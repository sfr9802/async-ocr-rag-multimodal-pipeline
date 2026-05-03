"""SearchUnit indexing worker loop.

This is deliberately separate from OCR callbacks and normal Redis job
execution. Spring remains the source of truth for claim state; this loop only
claims SearchUnits, indexes them into ragmeta/FAISS, and reports completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.capabilities.rag.search_unit_indexing import (
    SearchUnitIndexDocument,
    SearchUnitVectorIndexer,
    document_from_claim,
    is_indexable_claim,
)
from app.clients.core_api_client import CoreApiClient
from app.clients.schemas import (
    SearchUnitIndexClaimRequest,
    SearchUnitIndexEmbeddedRequest,
    SearchUnitIndexFailedRequest,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchUnitIndexingRunSummary:
    claimed: int = 0
    indexed: int = 0
    failed: int = 0
    stale: int = 0
    skipped_local: int = 0
    dry_run: bool = False


class SearchUnitIndexingWorker:
    def __init__(
        self,
        *,
        core_api: CoreApiClient,
        indexer: SearchUnitVectorIndexer,
        worker_id: str,
        batch_size: int,
        stale_after_seconds: Optional[int] = None,
    ) -> None:
        self._core_api = core_api
        self._indexer = indexer
        self._worker_id = worker_id
        self._batch_size = int(batch_size)
        self._stale_after_seconds = stale_after_seconds

    def run_once(self, *, dry_run: bool = False) -> SearchUnitIndexingRunSummary:
        if dry_run:
            log.info(
                "SearchUnit indexing dry-run: worker_id=%s batch_size=%d "
                "stale_after_seconds=%s; no claim or index mutation performed",
                self._worker_id,
                self._batch_size,
                self._stale_after_seconds,
            )
            return SearchUnitIndexingRunSummary(dry_run=True)

        claim = self._core_api.claim_search_unit_indexing(
            SearchUnitIndexClaimRequest(
                worker_id=self._worker_id,
                batch_size=self._batch_size,
                stale_after_seconds=self._stale_after_seconds,
            )
        )
        docs = [
            document_from_claim(unit.model_dump(by_alias=True))
            for unit in claim.units
        ]
        if not docs:
            log.info("SearchUnit indexing: no units claimed")
            return SearchUnitIndexingRunSummary()

        indexable: list[SearchUnitIndexDocument] = []
        skipped_local = 0
        failed = 0
        stale = 0
        for doc in docs:
            if is_indexable_claim(doc):
                indexable.append(doc)
                continue
            skipped_local += 1
            result = self._mark_failed(
                doc,
                "NOT_EMBEDDABLE: blank text, content hash missing, or metadata_json.indexable=false",
            )
            if result == "stale":
                stale += 1
            else:
                failed += 1

        if not indexable:
            return SearchUnitIndexingRunSummary(
                claimed=len(docs),
                failed=failed,
                stale=stale,
                skipped_local=skipped_local,
            )

        try:
            indexed = self._indexer.index_documents(indexable)
        except Exception as ex:
            detail = f"{type(ex).__name__}: {ex}"
            log.exception(
                "SearchUnit indexing batch failed claimed=%d detail=%s",
                len(indexable),
                detail,
            )
            for doc in indexable:
                result = self._mark_failed(doc, detail)
                if result == "stale":
                    stale += 1
                else:
                    failed += 1
            return SearchUnitIndexingRunSummary(
                claimed=len(docs),
                failed=failed,
                stale=stale,
                skipped_local=skipped_local,
            )

        applied = 0
        for item in indexed.indexed:
            response = self._core_api.mark_search_unit_embedded(
                item.search_unit_id,
                SearchUnitIndexEmbeddedRequest(
                    claim_token=item.claim_token,
                    content_sha256=item.content_sha256,
                    index_id=item.index_id,
                ),
            )
            if response.stale:
                stale += 1
                log.info(
                    "SearchUnit indexing completion stale searchUnitId=%s detail=%s",
                    item.search_unit_id,
                    response.detail,
                )
            elif response.applied:
                applied += 1
            else:
                failed += 1
                log.warning(
                    "SearchUnit indexing completion not applied searchUnitId=%s detail=%s",
                    item.search_unit_id,
                    response.detail,
                )

        return SearchUnitIndexingRunSummary(
            claimed=len(docs),
            indexed=applied,
            failed=failed,
            stale=stale,
            skipped_local=skipped_local,
        )

    def run_loop(
        self,
        *,
        interval_seconds: float,
        dry_run: bool = False,
    ) -> None:
        interval = max(0.1, float(interval_seconds))
        while True:
            summary = self.run_once(dry_run=dry_run)
            log.info("SearchUnit indexing loop summary: %s", summary)
            time.sleep(interval)

    def _mark_failed(self, doc: SearchUnitIndexDocument, detail: str) -> str:
        response = self._core_api.mark_search_unit_indexing_failed(
            doc.search_unit_id,
            SearchUnitIndexFailedRequest(
                claim_token=doc.claim_token,
                content_sha256=doc.content_sha256,
                detail=detail[:1900],
            ),
        )
        if response.stale:
            log.info(
                "SearchUnit indexing failure callback stale searchUnitId=%s detail=%s",
                doc.search_unit_id,
                response.detail,
            )
            return "stale"
        if not response.applied:
            log.warning(
                "SearchUnit indexing failure callback not applied searchUnitId=%s detail=%s",
                doc.search_unit_id,
                response.detail,
            )
        return "failed"
