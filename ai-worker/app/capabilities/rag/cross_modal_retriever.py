"""Cross-modal retriever — text + image RRF fusion.

Queries the text FAISS index (existing RAG) and the CLIP image index in
parallel (logically — both are synchronous in phase 2), then merges the
two ranked lists with Reciprocal Rank Fusion (RRF).

Image hits are mapped to :class:`RetrievedChunk` with ``section="image"``
and the chunk text set to the ingest-time caption (if available) or a
minimal metadata placeholder.  CLIP is NOT a caption generator — we only
surface metadata that was provided at index build time.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.image_embeddings import ImageEmbedder
from app.capabilities.rag.image_index import ImageFaissIndex
from app.capabilities.rag.image_metadata_store import ImageMetadataStore
from app.capabilities.rag.retriever import RetrievalReport, Retriever

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RankedItem:
    """Internal: an item participating in RRF."""

    chunk_id: str
    doc_id: str
    section: str
    text: str


class CrossModalRetriever:
    """Merges text-RAG and CLIP-image retrieval via RRF."""

    def __init__(
        self,
        *,
        text_retriever: Retriever,
        image_embedder: ImageEmbedder,
        image_index: ImageFaissIndex,
        image_metadata: ImageMetadataStore,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._text = text_retriever
        self._img_embedder = image_embedder
        self._img_index = image_index
        self._img_meta = image_metadata
        self._top_k = int(top_k)
        self._rrf_k = int(rrf_k)

    def retrieve_multimodal(self, query_text: str) -> RetrievalReport:
        """Retrieve from both text and image indexes, merge via RRF."""

        # 1. Text retrieval (existing path)
        text_report = self._text.retrieve(query_text)
        text_hits: List[RetrievedChunk] = text_report.results

        # 2. Image retrieval via CLIP text encoder
        image_hits = self._retrieve_images(query_text)

        # 3. RRF fusion
        merged = self._rrf_fuse(text_hits, image_hits)

        return RetrievalReport(
            query=query_text,
            top_k=self._top_k,
            index_version=text_report.index_version,
            embedding_model=text_report.embedding_model,
            results=merged[:self._top_k],
        )

    def _retrieve_images(self, query_text: str) -> List[RetrievedChunk]:
        """Encode query via CLIP text encoder and search image index."""
        vectors = self._img_embedder.encode_texts([query_text])
        hits = self._img_index.search(vectors, top_k=self._top_k)
        if not hits or not hits[0]:
            return []

        row_ids = [row_id for row_id, _score in hits[0]]
        looked_up = self._img_meta.lookup_by_faiss_rows(
            self._img_index.info.index_version, row_ids
        )
        if len(looked_up) < len(row_ids):
            log.warning(
                "Image metadata mismatch: FAISS returned %d hits but only "
                "%d found in ragmeta.image_chunks (index_version=%s). "
                "Possible stale index or incomplete ingest.",
                len(row_ids), len(looked_up), self._img_index.info.index_version,
            )
        score_by_row = {row_id: score for row_id, score in hits[0]}

        results: List[RetrievedChunk] = []
        for hit in looked_up:
            # Use caption from metadata if available; otherwise minimal placeholder
            if hit.caption:
                text = hit.caption
            else:
                text = f"[image: {hit.image_id}]"
            results.append(RetrievedChunk(
                chunk_id=hit.image_id,
                doc_id=hit.doc_id,
                section="image",
                text=text,
                score=float(score_by_row.get(hit.faiss_row_id, 0.0)),
            ))
        return results

    def _rrf_fuse(
        self,
        text_hits: List[RetrievedChunk],
        image_hits: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Reciprocal Rank Fusion over two ranked lists.

        score(item) = sum( 1 / (rrf_k + rank_i) ) across lists where
        item appears. Items are keyed by (doc_id, chunk_id).
        """
        k = self._rrf_k
        rrf_scores: defaultdict[tuple[str, str], float] = defaultdict(float)
        item_map: dict[tuple[str, str], RetrievedChunk] = {}

        for rank, chunk in enumerate(text_hits, start=1):
            key = (chunk.doc_id, chunk.chunk_id)
            rrf_scores[key] += 1.0 / (k + rank)
            item_map[key] = chunk

        for rank, chunk in enumerate(image_hits, start=1):
            key = (chunk.doc_id, chunk.chunk_id)
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in item_map:
                item_map[key] = chunk

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=item_map[key].chunk_id,
                doc_id=item_map[key].doc_id,
                section=item_map[key].section,
                text=item_map[key].text,
                score=rrf_scores[key],
            )
            for key in sorted_keys
        ]
