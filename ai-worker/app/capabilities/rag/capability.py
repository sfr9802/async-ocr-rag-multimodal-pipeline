"""RAG capability — the thing the worker's task runner calls.

Given a job's INPUT_TEXT artifact, this capability:

  1. Builds the retrieval query from the input bytes
  2. Asks the Retriever for top-k chunks
  3. Hands the chunks to the GenerationProvider for a grounded answer
  4. Emits two output artifacts:
       - RETRIEVAL_RESULT  : JSON payload of the retrieval report
                             (which chunks were retrieved, scores, the
                              index version and embedding model used)
       - FINAL_RESPONSE    : the markdown answer produced by the
                             generation provider
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.generation import GenerationProvider, RetrievedChunk
from app.capabilities.rag.retrieval_contract import retrieval_result_row
from app.capabilities.rag.retriever import RetrievalReport, Retriever

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagCapabilityConfig:
    top_k: int
    excerpt_chars: int = 400


class RagCapability(Capability):
    name = "RAG"

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: GenerationProvider,
        config: RagCapabilityConfig,
        audit_store: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._config = config
        self._audit_store = audit_store

    # ------------------------------------------------------------------

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        query = self._extract_query(input)
        if not query:
            raise CapabilityError(
                "EMPTY_QUERY",
                "RAG job has no INPUT_TEXT content to run retrieval on.",
            )
        log.info("RAG retrieval start jobId=%s query=%r", input.job_id, query[:120])

        report = self._retriever.retrieve(query)
        log.info(
            "RAG retrieval done jobId=%s hits=%d index_version=%s",
            input.job_id, len(report.results), report.index_version,
        )
        self._record_audit(input, report)

        answer_md = self._generator.generate(query, report.results)

        retrieval_artifact = CapabilityOutputArtifact(
            type="RETRIEVAL_RESULT",
            filename="retrieval.json",
            content_type="application/json",
            content=self._retrieval_payload(report).encode("utf-8"),
        )
        final_artifact = CapabilityOutputArtifact(
            type="FINAL_RESPONSE",
            filename="answer.md",
            content_type="text/markdown; charset=utf-8",
            content=answer_md.encode("utf-8"),
        )
        return CapabilityOutput(outputs=[retrieval_artifact, final_artifact])

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query(input: CapabilityInput) -> str:
        for artifact in input.inputs:
            if artifact.type == "INPUT_TEXT":
                try:
                    return artifact.content.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
        # Last resort: any input artifact, decoded best-effort.
        for artifact in input.inputs:
            try:
                text = artifact.content.decode("utf-8").strip()
                if text:
                    return text
            except UnicodeDecodeError:
                continue
        return ""

    @staticmethod
    def _retrieval_payload(report: RetrievalReport) -> str:
        body = {
            "query": report.query,
            "topK": report.top_k,
            "indexVersion": report.index_version,
            "embeddingModel": report.embedding_model,
            "hitCount": len(report.results),
            "results": [
                retrieval_result_row(i + 1, r)
                for i, r in enumerate(report.results)
            ],
        }
        if report.parsed_query is not None:
            body["parsedQuery"] = report.parsed_query.to_dict()
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _record_audit(self, input: CapabilityInput, report: RetrievalReport) -> None:
        if self._audit_store is None:
            return
        try:
            self._audit_store.record_retrieval(
                report,
                request_id=input.job_id,
                user_id=None,
            )
        except Exception as ex:
            log.warning(
                "RAG retrieval audit failed jobId=%s error=%s",
                input.job_id,
                ex,
                exc_info=True,
            )


def _make_retrieved(
    *, chunk_id: str, doc_id: str, section: str, text: str, score: float
) -> RetrievedChunk:
    # Helper for tests so they don't have to import from generation.py.
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=doc_id, section=section, text=text, score=score
    )
