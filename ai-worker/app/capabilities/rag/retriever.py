"""Retrieval service — query text → top-k chunks.

Holds four live objects:

  1. An embedding provider (to vectorize the query)
  2. A loaded FAISS index (to do nearest-neighbour search)
  3. A ragmeta store handle (to look up text + metadata for each hit)
  4. A reranker provider (cross-encoder by default; NoOp in CI/offline)

The retriever is built once at worker startup and used for every job.
The bi-encoder fetches ``candidate_k`` candidates from FAISS; the
reranker then re-scores those candidates with a cross-encoder and
returns the top ``top_k``. When the reranker is ``NoOpReranker``, the
bi-encoder's top-k is returned unchanged — bit-for-bit identical to the
pre-reranker Phase 0 baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.metadata_store import RagMetadataStore
from app.capabilities.rag.reranker import NoOpReranker, RerankerProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalReport:
    query: str
    top_k: int
    index_version: str
    embedding_model: str
    results: List[RetrievedChunk]
    reranker_name: str = "noop"
    candidate_k: int = 0
    topk_gap: Optional[float] = None
    topk_rel_gap: Optional[float] = None


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        index: FaissIndex,
        metadata: RagMetadataStore,
        top_k: int,
        reranker: Optional[RerankerProvider] = None,
        candidate_k: Optional[int] = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._metadata = metadata
        self._top_k = int(top_k)
        self._reranker: RerankerProvider = reranker or NoOpReranker()
        # ``candidate_k`` controls how many bi-encoder candidates flow
        # into the reranker. None / non-positive collapses to top_k so a
        # NoOpReranker path reproduces the exact Phase 0 behaviour.
        if candidate_k is None or int(candidate_k) <= 0:
            self._candidate_k = self._top_k
        else:
            self._candidate_k = max(self._top_k, int(candidate_k))
        self._info: IndexBuildInfo | None = None

    def ensure_ready(self) -> None:
        """Load the index and strictly verify runtime model == build model.

        Two checks, in order:

          1. Model name equality. Mixing two different embedding models
             over the same FAISS index silently corrupts retrieval even
             when dimensions happen to agree, so we fail hard on any
             mismatch — no warning fallback.
          2. Dimension equality. Redundant in practice when (1) passes,
             but kept as a belt-and-suspenders guard against tampered
             build.json files or misbehaving embedder implementations.

        Both failures are raised as RuntimeError so the registry wraps
        them into a clean "RAG capability NOT registered" startup log
        without taking down the MOCK capability.
        """
        self._info = self._index.load()
        log.info(
            "Retriever readiness check: configured_model=%r index_model=%r "
            "configured_dim=%d index_dim=%d index_version=%s chunk_count=%d "
            "reranker=%s candidate_k=%d top_k=%d",
            self._embedder.model_name,
            self._info.embedding_model,
            self._embedder.dimension,
            self._info.dimension,
            self._info.index_version,
            self._info.chunk_count,
            self._reranker.name,
            self._candidate_k,
            self._top_k,
        )
        if self._embedder.model_name != self._info.embedding_model:
            raise RuntimeError(
                "Embedding MODEL mismatch: runtime embedder="
                f"{self._embedder.model_name!r} vs index build="
                f"{self._info.embedding_model!r}. "
                "When the embedding model changes, the FAISS index must be "
                "rebuilt AND the worker restarted — mixing models over the "
                "same index silently corrupts retrieval quality. "
                "Fix: either rebuild the index with the runtime model "
                "(`python -m scripts.build_rag_index --fixture`) or set "
                "AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL back to "
                f"{self._info.embedding_model!r}, then restart the worker."
            )
        if self._embedder.dimension != self._info.dimension:
            raise RuntimeError(
                "Embedding DIMENSION mismatch: runtime embedder dim="
                f"{self._embedder.dimension} vs index dim="
                f"{self._info.dimension}. Model names matched "
                f"({self._embedder.model_name!r}) so this almost always "
                "means build.json is stale or hand-edited. "
                "Rebuild the index (`python -m scripts.build_rag_index "
                "--fixture`) and restart the worker."
            )
        log.info(
            "Retriever ready: model=%s dim=%d index_version=%s reranker=%s",
            self._embedder.model_name,
            self._embedder.dimension,
            self._info.index_version,
            self._reranker.name,
        )

    def retrieve(self, query: str) -> RetrievalReport:
        if self._info is None:
            raise RuntimeError("Retriever is not ready — call ensure_ready() first")
        vectors = self._embedder.embed_queries([query])
        hits = self._index.search(vectors, top_k=self._candidate_k)
        if not hits or not hits[0]:
            return RetrievalReport(
                query=query,
                top_k=self._top_k,
                index_version=self._info.index_version,
                embedding_model=self._info.embedding_model,
                results=[],
                reranker_name=self._reranker.name,
                candidate_k=self._candidate_k,
                topk_gap=None,
                topk_rel_gap=None,
            )
        row_ids = [row_id for row_id, _score in hits[0]]
        looked_up = self._metadata.lookup_chunks_by_faiss_rows(
            self._info.index_version, row_ids
        )
        score_by_row = {row_id: score for row_id, score in hits[0]}

        candidates: List[RetrievedChunk] = []
        for hit in looked_up:
            candidates.append(RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                section=hit.section or "",
                text=hit.text,
                score=float(score_by_row.get(hit.faiss_row_id, 0.0)),
            ))

        results = self._reranker.rerank(query, candidates, k=self._top_k)

        topk_gap, topk_rel_gap = _compute_topk_gap(results)

        return RetrievalReport(
            query=query,
            top_k=self._top_k,
            index_version=self._info.index_version,
            embedding_model=self._info.embedding_model,
            results=results,
            reranker_name=self._reranker.name,
            candidate_k=self._candidate_k,
            topk_gap=topk_gap,
            topk_rel_gap=topk_rel_gap,
        )


def _compute_topk_gap(
    results: List[RetrievedChunk],
) -> tuple[Optional[float], Optional[float]]:
    """Compute absolute and relative gap between rank-1 and rank-k scores.

    Prefers ``rerank_score`` when present (the reranker is the live signal
    driving ordering); falls back to the bi-encoder ``score``. Returns
    ``(None, None)`` when the list has fewer than two items or both
    scores are missing — the caller surfaces None so ops can distinguish
    "no gap data" from "gap is 0".
    """
    if len(results) < 2:
        return None, None

    def _pick(c: RetrievedChunk) -> Optional[float]:
        return c.rerank_score if c.rerank_score is not None else c.score

    s1 = _pick(results[0])
    s2 = _pick(results[-1])
    if s1 is None or s2 is None:
        return None, None
    abs_gap = round(float(s1) - float(s2), 4)
    rel_gap = round(abs_gap / float(s1), 4) if abs(s1) > 1e-9 else None
    return abs_gap, rel_gap
