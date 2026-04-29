"""Phase 2 — RRF fusion of dense + sparse retrievers for eval-side A/B.

``RRFHybridEvalRetriever`` calls a dense retriever and a sparse
retriever (typically ``BM25EvalRetriever``) on the same query and fuses
the two ranked lists via Reciprocal Rank Fusion. The fused list is
returned as a duck-typed ``RetrievalReport`` so the existing
``run_retrieval_eval`` harness consumes it without changes.

Design notes:

  - **Eval-only**. This wrapper lives under ``eval/harness/`` and must
    not be imported from production code. The production ``Retriever``
    already has its own ``_rrf_merge`` for *multi-query* fusion within
    a single dense backend; that path is unrelated to this hybrid.
  - **Per-stage latency** is preserved. Dense wall-clock surfaces as
    ``dense_retrieval_ms``, sparse wall-clock as ``rerank_ms`` (the
    closest existing slot — BM25 is the "second-stage" reranker
    analogue for this hybrid). Total wall-clock includes the fusion.
  - **Candidate pool** surfacing: the report's ``candidate_doc_ids``
    is the deduplicated union of the dense + sparse candidate lists,
    in fused-rank order, so Phase 1 candidate hit@K metrics fire on
    the real combined pool.
  - **k_rrf** defaults to 60 (the value the production
    ``_rrf_merge`` and most Pyserini examples use). Lower k_rrf
    sharpens the contribution of high-rank items; higher k_rrf
    flattens it.
  - **Tie behaviour**: the same chunk_id appearing in both ranked
    lists has its fusion weights summed; ties on fused score keep
    the dense list's order (dense rank wins) so the fused output is
    deterministic and replay-safe.

Public surface:

  - ``RRFHybridEvalRetriever`` — wraps two retrievers, exposes
    ``retrieve(query) -> HybridReport``.
  - ``HybridReport`` — duck-typed report.
  - ``rrf_fuse_ranked_lists(*lists, k_rrf=60)`` — pure helper that
    can be reused by callers wanting to fuse already-retrieved lists
    without going through the wrapper.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from app.capabilities.rag.generation import RetrievedChunk

log = logging.getLogger(__name__)


DEFAULT_K_RRF = 60
DEFAULT_FINAL_TOP_K = 10
DEFAULT_PER_BACKEND_TOP_K = 100


class _RetrieverLike(Protocol):
    def retrieve(self, query: str) -> Any: ...


@dataclass
class HybridReport:
    """RetrievalReport-shaped output for the hybrid retriever.

    Mirrors the optional fields the eval harness reads off any report
    so a hybrid run produces the same Phase 1 metrics a dense run does.
    """

    results: List[RetrievedChunk]
    candidate_doc_ids: List[str] = field(default_factory=list)
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    rerank_ms: Optional[float] = None
    dense_retrieval_ms: Optional[float] = None
    rerank_breakdown_ms: Optional[Dict[str, float]] = None
    # Hybrid-specific provenance: how many chunks each backend
    # contributed before fusion. Surfaced on the report so an operator
    # can debug "why does the fused list look like the BM25 list" by
    # checking whether the dense backend returned anything at all.
    dense_candidate_count: int = 0
    sparse_candidate_count: int = 0
    fused_candidate_count: int = 0


def rrf_fuse_ranked_lists(
    *ranked_lists: Sequence[Tuple[str, Any]],
    k_rrf: int = DEFAULT_K_RRF,
) -> List[Tuple[str, float]]:
    """Fuse N ranked lists of ``(key, payload)`` pairs via RRF.

    Returns ``[(key, fused_score), ...]`` sorted by fused_score
    descending. Ties keep the order of the first list the key appeared
    in — important for replay determinism.

    The contract uses ``(key, payload)`` so callers can hand the
    payload (e.g. the ``RetrievedChunk``) along with the rank key
    (e.g. ``chunk_id``). Only the key is used for fusion; the payload
    rides along for downstream materialisation.

    ``k_rrf`` is the same constant as the production helper; values
    smaller than 1 are clamped to 1 (RRF is undefined at 0).
    """
    k = max(1, int(k_rrf))
    fused_scores: Dict[str, float] = {}
    first_list_seen: Dict[str, int] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        for rank, (key, _payload) in enumerate(ranked, start=1):
            if not key:
                continue
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            first_list_seen.setdefault(key, list_idx)
    return sorted(
        fused_scores.items(),
        key=lambda kv: (-kv[1], first_list_seen.get(kv[0], 0)),
    )


class RRFHybridEvalRetriever:
    """Wrap a dense + sparse retriever and fuse their results via RRF.

    Construction:

        RRFHybridEvalRetriever(
            dense=dense_retriever,
            sparse=sparse_retriever,
            k_rrf=60,
            final_top_k=10,
            per_backend_top_k=100,
            name="hybrid:dense+bm25",
        )

    ``retrieve(query)`` returns a ``HybridReport``. ``per_backend_top_k``
    caps how many results each backend contributes to the fusion; the
    fused list is then sliced to ``final_top_k``. The ``candidate_doc_ids``
    field on the report is the deduplicated union before slicing, in
    fused-rank order, so candidate@50 / @100 metrics fire on the right
    pool.
    """

    def __init__(
        self,
        *,
        dense: _RetrieverLike,
        sparse: _RetrieverLike,
        k_rrf: int = DEFAULT_K_RRF,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        per_backend_top_k: int = DEFAULT_PER_BACKEND_TOP_K,
        name: str = "hybrid:dense+bm25",
        index_version: Optional[str] = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._k_rrf = max(1, int(k_rrf))
        self._final_top_k = max(1, int(final_top_k))
        self._per_backend_top_k = max(1, int(per_backend_top_k))
        self._name = str(name)
        self._index_version = index_version

    @property
    def name(self) -> str:
        return self._name

    @property
    def k_rrf(self) -> int:
        return self._k_rrf

    def retrieve(self, query: str) -> HybridReport:
        """Run both backends, fuse, and emit a hybrid report."""
        # Run dense first so its order wins on fused-score ties.
        dense_started = time.perf_counter()
        dense_report = self._dense.retrieve(query)
        dense_elapsed = round(
            (time.perf_counter() - dense_started) * 1000.0, 3
        )
        sparse_started = time.perf_counter()
        sparse_report = self._sparse.retrieve(query)
        sparse_elapsed = round(
            (time.perf_counter() - sparse_started) * 1000.0, 3
        )

        dense_results = list(getattr(dense_report, "results", []) or [])
        sparse_results = list(getattr(sparse_report, "results", []) or [])
        dense_results = dense_results[: self._per_backend_top_k]
        sparse_results = sparse_results[: self._per_backend_top_k]

        # Build by-key payload tables so we can recover the chunk
        # object after fusion. Dense wins on chunk_id collision.
        chunk_by_id: Dict[str, RetrievedChunk] = {}
        for chunk in sparse_results:
            cid = getattr(chunk, "chunk_id", None) or ""
            if cid:
                chunk_by_id[cid] = chunk
        for chunk in dense_results:
            cid = getattr(chunk, "chunk_id", None) or ""
            if cid:
                chunk_by_id[cid] = chunk

        dense_keys = [
            (getattr(c, "chunk_id", None) or "", c) for c in dense_results
        ]
        sparse_keys = [
            (getattr(c, "chunk_id", None) or "", c) for c in sparse_results
        ]
        fused = rrf_fuse_ranked_lists(
            dense_keys, sparse_keys, k_rrf=self._k_rrf,
        )

        materialised: List[RetrievedChunk] = []
        candidate_doc_ids: List[str] = []
        for key, fused_score in fused:
            chunk = chunk_by_id.get(key)
            if chunk is None:
                continue
            # Replace ``score`` with the fused RRF score. The original
            # dense score is no longer meaningful as a ranking key for
            # the fused list; the fused score is what reflects the
            # final ordering.
            materialised.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section=chunk.section,
                text=chunk.text,
                score=float(fused_score),
                rerank_score=getattr(chunk, "rerank_score", None),
            ))
            doc_id = getattr(chunk, "doc_id", None) or ""
            if doc_id and doc_id not in candidate_doc_ids:
                candidate_doc_ids.append(doc_id)

        sliced = materialised[: self._final_top_k]
        return HybridReport(
            results=sliced,
            candidate_doc_ids=candidate_doc_ids,
            index_version=(
                self._index_version
                or getattr(dense_report, "index_version", None)
                or getattr(sparse_report, "index_version", None)
            ),
            embedding_model=getattr(dense_report, "embedding_model", None),
            reranker_name=self._name,
            # Surface dense wall-clock under dense_retrieval_ms so the
            # Phase 1 latency tables map cleanly. BM25 wall-clock goes
            # under rerank_ms — it's the second stage of the hybrid
            # pipeline and the harness's existing rerank latency
            # aggregator picks it up there.
            dense_retrieval_ms=dense_elapsed,
            rerank_ms=sparse_elapsed,
            rerank_breakdown_ms=None,
            dense_candidate_count=len(dense_results),
            sparse_candidate_count=len(sparse_results),
            fused_candidate_count=len(materialised),
        )
