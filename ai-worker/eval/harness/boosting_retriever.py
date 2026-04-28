"""Phase 2B boosting retriever wrapper + report.

Composes a dense-only base ``Retriever`` (NoOp reranker, top_k =
candidate-pool size) with a ``MetadataBoostReranker`` and an optional
post-boost cross-encoder reranker into a single retriever-like object
that the eval harness can drive query-by-query.

The wrapper records full per-call provenance — dense candidates,
boost breakdown, post-rerank chunks — so the eval harness can later
attribute every result to a specific stage. The recorded calls are
kept in insertion order and indexed by both query string and call
order so the boost summary can be re-built without re-running
retrieval.

Contract guarantees:

  - The ``BoostingRetrievalReport`` returned by ``retrieve`` has every
    field ``run_retrieval_eval`` reads off ``RetrievalReport`` (the
    base retriever's own report shape), with the ``rerank_ms`` /
    ``rerank_breakdown_ms`` slots populated by the post-boost
    reranker when present.
  - When the boost config is ``BoostConfig.disabled()`` AND no
    post-boost reranker is attached, the wrapper's results are byte-
    identical to the base retriever's dense-only output.
  - The base retriever's ``top_k`` MUST be ≥ the boost ``rerank_top_k``;
    we cannot boost a candidate the dense stage didn't surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import RerankerProvider
from eval.harness.boost_scorer import BoostScore, MetadataBoostReranker
from eval.harness.query_normalizer import NormalizedQuery

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateRecord:
    """Pre-boost dense candidate snapshot."""

    rank: int
    chunk_id: str
    doc_id: str
    section: str
    dense_score: float


@dataclass(frozen=True)
class BoostedCandidateRecord:
    """Post-boost candidate snapshot with full score breakdown."""

    rank: int
    chunk_id: str
    doc_id: str
    section: str
    dense_score: float
    boost_total: float
    final_score: float
    boost_breakdown: BoostScore


@dataclass(frozen=True)
class FinalResultRecord:
    """Post-rerank (or post-boost when no reranker) final result."""

    rank: int
    chunk_id: str
    doc_id: str
    section: str
    final_score: float
    rerank_score: Optional[float]


@dataclass
class BoostingRetrievalReport:
    """RetrievalReport-shaped payload + boost-specific extras.

    The first block of fields mirrors ``app.capabilities.rag.retriever.
    RetrievalReport`` so ``run_retrieval_eval`` can consume it without
    any awareness of the boost wrapper.
    """

    query: str
    top_k: int
    index_version: str
    embedding_model: str
    results: List[RetrievedChunk]
    reranker_name: str = "metadata_boost"
    candidate_k: int = 0
    topk_gap: Optional[float] = None
    topk_rel_gap: Optional[float] = None
    use_mmr: bool = False
    mmr_lambda: Optional[float] = None
    dup_rate: float = 0.0
    parsed_query: Optional[Any] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    filter_produced_no_docs: bool = False
    rerank_ms: Optional[float] = None
    dense_retrieval_ms: Optional[float] = None
    rerank_breakdown_ms: Optional[Dict[str, float]] = None

    # Boost-specific provenance.
    boost_ms: Optional[float] = None
    post_rerank_ms: Optional[float] = None
    post_rerank_breakdown_ms: Optional[Dict[str, float]] = None
    normalized_query: Optional[NormalizedQuery] = None
    dense_candidates: List[CandidateRecord] = field(default_factory=list)
    boosted_candidates: List[BoostedCandidateRecord] = field(default_factory=list)
    final_results: List[FinalResultRecord] = field(default_factory=list)


class BoostingEvalRetriever:
    """Composes dense + boost (+ optional rerank) for offline eval.

    Construction parameters
    -----------------------
    base_retriever
        A dense-only retriever (NoOp reranker, top_k ≥ ``boost_top_k``)
        whose ``retrieve(query)`` returns a ``RetrievalReport`` with
        ``.results``, ``.dense_retrieval_ms``, etc.
    boost_reranker
        ``MetadataBoostReranker`` instance. Pass a disabled config when
        you want a "boost-off" baseline that still goes through the
        wrapper (so the eval flow is identical between the two).
    post_reranker
        Optional second-stage reranker (typically the cross-encoder
        used in Phase 2A). Receives the boost-reordered candidates
        and produces the final top-K.
    boost_top_k
        How many candidates the boost stage emits. Defaults to the
        base retriever's top_k. Must be ≤ the base's top_k.
    final_top_k
        How many results the wrapper returns. Defaults to ``boost_top_k``.

    Recording
    ---------
    Every ``retrieve`` call appends a ``BoostingRetrievalReport`` to
    ``call_log``; ``last_call`` returns the most recent one. The eval
    runner reads back ``call_log`` after a sweep to compute boost-
    specific summary metrics (boost_applied_count, rescued / regressed,
    avg boost score, …).
    """

    def __init__(
        self,
        *,
        base_retriever: Any,
        boost_reranker: MetadataBoostReranker,
        post_reranker: Optional[RerankerProvider] = None,
        boost_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> None:
        if base_retriever is None:
            raise ValueError("base_retriever is required")
        if boost_reranker is None:
            raise ValueError("boost_reranker is required")
        self._base = base_retriever
        self._boost = boost_reranker
        self._post = post_reranker
        # Cache the base's effective output cap — Retriever exposes it
        # as ``_top_k``; we tolerate other retriever-likes via getattr.
        base_top_k = (
            getattr(base_retriever, "_top_k", None)
            or getattr(base_retriever, "top_k", None)
            or 10
        )
        self._base_top_k = int(base_top_k)
        self._boost_top_k = int(boost_top_k) if boost_top_k else self._base_top_k
        if self._boost_top_k > self._base_top_k:
            raise ValueError(
                f"boost_top_k ({self._boost_top_k}) cannot exceed the "
                f"base retriever's top_k ({self._base_top_k}); the boost "
                "stage cannot surface a candidate the dense stage didn't fetch."
            )
        self._final_top_k = (
            int(final_top_k) if final_top_k else self._boost_top_k
        )
        if self._final_top_k > self._boost_top_k:
            raise ValueError(
                f"final_top_k ({self._final_top_k}) cannot exceed "
                f"boost_top_k ({self._boost_top_k}); cannot truncate "
                "above the boost output."
            )
        self._call_log: List[BoostingRetrievalReport] = []

    @property
    def base_retriever(self) -> Any:
        return self._base

    @property
    def boost_reranker(self) -> MetadataBoostReranker:
        return self._boost

    @property
    def post_reranker(self) -> Optional[RerankerProvider]:
        return self._post

    @property
    def boost_top_k(self) -> int:
        return self._boost_top_k

    @property
    def final_top_k(self) -> int:
        return self._final_top_k

    @property
    def call_log(self) -> List[BoostingRetrievalReport]:
        """All recorded ``retrieve`` calls in insertion order."""
        return list(self._call_log)

    @property
    def last_call(self) -> Optional[BoostingRetrievalReport]:
        return self._call_log[-1] if self._call_log else None

    def reset_call_log(self) -> None:
        self._call_log = []

    def retrieve(self, query: str) -> BoostingRetrievalReport:
        base_report = self._base.retrieve(query)
        dense_chunks: List[RetrievedChunk] = list(
            getattr(base_report, "results", []) or []
        )

        dense_records = [
            CandidateRecord(
                rank=i + 1,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                section=c.section,
                dense_score=float(c.score),
            )
            for i, c in enumerate(dense_chunks)
        ]

        boost_t0 = time.perf_counter()
        boosted_chunks = self._boost.rerank(
            query, dense_chunks, k=self._boost_top_k
        )
        boost_ms = round((time.perf_counter() - boost_t0) * 1000.0, 3)

        boost_breakdown = self._boost.last_boost_breakdown
        normalized_query = self._boost.last_normalized_query

        boosted_records: List[BoostedCandidateRecord] = []
        for i, chunk in enumerate(boosted_chunks):
            score_brk = boost_breakdown.get(chunk.chunk_id, BoostScore.empty())
            dense_score = float(chunk.score) - float(score_brk.total)
            boosted_records.append(
                BoostedCandidateRecord(
                    rank=i + 1,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    dense_score=round(dense_score, 6),
                    boost_total=round(float(score_brk.total), 6),
                    final_score=round(float(chunk.score), 6),
                    boost_breakdown=score_brk,
                )
            )

        if self._post is not None:
            post_t0 = time.perf_counter()
            final_chunks = self._post.rerank(
                query, boosted_chunks, k=self._final_top_k
            )
            post_ms = round((time.perf_counter() - post_t0) * 1000.0, 3)
            post_brk_attr = getattr(self._post, "last_breakdown_ms", None)
            post_breakdown_ms: Optional[Dict[str, float]] = (
                dict(post_brk_attr)
                if isinstance(post_brk_attr, dict) and post_brk_attr
                else None
            )
            reported_rerank_ms = (
                None if self._post.name == "noop" else post_ms
            )
        else:
            final_chunks = boosted_chunks[: self._final_top_k]
            post_ms = None
            post_breakdown_ms = None
            reported_rerank_ms = None

        final_records = [
            FinalResultRecord(
                rank=i + 1,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                section=c.section,
                final_score=float(c.score),
                rerank_score=(
                    None if c.rerank_score is None else float(c.rerank_score)
                ),
            )
            for i, c in enumerate(final_chunks)
        ]

        # Compose the reranker_name string the eval row records. When the
        # boost is disabled and no post-reranker is attached, the contract
        # is byte-identical to the base retriever — so we surface its name.
        if self._boost.config.is_disabled() and self._post is None:
            reranker_name = getattr(base_report, "reranker_name", "noop")
        elif self._post is not None and not self._boost.config.is_disabled():
            reranker_name = f"{self._boost.name}+{self._post.name}"
        elif self._post is not None:
            reranker_name = self._post.name
        else:
            reranker_name = self._boost.name

        report = BoostingRetrievalReport(
            query=query,
            top_k=self._final_top_k,
            index_version=getattr(base_report, "index_version", ""),
            embedding_model=getattr(base_report, "embedding_model", ""),
            results=final_chunks,
            reranker_name=reranker_name,
            candidate_k=getattr(base_report, "candidate_k", 0),
            topk_gap=getattr(base_report, "topk_gap", None),
            topk_rel_gap=getattr(base_report, "topk_rel_gap", None),
            use_mmr=getattr(base_report, "use_mmr", False),
            mmr_lambda=getattr(base_report, "mmr_lambda", None),
            dup_rate=getattr(base_report, "dup_rate", 0.0),
            parsed_query=getattr(base_report, "parsed_query", None),
            filters=dict(getattr(base_report, "filters", {}) or {}),
            filter_produced_no_docs=getattr(
                base_report, "filter_produced_no_docs", False
            ),
            rerank_ms=reported_rerank_ms,
            dense_retrieval_ms=getattr(base_report, "dense_retrieval_ms", None),
            rerank_breakdown_ms=post_breakdown_ms,
            boost_ms=boost_ms,
            post_rerank_ms=post_ms,
            post_rerank_breakdown_ms=post_breakdown_ms,
            normalized_query=normalized_query,
            dense_candidates=dense_records,
            boosted_candidates=boosted_records,
            final_results=final_records,
        )
        self._call_log.append(report)
        return report
