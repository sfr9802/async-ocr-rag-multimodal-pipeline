"""Phase 7.1 — paired A/B between dense baseline and reranker over the
Phase 7.0 ``retrieval_title_section`` index.

The Phase 7.0 verdict promoted ``retrieval_title_section`` to the default
candidate variant. This module reuses the same FAISS index (no rebuild)
and asks: does layering a cross-encoder on top of those dense candidates
*still* improve hit@k / MRR / nDCG, or does the +22pt dense lift already
saturate the candidate ordering?

The harness contract:

  - **baseline**: dense top ``final_k`` straight from the retriever.
    The retriever is configured with ``candidate_k = final_k`` and a
    ``NoOpReranker``, so this is bit-for-bit Phase 7.0's candidate side.
  - **candidate**: dense top ``candidate_k`` (default 40, configurable to
    20 / 80) handed to the cross-encoder, which returns the top
    ``final_k``. Two scoring modes:

    1. ``reranker_only`` — the reranker's score is the sole ordering
       signal. ``rerank_score`` populated on each chunk; ``score`` is
       preserved for diagnostics.
    2. ``weighted_dense_rerank`` — the chunk's final score is
       ``dense_weight * z(dense) + rerank_weight * z(rerank)`` where
       ``z(.)`` is the per-query min-max normalised score across the
       candidate pool. Defaults: ``dense_weight=0.3``,
       ``rerank_weight=0.7`` (rerank-leaning, the regime cross-encoders
       earn their place in).

  - All metrics — hit@1 / hit@3 / hit@10 / MRR@10 / nDCG@10 — are computed
    against the same silver query set Phase 7.0 used (200 queries
    stratified across main_work / subpage_generic / subpage_named).
    Phase 7.0's classifier is reused: a per-query status of one of
    {improved, regressed, both_hit, both_missed} based on best-rank
    movement under the baseline → candidate transition.

  - Latency: each rerank-side call is wall-clocked end-to-end; the
    ``LatencySummary`` carries p50/p90/p99 / mean / max in ms over the
    full query set.

  - **Diagnostic columns asked for by the spec**:
      - ``gold_in_input`` — was the expected doc_id present in the
        candidate pool the reranker received? (Distinguishes a rerank
        failure from a recall ceiling.)
      - ``rank_before_rerank`` — gold's 1-indexed dense rank inside
        the candidate pool, or -1.
      - ``rank_after_rerank`` — gold's 1-indexed rank in the post-rerank
        ``final_k`` window, or -1.
      - ``regression_severity`` — when ``status == "regressed"``,
        baseline_rank - candidate_rank (positive = larger drop).

Production code is NOT touched. The retriever / reranker contracts in
``app/capabilities/rag/`` are reused as-is; this module composes them
into a paired A/B and writes the artefact bundle Phase 7.1 asks for.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import RerankerProvider
from app.capabilities.rag.retriever import RetrievalReport, Retriever

from eval.harness.v4_ab_eval import (
    PerQueryMetrics,
    QueryRecord,
    _classify,
    _compute_dup_rate,
    _generic_collision_count,
    _ndcg_at_10,
    _per_query_metrics,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCORE_MODE_RERANKER_ONLY = "reranker_only"
SCORE_MODE_WEIGHTED = "weighted_dense_rerank"
SCORE_MODES: Tuple[str, ...] = (SCORE_MODE_RERANKER_ONLY, SCORE_MODE_WEIGHTED)

_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
_DEFAULT_FINAL_K = 10
_DEFAULT_CANDIDATE_K = 40
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_DENSE_WEIGHT = 0.3
_DEFAULT_RERANK_WEIGHT = 0.7
_RANK_MISS = -1


@dataclass(frozen=True)
class RerankerAbConfig:
    """Knobs the harness needs to reproduce a run.

    The dataclass is the source of truth — the CLI maps argparse args
    onto this and the JSON summary serialises it back. A reviewer can
    tell from one block exactly which ``score_mode`` / ``candidate_k``
    / weights produced a given report.
    """

    candidate_k: int = _DEFAULT_CANDIDATE_K
    final_k: int = _DEFAULT_FINAL_K
    reranker_batch_size: int = _DEFAULT_BATCH_SIZE
    score_mode: str = SCORE_MODE_RERANKER_ONLY
    dense_weight: float = _DEFAULT_DENSE_WEIGHT
    rerank_weight: float = _DEFAULT_RERANK_WEIGHT

    def validate(self) -> "RerankerAbConfig":
        if self.candidate_k <= 0:
            raise ValueError(
                f"candidate_k must be positive, got {self.candidate_k}."
            )
        if self.final_k <= 0:
            raise ValueError(
                f"final_k must be positive, got {self.final_k}."
            )
        if self.candidate_k < self.final_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be >= "
                f"final_k ({self.final_k}); reranker cannot expand the pool."
            )
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score_mode={self.score_mode!r}; "
                f"expected one of {SCORE_MODES}."
            )
        if self.score_mode == SCORE_MODE_WEIGHTED:
            if self.dense_weight < 0.0 or self.rerank_weight < 0.0:
                raise ValueError(
                    "dense_weight and rerank_weight must be non-negative."
                )
            if (self.dense_weight + self.rerank_weight) <= 0.0:
                raise ValueError(
                    "dense_weight + rerank_weight must be > 0 for weighted mode."
                )
        if self.reranker_batch_size <= 0:
            raise ValueError(
                f"reranker_batch_size must be positive, "
                f"got {self.reranker_batch_size}."
            )
        return self


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------


def _minmax(values: Sequence[float]) -> List[float]:
    """Min-max normalise to [0, 1]; degenerate ranges collapse to 0.5.

    A constant array would otherwise produce a divide-by-zero. Returning
    0.5 keeps the blend stable — every chunk contributes the same dense
    component, so the rerank component (which usually has spread) does
    the actual ordering work.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return [0.5] * len(values)
    return [(float(v) - lo) / span for v in values]


def blend_scores(
    chunks: Sequence[RetrievedChunk],
    *,
    dense_weight: float,
    rerank_weight: float,
) -> List[Tuple[RetrievedChunk, float]]:
    """Compute weighted score per chunk.

    Both score columns are min-max normalised inside the candidate pool
    so the weights map predictably regardless of the raw dense / rerank
    score scales (bge-m3 IP scores hover near 0.5–0.9; bge-reranker-v2-m3
    sigmoid scores span -10..+10 pre-sigmoid). Without normalisation, a
    50/50 weight would in practice be a 99/1 weight — we make the weights
    mean what they say.
    """
    if not chunks:
        return []
    dense = [c.score if c.score is not None else 0.0 for c in chunks]
    rerank = [
        c.rerank_score if c.rerank_score is not None else 0.0 for c in chunks
    ]
    dense_n = _minmax(dense)
    rerank_n = _minmax(rerank)
    total = float(dense_weight) + float(rerank_weight)
    if total <= 0.0:
        # Already validated upstream, but degrade gracefully so a
        # downstream caller can't silently divide by zero.
        return [(c, 0.0) for c in chunks]
    out: List[Tuple[RetrievedChunk, float]] = []
    for c, d, r in zip(chunks, dense_n, rerank_n):
        blended = (float(dense_weight) * d + float(rerank_weight) * r) / total
        out.append((c, blended))
    return out


# ---------------------------------------------------------------------------
# Reranker run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RerankCandidateOutcome:
    """One query's reranker side: ordered top-k + diagnostic flags."""

    results: List[RetrievedChunk]
    candidate_pool: List[RetrievedChunk]
    rerank_latency_ms: float
    rank_before_rerank: int
    rank_after_rerank: int
    gold_in_input: bool
    gold_was_demoted: bool


def _best_rank(
    results: Sequence[RetrievedChunk], expected: Sequence[str],
) -> int:
    if not expected or not results:
        return _RANK_MISS
    expected_set = set(expected)
    for i, c in enumerate(results, start=1):
        if c.doc_id in expected_set:
            return i
    return _RANK_MISS


def run_reranker_candidate(
    query: str,
    *,
    candidate_pool: Sequence[RetrievedChunk],
    reranker: RerankerProvider,
    config: RerankerAbConfig,
    expected_doc_ids: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> _RerankCandidateOutcome:
    """Apply the reranker to a candidate pool and order by ``score_mode``.

    The candidate pool MUST already be the dense top-``candidate_k`` for
    the current query — this function never re-runs FAISS. ``score_mode``
    decides whether the resulting list is ordered by ``rerank_score``
    alone or by the weighted blend.

    The returned outcome carries every diagnostic Phase 7.1's regression
    analysis asks for:

      - ``rank_before_rerank``: gold's 1-indexed position in the dense
        pool (-1 = recall miss / pool ceiling).
      - ``rank_after_rerank``: gold's 1-indexed position in the final-k
        result, -1 if dropped or absent.
      - ``gold_in_input``: ``True`` iff gold was anywhere in the dense
        pool. Distinguishes a rerank failure from a recall miss.
      - ``gold_was_demoted``: dense had gold in the top-final_k window
        but the rerank ordering pushed it past final_k. Strictly subset
        of regressions caused by the reranker, not by candidate-set
        composition.
    """
    config.validate()
    pool = list(candidate_pool)
    rank_before = _best_rank(pool, expected_doc_ids)
    gold_in_input = rank_before > 0

    t0 = clock()
    if not pool:
        rerank_ms = round((clock() - t0) * 1000.0, 3)
        return _RerankCandidateOutcome(
            results=[],
            candidate_pool=[],
            rerank_latency_ms=rerank_ms,
            rank_before_rerank=_RANK_MISS,
            rank_after_rerank=_RANK_MISS,
            gold_in_input=False,
            gold_was_demoted=False,
        )

    # Ask the reranker for the FULL pool so we always have rerank_score
    # on every candidate; the post-rank trim picks the final_k slot.
    # When score_mode is reranker_only this is identical to asking for
    # the top-final_k directly, but having scores on every candidate
    # lets the weighted blend mode reason about the whole pool.
    rerank_k = len(pool)
    reranked = reranker.rerank(query, pool, k=rerank_k)
    rerank_ms = round((clock() - t0) * 1000.0, 3)

    if config.score_mode == SCORE_MODE_RERANKER_ONLY:
        ordered = list(reranked)
    elif config.score_mode == SCORE_MODE_WEIGHTED:
        scored = blend_scores(
            reranked,
            dense_weight=config.dense_weight,
            rerank_weight=config.rerank_weight,
        )
        scored.sort(key=lambda t: t[1], reverse=True)
        ordered = [c for c, _ in scored]
    else:  # pragma: no cover — validate() blocks this branch upstream.
        raise ValueError(f"unknown score_mode={config.score_mode!r}")

    final = ordered[: config.final_k]
    rank_after = _best_rank(final, expected_doc_ids)

    # ``gold_was_demoted`` flags only the rerank-attributable regression:
    # the dense pool had gold inside the final_k window, but the rerank
    # ordering knocked it out. Pool-ceiling misses (rank_before > final_k
    # but the reranker happens to push it further down) are NOT counted
    # as a demotion — that is a recall problem the reranker cannot fix.
    gold_was_demoted = (
        gold_in_input
        and rank_before <= config.final_k
        and rank_after == _RANK_MISS
    )

    return _RerankCandidateOutcome(
        results=final,
        candidate_pool=pool,
        rerank_latency_ms=rerank_ms,
        rank_before_rerank=rank_before,
        rank_after_rerank=rank_after,
        gold_in_input=gold_in_input,
        gold_was_demoted=gold_was_demoted,
    )


# ---------------------------------------------------------------------------
# Latency summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencySummary:
    """p50 / p90 / p99 / mean / max in ms across one query set.

    Empty input collapses to all-zeros — the JSON writer treats this as
    "harness ran but reranker was never invoked", which is the only way
    you'd get an empty list from ``run_reranker_ab`` with valid inputs
    (a no-rerank dry run for a smoke test, for example).
    """

    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


def summarize_latency(samples: Sequence[float]) -> LatencySummary:
    if not samples:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = sorted(float(s) for s in samples)
    n = len(arr)
    mean = sum(arr) / n
    return LatencySummary(
        count=n,
        mean_ms=round(mean, 3),
        p50_ms=round(_quantile(arr, 0.50), 3),
        p90_ms=round(_quantile(arr, 0.90), 3),
        p99_ms=round(_quantile(arr, 0.99), 3),
        max_ms=round(arr[-1], 3),
    )


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile.

    Same definition NumPy / SciPy / pandas use by default
    (``method="linear"``). Avoids importing numpy here so the harness
    has zero numerical-runtime cost on top of stdlib.
    """
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    # 0-indexed position with linear interpolation.
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


# ---------------------------------------------------------------------------
# Per-query record + serialisation
# ---------------------------------------------------------------------------


def _per_query_metrics_from_results(
    results: Sequence[RetrievedChunk],
    expected_doc_ids: Sequence[str],
) -> PerQueryMetrics:
    """Reuse Phase 7.0's per-query metric builder against an arbitrary list.

    Phase 7.0's ``_per_query_metrics`` takes a ``RetrievalReport`` —
    Phase 7.1 sometimes builds the result list directly from a reranker
    pass without producing a full report, so we wrap it in a small
    shim that exposes only the ``results`` attribute the metric path
    uses.
    """

    class _Shim:
        def __init__(self, rs: Sequence[RetrievedChunk]) -> None:
            self.results = list(rs)

    return _per_query_metrics(_Shim(results), expected_doc_ids)


def _serialise_outcome(
    outcome: _RerankCandidateOutcome,
    metrics: PerQueryMetrics,
) -> Dict[str, Any]:
    """Pack candidate-side metrics + diagnostics for the JSONL row."""
    return {
        "rank": metrics.rank,
        "hit_at": {str(k): metrics.hit_at[k] for k in _K_VALUES},
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
        "dup_rate": metrics.dup_rate,
        "same_title_collisions": metrics.same_title_collisions,
        "top_results": metrics.top_results,
        "rank_before_rerank": outcome.rank_before_rerank,
        "rank_after_rerank": outcome.rank_after_rerank,
        "gold_in_input": outcome.gold_in_input,
        "gold_was_demoted": outcome.gold_was_demoted,
        "rerank_latency_ms": outcome.rerank_latency_ms,
    }


def _serialise_baseline(
    metrics: PerQueryMetrics,
) -> Dict[str, Any]:
    """Baseline side: dense-only, no rerank diagnostics."""
    return {
        "rank": metrics.rank,
        "hit_at": {str(k): metrics.hit_at[k] for k in _K_VALUES},
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
        "dup_rate": metrics.dup_rate,
        "same_title_collisions": metrics.same_title_collisions,
        "top_results": metrics.top_results,
    }


def _candidate_pool_preview(
    pool: Sequence[RetrievedChunk], limit: int = 10,
) -> List[Dict[str, Any]]:
    """Compact view of the dense candidate pool for the regressed dump."""
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(pool[:limit], start=1):
        out.append({
            "rank": i,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "section": c.section,
            "score": float(c.score) if c.score is not None else None,
            "rerank_score": (
                float(c.rerank_score) if c.rerank_score is not None else None
            ),
        })
    return out


def _row_status_severity(
    baseline_metrics: PerQueryMetrics,
    candidate_metrics: PerQueryMetrics,
) -> Optional[int]:
    """Regression severity = baseline_rank - candidate_rank (positive = worse).

    None when the row is not a regression, or when either side is a
    miss (-1) and severity isn't well-defined as an integer drop.
    """
    if not (
        0 < baseline_metrics.rank <= _DEFAULT_FINAL_K
    ):
        return None
    cand_rank = candidate_metrics.rank
    if cand_rank == _RANK_MISS:
        # Dropped past final_k entirely — severity is the deepest possible.
        return _DEFAULT_FINAL_K + 1 - baseline_metrics.rank
    if cand_rank > baseline_metrics.rank:
        return cand_rank - baseline_metrics.rank
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_side(rows: Sequence[Mapping[str, Any]], side: str) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            **{f"hit_at_{k}": 0.0 for k in _K_VALUES},
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "dup_rate": 0.0,
            "same_title_collisions_avg": 0.0,
        }
    n = len(rows)
    payload = {
        "count": n,
        **{
            f"hit_at_{k}":
                sum(r[side]["hit_at"][str(k)] for r in rows) / n
            for k in _K_VALUES
        },
        "mrr_at_10": sum(r[side]["mrr_at_10"] for r in rows) / n,
        "ndcg_at_10": sum(r[side]["ndcg_at_10"] for r in rows) / n,
        "dup_rate": sum(r[side]["dup_rate"] for r in rows) / n,
        "same_title_collisions_avg":
            sum(r[side]["same_title_collisions"] for r in rows) / n,
    }
    return payload


def _aggregate_by_bucket(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(r.get("bucket") or "<unbucketed>", []).append(r)
    return {
        bucket: {
            "count": len(rs),
            "baseline": _aggregate_side(rs, "baseline"),
            "candidate": _aggregate_side(rs, "candidate"),
            "status_counts": _status_counts(rs),
        }
        for bucket, rs in sorted(by.items())
    }


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {
        "improved": 0, "regressed": 0, "both_hit": 0,
        "both_missed": 0, "unchanged": 0,
    }
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def _gold_input_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count how many rows had gold in the candidate pool / were demoted.

    Surfaced at aggregate + per-bucket level. Two counters:

      - ``gold_in_input``: rerank pool contained gold. Upper bound on
        what the reranker could ever have promoted.
      - ``gold_was_demoted``: gold was inside the *dense final_k* window
        but the reranker pushed it past final_k. Strictly a reranker
        defect, isolated from recall ceiling.
    """
    return {
        "gold_in_input": sum(
            1 for r in rows if r["candidate"].get("gold_in_input")
        ),
        "gold_was_demoted": sum(
            1 for r in rows if r["candidate"].get("gold_was_demoted")
        ),
    }


# ---------------------------------------------------------------------------
# Result container + main entry point
# ---------------------------------------------------------------------------


@dataclass
class RerankerAbResult:
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    latency_summary: Dict[str, Any] = field(default_factory=dict)


def run_reranker_ab(
    queries: Sequence[QueryRecord],
    *,
    baseline_retriever: Retriever,
    candidate_retriever: Retriever,
    reranker: RerankerProvider,
    config: RerankerAbConfig,
    baseline_label: str = "retrieval_title_section_dense",
    candidate_label: str = "retrieval_title_section_dense_plus_rerank",
    progress_log_every: int = 25,
    clock: Callable[[], float] = time.perf_counter,
) -> RerankerAbResult:
    """Drive the paired A/B between dense-only and dense+rerank.

    ``baseline_retriever`` MUST be configured with ``candidate_k =
    final_k`` and a NoOp reranker — i.e. it is the Phase 7.0 candidate
    side, used here as the dense-only baseline.

    ``candidate_retriever`` MUST be configured with ``candidate_k =
    config.candidate_k`` (typically 40) and a NoOp reranker; this
    function then applies ``reranker`` outside the retriever so we can
    record the dense pool order before rerank runs (Phase 7.1's
    ``rank_before_rerank`` diagnostic).

    Both retrievers must already be ``ensure_ready()``-ed and the
    silver query set must be the Phase 7.0 200-query set (or a strict
    subset / superset — the harness doesn't care about size, only
    that ``v4_meta.bucket`` is populated for the bucket aggregation).
    """
    config = config.validate()
    rows: List[Dict[str, Any]] = []
    rerank_latencies: List[float] = []

    for i, q in enumerate(queries, start=1):
        # Baseline: dense-only top-final_k.
        b_report = baseline_retriever.retrieve(q.query)
        baseline_metrics = _per_query_metrics_from_results(
            b_report.results, q.expected_doc_ids,
        )

        # Candidate: dense top-candidate_k, then rerank.
        c_report = candidate_retriever.retrieve(q.query)
        outcome = run_reranker_candidate(
            q.query,
            candidate_pool=c_report.results,
            reranker=reranker,
            config=config,
            expected_doc_ids=q.expected_doc_ids,
            clock=clock,
        )
        candidate_metrics = _per_query_metrics_from_results(
            outcome.results, q.expected_doc_ids,
        )

        status = _classify(baseline_metrics, candidate_metrics)
        severity = _row_status_severity(baseline_metrics, candidate_metrics)
        rerank_latencies.append(outcome.rerank_latency_ms)

        row = {
            "qid": q.qid,
            "query": q.query,
            "expected_doc_ids": list(q.expected_doc_ids),
            "answer_type": q.answer_type,
            "difficulty": q.difficulty,
            "bucket": q.bucket,
            "v4_meta": q.v4_meta,
            "status": status,
            "regression_severity": severity,
            "baseline": _serialise_baseline(baseline_metrics),
            "candidate": _serialise_outcome(outcome, candidate_metrics),
            "candidate_pool_preview": _candidate_pool_preview(
                outcome.candidate_pool,
            ),
        }
        rows.append(row)

        if progress_log_every and (i % progress_log_every == 0):
            log.info("phase7.1 ab progress: %d/%d", i, len(queries))

    aggregate = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "config": asdict(config),
        "n_queries": len(rows),
        "k_values": list(_K_VALUES),
        "baseline": _aggregate_side(rows, "baseline"),
        "candidate": _aggregate_side(rows, "candidate"),
        "status_counts": _status_counts(rows),
        "gold_input_counts": _gold_input_counts(rows),
        "by_bucket": {
            bucket: {
                **payload,
                "gold_input_counts": _gold_input_counts(
                    [r for r in rows if (r.get("bucket") or "<unbucketed>") == bucket]
                ),
            }
            for bucket, payload in _aggregate_by_bucket(rows).items()
        },
    }

    latency = summarize_latency(rerank_latencies)
    latency_dict = {
        "config": asdict(config),
        "n_queries": len(rerank_latencies),
        "mean_ms": latency.mean_ms,
        "p50_ms": latency.p50_ms,
        "p90_ms": latency.p90_ms,
        "p99_ms": latency.p99_ms,
        "max_ms": latency.max_ms,
    }
    return RerankerAbResult(
        per_query=rows,
        aggregate=aggregate,
        latency_summary=latency_dict,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_ab_outputs(
    result: RerankerAbResult,
    *,
    out_dir: Path,
    summary_json_name: str = "ab_summary.json",
    summary_md_name: str = "ab_summary.md",
    per_query_name: str = "per_query_comparison.jsonl",
    improved_name: str = "improved_queries.jsonl",
    regressed_name: str = "regressed_queries.jsonl",
    latency_name: str = "reranker_latency_summary.json",
) -> Dict[str, Path]:
    """Persist the artefact bundle Phase 7.1 asks for.

    The shape mirrors Phase 7.0's writer plus three additions:

      - per-query rows carry a ``candidate.rank_before_rerank`` /
        ``rank_after_rerank`` / ``gold_in_input`` / ``gold_was_demoted``
        block.
      - regressed rows include the dense candidate-pool preview so a
        reviewer can read the rerank failure without re-running anything.
      - ``reranker_latency_summary.json`` is written separately so
        latency-only diff tools can ingest it without parsing the full
        per_query bundle.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / summary_json_name
    summary_md = out_dir / summary_md_name
    per_query = out_dir / per_query_name
    improved = out_dir / improved_name
    regressed = out_dir / regressed_name
    latency = out_dir / latency_name

    summary_json.write_text(
        json.dumps(result.aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(
        render_summary_md(result.aggregate, result.latency_summary),
        encoding="utf-8",
    )
    with per_query.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with improved.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            if r["status"] == "improved":
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with regressed.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            if r["status"] == "regressed":
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    latency.write_text(
        json.dumps(result.latency_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "per_query": per_query,
        "improved": improved,
        "regressed": regressed,
        "latency": latency,
    }


def render_summary_md(
    agg: Mapping[str, Any], latency: Mapping[str, Any],
) -> str:
    """Render the summary markdown.

    Layout: headline metrics → status counts → gold-in-input counts →
    bucket breakdown → latency. Caveats live in the final report doc;
    keeping the auto-rendered summary terse means a reviewer can scan
    it in 30s without scrolling.
    """
    cfg = agg.get("config") or {}
    lines: List[str] = []
    lines.append(
        f"# Phase 7.1 reranker A/B — "
        f"{agg.get('baseline_label')} vs {agg.get('candidate_label')}"
    )
    lines.append("")
    lines.append(f"- n_queries: **{agg.get('n_queries', 0)}**")
    lines.append(f"- k_values: {agg.get('k_values', [])}")
    lines.append(
        f"- score_mode: **{cfg.get('score_mode')}** "
        f"(candidate_k={cfg.get('candidate_k')}, "
        f"final_k={cfg.get('final_k')}, "
        f"batch={cfg.get('reranker_batch_size')})"
    )
    if cfg.get("score_mode") == SCORE_MODE_WEIGHTED:
        lines.append(
            f"- weights: dense={cfg.get('dense_weight')}, "
            f"rerank={cfg.get('rerank_weight')}"
        )
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| metric | baseline | candidate | Δ (cand − base) |")
    lines.append("|---|---:|---:|---:|")
    base = agg.get("baseline") or {}
    cand = agg.get("candidate") or {}
    for k in agg.get("k_values", _K_VALUES):
        b = base.get(f"hit_at_{k}", 0.0)
        c = cand.get(f"hit_at_{k}", 0.0)
        lines.append(f"| hit@{k} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
    for key in (
        "mrr_at_10", "ndcg_at_10", "dup_rate", "same_title_collisions_avg",
    ):
        b = base.get(key, 0.0)
        c = cand.get(key, 0.0)
        lines.append(f"| {key} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
    lines.append("")

    lines.append("## Status counts")
    lines.append("")
    for k, v in sorted((agg.get("status_counts") or {}).items()):
        lines.append(f"- {k}: **{v}**")
    lines.append("")

    gic = agg.get("gold_input_counts") or {}
    lines.append("## Gold-in-input diagnostics")
    lines.append("")
    lines.append(
        f"- gold_in_input (gold present in dense candidate pool): "
        f"**{gic.get('gold_in_input', 0)}**"
    )
    lines.append(
        f"- gold_was_demoted (gold was in dense top-final_k but rerank "
        f"dropped it past final_k): **{gic.get('gold_was_demoted', 0)}**"
    )
    lines.append("")

    lines.append("## Latency (ms, reranker only)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in ("mean_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms"):
        lines.append(f"| {key} | {latency.get(key, 0.0):.3f} |")
    lines.append("")

    lines.append("## By bucket")
    lines.append("")
    for bucket, payload in (agg.get("by_bucket") or {}).items():
        lines.append(f"### {bucket} (n={payload['count']})")
        lines.append("")
        lines.append("| metric | baseline | candidate | Δ |")
        lines.append("|---|---:|---:|---:|")
        for k in agg.get("k_values", _K_VALUES):
            b = payload["baseline"].get(f"hit_at_{k}", 0.0)
            c = payload["candidate"].get(f"hit_at_{k}", 0.0)
            lines.append(f"| hit@{k} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
        for key in ("mrr_at_10", "ndcg_at_10"):
            b = payload["baseline"].get(key, 0.0)
            c = payload["candidate"].get(key, 0.0)
            lines.append(f"| {key} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
        lines.append("")
        lines.append(
            "- status: " + ", ".join(
                f"{k}={v}" for k, v in sorted(
                    payload["status_counts"].items()
                )
            )
        )
        gic_b = payload.get("gold_input_counts") or {}
        lines.append(
            f"- gold_in_input={gic_b.get('gold_in_input', 0)}, "
            f"gold_was_demoted={gic_b.get('gold_was_demoted', 0)}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"
