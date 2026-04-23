"""Text RAG evaluation harness.

Takes an already-constructed Retriever + GenerationProvider — the same
two objects the production RAG capability uses — and runs them over a
JSONL dataset of `{query, expected_doc_ids?, expected_keywords?}` rows.

Why pluggable vs. building the RAG stack inside this module:

  - Tests can inject a HashingEmbedder + in-memory FaissIndex + a
    fake metadata store without touching a real model or Postgres.
  - The CLI can build the real production stack from WorkerSettings
    and hand it in unchanged.
  - This module stays a pure function of (retriever, generator,
    dataset) → report.

Emitted metrics per row:

  - hit@k            : retrieval hit rate (None if no expected_doc_ids)
  - recall_at_k      : distinct-gold recall over top-k (None if no expected_doc_ids)
  - reciprocal_rank  : 1/rank of first match, 0 if no match (None if no expected_doc_ids)
  - keyword_coverage : fraction of expected_keywords in the final answer
                       (None if no expected_keywords)
  - dup_rate         : fraction of duplicate ids in the top-k
  - topk_gap / topk_rel_gap : score headroom between rank 1 and rank k
  - retrieval_ms / generation_ms / total_ms : per-call latency

Aggregations in the summary:

  - mean_hit_at_k / mean_recall_at_k / mrr over rows that had expected_doc_ids
  - mean_keyword_coverage over rows that had expected_keywords
  - mean_dup_rate / mean_topk_gap over rows where the signal is defined
  - latency mean / p50 / p95 / max

A `misses` list (capped at 20) enumerates the rows where hit@k == 0 with
their top-3 retrieved doc_ids and scores — the quickest way to see what
a reranker or query parser would have to fix.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol

from eval.harness.metrics import (
    dup_rate,
    hit_at_k,
    keyword_coverage,
    p_percentile,
    recall_at_k,
    reciprocal_rank,
    topk_gap,
)

MISSES_CAP = 20

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal structural types so the harness doesn't import the app package
# just to annotate its parameters. Anything that quacks like a Retriever
# / GenerationProvider will work.
# ---------------------------------------------------------------------------


class _RetrieverLike(Protocol):
    def retrieve(self, query: str) -> Any: ...


class _GeneratorLike(Protocol):
    def generate(self, query: str, chunks: List[Any]) -> str: ...


# ---------------------------------------------------------------------------
# Row + summary dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class RagEvalRow:
    """One row of the generated report.

    Fields map 1:1 to the JSON/CSV output. Everything is
    JSON-serializable; list-valued fields get pipe-joined into CSV
    cells by the writer.
    """

    query: str
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    rerank_scores: List[Optional[float]] = field(default_factory=list)
    hit_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    keyword_coverage: Optional[float] = None
    dup_rate: float = 0.0
    topk_gap: Optional[float] = None
    topk_rel_gap: Optional[float] = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    candidate_k: Optional[int] = None
    use_mmr: Optional[bool] = None
    mmr_lambda: Optional[float] = None
    notes: Optional[str] = None
    answer: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RagEvalSummary:
    dataset_path: str
    row_count: int
    rows_with_expected_doc_ids: int
    rows_with_expected_keywords: int
    top_k: int
    mean_hit_at_k: Optional[float]
    mean_recall_at_k: Optional[float]
    mrr: Optional[float]
    mean_keyword_coverage: Optional[float]
    mean_dup_rate: float
    mean_topk_gap: Optional[float]
    mean_retrieval_ms: float
    p50_retrieval_ms: float
    p95_retrieval_ms: float
    max_retrieval_ms: float
    mean_generation_ms: float
    mean_total_ms: float
    error_count: int
    misses: List[Dict[str, Any]] = field(default_factory=list)
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    candidate_k: Optional[int] = None
    use_mmr: Optional[bool] = None
    mmr_lambda: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def run_rag_eval(
    dataset: List[Mapping[str, Any]],
    *,
    retriever: _RetrieverLike,
    generator: _GeneratorLike,
    top_k: int,
    dataset_path: Optional[str] = None,
    include_answer_in_report: bool = True,
    answer_excerpt_chars: int = 600,
) -> tuple[RagEvalSummary, List[RagEvalRow]]:
    """Run the RAG eval over `dataset`, returning (summary, rows).

    The harness never raises on a single-row failure: an exception in
    `retrieve` or `generate` becomes an `error` field on the row and
    an increment to `summary.error_count`. This matches how a developer
    iterates locally — one malformed query shouldn't abort the whole
    run.

    `dataset` is a list of mappings shaped like:
        {
          "query": "who runs the bookshop?",
          "expected_doc_ids": ["anime-003"],        # optional
          "expected_keywords": ["bookshop", "translator"],  # optional
          "notes": "cozy mystery"                   # optional, free-form
        }
    """
    started_at = _now_iso()
    run_start = time.perf_counter()

    rows: List[RagEvalRow] = []
    errors = 0

    for idx, raw in enumerate(dataset, start=1):
        query = _require_str(raw, "query", row_index=idx)
        expected_doc_ids = _list_of_str(raw.get("expected_doc_ids"))
        expected_keywords = _list_of_str(raw.get("expected_keywords"))
        notes = raw.get("notes")

        row = RagEvalRow(
            query=query,
            expected_doc_ids=expected_doc_ids,
            expected_keywords=expected_keywords,
            notes=str(notes) if notes is not None else None,
        )

        try:
            t0 = time.perf_counter()
            report = retriever.retrieve(query)
            t1 = time.perf_counter()
            retrieved = list(getattr(report, "results", []) or [])
            retrieved_doc_ids = [getattr(r, "doc_id", "") for r in retrieved]
            retrieval_scores = [float(getattr(r, "score", 0.0)) for r in retrieved]
            rerank_scores_raw = [getattr(r, "rerank_score", None) for r in retrieved]

            row.retrieved_doc_ids = retrieved_doc_ids
            row.retrieval_scores = [round(s, 6) for s in retrieval_scores]
            row.rerank_scores = [
                round(float(s), 6) if s is not None else None
                for s in rerank_scores_raw
            ]
            row.index_version = getattr(report, "index_version", None)
            row.embedding_model = getattr(report, "embedding_model", None)
            row.reranker_name = getattr(report, "reranker_name", None)
            row.candidate_k = getattr(report, "candidate_k", None)
            row.use_mmr = getattr(report, "use_mmr", None)
            row.mmr_lambda = getattr(report, "mmr_lambda", None)

            # Generation is inside the try/except too so a bad generator
            # doesn't abort the full eval.
            answer = generator.generate(query, retrieved)
            t2 = time.perf_counter()

            if include_answer_in_report:
                row.answer = _truncate(answer, answer_excerpt_chars)

            row.retrieval_ms = round((t1 - t0) * 1000.0, 3)
            row.generation_ms = round((t2 - t1) * 1000.0, 3)
            row.total_ms = round((t2 - t0) * 1000.0, 3)

            row.hit_at_k = hit_at_k(
                retrieved_doc_ids, expected_doc_ids, k=top_k
            )
            row.recall_at_k = recall_at_k(
                retrieved_doc_ids, expected_doc_ids, k=top_k
            )
            row.reciprocal_rank = reciprocal_rank(
                retrieved_doc_ids, expected_doc_ids
            )
            row.keyword_coverage = keyword_coverage(
                answer, expected_keywords
            )
            # Prefer the dup_rate the Retriever already computed — that
            # captures the real final top-k (post-MMR, post-rerank) and
            # stays consistent across retriever variants. Fall back to
            # the harness-side metric only for old retrievers / test
            # doubles that don't surface the field.
            report_dup = getattr(report, "dup_rate", None)
            row.dup_rate = (
                round(float(report_dup), 4)
                if report_dup is not None
                else round(dup_rate(retrieved_doc_ids[:top_k]), 4)
            )
            gap_abs, gap_rel = topk_gap(retrieval_scores[:top_k])
            row.topk_gap = round(gap_abs, 6) if gap_abs is not None else None
            row.topk_rel_gap = round(gap_rel, 6) if gap_rel is not None else None
        except Exception as ex:
            errors += 1
            row.error = f"{type(ex).__name__}: {ex}"
            log.exception("RAG eval row %d (%r) failed", idx, query)

        rows.append(row)

    run_end = time.perf_counter()

    summary = _aggregate(
        rows,
        top_k=top_k,
        dataset_path=dataset_path or "<inline>",
        errors=errors,
    )
    summary.started_at = started_at
    summary.finished_at = _now_iso()
    summary.duration_ms = round((run_end - run_start) * 1000.0, 3)

    _log_summary(summary)
    return summary, rows


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate(
    rows: List[RagEvalRow],
    *,
    top_k: int,
    dataset_path: str,
    errors: int,
) -> RagEvalSummary:
    hit_values = [r.hit_at_k for r in rows if r.hit_at_k is not None]
    recall_values = [r.recall_at_k for r in rows if r.recall_at_k is not None]
    rr_values = [r.reciprocal_rank for r in rows if r.reciprocal_rank is not None]
    kc_values = [r.keyword_coverage for r in rows if r.keyword_coverage is not None]
    dup_values = [r.dup_rate for r in rows if r.error is None]
    gap_values = [r.topk_gap for r in rows if r.topk_gap is not None]

    retrieval_latencies = [r.retrieval_ms for r in rows if r.error is None]
    generation_latencies = [r.generation_ms for r in rows if r.error is None]
    total_latencies = [r.total_ms for r in rows if r.error is None]

    # Capture the first non-None index_version / embedding_model we saw so
    # the summary header tells ops exactly which build was evaluated.
    index_version = next(
        (r.index_version for r in rows if r.index_version), None
    )
    embedding_model = next(
        (r.embedding_model for r in rows if r.embedding_model), None
    )
    reranker_name = next(
        (r.reranker_name for r in rows if r.reranker_name), None
    )
    candidate_k = next(
        (r.candidate_k for r in rows if r.candidate_k is not None), None
    )
    use_mmr = next(
        (r.use_mmr for r in rows if r.use_mmr is not None), None
    )
    mmr_lambda = next(
        (r.mmr_lambda for r in rows if r.mmr_lambda is not None), None
    )

    return RagEvalSummary(
        dataset_path=dataset_path,
        row_count=len(rows),
        rows_with_expected_doc_ids=len(hit_values),
        rows_with_expected_keywords=len(kc_values),
        top_k=top_k,
        mean_hit_at_k=_mean_or_none(hit_values),
        mean_recall_at_k=_mean_or_none(recall_values),
        mrr=_mean_or_none(rr_values),
        mean_keyword_coverage=_mean_or_none(kc_values),
        mean_dup_rate=round(statistics.fmean(dup_values), 4) if dup_values else 0.0,
        mean_topk_gap=_mean_or_none(gap_values),
        mean_retrieval_ms=_mean_or_zero(retrieval_latencies),
        p50_retrieval_ms=_p50_or_zero(retrieval_latencies),
        p95_retrieval_ms=round(p_percentile(retrieval_latencies, 95.0), 3),
        max_retrieval_ms=round(max(retrieval_latencies), 3) if retrieval_latencies else 0.0,
        mean_generation_ms=_mean_or_zero(generation_latencies),
        mean_total_ms=_mean_or_zero(total_latencies),
        error_count=errors,
        misses=_collect_misses(rows),
        index_version=index_version,
        embedding_model=embedding_model,
        reranker_name=reranker_name,
        candidate_k=candidate_k,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )


def _collect_misses(rows: List[RagEvalRow]) -> List[Dict[str, Any]]:
    """Rows with hit@k == 0, capped at `MISSES_CAP` for report size.

    Only the first three retrieved (doc_id, score) pairs are included —
    the quickest signal for "what did the retriever return instead?"
    without blowing up the JSON report on long top-k values.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.hit_at_k != 0.0:
            continue
        top3 = [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in zip(row.retrieved_doc_ids[:3], row.retrieval_scores[:3])
        ]
        out.append(
            {
                "query": row.query,
                "expected_doc_ids": list(row.expected_doc_ids),
                "top3": top3,
            }
        )
        if len(out) >= MISSES_CAP:
            break
    return out


def _log_summary(summary: RagEvalSummary) -> None:
    log.info(
        "RAG eval complete: rows=%d errors=%d hit@%d=%s recall@%d=%s mrr=%s "
        "kw_cov=%s dup=%s p95_ret_ms=%.1f mean_total_ms=%.1f misses=%d",
        summary.row_count,
        summary.error_count,
        summary.top_k,
        _fmt_opt(summary.mean_hit_at_k),
        summary.top_k,
        _fmt_opt(summary.mean_recall_at_k),
        _fmt_opt(summary.mrr),
        _fmt_opt(summary.mean_keyword_coverage),
        f"{summary.mean_dup_rate:.3f}",
        summary.p95_retrieval_ms,
        summary.mean_total_ms,
        len(summary.misses),
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _require_str(raw: Mapping[str, Any], key: str, *, row_index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Row {row_index} is missing required string field {key!r}"
        )
    return value.strip()


def _list_of_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    raise ValueError(
        f"Expected list of strings, got {type(value).__name__}: {value!r}"
    )


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.fmean(values), 3)


def _p50_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.median(values), 3)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _fmt_opt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _now_iso() -> str:
    # Local-time ISO string is plenty for a local eval log — no need to
    # drag in a timezone library. Precision to seconds is enough for
    # human-eyeball comparison between runs.
    from datetime import datetime

    return datetime.now().replace(microsecond=0).isoformat()


def summary_to_dict(summary: RagEvalSummary) -> Dict[str, Any]:
    """Public helper for the CLI / writer code."""
    return asdict(summary)


def row_to_dict(row: RagEvalRow) -> Dict[str, Any]:
    return asdict(row)


# ---------------------------------------------------------------------------
# Agent compare harness (phase 6).
#
# Runs the same dataset twice — once with the loop disabled (single-pass
# Phase 5 AUTO behaviour) and once with the loop enabled — so the Phase
# 6 metrics (loop_recovery_rate, answer_recall_delta, avg_cost_multiplier,
# iter_count_mean) have something to average over.
#
# The harness takes a single ``agent_run_fn`` callback that knows how to
# execute one row end-to-end given ``(query, loop_enabled)``. Passing the
# callback instead of building the agent stack inline keeps this module
# testable (tests can supply a deterministic fake runner) and keeps the
# CLI free to wire a live AgentCapability without bleeding its types
# into the harness.
# ---------------------------------------------------------------------------


@dataclass
class AgentCompareRow:
    """One row of the compare report.

    Carries the per-row fields needed by the Phase 6 metric helpers
    (``iter0_keyword_coverage``, ``final_keyword_coverage``,
    ``iter0_tokens``, ``total_tokens``, ``iter_count``) plus the raw
    answers so a human reviewer can eyeball what the loop rewrote. The
    loop-off snapshot is the ``iter0_*`` fields; the loop-on snapshot is
    ``final_*``.

    Phase 8 additions: ``difficulty`` + per-mode retrieval metrics
    (``{iter0,final}_{recall,mrr}``) so the decision gate can judge
    per-difficulty recovery.
    """

    query: str
    difficulty: Optional[str] = None
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    # loop=off snapshot
    iter0_answer: str = ""
    iter0_retrieved_doc_ids: List[str] = field(default_factory=list)
    iter0_recall_at_k: Optional[float] = None
    iter0_reciprocal_rank: Optional[float] = None
    iter0_keyword_coverage: Optional[float] = None
    iter0_tokens: int = 0
    iter0_ms: float = 0.0
    # loop=on snapshot
    final_answer: str = ""
    final_retrieved_doc_ids: List[str] = field(default_factory=list)
    final_recall_at_k: Optional[float] = None
    final_reciprocal_rank: Optional[float] = None
    final_keyword_coverage: Optional[float] = None
    total_tokens: int = 0
    total_ms: float = 0.0
    iter_count: int = 0
    stop_reason: Optional[str] = None
    notes: Optional[str] = None
    error: Optional[str] = None


# Signature for the per-row agent runner. Returns an ``AgentRunResult``
# dataclass so the harness doesn't have to care whether the agent is a
# real AgentCapability or a deterministic test fake.
@dataclass
class AgentRunResult:
    answer: str
    retrieved_doc_ids: List[str]
    tokens_used: int
    elapsed_ms: float
    iter_count: int
    stop_reason: Optional[str]


def run_agent_compare_eval(
    dataset: List[Mapping[str, Any]],
    *,
    agent_run_fn: Any,
    dataset_path: Optional[str] = None,
    top_k: int = 5,
) -> List[AgentCompareRow]:
    """Run the compare harness over ``dataset``.

    ``agent_run_fn(query, loop_enabled) -> AgentRunResult`` is called
    twice per row — once with ``loop_enabled=False`` (captures the
    iter0 baseline) and once with ``loop_enabled=True`` (captures the
    final loop result). The per-row ``keyword_coverage`` is computed
    against the row's ``expected_keywords`` and ``recall_at_k`` /
    ``reciprocal_rank`` against ``expected_doc_ids``.
    """
    rows: List[AgentCompareRow] = []
    for idx, raw in enumerate(dataset, start=1):
        query = _require_str(raw, "query", row_index=idx)
        expected_doc_ids = _list_of_str(raw.get("expected_doc_ids"))
        expected_keywords = _list_of_str(raw.get("expected_keywords"))
        difficulty_raw = raw.get("difficulty")
        difficulty = (
            str(difficulty_raw).strip().lower() if difficulty_raw else None
        )
        notes = raw.get("notes")

        row = AgentCompareRow(
            query=query,
            difficulty=difficulty,
            expected_doc_ids=expected_doc_ids,
            expected_keywords=expected_keywords,
            notes=str(notes) if notes is not None else None,
        )

        try:
            off = agent_run_fn(query, False)
            row.iter0_answer = off.answer or ""
            row.iter0_retrieved_doc_ids = list(off.retrieved_doc_ids or [])
            row.iter0_tokens = int(off.tokens_used or 0)
            row.iter0_ms = round(float(off.elapsed_ms or 0.0), 3)
            row.iter0_keyword_coverage = keyword_coverage(
                row.iter0_answer, expected_keywords
            )
            row.iter0_recall_at_k = recall_at_k(
                row.iter0_retrieved_doc_ids, expected_doc_ids, k=top_k
            )
            row.iter0_reciprocal_rank = reciprocal_rank(
                row.iter0_retrieved_doc_ids, expected_doc_ids
            )

            on = agent_run_fn(query, True)
            row.final_answer = on.answer or ""
            row.final_retrieved_doc_ids = list(on.retrieved_doc_ids or [])
            row.total_tokens = int(on.tokens_used or 0)
            row.total_ms = round(float(on.elapsed_ms or 0.0), 3)
            row.iter_count = int(on.iter_count or 0)
            row.stop_reason = on.stop_reason
            row.final_keyword_coverage = keyword_coverage(
                row.final_answer, expected_keywords
            )
            row.final_recall_at_k = recall_at_k(
                row.final_retrieved_doc_ids, expected_doc_ids, k=top_k
            )
            row.final_reciprocal_rank = reciprocal_rank(
                row.final_retrieved_doc_ids, expected_doc_ids
            )
        except Exception as ex:
            row.error = f"{type(ex).__name__}: {ex}"
            log.exception("AGENT compare row %d (%r) failed", idx, query)

        rows.append(row)

    log.info(
        "AGENT compare eval complete: rows=%d errors=%d dataset=%s",
        len(rows),
        sum(1 for r in rows if r.error is not None),
        dataset_path or "<inline>",
    )
    return rows


def agent_compare_row_to_dict(row: AgentCompareRow) -> Dict[str, Any]:
    return asdict(row)


# ---------------------------------------------------------------------------
# Agent compare summary (phase 8 decision gate).
# ---------------------------------------------------------------------------


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def _p_latency(values: List[float], p: float) -> float:
    from eval.harness.metrics import p_percentile

    return round(p_percentile(values, p), 3) if values else 0.0


def _bucket_stats(
    rows: List[AgentCompareRow],
    mode: str,
) -> Dict[str, Any]:
    """Per-mode slice of compare rows: retrieval + keyword + latency."""
    if mode == "off":
        recalls = [r.iter0_recall_at_k for r in rows if r.iter0_recall_at_k is not None]
        rrs = [r.iter0_reciprocal_rank for r in rows if r.iter0_reciprocal_rank is not None]
        kcs = [r.iter0_keyword_coverage for r in rows if r.iter0_keyword_coverage is not None]
        latencies = [r.iter0_ms for r in rows if r.error is None]
        tokens = [r.iter0_tokens for r in rows if r.error is None]
    else:
        recalls = [r.final_recall_at_k for r in rows if r.final_recall_at_k is not None]
        rrs = [r.final_reciprocal_rank for r in rows if r.final_reciprocal_rank is not None]
        kcs = [r.final_keyword_coverage for r in rows if r.final_keyword_coverage is not None]
        latencies = [r.total_ms for r in rows if r.error is None]
        tokens = [r.total_tokens for r in rows if r.error is None]
    return {
        "row_count": len(rows),
        "mean_recall_at_k": _mean(recalls),
        "mrr": _mean(rrs),
        "mean_keyword_coverage": _mean(kcs),
        "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "p50_latency_ms": _p_latency(latencies, 50.0),
        "p95_latency_ms": _p_latency(latencies, 95.0),
        "mean_tokens": round(statistics.fmean(tokens), 3) if tokens else 0.0,
    }


def summarize_agent_compare(
    rows: List[AgentCompareRow],
    *,
    top_k: int = 5,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate a compare run into the Phase 8 decision-gate report.

    Returns a dict with:
      - overall {off, on} stats
      - per_difficulty[{easy, hard, impossible}] {off, on}
      - agent_metrics (loop_recovery_rate[overall/hard], avg_cost_multiplier,
        iter_count_mean, answer_recall_delta)
      - stop_reason_distribution (loop=on only)
      - latency_delta (p50/p95/mean deltas on-minus-off)
      - decision inputs (numbers the gate checks against the rule)
    """
    from collections import Counter

    from eval.harness.metrics import (
        answer_recall_delta,
        avg_cost_multiplier,
        iter_count_mean,
        loop_recovery_rate,
    )

    overall = {
        "off": _bucket_stats(rows, "off"),
        "on": _bucket_stats(rows, "on"),
    }

    per_difficulty: Dict[str, Dict[str, Any]] = {}
    for tag in ("easy", "hard", "impossible"):
        slice_rows = [r for r in rows if (r.difficulty or "").lower() == tag]
        per_difficulty[tag] = {
            "off": _bucket_stats(slice_rows, "off"),
            "on": _bucket_stats(slice_rows, "on"),
        }

    hard_rows = [r for r in rows if (r.difficulty or "").lower() == "hard"]
    agent_metrics = {
        "loop_recovery_rate_overall": loop_recovery_rate(rows),
        "loop_recovery_rate_hard": loop_recovery_rate(hard_rows),
        "avg_cost_multiplier": avg_cost_multiplier(rows),
        "iter_count_mean": iter_count_mean(rows),
        "answer_recall_delta": answer_recall_delta(rows),
    }

    stop_counter: Counter = Counter()
    for r in rows:
        if r.stop_reason:
            stop_counter[r.stop_reason] += 1
    stop_total = sum(stop_counter.values())
    stop_distribution = {
        reason: {
            "count": int(count),
            "fraction": round(count / stop_total, 4) if stop_total else 0.0,
        }
        for reason, count in stop_counter.most_common()
    }

    # Latency deltas: loop_on - loop_off. p95 delta is what the gate checks
    # (<= 8s budget).
    latency_delta = {
        "mean_ms": round(
            overall["on"]["mean_latency_ms"] - overall["off"]["mean_latency_ms"],
            3,
        ),
        "p50_ms": round(
            overall["on"]["p50_latency_ms"] - overall["off"]["p50_latency_ms"],
            3,
        ),
        "p95_ms": round(
            overall["on"]["p95_latency_ms"] - overall["off"]["p95_latency_ms"],
            3,
        ),
    }

    return {
        "dataset_path": dataset_path or "<inline>",
        "row_count": len(rows),
        "error_count": sum(1 for r in rows if r.error is not None),
        "top_k": int(top_k),
        "overall": overall,
        "per_difficulty": per_difficulty,
        "agent_metrics": agent_metrics,
        "stop_reason_distribution": stop_distribution,
        "latency_delta": latency_delta,
    }
