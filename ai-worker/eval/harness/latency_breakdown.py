"""Phase 2A-L latency-breakdown analysis.

Reads a single retrieval_eval_report.json (typically a retrieval-rerank
run with ``collect_stage_timings`` on) and emits a per-stage latency
report covering:

  - dense_retrieval_ms      (FAISS + embedder, before the reranker)
  - pair_build_ms           ((query, passage) construction, host-side)
  - tokenize_ms             (cross-encoder tokenizer)
  - forward_ms              (cross-encoder model forward + activation)
  - postprocess_ms          (sort + slice top-k after reranker)
  - total_rerank_ms         (sum of the four reranker stages)
  - total_query_ms          (full retrieve() wall-clock — the row's
                             ``retrieval_ms`` field)

For each stage the report computes ``avg / p50 / p90 / p95 / p99 / max``
plus the row count that contributed. Pure post-processing — never
re-runs retrieval, never touches CUDA — so the analysis pass is fast
and reproducible from any committed report on disk.

Public surface:

  - ``LatencyStats``                 — per-stage stats dataclass
  - ``LatencyBreakdownReport``       — full report dataclass (all stages)
  - ``compute_latency_stats(values)``
  - ``build_latency_breakdown(report_path)``
  - ``render_latency_breakdown_markdown(report)``
  - ``latency_breakdown_to_dict(report)``

Schema string ``phase2a-latency-breakdown.v1`` is pinned on every
emitted JSON document so downstream tooling can assert the contract.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

log = logging.getLogger(__name__)


SCHEMA_VERSION = "phase2a-latency-breakdown.v1"

# Pipeline-order known stages; the writer renders them in this sequence
# regardless of dict iteration order. Other stages fall to the bottom
# alphabetically so future additions never break the headline order.
KNOWN_STAGES: tuple = (
    "dense_retrieval_ms",
    "pair_build_ms",
    "tokenize_ms",
    "forward_ms",
    "postprocess_ms",
    "total_rerank_ms",
    "total_query_ms",
)

# Pin the percentile cutoffs the headline report quotes; the helpers
# accept arbitrary cutoffs but the standard report shape is fixed so
# the markdown table column layout never drifts between runs.
HEADLINE_PERCENTILES: tuple = (50.0, 90.0, 95.0, 99.0)


# ---------------------------------------------------------------------------
# Stats helpers.
# ---------------------------------------------------------------------------


def _percentile_nearest_rank(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of ``values`` (0.0 on empty input).

    Mirrors ``eval.harness.metrics.p_percentile`` so the breakdown
    aggregator quotes percentiles consistent with the rest of the
    retrieval eval harness — same ceil-based index, same edge-case
    behaviour. Imported lazily to keep this module a leaf in the
    dependency graph.
    """
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(xs))) - 1
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


@dataclass
class LatencyStats:
    """Per-stage latency aggregate.

    All fields are floats (or None for empty input). ``count`` records
    how many rows contributed a measurement so a stage that only fired
    on some rows (e.g. rerank breakdown on the OOM-fallback path) is
    distinguishable from the empty case.
    """

    stage: str
    count: int
    avg_ms: Optional[float]
    p50_ms: Optional[float]
    p90_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    max_ms: Optional[float]


def compute_latency_stats(stage: str, values: Sequence[float]) -> LatencyStats:
    """Build a ``LatencyStats`` from a list of per-row measurements.

    Empty input yields an all-None stats record with ``count=0`` so the
    JSON shape stays stable even when no row contributed (e.g. a noop
    run with the breakdown aggregator pointed at it).
    """
    cleaned: List[float] = []
    for v in values:
        try:
            cleaned.append(float(v))
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return LatencyStats(
            stage=stage,
            count=0,
            avg_ms=None,
            p50_ms=None,
            p90_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
        )
    return LatencyStats(
        stage=stage,
        count=len(cleaned),
        avg_ms=round(statistics.fmean(cleaned), 3),
        p50_ms=round(statistics.median(cleaned), 3),
        p90_ms=round(_percentile_nearest_rank(cleaned, 90.0), 3),
        p95_ms=round(_percentile_nearest_rank(cleaned, 95.0), 3),
        p99_ms=round(_percentile_nearest_rank(cleaned, 99.0), 3),
        max_ms=round(max(cleaned), 3),
    )


@dataclass
class LatencyBreakdownReport:
    """Top-level latency-breakdown report (one retrieval-rerank run).

    Caller-relevant provenance fields (``corpus_path``, ``dense_top_n``,
    ``final_top_k``, ``reranker_model``) are read from the run's
    metadata + summary blocks. ``stages`` is a dict keyed on the stage
    name with one ``LatencyStats`` per stage that has at least one
    measurement.
    """

    schema: str
    report_path: str
    label: Optional[str]
    corpus_path: Optional[str]
    reranker_name: Optional[str]
    reranker_model: Optional[str]
    dense_top_n: Optional[int]
    final_top_k: Optional[int]
    reranker_batch_size: Optional[int]
    row_count: int
    rows_with_rerank_breakdown: int
    rows_with_dense_retrieval_ms: int
    stages: Dict[str, LatencyStats] = field(default_factory=dict)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_latency_breakdown(
    report_path: Path,
    *,
    label: Optional[str] = None,
) -> LatencyBreakdownReport:
    """Read a retrieval_eval_report.json and emit a breakdown report.

    The function never re-runs retrieval — it walks the rows already
    persisted and computes per-stage stats over their measurements.
    Rows with ``error`` set are skipped (the wall-clock measurement is
    meaningless when the retrieve() raised). Stages absent from every
    row are absent from the output (the JSON shape is "what we
    measured", not "all possible stages").
    """
    p = Path(report_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Retrieval report not found: {p}. The latency-breakdown "
            f"analysis is purely post-processing — point it at the "
            f"out-dir of a finished retrieval-rerank run."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    summary = dict(payload.get("summary") or {})
    rows = list(payload.get("rows") or [])

    valid_rows = [r for r in rows if not r.get("error")]

    # Per-row latencies for the headline series. ``retrieval_ms`` is
    # always present; ``dense_retrieval_ms`` is Phase 2A-L's addition
    # and may be absent on older runs (the helper just produces an
    # empty-stats record in that case).
    total_query: List[float] = []
    dense: List[float] = []
    rerank_breakdown_buckets: Dict[str, List[float]] = {}
    rerank_total_per_row: List[float] = []

    for r in valid_rows:
        try:
            total_query.append(float(r.get("retrieval_ms")))
        except (TypeError, ValueError):
            pass
        dense_value = r.get("dense_retrieval_ms")
        if dense_value is not None:
            try:
                dense.append(float(dense_value))
            except (TypeError, ValueError):
                pass
        breakdown = r.get("rerank_breakdown_ms")
        if isinstance(breakdown, dict) and breakdown:
            for stage, value in breakdown.items():
                try:
                    rerank_breakdown_buckets.setdefault(
                        str(stage), []
                    ).append(float(value))
                except (TypeError, ValueError):
                    continue
            tot = breakdown.get("total_rerank_ms")
            if tot is not None:
                try:
                    rerank_total_per_row.append(float(tot))
                except (TypeError, ValueError):
                    pass

    stages: Dict[str, LatencyStats] = {}
    if dense:
        stages["dense_retrieval_ms"] = compute_latency_stats(
            "dense_retrieval_ms", dense,
        )
    for stage_name, values in rerank_breakdown_buckets.items():
        stages[stage_name] = compute_latency_stats(stage_name, values)
    if total_query:
        stages["total_query_ms"] = compute_latency_stats(
            "total_query_ms", total_query,
        )

    return LatencyBreakdownReport(
        schema=SCHEMA_VERSION,
        report_path=str(p),
        label=label,
        corpus_path=summary.get("corpus_path") or metadata.get("corpus_path"),
        reranker_name=summary.get("reranker_name"),
        reranker_model=metadata.get("reranker_model"),
        dense_top_n=_safe_int(metadata.get("dense_top_n") or metadata.get("candidate_k")),
        final_top_k=_safe_int(metadata.get("final_top_k") or summary.get("top_k")),
        reranker_batch_size=_safe_int(metadata.get("reranker_batch_size")),
        row_count=len(valid_rows),
        rows_with_rerank_breakdown=len(rerank_total_per_row),
        rows_with_dense_retrieval_ms=len(dense),
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


def latency_breakdown_to_dict(report: LatencyBreakdownReport) -> Dict[str, Any]:
    """asdict() with the stage map preserved as a stable nested dict.

    ``dataclasses.asdict`` already recurses through ``LatencyStats``, but
    the nested dict keys come out in dataclass insertion order. Re-walk
    the result so the headline pipeline order (``KNOWN_STAGES`` first,
    others alphabetical) is what's persisted on disk — keeps the JSON
    diff small between runs.
    """
    raw = asdict(report)
    stages_in = dict(raw.get("stages") or {})
    ordered = []
    seen = set()
    for stage in KNOWN_STAGES:
        if stage in stages_in:
            ordered.append((stage, stages_in[stage]))
            seen.add(stage)
    for stage in sorted(k for k in stages_in.keys() if k not in seen):
        ordered.append((stage, stages_in[stage]))
    raw["stages"] = {k: v for k, v in ordered}
    return raw


# ---------------------------------------------------------------------------
# Markdown.
# ---------------------------------------------------------------------------


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_or_dash_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def render_latency_breakdown_markdown(
    report: LatencyBreakdownReport,
) -> str:
    """Render the breakdown report as a one-page markdown table."""
    lines: List[str] = []
    lines.append("# Phase 2A-L reranker latency breakdown")
    lines.append("")
    lines.append(
        "Stage-level latency profile for one retrieval-rerank run. "
        "Pure post-processing — never re-runs retrieval. CUDA "
        "synchronize is applied around the cross-encoder forward "
        "pass when the run was on a GPU; CPU-only runs see "
        "approximate forward times since async semantics don't apply."
    )
    lines.append("")
    if report.label:
        lines.append(f"- label: `{report.label}`")
    lines.append(f"- report: `{report.report_path}`")
    if report.corpus_path:
        lines.append(f"- corpus: `{report.corpus_path}`")
    lines.append(f"- reranker_name: {report.reranker_name or '-'}")
    if report.reranker_model:
        lines.append(f"- reranker_model: `{report.reranker_model}`")
    lines.append(
        f"- dense_top_n: {_fmt_or_dash_int(report.dense_top_n)} | "
        f"final_top_k: {_fmt_or_dash_int(report.final_top_k)} | "
        f"reranker_batch_size: {_fmt_or_dash_int(report.reranker_batch_size)}"
    )
    lines.append(f"- rows: {report.row_count}")
    lines.append(
        f"- rows_with_rerank_breakdown: {report.rows_with_rerank_breakdown}"
    )
    lines.append(
        f"- rows_with_dense_retrieval_ms: {report.rows_with_dense_retrieval_ms}"
    )
    lines.append("")

    if not report.stages:
        lines.append(
            "_No stages produced measurements — the run likely used the "
            "noop reranker or the cross-encoder was on the OOM-fallback "
            "path._"
        )
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Per-stage latency (ms)")
    lines.append("")
    lines.append("| stage | n | avg | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    ordered_keys: List[str] = []
    seen: set = set()
    for stage in KNOWN_STAGES:
        if stage in report.stages:
            ordered_keys.append(stage)
            seen.add(stage)
    ordered_keys.extend(sorted(k for k in report.stages.keys() if k not in seen))

    for stage in ordered_keys:
        stats = report.stages[stage]
        lines.append(
            f"| `{stage}` | "
            f"{stats.count} | "
            f"{_fmt_ms(stats.avg_ms)} | "
            f"{_fmt_ms(stats.p50_ms)} | "
            f"{_fmt_ms(stats.p90_ms)} | "
            f"{_fmt_ms(stats.p95_ms)} | "
            f"{_fmt_ms(stats.p99_ms)} | "
            f"{_fmt_ms(stats.max_ms)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
