"""Phase 2A-L accuracy ↔ latency Pareto frontier.

Walks a ``TopNSweepReport`` and labels each entry as on-frontier or
dominated by another entry. A point ``A`` *dominates* a point ``B``
when every objective of A is at least as good as B's and at least one
is strictly better. Phase 2A-L's objectives are:

  - higher ``primary_metric`` (default: hit@1; configurable to mrr@10)
  - lower ``primary_latency`` (default: rerank_p95_ms; configurable to
    total_query_p95_ms / total_query_p99_ms)

Pure post-processing — same input contract as the sweep aggregator.
The output is a list of ``ParetoPoint`` records each carrying the
sweep entry's headline numbers plus a ``dominated_by`` field naming
which entry (if any) dominates it.

Public surface:

  - ``ParetoObjective`` (Enum-like literal contract)
  - ``ParetoPoint``     — one annotated entry
  - ``ParetoFrontierReport``
  - ``compute_pareto_frontier(sweep, *, metric, latency)``
  - ``render_pareto_markdown(report)``
  - ``pareto_to_dict(report)``
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from eval.harness.topn_sweep import TopNSweepEntry, TopNSweepReport

log = logging.getLogger(__name__)


SCHEMA_VERSION = "phase2a-pareto-frontier.v1"


# Names of metric series the sweep entry can supply for the "higher is
# better" objective. Restricting the set keeps the report contract
# obvious — a typo in the CLI flag fails fast instead of producing a
# silently-empty frontier.
ACCURACY_METRICS = (
    "mean_hit_at_1",
    "mean_hit_at_3",
    "mean_hit_at_5",
    "mean_mrr_at_10",
    "mean_ndcg_at_10",
)

# Names of latency series the entry can supply for the "lower is better"
# objective. The default for Phase 2A-L is ``rerank_p95_ms`` (the
# reranker bottleneck the sweep was designed to investigate); the
# CLI exposes the others so a downstream user can re-run the same data
# against total-query budgets.
LATENCY_METRICS = (
    "rerank_avg_ms",
    "rerank_p50_ms",
    "rerank_p90_ms",
    "rerank_p95_ms",
    "rerank_p99_ms",
    "total_query_avg_ms",
    "total_query_p50_ms",
    "total_query_p90_ms",
    "total_query_p95_ms",
    "total_query_p99_ms",
)


@dataclass
class ParetoPoint:
    """One sweep entry with frontier annotations.

    ``metric`` and ``latency`` are the values used to compute dominance
    (round-tripped from the sweep entry); ``on_frontier`` is True when
    the point is non-dominated; ``dominated_by`` names the
    dense_top_n / label of the entry that dominates this one (None on
    frontier). Both objectives are required for dominance — entries
    missing either are excluded from the frontier and surface as
    ``on_frontier = False`` with ``dominated_by = "(missing-data)"``.
    """

    label: str
    dense_top_n: Optional[int]
    metric: Optional[float]
    latency_ms: Optional[float]
    on_frontier: bool
    dominated_by: Optional[str]
    # Carry the headline metrics + key latency series for the markdown
    # writer so the frontier report doesn't force the reader to flip
    # back to the sweep document. All Optional because individual
    # entries may not have populated each field.
    mean_hit_at_1: Optional[float]
    mean_hit_at_3: Optional[float]
    mean_hit_at_5: Optional[float]
    mean_mrr_at_10: Optional[float]
    mean_ndcg_at_10: Optional[float]
    rerank_p95_ms: Optional[float]
    rerank_p99_ms: Optional[float]
    total_query_p95_ms: Optional[float]
    total_query_p99_ms: Optional[float]


@dataclass
class ParetoFrontierReport:
    schema: str
    metric_field: str
    latency_field: str
    entries: List[ParetoPoint] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def _get_field(entry: TopNSweepEntry, field_name: str) -> Optional[float]:
    """Read a numeric field off a sweep entry, or ``None``."""
    value = getattr(entry, field_name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_pareto_frontier(
    sweep: TopNSweepReport,
    *,
    metric: str = "mean_hit_at_1",
    latency: str = "rerank_p95_ms",
    caveats: Optional[List[str]] = None,
) -> ParetoFrontierReport:
    """Compute the accuracy ↔ latency Pareto frontier for a sweep.

    Two-objective dominance: A dominates B when ``A.metric >= B.metric``
    AND ``A.latency <= B.latency`` AND at least one of the two is
    strictly better. Ties on both objectives produce mutually-undominated
    points — the frontier keeps both, which matches the standard
    multi-objective definition.
    """
    if metric not in ACCURACY_METRICS:
        raise ValueError(
            f"Unknown accuracy metric {metric!r}; expected one of "
            f"{ACCURACY_METRICS}."
        )
    if latency not in LATENCY_METRICS:
        raise ValueError(
            f"Unknown latency metric {latency!r}; expected one of "
            f"{LATENCY_METRICS}."
        )

    raw_points: List[Dict[str, Any]] = []
    for e in sweep.entries:
        m = _get_field(e, metric)
        l = _get_field(e, latency)
        raw_points.append({"entry": e, "metric": m, "latency": l})

    annotated: List[ParetoPoint] = []
    for point in raw_points:
        e: TopNSweepEntry = point["entry"]
        m = point["metric"]
        l = point["latency"]
        if m is None or l is None:
            annotated.append(
                ParetoPoint(
                    label=e.label,
                    dense_top_n=e.dense_top_n,
                    metric=m,
                    latency_ms=l,
                    on_frontier=False,
                    dominated_by="(missing-data)",
                    mean_hit_at_1=e.mean_hit_at_1,
                    mean_hit_at_3=e.mean_hit_at_3,
                    mean_hit_at_5=e.mean_hit_at_5,
                    mean_mrr_at_10=e.mean_mrr_at_10,
                    mean_ndcg_at_10=e.mean_ndcg_at_10,
                    rerank_p95_ms=e.rerank_p95_ms,
                    rerank_p99_ms=e.rerank_p99_ms,
                    total_query_p95_ms=e.total_query_p95_ms,
                    total_query_p99_ms=e.total_query_p99_ms,
                )
            )
            continue

        dominator: Optional[str] = None
        for other in raw_points:
            if other is point:
                continue
            om = other["metric"]
            ol = other["latency"]
            if om is None or ol is None:
                continue
            # other dominates point iff:
            #   om >= m AND ol <= l AND (om > m OR ol < l)
            if om >= m and ol <= l and (om > m or ol < l):
                other_e: TopNSweepEntry = other["entry"]
                dominator = (
                    f"{other_e.label} (dense_top_n="
                    f"{other_e.dense_top_n})"
                )
                break

        annotated.append(
            ParetoPoint(
                label=e.label,
                dense_top_n=e.dense_top_n,
                metric=m,
                latency_ms=l,
                on_frontier=dominator is None,
                dominated_by=dominator,
                mean_hit_at_1=e.mean_hit_at_1,
                mean_hit_at_3=e.mean_hit_at_3,
                mean_hit_at_5=e.mean_hit_at_5,
                mean_mrr_at_10=e.mean_mrr_at_10,
                mean_ndcg_at_10=e.mean_ndcg_at_10,
                rerank_p95_ms=e.rerank_p95_ms,
                rerank_p99_ms=e.rerank_p99_ms,
                total_query_p95_ms=e.total_query_p95_ms,
                total_query_p99_ms=e.total_query_p99_ms,
            )
        )

    # Sort: frontier points first by ascending latency, then dominated
    # points in the same latency order. The headline reading order is
    # "fast → slow" which is what an operator picks against.
    annotated.sort(
        key=lambda p: (
            0 if p.on_frontier else 1,
            (p.latency_ms if p.latency_ms is not None else float("inf")),
        )
    )

    return ParetoFrontierReport(
        schema=SCHEMA_VERSION,
        metric_field=metric,
        latency_field=latency,
        entries=annotated,
        caveats=list(caveats or _default_caveats()),
    )


def _default_caveats() -> List[str]:
    return [
        "Pareto frontier는 두 objective(accuracy↑, latency↓)에 대해서만 dominance를 판정한다. 실제 운영 결정은 quality variance, GPU memory, batch_size 등 추가 제약을 함께 본다.",
        "primary_metric / primary_latency는 CLI에서 변경 가능하다 — 보통 hit@1 + rerank_p95_ms로 본다.",
        "tie의 경우(metric/latency 모두 동일) 양쪽 다 frontier에 남는다.",
        "missing 데이터(metric/latency 둘 중 하나가 None)인 entry는 frontier 비교에서 제외되고 (missing-data) 라벨을 받는다.",
    ]


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


def pareto_to_dict(report: ParetoFrontierReport) -> Dict[str, Any]:
    return asdict(report)


# ---------------------------------------------------------------------------
# Markdown.
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


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


def render_pareto_markdown(report: ParetoFrontierReport) -> str:
    lines: List[str] = []
    lines.append("# Phase 2A-L accuracy ↔ latency Pareto frontier")
    lines.append("")
    lines.append(
        f"- objectives: maximise `{report.metric_field}` (↑) "
        f"vs minimise `{report.latency_field}` (↓)"
    )
    lines.append(f"- entries: {len(report.entries)}")
    on_frontier = [e for e in report.entries if e.on_frontier]
    lines.append(f"- on frontier: {len(on_frontier)}")
    dominated = [
        e for e in report.entries
        if not e.on_frontier and e.dominated_by != "(missing-data)"
    ]
    lines.append(f"- dominated: {len(dominated)}")
    lines.append("")

    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")

    if not report.entries:
        lines.append("_No entries to plot._")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Frontier")
    lines.append("")
    lines.append(
        "| label | dense_top_n | metric | latency_ms | on_frontier | dominated_by |"
    )
    lines.append(
        "|---|---:|---:|---:|:-:|---|"
    )
    for e in report.entries:
        marker = "✓" if e.on_frontier else ""
        lines.append(
            f"| {e.label} | "
            f"{_fmt_or_dash_int(e.dense_top_n)} | "
            f"{_fmt(e.metric)} | "
            f"{_fmt_ms(e.latency_ms)} | "
            f"{marker} | "
            f"{e.dominated_by or '-'} |"
        )
    lines.append("")

    lines.append("## Headline metrics + latency for every entry")
    lines.append("")
    lines.append(
        "| label | dense_top_n | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
        "rerank_p95 | rerank_p99 | total_p95 | total_p99 |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for e in report.entries:
        lines.append(
            f"| {e.label} | "
            f"{_fmt_or_dash_int(e.dense_top_n)} | "
            f"{_fmt(e.mean_hit_at_1)} | "
            f"{_fmt(e.mean_hit_at_3)} | "
            f"{_fmt(e.mean_hit_at_5)} | "
            f"{_fmt(e.mean_mrr_at_10)} | "
            f"{_fmt(e.mean_ndcg_at_10)} | "
            f"{_fmt_ms(e.rerank_p95_ms)} | "
            f"{_fmt_ms(e.rerank_p99_ms)} | "
            f"{_fmt_ms(e.total_query_p95_ms)} | "
            f"{_fmt_ms(e.total_query_p99_ms)} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"
