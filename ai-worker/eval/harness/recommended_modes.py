"""Phase 2A-L recommended operating modes.

Picks three configurations — ``fast`` / ``balanced`` / ``quality`` —
out of a topN sweep + Pareto frontier and emits a one-page recommendation.

Selection strategy:

  - ``fast``     : the on-frontier entry with the lowest p95 latency.
                   Optionally, a dense-only baseline (dense_top_n
                   small / no rerank) can be passed in via the sweep
                   if such a slice is included.
  - ``balanced`` : the on-frontier entry that maximises ``primary_metric``
                   subject to ``p95_latency <= balanced_p95_budget_ms``.
                   When the budget is None the helper picks the
                   on-frontier entry with the median rank by latency,
                   which is the natural midpoint between fast / quality.
  - ``quality``  : the on-frontier entry with the highest
                   ``primary_metric``. Acceptable when the operator can
                   tolerate the resulting p95 latency.

The recommendation is a guidance document, not a config switch — the
JSON shape includes every candidate the helper considered so a human
reviewer can override at any tier.

Public surface:

  - ``RecommendedMode``
  - ``RecommendedModesReport``
  - ``recommend_modes(sweep, frontier, ...)``
  - ``render_recommended_modes_markdown(report)``
  - ``recommended_modes_to_dict(report)``
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from eval.harness.pareto_frontier import ParetoFrontierReport, ParetoPoint
from eval.harness.topn_sweep import TopNSweepEntry, TopNSweepReport

log = logging.getLogger(__name__)


SCHEMA_VERSION = "phase2a-recommended-modes.v1"

# Tier names — kept in this fixed order in every emitted document.
TIER_ORDER = ("fast", "balanced", "quality")


@dataclass
class RecommendedMode:
    """One operating-mode recommendation.

    ``rationale`` is a short string explaining why this mode picked the
    entry it did (e.g. "lowest on-frontier p95 latency"). ``notes`` may
    surface extra caveats (e.g. "the cheaper top10 was within ε of
    top15 — using top15 because tie-breaking favoured higher hit@1").
    """

    tier: str
    label: Optional[str]
    dense_top_n: Optional[int]
    final_top_k: Optional[int]
    primary_metric_field: str
    primary_metric_value: Optional[float]
    latency_field: str
    latency_p95_ms: Optional[float]
    rationale: str
    notes: List[str] = field(default_factory=list)
    # Carry the full headline metrics so the markdown writer can show
    # the operator the trade-off without reading the sweep document.
    mean_hit_at_1: Optional[float] = None
    mean_hit_at_3: Optional[float] = None
    mean_hit_at_5: Optional[float] = None
    mean_mrr_at_10: Optional[float] = None
    mean_ndcg_at_10: Optional[float] = None
    rerank_p95_ms: Optional[float] = None
    rerank_p99_ms: Optional[float] = None
    total_query_p95_ms: Optional[float] = None
    total_query_p99_ms: Optional[float] = None


@dataclass
class RecommendedModesReport:
    schema: str
    primary_metric_field: str
    latency_field: str
    fast_p95_budget_ms: Optional[float]
    balanced_p95_budget_ms: Optional[float]
    quality_target_metric: Optional[float]
    modes: List[RecommendedMode] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def _entry_by_label(
    sweep: TopNSweepReport, label: str
) -> Optional[TopNSweepEntry]:
    for e in sweep.entries:
        if e.label == label:
            return e
    return None


def _frontier_points(frontier: ParetoFrontierReport) -> List[ParetoPoint]:
    return [
        e for e in frontier.entries
        if e.on_frontier and e.metric is not None and e.latency_ms is not None
    ]


def _fill_mode_from_entry(
    *,
    tier: str,
    rationale: str,
    point: ParetoPoint,
    sweep_entry: Optional[TopNSweepEntry],
    primary_metric_field: str,
    latency_field: str,
    notes: Optional[List[str]] = None,
) -> RecommendedMode:
    final_top_k = sweep_entry.final_top_k if sweep_entry is not None else None
    return RecommendedMode(
        tier=tier,
        label=point.label,
        dense_top_n=point.dense_top_n,
        final_top_k=final_top_k,
        primary_metric_field=primary_metric_field,
        primary_metric_value=point.metric,
        latency_field=latency_field,
        latency_p95_ms=point.latency_ms,
        rationale=rationale,
        notes=list(notes or []),
        mean_hit_at_1=point.mean_hit_at_1,
        mean_hit_at_3=point.mean_hit_at_3,
        mean_hit_at_5=point.mean_hit_at_5,
        mean_mrr_at_10=point.mean_mrr_at_10,
        mean_ndcg_at_10=point.mean_ndcg_at_10,
        rerank_p95_ms=point.rerank_p95_ms,
        rerank_p99_ms=point.rerank_p99_ms,
        total_query_p95_ms=point.total_query_p95_ms,
        total_query_p99_ms=point.total_query_p99_ms,
    )


def recommend_modes(
    sweep: TopNSweepReport,
    frontier: ParetoFrontierReport,
    *,
    fast_p95_budget_ms: Optional[float] = None,
    balanced_p95_budget_ms: Optional[float] = None,
    quality_target_metric: Optional[float] = None,
    caveats: Optional[List[str]] = None,
) -> RecommendedModesReport:
    """Emit fast / balanced / quality recommendations from the frontier.

    Selection rules:

      - ``fast``    : on-frontier entry with the lowest latency. When
                      ``fast_p95_budget_ms`` is provided the helper
                      prefers entries with latency ≤ budget; if none
                      qualify it surfaces the closest entry and adds a
                      "exceeds budget" note.
      - ``balanced``: on-frontier entry maximising ``primary_metric``
                      subject to latency ≤ ``balanced_p95_budget_ms``.
                      Default budget is ``None`` → median-rank entry by
                      latency.
      - ``quality`` : on-frontier entry with the highest ``primary_metric``.
                      ``quality_target_metric`` only adds a note when
                      the chosen entry doesn't reach the target.

    Empty / single-point frontiers degrade gracefully — every mode that
    can't be picked surfaces with ``label = None`` and an explanatory
    rationale. The JSON shape stays the same so downstream tooling
    doesn't have to special-case the missing case.
    """
    primary_metric_field = frontier.metric_field
    latency_field = frontier.latency_field

    points = _frontier_points(frontier)

    modes: List[RecommendedMode] = []

    # FAST
    if not points:
        modes.append(
            RecommendedMode(
                tier="fast",
                label=None,
                dense_top_n=None,
                final_top_k=None,
                primary_metric_field=primary_metric_field,
                primary_metric_value=None,
                latency_field=latency_field,
                latency_p95_ms=None,
                rationale=(
                    "no Pareto-frontier points are available; the sweep "
                    "produced no entry with both metric and latency."
                ),
            )
        )
    else:
        sorted_by_latency = sorted(
            points, key=lambda p: p.latency_ms or float("inf")
        )
        if fast_p95_budget_ms is not None:
            within_budget = [
                p for p in sorted_by_latency
                if p.latency_ms is not None and p.latency_ms <= fast_p95_budget_ms
            ]
            if within_budget:
                # Within budget: pick the entry that maximises metric so
                # we don't sacrifice quality unnecessarily — fastest
                # within budget can be the same point or a slower one
                # with higher metric.
                pick = max(
                    within_budget,
                    key=lambda p: (
                        p.metric or float("-inf"),
                        -(p.latency_ms or float("inf")),
                    ),
                )
                rationale = (
                    f"on-frontier entry with the highest "
                    f"{primary_metric_field} that fits the "
                    f"fast_p95_budget_ms={fast_p95_budget_ms:.2f}."
                )
                notes: List[str] = []
            else:
                pick = sorted_by_latency[0]
                rationale = (
                    f"no on-frontier entry meets "
                    f"fast_p95_budget_ms={fast_p95_budget_ms:.2f}; "
                    f"falling back to the lowest-latency frontier point."
                )
                notes = [
                    f"latency_p95_ms={pick.latency_ms:.2f} exceeds "
                    f"the fast budget."
                ]
        else:
            pick = sorted_by_latency[0]
            rationale = "lowest-latency on-frontier entry."
            notes = []
        modes.append(
            _fill_mode_from_entry(
                tier="fast",
                rationale=rationale,
                point=pick,
                sweep_entry=_entry_by_label(sweep, pick.label),
                primary_metric_field=primary_metric_field,
                latency_field=latency_field,
                notes=notes,
            )
        )

    # BALANCED
    if not points:
        modes.append(
            RecommendedMode(
                tier="balanced",
                label=None,
                dense_top_n=None,
                final_top_k=None,
                primary_metric_field=primary_metric_field,
                primary_metric_value=None,
                latency_field=latency_field,
                latency_p95_ms=None,
                rationale="no Pareto-frontier points available.",
            )
        )
    else:
        if balanced_p95_budget_ms is not None:
            within_budget = [
                p for p in points
                if p.latency_ms is not None
                and p.latency_ms <= balanced_p95_budget_ms
            ]
            if within_budget:
                pick = max(
                    within_budget,
                    key=lambda p: (
                        p.metric or float("-inf"),
                        -(p.latency_ms or float("inf")),
                    ),
                )
                rationale = (
                    f"on-frontier entry maximising {primary_metric_field} "
                    f"subject to latency ≤ {balanced_p95_budget_ms:.2f} ms."
                )
                notes = []
            else:
                # No frontier point fits; degrade to the lowest-latency
                # frontier point and flag.
                pick = min(points, key=lambda p: p.latency_ms or float("inf"))
                rationale = (
                    f"no on-frontier entry meets "
                    f"balanced_p95_budget_ms={balanced_p95_budget_ms:.2f}; "
                    f"falling back to lowest-latency frontier point."
                )
                notes = [
                    f"latency_p95_ms={pick.latency_ms:.2f} exceeds "
                    f"the balanced budget."
                ]
        else:
            # No budget set — pick the median-rank frontier point by
            # latency. For an even number of points we pick the one
            # closer to the high-metric end so balanced still trends
            # toward quality rather than fast.
            sorted_pts = sorted(
                points, key=lambda p: p.latency_ms or float("inf"),
            )
            mid_idx = len(sorted_pts) // 2
            if mid_idx >= len(sorted_pts):
                mid_idx = len(sorted_pts) - 1
            pick = sorted_pts[mid_idx]
            rationale = (
                "median-latency on-frontier entry "
                f"({mid_idx + 1} of {len(sorted_pts)} sorted by latency)."
            )
            notes = []
        modes.append(
            _fill_mode_from_entry(
                tier="balanced",
                rationale=rationale,
                point=pick,
                sweep_entry=_entry_by_label(sweep, pick.label),
                primary_metric_field=primary_metric_field,
                latency_field=latency_field,
                notes=notes,
            )
        )

    # QUALITY
    if not points:
        modes.append(
            RecommendedMode(
                tier="quality",
                label=None,
                dense_top_n=None,
                final_top_k=None,
                primary_metric_field=primary_metric_field,
                primary_metric_value=None,
                latency_field=latency_field,
                latency_p95_ms=None,
                rationale="no Pareto-frontier points available.",
            )
        )
    else:
        pick = max(
            points,
            key=lambda p: (
                p.metric or float("-inf"),
                -(p.latency_ms or float("inf")),
            ),
        )
        rationale = f"on-frontier entry with the highest {primary_metric_field}."
        notes = []
        if quality_target_metric is not None and pick.metric is not None:
            if pick.metric < quality_target_metric:
                notes.append(
                    f"{primary_metric_field}={pick.metric:.4f} is below "
                    f"the requested quality_target_metric="
                    f"{quality_target_metric:.4f}."
                )
        modes.append(
            _fill_mode_from_entry(
                tier="quality",
                rationale=rationale,
                point=pick,
                sweep_entry=_entry_by_label(sweep, pick.label),
                primary_metric_field=primary_metric_field,
                latency_field=latency_field,
                notes=notes,
            )
        )

    return RecommendedModesReport(
        schema=SCHEMA_VERSION,
        primary_metric_field=primary_metric_field,
        latency_field=latency_field,
        fast_p95_budget_ms=fast_p95_budget_ms,
        balanced_p95_budget_ms=balanced_p95_budget_ms,
        quality_target_metric=quality_target_metric,
        modes=modes,
        caveats=list(caveats or _default_caveats()),
    )


def _default_caveats() -> List[str]:
    return [
        "추천 모드는 sweep + frontier로부터 자동 선정한 가이드일 뿐 production default를 변경하지 않는다.",
        "Pareto-dominated configuration은 무조건 후보에서 제외되므로, fast/balanced/quality는 항상 frontier 위에서 선택된다.",
        "balanced 기본값은 latency 중앙값(frontier의 정렬된 중간 entry)이며, --balanced-p95-budget-ms로 조정 가능하다.",
        "fast 모드도 budget이 명시되면 그 안에서 metric을 최대화하는 frontier point를 고른다 — 무조건 가장 빠른 점은 아니다.",
        "quality 모드는 정확도 최우선; quality_target_metric에 미달할 경우 notes에 명시된다.",
    ]


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


def recommended_modes_to_dict(report: RecommendedModesReport) -> Dict[str, Any]:
    raw = asdict(report)
    # Pin tier order regardless of dict iteration order.
    by_tier = {m["tier"]: m for m in raw.get("modes", [])}
    raw["modes"] = [
        by_tier[t] for t in TIER_ORDER if t in by_tier
    ] + [
        m for m in raw.get("modes", []) if m["tier"] not in TIER_ORDER
    ]
    return raw


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


def render_recommended_modes_markdown(
    report: RecommendedModesReport,
) -> str:
    lines: List[str] = []
    lines.append("# Phase 2A-L recommended operating modes")
    lines.append("")
    lines.append(
        f"- primary_metric: `{report.primary_metric_field}` (↑) | "
        f"latency: `{report.latency_field}` (↓)"
    )
    if report.fast_p95_budget_ms is not None:
        lines.append(
            f"- fast_p95_budget_ms: {report.fast_p95_budget_ms:.2f}"
        )
    if report.balanced_p95_budget_ms is not None:
        lines.append(
            f"- balanced_p95_budget_ms: {report.balanced_p95_budget_ms:.2f}"
        )
    if report.quality_target_metric is not None:
        lines.append(
            f"- quality_target_metric: {report.quality_target_metric:.4f}"
        )
    lines.append("")

    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")

    by_tier = {m.tier: m for m in report.modes}
    for tier in TIER_ORDER:
        m = by_tier.get(tier)
        if m is None:
            continue
        lines.append(f"## `{tier}`")
        lines.append("")
        if m.label is None:
            lines.append(f"- not selected: {m.rationale}")
            lines.append("")
            continue
        lines.append(f"- label: `{m.label}`")
        lines.append(
            f"- dense_top_n: {_fmt_or_dash_int(m.dense_top_n)} | "
            f"final_top_k: {_fmt_or_dash_int(m.final_top_k)}"
        )
        lines.append(
            f"- {m.primary_metric_field}: {_fmt(m.primary_metric_value)}"
        )
        lines.append(
            f"- {m.latency_field}: {_fmt_ms(m.latency_p95_ms)}"
        )
        lines.append(f"- rationale: {m.rationale}")
        if m.notes:
            for note in m.notes:
                lines.append(f"  - note: {note}")
        lines.append("")
        lines.append("**Headline metrics**")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| hit@1 | {_fmt(m.mean_hit_at_1)} |")
        lines.append(f"| hit@3 | {_fmt(m.mean_hit_at_3)} |")
        lines.append(f"| hit@5 | {_fmt(m.mean_hit_at_5)} |")
        lines.append(f"| mrr@10 | {_fmt(m.mean_mrr_at_10)} |")
        lines.append(f"| ndcg@10 | {_fmt(m.mean_ndcg_at_10)} |")
        lines.append(f"| rerank_p95_ms | {_fmt_ms(m.rerank_p95_ms)} |")
        lines.append(f"| rerank_p99_ms | {_fmt_ms(m.rerank_p99_ms)} |")
        lines.append(f"| total_query_p95_ms | {_fmt_ms(m.total_query_p95_ms)} |")
        lines.append(f"| total_query_p99_ms | {_fmt_ms(m.total_query_p99_ms)} |")
        lines.append("")

    return "\n".join(lines) + "\n"
