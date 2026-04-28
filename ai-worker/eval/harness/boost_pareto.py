"""Phase 2B boost-aware Pareto frontier.

Composes the existing Phase 2A topN sweep entries (dense + cross-
encoder rerank, no boost) with new Phase 2B entries (dense + boost
with optional rerank) and computes a unified accuracy↔latency
Pareto frontier so a reviewer can see at a glance whether the boost
pipeline pushes the frontier.

Reuses ``compute_pareto_frontier`` and ``ParetoFrontierReport``
from Phase 2A so the JSON shape is consistent — Phase 2B entries
get a ``track`` tag (``"phase2a"`` or ``"phase2b"``) and a
``boost_config_label`` so the markdown renderer can group them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.harness.pareto_frontier import (
    ACCURACY_METRICS,
    LATENCY_METRICS,
    compute_pareto_frontier,
    pareto_to_dict,
)
from eval.harness.topn_sweep import (
    TopNSweepEntry,
    TopNSweepReport,
)


@dataclass
class BoostParetoEntry:
    """A single point on the boost-aware frontier.

    Same shape as ``ParetoPoint`` plus ``track`` and ``boost_config_label``
    so the markdown writer can split rows into Phase 2A baseline vs
    Phase 2B boost groups without consulting the source path.
    """

    label: str
    track: str
    boost_config_label: Optional[str]
    dense_top_n: Optional[int]
    final_top_k: Optional[int]
    metric_field: str
    metric: float
    latency_field: str
    latency_ms: float
    on_frontier: bool
    dominated_by: Optional[str]
    headline: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoostParetoReport:
    schema: str = "phase2b-boost-pareto-frontier.v1"
    metric_field: str = ""
    latency_field: str = ""
    entries: List[BoostParetoEntry] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def _tag_track(
    sweep: TopNSweepReport,
    *,
    track: str,
    boost_config_label: Optional[str],
) -> List[Tuple[TopNSweepEntry, str, Optional[str]]]:
    """Annotate sweep entries with track + boost label for downstream merge."""
    return [(e, track, boost_config_label) for e in sweep.entries]


def _entry_metric(entry: TopNSweepEntry, field_name: str) -> Optional[float]:
    return getattr(entry, field_name, None)


def _headline_payload(entry: TopNSweepEntry) -> Dict[str, Any]:
    """Carry the same headline metrics the Phase 2A frontier surfaces."""
    out: Dict[str, Any] = {}
    keys = (
        "mean_hit_at_1",
        "mean_hit_at_3",
        "mean_hit_at_5",
        "mean_mrr_at_10",
        "mean_ndcg_at_10",
        "rerank_p95_ms",
        "rerank_p99_ms",
        "total_query_p95_ms",
        "total_query_p99_ms",
    )
    for k in keys:
        v = getattr(entry, k, None)
        if v is not None:
            out[k] = v
    return out


def merge_sweeps(
    *,
    phase2a_sweep: TopNSweepReport,
    phase2b_sweep: TopNSweepReport,
    phase2b_label: Optional[str] = None,
) -> List[Tuple[TopNSweepEntry, str, Optional[str]]]:
    """Concat Phase 2A baseline entries + Phase 2B boost entries."""
    out: List[Tuple[TopNSweepEntry, str, Optional[str]]] = []
    out.extend(_tag_track(phase2a_sweep, track="phase2a", boost_config_label=None))
    out.extend(
        _tag_track(
            phase2b_sweep,
            track="phase2b",
            boost_config_label=phase2b_label,
        )
    )
    return out


def compute_boost_pareto_frontier(
    *,
    phase2a_sweep: TopNSweepReport,
    phase2b_sweep: TopNSweepReport,
    metric: str = "mean_hit_at_1",
    latency: str = "rerank_p95_ms",
    phase2b_label: Optional[str] = None,
    caveats: Optional[Sequence[str]] = None,
) -> BoostParetoReport:
    """Build a single frontier across Phase 2A + Phase 2B entries.

    The frontier reuses ``compute_pareto_frontier`` over a synthetic
    sweep (the concatenation of both inputs) so the dominance logic
    is identical between the Phase 2A-only and Phase 2B-aware paths.
    The ``track`` / ``boost_config_label`` tags travel through the
    label namespace so we can recover them after the dominance pass.
    """
    if metric not in ACCURACY_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; choose from "
            + ", ".join(sorted(ACCURACY_METRICS))
        )
    if latency not in LATENCY_METRICS:
        raise ValueError(
            f"unknown latency {latency!r}; choose from "
            + ", ".join(sorted(LATENCY_METRICS))
        )

    tagged = merge_sweeps(
        phase2a_sweep=phase2a_sweep,
        phase2b_sweep=phase2b_sweep,
        phase2b_label=phase2b_label,
    )

    # Build a unified sweep with namespaced labels so dominance can run
    # without hitting label collisions when 2A and 2B both have "top10".
    namespaced_entries: List[TopNSweepEntry] = []
    namespace_to_track: Dict[str, str] = {}
    namespace_to_boost_label: Dict[str, Optional[str]] = {}
    namespace_to_original: Dict[str, TopNSweepEntry] = {}
    for original, track, boost_label in tagged:
        ns_label = f"{track}::{original.label}"
        if ns_label in namespace_to_track:
            # Disambiguate further if multiple boost configs collide.
            ns_label = f"{ns_label}::{boost_label or 'default'}"
        namespace_to_track[ns_label] = track
        namespace_to_boost_label[ns_label] = boost_label
        namespace_to_original[ns_label] = original
        # Build a clone of the entry with the new label.
        new_kwargs = asdict(original)
        new_kwargs["label"] = ns_label
        namespaced_entries.append(TopNSweepEntry(**new_kwargs))

    synthetic_sweep = TopNSweepReport(
        schema="phase2b-boost-pareto-merge.v1",
        entries=namespaced_entries,
        caveats=list(phase2a_sweep.caveats) + list(phase2b_sweep.caveats),
    )

    frontier = compute_pareto_frontier(
        synthetic_sweep,
        metric=metric,
        latency=latency,
        caveats=list(caveats or []),
    )

    out_entries: List[BoostParetoEntry] = []
    for point in frontier.entries:
        ns = point.label
        original = namespace_to_original[ns]
        track = namespace_to_track[ns]
        boost_label = namespace_to_boost_label[ns]
        # Strip namespace prefix for human display; original entry's
        # label is preserved on the report payload via ``label``.
        display_label = ns.split("::", 1)[1] if "::" in ns else ns
        out_entries.append(
            BoostParetoEntry(
                label=display_label,
                track=track,
                boost_config_label=boost_label,
                dense_top_n=original.dense_top_n,
                final_top_k=original.final_top_k,
                metric_field=metric,
                metric=point.metric,
                latency_field=latency,
                latency_ms=point.latency_ms,
                on_frontier=point.on_frontier,
                dominated_by=(
                    point.dominated_by.split("::", 1)[1]
                    if point.dominated_by and "::" in point.dominated_by
                    else point.dominated_by
                ),
                headline=_headline_payload(original),
            )
        )

    return BoostParetoReport(
        metric_field=metric,
        latency_field=latency,
        entries=out_entries,
        caveats=list(caveats or []),
    )


def boost_pareto_to_dict(report: BoostParetoReport) -> Dict[str, Any]:
    return asdict(report)


def render_boost_pareto_markdown(report: BoostParetoReport) -> str:
    """Markdown renderer that groups entries by track for at-a-glance compare."""
    lines: List[str] = []
    lines.append("# Phase 2B boost Pareto frontier")
    lines.append("")
    lines.append(f"- metric:  `{report.metric_field}`")
    lines.append(f"- latency: `{report.latency_field}`")
    lines.append("")
    lines.append("## Frontier (combined)")
    lines.append("")
    lines.append(
        "| track | label | "
        f"{report.metric_field} | {report.latency_field} | on_frontier | "
        "dominated_by |"
    )
    lines.append("|---|---|---:|---:|:---:|---|")
    # Sort by latency ascending so the table reads left-to-right.
    sorted_entries = sorted(report.entries, key=lambda e: e.latency_ms)
    for e in sorted_entries:
        on = "✓" if e.on_frontier else " "
        dom = e.dominated_by or "-"
        lines.append(
            f"| {e.track} | {e.label} | "
            f"{e.metric:.4f} | {e.latency_ms:.2f} | {on} | {dom} |"
        )
    lines.append("")
    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines) + "\n"
