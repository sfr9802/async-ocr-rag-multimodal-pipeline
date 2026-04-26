"""Compare two retrieval-eval baselines side-by-side.

Used in Phase 0 to produce ``retrieval-baseline-comparison.{json,md}``
that puts the deterministic silver and the Opus-generated silver
runs side-by-side. Also computes a third synthetic slice —
``deterministic_without_character_relation`` — because the
deterministic generator's keyword extractor is broken for the
``character_relation`` answer_type and that slice drags the
deterministic aggregate down by ~40 rows.

The comparison reads existing ``retrieval_eval_report.json`` files
(plus the per-row JSON they carry) and never re-runs retrieval. All
numbers are recomputed from row-level fields so the slice with
``character_relation`` excluded is byte-identical in formula to the
"all" slices.

Public surface
--------------

- ``BaselineSlice`` — one row in the compare table
- ``compute_baseline_slice`` — rebuilds aggregates from a row list
- ``run_comparison`` — given the three loaded reports, return the
  comparison dataclass
- ``comparison_to_dict`` / ``render_comparison_markdown``

The output JSON shape mirrors what the existing retrieval_eval_report
emits, so downstream tools (jq, dashboards) can read either
identically. The markdown report is the eyeballing surface.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


METRIC_KEYS: tuple[str, ...] = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_10",
    "dup_rate",
    "unique_doc_coverage",
    "expected_keyword_match_rate",
    "avg_context_token_count",
    "top1_score_margin",
)


@dataclass
class BaselineSlice:
    label: str
    description: str
    dataset_path: Optional[str]
    row_count: int
    rows_with_expected_doc_ids: int
    rows_with_expected_keywords: int
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    per_answer_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_language: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class BaselineComparison:
    slices: List[BaselineSlice]
    metric_keys: List[str] = field(default_factory=lambda: list(METRIC_KEYS))


def compute_baseline_slice(
    *,
    label: str,
    description: str,
    dataset_path: Optional[str],
    rows: Sequence[Mapping[str, Any]],
    exclude_answer_types: Sequence[str] = (),
) -> BaselineSlice:
    """Recompute aggregates from ``rows`` (filtered by exclude list)."""
    excl = {t for t in exclude_answer_types if t}
    kept: List[Mapping[str, Any]] = []
    for r in rows:
        atype = r.get("answer_type")
        if atype and atype in excl:
            continue
        kept.append(r)

    rows_with_doc_ids = sum(
        1 for r in kept if r.get("expected_doc_ids")
    )
    rows_with_keywords = sum(
        1 for r in kept if r.get("expected_section_keywords")
    )

    metrics = {key: _mean_field(kept, key) for key in METRIC_KEYS}

    return BaselineSlice(
        label=label,
        description=description,
        dataset_path=dataset_path,
        row_count=len(kept),
        rows_with_expected_doc_ids=rows_with_doc_ids,
        rows_with_expected_keywords=rows_with_keywords,
        metrics=metrics,
        per_answer_type=_breakdown(kept, key="answer_type"),
        per_difficulty=_breakdown(kept, key="difficulty"),
        per_language=_breakdown(kept, key="language"),
    )


def run_comparison(
    *,
    deterministic_rows: Sequence[Mapping[str, Any]],
    deterministic_dataset_path: Optional[str],
    opus_rows: Sequence[Mapping[str, Any]],
    opus_dataset_path: Optional[str],
    excluded_answer_type: str = "character_relation",
) -> BaselineComparison:
    """Build the three-slice comparison.

    The ``deterministic_without_character_relation`` slice drops every
    row whose ``answer_type`` is the deterministic generator's broken
    bucket; the others use all rows as-is. We don't drop the same
    answer_type from the opus slice because the opus generator emits
    well-formed character_relation queries.
    """
    slices: List[BaselineSlice] = []

    slices.append(
        compute_baseline_slice(
            label="deterministic_all",
            description=(
                "deterministic anime_silver_200 — all 200 rows. "
                "character_relation rows here are degraded by the "
                "generator's keyword extractor, which drags the aggregate "
                "down."
            ),
            dataset_path=deterministic_dataset_path,
            rows=deterministic_rows,
        )
    )
    slices.append(
        compute_baseline_slice(
            label=f"deterministic_without_{excluded_answer_type}",
            description=(
                "deterministic anime_silver_200 with the broken "
                f"answer_type='{excluded_answer_type}' rows removed. "
                "Treat as the deterministic baseline's healthy ceiling, "
                "not the headline number."
            ),
            dataset_path=deterministic_dataset_path,
            rows=deterministic_rows,
            exclude_answer_types=(excluded_answer_type,),
        )
    )
    slices.append(
        compute_baseline_slice(
            label="opus_all",
            description=(
                "opus-generated anime_silver_200_opus — all 200 rows. "
                "Higher-quality queries; the headline pre-improvement "
                "baseline."
            ),
            dataset_path=opus_dataset_path,
            rows=opus_rows,
        )
    )

    return BaselineComparison(slices=slices)


def _mean_field(
    rows: Iterable[Mapping[str, Any]], key: str
) -> Optional[float]:
    values: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> Dict[str, Dict[str, Any]]:
    """Group by ``row[key]`` and aggregate the headline metrics.

    Mirrors ``retrieval_eval._breakdown`` so the per-axis tables here
    line up exactly with the existing eval reports — same metric set
    (hit@5, mrr@10, ndcg@10, count) computed the same way.
    """
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        bucket = r.get(key)
        if not bucket:
            continue
        buckets.setdefault(str(bucket), []).append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for k, brows in sorted(buckets.items()):
        out[k] = {
            "row_count": len(brows),
            "mean_hit_at_5": _mean_field(brows, "hit_at_5"),
            "mean_mrr_at_10": _mean_field(brows, "mrr_at_10"),
            "mean_ndcg_at_10": _mean_field(brows, "ndcg_at_10"),
            "mean_expected_keyword_match_rate": _mean_field(
                brows, "expected_keyword_match_rate"
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Serializers.
# ---------------------------------------------------------------------------


def comparison_to_dict(comparison: BaselineComparison) -> Dict[str, Any]:
    return {
        "metric_keys": list(comparison.metric_keys),
        "slices": [asdict(s) for s in comparison.slices],
    }


def render_comparison_markdown(comparison: BaselineComparison) -> str:
    """Compose retrieval-baseline-comparison.md.

    Three columns per metric (one per slice). Per-axis breakdowns are
    rendered as one section per axis (answer_type, difficulty,
    language), each with one table per slice. Verbose, but reviewers
    asked specifically for the three slices side-by-side.
    """
    lines: List[str] = []
    lines.append("# Retrieval baseline comparison")
    lines.append("")
    lines.append("Three slices over the same retriever (BAAI/bge-m3, FAISS, no reranker).")
    lines.append("")

    # Slice descriptions block.
    for s in comparison.slices:
        lines.append(f"### `{s.label}`")
        lines.append("")
        lines.append(s.description)
        lines.append("")
        lines.append(f"- dataset: `{s.dataset_path or '<unknown>'}`")
        lines.append(
            f"- rows: {s.row_count} "
            f"(with expected_doc_ids: {s.rows_with_expected_doc_ids}, "
            f"with expected_keywords: {s.rows_with_expected_keywords})"
        )
        lines.append("")

    # Headline metrics table — one column per slice.
    lines.append("## Headline metrics")
    lines.append("")
    header = ["metric", *(s.label for s in comparison.slices)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---|" + "---:|" * len(comparison.slices))
    for key in METRIC_KEYS:
        row = [key]
        for s in comparison.slices:
            row.append(_fmt(s.metrics.get(key)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-axis sections (answer_type, difficulty, language).
    for axis_name, accessor in (
        ("answer_type", lambda s: s.per_answer_type),
        ("difficulty", lambda s: s.per_difficulty),
        ("language", lambda s: s.per_language),
    ):
        # Skip the axis if no slice has any data for it.
        if not any(accessor(s) for s in comparison.slices):
            continue
        lines.append(f"## Per {axis_name}")
        lines.append("")
        for s in comparison.slices:
            data = accessor(s)
            if not data:
                continue
            lines.append(f"### {s.label}")
            lines.append("")
            lines.append(
                f"| {axis_name} | n | hit@5 | mrr@10 | ndcg@10 | kw_match |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for bucket, agg in data.items():
                lines.append(
                    f"| {bucket} | {agg['row_count']} | "
                    f"{_fmt(agg.get('mean_hit_at_5'))} | "
                    f"{_fmt(agg.get('mean_mrr_at_10'))} | "
                    f"{_fmt(agg.get('mean_ndcg_at_10'))} | "
                    f"{_fmt(agg.get('mean_expected_keyword_match_rate'))} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
