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


# Allowed values for ``BaselineSlice.kind``. The renderer groups slices
# by kind into separate sections so baseline headline numbers never
# share a table row with hyperparameter-tuned variants.
KIND_BASELINE = "baseline"
KIND_DIAGNOSTIC = "diagnostic"
KIND_TUNED = "tuned"
KIND_ORDER: tuple[str, ...] = (KIND_BASELINE, KIND_DIAGNOSTIC, KIND_TUNED)
KIND_HEADINGS: Dict[str, str] = {
    KIND_BASELINE: "Baselines",
    KIND_DIAGNOSTIC: "Diagnostic slices (do not quote as headlines)",
    KIND_TUNED: "Tuned variants (hyperparameter-modified — kept separate from baselines)",
}


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
    # Retriever-config snapshot the slice was scored under. Free-form
    # so we can record max_seq_length, embedding_model, reranker,
    # candidate_k, etc. without forcing each future caller to update a
    # struct. Surfaced both in JSON and in the per-slice description
    # block so a reviewer can spot apples-to-oranges configs at a glance.
    retriever_config: Dict[str, Any] = field(default_factory=dict)
    # Bucket the renderer groups by. ``baseline`` slices share a
    # headline table; ``tuned`` slices are rendered in their own
    # section so hyperparameter-modified runs never get summed,
    # subtracted, or visually compared against baseline numbers in
    # the same row. ``diagnostic`` is for derived ceiling/filter
    # slices like ``deterministic_without_<answer_type>``.
    kind: str = KIND_BASELINE


@dataclass
class BaselineComparison:
    slices: List[BaselineSlice]
    metric_keys: List[str] = field(default_factory=lambda: list(METRIC_KEYS))
    # Free-text warnings that go at the top of the markdown report,
    # before the slice descriptions. Used for "this is not a strict
    # apples-to-apples comparison" callouts, broken-bucket flags, etc.
    caveats: List[str] = field(default_factory=list)


def compute_baseline_slice(
    *,
    label: str,
    description: str,
    dataset_path: Optional[str],
    rows: Sequence[Mapping[str, Any]],
    exclude_answer_types: Sequence[str] = (),
    retriever_config: Optional[Mapping[str, Any]] = None,
    kind: str = KIND_BASELINE,
) -> BaselineSlice:
    """Recompute aggregates from ``rows`` (filtered by exclude list).

    ``kind`` controls which section the renderer puts the slice into;
    see ``KIND_ORDER`` for the canonical bucket names.
    """
    if kind not in KIND_ORDER:
        raise ValueError(
            f"unknown slice kind {kind!r}; expected one of {KIND_ORDER}"
        )
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
        retriever_config=dict(retriever_config or {}),
        kind=kind,
    )


def run_comparison(
    *,
    deterministic_rows: Sequence[Mapping[str, Any]],
    deterministic_dataset_path: Optional[str],
    opus_rows: Sequence[Mapping[str, Any]],
    opus_dataset_path: Optional[str],
    excluded_answer_type: str = "character_relation",
    deterministic_retriever_config: Optional[Mapping[str, Any]] = None,
    opus_retriever_config: Optional[Mapping[str, Any]] = None,
    caveats: Optional[Sequence[str]] = None,
    deterministic_kind: str = KIND_BASELINE,
    opus_kind: str = KIND_BASELINE,
) -> BaselineComparison:
    """Build the three-slice comparison.

    The ``deterministic_without_character_relation`` slice drops every
    row whose ``answer_type`` is the deterministic generator's broken
    bucket; the others use all rows as-is. We don't drop the same
    answer_type from the opus slice because the opus generator emits
    well-formed character_relation queries.

    ``*_retriever_config`` payloads are propagated into each slice's
    ``retriever_config`` field so the rendered report can call out
    apples-to-oranges differences (e.g. ``max_seq_length=8192`` vs
    ``=1024``). ``caveats`` lines are emitted verbatim above the slice
    descriptions in the markdown report.

    ``*_kind`` chooses which section each slice lands in. Pass
    ``KIND_TUNED`` (``"tuned"``) when a slice came from a
    hyperparameter-modified retriever run — that slice (and the
    derived diagnostic, if applicable) is then rendered in a separate
    section so its numbers cannot be mistaken for, or table-joined
    against, the baseline headlines.
    """
    if deterministic_kind not in (KIND_BASELINE, KIND_TUNED):
        raise ValueError(
            f"deterministic_kind must be one of {{baseline, tuned}}; "
            f"got {deterministic_kind!r}"
        )
    if opus_kind not in (KIND_BASELINE, KIND_TUNED):
        raise ValueError(
            f"opus_kind must be one of {{baseline, tuned}}; "
            f"got {opus_kind!r}"
        )

    slices: List[BaselineSlice] = []

    det_cfg = dict(deterministic_retriever_config or {})
    opus_cfg = dict(opus_retriever_config or {})

    # The diagnostic slice is filtered from the deterministic rows, so
    # its retriever-config provenance is the deterministic one. If the
    # deterministic side is tuned, the diagnostic is *also* a tuned
    # measurement and must travel with it — never collapse a tuned
    # diagnostic back into the baseline section.
    diagnostic_kind = (
        KIND_DIAGNOSTIC if deterministic_kind == KIND_BASELINE else KIND_TUNED
    )
    diagnostic_label_prefix = (
        "deterministic" if deterministic_kind == KIND_BASELINE else "tuned"
    )

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
            retriever_config=det_cfg,
            kind=deterministic_kind,
        )
    )
    slices.append(
        compute_baseline_slice(
            label=f"{diagnostic_label_prefix}_without_{excluded_answer_type}",
            description=(
                f"{diagnostic_label_prefix} anime_silver_200 with the "
                f"broken answer_type='{excluded_answer_type}' rows "
                "removed. **Diagnostic ceiling slice — not an official "
                "headline number.** Use it to estimate the retriever's "
                "ceiling on well-formed queries; do not quote it as the "
                "deterministic headline."
            ),
            dataset_path=deterministic_dataset_path,
            rows=deterministic_rows,
            exclude_answer_types=(excluded_answer_type,),
            retriever_config=det_cfg,
            kind=diagnostic_kind,
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
            retriever_config=opus_cfg,
            kind=opus_kind,
        )
    )

    return BaselineComparison(slices=slices, caveats=list(caveats or []))


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


def _group_slices_by_kind(
    slices: Sequence[BaselineSlice],
) -> List[tuple[str, List[BaselineSlice]]]:
    """Group slices into ``(kind, [slices])`` tuples in canonical order.

    Iterates in ``KIND_ORDER`` first so baselines always come before
    diagnostics and tuned variants in the rendered report. Any
    non-canonical kind landed via direct ``BaselineSlice`` construction
    is appended afterward in insertion order so it isn't silently
    dropped.
    """
    by_kind: Dict[str, List[BaselineSlice]] = {}
    for s in slices:
        by_kind.setdefault(s.kind, []).append(s)
    out: List[tuple[str, List[BaselineSlice]]] = []
    seen: set[str] = set()
    for k in KIND_ORDER:
        if by_kind.get(k):
            out.append((k, by_kind[k]))
            seen.add(k)
    for k, group in by_kind.items():
        if k not in seen:
            out.append((k, group))
    return out


def render_comparison_markdown(comparison: BaselineComparison) -> str:
    """Compose retrieval-baseline-comparison.md.

    Slices are grouped by ``kind`` before rendering — baselines,
    diagnostics, and tuned variants each get their own description
    block, headline-metrics table, and per-axis breakdown. The
    grouping is structural, not cosmetic: a tuned slice never shares a
    headline-metrics row with a baseline slice, so reviewers cannot
    accidentally subtract or jq-join hyperparameter-modified numbers
    against baseline numbers.
    """
    lines: List[str] = []
    lines.append("# Retrieval baseline comparison")
    lines.append("")
    lines.append("Slices over the same retriever family (BAAI/bge-m3, FAISS, no reranker).")
    lines.append("")

    # Caveats block — apples-to-apples warning, retriever-config diff,
    # and any other reviewer-facing flags.
    if comparison.caveats:
        lines.append("## Caveats")
        lines.append("")
        for caveat in comparison.caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    grouped = _group_slices_by_kind(comparison.slices)

    # Slice descriptions, grouped by kind.
    for kind, group in grouped:
        heading = KIND_HEADINGS.get(kind, kind.title())
        lines.append(f"## {heading}")
        lines.append("")
        for s in group:
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
            if s.retriever_config:
                cfg_str = ", ".join(
                    f"{k}={v}" for k, v in s.retriever_config.items()
                )
                lines.append(f"- retriever: {cfg_str}")
            lines.append("")

    # Headline metrics — one table per kind. Different kinds never
    # share a row in the same table.
    for kind, group in grouped:
        heading = KIND_HEADINGS.get(kind, kind.title())
        lines.append(f"## Headline metrics — {heading}")
        lines.append("")
        header = ["metric", *(s.label for s in group)]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|" + "---:|" * len(group))
        for key in METRIC_KEYS:
            row = [key]
            for s in group:
                row.append(_fmt(s.metrics.get(key)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Per-axis sections (answer_type, difficulty, language) — also
    # split by kind so a tuned slice's per-bucket numbers don't sit
    # next to a baseline's per-bucket numbers.
    for axis_name, accessor in (
        ("answer_type", lambda s: s.per_answer_type),
        ("difficulty", lambda s: s.per_difficulty),
        ("language", lambda s: s.per_language),
    ):
        # Skip the axis entirely if no slice has any data for it.
        if not any(accessor(s) for s in comparison.slices):
            continue
        for kind, group in grouped:
            if not any(accessor(s) for s in group):
                continue
            heading = KIND_HEADINGS.get(kind, kind.title())
            lines.append(f"## Per {axis_name} — {heading}")
            lines.append("")
            for s in group:
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
