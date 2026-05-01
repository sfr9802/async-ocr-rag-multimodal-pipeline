"""Per-row leakage detection on top of ``lexical_overlap``.

The lexical_overlap module emits a *raw* per-row
``overlap_risk`` (high / medium / low / not_applicable) that says
"how close is the query to its target on a string-overlap basis".
That signal alone is **not** the leakage call — a ``direct_title``
query is *supposed* to repeat the title verbatim, so a high overlap
there is by design, not leakage.

This module adds the *type-aware* leakage call:

  - ``leakage_risk = "high"``   the query's overlap_risk is high AND
                                 the query_type is one of the
                                 paraphrase / indirect / section_intent
                                 family (where high overlap means the
                                 query effectively echoed corpus text).
  - ``leakage_risk = "medium"`` overlap_risk is high but the type is
                                 ambiguous; OR overlap_risk is medium
                                 and the type is paraphrase/indirect/
                                 section_intent.
  - ``leakage_risk = "low"``    everything else (including high
                                 overlap on direct_title / alias_variant —
                                 those are by-design).
  - ``leakage_risk = "not_applicable"`` for not_in_corpus rows.

The aggregated reporter ``summarize_leakage`` walks all records and
emits a JSON-friendly block ready to drop into the silver-500 summary,
plus a short markdown table (``render_leakage_md``) for the .md report.

This module never reads raw chunk text — it works only on the
already-computed ``lexical_overlap`` dict and the row's ``query_type``.
That keeps it cheap to call on the full 500-row set in CI.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# Frozen taxonomies — these names match the spec verbatim. Tests grep them.
# ---------------------------------------------------------------------------


QUERY_TYPE_DIRECT_TITLE = "direct_title"
QUERY_TYPE_PARAPHRASE_SEMANTIC = "paraphrase_semantic"
QUERY_TYPE_SECTION_INTENT = "section_intent"
QUERY_TYPE_INDIRECT_ENTITY = "indirect_entity"
QUERY_TYPE_ALIAS_VARIANT = "alias_variant"
QUERY_TYPE_AMBIGUOUS = "ambiguous"
QUERY_TYPE_UNANSWERABLE = "unanswerable_or_not_in_corpus"


ALL_QUERY_TYPES: Tuple[str, ...] = (
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_SECTION_INTENT,
    QUERY_TYPE_INDIRECT_ENTITY,
    QUERY_TYPE_ALIAS_VARIANT,
    QUERY_TYPE_AMBIGUOUS,
    QUERY_TYPE_UNANSWERABLE,
)


# Types where high overlap with the title is **expected** — these are
# the queries that mention the work name on purpose. A high overlap
# here is not leakage.
LEAKAGE_BENIGN_TYPES: frozenset = frozenset({
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_ALIAS_VARIANT,
})


# Types where high overlap is a leakage red flag — the query was
# supposed to *not* echo the title / chunk text.
LEAKAGE_SENSITIVE_TYPES: frozenset = frozenset({
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_INDIRECT_ENTITY,
    QUERY_TYPE_SECTION_INTENT,
})


LEAKAGE_RISK_HIGH = "high"
LEAKAGE_RISK_MEDIUM = "medium"
LEAKAGE_RISK_LOW = "low"
LEAKAGE_RISK_NA = "not_applicable"


# ---------------------------------------------------------------------------
# Per-row classifier
# ---------------------------------------------------------------------------


def classify_leakage_risk(
    *,
    query_type: str,
    overlap_risk: str,
) -> str:
    """Cross-reference ``query_type`` × ``overlap_risk`` → leakage_risk.

    The matrix:

      query_type \\ overlap_risk | not_applicable | low    | medium | high
      ---------------------------|----------------|--------|--------|------
      unanswerable               | not_applicable | n/a    | n/a    | n/a
      direct_title / alias       | not_applicable | low    | low    | low
      paraphrase / indirect /    |                |        |        |
        section_intent           | not_applicable | low    | medium | high
      ambiguous                  | not_applicable | low    | low    | medium

    The ``ambiguous`` row is a special case: ambiguous queries are
    *expected* to be short / under-specified, so high lexical overlap
    can also be a quirk of the title being a single common word
    ("도라에몽") rather than leakage. We downgrade high → medium for
    those rows so the summary still surfaces them, but they don't
    trigger the "must-fix" red-flag bucket.
    """
    if query_type == QUERY_TYPE_UNANSWERABLE or overlap_risk == LEAKAGE_RISK_NA:
        return LEAKAGE_RISK_NA
    if query_type in LEAKAGE_BENIGN_TYPES:
        return LEAKAGE_RISK_LOW
    if query_type in LEAKAGE_SENSITIVE_TYPES:
        if overlap_risk == "high":
            return LEAKAGE_RISK_HIGH
        if overlap_risk == "medium":
            return LEAKAGE_RISK_MEDIUM
        return LEAKAGE_RISK_LOW
    if query_type == QUERY_TYPE_AMBIGUOUS:
        if overlap_risk == "high":
            return LEAKAGE_RISK_MEDIUM
        return LEAKAGE_RISK_LOW
    # Unknown query_type — be conservative.
    return LEAKAGE_RISK_LOW


# ---------------------------------------------------------------------------
# Aggregate reporter
# ---------------------------------------------------------------------------


def summarize_leakage(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-row leakage signals into the summary JSON block.

    Input shape: each record carries
      - ``query_id``, ``query_type``, ``bucket``
      - ``lexical_overlap.overlap_risk``
      - ``leakage_risk``  (set per row by ``annotate_leakage``)

    The block returned lives at ``summary["leakage"]`` and contains:

      - ``per_query_type``    counts of overlap_risk × leakage_risk
      - ``per_bucket``        counts of leakage_risk per bucket
      - ``high_risk_query_ids`` list of qids that flagged HIGH
      - ``benign_high_overlap`` count of rows where overlap_risk=high
                                but type is direct_title / alias_variant
                                (these are *expected* high — reported
                                so the reviewer sees the breakdown).
      - ``ambiguous_high_overlap`` count of ambiguous rows where
                                overlap_risk=high (downgraded to
                                medium leakage_risk per the matrix).
    """
    per_type_overlap: Dict[str, Counter] = defaultdict(Counter)
    per_type_leakage: Dict[str, Counter] = defaultdict(Counter)
    per_bucket_leakage: Dict[str, Counter] = defaultdict(Counter)
    high_qids: List[str] = []
    benign_high = 0
    ambiguous_high = 0

    for r in records:
        qt = str(r.get("query_type") or "")
        bucket = str(r.get("bucket") or "")
        overlap = r.get("lexical_overlap") or {}
        orisk = str(overlap.get("overlap_risk") or "low")
        lrisk = str(r.get("leakage_risk") or "low")

        per_type_overlap[qt][orisk] += 1
        per_type_leakage[qt][lrisk] += 1
        per_bucket_leakage[bucket][lrisk] += 1

        if lrisk == LEAKAGE_RISK_HIGH:
            qid = str(r.get("query_id") or "")
            if qid:
                high_qids.append(qid)

        if orisk == "high" and qt in LEAKAGE_BENIGN_TYPES:
            benign_high += 1
        if orisk == "high" and qt == QUERY_TYPE_AMBIGUOUS:
            ambiguous_high += 1

    return {
        "per_query_type_overlap_risk": {
            k: dict(v) for k, v in per_type_overlap.items()
        },
        "per_query_type_leakage_risk": {
            k: dict(v) for k, v in per_type_leakage.items()
        },
        "per_bucket_leakage_risk": {
            k: dict(v) for k, v in per_bucket_leakage.items()
        },
        "high_risk_query_ids": sorted(high_qids),
        "benign_high_overlap_count": benign_high,
        "ambiguous_high_overlap_count": ambiguous_high,
    }


def render_leakage_md(
    leakage_block: Mapping[str, Any],
) -> str:
    """Render the markdown body that goes under "## Leakage" in summary.md.

    The renderer lays out three small tables (one per matrix axis) and
    a numbered list of the high-risk qids. It deliberately omits any
    field the input doesn't carry so a partial input still produces a
    valid markdown body.
    """
    lines: List[str] = []
    lines.append("## Leakage")
    lines.append("")
    lines.append(
        "Per-row leakage_risk is the cross of `query_type` and "
        "`lexical_overlap.overlap_risk`. **High** flags rows where a "
        "paraphrase / indirect / section_intent query overlaps the "
        "silver target enough to be answered by string match alone — "
        "those are leakage candidates. **Medium** is the same axis at "
        "the medium overlap threshold. `direct_title` and "
        "`alias_variant` rows with high overlap are by design and "
        "show up under `benign_high_overlap`."
    )
    lines.append("")

    # Table 1: per-query-type leakage risk
    lines.append("### Per-query-type leakage_risk distribution")
    lines.append("")
    lines.append("| query_type | low | medium | high | not_applicable |")
    lines.append("|---|---:|---:|---:|---:|")
    per_type = leakage_block.get("per_query_type_leakage_risk") or {}
    for qt in ALL_QUERY_TYPES:
        c = per_type.get(qt) or {}
        lines.append(
            f"| {qt} | "
            f"{int(c.get('low', 0))} | "
            f"{int(c.get('medium', 0))} | "
            f"{int(c.get('high', 0))} | "
            f"{int(c.get('not_applicable', 0))} |"
        )
    lines.append("")

    # Table 2: per-bucket leakage risk
    lines.append("### Per-bucket leakage_risk distribution")
    lines.append("")
    lines.append("| bucket | low | medium | high | not_applicable |")
    lines.append("|---|---:|---:|---:|---:|")
    per_bucket = leakage_block.get("per_bucket_leakage_risk") or {}
    for bucket in sorted(per_bucket.keys()):
        c = per_bucket.get(bucket) or {}
        lines.append(
            f"| {bucket} | "
            f"{int(c.get('low', 0))} | "
            f"{int(c.get('medium', 0))} | "
            f"{int(c.get('high', 0))} | "
            f"{int(c.get('not_applicable', 0))} |"
        )
    lines.append("")

    # Table 3: notes on benign-by-design high-overlap rows
    benign = int(leakage_block.get("benign_high_overlap_count") or 0)
    ambig = int(leakage_block.get("ambiguous_high_overlap_count") or 0)
    lines.append("### High-overlap rows allowed by design")
    lines.append("")
    lines.append(
        f"- `direct_title` / `alias_variant` rows with overlap_risk=high: "
        f"**{benign}** (expected; these queries mention the title on purpose)."
    )
    lines.append(
        f"- `ambiguous` rows with overlap_risk=high: **{ambig}** "
        "(downgraded to medium leakage_risk; the title may itself be a "
        "single common word)."
    )
    lines.append("")

    # High-risk qid list (truncated)
    high_qids: List[str] = list(leakage_block.get("high_risk_query_ids") or [])
    lines.append("### High-leakage_risk query ids")
    lines.append("")
    if not high_qids:
        lines.append("- _(none)_")
    else:
        for qid in high_qids[:50]:
            lines.append(f"- `{qid}`")
        if len(high_qids) > 50:
            lines.append(f"- _… {len(high_qids) - 50} more_")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-record annotator (used by the build pipeline)
# ---------------------------------------------------------------------------


def annotate_leakage(record: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``record["leakage_risk"]`` from the row's overlap_risk + type.

    Mutates and returns the record so the caller can chain in a list
    comprehension. Idempotent — calling twice on the same record gives
    the same value.
    """
    qt = str(record.get("query_type") or "")
    overlap = record.get("lexical_overlap") or {}
    orisk = str(overlap.get("overlap_risk") or "low")
    record["leakage_risk"] = classify_leakage_risk(
        query_type=qt, overlap_risk=orisk,
    )
    return record
