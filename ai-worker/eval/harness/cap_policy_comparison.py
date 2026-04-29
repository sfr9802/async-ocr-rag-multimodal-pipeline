"""Verdict logic for the rerank-input cap policy confirm sweep.

Sister of ``reranker_format_comparison``. The matrix here is one cap
policy per cell, all sharing the title_section index × title_plus_chunk
reranker input format that won the prior sweep. Anchor: the
``title_cap_rerank_input=1`` policy that came out of the prior verdict.

This module provides:

  - per-policy deltas vs the anchor (``compute_cap_policy_deltas``,
    a thin re-use of ``compute_variant_deltas`` from
    ``variant_comparison``);
  - the spec's six-way verdict
    (``KEEP_TITLE_CAP_RERANK_INPUT_1``,
    ``ADOPT_TITLE_CAP_RERANK_INPUT_2``,
    ``ADOPT_TITLE_CAP_RERANK_INPUT_3``,
    ``ADOPT_DOC_ID_LEVEL_CAP``,
    ``ADOPT_NO_CAP_RERANK_INPUT``,
    ``NEED_SCHEMA_ENRICHMENT``) via
    ``decide_cap_policy_verdict``.

The schema-enrichment verdict fires when no policy alleviates the
``character_relation`` bucket *and* the gold-was-capped-out counter
doesn't drop materially across policies — i.e. cap tuning isn't the
real bottleneck and the dataset itself needs richer entity / section
metadata to disambiguate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from eval.harness.confirm_wide_mmr_helpers import EPS_HIT, EPS_MRR
from eval.harness.variant_comparison import (
    EPS_CANDIDATE,
    LATENCY_RATIO_LIMIT,
    VariantDeltas,
    compute_variant_deltas,
)


# Policy labels — these are the keys the report writer + tests pin.
POLICY_TITLE_CAP_1 = "title_cap_1"
POLICY_TITLE_CAP_2 = "title_cap_2"
POLICY_TITLE_CAP_3 = "title_cap_3"
POLICY_DOC_ID_CAP = "doc_id_cap"
POLICY_NO_CAP = "no_cap"
POLICY_SECTION_PATH_CAP = "section_path_cap"

ANCHOR_POLICY = POLICY_TITLE_CAP_1


# Verdict labels — matched against the spec's A/B/C/D/E/F enumeration.
VERDICT_KEEP_TITLE_CAP_1 = "KEEP_TITLE_CAP_RERANK_INPUT_1"
VERDICT_ADOPT_TITLE_CAP_2 = "ADOPT_TITLE_CAP_RERANK_INPUT_2"
VERDICT_ADOPT_TITLE_CAP_3 = "ADOPT_TITLE_CAP_RERANK_INPUT_3"
VERDICT_ADOPT_DOC_ID_CAP = "ADOPT_DOC_ID_LEVEL_CAP"
VERDICT_ADOPT_NO_CAP = "ADOPT_NO_CAP_RERANK_INPUT"
VERDICT_NEED_SCHEMA_ENRICHMENT = "NEED_SCHEMA_ENRICHMENT"


# Threshold below which the character_relation bucket is "still bad"
# even after cap relaxation — when every policy stays under this on
# hit@5 and the gold-was-capped-out delta is small, the bottleneck is
# dataset schema, not cap policy.
CHARACTER_RELATION_HIT5_FLOOR = 0.45

# Drop in gold_was_capped_out (count) under cap relaxation that we
# treat as "cap policy actually moved the needle". Below this the
# audit data says the cap wasn't the constraint that was costing
# character-relation queries.
GOLD_CAPPED_OUT_DROP_THRESHOLD = 5


def compute_cap_policy_deltas(
    *,
    policy_label: str,
    policy_summary: Any,
    anchor_summary: Any,
) -> VariantDeltas:
    """Re-use ``compute_variant_deltas`` with the policy label as the
    "variant" key. Treating the anchor policy as ``ANCHOR_VARIANT`` of
    ``variant_comparison`` lets the existing deltas dataclass + grade
    contract work without a new dataclass.
    """
    return compute_variant_deltas(
        cell_label="rerank_input_cap_policy",
        variant=policy_label if policy_label != ANCHOR_POLICY else "raw",
        variant_summary=policy_summary,
        raw_summary=anchor_summary,
    )


def _has_quality_lift(deltas: Optional[VariantDeltas]) -> bool:
    """Quality lift ≥ EPS on hit@5 OR MRR@10 with no hit@5 regression."""
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    h5_safe = (h5 is None) or (h5 >= -EPS_HIT)
    has_lift = (
        (h5 is not None and h5 >= EPS_HIT)
        or (mrr is not None and mrr >= EPS_MRR)
    )
    return has_lift and h5_safe


def _has_regression(deltas: Optional[VariantDeltas]) -> bool:
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    cand50 = deltas.delta_candidate_hit_at_50
    return (
        (h5 is not None and h5 <= -EPS_HIT)
        or (mrr is not None and mrr <= -EPS_MRR)
        or (cand50 is not None and cand50 <= -EPS_CANDIDATE)
    )


def _latency_within_budget(deltas: Optional[VariantDeltas]) -> bool:
    if deltas is None:
        return True
    ratio = deltas.latency_ratio_p95
    return ratio is None or ratio <= LATENCY_RATIO_LIMIT


def _composite_quality_score(deltas: Optional[VariantDeltas]) -> float:
    """Δhit@5 + Δmrr@10 — same composite the format-comparison verdict
    uses. Cand@K is excluded because every policy here shares the same
    index, so its delta is a noise floor, not a signal.
    """
    if deltas is None:
        return float("-inf")
    h5 = deltas.delta_hit_at_5 or 0.0
    mrr = deltas.delta_mrr_at_10 or 0.0
    return float(h5) + float(mrr)


def _character_relation_hit5(
    bucket_metrics: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not bucket_metrics:
        return None
    payload = bucket_metrics.get("character_relation")
    if not isinstance(payload, dict):
        return None
    v = payload.get("mean_hit_at_5")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gold_capped_out_count(
    audit_summary: Optional[Dict[str, Any]],
) -> Optional[int]:
    if not audit_summary:
        return None
    v = audit_summary.get("gold_was_capped_out_count")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def decide_cap_policy_verdict(
    *,
    deltas_by_policy: Dict[str, VariantDeltas],
    bucket_metrics_by_policy: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    audit_summary_by_policy: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
) -> Tuple[str, str]:
    """Return ``(verdict, rationale)`` over the 6-way cap-policy matrix.

    Decision tree (first match wins):

      1. If no non-anchor policy is *eligible* (quality lift, no
         regression, latency within budget) AND character_relation
         hit@5 stays below ``CHARACTER_RELATION_HIT5_FLOOR`` for all
         policies AND the largest drop in gold_was_capped_out is
         smaller than ``GOLD_CAPPED_OUT_DROP_THRESHOLD`` → schema
         enrichment is the next axis.
      2. Among eligible policies pick the one with the highest
         composite quality score; emit the matching ``ADOPT_*``
         verdict. Ties go to the policy with the smaller dup@10
         delta (more diversity-preserving wins).
      3. Otherwise keep the anchor policy.

    The function is conservative — ``ADOPT_*`` requires both lift and
    no-regression. ``NEED_SCHEMA_ENRICHMENT`` requires three signals
    pointing the same way (no lift + character_relation low + cap
    audit unmoved); any one of them missing falls back to KEEP.
    """
    candidates = [
        p for p in deltas_by_policy
        if p != ANCHOR_POLICY
    ]

    eligibility: Dict[str, bool] = {}
    for p in candidates:
        d = deltas_by_policy.get(p)
        eligibility[p] = bool(
            _has_quality_lift(d)
            and not _has_regression(d)
            and _latency_within_budget(d)
        )
    eligible = [p for p, ok in eligibility.items() if ok]

    if eligible:
        def _score(p: str) -> Tuple[float, float]:
            d = deltas_by_policy.get(p)
            qs = _composite_quality_score(d)
            # Secondary key: smaller (less positive / more negative) dup@10
            # delta wins — diversity preserving.
            dup = (
                0.0 if d is None or d.delta_duplicate_ratio_at_10 is None
                else float(d.delta_duplicate_ratio_at_10)
            )
            return (qs, -dup)

        leader = max(eligible, key=_score)
        # Tie-break — when the spec's named policies (doc_id_cap >
        # no_cap > section_path_cap) are within EPS_MRR of the leader's
        # composite score, prefer the simpler / spec-aligned policy.
        # The spec defines six verdicts (A–F): doc_id_cap and no_cap
        # have first-class verdict labels, section_path_cap does not,
        # so resolving to the labelled policy avoids "verdict says X
        # but leader says Y" confusion in the report.
        leader_score = _score(leader)[0]
        for preferred in (POLICY_DOC_ID_CAP, POLICY_NO_CAP):
            if preferred in eligible and preferred != leader:
                pscore = _score(preferred)[0]
                if leader_score - pscore <= EPS_MRR:
                    leader = preferred
                    break

        d = deltas_by_policy[leader]
        qs = _composite_quality_score(d)
        rationale_template = (
            "Policy `{leader}` leads on composite quality "
            "({qs:+.4f}; Δhit@5={h5}, Δmrr@10={mrr}) among the eligible "
            "policies, no regression on hit@5/MRR/cand@50, latency "
            "within budget. Adopt over the title_cap=1 anchor."
        )
        rationale = rationale_template.format(
            leader=leader,
            qs=qs,
            h5=_fmt_signed(d.delta_hit_at_5),
            mrr=_fmt_signed(d.delta_mrr_at_10),
        )
        return _verdict_for_policy(leader), rationale

    # No eligible policy. Check schema-enrichment trigger.
    if (
        bucket_metrics_by_policy is not None
        and audit_summary_by_policy is not None
    ):
        char_floor_hit_for_all = True
        for p in deltas_by_policy:
            cr_h5 = _character_relation_hit5(
                bucket_metrics_by_policy.get(p),
            )
            if cr_h5 is None or cr_h5 >= CHARACTER_RELATION_HIT5_FLOOR:
                char_floor_hit_for_all = False
                break

        anchor_capped_out = _gold_capped_out_count(
            audit_summary_by_policy.get(ANCHOR_POLICY),
        )
        max_drop = 0
        if anchor_capped_out is not None:
            for p, audit in audit_summary_by_policy.items():
                c = _gold_capped_out_count(audit)
                if c is None:
                    continue
                drop = anchor_capped_out - c
                if drop > max_drop:
                    max_drop = drop

        if (
            char_floor_hit_for_all
            and max_drop < GOLD_CAPPED_OUT_DROP_THRESHOLD
        ):
            return (
                VERDICT_NEED_SCHEMA_ENRICHMENT,
                (
                    "No cap policy clears EPS on hit@5/MRR over the "
                    "title_cap=1 anchor; character_relation bucket "
                    "hit@5 stays below "
                    f"{CHARACTER_RELATION_HIT5_FLOOR} for every policy "
                    "(including no-cap); gold_was_capped_out drops by "
                    f"at most {max_drop} across policies. Cap tuning "
                    "is not the real bottleneck — dataset schema needs "
                    "richer fields (work_title, entity_name, "
                    "entity_type, section_path, source_doc_id) to "
                    "disambiguate the same-franchise / same-character "
                    "cases the reranker can't resolve from chunk text "
                    "alone."
                ),
            )

    # Fallback — keep the anchor.
    best_non_anchor = max(
        candidates,
        key=lambda p: _composite_quality_score(deltas_by_policy.get(p)),
        default=None,
    )
    rationale = (
        f"No alternative cap policy lifts hit@5/MRR by ≥ {EPS_HIT} "
        "without regressing the cap/latency contract."
    )
    if best_non_anchor is not None:
        d = deltas_by_policy.get(best_non_anchor)
        rationale += (
            f" Best non-anchor candidate `{best_non_anchor}` carries "
            f"Δhit@5={_fmt_signed(getattr(d, 'delta_hit_at_5', None))} "
            f"Δmrr@10={_fmt_signed(getattr(d, 'delta_mrr_at_10', None))}, "
            "below the adoption threshold."
        )
    rationale += " Keep title_cap_rerank_input=1."
    return VERDICT_KEEP_TITLE_CAP_1, rationale


def _verdict_for_policy(policy: str) -> str:
    return {
        POLICY_TITLE_CAP_2: VERDICT_ADOPT_TITLE_CAP_2,
        POLICY_TITLE_CAP_3: VERDICT_ADOPT_TITLE_CAP_3,
        POLICY_DOC_ID_CAP: VERDICT_ADOPT_DOC_ID_CAP,
        POLICY_NO_CAP: VERDICT_ADOPT_NO_CAP,
        POLICY_SECTION_PATH_CAP: VERDICT_ADOPT_DOC_ID_CAP,
    }.get(policy, VERDICT_KEEP_TITLE_CAP_1)


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return str(value)
