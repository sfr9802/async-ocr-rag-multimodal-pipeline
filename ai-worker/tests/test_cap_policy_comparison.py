"""Tests for the cap-policy comparison verdict logic.

Covers all six verdicts:
  - KEEP_TITLE_CAP_RERANK_INPUT_1 (no eligible alternative).
  - ADOPT_TITLE_CAP_RERANK_INPUT_2 (cap=2 wins).
  - ADOPT_TITLE_CAP_RERANK_INPUT_3 (cap=3 wins).
  - ADOPT_DOC_ID_LEVEL_CAP (doc_id_cap wins).
  - ADOPT_NO_CAP_RERANK_INPUT (no_cap wins).
  - NEED_SCHEMA_ENRICHMENT (no eligible policy + character_relation
    plateau + capped-out unmoved).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from eval.harness.cap_policy_comparison import (
    ANCHOR_POLICY,
    POLICY_DOC_ID_CAP,
    POLICY_NO_CAP,
    POLICY_SECTION_PATH_CAP,
    POLICY_TITLE_CAP_2,
    POLICY_TITLE_CAP_3,
    VERDICT_ADOPT_DOC_ID_CAP,
    VERDICT_ADOPT_NO_CAP,
    VERDICT_ADOPT_TITLE_CAP_2,
    VERDICT_ADOPT_TITLE_CAP_3,
    VERDICT_KEEP_TITLE_CAP_1,
    VERDICT_NEED_SCHEMA_ENRICHMENT,
    compute_cap_policy_deltas,
    decide_cap_policy_verdict,
)
from eval.harness.variant_comparison import VariantDeltas


def _make_summary(
    *,
    hit5: float,
    mrr10: float,
    cand50: float = 0.85,
    p95: float = 540.0,
    dup10: float = 0.0,
    uniq10: float = 8.0,
) -> Dict[str, Any]:
    return {
        "mean_hit_at_1": 0.6,
        "mean_hit_at_3": 0.7,
        "mean_hit_at_5": hit5,
        "mean_mrr_at_10": mrr10,
        "mean_ndcg_at_10": 0.7,
        "candidate_hit_rates": {"10": 0.7, "20": 0.8, "50": cand50, "100": cand50 + 0.02},
        "duplicate_doc_ratios": {"5": 0.0, "10": dup10},
        "unique_doc_counts": {"10": uniq10},
        "p50_retrieval_ms": 530.0,
        "p95_total_retrieval_ms": p95,
        "p95_retrieval_ms": p95,
        "p99_retrieval_ms": p95 * 1.05,
    }


def _deltas(
    summary_by_policy: Dict[str, Dict[str, Any]],
) -> Dict[str, VariantDeltas]:
    anchor = summary_by_policy[ANCHOR_POLICY]
    out: Dict[str, VariantDeltas] = {}
    for policy, summary in summary_by_policy.items():
        out[policy] = compute_cap_policy_deltas(
            policy_label=policy,
            policy_summary=summary,
            anchor_summary=anchor,
        )
    return out


# ---------------------------------------------------------------------------
# Adoption verdicts
# ---------------------------------------------------------------------------


class TestAdoptionVerdicts:
    def test_adopt_title_cap_2(self):
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.76, mrr10=0.70),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_ADOPT_TITLE_CAP_2

    def test_adopt_title_cap_3_when_3_dominates(self):
        # Both cap_2 and cap_3 lift; cap_3 lifts more.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.745, mrr10=0.675),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.77, mrr10=0.71),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_ADOPT_TITLE_CAP_3

    def test_adopt_doc_id_cap_when_only_one_eligible(self):
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.78, mrr10=0.71),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_ADOPT_DOC_ID_CAP

    def test_adopt_no_cap_when_no_cap_dominates(self):
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.78, mrr10=0.72, dup10=0.05),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_ADOPT_NO_CAP


# ---------------------------------------------------------------------------
# Keep / schema enrichment
# ---------------------------------------------------------------------------


class TestNonAdoptVerdicts:
    def test_keep_anchor_when_no_lift(self):
        # All policies match the anchor — nothing eligible to adopt.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_KEEP_TITLE_CAP_1

    def test_keep_anchor_when_alternative_regresses(self):
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.71, mrr10=0.65),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_KEEP_TITLE_CAP_1

    def test_need_schema_enrichment_when_character_relation_plateaus(self):
        # No lift on any policy + character_relation hit@5 stays below
        # the floor (0.45) for every policy + capped_out doesn't drop.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        bucket = {
            p: {"character_relation": {"mean_hit_at_5": 0.40}}
            for p in summaries
        }
        audit = {
            p: {"gold_was_capped_out_count": 12} for p in summaries
        }
        # All policies stuck at 12 capped-outs → max_drop = 0 < 5.
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
            bucket_metrics_by_policy=bucket,
            audit_summary_by_policy=audit,
        )
        assert verdict == VERDICT_NEED_SCHEMA_ENRICHMENT

    def test_no_schema_verdict_when_capped_out_drops(self):
        # No lift but a policy DROPS capped-out by ≥ 5 — bottleneck is
        # cap, but no policy clears EPS yet → KEEP, not SCHEMA.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        bucket = {
            p: {"character_relation": {"mean_hit_at_5": 0.40}}
            for p in summaries
        }
        audit = {
            ANCHOR_POLICY: {"gold_was_capped_out_count": 12},
            POLICY_TITLE_CAP_2: {"gold_was_capped_out_count": 12},
            POLICY_TITLE_CAP_3: {"gold_was_capped_out_count": 12},
            POLICY_NO_CAP: {"gold_was_capped_out_count": 0},  # massive drop
            POLICY_DOC_ID_CAP: {"gold_was_capped_out_count": 12},
            POLICY_SECTION_PATH_CAP: {"gold_was_capped_out_count": 12},
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
            bucket_metrics_by_policy=bucket,
            audit_summary_by_policy=audit,
        )
        # Schema verdict requires capped_out NOT moving — here it moved
        # by 12 with no_cap → fall through to KEEP.
        assert verdict == VERDICT_KEEP_TITLE_CAP_1

    def test_no_schema_verdict_when_character_relation_above_floor(self):
        # No quality lift but character_relation > 0.45 on at least one
        # policy → don't claim schema bottleneck.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67),
        }
        bucket = {
            ANCHOR_POLICY: {"character_relation": {"mean_hit_at_5": 0.40}},
            POLICY_TITLE_CAP_2: {"character_relation": {"mean_hit_at_5": 0.50}},
            POLICY_TITLE_CAP_3: {"character_relation": {"mean_hit_at_5": 0.40}},
            POLICY_NO_CAP: {"character_relation": {"mean_hit_at_5": 0.40}},
            POLICY_DOC_ID_CAP: {"character_relation": {"mean_hit_at_5": 0.40}},
            POLICY_SECTION_PATH_CAP: {"character_relation": {"mean_hit_at_5": 0.40}},
        }
        audit = {p: {"gold_was_capped_out_count": 12} for p in summaries}
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
            bucket_metrics_by_policy=bucket,
            audit_summary_by_policy=audit,
        )
        assert verdict == VERDICT_KEEP_TITLE_CAP_1


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------


class TestTieBreakPreferredPolicy:
    def test_section_path_within_eps_falls_back_to_doc_id(self):
        # section_path_cap leads by < EPS_MRR over doc_id_cap. Tie-
        # break should prefer doc_id_cap (spec-labelled) so the
        # verdict label aligns with the rationale.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.78, mrr10=0.7203),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.78, mrr10=0.7203),
            # section_path edges out by 0.0025 mrr — within EPS_MRR=0.005.
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.78, mrr10=0.7228),
        }
        verdict, rationale = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        assert verdict == VERDICT_ADOPT_DOC_ID_CAP
        assert "doc_id_cap" in rationale

    def test_section_path_dominates_outside_eps(self):
        # section_path leads by > EPS_MRR — keep it as leader.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.78, mrr10=0.71),
        }
        verdict, rationale = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        # section_path_cap → ADOPT_DOC_ID_LEVEL_CAP family, but
        # rationale should reference section_path_cap as the actual leader.
        assert verdict == VERDICT_ADOPT_DOC_ID_CAP
        assert "section_path_cap" in rationale


class TestLatencyBudget:
    def test_lift_with_excess_latency_falls_back(self):
        # Cap=2 lifts but p95 ratio > 1.5x (LATENCY_RATIO_LIMIT) → not eligible.
        summaries = {
            ANCHOR_POLICY: _make_summary(hit5=0.74, mrr10=0.67, p95=400.0),
            POLICY_TITLE_CAP_2: _make_summary(hit5=0.78, mrr10=0.71, p95=620.0),
            POLICY_TITLE_CAP_3: _make_summary(hit5=0.74, mrr10=0.67, p95=400.0),
            POLICY_NO_CAP: _make_summary(hit5=0.74, mrr10=0.67, p95=400.0),
            POLICY_DOC_ID_CAP: _make_summary(hit5=0.74, mrr10=0.67, p95=400.0),
            POLICY_SECTION_PATH_CAP: _make_summary(hit5=0.74, mrr10=0.67, p95=400.0),
        }
        verdict, _ = decide_cap_policy_verdict(
            deltas_by_policy=_deltas(summaries),
        )
        # 620/400 = 1.55 > 1.5 → cap_2 not eligible → fall back to KEEP.
        assert verdict == VERDICT_KEEP_TITLE_CAP_1
