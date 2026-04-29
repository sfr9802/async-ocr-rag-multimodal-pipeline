"""Tests for ``eval/harness/reranker_format_comparison.py``.

Pin the 5-way verdict logic for the reranker-input-format confirm
sweep. The verdict output drives the next-step recommendation in the
markdown report; future threshold drift would surface here rather
than as silent shifts in the report's verdict line.

All tests are pure-Python, no FAISS / no embedder / no reranker.
"""

from __future__ import annotations

from typing import Dict, Tuple

from eval.harness.reranker_format_comparison import (
    ANCHOR_FORMAT,
    ANCHOR_VARIANT,
    CHUNK_ONLY_FORMAT,
    COMPACT_PREFIX_FORMAT,
    DIVERSITY_DUP_LIFT_THRESHOLD,
    TITLE_PREFIX_FORMAT,
    TITLE_SECTION_PREFIX_FORMAT,
    VERDICT_ADOPT_COMPACT_METADATA_PREFIX,
    VERDICT_ADOPT_TITLE_PREFIX,
    VERDICT_ADOPT_TITLE_SECTION_PREFIX,
    VERDICT_KEEP_CHUNK_ONLY,
    VERDICT_NEED_CHUNKING_DIVERSITY,
    decide_reranker_format_verdict,
)
from eval.harness.variant_comparison import VariantDeltas


def _deltas(
    *,
    variant: str = "title_section",
    fmt: str = TITLE_SECTION_PREFIX_FORMAT,
    grade: str = "promising_quality",
    dh5: float = 0.0,
    dmrr: float = 0.0,
    dcand50: float = 0.0,
    dcand100: float = 0.0,
    ddup10: float = 0.0,
    latency_ratio: float = 1.0,
) -> VariantDeltas:
    return VariantDeltas(
        cell_label="optuna_winner_top8",
        variant=f"{variant}/{fmt}",
        grade=grade,
        reason="test stub",
        delta_hit_at_1=0.0,
        delta_hit_at_3=0.0,
        delta_hit_at_5=dh5,
        delta_mrr_at_10=dmrr,
        delta_ndcg_at_10=0.0,
        delta_candidate_hit_at_10=0.0,
        delta_candidate_hit_at_20=0.0,
        delta_candidate_hit_at_50=dcand50,
        delta_candidate_hit_at_100=dcand100,
        delta_duplicate_ratio_at_5=0.0,
        delta_duplicate_ratio_at_10=ddup10,
        delta_unique_doc_count_at_10=0.0,
        delta_p50ms=0.0,
        delta_p95ms=0.0,
        delta_p99ms=0.0,
        latency_ratio_p95=latency_ratio,
    )


def _anchor_self() -> VariantDeltas:
    """The anchor pair (raw, chunk_only) self-grades to all-zero deltas."""
    return _deltas(
        variant=ANCHOR_VARIANT, fmt=ANCHOR_FORMAT, grade="baseline",
    )


def _build_minimal_matrix(
    *,
    title_dh5: float = 0.0,
    title_dmrr: float = 0.0,
    ts_dh5: float = 0.0,
    ts_dmrr: float = 0.0,
    compact_dh5: float = 0.0,
    compact_dmrr: float = 0.0,
    chunk_only_other_index_dcand50: float = 0.0,
    chunk_only_other_index_dh5: float = 0.0,
    title_section_dup_persist: float = 0.0,
) -> Dict[Tuple[str, str], VariantDeltas]:
    """Build a small (variant, format) → deltas matrix for the verdict.

    Variants: raw + title_section. Formats: chunk_only +
    title/title_section/compact. Returns a dict keyed by
    (variant, format).
    """
    matrix: Dict[Tuple[str, str], VariantDeltas] = {}

    # Anchor.
    matrix[(ANCHOR_VARIANT, ANCHOR_FORMAT)] = _anchor_self()

    # Anchor variant + prefix formats.
    matrix[(ANCHOR_VARIANT, TITLE_PREFIX_FORMAT)] = _deltas(
        variant=ANCHOR_VARIANT, fmt=TITLE_PREFIX_FORMAT,
        dh5=title_dh5, dmrr=title_dmrr,
    )
    matrix[(ANCHOR_VARIANT, TITLE_SECTION_PREFIX_FORMAT)] = _deltas(
        variant=ANCHOR_VARIANT, fmt=TITLE_SECTION_PREFIX_FORMAT,
        dh5=ts_dh5, dmrr=ts_dmrr,
    )
    matrix[(ANCHOR_VARIANT, COMPACT_PREFIX_FORMAT)] = _deltas(
        variant=ANCHOR_VARIANT, fmt=COMPACT_PREFIX_FORMAT,
        dh5=compact_dh5, dmrr=compact_dmrr,
    )

    # title_section variant + chunk_only format = "the dense pool moved
    # but no prefix unlocked it" baseline for the chunking-diversity case.
    matrix[("title_section", CHUNK_ONLY_FORMAT)] = _deltas(
        variant="title_section", fmt=CHUNK_ONLY_FORMAT,
        dh5=chunk_only_other_index_dh5,
        dcand50=chunk_only_other_index_dcand50,
        dcand100=chunk_only_other_index_dcand50,
    )
    # title_section + prefix formats — for the dup-persistence signal.
    matrix[("title_section", TITLE_PREFIX_FORMAT)] = _deltas(
        variant="title_section", fmt=TITLE_PREFIX_FORMAT,
        dh5=0.0, dmrr=0.0,
        ddup10=title_section_dup_persist,
    )
    matrix[("title_section", TITLE_SECTION_PREFIX_FORMAT)] = _deltas(
        variant="title_section", fmt=TITLE_SECTION_PREFIX_FORMAT,
        dh5=0.0, dmrr=0.0,
        ddup10=title_section_dup_persist,
    )
    matrix[("title_section", COMPACT_PREFIX_FORMAT)] = _deltas(
        variant="title_section", fmt=COMPACT_PREFIX_FORMAT,
        dh5=0.0, dmrr=0.0,
        ddup10=title_section_dup_persist,
    )
    return matrix


# ---------------------------------------------------------------------------
# Case A — title_section_plus_chunk wins
# ---------------------------------------------------------------------------


class TestVerdictTitleSectionPrefix:
    def test_title_section_lifts_quality_uncontested(self):
        matrix = _build_minimal_matrix(
            ts_dh5=0.020, ts_dmrr=0.020,
            title_dh5=0.005, title_dmrr=0.005,
            compact_dh5=0.005, compact_dmrr=0.005,
        )
        verdict, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert verdict == VERDICT_ADOPT_TITLE_SECTION_PREFIX
        assert "title_section" in rationale.lower()

    def test_title_section_wins_even_when_others_clear_eps(self):
        matrix = _build_minimal_matrix(
            ts_dh5=0.030, ts_dmrr=0.030,
            title_dh5=0.010, title_dmrr=0.010,
            compact_dh5=0.015, compact_dmrr=0.015,
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        assert v == VERDICT_ADOPT_TITLE_SECTION_PREFIX


# ---------------------------------------------------------------------------
# Case B — title_plus_chunk wins
# ---------------------------------------------------------------------------


class TestVerdictTitlePrefix:
    def test_title_only_lifts_when_title_section_regresses(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.020, title_dmrr=0.020,
            ts_dh5=-0.020, ts_dmrr=-0.020,
            compact_dh5=0.005, compact_dmrr=0.005,
        )
        v, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert v == VERDICT_ADOPT_TITLE_PREFIX
        assert "title_plus_chunk" in rationale.lower() or "title-only" in rationale.lower()

    def test_title_only_lifts_when_title_section_below_eps(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.020, title_dmrr=0.020,
            ts_dh5=0.001, ts_dmrr=0.001,  # below EPS
            compact_dh5=0.005, compact_dmrr=0.005,
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        assert v == VERDICT_ADOPT_TITLE_PREFIX

    def test_title_leads_when_all_three_eligible_but_title_top(self):
        """Spec wording: 가장 좋으면 선택 (pick the best). When every
        prefix format is eligible AND title leads on composite quality,
        title wins regardless of whether title_section also clears EPS.

        Mirrors the actual silver_200 sweep result: at the leader pair
        (title_section index × title_plus_chunk) hit@5=+0.015 /
        MRR=+0.043 → title_qs=+0.058 just above the title_section
        prefix at +0.054.
        """
        matrix = _build_minimal_matrix(
            title_dh5=0.015, title_dmrr=0.043,         # qs=+0.058
            ts_dh5=0.015, ts_dmrr=0.039,               # qs=+0.054
            compact_dh5=0.015, compact_dmrr=0.034,     # qs=+0.049
        )
        v, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert v == VERDICT_ADOPT_TITLE_PREFIX
        assert "title_plus_chunk" in rationale.lower() or "title-only" in rationale.lower()


# ---------------------------------------------------------------------------
# Case C — compact_metadata_plus_chunk wins
# ---------------------------------------------------------------------------


class TestVerdictCompactPrefix:
    def test_compact_wins_when_strictly_above_others(self):
        matrix = _build_minimal_matrix(
            compact_dh5=0.020, compact_dmrr=0.020,
            title_dh5=0.005, title_dmrr=0.005,
            ts_dh5=0.001, ts_dmrr=0.001,
        )
        v, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert v == VERDICT_ADOPT_COMPACT_METADATA_PREFIX
        assert "compact" in rationale.lower()


# ---------------------------------------------------------------------------
# Case D — keep chunk_only
# ---------------------------------------------------------------------------


class TestVerdictKeepChunkOnly:
    def test_no_format_clears_eps(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.001, title_dmrr=0.001,
            ts_dh5=0.001, ts_dmrr=0.001,
            compact_dh5=0.001, compact_dmrr=0.001,
            chunk_only_other_index_dcand50=0.0,
            title_section_dup_persist=0.0,
        )
        v, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert v == VERDICT_KEEP_CHUNK_ONLY
        assert "chunk_only" in rationale.lower()

    def test_all_prefixes_regress(self):
        matrix = _build_minimal_matrix(
            title_dh5=-0.020, title_dmrr=-0.020,
            ts_dh5=-0.020, ts_dmrr=-0.020,
            compact_dh5=-0.020, compact_dmrr=-0.020,
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        # Without dense-pool lift OR dup persistence we land on D.
        assert v == VERDICT_KEEP_CHUNK_ONLY


# ---------------------------------------------------------------------------
# Case E — chunking-diversity bottleneck
# ---------------------------------------------------------------------------


class TestVerdictNeedChunkingDiversity:
    def test_dense_pool_moved_no_format_lift_dup_persistent(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.000, title_dmrr=0.000,
            ts_dh5=0.000, ts_dmrr=0.000,
            compact_dh5=0.000, compact_dmrr=0.000,
            chunk_only_other_index_dcand50=0.045,
            chunk_only_other_index_dh5=-0.045,
            title_section_dup_persist=DIVERSITY_DUP_LIFT_THRESHOLD + 0.01,
        )
        v, rationale = decide_reranker_format_verdict(
            deltas_by_pair=matrix,
        )
        assert v == VERDICT_NEED_CHUNKING_DIVERSITY
        assert "chunking diversity" in rationale.lower()

    def test_dup_persistence_alone_not_enough_without_dense_pool_lift(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.000, title_dmrr=0.000,
            ts_dh5=0.000, ts_dmrr=0.000,
            compact_dh5=0.000, compact_dmrr=0.000,
            chunk_only_other_index_dcand50=0.0,  # no dense-pool lift
            title_section_dup_persist=DIVERSITY_DUP_LIFT_THRESHOLD + 0.01,
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        assert v == VERDICT_KEEP_CHUNK_ONLY

    def test_dense_pool_lift_alone_not_enough_without_dup_persistence(self):
        matrix = _build_minimal_matrix(
            title_dh5=0.000, title_dmrr=0.000,
            ts_dh5=0.000, ts_dmrr=0.000,
            compact_dh5=0.000, compact_dmrr=0.000,
            chunk_only_other_index_dcand50=0.045,
            title_section_dup_persist=0.0,  # dup didn't actually persist
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        assert v == VERDICT_KEEP_CHUNK_ONLY


# ---------------------------------------------------------------------------
# Latency budget — quality lift but high latency degrades to KEEP_CHUNK_ONLY
# ---------------------------------------------------------------------------


class TestLatencyBudget:
    def test_high_latency_excludes_format_from_eligibility(self):
        # title_plus_chunk lifts hit@5 but at 2x p95 latency — should
        # be filtered out, falling through to D.
        matrix: Dict[Tuple[str, str], VariantDeltas] = {}
        matrix[(ANCHOR_VARIANT, ANCHOR_FORMAT)] = _anchor_self()
        matrix[(ANCHOR_VARIANT, TITLE_PREFIX_FORMAT)] = _deltas(
            variant=ANCHOR_VARIANT, fmt=TITLE_PREFIX_FORMAT,
            dh5=0.020, dmrr=0.020, latency_ratio=2.0,
        )
        matrix[(ANCHOR_VARIANT, TITLE_SECTION_PREFIX_FORMAT)] = _deltas(
            variant=ANCHOR_VARIANT, fmt=TITLE_SECTION_PREFIX_FORMAT,
            dh5=0.001, dmrr=0.001,
        )
        matrix[(ANCHOR_VARIANT, COMPACT_PREFIX_FORMAT)] = _deltas(
            variant=ANCHOR_VARIANT, fmt=COMPACT_PREFIX_FORMAT,
            dh5=0.001, dmrr=0.001,
        )
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        # title_plus_chunk filtered out by latency; nothing else clears
        # EPS — KEEP_CHUNK_ONLY.
        assert v == VERDICT_KEEP_CHUNK_ONLY


# ---------------------------------------------------------------------------
# Empty matrix — defensive
# ---------------------------------------------------------------------------


class TestEmptyMatrix:
    def test_only_anchor_returns_keep_chunk_only(self):
        matrix = {(ANCHOR_VARIANT, ANCHOR_FORMAT): _anchor_self()}
        v, _ = decide_reranker_format_verdict(deltas_by_pair=matrix)
        assert v == VERDICT_KEEP_CHUNK_ONLY
