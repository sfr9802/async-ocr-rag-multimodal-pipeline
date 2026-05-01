"""Phase 7.5 — tests for the MMR confirm sweep harness.

Targets ``eval.harness.phase7_mmr_confirm_sweep``. The bar is:

  - grid generation produces every (candidate_k × mmr_lambda) pair
    under the spec defaults (3 × 5 = 15) with deterministic names.
  - selection rule disqualifies on every blocking warning code:
    silver hit@5 regression, silver subpage_named bucket regression,
    gold subpage_named hold violation, gold main_work collapse.
  - section_hit caveat fires as a *warning* but is NOT a disqualifier.
  - plateau analysis labels neighbours-within-epsilon as PLATEAU_OK
    and a single-point peak as PLATEAU_OVERFIT_WARNING.
  - the env / json writers carry the recommended config exactly when
    a winner exists, and rollback wording when no winner exists.
  - the report renderer never emits any of the FORBIDDEN phrases that
    would frame `cand_title_section_top10` as the promotion target —
    this is the regression test the spec calls out explicitly.
  - mmr_select_post_hoc reproduces the production retriever's MMR
    selection on equivalent inputs.

All tests are pure-Python: no FAISS, no embedder.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

from eval.harness.phase7_human_gold_tune import (
    GROUP_STRICT_POSITIVE,
    HUMAN_FOCUS_DISCLAIMER,
    PROMOTION_TARGET_FRAMING,
    GoldQueryEvalRow,
    GoldSeedDataset,
    GoldSummary,
    RetrievedDoc,
    SilverDataset,
    SilverSummary,
    evaluate_gold,
    evaluate_silver,
    summarize_gold,
    summarize_silver,
)
from eval.harness.phase7_mmr_confirm_sweep import (
    DEFAULT_CANDIDATE_K_GRID,
    DEFAULT_MMR_LAMBDA_GRID,
    DEFAULT_TOP_K,
    FORBIDDEN_PROMOTION_TARGET_PHRASES,
    LAMBDA_POLICY_NO_PLATEAU_FALLBACK,
    LAMBDA_POLICY_PLATEAU_NEAREST,
    LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST,
    MAIN_WORK_REGRESSION_THRESHOLD,
    MAIN_WORK_REGRESSION_WARNING,
    PLATEAU_OK,
    PLATEAU_OVERFIT_WARNING,
    PRODUCTION_PLATEAU_EPSILON,
    PRODUCTION_RECOMMENDED_LAMBDA,
    PROMOTION_TARGET_CLARIFICATION,
    SECTION_HIT_CAVEAT,
    SECTION_HIT_HALVING_FACTOR,
    SECTION_RETRIEVAL_WARNING,
    SUBPAGE_NAMED_NOT_FIXED_WARNING,
    CandidateScore,
    ConfirmSweepResult,
    ProductionRecommendation,
    SweepCandidate,
    analyze_plateau,
    append_production_recommendation_to_report,
    apply_variant_to_candidates,
    evaluate_main_work_guardrail,
    evaluate_section_hit_caveat,
    evaluate_subpage_named_hold,
    make_confirm_sweep_grid,
    mmr_select_post_hoc,
    render_confirm_sweep_report,
    render_production_recommendation_section,
    select_confirmed_best,
    select_production_recommended_lambda,
    write_confirm_sweep_report_md,
    write_confirm_sweep_results_jsonl,
    write_confirm_sweep_summary_json,
    write_confirmed_best_config_env,
    write_confirmed_best_config_json,
    write_production_recommended_config_env,
    write_production_recommended_config_json,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic GoldSummary / SilverSummary builders
# ---------------------------------------------------------------------------


def _make_gold_summary(
    *,
    primary: float = 0.7327,
    weighted_hit_at_5: float = 0.7858,
    weighted_mrr_at_10: float = 0.6697,
    weighted_ndcg_at_10: float = 0.7232,
    strict_hit_at_5: float = 0.8333,
    section_hit_at_5: float = 0.045,
    main_work_h5: float = 0.6512,
    subpage_named_h5: float = 0.7107,
    subpage_generic_h5: float = 0.9301,
) -> GoldSummary:
    """Build a GoldSummary skeleton — only the fields the sweep reads."""
    by_bucket: Dict[str, Dict[str, float]] = {
        "main_work": {
            "n_total": 13.0, "n_positive": 11.0,
            "hit_at_5": 0.5, "mrr_at_10": 0.5, "ndcg_at_10": 0.5,
            "weighted_hit_at_5": main_work_h5,
            "weighted_mrr_at_10": main_work_h5 * 0.9,
            "weighted_ndcg_at_10": main_work_h5 * 0.95,
        },
        "subpage_named": {
            "n_total": 17.0, "n_positive": 17.0,
            "hit_at_5": 0.7, "mrr_at_10": 0.6, "ndcg_at_10": 0.7,
            "weighted_hit_at_5": subpage_named_h5,
            "weighted_mrr_at_10": subpage_named_h5 * 0.9,
            "weighted_ndcg_at_10": subpage_named_h5 * 0.95,
        },
        "subpage_generic": {
            "n_total": 17.0, "n_positive": 16.0,
            "hit_at_5": 0.9, "mrr_at_10": 0.7, "ndcg_at_10": 0.8,
            "weighted_hit_at_5": subpage_generic_h5,
            "weighted_mrr_at_10": subpage_generic_h5 * 0.8,
            "weighted_ndcg_at_10": subpage_generic_h5 * 0.85,
        },
    }
    return GoldSummary(
        n_total=50,
        n_strict_positive=30,
        n_soft_positive=14,
        n_ambiguous_probe=3,
        n_abstain_test=3,
        hit_at_1=0.5, hit_at_3=0.6, hit_at_5=0.75, hit_at_10=0.86,
        mrr_at_10=0.6, ndcg_at_10=0.7,
        weighted_hit_at_1=0.55,
        weighted_hit_at_3=0.7,
        weighted_hit_at_5=weighted_hit_at_5,
        weighted_hit_at_10=0.89,
        weighted_mrr_at_10=weighted_mrr_at_10,
        weighted_ndcg_at_10=weighted_ndcg_at_10,
        strict_hit_at_5=strict_hit_at_5,
        strict_mrr_at_10=0.71,
        section_hit_at_5_when_defined=section_hit_at_5,
        section_hit_at_10_when_defined=section_hit_at_5 * 2.0,
        chunk_hit_at_10_when_defined=None,
        primary_score=primary,
        by_bucket=by_bucket,
        by_query_type={},
        by_normalized_group={},
        by_leakage_risk={},
    )


def _make_silver_summary(
    *,
    hit_at_5: float = 0.78,
    subpage_named_h5: float = 0.85,
    main_work_h5: float = 0.65,
) -> SilverSummary:
    by_bucket = {
        "main_work": {
            "n_total": 150.0, "n_scored": 150.0,
            "hit_at_1": 0.4, "hit_at_3": 0.6, "hit_at_5": main_work_h5,
            "hit_at_10": 0.7, "mrr_at_10": 0.5,
        },
        "subpage_named": {
            "n_total": 100.0, "n_scored": 100.0,
            "hit_at_1": 0.6, "hit_at_3": 0.8, "hit_at_5": subpage_named_h5,
            "hit_at_10": 0.93, "mrr_at_10": 0.7,
        },
        "subpage_generic": {
            "n_total": 225.0, "n_scored": 225.0,
            "hit_at_1": 0.6, "hit_at_3": 0.78, "hit_at_5": 0.84,
            "hit_at_10": 0.88, "mrr_at_10": 0.7,
        },
    }
    return SilverSummary(
        n_total=500, n_scored=475,
        hit_at_1=0.55, hit_at_3=0.73, hit_at_5=hit_at_5, hit_at_10=0.83,
        mrr_at_10=0.65,
        by_bucket=by_bucket,
        by_query_type={}, by_leakage_risk={}, by_overlap_risk={},
    )


def _baseline_pair() -> Tuple[GoldSummary, SilverSummary]:
    return _make_gold_summary(), _make_silver_summary()


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def test_make_confirm_sweep_grid_default_size() -> None:
    """The default grid is 3 candidate_k × 5 lambdas = 15 candidates.

    Pinned because the spec explicitly calls out "총 15개 candidate".
    """
    grid = make_confirm_sweep_grid()
    assert len(grid) == 15
    # Order: outer loop candidate_k, inner loop mmr_lambda. The first
    # 5 entries should share candidate_k=20.
    assert grid[0].candidate_k == 20
    assert grid[5].candidate_k == 30
    assert grid[10].candidate_k == 40
    # Every entry uses the production-default index by default.
    for c in grid:
        assert c.cache_dir_relative == (
            "namu-v4-2008-2026-04-retrieval-title-section-mseq512"
        )
        assert c.use_mmr is True
        assert c.top_k == 10


def test_make_confirm_sweep_grid_names_are_deterministic() -> None:
    """The variant name encodes both axes so a reviewer can map it back.

    Pinned shape: ``cand_candk{NN}_mmr_lambda{NNN}``.
    """
    grid = make_confirm_sweep_grid(
        candidate_ks=[20, 30, 40],
        mmr_lambdas=[0.65, 0.7],
    )
    names = {c.name for c in grid}
    assert "cand_candk20_mmr_lambda065" in names
    assert "cand_candk30_mmr_lambda070" in names
    assert "cand_candk40_mmr_lambda065" in names


def test_make_confirm_sweep_grid_axes_pinned_to_spec() -> None:
    """DEFAULT_*_GRID values match the spec: candidate_k {20,30,40},
    mmr_lambda {0.6, 0.65, 0.7, 0.75, 0.8}."""
    assert DEFAULT_CANDIDATE_K_GRID == (20, 30, 40)
    assert DEFAULT_MMR_LAMBDA_GRID == (0.60, 0.65, 0.70, 0.75, 0.80)
    assert DEFAULT_TOP_K == 10


# ---------------------------------------------------------------------------
# Post-hoc MMR
# ---------------------------------------------------------------------------


def _doc(rank: int, page_id: str, score: float) -> RetrievedDoc:
    return RetrievedDoc(
        rank=rank, chunk_id=f"{page_id}-c{rank}", page_id=page_id,
        title=page_id, section_path=("개요",), score=score,
    )


def test_mmr_select_post_hoc_lambda_one_is_relevance_only() -> None:
    """λ=1.0 means MMR == relevance-only ordering (production contract)."""
    pool = [_doc(1, "P-A", 0.9), _doc(2, "P-A", 0.8), _doc(3, "P-B", 0.7)]
    out = mmr_select_post_hoc(pool, top_k=3, mmr_lambda=1.0)
    assert [d.page_id for d in out] == ["P-A", "P-A", "P-B"]


def test_mmr_select_post_hoc_lambda_zero_pure_diversity() -> None:
    """λ=0 means only the diversity penalty matters; same doc_id is
    always penalised so a different doc_id wins after the first pick."""
    pool = [_doc(1, "P-A", 0.9), _doc(2, "P-A", 0.85), _doc(3, "P-B", 0.5)]
    out = mmr_select_post_hoc(pool, top_k=2, mmr_lambda=0.0)
    # First pick is always max-relevance (penalty=0 with empty selected).
    # Second pick: P-A penalty>0 vs P-B penalty=0 → P-B wins.
    assert [d.page_id for d in out] == ["P-A", "P-B"]


def test_mmr_select_post_hoc_renumbers_ranks() -> None:
    """The selector renumbers picked docs to 1..k so downstream metric
    code doesn't get confused by the input pool's ranks."""
    pool = [_doc(5, "P-A", 0.9), _doc(7, "P-B", 0.8)]
    out = mmr_select_post_hoc(pool, top_k=2, mmr_lambda=0.7)
    assert [d.rank for d in out] == [1, 2]


def test_apply_variant_to_candidates_no_mmr_truncates_pool() -> None:
    """use_mmr=False should be a plain top-k slice."""
    pool = [_doc(i, f"P-{i}", 1.0 - i * 0.1) for i in range(1, 11)]
    out = apply_variant_to_candidates(
        pool, candidate_k=8, use_mmr=False, mmr_lambda=0.7, top_k=5,
    )
    assert len(out) == 5
    assert [d.page_id for d in out] == ["P-1", "P-2", "P-3", "P-4", "P-5"]


def test_apply_variant_to_candidates_respects_candidate_k_for_mmr_pool() -> None:
    """When use_mmr=True, MMR runs over candidate_k items, then truncates
    to top_k. Picking candidate_k=2 with a same-doc pool means MMR has
    nothing to diversify against."""
    pool = [_doc(1, "P-A", 0.9), _doc(2, "P-A", 0.85), _doc(3, "P-B", 0.5)]
    out = apply_variant_to_candidates(
        pool, candidate_k=2, use_mmr=True, mmr_lambda=0.0, top_k=2,
    )
    # Both P-A docs are in the candidate window; MMR with λ=0 picks
    # P-A then... still has only P-A available. So output is two P-A.
    assert [d.page_id for d in out] == ["P-A", "P-A"]


# ---------------------------------------------------------------------------
# Guardrail evaluators
# ---------------------------------------------------------------------------


def test_evaluate_main_work_guardrail_fires_on_5pp_drop() -> None:
    base = _make_gold_summary(main_work_h5=0.70)
    cand = _make_gold_summary(main_work_h5=0.64)  # -6pp
    warn = evaluate_main_work_guardrail(baseline=base, candidate=cand)
    assert warn is not None
    assert warn.code == MAIN_WORK_REGRESSION_WARNING
    assert warn.bucket == "main_work"
    assert warn.threshold == pytest.approx(MAIN_WORK_REGRESSION_THRESHOLD)


def test_evaluate_main_work_guardrail_silent_below_threshold() -> None:
    base = _make_gold_summary(main_work_h5=0.70)
    cand = _make_gold_summary(main_work_h5=0.66)  # -4pp; under 5pp threshold
    assert evaluate_main_work_guardrail(baseline=base, candidate=cand) is None


def test_evaluate_subpage_named_hold_fires_on_any_regression() -> None:
    """The subpage_named bucket must be HELD; even a small dip warns."""
    base = _make_gold_summary(subpage_named_h5=0.71)
    cand = _make_gold_summary(subpage_named_h5=0.70)
    warn = evaluate_subpage_named_hold(baseline=base, candidate=cand)
    assert warn is not None
    assert warn.code == SUBPAGE_NAMED_NOT_FIXED_WARNING


def test_evaluate_subpage_named_hold_silent_when_improved() -> None:
    base = _make_gold_summary(subpage_named_h5=0.71)
    cand = _make_gold_summary(subpage_named_h5=0.94)
    assert evaluate_subpage_named_hold(baseline=base, candidate=cand) is None


def test_evaluate_section_hit_caveat_fires_when_halved() -> None:
    base = _make_gold_summary(section_hit_at_5=0.045)
    cand = _make_gold_summary(section_hit_at_5=0.022)  # below 50% threshold
    warn = evaluate_section_hit_caveat(baseline=base, candidate=cand)
    assert warn is not None
    assert warn.code == SECTION_RETRIEVAL_WARNING
    assert warn.threshold == pytest.approx(SECTION_HIT_HALVING_FACTOR)


def test_evaluate_section_hit_caveat_silent_when_within_factor() -> None:
    base = _make_gold_summary(section_hit_at_5=0.045)
    cand = _make_gold_summary(section_hit_at_5=0.030)  # well above half
    assert evaluate_section_hit_caveat(baseline=base, candidate=cand) is None


def test_evaluate_section_hit_caveat_silent_when_baseline_zero() -> None:
    """When baseline section_hit is 0 the metric is undefined; no caveat."""
    base = _make_gold_summary(section_hit_at_5=0.0)
    cand = _make_gold_summary(section_hit_at_5=0.0)
    assert evaluate_section_hit_caveat(baseline=base, candidate=cand) is None


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _make_grid_two_candidates() -> List[SweepCandidate]:
    return [
        SweepCandidate(
            name="cand_candk30_mmr_lambda065",
            candidate_k=30, mmr_lambda=0.65, top_k=10, use_mmr=True,
            cache_dir_relative="x", rag_chunks_path_relative="y",
        ),
        SweepCandidate(
            name="cand_candk30_mmr_lambda070",
            candidate_k=30, mmr_lambda=0.70, top_k=10, use_mmr=True,
            cache_dir_relative="x", rag_chunks_path_relative="y",
        ),
    ]


def test_select_confirmed_best_picks_highest_primary_when_clean() -> None:
    """Two clean wins; the higher primary_score is the confirmed best."""
    base_g, base_s = _baseline_pair()
    grid = _make_grid_two_candidates()
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=0.78, subpage_named_h5=0.93),
            _make_silver_summary(hit_at_5=0.82),
        ),
        grid[1].name: (
            _make_gold_summary(primary=0.79, subpage_named_h5=0.94),
            _make_silver_summary(hit_at_5=0.83),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is not None
    assert result.confirmed_best.name == grid[1].name
    assert result.promotion_recommended is True


def test_select_confirmed_best_rejects_silver_regression() -> None:
    """Silver hit@5 dropping > 3pp is a hard disqualifier even if gold gains."""
    base_g, base_s = _baseline_pair()
    grid = _make_grid_two_candidates()
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=0.79, subpage_named_h5=0.95),
            _make_silver_summary(hit_at_5=0.74),  # -4pp from 0.78
        ),
        grid[1].name: (
            _make_gold_summary(primary=0.79, subpage_named_h5=0.95),
            _make_silver_summary(hit_at_5=0.74),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None
    assert result.promotion_recommended is False


def test_select_confirmed_best_rejects_subpage_named_regression() -> None:
    """gold subpage_named must be HELD; regression is a disqualifier."""
    base_g, base_s = _baseline_pair()
    grid = _make_grid_two_candidates()
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=0.79, subpage_named_h5=0.65),  # -6pp
            _make_silver_summary(hit_at_5=0.83),
        ),
        grid[1].name: (
            _make_gold_summary(primary=0.79, subpage_named_h5=0.65),
            _make_silver_summary(hit_at_5=0.83),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None
    # The rejection reasons should mention the subpage_named warning.
    reasons = " ".join(r.rejection_reason for r in result.candidates)
    assert "SUBPAGE_NAMED_NOT_FIXED_WARNING" in reasons


def test_select_confirmed_best_rejects_main_work_collapse() -> None:
    """gold main_work dropping > 5pp is a disqualifier."""
    base_g, base_s = _baseline_pair()
    grid = _make_grid_two_candidates()
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=0.79, main_work_h5=0.55),  # -10pp from 0.65
            _make_silver_summary(hit_at_5=0.83),
        ),
        grid[1].name: (
            _make_gold_summary(primary=0.79, main_work_h5=0.55),
            _make_silver_summary(hit_at_5=0.83),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None
    reasons = " ".join(r.rejection_reason for r in result.candidates)
    assert MAIN_WORK_REGRESSION_WARNING in reasons


def test_select_confirmed_best_keeps_section_caveat_as_warning_only() -> None:
    """Section_hit halving fires a caveat — but does NOT disqualify."""
    base_g = _make_gold_summary(section_hit_at_5=0.045)
    base_s = _make_silver_summary()
    grid = _make_grid_two_candidates()
    # Both candidates have section_hit halved — caveat fires, but
    # primary still ahead and other guardrails clean → still a winner.
    cand_results = {
        grid[0].name: (
            _make_gold_summary(
                primary=0.78, subpage_named_h5=0.93, section_hit_at_5=0.022,
            ),
            _make_silver_summary(hit_at_5=0.82),
        ),
        grid[1].name: (
            _make_gold_summary(
                primary=0.79, subpage_named_h5=0.94, section_hit_at_5=0.022,
            ),
            _make_silver_summary(hit_at_5=0.83),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is not None
    assert result.confirmed_best.name == grid[1].name
    # Section caveat appears as a warning on at least one candidate.
    has_section_warn = any(
        any(w.code == SECTION_RETRIEVAL_WARNING for w in c.warnings)
        for c in result.candidates
    )
    assert has_section_warn is True


def test_select_confirmed_best_below_epsilon_keeps_baseline() -> None:
    """A candidate with a primary_score delta below the epsilon is rejected."""
    base_g, base_s = _baseline_pair()
    grid = _make_grid_two_candidates()
    cand_results = {
        grid[0].name: (
            _make_gold_summary(
                primary=base_g.primary_score + 0.0001,  # below 0.0005 default
                subpage_named_h5=0.95,
            ),
            _make_silver_summary(hit_at_5=0.83),
        ),
        grid[1].name: (
            _make_gold_summary(
                primary=base_g.primary_score + 0.0001,
                subpage_named_h5=0.95,
            ),
            _make_silver_summary(hit_at_5=0.83),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None


# ---------------------------------------------------------------------------
# Plateau analysis
# ---------------------------------------------------------------------------


def test_analyze_plateau_ok_when_neighbours_within_epsilon() -> None:
    grid = make_confirm_sweep_grid()
    best = next(c for c in grid if c.candidate_k == 30 and c.mmr_lambda == 0.70)
    sweep = {
        c.name: _make_gold_summary(primary=0.78)
        for c in grid if c.candidate_k == 30
    }
    p = analyze_plateau(
        best_candidate=best, best_primary_score=0.78,
        sweep_results_by_name=sweep, grid=grid,
    )
    assert p.status == PLATEAU_OK
    # Both neighbours (0.65, 0.75) should have been compared.
    neighbour_lambdas = {lam for lam, _ in p.neighbours}
    assert neighbour_lambdas == {0.65, 0.75}


def test_analyze_plateau_warns_on_single_point_peak() -> None:
    grid = make_confirm_sweep_grid()
    best = next(c for c in grid if c.candidate_k == 30 and c.mmr_lambda == 0.70)
    # λ=0.7 wins big but neighbours collapse → overfit warning.
    sweep: Dict[str, GoldSummary] = {}
    for c in grid:
        if c.candidate_k != 30:
            continue
        if c.mmr_lambda == 0.70:
            sweep[c.name] = _make_gold_summary(primary=0.79)
        else:
            sweep[c.name] = _make_gold_summary(primary=0.65)
    p = analyze_plateau(
        best_candidate=best, best_primary_score=0.79,
        sweep_results_by_name=sweep, grid=grid,
    )
    assert p.status == PLATEAU_OVERFIT_WARNING


# ---------------------------------------------------------------------------
# Env / JSON writers
# ---------------------------------------------------------------------------


def _ok_result_with_winner() -> ConfirmSweepResult:
    grid = make_confirm_sweep_grid(
        candidate_ks=[30],
        mmr_lambdas=[0.65, 0.70, 0.75],
    )
    base_g, base_s = _baseline_pair()
    cand_results = {
        c.name: (
            _make_gold_summary(
                primary=0.78 if c.mmr_lambda == 0.70 else 0.77,
                subpage_named_h5=0.94,
            ),
            _make_silver_summary(hit_at_5=0.83),
        )
        for c in grid
    }
    return select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )


def test_write_confirmed_best_config_env_emits_promote_lines(tmp_path: Path) -> None:
    result = _ok_result_with_winner()
    p = write_confirmed_best_config_env(
        tmp_path / "best.env", result=result,
    )
    text = p.read_text(encoding="utf-8")
    assert "AIPIPELINE_WORKER_RAG_TOP_K=10" in text
    assert "AIPIPELINE_WORKER_RAG_USE_MMR=true" in text
    assert "AIPIPELINE_WORKER_RAG_CANDIDATE_K=30" in text
    assert "AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000" in text
    # Promotion target clarification carried as a comment.
    assert "Promotion target" in text


def test_write_confirmed_best_config_env_emits_rollback_when_no_winner(
    tmp_path: Path,
) -> None:
    """When no candidate wins, the env file lists a no-op rollback."""
    base_g, base_s = _baseline_pair()
    grid = make_confirm_sweep_grid(
        candidate_ks=[30], mmr_lambdas=[0.7],
    )
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=base_g.primary_score - 0.1),
            _make_silver_summary(hit_at_5=0.5),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    p = write_confirmed_best_config_env(
        tmp_path / "best.env", result=result,
    )
    text = p.read_text(encoding="utf-8")
    assert "promotion_recommended=false" in text
    assert "AIPIPELINE_WORKER_RAG_USE_MMR=false" in text
    assert "Keep baseline" in text


def test_write_confirmed_best_config_json_carries_winner_dict(tmp_path: Path) -> None:
    result = _ok_result_with_winner()
    p = write_confirmed_best_config_json(
        tmp_path / "best.json", result=result,
    )
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["promotion_recommended"] is True
    assert payload["confirmed_best"]["candidate_k"] == 30
    assert payload["confirmed_best"]["mmr_lambda"] == pytest.approx(0.70)
    assert payload["config"]["use_mmr"] is True


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------


def test_render_confirm_sweep_report_includes_clarifications() -> None:
    """The report must include both the human-focus disclaimer AND the
    promotion-target clarification AND the section_hit caveat verbatim.

    These are pinned because the spec is explicit that the report
    cannot lie about what's being tested or what the metric base is.
    """
    result = _ok_result_with_winner()
    base_g, base_s = _baseline_pair()
    md = render_confirm_sweep_report(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )
    assert HUMAN_FOCUS_DISCLAIMER in md
    assert PROMOTION_TARGET_CLARIFICATION in md
    assert SECTION_HIT_CAVEAT in md
    # The headline columns are present.
    assert "primary_score" in md
    assert "subpage_named" in md
    assert "main_work" in md


def test_render_confirm_sweep_report_no_forbidden_promotion_phrases() -> None:
    """Regression check on the wording: NEVER frame `cand_title_section_top10`
    as the promotion target. The selection rule must keep that
    candidate out of the winner slot, but if a future contributor
    accidentally pipes it through as the previous_best the renderer
    should still not produce text that promotes it."""
    result = _ok_result_with_winner()
    base_g, base_s = _baseline_pair()
    md = render_confirm_sweep_report(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        assert phrase not in md, (
            f"forbidden promotion-target phrase leaked into the report: "
            f"{phrase!r}"
        )


def test_write_confirm_sweep_report_md_blocks_forbidden_phrases(
    tmp_path: Path,
) -> None:
    """The writer raises if the rendered MD ever contains a forbidden
    phrase. Belt-and-suspenders against future contributors who edit
    the renderer carelessly."""
    result = _ok_result_with_winner()
    base_g, base_s = _baseline_pair()
    p = write_confirm_sweep_report_md(
        tmp_path / "r.md", result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )
    assert p.exists()
    md = p.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        assert phrase not in md


def test_render_confirm_sweep_report_keeps_baseline_when_no_winner(
    tmp_path: Path,
) -> None:
    """When no candidate wins, the report says 'keep baseline'."""
    base_g, base_s = _baseline_pair()
    grid = make_confirm_sweep_grid(
        candidate_ks=[30], mmr_lambdas=[0.7],
    )
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=base_g.primary_score - 0.1),
            _make_silver_summary(hit_at_5=0.5),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    md = render_confirm_sweep_report(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert "Keep baseline" in md
    assert "promotion_recommended: **NO**" in md.lower() or \
        "promotion recommended: **NO**" in md


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def test_write_confirm_sweep_results_jsonl_includes_baseline_and_grid(
    tmp_path: Path,
) -> None:
    result = _ok_result_with_winner()
    base_g, base_s = _baseline_pair()
    cand_results = {
        c.name: (_make_gold_summary(primary=0.78), _make_silver_summary())
        for c in result.grid
    }
    p = write_confirm_sweep_results_jsonl(
        tmp_path / "results.jsonl",
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    rows = [
        json.loads(line) for line in
        p.read_text(encoding="utf-8").splitlines() if line
    ]
    # Header row = baseline. Then one per grid entry.
    assert rows[0]["role"] == "baseline"
    grid_names = {c.name for c in result.grid}
    seen_candidate_names = {r["variant"] for r in rows[1:]}
    assert grid_names <= seen_candidate_names


# ---------------------------------------------------------------------------
# End-to-end integration with the base harness
# ---------------------------------------------------------------------------


def test_full_select_and_render_with_synthetic_pool(tmp_path: Path) -> None:
    """Drive the harness end-to-end with a synthetic candidate pool.

    Builds a 3-query gold + 3-query silver, applies the selection rule,
    renders the MD, and checks every output file's existence + key
    content.
    """
    # Build a 3-candidate grid (1 candidate_k × 3 lambdas) for speed.
    grid = make_confirm_sweep_grid(
        candidate_ks=[30],
        mmr_lambdas=[0.65, 0.70, 0.75],
    )
    base_g, base_s = _baseline_pair()
    # Two candidates with identical scores; tiebreak should pick the
    # first by name. This pins the deterministic ordering so a future
    # contributor can't accidentally make the picker non-deterministic.
    same = (
        _make_gold_summary(primary=0.78, subpage_named_h5=0.93),
        _make_silver_summary(hit_at_5=0.83),
    )
    cand_results = {
        grid[0].name: same,
        grid[1].name: same,
        grid[2].name: same,
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is not None
    # Tiebreak: name asc → cand_candk30_mmr_lambda065 first.
    assert result.confirmed_best.name == grid[0].name

    write_confirm_sweep_summary_json(
        tmp_path / "summary.json", result=result,
    )
    write_confirm_sweep_results_jsonl(
        tmp_path / "results.jsonl",
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    write_confirm_sweep_report_md(
        tmp_path / "report.md",
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )
    write_confirmed_best_config_json(
        tmp_path / "best.json", result=result,
    )
    write_confirmed_best_config_env(
        tmp_path / "best.env", result=result,
    )
    for fname in [
        "summary.json", "results.jsonl", "report.md",
        "best.json", "best.env",
    ]:
        assert (tmp_path / fname).exists()
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        assert phrase not in md


# ---------------------------------------------------------------------------
# Production recommendation — plateau-aware λ policy
# ---------------------------------------------------------------------------


def _flat_plateau_result_at_candidate_k_40() -> ConfirmSweepResult:
    """A confirm sweep result that mimics the live Phase 7.5 outcome:
    every candidate at candidate_k=40 plateaus at the same primary
    score, so the metric-best lexicographic tie-break picks λ=0.60.
    """
    grid = make_confirm_sweep_grid(
        candidate_ks=[20, 30, 40],
        mmr_lambdas=[0.60, 0.65, 0.70, 0.75, 0.80],
    )
    base_g, base_s = _baseline_pair()
    cand_results: Dict[str, Tuple[GoldSummary, SilverSummary]] = {}
    for c in grid:
        if c.candidate_k == 40:
            primary = 0.813
            silver = 0.836
            sub_named = 0.937
        elif c.candidate_k == 30:
            primary = 0.795
            silver = 0.834
            sub_named = 0.937
        else:
            primary = 0.792
            silver = 0.832
            sub_named = 0.937
        cand_results[c.name] = (
            _make_gold_summary(
                primary=primary,
                weighted_hit_at_5=0.92,
                weighted_mrr_at_10=0.70,
                weighted_ndcg_at_10=0.76,
                section_hit_at_5=0.022,
                main_work_h5=0.6977,
                subpage_named_h5=sub_named,
                subpage_generic_h5=1.0,
            ),
            _make_silver_summary(hit_at_5=silver),
        )
    return select_confirmed_best(
        grid=grid, baseline_name="baseline_retrieval_title_section_top10",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )


def test_production_recommendation_picks_previous_best_on_plateau() -> None:
    """When the metric-best λ is on a plateau and λ=0.70 is part of the
    plateau, the production recommendation is λ=0.70 (the previous
    Phase 7.x best), NOT the lexicographic tie-break value λ=0.60."""
    result = _flat_plateau_result_at_candidate_k_40()
    assert result.confirmed_best is not None
    assert result.confirmed_best.mmr_lambda == pytest.approx(0.60)
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    assert rec.recommended_lambda == pytest.approx(0.70)
    assert rec.candidate_k == 40
    assert rec.use_mmr is True
    assert rec.selected_lambda_policy == LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST
    assert "consistent" in rec.selected_lambda_reason or \
        "matches the prior best" in rec.selected_lambda_reason
    # Plateau should have all five λ values.
    plateau_lams = [lam for lam, _ in rec.plateau_lambdas]
    assert len(plateau_lams) == 5
    # Recommended variant name should resolve.
    assert rec.recommended_variant_name == "cand_candk40_mmr_lambda070"


def test_production_recommendation_falls_back_when_single_point_peak() -> None:
    """When the metric-best λ is a single-point peak (no λ-row
    neighbour is within epsilon), the recommendation falls back to
    the metric-best λ — the policy field carries the fallback marker
    so a reviewer can tell the picker had no other plateau choice."""
    grid = make_confirm_sweep_grid(
        candidate_ks=[30],
        mmr_lambdas=[0.60, 0.65, 0.70, 0.75, 0.80],
    )
    base_g, base_s = _baseline_pair()
    # Single sharp peak at λ=0.65; every other λ is far below.
    cand_results = {}
    for c in grid:
        if c.mmr_lambda == 0.65:
            primary = 0.85  # well above neighbours
        else:
            primary = 0.74  # passes guardrails but well below the peak
        cand_results[c.name] = (
            _make_gold_summary(
                primary=primary, subpage_named_h5=0.94,
            ),
            _make_silver_summary(hit_at_5=0.82),
        )
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is not None
    assert result.confirmed_best.mmr_lambda == pytest.approx(0.65)
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    assert rec.recommended_lambda == pytest.approx(0.65)
    assert rec.selected_lambda_policy == LAMBDA_POLICY_NO_PLATEAU_FALLBACK


def test_production_recommendation_picks_nearest_when_prefer_off_plateau() -> None:
    """A multi-point plateau that does NOT include prefer_lambda → pick
    the plateau λ closest to prefer_lambda."""
    grid = make_confirm_sweep_grid(
        candidate_ks=[30],
        mmr_lambdas=[0.60, 0.65, 0.70, 0.75, 0.80],
    )
    base_g, base_s = _baseline_pair()
    cand_results = {}
    for c in grid:
        if c.mmr_lambda in (0.60, 0.65):
            primary = 0.83  # multi-point plateau at λ=0.60, 0.65
        else:
            primary = 0.79  # below plateau but still passes guardrails
        cand_results[c.name] = (
            _make_gold_summary(primary=primary, subpage_named_h5=0.94),
            _make_silver_summary(hit_at_5=0.82),
        )
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is not None
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        prefer_lambda=0.70,
    )
    assert rec is not None
    # 0.70 not on plateau → closest plateau value (0.65) wins.
    assert rec.recommended_lambda == pytest.approx(0.65)
    assert rec.selected_lambda_policy == LAMBDA_POLICY_PLATEAU_NEAREST


def test_production_recommendation_returns_none_when_no_winner() -> None:
    """No metric-best winner → no production recommendation possible."""
    base_g, base_s = _baseline_pair()
    grid = make_confirm_sweep_grid(candidate_ks=[30], mmr_lambdas=[0.7])
    cand_results = {
        grid[0].name: (
            _make_gold_summary(primary=base_g.primary_score - 0.1),
            _make_silver_summary(hit_at_5=0.5),
        ),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is None


def test_write_production_recommended_config_env_emits_recommended_lambda(
    tmp_path: Path,
) -> None:
    """The env writer must carry the recommended λ (0.70 by default),
    NOT the metric-best lexicographic tie-break value (0.60)."""
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    p = write_production_recommended_config_env(
        tmp_path / "best.production_recommended.env",
        recommendation=rec,
    )
    text = p.read_text(encoding="utf-8")
    assert "AIPIPELINE_WORKER_RAG_TOP_K=10" in text
    assert "AIPIPELINE_WORKER_RAG_CANDIDATE_K=40" in text
    assert "AIPIPELINE_WORKER_RAG_USE_MMR=true" in text
    assert "AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000" in text
    # Must NOT carry the metric-best λ as the live env value.
    assert "AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.6000" not in text
    # Should carry the policy / reason as comments.
    assert LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST in text
    assert "metric_best_lambda=0.6000" in text


def test_write_production_recommended_config_json_carries_full_record(
    tmp_path: Path,
) -> None:
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    p = write_production_recommended_config_json(
        tmp_path / "best.production_recommended.json",
        recommendation=rec,
    )
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["recommended_lambda"] == pytest.approx(0.70)
    assert payload["mmr_lambda"] == pytest.approx(0.70)
    assert payload["candidate_k"] == 40
    assert payload["use_mmr"] is True
    assert payload["confirmed_best_lambda"] == pytest.approx(0.60)
    assert (
        payload["selected_lambda_policy"]
        == LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST
    )
    assert "baseline_metrics" in payload
    assert "confirmed_metrics" in payload
    # Required reviewer-facing fields.
    for key in (
        "rollback_env", "plateau_lambdas", "section_hit_caveat",
        "promotion_target_clarification",
    ):
        assert key in payload


def test_render_production_recommendation_section_carries_pinned_phrases() -> None:
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    md = render_production_recommendation_section(
        recommendation=rec,
        section_caveat_section_hit_at_5=(0.0455, 0.0227),
    )
    assert "## Production recommendation" in md
    assert "production recommended lambda: **0.7000**" in md
    assert "λ-plateau" in md
    assert "0.0455" in md  # section_hit baseline carried in caveat
    assert "0.0227" in md
    assert "AIPIPELINE_WORKER_RAG_CANDIDATE_K=40" in md
    assert "AIPIPELINE_WORKER_RAG_USE_MMR=true" in md
    # Must NOT promote the lexicographic tie-break λ.
    assert "production recommended lambda: **0.6000**" not in md


def test_append_production_recommendation_to_report_idempotent(
    tmp_path: Path,
) -> None:
    """Re-running the appender replaces the existing section in place
    rather than appending a second copy."""
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None

    report_path = tmp_path / "confirm_sweep_report.md"
    write_confirm_sweep_report_md(
        report_path,
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )

    append_production_recommendation_to_report(
        report_path, recommendation=rec,
    )
    md1 = report_path.read_text(encoding="utf-8")
    n1 = md1.count("## Production recommendation")
    assert n1 == 1

    # Re-run → still exactly one Production recommendation section.
    append_production_recommendation_to_report(
        report_path, recommendation=rec,
    )
    md2 = report_path.read_text(encoding="utf-8")
    n2 = md2.count("## Production recommendation")
    assert n2 == 1
    # And no forbidden phrases got introduced.
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        assert phrase not in md2


def test_append_production_recommendation_to_report_inserts_before_reminders(
    tmp_path: Path,
) -> None:
    """The appender must insert the section before ``## Reminders`` so
    the reminders block stays at the bottom of the report."""
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None

    report_path = tmp_path / "report.md"
    write_confirm_sweep_report_md(
        report_path,
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )
    append_production_recommendation_to_report(
        report_path, recommendation=rec,
    )
    md = report_path.read_text(encoding="utf-8")
    prod_idx = md.find("## Production recommendation")
    rem_idx = md.find("## Reminders")
    assert prod_idx != -1 and rem_idx != -1
    assert prod_idx < rem_idx


def test_production_recommended_lambda_default_is_seventy() -> None:
    """The default prefer_lambda is the previous Phase 7.x best (0.70)."""
    assert PRODUCTION_RECOMMENDED_LAMBDA == pytest.approx(0.70)
    assert PRODUCTION_PLATEAU_EPSILON > 0


def test_production_recommendation_to_dict_round_trip() -> None:
    """The to_dict() shape carries every reviewer-facing key without
    losing precision."""
    result = _flat_plateau_result_at_candidate_k_40()
    base_g, base_s = _baseline_pair()
    rec = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
    )
    assert rec is not None
    d = rec.to_dict()
    expected_keys = {
        "confirmed_best_name", "confirmed_best_lambda",
        "confirmed_best_primary_score", "recommended_lambda",
        "recommended_variant_name", "top_k", "candidate_k", "use_mmr",
        "mmr_lambda", "cache_dir_relative", "plateau_status",
        "plateau_lambdas", "selected_lambda_policy",
        "selected_lambda_reason", "rollback_env", "baseline_metrics",
        "confirmed_metrics", "section_hit_caveat",
        "promotion_target_clarification", "human_focus_disclaimer",
    }
    assert expected_keys <= set(d.keys())
