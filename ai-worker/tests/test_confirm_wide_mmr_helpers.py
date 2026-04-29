"""Tests for the wide-MMR confirm sweep helpers.

Covers ``ConfirmCellSpec`` roster, ``compute_cell_deltas`` grader,
``decide_verdict`` head-to-head logic, ``per_query_diff`` flip
accounting, and ``candidate_pool_recoverable_misses``.

All tests are pure-Python — no FAISS, no embedder, no reranker. Pin
the contract the markdown writer + regression_guard rely on so a
future refactor on the metric thresholds can't drift silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from eval.harness.confirm_wide_mmr_helpers import (
    DEFAULT_CONFIRM_CELLS,
    EPS_HIT,
    EPS_MRR,
    GRADE_BASELINE,
    GRADE_DIAG_ONLY,
    GRADE_INCONCLUSIVE,
    GRADE_PROMISING,
    GRADE_REGRESSION,
    GROUP_BASELINE,
    GROUP_CAP_FINAL,
    GROUP_FINAL_TOPK,
    GROUP_LAMBDA,
    GROUP_OPTUNA_WINNER,
    GROUP_PHASE1,
    LATENCY_RATIO_LIMIT,
    PerQueryDiffEntry,
    THRESHOLD_PROMISING_HIT,
    THRESHOLD_PROMISING_MRR,
    VERDICT_ADOPT_OPTUNA,
    VERDICT_INCONCLUSIVE,
    VERDICT_KEEP_PHASE1,
    candidate_pool_recoverable_misses,
    compute_cell_deltas,
    decide_verdict,
    default_confirm_cells,
    per_query_diff,
)


# ---------------------------------------------------------------------------
# Helpers — stub summary objects matching the dict / dataclass duck shape
# ``compute_cell_deltas`` reads.
# ---------------------------------------------------------------------------


@dataclass
class _StubSummary:
    """Minimal summary surface — fields the helpers actually read."""

    mean_hit_at_1: Optional[float] = None
    mean_hit_at_3: Optional[float] = None
    mean_hit_at_5: Optional[float] = None
    mean_mrr_at_10: Optional[float] = None
    mean_ndcg_at_10: Optional[float] = None
    candidate_hit_rates: Dict[str, Optional[float]] = field(default_factory=dict)
    candidate_recalls: Dict[str, Optional[float]] = field(default_factory=dict)
    duplicate_doc_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    unique_doc_counts: Dict[str, Optional[float]] = field(default_factory=dict)
    p50_retrieval_ms: Optional[float] = None
    p95_retrieval_ms: Optional[float] = None
    p99_retrieval_ms: Optional[float] = None
    p95_total_retrieval_ms: Optional[float] = None
    rerank_uplift_hit_at_5: Optional[float] = None
    rerank_uplift_mrr_at_10: Optional[float] = None


def _baseline_summary() -> _StubSummary:
    return _StubSummary(
        mean_hit_at_1=0.55,
        mean_hit_at_3=0.65,
        mean_hit_at_5=0.7200,
        mean_mrr_at_10=0.6587,
        mean_ndcg_at_10=0.6743,
        candidate_hit_rates={"10": 0.72, "20": 0.78, "50": 0.80, "100": 0.80},
        candidate_recalls={"50": 0.80, "100": 0.80},
        duplicate_doc_ratios={"5": 0.30, "10": 0.636},
        unique_doc_counts={"10": 3.64},
        p50_retrieval_ms=900.0,
        p95_retrieval_ms=1133.0,
        p99_retrieval_ms=1200.0,
    )


# ---------------------------------------------------------------------------
# 1. Default cell roster
# ---------------------------------------------------------------------------


class TestDefaultCellRoster:
    def test_default_cells_count_and_groups(self):
        cells = default_confirm_cells()
        # 12 cells per the spec.
        assert len(cells) == 12
        groups = {c.group for c in cells}
        assert groups == {
            GROUP_BASELINE, GROUP_PHASE1, GROUP_OPTUNA_WINNER,
            GROUP_CAP_FINAL, GROUP_LAMBDA, GROUP_FINAL_TOPK,
        }

    def test_baseline_is_first_and_singleton(self):
        cells = default_confirm_cells()
        assert cells[0].label == "baseline_k50_top5"
        assert cells[0].group == GROUP_BASELINE
        baselines = [c for c in cells if c.group == GROUP_BASELINE]
        assert len(baselines) == 1

    def test_phase1_two_cells(self):
        cells = [c for c in default_confirm_cells() if c.group == GROUP_PHASE1]
        labels = {c.label for c in cells}
        assert labels == {"phase1_best_cap2_top8", "phase1_cap1_top8"}
        cap2 = next(c for c in cells if c.label == "phase1_best_cap2_top8")
        assert cap2.candidate_k == 200
        assert cap2.title_cap_rerank_input == 2
        assert cap2.title_cap_final == 2
        assert cap2.rerank_in == 32
        assert cap2.use_mmr is True

    def test_optuna_winner_recipe_locked(self):
        cells = default_confirm_cells()
        winner = next(c for c in cells if c.label == "optuna_winner_top8")
        assert winner.group == GROUP_OPTUNA_WINNER
        assert winner.candidate_k == 100
        assert winner.rerank_in == 16
        assert winner.use_mmr is True
        assert winner.title_cap_rerank_input == 1
        # cap_final defaults to 2 per the spec — sensitivity cells
        # explore the 1 / 3 alternatives separately.
        assert winner.title_cap_final == 2
        assert winner.final_top_k == 8

    def test_cap_final_sensitivity_covers_1_and_3(self):
        cells = [
            c for c in default_confirm_cells()
            if c.group == GROUP_CAP_FINAL
        ]
        cap_finals = sorted(c.title_cap_final for c in cells)
        assert cap_finals == [1, 3]
        # All cells share the winner anchor recipe.
        for c in cells:
            assert c.candidate_k == 100
            assert c.rerank_in == 16
            assert c.title_cap_rerank_input == 1
            assert c.final_top_k == 8

    def test_lambda_sensitivity_covers_unsampled_edges(self):
        cells = [
            c for c in default_confirm_cells()
            if c.group == GROUP_LAMBDA
        ]
        lambdas = sorted(round(c.mmr_lambda, 2) for c in cells)
        # 0.55 and 0.75 ARE the round_05 unsampled edges; 0.60 / 0.70
        # bracket the winner default.
        assert 0.55 in lambdas
        assert 0.75 in lambdas
        assert 0.60 in lambdas
        assert 0.70 in lambdas
        # The winner default 0.65 is NOT duplicated here — it lives on
        # the optuna_winner_top8 cell so the sensitivity table is just
        # the deltas.
        assert 0.65 not in lambdas

    def test_final_topk_sensitivity_covers_5_and_10(self):
        cells = [
            c for c in default_confirm_cells()
            if c.group == GROUP_FINAL_TOPK
        ]
        topks = sorted(c.final_top_k for c in cells)
        assert topks == [5, 10]

    def test_default_cells_immutable_tuple(self):
        # The exported alias must be a tuple so callers can't mutate
        # the canonical list at module level.
        assert isinstance(DEFAULT_CONFIRM_CELLS, tuple)
        assert len(DEFAULT_CONFIRM_CELLS) == 12
        # Each entry is frozen — assignment must raise.
        with pytest.raises((AttributeError, Exception)):
            DEFAULT_CONFIRM_CELLS[0].candidate_k = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. compute_cell_deltas grader
# ---------------------------------------------------------------------------


class TestComputeCellDeltas:
    def test_baseline_self_grade(self):
        baseline = _baseline_summary()
        deltas = compute_cell_deltas(
            label="baseline_k50_top5",
            group=GROUP_BASELINE,
            cell_summary=baseline,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_BASELINE
        assert deltas.delta_hit_at_5 == 0.0
        assert deltas.delta_mrr_at_10 == 0.0
        assert deltas.latency_ratio_p95 == 1.0

    def test_promising_quality_when_hit_jumps(self):
        baseline = _baseline_summary()
        cell = _StubSummary(
            mean_hit_at_5=baseline.mean_hit_at_5 + 0.020,
            mean_mrr_at_10=baseline.mean_mrr_at_10 + 0.011,
            candidate_hit_rates={"50": 0.80, "100": 0.815},
            duplicate_doc_ratios={"5": 0.20, "10": 0.20},
            unique_doc_counts={"10": 8.0},
            p50_retrieval_ms=920.0, p95_retrieval_ms=1170.0,
        )
        deltas = compute_cell_deltas(
            label="phase1_best_cap2_top8",
            group=GROUP_PHASE1,
            cell_summary=cell,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_PROMISING
        assert pytest.approx(deltas.delta_hit_at_5, abs=1e-6) == 0.020
        assert deltas.delta_mrr_at_10 > 0
        # latency_ratio close to 1.0 keeps the cell out of diagnostic_only.
        assert deltas.latency_ratio_p95 < LATENCY_RATIO_LIMIT

    def test_regression_when_mrr_drops(self):
        baseline = _baseline_summary()
        cell = _StubSummary(
            mean_hit_at_5=baseline.mean_hit_at_5 - 0.001,  # within EPS
            mean_mrr_at_10=baseline.mean_mrr_at_10 - 0.020,  # past EPS
            candidate_hit_rates={"50": 0.80, "100": 0.80},
            p95_retrieval_ms=baseline.p95_retrieval_ms,
        )
        deltas = compute_cell_deltas(
            label="some_regressing_cell",
            group=GROUP_OPTUNA_WINNER,
            cell_summary=cell,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_REGRESSION
        assert deltas.delta_mrr_at_10 < -EPS_MRR

    def test_inconclusive_when_within_epsilon(self):
        baseline = _baseline_summary()
        # Hold both metrics inside the epsilon band.
        cell = _StubSummary(
            mean_hit_at_5=baseline.mean_hit_at_5 + 0.001,
            mean_mrr_at_10=baseline.mean_mrr_at_10 + 0.001,
            candidate_hit_rates={"50": 0.80, "100": 0.80},
            p95_retrieval_ms=baseline.p95_retrieval_ms,
        )
        deltas = compute_cell_deltas(
            label="tiny_change",
            group=GROUP_OPTUNA_WINNER,
            cell_summary=cell,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_INCONCLUSIVE

    def test_diagnostic_only_when_quality_up_but_latency_too_high(self):
        baseline = _baseline_summary()
        cell = _StubSummary(
            mean_hit_at_5=baseline.mean_hit_at_5 + 0.020,
            mean_mrr_at_10=baseline.mean_mrr_at_10 + 0.015,
            candidate_hit_rates={"50": 0.80, "100": 0.85},
            # 2x latency — past LATENCY_RATIO_LIMIT.
            p50_retrieval_ms=1800.0,
            p95_retrieval_ms=baseline.p95_retrieval_ms * 2.5,
        )
        deltas = compute_cell_deltas(
            label="too_slow",
            group=GROUP_LAMBDA,
            cell_summary=cell,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_DIAG_ONLY
        assert deltas.latency_ratio_p95 > LATENCY_RATIO_LIMIT

    def test_compute_with_dict_summary_works(self):
        """The grader accepts both dataclass and dict shapes."""
        baseline = {
            "mean_hit_at_5": 0.72,
            "mean_mrr_at_10": 0.6587,
            "candidate_hit_rates": {"50": 0.80, "100": 0.80},
            "duplicate_doc_ratios": {"10": 0.636},
            "unique_doc_counts": {"10": 3.64},
            "p95_retrieval_ms": 1133.0,
        }
        cell = {
            "mean_hit_at_5": 0.7400,
            "mean_mrr_at_10": 0.6699,
            "candidate_hit_rates": {"50": 0.80, "100": 0.815},
            "duplicate_doc_ratios": {"10": 0.20},
            "unique_doc_counts": {"10": 8.0},
            "p95_retrieval_ms": 1165.0,
        }
        deltas = compute_cell_deltas(
            label="dict_shape",
            group=GROUP_PHASE1,
            cell_summary=cell,
            baseline_summary=baseline,
        )
        assert deltas.grade == GRADE_PROMISING
        assert deltas.delta_hit_at_5 == pytest.approx(0.02, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. decide_verdict
# ---------------------------------------------------------------------------


class TestDecideVerdict:
    def test_inconclusive_when_both_regress(self):
        baseline = _StubSummary(mean_hit_at_5=0.80, mean_mrr_at_10=0.70)
        phase1 = _StubSummary(mean_hit_at_5=0.78, mean_mrr_at_10=0.68)
        optuna = _StubSummary(mean_hit_at_5=0.78, mean_mrr_at_10=0.69)
        verdict, rationale = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.0,
            latency_ratio_optuna=1.0,
        )
        assert verdict == VERDICT_INCONCLUSIVE
        assert "regress" in rationale.lower()

    def test_adopt_optuna_when_winner_clearly_better(self):
        # Optuna MRR uplift over phase1 must clear EPS_MRR (0.005);
        # 0.020 is well above that boundary.
        baseline = _StubSummary(mean_hit_at_5=0.72, mean_mrr_at_10=0.6587)
        phase1 = _StubSummary(mean_hit_at_5=0.74, mean_mrr_at_10=0.6699)
        optuna = _StubSummary(
            mean_hit_at_5=0.74, mean_mrr_at_10=0.6920,
        )
        verdict, rationale = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.02,
            latency_ratio_optuna=1.05,
        )
        assert verdict == VERDICT_ADOPT_OPTUNA
        assert "MRR" in rationale or "mrr" in rationale.lower()

    def test_realistic_optuna_uplift_below_epsilon_is_inconclusive(self):
        """The actual MRR gap between phase1 (0.6699) and optuna
        (0.6745) on the 100-row subset is 0.0046 — below EPS_MRR. The
        verdict for this case should be INCONCLUSIVE, not ADOPT.
        """
        baseline = _StubSummary(mean_hit_at_5=0.72, mean_mrr_at_10=0.6587)
        phase1 = _StubSummary(mean_hit_at_5=0.74, mean_mrr_at_10=0.6699)
        optuna = _StubSummary(
            mean_hit_at_5=0.74, mean_mrr_at_10=0.6745,
        )
        verdict, _ = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.02,
            latency_ratio_optuna=1.05,
        )
        assert verdict == VERDICT_INCONCLUSIVE

    def test_keep_phase1_when_phase1_clearly_better(self):
        baseline = _StubSummary(mean_hit_at_5=0.72, mean_mrr_at_10=0.6587)
        phase1 = _StubSummary(mean_hit_at_5=0.74, mean_mrr_at_10=0.6745)
        optuna = _StubSummary(
            mean_hit_at_5=0.72, mean_mrr_at_10=0.6620,
        )
        verdict, rationale = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.02,
            latency_ratio_optuna=0.95,
        )
        assert verdict == VERDICT_KEEP_PHASE1

    def test_inconclusive_when_close_to_each_other(self):
        baseline = _StubSummary(mean_hit_at_5=0.72, mean_mrr_at_10=0.6587)
        phase1 = _StubSummary(mean_hit_at_5=0.74, mean_mrr_at_10=0.6699)
        optuna = _StubSummary(
            mean_hit_at_5=0.74, mean_mrr_at_10=0.6701,  # +0.0002 — < EPS
        )
        verdict, _ = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.02,
            latency_ratio_optuna=0.85,
        )
        assert verdict == VERDICT_INCONCLUSIVE

    def test_adopt_blocked_by_latency_breach(self):
        """A large MRR uplift must not adopt if latency ratio > limit."""
        baseline = _StubSummary(mean_hit_at_5=0.72, mean_mrr_at_10=0.6587)
        phase1 = _StubSummary(mean_hit_at_5=0.74, mean_mrr_at_10=0.6699)
        optuna = _StubSummary(
            mean_hit_at_5=0.74, mean_mrr_at_10=0.6800,  # nice uplift
        )
        verdict, _ = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.02,
            latency_ratio_optuna=2.5,  # 2.5x > LATENCY_RATIO_LIMIT
        )
        assert verdict != VERDICT_ADOPT_OPTUNA

    def test_decide_verdict_handles_dict_summaries(self):
        baseline = {"mean_hit_at_5": 0.72, "mean_mrr_at_10": 0.6587}
        phase1 = {"mean_hit_at_5": 0.74, "mean_mrr_at_10": 0.6699}
        # 0.020 MRR uplift over phase1 — clearly past EPS_MRR.
        optuna = {"mean_hit_at_5": 0.74, "mean_mrr_at_10": 0.6920}
        verdict, _ = decide_verdict(
            baseline_summary=baseline,
            phase1_best_summary=phase1,
            optuna_winner_summary=optuna,
            latency_ratio_phase1=1.0,
            latency_ratio_optuna=1.0,
        )
        assert verdict == VERDICT_ADOPT_OPTUNA


# ---------------------------------------------------------------------------
# 4. per_query_diff
# ---------------------------------------------------------------------------


class TestPerQueryDiff:
    def test_improvement_and_regression_classified(self):
        baseline_rows = [
            {"id": "q1", "query": "alpha", "hit_at_5": 1.0,
             "expected_doc_ids": ["d1"], "retrieved_doc_ids": ["d1"]},
            {"id": "q2", "query": "beta", "hit_at_5": 0.0,
             "expected_doc_ids": ["d2"], "retrieved_doc_ids": ["dx"]},
            {"id": "q3", "query": "gamma", "hit_at_5": 0.0,
             "expected_doc_ids": ["d3"], "retrieved_doc_ids": ["dy"]},
        ]
        cell_rows = [
            # q1 regressed.
            {"id": "q1", "query": "alpha", "hit_at_5": 0.0,
             "expected_doc_ids": ["d1"], "retrieved_doc_ids": ["dz"]},
            # q2 improved.
            {"id": "q2", "query": "beta", "hit_at_5": 1.0,
             "expected_doc_ids": ["d2"], "retrieved_doc_ids": ["d2"]},
            # q3 still missed — should NOT show in either bucket.
            {"id": "q3", "query": "gamma", "hit_at_5": 0.0,
             "expected_doc_ids": ["d3"], "retrieved_doc_ids": ["dz"]},
        ]
        improvements, regressions = per_query_diff(
            baseline_rows, cell_rows, cell_label="cellA",
        )
        assert {e.id for e in improvements} == {"q2"}
        assert {e.id for e in regressions} == {"q1"}
        # Top doc_ids surface for both sides.
        improved = improvements[0]
        assert improved.cell == "cellA"
        assert improved.flip_direction == "improved"
        assert improved.cell_top_doc_ids == ["d2"]
        assert improved.baseline_top_doc_ids == ["dx"]

    def test_skips_rows_with_none_hit(self):
        baseline_rows = [
            {"id": "q1", "query": "x", "hit_at_5": None,
             "expected_doc_ids": [], "retrieved_doc_ids": []},
        ]
        cell_rows = [
            {"id": "q1", "query": "x", "hit_at_5": 1.0,
             "expected_doc_ids": ["d1"], "retrieved_doc_ids": ["d1"]},
        ]
        improvements, regressions = per_query_diff(
            baseline_rows, cell_rows, cell_label="cellA",
        )
        assert improvements == []
        assert regressions == []

    def test_skips_unknown_ids(self):
        baseline_rows = [
            {"id": "q1", "hit_at_5": 0.0,
             "expected_doc_ids": ["d1"], "retrieved_doc_ids": []},
        ]
        cell_rows = [
            {"id": "q2", "hit_at_5": 1.0,
             "expected_doc_ids": ["d2"], "retrieved_doc_ids": ["d2"]},
        ]
        improvements, regressions = per_query_diff(
            baseline_rows, cell_rows, cell_label="cellA",
        )
        # q2 not in baseline — silently skipped.
        assert improvements == []
        assert regressions == []


# ---------------------------------------------------------------------------
# 5. candidate_pool_recoverable_misses
# ---------------------------------------------------------------------------


class TestCandidatePoolRecoverable:
    def test_finds_pool_hit_rerank_miss(self):
        rows = [
            # Gold doc IS in candidate pool but NOT in retrieved top-k.
            {
                "id": "q1", "query": "alpha",
                "expected_doc_ids": ["dG"],
                "candidate_doc_ids": ["dG", "dB", "dC"],
                "retrieved_doc_ids": ["dB", "dC", "dA"],
            },
            # Gold doc IS in retrieved top-k — should NOT surface.
            {
                "id": "q2", "query": "beta",
                "expected_doc_ids": ["dG"],
                "candidate_doc_ids": ["dG"],
                "retrieved_doc_ids": ["dG"],
            },
            # Gold doc NOT in candidate pool — outside helper's scope.
            {
                "id": "q3", "query": "gamma",
                "expected_doc_ids": ["dG"],
                "candidate_doc_ids": ["dA", "dB"],
                "retrieved_doc_ids": ["dA", "dB"],
            },
        ]
        recoverable = candidate_pool_recoverable_misses(rows)
        assert len(recoverable) == 1
        assert recoverable[0]["id"] == "q1"
        assert recoverable[0]["expected_doc_ids"] == ["dG"]
        assert recoverable[0]["retrieved_top_doc_ids"] == ["dB", "dC", "dA"]

    def test_no_expected_doc_ids_skipped(self):
        rows = [
            {
                "id": "q1", "query": "alpha",
                "expected_doc_ids": [],
                "candidate_doc_ids": ["dA"],
                "retrieved_doc_ids": ["dB"],
            },
        ]
        assert candidate_pool_recoverable_misses(rows) == []
