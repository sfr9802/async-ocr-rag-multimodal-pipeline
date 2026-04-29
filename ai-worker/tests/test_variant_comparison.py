"""Tests for ``eval/harness/variant_comparison.py``.

Pin the verdict logic (A/B/C/D), the delta computation contract, the
reranker-audit sample shape, and the per-query diff aggregator. All
tests are pure-Python — no FAISS, no embedder, no reranker. The
markdown writer + regression_guard rely on these contracts; future
threshold drift would surface in unit-test failures rather than as
silent report shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from eval.harness.confirm_wide_mmr_helpers import (
    GRADE_BASELINE,
    GRADE_DIAG_ONLY,
    GRADE_INCONCLUSIVE,
    GRADE_PROMISING,
    GRADE_REGRESSION,
)
from eval.harness.variant_comparison import (
    ANCHOR_VARIANT,
    EPS_CANDIDATE,
    RerankerAuditSample,
    TITLE_SECTION_VARIANT,
    TITLE_VARIANT,
    VERDICT_ADOPT_TITLE,
    VERDICT_ADOPT_TITLE_SECTION,
    VERDICT_KEEP_RAW,
    VERDICT_NEED_RERANKER_AUDIT,
    VariantDeltas,
    VariantPerQueryDelta,
    candidate_pool_recoverable_miss_count,
    candidate_pool_unrecoverable_miss_count,
    collect_reranker_audit_samples,
    compute_variant_deltas,
    decide_variant_verdict,
    variant_per_query_diff,
)


# ---------------------------------------------------------------------------
# Stub summary objects that mimic the dataclass / dict surfaces the
# helpers read off ``RetrievalEvalSummary``.
# ---------------------------------------------------------------------------


@dataclass
class _StubSummary:
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


def _raw_anchor_summary() -> _StubSummary:
    """Stand-in for the raw variant's optuna_winner_top8 row.

    Numbers chosen to match the actual confirm result on silver_200
    (hit@5 = 0.74, MRR = 0.6698, cand@50 = 0.80, p95 = 537 ms) so the
    delta math here mirrors realistic deltas.
    """
    return _StubSummary(
        mean_hit_at_1=0.55,
        mean_hit_at_3=0.65,
        mean_hit_at_5=0.74,
        mean_mrr_at_10=0.6698,
        mean_ndcg_at_10=0.6743,
        candidate_hit_rates={"10": 0.72, "20": 0.78, "50": 0.80, "100": 0.80},
        duplicate_doc_ratios={"5": 0.30, "10": 0.50},
        unique_doc_counts={"10": 4.0},
        p50_retrieval_ms=420.0,
        p95_retrieval_ms=537.0,
        p99_retrieval_ms=600.0,
    )


def _shifted(base: _StubSummary, *, dh5: float = 0.0, dmrr: float = 0.0,
             dcand50: float = 0.0, dcand100: float = 0.0,
             ddup10: float = 0.0, dp95: float = 0.0) -> _StubSummary:
    """Return a stub summary with the named deltas applied."""
    return _StubSummary(
        mean_hit_at_1=base.mean_hit_at_1,
        mean_hit_at_3=base.mean_hit_at_3,
        mean_hit_at_5=(base.mean_hit_at_5 or 0.0) + dh5,
        mean_mrr_at_10=(base.mean_mrr_at_10 or 0.0) + dmrr,
        mean_ndcg_at_10=base.mean_ndcg_at_10,
        candidate_hit_rates={
            "10": base.candidate_hit_rates.get("10"),
            "20": base.candidate_hit_rates.get("20"),
            "50": (base.candidate_hit_rates.get("50") or 0.0) + dcand50,
            "100": (base.candidate_hit_rates.get("100") or 0.0) + dcand100,
        },
        duplicate_doc_ratios={
            "5": base.duplicate_doc_ratios.get("5"),
            "10": (base.duplicate_doc_ratios.get("10") or 0.0) + ddup10,
        },
        unique_doc_counts=dict(base.unique_doc_counts),
        p50_retrieval_ms=base.p50_retrieval_ms,
        p95_retrieval_ms=(base.p95_retrieval_ms or 0.0) + dp95,
        p99_retrieval_ms=base.p99_retrieval_ms,
    )


# ---------------------------------------------------------------------------
# 1. compute_variant_deltas — anchor self-grade + grade transitions
# ---------------------------------------------------------------------------


class TestComputeVariantDeltas:
    def test_raw_anchor_self_grades_baseline(self):
        s = _raw_anchor_summary()
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=ANCHOR_VARIANT,
            variant_summary=s,
            raw_summary=s,
        )
        assert d.grade == GRADE_BASELINE
        assert d.delta_hit_at_5 == 0.0
        assert d.delta_mrr_at_10 == 0.0
        assert d.latency_ratio_p95 == 1.0
        assert d.delta_candidate_hit_at_50 == 0.0

    def test_promising_when_hit5_lifts_with_no_latency_blow(self):
        raw = _raw_anchor_summary()
        v = _shifted(raw, dh5=0.020, dmrr=0.018, dcand50=0.020, dp95=10.0)
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_SECTION_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_PROMISING
        assert d.delta_hit_at_5 == pytest.approx(0.020, abs=1e-4)
        assert d.delta_mrr_at_10 == pytest.approx(0.018, abs=1e-4)
        assert d.delta_candidate_hit_at_50 == pytest.approx(0.020, abs=1e-4)

    def test_regression_when_hit5_drops(self):
        raw = _raw_anchor_summary()
        v = _shifted(raw, dh5=-0.010, dmrr=-0.005)
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_REGRESSION

    def test_regression_when_cand50_drops(self):
        raw = _raw_anchor_summary()
        # hit@5 / mrr unchanged but cand@50 drops past EPS.
        v = _shifted(raw, dcand50=-0.010)
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_REGRESSION

    def test_inconclusive_when_all_inside_epsilon(self):
        raw = _raw_anchor_summary()
        v = _shifted(raw, dh5=0.001, dmrr=0.001)  # below EPS
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_INCONCLUSIVE

    def test_diag_only_when_pool_lifts_but_final_flat(self):
        raw = _raw_anchor_summary()
        # cand@50 lifts ≥ EPS_CANDIDATE but final hit@5/MRR are inside EPS.
        v = _shifted(raw, dcand50=0.015, dh5=0.001)
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_SECTION_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_DIAG_ONLY
        assert "reranker is not propagating" in d.reason

    def test_diag_only_when_quality_up_but_latency_blown(self):
        raw = _raw_anchor_summary()
        # hit@5 lifts but p95 latency triples.
        v = _shifted(raw, dh5=0.020, dp95=raw.p95_retrieval_ms * 2)
        d = compute_variant_deltas(
            cell_label="optuna_winner_top8",
            variant=TITLE_SECTION_VARIANT,
            variant_summary=v,
            raw_summary=raw,
        )
        assert d.grade == GRADE_DIAG_ONLY
        assert "latency ratio" in d.reason

    def test_dict_summaries_equivalent_to_dataclass(self):
        raw_dataclass = _raw_anchor_summary()
        raw_dict = {
            "mean_hit_at_5": raw_dataclass.mean_hit_at_5,
            "mean_mrr_at_10": raw_dataclass.mean_mrr_at_10,
            "mean_ndcg_at_10": raw_dataclass.mean_ndcg_at_10,
            "mean_hit_at_1": raw_dataclass.mean_hit_at_1,
            "mean_hit_at_3": raw_dataclass.mean_hit_at_3,
            "candidate_hit_rates": dict(raw_dataclass.candidate_hit_rates),
            "duplicate_doc_ratios": dict(raw_dataclass.duplicate_doc_ratios),
            "unique_doc_counts": dict(raw_dataclass.unique_doc_counts),
            "p50_retrieval_ms": raw_dataclass.p50_retrieval_ms,
            "p95_retrieval_ms": raw_dataclass.p95_retrieval_ms,
            "p99_retrieval_ms": raw_dataclass.p99_retrieval_ms,
        }
        # Both should yield the same deltas.
        d_a = compute_variant_deltas(
            cell_label="x", variant="title",
            variant_summary=_shifted(raw_dataclass, dh5=0.02),
            raw_summary=raw_dataclass,
        )
        d_b = compute_variant_deltas(
            cell_label="x", variant="title",
            variant_summary=_shifted(raw_dataclass, dh5=0.02),
            raw_summary=raw_dict,
        )
        assert d_a.delta_hit_at_5 == d_b.delta_hit_at_5


# ---------------------------------------------------------------------------
# 2. decide_variant_verdict — A / B / C / D pinning
# ---------------------------------------------------------------------------


def _stub_deltas(
    *,
    cell="optuna_winner_top8",
    variant=TITLE_SECTION_VARIANT,
    grade=GRADE_PROMISING,
    dh5=0.02, dmrr=0.018,
    dcand10=0.0, dcand20=0.0, dcand50=0.0, dcand100=0.0,
    ddup5=0.0, ddup10=0.0, duniq10=0.0,
    dp50=0.0, dp95=0.0, dp99=0.0,
    latency_ratio=1.02,
) -> VariantDeltas:
    return VariantDeltas(
        cell_label=cell, variant=variant, grade=grade, reason="stub",
        delta_hit_at_1=0.0, delta_hit_at_3=0.0,
        delta_hit_at_5=dh5,
        delta_mrr_at_10=dmrr,
        delta_ndcg_at_10=0.0,
        delta_candidate_hit_at_10=dcand10,
        delta_candidate_hit_at_20=dcand20,
        delta_candidate_hit_at_50=dcand50,
        delta_candidate_hit_at_100=dcand100,
        delta_duplicate_ratio_at_5=ddup5,
        delta_duplicate_ratio_at_10=ddup10,
        delta_unique_doc_count_at_10=duniq10,
        delta_p50ms=dp50, delta_p95ms=dp95, delta_p99ms=dp99,
        latency_ratio_p95=latency_ratio,
    )


class TestDecideVariantVerdict:
    def test_case_a_adopt_title_section(self):
        ts = _stub_deltas(dh5=0.02, dmrr=0.018, dcand50=0.02)
        title = _stub_deltas(dh5=0.005, dmrr=0.004)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_ADOPT_TITLE_SECTION

    def test_case_b_adopt_title_only(self):
        # Title is promising; title_section regresses on hit@5.
        ts = _stub_deltas(dh5=-0.01, dmrr=0.0, dcand50=0.0)
        title = _stub_deltas(dh5=0.015, dmrr=0.012, dcand50=0.0)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_ADOPT_TITLE

    def test_case_c_keep_raw_when_neither_moves(self):
        ts = _stub_deltas(dh5=0.001, dmrr=0.0)
        title = _stub_deltas(dh5=-0.001, dmrr=0.0)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_KEEP_RAW

    def test_case_d_reranker_audit_when_pool_lifts_but_final_flat(self):
        # title_section: cand@50 ↑ 0.02 but final hit@5 inside epsilon.
        ts = _stub_deltas(
            dh5=0.001, dmrr=0.001,
            dcand50=0.020, dcand100=0.020,
        )
        title = _stub_deltas(dh5=0.0, dmrr=0.0)
        verdict, rationale = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_NEED_RERANKER_AUDIT
        assert "Candidate-pool" in rationale or "candidate-pool" in rationale.lower()

    def test_case_d_reranker_audit_when_pool_lifts_but_final_regressed(self):
        """Spec D wording — *cand@K up, final not improving* — covers
        the regression case too. The silver_200 result (cand@50 +0.045,
        final hit@5 -0.045) was misclassified as ``KEEP_RAW`` by an
        earlier draft of ``_has_dense_pool_lift_only`` that required a
        flat final. Pin the corrected behaviour."""
        ts = _stub_deltas(
            dh5=-0.045, dmrr=-0.030,
            dcand50=0.045, dcand100=0.040,
        )
        title = _stub_deltas(
            dh5=-0.060, dmrr=-0.051,
            dcand50=0.035, dcand100=0.030,
        )
        verdict, rationale = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_NEED_RERANKER_AUDIT
        assert "Candidate-pool" in rationale or "candidate-pool" in rationale.lower()

    def test_case_d_when_pool_lift_with_mixed_final(self):
        """One of (h5, mrr) flat and the other regressed still triggers D."""
        ts = _stub_deltas(
            dh5=-0.020, dmrr=0.001,
            dcand50=0.030,
        )
        title = _stub_deltas(dh5=0.0, dmrr=0.0)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_NEED_RERANKER_AUDIT

    def test_case_a_wins_over_d_when_both_apply(self):
        # title_section is genuinely promising AND cand lifts —
        # adoption should win over the audit-first fallback.
        ts = _stub_deltas(
            dh5=0.020, dmrr=0.018,
            dcand50=0.030, dcand100=0.030,
        )
        title = _stub_deltas(dh5=0.0, dmrr=0.0)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_ADOPT_TITLE_SECTION

    def test_latency_blow_blocks_adoption(self):
        # title_section is quality-positive but latency ratio > limit
        # → not adopt. With title not quality-positive either, fall
        # back to KEEP_RAW.
        ts = _stub_deltas(dh5=0.020, dmrr=0.018, latency_ratio=2.0)
        title = _stub_deltas(dh5=0.0, dmrr=0.0)
        verdict, _ = decide_variant_verdict(
            title_deltas=title, title_section_deltas=ts,
        )
        assert verdict == VERDICT_KEEP_RAW

    def test_none_inputs_return_keep_raw(self):
        verdict, _ = decide_variant_verdict(
            title_deltas=None, title_section_deltas=None,
        )
        assert verdict == VERDICT_KEEP_RAW


# ---------------------------------------------------------------------------
# 3. Reranker audit sample collection
# ---------------------------------------------------------------------------


def _row_dict(
    *, rid, query, expected, retrieved, candidates, hit_at_5,
):
    return {
        "id": rid,
        "query": query,
        "expected_doc_ids": expected,
        "retrieved_doc_ids": retrieved,
        "candidate_doc_ids": candidates,
        "hit_at_5": hit_at_5,
    }


class TestCollectRerankerAuditSamples:
    def test_qualifies_when_gold_in_pool_but_not_in_top5(self):
        rows = [
            _row_dict(
                rid="q1", query="bookshop", expected=["doc-book"],
                retrieved=["doc-cats", "doc-aoi", "doc-mech",
                           "doc-other", "doc-x"],
                candidates=["doc-cats", "doc-book", "doc-aoi"],
                hit_at_5=0.0,
            ),
        ]
        samples = collect_reranker_audit_samples(
            cell_label="optuna_winner_top8",
            variant="title_section",
            rows=rows,
            chunk_text_lookup={"doc-book": "Bookshop\noverview\nbody text"},
            chunk_title_lookup={"doc-book": "Bookshop"},
            truncation_threshold_chars=800,
            limit=5,
        )
        assert len(samples) == 1
        s = samples[0]
        assert isinstance(s, RerankerAuditSample)
        assert s.query_id == "q1"
        assert s.gold_in_candidates
        assert s.gold_passage_has_title  # title appears in preview
        assert not s.gold_passage_truncated

    def test_skips_rows_with_gold_in_top5(self):
        rows = [
            _row_dict(
                rid="q1", query="x", expected=["doc-a"],
                retrieved=["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"],
                candidates=["doc-a", "doc-b"],
                hit_at_5=1.0,
            ),
        ]
        samples = collect_reranker_audit_samples(
            cell_label="x", variant="raw", rows=rows,
            chunk_text_lookup={}, chunk_title_lookup={},
        )
        assert samples == []

    def test_skips_when_gold_not_in_candidates(self):
        # The reranker can't be blamed if dense never surfaced gold.
        rows = [
            _row_dict(
                rid="q1", query="x", expected=["doc-z"],
                retrieved=["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"],
                candidates=["doc-a", "doc-b", "doc-c"],
                hit_at_5=0.0,
            ),
        ]
        samples = collect_reranker_audit_samples(
            cell_label="x", variant="raw", rows=rows,
            chunk_text_lookup={}, chunk_title_lookup={},
        )
        assert samples == []

    def test_truncation_flag_fires_above_threshold(self):
        long_text = "PaddedTitle\n" + ("x" * 2000)
        rows = [
            _row_dict(
                rid="q1", query="x", expected=["doc-a"],
                retrieved=["doc-b"] * 5,
                candidates=["doc-a", "doc-b"],
                hit_at_5=0.0,
            ),
        ]
        samples = collect_reranker_audit_samples(
            cell_label="x", variant="title",
            rows=rows,
            chunk_text_lookup={"doc-a": long_text},
            chunk_title_lookup={"doc-a": "PaddedTitle"},
            truncation_threshold_chars=800,
        )
        assert samples and samples[0].gold_passage_truncated

    def test_limit_caps_sample_count(self):
        rows = [
            _row_dict(
                rid=f"q{i}", query="x", expected=[f"doc-{i}"],
                retrieved=["doc-z"] * 5,
                candidates=[f"doc-{i}"],
                hit_at_5=0.0,
            ) for i in range(20)
        ]
        samples = collect_reranker_audit_samples(
            cell_label="x", variant="raw", rows=rows,
            chunk_text_lookup={}, chunk_title_lookup={}, limit=3,
        )
        assert len(samples) == 3

    def test_skips_rows_without_expected(self):
        rows = [
            _row_dict(
                rid="q1", query="x", expected=[],
                retrieved=["doc-a"], candidates=["doc-a"], hit_at_5=None,
            ),
        ]
        samples = collect_reranker_audit_samples(
            cell_label="x", variant="raw", rows=rows,
            chunk_text_lookup={}, chunk_title_lookup={},
        )
        assert samples == []


# ---------------------------------------------------------------------------
# 4. variant_per_query_diff
# ---------------------------------------------------------------------------


class TestVariantPerQueryDiff:
    def test_improvements_and_regressions_partition(self):
        raw_rows = [
            _row_dict(rid="q1", query="a", expected=["d-a"],
                      retrieved=["d-x"], candidates=[], hit_at_5=0.0),
            _row_dict(rid="q2", query="b", expected=["d-b"],
                      retrieved=["d-b"], candidates=[], hit_at_5=1.0),
            _row_dict(rid="q3", query="c", expected=["d-c"],
                      retrieved=["d-c"], candidates=[], hit_at_5=1.0),
        ]
        var_rows = [
            _row_dict(rid="q1", query="a", expected=["d-a"],
                      retrieved=["d-a"], candidates=[], hit_at_5=1.0),
            _row_dict(rid="q2", query="b", expected=["d-b"],
                      retrieved=["d-x"], candidates=[], hit_at_5=0.0),
            _row_dict(rid="q3", query="c", expected=["d-c"],
                      retrieved=["d-c"], candidates=[], hit_at_5=1.0),
        ]
        improved, regressed = variant_per_query_diff(
            cell_label="optuna_winner_top8",
            variant=TITLE_SECTION_VARIANT,
            raw_rows=raw_rows,
            variant_rows=var_rows,
        )
        assert [e.id for e in improved] == ["q1"]
        assert [e.id for e in regressed] == ["q2"]

    def test_skips_rows_with_none_hit(self):
        raw_rows = [
            _row_dict(rid="q1", query="a", expected=[],
                      retrieved=[], candidates=[], hit_at_5=None),
        ]
        var_rows = [
            _row_dict(rid="q1", query="a", expected=[],
                      retrieved=[], candidates=[], hit_at_5=None),
        ]
        improved, regressed = variant_per_query_diff(
            cell_label="x", variant="title",
            raw_rows=raw_rows, variant_rows=var_rows,
        )
        assert improved == []
        assert regressed == []


# ---------------------------------------------------------------------------
# 5. Recoverable / unrecoverable miss counters
# ---------------------------------------------------------------------------


class TestMissCounters:
    def test_unrecoverable_misses_count_pure_dense_misses(self):
        rows = [
            _row_dict(rid="q1", query="x", expected=["d-a"],
                      retrieved=["d-x"], candidates=["d-x", "d-y"],
                      hit_at_5=0.0),
            _row_dict(rid="q2", query="y", expected=["d-b"],
                      retrieved=["d-b"], candidates=["d-b"],
                      hit_at_5=1.0),
        ]
        assert candidate_pool_unrecoverable_miss_count(rows) == 1

    def test_recoverable_misses_count_only_pool_hits(self):
        rows = [
            # Gold in candidates but not in top-5 → recoverable.
            _row_dict(rid="q1", query="x", expected=["d-a"],
                      retrieved=["d-x", "d-y", "d-z", "d-w", "d-v"],
                      candidates=["d-a", "d-x", "d-y"], hit_at_5=0.0),
            # Gold in top-5 → not recoverable miss.
            _row_dict(rid="q2", query="y", expected=["d-b"],
                      retrieved=["d-b"], candidates=["d-b"],
                      hit_at_5=1.0),
        ]
        assert candidate_pool_recoverable_miss_count(rows) == 1

    def test_skips_rows_without_expected(self):
        rows = [_row_dict(rid="q", query="x", expected=[],
                          retrieved=["d"], candidates=["d"], hit_at_5=None)]
        assert candidate_pool_unrecoverable_miss_count(rows) == 0
        assert candidate_pool_recoverable_miss_count(rows) == 0
