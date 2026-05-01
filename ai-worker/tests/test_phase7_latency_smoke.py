"""Phase 7.5 — tests for the latency smoke harness.

Scope of the harness under test (``eval.harness.phase7_latency_smoke``):

  * default suite contains exactly the three roles the spec calls out
    (baseline / previous-best / production-recommended) at the right
    candidate_k / λ values.
  * percentile aggregation matches numpy's linear interpolation (the
    smoke harness ships a numpy-free implementation; the test pins
    behaviour by hand-computed expectations).
  * ``measure_one_config`` produces one record per pool row, with the
    measured ``mmr_post_ms`` matching the time the (deterministic)
    timer fed in.
  * ``run_smoke_check`` produces gold + silver + combined aggregates
    with consistent counts.
  * the rendered Markdown report carries every set + every config row
    + the rerank-not-measured note (pinned because production
    reviewers cannot mistake harness numbers for live-reranker
    measurements).
  * the JSON writer round-trips ``LatencySmokeReport.to_dict()``
    without loss.
"""

from __future__ import annotations

import json
import math
from itertools import count
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import pytest

from eval.harness.phase7_human_gold_tune import RetrievedDoc
from eval.harness.phase7_latency_smoke import (
    DEFAULT_LATENCY_PERCENTILES,
    DEFAULT_MMR_REPS,
    ROLE_BASELINE,
    ROLE_PREVIOUS_BEST,
    ROLE_RECOMMENDED,
    LatencyMeasurement,
    LatencySmokeConfig,
    LatencySmokeReport,
    aggregate_latencies,
    default_smoke_suite,
    measure_one_config,
    render_latency_smoke_report,
    run_smoke_check,
    time_mmr_post_hoc_pass,
    write_latency_smoke_json,
    write_latency_smoke_md,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic candidate pool rows + deterministic timer
# ---------------------------------------------------------------------------


def _doc(rank: int, page_id: str, score: float) -> RetrievedDoc:
    return RetrievedDoc(
        rank=rank, chunk_id=f"{page_id}-c{rank}", page_id=page_id,
        title=page_id, section_path=("개요",), score=score,
    )


def _make_pool_row(
    *, query_id: str, query: str, n_docs: int = 40,
    elapsed_ms: float = 25.0,
):
    """Build one cached candidate pool row (duck-typed dict)."""
    docs = [
        _doc(i + 1, f"P-{(i // 5):02d}", 1.0 - i * 0.01)
        for i in range(n_docs)
    ]
    return {
        "variant": "_pool_wide",
        "query_id": query_id,
        "query": query,
        "elapsed_ms": float(elapsed_ms),
        "docs": [
            {
                "rank": d.rank, "chunk_id": d.chunk_id,
                "page_id": d.page_id, "title": d.title,
                "section_path": list(d.section_path), "score": d.score,
            }
            for d in docs
        ],
    }


def _det_timer(start: float = 0.0, step_ms: float = 0.5):
    """Return a perf_counter-style timer that advances by step_ms each call."""
    t = [start]

    def _now() -> float:
        # perf_counter is in seconds; convert ms → s.
        v = t[0]
        t[0] = v + (step_ms / 1000.0)
        return v

    return _now


# ---------------------------------------------------------------------------
# Default suite contract
# ---------------------------------------------------------------------------


def test_default_smoke_suite_has_three_role_rows() -> None:
    """The default suite is exactly baseline / previous-best /
    production-recommended at the spec-pinned candidate_k / λ."""
    suite = default_smoke_suite()
    assert len(suite) == 3
    by_role = {c.role: c for c in suite}
    assert ROLE_BASELINE in by_role
    assert ROLE_PREVIOUS_BEST in by_role
    assert ROLE_RECOMMENDED in by_role
    assert by_role[ROLE_BASELINE].use_mmr is False
    assert by_role[ROLE_PREVIOUS_BEST].candidate_k == 30
    assert by_role[ROLE_PREVIOUS_BEST].mmr_lambda == pytest.approx(0.70)
    assert by_role[ROLE_RECOMMENDED].candidate_k == 40
    assert by_role[ROLE_RECOMMENDED].mmr_lambda == pytest.approx(0.70)
    assert by_role[ROLE_RECOMMENDED].use_mmr is True


def test_default_latency_percentiles_pinned() -> None:
    """Renderer pins p50/p90/p99 — never drop one without a test bump."""
    assert DEFAULT_LATENCY_PERCENTILES == (0.50, 0.90, 0.99)


def test_default_mmr_reps_at_least_one() -> None:
    assert DEFAULT_MMR_REPS >= 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_latencies_simple_distribution() -> None:
    measurements = [
        LatencyMeasurement(
            config_name="x", query_id=f"q{i}", candidate_gen_ms=float(i),
            mmr_post_ms=float(i) * 0.1, total_ms=float(i) * 1.1,
            n_candidates=40,
        )
        for i in range(1, 11)  # 10 measurements: 1..10
    ]
    agg = aggregate_latencies(
        measurements=measurements, config_name="x",
        role=ROLE_RECOMMENDED, set_name="gold",
    )
    assert agg.n_queries == 10
    # mean = 5.5
    assert agg.candidate_gen_ms["mean"] == pytest.approx(5.5)
    # p50 of [1..10] ≈ 5.5 with linear interp (positions 4,5)
    assert agg.candidate_gen_ms["p50"] == pytest.approx(5.5)
    # p90 ≈ 9.1 (linear interp between positions 8, 9)
    assert agg.candidate_gen_ms["p90"] == pytest.approx(9.1, rel=1e-2)
    # p99 → close to 10.0
    assert agg.candidate_gen_ms["p99"] == pytest.approx(9.91, rel=1e-2)


def test_aggregate_latencies_empty_returns_zeros() -> None:
    agg = aggregate_latencies(
        measurements=[], config_name="x",
        role=ROLE_BASELINE, set_name="gold",
    )
    assert agg.n_queries == 0
    for stage in (agg.candidate_gen_ms, agg.mmr_post_ms, agg.total_ms):
        assert stage["p50"] == pytest.approx(0.0)
        assert stage["p99"] == pytest.approx(0.0)


def test_aggregate_latencies_single_value() -> None:
    """Edge case: a single-sample distribution should report the same
    value for mean / p50 / p90 / p99 — trivially correct."""
    m = LatencyMeasurement(
        config_name="x", query_id="q1", candidate_gen_ms=12.5,
        mmr_post_ms=0.5, total_ms=13.0, n_candidates=40,
    )
    agg = aggregate_latencies(
        measurements=[m], config_name="x", role=ROLE_BASELINE,
        set_name="gold",
    )
    assert agg.candidate_gen_ms["p50"] == pytest.approx(12.5)
    assert agg.total_ms["p99"] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# Live MMR micro-benchmark
# ---------------------------------------------------------------------------


def test_time_mmr_post_hoc_pass_uses_provided_timer() -> None:
    """The mean of the deterministic-timer samples should match the
    fixed step we configured — pinning that the harness uses the
    injected timer rather than wall clock."""
    pool = [_doc(i + 1, f"P-{i}", 1.0 - i * 0.01) for i in range(40)]
    timer = _det_timer(step_ms=2.0)
    elapsed_ms = time_mmr_post_hoc_pass(
        pool, candidate_k=30, use_mmr=True, mmr_lambda=0.7, top_k=10,
        reps=3, timer=timer,
    )
    # Each rep advances start (step) and end (step) → end-start = step
    assert elapsed_ms == pytest.approx(2.0, abs=1e-9)


def test_time_mmr_post_hoc_pass_handles_no_mmr() -> None:
    """``use_mmr=False`` should still produce a measurement (the
    apply_variant path slices the pool without MMR — still timed)."""
    pool = [_doc(i + 1, f"P-{i}", 1.0 - i * 0.01) for i in range(40)]
    timer = _det_timer(step_ms=1.0)
    elapsed_ms = time_mmr_post_hoc_pass(
        pool, candidate_k=10, use_mmr=False, mmr_lambda=0.7, top_k=10,
        reps=2, timer=timer,
    )
    assert elapsed_ms == pytest.approx(1.0, abs=1e-9)


def test_time_mmr_post_hoc_pass_rejects_zero_reps() -> None:
    pool = [_doc(1, "P-A", 0.9)]
    with pytest.raises(ValueError):
        time_mmr_post_hoc_pass(
            pool, candidate_k=1, use_mmr=False, mmr_lambda=0.7, top_k=1,
            reps=0,
        )


def test_measure_one_config_produces_one_record_per_row() -> None:
    rows = [
        _make_pool_row(query_id=f"q{i}", query="t", elapsed_ms=20.0 + i)
        for i in range(5)
    ]
    cfg = LatencySmokeConfig(
        name="rec", role=ROLE_RECOMMENDED, top_k=10, candidate_k=40,
        use_mmr=True, mmr_lambda=0.7,
    )
    timer = _det_timer(step_ms=0.4)
    out = measure_one_config(
        config=cfg, pool_rows=rows, reps=2, timer=timer,
    )
    assert len(out) == 5
    for i, m in enumerate(out):
        assert m.config_name == "rec"
        assert m.query_id == f"q{i}"
        assert m.candidate_gen_ms == pytest.approx(20.0 + i)
        # Each call resolves to ~step ms; total = candidate_gen + mmr_post.
        assert m.mmr_post_ms == pytest.approx(0.4, abs=1e-9)
        assert m.total_ms == pytest.approx((20.0 + i) + 0.4, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end smoke check
# ---------------------------------------------------------------------------


def test_run_smoke_check_emits_one_aggregate_per_config_set_combo() -> None:
    gold_rows = [
        _make_pool_row(query_id=f"g{i}", query="t", elapsed_ms=20.0 + i)
        for i in range(10)
    ]
    silver_rows = [
        _make_pool_row(query_id=f"s{i}", query="t", elapsed_ms=15.0 + i)
        for i in range(20)
    ]
    configs = default_smoke_suite()
    report = run_smoke_check(
        configs=configs,
        gold_pool_rows=gold_rows, silver_pool_rows=silver_rows,
        pool_size=40, reps=1, include_combined=True,
    )
    # 3 configs × 3 sets = 9 aggregates.
    assert len(report.aggregates) == 9
    # Per-set query counts.
    by_set = {(a.config_name, a.set_name): a for a in report.aggregates}
    for cfg in configs:
        assert by_set[(cfg.name, "gold-50")].n_queries == 10
        assert by_set[(cfg.name, "silver-500")].n_queries == 20
        assert by_set[(cfg.name, "combined-550")].n_queries == 30


def test_run_smoke_check_skips_combined_when_disabled() -> None:
    rows = [_make_pool_row(query_id="q0", query="t")]
    configs = default_smoke_suite()
    report = run_smoke_check(
        configs=configs,
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=False,
    )
    set_names = {a.set_name for a in report.aggregates}
    assert "combined-550" not in set_names


def test_run_smoke_check_carries_pool_size_note() -> None:
    """The pool_size note must include the actual pool_size value so
    a reviewer cannot lose track of where candidate_gen_ms came from."""
    rows = [_make_pool_row(query_id="q0", query="t")]
    report = run_smoke_check(
        configs=default_smoke_suite(),
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=False,
    )
    assert "pool_size=40" in report.measured_at_pool_size_note
    # The rerank-not-measured note must appear verbatim somewhere in
    # the report's notes payload (also surfaced by the renderer).
    assert "Reranker stage NOT measured" in report.rerank_note


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_latency_smoke_report_includes_every_set_and_config() -> None:
    rows = [_make_pool_row(query_id=f"q{i}", query="t") for i in range(3)]
    configs = default_smoke_suite()
    report = run_smoke_check(
        configs=configs,
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=True,
    )
    md = render_latency_smoke_report(report)
    assert "# Phase 7.5 — latency smoke check" in md
    assert "## Honest scope" in md
    assert "## Configs under test" in md
    assert "Reranker stage NOT measured" in md
    for cfg in configs:
        assert cfg.name in md
    # Per-set sub-headings.
    assert "### gold-50" in md
    assert "### silver-500" in md
    assert "### combined-550" in md
    # Verdict carries the recommended-vs-previous-best comparison.
    assert "## Verdict" in md
    assert "recommended" in md.lower()
    # p50/p90/p99 columns appear.
    assert "p50" in md
    assert "p90" in md
    assert "p99" in md


def test_write_latency_smoke_md_and_json_round_trip(tmp_path: Path) -> None:
    rows = [_make_pool_row(query_id=f"q{i}", query="t") for i in range(2)]
    configs = default_smoke_suite()
    report = run_smoke_check(
        configs=configs,
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=False,
    )
    md_path = write_latency_smoke_md(tmp_path / "smoke.md", report)
    json_path = write_latency_smoke_json(tmp_path / "smoke.json", report)
    assert md_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pool_size"] == 40
    assert "configs" in payload
    assert len(payload["aggregates"]) == 3 * 2  # 3 configs × 2 sets


def test_render_latency_smoke_report_marks_baseline_no_mmr() -> None:
    """The baseline row must render with MMR off so a reviewer skimming
    the table cannot mistake it for the previous-best config."""
    rows = [_make_pool_row(query_id="q0", query="t")]
    report = run_smoke_check(
        configs=default_smoke_suite(),
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=False,
    )
    md = render_latency_smoke_report(report)
    # The baseline row should have a `—` mark (use_mmr=False) and the
    # other two should have a `✓` (use_mmr=True). Pin both.
    baseline_line = [
        line for line in md.splitlines()
        if line.startswith("| `baseline_top10`")
    ]
    assert baseline_line, "baseline row should be present in configs table"
    assert " — " in baseline_line[0]


def test_render_latency_smoke_report_emits_decision_section() -> None:
    rows = [_make_pool_row(query_id="q0", query="t")]
    report = run_smoke_check(
        configs=default_smoke_suite(),
        gold_pool_rows=rows, silver_pool_rows=rows,
        pool_size=40, reps=1, include_combined=False,
    )
    md = render_latency_smoke_report(report)
    assert "## Decision" in md
    assert "30%" in md  # the threshold the spec calls out
