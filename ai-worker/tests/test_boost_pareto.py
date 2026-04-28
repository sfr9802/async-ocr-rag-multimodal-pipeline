"""Tests for the Phase 2B boost-aware Pareto frontier merger.

Builds two synthetic ``TopNSweepReport`` instances (Phase 2A baseline
+ Phase 2B boost) and asserts the unified frontier:
  - keeps every entry from both inputs
  - tags each with a track string
  - calls dominance correctly when Phase 2B improves accuracy at
    same latency
  - rejects unknown metric / latency field names
  - markdown renderer produces a header, table, and per-row tag
"""

from __future__ import annotations

import pytest

from eval.harness.boost_pareto import (
    BoostParetoEntry,
    BoostParetoReport,
    boost_pareto_to_dict,
    compute_boost_pareto_frontier,
    render_boost_pareto_markdown,
)
from eval.harness.topn_sweep import TopNSweepEntry, TopNSweepReport


def _entry(label, *, dense_top_n, final_top_k, hit, p95):
    return TopNSweepEntry(
        label=label,
        report_path="<inline>",
        dense_top_n=dense_top_n,
        final_top_k=final_top_k,
        reranker_batch_size=16,
        reranker_model="bge",
        row_count=200,
        rows_with_expected_doc_ids=200,
        mean_hit_at_1=hit,
        mean_hit_at_3=hit,
        mean_hit_at_5=hit,
        mean_mrr_at_10=hit,
        mean_ndcg_at_10=hit,
        candidate_recall={},
        mean_dup_rate=0.0,
        mean_avg_context_token_count=200.0,
        rerank_avg_ms=p95,
        rerank_p50_ms=p95,
        rerank_p90_ms=p95,
        rerank_p95_ms=p95,
        rerank_p99_ms=p95,
        rerank_max_ms=p95,
        rerank_row_count=200,
        total_query_avg_ms=p95 + 5,
        total_query_p50_ms=p95 + 5,
        total_query_p90_ms=p95 + 5,
        total_query_p95_ms=p95 + 5,
        total_query_p99_ms=p95 + 5,
        total_query_max_ms=p95 + 5,
        total_query_row_count=200,
        dense_retrieval_avg_ms=10.0,
        dense_retrieval_p50_ms=10.0,
        dense_retrieval_p95_ms=10.0,
        dense_retrieval_row_count=200,
    )


def _phase2a_sweep():
    return TopNSweepReport(
        schema="phase2a-topn-sweep.v1",
        entries=[
            _entry("top5", dense_top_n=5, final_top_k=5, hit=0.50, p95=160.0),
            _entry("top10", dense_top_n=10, final_top_k=10, hit=0.60, p95=170.0),
            _entry("top15", dense_top_n=15, final_top_k=15, hit=0.62, p95=180.0),
        ],
    )


def _phase2b_sweep():
    return TopNSweepReport(
        schema="phase2b-boost-topn-sweep.v1",
        entries=[
            _entry("top5", dense_top_n=5, final_top_k=5, hit=0.55, p95=160.5),
            _entry("top10", dense_top_n=10, final_top_k=10, hit=0.65, p95=170.5),
            _entry("top15", dense_top_n=15, final_top_k=15, hit=0.66, p95=180.5),
        ],
    )


class TestComputeBoostPareto:
    def test_all_entries_present_with_track_tag(self):
        report = compute_boost_pareto_frontier(
            phase2a_sweep=_phase2a_sweep(),
            phase2b_sweep=_phase2b_sweep(),
            metric="mean_hit_at_1",
            latency="rerank_p95_ms",
        )
        tracks = sorted({(e.track, e.label) for e in report.entries})
        assert ("phase2a", "top5") in tracks
        assert ("phase2b", "top10") in tracks

    def test_phase2b_dominates_at_same_latency_band(self):
        report = compute_boost_pareto_frontier(
            phase2a_sweep=_phase2a_sweep(),
            phase2b_sweep=_phase2b_sweep(),
            metric="mean_hit_at_1",
            latency="rerank_p95_ms",
        )
        # Phase 2B's top10 has higher hit@1 (0.65 vs 0.60) at slightly
        # higher latency (170.5 vs 170.0). 2A's top10 (0.60 @ 170) is
        # dominated by 2B's top10 since 2B has both higher hit AND
        # lower latency than 2A's top15 (0.62 @ 180), which means in
        # the merged set 2A top10 may be dominated by 2B top10.
        # We only check that 2B's strongest point is on the frontier.
        on_frontier = {(e.track, e.label) for e in report.entries if e.on_frontier}
        assert ("phase2b", "top15") in on_frontier
        assert ("phase2b", "top5") in on_frontier

    def test_label_collision_disambiguation(self):
        # Both sweeps have "top10" — namespacing should keep them
        # distinct; merged report has both.
        report = compute_boost_pareto_frontier(
            phase2a_sweep=_phase2a_sweep(),
            phase2b_sweep=_phase2b_sweep(),
        )
        labels = sorted((e.track, e.label) for e in report.entries)
        # 6 entries (3 phase2a + 3 phase2b).
        assert len(labels) == 6

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            compute_boost_pareto_frontier(
                phase2a_sweep=_phase2a_sweep(),
                phase2b_sweep=_phase2b_sweep(),
                metric="not_a_metric",
            )

    def test_invalid_latency_raises(self):
        with pytest.raises(ValueError):
            compute_boost_pareto_frontier(
                phase2a_sweep=_phase2a_sweep(),
                phase2b_sweep=_phase2b_sweep(),
                latency="not_a_latency",
            )


class TestSerialization:
    def test_to_dict_carries_schema_and_entries(self):
        report = compute_boost_pareto_frontier(
            phase2a_sweep=_phase2a_sweep(),
            phase2b_sweep=_phase2b_sweep(),
        )
        out = boost_pareto_to_dict(report)
        assert out["schema"] == "phase2b-boost-pareto-frontier.v1"
        assert out["metric_field"] == "mean_hit_at_1"
        assert len(out["entries"]) == 6

    def test_markdown_includes_track_column(self):
        report = compute_boost_pareto_frontier(
            phase2a_sweep=_phase2a_sweep(),
            phase2b_sweep=_phase2b_sweep(),
        )
        md = render_boost_pareto_markdown(report)
        assert "Phase 2B boost Pareto frontier" in md
        assert "phase2a" in md
        assert "phase2b" in md
