"""Phase 2A-L latency-breakdown / topN sweep / Pareto / modes tests.

Six blocks of coverage, all fully offline (no torch / CUDA / FAISS):

  1. ``compute_latency_stats`` percentile + aggregation contract
     - Empty input → all-None stats record with count=0.
     - Known input → exact avg / p50 / p90 / p95 / p99 / max values.
     - Non-numeric values are skipped silently.

  2. ``build_latency_breakdown`` round-trips a retrieval report:
     - Reads ``rerank_breakdown_ms`` per row, aggregates per stage.
     - Surfaces ``dense_retrieval_ms`` + ``total_query_ms`` separately.
     - Missing-data path: a report without breakdowns yields an empty
       stages dict but still carries valid provenance.

  3. ``build_topn_sweep`` produces a stable table from N retrieval
     reports — including a candidate-recall sibling that backfills
     ``candidate_recall@N`` onto every entry.

  4. ``compute_pareto_frontier`` — dominance contract:
     - Better metric AND lower latency dominates.
     - Tie-on-both keeps both points on the frontier.
     - Missing-data points get the (missing-data) sentinel.

  5. ``recommend_modes`` — fast / balanced / quality picks under
     several budget configurations, including the empty-frontier fallback.

  6. CrossEncoderReranker.collect_stage_timings hook — exercises a
     stand-in CrossEncoder that has tokenizer / model / activation_fn /
     config attributes so the breakdown round-trips through rerank()
     without importing torch in this process. The non-instrumented
     default path still works the same way (existing tests pin that).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from eval.harness.latency_breakdown import (
    KNOWN_STAGES,
    SCHEMA_VERSION as LATENCY_SCHEMA_VERSION,
    LatencyBreakdownReport,
    LatencyStats,
    build_latency_breakdown,
    compute_latency_stats,
    latency_breakdown_to_dict,
    render_latency_breakdown_markdown,
)
from eval.harness.pareto_frontier import (
    ACCURACY_METRICS,
    LATENCY_METRICS,
    SCHEMA_VERSION as PARETO_SCHEMA_VERSION,
    compute_pareto_frontier,
    pareto_to_dict,
    render_pareto_markdown,
)
from eval.harness.recommended_modes import (
    SCHEMA_VERSION as MODES_SCHEMA_VERSION,
    TIER_ORDER,
    recommend_modes,
    recommended_modes_to_dict,
    render_recommended_modes_markdown,
)
from eval.harness.topn_sweep import (
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
    TopNSweepEntry,
    TopNSweepReport,
    build_topn_sweep,
    render_topn_sweep_markdown,
    topn_sweep_to_dict,
)


# ---------------------------------------------------------------------------
# 1. compute_latency_stats / percentile contract.
# ---------------------------------------------------------------------------


def test_compute_latency_stats_empty_returns_all_none_with_count_zero():
    stats = compute_latency_stats("forward_ms", [])
    assert stats.stage == "forward_ms"
    assert stats.count == 0
    assert stats.avg_ms is None
    assert stats.p50_ms is None
    assert stats.p90_ms is None
    assert stats.p95_ms is None
    assert stats.p99_ms is None
    assert stats.max_ms is None


def test_compute_latency_stats_known_values_exact():
    """Pin the percentile contract numerically against a tractable input.

    Values 1..100. The aggregator uses ``statistics.median`` for p50
    (matches the rest of the retrieval-eval harness — consistent with
    ``_p50_or_zero``) and nearest-rank for p90 / p95 / p99 / max.

      - mean = 50.5
      - median over an even-length list = (50 + 51) / 2 = 50.5
      - p90 nearest-rank index = ceil(0.90 * 100) - 1 = 89 → value 90
      - p95 nearest-rank → value 95
      - p99 nearest-rank → value 99
      - max → 100
    """
    values = [float(i) for i in range(1, 101)]
    stats = compute_latency_stats("forward_ms", values)
    assert stats.count == 100
    assert stats.avg_ms == 50.5
    assert stats.p50_ms == 50.5
    assert stats.p90_ms == 90
    assert stats.p95_ms == 95
    assert stats.p99_ms == 99
    assert stats.max_ms == 100


def test_compute_latency_stats_skips_non_numeric_values():
    stats = compute_latency_stats(
        "tokenize_ms", [10.0, "bad", None, 20.0, float("nan")],
    )
    # NaN is float → kept (caller's choice); strings/Nones are dropped.
    # Sanity: avg over (10, 20, nan). NaN propagates through fmean,
    # so we don't assert on avg here — assert on count instead.
    assert stats.count == 3
    assert stats.max_ms == 20.0 or (
        stats.max_ms is not None and math.isnan(stats.max_ms)
    )


def test_compute_latency_stats_single_value_collapses_to_value():
    stats = compute_latency_stats("postprocess_ms", [42.5])
    assert stats.count == 1
    assert stats.avg_ms == 42.5
    assert stats.p50_ms == 42.5
    assert stats.p95_ms == 42.5
    assert stats.p99_ms == 42.5
    assert stats.max_ms == 42.5


# ---------------------------------------------------------------------------
# 2. build_latency_breakdown round-trip.
# ---------------------------------------------------------------------------


def _write_retrieval_report(
    tmp_path: Path,
    *,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
    summary: Dict[str, Any] | None = None,
) -> Path:
    payload = {
        "metadata": metadata or {
            "corpus_path": "eval/corpora/x/corpus.jsonl",
            "dense_top_n": 20,
            "final_top_k": 10,
            "reranker_batch_size": 16,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        },
        "summary": summary or {
            "corpus_path": "eval/corpora/x/corpus.jsonl",
            "reranker_name": "cross-encoder:test",
            "top_k": 10,
        },
        "rows": rows,
    }
    out = tmp_path / "retrieval_eval_report.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_build_latency_breakdown_aggregates_per_stage(tmp_path):
    rows = [
        {
            "id": "q1",
            "retrieval_ms": 700.0,
            "dense_retrieval_ms": 15.0,
            "rerank_breakdown_ms": {
                "pair_build_ms": 0.5,
                "tokenize_ms": 10.0,
                "forward_ms": 600.0,
                "postprocess_ms": 0.2,
                "total_rerank_ms": 610.7,
            },
        },
        {
            "id": "q2",
            "retrieval_ms": 720.0,
            "dense_retrieval_ms": 18.0,
            "rerank_breakdown_ms": {
                "pair_build_ms": 0.6,
                "tokenize_ms": 12.0,
                "forward_ms": 620.0,
                "postprocess_ms": 0.3,
                "total_rerank_ms": 632.9,
            },
        },
    ]
    report_path = _write_retrieval_report(tmp_path, rows=rows)
    report = build_latency_breakdown(report_path, label="rerank-top20")

    assert report.schema == LATENCY_SCHEMA_VERSION
    assert report.label == "rerank-top20"
    assert report.row_count == 2
    assert report.rows_with_rerank_breakdown == 2
    assert report.rows_with_dense_retrieval_ms == 2
    # All five rerank stages plus dense + total query.
    expected_stages = {
        "dense_retrieval_ms",
        "pair_build_ms",
        "tokenize_ms",
        "forward_ms",
        "postprocess_ms",
        "total_rerank_ms",
        "total_query_ms",
    }
    assert expected_stages.issubset(set(report.stages.keys()))
    forward = report.stages["forward_ms"]
    assert forward.count == 2
    assert forward.avg_ms == pytest.approx(610.0)
    assert forward.max_ms == pytest.approx(620.0)
    total_query = report.stages["total_query_ms"]
    assert total_query.count == 2
    assert total_query.avg_ms == pytest.approx(710.0)


def test_build_latency_breakdown_handles_missing_breakdowns(tmp_path):
    """Report without rerank_breakdown_ms → empty stages for those keys
    but dense + total_query still populated when the rows have them."""
    rows = [
        {"id": "q1", "retrieval_ms": 30.0, "dense_retrieval_ms": 28.0},
        {"id": "q2", "retrieval_ms": 32.0, "dense_retrieval_ms": 30.0},
    ]
    report_path = _write_retrieval_report(tmp_path, rows=rows)
    report = build_latency_breakdown(report_path)

    assert report.rows_with_rerank_breakdown == 0
    assert report.rows_with_dense_retrieval_ms == 2
    # Stages present: dense + total_query only; no rerank stages.
    assert "dense_retrieval_ms" in report.stages
    assert "total_query_ms" in report.stages
    for s in (
        "pair_build_ms",
        "tokenize_ms",
        "forward_ms",
        "postprocess_ms",
        "total_rerank_ms",
    ):
        assert s not in report.stages


def test_build_latency_breakdown_skips_error_rows(tmp_path):
    rows = [
        {
            "id": "q1",
            "retrieval_ms": 700.0,
            "dense_retrieval_ms": 15.0,
            "rerank_breakdown_ms": {
                "pair_build_ms": 0.5,
                "tokenize_ms": 10.0,
                "forward_ms": 600.0,
                "postprocess_ms": 0.2,
                "total_rerank_ms": 610.7,
            },
        },
        {"id": "q2", "retrieval_ms": 9999.0, "error": "boom"},
    ]
    report_path = _write_retrieval_report(tmp_path, rows=rows)
    report = build_latency_breakdown(report_path)
    # The error row is excluded everywhere; total_query / dense / rerank
    # all see exactly one valid row.
    assert report.row_count == 1
    assert report.stages["forward_ms"].count == 1
    assert report.stages["total_query_ms"].count == 1


def test_latency_breakdown_to_dict_orders_known_stages_first(tmp_path):
    rows = [
        {
            "id": "q1",
            "retrieval_ms": 100.0,
            "dense_retrieval_ms": 10.0,
            "rerank_breakdown_ms": {
                "pair_build_ms": 1.0,
                "tokenize_ms": 2.0,
                "forward_ms": 80.0,
                "postprocess_ms": 0.5,
                "total_rerank_ms": 83.5,
                "ZZZ_custom_stage": 0.1,  # alphabetic sort tail
            },
        },
    ]
    report_path = _write_retrieval_report(tmp_path, rows=rows)
    report = build_latency_breakdown(report_path)
    raw = latency_breakdown_to_dict(report)
    keys = list(raw["stages"].keys())
    # KNOWN_STAGES intersection appears first in pipeline order; then the
    # custom stage at the bottom.
    known = [s for s in KNOWN_STAGES if s in keys]
    custom = [s for s in keys if s not in KNOWN_STAGES]
    assert keys == known + custom
    assert custom == ["ZZZ_custom_stage"]


def test_render_latency_breakdown_markdown_smoke(tmp_path):
    rows = [
        {
            "id": "q1",
            "retrieval_ms": 700.0,
            "dense_retrieval_ms": 15.0,
            "rerank_breakdown_ms": {
                "pair_build_ms": 0.5,
                "tokenize_ms": 10.0,
                "forward_ms": 600.0,
                "postprocess_ms": 0.2,
                "total_rerank_ms": 610.7,
            },
        },
    ]
    report_path = _write_retrieval_report(tmp_path, rows=rows)
    md = render_latency_breakdown_markdown(
        build_latency_breakdown(report_path, label="anchor-top20")
    )
    assert "Phase 2A-L reranker latency breakdown" in md
    assert "anchor-top20" in md
    assert "forward_ms" in md
    assert "dense_retrieval_ms" in md


# ---------------------------------------------------------------------------
# 3. build_topn_sweep aggregation.
# ---------------------------------------------------------------------------


def _make_sweep_payload(
    *,
    dense_top_n: int,
    hit_at_1: float,
    rerank_p95_ms: float,
    total_p95_ms: float,
    extra_hits: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    # Synthesize 5 rows so the sweep aggregator can compute sane
    # percentiles. retrieval_ms = dense + rerank approx; we hard-set
    # the headline rerank_p95_ms via summary so the recompute matches.
    for i in range(5):
        rows.append({
            "id": f"r-{i}",
            "retrieval_ms": float(total_p95_ms),
            "dense_retrieval_ms": float(total_p95_ms - rerank_p95_ms),
            "rerank_breakdown_ms": {
                "pair_build_ms": 0.5,
                "tokenize_ms": 10.0,
                "forward_ms": rerank_p95_ms - 11.0,
                "postprocess_ms": 0.5,
                "total_rerank_ms": rerank_p95_ms,
            },
        })
    return {
        "metadata": {
            "dense_top_n": dense_top_n,
            "final_top_k": 10,
            "reranker_batch_size": 16,
            "reranker_model": "test",
            "corpus_path": "eval/corpora/x/corpus.jsonl",
        },
        "summary": {
            "row_count": len(rows),
            "rows_with_expected_doc_ids": len(rows),
            "top_k": 10,
            "mean_hit_at_1": hit_at_1,
            "mean_hit_at_3": hit_at_1 + 0.05,
            "mean_hit_at_5": hit_at_1 + 0.10,
            "mean_mrr_at_10": hit_at_1 + 0.02,
            "mean_ndcg_at_10": hit_at_1 + 0.04,
            "mean_dup_rate": 0.20,
            "mean_avg_context_token_count": 290.0,
            "mean_extra_hits": extra_hits or {},
            "mean_rerank_ms": rerank_p95_ms - 5.0,
            "p50_rerank_ms": rerank_p95_ms - 8.0,
            "p90_rerank_ms": rerank_p95_ms - 1.0,
            "p95_rerank_ms": rerank_p95_ms,
            "p99_rerank_ms": rerank_p95_ms + 2.0,
            "max_rerank_ms": rerank_p95_ms + 5.0,
            "rerank_row_count": len(rows),
            "reranker_name": "cross-encoder:test",
            "embedding_model": "BAAI/bge-m3",
            "index_version": "offline-x",
            "corpus_path": "eval/corpora/x/corpus.jsonl",
        },
        "rows": rows,
    }


def test_build_topn_sweep_orders_by_dense_top_n_and_aggregates(tmp_path):
    paths = []
    for n, h1, r95, t95 in [
        (20, 0.605, 706.0, 723.0),
        (5,  0.580, 200.0, 220.0),
        (50, 0.615, 1840.0, 1855.0),
    ]:
        p = tmp_path / f"top{n}.json"
        p.write_text(
            json.dumps(_make_sweep_payload(
                dense_top_n=n,
                hit_at_1=h1,
                rerank_p95_ms=r95,
                total_p95_ms=t95,
            )),
            encoding="utf-8",
        )
        paths.append((f"top{n}", p))

    sweep = build_topn_sweep(paths)
    assert sweep.schema == SWEEP_SCHEMA_VERSION
    # Sorted ascending by dense_top_n.
    assert [e.dense_top_n for e in sweep.entries] == [5, 20, 50]
    # Latency series surfaced from both the summary and recomputed rows.
    top20 = sweep.entries[1]
    assert top20.rerank_p95_ms == 706.0
    assert top20.rerank_p50_ms == 698.0
    assert top20.total_query_p95_ms is not None
    assert top20.total_query_row_count == 5


def test_build_topn_sweep_inherits_candidate_recall_from_sibling(tmp_path):
    candidate_recall = tmp_path / "cr.json"
    candidate_recall.write_text(
        json.dumps(_make_sweep_payload(
            dense_top_n=50,
            hit_at_1=0.54,
            rerank_p95_ms=20.0,
            total_p95_ms=21.0,
            extra_hits={"10": 0.715, "20": 0.770, "50": 0.800},
        )),
        encoding="utf-8",
    )

    rerank_paths = []
    for n in (10, 20, 50):
        p = tmp_path / f"top{n}.json"
        p.write_text(
            json.dumps(_make_sweep_payload(
                dense_top_n=n,
                hit_at_1=0.59,
                rerank_p95_ms=300.0,
                total_p95_ms=320.0,
            )),
            encoding="utf-8",
        )
        rerank_paths.append((f"top{n}", p))

    sweep = build_topn_sweep(
        rerank_paths, candidate_recall_path=candidate_recall,
    )
    for e in sweep.entries:
        assert e.candidate_recall.get("10") == 0.715
        assert e.candidate_recall.get("20") == 0.770
        assert e.candidate_recall.get("50") == 0.800


def test_build_topn_sweep_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_topn_sweep([("topX", tmp_path / "nonexistent.json")])


def test_topn_sweep_to_dict_round_trips_through_recommended_modes(tmp_path):
    """Pin that the JSON shape ``topn_sweep_to_dict`` emits is consumable
    by ``_run_phase2a_recommended_modes_cli`` field-by-field
    re-hydration. The fields the dataclass declares MUST be on the
    dict; extra fields would be ignored but never trip the loader."""
    paths = []
    for n in (10, 20):
        p = tmp_path / f"top{n}.json"
        p.write_text(
            json.dumps(_make_sweep_payload(
                dense_top_n=n, hit_at_1=0.6,
                rerank_p95_ms=300.0, total_p95_ms=320.0,
            )),
            encoding="utf-8",
        )
        paths.append((f"top{n}", p))
    sweep = build_topn_sweep(paths)
    raw = topn_sweep_to_dict(sweep)
    declared = set(TopNSweepEntry.__dataclass_fields__.keys())
    for entry in raw["entries"]:
        # Every declared field must be present on disk.
        assert declared.issubset(set(entry.keys())), (
            f"missing fields: {declared - set(entry.keys())}"
        )


def test_render_topn_sweep_markdown_includes_each_section(tmp_path):
    paths = []
    for n in (10, 50):
        p = tmp_path / f"top{n}.json"
        p.write_text(
            json.dumps(_make_sweep_payload(
                dense_top_n=n, hit_at_1=0.55,
                rerank_p95_ms=300.0, total_p95_ms=320.0,
                extra_hits={"50": 0.80} if n == 50 else None,
            )),
            encoding="utf-8",
        )
        paths.append((f"top{n}", p))
    md = render_topn_sweep_markdown(build_topn_sweep(paths))
    assert "Headline accuracy" in md
    assert "Rerank latency" in md
    assert "Total query latency" in md
    assert "hit@50" in md  # candidate-recall section appeared


# ---------------------------------------------------------------------------
# 4. compute_pareto_frontier dominance contract.
# ---------------------------------------------------------------------------


def _entry(
    *,
    label: str,
    n: int,
    hit_at_1: float,
    rerank_p95: float,
    total_p95: float | None = None,
) -> TopNSweepEntry:
    return TopNSweepEntry(
        label=label,
        report_path=f"/tmp/{label}.json",
        dense_top_n=n,
        final_top_k=10,
        reranker_batch_size=16,
        reranker_model="test",
        row_count=5,
        rows_with_expected_doc_ids=5,
        mean_hit_at_1=hit_at_1,
        mean_hit_at_3=hit_at_1 + 0.05,
        mean_hit_at_5=hit_at_1 + 0.10,
        mean_mrr_at_10=hit_at_1 + 0.02,
        mean_ndcg_at_10=hit_at_1 + 0.04,
        rerank_p95_ms=rerank_p95,
        rerank_p99_ms=rerank_p95 + 5,
        total_query_p95_ms=total_p95 if total_p95 is not None else rerank_p95 + 20.0,
        total_query_p99_ms=(total_p95 or rerank_p95 + 20.0) + 10.0,
        rerank_row_count=5,
        total_query_row_count=5,
    )


def _make_sweep(entries: List[TopNSweepEntry]) -> TopNSweepReport:
    return TopNSweepReport(
        schema=SWEEP_SCHEMA_VERSION, entries=entries, caveats=[],
    )


def test_pareto_frontier_dominance_basic():
    """top20 dominates top30 (higher hit@1, lower latency)."""
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
        # top30 strictly dominated by top20 (lower metric AND higher latency).
        _entry(label="top30", n=30, hit_at_1=0.62, rerank_p95=1100.0),
        _entry(label="top50", n=50, hit_at_1=0.70, rerank_p95=1840.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    assert frontier.schema == PARETO_SCHEMA_VERSION

    on_frontier_labels = {p.label for p in frontier.entries if p.on_frontier}
    assert on_frontier_labels == {"top10", "top20", "top50"}
    dominated = [p for p in frontier.entries if not p.on_frontier]
    assert len(dominated) == 1
    assert dominated[0].label == "top30"
    assert dominated[0].dominated_by is not None
    assert "top20" in (dominated[0].dominated_by or "")


def test_pareto_frontier_tie_keeps_both_points():
    """Two points with identical (metric, latency) are mutually
    undominated; both stay on the frontier."""
    entries = [
        _entry(label="A", n=10, hit_at_1=0.60, rerank_p95=300.0),
        _entry(label="B", n=10, hit_at_1=0.60, rerank_p95=300.0),
    ]
    frontier = compute_pareto_frontier(_make_sweep(entries))
    assert all(p.on_frontier for p in frontier.entries)


def test_pareto_frontier_missing_data_is_excluded():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        TopNSweepEntry(
            label="malformed",
            report_path="/tmp/x.json",
            dense_top_n=20,
            final_top_k=10,
            reranker_batch_size=None,
            reranker_model=None,
            row_count=0,
            rows_with_expected_doc_ids=0,
            mean_hit_at_1=None,  # no metric
            mean_hit_at_3=None,
            mean_hit_at_5=None,
            mean_mrr_at_10=None,
            mean_ndcg_at_10=None,
            rerank_p95_ms=None,
        ),
    ]
    frontier = compute_pareto_frontier(_make_sweep(entries))
    by_label = {p.label: p for p in frontier.entries}
    assert by_label["malformed"].on_frontier is False
    assert by_label["malformed"].dominated_by == "(missing-data)"
    assert by_label["top10"].on_frontier is True


def test_pareto_frontier_rejects_unknown_metric_or_latency():
    sweep = _make_sweep([_entry(label="A", n=10, hit_at_1=0.6, rerank_p95=300.0)])
    with pytest.raises(ValueError):
        compute_pareto_frontier(sweep, metric="bogus_metric")
    with pytest.raises(ValueError):
        compute_pareto_frontier(sweep, latency="bogus_latency")


def test_pareto_frontier_field_lists_are_consistent():
    """``ACCURACY_METRICS`` and ``LATENCY_METRICS`` must reference fields
    that exist on ``TopNSweepEntry`` — otherwise the Pareto helper
    silently treats them as missing-data."""
    declared = set(TopNSweepEntry.__dataclass_fields__.keys())
    for m in ACCURACY_METRICS:
        assert m in declared
    for l in LATENCY_METRICS:
        assert l in declared


def test_pareto_frontier_markdown_smoke():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top50", n=50, hit_at_1=0.70, rerank_p95=1840.0),
    ]
    md = render_pareto_markdown(
        compute_pareto_frontier(_make_sweep(entries))
    )
    assert "Frontier" in md
    assert "top10" in md and "top50" in md


# ---------------------------------------------------------------------------
# 5. recommend_modes.
# ---------------------------------------------------------------------------


def test_recommend_modes_picks_fast_balanced_quality():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
        _entry(label="top50", n=50, hit_at_1=0.70, rerank_p95=1840.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    modes = recommend_modes(sweep, frontier)
    assert modes.schema == MODES_SCHEMA_VERSION
    by_tier = {m.tier: m for m in modes.modes}
    assert set(by_tier.keys()) == set(TIER_ORDER)
    assert by_tier["fast"].label == "top10"
    assert by_tier["fast"].dense_top_n == 10
    assert by_tier["quality"].label == "top50"
    assert by_tier["quality"].dense_top_n == 50
    # Balanced uses median-rank by default → top20.
    assert by_tier["balanced"].label == "top20"


def test_recommend_modes_balanced_budget_picks_highest_metric_under_budget():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
        _entry(label="top50", n=50, hit_at_1=0.70, rerank_p95=1840.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)

    modes_750 = recommend_modes(sweep, frontier, balanced_p95_budget_ms=750.0)
    by_tier_750 = {m.tier: m for m in modes_750.modes}
    # top20 fits the 750 ms budget; top50 doesn't. balanced picks
    # top20 because it has the highest hit@1 within budget.
    assert by_tier_750["balanced"].label == "top20"

    # Now tighten the budget below the lowest-latency frontier point —
    # balanced falls back to the lowest-latency entry with a budget-
    # exceeded note.
    modes_100 = recommend_modes(sweep, frontier, balanced_p95_budget_ms=100.0)
    by_tier_100 = {m.tier: m for m in modes_100.modes}
    assert by_tier_100["balanced"].label == "top10"
    assert any("exceeds" in n for n in by_tier_100["balanced"].notes)


def test_recommend_modes_fast_budget_prefers_high_metric_within_budget():
    entries = [
        _entry(label="top5",  n=5,  hit_at_1=0.40, rerank_p95=150.0),
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    # 350 ms budget should allow top10 (300 ms) and top5 (150 ms);
    # the helper picks top10 because hit@1 is higher.
    modes = recommend_modes(sweep, frontier, fast_p95_budget_ms=350.0)
    by_tier = {m.tier: m for m in modes.modes}
    assert by_tier["fast"].label == "top10"


def test_recommend_modes_quality_target_emits_note_when_below():
    entries = [
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    modes = recommend_modes(
        sweep, frontier, quality_target_metric=0.80,
    )
    by_tier = {m.tier: m for m in modes.modes}
    quality = by_tier["quality"]
    assert quality.label == "top20"
    assert any("below" in n for n in quality.notes)


def test_recommend_modes_empty_frontier_falls_back_to_unselected():
    sweep = _make_sweep([])
    frontier = compute_pareto_frontier(sweep)
    modes = recommend_modes(sweep, frontier)
    by_tier = {m.tier: m for m in modes.modes}
    for tier in TIER_ORDER:
        assert by_tier[tier].label is None
        assert "no Pareto-frontier" in by_tier[tier].rationale


def test_recommended_modes_to_dict_pins_tier_order():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top20", n=20, hit_at_1=0.65, rerank_p95=700.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    modes = recommend_modes(sweep, frontier)
    raw = recommended_modes_to_dict(modes)
    tiers = [m["tier"] for m in raw["modes"]]
    assert tiers == list(TIER_ORDER)


def test_render_recommended_modes_markdown_smoke():
    entries = [
        _entry(label="top10", n=10, hit_at_1=0.55, rerank_p95=300.0),
        _entry(label="top50", n=50, hit_at_1=0.70, rerank_p95=1840.0),
    ]
    sweep = _make_sweep(entries)
    frontier = compute_pareto_frontier(sweep)
    md = render_recommended_modes_markdown(
        recommend_modes(sweep, frontier)
    )
    assert "fast" in md and "balanced" in md and "quality" in md


# ---------------------------------------------------------------------------
# 6. CrossEncoderReranker.collect_stage_timings hook (no torch required).
# ---------------------------------------------------------------------------


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, batch, **_kwargs):
        # Return a dict-shaped object with a ``.to(device)`` no-op so the
        # instrumented predict can pretend to host→device-transfer.
        return _FakeBatchEncoding({
            "input_ids": list(range(len(batch))),
        })


@dataclass
class _FakeLogits:
    """Stand-in for a torch tensor with ``ndim`` and basic numpy bridging."""
    values: List[float]

    def cpu(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self.values, dtype=float)


class _FakeWrappedTensor:
    """Wrap a float in a tensor-like with a ``[0]`` accessor."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __getitem__(self, idx):
        return _FakeWrappedTensor(self.value) if idx == 0 else self

    def cpu(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self.value, dtype=float)


@dataclass
class _FakeModelPredictions:
    logits: Any


class _FakeDevice:
    def __repr__(self) -> str:
        return "cpu"


class _FakeModel:
    device = _FakeDevice()

    def __call__(self, *, return_dict, **features):
        # Return one logit per "input_ids" entry so the instrumented
        # predict's loop produces the expected score count.
        ids = features.get("input_ids") or []
        # Score by index so the test can assert on ordering.
        return _FakeModelPredictions(
            logits=_FakeWrappedTensorRow(
                [float(0.1 * i) for i in range(len(ids))]
            )
        )


class _FakeWrappedTensorRow:
    """A list-of-tensors stand-in that supports iteration into [0] entries."""

    def __init__(self, values: List[float]) -> None:
        self._values = list(values)

    def __iter__(self):
        for v in self._values:
            yield _FakeWrappedTensor(v)


class _FakeConfig:
    num_labels = 1


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()
        self.config = _FakeConfig()

    def eval(self):
        return self

    def activation_fn(self, x):
        return x


def _patch_loader_for_breakdown(monkeypatch, fake) -> None:
    from app.capabilities.rag import reranker as reranker_module

    reranker_module._load_cross_encoder.cache_clear()
    monkeypatch.setattr(
        reranker_module,
        "_load_cross_encoder",
        lambda model, max_length, device: fake,
    )
    # Block torch import inside the breakdown predict so the test
    # process never triggers a torch CUDA init.
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch blocked in unit test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)


@pytest.fixture(autouse=True)
def _no_cuda_log(monkeypatch):
    """Mirror the production stub from ``test_phase2a_reranker.py`` so
    the reranker's CUDA helpers don't trigger a torch import here."""
    from app.capabilities.rag import (
        embeddings as embeddings_module,
        reranker as reranker_module,
    )

    monkeypatch.setattr(reranker_module, "_log_cuda_memory", lambda _label: None)
    monkeypatch.setattr(embeddings_module, "_log_cuda_memory", lambda _label: None)


def test_cross_encoder_reranker_breakdown_round_trips(monkeypatch):
    from app.capabilities.rag.generation import RetrievedChunk
    from app.capabilities.rag.reranker import CrossEncoderReranker

    fake = _FakeCrossEncoder()
    _patch_loader_for_breakdown(monkeypatch, fake)

    reranker = CrossEncoderReranker(
        batch_size=2,
        text_max_chars=20,
        device="cpu",
        collect_stage_timings=True,
    )
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id=f"d{i}", section="s",
            text=f"passage {i}", score=0.5,
        )
        for i in range(3)
    ]
    out = reranker.rerank("q", chunks, k=2)
    assert len(out) == 2
    assert reranker.last_breakdown_ms is not None
    bd = reranker.last_breakdown_ms
    for stage in (
        "pair_build_ms",
        "tokenize_ms",
        "forward_ms",
        "postprocess_ms",
        "total_rerank_ms",
    ):
        assert stage in bd
        assert bd[stage] >= 0.0
    # total_rerank_ms is the sum of the four explicit stages.
    explicit_sum = (
        bd["pair_build_ms"]
        + bd["tokenize_ms"]
        + bd["forward_ms"]
        + bd["postprocess_ms"]
    )
    assert abs(bd["total_rerank_ms"] - explicit_sum) < 0.005


def test_cross_encoder_reranker_breakdown_default_off_returns_none(monkeypatch):
    from app.capabilities.rag.generation import RetrievedChunk
    from app.capabilities.rag.reranker import CrossEncoderReranker

    fake = _FakeCrossEncoder()
    _patch_loader_for_breakdown(monkeypatch, fake)

    reranker = CrossEncoderReranker(
        batch_size=2,
        device="cpu",
        # collect_stage_timings defaults to False — no breakdown is
        # captured even after a successful rerank().
    )
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id=f"d{i}", section="s",
            text=f"passage {i}", score=0.5,
        )
        for i in range(3)
    ]
    # Patch ``_predict`` to bypass the real upstream CrossEncoder.predict
    # (which our fake doesn't implement).
    monkeypatch.setattr(
        reranker,
        "_predict",
        lambda enc, pairs, bs: [0.1 * i for i in range(len(pairs))],
    )
    out = reranker.rerank("q", chunks, k=2)
    assert len(out) == 2
    assert reranker.last_breakdown_ms is None


# ---------------------------------------------------------------------------
# 7. retrieval_eval round-trip with new fields.
# ---------------------------------------------------------------------------


def test_retrieval_eval_aggregates_dense_retrieval_ms_and_breakdown():
    """The harness must roll up dense_retrieval_ms + per-stage breakdown
    onto the summary so downstream analysis can read off a single
    document. NoOp paths still leave the rerank breakdown empty."""
    from app.capabilities.rag.generation import RetrievedChunk
    from eval.harness.retrieval_eval import run_retrieval_eval

    class _StubRetriever:
        def __init__(self):
            self._calls = 0

        def retrieve(self, query):
            self._calls += 1
            return _StubReport(self._calls)

    class _StubReport:
        def __init__(self, idx):
            self.results = [
                RetrievedChunk(
                    chunk_id=f"c{idx}", doc_id=f"doc-{idx}", section="s",
                    text="x", score=0.9,
                ),
            ]
            self.rerank_ms = 600.0 + idx * 5.0
            self.dense_retrieval_ms = 15.0 + idx * 1.0
            self.rerank_breakdown_ms = {
                "pair_build_ms": 0.5,
                "tokenize_ms": 10.0,
                "forward_ms": 580.0 + idx * 5.0,
                "postprocess_ms": 0.5,
                "total_rerank_ms": 591.0 + idx * 5.0,
            }
            self.reranker_name = "cross-encoder:test"
            self.index_version = "v"
            self.embedding_model = "m"
            self.candidate_k = 20
            self.use_mmr = False
            self.mmr_lambda = None
            self.dup_rate = 0.0

    dataset = [
        {"id": "r1", "query": "q1", "expected_doc_ids": ["doc-1"]},
        {"id": "r2", "query": "q2", "expected_doc_ids": ["doc-2"]},
    ]
    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=_StubRetriever(), top_k=5, mrr_k=10, ndcg_k=10,
    )

    assert summary.dense_retrieval_row_count == 2
    assert summary.mean_dense_retrieval_ms == pytest.approx(16.5)
    # p99 / p90 series populated on the headline retrieval/rerank.
    assert summary.p99_retrieval_ms is not None
    assert summary.p90_rerank_ms is not None
    # Breakdown stats — five stages each with count=2.
    stages = summary.rerank_breakdown_stats
    assert set(stages.keys()) == {
        "pair_build_ms",
        "tokenize_ms",
        "forward_ms",
        "postprocess_ms",
        "total_rerank_ms",
    }
    forward = stages["forward_ms"]
    assert int(forward.get("count") or 0) == 2
    assert forward["avg"] == pytest.approx(587.5, rel=1e-4)
    # Rows surface the breakdown dict verbatim.
    for r in rows:
        assert r.dense_retrieval_ms is not None
        assert r.rerank_breakdown_ms is not None
        assert r.rerank_breakdown_ms["forward_ms"] > 0


def test_retrieval_eval_breakdown_stats_empty_when_no_row_has_breakdown():
    """Backwards-compat: without breakdowns, the new field is an empty
    dict and nothing else changes — the summary keeps the existing
    contract for noop runs."""
    from app.capabilities.rag.generation import RetrievedChunk
    from eval.harness.retrieval_eval import run_retrieval_eval

    class _StubRetriever:
        def retrieve(self, query):
            return _StubReportNoBreakdown()

    class _StubReportNoBreakdown:
        results = [
            RetrievedChunk(
                chunk_id="c1", doc_id="doc-1", section="s",
                text="x", score=0.9,
            ),
        ]
        rerank_ms = None
        dense_retrieval_ms = None
        rerank_breakdown_ms = None
        reranker_name = "noop"
        index_version = "v"
        embedding_model = "m"
        candidate_k = 5
        use_mmr = False
        mmr_lambda = None
        dup_rate = 0.0

    dataset = [{"id": "r1", "query": "q", "expected_doc_ids": ["doc-1"]}]
    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=_StubRetriever(), top_k=5, mrr_k=10, ndcg_k=10,
    )
    assert summary.rerank_breakdown_stats == {}
    assert summary.dense_retrieval_row_count == 0
    assert rows[0].rerank_breakdown_ms is None
