"""Tests for the production-recommendation finaliser CLI helpers.

The CLI itself is thin glue around the harness; these tests cover the
deserialiser round-trip (so a saved confirm_sweep_summary.json can
always reconstruct the in-memory ConfirmSweepResult shape) and the
end-to-end behaviour of the finaliser when run against a synthetic
sweep bundle on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest

from eval.harness.phase7_human_gold_tune import GoldSummary, SilverSummary
from eval.harness.phase7_mmr_confirm_sweep import (
    LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST,
    SweepCandidate,
    make_confirm_sweep_grid,
    select_confirmed_best,
    write_confirm_sweep_results_jsonl,
    write_confirm_sweep_summary_json,
)
from scripts.finalize_phase7_5_production_recommendation import (
    load_baseline_summaries_from_jsonl,
    load_confirm_sweep_summary,
    main as finalize_main,
)


def _make_gold(primary: float = 0.74) -> GoldSummary:
    return GoldSummary(
        n_total=50, n_strict_positive=30, n_soft_positive=14,
        n_ambiguous_probe=3, n_abstain_test=3,
        hit_at_1=0.5, hit_at_3=0.6, hit_at_5=0.75, hit_at_10=0.86,
        mrr_at_10=0.6, ndcg_at_10=0.7,
        weighted_hit_at_1=0.55, weighted_hit_at_3=0.7,
        weighted_hit_at_5=0.79, weighted_hit_at_10=0.89,
        weighted_mrr_at_10=0.67, weighted_ndcg_at_10=0.72,
        strict_hit_at_5=0.83, strict_mrr_at_10=0.71,
        section_hit_at_5_when_defined=0.045,
        section_hit_at_10_when_defined=0.09,
        chunk_hit_at_10_when_defined=None,
        primary_score=primary,
        by_bucket={
            "main_work": {
                "n_total": 13.0, "n_positive": 11.0,
                "hit_at_5": 0.5, "mrr_at_10": 0.5, "ndcg_at_10": 0.5,
                "weighted_hit_at_5": 0.65,
                "weighted_mrr_at_10": 0.6, "weighted_ndcg_at_10": 0.62,
            },
            "subpage_named": {
                "n_total": 17.0, "n_positive": 17.0,
                "hit_at_5": 0.7, "mrr_at_10": 0.6, "ndcg_at_10": 0.7,
                "weighted_hit_at_5": 0.94,
                "weighted_mrr_at_10": 0.85, "weighted_ndcg_at_10": 0.89,
            },
            "subpage_generic": {
                "n_total": 17.0, "n_positive": 16.0,
                "hit_at_5": 0.9, "mrr_at_10": 0.7, "ndcg_at_10": 0.8,
                "weighted_hit_at_5": 0.95,
                "weighted_mrr_at_10": 0.78, "weighted_ndcg_at_10": 0.85,
            },
        },
    )


def _make_silver(hit_at_5: float = 0.78) -> SilverSummary:
    return SilverSummary(
        n_total=500, n_scored=475,
        hit_at_1=0.55, hit_at_3=0.73, hit_at_5=hit_at_5, hit_at_10=0.83,
        mrr_at_10=0.65,
        by_bucket={
            "main_work": {
                "n_total": 150.0, "n_scored": 150.0,
                "hit_at_1": 0.4, "hit_at_3": 0.6, "hit_at_5": 0.7,
                "hit_at_10": 0.7, "mrr_at_10": 0.5,
            },
            "subpage_named": {
                "n_total": 100.0, "n_scored": 100.0,
                "hit_at_1": 0.6, "hit_at_3": 0.8, "hit_at_5": 0.9,
                "hit_at_10": 0.93, "mrr_at_10": 0.7,
            },
            "subpage_generic": {
                "n_total": 225.0, "n_scored": 225.0,
                "hit_at_1": 0.6, "hit_at_3": 0.78, "hit_at_5": 0.84,
                "hit_at_10": 0.88, "mrr_at_10": 0.7,
            },
        },
    )


def _persist_sweep_bundle(tmp_path: Path) -> Path:
    """Write a synthetic confirm_sweep_summary.json + results.jsonl
    that mimics the live Phase 7.5 plateau outcome — every λ at
    candidate_k=40 plateaus at the same primary score."""
    grid = make_confirm_sweep_grid(
        candidate_ks=[20, 30, 40],
        mmr_lambdas=[0.60, 0.65, 0.70, 0.75, 0.80],
    )
    base_g = _make_gold(primary=0.733)
    base_s = _make_silver(hit_at_5=0.78)
    cand_results: Dict[str, Tuple[GoldSummary, SilverSummary]] = {}
    for c in grid:
        if c.candidate_k == 40:
            primary, silver = 0.813, 0.836
        elif c.candidate_k == 30:
            primary, silver = 0.795, 0.834
        else:
            primary, silver = 0.792, 0.832
        cand_results[c.name] = (
            _make_gold(primary=primary),
            _make_silver(hit_at_5=silver),
        )
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline_retrieval_title_section_top10",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    report_dir = tmp_path / "confirm_sweep"
    report_dir.mkdir()
    write_confirm_sweep_summary_json(
        report_dir / "confirm_sweep_summary.json", result=result,
    )
    write_confirm_sweep_results_jsonl(
        report_dir / "confirm_sweep_results.jsonl",
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    return report_dir


def test_load_confirm_sweep_summary_round_trips(tmp_path: Path) -> None:
    report_dir = _persist_sweep_bundle(tmp_path)
    loaded = load_confirm_sweep_summary(
        report_dir / "confirm_sweep_summary.json",
    )
    assert loaded.confirmed_best is not None
    assert loaded.confirmed_best.candidate_k == 40
    assert loaded.confirmed_best.mmr_lambda == pytest.approx(0.60)
    # Every grid entry must round-trip.
    assert len(loaded.grid) == 15


def test_load_baseline_summaries_round_trips(tmp_path: Path) -> None:
    report_dir = _persist_sweep_bundle(tmp_path)
    gold, silver = load_baseline_summaries_from_jsonl(
        report_dir / "confirm_sweep_results.jsonl",
    )
    assert gold.primary_score == pytest.approx(0.733, rel=1e-3)
    assert silver.hit_at_5 == pytest.approx(0.78, rel=1e-3)


def test_finalize_main_writes_production_recommended_files(
    tmp_path: Path,
) -> None:
    """End-to-end: the CLI consumes a real sweep bundle and emits
    best_config.production_recommended.{env,json} + splices the
    Production recommendation section into the existing report MD."""
    report_dir = _persist_sweep_bundle(tmp_path)
    # Pre-write the MD we want spliced (the harness writer would do
    # this in the real flow).
    from eval.harness.phase7_mmr_confirm_sweep import (
        write_confirm_sweep_report_md,
    )
    base_g = _make_gold(primary=0.733)
    base_s = _make_silver(hit_at_5=0.78)
    loaded = load_confirm_sweep_summary(
        report_dir / "confirm_sweep_summary.json",
    )
    write_confirm_sweep_report_md(
        report_dir / "confirm_sweep_report.md",
        result=loaded,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        previous_best_name="cand_top10_mmr_lambda07",
    )

    rc = finalize_main([
        "--report-dir", str(report_dir),
        "--log-level", "ERROR",
    ])
    assert rc == 0
    env_path = report_dir / "best_config.production_recommended.env"
    json_path = report_dir / "best_config.production_recommended.json"
    md_path = report_dir / "confirm_sweep_report.md"
    assert env_path.exists()
    assert json_path.exists()

    env_text = env_path.read_text(encoding="utf-8")
    assert "AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000" in env_text
    assert "AIPIPELINE_WORKER_RAG_CANDIDATE_K=40" in env_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["recommended_lambda"] == pytest.approx(0.70)
    assert (
        payload["selected_lambda_policy"]
        == LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST
    )

    md = md_path.read_text(encoding="utf-8")
    assert "## Production recommendation" in md
    # Splice is idempotent.
    rc2 = finalize_main([
        "--report-dir", str(report_dir),
        "--log-level", "ERROR",
    ])
    assert rc2 == 0
    md_after = md_path.read_text(encoding="utf-8")
    assert md_after.count("## Production recommendation") == 1


def test_finalize_main_no_winner_returns_zero(tmp_path: Path) -> None:
    """When the sweep has no metric-best winner, the finaliser emits
    a warning and returns 0 (no artefacts to write)."""
    grid = make_confirm_sweep_grid(candidate_ks=[30], mmr_lambdas=[0.7])
    base_g = _make_gold(primary=0.85)  # baseline already very high
    base_s = _make_silver(hit_at_5=0.78)
    cand_results = {
        grid[0].name: (_make_gold(primary=0.74), _make_silver(hit_at_5=0.5)),
    }
    result = select_confirmed_best(
        grid=grid, baseline_name="baseline",
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    assert result.confirmed_best is None

    report_dir = tmp_path / "confirm_sweep"
    report_dir.mkdir()
    write_confirm_sweep_summary_json(
        report_dir / "confirm_sweep_summary.json", result=result,
    )
    write_confirm_sweep_results_jsonl(
        report_dir / "confirm_sweep_results.jsonl",
        result=result,
        baseline_summary_gold=base_g, baseline_summary_silver=base_s,
        candidate_results=cand_results,
    )
    rc = finalize_main([
        "--report-dir", str(report_dir),
        "--log-level", "ERROR",
    ])
    assert rc == 0
    assert not (report_dir / "best_config.production_recommended.env").exists()
