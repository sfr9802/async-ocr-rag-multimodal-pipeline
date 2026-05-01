"""Phase 7 — tests for the silver-500 full-eval orchestrator.

Pure unit tests over the path-construction / command-building / report-
rendering surface. The orchestrator never embeds, indexes, or reranks
in this test bundle; the heavy stage executions are exercised by the
individual Phase 7.x test modules. What we lock in here:

  1. Path layout matches the spec (silver500 in the names).
  2. Generated stage commands carry the silver500 paths.
  3. The orchestration report includes the silver/gold disclaimer and
     refuses to claim human-verified accuracy.
  4. The human-seed-export command consumes silver500 outputs.
  5. The production-like rewrite refusal count surfaces in the report
     when present in the Phase 7.4 summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.run_phase7_silver_500_full_eval import (
    HUMAN_LABELS_REQUIRED_DISCLAIMER,
    HUMAN_SEED_BASE_NAME,
    NO_PR_CLAIM_DISCLAIMER,
    OrchestrationPlan,
    Phase7SilverPaths,
    PHASE7_0_AB_SUMMARY_NAME,
    PHASE7_0_CHUNKS_NAME,
    PHASE7_0_PER_QUERY_NAME,
    PHASE7_3_CONFIDENCE_NAME,
    PHASE7_3_SUMMARY_NAME,
    PHASE7_4_ATTEMPTS_NAME,
    PHASE7_4_SUMMARY_NAME,
    ReportSnapshot,
    SILVER_JSONL_NAME,
    SILVER_NOT_GOLD_DISCLAIMER,
    SILVER_SUMMARY_JSON_NAME,
    STAGE_HUMAN_SEED,
    STAGE_PHASE7_0,
    STAGE_PHASE7_3,
    STAGE_PHASE7_4,
    STAGE_SILVER,
    build_human_seed_command,
    build_phase7_0_command,
    build_phase7_3_command,
    build_phase7_4_command,
    build_plan,
    build_silver_command,
    load_report_snapshot,
    render_final_report,
    stage_existing_chunks,
    write_final_report,
)


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


def test_silver500_path_layout_groups_under_phase7(tmp_path: Path):
    paths = Phase7SilverPaths(reports_root=tmp_path)
    # All silver500 stages live under <reports>/phase7/silver500/.
    assert paths.silver_dir == tmp_path / "phase7" / "silver500" / "queries"
    assert paths.phase7_0_dir == tmp_path / "phase7" / "silver500" / "retrieval"
    assert paths.phase7_3_dir == tmp_path / "phase7" / "silver500" / "confidence"
    assert paths.phase7_4_dir == tmp_path / "phase7" / "silver500" / "recovery"
    # Human-seed export lives under the seeds/ tree, not silver500/.
    assert paths.human_seed_dir == tmp_path / "phase7" / "seeds" / "human_seed_50"
    # Final report lives under the silver dir.
    assert paths.final_report.name == "PHASE7_SILVER500_FULL_EVAL_REPORT.md"


def test_silver500_artefact_filenames_match_spec(tmp_path: Path):
    paths = Phase7SilverPaths(reports_root=tmp_path)
    assert paths.silver_queries_jsonl.name == SILVER_JSONL_NAME
    assert paths.silver_summary_json.name == SILVER_SUMMARY_JSON_NAME
    assert paths.phase7_0_per_query.name == PHASE7_0_PER_QUERY_NAME
    assert paths.phase7_0_chunks.name == PHASE7_0_CHUNKS_NAME
    assert paths.phase7_0_ab_summary.name == PHASE7_0_AB_SUMMARY_NAME
    assert paths.phase7_3_confidence.name == PHASE7_3_CONFIDENCE_NAME
    assert paths.phase7_3_summary.name == PHASE7_3_SUMMARY_NAME
    assert paths.phase7_4_attempts.name == PHASE7_4_ATTEMPTS_NAME
    assert paths.phase7_4_summary.name == PHASE7_4_SUMMARY_NAME
    assert paths.human_seed_jsonl.name == f"{HUMAN_SEED_BASE_NAME}.jsonl"
    assert paths.human_seed_csv.name == f"{HUMAN_SEED_BASE_NAME}.csv"
    assert paths.human_seed_md.name == f"{HUMAN_SEED_BASE_NAME}.md"


# ---------------------------------------------------------------------------
# Stage command builders
# ---------------------------------------------------------------------------


def _paths(tmp_path: Path) -> Phase7SilverPaths:
    return Phase7SilverPaths(reports_root=tmp_path)


def test_silver_command_writes_to_silver_dir(tmp_path: Path):
    paths = _paths(tmp_path)
    pages = tmp_path / "pages_v4.jsonl"
    cmd = build_silver_command(paths, pages_v4=pages, seed=42)
    assert cmd.stage == STAGE_SILVER
    assert cmd.module == "scripts.run_phase7_silver_500"
    # silver500 generator must point its --out-dir at the silver dir.
    assert "--out-dir" in cmd.argv
    out_dir_idx = cmd.argv.index("--out-dir") + 1
    assert Path(cmd.argv[out_dir_idx]) == paths.silver_dir
    # And read from the requested pages_v4.
    assert "--pages-v4" in cmd.argv
    assert cmd.argv[cmd.argv.index("--pages-v4") + 1] == str(pages)


def test_silver_command_uses_500_split_targets_by_default(tmp_path: Path):
    paths = _paths(tmp_path)
    cmd = build_silver_command(
        paths, pages_v4=tmp_path / "p.jsonl", seed=42,
    )
    # Default per-bucket targets sum to 500.
    main = int(cmd.argv[cmd.argv.index("--main-work-target") + 1])
    sub_g = int(cmd.argv[cmd.argv.index("--subpage-generic-target") + 1])
    sub_n = int(cmd.argv[cmd.argv.index("--subpage-named-target") + 1])
    assert main + sub_g + sub_n == 500


def test_phase7_0_command_uses_silver500_queries_and_skips_heavy_steps(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    cmd = build_phase7_0_command(
        paths,
        rag_chunks=tmp_path / "rag_chunks.jsonl",
        pages_v4=tmp_path / "pages_v4.jsonl",
        index_root=tmp_path / "indexes",
    )
    assert cmd.stage == STAGE_PHASE7_0
    assert cmd.module == "scripts.run_phase7_0_retrieval_title_ab"
    # Must point --report-dir at the silver500 retrieval stage dir.
    rd_idx = cmd.argv.index("--report-dir") + 1
    assert Path(cmd.argv[rd_idx]) == paths.phase7_0_dir
    # Must wire --queries to the silver-500 jsonl.
    qid = cmd.argv.index("--queries") + 1
    assert Path(cmd.argv[qid]) == paths.silver_queries_jsonl
    assert paths.silver_queries_jsonl.name == SILVER_JSONL_NAME
    # Must skip the heavy export / diff / index-build steps so we reuse
    # the existing Phase 7.0 chunks / indexes.
    for skip in ("--skip-export", "--skip-diff", "--skip-index-build"):
        assert skip in cmd.argv


def test_phase7_3_command_consumes_silver500_paths(tmp_path: Path):
    paths = _paths(tmp_path)
    cmd = build_phase7_3_command(paths)
    assert cmd.stage == STAGE_PHASE7_3
    assert cmd.module == "scripts.run_phase7_3_confidence_eval"
    # Must read per-query / chunks / silver-queries from the silver500 dirs.
    pq = cmd.argv[cmd.argv.index("--per-query") + 1]
    cs = cmd.argv[cmd.argv.index("--chunks") + 1]
    sq = cmd.argv[cmd.argv.index("--silver-queries") + 1]
    rd = cmd.argv[cmd.argv.index("--report-dir") + 1]
    assert Path(pq) == paths.phase7_0_per_query
    assert Path(cs) == paths.phase7_0_chunks
    assert Path(sq) == paths.silver_queries_jsonl
    assert Path(rd) == paths.phase7_3_dir


def test_phase7_4_command_consumes_silver500_paths(tmp_path: Path):
    paths = _paths(tmp_path)
    cmd = build_phase7_4_command(paths, rewrite_mode="both")
    assert cmd.stage == STAGE_PHASE7_4
    assert cmd.module == "scripts.run_phase7_4_controlled_recovery"
    cj = cmd.argv[cmd.argv.index("--confidence-jsonl") + 1]
    pq = cmd.argv[cmd.argv.index("--per-query") + 1]
    cs = cmd.argv[cmd.argv.index("--chunks") + 1]
    sq = cmd.argv[cmd.argv.index("--silver-queries") + 1]
    rd = cmd.argv[cmd.argv.index("--report-dir") + 1]
    rm = cmd.argv[cmd.argv.index("--rewrite-mode") + 1]
    assert Path(cj) == paths.phase7_3_confidence
    assert Path(pq) == paths.phase7_0_per_query
    assert Path(cs) == paths.phase7_0_chunks
    assert Path(sq) == paths.silver_queries_jsonl
    assert Path(rd) == paths.phase7_4_dir
    assert rm == "both"


def test_phase7_4_command_strict_label_leakage_default_on(tmp_path: Path):
    """Default Phase 7.4 invocation must keep the leakage guard ON."""
    paths = _paths(tmp_path)
    cmd = build_phase7_4_command(paths)
    assert "--no-strict-label-leakage" not in cmd.argv


def test_phase7_4_command_strict_label_leakage_optional_off(tmp_path: Path):
    paths = _paths(tmp_path)
    cmd = build_phase7_4_command(paths, no_strict_label_leakage=True)
    assert "--no-strict-label-leakage" in cmd.argv


def test_human_seed_command_consumes_silver500_outputs(tmp_path: Path):
    paths = _paths(tmp_path)
    cmd = build_human_seed_command(paths)
    assert cmd.stage == STAGE_HUMAN_SEED
    assert cmd.module == "scripts.export_phase7_human_gold_seed"
    sq = cmd.argv[cmd.argv.index("--silver-queries") + 1]
    pq = cmd.argv[cmd.argv.index("--per-query") + 1]
    cf = cmd.argv[cmd.argv.index("--confidence") + 1]
    rec = cmd.argv[cmd.argv.index("--recovery") + 1]
    cs = cmd.argv[cmd.argv.index("--chunks") + 1]
    od = cmd.argv[cmd.argv.index("--out-dir") + 1]
    bn = cmd.argv[cmd.argv.index("--base-name") + 1]
    # Every consumed input must be a silver500 path.
    assert Path(sq) == paths.silver_queries_jsonl
    assert Path(pq) == paths.phase7_0_per_query
    assert Path(cf) == paths.phase7_3_confidence
    assert Path(rec) == paths.phase7_4_attempts
    assert Path(cs) == paths.phase7_0_chunks
    assert Path(od) == paths.human_seed_dir
    assert bn == HUMAN_SEED_BASE_NAME


def test_human_seed_command_default_target_total_50(tmp_path: Path):
    """Spec requires 50-row default."""
    paths = _paths(tmp_path)
    cmd = build_human_seed_command(paths)
    tt = cmd.argv[cmd.argv.index("--target-total") + 1]
    assert int(tt) == 50


def test_build_plan_returns_five_stages_in_canonical_order(tmp_path: Path):
    paths = _paths(tmp_path)
    plan = build_plan(
        paths=paths,
        pages_v4=tmp_path / "p.jsonl",
        rag_chunks=tmp_path / "c.jsonl",
        index_root=tmp_path / "ix",
    )
    stages = [c.stage for c in plan.as_ordered_list()]
    assert stages == [
        STAGE_SILVER,
        STAGE_PHASE7_0,
        STAGE_PHASE7_3,
        STAGE_PHASE7_4,
        STAGE_HUMAN_SEED,
    ]


def test_stage_commands_have_a_documented_shell_form(tmp_path: Path):
    """Every stage's ``shell_form`` should round-trip the module name."""
    paths = _paths(tmp_path)
    plan = build_plan(
        paths=paths,
        pages_v4=tmp_path / "p.jsonl",
        rag_chunks=tmp_path / "c.jsonl",
        index_root=tmp_path / "ix",
    )
    for cmd in plan.as_ordered_list():
        s = cmd.shell_form()
        assert s.startswith("python -m " + cmd.module)


# ---------------------------------------------------------------------------
# Final orchestration report
# ---------------------------------------------------------------------------


def _silver_summary_fixture() -> Dict[str, Any]:
    return {
        "schema": "queries-v4-silver-500.summary.v1",
        "is_silver_not_gold": True,
        "seed": 42,
        "target_total": 500,
        "requested_total": 500,
        "actual_total": 487,
        "bucket_targets": {
            "main_work": 150,
            "subpage_generic": 200,
            "subpage_named": 150,
        },
        "bucket_actual_counts": {
            "main_work": 150,
            "subpage_generic": 200,
            "subpage_named": 137,
        },
        "bucket_deficits": {
            "main_work": 0, "subpage_generic": 0, "subpage_named": 13,
        },
        "candidate_pool_counts": {
            "main_work": 250, "subpage_generic": 350, "subpage_named": 137,
        },
        "template_kind_counts": {
            "title_lookup": 60, "plot_summary": 60, "evaluation": 30,
            "alias_lookup": 0, "ambiguous_short": 0,
            "section_lookup": 100, "section_detail": 50,
            "section_question": 50,
            "named_lookup": 80, "named_question": 50, "named_alias": 7,
        },
        "label_confidence_counts": {
            "high": 280, "medium": 180, "low": 27,
        },
    }


def _phase7_0_ab_summary_fixture() -> Dict[str, Any]:
    """Mirror the actual Phase 7.0 ab_summary schema (hit_at_K direct keys)."""
    return {
        "baseline_variant": "title_section",
        "candidate_variant": "retrieval_title_section",
        "n_queries": 500,
        "k_values": [1, 3, 5, 10],
        "candidate": {
            "count": 500,
            "hit_at_1": 0.81, "hit_at_3": 0.93, "hit_at_5": 0.95,
            "hit_at_10": 0.98, "mrr_at_10": 0.88, "ndcg_at_10": 0.90,
            "dup_rate": 0.71, "same_title_collisions_avg": 4.5,
        },
        "baseline": {
            "count": 500,
            "hit_at_1": 0.59, "hit_at_3": 0.71, "hit_at_5": 0.74,
            "hit_at_10": 0.79, "mrr_at_10": 0.66, "ndcg_at_10": 0.69,
            "dup_rate": 0.68, "same_title_collisions_avg": 5.4,
        },
        "status_counts": {
            "improved": 110, "regressed": 5, "both_hit": 360,
            "both_missed": 12, "unchanged": 0,
        },
    }


def _phase7_3_summary_fixture() -> Dict[str, Any]:
    return {
        "labels": {
            "CONFIDENT": 30, "AMBIGUOUS": 380,
            "LOW_CONFIDENCE": 60, "FAILED": 17,
        },
        "actions": {
            "ANSWER": 30, "ANSWER_WITH_CAUTION": 380,
            "HYBRID_RECOVERY": 25, "QUERY_REWRITE": 35,
            "ASK_CLARIFICATION": 0, "INSUFFICIENT_EVIDENCE": 17,
        },
    }


def _phase7_4_summary_fixture(*, leakage_count: int = 35) -> Dict[str, Any]:
    return {
        "config": {
            "rewrite_mode": "both",
            "final_k": 10, "hybrid_top_k": 10,
            "bm25_pool_size": 100, "k_rrf": 60,
            "top_n_for_production": 5, "strict_label_leakage": True,
            "side": "candidate",
        },
        "n_queries": 487,
        "totals": {
            "n_decisions": 487,
            "skipped": 432,
            "attempted": 55,
            "recovered": 7,
            "regressed": 2,
            "gold_newly_entered_candidates": 4,
        },
        "by_action": {
            "NOOP": {"count": 30, "attempted": 0, "recovered": 0, "regressed": 0},
            "SKIP_REFUSE": {"count": 17, "attempted": 0, "recovered": 0, "regressed": 0},
            "SKIP_CAUTION": {"count": 380, "attempted": 0, "recovered": 0, "regressed": 0},
            "SKIP_DEFER": {"count": leakage_count, "attempted": 0, "recovered": 0, "regressed": 0},
            "ATTEMPT_HYBRID": {"count": 25, "attempted": 25, "recovered": 5, "regressed": 1},
            "ATTEMPT_REWRITE": {"count": 35, "attempted": 30, "recovered": 2, "regressed": 1},
        },
        "by_rewrite_mode": {
            "oracle": {"count": 35, "attempted": 35, "recovered": 12, "regressed": 1},
            "production_like": {"count": 35, "attempted": 0, "recovered": 0, "regressed": 0},
        },
        "invariants": {
            "answer_with_caution_recovered": False,
            "answer_with_caution_skip_count": 380,
            "insufficient_evidence_recovered": False,
            "insufficient_evidence_refused_count": 17,
            "ask_clarification_deferred_count": 0,
            "label_leakage_refused_count": leakage_count,
        },
    }


def _populate_artefacts(
    paths: Phase7SilverPaths, *, leakage_count: int = 35,
) -> None:
    """Write fixture json/jsonl into every stage directory."""
    for d in paths.all_stage_dirs().values():
        d.mkdir(parents=True, exist_ok=True)
    paths.silver_summary_json.write_text(
        json.dumps(_silver_summary_fixture(), ensure_ascii=False),
        encoding="utf-8",
    )
    # silver queries: 487 lines (matches actual_total)
    paths.silver_queries_jsonl.write_text(
        "\n".join(json.dumps({"id": f"v4-silver-500-{i:04d}"}) for i in range(487)),
        encoding="utf-8",
    )
    paths.phase7_0_ab_summary.write_text(
        json.dumps(_phase7_0_ab_summary_fixture(), ensure_ascii=False),
        encoding="utf-8",
    )
    # per_query_comparison.jsonl - 487 lines
    paths.phase7_0_per_query.write_text(
        "\n".join(json.dumps({"qid": f"v4-silver-500-{i:04d}"}) for i in range(487)),
        encoding="utf-8",
    )
    paths.phase7_3_summary.write_text(
        json.dumps(_phase7_3_summary_fixture(), ensure_ascii=False),
        encoding="utf-8",
    )
    paths.phase7_3_confidence.write_text(
        "\n".join(json.dumps({"query_id": f"v4-silver-500-{i:04d}"}) for i in range(487)),
        encoding="utf-8",
    )
    paths.phase7_4_summary.write_text(
        json.dumps(
            _phase7_4_summary_fixture(leakage_count=leakage_count),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.phase7_4_attempts.write_text(
        "\n".join(json.dumps({"decision": {"query_id": f"v4-silver-500-{i:04d}"}}) for i in range(60)),
        encoding="utf-8",
    )
    paths.human_seed_jsonl.write_text(
        "\n".join(json.dumps({"query_id": f"q-{i}"}) for i in range(50)),
        encoding="utf-8",
    )


def test_final_report_carries_silver_disclaimer(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    assert SILVER_NOT_GOLD_DISCLAIMER in md
    assert NO_PR_CLAIM_DISCLAIMER in md
    assert HUMAN_LABELS_REQUIRED_DISCLAIMER in md


def test_final_report_does_not_claim_human_verified_accuracy(tmp_path: Path):
    """Spec: report must NOT claim human-verified precision/recall/accuracy.

    The disclaimer mentions 'human-verified' inside a *negative* phrase
    ('NOT human-verified gold'), which is desired. The forbidden pattern
    is any positive claim like 'precision: 0.81' or
    'human-verified accuracy of...'. We grep for those.
    """
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    forbidden_substrings = (
        "human-verified accuracy",
        "human-verified precision",
        "human-verified recall",
    )
    for sub in forbidden_substrings:
        assert sub not in md.lower(), (
            f"Report unexpectedly claims {sub!r}: {md[:200]}"
        )
    # Hits are reported as silver-agreement, not as accuracy.
    assert "silver-agreement" in md.lower()


def test_final_report_renders_phase7_0_metric_table(tmp_path: Path):
    """Spec §2: retrieval metrics must be rendered as silver-agreement
    metrics — *not* as accuracy claims.

    The fixture's hit_at_1 / mrr_at_10 numbers must surface in the
    report so a reviewer can sanity-check what they're looking at.
    """
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # Canonical metric names appear (with @, not _at_).
    assert "| hit@1 |" in md
    assert "| mrr@10 |" in md
    # Section header is silver-agreement.
    assert "silver-agreement" in md.lower()
    # Numeric values from the fixture surface.
    assert "0.8100" in md  # candidate hit@1
    assert "0.5900" in md  # baseline hit@1


def test_final_report_lists_silver_bucket_distribution(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    assert "Bucket distribution" in md
    assert "main_work" in md
    assert "subpage_generic" in md
    assert "subpage_named" in md
    # Numbers from the fixture must appear (target 150/200/150).
    assert "150" in md
    assert "200" in md


def test_final_report_lists_confidence_label_distribution(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    for label in ("CONFIDENT", "AMBIGUOUS", "LOW_CONFIDENCE", "FAILED"):
        assert label in md


def test_final_report_lists_recommended_action_distribution(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    for action in (
        "ANSWER",
        "ANSWER_WITH_CAUTION",
        "HYBRID_RECOVERY",
        "QUERY_REWRITE",
        "INSUFFICIENT_EVIDENCE",
    ):
        assert action in md


def test_final_report_surfaces_recovery_attempt_counts(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # Phase 7.4 attempts (55 attempted, 7 recovered) must surface.
    assert "55" in md
    assert "recovered: **7**" in md
    assert "regressed: **2**" in md
    # Action breakdown table must list the rewrite + hybrid actions.
    assert "ATTEMPT_HYBRID" in md
    assert "ATTEMPT_REWRITE" in md


def test_final_report_surfaces_production_like_rewrite_refusal_count(
    tmp_path: Path,
):
    """Spec §3: production-like rewrite leakage refusal count must be
    explicitly visible in the report when present."""
    paths = _paths(tmp_path)
    _populate_artefacts(paths, leakage_count=35)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # The report must spell out "production_like" + refusal count.
    assert "production_like" in md
    assert "production_like LabelLeakageError refused: **35**" in md
    # The report must also show the oracle vs production-like breakdown.
    assert "oracle attempts" in md
    assert "production_like attempts" in md


def test_final_report_oracle_and_production_like_attempts_shown_separately(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    _populate_artefacts(paths, leakage_count=35)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # Oracle had 35 attempts in the fixture; production_like had 0
    # (because all were refused). Both numbers must surface.
    assert "oracle attempts: **35**" in md
    assert "production_like attempts: **0**" in md


def test_final_report_explains_expected_title_caused_refusal(
    tmp_path: Path,
):
    """When leakage > 0, the report must call out *why* the production
    rewriter refused — `expected_title` presence on the QUERY_REWRITE
    rows is the load-bearing signal."""
    paths = _paths(tmp_path)
    _populate_artefacts(paths, leakage_count=35)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    assert "expected_title" in md
    assert "leakage guard" in md.lower()
    # And the report must say the guard is NOT being fixed in this phase.
    assert "NOT fixed" in md or "not fixed" in md.lower()


def test_final_report_no_refusal_message_when_leakage_zero(tmp_path: Path):
    """When leakage_count is 0 the report should still say so explicitly."""
    paths = _paths(tmp_path)
    _populate_artefacts(paths, leakage_count=0)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # Must still report the count (zero is informative).
    assert "production_like LabelLeakageError refused: **0**" in md
    assert (
        "No production_like rewrite was refused" in md
        or "no production_like rewrite" in md.lower()
    )


def test_final_report_human_seed_count_appears(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # The human seed JSONL has 50 lines in the fixture — that count must
    # surface in the report.
    assert "audit rows emitted: **50**" in md
    # Every human_label column must be flagged blank by design.
    assert "blank" in md.lower()
    # The seed must be flagged silver-derived.
    assert "silver-derived" in md.lower()


def test_final_report_lists_artefact_paths(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    # Each stage directory must be referenced by its rendered path.
    # The rendered paths use os.sep; check both representations to be
    # platform-agnostic.
    for expected in (paths.phase7_0_dir, paths.phase7_3_dir,
                     paths.phase7_4_dir, paths.human_seed_dir):
        as_posix = expected.as_posix()
        as_native = str(expected)
        assert (as_posix in md) or (as_native in md), (
            f"expected {expected} in rendered report, got: {md!r}"
        )
    assert SILVER_JSONL_NAME in md


def test_final_report_handles_missing_artefacts(tmp_path: Path):
    """No artefacts on disk → report still renders with placeholder
    notes. The disclaimer is still present."""
    paths = _paths(tmp_path)
    # Don't populate anything; the silver_dir doesn't even exist yet.
    snap = load_report_snapshot(paths)
    md = render_final_report(snap)
    assert SILVER_NOT_GOLD_DISCLAIMER in md
    # Each stage section must surface the "stage did not run" placeholder.
    assert "stage A did not run" in md
    assert "stage B did not run" in md
    assert "stage C did not run" in md
    assert "stage D did not run" in md


def test_write_final_report_persists_to_silver_dir(tmp_path: Path):
    paths = _paths(tmp_path)
    _populate_artefacts(paths)
    snap = load_report_snapshot(paths)
    out = write_final_report(snap)
    assert out == paths.final_report
    assert paths.final_report.exists()
    body = paths.final_report.read_text(encoding="utf-8")
    assert SILVER_NOT_GOLD_DISCLAIMER in body


def test_load_report_snapshot_is_resilient_to_garbage_json(tmp_path: Path):
    paths = _paths(tmp_path)
    paths.silver_dir.mkdir(parents=True, exist_ok=True)
    paths.silver_summary_json.write_text("not-json", encoding="utf-8")
    snap = load_report_snapshot(paths)
    # Garbage JSON ⇒ summary stays None instead of raising.
    assert snap.silver_summary is None


# ---------------------------------------------------------------------------
# Regression: produced shell command embeds silver500 paths
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 7.0 chunks pre-staging (existing-export hardlink/copy)
# ---------------------------------------------------------------------------


def test_stage_existing_chunks_links_from_canonical_phase7_0_dir(
    tmp_path: Path,
):
    """When the chunks file is missing, the orchestrator hardlinks (or
    copies) it from the existing Phase 7.0 export into the silver500
    phase7_0 dir. This unblocks the Phase 7.0 ``--skip-export``
    validation without re-running the heavy export."""
    paths = _paths(tmp_path)
    src_dir = (
        Path(paths.reports_root) / "phase7" / "7.0_retrieval_title_ab"
    )
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / PHASE7_0_CHUNKS_NAME
    src_file.write_text("dummy chunks file content", encoding="utf-8")
    # Sanity: dst doesn't exist before the call.
    assert not paths.phase7_0_chunks.exists()
    out = stage_existing_chunks(paths)
    assert out == paths.phase7_0_chunks
    assert paths.phase7_0_chunks.exists()
    assert paths.phase7_0_chunks.read_text(encoding="utf-8") == src_file.read_text(encoding="utf-8")


def test_stage_existing_chunks_no_op_when_target_exists(tmp_path: Path):
    paths = _paths(tmp_path)
    paths.phase7_0_dir.mkdir(parents=True, exist_ok=True)
    paths.phase7_0_chunks.write_text("preserved", encoding="utf-8")
    out = stage_existing_chunks(paths)
    assert out == paths.phase7_0_chunks
    # Target was preserved, not overwritten.
    assert paths.phase7_0_chunks.read_text(encoding="utf-8") == "preserved"


def test_stage_existing_chunks_returns_none_when_source_missing(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    out = stage_existing_chunks(paths)
    # No source file → returns None and never creates the target.
    assert out is None
    assert not paths.phase7_0_chunks.exists()


def test_stage_existing_chunks_accepts_explicit_source_dir(tmp_path: Path):
    """The ``source_phase7_0_dir`` override must be honoured."""
    paths = _paths(tmp_path)
    custom = tmp_path / "alt_phase7_0_export"
    custom.mkdir(parents=True, exist_ok=True)
    src_file = custom / PHASE7_0_CHUNKS_NAME
    src_file.write_text("alt-export", encoding="utf-8")
    out = stage_existing_chunks(paths, source_phase7_0_dir=custom)
    assert out == paths.phase7_0_chunks
    assert paths.phase7_0_chunks.read_text(encoding="utf-8") == "alt-export"


def test_planned_shell_commands_carry_silver500_paths(tmp_path: Path):
    paths = _paths(tmp_path)
    plan = build_plan(
        paths=paths,
        pages_v4=tmp_path / "p.jsonl",
        rag_chunks=tmp_path / "c.jsonl",
        index_root=tmp_path / "ix",
    )
    for cmd in plan.as_ordered_list():
        s = cmd.shell_form()
        # Every stage's shell form should reference the silver-500
        # generator output, the silver500 stage dirs, OR (for stage A
        # itself) the silver dir.
        if cmd.stage == STAGE_SILVER:
            assert paths.silver_dir.name in s
        elif cmd.stage == STAGE_PHASE7_0:
            assert SILVER_JSONL_NAME in s
            assert paths.phase7_0_dir.name in s
        elif cmd.stage == STAGE_PHASE7_3:
            assert paths.phase7_3_dir.name in s
            assert SILVER_JSONL_NAME in s
        elif cmd.stage == STAGE_PHASE7_4:
            assert paths.phase7_4_dir.name in s
            assert SILVER_JSONL_NAME in s
        elif cmd.stage == STAGE_HUMAN_SEED:
            assert paths.human_seed_dir.name in s
            assert SILVER_JSONL_NAME in s
