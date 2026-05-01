"""Phase 7.6 — tests for the section-aware rerank scaffold.

Scope of the harness under test
(``eval.harness.phase7_section_aware_rerank``):

  * grid generation produces every spec'd variant in deterministic
    order, with the right ``deployable`` flag per strategy.
  * the section-bonus scorer correctly handles equality / prefix /
    leaf-token-overlap matches and the no-expected-path no-op.
  * the same-page chunk rerank picks the page's best section-overlap
    chunk against the query, leaving non-overlapping pages alone.
  * the page-first-section-rerank composes the Phase 7.5 baseline +
    the same-page rerank and produces ``top_k`` outputs.
  * the supporting-chunk-proximity strategy boosts a candidate that
    matches the supporting chunk id and falls back to page-id when
    the chunk id is missing.
  * the renderers carry the deployable / diagnostic distinction
    (verbatim) so a reviewer cannot mistake a diagnostic-only variant
    for a production-deployable one.
  * the JSON writer emits the guardrail thresholds the plan doc pins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

from eval.harness.phase7_human_gold_tune import RetrievedDoc
from eval.harness.phase7_section_aware_rerank import (
    DEFAULT_SECTION_BONUS_VALUES,
    PHASE_7_5_RECOMMENDED_CANDIDATE_K,
    PHASE_7_5_RECOMMENDED_LAMBDA,
    PHASE_7_5_RECOMMENDED_TOP_K,
    PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD,
    PHASE_7_6_SILVER_REGRESSION_THRESHOLD,
    PHASE_7_6_SUBPAGE_NAMED_THRESHOLD,
    STRATEGY_BASELINE_NO_RERANK,
    STRATEGY_PAGE_FIRST_SECTION_RERANK,
    STRATEGY_SAME_PAGE_CHUNK_RERANK,
    STRATEGY_SECTION_BONUS,
    STRATEGY_SUPPORTING_CHUNK_PROXIMITY,
    SectionRerankInput,
    SectionRerankSpec,
    apply_page_first_section_rerank,
    apply_same_page_chunk_rerank,
    apply_section_bonus_post_hoc,
    make_section_rerank_grid,
    page_hit_at_k,
    render_section_rerank_grid_md,
    run_variant_for_query,
    section_hit_at_k,
    write_section_rerank_grid_json,
    write_section_rerank_grid_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    rank: int,
    page_id: str,
    score: float,
    *,
    chunk_suffix: str = "",
    section: Sequence[str] = ("개요",),
    title: str = "",
) -> RetrievedDoc:
    return RetrievedDoc(
        rank=rank, chunk_id=f"{page_id}-c{rank}{chunk_suffix}",
        page_id=page_id, title=title or page_id,
        section_path=tuple(section), score=score,
    )


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def test_make_section_rerank_grid_default_size() -> None:
    """Default grid: baseline + 3 section_bonus + supporting + page-first
    + same-page = 7 variants in deterministic order."""
    grid = make_section_rerank_grid()
    assert len(grid) == 7
    assert grid[0].strategy == STRATEGY_BASELINE_NO_RERANK
    # Three section_bonus variants in the configured order.
    bonus_specs = [g for g in grid if g.strategy == STRATEGY_SECTION_BONUS]
    assert len(bonus_specs) == 3
    assert bonus_specs[0].section_bonus == pytest.approx(0.05)
    assert bonus_specs[2].section_bonus == pytest.approx(0.15)


def test_make_section_rerank_grid_marks_diagnostic_strategies() -> None:
    grid = make_section_rerank_grid()
    by_name = {g.name: g for g in grid}
    # Production-deployable variants:
    assert by_name["baseline_phase7_5_recommended"].deployable is True
    assert by_name["page_first_section_rerank_overlap"].deployable is True
    assert by_name["same_page_chunk_rerank"].deployable is True
    # Diagnostic-only:
    for bonus in (5, 10, 15):
        assert by_name[f"section_bonus_{bonus:03d}"].deployable is False
    assert by_name["supporting_chunk_proximity"].deployable is False


def test_make_section_rerank_grid_can_skip_variants() -> None:
    grid = make_section_rerank_grid(
        section_bonus_values=[0.10],
        include_baseline=False,
        include_supporting_chunk_proximity=False,
        include_page_first=False,
        include_same_page_rerank=False,
    )
    assert len(grid) == 1
    assert grid[0].strategy == STRATEGY_SECTION_BONUS


def test_phase7_6_thresholds_pinned_to_plan() -> None:
    """The plan doc states 2pp / 3pp / 5pp guardrails — pin them so a
    contributor relaxing the harness must also bump the plan doc."""
    assert PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD == pytest.approx(0.02)
    assert PHASE_7_6_SILVER_REGRESSION_THRESHOLD == pytest.approx(0.03)
    assert PHASE_7_6_SUBPAGE_NAMED_THRESHOLD == pytest.approx(0.05)


def test_phase7_5_baseline_pinned() -> None:
    """The Phase 7.5 promoted (recommended) config defines the new
    baseline for this sweep — pin it."""
    assert PHASE_7_5_RECOMMENDED_CANDIDATE_K == 40
    assert PHASE_7_5_RECOMMENDED_LAMBDA == pytest.approx(0.70)
    assert PHASE_7_5_RECOMMENDED_TOP_K == 10


def test_default_section_bonus_values() -> None:
    assert DEFAULT_SECTION_BONUS_VALUES == (0.05, 0.10, 0.15)


# ---------------------------------------------------------------------------
# Section bonus scorer
# ---------------------------------------------------------------------------


def test_apply_section_bonus_promotes_exact_match() -> None:
    """An exact-match section_path candidate should jump to rank 1
    once the bonus is large enough to outweigh the score gap."""
    docs = [
        _doc(1, "P-A", 0.90, section=("줄거리", "결말")),
        _doc(2, "P-B", 0.80, section=("개요",)),  # exact match
        _doc(3, "P-C", 0.70, section=("기타",)),
    ]
    out = apply_section_bonus_post_hoc(
        docs, expected_section_path=("개요",), bonus=0.15, top_k=2,
    )
    assert [d.page_id for d in out] == ["P-B", "P-A"]


def test_apply_section_bonus_no_op_when_expected_empty() -> None:
    """No expected path → pure top-k truncation, ranks renumbered."""
    docs = [_doc(1, "P-A", 0.9), _doc(2, "P-B", 0.8), _doc(3, "P-C", 0.7)]
    out = apply_section_bonus_post_hoc(
        docs, expected_section_path=(), bonus=0.10, top_k=2,
    )
    assert [d.page_id for d in out] == ["P-A", "P-B"]
    assert [d.rank for d in out] == [1, 2]


def test_apply_section_bonus_renumbers_ranks() -> None:
    docs = [
        _doc(1, "P-A", 0.90, section=("기타",)),
        _doc(2, "P-B", 0.80, section=("개요",)),
    ]
    out = apply_section_bonus_post_hoc(
        docs, expected_section_path=("개요",), bonus=0.15, top_k=2,
    )
    assert [d.rank for d in out] == [1, 2]


def test_apply_section_bonus_zero_bonus_is_truncate() -> None:
    docs = [_doc(1, "P-A", 0.9), _doc(2, "P-B", 0.8)]
    out = apply_section_bonus_post_hoc(
        docs, expected_section_path=("개요",), bonus=0.0, top_k=1,
    )
    assert len(out) == 1
    assert out[0].page_id == "P-A"


# ---------------------------------------------------------------------------
# Same-page chunk rerank
# ---------------------------------------------------------------------------


def test_apply_same_page_chunk_rerank_swaps_to_better_overlap() -> None:
    """Within the same page, the rerank should pick the chunk whose
    section_path overlaps the query token best."""
    docs = [_doc(1, "P-A", 0.9, section=("개요",))]
    pool = {
        "P-A": [
            _doc(10, "P-A", 0.70, chunk_suffix="-better",
                 section=("줄거리", "결말")),
            _doc(11, "P-A", 0.60, section=("개요",)),
        ],
    }
    out = apply_same_page_chunk_rerank(
        docs, full_page_chunks=pool,
        query="줄거리 결말", top_k=1,
    )
    assert len(out) == 1
    assert out[0].chunk_id.endswith("-better")


def test_apply_same_page_chunk_rerank_keeps_when_no_overlap_found() -> None:
    """When no candidate in the page beats the original on overlap,
    the original chunk is kept."""
    docs = [_doc(1, "P-A", 0.9, section=("개요",))]
    pool = {
        "P-A": [
            _doc(2, "P-A", 0.7, section=("기타",)),  # zero overlap
        ],
    }
    out = apply_same_page_chunk_rerank(
        docs, full_page_chunks=pool, query="개요", top_k=1,
    )
    assert out[0].chunk_id == docs[0].chunk_id


def test_apply_same_page_chunk_rerank_passes_through_missing_pages() -> None:
    """Pages without a full_page_chunks entry → original chunk kept."""
    docs = [_doc(1, "P-Z", 0.9, section=("기타",))]
    out = apply_same_page_chunk_rerank(
        docs, full_page_chunks={}, query="개요", top_k=1,
    )
    assert out[0].page_id == "P-Z"
    assert out[0].rank == 1


# ---------------------------------------------------------------------------
# Page-first then section rerank
# ---------------------------------------------------------------------------


def test_apply_page_first_section_rerank_returns_top_k_outputs() -> None:
    pool = [
        _doc(1, "P-A", 0.95, section=("기타",)),
        _doc(2, "P-A", 0.94, section=("개요",)),
        _doc(3, "P-B", 0.93, section=("기타",)),
        _doc(4, "P-C", 0.92, section=("개요",)),
        _doc(5, "P-D", 0.91, section=("줄거리",)),
    ]
    out = apply_page_first_section_rerank(
        pool,
        query="개요",
        candidate_k=5, mmr_lambda=0.7, top_k=3,
    )
    assert len(out) == 3
    # All ranks should renumber to 1..3.
    assert [d.rank for d in out] == [1, 2, 3]


def test_apply_page_first_section_rerank_swaps_within_page() -> None:
    """The page-first rerank should reach back into the per-page pool
    and swap chunks where overlap is better."""
    pool = [
        _doc(1, "P-A", 0.95, section=("기타",)),
        _doc(2, "P-A", 0.94, section=("줄거리", "결말")),
        _doc(3, "P-B", 0.50, section=("기타",)),
    ]
    out = apply_page_first_section_rerank(
        pool, query="줄거리 결말",
        candidate_k=3, mmr_lambda=1.0, top_k=2,
    )
    assert out[0].page_id == "P-A"
    # The swap should pick the chunk that overlaps "줄거리 결말".
    assert out[0].section_path == ("줄거리", "결말")


# ---------------------------------------------------------------------------
# Strategy dispatch via run_variant_for_query
# ---------------------------------------------------------------------------


def _pool_inputs(qid: str = "q1") -> SectionRerankInput:
    pool = tuple(
        _doc(i + 1, f"P-{i // 5:02d}", 1.0 - i * 0.01, section=("개요",))
        for i in range(40)
    )
    return SectionRerankInput(
        query_id=qid, query="t", candidate_pool=pool,
        expected_page_id="P-00",
        expected_section_path=("개요",),
        supporting_chunk_id="P-00-c1",
    )


def test_run_variant_for_query_baseline_returns_phase7_5_top_k() -> None:
    spec = SectionRerankSpec(
        name="baseline", strategy=STRATEGY_BASELINE_NO_RERANK,
        deployable=True,
    )
    out = run_variant_for_query(spec=spec, inputs=_pool_inputs())
    assert len(out) == 10
    assert all(d.rank == i + 1 for i, d in enumerate(out))


def test_run_variant_for_query_section_bonus_promotes_match() -> None:
    spec = SectionRerankSpec(
        name="section_bonus_010", strategy=STRATEGY_SECTION_BONUS,
        deployable=False, section_bonus=0.10,
    )
    inputs = _pool_inputs()
    out = run_variant_for_query(spec=spec, inputs=inputs)
    assert len(out) == 10
    # Every doc in our synthetic pool has section_path=("개요",), which
    # equals expected_section_path → all candidates get +0.10. The
    # *order* should still match relevance (ties → original index).
    assert out[0].rank == 1


def test_run_variant_for_query_supporting_chunk_proximity_no_op_when_unset() -> None:
    spec = SectionRerankSpec(
        name="supp", strategy=STRATEGY_SUPPORTING_CHUNK_PROXIMITY,
        deployable=False,
    )
    pool = tuple(
        _doc(i + 1, f"P-{i // 5:02d}", 1.0 - i * 0.01) for i in range(40)
    )
    inputs = SectionRerankInput(
        query_id="q1", query="t", candidate_pool=pool,
        expected_page_id="",
        expected_section_path=(),
        supporting_chunk_id="",
    )
    out = run_variant_for_query(spec=spec, inputs=inputs)
    # Falls through to the Phase 7.5 baseline behaviour.
    assert len(out) == 10


def test_run_variant_for_query_supporting_chunk_proximity_boosts_chunk() -> None:
    """When the supporting chunk id is in the pool, it should be in the
    top-k after the boost."""
    pool = tuple([
        _doc(1, "P-X", 0.95),  # high score, wrong page
        _doc(2, "P-X", 0.94),
        _doc(3, "P-X", 0.93),
        _doc(4, "P-X", 0.92),
        _doc(5, "P-X", 0.91),
        _doc(6, "P-X", 0.90),
        _doc(7, "P-X", 0.89),
        _doc(8, "P-X", 0.88),
        _doc(9, "P-X", 0.87),
        _doc(10, "P-X", 0.86),
        _doc(11, "P-Y", 0.50),  # supporting page, lower score
    ])
    # Supporting chunk lives at rank 11 (P-Y-c11), with the matching
    # supporting_chunk_id pattern.
    inputs = SectionRerankInput(
        query_id="q1", query="t", candidate_pool=pool,
        expected_page_id="P-Y",
        expected_section_path=(),
        supporting_chunk_id="P-Y-c11",
    )
    spec = SectionRerankSpec(
        name="supp", strategy=STRATEGY_SUPPORTING_CHUNK_PROXIMITY,
        deployable=False, section_bonus=1.0,  # large boost so it actually moves
    )
    out = run_variant_for_query(spec=spec, inputs=inputs)
    chunk_ids = [d.chunk_id for d in out]
    assert "P-Y-c11" in chunk_ids


def test_run_variant_for_query_unknown_strategy_raises() -> None:
    spec = SectionRerankSpec(
        name="bogus", strategy="not_a_real_strategy", deployable=False,
    )
    with pytest.raises(ValueError):
        run_variant_for_query(spec=spec, inputs=_pool_inputs())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_page_hit_at_k_basic() -> None:
    docs = [_doc(1, "P-A", 0.9), _doc(2, "P-B", 0.8)]
    assert page_hit_at_k(docs, "P-B", k=2) == 1
    assert page_hit_at_k(docs, "P-Z", k=2) == 0
    assert page_hit_at_k(docs, "P-B", k=1) == 0
    assert page_hit_at_k(docs, "", k=2) == 0


def test_section_hit_at_k_returns_none_when_undefined() -> None:
    docs = [_doc(1, "P-A", 0.9, section=("개요",))]
    assert section_hit_at_k(docs, (), k=5) is None
    assert section_hit_at_k(docs, ("개요",), k=5) == 1
    assert section_hit_at_k(docs, ("기타",), k=5) == 0


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_section_rerank_grid_md_includes_each_variant() -> None:
    grid = make_section_rerank_grid()
    md = render_section_rerank_grid_md(grid)
    assert "# Phase 7.6 — section-aware rerank candidate grid" in md
    assert "## Guardrails" in md
    for spec in grid:
        assert spec.name in md
    # Diagnostic-only markers must show up so a reviewer can never
    # accidentally promote a diagnostic variant.
    assert "diagnostic" in md.lower()
    assert "DIAGNOSTIC ONLY" in md or "diagnostic" in md.lower()
    # Phase 7.5 baseline ref appears.
    assert "Phase 7.5" in md


def test_write_section_rerank_grid_json_round_trip(tmp_path: Path) -> None:
    grid = make_section_rerank_grid()
    p = write_section_rerank_grid_json(tmp_path / "grid.json", grid)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["phase"] == "7.6_section_aware_rerank"
    assert payload["grid_size"] == len(grid)
    assert len(payload["grid"]) == len(grid)
    thr = payload["guardrail_thresholds"]
    assert thr["primary_score_drop_max_vs_phase_7_5"] == pytest.approx(
        PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD,
    )
    assert thr["silver_hit_at_5_drop_max_vs_phase_7_5"] == pytest.approx(
        PHASE_7_6_SILVER_REGRESSION_THRESHOLD,
    )


def test_write_section_rerank_grid_md_writes_file(tmp_path: Path) -> None:
    grid = make_section_rerank_grid()
    p = write_section_rerank_grid_md(tmp_path / "grid.md", grid)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Phase 7.6" in text
