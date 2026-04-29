"""Eval-only helpers for the wide-MMR confirm sweep.

Sister of ``wide_retrieval_helpers``. The Phase 1 wide-MMR-titlecap
sweep + the Phase 2 5-round Optuna refinement converged on different
recipes; this module provides the spec / grader logic for an out-of-
loop confirm run that scores both recipes (and their immediate
sensitivity neighbours) on the *full* silver_200 set.

Production code is NOT touched. The sweep driver in
``scripts.confirm_wide_mmr_best_configs`` imports the spec list, the
grader, and the per-query diff helper from here so unit tests can pin
the comparison logic without spinning up FAISS / embedder / reranker.

Layout:

  - ``ConfirmCellSpec`` — per-cell wide-MMR knobs + presentation
    metadata (which comparison group the cell belongs to, so the
    markdown can render sub-tables for the sensitivity cohorts).
  - ``DEFAULT_CONFIRM_CELLS`` — the 12-cell roster the spec asks for.
    Layered by group: ``baseline``, ``phase1``, ``optuna_winner``,
    ``cap_final``, ``lambda``, ``final_topk``.
  - ``compute_cell_deltas`` — reuses the Phase 1 grader's epsilon /
    threshold contract (regression at -0.005, promising at +0.01)
    and surfaces deltas + a verdict the report can render.
  - ``decide_verdict`` — final ``KEEP_PHASE1_BEST`` /
    ``ADOPT_OPTUNA_WINNER`` / ``INCONCLUSIVE_REPRESENTATION_BOTTLENECK``
    judgement based on the head-to-head baseline / phase1 / optuna
    deltas. Pure function — testable without retrieval.
  - ``per_query_diff`` — per-query gain/loss accounting so the report
    can name which queries flipped between Phase 1 best and Optuna
    winner without re-implementing the diff inside the markdown
    writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Cell groups — these are stable strings, written into the JSON
# artefacts so a downstream consumer can filter by group without
# re-parsing the cell name.
GROUP_BASELINE = "baseline"
GROUP_PHASE1 = "phase1"
GROUP_OPTUNA_WINNER = "optuna_winner"
GROUP_CAP_FINAL = "cap_final_sensitivity"
GROUP_LAMBDA = "lambda_sensitivity"
GROUP_FINAL_TOPK = "final_topk_sensitivity"

# Final verdict labels — chosen to match the spec's KEEP_PHASE1_BEST /
# ADOPT_OPTUNA_WINNER / INCONCLUSIVE_REPRESENTATION_BOTTLENECK contract.
VERDICT_KEEP_PHASE1 = "KEEP_PHASE1_BEST"
VERDICT_ADOPT_OPTUNA = "ADOPT_OPTUNA_WINNER"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_REPRESENTATION_BOTTLENECK"

# Grade labels mirror the Phase 1 sweep grader so the markdown can be
# read alongside the prior reports without label drift.
GRADE_BASELINE = "baseline"
GRADE_PROMISING = "promising_quality"
GRADE_REGRESSION = "regression"
GRADE_INCONCLUSIVE = "inconclusive"
GRADE_DIAG_ONLY = "diagnostic_only"

# Epsilon / threshold contract — copied verbatim from
# ``eval_wide_mmr_titlecap_sweep._grade_cell``. Epsilons set the
# minimum *meaningful* change; thresholds set the bar for "promising".
EPS_HIT = 0.005
EPS_MRR = 0.005
THRESHOLD_PROMISING_HIT = 0.01
THRESHOLD_PROMISING_MRR = 0.01
LATENCY_RATIO_LIMIT = 1.5


@dataclass(frozen=True)
class ConfirmCellSpec:
    """One wide-MMR config to score in the confirm sweep.

    ``label`` is the artefact key — appears in the CSV / markdown / per-
    query JSONL. ``group`` controls which comparison sub-table the
    markdown writer assigns this cell to. ``description`` is a one-
    liner the markdown surfaces above the headline metric for the cell.

    The wide-retrieval knobs match ``WideRetrievalConfig`` field-for-
    field — the sweep driver re-packs them into a
    ``WideRetrievalConfig`` per cell.
    """

    label: str
    group: str
    description: str
    candidate_k: int
    final_top_k: int
    rerank_in: int
    use_mmr: bool = False
    mmr_lambda: float = 0.65
    mmr_k: int = 48
    title_cap_rerank_input: Optional[int] = None
    title_cap_final: Optional[int] = None


def default_confirm_cells() -> List[ConfirmCellSpec]:
    """Return the 12-cell roster called out by the confirm spec.

    Stable order — the markdown writer surfaces cells in this order
    for the report. New cells should be appended, not interleaved,
    so prior artefacts diff cleanly when this list grows.
    """
    return [
        # A: historical / current baseline (Phase 1 baseline cell).
        ConfirmCellSpec(
            label="baseline_k50_top5",
            group=GROUP_BASELINE,
            description=(
                "Phase 1 / production-equivalent baseline — "
                "candidate_k=50, no MMR, no title cap, rerank_in=32, "
                "final_top_k=5."
            ),
            candidate_k=50, final_top_k=5, rerank_in=32,
        ),
        # B: Phase 1 promising winner (cap=2, top_k=8).
        ConfirmCellSpec(
            label="phase1_best_cap2_top8",
            group=GROUP_PHASE1,
            description=(
                "Phase 1 wide-MMR-titlecap promising_quality cell — "
                "candidate_k=200, MMR λ=0.65, mmr_k=64, cap_rr=2, "
                "cap_final=2, rerank_in=32, top_k=8."
            ),
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.65, mmr_k=64,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
        # C: Phase 1 cap=1 / top_k=8 alternative.
        ConfirmCellSpec(
            label="phase1_cap1_top8",
            group=GROUP_PHASE1,
            description=(
                "Phase 1 cap=1 alternative — same recipe as cap=2 but "
                "title_cap_rerank_input=1 and title_cap_final=1; tied "
                "on the diagnostic 200-row run."
            ),
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.65, mmr_k=64,
            title_cap_rerank_input=1, title_cap_final=1,
        ),
        # D: Optuna winner — cand_k=100, rerank_in=16, cap_rr=1.
        ConfirmCellSpec(
            label="optuna_winner_top8",
            group=GROUP_OPTUNA_WINNER,
            description=(
                "Optuna 5-round winner — candidate_k=100, rerank_in=16, "
                "MMR λ=0.65, mmr_k=48, cap_rr=1, cap_final=2, top_k=8. "
                "Found best on first 100 rows; this run validates on the "
                "full 200."
            ),
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.65, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        # E: title_cap_final sensitivity (cap_final=1 and cap_final=3;
        # cap_final=2 is the winner cell above).
        ConfirmCellSpec(
            label="optuna_winner_top8_capfinal1",
            group=GROUP_CAP_FINAL,
            description=(
                "Winner recipe with title_cap_final=1 — checks if a "
                "stricter final cap moves quality on the full 200."
            ),
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.65, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=1,
        ),
        ConfirmCellSpec(
            label="optuna_winner_top8_capfinal3",
            group=GROUP_CAP_FINAL,
            description=(
                "Winner recipe with title_cap_final=3 — checks if a "
                "looser final cap moves quality on the full 200."
            ),
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.65, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=3,
        ),
        # F: mmr_lambda sensitivity (0.55 / 0.60 / 0.70 / 0.75).
        ConfirmCellSpec(
            label="optuna_winner_top8_lambda055",
            group=GROUP_LAMBDA,
            description=(
                "Winner recipe with MMR λ=0.55 — UNSAMPLED edge in the "
                "Optuna study; confirm whether the lower edge is safe."
            ),
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.55, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        ConfirmCellSpec(
            label="optuna_winner_top8_lambda060",
            group=GROUP_LAMBDA,
            description="Winner recipe with MMR λ=0.60.",
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.60, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        ConfirmCellSpec(
            label="optuna_winner_top8_lambda070",
            group=GROUP_LAMBDA,
            description="Winner recipe with MMR λ=0.70.",
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.70, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        ConfirmCellSpec(
            label="optuna_winner_top8_lambda075",
            group=GROUP_LAMBDA,
            description=(
                "Winner recipe with MMR λ=0.75 — UNSAMPLED edge in "
                "round_05; confirm whether the upper edge is safe."
            ),
            candidate_k=100, final_top_k=8, rerank_in=16,
            use_mmr=True, mmr_lambda=0.75, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        # G: final_top_k sensitivity (5 / 10; 8 is the winner cell).
        ConfirmCellSpec(
            label="optuna_winner_top5",
            group=GROUP_FINAL_TOPK,
            description=(
                "Winner recipe with final_top_k=5 — alignment with the "
                "Phase 1 baseline top_k for direct hit@5 comparison."
            ),
            candidate_k=100, final_top_k=5, rerank_in=16,
            use_mmr=True, mmr_lambda=0.65, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
        ConfirmCellSpec(
            label="optuna_winner_top10",
            group=GROUP_FINAL_TOPK,
            description=(
                "Winner recipe with final_top_k=10 — checks whether top "
                "10 chunks change MRR / nDCG measurably."
            ),
            candidate_k=100, final_top_k=10, rerank_in=16,
            use_mmr=True, mmr_lambda=0.65, mmr_k=48,
            title_cap_rerank_input=1, title_cap_final=2,
        ),
    ]


# Stable spec list aliased for tests / external callers.
DEFAULT_CONFIRM_CELLS: Tuple[ConfirmCellSpec, ...] = tuple(default_confirm_cells())


def _delta_or_none(
    curr: Optional[float], base: Optional[float],
) -> Optional[float]:
    if curr is None or base is None:
        return None
    return round(float(curr) - float(base), 6)


def _ratio_or_none(
    curr: Optional[float], base: Optional[float],
) -> Optional[float]:
    if curr is None or base is None or float(base) <= 0:
        return None
    return round(float(curr) / float(base), 4)


@dataclass
class CellDeltas:
    """All deltas a confirm cell carries vs the baseline.

    Tuned for the report's needs: header metrics, candidate / dup /
    latency aux, plus a grade string the markdown writer can echo.
    """

    label: str
    group: str
    grade: str
    reason: str
    delta_hit_at_1: Optional[float]
    delta_hit_at_3: Optional[float]
    delta_hit_at_5: Optional[float]
    delta_mrr_at_10: Optional[float]
    delta_ndcg_at_10: Optional[float]
    delta_candidate_hit_at_50: Optional[float]
    delta_candidate_hit_at_100: Optional[float]
    delta_duplicate_ratio_at_10: Optional[float]
    delta_unique_doc_count_at_10: Optional[float]
    delta_p50ms: Optional[float]
    delta_p95ms: Optional[float]
    delta_p99ms: Optional[float]
    latency_ratio_p95: Optional[float]
    rerank_uplift_hit_at_5: Optional[float]
    rerank_uplift_mrr_at_10: Optional[float]


def _read_metric(
    summary: Mapping[str, Any] | Any, key: str,
) -> Optional[float]:
    """Pull a metric off a summary in dict-or-dataclass form."""
    if summary is None:
        return None
    if isinstance(summary, Mapping):
        v = summary.get(key)
    else:
        v = getattr(summary, key, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_dict_metric(
    summary: Mapping[str, Any] | Any,
    field_name: str,
    key: str,
) -> Optional[float]:
    """Pull a nested numeric metric out of a dict-on-dataclass field."""
    if summary is None:
        return None
    if isinstance(summary, Mapping):
        nested = summary.get(field_name) or {}
    else:
        nested = getattr(summary, field_name, None) or {}
    if not isinstance(nested, Mapping):
        return None
    v = nested.get(str(key))
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_cell_deltas(
    *,
    label: str,
    group: str,
    cell_summary: Any,
    baseline_summary: Any,
) -> CellDeltas:
    """Compute the per-cell deltas + grade vs the baseline cell.

    Mirrors ``eval_wide_mmr_titlecap_sweep._grade_cell`` so a confirm
    report grade reads consistently with the Phase 1 grade. The
    baseline cell is always self-graded ``GRADE_BASELINE``.
    """
    if label == "baseline_k50_top5" or cell_summary is baseline_summary:
        # Self-grade: zero deltas, latency ratio 1.0.
        return CellDeltas(
            label=label,
            group=group,
            grade=GRADE_BASELINE,
            reason="reference",
            delta_hit_at_1=0.0,
            delta_hit_at_3=0.0,
            delta_hit_at_5=0.0,
            delta_mrr_at_10=0.0,
            delta_ndcg_at_10=0.0,
            delta_candidate_hit_at_50=0.0,
            delta_candidate_hit_at_100=0.0,
            delta_duplicate_ratio_at_10=0.0,
            delta_unique_doc_count_at_10=0.0,
            delta_p50ms=0.0,
            delta_p95ms=0.0,
            delta_p99ms=0.0,
            latency_ratio_p95=1.0,
            rerank_uplift_hit_at_5=_read_metric(
                cell_summary, "rerank_uplift_hit_at_5",
            ),
            rerank_uplift_mrr_at_10=_read_metric(
                cell_summary, "rerank_uplift_mrr_at_10",
            ),
        )

    d_h1 = _delta_or_none(
        _read_metric(cell_summary, "mean_hit_at_1"),
        _read_metric(baseline_summary, "mean_hit_at_1"),
    )
    d_h3 = _delta_or_none(
        _read_metric(cell_summary, "mean_hit_at_3"),
        _read_metric(baseline_summary, "mean_hit_at_3"),
    )
    d_h5 = _delta_or_none(
        _read_metric(cell_summary, "mean_hit_at_5"),
        _read_metric(baseline_summary, "mean_hit_at_5"),
    )
    d_mrr = _delta_or_none(
        _read_metric(cell_summary, "mean_mrr_at_10"),
        _read_metric(baseline_summary, "mean_mrr_at_10"),
    )
    d_ndcg = _delta_or_none(
        _read_metric(cell_summary, "mean_ndcg_at_10"),
        _read_metric(baseline_summary, "mean_ndcg_at_10"),
    )
    d_cand50 = _delta_or_none(
        _read_dict_metric(cell_summary, "candidate_hit_rates", "50"),
        _read_dict_metric(baseline_summary, "candidate_hit_rates", "50"),
    )
    d_cand100 = _delta_or_none(
        _read_dict_metric(cell_summary, "candidate_hit_rates", "100"),
        _read_dict_metric(baseline_summary, "candidate_hit_rates", "100"),
    )
    d_dup10 = _delta_or_none(
        _read_dict_metric(cell_summary, "duplicate_doc_ratios", "10"),
        _read_dict_metric(baseline_summary, "duplicate_doc_ratios", "10"),
    )
    d_uniq10 = _delta_or_none(
        _read_dict_metric(cell_summary, "unique_doc_counts", "10"),
        _read_dict_metric(baseline_summary, "unique_doc_counts", "10"),
    )

    p50_curr = _read_metric(cell_summary, "p50_retrieval_ms")
    p50_base = _read_metric(baseline_summary, "p50_retrieval_ms")
    p95_curr = (
        _read_metric(cell_summary, "p95_total_retrieval_ms")
        or _read_metric(cell_summary, "p95_retrieval_ms")
    )
    p95_base = (
        _read_metric(baseline_summary, "p95_total_retrieval_ms")
        or _read_metric(baseline_summary, "p95_retrieval_ms")
    )
    p99_curr = _read_metric(cell_summary, "p99_retrieval_ms")
    p99_base = _read_metric(baseline_summary, "p99_retrieval_ms")
    d_p50 = _delta_or_none(p50_curr, p50_base)
    d_p95 = _delta_or_none(p95_curr, p95_base)
    d_p99 = _delta_or_none(p99_curr, p99_base)
    latency_ratio = _ratio_or_none(p95_curr, p95_base)

    if (
        (d_h5 is not None and d_h5 <= -EPS_HIT)
        or (d_mrr is not None and d_mrr <= -EPS_MRR)
        or (d_cand50 is not None and d_cand50 <= -EPS_HIT)
    ):
        grade = GRADE_REGRESSION
        reason = (
            f"Δhit@5={d_h5} Δmrr={d_mrr} Δcand@50={d_cand50}"
        )
    elif (
        (d_h5 is not None and d_h5 >= THRESHOLD_PROMISING_HIT)
        or (d_mrr is not None and d_mrr >= THRESHOLD_PROMISING_MRR)
    ):
        if (
            latency_ratio is not None
            and latency_ratio > LATENCY_RATIO_LIMIT
        ):
            grade = GRADE_DIAG_ONLY
            reason = (
                f"quality up (Δhit@5={d_h5}, Δmrr={d_mrr}) but "
                f"latency ratio {latency_ratio:.2f}x > "
                f"{LATENCY_RATIO_LIMIT}x"
            )
        else:
            grade = GRADE_PROMISING
            reason = (
                f"Δhit@5={d_h5} Δmrr={d_mrr} latency_ratio="
                f"{latency_ratio}"
            )
    elif (
        (d_h5 is None or abs(d_h5) < EPS_HIT)
        and (d_mrr is None or abs(d_mrr) < EPS_MRR)
    ):
        grade = GRADE_INCONCLUSIVE
        reason = (
            f"deltas within epsilon (Δhit@5={d_h5}, Δmrr={d_mrr})"
        )
    else:
        grade = GRADE_DIAG_ONLY
        reason = (
            f"Δhit@5={d_h5} Δmrr={d_mrr} Δcand@50={d_cand50} "
            f"latency_ratio={latency_ratio}"
        )

    return CellDeltas(
        label=label,
        group=group,
        grade=grade,
        reason=reason,
        delta_hit_at_1=d_h1,
        delta_hit_at_3=d_h3,
        delta_hit_at_5=d_h5,
        delta_mrr_at_10=d_mrr,
        delta_ndcg_at_10=d_ndcg,
        delta_candidate_hit_at_50=d_cand50,
        delta_candidate_hit_at_100=d_cand100,
        delta_duplicate_ratio_at_10=d_dup10,
        delta_unique_doc_count_at_10=d_uniq10,
        delta_p50ms=d_p50,
        delta_p95ms=d_p95,
        delta_p99ms=d_p99,
        latency_ratio_p95=latency_ratio,
        rerank_uplift_hit_at_5=_read_metric(
            cell_summary, "rerank_uplift_hit_at_5",
        ),
        rerank_uplift_mrr_at_10=_read_metric(
            cell_summary, "rerank_uplift_mrr_at_10",
        ),
    )


def decide_verdict(
    *,
    baseline_summary: Any,
    phase1_best_summary: Any,
    optuna_winner_summary: Any,
    latency_ratio_phase1: Optional[float] = None,
    latency_ratio_optuna: Optional[float] = None,
) -> Tuple[str, str]:
    """Return ``(verdict, rationale)`` over the head-to-head comparison.

    Logic:

      1. If both phase1_best and optuna_winner regress vs baseline on
         hit@5 AND MRR (both ≤ -EPS), → INCONCLUSIVE_REPRESENTATION.
      2. Else, if optuna_winner ≥ phase1_best on MRR by ≥ EPS *and*
         doesn't regress on hit@5 by more than EPS, *and* its latency
         ratio is within ``LATENCY_RATIO_LIMIT`` of baseline, →
         ADOPT_OPTUNA_WINNER.
      3. Else, if phase1_best ≥ optuna_winner on MRR by ≥ EPS, →
         KEEP_PHASE1_BEST.
      4. Else (deltas inside epsilon both ways) →
         INCONCLUSIVE_REPRESENTATION_BOTTLENECK — no statistically
         compelling differentiation; defer to the next-stage
         embedding text variant work.

    The verdict is **conservative** — it never says ADOPT unless
    Optuna wins on MRR. The rationale string surfaces the key deltas
    so the report can render them inline.
    """
    base_h5 = _read_metric(baseline_summary, "mean_hit_at_5")
    base_mrr = _read_metric(baseline_summary, "mean_mrr_at_10")
    p1_h5 = _read_metric(phase1_best_summary, "mean_hit_at_5")
    p1_mrr = _read_metric(phase1_best_summary, "mean_mrr_at_10")
    op_h5 = _read_metric(optuna_winner_summary, "mean_hit_at_5")
    op_mrr = _read_metric(optuna_winner_summary, "mean_mrr_at_10")

    delta_p1_h5 = _delta_or_none(p1_h5, base_h5)
    delta_p1_mrr = _delta_or_none(p1_mrr, base_mrr)
    delta_op_h5 = _delta_or_none(op_h5, base_h5)
    delta_op_mrr = _delta_or_none(op_mrr, base_mrr)
    delta_op_vs_p1_h5 = _delta_or_none(op_h5, p1_h5)
    delta_op_vs_p1_mrr = _delta_or_none(op_mrr, p1_mrr)

    # Case 1: both regress vs baseline.
    p1_regresses = (
        (delta_p1_h5 is not None and delta_p1_h5 <= -EPS_HIT)
        and (delta_p1_mrr is not None and delta_p1_mrr <= -EPS_MRR)
    )
    op_regresses = (
        (delta_op_h5 is not None and delta_op_h5 <= -EPS_HIT)
        and (delta_op_mrr is not None and delta_op_mrr <= -EPS_MRR)
    )
    if p1_regresses and op_regresses:
        return (
            VERDICT_INCONCLUSIVE,
            (
                f"Both phase1_best and optuna_winner regress vs baseline "
                f"on hit@5 AND MRR (Δp1_h5={delta_p1_h5}, Δp1_mrr="
                f"{delta_p1_mrr}, Δop_h5={delta_op_h5}, Δop_mrr="
                f"{delta_op_mrr}). The bottleneck is upstream — "
                "embedding/chunk/representation, not MMR/cap tuning."
            ),
        )

    op_latency_within = (
        latency_ratio_optuna is None
        or latency_ratio_optuna <= LATENCY_RATIO_LIMIT
    )

    # Case 2: optuna_winner clearly better than phase1_best.
    if (
        delta_op_vs_p1_mrr is not None
        and delta_op_vs_p1_mrr >= EPS_MRR
        and (
            delta_op_vs_p1_h5 is None or delta_op_vs_p1_h5 >= -EPS_HIT
        )
        and op_latency_within
    ):
        return (
            VERDICT_ADOPT_OPTUNA,
            (
                f"optuna_winner ≥ phase1_best on MRR by "
                f"{delta_op_vs_p1_mrr} (≥ EPS_MRR={EPS_MRR}); hit@5 "
                f"delta {delta_op_vs_p1_h5} not regressing; latency "
                f"ratio vs baseline {latency_ratio_optuna} ≤ "
                f"{LATENCY_RATIO_LIMIT}x. Adoption-eligible."
            ),
        )

    # Case 3: phase1_best clearly better than optuna_winner.
    if (
        delta_op_vs_p1_mrr is not None
        and delta_op_vs_p1_mrr <= -EPS_MRR
    ):
        return (
            VERDICT_KEEP_PHASE1,
            (
                f"phase1_best ≥ optuna_winner on MRR by "
                f"{-delta_op_vs_p1_mrr} (≥ EPS_MRR={EPS_MRR}). The "
                "Optuna winner from the 100-row subset doesn't transfer "
                "to the full 200-row set."
            ),
        )

    # Case 4: tie within epsilon — inconclusive.
    return (
        VERDICT_INCONCLUSIVE,
        (
            f"phase1_best vs optuna_winner deltas inside epsilon "
            f"(ΔMRR={delta_op_vs_p1_mrr}, Δhit@5={delta_op_vs_p1_h5}). "
            "Neither recipe carries enough signal on the full 200-row "
            "set to displace the other; the next bottleneck is upstream "
            "(embedding text variant / candidate representation), not "
            "MMR/cap tuning."
        ),
    )


# ---------------------------------------------------------------------------
# Per-query diff helper — used by the markdown writer to surface query
# lists for "improved vs baseline" / "regressed vs baseline" / "phase1
# vs optuna split". Pure dict logic, no dataclass round-trip needed.
# ---------------------------------------------------------------------------


def _row_hit_at_5(row: Mapping[str, Any]) -> Optional[float]:
    v = row.get("hit_at_5")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class PerQueryDiffEntry:
    """One per-query crossover entry for the markdown writer.

    ``baseline_hit_at_5`` / ``cell_hit_at_5`` are the row-level hit@5
    for the two cells being diffed; ``flip_direction`` is one of
    ``"improved"`` (baseline=0, cell=1) or ``"regressed"`` (baseline=1,
    cell=0). Other shapes (both 0 or both 1) are excluded by
    construction so the caller can iterate the entries directly.
    """

    cell: str
    id: str
    query: str
    baseline_hit_at_5: float
    cell_hit_at_5: float
    flip_direction: str
    expected_doc_ids: List[str] = field(default_factory=list)
    cell_top_doc_ids: List[str] = field(default_factory=list)
    baseline_top_doc_ids: List[str] = field(default_factory=list)


def per_query_diff(
    baseline_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    *,
    cell_label: str,
) -> Tuple[List[PerQueryDiffEntry], List[PerQueryDiffEntry]]:
    """Return ``(improvements, regressions)`` between baseline and cell.

    Only rows where both sides have a populated ``hit_at_5`` are
    considered — rows without expected_doc_ids are excluded since their
    hit@5 is ``None`` by construction.
    """
    base_by_id = {str(r.get("id")): r for r in baseline_rows if r.get("id")}
    improvements: List[PerQueryDiffEntry] = []
    regressions: List[PerQueryDiffEntry] = []
    for row in cell_rows:
        rid = str(row.get("id") or "")
        if not rid or rid not in base_by_id:
            continue
        base = base_by_id[rid]
        b_h5 = _row_hit_at_5(base)
        c_h5 = _row_hit_at_5(row)
        if b_h5 is None or c_h5 is None:
            continue
        # Only flag genuine flips — both 0 or both 1 doesn't surface.
        if c_h5 > 0.5 and b_h5 <= 0.5:
            improvements.append(PerQueryDiffEntry(
                cell=cell_label,
                id=rid,
                query=str(row.get("query") or ""),
                baseline_hit_at_5=b_h5,
                cell_hit_at_5=c_h5,
                flip_direction="improved",
                expected_doc_ids=list(row.get("expected_doc_ids") or []),
                cell_top_doc_ids=list(
                    (row.get("retrieved_doc_ids") or [])[:5]
                ),
                baseline_top_doc_ids=list(
                    (base.get("retrieved_doc_ids") or [])[:5]
                ),
            ))
        elif b_h5 > 0.5 and c_h5 <= 0.5:
            regressions.append(PerQueryDiffEntry(
                cell=cell_label,
                id=rid,
                query=str(row.get("query") or ""),
                baseline_hit_at_5=b_h5,
                cell_hit_at_5=c_h5,
                flip_direction="regressed",
                expected_doc_ids=list(row.get("expected_doc_ids") or []),
                cell_top_doc_ids=list(
                    (row.get("retrieved_doc_ids") or [])[:5]
                ),
                baseline_top_doc_ids=list(
                    (base.get("retrieved_doc_ids") or [])[:5]
                ),
            ))
    return improvements, regressions


def candidate_pool_recoverable_misses(
    cell_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Find queries where the gold doc is in the candidate pool but
    the reranker drops it from the final top-k.

    Returns one entry per such query with cell label, query text,
    expected doc_ids, and the truncated final top doc_ids — the
    markdown writer uses this for the "정답 후보가 candidate pool에는
    있으나 reranker 후 밀리는 query" section the spec asks for.
    """
    out: List[Dict[str, Any]] = []
    for row in cell_rows:
        expected = [str(d) for d in (row.get("expected_doc_ids") or []) if d]
        if not expected:
            continue
        retrieved = [
            str(d) for d in (row.get("retrieved_doc_ids") or []) if d
        ]
        candidates = [
            str(d) for d in (row.get("candidate_doc_ids") or []) if d
        ]
        retrieved_set = set(retrieved)
        candidate_set = set(candidates)
        if any(d in candidate_set and d not in retrieved_set for d in expected):
            out.append({
                "id": str(row.get("id") or ""),
                "query": str(row.get("query") or ""),
                "expected_doc_ids": expected,
                "retrieved_top_doc_ids": retrieved[:5],
                "candidate_count": len(candidates),
            })
    return out
