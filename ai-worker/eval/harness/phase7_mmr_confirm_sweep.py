"""Phase 7.5 — MMR confirm sweep harness.

Phase 7.x's first pass landed on a *single* candidate winner
(``cand_top10_mmr_lambda07``: candidate_k=30, use_mmr=True, λ=0.7) on top
of the production-default ``retrieval-title-section`` index. Before
promoting that to production we want to confirm the win is a stable
plateau and not a single-point peak — pick the wrong λ and the gain
disappears, or worse, a different bucket regresses.

This module is the *confirm sweep* layer on top of
``phase7_human_gold_tune``:

  * **Grid generation** — a small ``(candidate_k × mmr_lambda)`` grid
    around the previous best, all on the *same* index variant the
    Phase 7.0–7.4 work used. The grid is intentionally narrow: 3
    candidate_k values × 5 λ values = 15 candidates. Wider sweeps
    belong in a separate experiment.

  * **Selection rule** — gold ``primary_score`` improvement is necessary
    but not sufficient. A candidate must also hold the silver hit@5
    guardrail, hold ``subpage_named`` (the bucket the gold-50 set was
    curated to fix), and not collapse the ``main_work`` bucket. If the
    candidate fails any of those, it is rejected before the
    primary-score ranking even runs.

  * **Plateau analysis** — for the chosen winner we look at its λ-grid
    neighbours: if they sit within an epsilon of the best primary_score
    we call it a plateau, otherwise we flag overfit risk. The point is
    to avoid the "λ=0.7 is +6pp but λ=0.65 and λ=0.75 are -3pp" case,
    which would be a single-point peak masquerading as a stable config.

  * **Section hit caveat** — Phase 7.x measured a section_hit@5 drop
    from 4.5pp to 2.3pp when MMR turned on. We carry that as a
    documented caveat (page-level retrieval improved, section-level
    exact match fell on a tiny base of 4.5pp). It is *not* a promotion
    blocker, but the report writer surfaces it explicitly so the
    reviewer knows to schedule a section-aware reranking experiment
    next.

  * **Promotion target clarification** — Phase 7.0–7.2 already promoted
    the ``retrieval-title-section`` index; the change being scored here
    is a retrieval *config* change (candidate_k + MMR), NOT another
    embedding-text variant promotion. The report writer makes that
    explicit so a reviewer can never confuse the two — the rejected
    ``cand_title_section_top10`` was a sanity check on the old index,
    not a candidate for promotion.

All scoring functions are pure: they take ``GoldSummary`` /
``SilverSummary`` / ``RetrievedDoc`` lists and return rankings. The
"actually run the retriever" wiring lives in
``scripts.run_phase7_mmr_confirm_sweep`` so tests can exercise the
selection logic without touching FAISS / bge-m3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from eval.harness.phase7_human_gold_tune import (
    GROUP_ABSTAIN_TEST,
    GROUP_AMBIGUOUS_PROBE,
    HUMAN_FOCUS_DISCLAIMER,
    POSITIVE_GROUPS,
    SILVER_HIT_AT_5_REGRESSION_THRESHOLD,
    SILVER_BUCKET_REGRESSION_THRESHOLD,
    SILVER_BUCKET_FOR_NAMED_GUARDRAIL,
    GoldSummary,
    GuardrailWarning,
    RetrievedDoc,
    SilverSummary,
    VariantResult,
    evaluate_silver_guardrail,
    gold_summary_to_dict,
    silver_summary_to_dict,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid configuration — narrow on purpose, see module docstring.
# ---------------------------------------------------------------------------


# The previous Phase 7.x best was candidate_k=30 / λ=0.7. The confirm
# sweep covers ±step around both axes so a contributor can see whether
# the win is a plateau or a single-point peak. Adding more axes (top_k,
# index variant, reranker) is out of scope here — that's a wider Phase 8
# experiment.
DEFAULT_CANDIDATE_K_GRID: Tuple[int, ...] = (20, 30, 40)
DEFAULT_MMR_LAMBDA_GRID: Tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_TOP_K: int = 10
# The doc_id-overlap penalty inside MMR. Production ``_mmr_select`` uses
# 0.6; the harness re-applies that constant when scoring candidates from
# pre-computed embeddings so post-hoc replay matches the live path bit-
# for-bit on the same input.
MMR_DOC_ID_PENALTY: float = 0.6


# Cache dir / rag_chunks defaults — pinned to the production-default
# index (``retrieval-title-section-mseq512``) so the confirm sweep can
# only vary the retrieval-config knobs, not the embedding-text variant.
PRODUCTION_INDEX_CACHE_DIR: str = (
    "namu-v4-2008-2026-04-retrieval-title-section-mseq512"
)
PRODUCTION_RAG_CHUNKS_PATH: str = (
    "eval/reports/phase7/silver500/retrieval/"
    "rag_chunks_retrieval_title_section.jsonl"
)


# ---------------------------------------------------------------------------
# Confirm-sweep-specific guardrail thresholds.
# ---------------------------------------------------------------------------


# main_work weighted_hit@5 must not drop more than 5pp vs baseline. The
# gold-50 main_work bucket is the most "general" slice — if MMR helps
# subpage_named at the cost of main_work we need to know.
MAIN_WORK_REGRESSION_THRESHOLD: float = 0.05
# subpage_named weighted_hit@5 must NOT regress vs baseline. The whole
# point of the gold-50 set is to fix subpage / named-subpage failures,
# so a candidate that improves primary_score by trading subpage_named
# is not actually solving the problem it was curated for.
SUBPAGE_NAMED_HOLD_THRESHOLD: float = 0.0
# section_hit@5 caveat: warn (not block) when section_hit@5 drops more
# than half. baseline section_hit@5 is tiny (~4.5pp) so half is the
# tightest meaningful threshold.
SECTION_HIT_HALVING_FACTOR: float = 0.5
# Plateau analysis: a candidate's λ-neighbour primary_score must be
# within this absolute delta to count as a plateau. 0.02 is one-third
# of the previous Phase 7.x best gain (~0.062), large enough to absorb
# the noise of a 50-row gold set but tight enough to flag a single-
# point peak.
PLATEAU_PRIMARY_DELTA: float = 0.02


# Warning codes — add to the existing SILVER_/BUCKET_ codes from the
# base harness. The report writer keys off the code string, so do not
# rename these without updating the renderer test.
SUBPAGE_NAMED_NOT_FIXED_WARNING: str = "SUBPAGE_NAMED_NOT_FIXED_WARNING"
MAIN_WORK_REGRESSION_WARNING: str = "MAIN_WORK_REGRESSION_WARNING"
SECTION_RETRIEVAL_WARNING: str = "SECTION_RETRIEVAL_WARNING"
PLATEAU_OK: str = "PLATEAU_OK"
PLATEAU_OVERFIT_WARNING: str = "PLATEAU_OVERFIT_WARNING"


# ---------------------------------------------------------------------------
# Promotion target clarification + section caveat — pinned by the test
# suite so the language stays stable across re-runs of the harness.
# ---------------------------------------------------------------------------


PROMOTION_TARGET_CLARIFICATION: str = (
    "Promotion target: this evaluation tests retrieval CONFIG changes "
    "(candidate_k, use_mmr, mmr_lambda) on top of the production-default "
    "retrieval_title_section index. It does NOT test another embedding-"
    "text variant promotion. The rejected `cand_title_section_top10` "
    "candidate is the *previous* (Phase 7.0) embedding-text variant and "
    "is included only as a sanity check that retrieval_title_section is "
    "still the right index choice — its large regression on both gold "
    "and silver confirms that decision and is not part of this "
    "promotion proposal."
)


# ---------------------------------------------------------------------------
# Production recommendation — plateau-aware lambda policy on top of the
# metric-best winner. The metric-best λ that lexicographic tie-break
# picks may not be the best PR target: when the entire λ-grid plateaus
# at the same primary_score, the tie-break value is the *first* one in
# alphabetical / numeric order, which says nothing about production
# fitness. A sweep that lands on a flat plateau should be promoted at
# the λ that's already shipped (or closest to it) so the resulting PR
# is a single-knob flip ("turn MMR on, widen candidate_k") rather than
# a "and λ moved 0.10 too" change that costs extra explanation.
# ---------------------------------------------------------------------------


# Default production-recommended lambda when the metric-best is on a
# plateau. 0.70 was the previous Phase 7.x first-pass best; staying at
# 0.70 keeps the PR consistent with the prior recommendation and lowers
# the explanation cost vs the lexicographic tie-break value (typically
# the lowest λ in the plateau row).
PRODUCTION_RECOMMENDED_LAMBDA: float = 0.70


# Lambda-policy strings written into the production-recommended JSON
# under ``selected_lambda_policy``. The CLI / report renderer keys off
# the string, so do not rename without also bumping the test suite.
LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST: str = "PLATEAU_TIE_BREAK_TO_PREVIOUS_BEST"
LAMBDA_POLICY_PLATEAU_NEAREST: str = "PLATEAU_TIE_BREAK_NEAREST"
LAMBDA_POLICY_NO_PLATEAU_FALLBACK: str = "NO_PLATEAU_FALLBACK_TO_METRIC_BEST"


# Plateau detection epsilon for the production-recommendation step.
# Keep separate from PLATEAU_PRIMARY_DELTA so a future contributor can
# use a tighter or looser threshold for "is this PR-target safe?" than
# the one used to flag overfit risk in the report.
PRODUCTION_PLATEAU_EPSILON: float = PLATEAU_PRIMARY_DELTA


SECTION_HIT_CAVEAT: str = (
    "Section_hit@5 caveat: page-level retrieval is meaningfully better "
    "under MMR; section-level exact-match fell relative to baseline. "
    "Baseline section_hit@5 is tiny (≈4.5pp on a 22-row defined-only "
    "subset), which makes the metric brittle: any reordering of "
    "neighbouring chunks within the same page can flip section_hit "
    "without affecting answer quality. We therefore document the drop "
    "but do NOT treat it as a promotion blocker. Section-aware reranking "
    "or chunk-level generation audit is the right follow-up to validate "
    "this assumption."
)


# Tokens the report writer must NOT emit — pinned by a regression test.
# Anything in the comparison_report.md or confirm_sweep_report.md that
# matches one of these strings means the writer accidentally framed the
# rejected ``cand_title_section_top10`` candidate as the promotion
# target. Update only when you have a concrete reason — the test fails
# loudly otherwise.
FORBIDDEN_PROMOTION_TARGET_PHRASES: Tuple[str, ...] = (
    "Adopt `cand_title_section_top10`",
    "promote cand_title_section_top10",
    "promote `cand_title_section_top10`",
    "cand_title_section_top10 should be promoted",
    "cand_title_section_top10 result justified",
    "cand_title_section_top10 justified the retrieval_title_section "
    "promotion",
    # Older accidental wording: "title_section variant is the recommended "
    # "production config".
    "title_section variant is the recommended production",
)


# ---------------------------------------------------------------------------
# Sweep candidate spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCandidate:
    """One candidate in the (candidate_k × mmr_lambda) confirm grid.

    All fields except ``name`` and ``description`` come from the grid
    axes; ``name`` is built deterministically so a contributor reading
    the report can map ``cand_candk30_mmr_lambda065`` straight back to
    the grid coordinates without consulting the manifest.
    """

    name: str
    candidate_k: int
    mmr_lambda: float
    top_k: int = DEFAULT_TOP_K
    use_mmr: bool = True
    cache_dir_relative: str = PRODUCTION_INDEX_CACHE_DIR
    rag_chunks_path_relative: str = PRODUCTION_RAG_CHUNKS_PATH
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _format_lambda(value: float) -> str:
    """Format a lambda value for the variant name — drops the decimal point.

    ``0.65 → "065"``, ``0.7 → "070"``. Keeps names file-safe and lets
    the report cells stay narrow.
    """
    return f"{int(round(float(value) * 100)):03d}"


def make_confirm_sweep_grid(
    *,
    candidate_ks: Sequence[int] = DEFAULT_CANDIDATE_K_GRID,
    mmr_lambdas: Sequence[float] = DEFAULT_MMR_LAMBDA_GRID,
    top_k: int = DEFAULT_TOP_K,
    cache_dir_relative: str = PRODUCTION_INDEX_CACHE_DIR,
    rag_chunks_path_relative: str = PRODUCTION_RAG_CHUNKS_PATH,
) -> List[SweepCandidate]:
    """Build the (candidate_k × mmr_lambda) confirm-sweep grid.

    Order: outer loop candidate_k, inner loop mmr_lambda. The order
    matters because the renderer prints the grid as a table indexed by
    the same axes; a contributor adding a dimension here must remember
    to update the renderer.
    """
    out: List[SweepCandidate] = []
    for ck in candidate_ks:
        for lam in mmr_lambdas:
            name = f"cand_candk{int(ck):02d}_mmr_lambda{_format_lambda(lam)}"
            desc = (
                f"candidate_k={int(ck)} + MMR(λ={float(lam):.2f}) on "
                f"{cache_dir_relative}; top_k={int(top_k)}."
            )
            out.append(SweepCandidate(
                name=name,
                candidate_k=int(ck),
                mmr_lambda=float(lam),
                top_k=int(top_k),
                use_mmr=True,
                cache_dir_relative=cache_dir_relative,
                rag_chunks_path_relative=rag_chunks_path_relative,
                description=desc,
            ))
    return out


# ---------------------------------------------------------------------------
# Post-hoc MMR — runs over a pre-computed candidate pool.
# ---------------------------------------------------------------------------


def mmr_select_post_hoc(
    candidates: Sequence[RetrievedDoc],
    *,
    top_k: int,
    mmr_lambda: float,
    doc_id_penalty: float = MMR_DOC_ID_PENALTY,
) -> List[RetrievedDoc]:
    """Pure-Python MMR over a pre-computed candidate pool of RetrievedDocs.

    Mirrors :func:`app.capabilities.rag.retriever._mmr_select` exactly
    (same penalty, same value formula, same first-pick tiebreak), but
    operates on the harness's lightweight :class:`RetrievedDoc` so the
    test suite can exercise the selector without instantiating any
    production retriever / chunk types.

    Used by both the live mode (apply MMR to the wide candidate pool we
    cached once per query) and the test suite (drive the selector with
    synthetic candidates).
    """
    k = max(0, int(top_k))
    if k == 0 or not candidates:
        return []

    remaining: List[RetrievedDoc] = list(candidates)
    selected: List[RetrievedDoc] = []
    selected_doc_ids: set[str] = set()

    while remaining and len(selected) < k:
        best_idx = 0
        best_value = float("-inf")
        for i, cand in enumerate(remaining):
            relevance = float(cand.score) if cand.score is not None else 0.0
            penalty = doc_id_penalty if cand.page_id in selected_doc_ids else 0.0
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * penalty
            if value > best_value:
                best_value = value
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        if chosen.page_id:
            selected_doc_ids.add(chosen.page_id)

    # Re-rank ``rank`` to be 1..len(selected) so downstream metric code
    # (which uses the rank for nDCG / first_hit_rank) doesn't think the
    # MMR-picked rank-7 is at position 7 of the original pool.
    out: List[RetrievedDoc] = []
    for new_rank, d in enumerate(selected, start=1):
        out.append(RetrievedDoc(
            rank=new_rank,
            chunk_id=d.chunk_id,
            page_id=d.page_id,
            title=d.title,
            section_path=d.section_path,
            score=d.score,
        ))
    return out


def apply_variant_to_candidates(
    pool: Sequence[RetrievedDoc],
    *,
    candidate_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    top_k: int,
) -> List[RetrievedDoc]:
    """Slice + rerank a pre-computed candidate pool for one variant config.

    The pool is assumed to be ordered by relevance score (descending),
    same shape as the live retriever returns from FAISS. We:

      1. Take the top ``candidate_k`` from the pool.
      2. If ``use_mmr`` is True, run MMR selection on those.
      3. Otherwise just take the top ``top_k`` directly.

    Returns at most ``top_k`` docs, with ranks renumbered.
    """
    if not pool:
        return []
    eff_candidate_k = max(int(top_k), int(candidate_k))
    sliced = list(pool)[:eff_candidate_k]
    if use_mmr:
        return mmr_select_post_hoc(
            sliced, top_k=int(top_k), mmr_lambda=float(mmr_lambda),
            doc_id_penalty=MMR_DOC_ID_PENALTY,
        )
    out: List[RetrievedDoc] = []
    for new_rank, d in enumerate(sliced[: int(top_k)], start=1):
        out.append(RetrievedDoc(
            rank=new_rank,
            chunk_id=d.chunk_id,
            page_id=d.page_id,
            title=d.title,
            section_path=d.section_path,
            score=d.score,
        ))
    return out


# ---------------------------------------------------------------------------
# Confirm-sweep guardrails
# ---------------------------------------------------------------------------


def _bucket_weighted_hit_at_5(
    summary: GoldSummary, bucket: str,
) -> Optional[float]:
    """Pick out a single bucket's weighted_hit_at_5 from a GoldSummary.

    Returns None when the bucket isn't in the by_bucket map at all so
    the caller can render "—" instead of a misleading 0.
    """
    cell = summary.by_bucket.get(bucket)
    if not cell:
        return None
    return float(cell.get("weighted_hit_at_5", 0.0))


def evaluate_main_work_guardrail(
    *,
    baseline: GoldSummary,
    candidate: GoldSummary,
    threshold: float = MAIN_WORK_REGRESSION_THRESHOLD,
) -> Optional[GuardrailWarning]:
    """Fire ``MAIN_WORK_REGRESSION_WARNING`` when the bucket drops too far.

    The main_work bucket is the broadest "is this even on the right
    work?" slice. A candidate that gains on subpage_named while
    collapsing main_work is solving the wrong problem.
    """
    base_v = _bucket_weighted_hit_at_5(baseline, "main_work")
    cand_v = _bucket_weighted_hit_at_5(candidate, "main_work")
    if base_v is None or cand_v is None:
        return None
    delta = cand_v - base_v
    if delta <= -float(threshold):
        return GuardrailWarning(
            code=MAIN_WORK_REGRESSION_WARNING,
            metric="weighted_hit_at_5",
            bucket="main_work",
            baseline=base_v,
            candidate=cand_v,
            delta=delta,
            threshold=float(threshold),
            message=(
                f"gold main_work weighted_hit@5 dropped "
                f"{abs(delta)*100:.2f}pp (>= {threshold*100:.1f}pp "
                f"threshold) — candidate is trading the broadest bucket "
                f"for subpage gains; do NOT promote without inspecting "
                f"main_work failures first."
            ),
        )
    return None


def evaluate_subpage_named_hold(
    *,
    baseline: GoldSummary,
    candidate: GoldSummary,
    hold_threshold: float = SUBPAGE_NAMED_HOLD_THRESHOLD,
) -> Optional[GuardrailWarning]:
    """Fire ``SUBPAGE_NAMED_NOT_FIXED_WARNING`` when the bucket regresses.

    The whole point of the gold-50 set is to surface subpage_named
    failures — a candidate that doesn't HOLD subpage_named, let alone
    improve it, isn't fixing the problem the set was curated for.
    """
    base_v = _bucket_weighted_hit_at_5(baseline, "subpage_named")
    cand_v = _bucket_weighted_hit_at_5(candidate, "subpage_named")
    if base_v is None or cand_v is None:
        return None
    delta = cand_v - base_v
    if delta < -float(hold_threshold):
        return GuardrailWarning(
            code=SUBPAGE_NAMED_NOT_FIXED_WARNING,
            metric="weighted_hit_at_5",
            bucket="subpage_named",
            baseline=base_v,
            candidate=cand_v,
            delta=delta,
            threshold=float(hold_threshold),
            message=(
                f"gold subpage_named weighted_hit@5 regressed "
                f"({delta:+.4f} vs baseline; hold threshold "
                f"{hold_threshold:+.4f}). The gold-50 set was curated to "
                f"fix this bucket — a candidate that doesn't hold it is "
                f"not solving the right problem."
            ),
        )
    return None


def evaluate_section_hit_caveat(
    *,
    baseline: GoldSummary,
    candidate: GoldSummary,
    halving_factor: float = SECTION_HIT_HALVING_FACTOR,
) -> Optional[GuardrailWarning]:
    """Note section_hit@5 drops >= half of baseline as a caveat.

    NOT a hard guardrail — section_hit@5 is auxiliary, the metric base
    is tiny, and Phase 7.x acknowledged the metric's brittleness. We
    only emit a warning so the report can carry the caveat text.
    """
    base_v = baseline.section_hit_at_5_when_defined
    cand_v = candidate.section_hit_at_5_when_defined
    if base_v is None or cand_v is None:
        return None
    if base_v <= 0.0:
        return None
    if cand_v > base_v * float(halving_factor):
        # Within factor (e.g. >= half of baseline) → no warning.
        return None
    delta = cand_v - base_v
    return GuardrailWarning(
        code=SECTION_RETRIEVAL_WARNING,
        metric="section_hit_at_5",
        bucket=None,
        baseline=base_v,
        candidate=cand_v,
        delta=delta,
        threshold=float(halving_factor),
        message=(
            f"gold section_hit@5 fell from {base_v:.4f} to {cand_v:.4f} "
            f"(<= {halving_factor*100:.0f}% of baseline). Documented as "
            f"a caveat — page_hit improved, section-level exact match "
            f"regressed on a tiny base. Not a promotion blocker; needs "
            f"section-aware reranking or chunk-level audit to validate."
        ),
    )


# ---------------------------------------------------------------------------
# Plateau analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateauAnalysis:
    """Verdict on whether the chosen winner sits on a stable plateau.

    ``status`` is one of :data:`PLATEAU_OK` or :data:`PLATEAU_OVERFIT_WARNING`.
    ``neighbours`` lists the (lambda, primary_score) of the points
    inspected — usually the immediate λ-neighbours within the same
    candidate_k row of the grid.
    """

    status: str
    best_variant: str
    best_primary_score: float
    candidate_k: int
    mmr_lambda: float
    neighbours: Tuple[Tuple[float, float], ...]
    epsilon: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "best_variant": self.best_variant,
            "best_primary_score": self.best_primary_score,
            "candidate_k": self.candidate_k,
            "mmr_lambda": self.mmr_lambda,
            "neighbours": [list(n) for n in self.neighbours],
            "epsilon": self.epsilon,
            "message": self.message,
        }


def analyze_plateau(
    *,
    best_candidate: SweepCandidate,
    best_primary_score: float,
    sweep_results_by_name: Mapping[str, GoldSummary],
    grid: Sequence[SweepCandidate],
    epsilon: float = PLATEAU_PRIMARY_DELTA,
) -> PlateauAnalysis:
    """Inspect the λ-neighbours of the chosen winner.

    For every candidate in ``grid`` that shares ``best_candidate``'s
    candidate_k AND sits one λ-step away (i.e. immediate neighbour on
    the grid), check that its primary_score is within ``epsilon`` of
    the winner's. If both neighbours satisfy this, the winner is on a
    plateau (:data:`PLATEAU_OK`); otherwise we flag potential overfit
    (:data:`PLATEAU_OVERFIT_WARNING`).
    """
    same_row = [
        c for c in grid
        if c.candidate_k == best_candidate.candidate_k
    ]
    same_row_sorted = sorted(same_row, key=lambda c: c.mmr_lambda)
    lambdas = [c.mmr_lambda for c in same_row_sorted]
    try:
        idx = lambdas.index(best_candidate.mmr_lambda)
    except ValueError:
        return PlateauAnalysis(
            status=PLATEAU_OVERFIT_WARNING,
            best_variant=best_candidate.name,
            best_primary_score=float(best_primary_score),
            candidate_k=best_candidate.candidate_k,
            mmr_lambda=best_candidate.mmr_lambda,
            neighbours=(),
            epsilon=float(epsilon),
            message=(
                f"best λ={best_candidate.mmr_lambda:.2f} is not a member "
                f"of the candidate_k={best_candidate.candidate_k} grid "
                f"row — cannot evaluate plateau."
            ),
        )

    neighbour_idxs: List[int] = []
    if idx > 0:
        neighbour_idxs.append(idx - 1)
    if idx < len(same_row_sorted) - 1:
        neighbour_idxs.append(idx + 1)

    neighbours: List[Tuple[float, float]] = []
    drops: List[Tuple[float, float]] = []
    for ni in neighbour_idxs:
        nb = same_row_sorted[ni]
        nb_summary = sweep_results_by_name.get(nb.name)
        if nb_summary is None:
            neighbours.append((nb.mmr_lambda, float("nan")))
            continue
        nb_score = float(nb_summary.primary_score)
        neighbours.append((nb.mmr_lambda, nb_score))
        if nb_score < float(best_primary_score) - float(epsilon):
            drops.append((nb.mmr_lambda, nb_score))

    if drops:
        msg = (
            f"λ={best_candidate.mmr_lambda:.2f} is the winner but its "
            f"neighbour(s) "
            + ", ".join(f"λ={lam:.2f} ({s:.4f})" for lam, s in drops)
            + f" sit > {epsilon:.4f} primary_score below the winner — "
            f"single-point peak risk; a small λ shift may erase the gain."
        )
        return PlateauAnalysis(
            status=PLATEAU_OVERFIT_WARNING,
            best_variant=best_candidate.name,
            best_primary_score=float(best_primary_score),
            candidate_k=best_candidate.candidate_k,
            mmr_lambda=best_candidate.mmr_lambda,
            neighbours=tuple(neighbours),
            epsilon=float(epsilon),
            message=msg,
        )
    if not neighbours:
        return PlateauAnalysis(
            status=PLATEAU_OVERFIT_WARNING,
            best_variant=best_candidate.name,
            best_primary_score=float(best_primary_score),
            candidate_k=best_candidate.candidate_k,
            mmr_lambda=best_candidate.mmr_lambda,
            neighbours=(),
            epsilon=float(epsilon),
            message=(
                "best λ has no immediate λ-grid neighbours (sole row "
                "member) — cannot validate as plateau."
            ),
        )
    return PlateauAnalysis(
        status=PLATEAU_OK,
        best_variant=best_candidate.name,
        best_primary_score=float(best_primary_score),
        candidate_k=best_candidate.candidate_k,
        mmr_lambda=best_candidate.mmr_lambda,
        neighbours=tuple(neighbours),
        epsilon=float(epsilon),
        message=(
            f"λ={best_candidate.mmr_lambda:.2f} winner is within "
            f"{epsilon:.4f} primary_score of all immediate neighbours "
            f"({len(neighbours)} compared); treated as a stable plateau."
        ),
    )


# ---------------------------------------------------------------------------
# Confirmed-best selection
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    """Per-candidate computed deltas + accumulated guardrail warnings.

    Used as the row record for the confirm_sweep_summary.json + the
    headline table in confirm_sweep_report.md.
    """

    name: str
    candidate_k: int
    mmr_lambda: float
    primary_score: float
    weighted_hit_at_5: float
    weighted_mrr_at_10: float
    weighted_ndcg_at_10: float
    silver_hit_at_5: float
    main_work_weighted_hit_at_5: Optional[float]
    subpage_named_weighted_hit_at_5: Optional[float]
    subpage_generic_weighted_hit_at_5: Optional[float]
    section_hit_at_5: Optional[float]
    deltas: Dict[str, float]
    warnings: List[GuardrailWarning] = field(default_factory=list)
    accepted: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "candidate_k": self.candidate_k,
            "mmr_lambda": self.mmr_lambda,
            "primary_score": self.primary_score,
            "weighted_hit_at_5": self.weighted_hit_at_5,
            "weighted_mrr_at_10": self.weighted_mrr_at_10,
            "weighted_ndcg_at_10": self.weighted_ndcg_at_10,
            "silver_hit_at_5": self.silver_hit_at_5,
            "main_work_weighted_hit_at_5": self.main_work_weighted_hit_at_5,
            "subpage_named_weighted_hit_at_5": (
                self.subpage_named_weighted_hit_at_5
            ),
            "subpage_generic_weighted_hit_at_5": (
                self.subpage_generic_weighted_hit_at_5
            ),
            "section_hit_at_5": self.section_hit_at_5,
            "deltas": self.deltas,
            "warnings": [w.to_dict() for w in self.warnings],
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ConfirmSweepResult:
    """The verdict + per-candidate detail of one confirm sweep.

    ``confirmed_best`` is None when no candidate cleared the selection
    rule; in that case ``promotion_recommended`` is False and the
    report tells the reviewer to keep the baseline.
    """

    baseline_name: str
    baseline_primary_score: float
    candidates: List[CandidateScore]
    confirmed_best: Optional[CandidateScore]
    plateau: Optional[PlateauAnalysis]
    promotion_recommended: bool
    promotion_reason: str
    grid: List[SweepCandidate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_primary_score": self.baseline_primary_score,
            "candidates": [c.to_dict() for c in self.candidates],
            "confirmed_best": (
                self.confirmed_best.to_dict()
                if self.confirmed_best is not None else None
            ),
            "plateau": self.plateau.to_dict() if self.plateau else None,
            "promotion_recommended": self.promotion_recommended,
            "promotion_reason": self.promotion_reason,
            "grid": [g.to_dict() for g in self.grid],
            "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
            "promotion_target_clarification": PROMOTION_TARGET_CLARIFICATION,
            "section_hit_caveat": SECTION_HIT_CAVEAT,
        }


def _candidate_score_from(
    *,
    spec: SweepCandidate,
    cand_summary_gold: GoldSummary,
    cand_summary_silver: SilverSummary,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
) -> CandidateScore:
    """Build the row record for one candidate including warnings + deltas."""
    primary = float(cand_summary_gold.primary_score)
    base_primary = float(baseline_summary_gold.primary_score)
    main_v = _bucket_weighted_hit_at_5(cand_summary_gold, "main_work")
    sub_named_v = _bucket_weighted_hit_at_5(cand_summary_gold, "subpage_named")
    sub_generic_v = _bucket_weighted_hit_at_5(
        cand_summary_gold, "subpage_generic",
    )
    base_main = _bucket_weighted_hit_at_5(baseline_summary_gold, "main_work")
    base_sub_named = _bucket_weighted_hit_at_5(
        baseline_summary_gold, "subpage_named",
    )
    base_sub_generic = _bucket_weighted_hit_at_5(
        baseline_summary_gold, "subpage_generic",
    )
    deltas = {
        "primary_score": primary - base_primary,
        "weighted_hit_at_5": (
            cand_summary_gold.weighted_hit_at_5
            - baseline_summary_gold.weighted_hit_at_5
        ),
        "weighted_mrr_at_10": (
            cand_summary_gold.weighted_mrr_at_10
            - baseline_summary_gold.weighted_mrr_at_10
        ),
        "weighted_ndcg_at_10": (
            cand_summary_gold.weighted_ndcg_at_10
            - baseline_summary_gold.weighted_ndcg_at_10
        ),
        "strict_hit_at_5": (
            cand_summary_gold.strict_hit_at_5
            - baseline_summary_gold.strict_hit_at_5
        ),
        "silver_hit_at_5": (
            cand_summary_silver.hit_at_5
            - baseline_summary_silver.hit_at_5
        ),
        "main_work_weighted_hit_at_5": (
            (main_v - base_main)
            if (main_v is not None and base_main is not None)
            else 0.0
        ),
        "subpage_named_weighted_hit_at_5": (
            (sub_named_v - base_sub_named)
            if (sub_named_v is not None and base_sub_named is not None)
            else 0.0
        ),
        "subpage_generic_weighted_hit_at_5": (
            (sub_generic_v - base_sub_generic)
            if (sub_generic_v is not None and base_sub_generic is not None)
            else 0.0
        ),
        "section_hit_at_5": (
            (cand_summary_gold.section_hit_at_5_when_defined or 0.0)
            - (baseline_summary_gold.section_hit_at_5_when_defined or 0.0)
        ),
    }

    warnings: List[GuardrailWarning] = list(
        evaluate_silver_guardrail(
            baseline=baseline_summary_silver,
            candidate=cand_summary_silver,
        )
    )
    sub_named = evaluate_subpage_named_hold(
        baseline=baseline_summary_gold, candidate=cand_summary_gold,
    )
    if sub_named is not None:
        warnings.append(sub_named)
    main_work = evaluate_main_work_guardrail(
        baseline=baseline_summary_gold, candidate=cand_summary_gold,
    )
    if main_work is not None:
        warnings.append(main_work)
    section = evaluate_section_hit_caveat(
        baseline=baseline_summary_gold, candidate=cand_summary_gold,
    )
    if section is not None:
        warnings.append(section)

    return CandidateScore(
        name=spec.name,
        candidate_k=spec.candidate_k,
        mmr_lambda=spec.mmr_lambda,
        primary_score=primary,
        weighted_hit_at_5=cand_summary_gold.weighted_hit_at_5,
        weighted_mrr_at_10=cand_summary_gold.weighted_mrr_at_10,
        weighted_ndcg_at_10=cand_summary_gold.weighted_ndcg_at_10,
        silver_hit_at_5=cand_summary_silver.hit_at_5,
        main_work_weighted_hit_at_5=main_v,
        subpage_named_weighted_hit_at_5=sub_named_v,
        subpage_generic_weighted_hit_at_5=sub_generic_v,
        section_hit_at_5=cand_summary_gold.section_hit_at_5_when_defined,
        deltas=deltas,
        warnings=warnings,
    )


def select_confirmed_best(
    *,
    grid: Sequence[SweepCandidate],
    baseline_name: str,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
    candidate_results: Mapping[str, Tuple[GoldSummary, SilverSummary]],
    primary_min_delta: float = 0.0005,
) -> ConfirmSweepResult:
    """Apply the confirm-sweep selection rule across the grid.

    Selection (matches the spec):

      1. Build CandidateScore per grid entry; collect warnings.
      2. Disqualify candidates with:
         - primary_score delta < ``primary_min_delta`` (no improvement)
         - SILVER_REGRESSION_WARNING (silver hit@5 regressed)
         - BUCKET_REGRESSION_WARNING (silver subpage_named regressed)
         - SUBPAGE_NAMED_NOT_FIXED_WARNING (gold subpage_named regressed)
         - MAIN_WORK_REGRESSION_WARNING (gold main_work collapsed)
      3. Of the survivors, rank by (primary_score desc, weighted_mrr
         desc, name asc). The top is the confirmed best.
      4. If no survivor: keep baseline.
      5. Run plateau analysis on the winner's λ-grid neighbours; the
         result is attached but does NOT veto the win — the report
         surfaces the warning and the reviewer decides.

    Section-hit caveat is informational only and is captured as a
    warning on the candidate row, not a disqualifier.
    """
    rows: List[CandidateScore] = []
    for spec in grid:
        cand = candidate_results.get(spec.name)
        if cand is None:
            log.warning(
                "confirm sweep: missing candidate result for %s — skipping.",
                spec.name,
            )
            continue
        cand_gold, cand_silver = cand
        rows.append(_candidate_score_from(
            spec=spec,
            cand_summary_gold=cand_gold,
            cand_summary_silver=cand_silver,
            baseline_summary_gold=baseline_summary_gold,
            baseline_summary_silver=baseline_summary_silver,
        ))

    blocking_codes = {
        "SILVER_REGRESSION_WARNING",
        "BUCKET_REGRESSION_WARNING",
        SUBPAGE_NAMED_NOT_FIXED_WARNING,
        MAIN_WORK_REGRESSION_WARNING,
    }
    survivors: List[CandidateScore] = []
    for r in rows:
        primary_delta = r.deltas.get("primary_score", 0.0)
        if primary_delta < primary_min_delta:
            r.rejection_reason = (
                f"primary_score delta {primary_delta:+.4f} below "
                f"epsilon {primary_min_delta:+.4f}."
            )
            continue
        blocking = [w for w in r.warnings if w.code in blocking_codes]
        if blocking:
            r.rejection_reason = "; ".join(
                f"{w.code}({w.metric})" for w in blocking
            )
            continue
        r.accepted = True
        survivors.append(r)

    confirmed_best: Optional[CandidateScore] = None
    plateau: Optional[PlateauAnalysis] = None
    promotion_recommended = False
    promotion_reason = ""

    if survivors:
        survivors.sort(
            key=lambda x: (-x.primary_score, -x.weighted_mrr_at_10, x.name)
        )
        confirmed_best = survivors[0]
        # Run plateau analysis using the gold summaries we already have.
        sweep_summaries: Dict[str, GoldSummary] = {}
        for spec in grid:
            cand = candidate_results.get(spec.name)
            if cand is None:
                continue
            sweep_summaries[spec.name] = cand[0]
        best_spec = next(g for g in grid if g.name == confirmed_best.name)
        plateau = analyze_plateau(
            best_candidate=best_spec,
            best_primary_score=confirmed_best.primary_score,
            sweep_results_by_name=sweep_summaries,
            grid=grid,
        )
        promotion_recommended = True
        promotion_reason = (
            f"confirmed_best={confirmed_best.name}, primary_score="
            f"{confirmed_best.primary_score:.6f} "
            f"({confirmed_best.deltas['primary_score']:+.6f} vs baseline). "
            f"Plateau: {plateau.status if plateau else 'NOT_EVALUATED'}. "
            f"All hard guardrails (silver hit@5, silver subpage_named, "
            f"gold subpage_named hold, gold main_work) passed."
        )
    else:
        promotion_reason = (
            "no candidate cleared the primary_score epsilon AND survived "
            "the silver + main_work + subpage_named guardrails — keep "
            "baseline."
        )

    return ConfirmSweepResult(
        baseline_name=baseline_name,
        baseline_primary_score=float(baseline_summary_gold.primary_score),
        candidates=rows,
        confirmed_best=confirmed_best,
        plateau=plateau,
        promotion_recommended=promotion_recommended,
        promotion_reason=promotion_reason,
        grid=list(grid),
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _row_for_grid_table(
    score: CandidateScore, baseline_primary: float,
) -> List[str]:
    """Helper: format a CandidateScore as one row of the grid table."""
    delta = score.primary_score - baseline_primary
    accepted_marker = "✓" if score.accepted else "✗"
    return [
        score.name,
        f"{score.candidate_k}",
        f"{score.mmr_lambda:.2f}",
        f"{score.primary_score:.4f} ({delta:+.4f})",
        f"{score.weighted_hit_at_5:.4f}",
        f"{score.weighted_mrr_at_10:.4f}",
        f"{score.silver_hit_at_5:.4f}",
        f"{(score.subpage_named_weighted_hit_at_5 or 0.0):.4f}",
        f"{(score.main_work_weighted_hit_at_5 or 0.0):.4f}",
        f"{(score.section_hit_at_5 or 0.0):.4f}",
        accepted_marker,
        score.rejection_reason or "—",
    ]


def render_confirm_sweep_report(
    *,
    result: ConfirmSweepResult,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
    previous_best_name: Optional[str] = None,
    previous_best_score: Optional[CandidateScore] = None,
) -> str:
    """Render the human-facing confirm_sweep_report.md."""
    lines: List[str] = []
    lines.append("# Phase 7.5 — MMR confirm sweep")
    lines.append("")
    lines.append(f"> {HUMAN_FOCUS_DISCLAIMER}")
    lines.append("")
    lines.append("## Promotion target clarification")
    lines.append("")
    lines.append(PROMOTION_TARGET_CLARIFICATION)
    lines.append("")

    base_primary = result.baseline_primary_score
    lines.append(
        f"- baseline: **{result.baseline_name}** — primary_score="
        f"{base_primary:.4f}"
    )
    if previous_best_name is not None:
        lines.append(
            f"- previous best (Phase 7.x first pass): **{previous_best_name}**"
        )
    if previous_best_score is not None:
        lines.append(
            f"  - primary_score={previous_best_score.primary_score:.4f} "
            f"({previous_best_score.deltas['primary_score']:+.4f})"
        )
    if result.confirmed_best is not None:
        cb = result.confirmed_best
        lines.append(
            f"- confirmed best: **{cb.name}** — primary_score="
            f"{cb.primary_score:.4f} "
            f"({cb.deltas['primary_score']:+.4f} vs baseline)"
        )
    else:
        lines.append("- confirmed best: **(none — keep baseline)**")
    lines.append(
        f"- promotion recommended: "
        f"**{'YES' if result.promotion_recommended else 'NO'}** — "
        f"{result.promotion_reason}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Headline comparison
    # ------------------------------------------------------------------
    lines.append("## Headline comparison")
    lines.append("")
    lines.append(
        "| metric | baseline | previous best | confirmed best |"
    )
    lines.append("|---|---:|---:|---:|")
    cb = result.confirmed_best
    pb = previous_best_score
    metric_rows = (
        ("gold primary_score", base_primary,
         pb.primary_score if pb else None,
         cb.primary_score if cb else None),
        ("gold weighted_hit@5", baseline_summary_gold.weighted_hit_at_5,
         pb.weighted_hit_at_5 if pb else None,
         cb.weighted_hit_at_5 if cb else None),
        ("gold weighted_MRR@10", baseline_summary_gold.weighted_mrr_at_10,
         pb.weighted_mrr_at_10 if pb else None,
         cb.weighted_mrr_at_10 if cb else None),
        ("gold weighted_nDCG@10", baseline_summary_gold.weighted_ndcg_at_10,
         pb.weighted_ndcg_at_10 if pb else None,
         cb.weighted_ndcg_at_10 if cb else None),
        ("silver hit@5", baseline_summary_silver.hit_at_5,
         pb.silver_hit_at_5 if pb else None,
         cb.silver_hit_at_5 if cb else None),
        ("silver MRR@10", baseline_summary_silver.mrr_at_10,
         None, None),
        ("subpage_named weighted_hit@5",
         _bucket_weighted_hit_at_5(baseline_summary_gold, "subpage_named") or 0.0,
         pb.subpage_named_weighted_hit_at_5 if pb else None,
         cb.subpage_named_weighted_hit_at_5 if cb else None),
        ("subpage_generic weighted_hit@5",
         _bucket_weighted_hit_at_5(baseline_summary_gold, "subpage_generic") or 0.0,
         pb.subpage_generic_weighted_hit_at_5 if pb else None,
         cb.subpage_generic_weighted_hit_at_5 if cb else None),
        ("main_work weighted_hit@5",
         _bucket_weighted_hit_at_5(baseline_summary_gold, "main_work") or 0.0,
         pb.main_work_weighted_hit_at_5 if pb else None,
         cb.main_work_weighted_hit_at_5 if cb else None),
        ("section_hit@5",
         baseline_summary_gold.section_hit_at_5_when_defined or 0.0,
         pb.section_hit_at_5 if pb else None,
         cb.section_hit_at_5 if cb else None),
    )

    def _fmt(v: Optional[float]) -> str:
        return "—" if v is None else f"{float(v):.4f}"

    for label, b, pbv, cbv in metric_rows:
        lines.append(
            f"| {label} | {_fmt(b)} | {_fmt(pbv)} | {_fmt(cbv)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Full sweep grid
    # ------------------------------------------------------------------
    lines.append("## Sweep grid (15 candidates)")
    lines.append("")
    lines.append(
        "| variant | candidate_k | λ | primary | wh@5 | wMRR@10 | "
        "silver_h@5 | subpage_named_wh@5 | main_work_wh@5 | section_h@5 | "
        "accepted | rejection |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|"
    )
    sorted_rows = sorted(
        result.candidates,
        key=lambda r: (r.candidate_k, r.mmr_lambda),
    )
    for r in sorted_rows:
        cells = _row_for_grid_table(r, base_primary)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------
    lines.append("## Guardrail warnings")
    lines.append("")
    any_warn = False
    for r in sorted_rows:
        if not r.warnings:
            continue
        any_warn = True
        lines.append(f"- **{r.name}**:")
        for w in r.warnings:
            lines.append(
                f"  - `{w.code}` ({w.metric}, bucket={w.bucket!s}): "
                f"baseline={w.baseline:.4f} → candidate={w.candidate:.4f} "
                f"(Δ={w.delta:+.4f}; threshold {w.threshold}). {w.message}"
            )
    if not any_warn:
        lines.append("- (none)")
    lines.append("")

    # ------------------------------------------------------------------
    # Plateau analysis
    # ------------------------------------------------------------------
    lines.append("## Plateau analysis")
    lines.append("")
    if result.plateau is None:
        lines.append("- no winner → plateau analysis skipped.")
    else:
        p = result.plateau
        lines.append(f"- status: **{p.status}**")
        lines.append(
            f"- best variant: `{p.best_variant}` (candidate_k="
            f"{p.candidate_k}, λ={p.mmr_lambda:.2f})"
        )
        if p.neighbours:
            lines.append("- λ-neighbours (same candidate_k row):")
            for lam, score in p.neighbours:
                lines.append(f"  - λ={lam:.2f}: primary_score={score:.4f}")
        lines.append(f"- {p.message}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section-hit caveat
    # ------------------------------------------------------------------
    lines.append("## Section_hit@5 caveat")
    lines.append("")
    lines.append(SECTION_HIT_CAVEAT)
    lines.append("")

    # ------------------------------------------------------------------
    # Recommendation + rollback
    # ------------------------------------------------------------------
    lines.append("## Recommendation")
    lines.append("")
    if result.confirmed_best is not None and result.promotion_recommended:
        cb = result.confirmed_best
        lines.append(
            f"- **Promote** retrieval config "
            f"`top_k={DEFAULT_TOP_K}, candidate_k={cb.candidate_k}, "
            f"use_mmr=true, mmr_lambda={cb.mmr_lambda:.2f}` on the "
            f"production-default `{PRODUCTION_INDEX_CACHE_DIR}` index."
        )
    else:
        lines.append(
            "- **Keep baseline.** No candidate cleared the selection rule."
        )
    lines.append("")
    lines.append("### Rollback")
    lines.append("")
    lines.append(
        "- Set `AIPIPELINE_WORKER_RAG_USE_MMR=false` and remove the "
        "`AIPIPELINE_WORKER_RAG_CANDIDATE_K` override to restore the "
        "exact pre-promotion retrieval behaviour. The index variant "
        "itself was promoted in Phase 7.2 and is NOT touched by this "
        "change."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------
    lines.append("## Reminders")
    lines.append("")
    lines.append(
        "- gold-50 is a *human-weighted focus set* drawn from "
        "queries_v4_llm_silver_500. Improvements only mean we got "
        "better at the subpage / named-subpage failures the gold-50 "
        "set was curated to expose. NOT a generic retrieval benchmark."
    )
    lines.append(
        "- silver-500 is LLM-generated and serves as the "
        "**overfitting guardrail / sanity check**, NOT the primary "
        "tuning objective."
    )
    lines.append(
        "- production retrieval config MUST NOT be changed off this "
        "report alone — promote via the standard config-change PR."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_confirm_sweep_results_jsonl(
    path: Path,
    *,
    result: ConfirmSweepResult,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
    candidate_results: Mapping[str, Tuple[GoldSummary, SilverSummary]],
) -> Path:
    """Persist one row per candidate (config + gold + silver summary).

    Replay-friendly: the row shape is the same as
    ``candidate_results.jsonl`` from the base harness, so a downstream
    script can mash the two together without parsing the report MD.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        # Write baseline first so the file is self-contained.
        fp.write(json.dumps({
            "variant": result.baseline_name,
            "role": "baseline",
            "config": {"name": result.baseline_name},
            "gold_summary": gold_summary_to_dict(baseline_summary_gold),
            "silver_summary": silver_summary_to_dict(baseline_summary_silver),
        }, ensure_ascii=False) + "\n")
        for spec in result.grid:
            cand = candidate_results.get(spec.name)
            if cand is None:
                continue
            cand_gold, cand_silver = cand
            score_row = next(
                (c for c in result.candidates if c.name == spec.name), None
            )
            fp.write(json.dumps({
                "variant": spec.name,
                "role": "candidate",
                "config": spec.to_dict(),
                "gold_summary": gold_summary_to_dict(cand_gold),
                "silver_summary": silver_summary_to_dict(cand_silver),
                "score_row": score_row.to_dict() if score_row else None,
            }, ensure_ascii=False) + "\n")
    return path


def write_confirm_sweep_summary_json(
    path: Path, *, result: ConfirmSweepResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_confirm_sweep_report_md(
    path: Path,
    *,
    result: ConfirmSweepResult,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
    previous_best_name: Optional[str] = None,
    previous_best_score: Optional[CandidateScore] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_confirm_sweep_report(
        result=result,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
        previous_best_name=previous_best_name,
        previous_best_score=previous_best_score,
    )
    # Regression check inside the writer too: never emit any of the
    # forbidden phrases. Catches a reviewer accidentally passing the
    # rejected variant as the previous_best.
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        if phrase in md:
            raise ValueError(
                f"render_confirm_sweep_report produced a forbidden phrase: "
                f"{phrase!r}. Check the input arguments."
            )
    path.write_text(md, encoding="utf-8")
    return path


def write_confirmed_best_config_json(
    path: Path, *, result: ConfirmSweepResult,
) -> Path:
    """Emit the promote-or-keep config JSON.

    Always written, even when no candidate cleared the rule (the
    reviewer can grep the file to know an empty pass happened).
    """
    if result.confirmed_best is None:
        payload: Dict[str, Any] = {
            "promotion_recommended": False,
            "reason": result.promotion_reason,
            "baseline_name": result.baseline_name,
            "baseline_primary_score": result.baseline_primary_score,
            "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
            "promotion_target_clarification": PROMOTION_TARGET_CLARIFICATION,
        }
    else:
        cb = result.confirmed_best
        spec = next(g for g in result.grid if g.name == cb.name)
        payload = {
            "promotion_recommended": True,
            "reason": result.promotion_reason,
            "baseline_name": result.baseline_name,
            "baseline_primary_score": result.baseline_primary_score,
            "confirmed_best": cb.to_dict(),
            "config": spec.to_dict(),
            "plateau": (
                result.plateau.to_dict() if result.plateau else None
            ),
            "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
            "promotion_target_clarification": PROMOTION_TARGET_CLARIFICATION,
            "section_hit_caveat": SECTION_HIT_CAVEAT,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_confirmed_best_config_env(
    path: Path, *, result: ConfirmSweepResult,
) -> Path:
    """Emit the env-snippet form of the promote-or-keep config.

    When promotion is NOT recommended the file lists the rollback /
    no-op env values so a reviewer can paste it without thinking.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Phase 7.5 confirm-sweep — recommended retrieval config",
        "# DO NOT auto-apply. Promote via the standard config-change PR.",
        f"# {HUMAN_FOCUS_DISCLAIMER}",
        f"# Promotion target: {PROMOTION_TARGET_CLARIFICATION}",
    ]
    if result.confirmed_best is None:
        lines.append("# promotion_recommended=false")
        lines.append(f"# reason={result.promotion_reason}")
        lines.append("# Keep baseline — env values unchanged. Rollback is a no-op.")
        lines.append("AIPIPELINE_WORKER_RAG_USE_MMR=false")
        lines.append(f"AIPIPELINE_WORKER_RAG_TOP_K={DEFAULT_TOP_K}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    cb = result.confirmed_best
    spec = next(g for g in result.grid if g.name == cb.name)
    lines.append(f"# promotion_recommended=true")
    lines.append(f"# confirmed_best={cb.name}")
    lines.append(f"# reason={result.promotion_reason}")
    if result.plateau is not None:
        lines.append(f"# plateau_status={result.plateau.status}")
    lines.append(f"AIPIPELINE_WORKER_RAG_TOP_K={spec.top_k}")
    lines.append(f"AIPIPELINE_WORKER_RAG_CANDIDATE_K={spec.candidate_k}")
    lines.append(
        f"AIPIPELINE_WORKER_RAG_USE_MMR="
        f"{'true' if spec.use_mmr else 'false'}"
    )
    lines.append(f"AIPIPELINE_WORKER_RAG_MMR_LAMBDA={spec.mmr_lambda:.4f}")
    lines.append(f"# index variant cache: {spec.cache_dir_relative}")
    lines.append(
        "# Rollback: drop AIPIPELINE_WORKER_RAG_CANDIDATE_K and set "
        "AIPIPELINE_WORKER_RAG_USE_MMR=false to restore baseline."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Production recommendation — promotion artefact + lambda policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionRecommendation:
    """The plateau-aware production-promotion record.

    Captures the metric-best winner (what the selection rule picked),
    the production-recommended config (what the PR should ship), and
    the policy that mapped one to the other. Persisted as the JSON
    body of ``best_config.production_recommended.json`` and rendered
    into the report's ``## Production recommendation`` section.

    ``rollback_env`` is the env-snippet a reviewer should paste to
    undo the change — restoring whatever was shipping before the PR
    landed (typically ``USE_MMR=false`` with no candidate_k override).

    ``baseline_metrics`` and ``confirmed_metrics`` are populated from
    the same gold/silver summaries the confirm sweep already
    aggregated; the PR description renderer reuses them so a reviewer
    sees the same numbers the metric-best record carried.
    """

    confirmed_best_name: str
    confirmed_best_lambda: float
    confirmed_best_primary_score: float
    recommended_lambda: float
    recommended_variant_name: Optional[str]
    candidate_k: int
    top_k: int
    use_mmr: bool
    cache_dir_relative: str
    plateau_status: str
    plateau_lambdas: Tuple[Tuple[float, float], ...]
    selected_lambda_policy: str
    selected_lambda_reason: str
    rollback_env: Tuple[Tuple[str, str], ...]
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    confirmed_metrics: Dict[str, float] = field(default_factory=dict)
    section_hit_caveat: str = SECTION_HIT_CAVEAT
    promotion_target_clarification: str = PROMOTION_TARGET_CLARIFICATION
    human_focus_disclaimer: str = HUMAN_FOCUS_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirmed_best_name": self.confirmed_best_name,
            "confirmed_best_lambda": self.confirmed_best_lambda,
            "confirmed_best_primary_score": (
                self.confirmed_best_primary_score
            ),
            "recommended_lambda": self.recommended_lambda,
            "recommended_variant_name": self.recommended_variant_name,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "use_mmr": self.use_mmr,
            "mmr_lambda": self.recommended_lambda,
            "cache_dir_relative": self.cache_dir_relative,
            "plateau_status": self.plateau_status,
            "plateau_lambdas": [list(p) for p in self.plateau_lambdas],
            "selected_lambda_policy": self.selected_lambda_policy,
            "selected_lambda_reason": self.selected_lambda_reason,
            "rollback_env": [list(p) for p in self.rollback_env],
            "baseline_metrics": dict(self.baseline_metrics),
            "confirmed_metrics": dict(self.confirmed_metrics),
            "section_hit_caveat": self.section_hit_caveat,
            "promotion_target_clarification": (
                self.promotion_target_clarification
            ),
            "human_focus_disclaimer": self.human_focus_disclaimer,
        }


def _baseline_metrics_from_summaries(
    *,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
) -> Dict[str, float]:
    """Pick the metrics the production recommendation card carries."""
    return {
        "primary_score": float(baseline_summary_gold.primary_score),
        "weighted_hit_at_5": float(
            baseline_summary_gold.weighted_hit_at_5
        ),
        "weighted_mrr_at_10": float(
            baseline_summary_gold.weighted_mrr_at_10
        ),
        "weighted_ndcg_at_10": float(
            baseline_summary_gold.weighted_ndcg_at_10
        ),
        "silver_hit_at_5": float(baseline_summary_silver.hit_at_5),
        "subpage_named_weighted_hit_at_5": float(
            _bucket_weighted_hit_at_5(
                baseline_summary_gold, "subpage_named",
            ) or 0.0
        ),
        "subpage_generic_weighted_hit_at_5": float(
            _bucket_weighted_hit_at_5(
                baseline_summary_gold, "subpage_generic",
            ) or 0.0
        ),
        "main_work_weighted_hit_at_5": float(
            _bucket_weighted_hit_at_5(
                baseline_summary_gold, "main_work",
            ) or 0.0
        ),
        "section_hit_at_5": float(
            baseline_summary_gold.section_hit_at_5_when_defined or 0.0
        ),
    }


def _confirmed_metrics_from_score(score: CandidateScore) -> Dict[str, float]:
    """Pick the metrics the production recommendation card carries."""
    return {
        "primary_score": float(score.primary_score),
        "weighted_hit_at_5": float(score.weighted_hit_at_5),
        "weighted_mrr_at_10": float(score.weighted_mrr_at_10),
        "weighted_ndcg_at_10": float(score.weighted_ndcg_at_10),
        "silver_hit_at_5": float(score.silver_hit_at_5),
        "subpage_named_weighted_hit_at_5": float(
            score.subpage_named_weighted_hit_at_5 or 0.0
        ),
        "subpage_generic_weighted_hit_at_5": float(
            score.subpage_generic_weighted_hit_at_5 or 0.0
        ),
        "main_work_weighted_hit_at_5": float(
            score.main_work_weighted_hit_at_5 or 0.0
        ),
        "section_hit_at_5": float(score.section_hit_at_5 or 0.0),
        "delta_primary_score": float(
            score.deltas.get("primary_score", 0.0)
        ),
        "delta_weighted_hit_at_5": float(
            score.deltas.get("weighted_hit_at_5", 0.0)
        ),
        "delta_silver_hit_at_5": float(
            score.deltas.get("silver_hit_at_5", 0.0)
        ),
        "delta_subpage_named_weighted_hit_at_5": float(
            score.deltas.get("subpage_named_weighted_hit_at_5", 0.0)
        ),
        "delta_main_work_weighted_hit_at_5": float(
            score.deltas.get("main_work_weighted_hit_at_5", 0.0)
        ),
        "delta_section_hit_at_5": float(
            score.deltas.get("section_hit_at_5", 0.0)
        ),
    }


def _plateau_lambda_set(
    *,
    result: ConfirmSweepResult,
    candidate_k: int,
    best_primary_score: float,
    epsilon: float,
) -> List[Tuple[float, float]]:
    """Collect (λ, primary) for every grid entry in the same candidate_k
    row whose primary_score is within ``epsilon`` of the best.

    The set INCLUDES the best entry itself. Returned ordered by λ asc.
    """
    out: List[Tuple[float, float]] = []
    for c in result.candidates:
        if c.candidate_k != int(candidate_k):
            continue
        if abs(c.primary_score - float(best_primary_score)) <= float(epsilon):
            out.append((float(c.mmr_lambda), float(c.primary_score)))
    out.sort(key=lambda p: p[0])
    return out


def _default_rollback_env() -> Tuple[Tuple[str, str], ...]:
    """Restore-to-baseline env snippet — pre-promotion shipping config.

    The Phase 7.4 baseline shipping config is ``use_mmr=false`` with
    the production defaults for candidate_k (30) and mmr_lambda (0.7).
    The rollback flips ``USE_MMR`` off; the candidate_k and
    mmr_lambda lines are written commented-out so a reviewer can opt
    to also unset them. ``RAG_TOP_K`` is left alone — top_k=10 was
    already promoted in Phase 7.4.
    """
    return (
        ("AIPIPELINE_WORKER_RAG_USE_MMR", "false"),
        ("# unset (default 30)", "AIPIPELINE_WORKER_RAG_CANDIDATE_K"),
        ("# unset (default 0.7)", "AIPIPELINE_WORKER_RAG_MMR_LAMBDA"),
    )


def select_production_recommended_lambda(
    *,
    result: ConfirmSweepResult,
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
    prefer_lambda: float = PRODUCTION_RECOMMENDED_LAMBDA,
    plateau_epsilon: float = PRODUCTION_PLATEAU_EPSILON,
    rollback_env: Optional[Sequence[Tuple[str, str]]] = None,
) -> Optional[ProductionRecommendation]:
    """Map a metric-best winner to a plateau-aware promotion target.

    Policy:

      1. If no metric-best winner exists → return None (no promotion).
      2. Build the plateau set: every entry in the winner's
         ``candidate_k`` row whose primary_score is within
         ``plateau_epsilon`` of the winner.
      3. If ``prefer_lambda`` is in the plateau → recommend
         ``prefer_lambda``.
         policy = ``LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST``
      4. Otherwise pick the plateau λ closest to ``prefer_lambda``
         (lexicographic tie-break: smallest λ wins on tie).
         policy = ``LAMBDA_POLICY_PLATEAU_NEAREST``
      5. Plateau is empty (single-point peak) → recommend
         ``confirmed_best_lambda`` unchanged.
         policy = ``LAMBDA_POLICY_NO_PLATEAU_FALLBACK``

    The returned record carries the metric-best name, the recommended
    variant name (if a grid match exists), the recommended lambda,
    candidate_k / top_k / use_mmr, the plateau set, the rollback env,
    and a baseline / confirmed metrics snapshot.
    """
    if result.confirmed_best is None:
        return None
    cb = result.confirmed_best
    plateau_set = _plateau_lambda_set(
        result=result,
        candidate_k=cb.candidate_k,
        best_primary_score=cb.primary_score,
        epsilon=plateau_epsilon,
    )
    plateau_lambdas = tuple(plateau_set)
    plateau_lambda_values = [lam for lam, _ in plateau_set]

    # The plateau set always includes the winner itself (Δ=0 from
    # itself). A "real" plateau means at least one *other* point in
    # the same λ-row scored within epsilon — i.e. the picker has more
    # than one option to choose from. If only the winner is in the
    # set, treat it as a single-point peak and fall back to that λ.
    has_real_plateau = len(plateau_set) >= 2

    # Selection.
    if not has_real_plateau:
        recommended_lambda = float(cb.mmr_lambda)
        policy = LAMBDA_POLICY_NO_PLATEAU_FALLBACK
        reason = (
            f"no λ-row plateau within epsilon={plateau_epsilon:.4f} "
            f"(only the metric-best λ={recommended_lambda:.4f} is in the "
            f"set); single-point peak → recommendation stays at the "
            f"metric-best."
        )
    elif any(
        abs(lam - float(prefer_lambda)) < 1e-9 for lam in plateau_lambda_values
    ):
        recommended_lambda = float(prefer_lambda)
        policy = LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST
        reason = (
            f"λ-row at candidate_k={cb.candidate_k} plateaus within "
            f"epsilon={plateau_epsilon:.4f} of the metric-best primary_"
            f"score; recommended λ={recommended_lambda:.4f} matches the "
            f"prior best, lowering the PR explanation cost vs the "
            f"lexicographic tie-break value λ={cb.mmr_lambda:.4f}."
        )
    else:
        # Pick plateau λ closest to prefer_lambda; ties → smallest λ.
        plateau_set_sorted = sorted(
            plateau_set,
            key=lambda p: (abs(p[0] - float(prefer_lambda)), p[0]),
        )
        recommended_lambda = float(plateau_set_sorted[0][0])
        policy = LAMBDA_POLICY_PLATEAU_NEAREST
        reason = (
            f"λ-row plateau exists but does not include prefer_lambda="
            f"{float(prefer_lambda):.4f}; recommended λ="
            f"{recommended_lambda:.4f} is the plateau entry closest to "
            f"the prior best (epsilon={plateau_epsilon:.4f})."
        )

    # Resolve the recommended variant name on the grid (best-effort).
    recommended_variant_name: Optional[str] = None
    for spec in result.grid:
        if (
            spec.candidate_k == cb.candidate_k
            and abs(spec.mmr_lambda - recommended_lambda) < 1e-9
        ):
            recommended_variant_name = spec.name
            break

    # Resolve the cache dir from the metric-best's grid spec.
    spec_cb = next(g for g in result.grid if g.name == cb.name)

    # Baseline + confirmed metric snapshot for reviewer reference.
    baseline_metrics = _baseline_metrics_from_summaries(
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
    )
    confirmed_metrics = _confirmed_metrics_from_score(cb)

    return ProductionRecommendation(
        confirmed_best_name=cb.name,
        confirmed_best_lambda=float(cb.mmr_lambda),
        confirmed_best_primary_score=float(cb.primary_score),
        recommended_lambda=recommended_lambda,
        recommended_variant_name=recommended_variant_name,
        candidate_k=int(cb.candidate_k),
        top_k=int(spec_cb.top_k),
        use_mmr=bool(spec_cb.use_mmr),
        cache_dir_relative=spec_cb.cache_dir_relative,
        plateau_status=(
            result.plateau.status if result.plateau else PLATEAU_OVERFIT_WARNING
        ),
        plateau_lambdas=plateau_lambdas,
        selected_lambda_policy=policy,
        selected_lambda_reason=reason,
        rollback_env=tuple(
            tuple(p) for p in (
                rollback_env if rollback_env is not None
                else _default_rollback_env()
            )
        ),
        baseline_metrics=baseline_metrics,
        confirmed_metrics=confirmed_metrics,
    )


def render_production_recommendation_section(
    *,
    recommendation: ProductionRecommendation,
    section_caveat_section_hit_at_5: Optional[Tuple[float, float]] = None,
) -> str:
    """Render the ``## Production recommendation`` markdown section.

    The renderer is pure: it takes the structured recommendation +
    optional ``(baseline, candidate)`` section_hit pair (so the caveat
    sub-section can carry concrete numbers) and returns the markdown
    chunk. The caller decides whether to splice it into an existing
    confirm_sweep_report.md or write it standalone.
    """
    lines: List[str] = []
    lines.append("## Production recommendation")
    lines.append("")
    lines.append(
        f"- confirmed best by metric: `{recommendation.confirmed_best_name}` "
        f"(candidate_k={recommendation.candidate_k}, MMR enabled, λ-plateau)"
    )
    lines.append(
        f"- production recommended lambda: **"
        f"{recommendation.recommended_lambda:.4f}**"
    )
    if recommendation.recommended_variant_name is not None:
        lines.append(
            f"- production recommended variant: "
            f"`{recommendation.recommended_variant_name}`"
        )
    lines.append(
        f"- selected_lambda_policy: `{recommendation.selected_lambda_policy}`"
    )
    lines.append(f"- reason: {recommendation.selected_lambda_reason}")
    if recommendation.plateau_lambdas:
        lams = ", ".join(
            f"λ={lam:.2f} → {score:.4f}"
            for lam, score in recommendation.plateau_lambdas
        )
        lines.append(
            f"- plateau set (candidate_k={recommendation.candidate_k}): {lams}"
        )
    lines.append("")
    lines.append("### Why not the metric-best λ?")
    lines.append("")
    lines.append(
        f"- the lexicographic tie-break value is "
        f"λ={recommendation.confirmed_best_lambda:.4f}"
    )
    lines.append(
        f"- λ={recommendation.recommended_lambda:.4f} sits on the same "
        f"plateau (within epsilon)"
    )
    lines.append(
        "- λ=0.70 (the prior Phase 7.x best) is more familiar to a "
        "reviewer; promoting at that value keeps the PR a single-knob "
        "change (`USE_MMR=true`, widen `CANDIDATE_K`) instead of also "
        "moving λ"
    )
    lines.append("")
    lines.append("### Improvement summary (vs baseline)")
    lines.append("")
    cm = recommendation.confirmed_metrics
    bm = recommendation.baseline_metrics

    def _delta(name: str) -> str:
        d = cm.get(f"delta_{name}")
        if d is None:
            d = float(cm.get(name, 0.0)) - float(bm.get(name, 0.0))
        return f"{d:+.4f}"

    lines.append(
        f"- gold primary_score: {bm.get('primary_score', 0.0):.4f} → "
        f"{cm.get('primary_score', 0.0):.4f} ({_delta('primary_score')})"
    )
    lines.append(
        f"- gold weighted_hit@5: {bm.get('weighted_hit_at_5', 0.0):.4f} → "
        f"{cm.get('weighted_hit_at_5', 0.0):.4f} "
        f"({_delta('weighted_hit_at_5')})"
    )
    lines.append(
        f"- silver hit@5: {bm.get('silver_hit_at_5', 0.0):.4f} → "
        f"{cm.get('silver_hit_at_5', 0.0):.4f} ({_delta('silver_hit_at_5')})"
    )
    lines.append(
        f"- subpage_named weighted_hit@5: "
        f"{bm.get('subpage_named_weighted_hit_at_5', 0.0):.4f} → "
        f"{cm.get('subpage_named_weighted_hit_at_5', 0.0):.4f} "
        f"({_delta('subpage_named_weighted_hit_at_5')})"
    )
    lines.append(
        f"- main_work weighted_hit@5: "
        f"{bm.get('main_work_weighted_hit_at_5', 0.0):.4f} → "
        f"{cm.get('main_work_weighted_hit_at_5', 0.0):.4f} "
        f"({_delta('main_work_weighted_hit_at_5')})"
    )
    lines.append("")

    # Section hit caveat — concrete numbers if provided.
    lines.append("### Known caveat: section_hit@5")
    lines.append("")
    if section_caveat_section_hit_at_5 is not None:
        b, c = section_caveat_section_hit_at_5
        lines.append(
            f"- baseline {b:.4f} → recommended {c:.4f} "
            f"(Δ={c - b:+.4f})"
        )
    else:
        lines.append(
            f"- baseline section_hit@5: "
            f"{bm.get('section_hit_at_5', 0.0):.4f}"
        )
        lines.append(
            f"- recommended section_hit@5: "
            f"{cm.get('section_hit_at_5', 0.0):.4f}"
        )
    lines.append(
        "- production blocker? **NO** — caveat documented in the main "
        "report; deferred to Phase 7.6 section-aware reranking."
    )
    lines.append("")

    # Recommended env snippet (paste-target).
    lines.append("### Recommended env snippet")
    lines.append("")
    lines.append("```")
    lines.append(f"AIPIPELINE_WORKER_RAG_TOP_K={recommendation.top_k}")
    lines.append(
        f"AIPIPELINE_WORKER_RAG_CANDIDATE_K={recommendation.candidate_k}"
    )
    lines.append(
        f"AIPIPELINE_WORKER_RAG_USE_MMR="
        f"{'true' if recommendation.use_mmr else 'false'}"
    )
    lines.append(
        f"AIPIPELINE_WORKER_RAG_MMR_LAMBDA="
        f"{recommendation.recommended_lambda:.4f}"
    )
    lines.append("```")
    lines.append("")

    # Fallback configs.
    lines.append("### Fallback configs")
    lines.append("")
    lines.append(
        "- intermediate fallback (smaller candidate_k pool, same λ): "
        f"`candidate_k=30, mmr_lambda={recommendation.recommended_lambda:.4f}, "
        "use_mmr=true` — matches the Phase 7.x first-pass best."
    )
    lines.append(
        "- full rollback (restore baseline retrieval path): set "
        "`AIPIPELINE_WORKER_RAG_USE_MMR=false` and unset / drop "
        "`AIPIPELINE_WORKER_RAG_CANDIDATE_K` + "
        "`AIPIPELINE_WORKER_RAG_MMR_LAMBDA` overrides."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_production_recommended_config_env(
    path: Path, *, recommendation: ProductionRecommendation,
) -> Path:
    """Emit the env-snippet form of the production-recommended config.

    Distinct from ``best_config.confirmed.env`` in that the λ value is
    chosen by the plateau-aware policy, not by the metric-best
    lexicographic tie-break. The promotion-target clarification and
    section-hit caveat are carried as comments so a reviewer pasting
    this into a PR description / config-change ticket does not lose
    context.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Phase 7.5 confirm-sweep — PRODUCTION-RECOMMENDED retrieval config",
        "# DO NOT auto-apply. Promote via the standard config-change PR.",
        f"# {recommendation.human_focus_disclaimer}",
        f"# Promotion target: {recommendation.promotion_target_clarification}",
        f"# selected_lambda_policy={recommendation.selected_lambda_policy}",
        f"# selected_lambda_reason={recommendation.selected_lambda_reason}",
        f"# metric_best_variant={recommendation.confirmed_best_name}",
        f"# metric_best_lambda={recommendation.confirmed_best_lambda:.4f}",
        f"# metric_best_primary_score="
        f"{recommendation.confirmed_best_primary_score:.6f}",
        f"# plateau_status={recommendation.plateau_status}",
    ]
    if recommendation.plateau_lambdas:
        plateau_str = ", ".join(
            f"({lam:.2f},{score:.4f})"
            for lam, score in recommendation.plateau_lambdas
        )
        lines.append(f"# plateau_lambdas={plateau_str}")
    lines.append(f"AIPIPELINE_WORKER_RAG_TOP_K={recommendation.top_k}")
    lines.append(
        f"AIPIPELINE_WORKER_RAG_CANDIDATE_K={recommendation.candidate_k}"
    )
    lines.append(
        f"AIPIPELINE_WORKER_RAG_USE_MMR="
        f"{'true' if recommendation.use_mmr else 'false'}"
    )
    lines.append(
        f"AIPIPELINE_WORKER_RAG_MMR_LAMBDA="
        f"{recommendation.recommended_lambda:.4f}"
    )
    lines.append(
        f"# index variant cache: {recommendation.cache_dir_relative}"
    )
    lines.append("# Rollback (restore pre-promotion baseline):")
    for left, right in recommendation.rollback_env:
        if left.startswith("#"):
            # Marker for an env name to recommend unsetting.
            lines.append(f"# {right}=  {left}")
        else:
            lines.append(f"# {left}={right}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_production_recommended_config_json(
    path: Path, *, recommendation: ProductionRecommendation,
) -> Path:
    """Emit the JSON form of the production-recommended config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(recommendation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


_PRODUCTION_RECO_HEADER = "## Production recommendation"


def append_production_recommendation_to_report(
    path: Path,
    *,
    recommendation: ProductionRecommendation,
    section_caveat_section_hit_at_5: Optional[Tuple[float, float]] = None,
) -> Path:
    """Splice the production-recommendation section into an existing report.

    If the report already contains a ``## Production recommendation``
    heading, the existing block (up to the next ``## `` heading) is
    replaced; otherwise the new section is appended just before the
    ``## Reminders`` heading (or at the end of the file as a fallback).
    """
    md = path.read_text(encoding="utf-8")
    section = render_production_recommendation_section(
        recommendation=recommendation,
        section_caveat_section_hit_at_5=section_caveat_section_hit_at_5,
    )

    if _PRODUCTION_RECO_HEADER in md:
        before, _, rest = md.partition(_PRODUCTION_RECO_HEADER)
        # Find the next "## " heading to know where the existing block
        # ends. partition gives us "rest = '...existing body...\n## Next'".
        next_heading_idx = rest.find("\n## ")
        if next_heading_idx == -1:
            # No following heading → recommendation was the last section.
            new_md = before.rstrip() + "\n\n" + section
        else:
            after = rest[next_heading_idx + 1:]
            new_md = (
                before.rstrip() + "\n\n" + section + "\n" + after
            )
    else:
        if "## Reminders" in md:
            head, _, tail = md.partition("## Reminders")
            new_md = head.rstrip() + "\n\n" + section + "\n## Reminders" + tail
        else:
            new_md = md.rstrip() + "\n\n" + section

    # Belt-and-suspenders: never emit a forbidden phrase via this writer.
    for phrase in FORBIDDEN_PROMOTION_TARGET_PHRASES:
        if phrase in new_md:
            raise ValueError(
                f"append_production_recommendation_to_report produced a "
                f"forbidden phrase: {phrase!r}."
            )
    path.write_text(new_md, encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_CANDIDATE_K_GRID",
    "DEFAULT_MMR_LAMBDA_GRID",
    "DEFAULT_TOP_K",
    "MMR_DOC_ID_PENALTY",
    "PRODUCTION_INDEX_CACHE_DIR",
    "PRODUCTION_RAG_CHUNKS_PATH",
    "PRODUCTION_RECOMMENDED_LAMBDA",
    "PRODUCTION_PLATEAU_EPSILON",
    "MAIN_WORK_REGRESSION_THRESHOLD",
    "SUBPAGE_NAMED_HOLD_THRESHOLD",
    "SECTION_HIT_HALVING_FACTOR",
    "PLATEAU_PRIMARY_DELTA",
    "SUBPAGE_NAMED_NOT_FIXED_WARNING",
    "MAIN_WORK_REGRESSION_WARNING",
    "SECTION_RETRIEVAL_WARNING",
    "PLATEAU_OK",
    "PLATEAU_OVERFIT_WARNING",
    "LAMBDA_POLICY_PLATEAU_PREVIOUS_BEST",
    "LAMBDA_POLICY_PLATEAU_NEAREST",
    "LAMBDA_POLICY_NO_PLATEAU_FALLBACK",
    "PROMOTION_TARGET_CLARIFICATION",
    "SECTION_HIT_CAVEAT",
    "FORBIDDEN_PROMOTION_TARGET_PHRASES",
    "SweepCandidate",
    "CandidateScore",
    "ConfirmSweepResult",
    "PlateauAnalysis",
    "ProductionRecommendation",
    "make_confirm_sweep_grid",
    "mmr_select_post_hoc",
    "apply_variant_to_candidates",
    "evaluate_main_work_guardrail",
    "evaluate_subpage_named_hold",
    "evaluate_section_hit_caveat",
    "analyze_plateau",
    "select_confirmed_best",
    "select_production_recommended_lambda",
    "render_confirm_sweep_report",
    "render_production_recommendation_section",
    "write_confirm_sweep_results_jsonl",
    "write_confirm_sweep_summary_json",
    "write_confirm_sweep_report_md",
    "write_confirmed_best_config_json",
    "write_confirmed_best_config_env",
    "write_production_recommended_config_env",
    "write_production_recommended_config_json",
    "append_production_recommendation_to_report",
]
