"""Phase 7.6 — section-aware reranking experiment harness (scaffold).

Phase 7.5's MMR confirm sweep landed a meaningful page-level win on
the gold-50 focus set but section_hit@5 fell from 0.0455 → 0.0227 on
a tiny defined-only subset. Phase 7.6 is the experiment that figures
out whether that section-level regression is real or an artefact of
a brittle small-base metric.

This module is a *scaffold*. The strategy specs and the grid
generator are fully implemented and tested; the production-deployable
"page-first then section rerank" path is implemented as well. The
*diagnostic-only* paths (section prefix bonus, supporting-chunk
proximity bonus) are also implemented but explicitly flagged in the
strategy record as ``deployable=False`` so the report renderer cannot
accidentally promote one of them.

The CLI in ``scripts.run_phase7_section_aware_rerank`` wires this
harness up to the cached candidate pool from Phase 7.5; the harness
itself is pure-Python and depends only on the
:class:`RetrievedDoc` type from ``eval.harness.phase7_human_gold_tune``.

Design contract:

  * **No production retriever code is imported here.** Reranking
    operates on pre-computed :class:`RetrievedDoc` lists; the live
    FAISS / bge-m3 wiring lives in the CLI.
  * **Oracle-grounded variants are clearly labelled** so the report
    renderer cannot frame them as production-deployable. The
    ``deployable`` flag on :class:`SectionRerankSpec` is the source
    of truth.
  * **Phase 7.5's promoted config is the new baseline** — Phase 7.6
    is asking "can section-aware rerank improve on Phase 7.5?", not
    "is Phase 7.5 itself an improvement?".
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from eval.harness.phase7_human_gold_tune import RetrievedDoc
from eval.harness.phase7_mmr_confirm_sweep import (
    DEFAULT_TOP_K,
    MMR_DOC_ID_PENALTY,
    PRODUCTION_RECOMMENDED_LAMBDA,
    apply_variant_to_candidates,
    mmr_select_post_hoc,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy strings — the report renderer keys off these. Don't rename.
# ---------------------------------------------------------------------------


STRATEGY_BASELINE_NO_RERANK: str = "baseline_no_section_rerank"
STRATEGY_SECTION_BONUS: str = "section_bonus"
STRATEGY_SUPPORTING_CHUNK_PROXIMITY: str = "supporting_chunk_proximity"
STRATEGY_PAGE_FIRST_SECTION_RERANK: str = "page_first_section_rerank"
STRATEGY_SAME_PAGE_CHUNK_RERANK: str = "same_page_chunk_rerank"
STRATEGY_ANSWERABILITY_AUDIT_PREP: str = "answerability_audit_prep"


# Default Phase 7.5 promoted config used as the post-hoc rerank input
# pool. Variants in the Phase 7.6 sweep operate on top of this.
PHASE_7_5_RECOMMENDED_CANDIDATE_K: int = 40
PHASE_7_5_RECOMMENDED_LAMBDA: float = PRODUCTION_RECOMMENDED_LAMBDA
PHASE_7_5_RECOMMENDED_TOP_K: int = DEFAULT_TOP_K


# Section-bonus grid axis defaults.
DEFAULT_SECTION_BONUS_VALUES: Tuple[float, ...] = (0.05, 0.10, 0.15)


# Phase 7.6 guardrail thresholds (vs the Phase 7.5 production-
# recommended config, NOT vs the original Phase 7.4 baseline).
PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD: float = 0.02  # 2pp on gold primary
PHASE_7_6_SILVER_REGRESSION_THRESHOLD: float = 0.03   # 3pp on silver hit@5
PHASE_7_6_SUBPAGE_NAMED_THRESHOLD: float = 0.05       # 5pp


# ---------------------------------------------------------------------------
# Strategy spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionRerankSpec:
    """One rerank variant to try in the Phase 7.6 sweep.

    ``deployable`` is the contract that distinguishes diagnostic-only
    candidates (which need oracle access at inference time, e.g. the
    expected_section_path or the supporting_chunk_id) from
    production-deployable strategies that only use the query +
    candidate metadata.

    The renderer reads ``deployable`` to gate the
    ``promote-to-production`` recommendation: a non-deployable variant
    can only ever earn a "best diagnostic upper bound" tag, never a
    production-promotion recommendation.
    """

    name: str
    strategy: str
    deployable: bool
    base_candidate_k: int = PHASE_7_5_RECOMMENDED_CANDIDATE_K
    base_mmr_lambda: float = PHASE_7_5_RECOMMENDED_LAMBDA
    base_top_k: int = PHASE_7_5_RECOMMENDED_TOP_K
    section_bonus: float = 0.0
    page_first_inner_top_k: int = 5  # used by page-first variants
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_section_rerank_grid(
    *,
    section_bonus_values: Sequence[float] = DEFAULT_SECTION_BONUS_VALUES,
    include_baseline: bool = True,
    include_supporting_chunk_proximity: bool = True,
    include_page_first: bool = True,
    include_same_page_rerank: bool = True,
) -> List[SectionRerankSpec]:
    """Build the Phase 7.6 sweep grid.

    Order matters — the report renderer prints rows in grid order, and
    the test suite pins the row count + first row.
    """
    grid: List[SectionRerankSpec] = []

    if include_baseline:
        grid.append(SectionRerankSpec(
            name="baseline_phase7_5_recommended",
            strategy=STRATEGY_BASELINE_NO_RERANK,
            deployable=True,
            description=(
                "Phase 7.5 production-recommended: candidate_k=40, MMR "
                "λ=0.70. No section-aware rerank. New baseline for "
                "Phase 7.6."
            ),
        ))

    for bonus in section_bonus_values:
        grid.append(SectionRerankSpec(
            name=f"section_bonus_{int(round(bonus * 100)):03d}",
            strategy=STRATEGY_SECTION_BONUS,
            deployable=False,  # needs expected_section_path → diagnostic
            section_bonus=float(bonus),
            description=(
                f"Add a {bonus:+.2f} bonus to candidates whose "
                f"section_path matches the gold expected_section_path "
                f"(prefix or substring). DIAGNOSTIC ONLY — needs "
                f"oracle access, not production-deployable."
            ),
        ))

    if include_supporting_chunk_proximity:
        grid.append(SectionRerankSpec(
            name="supporting_chunk_proximity",
            strategy=STRATEGY_SUPPORTING_CHUNK_PROXIMITY,
            deployable=False,
            description=(
                "Boost candidates that share the gold row's "
                "human_supporting_chunk_id page AND a small "
                "section-path edit distance. DIAGNOSTIC ONLY — needs "
                "oracle supporting-chunk annotation."
            ),
        ))

    if include_page_first:
        grid.append(SectionRerankSpec(
            name="page_first_section_rerank_overlap",
            strategy=STRATEGY_PAGE_FIRST_SECTION_RERANK,
            deployable=True,
            description=(
                "Two-pass rerank: page-level diversification then "
                "section-name token overlap with the query inside each "
                "retained page. PRODUCTION-DEPLOYABLE."
            ),
        ))

    if include_same_page_rerank:
        grid.append(SectionRerankSpec(
            name="same_page_chunk_rerank",
            strategy=STRATEGY_SAME_PAGE_CHUNK_RERANK,
            deployable=True,
            description=(
                "Within each page in the Phase 7.5 top-k, swap the "
                "represented chunk for the page's best section-name "
                "overlap chunk. PRODUCTION-DEPLOYABLE."
            ),
        ))

    return grid


# ---------------------------------------------------------------------------
# Section bonus scoring — DIAGNOSTIC ONLY
# ---------------------------------------------------------------------------


def _section_path_overlap(
    candidate_path: Sequence[str],
    expected_path: Sequence[str],
) -> float:
    """Score a candidate's section_path against the gold expected_path.

    Returns:
      * 1.0 when the candidate's section_path *equals* the expected
      * 0.5 when the expected_path is a prefix of the candidate's
        path (deeper section under the expected one)
      * 0.5 when the candidate's path is a prefix of the expected
        (broader section that contains the expected one)
      * a substring-style score in [0, 0.4] when the leaf section
        names share token overlap
      * 0.0 when nothing matches

    The scoring is intentionally coarse — section_path is noisy text
    and we want a stable score that survives small typographical
    variation without leaking a near-zero signal everywhere.
    """
    cand = tuple(s.strip() for s in candidate_path if s.strip())
    exp = tuple(s.strip() for s in expected_path if s.strip())
    if not cand or not exp:
        return 0.0
    if cand == exp:
        return 1.0
    # Prefix match either direction.
    if len(exp) <= len(cand) and cand[: len(exp)] == exp:
        return 0.5
    if len(cand) <= len(exp) and exp[: len(cand)] == cand:
        return 0.5
    # Token overlap on the leaf segments.
    cand_leaf_tokens = set(cand[-1].split())
    exp_leaf_tokens = set(exp[-1].split())
    if not cand_leaf_tokens or not exp_leaf_tokens:
        return 0.0
    overlap = cand_leaf_tokens & exp_leaf_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / float(len(exp_leaf_tokens))
    return min(0.4, 0.4 * coverage)


def apply_section_bonus_post_hoc(
    docs: Sequence[RetrievedDoc],
    *,
    expected_section_path: Sequence[str],
    bonus: float,
    top_k: int,
) -> List[RetrievedDoc]:
    """Re-rank ``docs`` by ``score + bonus * section_overlap``.

    Diagnostic-only path (needs oracle ``expected_section_path``).
    Returns the top ``top_k`` after the rerank, with renumbered ranks.

    When ``expected_section_path`` is empty / unset, the function is a
    no-op pass-through that just truncates to ``top_k`` — this keeps
    the grid step uniform across rows (some gold rows do not define
    a section path).
    """
    if not docs:
        return []
    if not expected_section_path or float(bonus) == 0.0:
        return [
            RetrievedDoc(
                rank=i + 1, chunk_id=d.chunk_id, page_id=d.page_id,
                title=d.title, section_path=d.section_path, score=d.score,
            )
            for i, d in enumerate(list(docs)[: int(top_k)])
        ]

    scored: List[Tuple[float, int, RetrievedDoc]] = []
    for original_idx, d in enumerate(docs):
        base = float(d.score) if d.score is not None else 0.0
        overlap = _section_path_overlap(d.section_path, expected_section_path)
        rerank_score = base + float(bonus) * overlap
        scored.append((rerank_score, original_idx, d))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out: List[RetrievedDoc] = []
    for new_rank, (_, _, d) in enumerate(scored[: int(top_k)], start=1):
        out.append(RetrievedDoc(
            rank=new_rank, chunk_id=d.chunk_id, page_id=d.page_id,
            title=d.title, section_path=d.section_path, score=d.score,
        ))
    return out


# ---------------------------------------------------------------------------
# Same-page chunk rerank — PRODUCTION-DEPLOYABLE
# ---------------------------------------------------------------------------


def _query_section_overlap_score(
    query_tokens: Sequence[str], section_path: Sequence[str],
) -> float:
    """Token-overlap score between the query and a candidate's section_path.

    Production-deployable: uses only the query and the candidate
    metadata. Returns coverage of section-tokens by query-tokens —
    "section_path words present in the query, normalised by section
    word count". Empty section / empty query → 0.
    """
    qt = {t for t in query_tokens if t}
    st: List[str] = []
    for s in section_path:
        for tok in s.split():
            tok = tok.strip()
            if tok:
                st.append(tok)
    if not qt or not st:
        return 0.0
    overlap = sum(1 for s in st if s in qt)
    if overlap == 0:
        return 0.0
    return float(overlap) / float(len(st))


def apply_same_page_chunk_rerank(
    docs: Sequence[RetrievedDoc],
    *,
    full_page_chunks: Mapping[str, Sequence[RetrievedDoc]],
    query: str,
    top_k: int,
) -> List[RetrievedDoc]:
    """Within each retained page in ``docs``, swap the picked chunk
    for the page's best section-name-overlap chunk.

    ``full_page_chunks`` is a per-page list of candidate chunks the
    eval driver supplied (typically the FAISS top-N within that page,
    or the entire page's chunk inventory when the index supports
    page-grouped queries). When a page is not present in
    ``full_page_chunks`` the original chunk is kept.

    Production-deployable: uses only the query and the candidate
    metadata. Idempotent in the absence of a better same-page
    overlap. Returns the same number of rows as ``docs[:top_k]``,
    with ranks renumbered.
    """
    if not docs:
        return []
    query_tokens = [t.strip().lower() for t in query.split() if t.strip()]
    out: List[RetrievedDoc] = []
    for d in list(docs)[: int(top_k)]:
        page_chunks = full_page_chunks.get(d.page_id) or []
        best = d
        best_score = _query_section_overlap_score(
            query_tokens, d.section_path,
        )
        for pc in page_chunks:
            s = _query_section_overlap_score(query_tokens, pc.section_path)
            if s > best_score + 1e-9:
                best = pc
                best_score = s
        out.append(RetrievedDoc(
            rank=len(out) + 1,
            chunk_id=best.chunk_id, page_id=best.page_id,
            title=best.title, section_path=best.section_path,
            score=best.score,
        ))
    return out


# ---------------------------------------------------------------------------
# Page-first then section rerank — PRODUCTION-DEPLOYABLE
# ---------------------------------------------------------------------------


def apply_page_first_section_rerank(
    pool: Sequence[RetrievedDoc],
    *,
    query: str,
    candidate_k: int = PHASE_7_5_RECOMMENDED_CANDIDATE_K,
    mmr_lambda: float = PHASE_7_5_RECOMMENDED_LAMBDA,
    top_k: int = PHASE_7_5_RECOMMENDED_TOP_K,
    inner_top_k: int = 5,
) -> List[RetrievedDoc]:
    """Two-pass rerank: page-level MMR then within-page section overlap.

    Pass 1: page-level diversification on the wide pool (the Phase
    7.5 promoted config). Pass 2: within each page that contributed
    a chunk to the top-k, look at *all* candidates in the pool from
    the same page (up to ``inner_top_k``) and rerank them by query /
    section_path token overlap; pick the best.

    The function does NOT re-do FAISS retrieval — it only operates on
    ``pool`` (typically the cached candidate pool). Production
    deployability is preserved because the second pass uses only
    query + candidate metadata.
    """
    if not pool:
        return []
    # Pass 1: standard Phase 7.5 path.
    page_top = apply_variant_to_candidates(
        pool,
        candidate_k=int(candidate_k),
        use_mmr=True,
        mmr_lambda=float(mmr_lambda),
        top_k=int(top_k),
    )
    # Build the within-page candidate map from the full pool.
    by_page: Dict[str, List[RetrievedDoc]] = {}
    for d in pool:
        by_page.setdefault(d.page_id, []).append(d)
    # Pass 2: within-page rerank.
    return apply_same_page_chunk_rerank(
        page_top,
        full_page_chunks={k: v[: int(inner_top_k)] for k, v in by_page.items()},
        query=query,
        top_k=int(top_k),
    )


# ---------------------------------------------------------------------------
# Variant evaluation glue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionRerankInput:
    """Per-query inputs the harness needs to drive one variant.

    ``query`` and ``candidate_pool`` come from the cached pool.
    ``expected_section_path`` and ``supporting_chunk_id`` are oracle
    fields the diagnostic-only strategies read; they are None / empty
    for variants that don't need them.
    """

    query_id: str
    query: str
    candidate_pool: Tuple[RetrievedDoc, ...]
    expected_page_id: str = ""
    expected_section_path: Tuple[str, ...] = ()
    supporting_chunk_id: str = ""


def run_variant_for_query(
    *,
    spec: SectionRerankSpec,
    inputs: SectionRerankInput,
) -> List[RetrievedDoc]:
    """Apply ``spec``'s strategy to one query's candidate pool.

    Returns the variant's top-k. Strategies that do not have an
    implementation here (e.g. ``STRATEGY_ANSWERABILITY_AUDIT_PREP``)
    fall through to the Phase 7.5 baseline behaviour — they are not
    rerank strategies, only data-collection hooks.
    """
    pool = inputs.candidate_pool
    base_top_k = int(spec.base_top_k)
    base_candidate_k = int(spec.base_candidate_k)
    base_lambda = float(spec.base_mmr_lambda)

    if spec.strategy == STRATEGY_BASELINE_NO_RERANK:
        return apply_variant_to_candidates(
            pool,
            candidate_k=base_candidate_k,
            use_mmr=True,
            mmr_lambda=base_lambda,
            top_k=base_top_k,
        )

    if spec.strategy == STRATEGY_SECTION_BONUS:
        # Run the Phase 7.5 baseline first, then apply the bonus.
        prelim = apply_variant_to_candidates(
            pool,
            candidate_k=base_candidate_k,
            use_mmr=True,
            mmr_lambda=base_lambda,
            top_k=base_candidate_k,  # keep wide pool for the rerank
        )
        return apply_section_bonus_post_hoc(
            prelim,
            expected_section_path=inputs.expected_section_path,
            bonus=float(spec.section_bonus),
            top_k=base_top_k,
        )

    if spec.strategy == STRATEGY_SUPPORTING_CHUNK_PROXIMITY:
        # Boost any candidate that shares the supporting-chunk's page.
        # Diagnostic only — gold-50 has only ≈22 rows with a defined
        # supporting chunk id.
        if not inputs.supporting_chunk_id:
            return apply_variant_to_candidates(
                pool,
                candidate_k=base_candidate_k,
                use_mmr=True,
                mmr_lambda=base_lambda,
                top_k=base_top_k,
            )
        # Use the expected_page_id from the supporting chunk's
        # context. The supporting_chunk_id alone doesn't tell us the
        # page; the SectionRerankInput provides expected_page_id
        # alongside.
        target_page = inputs.expected_page_id
        if not target_page:
            return apply_variant_to_candidates(
                pool,
                candidate_k=base_candidate_k,
                use_mmr=True,
                mmr_lambda=base_lambda,
                top_k=base_top_k,
            )
        prelim = apply_variant_to_candidates(
            pool,
            candidate_k=base_candidate_k,
            use_mmr=True,
            mmr_lambda=base_lambda,
            top_k=base_candidate_k,
        )
        bonus = float(spec.section_bonus or 0.10)
        scored: List[Tuple[float, int, RetrievedDoc]] = []
        for original_idx, d in enumerate(prelim):
            base = float(d.score) if d.score is not None else 0.0
            chunk_match = (
                bonus * 1.5 if d.chunk_id == inputs.supporting_chunk_id
                else (bonus if d.page_id == target_page else 0.0)
            )
            scored.append((base + chunk_match, original_idx, d))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out: List[RetrievedDoc] = []
        for new_rank, (_, _, d) in enumerate(scored[: base_top_k], start=1):
            out.append(RetrievedDoc(
                rank=new_rank, chunk_id=d.chunk_id, page_id=d.page_id,
                title=d.title, section_path=d.section_path, score=d.score,
            ))
        return out

    if spec.strategy == STRATEGY_PAGE_FIRST_SECTION_RERANK:
        return apply_page_first_section_rerank(
            pool,
            query=inputs.query,
            candidate_k=base_candidate_k,
            mmr_lambda=base_lambda,
            top_k=base_top_k,
            inner_top_k=int(spec.page_first_inner_top_k),
        )

    if spec.strategy == STRATEGY_SAME_PAGE_CHUNK_RERANK:
        # Build the per-page chunk map from the full candidate pool.
        by_page: Dict[str, List[RetrievedDoc]] = {}
        for d in pool:
            by_page.setdefault(d.page_id, []).append(d)
        prelim = apply_variant_to_candidates(
            pool,
            candidate_k=base_candidate_k,
            use_mmr=True,
            mmr_lambda=base_lambda,
            top_k=base_top_k,
        )
        return apply_same_page_chunk_rerank(
            prelim,
            full_page_chunks={
                k: v[: int(spec.page_first_inner_top_k)]
                for k, v in by_page.items()
            },
            query=inputs.query,
            top_k=base_top_k,
        )

    if spec.strategy == STRATEGY_ANSWERABILITY_AUDIT_PREP:
        # Pure passthrough — this strategy only freezes the top-k for
        # later answerability audit; it doesn't rerank.
        return apply_variant_to_candidates(
            pool,
            candidate_k=base_candidate_k,
            use_mmr=True,
            mmr_lambda=base_lambda,
            top_k=base_top_k,
        )

    raise ValueError(f"unknown rerank strategy: {spec.strategy!r}")


# ---------------------------------------------------------------------------
# Section / page hit metrics — minimal renderer-helpful aggregates.
# ---------------------------------------------------------------------------


def page_hit_at_k(
    docs: Sequence[RetrievedDoc],
    expected_page_id: str,
    *,
    k: int = 5,
) -> int:
    if not expected_page_id or k <= 0:
        return 0
    for d in docs[: int(k)]:
        if d.page_id == expected_page_id:
            return 1
    return 0


def section_hit_at_k(
    docs: Sequence[RetrievedDoc],
    expected_section_path: Sequence[str],
    *,
    k: int = 5,
) -> Optional[int]:
    if not expected_section_path:
        return None
    if k <= 0:
        return 0
    target = tuple(s.strip() for s in expected_section_path if s.strip())
    if not target:
        return None
    for d in docs[: int(k)]:
        cand = tuple(s.strip() for s in d.section_path if s.strip())
        if cand == target:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Persistence helpers — small JSON renderers, paired with the report.
# ---------------------------------------------------------------------------


def write_section_rerank_grid_json(
    path: Path, grid: Sequence[SectionRerankSpec],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "7.6_section_aware_rerank",
        "grid_size": len(grid),
        "grid": [g.to_dict() for g in grid],
        "guardrail_thresholds": {
            "primary_score_drop_max_vs_phase_7_5": (
                PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD
            ),
            "silver_hit_at_5_drop_max_vs_phase_7_5": (
                PHASE_7_6_SILVER_REGRESSION_THRESHOLD
            ),
            "subpage_named_drop_max_vs_phase_7_5": (
                PHASE_7_6_SUBPAGE_NAMED_THRESHOLD
            ),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def render_section_rerank_grid_md(grid: Sequence[SectionRerankSpec]) -> str:
    lines: List[str] = []
    lines.append("# Phase 7.6 — section-aware rerank candidate grid")
    lines.append("")
    lines.append(
        "> Scaffolding artefact. Phase 7.6 lands an eval harness; "
        "production promotion is a separate Phase 7.6.x PR if a "
        "winner survives the guardrails. The Phase 7.5 production-"
        "recommended config is the **new baseline** for this sweep."
    )
    lines.append("")
    lines.append("## Grid")
    lines.append("")
    lines.append(
        "| name | strategy | deployable | base candidate_k | "
        "base λ | base top_k | section_bonus | inner_top_k | "
        "description |"
    )
    lines.append(
        "|---|---|:---:|---:|---:|---:|---:|---:|---|"
    )
    for g in grid:
        lines.append(
            f"| `{g.name}` | `{g.strategy}` | "
            f"{'✓' if g.deployable else 'diagnostic'} | "
            f"{g.base_candidate_k} | {g.base_mmr_lambda:.2f} | "
            f"{g.base_top_k} | {g.section_bonus:.2f} | "
            f"{g.page_first_inner_top_k} | {g.description} |"
        )
    lines.append("")
    lines.append("## Guardrails (vs Phase 7.5 production-recommended)")
    lines.append("")
    lines.append(
        f"- gold primary_score may not drop more than "
        f"{PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD * 100:.0f}pp"
    )
    lines.append(
        f"- silver hit@5 may not drop more than "
        f"{PHASE_7_6_SILVER_REGRESSION_THRESHOLD * 100:.0f}pp"
    )
    lines.append(
        f"- subpage_named weighted_hit@5 may not drop more than "
        f"{PHASE_7_6_SUBPAGE_NAMED_THRESHOLD * 100:.0f}pp"
    )
    lines.append(
        "- page-level `weighted_hit@5` may not drop below the original "
        "Phase 7.4 baseline."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Strategies marked DIAGNOSTIC ONLY require oracle access "
        "(expected_section_path or supporting_chunk_id) at inference "
        "time. They cannot be promoted to production; their job is "
        "to set an upper bound on the section_hit@5 metric."
    )
    lines.append(
        "- Production-deployable strategies use only the query and "
        "the candidate metadata."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_section_rerank_grid_md(
    path: Path, grid: Sequence[SectionRerankSpec],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_section_rerank_grid_md(grid)
    path.write_text(md, encoding="utf-8")
    return path


__all__ = [
    "STRATEGY_BASELINE_NO_RERANK",
    "STRATEGY_SECTION_BONUS",
    "STRATEGY_SUPPORTING_CHUNK_PROXIMITY",
    "STRATEGY_PAGE_FIRST_SECTION_RERANK",
    "STRATEGY_SAME_PAGE_CHUNK_RERANK",
    "STRATEGY_ANSWERABILITY_AUDIT_PREP",
    "DEFAULT_SECTION_BONUS_VALUES",
    "PHASE_7_5_RECOMMENDED_CANDIDATE_K",
    "PHASE_7_5_RECOMMENDED_LAMBDA",
    "PHASE_7_5_RECOMMENDED_TOP_K",
    "PHASE_7_6_PRIMARY_REGRESSION_THRESHOLD",
    "PHASE_7_6_SILVER_REGRESSION_THRESHOLD",
    "PHASE_7_6_SUBPAGE_NAMED_THRESHOLD",
    "SectionRerankSpec",
    "SectionRerankInput",
    "make_section_rerank_grid",
    "apply_section_bonus_post_hoc",
    "apply_same_page_chunk_rerank",
    "apply_page_first_section_rerank",
    "run_variant_for_query",
    "page_hit_at_k",
    "section_hit_at_k",
    "render_section_rerank_grid_md",
    "write_section_rerank_grid_md",
    "write_section_rerank_grid_json",
]
