"""Verdict logic for the reranker-input-format confirm sweep.

The matrix is (index_variant) × (reranker_format) for one cell
(``optuna_winner_top8``). Anchor: ``raw`` index × ``chunk_only``
format — that's the production-equivalent baseline.

This module computes:

  - per-(variant, format) deltas vs the anchor — same shape as
    ``variant_comparison.VariantDeltas`` so the existing report
    writers can render them — via
    ``compute_format_deltas`` (a thin pass-through to
    ``compute_variant_deltas`` with the variant key replaced by
    a ``(variant, format)`` composite).
  - the spec's five-way verdict
    (``ADOPT_RERANKER_TITLE_PREFIX``,
    ``ADOPT_RERANKER_TITLE_SECTION_PREFIX``,
    ``ADOPT_COMPACT_METADATA_PREFIX``,
    ``KEEP_CHUNK_ONLY_RERANKER_INPUT``,
    ``NEED_CHUNKING_DIVERSITY_EXPERIMENT``) via
    ``decide_reranker_format_verdict``.

The five-way verdict is the headline output: it tells the next sweep
which axis to attack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from eval.harness.confirm_wide_mmr_helpers import EPS_HIT, EPS_MRR
from eval.harness.variant_comparison import (
    EPS_CANDIDATE,
    LATENCY_RATIO_LIMIT,
    VariantDeltas,
)


VERDICT_ADOPT_TITLE_PREFIX = "ADOPT_RERANKER_TITLE_PREFIX"
VERDICT_ADOPT_TITLE_SECTION_PREFIX = "ADOPT_RERANKER_TITLE_SECTION_PREFIX"
VERDICT_ADOPT_COMPACT_METADATA_PREFIX = "ADOPT_COMPACT_METADATA_PREFIX"
VERDICT_KEEP_CHUNK_ONLY = "KEEP_CHUNK_ONLY_RERANKER_INPUT"
VERDICT_NEED_CHUNKING_DIVERSITY = "NEED_CHUNKING_DIVERSITY_EXPERIMENT"


# Anchor (variant, format) — the production-equivalent recipe.
ANCHOR_VARIANT = "raw"
ANCHOR_FORMAT = "chunk_only"

CHUNK_ONLY_FORMAT = "chunk_only"
TITLE_PREFIX_FORMAT = "title_plus_chunk"
TITLE_SECTION_PREFIX_FORMAT = "title_section_plus_chunk"
COMPACT_PREFIX_FORMAT = "compact_metadata_plus_chunk"


# Diversity-bottleneck signal threshold — when title_section index
# already lifts cand@50 by ≥ this much vs the raw anchor but no prefix
# format clears EPS on hit@5, the next axis is chunking diversity, not
# more reranker prefixes.
DIVERSITY_DUP_LIFT_THRESHOLD = 0.01


def _has_quality_lift(deltas: Optional[VariantDeltas]) -> bool:
    """Hit@5 OR MRR@10 lifted by ≥ EPS without a hit@5 regression."""
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    h5_safe = (h5 is None) or (h5 >= -EPS_HIT)
    has_lift = (
        (h5 is not None and h5 >= EPS_HIT)
        or (mrr is not None and mrr >= EPS_MRR)
    )
    return has_lift and h5_safe


def _has_regression(deltas: Optional[VariantDeltas]) -> bool:
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    cand50 = deltas.delta_candidate_hit_at_50
    return (
        (h5 is not None and h5 <= -EPS_HIT)
        or (mrr is not None and mrr <= -EPS_MRR)
        or (cand50 is not None and cand50 <= -EPS_CANDIDATE)
    )


def _latency_within_budget(deltas: Optional[VariantDeltas]) -> bool:
    if deltas is None:
        return True
    ratio = deltas.latency_ratio_p95
    return ratio is None or ratio <= LATENCY_RATIO_LIMIT


def _quality_score(deltas: Optional[VariantDeltas]) -> float:
    """Composite quality lift used to break ties between adopt-eligible
    formats. Sum of Δhit@5 + Δmrr@10 — both metrics matter equally for
    the headline verdict; cand@K is excluded because it's an upstream
    signal that all formats share when the index variant is fixed.
    """
    if deltas is None:
        return float("-inf")
    h5 = deltas.delta_hit_at_5 or 0.0
    mrr = deltas.delta_mrr_at_10 or 0.0
    return float(h5) + float(mrr)


def _has_dense_pool_lift(
    variant_anchor: Optional[VariantDeltas],
) -> bool:
    """True when an index variant lifts cand@50 OR cand@100 ≥ EPS over
    raw, regardless of the reranker format. Used to detect the
    chunking-diversity verdict (E): dense pool moved upstream but no
    reranker format unlocks the gain.
    """
    if variant_anchor is None:
        return False
    cand50 = variant_anchor.delta_candidate_hit_at_50
    cand100 = variant_anchor.delta_candidate_hit_at_100
    return (
        (cand50 is not None and cand50 >= EPS_CANDIDATE)
        or (cand100 is not None and cand100 >= EPS_CANDIDATE)
    )


def _has_dup_persistence(
    deltas: Optional[VariantDeltas],
) -> bool:
    """True when dup@10 stays elevated (≥ DIVERSITY_DUP_LIFT_THRESHOLD)
    vs the raw chunk_only anchor. Marker for "more dense candidates
    aren't translating into more *diverse* finals" — the chunking-
    diversity bottleneck signal.
    """
    if deltas is None:
        return False
    dup = deltas.delta_duplicate_ratio_at_10
    return dup is not None and dup >= DIVERSITY_DUP_LIFT_THRESHOLD


def decide_reranker_format_verdict(
    *,
    deltas_by_pair: Dict[Tuple[str, str], VariantDeltas],
) -> Tuple[str, str]:
    """Return ``(verdict, rationale)`` over (variant, format) deltas.

    The five-way verdict mirrors the spec's A/B/C/D/E enumeration. The
    decision is anchored on the *quality leader* among eligible
    formats, not on a strict format-priority order — when both
    title and title_section clear EPS, the spec says "가장 좋으면
    선택" (pick the best), not "title only when title_section regresses".

    Definitions:

      - ``best_by_fmt[fmt]`` is the (variant, fmt) pair with the
        highest composite quality score (``Δhit@5 + Δmrr@10``) on
        that format, preferring a non-regressing + within-latency
        candidate when one exists. Picking the best variant *per
        format* lets the verdict ask "is this format ever a good
        idea on any index?", which is the right question.
      - A format is *eligible* when the best pair for it shows a
        ≥ EPS lift on hit@5 OR MRR@10, no hit@5/MRR/cand@50
        regression, and stays within ``LATENCY_RATIO_LIMIT``.
      - The *leader* is the eligible format with the highest
        composite quality score. Ties go to the spec's preferred
        order (title_section > title > compact) — that order
        captures the spec's preference for richer metadata when
        all else is equal.

    Decision tree (first match wins):

      1. Eligible leader is ``title_section_plus_chunk`` →
         ``ADOPT_RERANKER_TITLE_SECTION_PREFIX`` (case B).
      2. Eligible leader is ``title_plus_chunk`` →
         ``ADOPT_RERANKER_TITLE_PREFIX`` (case A).
      3. Eligible leader is ``compact_metadata_plus_chunk`` →
         ``ADOPT_COMPACT_METADATA_PREFIX`` (case C).
      4. No eligible format AND a non-anchor index variant lifts
         cand@50/cand@100 by ≥ EPS AND dup@10 stays elevated on
         the title_section index → ``NEED_CHUNKING_DIVERSITY_
         EXPERIMENT`` (case E).
      5. Otherwise → ``KEEP_CHUNK_ONLY_RERANKER_INPUT`` (case D).
    """
    formats = (
        TITLE_PREFIX_FORMAT,
        TITLE_SECTION_PREFIX_FORMAT,
        COMPACT_PREFIX_FORMAT,
    )

    best_by_fmt: Dict[str, Optional[VariantDeltas]] = {}
    for fmt in formats:
        candidates = [
            d for (v, f), d in deltas_by_pair.items() if f == fmt
        ]
        if not candidates:
            best_by_fmt[fmt] = None
            continue
        passing = [
            d for d in candidates
            if not _has_regression(d) and _latency_within_budget(d)
        ]
        pool = passing if passing else candidates
        best_by_fmt[fmt] = max(pool, key=_quality_score)

    title_d = best_by_fmt.get(TITLE_PREFIX_FORMAT)
    ts_d = best_by_fmt.get(TITLE_SECTION_PREFIX_FORMAT)
    compact_d = best_by_fmt.get(COMPACT_PREFIX_FORMAT)

    title_qs = _quality_score(title_d)
    ts_qs = _quality_score(ts_d)
    compact_qs = _quality_score(compact_d)

    eligibility: Dict[str, bool] = {
        TITLE_PREFIX_FORMAT: bool(
            _has_quality_lift(title_d)
            and not _has_regression(title_d)
            and _latency_within_budget(title_d)
        ),
        TITLE_SECTION_PREFIX_FORMAT: bool(
            _has_quality_lift(ts_d)
            and not _has_regression(ts_d)
            and _latency_within_budget(ts_d)
        ),
        COMPACT_PREFIX_FORMAT: bool(
            _has_quality_lift(compact_d)
            and not _has_regression(compact_d)
            and _latency_within_budget(compact_d)
        ),
    }
    qs_by_fmt: Dict[str, float] = {
        TITLE_PREFIX_FORMAT: title_qs,
        TITLE_SECTION_PREFIX_FORMAT: ts_qs,
        COMPACT_PREFIX_FORMAT: compact_qs,
    }

    eligible_formats = [f for f, ok in eligibility.items() if ok]
    if eligible_formats:
        # Tie-break order: title_section (richest) > title > compact.
        # On a strict tie this captures the spec's preference for the
        # most metadata when quality scores are indistinguishable.
        priority = {
            TITLE_SECTION_PREFIX_FORMAT: 0,
            TITLE_PREFIX_FORMAT: 1,
            COMPACT_PREFIX_FORMAT: 2,
        }
        leader = max(
            eligible_formats,
            key=lambda f: (qs_by_fmt[f], -priority[f]),
        )
        if leader == TITLE_SECTION_PREFIX_FORMAT:
            return (
                VERDICT_ADOPT_TITLE_SECTION_PREFIX,
                (
                    f"title_section_plus_chunk leads on composite "
                    f"quality ({ts_qs:+.4f}) among eligible formats "
                    f"(title_only {title_qs:+.4f}, compact "
                    f"{compact_qs:+.4f}) and stays within the "
                    f"non-regression / latency budget. Adopt the "
                    f"title+section prefix — the richer metadata "
                    f"surface translated into the cleanest reranker "
                    f"uplift on this dataset."
                ),
            )
        if leader == TITLE_PREFIX_FORMAT:
            return (
                VERDICT_ADOPT_TITLE_PREFIX,
                (
                    f"title_plus_chunk leads on composite quality "
                    f"({title_qs:+.4f}) among eligible formats "
                    f"(title_section {ts_qs:+.4f}, compact "
                    f"{compact_qs:+.4f}) and stays within the "
                    f"non-regression / latency budget. Adopt the "
                    f"title-only prefix — the shorter prefix preserves "
                    f"the cross-encoder's text_max_chars budget while "
                    f"still surfacing the title disambiguation signal."
                ),
            )
        if leader == COMPACT_PREFIX_FORMAT:
            return (
                VERDICT_ADOPT_COMPACT_METADATA_PREFIX,
                (
                    f"compact_metadata_plus_chunk leads on composite "
                    f"quality ({compact_qs:+.4f}) among eligible "
                    f"formats (title_only {title_qs:+.4f}, title_section "
                    f"{ts_qs:+.4f}). The compact ``[title / section]`` "
                    f"head beats the longer Korean-prefixed forms — "
                    f"adopt the compact prefix to keep the "
                    f"text_max_chars budget for the actual chunk body."
                ),
            )

    # No eligible format. Check the chunking-diversity signal.
    chunk_only_anchor_lifts = [
        d for (v, f), d in deltas_by_pair.items()
        if f == CHUNK_ONLY_FORMAT and v != ANCHOR_VARIANT
    ]
    dense_pool_moved = any(
        _has_dense_pool_lift(d) for d in chunk_only_anchor_lifts
    )
    dup_persistent = any(
        _has_dup_persistence(d)
        for (v, f), d in deltas_by_pair.items()
        if f != CHUNK_ONLY_FORMAT
    )

    if dense_pool_moved and dup_persistent:
        return (
            VERDICT_NEED_CHUNKING_DIVERSITY,
            (
                "Dense candidate pool lifts cand@50/cand@100 by ≥ EPS "
                "with the title/title_section indexes, but no reranker "
                "input format clears EPS on hit@5/MRR over the raw "
                "chunk_only anchor AND dup@10 stays elevated. The "
                "reranker is fed enough gold but cannot disambiguate "
                "near-duplicate chunks regardless of prefix — the next "
                "axis is chunking diversity, not more reranker formats."
            ),
        )

    return (
        VERDICT_KEEP_CHUNK_ONLY,
        (
            f"No reranker input format clears EPS on hit@5/MRR over "
            f"the raw chunk_only anchor (best lifts: title "
            f"{title_qs:+.4f}, title_section {ts_qs:+.4f}, compact "
            f"{compact_qs:+.4f}). The reranker passage formatting "
            f"isn't the bottleneck on this dataset; keep chunk_only."
        ),
    )
