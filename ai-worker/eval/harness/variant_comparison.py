"""Eval-only — variant-axis comparison helpers.

Sister to ``confirm_wide_mmr_helpers``. The wide-MMR confirm sweep
asked which (cap × λ × top_k × rerank_in) recipe wins on silver_200;
this module asks which *embedding-text variant* wins for a fixed
recipe. The independent variable here is the bi-encoder's input
string (``raw`` / ``title`` / ``title_section``); everything
downstream (MMR / cap / reranker) is held constant.

Layered to mirror ``confirm_wide_mmr_helpers`` for readability:

  - ``VariantDeltas`` — per-(variant, cell) deltas vs the same cell on
    the ``raw`` anchor variant. Mirrors ``CellDeltas`` but the baseline
    is a same-cell different-variant pair instead of a same-variant
    different-cell pair. Pure data; tests pin the field set.
  - ``compute_variant_deltas`` — populate ``VariantDeltas``. Reuses the
    ``_delta_or_none`` / ``_ratio_or_none`` / ``_read_metric``
    primitives from the wide-MMR helpers so the metric extraction
    contract is identical between the two comparison axes.
  - ``decide_variant_verdict`` — A / B / C / D verdict over
    ``ADOPT_TITLE_SECTION_INDEX``, ``ADOPT_TITLE_INDEX``,
    ``KEEP_RAW_INDEX``, ``NEED_RERANKER_INPUT_AUDIT_FIRST``. Pure
    function over the per-cell deltas; testable without retrieval.
  - ``RerankerAuditSample`` + ``collect_reranker_audit_samples`` —
    surface the queries where the gold doc is in the candidate pool
    but the reranker dropped it from the top-k, with the candidate-
    pool passage previews / title-included flag / truncation flag
    that the spec asks for.

Verdict logic:

  A. ``ADOPT_TITLE_SECTION_INDEX`` —
     ``title_section`` improves hit@5 (or MRR@10) by ≥ EPS over raw,
     does NOT regress hit@5 by more than EPS, and latency / dup ratio
     stay within bounds.

  B. ``ADOPT_TITLE_INDEX`` —
     ``title`` improves hit@5 / MRR@10 over raw by ≥ EPS, AND
     ``title_section`` either fails the same check or regresses on
     hit@5. Title is the safer pick when title_section adds noise.

  C. ``KEEP_RAW_INDEX`` —
     Neither ``title`` nor ``title_section`` clears EPS on hit@5
     OR MRR@10, OR both regress on hit@5 / dup / latency.

  D. ``NEED_RERANKER_INPUT_AUDIT_FIRST`` —
     Either variant lifts ``cand@K`` (hit@K of the candidate pool)
     by ≥ EPS but the *final* hit@5 / MRR@10 is inside epsilon. The
     dense pool is finding more gold candidates but the reranker is
     dropping them on the floor — the next bottleneck is reranker
     input formatting, not the embedding text variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.harness.confirm_wide_mmr_helpers import (
    EPS_HIT,
    EPS_MRR,
    GRADE_BASELINE,
    GRADE_DIAG_ONLY,
    GRADE_INCONCLUSIVE,
    GRADE_PROMISING,
    GRADE_REGRESSION,
    LATENCY_RATIO_LIMIT,
)


# Verdict labels — exact strings the report writer + tests pin.
VERDICT_ADOPT_TITLE_SECTION = "ADOPT_TITLE_SECTION_INDEX"
VERDICT_ADOPT_TITLE = "ADOPT_TITLE_INDEX"
VERDICT_KEEP_RAW = "KEEP_RAW_INDEX"
VERDICT_NEED_RERANKER_AUDIT = "NEED_RERANKER_INPUT_AUDIT_FIRST"

# How big a candidate-pool lift counts as "the dense pool moved" for
# the reranker-audit verdict trigger. Same epsilon as hit@5 — these
# are 200-row metrics, single-query swings are 0.005.
EPS_CANDIDATE = 0.005

# Anchor variant — the regression baseline. ``raw`` matches production.
ANCHOR_VARIANT = "raw"

# The two adoption candidates the spec calls out. Order matters: the
# verdict logic checks ``title_section`` first (it's the more
# aggressive variant; if it wins, no need to fall back to title).
TITLE_SECTION_VARIANT = "title_section"
TITLE_VARIANT = "title"


# ---------------------------------------------------------------------------
# Metric-read helpers (kept local so this module doesn't reach into the
# wide-MMR helpers' private functions).
# ---------------------------------------------------------------------------


def _read_metric(
    summary: Mapping[str, Any] | Any, key: str,
) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Deltas dataclass + computation
# ---------------------------------------------------------------------------


@dataclass
class VariantDeltas:
    """All deltas a (variant, cell) carries vs the raw anchor.

    Tuned for the report's needs: header metrics, candidate / dup /
    latency aux. ``grade`` mirrors the wide-MMR grades so the markdown
    can be read alongside the prior sweeps. The ``cand@K`` deltas live
    here too because the reranker-audit verdict turns on whether dense
    candidate hit-rate moved.
    """

    cell_label: str
    variant: str
    grade: str
    reason: str
    delta_hit_at_1: Optional[float]
    delta_hit_at_3: Optional[float]
    delta_hit_at_5: Optional[float]
    delta_mrr_at_10: Optional[float]
    delta_ndcg_at_10: Optional[float]
    delta_candidate_hit_at_10: Optional[float]
    delta_candidate_hit_at_20: Optional[float]
    delta_candidate_hit_at_50: Optional[float]
    delta_candidate_hit_at_100: Optional[float]
    delta_duplicate_ratio_at_5: Optional[float]
    delta_duplicate_ratio_at_10: Optional[float]
    delta_unique_doc_count_at_10: Optional[float]
    delta_p50ms: Optional[float]
    delta_p95ms: Optional[float]
    delta_p99ms: Optional[float]
    latency_ratio_p95: Optional[float]


def compute_variant_deltas(
    *,
    cell_label: str,
    variant: str,
    variant_summary: Any,
    raw_summary: Any,
) -> VariantDeltas:
    """Build a ``VariantDeltas`` for ``variant`` against the raw anchor.

    The raw variant always self-grades to ``GRADE_BASELINE`` with zero
    deltas — its presence in the report is an anchor, not a comparison.

    Grading mirrors ``confirm_wide_mmr_helpers.compute_cell_deltas`` so
    the markdown reads consistently across reports:

      - ``GRADE_REGRESSION``: hit@5 OR MRR@10 OR cand@50 dropped by
        ≥ EPS (EPS_HIT = EPS_MRR = 0.005). Don't adopt.
      - ``GRADE_PROMISING``: hit@5 OR MRR@10 lifted by ≥ EPS *and*
        latency ratio within ``LATENCY_RATIO_LIMIT`` (1.5x).
      - ``GRADE_DIAG_ONLY``: lift exists but exceeds latency budget,
        OR mixed signals (cand@K up but final flat).
      - ``GRADE_INCONCLUSIVE``: every delta inside epsilon.
    """
    if variant == ANCHOR_VARIANT or variant_summary is raw_summary:
        return VariantDeltas(
            cell_label=cell_label,
            variant=variant,
            grade=GRADE_BASELINE,
            reason="raw anchor",
            delta_hit_at_1=0.0,
            delta_hit_at_3=0.0,
            delta_hit_at_5=0.0,
            delta_mrr_at_10=0.0,
            delta_ndcg_at_10=0.0,
            delta_candidate_hit_at_10=0.0,
            delta_candidate_hit_at_20=0.0,
            delta_candidate_hit_at_50=0.0,
            delta_candidate_hit_at_100=0.0,
            delta_duplicate_ratio_at_5=0.0,
            delta_duplicate_ratio_at_10=0.0,
            delta_unique_doc_count_at_10=0.0,
            delta_p50ms=0.0,
            delta_p95ms=0.0,
            delta_p99ms=0.0,
            latency_ratio_p95=1.0,
        )

    d_h1 = _delta_or_none(
        _read_metric(variant_summary, "mean_hit_at_1"),
        _read_metric(raw_summary, "mean_hit_at_1"),
    )
    d_h3 = _delta_or_none(
        _read_metric(variant_summary, "mean_hit_at_3"),
        _read_metric(raw_summary, "mean_hit_at_3"),
    )
    d_h5 = _delta_or_none(
        _read_metric(variant_summary, "mean_hit_at_5"),
        _read_metric(raw_summary, "mean_hit_at_5"),
    )
    d_mrr = _delta_or_none(
        _read_metric(variant_summary, "mean_mrr_at_10"),
        _read_metric(raw_summary, "mean_mrr_at_10"),
    )
    d_ndcg = _delta_or_none(
        _read_metric(variant_summary, "mean_ndcg_at_10"),
        _read_metric(raw_summary, "mean_ndcg_at_10"),
    )

    def _dcand(k: str) -> Optional[float]:
        return _delta_or_none(
            _read_dict_metric(variant_summary, "candidate_hit_rates", k),
            _read_dict_metric(raw_summary, "candidate_hit_rates", k),
        )

    d_cand10 = _dcand("10")
    d_cand20 = _dcand("20")
    d_cand50 = _dcand("50")
    d_cand100 = _dcand("100")

    d_dup5 = _delta_or_none(
        _read_dict_metric(variant_summary, "duplicate_doc_ratios", "5"),
        _read_dict_metric(raw_summary, "duplicate_doc_ratios", "5"),
    )
    d_dup10 = _delta_or_none(
        _read_dict_metric(variant_summary, "duplicate_doc_ratios", "10"),
        _read_dict_metric(raw_summary, "duplicate_doc_ratios", "10"),
    )
    d_uniq10 = _delta_or_none(
        _read_dict_metric(variant_summary, "unique_doc_counts", "10"),
        _read_dict_metric(raw_summary, "unique_doc_counts", "10"),
    )

    p50_curr = _read_metric(variant_summary, "p50_retrieval_ms")
    p50_base = _read_metric(raw_summary, "p50_retrieval_ms")
    p95_curr = (
        _read_metric(variant_summary, "p95_total_retrieval_ms")
        or _read_metric(variant_summary, "p95_retrieval_ms")
    )
    p95_base = (
        _read_metric(raw_summary, "p95_total_retrieval_ms")
        or _read_metric(raw_summary, "p95_retrieval_ms")
    )
    p99_curr = _read_metric(variant_summary, "p99_retrieval_ms")
    p99_base = _read_metric(raw_summary, "p99_retrieval_ms")
    d_p50 = _delta_or_none(p50_curr, p50_base)
    d_p95 = _delta_or_none(p95_curr, p95_base)
    d_p99 = _delta_or_none(p99_curr, p99_base)
    latency_ratio = _ratio_or_none(p95_curr, p95_base)

    if (
        (d_h5 is not None and d_h5 <= -EPS_HIT)
        or (d_mrr is not None and d_mrr <= -EPS_MRR)
        or (d_cand50 is not None and d_cand50 <= -EPS_CANDIDATE)
    ):
        grade = GRADE_REGRESSION
        reason = (
            f"Δhit@5={d_h5} Δmrr={d_mrr} Δcand@50={d_cand50}"
        )
    elif (
        (d_h5 is not None and d_h5 >= EPS_HIT)
        or (d_mrr is not None and d_mrr >= EPS_MRR)
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
                f"Δhit@5={d_h5} Δmrr={d_mrr} latency_ratio={latency_ratio}"
            )
    elif (
        (d_h5 is None or abs(d_h5) < EPS_HIT)
        and (d_mrr is None or abs(d_mrr) < EPS_MRR)
        and (d_cand50 is not None and d_cand50 >= EPS_CANDIDATE)
    ):
        # Mixed signal: dense pool moved but final ranking didn't.
        grade = GRADE_DIAG_ONLY
        reason = (
            f"cand@50 up by {d_cand50} but final hit@5 / MRR flat — "
            "reranker is not propagating the dense pool gain."
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

    return VariantDeltas(
        cell_label=cell_label,
        variant=variant,
        grade=grade,
        reason=reason,
        delta_hit_at_1=d_h1,
        delta_hit_at_3=d_h3,
        delta_hit_at_5=d_h5,
        delta_mrr_at_10=d_mrr,
        delta_ndcg_at_10=d_ndcg,
        delta_candidate_hit_at_10=d_cand10,
        delta_candidate_hit_at_20=d_cand20,
        delta_candidate_hit_at_50=d_cand50,
        delta_candidate_hit_at_100=d_cand100,
        delta_duplicate_ratio_at_5=d_dup5,
        delta_duplicate_ratio_at_10=d_dup10,
        delta_unique_doc_count_at_10=d_uniq10,
        delta_p50ms=d_p50,
        delta_p95ms=d_p95,
        delta_p99ms=d_p99,
        latency_ratio_p95=latency_ratio,
    )


# ---------------------------------------------------------------------------
# Verdict logic (A / B / C / D)
# ---------------------------------------------------------------------------


def _has_promising_quality(deltas: Optional[VariantDeltas]) -> bool:
    """Quality lift ≥ EPS on hit@5 OR MRR@10 *and* no hit@5 regression."""
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    h5_ok = (h5 is None) or (h5 >= -EPS_HIT)
    has_lift = (
        (h5 is not None and h5 >= EPS_HIT)
        or (mrr is not None and mrr >= EPS_MRR)
    )
    return has_lift and h5_ok


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


def _has_dense_pool_lift_only(deltas: Optional[VariantDeltas]) -> bool:
    """``cand@K`` up by ≥ EPS but final hit@5 / MRR not improving.

    "Not improving" intentionally includes both *flat* and *regressed*
    final metrics. The spec's D verdict ("cand@K는 개선되는데 final
    hit/MRR이 개선되지 않으면 reranker input formatting이 더 큰 병목")
    treats a final regression as the strongest possible "not improving"
    signal — the dense pool surfaced more gold candidates but the
    cross-encoder reordering made things worse, which is exactly the
    diagnostic the reranker-audit verdict is meant to surface. An
    earlier draft of this helper required ``final_flat`` strictly
    (|Δh5| < EPS); that left the silver_200 result (cand@50 +0.045,
    final hit@5 -0.045) misclassified as plain ``KEEP_RAW``, hiding
    the reranker bottleneck the data made obvious.
    """
    if deltas is None:
        return False
    h5 = deltas.delta_hit_at_5
    mrr = deltas.delta_mrr_at_10
    cand_lifts = [
        v for v in (
            deltas.delta_candidate_hit_at_50,
            deltas.delta_candidate_hit_at_100,
        ) if v is not None
    ]
    if not cand_lifts:
        return False
    cand_up = any(v >= EPS_CANDIDATE for v in cand_lifts)
    # final_not_improving: hit@5 AND MRR both fail to clear +EPS.
    # Flat (|Δ| < EPS) and regressed (Δ ≤ -EPS) both qualify.
    h5_not_improving = (h5 is None) or (h5 < EPS_HIT)
    mrr_not_improving = (mrr is None) or (mrr < EPS_MRR)
    final_not_improving = h5_not_improving and mrr_not_improving
    return cand_up and final_not_improving


def _latency_within_budget(deltas: Optional[VariantDeltas]) -> bool:
    if deltas is None:
        return False
    ratio = deltas.latency_ratio_p95
    return ratio is None or ratio <= LATENCY_RATIO_LIMIT


def decide_variant_verdict(
    *,
    title_deltas: Optional[VariantDeltas],
    title_section_deltas: Optional[VariantDeltas],
) -> Tuple[str, str]:
    """Return ``(verdict, rationale)`` over the variant deltas.

    Decision tree (in order; first match wins):

      1. ``title_section`` is promising AND not regressing AND latency
         within budget → ``ADOPT_TITLE_SECTION_INDEX``.
      2. ``title`` is promising AND ``title_section`` is regressing OR
         not promising AND latency within budget for title →
         ``ADOPT_TITLE_INDEX``.
      3. Either variant has ``cand@K`` lift but final hit@5 *not
         improving* (flat OR regressed) →
         ``NEED_RERANKER_INPUT_AUDIT_FIRST``. Spec D's wording ("cand@K
         는 개선되는데 final hit/MRR이 개선되지 않으면") covers both
         flat and regressed finals — when dense pulls in more gold
         and the reranker still can't surface it, the bottleneck is
         the reranker, regardless of whether the final delta is zero
         or negative.
      4. Neither variant moves quality OR both regress without a
         compensating cand@K lift → ``KEEP_RAW_INDEX``.
    """
    ts_promising = _has_promising_quality(title_section_deltas)
    ts_regression = _has_regression(title_section_deltas)
    ts_latency_ok = _latency_within_budget(title_section_deltas)
    t_promising = _has_promising_quality(title_deltas)
    t_regression = _has_regression(title_deltas)
    t_latency_ok = _latency_within_budget(title_deltas)
    ts_pool_only = _has_dense_pool_lift_only(title_section_deltas)
    t_pool_only = _has_dense_pool_lift_only(title_deltas)

    # Case A — title_section is the cleanest win.
    if ts_promising and not ts_regression and ts_latency_ok:
        return (
            VERDICT_ADOPT_TITLE_SECTION,
            (
                "title_section lifts hit@5 / MRR@10 by ≥ EPS over raw "
                "without regressing the cap/latency contract; safe to "
                "adopt as the new dense-index reindex target."
            ),
        )

    # Case B — title alone wins; title_section adds noise.
    if (
        t_promising
        and not t_regression
        and t_latency_ok
        and (not ts_promising or ts_regression)
    ):
        return (
            VERDICT_ADOPT_TITLE,
            (
                "title-only prefix lifts quality without regression, "
                "while title_section either regresses or doesn't clear "
                "EPS — title is the safer adopt target."
            ),
        )

    # Case D — dense pool moved but rerank didn't propagate the gain.
    # ``final not improving`` covers both ``flat`` (|Δhit@5| < EPS) and
    # ``regressed`` (Δhit@5 ≤ -EPS); when the dense pool surfaces more
    # gold candidates and the reranker drops them anyway, the bottleneck
    # is the reranker regardless of the final delta's sign.
    if ts_pool_only or t_pool_only:
        return (
            VERDICT_NEED_RERANKER_AUDIT,
            (
                "Candidate-pool hit-rate (cand@50/cand@100) lifts by "
                "≥ EPS but final hit@5 / MRR@10 fail to improve — the "
                "dense pool is finding more gold, but the cross-encoder "
                "is dropping it (or worse, demoting it below the raw "
                "result). Audit reranker input formatting "
                "(title/section in passage, max_chars truncation) "
                "before another embedding-text reindex."
            ),
        )

    # Case C — fallback. Neither variant moved quality measurably.
    return (
        VERDICT_KEEP_RAW,
        (
            "Neither title nor title_section variants clear EPS on "
            "hit@5 / MRR@10 over raw — the embedding-text axis isn't "
            "the bottleneck on this dataset. Keep the raw index."
        ),
    )


# ---------------------------------------------------------------------------
# Reranker input audit — surface gold-in-pool-but-rerank-dropped queries
# ---------------------------------------------------------------------------


@dataclass
class RerankerAuditSample:
    """One ``gold-in-pool-but-rerank-dropped`` audit row.

    Captures everything the spec's report section needs: the query,
    the gold doc id list, the top-K retrieved doc list (post rerank),
    one or two candidate-pool passage previews, and two boolean flags
    indicating whether the gold passage carried a title/section in
    its preview text and whether the preview hit the truncation cap.
    """

    cell_label: str
    variant: str
    query_id: str
    query: str
    expected_doc_ids: List[str]
    candidate_doc_ids: List[str]
    retrieved_top_doc_ids: List[str]
    candidate_count: int
    gold_in_candidates: bool
    gold_passage_preview: Optional[str] = None
    gold_passage_has_title: bool = False
    gold_passage_truncated: bool = False
    rerank_top_passage_preview: Optional[str] = None


def _row_get(row: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _passage_carries_title(text: str, title: Optional[str]) -> bool:
    """Heuristic: does ``text`` start with the chunk's doc title?

    Returns False when ``title`` is None or empty (no signal). The
    audit uses this to surface whether a variant's prefix actually
    made it into the reranker's view of the passage — which is the
    debugging question the spec calls out (``title/section 포함 여부``).
    """
    if not title:
        return False
    needle = str(title).strip().casefold()
    if not needle:
        return False
    haystack = (text or "")[: max(len(needle) * 4, 200)].casefold()
    return needle in haystack


def collect_reranker_audit_samples(
    *,
    cell_label: str,
    variant: str,
    rows: Sequence[Mapping[str, Any] | Any],
    chunk_text_lookup: Optional[Dict[str, str]] = None,
    chunk_title_lookup: Optional[Dict[str, Optional[str]]] = None,
    truncation_threshold_chars: int = 800,
    limit: int = 5,
) -> List[RerankerAuditSample]:
    """Return up to ``limit`` queries with gold-in-pool-but-rerank-dropped.

    A row qualifies when:
      - it has expected_doc_ids, AND
      - at least one expected doc_id is in ``candidate_doc_ids``, AND
      - none of the expected doc_ids appear in the final top-5 of
        ``retrieved_doc_ids`` (the rerank dropped it).

    The audit fills ``gold_passage_preview`` from
    ``chunk_text_lookup[doc_id]`` when supplied (typically the corpus
    title text); ``gold_passage_truncated`` fires when the preview
    matches the reranker's text_max_chars cap. ``gold_passage_has_title``
    asks whether the variant's prefix actually surfaces in the passage
    the reranker would have scored, which the spec calls out as the
    central debugging question.
    """
    out: List[RerankerAuditSample] = []
    for row in rows:
        if len(out) >= limit:
            break
        expected_raw = _row_get(row, "expected_doc_ids") or []
        expected = [str(d) for d in expected_raw if d]
        if not expected:
            continue
        retrieved_raw = _row_get(row, "retrieved_doc_ids") or []
        retrieved = [str(d) for d in retrieved_raw if d]
        candidates_raw = _row_get(row, "candidate_doc_ids") or []
        candidates = [str(d) for d in candidates_raw if d]
        cand_set = set(candidates)
        retrieved_top5 = set(retrieved[:5])
        gold_in_candidates = any(d in cand_set for d in expected)
        gold_dropped = (
            gold_in_candidates
            and not any(d in retrieved_top5 for d in expected)
        )
        if not gold_dropped:
            continue

        gold_doc_id = next((d for d in expected if d in cand_set), None)
        preview: Optional[str] = None
        truncated = False
        has_title = False
        if gold_doc_id and chunk_text_lookup:
            preview_full = chunk_text_lookup.get(gold_doc_id)
            if preview_full is not None:
                preview = str(preview_full)[: int(truncation_threshold_chars)]
                truncated = (
                    len(preview_full) > int(truncation_threshold_chars)
                )
                title_for_doc = (
                    None
                    if chunk_title_lookup is None
                    else chunk_title_lookup.get(gold_doc_id)
                )
                has_title = _passage_carries_title(preview_full, title_for_doc)

        rerank_preview: Optional[str] = None
        if retrieved and chunk_text_lookup:
            rerank_doc_id = retrieved[0]
            rerank_full = chunk_text_lookup.get(rerank_doc_id)
            if rerank_full is not None:
                rerank_preview = str(rerank_full)[
                    : int(truncation_threshold_chars)
                ]

        out.append(RerankerAuditSample(
            cell_label=cell_label,
            variant=variant,
            query_id=str(_row_get(row, "id") or ""),
            query=str(_row_get(row, "query") or ""),
            expected_doc_ids=expected,
            candidate_doc_ids=candidates[:20],
            retrieved_top_doc_ids=retrieved[:5],
            candidate_count=len(candidates),
            gold_in_candidates=gold_in_candidates,
            gold_passage_preview=preview,
            gold_passage_has_title=has_title,
            gold_passage_truncated=truncated,
            rerank_top_passage_preview=rerank_preview,
        ))
    return out


# ---------------------------------------------------------------------------
# Per-variant per-query diff aggregator
# ---------------------------------------------------------------------------


@dataclass
class VariantPerQueryDelta:
    """One per-query crossover entry between raw and a variant.

    Mirrors ``PerQueryDiffEntry`` from the wide-MMR helpers but keyed
    on the variant label instead of cell label, so the report writer
    can render a single table per (cell, variant_pair).
    """

    variant: str
    cell_label: str
    id: str
    query: str
    raw_hit_at_5: float
    variant_hit_at_5: float
    direction: str  # "improved" or "regressed"
    expected_doc_ids: List[str] = field(default_factory=list)
    raw_top_doc_ids: List[str] = field(default_factory=list)
    variant_top_doc_ids: List[str] = field(default_factory=list)


def variant_per_query_diff(
    *,
    cell_label: str,
    variant: str,
    raw_rows: Sequence[Mapping[str, Any] | Any],
    variant_rows: Sequence[Mapping[str, Any] | Any],
) -> Tuple[List[VariantPerQueryDelta], List[VariantPerQueryDelta]]:
    """Return ``(improved_against_raw, regressed_against_raw)``.

    Improvements: raw hit@5 = 0, variant hit@5 = 1 — the variant
    rescued a query the raw index missed.
    Regressions: raw hit@5 = 1, variant hit@5 = 0 — the variant
    broke a query the raw index had right. (Useful for spotting when
    a title prefix collides with the user query phrasing.)
    """
    base_by_id: Dict[str, Mapping[str, Any] | Any] = {}
    for row in raw_rows:
        rid = _row_get(row, "id")
        if rid:
            base_by_id[str(rid)] = row

    improved: List[VariantPerQueryDelta] = []
    regressed: List[VariantPerQueryDelta] = []
    for row in variant_rows:
        rid = str(_row_get(row, "id") or "")
        if not rid or rid not in base_by_id:
            continue
        base = base_by_id[rid]
        b_h5 = _row_get(base, "hit_at_5")
        v_h5 = _row_get(row, "hit_at_5")
        if b_h5 is None or v_h5 is None:
            continue
        try:
            b_f = float(b_h5)
            v_f = float(v_h5)
        except (TypeError, ValueError):
            continue
        expected = [str(d) for d in (_row_get(row, "expected_doc_ids") or []) if d]
        v_top = [str(d) for d in (_row_get(row, "retrieved_doc_ids") or [])][:5]
        b_top = [str(d) for d in (_row_get(base, "retrieved_doc_ids") or [])][:5]
        if v_f > 0.5 and b_f <= 0.5:
            improved.append(VariantPerQueryDelta(
                variant=variant,
                cell_label=cell_label,
                id=rid,
                query=str(_row_get(row, "query") or ""),
                raw_hit_at_5=b_f,
                variant_hit_at_5=v_f,
                direction="improved",
                expected_doc_ids=expected,
                raw_top_doc_ids=b_top,
                variant_top_doc_ids=v_top,
            ))
        elif b_f > 0.5 and v_f <= 0.5:
            regressed.append(VariantPerQueryDelta(
                variant=variant,
                cell_label=cell_label,
                id=rid,
                query=str(_row_get(row, "query") or ""),
                raw_hit_at_5=b_f,
                variant_hit_at_5=v_f,
                direction="regressed",
                expected_doc_ids=expected,
                raw_top_doc_ids=b_top,
                variant_top_doc_ids=v_top,
            ))
    return improved, regressed


def candidate_pool_recoverable_miss_count(
    rows: Sequence[Mapping[str, Any] | Any],
) -> int:
    """Number of queries with gold in candidates but not in top-5.

    Same predicate as ``confirm_wide_mmr_helpers.candidate_pool_
    recoverable_misses`` but returns the count only — the variant
    report needs the *delta* between raw and variant, not the full
    list per variant.
    """
    n = 0
    for row in rows:
        expected = [str(d) for d in (_row_get(row, "expected_doc_ids") or []) if d]
        if not expected:
            continue
        retrieved = [str(d) for d in (_row_get(row, "retrieved_doc_ids") or []) if d]
        candidates = [str(d) for d in (_row_get(row, "candidate_doc_ids") or []) if d]
        cand_set = set(candidates)
        retrieved_set = set(retrieved[:5])
        if any(d in cand_set and d not in retrieved_set for d in expected):
            n += 1
    return n


def candidate_pool_unrecoverable_miss_count(
    rows: Sequence[Mapping[str, Any] | Any],
) -> int:
    """Number of queries with gold NOT in the candidate pool at all.

    Pure dense-side miss: the bi-encoder didn't surface the gold
    document at any rank. Reducing this is exactly what the variant
    axis is meant to do, so its delta from raw → variant is the
    headline candidate-pool metric the report leads with.
    """
    n = 0
    for row in rows:
        expected = [str(d) for d in (_row_get(row, "expected_doc_ids") or []) if d]
        if not expected:
            continue
        candidates = [str(d) for d in (_row_get(row, "candidate_doc_ids") or []) if d]
        cand_set = set(candidates)
        if not any(d in cand_set for d in expected):
            n += 1
    return n
