"""Phase 7.4 — recovery policy layer for the controlled recovery loop.

Phase 7.3 emitted a per-query verdict whose ``recommended_action`` is
one of {ANSWER, ANSWER_WITH_CAUTION, HYBRID_RECOVERY, QUERY_REWRITE,
ASK_CLARIFICATION, INSUFFICIENT_EVIDENCE}. Phase 7.4 turns each verdict
into a *recovery decision* — a typed instruction the controlled loop
consumes to either retry retrieval (hybrid or query rewrite) or to
explicitly refuse / defer.

Design constraints (frozen contracts the loop relies on):

  - Production code is NOT touched. The policy is a pure planning
    function over a verdict dict; the loop calls it once per query and
    routes to the BM25 / hybrid path it returns.
  - Determinism. No LLM rewriter — the production-like rewrite uses
    only the top-N candidate canonical titles / aliases that were
    *already in the candidate pool*. The oracle rewrite uses the silver
    ``expected_title`` and is explicitly marked as upper-bound.
  - Label leakage refusal. Asking the policy for a production-like
    rewrite while passing ``expected_title`` will either *raise* (when
    ``strict_label_leakage=True``) or silently strip ``expected_title``
    from the rewriter input (when ``strict_label_leakage=False`` —
    the loop runs in non-strict mode for diagnostic comparisons but
    every emitted attempt records the strict-mode flag so a reviewer
    can tell which guarantee a row carries).

Action handling (the spec the Phase 7.4 brief asked for):

  - INSUFFICIENT_EVIDENCE → SKIP_REFUSE. No retrieval, mark refusal.
  - HYBRID_RECOVERY → ATTEMPT_HYBRID. BM25 wide pool over the original
    query, fused with the frozen dense top-N via RRF.
  - QUERY_REWRITE → ATTEMPT_REWRITE. Mode-dependent:
      * oracle: rewrite uses ``expected_title`` (+ aliases from silver).
        Marked oracle_upper_bound=True.
      * production_like: rewrite uses only top-N candidate canonical
        titles / aliases / retrieval_titles that the verdict already
        saw. ``expected_title`` is rejected/flagged.
  - ANSWER_WITH_CAUTION → SKIP_CAUTION. Kept for calibration reporting,
    no recovery this phase.
  - ASK_CLARIFICATION → SKIP_DEFER. Phase 7.4 does not implement
    interactive clarification; recorded as deferred.
  - ANSWER → NOOP (the loop never sees these — they're filtered by
    the recommended_recovery_queries.jsonl loader — but the policy
    handles the case for completeness).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from eval.harness.v4_confidence_detector import (
    ACTION_ANSWER,
    ACTION_ANSWER_WITH_CAUTION,
    ACTION_ASK_CLARIFICATION,
    ACTION_HYBRID_RECOVERY,
    ACTION_INSUFFICIENT_EVIDENCE,
    ACTION_QUERY_REWRITE,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen taxonomies — strings are part of the JSONL contract
# ---------------------------------------------------------------------------


REWRITE_MODE_ORACLE = "oracle"
REWRITE_MODE_PRODUCTION_LIKE = "production_like"
REWRITE_MODE_BOTH = "both"

REWRITE_MODES: Tuple[str, ...] = (
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
    REWRITE_MODE_BOTH,
)


# RecoveryAction values mirror the Phase 7.3 ACTION_* constants for the
# inputs we route on, plus three Phase 7.4-specific outcomes:
#
#  - SKIP_REFUSE       — INSUFFICIENT_EVIDENCE; no retrieval, refusal.
#  - SKIP_CAUTION      — ANSWER_WITH_CAUTION; calibration only.
#  - SKIP_DEFER        — ASK_CLARIFICATION; no recovery this phase.
#  - ATTEMPT_HYBRID    — HYBRID_RECOVERY; BM25 + dense RRF fusion.
#  - ATTEMPT_REWRITE   — QUERY_REWRITE; oracle or production_like.
#  - NOOP              — ANSWER (already-confident); should not appear
#                        in the recovery loop input but kept for
#                        completeness.
RECOVERY_ACTION_SKIP_REFUSE = "SKIP_REFUSE"
RECOVERY_ACTION_SKIP_CAUTION = "SKIP_CAUTION"
RECOVERY_ACTION_SKIP_DEFER = "SKIP_DEFER"
RECOVERY_ACTION_ATTEMPT_HYBRID = "ATTEMPT_HYBRID"
RECOVERY_ACTION_ATTEMPT_REWRITE = "ATTEMPT_REWRITE"
RECOVERY_ACTION_NOOP = "NOOP"

RECOVERY_ACTIONS: Tuple[str, ...] = (
    RECOVERY_ACTION_NOOP,
    RECOVERY_ACTION_SKIP_REFUSE,
    RECOVERY_ACTION_SKIP_CAUTION,
    RECOVERY_ACTION_SKIP_DEFER,
    RECOVERY_ACTION_ATTEMPT_HYBRID,
    RECOVERY_ACTION_ATTEMPT_REWRITE,
)


SKIP_REASON_REFUSED_INSUFFICIENT = "REFUSED_INSUFFICIENT_EVIDENCE"
SKIP_REASON_CAUTION_NOT_RECOVERED = "CAUTION_NOT_RECOVERED_THIS_PHASE"
SKIP_REASON_CLARIFICATION_DEFERRED = "CLARIFICATION_DEFERRED_THIS_PHASE"
SKIP_REASON_ALREADY_CONFIDENT = "ALREADY_CONFIDENT"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LabelLeakageError(ValueError):
    """Raised when a production-like rewrite is requested but the input
    carries ``expected_title`` and ``strict_label_leakage=True``.

    The error is preferable to silent stripping when the caller has
    asked for the strict guarantee — a downstream reader of
    ``oracle_rewrite_upper_bound.jsonl`` vs the production-like JSONL
    must be able to trust that the production-like rows never saw
    silver labels.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryDecision:
    """Per-query plan emitted by the policy.

    Carries enough information for the loop to execute the attempt
    without re-reading the verdict — the loop only reads
    ``query_text``, ``rewritten_query``, ``recovery_action``,
    ``rewrite_mode`` and the ``expected_*`` fields that travel through
    for metric computation only (never for retrieval).
    """

    query_id: str
    bucket: str
    original_action: str
    recovery_action: str
    skip_reason: Optional[str] = None
    rewrite_mode: Optional[str] = None
    oracle_upper_bound: bool = False
    query_text: str = ""
    rewritten_query: Optional[str] = None
    rewrite_terms: Tuple[str, ...] = ()
    rewrite_source: Optional[str] = None
    expected_title: Optional[str] = None
    gold_doc_id: Optional[str] = None
    gold_page_id: Optional[str] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryAttempt:
    """One executed (or explicitly skipped) recovery attempt.

    The loop builds this *after* it consumes ``RecoveryDecision`` and
    runs (or refuses to run) the corresponding retrieval. Latency is
    optional — populated only for ATTEMPT_* actions where the retrieval
    actually fired.
    """

    decision: RecoveryDecision
    before_rank: int
    before_top_doc_ids: Tuple[str, ...]
    before_top_chunk_ids: Tuple[str, ...]
    before_in_top_k: bool
    before_top1_score: Optional[float]
    after_rank: int
    after_top_doc_ids: Tuple[str, ...]
    after_top_chunk_ids: Tuple[str, ...]
    after_in_top_k: bool
    after_top1_score: Optional[float]
    final_k: int
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class RecoveryResult:
    """Per-query verdict on whether the recovery loop helped.

    Computed from a ``RecoveryAttempt`` by ``classify_attempt``. A row
    is ``recovered`` iff gold was missing or below ``final_k`` before
    and is now at-or-above ``final_k``. ``regression`` flags the case
    where gold *was* in the top-k and is no longer (or moved deeper
    in a strictly worse direction).

    Skipped (non-attempt) rows always have ``recovered=False`` and
    ``regression=False`` — the policy refused to run anything, so
    there is nothing to grade.
    """

    query_id: str
    bucket: str
    recovery_action: str
    rewrite_mode: Optional[str]
    oracle_upper_bound: bool
    skipped: bool
    recovered: bool
    regression: bool
    gold_newly_entered_candidates: bool
    before_rank: int
    after_rank: int
    rank_delta: Optional[int]
    final_k: int
    latency_ms: Optional[float]
    notes: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Helpers — verdict-row reading and rewrite construction
# ---------------------------------------------------------------------------


def _verdict_input_payload(verdict_row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded ``input`` block of a per-query verdict row.

    Phase 7.3 wraps the originating ConfidenceQueryInput summary under
    the ``input`` key (see ``_input_summary_for_jsonl``). Older or
    hand-written rows may put the fields at the top level — the policy
    accepts both shapes so it can be tested in isolation without the
    full JSONL writer.
    """
    inp = verdict_row.get("input")
    if isinstance(inp, Mapping):
        return inp
    return verdict_row


def _read_top_candidates_preview(
    verdict_row: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Pull the top-N candidate preview list off a verdict row.

    Phase 7.3 caps this at 5 entries; if the policy needs more (the
    ``top_n_for_production`` knob), the caller should supply the raw
    ``top_results`` from Phase 7.0 separately.
    """
    inp = _verdict_input_payload(verdict_row)
    preview = inp.get("top_candidates_preview")
    if isinstance(preview, list):
        return [p for p in preview if isinstance(p, Mapping)]
    return []


def _gather_candidate_title_terms(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    expected_title: Optional[str],
    drop_expected_title: bool,
) -> Tuple[List[str], List[str]]:
    """Collect canonical title / retrieval_title / alias-like tokens.

    Returns (terms, sources) — ``sources`` is parallel to ``terms`` and
    records where the term came from (``title``, ``retrieval_title``,
    ``section_path``) so a reviewer reading the JSONL can audit what
    the production-like rewrite actually saw.

    When ``drop_expected_title`` is True, any term whose normalised form
    matches ``expected_title`` is removed from the output. This is the
    leakage guard for production-like mode.
    """
    seen: Dict[str, str] = {}  # normalised → first-seen source
    out_terms: List[str] = []
    out_sources: List[str] = []

    expected_norm = _norm_title(expected_title) if expected_title else ""

    def _add(term: Optional[str], source: str) -> None:
        if not term:
            return
        norm = _norm_title(term)
        if not norm:
            return
        if norm in seen:
            return
        if drop_expected_title and expected_norm and norm == expected_norm:
            return
        seen[norm] = source
        out_terms.append(term)
        out_sources.append(source)

    for c in candidates[:max(1, int(limit))]:
        # Prefer retrieval_title (Phase 7.0 promoted it); fall back to
        # plain title. Both are pulled because the chunk-level metadata
        # may carry one but not the other.
        _add(c.get("retrieval_title"), "retrieval_title")
        _add(c.get("title"), "title")
        sp = c.get("section_path")
        if isinstance(sp, list) and sp:
            # Section_path[0] is typically the page-section anchor — useful
            # only when the title fields are empty (which happens on
            # un-enriched chunks). Keep it as a last-resort source so the
            # term pool isn't empty.
            _add(str(sp[0]), "section_path[0]")

    return out_terms, out_sources


def _norm_title(s: Optional[str]) -> str:
    """Loose canonical form for de-duping / comparing titles.

    Mirrors the comparator in v4_confidence_detector but we keep a
    local copy so the policy module has no inadvertent dependency on
    the detector's *private* helpers (the detector's
    ``_normalise_for_match`` is module-private).
    """
    if not s:
        return ""
    out = s.strip().lower()
    for sep in (" / ", " > ", "/", ">", "·"):
        out = out.replace(sep, " ")
    return " ".join(out.split())


def build_rewritten_query(
    *,
    query_text: str,
    expected_title: Optional[str],
    candidate_preview: Sequence[Mapping[str, Any]],
    mode: str,
    top_n_for_production: int = 5,
    strict_label_leakage: bool = True,
) -> Tuple[Optional[str], Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    """Compose the rewritten query for one decision.

    Returns ``(rewritten_query, terms, sources, source_label)``:

      - ``rewritten_query``: the rewritten string the loop hands to BM25.
        ``None`` when no useful term could be extracted (the loop falls
        back to original-query BM25 in that case).
      - ``terms``: parallel list of the title/alias terms that were
        appended to the original query. Audit trail for the JSONL row.
      - ``sources``: parallel list of "where each term came from".
      - ``source_label``: high-level description of the rewriter mode
        actually executed — "oracle:expected_title",
        "production_like:top_n_titles", or "production_like:noop"
        when the candidate pool yielded zero usable titles.

    Modes:

      - ``oracle``: prepends ``expected_title`` then any aliases the
        candidate pool happens to surface. Loop must mark the row
        ``oracle_upper_bound=True``.
      - ``production_like``: uses ONLY candidate titles / retrieval
        titles. ``expected_title`` is forbidden — when present and
        ``strict_label_leakage=True`` we raise; otherwise we strip it
        from the term pool and emit a note in the source label so a
        downstream reader can spot the leniency.
      - ``both`` is not a valid value here — the loop runs the policy
        twice (once per mode) when ``--rewrite-mode both`` is set.

    The rewritten string layout: ``"<original> | <term1> <term2> ..."``.
    The pipe separator keeps the BM25 tokeniser from collapsing the
    appended terms into a single token, and reads cleanly when a
    reviewer dumps the JSONL.
    """
    if mode == REWRITE_MODE_ORACLE:
        return _build_oracle_rewrite(
            query_text=query_text,
            expected_title=expected_title,
            candidate_preview=candidate_preview,
            top_n_for_production=top_n_for_production,
        )
    if mode == REWRITE_MODE_PRODUCTION_LIKE:
        return _build_production_like_rewrite(
            query_text=query_text,
            expected_title=expected_title,
            candidate_preview=candidate_preview,
            top_n_for_production=top_n_for_production,
            strict_label_leakage=strict_label_leakage,
        )
    if mode == REWRITE_MODE_BOTH:
        raise ValueError(
            "build_rewritten_query: 'both' is a CLI-level fan-out, not "
            "a per-call mode. Call once per concrete mode."
        )
    raise ValueError(f"Unknown rewrite mode {mode!r}; expected one of {REWRITE_MODES}.")


def _build_oracle_rewrite(
    *,
    query_text: str,
    expected_title: Optional[str],
    candidate_preview: Sequence[Mapping[str, Any]],
    top_n_for_production: int,
) -> Tuple[Optional[str], Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    terms: List[str] = []
    sources: List[str] = []
    if expected_title:
        terms.append(str(expected_title))
        sources.append("expected_title")
    # Oracle still gets to see the candidate pool's titles — they may
    # carry aliases the silver retrieval_title doesn't list. This keeps
    # the oracle's upper-bound number meaningful as "best the rewrite
    # path could possibly do" rather than "expected_title alone".
    extra, extra_sources = _gather_candidate_title_terms(
        candidate_preview,
        limit=top_n_for_production,
        expected_title=expected_title,
        drop_expected_title=True,  # already added above
    )
    terms.extend(extra)
    sources.extend(extra_sources)
    if not terms:
        return None, (), (), "oracle:noop"
    rewritten = f"{query_text} | " + " ".join(terms)
    return rewritten, tuple(terms), tuple(sources), "oracle:expected_title"


def _build_production_like_rewrite(
    *,
    query_text: str,
    expected_title: Optional[str],
    candidate_preview: Sequence[Mapping[str, Any]],
    top_n_for_production: int,
    strict_label_leakage: bool,
) -> Tuple[Optional[str], Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    if expected_title and strict_label_leakage:
        raise LabelLeakageError(
            "production_like rewrite requested with expected_title set "
            "and strict_label_leakage=True. Refusing to leak silver "
            "labels into a production-like evaluation."
        )
    terms, sources = _gather_candidate_title_terms(
        candidate_preview,
        limit=top_n_for_production,
        expected_title=expected_title,
        drop_expected_title=True,
    )
    if not terms:
        # Production-like has no viable rewriter input — the candidate
        # pool yielded no titles distinct from expected_title (possibly
        # because every candidate already shared the expected title and
        # we filtered them all out). The loop should treat this as a
        # no-op rewrite and fall back to original-query BM25.
        return None, (), (), "production_like:noop"
    rewritten = f"{query_text} | " + " ".join(terms)
    return rewritten, tuple(terms), tuple(sources), "production_like:top_n_titles"


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------


def decide_recovery(
    verdict_row: Mapping[str, Any],
    *,
    rewrite_mode: str = REWRITE_MODE_PRODUCTION_LIKE,
    top_n_for_production: int = 5,
    strict_label_leakage: bool = True,
) -> RecoveryDecision:
    """Map one Phase 7.3 verdict row to a RecoveryDecision.

    The policy never *executes* retrieval — it only decides which path
    to send the row down. The loop reads the returned decision and
    handles execution / metric capture.

    ``rewrite_mode`` only affects QUERY_REWRITE rows. HYBRID rows are
    routed identically regardless of mode. INSUFFICIENT_EVIDENCE /
    ANSWER_WITH_CAUTION / ASK_CLARIFICATION are short-circuited to
    SKIP_* outcomes with a reason string Phase 7.4's summary table
    keys off.
    """
    if rewrite_mode not in REWRITE_MODES:
        raise ValueError(
            f"unknown rewrite_mode={rewrite_mode!r}; "
            f"expected one of {REWRITE_MODES}."
        )

    qid = str(verdict_row.get("query_id") or verdict_row.get("qid") or "")
    bucket = str(verdict_row.get("bucket") or "")
    inp = _verdict_input_payload(verdict_row)
    query_text = str(inp.get("query_text") or verdict_row.get("query") or "")
    expected_title = inp.get("expected_title")
    expected_title_s = str(expected_title) if expected_title else None
    gold_doc_id = inp.get("gold_doc_id")
    gold_page_id = inp.get("gold_page_id")
    gold_doc_id_s = str(gold_doc_id) if gold_doc_id else None
    gold_page_id_s = str(gold_page_id) if gold_page_id else None
    original_action = str(verdict_row.get("recommended_action") or "")

    base_kwargs: Dict[str, Any] = {
        "query_id": qid,
        "bucket": bucket,
        "original_action": original_action,
        "query_text": query_text,
        "expected_title": expected_title_s,
        "gold_doc_id": gold_doc_id_s,
        "gold_page_id": gold_page_id_s,
    }

    if original_action == ACTION_INSUFFICIENT_EVIDENCE:
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_SKIP_REFUSE,
            skip_reason=SKIP_REASON_REFUSED_INSUFFICIENT,
            **base_kwargs,
        )
    if original_action == ACTION_ANSWER_WITH_CAUTION:
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_SKIP_CAUTION,
            skip_reason=SKIP_REASON_CAUTION_NOT_RECOVERED,
            **base_kwargs,
        )
    if original_action == ACTION_ASK_CLARIFICATION:
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_SKIP_DEFER,
            skip_reason=SKIP_REASON_CLARIFICATION_DEFERRED,
            **base_kwargs,
        )
    if original_action == ACTION_ANSWER:
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_NOOP,
            skip_reason=SKIP_REASON_ALREADY_CONFIDENT,
            **base_kwargs,
        )
    if original_action == ACTION_HYBRID_RECOVERY:
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_ATTEMPT_HYBRID,
            **base_kwargs,
        )
    if original_action == ACTION_QUERY_REWRITE:
        if rewrite_mode == REWRITE_MODE_BOTH:
            raise ValueError(
                "decide_recovery cannot route a QUERY_REWRITE row when "
                "rewrite_mode='both'; the loop must call this with "
                "concrete oracle / production_like."
            )
        candidate_preview = _read_top_candidates_preview(verdict_row)
        rewritten, terms, sources, source_label = build_rewritten_query(
            query_text=query_text,
            expected_title=expected_title_s,
            candidate_preview=candidate_preview,
            mode=rewrite_mode,
            top_n_for_production=top_n_for_production,
            strict_label_leakage=strict_label_leakage,
        )
        return RecoveryDecision(
            recovery_action=RECOVERY_ACTION_ATTEMPT_REWRITE,
            rewrite_mode=rewrite_mode,
            oracle_upper_bound=(rewrite_mode == REWRITE_MODE_ORACLE),
            rewritten_query=rewritten,
            rewrite_terms=terms,
            rewrite_source=source_label,
            **base_kwargs,
        )
    # Unknown action — treat as defer, log so the loop can surface it.
    log.warning(
        "decide_recovery: unknown recommended_action=%r for qid=%s; "
        "deferring.", original_action, qid,
    )
    return RecoveryDecision(
        recovery_action=RECOVERY_ACTION_SKIP_DEFER,
        skip_reason=SKIP_REASON_CLARIFICATION_DEFERRED,
        notes=(f"unknown_action={original_action!r}",),
        **base_kwargs,
    )


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


_RANK_MISS = -1


def classify_attempt(
    attempt: RecoveryAttempt,
    *,
    final_k: Optional[int] = None,
) -> RecoveryResult:
    """Decide whether ``attempt`` represents recovery / regression / unchanged.

    Skipped (SKIP_*) attempts always classify as ``recovered=False,
    regression=False, gold_newly_entered_candidates=False``. The loop
    still emits one ``RecoveryResult`` per skipped row so the summary
    can count refusals / caution-deferrals alongside actual recoveries.

    For ATTEMPT_* rows:
      - ``recovered = True``  iff the gold rank moved from missing /
        > final_k to ≤ final_k.
      - ``gold_newly_entered_candidates = True`` iff before_rank was
        a miss (-1) and after_rank is now any positive number,
        regardless of whether it crossed final_k. This counts the
        looser pool-recall signal HYBRID is meant to deliver.
      - ``regression = True`` iff before_rank was ≤ final_k and
        after_rank is now > final_k (or a miss).
    """
    k = int(final_k if final_k is not None else attempt.final_k)
    skipped = attempt.decision.recovery_action in (
        RECOVERY_ACTION_SKIP_REFUSE,
        RECOVERY_ACTION_SKIP_CAUTION,
        RECOVERY_ACTION_SKIP_DEFER,
        RECOVERY_ACTION_NOOP,
    )

    before = attempt.before_rank
    after = attempt.after_rank

    rank_delta: Optional[int]
    if skipped:
        rank_delta = None
    elif before == _RANK_MISS and after == _RANK_MISS:
        rank_delta = None
    elif before == _RANK_MISS:
        # Negative delta = improvement (rank got smaller). Use a sentinel-ish
        # "k+1 minus after" to keep the magnitude meaningful.
        rank_delta = -(k + 1 - after) if after > 0 else None
    elif after == _RANK_MISS:
        rank_delta = (k + 1) - before  # positive = worse
    else:
        rank_delta = after - before

    if skipped:
        return RecoveryResult(
            query_id=attempt.decision.query_id,
            bucket=attempt.decision.bucket,
            recovery_action=attempt.decision.recovery_action,
            rewrite_mode=attempt.decision.rewrite_mode,
            oracle_upper_bound=attempt.decision.oracle_upper_bound,
            skipped=True,
            recovered=False,
            regression=False,
            gold_newly_entered_candidates=False,
            before_rank=before,
            after_rank=after,
            rank_delta=rank_delta,
            final_k=k,
            latency_ms=attempt.latency_ms,
            notes=attempt.decision.notes,
        )

    before_in = (before != _RANK_MISS) and (before <= k)
    after_in = (after != _RANK_MISS) and (after <= k)

    recovered = (not before_in) and after_in
    regression = before_in and (not after_in)
    gold_newly_entered = (
        before == _RANK_MISS and after != _RANK_MISS
    )

    notes: List[str] = list(attempt.decision.notes)
    if attempt.error:
        notes.append(f"error={attempt.error}")

    return RecoveryResult(
        query_id=attempt.decision.query_id,
        bucket=attempt.decision.bucket,
        recovery_action=attempt.decision.recovery_action,
        rewrite_mode=attempt.decision.rewrite_mode,
        oracle_upper_bound=attempt.decision.oracle_upper_bound,
        skipped=False,
        recovered=recovered,
        regression=regression,
        gold_newly_entered_candidates=gold_newly_entered,
        before_rank=before,
        after_rank=after,
        rank_delta=rank_delta,
        final_k=k,
        latency_ms=attempt.latency_ms,
        notes=tuple(notes),
    )
