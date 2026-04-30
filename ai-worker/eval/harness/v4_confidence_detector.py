"""Phase 7.3 — Retrieval Confidence Detector / Failure Classifier.

Classifies a query's retrieval result into one of
{CONFIDENT, AMBIGUOUS, LOW_CONFIDENCE, FAILED} with a list of failure
reasons and a recommended action. The classifier is the gate Phase 7.4
will use to decide whether to trigger an agentic recovery loop — this
phase ships only the gate, not the loop.

Design constraints:

  - Production answer generation is NOT touched. The classifier reads
    Phase 7.0's ``per_query_comparison.jsonl`` (and optionally Phase
    7.1's reranker per-query bundle) and emits its own JSONL artefact;
    nothing in ``app/capabilities/rag/`` calls into this module yet.

  - Reranker is NOT required. Dense-only Phase 7.0 input is fully
    sufficient. When Phase 7.1 reranker rows are also supplied, the
    classifier additionally surfaces ``rerank_demoted_gold`` and uses
    rerank scores in the top-1/top-2 margin calculation.

  - Heuristic thresholds live in :class:`ConfidenceConfig`. Defaults
    are conservative on purpose — Phase 7.4 will tune them against the
    label distribution this phase produces. Threshold optimisation is
    explicitly out of scope here, the goal is reproducible classification
    that exposes its inputs.

Failure reason taxonomy (every reason emitted by :func:`decide` is one
of these strings, frozen for downstream consumers):

  - ``LOW_TOP1_SCORE``           dense / rerank top-1 below threshold
  - ``LOW_MARGIN``               top1-top2 score gap too small
  - ``PAGE_ID_DISAGREEMENT``     top-k chunks disagree on doc/page id
  - ``TITLE_ALIAS_MISMATCH``     top-1 title doesn't match the query's
                                 expected title / retrieval_title
  - ``SECTION_INTENT_MISMATCH``  top-1 section_type ≠ expected page_type
  - ``GENERIC_COLLISION``        too many top-k chunks live under
                                 generic page titles (등장인물 / 평가 / …)
  - ``HIGH_DUPLICATE_RATE``      one doc dominates the top-k
  - ``GOLD_NOT_IN_CANDIDATES``   gold doc id not in the top-k
  - ``GOLD_LOW_RANK``            gold present but past the
                                 ``gold_low_rank_threshold`` slot
  - ``RERANK_DEMOTED_GOLD``      Phase 7.1 reranker dropped gold past
                                 final_k when it had been inside it
  - ``INSUFFICIENT_EVIDENCE``    fewer than ``min_evidence_chunks_same_page``
                                 chunks share the top-1 doc / page

Label assignment:

  - ``FAILED``         GOLD_NOT_IN_CANDIDATES with the gold available
                       (i.e. the gold IS known and we missed it). When
                       gold is unknown this label is never assigned.
  - ``LOW_CONFIDENCE`` at least one HARD reason (LOW_TOP1_SCORE,
                       INSUFFICIENT_EVIDENCE, PAGE_ID_DISAGREEMENT with
                       sub-threshold same_page_ratio, RERANK_DEMOTED_GOLD,
                       TITLE_ALIAS_MISMATCH).
  - ``AMBIGUOUS``      one or more SOFT reasons only (LOW_MARGIN,
                       GENERIC_COLLISION, HIGH_DUPLICATE_RATE,
                       SECTION_INTENT_MISMATCH, GOLD_LOW_RANK).
  - ``CONFIDENT``      no reasons emitted.

Action assignment uses the strongest action implied by the worst
reason — actions are ordered:
``INSUFFICIENT_EVIDENCE > HYBRID_RECOVERY > QUERY_REWRITE >
ASK_CLARIFICATION > ANSWER_WITH_CAUTION > ANSWER``.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen taxonomies — strings are part of the JSONL contract
# ---------------------------------------------------------------------------

LABEL_CONFIDENT = "CONFIDENT"
LABEL_AMBIGUOUS = "AMBIGUOUS"
LABEL_LOW_CONFIDENCE = "LOW_CONFIDENCE"
LABEL_FAILED = "FAILED"

CONFIDENCE_LABELS: Tuple[str, ...] = (
    LABEL_CONFIDENT,
    LABEL_AMBIGUOUS,
    LABEL_LOW_CONFIDENCE,
    LABEL_FAILED,
)

ACTION_ANSWER = "ANSWER"
ACTION_ANSWER_WITH_CAUTION = "ANSWER_WITH_CAUTION"
ACTION_HYBRID_RECOVERY = "HYBRID_RECOVERY"
ACTION_QUERY_REWRITE = "QUERY_REWRITE"
ACTION_ASK_CLARIFICATION = "ASK_CLARIFICATION"
ACTION_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

RECOMMENDED_ACTIONS: Tuple[str, ...] = (
    ACTION_ANSWER,
    ACTION_ANSWER_WITH_CAUTION,
    ACTION_HYBRID_RECOVERY,
    ACTION_QUERY_REWRITE,
    ACTION_ASK_CLARIFICATION,
    ACTION_INSUFFICIENT_EVIDENCE,
)

# Action precedence: when multiple reasons fire, this ordering picks the
# strongest recovery instruction. Higher index = stronger.
_ACTION_PRECEDENCE: Tuple[str, ...] = (
    ACTION_ANSWER,
    ACTION_ANSWER_WITH_CAUTION,
    ACTION_ASK_CLARIFICATION,
    ACTION_QUERY_REWRITE,
    ACTION_HYBRID_RECOVERY,
    ACTION_INSUFFICIENT_EVIDENCE,
)


REASON_LOW_TOP1_SCORE = "LOW_TOP1_SCORE"
REASON_LOW_MARGIN = "LOW_MARGIN"
REASON_PAGE_ID_DISAGREEMENT = "PAGE_ID_DISAGREEMENT"
REASON_TITLE_ALIAS_MISMATCH = "TITLE_ALIAS_MISMATCH"
REASON_SECTION_INTENT_MISMATCH = "SECTION_INTENT_MISMATCH"
REASON_GENERIC_COLLISION = "GENERIC_COLLISION"
REASON_HIGH_DUPLICATE_RATE = "HIGH_DUPLICATE_RATE"
REASON_GOLD_NOT_IN_CANDIDATES = "GOLD_NOT_IN_CANDIDATES"
REASON_GOLD_LOW_RANK = "GOLD_LOW_RANK"
REASON_RERANK_DEMOTED_GOLD = "RERANK_DEMOTED_GOLD"
REASON_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

FAILURE_REASONS: Tuple[str, ...] = (
    REASON_LOW_TOP1_SCORE,
    REASON_LOW_MARGIN,
    REASON_PAGE_ID_DISAGREEMENT,
    REASON_TITLE_ALIAS_MISMATCH,
    REASON_SECTION_INTENT_MISMATCH,
    REASON_GENERIC_COLLISION,
    REASON_HIGH_DUPLICATE_RATE,
    REASON_GOLD_NOT_IN_CANDIDATES,
    REASON_GOLD_LOW_RANK,
    REASON_RERANK_DEMOTED_GOLD,
    REASON_INSUFFICIENT_EVIDENCE,
)


# Generic page-title tokens reused from the Phase 7.0 collision counter.
# Kept verbatim so a shift in one place is visible in the other; the
# canonical list lives in ``corpus_title_noise.md`` notes (Phase 6.3).
_GENERIC_TITLE_KEYWORDS: frozenset = frozenset({
    "등장인물", "평가", "OST", "기타", "회차", "에피소드", "주제가",
    "음악", "회차 목록", "에피소드 가이드", "미디어 믹스", "기타 등장인물",
    "설정", "줄거리", "스태프", "성우진",
})


# v4 silver page_type values that have a direct counterpart in the
# chunk-level section_type vocabulary. Only these are forwarded to the
# classifier as ``expected_section_type`` by the loader; values like
# "main" / "setting" do NOT have a direct chunk_section_type match
# (chunks on a "main" page are typed as "summary" / "story" / etc),
# so feeding them through would over-fire SECTION_INTENT_MISMATCH.
_DIRECT_PAGE_TYPE_MATCHES: frozenset = frozenset({
    "character", "story", "reception", "summary",
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceConfig:
    """Heuristic thresholds.

    Defaults are deliberately conservative. The intent of Phase 7.3 is
    to *describe* the retrieval distribution, not to lock in operating
    points. Phase 7.4 will tune these against the label distribution
    this phase exports.

    Attributes:
      min_top1_score: floor on the top-1 chunk's score (rerank when
        present, dense otherwise). Below this, fire LOW_TOP1_SCORE.
        bge-m3 IP scores typically span [0.4, 0.95]; 0.55 is roughly
        the bottom quartile observed on the v4 silver set.
      min_margin: floor on top1−top2 gap on the comparable score
        column. A gap below this means rank 1 vs rank 2 is essentially
        a coin flip and the top-1 should not be trusted alone.
      min_same_page_ratio: floor on the share of top-k chunks that
        share the most-common doc_id. Below this, the retriever is
        spreading evidence across pages rather than converging on
        a single source.
      max_duplicate_rate: ceiling on the duplicate-doc ratio. Above
        this the top-k is one page repeating itself and there is
        almost no diversity left to disambiguate against.
      max_generic_collision_count: ceiling on the number of top-k
        chunks whose section path leads with a Phase 6.3 generic
        token (등장인물 / 평가 / …). High values mean the retriever
        is being asked to disambiguate noisy buckets.
      min_evidence_chunks_same_page: minimum chunks that must share
        the top-1 doc_id for the answer to be considered grounded.
        At least 2 means we want at least one corroborating chunk
        before we say "ANSWER".
      gold_low_rank_threshold: rank above which gold (when known)
        is considered too far down to be served confidently. Default
        of 5 keeps the top-1/top-3 zones clean.
    """

    min_top1_score: float = 0.55
    min_margin: float = 0.04
    min_same_page_ratio: float = 0.30
    # 0.90 chosen so that 10-of-10 same-doc top-K (dup=0.9) does NOT fire
    # the "noise" reason — full same-page convergence is the *desired*
    # signal, and we only want to flag pathological repetition (>0.9 means
    # K>10 with full repetition, which is pool ceiling).
    max_duplicate_rate: float = 0.90
    max_generic_collision_count: int = 6
    min_evidence_chunks_same_page: int = 2
    gold_low_rank_threshold: int = 5

    def validate(self) -> "ConfidenceConfig":
        if self.min_top1_score < 0.0:
            raise ValueError(
                f"min_top1_score must be non-negative, got {self.min_top1_score}."
            )
        if self.min_margin < 0.0:
            raise ValueError(
                f"min_margin must be non-negative, got {self.min_margin}."
            )
        if not (0.0 <= self.min_same_page_ratio <= 1.0):
            raise ValueError(
                f"min_same_page_ratio must be in [0, 1], "
                f"got {self.min_same_page_ratio}."
            )
        if not (0.0 <= self.max_duplicate_rate <= 1.0):
            raise ValueError(
                f"max_duplicate_rate must be in [0, 1], "
                f"got {self.max_duplicate_rate}."
            )
        if self.max_generic_collision_count < 0:
            raise ValueError(
                f"max_generic_collision_count must be non-negative, "
                f"got {self.max_generic_collision_count}."
            )
        if self.min_evidence_chunks_same_page < 1:
            raise ValueError(
                f"min_evidence_chunks_same_page must be >= 1, "
                f"got {self.min_evidence_chunks_same_page}."
            )
        if self.gold_low_rank_threshold < 1:
            raise ValueError(
                f"gold_low_rank_threshold must be >= 1, "
                f"got {self.gold_low_rank_threshold}."
            )
        return self


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateChunk:
    """One row of the top-k candidate list, post-rerank when applicable."""

    rank: int
    chunk_id: str
    doc_id: str
    title: Optional[str] = None
    retrieval_title: Optional[str] = None
    section_path: Tuple[str, ...] = ()
    section_type: Optional[str] = None
    section: Optional[str] = None
    dense_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    page_id: Optional[str] = None

    def effective_score(self) -> Optional[float]:
        """Score the classifier compares with ``min_top1_score``.

        Preference order: ``final_score`` (weighted blend output when
        Phase 7.1 ran in weighted mode) > ``rerank_score`` (Phase 7.1
        reranker_only) > ``dense_score`` (Phase 7.0 dense-only). The
        thresholding scale-shifts when the upstream score column
        changes; the config's ``min_top1_score`` default is dense-tuned,
        a tighter rerank-tuned value can be set explicitly when
        feeding rerank rows.
        """
        if self.final_score is not None:
            return float(self.final_score)
        if self.rerank_score is not None:
            return float(self.rerank_score)
        if self.dense_score is not None:
            return float(self.dense_score)
        return None

    def page_or_doc_id(self) -> str:
        """Return ``page_id`` if set, else ``doc_id``.

        In the v4 namu corpus chunks within a page share doc_id, so
        ``doc_id`` is the page-level grouping key by default. ``page_id``
        is left as a separate field for callers that index at a finer
        granularity (e.g. multi-page works split into per-section
        documents).
        """
        return self.page_id if self.page_id else self.doc_id


@dataclass(frozen=True)
class ConfidenceQueryInput:
    """Per-query input to :func:`decide`.

    Most fields are optional — gold IDs are only set on labelled
    silver queries; expected_title / expected_section_type only when
    the silver schema carries them (the Phase 7.0 v4 silver set does).
    """

    query_id: str
    query_text: str
    bucket: str = ""
    gold_doc_id: Optional[str] = None
    gold_page_id: Optional[str] = None
    expected_title: Optional[str] = None
    expected_section_type: Optional[str] = None
    extracted_query_terms: Tuple[str, ...] = ()
    top_candidates: Tuple[CandidateChunk, ...] = ()
    rerank_demoted_gold: Optional[bool] = None


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceSignals:
    """Numerical signals used (or computable) for the verdict."""

    top1_score: Optional[float]
    top1_top2_margin: Optional[float]
    page_id_consistency: float
    same_page_top_k_count: int
    candidate_count: int
    title_match: Optional[bool]
    retrieval_title_match: Optional[bool]
    section_type_match: Optional[bool]
    generic_collision_count: int
    duplicate_rate: float
    gold_in_top_k: Optional[bool]
    gold_rank: Optional[int]
    rerank_demoted_gold: Optional[bool]


@dataclass(frozen=True)
class ConfidenceVerdict:
    """Per-query classification result."""

    query_id: str
    bucket: str
    confidence_label: str
    failure_reasons: Tuple[str, ...]
    recommended_action: str
    signals: ConfidenceSignals
    debug_summary: str


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------


def _normalise_for_match(s: Optional[str]) -> str:
    """Canonical form used for title / retrieval_title comparison.

    Lowercased, whitespace-collapsed, separator-stripped. The match is
    deliberately loose: silver-meta retrieval_title takes the form
    ``"마법과고교의 열등생 / 설정"`` while the chunk-level
    retrieval_title can be either the full form or the work title alone
    depending on whether the chunk is on a sub-page or the main page.
    Loose matching keeps either form acceptable; a stricter equality
    check would over-fire the TITLE_ALIAS_MISMATCH reason.
    """
    if not s:
        return ""
    out = s.strip().lower()
    for sep in (" / ", " > ", "/", ">", "·"):
        out = out.replace(sep, " ")
    out = " ".join(out.split())
    return out


def _title_overlap(expected: Optional[str], observed: Optional[str]) -> Optional[bool]:
    """Decide whether a chunk's title aligns with the query's expected title.

    Returns ``None`` when either side is missing. Returns ``True`` when
    the normalised observed title contains the normalised expected
    title, or vice-versa — this admits both ``"마법과고교의 열등생"`` ⊂
    ``"마법과고교의 열등생 / 설정"`` and the reverse.
    """
    if expected is None or observed is None:
        return None
    e = _normalise_for_match(expected)
    o = _normalise_for_match(observed)
    if not e or not o:
        return None
    return (e in o) or (o in e)


def _generic_collision_count(candidates: Sequence[CandidateChunk]) -> int:
    """Count candidates whose first section component is a generic token.

    Mirrors ``v4_ab_eval._generic_collision_count`` but operates on the
    structured ``section_path`` tuple so we don't have to re-split the
    joined ``" > "`` string. Falls back to substring matching against
    the joined ``section`` field when ``section_path`` is empty.
    """
    if not candidates:
        return 0
    n = 0
    for c in candidates:
        if c.section_path:
            head = c.section_path[0] or ""
            if any(kw in head for kw in _GENERIC_TITLE_KEYWORDS):
                n += 1
                continue
        s = c.section or ""
        if any(kw in s for kw in _GENERIC_TITLE_KEYWORDS):
            n += 1
    return n


def _duplicate_rate(candidates: Sequence[CandidateChunk]) -> float:
    """Fraction of candidates whose doc_id was already seen at a higher rank.

    Same definition as ``v4_ab_eval._compute_dup_rate`` — kept duplicated
    here so this module has no source dep on v4_ab_eval (the loader
    bridges them at the artefact level instead).
    """
    if not candidates:
        return 0.0
    seen: set = set()
    dups = 0
    for c in candidates:
        if c.doc_id in seen:
            dups += 1
        else:
            seen.add(c.doc_id)
    return dups / len(candidates)


def _page_consistency(candidates: Sequence[CandidateChunk]) -> Tuple[float, int]:
    """Return (most_common_share, most_common_count) over page_or_doc_id.

    A perfectly consistent top-k (every chunk same page) returns
    ``(1.0, len(candidates))``. An evenly-spread top-k across n pages
    returns approximately ``(1/n, len(candidates) / n)``.
    """
    if not candidates:
        return 0.0, 0
    counts = Counter(c.page_or_doc_id() for c in candidates)
    _, top_count = counts.most_common(1)[0]
    return top_count / len(candidates), top_count


def _gold_rank(
    candidates: Sequence[CandidateChunk],
    gold_doc_id: Optional[str],
    gold_page_id: Optional[str],
) -> Optional[int]:
    """1-indexed rank of the first gold-matching chunk, or ``None`` if absent.

    Both ``gold_doc_id`` and ``gold_page_id`` are accepted; if both are
    set, either match counts. When neither is set, returns ``None`` so
    the caller can distinguish "gold unknown" from "gold missing".
    """
    if not gold_doc_id and not gold_page_id:
        return None
    for i, c in enumerate(candidates, start=1):
        if gold_doc_id and c.doc_id == gold_doc_id:
            return i
        if gold_page_id and c.page_id == gold_page_id:
            return i
    return -1


def _section_type_match(
    expected: Optional[str], observed: Optional[str],
) -> Optional[bool]:
    """Loose match: ``None`` when either side missing, else case-insensitive eq.

    ``expected_section_type`` carries the silver query's ``page_type``
    (e.g. ``setting`` / ``character`` / ``main``); the chunk's
    ``section_type`` is the per-section classifier's output (e.g.
    ``summary`` / ``character`` / ``story``). They overlap on the
    high-signal cases but aren't bijective — see
    ``corpus_title_noise.md``. We treat exact case-folded match as a
    positive signal and anything else as negative; this is intentionally
    biased to over-fire SECTION_INTENT_MISMATCH so callers can choose
    to demote it from a HARD reason to a SOFT one if they want.
    """
    if expected is None or observed is None:
        return None
    e = expected.strip().lower()
    o = observed.strip().lower()
    if not e or not o:
        return None
    return e == o


def _compute_signals(
    inp: ConfidenceQueryInput,
    *,
    config: ConfidenceConfig,
) -> ConfidenceSignals:
    """Materialise every signal :func:`decide` consults.

    Kept pure: takes only the input + config and returns a frozen
    dataclass. The decision step is split out so threshold tuning is
    cheap (re-decide existing signals against a new config without
    re-loading anything).
    """
    cands = list(inp.top_candidates)
    n = len(cands)

    top1 = cands[0] if cands else None
    top2 = cands[1] if len(cands) >= 2 else None

    top1_score = top1.effective_score() if top1 is not None else None
    top2_score = top2.effective_score() if top2 is not None else None
    margin: Optional[float] = None
    if top1_score is not None and top2_score is not None:
        margin = float(top1_score - top2_score)

    page_consistency_ratio, same_page_count = _page_consistency(cands)

    title_match = (
        _title_overlap(inp.expected_title, top1.title) if top1 else None
    )
    retrieval_title_match = (
        _title_overlap(inp.expected_title, top1.retrieval_title)
        if top1 else None
    )
    section_type_match = (
        _section_type_match(inp.expected_section_type, top1.section_type)
        if top1 else None
    )

    generic_count = _generic_collision_count(cands)
    dup_rate = _duplicate_rate(cands)

    if inp.gold_doc_id or inp.gold_page_id:
        gold_rank = _gold_rank(cands, inp.gold_doc_id, inp.gold_page_id)
        if gold_rank is None:
            gold_in_top_k: Optional[bool] = None
        else:
            gold_in_top_k = gold_rank > 0
    else:
        gold_rank = None
        gold_in_top_k = None

    return ConfidenceSignals(
        top1_score=top1_score,
        top1_top2_margin=margin,
        page_id_consistency=page_consistency_ratio,
        same_page_top_k_count=same_page_count,
        candidate_count=n,
        title_match=title_match,
        retrieval_title_match=retrieval_title_match,
        section_type_match=section_type_match,
        generic_collision_count=generic_count,
        duplicate_rate=dup_rate,
        gold_in_top_k=gold_in_top_k,
        gold_rank=gold_rank,
        rerank_demoted_gold=inp.rerank_demoted_gold,
    )


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


# HARD reasons map to LOW_CONFIDENCE / FAILED. SOFT reasons map to
# AMBIGUOUS unless combined with a HARD reason. Ordering inside each
# tuple is purely for documentation — the decision uses set membership.
_HARD_REASONS: frozenset = frozenset({
    REASON_LOW_TOP1_SCORE,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_PAGE_ID_DISAGREEMENT,
    REASON_RERANK_DEMOTED_GOLD,
    REASON_TITLE_ALIAS_MISMATCH,
    REASON_GOLD_NOT_IN_CANDIDATES,
})

_SOFT_REASONS: frozenset = frozenset({
    REASON_LOW_MARGIN,
    REASON_GENERIC_COLLISION,
    REASON_HIGH_DUPLICATE_RATE,
    REASON_SECTION_INTENT_MISMATCH,
    REASON_GOLD_LOW_RANK,
})


# Each reason maps to the action it would imply if it fired alone. The
# verdict picks the strongest (highest precedence) action across all
# fired reasons.
_REASON_ACTION: Dict[str, str] = {
    REASON_GOLD_NOT_IN_CANDIDATES: ACTION_INSUFFICIENT_EVIDENCE,
    REASON_INSUFFICIENT_EVIDENCE: ACTION_INSUFFICIENT_EVIDENCE,
    REASON_LOW_TOP1_SCORE: ACTION_HYBRID_RECOVERY,
    REASON_PAGE_ID_DISAGREEMENT: ACTION_HYBRID_RECOVERY,
    REASON_RERANK_DEMOTED_GOLD: ACTION_HYBRID_RECOVERY,
    REASON_TITLE_ALIAS_MISMATCH: ACTION_QUERY_REWRITE,
    REASON_LOW_MARGIN: ACTION_ANSWER_WITH_CAUTION,
    REASON_GENERIC_COLLISION: ACTION_ANSWER_WITH_CAUTION,
    REASON_HIGH_DUPLICATE_RATE: ACTION_ANSWER_WITH_CAUTION,
    REASON_SECTION_INTENT_MISMATCH: ACTION_ANSWER_WITH_CAUTION,
    REASON_GOLD_LOW_RANK: ACTION_ANSWER_WITH_CAUTION,
}


def _strongest_action(reasons: Sequence[str]) -> str:
    """Pick the action with the highest precedence among triggered reasons."""
    if not reasons:
        return ACTION_ANSWER
    best_idx = 0
    best_action = ACTION_ANSWER
    for r in reasons:
        a = _REASON_ACTION.get(r, ACTION_ANSWER)
        idx = _ACTION_PRECEDENCE.index(a) if a in _ACTION_PRECEDENCE else 0
        if idx > best_idx:
            best_idx = idx
            best_action = a
    return best_action


def _label_from_reasons(
    reasons: Sequence[str],
    *,
    gold_known: bool,
) -> str:
    """Map fired reasons to a confidence label.

    GOLD_NOT_IN_CANDIDATES with a known gold maps to FAILED — the
    retrieval result is provably wrong and no confidence dial can
    rescue it. Without a known gold the same reason cannot fire, so
    FAILED is reserved for the labelled-silver case.

    Any HARD reason → LOW_CONFIDENCE. Any SOFT reason (with no HARD
    reason) → AMBIGUOUS. No reasons → CONFIDENT.
    """
    if not reasons:
        return LABEL_CONFIDENT
    rs = set(reasons)
    if gold_known and REASON_GOLD_NOT_IN_CANDIDATES in rs:
        return LABEL_FAILED
    if rs & _HARD_REASONS:
        return LABEL_LOW_CONFIDENCE
    if rs & _SOFT_REASONS:
        return LABEL_AMBIGUOUS
    return LABEL_CONFIDENT


def _build_debug_summary(
    inp: ConfidenceQueryInput,
    signals: ConfidenceSignals,
    reasons: Sequence[str],
) -> str:
    """Return a one-line human-readable trace for the JSONL row.

    Layout: ``label-friendly trace | top1=<score> margin=<v>
    page_consistency=<v>/<n> dup=<v> generic=<n> [gold_rank=<n>]
    [rerank_demoted=<bool>] reasons=[...]``. The trace is stable in
    field order so a downstream grep / sort works without parsing JSON.
    """
    parts: List[str] = []
    parts.append(f"qid={inp.query_id}")
    parts.append(f"bucket={inp.bucket or '<none>'}")
    if signals.top1_score is not None:
        parts.append(f"top1={signals.top1_score:.4f}")
    if signals.top1_top2_margin is not None:
        parts.append(f"margin={signals.top1_top2_margin:+.4f}")
    parts.append(
        f"page={signals.page_id_consistency:.2f}"
        f"({signals.same_page_top_k_count}/{signals.candidate_count})"
    )
    parts.append(f"dup={signals.duplicate_rate:.2f}")
    parts.append(f"generic={signals.generic_collision_count}")
    if signals.gold_rank is not None:
        parts.append(f"gold_rank={signals.gold_rank}")
    if signals.rerank_demoted_gold is not None:
        parts.append(f"rerank_demoted={signals.rerank_demoted_gold}")
    parts.append("reasons=[" + ",".join(reasons) + "]")
    return " ".join(parts)


def decide(
    inp: ConfidenceQueryInput,
    *,
    config: Optional[ConfidenceConfig] = None,
) -> ConfidenceVerdict:
    """Classify one query's retrieval result.

    Determinism: every reason fires from a single, idempotent rule
    against the signal block — no probabilistic gates, no time-of-day
    dependency. The same input + config will always produce the same
    verdict, which is what makes the JSONL output reproducible.
    """
    cfg = (config or ConfidenceConfig()).validate()
    signals = _compute_signals(inp, config=cfg)
    reasons: List[str] = []

    # 1) Empty / sparse top-k → INSUFFICIENT_EVIDENCE. Triggers before
    #    any score-quality reason because score-on-zero-chunks is
    #    undefined and we don't want to read NoneType.
    if signals.candidate_count == 0:
        reasons.append(REASON_INSUFFICIENT_EVIDENCE)

    # 2) Gold-aware reasons. Need to fire before score-quality so that
    #    a "gold missing entirely" case is labelled FAILED rather than
    #    LOW_CONFIDENCE — the user-facing recovery flow differs.
    if signals.gold_rank is not None:
        if signals.gold_rank == -1:
            reasons.append(REASON_GOLD_NOT_IN_CANDIDATES)
        elif signals.gold_rank > cfg.gold_low_rank_threshold:
            reasons.append(REASON_GOLD_LOW_RANK)

    # 3) Reranker-attributable demotion (Phase 7.1 signal).
    if signals.rerank_demoted_gold is True:
        reasons.append(REASON_RERANK_DEMOTED_GOLD)

    # 4) Score quality.
    if (
        signals.top1_score is not None
        and signals.top1_score < cfg.min_top1_score
    ):
        reasons.append(REASON_LOW_TOP1_SCORE)
    if (
        signals.top1_top2_margin is not None
        and signals.top1_top2_margin < cfg.min_margin
    ):
        reasons.append(REASON_LOW_MARGIN)

    # 5) Evidence convergence. PAGE_ID_DISAGREEMENT is a HARD reason
    #    only when the same_page ratio is genuinely low — a 0.6 ratio
    #    on a top-10 (top-1's page has 6 chunks) is still a strong
    #    convergence even if it isn't 1.0, so we don't fire it.
    if (
        signals.candidate_count > 0
        and signals.page_id_consistency < cfg.min_same_page_ratio
    ):
        reasons.append(REASON_PAGE_ID_DISAGREEMENT)
    if (
        signals.candidate_count > 0
        and signals.same_page_top_k_count < cfg.min_evidence_chunks_same_page
    ):
        # Only count as INSUFFICIENT_EVIDENCE if not already triggered.
        if REASON_INSUFFICIENT_EVIDENCE not in reasons:
            reasons.append(REASON_INSUFFICIENT_EVIDENCE)

    # 6) Title / section semantics.
    if signals.retrieval_title_match is False:
        reasons.append(REASON_TITLE_ALIAS_MISMATCH)
    elif (
        signals.retrieval_title_match is None
        and signals.title_match is False
    ):
        # Only fall back to plain title check when retrieval_title is
        # missing on the chunk side; some indexes don't carry it.
        reasons.append(REASON_TITLE_ALIAS_MISMATCH)
    if signals.section_type_match is False:
        reasons.append(REASON_SECTION_INTENT_MISMATCH)

    # 7) Noise.
    if signals.generic_collision_count > cfg.max_generic_collision_count:
        reasons.append(REASON_GENERIC_COLLISION)
    if signals.duplicate_rate > cfg.max_duplicate_rate:
        reasons.append(REASON_HIGH_DUPLICATE_RATE)

    # Deduplicate while preserving fire order so the JSONL row reads
    # in the same order the rules ran.
    seen: set = set()
    ordered_reasons: List[str] = []
    for r in reasons:
        if r not in seen:
            ordered_reasons.append(r)
            seen.add(r)

    gold_known = (inp.gold_doc_id is not None or inp.gold_page_id is not None)
    label = _label_from_reasons(ordered_reasons, gold_known=gold_known)
    action = _strongest_action(ordered_reasons)

    return ConfidenceVerdict(
        query_id=inp.query_id,
        bucket=inp.bucket,
        confidence_label=label,
        failure_reasons=tuple(ordered_reasons),
        recommended_action=action,
        signals=signals,
        debug_summary=_build_debug_summary(inp, signals, ordered_reasons),
    )


# ---------------------------------------------------------------------------
# Loader: Phase 7.0 / 7.1 artefacts → ConfidenceQueryInput
# ---------------------------------------------------------------------------


def load_chunk_metadata(
    chunks_jsonl: Path,
) -> Dict[str, Dict[str, Any]]:
    """Index ``rag_chunks_*.jsonl`` by chunk_id for O(1) enrichment.

    Keys retained: title, retrieval_title, display_title, section_path,
    section_type, page_id (if present), doc_id. The full chunk text is
    NOT carried — the classifier never reads it, and per-row text would
    bloat the in-memory index by 100×+.
    """
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(chunks_jsonl)
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                continue
            sp = rec.get("section_path")
            section_path: Tuple[str, ...]
            if isinstance(sp, list):
                section_path = tuple(str(x) for x in sp)
            else:
                section_path = ()
            out[str(chunk_id)] = {
                "doc_id": str(rec.get("doc_id") or ""),
                "title": rec.get("title"),
                "retrieval_title": rec.get("retrieval_title"),
                "display_title": rec.get("display_title"),
                "section_path": section_path,
                "section_type": rec.get("section_type"),
                "page_id": rec.get("page_id"),
            }
    return out


def load_silver_queries(
    silver_jsonl: Path,
) -> Dict[str, Dict[str, Any]]:
    """Index ``queries_v4_silver.jsonl`` by qid.

    Returns the full record dict (the loader doesn't pre-pick fields
    because both the classifier and the report writer use different
    subsets — keeping the raw dict avoids a second pass over the file).
    """
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(silver_jsonl)
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("id") or rec.get("qid")
            if not qid:
                continue
            out[str(qid)] = rec
    return out


def _candidate_from_top_result(
    rank: int,
    row: Mapping[str, Any],
    *,
    chunk_index: Mapping[str, Mapping[str, Any]],
    rerank_score_lookup: Mapping[str, float],
    final_score_lookup: Mapping[str, float],
) -> CandidateChunk:
    """Build a CandidateChunk from one ``top_results`` entry, enriching."""
    chunk_id = str(row.get("chunk_id") or "")
    doc_id_row = row.get("doc_id")
    section_str = row.get("section")
    score = row.get("score")
    enrichment = chunk_index.get(chunk_id, {}) if chunk_index else {}
    section_path = tuple(enrichment.get("section_path") or ())
    title = enrichment.get("title")
    retrieval_title = enrichment.get("retrieval_title")
    section_type = enrichment.get("section_type")
    page_id = enrichment.get("page_id")
    doc_id = str(doc_id_row or enrichment.get("doc_id") or "")
    return CandidateChunk(
        rank=rank,
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        retrieval_title=retrieval_title,
        section_path=section_path,
        section_type=section_type,
        section=str(section_str) if section_str is not None else None,
        dense_score=float(score) if score is not None else None,
        rerank_score=rerank_score_lookup.get(chunk_id),
        final_score=final_score_lookup.get(chunk_id),
        page_id=page_id,
    )


def _build_input_from_phase7_row(
    per_query_row: Mapping[str, Any],
    *,
    silver_record: Optional[Mapping[str, Any]],
    chunk_index: Mapping[str, Mapping[str, Any]],
    side: str,
    rerank_pool: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ConfidenceQueryInput:
    """Compose a ConfidenceQueryInput from one Phase 7.0 per-query row.

    ``side`` selects ``baseline`` or ``candidate``; the candidate side
    is the production default after Phase 7.0 promoted
    retrieval_title_section. ``rerank_pool`` is Phase 7.1's
    ``candidate_pool_preview`` block — when provided, its rerank_score
    column overrides whatever score is on the top_results entries.
    """
    qid = str(per_query_row.get("qid") or "")
    query_text = str(per_query_row.get("query") or "")
    bucket = str(per_query_row.get("bucket") or "")
    v4_meta = per_query_row.get("v4_meta") or {}
    if isinstance(v4_meta, Mapping):
        bucket = bucket or str(v4_meta.get("bucket") or "")
    expected_doc_ids = per_query_row.get("expected_doc_ids") or []
    gold_doc_id: Optional[str] = None
    if expected_doc_ids:
        gold_doc_id = str(expected_doc_ids[0])

    expected_title: Optional[str] = None
    expected_section_type: Optional[str] = None
    if isinstance(v4_meta, Mapping):
        # retrieval_title is the most informative title field from the
        # silver-meta — it stitches work_title + page_title for sub-pages
        # and keeps just work_title for main pages.
        rt = v4_meta.get("retrieval_title")
        wt = v4_meta.get("work_title")
        expected_title = str(rt or wt) if (rt or wt) else None
        # page_type and chunk-level section_type live in different
        # vocabularies (e.g. page_type="main" vs section_type="summary"),
        # so we only forward page_type when it has a direct section_type
        # counterpart. Otherwise the SECTION_INTENT_MISMATCH reason would
        # over-fire on every main_work query. Phase 7.4 can enrich this
        # with a richer query-intent classifier when needed.
        pt = v4_meta.get("page_type")
        if pt and str(pt).strip().lower() in _DIRECT_PAGE_TYPE_MATCHES:
            expected_section_type = str(pt)

    # Pull side-specific block (baseline / candidate).
    side_block = per_query_row.get(side) or {}
    top_results = side_block.get("top_results") or []

    # Reranker-side scores: prefer the dedicated candidate_pool_preview
    # bundle when supplied, fall back to whatever the per-query row has.
    rerank_score_lookup: Dict[str, float] = {}
    final_score_lookup: Dict[str, float] = {}
    pool = rerank_pool
    if pool is None:
        pool = per_query_row.get("candidate_pool_preview") or []
    if pool:
        for entry in pool:
            cid = entry.get("chunk_id")
            if not cid:
                continue
            rs = entry.get("rerank_score")
            if rs is not None:
                rerank_score_lookup[str(cid)] = float(rs)
            fs = entry.get("final_score") or entry.get("blended_score")
            if fs is not None:
                final_score_lookup[str(cid)] = float(fs)

    candidates: List[CandidateChunk] = []
    for rank, row in enumerate(top_results, start=1):
        candidates.append(
            _candidate_from_top_result(
                rank,
                row,
                chunk_index=chunk_index,
                rerank_score_lookup=rerank_score_lookup,
                final_score_lookup=final_score_lookup,
            )
        )

    # Optional Phase 7.1 demotion flag — only present on rerank rows.
    rerank_demoted: Optional[bool] = None
    cand_block = per_query_row.get("candidate") or {}
    if "gold_was_demoted" in cand_block:
        rerank_demoted = bool(cand_block.get("gold_was_demoted"))

    extracted_terms: Tuple[str, ...] = ()
    if silver_record is not None:
        kw = silver_record.get("expected_section_keywords") or []
        if isinstance(kw, list):
            extracted_terms = tuple(str(x) for x in kw)
        # Prefer silver-record gold_page_id if present (rare on v4
        # silver but the loader supports it for forward compat).
        gpid = silver_record.get("gold_page_id")
        gold_page_id: Optional[str] = str(gpid) if gpid else None
    else:
        gold_page_id = None

    return ConfidenceQueryInput(
        query_id=qid,
        query_text=query_text,
        bucket=bucket,
        gold_doc_id=gold_doc_id,
        gold_page_id=gold_page_id,
        expected_title=expected_title,
        expected_section_type=expected_section_type,
        extracted_query_terms=extracted_terms,
        top_candidates=tuple(candidates),
        rerank_demoted_gold=rerank_demoted,
    )


def load_inputs_from_phase7_artifacts(
    per_query_path: Path,
    *,
    chunks_jsonl: Optional[Path] = None,
    silver_queries_path: Optional[Path] = None,
    side: str = "candidate",
    rerank_per_query_path: Optional[Path] = None,
) -> List[ConfidenceQueryInput]:
    """Build the per-query input list off Phase 7.0 / 7.1 outputs.

    Args:
      per_query_path: Phase 7.0's per_query_comparison.jsonl. The
        baseline / candidate split inside this file is honoured by
        ``side`` ("candidate" is the post-Phase-7.0 default).
      chunks_jsonl: optional rag_chunks_*.jsonl whose chunk_id index
        carries title / retrieval_title / section_type. When omitted,
        title / retrieval_title / section_type signals stay None and
        the corresponding reasons cannot fire — useful when running
        the classifier on a corpus without exported chunks.
      silver_queries_path: optional queries_v4_silver.jsonl. When set,
        provides expected_section_keywords and any gold_page_id.
      side: "candidate" (default) or "baseline".
      rerank_per_query_path: optional Phase 7.1 per_query_comparison.jsonl.
        When set, the rerank-side ``candidate.gold_was_demoted`` flag
        and the ``candidate_pool_preview`` rerank_score column are
        merged into the input. Matched by qid.
    """
    if side not in ("baseline", "candidate"):
        raise ValueError(f"side must be baseline / candidate, got {side!r}.")

    chunk_index: Dict[str, Dict[str, Any]] = {}
    if chunks_jsonl is not None:
        chunk_index = load_chunk_metadata(chunks_jsonl)

    silver_index: Dict[str, Dict[str, Any]] = {}
    if silver_queries_path is not None:
        silver_index = load_silver_queries(silver_queries_path)

    rerank_index: Dict[str, Mapping[str, Any]] = {}
    if rerank_per_query_path is not None:
        with Path(rerank_per_query_path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("qid")
                if qid:
                    rerank_index[str(qid)] = rec

    out: List[ConfidenceQueryInput] = []
    with Path(per_query_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            silver_rec = silver_index.get(qid)
            rerank_pool: Optional[Sequence[Mapping[str, Any]]] = None
            row_for_input: Mapping[str, Any] = row
            if rerank_index and qid in rerank_index:
                # Use the rerank-side row for top_results (final_k post
                # rerank) and the rerank candidate_pool_preview for
                # score columns.
                rerank_row = rerank_index[qid]
                rerank_pool = rerank_row.get("candidate_pool_preview") or []
                # Merge: keep per_query row's expected_doc_ids / v4_meta /
                # bucket but override side block with the rerank row's
                # candidate side, since we're classifying the rerank
                # output.
                merged: Dict[str, Any] = dict(row)
                merged["candidate"] = rerank_row.get("candidate") or {}
                row_for_input = merged
            inp = _build_input_from_phase7_row(
                row_for_input,
                silver_record=silver_rec,
                chunk_index=chunk_index,
                side=side,
                rerank_pool=rerank_pool,
            )
            out.append(inp)
    return out


# ---------------------------------------------------------------------------
# Aggregation + serialisation
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceEvalResult:
    """Container the writer takes; keeps verdicts + originating inputs."""

    verdicts: List[ConfidenceVerdict] = field(default_factory=list)
    inputs: List[ConfidenceQueryInput] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)


def _verdict_to_dict(v: ConfidenceVerdict) -> Dict[str, Any]:
    """Stable JSONL row shape."""
    return {
        "query_id": v.query_id,
        "bucket": v.bucket,
        "confidence_label": v.confidence_label,
        "failure_reasons": list(v.failure_reasons),
        "recommended_action": v.recommended_action,
        "signals": {
            "top1_score": v.signals.top1_score,
            "top1_top2_margin": v.signals.top1_top2_margin,
            "page_id_consistency": v.signals.page_id_consistency,
            "same_page_top_k_count": v.signals.same_page_top_k_count,
            "candidate_count": v.signals.candidate_count,
            "title_match": v.signals.title_match,
            "retrieval_title_match": v.signals.retrieval_title_match,
            "section_type_match": v.signals.section_type_match,
            "generic_collision_count": v.signals.generic_collision_count,
            "duplicate_rate": v.signals.duplicate_rate,
            "gold_in_top_k": v.signals.gold_in_top_k,
            "gold_rank": v.signals.gold_rank,
            "rerank_demoted_gold": v.signals.rerank_demoted_gold,
        },
        "debug_summary": v.debug_summary,
    }


def _input_summary_for_jsonl(inp: ConfidenceQueryInput) -> Dict[str, Any]:
    """Compact view of the originating input for the per-query JSONL.

    Only the fields a reviewer needs to make sense of the verdict —
    skips the full top_candidates list (a reader who needs that should
    fetch the Phase 7.0 per-query bundle by qid).
    """
    return {
        "query_text": inp.query_text,
        "gold_doc_id": inp.gold_doc_id,
        "gold_page_id": inp.gold_page_id,
        "expected_title": inp.expected_title,
        "expected_section_type": inp.expected_section_type,
        "candidate_count": len(inp.top_candidates),
        "top_candidates_preview": [
            {
                "rank": c.rank,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "retrieval_title": c.retrieval_title,
                "section_path": list(c.section_path),
                "section_type": c.section_type,
                "dense_score": c.dense_score,
                "rerank_score": c.rerank_score,
                "final_score": c.final_score,
            }
            for c in inp.top_candidates[:5]
        ],
    }


def aggregate_verdicts(
    verdicts: Sequence[ConfidenceVerdict],
    config: ConfidenceConfig,
) -> Dict[str, Any]:
    """Compute the summary block (counts, distributions, top-N reasons)."""
    n = len(verdicts)
    label_counts: Counter = Counter(v.confidence_label for v in verdicts)
    action_counts: Counter = Counter(v.recommended_action for v in verdicts)
    reason_counts: Counter = Counter()
    for v in verdicts:
        reason_counts.update(v.failure_reasons)

    by_bucket: Dict[str, Dict[str, Any]] = {}
    by_bucket_groups: Dict[str, List[ConfidenceVerdict]] = {}
    for v in verdicts:
        by_bucket_groups.setdefault(v.bucket or "<unbucketed>", []).append(v)
    for bucket, vs in sorted(by_bucket_groups.items()):
        b_label = Counter(v.confidence_label for v in vs)
        b_action = Counter(v.recommended_action for v in vs)
        b_reason: Counter = Counter()
        for v in vs:
            b_reason.update(v.failure_reasons)
        by_bucket[bucket] = {
            "count": len(vs),
            "labels": {lab: int(b_label.get(lab, 0)) for lab in CONFIDENCE_LABELS},
            "actions": {a: int(b_action.get(a, 0)) for a in RECOMMENDED_ACTIONS},
            "reasons": {r: int(b_reason.get(r, 0)) for r in FAILURE_REASONS},
        }

    return {
        "config": asdict(config),
        "n_queries": n,
        "labels": {
            lab: int(label_counts.get(lab, 0)) for lab in CONFIDENCE_LABELS
        },
        "actions": {
            a: int(action_counts.get(a, 0)) for a in RECOMMENDED_ACTIONS
        },
        "reasons": {
            r: int(reason_counts.get(r, 0)) for r in FAILURE_REASONS
        },
        "by_bucket": by_bucket,
        "confidence_labels": list(CONFIDENCE_LABELS),
        "recommended_actions": list(RECOMMENDED_ACTIONS),
        "failure_reasons": list(FAILURE_REASONS),
    }


def split_verdicts_by_label(
    verdicts: Sequence[ConfidenceVerdict],
    inputs: Sequence[ConfidenceQueryInput],
) -> Dict[str, List[Tuple[ConfidenceVerdict, ConfidenceQueryInput]]]:
    """Group (verdict, input) pairs by confidence_label."""
    by_qid: Dict[str, ConfidenceQueryInput] = {i.query_id: i for i in inputs}
    out: Dict[str, List[Tuple[ConfidenceVerdict, ConfidenceQueryInput]]] = {
        lab: [] for lab in CONFIDENCE_LABELS
    }
    for v in verdicts:
        out.setdefault(v.confidence_label, []).append(
            (v, by_qid.get(v.query_id, ConfidenceQueryInput(
                query_id=v.query_id, query_text=""
            )))
        )
    return out


def find_confident_but_wrong(
    verdicts: Sequence[ConfidenceVerdict],
) -> List[ConfidenceVerdict]:
    """CONFIDENT label but gold is missing — the high-precision failures.

    Useful for Phase 7.4 to size how many queries the gate would let
    through to ANSWER while the retrieval result is provably wrong.
    Requires gold to be known (gold_in_top_k must be a real bool).
    """
    out: List[ConfidenceVerdict] = []
    for v in verdicts:
        gitk = v.signals.gold_in_top_k
        if v.confidence_label == LABEL_CONFIDENT and gitk is False:
            out.append(v)
    return out


def find_low_confidence_but_correct(
    verdicts: Sequence[ConfidenceVerdict],
) -> List[ConfidenceVerdict]:
    """LOW_CONFIDENCE / FAILED but gold actually IS in top-1.

    Suggests the threshold pack is too aggressive — Phase 7.4 will use
    these as the false-negative bucket when re-tuning thresholds.
    """
    out: List[ConfidenceVerdict] = []
    for v in verdicts:
        gitk = v.signals.gold_in_top_k
        gr = v.signals.gold_rank
        if v.confidence_label in (LABEL_LOW_CONFIDENCE, LABEL_FAILED) and (
            gitk is True and gr is not None and gr == 1
        ):
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    result: ConfidenceEvalResult,
    *,
    out_dir: Path,
    summary_json_name: str = "confidence_summary.json",
    summary_md_name: str = "confidence_summary.md",
    per_query_name: str = "per_query_confidence.jsonl",
    low_confidence_name: str = "low_confidence_queries.jsonl",
    ambiguous_name: str = "ambiguous_queries.jsonl",
    failed_name: str = "failed_queries.jsonl",
    recovery_name: str = "recommended_recovery_queries.jsonl",
) -> Dict[str, Path]:
    """Persist the artefact bundle Phase 7.3 asks for.

    The ``recommended_recovery_queries.jsonl`` file is the union of
    every query whose recommended_action ≠ ANSWER — Phase 7.4 will
    consume this list as the recovery loop's input.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / summary_json_name
    summary_md = out_dir / summary_md_name
    per_query = out_dir / per_query_name
    low_conf = out_dir / low_confidence_name
    ambiguous = out_dir / ambiguous_name
    failed = out_dir / failed_name
    recovery = out_dir / recovery_name

    by_qid: Dict[str, ConfidenceQueryInput] = {
        i.query_id: i for i in result.inputs
    }

    summary_json.write_text(
        json.dumps(result.aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(
        render_summary_md(result),
        encoding="utf-8",
    )

    with per_query.open("w", encoding="utf-8") as fp:
        for v in result.verdicts:
            row = _verdict_to_dict(v)
            inp = by_qid.get(v.query_id)
            if inp is not None:
                row["input"] = _input_summary_for_jsonl(inp)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _dump_subset(path: Path, predicate) -> None:
        with path.open("w", encoding="utf-8") as fpp:
            for v in result.verdicts:
                if predicate(v):
                    row = _verdict_to_dict(v)
                    inp = by_qid.get(v.query_id)
                    if inp is not None:
                        row["input"] = _input_summary_for_jsonl(inp)
                    fpp.write(json.dumps(row, ensure_ascii=False) + "\n")

    _dump_subset(low_conf, lambda v: v.confidence_label == LABEL_LOW_CONFIDENCE)
    _dump_subset(ambiguous, lambda v: v.confidence_label == LABEL_AMBIGUOUS)
    _dump_subset(failed, lambda v: v.confidence_label == LABEL_FAILED)
    _dump_subset(recovery, lambda v: v.recommended_action != ACTION_ANSWER)

    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "per_query": per_query,
        "low_confidence": low_conf,
        "ambiguous": ambiguous,
        "failed": failed,
        "recovery": recovery,
    }


def render_summary_md(result: ConfidenceEvalResult) -> str:
    """Render a human-readable confidence_summary.md.

    Layout: header → label counts → action counts → reason counts →
    bucket breakdown → confident-but-wrong / low-confidence-but-correct
    samples. The "samples" section is capped at 10 each to keep the
    file scannable; full lists live in the JSONL writers.
    """
    agg = result.aggregate
    cfg = agg.get("config") or {}
    n = agg.get("n_queries", 0)
    lines: List[str] = []
    lines.append("# Phase 7.3 — Retrieval Confidence Detector summary")
    lines.append("")
    lines.append(f"- n_queries: **{n}**")
    lines.append("- thresholds:")
    for k in (
        "min_top1_score", "min_margin", "min_same_page_ratio",
        "max_duplicate_rate", "max_generic_collision_count",
        "min_evidence_chunks_same_page", "gold_low_rank_threshold",
    ):
        lines.append(f"  - {k}: {cfg.get(k)}")
    lines.append("")

    lines.append("## Confidence label distribution")
    lines.append("")
    lines.append("| label | count | share |")
    lines.append("|---|---:|---:|")
    labels = agg.get("labels") or {}
    for lab in CONFIDENCE_LABELS:
        c = int(labels.get(lab, 0))
        share = (c / n) if n else 0.0
        lines.append(f"| {lab} | {c} | {share:.2%} |")
    lines.append("")

    lines.append("## Recommended action distribution")
    lines.append("")
    lines.append("| action | count | share |")
    lines.append("|---|---:|---:|")
    actions = agg.get("actions") or {}
    for a in RECOMMENDED_ACTIONS:
        c = int(actions.get(a, 0))
        share = (c / n) if n else 0.0
        lines.append(f"| {a} | {c} | {share:.2%} |")
    lines.append("")

    lines.append("## Failure reason counts")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    reasons = agg.get("reasons") or {}
    for r in FAILURE_REASONS:
        lines.append(f"| {r} | {int(reasons.get(r, 0))} |")
    lines.append("")

    lines.append("## By bucket")
    lines.append("")
    by_bucket = agg.get("by_bucket") or {}
    for bucket, payload in by_bucket.items():
        lines.append(f"### {bucket} (n={payload.get('count', 0)})")
        lines.append("")
        b_labels = payload.get("labels") or {}
        b_actions = payload.get("actions") or {}
        b_reasons = payload.get("reasons") or {}
        lines.append(
            "- labels: "
            + ", ".join(f"{lab}={int(b_labels.get(lab, 0))}" for lab in CONFIDENCE_LABELS)
        )
        lines.append(
            "- actions: "
            + ", ".join(f"{a}={int(b_actions.get(a, 0))}" for a in RECOMMENDED_ACTIONS)
        )
        top_reasons = sorted(
            b_reasons.items(), key=lambda kv: kv[1], reverse=True,
        )[:5]
        if top_reasons:
            lines.append(
                "- top reasons: "
                + ", ".join(f"{r}={c}" for r, c in top_reasons if c > 0)
            )
        lines.append("")

    cb_wrong = find_confident_but_wrong(result.verdicts)
    lc_correct = find_low_confidence_but_correct(result.verdicts)
    lines.append("## Calibration cross-tabs")
    lines.append("")
    lines.append(
        f"- confident_but_wrong (CONFIDENT label, gold not in top-k): "
        f"**{len(cb_wrong)}**"
    )
    lines.append(
        f"- low_confidence_but_correct (LOW_CONFIDENCE/FAILED label, "
        f"gold at rank 1): **{len(lc_correct)}**"
    )
    lines.append("")
    if cb_wrong:
        lines.append("### Sample confident-but-wrong (≤10)")
        lines.append("")
        for v in cb_wrong[:10]:
            lines.append(
                f"- `{v.query_id}` ({v.bucket}): "
                f"top1={v.signals.top1_score} "
                f"page_consistency={v.signals.page_id_consistency:.2f} "
                f"gold_rank={v.signals.gold_rank}"
            )
        lines.append("")
    if lc_correct:
        lines.append("### Sample low-confidence-but-correct (≤10)")
        lines.append("")
        for v in lc_correct[:10]:
            lines.append(
                f"- `{v.query_id}` ({v.bucket}): "
                f"reasons=[{','.join(v.failure_reasons)}] "
                f"top1={v.signals.top1_score} "
                f"margin={v.signals.top1_top2_margin}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
