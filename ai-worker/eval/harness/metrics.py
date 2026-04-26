"""Metrics used by the eval harnesses.

All pure-Python, zero external dependencies — deliberately: this file
is the one place in the repo that never needs updating when a new
model / engine is wired in. The implementations are obvious, not
optimized, and small enough to audit in one sitting.

What's here:

  - edit_distance : Levenshtein over any equal-length-comparable list
  - cer           : character error rate
  - wer           : word error rate (whitespace-tokenized)
  - hit_at_k      : 1.0 if any expected id is in the top-k, else 0.0
  - recall_at_k   : distinct-gold recall over top-k (normalized ids)
  - reciprocal_rank : 1/rank of the first matching expected id, else 0.0
  - keyword_coverage : fraction of expected keywords present (case-insensitive)
  - dup_rate      : 1 - unique/len for a top-k id list
  - p_percentile  : nearest-rank percentile over a list of values
  - topk_gap      : (absolute, relative) score gap between rank 1 and rank k

Retrieval-diagnostic metrics (added for the `retrieval` eval mode that
measures dense-retrieval baseline quality without invoking a generator):

  - reciprocal_rank_at_k       : MRR with an explicit cutoff (MRR@10)
  - ndcg_at_k                  : binary-relevance NDCG@k
  - unique_doc_coverage        : distinct doc_ids in top-k / k
  - top1_score_margin          : scores[0] - scores[1] (rank-1 vs rank-2)
  - count_whitespace_tokens    : crude token count for context-length budgets
  - expected_keyword_match_rate: fraction of expected keywords found in any
                                 top-k chunk text/section name
  - normalized_text_hash       : NFKC + lowercase + ws-collapse + sha1[:16]
                                 — near-duplicate detector for chunk text

None of these handle "language nuance" — CER is raw character edit
distance after an optional normalization pass, and WER is whitespace
split. For CJK languages, WER is generally not meaningful; use CER
only, which is how `ocr_eval.py` defaults per-row language.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Levenshtein (edit distance).
# ---------------------------------------------------------------------------


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Classic Wagner-Fischer with a two-row rolling buffer.

    Works on any sequence of hashable/equality-comparable items — use it
    with `list(str)` for character edit distance and with
    `str.split()` for word edit distance.

    Returns the integer count of insert + delete + substitute operations
    needed to turn `a` into `b`.
    """
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, token_a in enumerate(a, start=1):
        curr[0] = i
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev, curr = curr, prev
    return prev[-1]


# ---------------------------------------------------------------------------
# Normalization helpers for CER/WER.
# ---------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")


def _normalize_for_cer(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip ends.

    We deliberately do NOT lowercase here — case matters for most
    real-world OCR targets (capitalized proper nouns, acronyms). If
    callers want case-insensitive CER, they can lowercase before
    passing in.
    """
    return _WS_RE.sub(" ", text).strip()


def _normalize_for_wer(text: str) -> List[str]:
    """Whitespace tokenization after the same normalization as CER."""
    normalized = _normalize_for_cer(text)
    return normalized.split(" ") if normalized else []


# ---------------------------------------------------------------------------
# Character / word error rates.
# ---------------------------------------------------------------------------


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate.

    Returns the edit distance between hypothesis and reference,
    divided by the reference character count. Whitespace is normalized
    first so that trivial differences like "a  b" vs "a b" don't
    dominate the score.

    Edge cases:
      - empty reference + empty hypothesis → 0.0
      - empty reference + non-empty hypothesis → 1.0
        (every character is an insertion; the rate is defined relative
         to the hypothesis length in that degenerate case)
      - non-empty reference → distance / len(reference)
    """
    ref = _normalize_for_cer(reference)
    hyp = _normalize_for_cer(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    distance = edit_distance(list(hyp), list(ref))
    return distance / len(ref)


def wer(hypothesis: str, reference: str) -> float:
    """Word error rate, whitespace-tokenized.

    Same contract as `cer` but at the whitespace-delimited word level.
    Not meaningful for CJK languages that don't use spaces — the OCR
    harness falls back to CER-only for those rows.
    """
    ref_words = _normalize_for_wer(reference)
    hyp_words = _normalize_for_wer(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    distance = edit_distance(hyp_words, ref_words)
    return distance / len(ref_words)


# ---------------------------------------------------------------------------
# Retrieval metrics.
# ---------------------------------------------------------------------------


def hit_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Iterable[str],
    *,
    k: int,
) -> Optional[float]:
    """Binary hit@k: 1.0 if ANY expected id appears in the top-k, else 0.0.

    Returns None when `expected_doc_ids` is empty — the caller is
    responsible for treating None rows as "excluded from aggregation"
    rather than failing them. This matches the dataset convention
    where `expected_doc_ids` is optional per row.
    """
    expected = {d for d in expected_doc_ids if d}
    if not expected:
        return None
    top_k = list(retrieved_doc_ids)[: max(0, int(k))]
    return 1.0 if any(d in expected for d in top_k) else 0.0


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Iterable[str],
) -> Optional[float]:
    """Reciprocal rank of the first matching expected id.

    Returns 1/rank (1-indexed) of the first match in the ranked list,
    0.0 if no expected id appears in the list at all, and None if the
    row has no expected_doc_ids at all. Document-level averaging
    over this metric is the standard Mean Reciprocal Rank (MRR).
    """
    expected = {d for d in expected_doc_ids if d}
    if not expected:
        return None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def _normalize_doc_id(raw: str) -> str:
    """NFKC + lowercase + strip non-alphanumeric.

    Retrieval can round-trip ids through embeddings, stores, and user
    fixtures — unicode width, casing, and separator drift all show up
    as false misses in recall calculations. Normalize before comparing.
    """
    if not raw:
        return ""
    folded = unicodedata.normalize("NFKC", raw).casefold()
    return _NON_ALNUM_RE.sub("", folded)


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Iterable[str],
    *,
    k: int,
) -> Optional[float]:
    """Distinct-gold recall over the top-k retrieved ids.

    Counts how many unique ids from `expected_doc_ids` appear in the
    first k retrieved ids, divided by the number of unique expected
    ids. A gold id surfacing multiple times in the top-k is only
    credited once — this is what "recall" means, and it differs from
    hit@k which only asks whether any gold landed.

    Returns None when `expected_doc_ids` is empty so callers can skip
    the row during aggregation, matching the hit@k contract.
    """
    expected = {_normalize_doc_id(d) for d in expected_doc_ids if d}
    expected.discard("")
    if not expected:
        return None
    top_k = list(retrieved_doc_ids)[: max(0, int(k))]
    matched: set[str] = set()
    for doc_id in top_k:
        norm = _normalize_doc_id(doc_id)
        if norm in expected:
            matched.add(norm)
            if len(matched) == len(expected):
                break
    return len(matched) / len(expected)


def dup_rate(doc_ids_topk: Sequence[str]) -> float:
    """Fraction of duplicates in a top-k id list: 1 - unique/len.

    A high value means the retriever is returning the same doc under
    multiple chunks — useful as a diversity signal ahead of rerankers.
    Returns 0.0 for 0- and 1-element lists (no duplication possible).
    """
    n = len(doc_ids_topk)
    if n <= 1:
        return 0.0
    return 1.0 - (len(set(doc_ids_topk)) / float(n))


def p_percentile(values: Sequence[float], p: float = 95.0) -> float:
    """Nearest-rank percentile of `values` (0.0 on empty input).

    Uses the ceil(p/100 * n) - 1 index on the sorted list — deliberate
    match of the port/rag convention so eval reports across the two
    repos stay comparable. For latency tracking this is close enough
    to what percentile libraries give and needs no extra dependency.
    """
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(xs))) - 1
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def topk_gap(scores: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Absolute and relative score gap between rank 1 and rank k.

    Returns (abs_gap, rel_gap) where abs_gap = scores[0] - scores[-1]
    and rel_gap = abs_gap / |scores[0]|. A wide gap suggests the top
    hit is clearly better than the tail and a reranker has less to do;
    a narrow gap suggests rerank room.

    Returns (None, None) when fewer than 2 scores exist — the concept
    is undefined in that case. rel_gap is None when the top score is
    zero (avoids dividing by zero while keeping abs_gap meaningful).
    """
    if len(scores) < 2:
        return (None, None)
    top = float(scores[0])
    bottom = float(scores[-1])
    abs_gap = top - bottom
    denom = abs(top)
    rel_gap = (abs_gap / denom) if denom > 0.0 else None
    return (abs_gap, rel_gap)


# ---------------------------------------------------------------------------
# Keyword coverage for generated responses.
# ---------------------------------------------------------------------------


def keyword_coverage(
    response_text: str,
    expected_keywords: Iterable[str],
    *,
    case_insensitive: bool = True,
) -> Optional[float]:
    """Fraction of `expected_keywords` that appear as substrings.

    Returns a value in [0.0, 1.0], or None when the row has no
    expected keywords. Substring matching is deliberate — it's the
    cheapest reasonable signal for "the generator actually mentioned
    the thing we asked about". For stricter matching (whole-word,
    stemming, etc.), replace this function; the harness only depends
    on its signature.
    """
    keywords = [k for k in expected_keywords if k]
    if not keywords:
        return None

    haystack = response_text.lower() if case_insensitive else response_text
    hits = 0
    for keyword in keywords:
        needle = keyword.lower() if case_insensitive else keyword
        if needle and needle in haystack:
            hits += 1
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Agent loop metrics (phase 6).
# ---------------------------------------------------------------------------


def _get_attr(row: object, name: str) -> object:
    """Dict-or-object field accessor used by the agent metrics.

    The loop metrics are defined over the "compare" eval rows where each
    row carries both a loop-off snapshot (``iter0``) and a loop-on
    snapshot (``final``). Rows are plain dicts in the JSON output but
    dataclasses in memory; accepting both keeps the metrics callable
    from both the writer and in-memory tests.
    """
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def iter_count_mean(rows: Sequence[object]) -> Optional[float]:
    """Mean of per-row ``iter_count`` values (loop iterations run).

    Rows without an ``iter_count`` field are skipped. Returns None when
    no row contributed a value — callers treat None as "metric not
    defined for this dataset" rather than failing.
    """
    values: List[float] = []
    for row in rows:
        v = _get_attr(row, "iter_count")
        if v is None:
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return round(sum(values) / float(len(values)), 4)


def loop_recovery_rate(
    rows: Sequence[object],
    *,
    keyword_threshold: float = 0.5,
) -> Optional[float]:
    """Fraction of failing iter0 rows whose final answer clears the bar.

    A row "recovered" when ``iter0_keyword_coverage < keyword_threshold``
    AND ``final_keyword_coverage >= keyword_threshold``. The metric is
    ``recovered / needing_recovery``: how often did the loop actually
    rescue a weak first pass.

    Returns None when no row was "needing_recovery" — dividing by zero
    would be meaningless and the caller skips aggregation.
    """
    needing: int = 0
    recovered: int = 0
    for row in rows:
        iter0 = _get_attr(row, "iter0_keyword_coverage")
        final = _get_attr(row, "final_keyword_coverage")
        if iter0 is None or final is None:
            continue
        try:
            iter0_f = float(iter0)
            final_f = float(final)
        except (TypeError, ValueError):
            continue
        if iter0_f < keyword_threshold:
            needing += 1
            if final_f >= keyword_threshold:
                recovered += 1
    if needing == 0:
        return None
    return round(recovered / float(needing), 4)


def avg_cost_multiplier(rows: Sequence[object]) -> Optional[float]:
    """Mean of per-row ``total_tokens / iter0_tokens`` ratios.

    The iter0 tokens approximate "what a single-shot (loop-off) run
    would have cost"; the total tokens count every iteration. A value
    of 1.0 means the loop didn't add cost (converged at iter 0). Higher
    values indicate the loop paid for additional iterations. Rows with
    zero or missing iter0 tokens are excluded — cost multiplier is
    undefined in that case.
    """
    ratios: List[float] = []
    for row in rows:
        total = _get_attr(row, "total_tokens")
        iter0 = _get_attr(row, "iter0_tokens")
        if total is None or iter0 is None:
            continue
        try:
            total_f = float(total)
            iter0_f = float(iter0)
        except (TypeError, ValueError):
            continue
        if iter0_f <= 0.0:
            continue
        ratios.append(total_f / iter0_f)
    if not ratios:
        return None
    return round(sum(ratios) / float(len(ratios)), 4)


def answer_recall_delta(rows: Sequence[object]) -> Optional[float]:
    """Mean of ``final_keyword_coverage - iter0_keyword_coverage``.

    Positive means the loop improved answer coverage on average;
    negative would mean it regressed (which Phase 8 would flag as a
    failure). Rows where either side is missing are skipped so the
    metric is taken only over the sample it can actually compare.
    """
    deltas: List[float] = []
    for row in rows:
        iter0 = _get_attr(row, "iter0_keyword_coverage")
        final = _get_attr(row, "final_keyword_coverage")
        if iter0 is None or final is None:
            continue
        try:
            deltas.append(float(final) - float(iter0))
        except (TypeError, ValueError):
            continue
    if not deltas:
        return None
    return round(sum(deltas) / float(len(deltas)), 4)


# ---------------------------------------------------------------------------
# Retrieval-diagnostic metrics (added for the `retrieval` eval mode).
#
# These are pure functions over (retrieved ranked list, expected gold,
# k). They never depend on a generator output — the goal is to measure
# dense-retrieval baseline quality in isolation. Existing functions
# above (`hit_at_k`, `recall_at_k`, `reciprocal_rank`, `dup_rate`,
# `topk_gap`, `keyword_coverage`) are unchanged so the older `rag` eval
# path keeps producing byte-identical numbers.
# ---------------------------------------------------------------------------


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Iterable[str],
    *,
    k: int,
) -> Optional[float]:
    """Reciprocal rank with an explicit cutoff.

    Differs from ``reciprocal_rank`` (which scans the entire ranked list)
    by capping at the first ``k`` results — matches the standard MRR@k
    definition. A gold id appearing at rank > k contributes 0.0 just like
    a complete miss. Returns ``None`` when ``expected_doc_ids`` is empty
    so the caller can skip the row from aggregation, matching the
    contract of the other retrieval metrics in this module.
    """
    expected = {_normalize_doc_id(d) for d in expected_doc_ids if d}
    expected.discard("")
    if not expected:
        return None
    cutoff = max(0, int(k))
    for rank, doc_id in enumerate(list(retrieved_doc_ids)[:cutoff], start=1):
        if _normalize_doc_id(doc_id) in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Iterable[str],
    *,
    k: int,
) -> Optional[float]:
    """Binary-relevance NDCG@k.

    Each retrieved id is scored 1 if it normalizes into the gold set and
    0 otherwise. Both the DCG sum and the ideal DCG (which assumes every
    gold id is packed into the top of the ranking) use the standard
    ``rel_i / log2(i + 1)`` discount. The denominator is
    ``min(|gold|, k)`` so a gold set larger than k cannot drag the ideal
    above 1.0 by counting positions the retriever was never asked about.

    Returns ``None`` when ``expected_doc_ids`` is empty (same contract as
    ``recall_at_k``). Returns 0.0 when no gold id appears in the top-k.
    """
    expected = {_normalize_doc_id(d) for d in expected_doc_ids if d}
    expected.discard("")
    if not expected:
        return None
    cutoff = max(0, int(k))
    if cutoff == 0:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(list(retrieved_doc_ids)[:cutoff], start=1):
        norm = _normalize_doc_id(doc_id)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in expected:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(expected), cutoff)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def unique_doc_coverage(
    retrieved_doc_ids: Sequence[str],
    *,
    k: int,
) -> Optional[float]:
    """Fraction of distinct doc_ids in the top-k slot budget.

    ``len({normalized_doc_ids[:k]}) / k``, in [0.0, 1.0]. The complement
    of ``dup_rate`` when computed over the same slice — surfaced as its
    own metric so retrieval reports can quote diversity directly without
    the caller having to invert.

    Returns ``None`` when ``retrieved_doc_ids`` is empty. Returns 0.0
    when ``k <= 0`` (no slots to cover, nothing meaningful to report).
    """
    if not retrieved_doc_ids:
        return None
    cutoff = max(0, int(k))
    if cutoff == 0:
        return 0.0
    top = [_normalize_doc_id(d) for d in retrieved_doc_ids[:cutoff]]
    top = [d for d in top if d]
    if not top:
        return 0.0
    distinct = len(set(top))
    return distinct / float(cutoff)


def top1_score_margin(scores: Sequence[float]) -> Optional[float]:
    """Score gap between rank 1 and rank 2.

    A wider margin means the top hit is clearly differentiated from the
    second; a narrow margin means the retriever is genuinely uncertain
    between the top two candidates. Distinct from ``topk_gap`` which
    spans rank 1 to rank k — top1_score_margin asks specifically about
    "is the leader convincing", not "is the tail far behind".

    Returns ``None`` when fewer than two scores are present (the gap is
    undefined). Negative values are possible — and meaningful — when the
    upstream reranker has reordered candidates against the bi-encoder
    score, so we do NOT clip to >= 0.
    """
    if scores is None or len(scores) < 2:
        return None
    return float(scores[0]) - float(scores[1])


_WS_TOKEN_RE = re.compile(r"\s+")


def count_whitespace_tokens(text: str) -> int:
    """Crude whitespace-tokenized word count.

    Used to estimate ``avg_context_token_count`` — the rough size of the
    text the retriever would hand to a downstream LLM. This is NOT a
    BPE / sentencepiece tokenizer; it's a coarse signal sufficient for
    "is the context budget blowing up?" diagnostics. For CJK text the
    count under-reports by ~4-6x compared to a real tokenizer, but the
    relative trend across runs is what the eval is measuring.
    """
    if not text:
        return 0
    parts = _WS_TOKEN_RE.split(text.strip())
    return sum(1 for p in parts if p)


def expected_keyword_match_rate(
    chunk_texts: Iterable[str],
    expected_keywords: Iterable[str],
    *,
    case_insensitive: bool = True,
) -> Optional[float]:
    """Fraction of expected keywords that appear in ANY retrieved chunk.

    Differs from ``keyword_coverage`` (which scores against a single
    generated answer string): this scores against the union of all
    retrieved chunk texts, which is the right denominator for "did the
    retriever surface evidence for what the eval row asked about".

    Returns ``None`` when ``expected_keywords`` is empty — same skip
    contract as ``keyword_coverage`` so aggregation drops the row.
    """
    keywords = [k for k in expected_keywords if k]
    if not keywords:
        return None
    haystack_parts: List[str] = []
    for text in chunk_texts:
        if not text:
            continue
        haystack_parts.append(text.lower() if case_insensitive else text)
    haystack = "\n".join(haystack_parts)
    if not haystack:
        return 0.0
    hits = 0
    for kw in keywords:
        needle = kw.lower() if case_insensitive else kw
        if needle and needle in haystack:
            hits += 1
    return hits / len(keywords)


def normalized_text_hash(text: str, *, prefix_chars: int = 512) -> str:
    """NFKC + casefold + whitespace-collapse + sha1[:16] over a prefix.

    A deterministic near-duplicate detector for chunk text. Two chunks
    that differ only in case, unicode width, or whitespace collapse
    to the same hash. Truncating to the first ``prefix_chars`` keeps
    the cost bounded on long body chunks while still discriminating
    between unrelated passages — same ranks of duplication near the
    head of two passages is what the duplicate-analysis report flags.

    Returns the empty string for empty input so callers can use
    "hash present" as a non-empty-text guard.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    collapsed = _WS_TOKEN_RE.sub(" ", folded).strip()
    if not collapsed:
        return ""
    head = collapsed[: max(1, int(prefix_chars))]
    digest = hashlib.sha1(head.encode("utf-8")).hexdigest()
    return digest[:16]
