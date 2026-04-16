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
  - reciprocal_rank : 1/rank of the first matching expected id, else 0.0
  - keyword_coverage : fraction of expected keywords present (case-insensitive)

None of these handle "language nuance" — CER is raw character edit
distance after an optional normalization pass, and WER is whitespace
split. For CJK languages, WER is generally not meaningful; use CER
only, which is how `ocr_eval.py` defaults per-row language.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence


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
