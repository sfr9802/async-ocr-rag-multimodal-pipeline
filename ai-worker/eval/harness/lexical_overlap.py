"""Deterministic lexical overlap measures for the LLM silver-500 set.

Why this module exists
----------------------
The LLM silver-500 set author (the LLM in the outer loop) is supposed
to produce queries that *look* like a real user, not queries that
echo the corpus retrieval_title verbatim. The eval harness therefore
needs a way to flag, for each authored query, *how close* the query
is to its silver target on a purely lexical basis::

  - high overlap with the title  → the query reuses the title text;
  - high overlap with chunk text → the query copies content prose;
  - high BM25 rank for the page  → BM25 alone would already nail it.

When a query is *meant* to be a paraphrase / indirect / section-intent
test, a high lexical overlap is **leakage** — the silver-500 inflates
hit@k via string match instead of testing real semantic retrieval.
The leakage_guard module reads ``overlap_risk`` off this module to
flag those rows.

Determinism / dependency posture
--------------------------------
  - **No** Korean morphological analyser. The corpus mixes Hangul +
    English + numbers + special chars, and bringing in mecab/khaiii
    would tie the eval harness to a tokenizer install that the rest
    of the project doesn't carry. We work at the **character n-gram**
    level instead — robust across languages, no extra deps, byte-
    deterministic across machines.
  - **No** randomness. Every value the module emits is a pure function
    of the inputs.

Algorithms
----------

1. ``normalize_text(s)`` — collapse whitespace, strip punctuation /
   special chars (``♪``, ``!!``, brackets, ``·``, ``、``…), keep
   Hangul / Latin / digits, lowercase ASCII. Returns the canonical
   form used by every n-gram step below.

2. ``char_ngrams(s, n)`` — frozenset of overlapping char ``n``-grams
   over the *normalized* string. We use frozenset (not Counter) so
   Jaccard / containment are pure set ops — much faster than tf-based
   measures and adequate for a leakage signal.

3. ``jaccard(a, b)`` — ``|a ∩ b| / |a ∪ b|``. Returns 0.0 for two
   empty sets (rather than NaN) so downstream comparisons work.

4. ``containment(query_grams, target_grams)`` —
   ``|query_grams ∩ target_grams| / |query_grams|``. Asymmetric on
   purpose: we ask "how much of the *query* is buried in the target?"
   not "how similar are they overall?".

5. ``compute_overlap(...)`` — the public entry: takes a query string,
   the silver-target's title, the joined section path, and the target
   text (concat of chunk_text), and emits the dict that lands inside
   ``record["lexical_overlap"]``.

6. ``classify_overlap_risk(...)`` — applies the spec's thresholds:

       overlap_risk = "high"   if title_char2_jaccard       >= 0.55
                              or chunk_char4_containment   >= 0.35
                              or bm25_first_rank           <= 3
                    = "medium" if title_char2_jaccard       >= 0.30
                              or chunk_char4_containment   >= 0.20
                              or bm25_first_rank           <= 10
                    = "low"    otherwise.

   For ``unanswerable_or_not_in_corpus`` we emit
   ``"not_applicable"`` (the input dict carries ``None`` for every
   numeric field — the threshold check returns "not_applicable"
   without looking at any value).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, Optional


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


# Characters we collapse to whitespace before computing n-grams. The
# spec says "괄호/구두점/특수문자 제거 또는 공백화"; we go with
# whitespace-replacement so a token boundary is preserved (``A♪B`` →
# ``A B`` not ``AB``). Keeping Hangul (U+AC00–U+D7A3) + Latin + digits
# is enough for this corpus.
_NON_SEMANTIC_CHARS_RE = re.compile(
    # Anything that is NOT Hangul syllable + Hangul Jamo + Latin
    # alphanumeric + ASCII digit + whitespace becomes a space.
    r"[^"
    r"ㄱ-ㆎ"     # Hangul compatibility Jamo (ㄱ-ㆎ)
    r"가-힣"     # Hangul syllables
    r"A-Za-z0-9"
    r"\s"
    r"]+"
)
_WS_RE = re.compile(r"\s+")


def normalize_text(s: Optional[str]) -> str:
    """Canonical form for n-gram computation.

    Steps (order matters — NFKC first so half-width / full-width
    variants of the same char become identical before we strip):

      1. Unicode NFKC.
      2. Replace any non-Hangul / non-Latin-alnum / non-digit /
         non-whitespace char with a single space.
      3. Lowercase the ASCII portion (Hangul has no case).
      4. Collapse runs of whitespace.

    Empty/``None`` input returns ``""``.
    """
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", s)
    t = _NON_SEMANTIC_CHARS_RE.sub(" ", t)
    t = t.lower()
    t = _WS_RE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Char n-grams + set similarity
# ---------------------------------------------------------------------------


def char_ngrams(s: Optional[str], n: int) -> FrozenSet[str]:
    """Frozenset of overlapping char ``n``-grams over the normalized text.

    Whitespace is *kept* inside the n-grams so multi-word query
    boundaries influence the gram set (e.g. ``"원피스 줄거리"`` and
    ``"원피스 등장인물"`` differ on more than just the leaf).

    For ``len(normalized) < n`` we return a frozenset containing the
    whole string as a single "short" gram — this avoids zero-division
    in Jaccard for very short queries while still being deterministic.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    text = normalize_text(s)
    if not text:
        return frozenset()
    if len(text) < n:
        return frozenset({text})
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard similarity on two frozensets of grams.

    Returns 0.0 (not NaN) when both sets are empty — empty grams happen
    on a normalized empty string. The threshold compare in
    ``classify_overlap_risk`` then naturally reads "no overlap".
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def containment(query_grams: FrozenSet[str], target_grams: FrozenSet[str]) -> float:
    """Asymmetric containment: how much of ``query_grams`` is in target.

    ``|q ∩ t| / |q|``. Returns 0.0 when ``q`` is empty so the call is
    safe on empty queries.
    """
    if not query_grams:
        return 0.0
    inter = len(query_grams & target_grams)
    return inter / len(query_grams)


# ---------------------------------------------------------------------------
# Public entry: compute the lexical_overlap block
# ---------------------------------------------------------------------------


def compute_overlap(
    query: str,
    *,
    expected_title: Optional[str],
    expected_section_path: Optional[Iterable[str]],
    target_text: Optional[str],
    bm25_first_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce the ``lexical_overlap`` block for one query record.

    Inputs:
      query                   — the user-facing query text.
      expected_title          — silver_expected_title (None when
                                ``expected_not_in_corpus``).
      expected_section_path   — list/tuple of section path components,
                                joined with " " before n-gramming.
                                None when no path is available.
      target_text             — concat of the silver target's
                                chunk_text(s). None when no target.
      bm25_first_rank         — pre-computed BM25 first-chunk rank for
                                the silver page; None when not run.

    Output dict keys follow the spec verbatim:

      - title_char2_jaccard
      - section_char2_jaccard
      - chunk_char4_containment
      - bm25_expected_page_first_rank
      - overlap_risk
    """
    title_j: Optional[float]
    section_j: Optional[float]
    chunk_c: Optional[float]

    if expected_title:
        q2 = char_ngrams(query, 2)
        t2 = char_ngrams(expected_title, 2)
        title_j = round(jaccard(q2, t2), 6)
    else:
        title_j = None

    if expected_section_path:
        joined = " ".join(str(s) for s in expected_section_path if s)
        if joined:
            q2 = char_ngrams(query, 2)
            s2 = char_ngrams(joined, 2)
            section_j = round(jaccard(q2, s2), 6)
        else:
            section_j = None
    else:
        section_j = None

    if target_text:
        q4 = char_ngrams(query, 4)
        t4 = char_ngrams(target_text, 4)
        chunk_c = round(containment(q4, t4), 6)
    else:
        chunk_c = None

    risk = classify_overlap_risk(
        title_char2_jaccard=title_j,
        chunk_char4_containment=chunk_c,
        bm25_first_rank=bm25_first_rank,
        is_not_in_corpus=(expected_title is None and target_text is None),
    )

    return {
        "title_char2_jaccard": title_j,
        "section_char2_jaccard": section_j,
        "chunk_char4_containment": chunk_c,
        "bm25_expected_page_first_rank": bm25_first_rank,
        "overlap_risk": risk,
    }


# ---------------------------------------------------------------------------
# Risk classification — frozen thresholds from the spec
# ---------------------------------------------------------------------------


# Frozen thresholds. Public so tests can grep them; do NOT change
# without updating the spec + summary report copy.
TITLE_HIGH = 0.55
TITLE_MEDIUM = 0.30
CHUNK_HIGH = 0.35
CHUNK_MEDIUM = 0.20
BM25_HIGH = 3
BM25_MEDIUM = 10


def classify_overlap_risk(
    *,
    title_char2_jaccard: Optional[float],
    chunk_char4_containment: Optional[float],
    bm25_first_rank: Optional[int],
    is_not_in_corpus: bool = False,
) -> str:
    """Apply the spec's high / medium / low thresholds.

    ``is_not_in_corpus`` short-circuits to ``"not_applicable"`` because
    the lexical_overlap fields are all None for those rows — there is
    nothing to compare against. The check is explicit (not derived from
    the input being None) so a degenerate case where one input happens
    to be None doesn't accidentally return ``"not_applicable"``.
    """
    if is_not_in_corpus:
        return "not_applicable"
    # high
    if (title_char2_jaccard is not None and title_char2_jaccard >= TITLE_HIGH):
        return "high"
    if (chunk_char4_containment is not None and chunk_char4_containment >= CHUNK_HIGH):
        return "high"
    if (bm25_first_rank is not None and bm25_first_rank <= BM25_HIGH):
        return "high"
    # medium
    if (title_char2_jaccard is not None and title_char2_jaccard >= TITLE_MEDIUM):
        return "medium"
    if (chunk_char4_containment is not None and chunk_char4_containment >= CHUNK_MEDIUM):
        return "medium"
    if (bm25_first_rank is not None and bm25_first_rank <= BM25_MEDIUM):
        return "medium"
    return "low"
