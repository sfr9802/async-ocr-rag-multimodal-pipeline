"""Lightweight query normalization for Phase 2B candidate boost.

Deliberately conservative — NO query expansion, NO LLM rewriting, NO
synonym lookup. The point is to make string-match boost (title /
section) robust against trivial drift like double spaces, fullwidth
brackets, or stray punctuation. Anything more aggressive belongs in
the parser, not here.

What it does:

  - NFKC unicode normalization so fullwidth Latin / katakana ↔ ASCII /
    halfwidth match each other.
  - Bracket folding (「」, 〈〉, （）, 『』 …) → ASCII <>(){}[].
  - Quote folding (curly ’ ” → straight ' ").
  - Punctuation collapse (?!,.;:`~ → space) so a trailing period in
    the query doesn't break a substring match against a title.
  - Whitespace normalization (collapse runs to a single space).
  - ASCII letters are lowercased so "MUSASHI" boosts against
    "musashi" stored as a normalized title; Korean characters are
    LEFT INTACT (no morphological analysis).

What it does NOT do:

  - Tokenize Korean. The boost layer is a substring check; word-
    boundary handling on Korean would require a real morphology pass
    and isn't justified by Phase 2B's scope.
  - Generate query rewrites. The Phase 2B spec explicitly forbids LLM
    expansion or aggressive synonym maps.
  - Mutate the original string the embedder sees. The retriever
    embeds whatever the parser hands it; this normalizer's output is
    only consumed by the metadata boost scorer.

The result is a small frozen dataclass holding both the raw and the
normalized form so the failure-analysis report can show readers what
the matcher actually compared against.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


_BRACKET_PAIRS = {
    "「": "<", "」": ">",
    "『": "<", "』": ">",
    "〈": "<", "〉": ">",
    "《": "<", "》": ">",
    "（": "(", "）": ")",
    "｛": "{", "｝": "}",
    "［": "[", "］": "]",
}

_QUOTE_FOLDS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "′": "'", "″": '"',
}

_PUNCT_RE = re.compile(r"[?!,.;:`~]+")
_MULTI_WS_RE = re.compile(r"\s+")
_BRACKET_CONTENT_RE = re.compile(r"[<\(\[\{]([^<>\(\)\[\]\{\}]+)[>\)\]\}]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True)
class NormalizedQuery:
    """Container holding both the input and the canonical form.

    ``raw`` is preserved so reports can show the original query string;
    ``normalized`` is what the boost scorer actually compares against.
    ``title_tokens`` is a best-effort hint for the failure-analysis
    writer — never used by the scorer itself.
    """

    raw: str
    normalized: str
    title_tokens: Tuple[str, ...] = field(default_factory=tuple)


def fold_text(text: str) -> str:
    """Apply bracket / quote / punctuation folding without case change.

    Useful when downstream code wants to keep case (e.g. extracting
    proper-noun title tokens). For the boost-match form, prefer
    ``normalize_for_match`` which composes this with NFKC + lowercase.
    """
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch in _BRACKET_PAIRS:
            out_chars.append(_BRACKET_PAIRS[ch])
        elif ch in _QUOTE_FOLDS:
            out_chars.append(_QUOTE_FOLDS[ch])
        else:
            out_chars.append(ch)
    folded = "".join(out_chars)
    folded = _PUNCT_RE.sub(" ", folded)
    folded = _MULTI_WS_RE.sub(" ", folded).strip()
    return folded


def normalize_for_match(text: str) -> str:
    """Canonical lowercased form used for boost substring checks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    folded = fold_text(text)
    return folded.lower()


def extract_title_tokens(
    query: str,
    *,
    min_len: int = 2,
) -> Tuple[str, ...]:
    """Best-effort proper-noun extraction for failure-analysis hints.

    Pulls bracketed segments (the eval-query author often wraps title
    fragments in brackets like ``MUSASHI -GUN道-``) and Latin words
    >= 3 chars (model names, brand acronyms). Korean noun extraction
    is intentionally OUT OF SCOPE — see module docstring.
    """
    if not query:
        return ()
    folded = fold_text(query)
    tokens: List[str] = []
    seen: set = set()
    for match in _BRACKET_CONTENT_RE.findall(folded):
        token = match.strip()
        low = token.lower()
        if token and low not in seen and len(token) >= min_len:
            tokens.append(token)
            seen.add(low)
    for match in _LATIN_TOKEN_RE.findall(folded):
        if len(match) >= max(min_len, 3) and match.lower() not in seen:
            tokens.append(match)
            seen.add(match.lower())
    return tuple(tokens)


def normalize_query(
    query: str,
    *,
    extract_titles: bool = False,
) -> NormalizedQuery:
    """Compute the boost-friendly normalized form for a query string.

    ``extract_titles=True`` adds best-effort title-token hints; the
    scorer itself does not consume them — they're surfaced in reports
    so a reviewer can see what the partial-match logic likely picked up.
    """
    norm = normalize_for_match(query)
    titles = extract_title_tokens(query) if extract_titles else ()
    return NormalizedQuery(raw=query, normalized=norm, title_tokens=titles)


def normalize_iter(values: Iterable[str]) -> List[str]:
    """Convenience: ``normalize_for_match`` over an iterable, drop empties."""
    out: List[str] = []
    for v in values:
        n = normalize_for_match(v)
        if n:
            out.append(n)
    return out
