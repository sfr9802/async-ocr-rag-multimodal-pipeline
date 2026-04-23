"""QueryParserProvider contract + offline implementations.

A query parser is a pre-retrieval stage that converts a raw user query
into a structured ``ParsedQuery`` the Retriever can reason over: a
normalized form the embedder actually sees, a small list of keywords
the downstream evaluator can diff against gold references, an intent
tag (reserved for future routing), and an optional list of alternate
phrasings. When ``rewrites`` is non-empty the Retriever runs one FAISS
search per rewrite and merges them via Reciprocal Rank Fusion before
reranking, which is the multi-query path this seam was built to unlock.

This module is a new provider seam parallel to EmbeddingProvider,
RerankerProvider, and GenerationProvider. It is NOT a method on
Retriever: the Retriever composes a parser the same way it composes
an embedder, so the registry can swap between NoOpQueryParser,
RegexQueryParser, and (later, in phase 4) an LLM-backed parser
without the Retriever itself ever growing parser-specific code.

Two implementations ship here:

  1. NoOpQueryParser — passthrough default. ``normalized`` equals the
     raw query; everything else is empty. Single-query behaviour is
     bit-for-bit identical to the pre-parser path: the Retriever
     guards the RRF merge with ``if parsed.rewrites:`` so an empty
     rewrites list means zero behaviour change.

  2. RegexQueryParser — offline, zero-dependency tokenizer. Strips
     unicode quotes and backticks, collapses whitespace, lowercases,
     tokenizes on whitespace + Korean punctuation, drops len<2 tokens
     and a small hardcoded KR+EN stopword list, deduplicates while
     preserving first-seen order, and caps at 10 keywords. Always
     returns ``intent='other'`` and ``rewrites=[]`` because this
     phase ships keyword extraction only — intent classification and
     query rewriting arrive with the LLM-backed parser in phase 4.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


# Intent tag reserved for future routing (definition-vs-procedure-vs-
# factoid prompting, stopword tuning, etc.). Phase 3 always emits
# 'other' because neither offline parser can infer intent reliably.
Intent = Literal["definition", "comparison", "procedure", "factoid", "other"]


_VALID_INTENTS = frozenset(
    {"definition", "comparison", "procedure", "factoid", "other"}
)


@dataclass(frozen=True)
class ParsedQuery:
    """Structured representation of a retrieval query.

    Frozen so downstream code (RetrievalReport, artifact payloads) can
    treat it as a value object without defensive copies. The fields are
    all JSON-serializable primitives so the whole thing round-trips
    through ``json.dumps`` without a custom encoder.
    """

    original: str
    normalized: str
    keywords: list[str]
    intent: Intent
    rewrites: list[str]
    filters: dict[str, str]
    parser_name: str

    def __post_init__(self) -> None:
        # Intent is declared as Literal[...] in the type signature, but
        # Python doesn't enforce that at runtime — a stray "summarize"
        # would silently flow into artifacts and break downstream
        # consumers. Fail loud here instead.
        if self.intent not in _VALID_INTENTS:
            raise ValueError(
                f"ParsedQuery.intent must be one of {sorted(_VALID_INTENTS)}; "
                f"got {self.intent!r}"
            )

    def to_dict(self) -> dict:
        """JSON-serializable dict for RetrievalReport / artifact payloads."""
        return {
            "original": self.original,
            "normalized": self.normalized,
            "keywords": list(self.keywords),
            "intent": self.intent,
            "rewrites": list(self.rewrites),
            "filters": dict(self.filters),
            "parserName": self.parser_name,
        }


class QueryParserProvider(ABC):
    """Converts a raw query string into a ``ParsedQuery``."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def parse(self, query: str) -> ParsedQuery:
        ...


# ---------------------------------------------------------------------------
# NoOp parser — default, zero behaviour change
# ---------------------------------------------------------------------------


class NoOpQueryParser(QueryParserProvider):
    """Identity parser.

    ``normalized`` is the raw query; ``keywords`` / ``rewrites`` /
    ``filters`` are empty. The Retriever's RRF merge guards on
    ``if parsed.rewrites:`` so the single-query path remains bit-for-
    bit identical to the pre-parser behaviour when this parser is in
    play — which is what CI / env-unset must reproduce.
    """

    @property
    def name(self) -> str:
        return "noop"

    def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(
            original=query,
            normalized=query,
            keywords=[],
            intent="other",
            rewrites=[],
            filters={},
            parser_name="noop",
        )


# ---------------------------------------------------------------------------
# Regex parser — offline tokenizer
# ---------------------------------------------------------------------------


# Unicode curly quotes + backtick + ASCII single/double quotes. The raw
# character class is easier to audit than an \u escape soup, and all of
# these characters are pure punctuation in the queries we care about —
# stripping them never loses semantic content.
_QUOTE_CHARS = "\"'`\u201C\u201D\u2018\u2019\u00AB\u00BB"
_QUOTE_STRIP_RE = re.compile(f"[{re.escape(_QUOTE_CHARS)}]")

# Split on ASCII whitespace, ASCII punctuation, and the Korean punctuation
# commonly typed in queries. Emoji / CJK-extension-B is NOT a splitter —
# those stay inside tokens. We keep the class explicit (not \W) because
# \W eats Korean word characters in Python's re flavour depending on the
# UNICODE flag, which would be a subtle footgun to leave here.
_TOKEN_SPLIT_RE = re.compile(
    r"[\s,\.\?\!;:/\\(){}\[\]<>\-\u2013\u2014"
    r"\u3001\u3002\uFF01\uFF1F\uFF0C\uFF1A\uFF1B]+"
)

_WHITESPACE_RE = re.compile(r"\s+")

_KEYWORD_CAP = 10
_MIN_KEYWORD_LEN = 2

# Hardcoded KR+EN stopwords. Kept small and explicit — a full stopword
# list belongs in the LLM-backed parser (phase 4), not in a regex
# fallback. The Korean particles cover the most common josa that
# survive whitespace tokenization; the English entries cover the
# half-dozen function words most likely to leak through from mixed-
# language queries. Everything else is the caller's problem.
_STOPWORDS_KR = frozenset({
    "이", "가", "은", "는", "을", "를", "의", "에", "와", "과",
})
_STOPWORDS_EN = frozenset({
    "of", "the", "a", "to", "in", "on", "for", "is",
})
_STOPWORDS = _STOPWORDS_KR | _STOPWORDS_EN


class RegexQueryParser(QueryParserProvider):
    """Deterministic offline parser.

    Produces a normalized string, up to 10 deduplicated keywords, and
    empty rewrites/filters. Intent is always ``'other'`` — distinguishing
    definitions from procedures reliably needs an LLM and arrives in
    phase 4. Zero external dependencies.
    """

    @property
    def name(self) -> str:
        return "regex"

    def parse(self, query: str) -> ParsedQuery:
        normalized = self._normalize(query)
        keywords = self._extract_keywords(normalized)
        return ParsedQuery(
            original=query,
            normalized=normalized,
            keywords=keywords,
            intent="other",
            rewrites=[],
            filters={},
            parser_name="regex",
        )

    @staticmethod
    def _normalize(query: str) -> str:
        stripped = _QUOTE_STRIP_RE.sub(" ", query or "")
        collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
        return collapsed

    @staticmethod
    def _extract_keywords(normalized: str) -> list[str]:
        if not normalized:
            return []
        raw_tokens = _TOKEN_SPLIT_RE.split(normalized)
        seen: set[str] = set()
        keywords: list[str] = []
        for raw in raw_tokens:
            token = raw.strip().lower()
            if len(token) < _MIN_KEYWORD_LEN:
                continue
            if token in _STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= _KEYWORD_CAP:
                break
        return keywords
