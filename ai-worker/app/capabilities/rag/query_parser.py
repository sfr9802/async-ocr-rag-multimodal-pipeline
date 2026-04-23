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

Three implementations ship here:

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
     returns ``intent='other'`` and ``rewrites=[]``.

  3. LlmQueryParser — phase 4. Wraps a shared ``LlmChatProvider`` and
     asks it for a JSON object with the same fields the regex parser
     produces plus a real intent classification and up to 3 alternate
     phrasings. LRU-caches per-query so repeated calls don't re-hit
     the LLM. Any provider failure or schema violation falls back to
     the regex parser, with ``parser_name='llm-fallback-regex'`` so
     downgrades are visible in metrics.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# LLM parser — phase 4. Uses a shared LlmChatProvider to classify intent,
# extract keywords, and propose alternate phrasings. On any provider failure
# (network, timeout, invalid JSON, schema violation) it degrades to the
# RegexQueryParser so a broken LLM never takes down retrieval — the
# downgrade is visible in ParsedQuery.parser_name for the metrics layer.
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = (
    "You extract retrieval intent. Respond ONLY with a JSON object.\n"
    "Fields:\n"
    "  normalized (string): the query rewritten into a clean, self-contained "
    "form suitable for dense retrieval. Strip filler words but keep the "
    "meaning.\n"
    "  keywords (array of strings, up to 10): the most salient content "
    "tokens. Lowercase; do not include stopwords.\n"
    "  intent (one of 'definition', 'comparison', 'procedure', 'factoid', "
    "'other'): what the user is actually trying to learn.\n"
    "  rewrites (array of strings, up to 3): alternate phrasings that would "
    "also retrieve relevant passages. Empty array is fine.\n"
    "  filters (object): metadata filters that narrow the search space. "
    "filters MAY ONLY contain these keys, and ONLY when the query makes "
    "the value unambiguous; otherwise OMIT the key entirely (do not guess):\n"
    "    domain   in {'anime', 'enterprise'}\n"
    "    category in {'hr', 'finance', 'it', 'product', 'legal'}\n"
    "    language in {'ko', 'en'}\n"
    "  An empty object {} is the correct answer when no signal is strong.\n"
    "Return nothing except the JSON object."
)

_LLM_SCHEMA_HINT = (
    '{"normalized": string, "keywords": [string, ...], '
    '"intent": "definition"|"comparison"|"procedure"|"factoid"|"other", '
    '"rewrites": [string, ...], '
    '"filters": {"domain"?: "anime"|"enterprise", '
    '"category"?: "hr"|"finance"|"it"|"product"|"legal", '
    '"language"?: "ko"|"en"}}'
)


# The LLM is the ONLY parser that emits filters. Mirrors the SQL
# whitelist on RagMetadataStore.doc_ids_matching: any value Claude
# proposes outside these enums is silently dropped at materialize
# time so a hallucinated "domain":"news" can't reach the SQL layer.
_FILTER_WHITELIST: dict[str, frozenset[str]] = {
    "domain":   frozenset({"anime", "enterprise"}),
    "category": frozenset({"hr", "finance", "it", "product", "legal"}),
    "language": frozenset({"ko", "en"}),
}

_LLM_KEYWORD_CAP = 10
_LLM_REWRITE_CAP = 3
_LLM_CACHE_DEFAULT = 256


class LlmQueryParser(QueryParserProvider):
    """LLM-backed parser that falls back to the regex parser on failure.

    Wraps a single ``LlmChatProvider`` instance and uses ``chat_json``
    with a fixed schema. Repeated parses of the same query are served
    from an LRU cache so a hot query doesn't re-hit the LLM — the cache
    size defaults to 256 entries per parser instance.

    Failure modes that fall back to regex instead of raising:
      * LlmChatError from the underlying provider (network, timeout,
        invalid JSON, empty response).
      * Schema violation at our layer (missing field, wrong type, intent
        not in the 5-enum, keywords/rewrites not list-of-str).

    The fallback's ``parser_name`` is ``"llm-fallback-regex"`` — distinct
    from both ``"llm"`` (clean LLM path) and ``"regex"`` (configured
    regex parser) so the metrics layer can measure LLM downgrade rate
    separately from baseline regex usage.
    """

    def __init__(
        self,
        chat: Any,
        *,
        cache_size: int = _LLM_CACHE_DEFAULT,
    ) -> None:
        from app.clients.llm_chat import LlmChatProvider  # local import - avoids cycles

        if not isinstance(chat, LlmChatProvider):
            raise TypeError(
                "LlmQueryParser requires an LlmChatProvider instance; "
                f"got {type(chat).__name__}"
            )
        self._chat = chat
        self._regex_fallback = RegexQueryParser()
        # Bind the LRU cache to the instance so different parser
        # instances don't leak entries across each other and so test
        # setup/teardown can discard the cache by dropping the instance.
        self._parse_cached = lru_cache(maxsize=cache_size)(self._parse_uncached)

    @property
    def name(self) -> str:
        return "llm"

    def parse(self, query: str) -> ParsedQuery:
        return self._parse_cached(query)

    # ------------------------------------------------------------------

    def _parse_uncached(self, query: str) -> ParsedQuery:
        from app.clients.llm_chat import ChatMessage, LlmChatError

        messages = [
            ChatMessage(role="system", content=_LLM_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Query: {query}\n\n"
                    "Extract normalized + keywords + intent + rewrites + filters."
                ),
            ),
        ]
        enable_thinking = bool(self._chat.capabilities.get("thinking"))

        try:
            data = self._chat.chat_json(
                messages,
                schema_hint=_LLM_SCHEMA_HINT,
                enable_thinking=enable_thinking,
            )
        except LlmChatError as ex:
            log.warning(
                "LlmQueryParser: provider failure, falling back to regex (%s)", ex,
            )
            return self._fallback(query)

        try:
            return self._materialize(query, data)
        except ValueError as ex:
            log.warning(
                "LlmQueryParser: schema violation, falling back to regex (%s)", ex,
            )
            return self._fallback(query)

    def _fallback(self, query: str) -> ParsedQuery:
        base = self._regex_fallback.parse(query)
        # Re-wrap so parser_name records the downgrade for metrics.
        return ParsedQuery(
            original=base.original,
            normalized=base.normalized,
            keywords=base.keywords,
            intent=base.intent,
            rewrites=base.rewrites,
            filters=base.filters,
            parser_name="llm-fallback-regex",
        )

    @staticmethod
    def _materialize(query: str, data: dict) -> ParsedQuery:
        if not isinstance(data, dict):
            raise ValueError(
                f"LLM response must be a JSON object; got {type(data).__name__}"
            )

        raw_normalized = data.get("normalized", query)
        if not isinstance(raw_normalized, str):
            raise ValueError("'normalized' must be a string")
        normalized = raw_normalized.strip() or query

        raw_keywords = data.get("keywords", [])
        if not isinstance(raw_keywords, list):
            raise ValueError("'keywords' must be a list of strings")
        keywords: list[str] = []
        for kw in raw_keywords:
            if not isinstance(kw, str):
                raise ValueError("'keywords' entries must be strings")
            kw = kw.strip()
            if kw:
                keywords.append(kw)
            if len(keywords) >= _LLM_KEYWORD_CAP:
                break

        intent = data.get("intent", "other")
        if not isinstance(intent, str):
            raise ValueError("'intent' must be a string")
        intent = intent.strip().lower()
        if intent not in _VALID_INTENTS:
            raise ValueError(
                f"'intent' must be one of {sorted(_VALID_INTENTS)}; got {intent!r}"
            )

        raw_rewrites = data.get("rewrites", [])
        if not isinstance(raw_rewrites, list):
            raise ValueError("'rewrites' must be a list of strings")
        rewrites: list[str] = []
        for rw in raw_rewrites:
            if not isinstance(rw, str):
                raise ValueError("'rewrites' entries must be strings")
            rw = rw.strip()
            if rw:
                rewrites.append(rw)
            if len(rewrites) >= _LLM_REWRITE_CAP:
                break

        # Filters are optional. We accept ONLY keys/values that match the
        # whitelist — any out-of-vocabulary value is silently dropped
        # rather than falling back, since the parser already proved it
        # could produce structured JSON. A whole-payload fallback would
        # over-correct on a single noisy field.
        raw_filters = data.get("filters", {}) or {}
        if not isinstance(raw_filters, dict):
            raise ValueError("'filters' must be an object (or omitted).")
        filters: dict[str, str] = {}
        for key, value in raw_filters.items():
            if key not in _FILTER_WHITELIST:
                continue
            if not isinstance(value, str):
                continue
            value = value.strip().lower()
            if value in _FILTER_WHITELIST[key]:
                filters[key] = value

        return ParsedQuery(
            original=query,
            normalized=normalized,
            keywords=keywords,
            intent=intent,  # type: ignore[arg-type]
            rewrites=rewrites,
            filters=filters,
            parser_name="llm",
        )
