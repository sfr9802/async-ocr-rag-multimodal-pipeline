"""QueryParserProvider tests.

Three scenario groups, all fully offline:

  1. NoOpQueryParser is a passthrough: ``normalized`` equals the raw
     query, every list/dict field is empty, intent is 'other'. This is
     the "env unset is a no-op" contract the retriever's RRF guard
     depends on.

  2. RegexQueryParser extracts keywords deterministically:
       - lowercases, dedupes while preserving first-seen order
       - drops len<2 tokens and the hardcoded KR+EN stopword list
       - handles Korean+English mixed queries
       - strips unicode curly quotes + backticks before tokenizing
       - caps the keyword list at 10

  3. ParsedQuery is frozen (mutation raises), supports equality, and
     rejects intent values outside the Literal whitelist at construction
     time — a runtime guard is needed because ``Literal`` is purely a
     type hint in Python.

Zero new deps; everything runs under the stdlib + the parser module.
"""

from __future__ import annotations

import json

import pytest

from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)


# ---------------------------------------------------------------------------
# 1. NoOpQueryParser passthrough.
# ---------------------------------------------------------------------------


def test_noop_parser_name():
    parser = NoOpQueryParser()
    assert parser.name == "noop"


def test_noop_parser_passes_query_through_unchanged():
    parser = NoOpQueryParser()
    raw = "  FooBar   with   weird   whitespace  "
    parsed = parser.parse(raw)

    assert parsed.original == raw
    # NoOp explicitly does NOT normalize — retriever RRF guard + the
    # "zero behaviour change" contract both depend on this.
    assert parsed.normalized == raw
    assert parsed.keywords == []
    assert parsed.intent == "other"
    assert parsed.rewrites == []
    assert parsed.filters == {}
    assert parsed.parser_name == "noop"


def test_noop_parser_handles_empty_query():
    parser = NoOpQueryParser()
    parsed = parser.parse("")
    assert parsed.original == ""
    assert parsed.normalized == ""
    assert parsed.keywords == []


# ---------------------------------------------------------------------------
# 2. RegexQueryParser deterministic tokenization.
# ---------------------------------------------------------------------------


def test_regex_parser_name():
    parser = RegexQueryParser()
    assert parser.name == "regex"
    parsed = parser.parse("hello")
    assert parsed.parser_name == "regex"


def test_regex_parser_dedupes_keywords_preserving_first_seen_order():
    parser = RegexQueryParser()
    parsed = parser.parse("faiss index faiss tuning faiss")

    # 'faiss' appears three times; must appear once and at position 0.
    assert parsed.keywords == ["faiss", "index", "tuning"]


def test_regex_parser_drops_english_stopwords():
    parser = RegexQueryParser()
    parsed = parser.parse("What is the role of a reranker in retrieval")

    # 'is', 'the', 'of', 'a', 'in' are dropped. 'what' is kept because
    # it isn't on our minimal stopword list — the list is deliberately
    # small in phase 3.
    assert "is" not in parsed.keywords
    assert "the" not in parsed.keywords
    assert "of" not in parsed.keywords
    assert "a" not in parsed.keywords
    assert "in" not in parsed.keywords
    assert "reranker" in parsed.keywords
    assert "retrieval" in parsed.keywords


def test_regex_parser_mixes_kr_en_tokens():
    """Korean/English mixed queries produce a mixed keyword list.

    The regex parser does NOT do Korean morphological analysis — if a
    query happens to have whitespace between a word and its particle
    (unnatural but seen in UI-typed text), the particle will be a
    standalone len=1 token and get filtered out by the length guard
    before the stopword list even matters. The more common case is
    that particles stay attached to their head noun as a compound
    token, which the parser keeps intact.
    """
    parser = RegexQueryParser()
    parsed = parser.parse("RAG reranker 한국어 쿼리 분석")

    # English word lowercased, Korean words kept verbatim, deduped.
    assert "rag" in parsed.keywords
    assert "reranker" in parsed.keywords
    assert "한국어" in parsed.keywords
    assert "쿼리" in parsed.keywords
    assert "분석" in parsed.keywords


def test_regex_parser_drops_standalone_korean_stopword_tokens():
    """When a Korean particle happens to appear as a standalone token
    (whitespace on both sides — unnatural, but the parser must behave
    if it does), the stopword list drops it. The length guard would
    also drop it; the stopword list is a belt-and-suspenders check."""
    parser = RegexQueryParser()
    parsed = parser.parse("검색 의 품질 과 성능")

    # Standalone particles must not appear.
    assert "의" not in parsed.keywords
    assert "과" not in parsed.keywords
    # Real content words survive.
    assert "검색" in parsed.keywords
    assert "품질" in parsed.keywords
    assert "성능" in parsed.keywords


def test_regex_parser_drops_len1_tokens():
    parser = RegexQueryParser()
    parsed = parser.parse("a b cc ddd x y zzz")

    # Every single-character token must disappear, regardless of stopword
    # membership.
    for kw in parsed.keywords:
        assert len(kw) >= 2
    assert set(parsed.keywords) == {"cc", "ddd", "zzz"}


def test_regex_parser_strips_unicode_quotes_and_backticks():
    parser = RegexQueryParser()
    # Left+right double+single quotes (U+201C..U+2019), angle quotes,
    # ASCII backticks. Everything inside survives as normal tokens.
    raw = "\u201cvector\u201d \u2018database\u2019 \u00abfaiss\u00bb `index`"
    parsed = parser.parse(raw)

    # Quotes are stripped out of normalized, leaving whitespace-separated words.
    assert "\u201c" not in parsed.normalized
    assert "`" not in parsed.normalized
    assert set(parsed.keywords) == {"vector", "database", "faiss", "index"}


def test_regex_parser_collapses_whitespace_in_normalized():
    parser = RegexQueryParser()
    parsed = parser.parse("  vector\n\n  database\t index  ")
    assert parsed.normalized == "vector database index"


def test_regex_parser_caps_keywords_at_ten():
    parser = RegexQueryParser()
    words = [f"word{i}" for i in range(20)]
    parsed = parser.parse(" ".join(words))

    assert len(parsed.keywords) == 10
    # Must be the FIRST ten, not a random slice — the cap is applied
    # after dedup-while-preserving-order.
    assert parsed.keywords == [f"word{i}" for i in range(10)]


def test_regex_parser_empty_query_returns_no_keywords():
    parser = RegexQueryParser()
    parsed = parser.parse("")
    assert parsed.normalized == ""
    assert parsed.keywords == []


def test_regex_parser_rewrites_and_filters_are_empty_in_phase3():
    """Phase 3 contract: offline parser never emits rewrites/filters.
    The RRF path on the Retriever is dead while this is true."""
    parser = RegexQueryParser()
    parsed = parser.parse("some long real-world-ish korean 한국어 query")
    assert parsed.rewrites == []
    assert parsed.filters == {}
    assert parsed.intent == "other"


# ---------------------------------------------------------------------------
# 3. ParsedQuery value-object contracts.
# ---------------------------------------------------------------------------


def test_parsed_query_is_frozen():
    pq = ParsedQuery(
        original="q",
        normalized="q",
        keywords=[],
        intent="other",
        rewrites=[],
        filters={},
        parser_name="noop",
    )
    with pytest.raises(Exception):
        # frozen=True -> FrozenInstanceError (subclass of AttributeError
        # on CPython); the exact class is a Python-version detail, so we
        # just assert that assignment is rejected.
        pq.original = "hacked"  # type: ignore[misc]


def test_parsed_query_equality_by_value():
    a = ParsedQuery(
        original="q", normalized="q", keywords=["x"], intent="other",
        rewrites=[], filters={}, parser_name="regex",
    )
    b = ParsedQuery(
        original="q", normalized="q", keywords=["x"], intent="other",
        rewrites=[], filters={}, parser_name="regex",
    )
    c = ParsedQuery(
        original="q", normalized="q", keywords=["y"], intent="other",
        rewrites=[], filters={}, parser_name="regex",
    )
    assert a == b
    assert a != c


def test_parsed_query_rejects_unknown_intent():
    with pytest.raises(ValueError):
        ParsedQuery(
            original="q", normalized="q", keywords=[], intent="summarize",
            rewrites=[], filters={}, parser_name="noop",
        )


def test_parsed_query_to_dict_is_json_serializable():
    pq = ParsedQuery(
        original="q", normalized="q", keywords=["x", "y"],
        intent="definition", rewrites=["q prime"],
        filters={"lang": "ko"}, parser_name="regex",
    )
    as_dict = pq.to_dict()
    # json.dumps round-trips without a custom encoder — that's the
    # acceptance bar since the RetrievalReport must be JSON-serializable.
    rendered = json.dumps(as_dict, ensure_ascii=False)
    assert '"parserName": "regex"' in rendered
    parsed_back = json.loads(rendered)
    assert parsed_back["keywords"] == ["x", "y"]
    assert parsed_back["intent"] == "definition"
    assert parsed_back["rewrites"] == ["q prime"]
    assert parsed_back["filters"] == {"lang": "ko"}


def test_query_parser_provider_subclass_contract():
    """Each provider exposes name + parse. This is the smallest check
    that guards against future refactors accidentally breaking the
    seam the registry depends on."""
    assert issubclass(NoOpQueryParser, QueryParserProvider)
    assert issubclass(RegexQueryParser, QueryParserProvider)
