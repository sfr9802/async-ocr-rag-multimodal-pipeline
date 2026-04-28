"""Tests for the Phase 2B query normalizer.

The normalizer is intentionally narrow — bracket / quote / punctuation
folding, NFKC, lowercase ASCII. These tests pin that contract:

  - common bracket variants fold to ASCII
  - punctuation collapses to whitespace
  - Korean code points are preserved unchanged
  - Latin letters lowercase but Korean does not
  - NFKC widens / narrows fullwidth alphanumerics
  - title-token extraction picks up bracketed segments and Latin words
  - the dataclass round-trips raw alongside normalized

The boost scorer relies on substring matching against the normalized
form, so any drift in this layer would silently break boost recall.
"""

from __future__ import annotations

import pytest

from eval.harness.query_normalizer import (
    NormalizedQuery,
    extract_title_tokens,
    fold_text,
    normalize_for_match,
    normalize_iter,
    normalize_query,
)


class TestFoldText:
    def test_corner_brackets_fold(self):
        assert fold_text("「nausea」") == "<nausea>"
        assert fold_text("『코어』") == "<코어>"

    def test_angle_brackets_fold(self):
        assert fold_text("〈제목〉") == "<제목>"
        assert fold_text("《book》") == "<book>"

    def test_fullwidth_parens_fold(self):
        assert fold_text("（주의）") == "(주의)"

    def test_curly_quotes_fold(self):
        # The single curly quote pair shouldn't introduce extra spaces.
        assert fold_text("그의 ‘답’") == "그의 '답'"
        assert fold_text("a “b” c") == 'a "b" c'

    def test_punctuation_collapses_to_space(self):
        # Period, question mark, comma all become whitespace; runs collapse.
        out = fold_text("Hello, world?!")
        assert out == "Hello world"

    def test_multiple_whitespace_collapses(self):
        assert fold_text("a   b\t\tc\n d") == "a b c d"

    def test_empty_input(self):
        assert fold_text("") == ""
        assert fold_text(None) == ""  # type: ignore[arg-type]


class TestNormalizeForMatch:
    def test_lowercases_ascii_only(self):
        # "MUSASHI" → "musashi" but Korean stays as-is.
        out = normalize_for_match("MUSASHI 검사")
        assert out == "musashi 검사"

    def test_korean_unchanged(self):
        out = normalize_for_match("템플의 주요 주제")
        assert out == "템플의 주요 주제"

    def test_nfkc_widens_fullwidth(self):
        # Fullwidth ABC should normalize to halfwidth then lowercase.
        out = normalize_for_match("ＡＢＣ")
        assert out == "abc"

    def test_combines_all_steps(self):
        # A query mixing brackets, punctuation, casing, and Korean.
        out = normalize_for_match("「MUSASHI -GUN道-」, what?")
        # "「」" → "<>", comma + question mark → spaces
        assert out == "<musashi -gun道-> what"

    def test_empty_input(self):
        assert normalize_for_match("") == ""


class TestExtractTitleTokens:
    def test_picks_bracketed_segments(self):
        tokens = extract_title_tokens("〈my title〉 and stuff")
        assert "my title" in tokens

    def test_picks_latin_words(self):
        tokens = extract_title_tokens("the MUSASHI doc")
        # >= 3 chars, original case preserved
        assert "MUSASHI" in tokens
        assert "doc" in tokens  # 3 chars meets the floor
        # 'a' (single char) would be filtered; 'the' clears the floor.
        assert "the" in tokens

    def test_filters_latin_below_min_len(self):
        # min_len caps below 3 still respect the Latin-token floor of 3,
        # so an "ab" token never appears.
        tokens = extract_title_tokens("ab cd MUSASHI", min_len=2)
        assert "ab" not in tokens
        assert "cd" not in tokens
        assert "MUSASHI" in tokens

    def test_dedupes_by_lowercase(self):
        # Both ABC variants should collapse; "and" remains as its own
        # 3-char Latin token. Result: ABC + and == 2 unique tokens.
        tokens = extract_title_tokens("ABC and abc and ABC")
        assert len(tokens) == 2
        assert "ABC" in tokens
        assert "and" in tokens

    def test_empty_returns_empty_tuple(self):
        assert extract_title_tokens("") == ()
        assert extract_title_tokens("그") == ()  # 1-char Korean below min_len


class TestNormalizeQuery:
    def test_returns_dataclass_with_raw_preserved(self):
        nq = normalize_query("Hello,  world?")
        assert isinstance(nq, NormalizedQuery)
        assert nq.raw == "Hello,  world?"
        assert nq.normalized == "hello world"
        assert nq.title_tokens == ()

    def test_extract_titles_flag(self):
        nq = normalize_query("the MUSASHI doc", extract_titles=True)
        assert "MUSASHI" in nq.title_tokens

    def test_extract_titles_off_returns_empty(self):
        nq = normalize_query("the MUSASHI doc", extract_titles=False)
        assert nq.title_tokens == ()


class TestNormalizeIter:
    def test_filters_empty(self):
        out = normalize_iter(["abc", "", "  ", "DEF"])
        assert out == ["abc", "def"]

    def test_handles_empty_input(self):
        assert normalize_iter([]) == []
