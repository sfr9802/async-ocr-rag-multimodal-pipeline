"""Unit tests for ``eval.harness.lexical_overlap``."""

from __future__ import annotations

import pytest

from eval.harness.lexical_overlap import (
    BM25_HIGH,
    BM25_MEDIUM,
    CHUNK_HIGH,
    CHUNK_MEDIUM,
    TITLE_HIGH,
    TITLE_MEDIUM,
    char_ngrams,
    classify_overlap_risk,
    compute_overlap,
    containment,
    jaccard,
    normalize_text,
)


class TestNormalizeText:
    def test_empty_returns_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""

    def test_strips_special_chars(self):
        assert normalize_text("스즈메♪의 문단속!") == "스즈메 의 문단속"

    def test_collapses_whitespace(self):
        assert normalize_text("a   b\tc") == "a b c"

    def test_lowercase_ascii(self):
        assert normalize_text("ARIA The ORIGINATION") == "aria the origination"

    def test_keeps_hangul_and_digits(self):
        assert normalize_text("원피스 1~516화") == "원피스 1 516화"

    def test_nfkc_normalization(self):
        # full-width "Ａ" → ascii "a"
        assert normalize_text("Ａbc") == "abc"


class TestCharNgrams:
    def test_n_must_be_positive(self):
        with pytest.raises(ValueError):
            char_ngrams("abc", 0)

    def test_short_string_returns_whole(self):
        # When normalized text length < n, return frozenset({whole})
        assert char_ngrams("a", 3) == frozenset({"a"})

    def test_2gram_sliding(self):
        ng = char_ngrams("abc", 2)
        assert ng == frozenset({"ab", "bc"})

    def test_4gram_sliding(self):
        ng = char_ngrams("abcde", 4)
        assert ng == frozenset({"abcd", "bcde"})

    def test_deterministic(self):
        # Same input → same output, byte-for-byte.
        a = char_ngrams("원피스 줄거리", 2)
        b = char_ngrams("원피스 줄거리", 2)
        assert a == b


class TestJaccard:
    def test_empty_sets_zero(self):
        assert jaccard(frozenset(), frozenset()) == 0.0

    def test_identical_sets_one(self):
        s = frozenset({"a", "b"})
        assert jaccard(s, s) == 1.0

    def test_disjoint_zero(self):
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        assert jaccard(a, b) == 0.0

    def test_partial(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        # |∩|=2, |∪|=4 → 0.5
        assert jaccard(a, b) == 0.5


class TestContainment:
    def test_empty_query_zero(self):
        assert containment(frozenset(), frozenset({"a"})) == 0.0

    def test_full_containment_one(self):
        q = frozenset({"a", "b"})
        t = frozenset({"a", "b", "c", "d"})
        assert containment(q, t) == 1.0

    def test_partial(self):
        q = frozenset({"a", "b", "c"})
        t = frozenset({"a"})
        # 1 / 3
        assert containment(q, t) == pytest.approx(1.0 / 3.0)


class TestClassifyOverlapRisk:
    def test_not_in_corpus_returns_na(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=None,
            chunk_char4_containment=None,
            bm25_first_rank=None,
            is_not_in_corpus=True,
        )
        assert risk == "not_applicable"

    def test_high_via_title(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=TITLE_HIGH,
            chunk_char4_containment=0.0,
            bm25_first_rank=None,
        )
        assert risk == "high"

    def test_high_via_chunk(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=0.0,
            chunk_char4_containment=CHUNK_HIGH,
            bm25_first_rank=None,
        )
        assert risk == "high"

    def test_high_via_bm25(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=0.0,
            chunk_char4_containment=0.0,
            bm25_first_rank=BM25_HIGH,
        )
        assert risk == "high"

    def test_medium_via_title(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=TITLE_MEDIUM,
            chunk_char4_containment=0.0,
            bm25_first_rank=None,
        )
        assert risk == "medium"

    def test_low_when_all_below(self):
        risk = classify_overlap_risk(
            title_char2_jaccard=0.1,
            chunk_char4_containment=0.05,
            bm25_first_rank=50,
        )
        assert risk == "low"

    def test_thresholds_inclusive(self):
        # Exactly at HIGH should classify as high.
        assert classify_overlap_risk(
            title_char2_jaccard=TITLE_HIGH,
            chunk_char4_containment=None,
            bm25_first_rank=None,
        ) == "high"
        # One tick below should be medium.
        assert classify_overlap_risk(
            title_char2_jaccard=TITLE_HIGH - 0.001,
            chunk_char4_containment=None,
            bm25_first_rank=None,
        ) in ("medium", "low")  # depends on whether above MEDIUM


class TestComputeOverlap:
    def test_not_in_corpus_all_null(self):
        out = compute_overlap(
            "재미있는 애니 추천",
            expected_title=None,
            expected_section_path=None,
            target_text=None,
        )
        assert out["title_char2_jaccard"] is None
        assert out["section_char2_jaccard"] is None
        assert out["chunk_char4_containment"] is None
        assert out["bm25_expected_page_first_rank"] is None
        assert out["overlap_risk"] == "not_applicable"

    def test_full_with_target(self):
        out = compute_overlap(
            "원피스 줄거리",
            expected_title="원피스(애니메이션)",
            expected_section_path=["줄거리"],
            target_text="원피스는 일본 만화 작품으로 ...",
        )
        assert out["title_char2_jaccard"] is not None
        assert out["section_char2_jaccard"] is not None
        assert out["chunk_char4_containment"] is not None
        assert out["overlap_risk"] in ("low", "medium", "high")

    def test_deterministic(self):
        a = compute_overlap(
            "코난 등장인물",
            expected_title="명탐정 코난 / 등장인물",
            expected_section_path=["등장인물"],
            target_text="코난 등장인물 리스트",
        )
        b = compute_overlap(
            "코난 등장인물",
            expected_title="명탐정 코난 / 등장인물",
            expected_section_path=["등장인물"],
            target_text="코난 등장인물 리스트",
        )
        assert a == b
