"""Tests for the Phase 2B candidate miss bucket classifier.

The classifier reads existing retrieval-eval row dicts (no live
retriever) and partitions every miss into one of eight buckets at
multiple top-K cutoffs. These tests pin the priority order and
the per-bucket heuristics so future tuning does not silently shuffle
which bucket a query lands in.

For every bucket below corpus_missing in priority, the fixture must
keep the expected doc INSIDE the deep_pool but OUTSIDE the top-K
under test — otherwise corpus_missing fires first and masks the
bucket-of-interest.
"""

from __future__ import annotations

import pytest

from eval.harness.candidate_miss_analysis import (
    BUCKET_ALIAS_SYNONYM,
    BUCKET_AMBIGUOUS_LABEL,
    BUCKET_CHARACTER_MISMATCH,
    BUCKET_CORPUS_MISSING,
    BUCKET_LEXICAL_MISMATCH,
    BUCKET_OVERLY_BROAD,
    BUCKET_SECTION_MISMATCH,
    BUCKET_TITLE_MISMATCH,
    candidate_miss_report_to_dict,
    classify_candidate_misses,
    render_candidate_miss_markdown,
)


def _row(*, rid, query, expected, retrieved, keywords=(), atype="x", diff="medium"):
    return {
        "id": rid,
        "query": query,
        "expected_doc_ids": list(expected),
        "expected_section_keywords": list(keywords),
        "retrieved_doc_ids": list(retrieved),
        "answer_type": atype,
        "difficulty": diff,
    }


def _dump(*, qid, rank, doc_id, section="", text="", matched=()):
    return {
        "query_id": qid,
        "rank": rank,
        "doc_id": doc_id,
        "section_path": section,
        "chunk_preview": text,
        "matched_expected_keyword": list(matched),
    }


def _padded_retrieve(noise_count, *, gold_at=None, gold="gold"):
    """Build a retrieved list with `noise_count` distractors and gold
    optionally placed at a specific 1-based rank within the deep pool.
    """
    out = [f"noise-{i}" for i in range(noise_count)]
    if gold_at is not None:
        # 1-based: insert gold so that retrieved[gold_at - 1] == gold.
        if gold_at <= 0 or gold_at > len(out) + 1:
            out.append(gold)
        else:
            out.insert(gold_at - 1, gold)
    return out


class TestCorpusMissingPriority:
    def test_expected_not_in_deep_pool(self):
        # Gold not in retrieved list at all → corpus_missing.
        row = _row(
            rid="q1",
            query="템플의 무언가에 대해 알려줘 정말 길게",
            expected=["gold"],
            retrieved=["a", "b", "c", "d"],
        )
        report = classify_candidate_misses(
            [row], top_ks=(5,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_CORPUS_MISSING


class TestTitleMismatch:
    def test_query_title_overlap_with_sibling_in_top_k(self):
        # gold sits at deep_pool position 8 (outside top-3 but inside
        # deep_k=10). Query has the title token "템플" exactly so the
        # token-level overlap fires; sibling docs occupy top-3 → bucket
        # is title_mismatch.
        row = _row(
            rid="q1",
            query="템플 다른 무언가 추가 단어 더",
            expected=["gold"],
            retrieved=_padded_retrieve(7, gold_at=8),
        )
        meta = {"gold": {"title": "템플 시즌2"}}
        report = classify_candidate_misses(
            [row], doc_metadata=meta, top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_TITLE_MISMATCH


class TestCharacterMismatch:
    def test_proper_noun_keywords_unmatched(self):
        # Title doesn't overlap query (no 템플 etc.); keywords are
        # a Latin proper noun ("MUSASHI") and the matched_expected_keyword
        # list is empty.
        row = _row(
            rid="q2",
            query="검사 주인공이 누구야 정말 궁금해",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["MUSASHI"],
        )
        meta = {"gold": {"title": "전혀 다른 제목"}}  # hangul-only title
        dump = [
            _dump(qid="q2", rank=1, doc_id="noise-0", text="anything"),
        ]
        # Since the title is hangul-only AND the keyword carries Latin,
        # alias_or_synonym would fire first if we allowed it. To pin
        # character_mismatch, we use a hangul keyword instead.
        row3 = _row(
            rid="q3",
            query="검사 주인공이 누구야 정말 궁금해",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["라떼"],  # hangul proper-noun-ish (kana check fails;
            # _looks_like_proper_noun checks for katakana — let's use kana.
        )
        # Actually our _looks_like_proper_noun matches LATIN or KATAKANA.
        # Hangul "라떼" won't pass that check. So we need a Latin or
        # katakana keyword without the alias_or_synonym path firing.
        # alias_or_synonym fires when keyword has Latin AND title has no
        # Latin AND title has hangul. So we set the title to ALSO include
        # Latin to bypass alias_or_synonym.
        row4 = _row(
            rid="q4",
            query="검사 주인공이 누구야 정말 궁금해",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["MUSASHI"],
        )
        meta4 = {"gold": {"title": "MUSASHI 검사 다른 제목"}}  # has Latin
        dump4 = [
            _dump(qid="q4", rank=1, doc_id="noise-0", text="anything"),
        ]
        # Now title has Latin → alias_or_synonym does NOT fire.
        # Title overlaps query? query has no MUSASHI (we removed it).
        # Wait, query says "검사 주인공" — the title also has 검사.
        # That triggers title_mismatch (token overlap on "검사").
        # To avoid title_mismatch, query should not overlap title.
        # Use query without 검사:
        row5 = _row(
            rid="q5",
            query="이 작품의 주인공이 누구야 정말 궁금하군",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["MUSASHI"],
        )
        meta5 = {"gold": {"title": "MUSASHI 강철검사 제목"}}
        dump5 = [
            _dump(qid="q5", rank=1, doc_id="noise-0", text="anything"),
        ]
        report = classify_candidate_misses(
            [row5], dump_rows=dump5, doc_metadata=meta5,
            top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_CHARACTER_MISMATCH


class TestLexicalMismatch:
    def test_low_keyword_coverage_classified(self):
        # No title overlap, no proper-noun keywords, coverage 0/3 → lexical.
        row = _row(
            rid="q1",
            query="여러 단어가 길게 들어가 있는 충분히 긴 쿼리이다",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["가나다", "라마바", "사아자"],
        )
        dump = [
            _dump(qid="q1", rank=1, doc_id="noise-0", text="completely unrelated"),
        ]
        meta = {"gold": {"title": ""}}  # no title overlap
        report = classify_candidate_misses(
            [row], dump_rows=dump, doc_metadata=meta,
            top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_LEXICAL_MISMATCH


class TestAliasSynonym:
    def test_latin_keyword_against_hangul_title(self):
        # Latin keyword + hangul-only title → alias_or_synonym wins
        # (priority above title/character/lexical heuristics).
        row = _row(
            rid="q1",
            query="이 작품 어떤 거에 관한 거야 더 길게 길게 길게",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["MUSASHI"],
        )
        meta = {"gold": {"title": "강철검사"}}  # hangul-only title
        report = classify_candidate_misses(
            [row], doc_metadata=meta, top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_ALIAS_SYNONYM


class TestOverlyBroad:
    def test_short_query_falls_to_broad(self):
        # 3 tokens, no title overlap, coverage on hangul keyword high
        # enough → falls to overly_broad.
        row = _row(
            rid="q1",
            query="이 작품 알려줘",  # 3 tokens
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["가나다"],
        )
        dump = [_dump(qid="q1", rank=1, doc_id="noise-0", text="가나다 나옴")]
        meta = {"gold": {"title": "전혀 무관"}}
        report = classify_candidate_misses(
            [row], dump_rows=dump, doc_metadata=meta,
            top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_OVERLY_BROAD


class TestAmbiguousFallthrough:
    def test_long_query_no_signal_falls_through(self):
        row = _row(
            rid="q1",
            query="이것은 매우 길고 일반적인 그러나 무엇에 대한 것인지 명확하지 않은 쿼리야 정말로",
            expected=["gold"],
            retrieved=_padded_retrieve(5, gold_at=6),
            keywords=["어떤_키워드"],
        )
        dump = [_dump(qid="q1", rank=1, doc_id="noise-0", text="어떤_키워드 found")]
        meta = {"gold": {"title": "전혀_무관한"}}
        report = classify_candidate_misses(
            [row], dump_rows=dump, doc_metadata=meta,
            top_ks=(3,), deep_k=10,
        )
        bucket = next(b for b in report.per_top_k[0].buckets if b.count > 0)
        assert bucket.name == BUCKET_AMBIGUOUS_LABEL


class TestMultipleTopKs:
    def test_independent_buckets_per_top_k(self):
        # Place gold at deep position 13 — outside top-5 / top-10,
        # inside top-15.
        row = _row(
            rid="q1",
            query="템플 어쩌구 무언가",
            expected=["gold"],
            retrieved=_padded_retrieve(14, gold_at=13),
        )
        meta = {"gold": {"title": "템플"}}
        report = classify_candidate_misses(
            [row], doc_metadata=meta, top_ks=(5, 10, 15), deep_k=15,
        )
        result_5 = report.per_top_k[0]
        result_10 = report.per_top_k[1]
        result_15 = report.per_top_k[2]
        assert result_5.queries_missed == 1
        assert result_10.queries_missed == 1
        # gold at rank 13 → in top-15 → hit.
        assert result_15.queries_missed == 0


class TestRowsSkipped:
    def test_rows_without_expected_skipped(self):
        rows = [
            _row(rid="q1", query="x", expected=[], retrieved=["a"]),
            _row(rid="q2", query="y", expected=["g"], retrieved=["a"]),
        ]
        report = classify_candidate_misses(rows, top_ks=(3,), deep_k=10)
        assert report.rows_skipped == 1
        assert report.rows_evaluated == 1


class TestSerialization:
    def test_to_dict_carries_all_buckets(self):
        rows = [_row(rid="q1", query="x", expected=["g"], retrieved=["a"])]
        report = classify_candidate_misses(rows, top_ks=(5,), deep_k=10)
        out = candidate_miss_report_to_dict(report)
        assert out["schema"] == "phase2b-candidate-miss-analysis.v1"
        assert out["per_top_k"][0]["top_k"] == 5
        assert "bucket_definitions" in out

    def test_markdown_renders(self):
        rows = [_row(rid="q1", query="x", expected=["g"], retrieved=["a"])]
        report = classify_candidate_misses(rows, top_ks=(5,), deep_k=10)
        md = render_candidate_miss_markdown(report)
        assert "Phase 2B candidate miss analysis" in md
        assert "top-5" in md
