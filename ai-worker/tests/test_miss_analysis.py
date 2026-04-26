"""Tests for the retrieval miss-bucket analyzer.

Operates on plain dict rows / dump rows so the analyzer can be
re-run from the JSON written by the eval CLI without re-running
retrieval. Coverage targets:

  - 4-bucket assignment is correct in every cell of the 2x2 cross-tab
  - dump-row matched_expected_keyword wins over fallback substring
    re-derivation when both are present
  - rows with no expected_doc_ids are skipped (with a clean reason)
  - sample cap honored, samples carry top_5 dump entries
  - per-axis breakdowns sum back to row_count
  - markdown report renders without raising
"""

from __future__ import annotations

import json

import pytest

from eval.harness.miss_analysis import (
    BUCKET_DOC_HIT_KW_HIT,
    BUCKET_DOC_HIT_KW_MISS,
    BUCKET_DOC_MISS_KW_HIT,
    BUCKET_DOC_MISS_KW_MISS,
    BUCKET_ORDER,
    classify_rows,
    miss_analysis_to_dict,
    render_miss_analysis_markdown,
)


def _row(
    *,
    rid: str,
    expected_doc_ids,
    retrieved_doc_ids,
    expected_keywords,
    answer_type="title_lookup",
    difficulty="easy",
    language="ko",
    query=None,
):
    return {
        "id": rid,
        "query": query or f"query for {rid}",
        "language": language,
        "expected_doc_ids": list(expected_doc_ids),
        "expected_section_keywords": list(expected_keywords),
        "answer_type": answer_type,
        "difficulty": difficulty,
        "retrieved_doc_ids": list(retrieved_doc_ids),
    }


def _dump(*, qid, rank, doc_id, section, text, score=0.5, matched=()):
    return {
        "query_id": qid,
        "rank": rank,
        "doc_id": doc_id,
        "section_path": section,
        "score": score,
        "chunk_preview": text,
        "matched_expected_keyword": list(matched),
    }


class TestClassifyRows4Buckets:
    def test_doc_hit_keyword_hit(self):
        row = _row(
            rid="q1",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["gold", "x"],
            expected_keywords=["bookshop"],
        )
        dumps = [
            _dump(qid="q1", rank=1, doc_id="gold", section="overview",
                  text="bookshop"),
            _dump(qid="q1", rank=2, doc_id="x", section="other",
                  text="off-topic"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_HIT_KW_HIT] == 1
        assert sum(counts.values()) == 1

    def test_doc_hit_keyword_miss(self):
        row = _row(
            rid="q2",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["gold", "x"],
            expected_keywords=["the_specific_term"],
        )
        dumps = [
            _dump(qid="q2", rank=1, doc_id="gold", section="overview",
                  text="something completely different"),
            _dump(qid="q2", rank=2, doc_id="x", section="other",
                  text="and another"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_HIT_KW_MISS] == 1

    def test_doc_miss_keyword_hit(self):
        # gold doc not retrieved but a sibling chunk happens to contain
        # the keyword (typical Phase-0 leakage signal).
        row = _row(
            rid="q3",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["sibling-1", "sibling-2"],
            expected_keywords=["bookshop"],
        )
        dumps = [
            _dump(qid="q3", rank=1, doc_id="sibling-1", section="overview",
                  text="generic bookshop article"),
            _dump(qid="q3", rank=2, doc_id="sibling-2", section="other",
                  text="unrelated"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_MISS_KW_HIT] == 1
        # Sample populated for this failure bucket.
        samples = result.samples[BUCKET_DOC_MISS_KW_HIT]
        assert len(samples) == 1
        assert samples[0].query_id == "q3"
        assert "bookshop" in samples[0].matched_expected_keyword
        # Top-5 carries the dump entries (we only had 2).
        assert len(samples[0].top_5) == 2
        assert samples[0].top_5[0].doc_id == "sibling-1"

    def test_doc_miss_keyword_miss(self):
        row = _row(
            rid="q4",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["a", "b"],
            expected_keywords=["the_specific_term"],
        )
        dumps = [
            _dump(qid="q4", rank=1, doc_id="a", section="overview",
                  text="off-topic"),
            _dump(qid="q4", rank=2, doc_id="b", section="other",
                  text="also off-topic"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_MISS_KW_MISS] == 1
        samples = result.samples[BUCKET_DOC_MISS_KW_MISS]
        assert len(samples) == 1
        # No keyword matched — sample carries empty matched list.
        assert samples[0].matched_expected_keyword == []


class TestEdgeCases:
    def test_skip_rows_without_expected_doc_ids(self):
        row = _row(
            rid="q-no-doc",
            expected_doc_ids=[],
            retrieved_doc_ids=["a"],
            expected_keywords=["bookshop"],
        )
        dumps = [
            _dump(qid="q-no-doc", rank=1, doc_id="a", section="s",
                  text="bookshop"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        assert result.rows_evaluated == 0
        assert result.rows_skipped == 1
        assert all(b.count == 0 for b in result.buckets)
        assert "expected_doc_ids" in result.skip_reason

    def test_dump_matched_keyword_wins_over_fallback(self):
        # The dump pre-computed the matched keyword from the FULL chunk
        # text, but the analyzer here only sees the chunk_preview. The
        # preview happens NOT to contain the keyword — analyzer should
        # still trust the dump's matched_expected_keyword and call this
        # a doc_miss_keyword_hit.
        row = _row(
            rid="q-leak",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["sibling"],
            expected_keywords=["very_specific_term"],
        )
        dumps = [
            _dump(
                qid="q-leak", rank=1, doc_id="sibling",
                section="overview",
                text="preview that does not contain it",
                matched=["very_specific_term"],
            ),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_MISS_KW_HIT] == 1

    def test_keyword_match_uses_section_path(self):
        # Keyword only appears in the section_path, not the chunk text.
        # This mirrors the dump's matched_expected_keyword behavior and
        # the harness's expected_keyword_match_rate logic.
        row = _row(
            rid="q-sec",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["sibling"],
            expected_keywords=["만화"],
        )
        dumps = [
            _dump(qid="q-sec", rank=1, doc_id="sibling", section="만화/원작",
                  text="completely off-topic content"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_MISS_KW_HIT] == 1

    def test_top_k_cuts_off_late_hit(self):
        # gold doc at rank 11 — top_k=10 should treat this as a miss.
        retrieved = [f"doc-{i}" for i in range(1, 12)]
        retrieved[10] = "gold"  # rank 11
        row = _row(
            rid="q-cut",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=retrieved,
            expected_keywords=["off"],
        )
        dumps = [
            _dump(qid="q-cut", rank=i + 1, doc_id=retrieved[i], section="s",
                  text="something")
            for i in range(11)
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        counts = {b.name: b.count for b in result.buckets}
        assert counts[BUCKET_DOC_MISS_KW_MISS] == 1
        assert counts[BUCKET_DOC_HIT_KW_MISS] == 0


class TestSampleCap:
    def test_sample_limit_honored(self):
        rows = []
        dumps = []
        for i in range(40):
            qid = f"miss-{i:02d}"
            rows.append(
                _row(
                    rid=qid,
                    expected_doc_ids=["gold"],
                    retrieved_doc_ids=["other"],
                    expected_keywords=["nothing-here"],
                )
            )
            dumps.append(
                _dump(qid=qid, rank=1, doc_id="other", section="x", text="y")
            )
        result = classify_rows(rows, dump_rows=dumps, top_k=10, sample_limit=5)
        # 40 rows all in doc_miss_keyword_miss
        miss_count = next(
            b.count for b in result.buckets if b.name == BUCKET_DOC_MISS_KW_MISS
        )
        assert miss_count == 40
        assert len(result.samples[BUCKET_DOC_MISS_KW_MISS]) == 5


class TestPerAxisBreakdown:
    def test_per_answer_type_sums_to_row_count(self):
        rows = [
            _row(rid="a", expected_doc_ids=["g"], retrieved_doc_ids=["g"],
                 expected_keywords=["x"], answer_type="title_lookup"),
            _row(rid="b", expected_doc_ids=["g"], retrieved_doc_ids=["g"],
                 expected_keywords=["x"], answer_type="summary_plot"),
            _row(rid="c", expected_doc_ids=["g"], retrieved_doc_ids=["other"],
                 expected_keywords=["x"], answer_type="summary_plot"),
        ]
        dumps = [
            _dump(qid="a", rank=1, doc_id="g", section="s", text="x"),
            _dump(qid="b", rank=1, doc_id="g", section="s", text="x"),
            _dump(qid="c", rank=1, doc_id="other", section="s", text="off"),
        ]
        result = classify_rows(rows, dump_rows=dumps, top_k=10)
        for atype, counts in result.per_answer_type.items():
            n_for_type = sum(1 for r in rows if r["answer_type"] == atype)
            assert sum(counts.values()) == n_for_type


class TestSerialization:
    def test_to_dict_round_trips_through_json(self):
        row = _row(
            rid="q1",
            expected_doc_ids=["g"],
            retrieved_doc_ids=["g"],
            expected_keywords=["x"],
        )
        dumps = [
            _dump(qid="q1", rank=1, doc_id="g", section="s", text="x"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        payload = miss_analysis_to_dict(result)
        # Must be JSON-serializable.
        json_str = json.dumps(payload, ensure_ascii=False)
        assert "doc_hit_keyword_hit" in json_str
        # Round trip preserves structure.
        loaded = json.loads(json_str)
        assert loaded["top_k"] == 10
        assert loaded["rows_evaluated"] == 1
        assert [b["name"] for b in loaded["buckets"]] == list(BUCKET_ORDER)

    def test_markdown_report_renders(self):
        row = _row(
            rid="q1",
            expected_doc_ids=["gold"],
            retrieved_doc_ids=["other"],
            expected_keywords=["bookshop"],
        )
        dumps = [
            _dump(qid="q1", rank=1, doc_id="other", section="s",
                  text="bookshop appears here"),
        ]
        result = classify_rows([row], dump_rows=dumps, top_k=10)
        md = render_miss_analysis_markdown(result)
        assert "# Retrieval miss-bucket analysis" in md
        assert "## Buckets" in md
        assert "doc_miss_keyword_hit" in md
        # Sample section appears when miss bucket has samples.
        assert "Samples — doc_miss_keyword_hit" in md
