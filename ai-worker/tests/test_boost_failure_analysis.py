"""Tests for the Phase 2B boost-vs-baseline failure analysis.

Each test pins one of the five outcome groups by constructing
matched dense/boost (and optional rerank) row dicts keyed by query
id, then asserts the right bucket fires.
"""

from __future__ import annotations

import pytest

from eval.harness.boost_failure_analysis import (
    GROUP_BOOST_HIT_RERANK_HIT,
    GROUP_BOOST_HIT_RERANK_MISS,
    GROUP_BOTH_MISS,
    GROUP_DENSE_HIT_BOOST_MISS,
    GROUP_DENSE_MISS_BOOST_HIT,
    GROUP_NEUTRAL,
    GROUP_ORDER,
    boost_failure_analysis_to_dict,
    classify_boost_failures,
    render_boost_failure_markdown,
)


def _row(qid, expected, retrieved, scores=None, query="query text"):
    return {
        "id": qid,
        "query": query,
        "expected_doc_ids": list(expected),
        "expected_section_keywords": [],
        "retrieved_doc_ids": list(retrieved),
        "retrieval_scores": list(scores or [0.5] * len(retrieved)),
        "answer_type": "x",
        "difficulty": "medium",
    }


class TestGroupAssignment:
    def test_dense_miss_boost_hit(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a", "b", "c"])],
            boost_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_DENSE_MISS_BOOST_HIT] == 1
        assert sum(counts.values()) == 1

    def test_dense_hit_boost_miss(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            boost_rows=[_row("q1", ["g"], ["a", "b", "c"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_DENSE_HIT_BOOST_MISS] == 1

    def test_boost_hit_rerank_hit_when_rerank_provided(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a", "b", "g"])],
            boost_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            rerank_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_BOOST_HIT_RERANK_HIT] == 1

    def test_boost_hit_rerank_miss_when_rerank_drops(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a", "b", "g"])],
            boost_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            rerank_rows=[_row("q1", ["g"], ["a", "b", "c"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_BOOST_HIT_RERANK_MISS] == 1

    def test_both_miss(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a", "b", "c"])],
            boost_rows=[_row("q1", ["g"], ["a", "b", "c"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_BOTH_MISS] == 1

    def test_neutral_when_dense_and_boost_both_hit_no_rerank(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            boost_rows=[_row("q1", ["g"], ["g", "a", "b"])],
            top_k=3,
        )
        counts = {g.name: g.count for g in analysis.groups}
        assert counts[GROUP_NEUTRAL] == 1


class TestSkip:
    def test_query_missing_from_boost_skipped(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a"])],
            boost_rows=[],
            top_k=3,
        )
        assert analysis.queries_skipped == 1
        assert analysis.queries_evaluated == 0

    def test_query_without_expected_skipped(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", [], ["a"])],
            boost_rows=[_row("q1", [], ["a"])],
            top_k=3,
        )
        assert analysis.queries_skipped == 1
        assert analysis.queries_evaluated == 0


class TestSampleCap:
    def test_sample_capped_per_bucket(self):
        # 50 rescue rows; cap at sample_limit=10.
        dense_rows = [
            _row(f"q{i}", ["g"], ["a", "b", "c"]) for i in range(50)
        ]
        boost_rows = [
            _row(f"q{i}", ["g"], ["g", "a", "b"]) for i in range(50)
        ]
        analysis = classify_boost_failures(
            dense_rows=dense_rows,
            boost_rows=boost_rows,
            top_k=3,
            sample_limit=10,
        )
        rescued = analysis.samples[GROUP_DENSE_MISS_BOOST_HIT]
        assert len(rescued) == 10

    def test_sample_includes_dense_and_boost_top5(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a", "b", "c", "d"])],
            boost_rows=[_row("q1", ["g"], ["g", "a", "b", "c"])],
            top_k=3,
        )
        sample = analysis.samples[GROUP_DENSE_MISS_BOOST_HIT][0]
        assert sample.dense_top5_doc_ids[0] == "a"
        assert sample.boost_top5_doc_ids[0] == "g"


class TestSerialization:
    def test_to_dict_round_trips(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a"])],
            boost_rows=[_row("q1", ["g"], ["g"])],
            top_k=1,
        )
        out = boost_failure_analysis_to_dict(analysis)
        assert out["schema"] == "phase2b-boost-failure-analysis.v1"
        assert out["top_k"] == 1
        # All groups appear in the output even when count == 0.
        names = {g["name"] for g in out["groups"]}
        for name in GROUP_ORDER:
            assert name in names

    def test_markdown_renders(self):
        analysis = classify_boost_failures(
            dense_rows=[_row("q1", ["g"], ["a"])],
            boost_rows=[_row("q1", ["g"], ["g"])],
            top_k=1,
        )
        md = render_boost_failure_markdown(analysis)
        assert "Phase 2B boost failure analysis" in md
