"""Tests for the Phase 2B boost eval orchestrator.

End-to-end coverage of ``run_boost_retrieval_eval`` over a stub
retriever — verifies:

  - the standard retrieval-eval artifacts are produced in the same
    shape as a Phase 2A retrieval-rerank run
  - boost summary aggregates ``boost_applied_count``, title_match_count,
    section_match_count, avg_boost_score correctly
  - rescued / regressed events are detected at the boost_top_k cutoff
  - boost_dump emits one entry per (query, stage, rank)
  - write_boost_artifacts persists all eight files

Stubs avoid any FAISS / SentenceTransformer dependency so the test
runs offline in <1s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from app.capabilities.rag.generation import RetrievedChunk
from eval.harness.boost_eval import (
    build_boost_dump,
    compute_boost_summary,
    run_boost_retrieval_eval,
    write_boost_artifacts,
)
from eval.harness.boost_metadata import doc_metadata_from_records
from eval.harness.boost_scorer import BoostConfig, MetadataBoostReranker
from eval.harness.boosting_retriever import BoostingEvalRetriever


class _StubReport:
    def __init__(self, results, top_k=5):
        self.results = list(results)
        self.dense_retrieval_ms = 5.0
        self.rerank_ms = None
        self.rerank_breakdown_ms = None
        self.candidate_k = top_k
        self.topk_gap = None
        self.topk_rel_gap = None
        self.use_mmr = False
        self.mmr_lambda = None
        self.dup_rate = 0.0
        self.parsed_query = None
        self.filters = {}
        self.filter_produced_no_docs = False
        self.index_version = "stub-v1"
        self.embedding_model = "stub-emb"
        self.reranker_name = "noop"
        self.top_k = top_k


class _ScriptedRetriever:
    """Returns a different result list per query, keyed by query string."""

    def __init__(self, scripts, top_k=5):
        self._scripts = dict(scripts)
        self._top_k = top_k

    def retrieve(self, query):
        results = self._scripts.get(query, [])
        return _StubReport(results, top_k=self._top_k)


def _chunk(cid, did, section, score):
    return RetrievedChunk(cid, did, section, "x", score)


def _doc_meta():
    return doc_metadata_from_records(
        [
            {
                "doc_id": "good",
                "title": "템플",
                "seed": "템플",
                "sections": {"줄거리": {"chunks": ["x"]}, "본문": {"chunks": ["x"]}},
            },
            {
                "doc_id": "bad",
                "title": "다른",
                "seed": "다른",
                "sections": {"줄거리": {"chunks": ["x"]}, "본문": {"chunks": ["x"]}},
            },
        ]
    )


def _dataset():
    return [
        # q1: dense puts bad above good; boost should rescue good.
        {
            "id": "q1",
            "query": "템플의 주요 주제",
            "expected_doc_ids": ["good"],
        },
        # q2: dense already finds good first; boost is neutral.
        {
            "id": "q2",
            "query": "템플 다른 무언가",
            "expected_doc_ids": ["good"],
        },
    ]


def _make_retriever(boost_cfg, *, post=None):
    base = _ScriptedRetriever(
        scripts={
            "템플의 주요 주제": [
                _chunk("c-bad", "bad", "줄거리", 0.55),
                _chunk("c-good", "good", "줄거리", 0.40),
            ],
            "템플 다른 무언가": [
                _chunk("c-good", "good", "줄거리", 0.55),
                _chunk("c-bad", "bad", "줄거리", 0.40),
            ],
        },
        top_k=5,
    )
    boost = MetadataBoostReranker(
        config=boost_cfg, doc_metadata=_doc_meta(),
    )
    return BoostingEvalRetriever(
        base_retriever=base,
        boost_reranker=boost,
        post_reranker=post,
        boost_top_k=5,
        final_top_k=5,
    )


# ---------------------------------------------------------------------------
# Standard artifacts
# ---------------------------------------------------------------------------


class TestStandardArtifacts:
    def test_summary_records_hit_at_1_after_boost(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
            mrr_k=10,
            ndcg_k=10,
            extra_hit_ks=(3, 5),
        )
        # Both queries hit on rank 1 after boost (q1 rescued, q2 already hit).
        assert artifacts.summary.mean_hit_at_1 == pytest.approx(1.0)

    def test_dense_baseline_records_lower_hit_at_1(self):
        retriever = _make_retriever(BoostConfig.disabled())
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        # q1 misses at rank 1 dense (bad > good); q2 hits → mean 0.5.
        assert artifacts.summary.mean_hit_at_1 == pytest.approx(0.5)

    def test_top_k_dump_has_one_row_per_query_per_rank(self):
        retriever = _make_retriever(BoostConfig.disabled())
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        # 2 queries × 2 chunks per query = 4 dump rows.
        assert len(artifacts.top_k_dump) == 4


# ---------------------------------------------------------------------------
# Boost summary
# ---------------------------------------------------------------------------


class TestBoostSummary:
    def test_rescue_detected(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        # Both queries already have good in dense top-5 (boost_top_k=5),
        # so no rescue at the boost cutoff. To assert rescue we need to
        # narrow the boost cutoff so q1's dense top-1 misses good.
        # See test_rescue_at_top_1 for that.
        assert artifacts.boost_summary.boosted_rescued_count == 0
        # But avg_boost_score should be > 0 because chunks got boosted.
        assert artifacts.boost_summary.boost_applied_count > 0
        assert artifacts.boost_summary.avg_boost_score > 0

    def test_rescue_at_top_1(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=1,  # narrow window
            boost_top_k=1,
        )
        # q1: dense top-1 = bad → miss; boost top-1 = good → hit. RESCUED.
        # q2: dense top-1 = good → hit; boost top-1 = good → hit. NEUTRAL.
        assert artifacts.boost_summary.boosted_rescued_count == 1
        assert artifacts.boost_summary.boosted_regressed_count == 0
        assert artifacts.boost_summary.boost_neutral_count == 1

    def test_title_match_counted(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        # q1 "템플의 주요 주제": only "템플" appears → 1 exact match.
        # q2 "템플 다른 무언가": both "템플" and "다른" appear → 2 matches.
        # Total = 3 chunks with title_exact match (good gets 2, bad gets 1).
        assert artifacts.boost_summary.title_exact_match_count == 3
        assert artifacts.boost_summary.title_partial_match_count == 0
        # Both queries had at least one boosted chunk → 2 queries with boost.
        assert artifacts.boost_summary.queries_with_any_boost == 2

    def test_disabled_boost_records_zero_aggregates(self):
        retriever = _make_retriever(BoostConfig.disabled())
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        assert artifacts.boost_summary.boost_applied_count == 0
        assert artifacts.boost_summary.title_exact_match_count == 0
        assert artifacts.boost_summary.boosted_rescued_count == 0


# ---------------------------------------------------------------------------
# Boost dump
# ---------------------------------------------------------------------------


class TestBoostDump:
    def test_dump_has_three_stages_per_query(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        stages = {entry.stage for entry in artifacts.boost_dump}
        assert stages == {"dense", "boosted", "final"}

    def test_dump_row_breakdown_split(self):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        boosted_for_q1 = [
            e for e in artifacts.boost_dump
            if e.query_id == "q1" and e.stage == "boosted"
        ]
        # The "good" doc chunk should carry boost_score > 0 and a title
        # match kind on its boosted-stage entry.
        good_entry = next(e for e in boosted_for_q1 if e.doc_id == "good")
        assert good_entry.boost_score > 0
        assert good_entry.title_match_kind == "exact"
        # And dense_score + boost_score should equal final_score.
        assert pytest.approx(
            good_entry.dense_score + good_entry.boost_score, abs=1e-6
        ) == good_entry.final_score


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_writes_all_eight_files(self, tmp_path: Path):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
        )
        write_boost_artifacts(
            artifacts, tmp_path, metadata={"harness": "test"},
        )
        for name in (
            "retrieval_eval_report.json",
            "retrieval_eval_report.md",
            "top_k_dump.jsonl",
            "duplicate_analysis.json",
            "boost_summary.json",
            "boost_dump.jsonl",
            "boost_rescued.json",
            "boost_regressed.json",
            "boost_summary.md",
        ):
            assert (tmp_path / name).exists(), f"missing {name}"

    def test_boost_summary_json_round_trips(self, tmp_path: Path):
        retriever = _make_retriever(BoostConfig(title_exact_boost=0.20))
        artifacts = run_boost_retrieval_eval(
            _dataset(),
            retriever=retriever,
            final_top_k=5,
            boost_top_k=5,
            config={"title_exact_boost": 0.20},
        )
        write_boost_artifacts(
            artifacts, tmp_path, metadata={"harness": "test"},
        )
        loaded = json.loads(
            (tmp_path / "boost_summary.json").read_text(encoding="utf-8")
        )
        assert loaded["schema"] == "phase2b-boost-summary.v1"
        assert loaded["boost_top_k"] == 5
        assert loaded["title_exact_match_count"] == 3
        assert loaded["config"]["title_exact_boost"] == 0.20
