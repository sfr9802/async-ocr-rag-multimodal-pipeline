"""Tests for the Phase 2B BoostingEvalRetriever wrapper.

The wrapper composes a dense-only base retriever with a boost reranker
and an optional post-boost reranker. These tests use a simple
``StubRetriever`` to drive deterministic candidate lists into the
wrapper so we can pin the contract without spinning up FAISS.

Coverage:
  - byte-identical pass-through when boost is disabled and no post
    reranker is attached
  - boost reorders the wrapper's final results
  - post reranker runs AFTER the boost, on the boost-reordered list
  - rerank_ms is None on the noop / no-post-reranker path
  - rerank_ms is recorded when post reranker is non-noop
  - call_log appends one entry per retrieve(); reset_call_log clears it
  - dense_candidates / boosted_candidates / final_results capture each
    stage with the correct dense / boost / final score breakdown
  - construction validates boost_top_k vs base_top_k and
    final_top_k vs boost_top_k
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import NoOpReranker
from eval.harness.boost_metadata import doc_metadata_from_records
from eval.harness.boost_scorer import BoostConfig, MetadataBoostReranker
from eval.harness.boosting_retriever import BoostingEvalRetriever


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubReport:
    def __init__(self, results, dense_ms=10.0, top_k=15):
        self.results = list(results)
        self.dense_retrieval_ms = dense_ms
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


class _StubRetriever:
    def __init__(self, results, *, top_k=15):
        self._results = list(results)
        self._top_k = top_k

    @property
    def _stub_top_k(self):
        return self._top_k

    def retrieve(self, query):
        return _StubReport(self._results, top_k=self._top_k)


def _chunk(cid, did, section, score, text="x"):
    return RetrievedChunk(cid, did, section, text, score)


def _meta_records(*titles):
    """Build doc metadata records for a few docs sharing the same sections."""
    out = []
    for did, title in titles:
        out.append(
            {
                "doc_id": did,
                "title": title,
                "seed": title,
                "sections": {n: {"chunks": ["x"]} for n in ("요약", "줄거리", "본문")},
            }
        )
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_base_retriever(self):
        with pytest.raises(ValueError):
            BoostingEvalRetriever(
                base_retriever=None,
                boost_reranker=MetadataBoostReranker(
                    config=BoostConfig.disabled(), doc_metadata={},
                ),
            )

    def test_requires_boost_reranker(self):
        base = _StubRetriever([])
        with pytest.raises(ValueError):
            BoostingEvalRetriever(base_retriever=base, boost_reranker=None)

    def test_boost_top_k_cannot_exceed_base_top_k(self):
        base = _StubRetriever([], top_k=10)
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata={},
        )
        with pytest.raises(ValueError):
            BoostingEvalRetriever(
                base_retriever=base,
                boost_reranker=boost,
                boost_top_k=20,
            )

    def test_final_top_k_cannot_exceed_boost_top_k(self):
        base = _StubRetriever([], top_k=15)
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata={},
        )
        with pytest.raises(ValueError):
            BoostingEvalRetriever(
                base_retriever=base,
                boost_reranker=boost,
                boost_top_k=10,
                final_top_k=15,
            )

    def test_defaults_apply_when_top_k_omitted(self):
        base = _StubRetriever([], top_k=15)
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata={},
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base, boost_reranker=boost,
        )
        assert wrapper.boost_top_k == 15
        assert wrapper.final_top_k == 15


# ---------------------------------------------------------------------------
# Off-mode contract
# ---------------------------------------------------------------------------


class TestOffMode:
    def test_byte_identical_to_base_when_disabled(self):
        base = _StubRetriever(
            [
                _chunk("c1", "d1", "요약", 0.7),
                _chunk("c2", "d2", "줄거리", 0.6),
                _chunk("c3", "d3", "본문", 0.5),
            ],
            top_k=10,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base,
            boost_reranker=MetadataBoostReranker(
                config=BoostConfig.disabled(), doc_metadata={},
            ),
        )
        report = wrapper.retrieve("any query")
        assert [r.chunk_id for r in report.results] == ["c1", "c2", "c3"]
        assert [r.score for r in report.results] == [0.7, 0.6, 0.5]
        # In off mode + no post reranker, the reranker_name surfaces the
        # base retriever's value so downstream eval doesn't think it
        # was reranked.
        assert report.reranker_name == "noop"
        assert report.rerank_ms is None


# ---------------------------------------------------------------------------
# Boost reorders
# ---------------------------------------------------------------------------


class TestBoostReordersFinal:
    def test_boost_pulls_title_match_to_top(self):
        meta = doc_metadata_from_records(
            _meta_records(("d-good", "템플"), ("d-bad", "다른"))
        )
        base = _StubRetriever(
            [
                _chunk("c-bad", "d-bad", "요약", 0.55),
                _chunk("c-good", "d-good", "줄거리", 0.40),
            ],
            top_k=10,
        )
        boost = MetadataBoostReranker(
            config=BoostConfig(title_exact_boost=0.20),
            doc_metadata=meta,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base, boost_reranker=boost,
        )
        report = wrapper.retrieve("템플의 주요 주제")
        assert [r.chunk_id for r in report.results] == ["c-good", "c-bad"]
        assert pytest.approx(report.results[0].score) == 0.60

    def test_dense_candidates_preserved_pre_boost(self):
        meta = doc_metadata_from_records(
            _meta_records(("d1", "템플"), ("d2", "다른"))
        )
        base = _StubRetriever(
            [
                _chunk("c-bad", "d2", "요약", 0.55),
                _chunk("c-good", "d1", "줄거리", 0.40),
            ],
            top_k=10,
        )
        boost = MetadataBoostReranker(
            config=BoostConfig(title_exact_boost=0.20), doc_metadata=meta,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base, boost_reranker=boost,
        )
        report = wrapper.retrieve("템플의 주요 주제")
        # Dense ordering preserved (pre-boost).
        assert [c.chunk_id for c in report.dense_candidates] == [
            "c-bad", "c-good",
        ]
        # Boosted ordering reflects the reorder.
        assert [c.chunk_id for c in report.boosted_candidates] == [
            "c-good", "c-bad",
        ]
        # Boost score breakdown attached.
        good = next(c for c in report.boosted_candidates if c.chunk_id == "c-good")
        assert good.boost_total > 0
        assert good.boost_breakdown.title_match_kind == "exact"


# ---------------------------------------------------------------------------
# Post-rerank chain
# ---------------------------------------------------------------------------


class _FixedScoreReranker:
    """Test reranker that returns chunks in a fixed predetermined order."""

    def __init__(self, *, order_chunk_ids, name="fake_post"):
        self._order = list(order_chunk_ids)
        self._name = name

    @property
    def name(self):
        return self._name

    def rerank(self, query, chunks, k):
        by_id = {c.chunk_id: c for c in chunks}
        out = []
        for cid in self._order:
            if cid in by_id:
                out.append(by_id[cid])
            if len(out) >= k:
                break
        return out


class TestPostRerankChain:
    def test_post_reranker_runs_after_boost(self):
        meta = doc_metadata_from_records(
            _meta_records(("d1", "템플"), ("d2", "다른"))
        )
        base = _StubRetriever(
            [
                _chunk("c1", "d1", "요약", 0.40),
                _chunk("c2", "d2", "본문", 0.55),
            ],
            top_k=10,
        )
        # Boost would reorder to [c1, c2]; post reranker forces [c2, c1].
        boost = MetadataBoostReranker(
            config=BoostConfig(title_exact_boost=0.20), doc_metadata=meta,
        )
        post = _FixedScoreReranker(order_chunk_ids=["c2", "c1"])
        wrapper = BoostingEvalRetriever(
            base_retriever=base,
            boost_reranker=boost,
            post_reranker=post,
            final_top_k=2,
        )
        report = wrapper.retrieve("템플 어쩌구")
        assert [r.chunk_id for r in report.results] == ["c2", "c1"]
        assert report.reranker_name == "metadata_boost+fake_post"
        # rerank_ms reflects the post reranker step (non-noop name).
        assert report.rerank_ms is not None

    def test_noop_post_reranker_keeps_rerank_ms_none(self):
        meta = doc_metadata_from_records(_meta_records(("d1", "x")))
        base = _StubRetriever(
            [_chunk("c1", "d1", "요약", 0.5)], top_k=10,
        )
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata=meta,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base,
            boost_reranker=boost,
            post_reranker=NoOpReranker(),
            final_top_k=1,
        )
        report = wrapper.retrieve("q")
        assert report.rerank_ms is None


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------


class TestCallLog:
    def test_call_log_grows_per_retrieve(self):
        meta = doc_metadata_from_records(_meta_records(("d1", "x")))
        base = _StubRetriever([_chunk("c1", "d1", "요약", 0.5)], top_k=5)
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata=meta,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base, boost_reranker=boost,
        )
        wrapper.retrieve("q1")
        wrapper.retrieve("q2")
        assert len(wrapper.call_log) == 2
        assert wrapper.call_log[0].query == "q1"
        assert wrapper.call_log[1].query == "q2"

    def test_reset_call_log_clears(self):
        meta = doc_metadata_from_records(_meta_records(("d1", "x")))
        base = _StubRetriever([_chunk("c1", "d1", "요약", 0.5)], top_k=5)
        boost = MetadataBoostReranker(
            config=BoostConfig.disabled(), doc_metadata=meta,
        )
        wrapper = BoostingEvalRetriever(
            base_retriever=base, boost_reranker=boost,
        )
        wrapper.retrieve("q1")
        wrapper.reset_call_log()
        assert wrapper.call_log == []
        assert wrapper.last_call is None
