"""Tests for the wide-retrieval / MMR / title-cap eval helpers.

Covers:
  - apply_title_cap behaviour at cap=1 and cap=2 with title_provider
    AND fallback to doc_id.
  - mmr_select_score_fallback determinism + degenerate cases (lambda=
    1 reproduces relevance order; lambda=0 picks by penalty alone).
  - WideRetrievalEvalAdapter pipeline composition: pool fetch, MMR
    application, title-cap on rerank input, rerank_in slicing,
    title-cap on final, final_top_k truncation.
  - query_type heuristic tagging on representative Korean queries.
  - DocTitleResolver loads titles from a tiny synthetic corpus.

All tests are offline. WideRetrievalEvalAdapter uses a stub
``Retriever`` that exposes the same ``_top_k`` / ``_candidate_k`` /
``_reranker`` mutators the real one does, plus a scripted
``retrieve(query)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from eval.harness.query_type_heuristic import (
    QT_AMBIGUOUS,
    QT_CHARACTER_ATTRIBUTE,
    QT_CHARACTER_RELATION,
    QT_COMPARISON,
    QT_PLOT_EVENT,
    QT_REVIEW_RECEPTION,
    QT_SETTING,
    QT_TITLE_DIRECT,
    QT_UNKNOWN,
    summarize_distribution,
    tag_query,
    tag_rows,
    write_draft_jsonl,
)
from eval.harness.wide_retrieval_adapter import (
    WideRetrievalConfig,
    WideRetrievalEvalAdapter,
)
from eval.harness.wide_retrieval_helpers import (
    DEFAULT_DOC_ID_PENALTY,
    DEFAULT_TITLE_PENALTY,
    DocTitleResolver,
    apply_title_cap,
    count_keys,
    mmr_select_score_fallback,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny chunk + retriever stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubChunk:
    chunk_id: str
    doc_id: str
    section: str = "overview"
    text: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None
    title: Optional[str] = None  # used by tests with title_provider


def _title_of(chunk: Any) -> Optional[str]:
    return getattr(chunk, "title", None)


# ---------------------------------------------------------------------------
# 1. apply_title_cap
# ---------------------------------------------------------------------------


class TestTitleCap:
    def test_no_cap_returns_copy(self):
        chunks = [_StubChunk(f"c{i}", "doc-A") for i in range(5)]
        out = apply_title_cap(chunks, cap=None)
        assert len(out) == 5
        # Different list object, same content.
        assert out is not chunks
        out2 = apply_title_cap(chunks, cap=0)
        assert len(out2) == 5

    def test_cap_one_drops_repeats(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
            _StubChunk("c4", "doc-A"),
            _StubChunk("c5", "doc-B"),
        ]
        out = apply_title_cap(chunks, cap=1)
        ids = [c.chunk_id for c in out]
        assert ids == ["c1", "c3"]

    def test_cap_two_keeps_pair_per_doc(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-B"),
        ]
        out = apply_title_cap(chunks, cap=2)
        ids = [c.chunk_id for c in out]
        assert ids == ["c1", "c2", "c4"]

    def test_title_provider_groups_by_title(self):
        # Two different doc_ids, same title — title cap collapses them.
        chunks = [
            _StubChunk("c1", "doc-A", title="Series-1"),
            _StubChunk("c2", "doc-B", title="Series-1"),
            _StubChunk("c3", "doc-C", title="Series-2"),
        ]
        out = apply_title_cap(chunks, cap=1, title_provider=_title_of)
        ids = [c.chunk_id for c in out]
        assert ids == ["c1", "c3"]

    def test_missing_title_falls_back_to_doc_id(self):
        # title_provider returns None for some chunks → doc_id grouping.
        def provider(c):
            return None  # always None — like a missing resolver
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        out = apply_title_cap(chunks, cap=1, title_provider=provider)
        ids = [c.chunk_id for c in out]
        assert ids == ["c1", "c3"]

    def test_empty_doc_id_passes_through(self):
        chunks = [
            _StubChunk("c1", ""),
            _StubChunk("c2", ""),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-A"),
        ]
        out = apply_title_cap(chunks, cap=1)
        # both empty-id chunks survive (no group key); doc-A capped at 1.
        ids = [c.chunk_id for c in out]
        assert ids == ["c1", "c2", "c3"]

    def test_count_keys_diagnostic(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        counts = count_keys(chunks)
        assert counts.get("doc-a") == 2
        assert counts.get("doc-b") == 1


# ---------------------------------------------------------------------------
# 2. mmr_select_score_fallback
# ---------------------------------------------------------------------------


class TestMMR:
    def _ordered_chunks(self) -> List[_StubChunk]:
        # Relevance descending. Multiple chunks per doc to exercise
        # diversity penalty.
        return [
            _StubChunk("c1", "doc-A", score=0.95),
            _StubChunk("c2", "doc-A", score=0.94),
            _StubChunk("c3", "doc-B", score=0.92),
            _StubChunk("c4", "doc-A", score=0.91),
            _StubChunk("c5", "doc-C", score=0.90),
        ]

    def test_lambda_one_collapses_to_relevance(self):
        chunks = self._ordered_chunks()
        out = mmr_select_score_fallback(chunks, top_k=3, lambda_val=1.0)
        # No diversity penalty → just take top 3 by relevance.
        assert [c.chunk_id for c in out] == ["c1", "c2", "c3"]

    def test_lambda_below_one_promotes_diversity(self):
        chunks = self._ordered_chunks()
        out = mmr_select_score_fallback(
            chunks, top_k=3, lambda_val=0.5,
            doc_id_penalty=DEFAULT_DOC_ID_PENALTY,
        )
        # First pick is c1. Second pick: c2 (val 0.5*0.94 - 0.5*0.6 = 0.17)
        # vs c3 (val 0.5*0.92 - 0 = 0.46). c3 wins. Then c5 over c2 / c4.
        ids = [c.chunk_id for c in out]
        assert ids[0] == "c1"
        assert ids[1] == "c3"
        # Third pick must be from a doc not yet selected (doc-C).
        assert ids[2] == "c5"

    def test_title_penalty_groups_franchises(self):
        chunks = [
            _StubChunk("c1", "doc-A", score=0.95, title="Series-1"),
            _StubChunk("c2", "doc-B", score=0.93, title="Series-1"),
            _StubChunk("c3", "doc-C", score=0.90, title="Series-2"),
        ]
        out = mmr_select_score_fallback(
            chunks, top_k=2, lambda_val=0.5,
            title_provider=_title_of,
            doc_id_penalty=DEFAULT_DOC_ID_PENALTY,
            title_penalty=DEFAULT_TITLE_PENALTY,
        )
        # c1 picks first. Second: c2 has title penalty (Series-1
        # already selected) → 0.5*0.93 - 0.5*0.4 = 0.265. c3 has no
        # penalty: 0.5*0.90 - 0 = 0.45. c3 wins.
        assert [c.chunk_id for c in out] == ["c1", "c3"]

    def test_top_k_zero_or_empty_returns_empty(self):
        chunks = self._ordered_chunks()
        assert mmr_select_score_fallback(chunks, top_k=0, lambda_val=0.7) == []
        assert mmr_select_score_fallback([], top_k=5, lambda_val=0.7) == []

    def test_deterministic_under_ties(self):
        # Same score, same doc_id — order of input must be preserved
        # at every selection step (no random tiebreaks).
        chunks = [
            _StubChunk("c1", "doc-A", score=0.5),
            _StubChunk("c2", "doc-A", score=0.5),
            _StubChunk("c3", "doc-A", score=0.5),
        ]
        out_a = mmr_select_score_fallback(chunks, top_k=3, lambda_val=0.7)
        out_b = mmr_select_score_fallback(chunks, top_k=3, lambda_val=0.7)
        assert [c.chunk_id for c in out_a] == [c.chunk_id for c in out_b]
        assert [c.chunk_id for c in out_a] == ["c1", "c2", "c3"]

    def test_uses_rerank_score_when_present(self):
        chunks = [
            _StubChunk("c1", "doc-A", score=0.1, rerank_score=0.99),
            _StubChunk("c2", "doc-B", score=0.99, rerank_score=0.10),
        ]
        out = mmr_select_score_fallback(chunks, top_k=2, lambda_val=1.0)
        # rerank_score wins → c1 first.
        assert [c.chunk_id for c in out] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# 3. WideRetrievalEvalAdapter
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Minimal stub matching the ``Retriever`` mutator surface.

    Returns a scripted result on ``retrieve()`` capped at the current
    ``_top_k``. Mirrors the production attributes the adapter mutates.
    """

    def __init__(self, pool: List[_StubChunk]):
        self._pool = list(pool)
        self._top_k = 10
        self._candidate_k = 10
        self._reranker = None
        self._use_mmr = False
        self._mmr_lambda = 0.7

    def retrieve(self, query: str):
        return SimpleNamespace(
            results=self._pool[: self._top_k],
            index_version="stub-v1",
            embedding_model="stub-model",
            reranker_name="noop",
            rerank_ms=None,
            dense_retrieval_ms=12.0,
            rerank_breakdown_ms=None,
            candidate_doc_ids=[],
        )


class _StubReranker:
    """Reverses the input list to prove the adapter forwards the slice."""

    name = "stub-reverser"
    last_breakdown_ms = None

    def rerank(self, query, chunks, k):
        # Reverse the chunks (deterministic ordering signal).
        return list(reversed(chunks))[:k]


class TestWideRetrievalAdapter:
    def _pool(self) -> List[_StubChunk]:
        # 12 chunks across 4 doc_ids with 2 titles, decreasing relevance.
        return [
            _StubChunk("c01", "doc-A", title="Series-1", score=1.00),
            _StubChunk("c02", "doc-A", title="Series-1", score=0.99),
            _StubChunk("c03", "doc-A", title="Series-1", score=0.98),
            _StubChunk("c04", "doc-B", title="Series-1", score=0.97),
            _StubChunk("c05", "doc-B", title="Series-1", score=0.96),
            _StubChunk("c06", "doc-C", title="Series-2", score=0.95),
            _StubChunk("c07", "doc-C", title="Series-2", score=0.94),
            _StubChunk("c08", "doc-D", title="Series-3", score=0.93),
            _StubChunk("c09", "doc-D", title="Series-3", score=0.92),
            _StubChunk("c10", "doc-D", title="Series-3", score=0.91),
            _StubChunk("c11", "doc-A", title="Series-1", score=0.90),
            _StubChunk("c12", "doc-B", title="Series-1", score=0.89),
        ]

    def test_candidate_k_set_on_summary_via_pool_fetch(self):
        retriever = _StubRetriever(self._pool())
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=10, final_top_k=5, rerank_in=8,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        report = adapter.retrieve("q")
        # candidate_doc_ids fires on the candidate_k=10 pool.
        assert len(report.candidate_doc_ids) >= 4
        assert report.dense_retrieval_ms is not None

    def test_pipeline_truncates_to_rerank_in(self):
        retriever = _StubRetriever(self._pool())
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=12, final_top_k=10, rerank_in=4,
            ),
            final_reranker=_StubReranker(),
        )
        report = adapter.retrieve("q")
        # rerank_in caps the input AFTER any MMR/cap (here both off).
        # The reranker reversed [c01..c04] into [c04, c03, c02, c01].
        ids = [c.chunk_id for c in report.results]
        assert ids == ["c04", "c03", "c02", "c01"]

    def test_title_cap_on_rerank_input(self):
        retriever = _StubRetriever(self._pool())
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=12, final_top_k=8, rerank_in=8,
                title_cap_rerank_input=2,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        report = adapter.retrieve("q")
        # title_cap=2 over Series-1 / Series-2 / Series-3 leaves
        # at most 2 of each. Series-1 has 7 chunks → 2 land; Series-2
        # has 2 → both land; Series-3 has 3 → 2 land. Total 6.
        # rerank_in=8 doesn't bite because input is already 6.
        assert len(report.results) <= 6

    def test_final_top_k_truncates_after_cap(self):
        retriever = _StubRetriever(self._pool())
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=12, final_top_k=3, rerank_in=8,
                title_cap_rerank_input=2,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        report = adapter.retrieve("q")
        assert len(report.results) == 3

    def test_mmr_runs_before_cap(self):
        retriever = _StubRetriever(self._pool())
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=12, final_top_k=4, rerank_in=4,
                use_mmr=True, mmr_lambda=0.5, mmr_k=8,
                title_cap_rerank_input=1, title_cap_final=1,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        report = adapter.retrieve("q")
        # cap_final=1 over titles → at most 3 unique titles in output.
        out_titles = {c.title for c in report.results}
        assert len(out_titles) == len(report.results)

    def test_retriever_state_restored_after_call(self):
        retriever = _StubRetriever(self._pool())
        retriever._top_k = 7
        retriever._candidate_k = 7
        retriever._use_mmr = False
        retriever._mmr_lambda = 0.7
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=12, final_top_k=5, rerank_in=8,
            ),
            final_reranker=_StubReranker(),
        )
        adapter.retrieve("q")
        # All four mutated attributes restored to pre-call.
        assert retriever._top_k == 7
        assert retriever._candidate_k == 7
        assert retriever._use_mmr is False
        assert retriever._mmr_lambda == 0.7


# ---------------------------------------------------------------------------
# 4. query_type heuristic
# ---------------------------------------------------------------------------


class TestQueryTypeHeuristic:
    def test_relation_keyword_routes(self):
        tag = tag_query("주인공과 라이벌의 관계는 어떻게 되나요?")
        assert tag.query_type == QT_CHARACTER_RELATION

    def test_attribute_keyword_routes(self):
        tag = tag_query("이 캐릭터의 능력과 성격은 무엇인가요?")
        assert tag.query_type == QT_CHARACTER_ATTRIBUTE

    def test_plot_event_routes(self):
        tag = tag_query("작품의 줄거리를 알려주세요.")
        assert tag.query_type == QT_PLOT_EVENT

    def test_setting_routes(self):
        tag = tag_query("이 작품의 세계관과 조직 구조를 설명해 주세요.")
        assert tag.query_type == QT_SETTING

    def test_review_reception_routes(self):
        tag = tag_query("이 애니메이션의 평가와 흥행 성적은?")
        assert tag.query_type == QT_REVIEW_RECEPTION

    def test_comparison_routes(self):
        tag = tag_query("두 작품의 차이를 비교해 주세요.")
        assert tag.query_type == QT_COMPARISON

    def test_title_direct_routes(self):
        tag = tag_query("나루토에 대해 설명해 주세요.")
        assert tag.query_type == QT_TITLE_DIRECT

    def test_short_query_is_ambiguous(self):
        # Even though "관계" matches, the query is too short.
        tag = tag_query("관계")
        assert tag.query_type == QT_AMBIGUOUS

    def test_unknown_when_no_keyword_hits(self):
        tag = tag_query("어떤 작품인가요?")
        # No specific keyword in our table fires for this generic
        # phrasing → unknown bucket.
        assert tag.query_type in (QT_UNKNOWN, QT_TITLE_DIRECT)
        # Confidence stays low.
        assert tag.confidence < 0.7

    def test_tag_rows_preserves_count_and_fields(self):
        rows = [
            {"id": "q1", "query": "주인공과 적의 관계?",
             "expected_doc_ids": ["d1"]},
            {"id": "q2", "query": "스토리 결말은?", "language": "ko"},
            {"id": "q3", "query": ""},
        ]
        out = tag_rows(rows)
        assert len(out) == len(rows)
        # All original fields preserved.
        assert out[0]["expected_doc_ids"] == ["d1"]
        assert out[1]["language"] == "ko"
        # New fields injected.
        for row in out:
            assert "query_type" in row
            assert "query_type_confidence" in row
            assert "query_type_reason" in row
        # Empty query → unknown with confidence 0.
        assert out[2]["query_type"] == QT_UNKNOWN
        assert out[2]["query_type_confidence"] == 0.0

    def test_summarize_distribution_counts(self):
        rows = [
            {"query_type": QT_CHARACTER_RELATION, "query_type_confidence": 0.7,
             "query_type_reason": "matched 'sample'"},
            {"query_type": QT_PLOT_EVENT, "query_type_confidence": 0.7,
             "query_type_reason": "matched 'sample'"},
            {"query_type": QT_UNKNOWN, "query_type_confidence": 0.1,
             "query_type_reason": "no heuristic matched"},
            {"query_type": QT_AMBIGUOUS, "query_type_confidence": 0.2,
             "query_type_reason": "competing labels"},
        ]
        summary = summarize_distribution(rows)
        assert summary["total_rows"] == 4
        assert summary["per_type"][QT_CHARACTER_RELATION]["count"] == 1
        # Two rows below 0.5 confidence threshold.
        assert summary["low_confidence_count"] == 2
        # One row mentions "competing".
        assert summary["competing_count"] == 1


class TestWriteDraftJsonl:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        rows = [
            {"id": "q1", "query": "관계는?", "query_type": QT_CHARACTER_RELATION,
             "query_type_confidence": 0.7,
             "query_type_reason": "matched '관계'"},
        ]
        out = tmp_path / "draft.jsonl"
        write_draft_jsonl(rows, out)
        assert out.exists()
        with out.open("r", encoding="utf-8") as fp:
            line = fp.readline().strip()
        loaded = json.loads(line)
        assert loaded["id"] == "q1"
        assert loaded["query_type"] == QT_CHARACTER_RELATION


# ---------------------------------------------------------------------------
# 5. DocTitleResolver
# ---------------------------------------------------------------------------


class TestDocTitleResolver:
    def test_loads_titles_from_corpus(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        rows = [
            {"doc_id": "d1", "title": "Title-One", "sections": {}},
            {"doc_id": "d2", "title": "Title-Two", "sections": {}},
            {"seed": "d3", "title": "Title-Three", "sections": {}},
        ]
        with corpus.open("w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        resolver = DocTitleResolver.from_corpus(corpus)
        assert resolver.title_for_doc("d1") == "Title-One"
        assert resolver.title_for_doc("d2") == "Title-Two"
        assert resolver.title_for_doc("d3") == "Title-Three"
        assert resolver.title_for_doc("missing") is None

    def test_provider_returns_none_for_missing(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("", encoding="utf-8")
        resolver = DocTitleResolver.from_corpus(corpus)
        provider = resolver.title_provider()
        chunk = SimpleNamespace(doc_id="missing")
        assert provider(chunk) is None

    def test_handles_missing_corpus_file(self, tmp_path: Path):
        # Resolver must not crash when corpus file doesn't exist —
        # this happens in tests that don't have the production corpus
        # but want to instantiate the adapter with a default provider.
        resolver = DocTitleResolver.from_corpus(tmp_path / "nope.jsonl")
        assert resolver.title_for_doc("anything") is None
