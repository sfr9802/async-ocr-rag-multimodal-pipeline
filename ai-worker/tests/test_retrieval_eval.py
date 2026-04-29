"""Tests for the retrieval-eval harness.

Three layers, all fully offline:

  1. New metric unit tests (reciprocal_rank_at_k, ndcg_at_k,
     unique_doc_coverage, top1_score_margin, count_whitespace_tokens,
     expected_keyword_match_rate, normalized_text_hash).

  2. End-to-end harness smoke against an in-memory HashingEmbedder +
     FAISS index + fake metadata store — same pattern as the existing
     test_eval_harness.py.

  3. Deterministic generator smoke: target ratios obeyed, no key
     fabricated outside the source corpus, output schema valid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List

import pytest

from eval.harness import (
    DuplicateAnalysis,
    RetrievalEvalRow,
    RetrievalEvalSummary,
    TopKDumpRow,
    count_whitespace_tokens,
    expected_keyword_match_rate,
    ndcg_at_k,
    normalized_text_hash,
    reciprocal_rank_at_k,
    render_markdown_report,
    run_retrieval_eval,
    top1_score_margin,
    unique_doc_coverage,
)


# ---------------------------------------------------------------------------
# 1. Metric unit tests.
# ---------------------------------------------------------------------------


class TestReciprocalRankAtK:
    def test_first_position_is_one(self):
        assert reciprocal_rank_at_k(["gold", "x", "y"], ["gold"], k=10) == 1.0

    def test_third_position_is_one_third(self):
        assert reciprocal_rank_at_k(["x", "y", "gold"], ["gold"], k=10) == pytest.approx(1 / 3)

    def test_cutoff_drops_late_hit(self):
        # gold is at rank 5 — cutoff k=3 → 0.0
        assert reciprocal_rank_at_k(["a", "b", "c", "d", "gold"], ["gold"], k=3) == 0.0

    def test_no_hit_is_zero(self):
        assert reciprocal_rank_at_k(["a", "b", "c"], ["gold"], k=10) == 0.0

    def test_empty_expected_is_none(self):
        assert reciprocal_rank_at_k(["a"], [], k=10) is None

    def test_normalizes_doc_ids(self):
        # Different unicode width / casing should still match.
        assert reciprocal_rank_at_k(["ＧＯＬＤ"], ["gold"], k=10) == 1.0


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["gold"], ["gold"], k=5) == 1.0

    def test_ranked_below_top_drops_below_one(self):
        # gold at rank 2: dcg = 1/log2(3); idcg (1 gold, k=5) = 1/log2(2) = 1.0
        v = ndcg_at_k(["x", "gold", "y"], ["gold"], k=5)
        assert v == pytest.approx(1 / math.log2(3))

    def test_idcg_uses_min_of_gold_and_k(self):
        # 3 golds, k=2 → idcg over 2 positions only
        v = ndcg_at_k(["g1", "g2", "g3"], ["g1", "g2", "g3"], k=2)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, 3))
        dcg = sum(1 / math.log2(i + 1) for i in range(1, 3))
        assert v == pytest.approx(dcg / idcg)

    def test_no_gold_in_topk_is_zero(self):
        assert ndcg_at_k(["a", "b"], ["gold"], k=2) == 0.0

    def test_empty_expected_is_none(self):
        assert ndcg_at_k(["a"], [], k=5) is None

    def test_duplicate_gold_in_topk_credited_once(self):
        # gold appearing twice in top-k must not double-count
        v = ndcg_at_k(["gold", "gold", "x"], ["gold"], k=3)
        assert v == 1.0


class TestUniqueDocCoverage:
    def test_all_distinct(self):
        assert unique_doc_coverage(["a", "b", "c", "d", "e"], k=5) == 1.0

    def test_all_same_is_one_over_k(self):
        assert unique_doc_coverage(["a"] * 5, k=5) == pytest.approx(0.2)

    def test_empty_is_none(self):
        assert unique_doc_coverage([], k=5) is None


class TestTop1ScoreMargin:
    def test_basic_positive(self):
        v = top1_score_margin([0.9, 0.7, 0.5])
        assert v is not None and v == pytest.approx(0.2)

    def test_too_short_returns_none(self):
        assert top1_score_margin([0.5]) is None
        assert top1_score_margin([]) is None

    def test_negative_margin_allowed(self):
        # Reranker reordered such that rank-1 < rank-2 raw bi-encoder
        v = top1_score_margin([0.5, 0.6])
        assert v == pytest.approx(-0.1)


class TestCountWhitespaceTokens:
    def test_basic(self):
        assert count_whitespace_tokens("hello world foo") == 3

    def test_empty(self):
        assert count_whitespace_tokens("") == 0

    def test_multiple_whitespace(self):
        assert count_whitespace_tokens("a   b\nc\td") == 4


class TestExpectedKeywordMatchRate:
    def test_full_match_across_chunks(self):
        chunks = ["The bookshop is closed.", "Run by a translator."]
        assert expected_keyword_match_rate(chunks, ["bookshop", "translator"]) == 1.0

    def test_partial_match(self):
        chunks = ["Only bookshop.", "No second word."]
        assert expected_keyword_match_rate(chunks, ["bookshop", "missing"]) == 0.5

    def test_zero_match(self):
        assert expected_keyword_match_rate(["nothing"], ["alpha", "beta"]) == 0.0

    def test_none_when_empty_keywords(self):
        assert expected_keyword_match_rate(["text"], []) is None

    def test_zero_when_empty_chunks(self):
        # Keywords exist, but no chunk text — clean zero.
        assert expected_keyword_match_rate([], ["x"]) == 0.0


class TestNormalizedTextHash:
    def test_case_insensitive(self):
        assert normalized_text_hash("Hello") == normalized_text_hash("hello")

    def test_whitespace_collapse(self):
        assert normalized_text_hash("foo  bar") == normalized_text_hash("foo bar")

    def test_unicode_width(self):
        # Full-width ASCII should normalize to half-width via NFKC.
        assert normalized_text_hash("ＡＢＣ") == normalized_text_hash("abc")

    def test_empty_is_empty_string(self):
        assert normalized_text_hash("") == ""
        assert normalized_text_hash("   ") == ""

    def test_different_text_different_hash(self):
        a = normalized_text_hash("the quick brown fox")
        b = normalized_text_hash("the lazy dog")
        assert a != b
        assert len(a) == 16


# ---------------------------------------------------------------------------
# 2. End-to-end harness smoke (in-memory FAISS + HashingEmbedder).
# ---------------------------------------------------------------------------


def _build_in_memory_retriever(tmp_path: Path):
    """Tiny real Retriever against an in-memory FAISS index.

    Same pattern as the existing test_eval_harness.py — keeps these
    tests offline + fast.
    """
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.metadata_store import ChunkLookupResult
    from app.capabilities.rag.retriever import Retriever

    passages = [
        ("c1", "doc-book", "overview",
         "A retired translator runs a secondhand bookshop on a dying railway line."),
        ("c2", "doc-book", "overview",
         "The bookshop has tea and old translations the translator curates herself."),
        ("c3", "doc-cats", "overview",
         "An elderly fisherman feeds the stray cats of a small harbor every morning."),
        ("c4", "doc-mech", "plot",
         "Ironclad Academy students pilot construction mechs to reinforce a coastal dam."),
        ("c5", "doc-aoi", "overview",
         "Aoi tends luminescent gardens suspended above the clouds."),
    ]
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed_passages([p[3] for p in passages])
    index = FaissIndex(tmp_path / "idx")
    index.build(vectors, index_version="ret-test-v1", embedding_model=embedder.model_name)

    rows = [
        ChunkLookupResult(
            chunk_id=p[0], doc_id=p[1], section=p[2], text=p[3], faiss_row_id=i
        )
        for i, p in enumerate(passages)
    ]

    class _FakeMetadataStore:
        def __init__(self, version: str, rows: list):
            self._version = version
            self._by_row = {r.faiss_row_id: r for r in rows}

        def lookup_chunks_by_faiss_rows(
            self, index_version: str, faiss_row_ids: Iterable[int]
        ):
            assert index_version == self._version
            return [self._by_row[i] for i in faiss_row_ids if i in self._by_row]

    metadata = _FakeMetadataStore("ret-test-v1", rows)
    retriever = Retriever(embedder=embedder, index=index, metadata=metadata, top_k=5)
    retriever.ensure_ready()
    return retriever


def test_harness_end_to_end_emits_four_payloads(tmp_path: Path):
    retriever = _build_in_memory_retriever(tmp_path)
    dataset = [
        {
            "id": "test-001",
            "query": "who runs the bookshop?",
            "language": "en",
            "expected_doc_ids": ["doc-book"],
            "expected_section_keywords": ["bookshop", "translator"],
            "answer_type": "character_relation",
            "difficulty": "easy",
            "tags": ["test", "smoke"],
        },
        {
            "id": "test-002",
            "query": "fisherman feeding harbor cats",
            "expected_doc_ids": ["doc-cats"],
            "expected_section_keywords": ["fisherman", "cats"],
            "answer_type": "summary_plot",
            "difficulty": "easy",
            "tags": ["test"],
        },
    ]

    summary, rows, dump, dup = run_retrieval_eval(
        dataset,
        retriever=retriever,
        top_k=5,
        mrr_k=10,
        ndcg_k=10,
        dataset_path="test://in-memory",
    )

    # Summary shape.
    assert isinstance(summary, RetrievalEvalSummary)
    assert summary.row_count == 2
    assert summary.error_count == 0
    assert summary.rows_with_expected_doc_ids == 2
    assert summary.top_k == 5 and summary.mrr_k == 10 and summary.ndcg_k == 10
    assert summary.embedding_model == "hashing-embedder-dim64"
    assert summary.index_version == "ret-test-v1"
    # Hashing embedder generally lands the bookshop query — accept any
    # non-None aggregation rather than asserting exact values, since
    # HashingEmbedder math is not what we're testing here.
    assert summary.mean_hit_at_5 is not None
    assert summary.mean_mrr_at_10 is not None
    assert summary.mean_ndcg_at_10 is not None

    # Per-row shape.
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, RetrievalEvalRow)
        assert row.id and row.query
        assert row.retrieved_doc_ids, "retrieved_doc_ids should not be empty"
        assert row.hit_at_1 in (0.0, 1.0)
        assert row.hit_at_3 in (0.0, 1.0)
        assert row.hit_at_5 in (0.0, 1.0)
        assert row.dup_rate is not None
        assert row.unique_doc_coverage is not None
        # avg_context_token_count defined when chunks were retrieved.
        assert row.avg_context_token_count is not None
        # expected_keyword_match_rate defined when row had keywords.
        assert row.expected_keyword_match_rate is not None

    # Top-k dump: one record per (query, rank) pair, capped at top_k.
    assert len(dump) <= 2 * 5
    for d in dump:
        assert isinstance(d, TopKDumpRow)
        assert d.query_id and d.query
        assert 1 <= d.rank <= 5
        assert d.doc_id and d.chunk_id
        assert d.normalized_score is not None
        assert 0.0 <= d.normalized_score <= 1.0
        assert isinstance(d.is_expected_doc, bool)
        # chunk_preview is bounded
        assert len(d.chunk_preview) <= 161  # PREVIEW_CHARS + ellipsis

    # Duplicate analysis exists, ratios in [0, 1].
    assert isinstance(dup, DuplicateAnalysis)
    assert dup.queries_evaluated == 2
    assert 0.0 <= dup.queries_with_doc_dup_ratio <= 1.0
    assert 0.0 <= dup.queries_with_section_dup_ratio <= 1.0
    assert 0.0 <= dup.queries_with_text_dup_ratio <= 1.0


def test_top_k_dump_marks_expected_doc(tmp_path: Path):
    retriever = _build_in_memory_retriever(tmp_path)
    dataset = [
        {
            "id": "exp-test",
            "query": "translator bookshop",
            "expected_doc_ids": ["doc-book"],
            "expected_section_keywords": ["bookshop"],
            "answer_type": "title_lookup",
        }
    ]
    _, _, dump, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    # At least one dump row for doc-book should have is_expected_doc=True.
    expected_rows = [d for d in dump if d.doc_id == "doc-book"]
    if expected_rows:  # hashing may or may not surface it; skip if not
        assert any(d.is_expected_doc for d in expected_rows)
        # And the matched_expected_keyword field carries the substring hits.
        assert any("bookshop" in (d.matched_expected_keyword or []) for d in expected_rows)


def test_harness_records_error_without_raising(tmp_path: Path):
    """A retriever that always raises must not abort the harness."""

    class _BoomRetriever:
        def retrieve(self, query: str):
            raise RuntimeError("simulated failure")

    dataset = [
        {"id": "boom-1", "query": "anything", "expected_doc_ids": ["x"]},
        {"id": "boom-2", "query": "another",  "expected_doc_ids": ["y"]},
    ]
    summary, rows, dump, dup = run_retrieval_eval(
        dataset, retriever=_BoomRetriever(), top_k=5,
    )
    assert summary.error_count == 2
    assert summary.row_count == 2
    assert all(r.error and "RuntimeError" in r.error for r in rows)
    # No dump rows since nothing retrieved successfully.
    assert dump == []
    # Duplicate analysis still safely emitted.
    assert dup.queries_evaluated == 0
    assert dup.queries_with_doc_dup_ratio == 0.0


def test_markdown_report_renders(tmp_path: Path):
    retriever = _build_in_memory_retriever(tmp_path)
    dataset = [
        {
            "id": "md-1",
            "query": "translator bookshop",
            "expected_doc_ids": ["doc-book"],
            "expected_section_keywords": ["bookshop"],
            "answer_type": "title_lookup",
            "difficulty": "easy",
        }
    ]
    summary, rows, _, dup = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    md = render_markdown_report(summary, rows, dup)
    assert "# Retrieval eval report" in md
    assert "## Headline metrics" in md
    assert "| hit@1 |" in md
    assert "## Latency (ms)" in md
    assert "## Per answer_type" in md
    assert "## Per difficulty" in md
    assert "## Duplicate analysis" in md


# ---------------------------------------------------------------------------
# 3. Deterministic synthetic generator.
# ---------------------------------------------------------------------------


def _write_tiny_corpus(tmp_path: Path) -> Path:
    """Write a 3-doc anime corpus that exercises all builders."""
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "doc_id": "doc-001",
            "title": "신비한 모험의 책",
            "summary": "주인공 미아와 로엔은 마법의 책을 찾아 떠나는 모험 이야기입니다.",
            "summary_bullets": [
                "미아는 마법사 견습생이다.",
                "로엔은 미아의 친구이자 동료이다.",
                "두 사람은 환상적인 세계관에서 활약한다.",
            ],
            "sections": {
                "요약": {
                    "text": "주인공 미아와 로엔은 마법의 책을 찾아 떠나는 모험 이야기입니다.",
                    "chunks": ["주인공 미아와 로엔은 마법의 책을 찾아 떠나는 모험 이야기입니다."],
                },
                "본문": {
                    "text": "이야기의 본문에는 미아의 마법 수련과 로엔의 검술 훈련이 자세히 묘사된다. 두 인물은 협력하여 다양한 위기를 극복한다.",
                    "chunks": [
                        "이야기의 본문에는 미아의 마법 수련과 로엔의 검술 훈련이 자세히 묘사된다.",
                        "두 인물은 협력하여 다양한 위기를 극복하며, 마법의 책의 비밀을 풀어간다.",
                    ],
                },
                "등장인물": {
                    "text": "미아: 마법사 견습생.\n로엔: 검사 출신의 동료.",
                    "chunks": ["미아: 마법사 견습생.", "로엔: 검사 출신의 동료."],
                },
            },
            "section_order": ["요약", "본문", "등장인물"],
        },
        {
            "doc_id": "doc-002",
            "title": "별들의 항해사",
            "summary": "우주 항해사 카일이 잃어버린 항성을 찾아 떠나는 SF 이야기.",
            "summary_bullets": [
                "카일은 우주 항해사이다.",
                "잃어버린 항성을 찾는 임무를 수행한다.",
            ],
            "sections": {
                "요약": {
                    "text": "우주 항해사 카일이 잃어버린 항성을 찾아 떠나는 SF 이야기.",
                    "chunks": ["우주 항해사 카일이 잃어버린 항성을 찾아 떠나는 SF 이야기."],
                },
                "본문": {
                    "text": "본문은 우주선의 운항과 항성계의 묘사로 가득하다.",
                    "chunks": ["본문은 우주선의 운항과 항성계의 묘사로 가득하다."],
                },
                "설정": {
                    "text": "은하 연방은 12개의 항성계로 구성되며, 항해사는 특수 자격을 갖춘 자만이 될 수 있다.",
                    "chunks": ["은하 연방은 12개의 항성계로 구성되며, 항해사는 특수 자격을 갖춘 자만이 될 수 있다."],
                },
            },
            "section_order": ["요약", "본문", "설정"],
        },
        {
            "doc_id": "doc-003",
            "title": "조용한 숲의 정원사",
            "summary": "정원사 하나가 숲의 비밀을 발견하는 잔잔한 이야기.",
            "summary_bullets": ["하나는 정원사이다.", "숲의 비밀을 우연히 발견한다."],
            "sections": {
                "요약": {
                    "text": "정원사 하나가 숲의 비밀을 발견하는 잔잔한 이야기.",
                    "chunks": ["정원사 하나가 숲의 비밀을 발견하는 잔잔한 이야기."],
                },
                "본문": {
                    "text": "정원사 하나는 매일 숲을 가꾼다.",
                    "chunks": ["정원사 하나는 매일 숲을 가꾼다."],
                },
            },
            "section_order": ["요약", "본문"],
        },
    ]
    with corpus.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return corpus


def test_deterministic_generator_emits_valid_schema(tmp_path: Path):
    from eval.harness.generate_eval_queries import (
        ANSWER_TYPES,
        DIFFICULTY_BY_TYPE,
        _iter_corpus,
        generate_deterministic,
    )

    corpus_path = _write_tiny_corpus(tmp_path)
    docs = list(_iter_corpus(corpus_path))
    assert len(docs) == 3

    rows = generate_deterministic(docs, target=12, seed=42)
    # Some rows must be produced — the bigger source signal types
    # (summary_plot, title_lookup, theme_genre) are guaranteed by the
    # tiny corpus.
    assert len(rows) > 0
    # Schema check.
    valid_types = set(ANSWER_TYPES)
    for r in rows:
        assert set(r.keys()) >= {
            "id", "query", "language", "expected_doc_ids",
            "expected_section_keywords", "answer_type", "difficulty", "tags",
        }
        assert r["language"] == "ko"
        assert r["answer_type"] in valid_types
        assert r["difficulty"] == DIFFICULTY_BY_TYPE[r["answer_type"]]
        assert r["expected_doc_ids"] and r["expected_doc_ids"][0]
        assert r["expected_section_keywords"]
        assert "synthetic" in r["tags"]


def test_deterministic_generator_is_seed_stable(tmp_path: Path):
    from eval.harness.generate_eval_queries import _iter_corpus, generate_deterministic

    corpus_path = _write_tiny_corpus(tmp_path)
    docs = list(_iter_corpus(corpus_path))
    rows_a = generate_deterministic(docs, target=10, seed=123)
    rows_b = generate_deterministic(docs, target=10, seed=123)
    assert rows_a == rows_b

    rows_c = generate_deterministic(docs, target=10, seed=456)
    # Different seed produces a different ordering / row mix.
    assert rows_a != rows_c


# ---------------------------------------------------------------------------
# 4. Phase 1 metric / diagnostics extensions.
#
# These tests pin the new candidate / pre-rerank / diversity / quality-score
# / query-type / diagnostics behavior. Each test maps to one of the 12
# acceptance criteria called out in the work-plan; the docstring lists which.
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _phase1_dataclass
from typing import Optional as _Phase1Optional

from eval.harness import (
    DEFAULT_CANDIDATE_KS,
    DEFAULT_QUERY_TYPE_UNKNOWN,
    DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50,
    DIAG_HIGH_DUPLICATE_RATIO_AT_10,
    DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5,
    DIAG_RERANKER_NEGATIVE_UPLIFT_MRR_AT_10,
    DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5,
    QUALITY_SCORE_WEIGHT_HIT_AT_1,
    QUALITY_SCORE_WEIGHT_HIT_AT_5,
    QUALITY_SCORE_WEIGHT_MRR,
    QUALITY_SCORE_WEIGHT_NDCG,
    compute_retrieval_diagnostics,
    duplicate_doc_ratio_at_k,
    efficiency_score,
    quality_score,
    section_diversity_at_k,
    unique_doc_count_at_k,
)


# --- helper: minimal fake retriever that emulates the production shape ---


@_phase1_dataclass
class _PhaseChunk:
    """Stand-in for ``RetrievedChunk``.

    Mirrors the duck-typed contract ``run_retrieval_eval`` reads off
    each result: ``doc_id``, ``chunk_id``, ``section``, ``text``,
    ``score``, ``rerank_score``. Section is required because the
    Phase 1 diversity / dup metrics walk over ``section_paths``.
    """

    doc_id: str
    chunk_id: str
    section: str
    text: str
    score: float
    rerank_score: _Phase1Optional[float] = None


@_phase1_dataclass
class _PhaseReport:
    """Stand-in for ``RetrievalReport``.

    Optional fields exposed: ``candidate_doc_ids`` for the wider
    pre-rerank pool, ``rerank_ms`` / ``dense_retrieval_ms`` for
    latency aggregation. Defaults match a NoOp reranker run so a
    test that doesn't touch reranker fields gets a backward-compat
    shape.
    """

    results: list
    candidate_doc_ids: _Phase1Optional[list] = None
    rerank_ms: _Phase1Optional[float] = None
    dense_retrieval_ms: _Phase1Optional[float] = None
    index_version: str = "phase1-v1"
    embedding_model: str = "phase1-embed"
    reranker_name: str = "phase1-rerank"


class _ScriptedRetriever:
    """Returns a per-query scripted ``_PhaseReport``."""

    def __init__(self, by_query):
        self._by_query = by_query

    def retrieve(self, query: str):
        if query not in self._by_query:
            raise KeyError(f"Unscripted query: {query!r}")
        return self._by_query[query]


def test_phase1_singular_expected_doc_id_normalizes_to_list():
    """1. ``expected_doc_id`` (singular) on the dataset row must
    surface as ``RetrievalEvalRow.expected_doc_ids`` containing the
    same id, so per-row hit/recall metrics still fire even on the
    older schema.
    """
    chunks = [
        _PhaseChunk("doc-A", "c1", "intro", "alpha", 0.9),
        _PhaseChunk("doc-B", "c2", "intro", "beta", 0.7),
    ]
    retriever = _ScriptedRetriever({
        "q1": _PhaseReport(results=chunks),
    })
    dataset = [{
        "id": "row-1",
        "query": "q1",
        # singular field only — no expected_doc_ids list
        "expected_doc_id": "doc-A",
    }]
    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    row = rows[0]
    assert row.expected_doc_ids == ["doc-A"]
    assert row.hit_at_1 == 1.0
    assert summary.rows_with_expected_doc_ids == 1


def test_phase1_candidate_recall_with_multiple_expected_docs():
    """2. ``expected_doc_ids`` with multiple gold ids: candidate recall
    must reflect ``matched / |gold|`` over the candidate pool, not the
    final top-k.
    """
    chunks = [_PhaseChunk(f"doc-final-{i}", f"c{i}", "s", "t", 0.5) for i in range(3)]
    candidate_pool = [
        # Two of the three expected docs surface in the candidate pool
        # at ranks 1 and 4. The third is missed entirely → recall@10
        # = 2/3.
        "doc-gold-1",
        "doc-other-1",
        "doc-other-2",
        "doc-gold-2",
        "doc-other-3",
    ]
    retriever = _ScriptedRetriever({
        "q1": _PhaseReport(results=chunks, candidate_doc_ids=candidate_pool),
    })
    dataset = [{
        "id": "row-1",
        "query": "q1",
        "expected_doc_ids": ["doc-gold-1", "doc-gold-2", "doc-gold-3"],
    }]
    _, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    row = rows[0]
    assert row.candidate_recalls.get("10") == pytest.approx(2 / 3)
    # Also: candidate_hit@10 fires because at least one gold landed.
    assert row.candidate_hits.get("10") == 1.0


def test_phase1_candidate_metrics_separate_from_final():
    """3. Candidate hit@K reads from the candidate pool; final hit@K
    reads from the final top-k. The two MUST diverge when the candidate
    pool contains the gold doc but the final top-k doesn't.
    """
    final_chunks = [_PhaseChunk("doc-other", "c1", "s", "t", 0.9)]
    candidate_pool = ["doc-other", "doc-other2", "doc-gold"]
    retriever = _ScriptedRetriever({
        "q1": _PhaseReport(results=final_chunks, candidate_doc_ids=candidate_pool),
    })
    dataset = [{
        "id": "row-1",
        "query": "q1",
        "expected_doc_ids": ["doc-gold"],
    }]
    _, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    row = rows[0]
    # Final top-k missed the gold → hit@5 == 0
    assert row.hit_at_5 == 0.0
    # Candidate pool contained the gold at rank 3 → hit@10 == 1
    assert row.candidate_hits.get("10") == 1.0


def test_phase1_rerank_uplift_computed_from_pre_vs_final():
    """4. With reranker pulling the gold from rank 5 to rank 1:
    pre_rerank_hit_at_1 == 0, final hit_at_1 == 1, uplift = +1.
    """
    chunks = [
        # Final order: gold first (reranker did its job)
        _PhaseChunk("doc-gold", "g", "s", "t", 0.40, rerank_score=0.95),
        _PhaseChunk("doc-A", "a", "s", "t", 0.55, rerank_score=0.55),
        _PhaseChunk("doc-B", "b", "s", "t", 0.50, rerank_score=0.40),
        _PhaseChunk("doc-C", "c", "s", "t", 0.45, rerank_score=0.30),
        _PhaseChunk("doc-D", "d", "s", "t", 0.42, rerank_score=0.20),
    ]
    retriever = _ScriptedRetriever({"q1": _PhaseReport(results=chunks)})
    dataset = [{
        "id": "row-1",
        "query": "q1",
        "expected_doc_ids": ["doc-gold"],
    }]
    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    row = rows[0]
    # Pre-rerank order = sorted by raw dense score:
    #   doc-A (0.55), doc-B (0.50), doc-C (0.45), doc-D (0.42), doc-gold (0.40)
    # so pre_rerank_hit_at_1 = 0 (gold at rank 5).
    assert row.pre_rerank_hit_at_1 == 0.0
    assert row.hit_at_1 == 1.0
    assert summary.mean_pre_rerank_hit_at_1 == 0.0
    assert summary.mean_hit_at_1 == 1.0
    assert summary.rerank_uplift_hit_at_1 == pytest.approx(1.0)


def test_phase1_negative_uplift_flags_diagnostic():
    """5. When reranker hurts hit@5 (final < pre by more than the
    epsilon), ``diagnostics.rerankerNegativeUplift.flagged`` must be
    True.
    """
    # Build 6 rows: dense ordering would put gold in top-5; rerank
    # demotes gold below top-5.
    queries = {}
    dataset = []
    for i in range(6):
        chunks = [
            _PhaseChunk("doc-A", f"a{i}", "s", "t", 0.90, rerank_score=0.90),
            _PhaseChunk("doc-B", f"b{i}", "s", "t", 0.85, rerank_score=0.85),
            _PhaseChunk("doc-C", f"c{i}", "s", "t", 0.80, rerank_score=0.80),
            _PhaseChunk("doc-D", f"d{i}", "s", "t", 0.75, rerank_score=0.75),
            _PhaseChunk("doc-E", f"e{i}", "s", "t", 0.70, rerank_score=0.70),
            _PhaseChunk("doc-gold", f"g{i}", "s", "t", 0.65, rerank_score=0.10),
        ]
        # Final order matches rerank score → gold last (rank 6)
        # Pre-rerank order = sorted by score → gold also at rank 6 here,
        # which would mean uplift = 0. Instead, swap so dense puts gold
        # at rank 5 (above doc-E):
        chunks[5].score = 0.71  # gold dense > doc-E dense (0.70)
        # Final order is the list as-is (which reflects rerank ordering).
        queries[f"q{i}"] = _PhaseReport(results=chunks)
        dataset.append({
            "id": f"row-{i}",
            "query": f"q{i}",
            "expected_doc_ids": ["doc-gold"],
        })
    retriever = _ScriptedRetriever(queries)
    summary, _, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=10,
    )
    # Pre-rerank hit@5 should be 1.0 across all 6 rows; final hit@5
    # should be 0.0 → uplift = -1.0, well past the negative threshold.
    assert summary.mean_pre_rerank_hit_at_5 == pytest.approx(1.0)
    assert summary.mean_hit_at_5 == 0.0
    assert summary.rerank_uplift_hit_at_5 == pytest.approx(-1.0)
    diag = summary.diagnostics
    assert diag["rerankerNegativeUplift"]["flagged"] is True


def test_phase1_query_type_missing_falls_back_to_unknown():
    """6. Rows without ``query_type`` land in the ``unknown`` bucket of
    the by_query_type breakdown so they remain visible.
    """
    chunks = [_PhaseChunk("doc-X", "c", "s", "t", 0.9)]
    retriever = _ScriptedRetriever({
        "q1": _PhaseReport(results=chunks),
        "q2": _PhaseReport(results=chunks),
    })
    dataset = [
        {"id": "r1", "query": "q1", "expected_doc_ids": ["doc-X"]},
        {"id": "r2", "query": "q2", "expected_doc_ids": ["doc-X"],
         "query_type": "title_direct"},
    ]
    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    bt = summary.by_query_type
    assert DEFAULT_QUERY_TYPE_UNKNOWN in bt
    assert "title_direct" in bt
    assert bt[DEFAULT_QUERY_TYPE_UNKNOWN]["count"] == 1
    assert bt["title_direct"]["count"] == 1
    # And the per-row column on the unknown row stays None — the
    # source-of-truth is "the dataset didn't say".
    assert rows[0].query_type is None
    assert rows[1].query_type == "title_direct"


def test_phase1_query_type_breakdown_has_count_and_metrics():
    """7. byQueryType breakdown carries count + the spec-required fields
    (``hit_at_1``, ``hit_at_5``, ``mrr_at_10``, ``ndcg_at_10``,
    ``candidate_hit_at_50``, ``candidate_recall_at_50``,
    ``avg_total_retrieval_ms``, ``p95_total_retrieval_ms``,
    ``duplicate_doc_ratio_at_10``).
    """
    chunks = [_PhaseChunk("doc-G", "c", "s", "t", 0.9)]
    candidate_pool = ["doc-G"] + [f"doc-{i}" for i in range(60)]
    retriever = _ScriptedRetriever({
        "q1": _PhaseReport(results=chunks, candidate_doc_ids=candidate_pool),
    })
    dataset = [
        {"id": f"r{i}", "query": "q1", "expected_doc_ids": ["doc-G"],
         "query_type": "plot_event"}
        for i in range(7)
    ]
    summary, _, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5,
    )
    bucket = summary.by_query_type["plot_event"]
    expected_keys = {
        "count", "hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_10",
        "ndcg_at_10", "candidate_hit_at_50", "candidate_recall_at_50",
        "avg_total_retrieval_ms", "p95_total_retrieval_ms",
        "duplicate_doc_ratio_at_10",
    }
    assert expected_keys.issubset(set(bucket.keys()))
    assert bucket["count"] == 7
    assert bucket["hit_at_1"] == 1.0
    assert bucket["candidate_hit_at_50"] == 1.0


def test_phase1_duplicate_doc_ratio_at_k_with_repeated_docs():
    """8. ``duplicate_doc_ratio_at_k`` over a list with repeated doc_ids
    must equal ``1 - unique/k``.
    """
    # 5 results, 2 distinct doc_ids → dup ratio = 1 - 2/5 = 0.6
    assert duplicate_doc_ratio_at_k(["a", "a", "a", "b", "b"], k=5) == pytest.approx(0.6)
    # All same → 1 - 1/5 = 0.8
    assert duplicate_doc_ratio_at_k(["a"] * 5, k=5) == pytest.approx(0.8)
    # All distinct → 0
    assert duplicate_doc_ratio_at_k(["a", "b", "c", "d", "e"], k=5) == pytest.approx(0.0)
    assert unique_doc_count_at_k(["a", "a", "b", "c", "c"], k=5) == 3
    # Empty input: None (metric undefined)
    assert duplicate_doc_ratio_at_k([], k=5) is None
    assert unique_doc_count_at_k([], k=5) is None


def test_phase1_section_diversity_safe_when_sections_missing():
    """9. ``section_diversity_at_k`` returns ``None`` (not 0) when no
    section info was populated — the harness must skip the metric
    rather than zero it out.
    """
    assert section_diversity_at_k([None, None, None], k=3) is None
    assert section_diversity_at_k(["", "", ""], k=3) is None
    # Mixed: 2 distinct non-empty sections out of 3 → 2/3
    assert section_diversity_at_k(["a", "b", None], k=3) == pytest.approx(2 / 3)


def test_phase1_quality_score_matches_definition():
    """10. ``quality_score`` must equal
    ``0.30·hit@1 + 0.25·hit@5 + 0.25·MRR + 0.20·NDCG``.
    """
    assert (
        QUALITY_SCORE_WEIGHT_HIT_AT_1
        + QUALITY_SCORE_WEIGHT_HIT_AT_5
        + QUALITY_SCORE_WEIGHT_MRR
        + QUALITY_SCORE_WEIGHT_NDCG
    ) == pytest.approx(1.0)
    score = quality_score(
        hit_at_1=0.5, hit_at_5=0.7, mrr_at_10=0.65, ndcg_at_10=0.68,
    )
    expected = 0.30 * 0.5 + 0.25 * 0.7 + 0.25 * 0.65 + 0.20 * 0.68
    assert score == pytest.approx(expected, rel=1e-6)
    # Any None input → None composite
    assert quality_score(hit_at_1=None, hit_at_5=0.7, mrr_at_10=0.5, ndcg_at_10=0.5) is None


def test_phase1_efficiency_score_safe_when_p95_missing_or_zero():
    """11. Efficiency score must not raise when p95 is missing or
    non-positive — it returns ``None`` instead.
    """
    assert efficiency_score(0.5, None) is None
    assert efficiency_score(0.5, 0.0) is None
    assert efficiency_score(0.5, -1.0) is None
    assert efficiency_score(None, 10.0) is None
    # Positive p95 → finite value with the natural-log denominator
    v = efficiency_score(0.65, 16.5)
    assert v is not None
    assert v == pytest.approx(0.65 / math.log(1 + 16.5), rel=1e-6)


def test_phase1_legacy_report_still_renders_without_phase1_data():
    """12. A run that produced no candidate pool / no rerank scores
    must still render through ``render_markdown_report`` cleanly. The
    Phase 1 sections silently skip; the legacy headline + latency +
    duplicate-analysis sections remain in their original shape so an
    older reader sees a byte-compatible (apart from new tail sections)
    report.
    """
    retriever = _build_in_memory_retriever(Path("/tmp")) if False else None
    # Use a stub retriever that returns 1 chunk per query, no candidate
    # pool, no rerank_score, no dense_retrieval_ms — purely Phase 0 shape.

    class _LegacyChunk:
        def __init__(self, doc_id, chunk_id, section, text, score):
            self.doc_id = doc_id
            self.chunk_id = chunk_id
            self.section = section
            self.text = text
            self.score = score
            self.rerank_score = None

    class _LegacyReport:
        def __init__(self, results):
            self.results = results
            self.index_version = "legacy-v1"
            self.embedding_model = "legacy-embed"
            self.reranker_name = "noop"
            # No candidate_doc_ids, no rerank_ms, no dense_retrieval_ms.

    class _LegacyRetriever:
        def retrieve(self, query):
            return _LegacyReport([
                _LegacyChunk("doc-X", "c1", "s", "alpha beta", 0.9),
            ])

    dataset = [
        {"id": "r1", "query": "anything", "expected_doc_ids": ["doc-X"]},
    ]
    summary, rows, _, dup = run_retrieval_eval(
        dataset, retriever=_LegacyRetriever(), top_k=5,
    )
    # Headline metrics intact.
    assert summary.mean_hit_at_5 is not None
    # Phase 1 candidate metrics absent (no candidate pool surfaced).
    assert summary.candidate_hit_rates.get("10") is None
    # Pre-rerank metrics equal final metrics (no rerank scores → uplift 0).
    assert summary.rerank_uplift_hit_at_5 == pytest.approx(0.0)
    # Markdown renderer must not raise; legacy sections still present.
    md = render_markdown_report(summary, rows, dup)
    assert "## Headline metrics" in md
    assert "## Latency (ms)" in md
    assert "## Duplicate analysis" in md
    # Phase 1 candidate / reranker sections silently skipped; presence
    # is conditional on the data, so absence here is the contract.
    assert "## Candidate Retrieval Quality" not in md


def test_phase1_diagnostics_thresholds_pin_constants():
    """Pin the Phase 1 diagnostic thresholds so a future refactor
    can't silently drift them. Constants are intentionally exposed
    via the public ``eval.harness`` namespace.
    """
    assert DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50 == 0.80
    assert DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5 == 0.01
    assert DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5 == -0.005
    assert DIAG_RERANKER_NEGATIVE_UPLIFT_MRR_AT_10 == -0.005
    assert DIAG_HIGH_DUPLICATE_RATIO_AT_10 == 0.50


def test_phase1_high_duplicate_ratio_diagnostic_fires():
    """``highDuplicateRatio`` flag must trigger when top-10 dup ratio
    crosses the ceiling.
    """
    # 10 chunks all of the same doc_id → dup_ratio_at_10 = 0.9 > 0.5
    chunks = [_PhaseChunk("doc-A", f"c{i}", "s", "t", 0.9 - i * 0.01) for i in range(10)]
    retriever = _ScriptedRetriever({"q1": _PhaseReport(results=chunks)})
    dataset = [{"id": "r1", "query": "q1", "expected_doc_ids": ["doc-A"]}]
    summary, _, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=10,
    )
    assert summary.duplicate_doc_ratios["10"] == pytest.approx(0.9)
    assert summary.diagnostics["highDuplicateRatio"]["flagged"] is True


def test_phase1_candidate_recall_bottleneck_diagnostic():
    """``candidateRecallBottleneck`` flag fires when candidate hit@50
    falls below 0.80.
    """
    # 4 rows: only 1 has candidate hit@50 → rate = 0.25 < 0.80
    queries = {}
    dataset = []
    for i in range(4):
        if i == 0:
            cand = ["doc-gold"] + [f"doc-{j}" for j in range(60)]
        else:
            cand = [f"doc-{j}" for j in range(60)]  # gold absent
        chunks = [_PhaseChunk("doc-other", f"c{i}", "s", "t", 0.5)]
        queries[f"q{i}"] = _PhaseReport(results=chunks, candidate_doc_ids=cand)
        dataset.append({
            "id": f"r{i}", "query": f"q{i}", "expected_doc_ids": ["doc-gold"],
        })
    retriever = _ScriptedRetriever(queries)
    summary, _, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=10,
    )
    assert summary.candidate_hit_rates["50"] == pytest.approx(0.25)
    assert summary.diagnostics["candidateRecallBottleneck"]["flagged"] is True


def test_phase1_compute_diagnostics_handles_missing_metrics_gracefully():
    """`compute_retrieval_diagnostics` must return ``None`` for the
    ``flagged`` field when the underlying metric isn't measurable —
    we don't want missing data to render as "looks fine".
    """
    # Build a fresh summary with a minimum of fields populated.
    minimal = RetrievalEvalSummary(
        dataset_path="<inline>", corpus_path=None, row_count=0,
        rows_with_expected_doc_ids=0, rows_with_expected_keywords=0,
        top_k=10, mrr_k=10, ndcg_k=10,
        mean_hit_at_1=None, mean_hit_at_3=None, mean_hit_at_5=None,
        mean_mrr_at_10=None, mean_ndcg_at_10=None,
        mean_dup_rate=None, mean_unique_doc_coverage=None,
        mean_top1_score_margin=None, mean_avg_context_token_count=None,
        mean_expected_keyword_match_rate=None,
        mean_retrieval_ms=0.0, p50_retrieval_ms=0.0,
        p95_retrieval_ms=0.0, max_retrieval_ms=0.0,
    )
    diag = compute_retrieval_diagnostics(minimal)
    assert diag["candidateRecallBottleneck"]["flagged"] is None
    assert diag["highDuplicateRatio"]["flagged"] is None
    assert diag["rerankerUpliftLow"]["flagged"] is None
    assert diag["rerankerNegativeUplift"]["flagged"] is None
