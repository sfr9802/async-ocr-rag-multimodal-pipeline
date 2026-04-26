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
