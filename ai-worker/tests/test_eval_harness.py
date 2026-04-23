"""Tests for the eval harness.

Three layers, all fully offline:

  1. Pure-metric unit tests (CER / WER / hit@k / MRR / keyword coverage /
     edit distance). No harness, no fake providers — just input tables.

  2. I/O smoke: JSONL loader rejects malformed lines, JSON/CSV writers
     produce parseable files with the expected column set.

  3. Harness end-to-end smoke:
      - RAG harness driven against an in-memory HashingEmbedder +
        FAISS index + fake metadata store (the same pattern the
        existing test_rag_capability.py uses). Verifies that metrics
        flow through the report and that latency is populated.
      - OCR harness driven against a tiny FakeOcrProvider. Uses a
        one-byte dummy file so we never touch Pillow / PyMuPDF /
        Tesseract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pytest

from eval.harness import (
    cer,
    dup_rate,
    edit_distance,
    hit_at_k,
    keyword_coverage,
    load_jsonl,
    p_percentile,
    recall_at_k,
    reciprocal_rank,
    run_ocr_eval,
    run_rag_eval,
    topk_gap,
    wer,
    write_csv_report,
    write_json_report,
)
from eval.harness.ocr_eval import OcrEvalRow
from eval.harness.rag_eval import RagEvalRow


# ---------------------------------------------------------------------------
# 1. Metrics.
# ---------------------------------------------------------------------------


class TestEditDistance:
    def test_both_empty_is_zero(self):
        assert edit_distance([], []) == 0

    def test_empty_vs_nonempty(self):
        assert edit_distance([], list("abc")) == 3
        assert edit_distance(list("abc"), []) == 3

    def test_identical_is_zero(self):
        assert edit_distance(list("hello"), list("hello")) == 0

    def test_single_substitution(self):
        assert edit_distance(list("cat"), list("bat")) == 1

    def test_insertion_and_deletion(self):
        assert edit_distance(list("kitten"), list("sitting")) == 3  # classic example

    def test_word_level(self):
        a = "the quick brown fox".split()
        b = "the slow brown dog".split()
        assert edit_distance(a, b) == 2  # quick→slow, fox→dog


class TestCer:
    def test_perfect_match(self):
        assert cer("hello world", "hello world") == 0.0

    def test_both_empty(self):
        assert cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert cer("spurious", "") == 1.0

    def test_empty_hypothesis_nonempty_reference(self):
        # All chars are deletions → distance == len(ref)
        assert cer("", "target") == 1.0

    def test_single_substitution_normalized(self):
        # "cat" vs "bat" — 1 sub over 3 chars = 1/3
        assert cer("bat", "cat") == pytest.approx(1 / 3)

    def test_whitespace_normalization(self):
        # Triple space should collapse to single.
        assert cer("hello   world", "hello world") == 0.0

    def test_case_is_preserved(self):
        # CER does NOT lowercase by design — case counts as a substitution.
        assert cer("HELLO", "hello") == 1.0


class TestWer:
    def test_perfect_match(self):
        assert wer("the quick brown fox", "the quick brown fox") == 0.0

    def test_single_word_substitution(self):
        assert wer("the slow brown fox", "the quick brown fox") == pytest.approx(0.25)

    def test_both_empty(self):
        assert wer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert wer("extra words here", "") == 1.0

    def test_missing_words(self):
        # hyp drops two of four ref words
        assert wer("the brown", "the quick brown fox") == pytest.approx(0.5)


class TestHitAtK:
    def test_hit_at_rank_one(self):
        assert hit_at_k(["a", "b", "c"], ["a"], k=3) == 1.0

    def test_hit_at_last_rank_within_k(self):
        assert hit_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_miss_outside_k(self):
        assert hit_at_k(["x", "y", "z"], ["a"], k=3) == 0.0

    def test_k_truncates_retrieved(self):
        # "a" is at rank 5, k=3 → miss
        assert hit_at_k(["w", "x", "y", "z", "a"], ["a"], k=3) == 0.0

    def test_no_expected_returns_none(self):
        assert hit_at_k(["a", "b"], [], k=3) is None

    def test_any_expected_counts(self):
        # Two expected ids; one of them hits → 1.0
        assert hit_at_k(["a", "b", "c"], ["zzz", "b"], k=3) == 1.0


class TestReciprocalRank:
    def test_rank_one(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_rank_two(self):
        assert reciprocal_rank(["b", "a"], ["a"]) == 0.5

    def test_no_match_is_zero(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_no_expected_is_none(self):
        assert reciprocal_rank(["a", "b"], []) is None

    def test_picks_first_match(self):
        # "c" at rank 2 comes before "d" at rank 4
        assert reciprocal_rank(["a", "c", "b", "d"], ["d", "c"]) == 0.5


class TestRecallAtK:
    def test_single_gold_hit(self):
        assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0

    def test_multi_gold_partial(self):
        # Two expected, one lands in top-k → 0.5
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_multi_gold_all_hit(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_duplicate_gold_match_counts_once(self):
        # "a" appearing twice in top-k must not inflate recall past 1.0
        # when there's only one gold doc.
        assert recall_at_k(["a", "a", "c"], ["a", "b"], k=3) == 0.5

    def test_k_truncates_retrieved(self):
        # Second gold is at rank 4, outside k=3 → 1 of 2 = 0.5
        assert recall_at_k(["a", "x", "y", "b"], ["a", "b"], k=3) == 0.5

    def test_empty_expected_is_none(self):
        assert recall_at_k(["a", "b"], [], k=3) is None

    def test_no_retrieved_is_zero(self):
        assert recall_at_k([], ["a"], k=3) == 0.0

    def test_unicode_normalization_matches(self):
        # Fullwidth digits + uppercase + separator drift — all three
        # should collapse to the same normalized key so the gold lands.
        retrieved = ["ＤＯＣ_001"]
        expected = ["doc-001"]
        assert recall_at_k(retrieved, expected, k=1) == 1.0

    def test_normalization_strips_punctuation(self):
        assert recall_at_k(["doc_001"], ["doc-001"], k=1) == 1.0


class TestDupRate:
    def test_empty_is_zero(self):
        assert dup_rate([]) == 0.0

    def test_single_element_is_zero(self):
        assert dup_rate(["a"]) == 0.0

    def test_all_unique_is_zero(self):
        assert dup_rate(["a", "b", "c"]) == 0.0

    def test_all_same_approaches_one(self):
        # 3 copies, 1 unique → 1 - 1/3 = 0.666...
        assert dup_rate(["a", "a", "a"]) == pytest.approx(2 / 3)

    def test_half_duplicates(self):
        # 4 items, 2 unique → 1 - 2/4 = 0.5
        assert dup_rate(["a", "a", "b", "b"]) == 0.5


class TestPPercentile:
    def test_empty_is_zero(self):
        assert p_percentile([]) == 0.0
        assert p_percentile([], p=50.0) == 0.0

    def test_single_value(self):
        assert p_percentile([42.0]) == 42.0
        assert p_percentile([42.0], p=50.0) == 42.0

    def test_nearest_rank_default_p95(self):
        # 20 values, ceil(0.95*20) - 1 = 18 → sorted[18] == 18.0
        values = [float(i) for i in range(20)]
        assert p_percentile(values, p=95.0) == 18.0

    def test_nearest_rank_small_sample_p95_clamps_to_last(self):
        # 5 values, ceil(0.95*5) - 1 = 4 → last element
        assert p_percentile([10.0, 20.0, 30.0, 40.0, 50.0], p=95.0) == 50.0

    def test_median_matches_nearest_rank(self):
        # Nearest-rank median of [1..10]: ceil(0.5*10) - 1 = 4 → 5.0
        assert p_percentile([float(i) for i in range(1, 11)], p=50.0) == 5.0

    def test_unsorted_input_is_sorted_internally(self):
        assert p_percentile([3.0, 1.0, 2.0], p=50.0) == 2.0

    def test_clamps_to_last_when_p_is_100(self):
        assert p_percentile([1.0, 2.0, 3.0], p=100.0) == 3.0


class TestTopkGap:
    def test_too_short_returns_none_tuple(self):
        assert topk_gap([]) == (None, None)
        assert topk_gap([0.9]) == (None, None)

    def test_abs_and_rel_gap_two_values(self):
        abs_gap, rel_gap = topk_gap([0.9, 0.5])
        assert abs_gap == pytest.approx(0.4)
        assert rel_gap == pytest.approx(0.4 / 0.9)

    def test_gap_uses_first_and_last(self):
        # Middle values do not enter the calculation.
        abs_gap, rel_gap = topk_gap([0.9, 0.7, 0.4])
        assert abs_gap == pytest.approx(0.5)
        assert rel_gap == pytest.approx(0.5 / 0.9)

    def test_zero_top_gives_none_relative(self):
        abs_gap, rel_gap = topk_gap([0.0, -0.2])
        assert abs_gap == pytest.approx(0.2)
        assert rel_gap is None

    def test_equal_scores_are_zero_gap(self):
        abs_gap, rel_gap = topk_gap([0.5, 0.5])
        assert abs_gap == 0.0
        assert rel_gap == 0.0


class TestKeywordCoverage:
    def test_full_coverage(self):
        answer = "The bookshop is run by a retired translator."
        assert keyword_coverage(answer, ["bookshop", "translator"]) == 1.0

    def test_partial_coverage(self):
        answer = "The bookshop is run by someone old."
        assert keyword_coverage(answer, ["bookshop", "translator"]) == 0.5

    def test_zero_coverage(self):
        assert keyword_coverage("nothing matches", ["alpha", "beta"]) == 0.0

    def test_none_when_no_keywords(self):
        assert keyword_coverage("anything", []) is None

    def test_case_insensitive_by_default(self):
        assert keyword_coverage("BOOKSHOP", ["bookshop"]) == 1.0

    def test_case_sensitive_flag(self):
        assert (
            keyword_coverage("BOOKSHOP", ["bookshop"], case_insensitive=False)
            == 0.0
        )


# ---------------------------------------------------------------------------
# 2. I/O.
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_happy_path(self, tmp_path: Path):
        src = tmp_path / "data.jsonl"
        src.write_text(
            '{"query": "q1"}\n'
            "\n"                     # blank line should be ignored
            '# a comment to ignore\n'
            '{"query": "q2"}\n',
            encoding="utf-8",
        )
        rows = load_jsonl(src)
        assert [r["query"] for r in rows] == ["q1", "q2"]

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_malformed_line_raises_with_line_number(self, tmp_path: Path):
        src = tmp_path / "bad.jsonl"
        src.write_text('{"ok": 1}\n{not valid json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            load_jsonl(src)

    def test_top_level_array_rejected(self, tmp_path: Path):
        src = tmp_path / "array.jsonl"
        src.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_jsonl(src)


class TestReportWriters:
    def test_json_report_roundtrip(self, tmp_path: Path):
        out = tmp_path / "reports" / "run.json"
        write_json_report(
            out,
            summary={"mean_cer": 0.12, "row_count": 3},
            rows=[{"file": "a.png", "cer": 0.1}, {"file": "b.png", "cer": 0.14}],
            metadata={"harness": "ocr", "engine": "fake-ocr-1.0"},
        )
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metadata"]["harness"] == "ocr"
        assert payload["summary"]["mean_cer"] == 0.12
        assert len(payload["rows"]) == 2

    def test_csv_report_writes_header_and_rows(self, tmp_path: Path):
        out = tmp_path / "reports" / "run.csv"
        write_csv_report(
            out,
            rows=[
                {"file": "a.png", "cer": 0.1, "tags": ["en", "short"]},
                {"file": "b.png", "cer": 0.14, "tags": ["en"]},
            ],
            columns=["file", "cer", "tags"],
        )
        text = out.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        assert lines[0] == "file,cer,tags"
        # List field is pipe-joined, not comma-joined.
        assert "en|short" in lines[1]
        # Missing column in extrasaction=ignore mode shouldn't break.

    def test_csv_report_auto_columns(self, tmp_path: Path):
        out = tmp_path / "reports" / "auto.csv"
        write_csv_report(
            out,
            rows=[{"a": 1, "b": 2}, {"a": 3, "c": 4}],
        )
        header = out.read_text(encoding="utf-8").splitlines()[0]
        # Sorted union of keys.
        assert header == "a,b,c"


# ---------------------------------------------------------------------------
# 3. RAG harness end-to-end (offline).
# ---------------------------------------------------------------------------


def _build_in_memory_rag_eval_stack(tmp_path: Path):
    """Builds a tiny real Retriever + Generator against an in-memory
    FAISS index, using the HashingEmbedder + a fake metadata store.

    Returns (retriever, generator).
    """
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.generation import ExtractiveGenerator
    from app.capabilities.rag.metadata_store import ChunkLookupResult
    from app.capabilities.rag.retriever import Retriever

    passages = [
        ("chunk-1", "doc-book", "overview", "A retired translator runs a secondhand bookshop at the last station on a dying railway line."),
        ("chunk-2", "doc-cats", "overview", "An elderly fisherman feeds the stray cats of a small harbor every morning."),
        ("chunk-3", "doc-mech", "plot",     "Ironclad Academy students pilot construction mechs to reinforce a coastal dam before a typhoon."),
        ("chunk-4", "doc-aoi",  "overview", "Aoi tends luminescent gardens suspended above the clouds."),
    ]
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed_passages([p[3] for p in passages])

    index = FaissIndex(tmp_path / "idx")
    index.build(vectors, index_version="eval-test-v1", embedding_model=embedder.model_name)

    rows = [
        ChunkLookupResult(chunk_id=p[0], doc_id=p[1], section=p[2], text=p[3], faiss_row_id=i)
        for i, p in enumerate(passages)
    ]

    class _FakeMetadataStore:
        def __init__(self, version: str, rows: list):
            self._version = version
            self._by_row = {r.faiss_row_id: r for r in rows}

        def lookup_chunks_by_faiss_rows(self, index_version: str, faiss_row_ids: Iterable[int]):
            assert index_version == self._version
            return [self._by_row[i] for i in faiss_row_ids if i in self._by_row]

    metadata = _FakeMetadataStore("eval-test-v1", rows)

    retriever = Retriever(embedder=embedder, index=index, metadata=metadata, top_k=3)
    retriever.ensure_ready()
    return retriever, ExtractiveGenerator()


def test_rag_harness_end_to_end_on_hashing_index(tmp_path: Path):
    retriever, generator = _build_in_memory_rag_eval_stack(tmp_path)

    dataset = [
        {
            "query": "who runs the bookshop?",
            "expected_doc_ids": ["doc-book"],
            "expected_keywords": ["bookshop", "translator"],
            "notes": "cozy mystery",
        },
        {
            "query": "fisherman feeding harbor cats",
            "expected_doc_ids": ["doc-cats"],
            "expected_keywords": ["fisherman", "cats"],
        },
    ]

    summary, rows = run_rag_eval(
        dataset,
        retriever=retriever,
        generator=generator,
        top_k=3,
        dataset_path="test://in-memory",
    )

    assert summary.row_count == 2
    assert summary.error_count == 0
    assert summary.rows_with_expected_doc_ids == 2
    assert summary.mean_hit_at_k == 1.0  # hashing embedder lands both queries
    assert summary.mean_recall_at_k == 1.0
    assert summary.mrr is not None and summary.mrr > 0.0
    assert summary.mean_total_ms >= 0.0
    assert summary.index_version == "eval-test-v1"
    assert summary.embedding_model == "hashing-embedder-dim64"

    # New diagnostic aggregates flow into the summary.
    assert summary.mean_dup_rate == pytest.approx(0.0)
    assert summary.mean_topk_gap is not None and summary.mean_topk_gap >= 0.0
    assert summary.p95_retrieval_ms >= summary.p50_retrieval_ms
    assert summary.misses == []  # nothing to diagnose when everything hits

    # Per-row fields are populated.
    for row in rows:
        assert isinstance(row, RagEvalRow)
        assert row.retrieved_doc_ids, "retrieved_doc_ids should not be empty"
        assert row.hit_at_k in (0.0, 1.0)
        assert row.recall_at_k in (0.0, 1.0)
        assert row.reciprocal_rank is not None
        assert row.answer is not None and row.answer  # non-empty
        assert row.total_ms >= 0.0
        assert row.dup_rate >= 0.0
        assert row.topk_gap is not None


def test_rag_harness_collects_misses_for_failed_rows(tmp_path: Path):
    retriever, generator = _build_in_memory_rag_eval_stack(tmp_path)

    dataset = [
        {
            "query": "who runs the bookshop?",
            "expected_doc_ids": ["doc-book"],  # hashing lands this
        },
        {
            "query": "completely unrelated nonsense query xyz",
            "expected_doc_ids": ["doc-this-id-is-not-indexed"],  # guaranteed miss
        },
    ]
    summary, rows = run_rag_eval(
        dataset, retriever=retriever, generator=generator, top_k=3
    )

    assert len(summary.misses) == 1
    miss = summary.misses[0]
    assert miss["query"].startswith("completely unrelated")
    assert miss["expected_doc_ids"] == ["doc-this-id-is-not-indexed"]
    assert len(miss["top3"]) <= 3
    for entry in miss["top3"]:
        assert "doc_id" in entry and "score" in entry


def test_offline_corpus_builds_working_retriever(tmp_path: Path):
    """The --offline-corpus path should chunk, embed, and index a JSONL
    fixture, then return a retriever that surfaces the indexed doc ids.
    Uses HashingEmbedder so the test stays offline and fast."""
    from app.capabilities.rag.embeddings import HashingEmbedder

    from eval.harness.offline_corpus import build_offline_rag_stack

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id": "doc-a", "title": "A", "sections": {"overview": {"chunks": ['
        '"The retired translator runs the bookshop at the last station."]}}}\n'
        '{"doc_id": "doc-b", "title": "B", "sections": {"plot": {"chunks": ['
        '"The fisherman feeds the harbor cats every morning before dawn."]}}}\n',
        encoding="utf-8",
    )

    retriever, generator, info = build_offline_rag_stack(
        corpus,
        embedder=HashingEmbedder(dim=64),
        index_dir=tmp_path / "idx",
        top_k=2,
    )
    assert info.document_count == 2
    assert info.chunk_count >= 2
    assert info.embedding_model == "hashing-embedder-dim64"
    report = retriever.retrieve("bookshop translator")
    doc_ids = [r.doc_id for r in report.results]
    assert "doc-a" in doc_ids
    # Generator is a real ExtractiveGenerator instance, not just a protocol.
    assert generator.generate("q", report.results)


def test_rag_harness_reports_none_for_rows_without_expected_ids(tmp_path: Path):
    retriever, generator = _build_in_memory_rag_eval_stack(tmp_path)

    dataset = [{"query": "who runs the bookshop?"}]
    summary, rows = run_rag_eval(
        dataset, retriever=retriever, generator=generator, top_k=3
    )

    assert summary.rows_with_expected_doc_ids == 0
    assert summary.rows_with_expected_keywords == 0
    assert summary.mean_hit_at_k is None
    assert summary.mrr is None
    assert summary.mean_keyword_coverage is None
    assert rows[0].hit_at_k is None
    assert rows[0].reciprocal_rank is None
    assert rows[0].keyword_coverage is None


def test_rag_harness_continues_on_per_row_error(tmp_path: Path):
    """A single query raising shouldn't abort the whole dataset."""
    retriever, generator = _build_in_memory_rag_eval_stack(tmp_path)

    class _FlakyRetriever:
        def __init__(self, inner):
            self._inner = inner
            self._calls = 0

        def retrieve(self, query):
            self._calls += 1
            if self._calls == 2:
                raise RuntimeError("simulated retriever failure")
            return self._inner.retrieve(query)

    dataset = [
        {"query": "first query", "expected_doc_ids": ["doc-book"]},
        {"query": "second query (will fail)", "expected_doc_ids": ["doc-cats"]},
        {"query": "third query", "expected_doc_ids": ["doc-mech"]},
    ]
    summary, rows = run_rag_eval(
        dataset,
        retriever=_FlakyRetriever(retriever),
        generator=generator,
        top_k=3,
    )
    assert summary.error_count == 1
    assert rows[1].error is not None
    assert "simulated retriever failure" in rows[1].error
    # The other two rows still have hit_at_k populated.
    assert rows[0].hit_at_k is not None
    assert rows[2].hit_at_k is not None


# ---------------------------------------------------------------------------
# 4. OCR harness end-to-end (offline).
# ---------------------------------------------------------------------------


class _FakeOcrResult:
    """Duck-typed OcrPageResult / OcrDocumentResult stand-in.

    Keeps the eval tests fully decoupled from `app.capabilities.ocr`
    so this file can run even if the OCR package disappears in a
    future refactor.
    """

    def __init__(self, text: str, avg_confidence: Optional[float] = None):
        self.text = text
        self.avg_confidence = avg_confidence


class _FakeDocumentResult:
    def __init__(self, pages: List[_FakeOcrResult]):
        self.pages = pages
        self.engine_name = "fake-ocr-1.0"

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def avg_confidence(self) -> Optional[float]:
        confs = [p.avg_confidence for p in self.pages if p.avg_confidence is not None]
        return sum(confs) / len(confs) if confs else None


class _FakeOcrProvider:
    def __init__(self, image_text: str, image_conf: Optional[float] = 88.0):
        self._image_text = image_text
        self._image_conf = image_conf
        self.calls: List[str] = []

    @property
    def name(self) -> str:
        return "fake-ocr-1.0"

    def ocr_image(self, image_bytes: bytes, *, mime_type=None):
        self.calls.append("image")
        return _FakeOcrResult(text=self._image_text, avg_confidence=self._image_conf)

    def ocr_pdf(self, pdf_bytes: bytes):  # pragma: no cover — exercised in test_pdf
        self.calls.append("pdf")
        return _FakeDocumentResult([
            _FakeOcrResult("pdf page one", 90.0),
            _FakeOcrResult("pdf page two", 85.0),
        ])


def _make_ocr_dataset_dir(tmp_path: Path) -> Path:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    # The fake provider ignores file contents; one byte is enough to
    # make the harness see "file exists".
    (samples_dir / "hello.png").write_bytes(b"\x89")
    (samples_dir / "blank.png").write_bytes(b"\x89")
    return tmp_path


def test_ocr_harness_end_to_end_image_happy_path(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)

    dataset = [
        {
            "file": "samples/hello.png",
            "ground_truth": "HELLO WORLD",
            "language": "eng",
            "notes": "smoke",
        },
    ]
    provider = _FakeOcrProvider(image_text="HELLO WORLD")

    summary, rows = run_ocr_eval(
        dataset,
        provider=provider,
        dataset_dir=dataset_dir,
        dataset_path=str(dataset_dir / "ocr_sample.jsonl"),
    )

    assert summary.row_count == 1
    assert summary.evaluated_rows == 1
    assert summary.error_count == 0
    assert summary.mean_cer == 0.0
    assert summary.mean_wer == 0.0
    assert summary.empty_rate == 0.0
    assert summary.engine_name == "fake-ocr-1.0"

    assert isinstance(rows[0], OcrEvalRow)
    assert rows[0].cer == 0.0
    assert rows[0].wer == 0.0
    assert rows[0].is_empty is False
    assert rows[0].text_length == len("HELLO WORLD")
    assert rows[0].avg_confidence == 88.0
    assert provider.calls == ["image"]


def test_ocr_harness_scores_imperfect_output(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)
    dataset = [
        {
            "file": "samples/hello.png",
            "ground_truth": "HELLO WORLD",
            "language": "eng",
        }
    ]
    # Drops one character → CER = 1/11 ≈ 0.09
    provider = _FakeOcrProvider(image_text="HELO WORLD")

    summary, rows = run_ocr_eval(
        dataset, provider=provider, dataset_dir=dataset_dir
    )
    assert rows[0].cer == pytest.approx(1 / 11, abs=1e-4)
    assert rows[0].wer == pytest.approx(0.5, abs=1e-4)  # 1 of 2 words wrong
    assert summary.mean_cer == pytest.approx(1 / 11, abs=1e-4)


def test_ocr_harness_flags_empty_extraction(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)
    dataset = [
        {
            "file": "samples/blank.png",
            "ground_truth": "anything",
            "language": "eng",
        }
    ]
    provider = _FakeOcrProvider(image_text="", image_conf=None)

    summary, rows = run_ocr_eval(
        dataset, provider=provider, dataset_dir=dataset_dir
    )
    assert rows[0].is_empty is True
    assert rows[0].text_length == 0
    assert summary.empty_rate == 1.0
    assert summary.mean_cer == 1.0   # nothing vs "anything" → CER = 1.0


def test_ocr_harness_cjk_language_reports_none_wer(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)
    dataset = [
        {
            "file": "samples/hello.png",
            "ground_truth": "안녕 세계",
            "language": "kor",
        }
    ]
    provider = _FakeOcrProvider(image_text="안녕 세계")
    summary, rows = run_ocr_eval(
        dataset, provider=provider, dataset_dir=dataset_dir
    )
    assert rows[0].wer is None
    assert summary.mean_wer is None
    # CER is still computed.
    assert rows[0].cer == 0.0


def test_ocr_harness_skips_missing_files_by_default(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)
    dataset = [
        {"file": "samples/hello.png", "ground_truth": "HELLO WORLD", "language": "eng"},
        {"file": "samples/does_not_exist.png", "ground_truth": "anything", "language": "eng"},
    ]
    provider = _FakeOcrProvider(image_text="HELLO WORLD")

    summary, rows = run_ocr_eval(
        dataset, provider=provider, dataset_dir=dataset_dir
    )
    assert summary.row_count == 2
    assert summary.evaluated_rows == 1
    assert summary.skipped_rows == 1
    assert summary.error_count == 0
    assert rows[1].skipped_reason is not None
    assert "file not found" in rows[1].skipped_reason


def test_ocr_harness_fail_missing_files_flag(tmp_path: Path):
    dataset_dir = _make_ocr_dataset_dir(tmp_path)
    dataset = [
        {"file": "samples/does_not_exist.png", "ground_truth": "anything", "language": "eng"},
    ]
    provider = _FakeOcrProvider(image_text="")

    summary, rows = run_ocr_eval(
        dataset,
        provider=provider,
        dataset_dir=dataset_dir,
        skip_missing_files=False,
    )
    assert summary.error_count == 1
    assert rows[0].error is not None and "file not found" in rows[0].error


def test_ocr_harness_rejects_unsupported_extension(tmp_path: Path):
    (tmp_path / "weird.txt").write_bytes(b"not an image")
    dataset = [
        {"file": "weird.txt", "ground_truth": "hello", "language": "eng"},
    ]
    provider = _FakeOcrProvider(image_text="never called")

    summary, rows = run_ocr_eval(
        dataset, provider=provider, dataset_dir=tmp_path
    )
    assert summary.error_count == 1
    assert rows[0].error is not None
    assert "unsupported extension" in rows[0].error


# ---------------------------------------------------------------------------
# 5. Committed sample datasets parse cleanly.
# ---------------------------------------------------------------------------


class TestCommittedSampleDatasets:
    """Guards against a stray comma breaking the shipped examples."""

    def test_rag_sample_parses(self):
        path = Path("eval/datasets/rag_sample.jsonl")
        rows = load_jsonl(path)
        assert len(rows) >= 1
        for row in rows:
            assert "query" in row and isinstance(row["query"], str)

    def test_ocr_sample_parses(self):
        path = Path("eval/datasets/ocr_sample.jsonl")
        rows = load_jsonl(path)
        assert len(rows) >= 1
        for row in rows:
            assert "file" in row and isinstance(row["file"], str)
            assert "ground_truth" in row and isinstance(row["ground_truth"], str)

    def test_multimodal_sample_parses(self):
        path = Path("eval/datasets/multimodal_sample.jsonl")
        rows = load_jsonl(path)
        assert len(rows) >= 1
        for row in rows:
            assert "image" in row and isinstance(row["image"], str)
            assert "question" in row and isinstance(row["question"], str)
