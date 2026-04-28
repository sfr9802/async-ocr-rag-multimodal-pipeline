"""Tests for the long-chunk audit + length comparison report.

Same pattern as ``test_analyze_corpus_lengths.py``: write a small JSONL
corpus into ``tmp_path``, run the analyzer through the production
chunker with a stub token counter, and assert on the dataclass shape +
serialized output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence

import pytest

from eval.harness.corpus_audit import (
    DEFAULT_AUDIT_TOP_N,
    LengthComparisonReport,
    LongChunkAuditReport,
    audit_long_chunks,
    audit_to_dict,
    compare_raw_vs_cleaned,
    length_comparison_to_dict,
    render_audit_markdown,
    render_length_comparison_markdown,
)


def _tokens_per_char_counter(ratio: float = 1.0):
    def _count(batch: Sequence[str]) -> List[int]:
        return [max(1, int(math.ceil(len(t) * ratio))) for t in batch]

    return _count


def _write_corpus(tmp_path: Path, docs: list[dict]) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for d in docs:
            fp.write(json.dumps(d, ensure_ascii=False))
            fp.write("\n")
    return path


# --- audit_long_chunks --------------------------------------------------


class TestLongChunkAudit:
    def test_top_n_is_capped_and_ordered(self, tmp_path: Path):
        # Make 5 sections of distinct lengths.
        sections = {
            f"s{i}": {"chunks": ["x" * (i * 200)]}
            for i in range(1, 6)
        }
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "title": "T", "sections": sections}],
        )
        report = audit_long_chunks(
            corpus,
            top_n=3,
            token_counter=_tokens_per_char_counter(),
        )
        assert isinstance(report, LongChunkAuditReport)
        assert report.document_count == 1
        assert len(report.long_chunks) <= 3
        # Token counts are non-increasing.
        tokens = [e.token_count for e in report.long_chunks]
        assert tokens == sorted(tokens, reverse=True)

    def test_default_top_n_constant(self):
        # Pin the constant so contract changes are explicit.
        assert DEFAULT_AUDIT_TOP_N == 200

    def test_top_n_zero_emits_empty_list(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["abc"]}}}],
        )
        report = audit_long_chunks(
            corpus, top_n=0, token_counter=_tokens_per_char_counter(),
        )
        assert report.long_chunks == []

    def test_negative_top_n_rejected(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["abc"]}}}],
        )
        with pytest.raises(ValueError):
            audit_long_chunks(
                corpus,
                top_n=-1,
                token_counter=_tokens_per_char_counter(),
            )

    def test_missing_corpus_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            audit_long_chunks(
                tmp_path / "nope.jsonl",
                token_counter=_tokens_per_char_counter(),
            )

    def test_noise_signal_summary_picks_up_known_residue(self, tmp_path: Path):
        # One chunk with category footers (line-level kill candidate),
        # one clean chunk. The summary covers the whole corpus.
        noisy_text = (
            "본문 시작.\n"
            "분류: 일본 애니메이션\n"
            "분류: 2010년 작품"
        )
        clean_text = "주인공은 테이토 클라인이다. 사관학교를 탈출한다."
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "title": "T",
                    "sections": {
                        "noisy": {"chunks": [noisy_text]},
                        "clean": {"chunks": [clean_text]},
                    },
                }
            ],
        )
        report = audit_long_chunks(
            corpus, top_n=10, token_counter=_tokens_per_char_counter(),
        )
        assert "ui_category_footer" in report.noise_signal_summary
        assert report.noise_signal_summary["ui_category_footer"] >= 2

    def test_long_chunk_carries_per_chunk_signals(self, tmp_path: Path):
        # The per-chunk signals on a long entry must exclude signals
        # that fired in *other* chunks.
        noisy = "본문 X. " * 50 + "\n분류: 카테고리"
        clean = "본문 Y. " * 50
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "sections": {
                        "noisy": {"chunks": [noisy]},
                        "clean": {"chunks": [clean]},
                    },
                }
            ],
        )
        report = audit_long_chunks(
            corpus, top_n=2, token_counter=_tokens_per_char_counter(),
        )
        # Find the noisy entry by checking which one has the category signal.
        with_cat = [
            e for e in report.long_chunks
            if any(s.name == "ui_category_footer" for s in e.detected_noise_signals)
        ]
        without_cat = [
            e for e in report.long_chunks
            if all(s.name != "ui_category_footer" for s in e.detected_noise_signals)
        ]
        assert with_cat and without_cat


# --- compare_raw_vs_cleaned --------------------------------------------


class TestLengthComparison:
    def test_in_domain_only_corpus_yields_no_drops(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "sections": {
                        "s1": {"chunks": ["주인공 테이토 클라인은 사관학교를 탈출한다."]},
                        "s2": {"chunks": ["부주인공 미카게가 도와준다."]},
                    },
                }
            ],
        )
        report = compare_raw_vs_cleaned(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        assert isinstance(report, LengthComparisonReport)
        assert report.raw_chunk_count == report.cleaned_chunk_count
        assert report.dropped_chunk_count == 0
        assert report.drop_reasons == {}
        # Cleaner shouldn't shrink in-domain text.
        assert report.cleaned.char.max == report.raw.char.max

    def test_pure_noise_chunk_drops(self, tmp_path: Path):
        # A section whose entire chunk is noise. After cleaning it becomes
        # empty and is dropped from the cleaned bucket.
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "sections": {
                        "noise": {"chunks": [
                            "분류: 애니메이션\nPowered by namu-wiki\nCC BY-NC-SA"
                        ]},
                        "real": {"chunks": ["주인공은 테이토."]},
                    },
                }
            ],
        )
        report = compare_raw_vs_cleaned(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        assert report.dropped_chunk_count >= 1
        # The drop reason must be the post-clean-empty one (not empty
        # input — the raw chunk did have content).
        assert "empty_after_clean" in report.drop_reasons
        assert report.cleaned_chunk_count == report.raw_chunk_count - report.dropped_chunk_count

    def test_thresholds_passed_through(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["a" * 1500]}}}],
        )
        report = compare_raw_vs_cleaned(
            corpus,
            token_counter=_tokens_per_char_counter(),
            thresholds=(100, 1000, 2000),
        )
        for t in (100, 1000, 2000):
            assert t in report.raw.chunks_over_token_threshold
            assert t in report.cleaned.chunks_over_token_threshold

    def test_uses_p_percentile_from_metrics(self, tmp_path: Path):
        # Build a corpus whose char distribution is known. ratio=1
        # makes char/token equal so the assertion talks about both.
        sections = {
            f"s{i}": {"chunks": ["x" * i]}
            for i in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        }
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": sections}],
        )
        report = compare_raw_vs_cleaned(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        # The chunker may merge short chunks into one window — what we
        # really want to pin is monotonicity, which is the property the
        # nearest-rank percentile guarantees.
        assert report.raw.token.max >= report.raw.token.p99
        assert report.raw.token.p99 >= report.raw.token.p95
        assert report.raw.token.p95 >= report.raw.token.p50


# --- Serialization ------------------------------------------------------


class TestSerialization:
    def test_audit_to_dict_round_trips_through_json(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello world"]}}}],
        )
        report = audit_long_chunks(
            corpus, top_n=1, token_counter=_tokens_per_char_counter(),
        )
        payload = audit_to_dict(report)
        loaded = json.loads(json.dumps(payload, ensure_ascii=False))
        assert loaded["chunk_count"] == report.chunk_count
        assert loaded["top_n"] == report.top_n
        assert isinstance(loaded["long_chunks"], list)

    def test_length_comparison_to_dict_serializes_thresholds(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello"]}}}],
        )
        report = compare_raw_vs_cleaned(
            corpus,
            token_counter=_tokens_per_char_counter(),
            thresholds=(8,),
        )
        payload = length_comparison_to_dict(report)
        assert "8" in payload["raw"]["chunks_over_token_threshold"]
        assert "8" in payload["cleaned"]["chunks_over_token_threshold"]

    def test_audit_markdown_includes_required_sections(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello world"]}}}],
        )
        report = audit_long_chunks(
            corpus, top_n=1, token_counter=_tokens_per_char_counter(),
        )
        md = render_audit_markdown(report)
        assert "# Long-chunk corpus audit" in md
        assert "Noise signal summary" in md
        assert "longest chunks" in md

    def test_length_comparison_markdown_includes_required_sections(
        self, tmp_path: Path
    ):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello"]}}}],
        )
        report = compare_raw_vs_cleaned(
            corpus,
            token_counter=_tokens_per_char_counter(),
            thresholds=(8,),
        )
        md = render_length_comparison_markdown(report)
        assert "# Corpus length comparison" in md
        assert "Char distribution" in md
        assert "Token distribution" in md
        assert "Chunks over token threshold" in md
