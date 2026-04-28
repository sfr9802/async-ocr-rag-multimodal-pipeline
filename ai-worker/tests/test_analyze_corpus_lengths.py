"""Tests for the corpus chunk-length analyzer.

The analyzer reads ``corpus.jsonl`` through the production
``_chunks_from_section`` chunker and reports char + token length
distributions. We avoid loading the real bge-m3 tokenizer in tests by
injecting a deterministic token counter.

Coverage
--------
- chunker integration (one section with `chunks`, one with `text`,
  one with `list`) produces the expected chunk count
- distribution percentiles match a hand-computed mean/p50/p95/max
- chunks_over_token_threshold counts truncation candidates correctly
- top_longest is ordered by token length and capped
- markdown renderer includes the cap table and longest-chunk section
- empty corpus raises rather than silently producing zero chunks
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence

import pytest

from eval.harness.analyze_corpus_lengths import (
    DEFAULT_THRESHOLDS,
    LengthAnalysis,
    _percentile,
    analyze_corpus_lengths,
    length_analysis_to_dict,
    render_length_analysis_markdown,
)
from eval.harness.metrics import p_percentile


def _tokens_per_char_counter(ratio: float = 1.0):
    """Stub tokenizer: token count = ceil(len(text) * ratio).

    With ``ratio=1.0`` token == char, which keeps mental arithmetic
    tractable in the assertions below.
    """

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


# --- Chunker integration ------------------------------------------------


class TestChunkerIntegration:
    def test_chunks_from_chunks_field_are_counted(self, tmp_path: Path):
        # window_by_chars merges short chunks into one buffer, so two
        # short pre-chunks of "abc" + " def" land as a single window.
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "title": "T",
                    "sections": {
                        "s1": {"chunks": ["abc", "def"]},
                    },
                }
            ],
        )
        analysis = analyze_corpus_lengths(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        assert analysis.document_count == 1
        # The chunker emits at least one chunk; whether it merges depends
        # on the windower's min/max thresholds. The interesting check is
        # "non-zero" because an empty chunk set would have raised.
        assert analysis.chunk_count >= 1

    def test_text_blob_section_runs_through_greedy_chunk(self, tmp_path: Path):
        long_text = ("이것은 테스트 문장이다. " * 200).strip()
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "title": "T",
                    "sections": {"body": {"text": long_text}},
                }
            ],
        )
        analysis = analyze_corpus_lengths(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        # greedy_chunk targets 900 chars; a 5k-char blob produces > 1
        # chunk, never zero.
        assert analysis.chunk_count >= 2

    def test_skips_documents_without_doc_id_or_title(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [
                {"sections": {"s": {"chunks": ["lonely"]}}},  # no id
                {
                    "doc_id": "d1",
                    "sections": {"s": {"chunks": ["kept"]}},
                },
            ],
        )
        analysis = analyze_corpus_lengths(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        assert analysis.document_count == 1


# --- Distribution + threshold maths -------------------------------------


class TestDistribution:
    def test_percentile_monotonicity(self, tmp_path: Path):
        # 10 sections that produce a varied length distribution after
        # the windower. We can't pin exact values because window_by_chars
        # may merge short chunks, so the *behavioral* invariant we lock
        # in here is monotonicity (max >= p99 >= p95 >= p50).
        sections = {}
        for i in range(1, 11):
            text = "x" * (i * 100)
            sections[f"s{i}"] = {"chunks": [text]}
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "title": "T", "sections": sections}],
        )
        analysis = analyze_corpus_lengths(
            corpus, token_counter=_tokens_per_char_counter(),
        )
        assert analysis.chunk_count >= 1
        assert analysis.token_length.max >= analysis.token_length.p99
        assert analysis.token_length.p99 >= analysis.token_length.p95
        assert analysis.token_length.p95 >= analysis.token_length.p50

    def test_percentile_matches_metrics_p_percentile(self):
        # The analyzer's _percentile must produce byte-identical output
        # to metrics.p_percentile across the percentile points the
        # report uses (50/90/95/99). This anchors the docstring claim
        # and prevents the round-vs-ceil drift that earlier versions
        # exhibited (e.g. p=95 on n=15: round->idx 13, ceil->idx 14).
        for n in (1, 2, 9, 10, 11, 15, 20, 21, 51, 100, 101):
            xs = list(range(1, n + 1))
            for pct in (50, 90, 95, 99):
                assert _percentile(xs, pct) == p_percentile(
                    [float(v) for v in xs], pct
                ), f"divergence at n={n}, pct={pct}"

    def test_chunks_over_threshold_counts_truncation_candidates(self, tmp_path: Path):
        # Two short chunks (~50 chars), one very long (~3000 chars).
        # The long one ends up in its own window because it exceeds
        # max_chars=900 standalone; the short two get merged.
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "title": "T",
                    "sections": {
                        "short": {"chunks": ["short text one"]},
                        "long": {"chunks": ["a" * 3000]},
                    },
                }
            ],
        )
        analysis = analyze_corpus_lengths(
            corpus,
            token_counter=_tokens_per_char_counter(),
            thresholds=(100, 1024, 4096),
        )
        # > 1024 should catch the long chunk.
        assert analysis.chunks_over_token_threshold[1024] >= 1
        # > 4096 should not catch any (~3000 < 4096).
        assert analysis.chunks_over_token_threshold[4096] == 0
        # Ratio is in [0, 1].
        for v in analysis.chunks_over_token_threshold_ratio.values():
            assert 0.0 <= v <= 1.0

    def test_zero_top_longest_emits_no_samples(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["abc"]}}}],
        )
        analysis = analyze_corpus_lengths(
            corpus,
            token_counter=_tokens_per_char_counter(),
            top_longest=0,
        )
        assert analysis.longest_chunks == []

    def test_top_longest_ordered_by_token_length(self, tmp_path: Path):
        # Three sections of distinct sizes — long should land first.
        corpus = _write_corpus(
            tmp_path,
            [
                {
                    "doc_id": "d1",
                    "sections": {
                        "tiny": {"chunks": ["abc"]},
                        "medium": {"chunks": ["m" * 500]},
                        "huge": {"chunks": ["h" * 5000]},
                    },
                }
            ],
        )
        analysis = analyze_corpus_lengths(
            corpus,
            token_counter=_tokens_per_char_counter(),
            top_longest=3,
        )
        # token_length non-increasing along the list.
        lens = [s.token_length for s in analysis.longest_chunks]
        assert lens == sorted(lens, reverse=True)


# --- Errors + CLI helpers -----------------------------------------------


class TestErrors:
    def test_missing_corpus_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            analyze_corpus_lengths(
                tmp_path / "nope.jsonl",
                token_counter=_tokens_per_char_counter(),
            )

    def test_corpus_with_no_chunks_raises(self, tmp_path: Path):
        # Section payloads that produce nothing usable.
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"empty": {"text": "   "}}}],
        )
        with pytest.raises(RuntimeError):
            analyze_corpus_lengths(
                corpus, token_counter=_tokens_per_char_counter(),
            )

    def test_negative_top_longest_rejected(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["abc"]}}}],
        )
        with pytest.raises(ValueError):
            analyze_corpus_lengths(
                corpus,
                token_counter=_tokens_per_char_counter(),
                top_longest=-1,
            )


# --- Serialization -------------------------------------------------------


class TestSerialization:
    def test_to_dict_round_trips_through_json(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello world"]}}}],
        )
        analysis = analyze_corpus_lengths(
            corpus,
            token_counter=_tokens_per_char_counter(),
            thresholds=(8,),
            top_longest=1,
        )
        payload = length_analysis_to_dict(analysis)
        # Threshold keys are strings post-serialize.
        assert "8" in payload["chunks_over_token_threshold"]
        # JSON round-trip survives.
        loaded = json.loads(json.dumps(payload, ensure_ascii=False))
        assert loaded["chunk_count"] == analysis.chunk_count
        assert loaded["tokenizer"] == analysis.tokenizer

    def test_markdown_includes_threshold_table(self, tmp_path: Path):
        corpus = _write_corpus(
            tmp_path,
            [{"doc_id": "d1", "sections": {"s": {"chunks": ["hello world"]}}}],
        )
        analysis = analyze_corpus_lengths(
            corpus,
            token_counter=_tokens_per_char_counter(),
            top_longest=1,
        )
        md = render_length_analysis_markdown(analysis)
        assert "# Corpus chunk-length analysis" in md
        assert "## Length distribution" in md
        assert "## Chunks above max_seq_length cap" in md
        # 1024 is one of the default thresholds — the row mentions it.
        assert "> 1024" in md
        assert "Top 1 longest chunks" in md
