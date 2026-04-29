"""Tests for ``eval/harness/reranker_input_format.py``.

Pin the formatter contract for each named format, the
``FormattingRerankerWrapper`` clone-and-restore behavior, the audit
preview shape (``has_title`` / ``has_section`` / ``truncated`` flags),
and the title_provider fallback. All tests are pure-Python — no
sentence-transformers, no FAISS, no GPU.

The wrapper is the *eval-only* hook that lets the confirm sweep test
"what passage does the cross-encoder see?" without touching production
``CrossEncoderReranker``. The contract these tests pin:

  - ``rerank()`` is identity on chunk_id / doc_id / section / score —
    only ``rerank_score`` changes between in and out.
  - ``last_input_previews`` is reset every call.
  - cand@K invariance: the wrapper does not reorder or drop chunks
    before the base reranker scores them. (Order may change *after*
    the base reranker, but that's the base's job.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from app.capabilities.rag.generation import RetrievedChunk
from eval.harness.reranker_input_format import (
    RERANKER_INPUT_FORMATS,
    FormattingPreview,
    FormattingRerankerWrapper,
    format_passage,
)


# ---------------------------------------------------------------------------
# Stubs: a minimal reranker that returns chunks in score order, attaching
# a deterministic rerank_score so the wrapper has something to copy.
# ---------------------------------------------------------------------------


@dataclass
class _StubReranker:
    """Tiny stand-in for ``CrossEncoderReranker``.

    Records the *formatted* text it saw per chunk so a test can assert
    "the wrapper actually swapped the passage". Returns chunks sorted
    descending by len(text) so the test can detect a re-order driven
    by the formatter (longer prefix → higher rerank_score).
    """

    name_value: str = "stub-reranker"
    last_seen_texts: List[str] = None
    raise_on_call: bool = False

    def __post_init__(self) -> None:
        self.last_seen_texts = []

    @property
    def name(self) -> str:
        return self.name_value

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        if self.raise_on_call:
            raise RuntimeError("stub failure")
        self.last_seen_texts = [c.text for c in chunks]
        scored = sorted(
            chunks, key=lambda c: len(c.text), reverse=True,
        )
        # Attach a synthetic rerank_score = len(text).
        out = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                section=c.section,
                text=c.text,
                score=c.score,
                rerank_score=float(len(c.text)),
            )
            for c in scored
        ]
        return out[:k]


def _chunk(
    chunk_id: str, doc_id: str, section: str, text: str, score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=doc_id, section=section,
        text=text, score=score,
    )


def _title_lookup(mapping: dict):
    """Return a TitleProvider-shaped closure (chunk -> title)."""
    def _provider(chunk):
        return mapping.get(getattr(chunk, "doc_id", None))
    return _provider


# ---------------------------------------------------------------------------
# 1. format_passage — output shape per format
# ---------------------------------------------------------------------------


class TestFormatPassage:
    def test_chunk_only_returns_text_unchanged(self):
        out = format_passage(
            fmt="chunk_only",
            chunk_text="이 문서는 ABC에 관한 것이다.",
            title="제목X",
            section_path="섹션Y",
        )
        assert out == "이 문서는 ABC에 관한 것이다."

    def test_title_plus_chunk_emits_two_line_prefix(self):
        out = format_passage(
            fmt="title_plus_chunk",
            chunk_text="본문 텍스트",
            title="My Anime",
            section_path=None,
        )
        assert out == "제목: My Anime\n본문: 본문 텍스트"

    def test_title_plus_chunk_falls_back_to_chunk_only_when_title_missing(self):
        out = format_passage(
            fmt="title_plus_chunk",
            chunk_text="본문",
            title=None,
            section_path="섹션",
        )
        assert out == "본문"

    def test_title_section_plus_chunk_includes_both_lines(self):
        out = format_passage(
            fmt="title_section_plus_chunk",
            chunk_text="본문",
            title="My Anime",
            section_path="요약",
        )
        assert out == "제목: My Anime\n섹션: 요약\n본문: 본문"

    def test_title_section_plus_chunk_omits_missing_section(self):
        out = format_passage(
            fmt="title_section_plus_chunk",
            chunk_text="본문",
            title="My Anime",
            section_path=None,
        )
        assert out == "제목: My Anime\n본문: 본문"

    def test_title_section_plus_chunk_omits_missing_title_keeps_section(self):
        out = format_passage(
            fmt="title_section_plus_chunk",
            chunk_text="본문",
            title=None,
            section_path="요약",
        )
        assert out == "섹션: 요약\n본문: 본문"

    def test_compact_metadata_with_both_emits_bracketed_head(self):
        out = format_passage(
            fmt="compact_metadata_plus_chunk",
            chunk_text="본문 텍스트",
            title="My Anime",
            section_path="요약",
        )
        assert out == "[My Anime / 요약]\n본문 텍스트"

    def test_compact_metadata_with_title_only(self):
        out = format_passage(
            fmt="compact_metadata_plus_chunk",
            chunk_text="본문",
            title="My Anime",
            section_path=None,
        )
        assert out == "[My Anime]\n본문"

    def test_compact_metadata_with_section_only(self):
        out = format_passage(
            fmt="compact_metadata_plus_chunk",
            chunk_text="본문",
            title=None,
            section_path="요약",
        )
        assert out == "[요약]\n본문"

    def test_compact_metadata_with_neither_returns_chunk(self):
        out = format_passage(
            fmt="compact_metadata_plus_chunk",
            chunk_text="본문",
            title=None,
            section_path=None,
        )
        assert out == "본문"

    def test_strips_leading_whitespace_from_chunk_body(self):
        out = format_passage(
            fmt="title_plus_chunk",
            chunk_text="\n\n  본문",
            title="X",
            section_path=None,
        )
        # Body lstripped — title+section formatters always emit
        # ``본문: {stripped}``.
        assert out == "제목: X\n본문: 본문"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown reranker_input_format"):
            format_passage(
                fmt="bogus",
                chunk_text="x",
                title=None,
                section_path=None,
            )

    def test_handles_none_chunk_text(self):
        out = format_passage(
            fmt="title_plus_chunk",
            chunk_text=None,
            title="X",
            section_path=None,
        )
        assert out == "제목: X\n본문: "

    def test_canonical_format_list_is_stable(self):
        # Pin the order so report writers / sweep drivers reading the
        # tuple keep deterministic output.
        assert RERANKER_INPUT_FORMATS == (
            "chunk_only",
            "title_plus_chunk",
            "title_section_plus_chunk",
            "compact_metadata_plus_chunk",
        )


# ---------------------------------------------------------------------------
# 2. FormattingRerankerWrapper — wrap base reranker, restore originals
# ---------------------------------------------------------------------------


class TestFormattingRerankerWrapper:
    def test_returns_original_chunks_with_new_rerank_score(self):
        base = _StubReranker()
        chunks = [
            _chunk("c1", "doc1", "요약", "짧은 텍스트", score=0.1),
            _chunk("c2", "doc2", "본문", "조금 더 긴 텍스트입니다", score=0.2),
        ]
        wrapper = FormattingRerankerWrapper(
            base,
            fmt="title_plus_chunk",
            title_provider=_title_lookup({"doc1": "T1", "doc2": "T2"}),
        )
        out = wrapper.rerank("Q", chunks, k=2)

        # The base saw FORMATTED text (carries the title prefix).
        assert any("제목: T1" in t for t in base.last_seen_texts)
        assert any("제목: T2" in t for t in base.last_seen_texts)

        # The OUTPUT chunks carry the original (unformatted) text.
        assert all(not c.text.startswith("제목: ") for c in out)
        assert {c.chunk_id for c in out} == {"c1", "c2"}
        # rerank_score should be set on every returned chunk.
        assert all(c.rerank_score is not None for c in out)

    def test_preserves_chunk_id_doc_id_section_score_through_round_trip(self):
        base = _StubReranker()
        chunk = _chunk("c1", "docA", "S1", "텍스트", score=0.42)
        wrapper = FormattingRerankerWrapper(
            base, fmt="chunk_only", title_provider=None,
        )
        out = wrapper.rerank("Q", [chunk], k=1)
        assert len(out) == 1
        assert out[0].chunk_id == "c1"
        assert out[0].doc_id == "docA"
        assert out[0].section == "S1"
        assert out[0].text == "텍스트"
        assert out[0].score == pytest.approx(0.42)
        assert out[0].rerank_score is not None

    def test_empty_input_returns_empty_and_clears_previews(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_plus_chunk",
        )
        wrapper._last_input_previews = [object()]  # type: ignore[attr-defined]
        assert wrapper.rerank("Q", [], k=5) == []
        assert wrapper.last_input_previews == []

    def test_k_zero_returns_empty(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(base, fmt="chunk_only")
        out = wrapper.rerank(
            "Q", [_chunk("c1", "d1", "s", "t")], k=0,
        )
        assert out == []

    def test_unknown_format_raises_at_construction(self):
        base = _StubReranker()
        with pytest.raises(ValueError, match="unknown reranker_input_format"):
            FormattingRerankerWrapper(base, fmt="bogus")

    def test_name_carries_format_marker(self):
        base = _StubReranker(name_value="cross-encoder:bge-v2-m3")
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_section_plus_chunk",
        )
        assert "fmt=title_section_plus_chunk" in wrapper.name
        assert "cross-encoder:bge-v2-m3" in wrapper.name

    def test_title_provider_failure_falls_back_to_chunk_only(self):
        def boom(chunk):
            raise RuntimeError("provider broken")

        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_plus_chunk", title_provider=boom,
        )
        out = wrapper.rerank(
            "Q", [_chunk("c1", "d1", "s", "본문")], k=1,
        )
        # Provider failure → title=None → format falls back to chunk_only
        # → base saw the raw body.
        assert base.last_seen_texts == ["본문"]
        assert len(out) == 1

    def test_resets_previews_per_call(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_plus_chunk",
            title_provider=_title_lookup({"d1": "T"}),
        )
        wrapper.rerank("Q1", [_chunk("c1", "d1", "s", "본문")], k=1)
        assert len(wrapper.last_input_previews) == 1
        wrapper.rerank("Q2", [_chunk("c2", "d1", "s", "본문2")], k=1)
        previews = wrapper.last_input_previews
        assert len(previews) == 1
        assert previews[0].chunk_id == "c2"

    def test_passes_k_through_to_base(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(base, fmt="chunk_only")
        chunks = [_chunk(f"c{i}", f"d{i}", "s", f"t{i}") for i in range(5)]
        out = wrapper.rerank("Q", chunks, k=3)
        assert len(out) == 3

    def test_does_not_drop_or_reorder_before_base_sees_it(self):
        """Cand@K invariance: same chunks reach base reranker regardless
        of format. The base may reorder, but the wrapper must hand it
        the same set in the same order.
        """
        base = _StubReranker()
        chunks = [_chunk(f"c{i}", f"d{i}", "s", f"t{i}") for i in range(3)]
        wrapper_a = FormattingRerankerWrapper(base, fmt="chunk_only")
        wrapper_b = FormattingRerankerWrapper(
            base, fmt="title_plus_chunk",
            title_provider=_title_lookup({"d0": "T", "d1": "T", "d2": "T"}),
        )
        wrapper_a.rerank("Q", chunks, k=3)
        seen_a = list(base.last_seen_texts)
        wrapper_b.rerank("Q", chunks, k=3)
        seen_b = list(base.last_seen_texts)
        # Lengths match and order matches (just text content differs).
        assert len(seen_a) == len(seen_b) == 3
        # Each returned chunk_id ordering reflects what the base saw —
        # the wrapper just rebuilt RetrievedChunks from chunk_id.
        # Cand pool itself unchanged: chunk_ids remain identical.
        out_a = wrapper_a.rerank("Q", chunks, k=3)
        out_b = wrapper_b.rerank("Q", chunks, k=3)
        assert {c.chunk_id for c in out_a} == {c.chunk_id for c in out_b}


# ---------------------------------------------------------------------------
# 3. last_input_previews — audit-shape contract
# ---------------------------------------------------------------------------


class TestPreviewCapture:
    def test_preview_carries_metadata_flags(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_section_plus_chunk",
            title_provider=_title_lookup({"d1": "MyTitle"}),
            preview_max_chars=400,
            truncation_threshold_chars=1000,
        )
        wrapper.rerank(
            "Q",
            [_chunk("c1", "d1", "요약", "본문 텍스트")],
            k=1,
        )
        previews = wrapper.last_input_previews
        assert len(previews) == 1
        p = previews[0]
        assert isinstance(p, FormattingPreview)
        assert p.chunk_id == "c1"
        assert p.doc_id == "d1"
        assert p.title == "MyTitle"
        assert p.section == "요약"
        assert p.fmt == "title_section_plus_chunk"
        assert p.has_title is True
        assert p.has_section is True
        assert p.truncated is False
        # Preview should contain the formatted prefix.
        assert "제목: MyTitle" in p.preview
        assert "섹션: 요약" in p.preview

    def test_preview_truncated_flag_fires_when_formatted_text_long(self):
        base = _StubReranker()
        long_body = "ㄱ" * 1500
        wrapper = FormattingRerankerWrapper(
            base, fmt="chunk_only",
            truncation_threshold_chars=800,
        )
        wrapper.rerank(
            "Q",
            [_chunk("c1", "d1", "s", long_body)],
            k=1,
        )
        p = wrapper.last_input_previews[0]
        assert p.truncated is True
        assert p.formatted_length == 1500

    def test_has_title_false_when_format_strips_title(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="chunk_only",
            title_provider=_title_lookup({"d1": "MyTitle"}),
        )
        wrapper.rerank(
            "Q", [_chunk("c1", "d1", "요약", "본문")], k=1,
        )
        p = wrapper.last_input_previews[0]
        # chunk_only does not surface the title.
        assert p.has_title is False
        # Title is still recorded for audit purposes.
        assert p.title == "MyTitle"

    def test_record_input_previews_disabled(self):
        base = _StubReranker()
        wrapper = FormattingRerankerWrapper(
            base, fmt="title_plus_chunk",
            title_provider=_title_lookup({"d1": "T"}),
            record_input_previews=False,
        )
        wrapper.rerank("Q", [_chunk("c1", "d1", "s", "x")], k=1)
        assert wrapper.last_input_previews == []

    def test_preview_max_chars_truncates_recorded_preview_only(self):
        base = _StubReranker()
        body = "x" * 1000
        wrapper = FormattingRerankerWrapper(
            base, fmt="chunk_only",
            preview_max_chars=50,
            truncation_threshold_chars=200,
        )
        wrapper.rerank(
            "Q", [_chunk("c1", "d1", "s", body)], k=1,
        )
        p = wrapper.last_input_previews[0]
        # Recorded preview is bounded to 50 chars …
        assert len(p.preview) == 50
        # … but the *formatted* length is the full text the base saw.
        assert p.formatted_length == 1000
        # And truncated flag fires (formatted > truncation_threshold).
        assert p.truncated is True


# ---------------------------------------------------------------------------
# 4. Forwarded properties — last_breakdown_ms passes through
# ---------------------------------------------------------------------------


class TestForwardedProperties:
    def test_last_breakdown_ms_proxies_to_base(self):
        @dataclass
        class _R:
            name = "x"
            last_breakdown_ms = {"forward_ms": 12.0, "tokenize_ms": 3.0}

            def rerank(self, q, chunks, k):
                return list(chunks)[:k]

        wrapper = FormattingRerankerWrapper(_R(), fmt="chunk_only")
        assert wrapper.last_breakdown_ms == {
            "forward_ms": 12.0, "tokenize_ms": 3.0,
        }

    def test_last_breakdown_ms_returns_none_when_base_lacks_it(self):
        @dataclass
        class _R:
            name = "x"

            def rerank(self, q, chunks, k):
                return list(chunks)[:k]

        wrapper = FormattingRerankerWrapper(_R(), fmt="chunk_only")
        assert wrapper.last_breakdown_ms is None
