"""Tests for the cap policy abstraction.

Covers:
  - TitleCapPolicy with / without title_provider; doc_id fallback.
  - DocIdCapPolicy strict per-doc grouping.
  - NoCapPolicy passthrough — never drops, always returns input copy.
  - SectionPathCapPolicy per-(doc_id, section) grouping.
  - cap_out_records reports the dropped chunks per group.
  - apply_cap_policy thin wrapper.

All tests are offline. Stubs exposed match the production
``RetrievedChunk`` surface (chunk_id / doc_id / section / text /
score / rerank_score) plus an optional ``title`` for the
title_provider tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from eval.harness.cap_policy import (
    DocIdCapPolicy,
    NoCapPolicy,
    SectionPathCapPolicy,
    TitleCapPolicy,
    apply_cap_policy,
)


@dataclass
class _StubChunk:
    chunk_id: str
    doc_id: str
    section: str = "overview"
    text: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None
    title: Optional[str] = None


def _title_of(chunk: Any) -> Optional[str]:
    return getattr(chunk, "title", None)


# ---------------------------------------------------------------------------
# TitleCapPolicy
# ---------------------------------------------------------------------------


class TestTitleCapPolicy:
    def test_no_cap_returns_copy(self):
        chunks = [_StubChunk(f"c{i}", "doc-A") for i in range(3)]
        out = TitleCapPolicy(None).apply(chunks)
        assert len(out) == 3
        assert out is not chunks

    def test_cap_zero_returns_copy(self):
        chunks = [_StubChunk(f"c{i}", "doc-A") for i in range(3)]
        out = TitleCapPolicy(0).apply(chunks)
        assert len(out) == 3

    def test_cap_one_drops_repeats(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        out = TitleCapPolicy(1).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c3"]

    def test_title_provider_collapses_franchise(self):
        chunks = [
            _StubChunk("c1", "doc-A", title="Series-1"),
            _StubChunk("c2", "doc-B", title="Series-1"),
            _StubChunk("c3", "doc-C", title="Series-2"),
        ]
        out = TitleCapPolicy(1, title_provider=_title_of).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c3"]

    def test_doc_id_fallback_when_title_none(self):
        # provider returns None — fall back to doc_id grouping.
        def provider(c):
            return None
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        out = TitleCapPolicy(1, title_provider=provider).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c3"]

    def test_empty_doc_id_passes_through(self):
        chunks = [
            _StubChunk("c1", ""),
            _StubChunk("c2", ""),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-A"),
        ]
        out = TitleCapPolicy(1).apply(chunks)
        # Empty-key chunks bypass the cap; doc-A capped at 1.
        assert [c.chunk_id for c in out] == ["c1", "c2", "c3"]

    def test_provider_exception_swallowed(self):
        def provider(c):
            raise RuntimeError("boom")
        chunks = [_StubChunk("c1", "doc-A"), _StubChunk("c2", "doc-A")]
        out = TitleCapPolicy(1, title_provider=provider).apply(chunks)
        # Falls back to doc_id grouping → cap=1 → only c1.
        assert [c.chunk_id for c in out] == ["c1"]

    def test_describe_includes_metadata(self):
        p = TitleCapPolicy(2, title_provider=_title_of)
        d = p.describe()
        assert d["name"] == "title"
        assert d["cap"] == 2
        assert d["title_provider"] == "supplied"


# ---------------------------------------------------------------------------
# DocIdCapPolicy
# ---------------------------------------------------------------------------


class TestDocIdCapPolicy:
    def test_strict_doc_id_grouping(self):
        # Same title, different doc_ids — DocIdCap does NOT collapse.
        chunks = [
            _StubChunk("c1", "doc-A", title="Series-1"),
            _StubChunk("c2", "doc-B", title="Series-1"),
            _StubChunk("c3", "doc-A", title="Series-1"),
        ]
        out = DocIdCapPolicy(1).apply(chunks)
        # doc-A capped at 1; doc-B survives.
        assert [c.chunk_id for c in out] == ["c1", "c2"]

    def test_cap_two_keeps_two_per_doc(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-B"),
        ]
        out = DocIdCapPolicy(2).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c2", "c4"]

    def test_no_title_provider_signal(self):
        # Even when chunks carry titles, DocIdCapPolicy ignores them.
        chunks = [
            _StubChunk("c1", "doc-A", title="X"),
            _StubChunk("c2", "doc-A", title="Y"),  # different title, same doc
            _StubChunk("c3", "doc-B", title="X"),
        ]
        out = DocIdCapPolicy(1).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c3"]


# ---------------------------------------------------------------------------
# NoCapPolicy
# ---------------------------------------------------------------------------


class TestNoCapPolicy:
    def test_passthrough(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-A"),
            _StubChunk("c5", "doc-A"),
        ]
        out = NoCapPolicy().apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c2", "c3", "c4", "c5"]
        assert out is not chunks

    def test_cap_out_records_empty(self):
        chunks = [_StubChunk("c1", "doc-A"), _StubChunk("c2", "doc-A")]
        records = NoCapPolicy().cap_out_records(chunks)
        assert records == {}


# ---------------------------------------------------------------------------
# SectionPathCapPolicy
# ---------------------------------------------------------------------------


class TestSectionPathCapPolicy:
    def test_same_doc_different_section_survives(self):
        chunks = [
            _StubChunk("c1", "doc-A", section="요약"),
            _StubChunk("c2", "doc-A", section="요약"),
            _StubChunk("c3", "doc-A", section="본문"),
        ]
        out = SectionPathCapPolicy(1).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c3"]

    def test_same_doc_same_section_capped(self):
        chunks = [
            _StubChunk("c1", "doc-A", section="요약"),
            _StubChunk("c2", "doc-A", section="요약"),
            _StubChunk("c3", "doc-A", section="요약"),
            _StubChunk("c4", "doc-B", section="요약"),
        ]
        out = SectionPathCapPolicy(2).apply(chunks)
        assert [c.chunk_id for c in out] == ["c1", "c2", "c4"]

    def test_no_section_falls_back_to_doc_id(self):
        chunks = [
            _StubChunk("c1", "doc-A", section=""),
            _StubChunk("c2", "doc-A", section=""),
            _StubChunk("c3", "doc-A", section="요약"),
        ]
        out = SectionPathCapPolicy(1).apply(chunks)
        # ("doc-a", "") is one key, ("doc-a", "요약") is another.
        assert [c.chunk_id for c in out] == ["c1", "c3"]


# ---------------------------------------------------------------------------
# cap_out_records + group_sizes
# ---------------------------------------------------------------------------


class TestCapOutRecords:
    def test_records_dropped_chunks_per_group(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-A"),
            _StubChunk("c4", "doc-B"),
        ]
        records = TitleCapPolicy(1).cap_out_records(chunks)
        # doc-A keeps c1, drops c2 and c3. doc-B has only c4 → no drops.
        assert "doc-a" in records
        assert [c.chunk_id for c in records["doc-a"]] == ["c2", "c3"]
        assert "doc-b" not in records  # no dropped

    def test_no_cap_yields_no_records(self):
        chunks = [_StubChunk("c1", "doc-A"), _StubChunk("c2", "doc-A")]
        assert TitleCapPolicy(None).cap_out_records(chunks) == {}
        assert TitleCapPolicy(0).cap_out_records(chunks) == {}

    def test_group_sizes_diagnostic(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        sizes = TitleCapPolicy(99).group_sizes(chunks)
        assert sizes == {"doc-a": 2, "doc-b": 1}


class TestApplyCapPolicy:
    def test_thin_wrapper(self):
        chunks = [
            _StubChunk("c1", "doc-A"),
            _StubChunk("c2", "doc-A"),
            _StubChunk("c3", "doc-B"),
        ]
        policy = TitleCapPolicy(1)
        out = apply_cap_policy(chunks, policy=policy)
        assert [c.chunk_id for c in out] == ["c1", "c3"]
