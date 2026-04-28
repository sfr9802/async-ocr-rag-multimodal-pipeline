"""token_aware_emit smoke tests — focus on the schema invariants that
the production retrieval stack relies on.

Critical invariant
------------------
``_rewrite_section`` MUST clear both ``text`` and ``list`` after
populating ``chunks`` with token-aware output. ``ingest._chunks_from_section``
concatenates ``chunks`` + ``list`` (only ``text`` is gated on
emptiness), so leaving ``list`` populated would make the offline
retrieval stack feed the original raw list entries through
``window_by_chars`` on top of our token-aware chunks — defeating the
hard-cap invariant. Phase 1C learned this the hard way; this test
exists to keep the lesson sticking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from app.capabilities.rag.token_aware_chunker import (
    TokenAwareChunk,
    TokenAwareConfig,
)
from eval.harness.token_aware_emit import (
    EmitConfig,
    _count_input_units,
    _rewrite_section,
    emit_token_aware_corpus,
)


# ---------------------------------------------------------------------------
# Stubs.
# ---------------------------------------------------------------------------


def _ws_count(text: str) -> int:
    return len([t for t in (text or "").split() if t])


def _make_chunk(idx: int, text: str) -> TokenAwareChunk:
    return TokenAwareChunk(
        text=text,
        token_count=_ws_count(text),
        char_count=len(text),
        chunk_index=idx,
        split_strategy="paragraph",
        fallback_used=False,
    )


# ---------------------------------------------------------------------------
# _rewrite_section invariants.
# ---------------------------------------------------------------------------


def test_rewrite_section_clears_list_to_empty_array():
    """Without this, the offline retrieval stack runs window_by_chars
    over the raw list entries on top of our token-aware chunks."""
    payload = {
        "chunks": ["original-chunk"],
        "list": [
            {"name": "alice", "desc": "hero"},
            {"name": "bob", "desc": "villain"},
        ],
        "text": "original text blob",
        "urls": ["https://x"],
    }
    new_chunks = [_make_chunk(0, "ta-chunk-0"), _make_chunk(1, "ta-chunk-1")]
    out = _rewrite_section(payload, new_chunks)
    assert out["chunks"] == ["ta-chunk-0", "ta-chunk-1"]
    assert out["list"] == [], (
        f"list must be cleared, got: {out['list']!r}"
    )
    assert "text" not in out, (
        "text blob must be removed so it doesn't leak back into the "
        "chunker's source"
    )


def test_rewrite_section_records_original_sizes_for_provenance():
    payload = {
        "chunks": ["c"],
        "list": [{"name": "a", "desc": "b"}],
        "text": "x" * 200,
    }
    out = _rewrite_section(payload, [_make_chunk(0, "ta")])
    assert out.get("text_chars_original") == 200
    assert out.get("list_entries_original") == 1


def test_rewrite_section_preserves_unrelated_metadata():
    payload = {
        "chunks": ["c"],
        "list": [{"name": "a", "desc": "b"}],
        "text": "blob",
        "urls": ["https://example"],
        "model": "deterministic",
        "ts": "2026-04-28T00:00:00",
    }
    out = _rewrite_section(payload, [_make_chunk(0, "ta")])
    assert out["urls"] == ["https://example"]
    assert out["model"] == "deterministic"
    assert out["ts"] == "2026-04-28T00:00:00"


def test_rewrite_section_handles_missing_optional_fields():
    payload = {"chunks": ["c"]}  # no text, no list
    out = _rewrite_section(payload, [_make_chunk(0, "ta")])
    assert out["chunks"] == ["ta"]
    assert "text" not in out
    assert "list" not in out
    assert "text_chars_original" not in out
    assert "list_entries_original" not in out


# ---------------------------------------------------------------------------
# _count_input_units mirrors corpus_preprocessor.chunks_processed.
# ---------------------------------------------------------------------------


def test_count_input_units_counts_chunks_first():
    payload = {
        "chunks": ["one", "two", " "],  # blank stripped
        "list": [{"name": "a", "desc": "b"}],
        "text": "blob",
    }
    assert _count_input_units(payload) == 2  # only non-empty chunks


def test_count_input_units_falls_back_to_list_then_text():
    assert _count_input_units({
        "chunks": [], "list": [{"name": "x"}, {"name": "", "desc": ""}],
    }) == 1
    assert _count_input_units({
        "chunks": [], "list": [], "text": "blob",
    }) == 1
    assert _count_input_units({
        "chunks": [], "list": [], "text": "",
    }) == 0


# ---------------------------------------------------------------------------
# Round-trip emit through ``ingest._chunks_from_section``.
# ---------------------------------------------------------------------------


def test_emit_token_aware_corpus_output_survives_chunks_from_section(tmp_path: Path):
    """Round-trip a tiny corpus through emit + _chunks_from_section.

    After emit, the offline retrieval stack reads
    ``sections.<name>.chunks`` and runs ``window_by_chars``. With the
    list/text clearing invariant in place, the final retrievable
    chunks must match (or be a paragraph re-window of) the
    token-aware chunks — NOT include the raw list entries again.
    """
    from app.capabilities.rag.ingest import _chunks_from_section

    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({
        "doc_id": "doc1",
        "title": "Title",
        "sections": {
            "etymology": {
                "chunks": ["chunk-A short", "chunk-B short"],
                "list": [
                    {"name": "alice", "desc": "would-be-large-list-entry " * 20},
                    {"name": "bob", "desc": "another-large-list-entry " * 20},
                ],
                "text": "ignored blob",
            }
        }
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    out = tmp_path / "out.jsonl"

    cfg = TokenAwareConfig(
        target_tokens=8, soft_max_tokens=12, hard_max_tokens=20,
        overlap_tokens=2,
    )

    emit_token_aware_corpus(
        src, out,
        config=EmitConfig(chunker=cfg, write_provenance=False),
        token_counter=_ws_count,
        encode_fn=None, decode_fn=None,
        provenance_path=None,
    )

    # Reload the emitted corpus and run the same code the offline
    # retrieval stack runs.
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    section_payload = payload["sections"]["etymology"]
    # Invariant: list cleared, text dropped.
    assert section_payload["list"] == []
    assert "text" not in section_payload
    assert section_payload["chunks"], "expected token-aware chunks present"

    final_chunks = _chunks_from_section(section_payload)
    assert final_chunks, "production chunker must emit something"
    # The phrase "would-be-large-list-entry" must NOT appear because
    # that came from the original list entries, which our token-aware
    # output already condensed into the section's ``chunks`` body.
    # We only assert the *raw* list-entry shape ("alice: ..." /
    # "bob: ...") doesn't sneak back in twice — once via chunks (via
    # the token-aware splitter) and once via list (which would happen
    # if list weren't cleared).
    joined = "\n\n".join(final_chunks)
    # alice/bob should each appear at most once per emitted final
    # chunk's window — counting "alice: " ensures we didn't get a
    # second copy from the (uncleared) list field.
    alice_count = joined.count("alice:")
    bob_count = joined.count("bob:")
    # The token-aware chunker may include each name once because we
    # joined raw_section_units with paragraph separators — never
    # twice, which would be the regression signature.
    assert alice_count <= 1, (
        f"alice should appear at most once, got {alice_count} — list "
        "field probably wasn't cleared"
    )
    assert bob_count <= 1, (
        f"bob should appear at most once, got {bob_count} — list "
        "field probably wasn't cleared"
    )
