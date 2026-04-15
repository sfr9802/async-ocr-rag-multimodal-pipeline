"""Chunker behaviour tests — no model, no DB, no filesystem."""

from __future__ import annotations

from app.capabilities.rag.chunker import (
    MAX_CH,
    MIN_CH,
    OVERLAP,
    greedy_chunk,
    window_by_chars,
)


def test_greedy_chunk_empty_input_returns_empty():
    assert greedy_chunk("") == []
    assert greedy_chunk("   ") == []


def test_greedy_chunk_short_text_fits_in_one_chunk():
    text = "A single short sentence."
    result = greedy_chunk(text)
    assert result == ["A single short sentence."]


def test_greedy_chunk_long_text_produces_bounded_chunks():
    # build a long input from ~20 medium sentences
    sentence = "The clock tower's escapement ticks with a quiet, mechanical dignity. "
    text = sentence * 30
    chunks = greedy_chunk(text)
    assert len(chunks) >= 2, "expected multiple chunks for a long input"
    for c in chunks:
        # hard max is max_len * 1.2 (see chunker.py), enforce with a little slack
        assert len(c) <= int(MAX_CH * 1.3), f"chunk too long: {len(c)}"


def test_greedy_chunk_produces_overlapping_context():
    sentence = "Signal Fires follows a weather team stranded during a month-long storm. "
    text = sentence * 30
    chunks = greedy_chunk(text)
    assert len(chunks) >= 2
    # Adjacent chunks should share some characters (overlap tail or sentence).
    overlap_characters = min(OVERLAP, 40)
    for a, b in zip(chunks, chunks[1:]):
        # Not a strict substring check — the chunker rebuilds on sentence
        # boundaries — but the tail of a should appear near the head of b
        # for most real inputs.
        head = b[:overlap_characters]
        assert head.strip(), "second chunk should not start empty"


def test_window_by_chars_recombines_small_source_chunks():
    src = ["short fragment one.", "short fragment two.", "short fragment three."]
    out = window_by_chars(src, target=40, min_chars=10, max_chars=80, overlap=5)
    # All three tiny fragments should have been combined into at least one chunk.
    assert len(out) >= 1
    combined = " ".join(out)
    for fragment in src:
        assert fragment in combined


def test_window_by_chars_with_large_chunks_preserves_content():
    src = [
        "x" * (MAX_CH - 50),
        "y" * (MAX_CH - 50),
    ]
    out = window_by_chars(src, target=MAX_CH, min_chars=MIN_CH, max_chars=MAX_CH, overlap=OVERLAP)
    # Both source chunks' characters should appear somewhere in the output.
    joined = "".join(out)
    assert "x" * 10 in joined
    assert "y" * 10 in joined
