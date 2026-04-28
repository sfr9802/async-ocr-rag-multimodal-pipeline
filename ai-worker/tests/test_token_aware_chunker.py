"""Token-aware chunker behaviour tests — no real tokenizer needed.

The chunker takes a ``token_counter`` callable and (optionally) an
``encode_fn`` / ``decode_fn`` pair. We feed it a deterministic
whitespace-token stub so the tests run instantly and the assertions
are easy to read: 1 whitespace-separated word == 1 token.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import pytest

from app.capabilities.rag.token_aware_chunker import (
    CHUNKER_VERSION,
    DEFAULT_HARD_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_SOFT_MAX_TOKENS,
    DEFAULT_TARGET_TOKENS,
    STRATEGY_HARD_TOKEN,
    STRATEGY_LINE_BREAK,
    STRATEGY_PARAGRAPH,
    STRATEGY_SENTENCE,
    STRATEGY_SHORT,
    TokenAwareChunk,
    TokenAwareConfig,
    chunk_text_token_aware,
    raw_section_units,
    token_aware_chunks_from_section,
)


# ---------------------------------------------------------------------------
# Token-counter stub: 1 whitespace token == 1 model token.
# ---------------------------------------------------------------------------


def _ws_count(text: str) -> int:
    """Count whitespace-separated tokens. Pure deterministic stub."""
    if not text:
        return 0
    return len([t for t in text.split() if t])


def _ws_encode(text: str) -> List[int]:
    """Tokenize a string into integer ids (one id per word)."""
    if not text:
        return []
    return list(range(len([t for t in text.split() if t])))


def _ws_decode_factory(text: str):
    """Build a decoder bound to a tokenization of ``text``.

    The hard-token splitter calls ``encode(text)`` then asks
    ``decode(slice)``. We make decode return the corresponding word
    slice so the round trip preserves content. Tests using the
    fallback path build a fresh decode per text.
    """
    words = [t for t in text.split() if t]

    def _decode(ids: Sequence[int]) -> str:
        return " ".join(words[i] for i in ids if 0 <= i < len(words))

    return _decode


# ---------------------------------------------------------------------------
# Config sanity.
# ---------------------------------------------------------------------------


def test_default_config_invariants():
    cfg = TokenAwareConfig()
    assert cfg.target_tokens == DEFAULT_TARGET_TOKENS == 512
    assert cfg.soft_max_tokens == DEFAULT_SOFT_MAX_TOKENS == 768
    assert cfg.hard_max_tokens == DEFAULT_HARD_MAX_TOKENS == 1024
    assert cfg.overlap_tokens == DEFAULT_OVERLAP_TOKENS == 80
    assert CHUNKER_VERSION == "token-aware-v1"


def test_config_rejects_inverted_max():
    with pytest.raises(ValueError):
        TokenAwareConfig(target_tokens=512, soft_max_tokens=400)
    with pytest.raises(ValueError):
        TokenAwareConfig(soft_max_tokens=768, hard_max_tokens=600)
    with pytest.raises(ValueError):
        TokenAwareConfig(target_tokens=512, overlap_tokens=512)
    with pytest.raises(ValueError):
        TokenAwareConfig(target_tokens=0)


# ---------------------------------------------------------------------------
# Empty / short inputs — no splitting, no surprises.
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    cfg = TokenAwareConfig()
    assert chunk_text_token_aware("", config=cfg, token_counter=_ws_count) == []
    assert chunk_text_token_aware("   \n\n  ", config=cfg, token_counter=_ws_count) == []


def test_short_text_emits_one_short_chunk():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    text = "alpha beta gamma"
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert len(out) == 1
    only = out[0]
    assert only.text == text
    assert only.token_count == 3
    assert only.char_count == len(text)
    assert only.split_strategy == STRATEGY_SHORT
    assert only.fallback_used is False
    assert only.chunk_index == 0


def test_hard_max_text_emits_at_most_one_short_chunk():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    text = " ".join(f"w{i}" for i in range(15))  # 15 tokens, == soft_max
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert len(out) == 1
    assert out[0].token_count <= cfg.soft_max_tokens
    assert out[0].split_strategy == STRATEGY_SHORT


# ---------------------------------------------------------------------------
# Splitting + hard-cap enforcement.
# ---------------------------------------------------------------------------


def test_oversize_text_splits_into_multiple_chunks():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    # Build 60 tokens organised as paragraphs so paragraph splitter fires.
    paragraphs = ["\n\n".join(f"p{p}_w{i}" for i in range(6)) for p in range(10)]
    text = "\n\n".join(paragraphs)
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert len(out) >= 2
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens


def test_no_emitted_chunk_exceeds_hard_max():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    # 200 tokens with NO obvious boundaries — forces hard-token fallback.
    text = " ".join(f"x{i}" for i in range(200))
    encode_text = text  # hard-token slicer needs an encode fn
    out = chunk_text_token_aware(
        text, config=cfg, token_counter=_ws_count,
        encode_fn=_ws_encode, decode_fn=_ws_decode_factory(encode_text),
    )
    assert out, "expected non-empty output"
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens, (
            f"chunk over hard_max: {ch.token_count}"
        )
    # Fallback should have been recorded somewhere.
    assert any(ch.fallback_used for ch in out)
    assert any(ch.split_strategy == STRATEGY_HARD_TOKEN for ch in out)


def test_no_emitted_chunk_exceeds_hard_max_without_encode_fn():
    """The char-fallback path (no encode_fn) must also respect hard_max."""
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    text = " ".join(f"x{i}" for i in range(200))
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert out
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens


def test_paragraph_boundary_split_preferred_over_hard_token():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    # Two paragraphs, each 10 tokens — paragraph split should fire,
    # hard-token fallback should NOT.
    para_a = " ".join(f"a{i}" for i in range(10))
    para_b = " ".join(f"b{i}" for i in range(10))
    text = f"{para_a}\n\n{para_b}"
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert len(out) >= 1
    # No fallback should have been needed.
    assert not any(ch.fallback_used for ch in out)
    assert all(ch.split_strategy != STRATEGY_HARD_TOKEN for ch in out)


def test_bullet_boundary_preserved_when_possible():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    # 5 bullets, each 8 tokens. Linear split should land on bullet
    # boundaries (line-bullet ladder rung) — no hard-token fallback.
    bullets = [
        "- " + " ".join(f"b{p}_w{i}" for i in range(8)) for p in range(5)
    ]
    text = "\n".join(bullets)
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert out
    assert not any(ch.fallback_used for ch in out)


def test_sentence_fallback_fires_when_no_paragraphs_or_bullets():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    # 3 sentences glued onto one line, each 10 tokens.
    s = " ".join(f"w{i}" for i in range(10)) + "."
    text = " ".join([s, s, s])  # 30 tokens, single line
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert len(out) >= 2
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens
    # Sentence boundary should have been picked at least once.
    assert any(ch.split_strategy == STRATEGY_SENTENCE for ch in out)


def test_overlap_is_applied_between_chunks():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=4)
    # Two paragraphs of 10 tokens each → packer should flush between
    # them and prepend an overlap tail to chunk #1.
    para_a = " ".join(f"a{i}" for i in range(10))
    para_b = " ".join(f"b{i}" for i in range(10))
    text = f"{para_a}\n\n{para_b}"
    out = chunk_text_token_aware(
        text, config=cfg, token_counter=_ws_count,
        encode_fn=_ws_encode, decode_fn=_ws_decode_factory(text),
    )
    assert len(out) >= 2
    # Chunk #1 should contain the tail of chunk #0 (some "a*" tokens
    # from the end of paragraph a).
    second = out[1].text
    assert any(f"a{i}" in second for i in range(6, 10)), (
        f"expected overlap from end of paragraph a in: {second!r}"
    )


def test_overlap_zero_disables_tail():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=0)
    para_a = " ".join(f"a{i}" for i in range(10))
    para_b = " ".join(f"b{i}" for i in range(10))
    text = f"{para_a}\n\n{para_b}"
    out = chunk_text_token_aware(
        text, config=cfg, token_counter=_ws_count,
        encode_fn=_ws_encode, decode_fn=_ws_decode_factory(text),
    )
    assert len(out) >= 2
    second = out[1].text
    # No "a" tokens should be in the second chunk when overlap is off.
    assert not any(f"a{i}" in second for i in range(10))


def test_chunker_is_deterministic():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    paragraphs = ["\n\n".join(f"p{p}_w{i}" for i in range(6)) for p in range(10)]
    text = "\n\n".join(paragraphs)
    a = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    b = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert [(c.text, c.token_count, c.split_strategy, c.fallback_used) for c in a] == \
           [(c.text, c.token_count, c.split_strategy, c.fallback_used) for c in b]


def test_chunk_metadata_is_complete():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    paragraphs = ["\n\n".join(f"p{p}_w{i}" for i in range(6)) for p in range(10)]
    text = "\n\n".join(paragraphs)
    out = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    indices = [c.chunk_index for c in out]
    assert indices == list(range(len(out)))
    for c in out:
        assert isinstance(c.text, str) and c.text
        assert c.token_count > 0
        assert c.char_count > 0
        assert c.split_strategy
        assert isinstance(c.fallback_used, bool)


def test_chunk_index_resets_per_call():
    cfg = TokenAwareConfig(target_tokens=10, soft_max_tokens=15, hard_max_tokens=20, overlap_tokens=2)
    paragraphs = ["\n\n".join(f"p{p}_w{i}" for i in range(6)) for p in range(10)]
    text = "\n\n".join(paragraphs)
    a = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    b = chunk_text_token_aware(text, config=cfg, token_counter=_ws_count)
    assert a[0].chunk_index == 0
    assert b[0].chunk_index == 0


# ---------------------------------------------------------------------------
# Section-payload helpers.
# ---------------------------------------------------------------------------


def test_raw_section_units_concatenates_chunks_then_list():
    """Mirrors ``_chunks_from_section``: chunks + list both feed into the
    chunker (text is the only fallback that's gated on emptiness)."""
    payload = {
        "chunks": ["one", "two", "three"],
        "list": [{"name": "n", "desc": "d"}],
        "text": "ignored",
    }
    units = raw_section_units(payload)
    # Chunks first, then list entries — text is only used when both
    # are empty.
    assert units == ["one", "two", "three", "n: d"]


def test_raw_section_units_falls_back_to_list_then_text():
    units = raw_section_units({
        "chunks": [],
        "list": [{"name": "alice", "desc": "hero"}, {"name": "", "desc": "lone desc"}],
    })
    assert units == ["alice: hero", "lone desc"]

    units2 = raw_section_units({
        "chunks": [],
        "list": [],
        "text": "blob fallback",
    })
    assert units2 == ["blob fallback"]


def test_raw_section_units_empty_section_returns_empty():
    assert raw_section_units({}) == []
    assert raw_section_units({"chunks": [], "list": [], "text": ""}) == []


def test_token_aware_chunks_from_section_uses_paragraph_join():
    cfg = TokenAwareConfig(target_tokens=8, soft_max_tokens=12, hard_max_tokens=18, overlap_tokens=2)
    payload = {
        "chunks": [
            " ".join(f"u0_w{i}" for i in range(8)),
            " ".join(f"u1_w{i}" for i in range(8)),
            " ".join(f"u2_w{i}" for i in range(8)),
        ],
    }
    out = token_aware_chunks_from_section(
        payload, config=cfg, token_counter=_ws_count,
    )
    assert out, "expected at least one chunk"
    # Chunks should be paragraph-strategy because we joined units with
    # double newlines under the hood.
    assert all(ch.split_strategy in (STRATEGY_PARAGRAPH, STRATEGY_SHORT)
               for ch in out)
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens


def test_token_aware_chunks_from_section_handles_empty_section():
    cfg = TokenAwareConfig(target_tokens=8, soft_max_tokens=12, hard_max_tokens=18, overlap_tokens=2)
    assert token_aware_chunks_from_section(
        {}, config=cfg, token_counter=_ws_count,
    ) == []


# ---------------------------------------------------------------------------
# Long-input stress: many tokens, no boundary signals.
# ---------------------------------------------------------------------------


def test_long_unbounded_text_emits_bounded_chunks_with_real_token_split():
    cfg = TokenAwareConfig(
        target_tokens=20, soft_max_tokens=30, hard_max_tokens=40, overlap_tokens=4,
    )
    # 500 tokens, no paragraphs, no bullets, no sentence punctuation
    # — purely opaque to the boundary regexes.
    text = " ".join(f"tok{i}" for i in range(500))
    out = chunk_text_token_aware(
        text, config=cfg, token_counter=_ws_count,
        encode_fn=_ws_encode, decode_fn=_ws_decode_factory(text),
    )
    assert out
    for ch in out:
        assert ch.token_count <= cfg.hard_max_tokens
    # All chunks should be marked fallback (hard-token slicing fired).
    assert all(ch.fallback_used for ch in out)
