"""Token-aware chunker (Phase 1C).

Hard-caps every emitted chunk at ``hard_max_tokens`` measured by the
target embedding tokenizer (default: BAAI/bge-m3). Designed as a
*separate strategy* — the existing ``chunker.py`` (greedy_chunk +
window_by_chars) stays untouched and remains the production default
until the retrieval-quality A/B says otherwise.

Why this exists
---------------
Phase 1A/1B established that line-level cleaning + page-prefix /
inline-edit preprocessing cannot reach the dominant cause of long
chunks: a small number of namu-wiki sections (mostly ``등장인물``)
glue dozens of subentries into one ``chunks[i]`` payload that the
character-windowed chunker cannot meaningfully split (max=900 chars
but a hard-max relief valve allows single oversized sentences /
pieces through). The Phase 1B combined preprocessor still leaves
2,799 chunks > 1024 tokens (and 609 > 8192). Token-aware-v1 attacks
that head-on with a hard token cap.

Strategy
--------
Each input string is split with a layered fallback chain. The first
strategy that produces a unit fitting under ``soft_max_tokens`` wins
for that piece; otherwise we drop to the next layer:

  1. ``paragraph``     — split on blank lines (``\\n\\s*\\n``)
  2. ``line_bullet``   — split on per-line bullet / numbered prefixes
                         when the unit is a list-like block
  3. ``line_break``    — split on single ``\\n`` boundaries
  4. ``sentence``      — split on Korean / English sentence punctuation
  5. ``hard_token``    — last-resort fallback that walks the token
                         stream and slices on token-count alone.
                         Records ``fallback_used=true`` in metadata.

Greedy packing then walks the produced units and concatenates them
into chunks targeting ``target_tokens``, flushing whenever adding the
next unit would push the buffer past ``soft_max_tokens``. Adjacent
chunks share an ``overlap_tokens`` tail so retrieval still sees
context across the boundary.

Public surface
--------------
- ``CHUNKER_VERSION``               — bumped on rule change
- ``DEFAULT_TARGET_TOKENS``         — 512
- ``DEFAULT_SOFT_MAX_TOKENS``       — 768
- ``DEFAULT_HARD_MAX_TOKENS``       — 1024
- ``DEFAULT_OVERLAP_TOKENS``        — 80
- ``TokenAwareConfig``              — frozen dataclass with the knobs
- ``TokenAwareChunk``               — a single emitted chunk + metadata
- ``chunk_text_token_aware``        — the entry point
- ``token_aware_chunks_from_section`` — section-payload helper
- ``raw_section_units``             — surface for diagnostics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


CHUNKER_VERSION = "token-aware-v1"

DEFAULT_TARGET_TOKENS = 512
DEFAULT_SOFT_MAX_TOKENS = 768
DEFAULT_HARD_MAX_TOKENS = 1024
DEFAULT_OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class TokenAwareConfig:
    """Knobs for token-aware-v1.

    Invariants enforced in ``__post_init__``:
      - target_tokens > 0
      - soft_max_tokens >= target_tokens
      - hard_max_tokens >= soft_max_tokens
      - 0 <= overlap_tokens < target_tokens

    ``allow_hard_max_overflow`` exists for the diagnose path: when set,
    the chunker emits oversize chunks instead of hard-splitting and
    annotates ``fallback_used=True`` so the report can flag them. The
    production emit path always leaves it false.
    """

    target_tokens: int = DEFAULT_TARGET_TOKENS
    soft_max_tokens: int = DEFAULT_SOFT_MAX_TOKENS
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    allow_hard_max_overflow: bool = False

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.soft_max_tokens < self.target_tokens:
            raise ValueError("soft_max_tokens must be >= target_tokens")
        if self.hard_max_tokens < self.soft_max_tokens:
            raise ValueError("hard_max_tokens must be >= soft_max_tokens")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                "overlap_tokens must be < target_tokens"
            )


# ---------------------------------------------------------------------------
# Chunk dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenAwareChunk:
    """A single emitted chunk plus provenance metadata.

    ``split_strategy`` records the deepest layer that contributed to
    this chunk. ``fallback_used`` is True iff the chunk required the
    last-resort token slicer.
    """

    text: str
    token_count: int
    char_count: int
    chunk_index: int
    split_strategy: str
    fallback_used: bool


# ---------------------------------------------------------------------------
# Tokenizer plumbing.
# ---------------------------------------------------------------------------


# Counts tokens for a single string. Modules that have a batch
# tokenizer wrap it into a per-string callable.
SingleTokenCounter = Callable[[str], int]

# Tokenize a single string into its token-id list. The hard-token
# fallback needs the actual token stream so it can slice on token
# offsets and decode each window back to text. When omitted, the
# chunker degrades to a character-proportional split for the
# fallback path (still bounded, but coarser).
TokenizerEncodeFn = Callable[[str], List[int]]
TokenizerDecodeFn = Callable[[Sequence[int]], str]


# ---------------------------------------------------------------------------
# Boundary regexes.
# ---------------------------------------------------------------------------


# Two or more consecutive newlines (with optional whitespace between).
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")

# Bullet / numbered-list line starts. Conservative: must begin a line
# (after start-of-string or newline) and must be followed by at least
# one whitespace + non-empty content. Korean ordinal markers like "1."
# / "1)" / "가." also match.
_BULLET_LINE = re.compile(
    r"(?m)^\s*(?:[-*•·▪◦‣⁃]|\d{1,3}[.)]|[가-힣]\.)\s+"
)

# Sentence boundaries: copied from chunker.py for compatibility, plus a
# few common Korean enders.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|"
    r"(?<=습니다\.)\s+|(?<=합니다\.)\s+"
)


# Strategy labels used by callers and reports. Kept as constants so
# downstream code can branch on them without typo risk.
STRATEGY_SHORT = "short"  # input fit in one chunk
STRATEGY_PARAGRAPH = "paragraph"
STRATEGY_LINE_BULLET = "line_bullet"
STRATEGY_LINE_BREAK = "line_break"
STRATEGY_SENTENCE = "sentence"
STRATEGY_HARD_TOKEN = "hard_token"
STRATEGY_HARD_CHAR = "hard_char"  # used when no tokenizer encode fn


# Order matters — outer loops walk this list in order, picking the
# first strategy whose units all fit under soft_max_tokens.
_SPLIT_LADDER: Tuple[str, ...] = (
    STRATEGY_PARAGRAPH,
    STRATEGY_LINE_BULLET,
    STRATEGY_LINE_BREAK,
    STRATEGY_SENTENCE,
)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def chunk_text_token_aware(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn] = None,
    decode_fn: Optional[TokenizerDecodeFn] = None,
) -> List[TokenAwareChunk]:
    """Split ``text`` into ``TokenAwareChunk`` records.

    Returns ``[]`` for empty / whitespace-only input.

    The output list is deterministic: same input + same config + same
    tokenizer always produces the same chunks (no random tie-breaking
    inside the packer).

    Provenance:
      - ``split_strategy`` records the deepest layer that contributed.
      - ``fallback_used=True`` iff the hard token / char slicer fired.
      - ``chunk_index`` is 0-based monotonic over the returned list.
      - ``char_count`` / ``token_count`` reflect the emitted text
        exactly (including any overlap tail prepended from the
        previous chunk).
    """
    if not text or not text.strip():
        return []

    # Cheap fast path: if the whole input fits, emit one chunk and skip
    # all the splitting / packing machinery.
    total_tokens = token_counter(text)
    if total_tokens <= config.soft_max_tokens:
        cleaned = text.strip()
        return [
            TokenAwareChunk(
                text=cleaned,
                token_count=token_counter(cleaned),
                char_count=len(cleaned),
                chunk_index=0,
                split_strategy=STRATEGY_SHORT,
                fallback_used=False,
            )
        ]

    units = _split_into_units(
        text,
        config=config,
        token_counter=token_counter,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
    )

    return _greedy_pack(
        units,
        config=config,
        token_counter=token_counter,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
    )


# ---------------------------------------------------------------------------
# Splitting layer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    """Internal sub-unit produced by the splitter."""

    text: str
    tokens: int
    strategy: str
    fallback_used: bool


def _split_into_units(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn],
    decode_fn: Optional[TokenizerDecodeFn],
) -> List[_Unit]:
    """Walk the strategy ladder until every unit fits under soft_max."""
    pieces: List[Tuple[str, str]] = [(text, STRATEGY_PARAGRAPH)]

    units: List[_Unit] = []
    for piece_text, _initial_strategy in pieces:
        units.extend(
            _split_one(
                piece_text,
                config=config,
                token_counter=token_counter,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
            )
        )
    return units


def _split_one(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn],
    decode_fn: Optional[TokenizerDecodeFn],
) -> List[_Unit]:
    """Recursively split a single string until each unit fits."""
    cleaned = text.strip()
    if not cleaned:
        return []

    tokens = token_counter(cleaned)
    if tokens <= config.soft_max_tokens:
        return [_Unit(text=cleaned, tokens=tokens, strategy=STRATEGY_SHORT,
                     fallback_used=False)]

    # Walk the ladder, descending only when the current layer didn't
    # produce more than one piece.
    for strategy in _SPLIT_LADDER:
        sub_pieces = _try_split(cleaned, strategy)
        if len(sub_pieces) <= 1:
            continue
        out: List[_Unit] = []
        for sp in sub_pieces:
            sp_clean = sp.strip()
            if not sp_clean:
                continue
            sp_tokens = token_counter(sp_clean)
            if sp_tokens <= config.soft_max_tokens:
                out.append(_Unit(
                    text=sp_clean, tokens=sp_tokens, strategy=strategy,
                    fallback_used=False,
                ))
            else:
                # Recurse — this sub-piece is still too big, drop into the
                # next ladder rung.
                deeper = _split_one(
                    sp_clean,
                    config=config,
                    token_counter=token_counter,
                    encode_fn=encode_fn,
                    decode_fn=decode_fn,
                )
                out.extend(deeper)
        if out:
            return out

    # Fallback: hard token (or char) slicer.
    return _hard_split(
        cleaned,
        config=config,
        token_counter=token_counter,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
    )


def _try_split(text: str, strategy: str) -> List[str]:
    """Apply one boundary regex, return non-empty pieces."""
    if strategy == STRATEGY_PARAGRAPH:
        parts = _PARAGRAPH_BOUNDARY.split(text)
    elif strategy == STRATEGY_LINE_BULLET:
        # Split *before* each bullet boundary so the bullet stays with
        # its body. Use re.split with a lookahead on the bullet pattern
        # rebuilt as a multiline anchor.
        # Note: this only fires when the text is bullet-shaped at all.
        if not _BULLET_LINE.search(text):
            return [text]
        parts = re.split(r"(?m)(?=^\s*(?:[-*•·▪◦‣⁃]|\d{1,3}[.)]|[가-힣]\.)\s+)", text)
    elif strategy == STRATEGY_LINE_BREAK:
        parts = text.split("\n")
    elif strategy == STRATEGY_SENTENCE:
        parts = _SENTENCE_BOUNDARY.split(text)
    else:
        return [text]

    out = [p.strip() for p in parts if p and p.strip()]
    return out


def _hard_split(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn],
    decode_fn: Optional[TokenizerDecodeFn],
) -> List[_Unit]:
    """Last-resort split when no boundary fits the token budget.

    If a real tokenizer encode/decode pair is available, slice on
    token-id windows with overlap. Otherwise fall back to a
    proportional character slicer (still bounded, but coarser).
    """
    if encode_fn is not None and decode_fn is not None:
        return _hard_token_split(
            text,
            config=config,
            token_counter=token_counter,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
        )
    return _hard_char_split(
        text,
        config=config,
        token_counter=token_counter,
    )


def _hard_token_split(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: TokenizerEncodeFn,
    decode_fn: TokenizerDecodeFn,
) -> List[_Unit]:
    ids = encode_fn(text)
    n = len(ids)
    if n == 0:
        return []
    # Use target_tokens for window size — conservatively under soft_max
    # so the recombination loop has room to add overlap before hitting
    # hard_max.
    window = max(1, config.target_tokens)
    step = max(1, window)
    out: List[_Unit] = []
    pos = 0
    while pos < n:
        end = min(pos + window, n)
        slice_ids = ids[pos:end]
        text_slice = decode_fn(slice_ids).strip()
        if text_slice:
            tokens = token_counter(text_slice)
            out.append(_Unit(
                text=text_slice, tokens=tokens,
                strategy=STRATEGY_HARD_TOKEN, fallback_used=True,
            ))
        if end >= n:
            break
        pos = end  # no overlap inside the unit-list; the packer will
                   # add chunk-level overlap on its own.
    return out


def _hard_char_split(
    text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
) -> List[_Unit]:
    """Proportional char slicer for environments without an encode_fn.

    Estimates a chars-per-target-token ratio from the input, then
    slices the text on character offsets close to that ratio. Each
    slice is verified against ``hard_max_tokens``; if it overshoots,
    the slice is shortened and re-emitted.
    """
    total_tokens = max(1, token_counter(text))
    chars_per_token = max(1.0, len(text) / total_tokens)
    window_chars = max(8, int(config.target_tokens * chars_per_token))
    out: List[_Unit] = []
    n = len(text)
    pos = 0
    safety = 0
    while pos < n and safety < 10000:
        safety += 1
        end = min(pos + window_chars, n)
        candidate = text[pos:end].strip()
        if not candidate:
            pos = end
            continue
        tokens = token_counter(candidate)
        # Shrink if we overshot the hard cap. Halve repeatedly.
        shrink_safety = 0
        while tokens > config.hard_max_tokens and shrink_safety < 32:
            shrink_safety += 1
            window_chars = max(8, window_chars // 2)
            end = min(pos + window_chars, n)
            candidate = text[pos:end].strip()
            if not candidate:
                break
            tokens = token_counter(candidate)
        if candidate:
            out.append(_Unit(
                text=candidate, tokens=tokens,
                strategy=STRATEGY_HARD_CHAR, fallback_used=True,
            ))
        if end >= n:
            break
        pos = end
    return out


# ---------------------------------------------------------------------------
# Greedy packer.
# ---------------------------------------------------------------------------


def _greedy_pack(
    units: Sequence[_Unit],
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn],
    decode_fn: Optional[TokenizerDecodeFn],
) -> List[TokenAwareChunk]:
    """Concatenate units into chunks bounded by soft_max_tokens.

    Builds a chunk by appending units until adding the next one would
    push past soft_max_tokens, then flushes. Adjacent chunks share an
    overlap tail (built by re-tokenizing the trailing portion of the
    just-flushed chunk).
    """
    chunks: List[TokenAwareChunk] = []
    if not units:
        return chunks

    buf_units: List[_Unit] = []
    buf_tokens = 0
    overlap_text = ""

    def _flush(idx: int) -> int:
        nonlocal buf_units, buf_tokens, overlap_text
        if not buf_units:
            return idx
        body = "\n\n".join(u.text for u in buf_units).strip()
        text = (overlap_text + ("\n\n" if overlap_text else "") + body).strip() if overlap_text else body
        tokens = token_counter(text)
        # Final hard-cap guard. If the prepended overlap pushed the
        # chunk over hard_max_tokens, drop the overlap and re-emit.
        if tokens > config.hard_max_tokens and overlap_text:
            text = body
            tokens = token_counter(text)
        # If the body alone is still over hard_max (can happen when a
        # single unit was a hard-token slice that just barely exceeded
        # the cap), we have two options:
        #   - allow_hard_max_overflow=True: emit as-is with
        #     fallback_used=True (and warn upstream).
        #   - allow_hard_max_overflow=False: re-slice the body via the
        #     hard splitter and emit one chunk per slice.
        if tokens > config.hard_max_tokens and not config.allow_hard_max_overflow:
            re_units = _hard_split(
                text,
                config=config,
                token_counter=token_counter,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
            )
            for ru in re_units:
                chunks.append(TokenAwareChunk(
                    text=ru.text,
                    token_count=ru.tokens,
                    char_count=len(ru.text),
                    chunk_index=idx,
                    split_strategy=ru.strategy,
                    fallback_used=True,
                ))
                idx += 1
            buf_units = []
            buf_tokens = 0
            # Build overlap from the last emitted chunk's tail.
            overlap_text = _build_overlap(
                chunks[-1].text if chunks else "",
                config=config,
                token_counter=token_counter,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
            )
            return idx

        # Determine the chunk's split_strategy: take the deepest
        # strategy from the units that contributed (strategies later in
        # the ladder are deeper).
        strategies = {u.strategy for u in buf_units}
        chosen_strategy = _pick_strategy(strategies)
        chunk_fallback = any(u.fallback_used for u in buf_units)

        chunks.append(TokenAwareChunk(
            text=text,
            token_count=tokens,
            char_count=len(text),
            chunk_index=idx,
            split_strategy=chosen_strategy,
            fallback_used=chunk_fallback,
        ))
        idx += 1
        buf_units = []
        buf_tokens = 0
        overlap_text = _build_overlap(
            text,
            config=config,
            token_counter=token_counter,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
        )
        return idx

    chunk_idx = 0
    for unit in units:
        prospective = buf_tokens + unit.tokens
        if buf_units and prospective > config.soft_max_tokens:
            chunk_idx = _flush(chunk_idx)
        buf_units.append(unit)
        buf_tokens += unit.tokens
    chunk_idx = _flush(chunk_idx)
    return chunks


def _pick_strategy(strategies: set) -> str:
    """Pick the deepest strategy label from a set.

    Ladder order: paragraph < line_bullet < line_break < sentence <
    hard_token / hard_char. ``short`` only applies when the input was
    one piece — we promote to whichever boundary actually contributed.
    """
    order = (
        STRATEGY_HARD_TOKEN, STRATEGY_HARD_CHAR,
        STRATEGY_SENTENCE, STRATEGY_LINE_BREAK,
        STRATEGY_LINE_BULLET, STRATEGY_PARAGRAPH,
        STRATEGY_SHORT,
    )
    for s in order:
        if s in strategies:
            return s
    return STRATEGY_SHORT


def _build_overlap(
    chunk_text: str,
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn],
    decode_fn: Optional[TokenizerDecodeFn],
) -> str:
    """Build the overlap tail to prepend to the next chunk.

    With a real tokenizer (encode_fn/decode_fn): take the trailing
    ``overlap_tokens`` tokens of the chunk and decode. Without one:
    proportionally compute a character tail of similar size and use
    that as a coarser approximation.
    """
    if config.overlap_tokens <= 0 or not chunk_text:
        return ""
    if encode_fn is not None and decode_fn is not None:
        ids = encode_fn(chunk_text)
        if not ids:
            return ""
        tail = ids[-config.overlap_tokens:]
        decoded = decode_fn(tail).strip()
        return decoded
    # Proportional char tail.
    tokens = max(1, token_counter(chunk_text))
    chars_per_token = max(1.0, len(chunk_text) / tokens)
    tail_chars = max(8, int(config.overlap_tokens * chars_per_token))
    return chunk_text[-tail_chars:].strip()


# ---------------------------------------------------------------------------
# Section-payload helper.
# ---------------------------------------------------------------------------


def raw_section_units(raw_section: Mapping[str, Any]) -> List[str]:
    """Extract raw text units from one section payload, no re-windowing.

    Mirrors ``app.capabilities.rag.ingest._chunks_from_section`` minus
    the ``window_by_chars`` re-window — diagnostics + the token-aware
    emit path use this to feed text into the new chunker without the
    old chunker having a chance to chew on it first.

    Returns the units in source order: pre-chunked strings first,
    then per-list-entry one-liners, then the text blob fallback.
    """
    units: List[str] = []

    pre = raw_section.get("chunks")
    if isinstance(pre, list):
        units.extend(
            str(x) for x in pre if isinstance(x, (str, int, float))
        )

    list_entries = raw_section.get("list")
    if isinstance(list_entries, list):
        for entry in list_entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            if name and desc:
                units.append(f"{name}: {desc}")
            elif desc:
                units.append(desc)
            elif name:
                units.append(name)

    if not units:
        blob = raw_section.get("text")
        if isinstance(blob, str) and blob.strip():
            units.append(blob)

    return units


def token_aware_chunks_from_section(
    raw_section: Mapping[str, Any],
    *,
    config: TokenAwareConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn] = None,
    decode_fn: Optional[TokenizerDecodeFn] = None,
) -> List[TokenAwareChunk]:
    """Run the token-aware chunker over one section payload.

    Concatenates the section's raw units with paragraph separators so
    the chunker's first split layer is a meaningful boundary, then
    delegates to ``chunk_text_token_aware``. ``chunk_index`` in each
    returned ``TokenAwareChunk`` is *section-local* — ingest/emit code
    is responsible for any global numbering it needs.
    """
    units = raw_section_units(raw_section)
    if not units:
        return []
    combined = "\n\n".join(units)
    return chunk_text_token_aware(
        combined,
        config=config,
        token_counter=token_counter,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
    )
