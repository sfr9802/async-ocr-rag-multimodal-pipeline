"""Text chunker.

Ported from port/rag's domain/chunker.py with only the minimal surface we
need for phase 2 (greedy packing + character windowing). Constants match
the source so documents chunked by either code path produce comparable
retrieval quality.

Target shape:
  - min_len     ~ 450 chars  (~250-300 tokens)
  - max_len     ~ 900 chars  (~500-600 tokens)
  - overlap     ~ 120 chars

The chunker is sentence-aware for Korean and also handles English-style
sentence boundaries as a pleasant side effect (the sentence regex is a
union of both punctuation systems).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

MIN_CH = 450
MAX_CH = 900
OVERLAP = 120

# Sentence boundary regex. Matches Korean polite/declarative endings and
# common English sentence punctuation. Captures the terminator so the join
# step can reattach it.
_SENT_SEP = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=습니다\.)\s+|(?<=합니다\.)\s+"
)


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with its section pointer and order."""

    section: str
    order: int
    text: str


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SEP.split(text) if p and p.strip()]
    return parts or [text]


def greedy_chunk(
    text: str,
    min_len: int = MIN_CH,
    max_len: int = MAX_CH,
    overlap: int = OVERLAP,
) -> List[str]:
    """Pack sentences greedily into [min_len, max_len]-sized buffers.

    Each output chunk is seeded with the last `overlap` characters of the
    previous chunk so that adjacent chunks share context. Sentences longer
    than `max_len * 1.2` are split on whitespace as an escape hatch.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: List[str] = []
    buf = ""
    hard_max = int(max_len * 1.2)

    for sentence in sentences:
        if len(sentence) > hard_max:
            # Emit current buffer first, then hard-split the long sentence.
            if buf:
                chunks.append(buf.strip())
                buf = buf[-overlap:] if overlap > 0 else ""
            words = sentence.split()
            piece = ""
            for w in words:
                if len(piece) + len(w) + 1 > max_len:
                    if piece:
                        chunks.append(piece.strip())
                    piece = w
                else:
                    piece = (piece + " " + w) if piece else w
            if piece:
                chunks.append(piece.strip())
            continue

        candidate = (buf + " " + sentence).strip() if buf else sentence

        if len(candidate) <= max_len:
            buf = candidate
            continue

        if len(buf) >= min_len:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] + " " + sentence).strip() if overlap > 0 else sentence
        else:
            # Under the min floor and the next sentence won't fit. Emit as-is
            # and move on rather than bloating past max.
            if buf:
                chunks.append(buf.strip())
            buf = sentence

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def window_by_chars(
    pre_chunks: Iterable[str],
    *,
    target: int = MAX_CH,
    min_chars: int = MIN_CH,
    max_chars: int = MAX_CH,
    overlap: int = OVERLAP,
) -> List[str]:
    """Re-window an already-chunked text list to hit `target` character size.

    Useful when the source dataset is pre-chunked (like port/rag's
    sections[name].chunks) but the upstream chunks are uneven — some are
    two sentences, some are paragraphs. This pass recombines / splits so
    every output chunk is close to the target size.
    """
    out: List[str] = []
    buf = ""
    for raw in pre_chunks:
        if not raw:
            continue
        piece = raw.strip()
        if not piece:
            continue
        if not buf:
            buf = piece
            continue
        if len(buf) + 1 + len(piece) <= max_chars:
            buf = buf + " " + piece
            if len(buf) >= target:
                out.append(buf)
                buf = buf[-overlap:] if overlap > 0 else ""
            continue
        # Would overflow — flush current buffer.
        if len(buf) >= min_chars or not out:
            out.append(buf)
            buf = (buf[-overlap:] + " " + piece).strip() if overlap > 0 else piece
        else:
            # Buffer is too small to emit alone; absorb the new piece even
            # if it nudges us slightly over max_chars. This keeps the
            # minimum-length invariant from being violated by runs of tiny
            # source chunks.
            buf = buf + " " + piece
    if buf.strip():
        out.append(buf.strip())
    return out
