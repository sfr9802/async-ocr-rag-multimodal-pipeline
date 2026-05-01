"""Self-contained BM25 helper for the LLM silver-500 build.

Why a separate file
-------------------
``eval/harness/bm25_retriever.py`` already implements a fully featured
BM25Okapi over the corpus, but it imports
``app.capabilities.rag.generation.RetrievedChunk`` from production code
and pulls in the ``embedding_text_builder`` machinery that's overkill
for "given a query, what's the rank of expected_page_id's first
chunk?". The Phase 7 spec for the LLM silver set explicitly says
"production code는 절대 수정하지 않습니다" — to keep that line clean
even from indirect imports, we ship a small, pure-stdlib BM25 here
that the silver-500 builder uses solely to populate the
``bm25_expected_page_first_rank`` field of ``lexical_overlap``.

Design
------
  - Tokenization is lifted from ``bm25_retriever.tokenize_for_bm25``
    *in spirit* but reimplemented locally so this module has zero
    cross-file dependency. The two tokenizers will diverge only if
    a future production-side rewrite changes the canonical form;
    the silver set's BM25 rank is meant to be a stable historical
    signal, so it's correct that it doesn't follow that drift.
  - ``BM25SimpleIndex.first_rank_for_page(query, page_id)`` walks the
    sorted score list once and returns the 1-based rank of the first
    chunk whose ``page_id`` matches. Returns ``None`` when the page
    has no chunk in the top-K considered (default: full corpus).
  - The whole index is built in-memory off a JSONL chunks file. For
    the 135K-chunk namu corpus this takes ~30s and ~600MB peak.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization (local copy — see "Design" note in the docstring).
# ---------------------------------------------------------------------------


_WS_SPLIT_RE = re.compile(r"\s+")
_NON_ALNUM_TAIL_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_CJK_RE = re.compile(
    r"[가-힯ᄀ-ᇿ㄰-㆏"
    r"一-鿿㐀-䶿"
    r"぀-ゟ゠-ヿ]"
)


def tokenize(text: Optional[str]) -> List[str]:
    """Whitespace + CJK 1-gram tokenization.

    Mirrors the bm25_retriever tokenizer's logic without importing
    from it. Returns an empty list on empty/``None`` input.
    """
    if not text:
        return []
    folded = unicodedata.normalize("NFKC", text).casefold()
    out: List[str] = []
    for raw in _WS_SPLIT_RE.split(folded):
        if not raw:
            continue
        stripped = _NON_ALNUM_TAIL_RE.sub("", raw)
        if stripped:
            out.append(stripped)
        for ch in raw:
            if _CJK_RE.match(ch):
                out.append(ch)
    return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class _IndexedChunk:
    chunk_id: str
    page_id: str   # corpus uses doc_id == page_id at the page level
    tokens: Tuple[str, ...]


@dataclass
class BM25SimpleIndex:
    """In-memory BM25 index over a chunk JSONL file.

    Constructed with ``build_from_chunks_file``; once built it serves
    ``score(query)`` and ``first_rank_for_page(query, page_id)``.
    """

    chunks: List[_IndexedChunk]
    doc_lengths: List[int]
    avg_doc_length: float
    inverted_index: Dict[str, Dict[int, int]]
    document_frequency: Dict[str, int]
    k1: float
    b: float
    n_chunks: int

    def score(self, query_tokens: Sequence[str]) -> List[Tuple[int, float]]:
        """Score every chunk; return ``(chunk_index, score)`` sorted.

        Stable: ties broken by chunk_index ascending. Out-of-vocab
        terms are skipped (df=0 → no contribution) so a query whose
        every term is OOV returns an empty list rather than a flat
        zero-score listing.
        """
        scores: Dict[int, float] = {}
        if not query_tokens:
            return []
        for term in query_tokens:
            df = self.document_frequency.get(term, 0)
            if df == 0:
                continue
            idf = math.log(
                1.0 + (self.n_chunks - df + 0.5) / (df + 0.5)
            )
            postings = self.inverted_index.get(term, {})
            for chunk_idx, tf in postings.items():
                doc_len = self.doc_lengths[chunk_idx]
                if self.avg_doc_length <= 0.0:
                    norm = 1.0
                else:
                    norm = (
                        1.0 - self.b
                        + self.b * (doc_len / self.avg_doc_length)
                    )
                denom = tf + self.k1 * norm
                if denom <= 0.0:
                    continue
                contribution = idf * (tf * (self.k1 + 1.0)) / denom
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + contribution
        return sorted(scores.items(), key=lambda p: (-p[1], p[0]))

    def first_rank_for_page(
        self, query: str, page_id: str, *, max_rank: Optional[int] = None,
    ) -> Optional[int]:
        """1-based rank of the first ranked chunk whose page_id matches.

        Returns ``None`` when the page has no chunk in the ranked
        result list. ``max_rank`` caps the walk (so we don't pay for
        scanning the long tail when the silver target was clearly
        not retrieved); pass ``None`` to walk the whole list.
        """
        if not page_id:
            return None
        ranked = self.score(tokenize(query))
        for rank, (chunk_idx, _score) in enumerate(ranked, start=1):
            if max_rank is not None and rank > max_rank:
                return None
            if self.chunks[chunk_idx].page_id == page_id:
                return rank
        return None


def build_from_chunks_file(
    chunks_path: Path,
    *,
    text_field: str = "chunk_text",
    page_id_fields: Sequence[str] = ("page_id", "doc_id"),
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25SimpleIndex:
    """Build an index over a JSONL chunks file.

    ``text_field`` selects which field carries the body to tokenize
    (default ``chunk_text`` — what the production embedder also reads).
    ``page_id_fields`` is the lookup chain for the page identifier;
    we accept either ``page_id`` or ``doc_id`` since the namu corpus
    uses them interchangeably at the page level.
    """
    chunks: List[_IndexedChunk] = []
    inverted: Dict[str, Dict[int, int]] = {}
    df: Counter = Counter()
    doc_lengths: List[int] = []
    total_length = 0

    with Path(chunks_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get(text_field) or ""
            tokens = tuple(tokenize(text))
            idx = len(chunks)
            page_id = ""
            for f in page_id_fields:
                v = d.get(f)
                if v:
                    page_id = str(v)
                    break
            chunks.append(_IndexedChunk(
                chunk_id=str(d.get("chunk_id") or f"row-{idx}"),
                page_id=page_id,
                tokens=tokens,
            ))
            doc_lengths.append(len(tokens))
            total_length += len(tokens)
            for term, count in Counter(tokens).items():
                inverted.setdefault(term, {})[idx] = count
                df[term] += 1

    n = len(chunks)
    avgdl = (total_length / n) if n > 0 else 0.0
    return BM25SimpleIndex(
        chunks=chunks,
        doc_lengths=doc_lengths,
        avg_doc_length=avgdl,
        inverted_index=inverted,
        document_frequency=dict(df),
        k1=float(k1),
        b=float(b),
        n_chunks=n,
    )
