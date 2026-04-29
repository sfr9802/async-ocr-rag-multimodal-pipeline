"""Phase 2 — pure-Python BM25 retriever for offline eval comparisons.

Implements a small BM25Okapi over an in-memory chunk list and exposes
it through the same ``retrieve(query) -> Report`` shape that
``run_retrieval_eval`` reads off any production retriever. No external
deps (rank_bm25 / scikit / etc.); the implementation is the standard
Robertson / BM25Okapi formulation:

    score(d, q) = Σ_t IDF(t) · (tf(t, d) · (k1 + 1)) /
                                (tf(t, d) + k1 · (1 - b + b · |d| / avgdl))

Default knobs: ``k1=1.5``, ``b=0.75``. Tokenization is whitespace +
unicode-NFKC casefold, with a CJK-aware character fallback so Korean /
Japanese / Chinese chunks (where whitespace tokens collapse the entire
sentence into a single token) still produce useful term frequency
counts. The character-fallback path emits 1-grams for CJK runs; this
matches what the production embedder would see *post*-tokenization on
average and is sufficient for an A/B against dense retrieval.

The retriever is **eval-only** — it lives under ``eval/harness/`` and
must never be imported from production code. It plugs into:

  - ``run_retrieval_eval`` directly (its ``retrieve(query)`` returns a
    duck-typed report).
  - ``RRFHybridEvalRetriever`` as the ``sparse`` arm.

Performance posture: O(|corpus| · |query_tokens|) per query against a
precomputed inverted-index dict-of-Counters; fast enough for a 50k
chunk corpus on a single machine without a vector store.
"""

from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.capabilities.rag.generation import RetrievedChunk

from eval.harness.embedding_text_builder import (
    EMBEDDING_TEXT_VARIANTS,
    EmbeddingTextInput,
    VARIANT_RAW,
    build_embedding_text,
)

log = logging.getLogger(__name__)


DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
# How many chunks the retriever returns per query. Mirrors the
# ``DEFAULT_CANDIDATE_KS`` ceiling so a single-call hybrid run can
# still feed candidate@100 metrics without re-querying the BM25 side.
DEFAULT_TOP_K = 100

# Whitespace token regex. Splits on Unicode whitespace; preserves any
# alphanumeric / CJK run as a single token before the CJK fallback.
_WS_SPLIT_RE = re.compile(r"\s+")
# CJK character class — covers Hangul syllables, Hangul jamo, Han
# ideographs (CJK Unified + Extension A), Hiragana, Katakana. Used by
# the per-token fallback that emits character 1-grams for languages
# without space-delimited words.
_CJK_RE = re.compile(
    r"[가-힯ᄀ-ᇿ㄰-㆏"
    r"一-鿿㐀-䶿"
    r"぀-ゟ゠-ヿ]"
)
# Strip non-alphanumeric characters from the tail of a whitespace token
# (commas, periods, brackets) so "bge-m3,," tokenizes the same as "bge-m3".
_NON_ALNUM_TAIL_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class BM25IndexedChunk:
    """One chunk after tokenization but before scoring.

    Carries the original chunk fields needed by ``RetrievedChunk`` plus
    the tokenized form. Frozen so the index can hash / cache safely.
    """

    chunk_id: str
    doc_id: str
    section: str
    text: str
    tokens: Tuple[str, ...]


@dataclass
class BM25Report:
    """Duck-typed report shaped to match ``RetrievalReport``.

    ``run_retrieval_eval`` reads off ``results``, ``index_version``,
    ``embedding_model``, ``reranker_name``, ``rerank_ms``,
    ``dense_retrieval_ms``, ``rerank_breakdown_ms``, and the optional
    ``candidate_doc_ids`` field this harness uses for Phase 1 candidate
    metrics. Anything not surfaced here stays at ``None`` / empty.
    """

    results: List[RetrievedChunk]
    candidate_doc_ids: List[str] = field(default_factory=list)
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    rerank_ms: Optional[float] = None
    dense_retrieval_ms: Optional[float] = None
    rerank_breakdown_ms: Optional[Dict[str, float]] = None


def tokenize_for_bm25(text: str) -> List[str]:
    """Return BM25-ready tokens for ``text``.

    Pipeline:
      1. NFKC normalise + casefold (collapses unicode width / case).
      2. Whitespace split.
      3. Strip non-alnum head/tail of each whitespace token.
      4. For tokens that contain CJK characters, *also* emit each CJK
         character as its own token (1-gram fallback). Non-CJK tokens
         keep their whitespace-token form.

    Empty / whitespace-only inputs return an empty list. The function is
    pure and deterministic — the retriever caches no state on it so
    callers can invoke it for both indexing and querying.
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
        # CJK fallback: each CJK character becomes its own token. The
        # whitespace token is also kept so Latin words inside a Korean
        # query (e.g. "bge-m3") survive intact alongside their CJK
        # siblings.
        for ch in raw:
            if _CJK_RE.match(ch):
                out.append(ch)
    return out


@dataclass
class BM25Index:
    """Precomputed BM25 statistics over an indexed chunk list.

    Built once at construction time; ``score(query_tokens) -> dict``
    is O(|query_tokens| · avg_docs_per_term). The eval driver builds
    one index per (corpus, prefix_variant) combination and reuses it
    across the dataset query loop.
    """

    chunks: List[BM25IndexedChunk]
    doc_lengths: List[int]
    avg_doc_length: float
    # term -> {chunk_index: term_frequency}
    inverted_index: Dict[str, Dict[int, int]]
    # term -> document frequency (number of chunks containing the term)
    document_frequency: Dict[str, int]
    k1: float
    b: float
    n_chunks: int

    def score(self, query_tokens: Sequence[str]) -> List[Tuple[int, float]]:
        """Score every chunk; return ``(chunk_index, score)`` pairs.

        Skips terms that don't appear in any indexed chunk (df == 0)
        rather than treating them as no-op contributions — a totally
        out-of-vocab term doesn't move ranking. Idiomatic Robertson
        IDF with the +0.5 / +0.5 smoothing keeps single-doc terms from
        producing infinite-IDF spikes.
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
        # Stable rank: highest score first, ties broken by chunk_index
        # ascending (the original index order from the source corpus).
        return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


def build_bm25_index(
    chunks: Iterable[Any],
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    embedding_text_variant: str = VARIANT_RAW,
    keyword_provider: Optional[Any] = None,
) -> BM25Index:
    """Build a ``BM25Index`` from any iterable of chunk-like objects.

    Each chunk is read via duck-typed ``.chunk_id``, ``.doc_id``,
    ``.section``, ``.text``, ``.title`` (optional), ``.keywords``
    (optional). Title and keyword fields are only consulted when the
    embedding-text variant requests them; the BM25 index then tokenizes
    *exactly the prefix-built string*, which keeps the lexical retriever
    consistent with whatever a parallel dense reindex would have seen.

    ``keyword_provider`` is an optional ``(chunk) -> Iterable[str]``
    callable for datasets where keywords aren't on the chunk object —
    e.g. project-specific metadata that the test harness assembles
    on-the-fly. Falls back to ``getattr(chunk, "keywords", ())``.
    """
    if embedding_text_variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {embedding_text_variant!r}"
        )
    indexed: List[BM25IndexedChunk] = []
    inverted: Dict[str, Dict[int, int]] = {}
    df: Counter = Counter()
    doc_lengths: List[int] = []
    total_length = 0

    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        title = getattr(chunk, "title", None)
        section = getattr(chunk, "section", None)
        if keyword_provider is not None:
            keywords = tuple(keyword_provider(chunk) or ())
        else:
            keywords = tuple(getattr(chunk, "keywords", ()) or ())
        embed_text = build_embedding_text(
            EmbeddingTextInput(
                text=text,
                title=title,
                section=section,
                keywords=tuple(str(k) for k in keywords),
            ),
            variant=embedding_text_variant,
        )
        tokens = tuple(tokenize_for_bm25(embed_text))
        idx = len(indexed)
        indexed.append(BM25IndexedChunk(
            chunk_id=str(getattr(chunk, "chunk_id", "") or f"row-{idx}"),
            doc_id=str(getattr(chunk, "doc_id", "") or ""),
            section=str(section or ""),
            text=text,
            tokens=tokens,
        ))
        doc_lengths.append(len(tokens))
        total_length += len(tokens)
        token_counts = Counter(tokens)
        for term, count in token_counts.items():
            inverted.setdefault(term, {})[idx] = count
            df[term] += 1

    n = len(indexed)
    avgdl = (total_length / n) if n > 0 else 0.0
    return BM25Index(
        chunks=indexed,
        doc_lengths=doc_lengths,
        avg_doc_length=avgdl,
        inverted_index=inverted,
        document_frequency=dict(df),
        k1=float(k1),
        b=float(b),
        n_chunks=n,
    )


class BM25EvalRetriever:
    """Eval-side BM25 retriever conforming to the harness ``retrieve()`` shape.

    Construction:
      ``BM25EvalRetriever(index, top_k=DEFAULT_TOP_K, name="bm25-okapi")``

    The retriever surfaces ``candidate_doc_ids`` (the deduplicated
    doc_id list across the BM25 score top-N) so Phase 1 candidate@K
    metrics fire even on the dense-vs-BM25 baseline. ``rerank_ms`` /
    ``dense_retrieval_ms`` stay None — BM25 has no reranker stage and
    the dense-stage timing belongs to the dense retriever.
    """

    def __init__(
        self,
        index: BM25Index,
        *,
        top_k: int = DEFAULT_TOP_K,
        name: str = "bm25-okapi",
        index_version: Optional[str] = None,
    ) -> None:
        self._index = index
        self._top_k = max(1, int(top_k))
        self._name = str(name)
        self._index_version = index_version

    @property
    def index(self) -> BM25Index:
        return self._index

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, query: str) -> BM25Report:
        """Score the corpus against ``query`` and return a top-k report."""
        started = time.perf_counter()
        tokens = tokenize_for_bm25(query)
        ranked = self._index.score(tokens)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        results: List[RetrievedChunk] = []
        seen_doc_ids: List[str] = []
        for chunk_idx, score in ranked[: self._top_k]:
            chunk = self._index.chunks[chunk_idx]
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section=chunk.section,
                text=chunk.text,
                score=float(score),
                rerank_score=None,
            ))
            if chunk.doc_id and chunk.doc_id not in seen_doc_ids:
                seen_doc_ids.append(chunk.doc_id)
        return BM25Report(
            results=results,
            candidate_doc_ids=seen_doc_ids,
            index_version=self._index_version,
            embedding_model=None,
            reranker_name=self._name,
            rerank_ms=None,
            # Surface BM25 wall-clock under dense_retrieval_ms so the
            # latency aggregator captures it — the harness treats
            # "dense" loosely as "primary retrieval stage" and BM25 is
            # the primary stage when it's the only retriever.
            dense_retrieval_ms=elapsed_ms,
        )
