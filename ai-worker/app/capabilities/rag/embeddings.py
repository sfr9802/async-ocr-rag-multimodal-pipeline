"""EmbeddingProvider contract + sentence-transformers implementation.

Exposing both passages and queries as separate methods so that we can
honour asymmetric-retrieval models (E5, bge-m3) without forcing the rest
of the codebase to know about their prefix quirks. Models that don't
need prefixes simply leave `query_prefix` / `passage_prefix` empty.

The sentence-transformers provider is the only real implementation in
phase 2. A deterministic fallback lives in the same file so unit tests
can run without downloading a real model, but the capability itself
uses sentence-transformers unless a test explicitly substitutes the
fallback.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Produces L2-normalized float32 vectors for passages and queries."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def embed_passages(self, texts: List[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Real embedding provider backed by sentence-transformers.

    Loads the model lazily on first embed() call so that importing this
    module doesn't trigger a multi-hundred-megabyte download. Output
    vectors are L2-normalized, which lets us use FAISS IndexFlatIP
    (inner product) as a cosine-similarity search.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        *,
        max_seq_length: Optional[int] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._max_seq_length = max_seq_length
        self._batch_size = int(batch_size)
        self._show_progress_bar = bool(show_progress_bar)
        self._model = None  # lazy
        self._dim: Optional[int] = None

    # -- abstract contract -----------------------------------------------

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return self._embed([self._passage_prefix + t for t in texts])

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._embed([self._query_prefix + t for t in texts])

    # -- internals -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        log.info("Loading sentence-transformers model: %s", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        # Cap the model's input window when the caller asked for it.
        # Default (None) preserves the model's own max_seq_length, which
        # for bge-m3 is 8192 — fine for short passages, but pathologically
        # expensive on a few outlier chunks (~100k chars exist in the
        # anime corpus) because attention is O(seq_len^2). Setting a cap
        # truncates only the tail of those long chunks, which is the
        # right trade-off for offline eval against namu-wiki dumps.
        if self._max_seq_length is not None:
            self._model.max_seq_length = int(self._max_seq_length)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        log.info(
            "Embedding model ready (dim=%d, max_seq_length=%s)",
            self._dim, self._model.max_seq_length,
        )

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=self._show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)


class HashingEmbedder(EmbeddingProvider):
    """Deterministic fallback provider.

    Not used at runtime — kept so unit tests can exercise the chunker /
    retriever plumbing without pulling a model off HuggingFace Hub. It
    produces L2-normalized vectors via token hashing + sign encoding,
    which is enough for the tests to verify that similar queries find
    similar passages. Semantic quality is obviously worse than a real
    transformer; don't ship it.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hashing-embedder-dim{self._dim}"

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = [t for t in text.lower().split() if t]
            if not tokens:
                continue
            v = np.zeros(self._dim, dtype=np.float32)
            for tok in tokens:
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if (h[4] & 1) else -1.0
                v[idx] += sign
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            out[i] = v
        return out
