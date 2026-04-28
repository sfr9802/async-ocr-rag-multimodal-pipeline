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
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA allocator + memory-stat helpers (Phase 1C).
#
# Phase 1C diagnosed that RTX-class GPUs running bge-m3 over the namu-wiki
# corpus see a slow upward ramp in `nvidia-smi memory.used` because PyTorch's
# default caching allocator keeps a separate free pool per power-of-two size
# class. Variable-length batches (each padded to a different L) create new
# size classes monotonically, so reserved memory only ever grows. PyTorch
# 2.1+ ships an ``expandable_segments`` allocator backend that keeps a
# single growable segment instead — fragmentation effectively disappears
# for inference workloads. This is opt-in, so we set the env var ourselves
# from the embedder's lazy ``_load`` (before the model touches CUDA).
# ---------------------------------------------------------------------------


def apply_cuda_alloc_conf(value: Optional[str]) -> Optional[str]:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` if the caller asked for a value.

    No-op when ``value`` is None or empty (caller opted out, or settings
    explicitly cleared the knob). When ``PYTORCH_CUDA_ALLOC_CONF`` is
    already set (e.g. from the shell), the existing value wins so an
    operator can override settings without touching the code.

    Returns the value actually in effect after this call (the env-var's
    current value), so callers can log it for the audit trail.
    """
    if not value:
        return os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing:
        # Operator-set value wins; record the divergence for the log.
        if existing != value:
            log.info(
                "PYTORCH_CUDA_ALLOC_CONF already set to %r; "
                "leaving as-is (settings wanted %r)",
                existing, value,
            )
        return existing
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    log.info("Set PYTORCH_CUDA_ALLOC_CONF=%s", value)
    return value


def cuda_memory_stats() -> Optional[dict]:
    """Return a dict of (allocated, reserved, max_reserved) in MiB, or None.

    Returns None when torch isn't importable or CUDA isn't initialised —
    the caller should treat None as "no GPU stats available" and skip
    the log line. Stats are reported in MiB with one-decimal precision so
    they line up nicely with ``nvidia-smi`` numbers.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    mib = 1024 * 1024
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / mib, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / mib, 1),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / mib, 1),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / mib, 1),
    }


def reset_cuda_peak_stats() -> None:
    """Reset PyTorch's peak-memory counters.

    Called after a `_load()` so the post-encode max stats reflect the
    encode pass only, not the model-load spike. Silent no-op when CUDA
    is unavailable.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    torch.cuda.reset_peak_memory_stats()


def _log_cuda_memory(label: str) -> None:
    stats = cuda_memory_stats()
    if stats is None:
        return
    log.info(
        "CUDA mem [%s]: allocated=%.1f MiB · reserved=%.1f MiB · "
        "max_reserved=%.1f MiB · max_allocated=%.1f MiB",
        label,
        stats["allocated_mib"], stats["reserved_mib"],
        stats["max_reserved_mib"], stats["max_allocated_mib"],
    )


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    """Detect CUDA OOM errors across torch versions.

    torch >= 2.x exposes ``torch.cuda.OutOfMemoryError`` (subclass of
    ``RuntimeError``). Older versions raise a plain ``RuntimeError``
    whose message contains ``"CUDA out of memory"``. Handle both.
    """
    try:
        import torch
        oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_class is not None and isinstance(exc, oom_class):
            return True
    except ImportError:
        pass
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


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


def resolve_max_seq_length(value: Optional[int]) -> Optional[int]:
    """Normalise a configured ``rag_embedding_max_seq_length`` to ``None``
    or a positive int.

    The setting accepts ``0`` (or any non-positive value) as the
    explicit "no cap, use the model's own default" escape hatch. Any
    positive value is used verbatim. Centralising this keeps
    registry / scripts / eval CLIs reading the same setting in the
    same way.
    """
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


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
        cuda_alloc_conf: Optional[str] = None,
        oom_fallback_batch_size: Optional[int] = None,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._max_seq_length = max_seq_length
        self._batch_size = int(batch_size)
        self._show_progress_bar = bool(show_progress_bar)
        self._cuda_alloc_conf = cuda_alloc_conf
        # Default OOM fallback halves the batch_size once. Caller can
        # set to None to disable the retry, or to an explicit smaller
        # value to control where the fallback lands.
        if oom_fallback_batch_size is None:
            self._oom_fallback_batch_size = max(1, self._batch_size // 2)
        else:
            self._oom_fallback_batch_size = max(1, int(oom_fallback_batch_size))
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
        # IMPORTANT: set the CUDA allocator env var BEFORE the
        # SentenceTransformer import touches torch.cuda. PyTorch reads
        # PYTORCH_CUDA_ALLOC_CONF at first-cuda-init time and ignores
        # later changes; lazy-loading + setting it here is the
        # single-threaded safe way to apply Phase 1C's
        # ``expandable_segments`` recommendation.
        apply_cuda_alloc_conf(self._cuda_alloc_conf)

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
            "Embedding model ready (dim=%d, max_seq_length=%s, "
            "batch_size=%d, oom_fallback_batch_size=%d)",
            self._dim, self._model.max_seq_length,
            self._batch_size, self._oom_fallback_batch_size,
        )
        _log_cuda_memory("after model load")
        # Reset peak counters so the next ``_embed`` call's max_*
        # numbers reflect that pass only, not the model-load spike.
        reset_cuda_peak_stats()

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        assert self._model is not None

        bs = self._batch_size
        try:
            vectors = self._encode_with(texts, bs)
        except BaseException as exc:  # noqa: BLE001 — narrow next line
            if not _is_cuda_oom_exception(exc):
                raise
            new_bs = self._oom_fallback_batch_size
            if new_bs >= bs:
                raise
            log.warning(
                "CUDA OOM at batch_size=%d (%s); retrying once at "
                "batch_size=%d. Phase 1C plan: do not loop further; "
                "raise if this also OOMs.",
                bs, type(exc).__name__, new_bs,
            )
            # Phase 1C decision: do NOT call torch.cuda.empty_cache()
            # here. PyTorch's caching allocator already empties its
            # free pool internally before raising OOM, so a manual
            # empty_cache adds no headroom for the retry. Avoiding
            # the call also keeps this fallback path free of any
            # implicit CUDA-driver init, which lets the OOM-fallback
            # unit tests run in CPU-only environments without
            # touching torch.cuda module state.
            vectors = self._encode_with(texts, new_bs)

        _log_cuda_memory(f"after encode ({len(texts)} texts)")
        return vectors.astype(np.float32, copy=False)

    def _encode_with(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Single ``model.encode`` call with the given batch_size.

        Split out so the OOM-retry path can reuse it without
        duplicating the kwargs.
        """
        assert self._model is not None
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self._show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


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
