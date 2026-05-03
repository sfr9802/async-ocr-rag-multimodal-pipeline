"""RerankerProvider contract + cross-encoder implementation.

A reranker is a second-stage scorer that consumes the bi-encoder's
candidate list and re-ranks it with a cross-encoder model that sees the
(query, passage) pair jointly. For multilingual RAG this is consistently
the single highest-ROI retrieval change — bi-encoder recall@30 plus
cross-encoder precision@k beats bi-encoder precision@k alone by a wide
margin, especially on Korean queries.

This module is a new provider seam parallel to EmbeddingProvider and
GenerationProvider. It is NOT a method on Retriever: the Retriever
composes a reranker in the same way it composes an embedder, so the
registry can swap between CrossEncoderReranker and NoOpReranker based on
environment without Retriever itself ever growing model-specific code.

Two implementations ship here:

  1. CrossEncoderReranker — production path, lazy-loads
     sentence_transformers.CrossEncoder (BAAI/bge-reranker-v2-m3 by
     default). On a CUDA OOM the predictor halves the batch size once
     and retries; any other exception (or a second OOM after the halved
     retry) is swallowed and the original top-k is returned unchanged,
     so a flaky GPU driver or a transient OOM cannot take down the RAG
     capability. CUDA memory is logged at model load and around predict
     so the Phase 2A reranker eval can correlate latency with peak mem.

  2. NoOpReranker — offline / CI default. Returns chunks[:k] untouched.
     Registry falls back to this on init failure of the real reranker,
     which keeps the MOCK / OCR / RAG failure-isolation pattern intact:
     if the cross-encoder model can't load, RAG still registers.

Phase 2A-L latency-breakdown hook
---------------------------------
``CrossEncoderReranker`` accepts an opt-in ``collect_stage_timings`` flag
that swaps the upstream ``CrossEncoder.predict`` for a per-stage
instrumented variant — pair_build / tokenize / forward / postprocess —
recorded into ``last_breakdown_ms``. The default is ``False`` and the
production code path (Retriever / registry / RAG capability) does not
enable it, so behaviour is bit-for-bit identical when the flag is off.
The eval harness flips it on for the Phase 2A-L topN sweep so the
latency report can attribute time inside ``predict`` rather than treat
it as a single black box.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.capabilities.rag.embeddings import (
    _is_cuda_oom_exception,
    _log_cuda_memory,
    apply_cuda_alloc_conf,
)
from app.capabilities.rag.generation import RetrievedChunk

log = logging.getLogger(__name__)


class RerankerProvider(ABC):
    """Produces a re-ranked top-k list from a candidate list."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        ...


# ---------------------------------------------------------------------------
# CrossEncoder-backed reranker
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, max_length: int, device: str):
    """Process-wide cached CrossEncoder loader.

    CrossEncoder load is the expensive step (~1s + model download on
    cold miss). The cache is keyed by (model, max_length, device) so a
    test-time swap or a mid-process device change yields a fresh
    instance instead of silently reusing the previous one.
    """
    from sentence_transformers import CrossEncoder

    log.info(
        "Loading CrossEncoder reranker: model=%s max_length=%d device=%s",
        model_name, max_length, device,
    )
    encoder = CrossEncoder(model_name, max_length=max_length, device=device)
    _log_cuda_memory(f"after CrossEncoder load ({model_name})")
    return encoder


class CrossEncoderReranker(RerankerProvider):
    """Cross-encoder reranker backed by sentence-transformers.

    Scoring is deterministic: for each chunk we build a (query, passage)
    pair where the passage is truncated to ``text_max_chars`` to keep
    the tokenizer from blowing past ``max_length`` on very long chunks.
    The CrossEncoder.predict() result is attached to each chunk as
    ``rerank_score`` and the list is sorted descending.

    Any exception raised by CrossEncoder load or predict is logged at
    WARNING level and the caller gets chunks[:k] back unchanged. This
    preserves the MOCK/OCR/RAG failure-isolation pattern: a flaky
    reranker degrades to bi-encoder-only retrieval, it does not take
    down the RAG capability.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        batch_size: int = 64,
        text_max_chars: int = 800,
        device: str | None = None,
        oom_fallback_batch_size: int | None = None,
        cuda_alloc_conf: str | None = "expandable_segments:True",
        collect_stage_timings: bool = False,
    ) -> None:
        self._model_name = model_name
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        self._text_max_chars = int(text_max_chars)
        self._device_override = device
        # Default OOM fallback halves the batch_size once. Mirrors the
        # SentenceTransformerEmbedder OOM contract — caller can pin an
        # explicit smaller value, or pass 0 / negative to disable the
        # retry (we still swallow the second exception so the contract
        # of "broken predict → original top-k" is preserved).
        if oom_fallback_batch_size is None:
            self._oom_fallback_batch_size = max(1, self._batch_size // 2)
        else:
            self._oom_fallback_batch_size = max(1, int(oom_fallback_batch_size))
        # Defensive ``PYTORCH_CUDA_ALLOC_CONF`` setting for reranker-only
        # CLI paths that don't go through SentenceTransformerEmbedder
        # first. The embedder normally applies this before its own model
        # load; when it does, ``apply_cuda_alloc_conf`` is a no-op here
        # because the env var is already set. Pass ``None`` to opt out.
        self._cuda_alloc_conf = cuda_alloc_conf
        self._encoder = None  # lazy, via _load_cross_encoder cache
        # Phase 2A-L: opt-in latency-breakdown hook. Default OFF keeps the
        # production RAG path bit-for-bit identical; the eval harness
        # flips it on so the topN sweep can attribute time per stage.
        self._collect_stage_timings = bool(collect_stage_timings)
        self._last_breakdown_ms: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    @property
    def batch_size(self) -> int:
        """Configured CrossEncoder.predict batch_size.

        Surfaced read-only so the eval harness can record the value in
        a run's metadata without round-tripping through the encoder.
        The actual predict() call may use the OOM-fallback batch size
        on a retry; that's logged at WARNING and recoverable.
        """
        return self._batch_size

    @property
    def last_breakdown_ms(self) -> Optional[Dict[str, float]]:
        """Per-stage latencies recorded by the most recent ``rerank()``.

        Populated only when ``collect_stage_timings=True`` and the
        cross-encoder succeeded on the primary batch_size (no OOM
        retry). Keys: ``pair_build_ms``, ``tokenize_ms``, ``forward_ms``,
        ``postprocess_ms``, ``total_rerank_ms``. ``total_rerank_ms`` is
        the sum of the four stages — wall-clock of the same window may
        differ marginally due to instrumentation gaps. ``None`` when the
        timing hook is off, when reranking degraded to chunks[:k], or
        when the OOM-retry path took over (mixing primary + retry timings
        would be misleading).
        """
        if self._last_breakdown_ms is None:
            return None
        return dict(self._last_breakdown_ms)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        # Reset per-call so a previous breakdown can never leak forward
        # into a degenerate / OOM-retry call where the hook should
        # surface ``None`` to the eval harness.
        self._last_breakdown_ms = None

        if not chunks:
            return []
        k = max(0, int(k))
        if k == 0:
            return []
        try:
            encoder = self._ensure_encoder()
        except Exception as ex:  # noqa: BLE001 — narrow logging only
            log.warning(
                "CrossEncoderReranker model load failed (%s: %s); "
                "returning original top-%d order.",
                type(ex).__name__, ex, k,
            )
            return list(chunks[:k])

        t_pair_start = time.perf_counter()
        pairs = [
            (query, (c.text or "")[: self._text_max_chars])
            for c in chunks
        ]
        t_pair_end = time.perf_counter()

        _log_cuda_memory(f"before rerank predict ({len(pairs)} pairs)")
        predict_breakdown: Optional[Dict[str, float]] = None
        try:
            if self._collect_stage_timings:
                scores, predict_breakdown = self._predict_with_breakdown(
                    encoder, pairs, self._batch_size,
                )
            else:
                scores = self._predict(encoder, pairs, self._batch_size)
        except BaseException as ex:  # noqa: BLE001 — narrow next two lines
            if not _is_cuda_oom_exception(ex):
                log.warning(
                    "CrossEncoderReranker.predict failed (%s: %s); "
                    "returning original top-%d order.",
                    type(ex).__name__, ex, k,
                )
                return list(chunks[:k])
            new_bs = self._oom_fallback_batch_size
            if new_bs >= self._batch_size:
                log.warning(
                    "CrossEncoderReranker hit CUDA OOM at batch_size=%d "
                    "but oom_fallback_batch_size=%d is not smaller; "
                    "returning original top-%d order.",
                    self._batch_size, new_bs, k,
                )
                return list(chunks[:k])
            log.warning(
                "CrossEncoderReranker CUDA OOM at batch_size=%d (%s); "
                "retrying once at batch_size=%d.",
                self._batch_size, type(ex).__name__, new_bs,
            )
            try:
                # Retry path: skip stage breakdown — recording the OOM
                # half-batch retry's tokenize/forward would mix two
                # measurement regimes and confuse the eval report.
                scores = self._predict(encoder, pairs, new_bs)
                predict_breakdown = None
            except Exception as retry_ex:  # noqa: BLE001 — narrow logging
                log.warning(
                    "CrossEncoderReranker.predict still failing at "
                    "batch_size=%d (%s: %s); returning original top-%d "
                    "order.",
                    new_bs, type(retry_ex).__name__, retry_ex, k,
                )
                return list(chunks[:k])

        _log_cuda_memory(f"after rerank predict ({len(pairs)} pairs)")

        t_post_start = time.perf_counter()
        scored = [
            _replace_rerank_score(c, float(s))
            for c, s in zip(chunks, scores)
        ]
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        result = scored[:k]
        t_post_end = time.perf_counter()

        if self._collect_stage_timings and predict_breakdown is not None:
            pair_build_ms = round((t_pair_end - t_pair_start) * 1000.0, 3)
            postprocess_ms = round((t_post_end - t_post_start) * 1000.0, 3)
            tokenize_ms = float(predict_breakdown.get("tokenize_ms", 0.0))
            forward_ms = float(predict_breakdown.get("forward_ms", 0.0))
            total = pair_build_ms + tokenize_ms + forward_ms + postprocess_ms
            self._last_breakdown_ms = {
                "pair_build_ms": pair_build_ms,
                "tokenize_ms": round(tokenize_ms, 3),
                "forward_ms": round(forward_ms, 3),
                "postprocess_ms": postprocess_ms,
                "total_rerank_ms": round(total, 3),
            }

        return result

    # -- internals -------------------------------------------------------

    def _ensure_encoder(self):
        if self._encoder is not None:
            return self._encoder
        # IMPORTANT: set ``PYTORCH_CUDA_ALLOC_CONF`` BEFORE the
        # CrossEncoder import touches ``torch.cuda``. PyTorch reads
        # this env var at first-cuda-init time and ignores later
        # changes — same constraint the embedder honours. This is a
        # no-op when the embedder already applied the value (the
        # variable wins over a later setattr), so the eval CLI's
        # embedder-then-reranker order keeps a single warm CUDA
        # context with one consistent allocator config.
        apply_cuda_alloc_conf(self._cuda_alloc_conf)
        device = self._device_override or _auto_device()
        self._encoder = _load_cross_encoder(
            self._model_name, self._max_length, device,
        )
        return self._encoder

    def _predict(self, encoder, pairs, batch_size: int):
        """Single CrossEncoder.predict pass with the given batch_size.

        Split out so the OOM-retry path can reuse it without
        duplicating the kwargs.
        """
        return encoder.predict(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _predict_with_breakdown(
        self,
        encoder,
        pairs: List[Tuple[str, str]],
        batch_size: int,
    ) -> Tuple[Any, Dict[str, float]]:
        """Instrumented predict that records tokenize / forward per stage.

        Mirrors ``sentence_transformers.CrossEncoder.predict`` for the
        sigmoid / num_labels=1 path used by the bge-reranker-v2-m3
        family — the only path Phase 2A-L exercises. The output is the
        same numpy array of scores ``predict`` would emit, so callers
        can drop this in alongside the production ``_predict`` without
        observing any behavioural difference beyond the extra timing
        dict.

        Per-stage measurement strategy:
          - ``tokenize_ms`` covers ``encoder.tokenizer(...)`` only,
            which is a pure host-side step.
          - ``forward_ms`` covers the host→device transfer
            (``features.to(device)``), the model forward, and the
            activation. CUDA work is asynchronous, so we
            ``torch.cuda.synchronize()`` before reading ``perf_counter``
            on either side; without sync the forward time would leak
            into the next batch's tokenize and the breakdown numbers
            would be meaningless. Sync is gated on
            ``torch.cuda.is_available()`` so a CPU-only run still gets
            a usable (if approximate) forward time.

        Per-batch sums are accumulated into ``tokenize_ms`` and
        ``forward_ms`` totals; the caller adds ``pair_build_ms`` and
        ``postprocess_ms`` measured outside this method.

        Falls back to a string-match path if ``torch`` is not importable
        in this process — the unit tests rely on that for a torch-free
        breakdown smoke check.
        """
        import numpy as np

        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover — torch missing means no GPU
            torch = None  # type: ignore[assignment]

        cuda_sync = False
        if torch is not None:
            try:
                cuda_sync = bool(torch.cuda.is_available())
            except Exception:  # pragma: no cover — defensive
                cuda_sync = False

        tokenize_ms_total = 0.0
        forward_ms_total = 0.0
        pred_scores: List[Any] = []

        encoder.eval()
        # Wrap the whole loop in inference_mode when torch is around so
        # we match the upstream predict's autograd-disabled posture; CPU
        # fallback (torch == None) just runs without the context.
        if torch is not None:
            inference_ctx = torch.inference_mode()
        else:
            from contextlib import nullcontext

            inference_ctx = nullcontext()

        with inference_ctx:
            for start_index in range(0, len(pairs), batch_size):
                batch = list(pairs[start_index : start_index + batch_size])

                t0 = time.perf_counter()
                features = encoder.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                t1 = time.perf_counter()
                tokenize_ms_total += (t1 - t0) * 1000.0

                if cuda_sync:
                    try:
                        torch.cuda.synchronize()
                    except Exception:  # pragma: no cover — defensive
                        cuda_sync = False
                t2 = time.perf_counter()
                features = features.to(encoder.model.device)
                model_predictions = encoder.model(**features, return_dict=True)
                logits = encoder.activation_fn(model_predictions.logits)
                if cuda_sync:
                    try:
                        torch.cuda.synchronize()
                    except Exception:  # pragma: no cover — defensive
                        pass
                t3 = time.perf_counter()
                forward_ms_total += (t3 - t2) * 1000.0

                pred_scores.extend(logits)

        if encoder.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        # Convert to numpy on the way out — same shape and dtype the
        # upstream predict() would have produced for convert_to_numpy=True.
        scores_np = np.asarray(
            [s.cpu().detach().float().numpy() for s in pred_scores]
        )

        return scores_np, {
            "tokenize_ms": round(tokenize_ms_total, 3),
            "forward_ms": round(forward_ms_total, 3),
        }


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _replace_rerank_score(
    chunk: RetrievedChunk, score: float
) -> RetrievedChunk:
    """Return a new RetrievedChunk with ``rerank_score`` attached.

    RetrievedChunk is a frozen dataclass, so we rebuild rather than
    mutate. The original ``score`` (bi-encoder similarity) is preserved
    so downstream consumers that want to log both can do so.
    """
    return replace(chunk, rerank_score=score)


# ---------------------------------------------------------------------------
# No-op reranker
# ---------------------------------------------------------------------------


class NoOpReranker(RerankerProvider):
    """Identity reranker — returns chunks[:k] unchanged.

    This is the default when reranking is disabled and the fallback
    when CrossEncoderReranker init fails. It never touches the score
    field so a Phase 0 baseline (no rerank) reproduces bit-for-bit.
    """

    @property
    def name(self) -> str:
        return "noop"

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        k = max(0, int(k))
        return list(chunks[:k])
