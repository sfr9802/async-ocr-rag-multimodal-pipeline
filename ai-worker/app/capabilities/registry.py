"""Capability registry.

A simple name → instance map. Kept deliberately dumb: no auto-discovery,
no plugin loader. New capabilities are added in one place when they
arrive.

Registrations are opportunistic — if RAG, OCR, or MULTIMODAL init
fails (missing FAISS index, missing Tesseract binary, missing ragmeta
schema, Pillow ImportError, etc.), the worker still boots with the
MockProcessor available and logs a clear warning for each missing
capability. A job for a capability that isn't ready fails with a
clean UNKNOWN_CAPABILITY error at runtime rather than crashing the
worker at startup.

MULTIMODAL is a dependent capability: it needs both a working OCR
provider and a working RAG retriever. The registry enforces this
dependency explicitly — if either parent is missing, MULTIMODAL is
skipped with a warning that names the missing parent. MOCK, RAG,
and OCR register independently of multimodal state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.capabilities.base import Capability, CapabilityError
from app.capabilities.mock_processor import MockProcessor
from app.core.config import WorkerSettings

log = logging.getLogger(__name__)


# Module-level cache of shared sub-components so RAG + MULTIMODAL
# don't each load their own sentence-transformers model and FAISS
# index, and OCR + MULTIMODAL don't each probe Tesseract twice.
#
# The cache is cleared at the start of every build_default_registry
# call so repeat invocations (tests, worker restarts in-process, etc.)
# get fresh instances without leaking state across runs.
_shared_component_cache: dict[str, Any] = {}


class CapabilityRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if not capability.name:
            raise ValueError("Capability.name must not be blank")
        self._by_name[capability.name] = capability

    def get(self, name: str) -> Capability:
        if name not in self._by_name:
            raise CapabilityError("UNKNOWN_CAPABILITY", f"no capability registered for {name}")
        return self._by_name[name]

    def available(self) -> list[str]:
        return sorted(self._by_name.keys())


def build_default_registry(settings: WorkerSettings) -> CapabilityRegistry:
    _shared_component_cache.clear()

    registry = CapabilityRegistry()
    registry.register(MockProcessor())

    rag_registered = False
    ocr_registered = False

    if settings.rag_enabled:
        log.info(
            "RAG init: configured_model=%s query_prefix=%r passage_prefix=%r "
            "index_dir=%s top_k=%d",
            settings.rag_embedding_model,
            settings.rag_embedding_prefix_query,
            settings.rag_embedding_prefix_passage,
            settings.rag_index_dir,
            settings.rag_top_k,
        )
        try:
            registry.register(_build_rag_capability(settings))
            rag_registered = True
            log.info("RAG capability registered.")
        except Exception as ex:
            log.warning(
                "RAG capability NOT registered (%s: %s). "
                "Worker still serves the MOCK capability. "
                "To enable RAG: rebuild the FAISS index with the configured "
                "embedding model (python -m scripts.build_rag_index --fixture), "
                "ensure the ragmeta schema exists, then restart the worker.",
                type(ex).__name__, ex,
            )

    if settings.ocr_enabled:
        log.info(
            "OCR init: languages=%s pdf_dpi=%d tesseract_cmd=%s "
            "min_conf_warn=%.1f max_pages=%d",
            settings.ocr_languages,
            settings.ocr_pdf_dpi,
            settings.ocr_tesseract_cmd or "<PATH>",
            settings.ocr_min_confidence_warn,
            settings.ocr_max_pages,
        )
        try:
            registry.register(_build_ocr_capability(settings))
            ocr_registered = True
            log.info("OCR capability registered.")
        except Exception as ex:
            log.warning(
                "OCR capability NOT registered (%s: %s). "
                "Worker still serves the other registered capabilities. "
                "To enable OCR: install Tesseract (https://tesseract-ocr.github.io/), "
                "pip install pytesseract pymupdf, make sure the configured "
                "language packs are present, then restart the worker.",
                type(ex).__name__, ex,
            )

    if settings.multimodal_enabled:
        if not ocr_registered:
            log.warning(
                "MULTIMODAL capability NOT registered: OCR capability is "
                "unavailable. MULTIMODAL v1 reuses the OCR provider — enable "
                "and fix OCR first, then restart the worker. MOCK, RAG "
                "remain registered."
            )
        elif not rag_registered:
            log.warning(
                "MULTIMODAL capability NOT registered: RAG capability is "
                "unavailable. MULTIMODAL v1 reuses the RAG retriever + "
                "generator to feed the fused OCR + vision context into the "
                "existing text-RAG path — enable and fix RAG first, then "
                "restart the worker. MOCK, OCR remain registered."
            )
        else:
            log.info(
                "MULTIMODAL init: vision_provider=%s pdf_vision_dpi=%d "
                "emit_trace=%s default_question=%r (retrieval top-k "
                "inherited from RAG=%d)",
                settings.multimodal_vision_provider,
                settings.multimodal_pdf_vision_dpi,
                settings.multimodal_emit_trace,
                settings.multimodal_default_question,
                settings.rag_top_k,
            )
            try:
                registry.register(_build_multimodal_capability(settings))
                log.info("MULTIMODAL capability registered.")
            except Exception as ex:
                log.warning(
                    "MULTIMODAL capability NOT registered (%s: %s). "
                    "MOCK, RAG, OCR continue to serve their own capabilities. "
                    "Check the vision provider config (Pillow decode error?) "
                    "and the multimodal_* settings, then restart the worker.",
                    type(ex).__name__, ex,
                )

    log.info("Active capabilities: %s", registry.available())
    return registry


# ----------------------------------------------------------------------
# individual capability builders
# ----------------------------------------------------------------------


def _build_rag_capability(settings: WorkerSettings) -> Capability:
    # Imports are local so a broken RAG subsystem can't take down mock-only
    # deployments on startup — ImportError on faiss / psycopg2 surfaces as
    # a clean "RAG not registered" warning.
    from app.capabilities.rag.capability import RagCapability, RagCapabilityConfig

    retriever, generator = _get_shared_retriever_bundle(settings)
    return RagCapability(
        retriever=retriever,
        generator=generator,
        config=RagCapabilityConfig(top_k=settings.rag_top_k),
    )


def _build_ocr_capability(settings: WorkerSettings) -> Capability:
    # Local imports so import failures (pytesseract / pymupdf not installed)
    # surface cleanly as "OCR not registered" instead of breaking the
    # worker's module-load phase for RAG/MOCK.
    from app.capabilities.ocr.capability import OcrCapability, OcrCapabilityConfig

    provider = _get_shared_ocr_provider(settings)
    return OcrCapability(
        provider=provider,
        config=OcrCapabilityConfig(
            min_confidence_warn=settings.ocr_min_confidence_warn,
            max_pages=settings.ocr_max_pages,
        ),
    )


def _build_multimodal_capability(settings: WorkerSettings) -> Capability:
    """Build the MULTIMODAL capability.

    Only called when BOTH OCR and RAG have already registered
    successfully, so the shared OcrProvider / Retriever instances are
    already in `_shared_component_cache`. Failures here surface as a
    clean "MULTIMODAL NOT registered" warning upstream.
    """
    # Local imports for the same reason as _build_rag_capability:
    # a broken multimodal subsystem shouldn't take down MOCK/RAG/OCR.
    from app.capabilities.multimodal.capability import (
        MultimodalCapability,
        MultimodalCapabilityConfig,
    )

    ocr_provider = _get_shared_ocr_provider(settings)
    retriever, generator = _get_shared_retriever_bundle(settings)
    vision_provider = _build_vision_provider(settings)

    return MultimodalCapability(
        ocr_provider=ocr_provider,
        vision_provider=vision_provider,
        retriever=retriever,
        generator=generator,
        config=MultimodalCapabilityConfig(
            pdf_vision_dpi=settings.multimodal_pdf_vision_dpi,
            emit_trace=settings.multimodal_emit_trace,
            default_user_question=settings.multimodal_default_question,
        ),
    )


# ----------------------------------------------------------------------
# shared sub-component helpers — cached per build_default_registry call
# ----------------------------------------------------------------------


def _get_shared_ocr_provider(settings: WorkerSettings):
    """Return the process's single OcrProvider instance.

    Built on first call, cached under a settings-derived key so that
    a test-time WorkerSettings override with different OCR knobs gets
    its own instance instead of inheriting a stale one. The cache is
    wiped at the top of `build_default_registry`, so in a long-lived
    worker process there's exactly one live OcrProvider per capability
    registry build.
    """
    key = (
        "ocr_provider",
        settings.ocr_languages,
        settings.ocr_pdf_dpi,
        settings.ocr_tesseract_cmd,
    )
    cached = _shared_component_cache.get(key)
    if cached is not None:
        return cached

    from app.capabilities.ocr.tesseract_provider import TesseractOcrProvider

    provider = TesseractOcrProvider(
        languages=settings.ocr_languages,
        pdf_dpi=settings.ocr_pdf_dpi,
        tesseract_cmd=settings.ocr_tesseract_cmd,
    )
    # Probe the Tesseract binary + language packs NOW so a missing
    # install surfaces at startup as a clean warning.
    provider.ensure_ready()
    _shared_component_cache[key] = provider
    return provider


def _get_shared_retriever_bundle(settings: WorkerSettings):
    """Return (retriever, generator) pair shared by RAG + MULTIMODAL.

    Built once per `build_default_registry` call. The RAG and MM
    capabilities both want a live `Retriever` pointed at the same
    FAISS index with the same embedding model loaded — loading it
    twice would double memory for no benefit.
    """
    key = (
        "retriever_bundle",
        settings.rag_embedding_model,
        settings.rag_embedding_prefix_query,
        settings.rag_embedding_prefix_passage,
        settings.rag_index_dir,
        settings.rag_top_k,
        settings.rag_db_dsn,
    )
    cached = _shared_component_cache.get(key)
    if cached is not None:
        return cached

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.generation import ExtractiveGenerator
    from app.capabilities.rag.metadata_store import RagMetadataStore
    from app.capabilities.rag.retriever import Retriever

    metadata = RagMetadataStore(settings.rag_db_dsn)
    metadata.ping()

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    index = FaissIndex(Path(settings.rag_index_dir))
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=settings.rag_top_k,
    )
    # ensure_ready() is the strict gate: on any model/dim mismatch
    # between the runtime embedder and the on-disk build.json it
    # raises RuntimeError which the caller above converts into a
    # clean "not registered" warning.
    retriever.ensure_ready()

    generator = ExtractiveGenerator()
    bundle = (retriever, generator)
    _shared_component_cache[key] = bundle
    return bundle


def _build_vision_provider(settings: WorkerSettings):
    """Build the vision description provider named by settings.

    v1 ships only the 'heuristic' provider. Adding a real VLM later
    is a single `elif` branch — the rest of the multimodal capability
    stays put because the provider seam is already abstracted.
    """
    from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider

    provider_name = (settings.multimodal_vision_provider or "heuristic").strip().lower()
    if provider_name in ("", "heuristic", "pillow", "default"):
        return HeuristicVisionProvider()
    raise RuntimeError(
        f"Unknown multimodal vision provider {provider_name!r}. "
        "v1 supports 'heuristic' only — future phases can add 'blip2', "
        "'claude-vision', etc."
    )
