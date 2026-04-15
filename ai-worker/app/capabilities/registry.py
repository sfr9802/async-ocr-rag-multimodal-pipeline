"""Capability registry.

A simple name → instance map. Kept deliberately dumb: no auto-discovery,
no plugin loader. New capabilities are added in one place when they
arrive.

Phase 2 registers RagCapability and OcrCapability opportunistically —
if their initialization fails (missing FAISS index, missing Tesseract
binary, missing ragmeta schema, etc.), the worker still boots with the
MockProcessor available and logs a clear warning for each missing
capability. That way a job for a capability that isn't ready fails with
a clean UNKNOWN_CAPABILITY error at runtime rather than crashing the
worker at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.capabilities.base import Capability, CapabilityError
from app.capabilities.mock_processor import MockProcessor
from app.core.config import WorkerSettings

log = logging.getLogger(__name__)


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
    registry = CapabilityRegistry()
    registry.register(MockProcessor())

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

    log.info("Active capabilities: %s", registry.available())
    return registry


def _build_rag_capability(settings: WorkerSettings) -> Capability:
    # Imports are local so a broken RAG subsystem can't take down mock-only
    # deployments on startup — ImportError on faiss / psycopg2 surfaces as
    # a clean "RAG not registered" warning.
    from app.capabilities.rag.capability import RagCapability, RagCapabilityConfig
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
    # ensure_ready() is the strict gate: on any model/dim mismatch between
    # the runtime embedder and the on-disk build.json it raises RuntimeError
    # which the caller above converts into a clean "not registered" warning.
    retriever.ensure_ready()

    return RagCapability(
        retriever=retriever,
        generator=ExtractiveGenerator(),
        config=RagCapabilityConfig(top_k=settings.rag_top_k),
    )


def _build_ocr_capability(settings: WorkerSettings) -> Capability:
    # Local imports so import failures (pytesseract / pymupdf not installed)
    # surface cleanly as "OCR not registered" instead of breaking the
    # worker's module-load phase for RAG/MOCK.
    from app.capabilities.ocr.capability import OcrCapability, OcrCapabilityConfig
    from app.capabilities.ocr.tesseract_provider import TesseractOcrProvider

    provider = TesseractOcrProvider(
        languages=settings.ocr_languages,
        pdf_dpi=settings.ocr_pdf_dpi,
        tesseract_cmd=settings.ocr_tesseract_cmd,
    )
    # Probe the Tesseract binary + language packs NOW so a missing install
    # surfaces at startup as a clean warning, not as a per-job crash.
    provider.ensure_ready()

    return OcrCapability(
        provider=provider,
        config=OcrCapabilityConfig(
            min_confidence_warn=settings.ocr_min_confidence_warn,
            max_pages=settings.ocr_max_pages,
        ),
    )
