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
from typing import Any, Optional

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

    # Build the shared LlmChatProvider once per registry build so that
    # the query parser today and the agent router / critic / rewriter
    # in later phases share a single client (and, crucially, a single
    # keep-alive slot against Ollama). Failures are logged and
    # downgraded to NoOpChatProvider — RAG falls back to regex; the
    # worker never crashes because of a broken LLM backend.
    _get_shared_llm_chat(settings)

    rag_registered = False
    ocr_registered = False

    if settings.rag_enabled:
        log.info(
            "RAG init: configured_model=%s query_prefix=%r passage_prefix=%r "
            "index_dir=%s top_k=%d generator=%s reranker=%s candidate_k=%d "
            "use_mmr=%s mmr_lambda=%.3f query_parser=%s rrf_k=%d "
            "llm_backend=%s",
            settings.rag_embedding_model,
            settings.rag_embedding_prefix_query,
            settings.rag_embedding_prefix_passage,
            settings.rag_index_dir,
            settings.rag_top_k,
            settings.rag_generator,
            settings.rag_reranker,
            settings.rag_candidate_k,
            settings.rag_use_mmr,
            settings.rag_mmr_lambda,
            settings.rag_query_parser,
            settings.rag_multi_query_rrf_k,
            settings.llm_backend,
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

    multimodal_registered = False
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
                "MULTIMODAL init: vision_provider=%s max_vision_pages=%d "
                "pdf_vision_dpi=%d emit_trace=%s default_question=%r "
                "generator=%s (retrieval top-k inherited from RAG=%d)",
                settings.multimodal_vision_provider,
                settings.multimodal_max_vision_pages,
                settings.multimodal_pdf_vision_dpi,
                settings.multimodal_emit_trace,
                settings.multimodal_default_question,
                settings.rag_generator,
                settings.rag_top_k,
            )
            try:
                registry.register(_build_multimodal_capability(settings))
                multimodal_registered = True
                log.info("MULTIMODAL capability registered.")
            except Exception as ex:
                log.warning(
                    "MULTIMODAL capability NOT registered (%s: %s). "
                    "MOCK, RAG, OCR continue to serve their own capabilities. "
                    "Check the vision provider config (Pillow decode error?) "
                    "and the multimodal_* settings, then restart the worker.",
                    type(ex).__name__, ex,
                )

    # AUTO / AGENT depend on at least one of RAG / OCR / MULTIMODAL being
    # registered — they have nothing to dispatch to otherwise. Missing
    # sub-capabilities are not fatal: the router simply can't route to
    # them and the AutoCapability returns a clarify response with a
    # reason when a missing-sub action is selected. Both AUTO and AGENT
    # are registered when any downstream exists — AUTO is the Phase 5
    # single-pass dispatcher (loop forced off), AGENT honours
    # agent_loop to enable the Phase 6 loop.
    if rag_registered or ocr_registered or multimodal_registered:
        try:
            auto_cap, agent_cap = _build_agent_capabilities(
                settings,
                registry=registry,
                rag_registered=rag_registered,
                ocr_registered=ocr_registered,
                multimodal_registered=multimodal_registered,
            )
            registry.register(auto_cap)
            registry.register(agent_cap)
            log.info(
                "AUTO + AGENT capabilities registered (router=%s "
                "confidence_threshold=%.2f rag=%s ocr=%s multimodal=%s "
                "agent_loop=%s agent_critic=%s agent_max_iter=%d).",
                settings.agent_router,
                settings.agent_confidence_threshold,
                rag_registered, ocr_registered, multimodal_registered,
                settings.agent_loop, settings.agent_critic,
                settings.agent_max_iter,
            )
        except Exception as ex:
            log.warning(
                "AUTO/AGENT capabilities NOT registered (%s: %s). MOCK, "
                "RAG, OCR, MULTIMODAL continue to serve their own "
                "capabilities. Check the agent_router / "
                "agent_confidence_threshold / agent_loop settings, "
                "then restart the worker.",
                type(ex).__name__, ex,
            )
    else:
        log.info(
            "AUTO / AGENT capabilities NOT registered: no downstream "
            "capabilities (RAG / OCR / MULTIMODAL) are available. "
            "Enable at least one, then restart the worker. MOCK "
            "remains registered."
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

    # Resolve max_vision_pages: 0/negative → all pages, but always
    # capped by ocr_max_pages so a huge PDF doesn't run away.
    raw_max = settings.multimodal_max_vision_pages
    safety_cap = settings.ocr_max_pages
    if raw_max <= 0:
        effective_max_vision_pages = safety_cap
    else:
        effective_max_vision_pages = min(raw_max, safety_cap)

    cross_modal = None
    if settings.cross_modal_enabled:
        try:
            cross_modal = _build_cross_modal_retriever(settings, retriever)
            log.info("Cross-modal retriever ready (CLIP + RRF).")
        except Exception as ex:
            log.warning(
                "Cross-modal retriever NOT available (%s: %s). "
                "MULTIMODAL will use text-only retrieval.",
                type(ex).__name__, ex,
            )

    return MultimodalCapability(
        ocr_provider=ocr_provider,
        vision_provider=vision_provider,
        retriever=retriever,
        generator=generator,
        config=MultimodalCapabilityConfig(
            pdf_vision_dpi=settings.multimodal_pdf_vision_dpi,
            max_vision_pages=effective_max_vision_pages,
            emit_trace=settings.multimodal_emit_trace,
            default_user_question=settings.multimodal_default_question,
            use_cross_modal_retrieval=cross_modal is not None,
        ),
        cross_modal_retriever=cross_modal,
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
        settings.rag_generator,
        settings.rag_claude_generation_model,
        settings.rag_reranker,
        settings.rag_candidate_k,
        settings.rag_rerank_batch,
        settings.rag_use_mmr,
        settings.rag_mmr_lambda,
        settings.rag_query_parser,
        settings.rag_multi_query_rrf_k,
    )
    cached = _shared_component_cache.get(key)
    if cached is not None:
        return cached

    from app.capabilities.rag.embeddings import (
        SentenceTransformerEmbedder,
        resolve_max_seq_length,
    )
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
        max_seq_length=resolve_max_seq_length(settings.rag_embedding_max_seq_length),
        batch_size=int(settings.rag_embedding_batch_size),
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    index = FaissIndex(Path(settings.rag_index_dir))
    reranker = _build_reranker(settings)
    query_parser = _build_query_parser(settings)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=settings.rag_top_k,
        reranker=reranker,
        candidate_k=settings.rag_candidate_k,
        use_mmr=settings.rag_use_mmr,
        mmr_lambda=settings.rag_mmr_lambda,
        query_parser=query_parser,
        multi_query_rrf_k=settings.rag_multi_query_rrf_k,
    )
    # ensure_ready() is the strict gate: on any model/dim mismatch
    # between the runtime embedder and the on-disk build.json it
    # raises RuntimeError which the caller above converts into a
    # clean "not registered" warning.
    retriever.ensure_ready()

    generator_name = (settings.rag_generator or "extractive").strip().lower()
    if generator_name == "claude":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — required when "
                "rag_generator='claude'. Set "
                "AIPIPELINE_WORKER_ANTHROPIC_API_KEY in the environment."
            )
        from app.capabilities.rag.claude_generation import ClaudeGenerationProvider

        generator = ClaudeGenerationProvider(
            api_key=settings.anthropic_api_key,
            model=settings.rag_claude_generation_model,
            timeout_seconds=settings.rag_claude_timeout_seconds,
            fallback_on_error=settings.rag_generator_fallback_on_error,
        )
        log.info(
            "RAG generator: claude-generation-v1 model=%s fallback=%s",
            settings.rag_claude_generation_model,
            settings.rag_generator_fallback_on_error,
        )
    else:
        generator = ExtractiveGenerator()

    bundle = (retriever, generator)
    _shared_component_cache[key] = bundle
    return bundle


def _build_query_parser(settings: WorkerSettings):
    """Build the query parser named by settings.rag_query_parser.

    Supported values:
      - 'off' / 'noop' / '' (default): NoOpQueryParser — passthrough,
        single-query retrieval reproduces the pre-parser Phase 2 path
        bit-for-bit.
      - 'regex': RegexQueryParser — offline keyword extractor. Emits
        ``rewrites=[]`` so the Retriever's RRF path stays dead; only
        ``normalized`` + ``keywords`` are active.
      - 'llm': LlmQueryParser — wraps the shared LlmChatProvider. On
        any provider failure it falls back to the regex parser and
        stamps ``parser_name='llm-fallback-regex'``. If the configured
        LLM backend itself downgraded to NoOp at worker startup, the
        LlmQueryParser still builds but every parse call exercises
        the fallback — visible in metrics.
    """
    from app.capabilities.rag.query_parser import (
        LlmQueryParser,
        NoOpQueryParser,
        RegexQueryParser,
    )

    name = (settings.rag_query_parser or "off").strip().lower()
    if name in ("", "off", "noop", "none", "false", "0"):
        log.info("RAG query parser disabled (noop).")
        return NoOpQueryParser()
    if name == "regex":
        log.info("RAG query parser active: regex")
        return RegexQueryParser()
    if name == "llm":
        chat = _get_shared_llm_chat(settings)
        log.info("RAG query parser active: llm (backend=%s)", chat.name)
        return LlmQueryParser(chat)

    log.warning(
        "Unknown rag_query_parser=%r. Falling back to NoOpQueryParser. "
        "Supported: 'off', 'regex', 'llm'.",
        settings.rag_query_parser,
    )
    return NoOpQueryParser()


def _get_shared_llm_chat(settings: WorkerSettings):
    """Return the process's single LlmChatProvider instance.

    Built on the first call per ``build_default_registry`` invocation.
    Any backend init failure is logged and downgraded to
    ``NoOpChatProvider`` so dependent consumers (LlmQueryParser today;
    agent router / critic / rewriter in later phases) exercise their
    fallback paths instead of taking the worker down.
    """
    key = (
        "llm_chat",
        settings.llm_backend,
        settings.llm_ollama_base_url,
        settings.llm_ollama_model,
        settings.llm_ollama_keep_alive,
        settings.llm_claude_model,
        settings.llm_timeout_seconds,
    )
    cached = _shared_component_cache.get(key)
    if cached is not None:
        return cached

    from app.clients.llm_chat import (
        ClaudeChatProvider,
        NoOpChatProvider,
        OllamaChatProvider,
    )

    backend = (settings.llm_backend or "noop").strip().lower()
    provider = None

    if backend in ("", "noop", "off", "none", "false", "0"):
        provider = NoOpChatProvider()
        log.info("LLM chat backend disabled (noop).")
    elif backend == "ollama":
        try:
            provider = OllamaChatProvider(
                base_url=settings.llm_ollama_base_url,
                model=settings.llm_ollama_model,
                timeout_s=settings.llm_timeout_seconds,
                keep_alive=settings.llm_ollama_keep_alive,
            )
            log.info(
                "LLM chat backend active: ollama base_url=%s model=%s "
                "keep_alive=%s timeout=%.1fs",
                settings.llm_ollama_base_url,
                settings.llm_ollama_model,
                settings.llm_ollama_keep_alive,
                settings.llm_timeout_seconds,
            )
        except Exception as ex:
            log.warning(
                "LLM chat backend init failed (ollama, %s: %s). "
                "Falling back to NoOpChatProvider — dependent consumers "
                "will use their offline fallback path. Ensure Ollama is "
                "running (docker compose --profile llm up -d ollama "
                "ollama-bootstrap) and AIPIPELINE_WORKER_LLM_OLLAMA_BASE_URL "
                "points at it.",
                type(ex).__name__, ex,
            )
            provider = NoOpChatProvider()
    elif backend == "claude":
        if not settings.anthropic_api_key:
            log.warning(
                "LLM chat backend 'claude' requested but "
                "AIPIPELINE_WORKER_ANTHROPIC_API_KEY is unset. Falling "
                "back to NoOpChatProvider."
            )
            provider = NoOpChatProvider()
        else:
            try:
                import anthropic

                client = anthropic.Anthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.llm_timeout_seconds,
                )
                provider = ClaudeChatProvider(
                    anthropic_client=client,
                    model=settings.llm_claude_model,
                )
                log.info(
                    "LLM chat backend active: claude model=%s",
                    settings.llm_claude_model,
                )
            except Exception as ex:
                log.warning(
                    "LLM chat backend init failed (claude, %s: %s). "
                    "Falling back to NoOpChatProvider.",
                    type(ex).__name__, ex,
                )
                provider = NoOpChatProvider()
    else:
        log.warning(
            "Unknown llm_backend=%r. Falling back to NoOpChatProvider. "
            "Supported: 'noop', 'ollama', 'claude'.",
            settings.llm_backend,
        )
        provider = NoOpChatProvider()

    _shared_component_cache[key] = provider
    return provider


def _build_reranker(settings: WorkerSettings):
    """Build the reranker named by settings.rag_reranker.

    Supported values:
      - 'off' / 'noop' / '' (default): NoOpReranker (no behaviour
        change, reproduces the Phase 0 bi-encoder-only baseline).
      - 'cross_encoder' / 'cross-encoder' / 'ce': CrossEncoderReranker
        loading sentence_transformers.CrossEncoder. On any init failure
        (ImportError on sentence_transformers / torch, CUDA init
        failure, model download failure), the registry downgrades to
        NoOpReranker and logs a warning — RAG still registers and
        continues to serve bi-encoder-only retrieval.
    """
    from app.capabilities.rag.reranker import (
        CrossEncoderReranker,
        NoOpReranker,
        RerankerProvider,
    )

    name = (settings.rag_reranker or "off").strip().lower()
    if name in ("", "off", "noop", "none", "false", "0"):
        log.info("RAG reranker disabled (noop).")
        return NoOpReranker()

    if name in ("cross_encoder", "cross-encoder", "ce"):
        try:
            reranker: RerankerProvider = CrossEncoderReranker(
                batch_size=settings.rag_rerank_batch,
            )
            log.info("RAG reranker active: %s", reranker.name)
            return reranker
        except Exception as ex:
            log.warning(
                "RAG reranker init failed (%s: %s). "
                "Falling back to NoOpReranker — RAG continues to serve "
                "bi-encoder-only retrieval. To enable: pip install "
                "sentence-transformers>=2.2, ensure network access for "
                "the BAAI/bge-reranker-v2-m3 model download, then "
                "restart the worker.",
                type(ex).__name__, ex,
            )
            return NoOpReranker()

    log.warning(
        "Unknown rag_reranker=%r. Falling back to NoOpReranker. "
        "Supported: 'off', 'cross_encoder'.",
        settings.rag_reranker,
    )
    return NoOpReranker()


def _build_cross_modal_retriever(settings: WorkerSettings, text_retriever):
    """Build the CLIP image index + RRF cross-modal retriever.

    Only called when ``settings.cross_modal_enabled`` is True. Failures
    are caught by the caller and downgraded to a warning — MULTIMODAL
    falls back to text-only retrieval.
    """
    from app.capabilities.rag.cross_modal_retriever import CrossModalRetriever
    from app.capabilities.rag.image_embeddings import ClipImageEmbedder
    from app.capabilities.rag.image_index import ImageFaissIndex
    from app.capabilities.rag.image_metadata_store import ImageMetadataStore

    cache_key = ("cross_modal", settings.cross_modal_clip_model, settings.rag_index_dir, settings.cross_modal_rrf_k)
    cached = _shared_component_cache.get(cache_key)
    if cached is not None:
        return cached

    embedder = ClipImageEmbedder(model_name=settings.cross_modal_clip_model)
    image_index = ImageFaissIndex(Path(settings.rag_index_dir))
    image_index.load()

    image_meta = ImageMetadataStore(settings.rag_db_dsn)
    image_meta.ping()

    retriever = CrossModalRetriever(
        text_retriever=text_retriever,
        image_embedder=embedder,
        image_index=image_index,
        image_metadata=image_meta,
        top_k=settings.rag_top_k,
        rrf_k=settings.cross_modal_rrf_k,
    )
    _shared_component_cache[cache_key] = retriever
    return retriever


def _build_agent_capabilities(
    settings: WorkerSettings,
    *,
    registry: "CapabilityRegistry",
    rag_registered: bool,
    ocr_registered: bool,
    multimodal_registered: bool,
) -> tuple[Capability, Capability]:
    """Build ``(AutoCapability, AgentCapability)`` sharing wiring.

    Reuses the already-registered RAG / OCR / MULTIMODAL Capability
    instances (pulled from the registry) so there is exactly one
    RagCapability / OcrCapability / MultimodalCapability in the
    process regardless of whether AUTO is enabled. Missing
    sub-capabilities become ``None`` — AgentCapability's missing-sub
    branch handles the dispatch-time failure with a typed
    ``AUTO_<sub>_UNAVAILABLE`` error code.

    AUTO is always built with ``loop_enabled=False`` so the Phase 5
    contract is bit-for-bit preserved. AGENT honours
    ``settings.agent_loop`` — ``on`` enables the loop, ``off`` (the
    default) mirrors AUTO behaviour for safe Phase 5 parity until Phase
    8 measures ``loop_recovery_rate``.

    The router is built from ``settings.agent_router``. When the env
    asks for ``llm`` but the configured LLM backend has downgraded to
    NoOpChatProvider (either because ``llm_backend`` was unset or the
    remote backend was unreachable), the registry auto-downgrades to
    the rule router with a warning — AUTO / AGENT never go down because
    of a missing LLM. The same downgrade logic applies to the critic:
    ``agent_critic=llm`` on a noop backend falls back to the rule
    critic with a warning.
    """
    from app.capabilities.agent.capability import (
        AgentCapability,
        AutoCapability,
    )
    from app.capabilities.agent.critic import (
        AgentCriticProvider,
        LlmCritic,
        NoOpCritic,
        RuleCritic,
    )
    from app.capabilities.agent.loop import LoopBudget
    from app.capabilities.agent.rewriter import (
        LlmQueryRewriter,
        NoOpQueryRewriter,
        QueryRewriterProvider,
    )
    from app.capabilities.agent.router import (
        AgentRouterProvider,
        LlmAgentRouter,
        RuleBasedAgentRouter,
    )
    from app.capabilities.agent.synthesizer import AgentSynthesizer
    from app.clients.llm_chat import NoOpChatProvider

    chat = _get_shared_llm_chat(settings)
    parser = _build_query_parser(settings)

    requested = (settings.agent_router or "rule").strip().lower()
    router: AgentRouterProvider
    if requested in ("", "rule", "off", "noop", "none", "false", "0"):
        router = RuleBasedAgentRouter()
        log.info("AUTO/AGENT router active: rule")
    elif requested == "llm":
        if isinstance(chat, NoOpChatProvider):
            log.warning(
                "AUTO/AGENT router requested=llm but llm_backend is noop "
                "(or downgraded). Falling back to rule-based router — "
                "set AIPIPELINE_WORKER_LLM_BACKEND=ollama|claude and "
                "ensure the backend is reachable to re-enable the LLM "
                "router."
            )
            router = RuleBasedAgentRouter()
        else:
            router = LlmAgentRouter(
                chat,
                parser,
                confidence_threshold=settings.agent_confidence_threshold,
            )
            log.info(
                "AUTO/AGENT router active: %s (threshold=%.2f)",
                router.name, settings.agent_confidence_threshold,
            )
    else:
        log.warning(
            "Unknown agent_router=%r. Falling back to rule-based router. "
            "Supported: 'rule', 'llm'.",
            settings.agent_router,
        )
        router = RuleBasedAgentRouter()

    rag_capability = registry.get("RAG") if rag_registered else None
    ocr_capability = registry.get("OCR") if ocr_registered else None
    multimodal_capability = registry.get("MULTIMODAL") if multimodal_registered else None

    chat_for_capability = chat if not isinstance(chat, NoOpChatProvider) else None

    # AUTO: always loop_enabled=False, no loop wiring needed.
    auto_cap = AutoCapability(
        router=router,
        parser=parser,
        rag=rag_capability,
        ocr=ocr_capability,
        multimodal=multimodal_capability,
        chat=chat_for_capability,
        direct_answer_max_tokens=settings.agent_direct_answer_max_tokens,
    )

    # AGENT: loop wiring is only activated when agent_loop='on'.
    loop_enabled = _parse_loop_flag(settings.agent_loop)
    critic: AgentCriticProvider
    rewriter: QueryRewriterProvider
    synthesizer: Optional[AgentSynthesizer] = None
    retriever = None
    generator = None

    if loop_enabled:
        critic = _build_agent_critic(settings, chat)
        rewriter = _build_agent_rewriter(settings, chat, parser)
        # Loop wiring reuses the RAG retriever + generator. We only
        # fetch them when RAG has registered (they're the same shared
        # instances used by RagCapability + MultimodalCapability). The
        # loop is only meaningfully useful when retrieval is available.
        if rag_registered:
            retriever, generator = _get_shared_retriever_bundle(settings)
            synthesizer = AgentSynthesizer(generator)
        else:
            # Loop requested but no retriever — the AGENT capability
            # falls through to the Phase 5 single-pass path because its
            # loop gate requires retriever + generator + synthesizer.
            log.warning(
                "AGENT loop requested but RAG is not registered — the "
                "loop will be inactive until a RAG retriever is "
                "available. AGENT continues to serve single-pass "
                "dispatch until then."
            )
    else:
        critic = NoOpCritic()
        rewriter = NoOpQueryRewriter()

    budget = LoopBudget(
        max_iter=settings.agent_max_iter,
        max_total_ms=settings.agent_max_total_ms,
        max_llm_tokens=settings.agent_max_llm_tokens,
        min_confidence_to_stop=settings.agent_min_stop_confidence,
    )

    agent_cap = AgentCapability(
        router=router,
        parser=parser,
        rag=rag_capability,
        ocr=ocr_capability,
        multimodal=multimodal_capability,
        chat=chat_for_capability,
        direct_answer_max_tokens=settings.agent_direct_answer_max_tokens,
        loop_enabled=loop_enabled,
        critic=critic,
        rewriter=rewriter,
        synthesizer=synthesizer,
        retriever=retriever,
        generator=generator,
        budget=budget,
    )

    return auto_cap, agent_cap


def _parse_loop_flag(raw: str) -> bool:
    """Lenient truthy parser for ``agent_loop``: on/true/1/yes -> True."""
    lowered = (raw or "").strip().lower()
    if lowered in ("on", "true", "1", "yes", "y", "enable", "enabled"):
        return True
    return False


def _build_agent_critic(settings: WorkerSettings, chat):
    """Build the AgentCriticProvider named by ``settings.agent_critic``.

    Supported values:
      - 'llm'  : LlmCritic, degrades to RuleCritic on provider failure.
        If the LLM backend downgraded to NoOp, the registry falls back
        to RuleCritic with a warning — the loop stays alive but
        semantically identical to rule.
      - 'rule' (default): RuleCritic.
      - 'noop' / 'off': NoOpCritic (loop degenerates to single-pass).

    Anything else degrades to RuleCritic with a warning.
    """
    from app.capabilities.agent.critic import (
        LlmCritic,
        NoOpCritic,
        RuleCritic,
    )
    from app.clients.llm_chat import NoOpChatProvider

    name = (settings.agent_critic or "rule").strip().lower()
    if name in ("off", "noop", "none", "false", "0"):
        log.info("AGENT critic: noop (loop effectively disabled)")
        return NoOpCritic()
    if name == "rule":
        log.info("AGENT critic: rule")
        return RuleCritic()
    if name == "llm":
        if isinstance(chat, NoOpChatProvider):
            log.warning(
                "AGENT critic requested=llm but llm_backend is noop. "
                "Falling back to rule critic — set "
                "AIPIPELINE_WORKER_LLM_BACKEND=ollama|claude to re-enable."
            )
            return RuleCritic()
        log.info("AGENT critic: llm (backend=%s)", chat.name)
        return LlmCritic(chat)
    log.warning(
        "Unknown agent_critic=%r. Falling back to rule critic. "
        "Supported: 'llm', 'rule', 'noop'.",
        settings.agent_critic,
    )
    return RuleCritic()


def _build_agent_rewriter(settings: WorkerSettings, chat, parser):
    """Build a ``QueryRewriterProvider`` for the agent loop.

    Always attempts the LLM rewriter when a live chat backend is
    available. On a NoOp backend the registry hands back a
    ``NoOpQueryRewriter`` so the loop still composes — but a loop with
    a NoOp rewriter is equivalent to a single-pass flow because the
    second iteration would re-run the same query; operators are
    expected to enable the LLM backend before turning the loop on.
    """
    from app.capabilities.agent.rewriter import (
        LlmQueryRewriter,
        NoOpQueryRewriter,
    )
    from app.clients.llm_chat import NoOpChatProvider

    if isinstance(chat, NoOpChatProvider):
        log.info("AGENT rewriter: noop (llm backend unavailable)")
        return NoOpQueryRewriter()
    log.info("AGENT rewriter: llm (backend=%s)", chat.name)
    return LlmQueryRewriter(chat)


def _build_vision_provider(settings: WorkerSettings):
    """Build the vision description provider named by settings.

    Supported values for multimodal_vision_provider:
      - 'heuristic' (default): deterministic Pillow-based fallback
      - 'claude': Claude Vision via Anthropic API (requires API key)
      - 'gemma': reuses the shared LlmChatProvider (Ollama gemma4:e2b
        by default). Auto-downgrades to the heuristic provider with a
        warning if the chat backend does not advertise vision
        capability — this keeps MULTIMODAL registrable even when the
        LLM backend is NoOp or a text-only Ollama tag.
    """
    provider_name = (settings.multimodal_vision_provider or "heuristic").strip().lower()
    if provider_name in ("", "heuristic", "pillow", "default"):
        from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider

        return HeuristicVisionProvider()

    if provider_name == "claude":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — required when "
                "multimodal_vision_provider='claude'. Set "
                "AIPIPELINE_WORKER_ANTHROPIC_API_KEY in the environment."
            )
        from app.capabilities.multimodal.claude_vision import ClaudeVisionProvider

        return ClaudeVisionProvider(
            api_key=settings.anthropic_api_key,
            model=settings.multimodal_claude_vision_model,
            timeout_seconds=settings.multimodal_claude_timeout_seconds,
        )

    if provider_name == "gemma":
        from app.capabilities.multimodal.gemma_vision import GemmaVisionProvider
        from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider

        chat = _get_shared_llm_chat(settings)
        if not chat.capabilities.get("vision"):
            log.warning(
                "MULTIMODAL vision_provider='gemma' requested but the "
                "shared chat backend %r does not advertise vision "
                "capability (capabilities=%r). Falling back to the "
                "heuristic vision provider — set "
                "AIPIPELINE_WORKER_LLM_BACKEND=ollama with a multimodal "
                "model tag (e.g. gemma4:e2b) to enable gemma vision.",
                chat.name, chat.capabilities,
            )
            return HeuristicVisionProvider()

        log.info(
            "MULTIMODAL vision provider active: gemma (backend=%s "
            "default_token_budget=%d)",
            chat.name, settings.multimodal_gemma_token_budget,
        )
        return GemmaVisionProvider(
            chat,
            default_token_budget=settings.multimodal_gemma_token_budget,
        )

    raise RuntimeError(
        f"Unknown multimodal vision provider {provider_name!r}. "
        "Supported: 'heuristic', 'claude', 'gemma'."
    )
