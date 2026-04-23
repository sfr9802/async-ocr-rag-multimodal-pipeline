"""Worker configuration.

Settings are loaded from environment variables (optionally from a `.env`
file in the worker directory). Keep this list small — everything worth
configuring surfaces via explicit names, not ad-hoc `os.environ` lookups.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Runtime configuration for the worker process."""

    # --- identity ---
    worker_id: str = Field(
        default="worker-local-1",
        description="Stable identifier used as the claim token on core-api.",
    )

    # --- core-api ---
    core_api_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for core-api, used for claim / callback / artifact upload.",
    )
    core_api_request_timeout_seconds: float = 15.0
    internal_secret: Optional[str] = Field(
        default=None,
        description=(
            "Shared secret sent as X-Internal-Secret header on /api/internal/* "
            "calls. Must match AIPIPELINE_INTERNAL_SECRET on core-api. When "
            "unset the header is omitted (core-api dev mode passes through)."
        ),
    )

    # --- redis queue ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used by the queue consumer.",
    )
    queue_pending_key: str = Field(
        default="aipipeline:jobs:pending",
        description="Redis list key the core-api LPUSHes dispatch messages onto.",
    )
    queue_block_timeout_seconds: int = Field(
        default=5,
        description="BRPOP block timeout. Shorter = more responsive shutdown.",
    )

    # --- storage resolution ---
    local_storage_root: str = Field(
        default="../local-storage",
        description=(
            "Filesystem path that mirrors the core-api local storage root. "
            "Used in phase 1 to resolve local:// URIs for reading/writing "
            "artifact bytes without routing everything through HTTP."
        ),
    )
    s3_endpoint: Optional[str] = Field(
        default=None,
        description=(
            "S3/MinIO endpoint URL for resolving s3:// storage URIs. "
            "Required when core-api uses backend=s3. Example: http://localhost:9000"
        ),
    )
    s3_region: str = Field(
        default="us-east-1",
        description="AWS region for S3. Use us-east-1 for MinIO.",
    )
    s3_access_key: Optional[str] = Field(
        default=None,
        description="S3/MinIO access key.",
    )
    s3_secret_key: Optional[str] = Field(
        default=None,
        description="S3/MinIO secret key.",
    )

    # --- processing ---
    mock_processing_delay_ms: int = Field(
        default=200,
        description="Artificial delay for the mock capability so the E2E trace is observable.",
    )

    # --- rag capability (phase 2) ---
    rag_enabled: bool = Field(
        default=True,
        description=(
            "Set to false to skip trying to initialize the RAG capability at "
            "worker startup. Useful if you only want the mock capability."
        ),
    )
    rag_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace model id used by the sentence-transformers embedder. "
            "Default is BAAI/bge-m3: multilingual, 1024-dim vectors, ~2.3 GB "
            "on first download. bge-m3 is trained WITHOUT query/passage "
            "prefixes in its dense-retrieval path, so both prefix knobs below "
            "default to empty strings. If you switch to an E5-family model "
            "(intfloat/multilingual-e5-small etc.) you MUST also set the "
            "query/passage prefixes; switching to all-MiniLM-L6-v2 gives a "
            "smaller, English-only model with 384-dim vectors. "
            "IMPORTANT: changing this value requires rebuilding the FAISS "
            "index (python -m scripts.build_rag_index --fixture) AND "
            "restarting the worker. The worker refuses to serve RAG if the "
            "runtime model name does not exactly match the one recorded in "
            "build.json."
        ),
    )
    rag_embedding_prefix_query: str = Field(
        default="",
        description=(
            "Asymmetric-retrieval prefix prepended to query texts before "
            "embedding. Leave empty for bge-m3 (default) and sentence-"
            "transformers/all-MiniLM-L6-v2. Set to 'query: ' for E5-family "
            "models. Must match what was used to build the index."
        ),
    )
    rag_embedding_prefix_passage: str = Field(
        default="",
        description=(
            "Asymmetric-retrieval prefix prepended to passage texts before "
            "embedding. Leave empty for bge-m3 (default) and sentence-"
            "transformers/all-MiniLM-L6-v2. Set to 'passage: ' for E5-family "
            "models. Must match what was used to build the index."
        ),
    )
    rag_index_dir: str = Field(
        default="../rag-data",
        description="Directory holding the FAISS index file and build metadata. Resolved relative to the worker's CWD.",
    )
    rag_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve per query.",
    )
    rag_reranker: str = Field(
        default="off",
        description=(
            "Which RerankerProvider to use after bi-encoder retrieval. "
            "Options: 'off' (NoOpReranker — returns bi-encoder top-k "
            "unchanged, default for CI / Phase 0 reproducibility), "
            "'cross_encoder' (sentence-transformers CrossEncoder, loads "
            "BAAI/bge-reranker-v2-m3 by default). On init failure the "
            "registry falls back to NoOpReranker + a warning; RAG is "
            "never taken down by a broken reranker."
        ),
    )
    rag_candidate_k: int = Field(
        default=30,
        description=(
            "Number of bi-encoder candidates fetched from FAISS before "
            "reranking. The reranker trims this down to rag_top_k. "
            "Values <= rag_top_k collapse to rag_top_k (no widening), "
            "which is what the NoOpReranker path wants."
        ),
    )
    rag_rerank_batch: int = Field(
        default=64,
        description=(
            "CrossEncoder.predict batch size. Bigger batches are faster "
            "on GPU and safe at bge-reranker-v2-m3's max_length=512. "
            "Lower to 16 or 32 if CPU OOMs on the rerank step."
        ),
    )
    rag_use_mmr: bool = Field(
        default=False,
        description=(
            "Enable the post-rerank MMR (Maximal Marginal Relevance) "
            "diversity pass. When true, the top-k is selected from the "
            "reranker's candidate list by a relevance-minus-penalty "
            "score that penalises candidates sharing a doc_id with an "
            "already-selected chunk. Off by default — env unset "
            "reproduces the Phase 1 rerank-only top-k bit-for-bit."
        ),
    )
    rag_mmr_lambda: float = Field(
        default=0.7,
        description=(
            "MMR trade-off weight in [0.0, 1.0]. value = lambda * "
            "relevance - (1 - lambda) * max_doc_id_penalty. lambda=1.0 "
            "degenerates to relevance-only (matches no-MMR ordering); "
            "lambda=0.0 maximises diversity at the cost of relevance. "
            "Default 0.7 matches the port/rag tuning sweep's best "
            "dup_rate/recall trade-off."
        ),
    )
    rag_db_dsn: str = Field(
        default="host=localhost port=5432 dbname=aipipeline user=aipipeline password=aipipeline_pw",
        description="libpq connection string for the ragmeta schema. Uses the same cluster as core-api.",
    )
    rag_generator: str = Field(
        default="extractive",
        description=(
            "Which GenerationProvider to use for RAG answers. Options: "
            "'extractive' (deterministic set-intersection heuristic, no "
            "API key needed), 'claude' (Claude LLM via Anthropic API — "
            "requires anthropic_api_key). Default is 'extractive' for "
            "CI/test compatibility."
        ),
    )
    rag_claude_generation_model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "Anthropic model id used by ClaudeGenerationProvider. "
            "Only relevant when rag_generator='claude'."
        ),
    )
    rag_generator_fallback_on_error: bool = Field(
        default=True,
        description=(
            "When rag_generator='claude' and the API call fails, "
            "fall back to the extractive generator instead of "
            "propagating the error. Set to False for strict mode."
        ),
    )
    rag_claude_timeout_seconds: float = Field(
        default=60.0,
        description=(
            "HTTP timeout (seconds) for Claude Generation API calls."
        ),
    )
    rag_query_parser: str = Field(
        default="off",
        description=(
            "Which QueryParserProvider to run before retrieval. Options: "
            "'off' (NoOpQueryParser — passthrough, single-query behaviour "
            "bit-for-bit identical to the pre-parser path, default), "
            "'regex' (RegexQueryParser — offline tokenizer that strips "
            "quotes, collapses whitespace, and extracts up to 10 "
            "deduplicated keywords with KR+EN stopword removal), "
            "'llm' (LlmQueryParser — wraps the shared LlmChatProvider, "
            "emits a real intent + up to 3 rewrites, falls back to "
            "regex on any provider failure with "
            "parser_name='llm-fallback-regex')."
        ),
    )
    rag_multi_query_rrf_k: int = Field(
        default=60,
        description=(
            "Reciprocal Rank Fusion constant for the multi-query path. "
            "Only used when the query parser emits rewrites (Phase 3's "
            "offline parsers never do — the code path is dead until the "
            "phase-4 LLM parser lands). 60 is Cormack et al.'s default; "
            "raise to flatten rank differences across rewrites."
        ),
    )

    # --- ocr capability (phase 2) ---
    ocr_enabled: bool = Field(
        default=True,
        description=(
            "Set to false to skip trying to initialize the OCR capability "
            "at worker startup. Useful when Tesseract/PyMuPDF aren't "
            "available locally. MOCK and RAG still register independently."
        ),
    )
    ocr_languages: str = Field(
        default="eng",
        description=(
            "Tesseract language pack string. Multiple languages are joined "
            "with '+' (e.g. 'eng+kor'). Every listed language must have its "
            "traineddata file installed or the OCR capability fails to "
            "register at startup."
        ),
    )
    ocr_tesseract_cmd: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to the tesseract binary. Leave unset to let "
            "pytesseract find it on PATH. On Windows this is typically "
            "'C:/Program Files/Tesseract-OCR/tesseract.exe'."
        ),
    )
    ocr_pdf_dpi: int = Field(
        default=200,
        description=(
            "DPI used to rasterize scanned PDF pages before handing them "
            "to Tesseract. 200 is a good speed/quality tradeoff; raise to "
            "300 for small fonts, lower for speed."
        ),
    )
    ocr_min_confidence_warn: float = Field(
        default=40.0,
        description=(
            "Document-level average confidence threshold (0..100). Below "
            "this, a warning is added to OCR_RESULT.warnings — the job "
            "still succeeds. Set to 0 to disable."
        ),
    )
    ocr_max_pages: int = Field(
        default=100,
        description=(
            "Hard cap on PDF page count. Above this, OCR fails with "
            "OCR_TOO_MANY_PAGES rather than silently chewing through a "
            "thousand-page scan. Raise explicitly if you need more."
        ),
    )

    # --- anthropic api (shared by claude vision + claude generation) ---
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Anthropic API key for Claude Vision and Claude Generation "
            "providers. Required when multimodal_vision_provider='claude' "
            "or rag_generator='claude'. SecretStr is avoided because "
            "pydantic-settings env parsing is simpler with plain Optional[str]."
        ),
    )

    # --- shared LLM chat backend (phase 4+) ---
    llm_backend: str = Field(
        default="noop",
        description=(
            "Which LlmChatProvider to build once per worker. The provider "
            "is shared by the LLM-backed query parser today and by the "
            "agent router / critic / rewriter in later phases. Options: "
            "'noop' (default, every chat call raises — consumers fall "
            "back to their offline path), 'ollama' (local Ollama server, "
            "gemma4:e2b by default, see llm_ollama_* knobs below), "
            "'claude' (Anthropic API, requires anthropic_api_key). "
            "Init failure downgrades to noop with a warning; RAG never "
            "goes down because of a broken LLM backend."
        ),
    )
    llm_timeout_seconds: float = Field(
        default=15.0,
        description=(
            "Default per-call timeout (seconds) passed to LlmChatProvider "
            "methods. Callers can still override per-call."
        ),
    )
    llm_ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Ollama HTTP API base URL. Use http://ollama:11434 when the "
            "worker runs inside the compose network alongside the ollama "
            "service; use http://localhost:11434 when the worker runs on "
            "the host and ollama is exposed on the default port."
        ),
    )
    llm_ollama_model: str = Field(
        default="gemma4:e2b",
        description=(
            "Model tag served by Ollama. The bootstrap companion in "
            "docker-compose.yml pulls this model on first compose up. "
            "Switch to a smaller gemma4/llama variant on CPU-only hosts."
        ),
    )
    llm_ollama_keep_alive: str = Field(
        default="30m",
        description=(
            "Duration Ollama keeps the model resident between requests "
            "(forwarded as the 'keep_alive' field on every /api/chat "
            "call). Shorter reclaims VRAM sooner; longer avoids cold "
            "loads for bursty workloads."
        ),
    )
    llm_claude_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description=(
            "Anthropic model id used when llm_backend='claude'. Defaults "
            "to the smallest latest-gen Claude because query parsing / "
            "routing / critique fit comfortably inside that budget."
        ),
    )

    # --- multimodal capability (phase 2, v1) ---
    multimodal_enabled: bool = Field(
        default=True,
        description=(
            "Set to false to skip trying to initialize the MULTIMODAL "
            "capability at worker startup. MULTIMODAL v1 depends on the "
            "OCR capability and the RAG retriever — if either of those "
            "is not available, the multimodal builder is skipped "
            "automatically with a clear warning (MOCK/RAG/OCR remain "
            "unaffected)."
        ),
    )
    multimodal_vision_provider: str = Field(
        default="heuristic",
        description=(
            "Which VisionDescriptionProvider to build. Options: "
            "'heuristic' (deterministic Pillow-based fallback, no API "
            "key needed), 'claude' (Claude Vision via Anthropic API — "
            "requires anthropic_api_key). Default is 'heuristic' for "
            "CI/offline compatibility."
        ),
    )
    multimodal_claude_vision_model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "Anthropic model id used by ClaudeVisionProvider. Only "
            "relevant when multimodal_vision_provider='claude'."
        ),
    )
    multimodal_claude_timeout_seconds: float = Field(
        default=30.0,
        description=(
            "HTTP timeout (seconds) for Claude Vision API calls. "
            "Includes retries — total wall-clock may be up to "
            "3x this value on transient failures."
        ),
    )
    multimodal_max_vision_pages: int = Field(
        default=3,
        description=(
            "Maximum number of PDF pages to send through the vision "
            "provider. 0 or negative means all pages, but the effective "
            "count is always capped by ocr_max_pages to prevent runaway "
            "rasterization on huge documents."
        ),
    )
    multimodal_pdf_vision_dpi: int = Field(
        default=150,
        description=(
            "DPI used to rasterize PDF pages for the vision provider. "
            "The OCR stage walks every page via PyMuPDF + Tesseract; "
            "the vision stage sends up to multimodal_max_vision_pages "
            "pages through the vision provider."
        ),
    )
    multimodal_emit_trace: bool = Field(
        default=False,
        description=(
            "When true, MULTIMODAL jobs also emit a MULTIMODAL_TRACE "
            "artifact alongside OCR_TEXT / VISION_RESULT / "
            "RETRIEVAL_RESULT / FINAL_RESPONSE. Off by default to keep "
            "the artifact count at 4 for most consumers."
        ),
    )
    multimodal_default_question: str = Field(
        default="",
        description=(
            "Fallback user question used when a MULTIMODAL job does "
            "NOT supply an INPUT_TEXT artifact. Leave empty to let the "
            "fusion helper choose a neutral default query on its own."
        ),
    )

    # --- auto / agent router (phase 3) ---
    agent_router: str = Field(
        default="rule",
        description=(
            "Which AgentRouterProvider to build for the AUTO capability. "
            "Options: 'rule' (RuleBasedAgentRouter — deterministic 5-branch "
            "decision tree over the (text, file) pair, default), 'llm' "
            "(LlmAgentRouter — wraps the shared LlmChatProvider with a "
            "function-calling / JSON-mode decision, falls back to the rule "
            "router on low confidence or provider failure). When 'llm' is "
            "selected but llm_backend is 'noop' the registry auto-downgrades "
            "to 'rule' with a warning so AUTO never goes down because of a "
            "missing LLM."
        ),
    )
    agent_confidence_threshold: float = Field(
        default=0.55,
        description=(
            "Minimum confidence the LlmAgentRouter must self-report for its "
            "decision to be accepted. Below this threshold the router "
            "degrades to its rule-based fallback and stamps "
            "router_name='llm-*-fallback-rule' so the downgrade is visible "
            "in trace / metrics. Ignored when agent_router='rule'."
        ),
    )
    agent_direct_answer_max_tokens: int = Field(
        default=512,
        description=(
            "Max tokens requested from the chat backend when the LLM router "
            "selects 'direct_answer' and AutoCapability answers inline. "
            "Keep it small — direct_answer is for trivial prompts, longer "
            "responses should go through RAG."
        ),
    )

    # --- agent loop (phase 6) ---
    agent_loop: str = Field(
        default="off",
        description=(
            "Enable the Phase 6 iterative agent loop on the AGENT "
            "capability. Options: 'on' or 'off' (default). When 'off', "
            "AGENT degenerates to the Phase 5 single-pass dispatcher "
            "(bit-for-bit identical to AUTO). When 'on', AGENT runs a "
            "critic/rewriter/retrieve cycle up to agent_max_iter times "
            "on rag/multimodal actions, then synthesizes the final "
            "answer over the UNION of every iteration's retrieved "
            "chunks. Off-by-default until Phase 8 measures the "
            "recovery rate."
        ),
    )
    agent_critic: str = Field(
        default="rule",
        description=(
            "Which AgentCriticProvider the loop uses. Options: 'llm' "
            "(function-calling + thinking mode on backends that advertise "
            "them, falls back to rule on any provider failure), 'rule' "
            "(deterministic heuristic — short answers or 'I don't know' "
            "markers flag missing_facts), 'noop' (always sufficient — "
            "loop degenerates to single-pass). Ignored when agent_loop='off'."
        ),
    )
    agent_max_iter: int = Field(
        default=3,
        description=(
            "Hard cap on loop iterations. Budget shared across critic + "
            "rewriter + retrieve calls per iteration. 3 is tuned to the "
            "Phase 6 acceptance criteria — enough to recover from a weak "
            "initial retrieval without paying runaway tokens."
        ),
    )
    agent_max_total_ms: int = Field(
        default=15_000,
        description=(
            "Hard wall-clock cap for the loop, in milliseconds. Breaching "
            "this triggers stop_reason='time_cap' and the best-so-far "
            "answer is returned. 15s stays below TaskRunner's default "
            "job-level deadline."
        ),
    )
    agent_max_llm_tokens: int = Field(
        default=4_000,
        description=(
            "Hard cap on LLM tokens accumulated across critic + rewriter "
            "calls. Breaching it triggers stop_reason='token_cap'. "
            "Execution-side token use (e.g. Claude generator) is counted "
            "too when the executor reports it."
        ),
    )
    agent_min_stop_confidence: float = Field(
        default=0.75,
        description=(
            "Minimum critic confidence required for a 'sufficient' verdict "
            "to actually stop the loop. A low-confidence sufficient "
            "judgment still triggers another iteration up to the iter "
            "budget. Range [0.0, 1.0]."
        ),
    )

    # --- cross-modal retrieval (opt-in) ---
    cross_modal_enabled: bool = Field(
        default=False,
        description=(
            "When true, MULTIMODAL jobs use CLIP image retrieval "
            "alongside text RAG, merged via Reciprocal Rank Fusion. "
            "Requires a CLIP image index built by "
            "build_rag_index.py --with-images. Off by default."
        ),
    )
    cross_modal_clip_model: str = Field(
        default="sentence-transformers/clip-ViT-B-32",
        description=(
            "sentence-transformers model id for CLIP image/text "
            "embeddings. 512-dim. Changing this requires rebuilding "
            "the image index."
        ),
    )
    cross_modal_rrf_k: int = Field(
        default=60,
        description=(
            "RRF constant k. Higher values flatten rank differences "
            "across text and image result lists. Default 60 is the "
            "standard value from Cormack et al."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="AIPIPELINE_WORKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_settings: WorkerSettings | None = None


def get_settings() -> WorkerSettings:
    """Return a process-global settings singleton.

    Tests may monkeypatch this to inject their own config.
    """
    global _settings
    if _settings is None:
        _settings = WorkerSettings()
    return _settings
