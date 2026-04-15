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
    rag_db_dsn: str = Field(
        default="host=localhost port=5432 dbname=aipipeline user=aipipeline password=aipipeline_pw",
        description="libpq connection string for the ragmeta schema. Uses the same cluster as core-api.",
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
