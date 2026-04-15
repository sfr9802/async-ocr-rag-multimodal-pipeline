"""Build (or rebuild) the FAISS RAG index.

Reads a JSONL dataset in the port/rag schema, chunks each document with
the same constants port/rag uses, embeds every chunk via
sentence-transformers, builds a FAISS IndexFlatIP, and persists both the
vectors (on disk) and the document/chunk metadata (PostgreSQL ragmeta
schema).

Usage (from ai-worker/):

  # fixture dataset (small, committed for tests + first-run smoke)
  python -m scripts.build_rag_index --fixture

  # real dataset from port/rag
  python -m scripts.build_rag_index --input D:/port/rag/app/scripts/namu_anime_v3.jsonl

The worker must be RESTARTED after a rebuild so that the long-lived
RagCapability picks up the new index. Rebuilds are not hot-reloaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.ingest import IngestService
from app.capabilities.rag.metadata_store import RagMetadataStore
from app.core.config import get_settings
from app.core.logging import configure_logging

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "anime_sample.jsonl"


def main() -> int:
    configure_logging()
    log = logging.getLogger("scripts.build_rag_index")

    parser = argparse.ArgumentParser(description="Build or rebuild the RAG index.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a JSONL dataset in the port/rag schema.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Ignore --input and use the committed test fixture.",
    )
    parser.add_argument(
        "--index-version",
        type=str,
        default=None,
        help="Override the index version label. Defaults to v-<epoch>.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Free-form notes recorded on the index_builds row.",
    )
    args = parser.parse_args()

    if args.fixture:
        input_path = DEFAULT_FIXTURE
    elif args.input is not None:
        input_path = args.input
    else:
        parser.error("Provide --input <path> or --fixture")
        return 2

    if not input_path.exists():
        log.error("Dataset not found: %s", input_path)
        return 2

    settings = get_settings()
    log.info("Dataset:        %s", input_path)
    log.info("Index dir:      %s", settings.rag_index_dir)
    log.info("Embed model:    %s", settings.rag_embedding_model)
    log.info("Query prefix:   %r", settings.rag_embedding_prefix_query)
    log.info("Passage prefix: %r", settings.rag_embedding_prefix_passage)
    log.info("DB DSN:         %s", _redact(settings.rag_db_dsn))

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    metadata = RagMetadataStore(settings.rag_db_dsn)
    metadata.ping()

    index = FaissIndex(Path(settings.rag_index_dir))
    ingester = IngestService(
        embedder=embedder,
        metadata_store=metadata,
        index=index,
    )

    started = time.time()
    result = ingester.ingest_jsonl(
        input_path,
        source_label=input_path.name,
        index_version=args.index_version,
        notes=args.notes,
    )
    elapsed = time.time() - started

    # Log the final build identity from the index itself, not from settings —
    # both should agree, but we want the log line to reflect exactly what was
    # written into build.json and ragmeta.index_builds so ops can grep for it.
    log.info(
        "Ingest done in %.1fs: %d documents, %d chunks, version=%s, "
        "embedding_model=%s, embedding_dim=%d",
        elapsed,
        result.document_count,
        result.chunk_count,
        result.info.index_version,
        result.info.embedding_model,
        result.info.dimension,
    )

    stats = metadata.stats()
    log.info("ragmeta stats: %s", stats)
    log.info(
        "Next steps: restart the worker so it picks up the new index. "
        "The worker will refuse to register the RAG capability if its "
        "configured embedding model does not match %r.",
        result.info.embedding_model,
    )
    return 0


def _redact(dsn: str) -> str:
    # Never print passwords in startup logs.
    out = []
    for part in dsn.split():
        if part.startswith("password="):
            out.append("password=****")
        else:
            out.append(part)
    return " ".join(out)


if __name__ == "__main__":
    sys.exit(main())
