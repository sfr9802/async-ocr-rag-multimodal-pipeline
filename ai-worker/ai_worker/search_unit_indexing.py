"""CLI entrypoint for SearchUnit indexing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from app.capabilities.rag.embeddings import (
    SentenceTransformerEmbedder,
    resolve_max_seq_length,
)
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.metadata_store import RagMetadataStore
from app.capabilities.rag.search_unit_indexing import SearchUnitVectorIndexer
from app.clients.core_api_client import CoreApiClient
from app.core.config import WorkerSettings
from app.core.logging import configure_logging
from app.services.search_unit_indexing_loop import SearchUnitIndexingWorker

log = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if not args.once and not args.loop:
        args.once = True

    settings = WorkerSettings()
    max_seq_length = resolve_max_seq_length(settings.rag_embedding_max_seq_length)
    core_api = CoreApiClient(
        settings.core_api_base_url,
        settings.core_api_request_timeout_seconds,
        settings.internal_secret,
    )
    try:
        embedder = SentenceTransformerEmbedder(
            model_name=settings.rag_embedding_model,
            query_prefix=settings.rag_embedding_prefix_query,
            passage_prefix=settings.rag_embedding_prefix_passage,
            max_seq_length=max_seq_length,
            batch_size=int(settings.rag_embedding_batch_size),
            cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
        )
        metadata = RagMetadataStore(settings.rag_db_dsn)
        index = FaissIndex(Path(settings.rag_index_dir))
        indexer = SearchUnitVectorIndexer(
            embedder=embedder,
            metadata_store=metadata,
            index=index,
            index_version=args.index_version,
            embedding_text_variant=settings.rag_embedding_text_variant,
            max_seq_length=max_seq_length,
        )
        worker = SearchUnitIndexingWorker(
            core_api=core_api,
            indexer=indexer,
            worker_id=settings.worker_id,
            batch_size=args.batch_size,
            stale_after_seconds=args.stale_after_seconds,
        )

        if args.loop:
            worker.run_loop(
                interval_seconds=args.interval_seconds,
                dry_run=args.dry_run,
            )
        else:
            summary = worker.run_once(dry_run=args.dry_run)
            log.info("SearchUnit indexing once summary: %s", summary)
        return 0
    except KeyboardInterrupt:
        log.info("SearchUnit indexing loop interrupted")
        return 0
    finally:
        core_api.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claim pending SearchUnits from core-api and index them into ragmeta/FAISS.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one claim/index/complete cycle.")
    mode.add_argument("--loop", action="store_true", help="Run continuously until interrupted.")
    parser.add_argument("--batch-size", type=int, default=50, help="SearchUnit claim batch size.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=10.0,
        help="Sleep interval between loop iterations.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=None,
        help="Ask Spring to reclaim stale EMBEDDING claims older than this many seconds.",
    )
    parser.add_argument(
        "--index-version",
        default=None,
        help="Optional index_version. Omit to use the currently loaded FAISS build version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI/config wiring without claiming SearchUnits or writing indexes.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
