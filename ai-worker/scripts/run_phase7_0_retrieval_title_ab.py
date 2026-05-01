"""Phase 7.0 — orchestrator CLI for the retrieval_title A/B.

Wraps the four steps the spec calls for:

  1. Export ``rag_chunks_<variant>.jsonl`` for both variants
     (``title_section`` + ``retrieval_title_section``) under the
     Phase 7.0 report directory, with per-variant manifests.
  2. Compute the variant diff report (json + md) using the page-level
     metadata in ``pages_v4.jsonl`` for breakdowns.
  3. Build (or load) the FAISS dense index for each variant under
     ``eval/indexes/namu-v4-2008-2026-04-<variant>``.
  4. Generate (or load) the v4 silver query set, drive the paired A/B
     across both retrievers, and emit ``ab_summary.json/md``,
     ``per_query_comparison.jsonl``, ``improved_queries.jsonl``,
     ``regressed_queries.jsonl`` under the same Phase 7.0 report dir.

Each step has a ``--skip-<step>`` flag so a partial rerun (e.g. only
the A/B over already-built indexes) is one command. Outputs are
deterministic: the same input artefacts + seed produce byte-identical
chunks files and a stable silver query set.

Usage::

    python -m scripts.run_phase7_0_retrieval_title_ab \\
        --rag-chunks PATH/rag_chunks.jsonl \\
        --pages-v4 PATH/pages_v4.jsonl \\
        --report-dir eval/reports/phase7/7.0_retrieval_title_ab \\
        --index-root eval/indexes \\
        --embedding-model BAAI/bge-m3 \\
        --top-k 10 --target-queries 200 --seed 42

Add ``--skip-index-build`` if the cache already has both indexes; the
A/B step will load via ``load_variant_dense_stack``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from eval.harness.embedding_text_builder import (
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_TITLE_SECTION,
)
from eval.harness.embedding_text_reindex import load_variant_dense_stack
from eval.harness.v4_ab_eval import (
    load_queries,
    run_paired_ab,
    write_ab_outputs,
)
from eval.harness.v4_chunk_export import (
    V4_EXPORT_VARIANTS,
    export_v4_chunks,
)
from eval.harness.v4_index_builder import (
    build_v4_variant_index,
    v4_default_cache_dir,
)
from eval.harness.v4_silver_queries import (
    generate_v4_silver_queries,
    write_v4_silver_queries,
)
from eval.harness.v4_variant_diff_report import (
    compute_variant_diff,
    write_variant_diff_report,
)


log = logging.getLogger("scripts.run_phase7_0_retrieval_title_ab")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7.0 retrieval_title A/B")
    p.add_argument(
        "--rag-chunks", type=Path, required=True,
        help="Phase 6.3 rag_chunks.jsonl input.",
    )
    p.add_argument(
        "--pages-v4", type=Path, required=True,
        help="Phase 6.3 pages_v4.jsonl input (for breakdowns + queries).",
    )
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output dir for phase 7.0 artefacts.",
    )
    p.add_argument(
        "--index-root", type=Path, required=True,
        help="Cache root under which per-variant indexes live.",
    )
    p.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="Sentence-transformers model for the dense index.",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="Embedder max_seq_length (used in slug + cache key).",
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Top-k for the A/B retrieval.",
    )
    p.add_argument(
        "--target-queries", type=int, default=200,
        help="Number of v4 silver queries to render.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for query generation.",
    )
    p.add_argument(
        "--queries", type=Path, default=None,
        help=(
            "Optional pre-rendered query JSONL. When set, query "
            "generation is skipped and this file is used as-is."
        ),
    )
    p.add_argument(
        "--embed-batch-size", type=int, default=None,
        help="Embedder batch size; None lets the provider choose.",
    )
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-diff", action="store_true")
    p.add_argument("--skip-index-build", action="store_true")
    p.add_argument("--skip-ab", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _step_export(args: argparse.Namespace) -> Dict[str, Path]:
    """Run step 1 — emit per-variant chunk files + manifests."""
    out_dir = args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for variant in V4_EXPORT_VARIANTS:
        out_path = out_dir / f"rag_chunks_{variant}.jsonl"
        log.info("Exporting variant=%s → %s", variant, out_path)
        summary = export_v4_chunks(
            args.rag_chunks, out_path, variant=variant,
        )
        log.info(
            "  total=%d changed=%d page_ids=%d sha256=%s",
            summary.total_chunks, summary.changed_embedding_text_count,
            summary.page_id_count, summary.embed_text_sha256[:16],
        )
        paths[variant] = out_path
    return paths


def _step_diff(
    args: argparse.Namespace, paths: Dict[str, Path],
) -> Path:
    """Run step 2 — diff report between the two exports."""
    report = compute_variant_diff(
        baseline_chunks_path=paths[VARIANT_TITLE_SECTION],
        candidate_chunks_path=paths[VARIANT_RETRIEVAL_TITLE_SECTION],
        pages_v4_path=args.pages_v4,
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    json_path, md_path = write_variant_diff_report(
        report, out_dir=args.report_dir,
    )
    log.info(
        "variant diff: changed=%d/%d (%.2f%%) → %s",
        report["changed_embedding_text_count"],
        report["total_chunks"],
        report["changed_embedding_text_ratio"] * 100,
        json_path,
    )
    return json_path


def _step_index(
    args: argparse.Namespace,
    paths: Dict[str, Path],
) -> Dict[str, Path]:
    """Run step 3 — build (or load) the per-variant dense index."""
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        max_seq_length=args.max_seq_length,
    )
    cache_dirs: Dict[str, Path] = {}
    for variant in V4_EXPORT_VARIANTS:
        cache_dir = v4_default_cache_dir(
            cache_root=args.index_root,
            embedding_model=args.embedding_model,
            max_seq_length=args.max_seq_length,
            variant=variant,
        )
        log.info("Building index for variant=%s → %s", variant, cache_dir)
        retriever, summary = build_v4_variant_index(
            paths[variant],
            embedder=embedder,
            cache_dir=cache_dir,
            variant=variant,
            top_k=args.top_k,
            embed_batch_size=args.embed_batch_size,
        )
        log.info(
            "  built index: chunks=%d docs=%d dim=%d sha256=%s",
            summary.chunk_count, summary.document_count,
            summary.dimension, summary.embed_text_sha256[:16],
        )
        cache_dirs[variant] = cache_dir
        del retriever  # free the wrapper; we'll reload below
    return cache_dirs


def _step_ab(
    args: argparse.Namespace,
    cache_dirs: Dict[str, Path],
) -> Dict[str, Path]:
    """Run step 4 — load both indexes, drive the paired A/B."""
    if args.queries is not None:
        query_path = args.queries
        log.info("Using pre-rendered queries: %s", query_path)
    else:
        query_path = args.report_dir / "queries_v4_silver.jsonl"
        log.info(
            "Generating v4 silver queries (target=%d, seed=%d) → %s",
            args.target_queries, args.seed, query_path,
        )
        rendered = generate_v4_silver_queries(
            args.pages_v4,
            target_total=args.target_queries,
            seed=args.seed,
        )
        write_v4_silver_queries(rendered, query_path)
        log.info("  wrote %d queries", len(rendered))

    queries = load_queries(query_path)
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        max_seq_length=args.max_seq_length,
    )

    base_retriever, _, _ = load_variant_dense_stack(
        cache_dirs[VARIANT_TITLE_SECTION],
        embedder=embedder,
        top_k=args.top_k,
    )
    cand_retriever, _, _ = load_variant_dense_stack(
        cache_dirs[VARIANT_RETRIEVAL_TITLE_SECTION],
        embedder=embedder,
        top_k=args.top_k,
    )

    result = run_paired_ab(
        queries,
        baseline_retriever=base_retriever,
        candidate_retriever=cand_retriever,
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    out_paths = write_ab_outputs(
        result,
        out_dir=args.report_dir,
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    log.info("A/B summary: %s", out_paths["summary_md"])
    return out_paths


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths: Dict[str, Path] = {}
    if not args.skip_export:
        paths = _step_export(args)
    else:
        for variant in V4_EXPORT_VARIANTS:
            paths[variant] = args.report_dir / f"rag_chunks_{variant}.jsonl"
            if not paths[variant].exists():
                raise SystemExit(
                    f"--skip-export was passed but {paths[variant]} is missing."
                )

    if not args.skip_diff:
        _step_diff(args, paths)

    cache_dirs: Dict[str, Path] = {}
    for variant in V4_EXPORT_VARIANTS:
        cache_dirs[variant] = v4_default_cache_dir(
            cache_root=args.index_root,
            embedding_model=args.embedding_model,
            max_seq_length=args.max_seq_length,
            variant=variant,
        )

    if not args.skip_index_build:
        cache_dirs = _step_index(args, paths)

    if not args.skip_ab:
        _step_ab(args, cache_dirs)

    log.info("Phase 7.0 A/B finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
