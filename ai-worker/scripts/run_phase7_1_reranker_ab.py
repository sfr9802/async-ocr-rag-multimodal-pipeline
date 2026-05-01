"""Phase 7.1 — orchestrator CLI for the reranker A/B over Phase 7.0
``retrieval_title_section`` indexes.

The Phase 7.0 verdict was that the dense-only ``retrieval_title_section``
candidate beat the ``title_section`` baseline by ~22pt on hit@1 and ~21pt
on MRR. Phase 7.1 asks: when we now layer the cross-encoder reranker on
top of those same dense candidates, does the +22pt margin survive,
shrink, or grow? And — equally important — how many of the 200 silver
queries does the reranker *regress* relative to the strong dense
baseline?

This driver:

  1. Loads the existing ``retrieval_title_section`` FAISS index from
     ``eval/indexes/namu-v4-2008-2026-04-retrieval-title-section-mseq512``
     (no rebuild).
  2. Configures TWO retrievers backed by that single index:
       - ``baseline``: ``candidate_k = final_k`` (e.g. 10), NoOpReranker.
         This is the Phase 7.0 candidate side reused as Phase 7.1's
         dense-only baseline.
       - ``candidate``: ``candidate_k = config.candidate_k`` (default 40),
         NoOpReranker. The Phase 7.1 module then applies the cross-encoder
         outside the retriever so it can record dense rank pre-rerank.
  3. Runs the paired A/B over ``queries_v4_silver.jsonl`` (the Phase 7.0
     200-query set, picked up from the Phase 7.0 report dir by default).
  4. Writes the artefact bundle to a new Phase 7.1 report dir.

The reranker model defaults to ``BAAI/bge-reranker-v2-m3`` (the
production default and the model already proven by Phase 2A's reranker
sweeps on the v3 corpus). Score-mode flags expose
``reranker_only`` (default) and ``weighted_dense_rerank``.

Usage::

    python -m scripts.run_phase7_1_reranker_ab \\
        --variant-cache eval/indexes/namu-v4-2008-2026-04-retrieval-title-section-mseq512 \\
        --queries eval/reports/phase7/7.0_retrieval_title_ab/queries_v4_silver.jsonl \\
        --report-dir eval/reports/phase7/7.1_reranker_ab/ \\
        --candidate-k 40 --final-k 10 \\
        --score-mode reranker_only

A second pass with ``--score-mode weighted_dense_rerank --dense-weight 0.3
--rerank-weight 0.7`` produces the blended-score variant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from app.capabilities.rag.reranker import CrossEncoderReranker, NoOpReranker
from eval.harness.embedding_text_reindex import load_variant_dense_stack
from eval.harness.v4_ab_eval import load_queries
from eval.harness.v4_rerank_ab import (
    RerankerAbConfig,
    SCORE_MODE_RERANKER_ONLY,
    SCORE_MODE_WEIGHTED,
    SCORE_MODES,
    run_reranker_ab,
    write_ab_outputs,
)


log = logging.getLogger("scripts.run_phase7_1_reranker_ab")


_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_RERANKER_MAX_LENGTH = 512


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # Description deliberately ASCII-only so Windows cp949 consoles can
    # render --help without UnicodeEncodeError. Module-level docstring
    # carries the prose detail.
    p = argparse.ArgumentParser(
        description=(
            "Phase 7.1 reranker A/B over the Phase 7.0 "
            "retrieval_title_section dense index."
        ),
    )
    p.add_argument(
        "--variant-cache", type=Path, required=True,
        help=(
            "Path to the Phase 7.0 dense index cache "
            "(retrieval_title_section variant). Must contain faiss.index, "
            "build.json, chunks.jsonl. NOT rebuilt."
        ),
    )
    p.add_argument(
        "--queries", type=Path, required=True,
        help=(
            "Path to queries_v4_silver.jsonl (the Phase 7.0 200-query "
            "silver set)."
        ),
    )
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output dir for Phase 7.1 artefacts.",
    )
    p.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="Bi-encoder model to load for query embedding.",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=512,
        help=(
            "Bi-encoder max_seq_length. Should match the value the index "
            "was built with - 512 for the Phase 7.0 v4 caches."
        ),
    )
    p.add_argument(
        "--candidate-k", type=int, default=40,
        choices=[20, 40, 80],
        help=(
            "Number of dense candidates handed to the reranker. "
            "Spec exposes 20/40/80; default 40."
        ),
    )
    p.add_argument(
        "--final-k", type=int, default=10,
        help="Final top-k after reranking.",
    )
    p.add_argument(
        "--score-mode", choices=SCORE_MODES, default=SCORE_MODE_RERANKER_ONLY,
        help=(
            "How to order the reranker output. "
            "reranker_only: order by rerank_score alone. "
            "weighted_dense_rerank: dense_weight*z(dense) + "
            "rerank_weight*z(rerank), per-query min-max normalised."
        ),
    )
    p.add_argument(
        "--dense-weight", type=float, default=0.3,
        help="Dense-side weight in weighted_dense_rerank mode.",
    )
    p.add_argument(
        "--rerank-weight", type=float, default=0.7,
        help="Rerank-side weight in weighted_dense_rerank mode.",
    )
    p.add_argument(
        "--reranker-model", default=_DEFAULT_RERANKER_MODEL,
        help="CrossEncoder model name.",
    )
    p.add_argument(
        "--reranker-max-length", type=int,
        default=_DEFAULT_RERANKER_MAX_LENGTH,
        help="CrossEncoder max_length.",
    )
    p.add_argument(
        "--reranker-batch-size", type=int, default=16,
        help="CrossEncoder.predict batch size.",
    )
    p.add_argument(
        "--reranker-text-max-chars", type=int, default=800,
        help="Per-passage truncation char cap before reranker tokenisation.",
    )
    p.add_argument(
        "--reranker-device", default=None,
        help="Force a device override (e.g. cuda / cpu); auto-detect when omitted.",
    )
    p.add_argument(
        "--use-noop-reranker", action="store_true",
        help=(
            "Replace the cross-encoder with a NoOpReranker. "
            "Useful for self-A/B sanity checks that prove the harness "
            "produces zero status changes when both sides are identical."
        ),
    )
    p.add_argument(
        "--baseline-label", default="retrieval_title_section_dense",
        help="Label written into the summary for the baseline side.",
    )
    p.add_argument(
        "--candidate-label",
        default="retrieval_title_section_dense_plus_rerank",
        help="Label written into the summary for the candidate side.",
    )
    p.add_argument(
        "--progress-log-every", type=int, default=25,
        help="Log every N queries (0 disables progress logging).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _build_reranker(args: argparse.Namespace):
    """Return a RerankerProvider. NoOp when --use-noop-reranker is set."""
    if args.use_noop_reranker:
        log.info("Reranker: NoOp (self-A/B mode)")
        return NoOpReranker()
    log.info(
        "Reranker: CrossEncoder model=%s max_length=%d batch=%d",
        args.reranker_model, args.reranker_max_length, args.reranker_batch_size,
    )
    return CrossEncoderReranker(
        model_name=args.reranker_model,
        max_length=args.reranker_max_length,
        batch_size=args.reranker_batch_size,
        text_max_chars=args.reranker_text_max_chars,
        device=args.reranker_device,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = RerankerAbConfig(
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        reranker_batch_size=args.reranker_batch_size,
        score_mode=args.score_mode,
        dense_weight=args.dense_weight,
        rerank_weight=args.rerank_weight,
    ).validate()
    log.info("Phase 7.1 config: %s", asdict(config))

    queries = load_queries(args.queries)
    log.info("Loaded %d queries from %s", len(queries), args.queries)

    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        max_seq_length=args.max_seq_length,
    )

    # Two retrievers, both backed by the SAME Phase 7.0 cache. The split
    # is purely so we can record the dense pool order pre-rerank.
    baseline_retriever, _, _ = load_variant_dense_stack(
        args.variant_cache,
        embedder=embedder,
        top_k=config.final_k,
        candidate_k=config.final_k,
    )
    candidate_retriever, _, _ = load_variant_dense_stack(
        args.variant_cache,
        embedder=embedder,
        top_k=config.candidate_k,
        candidate_k=config.candidate_k,
    )
    log.info(
        "Retrievers ready: cache=%s top_k(base)=%d top_k(cand)=%d",
        args.variant_cache, config.final_k, config.candidate_k,
    )

    reranker = _build_reranker(args)

    result = run_reranker_ab(
        queries,
        baseline_retriever=baseline_retriever,
        candidate_retriever=candidate_retriever,
        reranker=reranker,
        config=config,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        progress_log_every=args.progress_log_every,
    )

    out_paths = write_ab_outputs(result, out_dir=args.report_dir)
    log.info("Phase 7.1 A/B finished. Outputs:")
    for role, p in out_paths.items():
        log.info("  %s -> %s", role, p)

    summary = result.aggregate
    log.info(
        "Aggregate: hit@1 base=%.4f cand=%.4f Δ=%+.4f | "
        "mrr@10 base=%.4f cand=%.4f Δ=%+.4f | "
        "improved=%d regressed=%d",
        summary["baseline"].get("hit_at_1", 0.0),
        summary["candidate"].get("hit_at_1", 0.0),
        summary["candidate"].get("hit_at_1", 0.0)
            - summary["baseline"].get("hit_at_1", 0.0),
        summary["baseline"].get("mrr_at_10", 0.0),
        summary["candidate"].get("mrr_at_10", 0.0),
        summary["candidate"].get("mrr_at_10", 0.0)
            - summary["baseline"].get("mrr_at_10", 0.0),
        summary["status_counts"].get("improved", 0),
        summary["status_counts"].get("regressed", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
