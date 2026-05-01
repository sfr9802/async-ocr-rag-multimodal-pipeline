"""Phase 7.4 — orchestrator CLI for the controlled recovery loop.

Reads Phase 7.3's per-query confidence JSONL and (optionally) Phase
7.0's per_query_comparison.jsonl + chunks_jsonl to drive a recovery
loop over the queries flagged by Phase 7.3 with
``recommended_action ∈ {HYBRID_RECOVERY, QUERY_REWRITE}``. The loop
writes the artefact bundle Phase 7.4 asks for; production code is NOT
touched.

Usage::

    python -m scripts.run_phase7_4_controlled_recovery \\
        --confidence-jsonl eval/reports/phase7/7.3_confidence_eval/per_query_confidence.jsonl \\
        --per-query        eval/reports/phase7/7.0_retrieval_title_ab/per_query_comparison.jsonl \\
        --chunks           eval/reports/phase7/7.0_retrieval_title_ab/rag_chunks_retrieval_title_section.jsonl \\
        --silver-queries   eval/reports/phase7/7.0_retrieval_title_ab/queries_v4_silver.jsonl \\
        --report-dir       eval/reports/phase7/7.4_controlled_recovery/ \\
        --rewrite-mode     both

The harness is deterministic: all retrieval is post-hoc against the
existing Phase 7.0 dense top-N + a fresh BM25 index over the chunks
JSONL. No LLM rewriter is invoked.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from eval.harness.bm25_retriever import (
    BM25EvalRetriever,
    DEFAULT_TOP_K as BM25_DEFAULT_TOP_K,
    build_bm25_index,
)
from eval.harness.controlled_recovery_loop import (
    ControlledRecoveryConfig,
    ControlledRecoveryResult,
    load_chunks_for_bm25,
    load_frozen_dense_state,
    load_verdict_rows,
    run_controlled_recovery,
)
from eval.harness.embedding_text_builder import (
    EMBEDDING_TEXT_VARIANTS,
    VARIANT_TITLE_SECTION,
    VARIANT_RAW,
)
from eval.harness.recovery_metrics import (
    aggregate_results,
    write_outputs,
)
from eval.harness.recovery_policy import (
    REWRITE_MODE_BOTH,
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
    REWRITE_MODES,
)


log = logging.getLogger("scripts.run_phase7_4_controlled_recovery")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 7.4 controlled recovery loop. Reads Phase 7.3 verdicts "
            "and runs deterministic recovery (hybrid / query rewrite) "
            "against a fresh BM25 index + frozen Phase 7.0 dense top-N. "
            "Writes the recovery artefact bundle. Production code untouched."
        ),
    )
    p.add_argument(
        "--confidence-jsonl", type=Path, required=True,
        help=(
            "Phase 7.3 per_query_confidence.jsonl (or recommended_recovery_"
            "queries.jsonl — both shapes are accepted)."
        ),
    )
    p.add_argument(
        "--per-query", type=Path, default=None,
        help=(
            "Phase 7.0 per_query_comparison.jsonl. Provides the frozen "
            "dense top-N for the RRF fuse step. When omitted, the loop "
            "runs BM25-only recoveries (the dense list is empty)."
        ),
    )
    p.add_argument(
        "--chunks", type=Path, required=True,
        help=(
            "rag_chunks_*.jsonl. The loop builds a fresh BM25 index over "
            "this corpus for the recovery pass."
        ),
    )
    p.add_argument(
        "--silver-queries", type=Path, default=None,
        help=(
            "Optional queries_v4_silver.jsonl. Currently unused by the "
            "recovery executor (the verdict rows already carry the silver "
            "expected_title / gold_doc_id) but accepted for symmetry "
            "with the Phase 7.3 CLI signature."
        ),
    )
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output directory for the Phase 7.4 artefact bundle.",
    )
    p.add_argument(
        "--side", choices=("baseline", "candidate"), default="candidate",
        help=(
            "Which side of Phase 7.0's A/B to read frozen dense from. "
            "Default 'candidate' matches Phase 7.3's default."
        ),
    )
    p.add_argument(
        "--rewrite-mode", choices=REWRITE_MODES,
        default=REWRITE_MODE_PRODUCTION_LIKE,
        help=(
            "QUERY_REWRITE mode. 'oracle' uses expected_title (upper-bound). "
            "'production_like' uses only top-N candidate canonical titles "
            "(no silver leakage). 'both' fans out QUERY_REWRITE rows into "
            "two attempts, oracle and production-like."
        ),
    )
    p.add_argument(
        "--final-k", type=int, default=10,
        help=(
            "Top-K window used for recovered@k / regression. Mirrors "
            "Phase 7.0's 10."
        ),
    )
    p.add_argument(
        "--hybrid-top-k", type=int, default=10,
        help=(
            "Final-k passed to the RRF fuser. Defaults to 10 to match "
            "Phase 7.0's reporting window; higher values surface gold "
            "deeper but loosen the comparison."
        ),
    )
    p.add_argument(
        "--bm25-pool-size", type=int, default=100,
        help=(
            "BM25 retrieval pool size before fusion. Larger pool → "
            "more lexical recall going into the RRF, at the cost of "
            "deeper RRF lists (still capped by hybrid_top_k afterwards)."
        ),
    )
    p.add_argument(
        "--k-rrf", type=int, default=60,
        help="RRF k constant. Default 60 matches the Phase 2 hybrid retriever.",
    )
    p.add_argument(
        "--top-n-for-production", type=int, default=5,
        help=(
            "How many top-N candidate previews the production-like rewrite "
            "is allowed to read titles from. Default 5 matches Phase 7.3's "
            "preview cap."
        ),
    )
    p.add_argument(
        "--no-strict-label-leakage", action="store_true",
        help=(
            "Disable the LabelLeakageError raise when production_like "
            "is asked for a row that carries expected_title. Use only for "
            "diagnostic comparisons; the produced rows are flagged with "
            "strict_label_leakage=False so a reader can spot the leniency."
        ),
    )
    p.add_argument(
        "--bm25-embedding-text-variant", default=VARIANT_TITLE_SECTION,
        choices=tuple(EMBEDDING_TEXT_VARIANTS),
        help=(
            "Embedding-text variant the BM25 index tokenises. "
            "title_section is the Phase 7.0 promoted default; raw "
            "approximates the chunk_text the dense embedder would have "
            "seen pre-prefix."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ControlledRecoveryConfig(
        rewrite_mode=args.rewrite_mode,
        final_k=int(args.final_k),
        hybrid_top_k=int(args.hybrid_top_k),
        bm25_pool_size=int(args.bm25_pool_size),
        k_rrf=int(args.k_rrf),
        top_n_for_production=int(args.top_n_for_production),
        strict_label_leakage=not args.no_strict_label_leakage,
        side=args.side,
    ).validate()
    log.info("ControlledRecoveryConfig: %s", asdict(cfg))

    log.info("loading verdict rows: %s", args.confidence_jsonl)
    verdict_rows = load_verdict_rows(args.confidence_jsonl)
    log.info("loaded %d verdict rows", len(verdict_rows))

    if args.per_query is not None:
        log.info("loading frozen dense state: %s", args.per_query)
        frozen = load_frozen_dense_state(
            args.per_query, side=args.side, final_k=cfg.final_k,
        )
        log.info("loaded frozen dense for %d qids", len(frozen))
    else:
        frozen = {}
        log.info(
            "no --per-query provided; running BM25-only recoveries "
            "(dense list empty)."
        )

    log.info(
        "loading chunks for BM25: %s (variant=%s)",
        args.chunks, args.bm25_embedding_text_variant,
    )
    chunks = load_chunks_for_bm25(args.chunks)
    log.info("loaded %d chunks; building BM25 index ...", len(chunks))
    bm25_index = build_bm25_index(
        chunks,
        embedding_text_variant=args.bm25_embedding_text_variant,
    )
    bm25_retriever = BM25EvalRetriever(
        bm25_index,
        top_k=max(cfg.bm25_pool_size, BM25_DEFAULT_TOP_K),
        name="bm25-recovery",
    )

    log.info("running controlled recovery loop ...")
    result = run_controlled_recovery(
        verdict_rows=verdict_rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25_retriever,
        config=cfg,
    )

    aggregate = aggregate_results(result)
    out_paths = write_outputs(
        result, out_dir=args.report_dir, aggregate=aggregate,
    )
    log.info("Phase 7.4 controlled recovery finished. Outputs:")
    for role, p in out_paths.items():
        log.info("  %s -> %s", role, p)

    totals = aggregate.get("totals") or {}
    invariants = aggregate.get("invariants") or {}
    log.info(
        "n_decisions=%d attempted=%d recovered=%d regressed=%d "
        "newly_entered=%d insufficient_refused=%d caution_skipped=%d",
        aggregate.get("n_queries", 0),
        totals.get("attempted", 0),
        totals.get("recovered", 0),
        totals.get("regressed", 0),
        totals.get("gold_newly_entered_candidates", 0),
        invariants.get("insufficient_evidence_refused_count", 0),
        invariants.get("answer_with_caution_skip_count", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
