"""Unified CLI entry point for the eval harness.

Usage (from ai-worker/):

    # Text RAG eval against the live index + model configured for the worker
    python -m eval.run_eval rag \
        --dataset eval/datasets/rag_sample.jsonl \
        --out-json eval/reports/rag-latest.json \
        --out-csv  eval/reports/rag-latest.csv \
        --top-k 5

    # OCR eval using the Tesseract provider configured for the worker
    python -m eval.run_eval ocr \
        --dataset eval/datasets/ocr_sample.jsonl \
        --out-json eval/reports/ocr-latest.json \
        --out-csv  eval/reports/ocr-latest.csv

Both subcommands print a short human-readable summary to stdout and
exit 0 on success. Any provider/retriever failure at startup (missing
FAISS index, missing tesseract binary, etc.) exits 2 with a clear
error — the same failure mode a production worker would surface at
boot.

The CLI deliberately builds the real production stack: bge-m3 + the
committed FAISS index for RAG, tesseract + pymupdf for OCR. Unit tests
live in `tests/test_eval_harness.py` and use pluggable fake
providers so they run offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from eval.harness.io_utils import (
    load_jsonl,
    write_csv_report,
    write_json_report,
)
from eval.harness.multimodal_eval import (
    MultimodalEvalRow,
    MultimodalEvalSummary,
    row_to_dict as mm_row_to_dict,
    run_multimodal_eval,
    summary_to_dict as mm_summary_to_dict,
)
from eval.harness.ocr_eval import (
    OcrEvalRow,
    OcrEvalSummary,
    row_to_dict as ocr_row_to_dict,
    run_ocr_eval,
    summary_to_dict as ocr_summary_to_dict,
)
from eval.harness.rag_eval import (
    RagEvalRow,
    RagEvalSummary,
    row_to_dict as rag_row_to_dict,
    run_rag_eval,
    summary_to_dict as rag_summary_to_dict,
)
from eval.harness.retrieval_eval import (
    DuplicateAnalysis,
    RetrievalEvalRow,
    RetrievalEvalSummary,
    TopKDumpRow,
    duplicate_analysis_to_dict,
    dump_row_to_dict as retrieval_dump_row_to_dict,
    render_markdown_report,
    row_to_dict as retrieval_row_to_dict,
    run_retrieval_eval,
    summary_to_dict as retrieval_summary_to_dict,
)
from eval.harness.miss_analysis import (
    classify_rows as classify_miss_buckets,
    miss_analysis_to_dict,
    render_miss_analysis_markdown,
)
from eval.harness.baseline_comparison import (
    comparison_to_dict,
    render_comparison_markdown,
    run_comparison,
)
from eval.harness.analyze_corpus_lengths import (
    DEFAULT_THRESHOLDS as DEFAULT_TOKEN_THRESHOLDS,
    DEFAULT_TOP_LONGEST as DEFAULT_TOP_LONGEST_CHUNKS,
    DEFAULT_TOKENIZER_NAME,
    analyze_corpus_lengths,
    length_analysis_to_dict,
    render_length_analysis_markdown,
)
from eval.harness.corpus_audit import (
    DEFAULT_AUDIT_TOP_N,
    audit_long_chunks,
    audit_to_dict,
    compare_raw_vs_cleaned,
    length_comparison_to_dict,
    render_audit_markdown,
    render_length_comparison_markdown,
)
from eval.harness.corpus_preprocessor import (
    PREPROCESS_VERSION,
    CorpusPreprocessSummary,
    PreprocessConfig,
    corpus_preprocess_summary_to_dict,
    iter_preprocessed_documents,
    render_corpus_preprocess_summary_markdown,
    render_sample_diff_markdown,
)
from eval.harness.chunker_diagnostics import (
    DEFAULT_THRESHOLDS as DEFAULT_DIAGNOSE_THRESHOLDS,
    DEFAULT_TOP_N as DEFAULT_DIAGNOSE_TOP_N,
    chunker_diagnosis_to_dict,
    diagnose_chunker_long_tail,
    render_chunker_diagnosis_markdown,
    render_chunker_provenance_markdown,
    samples_to_dict_list as diagnose_samples_to_dict_list,
)
from eval.harness.token_aware_emit import (
    EmitConfig,
    build_default_tokenizer_callables,
    emit_summary_to_dict,
    emit_token_aware_corpus,
    render_emit_summary_markdown,
)
from app.capabilities.rag.token_aware_chunker import (
    CHUNKER_VERSION as TOKEN_AWARE_CHUNKER_VERSION,
    DEFAULT_HARD_MAX_TOKENS as DEFAULT_TA_HARD_MAX,
    DEFAULT_OVERLAP_TOKENS as DEFAULT_TA_OVERLAP,
    DEFAULT_SOFT_MAX_TOKENS as DEFAULT_TA_SOFT_MAX,
    DEFAULT_TARGET_TOKENS as DEFAULT_TA_TARGET,
    TokenAwareConfig,
)
from app.capabilities.rag.ingest import _iter_documents
from eval.harness.boost_eval import (
    run_boost_retrieval_eval,
    write_boost_artifacts,
)
from eval.harness.boost_failure_analysis import (
    boost_failure_analysis_to_dict,
    classify_boost_failures,
    render_boost_failure_markdown,
)
from eval.harness.boost_metadata import load_doc_metadata
from eval.harness.boost_pareto import (
    boost_pareto_to_dict,
    compute_boost_pareto_frontier,
    render_boost_pareto_markdown,
)
from eval.harness.boost_scorer import BoostConfig, MetadataBoostReranker
from eval.harness.boosting_retriever import BoostingEvalRetriever
from eval.harness.candidate_miss_analysis import (
    DEFAULT_DEEP_K as DEFAULT_MISS_DEEP_K,
    DEFAULT_TOP_KS as DEFAULT_MISS_TOP_KS,
    candidate_miss_report_to_dict,
    classify_candidate_misses,
    render_candidate_miss_markdown,
)
from eval.harness.topn_sweep import build_topn_sweep


log = logging.getLogger("eval")

_PHASE7_V4_HELP_NOTE = (
    "Phase 7 active eval/tuning uses the fail-closed "
    "eval/experiments/active.yaml guardrail and the v4 corpus under "
    "eval/corpora/namu-v4-structured-combined/. Any anime_namu_v3 or "
    "Phase 1/2 corpus examples in this CLI are LEGACY V3 ONLY for "
    "historical reproduction."
)


# ---------------------------------------------------------------------------
# Argparse wiring.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose)

    if args.mode == "rag":
        return _run_rag_cli(args)
    if args.mode == "ocr":
        return _run_ocr_cli(args)
    if args.mode == "multimodal":
        return _run_multimodal_cli(args)
    if args.mode == "retrieval":
        return _run_retrieval_cli(args)
    if args.mode == "retrieval-rerank":
        return _run_retrieval_rerank_cli(args)
    if args.mode == "phase2a-reranker-comparison":
        return _run_phase2a_reranker_comparison_cli(args)
    if args.mode == "phase2a-reranker-failure-analysis":
        return _run_phase2a_reranker_failure_cli(args)
    if args.mode == "phase2a-latency-sweep":
        return _run_phase2a_latency_sweep_cli(args)
    if args.mode == "phase2a-latency-breakdown":
        return _run_phase2a_latency_breakdown_cli(args)
    if args.mode == "phase2a-topn-sweep":
        return _run_phase2a_topn_sweep_cli(args)
    if args.mode == "phase2a-recommended-modes":
        return _run_phase2a_recommended_modes_cli(args)
    if args.mode == "retrieval-compare":
        return _run_retrieval_compare_cli(args)
    if args.mode == "retrieval-miss-analysis":
        return _run_miss_analysis_cli(args)
    if args.mode == "analyze-corpus-lengths":
        return _run_analyze_corpus_lengths_cli(args)
    if args.mode == "audit-corpus-noise":
        return _run_audit_corpus_noise_cli(args)
    if args.mode == "clean-corpus-dry-run":
        return _run_clean_corpus_dry_run_cli(args)
    if args.mode == "preprocess-corpus-dry-run":
        return _run_preprocess_corpus_dry_run_cli(args)
    if args.mode == "emit-preprocessed-corpus":
        return _run_emit_preprocessed_corpus_cli(args)
    if args.mode == "compare-corpus-lengths":
        return _run_compare_corpus_lengths_cli(args)
    if args.mode == "diagnose-chunker-long-tail":
        return _run_diagnose_chunker_long_tail_cli(args)
    if args.mode == "emit-token-aware-chunks":
        return _run_emit_token_aware_chunks_cli(args)
    if args.mode == "compare-chunker-lengths":
        return _run_compare_chunker_lengths_cli(args)
    if args.mode == "retrieval-candidate-boost":
        return _run_retrieval_candidate_boost_cli(args)
    if args.mode == "retrieval-candidate-miss-analysis":
        return _run_retrieval_candidate_miss_analysis_cli(args)
    if args.mode == "retrieval-boost-failure-analysis":
        return _run_retrieval_boost_failure_analysis_cli(args)
    if args.mode == "retrieval-boost-pareto":
        return _run_retrieval_boost_pareto_cli(args)
    parser.error(f"unknown mode: {args.mode}")
    return 2  # unreachable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Run an eval harness over a JSONL dataset.",
        epilog=_PHASE7_V4_HELP_NOTE,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="More log output (DEBUG level).",
    )
    subs = parser.add_subparsers(dest="mode", required=True)

    # --- rag ---
    rag = subs.add_parser("rag", help="Run the text RAG eval harness.")
    rag.add_argument("--dataset", required=True, type=Path,
                     help="Path to the RAG eval JSONL dataset.")
    rag.add_argument("--out-json", type=Path, default=None,
                     help="Path for the JSON report. Defaults to "
                          "eval/reports/rag-<timestamp>.json.")
    rag.add_argument("--out-csv", type=Path, default=None,
                     help="Path for the CSV report. Defaults to "
                          "eval/reports/rag-<timestamp>.csv.")
    rag.add_argument("--top-k", type=int, default=None,
                     help="Override the retriever's top_k for this run. "
                          "Defaults to the worker setting.")
    rag.add_argument("--offline-corpus", type=Path, default=None,
                     help="Skip the live ragmeta/FAISS stack and build an "
                          "in-memory retriever from this JSONL corpus. Uses "
                          "the configured embedding model but no Postgres.")
    rag.add_argument("--agent-mode", type=str, default=None,
                     choices=["compare"],
                     help="If set to 'compare', run each row twice (agent loop "
                          "off vs on) and emit the Phase 8 decision-gate "
                          "report alongside a per-row compare CSV. Requires "
                          "the dataset to carry a 'difficulty' field for "
                          "per-difficulty aggregation.")
    rag.add_argument("--cross-domain", action="store_true",
                     help="Score the dataset under Phase 9 cross-domain mode: "
                          "every row carries 'filters' that point to the "
                          "WRONG domain, and the gate is whether the "
                          "generator refuses rather than hallucinates from "
                          "the filtered-in corpus.")
    rag.add_argument("--no-csv", action="store_true",
                     help="Skip CSV output.")

    # --- ocr ---
    ocr = subs.add_parser("ocr", help="Run the OCR eval harness.")
    ocr.add_argument("--dataset", required=True, type=Path,
                     help="Path to the OCR eval JSONL dataset.")
    ocr.add_argument("--out-json", type=Path, default=None,
                     help="Path for the JSON report. Defaults to "
                          "eval/reports/ocr-<timestamp>.json.")
    ocr.add_argument("--out-csv", type=Path, default=None,
                     help="Path for the CSV report. Defaults to "
                          "eval/reports/ocr-<timestamp>.csv.")
    ocr.add_argument("--fail-missing", action="store_true",
                     help="Treat missing fixture files as errors rather "
                          "than skips.")
    ocr.add_argument("--no-csv", action="store_true",
                     help="Skip CSV output.")

    # --- multimodal ---
    mm = subs.add_parser("multimodal", help="Run the multimodal eval harness.")
    mm.add_argument("--dataset", required=True, type=Path,
                    help="Path to the multimodal eval JSONL dataset.")
    mm.add_argument("--out-json", type=Path, default=None,
                    help="Path for the JSON report.")
    mm.add_argument("--out-csv", type=Path, default=None,
                    help="Path for the CSV report.")
    mm.add_argument("--no-csv", action="store_true",
                    help="Skip CSV output.")
    mm.add_argument("--require-ocr-only", action="store_true",
                    help="Only evaluate rows where requires_ocr is true.")
    mm.add_argument("--vision-provider", type=str, default=None,
                    choices=["heuristic", "claude"],
                    help="Override the vision provider for this run.")
    mm.add_argument("--cross-modal", action="store_true",
                    help="Enable CLIP cross-modal retrieval (text + image RRF).")

    # --- retrieval ---
    rt = subs.add_parser(
        "retrieval",
        help="Run the retrieval-quality eval harness (no generator).",
    )
    rt.add_argument("--dataset", required=True, type=Path,
                    help="Path to the eval-queries JSONL "
                         "(see ai-worker/eval/eval_queries/README.md).")
    rt.add_argument("--corpus", type=Path, default=None,
                    help="Offline corpus JSONL. LEGACY V3 ONLY example: "
                         "eval/corpora/anime_namu_v3/corpus.jsonl. "
                         "For Phase 7 active work, use v4-specific "
                         "harnesses/artifacts under "
                         "eval/corpora/namu-v4-structured-combined/. "
                         "When set, skips the live ragmeta/FAISS stack and "
                         "builds an in-memory retriever.")
    rt.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to drop the four output artifacts into. "
                         "Defaults to eval/reports/retrieval-<timestamp>/.")
    rt.add_argument("--top-k", type=int, default=10,
                    help="Top-k for retrieval scoring + dump (default: 10).")
    rt.add_argument("--mrr-k", type=int, default=10,
                    help="Cutoff k for MRR@k aggregation (default: 10).")
    rt.add_argument("--ndcg-k", type=int, default=10,
                    help="Cutoff k for NDCG@k aggregation (default: 10).")
    rt.add_argument("--max-seq-length", type=int, default=1024,
                    help="Cap the embedding model's max_seq_length when "
                         "building the offline corpus (default: 1024). "
                         "Lower → less GPU memory pressure on outlier-long "
                         "chunks at the cost of truncating their tails. "
                         "Only used with --corpus.")
    rt.add_argument("--embed-batch-size", type=int, default=32,
                    help="Embedding batch size for the offline corpus "
                         "build (default: 32). Halve this if you still "
                         "OOM on a small GPU.")
    rt.add_argument("--extra-hit-k", type=int, action="append", default=None,
                    help="Additional hit@k cutoff to compute on top of "
                         "the default {1,3,5}. LEGACY V3 ONLY: used for "
                         "the Phase 2A candidate-recall report — pass "
                         "--extra-hit-k 10 "
                         "--extra-hit-k 20 --extra-hit-k 50 to surface "
                         "hit@10 / hit@20 / hit@50 in the summary. The "
                         "harness will warn if any cutoff exceeds top-k. "
                         "Repeatable.")

    # --- retrieval-rerank ---
    #
    # LEGACY V3 ONLY: Phase 2A entrypoint. Same offline retrieval-eval flow as
    # ``retrieval``, but builds the in-memory Retriever with a
    # cross-encoder reranker stacked on top. dense-top-N candidates
    # flow into the reranker; the reranker returns final-top-K. All
    # rerank-specific knobs (model, batch_size, OOM-fallback) are
    # explicit on the CLI so a sweep is reproducible from a single
    # command line.
    rrk = subs.add_parser(
        "retrieval-rerank",
        description="LEGACY V3 ONLY / Phase 2A historical reproduction.",
        help="LEGACY V3 ONLY / Phase 2A historical reproduction: run "
             "retrieval eval with a cross-encoder reranker stacked on "
             "top of the dense retriever.",
    )
    rrk.add_argument("--dataset", required=True, type=Path,
                     help="Eval-queries JSONL.")
    rrk.add_argument("--corpus", required=True, type=Path,
                     help="Offline corpus JSONL. LEGACY V3 ONLY: Phase 2A "
                          "B2 token-aware-v1 corpus examples are historical "
                          "reproduction inputs, not Phase 7 defaults.")
    rrk.add_argument("--out-dir", type=Path, default=None,
                     help="Directory to drop the four output artifacts "
                          "into. Defaults to eval/reports/"
                          "retrieval-rerank-<timestamp>/.")
    rrk.add_argument("--final-top-k", type=int, default=10,
                     help="Final top-k after reranking (default: 10).")
    rrk.add_argument("--dense-top-n", type=int, default=20,
                     help="Number of bi-encoder candidates fetched from "
                          "FAISS before rerank (default: 20). MUST be "
                          ">= --final-top-k.")
    rrk.add_argument("--mrr-k", type=int, default=10)
    rrk.add_argument("--ndcg-k", type=int, default=10)
    rrk.add_argument("--max-seq-length", type=int, default=1024,
                     help="Cap the embedder's max_seq_length on the "
                          "offline corpus build (default: 1024).")
    rrk.add_argument("--embed-batch-size", type=int, default=32,
                     help="Embedding batch size for the offline corpus "
                          "build (default: 32).")
    rrk.add_argument("--reranker-model", type=str,
                     default="BAAI/bge-reranker-v2-m3",
                     help="Cross-encoder model name (default: "
                          "BAAI/bge-reranker-v2-m3 — multilingual M3 "
                          "reranker, the same family as the bge-m3 "
                          "embedder used for the bi-encoder).")
    rrk.add_argument("--reranker-batch-size", type=int, default=16,
                     help="CrossEncoder.predict batch_size (default: 16).")
    rrk.add_argument("--reranker-max-length", type=int, default=512,
                     help="Cross-encoder tokenizer max_length (default: "
                          "512). Truncates the (query, passage) pair "
                          "before encoding.")
    rrk.add_argument("--reranker-text-max-chars", type=int, default=800,
                     help="Per-passage character cap before the "
                          "cross-encoder tokenizer sees it. Keeps the "
                          "tokenizer from blowing past max_length on "
                          "very long chunks (default: 800).")
    rrk.add_argument("--reranker-device", type=str, default=None,
                     help="Force a CrossEncoder device ('cpu' / 'cuda'). "
                          "Defaults to auto: cuda when torch reports it "
                          "available, cpu otherwise.")
    rrk.add_argument("--reranker-oom-fallback-batch-size", type=int,
                     default=None,
                     help="Batch size to retry at on a single CUDA OOM "
                          "(default: half of --reranker-batch-size).")
    rrk.add_argument("--extra-hit-k", type=int, action="append",
                     default=None,
                     help="Additional hit@k cutoffs (repeatable). LEGACY "
                          "V3 ONLY: Phase 2A rerank reports usually leave "
                          "this empty — the candidate-recall report exists "
                          "for that.")

    # --- phase2a-reranker-comparison ---
    #
    # Pure post-processing: reads N labelled retrieval_eval_report.json
    # files and emits the comparison .{json,md} into a target directory.
    # Never re-runs retrieval, never re-embeds.
    rcomp = subs.add_parser(
        "phase2a-reranker-comparison",
        description="LEGACY V3 ONLY / Phase 2A historical reproduction.",
        help="LEGACY V3 ONLY / Phase 2A historical reproduction: build "
             "the side-by-side reranker comparison report from N labelled "
             "retrieval_eval_report.json files.",
    )
    rcomp.add_argument(
        "--slice", action="append", required=True,
        help="One labelled report. Format 'label:path/to/"
             "retrieval_eval_report.json'. Repeatable.",
    )
    rcomp.add_argument(
        "--out-json", type=Path, required=True,
        help="Output path for reranker-comparison.json.",
    )
    rcomp.add_argument(
        "--out-md", type=Path, required=True,
        help="Output path for reranker-comparison.md.",
    )
    rcomp.add_argument(
        "--caveat", type=str, action="append", default=None,
        help="Override the default caveat list (repeatable). When omitted, "
             "Phase 2A's standard caveats are emitted.",
    )

    # --- phase2a-reranker-failure-analysis ---
    rfail = subs.add_parser(
        "phase2a-reranker-failure-analysis",
        description="LEGACY V3 ONLY / Phase 2A historical reproduction.",
        help="LEGACY V3 ONLY / Phase 2A historical reproduction: cross-tab "
             "dense vs. rerank hit@1 and dump samples from the rescued / "
             "regressed / both-miss buckets.",
    )
    rfail.add_argument(
        "--dense-report-dir", required=True, type=Path,
        help="Run dir of the dense-only B2 retrieval eval (must contain "
             "retrieval_eval_report.json + top_k_dump.jsonl).",
    )
    rfail.add_argument(
        "--rerank-report-dir", required=True, type=Path,
        help="Run dir of the reranked B2 retrieval eval (must contain "
             "retrieval_eval_report.json + top_k_dump.jsonl).",
    )
    rfail.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write reranker-failure-analysis.{json,md}.",
    )
    rfail.add_argument(
        "--k-preview", type=int, default=5,
        help="Top-N preview rows per query (default: 5).",
    )
    rfail.add_argument(
        "--sample-cap", type=int, default=10,
        help="Cap on per-bucket samples emitted (default: 10).",
    )

    # --- phase2a-latency-sweep ---
    #
    # LEGACY V3 ONLY: Phase 2A-L entrypoint. Builds the offline corpus + FAISS index ONCE
    # and runs the retrieval-eval against it N times — once per
    # ``dense_top_n`` value — to populate the latency-breakdown topN
    # sweep + Pareto frontier + recommended-modes documents in a single
    # command. Reusing the index is the meat of the saving: the
    # repeated rebuild dominates the per-config wall-clock for a 47k-
    # chunk corpus.
    lswp = subs.add_parser(
        "phase2a-latency-sweep",
        description="LEGACY V3 ONLY / Phase 2A-L historical reproduction.",
        help="LEGACY V3 ONLY / Phase 2A-L historical reproduction: run a "
             "topN sweep over an offline corpus, then compute the "
             "latency-breakdown / Pareto frontier / recommended-modes "
             "documents.",
    )
    lswp.add_argument("--dataset", required=True, type=Path,
                      help="Eval-queries JSONL.")
    lswp.add_argument("--corpus", required=True, type=Path,
                      help="Offline corpus JSONL. LEGACY V3 ONLY: B2 "
                           "token-aware-v1 examples are historical "
                           "reproduction inputs, not Phase 7 defaults.")
    lswp.add_argument("--out-dir", type=Path, required=True,
                      help="Top-level directory; per-topN run dirs are "
                           "created underneath, plus the sweep / "
                           "frontier / modes / breakdown documents at "
                           "the top level.")
    lswp.add_argument("--final-top-k", type=int, default=10,
                      help="Final top-k after reranking (default: 10).")
    lswp.add_argument("--dense-top-n", type=int, action="append",
                      default=None,
                      help="Dense top-N values to sweep (repeatable). "
                           "Defaults to 5 10 15 20 30 50.")
    lswp.add_argument("--mrr-k", type=int, default=10)
    lswp.add_argument("--ndcg-k", type=int, default=10)
    lswp.add_argument("--max-seq-length", type=int, default=1024)
    lswp.add_argument("--embed-batch-size", type=int, default=32)
    lswp.add_argument("--reranker-model", type=str,
                      default="BAAI/bge-reranker-v2-m3")
    lswp.add_argument("--reranker-batch-size", type=int, default=16)
    lswp.add_argument("--reranker-max-length", type=int, default=512)
    lswp.add_argument("--reranker-text-max-chars", type=int, default=800)
    lswp.add_argument("--reranker-device", type=str, default=None)
    lswp.add_argument(
        "--breakdown-anchor-dense-top-n", type=int, default=20,
        help="Dense top-N to use as the headline latency-breakdown "
             "anchor (default: 20). The sweep runs all configured "
             "values; the breakdown doc is rendered from this one.",
    )
    lswp.add_argument(
        "--candidate-recall-extra-hit-k",
        type=int, action="append", default=None,
        help="Extra hit@k cutoffs to compute on the dense-only "
             "candidate-recall sibling run. LEGACY V3 ONLY: Phase 2A-L "
             "convention is 10 / 20 / 50. Pass empty to skip the "
             "dense-only run.",
    )
    lswp.add_argument(
        "--skip-candidate-recall", action="store_true",
        help="Skip the dense-only candidate-recall sibling run "
             "entirely. The sweep still produces accuracy metrics for "
             "each topN; only candidate_recall@N annotations on the "
             "sweep table are dropped.",
    )
    lswp.add_argument(
        "--metric", type=str, default="mean_hit_at_1",
        help="Pareto / recommended-modes accuracy metric "
             "(default: mean_hit_at_1).",
    )
    lswp.add_argument(
        "--latency", type=str, default="rerank_p95_ms",
        help="Pareto / recommended-modes latency metric "
             "(default: rerank_p95_ms).",
    )
    lswp.add_argument(
        "--fast-p95-budget-ms", type=float, default=None,
        help="When set, fast mode prefers entries whose p95 latency is "
             "under this budget.",
    )
    lswp.add_argument(
        "--balanced-p95-budget-ms", type=float, default=None,
        help="When set, balanced mode picks the highest-metric "
             "frontier point under this latency budget.",
    )
    lswp.add_argument(
        "--quality-target-metric", type=float, default=None,
        help="Optional quality-mode target value; documented in the "
             "recommended-modes doc when the chosen entry doesn't "
             "reach it.",
    )

    # --- phase2a-latency-breakdown ---
    lbd = subs.add_parser(
        "phase2a-latency-breakdown",
        help="Phase 2A-L post-process: stage-level latency breakdown "
             "for ONE retrieval-rerank report.",
    )
    lbd.add_argument("--report", required=True, type=Path,
                     help="Path to retrieval_eval_report.json.")
    lbd.add_argument("--label", type=str, default=None,
                     help="Optional label printed on the report header.")
    lbd.add_argument("--out-json", type=Path, required=True)
    lbd.add_argument("--out-md", type=Path, required=True)

    # --- phase2a-topn-sweep ---
    tswp = subs.add_parser(
        "phase2a-topn-sweep",
        help="Phase 2A-L post-process: assemble the topN sweep report "
             "from N labelled retrieval reports.",
    )
    tswp.add_argument(
        "--slice", action="append", required=True,
        help="One labelled report. Format 'label:path/to/"
             "retrieval_eval_report.json'. Repeatable.",
    )
    tswp.add_argument(
        "--candidate-recall-report", type=Path, default=None,
        help="Optional dense-only run with mean_extra_hits populated; "
             "used to annotate every sweep entry with the candidate-"
             "recall@N upper bound.",
    )
    tswp.add_argument("--out-json", type=Path, required=True)
    tswp.add_argument("--out-md", type=Path, required=True)
    tswp.add_argument(
        "--caveat", type=str, action="append", default=None,
        help="Override the default caveat list (repeatable).",
    )

    # --- phase2a-recommended-modes ---
    rmod = subs.add_parser(
        "phase2a-recommended-modes",
        help="Phase 2A-L post-process: emit fast / balanced / quality "
             "recommendations from a topN sweep.",
    )
    rmod.add_argument(
        "--sweep-json", required=True, type=Path,
        help="Path to topn-sweep.json.",
    )
    rmod.add_argument("--out-md", type=Path, required=True)
    rmod.add_argument("--out-frontier-json", type=Path, default=None)
    rmod.add_argument("--out-frontier-md", type=Path, default=None)
    rmod.add_argument("--out-modes-json", type=Path, default=None)
    rmod.add_argument(
        "--metric", type=str, default="mean_hit_at_1",
    )
    rmod.add_argument(
        "--latency", type=str, default="rerank_p95_ms",
    )
    rmod.add_argument(
        "--fast-p95-budget-ms", type=float, default=None,
    )
    rmod.add_argument(
        "--balanced-p95-budget-ms", type=float, default=None,
    )
    rmod.add_argument(
        "--quality-target-metric", type=float, default=None,
    )

    # --- retrieval-compare ---
    rc = subs.add_parser(
        "retrieval-compare",
        help="Compare two retrieval-eval reports side-by-side.",
    )
    rc.add_argument(
        "--deterministic-report", required=True, type=Path,
        help="Path to retrieval_eval_report.json from the deterministic run "
             "(e.g. eval/reports/_archive/silver200/baseline/"
             "retrieval_eval_report.json).",
    )
    rc.add_argument(
        "--opus-report", required=True, type=Path,
        help="Path to retrieval_eval_report.json from the opus run.",
    )
    rc.add_argument(
        "--exclude-answer-type", type=str, default="character_relation",
        help="answer_type to exclude when computing the "
             "'deterministic_without_<type>' slice (default: "
             "character_relation, the type the deterministic generator "
             "is broken on).",
    )
    rc.add_argument(
        "--deterministic-max-seq-length", type=int, default=8192,
        help="max_seq_length the deterministic baseline was embedded "
             "with. Recorded in the slice's retriever_config so the "
             "report can flag apples-to-oranges differences. Default "
             "8192 = bge-m3 model default (no truncation).",
    )
    rc.add_argument(
        "--opus-max-seq-length", type=int, default=1024,
        help="max_seq_length the opus baseline was embedded with "
             "(default: 1024).",
    )
    rc.add_argument(
        "--caveat", type=str, action="append", default=None,
        help="Free-text caveat to print at the top of the report. May "
             "be passed multiple times. If omitted, sensible defaults "
             "are added automatically (apples-to-oranges flag, "
             "max_seq_length difference if any, healthy-diagnostic "
             "reminder, plus a tuned-variant flag whenever a slice is "
             "marked --*-kind=tuned).",
    )
    rc.add_argument(
        "--deterministic-kind", type=str, default="baseline",
        choices=("baseline", "tuned"),
        help="Which section of the report the deterministic slice goes "
             "into. Use 'tuned' for hyperparameter-modified runs so "
             "their numbers are rendered in their own headline-metrics "
             "table and never share a row with baseline numbers. "
             "Default: baseline.",
    )
    rc.add_argument(
        "--opus-kind", type=str, default="baseline",
        choices=("baseline", "tuned"),
        help="Same as --deterministic-kind for the opus slice. "
             "Default: baseline.",
    )
    rc.add_argument(
        "--out-json", type=Path, required=True,
        help="Output path for the baseline comparison json (typically phase2/baseline_comparison.json).",
    )
    rc.add_argument(
        "--out-md", type=Path, required=True,
        help="Output path for the baseline comparison md (typically phase2/baseline_comparison.md).",
    )

    # --- retrieval-miss-analysis ---
    ma = subs.add_parser(
        "retrieval-miss-analysis",
        help="Compute miss-bucket analysis from an existing retrieval "
             "report (no re-embed needed).",
    )
    ma.add_argument(
        "--report-dir", required=True, type=Path,
        help="Directory containing retrieval_eval_report.json + "
             "top_k_dump.jsonl (the standard retrieval CLI output dir). "
             "miss_analysis.json + miss_analysis.md are written into the "
             "same directory.",
    )
    ma.add_argument(
        "--top-k", type=int, default=10,
        help="Top-k for the cross-tab classifier (default: 10).",
    )
    ma.add_argument(
        "--sample-limit", type=int, default=25,
        help="Cap on samples kept for each doc_miss_* bucket (default: 25).",
    )

    # --- analyze-corpus-lengths ---
    al = subs.add_parser(
        "analyze-corpus-lengths",
        help="Tokenizer-based char/token length distribution for a "
             "corpus.jsonl. Used to size max_seq_length caps honestly.",
    )
    al.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus.jsonl. LEGACY V3 ONLY example: "
             "eval/corpora/anime_namu_v3/corpus.jsonl. Phase 7 active "
             "work should start from namu-v4-structured-combined.",
    )
    al.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    al.add_argument(
        "--threshold", type=int, action="append", default=None,
        help="Token thresholds to count chunks_over_<t> for. May be "
             "passed multiple times. Defaults to "
             f"{list(DEFAULT_TOKEN_THRESHOLDS)}.",
    )
    al.add_argument(
        "--top-longest", type=int, default=DEFAULT_TOP_LONGEST_CHUNKS,
        help=f"Number of longest chunks to include in the report "
             f"(default: {DEFAULT_TOP_LONGEST_CHUNKS}).",
    )
    al.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (default: 256).",
    )
    al.add_argument(
        "--out-json", type=Path,
        default=Path("eval/reports/phase1/length_analysis.json"),
        help="Output path for the JSON report (default: "
             "eval/reports/phase1/length_analysis.json).",
    )
    al.add_argument(
        "--out-md", type=Path,
        default=Path("eval/reports/phase1/length_analysis.md"),
        help="Output path for the markdown report (default: "
             "eval/reports/phase1/length_analysis.md).",
    )

    # --- audit-corpus-noise ---
    #
    # Phase 1A entrypoint. Reads a corpus through the production
    # chunker, tokenizes everything, and writes both a long-chunk audit
    # and a raw-vs-cleaned length comparison to
    # eval/reports/phase1/1a_corpus_audit/. Does not modify the corpus.
    an = subs.add_parser(
        "audit-corpus-noise",
        help="Phase 1A long-chunk audit + raw-vs-cleaned length "
             "comparison. Does not modify the corpus.",
    )
    an.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus.jsonl. LEGACY V3 ONLY example: "
             "eval/corpora/anime_namu_v3/corpus.jsonl. Phase 7 active "
             "work should start from namu-v4-structured-combined.",
    )
    an.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    an.add_argument(
        "--top-n", type=int, default=DEFAULT_AUDIT_TOP_N,
        help=f"Number of longest chunks to include in the audit "
             f"(default: {DEFAULT_AUDIT_TOP_N}).",
    )
    an.add_argument(
        "--threshold", type=int, action="append", default=None,
        help="Token thresholds for the chunks-over-cap table. May be "
             f"passed multiple times. Defaults to {list(DEFAULT_TOKEN_THRESHOLDS)}.",
    )
    an.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (default: 256).",
    )
    an.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/reports/phase1/1a_corpus_audit"),
        help="Directory to write audit + length-comparison reports "
             "(default: eval/reports/phase1/1a_corpus_audit).",
    )

    # --- clean-corpus-dry-run ---
    #
    # Phase 1A entrypoint #2. Same length-comparison routine as
    # audit-corpus-noise but writes a focused cleaner-effect summary
    # only. The "dry-run" name is load-bearing: this command never
    # writes a cleaned corpus to disk, only the summary stats.
    cd = subs.add_parser(
        "clean-corpus-dry-run",
        help="Phase 1A cleaner-effect summary. Runs the cleaner "
             "in-memory and reports raw-vs-cleaned token/char "
             "distributions; the corpus.jsonl is not modified.",
    )
    cd.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus.jsonl.",
    )
    cd.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    cd.add_argument(
        "--threshold", type=int, action="append", default=None,
        help="Token thresholds for the chunks-over-cap table. May be "
             f"passed multiple times. Defaults to {list(DEFAULT_TOKEN_THRESHOLDS)}.",
    )
    cd.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (default: 256).",
    )
    cd.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/reports/phase1/1a_corpus_audit"),
        help="Directory to write the cleaner-effect summary "
             "(default: eval/reports/phase1/1a_corpus_audit).",
    )

    # --- preprocess-corpus-dry-run ---
    #
    # Phase 1B entrypoint #1. Streams a corpus.jsonl through the
    # ingest-side preprocessor (page-prefix and/or inline-edit-marker
    # strip) without writing a preprocessed corpus to disk. Emits a
    # summary + sample diffs into eval/reports/phase1/1b_preprocess/.
    pp = subs.add_parser(
        "preprocess-corpus-dry-run",
        help="Phase 1B preprocessor dry-run: streams the corpus through "
             "the ingest-side preprocessor, writes summary + sample "
             "diffs only. No artifact corpus is produced.",
    )
    pp.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to the source corpus.jsonl. Not modified.",
    )
    pp.add_argument(
        "--strip-page-prefix", action="store_true",
        help="Enable namu-wiki page-prefix metadata strip.",
    )
    pp.add_argument(
        "--strip-inline-edit", action="store_true",
        help="Enable inline [편집] / [원본 편집] / [소스 편집] strip.",
    )
    pp.add_argument(
        "--sample-diff-n", type=int, default=20,
        help="Number of sample before/after diffs to collect "
             "(default: 20).",
    )
    pp.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/reports/phase1/1b_preprocess"),
        help="Directory for the dry-run summary + sample diffs "
             "(default: eval/reports/phase1/1b_preprocess).",
    )

    # --- emit-preprocessed-corpus ---
    #
    # Phase 1B entrypoint #2. Same preprocessor pass, but writes a
    # preprocessed corpus.<variant>.jsonl plus a manifest.json into
    # eval/corpora/<dir>/. Source corpus is never modified.
    ep = subs.add_parser(
        "emit-preprocessed-corpus",
        description="LEGACY V3 ONLY / Phase 1B historical reproduction.",
        help="LEGACY V3 ONLY / Phase 1B historical reproduction: emit a "
             "preprocessed corpus.jsonl artifact (plus manifest.json). "
             "Source corpus is never modified.",
    )
    ep.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to the source corpus.jsonl. Not modified.",
    )
    ep.add_argument(
        "--strip-page-prefix", action="store_true",
        help="Enable namu-wiki page-prefix metadata strip.",
    )
    ep.add_argument(
        "--strip-inline-edit", action="store_true",
        help="Enable inline [편집] / [원본 편집] / [소스 편집] strip.",
    )
    ep.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/corpora/anime_namu_v3_preprocessed"),
        help="Directory to write the variant corpus + manifest "
             "(LEGACY V3 ONLY default: "
             "eval/corpora/anime_namu_v3_preprocessed).",
    )
    ep.add_argument(
        "--variant-name", type=str, default=None,
        help="Override the variant filename suffix. Defaults to the "
             "config-derived label (e.g. 'prefix-v1', "
             "'prefix-v1.inline-edit-v1').",
    )

    # --- compare-corpus-lengths ---
    #
    # Phase 1B: roll up multiple analyze-corpus-lengths reports into a
    # single side-by-side comparison table. Used to compare raw vs
    # prefix-v1 vs inline-edit-v1 vs combined corpus length
    # distributions in one pass.
    cl = subs.add_parser(
        "compare-corpus-lengths",
        help="Compare N analyze-corpus-lengths reports side-by-side. "
             "Each --analysis takes 'label:path' or just 'path' "
             "(label defaults to the file stem).",
    )
    cl.add_argument(
        "--analysis", action="append", required=True,
        help="One length-analysis report. Format 'label:path' or "
             "just 'path'. May be passed multiple times.",
    )
    cl.add_argument(
        "--out-json", type=Path, required=True,
        help="Output path for the JSON comparison.",
    )
    cl.add_argument(
        "--out-md", type=Path, required=True,
        help="Output path for the markdown comparison.",
    )

    # --- diagnose-chunker-long-tail ---
    #
    # Phase 1C entrypoint #1. Re-runs the corpus through the production
    # chunker, ranks emitted chunks by token count, and dumps the top-N
    # with full provenance (source payload type, original section
    # shape, split attribution). Reads only — corpus is not modified.
    dc = subs.add_parser(
        "diagnose-chunker-long-tail",
        help="Phase 1C: emit chunker provenance for the longest "
             "chunks. Reads only.",
    )
    dc.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus.jsonl (raw or preprocessed).",
    )
    dc.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    dc.add_argument(
        "--top-n", type=int, default=DEFAULT_DIAGNOSE_TOP_N,
        help=f"Number of longest provenance samples to dump "
             f"(default: {DEFAULT_DIAGNOSE_TOP_N}).",
    )
    dc.add_argument(
        "--threshold", type=int, action="append", default=None,
        help="Token thresholds for the chunks-over-cap table. May be "
             f"passed multiple times. Defaults to "
             f"{list(DEFAULT_DIAGNOSE_THRESHOLDS)}.",
    )
    dc.add_argument(
        "--long-chunk-threshold", type=int, default=1024,
        help="Token threshold above which a chunk is counted toward "
             "the long-chunk attribution rollup (default: 1024).",
    )
    dc.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (default: 256).",
    )
    dc.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/reports/phase1/1c_token_chunker"),
        help="Directory to write diagnosis reports "
             "(default: eval/reports/phase1/1c_token_chunker).",
    )

    # --- emit-token-aware-chunks ---
    #
    # Phase 1C entrypoint #2. Streams a corpus through the token-aware
    # chunker and writes a new corpus.<variant>.jsonl whose section
    # ``chunks`` lists are bounded by hard_max_tokens. Source corpus
    # is never modified.
    et = subs.add_parser(
        "emit-token-aware-chunks",
        help="Phase 1C: emit a token-aware chunked corpus + manifest. "
             "Source corpus is never modified.",
    )
    et.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to source corpus.jsonl. Not modified.",
    )
    et.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    et.add_argument(
        "--target-tokens", type=int, default=DEFAULT_TA_TARGET,
        help=f"Target chunk token count (default: {DEFAULT_TA_TARGET}).",
    )
    et.add_argument(
        "--soft-max-tokens", type=int, default=DEFAULT_TA_SOFT_MAX,
        help=f"Soft max - packer flushes when next unit would exceed "
             f"this (default: {DEFAULT_TA_SOFT_MAX}).",
    )
    et.add_argument(
        "--hard-max-tokens", type=int, default=DEFAULT_TA_HARD_MAX,
        help=f"Hard max - no emitted chunk may exceed this in "
             f"production emit (default: {DEFAULT_TA_HARD_MAX}).",
    )
    et.add_argument(
        "--overlap-tokens", type=int, default=DEFAULT_TA_OVERLAP,
        help=f"Adjacent-chunk overlap in tokens (default: "
             f"{DEFAULT_TA_OVERLAP}).",
    )
    et.add_argument(
        "--out-corpus", type=Path, required=True,
        help="Output path for the token-aware corpus.jsonl. The "
             "parent directory is created if needed. Refusing to "
             "overwrite the source corpus.",
    )
    et.add_argument(
        "--manifest", type=Path, default=None,
        help="Output path for the manifest.json (default: "
             "<out-corpus dir>/manifest.json).",
    )
    et.add_argument(
        "--provenance", type=Path, default=None,
        help="Optional path for the per-chunk provenance jsonl "
             "(default: <out-corpus dir>/chunks_provenance.jsonl).",
    )
    et.add_argument(
        "--no-provenance", action="store_true",
        help="Skip writing the provenance jsonl.",
    )
    et.add_argument(
        "--variant-label", type=str, default="combined.token-aware-v1",
        help="Stable label for this variant - recorded in the manifest. "
             "Default: combined.token-aware-v1.",
    )

    # --- compare-chunker-lengths ---
    #
    # Phase 1C entrypoint #3. Takes N corpora (label:path), runs the
    # production chunker length analysis on each, and emits a
    # side-by-side comparison. Unlike compare-corpus-lengths (which
    # consumes pre-computed analysis JSONs), this tokenizes from
    # scratch — useful when the corpora include token-aware-emitted
    # variants whose final chunks live in ``sections.<>.chunks``.
    ccl = subs.add_parser(
        "compare-chunker-lengths",
        help="Phase 1C: tokenize N corpora end-to-end and emit a "
             "length-comparison table.",
    )
    ccl.add_argument(
        "--corpus", action="append", required=True,
        help="One labelled corpus. Format 'label:path' or just 'path' "
             "(label defaults to the file stem). May be passed "
             "multiple times.",
    )
    ccl.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help=f"HuggingFace tokenizer name (default: {DEFAULT_TOKENIZER_NAME}).",
    )
    ccl.add_argument(
        "--threshold", type=int, action="append", default=None,
        help="Token thresholds for the chunks-over-cap table. Defaults "
             f"to {list(DEFAULT_TOKEN_THRESHOLDS)}.",
    )
    ccl.add_argument(
        "--top-longest", type=int, default=DEFAULT_TOP_LONGEST_CHUNKS,
        help=f"Per-corpus longest-chunks count for the per-corpus "
             f"length analysis (default: {DEFAULT_TOP_LONGEST_CHUNKS}).",
    )
    ccl.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (default: 256).",
    )
    ccl.add_argument(
        "--out-dir", type=Path,
        default=Path("eval/reports/phase1/1c_token_chunker"),
        help="Directory to write per-corpus length JSONs + the "
             "comparison report "
             "(default: eval/reports/phase1/1c_token_chunker).",
    )

    # --- retrieval-candidate-boost ---
    #
    # Phase 2B entrypoint #1. Wraps the offline retrieval-rerank stack
    # with a metadata-based boost reranker plugged in BEFORE the
    # cross-encoder (or used standalone for the boost-only baseline).
    # All boost weights are CLI-overridable so an experiment matrix
    # can sweep them without code changes; setting them all to 0 is
    # a byte-identical baseline against retrieval-rerank with the
    # same dense_top_n / final_top_k.
    rcb = subs.add_parser(
        "retrieval-candidate-boost",
        help="Phase 2B: dense retrieval + metadata boost (+ optional "
             "cross-encoder rerank) over an offline corpus.",
    )
    rcb.add_argument("--dataset", required=True, type=Path)
    rcb.add_argument("--corpus", required=True, type=Path,
                     help="B2 token-aware-v1 corpus JSONL.")
    rcb.add_argument("--out-dir", type=Path, default=None)
    rcb.add_argument("--top-n", type=int, default=15,
                     help="Final top-K returned by the boost stage. "
                          "Phase 2B keeps this at 5/10/15 — bringing "
                          "back top-20+ would defeat the goal.")
    rcb.add_argument("--final-top-k", type=int, default=None,
                     help="Optional. When a post-rerank stage is "
                          "attached this caps the final output below "
                          "--top-n; defaults to --top-n.")
    rcb.add_argument("--mrr-k", type=int, default=10)
    rcb.add_argument("--ndcg-k", type=int, default=10)
    rcb.add_argument("--max-seq-length", type=int, default=1024)
    rcb.add_argument("--embed-batch-size", type=int, default=32)
    rcb.add_argument(
        "--extra-hit-k", type=int, action="append", default=None,
        help="Additional hit@k cutoff (repeatable). Phase 2B usually "
             "passes 5/10/15 here so the candidate-recall metrics are "
             "in the same report.",
    )
    # Boost knobs.
    rcb.add_argument("--title-exact-boost", type=float, default=0.0,
                     help="Boost added when the doc title appears "
                          "verbatim in the (normalized) query.")
    rcb.add_argument("--title-partial-boost", type=float, default=0.0,
                     help="Boost added when any title token "
                          "(>= --title-min-len chars) appears in the "
                          "query. Falls back when --title-exact-boost "
                          "did not fire.")
    rcb.add_argument("--section-keyword-boost", type=float, default=0.0,
                     help="Boost when the chunk's section name appears "
                          "in the query and is not in --excluded-section.")
    rcb.add_argument("--section-path-boost", type=float, default=0.0,
                     help="Boost when ANY of the doc's other section "
                          "names appears in the query (weaker proxy).")
    rcb.add_argument("--max-boost", type=float, default=0.30,
                     help="Per-chunk total boost clamp; set to 0 to "
                          "disable clamping.")
    rcb.add_argument("--title-min-len", type=int, default=2,
                     help="Minimum title-token length for partial match.")
    rcb.add_argument(
        "--excluded-section", type=str, action="append", default=None,
        help="Section names to skip for keyword boost (repeatable). "
             "Defaults to 본문, 요약 — both appear on every doc.",
    )
    rcb.add_argument(
        "--reranker-model", type=str, default=None,
        help="Optional cross-encoder reranker model name. When set, "
             "the cross-encoder runs AFTER the boost reorder.",
    )
    rcb.add_argument("--reranker-batch-size", type=int, default=16)
    rcb.add_argument("--reranker-max-length", type=int, default=512)
    rcb.add_argument("--reranker-text-max-chars", type=int, default=800)
    rcb.add_argument("--reranker-device", type=str, default=None)
    rcb.add_argument("--reranker-oom-fallback-batch-size", type=int,
                     default=None)

    # --- retrieval-candidate-miss-analysis ---
    #
    # Phase 2B entrypoint #2. Reads an existing retrieval report
    # (typically the candidate-recall sibling that goes deep enough
    # to surface the corpus_missing rate) and re-tabulates misses at
    # multiple top-K cutoffs into the eight Phase 2B failure buckets.
    rcma = subs.add_parser(
        "retrieval-candidate-miss-analysis",
        help="Phase 2B: classify Phase 2A retrieval misses into "
             "title / character / section / lexical / alias / broad "
             "/ ambiguous / corpus-missing buckets at top-5/10/15.",
    )
    rcma.add_argument(
        "--report", required=True, type=Path,
        help="retrieval_eval_report.json from a Phase 2A run with "
             "deep-enough top_k (>= --deep-k).",
    )
    rcma.add_argument(
        "--top-k-dump", type=Path, default=None,
        help="top_k_dump.jsonl from the same run; enables section / "
             "lexical bucket detection.",
    )
    rcma.add_argument(
        "--corpus", type=Path, default=None,
        help="Optional corpus JSONL for doc title / section metadata "
             "lookup (enables title_mismatch + alias_or_synonym buckets).",
    )
    rcma.add_argument(
        "--top-k", type=int, action="append", default=None,
        help="Top-K cutoff to evaluate misses at. Repeatable. "
             "Defaults to (5, 10, 15).",
    )
    rcma.add_argument(
        "--deep-k", type=int, default=DEFAULT_MISS_DEEP_K,
        help=f"Cutoff for the corpus_missing check "
             f"(default: {DEFAULT_MISS_DEEP_K}).",
    )
    rcma.add_argument(
        "--sample-limit", type=int, default=25,
        help="Cap per-bucket sample list (default: 25).",
    )
    rcma.add_argument("--out-dir", type=Path, default=None)

    # --- retrieval-boost-failure-analysis ---
    #
    # Phase 2B entrypoint #3. Cross-tabs three retrieval reports
    # (dense baseline, dense+boost, optionally dense+boost+rerank)
    # into the five outcome groups so a reviewer can inspect every
    # query the boost rescued or hurt.
    rbfa = subs.add_parser(
        "retrieval-boost-failure-analysis",
        help="Phase 2B: bucket queries by their dense → boost (→ rerank) "
             "trajectory and emit per-bucket samples.",
    )
    rbfa.add_argument("--dense-report", required=True, type=Path)
    rbfa.add_argument("--boost-report", required=True, type=Path)
    rbfa.add_argument("--rerank-report", type=Path, default=None,
                      help="Optional dense+boost+rerank report.")
    rbfa.add_argument("--boost-dump", type=Path, default=None,
                      help="Optional boost_dump.jsonl from the boost "
                           "report; enables per-chunk boost scores in "
                           "the sample entries.")
    rbfa.add_argument("--top-k", type=int, default=10)
    rbfa.add_argument("--sample-limit", type=int, default=30)
    rbfa.add_argument("--out-dir", type=Path, default=None)

    # --- retrieval-boost-pareto ---
    #
    # Phase 2B entrypoint #4. Merges a Phase 2A topN sweep with one
    # or more Phase 2B sweeps and emits a unified accuracy↔latency
    # Pareto frontier so a reviewer can see whether boost pushes
    # the frontier vs the no-boost baseline.
    rbp = subs.add_parser(
        "retrieval-boost-pareto",
        help="Phase 2B: merge Phase 2A baseline + Phase 2B boost "
             "topN sweeps into one Pareto frontier.",
    )
    rbp.add_argument(
        "--phase2a-sweep", required=True, type=Path,
        help="Phase 2A topn-sweep.json.",
    )
    rbp.add_argument(
        "--phase2b-sweep", required=True, type=Path,
        help="Phase 2B boost topn-sweep.json (built from the boost "
             "retrieval reports via phase2a-topn-sweep CLI mode).",
    )
    rbp.add_argument("--metric", type=str, default="mean_hit_at_1")
    rbp.add_argument("--latency", type=str, default="rerank_p95_ms")
    rbp.add_argument("--phase2b-label", type=str, default=None,
                     help="Optional human-readable boost-config label "
                          "carried in the report alongside each Phase 2B "
                          "entry.")
    rbp.add_argument("--out-dir", type=Path, default=None)

    return parser


def _configure_logging(*, verbose: bool) -> None:
    try:
        from app.core.logging import configure_logging

        configure_logging()
    except Exception:  # pragma: no cover — CLI should work even before app is importable
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        )
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# RAG CLI path.
# ---------------------------------------------------------------------------


def _run_rag_cli(args: argparse.Namespace) -> int:
    offline_info = None
    try:
        if args.offline_corpus is not None:
            retriever, generator, settings, offline_info = _build_offline_rag_stack(args)
        else:
            retriever, generator, settings = _build_rag_stack(args)
    except Exception as ex:
        log.error(
            "Failed to build the RAG eval stack (%s: %s). "
            "Make sure the FAISS index is built, the ragmeta schema exists, "
            "and the configured embedding model matches the one in build.json. "
            "For a ragmeta-less run pass --offline-corpus <corpus.jsonl>.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    top_k = int(args.top_k) if args.top_k is not None else int(settings.rag_top_k)

    if args.agent_mode == "compare":
        return _run_rag_compare_cli(
            args,
            dataset=dataset,
            retriever=retriever,
            generator=generator,
            settings=settings,
            top_k=top_k,
            offline_info=offline_info,
        )

    if args.cross_domain:
        return _run_rag_cross_domain_cli(
            args,
            dataset=dataset,
            retriever=retriever,
            generator=generator,
            settings=settings,
            top_k=top_k,
            offline_info=offline_info,
        )

    summary, rows = run_rag_eval(
        dataset,
        retriever=retriever,
        generator=generator,
        top_k=top_k,
        dataset_path=str(args.dataset),
    )

    # Write reports BEFORE the console pretty-print so a stray Unicode
    # character (cp949 consoles on Windows) can't cost us the data.
    out_json = args.out_json or _default_report_path("rag", "json")
    write_json_report(
        out_json,
        summary=rag_summary_to_dict(summary),
        rows=[rag_row_to_dict(r) for r in rows],
        metadata=_rag_metadata(settings, top_k, offline_info=offline_info),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("rag", "csv")
        write_csv_report(
            out_csv,
            [rag_row_to_dict(r) for r in rows],
            columns=_rag_csv_columns(),
        )
    try:
        _print_rag_summary(summary)
    except UnicodeEncodeError as ex:  # pragma: no cover — win-cp949 fallback
        log.warning(
            "Pretty summary print failed (%s); report already written to %s",
            ex, out_json,
        )
    return 0


def _run_rag_compare_cli(
    args: argparse.Namespace,
    *,
    dataset: List[Mapping[str, Any]],
    retriever: Any,
    generator: Any,
    settings: Any,
    top_k: int,
    offline_info: Optional[Any],
) -> int:
    """Run the loop-off vs loop-on compare harness over ``dataset``.

    Builds a live AgentLoopController + RuleCritic + NoOpQueryRewriter
    stack using the same retriever + generator the regular RAG eval
    uses. This matches what the worker actually runs when the LLM
    backend is unavailable (the production registry degrades to
    RuleCritic + NoOpQueryRewriter in that case) — exactly the
    scenario the Phase 8 decision gate needs to measure.
    """
    from app.capabilities.agent.critic import RuleCritic
    from app.capabilities.agent.loop import AgentLoopController, LoopBudget
    from app.capabilities.agent.rewriter import NoOpQueryRewriter
    from app.capabilities.agent.synthesizer import AgentSynthesizer
    from app.capabilities.rag.query_parser import RegexQueryParser

    from eval.harness.io_utils import write_csv_report, write_json_report
    from eval.harness.rag_eval import (
        AgentRunResult,
        agent_compare_row_to_dict,
        run_agent_compare_eval,
        summarize_agent_compare,
    )

    parser = RegexQueryParser()
    critic = RuleCritic()
    rewriter = NoOpQueryRewriter()
    synthesizer = AgentSynthesizer(generator)
    budget = LoopBudget(
        max_iter=int(settings.agent_max_iter),
        max_total_ms=int(settings.agent_max_total_ms),
        max_llm_tokens=int(settings.agent_max_llm_tokens),
        min_confidence_to_stop=float(settings.agent_min_stop_confidence),
    )
    controller = AgentLoopController(
        critic=critic,
        rewriter=rewriter,
        parser=parser,
        budget=budget,
    )

    def agent_run_fn(query: str, loop_enabled: bool) -> AgentRunResult:
        import time as _t

        start = _t.perf_counter()
        parsed = parser.parse(query)
        if not loop_enabled:
            report = retriever.retrieve(parsed.normalized or query)
            chunks = list(report.results)
            answer = generator.generate(query, chunks)
            elapsed_ms = round((_t.perf_counter() - start) * 1000.0, 3)
            # loop=off rough token approximation: characters / 4, floor 1.
            tokens = max(1, len(answer) // 4)
            doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
            return AgentRunResult(
                answer=answer,
                retrieved_doc_ids=doc_ids,
                tokens_used=tokens,
                elapsed_ms=elapsed_ms,
                iter_count=1,
                stop_reason="loop_off",
            )

        def execute_fn(pq: Any) -> tuple:
            q = pq.normalized or pq.original or query
            rep = retriever.retrieve(q)
            chunks = list(rep.results)
            ans = generator.generate(query, chunks)
            # Rule critic has no LLM token cost; approximate execute
            # tokens from generator output so the token budget math in
            # the loop behaves realistically.
            return ans, chunks, max(1, len(ans) // 4)

        outcome = controller.run(
            question=query,
            initial_parsed_query=parsed,
            execute_fn=execute_fn,
        )
        final_answer = synthesizer.synthesize(query, outcome)
        elapsed_ms = round((_t.perf_counter() - start) * 1000.0, 3)
        agg_doc_ids = list(
            dict.fromkeys(c.doc_id for c in outcome.aggregated_chunks)
        )
        # Approximate final-answer synthesis tokens the same way as
        # iter0 so cost multiplier math is meaningful even though
        # offline runs use a deterministic extractive generator.
        total_tokens = outcome.total_llm_tokens + max(1, len(final_answer) // 4)
        return AgentRunResult(
            answer=final_answer,
            retrieved_doc_ids=agg_doc_ids,
            tokens_used=total_tokens,
            elapsed_ms=elapsed_ms,
            iter_count=len(outcome.steps),
            stop_reason=outcome.stop_reason,
        )

    compare_rows = run_agent_compare_eval(
        dataset,
        agent_run_fn=agent_run_fn,
        dataset_path=str(args.dataset),
        top_k=top_k,
    )
    compare_summary = summarize_agent_compare(
        compare_rows, top_k=top_k, dataset_path=str(args.dataset)
    )

    # Write reports BEFORE the console pretty-print so a stray Unicode
    # character (cp949 consoles on Windows) can't cost us the data.
    out_json = args.out_json or _default_report_path("rag-compare", "json")
    write_json_report(
        out_json,
        summary=compare_summary,
        rows=[agent_compare_row_to_dict(r) for r in compare_rows],
        metadata=_rag_compare_metadata(
            settings, top_k, budget, offline_info=offline_info
        ),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("rag-compare", "csv")
        write_csv_report(
            out_csv,
            [agent_compare_row_to_dict(r) for r in compare_rows],
            columns=_rag_compare_csv_columns(),
        )
    try:
        _print_agent_compare_summary(compare_summary)
    except UnicodeEncodeError as ex:  # pragma: no cover — win-cp949 fallback
        log.warning(
            "Pretty summary print failed (%s); reports already written "
            "to %s", ex, out_json,
        )
    return 0


def _run_rag_cross_domain_cli(
    args: argparse.Namespace,
    *,
    dataset: List[Mapping[str, Any]],
    retriever: Any,
    generator: Any,
    settings: Any,
    top_k: int,
    offline_info: Optional[Any],
) -> int:
    """Score a cross-domain unanswerable dataset (Phase 9).

    Swaps in an ExtractiveGenerator with ``low_relevance_threshold`` so
    the extractive path emits a Korean refusal when the filter allows
    only off-topic chunks through. The threshold defaults to
    ``settings.rag_cross_domain_relevance_threshold`` (0.35 for
    bge-m3 normalized IP) so on-topic queries still get their normal
    grounded answer when the filter DOES permit the right corpus.
    """
    from app.capabilities.rag.generation import ExtractiveGenerator
    from eval.harness.rag_eval import (
        cross_domain_row_to_dict,
        run_rag_cross_domain_eval,
    )

    # bge-m3 normalized IP on mixed KR/EN text typically gives
    # on-topic > 0.5 and cross-domain < 0.45 — 0.48 sits in the clear
    # space between the two distributions (on-topic min on the Phase 9
    # baselines is 0.519; cross-domain top score is 0.422).
    threshold = float(
        getattr(settings, "rag_cross_domain_relevance_threshold", 0.48)
    )
    cross_domain_generator = ExtractiveGenerator(
        low_relevance_threshold=threshold,
    )
    log.info(
        "Cross-domain eval using relevance-gated ExtractiveGenerator "
        "(threshold=%.3f)", threshold,
    )

    summary, rows = run_rag_cross_domain_eval(
        dataset,
        retriever=retriever,
        generator=cross_domain_generator,
        dataset_path=str(args.dataset),
    )

    # Write reports BEFORE the console pretty-print so a stray Unicode
    # character (cp949 consoles on Windows) can't cost us the data.
    out_json = args.out_json or _default_report_path("rag-cross-domain", "json")
    write_json_report(
        out_json,
        summary={
            **{k: v for k, v in summary.items() if k != "rows"},
            "top_k": top_k,
        },
        rows=summary["rows"],
        metadata=_rag_metadata(settings, top_k, offline_info=offline_info),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("rag-cross-domain", "csv")
        write_csv_report(
            out_csv,
            [cross_domain_row_to_dict(r) for r in rows],
            columns=[
                "query", "filters", "expected_action",
                "retrieved_doc_ids", "filter_produced_no_docs",
                "refusal_detected", "cross_domain_pass",
                "answer", "retrieval_ms", "generation_ms",
                "notes", "error",
            ],
        )
    try:
        print()
        print(f"Cross-domain RAG eval: {summary['dataset_path']}")
        print(f"  rows                       : {summary['row_count']}")
        print(f"  errors                     : {summary['error_count']}")
        print(f"  cross_domain_refusal_rate  : {summary['cross_domain_refusal_rate']}")
        print(f"  cross_domain_zero_results  : {summary['cross_domain_zero_results_rate']}")
        print(f"  passing rows               : {summary['passing_rows']}")
        print()
    except UnicodeEncodeError as ex:  # pragma: no cover — win-cp949 fallback
        log.warning(
            "Pretty summary print failed (%s); report already written to %s",
            ex, out_json,
        )
    return 0


def _build_rag_stack(args: argparse.Namespace):
    """Construct the same Retriever + ExtractiveGenerator the worker
    uses — but without any Spring/queue wiring. The import path is
    local to this function so a bare `python -m eval.run_eval ocr ...`
    call doesn't pull in faiss / psycopg2 / torch."""
    from pathlib import Path as _P

    from app.capabilities.registry import _build_reranker
    from app.capabilities.rag.embeddings import (
        SentenceTransformerEmbedder,
        resolve_max_seq_length,
    )
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.generation import ExtractiveGenerator
    from app.capabilities.rag.metadata_store import RagMetadataStore
    from app.capabilities.rag.retriever import Retriever
    from app.core.config import get_settings

    settings = get_settings()
    top_k = int(args.top_k) if args.top_k is not None else int(settings.rag_top_k)

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
    index = FaissIndex(_P(settings.rag_index_dir))
    reranker = _build_reranker(settings)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=top_k,
        reranker=reranker,
        candidate_k=settings.rag_candidate_k,
    )
    retriever.ensure_ready()
    return retriever, ExtractiveGenerator(), settings


def _build_offline_rag_stack(args: argparse.Namespace):
    """Same bge-m3 embedder as production but an in-memory metadata store.

    The point of this path is to produce a real retrieval-quality baseline
    on a dev machine that has the embedding model cached but no Postgres.
    Not a substitute for the live stack in production eval runs.
    """
    import tempfile
    from pathlib import Path as _P

    from app.capabilities.rag.embeddings import (
        SentenceTransformerEmbedder,
        resolve_max_seq_length,
    )
    from app.core.config import get_settings

    from eval.harness.offline_corpus import build_offline_rag_stack

    settings = get_settings()
    top_k = int(args.top_k) if args.top_k is not None else int(settings.rag_top_k)

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=resolve_max_seq_length(settings.rag_embedding_max_seq_length),
        batch_size=int(settings.rag_embedding_batch_size),
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    tmp_dir = _P(tempfile.mkdtemp(prefix="rag-eval-offline-"))
    retriever, generator, info = build_offline_rag_stack(
        _P(args.offline_corpus),
        embedder=embedder,
        index_dir=tmp_dir,
        top_k=top_k,
    )
    return retriever, generator, settings, info


def _rag_metadata(
    settings: Any,
    top_k: int,
    *,
    offline_info: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "harness": "rag",
        "embedding_model": settings.rag_embedding_model,
        "rag_index_dir": str(settings.rag_index_dir),
        "top_k": top_k,
        "reranker": settings.rag_reranker,
        "candidate_k": settings.rag_candidate_k,
        "rerank_batch": settings.rag_rerank_batch,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    if offline_info is not None:
        payload["offline_corpus"] = {
            "path": offline_info.corpus_path,
            "document_count": offline_info.document_count,
            "chunk_count": offline_info.chunk_count,
            "index_version": offline_info.index_version,
            "dimension": offline_info.dimension,
        }
    return payload


def _rag_csv_columns() -> List[str]:
    return [
        "query",
        "expected_doc_ids",
        "retrieved_doc_ids",
        "hit_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "keyword_coverage",
        "dup_rate",
        "topk_gap",
        "topk_rel_gap",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "index_version",
        "embedding_model",
        "reranker_name",
        "candidate_k",
        "retrieval_scores",
        "rerank_scores",
        "notes",
        "error",
    ]


def _rag_compare_metadata(
    settings: Any,
    top_k: int,
    budget: Any,
    *,
    offline_info: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "harness": "rag-agent-compare",
        "embedding_model": settings.rag_embedding_model,
        "rag_index_dir": str(settings.rag_index_dir),
        "top_k": top_k,
        "reranker": settings.rag_reranker,
        "candidate_k": settings.rag_candidate_k,
        "agent_critic": "rule",
        "agent_rewriter": "noop",
        "agent_budget": {
            "max_iter": budget.max_iter,
            "max_total_ms": budget.max_total_ms,
            "max_llm_tokens": budget.max_llm_tokens,
            "min_confidence_to_stop": budget.min_confidence_to_stop,
        },
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    if offline_info is not None:
        payload["offline_corpus"] = {
            "path": offline_info.corpus_path,
            "document_count": offline_info.document_count,
            "chunk_count": offline_info.chunk_count,
            "index_version": offline_info.index_version,
            "dimension": offline_info.dimension,
        }
    return payload


def _rag_compare_csv_columns() -> List[str]:
    return [
        "query",
        "difficulty",
        "expected_doc_ids",
        "expected_keywords",
        "iter0_retrieved_doc_ids",
        "iter0_recall_at_k",
        "iter0_reciprocal_rank",
        "iter0_keyword_coverage",
        "iter0_tokens",
        "iter0_ms",
        "final_retrieved_doc_ids",
        "final_recall_at_k",
        "final_reciprocal_rank",
        "final_keyword_coverage",
        "total_tokens",
        "total_ms",
        "iter_count",
        "stop_reason",
        "notes",
        "error",
    ]


def _print_agent_compare_summary(summary: Dict[str, Any]) -> None:
    print()
    print(f"Agent compare eval - {summary['dataset_path']}")
    print(f"  rows evaluated : {summary['row_count']} (errors={summary['error_count']})")
    print(f"  top_k          : {summary['top_k']}")

    def _fmt_bucket(label: str, bucket: Dict[str, Any]) -> None:
        print(
            f"    {label:<6} n={bucket['row_count']:>2}  "
            f"recall@k={_fmt(bucket['mean_recall_at_k'])}  "
            f"mrr={_fmt(bucket['mrr'])}  "
            f"kw_cov={_fmt(bucket['mean_keyword_coverage'])}  "
            f"p50_ms={bucket['p50_latency_ms']:.1f}  "
            f"p95_ms={bucket['p95_latency_ms']:.1f}  "
            f"mean_tokens={bucket['mean_tokens']:.1f}"
        )

    print("  overall:")
    _fmt_bucket("off", summary["overall"]["off"])
    _fmt_bucket("on",  summary["overall"]["on"])

    print("  per_difficulty:")
    for tag in ("easy", "hard", "impossible"):
        print(f"    {tag}:")
        _fmt_bucket("off", summary["per_difficulty"][tag]["off"])
        _fmt_bucket("on",  summary["per_difficulty"][tag]["on"])

    am = summary["agent_metrics"]
    print("  agent metrics:")
    print(f"    loop_recovery_rate_overall : {_fmt(am['loop_recovery_rate_overall'])}")
    print(f"    loop_recovery_rate_hard    : {_fmt(am['loop_recovery_rate_hard'])}")
    print(f"    avg_cost_multiplier        : {_fmt(am['avg_cost_multiplier'])}")
    print(f"    iter_count_mean            : {_fmt(am['iter_count_mean'])}")
    print(f"    answer_recall_delta        : {_fmt(am['answer_recall_delta'])}")

    ld = summary["latency_delta"]
    print("  latency delta (on - off):")
    print(f"    mean_ms={ld['mean_ms']:.1f}  p50_ms={ld['p50_ms']:.1f}  p95_ms={ld['p95_ms']:.1f}")

    print("  stop_reason distribution (loop=on):")
    for reason, info in summary["stop_reason_distribution"].items():
        print(f"    {reason:<14} count={info['count']:<3} frac={info['fraction']:.3f}")
    print()


def _print_rag_summary(summary: RagEvalSummary) -> None:
    print()
    print(f"RAG eval - {summary.dataset_path}")
    print(f"  rows evaluated       : {summary.row_count}")
    print(f"  with expected_doc_ids: {summary.rows_with_expected_doc_ids}")
    print(f"  with expected_kwds   : {summary.rows_with_expected_keywords}")
    print(f"  top_k                : {summary.top_k}")
    print(f"  mean hit@k           : {_fmt(summary.mean_hit_at_k)}")
    print(f"  mean recall@k        : {_fmt(summary.mean_recall_at_k)}")
    print(f"  MRR                  : {_fmt(summary.mrr)}")
    print(f"  mean keyword coverage: {_fmt(summary.mean_keyword_coverage)}")
    print(f"  mean dup_rate        : {summary.mean_dup_rate:.4f}")
    print(f"  mean top-k gap       : {_fmt(summary.mean_topk_gap)}")
    print(f"  mean retrieval_ms    : {summary.mean_retrieval_ms:.1f}")
    print(f"  p50 retrieval_ms     : {summary.p50_retrieval_ms:.1f}")
    print(f"  p95 retrieval_ms     : {summary.p95_retrieval_ms:.1f}")
    print(f"  mean generation_ms   : {summary.mean_generation_ms:.1f}")
    print(f"  mean total_ms        : {summary.mean_total_ms:.1f}")
    print(f"  errors               : {summary.error_count}")
    print(f"  misses (capped 20)   : {len(summary.misses)}")
    print(f"  index_version        : {summary.index_version}")
    print(f"  embedding_model      : {summary.embedding_model}")
    print(f"  reranker             : {summary.reranker_name or 'n/a'}")
    print(f"  candidate_k          : {summary.candidate_k if summary.candidate_k is not None else 'n/a'}")
    print()


# ---------------------------------------------------------------------------
# OCR CLI path.
# ---------------------------------------------------------------------------


def _run_ocr_cli(args: argparse.Namespace) -> int:
    try:
        provider, settings = _build_ocr_provider()
    except Exception as ex:
        log.error(
            "Failed to build the OCR eval provider (%s: %s). "
            "Install tesseract (https://tesseract-ocr.github.io/) + the "
            "configured language packs, install `pytesseract` and `pymupdf`, "
            "or set AIPIPELINE_WORKER_OCR_TESSERACT_CMD.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    dataset_dir = args.dataset.resolve().parent

    summary, rows = run_ocr_eval(
        dataset,
        provider=provider,
        dataset_dir=dataset_dir,
        dataset_path=str(args.dataset),
        skip_missing_files=(not args.fail_missing),
    )

    _print_ocr_summary(summary)

    out_json = args.out_json or _default_report_path("ocr", "json")
    write_json_report(
        out_json,
        summary=ocr_summary_to_dict(summary),
        rows=[ocr_row_to_dict(r) for r in rows],
        metadata=_ocr_metadata(settings),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("ocr", "csv")
        write_csv_report(
            out_csv,
            [ocr_row_to_dict(r) for r in rows],
            columns=_ocr_csv_columns(),
        )
    return 0


def _build_ocr_provider():
    from app.capabilities.ocr.tesseract_provider import TesseractOcrProvider
    from app.core.config import get_settings

    settings = get_settings()
    provider = TesseractOcrProvider(
        languages=settings.ocr_languages,
        pdf_dpi=settings.ocr_pdf_dpi,
        tesseract_cmd=settings.ocr_tesseract_cmd,
    )
    provider.ensure_ready()
    return provider, settings


def _ocr_metadata(settings: Any) -> Dict[str, Any]:
    return {
        "harness": "ocr",
        "ocr_languages": settings.ocr_languages,
        "ocr_pdf_dpi": settings.ocr_pdf_dpi,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }


def _ocr_csv_columns() -> List[str]:
    return [
        "file",
        "kind",
        "language",
        "expected_length",
        "text_length",
        "cer",
        "wer",
        "is_empty",
        "latency_ms",
        "avg_confidence",
        "page_count",
        "engine_name",
        "notes",
        "skipped_reason",
        "error",
    ]


def _print_ocr_summary(summary: OcrEvalSummary) -> None:
    print()
    print(f"OCR eval - {summary.dataset_path}")
    print(f"  rows in dataset  : {summary.row_count}")
    print(f"  evaluated        : {summary.evaluated_rows}")
    print(f"  skipped (missing): {summary.skipped_rows}")
    print(f"  errors           : {summary.error_count}")
    print(f"  empty_rate       : {summary.empty_rate:.3f}")
    print(f"  mean CER         : {_fmt(summary.mean_cer)}")
    print(f"  median CER       : {_fmt(summary.median_cer)}")
    print(f"  max CER          : {_fmt(summary.max_cer)}")
    print(f"  mean WER         : {_fmt(summary.mean_wer)}")
    print(f"  mean latency_ms  : {summary.mean_latency_ms:.1f}")
    print(f"  p50 latency_ms   : {summary.p50_latency_ms:.1f}")
    print(f"  engine           : {summary.engine_name}")
    print()


# ---------------------------------------------------------------------------
# Multimodal CLI path.
# ---------------------------------------------------------------------------


def _run_multimodal_cli(args: argparse.Namespace) -> int:
    try:
        capability, settings = _build_multimodal_stack(args)
    except Exception as ex:
        log.error(
            "Failed to build the multimodal eval stack (%s: %s). "
            "Make sure the FAISS index is built, Tesseract is available, "
            "and the configured embedding model matches the one in build.json.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    dataset_dir = args.dataset.resolve().parent

    summary, rows = run_multimodal_eval(
        dataset,
        capability=capability,
        input_builder=_build_multimodal_input,
        dataset_dir=dataset_dir,
        dataset_path=str(args.dataset),
        require_ocr_only=args.require_ocr_only,
    )

    _print_multimodal_summary(summary)

    out_json = args.out_json or _default_report_path("multimodal", "json")
    write_json_report(
        out_json,
        summary=mm_summary_to_dict(summary),
        rows=[mm_row_to_dict(r) for r in rows],
        metadata=_multimodal_metadata(
            settings, cross_modal=(cross_modal is not None),
        ),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("multimodal", "csv")
        write_csv_report(
            out_csv,
            [mm_row_to_dict(r) for r in rows],
            columns=_multimodal_csv_columns(),
        )
    return 0


def _build_multimodal_stack(args: argparse.Namespace):
    """Build the full MULTIMODAL capability from worker settings."""
    from app.capabilities.registry import (
        _build_vision_provider,
        _get_shared_ocr_provider,
        _get_shared_retriever_bundle,
    )
    from app.capabilities.multimodal.capability import (
        MultimodalCapability,
        MultimodalCapabilityConfig,
    )
    from app.core.config import get_settings

    settings = get_settings()

    # Optional CLI override for vision provider.
    if args.vision_provider:
        settings.multimodal_vision_provider = args.vision_provider

    ocr_provider = _get_shared_ocr_provider(settings)
    retriever, generator = _get_shared_retriever_bundle(settings)
    vision_provider = _build_vision_provider(settings)

    # Optional CLI override for cross-modal retrieval.
    cross_modal = None
    use_cross_modal = getattr(args, "cross_modal", False)
    if use_cross_modal:
        try:
            from app.capabilities.registry import _build_cross_modal_retriever
            cross_modal = _build_cross_modal_retriever(settings, retriever)
        except Exception as ex:
            log.warning(
                "Cross-modal retriever not available for eval (%s: %s). "
                "Falling back to text-only.",
                type(ex).__name__, ex,
            )

    capability = MultimodalCapability(
        ocr_provider=ocr_provider,
        vision_provider=vision_provider,
        retriever=retriever,
        generator=generator,
        config=MultimodalCapabilityConfig(
            pdf_vision_dpi=settings.multimodal_pdf_vision_dpi,
            emit_trace=True,  # always emit trace for eval latency breakdowns
            default_user_question=settings.multimodal_default_question,
            use_cross_modal_retrieval=cross_modal is not None,
        ),
        cross_modal_retriever=cross_modal,
    )
    return capability, settings


def _build_multimodal_input(
    image_path: str,
    image_bytes: bytes,
    question: str,
    filename: Optional[str],
) -> Any:
    """Build a CapabilityInput for the multimodal capability."""
    from app.capabilities.base import CapabilityInput, CapabilityInputArtifact

    artifacts = [
        CapabilityInputArtifact(
            artifact_id=f"eval-file-{hash(image_path) & 0xFFFFFFFF:08x}",
            type="INPUT_FILE",
            content=image_bytes,
            content_type=_guess_mime_from_path(image_path),
            filename=filename,
        ),
    ]
    if question:
        artifacts.append(
            CapabilityInputArtifact(
                artifact_id=f"eval-q-{hash(question) & 0xFFFFFFFF:08x}",
                type="INPUT_TEXT",
                content=question.encode("utf-8"),
                content_type="text/plain",
            )
        )
    return CapabilityInput(
        job_id=f"eval-mm-{hash(image_path) & 0xFFFFFFFF:08x}",
        capability="MULTIMODAL",
        attempt_no=1,
        inputs=artifacts,
    )


def _guess_mime_from_path(path: str) -> Optional[str]:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if p.endswith(".pdf"):
        return "application/pdf"
    return None


def _multimodal_metadata(settings: Any, *, cross_modal: bool = False) -> Dict[str, Any]:
    return {
        "harness": "multimodal",
        "vision_provider": settings.multimodal_vision_provider,
        "embedding_model": settings.rag_embedding_model,
        "rag_generator": settings.rag_generator,
        "cross_modal": cross_modal,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }


def _multimodal_csv_columns() -> List[str]:
    return [
        "image",
        "question",
        "expected_answer",
        "expected_keywords",
        "expected_labels",
        "requires_ocr",
        "language",
        "answer",
        "exact_match",
        "substring_match",
        "keyword_coverage",
        "label_precision",
        "label_recall",
        "latency_ms",
        "ocr_latency_ms",
        "vision_latency_ms",
        "rag_latency_ms",
        "vision_provider",
        "notes",
        "error",
        "skipped_reason",
    ]


def _print_multimodal_summary(summary: MultimodalEvalSummary) -> None:
    print()
    print(f"Multimodal eval - {summary.dataset_path}")
    print(f"  rows in dataset     : {summary.row_count}")
    print(f"  evaluated           : {summary.evaluated_rows}")
    print(f"  skipped             : {summary.skipped_rows}")
    print(f"  errors              : {summary.error_count}")
    print(f"  mean exact_match    : {_fmt(summary.mean_exact_match)}")
    print(f"  mean substring_match: {_fmt(summary.mean_substring_match)}")
    print(f"  mean keyword_cov    : {_fmt(summary.mean_keyword_coverage)}")
    print(f"  mean label_precision: {_fmt(summary.mean_label_precision)}")
    print(f"  mean label_recall   : {_fmt(summary.mean_label_recall)}")
    print(f"  mean latency_ms     : {summary.mean_latency_ms:.1f}")
    print(f"  p50 latency_ms      : {summary.p50_latency_ms:.1f}")
    print(f"  max latency_ms      : {summary.max_latency_ms:.1f}")
    print(f"  mean ocr_ms         : {summary.mean_ocr_latency_ms:.1f}")
    print(f"  mean vision_ms      : {summary.mean_vision_latency_ms:.1f}")
    print(f"  mean rag_ms         : {summary.mean_rag_latency_ms:.1f}")
    print(f"  vision_provider     : {summary.vision_provider}")
    print()


# ---------------------------------------------------------------------------
# Retrieval CLI path.
#
# Builds the same Retriever the production worker uses (or the offline
# in-memory variant when --corpus is passed) and scores it with the new
# `retrieval_eval` harness. Generator is intentionally NOT instantiated:
# this mode measures dense-retrieval baseline quality in isolation.
# ---------------------------------------------------------------------------


def _run_retrieval_cli(args: argparse.Namespace) -> int:
    import json as _json

    try:
        if args.corpus is not None:
            retriever, _, settings, offline_info = _build_offline_retrieval_stack(args)
            corpus_path: Optional[str] = str(args.corpus)
        else:
            retriever, _, settings = _build_rag_stack(args)
            corpus_path = None
            offline_info = None
    except Exception as ex:
        log.error(
            "Failed to build the retrieval eval stack (%s: %s). "
            "For an offline run, pass --corpus <path/to/corpus.jsonl>; "
            "see eval/corpora/<name>/README.md for re-staging instructions.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    top_k = int(args.top_k)
    mrr_k = int(args.mrr_k)
    ndcg_k = int(args.ndcg_k)
    extra_hit_ks = _resolve_extra_hit_ks(
        getattr(args, "extra_hit_k", None), top_k=top_k,
    )

    summary, rows, dump, dup = run_retrieval_eval(
        dataset,
        retriever=retriever,
        top_k=top_k,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
        extra_hit_ks=extra_hit_ks,
        dataset_path=str(args.dataset),
        corpus_path=corpus_path,
    )

    out_dir = args.out_dir or _default_retrieval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) retrieval_eval_report.json — summary + per-row + provenance
    report_payload = {
        "metadata": _retrieval_metadata(
            settings, args, top_k, mrr_k, ndcg_k,
            corpus_path=corpus_path, offline_info=offline_info,
        ),
        "summary": retrieval_summary_to_dict(summary),
        "rows": [retrieval_row_to_dict(r) for r in rows],
    }
    report_json = out_dir / "retrieval_eval_report.json"
    report_json.write_text(
        _json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", report_json)

    # 2) retrieval_eval_report.md — human-readable summary
    report_md = out_dir / "retrieval_eval_report.md"
    report_md.write_text(
        render_markdown_report(summary, rows, dup), encoding="utf-8"
    )
    log.info("Wrote %s", report_md)

    # 3) top_k_dump.jsonl — one record per (query, rank) pair
    dump_path = out_dir / "top_k_dump.jsonl"
    with dump_path.open("w", encoding="utf-8") as fp:
        for d in dump:
            fp.write(_json.dumps(retrieval_dump_row_to_dict(d), ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d rows)", dump_path, len(dump))

    # 4) duplicate_analysis.json — per-query + aggregate
    dup_path = out_dir / "duplicate_analysis.json"
    dup_path.write_text(
        _json.dumps(duplicate_analysis_to_dict(dup), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", dup_path)

    # 5) miss_analysis.{json,md} — 4-bucket cross-tab + capped samples
    rows_for_miss = [retrieval_row_to_dict(r) for r in rows]
    dump_for_miss = [retrieval_dump_row_to_dict(d) for d in dump]
    miss_analysis = classify_miss_buckets(
        rows_for_miss,
        dump_rows=dump_for_miss,
        top_k=top_k,
    )
    miss_json_path = out_dir / "miss_analysis.json"
    miss_json_path.write_text(
        _json.dumps(miss_analysis_to_dict(miss_analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", miss_json_path)
    miss_md_path = out_dir / "miss_analysis.md"
    miss_md_path.write_text(
        render_miss_analysis_markdown(miss_analysis), encoding="utf-8"
    )
    log.info("Wrote %s", miss_md_path)

    try:
        _print_retrieval_summary(summary, dup, out_dir)
    except UnicodeEncodeError as ex:  # pragma: no cover — win cp949 fallback
        log.warning(
            "Pretty summary print failed (%s); reports already written to %s",
            ex, out_dir,
        )
    return 0


def _build_offline_retrieval_stack(args: argparse.Namespace):
    """Build the in-memory retriever from --corpus.

    Re-uses ``eval.harness.offline_corpus.build_offline_rag_stack`` so the
    chunk/embed/index path is byte-identical to what the existing offline
    `rag` mode uses. We discard the generator the helper returns — the
    retrieval mode never invokes it.
    """
    import tempfile
    from pathlib import Path as _P

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.core.config import get_settings

    from eval.harness.offline_corpus import build_offline_rag_stack

    settings = get_settings()
    top_k = int(args.top_k)

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        # Offline namu-wiki dumps include extreme outlier chunks (>100k
        # chars). bge-m3 default max_seq_length is 8192 → attention is
        # O(L^2), so a single outlier in a batch of 32 OOM-thrashes the
        # GPU and blows up wall-clock by orders of magnitude. Cap to
        # 1024 tokens so the same retrieval baseline finishes in ~10
        # minutes instead of 2+ hours; the tail of those rare long
        # chunks is dropped, which doesn't materially change retrieval
        # quality on this corpus. CLI flag wins over the worker
        # setting so per-run experiments can override.
        max_seq_length=getattr(args, "max_seq_length", 1024),
        batch_size=getattr(args, "embed_batch_size", 32),
        show_progress_bar=True,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    tmp_dir = _P(tempfile.mkdtemp(prefix="retrieval-eval-offline-"))
    retriever, generator, info = build_offline_rag_stack(
        _P(args.corpus),
        embedder=embedder,
        index_dir=tmp_dir,
        top_k=top_k,
    )
    return retriever, generator, settings, info


def _retrieval_metadata(
    settings: Any,
    args: argparse.Namespace,
    top_k: int,
    mrr_k: int,
    ndcg_k: int,
    *,
    corpus_path: Optional[str],
    offline_info: Optional[Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "harness": "retrieval",
        "embedding_model": settings.rag_embedding_model,
        "rag_index_dir": str(settings.rag_index_dir),
        "top_k": top_k,
        "mrr_k": mrr_k,
        "ndcg_k": ndcg_k,
        "reranker": getattr(settings, "rag_reranker", None),
        "candidate_k": getattr(settings, "rag_candidate_k", None),
        "dataset": str(args.dataset),
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    if corpus_path:
        payload["corpus_path"] = corpus_path
    if offline_info is not None:
        payload["offline_corpus"] = {
            "path": offline_info.corpus_path,
            "document_count": offline_info.document_count,
            "chunk_count": offline_info.chunk_count,
            "index_version": offline_info.index_version,
            "dimension": offline_info.dimension,
        }
    return payload


def _print_retrieval_summary(
    summary: RetrievalEvalSummary,
    dup: DuplicateAnalysis,
    out_dir: Path,
) -> None:
    print()
    print(f"Retrieval eval - {summary.dataset_path}")
    if summary.corpus_path:
        print(f"  corpus              : {summary.corpus_path}")
    print(f"  rows / errors       : {summary.row_count} / {summary.error_count}")
    print(f"  top_k               : {summary.top_k} (mrr@{summary.mrr_k}, ndcg@{summary.ndcg_k})")
    print(f"  hit@1 / hit@3 / hit@5: "
          f"{_fmt(summary.mean_hit_at_1)} / "
          f"{_fmt(summary.mean_hit_at_3)} / "
          f"{_fmt(summary.mean_hit_at_5)}")
    print(f"  mrr@{summary.mrr_k}              : {_fmt(summary.mean_mrr_at_10)}")
    print(f"  ndcg@{summary.ndcg_k}             : {_fmt(summary.mean_ndcg_at_10)}")
    print(f"  dup_rate (top-{summary.top_k})   : {_fmt(summary.mean_dup_rate)}")
    print(f"  unique_doc_coverage : {_fmt(summary.mean_unique_doc_coverage)}")
    print(f"  top1_score_margin   : {_fmt(summary.mean_top1_score_margin)}")
    print(f"  avg_ctx_tok_count   : {_fmt(summary.mean_avg_context_token_count)}")
    print(f"  expected_kw_match   : {_fmt(summary.mean_expected_keyword_match_rate)}")
    print(f"  retrieval_ms p50/p95/max: "
          f"{summary.p50_retrieval_ms:.1f} / "
          f"{summary.p95_retrieval_ms:.1f} / "
          f"{summary.max_retrieval_ms:.1f}")
    print(f"  embedding model     : {summary.embedding_model}")
    print(f"  index version       : {summary.index_version}")
    print()
    print(f"  duplicates: doc {dup.queries_with_doc_dup_ratio:.3f}  "
          f"section {dup.queries_with_section_dup_ratio:.3f}  "
          f"text {dup.queries_with_text_dup_ratio:.3f}")
    print()
    print(f"  artifacts written to {out_dir}/")
    print()


def _default_retrieval_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/retrieval-{timestamp}")


def _resolve_extra_hit_ks(raw, *, top_k: int):
    """Normalise the --extra-hit-k argparse list into a sorted tuple.

    Values <= 0 are dropped; values > top_k log a warning because the
    metric they produce is degenerate (the harness clamps to the
    top-k slice it actually has). Duplicates are collapsed so the
    summary shape is stable.
    """
    if not raw:
        return ()
    seen: set[int] = set()
    out: list[int] = []
    for value in raw:
        try:
            k = int(value)
        except (TypeError, ValueError):
            continue
        if k <= 0 or k in seen:
            continue
        if k > top_k:
            log.warning(
                "extra-hit-k=%d exceeds top_k=%d; the metric will be "
                "computed over the top_k slice (effective hit@top_k).",
                k, top_k,
            )
        seen.add(k)
        out.append(k)
    out.sort()
    return tuple(out)


# ---------------------------------------------------------------------------
# Phase 2A retrieval-rerank CLI.
# ---------------------------------------------------------------------------


def _run_retrieval_rerank_cli(args: argparse.Namespace) -> int:
    """Run the offline retrieval eval with a cross-encoder reranker.

    Builds the same offline retrieval stack as ``retrieval`` but plugs
    a CrossEncoderReranker into the Retriever with candidate_k =
    --dense-top-n. The reranker truncates the bi-encoder candidates
    down to --final-top-k before the harness scores the top-k.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _P

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.core.config import get_settings
    from eval.harness.offline_corpus import build_offline_rag_stack

    final_top_k = int(args.final_top_k)
    dense_top_n = int(args.dense_top_n)
    if dense_top_n < final_top_k:
        log.error(
            "--dense-top-n (%d) must be >= --final-top-k (%d). The "
            "reranker can only re-order what the bi-encoder hands it; "
            "asking for more output than candidate input would force "
            "the retriever to return fewer than --final-top-k results.",
            dense_top_n, final_top_k,
        )
        return 2

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=True,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    try:
        reranker = CrossEncoderReranker(
            model_name=str(args.reranker_model),
            max_length=int(args.reranker_max_length),
            batch_size=int(args.reranker_batch_size),
            text_max_chars=int(args.reranker_text_max_chars),
            device=args.reranker_device or None,
            oom_fallback_batch_size=(
                int(args.reranker_oom_fallback_batch_size)
                if args.reranker_oom_fallback_batch_size is not None
                else None
            ),
        )
    except Exception as ex:
        log.error(
            "Failed to construct CrossEncoderReranker (%s: %s). "
            "Check that sentence-transformers is installed and that "
            "the requested model name is valid: %s",
            type(ex).__name__, ex, args.reranker_model,
        )
        return 2

    tmp_dir = _P(tempfile.mkdtemp(prefix="retrieval-rerank-offline-"))
    try:
        retriever, _generator, info = build_offline_rag_stack(
            _P(args.corpus),
            embedder=embedder,
            index_dir=tmp_dir,
            top_k=final_top_k,
            reranker=reranker,
            candidate_k=dense_top_n,
        )
    except Exception as ex:
        log.error(
            "Failed to build the rerank retrieval stack (%s: %s). "
            "Verify --corpus exists and has the expected sections "
            "shape; eval/corpora/<dir>/README.md has staging notes.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    extra_hit_ks = _resolve_extra_hit_ks(
        getattr(args, "extra_hit_k", None), top_k=final_top_k,
    )

    summary, rows, dump, dup = run_retrieval_eval(
        dataset,
        retriever=retriever,
        top_k=final_top_k,
        mrr_k=int(args.mrr_k),
        ndcg_k=int(args.ndcg_k),
        extra_hit_ks=extra_hit_ks,
        dataset_path=str(args.dataset),
        corpus_path=str(args.corpus),
    )

    out_dir = args.out_dir or _default_retrieval_rerank_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "harness": "retrieval-rerank",
        "embedding_model": settings.rag_embedding_model,
        "embedding_max_seq_length": int(args.max_seq_length),
        "embedding_batch_size": int(args.embed_batch_size),
        "rag_index_dir": str(settings.rag_index_dir),
        "corpus_path": str(args.corpus),
        "dataset": str(args.dataset),
        "final_top_k": final_top_k,
        "dense_top_n": dense_top_n,
        "candidate_k": dense_top_n,
        "mrr_k": int(args.mrr_k),
        "ndcg_k": int(args.ndcg_k),
        "extra_hit_ks": list(extra_hit_ks),
        "reranker": "cross_encoder",
        "reranker_model": str(args.reranker_model),
        "reranker_batch_size": int(args.reranker_batch_size),
        "reranker_max_length": int(args.reranker_max_length),
        "reranker_text_max_chars": int(args.reranker_text_max_chars),
        "reranker_device": args.reranker_device,
        "reranker_oom_fallback_batch_size": (
            int(args.reranker_oom_fallback_batch_size)
            if args.reranker_oom_fallback_batch_size is not None
            else None
        ),
        "offline_corpus": {
            "path": info.corpus_path,
            "document_count": info.document_count,
            "chunk_count": info.chunk_count,
            "index_version": info.index_version,
            "dimension": info.dimension,
        },
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }

    report_payload = {
        "metadata": metadata,
        "summary": retrieval_summary_to_dict(summary),
        "rows": [retrieval_row_to_dict(r) for r in rows],
    }
    (out_dir / "retrieval_eval_report.json").write_text(
        _json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "retrieval_eval_report.md").write_text(
        render_markdown_report(summary, rows, dup), encoding="utf-8",
    )
    with (out_dir / "top_k_dump.jsonl").open("w", encoding="utf-8") as fp:
        for d in dump:
            fp.write(
                _json.dumps(retrieval_dump_row_to_dict(d), ensure_ascii=False)
                + "\n"
            )
    (out_dir / "duplicate_analysis.json").write_text(
        _json.dumps(duplicate_analysis_to_dict(dup), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows_for_miss = [retrieval_row_to_dict(r) for r in rows]
    dump_for_miss = [retrieval_dump_row_to_dict(d) for d in dump]
    miss_analysis = classify_miss_buckets(
        rows_for_miss, dump_rows=dump_for_miss, top_k=final_top_k,
    )
    (out_dir / "miss_analysis.json").write_text(
        _json.dumps(miss_analysis_to_dict(miss_analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "miss_analysis.md").write_text(
        render_miss_analysis_markdown(miss_analysis), encoding="utf-8",
    )

    try:
        _print_retrieval_summary(summary, dup, out_dir)
    except UnicodeEncodeError as ex:  # pragma: no cover
        log.warning(
            "Pretty summary print failed (%s); reports already in %s",
            ex, out_dir,
        )
    return 0


def _default_retrieval_rerank_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/retrieval-rerank-{timestamp}")


# ---------------------------------------------------------------------------
# Phase 2A-L latency-sweep CLIs.
# ---------------------------------------------------------------------------


_DEFAULT_TOPN_SWEEP = (5, 10, 15, 20, 30, 50)
_DEFAULT_CANDIDATE_RECALL_KS = (10, 20, 50)


def _run_phase2a_latency_sweep_cli(args: argparse.Namespace) -> int:
    """Phase 2A-L: build a corpus once, run N retrieval-rerank passes.

    Each ``dense_top_n`` value gets its own sub-directory under
    ``--out-dir`` carrying a full retrieval_eval_report.json (with
    stage-timing breakdown enabled), the standard top_k_dump /
    duplicate / miss-analysis sidecars, and the per-config
    miss-analysis. The top-level ``--out-dir`` then receives the four
    Phase 2A-L documents:

      - reranker-latency-breakdown.{json,md} (anchored on
        ``--breakdown-anchor-dense-top-n``)
      - topn-sweep.{json,md}
      - accuracy-latency-frontier.{json,md}
      - recommended-modes.md

    Reusing the corpus + index across runs is the meat of the saving
    against running ``retrieval-rerank`` six times by hand: a 47k-chunk
    BGE-M3 build takes minutes whereas the per-config rerank loop is
    seconds (top5) to a handful of minutes (top50).
    """
    import json as _json
    import tempfile
    from pathlib import Path as _P

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import (
        CrossEncoderReranker,
        NoOpReranker,
    )
    from app.core.config import get_settings
    from eval.harness.latency_breakdown import (
        build_latency_breakdown,
        latency_breakdown_to_dict,
        render_latency_breakdown_markdown,
    )
    from eval.harness.offline_corpus import build_offline_rag_stack
    from eval.harness.pareto_frontier import (
        compute_pareto_frontier,
        pareto_to_dict,
        render_pareto_markdown,
    )
    from eval.harness.recommended_modes import (
        recommend_modes,
        recommended_modes_to_dict,
        render_recommended_modes_markdown,
    )
    from eval.harness.topn_sweep import (
        build_topn_sweep,
        render_topn_sweep_markdown,
        topn_sweep_to_dict,
    )

    final_top_k = int(args.final_top_k)
    raw_topns = args.dense_top_n if args.dense_top_n else list(_DEFAULT_TOPN_SWEEP)
    seen: set = set()
    dense_top_ns: List[int] = []
    for v in raw_topns:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv <= 0 or iv in seen:
            continue
        seen.add(iv)
        dense_top_ns.append(iv)
    dense_top_ns.sort()
    if not dense_top_ns:
        log.error("No valid --dense-top-n values supplied.")
        return 2
    # For Phase 2A-L the sweep is allowed to span dense_top_n values
    # smaller than --final-top-k (e.g. 5 vs 10). The retriever can only
    # ever return min(dense_top_n, final_top_k) results, so when
    # ``n < final_top_k`` we silently reduce the per-config final_top_k
    # to ``n``. This is the only semantic that makes a "dense_top_n=5"
    # config meaningfully different from "dense_top_n=10" — without the
    # reduction the retriever would clamp candidate_k back up to
    # final_top_k and the small slice would never actually run.
    for n in dense_top_ns:
        if n < final_top_k:
            log.warning(
                "Phase 2A-L sweep: dense_top_n=%d < final_top_k=%d; "
                "this config will use effective_final_top_k=%d so the "
                "smaller candidate pool actually exercises the "
                "reranker rather than getting clamped up.", n, final_top_k, n,
            )

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=True,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    try:
        reranker = CrossEncoderReranker(
            model_name=str(args.reranker_model),
            max_length=int(args.reranker_max_length),
            batch_size=int(args.reranker_batch_size),
            text_max_chars=int(args.reranker_text_max_chars),
            device=args.reranker_device or None,
            collect_stage_timings=True,
        )
    except Exception as ex:
        log.error(
            "Failed to construct CrossEncoderReranker (%s: %s).",
            type(ex).__name__, ex,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = _P(tempfile.mkdtemp(prefix="phase2a-latency-sweep-offline-"))
    try:
        max_topn = max(dense_top_ns)
        log.info(
            "Building offline corpus with candidate_k=%d (max sweep value); "
            "FAISS index reused across configs.",
            max_topn,
        )
        retriever, _generator, info = build_offline_rag_stack(
            _P(args.corpus),
            embedder=embedder,
            index_dir=tmp_dir,
            top_k=final_top_k,
            reranker=reranker,
            candidate_k=max_topn,
        )
    except Exception as ex:
        log.error(
            "Failed to build the rerank retrieval stack (%s: %s).",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)

    sweep_slices: List[Tuple[str, Path]] = []
    breakdown_anchor_label: Optional[str] = None
    breakdown_anchor_path: Optional[Path] = None

    breakdown_anchor_n = int(args.breakdown_anchor_dense_top_n)

    # Per-config retrieval-rerank runs. We swap candidate_k on the live
    # Retriever between calls — the underlying FAISS index + embedder
    # is the same throughout.
    for n in dense_top_ns:
        label = f"top{n}"
        run_dir = out_dir / f"rerank-{label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # When dense_top_n < final_top_k we reduce the per-config top_k
        # to ``n`` so the candidate pool is actually small. The Retriever
        # constructor would otherwise clamp candidate_k back up to top_k
        # and erase the difference between dense_top_n=5 and
        # dense_top_n=10. Mutating ``_top_k`` + ``_candidate_k`` directly
        # is intentionally tightly-coupled — these are owned attributes
        # of the Retriever and the sweep is the lone caller.
        effective_top_k = min(final_top_k, n)
        retriever._top_k = effective_top_k  # noqa: SLF001 — owned by us
        retriever._candidate_k = max(effective_top_k, n)  # noqa: SLF001

        log.info(
            "Phase 2A-L sweep: dense_top_n=%d, final_top_k=%d "
            "(effective_top_k=%d), out=%s",
            n, final_top_k, effective_top_k, run_dir,
        )
        summary, rows, dump, dup = run_retrieval_eval(
            dataset,
            retriever=retriever,
            top_k=effective_top_k,
            mrr_k=int(args.mrr_k),
            ndcg_k=int(args.ndcg_k),
            extra_hit_ks=(),
            dataset_path=str(args.dataset),
            corpus_path=str(args.corpus),
        )

        metadata: Dict[str, Any] = {
            "harness": "phase2a-latency-sweep",
            "embedding_model": settings.rag_embedding_model,
            "embedding_max_seq_length": int(args.max_seq_length),
            "embedding_batch_size": int(args.embed_batch_size),
            "rag_index_dir": str(settings.rag_index_dir),
            "corpus_path": str(args.corpus),
            "dataset": str(args.dataset),
            "final_top_k": effective_top_k,
            "requested_final_top_k": final_top_k,
            "dense_top_n": n,
            "candidate_k": n,
            "mrr_k": int(args.mrr_k),
            "ndcg_k": int(args.ndcg_k),
            "extra_hit_ks": [],
            "reranker": "cross_encoder",
            "reranker_model": str(args.reranker_model),
            "reranker_batch_size": int(args.reranker_batch_size),
            "reranker_max_length": int(args.reranker_max_length),
            "reranker_text_max_chars": int(args.reranker_text_max_chars),
            "reranker_device": args.reranker_device,
            "collect_stage_timings": True,
            "offline_corpus": {
                "path": info.corpus_path,
                "document_count": info.document_count,
                "chunk_count": info.chunk_count,
                "index_version": info.index_version,
                "dimension": info.dimension,
            },
            "run_at": datetime.now().isoformat(timespec="seconds"),
        }
        report_payload = {
            "metadata": metadata,
            "summary": retrieval_summary_to_dict(summary),
            "rows": [retrieval_row_to_dict(r) for r in rows],
        }
        report_path = run_dir / "retrieval_eval_report.json"
        report_path.write_text(
            _json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "retrieval_eval_report.md").write_text(
            render_markdown_report(summary, rows, dup), encoding="utf-8",
        )
        with (run_dir / "top_k_dump.jsonl").open("w", encoding="utf-8") as fp:
            for d in dump:
                fp.write(
                    _json.dumps(
                        retrieval_dump_row_to_dict(d), ensure_ascii=False,
                    ) + "\n"
                )
        (run_dir / "duplicate_analysis.json").write_text(
            _json.dumps(
                duplicate_analysis_to_dict(dup), ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        sweep_slices.append((label, report_path))
        if n == breakdown_anchor_n:
            breakdown_anchor_label = label
            breakdown_anchor_path = report_path

    # Optional candidate-recall sibling. Uses a NoOpReranker so the
    # retrieval ordering is bi-encoder-only; the dataset rows pick up
    # mean_extra_hits via ``extra_hit_ks`` so the sweep aggregator can
    # quote candidate_recall@N on every entry.
    candidate_recall_report_path: Optional[Path] = None
    if not args.skip_candidate_recall:
        cr_ks_raw = args.candidate_recall_extra_hit_k
        if cr_ks_raw:
            cr_ks_seen: set = set()
            cr_ks: List[int] = []
            for v in cr_ks_raw:
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv <= 0 or iv in cr_ks_seen:
                    continue
                cr_ks_seen.add(iv)
                cr_ks.append(iv)
            cr_ks.sort()
        else:
            cr_ks = list(_DEFAULT_CANDIDATE_RECALL_KS)
        cr_topn = max(cr_ks) if cr_ks else max(dense_top_ns)
        cr_dir = out_dir / "candidate-recall"
        cr_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Phase 2A-L candidate-recall sibling: dense_top_n=%d, "
            "extra_hit_ks=%s", cr_topn, cr_ks,
        )
        retriever._reranker = NoOpReranker()  # noqa: SLF001
        retriever._top_k = cr_topn  # noqa: SLF001
        retriever._candidate_k = cr_topn  # noqa: SLF001

        cr_summary, cr_rows, cr_dump, cr_dup = run_retrieval_eval(
            dataset,
            retriever=retriever,
            top_k=cr_topn,
            mrr_k=int(args.mrr_k),
            ndcg_k=int(args.ndcg_k),
            extra_hit_ks=tuple(cr_ks),
            dataset_path=str(args.dataset),
            corpus_path=str(args.corpus),
        )
        cr_metadata = {
            "harness": "phase2a-latency-sweep:candidate-recall",
            "embedding_model": settings.rag_embedding_model,
            "corpus_path": str(args.corpus),
            "dataset": str(args.dataset),
            "final_top_k": cr_topn,
            "dense_top_n": cr_topn,
            "candidate_k": cr_topn,
            "extra_hit_ks": list(cr_ks),
            "reranker": "noop",
            "offline_corpus": {
                "path": info.corpus_path,
                "document_count": info.document_count,
                "chunk_count": info.chunk_count,
                "index_version": info.index_version,
                "dimension": info.dimension,
            },
            "run_at": datetime.now().isoformat(timespec="seconds"),
        }
        cr_payload = {
            "metadata": cr_metadata,
            "summary": retrieval_summary_to_dict(cr_summary),
            "rows": [retrieval_row_to_dict(r) for r in cr_rows],
        }
        candidate_recall_report_path = cr_dir / "retrieval_eval_report.json"
        candidate_recall_report_path.write_text(
            _json.dumps(cr_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (cr_dir / "retrieval_eval_report.md").write_text(
            render_markdown_report(cr_summary, cr_rows, cr_dup),
            encoding="utf-8",
        )

    # 1) reranker-latency-breakdown anchored on the requested topN.
    if breakdown_anchor_path is None:
        breakdown_anchor_label = sweep_slices[0][0]
        breakdown_anchor_path = sweep_slices[0][1]
        log.warning(
            "breakdown_anchor_dense_top_n=%d not in sweep; falling "
            "back to %s.", breakdown_anchor_n, breakdown_anchor_label,
        )
    breakdown = build_latency_breakdown(
        breakdown_anchor_path, label=breakdown_anchor_label,
    )
    (out_dir / "reranker-latency-breakdown.json").write_text(
        _json.dumps(
            latency_breakdown_to_dict(breakdown),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "reranker-latency-breakdown.md").write_text(
        render_latency_breakdown_markdown(breakdown), encoding="utf-8",
    )

    # 2) topn-sweep.
    sweep_report = build_topn_sweep(
        sweep_slices, candidate_recall_path=candidate_recall_report_path,
    )
    (out_dir / "topn-sweep.json").write_text(
        _json.dumps(
            topn_sweep_to_dict(sweep_report), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "topn-sweep.md").write_text(
        render_topn_sweep_markdown(sweep_report), encoding="utf-8",
    )

    # 3) Pareto frontier.
    frontier_report = compute_pareto_frontier(
        sweep_report,
        metric=str(args.metric),
        latency=str(args.latency),
    )
    (out_dir / "accuracy-latency-frontier.json").write_text(
        _json.dumps(
            pareto_to_dict(frontier_report), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "accuracy-latency-frontier.md").write_text(
        render_pareto_markdown(frontier_report), encoding="utf-8",
    )

    # 4) Recommended modes.
    modes_report = recommend_modes(
        sweep_report, frontier_report,
        fast_p95_budget_ms=args.fast_p95_budget_ms,
        balanced_p95_budget_ms=args.balanced_p95_budget_ms,
        quality_target_metric=args.quality_target_metric,
    )
    (out_dir / "recommended-modes.md").write_text(
        render_recommended_modes_markdown(modes_report),
        encoding="utf-8",
    )
    (out_dir / "recommended-modes.json").write_text(
        _json.dumps(
            recommended_modes_to_dict(modes_report),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"phase2a-latency-sweep: {len(sweep_slices)} configs swept; "
        f"breakdown anchor={breakdown_anchor_label}, "
        f"out_dir={out_dir}"
    )
    return 0


def _run_phase2a_latency_breakdown_cli(args: argparse.Namespace) -> int:
    """Post-process: emit a stage-level latency breakdown for one report."""
    import json as _json

    from eval.harness.latency_breakdown import (
        build_latency_breakdown,
        latency_breakdown_to_dict,
        render_latency_breakdown_markdown,
    )

    try:
        report = build_latency_breakdown(
            Path(args.report), label=args.label,
        )
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        _json.dumps(
            latency_breakdown_to_dict(report), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    args.out_md.write_text(
        render_latency_breakdown_markdown(report), encoding="utf-8",
    )
    log.info("Wrote %s and %s", args.out_json, args.out_md)
    return 0


def _run_phase2a_topn_sweep_cli(args: argparse.Namespace) -> int:
    import json as _json

    from eval.harness.topn_sweep import (
        build_topn_sweep,
        render_topn_sweep_markdown,
        topn_sweep_to_dict,
    )

    slices: List[Tuple[str, Path]] = []
    for entry in args.slice:
        label, _, path = str(entry).partition(":")
        if not label or not path:
            log.error(
                "Bad --slice value %r; expected 'label:path/to/"
                "retrieval_eval_report.json'.",
                entry,
            )
            return 2
        slices.append((label.strip(), Path(path.strip())))

    try:
        report = build_topn_sweep(
            slices,
            candidate_recall_path=args.candidate_recall_report,
            caveats=args.caveat,
        )
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        _json.dumps(topn_sweep_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out_md.write_text(
        render_topn_sweep_markdown(report), encoding="utf-8",
    )
    log.info("Wrote %s and %s", args.out_json, args.out_md)
    return 0


def _run_phase2a_recommended_modes_cli(args: argparse.Namespace) -> int:
    import json as _json

    from eval.harness.pareto_frontier import (
        compute_pareto_frontier,
        pareto_to_dict,
        render_pareto_markdown,
    )
    from eval.harness.recommended_modes import (
        recommend_modes,
        recommended_modes_to_dict,
        render_recommended_modes_markdown,
    )
    from eval.harness.topn_sweep import (
        TopNSweepEntry,
        TopNSweepReport,
    )

    sweep_path = Path(args.sweep_json)
    if not sweep_path.exists():
        log.error("topn-sweep.json not found at %s", sweep_path)
        return 2
    payload = _json.loads(sweep_path.read_text(encoding="utf-8"))

    raw_entries = payload.get("entries") or []
    entries: List[TopNSweepEntry] = []
    for raw in raw_entries:
        # Re-hydrate via field-by-field construction so a stale on-disk
        # schema with extra fields doesn't crash the dataclass init —
        # we only pull the fields the dataclass declares.
        kwargs = {
            f.name: raw.get(f.name)
            for f in TopNSweepEntry.__dataclass_fields__.values()
        }
        # Nested dicts default to {}; the dataclass annotates them
        # with field(default_factory=dict).
        if kwargs.get("candidate_recall") is None:
            kwargs["candidate_recall"] = {}
        if kwargs.get("rerank_row_count") is None:
            kwargs["rerank_row_count"] = 0
        if kwargs.get("total_query_row_count") is None:
            kwargs["total_query_row_count"] = 0
        if kwargs.get("dense_retrieval_row_count") is None:
            kwargs["dense_retrieval_row_count"] = 0
        try:
            entries.append(TopNSweepEntry(**kwargs))
        except TypeError as ex:
            log.error(
                "topn-sweep.json entry incompatible with current "
                "TopNSweepEntry schema (%s). Field set on disk: %s",
                ex, sorted(raw.keys()),
            )
            return 2

    sweep_report = TopNSweepReport(
        schema=payload.get("schema") or "phase2a-topn-sweep.v1",
        entries=entries,
        caveats=list(payload.get("caveats") or []),
    )
    frontier = compute_pareto_frontier(
        sweep_report,
        metric=str(args.metric),
        latency=str(args.latency),
    )
    modes = recommend_modes(
        sweep_report, frontier,
        fast_p95_budget_ms=args.fast_p95_budget_ms,
        balanced_p95_budget_ms=args.balanced_p95_budget_ms,
        quality_target_metric=args.quality_target_metric,
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        render_recommended_modes_markdown(modes), encoding="utf-8",
    )
    if args.out_modes_json:
        args.out_modes_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_modes_json.write_text(
            _json.dumps(
                recommended_modes_to_dict(modes),
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    if args.out_frontier_json:
        args.out_frontier_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_frontier_json.write_text(
            _json.dumps(
                pareto_to_dict(frontier), ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    if args.out_frontier_md:
        args.out_frontier_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_frontier_md.write_text(
            render_pareto_markdown(frontier), encoding="utf-8",
        )

    log.info("Wrote %s", args.out_md)
    return 0


# ---------------------------------------------------------------------------
# Phase 2A reranker comparison + failure-analysis CLIs (post-processing).
# ---------------------------------------------------------------------------


def _run_phase2a_reranker_comparison_cli(args: argparse.Namespace) -> int:
    import json as _json

    from eval.harness.reranker_eval import (
        build_reranker_comparison,
        render_reranker_comparison_markdown,
    )

    slices: List[Tuple[str, Path]] = []
    for entry in args.slice:
        label, _, path = str(entry).partition(":")
        if not label or not path:
            log.error(
                "Bad --slice value %r; expected 'label:path/to/"
                "retrieval_eval_report.json'.",
                entry,
            )
            return 2
        slices.append((label.strip(), Path(path.strip())))

    try:
        comparison = build_reranker_comparison(slices, caveats=args.caveat)
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        _json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out_md.write_text(
        render_reranker_comparison_markdown(comparison),
        encoding="utf-8",
    )
    log.info("Wrote %s and %s", args.out_json, args.out_md)
    return 0


def _run_phase2a_reranker_failure_cli(args: argparse.Namespace) -> int:
    import json as _json

    from eval.harness.reranker_eval import (
        build_reranker_failure_analysis,
        render_reranker_failure_markdown,
    )

    dense_dir = Path(args.dense_report_dir)
    rerank_dir = Path(args.rerank_report_dir)

    dense_report_path = dense_dir / "retrieval_eval_report.json"
    rerank_report_path = rerank_dir / "retrieval_eval_report.json"
    dense_dump_path = dense_dir / "top_k_dump.jsonl"
    rerank_dump_path = rerank_dir / "top_k_dump.jsonl"
    for p in (dense_report_path, rerank_report_path, dense_dump_path, rerank_dump_path):
        if not p.exists():
            log.error("Missing artifact: %s", p)
            return 2

    dense_report = _json.loads(dense_report_path.read_text(encoding="utf-8"))
    rerank_report = _json.loads(rerank_report_path.read_text(encoding="utf-8"))

    def _read_dump(path: Path) -> List[Mapping[str, Any]]:
        rows: List[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rows.append(_json.loads(line))
        return rows

    analysis = build_reranker_failure_analysis(
        dense_rows=list(dense_report.get("rows") or []),
        rerank_rows=list(rerank_report.get("rows") or []),
        dense_dump=_read_dump(dense_dump_path),
        rerank_dump=_read_dump(rerank_dump_path),
        k_preview=int(args.k_preview),
        sample_cap=int(args.sample_cap),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "reranker-failure-analysis.json"
    out_md = out_dir / "reranker-failure-analysis.md"
    out_json.write_text(
        _json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(
        render_reranker_failure_markdown(analysis), encoding="utf-8",
    )
    log.info("Wrote %s and %s", out_json, out_md)
    return 0


# ---------------------------------------------------------------------------
# Retrieval comparison CLI path.
#
# Pure post-processing — reads two ``retrieval_eval_report.json`` files
# and emits a compare report with three slices: deterministic_all,
# deterministic_without_<excluded_answer_type>, and opus_all. Intended
# for the Phase-0 pre-improvement baseline pin so we can reason about
# what the dense-only retriever actually does on each dataset before
# touching MMR / reranker / hybrid.
# ---------------------------------------------------------------------------


def _run_retrieval_compare_cli(args: argparse.Namespace) -> int:
    import json as _json

    det_payload = _json.loads(args.deterministic_report.read_text(encoding="utf-8"))
    opus_payload = _json.loads(args.opus_report.read_text(encoding="utf-8"))

    det_rows = list(det_payload.get("rows") or [])
    opus_rows = list(opus_payload.get("rows") or [])
    det_dataset = (det_payload.get("summary") or {}).get("dataset_path")
    opus_dataset = (opus_payload.get("summary") or {}).get("dataset_path")

    det_meta = det_payload.get("metadata") or {}
    opus_meta = opus_payload.get("metadata") or {}

    det_cfg = {
        "embedding_model": det_meta.get("embedding_model", "BAAI/bge-m3"),
        "max_seq_length": int(args.deterministic_max_seq_length),
        "reranker": det_meta.get("reranker", "off"),
        "candidate_k": det_meta.get("candidate_k"),
        "top_k": det_meta.get("top_k"),
    }
    opus_cfg = {
        "embedding_model": opus_meta.get("embedding_model", "BAAI/bge-m3"),
        "max_seq_length": int(args.opus_max_seq_length),
        "reranker": opus_meta.get("reranker", "off"),
        "candidate_k": opus_meta.get("candidate_k"),
        "top_k": opus_meta.get("top_k"),
    }

    if args.caveat is not None:
        # Strip empty strings the user might have passed accidentally
        # (e.g. via shell glob); empty bullets render as blank lines.
        caveats = [c for c in args.caveat if c.strip()]
    else:
        caveats = _default_compare_caveats(
            det_cfg=det_cfg,
            opus_cfg=opus_cfg,
            excluded_answer_type=args.exclude_answer_type,
            deterministic_kind=args.deterministic_kind,
            opus_kind=args.opus_kind,
        )

    comparison = run_comparison(
        deterministic_rows=det_rows,
        deterministic_dataset_path=det_dataset,
        opus_rows=opus_rows,
        opus_dataset_path=opus_dataset,
        excluded_answer_type=args.exclude_answer_type,
        deterministic_retriever_config=det_cfg,
        opus_retriever_config=opus_cfg,
        caveats=caveats,
        deterministic_kind=args.deterministic_kind,
        opus_kind=args.opus_kind,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "harness": "retrieval-compare",
            "deterministic_report": str(args.deterministic_report),
            "opus_report": str(args.opus_report),
            "excluded_answer_type": args.exclude_answer_type,
            "deterministic_retriever_config": det_cfg,
            "deterministic_kind": args.deterministic_kind,
            "opus_retriever_config": opus_cfg,
            "opus_kind": args.opus_kind,
            "caveats": caveats,
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "comparison": comparison_to_dict(comparison),
    }
    args.out_json.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.out_md.write_text(
        render_comparison_markdown(comparison), encoding="utf-8"
    )
    log.info("Wrote %s", args.out_json)
    log.info("Wrote %s", args.out_md)
    return 0


def _default_compare_caveats(
    *,
    det_cfg: Mapping[str, Any],
    opus_cfg: Mapping[str, Any],
    excluded_answer_type: str,
    deterministic_kind: str = "baseline",
    opus_kind: str = "baseline",
) -> List[str]:
    """Default caveats block when --caveat is not passed.

    Always emits the apples-to-apples warning + the diagnostic-slice
    reminder. Adds a max_seq_length-difference flag if the two slices
    were embedded under different caps (the common Phase 0 case:
    deterministic at 8192, opus at 1024). When either side is marked
    ``tuned``, prepends a strong separation flag so reviewers cannot
    quote tuned numbers as if they were baselines.
    """
    caveats: List[str] = []
    tuned_sides = [
        name
        for name, kind in (
            ("deterministic", deterministic_kind),
            ("opus", opus_kind),
        )
        if kind == "tuned"
    ]
    if tuned_sides:
        caveats.append(
            "**Tuned variant present — not a baseline report.** The "
            f"following slice(s) were run with hyperparameter-modified "
            f"retriever configs: {', '.join(tuned_sides)}. Tuned "
            "numbers are rendered in their own headline-metrics table "
            "and must NOT be quoted as, table-joined against, or "
            "subtracted from baseline numbers. See the per-slice "
            "`retriever:` lines for the exact config delta."
        )
    caveats.append(
        "**Not a strict apples-to-apples comparison.** The two slices "
        "use different query sources (deterministic generator vs Opus) "
        "and may use different retriever configs (see per-slice "
        "`retriever:` lines below). Compare slice-by-slice; do not "
        "subtract aggregate metrics."
    )
    diag_prefix = (
        "deterministic" if deterministic_kind == "baseline" else "tuned"
    )
    caveats.append(
        f"`{diag_prefix}_without_{excluded_answer_type}` is a "
        "diagnostic ceiling slice, not an official headline number. "
        "Quote the headline-metrics tables in the matching section; "
        "use the diagnostic slice only to estimate the retriever's "
        "ceiling on well-formed queries."
    )
    det_msl = det_cfg.get("max_seq_length")
    opus_msl = opus_cfg.get("max_seq_length")
    if det_msl != opus_msl and det_msl is not None and opus_msl is not None:
        caveats.append(
            f"`max_seq_length` differs across slices "
            f"(deterministic={det_msl}, opus={opus_msl}). Long chunks "
            "are truncated more aggressively in the lower-cap slice, "
            "which slightly favors the higher-cap slice on queries "
            "whose gold content lives past the cap. See "
            "`eval/reports/phase1/length_analysis.md` for the measured "
            "fraction of chunks above each cap."
        )
    return caveats


# ---------------------------------------------------------------------------
# Retrieval miss-analysis CLI path.
#
# Pure post-processing — reads ``retrieval_eval_report.json`` +
# ``top_k_dump.jsonl`` from a previously-generated report dir and emits
# the miss_analysis pair into the same dir. Used to add the new
# artifact to historical baselines without paying the embedding cost
# of re-running retrieval.
# ---------------------------------------------------------------------------


def _run_miss_analysis_cli(args: argparse.Namespace) -> int:
    import json as _json

    report_dir: Path = args.report_dir
    if not report_dir.is_dir():
        log.error("--report-dir must be an existing directory: %s", report_dir)
        return 2

    report_json = report_dir / "retrieval_eval_report.json"
    dump_path = report_dir / "top_k_dump.jsonl"
    if not report_json.exists():
        log.error("Missing retrieval_eval_report.json in %s", report_dir)
        return 2
    if not dump_path.exists():
        log.error("Missing top_k_dump.jsonl in %s", report_dir)
        return 2

    payload = _json.loads(report_json.read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])

    dumps: List[Dict[str, Any]] = []
    with dump_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            dumps.append(_json.loads(line))

    miss_analysis = classify_miss_buckets(
        rows,
        dump_rows=dumps,
        top_k=int(args.top_k),
        sample_limit=int(args.sample_limit),
    )
    out_json = report_dir / "miss_analysis.json"
    out_md = report_dir / "miss_analysis.md"
    out_json.write_text(
        _json.dumps(miss_analysis_to_dict(miss_analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(render_miss_analysis_markdown(miss_analysis), encoding="utf-8")
    log.info("Wrote %s", out_json)
    log.info("Wrote %s", out_md)
    print(
        f"miss-analysis written for {report_dir.name}: "
        f"rows_evaluated={miss_analysis.rows_evaluated}, "
        f"buckets="
        + ", ".join(
            f"{b.name}={b.count}" for b in miss_analysis.buckets
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Corpus length-analyzer CLI path.
#
# Pure offline analysis — reads a corpus.jsonl, runs the production
# chunker over it, tokenizes each chunk with bge-m3 (or whatever the
# user passed via --tokenizer), and writes the char/token length
# distribution + max_seq_length cap impact + top-N longest chunks. Used
# to size max_seq_length truncation caveats with measured numbers
# instead of char-derived guesses.
# ---------------------------------------------------------------------------


def _run_analyze_corpus_lengths_cli(args: argparse.Namespace) -> int:
    import json as _json

    thresholds = (
        tuple(args.threshold)
        if args.threshold
        else DEFAULT_TOKEN_THRESHOLDS
    )

    try:
        analysis = analyze_corpus_lengths(
            args.corpus,
            tokenizer_name=args.tokenizer,
            thresholds=thresholds,
            top_longest=int(args.top_longest),
            batch_size=int(args.batch_size),
        )
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2
    except RuntimeError as ex:
        log.error("Analyzer failed: %s", ex)
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "harness": "analyze-corpus-lengths",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "thresholds": list(thresholds),
            "top_longest": int(args.top_longest),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "analysis": length_analysis_to_dict(analysis),
    }
    args.out_json.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out_md.write_text(
        render_length_analysis_markdown(analysis), encoding="utf-8"
    )
    log.info("Wrote %s", args.out_json)
    log.info("Wrote %s", args.out_md)
    print(
        f"corpus-length-analysis: chunks={analysis.chunk_count} "
        f"docs={analysis.document_count} "
        f"tokenizer={analysis.tokenizer} "
        f"token_p95={analysis.token_length.p95:.0f} "
        f"token_max={analysis.token_length.max} "
        + ", ".join(
            f">{t}={analysis.chunks_over_token_threshold[t]}"
            for t in sorted(analysis.chunks_over_token_threshold)
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase 1A — corpus-noise audit + cleaner dry-run CLI paths.
#
# Both commands are pure offline analysis. The audit emits the long-chunk
# top-N + per-chunk noise signals + raw-vs-cleaned length comparison so
# we can reason about the long tail before deciding whether the cleaner
# is worth shipping. The dry-run emits the cleaner-effect summary alone
# and is intended for quick iteration on cleaner pattern changes. Phase 0
# baselines under eval/reports/ are never overwritten — these commands
# write into eval/reports/phase1/1a_corpus_audit/.
# ---------------------------------------------------------------------------


def _run_audit_corpus_noise_cli(args: argparse.Namespace) -> int:
    import json as _json

    thresholds = (
        tuple(args.threshold)
        if args.threshold
        else DEFAULT_TOKEN_THRESHOLDS
    )

    try:
        audit = audit_long_chunks(
            args.corpus,
            tokenizer_name=args.tokenizer,
            top_n=int(args.top_n),
            batch_size=int(args.batch_size),
        )
        comparison = compare_raw_vs_cleaned(
            args.corpus,
            tokenizer_name=args.tokenizer,
            thresholds=thresholds,
            batch_size=int(args.batch_size),
        )
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2
    except RuntimeError as ex:
        log.error("Audit failed: %s", ex)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_payload = {
        "metadata": {
            "harness": "audit-corpus-noise",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "top_n": int(args.top_n),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "audit": audit_to_dict(audit),
    }
    comparison_payload = {
        "metadata": {
            "harness": "audit-corpus-noise:length-comparison",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "thresholds": list(thresholds),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "comparison": length_comparison_to_dict(comparison),
    }

    (out_dir / "long-chunk-audit.json").write_text(
        _json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "long-chunk-audit.md").write_text(
        render_audit_markdown(audit), encoding="utf-8"
    )
    (out_dir / "length-comparison.json").write_text(
        _json.dumps(comparison_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "length-comparison.md").write_text(
        render_length_comparison_markdown(comparison), encoding="utf-8"
    )

    log.info("Wrote %s", out_dir / "long-chunk-audit.json")
    log.info("Wrote %s", out_dir / "long-chunk-audit.md")
    log.info("Wrote %s", out_dir / "length-comparison.json")
    log.info("Wrote %s", out_dir / "length-comparison.md")

    print(
        f"audit-corpus-noise: chunks={audit.chunk_count} "
        f"docs={audit.document_count} "
        f"top_n={len(audit.long_chunks)} "
        f"raw_p95={comparison.raw.token.p95:.0f} "
        f"cleaned_p95={comparison.cleaned.token.p95:.0f} "
        f"dropped={comparison.dropped_chunk_count} "
        f"signals=" + (
            ",".join(
                f"{name}={count}"
                for name, count in sorted(audit.noise_signal_summary.items())
            )
            or "none"
        )
    )
    return 0


def _run_clean_corpus_dry_run_cli(args: argparse.Namespace) -> int:
    import json as _json

    thresholds = (
        tuple(args.threshold)
        if args.threshold
        else DEFAULT_TOKEN_THRESHOLDS
    )

    try:
        comparison = compare_raw_vs_cleaned(
            args.corpus,
            tokenizer_name=args.tokenizer,
            thresholds=thresholds,
            batch_size=int(args.batch_size),
        )
    except FileNotFoundError as ex:
        log.error("%s", ex)
        return 2
    except RuntimeError as ex:
        log.error("Dry-run failed: %s", ex)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "harness": "clean-corpus-dry-run",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "thresholds": list(thresholds),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "comparison": length_comparison_to_dict(comparison),
    }

    (out_dir / "clean-dry-run-summary.json").write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "clean-dry-run-summary.md").write_text(
        render_length_comparison_markdown(comparison), encoding="utf-8"
    )

    log.info("Wrote %s", out_dir / "clean-dry-run-summary.json")
    log.info("Wrote %s", out_dir / "clean-dry-run-summary.md")

    print(
        f"clean-corpus-dry-run: raw_chunks={comparison.raw_chunk_count} "
        f"cleaned_chunks={comparison.cleaned_chunk_count} "
        f"dropped={comparison.dropped_chunk_count} "
        f"removed_lines={comparison.cleaner_total_removed_lines} "
        f"collapsed_repeats={comparison.cleaner_total_collapsed_repeats} "
        f"raw_p95={comparison.raw.token.p95:.0f} "
        f"cleaned_p95={comparison.cleaned.token.p95:.0f}"
    )
    return 0


# ---------------------------------------------------------------------------
# Phase 1B — ingest-side preprocessor CLI paths.
#
# preprocess-corpus-dry-run:  streams the corpus through the prefix /
#   inline-edit transforms and writes a summary + sample diffs to
#   eval/reports/phase1/1b_preprocess/. No corpus artifact is produced.
#
# emit-preprocessed-corpus:   same pass, but writes a
#   corpus.<variant>.jsonl + manifest.json into the configured
#   corpora directory. The source corpus.jsonl is never modified.
#
# Both refuse to run with both transforms disabled, since "raw" is
# already on disk; pass at least one of --strip-page-prefix /
# --strip-inline-edit.
# ---------------------------------------------------------------------------


def _build_preprocess_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        strip_page_prefix=bool(args.strip_page_prefix),
        strip_inline_edit=bool(args.strip_inline_edit),
    )


def _refuse_if_no_transform(config: PreprocessConfig) -> Optional[int]:
    if not (config.strip_page_prefix or config.strip_inline_edit):
        log.error(
            "Refusing to run with no transform enabled — pass at least "
            "one of --strip-page-prefix / --strip-inline-edit. The "
            "'raw' variant is already the source corpus."
        )
        return 2
    return None


def _run_preprocess_corpus_dry_run_cli(args: argparse.Namespace) -> int:
    import json as _json

    config = _build_preprocess_config(args)
    rc = _refuse_if_no_transform(config)
    if rc is not None:
        return rc

    if not args.corpus.exists():
        log.error("Corpus not found: %s", args.corpus)
        return 2

    summary = CorpusPreprocessSummary(
        source_corpus=str(args.corpus), config=config,
    )

    # Drain the iterator so the summary is fully populated. We don't
    # write the preprocessed docs anywhere — this is the dry-run.
    consumed = 0
    for _ in iter_preprocessed_documents(
        _iter_documents(args.corpus),
        config=config,
        sample_diff_target=int(args.sample_diff_n),
        summary=summary,
    ):
        consumed += 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "metadata": {
            "harness": "preprocess-corpus-dry-run",
            "preprocess_version": PREPROCESS_VERSION,
            "corpus": str(args.corpus),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": corpus_preprocess_summary_to_dict(summary),
    }
    (out_dir / "preprocess-summary.json").write_text(
        _json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "preprocess-summary.md").write_text(
        render_corpus_preprocess_summary_markdown(summary),
        encoding="utf-8",
    )
    (out_dir / "preprocess-sample-diffs.md").write_text(
        render_sample_diff_markdown(summary.sample_diffs),
        encoding="utf-8",
    )

    log.info("Wrote %s", out_dir / "preprocess-summary.json")
    log.info("Wrote %s", out_dir / "preprocess-summary.md")
    log.info("Wrote %s", out_dir / "preprocess-sample-diffs.md")

    print(
        f"preprocess-corpus-dry-run: variant={config.variant_label} "
        f"docs={summary.document_count} "
        f"chunks_processed={summary.chunks_processed} "
        f"chunks_changed={summary.chunks_changed} "
        f"chunks_dropped={summary.chunks_dropped} "
        f"prefix_strips={summary.prefix_strip_count} "
        f"inline_edits_removed={summary.total_inline_edit_removals} "
        f"prefix_chars_removed={summary.total_removed_prefix_chars} "
        f"sample_diffs={len(summary.sample_diffs)}"
    )
    return 0


def _run_emit_preprocessed_corpus_cli(args: argparse.Namespace) -> int:
    import json as _json

    config = _build_preprocess_config(args)
    rc = _refuse_if_no_transform(config)
    if rc is not None:
        return rc

    if not args.corpus.exists():
        log.error("Corpus not found: %s", args.corpus)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = args.variant_name or config.variant_label
    out_corpus = out_dir / f"corpus.{variant}.jsonl"
    if out_corpus.resolve() == args.corpus.resolve():
        log.error(
            "Refusing to overwrite the source corpus %s — choose a "
            "different --out-dir or --variant-name.",
            args.corpus,
        )
        return 2

    summary = CorpusPreprocessSummary(
        source_corpus=str(args.corpus), config=config,
    )

    written = 0
    with out_corpus.open("w", encoding="utf-8") as fp:
        for new_doc in iter_preprocessed_documents(
            _iter_documents(args.corpus),
            config=config,
            sample_diff_target=20,
            summary=summary,
        ):
            fp.write(_json.dumps(new_doc, ensure_ascii=False))
            fp.write("\n")
            written += 1

    manifest = {
        "preprocess_version": PREPROCESS_VERSION,
        "source_corpus": str(args.corpus),
        "variant": variant,
        "options": {
            "strip_page_prefix": config.strip_page_prefix,
            "strip_inline_edit": config.strip_inline_edit,
        },
        "doc_count": summary.document_count,
        "sections_processed": summary.sections_processed,
        "chunks_processed": summary.chunks_processed,
        "chunks_changed": summary.chunks_changed,
        "chunks_dropped": summary.chunks_dropped,
        "text_blobs_changed": summary.text_blobs_changed,
        "list_entries_changed": summary.list_entries_changed,
        "total_removed_prefix_chars": summary.total_removed_prefix_chars,
        "total_inline_edit_removals": summary.total_inline_edit_removals,
        "prefix_strip_count": summary.prefix_strip_count,
        "output_corpus": str(out_corpus),
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / "manifest.json"
    # If a previous variant left a manifest behind, merge by variant
    # label so both runs survive in the directory. The manifest is a
    # dict-of-variants when more than one is present.
    if manifest_path.exists():
        try:
            existing = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict) and "variants" in existing:
            existing["variants"][variant] = manifest
            payload = existing
        elif isinstance(existing, dict) and "variant" in existing:
            payload = {
                "variants": {
                    existing["variant"]: existing,
                    variant: manifest,
                }
            }
        else:
            payload = {"variants": {variant: manifest}}
    else:
        payload = manifest

    manifest_path.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info("Wrote %s (%d docs)", out_corpus, written)
    log.info("Wrote %s", manifest_path)

    print(
        f"emit-preprocessed-corpus: variant={variant} "
        f"docs={summary.document_count} "
        f"chunks_changed={summary.chunks_changed} "
        f"chunks_dropped={summary.chunks_dropped} "
        f"prefix_strips={summary.prefix_strip_count} "
        f"inline_edits_removed={summary.total_inline_edit_removals} "
        f"out={out_corpus}"
    )
    return 0


# ---------------------------------------------------------------------------
# Phase 1B — multi-variant length comparison.
#
# Reads N analyze-corpus-lengths reports (one per preprocess variant)
# and emits a single wide table that lines up the headline distribution
# stats + chunks-over-cap counts. Cheap because the per-variant
# analyses are already on disk; this command never re-tokenizes.
# ---------------------------------------------------------------------------


def _run_compare_corpus_lengths_cli(args: argparse.Namespace) -> int:
    import json as _json

    bundles: List[Dict[str, Any]] = []
    for spec in args.analysis:
        if ":" in spec and not Path(spec).exists():
            label, _, raw_path = spec.partition(":")
        else:
            raw_path = spec
            label = Path(spec).stem
        path = Path(raw_path)
        if not path.exists():
            log.error("Analysis file not found: %s", path)
            return 2
        try:
            payload = _json.loads(path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError as ex:
            log.error("Bad JSON in %s: %s", path, ex)
            return 2
        analysis = payload.get("analysis", payload)
        bundles.append({
            "label": label,
            "path": str(path),
            "analysis": analysis,
        })

    if not bundles:
        log.error("No --analysis files provided.")
        return 2

    rows: List[Dict[str, Any]] = []
    for b in bundles:
        a = b["analysis"]
        char = a.get("char_length", {})
        tok = a.get("token_length", {})
        over = a.get("chunks_over_token_threshold", {}) or {}
        rows.append({
            "label": b["label"],
            "path": b["path"],
            "chunk_count": int(a.get("chunk_count", 0)),
            "document_count": int(a.get("document_count", 0)),
            "char_p50": float(char.get("p50", 0)),
            "char_p90": float(char.get("p90", 0)),
            "char_p95": float(char.get("p95", 0)),
            "char_p99": float(char.get("p99", 0)),
            "char_max": int(char.get("max", 0)),
            "token_p50": float(tok.get("p50", 0)),
            "token_p90": float(tok.get("p90", 0)),
            "token_p95": float(tok.get("p95", 0)),
            "token_p99": float(tok.get("p99", 0)),
            "token_max": int(tok.get("max", 0)),
            "over_512": int(over.get("512", 0)),
            "over_1024": int(over.get("1024", 0)),
            "over_2048": int(over.get("2048", 0)),
            "over_4096": int(over.get("4096", 0)),
            "over_8192": int(over.get("8192", 0)),
        })

    out_json: Path = args.out_json
    out_md: Path = args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "harness": "compare-corpus-lengths",
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rows": rows,
    }
    out_json.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines: List[str] = []
    md_lines.append("# Corpus length comparison across preprocess variants")
    md_lines.append("")
    md_lines.append("| variant | chunks | docs | char p50 | char p90 | char p95 | char p99 | char max | tok p50 | tok p90 | tok p95 | tok p99 | tok max | >512 | >1024 | >2048 | >4096 | >8192 |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['chunk_count']} | {r['document_count']} | "
            f"{r['char_p50']:.0f} | {r['char_p90']:.0f} | "
            f"{r['char_p95']:.0f} | {r['char_p99']:.0f} | {r['char_max']} | "
            f"{r['token_p50']:.0f} | {r['token_p90']:.0f} | "
            f"{r['token_p95']:.0f} | {r['token_p99']:.0f} | {r['token_max']} | "
            f"{r['over_512']} | {r['over_1024']} | {r['over_2048']} | "
            f"{r['over_4096']} | {r['over_8192']} |"
        )
    md_lines.append("")
    md_lines.append("**Source files:**")
    md_lines.append("")
    for r in rows:
        md_lines.append(f"- `{r['label']}` ← `{r['path']}`")
    md_lines.append("")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    log.info("Wrote %s", out_json)
    log.info("Wrote %s", out_md)
    print(
        f"compare-corpus-lengths: variants={len(rows)} "
        + " | ".join(
            f"{r['label']}: chunks={r['chunk_count']}, "
            f"tok_p95={r['token_p95']:.0f}, >1024={r['over_1024']}"
            for r in rows
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase 1C — token-aware chunker CLIs.
# ---------------------------------------------------------------------------


def _run_diagnose_chunker_long_tail_cli(args: argparse.Namespace) -> int:
    import json as _json

    if not args.corpus.exists():
        log.error("Corpus not found: %s", args.corpus)
        return 2

    thresholds = (
        tuple(args.threshold)
        if args.threshold
        else DEFAULT_DIAGNOSE_THRESHOLDS
    )

    summary, top_samples = diagnose_chunker_long_tail(
        args.corpus,
        tokenizer_name=args.tokenizer,
        thresholds=thresholds,
        long_chunk_threshold=int(args.long_chunk_threshold),
        top_n=int(args.top_n),
        batch_size=int(args.batch_size),
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "metadata": {
            "harness": "diagnose-chunker-long-tail",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "thresholds": list(thresholds),
            "long_chunk_threshold": int(args.long_chunk_threshold),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": chunker_diagnosis_to_dict(summary),
    }
    (out_dir / "chunker-diagnosis-summary.json").write_text(
        _json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "chunker-diagnosis-summary.md").write_text(
        render_chunker_diagnosis_markdown(summary),
        encoding="utf-8",
    )

    top_payload = {
        "metadata": {
            "harness": "diagnose-chunker-long-tail",
            "corpus": str(args.corpus),
            "tokenizer": args.tokenizer,
            "top_n": int(args.top_n),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "samples": diagnose_samples_to_dict_list(top_samples),
    }
    (out_dir / f"chunker-provenance-top{int(args.top_n)}.json").write_text(
        _json.dumps(top_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"chunker-provenance-top{int(args.top_n)}.md").write_text(
        render_chunker_provenance_markdown(top_samples, summary=summary),
        encoding="utf-8",
    )

    log.info("Wrote %s", out_dir / "chunker-diagnosis-summary.json")
    log.info("Wrote %s", out_dir / "chunker-diagnosis-summary.md")
    log.info("Wrote %s", out_dir / f"chunker-provenance-top{int(args.top_n)}.json")
    log.info("Wrote %s", out_dir / f"chunker-provenance-top{int(args.top_n)}.md")

    over_long = summary.chunks_over_token_threshold.get(
        int(args.long_chunk_threshold), 0
    )
    print(
        f"diagnose-chunker-long-tail: chunks={summary.chunk_count} "
        f"sections={summary.section_count} "
        f"over_{int(args.long_chunk_threshold)}={over_long} "
        f"sections_with_long={summary.sections_with_long_chunks} "
        f"out={out_dir}"
    )
    return 0


def _run_emit_token_aware_chunks_cli(args: argparse.Namespace) -> int:
    import json as _json

    if not args.corpus.exists():
        log.error("Corpus not found: %s", args.corpus)
        return 2

    out_corpus: Path = args.out_corpus
    if out_corpus.resolve() == args.corpus.resolve():
        log.error(
            "Refusing to overwrite the source corpus %s — choose a "
            "different --out-corpus path.",
            args.corpus,
        )
        return 2

    try:
        chunker_config = TokenAwareConfig(
            target_tokens=int(args.target_tokens),
            soft_max_tokens=int(args.soft_max_tokens),
            hard_max_tokens=int(args.hard_max_tokens),
            overlap_tokens=int(args.overlap_tokens),
        )
    except ValueError as ex:
        log.error("Invalid token-aware config: %s", ex)
        return 2

    counter, encode, decode = build_default_tokenizer_callables(args.tokenizer)

    out_corpus.parent.mkdir(parents=True, exist_ok=True)

    if args.no_provenance:
        provenance_path = None
    elif args.provenance is not None:
        provenance_path = Path(args.provenance)
    else:
        provenance_path = out_corpus.parent / "chunks_provenance.jsonl"

    summary = emit_token_aware_corpus(
        args.corpus,
        out_corpus,
        config=EmitConfig(chunker=chunker_config, write_provenance=not args.no_provenance),
        token_counter=counter,
        encode_fn=encode,
        decode_fn=decode,
        provenance_path=provenance_path,
    )

    manifest_path: Path = (
        Path(args.manifest)
        if args.manifest is not None
        else out_corpus.parent / "manifest.json"
    )

    manifest_entry = {
        **emit_summary_to_dict(summary),
        "variant": str(args.variant_label),
        "tokenizer": args.tokenizer,
        "provenance_path": str(provenance_path) if provenance_path else None,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Merge with existing manifest under "variants" to preserve prior
    # token-aware emits (mirrors the preprocess emit pattern).
    if manifest_path.exists():
        try:
            existing = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict) and "variants" in existing:
            existing["variants"][str(args.variant_label)] = manifest_entry
            payload = existing
        elif isinstance(existing, dict) and "variant" in existing:
            payload = {"variants": {
                existing["variant"]: existing,
                str(args.variant_label): manifest_entry,
            }}
        else:
            payload = {"variants": {str(args.variant_label): manifest_entry}}
    else:
        payload = manifest_entry

    manifest_path.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_md_path = out_corpus.parent / f"emit-summary.{args.variant_label}.md"
    summary_md_path.write_text(
        render_emit_summary_markdown(summary), encoding="utf-8",
    )

    log.info("Wrote %s", out_corpus)
    log.info("Wrote %s", manifest_path)
    log.info("Wrote %s", summary_md_path)

    print(
        f"emit-token-aware-chunks: variant={args.variant_label} "
        f"chunker={TOKEN_AWARE_CHUNKER_VERSION} "
        f"docs={summary.document_count} "
        f"sections={summary.section_count} "
        f"input_units={summary.input_payload_unit_count} "
        f"output_chunks={summary.output_chunk_count} "
        f"fallback={summary.fallback_used_count} "
        f"over_hard_max={summary.chunks_over_hard_max} "
        f"out={out_corpus}"
    )
    return 0


def _run_compare_chunker_lengths_cli(args: argparse.Namespace) -> int:
    import json as _json

    bundles: List[Tuple[str, Path]] = []
    for spec in args.corpus:
        if ":" in spec and not Path(spec).exists():
            label, _, raw_path = spec.partition(":")
            path = Path(raw_path)
        else:
            path = Path(spec)
            label = path.stem
        if not path.exists():
            log.error("Corpus file not found: %s", path)
            return 2
        bundles.append((label, path))

    if not bundles:
        log.error("No --corpus arguments provided.")
        return 2

    thresholds = (
        tuple(args.threshold)
        if args.threshold
        else DEFAULT_TOKEN_THRESHOLDS
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for label, path in bundles:
        log.info("Analyzing corpus: %s (%s)", label, path)
        analysis = analyze_corpus_lengths(
            path,
            tokenizer_name=args.tokenizer,
            thresholds=thresholds,
            top_longest=int(args.top_longest),
            batch_size=int(args.batch_size),
        )

        per_corpus_payload = {
            "metadata": {
                "harness": "compare-chunker-lengths · per-corpus",
                "corpus_label": label,
                "corpus_path": str(path),
                "tokenizer": args.tokenizer,
                "thresholds": list(thresholds),
                "top_longest": int(args.top_longest),
                "run_at": datetime.now().isoformat(timespec="seconds"),
            },
            "analysis": length_analysis_to_dict(analysis),
        }
        per_corpus_json = out_dir / f"length-{label}.json"
        per_corpus_md = out_dir / f"length-{label}.md"
        per_corpus_json.write_text(
            _json.dumps(per_corpus_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        per_corpus_md.write_text(
            render_length_analysis_markdown(analysis),
            encoding="utf-8",
        )
        log.info("Wrote %s", per_corpus_json)
        log.info("Wrote %s", per_corpus_md)

        char = analysis.char_length
        tok = analysis.token_length
        over = analysis.chunks_over_token_threshold
        rows.append({
            "label": label,
            "path": str(path),
            "chunk_count": analysis.chunk_count,
            "document_count": analysis.document_count,
            "char_p50": char.p50,
            "char_p90": char.p90,
            "char_p95": char.p95,
            "char_p99": char.p99,
            "char_max": char.max,
            "token_p50": tok.p50,
            "token_p90": tok.p90,
            "token_p95": tok.p95,
            "token_p99": tok.p99,
            "token_max": tok.max,
            **{f"over_{t}": int(over.get(int(t), 0)) for t in thresholds},
        })

    # Cross-corpus comparison artifact.
    base_row = rows[0] if rows else None
    deltas: List[Dict[str, Any]] = []
    for r in rows[1:] if len(rows) > 1 else []:
        if base_row is None:
            break
        d: Dict[str, Any] = {
            "label": r["label"],
            "vs_label": base_row["label"],
            "chunk_count_delta": r["chunk_count"] - base_row["chunk_count"],
            "chunk_count_ratio": (
                round(r["chunk_count"] / base_row["chunk_count"], 4)
                if base_row["chunk_count"] else 0.0
            ),
            "token_max_delta": r["token_max"] - base_row["token_max"],
            "token_p95_delta": r["token_p95"] - base_row["token_p95"],
        }
        for t in thresholds:
            key = f"over_{t}"
            d[f"{key}_delta"] = r.get(key, 0) - base_row.get(key, 0)
        deltas.append(d)

    cmp_json = out_dir / "length-comparison.json"
    cmp_md = out_dir / "length-comparison.md"
    cmp_payload = {
        "metadata": {
            "harness": "compare-chunker-lengths",
            "tokenizer": args.tokenizer,
            "thresholds": list(thresholds),
            "run_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rows": rows,
        "deltas_vs_first_row": deltas,
        "accounting_note": (
            "rows[*].chunk_count is the FINAL RETRIEVABLE CHUNK count "
            "(what the FAISS index would contain). It is NOT the same "
            "as the preprocess summary's 'chunks_processed', which counts "
            "TRANSFORMED PAYLOAD ENTRIES (one entry per "
            "sections.<name>.chunks[i] / list[i] / text). For "
            "token-aware corpora, sections.<name>.chunks[i] is already "
            "a final chunk; for raw / preprocessed corpora the production "
            "chunker re-windows them via window_by_chars before they "
            "become final chunks."
        ),
    }
    cmp_json.write_text(
        _json.dumps(cmp_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines: List[str] = []
    md_lines.append("# Phase 1C — chunker length comparison")
    md_lines.append("")
    md_lines.append(f"_tokenizer: `{args.tokenizer}`_")
    md_lines.append("")
    md_lines.append(
        "| variant | chunks | docs | char p50 | char p90 | char p95 | char p99 | char max | tok p50 | tok p90 | tok p95 | tok p99 | tok max | "
        + " | ".join(f">{t}" for t in thresholds)
        + " |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                    + "---:|" * len(thresholds))
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['chunk_count']} | {r['document_count']} | "
            f"{r['char_p50']:.0f} | {r['char_p90']:.0f} | "
            f"{r['char_p95']:.0f} | {r['char_p99']:.0f} | {r['char_max']} | "
            f"{r['token_p50']:.0f} | {r['token_p90']:.0f} | "
            f"{r['token_p95']:.0f} | {r['token_p99']:.0f} | {r['token_max']} | "
            + " | ".join(str(r.get(f"over_{t}", 0)) for t in thresholds)
            + " |"
        )
    md_lines.append("")
    if deltas:
        md_lines.append(f"## Deltas vs `{base_row['label']}`")
        md_lines.append("")
        md_lines.append(
            "| variant | chunk_count_delta | chunk_count_ratio | token_max_delta | token_p95_delta | "
            + " | ".join(f">{t}_delta" for t in thresholds)
            + " |"
        )
        md_lines.append(
            "|---|---:|---:|---:|---:|"
            + "---:|" * len(thresholds)
        )
        for d in deltas:
            md_lines.append(
                f"| {d['label']} | {d['chunk_count_delta']:+d} | "
                f"{d['chunk_count_ratio']:.3f} | {d['token_max_delta']:+d} | "
                f"{d['token_p95_delta']:+.0f} | "
                + " | ".join(
                    f"{d.get(f'over_{t}_delta', 0):+d}" for t in thresholds
                )
                + " |"
            )
        md_lines.append("")
    md_lines.append("## Accounting note")
    md_lines.append("")
    md_lines.append(
        "- `chunk_count` is the **final retrievable chunk count** (what "
        "the FAISS index would hold after the production chunker runs)."
    )
    md_lines.append(
        "- It is **not** the same number reported as `chunks_processed` "
        "in the Phase 1B preprocess summary; that one counts "
        "**transformed payload entries** (one per "
        "`sections.<name>.chunks[i]` / `list[i]` / `text`)."
    )
    md_lines.append(
        "- For token-aware corpora, `sections.<name>.chunks[i]` already "
        "*is* a final chunk; for raw / preprocessed corpora the "
        "production chunker (`window_by_chars`) re-windows them before "
        "they become final chunks."
    )
    md_lines.append("")
    md_lines.append("**Source corpora:**")
    md_lines.append("")
    for r in rows:
        md_lines.append(f"- `{r['label']}` ← `{r['path']}`")
    md_lines.append("")
    cmp_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    log.info("Wrote %s", cmp_json)
    log.info("Wrote %s", cmp_md)

    print(
        f"compare-chunker-lengths: variants={len(rows)} "
        + " | ".join(
            f"{r['label']}: chunks={r['chunk_count']}, "
            f"tok_p95={r['token_p95']:.0f}, tok_max={r['token_max']}, "
            f">1024={r.get('over_1024', 0)}"
            for r in rows
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase 2B candidate-boost CLIs.
# ---------------------------------------------------------------------------


def _default_phase2b_dir(suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/phase2/2b_candidate_boost-{suffix}-{timestamp}")


def _build_boost_config(args: argparse.Namespace) -> BoostConfig:
    """Translate CLI flags into a BoostConfig with safety validation."""
    excluded: Tuple[str, ...]
    if args.excluded_section:
        excluded = tuple(s for s in args.excluded_section if s)
    else:
        from eval.harness.boost_scorer import DEFAULT_EXCLUDED_SECTIONS
        excluded = DEFAULT_EXCLUDED_SECTIONS
    cfg = BoostConfig(
        title_exact_boost=float(args.title_exact_boost),
        title_partial_boost=float(args.title_partial_boost),
        section_keyword_boost=float(args.section_keyword_boost),
        section_path_boost=float(args.section_path_boost),
        max_boost=float(args.max_boost),
        title_min_len=int(args.title_min_len),
        excluded_sections=excluded,
    )
    errors = cfg.validate()
    if errors:
        raise SystemExit(
            "Invalid boost config:\n  - " + "\n  - ".join(errors)
        )
    return cfg


def _run_retrieval_candidate_boost_cli(args: argparse.Namespace) -> int:
    """Phase 2B: dense + boost (+ optional rerank) over an offline corpus.

    Mirrors the structure of ``_run_retrieval_rerank_cli`` so the same
    embedder / corpus path / extra-hit-k machinery is re-used. The
    additional work is:

      1. Build a NoOp-reranker base ``Retriever`` whose top_k equals
         ``--top-n`` (so the boost stage sees a candidate pool of
         that exact size; matches the spec's "don't grow the pool").
      2. Build the ``MetadataBoostReranker`` from the CLI flags.
      3. Optionally build a cross-encoder reranker that runs AFTER
         the boost reorder.
      4. Wrap (1)-(3) in a ``BoostingEvalRetriever`` and drive it
         through ``run_boost_retrieval_eval``.
      5. Persist the eight Phase 2B artifacts to ``--out-dir``.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _P

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import CrossEncoderReranker, NoOpReranker
    from app.core.config import get_settings
    from eval.harness.offline_corpus import build_offline_rag_stack

    top_n = int(args.top_n)
    final_top_k = int(args.final_top_k) if args.final_top_k is not None else top_n
    if final_top_k > top_n:
        log.error(
            "--final-top-k (%d) must be <= --top-n (%d). The boost stage "
            "cannot grow the candidate pool beyond what dense retrieval "
            "produced; the eval would be evaluating chunks the boost "
            "stage never saw.",
            final_top_k, top_n,
        )
        return 2

    boost_cfg = _build_boost_config(args)

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=True,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )

    post_reranker = None
    if args.reranker_model:
        try:
            post_reranker = CrossEncoderReranker(
                model_name=str(args.reranker_model),
                max_length=int(args.reranker_max_length),
                batch_size=int(args.reranker_batch_size),
                text_max_chars=int(args.reranker_text_max_chars),
                device=args.reranker_device or None,
                oom_fallback_batch_size=(
                    int(args.reranker_oom_fallback_batch_size)
                    if args.reranker_oom_fallback_batch_size is not None
                    else None
                ),
            )
        except Exception as ex:
            log.error(
                "Failed to construct CrossEncoderReranker (%s: %s).",
                type(ex).__name__, ex,
            )
            return 2

    tmp_dir = _P(tempfile.mkdtemp(prefix="retrieval-boost-offline-"))
    try:
        base_retriever, _generator, info = build_offline_rag_stack(
            _P(args.corpus),
            embedder=embedder,
            index_dir=tmp_dir,
            top_k=top_n,           # base produces the candidate pool
            reranker=NoOpReranker(),
            candidate_k=top_n,
        )
    except Exception as ex:
        log.error(
            "Failed to build the boost retrieval stack (%s: %s).",
            type(ex).__name__, ex,
        )
        return 2

    doc_metadata = load_doc_metadata(_P(args.corpus))
    boost_reranker = MetadataBoostReranker(
        config=boost_cfg, doc_metadata=doc_metadata,
    )
    boost_retriever = BoostingEvalRetriever(
        base_retriever=base_retriever,
        boost_reranker=boost_reranker,
        post_reranker=post_reranker,
        boost_top_k=top_n,
        final_top_k=final_top_k,
    )

    dataset = load_jsonl(args.dataset)
    extra_hit_ks = _resolve_extra_hit_ks(
        getattr(args, "extra_hit_k", None), top_k=final_top_k,
    )

    artifacts = run_boost_retrieval_eval(
        dataset,
        retriever=boost_retriever,
        final_top_k=final_top_k,
        boost_top_k=top_n,
        mrr_k=int(args.mrr_k),
        ndcg_k=int(args.ndcg_k),
        extra_hit_ks=extra_hit_ks,
        dataset_path=str(args.dataset),
        corpus_path=str(args.corpus),
        config=boost_cfg.to_dict(),
    )

    out_dir = args.out_dir or _default_phase2b_dir("candidate-boost")
    out_dir = Path(out_dir)
    metadata: Dict[str, Any] = {
        "harness": "retrieval-candidate-boost",
        "embedding_model": settings.rag_embedding_model,
        "embedding_max_seq_length": int(args.max_seq_length),
        "embedding_batch_size": int(args.embed_batch_size),
        "rag_index_dir": str(settings.rag_index_dir),
        "corpus_path": str(args.corpus),
        "dataset": str(args.dataset),
        "top_n": top_n,
        "final_top_k": final_top_k,
        "extra_hit_ks": list(extra_hit_ks),
        "boost_config": boost_cfg.to_dict(),
        "post_reranker": (
            None if post_reranker is None
            else {
                "name": post_reranker.name,
                "model": str(args.reranker_model),
                "batch_size": int(args.reranker_batch_size),
                "max_length": int(args.reranker_max_length),
                "text_max_chars": int(args.reranker_text_max_chars),
                "device": args.reranker_device,
            }
        ),
        "offline_corpus": {
            "path": info.corpus_path,
            "document_count": info.document_count,
            "chunk_count": info.chunk_count,
            "index_version": info.index_version,
            "dimension": info.dimension,
        },
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_boost_artifacts(artifacts, out_dir, metadata=metadata)

    print()
    print(f"retrieval-candidate-boost — {out_dir}")
    print(
        f"  hit@1={_fmt(artifacts.summary.mean_hit_at_1)} "
        f"hit@3={_fmt(artifacts.summary.mean_hit_at_3)} "
        f"hit@5={_fmt(artifacts.summary.mean_hit_at_5)} "
        f"mrr@10={_fmt(artifacts.summary.mean_mrr_at_10)} "
        f"ndcg@10={_fmt(artifacts.summary.mean_ndcg_at_10)}"
    )
    print(
        f"  boost: applied={artifacts.boost_summary.boost_applied_count} "
        f"title_match={artifacts.boost_summary.title_match_count} "
        f"section_match={artifacts.boost_summary.section_match_count} "
        f"avg_boost={artifacts.boost_summary.avg_boost_score:.4f} "
        f"rescued={artifacts.boost_summary.boosted_rescued_count} "
        f"regressed={artifacts.boost_summary.boosted_regressed_count}"
    )
    print()
    return 0


def _read_report_rows(path: Path) -> List[Mapping[str, Any]]:
    """Read the ``rows`` payload from a retrieval_eval_report.json."""
    import json as _json
    with Path(path).open("r", encoding="utf-8") as fp:
        data = _json.load(fp)
    return data.get("rows", []) or []


def _read_dump_rows(path: Optional[Path]) -> List[Mapping[str, Any]]:
    """Read top_k_dump.jsonl entries; tolerates a None path (returns [])."""
    if path is None:
        return []
    import json as _json
    out: List[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            out.append(_json.loads(line))
    return out


def _build_doc_meta_for_miss_analysis(
    corpus_path: Optional[Path],
) -> Dict[str, Mapping[str, Any]]:
    """Return doc_id → minimal {title, section_names} dict for the analyzer."""
    if corpus_path is None:
        return {}
    raw = load_doc_metadata(Path(corpus_path))
    return {
        did: {"title": meta.title, "section_names": meta.section_names}
        for did, meta in raw.items()
    }


def _run_retrieval_candidate_miss_analysis_cli(args: argparse.Namespace) -> int:
    """Phase 2B: classify Phase 2A misses into the eight failure buckets."""
    import json as _json

    rows = _read_report_rows(args.report)
    if not rows:
        log.error(
            "No rows found in %s — is this a retrieval_eval_report.json?",
            args.report,
        )
        return 2
    dump_rows = _read_dump_rows(args.top_k_dump)
    doc_meta = _build_doc_meta_for_miss_analysis(args.corpus)

    top_ks: Tuple[int, ...]
    if args.top_k:
        top_ks = tuple(sorted({int(k) for k in args.top_k if int(k) > 0}))
    else:
        top_ks = DEFAULT_MISS_TOP_KS

    report = classify_candidate_misses(
        rows,
        dump_rows=dump_rows,
        doc_metadata=doc_meta,
        top_ks=top_ks,
        deep_k=int(args.deep_k),
        sample_limit=int(args.sample_limit),
    )

    out_dir = args.out_dir or _default_phase2b_dir("candidate-boost")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate-miss-topk-analysis.json").write_text(
        _json.dumps(
            candidate_miss_report_to_dict(report),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "candidate-miss-topk-analysis.md").write_text(
        render_candidate_miss_markdown(report),
        encoding="utf-8",
    )

    print()
    print(f"retrieval-candidate-miss-analysis — {out_dir}")
    print(f"  rows_evaluated: {report.rows_evaluated}")
    for tkr in report.per_top_k:
        print(
            f"  top-{tkr.top_k}: missed={tkr.queries_missed} "
            f"miss_rate={tkr.miss_rate:.4f}"
        )
    print()
    return 0


def _run_retrieval_boost_failure_analysis_cli(args: argparse.Namespace) -> int:
    """Phase 2B: cross-tab dense → boost → rerank trajectories."""
    import json as _json

    dense_rows = _read_report_rows(args.dense_report)
    boost_rows = _read_report_rows(args.boost_report)
    rerank_rows = (
        _read_report_rows(args.rerank_report)
        if args.rerank_report is not None
        else None
    )
    boost_dump = _read_dump_rows(args.boost_dump) if args.boost_dump else None

    analysis = classify_boost_failures(
        dense_rows=dense_rows,
        boost_rows=boost_rows,
        rerank_rows=rerank_rows,
        boost_dump=boost_dump,
        top_k=int(args.top_k),
        sample_limit=int(args.sample_limit),
    )

    out_dir = args.out_dir or _default_phase2b_dir("candidate-boost")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "boost-failure-analysis.json").write_text(
        _json.dumps(
            boost_failure_analysis_to_dict(analysis),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "boost-failure-analysis.md").write_text(
        render_boost_failure_markdown(analysis),
        encoding="utf-8",
    )

    print()
    print(f"retrieval-boost-failure-analysis — {out_dir}")
    print(f"  queries_evaluated: {analysis.queries_evaluated}")
    for stat in analysis.groups:
        print(f"  {stat.name}: {stat.count} ({stat.ratio:.4f})")
    print()
    return 0


def _run_retrieval_boost_pareto_cli(args: argparse.Namespace) -> int:
    """Phase 2B: build the Phase 2A + Phase 2B unified Pareto frontier."""
    import json as _json
    from dataclasses import asdict as _asdict

    from eval.harness.topn_sweep import (
        TopNSweepEntry,
        TopNSweepReport,
    )

    def _load_sweep(path: Path) -> TopNSweepReport:
        with Path(path).open("r", encoding="utf-8") as fp:
            data = _json.load(fp)
        entries: List[TopNSweepEntry] = []
        for entry in data.get("entries", []) or []:
            kwargs = dict(entry)
            entries.append(TopNSweepEntry(**kwargs))
        return TopNSweepReport(
            schema=str(data.get("schema") or "phase2a-topn-sweep.v1"),
            entries=entries,
            caveats=list(data.get("caveats") or []),
        )

    sweep_a = _load_sweep(args.phase2a_sweep)
    sweep_b = _load_sweep(args.phase2b_sweep)

    report = compute_boost_pareto_frontier(
        phase2a_sweep=sweep_a,
        phase2b_sweep=sweep_b,
        metric=str(args.metric),
        latency=str(args.latency),
        phase2b_label=args.phase2b_label,
    )

    out_dir = args.out_dir or _default_phase2b_dir("candidate-boost")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "boost-pareto-frontier.json").write_text(
        _json.dumps(
            boost_pareto_to_dict(report), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "boost-pareto-frontier.md").write_text(
        render_boost_pareto_markdown(report),
        encoding="utf-8",
    )

    print()
    print(f"retrieval-boost-pareto — {out_dir}")
    on_frontier = [e for e in report.entries if e.on_frontier]
    print(f"  on-frontier points: {len(on_frontier)}")
    for e in sorted(on_frontier, key=lambda e: e.latency_ms):
        print(
            f"  - {e.track}::{e.label}: "
            f"{report.metric_field}={e.metric:.4f} "
            f"{report.latency_field}={e.latency_ms:.2f}"
        )
    print()
    return 0


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _default_report_path(mode: str, ext: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/{mode}-{timestamp}.{ext}")


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
