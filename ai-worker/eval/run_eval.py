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
from typing import Any, Dict, List, Mapping, Optional

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


log = logging.getLogger("eval")


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
    if args.mode == "retrieval-compare":
        return _run_retrieval_compare_cli(args)
    if args.mode == "retrieval-miss-analysis":
        return _run_miss_analysis_cli(args)
    parser.error(f"unknown mode: {args.mode}")
    return 2  # unreachable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Run an eval harness over a JSONL dataset.",
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
                    help="Offline corpus JSONL (e.g. "
                         "eval/corpora/anime_namu_v3/corpus.jsonl). "
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

    # --- retrieval-compare ---
    rc = subs.add_parser(
        "retrieval-compare",
        help="Compare two retrieval-eval reports side-by-side.",
    )
    rc.add_argument(
        "--deterministic-report", required=True, type=Path,
        help="Path to retrieval_eval_report.json from the deterministic run "
             "(e.g. eval/reports/retrieval-silver200-baseline/"
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
        "--out-json", type=Path, required=True,
        help="Output path for retrieval-baseline-comparison.json.",
    )
    rc.add_argument(
        "--out-md", type=Path, required=True,
        help="Output path for retrieval-baseline-comparison.md.",
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
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
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

    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.core.config import get_settings

    from eval.harness.offline_corpus import build_offline_rag_stack

    settings = get_settings()
    top_k = int(args.top_k) if args.top_k is not None else int(settings.rag_top_k)

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
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

    summary, rows, dump, dup = run_retrieval_eval(
        dataset,
        retriever=retriever,
        top_k=top_k,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
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
        # quality on this corpus.
        max_seq_length=getattr(args, "max_seq_length", 1024),
        batch_size=getattr(args, "embed_batch_size", 32),
        show_progress_bar=True,
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

    comparison = run_comparison(
        deterministic_rows=det_rows,
        deterministic_dataset_path=det_dataset,
        opus_rows=opus_rows,
        opus_dataset_path=opus_dataset,
        excluded_answer_type=args.exclude_answer_type,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "harness": "retrieval-compare",
            "deterministic_report": str(args.deterministic_report),
            "opus_report": str(args.opus_report),
            "excluded_answer_type": args.exclude_answer_type,
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
# Shared helpers.
# ---------------------------------------------------------------------------


def _default_report_path(mode: str, ext: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/{mode}-{timestamp}.{ext}")


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
