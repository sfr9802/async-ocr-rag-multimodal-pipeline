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
    try:
        retriever, generator, settings = _build_rag_stack(args)
    except Exception as ex:
        log.error(
            "Failed to build the RAG eval stack (%s: %s). "
            "Make sure the FAISS index is built, the ragmeta schema exists, "
            "and the configured embedding model matches the one in build.json.",
            type(ex).__name__, ex,
        )
        return 2

    dataset = load_jsonl(args.dataset)
    top_k = int(args.top_k) if args.top_k is not None else int(settings.rag_top_k)

    summary, rows = run_rag_eval(
        dataset,
        retriever=retriever,
        generator=generator,
        top_k=top_k,
        dataset_path=str(args.dataset),
    )

    _print_rag_summary(summary)

    out_json = args.out_json or _default_report_path("rag", "json")
    write_json_report(
        out_json,
        summary=rag_summary_to_dict(summary),
        rows=[rag_row_to_dict(r) for r in rows],
        metadata=_rag_metadata(settings, top_k),
    )
    if not args.no_csv:
        out_csv = args.out_csv or _default_report_path("rag", "csv")
        write_csv_report(
            out_csv,
            [rag_row_to_dict(r) for r in rows],
            columns=_rag_csv_columns(),
        )
    return 0


def _build_rag_stack(args: argparse.Namespace):
    """Construct the same Retriever + ExtractiveGenerator the worker
    uses — but without any Spring/queue wiring. The import path is
    local to this function so a bare `python -m eval.run_eval ocr ...`
    call doesn't pull in faiss / psycopg2 / torch."""
    from pathlib import Path as _P

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
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )
    retriever.ensure_ready()
    return retriever, ExtractiveGenerator(), settings


def _rag_metadata(settings: Any, top_k: int) -> Dict[str, Any]:
    return {
        "harness": "rag",
        "embedding_model": settings.rag_embedding_model,
        "rag_index_dir": str(settings.rag_index_dir),
        "top_k": top_k,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }


def _rag_csv_columns() -> List[str]:
    return [
        "query",
        "expected_doc_ids",
        "retrieved_doc_ids",
        "hit_at_k",
        "reciprocal_rank",
        "keyword_coverage",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "index_version",
        "embedding_model",
        "notes",
        "error",
    ]


def _print_rag_summary(summary: RagEvalSummary) -> None:
    print()
    print(f"RAG eval — {summary.dataset_path}")
    print(f"  rows evaluated       : {summary.row_count}")
    print(f"  with expected_doc_ids: {summary.rows_with_expected_doc_ids}")
    print(f"  with expected_kwds   : {summary.rows_with_expected_keywords}")
    print(f"  top_k                : {summary.top_k}")
    print(f"  mean hit@k           : {_fmt(summary.mean_hit_at_k)}")
    print(f"  MRR                  : {_fmt(summary.mrr)}")
    print(f"  mean keyword coverage: {_fmt(summary.mean_keyword_coverage)}")
    print(f"  mean retrieval_ms    : {summary.mean_retrieval_ms:.1f}")
    print(f"  p50 retrieval_ms     : {summary.p50_retrieval_ms:.1f}")
    print(f"  mean generation_ms   : {summary.mean_generation_ms:.1f}")
    print(f"  mean total_ms        : {summary.mean_total_ms:.1f}")
    print(f"  errors               : {summary.error_count}")
    print(f"  index_version        : {summary.index_version}")
    print(f"  embedding_model      : {summary.embedding_model}")
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
    print(f"OCR eval — {summary.dataset_path}")
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
# Shared helpers.
# ---------------------------------------------------------------------------


def _default_report_path(mode: str, ext: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"eval/reports/{mode}-{timestamp}.{ext}")


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
