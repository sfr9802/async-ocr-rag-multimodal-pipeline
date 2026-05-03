"""Build promotion-gate metrics from RAG ingestion smoke/eval reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_XLSX_REPORT = Path("reports/rag_ingestion_sample_batch_report.json")
DEFAULT_PDF_REPORT = Path("reports/rag_pdf_ingestion_sample_batch_report.json")
DEFAULT_RETRIEVAL_REPORT = Path("reports/rag_retrieval_eval_report.json")
DEFAULT_OCR_REPORT = Path("reports/rag_pdf_ocr_fallback_smoke_report.json")
DEFAULT_OUTPUT = Path("reports/rag_ingestion_promotion_gate_metrics.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    xlsx = read_json(Path(args.xlsx_report))
    pdf = read_json(Path(args.pdf_report))
    retrieval = read_json(Path(args.retrieval_report))
    ocr = read_json(Path(args.ocr_report)) if Path(args.ocr_report).exists() else {}

    xlsx_metrics = dict(xlsx.get("metrics") or {})
    pdf_metrics = dict(pdf.get("metrics") or {})
    retrieval_metrics = dict(retrieval.get("metrics") or {})
    ocr_db = dict(ocr.get("db_report") or {})
    gate_input_missing: list[str] = []

    missing_required = _required_int_source_metric(
        gate_input_missing,
        "missing_required_metadata_count",
        ("xlsx", xlsx_metrics),
        ("pdf", pdf_metrics),
    )
    zero_indexable = _required_int_source_metric(
        gate_input_missing,
        "zero_indexable_chunk_count",
        ("xlsx", xlsx_metrics),
        ("pdf", pdf_metrics),
    )
    missing_table_metadata = required_int_metric(
        gate_input_missing,
        xlsx_metrics,
        "missing_table_metadata_count",
        "xlsx",
    )
    table_detection_accuracy = 1.0 if missing_table_metadata == 0 else 0.0
    ocr_needed_count = int(ocr_db.get("ocr_search_unit_count") or 0) if ocr.get("status") == "PASSED" else 0
    unsupported_file_rate = _required_float_source_metric(
        gate_input_missing,
        "unsupported_file_rate",
        ("xlsx", xlsx_metrics),
        ("pdf", pdf_metrics),
    )
    fatal_warning_count = _required_int_source_metric(
        gate_input_missing,
        "fatal_warning_count",
        ("xlsx", xlsx_metrics),
        ("pdf", pdf_metrics),
        ("ocr", dict(ocr.get("metrics") or {})),
    )

    metrics = {
        "parser_success_rate": min(
            float_metric(xlsx_metrics, "parser_success_rate", default=0.0),
            float_metric(pdf_metrics, "parser_success_rate", default=0.0),
        ),
        "unsupported_file_rate": unsupported_file_rate,
        "zero_indexable_chunk_count": zero_indexable,
        "required_metadata_completeness": 1.0 if missing_required == 0 else 0.0,
        "missing_required_metadata_count": missing_required,
        "xlsx_citation_location_accuracy": float_metric(
            retrieval_metrics,
            "xlsx_citation_location_accuracy",
        ),
        "pdf_citation_location_accuracy": float_metric(
            retrieval_metrics,
            "pdf_citation_location_accuracy",
        ),
        "table_detection_accuracy": table_detection_accuracy,
        "OCR_needed_count": ocr_needed_count,
        "hit_at_10": float_metric(retrieval_metrics, "Hit@10", "hit_at_10"),
        "mrr_at_10": float_metric(retrieval_metrics, "MRR@10", "mrr_at_10"),
        "citation_accuracy": float_metric(retrieval_metrics, "citation_accuracy", "citation_location_accuracy"),
        "citation_location_accuracy": float_metric(retrieval_metrics, "citation_location_accuracy", "citation_accuracy"),
        "parsing_latency_p95_seconds": max(
            required_float_metric(
                gate_input_missing,
                xlsx_metrics,
                "parsing_latency_p95_seconds",
                "xlsx",
            ),
            required_float_metric(
                gate_input_missing,
                pdf_metrics,
                "parsing_latency_p95_seconds",
                "pdf",
            ),
        ),
        "indexing_latency_p95_seconds": max(
            required_float_metric(
                gate_input_missing,
                xlsx_metrics,
                "indexing_latency_p95_seconds",
                "xlsx",
            ),
            required_float_metric(
                gate_input_missing,
                pdf_metrics,
                "indexing_latency_p95_seconds",
                "pdf",
            ),
        ),
        "fatal_warning_count": fatal_warning_count,
        "gate_input_missing_count": 0,
        "gate_input_missing": gate_input_missing,
        "hidden_content_leakage_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "hidden_content_leakage_count",
            "retrieval",
        ),
        "embedding_filtered_eval": bool(retrieval_metrics.get("embedding_filtered_eval")),
        "required_embedding_status": retrieval_metrics.get("required_embedding_status"),
        "required_index_version": retrieval_metrics.get("required_index_version"),
        "indexing_filtered_hit_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "indexing_filtered_hit_count",
            "retrieval",
        ),
        "result_empty_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "result_empty_count",
            "retrieval",
        ),
        "gold_label_invalid_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "gold_label_invalid_count",
            "retrieval",
        ),
        "candidate_index_mismatch_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "candidate_index_mismatch_count",
            "retrieval",
        ),
        "embedding_status_mismatch_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "embedding_status_mismatch_count",
            "retrieval",
        ),
        "required_index_version_mismatch_count": required_int_metric(
            gate_input_missing,
            retrieval_metrics,
            "required_index_version_mismatch_count",
            "retrieval",
        ),
        "overall_failure_reason_counts": dict(retrieval_metrics.get("overall_failure_reason_counts") or {}),
        "bucket_failure_reason_counts": dict(retrieval_metrics.get("bucket_failure_reason_counts") or {}),
        "bucket_metrics": dict(retrieval.get("bucket_metrics") or {}),
        "retrieval_report_path": str(args.retrieval_report),
        "candidate_index_version": args.candidate_index_version,
        "baseline_index_version": args.baseline_index_version,
        "xlsx_hidden_search_unit_leakage_count": required_int_metric(
            gate_input_missing,
            xlsx_metrics,
            "hidden_search_unit_leakage_count",
            "xlsx",
        ),
        "pdf_missing_page_metadata_count": required_int_metric(
            gate_input_missing,
            pdf_metrics,
            "missing_page_metadata_count",
            "pdf",
        ),
        "pdf_inconsistent_location_page_metadata_count": required_int_metric(
            gate_input_missing,
            pdf_metrics,
            "inconsistent_location_page_metadata_count",
            "pdf",
        ),
        "source_reports": [
            str(args.xlsx_report),
            str(args.pdf_report),
            str(args.retrieval_report),
            str(args.ocr_report),
        ],
    }
    metrics["gate_input_missing_count"] = len(gate_input_missing)
    if ocr_needed_count > 0 and ocr_db.get("ocr_confidence_avg") is not None:
        metrics["OCR_confidence_avg"] = float_metric(ocr_db, "ocr_confidence_avg")

    payload = {
        "run_id": utc_run_id(),
        "status": "COMPLETED",
        "metrics": metrics,
        "notes": [
            "Latency p95 values are measured by smoke runners around extract job completion and DB validation.",
            "OCR confidence is required only when the OCR smoke report contains OCR-needed chunks.",
        ],
    }
    write_json(Path(args.output), payload)
    if args.baseline_output:
        baseline = {
            "candidate_snapshot": True,
            "notes": [
                "This file mirrors the current candidate retrieval metrics.",
                "Do not use it as an immutable baseline for promotion decisions.",
            ],
            "metrics": {
                "hit_at_10": metrics["hit_at_10"],
                "mrr_at_10": metrics["mrr_at_10"],
            },
        }
        write_json(Path(args.baseline_output), baseline)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx-report", default=str(DEFAULT_XLSX_REPORT))
    parser.add_argument("--pdf-report", default=str(DEFAULT_PDF_REPORT))
    parser.add_argument("--retrieval-report", default=str(DEFAULT_RETRIEVAL_REPORT))
    parser.add_argument("--ocr-report", default=str(DEFAULT_OCR_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--baseline-output")
    parser.add_argument("--candidate-index-version", default="rag-ingestion-v2-candidate")
    parser.add_argument("--baseline-index-version", default="rag-ingestion-v2-baseline")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def float_metric(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None and value != "":
            return float(value)
    return default


def int_metric(metrics: dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None or value == "":
        return 0
    return int(value)


def required_int_metric(
    missing: list[str],
    metrics: dict[str, Any],
    key: str,
    source_name: str,
) -> int:
    value = metrics.get(key)
    if value is None or value == "":
        missing.append(f"{source_name}.{key}")
        return 0
    return int(value)


def required_float_metric(
    missing: list[str],
    metrics: dict[str, Any],
    key: str,
    source_name: str,
) -> float:
    value = metrics.get(key)
    if value is None or value == "":
        missing.append(f"{source_name}.{key}")
        return 0.0
    return float(value)


def _required_float_source_metric(
    missing: list[str],
    key: str,
    *sources: tuple[str, dict[str, Any]],
) -> float:
    values = []
    for source_name, source in sources:
        if source.get(key) in {None, ""}:
            missing.append(f"{source_name}.{key}")
        else:
            values.append(source[key])
    if values:
        return max(float(value) for value in values)
    return 1.0


def _required_int_source_metric(
    missing: list[str],
    key: str,
    *sources: tuple[str, dict[str, Any]],
) -> int:
    values = []
    for source_name, source in sources:
        if source.get(key) in {None, ""}:
            missing.append(f"{source_name}.{key}")
        else:
            values.append(source[key])
    if values:
        return sum(int(value) for value in values)
    return 1


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
