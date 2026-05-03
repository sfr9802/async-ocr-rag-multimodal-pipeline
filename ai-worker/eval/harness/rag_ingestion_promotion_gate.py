"""Offline promotion gate for RAG ingestion candidate indexes.

The gate is deliberately deterministic and LLM-free. It consumes an aggregate
metrics payload from smoke/eval runners and returns a PASSED/BLOCKED decision
with blocking reasons that can later be persisted by API code.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DECISION_PASSED = "PASSED"
DECISION_BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class PromotionThresholds:
    parser_success_rate: float = 0.95
    unsupported_file_rate: float = 0.05
    zero_indexable_chunk_count: int = 0
    required_metadata_completeness: float = 0.98
    missing_required_metadata_count: int = 0
    xlsx_citation_location_accuracy: float = 0.90
    pdf_citation_location_accuracy: float = 0.85
    table_detection_accuracy: float = 0.80
    ocr_confidence_avg: float = 0.75
    hit_at_10_baseline_delta: float = -0.05
    mrr_at_10_baseline_delta: float = -0.05
    citation_accuracy: float = 0.85
    parsing_latency_p95: float = 30.0
    indexing_latency_p95: float = 60.0
    fatal_warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_success_rate": {"op": ">=", "value": self.parser_success_rate},
            "unsupported_file_rate": {"op": "<=", "value": self.unsupported_file_rate},
            "zero_indexable_chunk_count": {"op": "==", "value": self.zero_indexable_chunk_count},
            "required_metadata_completeness": {
                "op": ">=",
                "value": self.required_metadata_completeness,
            },
            "missing_required_metadata_count": {
                "op": "==",
                "value": self.missing_required_metadata_count,
            },
            "xlsx_citation_location_accuracy": {
                "op": ">=",
                "value": self.xlsx_citation_location_accuracy,
            },
            "pdf_citation_location_accuracy": {
                "op": ">=",
                "value": self.pdf_citation_location_accuracy,
            },
            "table_detection_accuracy": {"op": ">=", "value": self.table_detection_accuracy},
            "OCR_confidence_avg": {"op": ">=", "value": self.ocr_confidence_avg},
            "Hit@10": {"op": ">= baseline + delta", "value": self.hit_at_10_baseline_delta},
            "MRR@10": {"op": ">= baseline + delta", "value": self.mrr_at_10_baseline_delta},
            "citation_accuracy": {"op": ">=", "value": self.citation_accuracy},
            "parsing_latency_p95": {"op": "<=", "value": self.parsing_latency_p95},
            "indexing_latency_p95": {"op": "<=", "value": self.indexing_latency_p95},
            "fatal_warning_count": {"op": "==", "value": self.fatal_warning_count},
        }


@dataclass(frozen=True)
class PromotionGateResult:
    index_version: str
    decision: str
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "decision": self.decision,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "baseline_metrics": dict(self.baseline_metrics),
            "reasons": list(self.reasons),
        }


def evaluate_promotion_gate(
    *,
    index_version: str,
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None = None,
    thresholds: PromotionThresholds | None = None,
) -> PromotionGateResult:
    """Return a promotion decision for a candidate index version."""

    active_thresholds = thresholds or PromotionThresholds()
    reasons: list[str] = []
    baseline = baseline_metrics or {}
    if baseline.get("_baseline_candidate_snapshot"):
        reasons.append("baseline report must be an immutable baseline, not a candidate snapshot")

    _check_min(
        reasons,
        metrics,
        ("parser_success_rate",),
        "parser_success_rate",
        active_thresholds.parser_success_rate,
    )
    _check_max(
        reasons,
        metrics,
        ("unsupported_file_rate",),
        "unsupported_file_rate",
        active_thresholds.unsupported_file_rate,
    )
    _check_exact_int(
        reasons,
        metrics,
        "zero_indexable_chunk_count",
        active_thresholds.zero_indexable_chunk_count,
    )
    _check_min(
        reasons,
        metrics,
        ("required_metadata_completeness",),
        "required_metadata_completeness",
        active_thresholds.required_metadata_completeness,
    )

    _check_exact_int(
        reasons,
        metrics,
        "missing_required_metadata_count",
        active_thresholds.missing_required_metadata_count,
    )

    _check_min(
        reasons,
        metrics,
        ("xlsx_citation_location_accuracy",),
        "xlsx_citation_location_accuracy",
        active_thresholds.xlsx_citation_location_accuracy,
    )
    _check_min(
        reasons,
        metrics,
        ("pdf_citation_location_accuracy",),
        "pdf_citation_location_accuracy",
        active_thresholds.pdf_citation_location_accuracy,
    )
    _check_min(
        reasons,
        metrics,
        ("table_detection_accuracy",),
        "table_detection_accuracy",
        active_thresholds.table_detection_accuracy,
    )
    if _requires_ocr_confidence(metrics):
        _check_min(
            reasons,
            metrics,
            ("OCR_confidence_avg", "ocr_confidence_avg"),
            "OCR_confidence_avg",
            active_thresholds.ocr_confidence_avg,
            missing_reason="OCR_confidence_avg is required when OCR-needed bucket is present",
        )
    _check_baseline_delta(
        reasons,
        metrics,
        baseline,
        ("Hit@10", "hit_at_10"),
        ("Hit@10", "hit_at_10"),
        "Hit@10",
        active_thresholds.hit_at_10_baseline_delta,
    )
    _check_baseline_delta(
        reasons,
        metrics,
        baseline,
        ("MRR@10", "mrr_at_10"),
        ("MRR@10", "mrr_at_10"),
        "MRR@10",
        active_thresholds.mrr_at_10_baseline_delta,
    )
    _check_min(
        reasons,
        metrics,
        ("citation_accuracy", "citation_location_accuracy"),
        "citation_accuracy",
        active_thresholds.citation_accuracy,
    )
    _check_max(
        reasons,
        metrics,
        ("parsing_latency_p95", "parsing_latency_p95_seconds"),
        "parsing_latency_p95",
        active_thresholds.parsing_latency_p95,
    )
    _check_max(
        reasons,
        metrics,
        ("indexing_latency_p95", "indexing_latency_p95_seconds"),
        "indexing_latency_p95",
        active_thresholds.indexing_latency_p95,
    )
    _check_exact_int(reasons, metrics, "fatal_warning_count", active_thresholds.fatal_warning_count)

    _check_exact_int(reasons, metrics, "hidden_content_leakage_count", 0)

    if metrics.get("embedding_filtered_eval") is not True:
        reasons.append("retrieval eval must require embedded SearchUnit results")
    required_embedding_status = str(metrics.get("required_embedding_status") or "").upper()
    if required_embedding_status != "EMBEDDED":
        reasons.append("required_embedding_status must be EMBEDDED")
    required_index_version = _string_metric(metrics, "required_index_version")
    if not required_index_version:
        reasons.append("required_index_version is required for promotion")
    elif required_index_version != index_version:
        reasons.append("required_index_version must match promoted index_version")

    _check_exact_int(reasons, metrics, "gate_input_missing_count", 0)
    _check_exact_int(reasons, metrics, "indexing_filtered_hit_count", 0)
    _check_exact_int(reasons, metrics, "candidate_index_mismatch_count", 0)
    _check_exact_int(reasons, metrics, "required_index_version_mismatch_count", 0)
    _check_exact_int(reasons, metrics, "embedding_status_mismatch_count", 0)

    decision = DECISION_BLOCKED if reasons else DECISION_PASSED
    return PromotionGateResult(
        index_version=index_version,
        decision=decision,
        metrics=dict(metrics),
        thresholds=active_thresholds.to_dict(),
        baseline_metrics=dict(baseline),
        reasons=reasons,
    )


def build_failure_reason_payload(
    result: PromotionGateResult,
    *,
    eval_result_id: str | None = None,
    gate_report_path: str | None = None,
    retrieval_report_path: str | None = None,
) -> dict[str, Any]:
    """Return structured gate failure detail for report-only persistence."""

    overall_counts = _mapping_metric(result.metrics, "overall_failure_reason_counts")
    bucket_counts = _mapping_metric(result.metrics, "bucket_failure_reason_counts")
    return {
        "metric_threshold_failures": list(result.reasons),
        "bucket_level_failures": _bucket_level_failures(result.metrics),
        "failure_reason_distribution": {
            "overall": overall_counts,
            "by_bucket": bucket_counts,
        },
        "eval_result_id": eval_result_id,
        "retrieval_report_path": retrieval_report_path,
        "gate_report_path": gate_report_path,
        "candidate_index_version": _string_metric(result.metrics, "candidate_index_version") or result.index_version,
        "baseline_index_version": _string_metric(result.metrics, "baseline_index_version"),
    }


def _bucket_level_failures(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    bucket_metrics = _mapping_metric(metrics, "bucket_metrics")
    bucket_counts = _mapping_metric(metrics, "bucket_failure_reason_counts")
    failures: list[dict[str, Any]] = []
    for bucket in sorted(set(bucket_metrics) | set(bucket_counts)):
        reason_counts = bucket_counts.get(bucket) if isinstance(bucket_counts.get(bucket), Mapping) else {}
        bucket_payload = bucket_metrics.get(bucket) if isinstance(bucket_metrics.get(bucket), Mapping) else {}
        if reason_counts or bucket_payload.get("bucket_failure_reason_counts"):
            failures.append({
                "bucket": bucket,
                "failure_reason_counts": dict(reason_counts or bucket_payload.get("bucket_failure_reason_counts") or {}),
                "metrics": dict(bucket_payload),
            })
    return failures


def _mapping_metric(metrics: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = metrics.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _string_metric(metrics: Mapping[str, Any], key: str) -> str | None:
    value = metrics.get(key)
    if value is None or value == "":
        return None
    return str(value)


def persist_eval_result(
    connection: Any,
    *,
    dataset_id: str,
    result: PromotionGateResult,
    dataset_name: str | None = None,
    dataset_version: str = "report-only",
    baseline_index_version: str | None = None,
    report_path: str | None = None,
    report_uri: str | None = None,
    eval_result_id: str | None = None,
    created_at: datetime | None = None,
    commit: bool = True,
) -> str:
    """Insert a report-only promotion gate result into eval_result.

    The function accepts a DB-API compatible connection so tests can use a fake
    cursor and production callers can pass psycopg/psycopg2 connections without
    adding a hard dependency to the eval harness.
    """

    result_id = eval_result_id or f"eval_result_{uuid.uuid4().hex}"
    timestamp = created_at or datetime.now(timezone.utc)
    dataset_sql = """
INSERT INTO eval_dataset (
    id,
    name,
    version,
    created_at
) VALUES (
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (id) DO NOTHING
"""
    sql = """
INSERT INTO eval_result (
    id,
    dataset_id,
    eval_dataset_id,
    index_version,
    candidate_index_version,
    baseline_index_version,
    metrics_json,
    threshold_json,
    failure_reason_json,
    passed,
    status,
    report_path,
    report_uri,
    created_at
) VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s::jsonb,
    %s::jsonb,
    %s::jsonb,
    %s,
    %s,
    %s,
    %s,
    %s
)
"""
    params = (
        result_id,
        dataset_id,
        dataset_id,
        result.index_version,
        result.index_version,
        baseline_index_version,
        json.dumps(result.metrics, ensure_ascii=False, sort_keys=True),
        json.dumps(result.thresholds, ensure_ascii=False, sort_keys=True),
        json.dumps(
            build_failure_reason_payload(
                result,
                eval_result_id=result_id,
                gate_report_path=report_path,
                retrieval_report_path=_string_metric(result.metrics, "retrieval_report_path"),
            ),
            ensure_ascii=False,
            sort_keys=True,
        ),
        result.decision == DECISION_PASSED,
        result.decision,
        report_path,
        report_uri,
        timestamp,
    )
    cursor = connection.cursor()
    cursor.execute(dataset_sql, (
        dataset_id,
        dataset_name or dataset_id,
        dataset_version,
        timestamp,
    ))
    cursor.execute(sql, params)
    if commit and hasattr(connection, "commit"):
        connection.commit()
    return result_id


def persist_eval_result_from_dsn(
    dsn: str,
    *,
    dataset_id: str,
    result: PromotionGateResult,
    dataset_name: str | None = None,
    dataset_version: str = "report-only",
    baseline_index_version: str | None = None,
    report_path: str | None = None,
    report_uri: str | None = None,
    eval_result_id: str | None = None,
) -> str:
    connect = _load_psycopg_connect()
    with connect(dsn) as connection:
        return persist_eval_result(
            connection,
            dataset_id=dataset_id,
            result=result,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            baseline_index_version=baseline_index_version,
            report_path=report_path,
            report_uri=report_uri,
            eval_result_id=eval_result_id,
        )


def load_report_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        merged = dict(metrics)
        bucket_metrics = payload.get("bucket_metrics")
        if isinstance(bucket_metrics, Mapping):
            merged["bucket_metrics"] = dict(bucket_metrics)
        return merged
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"Report does not contain a metrics object: {path}")


def load_baseline_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = load_report_metrics(path)
    if isinstance(payload, Mapping) and payload.get("candidate_snapshot"):
        metrics["_baseline_candidate_snapshot"] = True
    return metrics


def write_gate_report(path: Path, result: PromotionGateResult, *, extra: Mapping[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["failure_reason_json"] = build_failure_reason_payload(
        result,
        eval_result_id=(extra or {}).get("eval_result_id") if extra else None,
        gate_report_path=str(path),
        retrieval_report_path=_string_metric(result.metrics, "retrieval_report_path"),
    )
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_report(payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-report", required=True, help="JSON report containing metrics or a metrics object")
    parser.add_argument("--baseline-report", help="Optional baseline JSON report")
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--gate-report", default="reports/rag_ingestion_promotion_gate_report.json")
    parser.add_argument("--db-dsn", help="Optional DB DSN for report-only eval_result persistence")
    parser.add_argument("--eval-dataset-id", default="rag-ingestion-gold-v0")
    parser.add_argument("--eval-dataset-name", default="rag-ingestion-gold-v0")
    parser.add_argument("--eval-dataset-version", default="v0")
    parser.add_argument("--baseline-index-version")
    parser.add_argument("--report-uri")
    parser.add_argument("--eval-result-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate_report_path = Path(args.gate_report)
    result = evaluate_promotion_gate(
        index_version=args.index_version,
        metrics=load_report_metrics(Path(args.metrics_report)),
        baseline_metrics=load_baseline_metrics(Path(args.baseline_report)) if args.baseline_report else None,
    )
    eval_result_id = None
    if args.db_dsn:
        eval_result_id = persist_eval_result_from_dsn(
            args.db_dsn,
            dataset_id=args.eval_dataset_id,
            result=result,
            dataset_name=args.eval_dataset_name,
            dataset_version=args.eval_dataset_version,
            baseline_index_version=args.baseline_index_version,
            report_path=str(gate_report_path),
            report_uri=args.report_uri,
            eval_result_id=args.eval_result_id,
        )
    payload = result.to_dict() | {
        "gate_report": str(gate_report_path),
        "eval_result_id": eval_result_id,
    }
    write_gate_report(
        gate_report_path,
        result,
        extra={
            "gate_report": str(gate_report_path),
            "eval_result_id": eval_result_id,
        },
    )
    print_report(payload)
    return 0 if result.decision == DECISION_PASSED else 2


def _check_min(
    reasons: list[str],
    metrics: Mapping[str, Any],
    keys: tuple[str, ...],
    display_key: str,
    minimum: float,
    *,
    missing_reason: str | None = None,
) -> None:
    value = _first_float_metric(metrics, keys)
    if value is None:
        reasons.append(missing_reason or f"{display_key} is required")
    elif value < minimum:
        reasons.append(f"{display_key} must be >= {minimum:.2f}")


def _check_max(
    reasons: list[str],
    metrics: Mapping[str, Any],
    keys: tuple[str, ...],
    display_key: str,
    maximum: float,
) -> None:
    value = _first_float_metric(metrics, keys)
    if value is None:
        reasons.append(f"{display_key} is required")
    elif value > maximum:
        reasons.append(f"{display_key} must be <= {maximum:.2f}")


def _check_exact_int(
    reasons: list[str],
    metrics: Mapping[str, Any],
    key: str,
    expected: int,
) -> None:
    value = _int_metric(metrics, key)
    if value is None:
        reasons.append(f"{key} is required")
    elif value != expected:
        reasons.append(f"{key} must be {expected}")


def _check_baseline_delta(
    reasons: list[str],
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    metric_keys: tuple[str, ...],
    baseline_keys: tuple[str, ...],
    display_key: str,
    delta: float,
) -> None:
    value = _first_float_metric(metrics, metric_keys)
    baseline_value = _first_float_metric(baseline_metrics, baseline_keys)
    if value is None:
        reasons.append(f"{display_key} is required")
        return
    if baseline_value is None:
        reasons.append(f"baseline {display_key} is required")
        return
    minimum = baseline_value + delta
    if value < minimum:
        reasons.append(f"{display_key} must be >= {_delta_label(delta)} ({minimum:.3f})")


def _delta_label(delta: float) -> str:
    if delta < 0:
        return f"baseline - {abs(delta):.2f}"
    if delta > 0:
        return f"baseline + {delta:.2f}"
    return "baseline"


def _requires_ocr_confidence(metrics: Mapping[str, Any]) -> bool:
    if _first_raw_metric(metrics, ("OCR_confidence_avg", "ocr_confidence_avg")) not in (None, ""):
        return True
    for key in (
        "OCR_needed_count",
        "ocr_needed_count",
        "ocr_needed_bucket_count",
        "ocr_bucket_count",
        "ocr_used_count",
        "pdf_ocr_noise_count",
        "pdf_ocr_noise_query_count",
    ):
        value = _float_metric(metrics, key)
        if value is not None and value > 0:
            return True
    bucket_metrics = metrics.get("bucket_metrics")
    if isinstance(bucket_metrics, Mapping):
        ocr_bucket = bucket_metrics.get("pdf_ocr_noise")
        if isinstance(ocr_bucket, Mapping):
            count = _float_metric(ocr_bucket, "count")
            if count is not None and count > 0:
                return True
    return False


def _first_raw_metric(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None and value != "":
            return value
    return None


def _first_float_metric(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float_metric(metrics, key)
        if value is not None:
            return value
    return None


def _float_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _int_metric(metrics: Mapping[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _load_psycopg_connect() -> Any:
    try:
        import psycopg  # type: ignore[import-not-found]

        return psycopg.connect
    except ModuleNotFoundError:
        try:
            import psycopg2  # type: ignore[import-not-found]

            return psycopg2.connect
        except ModuleNotFoundError as exc:
            raise RuntimeError("psycopg or psycopg2 is required for DSN persistence") from exc


__all__ = [
    "DECISION_BLOCKED",
    "DECISION_PASSED",
    "PromotionGateResult",
    "PromotionThresholds",
    "evaluate_promotion_gate",
    "load_report_metrics",
    "persist_eval_result",
    "persist_eval_result_from_dsn",
    "write_gate_report",
]


if __name__ == "__main__":
    sys.exit(main())
