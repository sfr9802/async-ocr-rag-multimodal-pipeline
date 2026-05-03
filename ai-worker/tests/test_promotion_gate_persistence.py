from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ai-worker" / "eval" / "harness" / "rag_ingestion_promotion_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rag_ingestion_promotion_gate_for_persistence", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate_module = load_module()


def test_promotion_gate_passes_full_p1_threshold_set():
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=_passing_metrics(ocr_needed=True),
        baseline_metrics=_baseline_metrics(),
    )

    assert result.decision == "PASSED"
    assert result.reasons == []
    assert result.thresholds["parser_success_rate"]["value"] == 0.95
    assert result.thresholds["OCR_confidence_avg"]["value"] == 0.75


def test_promotion_gate_blocks_p1_threshold_regressions():
    metrics = _passing_metrics(ocr_needed=True) | {
        "unsupported_file_rate": 0.06,
        "OCR_confidence_avg": 0.74,
        "MRR@10": 0.64,
    }

    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=metrics,
        baseline_metrics=_baseline_metrics(),
    )

    assert result.decision == "BLOCKED"
    assert "unsupported_file_rate must be <= 0.05" in result.reasons
    assert "OCR_confidence_avg must be >= 0.75" in result.reasons
    assert "MRR@10 must be >= baseline - 0.05 (0.650)" in result.reasons


def test_promotion_gate_does_not_require_ocr_confidence_without_ocr_bucket():
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=_passing_metrics(ocr_needed=False),
        baseline_metrics=_baseline_metrics(),
    )

    assert result.decision == "PASSED"


def test_promotion_gate_rejects_candidate_snapshot_baseline():
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=_passing_metrics(ocr_needed=False),
        baseline_metrics=_baseline_metrics() | {"_baseline_candidate_snapshot": True},
    )

    assert result.decision == "BLOCKED"
    assert "baseline report must be an immutable baseline, not a candidate snapshot" in result.reasons


def test_promotion_gate_blocks_index_contract_mismatch_counts():
    metrics = _passing_metrics(ocr_needed=False) | {
        "indexing_filtered_hit_count": 1,
        "candidate_index_mismatch_count": 1,
        "required_index_version_mismatch_count": 1,
        "embedding_status_mismatch_count": 1,
    }

    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=metrics,
        baseline_metrics=_baseline_metrics(),
    )

    assert result.decision == "BLOCKED"
    assert "indexing_filtered_hit_count must be 0" in result.reasons
    assert "candidate_index_mismatch_count must be 0" in result.reasons
    assert "required_index_version_mismatch_count must be 0" in result.reasons
    assert "embedding_status_mismatch_count must be 0" in result.reasons


def test_promotion_gate_blocks_missing_gate_inputs():
    metrics = _passing_metrics(ocr_needed=False) | {"gate_input_missing_count": 2}

    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=metrics,
        baseline_metrics=_baseline_metrics(),
    )

    assert result.decision == "BLOCKED"
    assert "gate_input_missing_count must be 0" in result.reasons


def test_persist_eval_result_inserts_report_only_payload_with_status():
    metrics = _passing_metrics(ocr_needed=False) | {
        "fatal_warning_count": 1,
        "required_index_version": "candidate-blocked",
    }
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-blocked",
        metrics=metrics,
        baseline_metrics=_baseline_metrics(),
    )
    connection = FakeConnection()
    created_at = datetime(2026, 5, 3, 12, 30, tzinfo=timezone.utc)

    result_id = gate_module.persist_eval_result(
        connection,
        dataset_id="gold-v0",
        result=result,
        baseline_index_version="baseline-v1",
        report_path="reports/rag_gate.json",
        report_uri="s3://bucket/rag_gate.json",
        eval_result_id="eval-result-1",
        created_at=created_at,
    )

    assert result_id == "eval-result-1"
    assert connection.committed
    assert "INSERT INTO eval_dataset" in connection.cursor_obj.statements[0][0]
    assert "INSERT INTO eval_result" in connection.cursor_obj.sql
    assert "metrics_json" in connection.cursor_obj.sql
    assert "threshold_json" in connection.cursor_obj.sql
    assert "failure_reason_json" in connection.cursor_obj.sql

    params = connection.cursor_obj.params
    assert params[0] == "eval-result-1"
    assert params[1] == "gold-v0"
    assert params[3] == "candidate-blocked"
    assert params[5] == "baseline-v1"
    assert json.loads(params[6])["fatal_warning_count"] == 1
    assert json.loads(params[7])["citation_accuracy"]["value"] == 0.85
    failure_payload = json.loads(params[8])
    assert failure_payload["metric_threshold_failures"] == ["fatal_warning_count must be 0"]
    assert failure_payload["eval_result_id"] == "eval-result-1"
    assert failure_payload["retrieval_report_path"] is None
    assert failure_payload["candidate_index_version"] == "candidate-blocked"
    assert params[9] is False
    assert params[10] == "BLOCKED"
    assert params[11] == "reports/rag_gate.json"
    assert params[12] == "s3://bucket/rag_gate.json"
    assert params[13] == created_at


def test_write_gate_report_can_include_persisted_eval_result_id(tmp_path):
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=_passing_metrics(ocr_needed=False),
        baseline_metrics=_baseline_metrics(),
    )
    report_path = tmp_path / "gate.json"

    gate_module.write_gate_report(
        report_path,
        result,
        extra={
            "gate_report": str(report_path),
            "eval_result_id": "eval-result-1",
        },
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "PASSED"
    assert payload["eval_result_id"] == "eval-result-1"
    assert payload["gate_report"] == str(report_path)
    assert payload["failure_reason_json"]["eval_result_id"] == "eval-result-1"


def test_failure_reason_payload_includes_bucket_distribution():
    metrics = _passing_metrics(ocr_needed=False) | {
        "candidate_index_version": "candidate-v2",
        "retrieval_report_path": "reports/rag_retrieval_eval_report.json",
        "overall_failure_reason_counts": {"expected_range_not_found": 2},
        "bucket_failure_reason_counts": {"xlsx_lookup": {"expected_range_not_found": 2}},
        "bucket_metrics": {
            "xlsx_lookup": {
                "Hit@10": 0.5,
                "bucket_failure_reason_counts": {"expected_range_not_found": 2},
            }
        },
    }
    result = gate_module.evaluate_promotion_gate(
        index_version="candidate-v2",
        metrics=metrics,
        baseline_metrics=_baseline_metrics(),
    )

    payload = gate_module.build_failure_reason_payload(
        result,
        eval_result_id="eval-result-1",
        gate_report_path="reports/gate.json",
        retrieval_report_path=metrics["retrieval_report_path"],
    )

    assert payload["failure_reason_distribution"]["overall"]["expected_range_not_found"] == 2
    assert payload["bucket_level_failures"][0]["bucket"] == "xlsx_lookup"
    assert payload["retrieval_report_path"] == "reports/rag_retrieval_eval_report.json"


def _passing_metrics(*, ocr_needed: bool) -> dict[str, object]:
    metrics: dict[str, object] = {
        "parser_success_rate": 0.95,
        "unsupported_file_rate": 0.05,
        "zero_indexable_chunk_count": 0,
        "required_metadata_completeness": 0.98,
        "missing_required_metadata_count": 0,
        "xlsx_citation_location_accuracy": 0.90,
        "pdf_citation_location_accuracy": 0.85,
        "table_detection_accuracy": 0.80,
        "Hit@10": 0.75,
        "MRR@10": 0.65,
        "citation_accuracy": 0.85,
        "parsing_latency_p95": 30.0,
        "indexing_latency_p95": 60.0,
        "fatal_warning_count": 0,
        "hidden_content_leakage_count": 0,
        "embedding_filtered_eval": True,
        "required_embedding_status": "EMBEDDED",
        "required_index_version": "candidate-v2",
        "gate_input_missing_count": 0,
        "indexing_filtered_hit_count": 0,
        "candidate_index_mismatch_count": 0,
        "required_index_version_mismatch_count": 0,
        "embedding_status_mismatch_count": 0,
    }
    if ocr_needed:
        metrics["OCR_needed_count"] = 1
        metrics["OCR_confidence_avg"] = 0.75
    return metrics


def _baseline_metrics() -> dict[str, float]:
    return {"Hit@10": 0.80, "MRR@10": 0.70}


class FakeCursor:
    def __init__(self) -> None:
        self.sql = ""
        self.params = ()
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.sql = sql
        self.params = params
        self.statements.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True
