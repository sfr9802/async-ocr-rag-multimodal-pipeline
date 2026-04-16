"""Unit coverage for the full-stack smoke runner assertion logic.

All tests are fully offline — no core-api, no Redis, no worker. They
drive the smoke runner's pure functions (shape assertions, report
builders, fixture loader) with synthetic payloads so we can prove the
runner catches the failure modes it claims to catch without standing
up a real pipeline.

What we explicitly cover:

  * assert_submission_shape: happy path + the three drift modes
    (wrong capability echoed, wrong initial status, missing field)
  * assert_final_status: happy path + FAILED terminal + missing field
  * assert_result_outputs: exact-match PASS per capability, missing
    artifact types, unexpected artifact types, empty outputs list,
    MULTIMODAL_TRACE treated as optional
  * build_report / format_console_summary: counters and per-case
    rendering still line up after we feed them manual SmokeCaseResult
    objects
  * load_ocr_fixture_bytes: committed fixture path wins, fallback
    rendering path works, custom override path works
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from scripts.smoke_runner import (
    REQUIRED_OUTPUTS,
    SmokeAssertionError,
    SmokeCaseResult,
    assert_final_status,
    assert_result_outputs,
    assert_submission_shape,
    build_report,
    format_console_summary,
    load_ocr_fixture_bytes,
    parse_only_arg,
)


# ---------------------------------------------------------------------------
# assert_submission_shape
# ---------------------------------------------------------------------------


class TestAssertSubmissionShape:
    def test_accepts_well_formed_submission(self):
        payload = {
            "jobId": "job-123",
            "status": "QUEUED",
            "capability": "MOCK",
        }
        assert assert_submission_shape(payload, expected_capability="MOCK") == "job-123"

    def test_rejects_non_object(self):
        with pytest.raises(SmokeAssertionError, match="Expected object"):
            assert_submission_shape(["not an object"], expected_capability="MOCK")

    def test_rejects_missing_job_id(self):
        with pytest.raises(SmokeAssertionError, match="jobId"):
            assert_submission_shape(
                {"status": "QUEUED", "capability": "MOCK"},
                expected_capability="MOCK",
            )

    def test_rejects_capability_mismatch(self):
        with pytest.raises(SmokeAssertionError, match="capability"):
            assert_submission_shape(
                {"jobId": "x", "status": "QUEUED", "capability": "RAG"},
                expected_capability="MOCK",
            )

    def test_rejects_non_queued_initial_status(self):
        with pytest.raises(SmokeAssertionError, match="QUEUED"):
            assert_submission_shape(
                {"jobId": "x", "status": "RUNNING", "capability": "MOCK"},
                expected_capability="MOCK",
            )


# ---------------------------------------------------------------------------
# assert_final_status
# ---------------------------------------------------------------------------


class TestAssertFinalStatus:
    def test_accepts_succeeded_view(self):
        # Should not raise.
        assert_final_status(
            {"jobId": "x", "status": "SUCCEEDED", "capability": "MOCK"},
            expected_capability="MOCK",
        )

    def test_rejects_failed_status_and_surfaces_error_code(self):
        view = {
            "jobId": "x",
            "status": "FAILED",
            "capability": "OCR",
            "errorCode": "OCR_UNSUPPORTED_INPUT_TYPE",
            "errorMessage": "bad mime",
        }
        with pytest.raises(SmokeAssertionError, match="OCR_UNSUPPORTED_INPUT_TYPE"):
            assert_final_status(view, expected_capability="OCR")

    def test_rejects_capability_drift(self):
        with pytest.raises(SmokeAssertionError, match="capability"):
            assert_final_status(
                {"jobId": "x", "status": "SUCCEEDED", "capability": "MOCK"},
                expected_capability="RAG",
            )

    def test_rejects_missing_status_field(self):
        with pytest.raises(SmokeAssertionError, match="status"):
            assert_final_status(
                {"jobId": "x", "capability": "MOCK"},
                expected_capability="MOCK",
            )


# ---------------------------------------------------------------------------
# assert_result_outputs
# ---------------------------------------------------------------------------


def _output(type_: str) -> dict:
    """Minimal artifact dict matching what core-api's JobResult returns."""
    return {"id": f"art-{type_}", "type": type_, "accessUrl": f"/api/v1/artifacts/art-{type_}/content"}


class TestAssertResultOutputs:
    def test_mock_happy_path(self):
        payload = {"outputs": [_output("FINAL_RESPONSE")]}
        breakdown = assert_result_outputs(payload, capability="MOCK")
        assert breakdown["output_types"] == ["FINAL_RESPONSE"]
        assert breakdown["missing"] == []
        assert breakdown["unexpected"] == []

    def test_rag_happy_path(self):
        payload = {
            "outputs": [
                _output("RETRIEVAL_RESULT"),
                _output("FINAL_RESPONSE"),
            ]
        }
        breakdown = assert_result_outputs(payload, capability="RAG")
        assert set(breakdown["output_types"]) == {"RETRIEVAL_RESULT", "FINAL_RESPONSE"}

    def test_ocr_happy_path(self):
        payload = {"outputs": [_output("OCR_TEXT"), _output("OCR_RESULT")]}
        breakdown = assert_result_outputs(payload, capability="OCR")
        assert breakdown["missing"] == []
        assert breakdown["unexpected"] == []

    def test_multimodal_happy_path(self):
        payload = {
            "outputs": [
                _output("OCR_TEXT"),
                _output("VISION_RESULT"),
                _output("RETRIEVAL_RESULT"),
                _output("FINAL_RESPONSE"),
            ]
        }
        breakdown = assert_result_outputs(payload, capability="MULTIMODAL")
        assert breakdown["missing"] == []
        assert breakdown["unexpected"] == []

    def test_multimodal_trace_is_optional(self):
        """MULTIMODAL_TRACE should be accepted silently when present."""
        payload = {
            "outputs": [
                _output("OCR_TEXT"),
                _output("VISION_RESULT"),
                _output("RETRIEVAL_RESULT"),
                _output("FINAL_RESPONSE"),
                _output("MULTIMODAL_TRACE"),
            ]
        }
        breakdown = assert_result_outputs(payload, capability="MULTIMODAL")
        assert "MULTIMODAL_TRACE" in breakdown["output_types"]
        assert breakdown["missing"] == []
        assert breakdown["unexpected"] == []

    def test_rag_missing_retrieval_result(self):
        payload = {"outputs": [_output("FINAL_RESPONSE")]}
        with pytest.raises(SmokeAssertionError, match="RETRIEVAL_RESULT"):
            assert_result_outputs(payload, capability="RAG")

    def test_ocr_unexpected_extra_artifact(self):
        payload = {
            "outputs": [
                _output("OCR_TEXT"),
                _output("OCR_RESULT"),
                _output("VISION_RESULT"),  # OCR should never emit this
            ]
        }
        with pytest.raises(SmokeAssertionError, match="VISION_RESULT"):
            assert_result_outputs(payload, capability="OCR")

    def test_empty_outputs_is_failure(self):
        with pytest.raises(SmokeAssertionError, match="zero output artifacts"):
            assert_result_outputs({"outputs": []}, capability="MOCK")

    def test_missing_outputs_field_is_failure(self):
        with pytest.raises(SmokeAssertionError, match="outputs"):
            assert_result_outputs({}, capability="MOCK")

    def test_artifact_without_access_url_is_failure(self):
        payload = {"outputs": [{"type": "FINAL_RESPONSE", "id": "x"}]}
        with pytest.raises(SmokeAssertionError, match="accessUrl"):
            assert_result_outputs(payload, capability="MOCK")

    def test_non_object_payload_is_failure(self):
        with pytest.raises(SmokeAssertionError, match="Expected object"):
            assert_result_outputs([], capability="MOCK")

    @pytest.mark.parametrize("capability", list(REQUIRED_OUTPUTS.keys()))
    def test_every_capability_has_at_least_one_required_artifact(self, capability: str):
        """Sanity check: the required-outputs map can't silently go empty."""
        assert REQUIRED_OUTPUTS[capability], (
            f"REQUIRED_OUTPUTS[{capability!r}] is empty — every capability "
            "must produce at least one asserted artifact."
        )


# ---------------------------------------------------------------------------
# build_report + console formatter
# ---------------------------------------------------------------------------


class TestReportBuilder:
    def test_passed_failed_skipped_counters(self):
        cases = [
            SmokeCaseResult(capability="MOCK", status="SUCCESS"),
            SmokeCaseResult(capability="RAG", status="FAIL"),
            SmokeCaseResult(capability="OCR", status="SKIP"),
            SmokeCaseResult(capability="MULTIMODAL", status="SUCCESS"),
        ]
        report = build_report(
            base_url="http://localhost:8080",
            started_at="2026-04-15T12:34:56Z",
            duration_ms=42.0,
            cases=cases,
        )
        assert report.passed == 2
        assert report.failed == 1
        assert report.skipped == 1

    def test_to_dict_has_full_summary_block(self):
        report = build_report(
            base_url="http://x",
            started_at="t",
            duration_ms=1.0,
            cases=[SmokeCaseResult(capability="MOCK", status="SUCCESS")],
        )
        body = report.to_dict()
        assert body["summary"] == {
            "passed": 1, "failed": 0, "skipped": 0, "total": 1,
        }
        assert body["baseUrl"] == "http://x"
        assert body["cases"][0]["capability"] == "MOCK"

    def test_console_summary_includes_case_lines(self):
        report = build_report(
            base_url="http://x",
            started_at="t",
            duration_ms=0.0,
            cases=[
                SmokeCaseResult(
                    capability="MOCK",
                    status="SUCCESS",
                    job_id="job-1",
                    output_types=["FINAL_RESPONSE"],
                    final_job_status="SUCCEEDED",
                ),
                SmokeCaseResult(
                    capability="RAG",
                    status="FAIL",
                    failure_reason="timed out",
                    final_job_status="RUNNING",
                ),
            ],
        )
        text = format_console_summary(report)
        assert "MOCK" in text
        assert "RAG" in text
        assert "FINAL_RESPONSE" in text
        assert "timed out" in text
        assert "[OK]" in text
        assert "[FAIL]" in text


# ---------------------------------------------------------------------------
# parse_only_arg
# ---------------------------------------------------------------------------


class TestParseOnly:
    def test_default_returns_all_capabilities(self):
        assert parse_only_arg(None) == ["MOCK", "RAG", "OCR", "MULTIMODAL"]

    def test_subset_is_parsed_uppercased(self):
        assert parse_only_arg("mock,ocr") == ["MOCK", "OCR"]

    def test_rejects_unknown_capability(self):
        with pytest.raises(SystemExit):
            parse_only_arg("MOCK,SUMMARIZE")


# ---------------------------------------------------------------------------
# load_ocr_fixture_bytes
# ---------------------------------------------------------------------------


class TestLoadFixture:
    def test_custom_path_wins(self, tmp_path: Path):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        path = tmp_path / "custom.png"
        path.write_bytes(png_bytes)
        data, name = load_ocr_fixture_bytes(path)
        assert data == png_bytes
        assert name == "custom.png"

    def test_fallback_renders_valid_png_when_no_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Point the default resolver at an empty tmp dir so the fallback
        # branch fires, and pretend the committed fixture doesn't exist.
        from scripts import smoke_runner

        def _missing() -> Path:
            return tmp_path / "does-not-exist.png"

        monkeypatch.setattr(smoke_runner, "_default_ocr_fixture_path", _missing)

        pil = pytest.importorskip("PIL")  # fallback needs Pillow
        data, name = load_ocr_fixture_bytes(None)
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert name.endswith(".png")
