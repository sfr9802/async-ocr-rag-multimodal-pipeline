"""Unit coverage for the normalized pipeline trace module.

These tests exercise the `PipelineTrace` / `StageRecord` / `TraceBuilder`
types directly — no capabilities, no providers, no HTTP. They cover:

  * TraceBuilder records stages in the order they're appended
  * to_dict / StageRecord.to_dict serialize to the camelCase wire form
  * finalize_ok / finalize_partial / finalize_failed set final_status
  * summary() format is stable across ok / warn / fail / skipped /
    fallback stages
  * _clip_message / _clip_details bound oversize values so trace
    payloads stay compact even when a provider returns a huge blob
  * STABLE_ERROR_CODES is a non-empty dict whose keys are all
    UPPER_SNAKE_CASE (enforced so the docs table stays inspectable)
  * elapsed_ms returns a plausible non-negative float
"""

from __future__ import annotations

import re
import time

import pytest

from app.capabilities.trace import (
    FINAL_FAILED,
    FINAL_OK,
    FINAL_PARTIAL,
    INPUT_KIND_IMAGE,
    STABLE_ERROR_CODES,
    STAGE_CLASSIFY,
    STAGE_FUSION,
    STAGE_GENERATE,
    STAGE_OCR,
    STAGE_RETRIEVE,
    STAGE_VISION,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_WARN,
    TRACE_SCHEMA_VERSION,
    PipelineTrace,
    StageRecord,
    TraceBuilder,
    _clip_details,
    _clip_message,
    elapsed_ms,
)


# ---------------------------------------------------------------------------
# StageRecord
# ---------------------------------------------------------------------------


class TestStageRecord:
    def test_to_dict_has_stable_camelcase_keys(self):
        rec = StageRecord(
            stage=STAGE_OCR,
            provider="tesseract-5.3.3",
            status=STATUS_OK,
            duration_ms=12.345,
            details={"pageCount": 1, "textLength": 42},
        )
        out = rec.to_dict()
        assert set(out.keys()) == {
            "stage", "provider", "status", "code", "message",
            "retryable", "fallbackUsed", "durationMs", "details",
        }
        assert out["stage"] == "ocr"
        assert out["provider"] == "tesseract-5.3.3"
        assert out["status"] == "ok"
        assert out["durationMs"] == 12.345
        assert out["fallbackUsed"] is False
        assert out["details"] == {"pageCount": 1, "textLength": 42}

    def test_to_dict_rounds_duration_ms_to_three_decimals(self):
        rec = StageRecord(
            stage="x", status=STATUS_OK, duration_ms=1.123456789
        )
        assert rec.to_dict()["durationMs"] == 1.123


# ---------------------------------------------------------------------------
# PipelineTrace
# ---------------------------------------------------------------------------


class TestPipelineTrace:
    def test_empty_trace_serializes_with_envelope_fields(self):
        trace = PipelineTrace(
            capability="OCR", input_kind=INPUT_KIND_IMAGE
        )
        out = trace.to_dict()
        assert out["schemaVersion"] == TRACE_SCHEMA_VERSION
        assert out["capability"] == "OCR"
        assert out["inputKind"] == "image"
        assert out["finalStatus"] == "ok"
        assert out["stages"] == []
        assert out["warnings"] == []
        assert out["summary"] == ""

    def test_summary_mixes_ok_warn_fail_skipped(self):
        trace = PipelineTrace(capability="MULTIMODAL", input_kind="image")
        trace.stages.append(StageRecord(
            stage=STAGE_CLASSIFY, status=STATUS_OK, duration_ms=0.5
        ))
        trace.stages.append(StageRecord(
            stage=STAGE_OCR,
            status=STATUS_WARN,
            code="OCR_EMPTY_TEXT",
            duration_ms=2.1,
            fallback_used=True,
        ))
        trace.stages.append(StageRecord(
            stage=STAGE_VISION,
            status=STATUS_FAIL,
            code="VISION_PROVIDER_TIMEOUT",
            duration_ms=5.0,
        ))
        trace.stages.append(StageRecord(
            stage=STAGE_FUSION, status=STATUS_SKIPPED
        ))
        summary = trace.summary()
        assert "classify:ok(0ms)" in summary
        assert "ocr:warn(OCR_EMPTY_TEXT" in summary
        assert "fallback" in summary
        assert "vision:fail(VISION_PROVIDER_TIMEOUT" in summary
        assert "fusion:skipped" in summary


# ---------------------------------------------------------------------------
# TraceBuilder
# ---------------------------------------------------------------------------


class TestTraceBuilder:
    def test_records_stages_in_order(self):
        b = TraceBuilder(capability="OCR", input_kind="image")
        b.record_ok(STAGE_CLASSIFY, duration_ms=0.1)
        b.record_ok(STAGE_OCR, provider="fake", duration_ms=1.0)
        trace = b.finalize_ok()
        assert [s.stage for s in trace.stages] == ["classify", "ocr"]
        assert trace.final_status == FINAL_OK

    def test_record_fail_marks_stage_and_preserves_fallback_flag(self):
        b = TraceBuilder(capability="MULTIMODAL", input_kind="image")
        b.record_fail(
            STAGE_VISION,
            provider="fake-vision",
            code="VISION_TIMEOUT",
            message="timed out after 3s",
            duration_ms=3000.0,
            retryable=True,
            fallback_used=True,
        )
        rec = b.trace.stages[0]
        assert rec.status == STATUS_FAIL
        assert rec.code == "VISION_TIMEOUT"
        assert rec.retryable is True
        assert rec.fallback_used is True

    def test_record_skipped_has_no_duration_or_provider(self):
        b = TraceBuilder(capability="MULTIMODAL", input_kind="image")
        b.record_skipped(STAGE_GENERATE, message="both providers failed")
        rec = b.trace.stages[0]
        assert rec.status == STATUS_SKIPPED
        assert rec.duration_ms == 0.0
        assert rec.provider is None

    def test_finalize_variants_set_final_status(self):
        b1 = TraceBuilder(capability="OCR", input_kind="image")
        assert b1.finalize_ok().final_status == FINAL_OK

        b2 = TraceBuilder(capability="OCR", input_kind="image")
        assert b2.finalize_partial().final_status == FINAL_PARTIAL

        b3 = TraceBuilder(capability="OCR", input_kind="image")
        assert b3.finalize_failed().final_status == FINAL_FAILED

    def test_add_warning_is_deduped_against_blanks(self):
        b = TraceBuilder(capability="OCR", input_kind="image")
        b.add_warning("something odd")
        b.add_warning("")  # should be ignored
        b.add_warning(None)  # type: ignore[arg-type] — defensive
        assert b.trace.warnings == ["something odd"]

    def test_summary_can_be_called_before_finalize(self):
        """Error-message formatting needs access to the summary string
        before finalize, so the builder exposes it pre-finalize too."""
        b = TraceBuilder(capability="MULTIMODAL", input_kind="image")
        b.record_ok(STAGE_CLASSIFY, duration_ms=0.5)
        b.record_fail(
            STAGE_OCR,
            provider="x",
            code="OCR_IMAGE_DECODE_FAILED",
            duration_ms=1.1,
        )
        s = b.summary()
        assert "classify:ok" in s
        assert "ocr:fail(OCR_IMAGE_DECODE_FAILED" in s


# ---------------------------------------------------------------------------
# clipping helpers
# ---------------------------------------------------------------------------


class TestClipping:
    def test_clip_message_handles_none(self):
        assert _clip_message(None) is None

    def test_clip_message_passes_through_short_text(self):
        assert _clip_message("short") == "short"

    def test_clip_message_truncates_long_text(self):
        result = _clip_message("x" * 400)
        assert len(result) <= 200
        assert result.endswith("...")

    def test_clip_details_passes_through_small_payload(self):
        payload = {"pageCount": 3, "textLength": 42, "model": "bge-m3"}
        assert _clip_details(payload) == payload

    def test_clip_details_truncates_oversize_string_values(self):
        big = "abcdef" * 200  # 1200 chars
        payload = {"preview": big, "pageCount": 1}
        out = _clip_details(payload)
        assert len(out["preview"]) <= 400
        assert out["preview"].endswith("...")
        # Non-string values pass through unchanged.
        assert out["pageCount"] == 1


# ---------------------------------------------------------------------------
# Stable error code registry
# ---------------------------------------------------------------------------


class TestStableErrorCodes:
    def test_registry_is_non_empty(self):
        assert len(STABLE_ERROR_CODES) > 0

    def test_every_key_is_upper_snake_case(self):
        pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
        for code in STABLE_ERROR_CODES:
            assert pattern.match(code), f"non-canonical code: {code}"

    def test_every_value_is_non_empty_string(self):
        for code, desc in STABLE_ERROR_CODES.items():
            assert isinstance(desc, str)
            assert desc, f"empty description for {code}"

    def test_includes_multimodal_terminal_codes(self):
        for expected in (
            "MULTIMODAL_ALL_PROVIDERS_FAILED",
            "MULTIMODAL_RETRIEVAL_FAILED",
            "MULTIMODAL_GENERATION_FAILED",
        ):
            assert expected in STABLE_ERROR_CODES

    def test_includes_ocr_provider_wrapped_codes(self):
        for expected in (
            "OCR_IMAGE_DECODE_FAILED",
            "OCR_PDF_OPEN_FAILED",
            "OCR_TOO_MANY_PAGES",
            "OCR_EMPTY_TEXT",
            "OCR_LOW_CONFIDENCE",
        ):
            assert expected in STABLE_ERROR_CODES


# ---------------------------------------------------------------------------
# elapsed_ms helper
# ---------------------------------------------------------------------------


class TestElapsedMs:
    def test_returns_non_negative_float(self):
        started = time.monotonic()
        value = elapsed_ms(started)
        assert isinstance(value, float)
        assert value >= 0

    def test_handles_fresh_monotonic_under_one_ms(self):
        """Calling elapsed_ms immediately after time.monotonic() should
        round-trip to a tiny value without flakiness."""
        value = elapsed_ms(time.monotonic())
        assert value >= 0
        assert value < 100  # far below 100ms, even on the slowest CI
