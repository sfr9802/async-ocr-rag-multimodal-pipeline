"""OCR capability tests.

All tests here are fully hermetic: they use a FakeOcrProvider instead of
Tesseract/PyMuPDF, which means the suite runs without the `tesseract`
binary, without PyMuPDF, and without any network / HuggingFace access.

The tests cover:

  1. Happy-path PNG image OCR → OCR_TEXT + OCR_RESULT artifacts with
     confidence propagated.
  2. Happy-path PDF OCR with multiple pages → page_count/text_length
     roll-ups correct, warnings aggregated.
  3. Unsupported input type rejected with a typed CapabilityError.
  4. Empty extraction path: zero characters still produces artifacts
     but adds a warning.
  5. Low-confidence path: avg below threshold adds a warning.
  6. Registry resilience: when OCR init raises, MOCK (and RAG, if
     enabled) still register.

The happy-path tests also verify that the OCR_TEXT output is ready to
be re-submitted as INPUT_TEXT for a later RAG job, which is the
contract the "OCR → RAG chaining" phase will build on.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from app.capabilities.base import (
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
)
from app.capabilities.ocr.capability import OcrCapability, OcrCapabilityConfig
from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrPageResult,
    OcrProvider,
)


# ---------------------------------------------------------------------------
# Fake provider + test fixtures.
# ---------------------------------------------------------------------------


class FakeOcrProvider(OcrProvider):
    """Canned OcrProvider used by all OCR capability tests.

    Constructor takes either an `image_result` or a `pdf_result` (or both)
    so individual tests can script exactly what the provider should
    return. Raising an `OcrError` is also supported via `raise_on`.
    """

    def __init__(
        self,
        *,
        image_result: Optional[OcrPageResult] = None,
        pdf_result: Optional[OcrDocumentResult] = None,
        raise_on: Optional[str] = None,
        raise_error: Optional[OcrError] = None,
    ) -> None:
        self._image_result = image_result
        self._pdf_result = pdf_result
        self._raise_on = raise_on
        self._raise_error = raise_error
        self.image_calls: List[bytes] = []
        self.pdf_calls: List[bytes] = []

    @property
    def name(self) -> str:
        return "fake-ocr-1.0"

    def ocr_image(self, image_bytes: bytes, *, mime_type=None) -> OcrPageResult:
        self.image_calls.append(image_bytes)
        if self._raise_on == "image":
            raise self._raise_error or OcrError("FAKE_IMG_FAIL", "fake image failure")
        assert self._image_result is not None, "test did not provide image_result"
        return self._image_result

    def ocr_pdf(self, pdf_bytes: bytes) -> OcrDocumentResult:
        self.pdf_calls.append(pdf_bytes)
        if self._raise_on == "pdf":
            raise self._raise_error or OcrError("FAKE_PDF_FAIL", "fake pdf failure")
        assert self._pdf_result is not None, "test did not provide pdf_result"
        return self._pdf_result


def _png_bytes() -> bytes:
    """Minimal valid PNG signature — enough to pass the capability's
    magic-byte sniff when content_type is unset. The fake provider
    never actually decodes this so 8 bytes is plenty."""
    return b"\x89PNG\r\n\x1a\n"


def _pdf_bytes() -> bytes:
    """Minimal PDF header so magic-byte classification succeeds."""
    return b"%PDF-1.4\n% fake\n"


def _make_job_input(
    *,
    job_id: str = "job-ocr-test",
    content: bytes,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    artifact_type: str = "INPUT_FILE",
) -> CapabilityInput:
    return CapabilityInput(
        job_id=job_id,
        capability="OCR",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="art-ocr-1",
                type=artifact_type,
                content=content,
                content_type=content_type,
                filename=filename,
            )
        ],
    )


def _find_output(result: CapabilityOutput, artifact_type: str):
    for artifact in result.outputs:
        if artifact.type == artifact_type:
            return artifact
    raise AssertionError(f"no {artifact_type} in outputs: {[o.type for o in result.outputs]}")


def _default_config() -> OcrCapabilityConfig:
    return OcrCapabilityConfig(min_confidence_warn=40.0, max_pages=100)


# ---------------------------------------------------------------------------
# 1. Happy-path image OCR.
# ---------------------------------------------------------------------------


def test_image_ocr_happy_path_emits_ocr_text_and_ocr_result():
    provider = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="Hello, OCR!\nSecond line of fake text.",
            avg_confidence=92.4,
            warnings=[],
        )
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(),
        content_type="image/png",
        filename="receipt.png",
    ))

    # Two output artifacts.
    types = {a.type for a in result.outputs}
    assert types == {"OCR_TEXT", "OCR_RESULT"}

    # OCR_TEXT is plain UTF-8 text — shape is identical to what a future
    # OCR→RAG chain will feed in as INPUT_TEXT.
    ocr_text = _find_output(result, "OCR_TEXT")
    assert ocr_text.content_type.startswith("text/plain")
    assert ocr_text.content.decode("utf-8") == "Hello, OCR!\nSecond line of fake text."

    # OCR_RESULT envelope has every required field populated.
    ocr_result = _find_output(result, "OCR_RESULT")
    body = json.loads(ocr_result.content.decode("utf-8"))
    assert body["filename"] == "receipt.png"
    assert body["mimeType"] == "image/png"
    assert body["kind"] == "image"
    assert body["engineName"] == "fake-ocr-1.0"
    assert body["pageCount"] == 1
    assert body["textLength"] == len("Hello, OCR!\nSecond line of fake text.")
    assert body["avgConfidence"] == 92.4
    assert body["warnings"] == []
    assert len(body["pages"]) == 1
    assert body["pages"][0]["pageNumber"] == 1
    assert body["pages"][0]["avgConfidence"] == 92.4

    # Provider saw the raw PNG bytes, unchanged.
    assert provider.image_calls == [_png_bytes()]
    assert provider.pdf_calls == []


def test_image_ocr_classifies_by_magic_bytes_when_content_type_missing():
    """Content_type may be absent in real uploads (e.g. curl without
    --header). The capability should still classify via PNG magic."""
    provider = FakeOcrProvider(
        image_result=OcrPageResult(page_number=1, text="ok", avg_confidence=80.0)
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(),
        content_type=None,
        filename=None,  # no filename either
    ))
    body = json.loads(_find_output(result, "OCR_RESULT").content)
    assert body["kind"] == "image"
    assert body["mimeType"] == "image/png"  # recovered from magic
    assert body["filename"] == "art-ocr-1.img"  # fallback from artifact_id


# ---------------------------------------------------------------------------
# 2. Happy-path PDF OCR (multi-page, mocked provider).
# ---------------------------------------------------------------------------


def test_pdf_ocr_multi_page_aggregates_pages_and_warnings():
    provider = FakeOcrProvider(
        pdf_result=OcrDocumentResult(
            pages=[
                OcrPageResult(
                    page_number=1,
                    text="Page one text with decent confidence.",
                    avg_confidence=88.0,
                    warnings=[],
                ),
                OcrPageResult(
                    page_number=2,
                    text="Page two, rasterized scan.",
                    avg_confidence=72.5,
                    warnings=["page 2: no text layer, ran OCR at 200 dpi"],
                ),
                OcrPageResult(
                    page_number=3,
                    text="Page three appendix.",
                    avg_confidence=90.0,
                    warnings=[],
                ),
            ],
            engine_name="fake-ocr-1.0",
            warnings=[],
        )
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_pdf_bytes(),
        content_type="application/pdf",
        filename="report.pdf",
    ))

    ocr_text = _find_output(result, "OCR_TEXT").content.decode("utf-8")
    # Text is joined in page order with blank lines between pages.
    assert "Page one text" in ocr_text
    assert "Page two, rasterized" in ocr_text
    assert "Page three appendix" in ocr_text
    assert ocr_text.index("Page one") < ocr_text.index("Page two") < ocr_text.index("Page three")

    body = json.loads(_find_output(result, "OCR_RESULT").content)
    assert body["kind"] == "pdf"
    assert body["pageCount"] == 3
    assert body["filename"] == "report.pdf"
    assert body["mimeType"] == "application/pdf"
    # Avg confidence is mean of 88, 72.5, 90 = 83.5
    assert body["avgConfidence"] == pytest.approx(83.5, abs=0.01)
    # Page-level warnings rolled up into the document warnings list.
    assert any("no text layer" in w for w in body["warnings"])
    # Per-page entries preserved in document order.
    assert [p["pageNumber"] for p in body["pages"]] == [1, 2, 3]
    assert provider.pdf_calls == [_pdf_bytes()]
    assert provider.image_calls == []


def test_pdf_ocr_above_max_pages_fails_with_typed_error():
    """Safety cap on PDF page count produces OCR_TOO_MANY_PAGES."""
    many_pages = [
        OcrPageResult(page_number=i + 1, text=f"p{i+1}", avg_confidence=80.0)
        for i in range(6)
    ]
    provider = FakeOcrProvider(
        pdf_result=OcrDocumentResult(
            pages=many_pages, engine_name="fake-ocr-1.0", warnings=[]
        )
    )
    cap = OcrCapability(
        provider=provider,
        config=OcrCapabilityConfig(min_confidence_warn=0.0, max_pages=5),
    )

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(content=_pdf_bytes(), content_type="application/pdf"))

    assert exc_info.value.code == "OCR_TOO_MANY_PAGES"
    assert "6" in exc_info.value.message and "5" in exc_info.value.message


# ---------------------------------------------------------------------------
# 3. Unsupported file type.
# ---------------------------------------------------------------------------


def test_unsupported_file_type_rejected_with_typed_error():
    provider = FakeOcrProvider()  # never actually called
    cap = OcrCapability(provider=provider, config=_default_config())

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=b"fake gif bytes GIF89a....",
            content_type="image/gif",
            filename="cat.gif",
        ))

    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"
    assert "PNG" in exc_info.value.message
    assert "PDF" in exc_info.value.message
    assert provider.image_calls == []
    assert provider.pdf_calls == []


def test_unsupported_file_type_rejected_when_all_signals_agree():
    """Even a seemingly-clean TXT upload should fail the classifier —
    we refuse to OCR text that isn't an image or PDF."""
    provider = FakeOcrProvider()
    cap = OcrCapability(provider=provider, config=_default_config())

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=b"just plain text, no magic bytes",
            content_type="text/plain",
            filename="notes.txt",
        ))
    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"


def test_ocr_without_input_file_artifact_fails_clearly():
    provider = FakeOcrProvider()
    cap = OcrCapability(provider=provider, config=_default_config())

    # An OCR job submitted with INPUT_TEXT instead of INPUT_FILE.
    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=b"hello",
            content_type="text/plain",
            filename="prompt.txt",
            artifact_type="INPUT_TEXT",
        ))
    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"


# ---------------------------------------------------------------------------
# 4. Empty extraction / low-confidence edge cases.
# ---------------------------------------------------------------------------


def test_empty_extraction_succeeds_with_warning():
    """Zero extracted characters is legal (blank page, unsupported
    script, low-contrast scan) — the job should still succeed and
    carry a warning the client can inspect."""
    provider = FakeOcrProvider(
        image_result=OcrPageResult(page_number=1, text="", avg_confidence=None)
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(), content_type="image/png", filename="blank.png"
    ))

    ocr_text = _find_output(result, "OCR_TEXT")
    assert ocr_text.content == b""

    body = json.loads(_find_output(result, "OCR_RESULT").content)
    assert body["textLength"] == 0
    assert body["avgConfidence"] is None
    assert any("zero characters" in w for w in body["warnings"])


def test_low_confidence_adds_warning_but_still_succeeds():
    provider = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="grainy scan of something",
            avg_confidence=22.5,  # well below default 40
        )
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(), content_type="image/png", filename="blurry.png"
    ))

    body = json.loads(_find_output(result, "OCR_RESULT").content)
    assert body["avgConfidence"] == 22.5
    assert any("below warn threshold" in w for w in body["warnings"])
    # But the job didn't fail.
    ocr_text = _find_output(result, "OCR_TEXT")
    assert ocr_text.content.decode("utf-8") == "grainy scan of something"


def test_provider_error_becomes_typed_capability_error():
    """OcrError from the provider maps to CapabilityError with the
    provider's code prefixed by `OCR_`, so ops can grep logs for the
    engine-side failure class."""
    provider = FakeOcrProvider(
        raise_on="image",
        raise_error=OcrError("IMAGE_DECODE_FAILED", "bad bytes"),
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=_png_bytes(), content_type="image/png", filename="broken.png"
        ))
    assert exc_info.value.code == "OCR_IMAGE_DECODE_FAILED"
    assert "bad bytes" in exc_info.value.message


# ---------------------------------------------------------------------------
# 5. Registry resilience.
# ---------------------------------------------------------------------------


def test_ocr_failure_still_registers_mock(monkeypatch):
    """If _build_ocr_capability raises (e.g. tesseract binary missing),
    MOCK must still register and the worker must still boot."""
    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    def _failing_ocr(_settings):
        raise RuntimeError(
            "simulated OCR init failure (tesseract binary not found)"
        )

    # Force RAG off so we isolate OCR-path behavior.
    monkeypatch.setattr(
        registry_module, "_build_ocr_capability", _failing_ocr
    )

    settings = WorkerSettings(rag_enabled=False, ocr_enabled=True)
    result = registry_module.build_default_registry(settings)

    assert "MOCK" in result.available()
    assert "OCR" not in result.available()


def test_ocr_failure_does_not_affect_rag_or_mock(monkeypatch):
    """When both RAG and OCR are enabled but OCR init fails, RAG (when
    it can init) and MOCK still register independently."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeRag(Capability):
        name = "RAG"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _fake_rag_builder(_settings):
        return _FakeRag()

    def _failing_ocr(_settings):
        raise RuntimeError("tesseract not installed")

    monkeypatch.setattr(registry_module, "_build_rag_capability", _fake_rag_builder)
    monkeypatch.setattr(registry_module, "_build_ocr_capability", _failing_ocr)

    settings = WorkerSettings(rag_enabled=True, ocr_enabled=True)
    result = registry_module.build_default_registry(settings)

    # MOCK and RAG present; OCR missing.
    assert "MOCK" in result.available()
    assert "RAG" in result.available()
    assert "OCR" not in result.available()


def test_ocr_disabled_leaves_other_capabilities_intact(monkeypatch):
    """rag_enabled=True + ocr_enabled=False → MOCK + RAG only, and
    _build_ocr_capability is never called."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeRag(Capability):
        name = "RAG"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _should_not_be_called(_settings):
        raise AssertionError("ocr_enabled=False must skip OCR construction")

    monkeypatch.setattr(registry_module, "_build_rag_capability", lambda _s: _FakeRag())
    monkeypatch.setattr(registry_module, "_build_ocr_capability", _should_not_be_called)

    settings = WorkerSettings(rag_enabled=True, ocr_enabled=False)
    result = registry_module.build_default_registry(settings)

    assert set(result.available()) == {"MOCK", "RAG"}


# ---------------------------------------------------------------------------
# Normalized trace (trace.v1) assertions on OCR_RESULT.
# ---------------------------------------------------------------------------


def _ocr_trace(result: CapabilityOutput) -> dict:
    """Extract the normalized trace dict embedded in OCR_RESULT."""
    body = json.loads(_find_output(result, "OCR_RESULT").content)
    return body["trace"]


def _find_stage(trace: dict, stage_name: str) -> dict:
    for rec in trace["stages"]:
        if rec["stage"] == stage_name:
            return rec
    raise AssertionError(
        f"no stage {stage_name!r} in trace; present: "
        f"{[r['stage'] for r in trace['stages']]}"
    )


def test_ocr_result_carries_normalized_trace_on_happy_path():
    """OCR_RESULT must embed a trace.v1 payload with classify + ocr
    stages both ok, finalStatus=ok, every canonical envelope field
    populated."""
    provider = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1, text="hello world", avg_confidence=92.0
        )
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(),
        content_type="image/png",
        filename="page.png",
    ))
    trace = _ocr_trace(result)
    assert trace["schemaVersion"] == "trace.v1"
    assert trace["capability"] == "OCR"
    assert trace["inputKind"] == "image"
    assert trace["finalStatus"] == "ok"
    stage_names = [rec["stage"] for rec in trace["stages"]]
    assert stage_names == ["classify", "ocr"]
    for rec in trace["stages"]:
        assert rec["status"] == "ok"
        assert rec["code"] is None
        assert rec["fallbackUsed"] is False

    ocr_stage = _find_stage(trace, "ocr")
    assert ocr_stage["provider"] == "fake-ocr-1.0"
    assert ocr_stage["details"]["textLength"] > 0
    assert ocr_stage["details"]["pageCount"] == 1
    assert ocr_stage["details"]["avgConfidence"] == 92.0


def test_ocr_result_trace_marks_empty_text_as_warn():
    """Zero-char extraction should mark ocr as warn(OCR_EMPTY_TEXT)
    and the trace as partial. Existing extraction semantics
    (OCR_RESULT.warnings list) must be unchanged."""
    provider = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1, text="", avg_confidence=None
        )
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    result = cap.run(_make_job_input(
        content=_png_bytes(),
        content_type="image/png",
        filename="blank.png",
    ))
    body = json.loads(_find_output(result, "OCR_RESULT").content)
    trace = body["trace"]

    assert trace["finalStatus"] == "partial"
    ocr_stage = _find_stage(trace, "ocr")
    assert ocr_stage["status"] == "warn"
    assert ocr_stage["code"] == "OCR_EMPTY_TEXT"

    # Existing OCR_RESULT.warnings still contains the empty-text warning.
    assert any("zero characters" in w for w in body["warnings"])
    # And the extraction fields are untouched.
    assert body["textLength"] == 0
    assert body["pageCount"] == 1


def test_ocr_result_trace_marks_low_confidence_as_warn():
    """Below-threshold avg confidence should mark ocr as
    warn(OCR_LOW_CONFIDENCE), partial trace."""
    provider = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1, text="some text", avg_confidence=20.0
        )
    )
    cap = OcrCapability(
        provider=provider,
        config=OcrCapabilityConfig(min_confidence_warn=40.0, max_pages=100),
    )

    result = cap.run(_make_job_input(
        content=_png_bytes(),
        content_type="image/png",
        filename="low.png",
    ))
    trace = _ocr_trace(result)
    ocr_stage = _find_stage(trace, "ocr")
    assert ocr_stage["status"] == "warn"
    assert ocr_stage["code"] == "OCR_LOW_CONFIDENCE"
    assert trace["finalStatus"] == "partial"


def test_ocr_unsupported_input_type_has_stable_code_and_no_trace_suffix():
    """The classifier runs before the trace builder exists, so an
    unsupported file type produces a bare stable error code with no
    ambiguous 'trace:' suffix."""
    provider = FakeOcrProvider()
    cap = OcrCapability(provider=provider, config=_default_config())

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=b"GIF89a\x01\x00",
            content_type="image/gif",
            filename="cat.gif",
        ))

    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"
    assert "trace:" not in exc_info.value.message


def test_ocr_provider_error_folds_trace_summary_into_message():
    """A hard OcrError raised mid-extraction should produce a typed
    CapabilityError whose message includes the stage flow summary so
    operators can see classify=ok + ocr=fail without downloading
    anything."""
    provider = FakeOcrProvider(
        raise_on="image",
        raise_error=OcrError("IMAGE_DECODE_FAILED", "Pillow blew up"),
    )
    cap = OcrCapability(provider=provider, config=_default_config())

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(_make_job_input(
            content=_png_bytes(),
            content_type="image/png",
            filename="broken.png",
        ))

    assert exc_info.value.code == "OCR_IMAGE_DECODE_FAILED"
    msg = exc_info.value.message
    assert "Pillow blew up" in msg
    assert "trace:" in msg
    assert "classify:ok" in msg
    assert "ocr:fail" in msg
    assert "OCR_IMAGE_DECODE_FAILED" in msg
