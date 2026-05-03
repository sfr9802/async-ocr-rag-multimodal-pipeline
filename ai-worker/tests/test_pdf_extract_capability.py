"""Tests for the PDF_EXTRACT native text capability."""

from __future__ import annotations

import json
import importlib
from io import BytesIO

import pytest

from app.capabilities.base import (
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
)
from app.capabilities.pdf.artifact_builder import (
    PDF_OCR_PIPELINE_VERSION,
    PDF_PARSED_JSON,
    PDF_PIPELINE_VERSION,
    PDF_PLAINTEXT,
)
from app.capabilities.pdf.service import PdfExtractCapability, PdfExtractService
from app.capabilities.ocr.models import OcrBlock, OcrDocument, OcrPage


def test_pdf_extract_emits_native_text_blocks_with_page_bbox_metadata():
    capability = PdfExtractCapability(service=PdfExtractService())

    result = capability.run(_input(_native_text_pdf_bytes()))

    assert [artifact.type for artifact in result.outputs] == [
        PDF_PARSED_JSON,
        PDF_PLAINTEXT,
    ]
    body = json.loads(result.outputs[0].content)
    assert body["document_version_id"] == "source-file-1"
    assert body["sourceRecordId"] == "source-file-1"
    assert body["parser_name"] == "pymupdf"
    assert body["parser_version"] == PDF_PIPELINE_VERSION
    assert body["file_type"] == "pdf"
    assert body["fileType"] == "pdf"
    assert body["warnings"] == []
    assert body["qualityScore"] > 0
    assert body["plainText"]
    assert "Native PDF contract text" in body["plainText"]

    page = body["pages"][0]
    assert page["physical_page_index"] == 0
    assert page["page_no"] == 1
    assert page["page_label"] == "1"
    assert page["width"] > 0
    assert page["height"] > 0
    assert page["text_layer_present"] is True
    assert page["ocr_used"] is False
    assert page["tables"] == []

    block = page["blocks"][0]
    assert block["block_id"] == "p0_b0"
    assert block["block_type"] == "paragraph"
    assert "Native PDF contract text" in block["text"]
    assert len(block["bbox"]) == 4
    assert all(isinstance(value, float) for value in block["bbox"])
    assert block["reading_order"] == 0
    assert block["section_path"] == []

    plain_text = result.outputs[1].content.decode("utf-8")
    assert "Native PDF contract text" in plain_text


def test_pdf_extract_warns_for_empty_text_layer_pages_without_ocr():
    capability = PdfExtractCapability(service=PdfExtractService())

    result = capability.run(_input(_empty_text_layer_pdf_bytes()))

    body = json.loads(result.outputs[0].content)
    page = body["pages"][0]
    assert page["text_layer_present"] is False
    assert page["ocr_used"] is False
    assert page["blocks"] == []
    assert body["plainText"] == ""
    assert body["qualityScore"] == 0

    warning_codes = [warning["code"] for warning in body["warnings"]]
    assert "PDF_TEXT_LAYER_EMPTY" in warning_codes
    assert "OCR_REQUIRED" in warning_codes


def test_pdf_extract_uses_paddle_fallback_only_for_empty_text_layer_pages():
    provider = _FakeOcrProvider()
    capability = PdfExtractCapability(
        service=PdfExtractService(
            ocr_fallback_enabled=True,
            ocr_lang="korean",
            ocr_provider=provider,
        )
    )

    result = capability.run(_input(_empty_text_layer_pdf_bytes()))

    body = json.loads(result.outputs[0].content)
    assert provider.calls == 1
    assert body["parser_version"] == PDF_OCR_PIPELINE_VERSION
    page = body["pages"][0]
    assert page["ocr_used"] is True
    assert page["ocr_engine"] == "paddleocr"
    assert page["ocr_confidence_avg"] == pytest.approx(0.82)
    block = page["blocks"][0]
    assert block["block_type"] == "ocr_line_group"
    assert block["ocr_used"] is True
    assert block["ocr_engine"] == "paddleocr"
    assert block["ocr_language"] == "korean"
    assert block["ocr_confidence"] == pytest.approx(0.82)
    assert "OCR fallback text" in body["plainText"]


def test_pdf_extract_does_not_ocr_normal_native_text_pages():
    provider = _FakeOcrProvider()
    capability = PdfExtractCapability(
        service=PdfExtractService(
            ocr_fallback_enabled=True,
            ocr_provider=provider,
        )
    )

    result = capability.run(_input(_native_text_pdf_bytes()))

    body = json.loads(result.outputs[0].content)
    assert provider.calls == 0
    assert body["parser_version"] == PDF_PIPELINE_VERSION
    assert body["pages"][0]["ocr_used"] is False


def test_pdf_extract_rejects_non_pdf_inputs():
    capability = PdfExtractCapability(service=PdfExtractService())

    with pytest.raises(CapabilityError) as raised:
        capability.run(
            _input(
                b"not a pdf",
                filename="notes.txt",
                content_type="text/plain",
            )
        )

    assert raised.value.code == "UNSUPPORTED_INPUT_TYPE"


def test_registry_registers_pdf_extract_without_ocr_rag_or_xlsx():
    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    settings = WorkerSettings(
        rag_enabled=False,
        ocr_enabled=False,
        ocr_extract_enabled=False,
        xlsx_extract_enabled=False,
        pdf_extract_enabled=True,
        multimodal_enabled=False,
    )

    result = registry_module.build_default_registry(settings)

    assert result.available() == ["MOCK", "PDF_EXTRACT"]


def _input(
    content: bytes,
    *,
    filename: str = "contract.pdf",
    content_type: str = "application/pdf",
) -> CapabilityInput:
    return CapabilityInput(
        job_id="job-1",
        capability="PDF_EXTRACT",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="input-artifact-1",
                source_file_id="source-file-1",
                type="INPUT_FILE",
                content=content,
                content_type=content_type,
                filename=filename,
            )
        ],
    )


def _native_text_pdf_bytes() -> bytes:
    fitz = pytest.importorskip("fitz", reason="PyMuPDF is required to generate PDF fixtures")
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 100), "Native PDF contract text")
    page.insert_text((72, 130), "Second line for reading order")
    return _save_pdf(document)


def _empty_text_layer_pdf_bytes() -> bytes:
    fitz = pytest.importorskip("fitz", reason="PyMuPDF is required to generate PDF fixtures")
    document = fitz.open()
    document.new_page(width=595, height=842)
    return _save_pdf(document)


def _save_pdf(document: object) -> bytes:
    if importlib.util.find_spec("fitz") is None:
        pytest.skip("PyMuPDF is required to generate PDF fixtures")
    stream = BytesIO()
    document.save(stream)
    document.close()
    return stream.getvalue()


class _FakeOcrProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, *_args, **kwargs) -> OcrDocument:
        self.calls += 1
        return OcrDocument(
            source_record_id=kwargs["source_record_id"],
            pipeline_version=kwargs["pipeline_version"],
            engine="paddleocr",
            pages=[
                OcrPage(
                    page_no=1,
                    blocks=[
                        OcrBlock(
                            text="OCR fallback text",
                            confidence=0.82,
                            bbox=[10, 20, 120, 40],
                        )
                    ],
                )
            ],
        )
