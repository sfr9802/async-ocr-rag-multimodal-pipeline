"""PDF_EXTRACT service and capability wrapper."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Any, Optional

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
)
from app.capabilities.pdf.artifact_builder import (
    PDF_OCR_PIPELINE_VERSION,
    PDF_PIPELINE_VERSION,
    build_output_artifacts,
)

log = logging.getLogger(__name__)

_PDF_MIME_TYPES = {"application/pdf"}
_PDF_EXTENSIONS = (".pdf",)
_MAX_BLOCK_TEXT = 30_000
_DEFAULT_MIN_NATIVE_CHARS = 20
_DEFAULT_MIN_TEXT_DENSITY = 0.000001


class PdfExtractService:
    def __init__(
        self,
        *,
        pipeline_version: str = PDF_PIPELINE_VERSION,
        ocr_pipeline_version: str = PDF_OCR_PIPELINE_VERSION,
        ocr_fallback_enabled: bool = False,
        ocr_lang: str = "en",
        ocr_pdf_dpi: int = 200,
        min_native_chars: int = _DEFAULT_MIN_NATIVE_CHARS,
        min_text_density: float = _DEFAULT_MIN_TEXT_DENSITY,
        ocr_provider: Any = None,
    ) -> None:
        self._pipeline_version = pipeline_version
        self._ocr_pipeline_version = ocr_pipeline_version
        self._ocr_fallback_enabled = ocr_fallback_enabled
        self._ocr_lang = ocr_lang
        self._ocr_pdf_dpi = int(ocr_pdf_dpi)
        self._min_native_chars = int(min_native_chars)
        self._min_text_density = float(min_text_density)
        self._ocr_provider = ocr_provider

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        artifact = self._pick_input_artifact(input)
        source_record_id = artifact.source_file_id or f"input-artifact:{artifact.artifact_id}"
        content_type, file_type = self._classify(artifact)
        filename = artifact.filename or f"{artifact.artifact_id}.pdf"

        log.info(
            "PDF extract start jobId=%s artifact=%s fileType=%s",
            input.job_id,
            artifact.artifact_id,
            file_type,
        )
        payload = self._extract(
            artifact.content,
            document_version_id=source_record_id,
            source_record_id=source_record_id,
            content_type=content_type,
            filename=filename,
        )
        return CapabilityOutput(outputs=build_output_artifacts(payload))

    @staticmethod
    def _pick_input_artifact(input: CapabilityInput) -> CapabilityInputArtifact:
        for candidate in input.inputs:
            if candidate.type == "INPUT_FILE":
                return candidate
        if not input.inputs:
            raise CapabilityError("NO_INPUT", "PDF_EXTRACT job has no input artifacts.")
        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "PDF_EXTRACT requires an INPUT_FILE artifact; got "
            + ", ".join(sorted({i.type for i in input.inputs})),
        )

    def _classify(self, artifact: CapabilityInputArtifact) -> tuple[str, str]:
        mime = (artifact.content_type or "").split(";", 1)[0].strip().lower()
        filename = (artifact.filename or "").lower()

        if filename.endswith(_PDF_EXTENSIONS) or mime in _PDF_MIME_TYPES:
            return mime or "application/pdf", "pdf"

        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "PDF_EXTRACT supports PDF input only. Received "
            f"content_type={artifact.content_type!r} filename={artifact.filename!r}.",
        )

    def _extract(
        self,
        content: bytes,
        *,
        document_version_id: str,
        source_record_id: str,
        content_type: Optional[str],
        filename: str,
    ) -> dict[str, Any]:
        try:
            import fitz
        except ImportError as ex:  # pragma: no cover - environment dependent
            raise CapabilityError(
                "DEPENDENCY_MISSING",
                "PDF_EXTRACT requires PyMuPDF. Install pymupdf so fitz can be imported.",
            ) from ex

        try:
            document = fitz.open(stream=BytesIO(content), filetype="pdf")
        except Exception as ex:
            raise CapabilityError(
                "INVALID_PDF",
                f"PDF_EXTRACT could not open the input as a PDF: {type(ex).__name__}: {ex}",
            ) from ex

        warnings: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_payload = self._extract_page(page, page_index, warnings)
                pages.append(page_payload)
        finally:
            document.close()

        if self._ocr_fallback_enabled and any(self._page_needs_ocr(page) for page in pages):
            self._apply_ocr_fallback(
                content,
                pages=pages,
                warnings=warnings,
                source_record_id=source_record_id,
                content_type=content_type,
                filename=filename,
            )

        plain_parts = [_page_plain_text(page) for page in pages]
        plain_parts = [text for text in plain_parts if text]
        plain_text = "\n\n".join(plain_parts).strip()
        quality_score = _quality_score(pages)
        any_ocr = any(page.get("ocr_used") for page in pages)
        return {
            "document_version_id": document_version_id,
            "sourceRecordId": source_record_id,
            "parser_name": "pymupdf+paddleocr" if any_ocr else "pymupdf",
            "parser_version": self._ocr_pipeline_version if any_ocr else self._pipeline_version,
            "file_type": "pdf",
            "fileType": "pdf",
            "contentType": content_type,
            "filename": filename,
            "pages": pages,
            "warnings": warnings,
            "plainText": plain_text,
            "qualityScore": quality_score,
            "quality_score": quality_score,
        }

    def _page_needs_ocr(self, page: dict[str, Any]) -> bool:
        if page.get("ocr_used"):
            return False
        if page.get("text_layer_present") is False:
            return True
        char_count = int(page.get("char_count") or len(_page_plain_text(page)))
        if char_count < self._min_native_chars:
            return True
        width = float(page.get("width") or 0)
        height = float(page.get("height") or 0)
        area = max(width * height, 1.0)
        return (char_count / area) < self._min_text_density

    def _apply_ocr_fallback(
        self,
        content: bytes,
        *,
        pages: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
        source_record_id: str,
        content_type: Optional[str],
        filename: str,
    ) -> None:
        try:
            provider = self._get_ocr_provider()
        except Exception as ex:  # pragma: no cover - depends on local Paddle install
            warnings.append({
                "code": "OCR_FALLBACK_UNAVAILABLE",
                "message": f"PaddleOCR fallback unavailable: {type(ex).__name__}: {ex}",
                "ocr_engine": "paddleocr",
            })
            return

        for page in pages:
            if not self._page_needs_ocr(page):
                continue
            page_no = int(page.get("page_no") or page.get("pageNo") or 0)
            try:
                image_bytes = _render_pdf_page_png(
                    content,
                    page_index=int(page.get("physical_page_index") or max(page_no - 1, 0)),
                    dpi=self._ocr_pdf_dpi,
                )
                ocr_document = provider.extract(
                    image_bytes,
                    source_record_id=source_record_id,
                    pipeline_version=self._ocr_pipeline_version,
                    content_type="image/png",
                    filename=f"{filename}.page-{page_no}.png",
                )
            except Exception as ex:  # pragma: no cover - depends on local Paddle install
                warnings.append({
                    "code": "OCR_FALLBACK_UNAVAILABLE",
                    "message": f"PaddleOCR fallback unavailable: {type(ex).__name__}: {ex}",
                    "physical_page_index": page.get("physical_page_index"),
                    "page_no": page_no,
                    "page_label": page.get("page_label"),
                    "ocr_engine": "paddleocr",
                })
                continue

            ocr_page = ocr_document.pages[0] if ocr_document.pages else None
            if ocr_page is None:
                warnings.append({
                    "code": "OCR_PAGE_MISSING",
                    "message": "PaddleOCR did not return OCR output for a required page.",
                    "physical_page_index": page.get("physical_page_index"),
                    "page_no": page_no,
                    "page_label": page.get("page_label"),
                    "ocr_engine": "paddleocr",
                })
                continue
            ocr_blocks: list[dict[str, Any]] = []
            for index, block in enumerate(ocr_page.blocks):
                if not block.text:
                    continue
                ocr_blocks.append({
                    "block_id": f"p{page.get('physical_page_index', page_no - 1)}_ocr_{index}",
                    "block_type": "ocr_line_group",
                    "text": _clean_text(block.text),
                    "bbox": [float(value) for value in block.bbox],
                    "reading_order": len(page.get("blocks") or []) + index,
                    "section_path": [],
                    "ocr_used": True,
                    "ocr_engine": "paddleocr",
                    "ocr_model": "PaddleOCR",
                    "ocr_language": self._ocr_lang,
                    "ocr_confidence": float(block.confidence),
                    "confidence": float(block.confidence),
                    "quality_score": round(float(block.confidence) * 0.8, 4),
                })
            if not ocr_blocks:
                warnings.append({
                    "code": "OCR_EMPTY_TEXT",
                    "message": "PaddleOCR returned no usable text for a required page.",
                    "physical_page_index": page.get("physical_page_index"),
                    "page_no": page_no,
                    "page_label": page.get("page_label"),
                    "ocr_engine": "paddleocr",
                })
                continue
            page["blocks"] = list(page.get("blocks") or []) + ocr_blocks
            confidences = [float(block["ocr_confidence"]) for block in ocr_blocks]
            page["ocr_used"] = True
            page["ocr_engine"] = "paddleocr"
            page["ocr_model"] = "PaddleOCR"
            page["ocr_language"] = self._ocr_lang
            page["ocr_confidence_avg"] = round(sum(confidences) / len(confidences), 4)
            page["quality_score"] = round(page["ocr_confidence_avg"] * 0.8, 4)
            page["char_count"] = len(_page_plain_text(page))

    def _get_ocr_provider(self) -> Any:
        if self._ocr_provider is not None:
            return self._ocr_provider
        from app.capabilities.ocr.paddle_provider import PaddleOcrProvider

        self._ocr_provider = PaddleOcrProvider(
            lang=self._ocr_lang,
            pdf_dpi=self._ocr_pdf_dpi,
        )
        return self._ocr_provider

    def _extract_page(
        self,
        page: Any,
        page_index: int,
        warnings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        page_no = page_index + 1
        page_label = _page_label(page, page_no)
        rect = page.rect

        blocks: list[dict[str, Any]] = []
        for raw in page.get_text("blocks", sort=True) or []:
            if len(raw) < 5:
                continue
            block_type_code = raw[6] if len(raw) > 6 else 0
            if block_type_code != 0:
                continue
            text = _clean_text(str(raw[4] or ""))
            if not text:
                continue
            reading_order = len(blocks)
            blocks.append({
                "block_id": f"p{page_index}_b{reading_order}",
                "block_type": "paragraph",
                "text": text,
                "bbox": _bbox(raw[:4]),
                "reading_order": reading_order,
                "section_path": [],
            })

        text_layer_present = bool(blocks)
        page_text = "\n".join(block["text"] for block in blocks if block.get("text")).strip()
        if not text_layer_present:
            warnings.append({
                "code": "PDF_TEXT_LAYER_EMPTY",
                "message": "Page has no extractable native text layer.",
                "physical_page_index": page_index,
                "page_no": page_no,
                "page_label": page_label,
            })
            warnings.append({
                "code": "OCR_REQUIRED",
                "message": "Page requires OCR fallback; OCR is not used by pdf-extract-v1.",
                "physical_page_index": page_index,
                "page_no": page_no,
                "page_label": page_label,
            })

        return {
            "physical_page_index": page_index,
            "page_no": page_no,
            "page_label": page_label,
            "width": round(float(rect.width), 2),
            "height": round(float(rect.height), 2),
            "text_layer_present": text_layer_present,
            "ocr_used": False,
            "char_count": len(page_text),
            "blocks": blocks,
            "tables": [],
        }


class PdfExtractCapability(Capability):
    name = "PDF_EXTRACT"

    def __init__(self, service: PdfExtractService) -> None:
        self._service = service

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        return self._service.run(input)


def _page_label(page: Any, fallback_page_no: int) -> str:
    get_label = getattr(page, "get_label", None)
    if callable(get_label):
        try:
            label = get_label()
            if label:
                return str(label)
        except Exception:
            pass
    return str(fallback_page_no)


def _bbox(values: Any) -> list[float]:
    return [round(float(value), 2) for value in values]


def _clean_text(value: str) -> str:
    text = value.replace("\x00", " ").strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= _MAX_BLOCK_TEXT:
        return text
    return text[: _MAX_BLOCK_TEXT - 20].rstrip() + "\n[truncated]"


def _page_plain_text(page: dict[str, Any]) -> str:
    return "\n".join(
        block["text"]
        for block in page.get("blocks") or []
        if block.get("text")
    ).strip()


def _render_pdf_page_png(content: bytes, *, page_index: int, dpi: int) -> bytes:
    try:
        import fitz
    except ImportError as ex:  # pragma: no cover - environment dependent
        raise CapabilityError(
            "DEPENDENCY_MISSING",
            "PyMuPDF is required to rasterize a PDF page for OCR fallback.",
        ) from ex

    document = fitz.open(stream=BytesIO(content), filetype="pdf")
    try:
        page = document.load_page(page_index)
        return page.get_pixmap(dpi=dpi).tobytes("png")
    finally:
        document.close()


def _quality_score(pages: list[dict[str, Any]]) -> float:
    if not pages:
        return 0.0
    text_pages = sum(1 for page in pages if page.get("text_layer_present"))
    block_count = sum(len(page.get("blocks") or []) for page in pages)
    page_coverage = text_pages / len(pages)
    block_signal = min(block_count / max(len(pages), 1), 5.0) / 5.0
    return round((page_coverage * 0.8) + (block_signal * 0.2), 4)
