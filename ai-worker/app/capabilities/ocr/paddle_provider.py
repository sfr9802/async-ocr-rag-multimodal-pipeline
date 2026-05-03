"""PaddleOCR-backed OCR-lite provider.

PaddleOCR is imported lazily so tests and fixture/local pipeline proof do not
need the heavy runtime dependency. This provider deliberately does plain OCR
only. PP-Structure, PP-ChatOCR, VLM, layout-aware chunking, and OCR-RAG
indexing are out of scope for this slice.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from app.capabilities.ocr.models import (
    OcrBlock,
    OcrDocument,
    OcrPage,
    OcrProviderError,
)


class PaddleOcrProvider:
    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        pdf_dpi: int = 200,
    ) -> None:
        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._pdf_dpi = int(pdf_dpi)
        self._client: Any = None

    @property
    def engine(self) -> str:
        return "paddleocr"

    def extract(
        self,
        content: bytes,
        *,
        source_record_id: str,
        pipeline_version: str,
        content_type: Optional[str],
        filename: Optional[str],
    ) -> OcrDocument:
        if _is_pdf(content, content_type, filename):
            pages = self._extract_pdf(content)
        else:
            suffix = _suffix_for(content_type, filename)
            pages = [self._extract_image_page(content, page_no=1, suffix=suffix)]

        return OcrDocument(
            source_record_id=source_record_id,
            pipeline_version=pipeline_version,
            engine=self.engine,
            pages=pages,
        )

    def _extract_pdf(self, content: bytes) -> list[OcrPage]:
        try:
            import fitz  # type: ignore
        except ImportError as ex:
            raise OcrProviderError(
                "PDF_SUPPORT_MISSING",
                "PyMuPDF is required to rasterize PDFs before PaddleOCR. "
                "Install pymupdf or submit PNG/JPEG input.",
            ) from ex

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as ex:
            raise OcrProviderError(
                "PDF_OPEN_FAILED",
                f"Could not open PDF bytes: {type(ex).__name__}: {ex}",
            ) from ex

        pages: list[OcrPage] = []
        try:
            if doc.page_count == 0:
                raise OcrProviderError("PDF_EMPTY", "PDF contains zero pages.")
            for index in range(doc.page_count):
                page = doc.load_page(index)
                pixmap = page.get_pixmap(dpi=self._pdf_dpi)
                png_bytes = pixmap.tobytes("png")
                pages.append(
                    self._extract_image_page(
                        png_bytes,
                        page_no=index + 1,
                        suffix=".png",
                    )
                )
        finally:
            doc.close()
        return pages

    def _extract_image_page(
        self,
        content: bytes,
        *,
        page_no: int,
        suffix: str,
    ) -> OcrPage:
        path = _write_temp_file(content, suffix=suffix)
        try:
            raw = _run_paddle_ocr(
                self._ocr_client(),
                path,
                use_textline_orientation=self._use_angle_cls,
            )
            return OcrPage(page_no=page_no, blocks=_normalize_blocks(raw))
        except OcrProviderError:
            raise
        except Exception as ex:
            raise OcrProviderError(
                "PADDLE_RUN_FAILED",
                f"PaddleOCR failed: {type(ex).__name__}: {ex}",
            ) from ex
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _ocr_client(self):
        if self._client is not None:
            return self._client
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as ex:
            raise OcrProviderError(
                "PADDLE_IMPORT_FAILED",
                "paddleocr is not installed. Install PaddleOCR in the worker "
                "runtime or use AIPIPELINE_WORKER_OCR_EXTRACT_PROVIDER=fixture "
                "for fixture-only local proof.",
            ) from ex
        try:
            self._client = PaddleOCR(
                lang=self._lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self._use_angle_cls,
            )
        except TypeError:
            self._client = PaddleOCR(
                lang=self._lang,
                use_angle_cls=self._use_angle_cls,
            )
        return self._client


def _run_paddle_ocr(
    client: Any,
    path: str,
    *,
    use_textline_orientation: bool,
) -> Any:
    predict = getattr(client, "predict", None)
    if callable(predict):
        try:
            return predict(
                path,
                use_textline_orientation=use_textline_orientation,
            )
        except TypeError as ex:
            if "use_textline_orientation" not in str(ex):
                raise
            return predict(path)

    ocr = getattr(client, "ocr", None)
    if not callable(ocr):
        raise OcrProviderError(
            "PADDLE_API_MISSING",
            "PaddleOCR client exposes neither predict(...) nor ocr(...).",
        )
    try:
        return ocr(path, cls=use_textline_orientation)
    except TypeError as ex:
        if "cls" not in str(ex):
            raise
        return ocr(path)


def _normalize_blocks(raw: Any) -> list[OcrBlock]:
    """Normalize common PaddleOCR result shapes into OCR-lite blocks."""
    blocks: list[OcrBlock] = []
    for item in _iter_paddle_lines(raw):
        parsed = _parse_line(item)
        if parsed is not None:
            blocks.append(parsed)
    return blocks


def _iter_paddle_lines(raw: Any):
    if raw is None:
        return
    if hasattr(raw, "res"):
        for nested in _iter_paddle_lines(raw.res):
            yield nested
        return
    if hasattr(raw, "json") and isinstance(raw.json, dict):
        for nested in _iter_paddle_lines(raw.json):
            yield nested
        return
    if isinstance(raw, dict):
        if isinstance(raw.get("res"), dict):
            for nested in _iter_paddle_lines(raw["res"]):
                yield nested
            return
        # PaddleOCR v3-style dictionaries often expose recognized text and
        # boxes as parallel arrays.
        texts = _first_present(raw, "rec_texts", "texts")
        scores = _first_present(raw, "rec_scores", "scores")
        boxes = _first_present(raw, "rec_boxes", "rec_polys", "dt_polys", "boxes")
        if texts is None:
            texts = []
        if scores is None:
            scores = []
        if boxes is None:
            boxes = []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else None
            yield [box, (text, score)]
        return
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and item and _looks_like_line(item):
                yield item
            elif isinstance(item, (list, tuple)):
                for nested in _iter_paddle_lines(item):
                    yield nested
            else:
                for nested in _iter_paddle_lines(item):
                    yield nested


def _looks_like_line(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and isinstance(item[1], (list, tuple))
        and len(item[1]) >= 2
    )


def _parse_line(item: Any) -> Optional[OcrBlock]:
    if not _looks_like_line(item):
        return None
    box = item[0]
    text_score = item[1]
    text = str(text_score[0]).strip()
    if not text:
        return None
    try:
        confidence = float(text_score[1])
    except (TypeError, ValueError):
        confidence = 0.0
    return OcrBlock(
        text=text,
        confidence=confidence,
        bbox=_bbox_from_points(box),
    )


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


def _bbox_from_points(box: Any) -> list[int]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if (
        isinstance(box, (list, tuple))
        and len(box) >= 4
        and all(isinstance(value, (int, float)) for value in box[:4])
    ):
        return [int(round(float(value))) for value in box[:4]]
    points: list[tuple[float, float]] = []
    if isinstance(box, (list, tuple)):
        for point in box:
            if hasattr(point, "tolist"):
                point = point.tolist()
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    if not points:
        return [0, 0, 100, 30]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _write_temp_file(content: bytes, *, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(content)
        return handle.name
    finally:
        handle.close()


def _is_pdf(
    content: bytes,
    content_type: Optional[str],
    filename: Optional[str],
) -> bool:
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in ("application/pdf", "application/x-pdf"):
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return content.startswith(b"%PDF-")


def _suffix_for(content_type: Optional[str], filename: Optional[str]) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg"):
            return suffix
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime == "image/png":
        return ".png"
    if mime in ("image/jpeg", "image/jpg"):
        return ".jpg"
    return ".img"
