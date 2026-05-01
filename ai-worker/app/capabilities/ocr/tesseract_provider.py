"""Tesseract-backed OcrProvider.

Uses two dependencies, both imported lazily so the rest of the worker
can start without them installed:

  - pytesseract: thin Python wrapper over the Tesseract CLI. Emits
    per-word confidences via `image_to_data`, which we average into a
    per-page score. Requires the `tesseract` binary on PATH (or a path
    set via the `tesseract_cmd` config knob).

  - PyMuPDF (imported as `fitz`): handles PDF I/O entirely in-process,
    no poppler/pdf2image dependency. For born-digital PDFs we use
    `page.get_text()` (no OCR, no confidence — but perfect fidelity);
    for scanned pages we rasterize with `page.get_pixmap(dpi=...)` and
    hand the PNG bytes to pytesseract.

The provider is constructed once at worker startup. Model/binary
probing happens in `__init__` via `ensure_ready()` so a misconfigured
Tesseract install surfaces as a clean "OCR capability NOT registered"
warning at boot, not a per-job crash.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrPageResult,
    OcrProvider,
)

log = logging.getLogger(__name__)


# Pages with fewer than this many extractable characters from the PDF
# text layer are treated as scanned pages and re-processed via OCR.
# Keeps "text layer present but empty" PDFs from slipping through.
_TEXT_LAYER_MIN_CHARS = 8


class TesseractOcrProvider(OcrProvider):
    """OcrProvider backed by Tesseract + PyMuPDF.

    Config surface is deliberately small:
      - languages: Tesseract language pack string ("eng", "eng+kor", ...)
      - pdf_dpi:   rasterization DPI for scanned PDF pages
      - tesseract_cmd: explicit path to the tesseract binary; None
                       means "whatever is on PATH"
    """

    def __init__(
        self,
        *,
        languages: str = "eng",
        pdf_dpi: int = 200,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self._languages = languages
        self._pdf_dpi = int(pdf_dpi)
        self._tesseract_cmd = tesseract_cmd
        self._version: Optional[str] = None

    # -- readiness --------------------------------------------------------

    def ensure_ready(self) -> None:
        """Probe Tesseract + PyMuPDF at startup so misconfigs fail early.

        Raises OcrError on any import failure, missing binary, or
        unavailable language pack. The registry converts this into a
        clean "OCR capability NOT registered" warning without taking
        down MOCK/RAG.
        """
        try:
            import pytesseract  # type: ignore
        except ImportError as ex:
            raise OcrError(
                "TESSERACT_IMPORT_FAILED",
                "pytesseract is not installed. Run `pip install pytesseract` "
                "in the worker environment.",
            ) from ex

        try:
            import fitz  # type: ignore  # noqa: F401  (imported for import-time check)
        except ImportError as ex:
            raise OcrError(
                "PYMUPDF_IMPORT_FAILED",
                "PyMuPDF is not installed. Run `pip install pymupdf` in the "
                "worker environment. PyMuPDF handles PDF page iteration "
                "without any external poppler/pdf2image dependency.",
            ) from ex

        if self._tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

        try:
            version = pytesseract.get_tesseract_version()
        except Exception as ex:
            raise OcrError(
                "TESSERACT_BINARY_MISSING",
                "Could not execute the tesseract binary. Install Tesseract "
                "(https://tesseract-ocr.github.io/) and make sure it is on "
                "PATH, or set AIPIPELINE_WORKER_OCR_TESSERACT_CMD to its "
                f"absolute path. Underlying error: {type(ex).__name__}: {ex}",
            ) from ex
        self._version = str(version)

        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:  # pragma: no cover — very old tesseract builds
            available = set()
        if available:
            requested = [lang for lang in self._languages.split("+") if lang]
            missing = [lang for lang in requested if lang not in available]
            if missing:
                raise OcrError(
                    "TESSERACT_LANG_MISSING",
                    f"Tesseract language pack(s) not installed: {missing}. "
                    f"Available on this install: {sorted(available)}. "
                    "Install the traineddata files or change "
                    "AIPIPELINE_WORKER_OCR_LANGUAGES.",
                )

        log.info(
            "TesseractOcrProvider ready: tesseract=%s languages=%s pdf_dpi=%d",
            self._version, self._languages, self._pdf_dpi,
        )

    # -- OcrProvider ------------------------------------------------------

    @property
    def name(self) -> str:
        if self._version:
            return f"tesseract-{self._version}"
        return "tesseract"

    def ocr_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
    ) -> OcrPageResult:
        image = self._load_image(image_bytes)
        text, confidence, warnings = self._run_tesseract(image)
        return OcrPageResult(
            page_number=1,
            text=text,
            avg_confidence=confidence,
            warnings=warnings,
        )

    def ocr_pdf(self, pdf_bytes: bytes) -> OcrDocumentResult:
        import fitz  # type: ignore

        pages: List[OcrPageResult] = []
        doc_warnings: List[str] = []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as ex:
            raise OcrError(
                "PDF_OPEN_FAILED",
                f"Could not open PDF bytes with PyMuPDF: {type(ex).__name__}: {ex}",
            ) from ex

        try:
            total_pages = doc.page_count
            if total_pages == 0:
                raise OcrError("PDF_EMPTY", "PDF contains zero pages.")

            for index in range(total_pages):
                page = doc.load_page(index)
                page_number = index + 1

                # 1. Try the born-digital text layer first.
                text_layer = (page.get_text() or "").strip()
                if len(text_layer) >= _TEXT_LAYER_MIN_CHARS:
                    pages.append(OcrPageResult(
                        page_number=page_number,
                        text=text_layer,
                        avg_confidence=None,  # text-layer extraction has no OCR confidence
                        warnings=[],
                    ))
                    continue

                # 2. Text layer missing/too short — rasterize + OCR this page.
                try:
                    pixmap = page.get_pixmap(dpi=self._pdf_dpi)
                    png_bytes = pixmap.tobytes("png")
                except Exception as ex:
                    pages.append(OcrPageResult(
                        page_number=page_number,
                        text="",
                        avg_confidence=None,
                        warnings=[
                            f"page {page_number}: rasterization failed "
                            f"({type(ex).__name__}: {ex})"
                        ],
                    ))
                    continue

                image = self._load_image(png_bytes)
                text, confidence, page_warnings = self._run_tesseract(image)
                page_warnings.insert(
                    0,
                    f"page {page_number}: no text layer, ran OCR at {self._pdf_dpi} dpi",
                )
                pages.append(OcrPageResult(
                    page_number=page_number,
                    text=text,
                    avg_confidence=confidence,
                    warnings=page_warnings,
                ))
        finally:
            doc.close()

        return OcrDocumentResult(
            pages=pages,
            engine_name=self.name,
            warnings=doc_warnings,
        )

    def pdf_page_count(self, pdf_bytes: bytes) -> int:
        """Cheap preflight used by OcrCapability before rasterizing pages."""
        import fitz  # type: ignore

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as ex:
            raise OcrError(
                "PDF_OPEN_FAILED",
                f"Could not open PDF bytes with PyMuPDF: {type(ex).__name__}: {ex}",
            ) from ex
        try:
            return int(doc.page_count)
        finally:
            doc.close()

    # -- helpers ----------------------------------------------------------

    def _load_image(self, image_bytes: bytes):
        try:
            from PIL import Image  # type: ignore
        except ImportError as ex:  # pragma: no cover — Pillow ships with sentence-transformers
            raise OcrError(
                "PIL_IMPORT_FAILED",
                "Pillow is not installed. `pip install Pillow`.",
            ) from ex

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as ex:
            raise OcrError(
                "IMAGE_DECODE_FAILED",
                f"Could not decode image bytes: {type(ex).__name__}: {ex}",
            ) from ex

        # Tesseract wants RGB or grayscale; convert palette / RGBA preemptively.
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        return image

    def _run_tesseract(self, image) -> tuple[str, Optional[float], List[str]]:
        import pytesseract  # type: ignore

        warnings: List[str] = []

        try:
            text = pytesseract.image_to_string(image, lang=self._languages)
        except Exception as ex:
            raise OcrError(
                "TESSERACT_RUN_FAILED",
                f"tesseract image_to_string failed: {type(ex).__name__}: {ex}",
            ) from ex

        avg_confidence: Optional[float] = None
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._languages,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                float(c) for c in data.get("conf", [])
                if c not in (None, "", "-1") and float(c) >= 0
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        except Exception as ex:  # pragma: no cover — non-fatal
            warnings.append(
                f"confidence extraction failed ({type(ex).__name__}: {ex})"
            )

        return text.strip(), avg_confidence, warnings
