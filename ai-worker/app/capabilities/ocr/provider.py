"""OcrProvider contract + result dataclasses + typed error.

Kept deliberately narrow: the capability layer doesn't care what engine
is behind the seam, only that the engine knows how to turn "image bytes"
or "pdf bytes" into per-page text (plus optional confidence). Swapping
Tesseract for EasyOCR / PaddleOCR / a cloud API is a single-file change
as long as the new provider honours this interface.

Two methods exist intentionally:
  - ocr_image:  for single-frame rasters (PNG, JPEG, ...)
  - ocr_pdf:    for multi-page PDFs — the provider owns page iteration
                so it can short-circuit on born-digital text layers and
                fall back to page rasterization for scanned PDFs.

The capability layer does mime dispatch; the provider does engine work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from app.capabilities.ocr.models import OcrDocument


@dataclass(frozen=True)
class OcrPageResult:
    """Result of running OCR on a single page / image.

    `page_number` is 1-indexed. For single-image inputs it is always 1.
    `avg_confidence` is 0..100 when the engine reports it (Tesseract
    does), otherwise None. Per-page warnings are surfaced so the
    capability can roll them up into the OCR_RESULT envelope.
    """

    page_number: int
    text: str
    avg_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OcrDocumentResult:
    """Aggregate result across all pages of a single input document.

    `pages` is in document order. `warnings` holds document-level
    warnings (e.g. "no text layer, fell back to OCR") that don't belong
    to any specific page.
    """

    pages: List[OcrPageResult]
    engine_name: str
    warnings: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate page texts with a blank line between pages.

        Empty pages are kept out of the joined string so a blank page
        in the middle of a document doesn't produce a double blank
        line — but their page entry is still present in `pages` so
        metadata consumers can see the gap.
        """
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def avg_confidence(self) -> Optional[float]:
        """Document-level mean of per-page confidences, None if no
        page has a confidence (e.g. a born-digital PDF handled via
        text-layer extraction)."""
        confs = [p.avg_confidence for p in self.pages if p.avg_confidence is not None]
        if not confs:
            return None
        return sum(confs) / len(confs)

    @property
    def total_text_length(self) -> int:
        return sum(len(p.text) for p in self.pages)


class OcrError(Exception):
    """Structured OCR failure.

    Providers raise this (or the capability layer raises it on behalf
    of the provider) so the capability can produce a clean typed
    CapabilityError with a stable `code` string. The error is NOT a
    CapabilityError itself because the provider layer must stay
    independent of the capability base module.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OcrProvider(ABC):
    """Abstract OCR engine."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier embedded into OCR_RESULT.engineName.

        Should include the underlying engine + version if the provider
        can determine it, so ops can correlate a result envelope with a
        specific engine install.
        """

    @abstractmethod
    def ocr_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
    ) -> OcrPageResult:
        """Run OCR on a single image. Raises OcrError on engine failure."""

    @abstractmethod
    def ocr_pdf(self, pdf_bytes: bytes) -> OcrDocumentResult:
        """Run OCR on a PDF document, page-by-page.

        Implementations should prefer the native text layer when it is
        present (born-digital PDFs) and only rasterize+OCR scanned
        pages. Raises OcrError on engine failure.
        """


class OcrLiteProvider(Protocol):
    """Provider protocol for the OCR_EXTRACT OCR-lite slice.

    The existing ``OcrProvider`` ABC above is kept for the phase-2 Tesseract
    OCR capability and multimodal reuse. OCR-lite has a narrower provider
    surface so PaddleOCR can stay isolated behind this protocol.
    """

    @property
    def engine(self) -> str:
        """Stable engine name embedded in OCR_RESULT_JSON."""

    def extract(
        self,
        content: bytes,
        *,
        source_record_id: str,
        pipeline_version: str,
        content_type: Optional[str],
        filename: Optional[str],
    ) -> OcrDocument:
        """Extract OCR-lite page/block data from image or PDF bytes."""
