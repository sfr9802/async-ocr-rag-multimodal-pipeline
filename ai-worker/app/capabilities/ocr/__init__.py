"""OCR capability package.

Public surface used by the rest of the worker:

  - OcrCapability, OcrCapabilityConfig  (app.capabilities.ocr.capability)
  - OcrProvider, OcrPageResult,
    OcrDocumentResult, OcrError          (app.capabilities.ocr.provider)
  - TesseractOcrProvider                 (app.capabilities.ocr.tesseract_provider)

The Tesseract provider is the only real engine implementation in phase 1.
Tests use a small fake provider; both honour the same OcrProvider seam
so swapping in EasyOCR / PaddleOCR / a cloud API later is a single-file
change.
"""

from app.capabilities.ocr.capability import OcrCapability, OcrCapabilityConfig
from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrPageResult,
    OcrProvider,
)

__all__ = [
    "OcrCapability",
    "OcrCapabilityConfig",
    "OcrDocumentResult",
    "OcrError",
    "OcrPageResult",
    "OcrProvider",
]
