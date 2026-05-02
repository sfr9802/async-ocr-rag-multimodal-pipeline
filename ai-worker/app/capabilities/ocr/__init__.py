"""OCR capability package.

Public surface used by the rest of the worker:

  - OcrCapability, OcrCapabilityConfig  (app.capabilities.ocr.capability)
  - OcrProvider, OcrPageResult,
    OcrDocumentResult, OcrError          (app.capabilities.ocr.provider)
  - TesseractOcrProvider                 (app.capabilities.ocr.tesseract_provider)
  - OCR_EXTRACT OCR-lite models/service/providers live in models.py,
    service.py, fixture_provider.py, and paddle_provider.py.

The phase-2 Tesseract provider remains the engine for the existing OCR
capability. OCR_EXTRACT uses a narrower provider seam so PaddleOCR can be
enabled for local runtime while tests keep using FixtureOcrProvider.
"""

from app.capabilities.ocr.capability import OcrCapability, OcrCapabilityConfig
from app.capabilities.ocr.artifact_builder import (
    OCR_LITE_PIPELINE_VERSION,
    OCR_RESULT_JSON,
    OCR_TEXT_MARKDOWN,
)
from app.capabilities.ocr.fixture_provider import FixtureOcrProvider
from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrLiteProvider,
    OcrPageResult,
    OcrProvider,
)
from app.capabilities.ocr.service import OcrExtractCapability, OcrExtractService

__all__ = [
    "OCR_LITE_PIPELINE_VERSION",
    "OCR_RESULT_JSON",
    "OCR_TEXT_MARKDOWN",
    "FixtureOcrProvider",
    "OcrCapability",
    "OcrCapabilityConfig",
    "OcrDocumentResult",
    "OcrError",
    "OcrExtractCapability",
    "OcrExtractService",
    "OcrLiteProvider",
    "OcrPageResult",
    "OcrProvider",
]
