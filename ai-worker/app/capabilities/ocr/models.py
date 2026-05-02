"""OCR-lite domain models.

These models are intentionally smaller than the existing Tesseract OCR
envelope. OCR-lite only needs page/block text, confidence, and a simple bbox
so the async pipeline can prove artifact handoff without layout-aware work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OcrBlock:
    text: str
    confidence: float
    bbox: list[int] = field(default_factory=lambda: [0, 0, 100, 30])


@dataclass(frozen=True)
class OcrPage:
    page_no: int
    blocks: list[OcrBlock]


@dataclass(frozen=True)
class OcrDocument:
    source_record_id: str
    pipeline_version: str
    engine: str
    pages: list[OcrPage]

    @property
    def plain_text(self) -> str:
        page_texts: list[str] = []
        for page in self.pages:
            text = "\n".join(block.text for block in page.blocks if block.text).strip()
            if text:
                page_texts.append(text)
        return "\n\n".join(page_texts)


class OcrProviderError(RuntimeError):
    """Raised by OCR-lite providers for clean worker failure callbacks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
