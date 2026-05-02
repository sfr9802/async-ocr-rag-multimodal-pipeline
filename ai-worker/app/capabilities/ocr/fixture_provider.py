"""Deterministic OCR-lite fixture provider for tests and local proof."""

from __future__ import annotations

from typing import Optional

from app.capabilities.ocr.models import OcrBlock, OcrDocument, OcrPage


class FixtureOcrProvider:
    """Provider that returns one predictable block without reading pixels."""

    def __init__(self, text: str = "extracted text", engine: str = "fixture") -> None:
        self._text = text
        self._engine = engine

    @property
    def engine(self) -> str:
        return self._engine

    def extract(
        self,
        content: bytes,
        *,
        source_record_id: str,
        pipeline_version: str,
        content_type: Optional[str],
        filename: Optional[str],
    ) -> OcrDocument:
        del content, content_type, filename
        return OcrDocument(
            source_record_id=source_record_id,
            pipeline_version=pipeline_version,
            engine=self.engine,
            pages=[
                OcrPage(
                    page_no=1,
                    blocks=[
                        OcrBlock(
                            text=self._text,
                            confidence=0.95,
                            bbox=[0, 0, 100, 30],
                        )
                    ],
                )
            ],
        )
