"""Artifact builders for the OCR_EXTRACT OCR-lite slice."""

from __future__ import annotations

import json

from app.capabilities.base import CapabilityOutputArtifact
from app.capabilities.ocr.models import OcrDocument

OCR_LITE_PIPELINE_VERSION = "ocr-lite-v1"
OCR_RESULT_JSON = "OCR_RESULT_JSON"
OCR_TEXT_MARKDOWN = "OCR_TEXT_MARKDOWN"


def build_result_json(document: OcrDocument) -> str:
    body = {
        "sourceRecordId": document.source_record_id,
        "pipelineVersion": document.pipeline_version,
        "engine": document.engine,
        "pages": [
            {
                "pageNo": page.page_no,
                "blocks": [
                    {
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": list(block.bbox),
                    }
                    for block in page.blocks
                ],
            }
            for page in document.pages
        ],
        "plainText": document.plain_text,
    }
    return json.dumps(body, ensure_ascii=False, indent=2)


def build_markdown(document: OcrDocument) -> str:
    lines = [
        "# OCR Text",
        "",
        f"Source: {document.source_record_id}",
        f"Pipeline: {document.pipeline_version}",
        f"Engine: {document.engine}",
        "",
    ]
    for page in document.pages:
        lines.append(f"## Page {page.page_no}")
        lines.append("")
        page_text = "\n".join(block.text for block in page.blocks if block.text).strip()
        lines.append(page_text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_output_artifacts(document: OcrDocument) -> list[CapabilityOutputArtifact]:
    return [
        CapabilityOutputArtifact(
            type=OCR_RESULT_JSON,
            filename="ocr-result.json",
            content_type="application/json",
            content=build_result_json(document).encode("utf-8"),
        ),
        CapabilityOutputArtifact(
            type=OCR_TEXT_MARKDOWN,
            filename="ocr-text.md",
            content_type="text/markdown; charset=utf-8",
            content=build_markdown(document).encode("utf-8"),
        ),
    ]
