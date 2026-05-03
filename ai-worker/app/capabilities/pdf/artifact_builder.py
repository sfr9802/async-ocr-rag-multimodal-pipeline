"""Artifact builders for PDF_EXTRACT outputs."""

from __future__ import annotations

import json
from typing import Any

from app.capabilities.base import CapabilityOutputArtifact

PDF_PIPELINE_VERSION = "pdf-extract-v1"
PDF_OCR_PIPELINE_VERSION = "pdf-extract-v2"
PDF_PARSED_JSON = "PDF_PARSED_JSON"
PDF_PLAINTEXT = "PDF_PLAINTEXT"


def build_parsed_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_plain_text(payload: dict[str, Any]) -> str:
    text = payload.get("plainText")
    if isinstance(text, str):
        return text.rstrip() + ("\n" if text else "")

    page_texts: list[str] = []
    for page in payload.get("pages") or []:
        blocks = page.get("blocks") or []
        joined = "\n".join(
            block.get("text", "").strip()
            for block in blocks
            if block.get("text")
        ).strip()
        if joined:
            page_texts.append(joined)
    return "\n\n".join(page_texts).rstrip() + ("\n" if page_texts else "")


def build_output_artifacts(payload: dict[str, Any]) -> list[CapabilityOutputArtifact]:
    return [
        CapabilityOutputArtifact(
            type=PDF_PARSED_JSON,
            filename="pdf-parsed.json",
            content_type="application/json",
            content=build_parsed_json(payload).encode("utf-8"),
        ),
        CapabilityOutputArtifact(
            type=PDF_PLAINTEXT,
            filename="pdf.txt",
            content_type="text/plain; charset=utf-8",
            content=build_plain_text(payload).encode("utf-8"),
        ),
    ]
