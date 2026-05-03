"""Artifact builders for XLSX_EXTRACT outputs."""

from __future__ import annotations

import json
from typing import Any

from app.capabilities.base import CapabilityOutputArtifact

XLSX_PIPELINE_VERSION = "xlsx-extract-v1"
XLSX_WORKBOOK_JSON = "XLSX_WORKBOOK_JSON"
XLSX_MARKDOWN = "XLSX_MARKDOWN"
XLSX_TABLE_JSON = "XLSX_TABLE_JSON"


def build_workbook_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# XLSX Workbook",
        "",
        f"Source: {payload.get('sourceRecordId') or ''}",
        f"Pipeline: {payload.get('pipelineVersion') or XLSX_PIPELINE_VERSION}",
        "",
    ]
    workbook = payload.get("workbook") or {}
    for sheet in workbook.get("sheets") or []:
        if sheet.get("hidden"):
            continue
        name = sheet.get("name") or "Sheet"
        lines.append(f"## {name}")
        lines.append("")
        used_range = sheet.get("usedRange")
        if used_range:
            lines.append(f"Range: {used_range}")
            lines.append("")
        compact = (sheet.get("compactText") or "").strip()
        if compact:
            lines.append(compact)
            lines.append("")
        for table in sheet.get("tables") or []:
            title = table.get("name") or table.get("id") or "Table"
            lines.append(f"### {title}")
            lines.append("")
            table_text = (table.get("markdown") or table.get("text") or "").strip()
            if table_text:
                lines.append(table_text)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_table_json(payload: dict[str, Any]) -> str:
    workbook = payload.get("workbook") or {}
    tables: list[dict[str, Any]] = []
    for sheet in workbook.get("sheets") or []:
        for table in sheet.get("tables") or []:
            tables.append({
                **table,
                "sheetName": sheet.get("name"),
                "sheetIndex": sheet.get("index"),
                "fileType": payload.get("fileType"),
            })
    return json.dumps(
        {
            "sourceRecordId": payload.get("sourceRecordId"),
            "pipelineVersion": payload.get("pipelineVersion"),
            "tables": tables,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_output_artifacts(payload: dict[str, Any]) -> list[CapabilityOutputArtifact]:
    return [
        CapabilityOutputArtifact(
            type=XLSX_WORKBOOK_JSON,
            filename="xlsx-workbook.json",
            content_type="application/json",
            content=build_workbook_json(payload).encode("utf-8"),
        ),
        CapabilityOutputArtifact(
            type=XLSX_MARKDOWN,
            filename="xlsx.md",
            content_type="text/markdown; charset=utf-8",
            content=build_markdown(payload).encode("utf-8"),
        ),
        CapabilityOutputArtifact(
            type=XLSX_TABLE_JSON,
            filename="xlsx-tables.json",
            content_type="application/json",
            content=build_table_json(payload).encode("utf-8"),
        ),
    ]
