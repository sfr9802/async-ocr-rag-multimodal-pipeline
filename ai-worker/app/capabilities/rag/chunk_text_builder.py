"""Build separate retrieval/display/citation texts for parsed document chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ChunkTexts:
    embedding_text: str
    bm25_text: str
    display_text: str
    citation_text: str
    debug_text: str


def build_xlsx_chunk_texts(
    *,
    source_file_name: str,
    location: dict[str, Any],
    display_text: str,
    title: Optional[str] = None,
) -> ChunkTexts:
    citation = xlsx_citation_text(source_file_name, location)
    headers = _list_text(location.get("header_path") or location.get("headerPath"))
    embedding_lines: list[str] = []
    _append(embedding_lines, "Source", source_file_name)
    _append(embedding_lines, "Title", title)
    _append(embedding_lines, "Sheet", _str(location.get("sheet_name") or location.get("sheetName")))
    _append(embedding_lines, "Range", _str(location.get("cell_range") or location.get("cellRange")))
    _append(embedding_lines, "Table", _str(location.get("table_id") or location.get("tableId")))
    _append(embedding_lines, "Headers", headers)
    body = _clean(display_text)
    embedding_text = "\n".join(embedding_lines)
    if body:
        embedding_text = (embedding_text + "\n\nContent:\n" + body).strip()
    bm25_text = _join_text(source_file_name, citation, title, headers, display_text)
    return ChunkTexts(
        embedding_text=embedding_text,
        bm25_text=bm25_text,
        display_text=display_text,
        citation_text=citation,
        debug_text=_debug("xlsx", location, title),
    )


def build_pdf_chunk_texts(
    *,
    source_file_name: str,
    location: dict[str, Any],
    display_text: str,
    title: Optional[str] = None,
) -> ChunkTexts:
    citation = pdf_citation_text(source_file_name, location)
    section = _list_text(location.get("section_path") or location.get("sectionPath"))
    page = _str(location.get("page_no") or location.get("pageNo") or location.get("page_label"))
    embedding_lines: list[str] = []
    _append(embedding_lines, "Source", source_file_name)
    _append(embedding_lines, "Title", title)
    _append(embedding_lines, "Page", page)
    _append(embedding_lines, "Section", section)
    _append(embedding_lines, "Block", _str(location.get("block_type") or location.get("blockType")))
    if location.get("ocr_used") is True:
        _append(embedding_lines, "OCR confidence", _str(location.get("ocr_confidence")))
    body = _clean(display_text)
    embedding_text = "\n".join(embedding_lines)
    if body:
        embedding_text = (embedding_text + "\n\nContent:\n" + body).strip()
    bm25_text = _join_text(source_file_name, citation, title, section, display_text)
    return ChunkTexts(
        embedding_text=embedding_text,
        bm25_text=bm25_text,
        display_text=display_text,
        citation_text=citation,
        debug_text=_debug("pdf", location, title),
    )


def xlsx_citation_text(source_file_name: str, location: dict[str, Any]) -> str:
    return " > ".join(
        part
        for part in (
            source_file_name,
            _str(location.get("sheet_name") or location.get("sheetName")),
            _str(location.get("cell_range") or location.get("cellRange")),
        )
        if part
    )


def pdf_citation_text(source_file_name: str, location: dict[str, Any]) -> str:
    page = _str(location.get("page_no") or location.get("pageNo") or location.get("page_label"))
    bbox = _bbox_text(location.get("bbox"))
    return " > ".join(
        part
        for part in (
            source_file_name,
            f"p.{page}" if page else None,
            f"bbox {bbox}" if bbox else None,
        )
        if part
    )


def _append(lines: list[str], label: str, value: Optional[str]) -> None:
    text = _str(value)
    if text:
        lines.append(f"{label}: {text}")


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _join_text(*values: Optional[str]) -> str:
    return "\n".join(text for value in values if (text := _clean(value)))


def _list_text(value: Any) -> Optional[str]:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list):
                text = " > ".join(_clean(str(part)) for part in item if _clean(str(part)))
            else:
                text = _clean(str(item))
            if text:
                parts.append(text)
        return " | ".join(parts) if parts else None
    return _str(value)


def _bbox_text(value: Any) -> Optional[str]:
    if not isinstance(value, list) or not value:
        return None
    return "[" + ",".join(_clean(str(part)) for part in value) + "]"


def _debug(file_type: str, location: dict[str, Any], title: Optional[str]) -> str:
    return json.dumps(
        {
            "file_type": file_type,
            "title": title,
            "location": location,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
