"""SearchUnit-aware retrieval result and citation contract helpers."""

from __future__ import annotations

from typing import Any

from app.capabilities.rag.generation import RetrievedChunk


def retrieval_result_row(rank: int, chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunkId": chunk.chunk_id,
        "docId": chunk.doc_id,
        "section": chunk.section,
        "score": round(chunk.score, 6),
        "text": chunk.text,
        "searchUnitId": chunk.search_unit_id,
        "sourceFileId": chunk.source_file_id,
        "sourceFileName": chunk.source_file_name,
        "extractedArtifactId": chunk.extracted_artifact_id,
        "artifactType": chunk.artifact_type,
        "unitType": chunk.unit_type or "CHUNK",
        "unitKey": chunk.unit_key,
        "title": chunk.title,
        "sectionPath": chunk.section_path,
        "pageStart": chunk.page_start,
        "pageEnd": chunk.page_end,
        "snippet": preview(chunk.text),
        "textPreview": preview(chunk.text),
        "denseScore": (
            round(chunk.dense_score, 6)
            if chunk.dense_score is not None
            else round(chunk.score, 6)
        ),
        "sparseScore": (
            round(chunk.sparse_score, 6) if chunk.sparse_score is not None else None
        ),
        "rerankScore": (
            round(chunk.rerank_score, 6) if chunk.rerank_score is not None else None
        ),
        "metadataJson": chunk.metadata_json,
        "citation": citation_payload(chunk),
        "grounding": grounding_readiness(chunk, selected_for_context=True),
    }


def citation_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = chunk.metadata_json or {}
    search_unit_id = chunk.search_unit_id or chunk.chunk_id
    unit_type = chunk.unit_type or "CHUNK"
    table_id = metadata.get("tableId") or metadata.get("tableName") or _id_from_unit_key(chunk.unit_key, "table")
    return {
        "sourceFileId": chunk.source_file_id or chunk.doc_id,
        "sourceFileName": chunk.source_file_name,
        "searchUnitId": search_unit_id,
        "unitId": search_unit_id,
        "unitType": unit_type,
        "unitKey": chunk.unit_key,
        "title": chunk.title,
        "pageStart": chunk.page_start,
        "pageEnd": chunk.page_end,
        "sectionPath": chunk.section_path or chunk.section,
        "sheetName": metadata.get("sheetName"),
        "sheetIndex": metadata.get("sheetIndex"),
        "cellRange": metadata.get("cellRange") or metadata.get("range") or metadata.get("usedRange"),
        "rowStart": metadata.get("rowStart"),
        "rowEnd": metadata.get("rowEnd"),
        "columnStart": metadata.get("columnStart"),
        "columnEnd": metadata.get("columnEnd"),
        "tableId": table_id,
        "imageId": _id_from_unit_key(chunk.unit_key, "image"),
        "bbox": metadata.get("bbox") or metadata.get("boundingBox"),
        "artifactId": chunk.extracted_artifact_id,
        "artifactType": chunk.artifact_type,
    }


def grounding_readiness(
    chunk: RetrievedChunk,
    *,
    selected_for_context: bool,
) -> dict[str, Any]:
    citation = citation_payload(chunk)
    has_page_range = chunk.page_start is not None and chunk.page_end is not None
    return {
        "hasCitation": citation is not None,
        "hasSearchUnitId": bool(chunk.search_unit_id),
        "hasSourceFileId": bool(chunk.source_file_id or chunk.doc_id),
        "hasPageRange": has_page_range,
        "hasTextPreview": bool(preview(chunk.text)),
        "selectedForContext": bool(selected_for_context),
    }


def preview(text: str, *, max_chars: int = 240) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _id_from_unit_key(unit_key: str | None, kind: str) -> str | None:
    if not unit_key:
        return None
    prefix = f"{kind}:"
    if unit_key.startswith(prefix):
        return unit_key[len(prefix):] or None
    infix = f":{kind}:"
    index = unit_key.find(infix)
    if index < 0:
        return None
    return unit_key[index + len(infix):] or None
