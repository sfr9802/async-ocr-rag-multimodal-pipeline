"""Worker-side adapter for SearchUnit indexing contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from app.capabilities.rag.metadata_store import ChunkRow, DocumentRow


@dataclass(frozen=True)
class SearchUnitIndexDocument:
    search_unit_id: str
    claim_token: str
    index_id: str
    source_file_id: str
    source_file_name: Optional[str]
    extracted_artifact_id: Optional[str]
    artifact_type: Optional[str]
    unit_type: str
    unit_key: str
    title: Optional[str]
    section_path: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text_content: str
    content_sha256: str
    metadata_json: Optional[dict[str, Any]] = None
    index_metadata: dict[str, Any] = field(default_factory=dict)


def document_from_claim(payload: dict[str, Any]) -> SearchUnitIndexDocument:
    metadata_json = payload.get("metadataJson")
    if isinstance(metadata_json, str):
        try:
            metadata_json = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata_json = None
    if not isinstance(metadata_json, dict):
        metadata_json = None
    index_metadata = payload.get("indexMetadata")
    if not isinstance(index_metadata, dict):
        index_metadata = {}
    return SearchUnitIndexDocument(
        search_unit_id=str(payload["searchUnitId"]),
        claim_token=str(payload["claimToken"]),
        index_id=str(payload["indexId"]),
        source_file_id=str(payload["sourceFileId"]),
        source_file_name=_str_or_none(payload.get("sourceFileName")),
        extracted_artifact_id=_str_or_none(payload.get("extractedArtifactId")),
        artifact_type=_str_or_none(payload.get("artifactType")),
        unit_type=str(payload.get("unitType") or "CHUNK"),
        unit_key=str(payload.get("unitKey") or payload["searchUnitId"]),
        title=_str_or_none(payload.get("title")),
        section_path=_str_or_none(payload.get("sectionPath")),
        page_start=_int_or_none(payload.get("pageStart")),
        page_end=_int_or_none(payload.get("pageEnd")),
        text_content=str(payload.get("textContent") or ""),
        content_sha256=str(payload.get("contentSha256") or ""),
        metadata_json=metadata_json,
        index_metadata=index_metadata,
    )


def to_document_row(doc: SearchUnitIndexDocument) -> DocumentRow:
    return DocumentRow(
        doc_id=doc.source_file_id,
        title=doc.source_file_name or doc.source_file_id,
        source=doc.source_file_name,
        category=None,
        metadata={
            "source_file_id": doc.source_file_id,
            "source_file_name": doc.source_file_name,
        },
    )


def to_chunk_row(
    doc: SearchUnitIndexDocument,
    *,
    faiss_row_id: int,
    index_version: str,
) -> ChunkRow:
    return ChunkRow(
        chunk_id=doc.index_id,
        doc_id=doc.source_file_id,
        section=doc.section_path or doc.title or doc.unit_type,
        chunk_order=0,
        text=doc.text_content,
        token_count=len(doc.text_content.split()),
        faiss_row_id=faiss_row_id,
        index_version=index_version,
        extra=index_metadata(doc),
    )


def index_metadata(doc: SearchUnitIndexDocument) -> dict[str, Any]:
    metadata = dict(doc.index_metadata)
    metadata.update({
        "search_unit_id": doc.search_unit_id,
        "searchUnitId": doc.search_unit_id,
        "source_file_id": doc.source_file_id,
        "sourceFileId": doc.source_file_id,
        "source_file_name": doc.source_file_name,
        "sourceFileName": doc.source_file_name,
        "extracted_artifact_id": doc.extracted_artifact_id,
        "extractedArtifactId": doc.extracted_artifact_id,
        "artifact_type": doc.artifact_type,
        "artifactType": doc.artifact_type,
        "unit_type": doc.unit_type,
        "unitType": doc.unit_type,
        "unit_key": doc.unit_key,
        "unitKey": doc.unit_key,
        "page_start": doc.page_start,
        "pageStart": doc.page_start,
        "page_end": doc.page_end,
        "pageEnd": doc.page_end,
        "section_path": doc.section_path,
        "sectionPath": doc.section_path,
        "title": doc.title,
        "content_hash": doc.content_sha256,
        "contentHash": doc.content_sha256,
    })
    return {key: value for key, value in metadata.items() if value is not None}


def stable_index_id(source_file_id: str, unit_type: str, unit_key: str) -> str:
    return f"source_file:{source_file_id}:unit:{unit_type}:{unit_key}"


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
