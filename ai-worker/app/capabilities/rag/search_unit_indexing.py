"""Worker-side adapter and live index writer for SearchUnit indexing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.capabilities.rag.embedding_text_builder import EMBEDDING_TEXT_BUILDER_VERSION
from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.metadata_store import ChunkRow, DocumentRow, RagMetadataStore

log = logging.getLogger(__name__)

_DEFAULT_SEARCH_UNIT_INDEX_VERSION = "search-unit-live-v1"
_INGEST_MANIFEST_FILE = "ingest_manifest.json"
_MANIFEST_SAMPLE_LIMIT = 5
_MANIFEST_PREVIEW_CHARS = 240
_MANIFEST_COMPAT_VARIANT = "retrieval_title_section"
_PDF_EMBEDDING_TEXT_VARIANT = "retrieval_title_section_search_unit_v1"
_SPREADSHEET_EMBEDDING_TEXT_VARIANT = "retrieval_title_section_spreadsheet_v1"


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


@dataclass(frozen=True)
class SearchUnitEmbeddingText:
    text: str
    variant: str
    sha256: str


@dataclass(frozen=True)
class IndexedSearchUnit:
    search_unit_id: str
    claim_token: str
    content_sha256: str
    index_id: str
    faiss_row_id: int


@dataclass(frozen=True)
class SearchUnitVectorIndexResult:
    indexed: list[IndexedSearchUnit]
    info: IndexBuildInfo
    index_version: str


class SearchUnitVectorIndexer:
    """Upsert SearchUnit rows into ragmeta and rebuild the flat FAISS file.

    The existing text index is an exact ``IndexFlatIP`` without a mutable
    delete/update layer. For the live SearchUnit path we therefore keep the
    public contract as an upsert by stable ``index_id`` while implementing the
    file write as a staged full rewrite of the current vectors plus the claimed
    SearchUnits. Existing row ids are preserved; new SearchUnits append.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        metadata_store: RagMetadataStore,
        index: FaissIndex,
        index_version: Optional[str] = None,
        embedding_text_variant: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        notes: str = "search-unit live indexing",
    ) -> None:
        self._embedder = embedder
        self._metadata = metadata_store
        self._index = index
        self._requested_index_version = index_version
        self._embedding_text_variant = embedding_text_variant or _MANIFEST_COMPAT_VARIANT
        self._max_seq_length = max_seq_length
        self._notes = notes

    def index_documents(
        self,
        documents: list[SearchUnitIndexDocument],
    ) -> SearchUnitVectorIndexResult:
        docs = list(documents)
        if not docs:
            info, index_version = self._empty_info()
            return SearchUnitVectorIndexResult(
                indexed=[],
                info=info,
                index_version=index_version,
            )

        for doc in docs:
            if not is_indexable_claim(doc):
                raise ValueError(f"SearchUnit {doc.search_unit_id} is not embeddable")

        index_version, existing_chunks, existing_vectors = self._load_current_index()
        log.info(
            "SearchUnit indexing batch start units=%d existing_chunks=%d version=%s",
            len(docs),
            len(existing_chunks),
            index_version,
        )

        prepared = [
            (
                doc,
                build_search_unit_embedding_text(doc),
            )
            for doc in docs
        ]
        to_embed = [
            (doc, embedding_text)
            for doc, embedding_text in prepared
            if not _is_duplicate_indexed(
                by_index_id=existing_chunks,
                doc=doc,
                embedding_text=embedding_text,
                embedding_model=self._embedder.model_name,
            )
        ]
        embedded_doc_ids = {id(doc) for doc, _ in to_embed}

        new_vectors = (
            self._embedder.embed_passages([embedding_text.text for _, embedding_text in to_embed])
            if to_embed
            else np.empty((0, self._embedder.dimension), dtype=np.float32)
        )
        if new_vectors.shape[0] != len(to_embed):
            raise RuntimeError(
                f"Embedder returned {new_vectors.shape[0]} vectors for {len(to_embed)} SearchUnits"
            )

        final_chunks = list(existing_chunks)
        final_vectors: list[np.ndarray] = [
            existing_vectors[i] for i in range(existing_vectors.shape[0])
        ]
        by_chunk_id = {chunk.chunk_id: i for i, chunk in enumerate(final_chunks)}
        changed_chunks: list[ChunkRow] = []
        indexed: list[IndexedSearchUnit] = []

        embed_vector_offset = 0
        for doc, embedding_text in prepared:
            faiss_row_id = by_chunk_id.get(doc.index_id)
            if faiss_row_id is not None and _chunk_matches_embedding(
                final_chunks[faiss_row_id],
                doc=doc,
                embedding_text=embedding_text,
                embedding_model=self._embedder.model_name,
            ):
                if id(doc) in embedded_doc_ids:
                    embed_vector_offset += 1
                indexed.append(IndexedSearchUnit(
                    search_unit_id=doc.search_unit_id,
                    claim_token=doc.claim_token,
                    content_sha256=doc.content_sha256,
                    index_id=doc.index_id,
                    faiss_row_id=faiss_row_id,
                ))
                continue

            vector = new_vectors[embed_vector_offset]
            embed_vector_offset += 1
            if faiss_row_id is None:
                faiss_row_id = len(final_chunks)
                by_chunk_id[doc.index_id] = faiss_row_id
                final_vectors.append(vector)
                final_chunks.append(to_chunk_row(
                    doc,
                    faiss_row_id=faiss_row_id,
                    index_version=index_version,
                    embedding_model=self._embedder.model_name,
                    embedding_text=embedding_text,
                ))
            else:
                final_vectors[faiss_row_id] = vector
                final_chunks[faiss_row_id] = to_chunk_row(
                    doc,
                    faiss_row_id=faiss_row_id,
                    index_version=index_version,
                    embedding_model=self._embedder.model_name,
                    embedding_text=embedding_text,
                )

            changed_chunks.append(final_chunks[faiss_row_id])
            indexed.append(IndexedSearchUnit(
                search_unit_id=doc.search_unit_id,
                claim_token=doc.claim_token,
                content_sha256=doc.content_sha256,
                index_id=doc.index_id,
                faiss_row_id=faiss_row_id,
            ))

        if not changed_chunks:
            info = self._index.info
            return SearchUnitVectorIndexResult(
                indexed=indexed,
                info=info,
                index_version=index_version,
            )

        all_vectors = np.vstack(final_vectors).astype(np.float32, copy=False)
        info, stage_dir = self._index.build_staged(
            all_vectors,
            index_version=index_version,
            embedding_model=self._embedder.model_name,
        )
        self._write_live_manifest(
            stage_dir,
            info=info,
            chunks=final_chunks,
            document_count=len({chunk.doc_id for chunk in final_chunks}),
        )

        try:
            self._metadata.upsert_index_rows(
                documents=_unique_document_rows(docs),
                chunks=changed_chunks,
            )
            self._metadata.record_index_build(
                index_version=index_version,
                embedding_model=self._embedder.model_name,
                embedding_dim=info.dimension,
                chunk_count=len(final_chunks),
                document_count=len({chunk.doc_id for chunk in final_chunks}),
                faiss_index_path=str(self._index.index_dir / "faiss.index"),
                notes=self._notes,
            )
            self._index.promote_staged(
                stage_dir,
                info,
                extra_files=(_INGEST_MANIFEST_FILE,),
            )
        except Exception:
            self._index.discard_staged(stage_dir)
            raise

        log.info(
            "SearchUnit indexing batch committed units=%d total_chunks=%d version=%s",
            len(indexed),
            info.chunk_count,
            index_version,
        )
        return SearchUnitVectorIndexResult(
            indexed=indexed,
            info=info,
            index_version=index_version,
        )

    def _load_current_index(self) -> tuple[str, list[ChunkRow], np.ndarray]:
        try:
            info = self._index.load()
        except FileNotFoundError:
            index_version = self._requested_index_version or _DEFAULT_SEARCH_UNIT_INDEX_VERSION
            return (
                index_version,
                [],
                np.empty((0, self._embedder.dimension), dtype=np.float32),
            )

        if self._requested_index_version and self._requested_index_version != info.index_version:
            raise RuntimeError(
                "Refusing to live-upsert SearchUnits into a different index_version: "
                f"requested={self._requested_index_version!r} loaded={info.index_version!r}. "
                "Use the existing version or rebuild explicitly."
            )
        if info.embedding_model != self._embedder.model_name:
            raise RuntimeError(
                "Embedding model mismatch for SearchUnit indexing: "
                f"runtime={self._embedder.model_name!r} index={info.embedding_model!r}"
            )
        if info.dimension != self._embedder.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch for SearchUnit indexing: "
                f"runtime={self._embedder.dimension} index={info.dimension}"
            )

        chunks = self._metadata.list_chunks(info.index_version)
        vectors = self._index.vectors()
        self._assert_contiguous_metadata(chunks, vectors)
        return info.index_version, chunks, vectors

    def _assert_contiguous_metadata(
        self,
        chunks: list[ChunkRow],
        vectors: np.ndarray,
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise RuntimeError(
                "ragmeta chunk count does not match FAISS vector count: "
                f"chunks={len(chunks)} vectors={vectors.shape[0]}"
            )
        for expected_row, chunk in enumerate(chunks):
            if chunk.faiss_row_id != expected_row:
                raise RuntimeError(
                    "ragmeta chunks are not contiguous by faiss_row_id: "
                    f"expected={expected_row} actual={chunk.faiss_row_id} "
                    f"chunk_id={chunk.chunk_id}"
                )

    def _empty_info(self) -> tuple[IndexBuildInfo, str]:
        try:
            info = self._index.info
            return info, info.index_version
        except RuntimeError:
            index_version = self._requested_index_version or _DEFAULT_SEARCH_UNIT_INDEX_VERSION
            return (
                IndexBuildInfo(
                    index_version=index_version,
                    embedding_model=self._embedder.model_name,
                    dimension=self._embedder.dimension,
                    chunk_count=0,
                ),
                index_version,
            )

    def _write_live_manifest(
        self,
        stage_dir: Path,
        *,
        info: IndexBuildInfo,
        chunks: list[ChunkRow],
        document_count: int,
    ) -> None:
        texts = [
            str((chunk.extra or {}).get("embeddingTextSha256") or chunk.text)
            for chunk in chunks
        ]
        manifest = {
            "embedding_text_variant": self._embedding_text_variant,
            "embedding_text_builder_version": EMBEDDING_TEXT_BUILDER_VERSION,
            "embedding_model": info.embedding_model,
            "max_seq_length": self._max_seq_length,
            "chunk_count": info.chunk_count,
            "document_count": int(document_count),
            "dimension": info.dimension,
            "index_version": info.index_version,
            "corpus_path": "search-unit-live-indexing",
            "embed_text_sha256": _digest_texts(texts),
            "embed_text_samples": [
                {
                    "row": i,
                    "chunk_id": chunk.chunk_id,
                    "preview": _preview(chunk.text),
                }
                for i, chunk in enumerate(chunks[:_MANIFEST_SAMPLE_LIMIT])
            ],
        }
        (stage_dir / _INGEST_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
    embedding_model: Optional[str] = None,
    embedding_text: Optional[SearchUnitEmbeddingText] = None,
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
        extra=index_metadata(
            doc,
            embedding_model=embedding_model,
            embedding_text=embedding_text,
        ),
    )


def index_metadata(
    doc: SearchUnitIndexDocument,
    *,
    embedding_model: Optional[str] = None,
    embedding_text: Optional[SearchUnitEmbeddingText] = None,
) -> dict[str, Any]:
    metadata = dict(doc.index_metadata)
    metadata.update({
        "index_id": doc.index_id,
        "indexId": doc.index_id,
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
        "content_sha256": doc.content_sha256,
        "contentSha256": doc.content_sha256,
    })
    source_metadata = _combined_metadata(doc)
    for key in (
        "fileType",
        "sheetName",
        "sheetIndex",
        "cellRange",
        "range",
        "rowStart",
        "rowEnd",
        "columnStart",
        "columnEnd",
        "tableId",
    ):
        if key not in metadata and key in source_metadata:
            metadata[key] = source_metadata[key]
    if "cellRange" not in metadata and "usedRange" in source_metadata:
        metadata["cellRange"] = source_metadata["usedRange"]
    if "range" not in metadata and "cellRange" in metadata:
        metadata["range"] = metadata["cellRange"]
    if embedding_model:
        metadata.update({
            "embedding_model": embedding_model,
            "embeddingModel": embedding_model,
        })
    if embedding_text:
        metadata.update({
            "embedding_text_variant": embedding_text.variant,
            "embeddingTextVariant": embedding_text.variant,
            "embedding_text_sha256": embedding_text.sha256,
            "embeddingTextSha256": embedding_text.sha256,
        })
    return {key: value for key, value in metadata.items() if value is not None}


def stable_index_id(source_file_id: str, unit_type: str, unit_key: str) -> str:
    return f"source_file:{source_file_id}:unit:{unit_type}:{unit_key}"


def is_indexable_claim(doc: SearchUnitIndexDocument) -> bool:
    if not doc.text_content or not doc.text_content.strip():
        return False
    metadata = doc.metadata_json or {}
    if isinstance(metadata, dict) and metadata.get("indexable") is False:
        return False
    if doc.unit_type.upper() == "CELL":
        return False
    if doc.unit_type.upper() == "DOCUMENT" and not _is_spreadsheet(doc):
        return False
    return bool(doc.content_sha256.strip())


def build_search_unit_embedding_text(doc: SearchUnitIndexDocument) -> SearchUnitEmbeddingText:
    """Build the canonical text handed to the embedder for a SearchUnit."""
    spreadsheet = _is_spreadsheet(doc)
    variant = _SPREADSHEET_EMBEDDING_TEXT_VARIANT if spreadsheet else _PDF_EMBEDDING_TEXT_VARIANT
    metadata = _combined_metadata(doc)

    header_lines: list[str] = []
    _append_line(header_lines, "Source", doc.source_file_name)
    _append_line(header_lines, "Title", doc.title)
    _append_line(header_lines, "Section", doc.section_path)

    if spreadsheet:
        _append_line(header_lines, "Sheet", _str_or_none(metadata.get("sheetName")))
        _append_line(header_lines, "Range", _str_or_none(
            metadata.get("cellRange") or metadata.get("range") or metadata.get("usedRange")
        ))
        _append_line(header_lines, "Table", _str_or_none(
            metadata.get("tableId") or metadata.get("tableName")
        ))
        headers = _headers_text(metadata)
        _append_line(header_lines, "Headers", headers)
    else:
        _append_line(header_lines, "Page", _page_range(doc.page_start, doc.page_end))

    body = _body_text(doc, metadata, spreadsheet=spreadsheet)
    parts: list[str] = []
    if header_lines:
        parts.append("\n".join(header_lines))
    if body:
        parts.append("Content:\n" + body)
    text = "\n\n".join(parts).strip()
    return SearchUnitEmbeddingText(
        text=text,
        variant=variant,
        sha256=_sha256(text),
    )


def _unique_document_rows(docs: list[SearchUnitIndexDocument]) -> list[DocumentRow]:
    by_id: dict[str, DocumentRow] = {}
    for doc in docs:
        row = to_document_row(doc)
        by_id[row.doc_id] = row
    return list(by_id.values())


def _is_duplicate_indexed(
    *,
    by_index_id: list[ChunkRow],
    doc: SearchUnitIndexDocument,
    embedding_text: SearchUnitEmbeddingText,
    embedding_model: str,
) -> bool:
    for chunk in by_index_id:
        if chunk.chunk_id == doc.index_id:
            return _chunk_matches_embedding(
                chunk,
                doc=doc,
                embedding_text=embedding_text,
                embedding_model=embedding_model,
            )
    return False


def _chunk_matches_embedding(
    chunk: ChunkRow,
    *,
    doc: SearchUnitIndexDocument,
    embedding_text: SearchUnitEmbeddingText,
    embedding_model: str,
) -> bool:
    extra = chunk.extra or {}
    return (
        str(_extra(extra, "sourceFileId", "source_file_id") or "") == doc.source_file_id
        and str(_extra(extra, "unitType", "unit_type") or "") == doc.unit_type
        and str(_extra(extra, "unitKey", "unit_key") or "") == doc.unit_key
        and str(_extra(extra, "contentSha256", "content_sha256", "content_hash") or "") == doc.content_sha256
        and str(_extra(extra, "embeddingModel", "embedding_model") or "") == embedding_model
        and str(_extra(extra, "embeddingTextVariant", "embedding_text_variant") or "") == embedding_text.variant
        and str(_extra(extra, "embeddingTextSha256", "embedding_text_sha256") or "") == embedding_text.sha256
    )


def _combined_metadata(doc: SearchUnitIndexDocument) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    if isinstance(doc.metadata_json, dict):
        combined.update(doc.metadata_json)
    if isinstance(doc.index_metadata, dict):
        combined.update(doc.index_metadata)
    return combined


def _is_spreadsheet(doc: SearchUnitIndexDocument) -> bool:
    metadata = _combined_metadata(doc)
    file_type = str(metadata.get("fileType") or metadata.get("file_type") or "").strip().lower()
    if file_type in {"xlsx", "xlsm", "spreadsheet"}:
        return True
    artifact_type = (doc.artifact_type or "").strip().upper()
    if artifact_type.startswith("XLSX_"):
        return True
    name = (doc.source_file_name or "").strip().lower()
    return name.endswith((".xlsx", ".xlsm"))


def _body_text(
    doc: SearchUnitIndexDocument,
    metadata: dict[str, Any],
    *,
    spreadsheet: bool,
) -> str:
    if spreadsheet and doc.unit_type.upper() == "DOCUMENT":
        sheet_names = _sheet_list_text(metadata)
        workbook_lines = []
        _append_line(workbook_lines, "Workbook", doc.source_file_name or doc.title)
        _append_line(workbook_lines, "Sheets", sheet_names)
        return "\n".join(workbook_lines).strip()
    return doc.text_content.strip()


def _headers_text(metadata: dict[str, Any]) -> Optional[str]:
    value = (
        metadata.get("headers")
        or metadata.get("header")
        or metadata.get("headerRow")
        or metadata.get("headerContext")
    )
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return " | ".join(
            str(item).strip() for item in value.values() if str(item).strip()
        )
    return _str_or_none(value)


def _sheet_list_text(metadata: dict[str, Any]) -> Optional[str]:
    value = metadata.get("sheetNames") or metadata.get("sheets")
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = _str_or_none(item.get("name") or item.get("sheetName"))
            else:
                text = _str_or_none(item)
            if text:
                names.append(text)
        return ", ".join(names) if names else None
    return _str_or_none(value)


def _append_line(lines: list[str], label: str, value: Optional[str]) -> None:
    text = _str_or_none(value)
    if text:
        lines.append(f"{label}: {text}")


def _page_range(page_start: Optional[int], page_end: Optional[int]) -> Optional[str]:
    if page_start is None and page_end is None:
        return None
    if page_end is None or page_end == page_start:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"


def _extra(extra: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in extra:
            return extra[key]
    return None


def _digest_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= _MANIFEST_PREVIEW_CHARS:
        return normalized
    return normalized[: _MANIFEST_PREVIEW_CHARS - 3] + "..."


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
