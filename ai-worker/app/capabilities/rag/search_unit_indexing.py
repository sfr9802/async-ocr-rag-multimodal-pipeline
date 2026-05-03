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
        self._embedding_text_variant = embedding_text_variant
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

        new_vectors = self._embedder.embed_passages([doc.text_content for doc in docs])
        if new_vectors.shape[0] != len(docs):
            raise RuntimeError(
                f"Embedder returned {new_vectors.shape[0]} vectors for {len(docs)} SearchUnits"
            )

        final_chunks = list(existing_chunks)
        final_vectors: list[np.ndarray] = [
            existing_vectors[i] for i in range(existing_vectors.shape[0])
        ]
        by_chunk_id = {chunk.chunk_id: i for i, chunk in enumerate(final_chunks)}
        changed_chunks: list[ChunkRow] = []
        indexed: list[IndexedSearchUnit] = []

        for vector_offset, doc in enumerate(docs):
            faiss_row_id = by_chunk_id.get(doc.index_id)
            if faiss_row_id is None:
                faiss_row_id = len(final_chunks)
                by_chunk_id[doc.index_id] = faiss_row_id
                final_vectors.append(new_vectors[vector_offset])
                final_chunks.append(to_chunk_row(
                    doc,
                    faiss_row_id=faiss_row_id,
                    index_version=index_version,
                ))
            else:
                final_vectors[faiss_row_id] = new_vectors[vector_offset]
                final_chunks[faiss_row_id] = to_chunk_row(
                    doc,
                    faiss_row_id=faiss_row_id,
                    index_version=index_version,
                )

            changed_chunks.append(final_chunks[faiss_row_id])
            indexed.append(IndexedSearchUnit(
                search_unit_id=doc.search_unit_id,
                claim_token=doc.claim_token,
                content_sha256=doc.content_sha256,
                index_id=doc.index_id,
                faiss_row_id=faiss_row_id,
            ))

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
        texts = [chunk.text for chunk in chunks]
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
    return {key: value for key, value in metadata.items() if value is not None}


def stable_index_id(source_file_id: str, unit_type: str, unit_key: str) -> str:
    return f"source_file:{source_file_id}:unit:{unit_type}:{unit_key}"


def is_indexable_claim(doc: SearchUnitIndexDocument) -> bool:
    if not doc.text_content or not doc.text_content.strip():
        return False
    metadata = doc.metadata_json or {}
    if isinstance(metadata, dict) and metadata.get("indexable") is False:
        return False
    return bool(doc.content_sha256.strip())


def _unique_document_rows(docs: list[SearchUnitIndexDocument]) -> list[DocumentRow]:
    by_id: dict[str, DocumentRow] = {}
    for doc in docs:
        row = to_document_row(doc)
        by_id[row.doc_id] = row
    return list(by_id.values())


def _digest_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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
