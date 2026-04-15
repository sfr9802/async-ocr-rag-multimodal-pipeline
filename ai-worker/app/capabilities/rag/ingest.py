"""Ingestion service — reads a dataset, chunks, embeds, persists.

Reads a JSONL file following the port/rag schema (doc_id / title / seed /
sections) and produces:

  1. Document and chunk rows in the PostgreSQL `ragmeta` schema.
  2. A FAISS index on disk in the configured `rag_index_dir`.
  3. An `index_builds` row recording the model + dimensions + counts.

Phase 2 runs ingestion as a one-shot CLI (`scripts/build_rag_index.py`),
not as a long-lived service. The RagCapability's serving path loads the
result at worker startup and does NOT re-read the dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from app.capabilities.rag.chunker import (
    MAX_CH,
    MIN_CH,
    OVERLAP,
    greedy_chunk,
    window_by_chars,
)
from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.metadata_store import ChunkRow, DocumentRow, RagMetadataStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document_count: int
    chunk_count: int
    info: IndexBuildInfo


def _stable_chunk_id(doc_id: str, section: str, order: int, text: str) -> str:
    h = hashlib.md5(f"{doc_id}|{section}|{order}|{text}".encode("utf-8")).hexdigest()[:24]
    return f"{h}_{order}"


def _iter_documents(jsonl_path: Path) -> Iterable[dict]:
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSONL line: %s", line[:80])


def _chunks_from_section(raw_section: dict) -> List[str]:
    """Extract text chunks from a single section payload.

    The port/rag schema gives us three useful signals per section:
      - `chunks`: a list of already-chunked strings (preferred input)
      - `text`:   a full blob to run through greedy_chunk as a fallback
      - `list`:   structured entries (e.g. character name/desc) — each
                  entry becomes one chunk string "name: desc"

    We concatenate whatever we find, then re-window with window_by_chars
    so the final chunks are uniformly close to the target size.
    """
    source_chunks: list[str] = []

    pre = raw_section.get("chunks")
    if isinstance(pre, list):
        source_chunks.extend([str(x) for x in pre if isinstance(x, (str, int, float))])

    list_entries = raw_section.get("list")
    if isinstance(list_entries, list):
        for entry in list_entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            if name and desc:
                source_chunks.append(f"{name}: {desc}")
            elif desc:
                source_chunks.append(desc)
            elif name:
                source_chunks.append(name)

    if not source_chunks:
        blob = raw_section.get("text")
        if isinstance(blob, str) and blob.strip():
            source_chunks.extend(greedy_chunk(blob))

    if not source_chunks:
        return []

    # Always re-window so chunks land near the target size regardless of
    # how uneven the source chunking was.
    return window_by_chars(
        source_chunks,
        target=MAX_CH,
        min_chars=MIN_CH,
        max_chars=MAX_CH,
        overlap=OVERLAP,
    )


class IngestService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        metadata_store: RagMetadataStore,
        index: FaissIndex,
    ) -> None:
        self._embedder = embedder
        self._metadata = metadata_store
        self._index = index

    def ingest_jsonl(
        self,
        jsonl_path: Path,
        *,
        source_label: str,
        index_version: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> IngestResult:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

        log.info("Reading dataset: %s", jsonl_path)

        docs: List[DocumentRow] = []
        chunks: List[ChunkRow] = []
        texts_to_embed: List[str] = []

        if index_version is None:
            index_version = f"v-{int(time.time())}"

        faiss_row_id = 0
        for raw in _iter_documents(jsonl_path):
            doc_id = str(raw.get("doc_id") or raw.get("seed") or raw.get("title") or "").strip()
            if not doc_id:
                continue
            title = str(raw.get("title") or raw.get("seed") or "")[:500]
            docs.append(DocumentRow(
                doc_id=doc_id,
                title=title or None,
                source=source_label,
                category=None,
                metadata={
                    k: raw.get(k)
                    for k in ("seed", "section_order", "created_at")
                    if raw.get(k) is not None
                },
            ))

            sections = raw.get("sections") or {}
            if not isinstance(sections, dict):
                continue

            for section_name, section_raw in sections.items():
                if not isinstance(section_raw, dict):
                    continue
                section_chunks = _chunks_from_section(section_raw)
                for order, text in enumerate(section_chunks):
                    chunk_text = text.strip()
                    if not chunk_text:
                        continue
                    chunk_id = _stable_chunk_id(doc_id, section_name, order, chunk_text)
                    chunks.append(ChunkRow(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section=section_name,
                        chunk_order=order,
                        text=chunk_text,
                        token_count=len(chunk_text.split()),
                        faiss_row_id=faiss_row_id,
                        index_version=index_version,
                    ))
                    texts_to_embed.append(chunk_text)
                    faiss_row_id += 1

        if not chunks:
            raise RuntimeError(
                f"Dataset at {jsonl_path} produced zero chunks. "
                f"Check that each document has sections with chunks/text/list."
            )

        log.info("Ingestion prepared: %d documents, %d chunks", len(docs), len(chunks))

        log.info("Embedding %d passages with model=%s", len(texts_to_embed), self._embedder.model_name)
        vectors = self._embedder.embed_passages(texts_to_embed)
        if vectors.shape[0] != len(chunks):
            raise RuntimeError(
                f"Embedder returned {vectors.shape[0]} vectors for {len(chunks)} chunks"
            )

        info = self._index.build(
            vectors,
            index_version=index_version,
            embedding_model=self._embedder.model_name,
        )

        self._metadata.replace_all(
            documents=docs,
            chunks=chunks,
            index_version=index_version,
            embedding_model=self._embedder.model_name,
            embedding_dim=info.dimension,
            faiss_index_path=str(self._index_dir_for_notes()),
            notes=notes,
        )
        return IngestResult(
            document_count=len(docs),
            chunk_count=len(chunks),
            info=info,
        )

    def _index_dir_for_notes(self) -> str:
        # Avoid poking at FaissIndex internals — we only need a human
        # readable path stored in index_builds.faiss_index_path.
        from app.capabilities.rag.faiss_index import _INDEX_FILE  # noqa: WPS433
        return str(self._index._dir / _INDEX_FILE)  # noqa: SLF001
