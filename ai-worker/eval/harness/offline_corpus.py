"""Offline RAG eval stack from a single JSONL corpus.

Builds the same Retriever + ExtractiveGenerator the worker uses but with
an in-memory metadata store. Lets the eval harness run against a committed
fixture without a live Postgres ragmeta — useful for phase-0 baselines on
machines that only have the embedding model cached (not the full infra).

Chunks via the same `_chunks_from_section` rules as `app.capabilities.rag.ingest`
so a fixture indexed online and offline produce the same chunk set, which
keeps retrieval numbers comparable across environments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.generation import ExtractiveGenerator
from app.capabilities.rag.ingest import _chunks_from_section, _iter_documents, _stable_chunk_id
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.retriever import Retriever

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineCorpusInfo:
    """Small summary returned alongside the retriever so reports can cite
    exactly what was indexed."""

    corpus_path: str
    document_count: int
    chunk_count: int
    index_version: str
    embedding_model: str
    dimension: int


class _InMemoryMetadataStore:
    """Quacks like RagMetadataStore for the one call Retriever makes.

    Retrieval only ever reads back chunks by faiss_row_id, so a dict
    keyed on the row id is sufficient — no persistence concerns apply
    inside a single eval run.
    """

    def __init__(self, index_version: str, rows: List[ChunkLookupResult]) -> None:
        self._version = index_version
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ChunkLookupResult]:
        if index_version != self._version:
            raise RuntimeError(
                f"Offline store version mismatch: got {index_version!r} "
                f"expected {self._version!r}"
            )
        return [self._by_row[i] for i in faiss_row_ids if i in self._by_row]


def build_offline_rag_stack(
    corpus_path: Path,
    *,
    embedder: EmbeddingProvider,
    index_dir: Path,
    top_k: int,
    index_version: Optional[str] = None,
) -> Tuple[Retriever, ExtractiveGenerator, OfflineCorpusInfo]:
    """Chunk, embed, and index `corpus_path` into an in-memory stack.

    Re-uses the production chunker and FAISS wrapper so the retrieval
    numbers from this path are comparable to the live worker within the
    fidelity of the embedder choice.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    version = index_version or f"offline-{int(time.time())}"

    texts: List[str] = []
    rows: List[ChunkLookupResult] = []
    doc_ids: set[str] = set()
    faiss_row_id = 0
    for raw in _iter_documents(corpus_path):
        doc_id = str(raw.get("doc_id") or raw.get("seed") or raw.get("title") or "").strip()
        if not doc_id:
            continue
        doc_ids.add(doc_id)
        sections = raw.get("sections") or {}
        if not isinstance(sections, dict):
            continue
        for section_name, section_raw in sections.items():
            if not isinstance(section_raw, dict):
                continue
            for order, text in enumerate(_chunks_from_section(section_raw)):
                chunk_text = text.strip()
                if not chunk_text:
                    continue
                chunk_id = _stable_chunk_id(doc_id, section_name, order, chunk_text)
                rows.append(
                    ChunkLookupResult(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section=section_name,
                        text=chunk_text,
                        faiss_row_id=faiss_row_id,
                    )
                )
                texts.append(chunk_text)
                faiss_row_id += 1

    if not rows:
        raise RuntimeError(
            f"Offline corpus {corpus_path} produced zero chunks."
        )

    log.info(
        "Offline corpus: %d docs → %d chunks, embedding with %s",
        len(doc_ids), len(rows), embedder.model_name,
    )
    vectors = embedder.embed_passages(texts)

    index = FaissIndex(index_dir)
    info = index.build(
        vectors,
        index_version=version,
        embedding_model=embedder.model_name,
    )

    store = _InMemoryMetadataStore(version, rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=top_k,
    )
    retriever.ensure_ready()

    return (
        retriever,
        ExtractiveGenerator(),
        OfflineCorpusInfo(
            corpus_path=str(corpus_path),
            document_count=len(doc_ids),
            chunk_count=len(rows),
            index_version=info.index_version,
            embedding_model=info.embedding_model,
            dimension=info.dimension,
        ),
    )
