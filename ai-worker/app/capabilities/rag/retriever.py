"""Retrieval service — query text → top-k chunks.

Holds three live objects:

  1. An embedding provider (to vectorize the query)
  2. A loaded FAISS index (to do nearest-neighbour search)
  3. A ragmeta store handle (to look up text + metadata for each hit)

The retriever is built once at worker startup and used for every job;
FAISS search plus a single small Postgres query per job is all that a
top-k retrieval costs at phase-2 scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.metadata_store import RagMetadataStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalReport:
    query: str
    top_k: int
    index_version: str
    embedding_model: str
    results: List[RetrievedChunk]


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        index: FaissIndex,
        metadata: RagMetadataStore,
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._metadata = metadata
        self._top_k = int(top_k)
        self._info: IndexBuildInfo | None = None

    def ensure_ready(self) -> None:
        """Load the index and strictly verify runtime model == build model.

        Two checks, in order:

          1. Model name equality. Mixing two different embedding models
             over the same FAISS index silently corrupts retrieval even
             when dimensions happen to agree, so we fail hard on any
             mismatch — no warning fallback.
          2. Dimension equality. Redundant in practice when (1) passes,
             but kept as a belt-and-suspenders guard against tampered
             build.json files or misbehaving embedder implementations.

        Both failures are raised as RuntimeError so the registry wraps
        them into a clean "RAG capability NOT registered" startup log
        without taking down the MOCK capability.
        """
        self._info = self._index.load()
        log.info(
            "Retriever readiness check: configured_model=%r index_model=%r "
            "configured_dim=%d index_dim=%d index_version=%s chunk_count=%d",
            self._embedder.model_name,
            self._info.embedding_model,
            self._embedder.dimension,
            self._info.dimension,
            self._info.index_version,
            self._info.chunk_count,
        )
        if self._embedder.model_name != self._info.embedding_model:
            raise RuntimeError(
                "Embedding MODEL mismatch: runtime embedder="
                f"{self._embedder.model_name!r} vs index build="
                f"{self._info.embedding_model!r}. "
                "When the embedding model changes, the FAISS index must be "
                "rebuilt AND the worker restarted — mixing models over the "
                "same index silently corrupts retrieval quality. "
                "Fix: either rebuild the index with the runtime model "
                "(`python -m scripts.build_rag_index --fixture`) or set "
                "AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL back to "
                f"{self._info.embedding_model!r}, then restart the worker."
            )
        if self._embedder.dimension != self._info.dimension:
            raise RuntimeError(
                "Embedding DIMENSION mismatch: runtime embedder dim="
                f"{self._embedder.dimension} vs index dim="
                f"{self._info.dimension}. Model names matched "
                f"({self._embedder.model_name!r}) so this almost always "
                "means build.json is stale or hand-edited. "
                "Rebuild the index (`python -m scripts.build_rag_index "
                "--fixture`) and restart the worker."
            )
        log.info(
            "Retriever ready: model=%s dim=%d index_version=%s",
            self._embedder.model_name,
            self._embedder.dimension,
            self._info.index_version,
        )

    def retrieve(self, query: str) -> RetrievalReport:
        if self._info is None:
            raise RuntimeError("Retriever is not ready — call ensure_ready() first")
        vectors = self._embedder.embed_queries([query])
        hits = self._index.search(vectors, top_k=self._top_k)
        if not hits or not hits[0]:
            return RetrievalReport(
                query=query,
                top_k=self._top_k,
                index_version=self._info.index_version,
                embedding_model=self._info.embedding_model,
                results=[],
            )
        row_ids = [row_id for row_id, _score in hits[0]]
        looked_up = self._metadata.lookup_chunks_by_faiss_rows(
            self._info.index_version, row_ids
        )
        score_by_row = {row_id: score for row_id, score in hits[0]}

        results: List[RetrievedChunk] = []
        for hit in looked_up:
            results.append(RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                section=hit.section or "",
                text=hit.text,
                score=float(score_by_row.get(hit.faiss_row_id, 0.0)),
            ))
        return RetrievalReport(
            query=query,
            top_k=self._top_k,
            index_version=self._info.index_version,
            embedding_model=self._info.embedding_model,
            results=results,
        )
