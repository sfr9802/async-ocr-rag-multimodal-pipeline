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
from app.capabilities.rag.reranker import RerankerProvider
from app.capabilities.rag.retriever import Retriever

log = logging.getLogger(__name__)


def _release_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()`` for the offline pipeline.

    Called between corpus encoding and downstream stack construction
    so the bulk-encode free pool doesn't crowd out the reranker (or any
    second model) on the same device.

    The torch import is gated on ``sys.modules`` membership: if nothing
    else in the process has imported torch yet (HashingEmbedder unit-
    test paths, CPU-only installs without torch on the path), we
    silently no-op rather than triggering a fresh torch init. Forcing
    a torch import inside an already-loaded pytest process has bitten
    the suite before — the existing comment in
    ``tests/test_rag_embeddings_helpers.py`` documents the segfault
    risk. The production retrieval-rerank CLI always loads torch via
    SentenceTransformerEmbedder first, so the gate doesn't suppress
    the cleanup in any real-world scenario.
    """
    import sys

    if "torch" not in sys.modules:
        return
    torch = sys.modules["torch"]
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            log.info("Released cached CUDA memory after corpus encode.")
    except Exception as ex:  # pragma: no cover — defensive
        log.warning(
            "torch.cuda.empty_cache() failed (%s: %s); continuing.",
            type(ex).__name__, ex,
        )


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
    reranker: Optional[RerankerProvider] = None,
    candidate_k: Optional[int] = None,
) -> Tuple[Retriever, ExtractiveGenerator, OfflineCorpusInfo]:
    """Chunk, embed, and index `corpus_path` into an in-memory stack.

    Re-uses the production chunker and FAISS wrapper so the retrieval
    numbers from this path are comparable to the live worker within the
    fidelity of the embedder choice.

    ``reranker`` and ``candidate_k`` are Phase 2A passthroughs: when both
    are set, the Retriever fetches ``candidate_k`` bi-encoder candidates
    and asks the reranker for the final top-k. Leave them None to reproduce
    the dense-only Phase 0/1 baseline byte-for-byte.
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

    # Release the bulk-encoding free pool back to CUDA before any
    # downstream component (reranker, second embedder, …) tries to
    # allocate. Phase 2A reranker eval stacks bge-m3 + bge-reranker-v2-m3
    # on the same GPU; on a 16 GB card the corpus-encode pass leaves a
    # multi-GB cached free pool that fragments the address space and
    # forces the reranker into expensive defragmentation loops. Best-
    # effort, swallows any exception so a pure-CPU run never breaks.
    _release_cuda_cache()

    store = _InMemoryMetadataStore(version, rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=top_k,
        reranker=reranker,
        candidate_k=candidate_k,
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
