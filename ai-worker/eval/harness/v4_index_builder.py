"""Phase 7.0 — variant-aware FAISS index over pre-chunked v4 rag_chunks.

The legacy :mod:`eval.harness.embedding_text_reindex` reads a *raw* v3
corpus and runs the production token-aware chunker before embedding,
so its iteration loop assumes the v3 ``{seed, title, sections: {...}}``
schema. Phase 6.3 already chunked the v4 corpus into
``rag_chunks.jsonl`` with stable ``chunk_id`` / ``doc_id``, so we
should index *those* rows directly rather than re-chunking — anything
else risks divergence between the chunk_id we report in the silver
queries' ``expected_doc_ids`` joins and the chunk_id stored in the
FAISS row metadata.

Layout (next to the FAISS index, consistent with the legacy reindex
helper so any future Retriever consumer can load either)::

    <cache_dir>/
      faiss.index
      build.json
      chunks.jsonl              (ChunkLookupResult-shaped rows)
      variant_manifest.json     (variant + corpus + sha + samples)

Index slug / cache key includes the variant per Phase 7.0 spec — same
contract as the v3 reindex helper, exposed here as
:func:`v4_variant_cache_key` so tests can pin it independently of the
v3 family.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.reranker import RerankerProvider
from app.capabilities.rag.retriever import Retriever
from eval.harness.embedding_text_builder import (
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_TITLE_SECTION,
)
from eval.harness.embedding_text_reindex import (
    _CHUNKS_FILE,
    _MANIFEST_FILE,
    VariantManifest,
    _digest_embed_texts,
    _persist_chunks,
    manifest_to_dict,
)
from eval.harness.offline_corpus import (
    OfflineCorpusInfo,
    _InMemoryMetadataStore,
    _release_cuda_cache,
)
from eval.harness.v4_chunk_export import iter_v4_chunk_records

log = logging.getLogger(__name__)


V4_INDEX_VARIANTS: Tuple[str, ...] = (
    VARIANT_TITLE_SECTION,
    VARIANT_RETRIEVAL_TITLE_SECTION,
)


def v4_variant_cache_key(
    rag_chunks_path: Path,
    embedding_model: str,
    max_seq_length: int,
    variant: str,
) -> str:
    """Stable 16-char digest over (chunk file, model, max_seq, variant).

    Includes the file's mtime_ns so a re-export overwrites the cache
    deterministically. The variant is the *last* component fed into
    the digest so collisions across variants are impossible: two
    variants on the same chunk file MUST hash differently.
    """
    if variant not in V4_INDEX_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} not supported for v4 index; "
            f"expected one of {V4_INDEX_VARIANTS}."
        )
    h = hashlib.sha256()
    h.update(str(Path(rag_chunks_path).resolve()).encode("utf-8"))
    h.update(b"|")
    try:
        mtime_ns = Path(rag_chunks_path).stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = 0
    h.update(str(mtime_ns).encode("utf-8"))
    h.update(b"|")
    h.update(str(embedding_model).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(max_seq_length)).encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    return h.hexdigest()[:16]


def v4_default_cache_dir(
    *,
    cache_root: Path,
    embedding_model: str,
    max_seq_length: int,
    variant: str,
    corpus_slug: str = "namu-v4-2008-2026-04",
) -> Path:
    """Canonical cache directory for a v4 variant index.

    Layout::

        <cache_root>/<corpus_slug>-<variant_slug>-mseq<N>

    ``corpus_slug`` defaults to the Phase 7.0 spec value. The variant
    slug uses hyphens so the directory name is filesystem-friendly.
    """
    if variant not in V4_INDEX_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} not supported for v4 index; "
            f"expected one of {V4_INDEX_VARIANTS}."
        )
    variant_slug = variant.replace("_", "-")
    return Path(cache_root) / (
        f"{corpus_slug}-{variant_slug}-mseq{int(max_seq_length)}"
    )


@dataclass(frozen=True)
class V4IndexBuildSummary:
    """Returned to the CLI / caller after a build completes."""

    variant: str
    cache_dir: str
    chunk_count: int
    document_count: int
    dimension: int
    embed_text_sha256: str
    index_version: str


def _iter_chunk_rows(
    rag_chunks_path: Path,
) -> Iterable[Tuple[ChunkLookupResult, str]]:
    """Yield ``(ChunkLookupResult, embed_text)`` from a v4 chunks file.

    The chunks file is the per-variant export — its ``embedding_text``
    is what the embedder will see. The metadata-store row stores the
    raw ``chunk_text`` so the reranker / generation paths are
    unaffected by which variant powered the dense ranking.
    """
    for row_id, rec in enumerate(iter_v4_chunk_records(rag_chunks_path)):
        chunk = ChunkLookupResult(
            chunk_id=str(rec.get("chunk_id") or ""),
            doc_id=str(rec.get("doc_id") or ""),
            section=" > ".join(
                str(s) for s in (rec.get("section_path") or []) if s
            ),
            text=str(rec.get("chunk_text") or ""),
            faiss_row_id=row_id,
        )
        embed_text = str(rec.get("embedding_text") or chunk.text)
        yield chunk, embed_text


def build_v4_variant_index(
    rag_chunks_path: Path,
    *,
    embedder: EmbeddingProvider,
    cache_dir: Path,
    variant: str,
    top_k: int = 10,
    candidate_k: Optional[int] = None,
    reranker: Optional[RerankerProvider] = None,
    index_version: Optional[str] = None,
    write_manifest: bool = True,
    embed_batch_size: Optional[int] = None,
    progress_log_every: int = 5000,
) -> Tuple[Retriever, V4IndexBuildSummary]:
    """Build (or rebuild) the FAISS index for ``variant`` from the chunks file.

    Streams the chunk file row-by-row to keep memory bounded; the only
    in-memory accumulation is the embedding-text list passed to the
    embedder. For 135k chunks at bge-m3 dim=1024 the resulting matrix
    is ≈540MB float32 — well within RAM, and the underlying embedder
    can sub-batch internally via ``embed_passages``.

    Returns a Retriever wired against the freshly persisted cache plus
    a summary dataclass for the CLI to log.
    """
    if variant not in V4_INDEX_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} not supported for v4 index; "
            f"expected one of {V4_INDEX_VARIANTS}."
        )
    rag_chunks_path = Path(rag_chunks_path)
    if not rag_chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {rag_chunks_path}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: List[ChunkLookupResult] = []
    embed_texts: List[str] = []
    doc_ids: set = set()
    for chunk, embed_text in _iter_chunk_rows(rag_chunks_path):
        if not chunk.chunk_id or not embed_text:
            continue
        rows.append(chunk)
        embed_texts.append(embed_text)
        doc_ids.add(chunk.doc_id)
        if progress_log_every and len(rows) % progress_log_every == 0:
            log.info("v4 index: collected %d chunks (variant=%s)",
                     len(rows), variant)

    if not rows:
        raise RuntimeError(
            f"Chunks file {rag_chunks_path} produced zero rows."
        )

    log.info(
        "v4 index build: variant=%s docs=%d chunks=%d model=%s",
        variant, len(doc_ids), len(rows), embedder.model_name,
    )

    # ``EmbeddingProvider.embed_passages`` does not accept a per-call
    # batch_size kwarg — the batch size is fixed at the embedder's
    # constructor. We swallow the parameter to keep the CLI signature
    # symmetric with future embedders that may expose it.
    if embed_batch_size is not None:
        log.info(
            "Note: embed_batch_size=%d ignored at embed-time; the "
            "embedder uses its constructor-configured batch size.",
            embed_batch_size,
        )
    vectors = embedder.embed_passages(embed_texts)

    index = FaissIndex(cache_dir)
    version = index_version or f"v4-{variant.replace('_', '-')}-{int(time.time())}"
    info = index.build(
        vectors,
        index_version=version,
        embedding_model=embedder.model_name,
    )
    _persist_chunks(cache_dir / _CHUNKS_FILE, rows)
    _release_cuda_cache()

    embed_sha, embed_samples = _digest_embed_texts(embed_texts)
    manifest = VariantManifest(
        variant=variant,
        variant_slug=variant.replace("_", "-"),
        embedding_model=embedder.model_name,
        max_seq_length=int(getattr(embedder, "max_seq_length", 0) or 0),
        corpus_path=str(rag_chunks_path.resolve()),
        document_count=len(doc_ids),
        chunk_count=len(rows),
        dimension=int(info.dimension),
        index_version=info.index_version,
        embed_text_sha256=embed_sha,
        embed_text_samples=embed_samples,
    )
    if write_manifest:
        (cache_dir / _MANIFEST_FILE).write_text(
            json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    store = _InMemoryMetadataStore(info.index_version, rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=int(top_k),
        reranker=reranker,
        candidate_k=candidate_k,
    )
    retriever.ensure_ready()

    summary = V4IndexBuildSummary(
        variant=variant,
        cache_dir=str(cache_dir.resolve()),
        chunk_count=len(rows),
        document_count=len(doc_ids),
        dimension=int(info.dimension),
        embed_text_sha256=embed_sha,
        index_version=info.index_version,
    )
    return retriever, summary
