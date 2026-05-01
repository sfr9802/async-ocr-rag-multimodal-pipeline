"""Ingestion service — reads a dataset, chunks, embeds, persists.

Reads a JSONL file following the port/rag schema (doc_id / title / seed /
sections) and produces:

  1. Document and chunk rows in the PostgreSQL `ragmeta` schema.
  2. A FAISS index on disk in the configured `rag_index_dir`.
  3. An `index_builds` row recording the model + dimensions + counts.
  4. An ``ingest_manifest.json`` sidecar (Phase 7.2) recording the
     embedding-text variant + builder version + a checksum of every
     embedded string so a downstream tool can verify what got
     embedded matches what the offline eval / Phase 7.0 export
     produced.

Phase 2 runs ingestion as a one-shot CLI (`scripts/build_rag_index.py`),
not as a long-lived service. The RagCapability's serving path loads the
result at worker startup and does NOT re-read the dataset.

Phase 7.2 promotes the canonical Phase 7.0 ``retrieval_title_section``
embedding-text format (lives in
``app.capabilities.rag.embedding_text_builder``) to the production
default. The raw chunk text is *still* what gets stored in
``ragmeta.rag_chunks.text`` — only the string handed to the embedder
changes. Reranker and generation paths consume the raw text and are
unaffected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from app.capabilities.rag.chunker import (
    MAX_CH,
    MIN_CH,
    OVERLAP,
    greedy_chunk,
    window_by_chars,
)
from app.capabilities.rag.embedding_text_builder import (
    DEFAULT_PRODUCTION_VARIANT,
    EMBEDDING_TEXT_BUILDER_VERSION,
    PRODUCTION_VARIANTS,
    build_embedding_text_from_v3_chunk,
    is_known_production_variant,
)
from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.metadata_store import ChunkRow, DocumentRow, RagMetadataStore

log = logging.getLogger(__name__)


# Phase 7.2: ingest manifest sidecar that lives alongside build.json. We
# keep it separate so adding a new field never invalidates a previously-
# valid build.json, and so the FaissIndex storage layer can stay
# variant-agnostic (it only knows about index_version / embedding_model
# / dimension / chunk_count, which are still its source of truth).
_INGEST_MANIFEST_FILE = "ingest_manifest.json"
_EMBED_TEXT_SAMPLE_LIMIT = 5
_EMBED_TEXT_PREVIEW_CHARS = 240


@dataclass(frozen=True)
class IngestManifest:
    """Phase 7.2 — embedding-text provenance recorded next to build.json.

    Carried fields:
      - ``embedding_text_variant`` — variant id the builder was driven
        with (``title_section`` / ``retrieval_title_section``). The
        retriever can compare this against the configured runtime
        variant and refuse to load on mismatch.
      - ``embedding_text_builder_version`` — version stamp that maps to
        ``EMBEDDING_TEXT_BUILDER_VERSION`` in the production builder.
        Bumping that constant invalidates every cached index whose
        manifest carries the old value.
      - ``embed_text_sha256`` — SHA-256 over the concatenation of
        every embedded string + ``\\n``. The Phase 7.0 eval export
        emits the same digest by construction, so a parity check is
        a one-line equality test.
      - ``embed_text_samples`` — first 5 truncated strings, for human
        eyeballing without loading the full chunks.jsonl.
      - ``index_version`` / ``embedding_model`` / ``dimension`` /
        ``max_seq_length`` / ``chunk_count`` / ``document_count`` /
        ``corpus_path`` — duplicated from build.json + the ingest
        invocation so the manifest is self-contained.
    """

    embedding_text_variant: str
    embedding_text_builder_version: str
    embedding_model: str
    max_seq_length: Optional[int]
    chunk_count: int
    document_count: int
    dimension: int
    index_version: str
    corpus_path: Optional[str]
    embed_text_sha256: str
    embed_text_samples: List[Dict[str, object]] = field(default_factory=list)


def _digest_embed_texts(texts: Iterable[str]) -> str:
    """SHA-256 the concatenation of texts joined by ``\\n``.

    Same convention the Phase 7.0 eval export uses
    (``v4_chunk_export.py``) so the two digests can be compared
    directly. Joiner is hard-coded to ``\\n`` so a future change
    cannot silently desync.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _embed_text_samples(
    texts: List[str], *, limit: int = _EMBED_TEXT_SAMPLE_LIMIT,
    preview_chars: int = _EMBED_TEXT_PREVIEW_CHARS,
) -> List[Dict[str, object]]:
    """Return the first ``limit`` embedded strings, truncated for preview."""
    samples: List[Dict[str, object]] = []
    for i, t in enumerate(texts[:limit]):
        preview = t if len(t) <= preview_chars else (
            t[: max(0, preview_chars - 1)] + "…"
        )
        samples.append({
            "row": i,
            "char_count": len(t),
            "preview": preview,
        })
    return samples


def _resolve_variant(variant: Optional[str]) -> str:
    """Validate + default a caller-supplied variant string."""
    if variant is None:
        return DEFAULT_PRODUCTION_VARIANT
    if not is_known_production_variant(variant):
        raise ValueError(
            f"Unknown embedding_text_variant {variant!r}; "
            f"expected one of {PRODUCTION_VARIANTS}."
        )
    return variant


def load_ingest_manifest(index_dir: Path) -> Optional[IngestManifest]:
    """Read ``ingest_manifest.json`` from ``index_dir`` if present.

    Returns ``None`` when the sidecar is absent (index built before
    Phase 7.2 or by a tool that doesn't emit the sidecar). Callers that
    want to fail-hard on a missing sidecar can do so explicitly; this
    function intentionally does not because the retriever's existing
    embedding-model check already covers the load-time correctness
    floor.
    """
    path = Path(index_dir) / _INGEST_MANIFEST_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return IngestManifest(
        embedding_text_variant=str(payload.get("embedding_text_variant") or ""),
        embedding_text_builder_version=str(
            payload.get("embedding_text_builder_version") or "",
        ),
        embedding_model=str(payload.get("embedding_model") or ""),
        max_seq_length=(
            int(payload["max_seq_length"])
            if payload.get("max_seq_length") is not None
            else None
        ),
        chunk_count=int(payload.get("chunk_count") or 0),
        document_count=int(payload.get("document_count") or 0),
        dimension=int(payload.get("dimension") or 0),
        index_version=str(payload.get("index_version") or ""),
        corpus_path=(
            str(payload["corpus_path"])
            if payload.get("corpus_path") is not None
            else None
        ),
        embed_text_sha256=str(payload.get("embed_text_sha256") or ""),
        embed_text_samples=list(payload.get("embed_text_samples") or []),
    )


@dataclass(frozen=True)
class IngestResult:
    document_count: int
    chunk_count: int
    info: IndexBuildInfo
    manifest: Optional[IngestManifest] = None


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
        embedding_text_variant: Optional[str] = None,
    ) -> None:
        """Construct the ingest service.

        ``embedding_text_variant`` controls which canonical builder
        variant the embedder sees. ``None`` resolves to
        ``DEFAULT_PRODUCTION_VARIANT`` (Phase 7.2 default
        ``retrieval_title_section``). Pass ``"title_section"`` to
        roll back to the pre-Phase-7.0 baseline.
        """
        self._embedder = embedder
        self._metadata = metadata_store
        self._index = index
        self._embedding_text_variant = _resolve_variant(embedding_text_variant)
        self._embedding_text_builder_version = EMBEDDING_TEXT_BUILDER_VERSION
        log.info(
            "IngestService configured: embedding_text_variant=%s "
            "embedding_text_builder_version=%s",
            self._embedding_text_variant,
            self._embedding_text_builder_version,
        )

    @property
    def embedding_text_variant(self) -> str:
        return self._embedding_text_variant

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

        log.info(
            "Reading dataset: %s (variant=%s)",
            jsonl_path, self._embedding_text_variant,
        )

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
            # Phase 7.2: ``retrieval_title`` may or may not be carried
            # at the doc level depending on the corpus generation.
            # Fixtures (anime_sample.jsonl etc.) don't carry it; future
            # v4-shaped fixtures may. The builder falls back to
            # ``page_title`` when retrieval_title is empty so v3
            # corpora produce the same byte-string under either
            # variant.
            doc_retrieval_title = str(raw.get("retrieval_title") or "")
            category_raw = raw.get("category")
            domain_raw = raw.get("domain")
            language_raw = raw.get("language")
            docs.append(DocumentRow(
                doc_id=doc_id,
                title=title or None,
                source=source_label,
                category=str(category_raw)[:32] if category_raw else None,
                metadata={
                    k: raw.get(k)
                    for k in ("seed", "section_order", "created_at")
                    if raw.get(k) is not None
                },
                domain=str(domain_raw)[:32] if domain_raw else None,
                language=str(language_raw)[:8] if language_raw else None,
            ))

            sections = raw.get("sections") or {}
            if not isinstance(sections, dict):
                continue

            for section_name, section_raw in sections.items():
                if not isinstance(section_raw, dict):
                    continue
                section_chunks = _chunks_from_section(section_raw)
                # Phase 7.2: ``section_type`` is part of the canonical
                # v4 schema. v3 fixtures don't carry it; we plumb it
                # through if the source row provides it (future-proofing
                # for v4 corpora that ingest through this same path).
                section_type = str(section_raw.get("section_type") or "")
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
                    # Phase 7.2: build the canonical embedding text
                    # via the production builder. The raw chunk text
                    # is still what got stored in ChunkRow.text above —
                    # reranker / generation paths consume that, so
                    # they're untouched by the variant change.
                    embed_text = build_embedding_text_from_v3_chunk(
                        chunk_text=chunk_text,
                        title=title,
                        section_name=section_name,
                        retrieval_title=doc_retrieval_title,
                        section_type=section_type,
                        variant=self._embedding_text_variant,
                    )
                    texts_to_embed.append(embed_text)
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

        info, stage_dir = self._index.build_staged(
            vectors,
            index_version=index_version,
            embedding_model=self._embedder.model_name,
        )

        manifest = self._build_manifest(
            texts_to_embed=texts_to_embed,
            info=info,
            document_count=len(docs),
            corpus_path=str(jsonl_path),
            max_seq_length=getattr(
                self._embedder, "max_seq_length", None,
            ),
        )
        try:
            self._write_manifest(manifest, index_dir=stage_dir)
            self._metadata.replace_all(
                documents=docs,
                chunks=chunks,
                index_version=index_version,
                embedding_model=self._embedder.model_name,
                embedding_dim=info.dimension,
                faiss_index_path=str(self._index_dir_for_notes()),
                notes=notes,
            )
            self._index.promote_staged(
                stage_dir,
                info,
                extra_files=(_INGEST_MANIFEST_FILE,),
            )
        except Exception:
            self._index.discard_staged(stage_dir)
            raise

        return IngestResult(
            document_count=len(docs),
            chunk_count=len(chunks),
            info=info,
            manifest=manifest,
        )

    def _build_manifest(
        self,
        *,
        texts_to_embed: List[str],
        info: IndexBuildInfo,
        document_count: int,
        corpus_path: Optional[str],
        max_seq_length: Optional[int],
    ) -> IngestManifest:
        """Pack the per-build manifest dataclass.

        Computes the embed-text sha256 + a small preview sample so the
        sidecar carries everything needed to spot-check against the
        Phase 7.0 eval export digests without re-loading the corpus.
        """
        return IngestManifest(
            embedding_text_variant=self._embedding_text_variant,
            embedding_text_builder_version=self._embedding_text_builder_version,
            embedding_model=info.embedding_model,
            max_seq_length=int(max_seq_length) if max_seq_length else None,
            chunk_count=int(info.chunk_count),
            document_count=int(document_count),
            dimension=int(info.dimension),
            index_version=info.index_version,
            corpus_path=corpus_path,
            embed_text_sha256=_digest_embed_texts(texts_to_embed),
            embed_text_samples=_embed_text_samples(texts_to_embed),
        )

    def _write_manifest(
        self,
        manifest: IngestManifest,
        *,
        index_dir: Optional[Path] = None,
    ) -> Path:
        """Persist the manifest dataclass to ``ingest_manifest.json``.

        Lives next to ``build.json`` under the FAISS index dir; the
        retriever can pick it up at load time without any IO contract
        change in :class:`FaissIndex` itself.
        """
        path = (index_dir or self._index_dir) / _INGEST_MANIFEST_FILE
        path.write_text(
            json.dumps({
                "embedding_text_variant": manifest.embedding_text_variant,
                "embedding_text_builder_version":
                    manifest.embedding_text_builder_version,
                "embedding_model": manifest.embedding_model,
                "max_seq_length": manifest.max_seq_length,
                "chunk_count": manifest.chunk_count,
                "document_count": manifest.document_count,
                "dimension": manifest.dimension,
                "index_version": manifest.index_version,
                "corpus_path": manifest.corpus_path,
                "embed_text_sha256": manifest.embed_text_sha256,
                "embed_text_samples": manifest.embed_text_samples,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Wrote ingest manifest: %s (variant=%s sha256=%s…)",
            path, manifest.embedding_text_variant,
            manifest.embed_text_sha256[:16],
        )
        return path

    @property
    def _index_dir(self) -> Path:
        # FaissIndex._dir is the canonical on-disk location for the
        # index files; we read it directly here so the manifest lands
        # next to build.json in lockstep without growing the public
        # FaissIndex surface.
        return self._index._dir  # noqa: SLF001

    def _index_dir_for_notes(self) -> str:
        # Avoid poking at FaissIndex internals — we only need a human
        # readable path stored in index_builds.faiss_index_path.
        from app.capabilities.rag.faiss_index import _INDEX_FILE  # noqa: WPS433
        return str(self._index._dir / _INDEX_FILE)  # noqa: SLF001
