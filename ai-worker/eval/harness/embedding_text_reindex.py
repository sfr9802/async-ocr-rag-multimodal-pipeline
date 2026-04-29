"""Eval-only — variant-aware FAISS reindex helper.

Phase 2 follow-up to ``confirm_wide_mmr_best_configs``. The candidate
pool ceiling on silver_200 (cand@50 = 0.80) is the open bottleneck after
the wide-MMR / Optuna confirm verdict landed
``INCONCLUSIVE_REPRESENTATION_BOTTLENECK``. The next experimental axis is
*what string the bi-encoder embeds* — currently the chunk text only,
matching production. This module re-encodes the full corpus under one
of the prefix variants exposed by ``embedding_text_builder``:

  - ``raw``           — chunk.text only (regression anchor — matches the
    pre-existing cached index)
  - ``title``         — title + chunk.text
  - ``title_section`` — title + section + chunk.text (most likely win)

Production code (``app/``) is **not touched**. The chunk metadata
persisted to ``chunks.jsonl`` always carries the *raw* chunk text — so
the reranker sees the same passage strings regardless of variant. Only
the FAISS dense vectors change between variants, which is the correct
experimental design (the variant axis isolates dense candidate
representation from rerank behaviour).

Outputs per variant (under the cache dir):

  - ``faiss.index``           — FAISS bytes (variant-specific embeddings)
  - ``build.json``            — index version, embedding model, chunk count
  - ``chunks.jsonl``          — ``ChunkLookupResult`` rows (raw text)
  - ``variant_manifest.json`` — variant id, sample embedded strings,
    corpus path, sha256, embedding model, chunk count, dimension

The manifest's purpose is auditability: a downstream reader can
*verify* the index was built with the variant it claims by hashing the
embedding-text strings and comparing against the manifest's recorded
sample digest. Cheap insurance against accidentally pointing the
confirm sweep at a stale cache.

Cache key contract (``variant_cache_key``):
  digest = sha256(``corpus_path|mtime|embedding_model|max_seq|variant``)
  Including the variant in the key is mandatory — different variants
  produce different vectors, and a cache-key collision between variants
  would silently feed the confirm sweep wrong data. Tests pin this.

Reuse posture:
  Re-uses ``app.capabilities.rag.faiss_index.FaissIndex`` for the
  storage layer, ``app.capabilities.rag.metadata_store.ChunkLookupResult``
  for the row type, ``eval.harness.offline_corpus._InMemoryMetadataStore``
  for query-time lookups, and ``eval.harness.embedding_text_builder``
  for the prefix logic. The chunk-iteration loop is duplicated from
  ``offline_corpus.build_offline_rag_stack`` so this module can stand
  alone — the duplication is ~30 LOC and lets us evolve the variant
  pipeline independently of the production-shaped offline stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.ingest import (
    _chunks_from_section,
    _iter_documents,
    _stable_chunk_id,
)
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.reranker import RerankerProvider
from app.capabilities.rag.retriever import Retriever

from eval.harness.embedding_text_builder import (
    EMBEDDING_TEXT_VARIANTS,
    EmbeddingTextInput,
    VARIANT_RAW,
    build_embedding_text,
)
from eval.harness.offline_corpus import (
    OfflineCorpusInfo,
    _InMemoryMetadataStore,
    _release_cuda_cache,
)

log = logging.getLogger(__name__)


_CHUNKS_FILE = "chunks.jsonl"
_MANIFEST_FILE = "variant_manifest.json"
_EMBED_TEXT_SAMPLE_LIMIT = 5
_EMBED_TEXT_PREVIEW_CHARS = 240


# ---------------------------------------------------------------------------
# Variant slug + cache-key helpers — pure functions; no I/O.
# ---------------------------------------------------------------------------


def variant_slug_for_path(variant: str) -> str:
    """Map ``variant`` → filesystem-friendly slug.

    The ``embedding_text_builder`` constants use underscores
    (``"title_section"``) to match Python identifier conventions, but
    cache directory names look cleaner with hyphens. This function is
    the single source of truth for that conversion so the report
    writer, the confirm sweep, and the cache loader all agree on
    the path layout.

    Examples::

        variant_slug_for_path("raw")            → "raw"
        variant_slug_for_path("title")          → "title"
        variant_slug_for_path("title_section")  → "title-section"
        variant_slug_for_path("title-section")  → "title-section"  (idempotent)

    Raises ``ValueError`` if ``variant`` is not a known builder variant.
    """
    if variant is None:
        raise ValueError("variant is required")
    normalized = str(variant).strip()
    if not normalized:
        raise ValueError("variant must be a non-empty string")
    if normalized not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
        )
    return normalized.replace("_", "-")


def model_slug_for_path(embedding_model: str) -> str:
    """Map an embedding model name → short, hyphenated slug.

    Drops the HuggingFace org prefix (``BAAI/bge-m3`` → ``bge-m3``)
    and replaces remaining slashes with hyphens. The shorter form
    keeps cache directory names scannable (``bge-m3-...`` rather than
    ``BAAI_bge-m3-...``) while still being unambiguous within the
    project's model set. Raises ``ValueError`` on empty input.
    """
    if not embedding_model:
        raise ValueError("embedding_model is required")
    name = str(embedding_model).strip()
    if not name:
        raise ValueError("embedding_model must be a non-empty string")
    # Drop the leading ``org/`` segment if present (HF convention).
    tail = name.split("/")[-1]
    return tail.replace("/", "-").replace("_", "-").lower()


# Suffixes we lop off the corpus directory name when composing the
# cache slug. ``-token-chunked`` / ``-chunked`` are the noisy bits the
# anime corpus suite emits (``anime_namu_v3_token_chunked``); trimming
# them keeps the slug short without losing identifying signal. Order
# matters — the longest match wins so ``-token-chunked`` is checked
# before ``-chunked``.
_CORPUS_SLUG_TRIM_SUFFIXES: Tuple[str, ...] = ("-token-chunked", "-chunked")


def corpus_slug_for_path(corpus_path: Path) -> str:
    """Best-effort corpus identifier slug for cache directory names.

    Pulls the corpus file's *parent directory* name, hyphenates it,
    and trims known noise suffixes (``-token-chunked``,
    ``-chunked``). Falls back to the raw parent name when no trim
    matches.

    Examples::

        corpus_slug_for_path(
            Path("eval/corpora/anime_namu_v3_token_chunked/corpus.jsonl")
        ) → "anime-namu-v3"

        corpus_slug_for_path(
            Path("eval/corpora/enterprise_v1/payload.jsonl")
        ) → "enterprise-v1"

    The slug is used purely for human readability — corpus identity
    in cache validation goes through ``build.json`` (corpus_path,
    chunk_count) and ``variant_manifest.json`` (embed_text_sha256),
    so a slug collision between two corpora that happen to share a
    parent name does not corrupt the cache, it merely produces a
    misleading directory label. The trim list can grow as new
    corpus families land.
    """
    parent_name = Path(corpus_path).parent.name or "unknown-corpus"
    slug = parent_name.replace("_", "-").lower()
    for trim in _CORPUS_SLUG_TRIM_SUFFIXES:
        if slug.endswith(trim):
            return slug[: -len(trim)]
    return slug


def variant_cache_key(
    corpus_path: Path,
    embedding_model: str,
    max_seq_length: int,
    variant: str,
) -> str:
    """Stable 16-char digest over (corpus, model, max_seq, variant).

    Mirrors ``eval_full_silver_minimal_sweep._dense_cache_key`` so the
    layout under ``eval/agent_loop_ab/_indexes/`` stays consistent, but
    *adds* the variant identifier to the digest. Two variants with the
    same (corpus, model, max_seq) MUST produce different keys —
    collisions would silently feed the confirm sweep wrong data. The
    test ``test_variant_cache_key_includes_variant`` pins this.
    """
    if variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
        )
    h = hashlib.sha256()
    h.update(str(Path(corpus_path).resolve()).encode("utf-8"))
    h.update(b"|")
    try:
        mtime_ns = Path(corpus_path).stat().st_mtime_ns
    except FileNotFoundError:
        # Tests may pass a non-existent path; treat as 0 mtime so the
        # digest stays deterministic per (path, model, variant).
        mtime_ns = 0
    h.update(str(mtime_ns).encode("utf-8"))
    h.update(b"|")
    h.update(str(embedding_model).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(max_seq_length)).encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    return h.hexdigest()[:16]


def default_cache_dir_for_variant(
    *,
    cache_root: Path,
    embedding_model: str,
    max_seq_length: int,
    corpus_path: Path,
    variant: str,
) -> Path:
    """Return the canonical, human-readable cache directory for ``variant``.

    Layout::

        <cache_root>/<model_slug>-<corpus_slug>-<variant_slug>-mseq<N>

    Earlier drafts appended a 16-char digest (``...-<sha16>``) so two
    corpora with the same parent directory name couldn't collide.
    That made the directory name unreadable for a benefit that
    ``build.json`` already provides (it records the resolved
    ``corpus_path`` + chunk count, so a stale cache is detectable at
    load time). The current layout drops the digest in favour of a
    slug a human can scan at a glance: ``bge-m3-anime-namu-v3-title-
    section-mseq1024`` is unambiguous in the project's corpus set.

    If a future corpus family produces ``corpus_slug`` collisions, the
    fix is to (a) extend ``_CORPUS_SLUG_TRIM_SUFFIXES`` so the offender
    gets a distinguishing suffix back, or (b) pass ``--cache-dir``
    explicitly to the CLI. ``variant_cache_key`` is preserved as a
    helper so callers that still want a deterministic short digest
    (legacy artefacts, future hash-suffixed layouts) have the same
    contract.
    """
    return Path(cache_root) / (
        f"{model_slug_for_path(embedding_model)}-"
        f"{corpus_slug_for_path(corpus_path)}-"
        f"{variant_slug_for_path(variant)}-"
        f"mseq{int(max_seq_length)}"
    )


# ---------------------------------------------------------------------------
# Manifest dataclass + writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantManifest:
    """Provenance manifest written next to the FAISS index.

    Fields chosen to make a downstream reader able to (a) tell which
    variant the index was built with, (b) sanity-check the embedded
    text composition, and (c) cross-reference against the corpus and
    embedding model it was built against. ``embed_text_samples`` are
    capped to ``_EMBED_TEXT_SAMPLE_LIMIT`` rows so the manifest stays
    tiny; ``embed_text_sha256`` digests the *full* embedded-text list
    so a tamper-check is still possible without storing ~50k strings.
    """

    variant: str
    variant_slug: str
    embedding_model: str
    max_seq_length: int
    corpus_path: str
    document_count: int
    chunk_count: int
    dimension: int
    index_version: str
    embed_text_sha256: str
    embed_text_samples: List[Dict[str, Any]] = field(default_factory=list)


def manifest_to_dict(manifest: VariantManifest) -> Dict[str, Any]:
    return {
        "schema": "variant-index-manifest.v1",
        "variant": manifest.variant,
        "variant_slug": manifest.variant_slug,
        "embedding_model": manifest.embedding_model,
        "max_seq_length": manifest.max_seq_length,
        "corpus_path": manifest.corpus_path,
        "document_count": manifest.document_count,
        "chunk_count": manifest.chunk_count,
        "dimension": manifest.dimension,
        "index_version": manifest.index_version,
        "embed_text_sha256": manifest.embed_text_sha256,
        "embed_text_samples": list(manifest.embed_text_samples),
    }


# ---------------------------------------------------------------------------
# Variant chunk iterator (mirrors ``build_offline_rag_stack`` loop body).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantChunkRecord:
    """One chunk after metadata extraction, before embedding.

    ``raw_text`` is what the reranker / metadata store sees; it always
    matches the production chunk text. ``embed_text`` is what the
    bi-encoder will embed for *this* variant — different from raw when
    the variant adds a title/section prefix. Keeping both side-by-side
    on the record lets the manifest sample tier surface them paired so
    a debugger can see exactly what got prefixed.
    """

    chunk_id: str
    doc_id: str
    section: str
    raw_text: str
    embed_text: str
    title: Optional[str]
    faiss_row_id: int


def iter_variant_chunk_records(
    corpus_path: Path,
    *,
    variant: str,
) -> Iterable[VariantChunkRecord]:
    """Yield ``VariantChunkRecord`` for each chunk in the corpus.

    Iteration order is stable: corpus order → section iteration order
    (insertion order on the dict) → ``_chunks_from_section`` output
    order. Same as ``build_offline_rag_stack`` so a comparison run
    against the raw cache lines up by ``faiss_row_id``.

    Documents missing a usable doc_id (no doc_id, no seed, no title)
    are skipped silently — the production ingest does the same.
    """
    if variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
        )
    faiss_row_id = 0
    for raw in _iter_documents(Path(corpus_path)):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id:
            continue
        title_raw = raw.get("title")
        title_str = str(title_raw).strip() if title_raw else None
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
                chunk_id = _stable_chunk_id(
                    doc_id, str(section_name), order, chunk_text,
                )
                embed_text = build_embedding_text(
                    EmbeddingTextInput(
                        text=chunk_text,
                        title=title_str,
                        section=str(section_name) if section_name else None,
                        keywords=(),
                    ),
                    variant=variant,
                )
                yield VariantChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section=str(section_name),
                    raw_text=chunk_text,
                    embed_text=embed_text,
                    title=title_str,
                    faiss_row_id=faiss_row_id,
                )
                faiss_row_id += 1


# ---------------------------------------------------------------------------
# Variant index build / load
# ---------------------------------------------------------------------------


def _persist_chunks(chunks_path: Path, rows: Sequence[ChunkLookupResult]) -> None:
    """Write ``ChunkLookupResult`` rows to ``chunks.jsonl`` (raw text only)."""
    with chunks_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps({
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "text": r.text,
                "faiss_row_id": r.faiss_row_id,
            }, ensure_ascii=False) + "\n")


def _load_chunks(chunks_path: Path) -> List[ChunkLookupResult]:
    rows: List[ChunkLookupResult] = []
    with chunks_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(ChunkLookupResult(
                chunk_id=str(obj["chunk_id"]),
                doc_id=str(obj["doc_id"]),
                section=str(obj["section"]),
                text=str(obj["text"]),
                faiss_row_id=int(obj["faiss_row_id"]),
            ))
    return rows


def _digest_embed_texts(
    embed_texts: Sequence[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return ``(sha256_hex, samples)`` over the embedding text list.

    ``samples`` is the first ``_EMBED_TEXT_SAMPLE_LIMIT`` entries with
    the embedded text truncated to ``_EMBED_TEXT_PREVIEW_CHARS`` so the
    manifest stays small. ``sha256_hex`` is computed over the *full*
    list with ``\\n``-joined lines so two manifests built from the same
    variant + corpus collide deterministically.
    """
    h = hashlib.sha256()
    for text in embed_texts:
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    samples: List[Dict[str, Any]] = []
    for i, text in enumerate(embed_texts[:_EMBED_TEXT_SAMPLE_LIMIT]):
        samples.append({
            "row": i,
            "preview": text[:_EMBED_TEXT_PREVIEW_CHARS],
            "char_count": len(text),
        })
    return h.hexdigest(), samples


def build_variant_dense_stack(
    corpus_path: Path,
    *,
    embedder: EmbeddingProvider,
    index_dir: Path,
    top_k: int,
    embedding_text_variant: str = VARIANT_RAW,
    index_version: Optional[str] = None,
    reranker: Optional[RerankerProvider] = None,
    candidate_k: Optional[int] = None,
    write_manifest: bool = True,
) -> Tuple[Retriever, OfflineCorpusInfo, VariantManifest]:
    """Encode the corpus under ``embedding_text_variant`` and return the stack.

    Mirrors ``offline_corpus.build_offline_rag_stack`` for the storage /
    metadata wiring but routes the embed-time text through
    ``build_embedding_text`` so the FAISS vectors carry the variant's
    prefix while the metadata store keeps the raw chunk text. Persists
    ``chunks.jsonl`` + ``variant_manifest.json`` next to the FAISS
    index so a follow-up confirm run finds a complete cache.

    The retriever produced here behaves exactly like the production
    Retriever at query time — query embeddings still run through
    ``embedder.embed_queries`` (no variant applied to queries; the
    variant axis is *passage-side only*). The rerank stage receives
    raw chunk text, so a variant only changes the dense ranking that
    gets fed *into* the reranker.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    if embedding_text_variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {embedding_text_variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
        )

    version = (
        index_version
        if index_version
        else f"variant-{embedding_text_variant}-{int(time.time())}"
    )

    rows: List[ChunkLookupResult] = []
    embed_texts: List[str] = []
    doc_ids: set[str] = set()

    for record in iter_variant_chunk_records(
        corpus_path, variant=embedding_text_variant,
    ):
        doc_ids.add(record.doc_id)
        rows.append(ChunkLookupResult(
            chunk_id=record.chunk_id,
            doc_id=record.doc_id,
            section=record.section,
            text=record.raw_text,
            faiss_row_id=record.faiss_row_id,
        ))
        embed_texts.append(record.embed_text)

    if not rows:
        raise RuntimeError(
            f"Corpus {corpus_path} produced zero chunks for variant "
            f"{embedding_text_variant!r}."
        )

    log.info(
        "variant=%s: %d docs → %d chunks; embedding with %s",
        embedding_text_variant, len(doc_ids), len(rows),
        embedder.model_name,
    )
    vectors = embedder.embed_passages(embed_texts)

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index = FaissIndex(Path(index_dir))
    info = index.build(
        vectors,
        index_version=version,
        embedding_model=embedder.model_name,
    )

    # Persist chunk metadata + manifest. Doing this *before* releasing
    # CUDA cache so a download-only failure mode (model loaded, encode
    # done, but FAISS write or persist fails) leaves no half-state.
    _persist_chunks(Path(index_dir) / _CHUNKS_FILE, rows)

    # Best-effort CUDA cleanup — same idiom used by the offline stack.
    _release_cuda_cache()

    embed_sha, embed_samples = _digest_embed_texts(embed_texts)
    manifest = VariantManifest(
        variant=embedding_text_variant,
        variant_slug=variant_slug_for_path(embedding_text_variant),
        embedding_model=embedder.model_name,
        max_seq_length=int(getattr(embedder, "max_seq_length", 0) or 0),
        corpus_path=str(Path(corpus_path).resolve()),
        document_count=len(doc_ids),
        chunk_count=len(rows),
        dimension=int(info.dimension),
        index_version=info.index_version,
        embed_text_sha256=embed_sha,
        embed_text_samples=embed_samples,
    )

    if write_manifest:
        (Path(index_dir) / _MANIFEST_FILE).write_text(
            json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Wrote variant manifest %s (sha256[:16]=%s)",
            Path(index_dir) / _MANIFEST_FILE,
            embed_sha[:16],
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

    offline_info = OfflineCorpusInfo(
        corpus_path=str(corpus_path),
        document_count=len(doc_ids),
        chunk_count=len(rows),
        index_version=info.index_version,
        embedding_model=info.embedding_model,
        dimension=info.dimension,
    )
    return retriever, offline_info, manifest


def load_variant_dense_stack(
    cache_dir: Path,
    *,
    embedder: EmbeddingProvider,
    top_k: int,
    reranker: Optional[RerankerProvider] = None,
    candidate_k: Optional[int] = None,
) -> Tuple[Retriever, OfflineCorpusInfo, Optional[VariantManifest]]:
    """Load a previously-built variant cache without re-encoding.

    Cache hit requires ``faiss.index``, ``build.json``, and
    ``chunks.jsonl`` to exist; ``variant_manifest.json`` is optional
    (older raw caches built before this module existed don't carry
    one — we still load them so the inherited
    ``bge-m3-anime-namu-v3-raw-mseq1024`` cache works as the raw
    anchor).
    """
    cache_dir = Path(cache_dir)
    faiss_path = cache_dir / "faiss.index"
    build_meta = cache_dir / "build.json"
    chunks_path = cache_dir / _CHUNKS_FILE
    if not (faiss_path.exists() and build_meta.exists() and chunks_path.exists()):
        raise FileNotFoundError(
            f"Variant cache {cache_dir} is incomplete: "
            "expected faiss.index / build.json / chunks.jsonl."
        )

    index = FaissIndex(cache_dir)
    info = index.load()
    if info.embedding_model != embedder.model_name:
        raise RuntimeError(
            f"Variant cache embedding_model={info.embedding_model!r} "
            f"differs from embedder={embedder.model_name!r}; "
            "rebuild with the matching model or pass --force-rebuild."
        )

    rows = _load_chunks(chunks_path)
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

    offline_info = OfflineCorpusInfo(
        corpus_path=str(cache_dir),
        document_count=len({r.doc_id for r in rows}),
        chunk_count=len(rows),
        index_version=info.index_version,
        embedding_model=info.embedding_model,
        dimension=info.dimension,
    )

    manifest_path = cache_dir / _MANIFEST_FILE
    manifest: Optional[VariantManifest] = None
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = VariantManifest(
                variant=str(payload.get("variant") or VARIANT_RAW),
                variant_slug=str(payload.get("variant_slug") or "raw"),
                embedding_model=str(payload.get("embedding_model") or ""),
                max_seq_length=int(payload.get("max_seq_length") or 0),
                corpus_path=str(payload.get("corpus_path") or ""),
                document_count=int(payload.get("document_count") or 0),
                chunk_count=int(payload.get("chunk_count") or 0),
                dimension=int(payload.get("dimension") or 0),
                index_version=str(payload.get("index_version") or ""),
                embed_text_sha256=str(payload.get("embed_text_sha256") or ""),
                embed_text_samples=list(payload.get("embed_text_samples") or []),
            )
        except (OSError, ValueError, json.JSONDecodeError) as ex:
            log.warning(
                "Failed to parse %s (%s); proceeding without manifest.",
                manifest_path, ex,
            )
    return retriever, offline_info, manifest
