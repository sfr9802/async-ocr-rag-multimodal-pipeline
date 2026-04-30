"""Phase 7.0 — v4 RAG-chunks export under a chosen embedding-text variant.

Phase 6.3's ``rag_chunks.jsonl`` already carries an ``embedding_text``
field built from ``page_title`` (== chunk ``title``). Phase 7.0 wants
to compare that against ``retrieval_title``-prefixed embeddings without
modifying the source artefacts. This module re-emits the chunk file
with the embedding_text recomputed per variant; everything else
(``chunk_id``, ``doc_id``, ``chunk_text``, metadata, ``display_title``,
``retrieval_title``) is preserved byte-for-byte so a downstream FAISS
build only differs in the embedded string.

Two responsibilities:
  1. Iterate the chunk records lazily (memory-bounded — the file is ~315MB
     and 135k rows so eagerly loading it into a list is wasteful when
     downstream consumers stream it anyway).
  2. Recompute ``embedding_text`` under one of:
       - ``title_section``           (Phase 6.3 baseline, byte-identical)
       - ``retrieval_title_section`` (Phase 7.0 candidate)
     using the v4-aware builder so the format never drifts from what
     Phase 6.3 stored.

Outputs a manifest (``manifest_<variant>.json``) so the audit trail is
machine-readable: source path + sha256, total / changed counts,
variant, schema version. The diff report module reads both manifests
and consumes the exported chunk files for breakdown statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from eval.harness.embedding_text_builder import (
    EMBEDDING_TEXT_VARIANTS,
    V4EmbeddingTextInput,
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_TITLE_SECTION,
    build_v4_embedding_text,
)

log = logging.getLogger(__name__)


V4_EXPORT_VARIANTS: Tuple[str, ...] = (
    VARIANT_TITLE_SECTION,
    VARIANT_RETRIEVAL_TITLE_SECTION,
)


@dataclass(frozen=True)
class V4ChunkExportSummary:
    """Returned from :func:`export_v4_chunks` so the CLI can log + manifest."""

    variant: str
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    total_chunks: int
    changed_embedding_text_count: int
    page_id_count: int
    embed_text_sha256: str


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            buf = fp.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def iter_v4_chunk_records(rag_chunks_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield chunk dicts from ``rag_chunks.jsonl`` (Phase 6.3 schema).

    No filtering; downstream callers decide which records they care
    about. Skips empty / blank lines so a manually edited file with a
    trailing newline does not yield ``None``.
    """
    with Path(rag_chunks_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def recompute_embedding_text(
    record: Dict[str, Any],
    *,
    variant: str,
) -> str:
    """Return the variant-aware embedding_text for a chunk record.

    Wraps ``V4EmbeddingTextInput.from_chunk_record`` + the v4 builder
    so callers don't have to touch the dataclass directly.
    """
    if variant not in V4_EXPORT_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} not supported for v4 export; "
            f"expected one of {V4_EXPORT_VARIANTS}."
        )
    return build_v4_embedding_text(
        V4EmbeddingTextInput.from_chunk_record(record),
        variant=variant,
    )


def export_v4_chunks(
    rag_chunks_path: Path,
    out_path: Path,
    *,
    variant: str,
    write_manifest: bool = True,
    manifest_path: Optional[Path] = None,
) -> V4ChunkExportSummary:
    """Stream chunks from ``rag_chunks_path`` → ``out_path`` under ``variant``.

    Preserves every existing field (``chunk_id``, ``doc_id``,
    ``chunk_text``, metadata, ``display_title``, ``retrieval_title``)
    and overwrites only ``embedding_text``. Counts how many chunks'
    new ``embedding_text`` differs from the source's stored value;
    the count is recorded in the manifest so a downstream diff report
    can sanity-check the change ratio matches the Phase 6.3 spec
    (~33% / 44,759 chunks).
    """
    if variant not in V4_EXPORT_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} not supported for v4 export; "
            f"expected one of {V4_EXPORT_VARIANTS}."
        )
    rag_chunks_path = Path(rag_chunks_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embed_hasher = hashlib.sha256()
    total = 0
    changed = 0
    page_ids: set = set()

    with out_path.open("w", encoding="utf-8") as out_fp:
        for record in iter_v4_chunk_records(rag_chunks_path):
            new_embed = recompute_embedding_text(record, variant=variant)
            if new_embed != record.get("embedding_text"):
                changed += 1
            record["embedding_text"] = new_embed
            doc_id = record.get("doc_id")
            if doc_id:
                page_ids.add(str(doc_id))
            embed_hasher.update(new_embed.encode("utf-8"))
            embed_hasher.update(b"\n")
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    summary = V4ChunkExportSummary(
        variant=variant,
        source_path=str(rag_chunks_path.resolve()),
        source_sha256=_file_sha256(rag_chunks_path),
        output_path=str(out_path.resolve()),
        output_sha256=_file_sha256(out_path),
        total_chunks=total,
        changed_embedding_text_count=changed,
        page_id_count=len(page_ids),
        embed_text_sha256=embed_hasher.hexdigest(),
    )

    if write_manifest:
        manifest_dest = (
            Path(manifest_path)
            if manifest_path is not None
            else out_path.with_name(f"manifest_{variant}.json")
        )
        manifest_dest.write_text(
            json.dumps(_manifest_to_dict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Wrote v4 chunk export manifest %s (variant=%s, changed=%d/%d)",
            manifest_dest, variant, changed, total,
        )
    return summary


def _manifest_to_dict(summary: V4ChunkExportSummary) -> Dict[str, Any]:
    return {
        "schema": "v4-chunk-export-manifest.v1",
        "variant": summary.variant,
        "source_path": summary.source_path,
        "source_sha256": summary.source_sha256,
        "output_path": summary.output_path,
        "output_sha256": summary.output_sha256,
        "total_chunks": summary.total_chunks,
        "changed_embedding_text_count": summary.changed_embedding_text_count,
        "page_id_count": summary.page_id_count,
        "embed_text_sha256": summary.embed_text_sha256,
    }
