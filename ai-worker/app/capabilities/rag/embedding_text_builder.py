"""Production canonical embedding-text builder.

Phase 7.2 promotes the Phase 7.0 ``retrieval_title_section`` format to
the production default. This module owns the *canonical* string shape
that gets handed to the bi-encoder at ingest time. The eval-side
builder (``eval/harness/embedding_text_builder.py``) is now a wrapper
around the symbols defined here so eval-export and production-ingest
produce byte-perfect-equivalent embedding text on the same input.

Why one canonical builder:

  - Phase 7.0 measured a +22pt hit@1 lift driven entirely by the
    embedding text shape — a one-character drift between eval and
    production would silently invalidate that gain.
  - The Phase 7.0 silver index sha256s the exact byte string handed
    to the embedder; if the production ingest builds a different
    string, an offline → online migration would re-encode 135k+
    chunks for nothing.
  - Centralising the format means the bump-the-version posture
    (``EMBEDDING_TEXT_BUILDER_VERSION``) only has to land in one
    place; downstream manifests pick it up automatically.

What is NOT here:

  - V3 prefix variants (``raw`` / ``title`` / ``section`` / ``keyword``
    / ``all``). Those are eval-only experimental knobs and continue
    to live in ``eval/harness/embedding_text_builder.py``. The
    production path never built indexes against them — keeping them
    eval-only avoids growing the production surface with experimental
    debt.
  - Reranker / generation logic. The reranker still consumes the raw
    chunk text (stored in ragmeta), not the embedding text — that
    contract did not change in Phase 7.2 and this module does not
    touch it.

Format contract (the byte string the embedder sees, when all v4 fields
are populated)::

    제목: {title}
    섹션: {section_path joined with ' > '}
    섹션타입: {section_type}

    본문:
    {chunk_text}

Each label-line is dropped if its source is empty — same hygiene as
Phase 7.0: a missing field becomes an absent line, not an empty
"제목: " sentinel that would confuse the embedder.

Format version contract:

  - ``EMBEDDING_TEXT_BUILDER_VERSION`` is the version a downstream
    manifest records so a retriever / migration tool can detect
    silent format drift. Bumping this string is a cache-invalidating
    change — every previously-built index whose
    ``ingest_manifest.json`` carries a stale version must be
    rebuilt OR the index loader must refuse to serve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANT_TITLE_SECTION = "title_section"
VARIANT_RETRIEVAL_TITLE_SECTION = "retrieval_title_section"

# Production-supported variants. Eval may know about more (raw / title /
# section / keyword / all) — those never reach this module.
PRODUCTION_VARIANTS: Tuple[str, ...] = (
    VARIANT_TITLE_SECTION,
    VARIANT_RETRIEVAL_TITLE_SECTION,
)

# Phase 7.2 default. The Phase 7.0 A/B promoted this verdict
# (+22pt hit@1, +21.9pt MRR over title_section) so it lands as the
# production default; ``rag_embedding_text_variant`` settings flip
# rolls back to ``title_section`` if needed.
DEFAULT_PRODUCTION_VARIANT = VARIANT_RETRIEVAL_TITLE_SECTION

# Format version. Increment on any format-breaking change so a stale
# index manifest can be detected at load time. This is the value that
# lives in ingest_manifest.json and gets compared by the retriever's
# format check.
#
# v4-1 = Phase 7.0 / 7.2 canonical format, identical byte-for-byte to
# the Phase 6.3 ``rag_chunks.jsonl.embedding_text`` and the Phase 7.0
# eval export.
EMBEDDING_TEXT_BUILDER_VERSION = "v4-1"


# ---------------------------------------------------------------------------
# Format constants — pinned because every cached index whose embed_text_sha256
# was computed under the previous values would silently desync if any of them
# changed. Tests pin them.
# ---------------------------------------------------------------------------

V4_SECTION_PATH_JOINER = " > "
V4_TITLE_LABEL = "제목"
V4_SECTION_LABEL = "섹션"
V4_SECTION_TYPE_LABEL = "섹션타입"
V4_BODY_LABEL = "본문"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V4EmbeddingTextInput:
    """Per-chunk inputs to :func:`build_v4_embedding_text`.

    Mirrors the Phase 6.3 ``rag_chunks.jsonl`` schema. ``page_title`` is
    the stored chunk-level title (== PageV4.page_title); ``retrieval_title``
    is the Phase 6.3-folded retrieval form ("{work_title} / {page_title}"
    when generic, else == page_title). Both can be empty strings — the
    formatter falls back to the other in that case so the output never
    leaks a sentinel.

    ``section_path`` is the heading list as it appears on the chunk;
    we accept both list-of-strings and a single pre-joined string so a
    caller passing already-collapsed metadata isn't forced to split it.
    """

    chunk_text: str
    page_title: str = ""
    retrieval_title: str = ""
    section_path: Tuple[str, ...] = field(default_factory=tuple)
    section_type: str = ""

    @classmethod
    def from_chunk_record(cls, record: Mapping[str, object]) -> "V4EmbeddingTextInput":
        """Construct from a Phase 6.3 ``rag_chunks.jsonl`` record dict.

        The record-level field names are the Phase 6.3 schema:
        ``chunk_text`` / ``title`` / ``retrieval_title`` /
        ``section_path`` / ``section_type``. We coerce everything to
        the dataclass shape; missing fields default to empty.
        """
        section_path_raw = record.get("section_path") or ()
        if isinstance(section_path_raw, str):
            section_path: Tuple[str, ...] = (section_path_raw,)
        else:
            section_path = tuple(
                str(s) for s in section_path_raw  # type: ignore[union-attr]
                if s
            )
        return cls(
            chunk_text=str(record.get("chunk_text") or ""),
            page_title=str(record.get("title") or ""),
            retrieval_title=str(record.get("retrieval_title") or ""),
            section_path=section_path,
            section_type=str(record.get("section_type") or ""),
        )

    @classmethod
    def from_v3_ingest_chunk(
        cls,
        *,
        chunk_text: str,
        title: str,
        section_name: str,
        retrieval_title: str = "",
        section_type: str = "",
    ) -> "V4EmbeddingTextInput":
        """Construct from v3-shaped ingest fields.

        The current production fixtures (anime_sample.jsonl, kr_sample.jsonl,
        anime_kr.jsonl, …) use the v3 schema:
          - ``title`` (== work title at the doc level)
          - ``sections.<section_name>`` (the dict key is the section name)
          - no ``retrieval_title``, no ``section_path``, no ``section_type``

        The mapping promotes ``section_name`` (a single string) into a
        single-segment ``section_path`` and lets ``retrieval_title`` /
        ``section_type`` default to empty. Under
        ``retrieval_title_section`` the title fallback resolves
        retrieval_title='' → page_title, so v3 corpora produce the
        same byte string under either variant — which is the right
        outcome (v3 has no parent-work prefix to fold in).
        """
        return cls(
            chunk_text=str(chunk_text or ""),
            page_title=str(title or ""),
            retrieval_title=str(retrieval_title or ""),
            section_path=(str(section_name),) if section_name else (),
            section_type=str(section_type or ""),
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _resolve_title(chunk: V4EmbeddingTextInput, *, variant: str) -> str:
    """Pick the title segment per v4 variant.

    For ``title_section``: always page_title (the Phase 6.3 baseline).
    For ``retrieval_title_section``: prefer non-empty retrieval_title,
    else fall back to page_title. The fallback matters because Phase 6.3
    leaves ``retrieval_title`` equal to ``page_title`` for non-generic
    pages — the variant should still produce a sensible string for
    those rows rather than dropping the title segment entirely.
    """
    page_title = (chunk.page_title or "").strip()
    if variant == VARIANT_TITLE_SECTION:
        return page_title
    if variant == VARIANT_RETRIEVAL_TITLE_SECTION:
        retrieval_title = (chunk.retrieval_title or "").strip()
        return retrieval_title or page_title
    raise ValueError(
        f"build_v4_embedding_text does not handle variant {variant!r}; "
        f"expected one of {PRODUCTION_VARIANTS}."
    )


def build_v4_embedding_text(
    chunk: V4EmbeddingTextInput,
    *,
    variant: str = DEFAULT_PRODUCTION_VARIANT,
) -> str:
    """Compose the canonical embedding text for ``chunk`` under ``variant``.

    Default variant is ``retrieval_title_section`` (the Phase 7.0
    verdict). The legacy ``title_section`` variant is preserved so a
    rollback is one config flip away.

    Empty fields drop their label-line entirely; this matches the
    Phase 6.3 rag_chunks.jsonl convention so a downstream sha256
    comparison between Phase 6.3 stored embedding_text, the eval
    export, and the production ingest text yields byte-perfect
    equality.
    """
    if variant not in PRODUCTION_VARIANTS:
        raise ValueError(
            f"Unknown production embedding-text variant {variant!r}; "
            f"expected one of {PRODUCTION_VARIANTS}."
        )

    title = _resolve_title(chunk, variant=variant)
    section_path = tuple(
        s for s in (chunk.section_path or ()) if s and s.strip()
    )
    section_str = V4_SECTION_PATH_JOINER.join(section_path)
    section_type = (chunk.section_type or "").strip()
    body = (chunk.chunk_text or "").strip()

    header_lines = []
    if title:
        header_lines.append(f"{V4_TITLE_LABEL}: {title}")
    if section_str:
        header_lines.append(f"{V4_SECTION_LABEL}: {section_str}")
    if section_type:
        header_lines.append(f"{V4_SECTION_TYPE_LABEL}: {section_type}")

    parts = []
    if header_lines:
        parts.append("\n".join(header_lines))
    if body:
        parts.append(f"{V4_BODY_LABEL}:\n{body}")
    # Phase 6.3 separator between header block and body block is a
    # blank line ("\n\n"); when the header is empty (defence in
    # depth) we just emit the body so we never produce a stray
    # leading "\n\n".
    return "\n\n".join(parts) if parts else ""


def is_known_production_variant(variant: str) -> bool:
    """Convenience predicate for argparse / config validation."""
    return variant in PRODUCTION_VARIANTS


# ---------------------------------------------------------------------------
# Convenience entry point used by ingest.py.
#
# A second, narrower wrapper that takes the v3-shape fields the
# ``IngestService`` already has on hand. Lets the ingest path stay a
# one-liner instead of growing a per-call dataclass build dance.
# ---------------------------------------------------------------------------


def build_embedding_text_from_v3_chunk(
    *,
    chunk_text: str,
    title: str,
    section_name: str,
    retrieval_title: str = "",
    section_type: str = "",
    variant: str = DEFAULT_PRODUCTION_VARIANT,
) -> str:
    """Build an embedding text from v3-shape ingest fields.

    Convenience that ``IngestService`` calls at the chunk loop. v3
    fixtures don't carry ``retrieval_title`` / ``section_type``; both
    default to empty so the call site can pass only what it has. The
    ``retrieval_title_section`` variant falls back to ``title`` in
    that case, so v3 corpora produce semantically-equivalent output
    under either variant.
    """
    return build_v4_embedding_text(
        V4EmbeddingTextInput.from_v3_ingest_chunk(
            chunk_text=chunk_text,
            title=title,
            section_name=section_name,
            retrieval_title=retrieval_title,
            section_type=section_type,
        ),
        variant=variant,
    )
