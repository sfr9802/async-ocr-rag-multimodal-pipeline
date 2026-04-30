"""Phase 2 — section-aware embedding-text composition for eval experiments.

Pure helper for building the *string* fed to an embedder for a chunk.
The production ingest path (``app.capabilities.rag.ingest``) embeds raw
chunk text only; this module lets eval-side experiments build prefixed
variants (``"<title>\\n<section>\\n<text>"`` etc.) without touching the
production ingest / retriever code. Two consumers:

  1. ``BM25EvalRetriever`` — passes the same text it indexes through
     this builder so prefix variants are tokenized consistently.
  2. ``DenseEvalReindex`` (later in this Phase) — re-embeds an existing
     corpus with a prefix variant for an A/B against the dense baseline.

Variants kept deliberately small and explicit. Each variant is *additive*
over the previous one so a downstream sweep can read the variant name
and predict what the embedding text contains:

  - ``raw``           — chunk.text only (matches production)
  - ``title``         — title prefix + chunk.text
  - ``section``       — section prefix + chunk.text
  - ``title_section`` — title + section + chunk.text (most common)
  - ``keyword``       — keyword prefix + chunk.text (when keywords exist)
  - ``all``           — title + section + keyword + chunk.text

When the requested prefix isn't available on the chunk (e.g. variant
``title`` but ``title is None``), the builder silently falls back to
the raw text — keeping the eval input runnable across mixed datasets.
The resulting string is *exactly* what gets embedded / tokenized; the
pipeline does no further mutation on it.

This module has zero deps and no side effects so it composes safely
with the ``run_retrieval_eval`` harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# Public variant labels — surfaced as module-level constants so the
# sweep driver can reference them by name and tests can pin them.
VARIANT_RAW = "raw"
VARIANT_TITLE = "title"
VARIANT_SECTION = "section"
VARIANT_TITLE_SECTION = "title_section"
VARIANT_KEYWORD = "keyword"
VARIANT_ALL = "all"
# Phase 7.0 — v4-aware retrieval-title variant. Pulls ``retrieval_title``
# (Phase 6.3 schema field) when set; falls back to ``page_title``. Used
# only by the v4 ``build_v4_embedding_text`` path; the legacy variants
# above continue to operate on the v3 EmbeddingTextInput interface.
VARIANT_RETRIEVAL_TITLE_SECTION = "retrieval_title_section"

EMBEDDING_TEXT_VARIANTS: Tuple[str, ...] = (
    VARIANT_RAW,
    VARIANT_TITLE,
    VARIANT_SECTION,
    VARIANT_TITLE_SECTION,
    VARIANT_KEYWORD,
    VARIANT_ALL,
    VARIANT_RETRIEVAL_TITLE_SECTION,
)

# Default separator between prefix segments and the chunk body. Newline
# matches what cross-encoder tokenizers do best with — they split on
# whitespace anyway and the newline keeps the boundary visible in
# debug printouts.
PREFIX_SEPARATOR = "\n"
# Joiner inside the keyword prefix line: space-separated lowercase
# keywords, capped at the first 8 to keep prefix length bounded.
KEYWORD_JOINER = " "
DEFAULT_KEYWORD_LIMIT = 8


@dataclass(frozen=True)
class EmbeddingTextInput:
    """Per-chunk inputs to ``build_embedding_text``.

    All optional except ``text``. Empty / whitespace-only fields are
    treated identically to ``None`` — the prefix segment is dropped
    from the output rather than left as an empty line. This matches
    how the production ingest treats missing metadata: silently absent,
    not a sentinel.
    """

    text: str
    title: Optional[str] = None
    section: Optional[str] = None
    keywords: Tuple[str, ...] = field(default_factory=tuple)


def build_embedding_text(
    chunk: EmbeddingTextInput,
    *,
    variant: str = VARIANT_RAW,
    keyword_limit: int = DEFAULT_KEYWORD_LIMIT,
    separator: str = PREFIX_SEPARATOR,
) -> str:
    """Compose the embedding text for ``chunk`` per ``variant``.

    Returns the assembled string. Variants that ask for a prefix the
    chunk doesn't carry (e.g. ``title`` but ``chunk.title is None``)
    silently degrade — the missing segment is dropped, NOT replaced
    with a sentinel, so a sweep over (variant) on a dataset where some
    rows lack metadata still produces interpretable embeddings rather
    than poisoned ones.

    The v4-only variant ``retrieval_title_section`` is rejected here —
    it requires the v4 schema fields (``retrieval_title`` /
    ``section_path`` / ``section_type``) carried by
    :class:`V4EmbeddingTextInput`, and silently treating it as a
    ``title_section`` would conflate variants in any sweep that mixes
    them. Callers must route through :func:`build_v4_embedding_text`.
    """
    if variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
        )
    if variant == VARIANT_RETRIEVAL_TITLE_SECTION:
        raise ValueError(
            f"Variant {variant!r} requires the v4 schema "
            "(retrieval_title / section_path / section_type); "
            "use build_v4_embedding_text with V4EmbeddingTextInput."
        )
    text = (chunk.text or "").strip()
    if variant == VARIANT_RAW:
        return text

    segments: List[str] = []
    want_title = variant in (VARIANT_TITLE, VARIANT_TITLE_SECTION, VARIANT_ALL)
    want_section = variant in (
        VARIANT_SECTION, VARIANT_TITLE_SECTION, VARIANT_ALL,
    )
    want_keyword = variant in (VARIANT_KEYWORD, VARIANT_ALL)

    if want_title:
        title = (chunk.title or "").strip()
        if title:
            segments.append(title)
    if want_section:
        section = (chunk.section or "").strip()
        if section:
            segments.append(section)
    if want_keyword:
        kw_segment = _format_keywords(chunk.keywords, limit=keyword_limit)
        if kw_segment:
            segments.append(kw_segment)
    segments.append(text)
    # Filter out any empties that snuck through (defence in depth).
    return separator.join(s for s in segments if s)


def _format_keywords(
    keywords: Iterable[str],
    *,
    limit: int,
) -> str:
    """Cap keyword list and join into a single space-separated line.

    Keywords are deduplicated case-insensitively (preserving first
    occurrence's original casing) so the prefix doesn't repeat tokens.
    Empties are dropped. ``limit`` caps the count after dedup so the
    prefix length stays bounded even if the source list is long.
    """
    if not keywords:
        return ""
    seen: set = set()
    out: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        token = str(kw).strip()
        if not token:
            continue
        norm = token.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(token)
        if len(out) >= max(1, int(limit)):
            break
    return KEYWORD_JOINER.join(out)


def is_known_variant(variant: str) -> bool:
    """Convenience predicate the sweep CLI uses for argument validation."""
    return variant in EMBEDDING_TEXT_VARIANTS


# ---------------------------------------------------------------------------
# Phase 7.0 — v4-aware embedding text builder.
#
# The Phase 6.3 ``rag_chunks.jsonl`` already ships an ``embedding_text``
# field that uses a Korean-prefixed format ("제목: ... \n섹션: ... \n
# 섹션타입: ... \n\n본문:\n..."). Phase 7.0 needs to compare that exact
# format against an otherwise-identical variant that swaps ``page_title``
# for ``retrieval_title``. We therefore reproduce the Phase 6.3 format
# byte-for-byte here so an A/B run is meaningful (any divergence from
# Phase 6.3's stored embedding_text would conflate "format change" with
# "title change").
# ---------------------------------------------------------------------------


# Phase 6.3 produces section_path as a list of headings (e.g. ["음악",
# "주제가", "OP"]). The stored embedding_text joins those with " > ", so
# we mirror that exact joiner.
V4_SECTION_PATH_JOINER = " > "

# The four labels are part of the Phase 6.3 embedding_text contract.
# Changing any of them silently invalidates every cached index whose
# embed_text_sha256 was computed against the previous format, so they
# live as constants the tests can pin.
V4_TITLE_LABEL = "제목"
V4_SECTION_LABEL = "섹션"
V4_SECTION_TYPE_LABEL = "섹션타입"
V4_BODY_LABEL = "본문"


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
    def from_chunk_record(cls, record: dict) -> "V4EmbeddingTextInput":
        """Construct from a Phase 6.3 ``rag_chunks.jsonl`` record dict."""
        section_path = record.get("section_path") or ()
        if isinstance(section_path, str):
            # accept a pre-joined string and treat as a single-segment path
            section_path = (section_path,)
        else:
            section_path = tuple(str(s) for s in section_path if s)
        return cls(
            chunk_text=str(record.get("chunk_text") or ""),
            page_title=str(record.get("title") or ""),
            retrieval_title=str(record.get("retrieval_title") or ""),
            section_path=section_path,
            section_type=str(record.get("section_type") or ""),
        )


def _resolve_v4_title(chunk: V4EmbeddingTextInput, *, variant: str) -> str:
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
        f"expected one of "
        f"({VARIANT_TITLE_SECTION!r}, {VARIANT_RETRIEVAL_TITLE_SECTION!r})."
    )


def build_v4_embedding_text(
    chunk: V4EmbeddingTextInput,
    *,
    variant: str,
) -> str:
    """Compose the Phase 6.3-format embedding text under ``variant``.

    Output format (matches what Phase 6.3 stores in ``rag_chunks.jsonl``
    when all fields are present)::

        제목: {title}
        섹션: {section_path joined with ' > '}
        섹션타입: {section_type}

        본문:
        {chunk_text}

    Each label-line is dropped if its source is empty — same hygiene
    as the v3 ``build_embedding_text``: missing metadata becomes a
    silently absent line, not an empty "제목: " sentinel that would
    confuse the embedder.
    """
    if variant not in (VARIANT_TITLE_SECTION, VARIANT_RETRIEVAL_TITLE_SECTION):
        raise ValueError(
            f"build_v4_embedding_text only supports "
            f"{VARIANT_TITLE_SECTION!r} and "
            f"{VARIANT_RETRIEVAL_TITLE_SECTION!r}; got {variant!r}."
        )

    title = _resolve_v4_title(chunk, variant=variant)
    section_path = tuple(s for s in (chunk.section_path or ()) if s and s.strip())
    section_str = V4_SECTION_PATH_JOINER.join(section_path)
    section_type = (chunk.section_type or "").strip()
    body = (chunk.chunk_text or "").strip()

    header_lines: List[str] = []
    if title:
        header_lines.append(f"{V4_TITLE_LABEL}: {title}")
    if section_str:
        header_lines.append(f"{V4_SECTION_LABEL}: {section_str}")
    if section_type:
        header_lines.append(f"{V4_SECTION_TYPE_LABEL}: {section_type}")

    parts: List[str] = []
    if header_lines:
        parts.append("\n".join(header_lines))
    if body:
        parts.append(f"{V4_BODY_LABEL}:\n{body}")
    # The Phase 6.3 separator between header block and body block is a
    # blank line ("\n\n"); when the header is empty (defence in depth)
    # we just emit the body.
    return "\n\n".join(parts) if parts else ""
