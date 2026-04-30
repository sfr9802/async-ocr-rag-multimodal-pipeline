"""Phase 2 — section-aware embedding-text composition for eval experiments.

Pure helper for building the *string* fed to an embedder for a chunk.
Two variant families coexist here:

  1. **v3 prefix variants** (eval-only) — ``raw`` / ``title`` /
     ``section`` / ``title_section`` / ``keyword`` / ``all``. These
     were the Phase 2 experimental knobs over the v3 corpus
     (``anime_namu_v3_token_chunked``). Production never embedded
     against them; they live in this module as legacy eval-side
     experiments and are pinned by tests.

  2. **v4 canonical variants** (production-aligned) — ``title_section``
     and ``retrieval_title_section``. These now live in
     ``app/capabilities/rag/embedding_text_builder`` (Phase 7.2) and
     this module re-exports them so existing eval-side callers keep
     working unchanged. The v4 builder is the *canonical* byte
     producer; eval and production both call into it so an
     ``embed_text_sha256`` computed offline matches the live ingest
     byte-for-byte.

Phase 7.2 split rationale:

  - Centralising the v4 byte format prevents silent drift between
    eval-side ``v4_chunk_export`` (where Phase 7.0's +22pt hit@1
    gain was measured) and the production ingest path (which now
    embeds against the same format).
  - The eval-only v3 variants stay here because production has no
    use for them and Phase 2's regression tests pin the exact byte
    output. Moving them would force production to take on
    experimental code.

Two consumers of this module:

  1. ``BM25EvalRetriever`` — passes the same text it indexes through
     this builder so prefix variants are tokenized consistently.
  2. ``DenseEvalReindex`` — re-embeds an existing corpus with a prefix
     variant for an A/B against the dense baseline.

Variants kept deliberately small and explicit. Each v3 variant is
*additive* over the previous one so a downstream sweep can read the
variant name and predict what the embedding text contains:

  - ``raw``           — chunk.text only (matches production v3 ingest)
  - ``title``         — title prefix + chunk.text
  - ``section``       — section prefix + chunk.text
  - ``title_section`` — title + section + chunk.text (eval-side v3 form;
                        and ALSO the v4-canonical title_section form
                        when called via :func:`build_v4_embedding_text`)
  - ``keyword``       — keyword prefix + chunk.text (when keywords exist)
  - ``all``           — title + section + keyword + chunk.text

When the requested prefix isn't available on the chunk (e.g. variant
``title`` but ``title is None``), the v3 builder silently falls back to
the raw text — keeping the eval input runnable across mixed datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

# Phase 7.2: v4-canonical builder lives in production. We re-export the
# names so existing eval-side imports
# (``from eval.harness.embedding_text_builder import V4EmbeddingTextInput``)
# keep working byte-for-byte unchanged.
from app.capabilities.rag.embedding_text_builder import (
    EMBEDDING_TEXT_BUILDER_VERSION,
    V4_BODY_LABEL,
    V4_SECTION_LABEL,
    V4_SECTION_PATH_JOINER,
    V4_SECTION_TYPE_LABEL,
    V4_TITLE_LABEL,
    V4EmbeddingTextInput,
    build_v4_embedding_text,
    is_known_production_variant,
)


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
    """Compose the eval-only v3 prefix embedding text for ``chunk``.

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
# Re-exported v4 symbols (canonical owner: production module).
#
# ``V4EmbeddingTextInput`` and ``build_v4_embedding_text`` now live in
# ``app/capabilities/rag/embedding_text_builder``. They are re-exported
# at this module's top by the import block above so:
#
#   from eval.harness.embedding_text_builder import V4EmbeddingTextInput
#
# continues to work after Phase 7.2's centralisation. The byte output is
# identical to the pre-Phase 7.2 eval-only implementation — Phase 7.2's
# parity tests pin that.
#
# Format constants (``V4_TITLE_LABEL`` etc.) are also re-exported so any
# downstream test or audit tool that reads them off this module keeps
# working.
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_KEYWORD_LIMIT",
    "EMBEDDING_TEXT_BUILDER_VERSION",
    "EMBEDDING_TEXT_VARIANTS",
    "EmbeddingTextInput",
    "KEYWORD_JOINER",
    "PREFIX_SEPARATOR",
    "VARIANT_ALL",
    "VARIANT_KEYWORD",
    "VARIANT_RAW",
    "VARIANT_RETRIEVAL_TITLE_SECTION",
    "VARIANT_SECTION",
    "VARIANT_TITLE",
    "VARIANT_TITLE_SECTION",
    "V4EmbeddingTextInput",
    "V4_BODY_LABEL",
    "V4_SECTION_LABEL",
    "V4_SECTION_PATH_JOINER",
    "V4_SECTION_TYPE_LABEL",
    "V4_TITLE_LABEL",
    "build_embedding_text",
    "build_v4_embedding_text",
    "is_known_production_variant",
    "is_known_variant",
]
