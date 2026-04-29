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

EMBEDDING_TEXT_VARIANTS: Tuple[str, ...] = (
    VARIANT_RAW,
    VARIANT_TITLE,
    VARIANT_SECTION,
    VARIANT_TITLE_SECTION,
    VARIANT_KEYWORD,
    VARIANT_ALL,
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
    """
    if variant not in EMBEDDING_TEXT_VARIANTS:
        raise ValueError(
            f"Unknown embedding-text variant {variant!r}; "
            f"expected one of {EMBEDDING_TEXT_VARIANTS}."
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
