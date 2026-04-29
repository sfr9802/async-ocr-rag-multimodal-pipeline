"""Eval-only reranker input passage formatter + wrapper.

The bge-reranker-v2-m3 cross-encoder scores a (query, passage) pair
where ``passage`` is whatever text we hand it. The production
``CrossEncoderReranker`` builds the passage from ``chunk.text`` only,
truncated to ``text_max_chars``. This module lets the eval harness
swap out that passage construction *without* touching production code:

  - ``format_passage(...)`` builds the passage string for one chunk
    according to a named format (``chunk_only`` / ``title_plus_chunk``
    / ``title_section_plus_chunk`` / ``compact_metadata_plus_chunk``).
  - ``FormattingRerankerWrapper`` is a ``RerankerProvider`` that wraps
    the real reranker. Per ``rerank()`` call it builds a list of
    formatted clones of the input chunks (same chunk_id / doc_id /
    section / score, only the ``text`` field replaced by the formatted
    string), forwards them to the wrapped reranker, then reattaches the
    returned ``rerank_score`` to the *original* chunks before returning
    the ranked list.

Why clone chunks?
  - ``RetrievedChunk`` is a frozen dataclass; we have to rebuild rather
    than mutate.
  - The ``WideRetrievalEvalAdapter`` already runs phase 4 (title cap on
    final) over the reranker's output. That cap reads ``doc_id`` /
    ``title_provider(chunk)``, never ``chunk.text``, so it is safe for
    the wrapper to return the *original* (unformatted) chunks with the
    rerank_score swapped in. Returning the formatted clones would also
    work but would leak prefix-augmented text into downstream consumers
    that don't expect it.

The wrapper records each call's input previews on
``last_input_previews`` so the audit step in
``confirm_reranker_input_format`` can surface what the cross-encoder
actually saw — including ``has_title`` / ``has_section`` / ``truncated``
flags that match the ``CrossEncoderReranker`` truncation behavior.

Production code is *not* modified — this is an eval-only adapter that
delegates to whatever ``RerankerProvider`` it wraps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.capabilities.rag.generation import RetrievedChunk

from eval.harness.wide_retrieval_helpers import TitleProvider

log = logging.getLogger(__name__)


# Public list of supported formats — order is the canonical sweep order
# the report writer surfaces. Add new formats by appending to keep prior
# artefact diffs stable.
RERANKER_INPUT_FORMATS = (
    "chunk_only",
    "title_plus_chunk",
    "title_section_plus_chunk",
    "compact_metadata_plus_chunk",
)


def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def format_passage(
    *,
    fmt: str,
    chunk_text: Optional[str],
    title: Optional[str],
    section_path: Optional[str],
) -> str:
    """Return the passage string the reranker should score.

    Behaviour per format:

      ``chunk_only``
        Returns the chunk text untouched (modulo a left-strip on the
        chunk text — cross-encoder tokenizers do not benefit from
        leading whitespace).
      ``title_plus_chunk``
        ``제목: {title}\\n본문: {text}``. Falls back to ``chunk_only``
        when title is missing — never emits a stray ``제목: ``.
      ``title_section_plus_chunk``
        ``제목: {title}\\n섹션: {section}\\n본문: {text}``, omitting
        any line whose value is missing. The body line is always
        present so the reranker has something to score on.
      ``compact_metadata_plus_chunk``
        ``[{title} / {section}]\\n{text}``. Compact form for the
        cross-encoder's ``text_max_chars`` budget — same semantic
        signal as ``title_section_plus_chunk`` but ~30 chars shorter
        for the typical Korean title+section.

    The function does NOT truncate to ``text_max_chars``; the wrapped
    cross-encoder applies its own truncation on the formatted result.
    Callers wanting a truncated *preview* should slice the return
    value themselves.
    """
    body = (chunk_text or "").lstrip()
    title_n = _normalize(title)
    section_n = _normalize(section_path)

    if fmt == "chunk_only":
        return body

    if fmt == "title_plus_chunk":
        if not title_n:
            return body
        return f"제목: {title_n}\n본문: {body}"

    if fmt == "title_section_plus_chunk":
        lines: List[str] = []
        if title_n:
            lines.append(f"제목: {title_n}")
        if section_n:
            lines.append(f"섹션: {section_n}")
        lines.append(f"본문: {body}")
        return "\n".join(lines)

    if fmt == "compact_metadata_plus_chunk":
        if title_n and section_n:
            head = f"[{title_n} / {section_n}]\n"
        elif title_n:
            head = f"[{title_n}]\n"
        elif section_n:
            head = f"[{section_n}]\n"
        else:
            head = ""
        return head + body

    raise ValueError(
        f"unknown reranker_input_format: {fmt!r}; expected one of "
        f"{RERANKER_INPUT_FORMATS}"
    )


@dataclass
class FormattingPreview:
    """One chunk's formatted passage, captured before the cross-encoder.

    The wrapper stores a list of these on ``last_input_previews`` after
    each ``rerank()`` call so the audit step can see exactly what the
    cross-encoder was scoring per chunk — including whether the variant
    prefix actually surfaced (``has_title`` / ``has_section``) and
    whether the formatted text would be truncated by the cross-encoder's
    ``text_max_chars`` cap.
    """

    chunk_id: str
    doc_id: str
    title: Optional[str]
    section: Optional[str]
    fmt: str
    preview: str
    formatted_length: int
    has_title: bool
    has_section: bool
    truncated: bool


def _passage_carries_title(text: str, title: Optional[str]) -> bool:
    """True if ``text`` contains the chunk's doc title (case-folded).

    Mirrors ``variant_comparison._passage_carries_title`` but checks
    *anywhere* in the formatted prefix window rather than only the
    leading 200 chars — the formatter may emit ``[title / section]\\n``
    where the title sits a few chars in. We bound the search to the
    first 800 chars so a chunk that happens to mention the title in its
    body doesn't false-positive.
    """
    if not title:
        return False
    needle = str(title).strip().casefold()
    if not needle:
        return False
    haystack = (text or "")[: max(len(needle) * 8, 800)].casefold()
    return needle in haystack


def _passage_carries_section(text: str, section: Optional[str]) -> bool:
    if not section:
        return False
    needle = str(section).strip().casefold()
    if not needle:
        return False
    haystack = (text or "")[: max(len(needle) * 8, 800)].casefold()
    return needle in haystack


class FormattingRerankerWrapper:
    """Eval-only ``RerankerProvider`` that re-formats passages.

    Wraps a real ``RerankerProvider``. Every ``rerank()`` call:

      1. Builds a list of formatted ``RetrievedChunk`` clones (same
         chunk_id / doc_id / section / score; only ``text`` replaced by
         the format's output).
      2. Forwards the formatted list to the base reranker.
      3. Maps the base's ranked output back to the original chunks by
         ``chunk_id``, copying the ``rerank_score`` over.
      4. Returns the ranked list of ORIGINAL chunks.

    Step 4's "original chunks" choice keeps downstream consumers
    (``WideRetrievalEvalAdapter`` phase 4 title cap, eval metric
    aggregator) reading the un-prefixed text — the formatting is a
    *reranker-only* signal, not a corpus mutation.

    Title lookup uses the supplied ``title_provider`` (typically
    ``DocTitleResolver.title_provider()``); when missing or returning
    None for a chunk the format falls back to chunk-only.

    Audit hook:
      ``last_input_previews`` is a list of ``FormattingPreview`` for
      the most recent ``rerank()`` call. Reset to ``[]`` at the start
      of every call so the eval driver can pick up a per-query view.
    """

    def __init__(
        self,
        base: Any,
        *,
        fmt: str,
        title_provider: Optional[TitleProvider] = None,
        record_input_previews: bool = True,
        preview_max_chars: int = 600,
        truncation_threshold_chars: int = 800,
    ) -> None:
        if fmt not in RERANKER_INPUT_FORMATS:
            raise ValueError(
                f"unknown reranker_input_format: {fmt!r}; expected one "
                f"of {RERANKER_INPUT_FORMATS}"
            )
        self._base = base
        self._fmt = fmt
        self._title_provider = title_provider
        self._record_input_previews = bool(record_input_previews)
        self._preview_max_chars = max(1, int(preview_max_chars))
        self._truncation_threshold_chars = max(
            1, int(truncation_threshold_chars)
        )
        self._last_input_previews: List[FormattingPreview] = []

    @property
    def name(self) -> str:
        base_name = getattr(self._base, "name", "reranker")
        return f"{base_name}+fmt={self._fmt}"

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def last_input_previews(self) -> List[FormattingPreview]:
        return list(self._last_input_previews)

    @property
    def last_breakdown_ms(self) -> Optional[Dict[str, float]]:
        """Forward the wrapped reranker's per-stage breakdown if any.

        The eval ``run_retrieval_eval`` aggregator reads
        ``last_breakdown_ms`` off the *retriever-side* reranker (via
        ``WideRetrievalEvalAdapter`` returning it on the report);
        forwarding here keeps that contract intact for runs that flip
        ``collect_stage_timings=True`` on the wrapped reranker.
        """
        return getattr(self._base, "last_breakdown_ms", None)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            self._last_input_previews = []
            return []
        k_int = max(0, int(k))
        if k_int == 0:
            self._last_input_previews = []
            return []

        formatted: List[RetrievedChunk] = []
        previews: List[FormattingPreview] = []
        original_by_id: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            original_by_id[chunk.chunk_id] = chunk
            title = self._lookup_title(chunk)
            section_path = chunk.section or None
            formatted_text = format_passage(
                fmt=self._fmt,
                chunk_text=chunk.text,
                title=title,
                section_path=section_path,
            )
            formatted.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section=chunk.section,
                text=formatted_text,
                score=chunk.score,
                rerank_score=chunk.rerank_score,
            ))
            if self._record_input_previews:
                preview = formatted_text[: self._preview_max_chars]
                previews.append(FormattingPreview(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=title,
                    section=section_path,
                    fmt=self._fmt,
                    preview=preview,
                    formatted_length=len(formatted_text),
                    has_title=_passage_carries_title(
                        formatted_text, title,
                    ),
                    has_section=_passage_carries_section(
                        formatted_text, section_path,
                    ),
                    truncated=(
                        len(formatted_text)
                        > self._truncation_threshold_chars
                    ),
                ))
        self._last_input_previews = previews

        ranked_formatted = self._base.rerank(query, formatted, k_int)

        out: List[RetrievedChunk] = []
        for r in ranked_formatted:
            origin = original_by_id.get(r.chunk_id)
            if origin is None:
                # Defensive: if the base reranker returned a chunk we
                # didn't supply, surface it untouched. Should not happen
                # with the production CrossEncoderReranker, which only
                # returns elements from its input list.
                out.append(r)
                continue
            out.append(RetrievedChunk(
                chunk_id=origin.chunk_id,
                doc_id=origin.doc_id,
                section=origin.section,
                text=origin.text,
                score=origin.score,
                rerank_score=r.rerank_score,
            ))
        return out

    # -- internals -------------------------------------------------------

    def _lookup_title(self, chunk: Any) -> Optional[str]:
        if self._title_provider is None:
            return None
        try:
            return self._title_provider(chunk)
        except Exception:  # noqa: BLE001 — defensive
            log.warning(
                "title_provider failed for chunk %r; falling back to None",
                getattr(chunk, "chunk_id", "?"),
            )
            return None
