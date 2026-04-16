"""Multimodal fusion helper.

Takes the three available signals for a multimodal v1 job —
  1. optional user question (INPUT_TEXT)
  2. OCR-extracted text (OcrDocumentResult.full_text)
  3. visual description (VisionDescriptionResult)
— and produces a deterministic `FusionResult` whose two output fields
drive the rest of the pipeline:

  - `retrieval_query`: the short query string handed to the existing
    text-RAG retriever. This is what actually hits the FAISS index, so
    it has to be a clean, short piece of text — not the full fused
    context.

  - `fused_context`: a longer, structured markdown block that will be
    injected as a synthetic retrieval chunk into the generation step.
    This is how the OCR + vision signal actually ends up in the
    FINAL_RESPONSE: the generator sees the fused block as its first
    "retrieved" chunk and grounds the short answer on it.

Everything here is deterministic. Given the same `(user_question,
ocr_text, vision)` tuple the output is byte-identical, which makes
the fusion logic inspectable (log the FusionResult, diff it against
expectations, etc.) and cleanly testable without any real models.

Deliberately kept as a module-level function instead of hiding the
logic inside the capability's `run()` method — ops should be able to
reason about "what did the retriever actually see?" without reading
the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.capabilities.multimodal.vision_provider import VisionDescriptionResult


# Tunables — exposed as function parameters so tests and future ops
# can override without editing this file. Defaults match v1 scope:
# short enough to keep the embedder happy, long enough to preserve
# a useful chunk of OCR context.
DEFAULT_MAX_QUERY_CHARS = 400
DEFAULT_MAX_OCR_PREVIEW_CHARS = 1200
# Questions with strictly fewer than this many whitespace-split tokens
# get enriched with OCR keywords. "what is the total?" → 4 tokens → enriched.
# "summarize the full invoice totals for me" → 7 tokens → not enriched.
DEFAULT_SHORT_QUERY_WORDS = 5


@dataclass(frozen=True)
class FusionResult:
    """What the fusion helper produces.

    Fields:
      retrieval_query:  short string fed to Retriever.retrieve(...)
      fused_context:    structured markdown block used as a synthetic
                        grounding chunk for the generator
      sources:          ordered list of signals that actually
                        contributed to the retrieval_query / context.
                        Values come from `{"user_question", "ocr_text",
                        "vision_description"}`.
      warnings:         non-fatal fusion-layer warnings (e.g. "no
                        inputs supplied, using default query")
    """

    retrieval_query: str
    fused_context: str
    sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_fusion(
    *,
    user_question: Optional[str],
    ocr_text: str,
    vision: Optional[VisionDescriptionResult],
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    max_ocr_preview_chars: int = DEFAULT_MAX_OCR_PREVIEW_CHARS,
    short_query_words: int = DEFAULT_SHORT_QUERY_WORDS,
) -> FusionResult:
    """Build a retrieval query + fused context from question + OCR + vision.

    The function is intentionally short and deterministic: given the
    same `(user_question, ocr_text, vision)` tuple it produces
    byte-identical output, so the MULTIMODAL_TRACE artifact can be
    diffed in tests and in ops.

    Fusion rules (in priority order):
      1. If the user supplied a non-blank question, that IS the
         retrieval query. If it is unusually short (< `short_query_words`
         words) and OCR text is available, a small OCR keyword suffix
         is appended so retrieval has something to anchor on.
      2. Else, fall back to an OCR text head (first `max_query_chars`).
      3. Else, fall back to the vision caption.
      4. Else, use a neutral default query and record a warning.

    The `fused_context` block is always emitted with the same three
    sections — even when a section is empty it is still present with
    an explicit "(empty)" marker, so downstream consumers can do
    deterministic parsing.
    """

    sources: List[str] = []
    warnings: List[str] = []

    question_clean = (user_question or "").strip()
    ocr_clean = (ocr_text or "").strip()
    vision_caption = vision.caption.strip() if vision and vision.caption else ""

    # ------------------------------------------------------------------
    # 1. Decide retrieval_query
    # ------------------------------------------------------------------
    if question_clean:
        retrieval_query = question_clean
        sources.append("user_question")

        if len(question_clean.split()) < short_query_words and ocr_clean:
            # Enrich short user questions with the first OCR tokens so
            # the FAISS lookup has material to embed. The boundary of 4
            # words is a heuristic — a query like "price?" is useless
            # to an embedder on its own; "price? ... invoice total
            # amount due" is something a retrieval model can actually
            # score against.
            keyword_prefix = " ".join(ocr_clean.split()[:20])
            retrieval_query = (
                f"{retrieval_query} | OCR context: {keyword_prefix}"
            )
            sources.append("ocr_text")
    elif ocr_clean:
        retrieval_query = ocr_clean[:max_query_chars].strip()
        sources.append("ocr_text")
    elif vision_caption:
        retrieval_query = vision_caption[:max_query_chars].strip()
        sources.append("vision_description")
    else:
        retrieval_query = "describe the submitted document"
        warnings.append(
            "fusion had no user question, no OCR text, and no vision "
            "description — retrieval will use the default query"
        )

    # Hard cap so a huge OCR dump doesn't blow up the embedder.
    if len(retrieval_query) > max_query_chars:
        retrieval_query = retrieval_query[: max_query_chars - 3].rstrip() + "..."

    # ------------------------------------------------------------------
    # 2. Build fused_context — always three sections, always in order
    # ------------------------------------------------------------------
    context_lines: List[str] = []

    context_lines.append("### User question")
    context_lines.append(question_clean if question_clean else "(none supplied)")
    context_lines.append("")

    context_lines.append("### Extracted text (OCR)")
    if ocr_clean:
        if len(ocr_clean) > max_ocr_preview_chars:
            preview = ocr_clean[: max_ocr_preview_chars - 3] + "..."
            warnings.append(
                f"OCR text was {len(ocr_clean)} chars — truncated to "
                f"{max_ocr_preview_chars} in the fused context"
            )
        else:
            preview = ocr_clean
        context_lines.append(preview)
        if "ocr_text" not in sources:
            sources.append("ocr_text")
    else:
        context_lines.append("(empty — OCR returned no text)")
    context_lines.append("")

    context_lines.append("### Visual description")
    if vision and vision_caption:
        context_lines.append(vision_caption)
        if vision.details:
            context_lines.append("")
            for detail in vision.details:
                context_lines.append(f"- {detail}")
        if "vision_description" not in sources:
            sources.append("vision_description")
    else:
        context_lines.append("(unavailable — no vision description returned)")

    fused_context = "\n".join(context_lines)

    return FusionResult(
        retrieval_query=retrieval_query,
        fused_context=fused_context,
        sources=sources,
        warnings=warnings,
    )
