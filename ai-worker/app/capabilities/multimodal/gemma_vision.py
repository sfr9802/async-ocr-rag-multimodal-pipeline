"""GemmaVisionProvider — vision over the shared LlmChatProvider seam.

Gemma 4 E2B is natively multimodal: the same /api/chat endpoint that
drives the agent router / critic / rewriter / query parser also accepts
image bytes alongside the text prompt. Wiring this provider in makes
the whole stack — router, critic, rewriter, parser, vision — reachable
through a single local model. Single-model on-prem deployment is the
story; `multimodal_vision_provider=gemma` + `llm_backend=ollama` is
the knob.

The provider is the minimal adapter between
``VisionDescriptionProvider`` (bytes in → caption out) and
``LlmChatProvider.chat_vision`` (prompt + image → raw text). It:

  1. Base64-encodes the image bytes.
  2. Builds a Korean/English-aware instruction prompt.
  3. Calls ``chat.chat_vision(...)`` with a token budget (Gemma 4's
     image-resolution knob — higher budget ≈ higher fidelity for
     OCR-heavy document pages, lower is fine for generic captioning).
  4. Splits the response into ``caption`` + ``details`` using the same
     first-sentence / bullet convention as ClaudeVisionProvider so the
     MULTIMODAL fusion layer treats the output identically.

Errors: any ``LlmChatError`` from the chat provider becomes a
``VisionError('VLM_API_FAILED', …)`` — same code ClaudeVisionProvider
uses, so the capability layer's fallback + warning wiring does not need
to special-case gemma.

Token-budget tiers (pass via ``token_budget=`` kwarg on
``describe_image`` or via the registry default):

  *  140 — generic captioning (portrait, photo, diagram sketch)
  *  280 — default, mixed-content document page
  *  560 — multi-column layout, dense screenshot
  * 1120 — OCR-heavy scanned page, long table, small-font text

Lower budgets reduce latency + VRAM; higher budgets let Gemma transcribe
more visible text before the reply is truncated.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from app.capabilities.multimodal.vision_provider import (
    VisionDescriptionProvider,
    VisionDescriptionResult,
    VisionError,
)
from app.clients.llm_chat import LlmChatError, LlmChatProvider

log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "Describe this {subject} in Korean if the image contains Korean text, "
    "otherwise in English. Include (1) a single-sentence caption of the "
    "main subject, (2) 3-5 bullet points covering layout, visible objects, "
    "tables, or charts, and (3) a verbatim transcription of any visible "
    "text. Do not speculate beyond what is visible."
)


class GemmaVisionProvider(VisionDescriptionProvider):
    """Vision provider that reuses the shared LlmChatProvider.

    The chat argument MUST advertise ``capabilities['vision'] is True``.
    Passing a non-vision chat provider (NoOp, a text-only Ollama tag,
    …) raises ``ValueError`` at init — the registry catches it and
    downgrades to the heuristic provider with a warning.
    """

    def __init__(
        self,
        chat: LlmChatProvider,
        *,
        default_token_budget: int = 280,
    ) -> None:
        if not chat.capabilities.get("vision"):
            raise ValueError(
                f"GemmaVisionProvider requires a chat provider with "
                f"capabilities['vision']=True; got {chat.name!r} with "
                f"capabilities={chat.capabilities!r}."
            )
        if default_token_budget <= 0:
            raise ValueError(
                f"default_token_budget must be positive; got {default_token_budget}."
            )
        self._chat = chat
        self._default_token_budget = int(default_token_budget)

    @property
    def name(self) -> str:
        return f"gemma-{self._chat.name}"

    def describe_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
        hint: Optional[str] = None,
        page_number: int = 1,
        token_budget: Optional[int] = None,
    ) -> VisionDescriptionResult:
        budget = int(token_budget) if token_budget else self._default_token_budget
        if budget <= 0:
            raise VisionError(
                "VLM_BAD_REQUEST",
                f"token_budget must be positive; got {token_budget}.",
            )

        subject = (hint.strip() if hint else "") or "document page"
        prompt = _PROMPT_TEMPLATE.format(subject=subject)

        started_at = time.perf_counter()
        try:
            raw_text = self._chat.chat_vision(
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=(mime_type or "image/png"),
                max_tokens=budget,
                temperature=0.2,
            )
        except LlmChatError as ex:
            raise VisionError(
                "VLM_API_FAILED",
                f"Gemma vision call via {self._chat.name} failed: {ex}",
            ) from ex

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if not raw_text.strip():
            raise VisionError(
                "VLM_BAD_RESPONSE",
                f"Gemma vision returned an empty response via {self._chat.name}.",
            )

        caption, details = _parse_response(raw_text)

        log.info(
            "GemmaVisionProvider described image: page=%d backend=%s "
            "token_budget=%d latency_ms=%.2f caption_len=%d detail_count=%d",
            page_number, self._chat.name, budget, elapsed_ms,
            len(caption), len(details),
        )

        return VisionDescriptionResult(
            provider_name=self.name,
            caption=caption,
            details=details,
            warnings=[],
            latency_ms=round(elapsed_ms, 3),
            page_number=page_number,
        )


def _parse_response(text: str) -> tuple[str, List[str]]:
    """Split raw model output into (caption, details).

    Mirrors ClaudeVisionProvider's parser: first non-bullet line becomes
    the caption; remaining lines (bullets or numbered) become details.
    A single unbroken paragraph degrades cleanly — the whole text is the
    caption and details is empty.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text.strip(), []

    caption = ""
    detail_start = 0
    for i, line in enumerate(lines):
        if not line.startswith(("-", "*", "•", "(")):
            caption = line.rstrip(".")
            caption = re.sub(r"^\(\d+\)\s*", "", caption)
            caption = re.sub(r"^\d+\.\s*", "", caption)
            if not caption.endswith("."):
                caption += "."
            detail_start = i + 1
            break

    if not caption:
        caption = lines[0]
        detail_start = 1

    details: List[str] = []
    for line in lines[detail_start:]:
        cleaned = re.sub(r"^[-*•]\s*", "", line)
        cleaned = re.sub(r"^\(\d+\)\s*", "", cleaned)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        if cleaned:
            details.append(cleaned)

    return caption, details
