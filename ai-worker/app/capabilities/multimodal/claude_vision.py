"""Claude Vision provider — real VLM behind the VisionDescriptionProvider seam.

Uses the Anthropic Python SDK to call Claude's vision endpoint. The model
receives the image as base64-encoded content and returns a structured
description that the multimodal capability fuses with OCR text for
grounded retrieval + generation.

The heuristic fallback (heuristic_vision.py) stays as the CI / offline /
test default. This provider is activated by setting:

    AIPIPELINE_WORKER_MULTIMODAL_VISION_PROVIDER=claude
    AIPIPELINE_WORKER_ANTHROPIC_API_KEY=sk-ant-...

Why Claude-only for now: the provider seam is model-agnostic, but this
phase deliberately ships a single VLM integration to keep the matrix
small. GPT-4o, Gemini, and local VLMs (LLaVA) are future work behind
the same interface.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import List, Optional

from app.capabilities.multimodal.vision_provider import (
    VisionDescriptionProvider,
    VisionDescriptionResult,
    VisionError,
)

log = logging.getLogger(__name__)

# Mime-to-media-type mapping for the Anthropic API.
_MIME_TO_MEDIA_TYPE = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}

_SYSTEM_PROMPT = (
    "You are a document-aware vision assistant. Analyze the image and produce:\n"
    "(1) a single-sentence factual caption of the main subject,\n"
    "(2) 3-5 bullet points of salient visual details,\n"
    "(3) verbatim transcription of any visible text.\n"
    "Respond in Korean if the image contains Korean text OR if the hint is "
    "in Korean; otherwise respond in English.\n"
    "Never speculate beyond what is visible."
)

_MAX_RETRIES = 2


class ClaudeVisionProvider(VisionDescriptionProvider):
    """Claude-backed vision description provider.

    Dependencies: `anthropic` SDK (pip install anthropic>=0.40.0).
    Requires AIPIPELINE_WORKER_ANTHROPIC_API_KEY at runtime.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        timeout_seconds: float = 30.0,
    ) -> None:
        import anthropic  # local import — registry catches ImportError

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
        )
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "claude-vision-v1"

    def describe_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
        hint: Optional[str] = None,
        page_number: int = 1,
    ) -> VisionDescriptionResult:
        import anthropic  # needed for exception types

        media_type = _MIME_TO_MEDIA_TYPE.get(
            (mime_type or "").lower(), "image/png"
        )
        b64_data = base64.standard_b64encode(image_bytes).decode("ascii")

        # Build the user message: image + optional hint text.
        content_blocks: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            }
        ]
        if hint:
            content_blocks.append({"type": "text", "text": f"Hint: {hint}"})

        started_at = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1 + _MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    temperature=0,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content_blocks}],
                )
                break
            except anthropic.APITimeoutError as ex:
                last_error = ex
                log.warning(
                    "ClaudeVisionProvider timeout attempt=%d/%d: %s",
                    attempt + 1, 1 + _MAX_RETRIES, ex,
                )
                if attempt >= _MAX_RETRIES:
                    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                    raise VisionError(
                        "VLM_TIMEOUT",
                        f"Claude Vision timed out after {self._timeout_seconds}s "
                        f"({1 + _MAX_RETRIES} attempts). Last error: {ex}",
                    ) from ex
            except anthropic.RateLimitError as ex:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                raise VisionError(
                    "VLM_RATE_LIMIT",
                    f"Claude Vision rate-limited: {ex}",
                ) from ex
            except anthropic.APIStatusError as ex:
                last_error = ex
                if ex.status_code >= 500:
                    log.warning(
                        "ClaudeVisionProvider 5xx attempt=%d/%d: %s",
                        attempt + 1, 1 + _MAX_RETRIES, ex,
                    )
                    if attempt >= _MAX_RETRIES:
                        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                        raise VisionError(
                            "VLM_API_FAILED",
                            f"Claude Vision API failed after "
                            f"{1 + _MAX_RETRIES} attempts. "
                            f"Last error: {ex.status_code} {ex}",
                        ) from ex
                else:
                    # 4xx is not retryable
                    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                    raise VisionError(
                        "VLM_API_FAILED",
                        f"Claude Vision API error: {ex.status_code} {ex}",
                    ) from ex
            except Exception as ex:
                # httpx transient errors (ConnectionError, etc.) — retry
                last_error = ex
                log.warning(
                    "ClaudeVisionProvider transient error attempt=%d/%d: %s: %s",
                    attempt + 1, 1 + _MAX_RETRIES,
                    type(ex).__name__, ex,
                )
                if attempt >= _MAX_RETRIES:
                    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                    raise VisionError(
                        "VLM_API_FAILED",
                        f"Claude Vision failed after {1 + _MAX_RETRIES} "
                        f"attempts: {type(ex).__name__}: {ex}",
                    ) from ex

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        # Extract text from the response.
        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        if not raw_text.strip():
            raise VisionError(
                "VLM_BAD_RESPONSE",
                "Claude Vision returned an empty response.",
            )

        caption, details = _parse_response(raw_text)

        log.info(
            "ClaudeVisionProvider described image: page=%d model=%s "
            "latency_ms=%.2f caption_len=%d detail_count=%d",
            page_number, self._model, elapsed_ms,
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
    """Extract a caption (first sentence) and bullet details from the model output.

    The model is prompted to produce (1) a caption sentence, (2) bullet
    points, (3) visible text transcription. We split on the first
    sentence boundary for the caption and collect bullet lines as details.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return text.strip(), []

    # Caption: first non-bullet line (or the first line if all are bullets).
    caption = ""
    detail_start = 0
    for i, line in enumerate(lines):
        if not line.startswith(("-", "*", "•", "(")):
            caption = line.rstrip(".")
            # Remove leading numbering like "(1)" or "1."
            caption = re.sub(r"^\(\d+\)\s*", "", caption)
            caption = re.sub(r"^\d+\.\s*", "", caption)
            if not caption.endswith("."):
                caption += "."
            detail_start = i + 1
            break

    if not caption:
        caption = lines[0]
        detail_start = 1

    # Details: remaining lines, cleaned of bullet markers.
    details: List[str] = []
    for line in lines[detail_start:]:
        cleaned = re.sub(r"^[-*•]\s*", "", line)
        cleaned = re.sub(r"^\(\d+\)\s*", "", cleaned)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        if cleaned:
            details.append(cleaned)

    return caption, details
