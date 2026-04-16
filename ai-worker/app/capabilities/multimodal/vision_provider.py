"""VisionDescriptionProvider contract + result dataclasses + typed error.

Kept deliberately narrow: the multimodal capability doesn't care what
model is behind the seam, only that the model knows how to turn "image
bytes" into a short structured description. Swapping in a real VLM
(BLIP-2, LLaVA, GPT-4V, Gemini Vision, Claude, ...) is a single-file
change as long as the new provider honours this interface.

For v1 the capability ships with a deterministic HeuristicVisionProvider
implementation (see `heuristic_vision.py`) so the multimodal pipeline
can be exercised end-to-end without pulling a multi-gigabyte VLM into
memory. The heuristic provider is NOT a quality bar — it's a "the
pipeline is open" bar.

One method exists intentionally:
  - describe_image: single-frame raster input (PNG, JPEG). PDF pages
    are rasterized to PNG by the capability layer and handed here one
    page at a time, so providers never need to implement PDF iteration.

The capability layer does mime dispatch + PDF rasterization; the
provider only does "bytes in → caption out".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class VisionDescriptionResult:
    """Result of running a vision description model on a single image.

    Fields:
      - provider_name: stable identifier for the model/version used.
                       Embedded into VISION_RESULT.provider so ops can
                       correlate a job's VISION_RESULT with a specific
                       provider install.
      - caption:       one-line natural-language caption of the image.
                       Must not be empty — a provider that can't produce
                       anything useful should raise VisionError instead.
      - details:       optional extra bullet-point details (dominant
                       colors, visible objects, text hints, etc.). May
                       be empty for minimal providers.
      - warnings:      non-fatal diagnostics attached to this result.
      - latency_ms:    provider-side wall clock latency in milliseconds.
                       Used purely for telemetry / MULTIMODAL_TRACE.
      - page_number:   1-indexed page number the caption was generated
                       from. For single-image inputs this is always 1.
                       Multi-page PDF inputs produce multiple results
                       with increasing page numbers.
    """

    provider_name: str
    caption: str
    details: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    page_number: int = 1


class VisionError(Exception):
    """Structured vision-description failure.

    Providers raise this (or the capability layer raises it on behalf of
    the provider) so the capability can produce a clean typed
    CapabilityError with a stable `code` string. The error is NOT a
    CapabilityError itself because the provider layer must stay
    independent of the capability base module.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class VisionDescriptionProvider(ABC):
    """Abstract vision-description model.

    Implementations must be deterministic-ish enough that the same image
    bytes produce *similar* outputs across calls, but they are not
    required to be bit-for-bit reproducible — a real VLM sampling at
    temperature > 0 is fine. The capability layer does not depend on
    byte-level stability.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier embedded into VISION_RESULT.provider."""

    @abstractmethod
    def describe_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
        hint: Optional[str] = None,
        page_number: int = 1,
    ) -> VisionDescriptionResult:
        """Produce a short description for a single image.

        Parameters:
          image_bytes: raw image bytes (PNG / JPEG / ...).
          mime_type:   advertised content-type, if known. Providers may
                       use this to pick the right decoder path, but
                       must also be tolerant of None (the capability
                       layer already validated magic bytes).
          hint:        optional short free-text hint from the user (the
                       question they asked). A real VLM provider can
                       condition on the hint; the heuristic fallback
                       records it in `details` for inspectability.
          page_number: 1-indexed page number (useful only when the
                       capability rasterized a specific PDF page).

        Raises:
          VisionError: on unrecoverable failures. The capability layer
                       catches this and falls back to "vision
                       unavailable" in the fused context rather than
                       failing the whole job.
        """
