"""Deterministic Pillow-based vision fallback.

This is the default VisionDescriptionProvider implementation for v1.
It is explicitly NOT a replacement for a real VLM — it's a
"the multimodal pipeline is open" provider that lets us exercise the
full INPUT_FILE → OCR + vision → fused context → retrieval → generation
flow without downloading a multi-gigabyte model or calling a paid API.

What it actually does:
  1. Decodes the image with Pillow.
  2. Computes a small set of statistical features over the image:
     dimensions, orientation, brightness, contrast, dominant color
     channel, and mode (grayscale / RGB / etc.).
  3. Converts those features into a short deterministic caption
     ("A portrait light-toned image (800x1000 pixels) dominated by
     red tones with moderate contrast.") and a bullet list of the
     raw features.

The output is useful in two ways:
  - It gives the fusion layer a non-empty "visual description" signal
    to fold into the retrieval context, so the retrieval + generation
    stages still have something to work with when OCR extracts nothing.
  - It is byte-for-byte deterministic for the same input, so the
    multimodal capability's behavior is testable without any network
    or model stochasticity.

Swap in a real VLM later by implementing VisionDescriptionProvider
and changing a single line in `registry._build_vision_provider`.
"""

from __future__ import annotations

import io
import logging
import time
from typing import List, Optional

from app.capabilities.multimodal.vision_provider import (
    VisionDescriptionProvider,
    VisionDescriptionResult,
    VisionError,
)

log = logging.getLogger(__name__)


# Feature classification thresholds. Kept as module-level constants so
# the tests (and ops) can audit them without digging into method bodies.
_BRIGHTNESS_DARK_MAX = 64.0
_BRIGHTNESS_MEDIUM_MAX = 160.0
_CONTRAST_LOW_MAX = 28.0
_CONTRAST_MODERATE_MAX = 64.0
_ASPECT_PORTRAIT_MAX = 0.9
_ASPECT_LANDSCAPE_MIN = 1.1
_DOMINANT_CHANNEL_DELTA = 6.0


class HeuristicVisionProvider(VisionDescriptionProvider):
    """Pillow-backed deterministic vision fallback.

    Dependencies: only Pillow, which is already pulled in by
    sentence-transformers / the OCR stack. No torch, no transformers,
    no HuggingFace, no network.
    """

    @property
    def name(self) -> str:
        return "heuristic-vision-v1"

    def describe_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
        hint: Optional[str] = None,
        page_number: int = 1,
    ) -> VisionDescriptionResult:
        try:
            from PIL import Image, ImageStat  # type: ignore
        except ImportError as ex:  # pragma: no cover — Pillow ships with sentence-transformers
            raise VisionError(
                "PIL_IMPORT_FAILED",
                "Pillow is not installed. `pip install Pillow`.",
            ) from ex

        started_at = time.perf_counter()

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as ex:
            raise VisionError(
                "IMAGE_DECODE_FAILED",
                f"HeuristicVisionProvider could not decode image bytes: "
                f"{type(ex).__name__}: {ex}",
            ) from ex

        original_mode = image.mode
        width, height = image.size
        if width <= 0 or height <= 0:
            raise VisionError(
                "IMAGE_DIMENSIONS_INVALID",
                f"Decoded image has invalid dimensions {width}x{height}.",
            )

        # Convert once for consistent statistics regardless of source mode.
        rgb = image.convert("RGB") if original_mode not in ("L", "RGB") else image
        grayscale = rgb.convert("L") if rgb.mode != "L" else rgb

        gray_stats = ImageStat.Stat(grayscale)
        brightness = float(gray_stats.mean[0])
        contrast = float(gray_stats.stddev[0])

        # Channel means give a rough "dominant hue" without calling a
        # full color analysis. Enough to distinguish "mostly red" from
        # "mostly blue" from "neutral".
        if rgb.mode == "RGB":
            rgb_stats = ImageStat.Stat(rgb)
            r_mean, g_mean, b_mean = (
                float(rgb_stats.mean[0]),
                float(rgb_stats.mean[1]),
                float(rgb_stats.mean[2]),
            )
        else:
            r_mean = g_mean = b_mean = brightness
        dominant_channel = _classify_dominant_channel(r_mean, g_mean, b_mean)

        aspect = width / max(1, height)
        orientation = _classify_orientation(aspect)
        lightness = _classify_brightness(brightness)
        contrast_word = _classify_contrast(contrast)

        caption = (
            f"A {orientation} {lightness}-toned image "
            f"({width}x{height} pixels) dominated by {dominant_channel} tones "
            f"with {contrast_word} contrast."
        )

        details: List[str] = [
            f"dimensions: {width}x{height} pixels ({orientation})",
            f"mean brightness: {brightness:.1f}/255 ({lightness})",
            f"contrast (stddev): {contrast:.1f} ({contrast_word})",
            f"dominant channel: {dominant_channel}",
            f"source mode: {original_mode}",
        ]
        if mime_type:
            details.append(f"mime type: {mime_type}")
        if hint:
            details.append(f"hint: {hint.strip()}")

        warnings: List[str] = []
        if brightness >= 250.0 or brightness <= 5.0:
            warnings.append(
                f"brightness {brightness:.1f}/255 is near the extreme — "
                "image may be blank, saturated, or OCR-hostile"
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        log.info(
            "HeuristicVisionProvider described image: "
            "page=%d size=%dx%d brightness=%.1f contrast=%.1f dominant=%s "
            "latency_ms=%.2f",
            page_number, width, height, brightness, contrast,
            dominant_channel, elapsed_ms,
        )

        return VisionDescriptionResult(
            provider_name=self.name,
            caption=caption,
            details=details,
            warnings=warnings,
            latency_ms=round(elapsed_ms, 3),
            page_number=page_number,
        )


# --------------------------------------------------------------------
# pure helpers — no Pillow import, deterministic, testable in isolation
# --------------------------------------------------------------------


def _classify_brightness(brightness: float) -> str:
    if brightness < _BRIGHTNESS_DARK_MAX:
        return "dark"
    if brightness < _BRIGHTNESS_MEDIUM_MAX:
        return "medium"
    return "light"


def _classify_contrast(contrast: float) -> str:
    if contrast < _CONTRAST_LOW_MAX:
        return "low"
    if contrast < _CONTRAST_MODERATE_MAX:
        return "moderate"
    return "high"


def _classify_orientation(aspect: float) -> str:
    if aspect < _ASPECT_PORTRAIT_MAX:
        return "portrait"
    if aspect > _ASPECT_LANDSCAPE_MIN:
        return "landscape"
    return "square"


def _classify_dominant_channel(r: float, g: float, b: float) -> str:
    """Return the channel name with the highest mean, or "neutral" if
    no channel is meaningfully above the others.

    The delta check prevents a balanced gray image from getting an
    arbitrary label flip due to sub-pixel noise.
    """
    ordered = sorted([("red", r), ("green", g), ("blue", b)], key=lambda kv: kv[1], reverse=True)
    top_name, top_value = ordered[0]
    _, second_value = ordered[1]
    if (top_value - second_value) < _DOMINANT_CHANNEL_DELTA:
        return "neutral"
    return top_name
