"""Generate synthetic OCR fixture images for the eval sample dataset.

Writes PNGs into `eval/datasets/samples/` whose contents exactly match
the `ground_truth` strings in `eval/datasets/ocr_sample.jsonl`. The
goal is a zero-cost local smoke test of the OCR eval pipeline without
committing binary image blobs to the repo.

Usage (from ai-worker/):

    python -m scripts.make_ocr_sample_fixtures

Overwrites existing files. Uses Pillow's default bitmap font when a
platform TTF is not discoverable, so the images look unglamorous but
are easy for Tesseract to read.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("scripts.make_ocr_sample_fixtures")


# (filename, lines, font_size_hint) — the eval dataset ground_truth
# strings are the `\n`-joined versions of `lines`.
_FIXTURES: List[Tuple[str, List[str], int]] = [
    ("hello_world.png", ["HELLO WORLD"], 48),
    ("invoice_snippet.png", ["Invoice #1024", "Total: $129.95"], 36),
    (
        "multi_paragraph.png",
        [
            "The quick brown fox jumps over the lazy dog.",
            "Sphinx of black quartz, judge my vow.",
        ],
        28,
    ),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error(
            "Pillow is required to generate OCR sample fixtures. "
            "Run `pip install -r requirements.txt`."
        )
        return 2

    out_dir = (
        Path(__file__).resolve().parent.parent
        / "eval"
        / "datasets"
        / "samples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing OCR sample PNGs to %s", out_dir)

    for filename, lines, font_size in _FIXTURES:
        _render(
            Image,
            ImageDraw,
            ImageFont,
            out_dir / filename,
            lines,
            font_size,
        )

    log.info("Done. Run `python -m eval.run_eval ocr --dataset eval/datasets/ocr_sample.jsonl`")
    return 0


def _render(Image, ImageDraw, ImageFont, path: Path, lines, font_size: int) -> None:
    font = _load_font(ImageFont, font_size)

    # Measure each line and pad with generous margins so Tesseract's
    # segmenter has breathing room.
    padding_x, padding_y, line_gap = 40, 32, 14
    dummy = Image.new("L", (4, 4), color=255)
    draw = ImageDraw.Draw(dummy)

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [box[2] - box[0] for box in line_boxes]
    line_heights = [box[3] - box[1] for box in line_boxes]
    width = 2 * padding_x + max(line_widths) if line_widths else 2 * padding_x
    height = 2 * padding_y + sum(line_heights) + line_gap * (len(lines) - 1)

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    y = padding_y
    for line, h in zip(lines, line_heights):
        draw.text((padding_x, y), line, fill=0, font=font)
        y += h + line_gap

    image.save(path, format="PNG")
    log.info("wrote %s (%dx%d)", path.name, width, height)


def _load_font(ImageFont, size: int):
    """Try a few common system TrueType fonts; fall back to Pillow's
    built-in bitmap font if none are discoverable.

    The bitmap fallback is tiny (~10px) and Tesseract still reads it
    fine for the short sample strings we generate."""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Linux (DejaVu ships in most distros)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    log.warning(
        "No system TTF found; falling back to Pillow's default bitmap font. "
        "OCR quality on the samples will still be fine."
    )
    return ImageFont.load_default()


if __name__ == "__main__":
    sys.exit(main())
