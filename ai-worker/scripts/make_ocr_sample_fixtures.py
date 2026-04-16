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

# Korean fixtures — require a Korean-capable font (NanumGothic, Malgun
# Gothic, or D2Coding). If no Korean font is found the script emits a
# warning and skips these without failing the harness.
_KR_FIXTURES: List[Tuple[str, List[str], int]] = [
    ("kr_hello.png", ["안녕하세요 세계"], 48),
    ("kr_notice.png", ["공지사항", "시스템 점검 안내"], 36),
    ("kr_policy.png", ["비밀번호는 최소 12자 이상이어야 합니다"], 28),
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

    # Korean fixtures — skip if no Korean font is available.
    kr_font = _load_korean_font(ImageFont, 36)
    if kr_font is not None:
        log.info("Korean font found — generating Korean OCR fixtures")
        for filename, lines, font_size in _KR_FIXTURES:
            kr_font_sized = _load_korean_font(ImageFont, font_size)
            _render(
                Image,
                ImageDraw,
                ImageFont,
                out_dir / filename,
                lines,
                font_size,
                font_override=kr_font_sized,
            )
    else:
        log.warning(
            "No Korean font found on this system — skipping Korean OCR "
            "fixtures. Korean OCR eval will not run until a Korean font "
            "(NanumGothic, Malgun Gothic, or D2Coding) is installed."
        )

    log.info("Done. Run `python -m eval.run_eval ocr --dataset eval/datasets/ocr_sample.jsonl`")
    return 0


def _render(Image, ImageDraw, ImageFont, path: Path, lines, font_size: int, *, font_override=None) -> None:
    font = font_override if font_override is not None else _load_font(ImageFont, font_size)

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


def _load_korean_font(ImageFont, size: int):
    """Try Korean-capable TrueType fonts; return None if none found.

    Searches common font paths on Windows, macOS, and Linux. Returns
    None (instead of raising) so the caller can skip Korean fixtures
    gracefully.
    """
    candidates = [
        # Windows
        r"C:\Windows\Fonts\malgun.ttf",      # Malgun Gothic
        r"C:\Windows\Fonts\malgunbd.ttf",     # Malgun Gothic Bold
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\D2Coding.ttf",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return None


if __name__ == "__main__":
    sys.exit(main())
