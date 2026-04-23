"""Generate synthetic multimodal eval fixture images.

Creates PNGs in `eval/datasets/samples/multimodal/` that cover three
eval row types:
  1. OCR-only: images with clear text that can be answered via OCR alone
  2. Visual-only: images with shapes/colors that require visual description
  3. OCR + Visual: images combining text and visual elements

Includes both English and Korean samples.

Usage (from ai-worker/):
    python -m scripts.make_multimodal_sample_fixtures
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("scripts.make_multimodal_sample_fixtures")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error(
            "Pillow is required. Run `pip install -r requirements.txt`."
        )
        return 2

    out_dir = (
        Path(__file__).resolve().parent.parent
        / "eval" / "datasets" / "samples" / "multimodal"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing multimodal sample PNGs to %s", out_dir)

    font = _load_font(ImageFont, 32)
    kr_font = _load_korean_font(ImageFont, 32)

    # 1. OCR-only: invoice text (EN)
    _render_text_image(
        Image, ImageDraw, out_dir / "invoice_text.png",
        lines=["Invoice #2024-0042", "Total: $257.50", "Due: 2026-05-01"],
        font=font, width=500, height=200,
    )
    log.info("wrote invoice_text.png (OCR-only, EN)")

    # 2. OCR-only: Korean notice
    if kr_font:
        _render_text_image(
            Image, ImageDraw, out_dir / "kr_notice_text.png",
            lines=["공지사항", "서버 점검 일시: 2026년 5월 1일", "오전 2시 ~ 오전 6시"],
            font=kr_font, width=520, height=200,
        )
        log.info("wrote kr_notice_text.png (OCR-only, KR)")
    else:
        log.warning("No Korean font — skipping kr_notice_text.png")

    # 3. Visual-only: colored shapes (no text)
    _render_shapes_image(
        Image, ImageDraw, out_dir / "shapes_visual.png",
        width=400, height=300,
    )
    log.info("wrote shapes_visual.png (visual-only)")

    # 4. Visual-only: chart-like bars (no text)
    _render_bar_chart(
        Image, ImageDraw, out_dir / "bar_chart.png",
        width=400, height=300,
    )
    log.info("wrote bar_chart.png (visual-only)")

    # 5. OCR + Visual: labeled diagram (EN)
    _render_labeled_diagram(
        Image, ImageDraw, out_dir / "labeled_diagram.png",
        font=font, width=500, height=350,
    )
    log.info("wrote labeled_diagram.png (OCR+visual, EN)")

    # 6. OCR + Visual: Korean labeled diagram
    if kr_font:
        _render_kr_labeled_diagram(
            Image, ImageDraw, out_dir / "kr_labeled_diagram.png",
            font=kr_font, width=500, height=350,
        )
        log.info("wrote kr_labeled_diagram.png (OCR+visual, KR)")
    else:
        log.warning("No Korean font — skipping kr_labeled_diagram.png")

    # 7. OCR-only: Korean security policy
    if kr_font:
        _render_text_image(
            Image, ImageDraw, out_dir / "kr_security_text.png",
            lines=["보안 정책 안내", "비밀번호 최소 12자 이상 필수"],
            font=kr_font, width=500, height=160,
        )
        log.info("wrote kr_security_text.png (OCR-only, KR)")
    else:
        log.warning("No Korean font — skipping kr_security_text.png")

    log.info("Done. Run `python -m eval.run_eval multimodal --dataset eval/datasets/multimodal_sample.jsonl`")
    return 0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_text_image(Image, ImageDraw, path: Path, *, lines, font, width, height):
    """White background with black text lines."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((30, y), line, fill=(0, 0, 0), font=font)
        y += 50
    img.save(path, format="PNG")


def _render_shapes_image(Image, ImageDraw, path: Path, *, width, height):
    """White background with colored geometric shapes — no text."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Red circle
    draw.ellipse([30, 30, 150, 150], fill=(220, 50, 50), outline=(180, 30, 30))
    # Blue rectangle
    draw.rectangle([180, 50, 320, 140], fill=(50, 50, 220), outline=(30, 30, 180))
    # Green triangle
    draw.polygon([(250, 200), (350, 280), (150, 280)], fill=(50, 180, 50), outline=(30, 140, 30))
    img.save(path, format="PNG")


def _render_bar_chart(Image, ImageDraw, path: Path, *, width, height):
    """Simple bar chart visual — colored bars of different heights."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Bars
    bars = [
        (50, 200, (65, 105, 225)),   # blue, tallest
        (120, 140, (220, 50, 50)),   # red, medium
        (190, 100, (50, 180, 50)),   # green, short
        (260, 170, (255, 165, 0)),   # orange, tall
    ]
    for x, bar_height, color in bars:
        draw.rectangle([x, height - 30 - bar_height, x + 50, height - 30], fill=color)
    # Baseline
    draw.line([(30, height - 30), (width - 30, height - 30)], fill=(0, 0, 0), width=2)
    img.save(path, format="PNG")


def _render_labeled_diagram(Image, ImageDraw, path: Path, *, font, width, height):
    """Shapes with text labels — requires both OCR and visual understanding."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Title
    draw.text((30, 10), "System Architecture", fill=(0, 0, 0), font=font)
    # Box A
    draw.rectangle([30, 70, 200, 140], fill=(200, 220, 255), outline=(0, 0, 150))
    draw.text((60, 90), "Frontend", fill=(0, 0, 100), font=font)
    # Box B
    draw.rectangle([280, 70, 450, 140], fill=(200, 255, 200), outline=(0, 150, 0))
    draw.text((310, 90), "Backend", fill=(0, 100, 0), font=font)
    # Arrow
    draw.line([(200, 105), (280, 105)], fill=(100, 100, 100), width=3)
    draw.polygon([(270, 95), (280, 105), (270, 115)], fill=(100, 100, 100))
    # Box C
    draw.rectangle([150, 200, 330, 270], fill=(255, 220, 200), outline=(150, 0, 0))
    draw.text((170, 220), "Database", fill=(100, 0, 0), font=font)
    # Arrow down
    draw.line([(365, 140), (240, 200)], fill=(100, 100, 100), width=2)
    img.save(path, format="PNG")


def _render_kr_labeled_diagram(Image, ImageDraw, path: Path, *, font, width, height):
    """Korean labeled diagram — requires OCR + visual."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 10), "시스템 구성도", fill=(0, 0, 0), font=font)
    # Box A
    draw.rectangle([30, 70, 200, 140], fill=(200, 220, 255), outline=(0, 0, 150))
    draw.text((50, 90), "웹 서버", fill=(0, 0, 100), font=font)
    # Box B
    draw.rectangle([280, 70, 470, 140], fill=(200, 255, 200), outline=(0, 150, 0))
    draw.text((290, 90), "API 서버", fill=(0, 100, 0), font=font)
    # Arrow
    draw.line([(200, 105), (280, 105)], fill=(100, 100, 100), width=3)
    draw.polygon([(270, 95), (280, 105), (270, 115)], fill=(100, 100, 100))
    # Box C
    draw.rectangle([140, 200, 350, 270], fill=(255, 220, 200), outline=(150, 0, 0))
    draw.text((150, 220), "데이터베이스", fill=(100, 0, 0), font=font)
    draw.line([(375, 140), (245, 200)], fill=(100, 100, 100), width=2)
    img.save(path, format="PNG")


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------


def _load_font(ImageFont, size: int):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_korean_font(ImageFont, size: int):
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    log.warning("No Korean font found — Korean fixtures will be skipped")
    return None


if __name__ == "__main__":
    sys.exit(main())
