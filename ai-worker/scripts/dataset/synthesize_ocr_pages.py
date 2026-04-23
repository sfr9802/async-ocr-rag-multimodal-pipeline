"""Render enterprise-style OCR eval pages from the synthetic KR corpus.

Produces PNG images of Korean document pages whose ``ground_truth`` is
exactly the rendered text. Each image carries a single page worth of
body text pulled from the corpus so the OCR harness sees realistic
prose rather than fabricated strings.

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.synthesize_ocr_pages \\
        --corpus fixtures/corpus_kr/index.jsonl \\
        --out-images eval/datasets/samples/ocr_enterprise \\
        --out-jsonl  eval/datasets/ocr_enterprise_kr.jsonl \\
        --count 50

Design notes
------------
* **Deterministic.** Picks the first N docs from the corpus and renders
  one page each. No model calls — pure Pillow.
* **Korean font required.** Same font resolution as the Phase-2 sample
  fixture generator; without a Korean font the script aborts rather
  than producing boxes that the OCR step would fail to recognize.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.dataset._common import configure_logging, read_jsonl, write_jsonl

log = logging.getLogger("scripts.dataset.synthesize_ocr_pages")


def _extract_body_for_doc(doc: Dict, *, max_chars: int) -> str:
    """Concatenate the document's sections into a single renderable blob."""
    sections = doc.get("sections") or {}
    lines: List[str] = []
    title = doc.get("title")
    if title:
        lines.append(str(title))
        lines.append("")
    order = doc.get("section_order") or list(sections.keys())
    for heading in order:
        payload = sections.get(heading)
        if isinstance(payload, dict):
            text = payload.get("text")
            if text:
                lines.append(f"{heading}")
                lines.append(str(text))
                lines.append("")
        if sum(len(l) for l in lines) >= max_chars:
            break
    blob = "\n".join(lines).strip()
    return blob[:max_chars]


def _render_page(
    Image, ImageDraw, ImageFont,
    path: Path, *, text: str, font,
    width: int = 1080, height: int = 1400, margin: int = 80,
) -> None:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = margin
    # Wrap lines by width using pixel measurements so long sentences
    # actually wrap inside the page frame. Simple greedy word-wrap by
    # whitespace since Korean sentences are still separated by spaces
    # around content words in this corpus.
    line_height = 36
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            y += line_height // 2
            continue
        words = raw_line.split(" ")
        line = ""
        for word in words:
            candidate = (line + " " + word).strip()
            # Measure by draw.textlength (Pillow >=9.2) or fallback.
            try:
                w = draw.textlength(candidate, font=font)
            except Exception:
                w = len(candidate) * 12
            if w <= width - 2 * margin:
                line = candidate
            else:
                if line:
                    draw.text((margin, y), line, fill=(20, 20, 20), font=font)
                    y += line_height
                line = word
            if y > height - margin:
                break
        if line and y <= height - margin:
            draw.text((margin, y), line, fill=(20, 20, 20), font=font)
            y += line_height
        if y > height - margin:
            break
    img.save(path, format="PNG")


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
    return None


def synthesize(
    *,
    corpus_path: Path,
    out_images_dir: Path,
    out_jsonl: Path,
    count: int,
    max_chars: int,
) -> int:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error("Pillow is required. Run `pip install Pillow`.")
        return 0

    font = _load_korean_font(ImageFont, 28)
    if font is None:
        log.error(
            "No Korean font found. Install malgun.ttf (Windows), "
            "NanumGothic (Linux) or AppleSDGothicNeo (macOS) and retry."
        )
        return 0

    docs = read_jsonl(corpus_path)
    if not docs:
        log.error("corpus is empty: %s", corpus_path)
        return 0

    out_images_dir.mkdir(parents=True, exist_ok=True)
    picks = docs[:count]
    jsonl_rows: List[Dict] = []

    for doc in picks:
        doc_id = str(doc.get("doc_id"))
        text = _extract_body_for_doc(doc, max_chars=max_chars)
        if not text:
            continue
        png_path = out_images_dir / f"{doc_id}.png"
        _render_page(Image, ImageDraw, ImageFont, png_path, text=text, font=font)
        # Relative path for the jsonl loader — same convention as
        # ocr_sample.jsonl (paths relative to the dataset file).
        try:
            rel = str(png_path.resolve().relative_to(out_jsonl.parent.resolve()))
        except (OSError, ValueError):
            rel = f"samples/ocr_enterprise/{doc_id}.png"
        jsonl_rows.append({
            "file": rel.replace("\\", "/"),
            "ground_truth": text,
            "language": "kor",
            "domain": "enterprise",
            "category": doc.get("category"),
            "notes": f"rendered from {doc_id}",
        })

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, jsonl_rows, header=(
        f"Synthesized from {corpus_path.name} by synthesize_ocr_pages\n"
        f"images under {out_images_dir.name}"
    ))
    log.info("Wrote %d images + %s", len(jsonl_rows), out_jsonl)
    return len(jsonl_rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-images", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        synthesize(
            corpus_path=args.corpus,
            out_images_dir=args.out_images,
            out_jsonl=args.out_jsonl,
            count=args.count,
            max_chars=args.max_chars,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("synthesis failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
