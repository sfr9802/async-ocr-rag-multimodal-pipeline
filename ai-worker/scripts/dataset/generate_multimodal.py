"""Generate enterprise multimodal eval rows from rendered OCR pages.

Pairs each PNG produced by ``synthesize_ocr_pages`` with a Korean
question that targets one specific fact in that page. The question is
drafted by Claude — the gold answer is ``null`` because the multimodal
harness scores keyword coverage, not exact match, on these rows.

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.generate_multimodal \\
        --corpus fixtures/corpus_kr/index.jsonl \\
        --pages-dir eval/datasets/samples/ocr_enterprise \\
        --out  eval/datasets/multimodal_enterprise_kr.jsonl \\
        --count 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.dataset._common import (
    ClaudeResponseError,
    GenerationLog,
    RateLimiter,
    ResumableJsonlWriter,
    claude_json_call,
    configure_logging,
    load_anthropic_client,
    log_call,
    read_jsonl,
    stable_seed,
)

log = logging.getLogger("scripts.dataset.generate_multimodal")


_SYSTEM_PROMPT = (
    "You author multimodal evaluation questions for Korean enterprise "
    "document images. Respond ONLY with a JSON object. Do NOT wrap it "
    "in code fences."
)


_SCHEMA_HINT = (
    '{"question": string, "expected_keywords": [string, ...], "notes": string}'
)


def _build_user_prompt(doc: Dict[str, Any], *, seed: int) -> str:
    sections = doc.get("sections") or {}
    body_lines: List[str] = []
    for heading, payload in sections.items():
        if isinstance(payload, dict) and payload.get("text"):
            body_lines.append(f"## {heading}\n{payload.get('text')}")
    body = "\n\n".join(body_lines)[:3000]
    return (
        f"Document title: {doc.get('title')}\n"
        f"Document id: {doc.get('doc_id')}\n"
        f"Seed (do not echo): {seed}\n\n"
        f"Document body:\n{body}\n\n"
        "Produce ONE Korean question that targets a single concrete "
        "fact (date, count, role, threshold) stated in the document. "
        "expected_keywords: 1-3 substrings the answer text should "
        "contain. notes: short rationale (max 12 words).\n"
        f"Return JSON of this shape:\n{_SCHEMA_HINT}"
    )


def generate(
    *,
    corpus_path: Path,
    pages_dir: Path,
    out_path: Path,
    count: int,
    model: str,
    rate_per_sec: float,
    dry_run: bool,
) -> int:
    docs = read_jsonl(corpus_path)
    if not docs:
        log.error("corpus empty: %s", corpus_path)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ResumableJsonlWriter(out_path, key_fn=lambda r: str(r.get("image", "")))
    gen_log = GenerationLog(out_path.parent / f"{out_path.stem}_generation_log.jsonl")
    limiter = RateLimiter(rate_per_sec)

    client = None
    if not dry_run:
        client = load_anthropic_client()

    new_count = 0
    for doc in docs[:count]:
        doc_id = str(doc.get("doc_id"))
        png_path = pages_dir / f"{doc_id}.png"
        if not png_path.exists():
            log.warning("page image not found, skipping: %s", png_path)
            continue
        try:
            rel_image = str(png_path.resolve().relative_to(out_path.parent.resolve()))
            rel_image = rel_image.replace("\\", "/")
        except (OSError, ValueError):
            rel_image = f"samples/ocr_enterprise/{doc_id}.png"

        if writer.has(rel_image):
            continue
        if dry_run:
            log.info("[dry-run] would draft a question for %s", doc_id)
            continue

        seed = stable_seed(doc_id, "multimodal")
        limiter.wait()
        try:
            with log_call(
                gen_log,
                script="generate_multimodal",
                provider="claude",
                model=model,
                seed=seed,
                note=doc_id,
            ) as slot:
                parsed = claude_json_call(
                    client, model=model,
                    system=_SYSTEM_PROMPT,
                    user=_build_user_prompt(doc, seed=seed),
                    max_tokens=400,
                    temperature=0.6,
                )
                usage = parsed.pop("_usage", {})
                slot["prompt_tokens"] = usage.get("input_tokens")
                slot["completion_tokens"] = usage.get("output_tokens")
        except (ClaudeResponseError, Exception) as ex:  # noqa: BLE001
            log.warning("multimodal question generation failed for %s: %s", doc_id, ex)
            continue

        question = str(parsed.get("question", "")).strip()
        keywords_raw = parsed.get("expected_keywords") or []
        if not isinstance(keywords_raw, list):
            keywords_raw = []
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()][:6]
        if not question:
            continue

        writer.append({
            "image": rel_image,
            "question": question,
            "expected_answer": None,
            "expected_keywords": keywords,
            "expected_labels": [],
            "requires_ocr": True,
            "language": "kor",
            "domain": "enterprise",
            "category": doc.get("category"),
            "notes": str(parsed.get("notes", "")).strip()[:120] or f"drafted for {doc_id}",
        })
        new_count += 1

    log.info("Wrote %d new multimodal rows to %s", new_count, out_path)
    return new_count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--rate-per-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        generate(
            corpus_path=args.corpus,
            pages_dir=args.pages_dir,
            out_path=args.out,
            count=args.count,
            model=args.model,
            rate_per_sec=args.rate_per_sec,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("multimodal generation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
