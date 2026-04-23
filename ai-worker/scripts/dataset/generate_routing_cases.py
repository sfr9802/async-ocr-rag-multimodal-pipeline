"""Generate routing eval cases for the AUTO capability.

Emits a JSONL where each row targets one of the five router actions:
``rag`` / ``ocr`` / ``multimodal`` / ``direct_answer`` / ``clarify``.
Quota defaults mirror the Phase 9 distribution (30 / 15 / 15 / 10 / 10).

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.generate_routing_cases \\
        --corpus fixtures/corpus_kr/index.jsonl \\
        --out    eval/datasets/routing_enterprise_kr.jsonl \\
        --total 80

For ``rag`` rows the script pulls realistic query stems from the corpus
via Claude so the query actually references something the retriever can
find. The other four buckets are drawn from hardcoded templates since
they are deliberately generic (greetings, file-only, empty queries).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
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
    write_jsonl,
)

log = logging.getLogger("scripts.dataset.generate_routing_cases")


_ACTIONS = ("rag", "ocr", "multimodal", "direct_answer", "clarify")
_DEFAULT_QUOTA = {"rag": 30, "ocr": 15, "multimodal": 15, "direct_answer": 10, "clarify": 10}


_DIRECT_ANSWER_TEMPLATES = [
    "안녕하세요",
    "반갑습니다",
    "고맙습니다",
    "감사합니다",
    "안녕히 계세요",
    "좋은 하루 되세요",
    "잘 부탁드립니다",
    "수고하셨습니다",
]

_CLARIFY_TEMPLATES = [
    "",
    "?",
    "ok",
    "네",
    "아",
    "흠",
    "음",
    "...",
    "hi",
    "여보세요",
]

_OCR_REQUEST_TEMPLATES = [
    "",
    "",
    "텍스트만 추출해줘",
    "OCR 해주세요",
    "이 파일의 내용을 글자로 옮겨줘",
]


_FILE_MIMES = ["image/png", "image/jpeg", "application/pdf"]


_SYSTEM_PROMPT_RAG = (
    "You extract realistic Korean document search queries. Respond ONLY "
    "with a JSON object. Do NOT wrap it in code fences."
)

_SCHEMA_HINT_RAG = '{"query": string, "notes": string}'


def _build_user_prompt_rag(doc: Dict[str, Any], *, seed: int) -> str:
    sections = doc.get("sections") or {}
    title = doc.get("title") or doc.get("doc_id")
    first_text = ""
    for payload in sections.values():
        if isinstance(payload, dict) and payload.get("text"):
            first_text = payload.get("text")
            break
    return (
        f"Document title: {title}\n"
        f"Document id: {doc.get('doc_id')}\n"
        f"Sample section text: {first_text[:500]}\n"
        f"Seed (do not echo): {seed}\n\n"
        f"Produce ONE Korean retrieval query a user would type if they "
        f"wanted this specific document back. Use one sentence.\n"
        f"Return JSON of this shape:\n{_SCHEMA_HINT_RAG}"
    )


@dataclass
class _Templates:
    rag_queries: List[str]


def _bucket_rows(
    action: str,
    *,
    count: int,
    templates: _Templates,
    corpus: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if action == "rag":
        for i in range(count):
            doc = corpus[i % max(1, len(corpus))] if corpus else {}
            if templates.rag_queries and i < len(templates.rag_queries):
                query = templates.rag_queries[i]
            else:
                query = f"{doc.get('title', 'enterprise policy')}에 대해 알려주세요"
            rows.append({
                "query": query,
                "has_file": False,
                "file_mime": None,
                "expected_action": "rag",
                "notes": f"corpus question -> rag ({doc.get('doc_id', '?')})",
            })
    elif action == "ocr":
        for i in range(count):
            query = rng.choice(_OCR_REQUEST_TEMPLATES)
            rows.append({
                "query": query,
                "has_file": True,
                "file_mime": rng.choice(_FILE_MIMES),
                "expected_action": "ocr",
                "notes": "file + no (or ocr-keyword) question -> ocr",
            })
    elif action == "multimodal":
        prompts = [
            "이 이미지에 무엇이 있나요?",
            "이 문서에서 법인카드 한도를 찾아주세요",
            "이 포스터의 제목은?",
            "첨부 PDF의 p99 응답 지연 수치는?",
            "이 스크린샷에서 오류 메시지를 알려주세요",
        ]
        for i in range(count):
            rows.append({
                "query": rng.choice(prompts),
                "has_file": True,
                "file_mime": rng.choice(_FILE_MIMES),
                "expected_action": "multimodal",
                "notes": "question about file -> multimodal",
            })
    elif action == "direct_answer":
        for q in _DIRECT_ANSWER_TEMPLATES[:count]:
            rows.append({
                "query": q,
                "has_file": False,
                "file_mime": None,
                "expected_action": "direct_answer",
                "notes": "greeting/closing -> direct_answer",
            })
    elif action == "clarify":
        for q in _CLARIFY_TEMPLATES[:count]:
            rows.append({
                "query": q,
                "has_file": False,
                "file_mime": None,
                "expected_action": "clarify",
                "notes": "empty or too-short -> clarify",
            })
    return rows


def _generate_rag_query_templates(
    *,
    corpus: List[Dict[str, Any]],
    count: int,
    model: str,
    limiter: RateLimiter,
    gen_log: GenerationLog,
    client: Any,
) -> List[str]:
    if not corpus or client is None:
        return []
    out: List[str] = []
    for i, doc in enumerate(corpus[:count]):
        doc_id = str(doc.get("doc_id"))
        seed = stable_seed(doc_id, "routing")
        limiter.wait()
        try:
            with log_call(
                gen_log,
                script="generate_routing_cases",
                provider="claude",
                model=model,
                seed=seed,
                note=f"{doc_id}",
            ) as slot:
                parsed = claude_json_call(
                    client, model=model,
                    system=_SYSTEM_PROMPT_RAG,
                    user=_build_user_prompt_rag(doc, seed=seed),
                    max_tokens=400,
                    temperature=0.6,
                )
                usage = parsed.pop("_usage", {})
                slot["prompt_tokens"] = usage.get("input_tokens")
                slot["completion_tokens"] = usage.get("output_tokens")
            query = str(parsed.get("query", "")).strip()
            if query:
                out.append(query)
        except (ClaudeResponseError, Exception) as ex:  # noqa: BLE001
            log.warning("failed to draft routing query for %s: %s", doc_id, ex)
    return out


def generate(
    *,
    corpus_path: Path,
    out_path: Path,
    total: int,
    quota: Dict[str, int],
    model: str,
    rate_per_sec: float,
    dry_run: bool,
) -> int:
    corpus = read_jsonl(corpus_path)
    if not corpus:
        log.warning("corpus is empty — rag bucket will use title templates")

    # Normalize quota to total.
    quota_sum = sum(quota.values())
    if quota_sum != total and quota_sum > 0:
        scale = total / quota_sum
        quota = {k: max(0, int(round(v * scale))) for k, v in quota.items()}
        # Ensure we hit total exactly by topping up the largest bucket.
        delta = total - sum(quota.values())
        if delta != 0:
            largest = max(quota, key=quota.get)
            quota[largest] += delta

    log.info("Quota per action: %s (total=%d)", quota, sum(quota.values()))

    client = None
    if not dry_run:
        client = load_anthropic_client()
    gen_log = GenerationLog(out_path.parent / f"{out_path.stem}_generation_log.jsonl")
    limiter = RateLimiter(rate_per_sec)

    rag_templates = _generate_rag_query_templates(
        corpus=corpus,
        count=quota.get("rag", 0),
        model=model,
        limiter=limiter,
        gen_log=gen_log,
        client=client,
    )
    templates = _Templates(rag_queries=rag_templates)
    rng = random.Random(42)

    rows: List[Dict[str, Any]] = []
    for action in _ACTIONS:
        rows.extend(_bucket_rows(
            action,
            count=quota.get(action, 0),
            templates=templates,
            corpus=corpus,
            rng=rng,
        ))

    write_jsonl(out_path, rows, header=(
        f"Generated by scripts.dataset.generate_routing_cases\n"
        f"quota={quota} total={len(rows)}"
    ))
    log.info("Wrote %d rows to %s", len(rows), out_path)
    return len(rows)


def _parse_quota(spec: Optional[str]) -> Dict[str, int]:
    if not spec:
        return dict(_DEFAULT_QUOTA)
    out = dict(_DEFAULT_QUOTA)
    for chunk in spec.split(","):
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        k = k.strip()
        if k in _ACTIONS:
            out[k] = int(v)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=80)
    parser.add_argument("--quota", type=str, default=None,
                        help="override quota, e.g. rag:30,ocr:15,multimodal:15,"
                             "direct_answer:10,clarify:10")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--rate-per-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        generate(
            corpus_path=args.corpus,
            out_path=args.out,
            total=args.total,
            quota=_parse_quota(args.quota),
            model=args.model,
            rate_per_sec=args.rate_per_sec,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("generation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
