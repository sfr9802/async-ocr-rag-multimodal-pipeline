"""Build a synthetic Korean enterprise-document corpus via Claude Sonnet 4.6.

Generates N documents per category across HR / finance / IT / product /
legal. Each document lands on disk as one pretty-printed JSON file plus
a row in ``index.jsonl`` — the same shape the production RAG ingest
consumes (doc_id / title / sections{name: {chunks}} + a ``section_order``
hint). A generation log is appended beside the corpus so every doc
traces back to the exact model call that produced it.

Usage (from ai-worker/)::

    python -m scripts.dataset.build_corpus \\
        --out fixtures/corpus_kr \\
        --categories hr,finance,it,product,legal \\
        --per-category 25

Design notes
------------
* **Resumable.** Existing ``<doc_id>.json`` files are re-read and their
  ids are added to ``index.jsonl`` on the first pass, so re-running the
  script only generates missing docs. Crashing halfway through is safe.
* **Deterministic seed per doc.** ``seed = blake2b(category, idx)`` is
  written into each doc's metadata and into the prompt so the generator
  has a stable ``seed N of M`` anchor — rerunning with the same seed is
  encouraged to give comparable content.
* **Rate limited.** Default 0.5 req/s stays well under Claude tier-1
  limits. Flag is exposed so a team machine can dial it up.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from scripts.dataset._common import (
    ClaudeResponseError,
    GenerationLog,
    RateLimiter,
    ResumableJsonlWriter,
    claude_json_call,
    configure_logging,
    load_anthropic_client,
    log_call,
    stable_seed,
)

log = logging.getLogger("scripts.dataset.build_corpus")


# ---------------------------------------------------------------------------
# Category prompt specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategorySpec:
    """Describes how to prompt Claude for a single category.

    Held as data (not a chain of if/elif in the prompt builder) so adding
    a sixth category is one dict entry, not a new branch.
    """

    name: str
    korean_label: str
    style_guide: str
    example_titles: List[str]
    example_sections: List[str]


_CATEGORIES: Dict[str, CategorySpec] = {
    "hr": CategorySpec(
        name="hr",
        korean_label="인사/총무",
        style_guide=(
            "사내 인사/총무 정책 문서. 공식 어투(합니다체). 근거 조항, 예외 조항, "
            "관련 양식/문서 참조를 포함. 숫자(일수/횟수/금액)를 구체적으로."
        ),
        example_titles=[
            "연차 휴가 운영 지침",
            "재택근무 신청 및 승인 절차",
            "사내 경조사 지원 규정",
            "복리후생 포인트 운영 가이드",
            "신입사원 온보딩 체크리스트",
        ],
        example_sections=["개요", "적용대상", "세부지침", "예외사항", "관련문서"],
    ),
    "finance": CategorySpec(
        name="finance",
        korean_label="재무/회계",
        style_guide=(
            "재무/회계/지출 처리 규정. 구체적인 결재선, 한도 금액, 증빙 서류 명시. "
            "RFC 스타일 번호 매기기 (3.1, 3.2...)."
        ),
        example_titles=[
            "법인카드 사용 지침",
            "출장비 정산 절차",
            "자본적 지출(CapEx) 승인 기준",
            "분기 결산 마감 일정",
            "외화 송금 승인 절차",
        ],
        example_sections=["개요", "용어정의", "결재권한", "세부절차", "예외처리", "관련서식"],
    ),
    "it": CategorySpec(
        name="it",
        korean_label="IT/보안",
        style_guide=(
            "IT 운영/보안 가이드. 기술 용어 정확하게(PostgreSQL, Redis, K8s 등). "
            "구체적 설정값/임계치 포함. 장애 대응 체크리스트 풍."
        ),
        example_titles=[
            "Kubernetes 클러스터 운영 가이드",
            "데이터베이스 백업 및 복구 절차",
            "API 게이트웨이 보안 설정 표준",
            "로그 보존 정책",
            "엔드포인트 MDM 가입 절차",
        ],
        example_sections=["개요", "범위", "설정기준", "운영절차", "장애대응", "관련문서"],
    ),
    "product": CategorySpec(
        name="product",
        korean_label="제품/서비스",
        style_guide=(
            "B2B SaaS 제품 기획/스펙 문서. 사용자 시나리오, 수락 기준(AC), "
            "오픈 이슈를 포함. 공식적이면서 실무적인 어투."
        ),
        example_titles=[
            "검색 랭킹 v3 기획서",
            "요금제 업그레이드 플로우 명세",
            "관리자 대시보드 개편 요건서",
            "알림 채널 확장 로드맵",
            "데이터 연동(Integration) 요구사항",
        ],
        example_sections=["배경", "목표", "요구사항", "수락기준", "오픈이슈", "롤아웃계획"],
    ),
    "legal": CategorySpec(
        name="legal",
        korean_label="법무/계약",
        style_guide=(
            "법무/계약 관련 내부 가이드. 조항 번호(제1조, 제2조 ...), 위반 시 조치, "
            "면책 조항 포함. 법률 용어 정확하게."
        ),
        example_titles=[
            "외부 협력사 비밀유지계약(NDA) 체결 절차",
            "소프트웨어 라이선스 준수 가이드",
            "개인정보 처리 방침 운영 지침",
            "지적재산권 귀속 기준",
            "계약서 검토 요청 프로세스",
        ],
        example_sections=["목적", "정의", "적용범위", "의무사항", "위반시조치", "부칙"],
    ),
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a professional Korean enterprise-document writer producing "
    "realistic internal policy / spec / runbook documents for an IT "
    "company headquartered in Seoul. Every document must feel like it "
    "was written by a Korean corporate author: proper honorific register "
    "(합니다/습니다), RFC-style numbered clauses, concrete figures "
    "(dates, amounts, retention periods, SLAs), named roles (팀장, CTO, "
    "법무팀, 재무팀), and cross-references to related documents.\n\n"
    "Return a single JSON object matching the schema the user describes. "
    "Do NOT wrap the JSON in Markdown code fences. Do NOT add any prose "
    "before or after the JSON."
)


_SCHEMA_BLOCK = (
    "{\n"
    '  "doc_id": "<doc_id placeholder>",\n'
    '  "title": "<Korean title, under 80 chars>",\n'
    '  "category": "<category name>",\n'
    '  "sections": [\n'
    '    {"heading": "<section heading>", '
    '"text": "<2-5 Korean sentences per section, concrete numbers and named roles>"}\n'
    "  ],\n"
    '  "related_docs": ["<referenced policy name>", "..."]\n'
    "}"
)


def _build_user_prompt(
    *,
    category: CategorySpec,
    doc_id: str,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> str:
    titles = "\n".join(f"  - {t}" for t in category.example_titles)
    sections = ", ".join(category.example_sections)
    # Use the seed as a stable "idea anchor" — the model doesn't need to
    # seed its own RNG, it just needs a stable handle we can cite in the
    # log so two runs of the same seed produce comparable content.
    schema = _SCHEMA_BLOCK.replace("<doc_id placeholder>", doc_id).replace(
        "<category name>", category.name
    )
    return (
        f"Write ONE Korean enterprise document for category "
        f"'{category.korean_label}' ({category.name}). Use seed={seed} "
        f"as a creative anchor — pick a topic variant that is clearly "
        f"distinct from ones a neighbouring seed would pick.\n\n"
        f"Style guide: {category.style_guide}\n\n"
        f"Example titles for inspiration (do NOT copy verbatim):\n{titles}\n\n"
        f"Suggested section structure: {sections} (adapt as needed; 3-6 sections).\n\n"
        f"Total body length: between {min_chars} and {max_chars} Korean characters.\n\n"
        f"Return JSON with this exact shape:\n{schema}\n\n"
        f"Rules:\n"
        f"  - doc_id MUST equal {doc_id!r}.\n"
        f"  - Minimum 3 sections, maximum 6.\n"
        f"  - Each section must cite at least one specific figure "
        f"(amount / day count / threshold) or a specific role.\n"
        f"  - Include one exception clause somewhere in the doc.\n"
        f"  - No placeholder text (no 'TBD', 'lorem ipsum', 'xxx')."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_VALID_DOC_ID = re.compile(r"^[a-z]{2,}-\d{3,}$")


def _validate_doc(raw: Dict, *, expected_doc_id: str, min_chars: int, max_chars: int) -> Dict:
    """Best-effort schema + length validation.

    Raises ``ClaudeResponseError`` on anything that would make the doc
    useless to the RAG ingest — missing fields, empty sections, or a
    total body that's shorter than ``min_chars / 2`` (Claude sometimes
    returns a short placeholder if the prompt was too constrained).
    """
    for field in ("doc_id", "title", "sections"):
        if field not in raw:
            raise ClaudeResponseError(f"Missing required field: {field}")
    if not isinstance(raw["sections"], list) or not raw["sections"]:
        raise ClaudeResponseError("'sections' must be a non-empty list.")
    if not (3 <= len(raw["sections"]) <= 6):
        raise ClaudeResponseError(
            f"Expected 3-6 sections, got {len(raw['sections'])}"
        )

    total = 0
    cleaned_sections: List[Dict] = []
    for i, section in enumerate(raw["sections"]):
        if not isinstance(section, dict):
            raise ClaudeResponseError(f"section {i} is not an object.")
        heading = str(section.get("heading", "")).strip()
        text = str(section.get("text", "")).strip()
        if not heading or not text:
            raise ClaudeResponseError(f"section {i} missing heading or text.")
        total += len(text)
        cleaned_sections.append({"heading": heading, "text": text})

    lower_bound = max(100, min_chars // 2)
    if total < lower_bound:
        raise ClaudeResponseError(
            f"Body too short: {total} chars (expected at least {lower_bound})"
        )
    if total > max_chars * 2:
        # Hard cap — truncate or regenerate. We choose to accept but warn;
        # the downstream chunker will windowize.
        log.warning(
            "Doc %s body length %d exceeds max_chars*2=%d — accepting but over budget.",
            expected_doc_id, total, max_chars * 2,
        )

    related = raw.get("related_docs") or []
    if not isinstance(related, list):
        related = []

    return {
        "doc_id": expected_doc_id,
        "title": str(raw["title"]).strip()[:200],
        "category": str(raw.get("category", "")).strip(),
        "sections": cleaned_sections,
        "related_docs": [str(x).strip() for x in related if str(x).strip()],
    }


def _to_index_row(doc: Dict, *, seed: int, created_at: str) -> Dict:
    """Transform the per-doc JSON into the shape ``ingest.ingest_jsonl`` consumes.

    The ingest wraps each section's text through ``greedy_chunk`` +
    ``window_by_chars`` so we just need ``sections: {heading: {text}}``.
    ``section_order`` preserves authoring order for the report.

    Phase 9: every row carries ``domain='enterprise'`` and
    ``language='ko'`` so the unified ingest picks them up as enterprise-
    Korean docs. The per-doc ``category`` is preserved as-is so the
    metadata filter can scope retrieval to a single category.
    """
    sections_map: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for s in doc["sections"]:
        heading = s["heading"]
        # Collapse duplicate headings by appending a disambiguating suffix.
        dedup = heading
        suffix = 2
        while dedup in sections_map:
            dedup = f"{heading}#{suffix}"
            suffix += 1
        sections_map[dedup] = {"text": s["text"]}
        order.append(dedup)
    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "category": doc["category"],
        "domain": "enterprise",
        "language": "ko",
        "sections": sections_map,
        "section_order": order,
        "seed": seed,
        "created_at": created_at,
        "related_docs": doc.get("related_docs", []),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _doc_id_for(category: str, index: int) -> str:
    # kr-hr-001, kr-it-024, etc. Flat three-digit zero-padded suffix so
    # we can produce up to 999 docs per category without collisions.
    return f"kr-{category}-{index:03d}"


def _category_dir(out_dir: Path, category: str) -> Path:
    return out_dir / category


def _existing_doc(out_dir: Path, category: str, doc_id: str) -> Optional[Dict]:
    p = _category_dir(out_dir, category) / f"{doc_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        log.warning("Skipping corrupt existing doc %s: %s", p, ex)
        return None


def _write_doc(out_dir: Path, category: str, doc: Dict) -> Path:
    cat_dir = _category_dir(out_dir, category)
    cat_dir.mkdir(parents=True, exist_ok=True)
    path = cat_dir / f"{doc['doc_id']}.json"
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_corpus(
    *,
    out_dir: Path,
    categories: List[str],
    per_category: int,
    model: str,
    min_chars: int,
    max_chars: int,
    rate_per_sec: float,
    dry_run: bool,
    max_retries: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_log = GenerationLog(out_dir / "generation_log.jsonl")
    index_writer = ResumableJsonlWriter(
        out_dir / "index.jsonl", key_fn=lambda row: str(row.get("doc_id", ""))
    )
    limiter = RateLimiter(rate_per_sec)

    client = None
    if not dry_run:
        client = load_anthropic_client()

    from datetime import datetime, timezone

    total_existing = 0
    total_new = 0
    total_failed = 0

    for category in categories:
        if category not in _CATEGORIES:
            log.error("Unknown category %r — skipping", category)
            continue
        spec = _CATEGORIES[category]

        for idx in range(1, per_category + 1):
            doc_id = _doc_id_for(category, idx)
            seed = stable_seed(category, idx)

            existing = _existing_doc(out_dir, category, doc_id)
            if existing is not None:
                # Preserve the existing doc but make sure it's indexed.
                if not index_writer.has(doc_id):
                    index_writer.append(_to_index_row(
                        existing,
                        seed=existing.get("seed", seed),
                        created_at=existing.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    ))
                total_existing += 1
                log.info("[skip] %s already on disk", doc_id)
                continue

            if dry_run:
                log.info("[dry-run] would generate %s (seed=%d)", doc_id, seed)
                continue

            limiter.wait()
            user_prompt = _build_user_prompt(
                category=spec,
                doc_id=doc_id,
                seed=seed,
                min_chars=min_chars,
                max_chars=max_chars,
            )

            doc: Optional[Dict] = None
            for attempt in range(1, max_retries + 1):
                try:
                    with log_call(
                        gen_log,
                        script="build_corpus",
                        provider="claude",
                        model=model,
                        seed=seed,
                        note=f"{doc_id} attempt={attempt}",
                    ) as slot:
                        parsed = claude_json_call(
                            client,
                            model=model,
                            system=_SYSTEM_PROMPT,
                            user=user_prompt,
                            max_tokens=2048,
                            temperature=0.7,
                        )
                        usage = parsed.pop("_usage", {})
                        slot["prompt_tokens"] = usage.get("input_tokens")
                        slot["completion_tokens"] = usage.get("output_tokens")
                    doc = _validate_doc(
                        parsed,
                        expected_doc_id=doc_id,
                        min_chars=min_chars,
                        max_chars=max_chars,
                    )
                    break
                except (ClaudeResponseError, Exception) as ex:  # noqa: BLE001
                    log.warning(
                        "Generation for %s failed on attempt %d/%d: %s: %s",
                        doc_id, attempt, max_retries, type(ex).__name__, ex,
                    )
                    if attempt == max_retries:
                        total_failed += 1

            if doc is None:
                continue

            created_at = datetime.now(timezone.utc).isoformat()
            doc_disk = {
                **doc,
                "seed": seed,
                "created_at": created_at,
                "generator_model": model,
            }
            _write_doc(out_dir, category, doc_disk)
            index_writer.append(_to_index_row(doc, seed=seed, created_at=created_at))
            total_new += 1
            log.info("[new] %s (%d sections)", doc_id, len(doc["sections"]))

    log.info(
        "Corpus build complete: new=%d existing=%d failed=%d",
        total_new, total_existing, total_failed,
    )
    return total_new + total_existing


def _parse_categories(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _default_out_dir() -> Path:
    # Matches the example in the task spec.
    return Path(__file__).resolve().parent.parent.parent / "fixtures" / "corpus_kr"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=_default_out_dir(),
                        help="Output directory (default: fixtures/corpus_kr)")
    parser.add_argument("--categories", type=str,
                        default="hr,finance,it,product,legal",
                        help="Comma-separated category list")
    parser.add_argument("--per-category", type=int, default=25,
                        help="Docs per category (default: 25 -> 125 total)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Anthropic model id for generation")
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--rate-per-sec", type=float, default=0.5,
                        help="Maximum calls per second (default: 0.5)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retries per doc on validation/API error")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — don't call the model")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)

    categories = _parse_categories(args.categories)
    if not categories:
        log.error("No categories to build.")
        return 2

    try:
        total = build_corpus(
            out_dir=args.out,
            categories=categories,
            per_category=args.per_category,
            model=args.model,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            rate_per_sec=args.rate_per_sec,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("Corpus build failed: %s", ex)
        return 1

    if total < len(categories) * args.per_category * 0.5 and not args.dry_run:
        log.warning(
            "Only %d docs on disk (target %d) — check the generation log.",
            total, len(categories) * args.per_category,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
