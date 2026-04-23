"""Generate Korean evaluation queries against the enterprise corpus.

Reads ``enterprise_corpus_kr/index.jsonl`` and emits a raw JSONL dataset
of ~400 rows to be narrowed down by ``validate_enterprise_dataset`` to
the ~200-row stratified eval set. Three categories of output:

  * **factoid / procedural / comparison** — per-doc generation. The
    script assigns a question type to each of N queries per doc via a
    largest-remainder split of ``--type-ratio``. The generator (Claude
    Haiku by default) sees the doc's body and must ground the query in
    it. Output rows carry ``expected_doc_ids=[<doc_id>]``,
    ``expected_keywords``, ``expected_answer_hint``, and
    ``evidence_chunk`` (a verbatim substring from the doc).
  * **unanswerable** — global batch. We derive a TF-IDF vocabulary
    across the corpus, then ask Haiku (per category) for plausible
    sibling terms that are NOT in the vocabulary. For each absent
    candidate the script does one more Haiku call to phrase a question
    around it. Output rows carry ``expected_doc_ids=[]``.

Usage (from ai-worker/)::

    python -m scripts.dataset.generate_enterprise_queries \\
        --corpus fixtures/enterprise_corpus_kr/index.jsonl \\
        --out    eval/datasets/rag_enterprise_kr_raw.jsonl \\
        --queries-per-doc 3 \\
        --type-ratio factoid:0.55,procedural:0.25,comparison:0.15,unanswerable:0.05 \\
        --generator claude:haiku
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

log = logging.getLogger("scripts.dataset.generate_enterprise_queries")


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


ANSWERABLE_TYPES: Tuple[str, ...] = ("factoid", "procedural", "comparison")
ALL_TYPES: Tuple[str, ...] = ANSWERABLE_TYPES + ("unanswerable",)


_TYPE_GUIDANCE: Dict[str, str] = {
    "factoid": (
        "단일 사실을 직접 묻는 질문. 정답이 특정 값(숫자/명사/역할/기간 등) "
        "하나로 떨어져야 합니다."
    ),
    "procedural": (
        "절차/단계/처리 순서를 묻는 질문. '어떻게', '어떤 순서로', "
        "'누구의 승인을 받아' 형태로 시작되는 경우가 많습니다."
    ),
    "comparison": (
        "두 항목/등급/금액 한도 사이의 차이를 묻는 질문. 답변에는 둘 "
        "이상의 개체가 등장해야 합니다."
    ),
    "unanswerable": (
        "회사에 있을 법한 업무 질문이지만, 현재 문서 코퍼스에서 명시적으로 "
        "다루지 않는 주제를 다룹니다. 답을 아는 척해서는 안 됩니다."
    ),
}


_CLAUDE_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4-5": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4-6": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "opus-4-7": "claude-opus-4-7",
}


def _resolve_generator(generator: str) -> Tuple[str, str]:
    if ":" in generator:
        provider, suffix = generator.split(":", 1)
    else:
        provider, suffix = "claude", generator
    if provider not in ("claude", "anthropic"):
        raise ValueError(f"Unsupported generator provider: {provider!r}")
    if suffix in _CLAUDE_MODEL_ALIASES:
        return "claude", _CLAUDE_MODEL_ALIASES[suffix]
    if suffix.startswith("claude-"):
        return "claude", suffix
    return "claude", f"claude-{suffix}"


# ---------------------------------------------------------------------------
# Ratio parsing + per-doc type assignment
# ---------------------------------------------------------------------------


def parse_type_ratio(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad --type-ratio token {chunk!r}")
        k, v = chunk.split(":", 1)
        k = k.strip()
        if k not in ALL_TYPES:
            raise ValueError(
                f"Unknown question type {k!r}. Expected one of {ALL_TYPES}."
            )
        out[k] = float(v)
    if not out:
        raise ValueError("--type-ratio produced no entries")
    total = sum(out.values())
    if total <= 0:
        raise ValueError("--type-ratio weights must be positive")
    return {k: v / total for k, v in out.items()}


def _assign_doc_types(
    doc_ids: List[str],
    ratio: Dict[str, float],
    queries_per_doc: int,
    seed: int,
) -> Dict[str, List[str]]:
    """Return doc_id -> list of N question types using largest-remainder.

    Unanswerable is intentionally handled separately and excluded from
    per-doc slots — the caller runs an extra batch for it.
    """
    slotted_types = [t for t in ANSWERABLE_TYPES if ratio.get(t, 0) > 0]
    if not slotted_types:
        return {doc_id: [] for doc_id in doc_ids}
    # Re-normalize over the types that go per-doc.
    total = sum(ratio[t] for t in slotted_types)
    per_doc_ratio = {t: ratio[t] / total for t in slotted_types}
    total_slots = len(doc_ids) * queries_per_doc
    raw = {t: per_doc_ratio[t] * total_slots for t in slotted_types}
    base = {t: int(raw[t]) for t in slotted_types}
    remainder = {t: raw[t] - base[t] for t in slotted_types}
    remaining = total_slots - sum(base.values())
    for t in sorted(remainder, key=lambda x: remainder[x], reverse=True)[:remaining]:
        base[t] += 1

    queue: List[str] = []
    for t in slotted_types:
        queue.extend([t] * base[t])
    # Deterministic shuffle keyed by the seed + doc_id ordering so the
    # assignment is stable across reruns.
    import random as _random
    rng = _random.Random(stable_seed(seed, "type_assignment"))
    rng.shuffle(queue)

    per_doc: Dict[str, List[str]] = {}
    for i, doc_id in enumerate(doc_ids):
        per_doc[doc_id] = queue[i * queries_per_doc:(i + 1) * queries_per_doc]
    return per_doc


def unanswerable_total(ratio: Dict[str, float], doc_count: int, queries_per_doc: int) -> int:
    """How many unanswerable queries to emit overall."""
    slotted = sum(ratio.get(t, 0) for t in ANSWERABLE_TYPES)
    total = doc_count * queries_per_doc
    if slotted <= 0:
        return total
    # Ratio is normalized; unanswerable = 1 - sum(answerable). Multiply by
    # TOTAL (answerable + unanswerable) = doc*q / slotted.
    total_all = total / slotted
    unanswerable = total_all - total
    return max(0, int(round(unanswerable)))


# ---------------------------------------------------------------------------
# TF-IDF over the corpus for unanswerable candidate mining
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-]+|"
    r"[\uAC00-\uD7A3]{2,}"
)


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _doc_terms(doc: Dict[str, Any]) -> List[str]:
    parts: List[str] = [str(doc.get("title", ""))]
    sections = doc.get("sections") or {}
    if isinstance(sections, dict):
        for body in sections.values():
            if not isinstance(body, dict):
                continue
            if body.get("text"):
                parts.append(str(body["text"]))
            chunks = body.get("chunks") or []
            if isinstance(chunks, list):
                parts.extend(str(c) for c in chunks if c)
    return _tokens("\n".join(parts))


@dataclass
class TfIdfVocab:
    """Very small bespoke TF-IDF — we don't need sklearn for 125 docs."""

    terms: List[str]
    idf: Dict[str, float]
    tf_by_doc: Dict[str, Counter]
    docs_by_category: Dict[str, List[str]]

    def top_terms_in_category(self, category: str, top_n: int) -> List[Tuple[str, float]]:
        doc_ids = self.docs_by_category.get(category, [])
        agg: Counter = Counter()
        for doc_id in doc_ids:
            agg.update(self.tf_by_doc.get(doc_id, {}))
        # Score = tf × idf, with idf pulling rare-across-corpus up.
        scored = [
            (term, count * self.idf.get(term, 0.0))
            for term, count in agg.items()
            if term in self.idf
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def contains(self, term: str) -> bool:
        return term.strip() in self.idf


def build_tfidf(docs: List[Dict[str, Any]]) -> TfIdfVocab:
    tf_by_doc: Dict[str, Counter] = {}
    docs_by_category: Dict[str, List[str]] = {}
    df: Counter = Counter()
    total_docs = 0
    for doc in docs:
        doc_id = str(doc.get("doc_id", ""))
        if not doc_id:
            continue
        terms = _doc_terms(doc)
        if not terms:
            continue
        total_docs += 1
        tf = Counter(terms)
        tf_by_doc[doc_id] = tf
        category = str(doc.get("category") or "uncategorized")
        docs_by_category.setdefault(category, []).append(doc_id)
        for term in tf:
            df[term] += 1
    idf: Dict[str, float] = {}
    if total_docs > 0:
        for term, count in df.items():
            idf[term] = math.log((total_docs + 1) / (count + 1)) + 1.0
    return TfIdfVocab(
        terms=sorted(idf),
        idf=idf,
        tf_by_doc=tf_by_doc,
        docs_by_category=docs_by_category,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are generating Korean-language evaluation queries for a RAG "
    "(retrieval-augmented generation) system over an internal Korean "
    "enterprise-policy corpus. Queries must sound like realistic "
    "employee questions; they must be grounded in the provided "
    "excerpts unless the prompt explicitly says otherwise. Respond "
    "with a single JSON object. Do NOT wrap the JSON in Markdown code "
    "fences and do NOT add any prose before or after the JSON."
)


def _extract_doc_body(doc: Dict[str, Any], max_chars: int = 5000) -> str:
    chunks: List[str] = [f"# {doc.get('title', '')}"]
    sections = doc.get("sections") or {}
    order = doc.get("section_order") or list(
        sections.keys() if isinstance(sections, dict) else []
    )
    for name in order:
        body = sections.get(name) if isinstance(sections, dict) else None
        if not isinstance(body, dict):
            continue
        chunks.append(f"## {name}")
        if body.get("text"):
            chunks.append(str(body["text"]))
        if isinstance(body.get("chunks"), list):
            chunks.extend(str(c) for c in body["chunks"] if c)
    joined = "\n".join(chunks)
    return joined[:max_chars]


def _build_answerable_user_prompt(
    *,
    doc: Dict[str, Any],
    question_type: str,
) -> str:
    body = _extract_doc_body(doc)
    guidance = _TYPE_GUIDANCE[question_type]
    return (
        f"Internal document (Korean):\n{body}\n\n"
        f"Generate ONE Korean evaluation query of type '{question_type}' "
        f"grounded in this document.\n\n"
        f"Type guidance:\n  {guidance}\n\n"
        f"Return JSON with this exact shape:\n"
        f"{{\n"
        f'  "query": "한국어 질문 (6-30 어절)",\n'
        f'  "expected_keywords": ["핵심 단어 1", "핵심 단어 2", "..."] ,\n'
        f'  "expected_answer_hint": "정답 요지 (2-3문장)",\n'
        f'  "evidence_chunk": "문서에서 근거가 되는 구절 (verbatim substring)"\n'
        f"}}\n\n"
        f"Rules:\n"
        f"  - query MUST be answerable from this document alone.\n"
        f"  - query MUST be in Korean.\n"
        f"  - expected_keywords: 2-5 Korean terms that a correct answer "
        f"must contain.\n"
        f"  - expected_answer_hint: a brief Korean sketch of the correct "
        f"answer, not the full text.\n"
        f"  - evidence_chunk: a VERBATIM substring (>= 20 chars) copied "
        f"from the document above.\n"
        f"  - Do NOT include the document title in the query itself — "
        f"imagine the user is searching broadly."
    )


def _build_unanswerable_user_prompt(
    *,
    category: str,
    absent_term: str,
    context_hint: str,
) -> str:
    return (
        f"Context: 우리 회사의 내부 정책 코퍼스는 '{category}' 영역을 다룹니다. "
        f"예를 들어 다음과 같은 용어들이 포함되어 있습니다: {context_hint}.\n\n"
        f"그러나 코퍼스는 '{absent_term}' 주제는 다루지 않습니다.\n\n"
        f"'{absent_term}' 에 대해 직원이 물어볼 법한 그럴듯한 한국어 "
        f"질문을 ONE 만들어 주세요. 질문은 자연스러워야 하고, 질문만 "
        f"보면 마치 코퍼스가 다룰 것 같은 인상을 줘야 합니다.\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'  "query": "한국어 질문 (6-30 어절)",\n'
        f'  "absent_term": "{absent_term}"\n'
        f"}}"
    )


def _build_sibling_term_prompt(
    *,
    category: str,
    present_terms: List[str],
    how_many: int,
) -> str:
    present_block = "\n".join(f"- {t}" for t in present_terms[:30])
    return (
        f"우리 회사의 '{category}' 영역 내부 정책 코퍼스에는 다음 용어들이 "
        f"많이 등장합니다:\n\n{present_block}\n\n"
        f"이 영역에서 일반적으로 함께 다뤄지지만, 위 목록에는 존재하지 "
        f"않는 '인접(sibling) 한국어 업무 용어'를 {how_many} 개 제안해 "
        f"주세요. 각 용어는 한국 회사의 실제 업무에서 쓰일 법한 표현이어야 "
        f"하고, 명사구여야 합니다.\n\n"
        f"Return JSON:\n"
        f'{{"absent_terms": ["term1", "term2", ...]}}'
    )


# ---------------------------------------------------------------------------
# Validation of generator output
# ---------------------------------------------------------------------------


def _validate_answerable_row(
    parsed: Dict[str, Any],
    *,
    doc_body: str,
) -> Dict[str, Any]:
    for field in ("query", "expected_keywords", "expected_answer_hint", "evidence_chunk"):
        if field not in parsed:
            raise ClaudeResponseError(f"missing field: {field}")
    query = str(parsed["query"]).strip()
    if not query:
        raise ClaudeResponseError("empty query")
    keywords = parsed["expected_keywords"]
    if not isinstance(keywords, list):
        raise ClaudeResponseError("expected_keywords must be a list")
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not (1 <= len(keywords) <= 8):
        raise ClaudeResponseError(
            f"expected_keywords must have 1-8 entries, got {len(keywords)}"
        )
    hint = str(parsed["expected_answer_hint"]).strip()
    if not hint:
        raise ClaudeResponseError("empty expected_answer_hint")
    evidence = str(parsed["evidence_chunk"]).strip()
    if not evidence:
        raise ClaudeResponseError("empty evidence_chunk")
    if len(evidence) < 15:
        raise ClaudeResponseError("evidence_chunk too short (< 15 chars)")
    # Soft check — evidence should appear in the doc body. Keep the row
    # even if not, but flag.
    evidence_in_doc = evidence in doc_body
    return {
        "query": query,
        "expected_keywords": keywords,
        "expected_answer_hint": hint[:400],
        "evidence_chunk": evidence[:800],
        "evidence_in_doc": evidence_in_doc,
    }


def _validate_unanswerable_row(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if "query" not in parsed:
        raise ClaudeResponseError("missing field: query")
    query = str(parsed["query"]).strip()
    if not query:
        raise ClaudeResponseError("empty query")
    absent_term = str(parsed.get("absent_term") or "").strip()
    return {"query": query, "absent_term": absent_term}


def _validate_sibling_terms(parsed: Dict[str, Any]) -> List[str]:
    terms = parsed.get("absent_terms")
    if not isinstance(terms, list):
        raise ClaudeResponseError("absent_terms must be a list")
    out = [str(t).strip() for t in terms if str(t).strip()]
    if not out:
        raise ClaudeResponseError("no candidate absent_terms returned")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _row_key(row: Dict[str, Any]) -> str:
    # Uniqueness over source_doc_id + question_type + slot_idx; for
    # unanswerable the source doc is the placeholder "unans:<category>".
    sdi = str(row.get("source_doc_id") or "")
    qt = str(row.get("question_type") or "")
    slot = str(row.get("slot_index") or 0)
    return f"{sdi}|{qt}|{slot}"


def _load_corpus(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    usable: List[Dict[str, Any]] = []
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        sections = row.get("sections")
        if not isinstance(sections, dict) or not sections:
            continue
        usable.append(row)
    return usable


def run(
    *,
    corpus_path: Path,
    out_path: Path,
    queries_per_doc: int,
    type_ratio: Dict[str, float],
    generator: str,
    seed: int,
    rate_per_sec: float,
    max_retries: int,
    sibling_terms_per_category: int,
    dry_run: bool,
) -> Dict[str, Any]:
    provider, model = _resolve_generator(generator)
    corpus = _load_corpus(corpus_path)
    if not corpus:
        raise SystemExit(f"corpus is empty: {corpus_path}")
    log.info("Loaded %d enterprise docs from %s", len(corpus), corpus_path)

    doc_ids = [str(d["doc_id"]) for d in corpus]
    assigned = _assign_doc_types(
        doc_ids, type_ratio, queries_per_doc=queries_per_doc, seed=seed,
    )
    unans_target = unanswerable_total(type_ratio, len(corpus), queries_per_doc)
    log.info(
        "Per-doc answerable slots: %s; unanswerable target: %d",
        {t: sum(1 for types in assigned.values() for x in types if x == t)
         for t in ANSWERABLE_TYPES},
        unans_target,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ResumableJsonlWriter(out_path, key_fn=_row_key)
    gen_log = GenerationLog(out_path.with_suffix(".generation_log.jsonl"))
    limiter = RateLimiter(rate_per_sec)

    client = None if dry_run else load_anthropic_client()

    stats = {
        "answerable_new": 0,
        "answerable_failed": 0,
        "unanswerable_new": 0,
        "unanswerable_failed": 0,
        "model": model,
        "provider": provider,
    }

    # ----- Answerable pass -------------------------------------------------
    by_doc = {str(d["doc_id"]): d for d in corpus}
    for doc_id, types in assigned.items():
        doc = by_doc.get(doc_id)
        if doc is None:
            continue
        for slot_index, qt in enumerate(types):
            row_key = f"{doc_id}|{qt}|{slot_index}"
            if writer.has(row_key):
                continue
            if dry_run:
                log.info("[dry-run] would gen answerable %s (%s, slot %d)",
                         doc_id, qt, slot_index)
                continue
            doc_body = _extract_doc_body(doc)
            doc_seed = stable_seed(seed, doc_id, qt, slot_index)
            user_prompt = _build_answerable_user_prompt(doc=doc, question_type=qt)

            parsed_row: Optional[Dict[str, Any]] = None
            for attempt in range(1, max_retries + 1):
                limiter.wait()
                try:
                    with log_call(
                        gen_log,
                        script="generate_enterprise_queries.answerable",
                        provider=provider,
                        model=model,
                        seed=doc_seed,
                        note=f"{doc_id}/{qt}/{slot_index} attempt={attempt}",
                    ) as slot:
                        raw = claude_json_call(
                            client,
                            model=model,
                            system=_SYSTEM_PROMPT,
                            user=user_prompt,
                            max_tokens=768,
                            temperature=0.6,
                        )
                        usage = raw.pop("_usage", {})
                        slot["prompt_tokens"] = usage.get("input_tokens")
                        slot["completion_tokens"] = usage.get("output_tokens")
                    parsed_row = _validate_answerable_row(raw, doc_body=doc_body)
                    break
                except Exception as ex:  # noqa: BLE001
                    log.warning(
                        "answerable %s/%s/%d attempt %d/%d failed: %s",
                        doc_id, qt, slot_index, attempt, max_retries, ex,
                    )
                    if attempt == max_retries:
                        stats["answerable_failed"] += 1
            if parsed_row is None:
                continue

            writer.append({
                "query": parsed_row["query"],
                "question_type": qt,
                "expected_doc_ids": [doc_id],
                "expected_keywords": parsed_row["expected_keywords"],
                "expected_answer_hint": parsed_row["expected_answer_hint"],
                "evidence_chunk": parsed_row["evidence_chunk"],
                "evidence_in_doc": parsed_row["evidence_in_doc"],
                "source_doc_id": doc_id,
                "category": doc.get("category"),
                "slot_index": slot_index,
                "generator": f"{provider}:{model}",
                "seed": doc_seed,
            })
            stats["answerable_new"] += 1

    # ----- Unanswerable pass ----------------------------------------------
    if unans_target > 0:
        vocab = build_tfidf(corpus)
        categories = sorted(vocab.docs_by_category)
        per_cat = max(1, math.ceil(unans_target / max(1, len(categories))))
        for category in categories:
            present_terms = [
                t for t, _ in vocab.top_terms_in_category(category, top_n=40)
            ]
            if not present_terms:
                continue
            if dry_run:
                log.info(
                    "[dry-run] would mine %d absent terms for category=%s",
                    sibling_terms_per_category, category,
                )
                continue
            cand_seed = stable_seed(seed, "unans-seed-terms", category)
            sibling_user = _build_sibling_term_prompt(
                category=category,
                present_terms=present_terms,
                how_many=sibling_terms_per_category,
            )
            absent_candidates: List[str] = []
            for attempt in range(1, max_retries + 1):
                limiter.wait()
                try:
                    with log_call(
                        gen_log,
                        script="generate_enterprise_queries.siblings",
                        provider=provider,
                        model=model,
                        seed=cand_seed,
                        note=f"{category} attempt={attempt}",
                    ) as slot:
                        raw = claude_json_call(
                            client,
                            model=model,
                            system=_SYSTEM_PROMPT,
                            user=sibling_user,
                            max_tokens=512,
                            temperature=0.7,
                        )
                        usage = raw.pop("_usage", {})
                        slot["prompt_tokens"] = usage.get("input_tokens")
                        slot["completion_tokens"] = usage.get("output_tokens")
                    absent_candidates = _validate_sibling_terms(raw)
                    break
                except Exception as ex:  # noqa: BLE001
                    log.warning(
                        "sibling-term mine %s attempt %d/%d failed: %s",
                        category, attempt, max_retries, ex,
                    )

            # Keep only those terms that are actually absent from the corpus.
            absent_candidates = [
                t for t in absent_candidates if not vocab.contains(t)
            ]
            log.info(
                "%s: kept %d absent sibling term(s) after corpus check",
                category, len(absent_candidates),
            )

            generated = 0
            for idx, term in enumerate(absent_candidates):
                if generated >= per_cat:
                    break
                row_key = f"unans:{category}|unanswerable|{idx}"
                if writer.has(row_key):
                    generated += 1
                    continue
                doc_seed = stable_seed(seed, "unans", category, term, idx)
                limiter.wait()
                user_prompt = _build_unanswerable_user_prompt(
                    category=category,
                    absent_term=term,
                    context_hint=", ".join(present_terms[:6]),
                )
                parsed_row = None
                for attempt in range(1, max_retries + 1):
                    try:
                        with log_call(
                            gen_log,
                            script="generate_enterprise_queries.unanswerable",
                            provider=provider,
                            model=model,
                            seed=doc_seed,
                            note=f"{category}/{term} attempt={attempt}",
                        ) as slot:
                            raw = claude_json_call(
                                client,
                                model=model,
                                system=_SYSTEM_PROMPT,
                                user=user_prompt,
                                max_tokens=384,
                                temperature=0.7,
                            )
                            usage = raw.pop("_usage", {})
                            slot["prompt_tokens"] = usage.get("input_tokens")
                            slot["completion_tokens"] = usage.get("output_tokens")
                        parsed_row = _validate_unanswerable_row(raw)
                        break
                    except Exception as ex:  # noqa: BLE001
                        log.warning(
                            "unans %s/%s attempt %d/%d failed: %s",
                            category, term, attempt, max_retries, ex,
                        )
                        if attempt == max_retries:
                            stats["unanswerable_failed"] += 1
                if parsed_row is None:
                    continue
                writer.append({
                    "query": parsed_row["query"],
                    "question_type": "unanswerable",
                    "expected_doc_ids": [],
                    "expected_keywords": [],
                    "expected_answer_hint": "",
                    "evidence_chunk": "",
                    "source_doc_id": f"unans:{category}:{term}",
                    "category": category,
                    "slot_index": idx,
                    "absent_term": parsed_row["absent_term"] or term,
                    "generator": f"{provider}:{model}",
                    "seed": doc_seed,
                })
                stats["unanswerable_new"] += 1
                generated += 1

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Generation done: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries-per-doc", type=int, default=3)
    parser.add_argument(
        "--type-ratio", type=str,
        default="factoid:0.55,procedural:0.25,comparison:0.15,unanswerable:0.05",
    )
    parser.add_argument("--generator", type=str, default="claude:haiku")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate-per-sec", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sibling-terms-per-category", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        ratio = parse_type_ratio(args.type_ratio)
    except ValueError as ex:
        parser.error(str(ex))
        return 2

    try:
        run(
            corpus_path=args.corpus,
            out_path=args.out,
            queries_per_doc=args.queries_per_doc,
            type_ratio=ratio,
            generator=args.generator,
            seed=args.seed,
            rate_per_sec=args.rate_per_sec,
            max_retries=args.max_retries,
            sibling_terms_per_category=args.sibling_terms_per_category,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("enterprise query generation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
