"""Generate a stratified Korean eval-query set grounded in the sampled anime corpus.

The script does NOT invent anime content. For every sampled title it:

  1. Deterministically picks ONE source excerpt (``evidence_chunk``) from
     the namu-wiki sections that were sampled into ``anime_corpus_kr.jsonl``.
     The picked section is biased by ``question_type`` (e.g. ``character``
     queries draw from ``등장인물``) so the evidence covers the right
     material.
  2. Hands that single excerpt to a generator (Claude Haiku by default;
     also supports Claude Sonnet or local Ollama Gemma) whose sole job is
     to (a) phrase a natural Korean question whose answer is inside the
     excerpt, and (b) pick 2-5 keywords from the excerpt as the
     ``expected_keywords``.
  3. Rejects the generator output unless every keyword is a verbatim
     substring of the excerpt — the only way the row can be accepted is
     if the generator stayed strictly within the source.

This mirrors the hand-crafted style of ``eval/datasets/rag_sample_kr.jsonl``
(queries cite specific facts; ``expected_keywords`` are terms the answer
must contain) but scales it across the 300-title anime fixture without
fabricating content. The resulting JSONL is the raw input to
``validate_anime_dataset.py`` which narrows it to ~200 rows with
difficulty labels.

Usage (from ai-worker/)::

    python -m scripts.dataset.generate_anime_queries \\
        --corpus fixtures/anime_corpus_kr.jsonl \\
        --out    eval/datasets/rag_anime_kr_raw.jsonl \\
        --queries-per-title 1 \\
        --question-type-ratio factoid:0.5,plot:0.25,character:0.15,theme:0.10 \\
        --generator claude:haiku

Output rows::

    {"query", "question_type", "expected_doc_ids", "expected_keywords",
     "evidence_chunk", "source_title", "generator", "seed"}

Resumable: the writer dedups on ``(source_title, question_type)`` so a
partially-finished run picks up right where it stopped. Every generator
call (success or failure) lands in a sibling ``generation_log.jsonl``
alongside the output for audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.dataset._common import (
    ClaudeResponseError,
    GenerationLog,
    OllamaConfig,
    RateLimiter,
    ResumableJsonlWriter,
    claude_json_call,
    configure_logging,
    load_anthropic_client,
    log_call,
    ollama_chat_json,
    ollama_config_from_env,
    read_jsonl,
    stable_seed,
)

log = logging.getLogger("scripts.dataset.generate_anime_queries")


# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------


QUESTION_TYPES: Tuple[str, ...] = ("factoid", "plot", "character", "theme")

_TYPE_GUIDANCE = {
    "factoid": (
        "구체적인 사실(캐릭터 이름, 연도, 에피소드 수, 메커니즘 등)을 묻는 질문. "
        "답이 1-2 단어/구절로 떨어져야 합니다."
    ),
    "plot": (
        "스토리 전개, 사건, 갈등, 결말을 묻는 질문. 답변에 여러 문장이 필요해도 됩니다."
    ),
    "character": (
        "특정 캐릭터의 성격, 동기, 행동, 배경을 묻는 질문. 답변에 이름이 반드시 들어가야 합니다."
    ),
    "theme": (
        "작품 전체의 주제, 메시지, 분위기, 상징을 묻는 추상적 질문. "
        "추론/해석이 일부 필요할 수 있습니다."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    seed_title: str
    sections: Dict[str, Dict[str, Any]]
    section_order: List[str]

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CorpusDoc":
        sections = row.get("sections") or {}
        order = row.get("section_order") or list(sections.keys())
        return cls(
            doc_id=str(row["doc_id"]),
            title=str(row.get("title") or ""),
            seed_title=str(row.get("seed_title") or row.get("title") or ""),
            sections=sections if isinstance(sections, dict) else {},
            section_order=list(order),
        )

    def all_chunks(self) -> List[Tuple[str, str]]:
        """List of (section_name, chunk_text) flattened in section_order."""
        out: List[Tuple[str, str]] = []
        for name in self.section_order:
            body = self.sections.get(name) or {}
            chunks = body.get("chunks") or []
            if isinstance(chunks, list):
                for c in chunks:
                    if isinstance(c, str) and c.strip():
                        out.append((name, c.strip()))
            text = body.get("text")
            if not chunks and isinstance(text, str) and text.strip():
                out.append((name, text.strip()))
        return out


def _parse_ratio(spec: str) -> Dict[str, float]:
    """Parse ``factoid:0.5,plot:0.25,...`` into a normalized dict."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: Dict[str, float] = {}
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad ratio token {p!r} — expected name:weight")
        name, weight = p.split(":", 1)
        name = name.strip()
        if name not in QUESTION_TYPES:
            raise ValueError(
                f"Unknown question_type {name!r}. "
                f"Expected one of {QUESTION_TYPES}."
            )
        out[name] = float(weight)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("All ratio weights are zero.")
    return {k: v / total for k, v in out.items()}


def _assign_types(
    docs: List[CorpusDoc],
    ratio: Dict[str, float],
    seed: int,
) -> Dict[str, str]:
    """Assign a question_type to each doc_id deterministically by seed.

    Uses the standard largest-remainder method so the counts match the
    requested ratio within ±1 per bucket.
    """
    n = len(docs)
    raw = {t: ratio.get(t, 0.0) * n for t in QUESTION_TYPES}
    base = {t: int(raw[t]) for t in QUESTION_TYPES}
    remainder = {t: raw[t] - base[t] for t in QUESTION_TYPES}
    short = n - sum(base.values())
    # Hand out the remaining slots to the types with the largest remainders.
    for t in sorted(remainder, key=remainder.get, reverse=True)[:short]:
        base[t] += 1

    counts = dict(base)
    # Deterministic shuffle by (seed, doc_id) so the assignment is stable
    # across reruns of this script.
    sorted_docs = sorted(docs, key=lambda d: stable_seed(seed, d.doc_id))

    assignment: Dict[str, str] = {}
    type_queue: List[str] = []
    for t in QUESTION_TYPES:
        type_queue.extend([t] * counts[t])
    # Length of type_queue equals n; pair up with sorted_docs.
    for doc, qt in zip(sorted_docs, type_queue):
        assignment[doc.doc_id] = qt
    return assignment


# Section-name hints that tell us which part of a namu-wiki entry likely
# contains the material for a given question_type. Each entry is a list
# of lowercased substring hints checked against the section name.
_SECTION_HINTS: Dict[str, Tuple[str, ...]] = {
    "factoid": ("요약", "개요", "정보", "설정", "작품", "제작"),
    "plot": ("줄거리", "본문", "스토리", "에피소드", "전개", "내용"),
    "character": ("등장인물", "인물", "캐릭터", "주요", "캐스트"),
    "theme": ("평가", "주제", "테마", "특징", "비판", "호평", "감상"),
}

_EVIDENCE_MIN_CHARS = 40
_EVIDENCE_MAX_CHARS = 900


def _section_matches(section_name: str, question_type: str) -> bool:
    hints = _SECTION_HINTS.get(question_type, ())
    lowered = section_name.lower()
    return any(h in lowered or h in section_name for h in hints)


def _select_evidence_chunk(
    doc: CorpusDoc,
    *,
    question_type: str,
    seed: int,
) -> Optional[Tuple[str, str]]:
    """Pick ONE grounded evidence chunk for this (doc, question_type).

    Deterministic and heavily biased toward the section whose name
    matches the question_type (e.g. ``character`` queries come from the
    ``등장인물`` section). Falls back to any non-empty chunk if no
    section matches. This is the sole source of truth for the query: the
    LLM sees only this one chunk and is forbidden from inventing facts
    beyond it, which is what keeps the generated queries grounded in
    real namu-wiki content.
    """
    all_chunks = doc.all_chunks()
    if not all_chunks:
        return None

    def _usable(pair: Tuple[str, str]) -> bool:
        _, text = pair
        return _EVIDENCE_MIN_CHARS <= len(text) <= _EVIDENCE_MAX_CHARS

    preferred: List[Tuple[str, str]] = [
        pair for pair in all_chunks
        if _usable(pair) and _section_matches(pair[0], question_type)
    ]
    fallback_usable: List[Tuple[str, str]] = [
        pair for pair in all_chunks if _usable(pair)
    ]
    # If nothing meets the length band, just use whatever we have so
    # that short-section titles (e.g. very terse anime pages) still
    # contribute a query rather than silently dropping.
    pool = preferred or fallback_usable or all_chunks
    rng = random.Random(stable_seed(seed, doc.doc_id, question_type, "evid"))
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a Korean reading-comprehension question author building an "
    "evaluation set for a RAG (retrieval-augmented generation) system "
    "over a real Korean-language anime knowledge base (namu-wiki excerpts). "
    "Your ONLY source of truth is the single excerpt the user provides. "
    "Do not invent character names, plot events, dates, or themes that "
    "are not literally present in that excerpt. Return a single JSON "
    "object. Do NOT wrap it in Markdown code fences. Do NOT add prose "
    "before or after the JSON."
)


def _build_answerable_user_prompt(
    *,
    doc: CorpusDoc,
    question_type: str,
    section_name: str,
    evidence_chunk: str,
) -> str:
    guidance = _TYPE_GUIDANCE[question_type]
    return (
        f"Anime title: {doc.title}\n"
        f"Section: {section_name}\n"
        f"Source excerpt (THE ONLY material you may use; do not introduce "
        f"information from outside this block):\n\n"
        f"\"\"\"\n{evidence_chunk}\n\"\"\"\n\n"
        f"Generate ONE Korean-language evaluation query of type "
        f"'{question_type}' whose correct answer is entirely extractable "
        f"from the excerpt above.\n\n"
        f"Type guidance:\n  {guidance}\n\n"
        f"Return JSON with this exact shape:\n"
        f"{{\n"
        f'  "query": "한국어 질문 (5-25 어절)",\n'
        f'  "expected_keywords": ["발췌문에서 그대로 복사한 단어/구절 1", "2", "3"]\n'
        f"}}\n\n"
        f"Rules:\n"
        f"  - query MUST be in Korean (한국어).\n"
        f"  - query MUST be answerable using ONLY the excerpt above — no "
        f"outside knowledge, no speculation.\n"
        f"  - expected_keywords: 2-5 short terms (Korean nouns, English "
        f"names, numbers, etc.) that a correct answer must contain. "
        f"EVERY keyword MUST appear verbatim as a substring of the excerpt "
        f"above — do not paraphrase, translate, or abbreviate.\n"
        f"  - Do NOT include the anime title in the query itself — imagine "
        f"the user has already selected the anime in a UI and is now "
        f"asking a follow-up.\n"
        f"  - Do NOT fabricate: if the excerpt only tells you the genre, "
        f"ask about the genre. Don't invent a character's backstory the "
        f"excerpt never mentions."
    )


def _build_unanswerable_user_prompt(*, title: str) -> str:
    return (
        f"Anime title: {title}\n\n"
        f"This anime's source material is NOT in the retrieval index — the "
        f"RAG system should refuse or say it doesn't know.\n\n"
        f"Generate ONE plausible Korean-language question that a user might "
        f"ask about this anime. The question should look reasonable but be "
        f"impossible to answer reliably from general knowledge alone.\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'  "query": "한국어 질문 (5-25 어절)",\n'
        f'  "expected_keywords": []\n'
        f"}}\n\n"
        f"Rules:\n"
        f"  - query MUST be in Korean.\n"
        f"  - Do NOT include the title in the query itself."
    )


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------


def _parse_generator_spec(spec: str) -> Tuple[str, str]:
    """Parse ``provider:model`` → ``(provider, model)``."""
    if ":" not in spec:
        raise ValueError(f"Generator spec {spec!r} must be 'provider:model'.")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in ("claude", "ollama"):
        raise ValueError(f"Unknown generator provider {provider!r}")
    if not model:
        raise ValueError("Model name is empty.")
    return provider, model


_CLAUDE_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


def _resolve_claude_model(name: str) -> str:
    return _CLAUDE_MODEL_ALIASES.get(name, name)


def _call_generator(
    provider: str,
    model: str,
    *,
    claude_client: Any = None,
    ollama_cfg: Optional[OllamaConfig] = None,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if provider == "claude":
        if claude_client is None:
            raise RuntimeError("Claude client is not initialised.")
        return claude_json_call(
            claude_client,
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if provider == "ollama":
        cfg = ollama_cfg or ollama_config_from_env()
        cfg = OllamaConfig(
            base_url=cfg.base_url,
            model=model,
            timeout_s=cfg.timeout_s,
            keep_alive=cfg.keep_alive,
        )
        schema = (
            '{"query": "string", "expected_keywords": ["string"], '
            '"evidence_chunk": "string"}'
        )
        return ollama_chat_json(
            cfg,
            system=system,
            user=user,
            schema_hint=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise RuntimeError(f"Unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Validation of generator output
# ---------------------------------------------------------------------------


def _validate_answerable(
    parsed: Dict[str, Any],
    *,
    evidence_chunk: str,
    min_keyword_len: int = 1,
) -> Dict[str, Any]:
    """Reject generator output that isn't strictly grounded in the excerpt.

    The evidence_chunk itself is NOT taken from the model — the caller
    pre-selected it. Here we enforce the only remaining trust boundary:
    every keyword the model proposed must literally appear as a
    substring of the excerpt. If the model paraphrased, translated, or
    invented a term, the row is rejected so the caller can retry.
    """
    for field in ("query", "expected_keywords"):
        if field not in parsed:
            raise ClaudeResponseError(f"missing field: {field}")
    query = str(parsed["query"]).strip()
    if not query:
        raise ClaudeResponseError("query is empty")
    keywords = parsed["expected_keywords"]
    if not isinstance(keywords, list):
        raise ClaudeResponseError("expected_keywords must be a list")
    cleaned_keywords = []
    for k in keywords:
        s = str(k).strip()
        if len(s) >= min_keyword_len:
            cleaned_keywords.append(s)
    if not (2 <= len(cleaned_keywords) <= 5):
        raise ClaudeResponseError(
            f"expected_keywords must have 2-5 entries, got {len(cleaned_keywords)}"
        )
    ungrounded = [k for k in cleaned_keywords if k not in evidence_chunk]
    if ungrounded:
        raise ClaudeResponseError(
            f"expected_keywords not found verbatim in evidence: {ungrounded}"
        )
    return {
        "query": query,
        "expected_keywords": cleaned_keywords,
    }


def _validate_unanswerable(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if "query" not in parsed:
        raise ClaudeResponseError("missing field: query")
    query = str(parsed["query"]).strip()
    if not query:
        raise ClaudeResponseError("query is empty")
    return {"query": query, "expected_keywords": []}


# ---------------------------------------------------------------------------
# Unanswerable titles sampler
# ---------------------------------------------------------------------------


def _pick_outside_corpus_titles(
    source_path: Path,
    *,
    excluded_seed_titles: set,
    how_many: int,
    seed: int,
) -> List[str]:
    """Reservoir-sample ``how_many`` seed_titles NOT in ``excluded_seed_titles``."""
    rng = random.Random(stable_seed(seed, "unanswerable"))
    reservoir: List[str] = []
    total_seen = 0
    seen_here: set = set()
    with source_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = str(rec.get("seed_title") or rec.get("title") or "").strip()
            if not st:
                continue
            if st in excluded_seed_titles or st in seen_here:
                continue
            seen_here.add(st)
            if len(reservoir) < how_many:
                reservoir.append(st)
            else:
                j = rng.randint(0, total_seen)
                if j < how_many:
                    reservoir[j] = st
            total_seen += 1
    return reservoir


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _row_key(row: Dict[str, Any]) -> str:
    return f"{row.get('source_title','')}|{row.get('question_type','')}"


def run(
    *,
    corpus_path: Path,
    out_path: Path,
    ratio: Dict[str, float],
    generator_spec: str,
    seed: int,
    rate_per_sec: float,
    max_retries: int,
    unanswerable_count: int,
    source_path: Optional[Path],
    dry_run: bool,
) -> int:
    provider, model_raw = _parse_generator_spec(generator_spec)
    model = _resolve_claude_model(model_raw) if provider == "claude" else model_raw

    corpus_rows = read_jsonl(corpus_path)
    if not corpus_rows:
        log.error("Empty corpus: %s", corpus_path)
        return 2
    docs = [CorpusDoc.from_row(r) for r in corpus_rows]
    log.info("Loaded %d docs from %s", len(docs), corpus_path)

    assignment = _assign_types(docs, ratio, seed)
    type_counts: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    for qt in assignment.values():
        type_counts[qt] = type_counts.get(qt, 0) + 1
    log.info("Question-type assignment: %s", type_counts)

    writer = ResumableJsonlWriter(out_path, key_fn=_row_key)
    gen_log = GenerationLog(out_path.with_suffix(".generation_log.jsonl"))
    limiter = RateLimiter(rate_per_sec)

    claude_client = None
    ollama_cfg: Optional[OllamaConfig] = None
    if not dry_run:
        if provider == "claude":
            claude_client = load_anthropic_client()
        else:
            ollama_cfg = ollama_config_from_env()

    total_new = 0
    total_failed = 0

    for doc in docs:
        question_type = assignment[doc.doc_id]
        row_key = f"{doc.title}|{question_type}"
        if writer.has(row_key):
            continue
        evidence = _select_evidence_chunk(
            doc, question_type=question_type, seed=seed,
        )
        if evidence is None:
            log.warning("doc %s has no usable chunks — skipping", doc.doc_id)
            continue
        section_name, evidence_chunk_text = evidence
        doc_seed = stable_seed(seed, doc.doc_id, question_type)

        if dry_run:
            log.info(
                "[dry-run] would gen %s / %s (section=%s, evidence=%d chars)",
                doc.doc_id, question_type, section_name, len(evidence_chunk_text),
            )
            continue

        limiter.wait()
        user_prompt = _build_answerable_user_prompt(
            doc=doc,
            question_type=question_type,
            section_name=section_name,
            evidence_chunk=evidence_chunk_text,
        )

        parsed_row: Optional[Dict[str, Any]] = None
        for attempt in range(1, max_retries + 1):
            try:
                with log_call(
                    gen_log,
                    script="generate_anime_queries",
                    provider=provider,
                    model=model,
                    seed=doc_seed,
                    note=f"{doc.doc_id}/{question_type} attempt={attempt}",
                ) as slot:
                    raw = _call_generator(
                        provider,
                        model,
                        claude_client=claude_client,
                        ollama_cfg=ollama_cfg,
                        system=_SYSTEM_PROMPT,
                        user=user_prompt,
                        temperature=0.5,
                        max_tokens=384,
                    )
                    usage = raw.pop("_usage", {})
                    slot["prompt_tokens"] = usage.get("input_tokens")
                    slot["completion_tokens"] = usage.get("output_tokens")
                parsed_row = _validate_answerable(
                    raw, evidence_chunk=evidence_chunk_text,
                )
                break
            except Exception as ex:  # noqa: BLE001
                log.warning(
                    "gen %s/%s attempt %d/%d failed: %s",
                    doc.doc_id, question_type, attempt, max_retries, ex,
                )
                if attempt == max_retries:
                    total_failed += 1
        if parsed_row is None:
            continue

        writer.append({
            "query": parsed_row["query"],
            "question_type": question_type,
            "expected_doc_ids": [doc.doc_id],
            "expected_keywords": parsed_row["expected_keywords"],
            "evidence_chunk": evidence_chunk_text,
            "source_title": doc.title,
            "source_seed_title": doc.seed_title,
            "source_section": section_name,
            "notes": f"{section_name} 섹션 기반 — {doc.title}",
            "generator": f"{provider}:{model}",
            "seed": doc_seed,
        })
        total_new += 1
        if total_new % 25 == 0:
            log.info("Progress: %d / %d answerable generated",
                     total_new, len(docs))

    # ---- unanswerable batch --------------------------------------------
    if unanswerable_count > 0 and source_path is not None and not dry_run:
        excluded = {d.seed_title for d in docs}
        titles_outside = _pick_outside_corpus_titles(
            source_path,
            excluded_seed_titles=excluded,
            how_many=max(1, unanswerable_count // 2),  # 2 queries per title
            seed=seed,
        )
        log.info("Selected %d outside-corpus titles for unanswerable queries",
                 len(titles_outside))

        queries_left = unanswerable_count
        for title in titles_outside:
            for sub in range(2):
                if queries_left <= 0:
                    break
                key = f"{title}|unanswerable#{sub}"
                if writer.has(key):
                    queries_left -= 1
                    continue
                limiter.wait()
                doc_seed = stable_seed(seed, "unans", title, sub)
                user_prompt = _build_unanswerable_user_prompt(title=title)
                parsed_row = None
                for attempt in range(1, max_retries + 1):
                    try:
                        with log_call(
                            gen_log,
                            script="generate_anime_queries",
                            provider=provider,
                            model=model,
                            seed=doc_seed,
                            note=f"unans {title} sub={sub} attempt={attempt}",
                        ) as slot:
                            raw = _call_generator(
                                provider,
                                model,
                                claude_client=claude_client,
                                ollama_cfg=ollama_cfg,
                                system=_SYSTEM_PROMPT,
                                user=user_prompt,
                                temperature=0.7,
                                max_tokens=256,
                            )
                            usage = raw.pop("_usage", {})
                            slot["prompt_tokens"] = usage.get("input_tokens")
                            slot["completion_tokens"] = usage.get("output_tokens")
                        parsed_row = _validate_unanswerable(raw)
                        break
                    except Exception as ex:  # noqa: BLE001
                        log.warning(
                            "unans %s sub=%d attempt %d/%d failed: %s",
                            title, sub, attempt, max_retries, ex,
                        )
                        if attempt == max_retries:
                            total_failed += 1
                if parsed_row is None:
                    continue
                writer.append({
                    "query": parsed_row["query"],
                    "question_type": "unanswerable",
                    "expected_doc_ids": [],
                    "expected_keywords": [],
                    "evidence_chunk": "",
                    "source_title": title,
                    "source_seed_title": title,
                    "generator": f"{provider}:{model}",
                    "seed": doc_seed,
                    "_subindex": sub,
                })
                total_new += 1
                queries_left -= 1

    log.info(
        "Generation complete: new=%d failed=%d (file: %s)",
        total_new, total_failed, out_path,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Sampled corpus JSONL (anime_corpus_kr.jsonl).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Raw query output JSONL.")
    parser.add_argument("--queries-per-title", type=int, default=1,
                        help="Queries per sampled title (currently only 1 supported).")
    parser.add_argument(
        "--question-type-ratio",
        type=str,
        default="factoid:0.5,plot:0.25,character:0.15,theme:0.10",
        help="Comma-separated weight ratio by question type.",
    )
    parser.add_argument("--generator", type=str, default="claude:haiku",
                        help="provider:model (e.g. claude:haiku, claude:sonnet, ollama:gemma4:e2b).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate-per-sec", type=float, default=1.0,
                        help="Rate limit for generator calls (default 1 req/s).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--unanswerable-count", type=int, default=10,
                        help="How many unanswerable queries to add on top.")
    parser.add_argument(
        "--source-full",
        type=Path,
        default=None,
        help=(
            "Path to the full source JSONL. Required when "
            "--unanswerable-count > 0; the script picks titles not present "
            "in the sampled corpus."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)

    if args.queries_per_title != 1:
        parser.error("Only --queries-per-title 1 is supported at present.")

    try:
        ratio = _parse_ratio(args.question_type_ratio)
    except ValueError as ex:
        parser.error(str(ex))
        return 2

    if args.unanswerable_count > 0 and args.source_full is None:
        parser.error(
            "--source-full is required when --unanswerable-count > 0."
        )

    return run(
        corpus_path=args.corpus,
        out_path=args.out,
        ratio=ratio,
        generator_spec=args.generator,
        seed=args.seed,
        rate_per_sec=args.rate_per_sec,
        max_retries=args.max_retries,
        unanswerable_count=args.unanswerable_count,
        source_path=args.source_full,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
