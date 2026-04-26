"""LLM-backed query generator that drives the local `claude` CLI.

Pattern parallels the optuna-round-refinement skill: each generation
batch is one fresh `claude --print --model <m>` subprocess. The outer
process does no LLM work itself — it just reads doc excerpts off the
corpus, spawns a headless Claude Code call, and parses the JSON the
model wrote to stdout.

Why this exists alongside the SDK-backed legacy generator
---------------------------------------------------------
The legacy ``scripts.dataset.generate_anime_queries`` script speaks
the Anthropic Python SDK directly and needs ``ANTHROPIC_API_KEY``. On
machines that have Claude Code installed but no separate API key, the
``claude`` CLI re-uses the user's existing OAuth auth — same model
access, no duplicate key-management. This module is the bridge.

Trade-off: subprocess overhead is real (~5s per call session init).
We amortize by batching ``DOCS_PER_BATCH`` (5) documents per call —
the model produces a JSON array of 5 query rows in one round-trip
instead of 5 independent calls.

Output rows match the schema produced by ``generate_eval_queries.py``
deterministic mode, so the two are interchangeable downstream:

    {
      "id":                       "anime-silver-llm-{seq}",
      "query":                    "Korean question",
      "language":                 "ko",
      "expected_doc_ids":         ["<doc_id>"],
      "expected_section_keywords":["<kw>", ...],
      "answer_type":              "summary_plot" | ...,
      "difficulty":               "easy" | "medium" | "hard",
      "tags":                     ["anime", "silver", "synthetic",
                                   "<answer_type>", "claude-cli:<model>"]
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("eval.harness.llm_subprocess_generator")

DOCS_PER_BATCH = 5
SUBPROCESS_TIMEOUT = 180  # seconds per call — generous for batch of 5
MAX_RETRIES_PER_BATCH = 2
EXCERPT_CHARS_PER_DOC = 800  # cap per-doc material handed to the model

VALID_ANSWER_TYPES = {
    "summary_plot",
    "title_lookup",
    "character_relation",
    "body_excerpt",
    "theme_genre",
    "setting_worldbuilding",
}

DIFFICULTY_BY_TYPE: Dict[str, str] = {
    "summary_plot": "medium",
    "title_lookup": "easy",
    "character_relation": "medium",
    "body_excerpt": "hard",
    "theme_genre": "medium",
    "setting_worldbuilding": "hard",
}


# ---------------------------------------------------------------------------
# CLI discovery + system prompt.
# ---------------------------------------------------------------------------


def find_claude_cli(explicit: Optional[str] = None) -> str:
    """Return the path to the local ``claude`` executable.

    Order: explicit override → ``$PATH`` lookup → the well-known Windows
    install location under ``%LOCALAPPDATA%/.local/bin``. Raises
    ``FileNotFoundError`` if the CLI cannot be located — we surface a
    specific install hint rather than failing inside subprocess.
    """
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"--claude-cli {explicit!r} does not exist")
        return explicit

    found = shutil.which("claude")
    if found:
        return found

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / ".local" / "bin" / "claude.exe",
        Path.home() / ".local" / "bin" / "claude.exe",
        Path.home() / ".local" / "bin" / "claude",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Could not locate the `claude` CLI. Install Claude Code "
        "(https://claude.com/claude-code) or pass --claude-cli <path>."
    )


SYSTEM_PROMPT = """You generate evaluation queries for a Korean anime retrieval system.

Each user message hands you a numbered batch of corpus excerpts.
Produce ONE Korean retrieval query for EACH excerpt and return ONLY a
JSON array of objects, no markdown, no preamble, no commentary.

Each object MUST have exactly these fields:

{
  "doc_id":                    "<copied verbatim from the excerpt header>",
  "query":                     "<natural Korean question whose answer is in the excerpt>",
  "expected_section_keywords": ["<2-4 substrings copied verbatim from the excerpt>"],
  "answer_type":               "<one of the allowed types below>",
  "difficulty":                "<easy | medium | hard>"
}

Allowed answer_type values:
  - summary_plot          (about the plot / story summary)
  - title_lookup          (about what the work is / its identity)
  - character_relation    (about a character or relationships between characters)
  - body_excerpt          (about a specific detail in the body text)
  - theme_genre           (about themes, genre, tone)
  - setting_worldbuilding (about the worldbuilding / setting / rules)

HARD CONSTRAINTS (enforced by validator — your row is dropped if you violate):
1. Every string in expected_section_keywords MUST appear verbatim somewhere
   in that excerpt's text. Copy them character-for-character. Casing matters.
2. The query must be answerable from THIS excerpt alone — do not assume
   external knowledge.
3. Diversify answer_type across the batch. Do not output the same
   answer_type for every entry.
4. Output ONLY the raw JSON array. No ```json fences, no explanation.

If you cannot construct a valid row for some excerpt (insufficient text,
ambiguous content), still include an entry but set "query" to "" — the
validator will skip it cleanly.
"""


# ---------------------------------------------------------------------------
# Doc preparation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DocExcerpt:
    doc_id: str
    title: str
    summary: str
    body_excerpt: str
    char_excerpt: Optional[str]  # 등장인물 section text, when present
    setting_excerpt: Optional[str]  # 설정 section text, when present


def _doc_to_excerpt(doc: Dict[str, Any]) -> Optional[_DocExcerpt]:
    """Project a raw corpus row into the bounded excerpt the model sees."""
    doc_id = str(doc.get("doc_id") or "").strip()
    title = str(doc.get("title") or "").strip()
    summary = str(doc.get("summary") or "").strip()
    if not doc_id or not title or not summary:
        return None
    sections = doc.get("sections") or {}
    if not isinstance(sections, dict):
        sections = {}

    def section_text(name: str) -> Optional[str]:
        s = sections.get(name)
        if not isinstance(s, dict):
            return None
        text = str(s.get("text") or "").strip()
        return text or None

    body_text = section_text("본문") or ""
    body_excerpt = body_text[:EXCERPT_CHARS_PER_DOC] if body_text else ""
    char_text = section_text("등장인물")
    char_excerpt = char_text[:EXCERPT_CHARS_PER_DOC] if char_text else None
    setting_text = section_text("설정") or section_text("세계관")
    setting_excerpt = setting_text[:EXCERPT_CHARS_PER_DOC] if setting_text else None
    return _DocExcerpt(
        doc_id=doc_id,
        title=title,
        summary=summary[:EXCERPT_CHARS_PER_DOC],
        body_excerpt=body_excerpt,
        char_excerpt=char_excerpt,
        setting_excerpt=setting_excerpt,
    )


def _format_batch_prompt(batch: List[_DocExcerpt]) -> str:
    """Render the user-message body for one batch."""
    chunks: List[str] = []
    for i, d in enumerate(batch, start=1):
        chunks.append(f"=== EXCERPT {i} ===")
        chunks.append(f"doc_id: {d.doc_id}")
        chunks.append(f"title: {d.title}")
        chunks.append(f"summary: {d.summary}")
        if d.char_excerpt:
            chunks.append(f"등장인물 section excerpt:\n{d.char_excerpt}")
        if d.setting_excerpt:
            chunks.append(f"설정 section excerpt:\n{d.setting_excerpt}")
        if d.body_excerpt:
            chunks.append(f"본문 excerpt:\n{d.body_excerpt}")
        chunks.append("")
    chunks.append(
        f"Return a JSON array of exactly {len(batch)} objects, one per "
        "excerpt above, in the same order. Output ONLY the array."
    )
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Subprocess call.
# ---------------------------------------------------------------------------


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _invoke_claude_cli(
    *,
    claude_path: str,
    model: str,
    user_message: str,
    timeout: float = SUBPROCESS_TIMEOUT,
) -> str:
    """Run ``claude --print --model <model>`` and return raw stdout.

    The system prompt is appended via ``--append-system-prompt`` so the
    model has the JSON contract loaded before reading the user message.
    Raises ``RuntimeError`` on non-zero exit; the caller decides whether
    to retry or abort.
    """
    cmd = [
        claude_path,
        "--print",
        "--model", model,
        "--append-system-prompt", SYSTEM_PROMPT,
        "--no-session-persistence",
        "--output-format", "text",
    ]
    log.debug("invoking claude CLI: %s", " ".join(cmd[:5] + ["<system>", "<no-session>"]))
    proc = subprocess.run(
        cmd,
        input=user_message,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:400]}"
        )
    return proc.stdout


def _parse_batch_response(
    raw: str, batch: List[_DocExcerpt]
) -> List[Dict[str, Any]]:
    """Extract the JSON array from raw stdout and pair with batch ids.

    Tolerates a few common LLM tics: leading/trailing prose, ```json
    fences, the array embedded inside a wider object. Returns an empty
    list on unrecoverable parse failure — the caller treats that as a
    batch-level failure and may retry.
    """
    text = raw.strip()
    if not text:
        return []

    # Strip ```json ... ``` fence if present.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("queries", "rows", "results", "data"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    # Last resort: regex-extract the first array-of-objects block.
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse JSON array from CLI output (len=%d). First 200 chars: %r",
                len(text), text[:200])
    return []


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def _excerpt_haystack(d: _DocExcerpt) -> str:
    """Concatenated text the validator considers 'inside the excerpt'."""
    parts = [d.title, d.summary]
    if d.char_excerpt:
        parts.append(d.char_excerpt)
    if d.setting_excerpt:
        parts.append(d.setting_excerpt)
    if d.body_excerpt:
        parts.append(d.body_excerpt)
    return "\n".join(parts)


def _validate_row(
    row: Dict[str, Any], excerpt: _DocExcerpt
) -> Optional[Dict[str, Any]]:
    """Coerce + ground-check one model-emitted row.

    Returns the cleaned row dict on success, or None if any constraint
    fails. Constraints (in order): non-empty query, valid answer_type,
    >= 1 keyword, every keyword is a verbatim substring of the excerpt
    haystack.
    """
    query = str(row.get("query") or "").strip()
    if not query:
        return None
    raw_kws = row.get("expected_section_keywords") or []
    if not isinstance(raw_kws, list):
        return None
    keywords = [str(k).strip() for k in raw_kws if str(k).strip()]
    if not keywords:
        return None
    haystack = _excerpt_haystack(excerpt)
    grounded = [k for k in keywords if k in haystack]
    if len(grounded) < max(1, len(keywords) // 2):
        # Reject when fewer than half the keywords were actually verbatim
        # in the excerpt. Catches paraphrase drift without being so
        # strict that one stray word kills an otherwise-good row.
        return None

    answer_type = str(row.get("answer_type") or "").strip()
    if answer_type not in VALID_ANSWER_TYPES:
        answer_type = "summary_plot"  # safe default
    difficulty = str(row.get("difficulty") or "").strip()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = DIFFICULTY_BY_TYPE.get(answer_type, "medium")

    return {
        "query": query,
        "expected_doc_ids": [excerpt.doc_id],
        "expected_section_keywords": grounded[:5],
        "answer_type": answer_type,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Main generator entry point.
# ---------------------------------------------------------------------------


def generate_via_claude_cli(
    docs: List[Dict[str, Any]],
    *,
    target: int,
    model: str = "opus",
    seed: int = 42,
    claude_cli_path: Optional[str] = None,
    log_path: Optional[Path] = None,
    docs_per_batch: int = DOCS_PER_BATCH,
) -> List[Dict[str, Any]]:
    """Iterate the corpus in batches, call the CLI, validate, accumulate.

    Stops as soon as ``len(rows) >= target`` or all docs exhausted.
    Each batch failure is logged and skipped; the caller doesn't have
    to micro-manage retries beyond ``MAX_RETRIES_PER_BATCH`` per batch.

    ``log_path`` is optional — when set, every CLI call (input batch +
    raw stdout + validated count) is appended as one JSON line so the
    operator can audit cost / quality after the fact. Recommended to set.
    """
    import random

    claude_path = find_claude_cli(claude_cli_path)
    rng = random.Random(seed)
    excerpts: List[_DocExcerpt] = []
    for d in docs:
        e = _doc_to_excerpt(d)
        if e is not None:
            excerpts.append(e)
    rng.shuffle(excerpts)

    log.info(
        "claude-cli generator: %d docs → target=%d, batch=%d, model=%s, cli=%s",
        len(excerpts), target, docs_per_batch, model, claude_path,
    )

    rows: List[Dict[str, Any]] = []
    log_fp = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("w", encoding="utf-8")

    try:
        for batch_idx, batch in enumerate(_chunked(excerpts, docs_per_batch)):
            if len(rows) >= target:
                break
            user_msg = _format_batch_prompt(batch)
            attempt = 0
            parsed_rows: List[Dict[str, Any]] = []
            raw_response = ""
            err: Optional[str] = None
            while attempt < MAX_RETRIES_PER_BATCH:
                attempt += 1
                t0 = time.perf_counter()
                try:
                    raw_response = _invoke_claude_cli(
                        claude_path=claude_path, model=model, user_message=user_msg,
                    )
                    parsed_rows = _parse_batch_response(raw_response, batch)
                    err = None
                    if parsed_rows:
                        break
                    log.warning(
                        "Batch %d attempt %d: empty parse — retrying.",
                        batch_idx + 1, attempt,
                    )
                except subprocess.TimeoutExpired:
                    err = f"timeout after {SUBPROCESS_TIMEOUT}s"
                    log.warning("Batch %d attempt %d: %s", batch_idx + 1, attempt, err)
                except Exception as ex:
                    err = f"{type(ex).__name__}: {ex}"
                    log.warning("Batch %d attempt %d: %s", batch_idx + 1, attempt, err)
                t1 = time.perf_counter()

            t_total = time.perf_counter() - t0
            accepted_in_batch = 0
            for excerpt, row in zip(batch, parsed_rows):
                cleaned = _validate_row(row or {}, excerpt)
                if cleaned is None:
                    continue
                cleaned["language"] = "ko"
                cleaned["tags"] = [
                    "anime", "silver", "synthetic",
                    cleaned["answer_type"], f"claude-cli:{model}",
                ]
                cleaned["id"] = f"anime-silver-llm-{len(rows) + 1:04d}"
                rows.append(cleaned)
                accepted_in_batch += 1
                if len(rows) >= target:
                    break

            log.info(
                "Batch %d: %d/%d accepted (cumulative %d/%d, last call %.1fs)",
                batch_idx + 1, accepted_in_batch, len(batch),
                len(rows), target, t_total,
            )
            if log_fp is not None:
                log_fp.write(json.dumps({
                    "batch": batch_idx + 1,
                    "doc_ids": [e.doc_id for e in batch],
                    "raw_response_chars": len(raw_response or ""),
                    "raw_response_preview": (raw_response or "")[:200],
                    "parsed_rows": len(parsed_rows),
                    "accepted": accepted_in_batch,
                    "elapsed_s": round(t_total, 2),
                    "error": err,
                }, ensure_ascii=False) + "\n")
                log_fp.flush()
    finally:
        if log_fp is not None:
            log_fp.close()

    log.info(
        "claude-cli generator finished: %d rows accepted (target=%d).",
        len(rows), target,
    )
    return rows


def _chunked(items: List[Any], n: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]
