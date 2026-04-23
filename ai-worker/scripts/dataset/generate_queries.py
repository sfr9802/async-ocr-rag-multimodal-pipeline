"""Generate retrieval evaluation queries from a corpus of source documents.

Calls Claude Sonnet 4.6 once per document and asks it for ``--queries-per-doc``
queries spread across an explicit difficulty target (e.g. ``easy:10,
medium:15,hard:5``). Each generated row carries the gold ``doc_id`` so a
later validation pass can run them against the live retriever and decide
whether to keep, downgrade, or drop the row.

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.generate_queries \\
        --corpus fixtures/anime_kr.jsonl \\
        --out    eval/datasets/rag_anime_extended_kr_raw.jsonl \\
        --queries-per-doc 3 \\
        --difficulty-target easy:10,medium:15,hard:5

Design notes
------------
* **Resumable.** The output JSONL is keyed by ``query`` so re-running the
  script will skip queries already on disk. A crash mid-corpus is safe.
* **Deterministic seed per doc.** ``stable_seed(doc_id, 'queries')`` is
  passed into the prompt so two runs of the same corpus produce
  comparable phrasings and difficulty distributions.
* **No live retriever needed.** Validation against the live FAISS index
  is a separate step (``scripts.dataset.validate_dataset``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
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
)

log = logging.getLogger("scripts.dataset.generate_queries")


_VALID_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class DifficultyTarget:
    """Per-difficulty target counts across the entire output dataset."""
    easy: int
    medium: int
    hard: int

    @property
    def total(self) -> int:
        return self.easy + self.medium + self.hard

    def per_doc_distribution(self, doc_count: int, queries_per_doc: int) -> Dict[str, int]:
        """Return how many of each difficulty each document should produce.

        We aim for the global target while distributing evenly across
        documents — the model picks the topic, but our prompt anchors
        the difficulty mix.
        """
        per_doc = {
            "easy": max(0, round(self.easy / max(1, doc_count))),
            "medium": max(0, round(self.medium / max(1, doc_count))),
            "hard": max(0, round(self.hard / max(1, doc_count))),
        }
        # Top up / trim so the total matches queries_per_doc exactly.
        delta = queries_per_doc - sum(per_doc.values())
        if delta > 0:
            # Prefer adding 'medium' since it's the bulk of most evals.
            per_doc["medium"] += delta
        elif delta < 0:
            # Remove from the largest bucket first.
            for k in sorted(per_doc, key=per_doc.get, reverse=True):
                if per_doc[k] <= 0:
                    continue
                take = min(-delta, per_doc[k])
                per_doc[k] -= take
                delta += take
                if delta == 0:
                    break
        return per_doc


def _parse_difficulty_target(spec: str) -> DifficultyTarget:
    parts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--difficulty-target entries must be name:count, got {chunk!r}"
            )
        name, count = chunk.split(":", 1)
        name = name.strip().lower()
        if name not in parts:
            raise ValueError(
                f"Unknown difficulty {name!r}; allowed: {sorted(parts)}"
            )
        try:
            parts[name] = int(count)
        except ValueError as ex:
            raise ValueError(
                f"--difficulty-target {name!r} count must be int, got {count!r}"
            ) from ex
    return DifficultyTarget(**parts)


_SYSTEM_PROMPT = (
    "You are a retrieval-evaluation query author. Given ONE source "
    "document, produce N realistic search queries a user might type "
    "if they wanted to find this document. Difficulty levels:\n"
    "  easy   : single fact directly stated; one section.\n"
    "  medium : requires combining 2 facts within the document.\n"
    "  hard   : paraphrased or implicit; the user knows roughly what "
    "they want but uses none of the document's exact wording.\n\n"
    "Respond ONLY with a JSON object. Do NOT wrap it in code fences."
)


_SCHEMA_HINT = (
    '{"queries": ['
    '{"query": string, "difficulty": "easy"|"medium"|"hard", '
    '"expected_keywords": [string, ...], "notes": string}'
    "]}"
)


def _build_user_prompt(
    *,
    doc: Dict[str, Any],
    seed: int,
    distribution: Dict[str, int],
    language: str,
) -> str:
    sections = doc.get("sections") or {}
    section_text = []
    for name, payload in sections.items():
        if isinstance(payload, dict):
            chunks = payload.get("chunks") or []
            if chunks:
                section_text.append(
                    f"## {name}\n" + "\n".join(str(c) for c in chunks)
                )
            elif payload.get("text"):
                section_text.append(f"## {name}\n{payload.get('text')}")
            elif payload.get("list"):
                rendered = []
                for item in payload.get("list", []):
                    if isinstance(item, dict):
                        rendered.append(
                            f"- {item.get('name', '')}: {item.get('desc', '')}"
                        )
                section_text.append(f"## {name}\n" + "\n".join(rendered))

    body = "\n\n".join(section_text)[:6000]
    requested = ", ".join(f"{n} {k}" for k, n in distribution.items() if n > 0)
    lang_label = "Korean" if language == "ko" else "English" if language == "en" else language

    return (
        f"Document title: {doc.get('title') or doc.get('seed') or doc.get('doc_id')}\n"
        f"Document id: {doc.get('doc_id')}\n"
        f"Target language: {lang_label}\n"
        f"Seed (do not echo): {seed}\n\n"
        f"Document body:\n{body}\n\n"
        f"Produce queries with this distribution: {requested}.\n"
        f"Each query must be ONE sentence in {lang_label}. expected_keywords "
        f"are 2-4 substrings the answer text should contain to count as a "
        f"correct generation. notes is a one-phrase rationale (max 12 words).\n\n"
        f"Return JSON of this shape:\n{_SCHEMA_HINT}"
    )


def _validate_queries(
    raw: Dict[str, Any],
    *,
    doc_id: str,
    distribution: Dict[str, int],
) -> List[Dict[str, Any]]:
    queries = raw.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ClaudeResponseError("'queries' must be a non-empty list.")
    out: List[Dict[str, Any]] = []
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise ClaudeResponseError(f"queries[{i}] is not an object.")
        text = str(q.get("query", "")).strip()
        if not text:
            raise ClaudeResponseError(f"queries[{i}] missing 'query' text.")
        difficulty = str(q.get("difficulty", "medium")).strip().lower()
        if difficulty not in _VALID_DIFFICULTIES:
            difficulty = "medium"
        keywords = q.get("expected_keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()][:6]
        notes = str(q.get("notes", "")).strip()[:120]
        out.append({
            "query": text,
            "difficulty": difficulty,
            "expected_doc_ids": [doc_id],
            "expected_keywords": keywords,
            "notes": notes or None,
        })
    return out


def _rebalance_difficulty(
    rows: List[Dict[str, Any]],
    target: DifficultyTarget,
) -> List[Dict[str, Any]]:
    """Trim/keep rows so the global counts approach the requested target.

    The prompt steers per-doc distribution but Claude is not perfectly
    obedient — this pass enforces global totals by truncating each
    bucket to the target and dropping the excess. We keep the rows in
    the order they were generated so resumability stays predictable.
    """
    by_difficulty: Dict[str, List[Dict[str, Any]]] = {
        "easy": [], "medium": [], "hard": [],
    }
    for row in rows:
        d = row.get("difficulty", "medium")
        by_difficulty.setdefault(d, []).append(row)
    out: List[Dict[str, Any]] = []
    out.extend(by_difficulty["easy"][: target.easy])
    out.extend(by_difficulty["medium"][: target.medium])
    out.extend(by_difficulty["hard"][: target.hard])
    return out


def generate_queries(
    *,
    corpus_path: Path,
    out_path: Path,
    queries_per_doc: int,
    difficulty_target: DifficultyTarget,
    model: str,
    rate_per_sec: float,
    max_retries: int,
    language: str,
    dry_run: bool,
) -> int:
    docs = read_jsonl(corpus_path)
    if not docs:
        raise SystemExit(f"corpus is empty: {corpus_path}")
    log.info("Read %d documents from %s", len(docs), corpus_path)

    distribution = difficulty_target.per_doc_distribution(
        doc_count=len(docs), queries_per_doc=queries_per_doc,
    )
    log.info("Per-doc distribution: %s", distribution)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ResumableJsonlWriter(
        out_path,
        key_fn=lambda row: str(row.get("query", "")).strip(),
    )
    gen_log = GenerationLog(out_path.parent / f"{out_path.stem}_generation_log.jsonl")
    limiter = RateLimiter(rate_per_sec)

    client = None
    if not dry_run:
        client = load_anthropic_client()

    new_count = 0
    for doc in docs:
        doc_id = str(doc.get("doc_id") or doc.get("seed") or doc.get("title"))
        if not doc_id:
            continue
        seed = stable_seed(doc_id, "queries")

        if dry_run:
            log.info("[dry-run] would generate %d queries for %s", queries_per_doc, doc_id)
            continue

        rows: List[Dict[str, Any]] = []
        for attempt in range(1, max_retries + 1):
            limiter.wait()
            try:
                with log_call(
                    gen_log,
                    script="generate_queries",
                    provider="claude",
                    model=model,
                    seed=seed,
                    note=f"{doc_id} attempt={attempt}",
                ) as slot:
                    parsed = claude_json_call(
                        client,
                        model=model,
                        system=_SYSTEM_PROMPT,
                        user=_build_user_prompt(
                            doc=doc, seed=seed,
                            distribution=distribution, language=language,
                        ),
                        max_tokens=1500,
                        temperature=0.6,
                    )
                    usage = parsed.pop("_usage", {})
                    slot["prompt_tokens"] = usage.get("input_tokens")
                    slot["completion_tokens"] = usage.get("output_tokens")
                rows = _validate_queries(parsed, doc_id=doc_id, distribution=distribution)
                break
            except (ClaudeResponseError, Exception) as ex:  # noqa: BLE001
                log.warning(
                    "Query generation failed for %s (attempt %d/%d): %s",
                    doc_id, attempt, max_retries, ex,
                )

        for row in rows:
            if writer.has(row["query"]):
                continue
            writer.append(row)
            new_count += 1

    if not dry_run:
        # Re-balance the file once at the end to enforce the global
        # difficulty target. Reads everything back in, trims, rewrites.
        all_rows = read_jsonl(out_path)
        balanced = _rebalance_difficulty(all_rows, difficulty_target)
        if len(balanced) < len(all_rows):
            from scripts.dataset._common import write_jsonl
            write_jsonl(out_path, balanced, header=(
                f"Generated by scripts.dataset.generate_queries\n"
                f"corpus={corpus_path.name} "
                f"queries_per_doc={queries_per_doc} "
                f"target={difficulty_target}\n"
                f"NOTE: validate against the live retriever before using "
                f"as a benchmark."
            ))
            log.info(
                "Trimmed %d -> %d rows to match difficulty target %s",
                len(all_rows), len(balanced), difficulty_target,
            )

    counts = Counter(
        r.get("difficulty", "medium")
        for r in read_jsonl(out_path)
    )
    log.info(
        "Done. Wrote %d new rows, total on disk %s, target=%s",
        new_count, dict(counts), difficulty_target,
    )
    return new_count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries-per-doc", type=int, default=3)
    parser.add_argument("--difficulty-target", type=str, default="easy:10,medium:15,hard:5")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--rate-per-sec", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    target = _parse_difficulty_target(args.difficulty_target)

    try:
        generate_queries(
            corpus_path=args.corpus,
            out_path=args.out,
            queries_per_doc=args.queries_per_doc,
            difficulty_target=target,
            model=args.model,
            rate_per_sec=args.rate_per_sec,
            max_retries=args.max_retries,
            language=args.language,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("query generation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
