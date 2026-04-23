"""Validate + stratify the raw enterprise eval set.

Consumes ``rag_enterprise_kr_raw.jsonl`` (output of
``generate_enterprise_queries``) and produces:

  * ``rag_enterprise_kr.jsonl`` — ~200 rows with ``difficulty`` labels
    assigned from bge-m3 retrieval behaviour on the enterprise-only
    corpus, dedup'd, capped at 4 queries per doc, stratified against
    a ``easy 70 / medium 80 / hard 40 / impossible 10`` target.
  * ``validation_report.json`` — bucket counts, per-category counts,
    drop reasons, imbalance warnings, optional Claude-judge ratings.

Usage (from ai-worker/)::

    python -m scripts.dataset.validate_enterprise_dataset \\
        --in  eval/datasets/rag_enterprise_kr_raw.jsonl \\
        --corpus fixtures/enterprise_corpus_kr/index.jsonl \\
        --out eval/datasets/rag_enterprise_kr.jsonl \\
        --target easy:70,medium:80,hard:40,impossible:10 \\
        --use-claude-judge

Design notes
------------
* **Self-contained retrieval.** We build an in-memory cosine ranker
  over the corpus (bge-m3 passages + queries) rather than standing up
  the full Postgres + FAISS stack. The scores land in the same regime
  as production retrieval because the embedder is identical and the
  chunker mirrors the ingest's greedy + windowing passes.
* **Difficulty is assigned, not inherited.** Any ``difficulty`` field
  on the input row is ignored. We rank every row against the live
  corpus and assign a bucket based on ``(gold_doc_rank, top_score)``
  for answerable rows and ``top_score`` for unanswerable rows.
* **Don't pad.** If a bucket is short, we log the warning and ship the
  smaller dataset. Synthesizing filler rows would defeat the point of
  the stratification.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.dataset._common import (
    GenerationLog,
    RateLimiter,
    claude_json_call,
    configure_logging,
    load_anthropic_client,
    log_call,
    read_jsonl,
    stable_seed,
    write_jsonl,
)

log = logging.getLogger("scripts.dataset.validate_enterprise_dataset")


DIFFICULTIES = ("easy", "medium", "hard", "impossible")


# ---------------------------------------------------------------------------
# Chunking for in-memory retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chunk:
    doc_id: str
    category: str
    section: str
    order: int
    text: str


def _chunk_corpus(corpus: List[Dict[str, Any]]) -> List[_Chunk]:
    """Walk the corpus and emit ingest-equivalent chunks in one list.

    Reuses the production chunker so ``validate_*`` scores a query
    against chunk boundaries matching ``app.capabilities.rag.ingest``.
    """
    from app.capabilities.rag.chunker import (
        MAX_CH,
        MIN_CH,
        OVERLAP,
        greedy_chunk,
        window_by_chars,
    )

    out: List[_Chunk] = []
    for row in corpus:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        category = str(row.get("category") or "uncategorized")
        sections = row.get("sections") or {}
        if not isinstance(sections, dict):
            continue
        order_hint = row.get("section_order") or list(sections.keys())
        for section_name in order_hint:
            body = sections.get(section_name)
            if not isinstance(body, dict):
                continue
            source_chunks: List[str] = []
            if isinstance(body.get("chunks"), list):
                source_chunks.extend(
                    str(c) for c in body["chunks"] if isinstance(c, (str, int, float))
                )
            if not source_chunks and isinstance(body.get("text"), str) and body["text"].strip():
                source_chunks.extend(greedy_chunk(body["text"]))
            if not source_chunks:
                continue
            windowed = window_by_chars(
                source_chunks,
                target=MAX_CH,
                min_chars=MIN_CH,
                max_chars=MAX_CH,
                overlap=OVERLAP,
            )
            for order, text in enumerate(windowed):
                text = text.strip()
                if not text:
                    continue
                out.append(_Chunk(
                    doc_id=doc_id,
                    category=category,
                    section=str(section_name),
                    order=order,
                    text=text,
                ))
    return out


# ---------------------------------------------------------------------------
# In-memory retriever
# ---------------------------------------------------------------------------


@dataclass
class _RetrievalHit:
    doc_id: str
    score: float
    rank: int


class _InMemoryRetriever:
    """Tiny self-contained cosine retriever for validation only."""

    def __init__(self, chunks: List[_Chunk], embedder: Any) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._matrix = None

    def build(self) -> None:
        import numpy as np

        if not self._chunks:
            self._matrix = np.zeros((0, 0), dtype="float32")
            return
        texts = [c.text for c in self._chunks]
        self._matrix = self._embedder.embed_passages(texts)
        log.info("Validator index ready: %d chunks, dim=%d",
                 self._matrix.shape[0], self._matrix.shape[1])

    def rank_docs(self, query: str, top_docs: int = 10) -> List[_RetrievalHit]:
        import numpy as np

        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q_vec = self._embedder.embed_queries([query])
        sims = (self._matrix @ q_vec.T).reshape(-1)
        # Aggregate chunk-level scores to doc-level (max).
        doc_scores: Dict[str, float] = {}
        for i, c in enumerate(self._chunks):
            s = float(sims[i])
            if s > doc_scores.get(c.doc_id, -1e9):
                doc_scores[c.doc_id] = s
        ordered = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            _RetrievalHit(doc_id=doc_id, score=score, rank=rank + 1)
            for rank, (doc_id, score) in enumerate(ordered[:top_docs])
        ]


# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifficultyThresholds:
    easy_score: float = 0.70
    medium_score: float = 0.55
    impossible_score: float = 0.55
    hard_max_rank: int = 10


def _assign_difficulty(
    *,
    question_type: str,
    expected_doc_ids: List[str],
    hits: List[_RetrievalHit],
    thresholds: DifficultyThresholds,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return ``(difficulty, metadata)``. ``None`` means drop the row."""
    top = hits[0] if hits else None
    top_score = top.score if top is not None else 0.0
    meta: Dict[str, Any] = {
        "retrieval_top_score": round(top_score, 4),
        "retrieval_top_doc_id": top.doc_id if top else None,
    }
    if question_type == "unanswerable":
        if top_score < thresholds.impossible_score:
            return "impossible", meta
        # Retriever is confidently matching something — still kind of
        # "hard" for the RAG to refuse, but not truly impossible.
        return "hard", {**meta, "note": "unanswerable-but-retriever-matched"}

    if not expected_doc_ids:
        return None, {**meta, "drop_reason": "no_expected_doc_ids"}

    gold = set(str(d) for d in expected_doc_ids)
    gold_rank = None
    gold_score = None
    for h in hits:
        if h.doc_id in gold:
            gold_rank = h.rank
            gold_score = h.score
            break
    meta["gold_rank"] = gold_rank
    meta["gold_score"] = round(gold_score, 4) if gold_score is not None else None

    if gold_rank is None or gold_rank > thresholds.hard_max_rank:
        return None, {**meta, "drop_reason": "gold_not_in_topk"}
    if gold_rank == 1 and (gold_score or 0.0) >= thresholds.easy_score:
        return "easy", meta
    if gold_rank <= 3 and (gold_score or 0.0) >= thresholds.medium_score:
        return "medium", meta
    return "hard", meta


# ---------------------------------------------------------------------------
# Dedup + stratification
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _dedup_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = _normalize_for_dedup(str(row.get("query", "")))
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def _apply_per_doc_cap(
    rows: List[Dict[str, Any]], *, cap: int, seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """At most ``cap`` rows per expected doc. Unanswerable rows bypass
    the cap since they don't have a gold doc.
    """
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unans: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("question_type") == "unanswerable":
            unans.append(row)
            continue
        gold = row.get("expected_doc_ids") or []
        if not gold:
            unans.append(row)
            continue
        by_doc[str(gold[0])].append(row)
    rng = random.Random(seed)
    kept: List[Dict[str, Any]] = list(unans)
    dropped = 0
    for doc_id, rows_for_doc in by_doc.items():
        if len(rows_for_doc) <= cap:
            kept.extend(rows_for_doc)
            continue
        rng.shuffle(rows_for_doc)
        kept.extend(rows_for_doc[:cap])
        dropped += len(rows_for_doc) - cap
    return kept, dropped


def _parse_target(spec: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--target entry must be name:count, got {chunk!r}")
        name, value = chunk.split(":", 1)
        name = name.strip()
        if name not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty {name!r}")
        out[name] = int(value)
    for d in DIFFICULTIES:
        out.setdefault(d, 0)
    return out


def _stratify(
    rows: List[Dict[str, Any]], *, target: Dict[str, int], seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = str(row.get("difficulty", "")).lower()
        if d in target:
            by_bucket[d].append(row)
    rng = random.Random(seed)
    kept: List[Dict[str, Any]] = []
    achieved: Dict[str, int] = {}
    shortfalls: Dict[str, int] = {}
    for difficulty in DIFFICULTIES:
        pool = by_bucket.get(difficulty, [])
        want = target[difficulty]
        rng.shuffle(pool)
        taken = pool[:want]
        kept.extend(taken)
        achieved[difficulty] = len(taken)
        shortfall = want - len(taken)
        if shortfall > 0:
            shortfalls[difficulty] = shortfall
    return kept, achieved, shortfalls


# ---------------------------------------------------------------------------
# Claude-as-judge (optional)
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = (
    "You are a meticulous Korean-reading evaluator. Given ONE "
    "evaluation query and the answer-hint it is paired with, rate (a) "
    "naturalness of the Korean phrasing and (b) consistency of the "
    "answer hint with the query, each from 1 (poor) to 3 (excellent). "
    "Respond with a JSON object only."
)


def _judge_prompt(row: Dict[str, Any]) -> str:
    q = row.get("query", "")
    hint = row.get("expected_answer_hint") or ""
    evidence = row.get("evidence_chunk") or ""
    return (
        f"Query: {q}\n"
        f"Answer hint: {hint}\n"
        f"Evidence: {evidence[:400]}\n\n"
        f"Return JSON:\n"
        f'{{"naturalness": 1|2|3, "consistency": 1|2|3, "note": "short comment"}}'
    )


def _run_claude_judge(
    *,
    rows: List[Dict[str, Any]],
    model: str,
    rate_per_sec: float,
    gen_log_path: Path,
    sample_cap: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    client = load_anthropic_client()
    limiter = RateLimiter(rate_per_sec)
    gen_log = GenerationLog(gen_log_path)
    rng = random.Random(seed)

    pool = list(rows)
    rng.shuffle(pool)
    if sample_cap is not None:
        pool = pool[:sample_cap]

    ratings: List[Dict[str, Any]] = []
    for row in pool:
        limiter.wait()
        try:
            with log_call(
                gen_log,
                script="validate_enterprise_dataset.judge",
                provider="claude",
                model=model,
                seed=stable_seed(row.get("query", ""), "judge"),
            ) as slot:
                raw = claude_json_call(
                    client,
                    model=model,
                    system=_JUDGE_SYSTEM,
                    user=_judge_prompt(row),
                    max_tokens=256,
                    temperature=0.0,
                )
                usage = raw.pop("_usage", {})
                slot["prompt_tokens"] = usage.get("input_tokens")
                slot["completion_tokens"] = usage.get("output_tokens")
            ratings.append({
                "query": row.get("query"),
                "difficulty": row.get("difficulty"),
                "naturalness": int(raw.get("naturalness", 0) or 0),
                "consistency": int(raw.get("consistency", 0) or 0),
                "note": str(raw.get("note", ""))[:200],
            })
        except Exception as ex:  # noqa: BLE001
            log.warning("Judge call failed: %s", ex)
    if not ratings:
        return {"ratings": [], "summary": {}}

    def _avg(key: str) -> float:
        vals = [r.get(key, 0) for r in ratings if r.get(key)]
        return round(sum(vals) / max(1, len(vals)), 3)

    def _good_rate(key: str) -> float:
        if not ratings:
            return 0.0
        good = sum(1 for r in ratings if int(r.get(key, 0) or 0) >= 2)
        return round(good / len(ratings), 3)

    summary = {
        "ratings_count": len(ratings),
        "mean_naturalness": _avg("naturalness"),
        "mean_consistency": _avg("consistency"),
        "good_naturalness_rate": _good_rate("naturalness"),
        "good_consistency_rate": _good_rate("consistency"),
    }
    log.info("Claude judge: %s", summary)
    return {"ratings": ratings, "summary": summary}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_embedder(model_name: str) -> Any:
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=model_name)


def validate(
    *,
    raw_path: Path,
    corpus_path: Path,
    out_path: Path,
    report_path: Path,
    target: Dict[str, int],
    thresholds: DifficultyThresholds,
    per_doc_cap: int,
    embed_model: str,
    judge_enabled: bool,
    judge_model: str,
    judge_sample_cap: Optional[int],
    rate_per_sec: float,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    raw_rows = read_jsonl(raw_path)
    if not raw_rows:
        raise SystemExit(f"raw input is empty: {raw_path}")
    corpus = read_jsonl(corpus_path)
    if not corpus:
        raise SystemExit(f"corpus is empty: {corpus_path}")
    log.info("Loaded raw_queries=%d corpus=%d", len(raw_rows), len(corpus))

    chunks = _chunk_corpus(corpus)
    log.info("Corpus -> %d chunks for validation retrieval", len(chunks))

    if dry_run:
        log.info("[dry-run] skipping retrieval; writing no output")
        return {"dry_run": True, "raw_rows": len(raw_rows)}

    embedder = _load_embedder(embed_model)
    retriever = _InMemoryRetriever(chunks, embedder)
    retriever.build()

    labeled: List[Dict[str, Any]] = []
    dropped_counts: Counter = Counter()

    for row in raw_rows:
        query = str(row.get("query") or "").strip()
        if not query:
            dropped_counts["empty_query"] += 1
            continue
        question_type = str(row.get("question_type") or "").lower()
        expected_doc_ids = [str(d) for d in (row.get("expected_doc_ids") or [])]
        hits = retriever.rank_docs(query, top_docs=max(20, thresholds.hard_max_rank + 5))
        difficulty, meta = _assign_difficulty(
            question_type=question_type,
            expected_doc_ids=expected_doc_ids,
            hits=hits,
            thresholds=thresholds,
        )
        if difficulty is None:
            reason = meta.get("drop_reason", "no_difficulty")
            dropped_counts[reason] += 1
            continue
        labeled.append({
            **row,
            "difficulty": difficulty,
            "retrieval_meta": meta,
        })

    log.info("Labeled %d rows; dropped by bucket assignment: %s",
             len(labeled), dict(dropped_counts))

    deduped, dup_dropped = _dedup_rows(labeled)
    capped, cap_dropped = _apply_per_doc_cap(deduped, cap=per_doc_cap, seed=seed)
    log.info("Dedup removed %d; per-doc cap removed %d",
             dup_dropped, cap_dropped)

    final_rows, achieved, shortfalls = _stratify(capped, target=target, seed=seed)

    cat_counts = Counter(str(r.get("category") or "uncategorized") for r in final_rows)
    diff_counts = Counter(str(r.get("difficulty")) for r in final_rows)
    log.info("Stratification achieved: %s  (target %s)",
             dict(diff_counts), target)
    log.info("Per-category counts: %s", dict(cat_counts))

    # ---- Tolerance warnings --------------------------------------------
    imbalance_warnings: List[str] = []
    for difficulty, want in target.items():
        got = achieved.get(difficulty, 0)
        tol = max(1, int(round(want * 0.20)))
        if abs(got - want) > tol:
            imbalance_warnings.append(
                f"bucket {difficulty}: got {got}, target {want} "
                f"(tolerance ±{tol})"
            )
    categories_present = sorted(cat_counts)
    if categories_present:
        avg = sum(cat_counts.values()) / len(categories_present)
        for cat in categories_present:
            if abs(cat_counts[cat] - avg) > avg * 0.25:
                imbalance_warnings.append(
                    f"category {cat}: {cat_counts[cat]} rows "
                    f"(avg {avg:.1f}; off by >25%)"
                )
    for w in imbalance_warnings:
        log.warning("[imbalance] %s", w)

    write_jsonl(
        out_path, final_rows,
        header=(
            f"Validated enterprise eval set\n"
            f"Raw input: {raw_path.name}  corpus: {corpus_path.name}\n"
            f"Stratification target: {target}\n"
            f"Achieved: {dict(diff_counts)}\n"
            f"Per-category counts: {dict(cat_counts)}"
        ),
    )
    log.info("Wrote %d rows to %s", len(final_rows), out_path)

    judge_result: Optional[Dict[str, Any]] = None
    if judge_enabled and final_rows:
        judge_result = _run_claude_judge(
            rows=final_rows,
            model=judge_model,
            rate_per_sec=rate_per_sec,
            gen_log_path=report_path.with_suffix(".judge.log.jsonl"),
            sample_cap=judge_sample_cap,
            seed=seed,
        )

    report = {
        "raw_rows": len(raw_rows),
        "labeled": len(labeled),
        "after_dedup": len(deduped),
        "after_per_doc_cap": len(capped),
        "kept": len(final_rows),
        "target": target,
        "achieved_by_difficulty": dict(diff_counts),
        "per_category_counts": dict(cat_counts),
        "dropped_by_reason": dict(dropped_counts),
        "dedup_dropped": dup_dropped,
        "per_doc_cap_dropped": cap_dropped,
        "imbalance_warnings": imbalance_warnings,
        "thresholds": {
            "easy_score": thresholds.easy_score,
            "medium_score": thresholds.medium_score,
            "impossible_score": thresholds.impossible_score,
            "hard_max_rank": thresholds.hard_max_rank,
        },
        "claude_judge": (judge_result.get("summary") if judge_result else None),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if judge_result and judge_result.get("ratings"):
        (report_path.with_suffix(".judge.jsonl")).write_text(
            "\n".join(
                json.dumps(r, ensure_ascii=False)
                for r in judge_result["ratings"]
            ) + "\n",
            encoding="utf-8",
        )
    log.info("Wrote validation_report -> %s", report_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="raw_path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="enterprise_corpus_kr/index.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None,
                        help="validation_report.json path (defaults to "
                             "sibling of --out)")
    parser.add_argument("--target", type=str,
                        default="easy:70,medium:80,hard:40,impossible:10")
    parser.add_argument("--per-doc-cap", type=int, default=4)
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--easy-score", type=float, default=0.70)
    parser.add_argument("--medium-score", type=float, default=0.55)
    parser.add_argument("--impossible-score", type=float, default=0.55)
    parser.add_argument("--hard-max-rank", type=int, default=10)
    parser.add_argument("--use-claude-judge", action="store_true")
    parser.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--judge-sample-cap", type=int, default=None)
    parser.add_argument("--rate-per-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        target = _parse_target(args.target)
    except ValueError as ex:
        parser.error(str(ex))
        return 2
    if args.report is None:
        args.report = args.out.parent / f"{args.out.stem}_validation_report.json"

    thresholds = DifficultyThresholds(
        easy_score=args.easy_score,
        medium_score=args.medium_score,
        impossible_score=args.impossible_score,
        hard_max_rank=args.hard_max_rank,
    )
    try:
        validate(
            raw_path=args.raw_path,
            corpus_path=args.corpus,
            out_path=args.out,
            report_path=args.report,
            target=target,
            thresholds=thresholds,
            per_doc_cap=args.per_doc_cap,
            embed_model=args.embed_model,
            judge_enabled=args.use_claude_judge,
            judge_model=args.judge_model,
            judge_sample_cap=args.judge_sample_cap,
            rate_per_sec=args.rate_per_sec,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("Enterprise dataset validation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
