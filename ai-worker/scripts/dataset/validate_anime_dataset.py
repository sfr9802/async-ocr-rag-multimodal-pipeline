"""Validate and difficulty-label the raw anime query set.

Pipeline
--------
1. Build an in-memory bge-m3 + FAISS retriever from the sampled corpus,
   reusing ``eval.harness.offline_corpus.build_offline_rag_stack`` so the
   retrieval numbers are comparable to what the production worker would
   produce.
2. Run top_k=50 retrieval for every raw query; record hit@5, top1_score,
   rank_of_expected.
3. Assign a difficulty bucket:
     hit@5 and top1_score >= 0.80                -> easy
     hit@5 and top1_score in [0.50, 0.80)        -> medium
     not hit@5 and rank_of_expected <= 20        -> hard
     not hit@5 and rank_of_expected > 20 / None  -> FLAG
     unanswerable (empty expected_doc_ids)       -> impossible
4. Dedup on bge-m3 query-embedding cosine > 0.92.
5. Cap at <=2 queries per source_title.
6. Stratify-sample to ~200 rows targeting easy=70 / medium=80 / hard=40 /
   impossible=10. If a bucket falls short, keep what we have and log a
   warning — don't inflate with flagged rows.
7. Optional: Claude-as-judge naturalness score (1-5); <3 flagged.
8. Optional: baseline regression check by running the existing
   rag_sample(_kr) queries against a combined retriever.

Usage (from ai-worker/)::

    python -m scripts.dataset.validate_anime_dataset \\
        --queries eval/datasets/rag_anime_kr_raw.jsonl \\
        --corpus  fixtures/anime_corpus_kr.jsonl \\
        --out     eval/datasets/rag_anime_kr.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.dataset._common import (
    ClaudeResponseError,
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

log = logging.getLogger("scripts.dataset.validate_anime_dataset")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


DIFFICULTY_TARGETS = {
    "easy": 70,
    "medium": 80,
    "hard": 40,
    "impossible": 10,
}
_TOTAL_TARGET = sum(DIFFICULTY_TARGETS.values())


@dataclass
class ScoredQuery:
    raw: Dict[str, Any]
    hit_at_5: int
    top1_score: Optional[float]
    rank_of_expected: Optional[int]
    retrieved_doc_ids: List[str]
    retrieved_scores: List[float]
    difficulty: str
    flagged: List[str] = field(default_factory=list)
    naturalness_score: Optional[int] = None


# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------


def _classify(
    *,
    hit_at_5: int,
    top1_score: Optional[float],
    rank_of_expected: Optional[int],
    is_unanswerable: bool,
) -> str:
    if is_unanswerable:
        return "impossible"
    if hit_at_5 and top1_score is not None and top1_score >= 0.80:
        return "easy"
    if hit_at_5 and top1_score is not None and 0.50 <= top1_score < 0.80:
        return "medium"
    if hit_at_5:
        # hit but low score (top1_score < 0.50) — treat as medium, better
        # than silently dropping.
        return "medium"
    if rank_of_expected is not None and rank_of_expected <= 20:
        return "hard"
    return "FLAG"


# ---------------------------------------------------------------------------
# Retrieval + scoring
# ---------------------------------------------------------------------------


def _build_retriever(corpus_path: Path, *, top_k: int, index_dir: Path):
    """Lazy-import so the module parses without the heavyweight embedder."""
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.core.config import get_settings
    from eval.harness.offline_corpus import build_offline_rag_stack

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    retriever, _, info = build_offline_rag_stack(
        corpus_path,
        embedder=embedder,
        index_dir=index_dir,
        top_k=top_k,
    )
    return retriever, embedder, info


def _score_one(
    retriever,
    *,
    query: str,
    expected_doc_ids: List[str],
    top_k_full: int = 50,
) -> Tuple[int, Optional[float], Optional[int], List[str], List[float]]:
    """Run retrieval, compute hit@5 / top1_score / rank_of_expected."""
    # Retriever's top_k is fixed at construction; for this offline use we
    # built it at top_k=50 so results already come back at full depth.
    report = retriever.retrieve(query)
    doc_ids = [r.doc_id for r in report.results]
    scores = [float(r.score) for r in report.results]
    top1 = scores[0] if scores else None

    if not expected_doc_ids:
        # Unanswerable: by definition no doc_id should match; any hit is a
        # failure for the system but not our classification problem here.
        return 0, top1, None, doc_ids, scores

    expected_set = set(expected_doc_ids)
    hit_at_5 = 1 if any(d in expected_set for d in doc_ids[:5]) else 0
    rank = None
    for i, d in enumerate(doc_ids, start=1):
        if d in expected_set:
            rank = i
            break
    return hit_at_5, top1, rank, doc_ids, scores


# ---------------------------------------------------------------------------
# Dedup + coverage cap + stratification
# ---------------------------------------------------------------------------


def _dedup_by_embedding(
    items: List[ScoredQuery],
    *,
    embedder,
    threshold: float = 0.92,
) -> Tuple[List[ScoredQuery], int]:
    """Greedy near-duplicate dedup on bge-m3 query vectors."""
    if not items:
        return [], 0
    queries = [it.raw.get("query", "") for it in items]
    vecs = embedder.embed_queries(queries).astype(np.float32, copy=False)
    # Vectors are already L2-normalized by the embedder (FAISS IndexFlatIP
    # contract requires that), so inner product == cosine.
    kept: List[ScoredQuery] = []
    kept_vecs: List[np.ndarray] = []
    dropped = 0
    for item, vec in zip(items, vecs):
        if kept_vecs:
            sims = np.dot(np.stack(kept_vecs), vec)
            if float(sims.max()) >= threshold:
                dropped += 1
                continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept, dropped


def _cap_per_source(
    items: List[ScoredQuery],
    *,
    max_per_title: int = 2,
) -> Tuple[List[ScoredQuery], int]:
    """Allow at most ``max_per_title`` queries per source_title.

    Preserves order so the sorting the caller did earlier still applies.
    """
    counter: Dict[str, int] = {}
    kept: List[ScoredQuery] = []
    dropped = 0
    for it in items:
        key = str(it.raw.get("source_title") or "")
        c = counter.get(key, 0)
        if c >= max_per_title:
            dropped += 1
            continue
        counter[key] = c + 1
        kept.append(it)
    return kept, dropped


def _stratify(
    items: List[ScoredQuery],
    *,
    seed: int,
) -> Tuple[List[ScoredQuery], Dict[str, int], Dict[str, int]]:
    """Stratified sample ~200 rows matching ``DIFFICULTY_TARGETS``.

    Returns (selected, kept_per_bucket, available_per_bucket).
    """
    buckets: Dict[str, List[ScoredQuery]] = {k: [] for k in DIFFICULTY_TARGETS}
    for it in items:
        if it.difficulty in buckets:
            buckets[it.difficulty].append(it)

    selected: List[ScoredQuery] = []
    kept_counts: Dict[str, int] = {}
    avail_counts: Dict[str, int] = {}
    for bucket, target in DIFFICULTY_TARGETS.items():
        pool = buckets[bucket]
        avail_counts[bucket] = len(pool)
        # Deterministic sort: seed-salted stable key so the choice is
        # reproducible across reruns.
        pool_sorted = sorted(
            pool,
            key=lambda it: stable_seed(seed, "strat", it.raw.get("source_title", ""),
                                       it.difficulty),
        )
        keep = pool_sorted[:target]
        if len(keep) < target:
            log.warning(
                "Bucket %r: available=%d target=%d — keeping all available.",
                bucket, len(pool), target,
            )
        kept_counts[bucket] = len(keep)
        selected.extend(keep)
    return selected, kept_counts, avail_counts


# ---------------------------------------------------------------------------
# Optional: Claude-as-judge naturalness
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = (
    "You are a Korean-language reading-comprehension expert. Score how "
    "natural the given Korean question sounds on a 1-5 scale. "
    "5 = native-fluent and clearly phrased; 3 = understandable but awkward; "
    "1 = grammatically wrong or nonsensical. Return JSON only: "
    '{"score": <int 1-5>, "note": "<brief reason>"}.'
)


def _judge_naturalness(
    client,
    *,
    model: str,
    query: str,
) -> Optional[int]:
    user = f"Question:\n{query}\n\nReturn JSON with score and note."
    try:
        parsed = claude_json_call(
            client,
            model=model,
            system=_JUDGE_SYSTEM,
            user=user,
            max_tokens=128,
            temperature=0.2,
        )
    except ClaudeResponseError as ex:
        log.warning("Judge parse failed: %s", ex)
        return None
    try:
        score = int(parsed.get("score"))
    except Exception:  # noqa: BLE001
        return None
    if score < 1 or score > 5:
        return None
    return score


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _to_output_row(sq: ScoredQuery) -> Dict[str, Any]:
    raw = sq.raw
    return {
        "query": raw["query"],
        "question_type": raw.get("question_type"),
        "expected_doc_ids": raw.get("expected_doc_ids", []),
        "expected_keywords": raw.get("expected_keywords", []),
        "evidence_chunk": raw.get("evidence_chunk", ""),
        "source_title": raw.get("source_title"),
        "source_section": raw.get("source_section"),
        "notes": raw.get("notes"),
        "difficulty": sq.difficulty,
        "top1_score": sq.top1_score,
        "rank_of_expected": sq.rank_of_expected,
        "hit_at_5": sq.hit_at_5,
        "generator": raw.get("generator"),
        "seed": raw.get("seed"),
        "naturalness_score": sq.naturalness_score,
    }


def _to_flagged_row(sq: ScoredQuery) -> Dict[str, Any]:
    out = _to_output_row(sq)
    out["flag_reasons"] = sq.flagged
    out["retrieved_doc_ids_top3"] = sq.retrieved_doc_ids[:3]
    return out


# ---------------------------------------------------------------------------
# Baseline regression check
# ---------------------------------------------------------------------------


def _run_baseline_regression(
    *,
    corpus_path: Path,
    baseline_fixtures: List[Path],
    baseline_queries: List[Path],
    top_k: int,
    index_dir: Path,
) -> Dict[str, Any]:
    """Build a combined retriever and compute recall@5 on each baseline set."""
    # Concatenate anime corpus + baseline fixtures into one temp JSONL so
    # the retriever covers both sets of doc_ids at once.
    merged_path = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    with merged_path.open("w", encoding="utf-8") as out:
        for p in [corpus_path, *baseline_fixtures]:
            if not p.exists():
                log.warning("baseline fixture missing: %s", p)
                continue
            out.write(p.read_text(encoding="utf-8"))
    log.info("Baseline regression: merged %d fixtures -> %s", 1 + len(baseline_fixtures), merged_path)

    retriever, _, info = _build_retriever(merged_path, top_k=top_k, index_dir=index_dir)

    reports: Dict[str, Dict[str, Any]] = {}
    for qpath in baseline_queries:
        if not qpath.exists():
            log.warning("baseline queries missing: %s", qpath)
            continue
        rows = read_jsonl(qpath)
        total = 0
        hits = 0
        for row in rows:
            query = str(row.get("query") or "").strip()
            expected = row.get("expected_doc_ids") or []
            if not query or not expected:
                continue
            total += 1
            hit, _, _, _, _ = _score_one(
                retriever, query=query, expected_doc_ids=list(expected),
            )
            hits += hit
        reports[qpath.name] = {
            "rows": total,
            "hit_at_5": (hits / total) if total else None,
        }
    reports["_combined_index"] = {
        "chunk_count": info.chunk_count,
        "embedding_model": info.embedding_model,
        "index_version": info.index_version,
    }
    try:
        merged_path.unlink()
    except OSError:
        pass
    return reports


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run(
    *,
    queries_path: Path,
    corpus_path: Path,
    out_path: Path,
    flagged_path: Optional[Path],
    report_path: Path,
    top_k: int,
    dedup_threshold: float,
    max_per_title: int,
    seed: int,
    judge_model: Optional[str],
    baseline_fixtures: List[Path],
    baseline_queries: List[Path],
) -> int:
    raw_rows = read_jsonl(queries_path)
    if not raw_rows:
        log.error("No queries to validate: %s", queries_path)
        return 2
    log.info("Loaded %d raw queries from %s", len(raw_rows), queries_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "faiss"
        log.info("Building offline retriever over %s ...", corpus_path)
        retriever, embedder, info = _build_retriever(
            corpus_path, top_k=top_k, index_dir=index_dir,
        )
        log.info(
            "Index ready: docs implicit via chunks, chunks=%d dim=%d model=%s",
            info.chunk_count, info.dimension, info.embedding_model,
        )

        scored: List[ScoredQuery] = []
        for row in raw_rows:
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            expected = row.get("expected_doc_ids") or []
            is_unanswerable = (row.get("question_type") == "unanswerable") or (not expected)
            started = time.time()
            hit, top1, rank, docs, scores = _score_one(
                retriever,
                query=query,
                expected_doc_ids=list(expected),
                top_k_full=top_k,
            )
            elapsed_ms = (time.time() - started) * 1000.0

            difficulty = _classify(
                hit_at_5=hit,
                top1_score=top1,
                rank_of_expected=rank,
                is_unanswerable=is_unanswerable,
            )
            flagged = []
            if difficulty == "FLAG":
                flagged.append(
                    f"miss_beyond_top20 (rank={rank}, top1={top1})"
                )
            scored.append(ScoredQuery(
                raw=row,
                hit_at_5=hit,
                top1_score=top1,
                rank_of_expected=rank,
                retrieved_doc_ids=docs,
                retrieved_scores=scores,
                difficulty=difficulty,
                flagged=flagged,
            ))
            if len(scored) % 50 == 0:
                log.info("Scored %d / %d (last %.0fms)",
                         len(scored), len(raw_rows), elapsed_ms)

        pre_counts = _count_by_bucket(scored)
        log.info("Post-scoring bucket counts: %s", pre_counts)

        flagged_items = [s for s in scored if s.difficulty == "FLAG"]
        keep_items = [s for s in scored if s.difficulty != "FLAG"]

        keep_items, dup_dropped = _dedup_by_embedding(
            keep_items, embedder=embedder, threshold=dedup_threshold,
        )
        log.info("Embedding dedup dropped %d near-duplicates (>=%.2f)",
                 dup_dropped, dedup_threshold)

        keep_items, cap_dropped = _cap_per_source(
            keep_items, max_per_title=max_per_title,
        )
        log.info("Per-title cap dropped %d overflow rows", cap_dropped)

        if judge_model:
            log.info("Running Claude-as-judge naturalness scoring (%d rows)", len(keep_items))
            client = load_anthropic_client()
            limiter = RateLimiter(1.0)
            judge_log = GenerationLog(out_path.with_suffix(".judge_log.jsonl"))
            for sq in keep_items:
                limiter.wait()
                with log_call(
                    judge_log,
                    script="validate_anime_dataset.judge",
                    provider="claude",
                    model=judge_model,
                    seed=None,
                    note=sq.raw.get("source_title"),
                ) as slot:
                    score = _judge_naturalness(
                        client, model=judge_model, query=sq.raw["query"],
                    )
                    slot["prompt_tokens"] = None
                    slot["completion_tokens"] = None
                sq.naturalness_score = score
                if score is not None and score < 3:
                    sq.flagged.append(f"naturalness_low ({score})")

        selected, kept_counts, avail_counts = _stratify(keep_items, seed=seed)
        log.info(
            "Stratification kept=%s available=%s target=%s",
            kept_counts, avail_counts, DIFFICULTY_TARGETS,
        )

        # Keep rows flagged by the judge out of the final set; they still
        # land in the flagged file for manual review.
        final_selected: List[ScoredQuery] = []
        moved_by_judge: List[ScoredQuery] = []
        for sq in selected:
            if any(f.startswith("naturalness_low") for f in sq.flagged):
                moved_by_judge.append(sq)
            else:
                final_selected.append(sq)
        log.info("Moved %d rows to flagged for low naturalness",
                 len(moved_by_judge))

        # Write outputs.
        out_rows = [_to_output_row(sq) for sq in final_selected]
        final_count = write_jsonl(
            out_path, out_rows,
            header=(
                "Validated anime RAG eval queries. Auto-generated by "
                "scripts.dataset.validate_anime_dataset.\n"
                f"Source corpus: {corpus_path.name} (chunks={info.chunk_count})\n"
                f"Seed: {seed}\n"
                "Do NOT hand-edit; rerun validate_anime_dataset to regenerate."
            ),
        )
        log.info("Wrote %d final rows to %s", final_count, out_path)

        if flagged_path:
            flagged_rows = [_to_flagged_row(s) for s in flagged_items + moved_by_judge]
            flagged_count = write_jsonl(
                flagged_path, flagged_rows,
                header="Flagged rows: excluded from final dataset.",
            )
            log.info("Wrote %d flagged rows to %s", flagged_count, flagged_path)

        # Baseline regression (optional).
        baseline_report: Optional[Dict[str, Any]] = None
        if baseline_fixtures and baseline_queries:
            baseline_report = _run_baseline_regression(
                corpus_path=corpus_path,
                baseline_fixtures=baseline_fixtures,
                baseline_queries=baseline_queries,
                top_k=top_k,
                index_dir=Path(tmpdir) / "combined_faiss",
            )
            log.info("Baseline regression: %s", baseline_report)

    # Final validation report.
    report = {
        "generated_at": time.time(),
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
        "out_path": str(out_path),
        "total_raw": len(raw_rows),
        "scored_bucket_counts": pre_counts,
        "dedup_dropped": dup_dropped,
        "per_title_cap_dropped": cap_dropped,
        "stratification_kept": kept_counts,
        "stratification_available": avail_counts,
        "stratification_target": DIFFICULTY_TARGETS,
        "final_count": final_count,
        "flagged_count": len(flagged_items) + (len(moved_by_judge) if judge_model else 0),
        "judge_model": judge_model,
        "baseline_regression": baseline_report,
        "seed": seed,
        "embedding_model": info.embedding_model,
        "index_chunk_count": info.chunk_count,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote validation report to %s", report_path)
    return 0


def _count_by_bucket(items: List[ScoredQuery]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it.difficulty] = out.get(it.difficulty, 0) + 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=Path, required=True,
                        help="Raw query JSONL from generate_anime_queries.py.")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Sampled anime corpus JSONL "
                             "(fixtures/anime_corpus_kr.jsonl).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Final validated JSONL path.")
    parser.add_argument("--flagged-out", type=Path, default=None,
                        help="Flagged-rows JSONL path. "
                             "Defaults to <out>_flagged.jsonl.")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="validation_report.json path. "
                             "Defaults to alongside --out.")
    parser.add_argument("--top-k-full", type=int, default=50,
                        help="Full retrieval depth for rank_of_expected (default 50).")
    parser.add_argument("--dedup-threshold", type=float, default=0.92,
                        help="Query-embedding cosine dedup threshold.")
    parser.add_argument("--max-per-title", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Optional Claude model id for naturalness judging.")
    parser.add_argument("--baseline-fixture", type=Path, action="append", default=[],
                        help="Additional fixtures to merge for baseline regression. "
                             "Repeat the flag to add more (e.g. --baseline-fixture "
                             "fixtures/anime_sample.jsonl).")
    parser.add_argument("--baseline-queries", type=Path, action="append", default=[],
                        help="Eval JSONLs to replay against the merged retriever.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)

    flagged_path = (
        args.flagged_out
        or args.out.with_name(args.out.stem + "_flagged.jsonl")
    )
    report_path = (
        args.report_out
        or args.out.with_name("validation_report.json")
    )

    return run(
        queries_path=args.queries,
        corpus_path=args.corpus,
        out_path=args.out,
        flagged_path=flagged_path,
        report_path=report_path,
        top_k=args.top_k_full,
        dedup_threshold=args.dedup_threshold,
        max_per_title=args.max_per_title,
        seed=args.seed,
        judge_model=args.judge_model,
        baseline_fixtures=list(args.baseline_fixture),
        baseline_queries=list(args.baseline_queries),
    )


if __name__ == "__main__":
    sys.exit(main())
