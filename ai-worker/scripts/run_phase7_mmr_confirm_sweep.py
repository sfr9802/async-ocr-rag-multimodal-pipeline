"""Phase 7.5 — MMR confirm sweep CLI.

Drives the confirm sweep harness over a (candidate_k × mmr_lambda) grid,
all on the production-default ``retrieval-title-section`` index. Two
modes:

  * **Live** (``--live``): pre-computes a wide candidate pool per query
    once (max(grid candidate_k), use_mmr=False), then for each grid
    point applies post-hoc MMR over the cached pool. This avoids
    re-embedding 550 queries × 15 variants — embed once, MMR many
    times. Saves ~25 minutes vs the naive loop.

  * **Replay** (default): consumes pre-computed retrieval JSONL files
    (one wide pool per query in ``--candidate-pool-results``, plus the
    baseline JSONLs from the prior phase).

Outputs (under ``--report-dir``):

  - ``confirm_sweep_results.jsonl``       (one row per candidate)
  - ``confirm_sweep_summary.json``
  - ``confirm_sweep_report.md``
  - ``best_config.confirmed.json``
  - ``best_config.confirmed.env``
  - ``manifest.json``
  - per-variant retrieval JSONL: ``retrieval_<variant>_gold.jsonl`` /
    ``retrieval_<variant>_silver.jsonl`` (only in live mode)

The CLI deliberately does NOT touch production retrieval code. The env
snippet is a paste-target, the JSON is a reviewer artefact, and any
config rollout still goes through the normal config-change PR review.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.harness.phase7_human_gold_tune import (
    HUMAN_FOCUS_DISCLAIMER,
    GoldSeedDataset,
    GoldSummary,
    RetrievedDoc,
    SilverDataset,
    SilverSummary,
    evaluate_gold,
    evaluate_silver,
    load_human_gold_seed_50,
    load_llm_silver_500,
    summarize_gold,
    summarize_silver,
)
from eval.harness.phase7_mmr_confirm_sweep import (
    DEFAULT_CANDIDATE_K_GRID,
    DEFAULT_MMR_LAMBDA_GRID,
    DEFAULT_TOP_K,
    PRODUCTION_INDEX_CACHE_DIR,
    PRODUCTION_RAG_CHUNKS_PATH,
    PROMOTION_TARGET_CLARIFICATION,
    SECTION_HIT_CAVEAT,
    CandidateScore,
    ConfirmSweepResult,
    SweepCandidate,
    apply_variant_to_candidates,
    make_confirm_sweep_grid,
    select_confirmed_best,
    write_confirm_sweep_report_md,
    write_confirm_sweep_results_jsonl,
    write_confirm_sweep_summary_json,
    write_confirmed_best_config_env,
    write_confirmed_best_config_json,
)
from scripts.phase7_human_gold_tune import (
    LiveResources,
    RetrievalResult,
    VariantSpec,
    _ensure_chunk_metadata,
    _ensure_embedder,
    _ensure_retriever,
    read_retrieval_jsonl,
    write_retrieval_jsonl,
)


log = logging.getLogger("scripts.run_phase7_mmr_confirm_sweep")


# ---------------------------------------------------------------------------
# Live precomputation: wide pool per query, no MMR.
# ---------------------------------------------------------------------------


def precompute_candidate_pool(
    *,
    queries: Sequence[Tuple[str, str]],
    cache_dir: Path,
    rag_chunks_path: Path,
    pool_size: int,
    resources: Optional[LiveResources] = None,
    embedding_model: str = "BAAI/bge-m3",
    max_seq_length: int = 512,
) -> List[RetrievalResult]:
    """Run the retriever once per query at ``candidate_k=pool_size``, no MMR.

    Returns a list of :class:`RetrievalResult`, one per query, each
    holding up to ``pool_size`` ranked docs. Variants in the sweep then
    apply post-hoc MMR on top of these pools.
    """
    res = resources or LiveResources(
        embedding_model=embedding_model,
        max_seq_length=int(max_seq_length),
    )
    chunk_meta, doc_titles = _ensure_chunk_metadata(res, rag_chunks_path)
    retriever = _ensure_retriever(
        res, cache_dir, default_top_k=int(pool_size),
    )
    # Force the retriever into a wide-pool, no-MMR config; this is the
    # only retrieve() call we'll ever make against this query, so the
    # pool needs to be at least max(grid candidate_k).
    retriever._top_k = int(pool_size)
    retriever._candidate_k = int(pool_size)
    retriever._use_mmr = False
    retriever._mmr_lambda = 0.7  # ignored when use_mmr=False

    rows: List[RetrievalResult] = []
    for i, (qid, query) in enumerate(queries, start=1):
        t0 = time.perf_counter()
        report = retriever.retrieve(query)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        docs: List[RetrievedDoc] = []
        for rank, c in enumerate(report.results[: int(pool_size)], start=1):
            cm = chunk_meta.get(c.chunk_id)
            title = (cm.title if cm else "") or doc_titles.get(c.doc_id, "")
            section_path: Tuple[str, ...] = ()
            if cm:
                section_path = cm.section_path
            elif c.section:
                section_path = tuple(
                    s.strip() for s in c.section.split(">") if s.strip()
                )
            docs.append(RetrievedDoc(
                rank=rank,
                chunk_id=c.chunk_id,
                page_id=c.doc_id,
                title=title,
                section_path=section_path,
                score=float(c.score) if c.score is not None else None,
            ))
        rows.append(RetrievalResult(
            variant="_pool_wide",
            query_id=qid,
            query=query,
            elapsed_ms=elapsed_ms,
            docs=tuple(docs),
        ))
        if i % 50 == 0:
            log.info("pool: %d/%d queries embedded", i, len(queries))
    return rows


def slice_pool_for_variant(
    *,
    spec: SweepCandidate,
    pool_rows: Sequence[RetrievalResult],
) -> List[RetrievalResult]:
    """Apply spec.candidate_k + spec.use_mmr + spec.mmr_lambda to each pool row.

    The caller can then feed the resulting :class:`RetrievalResult`
    list straight into ``evaluate_gold`` / ``evaluate_silver``.
    """
    out: List[RetrievalResult] = []
    for r in pool_rows:
        sliced = apply_variant_to_candidates(
            r.docs,
            candidate_k=spec.candidate_k,
            use_mmr=spec.use_mmr,
            mmr_lambda=spec.mmr_lambda,
            top_k=spec.top_k,
        )
        out.append(RetrievalResult(
            variant=spec.name,
            query_id=r.query_id,
            query=r.query,
            elapsed_ms=r.elapsed_ms,
            docs=tuple(sliced),
        ))
    return out


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------


def evaluate_variant_summaries(
    *,
    spec: SweepCandidate,
    gold: GoldSeedDataset,
    silver: SilverDataset,
    gold_retrievals: Sequence[RetrievalResult],
    silver_retrievals: Sequence[RetrievalResult],
) -> Tuple[GoldSummary, SilverSummary]:
    """Score one variant against gold + silver."""
    gold_map: Dict[str, List[RetrievedDoc]] = {
        r.query_id: list(r.docs) for r in gold_retrievals
    }
    silver_map: Dict[str, List[RetrievedDoc]] = {
        r.query_id: list(r.docs) for r in silver_retrievals
    }
    gold_rows = evaluate_gold(gold, gold_map)
    silver_rows = evaluate_silver(silver, silver_map)
    return summarize_gold(gold_rows), summarize_silver(silver_rows)


# ---------------------------------------------------------------------------
# CLI argparse
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.5 MMR confirm sweep. Tests retrieval CONFIG changes "
        "(candidate_k × mmr_lambda) on the production-default "
        "retrieval-title-section index — NOT another embedding-text "
        "variant promotion."
    ))
    p.add_argument("--gold-path", type=Path, required=True)
    p.add_argument("--silver-path", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--baseline-name", default="baseline_retrieval_title_section_top10")
    p.add_argument(
        "--baseline-gold-results", type=Path, required=True,
        help="JSONL of baseline retrieval results on gold (one row per query).",
    )
    p.add_argument(
        "--baseline-silver-results", type=Path, required=True,
        help="JSONL of baseline retrieval results on silver.",
    )
    p.add_argument(
        "--previous-best-name", default="cand_top10_mmr_lambda07",
        help=(
            "Name of the Phase 7.x first-pass best so the report can "
            "show baseline vs previous best vs confirmed best."
        ),
    )
    p.add_argument(
        "--previous-best-gold-results", type=Path, default=None,
        help=(
            "Optional: previous-best retrieval JSONL on gold; the report "
            "uses it to populate the 'previous best' column."
        ),
    )
    p.add_argument(
        "--previous-best-silver-results", type=Path, default=None,
    )

    # Live mode
    p.add_argument(
        "--live", action="store_true",
        help="Run the wide-pool retrieval pass once per query.",
    )
    p.add_argument(
        "--index-root", type=Path, default=None,
        help=(
            "FAISS index cache root. Required when --live. The cache "
            "dir resolved against this is the production-default "
            "retrieval-title-section variant."
        ),
    )
    p.add_argument(
        "--rag-chunks-root", type=Path, default=None,
        help=(
            "Root the variant's rag_chunks_*.jsonl path is resolved "
            "against (defaults to the working directory)."
        ),
    )
    p.add_argument("--embedding-model", default="BAAI/bge-m3")
    p.add_argument("--max-seq-length", type=int, default=512)

    # Replay mode
    p.add_argument(
        "--gold-pool-results", type=Path, default=None,
        help=(
            "(Replay) JSONL of wide-pool retrieval rows for gold queries, "
            "produced by a prior --live run. Each row's variant field is "
            "ignored — the file is treated as the pool."
        ),
    )
    p.add_argument(
        "--silver-pool-results", type=Path, default=None,
        help="(Replay) JSONL of wide-pool retrieval rows for silver queries.",
    )

    # Grid axes — defaults match the spec.
    p.add_argument(
        "--candidate-ks", type=str,
        default=",".join(str(k) for k in DEFAULT_CANDIDATE_K_GRID),
        help="Comma-separated candidate_k values for the sweep.",
    )
    p.add_argument(
        "--mmr-lambdas", type=str,
        default=",".join(f"{x:.2f}" for x in DEFAULT_MMR_LAMBDA_GRID),
        help="Comma-separated mmr_lambda values for the sweep.",
    )
    p.add_argument(
        "--pool-size", type=int, default=0,
        help=(
            "Wide-pool size. Defaults to max(candidate_ks). Increase only "
            "if your candidate_k axis exceeds the default 40."
        ),
    )

    p.add_argument(
        "--include-silver-guardrail", action="store_true",
        help="Also score the silver-500 set per variant.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _parse_int_list(s: str) -> List[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _filter_pool_by_ids(
    rows: Sequence[RetrievalResult], keep_ids: Sequence[str],
) -> List[RetrievalResult]:
    keep = set(keep_ids)
    return [r for r in rows if r.query_id in keep]


def _previous_best_score(
    *,
    name: str,
    gold: GoldSeedDataset,
    silver: SilverDataset,
    prev_gold_path: Optional[Path],
    prev_silver_path: Optional[Path],
    baseline_summary_gold: GoldSummary,
    baseline_summary_silver: SilverSummary,
) -> Optional[CandidateScore]:
    """Hydrate a CandidateScore for the previous-best variant if its JSONLs exist."""
    if prev_gold_path is None:
        return None
    prev_gold_rows = read_retrieval_jsonl(prev_gold_path)
    prev_silver_rows: List[RetrievalResult] = []
    if prev_silver_path is not None:
        prev_silver_rows = read_retrieval_jsonl(prev_silver_path)

    gold_ids = {r.query_id for r in gold.rows}
    silver_ids = {r.query_id for r in silver.rows}
    prev_gold_filtered = _filter_pool_by_ids(prev_gold_rows, list(gold_ids))
    prev_silver_filtered = _filter_pool_by_ids(prev_silver_rows, list(silver_ids))

    gold_map = {r.query_id: list(r.docs) for r in prev_gold_filtered}
    silver_map = {r.query_id: list(r.docs) for r in prev_silver_filtered}
    gold_eval = evaluate_gold(gold, gold_map)
    silver_eval = evaluate_silver(silver, silver_map)
    cand_gold = summarize_gold(gold_eval)
    cand_silver = summarize_silver(silver_eval)

    from eval.harness.phase7_mmr_confirm_sweep import _candidate_score_from
    spec = SweepCandidate(
        name=name, candidate_k=30, mmr_lambda=0.70, top_k=DEFAULT_TOP_K,
        use_mmr=True,
        cache_dir_relative=PRODUCTION_INDEX_CACHE_DIR,
        rag_chunks_path_relative=PRODUCTION_RAG_CHUNKS_PATH,
        description="Phase 7.x first-pass best (replayed).",
    )
    return _candidate_score_from(
        spec=spec,
        cand_summary_gold=cand_gold,
        cand_summary_silver=cand_silver,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading gold seed: %s", args.gold_path)
    gold = load_human_gold_seed_50(args.gold_path)
    log.info("loading silver: %s", args.silver_path)
    silver = load_llm_silver_500(args.silver_path)

    candidate_ks = _parse_int_list(args.candidate_ks)
    mmr_lambdas = _parse_float_list(args.mmr_lambdas)
    pool_size = int(args.pool_size) or max(candidate_ks)
    grid = make_confirm_sweep_grid(
        candidate_ks=candidate_ks,
        mmr_lambdas=mmr_lambdas,
        top_k=DEFAULT_TOP_K,
    )
    log.info(
        "grid size=%d (candidate_ks=%s × mmr_lambdas=%s); pool_size=%d",
        len(grid), candidate_ks, mmr_lambdas, pool_size,
    )

    # ------------------------------------------------------------------
    # Baseline summaries
    # ------------------------------------------------------------------
    base_gold_rows = read_retrieval_jsonl(args.baseline_gold_results)
    base_silver_rows = read_retrieval_jsonl(args.baseline_silver_results)
    base_gold_filtered = _filter_pool_by_ids(
        base_gold_rows, [r.query_id for r in gold.rows],
    )
    base_silver_filtered = _filter_pool_by_ids(
        base_silver_rows, [r.query_id for r in silver.rows],
    )
    base_gold_map = {r.query_id: list(r.docs) for r in base_gold_filtered}
    base_silver_map = {r.query_id: list(r.docs) for r in base_silver_filtered}
    baseline_summary_gold = summarize_gold(evaluate_gold(gold, base_gold_map))
    baseline_summary_silver = summarize_silver(
        evaluate_silver(silver, base_silver_map),
    )
    log.info(
        "baseline gold.primary=%.6f silver.hit@5=%.4f",
        baseline_summary_gold.primary_score,
        baseline_summary_silver.hit_at_5,
    )

    # ------------------------------------------------------------------
    # Live or replay candidate pool retrieval
    # ------------------------------------------------------------------
    if args.live:
        if args.index_root is None:
            raise SystemExit("--index-root is required with --live")
        cache_dir = Path(args.index_root) / PRODUCTION_INDEX_CACHE_DIR
        rcr = (
            Path(args.rag_chunks_root)
            if args.rag_chunks_root is not None else Path.cwd()
        )
        rag_chunks_arg = Path(PRODUCTION_RAG_CHUNKS_PATH)
        rag_chunks_path = (
            rag_chunks_arg if rag_chunks_arg.is_absolute()
            else (rcr / rag_chunks_arg).resolve()
        )
        log.info(
            "live: cache_dir=%s rag_chunks=%s pool_size=%d",
            cache_dir, rag_chunks_path, pool_size,
        )
        resources = LiveResources(
            embedding_model=args.embedding_model,
            max_seq_length=int(args.max_seq_length),
        )

        gold_queries = [(r.query_id, r.query) for r in gold.rows]
        gold_pool = precompute_candidate_pool(
            queries=gold_queries,
            cache_dir=cache_dir,
            rag_chunks_path=rag_chunks_path,
            pool_size=pool_size,
            resources=resources,
            embedding_model=args.embedding_model,
            max_seq_length=int(args.max_seq_length),
        )
        write_retrieval_jsonl(
            gold_pool, report_dir / "candidate_pool_gold.jsonl",
        )

        silver_pool: List[RetrievalResult] = []
        if args.include_silver_guardrail:
            silver_queries = [(r.query_id, r.query) for r in silver.rows]
            silver_pool = precompute_candidate_pool(
                queries=silver_queries,
                cache_dir=cache_dir,
                rag_chunks_path=rag_chunks_path,
                pool_size=pool_size,
                resources=resources,
                embedding_model=args.embedding_model,
                max_seq_length=int(args.max_seq_length),
            )
            write_retrieval_jsonl(
                silver_pool, report_dir / "candidate_pool_silver.jsonl",
            )
    else:
        if args.gold_pool_results is None:
            raise SystemExit(
                "replay mode requires --gold-pool-results (and "
                "--silver-pool-results if --include-silver-guardrail)."
            )
        gold_pool = read_retrieval_jsonl(args.gold_pool_results)
        gold_pool = _filter_pool_by_ids(
            gold_pool, [r.query_id for r in gold.rows],
        )
        silver_pool = []
        if args.include_silver_guardrail:
            if args.silver_pool_results is None:
                raise SystemExit(
                    "--include-silver-guardrail requires --silver-pool-results"
                )
            silver_pool = read_retrieval_jsonl(args.silver_pool_results)
            silver_pool = _filter_pool_by_ids(
                silver_pool, [r.query_id for r in silver.rows],
            )

    # ------------------------------------------------------------------
    # Per-variant slicing + scoring
    # ------------------------------------------------------------------
    candidate_results: Dict[str, Tuple[GoldSummary, SilverSummary]] = {}
    for spec in grid:
        gold_rows_for_variant = slice_pool_for_variant(
            spec=spec, pool_rows=gold_pool,
        )
        write_retrieval_jsonl(
            gold_rows_for_variant,
            report_dir / f"retrieval_{spec.name}_gold.jsonl",
        )
        silver_rows_for_variant: List[RetrievalResult] = []
        if silver_pool:
            silver_rows_for_variant = slice_pool_for_variant(
                spec=spec, pool_rows=silver_pool,
            )
            write_retrieval_jsonl(
                silver_rows_for_variant,
                report_dir / f"retrieval_{spec.name}_silver.jsonl",
            )
        cand_gold, cand_silver = evaluate_variant_summaries(
            spec=spec, gold=gold, silver=silver,
            gold_retrievals=gold_rows_for_variant,
            silver_retrievals=silver_rows_for_variant,
        )
        candidate_results[spec.name] = (cand_gold, cand_silver)
        log.info(
            "variant=%s primary=%.6f silver.hit@5=%.4f",
            spec.name, cand_gold.primary_score, cand_silver.hit_at_5,
        )

    # ------------------------------------------------------------------
    # Selection + reports
    # ------------------------------------------------------------------
    result = select_confirmed_best(
        grid=grid,
        baseline_name=args.baseline_name,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
        candidate_results=candidate_results,
    )

    prev_score = _previous_best_score(
        name=args.previous_best_name,
        gold=gold, silver=silver,
        prev_gold_path=args.previous_best_gold_results,
        prev_silver_path=args.previous_best_silver_results,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
    )

    write_confirm_sweep_results_jsonl(
        report_dir / "confirm_sweep_results.jsonl",
        result=result,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
        candidate_results=candidate_results,
    )
    write_confirm_sweep_summary_json(
        report_dir / "confirm_sweep_summary.json", result=result,
    )
    write_confirm_sweep_report_md(
        report_dir / "confirm_sweep_report.md",
        result=result,
        baseline_summary_gold=baseline_summary_gold,
        baseline_summary_silver=baseline_summary_silver,
        previous_best_name=args.previous_best_name,
        previous_best_score=prev_score,
    )
    write_confirmed_best_config_json(
        report_dir / "best_config.confirmed.json", result=result,
    )
    write_confirmed_best_config_env(
        report_dir / "best_config.confirmed.env", result=result,
    )

    # Manifest
    manifest = {
        "phase": "7.5_mmr_confirm_sweep",
        "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
        "promotion_target_clarification": PROMOTION_TARGET_CLARIFICATION,
        "section_hit_caveat": SECTION_HIT_CAVEAT,
        "baseline_name": args.baseline_name,
        "previous_best_name": args.previous_best_name,
        "candidate_ks": candidate_ks,
        "mmr_lambdas": mmr_lambdas,
        "pool_size": pool_size,
        "live": bool(args.live),
        "promotion_recommended": result.promotion_recommended,
        "promotion_reason": result.promotion_reason,
        "confirmed_best": (
            result.confirmed_best.to_dict()
            if result.confirmed_best else None
        ),
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote confirm sweep bundle to %s", report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
