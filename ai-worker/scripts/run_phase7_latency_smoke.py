"""Phase 7.5 — latency smoke check CLI.

Replays the cached candidate pool produced by ``run_phase7_mmr_confirm_sweep``
and times the post-hoc MMR pass for each config under test. The default
suite compares three rows:

  * pre-promotion baseline (use_mmr=false, top_k=10)
  * Phase 7.x first-pass best (candidate_k=30, MMR, λ=0.70)
  * Phase 7.5 production-recommended (candidate_k=40, MMR, λ=0.70)

Inputs:

  * ``--gold-pool-results candidate_pool_gold.jsonl`` — wide-pool rows
    for the 50 gold queries (one per query, ``elapsed_ms`` recorded
    live by the confirm sweep at ``pool_size=40``).
  * ``--silver-pool-results candidate_pool_silver.jsonl`` — same shape
    for the 500 silver queries.

Outputs (under ``--report-dir``):

  * ``latency_smoke_results.json`` — full ``LatencySmokeReport.to_dict()``
  * ``latency_smoke_report.md`` — rendered headline tables + verdict

Honest scope:

  * ``candidate_gen_ms`` reflects the cached pool's ``elapsed_ms``,
    measured live at ``pool_size=40``.
  * ``mmr_post_ms`` is timed live in this CLI run (one repetition
    average, ``DEFAULT_MMR_REPS`` per query).
  * **Reranker time is NOT measured** — see the harness module's
    "honest scope" notes. Do not paste these numbers into a
    production-latency dashboard without first confirming the rerank
    cost separately.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from eval.harness.phase7_latency_smoke import (
    DEFAULT_MMR_REPS,
    LatencySmokeConfig,
    default_smoke_suite,
    run_smoke_check,
    write_latency_smoke_json,
    write_latency_smoke_md,
)
from scripts.phase7_human_gold_tune import read_retrieval_jsonl


log = logging.getLogger("scripts.run_phase7_latency_smoke")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.5 latency smoke check — replays the cached candidate "
        "pool and times the post-hoc MMR pass for the production-"
        "promotion suite (baseline / previous-best / recommended)."
    ))
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help=(
            "Output directory. Latency artifacts are written next to "
            "the existing confirm-sweep bundle."
        ),
    )
    p.add_argument(
        "--gold-pool-results", type=Path, required=True,
        help="Wide-pool retrieval JSONL for the 50 gold queries.",
    )
    p.add_argument(
        "--silver-pool-results", type=Path, default=None,
        help=(
            "Wide-pool retrieval JSONL for the 500 silver queries. "
            "If omitted, only gold-50 measurements are produced."
        ),
    )
    p.add_argument(
        "--pool-size", type=int, default=40,
        help=(
            "The pool_size the candidate pool was generated at. "
            "Carried into the report's note section so a reader knows "
            "candidate_gen_ms is an upper-bound for candidate_k<pool_size."
        ),
    )
    p.add_argument(
        "--reps", type=int, default=DEFAULT_MMR_REPS,
        help="MMR micro-benchmark repetitions per query.",
    )
    p.add_argument(
        "--include-combined", action="store_true",
        help=(
            "Also produce a combined (gold+silver) aggregate row per "
            "config. Off by default; on for the standard CLI run."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    gold_pool = read_retrieval_jsonl(args.gold_pool_results)
    log.info(
        "loaded gold candidate pool: %d rows from %s",
        len(gold_pool), args.gold_pool_results,
    )

    silver_pool: List = []
    if args.silver_pool_results is not None:
        silver_pool = read_retrieval_jsonl(args.silver_pool_results)
        log.info(
            "loaded silver candidate pool: %d rows from %s",
            len(silver_pool), args.silver_pool_results,
        )

    configs: List[LatencySmokeConfig] = list(default_smoke_suite())
    log.info(
        "running smoke check: %d configs × %d sets, reps=%d",
        len(configs),
        2 + (1 if args.include_combined else 0),
        int(args.reps),
    )

    report = run_smoke_check(
        configs=configs,
        gold_pool_rows=gold_pool,
        silver_pool_rows=silver_pool,
        pool_size=int(args.pool_size),
        reps=int(args.reps),
        include_combined=bool(args.include_combined),
    )

    json_path = report_dir / "latency_smoke_results.json"
    md_path = report_dir / "latency_smoke_report.md"
    write_latency_smoke_json(json_path, report)
    write_latency_smoke_md(md_path, report)
    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
