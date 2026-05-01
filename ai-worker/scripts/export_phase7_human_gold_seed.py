"""Phase 7 human-gold-seed audit exporter — CLI.

Draws a stratified audit sample (default 50 rows) off the silver query
set + the Phase 7.3/7.4 outputs, and emits three sibling files a human
reviewer can fill in::

  phase7_human_gold_seed_50.jsonl
  phase7_human_gold_seed_50.csv
  phase7_human_gold_seed_50.md

The exported set is **silver-derived**, NOT human-verified gold —
``human_label`` is intentionally blank on every row. Fill it in, then
the corrected file is what should be used as gold for any
precision/recall/accuracy report.

Usage::

    python -m scripts.export_phase7_human_gold_seed \\
        --silver-queries  eval/reports/<run>/queries_v4_silver_500.jsonl \\
        --per-query       eval/reports/<run>/per_query_comparison.jsonl \\
        --confidence      eval/reports/<run>/per_query_confidence.jsonl \\
        --recovery        eval/reports/<run>/recovery_attempts.jsonl \\
        --chunks          eval/reports/<run>/rag_chunks_retrieval_title_section.jsonl \\
        --out-dir         eval/reports/<run>/phase7_human_gold_seed/

Targets default to (main_work=10, subpage_generic=20, subpage_named=20).
The exporter prioritises edge-case coverage over hitting the bucket
targets exactly when they conflict — see the module docstring of
``human_gold_seed_export`` for the picker's tie-break order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from eval.harness.human_gold_seed_export import (
    DEFAULT_BUCKET_TARGETS,
    HumanGoldSeedConfig,
    build_human_gold_seed,
    write_outputs,
)


log = logging.getLogger("scripts.export_phase7_human_gold_seed")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 7 human-gold-seed audit exporter. Reads the silver "
            "query set + Phase 7.3/7.4 outputs and emits a stratified "
            "audit seed for manual labelling. The exported file is "
            "silver-derived, NOT human-verified gold."
        ),
    )
    p.add_argument(
        "--silver-queries", type=Path, required=True,
        help="Silver query JSONL (queries_v4_silver_500.jsonl or v4_silver).",
    )
    p.add_argument(
        "--per-query", type=Path, default=None,
        help=(
            "Phase 7.0 per_query_comparison.jsonl. Optional; supplies "
            "the top_results list when no confidence file is provided."
        ),
    )
    p.add_argument(
        "--confidence", type=Path, default=None,
        help=(
            "Phase 7.3 per_query_confidence.jsonl. Optional; supplies "
            "confidence_label / failure_reasons / recommended_action."
        ),
    )
    p.add_argument(
        "--recovery", type=Path, default=None,
        help=(
            "Phase 7.4 recovery_attempts.jsonl. Optional; supplies "
            "recovery_action / rewrite_mode."
        ),
    )
    p.add_argument(
        "--chunks", type=Path, default=None,
        help=(
            "Phase 7.0 rag_chunks_*.jsonl. Optional; supplies enriched "
            "title / retrieval_title / section_path / snippet."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory where the JSONL / CSV / MD artefacts are written.",
    )
    p.add_argument(
        "--target-total", type=int, default=50,
        help="Total number of audit rows to emit (default 50).",
    )
    p.add_argument(
        "--main-work-target", type=int,
        default=DEFAULT_BUCKET_TARGETS["main_work"],
        help="Per-bucket target for main_work rows.",
    )
    p.add_argument(
        "--subpage-generic-target", type=int,
        default=DEFAULT_BUCKET_TARGETS["subpage_generic"],
        help="Per-bucket target for subpage_generic rows.",
    )
    p.add_argument(
        "--subpage-named-target", type=int,
        default=DEFAULT_BUCKET_TARGETS["subpage_named"],
        help="Per-bucket target for subpage_named rows.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Sampling seed.",
    )
    p.add_argument(
        "--side", choices=("baseline", "candidate"), default="candidate",
        help="Which side of the per_query A/B to read top_results from.",
    )
    p.add_argument(
        "--base-name", default="phase7_human_gold_seed_50",
        help=(
            "Filename stem for the three output files (default "
            "phase7_human_gold_seed_50)."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bucket_targets = {
        "main_work": int(args.main_work_target),
        "subpage_generic": int(args.subpage_generic_target),
        "subpage_named": int(args.subpage_named_target),
    }
    config = HumanGoldSeedConfig(
        target_total=int(args.target_total),
        bucket_targets=bucket_targets,
        seed=int(args.seed),
        side=args.side,
    ).validate()

    log.info(
        "Building human gold seed: silver=%s per_query=%s confidence=%s "
        "recovery=%s chunks=%s",
        args.silver_queries, args.per_query, args.confidence,
        args.recovery, args.chunks,
    )
    log.info(
        "Config: target_total=%d bucket_targets=%s seed=%d side=%s",
        config.target_total, dict(config.bucket_targets),
        config.seed, config.side,
    )

    export = build_human_gold_seed(
        silver_path=args.silver_queries,
        per_query_path=args.per_query,
        confidence_path=args.confidence,
        recovery_path=args.recovery,
        chunks_path=args.chunks,
        config=config,
    )

    out_paths = write_outputs(
        export,
        out_dir=args.out_dir,
        base_name=args.base_name,
        target_total=config.target_total,
    )

    log.info("Phase 7 human-gold-seed export finished. Outputs:")
    for role, p in out_paths.items():
        log.info("  %s -> %s", role, p)

    audit = export.audit_summary
    log.info(
        "Picked %d / %d rows. bucket_actual=%s deficits=%s missing_edges=%s",
        audit.get("actual_total"), audit.get("target_total"),
        audit.get("bucket_actual_counts"), audit.get("bucket_deficits"),
        audit.get("edge_cases_missing"),
    )
    log.info(
        "REMINDER: every row's human_label is BLANK by design. The "
        "audit seed is silver-derived, NOT human-verified gold. Fill "
        "the labels before reporting any P/R/F1 number."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
