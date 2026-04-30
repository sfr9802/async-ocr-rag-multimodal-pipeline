"""Phase 7.3 — orchestrator CLI for the retrieval confidence detector.

Reads Phase 7.0's per_query_comparison.jsonl (and optionally Phase
7.1's reranker per_query bundle) and writes a confidence-eval bundle
that Phase 7.4 will consume as the recovery loop's trigger list. No
production code is touched — this is a pure post-hoc analysis pass.

Usage::

    python -m scripts.run_phase7_3_confidence_eval \\
        --per-query eval/reports/.../per_query_comparison.jsonl \\
        --chunks eval/reports/.../rag_chunks_retrieval_title_section.jsonl \\
        --silver-queries eval/reports/.../queries_v4_silver.jsonl \\
        --report-dir eval/reports/...-phase7_3_confidence_eval/

Optionally include the Phase 7.1 reranker output to additionally surface
``rerank_demoted_gold`` and merge rerank scores::

    python -m scripts.run_phase7_3_confidence_eval ... \\
        --rerank-per-query eval/reports/.../per_query_comparison.jsonl \\
        --side candidate

The classifier's thresholds are exposed as flags but default to the
conservative pack baked into :class:`ConfidenceConfig` — the goal of
Phase 7.3 is to *describe* the distribution, not to lock in an
operating point. Phase 7.4 will tune the thresholds against the label
distribution this run produces.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from eval.harness.v4_confidence_detector import (
    ConfidenceConfig,
    ConfidenceEvalResult,
    aggregate_verdicts,
    decide,
    find_confident_but_wrong,
    find_low_confidence_but_correct,
    load_inputs_from_phase7_artifacts,
    write_outputs,
)


log = logging.getLogger("scripts.run_phase7_3_confidence_eval")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 7.3 retrieval confidence detector / failure classifier. "
            "Reads Phase 7.0 (and optionally Phase 7.1) per-query bundles, "
            "writes per-query confidence verdicts plus aggregate summaries."
        ),
    )
    p.add_argument(
        "--per-query", type=Path, required=True,
        help=(
            "Phase 7.0 per_query_comparison.jsonl. The classifier reads the "
            "side selected by --side from each row."
        ),
    )
    p.add_argument(
        "--chunks", type=Path, default=None,
        help=(
            "Optional rag_chunks_*.jsonl whose chunk_id index supplies "
            "title / retrieval_title / section_path / section_type. When "
            "omitted, the corresponding signals stay None and the related "
            "reasons cannot fire."
        ),
    )
    p.add_argument(
        "--silver-queries", type=Path, default=None,
        help=(
            "Optional queries_v4_silver.jsonl. When present, supplies "
            "expected_section_keywords and gold_page_id."
        ),
    )
    p.add_argument(
        "--rerank-per-query", type=Path, default=None,
        help=(
            "Optional Phase 7.1 per_query_comparison.jsonl. When set, "
            "rerank scores from candidate_pool_preview and the "
            "gold_was_demoted flag are merged into each input."
        ),
    )
    p.add_argument(
        "--side", choices=("baseline", "candidate"), default="candidate",
        help=(
            "Which side of the Phase 7.0 A/B to classify. Default is "
            "'candidate' (the retrieval_title_section variant promoted "
            "by Phase 7.0)."
        ),
    )
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output directory for the Phase 7.3 artefact bundle.",
    )

    # Threshold flags — defaults match ConfidenceConfig().
    cfg_default = ConfidenceConfig()
    p.add_argument(
        "--min-top1-score", type=float, default=cfg_default.min_top1_score,
        help="Floor on top-1 effective score. Default conservative.",
    )
    p.add_argument(
        "--min-margin", type=float, default=cfg_default.min_margin,
        help="Floor on top1−top2 score gap.",
    )
    p.add_argument(
        "--min-same-page-ratio", type=float,
        default=cfg_default.min_same_page_ratio,
        help="Floor on top-k share of the most-common page_or_doc_id.",
    )
    p.add_argument(
        "--max-duplicate-rate", type=float,
        default=cfg_default.max_duplicate_rate,
        help="Ceiling on duplicate-doc ratio across top-k.",
    )
    p.add_argument(
        "--max-generic-collision-count", type=int,
        default=cfg_default.max_generic_collision_count,
        help="Ceiling on generic-section-token collision count.",
    )
    p.add_argument(
        "--min-evidence-chunks-same-page", type=int,
        default=cfg_default.min_evidence_chunks_same_page,
        help="Minimum chunks that must share the top-1 doc_id / page_id.",
    )
    p.add_argument(
        "--gold-low-rank-threshold", type=int,
        default=cfg_default.gold_low_rank_threshold,
        help="Gold rank above which GOLD_LOW_RANK fires.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ConfidenceConfig(
        min_top1_score=args.min_top1_score,
        min_margin=args.min_margin,
        min_same_page_ratio=args.min_same_page_ratio,
        max_duplicate_rate=args.max_duplicate_rate,
        max_generic_collision_count=args.max_generic_collision_count,
        min_evidence_chunks_same_page=args.min_evidence_chunks_same_page,
        gold_low_rank_threshold=args.gold_low_rank_threshold,
    ).validate()
    log.info("ConfidenceConfig: %s", asdict(cfg))

    inputs = load_inputs_from_phase7_artifacts(
        args.per_query,
        chunks_jsonl=args.chunks,
        silver_queries_path=args.silver_queries,
        side=args.side,
        rerank_per_query_path=args.rerank_per_query,
    )
    log.info(
        "Loaded %d query inputs from %s (side=%s, chunks=%s, rerank=%s).",
        len(inputs), args.per_query, args.side, args.chunks,
        args.rerank_per_query,
    )

    verdicts = [decide(inp, config=cfg) for inp in inputs]
    aggregate = aggregate_verdicts(verdicts, cfg)
    result = ConfidenceEvalResult(
        verdicts=verdicts,
        inputs=inputs,
        aggregate=aggregate,
    )

    out_paths = write_outputs(result, out_dir=args.report_dir)
    log.info("Phase 7.3 confidence eval finished. Outputs:")
    for role, p in out_paths.items():
        log.info("  %s -> %s", role, p)

    cb_wrong = find_confident_but_wrong(verdicts)
    lc_correct = find_low_confidence_but_correct(verdicts)
    log.info(
        "Labels: %s | Actions: %s | confident_but_wrong=%d "
        "low_confidence_but_correct=%d",
        aggregate.get("labels"),
        aggregate.get("actions"),
        len(cb_wrong),
        len(lc_correct),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
