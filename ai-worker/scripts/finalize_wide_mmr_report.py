"""Post-process a wide-MMR-titlecap sweep into the final report bundle.

Reads ``sweep_summary.json`` from a ``retrieval-wide-mmr-titlecap-*``
directory and emits the remaining spec-required artifacts:

  - ``mmr_effect_comparison.csv``        — focused MMR-on vs MMR-off
                                           deltas at the same pool size.
  - ``title_cap_effect_comparison.csv``  — title_cap=1 vs cap=2 vs none.
  - ``pareto_quality_vs_latency.json``   — quality_score vs p95 frontier.
  - ``pareto_quality_vs_latency.md``     — markdown rendering.

Also re-runs the heuristic ``query_type`` tagger so the review markdown
lands inside the same report directory (the original tagging step
runs against the source silver file; this just copies its output to
the final report dir for self-containment).

Usage::

    python -m scripts.finalize_wide_mmr_report \\
        --report-dir eval/reports/retrieval-wide-mmr-titlecap-<TS>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("finalize_wide_mmr_report")


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def _write_mmr_effect(report_dir: Path, payload: Dict[str, Any]) -> None:
    """Compare MMR-on vs MMR-off cells at candidate_k=200.

    Highlights the diversity vs hit@5 trade — the spec wants a clear
    answer to "did MMR keep quality while lowering duplicates?".
    """
    rows = payload["cells"]
    headers = [
        "label", "use_mmr", "mmr_lambda", "mmr_k",
        "title_cap_rerank_input", "title_cap_final", "final_top_k",
        "hit@5", "mrr@10", "ndcg@10",
        "duplicateDocRatio@10", "uniqueDocCount@10",
        "p95TotalRetrievalMs", "grade",
    ]
    out = report_dir / "mmr_effect_comparison.csv"
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for cell in rows:
            spec = cell["spec"]
            s = cell["summary"]
            grade = cell.get("grade", {})
            writer.writerow([
                cell["label"],
                spec.get("use_mmr"),
                spec.get("mmr_lambda"),
                spec.get("mmr_k"),
                spec.get("title_cap_rerank_input"),
                spec.get("title_cap_final"),
                spec.get("final_top_k"),
                _safe_get(s, "mean_hit_at_5"),
                _safe_get(s, "mean_mrr_at_10"),
                _safe_get(s, "mean_ndcg_at_10"),
                _safe_get(s, "duplicate_doc_ratios", "10"),
                _safe_get(s, "unique_doc_counts", "10"),
                _safe_get(s, "p95_total_retrieval_ms")
                or _safe_get(s, "p95_retrieval_ms"),
                grade.get("grade", ""),
            ])
    log.info("Wrote %s", out)


def _write_title_cap_effect(report_dir: Path, payload: Dict[str, Any]) -> None:
    """Compare title_cap variants at the same MMR settings."""
    headers = [
        "label", "title_cap_rerank_input", "title_cap_final",
        "final_top_k", "hit@5", "mrr@10",
        "duplicateDocRatio@10", "uniqueDocCount@10",
        "sectionDiversity@10", "p95TotalRetrievalMs", "grade",
    ]
    out = report_dir / "title_cap_effect_comparison.csv"
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for cell in payload["cells"]:
            spec = cell["spec"]
            s = cell["summary"]
            grade = cell.get("grade", {})
            writer.writerow([
                cell["label"],
                spec.get("title_cap_rerank_input"),
                spec.get("title_cap_final"),
                spec.get("final_top_k"),
                _safe_get(s, "mean_hit_at_5"),
                _safe_get(s, "mean_mrr_at_10"),
                _safe_get(s, "duplicate_doc_ratios", "10"),
                _safe_get(s, "unique_doc_counts", "10"),
                _safe_get(s, "section_diversities", "10"),
                _safe_get(s, "p95_total_retrieval_ms")
                or _safe_get(s, "p95_retrieval_ms"),
                grade.get("grade", ""),
            ])
    log.info("Wrote %s", out)


def _write_pareto(report_dir: Path, payload: Dict[str, Any]) -> None:
    """Build a quality_score vs p95 latency Pareto frontier.

    Re-uses the existing TopNSweep adapter so the markdown output
    matches what the legacy sweep produces.
    """
    from eval.harness.pareto_frontier import (
        compute_pareto_frontier,
        pareto_to_dict,
        render_pareto_markdown,
    )
    from eval.harness.topn_sweep import TopNSweepEntry, TopNSweepReport

    entries: List[TopNSweepEntry] = []
    for cell in payload["cells"]:
        spec = cell["spec"]
        s = cell["summary"]
        entries.append(TopNSweepEntry(
            label=cell["label"],
            report_path=None,
            dense_top_n=int(spec.get("candidate_k") or 0),
            final_top_k=int(spec.get("final_top_k") or 0),
            reranker_batch_size=None,
            reranker_model=None,
            row_count=int(_safe_get(s, "row_count", default=0)),
            rows_with_expected_doc_ids=int(
                _safe_get(s, "rows_with_expected_doc_ids", default=0),
            ),
            mean_hit_at_1=_safe_get(s, "mean_hit_at_1"),
            mean_hit_at_3=_safe_get(s, "mean_hit_at_3"),
            mean_hit_at_5=_safe_get(s, "mean_hit_at_5"),
            mean_mrr_at_10=_safe_get(s, "mean_mrr_at_10"),
            mean_ndcg_at_10=_safe_get(s, "mean_ndcg_at_10"),
            candidate_recall=_safe_get(s, "candidate_recalls", "50"),
            mean_dup_rate=_safe_get(s, "mean_dup_rate"),
            mean_avg_context_token_count=_safe_get(
                s, "mean_avg_context_token_count",
            ),
            rerank_avg_ms=_safe_get(s, "mean_rerank_ms"),
            rerank_p50_ms=_safe_get(s, "p50_rerank_ms"),
            rerank_p90_ms=_safe_get(s, "p90_rerank_ms"),
            rerank_p95_ms=_safe_get(s, "p95_rerank_ms"),
            rerank_p99_ms=_safe_get(s, "p99_rerank_ms"),
            rerank_max_ms=_safe_get(s, "max_rerank_ms"),
            rerank_row_count=int(
                _safe_get(s, "rerank_row_count", default=0),
            ),
            total_query_avg_ms=(
                _safe_get(s, "avg_total_retrieval_ms")
                or _safe_get(s, "mean_retrieval_ms")
            ),
            total_query_p50_ms=_safe_get(s, "p50_retrieval_ms"),
            total_query_p90_ms=_safe_get(s, "p90_retrieval_ms"),
            total_query_p95_ms=(
                _safe_get(s, "p95_total_retrieval_ms")
                or _safe_get(s, "p95_retrieval_ms")
            ),
            total_query_p99_ms=_safe_get(s, "p99_retrieval_ms"),
            total_query_max_ms=_safe_get(s, "max_retrieval_ms"),
            total_query_row_count=int(
                _safe_get(s, "row_count", default=0),
            ),
            dense_retrieval_avg_ms=_safe_get(s, "mean_dense_retrieval_ms"),
            dense_retrieval_p50_ms=_safe_get(s, "p50_dense_retrieval_ms"),
            dense_retrieval_p95_ms=_safe_get(s, "p95_dense_retrieval_ms"),
            dense_retrieval_row_count=int(
                _safe_get(s, "dense_retrieval_row_count", default=0),
            ),
        ))
    topn = TopNSweepReport(
        schema="phase2-wide-mmr-titlecap.v1::topn-adapter",
        entries=entries,
    )
    pareto = compute_pareto_frontier(
        topn,
        metric="mean_hit_at_5",
        latency="total_query_p95_ms",
    )
    (report_dir / "pareto_quality_vs_latency.json").write_text(
        json.dumps(pareto_to_dict(pareto), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "pareto_quality_vs_latency.md").write_text(
        render_pareto_markdown(pareto), encoding="utf-8",
    )
    log.info("Wrote pareto JSON + MD")


def _maybe_copy_query_type_review(
    report_dir: Path, source_review: Path,
) -> None:
    """Copy the query_type review into the report dir if present.

    Idempotent: if the review already lives there, skip. The
    ``tag_query_types_draft`` script writes the review wherever the
    operator pointed ``--review-out``; the spec wants the review in
    the report dir, so we copy it here as a final step.
    """
    target = report_dir / "query_type_tagging_review.md"
    if target.exists():
        return
    if not source_review.exists():
        log.warning(
            "query_type_tagging_review not found at %s — skipping copy",
            source_review,
        )
        return
    shutil.copyfile(source_review, target)
    log.info("Copied query_type review → %s", target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, required=True,
        help="A retrieval-wide-mmr-titlecap-* report directory.",
    )
    parser.add_argument(
        "--query-type-review-source", type=Path, default=None,
        help="Optional source path to copy the query_type review from.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        log.error("Report dir not found: %s", report_dir)
        return 2
    summary_path = report_dir / "sweep_summary.json"
    if not summary_path.exists():
        log.error(
            "Missing sweep_summary.json in %s — finalize aborted.",
            report_dir,
        )
        return 2

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    _write_mmr_effect(report_dir, payload)
    _write_title_cap_effect(report_dir, payload)
    try:
        _write_pareto(report_dir, payload)
    except Exception as ex:  # pragma: no cover
        log.warning("Pareto rendering failed: %s: %s", type(ex).__name__, ex)

    if args.query_type_review_source is not None:
        _maybe_copy_query_type_review(
            report_dir, Path(args.query_type_review_source),
        )

    log.info("Finalized — artifacts in %s", report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
