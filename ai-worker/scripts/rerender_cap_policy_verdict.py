"""Rerender the cap-policy confirm sweep verdict + comparison report.

Reads the existing ``summary.json`` from a finished cap-policy confirm
sweep, recomputes the verdict using the current
``decide_cap_policy_verdict`` logic, and rewrites only the verdict
block in ``summary.json`` plus the ``Verdict`` / ``Next-step
recommendation`` sections of ``comparison_report.md``. All other
artefacts stay untouched.

Used when the verdict logic is tweaked after a sweep finishes — avoids
re-running the 200-row × 6-policy retrieval pass to pick up the
updated tie-break rule.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


def _load_summary(report_dir: Path) -> Dict[str, Any]:
    payload = json.loads(
        (report_dir / "summary.json").read_text(encoding="utf-8")
    )
    return payload


def _recompute_verdict(
    payload: Dict[str, Any],
) -> Any:
    from eval.harness.cap_policy_comparison import (
        compute_cap_policy_deltas,
        decide_cap_policy_verdict,
    )

    runs_by_policy = {r["policy"]: r for r in payload["runs"]}
    anchor_summary = runs_by_policy["title_cap_1"]["summary"]

    deltas: Dict[str, Any] = {}
    for policy, run in runs_by_policy.items():
        deltas[policy] = compute_cap_policy_deltas(
            policy_label=policy,
            policy_summary=run["summary"],
            anchor_summary=anchor_summary,
        )
    bucket_metrics = {
        p: r.get("bucket_metrics", {}) for p, r in runs_by_policy.items()
    }
    audit_summary = {
        p: r.get("audit_summary", {}) for p, r in runs_by_policy.items()
    }
    return decide_cap_policy_verdict(
        deltas_by_policy=deltas,
        bucket_metrics_by_policy=bucket_metrics,
        audit_summary_by_policy=audit_summary,
    )


def _next_step_block(verdict: str) -> str:
    """Return the bulleted next-step recommendation for ``verdict``."""
    if verdict == "ADOPT_TITLE_CAP_RERANK_INPUT_2":
        return (
            "1. Adopt ``title_cap_rerank_input=2`` as the production "
            "cap. Update the production retriever's title cap (or the "
            "wide-MMR config that wraps it) to ``cap=2`` at the rerank-"
            "input boundary while keeping the final cap at 2.\n"
            "2. Re-run the full retrieval eval (silver_200 + a smoke "
            "battery) with the new cap to confirm no latency regression."
        )
    if verdict == "ADOPT_TITLE_CAP_RERANK_INPUT_3":
        return (
            "1. Adopt ``title_cap_rerank_input=3`` — the loosest title "
            "cap that still beats the anchor. Watch dup@10 / latency."
        )
    if verdict == "ADOPT_DOC_ID_LEVEL_CAP":
        return (
            "1. Switch the rerank-input cap from title-level to doc_id-"
            "level grouping. The eval-only ``DocIdCapPolicy`` is the "
            "reference; mirror it in the production retriever.\n"
            "2. The production helper currently caps by title; adding a "
            "``cap_grouping=doc_id`` knob to the wide-MMR config is the "
            "minimal change."
        )
    if verdict == "ADOPT_NO_CAP_RERANK_INPUT":
        return (
            "1. Drop the rerank-input cap entirely. Retain the final "
            "cap (cap=2) so output diversity stays bounded.\n"
            "2. Audit dup@10 in production traffic — the no-cap pool "
            "feeds the cross-encoder more near-duplicates per query, "
            "which can shift the latency profile in ways the 200-row "
            "silver dataset doesn't surface."
        )
    if verdict == "NEED_SCHEMA_ENRICHMENT":
        return (
            "1. Cap policy is **not** the bottleneck. The "
            "``character_relation`` bucket stays below 0.45 hit@5 "
            "regardless of cap relaxation, and ``gold_was_capped_out`` "
            "doesn't drop materially across policies — the dataset is "
            "missing the disambiguation signals (work_title, "
            "entity_name, entity_type, section_path, source_doc_id) "
            "the reranker would need.\n"
            "2. Next axis: dataset schema enrichment. Add per-chunk "
            "entity / character / arc tags so the reranker can "
            "disambiguate same-franchise / same-character queries from "
            "structural metadata, not just chunk text."
        )
    return (
        "1. Keep ``title_cap_rerank_input=1``. No alternative cap "
        "policy clears EPS on hit@5 / MRR over the anchor.\n"
        "2. Two next directions: (a) reranker model headroom — "
        "``bge-reranker-v2-m3`` may saturate on this dataset; try a "
        "larger reranker. (b) chunking / section-level dedup before "
        "the dense pool, to reduce the reranker's same-title load "
        "without touching the cap."
    )


def _patch_markdown(
    report_dir: Path, verdict: str, rationale: str,
) -> None:
    md_path = report_dir / "comparison_report.md"
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Find the "## Verdict" header and replace through to the next "## "
    # header (which is "## Caveats" in the existing template).
    out: list = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("## Verdict"):
            # Inject the new verdict + next-step blocks.
            out.append("## Verdict")
            out.append("")
            out.append(f"**{verdict}** — {rationale}")
            out.append("")
            out.append("## Next-step recommendation")
            out.append("")
            for ln in _next_step_block(verdict).split("\n"):
                out.append(ln)
            out.append("")
            # Skip until the next "## " header.
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            # Skip the old "## Next-step recommendation" too.
            if i < len(lines) and lines[i].startswith("## Next-step"):
                i += 1
                while i < len(lines) and not lines[i].startswith("## "):
                    i += 1
            continue
        out.append(lines[i])
        i += 1
    md_path.write_text("\n".join(out), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args(argv)

    if not (args.report_dir / "summary.json").exists():
        print(f"summary.json not found in {args.report_dir}")
        return 2

    payload = _load_summary(args.report_dir)
    verdict, rationale = _recompute_verdict(payload)
    print(f"Recomputed verdict: {verdict}")
    print(f"Rationale: {rationale}")

    payload["verdict"] = {"label": verdict, "rationale": rationale}
    (args.report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _patch_markdown(args.report_dir, verdict, rationale)
    print(f"Updated {args.report_dir}/summary.json + comparison_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
