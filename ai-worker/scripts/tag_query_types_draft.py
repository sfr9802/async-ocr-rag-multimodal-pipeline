"""Tag a silver query JSONL with heuristic ``query_type`` labels.

Eval-only utility — does NOT modify the original silver file. Produces
two artifacts:

  - ``<input>.query_type_draft.jsonl`` next to the input by default
    (or wherever ``--out`` points).
  - ``query_type_tagging_review.md`` summarising the distribution and
    flagging low-confidence rows for manual review.

The heuristic is intentionally weak — keyword-based, no model. Any
downstream metric computed against this draft is *diagnostic only*
and must be flagged accordingly in reports. Rows with confidence
below ``--low-conf-threshold`` (default 0.5) are listed in the review
report.

Usage::

    python -m scripts.tag_query_types_draft \
        --input eval/eval_queries/anime_silver_200.jsonl \
        --out   eval/eval_queries/anime_silver_200.query_type_draft.jsonl \
        --review-out eval/reports/.../query_type_tagging_review.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

from eval.harness.io_utils import load_jsonl
from eval.harness.query_type_heuristic import (
    QUERY_TYPES,
    summarize_distribution,
    tag_rows,
    write_draft_jsonl,
)


log = logging.getLogger("tag_query_types_draft")


def _render_review_md(
    *,
    input_path: Path,
    draft_path: Path,
    summary: dict,
    tagged_rows: List[dict],
    low_conf_threshold: float,
    sample_per_bucket: int = 5,
) -> str:
    lines: List[str] = []
    lines.append("# Query type tagging — heuristic draft (manual review required)")
    lines.append("")
    lines.append(f"- input:  `{input_path}`")
    lines.append(f"- draft:  `{draft_path}`")
    lines.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "> **Diagnostic only.** This tagging was produced by a "
        "heuristic keyword matcher; rows below the confidence "
        "threshold need manual review before any byQueryType metric is "
        "trusted for adoption decisions."
    )
    lines.append("")

    lines.append("## Distribution")
    lines.append("")
    lines.append("| query_type | count | fraction |")
    lines.append("|---|---:|---:|")
    for qt, payload in summary["per_type"].items():
        lines.append(
            f"| {qt} | {payload['count']} | {payload['fraction']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"- total rows: {summary['total_rows']}"
    )
    lines.append(
        f"- low-confidence rows (<{summary['low_confidence_threshold']:.2f}): "
        f"{summary['low_confidence_count']} "
        f"({summary['low_confidence_fraction']:.2%})"
    )
    lines.append(
        f"- rows with competing labels: {summary['competing_count']}"
    )
    lines.append("")

    # Per-bucket samples — one short list per query_type so a reviewer
    # can sanity-check whether the keyword routing is roughly sane.
    lines.append("## Sample queries per bucket")
    lines.append("")
    by_bucket: dict = {qt: [] for qt in QUERY_TYPES}
    for row in tagged_rows:
        qt = str(row.get("query_type"))
        if qt in by_bucket and len(by_bucket[qt]) < sample_per_bucket:
            by_bucket[qt].append(row)
    for qt in QUERY_TYPES:
        rows = by_bucket[qt]
        if not rows:
            continue
        lines.append(f"### {qt}")
        lines.append("")
        for row in rows:
            conf = row.get("query_type_confidence")
            reason = row.get("query_type_reason") or ""
            qid = row.get("id")
            qtxt = (row.get("query") or "").strip()
            lines.append(
                f"- `{qid}` (conf={conf:.2f}): "
                f"{qtxt} — *{reason}*"
            )
        lines.append("")

    # Low-confidence callout — first 30 rows so reviewer can prioritise.
    low_rows = [
        r for r in tagged_rows
        if float(r.get("query_type_confidence") or 0.0) < low_conf_threshold
    ]
    if low_rows:
        lines.append("## Low-confidence rows (first 30)")
        lines.append("")
        lines.append("| id | query_type | confidence | query |")
        lines.append("|---|---|---:|---|")
        for r in low_rows[:30]:
            qid = r.get("id")
            qt = r.get("query_type")
            conf = r.get("query_type_confidence")
            qtxt = (r.get("query") or "").replace("|", "\\|").strip()
            if len(qtxt) > 80:
                qtxt = qtxt[:77] + "…"
            lines.append(f"| {qid} | {qt} | {conf:.2f} | {qtxt} |")
        lines.append("")
        lines.append(
            f"> Total low-confidence rows: {len(low_rows)} — "
            "manual review required."
        )
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "1. Review the sample queries per bucket above and confirm "
        "the bucket name actually fits the question pattern."
    )
    lines.append(
        "2. Spot-check 10-20 low-confidence rows; promote / re-tag as "
        "needed in a manual pass."
    )
    lines.append(
        "3. Once a manual gold tagging exists, regenerate "
        "``byQueryType`` metrics; the diagnostic draft results "
        "should NOT be quoted as adoption evidence."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("eval/eval_queries/anime_silver_200.jsonl"),
        help="Source silver JSONL — left untouched.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Destination draft JSONL. Defaults to "
             "<input>.query_type_draft.jsonl",
    )
    parser.add_argument(
        "--review-out", type=Path, default=None,
        help="Destination markdown review file.",
    )
    parser.add_argument(
        "--low-conf-threshold", type=float, default=0.5,
        help="Confidence threshold below which a row is flagged for "
             "manual review.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return 2

    out_path = args.out or input_path.with_name(
        input_path.stem + ".query_type_draft.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(load_jsonl(input_path))
    log.info("Loaded %d rows from %s", len(rows), input_path)
    tagged = tag_rows(rows)
    write_draft_jsonl(tagged, out_path)
    log.info("Wrote tagged draft → %s", out_path)

    summary = summarize_distribution(
        tagged, low_confidence_threshold=args.low_conf_threshold,
    )
    if args.review_out is not None:
        review_path = Path(args.review_out)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_md = _render_review_md(
            input_path=input_path,
            draft_path=out_path,
            summary=summary,
            tagged_rows=tagged,
            low_conf_threshold=args.low_conf_threshold,
        )
        review_path.write_text(review_md, encoding="utf-8")
        log.info("Wrote review markdown → %s", review_path)

    # Print headline distribution to stdout for the operator.
    counts = Counter(r["query_type"] for r in tagged)
    print(json.dumps({
        "total_rows": len(tagged),
        "per_type": {k: counts[k] for k in sorted(counts)},
        "low_confidence_count": summary["low_confidence_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
