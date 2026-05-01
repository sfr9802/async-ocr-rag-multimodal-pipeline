"""CLI: rewrite the legacy keyword-derived silver-500 as a sanity-only set.

Reads the existing ``queries_v4_silver_500.jsonl`` (the keyword-template
output of ``silver_500_generator.py``) and writes a sibling
``queries_v4_keyword_sanity_500.jsonl`` whose tags / metadata mark it
as a sanity / lexical smoke test — NOT the main retrieval eval set.
The new main eval set is the LLM-authored silver-500
(``queries_v4_llm_silver_500.jsonl``).

The transform is a one-pass JSONL rewrite. The semantic content
(``query``, ``expected_doc_ids``, ``v4_meta.bucket / page_title /
retrieval_title``) stays byte-identical so the existing Phase 7.0 /
7.1 baseline comparisons remain reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.keyword_sanity_set import (  # noqa: E402
    read_jsonl,
    retag_records,
    write_jsonl,
    write_summary_md,
)


DEFAULT_INPUT = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "silver500"
    / "queries"
    / "queries_v4_silver_500.jsonl"
)
DEFAULT_OUT_DIR = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "seeds"
    / "keyword_sanity_500"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="path to the legacy keyword silver-500 JSONL",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory for queries_v4_keyword_sanity_500.* files",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    rows = read_jsonl(args.input)
    retagged, stats = retag_records(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "queries_v4_keyword_sanity_500.jsonl"
    summary_path = args.out_dir / "queries_v4_keyword_sanity_500.summary.md"

    write_jsonl(retagged, jsonl_path)
    write_summary_md(retagged, stats, summary_path)

    print(f"rows_in:        {stats.rows_in}")
    print(f"rows_out:       {stats.rows_out}")
    print(f"ids_rewritten:  {stats.ids_rewritten}")
    if stats.forbidden_tags_seen:
        print(
            f"forbidden_tags_dropped: {len(stats.forbidden_tags_seen)} "
            f"(see summary.md for the qid list)",
            file=sys.stderr,
        )
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
