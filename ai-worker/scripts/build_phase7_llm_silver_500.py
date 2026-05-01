"""CLI: build the LLM-authored silver-500 retrieval eval set.

Reads the hand-authored ``QUERIES`` tuple from
:mod:`eval.harness.llm_silver_500` plus the corpus chunks file, runs
lexical_overlap + leakage_guard for every row, optionally computes
BM25 first-rank-for-page (``--with-bm25``), and writes:

  - ``queries_v4_llm_silver_500.jsonl``
  - ``queries_v4_llm_silver_500.summary.json``
  - ``queries_v4_llm_silver_500.summary.md``

The build is deterministic — two runs over the same inputs produce
byte-identical files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.llm_silver_500 import (  # noqa: E402
    _validate_distribution,
    build_records,
    get_full_queries,
    write_jsonl,
    write_summary_json,
    write_summary_md,
)
from eval.harness.leakage_guard import summarize_leakage  # noqa: E402


DEFAULT_CORPUS = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "7.0_retrieval_title_ab"
    / "rag_chunks_retrieval_title_section.jsonl"
)
DEFAULT_OUT_DIR = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "seeds"
    / "llm_silver_500"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help="path to rag_chunks_retrieval_title_section.jsonl",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="directory to write the JSONL + summary files",
    )
    ap.add_argument(
        "--with-bm25", action="store_true",
        help="compute bm25_expected_page_first_rank for every row "
             "(slow; default null)",
    )
    ap.add_argument(
        "--bm25-max-rank", type=int, default=100,
        help="rank cap for the BM25 walk (default 100)",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    bm25_index = None
    if args.with_bm25:
        from eval.harness.llm_silver_bm25 import build_from_chunks_file
        t0 = time.time()
        print(f"building BM25 index over {args.corpus.name} ...", file=sys.stderr)
        bm25_index = build_from_chunks_file(args.corpus)
        print(f"  built in {time.time() - t0:.1f}s, "
              f"chunks={bm25_index.n_chunks}", file=sys.stderr)

    queries = get_full_queries()
    audit = _validate_distribution(queries)
    if audit["deltas"] or audit["duplicate_query_count"] > 0:
        print(
            f"WARN: distribution deltas={len(audit['deltas'])} "
            f"duplicates={audit['duplicate_query_count']}",
            file=sys.stderr,
        )

    records = build_records(
        args.corpus,
        bm25_index=bm25_index,
        bm25_max_rank=args.bm25_max_rank,
    )
    leakage_block = summarize_leakage(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "queries_v4_llm_silver_500.jsonl"
    summary_json_path = args.out_dir / "queries_v4_llm_silver_500.summary.json"
    summary_md_path = args.out_dir / "queries_v4_llm_silver_500.summary.md"

    write_jsonl(records, jsonl_path)
    write_summary_json(records, leakage_block, audit, summary_json_path)
    write_summary_md(records, leakage_block, summary_md_path)

    print(f"wrote {len(records)} queries -> {jsonl_path}")
    print(f"wrote summary             -> {summary_json_path}")
    print(f"wrote summary             -> {summary_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
