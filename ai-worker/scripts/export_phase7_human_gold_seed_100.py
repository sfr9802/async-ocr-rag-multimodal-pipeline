"""CLI: export the 100-row human-audit seed off the LLM silver-500 set.

Reads ``queries_v4_llm_silver_500.jsonl`` and writes:

  - ``phase7_human_gold_seed_100.jsonl``
  - ``phase7_human_gold_seed_100.csv``
  - ``phase7_human_gold_seed_100.md``

Stratified picks (default targets per the spec):

  bucket          → main_work=25, subpage_generic=35, subpage_named=35,
                    not_in_corpus=5
  query_type      → direct_title=10, paraphrase=25, section_intent=20,
                    indirect_entity=20, alias_variant=10, ambiguous=10,
                    unanswerable=5

Determinism: a fixed seed shuffles each (bucket, query_type) candidate
list. Two runs on the same input yield byte-identical files.

The seed file carries the silver expected target verbatim; the
reviewer's verdict goes into the empty ``human_*`` columns. NO row
in the seed is gold until ``human_label`` is populated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.llm_silver_human_seed import (  # noqa: E402
    HumanSeedConfig,
    build_human_seed,
    write_outputs,
)


DEFAULT_INPUT = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "seeds"
    / "llm_silver_500"
    / "queries_v4_llm_silver_500.jsonl"
)
DEFAULT_OUT_DIR = (
    _REPO_ROOT
    / "eval"
    / "reports"
    / "phase7"
    / "seeds"
    / "human_gold_seed_100"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="path to queries_v4_llm_silver_500.jsonl",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="output directory for phase7_human_gold_seed_100.* files",
    )
    ap.add_argument(
        "--target-total", type=int, default=100,
        help="number of rows to sample (default 100)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    cfg = HumanSeedConfig(
        target_total=int(args.target_total),
        seed=int(args.seed),
    ).validate()

    export = build_human_seed(args.input, config=cfg)
    paths = write_outputs(
        export,
        out_dir=args.out_dir,
        base_name=f"phase7_human_gold_seed_{cfg.target_total}",
        target_total=cfg.target_total,
    )

    audit = export.audit
    print(f"actual_total: {audit.get('actual_total')}")
    print(f"bucket_actual: {audit.get('bucket_actual')}")
    print(f"bucket_deficits: {audit.get('bucket_deficits')}")
    print(f"qt_actual: {audit.get('query_type_actual')}")
    print(f"qt_deficits: {audit.get('query_type_deficits')}")
    for key, p in paths.items():
        print(f"wrote {key:>5s}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
