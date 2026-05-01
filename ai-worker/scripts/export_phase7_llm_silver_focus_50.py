"""CLI: export the 50-row focus subset from the LLM silver-500 set.

Reads ``queries_v4_llm_silver_500.jsonl`` and writes:

  - ``queries_v4_llm_silver_focus_50.jsonl``
  - ``queries_v4_llm_silver_focus_50.csv``
  - ``queries_v4_llm_silver_focus_50.md``

The 50-row picker uses the same audit-seed schema as the 100-row human
seed (so it carries the empty ``human_*`` columns), but with a smaller
cross-tab. See :mod:`eval.harness.llm_silver_focus_50` for the
distribution rationale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.llm_silver_focus_50 import build_focus50  # noqa: E402
from eval.harness.llm_silver_human_seed import write_outputs  # noqa: E402


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
    / "llm_silver_focus_50"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="path to queries_v4_llm_silver_500.jsonl",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="output directory for the focus-50 JSONL/CSV/MD",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    export = build_focus50(args.input, seed=int(args.seed))
    paths = write_outputs(
        export,
        out_dir=args.out_dir,
        base_name="queries_v4_llm_silver_focus_50",
        target_total=50,
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
