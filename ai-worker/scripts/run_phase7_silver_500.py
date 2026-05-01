"""Phase 7 silver-500 query set — orchestrator CLI.

Renders the expanded silver query set + summary report described by
:mod:`eval.harness.silver_500_generator`. Reads ``pages_v4.jsonl`` (the
Phase 6.3 page records) and writes three sibling files under the
configured output directory::

  queries_v4_silver_500.jsonl
  queries_v4_silver_500.summary.json
  queries_v4_silver_500.summary.md

The output is **silver**, not human-verified gold — the report leads
with the silver/gold disclaimer demanded by the spec.

Usage::

    python -m scripts.run_phase7_silver_500 \\
        --pages-v4 PATH/pages_v4.jsonl \\
        --out-dir  eval/reports/<run-name>/

Per-bucket targets default to (main_work=150, subpage_generic=200,
subpage_named=150). Override with ``--main-work-target / --subpage-
generic-target / --subpage-named-target`` when running against a
smaller corpus subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from eval.harness.silver_500_generator import (
    BUCKET_MAIN_WORK,
    BUCKET_SUBPAGE_GENERIC,
    BUCKET_SUBPAGE_NAMED,
    DEFAULT_BUCKET_TARGETS,
    generate_silver_500,
    write_silver_500_outputs,
)


log = logging.getLogger("scripts.run_phase7_silver_500")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 7 silver-500 query generator. Reads Phase 6.3 "
            "pages_v4.jsonl and writes a stratified silver query set "
            "(approximately 500 rows) plus a JSON / Markdown summary. "
            "Output is silver, NOT human-verified gold."
        ),
    )
    p.add_argument(
        "--pages-v4", type=Path, required=True,
        help="Phase 6.3 pages_v4.jsonl input.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory where the three output files are written.",
    )
    p.add_argument(
        "--main-work-target", type=int,
        default=DEFAULT_BUCKET_TARGETS[BUCKET_MAIN_WORK],
        help="Target row count for the main_work bucket.",
    )
    p.add_argument(
        "--subpage-generic-target", type=int,
        default=DEFAULT_BUCKET_TARGETS[BUCKET_SUBPAGE_GENERIC],
        help="Target row count for the subpage_generic bucket.",
    )
    p.add_argument(
        "--subpage-named-target", type=int,
        default=DEFAULT_BUCKET_TARGETS[BUCKET_SUBPAGE_NAMED],
        help="Target row count for the subpage_named bucket.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed used by the stratified sampler.",
    )
    p.add_argument(
        "--restrict-doc-ids", type=Path, default=None,
        help=(
            "Optional newline-separated list of page_ids — when set, "
            "the generator only considers pages whose ``page_id`` is in "
            "this list. Useful for re-running over a smoke subset."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_restrict(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    out: List[str] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if s:
                out.append(s)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bucket_targets = {
        BUCKET_MAIN_WORK: int(args.main_work_target),
        BUCKET_SUBPAGE_GENERIC: int(args.subpage_generic_target),
        BUCKET_SUBPAGE_NAMED: int(args.subpage_named_target),
    }
    target_total = sum(bucket_targets.values())

    log.info(
        "generating silver-500 from pages_v4=%s seed=%d targets=%s",
        args.pages_v4, args.seed, bucket_targets,
    )
    restrict = _load_restrict(args.restrict_doc_ids)
    if restrict:
        log.info("restricting to %d page_ids", len(restrict))

    result = generate_silver_500(
        args.pages_v4,
        target_total=target_total,
        bucket_targets=bucket_targets,
        seed=int(args.seed),
        restrict_doc_ids=restrict,
    )

    out_paths = write_silver_500_outputs(result, out_dir=args.out_dir)
    log.info("Phase 7 silver-500 generation finished. Outputs:")
    for role, p in out_paths.items():
        log.info("  %s -> %s", role, p)

    summary = result.summary
    log.info(
        "Result: actual_total=%d (target=%d), bucket_actual=%s, "
        "deficits=%s",
        summary.get("actual_total"),
        summary.get("requested_total"),
        summary.get("bucket_actual_counts"),
        summary.get("bucket_deficits"),
    )
    if any(int(d) > 0 for d in (summary.get("bucket_deficits") or {}).values()):
        log.warning(
            "One or more buckets did not reach their target — see "
            "summary.md for details. This is expected when the corpus "
            "is small or filtered by --restrict-doc-ids."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
