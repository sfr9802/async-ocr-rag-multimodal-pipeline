"""Phase 7.6 — section-aware rerank experiment CLI (scaffold).

Drives the candidate-grid scaffolding from
``eval.harness.phase7_section_aware_rerank``. The current scope of
this CLI is intentionally narrow — *land the harness scaffolding,
not the experiment results* — so the immediate output is just the
candidate grid + the guardrail summary as JSON / Markdown.

When the Phase 7.6 work is ready to actually run the sweep, the
``--score`` flag will be wired up to:

  1. read the cached candidate pool from Phase 7.5
     (``--gold-pool-results``, ``--silver-pool-results``)
  2. run each grid variant via ``run_variant_for_query``
  3. score each variant against the gold / silver datasets through
     the existing Phase 7 metric harness
  4. write a Phase 7.6 confirm sweep report and a Phase 7.6
     production-recommended bundle following the same artefact
     contract as Phase 7.5

For now, calling this without ``--score`` writes:

  * ``section_rerank_grid.json`` — full grid spec + guardrails
  * ``section_rerank_grid.md`` — human-readable grid table

so a reviewer can check the strategy list before the actual
experiment lands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from eval.harness.phase7_section_aware_rerank import (
    DEFAULT_SECTION_BONUS_VALUES,
    SectionRerankSpec,
    make_section_rerank_grid,
    write_section_rerank_grid_json,
    write_section_rerank_grid_md,
)


log = logging.getLogger("scripts.run_phase7_section_aware_rerank")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.6 section-aware rerank experiment CLI (scaffold). "
        "Lands the candidate grid + guardrail summary; the full "
        "scoring sweep is wired in via --score in a follow-up."
    ))
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output directory for the grid spec / scoring artefacts.",
    )
    p.add_argument(
        "--section-bonus-values", type=str,
        default=",".join(f"{x:.2f}" for x in DEFAULT_SECTION_BONUS_VALUES),
        help=(
            "Comma-separated list of section_bonus values to include "
            "in the diagnostic ``section_bonus`` strategy variants."
        ),
    )
    p.add_argument(
        "--no-baseline", action="store_true",
        help="Skip the Phase 7.5 baseline row in the grid.",
    )
    p.add_argument(
        "--no-supporting-chunk-proximity", action="store_true",
        help="Skip the supporting-chunk proximity diagnostic variant.",
    )
    p.add_argument(
        "--no-page-first", action="store_true",
        help="Skip the page-first then section rerank variant.",
    )
    p.add_argument(
        "--no-same-page-rerank", action="store_true",
        help="Skip the same-page chunk rerank variant.",
    )
    p.add_argument(
        "--score", action="store_true",
        help=(
            "Run the actual scoring sweep (NOT YET IMPLEMENTED — exits "
            "with a clear message). The grid + guardrail spec are "
            "still written."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    grid: List[SectionRerankSpec] = make_section_rerank_grid(
        section_bonus_values=_parse_float_list(args.section_bonus_values),
        include_baseline=not bool(args.no_baseline),
        include_supporting_chunk_proximity=(
            not bool(args.no_supporting_chunk_proximity)
        ),
        include_page_first=not bool(args.no_page_first),
        include_same_page_rerank=not bool(args.no_same_page_rerank),
    )

    grid_json = report_dir / "section_rerank_grid.json"
    grid_md = report_dir / "section_rerank_grid.md"
    write_section_rerank_grid_json(grid_json, grid)
    write_section_rerank_grid_md(grid_md, grid)
    log.info(
        "wrote Phase 7.6 grid scaffold: %d variants (%s, %s)",
        len(grid), grid_json, grid_md,
    )

    if args.score:
        log.error(
            "--score is not implemented yet. Phase 7.6 lands the "
            "scaffolding in this PR; the scoring sweep follows in a "
            "separate change once a reviewer signs off on the grid."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
