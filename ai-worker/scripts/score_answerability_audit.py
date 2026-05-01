"""Phase 7.7 / 7.7.1 — score a labelled audit file and render a report.

The CLI reads a labelled CSV / JSONL produced by
``scripts.export_answerability_audit`` (after a human reviewer fills
in label cells), routes it through
:mod:`eval.harness.answerability_audit` for validation + metric
computation, and writes a Markdown report. An optional ``--json-path``
also writes the structured summary the harness uses internally —
useful when wiring this into a downstream comparison or dashboard.

Three modes:

  * ``--mode row`` (default) — score the row-level (Phase 7.7)
    labelled file. Backwards-compatible: ``--labeled-path`` is the
    Phase 7.7 flag name and continues to work as an alias for
    ``--labeled-row-path``.
  * ``--mode bundle`` — score the bundle-level (Phase 7.7.1) labelled
    file. Reads ``--labeled-bundle-path``.
  * ``--mode combined`` — score both row and bundle labelled files
    in one pass and render a combined report. Both
    ``--labeled-row-path`` (or ``--labeled-path``) and
    ``--labeled-bundle-path`` are required.

This is a pure scoring + report-rendering script. It does not modify
any production config, does not pick a winning variant, and does not
attempt to compare against a baseline retrieval-side metric. The
report's interpretation guide explicitly tells reviewers not to read
``answerable@k`` / ``context_answerable@k`` as a hit@k replacement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from eval.harness.answerability_audit import (
    AnswerabilityLabeledRow,
    AnswerabilityValidationError,
    ContextBundleAuditRow,
    build_json_summary,
    read_bundle_labeled_csv,
    read_bundle_labeled_jsonl,
    read_labeled_csv,
    read_labeled_jsonl,
    render_markdown_report,
)


log = logging.getLogger("scripts.score_answerability_audit")


SUPPORTED_MODES = ("row", "bundle", "combined")


def _load_labeled_row(path: Path) -> List[AnswerabilityLabeledRow]:
    """Pick the row-level reader based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_labeled_csv(path)
    if suffix == ".jsonl":
        return read_labeled_jsonl(path)
    raise SystemExit(
        f"row labelled-path must end in .csv or .jsonl, "
        f"got {path.name!r}"
    )


def _load_labeled_bundle(path: Path) -> List[ContextBundleAuditRow]:
    """Pick the bundle reader based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_bundle_labeled_csv(path)
    if suffix == ".jsonl":
        return read_bundle_labeled_jsonl(path)
    raise SystemExit(
        f"bundle labelled-path must end in .csv or .jsonl, "
        f"got {path.name!r}"
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.7 / 7.7.1 — score a labelled answerability file and "
        "render a Markdown report. --mode row keeps the Phase 7.7 "
        "row-level scoring (default for backwards compat). --mode "
        "bundle scores the Phase 7.7.1 bundle-level file. --mode "
        "combined scores both files and renders a single combined "
        "report with a row-vs-bundle Δ table. Optional --json-path "
        "emits the structured summary alongside the report."
    ))
    p.add_argument(
        "--mode", choices=SUPPORTED_MODES, default="row",
        help=(
            "Scoring mode. row = labelled chunks (Phase 7.7); "
            "bundle = labelled bundles (Phase 7.7.1); "
            "combined = both files merged into one report."
        ),
    )

    # Row-level path. Two names: --labeled-path is the Phase 7.7 flag
    # (kept for backwards compat); --labeled-row-path is the
    # disambiguated name added in 7.7.1.
    p.add_argument(
        "--labeled-row-path", type=Path, default=None,
        help=(
            "Row-level labelled CSV / JSONL. Required for --mode row "
            "and --mode combined."
        ),
    )
    p.add_argument(
        "--labeled-path", type=Path, default=None,
        help=(
            "Backwards-compat alias for --labeled-row-path "
            "(Phase 7.7 flag name)."
        ),
    )
    p.add_argument(
        "--labeled-bundle-path", type=Path, default=None,
        help=(
            "Bundle-level labelled CSV / JSONL. Required for --mode "
            "bundle and --mode combined."
        ),
    )

    p.add_argument(
        "--report-path", type=Path, required=True,
        help="Output Markdown report path. Parent dir auto-created.",
    )
    p.add_argument(
        "--json-path", type=Path, default=None,
        help=(
            "Optional structured JSON summary path (matches the "
            "report's section order). Useful for downstream diffs."
        ),
    )
    p.add_argument(
        "--title", type=str, default=None,
        help=(
            "Override the H1 title. Default depends on mode: "
            "row → 'Phase 7.7 — answerability audit'; "
            "bundle → 'Phase 7.7.1 — bundle answerability audit'; "
            "combined → 'Phase 7.7 / 7.7.1 — combined audit'."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _resolve_row_path(args: argparse.Namespace) -> Optional[Path]:
    """Return the effective row-level path or None.

    ``--labeled-row-path`` wins when both are supplied; the legacy
    ``--labeled-path`` alias kicks in only when the disambiguated
    name is unset. Callers handle the "still None when required" case.
    """
    if args.labeled_row_path is not None:
        return Path(args.labeled_row_path)
    if args.labeled_path is not None:
        return Path(args.labeled_path)
    return None


def _default_title(mode: str) -> str:
    if mode == "row":
        return "Phase 7.7 — answerability audit"
    if mode == "bundle":
        return "Phase 7.7.1 — bundle answerability audit"
    return "Phase 7.7 / 7.7.1 — combined audit"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    row_path = _resolve_row_path(args)
    bundle_path = (
        Path(args.labeled_bundle_path)
        if args.labeled_bundle_path is not None
        else None
    )

    if args.mode == "row":
        if row_path is None:
            raise SystemExit(
                "--mode row requires --labeled-row-path "
                "(or its alias --labeled-path)"
            )
    elif args.mode == "bundle":
        if bundle_path is None:
            raise SystemExit(
                "--mode bundle requires --labeled-bundle-path"
            )
    elif args.mode == "combined":
        if row_path is None or bundle_path is None:
            raise SystemExit(
                "--mode combined requires both "
                "--labeled-row-path (or --labeled-path) and "
                "--labeled-bundle-path"
            )

    rows: List[AnswerabilityLabeledRow] = []
    bundle_rows: List[ContextBundleAuditRow] = []

    try:
        if row_path is not None and args.mode in {"row", "combined"}:
            log.info("loading row labelled file %s", row_path)
            rows = _load_labeled_row(row_path)
        if bundle_path is not None and args.mode in {
            "bundle", "combined",
        }:
            log.info("loading bundle labelled file %s", bundle_path)
            bundle_rows = _load_labeled_bundle(bundle_path)
    except AnswerabilityValidationError as ex:
        raise SystemExit(f"validation error: {ex}")
    except FileNotFoundError as ex:
        raise SystemExit(str(ex))

    log.info(
        "loaded %d row(s) and %d bundle row(s); rendering %s report",
        len(rows), len(bundle_rows), args.mode,
    )

    title = args.title or _default_title(args.mode)
    report_md = render_markdown_report(
        rows, title=title,
        bundle_rows=bundle_rows or None,
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    log.info("wrote markdown report: %s", report_path)

    if args.json_path is not None:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        summary = build_json_summary(
            rows, bundle_rows=bundle_rows or None,
        )
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("wrote structured json summary: %s", json_path)

    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    sys.exit(main())
