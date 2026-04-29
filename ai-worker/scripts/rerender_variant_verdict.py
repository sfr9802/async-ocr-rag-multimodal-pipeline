"""Re-render the variant-confirm sweep verdict + report from artifacts.

Eval-only / report-only. The Phase 2 variant confirm sweep persists
all per-query rows to ``per_query_results.jsonl`` and per-cell
summaries to ``summary.json`` — enough state to recompute the
verdict + comparison report without rerunning the (~3 hour)
encode + 1200-query loop.

Use when the verdict logic changes (e.g. spec D's wording is widened
to cover final-regressed cases) and you want to re-grade an existing
artifact bundle. The original ``summary.json`` /
``comparison_report.md`` are preserved verbatim; the rerendered
files land at ``summary.v2.json`` / ``comparison_report.v2.md`` so
the diff between the two verdict eras stays auditable.

Run::

    python -m scripts.rerender_variant_verdict \\
        --report-dir eval/reports/retrieval-embedding-text-variant-confirm-20260429-1940
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

log = logging.getLogger("rerender_variant_verdict")


_ANCHOR_VARIANT = "raw"
_OPTUNA_WINNER_LABEL = "optuna_winner_top8"


def _load_summary(report_dir: Path) -> Dict[str, Any]:
    return json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))


def _load_rows(report_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (report_dir / "per_query_results.jsonl").open(
        "r", encoding="utf-8",
    ) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed per_query_results row")
    return rows


def _summary_namespace(summary_dict: Dict[str, Any]) -> SimpleNamespace:
    """Wrap the JSON summary so ``compute_variant_deltas`` can read it.

    ``compute_variant_deltas`` accepts both Mapping and dataclass
    surfaces, but Mapping access is via .get on the top-level only.
    The nested dict fields (candidate_hit_rates, duplicate_doc_ratios,
    unique_doc_counts) are already plain dicts in the JSON, so we
    can wrap the whole thing in a SimpleNamespace and the reader
    helpers (``getattr`` path) will see them with the right names.
    """
    return SimpleNamespace(**summary_dict)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, required=True,
        help="Path to the report directory produced by "
             "scripts.confirm_embedding_text_variant.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="v2",
        help="Suffix for re-rendered artifacts (default: v2). "
             "Originals are never overwritten.",
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

    summary_payload = _load_summary(report_dir)
    rows = _load_rows(report_dir)
    log.info(
        "Loaded summary.json + %d per-query rows from %s",
        len(rows), report_dir,
    )

    from eval.harness.variant_comparison import (
        ANCHOR_VARIANT,
        TITLE_SECTION_VARIANT,
        TITLE_VARIANT,
        compute_variant_deltas,
        decide_variant_verdict,
    )

    runs = summary_payload.get("runs") or []
    # Index summaries by (variant, cell).
    summary_by_pair: Dict[Tuple[str, str], SimpleNamespace] = {}
    spec_by_cell: Dict[str, Any] = {}
    for run in runs:
        variant = str(run.get("variant"))
        cell = str(run.get("cell"))
        s_dict = run.get("summary") or {}
        summary_by_pair[(variant, cell)] = _summary_namespace(s_dict)
        if cell not in spec_by_cell:
            spec_by_cell[cell] = run.get("spec") or {}

    # Recompute deltas using the (possibly updated) variant-comparison
    # logic. The raw anchor pair is the baseline for each cell.
    deltas_by_pair: Dict[Tuple[str, str], Any] = {}
    for (variant, cell), summary_ns in summary_by_pair.items():
        raw_ns = summary_by_pair.get((ANCHOR_VARIANT, cell))
        if raw_ns is None:
            continue
        deltas_by_pair[(variant, cell)] = compute_variant_deltas(
            cell_label=cell,
            variant=variant,
            variant_summary=summary_ns,
            raw_summary=raw_ns,
        )

    # Per-cell verdicts (re-run with updated logic).
    cell_verdicts: Dict[str, Tuple[str, str]] = {}
    cells = sorted({cell for _, cell in summary_by_pair.keys()})
    for cell in cells:
        title_d = deltas_by_pair.get((TITLE_VARIANT, cell))
        ts_d = deltas_by_pair.get((TITLE_SECTION_VARIANT, cell))
        cell_verdicts[cell] = decide_variant_verdict(
            title_deltas=title_d,
            title_section_deltas=ts_d,
        )

    primary_verdict, primary_rationale = cell_verdicts.get(
        _OPTUNA_WINNER_LABEL,
        ("UNDETERMINED", "optuna_winner_top8 not in run."),
    )
    log.info(
        "Re-rendered verdict (primary cell %s): %s",
        _OPTUNA_WINNER_LABEL, primary_verdict,
    )

    # Persist rerendered summary.v2.json — preserves the original
    # ``run`` / ``stacks`` / ``runs`` blobs but rewrites ``verdict`` +
    # adds a ``rerendered`` provenance header.
    new_payload = dict(summary_payload)
    new_payload["verdict"] = {
        "label": primary_verdict,
        "rationale": primary_rationale,
        "per_cell": {
            cell: {"label": v[0], "rationale": v[1]}
            for cell, v in cell_verdicts.items()
        },
    }
    # Rewrite each run's deltas section with the recomputed deltas.
    for run in new_payload.get("runs") or []:
        key = (str(run.get("variant")), str(run.get("cell")))
        if key in deltas_by_pair:
            run["deltas"] = asdict(deltas_by_pair[key])
    new_payload["rerendered"] = {
        "schema": "phase2-embedding-text-variant-confirm.rerender.v1",
        "rerendered_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary": "summary.json",
        "verdict_module": "eval.harness.variant_comparison",
        "note": (
            "Re-rendered after the spec-D verdict trigger was widened "
            "to cover final-regressed cases (cand@K up + final not "
            "improving) — earlier draft required final flat. The "
            "underlying retrieval results are unchanged; only the "
            "verdict label may differ from summary.json."
        ),
    }
    out_summary = report_dir / f"summary.{args.out_suffix}.json"
    out_summary.write_text(
        json.dumps(new_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", out_summary)

    # Persist a small markdown footer the user can append to the
    # original comparison_report.md without rewriting its body.
    md: List[str] = []
    md.append(
        "# Re-rendered verdict (spec-D trigger widened "
        f"@ {datetime.now().isoformat(timespec='seconds')})"
    )
    md.append("")
    md.append(
        "An earlier draft of "
        "``eval.harness.variant_comparison._has_dense_pool_lift_only`` "
        "required ``final flat`` (|Δhit@5| < EPS) to trigger the "
        "``NEED_RERANKER_INPUT_AUDIT_FIRST`` verdict. Spec D's wording "
        "(\"cand@K는 개선되는데 final hit/MRR이 개선되지 않으면\") "
        "is broader — *not improving* covers both flat and regressed "
        "finals. The trigger was widened; the silver_200 result "
        "(cand@50 +0.045, final hit@5 -0.045) now correctly registers "
        "as case D rather than C."
    )
    md.append("")
    md.append("## Re-rendered verdict")
    md.append("")
    md.append(f"**{primary_verdict}** — {primary_rationale}")
    md.append("")
    md.append("## Per-cell verdicts (re-rendered)")
    md.append("")
    for cell, (label, why) in cell_verdicts.items():
        md.append(f"- `{cell}` → **{label}** — {why}")
    md.append("")
    md.append("## Re-graded deltas")
    md.append("")
    md.append(
        "| variant | cell | grade | Δhit@5 | Δmrr@10 | Δcand@50 | "
        "Δcand@100 | latRatioP95 |"
    )
    md.append(
        "|---|---|---|---:|---:|---:|---:|---:|"
    )

    def _fmt_signed(v: Any) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):+.4f}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    # Order: by cell, then by variant (raw, title, title_section).
    variant_order = ["raw", "title", "title_section"]
    for cell in cells:
        for variant in variant_order:
            d = deltas_by_pair.get((variant, cell))
            if d is None:
                continue
            md.append(
                f"| {variant} | {cell} | {d.grade} | "
                f"{_fmt_signed(d.delta_hit_at_5)} | "
                f"{_fmt_signed(d.delta_mrr_at_10)} | "
                f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
                f"{_fmt_signed(d.delta_candidate_hit_at_100)} | "
                f"{_fmt(d.latency_ratio_p95)} |"
            )
    md.append("")
    out_md = report_dir / f"comparison_report.{args.out_suffix}.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Wrote %s", out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
