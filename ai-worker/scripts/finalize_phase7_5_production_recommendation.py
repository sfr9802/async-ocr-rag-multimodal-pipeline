"""Phase 7.5 — render the production-recommended config artefacts.

The confirm sweep (``run_phase7_mmr_confirm_sweep.py``) produces the
metric-best winner via a deterministic lexicographic tie-break across
the (candidate_k × λ) grid. When the λ-row plateaus at the same
primary_score that tie-break value is essentially arbitrary — the
first λ in the plateau row wins by name, not by production fitness.

This finaliser turns that sweep result into a *promotion-target*
recommendation:

  * keeps the metric-best candidate_k (no change there)
  * picks a plateau-aware λ — defaults to the prior Phase 7.x best
    (λ=0.70) so the resulting PR is a single-knob flip ("turn MMR on,
    widen candidate_k") rather than "and λ moved 0.10 too"
  * writes ``best_config.production_recommended.{env,json}`` next to
    the existing ``best_config.confirmed.{env,json}`` so a reviewer
    can compare side-by-side
  * splices a ``## Production recommendation`` section into the
    existing ``confirm_sweep_report.md`` (idempotent — re-runs replace
    the existing block instead of duplicating it)

The script is *replay-only*: it reads the confirm sweep's persisted
artefacts (``confirm_sweep_summary.json``, ``confirm_sweep_results.jsonl``)
and writes new artefacts. It never touches FAISS, the embedder, or
the candidate pools. Re-runs are deterministic and idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.harness.phase7_human_gold_tune import (
    GoldSummary,
    SilverSummary,
)
from eval.harness.phase7_mmr_confirm_sweep import (
    PRODUCTION_PLATEAU_EPSILON,
    PRODUCTION_RECOMMENDED_LAMBDA,
    CandidateScore,
    ConfirmSweepResult,
    PlateauAnalysis,
    SweepCandidate,
    append_production_recommendation_to_report,
    select_production_recommended_lambda,
    write_production_recommended_config_env,
    write_production_recommended_config_json,
)


log = logging.getLogger("scripts.finalize_phase7_5_production_recommendation")


# ---------------------------------------------------------------------------
# Deserialisation helpers — the saved JSON / JSONL files are just
# ``dataclasses.asdict`` output, so we round-trip via the constructor.
# ---------------------------------------------------------------------------


def _gold_summary_from_dict(payload: Dict[str, Any]) -> GoldSummary:
    """Round-trip a dict produced by ``gold_summary_to_dict`` back to
    a :class:`GoldSummary`. Nested by_* dicts are passed through as-is."""
    fields = {k: payload.get(k) for k in (
        "n_total", "n_strict_positive", "n_soft_positive",
        "n_ambiguous_probe", "n_abstain_test",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "mrr_at_10", "ndcg_at_10",
        "weighted_hit_at_1", "weighted_hit_at_3", "weighted_hit_at_5",
        "weighted_hit_at_10", "weighted_mrr_at_10", "weighted_ndcg_at_10",
        "strict_hit_at_5", "strict_mrr_at_10",
        "section_hit_at_5_when_defined", "section_hit_at_10_when_defined",
        "chunk_hit_at_10_when_defined", "primary_score",
        "by_bucket", "by_query_type",
        "by_normalized_group", "by_leakage_risk",
    )}
    # Coerce ints + floats from JSON numbers; tolerate missing keys.
    int_keys = {
        "n_total", "n_strict_positive", "n_soft_positive",
        "n_ambiguous_probe", "n_abstain_test",
    }
    for k in int_keys:
        if fields.get(k) is None:
            fields[k] = 0
        else:
            fields[k] = int(fields[k])
    for k in (
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "mrr_at_10", "ndcg_at_10",
        "weighted_hit_at_1", "weighted_hit_at_3", "weighted_hit_at_5",
        "weighted_hit_at_10", "weighted_mrr_at_10", "weighted_ndcg_at_10",
        "strict_hit_at_5", "strict_mrr_at_10", "primary_score",
    ):
        if fields.get(k) is None:
            fields[k] = 0.0
        else:
            fields[k] = float(fields[k])
    for k in ("by_bucket", "by_query_type",
              "by_normalized_group", "by_leakage_risk"):
        if fields.get(k) is None:
            fields[k] = {}
    return GoldSummary(**fields)


def _silver_summary_from_dict(payload: Dict[str, Any]) -> SilverSummary:
    fields = {k: payload.get(k) for k in (
        "n_total", "n_scored",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10",
        "by_bucket", "by_query_type",
        "by_leakage_risk", "by_overlap_risk",
    )}
    int_keys = {"n_total", "n_scored"}
    for k in int_keys:
        if fields.get(k) is None:
            fields[k] = 0
        else:
            fields[k] = int(fields[k])
    for k in (
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10",
    ):
        if fields.get(k) is None:
            fields[k] = 0.0
        else:
            fields[k] = float(fields[k])
    for k in ("by_bucket", "by_query_type",
              "by_leakage_risk", "by_overlap_risk"):
        if fields.get(k) is None:
            fields[k] = {}
    return SilverSummary(**fields)


def _candidate_score_from_dict(payload: Dict[str, Any]) -> CandidateScore:
    """Reconstruct a CandidateScore including warning records.

    Warnings are reduced to plain dicts on disk; the production-
    recommendation logic only needs the primary fields, so we keep them
    as dicts on the round-trip ``warnings`` attribute. The downstream
    callers never re-trigger guardrail evaluation off this path.
    """
    from eval.harness.phase7_human_gold_tune import GuardrailWarning
    warnings_raw = payload.get("warnings") or []
    warnings: List[GuardrailWarning] = []
    for w in warnings_raw:
        warnings.append(GuardrailWarning(
            code=str(w.get("code") or ""),
            metric=str(w.get("metric") or ""),
            bucket=w.get("bucket"),
            baseline=float(w.get("baseline") or 0.0),
            candidate=float(w.get("candidate") or 0.0),
            delta=float(w.get("delta") or 0.0),
            threshold=float(w.get("threshold") or 0.0),
            message=str(w.get("message") or ""),
        ))
    return CandidateScore(
        name=str(payload["name"]),
        candidate_k=int(payload["candidate_k"]),
        mmr_lambda=float(payload["mmr_lambda"]),
        primary_score=float(payload["primary_score"]),
        weighted_hit_at_5=float(payload.get("weighted_hit_at_5", 0.0)),
        weighted_mrr_at_10=float(payload.get("weighted_mrr_at_10", 0.0)),
        weighted_ndcg_at_10=float(payload.get("weighted_ndcg_at_10", 0.0)),
        silver_hit_at_5=float(payload.get("silver_hit_at_5", 0.0)),
        main_work_weighted_hit_at_5=(
            None if payload.get("main_work_weighted_hit_at_5") is None
            else float(payload["main_work_weighted_hit_at_5"])
        ),
        subpage_named_weighted_hit_at_5=(
            None if payload.get("subpage_named_weighted_hit_at_5") is None
            else float(payload["subpage_named_weighted_hit_at_5"])
        ),
        subpage_generic_weighted_hit_at_5=(
            None if payload.get("subpage_generic_weighted_hit_at_5") is None
            else float(payload["subpage_generic_weighted_hit_at_5"])
        ),
        section_hit_at_5=(
            None if payload.get("section_hit_at_5") is None
            else float(payload["section_hit_at_5"])
        ),
        deltas=dict(payload.get("deltas") or {}),
        warnings=warnings,
        accepted=bool(payload.get("accepted", False)),
        rejection_reason=str(payload.get("rejection_reason") or ""),
    )


def _sweep_candidate_from_dict(payload: Dict[str, Any]) -> SweepCandidate:
    return SweepCandidate(
        name=str(payload["name"]),
        candidate_k=int(payload["candidate_k"]),
        mmr_lambda=float(payload["mmr_lambda"]),
        top_k=int(payload.get("top_k") or 10),
        use_mmr=bool(payload.get("use_mmr", True)),
        cache_dir_relative=str(payload.get("cache_dir_relative") or ""),
        rag_chunks_path_relative=str(
            payload.get("rag_chunks_path_relative") or ""
        ),
        description=str(payload.get("description") or ""),
    )


def _plateau_from_dict(
    payload: Optional[Dict[str, Any]],
) -> Optional[PlateauAnalysis]:
    if not payload:
        return None
    neighbours_raw = payload.get("neighbours") or []
    neighbours = tuple(
        (float(n[0]), float(n[1])) for n in neighbours_raw
    )
    return PlateauAnalysis(
        status=str(payload["status"]),
        best_variant=str(payload["best_variant"]),
        best_primary_score=float(payload["best_primary_score"]),
        candidate_k=int(payload["candidate_k"]),
        mmr_lambda=float(payload["mmr_lambda"]),
        neighbours=neighbours,
        epsilon=float(payload.get("epsilon") or 0.0),
        message=str(payload.get("message") or ""),
    )


def load_confirm_sweep_summary(path: Path) -> ConfirmSweepResult:
    """Round-trip a saved ``confirm_sweep_summary.json`` back to its
    in-memory :class:`ConfirmSweepResult`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        _candidate_score_from_dict(c) for c in payload.get("candidates") or []
    ]
    grid = [
        _sweep_candidate_from_dict(g) for g in payload.get("grid") or []
    ]
    confirmed_best_payload = payload.get("confirmed_best")
    confirmed_best = (
        _candidate_score_from_dict(confirmed_best_payload)
        if confirmed_best_payload else None
    )
    plateau = _plateau_from_dict(payload.get("plateau"))
    return ConfirmSweepResult(
        baseline_name=str(payload["baseline_name"]),
        baseline_primary_score=float(payload["baseline_primary_score"]),
        candidates=candidates,
        confirmed_best=confirmed_best,
        plateau=plateau,
        promotion_recommended=bool(payload.get("promotion_recommended", False)),
        promotion_reason=str(payload.get("promotion_reason") or ""),
        grid=grid,
    )


def load_baseline_summaries_from_jsonl(
    path: Path,
) -> Tuple[GoldSummary, SilverSummary]:
    """Pull the baseline gold + silver summaries from
    ``confirm_sweep_results.jsonl`` (first row, role=baseline)."""
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("role") == "baseline":
                gold = _gold_summary_from_dict(row["gold_summary"])
                silver = _silver_summary_from_dict(row["silver_summary"])
                return gold, silver
    raise SystemExit(
        f"no baseline row (role=baseline) found in {path}; cannot reconstruct "
        f"baseline summaries for the production-recommendation pass."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Render the production-recommended config bundle from a finished "
        "Phase 7.5 confirm sweep. Plateau-aware lambda policy: when the "
        "metric-best λ is on a plateau, recommend the prior Phase 7.x "
        "best λ (default 0.70) instead of the lexicographic tie-break."
    ))
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Confirm-sweep report directory (contains "
             "confirm_sweep_summary.json + confirm_sweep_results.jsonl).",
    )
    p.add_argument(
        "--prefer-lambda", type=float,
        default=PRODUCTION_RECOMMENDED_LAMBDA,
        help=(
            "Production-recommended λ when the metric-best is on a "
            "plateau. Default 0.70 (matches the prior Phase 7.x best)."
        ),
    )
    p.add_argument(
        "--plateau-epsilon", type=float,
        default=PRODUCTION_PLATEAU_EPSILON,
        help=(
            "Absolute primary_score window for plateau detection in the "
            "winner's λ-row. Default mirrors the confirm sweep's "
            "PLATEAU_PRIMARY_DELTA."
        ),
    )
    p.add_argument(
        "--report-md", type=Path, default=None,
        help=(
            "Override path to the confirm_sweep_report.md to splice the "
            "production-recommendation section into. Defaults to "
            "<report-dir>/confirm_sweep_report.md."
        ),
    )
    p.add_argument(
        "--skip-report-update", action="store_true",
        help="Do not splice the section into confirm_sweep_report.md.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report_dir = Path(args.report_dir)
    summary_path = report_dir / "confirm_sweep_summary.json"
    results_jsonl = report_dir / "confirm_sweep_results.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    if not results_jsonl.exists():
        raise SystemExit(f"missing {results_jsonl}")

    log.info("loading confirm sweep summary from %s", summary_path)
    result = load_confirm_sweep_summary(summary_path)
    if result.confirmed_best is None:
        log.warning(
            "confirm sweep had no metric-best winner — nothing to "
            "promote; skipping production-recommendation render."
        )
        return 0

    log.info("loading baseline gold/silver summaries from %s", results_jsonl)
    base_gold, base_silver = load_baseline_summaries_from_jsonl(results_jsonl)

    recommendation = select_production_recommended_lambda(
        result=result,
        baseline_summary_gold=base_gold,
        baseline_summary_silver=base_silver,
        prefer_lambda=float(args.prefer_lambda),
        plateau_epsilon=float(args.plateau_epsilon),
    )
    if recommendation is None:
        log.warning(
            "select_production_recommended_lambda returned None — "
            "no recommendation rendered."
        )
        return 0

    env_path = report_dir / "best_config.production_recommended.env"
    json_path = report_dir / "best_config.production_recommended.json"
    write_production_recommended_config_env(
        env_path, recommendation=recommendation,
    )
    write_production_recommended_config_json(
        json_path, recommendation=recommendation,
    )
    log.info("wrote %s", env_path)
    log.info("wrote %s", json_path)

    if not args.skip_report_update:
        report_md = (
            Path(args.report_md) if args.report_md is not None
            else report_dir / "confirm_sweep_report.md"
        )
        if not report_md.exists():
            log.warning(
                "%s does not exist — skipping report splice.",
                report_md,
            )
        else:
            base_section_hit = float(
                base_gold.section_hit_at_5_when_defined or 0.0
            )
            cand_section_hit = float(
                result.confirmed_best.section_hit_at_5 or 0.0
            )
            append_production_recommendation_to_report(
                report_md,
                recommendation=recommendation,
                section_caveat_section_hit_at_5=(
                    base_section_hit, cand_section_hit,
                ),
            )
            log.info("spliced production recommendation into %s", report_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
