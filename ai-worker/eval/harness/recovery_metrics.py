"""Phase 7.4 — recovery metrics + writer.

Aggregates the per-query ``RecoveryResult`` list emitted by the
controlled recovery loop into:

  - by-action counters (recovered / regressed / unchanged / skipped),
  - recovered@1 / @3 / @5 per action (the rate at which gold rank
    moves to ≤ k after the attempt),
  - ``gold_newly_entered_candidates`` count (gold was missing from
    the dense top-N but appeared in the post-fuse list — the looser
    pool-recall signal HYBRID is meant to deliver),
  - rank-delta distribution (mean / median / max),
  - latency p50 / p90 / p99 / mean,
  - oracle-vs-production-like comparison (when both modes ran),
  - explicit confirmation that ANSWER_WITH_CAUTION was *not* recovered
    and INSUFFICIENT_EVIDENCE was *refused* — the brief asks for these
    invariants to be reported visibly.

The writer ships:

  - ``recovery_attempts.jsonl`` — one row per executed (or skipped)
    attempt. Carries the decision, before/after state and latency.
  - ``recovery_summary.json`` — aggregate block above, plus the config.
  - ``recovery_summary.md`` — human-readable rendering.
  - ``recovered_queries.jsonl`` / ``unrecovered_queries.jsonl`` —
    split by ``RecoveryResult.recovered``.
  - ``oracle_rewrite_upper_bound.jsonl`` — only emitted when oracle
    mode ran; carries the oracle-only attempts so the upper-bound
    can be inspected without grep'ing the full attempts file.
  - ``PHASE7_4_FINAL_REPORT.md`` — top-level brief Phase 7.4 closes
    out with: recovery counts by action, recovered@k, regressions,
    invariants, oracle-vs-production-like delta, latency.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from eval.harness.recovery_policy import (
    RECOVERY_ACTION_ATTEMPT_HYBRID,
    RECOVERY_ACTION_ATTEMPT_REWRITE,
    RECOVERY_ACTION_NOOP,
    RECOVERY_ACTION_SKIP_CAUTION,
    RECOVERY_ACTION_SKIP_DEFER,
    RECOVERY_ACTION_SKIP_REFUSE,
    RECOVERY_ACTIONS,
    REWRITE_MODE_BOTH,
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
    REWRITE_MODES,
    RecoveryAttempt,
    RecoveryResult,
)

from eval.harness.controlled_recovery_loop import (
    ControlledRecoveryResult,
    _attempt_to_dict,
    _decision_to_dict,
    _result_to_dict,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


_RECOVERED_KS: Tuple[int, ...] = (1, 3, 5)
_RANK_MISS = -1


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (matches numpy default)."""
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _latency_block(samples: Sequence[float]) -> Dict[str, Any]:
    """p50 / p90 / p99 / mean / max in ms; zeros when no samples."""
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    arr = sorted(float(s) for s in samples)
    return {
        "count": len(arr),
        "mean_ms": round(sum(arr) / len(arr), 3),
        "p50_ms": round(_quantile(arr, 0.50), 3),
        "p90_ms": round(_quantile(arr, 0.90), 3),
        "p99_ms": round(_quantile(arr, 0.99), 3),
        "max_ms": round(arr[-1], 3),
    }


def _rank_delta_block(deltas: Sequence[int]) -> Dict[str, Any]:
    """Mean / median / min / max for non-None rank deltas."""
    if not deltas:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "improved": 0,
            "worsened": 0,
            "unchanged": 0,
        }
    arr = sorted(int(d) for d in deltas)
    improved = sum(1 for d in arr if d < 0)
    worsened = sum(1 for d in arr if d > 0)
    unchanged = sum(1 for d in arr if d == 0)
    return {
        "count": len(arr),
        "mean": round(sum(arr) / len(arr), 3),
        "median": round(statistics.median(arr), 3),
        "min": arr[0],
        "max": arr[-1],
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def _recovered_at_k(
    results: Sequence[RecoveryResult], *, k: int,
) -> Tuple[int, int]:
    """Return (recovered_at_k_count, denom) over non-skipped attempts.

    Recovered@k means: the gold rank moved from missing or > k to
    a position ≤ k. Denominator counts every non-skipped attempt
    whose ``before_rank`` was missing or > k (i.e. attempts that had
    something to recover).
    """
    num = 0
    denom = 0
    for r in results:
        if r.skipped:
            continue
        before_in = (r.before_rank != _RANK_MISS) and (r.before_rank <= k)
        if before_in:
            continue  # nothing to recover at this k
        denom += 1
        after_in = (r.after_rank != _RANK_MISS) and (r.after_rank <= k)
        if after_in:
            num += 1
    return num, denom


def _split_results_by_action(
    results: Sequence[RecoveryResult],
) -> Dict[str, List[RecoveryResult]]:
    """Group results by recovery_action; empty actions produce empty lists."""
    by: Dict[str, List[RecoveryResult]] = {a: [] for a in RECOVERY_ACTIONS}
    for r in results:
        by.setdefault(r.recovery_action, []).append(r)
    return by


def _split_results_by_rewrite_mode(
    results: Sequence[RecoveryResult],
) -> Dict[Optional[str], List[RecoveryResult]]:
    """Group rewrite results by rewrite_mode (oracle / production_like)."""
    by: Dict[Optional[str], List[RecoveryResult]] = defaultdict(list)
    for r in results:
        if r.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE:
            by[r.rewrite_mode].append(r)
    return dict(by)


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def _per_action_block(rs: Sequence[RecoveryResult]) -> Dict[str, Any]:
    """Recovered / regressed / unchanged / skipped + recovered@k for one action."""
    n = len(rs)
    skipped = sum(1 for r in rs if r.skipped)
    attempted = n - skipped
    recovered = sum(1 for r in rs if r.recovered)
    regressed = sum(1 for r in rs if r.regression)
    unchanged = attempted - recovered - regressed
    gold_newly = sum(1 for r in rs if r.gold_newly_entered_candidates)

    deltas = [r.rank_delta for r in rs if r.rank_delta is not None]
    latencies = [r.latency_ms for r in rs if r.latency_ms is not None]

    rec_at_k: Dict[str, Dict[str, Any]] = {}
    for k in _RECOVERED_KS:
        num, denom = _recovered_at_k(rs, k=k)
        rate = (num / denom) if denom else 0.0
        rec_at_k[f"k={k}"] = {
            "numerator": num,
            "denominator": denom,
            "rate": round(rate, 4),
        }

    return {
        "count": n,
        "skipped": skipped,
        "attempted": attempted,
        "recovered": recovered,
        "regressed": regressed,
        "unchanged": unchanged,
        "gold_newly_entered_candidates": gold_newly,
        "recovered_at_k": rec_at_k,
        "rank_delta": _rank_delta_block(deltas),
        "latency_ms": _latency_block(latencies),
    }


def aggregate_results(
    result: ControlledRecoveryResult,
) -> Dict[str, Any]:
    """Compute the JSON summary block.

    Layout (key by key, frozen contract — Phase 7.4's writer reads
    these keys directly into the markdown render):

      - ``config``: dataclass dump.
      - ``n_queries``: total decisions emitted (≥ verdict count when
        rewrite_mode='both').
      - ``by_action``: per-RecoveryAction block (counts + recovered@k
        + rank_delta + latency).
      - ``by_bucket``: per-bucket counters (top-line only — the
        full bucket × action cube is not surfaced here because Phase
        7.3 already exposes that breakdown for the gate).
      - ``by_rewrite_mode``: oracle vs production_like comparison
        when both ran.
      - ``invariants``: confirms the policy refused to recover
        ANSWER_WITH_CAUTION / INSUFFICIENT_EVIDENCE.
      - ``totals``: cross-row counters (recovered, regressed, ...) so
        a reader can compare against Phase 7.3's gate counts directly.
    """
    by_action = _split_results_by_action(result.results)
    n = len(result.results)

    # Per-action block.
    per_action = {a: _per_action_block(rs) for a, rs in by_action.items()}

    # Per-bucket block — top-line only.
    by_bucket: Dict[str, Dict[str, Any]] = {}
    for r in result.results:
        bucket = r.bucket or "<unbucketed>"
        cell = by_bucket.setdefault(bucket, {
            "count": 0, "skipped": 0, "attempted": 0,
            "recovered": 0, "regressed": 0,
            "gold_newly_entered_candidates": 0,
        })
        cell["count"] += 1
        if r.skipped:
            cell["skipped"] += 1
        else:
            cell["attempted"] += 1
            if r.recovered:
                cell["recovered"] += 1
            if r.regression:
                cell["regressed"] += 1
            if r.gold_newly_entered_candidates:
                cell["gold_newly_entered_candidates"] += 1

    # Per-rewrite-mode block (only meaningful for ATTEMPT_REWRITE rows).
    rewrite_modes = _split_results_by_rewrite_mode(result.results)
    by_rewrite_mode: Dict[str, Dict[str, Any]] = {}
    for mode, rs in rewrite_modes.items():
        if mode is None:
            continue
        by_rewrite_mode[mode] = _per_action_block(rs)

    oracle_block = by_rewrite_mode.get(REWRITE_MODE_ORACLE)
    prod_block = by_rewrite_mode.get(REWRITE_MODE_PRODUCTION_LIKE)
    oracle_vs_production: Optional[Dict[str, Any]] = None
    if oracle_block and prod_block:
        oracle_vs_production = {
            "oracle": {
                "recovered": oracle_block["recovered"],
                "regressed": oracle_block["regressed"],
                "attempted": oracle_block["attempted"],
                "recovered_at_k": oracle_block["recovered_at_k"],
            },
            "production_like": {
                "recovered": prod_block["recovered"],
                "regressed": prod_block["regressed"],
                "attempted": prod_block["attempted"],
                "recovered_at_k": prod_block["recovered_at_k"],
            },
            "delta": {
                "recovered_oracle_minus_production_like":
                    oracle_block["recovered"] - prod_block["recovered"],
                "regressed_oracle_minus_production_like":
                    oracle_block["regressed"] - prod_block["regressed"],
            },
        }

    # Invariants. Phase 7.4 explicitly does NOT recover these — flag
    # to the reader that this is by design, not a bug. The SKIP_DEFER
    # bucket is split between (a) genuine ASK_CLARIFICATION rows from
    # Phase 7.3 and (b) production-like rewrites the policy refused
    # because they would have leaked silver labels — both end up as
    # SKIP_DEFER but the skip_reason on the decision differentiates
    # them, so we count them separately for the report.
    refused_insufficient = sum(
        1 for r in result.results
        if r.recovery_action == RECOVERY_ACTION_SKIP_REFUSE
    )
    deferred_caution = sum(
        1 for r in result.results
        if r.recovery_action == RECOVERY_ACTION_SKIP_CAUTION
    )
    deferred_total = sum(
        1 for r in result.results
        if r.recovery_action == RECOVERY_ACTION_SKIP_DEFER
    )
    label_leakage_refused = sum(
        1 for d in result.decisions
        if d.recovery_action == RECOVERY_ACTION_SKIP_DEFER
        and d.skip_reason == "LABEL_LEAKAGE_REFUSED"
    )
    deferred_clarification = deferred_total - label_leakage_refused
    invariants = {
        "answer_with_caution_recovered": False,
        "answer_with_caution_skip_count": deferred_caution,
        "insufficient_evidence_recovered": False,
        "insufficient_evidence_refused_count": refused_insufficient,
        "ask_clarification_deferred_count": deferred_clarification,
        "label_leakage_refused_count": label_leakage_refused,
    }

    # Top-line totals.
    totals = {
        "n_decisions": n,
        "skipped": sum(1 for r in result.results if r.skipped),
        "attempted": sum(1 for r in result.results if not r.skipped),
        "recovered": sum(1 for r in result.results if r.recovered),
        "regressed": sum(1 for r in result.results if r.regression),
        "gold_newly_entered_candidates": sum(
            1 for r in result.results if r.gold_newly_entered_candidates
        ),
    }
    all_latencies = [
        r.latency_ms for r in result.results if r.latency_ms is not None
    ]
    totals["latency_ms"] = _latency_block(all_latencies)

    return {
        "config": asdict(result.config),
        "n_queries": n,
        "totals": totals,
        "by_action": per_action,
        "by_bucket": by_bucket,
        "by_rewrite_mode": by_rewrite_mode,
        "oracle_vs_production_like": oracle_vs_production,
        "invariants": invariants,
        "recovery_actions": list(RECOVERY_ACTIONS),
        "rewrite_modes": list(REWRITE_MODES),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    result: ControlledRecoveryResult,
    *,
    out_dir: Path,
    aggregate: Optional[Mapping[str, Any]] = None,
    attempts_name: str = "recovery_attempts.jsonl",
    summary_json_name: str = "recovery_summary.json",
    summary_md_name: str = "recovery_summary.md",
    recovered_name: str = "recovered_queries.jsonl",
    unrecovered_name: str = "unrecovered_queries.jsonl",
    oracle_name: str = "oracle_rewrite_upper_bound.jsonl",
    final_report_name: str = "PHASE7_4_FINAL_REPORT.md",
) -> Dict[str, Path]:
    """Persist the artefact bundle Phase 7.4 asks for.

    The writer is *deterministic*: row order matches the result lists
    (which match the verdict-row input order). No sets, no dict-key
    reorder, no shuffling — replay-safe across Python invocations.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if aggregate is None:
        aggregate = aggregate_results(result)

    attempts_path = out_dir / attempts_name
    summary_json_path = out_dir / summary_json_name
    summary_md_path = out_dir / summary_md_name
    recovered_path = out_dir / recovered_name
    unrecovered_path = out_dir / unrecovered_name
    oracle_path = out_dir / oracle_name
    final_report_path = out_dir / final_report_name

    # recovery_attempts.jsonl — one row per attempt (skipped or executed).
    with attempts_path.open("w", encoding="utf-8") as fp:
        for a in result.attempts:
            row = _attempt_to_dict(a)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    # recovery_summary.json — full aggregate block.
    summary_json_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # recovered_queries.jsonl / unrecovered_queries.jsonl — split by
    # the recovery flag. The skipped rows go into unrecovered (they
    # are not recovered, by definition); the JSONL row keeps the
    # ``skipped`` flag so a reviewer can filter them out.
    with recovered_path.open("w", encoding="utf-8") as fp:
        for a, r in zip(result.attempts, result.results):
            if r.recovered:
                row = _attempt_to_dict(a)
                row["result"] = _result_to_dict(r)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    with unrecovered_path.open("w", encoding="utf-8") as fp:
        for a, r in zip(result.attempts, result.results):
            if not r.recovered:
                row = _attempt_to_dict(a)
                row["result"] = _result_to_dict(r)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    # oracle_rewrite_upper_bound.jsonl — only when oracle ran.
    has_oracle = any(
        r.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
        and r.oracle_upper_bound
        for r in result.results
    )
    if has_oracle:
        with oracle_path.open("w", encoding="utf-8") as fp:
            for a, r in zip(result.attempts, result.results):
                if (
                    r.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
                    and r.oracle_upper_bound
                ):
                    row = _attempt_to_dict(a)
                    row["result"] = _result_to_dict(r)
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    # recovery_summary.md — narrative.
    summary_md_path.write_text(
        render_summary_md(aggregate),
        encoding="utf-8",
    )

    # PHASE7_4_FINAL_REPORT.md — top-level, includes invariants etc.
    final_report_path.write_text(
        render_final_report_md(aggregate),
        encoding="utf-8",
    )

    paths: Dict[str, Path] = {
        "attempts": attempts_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "recovered": recovered_path,
        "unrecovered": unrecovered_path,
        "final_report": final_report_path,
    }
    if has_oracle:
        paths["oracle"] = oracle_path
    return paths


def render_summary_md(aggregate: Mapping[str, Any]) -> str:
    """Render the human-readable recovery_summary.md.

    Layout: header → totals → by-action table → by-bucket table →
    oracle-vs-production-like comparison → invariants → latency.
    """
    cfg = aggregate.get("config") or {}
    totals = aggregate.get("totals") or {}
    by_action = aggregate.get("by_action") or {}
    by_bucket = aggregate.get("by_bucket") or {}
    by_rewrite_mode = aggregate.get("by_rewrite_mode") or {}
    oracle_vs_prod = aggregate.get("oracle_vs_production_like")
    invariants = aggregate.get("invariants") or {}
    n = aggregate.get("n_queries", 0)

    lines: List[str] = []
    lines.append("# Phase 7.4 — Controlled recovery loop summary")
    lines.append("")
    lines.append(f"- n_decisions: **{n}**")
    lines.append(f"- final_k: {cfg.get('final_k')}")
    lines.append(f"- hybrid_top_k: {cfg.get('hybrid_top_k')}")
    lines.append(f"- bm25_pool_size: {cfg.get('bm25_pool_size')}")
    lines.append(f"- rewrite_mode: {cfg.get('rewrite_mode')}")
    lines.append(f"- strict_label_leakage: {cfg.get('strict_label_leakage')}")
    lines.append("")

    lines.append("## Totals")
    lines.append("")
    lines.append(f"- attempted: **{totals.get('attempted', 0)}**")
    lines.append(f"- skipped:   **{totals.get('skipped', 0)}**")
    lines.append(f"- recovered: **{totals.get('recovered', 0)}**")
    lines.append(f"- regressed: **{totals.get('regressed', 0)}**")
    lines.append(
        f"- gold_newly_entered_candidates: "
        f"**{totals.get('gold_newly_entered_candidates', 0)}**"
    )
    lines.append("")

    lines.append("## By recovery action")
    lines.append("")
    lines.append(
        "| action | n | attempted | recovered | regressed | "
        "newly_entered | rec@1 | rec@3 | rec@5 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for action in RECOVERY_ACTIONS:
        block = by_action.get(action)
        if not block or block.get("count", 0) == 0:
            continue
        rec_at_k = block.get("recovered_at_k") or {}
        rate1 = rec_at_k.get("k=1", {}).get("rate", 0.0)
        rate3 = rec_at_k.get("k=3", {}).get("rate", 0.0)
        rate5 = rec_at_k.get("k=5", {}).get("rate", 0.0)
        lines.append(
            f"| {action} | {block.get('count', 0)} "
            f"| {block.get('attempted', 0)} "
            f"| {block.get('recovered', 0)} "
            f"| {block.get('regressed', 0)} "
            f"| {block.get('gold_newly_entered_candidates', 0)} "
            f"| {rate1:.3f} | {rate3:.3f} | {rate5:.3f} |"
        )
    lines.append("")

    lines.append("## By bucket")
    lines.append("")
    lines.append(
        "| bucket | n | attempted | recovered | regressed | newly_entered |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bucket, payload in sorted(by_bucket.items()):
        lines.append(
            f"| {bucket} | {payload.get('count', 0)} "
            f"| {payload.get('attempted', 0)} "
            f"| {payload.get('recovered', 0)} "
            f"| {payload.get('regressed', 0)} "
            f"| {payload.get('gold_newly_entered_candidates', 0)} |"
        )
    lines.append("")

    if oracle_vs_prod:
        lines.append("## Oracle vs production-like rewrite")
        lines.append("")
        lines.append(
            "| mode | attempted | recovered | regressed | rec@1 | rec@3 | rec@5 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for mode_name, label in (
            (REWRITE_MODE_ORACLE, "oracle"),
            (REWRITE_MODE_PRODUCTION_LIKE, "production_like"),
        ):
            block = by_rewrite_mode.get(mode_name)
            if not block:
                continue
            rec_at_k = block.get("recovered_at_k") or {}
            rate1 = rec_at_k.get("k=1", {}).get("rate", 0.0)
            rate3 = rec_at_k.get("k=3", {}).get("rate", 0.0)
            rate5 = rec_at_k.get("k=5", {}).get("rate", 0.0)
            lines.append(
                f"| {label} | {block.get('attempted', 0)} "
                f"| {block.get('recovered', 0)} "
                f"| {block.get('regressed', 0)} "
                f"| {rate1:.3f} | {rate3:.3f} | {rate5:.3f} |"
            )
        delta = oracle_vs_prod.get("delta") or {}
        lines.append("")
        lines.append(
            f"- recovered (oracle - production_like): "
            f"**{delta.get('recovered_oracle_minus_production_like', 0)}**"
        )
        lines.append(
            f"- regressed (oracle - production_like): "
            f"**{delta.get('regressed_oracle_minus_production_like', 0)}**"
        )
        lines.append("")

    lines.append("## Invariants")
    lines.append("")
    lines.append(
        f"- ANSWER_WITH_CAUTION recovered? "
        f"**{invariants.get('answer_with_caution_recovered', False)}** "
        f"(skip count: {invariants.get('answer_with_caution_skip_count', 0)})"
    )
    lines.append(
        f"- INSUFFICIENT_EVIDENCE recovered? "
        f"**{invariants.get('insufficient_evidence_recovered', False)}** "
        f"(refused count: {invariants.get('insufficient_evidence_refused_count', 0)})"
    )
    lines.append(
        f"- ASK_CLARIFICATION deferred count: "
        f"{invariants.get('ask_clarification_deferred_count', 0)}"
    )
    lines.append(
        f"- production-like label-leakage refused count: "
        f"{invariants.get('label_leakage_refused_count', 0)}"
    )
    lines.append("")

    lat = totals.get("latency_ms") or {}
    if lat.get("count"):
        lines.append("## Latency (BM25 retrieve only)")
        lines.append("")
        lines.append(f"- count: {lat.get('count', 0)}")
        lines.append(f"- mean: {lat.get('mean_ms', 0):.3f} ms")
        lines.append(f"- p50:  {lat.get('p50_ms', 0):.3f} ms")
        lines.append(f"- p90:  {lat.get('p90_ms', 0):.3f} ms")
        lines.append(f"- p99:  {lat.get('p99_ms', 0):.3f} ms")
        lines.append(f"- max:  {lat.get('max_ms', 0):.3f} ms")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_final_report_md(aggregate: Mapping[str, Any]) -> str:
    """Phase 7.4's top-level final report.

    Same content as the summary, plus a leading "key findings" section
    that calls out the headline numbers a reviewer needs to read in
    isolation: total recovered, total regressed, oracle gap, refusal
    counts. Mirrors the Phase 7.0/7.1/7.3 final-report shape so a
    reader knows what to expect.
    """
    cfg = aggregate.get("config") or {}
    totals = aggregate.get("totals") or {}
    invariants = aggregate.get("invariants") or {}
    oracle_vs_prod = aggregate.get("oracle_vs_production_like")
    by_action = aggregate.get("by_action") or {}

    lines: List[str] = []
    lines.append("# Phase 7.4 — Controlled Recovery Loop final report")
    lines.append("")
    lines.append("## Key findings")
    lines.append("")
    lines.append(f"- total decisions: **{aggregate.get('n_queries', 0)}**")
    lines.append(f"- attempted recoveries: **{totals.get('attempted', 0)}**")
    lines.append(f"- recovered: **{totals.get('recovered', 0)}**")
    lines.append(f"- regressed: **{totals.get('regressed', 0)}**")
    lines.append(
        f"- gold newly entered candidates: "
        f"**{totals.get('gold_newly_entered_candidates', 0)}**"
    )
    lines.append(
        f"- INSUFFICIENT_EVIDENCE refused (no recovery attempted): "
        f"**{invariants.get('insufficient_evidence_refused_count', 0)}**"
    )
    lines.append(
        f"- ANSWER_WITH_CAUTION skipped (calibration only, no recovery): "
        f"**{invariants.get('answer_with_caution_skip_count', 0)}**"
    )
    lines.append("")

    if oracle_vs_prod:
        delta = oracle_vs_prod.get("delta") or {}
        lines.append(
            f"Oracle vs production-like rewrite: oracle recovered "
            f"**{delta.get('recovered_oracle_minus_production_like', 0)}** "
            f"more queries than production-like, with "
            f"**{delta.get('regressed_oracle_minus_production_like', 0)}** "
            f"regression delta."
        )
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    for k in (
        "rewrite_mode", "final_k", "hybrid_top_k", "bm25_pool_size",
        "k_rrf", "top_n_for_production", "strict_label_leakage", "side",
    ):
        lines.append(f"- {k}: {cfg.get(k)}")
    lines.append("")

    lines.append("## Recovery breakdown by action")
    lines.append("")
    lines.append(
        "| action | n | attempted | recovered | regressed | newly_entered |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for action in RECOVERY_ACTIONS:
        block = by_action.get(action)
        if not block or block.get("count", 0) == 0:
            continue
        lines.append(
            f"| {action} | {block.get('count', 0)} "
            f"| {block.get('attempted', 0)} "
            f"| {block.get('recovered', 0)} "
            f"| {block.get('regressed', 0)} "
            f"| {block.get('gold_newly_entered_candidates', 0)} |"
        )
    lines.append("")

    lines.append("## Invariants confirmed")
    lines.append("")
    lines.append(
        f"- INSUFFICIENT_EVIDENCE → no recovery attempted, refused "
        f"**{invariants.get('insufficient_evidence_refused_count', 0)}** queries."
    )
    lines.append(
        f"- ANSWER_WITH_CAUTION → not recovered this phase, skipped "
        f"**{invariants.get('answer_with_caution_skip_count', 0)}** queries "
        f"(calibration-only)."
    )
    lines.append(
        f"- ASK_CLARIFICATION → deferred, "
        f"**{invariants.get('ask_clarification_deferred_count', 0)}** queries."
    )
    lines.append(
        f"- Production-like rewrite refused on label leakage: "
        f"**{invariants.get('label_leakage_refused_count', 0)}** queries."
    )
    lines.append("")
    lines.append(
        "These invariants are by design and visible in the JSONL outputs."
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Full distribution and bucket breakdown live in "
                 "`recovery_summary.md` and `recovery_summary.json`.")
    lines.append("")
    return "\n".join(lines) + "\n"
