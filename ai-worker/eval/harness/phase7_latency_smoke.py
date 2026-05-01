"""Phase 7.5 — latency smoke check harness.

Replays the cached candidate pool produced by the confirm sweep and
times the *post-hoc* slicing + MMR pass for each (candidate_k, λ,
use_mmr) config under test. The candidate-generation cost itself is
already measured live by the confirm sweep (``elapsed_ms`` per query
in ``candidate_pool_*.jsonl``); the smoke check sums the two so the
reviewer sees an end-to-end retrieval-pass estimate per config.

Honest scope:

  * **What is real**: ``candidate_gen_ms`` per query is the live
    FAISS+embed time the confirm sweep recorded at ``pool_size=40``.
    ``mmr_post_ms`` is timed live in this script. Both are real
    wall-clock numbers in the same Python process.
  * **What is approximate**: the candidate-gen pass was measured at
    ``pool_size=40`` — for ``candidate_k=20/30`` configs the live
    retriever would do *slightly* less work, so the numbers carried
    here are a small upper bound (FAISS NN lookup is O(log N + k)).
    The harness annotates this in the report.
  * **What is NOT measured**: the cross-encoder reranker's per-pass
    cost. The CI/eval environment runs with the NoOp reranker, so any
    reranker timing produced here would be an artefact of that
    fixture, not the production rerank pass. The harness emits the
    word "noop" in the rendered report so a reviewer cannot mistake
    it for a production-reranker measurement.

The aggregator is pure Python: hand it a list of per-query
``LatencyMeasurement`` rows and it returns a ``LatencyAggregate``
with mean / p50 / p90 / p99 per stage. The CLI in
``scripts.run_phase7_latency_smoke`` wires the I/O.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence,
    Tuple,
)

from eval.harness.phase7_human_gold_tune import RetrievedDoc
from eval.harness.phase7_mmr_confirm_sweep import (
    DEFAULT_TOP_K,
    PRODUCTION_INDEX_CACHE_DIR,
    apply_variant_to_candidates,
)


log = logging.getLogger(__name__)


# Number of micro-benchmark repetitions per query for the post-hoc MMR
# stage. The MMR call is sub-millisecond on a 40-doc pool so a single
# wall-clock sample is dominated by clock granularity; averaging over
# REPS smooths that without changing the order of magnitude. 5 is
# enough that the median is stable on typical CI hardware.
DEFAULT_MMR_REPS: int = 5


# Latency p-bands the report carries. Pinned by the test renderer
# regression test so a reviewer cannot accidentally drop p99 from the
# table.
DEFAULT_LATENCY_PERCENTILES: Tuple[float, ...] = (0.50, 0.90, 0.99)


# Suite roles — the smoke check reports per-role so a reader can tell
# at a glance which row corresponds to "what's shipping today" vs
# "what we want to ship". The strings are persisted in the JSON so
# do not rename without the test pin.
ROLE_BASELINE: str = "baseline"
ROLE_PREVIOUS_BEST: str = "previous_best"
ROLE_RECOMMENDED: str = "production_recommended"
ROLE_FALLBACK: str = "fallback"


# Default suite the CLI runs when ``--suite default`` is passed. Three
# rows: pre-promotion baseline (no MMR), Phase 7.x first-pass best
# (candidate_k=30, λ=0.7), Phase 7.5 production recommended
# (candidate_k=40, λ=0.7 plateau).
@dataclass(frozen=True)
class LatencySmokeConfig:
    """One config-under-test in the smoke suite."""

    name: str
    role: str
    top_k: int
    candidate_k: int
    use_mmr: bool
    mmr_lambda: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_smoke_suite() -> List[LatencySmokeConfig]:
    """The default 3-row suite: baseline / previous-best / recommended."""
    return [
        LatencySmokeConfig(
            name="baseline_top10",
            role=ROLE_BASELINE,
            top_k=DEFAULT_TOP_K,
            candidate_k=10,
            use_mmr=False,
            mmr_lambda=0.7,
            description=(
                "pre-promotion baseline (use_mmr=false, top_k=10). "
                "Mirrors the shipping retrieval path."
            ),
        ),
        LatencySmokeConfig(
            name="previous_best_candk30_lambda070",
            role=ROLE_PREVIOUS_BEST,
            top_k=DEFAULT_TOP_K,
            candidate_k=30,
            use_mmr=True,
            mmr_lambda=0.70,
            description=(
                "Phase 7.x first-pass best (candidate_k=30, MMR on, "
                "λ=0.70). Intermediate fallback config."
            ),
        ),
        LatencySmokeConfig(
            name="recommended_candk40_lambda070",
            role=ROLE_RECOMMENDED,
            top_k=DEFAULT_TOP_K,
            candidate_k=40,
            use_mmr=True,
            mmr_lambda=0.70,
            description=(
                "Phase 7.5 production-recommended (candidate_k=40, "
                "MMR on, λ=0.70 from the plateau-aware policy)."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Measurement records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyMeasurement:
    """One (config, query) latency record.

    ``candidate_gen_ms`` is the live FAISS+embed time the confirm sweep
    recorded for this query at ``pool_size=40``. For configs where
    candidate_k < 40 we treat that number as a small upper bound — see
    the module docstring's "honest scope" notes.

    ``mmr_post_ms`` is the time taken by the post-hoc MMR slicing
    pass against the cached pool, averaged over ``DEFAULT_MMR_REPS``
    repetitions to smooth clock granularity at sub-ms scales.

    ``total_ms`` is ``candidate_gen_ms + mmr_post_ms``. Reranker time
    is intentionally NOT included — see the module docstring.
    """

    config_name: str
    query_id: str
    candidate_gen_ms: float
    mmr_post_ms: float
    total_ms: float
    n_candidates: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyAggregate:
    """Per-config aggregate over a query set (gold or silver or combined)."""

    config_name: str
    role: str
    set_name: str
    n_queries: int
    candidate_gen_ms: Dict[str, float]
    mmr_post_ms: Dict[str, float]
    total_ms: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "role": self.role,
            "set_name": self.set_name,
            "n_queries": self.n_queries,
            "candidate_gen_ms": dict(self.candidate_gen_ms),
            "mmr_post_ms": dict(self.mmr_post_ms),
            "total_ms": dict(self.total_ms),
            "notes": list(self.notes),
        }


@dataclass
class LatencySmokeReport:
    """Top-level latency smoke report payload.

    The renderer keys off ``aggregates`` (one row per config × set) and
    ``configs`` (the suite definition). ``measured_at_pool_size`` is
    written into every row's notes so a reader cannot lose track of
    which pool the candidate-gen pass was timed against.
    """

    pool_size: int
    measured_at_pool_size_note: str
    rerank_note: str
    suite_note: str
    configs: List[LatencySmokeConfig]
    aggregates: List[LatencyAggregate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "measured_at_pool_size_note": self.measured_at_pool_size_note,
            "rerank_note": self.rerank_note,
            "suite_note": self.suite_note,
            "configs": [c.to_dict() for c in self.configs],
            "aggregates": [a.to_dict() for a in self.aggregates],
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, no numpy dep."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = float(q) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo]) + (float(s[hi]) - float(s[lo])) * frac


def _stage_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
            "min": 0.0, "max": 0.0, "n": 0.0,
        }
    return {
        "mean": float(sum(values) / len(values)),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": float(len(values)),
    }


def aggregate_latencies(
    *,
    measurements: Sequence[LatencyMeasurement],
    config_name: str,
    role: str,
    set_name: str,
    notes: Optional[Sequence[str]] = None,
) -> LatencyAggregate:
    """Aggregate per-stage timings into mean/p50/p90/p99 over a query set."""
    cgen = [m.candidate_gen_ms for m in measurements]
    mmr = [m.mmr_post_ms for m in measurements]
    total = [m.total_ms for m in measurements]
    return LatencyAggregate(
        config_name=config_name,
        role=role,
        set_name=set_name,
        n_queries=len(measurements),
        candidate_gen_ms=_stage_summary(cgen),
        mmr_post_ms=_stage_summary(mmr),
        total_ms=_stage_summary(total),
        notes=list(notes or ()),
    )


# ---------------------------------------------------------------------------
# Live MMR micro-benchmark
# ---------------------------------------------------------------------------


def time_mmr_post_hoc_pass(
    pool: Sequence[RetrievedDoc],
    *,
    candidate_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    top_k: int,
    reps: int = DEFAULT_MMR_REPS,
    timer: Callable[[], float] = time.perf_counter,
) -> float:
    """Time the post-hoc MMR slicing call against ``pool``, ``reps`` times.

    Returns the mean elapsed time in milliseconds. ``reps`` smooths out
    clock granularity since the call is sub-millisecond on a 40-doc
    pool — a single sample is dominated by the timer's resolution.
    """
    if reps <= 0:
        raise ValueError("reps must be >= 1")
    samples_ms: List[float] = []
    for _ in range(int(reps)):
        t0 = timer()
        apply_variant_to_candidates(
            pool,
            candidate_k=int(candidate_k),
            use_mmr=bool(use_mmr),
            mmr_lambda=float(mmr_lambda),
            top_k=int(top_k),
        )
        elapsed_ms = (timer() - t0) * 1000.0
        samples_ms.append(elapsed_ms)
    return float(sum(samples_ms) / len(samples_ms))


def measure_one_config(
    *,
    config: LatencySmokeConfig,
    pool_rows: Sequence[Any],
    reps: int = DEFAULT_MMR_REPS,
    timer: Callable[[], float] = time.perf_counter,
) -> List[LatencyMeasurement]:
    """Measure ``config`` against every row of ``pool_rows``.

    ``pool_rows`` is the deserialized candidate pool (one row per
    query), each row has ``query_id``, ``elapsed_ms``, and ``docs``.
    The function accepts ``RetrievalResult`` objects from the
    ``scripts.phase7_human_gold_tune`` module, but does not depend on
    that import — anything with ``query_id``, ``elapsed_ms``, ``docs``
    works (duck-typed).
    """
    out: List[LatencyMeasurement] = []
    for row in pool_rows:
        qid = getattr(row, "query_id", None) or row["query_id"]
        elapsed_ms = float(
            getattr(row, "elapsed_ms", None) or row.get("elapsed_ms", 0.0)
            if isinstance(row, dict)
            else getattr(row, "elapsed_ms", 0.0)
        )
        docs_attr = getattr(row, "docs", None)
        if docs_attr is None and isinstance(row, dict):
            docs_attr = row.get("docs") or []
        # Coerce dict docs to RetrievedDoc.
        coerced_docs: List[RetrievedDoc] = []
        for d in docs_attr or ():
            if isinstance(d, RetrievedDoc):
                coerced_docs.append(d)
                continue
            coerced_docs.append(RetrievedDoc(
                rank=int(d.get("rank") or 0),
                chunk_id=str(d.get("chunk_id") or ""),
                page_id=str(d.get("page_id") or ""),
                title=str(d.get("title") or ""),
                section_path=tuple(
                    str(x) for x in (d.get("section_path") or [])
                ),
                score=(
                    float(d["score"])
                    if d.get("score") is not None else None
                ),
            ))
        mmr_ms = time_mmr_post_hoc_pass(
            coerced_docs,
            candidate_k=config.candidate_k,
            use_mmr=config.use_mmr,
            mmr_lambda=config.mmr_lambda,
            top_k=config.top_k,
            reps=reps,
            timer=timer,
        )
        out.append(LatencyMeasurement(
            config_name=config.name,
            query_id=str(qid),
            candidate_gen_ms=elapsed_ms,
            mmr_post_ms=mmr_ms,
            total_ms=elapsed_ms + mmr_ms,
            n_candidates=len(coerced_docs),
        ))
    return out


# ---------------------------------------------------------------------------
# End-to-end smoke check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PoolBundle:
    name: str
    rows: Sequence[Any]


def run_smoke_check(
    *,
    configs: Sequence[LatencySmokeConfig],
    gold_pool_rows: Sequence[Any],
    silver_pool_rows: Sequence[Any],
    pool_size: int,
    reps: int = DEFAULT_MMR_REPS,
    include_combined: bool = True,
    timer: Callable[[], float] = time.perf_counter,
) -> LatencySmokeReport:
    """Run the smoke suite over gold + silver + combined sets.

    ``include_combined`` adds a "combined" aggregate row per config
    (gold + silver glued together). Off by default in the test harness
    so unit tests run faster, on by default for the CLI.
    """
    sets: List[_PoolBundle] = [
        _PoolBundle(name="gold-50", rows=gold_pool_rows),
        _PoolBundle(name="silver-500", rows=silver_pool_rows),
    ]
    if include_combined:
        combined_rows = list(gold_pool_rows) + list(silver_pool_rows)
        sets.append(_PoolBundle(name="combined-550", rows=combined_rows))

    aggregates: List[LatencyAggregate] = []
    for cfg in configs:
        for s in sets:
            measurements = measure_one_config(
                config=cfg, pool_rows=s.rows, reps=reps, timer=timer,
            )
            agg = aggregate_latencies(
                measurements=measurements,
                config_name=cfg.name, role=cfg.role, set_name=s.name,
                notes=[
                    f"candidate_gen_ms is the live FAISS+embed time the "
                    f"confirm sweep recorded at pool_size={int(pool_size)}; "
                    f"upper-bound estimate for candidate_k<{int(pool_size)}.",
                    f"mmr_post_ms timed live at reps={int(reps)} per query.",
                    "reranker timing NOT measured (NoOp in this env).",
                ],
            )
            aggregates.append(agg)
            log.info(
                "smoke: config=%s set=%s n=%d total_p50=%.3fms "
                "total_p90=%.3fms total_p99=%.3fms",
                cfg.name, s.name, agg.n_queries,
                agg.total_ms.get("p50", 0.0),
                agg.total_ms.get("p90", 0.0),
                agg.total_ms.get("p99", 0.0),
            )

    return LatencySmokeReport(
        pool_size=int(pool_size),
        measured_at_pool_size_note=(
            f"candidate_gen_ms reflects the cached pool's elapsed_ms — "
            f"the confirm sweep ran the live retriever at "
            f"pool_size={int(pool_size)} (no MMR). For candidate_k less "
            f"than the pool size the live retriever would do slightly "
            f"less work, so candidate_gen_ms here is a small upper bound."
        ),
        rerank_note=(
            "Reranker stage NOT measured in this smoke check. The eval "
            "environment runs with the NoOp reranker — any timing "
            "produced here would be a fixture artefact, not a "
            "production-reranker number."
        ),
        suite_note=(
            "Smoke compares pre-promotion baseline (no MMR), the Phase "
            "7.x first-pass best (candidate_k=30, MMR, λ=0.70), and the "
            "Phase 7.5 production-recommended (candidate_k=40, MMR, "
            "λ=0.70) on the same query set."
        ),
        configs=list(configs),
        aggregates=aggregates,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_latency_smoke_report(report: LatencySmokeReport) -> str:
    """Render the latency smoke report as Markdown."""
    lines: List[str] = []
    lines.append("# Phase 7.5 — latency smoke check")
    lines.append("")
    lines.append("> " + report.suite_note)
    lines.append("")
    lines.append("## Honest scope")
    lines.append("")
    lines.append(f"- **Pool size**: {report.pool_size}")
    lines.append(f"- **Pool note**: {report.measured_at_pool_size_note}")
    lines.append(f"- **Rerank note**: {report.rerank_note}")
    lines.append("")
    lines.append("## Configs under test")
    lines.append("")
    lines.append(
        "| name | role | top_k | candidate_k | use_mmr | mmr_lambda | description |"
    )
    lines.append("|---|---|---:|---:|:---:|---:|---|")
    for c in report.configs:
        lines.append(
            f"| `{c.name}` | {c.role} | {c.top_k} | {c.candidate_k} | "
            f"{'✓' if c.use_mmr else '—'} | {c.mmr_lambda:.2f} | "
            f"{c.description} |"
        )
    lines.append("")

    lines.append("## Per-set latency (ms)")
    lines.append("")
    set_names: List[str] = []
    for a in report.aggregates:
        if a.set_name not in set_names:
            set_names.append(a.set_name)

    for set_name in set_names:
        lines.append(f"### {set_name}")
        lines.append("")
        lines.append(
            "| config | role | n | candidate_gen p50/p90/p99 | "
            "mmr_post p50/p90/p99 | total p50/p90/p99 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for a in report.aggregates:
            if a.set_name != set_name:
                continue

            def _band(d: Mapping[str, float]) -> str:
                return (
                    f"{d.get('p50', 0.0):.3f} / "
                    f"{d.get('p90', 0.0):.3f} / "
                    f"{d.get('p99', 0.0):.3f}"
                )

            lines.append(
                f"| `{a.config_name}` | {a.role} | {a.n_queries} | "
                f"{_band(a.candidate_gen_ms)} | "
                f"{_band(a.mmr_post_ms)} | "
                f"{_band(a.total_ms)} |"
            )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    # Pull recommended vs previous_best vs baseline on the gold-50 set
    # for the headline call. Use total_p90 as the comparison point —
    # the spec calls out p90/p99 as the regression threshold.
    by_role_set: Dict[Tuple[str, str], LatencyAggregate] = {
        (a.role, a.set_name): a for a in report.aggregates
    }

    def _verdict_for_set(set_name: str) -> List[str]:
        rec = by_role_set.get((ROLE_RECOMMENDED, set_name))
        prev = by_role_set.get((ROLE_PREVIOUS_BEST, set_name))
        base = by_role_set.get((ROLE_BASELINE, set_name))
        out: List[str] = []
        if rec is None:
            return out
        if prev is not None:
            rec_p90 = rec.total_ms.get("p90", 0.0)
            prev_p90 = prev.total_ms.get("p90", 0.0)
            ratio = (
                (rec_p90 / prev_p90 - 1.0) * 100.0
                if prev_p90 > 0 else 0.0
            )
            out.append(
                f"- **{set_name}**: recommended total_p90={rec_p90:.3f}ms "
                f"vs previous-best {prev_p90:.3f}ms (Δ={ratio:+.1f}%)."
            )
        if base is not None:
            rec_p99 = rec.total_ms.get("p99", 0.0)
            base_p99 = base.total_ms.get("p99", 0.0)
            ratio = (
                (rec_p99 / base_p99 - 1.0) * 100.0
                if base_p99 > 0 else 0.0
            )
            out.append(
                f"  - vs baseline (no-MMR) total_p99={base_p99:.3f}ms → "
                f"recommended {rec_p99:.3f}ms (Δ={ratio:+.1f}%)."
            )
        return out

    any_verdict = False
    for set_name in set_names:
        v = _verdict_for_set(set_name)
        if v:
            any_verdict = True
            lines.extend(v)

    if not any_verdict:
        lines.append("- (no recommended/previous_best pair found; verdict skipped)")

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "- `candidate_k=40` is recommended to **promote** when the "
        "recommended-vs-previous-best total_p90 delta is below 30%. "
        "Above 30% with no clear silver-set quality gain, fall back "
        "to `candidate_k=30, mmr_lambda=0.70`."
    )
    lines.append(
        "- A noisy or unstable measurement (huge variance across "
        "runs) is itself a reason to recommend the smaller candidate_k "
        "fallback — production stability beats a marginal gain."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_latency_smoke_md(path: Path, report: LatencySmokeReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_latency_smoke_report(report)
    path.write_text(md, encoding="utf-8")
    return path


def write_latency_smoke_json(path: Path, report: LatencySmokeReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


__all__ = [
    "DEFAULT_MMR_REPS",
    "DEFAULT_LATENCY_PERCENTILES",
    "ROLE_BASELINE",
    "ROLE_PREVIOUS_BEST",
    "ROLE_RECOMMENDED",
    "ROLE_FALLBACK",
    "LatencySmokeConfig",
    "LatencyMeasurement",
    "LatencyAggregate",
    "LatencySmokeReport",
    "default_smoke_suite",
    "aggregate_latencies",
    "time_mmr_post_hoc_pass",
    "measure_one_config",
    "run_smoke_check",
    "render_latency_smoke_report",
    "write_latency_smoke_md",
    "write_latency_smoke_json",
]
