"""Phase 2A-L topN sweep aggregation.

Reads N labelled retrieval_eval_report.json files (one per
``dense_top_n`` configuration) and emits a sweep table that captures
the accuracy / latency trade-off for the cross-encoder reranker.

Each entry records:

  - dense_top_n          (the swept knob)
  - final_top_k          (constant across the sweep — usually 10)
  - hit@1 / hit@3 / hit@5 / mrr@10 / ndcg@10
  - candidate_recall@N when the dense-only sibling provided one
  - mean_dup_rate, mean_avg_context_token_count
  - rerank latency (avg, p50, p90, p95, p99, max, n)
  - total_query latency (avg, p50, p90, p95, p99, max, n)
  - dense_retrieval latency (avg, p50, p95, n) when present

Public surface:

  - ``TopNSweepEntry``           — one row of the sweep table
  - ``TopNSweepReport``          — the full sweep report
  - ``build_topn_sweep(slices)`` — load + assemble
  - ``render_topn_sweep_markdown(report)``
  - ``topn_sweep_to_dict(report)``

The schema string ``phase2a-topn-sweep.v1`` is pinned on every emitted
JSON document.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from eval.harness.latency_breakdown import _percentile_nearest_rank

log = logging.getLogger(__name__)


SCHEMA_VERSION = "phase2a-topn-sweep.v1"


@dataclass
class TopNSweepEntry:
    """One row of the topN sweep table.

    Latency aggregates duplicate the per-stage breakdown the underlying
    retrieval_eval_report.json already exposes — pulling them in here
    keeps the sweep document self-contained, so a Pareto plot or a
    recommended-modes computation can read off a single file.
    """

    label: str
    report_path: str
    dense_top_n: Optional[int]
    final_top_k: Optional[int]
    reranker_batch_size: Optional[int]
    reranker_model: Optional[str]
    row_count: int
    rows_with_expected_doc_ids: int
    # Headline accuracy
    mean_hit_at_1: Optional[float]
    mean_hit_at_3: Optional[float]
    mean_hit_at_5: Optional[float]
    mean_mrr_at_10: Optional[float]
    mean_ndcg_at_10: Optional[float]
    # Candidate recall: hit@N from a dense-only sibling sharing the
    # same corpus; the topN entry itself doesn't compute this — it's
    # just the upper bound the reranker can possibly reach. Empty
    # dict when no extra hits were measured.
    candidate_recall: Dict[str, Optional[float]] = field(default_factory=dict)
    # Diversity / context budget
    mean_dup_rate: Optional[float] = None
    mean_avg_context_token_count: Optional[float] = None
    # Rerank latency series (None when the run was noop)
    rerank_avg_ms: Optional[float] = None
    rerank_p50_ms: Optional[float] = None
    rerank_p90_ms: Optional[float] = None
    rerank_p95_ms: Optional[float] = None
    rerank_p99_ms: Optional[float] = None
    rerank_max_ms: Optional[float] = None
    rerank_row_count: int = 0
    # Total query latency (== retrieval_ms in the underlying report)
    total_query_avg_ms: Optional[float] = None
    total_query_p50_ms: Optional[float] = None
    total_query_p90_ms: Optional[float] = None
    total_query_p95_ms: Optional[float] = None
    total_query_p99_ms: Optional[float] = None
    total_query_max_ms: Optional[float] = None
    total_query_row_count: int = 0
    # Dense retrieval (Phase 2A-L addition; older runs leave this None)
    dense_retrieval_avg_ms: Optional[float] = None
    dense_retrieval_p50_ms: Optional[float] = None
    dense_retrieval_p95_ms: Optional[float] = None
    dense_retrieval_row_count: int = 0


@dataclass
class TopNSweepReport:
    schema: str
    entries: List[TopNSweepEntry] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile_or_none(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    return round(_percentile_nearest_rank(values, p), 3)


def _aggregate_total_query(rows: List[Mapping[str, Any]]) -> Tuple[List[float], int]:
    series: List[float] = []
    for r in rows:
        if r.get("error"):
            continue
        v = r.get("retrieval_ms")
        if v is None:
            continue
        try:
            series.append(float(v))
        except (TypeError, ValueError):
            continue
    return series, len(series)


def _aggregate_dense_retrieval(rows: List[Mapping[str, Any]]) -> Tuple[List[float], int]:
    series: List[float] = []
    for r in rows:
        if r.get("error"):
            continue
        v = r.get("dense_retrieval_ms")
        if v is None:
            continue
        try:
            series.append(float(v))
        except (TypeError, ValueError):
            continue
    return series, len(series)


def _entry_from_report(
    label: str,
    path: Path,
    payload: Mapping[str, Any],
    candidate_recall_source: Optional[Mapping[str, Any]] = None,
) -> TopNSweepEntry:
    metadata = dict(payload.get("metadata") or {})
    summary = dict(payload.get("summary") or {})
    rows = list(payload.get("rows") or [])

    candidate_recall: Dict[str, Optional[float]] = {}
    extras = dict(summary.get("mean_extra_hits") or {})
    if extras:
        candidate_recall = {
            str(k): _safe_float(v) for k, v in extras.items()
        }
    elif candidate_recall_source:
        ext = dict(candidate_recall_source.get("mean_extra_hits") or {})
        if ext:
            candidate_recall = {
                str(k): _safe_float(v) for k, v in ext.items()
            }

    rerank_avg = _safe_float(summary.get("mean_rerank_ms"))
    rerank_p50 = _safe_float(summary.get("p50_rerank_ms"))
    rerank_p90 = _safe_float(summary.get("p90_rerank_ms"))
    rerank_p95 = _safe_float(summary.get("p95_rerank_ms"))
    rerank_p99 = _safe_float(summary.get("p99_rerank_ms"))
    rerank_max = _safe_float(summary.get("max_rerank_ms"))
    rerank_row_count = int(summary.get("rerank_row_count") or 0)

    # Total query series — recompute from rows so we can produce p90/p99
    # even when older summaries don't carry them. This also keeps the
    # entry self-contained in case a run's summary block was edited.
    total_series, total_n = _aggregate_total_query(rows)
    if total_series:
        total_avg: Optional[float] = round(
            sum(total_series) / float(len(total_series)), 3,
        )
    else:
        total_avg = None
    total_p50 = _percentile_or_none(total_series, 50.0)
    total_p90 = _percentile_or_none(total_series, 90.0)
    total_p95 = _percentile_or_none(total_series, 95.0)
    total_p99 = _percentile_or_none(total_series, 99.0)
    total_max = round(max(total_series), 3) if total_series else None

    dense_series, dense_n = _aggregate_dense_retrieval(rows)
    if dense_series:
        dense_avg: Optional[float] = round(
            sum(dense_series) / float(len(dense_series)), 3,
        )
    else:
        dense_avg = None
    dense_p50 = _percentile_or_none(dense_series, 50.0)
    dense_p95 = _percentile_or_none(dense_series, 95.0)

    return TopNSweepEntry(
        label=label,
        report_path=str(path),
        dense_top_n=_safe_int(metadata.get("dense_top_n") or metadata.get("candidate_k")),
        final_top_k=_safe_int(metadata.get("final_top_k") or summary.get("top_k")),
        reranker_batch_size=_safe_int(metadata.get("reranker_batch_size")),
        reranker_model=metadata.get("reranker_model"),
        row_count=int(summary.get("row_count") or 0),
        rows_with_expected_doc_ids=int(summary.get("rows_with_expected_doc_ids") or 0),
        mean_hit_at_1=_safe_float(summary.get("mean_hit_at_1")),
        mean_hit_at_3=_safe_float(summary.get("mean_hit_at_3")),
        mean_hit_at_5=_safe_float(summary.get("mean_hit_at_5")),
        mean_mrr_at_10=_safe_float(summary.get("mean_mrr_at_10")),
        mean_ndcg_at_10=_safe_float(summary.get("mean_ndcg_at_10")),
        candidate_recall=candidate_recall,
        mean_dup_rate=_safe_float(summary.get("mean_dup_rate")),
        mean_avg_context_token_count=_safe_float(
            summary.get("mean_avg_context_token_count")
        ),
        rerank_avg_ms=rerank_avg,
        rerank_p50_ms=rerank_p50,
        rerank_p90_ms=rerank_p90,
        rerank_p95_ms=rerank_p95,
        rerank_p99_ms=rerank_p99,
        rerank_max_ms=rerank_max,
        rerank_row_count=rerank_row_count,
        total_query_avg_ms=total_avg,
        total_query_p50_ms=total_p50,
        total_query_p90_ms=total_p90,
        total_query_p95_ms=total_p95,
        total_query_p99_ms=total_p99,
        total_query_max_ms=total_max,
        total_query_row_count=total_n,
        dense_retrieval_avg_ms=dense_avg,
        dense_retrieval_p50_ms=dense_p50,
        dense_retrieval_p95_ms=dense_p95,
        dense_retrieval_row_count=dense_n,
    )


def build_topn_sweep(
    slices: List[Tuple[str, Path]],
    *,
    candidate_recall_path: Optional[Path] = None,
    caveats: Optional[List[str]] = None,
) -> TopNSweepReport:
    """Assemble a topN sweep report from N labelled retrieval reports.

    ``candidate_recall_path`` is an optional sibling pointing at a
    dense-only run with ``mean_extra_hits`` populated; when provided,
    every sweep entry that doesn't already carry candidate-recall
    numbers inherits them so the markdown writer can quote the upper
    bound on every row.
    """
    candidate_source: Optional[Mapping[str, Any]] = None
    if candidate_recall_path is not None:
        cp = Path(candidate_recall_path)
        if cp.exists():
            payload = json.loads(cp.read_text(encoding="utf-8"))
            candidate_source = dict(payload.get("summary") or {})
        else:
            log.warning(
                "candidate_recall report not found at %s — sweep entries "
                "will only quote candidate_recall when their own "
                "summary already carries mean_extra_hits.",
                cp,
            )

    entries: List[TopNSweepEntry] = []
    for label, path in slices:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Retrieval report not found: {p}. Pass --slice "
                f"'label:path/to/retrieval_eval_report.json' for each "
                f"finished retrieval-rerank run."
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        entries.append(
            _entry_from_report(label, p, payload, candidate_source)
        )

    # Sort by dense_top_n when known; entries without one fall to the
    # end so the table still reads in ascending sweep order even when
    # one slice is e.g. the dense-only baseline.
    entries.sort(
        key=lambda e: (e.dense_top_n if e.dense_top_n is not None else 1_000_000)
    )

    return TopNSweepReport(
        schema=SCHEMA_VERSION,
        entries=entries,
        caveats=list(caveats or _default_caveats()),
    )


def _default_caveats() -> List[str]:
    return [
        "topN sweep은 같은 corpus + 같은 dataset에서 dense_top_n만 변경한다 — corpus / chunker / reranker 모델은 고정.",
        "candidate_recall@N은 dense-only baseline의 hit@N이며, reranker가 도달할 수 있는 정확도 상한이다.",
        "rerank latency는 cross-encoder predict 단독 wall-clock; total_query는 retrieve() 전체이므로 dense 부분이 함께 포함된다.",
        "p95/p99는 nearest-rank percentile으로, 200-row dataset에서는 ±1 row 단위로 흔들릴 수 있다.",
        "이 리포트는 어떤 설정도 production default로 승격하지 않는다 — Pareto frontier + recommended-modes로 결정 근거를 분리한다.",
    ]


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


def topn_sweep_to_dict(report: TopNSweepReport) -> Dict[str, Any]:
    return asdict(report)


# ---------------------------------------------------------------------------
# Markdown.
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_or_dash_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def render_topn_sweep_markdown(report: TopNSweepReport) -> str:
    lines: List[str] = []
    lines.append("# Phase 2A-L topN sweep")
    lines.append("")
    lines.append(f"- entries: {len(report.entries)}")
    if report.entries:
        labels = ", ".join(
            f"{e.label}({_fmt_or_dash_int(e.dense_top_n)})" for e in report.entries
        )
        lines.append(f"- configs: {labels}")
    lines.append("")

    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")

    if not report.entries:
        lines.append("_No entries in this sweep._")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Headline accuracy")
    lines.append("")
    lines.append(
        "| label | dense_top_n | final_top_k | hit@1 | hit@3 | hit@5 | "
        "mrr@10 | ndcg@10 | dup_rate | avg_ctx_tok |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for e in report.entries:
        lines.append(
            f"| {e.label} | "
            f"{_fmt_or_dash_int(e.dense_top_n)} | "
            f"{_fmt_or_dash_int(e.final_top_k)} | "
            f"{_fmt(e.mean_hit_at_1)} | "
            f"{_fmt(e.mean_hit_at_3)} | "
            f"{_fmt(e.mean_hit_at_5)} | "
            f"{_fmt(e.mean_mrr_at_10)} | "
            f"{_fmt(e.mean_ndcg_at_10)} | "
            f"{_fmt(e.mean_dup_rate)} | "
            f"{_fmt(e.mean_avg_context_token_count)} |"
        )
    lines.append("")

    # Candidate-recall companion table — render only when at least one
    # entry carried it (otherwise it's an empty table that confuses the
    # reader).
    cutoffs = sorted(
        {
            int(k) for e in report.entries
            for k in (e.candidate_recall or {}).keys()
            if str(k).isdigit()
        }
    )
    if cutoffs:
        lines.append("## Candidate recall@N (dense-only upper bound)")
        lines.append("")
        lines.append(
            "| label | "
            + " | ".join(f"hit@{k}" for k in cutoffs)
            + " |"
        )
        lines.append(
            "|---" + "|---:" * len(cutoffs) + "|"
        )
        for e in report.entries:
            cells = [
                _fmt((e.candidate_recall or {}).get(str(k)))
                for k in cutoffs
            ]
            lines.append(f"| {e.label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Rerank latency (ms)")
    lines.append("")
    lines.append(
        "| label | dense_top_n | n | avg | p50 | p90 | p95 | p99 | max |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for e in report.entries:
        lines.append(
            f"| {e.label} | "
            f"{_fmt_or_dash_int(e.dense_top_n)} | "
            f"{e.rerank_row_count} | "
            f"{_fmt_ms(e.rerank_avg_ms)} | "
            f"{_fmt_ms(e.rerank_p50_ms)} | "
            f"{_fmt_ms(e.rerank_p90_ms)} | "
            f"{_fmt_ms(e.rerank_p95_ms)} | "
            f"{_fmt_ms(e.rerank_p99_ms)} | "
            f"{_fmt_ms(e.rerank_max_ms)} |"
        )
    lines.append("")

    lines.append("## Total query latency (ms)")
    lines.append("")
    lines.append(
        "| label | dense_top_n | n | avg | p50 | p90 | p95 | p99 | max |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for e in report.entries:
        lines.append(
            f"| {e.label} | "
            f"{_fmt_or_dash_int(e.dense_top_n)} | "
            f"{e.total_query_row_count} | "
            f"{_fmt_ms(e.total_query_avg_ms)} | "
            f"{_fmt_ms(e.total_query_p50_ms)} | "
            f"{_fmt_ms(e.total_query_p90_ms)} | "
            f"{_fmt_ms(e.total_query_p95_ms)} | "
            f"{_fmt_ms(e.total_query_p99_ms)} | "
            f"{_fmt_ms(e.total_query_max_ms)} |"
        )
    lines.append("")

    has_dense = any(e.dense_retrieval_row_count > 0 for e in report.entries)
    if has_dense:
        lines.append("## Dense-retrieval latency (ms)")
        lines.append("")
        lines.append(
            "| label | dense_top_n | n | avg | p50 | p95 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for e in report.entries:
            lines.append(
                f"| {e.label} | "
                f"{_fmt_or_dash_int(e.dense_top_n)} | "
                f"{e.dense_retrieval_row_count} | "
                f"{_fmt_ms(e.dense_retrieval_avg_ms)} | "
                f"{_fmt_ms(e.dense_retrieval_p50_ms)} | "
                f"{_fmt_ms(e.dense_retrieval_p95_ms)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
