"""Phase 2 — sweep driver for retrieval-experiment grids.

Drives a *grid* of retrieval-eval configurations (retriever kind ×
candidate_k × final_top_k × prefix variant) against a single dataset
and consolidates the per-cell ``RetrievalEvalSummary`` outputs into a
single sweep report. The report can then be fed to the existing
``compute_pareto_frontier`` helper for accuracy-vs-latency comparison.

The driver is **eval-only**. It:

  - Calls ``run_retrieval_eval`` once per cell with the cell's
    retriever instance, capturing per-row metrics + Phase 1 candidate /
    pre-rerank / diagnostic aggregates.
  - Records the cell's identifying knobs alongside the summary so a
    post-hoc reader knows which configuration produced which numbers.
  - Adapts each cell to the existing ``TopNSweepEntry`` shape via
    ``sweep_to_topn_sweep_report`` so the existing Pareto frontier
    tooling consumes Phase 2 results without modification.

Backward compatibility:

  - Production code is *not* touched. The driver only orchestrates
    eval-side modules (``BM25EvalRetriever``, ``RRFHybridEvalRetriever``,
    or any external retriever the caller wires in).
  - Any retriever satisfying the harness's ``retrieve(query) -> Report``
    duck-type works; the sweep driver itself is retriever-agnostic.
  - Cells the caller opted out of (e.g. no BM25 index built) are simply
    not in the grid; the sweep report records what ran.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from eval.harness.retrieval_eval import (
    DEFAULT_CANDIDATE_KS,
    DEFAULT_DIVERSITY_KS,
    DEFAULT_MRR_K,
    DEFAULT_NDCG_K,
    RetrievalEvalRow,
    RetrievalEvalSummary,
    run_retrieval_eval,
)
from eval.harness.topn_sweep import TopNSweepEntry, TopNSweepReport

log = logging.getLogger(__name__)


SCHEMA_VERSION = "phase2-retrieval-sweep.v1"

# Retriever-kind labels surfaced on each sweep cell. Free-form, but
# keeping the canonical set as constants helps the sweep CLI and tests
# stay aligned with the markdown writer's section headers.
KIND_DENSE = "dense"
KIND_BM25 = "bm25"
KIND_HYBRID = "hybrid"


@dataclass(frozen=True)
class RetrievalSweepConfig:
    """One cell of the sweep grid.

    Carries the identifying knobs (kind / variant / candidate_k /
    final_top_k) plus the *built* retriever instance. The driver
    delegates retriever construction to the caller so the grid stays
    decoupled from the various retriever factories — building a dense
    Retriever takes a FAISS index path, building a BM25 retriever
    takes a chunk corpus, etc. The caller wires those concerns and
    just hands the driver finished retrievers.

    ``extra`` is a free-form provenance bag (e.g. ``embedding_model``,
    ``reranker_name``, dataset chunk count) so the sweep report can
    quote the row alongside the numbers.
    """

    label: str
    retriever_kind: str
    embedding_text_variant: str
    candidate_k: Optional[int]
    final_top_k: int
    retriever: Any  # duck-typed retriever; not validated here
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalSweepCell:
    """One cell's run output: knobs + summary + per-row sample.

    The summary is the full ``RetrievalEvalSummary``; ``rows_sample``
    is a small subset of rows (default first 10) for sanity-check
    eyeballing in the markdown writer. The full row list isn't kept
    on the sweep report — it would balloon the JSON size; callers
    that need the rows should call ``run_retrieval_eval`` directly
    or persist per-cell reports separately.
    """

    label: str
    retriever_kind: str
    embedding_text_variant: str
    candidate_k: Optional[int]
    final_top_k: int
    extra: Dict[str, Any]
    summary: RetrievalEvalSummary
    rows_sample: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalSweepReport:
    """Top-level sweep output.

    ``cells`` lists every executed configuration. ``schema`` and
    ``dataset_path`` carry provenance. The sweep report is what gets
    written to disk; ``sweep_to_topn_sweep_report`` adapts it for the
    existing Pareto frontier tooling.
    """

    schema: str
    dataset_path: str
    corpus_path: Optional[str]
    cells: List[RetrievalSweepCell] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def run_retrieval_sweep(
    dataset: Sequence[Mapping[str, Any]],
    *,
    configs: Sequence[RetrievalSweepConfig],
    mrr_k: int = DEFAULT_MRR_K,
    ndcg_k: int = DEFAULT_NDCG_K,
    candidate_ks: Tuple[int, ...] = DEFAULT_CANDIDATE_KS,
    diversity_ks: Tuple[int, ...] = DEFAULT_DIVERSITY_KS,
    dataset_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    sample_rows: int = 10,
) -> RetrievalSweepReport:
    """Execute every config in ``configs`` against ``dataset``.

    Per-cell side effects: none beyond what each retriever does. The
    driver does NOT persist per-cell reports — callers wanting on-disk
    artifacts can iterate ``cells`` and write each summary separately.

    ``sample_rows`` controls how many per-cell row dicts are kept on
    the sweep report; default 10 is a balance between auditability
    (operators want to see *some* rows in the JSON) and report size.
    Pass ``0`` to drop them entirely.
    """
    if not configs:
        raise ValueError("retrieval sweep needs at least one config")

    cells: List[RetrievalSweepCell] = []
    for cfg in configs:
        log.info(
            "phase2 sweep: running cell label=%s kind=%s variant=%s "
            "cand_k=%s final_k=%d",
            cfg.label, cfg.retriever_kind, cfg.embedding_text_variant,
            cfg.candidate_k, cfg.final_top_k,
        )
        summary, rows, _, _ = run_retrieval_eval(
            list(dataset),
            retriever=cfg.retriever,
            top_k=cfg.final_top_k,
            mrr_k=mrr_k,
            ndcg_k=ndcg_k,
            candidate_ks=candidate_ks,
            diversity_ks=diversity_ks,
            dataset_path=dataset_path or "<inline>",
            corpus_path=corpus_path,
        )
        sample = []
        if sample_rows > 0:
            for row in rows[: max(0, int(sample_rows))]:
                sample.append({
                    "id": row.id,
                    "query": row.query,
                    "expected_doc_ids": list(row.expected_doc_ids),
                    "retrieved_doc_ids": list(row.retrieved_doc_ids[:5]),
                    "hit_at_5": row.hit_at_5,
                    "mrr_at_10": row.mrr_at_10,
                })
        cells.append(RetrievalSweepCell(
            label=cfg.label,
            retriever_kind=cfg.retriever_kind,
            embedding_text_variant=cfg.embedding_text_variant,
            candidate_k=cfg.candidate_k,
            final_top_k=cfg.final_top_k,
            extra=dict(cfg.extra),
            summary=summary,
            rows_sample=sample,
        ))
    return RetrievalSweepReport(
        schema=SCHEMA_VERSION,
        dataset_path=dataset_path or "<inline>",
        corpus_path=corpus_path,
        cells=cells,
    )


def sweep_to_topn_sweep_report(
    sweep: RetrievalSweepReport,
) -> TopNSweepReport:
    """Adapt a Phase 2 sweep report into the existing TopNSweepReport.

    The existing ``compute_pareto_frontier`` consumes ``TopNSweepReport``;
    rather than fork the Pareto algorithm, we project each Phase 2 cell
    into a ``TopNSweepEntry`` and reuse the existing tooling. Fields
    that don't have a Phase 2 equivalent (e.g. ``reranker_batch_size``,
    ``reranker_model``) stay ``None``; numeric fields with no value
    also stay ``None`` so the dominance comparison drops them as
    "(missing-data)" rather than zero-counting them as a strong score.

    ``dense_top_n`` is set to ``cell.candidate_k`` when present,
    falling back to ``cell.final_top_k`` so each cell still has an
    integer key the Pareto report can quote on its frontier label.
    """
    entries: List[TopNSweepEntry] = []
    for cell in sweep.cells:
        s = cell.summary
        dense_top_n = cell.candidate_k or cell.final_top_k
        entries.append(TopNSweepEntry(
            label=cell.label,
            report_path=None,
            dense_top_n=int(dense_top_n) if dense_top_n is not None else None,
            final_top_k=int(cell.final_top_k),
            reranker_batch_size=None,
            reranker_model=cell.extra.get("reranker_model"),
            row_count=int(s.row_count),
            rows_with_expected_doc_ids=int(s.rows_with_expected_doc_ids),
            mean_hit_at_1=s.mean_hit_at_1,
            mean_hit_at_3=s.mean_hit_at_3,
            mean_hit_at_5=s.mean_hit_at_5,
            mean_mrr_at_10=s.mean_mrr_at_10,
            mean_ndcg_at_10=s.mean_ndcg_at_10,
            candidate_recall=(s.candidate_recalls or {}).get("50"),
            mean_dup_rate=s.mean_dup_rate,
            mean_avg_context_token_count=s.mean_avg_context_token_count,
            rerank_avg_ms=s.mean_rerank_ms,
            rerank_p50_ms=s.p50_rerank_ms,
            rerank_p90_ms=s.p90_rerank_ms,
            rerank_p95_ms=s.p95_rerank_ms,
            rerank_p99_ms=s.p99_rerank_ms,
            rerank_max_ms=s.max_rerank_ms,
            rerank_row_count=int(s.rerank_row_count),
            total_query_avg_ms=s.avg_total_retrieval_ms or s.mean_retrieval_ms,
            total_query_p50_ms=s.p50_retrieval_ms,
            total_query_p90_ms=s.p90_retrieval_ms,
            total_query_p95_ms=s.p95_total_retrieval_ms or s.p95_retrieval_ms,
            total_query_p99_ms=s.p99_retrieval_ms,
            total_query_max_ms=s.max_retrieval_ms,
            total_query_row_count=int(s.row_count),
            dense_retrieval_avg_ms=s.mean_dense_retrieval_ms,
            dense_retrieval_p50_ms=s.p50_dense_retrieval_ms,
            dense_retrieval_p95_ms=s.p95_dense_retrieval_ms,
            dense_retrieval_row_count=int(s.dense_retrieval_row_count),
        ))
    return TopNSweepReport(
        schema=SCHEMA_VERSION + "::topn-adapter",
        entries=entries,
        caveats=list(sweep.caveats),
    )


def sweep_report_to_dict(sweep: RetrievalSweepReport) -> Dict[str, Any]:
    """JSON-friendly projection of a sweep report.

    ``RetrievalEvalSummary`` is dataclass-convertible; ``asdict`` walks
    nested dicts cleanly. The output preserves every cell's full
    summary so a downstream consumer can re-run any of the Phase 1
    diagnostics on the cell without re-querying the retriever.
    """
    return {
        "schema": sweep.schema,
        "dataset_path": sweep.dataset_path,
        "corpus_path": sweep.corpus_path,
        "caveats": list(sweep.caveats),
        "cells": [
            {
                "label": cell.label,
                "retriever_kind": cell.retriever_kind,
                "embedding_text_variant": cell.embedding_text_variant,
                "candidate_k": cell.candidate_k,
                "final_top_k": cell.final_top_k,
                "extra": dict(cell.extra),
                "summary": asdict(cell.summary),
                "rows_sample": list(cell.rows_sample),
            }
            for cell in sweep.cells
        ],
    }


def render_sweep_markdown(sweep: RetrievalSweepReport) -> str:
    """Compose a compact markdown overview of a sweep.

    Sections:
      1. One-line cell list
      2. Headline metrics table
      3. Latency table
      4. Composite scores (quality_score / efficiency_score)

    The Pareto frontier itself is rendered separately by
    ``render_pareto_markdown`` once the caller adapts the sweep via
    ``sweep_to_topn_sweep_report``. We don't double-render it here;
    keep this report focused on per-cell facts and let the Pareto
    report focus on dominance relationships.
    """
    lines: List[str] = []
    lines.append("# Phase 2 retrieval sweep")
    lines.append("")
    lines.append(f"- dataset: `{sweep.dataset_path}`")
    if sweep.corpus_path:
        lines.append(f"- corpus:  `{sweep.corpus_path}`")
    lines.append(f"- cells:   {len(sweep.cells)}")
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append(
        "| label | kind | variant | cand_k | final_k |"
    )
    lines.append("|---|---|---|---:|---:|")
    for cell in sweep.cells:
        lines.append(
            f"| {cell.label} | {cell.retriever_kind} | "
            f"{cell.embedding_text_variant} | "
            f"{_fmt_int(cell.candidate_k)} | {cell.final_top_k} |"
        )
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(
        "| label | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
        "cand_hit@50 | dup@10 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell in sweep.cells:
        s = cell.summary
        cand_h50 = (s.candidate_hit_rates or {}).get("50")
        dup_at_10 = (s.duplicate_doc_ratios or {}).get("10")
        lines.append(
            f"| {cell.label} | {_fmt(s.mean_hit_at_1)} | "
            f"{_fmt(s.mean_hit_at_3)} | {_fmt(s.mean_hit_at_5)} | "
            f"{_fmt(s.mean_mrr_at_10)} | {_fmt(s.mean_ndcg_at_10)} | "
            f"{_fmt(cand_h50)} | {_fmt(dup_at_10)} |"
        )
    lines.append("")
    lines.append("## Latency (ms)")
    lines.append("")
    lines.append(
        "| label | total p50 | total p95 | dense avg | rerank avg |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for cell in sweep.cells:
        s = cell.summary
        lines.append(
            f"| {cell.label} | {_fmt(s.p50_retrieval_ms)} | "
            f"{_fmt(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} | "
            f"{_fmt(s.mean_dense_retrieval_ms)} | "
            f"{_fmt(s.mean_rerank_ms)} |"
        )
    lines.append("")
    lines.append("## Composite scores")
    lines.append("")
    lines.append("| label | quality_score | efficiency_score |")
    lines.append("|---|---:|---:|")
    for cell in sweep.cells:
        s = cell.summary
        lines.append(
            f"| {cell.label} | {_fmt(s.quality_score)} | "
            f"{_fmt(s.efficiency_score)} |"
        )
    lines.append("")
    lines.append(
        "> _quality_score / efficiency_score are comparison aids, not "
        "adoption rules. See ``eval.harness.agent_loop_ab`` for the "
        "conservative recommendation rule._"
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "—"
    return str(int(value))
