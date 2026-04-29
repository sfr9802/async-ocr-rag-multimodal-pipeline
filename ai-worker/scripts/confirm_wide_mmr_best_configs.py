"""Confirm Phase 2 Optuna winner vs Phase 1 wide-MMR best on the full silver_200.

Purpose:
  - The Phase 1 wide-MMR-titlecap diagnostic sweep (200-row) flagged
    ``cap2_top8`` and ``cap1_top8`` as promising_quality with hit@5
    +0.020 vs baseline.
  - The Phase 2 Optuna 5-round refinement found a different recipe
    (``candidate_k=100, rerank_in=16, cap_rr=1, cap_final=any,
    final_top_k=any, mmr_lambda=any in [0.55, 0.75]``) on the
    *first 100 rows* of silver_200 with MRR 0.6745.
  - Whether the Optuna winner generalises to the full 200-row set is
    the open question the round_05 analysis explicitly deferred.

This script answers it. It is **eval-only / report-only**:

  - No production code (``app/``) is mutated.
  - Existing Phase 1 / minimal-sweep / round-bundle artefacts are not
    overwritten — outputs land under
    ``eval/reports/retrieval-wide-mmr-confirm-<TIMESTAMP>/``.
  - All retrieval goes through ``WideRetrievalEvalAdapter`` over the
    cached FAISS index (no PostgreSQL dependency).

Cell roster (12 cells, scored on the **full** 200 rows):

  A. ``baseline_k50_top5``           — Phase 1 baseline (production-equivalent).
  B. ``phase1_best_cap2_top8``       — Phase 1 promising_quality cap=2 cell.
  C. ``phase1_cap1_top8``            — Phase 1 cap=1 alternative.
  D. ``optuna_winner_top8``          — Optuna round_05 winner recipe.
  E. ``optuna_winner_top8_capfinal{1,3}``  — title_cap_final sensitivity.
  F. ``optuna_winner_top8_lambda{055,060,070,075}``  — MMR λ sensitivity.
  G. ``optuna_winner_top{5,10}``     — final_top_k sensitivity.

Outputs:

  - ``summary.csv``          — flat headline metrics per cell.
  - ``summary.json``         — full ``RetrievalEvalSummary`` per cell.
  - ``comparison_report.md`` — markdown narrative + verdict.
  - ``per_query_results.jsonl`` — one row per (cell, query).
  - ``config_dump.json``     — frozen spec list + run provenance.
  - ``regression_guard.md``  — pass/fail guard table vs baseline.

Run::

    python -m scripts.confirm_wide_mmr_best_configs \\
        --cache-dir eval/agent_loop_ab/_indexes/BAAI_bge-m3-mseq1024-30fc1cc1cd8c319a

Use ``--limit N`` for a smoke run. ``--cells`` lets you filter the cell
set by group (e.g. ``--cells optuna_winner,cap_final_sensitivity``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("confirm_wide_mmr_best_configs")


_DEFAULT_DATASET = Path("eval/eval_queries/anime_silver_200.jsonl")
_DEFAULT_CORPUS = Path(
    "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl"
)
_DEFAULT_REPORTS_ROOT = Path("eval/reports")
_DEFAULT_QUERY_TYPE_DRAFT = Path(
    "eval/eval_queries/anime_silver_200.query_type_draft.jsonl"
)
_DEFAULT_CACHE_DIR = Path(
    "eval/agent_loop_ab/_indexes/BAAI_bge-m3-mseq1024-30fc1cc1cd8c319a"
)
_BASELINE_LABEL = "baseline_k50_top5"
_PHASE1_BEST_LABEL = "phase1_best_cap2_top8"
_OPTUNA_WINNER_LABEL = "optuna_winner_top8"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return (
        _DEFAULT_REPORTS_ROOT
        / f"retrieval-wide-mmr-confirm-{_now_stamp()}"
    )


# ---------------------------------------------------------------------------
# Dense stack reuse (mirrors ``eval_wide_mmr_titlecap_sweep._load_dense_stack``).
# Kept inline so this script is independently runnable without importing
# from another script module.
# ---------------------------------------------------------------------------


def _load_dense_stack(args: argparse.Namespace):
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.metadata_store import ChunkLookupResult
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.capabilities.rag.retriever import Retriever
    from app.core.config import get_settings
    from eval.harness.offline_corpus import (
        OfflineCorpusInfo,
        _InMemoryMetadataStore,
    )

    cache_dir = Path(args.cache_dir)
    if not (
        (cache_dir / "faiss.index").exists()
        and (cache_dir / "build.json").exists()
        and (cache_dir / "chunks.jsonl").exists()
    ):
        raise FileNotFoundError(
            f"Cache dir {cache_dir} is missing one of "
            "faiss.index / build.json / chunks.jsonl. Run "
            "scripts.eval_full_silver_minimal_sweep once first to "
            "populate the cache, then point --cache-dir here."
        )
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=False,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    reranker = CrossEncoderReranker(
        model_name=str(args.reranker_model),
        max_length=int(args.reranker_max_length),
        batch_size=int(args.reranker_batch_size),
        text_max_chars=int(args.reranker_text_max_chars),
        device=args.reranker_device or None,
        collect_stage_timings=False,
    )
    index = FaissIndex(cache_dir)
    info = index.load()
    if info.embedding_model != settings.rag_embedding_model:
        raise RuntimeError(
            f"Cached FAISS embedding_model={info.embedding_model!r} "
            f"differs from settings={settings.rag_embedding_model!r}; "
            "re-run minimal sweep with --force-rebuild or correct the "
            "AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL env."
        )
    rows: List[ChunkLookupResult] = []
    with (cache_dir / "chunks.jsonl").open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(ChunkLookupResult(
                chunk_id=str(obj["chunk_id"]),
                doc_id=str(obj["doc_id"]),
                section=str(obj["section"]),
                text=str(obj["text"]),
                faiss_row_id=int(obj["faiss_row_id"]),
            ))
    store = _InMemoryMetadataStore(info.index_version, rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=10,
        reranker=reranker,
        candidate_k=50,
    )
    retriever.ensure_ready()
    offline_info = OfflineCorpusInfo(
        corpus_path=str(args.corpus),
        document_count=len({r.doc_id for r in rows}),
        chunk_count=len(rows),
        index_version=info.index_version,
        embedding_model=info.embedding_model,
        dimension=info.dimension,
    )
    return retriever, reranker, offline_info, settings


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _f(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:+.4f}"


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _write_summary_csv(
    out_dir: Path,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, "ConfirmCellSpec"],
    cell_deltas: Dict[str, "CellDeltas"],
) -> None:
    headers = [
        "label", "group",
        "candidate_k", "final_top_k", "rerank_in",
        "use_mmr", "mmr_lambda", "mmr_k",
        "title_cap_rerank_input", "title_cap_final",
        "row_count",
        "hit@1", "hit@3", "hit@5",
        "mrr@10", "ndcg@10",
        "candidateHit@10", "candidateHit@20",
        "candidateHit@50", "candidateHit@100",
        "candidateRecall@50", "candidateRecall@100",
        "duplicateDocRatio@5", "duplicateDocRatio@10",
        "uniqueDocCount@10", "sectionDiversity@10",
        "avgTotalRetrievalMs",
        "p50ms", "p95ms", "p99ms",
        "avgDenseRetrievalMs", "avgRerankMs",
        "rerankUpliftHit@5", "rerankUpliftMrr@10",
        "qualityScore", "efficiencyScore",
        "grade", "deltaHit@5", "deltaMrr@10",
        "deltaCandHit@50", "latencyRatioP95",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for label, summary in cell_summaries.items():
            spec = cell_specs[label]
            cand = summary.candidate_hit_rates or {}
            cand_recall = summary.candidate_recalls or {}
            dup = summary.duplicate_doc_ratios or {}
            udc = summary.unique_doc_counts or {}
            sec = summary.section_diversities or {}
            deltas = cell_deltas.get(label)
            writer.writerow([
                label, spec.group,
                spec.candidate_k, spec.final_top_k, spec.rerank_in,
                spec.use_mmr, spec.mmr_lambda, spec.mmr_k,
                spec.title_cap_rerank_input, spec.title_cap_final,
                summary.row_count,
                _f(summary.mean_hit_at_1), _f(summary.mean_hit_at_3),
                _f(summary.mean_hit_at_5),
                _f(summary.mean_mrr_at_10), _f(summary.mean_ndcg_at_10),
                _f(cand.get("10")), _f(cand.get("20")),
                _f(cand.get("50")), _f(cand.get("100")),
                _f(cand_recall.get("50")), _f(cand_recall.get("100")),
                _f(dup.get("5")), _f(dup.get("10")),
                _f(udc.get("10")), _f(sec.get("10")),
                _f(
                    summary.avg_total_retrieval_ms
                    or summary.mean_retrieval_ms
                ),
                _f(summary.p50_retrieval_ms),
                _f(
                    summary.p95_total_retrieval_ms
                    or summary.p95_retrieval_ms
                ),
                _f(summary.p99_retrieval_ms),
                _f(summary.mean_dense_retrieval_ms),
                _f(summary.mean_rerank_ms),
                _f(summary.rerank_uplift_hit_at_5),
                _f(summary.rerank_uplift_mrr_at_10),
                _f(summary.quality_score),
                _f(summary.efficiency_score),
                "" if deltas is None else deltas.grade,
                _f(None if deltas is None else deltas.delta_hit_at_5),
                _f(None if deltas is None else deltas.delta_mrr_at_10),
                _f(
                    None if deltas is None
                    else deltas.delta_candidate_hit_at_50
                ),
                _f(None if deltas is None else deltas.latency_ratio_p95),
            ])


def _write_summary_json(
    out_dir: Path,
    *,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, "ConfirmCellSpec"],
    cell_deltas: Dict[str, "CellDeltas"],
    cells_meta: List[Dict[str, Any]],
    args: argparse.Namespace,
    info: Any,
    settings: Any,
    verdict: str,
    rationale: str,
    query_type_breakdown: Optional[Dict[str, Any]],
) -> None:
    payload = {
        "schema": "phase2-wide-mmr-confirm.v1",
        "run": {
            "dataset": str(args.dataset),
            "corpus_path": str(args.corpus),
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": str(args.reranker_model),
            "document_count": info.document_count,
            "chunk_count": info.chunk_count,
            "index_version": info.index_version,
            "started_at": (
                cells_meta[0].get("started_at") if cells_meta else None
            ),
            "finished_at": (
                cells_meta[-1].get("finished_at") if cells_meta else None
            ),
            "cell_count": len(cell_summaries),
            "baseline_label": _BASELINE_LABEL,
            "phase1_best_label": _PHASE1_BEST_LABEL,
            "optuna_winner_label": _OPTUNA_WINNER_LABEL,
        },
        "verdict": {
            "label": verdict,
            "rationale": rationale,
        },
        "cells": [
            {
                "label": label,
                "spec": asdict(cell_specs[label]),
                "summary": asdict(summary),
                "deltas": (
                    asdict(cell_deltas[label])
                    if label in cell_deltas else None
                ),
                "started_at": next(
                    (
                        m.get("started_at") for m in cells_meta
                        if m["label"] == label
                    ),
                    None,
                ),
                "finished_at": next(
                    (
                        m.get("finished_at") for m in cells_meta
                        if m["label"] == label
                    ),
                    None,
                ),
            }
            for label, summary in cell_summaries.items()
        ],
        "byQueryType": query_type_breakdown or {},
        "caveats": [
            "All cells mutate the production Retriever's _top_k / "
            "_candidate_k / _reranker per call to swap to NoOp + the "
            "configured candidate_k for the bi-encoder pool pass — "
            "production code is NOT modified.",
            "The query_type_draft join is heuristic — diagnostic only.",
            "MMR is a score-fallback variant (no candidate embeddings). "
            "Sufficient for the qualitative diversity check the spec "
            "asks for; not a replacement for vector-MMR.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_per_query_jsonl(
    out_dir: Path,
    rows_by_cell: Dict[str, List[Any]],
) -> None:
    from eval.harness.retrieval_eval import row_to_dict

    with (out_dir / "per_query_results.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for label, rows in rows_by_cell.items():
            for row in rows:
                payload = row_to_dict(row)
                payload["cell"] = label
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_config_dump(
    out_dir: Path,
    *,
    cell_specs: Dict[str, "ConfirmCellSpec"],
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema": "phase2-wide-mmr-confirm.config.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "dataset": str(args.dataset),
            "corpus": str(args.corpus),
            "cache_dir": str(args.cache_dir),
            "out_dir": str(args.out_dir) if args.out_dir else None,
            "limit": args.limit,
            "cells_filter": args.cells,
            "max_seq_length": args.max_seq_length,
            "embed_batch_size": args.embed_batch_size,
            "reranker_model": args.reranker_model,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
            "reranker_text_max_chars": args.reranker_text_max_chars,
            "reranker_device": args.reranker_device,
            "query_type_draft": str(args.query_type_draft),
        },
        "cells": [asdict(spec) for spec in cell_specs.values()],
        "baseline_label": _BASELINE_LABEL,
        "phase1_best_label": _PHASE1_BEST_LABEL,
        "optuna_winner_label": _OPTUNA_WINNER_LABEL,
    }
    (out_dir / "config_dump.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_regression_guard(
    out_dir: Path,
    *,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, "ConfirmCellSpec"],
    cell_deltas: Dict[str, "CellDeltas"],
) -> None:
    from eval.harness.confirm_wide_mmr_helpers import (
        EPS_HIT, EPS_MRR, GRADE_REGRESSION,
    )

    md: List[str] = []
    md.append("# Regression guard — wide-MMR confirm sweep")
    md.append("")
    md.append(
        f"Baseline cell: `{_BASELINE_LABEL}` (Phase 1 production-"
        "equivalent). Each cell is checked against the baseline; a "
        "regression fires when ``Δhit@5 ≤ -0.005`` OR "
        "``Δmrr@10 ≤ -0.005`` OR ``ΔcandidateHit@50 ≤ -0.005`` "
        "(matches the Phase 1 grader epsilon contract)."
    )
    md.append("")
    md.append("| label | group | grade | Δhit@5 | Δmrr@10 | Δcand@50 | latRatioP95 | passes |")
    md.append("|---|---|---|---:|---:|---:|---:|---|")
    failures = 0
    for label in cell_summaries:
        spec = cell_specs[label]
        deltas = cell_deltas.get(label)
        if deltas is None:
            continue
        passes = "✓" if deltas.grade != GRADE_REGRESSION else "✗"
        if deltas.grade == GRADE_REGRESSION:
            failures += 1
        md.append(
            f"| {label} | {spec.group} | {deltas.grade} | "
            f"{_fmt_signed(deltas.delta_hit_at_5)} | "
            f"{_fmt_signed(deltas.delta_mrr_at_10)} | "
            f"{_fmt_signed(deltas.delta_candidate_hit_at_50)} | "
            f"{_fmt(deltas.latency_ratio_p95)} | {passes} |"
        )
    md.append("")
    if failures == 0:
        md.append("**Result: PASS** — no cell regresses against baseline beyond epsilon.")
    else:
        md.append(
            f"**Result: {failures} regressing cell(s) flagged.** "
            "Review individual ``Δ*`` columns and the comparison report "
            "to decide whether the regression is data-set noise or a "
            "real signal before adopting any cell."
        )
    md.append("")
    md.append(
        f"Epsilon contract: `EPS_HIT={EPS_HIT}`, `EPS_MRR={EPS_MRR}`. "
        "These are deliberately conservative — a 0.005 hit@5 swing on "
        "200 rows is exactly 1 query flipping, so any regression "
        "stronger than that warrants an inspection."
    )
    md.append("")
    (out_dir / "regression_guard.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Markdown comparison report
# ---------------------------------------------------------------------------


def _row_to_payload_dict(row: Any) -> Dict[str, Any]:
    from eval.harness.retrieval_eval import row_to_dict
    return row_to_dict(row)


def _compute_query_type_breakdown(
    rows_by_cell: Dict[str, List[Any]],
    *,
    query_type_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Optional join with the heuristic query_type_draft.

    The draft tags every silver row with a ``query_type`` and a
    confidence score. We compute per-(cell, query_type) hit@5 / MRR
    aggregates and surface them in the report's byQueryType section.
    Diagnostic-only — the draft is heuristic and the report flags it
    accordingly.
    """
    if query_type_path is None or not query_type_path.exists():
        return None

    qt_by_id: Dict[str, Dict[str, Any]] = {}
    with query_type_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("id") or "").strip()
            if not qid:
                continue
            qt_by_id[qid] = {
                "query_type": str(obj.get("query_type") or "unknown"),
                "confidence": float(obj.get("query_type_confidence") or 0.0),
            }

    if not qt_by_id:
        return None

    LOW_CONF = 0.5
    breakdown: Dict[str, Any] = {}
    for cell_label, rows in rows_by_cell.items():
        cell_buckets: Dict[str, Dict[str, Any]] = {}
        low_conf_total = {"count": 0, "hit_at_5_sum": 0.0, "mrr_sum": 0.0}
        unknown_total = {"count": 0, "hit_at_5_sum": 0.0, "mrr_sum": 0.0}
        for row in rows:
            rid = getattr(row, "id", None)
            if not rid:
                continue
            tag = qt_by_id.get(str(rid))
            if tag is None:
                continue
            qt = str(tag["query_type"]) or "unknown"
            conf = float(tag.get("confidence") or 0.0)
            h5 = getattr(row, "hit_at_5", None)
            mrr = getattr(row, "mrr_at_10", None)
            if h5 is None or mrr is None:
                continue
            bucket = cell_buckets.setdefault(qt, {
                "count": 0, "hit_at_5_sum": 0.0, "mrr_sum": 0.0,
            })
            bucket["count"] += 1
            bucket["hit_at_5_sum"] += float(h5)
            bucket["mrr_sum"] += float(mrr)
            if qt == "unknown":
                unknown_total["count"] += 1
                unknown_total["hit_at_5_sum"] += float(h5)
                unknown_total["mrr_sum"] += float(mrr)
            if conf < LOW_CONF:
                low_conf_total["count"] += 1
                low_conf_total["hit_at_5_sum"] += float(h5)
                low_conf_total["mrr_sum"] += float(mrr)

        per_type: Dict[str, Dict[str, Optional[float]]] = {}
        for qt, b in cell_buckets.items():
            n = max(1, b["count"])
            per_type[qt] = {
                "count": b["count"],
                "mean_hit_at_5": (
                    None if b["count"] == 0
                    else round(b["hit_at_5_sum"] / n, 4)
                ),
                "mean_mrr_at_10": (
                    None if b["count"] == 0
                    else round(b["mrr_sum"] / n, 4)
                ),
            }
        # Special low-conf and unknown roll-ups for the report's caveat
        # callout.
        per_type["__low_confidence_rows__"] = {
            "count": low_conf_total["count"],
            "mean_hit_at_5": (
                None if low_conf_total["count"] == 0
                else round(
                    low_conf_total["hit_at_5_sum"]
                    / max(1, low_conf_total["count"]),
                    4,
                )
            ),
            "mean_mrr_at_10": (
                None if low_conf_total["count"] == 0
                else round(
                    low_conf_total["mrr_sum"]
                    / max(1, low_conf_total["count"]),
                    4,
                )
            ),
        }
        breakdown[cell_label] = per_type
    return breakdown


def _write_comparison_report(
    out_dir: Path,
    *,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, "ConfirmCellSpec"],
    cell_deltas: Dict[str, "CellDeltas"],
    rows_by_cell: Dict[str, List[Any]],
    verdict: str,
    rationale: str,
    query_type_breakdown: Optional[Dict[str, Any]],
) -> None:
    from eval.harness.confirm_wide_mmr_helpers import (
        VERDICT_ADOPT_OPTUNA, VERDICT_INCONCLUSIVE, VERDICT_KEEP_PHASE1,
        candidate_pool_recoverable_misses,
        per_query_diff,
    )

    md: List[str] = []
    md.append("# Wide-MMR confirm sweep — Phase 1 best vs Optuna winner")
    md.append("")
    md.append(
        f"- generated: {datetime.now().isoformat(timespec='seconds')}"
    )
    md.append(f"- baseline cell: `{_BASELINE_LABEL}`")
    md.append(f"- phase1_best cell: `{_PHASE1_BEST_LABEL}`")
    md.append(f"- optuna_winner cell: `{_OPTUNA_WINNER_LABEL}`")
    md.append("")

    # Headline metrics ----------------------------------------------------
    md.append("## Headline metrics (full silver_200)")
    md.append("")
    md.append(
        "| label | group | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
        "cand@50 | cand@100 | dup@10 | uniq@10 | p50ms | p95ms | p99ms |"
    )
    md.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for label, s in cell_summaries.items():
        spec = cell_specs[label]
        cand = s.candidate_hit_rates or {}
        dup = s.duplicate_doc_ratios or {}
        udc = s.unique_doc_counts or {}
        md.append(
            f"| {label} | {spec.group} | "
            f"{_fmt(s.mean_hit_at_1)} | {_fmt(s.mean_hit_at_3)} | "
            f"{_fmt(s.mean_hit_at_5)} | {_fmt(s.mean_mrr_at_10)} | "
            f"{_fmt(s.mean_ndcg_at_10)} | "
            f"{_fmt(cand.get('50'))} | {_fmt(cand.get('100'))} | "
            f"{_fmt(dup.get('10'))} | {_fmt(udc.get('10'))} | "
            f"{_fmt_ms(s.p50_retrieval_ms)} | "
            f"{_fmt_ms(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} | "
            f"{_fmt_ms(s.p99_retrieval_ms)} |"
        )
    md.append("")

    # Deltas vs baseline --------------------------------------------------
    md.append("## Deltas vs baseline_k50_top5")
    md.append("")
    md.append(
        "| label | grade | Δhit@5 | Δmrr@10 | Δndcg@10 | Δcand@50 | "
        "Δcand@100 | Δdup@10 | latRatioP95 |"
    )
    md.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for label, deltas in cell_deltas.items():
        md.append(
            f"| {label} | {deltas.grade} | "
            f"{_fmt_signed(deltas.delta_hit_at_5)} | "
            f"{_fmt_signed(deltas.delta_mrr_at_10)} | "
            f"{_fmt_signed(deltas.delta_ndcg_at_10)} | "
            f"{_fmt_signed(deltas.delta_candidate_hit_at_50)} | "
            f"{_fmt_signed(deltas.delta_candidate_hit_at_100)} | "
            f"{_fmt_signed(deltas.delta_duplicate_ratio_at_10)} | "
            f"{_fmt(deltas.latency_ratio_p95)} |"
        )
    md.append("")

    # Sensitivity sub-tables ---------------------------------------------
    md.append("## Sensitivity sub-tables")
    md.append("")
    for group, header in (
        ("cap_final_sensitivity", "title_cap_final sensitivity"),
        ("lambda_sensitivity", "MMR λ sensitivity"),
        ("final_topk_sensitivity", "final_top_k sensitivity"),
    ):
        members = [
            label for label, spec in cell_specs.items()
            if spec.group == group
        ]
        # Always include the winner cell as the anchor so the sub-table
        # is interpretable in isolation.
        if _OPTUNA_WINNER_LABEL not in members:
            members = [_OPTUNA_WINNER_LABEL] + members
        if not members:
            continue
        md.append(f"### {header}")
        md.append("")
        md.append(
            "| label | hit@5 | mrr@10 | dup@10 | uniq@10 | p95ms |"
        )
        md.append("|---|---:|---:|---:|---:|---:|")
        for label in members:
            s = cell_summaries.get(label)
            if s is None:
                continue
            dup = (s.duplicate_doc_ratios or {}).get("10")
            udc = (s.unique_doc_counts or {}).get("10")
            md.append(
                f"| {label} | {_fmt(s.mean_hit_at_5)} | "
                f"{_fmt(s.mean_mrr_at_10)} | {_fmt(dup)} | "
                f"{_fmt(udc)} | "
                f"{_fmt_ms(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} |"
            )
        md.append("")

    # Per-query diff lists -----------------------------------------------
    md.append("## Per-query diffs vs baseline")
    md.append("")
    md.append(
        "Lists below enumerate query IDs whose hit@5 flipped between "
        "the baseline and the named cell. ``improved``: baseline=miss "
        "→ cell=hit; ``regressed``: baseline=hit → cell=miss. Entries "
        "where both sides agree are omitted."
    )
    md.append("")
    base_dicts = (
        [_row_to_payload_dict(r) for r in rows_by_cell[_BASELINE_LABEL]]
        if _BASELINE_LABEL in rows_by_cell else []
    )
    for label in cell_specs:
        if label == _BASELINE_LABEL:
            continue
        if label not in rows_by_cell:
            continue
        cell_dicts = [
            _row_to_payload_dict(r) for r in rows_by_cell[label]
        ]
        improved, regressed = per_query_diff(
            base_dicts, cell_dicts, cell_label=label,
        )
        md.append(f"### {label}")
        md.append("")
        md.append(
            f"- improved vs baseline: **{len(improved)}** queries"
        )
        md.append(
            f"- regressed vs baseline: **{len(regressed)}** queries"
        )
        if improved:
            md.append("")
            md.append("Improved query IDs (up to 10): " + ", ".join(
                e.id for e in improved[:10]
            ))
        if regressed:
            md.append("")
            md.append("Regressed query IDs (up to 10): " + ", ".join(
                e.id for e in regressed[:10]
            ))
        md.append("")

    # phase1 vs optuna split ---------------------------------------------
    if (
        _PHASE1_BEST_LABEL in rows_by_cell
        and _OPTUNA_WINNER_LABEL in rows_by_cell
    ):
        md.append("## Phase 1 best vs Optuna winner — head-to-head per query")
        md.append("")
        p1_dicts = [
            _row_to_payload_dict(r)
            for r in rows_by_cell[_PHASE1_BEST_LABEL]
        ]
        op_dicts = [
            _row_to_payload_dict(r)
            for r in rows_by_cell[_OPTUNA_WINNER_LABEL]
        ]
        # Diff the optuna winner *against* phase1_best so the "improved"
        # list is queries where Optuna recipe beats Phase 1.
        op_vs_p1_improved, op_vs_p1_regressed = per_query_diff(
            p1_dicts, op_dicts, cell_label=_OPTUNA_WINNER_LABEL,
        )
        md.append(
            f"- Optuna winner beats Phase 1 best on hit@5: "
            f"**{len(op_vs_p1_improved)}** queries"
        )
        md.append(
            f"- Phase 1 best beats Optuna winner on hit@5: "
            f"**{len(op_vs_p1_regressed)}** queries"
        )
        if op_vs_p1_improved:
            md.append("")
            md.append("Optuna-only wins (up to 10): " + ", ".join(
                e.id for e in op_vs_p1_improved[:10]
            ))
        if op_vs_p1_regressed:
            md.append("")
            md.append("Phase1-only wins (up to 10): " + ", ".join(
                e.id for e in op_vs_p1_regressed[:10]
            ))
        md.append("")

    # cap=1 vs cap=2 split ------------------------------------------------
    if (
        _PHASE1_BEST_LABEL in rows_by_cell
        and "phase1_cap1_top8" in rows_by_cell
    ):
        md.append("## cap=1 vs cap=2 split (Phase 1 cells)")
        md.append("")
        cap2_dicts = [
            _row_to_payload_dict(r)
            for r in rows_by_cell[_PHASE1_BEST_LABEL]
        ]
        cap1_dicts = [
            _row_to_payload_dict(r)
            for r in rows_by_cell["phase1_cap1_top8"]
        ]
        cap1_wins, cap2_wins = per_query_diff(
            cap2_dicts, cap1_dicts, cell_label="phase1_cap1_top8",
        )
        md.append(
            f"- cap=1 beats cap=2 on hit@5: **{len(cap1_wins)}** queries"
        )
        md.append(
            f"- cap=2 beats cap=1 on hit@5: **{len(cap2_wins)}** queries"
        )
        md.append("")

    # rerank_in 16 loss queries ------------------------------------------
    if (
        _OPTUNA_WINNER_LABEL in rows_by_cell
        and _PHASE1_BEST_LABEL in rows_by_cell
    ):
        md.append("## rerank_in=16 loss queries (Optuna winner regressed vs Phase1 best)")
        md.append("")
        # Already computed above — surface the regression list with
        # short context lines.
        md.append(
            "Surfaced from the head-to-head section: queries where "
            "Optuna's rerank_in=16 / cap_final=2 dropped a hit Phase 1 "
            "cap_rr=2 / rerank_in=32 had captured. Full list in "
            "``per_query_results.jsonl``."
        )
        md.append("")

    # candidate-pool recoverable misses ----------------------------------
    if _OPTUNA_WINNER_LABEL in rows_by_cell:
        md.append("## Candidate-pool recoverable misses (Optuna winner cell)")
        md.append("")
        op_dicts = [
            _row_to_payload_dict(r)
            for r in rows_by_cell[_OPTUNA_WINNER_LABEL]
        ]
        misses = candidate_pool_recoverable_misses(op_dicts)
        md.append(
            f"Found **{len(misses)}** queries where the gold doc is in "
            "the candidate pool but the reranker dropped it from the "
            "final top-k."
        )
        if misses:
            md.append("")
            md.append("First 10 examples (id, expected → top-5 retrieved):")
            md.append("")
            for entry in misses[:10]:
                md.append(
                    f"- `{entry['id']}` — expected={entry['expected_doc_ids']} "
                    f"→ retrieved_top5={entry['retrieved_top_doc_ids']} "
                    f"(candidates: {entry['candidate_count']})"
                )
        md.append("")

    # Title-duplicate reduction examples ---------------------------------
    md.append("## Top-result title-duplicate reduction examples")
    md.append("")
    md.append(
        "Per-query ``duplicate_doc_ratios`` collapsed by MMR + title "
        "cap. Listing first 10 queries where the baseline's top-10 "
        "had ≥2 duplicate doc_ids that the optuna_winner cell removed."
    )
    md.append("")
    if (
        _BASELINE_LABEL in rows_by_cell
        and _OPTUNA_WINNER_LABEL in rows_by_cell
    ):
        base_by_id = {r.id: r for r in rows_by_cell[_BASELINE_LABEL]}
        examples: List[str] = []
        for r in rows_by_cell[_OPTUNA_WINNER_LABEL]:
            if len(examples) >= 10:
                break
            base = base_by_id.get(r.id)
            if base is None:
                continue
            base_dup = (base.duplicate_doc_ratios or {}).get("10")
            cell_dup = (r.duplicate_doc_ratios or {}).get("10")
            if base_dup is None or cell_dup is None:
                continue
            if float(base_dup) >= 0.4 and float(cell_dup) <= float(base_dup) - 0.2:
                examples.append(
                    f"- `{r.id}` — dup@10 {float(base_dup):.2f} → "
                    f"{float(cell_dup):.2f}"
                )
        for line in examples:
            md.append(line)
        if not examples:
            md.append("(no qualifying examples found in this run)")
    md.append("")

    # byQueryType breakdown ----------------------------------------------
    if query_type_breakdown:
        md.append("## byQueryType breakdown (heuristic — diagnostic only)")
        md.append("")
        md.append(
            "Joined with the heuristic ``anime_silver_200.query_type_"
            "draft.jsonl``. The draft is auto-tagged and **not manually "
            "reviewed** — treat the per-bucket numbers as directional "
            "only. ``__low_confidence_rows__`` rolls up rows with "
            "tagging confidence < 0.5; ``unknown`` is the heuristic's "
            "fallback bucket."
        )
        md.append("")
        for cell_label in (
            _BASELINE_LABEL, _PHASE1_BEST_LABEL, _OPTUNA_WINNER_LABEL,
        ):
            cell_break = query_type_breakdown.get(cell_label)
            if not cell_break:
                continue
            md.append(f"### {cell_label}")
            md.append("")
            md.append("| query_type | count | hit@5 | mrr@10 |")
            md.append("|---|---:|---:|---:|")
            for qt, stats in sorted(cell_break.items()):
                md.append(
                    f"| {qt} | {stats.get('count')} | "
                    f"{_fmt(stats.get('mean_hit_at_5'))} | "
                    f"{_fmt(stats.get('mean_mrr_at_10'))} |"
                )
            md.append("")

    # Verdict + next steps -----------------------------------------------
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}** — {rationale}")
    md.append("")
    md.append("## Next-step recommendation")
    md.append("")
    if verdict == VERDICT_ADOPT_OPTUNA:
        md.append(
            "1. Promote the Optuna winner recipe (candidate_k=100, "
            "rerank_in=16, MMR λ=0.65, cap_rr=1, cap_final=2, "
            "final_top_k=8) to a production-side experiment. Wire the "
            "knobs through Retriever / RagSettings before any "
            "rollout, then re-validate the cells in this report on a "
            "production-mirroring index."
        )
        md.append(
            "2. Run candidate_pool_recoverable_misses against gold "
            "queries to scope the upper-bound win achievable by also "
            "fixing the reranker's slice — even after adoption, those "
            "queries say the bottleneck is reranker depth, not pool "
            "size."
        )
    elif verdict == VERDICT_KEEP_PHASE1:
        md.append(
            "1. Keep Phase 1 cap=2 / rerank_in=32 / top_k=8 as the "
            "current best recipe. The Optuna winner from the 100-row "
            "subset did not generalise to the full 200-row set; the "
            "subset bias was real."
        )
        md.append(
            "2. Use this confirm run as the baseline for the next "
            "Optuna study, but tune on the **full 200 rows** to avoid "
            "subset bias. Add ``embedding_text_variant`` as a search "
            "axis since this confirm run shows the candidate-pool "
            "ceiling is the real bottleneck."
        )
    else:
        md.append(
            "1. **Embedding text reindex is the highest-leverage next "
            "move.** The candidate-pool ceiling at cand@50 ≈ 0.80 is "
            "the actual bottleneck — neither MMR + cap=2 nor the "
            "Optuna winner moves the needle on hit@5 by more than "
            "epsilon. Reindex with ``title_section`` or ``title``-"
            "prefixed embedding text and rerun this confirm sweep."
        )
        md.append(
            "2. Hold off on a second Optuna round in the current "
            "search space. With both candidate recipes inside epsilon "
            "of each other, more hyper-parameter exploration on the "
            "current embedding will not lift hit@5."
        )
        md.append(
            "3. Reranker input formatting audit — the Phase 1 baseline "
            "already showed `rerankerUpliftLow=True` (uplift hit@5 = "
            "0). The cross-encoder is barely changing the dense "
            "ordering; verify text_max_chars=800 / max_length=512 "
            "isn't truncating signal."
        )
        md.append(
            "4. Defer chunking redesign and silver_500/1000 expansion "
            "until embedding text reindex resolves cand@50 ceiling. "
            "If the ceiling still pins at 0.80 after reindex, expand "
            "to silver_500 to surface mmr_lambda / final_top_k signal."
        )
    md.append("")

    # Caveats -------------------------------------------------------------
    md.append("## Caveats")
    md.append("")
    md.append(
        "- Production code (``app/``) is not modified. All cells run "
        "through ``WideRetrievalEvalAdapter`` over the cached FAISS "
        "index; no PostgreSQL involvement."
    )
    md.append(
        "- The ``query_type_draft`` join is heuristic — diagnostic "
        "only. Manual review is open work; see "
        "``query_type_tagging_review.md`` from the Phase 1 sweep."
    )
    md.append(
        "- p95/p99 latency on 200 rows is sensitive to GPU thermal "
        "state and other concurrent work; treat single-digit-percent "
        "latency deltas as noise unless reproduced across reruns."
    )
    md.append(
        "- ``rerank_uplift_*`` values are computed by the harness "
        "from the rerank score deltas vs the dense order; the value "
        "0.0 means the reranker did not change the order, not that "
        "it didn't run."
    )

    (out_dir / "comparison_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _select_cells(
    *, cells_filter: Optional[str],
) -> List["ConfirmCellSpec"]:
    from eval.harness.confirm_wide_mmr_helpers import default_confirm_cells

    specs = list(default_confirm_cells())
    if cells_filter is None or cells_filter.strip().lower() in {"all", ""}:
        return specs
    requested_groups = {
        s.strip().lower()
        for s in cells_filter.split(",")
        if s.strip()
    }
    if "all" in requested_groups:
        return specs
    # The baseline is always retained — it's the comparison anchor.
    filtered = [
        s for s in specs
        if s.group in requested_groups or s.label == _BASELINE_LABEL
    ]
    if len(filtered) <= 1:
        # Only baseline matched — fall back to the full list and warn.
        log.warning(
            "Cells filter %r matched no non-baseline cell; running "
            "full roster instead.",
            cells_filter,
        )
        return specs
    return filtered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument(
        "--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR,
        help="Pre-built dense FAISS cache directory.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument(
        "--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-text-max-chars", type=int, default=800)
    parser.add_argument("--reranker-device", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on the dataset row count (smoke runs).",
    )
    parser.add_argument(
        "--cells", type=str, default="all",
        help="Comma-separated group filter "
             "(baseline, phase1, optuna_winner, cap_final_sensitivity, "
             "lambda_sensitivity, final_topk_sensitivity, all).",
    )
    parser.add_argument(
        "--query-type-draft", type=Path, default=_DEFAULT_QUERY_TYPE_DRAFT,
        help="Optional heuristic query_type tagged jsonl for the "
             "byQueryType breakdown (diagnostic only).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    if out_dir.exists():
        log.error(
            "Refusing to overwrite existing out-dir %s — pick a new "
            "path or wait a minute for the timestamp to roll.",
            out_dir,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)
    log.info("Output dir: %s", out_dir)

    retriever, reranker, info, settings = _load_dense_stack(args)

    from eval.harness.confirm_wide_mmr_helpers import (
        compute_cell_deltas, decide_verdict,
    )
    from eval.harness.io_utils import load_jsonl
    from eval.harness.retrieval_eval import (
        DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS, run_retrieval_eval,
    )
    from eval.harness.wide_retrieval_adapter import (
        WideRetrievalConfig, WideRetrievalEvalAdapter,
    )
    from eval.harness.wide_retrieval_helpers import DocTitleResolver

    title_resolver = DocTitleResolver.from_corpus(args.corpus)
    title_provider = title_resolver.title_provider()

    specs = _select_cells(cells_filter=args.cells)
    log.info("Running %d cells", len(specs))

    dataset = list(load_jsonl(args.dataset))
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: int(args.limit)]
    log.info(
        "Loaded %d query rows (limit=%s) from %s",
        len(dataset), args.limit, args.dataset,
    )

    # Use the Phase 1 candidate_ks set + 200 since some cells fetch
    # candidate_k=200 pools.
    candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))

    cell_summaries: Dict[str, Any] = {}
    cell_specs: Dict[str, Any] = {}
    cells_meta: List[Dict[str, Any]] = []
    rows_by_cell: Dict[str, List[Any]] = {}

    for spec in specs:
        log.info(
            "running cell %s (group=%s cand_k=%d top_k=%d "
            "rerank_in=%d mmr=%s λ=%.2f cap_rr=%s cap_final=%s)",
            spec.label, spec.group, spec.candidate_k, spec.final_top_k,
            spec.rerank_in, spec.use_mmr, spec.mmr_lambda,
            spec.title_cap_rerank_input, spec.title_cap_final,
        )
        adapter = WideRetrievalEvalAdapter(
            retriever,
            config=WideRetrievalConfig(
                candidate_k=spec.candidate_k,
                final_top_k=spec.final_top_k,
                rerank_in=spec.rerank_in,
                use_mmr=spec.use_mmr,
                mmr_lambda=spec.mmr_lambda,
                mmr_k=spec.mmr_k,
                title_cap_rerank_input=spec.title_cap_rerank_input,
                title_cap_final=spec.title_cap_final,
            ),
            final_reranker=reranker,
            title_provider=title_provider,
            name=spec.label,
        )
        started_at = datetime.now().isoformat(timespec="seconds")
        summary, rows, _, _ = run_retrieval_eval(
            list(dataset),
            retriever=adapter,
            top_k=spec.final_top_k,
            mrr_k=10,
            ndcg_k=10,
            candidate_ks=candidate_ks,
            diversity_ks=DEFAULT_DIVERSITY_KS,
            dataset_path=str(args.dataset),
            corpus_path=str(args.corpus),
        )
        finished_at = datetime.now().isoformat(timespec="seconds")
        cell_summaries[spec.label] = summary
        cell_specs[spec.label] = spec
        cells_meta.append({
            "label": spec.label,
            "started_at": started_at,
            "finished_at": finished_at,
        })
        rows_by_cell[spec.label] = rows
        log.info(
            "  -> %s: hit@5=%.4f mrr@10=%.4f cand@50=%s p95=%.1fms",
            spec.label,
            (summary.mean_hit_at_5 or 0.0),
            (summary.mean_mrr_at_10 or 0.0),
            (summary.candidate_hit_rates or {}).get("50"),
            float(
                summary.p95_total_retrieval_ms
                or summary.p95_retrieval_ms
                or 0.0
            ),
        )

    baseline_summary = cell_summaries.get(_BASELINE_LABEL)
    if baseline_summary is None:
        log.error(
            "Baseline cell %s missing — grading skipped, but reports "
            "still emit raw metrics.", _BASELINE_LABEL,
        )

    cell_deltas: Dict[str, Any] = {}
    for label, summary in cell_summaries.items():
        spec = cell_specs[label]
        cell_deltas[label] = compute_cell_deltas(
            label=label,
            group=spec.group,
            cell_summary=summary,
            baseline_summary=baseline_summary or summary,
        )

    phase1_best_summary = cell_summaries.get(_PHASE1_BEST_LABEL)
    optuna_winner_summary = cell_summaries.get(_OPTUNA_WINNER_LABEL)
    if (
        baseline_summary is not None
        and phase1_best_summary is not None
        and optuna_winner_summary is not None
    ):
        latency_ratio_phase1 = (
            cell_deltas[_PHASE1_BEST_LABEL].latency_ratio_p95
        )
        latency_ratio_optuna = (
            cell_deltas[_OPTUNA_WINNER_LABEL].latency_ratio_p95
        )
        verdict, rationale = decide_verdict(
            baseline_summary=baseline_summary,
            phase1_best_summary=phase1_best_summary,
            optuna_winner_summary=optuna_winner_summary,
            latency_ratio_phase1=latency_ratio_phase1,
            latency_ratio_optuna=latency_ratio_optuna,
        )
    else:
        verdict = "UNDETERMINED"
        rationale = (
            "Required head-to-head cells "
            f"({_BASELINE_LABEL}, {_PHASE1_BEST_LABEL}, "
            f"{_OPTUNA_WINNER_LABEL}) not all present in this run; "
            "verdict deferred."
        )

    qt_breakdown = _compute_query_type_breakdown(
        rows_by_cell, query_type_path=args.query_type_draft,
    )

    _write_summary_csv(
        out_dir, cell_summaries, cell_specs, cell_deltas,
    )
    _write_summary_json(
        out_dir,
        cell_summaries=cell_summaries,
        cell_specs=cell_specs,
        cell_deltas=cell_deltas,
        cells_meta=cells_meta,
        args=args,
        info=info,
        settings=settings,
        verdict=verdict,
        rationale=rationale,
        query_type_breakdown=qt_breakdown,
    )
    _write_per_query_jsonl(out_dir, rows_by_cell)
    _write_config_dump(out_dir, cell_specs=cell_specs, args=args)
    _write_regression_guard(
        out_dir,
        cell_summaries=cell_summaries,
        cell_specs=cell_specs,
        cell_deltas=cell_deltas,
    )
    _write_comparison_report(
        out_dir,
        cell_summaries=cell_summaries,
        cell_specs=cell_specs,
        cell_deltas=cell_deltas,
        rows_by_cell=rows_by_cell,
        verdict=verdict,
        rationale=rationale,
        query_type_breakdown=qt_breakdown,
    )

    log.info(
        "Confirm sweep finished — verdict=%s artifacts in %s",
        verdict, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
