"""Wide retrieval + MMR + title-cap diagnostic sweep.

Phase 2 follow-up to ``eval_full_silver_minimal_sweep``. The minimal
sweep compared dense / BM25 / hybrid at the *current* candidate_k=50
ceiling. This sweep reuses the same FAISS cache + bi-encoder + cross-
encoder stack but explores three orthogonal knobs on the dense path:

  - candidate pool size: 50 / 100 / 200
  - eval-only MMR (score-fallback) over the pool
  - title cap (1 / 2) on the rerank input pool
  - rerank input slice size (24 / 32 / 40)
  - final top-k 5 / 8 / 10

Goal: find out whether the current 0.71 hit@5 / 0.80 candidateHit@50
ceiling is bottlenecked by the candidate pool (lift candidate_k to
100/200 and check), by reranker input size (32→40), or by lack of
diversity in the pool (MMR + title cap).

The sweep does NOT touch production. It mutates the production
``Retriever``'s candidate_k / top_k / reranker per call (same idiom
the minimal sweep uses) and runs an eval-only MMR / title-cap
selector before / after the cross-encoder.

Outputs land under ``eval/reports/_archive/confirm-runs/retrieval-wide-mmr-titlecap-<TS>/``.

Cells targeted by the spec (run all 9 by default; use ``--cells
core6`` to drop the four lambda variants):

  1. dense_baseline_k50_top5
  2. dense_baseline_k100_top5
  3. dense_baseline_k200_top5
  4. dense_wide_mmr_cap2_top5
  5. dense_wide_mmr_cap2_top8
  6. dense_wide_mmr_cap1_top8
  7. dense_wide_mmr_cap2_rerank40_top8
  8. dense_wide_mmr_cap2_lambda60_top8
  9. dense_wide_mmr_cap2_lambda70_top8

Run::

    python -m scripts.eval_wide_mmr_titlecap_sweep \\
        --dataset eval/eval_queries/anime_silver_200.jsonl \\
        --corpus eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \\
        --cache-dir eval/agent_loop_ab/_indexes/bge-m3-anime-namu-v3-raw-mseq1024
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("eval_wide_mmr_titlecap_sweep")


_DEFAULT_DATASET = Path("eval/eval_queries/anime_silver_200.jsonl")
_DEFAULT_CORPUS = Path(
    "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl"
)
_DEFAULT_REPORTS_ROOT = Path("eval/reports/_archive/confirm-runs")
_DEFAULT_CACHE_ROOT = Path("eval/_cache/dense_index")
_BASELINE_LABEL = "dense_baseline_k50_top5"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return (
        _DEFAULT_REPORTS_ROOT
        / f"retrieval-wide-mmr-titlecap-{_now_stamp()}"
    )


# ---------------------------------------------------------------------------
# Cell definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One sweep cell — knobs only, retriever wired by the driver."""

    label: str
    candidate_k: int
    final_top_k: int
    rerank_in: int
    use_mmr: bool = False
    mmr_lambda: float = 0.65
    mmr_k: int = 64
    title_cap_rerank_input: Optional[int] = None
    title_cap_final: Optional[int] = None


def _spec_grid(*, full: bool) -> List[CellSpec]:
    """Return the cell list — full 9 or core 6.

    ``rerank_in`` defaults to 32 except for the rerank40 cell. The
    spec lists rerank_in 24/32/40 but to keep the grid tractable we
    fix at 32 for most cells (closest to the existing pool_size=50)
    and add one rerank40 cell to test the upper bound.
    """
    base_specs = [
        # Baselines: candidate_k progression, no MMR / cap.
        CellSpec(
            label="dense_baseline_k50_top5",
            candidate_k=50, final_top_k=5, rerank_in=32,
        ),
        CellSpec(
            label="dense_baseline_k100_top5",
            candidate_k=100, final_top_k=5, rerank_in=32,
        ),
        CellSpec(
            label="dense_baseline_k200_top5",
            candidate_k=200, final_top_k=5, rerank_in=32,
        ),
        # Prior strategy cells — MMR + title cap on a wider pool.
        CellSpec(
            label="dense_wide_mmr_cap2_top5",
            candidate_k=200, final_top_k=5, rerank_in=32,
            use_mmr=True, mmr_lambda=0.65, mmr_k=64,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
        CellSpec(
            label="dense_wide_mmr_cap2_top8",
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.65, mmr_k=64,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
        CellSpec(
            label="dense_wide_mmr_cap1_top8",
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.65, mmr_k=64,
            title_cap_rerank_input=1, title_cap_final=1,
        ),
    ]
    if not full:
        return base_specs
    base_specs.extend([
        CellSpec(
            label="dense_wide_mmr_cap2_rerank40_top8",
            candidate_k=200, final_top_k=8, rerank_in=40,
            use_mmr=True, mmr_lambda=0.65, mmr_k=80,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
        CellSpec(
            label="dense_wide_mmr_cap2_lambda60_top8",
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.60, mmr_k=64,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
        CellSpec(
            label="dense_wide_mmr_cap2_lambda70_top8",
            candidate_k=200, final_top_k=8, rerank_in=32,
            use_mmr=True, mmr_lambda=0.70, mmr_k=64,
            title_cap_rerank_input=2, title_cap_final=2,
        ),
    ])
    return base_specs


# ---------------------------------------------------------------------------
# Dense stack reuse — borrows the cache layout from
# ``eval_full_silver_minimal_sweep``.
# ---------------------------------------------------------------------------


def _load_dense_stack(args: argparse.Namespace):
    """Load the bi-encoder + reranker stack from a pre-built cache dir.

    This driver does NOT (re)build the FAISS index — it requires the
    cache directory to already contain ``faiss.index``, ``build.json``,
    and ``chunks.jsonl``. Building should be done via
    ``eval_full_silver_minimal_sweep`` (which populates the cache).
    """
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
# Sweep driver
# ---------------------------------------------------------------------------


def _f(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _grade_cell(
    *,
    cell_summary: Any,
    baseline_summary: Any,
) -> Dict[str, Any]:
    """Compute the conservative cell grade per the spec.

    Returns a dict carrying ``grade``, ``reason``, and the deltas the
    grader looked at. Grades:
      - promising_quality
      - promising_latency
      - diagnostic_only
      - regression
      - inconclusive
    """
    EPS_HIT = 0.005
    EPS_MRR = 0.005
    THRESHOLD_PROMISING_HIT = 0.01
    THRESHOLD_PROMISING_MRR = 0.01
    LATENCY_RATIO_LIMIT = 1.5

    def _delta(curr, base):
        if curr is None or base is None:
            return None
        return round(float(curr) - float(base), 6)

    d_h5 = _delta(
        cell_summary.mean_hit_at_5, baseline_summary.mean_hit_at_5,
    )
    d_mrr = _delta(
        cell_summary.mean_mrr_at_10, baseline_summary.mean_mrr_at_10,
    )
    d_cand50 = _delta(
        (cell_summary.candidate_hit_rates or {}).get("50"),
        (baseline_summary.candidate_hit_rates or {}).get("50"),
    )
    p95_curr = (
        cell_summary.p95_total_retrieval_ms or cell_summary.p95_retrieval_ms
    )
    p95_base = (
        baseline_summary.p95_total_retrieval_ms
        or baseline_summary.p95_retrieval_ms
    )
    latency_ratio = None
    if p95_curr is not None and p95_base is not None and p95_base > 0:
        latency_ratio = round(float(p95_curr) / float(p95_base), 4)

    dup_curr = (cell_summary.duplicate_doc_ratios or {}).get("10")
    dup_base = (baseline_summary.duplicate_doc_ratios or {}).get("10")
    dup_delta = _delta(dup_curr, dup_base)

    # Regression first — any clear quality drop dominates.
    if (
        (d_h5 is not None and d_h5 <= -EPS_HIT)
        or (d_mrr is not None and d_mrr <= -EPS_MRR)
        or (d_cand50 is not None and d_cand50 <= -EPS_HIT)
    ):
        grade = "regression"
        reason = (
            f"Δhit@5={d_h5} Δmrr={d_mrr} Δcand@50={d_cand50}"
        )
    elif (
        (d_h5 is not None and d_h5 >= THRESHOLD_PROMISING_HIT)
        or (d_mrr is not None and d_mrr >= THRESHOLD_PROMISING_MRR)
    ):
        if latency_ratio is not None and latency_ratio > LATENCY_RATIO_LIMIT:
            grade = "diagnostic_only"
            reason = (
                f"quality up (Δhit@5={d_h5}, Δmrr={d_mrr}) but "
                f"latency ratio {latency_ratio:.2f}x > {LATENCY_RATIO_LIMIT}x"
            )
        else:
            grade = "promising_quality"
            reason = (
                f"Δhit@5={d_h5} Δmrr={d_mrr} latency_ratio="
                f"{latency_ratio}"
            )
    elif latency_ratio is not None and latency_ratio < 0.7:
        grade = "promising_latency"
        reason = (
            f"latency ratio {latency_ratio:.2f}x with quality "
            f"deltas Δhit@5={d_h5} Δmrr={d_mrr}"
        )
    elif (
        (d_h5 is None or abs(d_h5) < EPS_HIT)
        and (d_mrr is None or abs(d_mrr) < EPS_MRR)
    ):
        grade = "inconclusive"
        reason = (
            f"deltas within epsilon (Δhit@5={d_h5}, Δmrr={d_mrr})"
        )
    else:
        grade = "diagnostic_only"
        reason = (
            f"Δhit@5={d_h5} Δmrr={d_mrr} Δcand@50={d_cand50} "
            f"latency_ratio={latency_ratio}"
        )

    return {
        "grade": grade,
        "reason": reason,
        "delta_hit_at_5": d_h5,
        "delta_mrr_at_10": d_mrr,
        "delta_candidate_hit_at_50": d_cand50,
        "delta_duplicate_ratio_at_10": dup_delta,
        "latency_ratio_p95": latency_ratio,
    }


def _persist(
    out_dir: Path,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, CellSpec],
    cells_meta: List[Dict[str, Any]],
    grades: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    info: Any,
    settings: Any,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) sweep_summary.json — full per-cell summary as JSON.
    sweep_summary = {
        "schema": "phase2-wide-mmr-titlecap.v1",
        "run": {
            "dataset": str(args.dataset),
            "corpus_path": str(args.corpus),
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": str(args.reranker_model),
            "document_count": info.document_count,
            "chunk_count": info.chunk_count,
            "index_version": info.index_version,
            "started_at": cells_meta[0].get("started_at") if cells_meta else None,
            "finished_at": cells_meta[-1].get("finished_at") if cells_meta else None,
            "cell_count": len(cell_summaries),
            "baseline_label": _BASELINE_LABEL,
        },
        "cells": [
            {
                "label": label,
                "spec": asdict(cell_specs[label]),
                "summary": asdict(summary),
                "grade": grades.get(label, {}),
                "started_at": next(
                    (m.get("started_at") for m in cells_meta if m["label"] == label),
                    None,
                ),
                "finished_at": next(
                    (m.get("finished_at") for m in cells_meta if m["label"] == label),
                    None,
                ),
            }
            for label, summary in cell_summaries.items()
        ],
        "caveats": [
            "Every cell mutates the production Retriever's _top_k / "
            "_candidate_k / _reranker per call to swap to NoOp + the "
            "configured candidate_k for the bi-encoder pool pass.",
            "MMR is a score-fallback variant (no candidate embeddings); "
            "it uses doc_id + title penalty over rerank/bi-encoder "
            "score. This is sufficient for the qualitative diversity "
            "check the spec requested but is NOT a replacement for "
            "vector-MMR if production adoption is later considered.",
            "title_cap relies on a corpus-side title resolver "
            "(eval/corpora/anime_namu_v3_token_chunked/...). When the "
            "resolver can't find a title for a doc_id, title_cap "
            "collapses to doc_id-cap.",
        ],
    }
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) cell_comparison.csv — flat headline comparison.
    headers = [
        "label", "candidate_k", "final_top_k", "rerank_in",
        "use_mmr", "mmr_lambda", "mmr_k",
        "title_cap_rerank_input", "title_cap_final",
        "row_count",
        "hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@10",
        "candidateHit@10", "candidateHit@20", "candidateHit@50",
        "candidateHit@100",
        "candidateRecall@50", "candidateRecall@100",
        "duplicateDocRatio@5", "duplicateDocRatio@10",
        "uniqueDocCount@10", "sectionDiversity@10",
        "avgTotalRetrievalMs", "p95TotalRetrievalMs",
        "avgDenseRetrievalMs", "avgRerankMs",
        "qualityScore", "efficiencyScore",
        "grade", "deltaHit@5", "deltaMrr@10",
        "deltaCandidateHit@50", "latencyRatioP95",
    ]
    with (out_dir / "cell_comparison.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for label, summary in cell_summaries.items():
            spec = cell_specs[label]
            cand = summary.candidate_hit_rates or {}
            cand_recall = summary.candidate_recalls or {}
            dup = summary.duplicate_doc_ratios or {}
            udc = summary.unique_doc_counts or {}
            sec = summary.section_diversities or {}
            grade = grades.get(label, {})
            writer.writerow([
                label, spec.candidate_k, spec.final_top_k, spec.rerank_in,
                spec.use_mmr, spec.mmr_lambda, spec.mmr_k,
                spec.title_cap_rerank_input, spec.title_cap_final,
                summary.row_count,
                _f(summary.mean_hit_at_1), _f(summary.mean_hit_at_3),
                _f(summary.mean_hit_at_5), _f(summary.mean_mrr_at_10),
                _f(summary.mean_ndcg_at_10),
                _f(cand.get("10")), _f(cand.get("20")),
                _f(cand.get("50")), _f(cand.get("100")),
                _f(cand_recall.get("50")), _f(cand_recall.get("100")),
                _f(dup.get("5")), _f(dup.get("10")),
                _f(udc.get("10")), _f(sec.get("10")),
                _f(summary.avg_total_retrieval_ms or summary.mean_retrieval_ms),
                _f(summary.p95_total_retrieval_ms or summary.p95_retrieval_ms),
                _f(summary.mean_dense_retrieval_ms),
                _f(summary.mean_rerank_ms),
                _f(summary.quality_score), _f(summary.efficiency_score),
                grade.get("grade", ""),
                _f(grade.get("delta_hit_at_5")),
                _f(grade.get("delta_mrr_at_10")),
                _f(grade.get("delta_candidate_hit_at_50")),
                _f(grade.get("latency_ratio_p95")),
            ])

    # 3) diagnostics.json — per-cell flag dump.
    diagnostics = {
        label: {
            "diagnostics": dict(cell_summaries[label].diagnostics or {}),
            "row_count": cell_summaries[label].row_count,
            "rows_with_expected_doc_ids":
                cell_summaries[label].rows_with_expected_doc_ids,
            "error_count": cell_summaries[label].error_count,
        }
        for label in cell_summaries
    }
    (out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) sweep_report.md, candidate_k_ceiling.md, wide_mmr_titlecap_report.md
    _write_markdown_reports(
        out_dir, cell_summaries, cell_specs, grades,
    )


def _md_section_header(out_dir: Path, title: str) -> List[str]:
    return [f"# {title}", ""]


def _write_markdown_reports(
    out_dir: Path,
    cell_summaries: Dict[str, Any],
    cell_specs: Dict[str, CellSpec],
    grades: Dict[str, Dict[str, Any]],
) -> None:
    # --- sweep_report.md ---
    md: List[str] = []
    md.extend(_md_section_header(out_dir, "Wide MMR + title-cap sweep"))
    md.append(f"- baseline cell: `{_BASELINE_LABEL}`")
    md.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append("")

    md.append("## Configurations")
    md.append("")
    md.append(
        "| label | cand_k | final_k | rerank_in | mmr | λ | mmr_k "
        "| cap_rr | cap_final |"
    )
    md.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for label, spec in cell_specs.items():
        md.append(
            f"| {label} | {spec.candidate_k} | {spec.final_top_k} | "
            f"{spec.rerank_in} | {'Y' if spec.use_mmr else 'N'} | "
            f"{spec.mmr_lambda} | {spec.mmr_k} | "
            f"{spec.title_cap_rerank_input or '-'} | "
            f"{spec.title_cap_final or '-'} |"
        )
    md.append("")

    md.append("## Headline metrics")
    md.append("")
    md.append(
        "| label | hit@5 | mrr@10 | ndcg@10 | cand@50 | cand@100 "
        "| dup@10 | uniq@10 | p95ms |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, s in cell_summaries.items():
        md.append(
            f"| {label} | {_fmt(s.mean_hit_at_5)} | "
            f"{_fmt(s.mean_mrr_at_10)} | {_fmt(s.mean_ndcg_at_10)} | "
            f"{_fmt((s.candidate_hit_rates or {}).get('50'))} | "
            f"{_fmt((s.candidate_hit_rates or {}).get('100'))} | "
            f"{_fmt((s.duplicate_doc_ratios or {}).get('10'))} | "
            f"{_fmt((s.unique_doc_counts or {}).get('10'))} | "
            f"{_fmt(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} |"
        )
    md.append("")

    md.append("## Grades vs baseline")
    md.append("")
    md.append("| label | grade | Δhit@5 | Δmrr@10 | Δcand@50 | latRatio |")
    md.append("|---|---|---:|---:|---:|---:|")
    for label, grade in grades.items():
        md.append(
            f"| {label} | {grade.get('grade')} | "
            f"{_fmt(grade.get('delta_hit_at_5'))} | "
            f"{_fmt(grade.get('delta_mrr_at_10'))} | "
            f"{_fmt(grade.get('delta_candidate_hit_at_50'))} | "
            f"{_fmt(grade.get('latency_ratio_p95'))} |"
        )
    md.append("")

    md.append(
        "> Grade rules (per spec): promising_quality requires Δhit@5 ≥ "
        "0.01 OR Δmrr ≥ 0.01 with latency ratio ≤ 1.5x; regression "
        "fires when any of Δhit@5/Δmrr/Δcand@50 drops by ≥ 0.005; "
        "everything else is diagnostic_only / inconclusive."
    )
    (out_dir / "sweep_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )

    # --- candidate_k_ceiling.md ---
    cm: List[str] = []
    cm.extend(_md_section_header(out_dir, "Candidate-k ceiling check"))
    cm.append(
        "Compares the dense baseline at candidate_k = 50 / 100 / 200. "
        "Used to answer: is the candidate pool size the bottleneck for "
        "the current 0.80 candidate_hit@50 ceiling?"
    )
    cm.append("")
    cm.append(
        "| label | cand_k | cand@10 | cand@20 | cand@50 | cand@100 "
        "| hit@5 | mrr@10 | p95ms |"
    )
    cm.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, spec in cell_specs.items():
        if not label.startswith("dense_baseline"):
            continue
        s = cell_summaries[label]
        cand = s.candidate_hit_rates or {}
        cm.append(
            f"| {label} | {spec.candidate_k} | "
            f"{_fmt(cand.get('10'))} | {_fmt(cand.get('20'))} | "
            f"{_fmt(cand.get('50'))} | {_fmt(cand.get('100'))} | "
            f"{_fmt(s.mean_hit_at_5)} | {_fmt(s.mean_mrr_at_10)} | "
            f"{_fmt(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} |"
        )
    cm.append("")
    cm.append(
        "Interpretation: if cand@50 → cand@100 → cand@200 gains stay "
        "below ~0.02, the embedding/chunk/query representation is the "
        "bottleneck rather than pool size. If they keep climbing, lift "
        "the production candidate_k; otherwise the next tier of work "
        "is enriched embedding text or query rewriting."
    )
    (out_dir / "candidate_k_ceiling.md").write_text(
        "\n".join(cm) + "\n", encoding="utf-8",
    )

    # --- wide_mmr_titlecap_report.md ---
    wm: List[str] = []
    wm.extend(_md_section_header(out_dir, "Wide pool + MMR + title-cap report"))
    wm.append(
        "Focused view of the prior-strategy cells (MMR + title cap on "
        "a 200-candidate pool). The intent is to check whether Chroma-"
        "era *strategies* — not exact hyperparameter values — replicate "
        "in this FAISS harness."
    )
    wm.append("")
    wm.append(
        "| label | λ | mmr_k | cap_rr | cap_final | hit@5 | mrr@10 "
        "| dup@10 | uniq@10 | grade |"
    )
    wm.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, spec in cell_specs.items():
        if not spec.use_mmr:
            continue
        s = cell_summaries[label]
        grade = grades.get(label, {})
        wm.append(
            f"| {label} | {spec.mmr_lambda} | {spec.mmr_k} | "
            f"{spec.title_cap_rerank_input or '-'} | "
            f"{spec.title_cap_final or '-'} | "
            f"{_fmt(s.mean_hit_at_5)} | {_fmt(s.mean_mrr_at_10)} | "
            f"{_fmt((s.duplicate_doc_ratios or {}).get('10'))} | "
            f"{_fmt((s.unique_doc_counts or {}).get('10'))} | "
            f"{grade.get('grade', '')} |"
        )
    wm.append("")
    wm.append(
        "Read this against the baseline at candidate_k=50: a "
        "promising_quality cell maintains hit@5 / mrr@10 while "
        "lowering duplicate@10. A cell that only lowers duplicate "
        "without holding quality is regression (the diversity "
        "selector ate too much of the relevant pool)."
    )
    (out_dir / "wide_mmr_titlecap_report.md").write_text(
        "\n".join(wm) + "\n", encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _emit_failed_and_diffs(
    out_dir: Path,
    rows_by_cell: Dict[str, List[Any]],
) -> None:
    if _BASELINE_LABEL not in rows_by_cell:
        return
    failed_path = out_dir / "failed_queries.jsonl"
    with failed_path.open("w", encoding="utf-8") as fp:
        for label, rows in rows_by_cell.items():
            for row in rows:
                if row.error:
                    fp.write(json.dumps({
                        "cell": label,
                        "id": row.id,
                        "query": row.query,
                        "error": row.error,
                    }, ensure_ascii=False) + "\n")
    baseline_rows = {r.id: r for r in rows_by_cell[_BASELINE_LABEL]}
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    for label, rows in rows_by_cell.items():
        if label == _BASELINE_LABEL:
            continue
        for row in rows:
            base = baseline_rows.get(row.id)
            if base is None:
                continue
            if base.hit_at_5 is None or row.hit_at_5 is None:
                continue
            if base.hit_at_5 > 0.5 and row.hit_at_5 <= 0.5:
                regressions.append({
                    "cell": label, "id": row.id, "query": row.query,
                    "baseline_hit_at_5": base.hit_at_5,
                    "candidate_hit_at_5": row.hit_at_5,
                    "expected_doc_ids": list(row.expected_doc_ids),
                    "candidate_top_doc_ids": list(row.retrieved_doc_ids[:5]),
                })
            elif row.hit_at_5 > 0.5 and base.hit_at_5 <= 0.5:
                improvements.append({
                    "cell": label, "id": row.id, "query": row.query,
                    "baseline_hit_at_5": base.hit_at_5,
                    "candidate_hit_at_5": row.hit_at_5,
                    "expected_doc_ids": list(row.expected_doc_ids),
                    "candidate_top_doc_ids": list(row.retrieved_doc_ids[:5]),
                })
    with (out_dir / "top_regressions.jsonl").open("w", encoding="utf-8") as fp:
        for entry in regressions:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (out_dir / "top_improvements.jsonl").open("w", encoding="utf-8") as fp:
        for entry in improvements:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
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
        help="Optional cap on the dataset row count (for quick debug).",
    )
    parser.add_argument(
        "--cells", type=str, default="full",
        choices=["full", "core6"],
        help="Which cells to run. 'full' runs 9 cells; 'core6' runs "
             "the six required cells.",
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
    log.info("Output dir: %s", out_dir)

    retriever, reranker, info, settings = _load_dense_stack(args)

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

    specs = _spec_grid(full=(args.cells == "full"))
    log.info("Running %d cells", len(specs))

    dataset = list(load_jsonl(args.dataset))
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: int(args.limit)]
    log.info(
        "Loaded %d query rows (limit=%s) from %s",
        len(dataset), args.limit, args.dataset,
    )

    candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))

    cell_summaries: Dict[str, Any] = {}
    cell_specs: Dict[str, CellSpec] = {}
    cells_meta: List[Dict[str, Any]] = []
    rows_by_cell: Dict[str, List[Any]] = {}

    for spec in specs:
        log.info(
            "running cell %s (cand_k=%d top_k=%d rerank_in=%d "
            "mmr=%s λ=%.2f cap_rr=%s cap_final=%s)",
            spec.label, spec.candidate_k, spec.final_top_k,
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
            "  -> %s: hit@5=%.4f mrr@10=%.4f cand@50=%s cand@100=%s "
            "p95=%.1fms",
            spec.label,
            (summary.mean_hit_at_5 or 0.0),
            (summary.mean_mrr_at_10 or 0.0),
            (summary.candidate_hit_rates or {}).get("50"),
            (summary.candidate_hit_rates or {}).get("100"),
            float(
                summary.p95_total_retrieval_ms or summary.p95_retrieval_ms or 0.0
            ),
        )

    baseline_summary = cell_summaries.get(_BASELINE_LABEL)
    if baseline_summary is None:
        log.error(
            "Baseline cell %s missing — grading skipped.", _BASELINE_LABEL,
        )
        grades: Dict[str, Dict[str, Any]] = {}
    else:
        grades = {
            label: _grade_cell(
                cell_summary=summary, baseline_summary=baseline_summary,
            )
            for label, summary in cell_summaries.items()
        }
        # Self-grade for the baseline is always inconclusive.
        grades[_BASELINE_LABEL] = {
            "grade": "baseline",
            "reason": "reference",
            "delta_hit_at_5": 0.0,
            "delta_mrr_at_10": 0.0,
            "delta_candidate_hit_at_50": 0.0,
            "delta_duplicate_ratio_at_10": 0.0,
            "latency_ratio_p95": 1.0,
        }

    _persist(
        out_dir, cell_summaries, cell_specs, cells_meta, grades,
        args, info, settings,
    )
    _emit_failed_and_diffs(out_dir, rows_by_cell)

    log.info("Sweep finished — artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
