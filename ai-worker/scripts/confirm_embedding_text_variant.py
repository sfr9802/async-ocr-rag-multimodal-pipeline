"""LEGACY V3 ONLY - historical embedding-text variant confirmation.

This script keeps v3 corpus/cache defaults only for archived reproduction. Do
not use it as an active Phase 7 eval/tuning entrypoint; Phase 7 uses dataset v4
and the production embedding text builder guardrails.

Confirm sweep — embedding-text variant (raw / title / title_section).

Phase 2 follow-up to the wide-MMR confirm run. The verdict from
``confirm_wide_mmr_best_configs`` was ``INCONCLUSIVE_REPRESENTATION_
BOTTLENECK``: cand@50 ceilings at ~0.80 and the reranker uplift on
hit@5 is zero, so neither MMR-cap nor candidate-pool tuning can clear
the next 1-2 hit@5 points. The two upstream candidates are:

  1. Embedding-text representation — what string the bi-encoder
     embeds. This script tackles axis 1.
  2. Reranker input formatting — what string the cross-encoder sees.
     The verdict ``NEED_RERANKER_INPUT_AUDIT_FIRST`` (case D) flags
     this as the next-best move when the variant axis fails to lift
     final metrics despite a candidate-pool gain.

Independent variable: ``embedding_text_variant`` ∈ {raw, title,
title_section}. Held constant: the optuna_winner_top8 retrieval
recipe (candidate_k=100, rerank_in=16, MMR λ=0.65, mmr_k=48,
title_cap_rerank_input=1, title_cap_final=2). Optionally also runs
phase1_best_cap2_top8 per variant when ``--include-phase1`` is set.

Eval-only / report-only. Production code (``app/``) is **not modified**.

Outputs (under ``eval/reports/_archive/confirm-runs/retrieval-
embedding-text-variant-confirm-<TIMESTAMP>/``):

  - ``summary.csv``           — flat headline metrics per (variant, cell)
  - ``summary.json``          — full ``RetrievalEvalSummary`` per pair
  - ``comparison_report.md``  — narrative + verdict (A / B / C / D)
  - ``per_query_results.jsonl`` — one row per (variant, cell, query)
  - ``per_query_diffs.jsonl`` — improved / regressed query lists vs raw
  - ``config_dump.json``      — frozen run knobs + variant slug map
  - ``regression_guard.md``   — per-variant pass/fail vs raw
  - ``index_manifest.json``   — per-variant cache dir + manifest digest
  - ``reranker_audit.jsonl``  — gold-in-pool-but-rerank-dropped samples

Run::

    python -m scripts.confirm_embedding_text_variant \\
        --raw-cache-dir eval/agent_loop_ab/_indexes/bge-m3-anime-namu-v3-raw-mseq1024 \\
        --variants raw,title,title_section
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
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("confirm_embedding_text_variant")


_DEFAULT_DATASET = Path("eval/eval_queries/anime_silver_200.jsonl")
_DEFAULT_CORPUS = Path(
    "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl"
)
_DEFAULT_REPORTS_ROOT = Path("eval/reports/_archive/confirm-runs")
_DEFAULT_QUERY_TYPE_DRAFT = Path(
    "eval/eval_queries/anime_silver_200.query_type_draft.jsonl"
)
_DEFAULT_RAW_CACHE_DIR = Path(
    "eval/agent_loop_ab/_indexes/bge-m3-anime-namu-v3-raw-mseq1024"
)
_DEFAULT_CACHE_ROOT = Path("eval/agent_loop_ab/_indexes")

_OPTUNA_WINNER_LABEL = "optuna_winner_top8"
_PHASE1_BEST_LABEL = "phase1_best_cap2_top8"
_BASELINE_LABEL = "baseline_k50_top5"
_ANCHOR_VARIANT = "raw"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return (
        _DEFAULT_REPORTS_ROOT
        / f"retrieval-embedding-text-variant-confirm-{_now_stamp()}"
    )


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


# ---------------------------------------------------------------------------
# Stack builder per variant — load from cache when possible, otherwise build
# ---------------------------------------------------------------------------


@dataclass
class VariantStack:
    """One variant's runtime stack."""

    variant: str
    cache_dir: Path
    retriever: Any
    reranker: Any
    info: Any
    manifest: Any  # Optional[VariantManifest]
    used_cache: bool


def _resolve_variant_cache_dir(
    *,
    variant: str,
    args: argparse.Namespace,
    embedding_model: str,
) -> Path:
    """Pick the cache directory for ``variant``.

    Priority:
      1. Explicit ``--{variant}-cache-dir`` flag (the helper user passes
         when they want the raw anchor pointed at the existing
         pre-built cache).
      2. ``--cache-root / variant-canonical-name``.
    """
    from eval.harness.embedding_text_reindex import default_cache_dir_for_variant

    explicit = getattr(args, f"{variant}_cache_dir_arg", None)
    if explicit:
        return Path(explicit)
    return default_cache_dir_for_variant(
        cache_root=Path(args.cache_root),
        embedding_model=embedding_model,
        max_seq_length=int(args.max_seq_length),
        corpus_path=Path(args.corpus),
        variant=variant,
    )


def _build_or_load_variant_stack(
    *,
    variant: str,
    args: argparse.Namespace,
) -> VariantStack:
    """Build or load the dense stack for ``variant``.

    Reuses an existing cache dir when present (faiss.index +
    build.json + chunks.jsonl all exist). Otherwise re-encodes the
    corpus through ``build_variant_dense_stack`` and persists a
    fresh cache + variant manifest.
    """
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.core.config import get_settings

    from eval.harness.embedding_text_reindex import (
        build_variant_dense_stack,
        load_variant_dense_stack,
    )

    settings = get_settings()
    cache_dir = _resolve_variant_cache_dir(
        variant=variant,
        args=args,
        embedding_model=settings.rag_embedding_model,
    )
    log.info("variant=%s cache_dir=%s", variant, cache_dir)

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

    can_load = (
        not args.force_rebuild
        and (cache_dir / "faiss.index").exists()
        and (cache_dir / "build.json").exists()
        and (cache_dir / "chunks.jsonl").exists()
    )

    if can_load:
        log.info(
            "variant=%s cache hit at %s — skipping reindex.",
            variant, cache_dir,
        )
        retriever, info, manifest = load_variant_dense_stack(
            cache_dir,
            embedder=embedder,
            top_k=10,
            reranker=reranker,
            candidate_k=50,
        )
        used_cache = True
    else:
        log.info(
            "variant=%s cache miss — reindexing corpus %s",
            variant, args.corpus,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        retriever, info, manifest = build_variant_dense_stack(
            Path(args.corpus),
            embedder=embedder,
            index_dir=cache_dir,
            top_k=10,
            embedding_text_variant=variant,
            reranker=reranker,
            candidate_k=50,
            write_manifest=True,
        )
        used_cache = False

    return VariantStack(
        variant=variant,
        cache_dir=cache_dir,
        retriever=retriever,
        reranker=reranker,
        info=info,
        manifest=manifest,
        used_cache=used_cache,
    )


# ---------------------------------------------------------------------------
# Cell selection (uses ``confirm_wide_mmr_helpers`` cell roster)
# ---------------------------------------------------------------------------


def _resolve_cell_specs(
    *, include_phase1: bool, include_baseline: bool,
):
    """Return the cell specs used per variant.

    Always includes ``optuna_winner_top8`` — the spec's primary
    comparison axis. ``--include-phase1`` adds the Phase 1 cap=2 cell
    so we can confirm the variant signal is robust across recipes.
    ``--include-baseline`` adds ``baseline_k50_top5`` for an absolute
    anchor; turned off by default since the variant comparison is a
    same-cell-different-variant table, not vs production baseline.
    """
    from eval.harness.confirm_wide_mmr_helpers import default_confirm_cells

    all_specs = list(default_confirm_cells())
    by_label = {s.label: s for s in all_specs}
    out = [by_label[_OPTUNA_WINNER_LABEL]]
    if include_phase1 and _PHASE1_BEST_LABEL in by_label:
        out.append(by_label[_PHASE1_BEST_LABEL])
    if include_baseline and _BASELINE_LABEL in by_label:
        # Baseline goes first in the report so it anchors the table.
        out = [by_label[_BASELINE_LABEL]] + out
    return out


# ---------------------------------------------------------------------------
# Per-variant evaluation loop
# ---------------------------------------------------------------------------


@dataclass
class VariantCellRun:
    """One (variant, cell) result tuple."""

    variant: str
    cell_label: str
    cell_group: str
    cell_spec: Any
    summary: Any
    rows: List[Any]
    started_at: str
    finished_at: str


def _run_variant_cell(
    *,
    variant: str,
    cell_spec: Any,
    stack: VariantStack,
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    title_provider: Any,
) -> VariantCellRun:
    """Run one (variant, cell) through the wide-MMR adapter."""
    from eval.harness.retrieval_eval import (
        DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS, run_retrieval_eval,
    )
    from eval.harness.wide_retrieval_adapter import (
        WideRetrievalConfig, WideRetrievalEvalAdapter,
    )

    candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))
    log.info(
        "[variant=%s] cell=%s cand_k=%d top_k=%d rerank_in=%d "
        "mmr=%s λ=%.2f cap_rr=%s cap_final=%s",
        variant, cell_spec.label, cell_spec.candidate_k,
        cell_spec.final_top_k, cell_spec.rerank_in,
        cell_spec.use_mmr, cell_spec.mmr_lambda,
        cell_spec.title_cap_rerank_input, cell_spec.title_cap_final,
    )
    adapter = WideRetrievalEvalAdapter(
        stack.retriever,
        config=WideRetrievalConfig(
            candidate_k=cell_spec.candidate_k,
            final_top_k=cell_spec.final_top_k,
            rerank_in=cell_spec.rerank_in,
            use_mmr=cell_spec.use_mmr,
            mmr_lambda=cell_spec.mmr_lambda,
            mmr_k=cell_spec.mmr_k,
            title_cap_rerank_input=cell_spec.title_cap_rerank_input,
            title_cap_final=cell_spec.title_cap_final,
        ),
        final_reranker=stack.reranker,
        title_provider=title_provider,
        name=f"{variant}/{cell_spec.label}",
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    summary, rows, _, _ = run_retrieval_eval(
        list(dataset),
        retriever=adapter,
        top_k=cell_spec.final_top_k,
        mrr_k=10,
        ndcg_k=10,
        candidate_ks=candidate_ks,
        diversity_ks=DEFAULT_DIVERSITY_KS,
        dataset_path=str(args.dataset),
        corpus_path=str(args.corpus),
    )
    finished_at = datetime.now().isoformat(timespec="seconds")
    log.info(
        "  [variant=%s/%s] hit@5=%.4f mrr@10=%.4f cand@50=%s p95=%.1fms",
        variant, cell_spec.label,
        (summary.mean_hit_at_5 or 0.0),
        (summary.mean_mrr_at_10 or 0.0),
        (summary.candidate_hit_rates or {}).get("50"),
        float(
            summary.p95_total_retrieval_ms
            or summary.p95_retrieval_ms
            or 0.0
        ),
    )
    return VariantCellRun(
        variant=variant,
        cell_label=cell_spec.label,
        cell_group=cell_spec.group,
        cell_spec=cell_spec,
        summary=summary,
        rows=rows,
        started_at=started_at,
        finished_at=finished_at,
    )


# ---------------------------------------------------------------------------
# Reranker audit — chunk-text + title lookup builders
# ---------------------------------------------------------------------------


def _build_chunk_text_lookup(
    corpus_path: Path,
) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Return ``(doc_id -> first chunk text, doc_id -> title)``.

    The reranker audit needs *some* text to preview per doc_id; the
    simplest signal is the first chunk emitted by the same chunker the
    dense stack uses. For the title lookup we pull the document title
    so the audit can answer the spec's question "title/section 포함
    여부". Both maps are built once per run.
    """
    from app.capabilities.rag.ingest import (
        _chunks_from_section, _iter_documents,
    )

    text_by_doc: Dict[str, str] = {}
    title_by_doc: Dict[str, Optional[str]] = {}
    for raw in _iter_documents(Path(corpus_path)):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id or doc_id in text_by_doc:
            continue
        title_raw = raw.get("title")
        title_by_doc[doc_id] = (
            str(title_raw).strip() if title_raw else None
        )
        sections = raw.get("sections") or {}
        if not isinstance(sections, dict):
            continue
        for section_raw in sections.values():
            if not isinstance(section_raw, dict):
                continue
            for text in _chunks_from_section(section_raw):
                stripped = text.strip()
                if stripped:
                    text_by_doc[doc_id] = stripped
                    break
            if doc_id in text_by_doc:
                break
    return text_by_doc, title_by_doc


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_summary_csv(
    out_dir: Path,
    runs: List[VariantCellRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
) -> None:
    headers = [
        "variant", "cell", "group",
        "candidate_k", "final_top_k", "rerank_in",
        "use_mmr", "mmr_lambda", "mmr_k",
        "title_cap_rerank_input", "title_cap_final",
        "row_count",
        "hit@1", "hit@3", "hit@5",
        "mrr@10", "ndcg@10",
        "candidateHit@10", "candidateHit@20",
        "candidateHit@50", "candidateHit@100",
        "duplicateDocRatio@5", "duplicateDocRatio@10",
        "uniqueDocCount@10",
        "avgTotalRetrievalMs",
        "p50ms", "p95ms", "p99ms",
        "avgDenseRetrievalMs", "avgRerankMs",
        "rerankUpliftHit@5", "rerankUpliftMrr@10",
        "qualityScore", "efficiencyScore",
        "grade",
        "Δhit@5_vs_raw", "Δmrr@10_vs_raw",
        "Δcand@50_vs_raw", "latencyRatioP95_vs_raw",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for run in runs:
            spec = run.cell_spec
            s = run.summary
            cand = s.candidate_hit_rates or {}
            dup = s.duplicate_doc_ratios or {}
            udc = s.unique_doc_counts or {}
            d = deltas_by_pair.get((run.variant, run.cell_label))
            writer.writerow([
                run.variant, run.cell_label, run.cell_group,
                spec.candidate_k, spec.final_top_k, spec.rerank_in,
                spec.use_mmr, spec.mmr_lambda, spec.mmr_k,
                spec.title_cap_rerank_input, spec.title_cap_final,
                s.row_count,
                _f(s.mean_hit_at_1), _f(s.mean_hit_at_3), _f(s.mean_hit_at_5),
                _f(s.mean_mrr_at_10), _f(s.mean_ndcg_at_10),
                _f(cand.get("10")), _f(cand.get("20")),
                _f(cand.get("50")), _f(cand.get("100")),
                _f(dup.get("5")), _f(dup.get("10")),
                _f(udc.get("10")),
                _f(s.avg_total_retrieval_ms or s.mean_retrieval_ms),
                _f(s.p50_retrieval_ms),
                _f(s.p95_total_retrieval_ms or s.p95_retrieval_ms),
                _f(s.p99_retrieval_ms),
                _f(s.mean_dense_retrieval_ms),
                _f(s.mean_rerank_ms),
                _f(s.rerank_uplift_hit_at_5),
                _f(s.rerank_uplift_mrr_at_10),
                _f(s.quality_score),
                _f(s.efficiency_score),
                "" if d is None else d.grade,
                _f(None if d is None else d.delta_hit_at_5),
                _f(None if d is None else d.delta_mrr_at_10),
                _f(None if d is None else d.delta_candidate_hit_at_50),
                _f(None if d is None else d.latency_ratio_p95),
            ])


def _write_summary_json(
    out_dir: Path,
    *,
    runs: List[VariantCellRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
    stacks: Dict[str, VariantStack],
    args: argparse.Namespace,
    settings: Any,
    verdict: str,
    rationale: str,
    cell_verdicts: Dict[str, Tuple[str, str]],
) -> None:
    payload = {
        "schema": "phase2-embedding-text-variant-confirm.v1",
        "run": {
            "dataset": str(args.dataset),
            "corpus_path": str(args.corpus),
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": str(args.reranker_model),
            "anchor_variant": _ANCHOR_VARIANT,
            "variants": list(stacks.keys()),
            "started_at": runs[0].started_at if runs else None,
            "finished_at": runs[-1].finished_at if runs else None,
        },
        "verdict": {
            "label": verdict,
            "rationale": rationale,
            "per_cell": {
                cell: {"label": v[0], "rationale": v[1]}
                for cell, v in cell_verdicts.items()
            },
        },
        "stacks": {
            v: {
                "cache_dir": str(s.cache_dir),
                "used_cache": s.used_cache,
                "embedding_model": getattr(s.info, "embedding_model", None),
                "index_version": getattr(s.info, "index_version", None),
                "chunk_count": getattr(s.info, "chunk_count", None),
                "document_count": getattr(s.info, "document_count", None),
                "manifest": (
                    None if s.manifest is None
                    else {
                        "variant": s.manifest.variant,
                        "variant_slug": s.manifest.variant_slug,
                        "embed_text_sha256": s.manifest.embed_text_sha256,
                        "embed_text_samples": list(
                            s.manifest.embed_text_samples or []
                        )[:3],
                    }
                ),
            }
            for v, s in stacks.items()
        },
        "runs": [
            {
                "variant": r.variant,
                "cell": r.cell_label,
                "group": r.cell_group,
                "spec": asdict(r.cell_spec),
                "summary": asdict(r.summary),
                "deltas": (
                    None
                    if (r.variant, r.cell_label) not in deltas_by_pair
                    else asdict(deltas_by_pair[(r.variant, r.cell_label)])
                ),
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
            for r in runs
        ],
        "caveats": [
            "Production code (app/) is not modified. Each variant's "
            "FAISS index is built / cached under a variant-specific "
            "cache directory; the existing "
            "``bge-m3-anime-namu-v3-raw-mseq1024`` cache is the raw "
            "anchor and is reused as-is.",
            "Query embeddings are NOT prefixed — only passage "
            "embeddings carry the variant. This is the right symmetry "
            "for an A/B that asks whether the *passage* representation "
            "is the bottleneck.",
            "p95 / p99 latency on 200 rows is sensitive to GPU thermal "
            "state; treat single-digit-percent latency deltas as noise.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_per_query_jsonl(
    out_dir: Path,
    runs: List[VariantCellRun],
) -> None:
    from eval.harness.retrieval_eval import row_to_dict

    with (out_dir / "per_query_results.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for run in runs:
            for row in run.rows:
                payload = row_to_dict(row)
                payload["variant"] = run.variant
                payload["cell"] = run.cell_label
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_per_query_diffs(
    out_dir: Path,
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]],
) -> None:
    with (out_dir / "per_query_diffs.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for (variant, cell), (improved, regressed) in diffs_by_pair.items():
            for entry in improved:
                fp.write(json.dumps({
                    "variant": variant,
                    "cell": cell,
                    "direction": "improved",
                    **asdict(entry),
                }, ensure_ascii=False) + "\n")
            for entry in regressed:
                fp.write(json.dumps({
                    "variant": variant,
                    "cell": cell,
                    "direction": "regressed",
                    **asdict(entry),
                }, ensure_ascii=False) + "\n")


def _write_config_dump(
    out_dir: Path,
    *,
    runs: List[VariantCellRun],
    stacks: Dict[str, VariantStack],
    args: argparse.Namespace,
) -> None:
    from eval.harness.embedding_text_reindex import variant_slug_for_path

    payload = {
        "schema": "phase2-embedding-text-variant-confirm.config.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "dataset": str(args.dataset),
            "corpus": str(args.corpus),
            "out_dir": str(args.out_dir) if args.out_dir else None,
            "limit": args.limit,
            "variants": args.variants,
            "include_phase1": bool(args.include_phase1),
            "include_baseline": bool(args.include_baseline),
            "cache_root": str(args.cache_root),
            "raw_cache_dir": str(args.raw_cache_dir_arg)
                if args.raw_cache_dir_arg else None,
            "title_cache_dir": str(args.title_cache_dir_arg)
                if args.title_cache_dir_arg else None,
            "title_section_cache_dir": str(args.title_section_cache_dir_arg)
                if args.title_section_cache_dir_arg else None,
            "max_seq_length": args.max_seq_length,
            "embed_batch_size": args.embed_batch_size,
            "reranker_model": args.reranker_model,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
            "reranker_text_max_chars": args.reranker_text_max_chars,
            "force_rebuild": bool(args.force_rebuild),
        },
        "variant_slug_map": {
            v: variant_slug_for_path(v) for v in stacks.keys()
        },
        "cells": [asdict(r.cell_spec) for r in {
            r.cell_label: r for r in runs
        }.values()],
    }
    (out_dir / "config_dump.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_index_manifest(
    out_dir: Path,
    stacks: Dict[str, VariantStack],
) -> None:
    payload = {
        "schema": "phase2-embedding-text-variant-confirm.index-manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variants": {
            v: {
                "cache_dir": str(s.cache_dir),
                "used_cache": s.used_cache,
                "embedding_model": getattr(s.info, "embedding_model", None),
                "index_version": getattr(s.info, "index_version", None),
                "chunk_count": getattr(s.info, "chunk_count", None),
                "document_count": getattr(s.info, "document_count", None),
                "dimension": getattr(s.info, "dimension", None),
                "manifest": (
                    None if s.manifest is None
                    else {
                        "variant": s.manifest.variant,
                        "variant_slug": s.manifest.variant_slug,
                        "max_seq_length": s.manifest.max_seq_length,
                        "embed_text_sha256": s.manifest.embed_text_sha256,
                        "embed_text_samples": list(
                            s.manifest.embed_text_samples or []
                        ),
                    }
                ),
            }
            for v, s in stacks.items()
        },
    }
    (out_dir / "index_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_regression_guard(
    out_dir: Path,
    runs: List[VariantCellRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
) -> None:
    from eval.harness.confirm_wide_mmr_helpers import (
        EPS_HIT, EPS_MRR, GRADE_REGRESSION,
    )
    from eval.harness.variant_comparison import EPS_CANDIDATE

    md: List[str] = []
    md.append("# Regression guard — embedding-text variant confirm sweep")
    md.append("")
    md.append(
        f"Anchor variant: `{_ANCHOR_VARIANT}`. Each (variant, cell) is "
        "checked against the same cell on the raw anchor. A regression "
        "fires when ``Δhit@5 ≤ -0.005`` OR ``Δmrr@10 ≤ -0.005`` OR "
        "``Δcand@50 ≤ -0.005`` (matches the wide-MMR confirm grader)."
    )
    md.append("")
    md.append(
        f"Epsilon contract: `EPS_HIT={EPS_HIT}`, `EPS_MRR={EPS_MRR}`, "
        f"`EPS_CANDIDATE={EPS_CANDIDATE}`."
    )
    md.append("")
    md.append(
        "| variant | cell | grade | Δhit@5 | Δmrr@10 | Δcand@50 | latRatioP95 | passes |"
    )
    md.append(
        "|---|---|---|---:|---:|---:|---:|---|"
    )
    failures = 0
    for run in runs:
        d = deltas_by_pair.get((run.variant, run.cell_label))
        if d is None:
            continue
        passes = "✓" if d.grade != GRADE_REGRESSION else "✗"
        if d.grade == GRADE_REGRESSION:
            failures += 1
        md.append(
            f"| {run.variant} | {run.cell_label} | {d.grade} | "
            f"{_fmt_signed(d.delta_hit_at_5)} | "
            f"{_fmt_signed(d.delta_mrr_at_10)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
            f"{_fmt(d.latency_ratio_p95)} | {passes} |"
        )
    md.append("")
    if failures == 0:
        md.append(
            "**Result: PASS** — no (variant, cell) regresses against the "
            "raw anchor beyond epsilon."
        )
    else:
        md.append(
            f"**Result: {failures} regressing pair(s) flagged.** "
            "Review individual ``Δ*`` columns and the comparison report "
            "before adopting any variant."
        )
    md.append("")
    (out_dir / "regression_guard.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def _write_reranker_audit(
    out_dir: Path,
    runs_by_variant: Dict[str, List[VariantCellRun]],
    *,
    chunk_text_lookup: Dict[str, str],
    chunk_title_lookup: Dict[str, Optional[str]],
    truncation_chars: int,
    limit_per_pair: int,
) -> List[Any]:
    """Write reranker_audit.jsonl + return the samples list for the markdown.

    Audits each (variant, optuna_winner_top8) pair and surfaces the
    first ``limit_per_pair`` queries where the gold doc is in the
    candidate pool but the reranker dropped it from the top-5. Run
    against optuna_winner_top8 only since that's the headline cell;
    extending to phase1_best is straightforward but adds noise.
    """
    from eval.harness.variant_comparison import collect_reranker_audit_samples

    all_samples: List[Any] = []
    for variant, runs in runs_by_variant.items():
        for run in runs:
            if run.cell_label != _OPTUNA_WINNER_LABEL:
                continue
            samples = collect_reranker_audit_samples(
                cell_label=run.cell_label,
                variant=variant,
                rows=run.rows,
                chunk_text_lookup=chunk_text_lookup,
                chunk_title_lookup=chunk_title_lookup,
                truncation_threshold_chars=int(truncation_chars),
                limit=int(limit_per_pair),
            )
            all_samples.extend(samples)

    with (out_dir / "reranker_audit.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for sample in all_samples:
            fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    return all_samples


def _query_type_breakdown(
    runs: List[VariantCellRun],
    *,
    query_type_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Per-(variant, cell) per-bucket diagnostic join with the heuristic
    query_type draft. Mirrors the equivalent in the wide-MMR confirm
    sweep — diagnostic only, the draft is heuristic.
    """
    if query_type_path is None or not query_type_path.exists():
        return None
    qt_by_id: Dict[str, str] = {}
    qt_conf: Dict[str, float] = {}
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
            qt_by_id[qid] = str(obj.get("query_type") or "unknown")
            qt_conf[qid] = float(obj.get("query_type_confidence") or 0.0)
    if not qt_by_id:
        return None

    LOW_CONF = 0.5
    breakdown: Dict[str, Any] = {}
    for run in runs:
        key = f"{run.variant}/{run.cell_label}"
        buckets: Dict[str, Dict[str, Any]] = {}
        low_conf = {"count": 0, "h5_sum": 0.0, "mrr_sum": 0.0}
        for row in run.rows:
            rid = getattr(row, "id", None)
            if not rid:
                continue
            qt = qt_by_id.get(str(rid))
            if qt is None:
                continue
            conf = qt_conf.get(str(rid), 0.0)
            h5 = getattr(row, "hit_at_5", None)
            mrr = getattr(row, "mrr_at_10", None)
            if h5 is None or mrr is None:
                continue
            bucket = buckets.setdefault(qt, {
                "count": 0, "h5_sum": 0.0, "mrr_sum": 0.0,
            })
            bucket["count"] += 1
            bucket["h5_sum"] += float(h5)
            bucket["mrr_sum"] += float(mrr)
            if conf < LOW_CONF:
                low_conf["count"] += 1
                low_conf["h5_sum"] += float(h5)
                low_conf["mrr_sum"] += float(mrr)
        per_type = {}
        for qt, b in buckets.items():
            n = max(1, b["count"])
            per_type[qt] = {
                "count": b["count"],
                "mean_hit_at_5": round(b["h5_sum"] / n, 4),
                "mean_mrr_at_10": round(b["mrr_sum"] / n, 4),
            }
        per_type["__low_confidence_rows__"] = {
            "count": low_conf["count"],
            "mean_hit_at_5": (
                None if low_conf["count"] == 0
                else round(low_conf["h5_sum"] / max(1, low_conf["count"]), 4)
            ),
            "mean_mrr_at_10": (
                None if low_conf["count"] == 0
                else round(low_conf["mrr_sum"] / max(1, low_conf["count"]), 4)
            ),
        }
        breakdown[key] = per_type
    return breakdown


def _write_comparison_report(
    out_dir: Path,
    *,
    runs: List[VariantCellRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]],
    stacks: Dict[str, VariantStack],
    audit_samples: List[Any],
    miss_summary: Dict[Tuple[str, str], Dict[str, int]],
    verdict: str,
    rationale: str,
    cell_verdicts: Dict[str, Tuple[str, str]],
    qt_breakdown: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write the headline narrative report.

    Sections:
      1. Headline metrics per (variant, cell) — full table.
      2. Deltas vs raw anchor per cell.
      3. Candidate-miss + recoverable-miss deltas.
      4. Per-query improvements / regressions vs raw, with examples.
      5. Reranker audit (gold-in-pool-but-dropped) samples.
      6. byQueryType heuristic breakdown (diagnostic).
      7. Verdict + next-step recommendation.
      8. Caveats.
    """
    md: List[str] = []
    md.append("# Embedding-text variant confirm sweep — raw vs title vs title_section")
    md.append("")
    md.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- anchor variant: `{_ANCHOR_VARIANT}`")
    md.append(f"- variants in run: {', '.join(stacks.keys())}")
    md.append(f"- corpus: {args.corpus}")
    md.append(f"- dataset: {args.dataset}")
    md.append("")

    # 1. Headline metrics ------------------------------------------------
    md.append("## Headline metrics (silver_200)")
    md.append("")
    md.append(
        "| variant | cell | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
        "cand@50 | cand@100 | dup@10 | uniq@10 | p50ms | p95ms | p99ms |"
    )
    md.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for run in runs:
        s = run.summary
        cand = s.candidate_hit_rates or {}
        dup = s.duplicate_doc_ratios or {}
        udc = s.unique_doc_counts or {}
        md.append(
            f"| {run.variant} | {run.cell_label} | "
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

    # 2. Deltas vs raw ---------------------------------------------------
    cells_seen: List[str] = []
    for run in runs:
        if run.cell_label not in cells_seen:
            cells_seen.append(run.cell_label)
    for cell in cells_seen:
        md.append(f"## Deltas vs raw — cell `{cell}`")
        md.append("")
        md.append(
            "| variant | grade | Δhit@1 | Δhit@3 | Δhit@5 | Δmrr@10 | "
            "Δndcg@10 | Δcand@10 | Δcand@50 | Δcand@100 | Δdup@10 | "
            "Δuniq@10 | latRatioP95 | reason |"
        )
        md.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        )
        for run in runs:
            if run.cell_label != cell:
                continue
            d = deltas_by_pair.get((run.variant, run.cell_label))
            if d is None:
                continue
            md.append(
                f"| {run.variant} | {d.grade} | "
                f"{_fmt_signed(d.delta_hit_at_1)} | "
                f"{_fmt_signed(d.delta_hit_at_3)} | "
                f"{_fmt_signed(d.delta_hit_at_5)} | "
                f"{_fmt_signed(d.delta_mrr_at_10)} | "
                f"{_fmt_signed(d.delta_ndcg_at_10)} | "
                f"{_fmt_signed(d.delta_candidate_hit_at_10)} | "
                f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
                f"{_fmt_signed(d.delta_candidate_hit_at_100)} | "
                f"{_fmt_signed(d.delta_duplicate_ratio_at_10)} | "
                f"{_fmt_signed(d.delta_unique_doc_count_at_10)} | "
                f"{_fmt(d.latency_ratio_p95)} | {d.reason} |"
            )
        md.append("")

    # 3. Miss deltas ----------------------------------------------------
    md.append("## Candidate-pool miss deltas vs raw")
    md.append("")
    md.append(
        "*unrecoverable*: gold doc never appears in the candidate pool (pure "
        "dense miss). *recoverable*: gold is in the pool but the reranker "
        "dropped it from the final top-5 (reranker miss). The variant "
        "axis ought to chip away at unrecoverable misses; if it doesn't "
        "but recoverable misses go up, the dense pool is finding more "
        "candidates than the cross-encoder can re-rank correctly."
    )
    md.append("")
    md.append(
        "| variant | cell | unrecoverable_miss | recoverable_miss | "
        "Δunrecoverable_vs_raw | Δrecoverable_vs_raw |"
    )
    md.append("|---|---|---:|---:|---:|---:|")
    for run in runs:
        ms = miss_summary.get((run.variant, run.cell_label), {})
        md.append(
            f"| {run.variant} | {run.cell_label} | "
            f"{ms.get('unrecoverable', 'n/a')} | "
            f"{ms.get('recoverable', 'n/a')} | "
            f"{_fmt_signed(ms.get('unrecoverable_delta'))} | "
            f"{_fmt_signed(ms.get('recoverable_delta'))} |"
        )
    md.append("")

    # 4. Per-query diffs vs raw -----------------------------------------
    md.append("## Per-query diffs vs raw")
    md.append("")
    md.append(
        "Lists below enumerate query IDs whose hit@5 flipped between "
        "raw and the named variant on the same cell. ``improved``: "
        "raw=miss → variant=hit; ``regressed``: raw=hit → variant=miss."
    )
    md.append("")
    for (variant, cell), (improved, regressed) in diffs_by_pair.items():
        if variant == _ANCHOR_VARIANT:
            continue
        md.append(f"### `{variant}` on `{cell}`")
        md.append("")
        md.append(
            f"- improved vs raw: **{len(improved)}** queries"
        )
        md.append(
            f"- regressed vs raw: **{len(regressed)}** queries"
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
        if improved:
            md.append("")
            md.append("Examples — raw missed, variant hit:")
            md.append("")
            for entry in improved[:3]:
                md.append(
                    f"- `{entry.id}` query=`{entry.query[:60]}…` "
                    f"expected={entry.expected_doc_ids[:3]} "
                    f"variant_top5={entry.variant_top_doc_ids[:5]}"
                )
        if regressed:
            md.append("")
            md.append("Examples — raw hit, variant missed:")
            md.append("")
            for entry in regressed[:3]:
                md.append(
                    f"- `{entry.id}` query=`{entry.query[:60]}…` "
                    f"expected={entry.expected_doc_ids[:3]} "
                    f"variant_top5={entry.variant_top_doc_ids[:5]}"
                )
        md.append("")

    # 5. Reranker audit -------------------------------------------------
    md.append("## Reranker input audit (gold-in-pool-but-rerank-dropped)")
    md.append("")
    md.append(
        "Audit run against the optuna_winner_top8 cell only — the "
        "headline cell. Each row below is a query where the gold "
        "doc_id appears in the candidate pool but the cross-encoder "
        "dropped it from the final top-5. Passage previews are "
        "truncated to the reranker's text_max_chars (~800 by default); "
        "``has_title`` flags whether the gold passage's preview "
        "carries the doc title in its leading characters."
    )
    md.append("")
    md.append(
        "| variant | query_id | query | expected | retrieved_top5 | "
        "cand_count | gold_in_cand | has_title | truncated |"
    )
    md.append(
        "|---|---|---|---|---|---:|---|---|---|"
    )
    for sample in audit_samples[:30]:
        md.append(
            f"| {sample.variant} | `{sample.query_id}` | "
            f"`{sample.query[:50]}…` | "
            f"`{','.join(sample.expected_doc_ids[:2])}` | "
            f"`{','.join(sample.retrieved_top_doc_ids[:3])}` | "
            f"{sample.candidate_count} | "
            f"{'✓' if sample.gold_in_candidates else '✗'} | "
            f"{'✓' if sample.gold_passage_has_title else '✗'} | "
            f"{'✓' if sample.gold_passage_truncated else '✗'} |"
        )
    md.append("")
    if audit_samples:
        md.append(
            "Full samples (with passage previews) in "
            "``reranker_audit.jsonl``."
        )
    else:
        md.append(
            "(no qualifying gold-in-pool-but-rerank-dropped queries "
            "in this run)"
        )
    md.append("")

    # 6. byQueryType breakdown ------------------------------------------
    if qt_breakdown:
        md.append("## byQueryType breakdown (heuristic — diagnostic only)")
        md.append("")
        md.append(
            "Joined with the heuristic ``anime_silver_200.query_type_"
            "draft.jsonl``. Auto-tagged and **not manually reviewed** — "
            "treat the per-bucket numbers as directional. "
            "``__low_confidence_rows__`` rolls up rows with tagging "
            "confidence < 0.5."
        )
        md.append("")
        for key, per_type in qt_breakdown.items():
            md.append(f"### {key}")
            md.append("")
            md.append("| query_type | count | hit@5 | mrr@10 |")
            md.append("|---|---:|---:|---:|")
            for qt, stats in sorted(per_type.items()):
                md.append(
                    f"| {qt} | {stats.get('count')} | "
                    f"{_fmt(stats.get('mean_hit_at_5'))} | "
                    f"{_fmt(stats.get('mean_mrr_at_10'))} |"
                )
            md.append("")

    # 7. Verdict ---------------------------------------------------------
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}** — {rationale}")
    md.append("")
    if cell_verdicts:
        md.append("Per-cell verdicts (each cell is judged independently):")
        md.append("")
        for cell, (label, why) in cell_verdicts.items():
            md.append(f"- `{cell}` → **{label}** — {why}")
        md.append("")
    md.append("## Next-step recommendation")
    md.append("")
    from eval.harness.variant_comparison import (
        VERDICT_ADOPT_TITLE,
        VERDICT_ADOPT_TITLE_SECTION,
        VERDICT_KEEP_RAW,
        VERDICT_NEED_RERANKER_AUDIT,
    )

    if verdict == VERDICT_ADOPT_TITLE_SECTION:
        md.append(
            "1. **Adopt the title_section variant** as the production "
            "dense reindex target. Promote the variant cache as the "
            "active index and validate on a fresh 200-row pass; "
            "regression-guard the rerank-uplift signal stays ≥ 0."
        )
        md.append(
            "2. Run a focused reranker audit on the residual recoverable "
            "misses — even after this win, the cand@50 ceiling is the "
            "next bottleneck. Compare the rerank input passage "
            "formatting before / after the title_section variant to "
            "isolate which component closed the gap."
        )
    elif verdict == VERDICT_ADOPT_TITLE:
        md.append(
            "1. **Adopt the title-only variant.** title_section either "
            "regressed or didn't clear EPS; title is the safer pick. "
            "Promote the title cache and re-validate."
        )
        md.append(
            "2. Investigate why title_section didn't help — possible "
            "causes: section names diluting the prefix vector, or "
            "section duplication across docs hurting the dense ordering."
        )
    elif verdict == VERDICT_NEED_RERANKER_AUDIT:
        md.append(
            "1. **Reranker input audit is now the bottleneck.** The "
            "candidate-pool hit-rate moved with the variant axis but the "
            "final hit@5 didn't follow. Inspect the cross-encoder "
            "input format: does the passage carry the title? Are "
            "passages getting truncated at text_max_chars before the "
            "differentiating signal?"
        )
        md.append(
            "2. Hold off on another embedding reindex until the "
            "reranker input is fixed — more dense candidates won't "
            "help if the reranker drops them."
        )
        md.append(
            "3. Tactical experiment: run the same retrieval recipe on "
            "the title_section / title variant indexes but with the "
            "reranker text_max_chars cap raised to 1200 / 1600 to test "
            "whether truncation is the wall."
        )
    else:  # VERDICT_KEEP_RAW
        md.append(
            "1. **Keep the raw index.** Neither title nor title_section "
            "moved hit@5 / MRR@10 measurably; the embedding-text "
            "representation isn't the bottleneck on this dataset."
        )
        md.append(
            "2. The next two candidates are: (a) reranker input "
            "formatting (audit ``reranker_audit.jsonl`` for "
            "structural issues — title/section absence, truncation), "
            "and (b) chunking redesign — split sections more finely so "
            "the dense pool surfaces more candidate doc_ids per query."
        )
        md.append(
            "3. Consider expanding silver_200 → silver_500/1000 if the "
            "200-row epsilon (1 query = 0.005) is suppressing real "
            "signal. Variants may help on a longer-tail dataset where "
            "title disambiguation matters more."
        )
    md.append("")

    # 8. Caveats ---------------------------------------------------------
    md.append("## Caveats")
    md.append("")
    md.append(
        "- Production code (``app/``) is not modified. Every variant "
        "runs through ``WideRetrievalEvalAdapter`` over a variant-"
        "specific FAISS cache; no PostgreSQL involvement."
    )
    md.append(
        "- Query embeddings are NOT prefixed; only passage embeddings "
        "carry the variant. This is the right A/B for the *passage* "
        "representation question. A separate experiment would be "
        "needed for query-side prefixing."
    )
    md.append(
        "- The ``query_type_draft`` join is heuristic — diagnostic only. "
        "Manual review is open work; see "
        "``query_type_tagging_review.md`` from earlier sweeps."
    )
    md.append(
        "- ``has_title`` in the reranker audit is a coarse string-"
        "containment check; a true tokenizer-level audit would need "
        "to instrument the cross-encoder pre-processing."
    )
    md.append(
        "- p95/p99 latency on 200 rows is sensitive to GPU thermal "
        "state; treat single-digit-percent latency deltas as noise."
    )
    md.append("")

    (out_dir / "comparison_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _resolve_variants(args: argparse.Namespace) -> List[str]:
    from eval.harness.embedding_text_builder import EMBEDDING_TEXT_VARIANTS

    requested = [
        v.strip().lower() for v in str(args.variants).split(",") if v.strip()
    ]
    if not requested:
        raise ValueError(
            "--variants is required and must be non-empty."
        )
    if _ANCHOR_VARIANT not in requested:
        # Anchor must always be present so the deltas table has a base.
        requested = [_ANCHOR_VARIANT] + requested
    seen: List[str] = []
    for v in requested:
        if v not in EMBEDDING_TEXT_VARIANTS:
            raise ValueError(
                f"Unknown variant {v!r}; expected one of "
                f"{EMBEDDING_TEXT_VARIANTS}"
            )
        if v not in seen:
            seen.append(v)
    return seen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
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
        "--variants", type=str, default="raw,title,title_section",
        help="Comma-separated list of embedding-text variants. The "
             "`raw` anchor is auto-prepended if missing.",
    )
    parser.add_argument(
        "--include-phase1", action="store_true",
        help="Also run phase1_best_cap2_top8 per variant (default: only "
             "optuna_winner_top8 runs).",
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Include baseline_k50_top5 as an absolute anchor cell.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=_DEFAULT_CACHE_ROOT,
        help="Root directory for variant FAISS caches (one subdir per "
             "(variant, corpus, model, max_seq) tuple).",
    )
    parser.add_argument(
        "--raw-cache-dir", dest="raw_cache_dir_arg", type=Path,
        default=_DEFAULT_RAW_CACHE_DIR,
        help="Override the cache directory for the raw anchor variant. "
             "Defaults to the pre-built "
             "``bge-m3-anime-namu-v3-raw-mseq1024`` cache so we don't "
             "re-encode the corpus for raw.",
    )
    parser.add_argument(
        "--title-cache-dir", dest="title_cache_dir_arg",
        type=Path, default=None,
        help="Override the cache directory for the title variant.",
    )
    parser.add_argument(
        "--title-section-cache-dir", dest="title_section_cache_dir_arg",
        type=Path, default=None,
        help="Override the cache directory for the title_section variant.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore existing variant caches and rebuild from scratch.",
    )
    parser.add_argument(
        "--query-type-draft", type=Path, default=_DEFAULT_QUERY_TYPE_DRAFT,
        help="Optional heuristic query_type_draft jsonl for the "
             "byQueryType breakdown (diagnostic only).",
    )
    parser.add_argument(
        "--reranker-audit-limit", type=int, default=5,
        help="Per-(variant, optuna_winner) gold-in-pool-but-rerank-"
             "dropped sample cap.",
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

    variants = _resolve_variants(args)
    log.info("Variants in this run: %s", variants)

    from eval.harness.confirm_wide_mmr_helpers import GROUP_OPTUNA_WINNER
    from eval.harness.io_utils import load_jsonl
    from eval.harness.variant_comparison import (
        ANCHOR_VARIANT,
        TITLE_SECTION_VARIANT,
        TITLE_VARIANT,
        candidate_pool_recoverable_miss_count,
        candidate_pool_unrecoverable_miss_count,
        compute_variant_deltas,
        decide_variant_verdict,
        variant_per_query_diff,
    )
    from eval.harness.wide_retrieval_helpers import DocTitleResolver

    cell_specs = _resolve_cell_specs(
        include_phase1=bool(args.include_phase1),
        include_baseline=bool(args.include_baseline),
    )

    title_resolver = DocTitleResolver.from_corpus(args.corpus)
    title_provider = title_resolver.title_provider()

    dataset = list(load_jsonl(args.dataset))
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: int(args.limit)]
    log.info(
        "Loaded %d query rows (limit=%s) from %s",
        len(dataset), args.limit, args.dataset,
    )

    # Build / load every variant's stack first, then evaluate cells per
    # variant. Building all stacks up front amortises the embedder /
    # reranker model loads — `SentenceTransformerEmbedder` is created
    # fresh per variant so the FAISS index it caches inside doesn't get
    # mixed up between variants. Reranker is variant-agnostic but kept
    # local for symmetry.
    stacks: Dict[str, VariantStack] = {}
    for variant in variants:
        stacks[variant] = _build_or_load_variant_stack(variant=variant, args=args)
        log.info(
            "[variant=%s] stack ready (cache=%s, used_cache=%s, "
            "chunks=%s, dim=%s)",
            variant, stacks[variant].cache_dir, stacks[variant].used_cache,
            getattr(stacks[variant].info, "chunk_count", None),
            getattr(stacks[variant].info, "dimension", None),
        )

    # Per-(variant, cell) eval.
    runs: List[VariantCellRun] = []
    for variant, stack in stacks.items():
        for cell_spec in cell_specs:
            run = _run_variant_cell(
                variant=variant,
                cell_spec=cell_spec,
                stack=stack,
                dataset=dataset,
                args=args,
                title_provider=title_provider,
            )
            runs.append(run)

    # Index runs by (variant, cell) for fast lookup downstream.
    runs_by_pair: Dict[Tuple[str, str], VariantCellRun] = {
        (r.variant, r.cell_label): r for r in runs
    }
    runs_by_variant: Dict[str, List[VariantCellRun]] = {}
    for run in runs:
        runs_by_variant.setdefault(run.variant, []).append(run)

    # Compute deltas vs raw.
    deltas_by_pair: Dict[Tuple[str, str], Any] = {}
    for run in runs:
        raw_run = runs_by_pair.get((ANCHOR_VARIANT, run.cell_label))
        if raw_run is None:
            continue
        deltas_by_pair[(run.variant, run.cell_label)] = compute_variant_deltas(
            cell_label=run.cell_label,
            variant=run.variant,
            variant_summary=run.summary,
            raw_summary=raw_run.summary,
        )

    # Compute per-query diffs vs raw.
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]] = {}
    for run in runs:
        if run.variant == ANCHOR_VARIANT:
            continue
        raw_run = runs_by_pair.get((ANCHOR_VARIANT, run.cell_label))
        if raw_run is None:
            continue
        from eval.harness.retrieval_eval import row_to_dict
        improved, regressed = variant_per_query_diff(
            cell_label=run.cell_label,
            variant=run.variant,
            raw_rows=[row_to_dict(r) for r in raw_run.rows],
            variant_rows=[row_to_dict(r) for r in run.rows],
        )
        diffs_by_pair[(run.variant, run.cell_label)] = (improved, regressed)

    # Miss summary.
    miss_summary: Dict[Tuple[str, str], Dict[str, int]] = {}
    from eval.harness.retrieval_eval import row_to_dict
    for run in runs:
        rows_dict = [row_to_dict(r) for r in run.rows]
        unrecoverable = candidate_pool_unrecoverable_miss_count(rows_dict)
        recoverable = candidate_pool_recoverable_miss_count(rows_dict)
        raw_run = runs_by_pair.get((ANCHOR_VARIANT, run.cell_label))
        if raw_run is not None and run.variant != ANCHOR_VARIANT:
            raw_dict = [row_to_dict(r) for r in raw_run.rows]
            unrec_delta = unrecoverable - candidate_pool_unrecoverable_miss_count(raw_dict)
            rec_delta = recoverable - candidate_pool_recoverable_miss_count(raw_dict)
        else:
            unrec_delta = 0
            rec_delta = 0
        miss_summary[(run.variant, run.cell_label)] = {
            "unrecoverable": unrecoverable,
            "recoverable": recoverable,
            "unrecoverable_delta": unrec_delta,
            "recoverable_delta": rec_delta,
        }

    # Per-cell verdict, plus a global verdict that prioritises the
    # optuna_winner_top8 head-to-head — that's the spec's headline.
    cell_verdicts: Dict[str, Tuple[str, str]] = {}
    for cell_spec in cell_specs:
        title_d = deltas_by_pair.get((TITLE_VARIANT, cell_spec.label))
        ts_d = deltas_by_pair.get((TITLE_SECTION_VARIANT, cell_spec.label))
        cell_verdicts[cell_spec.label] = decide_variant_verdict(
            title_deltas=title_d,
            title_section_deltas=ts_d,
        )
    primary_verdict, primary_rationale = cell_verdicts.get(
        _OPTUNA_WINNER_LABEL,
        (
            "UNDETERMINED",
            f"{_OPTUNA_WINNER_LABEL} not in run; pick a different "
            "anchor cell or include it in --cells.",
        ),
    )

    # Reranker audit — uses corpus to build doc_id → preview maps.
    chunk_text_lookup, chunk_title_lookup = _build_chunk_text_lookup(args.corpus)
    audit_samples = _write_reranker_audit(
        out_dir,
        runs_by_variant=runs_by_variant,
        chunk_text_lookup=chunk_text_lookup,
        chunk_title_lookup=chunk_title_lookup,
        truncation_chars=int(args.reranker_text_max_chars),
        limit_per_pair=int(args.reranker_audit_limit),
    )

    qt_breakdown = _query_type_breakdown(
        runs, query_type_path=Path(args.query_type_draft),
    )

    from app.core.config import get_settings
    settings = get_settings()

    _write_summary_csv(out_dir, runs, deltas_by_pair)
    _write_summary_json(
        out_dir,
        runs=runs,
        deltas_by_pair=deltas_by_pair,
        stacks=stacks,
        args=args,
        settings=settings,
        verdict=primary_verdict,
        rationale=primary_rationale,
        cell_verdicts=cell_verdicts,
    )
    _write_per_query_jsonl(out_dir, runs)
    _write_per_query_diffs(out_dir, diffs_by_pair)
    _write_config_dump(out_dir, runs=runs, stacks=stacks, args=args)
    _write_index_manifest(out_dir, stacks)
    _write_regression_guard(out_dir, runs, deltas_by_pair)
    _write_comparison_report(
        out_dir,
        runs=runs,
        deltas_by_pair=deltas_by_pair,
        diffs_by_pair=diffs_by_pair,
        stacks=stacks,
        audit_samples=audit_samples,
        miss_summary=miss_summary,
        verdict=primary_verdict,
        rationale=primary_rationale,
        cell_verdicts=cell_verdicts,
        qt_breakdown=qt_breakdown,
        args=args,
    )

    log.info(
        "Variant confirm sweep finished — verdict=%s artifacts in %s",
        primary_verdict, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
