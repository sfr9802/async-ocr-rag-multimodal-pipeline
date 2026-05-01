"""Confirm sweep — reranker input passage formatting.

Phase 2 follow-up to ``confirm_embedding_text_variant``. The verdict
from that run was ``NEED_RERANKER_INPUT_AUDIT_FIRST``: the title /
title_section embedding-text variants lifted ``cand@50`` (+0.045 /
+0.045 vs raw) while final ``hit@5`` regressed (-0.060 / -0.045). The
dense pool surfaced more gold candidates but the cross-encoder couldn't
keep them in the top-5 — strong signal that the reranker input format
is the next bottleneck.

Independent variable: ``reranker_input_format`` ∈ {chunk_only,
title_plus_chunk, title_section_plus_chunk, compact_metadata_plus_chunk}.
Held constant per (variant, format): the optuna_winner_top8 retrieval
recipe (candidate_k=100, rerank_in=16, MMR λ=0.65, mmr_k=48,
title_cap_rerank_input=1, title_cap_final=2). The matrix:

    {raw, title, title_section} × {chunk_only, title_plus_chunk,
                                   title_section_plus_chunk,
                                   compact_metadata_plus_chunk}

Anchor: (raw, chunk_only) — the production-equivalent baseline.

Eval-only / report-only. Production code (``app/``) is **not modified**.
The wrapped reranker injects formatted passages just before the
cross-encoder; the underlying ``CrossEncoderReranker`` still applies its
own ``text_max_chars`` truncation and OOM-fallback contract.

Existing FAISS indexes / chunks caches are reused as-is — this run does
not regenerate any embedding.

Outputs (under ``eval/reports/_archive/confirm-runs/retrieval-
reranker-input-format-confirm-<TIMESTAMP>/``):

  - ``summary.csv``                — flat headline metrics per pair
  - ``summary.json``               — full RetrievalEvalSummary per pair
  - ``comparison_report.md``       — narrative + verdict (5-way)
  - ``per_query_results.jsonl``    — one row per (variant, format, query)
  - ``per_query_diffs.jsonl``      — improved/regressed query lists vs anchor
  - ``config_dump.json``           — frozen run knobs
  - ``regression_guard.md``        — per-pair pass/fail vs anchor
  - ``index_manifest.json``        — per-variant cache dir + manifest
  - ``reranker_input_audit.jsonl`` — per-pair formatted-input previews +
                                     gold-in-pool-but-rerank-dropped samples

Run::

    python -m scripts.confirm_reranker_input_format \\
        --variants raw,title_section \\
        --formats chunk_only,title_plus_chunk,title_section_plus_chunk,compact_metadata_plus_chunk
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("confirm_reranker_input_format")


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
_ANCHOR_VARIANT = "raw"
_ANCHOR_FORMAT = "chunk_only"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return (
        _DEFAULT_REPORTS_ROOT
        / f"retrieval-reranker-input-format-confirm-{_now_stamp()}"
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
# Stack management — load existing variant caches, never re-index here.
# ---------------------------------------------------------------------------


@dataclass
class VariantStack:
    variant: str
    cache_dir: Path
    retriever: Any
    base_reranker: Any
    info: Any
    manifest: Any


def _resolve_variant_cache_dir(
    *,
    variant: str,
    args: argparse.Namespace,
    embedding_model: str,
) -> Path:
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


def _load_variant_stack(
    *,
    variant: str,
    args: argparse.Namespace,
) -> VariantStack:
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.core.config import get_settings

    from eval.harness.embedding_text_reindex import load_variant_dense_stack

    settings = get_settings()
    cache_dir = _resolve_variant_cache_dir(
        variant=variant,
        args=args,
        embedding_model=settings.rag_embedding_model,
    )
    log.info("variant=%s cache_dir=%s", variant, cache_dir)

    if not (
        (cache_dir / "faiss.index").exists()
        and (cache_dir / "build.json").exists()
        and (cache_dir / "chunks.jsonl").exists()
    ):
        raise FileNotFoundError(
            f"Variant cache {cache_dir} is incomplete; this script "
            "does not re-index. Run confirm_embedding_text_variant.py "
            "first to build the cache."
        )

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=False,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    base_reranker = CrossEncoderReranker(
        model_name=str(args.reranker_model),
        max_length=int(args.reranker_max_length),
        batch_size=int(args.reranker_batch_size),
        text_max_chars=int(args.reranker_text_max_chars),
        device=args.reranker_device or None,
        collect_stage_timings=False,
    )
    retriever, info, manifest = load_variant_dense_stack(
        cache_dir,
        embedder=embedder,
        top_k=10,
        reranker=base_reranker,
        candidate_k=50,
    )
    return VariantStack(
        variant=variant,
        cache_dir=cache_dir,
        retriever=retriever,
        base_reranker=base_reranker,
        info=info,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Per (variant, format) eval loop
# ---------------------------------------------------------------------------


@dataclass
class PairRun:
    """One (variant, format) result tuple."""

    variant: str
    fmt: str
    cell_label: str
    cell_spec: Any
    summary: Any
    rows: List[Any]
    captured_input_previews: Dict[str, List[Any]]  # query_id -> [FormattingPreview]
    started_at: str
    finished_at: str


def _capture_query_id(query_row: Dict[str, Any]) -> str:
    return str(query_row.get("id") or "")


class _CapturingAdapterFactory:
    """Wraps WideRetrievalEvalAdapter.retrieve to snapshot the wrapper's
    last_input_previews per query.

    The eval driver (``run_retrieval_eval``) iterates over the dataset
    and calls ``retriever.retrieve(query)`` once per row. We need the
    formatted input previews keyed by the query id, but ``retrieve``
    only sees the query string. Solve: enumerate the dataset alongside
    the harness by composing a thin adapter that intercepts retrieve()
    and captures the wrapper's previews before yielding.
    """

    def __init__(self, adapter: Any, wrapper: Any) -> None:
        self._adapter = adapter
        self._wrapper = wrapper
        self.captured: Dict[str, List[Any]] = {}
        self._query_to_id: Dict[str, str] = {}

    def name_for_query(self, query: str, query_id: str) -> None:
        self._query_to_id[query] = query_id

    def retrieve(self, query: str) -> Any:
        report = self._adapter.retrieve(query)
        qid = self._query_to_id.get(query, "")
        previews = list(self._wrapper.last_input_previews)
        if qid:
            self.captured[qid] = previews
        return report

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._adapter, attr)


def _run_pair(
    *,
    variant: str,
    fmt: str,
    cell_spec: Any,
    stack: VariantStack,
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    title_provider: Any,
) -> PairRun:
    from eval.harness.retrieval_eval import (
        DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS, run_retrieval_eval,
    )
    from eval.harness.reranker_input_format import FormattingRerankerWrapper
    from eval.harness.wide_retrieval_adapter import (
        WideRetrievalConfig, WideRetrievalEvalAdapter,
    )

    candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))
    log.info(
        "[variant=%s fmt=%s] cell=%s cand_k=%d top_k=%d rerank_in=%d "
        "mmr=%s λ=%.2f cap_rr=%s cap_final=%s",
        variant, fmt, cell_spec.label, cell_spec.candidate_k,
        cell_spec.final_top_k, cell_spec.rerank_in,
        cell_spec.use_mmr, cell_spec.mmr_lambda,
        cell_spec.title_cap_rerank_input, cell_spec.title_cap_final,
    )

    wrapper = FormattingRerankerWrapper(
        stack.base_reranker,
        fmt=fmt,
        title_provider=title_provider,
        record_input_previews=True,
        preview_max_chars=int(args.preview_max_chars),
        truncation_threshold_chars=int(args.reranker_text_max_chars),
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
        final_reranker=wrapper,
        title_provider=title_provider,
        name=f"{variant}/{fmt}/{cell_spec.label}",
    )
    capturer = _CapturingAdapterFactory(adapter, wrapper)
    for row in dataset:
        capturer.name_for_query(
            str(row.get("query") or ""), _capture_query_id(row),
        )

    started_at = datetime.now().isoformat(timespec="seconds")
    summary, rows, _, _ = run_retrieval_eval(
        list(dataset),
        retriever=capturer,
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
        "  [variant=%s fmt=%s] hit@5=%.4f mrr@10=%.4f cand@50=%s p95=%.1fms",
        variant, fmt,
        (summary.mean_hit_at_5 or 0.0),
        (summary.mean_mrr_at_10 or 0.0),
        (summary.candidate_hit_rates or {}).get("50"),
        float(
            summary.p95_total_retrieval_ms
            or summary.p95_retrieval_ms
            or 0.0
        ),
    )
    return PairRun(
        variant=variant,
        fmt=fmt,
        cell_label=cell_spec.label,
        cell_spec=cell_spec,
        summary=summary,
        rows=rows,
        captured_input_previews=capturer.captured,
        started_at=started_at,
        finished_at=finished_at,
    )


# ---------------------------------------------------------------------------
# Reranker audit — corpus-side helpers (chunk text + title lookup)
# ---------------------------------------------------------------------------


def _build_chunk_text_lookup(
    corpus_path: Path,
) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Return ``(doc_id -> first chunk text, doc_id -> title)``."""
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
    runs: List[PairRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
) -> None:
    headers = [
        "variant", "format", "cell",
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
        "Δhit@5_vs_anchor", "Δmrr@10_vs_anchor",
        "Δcand@50_vs_anchor", "latencyRatioP95_vs_anchor",
    ]
    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8", newline="",
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for run in runs:
            spec = run.cell_spec
            s = run.summary
            cand = s.candidate_hit_rates or {}
            dup = s.duplicate_doc_ratios or {}
            udc = s.unique_doc_counts or {}
            d = deltas_by_pair.get((run.variant, run.fmt))
            writer.writerow([
                run.variant, run.fmt, run.cell_label,
                spec.candidate_k, spec.final_top_k, spec.rerank_in,
                spec.use_mmr, spec.mmr_lambda, spec.mmr_k,
                spec.title_cap_rerank_input, spec.title_cap_final,
                s.row_count,
                _f(s.mean_hit_at_1), _f(s.mean_hit_at_3),
                _f(s.mean_hit_at_5),
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
    runs: List[PairRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
    stacks: Dict[str, VariantStack],
    args: argparse.Namespace,
    settings: Any,
    verdict: str,
    rationale: str,
) -> None:
    payload = {
        "schema": "phase2-reranker-input-format-confirm.v1",
        "run": {
            "dataset": str(args.dataset),
            "corpus_path": str(args.corpus),
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": str(args.reranker_model),
            "anchor": {
                "variant": _ANCHOR_VARIANT,
                "format": _ANCHOR_FORMAT,
            },
            "variants": list(stacks.keys()),
            "formats": [run.fmt for run in runs],
            "started_at": runs[0].started_at if runs else None,
            "finished_at": runs[-1].finished_at if runs else None,
        },
        "verdict": {"label": verdict, "rationale": rationale},
        "stacks": {
            v: {
                "cache_dir": str(s.cache_dir),
                "embedding_model": getattr(s.info, "embedding_model", None),
                "index_version": getattr(s.info, "index_version", None),
                "chunk_count": getattr(s.info, "chunk_count", None),
                "document_count": getattr(s.info, "document_count", None),
            }
            for v, s in stacks.items()
        },
        "runs": [
            {
                "variant": r.variant,
                "format": r.fmt,
                "cell": r.cell_label,
                "spec": asdict(r.cell_spec),
                "summary": asdict(r.summary),
                "deltas": (
                    None
                    if (r.variant, r.fmt) not in deltas_by_pair
                    else asdict(deltas_by_pair[(r.variant, r.fmt)])
                ),
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
            for r in runs
        ],
        "caveats": [
            "Production code (app/) is not modified. Each (variant, "
            "format) pair runs through the existing FAISS cache for "
            "the variant and a FormattingRerankerWrapper that re-formats "
            "the passage before the cross-encoder.",
            "FAISS indexes are NOT regenerated; this script reuses the "
            "raw / title / title_section caches from the prior "
            "embedding-text variant confirm sweep.",
            "The independent variable is reranker input format only — "
            "candidate@K rates should be near-identical for two pairs "
            "sharing an index variant. A meaningful Δcand@K between "
            "format swaps on the same variant is a bug signal.",
            "p95 / p99 latency on 200 rows is sensitive to GPU thermal "
            "state; treat single-digit-percent latency deltas as noise. "
            "Longer prefixes (title_section, especially compact's "
            "title-only fallback) can grow the formatted text past "
            "text_max_chars and force more truncation — measurable in "
            "``reranker_input_audit.jsonl``.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_per_query_jsonl(
    out_dir: Path,
    runs: List[PairRun],
) -> None:
    from eval.harness.retrieval_eval import row_to_dict

    with (out_dir / "per_query_results.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for run in runs:
            for row in run.rows:
                payload = row_to_dict(row)
                payload["variant"] = run.variant
                payload["format"] = run.fmt
                payload["cell"] = run.cell_label
                fp.write(
                    json.dumps(payload, ensure_ascii=False) + "\n",
                )


def _write_per_query_diffs(
    out_dir: Path,
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]],
) -> None:
    with (out_dir / "per_query_diffs.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for (variant, fmt), (improved, regressed) in diffs_by_pair.items():
            for entry in improved:
                payload = {
                    "variant": variant,
                    "format": fmt,
                    "direction": "improved",
                    **asdict(entry),
                }
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for entry in regressed:
                payload = {
                    "variant": variant,
                    "format": fmt,
                    "direction": "regressed",
                    **asdict(entry),
                }
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_config_dump(
    out_dir: Path,
    *,
    runs: List[PairRun],
    stacks: Dict[str, VariantStack],
    args: argparse.Namespace,
) -> None:
    from eval.harness.embedding_text_reindex import variant_slug_for_path

    payload = {
        "schema": "phase2-reranker-input-format-confirm.config.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "dataset": str(args.dataset),
            "corpus": str(args.corpus),
            "out_dir": str(args.out_dir) if args.out_dir else None,
            "limit": args.limit,
            "variants": args.variants,
            "formats": args.formats,
            "cache_root": str(args.cache_root),
            "raw_cache_dir": (
                str(args.raw_cache_dir_arg) if args.raw_cache_dir_arg else None
            ),
            "title_cache_dir": (
                str(args.title_cache_dir_arg) if args.title_cache_dir_arg else None
            ),
            "title_section_cache_dir": (
                str(args.title_section_cache_dir_arg)
                if args.title_section_cache_dir_arg else None
            ),
            "max_seq_length": args.max_seq_length,
            "embed_batch_size": args.embed_batch_size,
            "reranker_model": args.reranker_model,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
            "reranker_text_max_chars": args.reranker_text_max_chars,
            "preview_max_chars": args.preview_max_chars,
            "reranker_audit_limit": args.reranker_audit_limit,
        },
        "anchor": {
            "variant": _ANCHOR_VARIANT,
            "format": _ANCHOR_FORMAT,
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
        "schema": "phase2-reranker-input-format-confirm.index-manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variants": {
            v: {
                "cache_dir": str(s.cache_dir),
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
    runs: List[PairRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
) -> None:
    from eval.harness.confirm_wide_mmr_helpers import (
        EPS_HIT, EPS_MRR, GRADE_REGRESSION,
    )
    from eval.harness.variant_comparison import EPS_CANDIDATE

    md: List[str] = []
    md.append("# Regression guard — reranker input format confirm sweep")
    md.append("")
    md.append(
        f"Anchor: variant=`{_ANCHOR_VARIANT}` format=`{_ANCHOR_FORMAT}`. "
        "Each (variant, format) pair is checked against the anchor on "
        "the optuna_winner_top8 cell. A regression fires when "
        "``Δhit@5 ≤ -0.005`` OR ``Δmrr@10 ≤ -0.005`` OR "
        "``Δcand@50 ≤ -0.005``."
    )
    md.append("")
    md.append(
        f"Epsilon contract: `EPS_HIT={EPS_HIT}`, `EPS_MRR={EPS_MRR}`, "
        f"`EPS_CANDIDATE={EPS_CANDIDATE}`."
    )
    md.append("")
    md.append(
        "| variant | format | grade | Δhit@5 | Δmrr@10 | Δcand@50 | "
        "latRatioP95 | passes |"
    )
    md.append("|---|---|---|---:|---:|---:|---:|---|")
    failures = 0
    for run in runs:
        d = deltas_by_pair.get((run.variant, run.fmt))
        if d is None:
            continue
        passes = "OK" if d.grade != GRADE_REGRESSION else "FAIL"
        if d.grade == GRADE_REGRESSION:
            failures += 1
        md.append(
            f"| {run.variant} | {run.fmt} | {d.grade} | "
            f"{_fmt_signed(d.delta_hit_at_5)} | "
            f"{_fmt_signed(d.delta_mrr_at_10)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
            f"{_fmt(d.latency_ratio_p95)} | {passes} |"
        )
    md.append("")
    if failures == 0:
        md.append(
            "**Result: PASS** — no (variant, format) pair regresses "
            "against the anchor beyond epsilon."
        )
    else:
        md.append(
            f"**Result: {failures} regressing pair(s) flagged.** "
            "Review individual ``Δ*`` columns and the comparison report "
            "before adopting any format."
        )
    md.append("")
    (out_dir / "regression_guard.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Reranker input audit — formatted previews + gold-in-pool-but-dropped
# ---------------------------------------------------------------------------


def _gold_dropped_qids(rows: List[Any]) -> List[Tuple[Any, str, List[str]]]:
    """Return rows whose gold doc is in the candidate pool but the
    reranker dropped it from the top-5. Shape: (row, gold_doc_id, top5).
    """
    from eval.harness.retrieval_eval import row_to_dict
    out: List[Tuple[Any, str, List[str]]] = []
    for row in rows:
        d = row_to_dict(row)
        expected = [str(x) for x in (d.get("expected_doc_ids") or []) if x]
        if not expected:
            continue
        retrieved = [str(x) for x in (d.get("retrieved_doc_ids") or []) if x]
        candidates = set(
            str(x) for x in (d.get("candidate_doc_ids") or []) if x
        )
        retrieved_top5 = set(retrieved[:5])
        if not any(e in candidates for e in expected):
            continue
        if any(e in retrieved_top5 for e in expected):
            continue
        gold = next((e for e in expected if e in candidates), expected[0])
        out.append((row, gold, retrieved[:5]))
    return out


def _write_reranker_input_audit(
    out_dir: Path,
    runs: List[PairRun],
    *,
    chunk_text_lookup: Dict[str, str],
    chunk_title_lookup: Dict[str, Optional[str]],
    limit_per_pair: int,
) -> List[Dict[str, Any]]:
    """Write the formatted-input audit jsonl + return summary samples.

    Per (variant, format) pair we emit up to ``limit_per_pair`` rows
    where the gold doc is in the candidate pool but the reranker
    dropped it. Each row carries the formatted preview the cross-
    encoder actually scored for the gold chunk, plus the metadata flags
    (``has_title`` / ``has_section`` / ``truncated``) so the report can
    answer "did the prefix surface in the passage?".
    """
    from eval.harness.retrieval_eval import row_to_dict

    samples: List[Dict[str, Any]] = []
    for run in runs:
        dropped = _gold_dropped_qids(run.rows)[:limit_per_pair]
        for row, gold_doc_id, top5 in dropped:
            row_dict = row_to_dict(row)
            qid = str(row_dict.get("id") or "")
            previews = run.captured_input_previews.get(qid, [])
            gold_preview = next(
                (p for p in previews if p.doc_id == gold_doc_id),
                None,
            )
            top1_preview = previews[0] if previews else None
            sample = {
                "variant": run.variant,
                "format": run.fmt,
                "cell": run.cell_label,
                "query_id": qid,
                "query": str(row_dict.get("query") or ""),
                "gold_doc_id": gold_doc_id,
                "gold_title": chunk_title_lookup.get(gold_doc_id),
                "final_top_doc_ids": top5,
                "final_top_titles": [
                    chunk_title_lookup.get(d) for d in top5
                ],
                "rerank_input_preview": (
                    None if gold_preview is None
                    else gold_preview.preview
                ),
                "rerank_input_length": (
                    None if gold_preview is None
                    else gold_preview.formatted_length
                ),
                "has_title": (
                    None if gold_preview is None
                    else gold_preview.has_title
                ),
                "has_section": (
                    None if gold_preview is None
                    else gold_preview.has_section
                ),
                "truncated": (
                    None if gold_preview is None
                    else gold_preview.truncated
                ),
                "top1_preview": (
                    None if top1_preview is None
                    else top1_preview.preview
                ),
                "top1_doc_id": (
                    None if top1_preview is None
                    else top1_preview.doc_id
                ),
                "top1_title": (
                    None if top1_preview is None
                    else top1_preview.title
                ),
                "failure_reason_guess": _failure_reason_guess(
                    gold_preview=gold_preview,
                    top1_preview=top1_preview,
                ),
            }
            samples.append(sample)

    with (out_dir / "reranker_input_audit.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return samples


def _failure_reason_guess(
    *, gold_preview: Any, top1_preview: Any,
) -> str:
    """Heuristic — surface why the reranker likely dropped the gold."""
    if gold_preview is None:
        return "gold_chunk_not_in_rerank_input_pool"
    if gold_preview.truncated:
        return "gold_passage_truncated"
    if not gold_preview.has_title and gold_preview.fmt != "chunk_only":
        return "format_did_not_surface_title"
    if (
        top1_preview is not None
        and gold_preview.title
        and top1_preview.title
        and gold_preview.title.casefold().strip()
        == top1_preview.title.casefold().strip()
    ):
        return "same_title_dup_pushed_gold_down"
    return "reranker_preferred_other_chunk"


# ---------------------------------------------------------------------------
# byQueryType breakdown — diagnostic only
# ---------------------------------------------------------------------------


def _query_type_breakdown(
    runs: List[PairRun],
    *,
    query_type_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
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
        key = f"{run.variant}/{run.fmt}"
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


# ---------------------------------------------------------------------------
# Comparison report (markdown narrative + verdict)
# ---------------------------------------------------------------------------


def _write_comparison_report(
    out_dir: Path,
    *,
    runs: List[PairRun],
    deltas_by_pair: Dict[Tuple[str, str], Any],
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]],
    stacks: Dict[str, VariantStack],
    audit_samples: List[Dict[str, Any]],
    miss_summary: Dict[Tuple[str, str], Dict[str, int]],
    verdict: str,
    rationale: str,
    qt_breakdown: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    from eval.harness.reranker_format_comparison import (
        VERDICT_ADOPT_TITLE_PREFIX,
        VERDICT_ADOPT_TITLE_SECTION_PREFIX,
        VERDICT_ADOPT_COMPACT_METADATA_PREFIX,
        VERDICT_KEEP_CHUNK_ONLY,
        VERDICT_NEED_CHUNKING_DIVERSITY,
    )

    md: List[str] = []
    md.append(
        "# Reranker input format confirm sweep — chunk_only vs prefix variants"
    )
    md.append("")
    md.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append(
        f"- anchor: variant=`{_ANCHOR_VARIANT}` format=`{_ANCHOR_FORMAT}`"
    )
    md.append(f"- variants: {', '.join(stacks.keys())}")
    md.append(
        f"- formats: {', '.join(sorted({r.fmt for r in runs}))}"
    )
    md.append(f"- corpus: {args.corpus}")
    md.append(f"- dataset: {args.dataset}")
    md.append("")

    # 1. Headline metrics --------------------------------------------------
    md.append("## Headline metrics (silver_200, optuna_winner_top8)")
    md.append("")
    md.append(
        "| variant | format | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
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
            f"| {run.variant} | {run.fmt} | "
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

    # 2. Deltas vs anchor --------------------------------------------------
    md.append(
        f"## Deltas vs anchor (variant=`{_ANCHOR_VARIANT}` "
        f"format=`{_ANCHOR_FORMAT}`)"
    )
    md.append("")
    md.append(
        "| variant | format | grade | Δhit@1 | Δhit@3 | Δhit@5 | "
        "Δmrr@10 | Δndcg@10 | Δcand@50 | Δcand@100 | Δdup@10 | "
        "Δuniq@10 | latRatioP95 | reason |"
    )
    md.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for run in runs:
        d = deltas_by_pair.get((run.variant, run.fmt))
        if d is None:
            continue
        md.append(
            f"| {run.variant} | {run.fmt} | {d.grade} | "
            f"{_fmt_signed(d.delta_hit_at_1)} | "
            f"{_fmt_signed(d.delta_hit_at_3)} | "
            f"{_fmt_signed(d.delta_hit_at_5)} | "
            f"{_fmt_signed(d.delta_mrr_at_10)} | "
            f"{_fmt_signed(d.delta_ndcg_at_10)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_100)} | "
            f"{_fmt_signed(d.delta_duplicate_ratio_at_10)} | "
            f"{_fmt_signed(d.delta_unique_doc_count_at_10)} | "
            f"{_fmt(d.latency_ratio_p95)} | {d.reason} |"
        )
    md.append("")

    # 3. cand@K invariance check ------------------------------------------
    md.append("## Candidate-pool invariance check")
    md.append("")
    md.append(
        "*Sanity check.* The reranker input format is a *post-dense* "
        "transform — it MUST NOT change cand@K within a fixed index "
        "variant. The table below shows max ``|Δcand@50|`` and "
        "max ``|Δcand@100|`` across formats for each variant; if any "
        "row shows non-zero deltas, the wrapper or adapter has a bug."
    )
    md.append("")
    md.append(
        "| variant | max |Δcand@50| across formats | max |Δcand@100| | passes |"
    )
    md.append("|---|---:|---:|---|")
    for variant in sorted({r.variant for r in runs}):
        cands50 = [
            (r.summary.candidate_hit_rates or {}).get("50")
            for r in runs if r.variant == variant
        ]
        cands100 = [
            (r.summary.candidate_hit_rates or {}).get("100")
            for r in runs if r.variant == variant
        ]
        cands50 = [c for c in cands50 if c is not None]
        cands100 = [c for c in cands100 if c is not None]
        max_d50 = (max(cands50) - min(cands50)) if cands50 else None
        max_d100 = (max(cands100) - min(cands100)) if cands100 else None
        passes = "OK" if (
            (max_d50 is not None and max_d50 < 1e-9)
            and (max_d100 is not None and max_d100 < 1e-9)
        ) else (
            "FAIL"
            if (
                (max_d50 is not None and max_d50 > 0.005)
                or (max_d100 is not None and max_d100 > 0.005)
            )
            else "tolerance"
        )
        md.append(
            f"| {variant} | {_fmt_signed(max_d50)} | "
            f"{_fmt_signed(max_d100)} | {passes} |"
        )
    md.append("")

    # 4. Miss deltas vs anchor --------------------------------------------
    md.append("## Miss counters vs anchor")
    md.append("")
    md.append(
        "*unrecoverable*: gold doc never appears in the candidate pool "
        "(pure dense miss; controlled by index variant only). "
        "*recoverable*: gold is in the pool but the reranker dropped "
        "it from the final top-5 (reranker miss). The format axis "
        "should chip away at recoverable misses; if it doesn't, the "
        "next axis is chunking diversity."
    )
    md.append("")
    md.append(
        "| variant | format | unrecoverable | recoverable | "
        "Δunrecoverable_vs_anchor | Δrecoverable_vs_anchor |"
    )
    md.append("|---|---|---:|---:|---:|---:|")
    for run in runs:
        ms = miss_summary.get((run.variant, run.fmt), {})
        md.append(
            f"| {run.variant} | {run.fmt} | "
            f"{ms.get('unrecoverable', 'n/a')} | "
            f"{ms.get('recoverable', 'n/a')} | "
            f"{_fmt_signed(ms.get('unrecoverable_delta'))} | "
            f"{_fmt_signed(ms.get('recoverable_delta'))} |"
        )
    md.append("")

    # 5. Per-query diffs ---------------------------------------------------
    md.append("## Per-query diffs vs anchor")
    md.append("")
    md.append(
        "Lists below enumerate query IDs whose hit@5 flipped between "
        f"the anchor (`{_ANCHOR_VARIANT}` × `{_ANCHOR_FORMAT}`) and "
        "the named (variant, format) pair. ``improved``: anchor=miss → "
        "pair=hit; ``regressed``: anchor=hit → pair=miss."
    )
    md.append("")
    for (variant, fmt), (improved, regressed) in diffs_by_pair.items():
        if variant == _ANCHOR_VARIANT and fmt == _ANCHOR_FORMAT:
            continue
        md.append(f"### `{variant}` × `{fmt}`")
        md.append("")
        md.append(f"- improved vs anchor: **{len(improved)}** queries")
        md.append(f"- regressed vs anchor: **{len(regressed)}** queries")
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

    # 6. Reranker input audit samples -------------------------------------
    md.append("## Reranker input audit (formatted-passage previews)")
    md.append("")
    md.append(
        "Audit run on every (variant, format) pair. Each row is a "
        "query where the gold doc appears in the candidate pool but "
        "the cross-encoder dropped it from the final top-5. "
        "``rerank_input_preview`` is what the cross-encoder actually "
        "scored for that gold chunk — useful for answering "
        "\"did the prefix surface?\" and \"was the passage truncated "
        "before the differentiating signal?\". "
        "Full samples in ``reranker_input_audit.jsonl``."
    )
    md.append("")
    md.append(
        "| variant | format | query_id | gold | top1_title | "
        "has_title | has_section | truncated | reason |"
    )
    md.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for sample in audit_samples[:40]:
        title_short = (sample.get("top1_title") or "")[:40]
        gold_title_short = (sample.get("gold_title") or "")[:30]
        md.append(
            f"| {sample['variant']} | {sample['format']} | "
            f"`{sample['query_id']}` | "
            f"`{gold_title_short}` | `{title_short}` | "
            f"{'OK' if sample.get('has_title') else '—'} | "
            f"{'OK' if sample.get('has_section') else '—'} | "
            f"{'OK' if sample.get('truncated') else '—'} | "
            f"{sample.get('failure_reason_guess', '')} |"
        )
    md.append("")

    # 7. byQueryType -------------------------------------------------------
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

    # 8. Verdict + next-step ----------------------------------------------
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}** — {rationale}")
    md.append("")
    md.append("## Next-step recommendation")
    md.append("")
    if verdict == VERDICT_ADOPT_TITLE_SECTION_PREFIX:
        md.append(
            "1. Adopt ``title_section_plus_chunk`` as the production "
            "reranker input format. Wire the formatter into "
            "``CrossEncoderReranker.rerank`` (or a thin adapter "
            "around it) and validate on a fresh 200-row pass."
        )
        md.append(
            "2. Re-run the candidate-pool / reranker uplift sweep with "
            "the new format to confirm the gain compounds with the "
            "title_section embedding-text variant."
        )
    elif verdict == VERDICT_ADOPT_TITLE_PREFIX:
        md.append(
            "1. Adopt ``title_plus_chunk`` as the production reranker "
            "input format. The shorter prefix preserves the "
            "``text_max_chars`` budget while still surfacing the title "
            "signal the cross-encoder needs."
        )
        md.append(
            "2. Hold off on title_section unless the section names "
            "vocabulary is known to be a high-signal axis on a future "
            "dataset."
        )
    elif verdict == VERDICT_ADOPT_COMPACT_METADATA_PREFIX:
        md.append(
            "1. Adopt the compact ``[title / section]\\n{chunk}`` "
            "format. Long prefixes are eating the ``text_max_chars`` "
            "budget; the compact form gives the cross-encoder the same "
            "metadata signal at lower cost."
        )
        md.append(
            "2. Track p95 latency and ``truncated`` rate after "
            "deploying — the compact form should reduce both."
        )
    elif verdict == VERDICT_NEED_CHUNKING_DIVERSITY:
        md.append(
            "1. Reranker input format is **not** the bottleneck. The "
            "dense pool brings more gold candidates (cand@50/cand@100 "
            "lift) but the cross-encoder cannot disambiguate near-"
            "duplicate chunks regardless of prefix. The next axis is "
            "chunking diversity — finer-grained section splits, "
            "section-level dedup, or a chunk-level re-id step before "
            "the dense pool."
        )
        md.append(
            "2. Tactical: rerun the wide-MMR sweep with stricter "
            "title_cap_rerank_input (cap=0 / no-op vs cap=1) to "
            "confirm the dup signal is chunking-driven, not cap-driven."
        )
        md.append(
            "3. Hold off on adopting any reranker format until the "
            "chunking experiment lands — adding a prefix on top of a "
            "dup-prone pool would mask the real bottleneck."
        )
    else:  # VERDICT_KEEP_CHUNK_ONLY
        md.append(
            "1. Keep ``chunk_only`` as the reranker input format. No "
            "prefix variant clears EPS on hit@5 / MRR over the anchor, "
            "and the dense pool didn't show a chunking-diversity "
            "signal either."
        )
        md.append(
            "2. The next two candidates are: (a) chunking redesign to "
            "split sections more finely, and (b) a different reranker "
            "model (e.g. larger context bge-reranker-v2-gemma) with "
            "more headroom on ``text_max_chars``. The current bge-"
            "reranker-v2-m3 may simply not benefit from the prefix "
            "structure on this dataset."
        )
        md.append(
            "3. Consider expanding silver_200 → silver_500/1000 if "
            "the 200-row epsilon (1 query = 0.005) is suppressing real "
            "format signal on a long-tail subset."
        )
    md.append("")

    # 9. Caveats -----------------------------------------------------------
    md.append("## Caveats")
    md.append("")
    md.append(
        "- Production code (``app/``) is not modified. Each (variant, "
        "format) pair runs through ``WideRetrievalEvalAdapter`` over "
        "the variant's existing FAISS cache, with a "
        "``FormattingRerankerWrapper`` injected between the adapter "
        "and the underlying ``CrossEncoderReranker``."
    )
    md.append(
        "- FAISS indexes are reused as-is; this run does NOT re-encode "
        "the corpus. cand@K rates are constant per index variant — a "
        "non-zero Δcand@K *across formats sharing a variant* is a bug "
        "signal flagged by the invariance check above."
    )
    md.append(
        "- ``has_title`` / ``has_section`` are coarse string-"
        "containment checks within the formatted prefix window; they "
        "say whether the cross-encoder *could* see the metadata, not "
        "whether the tokenizer's truncation boundary preserved it."
    )
    md.append(
        "- The ``query_type_draft`` join is heuristic — diagnostic "
        "only. Manual review of bucket assignments is open work."
    )
    md.append(
        "- p95/p99 latency on 200 rows is sensitive to GPU thermal "
        "state; treat single-digit-percent latency deltas as noise. "
        "Format-driven latency growth is most visible in the ``truncated`` "
        "rate of the audit — longer prefixes truncate more often."
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
        raise ValueError("--variants is required and must be non-empty.")
    if _ANCHOR_VARIANT not in requested:
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


def _resolve_formats(args: argparse.Namespace) -> List[str]:
    from eval.harness.reranker_input_format import RERANKER_INPUT_FORMATS

    requested = [
        f.strip() for f in str(args.formats).split(",") if f.strip()
    ]
    if not requested:
        raise ValueError("--formats is required and must be non-empty.")
    if _ANCHOR_FORMAT not in requested:
        requested = [_ANCHOR_FORMAT] + requested
    seen: List[str] = []
    for f in requested:
        if f not in RERANKER_INPUT_FORMATS:
            raise ValueError(
                f"Unknown format {f!r}; expected one of "
                f"{RERANKER_INPUT_FORMATS}"
            )
        if f not in seen:
            seen.append(f)
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
        "--variants", type=str, default="raw,title_section",
        help="Comma-separated index variants to evaluate. The `raw` "
             "anchor is auto-prepended if missing. Add `title` for the "
             "full 3-variant matrix.",
    )
    parser.add_argument(
        "--formats", type=str,
        default=(
            "chunk_only,title_plus_chunk,"
            "title_section_plus_chunk,compact_metadata_plus_chunk"
        ),
        help="Comma-separated reranker input formats. The "
             "`chunk_only` anchor is auto-prepended if missing.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=_DEFAULT_CACHE_ROOT,
        help="Root directory for variant FAISS caches.",
    )
    parser.add_argument(
        "--raw-cache-dir", dest="raw_cache_dir_arg", type=Path,
        default=_DEFAULT_RAW_CACHE_DIR,
    )
    parser.add_argument(
        "--title-cache-dir", dest="title_cache_dir_arg",
        type=Path, default=None,
    )
    parser.add_argument(
        "--title-section-cache-dir", dest="title_section_cache_dir_arg",
        type=Path, default=None,
    )
    parser.add_argument(
        "--query-type-draft", type=Path, default=_DEFAULT_QUERY_TYPE_DRAFT,
    )
    parser.add_argument(
        "--reranker-audit-limit", type=int, default=10,
        help="Per-(variant, format) gold-in-pool-but-rerank-dropped "
             "sample cap.",
    )
    parser.add_argument(
        "--preview-max-chars", type=int, default=600,
        help="Truncation cap for ``rerank_input_preview`` strings in "
             "the audit jsonl. Has no effect on what the cross-encoder "
             "actually sees — the wrapped CrossEncoderReranker still "
             "applies its own ``text_max_chars`` cap.",
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
    formats = _resolve_formats(args)
    log.info("Variants: %s | Formats: %s", variants, formats)

    from eval.harness.confirm_wide_mmr_helpers import default_confirm_cells
    from eval.harness.io_utils import load_jsonl
    from eval.harness.reranker_format_comparison import (
        decide_reranker_format_verdict,
    )
    from eval.harness.variant_comparison import (
        candidate_pool_recoverable_miss_count,
        candidate_pool_unrecoverable_miss_count,
        compute_variant_deltas,
        variant_per_query_diff,
    )
    from eval.harness.wide_retrieval_helpers import DocTitleResolver

    cell_specs = list(default_confirm_cells())
    by_label = {s.label: s for s in cell_specs}
    if _OPTUNA_WINNER_LABEL not in by_label:
        log.error(
            "Cell %s missing from default_confirm_cells; bail.",
            _OPTUNA_WINNER_LABEL,
        )
        return 3
    cell_spec = by_label[_OPTUNA_WINNER_LABEL]
    log.info("Cell: %s — %s", cell_spec.label, cell_spec.description)

    title_resolver = DocTitleResolver.from_corpus(args.corpus)
    title_provider = title_resolver.title_provider()

    dataset = list(load_jsonl(args.dataset))
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: int(args.limit)]
    log.info(
        "Loaded %d query rows (limit=%s) from %s",
        len(dataset), args.limit, args.dataset,
    )

    # Load every variant stack first.
    stacks: Dict[str, VariantStack] = {}
    for variant in variants:
        stacks[variant] = _load_variant_stack(variant=variant, args=args)
        log.info(
            "[variant=%s] cache=%s chunks=%s dim=%s",
            variant, stacks[variant].cache_dir,
            getattr(stacks[variant].info, "chunk_count", None),
            getattr(stacks[variant].info, "dimension", None),
        )

    # Run every (variant, format) pair.
    runs: List[PairRun] = []
    for variant in variants:
        stack = stacks[variant]
        for fmt in formats:
            run = _run_pair(
                variant=variant,
                fmt=fmt,
                cell_spec=cell_spec,
                stack=stack,
                dataset=dataset,
                args=args,
                title_provider=title_provider,
            )
            runs.append(run)

    # Index runs by (variant, format) for fast lookup.
    runs_by_pair: Dict[Tuple[str, str], PairRun] = {
        (r.variant, r.fmt): r for r in runs
    }

    anchor_run = runs_by_pair.get((_ANCHOR_VARIANT, _ANCHOR_FORMAT))
    if anchor_run is None:
        log.error(
            "Anchor pair (%s, %s) missing — cannot compute deltas.",
            _ANCHOR_VARIANT, _ANCHOR_FORMAT,
        )
        return 4

    # Compute deltas vs anchor. compute_variant_deltas expects a
    # variant key, but we want a (variant, format) composite — pass
    # the format as the variant key so the resulting VariantDeltas
    # reads ``variant=fmt`` in the table. Since the report writer
    # surfaces (variant, format) explicitly, this is just plumbing.
    deltas_by_pair: Dict[Tuple[str, str], Any] = {}
    for run in runs:
        # Key as (run.variant, run.fmt); pass the composite as the
        # variant string so the dataclass knows it's the anchor.
        composite = f"{run.variant}/{run.fmt}"
        anchor_composite = f"{_ANCHOR_VARIANT}/{_ANCHOR_FORMAT}"
        deltas_by_pair[(run.variant, run.fmt)] = compute_variant_deltas(
            cell_label=run.cell_label,
            variant=composite if composite != anchor_composite else "raw",
            variant_summary=run.summary,
            raw_summary=anchor_run.summary,
        )

    # Per-query diffs vs anchor.
    diffs_by_pair: Dict[Tuple[str, str], Tuple[List[Any], List[Any]]] = {}
    from eval.harness.retrieval_eval import row_to_dict
    anchor_rows_dict = [row_to_dict(r) for r in anchor_run.rows]
    for run in runs:
        if run.variant == _ANCHOR_VARIANT and run.fmt == _ANCHOR_FORMAT:
            continue
        improved, regressed = variant_per_query_diff(
            cell_label=run.cell_label,
            variant=f"{run.variant}/{run.fmt}",
            raw_rows=anchor_rows_dict,
            variant_rows=[row_to_dict(r) for r in run.rows],
        )
        diffs_by_pair[(run.variant, run.fmt)] = (improved, regressed)

    # Miss summary.
    miss_summary: Dict[Tuple[str, str], Dict[str, int]] = {}
    anchor_unrec = candidate_pool_unrecoverable_miss_count(anchor_rows_dict)
    anchor_rec = candidate_pool_recoverable_miss_count(anchor_rows_dict)
    for run in runs:
        rows_dict = [row_to_dict(r) for r in run.rows]
        unrec = candidate_pool_unrecoverable_miss_count(rows_dict)
        rec = candidate_pool_recoverable_miss_count(rows_dict)
        miss_summary[(run.variant, run.fmt)] = {
            "unrecoverable": unrec,
            "recoverable": rec,
            "unrecoverable_delta": unrec - anchor_unrec,
            "recoverable_delta": rec - anchor_rec,
        }

    # Reranker input audit.
    chunk_text_lookup, chunk_title_lookup = _build_chunk_text_lookup(
        args.corpus,
    )
    audit_samples = _write_reranker_input_audit(
        out_dir,
        runs,
        chunk_text_lookup=chunk_text_lookup,
        chunk_title_lookup=chunk_title_lookup,
        limit_per_pair=int(args.reranker_audit_limit),
    )

    qt_breakdown = _query_type_breakdown(
        runs, query_type_path=Path(args.query_type_draft),
    )

    # Verdict.
    verdict, rationale = decide_reranker_format_verdict(
        deltas_by_pair=deltas_by_pair,
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
        verdict=verdict,
        rationale=rationale,
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
        verdict=verdict,
        rationale=rationale,
        qt_breakdown=qt_breakdown,
        args=args,
    )

    log.info(
        "Reranker input format confirm sweep finished — verdict=%s "
        "artifacts in %s",
        verdict, out_dir,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
