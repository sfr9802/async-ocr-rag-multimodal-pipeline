"""Phase 2 minimal full-silver sweep: dense / BM25 / hybrid retrievers.

Eval-only driver. Encodes the full ``anime_namu_v3_token_chunked`` corpus
*once* via ``build_offline_rag_stack``, then reuses the embedder + FAISS
index across six retrieval cells, scoring each against the
``anime_silver_200`` query set:

  1. dense_only_current_top5   — bi-encoder + cross-encoder rerank, top 5
  2. dense_only_current_top10  — same stack, top 10
  3. bm25_raw_top5             — BM25Okapi over chunk text (raw variant)
  4. bm25_title_section_top5   — BM25Okapi with title+section prefix
  5. hybrid_rrf_top5           — RRF fusion of dense (50) + bm25 (50)
  6. hybrid_rrf_top10          — same fusion, final 10

The script does NOT change production code. It only:

  - Adds a tiny ``DenseEvalRetrieverAdapter`` that runs the bi-encoder
    pool once (NoOpReranker, top 50) so the harness picks up
    ``candidate_doc_ids`` for the dense cells, then runs the real
    reranker pass for the final results. Both passes share the
    underlying FAISS index + embedder + metadata, so the only extra
    cost is one bi-encoder forward pass per query.
  - Adds a ``DensePoolRetrieverAdapter`` (NoOp, top 50) for use as the
    dense backend of the RRF hybrid retriever.
  - Reuses the existing ``run_retrieval_sweep`` driver to score every
    cell and persist the consolidated sweep report + Pareto adapter.

Outputs land under
``eval/reports/retrieval-full-silver-minimal-<TIMESTAMP>/``. The run never
overwrites the legacy historical baseline at
``eval/reports/legacy-baseline-final/`` or the phase2a-latency artifacts.

Run::

    python -m scripts.eval_full_silver_minimal_sweep \
        --dataset eval/eval_queries/anime_silver_200.jsonl \
        --corpus eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("eval_full_silver_minimal_sweep")


_DEFAULT_DATASET = Path("eval/eval_queries/anime_silver_200.jsonl")
_DEFAULT_CORPUS = Path(
    "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl"
)
_DEFAULT_REPORTS_ROOT = Path("eval/reports")
_DEFAULT_CACHE_ROOT = Path("eval/_cache/dense_index")
_PER_BACKEND_TOP_K = 50
_K_RRF = 60
_CHUNKS_FILE = "chunks.jsonl"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return _DEFAULT_REPORTS_ROOT / f"retrieval-full-silver-minimal-{_now_stamp()}"


# ---------------------------------------------------------------------------
# Corpus chunking helper — mirrors ``build_offline_rag_stack`` so BM25 sees
# the same chunk set as dense.
# ---------------------------------------------------------------------------


def _iter_dense_chunks(corpus_path: Path) -> Iterable[Any]:
    """Yield chunk-like objects matching ``build_offline_rag_stack``.

    Each yielded object exposes ``chunk_id``, ``doc_id``, ``section``,
    ``text``, ``title``, ``keywords`` so ``build_bm25_index`` can compose
    the same prefix variants the dense embedder would have seen.
    """
    from app.capabilities.rag.ingest import (
        _chunks_from_section,
        _iter_documents,
        _stable_chunk_id,
    )

    for raw in _iter_documents(corpus_path):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id:
            continue
        title = raw.get("title")
        sections = raw.get("sections") or {}
        if not isinstance(sections, dict):
            continue
        for section_name, section_raw in sections.items():
            if not isinstance(section_raw, dict):
                continue
            for order, text in enumerate(_chunks_from_section(section_raw)):
                chunk_text = text.strip()
                if not chunk_text:
                    continue
                chunk_id = _stable_chunk_id(
                    doc_id, section_name, order, chunk_text,
                )
                yield SimpleNamespace(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section=section_name,
                    text=chunk_text,
                    title=title,
                    keywords=(),
                )


# ---------------------------------------------------------------------------
# Eval-only adapters around the production Retriever.
# ---------------------------------------------------------------------------


class DenseEvalRetrieverAdapter:
    """Two-pass adapter that surfaces ``candidate_doc_ids`` for dense.

    Pass 1: NoOpReranker, ``top_k = pool_size`` — captures the bi-encoder
    candidate list (50 docs).

    Pass 2: real reranker, ``top_k = final_top_k`` — produces the final
    reranked results.

    Both passes share the underlying FAISS index, embedder and metadata
    store via direct mutation of the Retriever's owned attributes — the
    same idiom ``_run_phase2a_latency_sweep_cli`` uses for its sibling
    candidate-recall pass.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        final_reranker: Any,
        final_top_k: int,
        pool_size: int = _PER_BACKEND_TOP_K,
    ) -> None:
        from app.capabilities.rag.reranker import NoOpReranker

        self._r = retriever
        self._final_reranker = final_reranker
        self._final_top_k = int(final_top_k)
        self._pool_size = int(pool_size)
        self._noop = NoOpReranker()

    def retrieve(self, query: str) -> Any:
        """Run pool + final passes; emit a duck-typed report."""
        prev_top_k = self._r._top_k  # noqa: SLF001 — owned by us
        prev_cand_k = self._r._candidate_k  # noqa: SLF001
        prev_reranker = self._r._reranker  # noqa: SLF001
        try:
            # Pool pass — bi-encoder only, no reranker.
            self._r._top_k = self._pool_size  # noqa: SLF001
            self._r._candidate_k = self._pool_size  # noqa: SLF001
            self._r._reranker = self._noop  # noqa: SLF001
            pool_t0 = time.perf_counter()
            pool_report = self._r.retrieve(query)
            pool_elapsed = round((time.perf_counter() - pool_t0) * 1000.0, 3)
            pool_doc_ids: List[str] = []
            for chunk in pool_report.results:
                doc_id = getattr(chunk, "doc_id", "") or ""
                if doc_id and doc_id not in pool_doc_ids:
                    pool_doc_ids.append(str(doc_id))

            # Final pass — real reranker, top final_top_k.
            self._r._top_k = self._final_top_k  # noqa: SLF001
            self._r._candidate_k = max(  # noqa: SLF001
                self._final_top_k, self._pool_size,
            )
            self._r._reranker = self._final_reranker  # noqa: SLF001
            final_report = self._r.retrieve(query)
        finally:
            self._r._top_k = prev_top_k  # noqa: SLF001
            self._r._candidate_k = prev_cand_k  # noqa: SLF001
            self._r._reranker = prev_reranker  # noqa: SLF001

        return SimpleNamespace(
            results=list(final_report.results),
            candidate_doc_ids=pool_doc_ids,
            index_version=getattr(final_report, "index_version", None),
            embedding_model=getattr(final_report, "embedding_model", None),
            reranker_name=getattr(final_report, "reranker_name", None),
            rerank_ms=getattr(final_report, "rerank_ms", None),
            # ``dense_retrieval_ms`` from the *final* pass; the pool pass
            # is supplemental and its wall-clock is separately captured
            # in ``pool_dense_retrieval_ms`` for diagnostics.
            dense_retrieval_ms=getattr(
                final_report, "dense_retrieval_ms", None,
            ),
            rerank_breakdown_ms=getattr(
                final_report, "rerank_breakdown_ms", None,
            ),
            # Diagnostic-only — not consumed by run_retrieval_eval.
            pool_dense_retrieval_ms=pool_elapsed,
            pool_size=self._pool_size,
        )


class DensePoolRetrieverAdapter:
    """Bi-encoder-only retrieve() for use as the dense arm of RRF hybrid.

    Mutates ``_top_k`` / ``_candidate_k`` / ``_reranker`` per call so the
    underlying Retriever (configured by the caller for dense+rerank)
    behaves like a NoOp top-50 retriever for the duration of the call.
    """

    def __init__(self, retriever: Any, *, pool_size: int = _PER_BACKEND_TOP_K) -> None:
        from app.capabilities.rag.reranker import NoOpReranker

        self._r = retriever
        self._pool_size = int(pool_size)
        self._noop = NoOpReranker()

    def retrieve(self, query: str) -> Any:
        prev_top_k = self._r._top_k  # noqa: SLF001
        prev_cand_k = self._r._candidate_k  # noqa: SLF001
        prev_reranker = self._r._reranker  # noqa: SLF001
        try:
            self._r._top_k = self._pool_size  # noqa: SLF001
            self._r._candidate_k = self._pool_size  # noqa: SLF001
            self._r._reranker = self._noop  # noqa: SLF001
            return self._r.retrieve(query)
        finally:
            self._r._top_k = prev_top_k  # noqa: SLF001
            self._r._candidate_k = prev_cand_k  # noqa: SLF001
            self._r._reranker = prev_reranker  # noqa: SLF001


# ---------------------------------------------------------------------------
# Sweep entry point.
# ---------------------------------------------------------------------------


def _dense_cache_key(
    corpus_path: Path,
    embedding_model: str,
    max_seq_length: int,
) -> str:
    """Stable digest over (corpus, model, max_seq_length).

    Different ``max_seq_length`` settings produce different bge-m3
    activations on long chunks, so the cache key includes it. The
    embedding batch size is *not* in the key — different batch sizes
    produce numerically identical vectors (the model is deterministic
    in eval mode), so reusing a 32-batch index from a 64-batch run is
    safe.
    """
    h = hashlib.sha256()
    h.update(str(corpus_path.resolve()).encode("utf-8"))
    h.update(b"|")
    h.update(str(corpus_path.stat().st_mtime_ns).encode("utf-8"))
    h.update(b"|")
    h.update(embedding_model.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(max_seq_length)).encode("utf-8"))
    return h.hexdigest()[:16]


def _persist_chunks(chunks_path: Path, rows: List[Any]) -> None:
    with chunks_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps({
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "text": r.text,
                "faiss_row_id": r.faiss_row_id,
            }, ensure_ascii=False) + "\n")


def _load_chunks(chunks_path: Path) -> List[Any]:
    from app.capabilities.rag.metadata_store import ChunkLookupResult

    rows: List[ChunkLookupResult] = []
    with chunks_path.open("r", encoding="utf-8") as fp:
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
    return rows


def _build_dense_stack(corpus_path: Path, args: argparse.Namespace):
    """Build (or restore from cache) the bi-encoder + cross-encoder stack.

    Cache layout (key derived from corpus + embedding model + max_seq):
      eval/_cache/dense_index/<key>/
        faiss.index    — FaissIndex artifact
        build.json     — IndexBuildInfo metadata
        chunks.jsonl   — InMemoryMetadataStore rows for offline lookup

    If all three files exist the embedder still loads (queries need it
    at retrieve-time) but the 47k-chunk passage encode is skipped. On a
    cache miss we run ``build_offline_rag_stack`` against the cache dir
    so a successful build populates the cache for the next run.
    """
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.capabilities.rag.retriever import Retriever
    from app.core.config import get_settings
    from eval.harness.offline_corpus import (
        OfflineCorpusInfo,
        _InMemoryMetadataStore,
        build_offline_rag_stack,
    )

    settings = get_settings()
    explicit_cache = Path(args.cache_dir) if args.cache_dir else None
    if (
        explicit_cache is not None
        and (explicit_cache / "faiss.index").exists()
        and (explicit_cache / "build.json").exists()
        and (explicit_cache / _CHUNKS_FILE).exists()
    ):
        # The caller pointed --cache-dir at a directory that already
        # contains the three artifacts. Skip the per-corpus hash subdir
        # and use the path directly. This is how we reuse the
        # ``agent_loop_ab/_indexes/<key>`` cache that the A/B harness
        # already built — no copy needed.
        cache_dir = explicit_cache
    else:
        cache_key = _dense_cache_key(
            corpus_path, settings.rag_embedding_model, int(args.max_seq_length),
        )
        cache_root = explicit_cache or _DEFAULT_CACHE_ROOT
        cache_dir = cache_root / cache_key
    chunks_path = cache_dir / _CHUNKS_FILE
    faiss_path = cache_dir / "faiss.index"
    build_meta = cache_dir / "build.json"
    cache_hit = (
        not args.force_rebuild
        and faiss_path.exists()
        and build_meta.exists()
        and chunks_path.exists()
    )

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=True,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    reranker = CrossEncoderReranker(
        model_name=str(args.reranker_model),
        max_length=int(args.reranker_max_length),
        batch_size=int(args.reranker_batch_size),
        text_max_chars=int(args.reranker_text_max_chars),
        device=args.reranker_device or None,
        collect_stage_timings=True,
    )

    if cache_hit:
        log.info(
            "Dense cache hit at %s — skipping passage embed pass.",
            cache_dir,
        )
        index = FaissIndex(cache_dir)
        info = index.load()
        if info.embedding_model != settings.rag_embedding_model:
            raise RuntimeError(
                f"Cached FAISS embedding_model={info.embedding_model!r} "
                f"differs from settings={settings.rag_embedding_model!r}; "
                "delete the cache dir or pass --force-rebuild."
            )
        rows = _load_chunks(chunks_path)
        store = _InMemoryMetadataStore(info.index_version, rows)
        retriever = Retriever(
            embedder=embedder,
            index=index,
            metadata=store,
            top_k=int(args.dense_final_top_k),
            reranker=reranker,
            candidate_k=_PER_BACKEND_TOP_K,
        )
        retriever.ensure_ready()
        offline_info = OfflineCorpusInfo(
            corpus_path=str(corpus_path),
            document_count=len({r.doc_id for r in rows}),
            chunk_count=len(rows),
            index_version=info.index_version,
            embedding_model=info.embedding_model,
            dimension=info.dimension,
        )
        return retriever, reranker, offline_info, settings

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Dense cache miss at %s — encoding %s with %s …",
        cache_dir, corpus_path, settings.rag_embedding_model,
    )
    retriever, _generator, offline_info = build_offline_rag_stack(
        corpus_path,
        embedder=embedder,
        index_dir=cache_dir,
        top_k=int(args.dense_final_top_k),
        reranker=reranker,
        candidate_k=_PER_BACKEND_TOP_K,
    )
    # Persist the chunk metadata alongside the FAISS index so a follow-up
    # run finds a complete cache.
    rows_for_cache = list(retriever._metadata._by_row.values())  # noqa: SLF001
    _persist_chunks(chunks_path, rows_for_cache)
    log.info("Cached %d chunk metadata rows to %s", len(rows_for_cache), chunks_path)
    return retriever, reranker, offline_info, settings


def _persist_outputs(
    out_dir: Path,
    sweep_report: Any,
    cells_meta: List[Dict[str, Any]],
    silver_dataset_path: Path,
    corpus_path: Path,
    settings: Any,
    info: Any,
    reranker_name: str,
) -> None:
    from eval.harness.pareto_frontier import (
        compute_pareto_frontier,
        pareto_to_dict,
        render_pareto_markdown,
    )
    from eval.harness.retrieval_sweep import (
        render_sweep_markdown,
        sweep_report_to_dict,
        sweep_to_topn_sweep_report,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) sweep_summary.json — full sweep report.
    sweep_payload = sweep_report_to_dict(sweep_report)
    sweep_payload["run"] = {
        "dataset": str(silver_dataset_path),
        "corpus_path": str(corpus_path),
        "embedding_model": settings.rag_embedding_model,
        "reranker_model": reranker_name,
        "per_backend_top_k": _PER_BACKEND_TOP_K,
        "k_rrf": _K_RRF,
        "document_count": info.document_count,
        "chunk_count": info.chunk_count,
        "index_version": info.index_version,
        "dimension": info.dimension,
        "started_at": cells_meta[0].get("started_at") if cells_meta else None,
        "finished_at": cells_meta[-1].get("finished_at") if cells_meta else None,
    }
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(sweep_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) sweep_report.md — markdown overview.
    (out_dir / "sweep_report.md").write_text(
        render_sweep_markdown(sweep_report), encoding="utf-8",
    )

    # 3) cell_comparison.csv — flat headline comparison.
    headers = [
        "label", "kind", "variant", "candidate_k", "final_top_k",
        "row_count",
        "hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@10",
        "candidateHit@10", "candidateHit@20", "candidateHit@50",
        "candidateRecall@10", "candidateRecall@20", "candidateRecall@50",
        "preRerankHit@5", "rerankUpliftHit@5", "rerankUpliftMrr@10",
        "duplicateDocRatio@5", "duplicateDocRatio@10",
        "uniqueDocCount@10", "sectionDiversity@10",
        "avgTotalRetrievalMs", "p95TotalRetrievalMs",
        "avgDenseRetrievalMs", "avgRerankMs",
        "avgCandidateCount", "avgFinalContextCount",
        "qualityScore", "efficiencyScore",
    ]
    with (out_dir / "cell_comparison.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for cell in sweep_report.cells:
            s = cell.summary
            cand = s.candidate_hit_rates or {}
            cand_recall = s.candidate_recalls or {}
            dup = s.duplicate_doc_ratios or {}
            udc = s.unique_doc_counts or {}
            sec_div = s.section_diversities or {}
            writer.writerow([
                cell.label, cell.retriever_kind, cell.embedding_text_variant,
                cell.candidate_k, cell.final_top_k,
                s.row_count,
                _f(s.mean_hit_at_1), _f(s.mean_hit_at_3), _f(s.mean_hit_at_5),
                _f(s.mean_mrr_at_10), _f(s.mean_ndcg_at_10),
                _f(cand.get("10")), _f(cand.get("20")), _f(cand.get("50")),
                _f(cand_recall.get("10")), _f(cand_recall.get("20")), _f(cand_recall.get("50")),
                _f(s.mean_pre_rerank_hit_at_5),
                _f(s.rerank_uplift_hit_at_5), _f(s.rerank_uplift_mrr_at_10),
                _f(dup.get("5")), _f(dup.get("10")),
                _f(udc.get("10")), _f(sec_div.get("10")),
                _f(s.avg_total_retrieval_ms or s.mean_retrieval_ms),
                _f(s.p95_total_retrieval_ms or s.p95_retrieval_ms),
                _f(s.mean_dense_retrieval_ms), _f(s.mean_rerank_ms),
                _f(s.avg_candidate_count), _f(s.avg_final_context_count),
                _f(s.quality_score), _f(s.efficiency_score),
            ])

    # 4) query_type_breakdown.csv — per-cell × per-bucket headlines.
    qt_headers = [
        "label", "query_type", "row_count",
        "hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@10",
        "candidateHit@50", "candidateRecall@50",
        "p95TotalRetrievalMs",
    ]
    with (out_dir / "query_type_breakdown.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(qt_headers)
        for cell in sweep_report.cells:
            for qtype, payload in (cell.summary.by_query_type or {}).items():
                # ``by_query_type`` flattens the headline metrics under
                # short keys (``hit_at_1`` not ``mean_hit_at_1``); see
                # ``_aggregate`` in retrieval_eval.py for the schema.
                writer.writerow([
                    cell.label, qtype, payload.get("count"),
                    _f(payload.get("hit_at_1")),
                    _f(payload.get("hit_at_3")),
                    _f(payload.get("hit_at_5")),
                    _f(payload.get("mrr_at_10")),
                    _f(payload.get("ndcg_at_10")),
                    _f(payload.get("candidate_hit_at_50")),
                    _f(payload.get("candidate_recall_at_50")),
                    _f(payload.get("p95_total_retrieval_ms")
                       or payload.get("p95_retrieval_ms")),
                ])

    # 5) diagnostics.json — per-cell flag dump.
    diagnostics = {
        cell.label: {
            "diagnostics": dict(cell.summary.diagnostics or {}),
            "row_count": cell.summary.row_count,
            "rows_with_expected_doc_ids": cell.summary.rows_with_expected_doc_ids,
            "error_count": cell.summary.error_count,
        }
        for cell in sweep_report.cells
    }
    (out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6) Pareto frontier (quality_score vs p95_total_retrieval_ms).
    topn = sweep_to_topn_sweep_report(sweep_report)
    try:
        pareto = compute_pareto_frontier(
            topn,
            metric="mean_hit_at_5",
            latency="total_query_p95_ms",
        )
        (out_dir / "pareto_quality_vs_latency.json").write_text(
            json.dumps(pareto_to_dict(pareto), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "pareto_quality_vs_latency.md").write_text(
            render_pareto_markdown(pareto), encoding="utf-8",
        )
    except Exception as ex:  # pragma: no cover — best-effort
        log.warning("Pareto rendering failed: %s: %s", type(ex).__name__, ex)


def _f(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _maybe_emit_failed_queries_and_diffs(
    out_dir: Path,
    sweep_report: Any,
    rows_by_cell: Dict[str, List[Any]],
) -> None:
    """Emit failed_queries.jsonl + top regressions/improvements.

    Top regressions / improvements compare every non-baseline cell's
    per-row hit@5 against ``dense_only_current_top5`` (the established
    legacy baseline). A regression is a query that the baseline got
    right but the candidate did not; an improvement is the inverse.
    """
    baseline_label = "dense_only_current_top5"
    if baseline_label not in rows_by_cell:
        return

    # failed_queries.jsonl — every row that errored in any cell.
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

    baseline_rows = {r.id: r for r in rows_by_cell[baseline_label]}

    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    for label, rows in rows_by_cell.items():
        if label == baseline_label:
            continue
        for row in rows:
            base = baseline_rows.get(row.id)
            if base is None:
                continue
            base_hit = base.hit_at_5
            cand_hit = row.hit_at_5
            if base_hit is None or cand_hit is None:
                continue
            if base_hit > 0.5 and cand_hit <= 0.5:
                regressions.append({
                    "cell": label,
                    "id": row.id,
                    "query": row.query,
                    "baseline_hit_at_5": base_hit,
                    "candidate_hit_at_5": cand_hit,
                    "expected_doc_ids": list(row.expected_doc_ids),
                    "candidate_top_doc_ids": list(row.retrieved_doc_ids[:5]),
                })
            elif cand_hit > 0.5 and base_hit <= 0.5:
                improvements.append({
                    "cell": label,
                    "id": row.id,
                    "query": row.query,
                    "baseline_hit_at_5": base_hit,
                    "candidate_hit_at_5": cand_hit,
                    "expected_doc_ids": list(row.expected_doc_ids),
                    "candidate_top_doc_ids": list(row.retrieved_doc_ids[:5]),
                })

    with (out_dir / "top_regressions.jsonl").open("w", encoding="utf-8") as fp:
        for entry in regressions:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (out_dir / "top_improvements.jsonl").open("w", encoding="utf-8") as fp:
        for entry in improvements:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Driver — orchestrates the per-cell run loop directly so per-cell row
# lists are kept for the diff dumps. ``run_retrieval_sweep`` only keeps
# a small per-cell ``rows_sample`` on the report, which isn't enough for
# regression / improvement analysis.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
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
    parser.add_argument("--dense-final-top-k", type=int, default=10)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on the dataset row count (for quick debug).",
    )
    parser.add_argument(
        "--skip-dense", action="store_true",
        help="Skip the two dense-only cells (debug only).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help=(
            "Override for the persistent FAISS cache root. "
            f"Defaults to {_DEFAULT_CACHE_ROOT}."
        ),
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore the cache and rebuild the FAISS index from scratch.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    if out_dir.exists():
        # We refuse to overwrite an existing report dir to honour the
        # spec's "기존 historical baseline 산출물은 절대 덮어쓰지 않는다"
        # rule. The caller should pick a fresh --out-dir or wait a minute
        # so the timestamped default rolls over.
        log.error(
            "Refusing to overwrite existing out-dir %s; pick a new path.",
            out_dir,
        )
        return 2

    log.info("Output dir: %s", out_dir)

    # --- 1. Build dense stack (embedder + FAISS + reranker) ---
    if args.skip_dense:
        # Stub mode — only used for quick BM25/hybrid debugging without
        # a GPU. Hybrid will fall back to a tiny in-process index.
        log.warning("Skipping dense stack — running BM25/hybrid only.")
        retriever = None
        reranker = None
        info = SimpleNamespace(
            document_count=0, chunk_count=0,
            index_version="skip-dense",
            embedding_model="<skip>",
            dimension=0,
            corpus_path=str(args.corpus),
        )
        from app.core.config import get_settings
        settings = get_settings()
    else:
        retriever, reranker, info, settings = _build_dense_stack(
            args.corpus, args,
        )

    # --- 2. Build BM25 indexes (raw + title_section) ---
    from eval.harness.bm25_retriever import BM25EvalRetriever, build_bm25_index
    from eval.harness.embedding_text_builder import (
        VARIANT_RAW, VARIANT_TITLE_SECTION,
    )
    from eval.harness.hybrid_retriever import RRFHybridEvalRetriever
    from eval.harness.io_utils import load_jsonl
    from eval.harness.retrieval_eval import (
        DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS,
    )
    from eval.harness.retrieval_sweep import (
        KIND_BM25, KIND_DENSE, KIND_HYBRID,
        RetrievalSweepCell, RetrievalSweepConfig, RetrievalSweepReport,
    )

    log.info("Reading corpus chunks for BM25 (re-using dense chunker)…")
    chunks = list(_iter_dense_chunks(args.corpus))
    log.info("Building BM25 index variant=raw (%d chunks)…", len(chunks))
    bm25_raw_index = build_bm25_index(
        chunks, embedding_text_variant=VARIANT_RAW,
    )
    log.info("Building BM25 index variant=title_section…")
    bm25_ts_index = build_bm25_index(
        chunks, embedding_text_variant=VARIANT_TITLE_SECTION,
    )
    bm25_raw_retriever = BM25EvalRetriever(
        bm25_raw_index, top_k=_PER_BACKEND_TOP_K, name="bm25-okapi-raw",
    )
    bm25_ts_retriever = BM25EvalRetriever(
        bm25_ts_index, top_k=_PER_BACKEND_TOP_K, name="bm25-okapi-title-section",
    )

    # --- 3. Compose retriever wrappers per cell ---
    cells_to_run: List[RetrievalSweepConfig] = []
    if not args.skip_dense:
        dense_top5 = DenseEvalRetrieverAdapter(
            retriever, final_reranker=reranker, final_top_k=5,
            pool_size=_PER_BACKEND_TOP_K,
        )
        dense_top10 = DenseEvalRetrieverAdapter(
            retriever, final_reranker=reranker, final_top_k=10,
            pool_size=_PER_BACKEND_TOP_K,
        )
        cells_to_run.append(RetrievalSweepConfig(
            label="dense_only_current_top5",
            retriever_kind=KIND_DENSE,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=_PER_BACKEND_TOP_K,
            final_top_k=5,
            retriever=dense_top5,
            extra={
                "reranker_model": args.reranker_model,
                "embedding_model": settings.rag_embedding_model,
                "pool_size": _PER_BACKEND_TOP_K,
            },
        ))
        cells_to_run.append(RetrievalSweepConfig(
            label="dense_only_current_top10",
            retriever_kind=KIND_DENSE,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=_PER_BACKEND_TOP_K,
            final_top_k=10,
            retriever=dense_top10,
            extra={
                "reranker_model": args.reranker_model,
                "embedding_model": settings.rag_embedding_model,
                "pool_size": _PER_BACKEND_TOP_K,
            },
        ))

    cells_to_run.append(RetrievalSweepConfig(
        label="bm25_raw_top5",
        retriever_kind=KIND_BM25,
        embedding_text_variant=VARIANT_RAW,
        candidate_k=_PER_BACKEND_TOP_K,
        final_top_k=5,
        retriever=bm25_raw_retriever,
        extra={"index_chunks": len(chunks), "k1": 1.5, "b": 0.75},
    ))
    cells_to_run.append(RetrievalSweepConfig(
        label="bm25_title_section_top5",
        retriever_kind=KIND_BM25,
        embedding_text_variant=VARIANT_TITLE_SECTION,
        candidate_k=_PER_BACKEND_TOP_K,
        final_top_k=5,
        retriever=bm25_ts_retriever,
        extra={"index_chunks": len(chunks), "k1": 1.5, "b": 0.75},
    ))

    if not args.skip_dense:
        dense_pool_for_hybrid = DensePoolRetrieverAdapter(
            retriever, pool_size=_PER_BACKEND_TOP_K,
        )
        hybrid_top5 = RRFHybridEvalRetriever(
            dense=dense_pool_for_hybrid,
            sparse=BM25EvalRetriever(
                bm25_raw_index, top_k=_PER_BACKEND_TOP_K,
                name="bm25-okapi-raw-hybrid",
            ),
            k_rrf=_K_RRF,
            final_top_k=5,
            per_backend_top_k=_PER_BACKEND_TOP_K,
            name="hybrid:dense+bm25_raw",
        )
        hybrid_top10 = RRFHybridEvalRetriever(
            dense=DensePoolRetrieverAdapter(
                retriever, pool_size=_PER_BACKEND_TOP_K,
            ),
            sparse=BM25EvalRetriever(
                bm25_raw_index, top_k=_PER_BACKEND_TOP_K,
                name="bm25-okapi-raw-hybrid",
            ),
            k_rrf=_K_RRF,
            final_top_k=10,
            per_backend_top_k=_PER_BACKEND_TOP_K,
            name="hybrid:dense+bm25_raw",
        )
        cells_to_run.append(RetrievalSweepConfig(
            label="hybrid_rrf_top5",
            retriever_kind=KIND_HYBRID,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=_PER_BACKEND_TOP_K,
            final_top_k=5,
            retriever=hybrid_top5,
            extra={
                "k_rrf": _K_RRF,
                "per_backend_top_k": _PER_BACKEND_TOP_K,
                "embedding_model": settings.rag_embedding_model,
            },
        ))
        cells_to_run.append(RetrievalSweepConfig(
            label="hybrid_rrf_top10",
            retriever_kind=KIND_HYBRID,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=_PER_BACKEND_TOP_K,
            final_top_k=10,
            retriever=hybrid_top10,
            extra={
                "k_rrf": _K_RRF,
                "per_backend_top_k": _PER_BACKEND_TOP_K,
                "embedding_model": settings.rag_embedding_model,
            },
        ))

    # --- 4. Load dataset ---
    dataset = load_jsonl(args.dataset)
    if args.limit is not None and args.limit > 0:
        dataset = list(dataset)[: int(args.limit)]
    log.info(
        "Loaded %d query rows from %s (limit=%s)",
        len(dataset), args.dataset, args.limit,
    )

    # --- 5. Run cells one at a time so we can keep all rows for diffs ---
    from eval.harness.retrieval_eval import run_retrieval_eval

    cells: List[RetrievalSweepCell] = []
    cells_meta: List[Dict[str, Any]] = []
    rows_by_cell: Dict[str, List[Any]] = {}

    for cfg in cells_to_run:
        log.info(
            "running cell %s (kind=%s variant=%s final_top_k=%d)",
            cfg.label, cfg.retriever_kind,
            cfg.embedding_text_variant, cfg.final_top_k,
        )
        cell_started = datetime.now().isoformat(timespec="seconds")
        summary, rows, _, _ = run_retrieval_eval(
            list(dataset),
            retriever=cfg.retriever,
            top_k=cfg.final_top_k,
            mrr_k=10,
            ndcg_k=10,
            candidate_ks=DEFAULT_CANDIDATE_KS,
            diversity_ks=DEFAULT_DIVERSITY_KS,
            dataset_path=str(args.dataset),
            corpus_path=str(args.corpus),
        )
        cell_finished = datetime.now().isoformat(timespec="seconds")

        sample = []
        for row in rows[:10]:
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
        cells_meta.append({
            "label": cfg.label,
            "started_at": cell_started,
            "finished_at": cell_finished,
        })
        rows_by_cell[cfg.label] = rows
        log.info(
            "  -> %s: hit@5=%.4f mrr@10=%.4f cand@50=%s p95=%.1fms",
            cfg.label,
            (summary.mean_hit_at_5 or 0.0),
            (summary.mean_mrr_at_10 or 0.0),
            (summary.candidate_hit_rates or {}).get("50"),
            float(
                summary.p95_total_retrieval_ms or summary.p95_retrieval_ms or 0.0
            ),
        )

    sweep_report = RetrievalSweepReport(
        schema="phase2-retrieval-sweep.v1",
        dataset_path=str(args.dataset),
        corpus_path=str(args.corpus),
        cells=cells,
        caveats=[
            "Dense cells re-run the bi-encoder pool with NoOpReranker so "
            "candidate_doc_ids fires; this doubles the per-query dense "
            "wall-clock relative to the legacy baseline.",
            "Hybrid cells call the dense backend with NoOpReranker (top "
            f"{_PER_BACKEND_TOP_K}); no cross-encoder rerank is applied "
            "after RRF fusion.",
        ],
    )

    # --- 6. Persist artifacts ---
    reranker_name = (
        args.reranker_model if not args.skip_dense else "<skip-dense>"
    )
    _persist_outputs(
        out_dir=out_dir,
        sweep_report=sweep_report,
        cells_meta=cells_meta,
        silver_dataset_path=args.dataset,
        corpus_path=args.corpus,
        settings=settings,
        info=info,
        reranker_name=reranker_name,
    )
    _maybe_emit_failed_queries_and_diffs(
        out_dir=out_dir,
        sweep_report=sweep_report,
        rows_by_cell=rows_by_cell,
    )

    log.info("Sweep finished — artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
