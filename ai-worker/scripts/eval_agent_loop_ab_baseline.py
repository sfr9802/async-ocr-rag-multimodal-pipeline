"""LEGACY V3 ONLY - historical legacy-baseline-final agent-loop A/B.

This script preserves the old legacy-baseline-final replay path. Do not use it
as an active Phase 7 eval/tuning source of truth.

Run the legacy-vs-graph A/B harness using legacy-baseline-final config.

Loads the locked legacy baseline manifest + ``selected_config.json``,
applies those values to a per-run ``WorkerSettings`` copy (without
mutating the process-global settings cache), wires both backends to
the same retriever / reranker / parser / critic / rewriter / budget the
baseline sweep used, and writes the standard A/B output set
(``raw_results.jsonl`` / ``summary.csv`` / ``comparison_summary.json``)
plus a ``comparison_report.md`` summarising the verdict.

Side-effect contract:

  * No Redis queue / TaskRunner involvement — same as the regular
    ``eval_agent_loop_ab.py``.
  * No callback emitted to core-api.
  * No DB writes (the metadata store is read-only via the retriever).
  * Production runtime defaults (``rag_top_k=5``, ``rag_reranker='off'``,
    ...) are NOT changed — settings are patched in memory for this run
    only.
  * ``agent_loop_backend`` default stays ``'legacy'``; the graph backend
    is only exercised through this offline harness.
  * ``AgentLoopGraph`` build/invoke failures are surfaced through
    ``runner.last_failure`` and recorded as ``success=False`` rows in
    the comparison summary so a degraded graph backend can never
    masquerade as a healthy one.

Usage::

    python -m scripts.eval_agent_loop_ab_baseline \\
        --baseline-dir eval/reports/legacy-baseline-final
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.capabilities.agent.critic import RuleCritic
from app.capabilities.agent.graph_loop import AgentLoopGraph
from app.capabilities.agent.loop import AgentLoopController, LoopBudget
from app.capabilities.agent.rewriter import NoOpQueryRewriter
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.rag.embeddings import (
    SentenceTransformerEmbedder,
    resolve_max_seq_length,
)
from app.capabilities.rag.generation import ExtractiveGenerator
from app.capabilities.rag.query_parser import RegexQueryParser
from app.capabilities.rag.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankerProvider,
)
from app.core.config import WorkerSettings, get_settings
from eval.harness.agent_loop_ab import (
    AgentLoopABComparisonRow,
    LATENCY_RATIO_ADOPT_CEILING,
    RECOMMENDATION_ADOPT,
    RECOMMENDATION_HOLD_EXPERIMENTAL,
    RECOMMENDATION_HOLD_NO_QUALITY_GAIN,
    RECOMMENDATION_HOLD_REVIEW_ERRORS,
    RECOMMENDATION_HOLD_REVIEW_REGRESSIONS,
    VERDICT_GRAPH_WIN,
    VERDICT_LEGACY_WIN,
    VERDICT_REGRESSION,
    VERDICT_TIE,
    has_quality_lift,
    load_query_rows,
    make_default_executor_builder,
    run_ab_eval,
    write_outputs,
)


log = logging.getLogger("scripts.eval_agent_loop_ab_baseline")


_DEFAULT_BASELINE_DIR = Path("eval/reports/legacy-baseline-final")
_DEFAULT_OUTPUT_ROOT = Path("eval/agent_loop_ab")
# Cache directory for the offline FAISS index + chunk metadata so that
# successive baseline runs can skip the 30-60-minute corpus embedding
# step. Bind the cache key to (corpus path + size + mtime + embedding
# model + max_seq_length) so a corpus refresh or a model swap forces a
# rebuild — a stale index silently corrupts retrieval otherwise.
_DEFAULT_CACHE_ROOT = Path("eval/agent_loop_ab/_indexes")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    baseline_dir = Path(args.baseline_dir).resolve()
    manifest_path = baseline_dir / "baseline_manifest.json"
    selected_path = baseline_dir / "selected_config.json"
    if not manifest_path.exists():
        log.error("Baseline manifest not found: %s", manifest_path)
        return 2
    if not selected_path.exists():
        log.error("Selected config not found: %s", selected_path)
        return 2

    manifest = _load_json(manifest_path)
    selected = _load_json(selected_path)
    log.info(
        "Loaded baseline: commit=%s queryCount=%s indexVersion=%s tier=%s",
        manifest.get("commitHash"),
        manifest.get("queryCount"),
        (manifest.get("corpusInfo") or {}).get("indexVersion"),
        manifest.get("selectedMode", {}).get("tier"),
    )

    queries_path = _resolve_query_path(manifest, args)
    if not queries_path.exists():
        log.error(
            "Query set not found at %s (resolved from baseline manifest "
            "querySetPath=%r). Pass --queries to override.",
            queries_path, manifest.get("querySetPath"),
        )
        return 2
    queries = load_query_rows(queries_path)
    if not queries:
        log.error("No queries loaded from %s — aborting.", queries_path)
        return 2
    if args.max_queries is not None and args.max_queries > 0:
        queries = queries[: args.max_queries]
        log.info("Trimmed query set to first %d rows (smoke).", len(queries))

    baseline_settings = _patch_settings_for_baseline(get_settings(), manifest, selected)
    log.info(
        "Baseline-derived settings: rag_top_k=%d rag_candidate_k=%d "
        "rag_reranker=%s rag_rerank_batch=%d rag_use_mmr=%s "
        "rag_embedding_model=%s rag_embedding_max_seq_length=%d",
        baseline_settings.rag_top_k,
        baseline_settings.rag_candidate_k,
        baseline_settings.rag_reranker,
        baseline_settings.rag_rerank_batch,
        baseline_settings.rag_use_mmr,
        baseline_settings.rag_embedding_model,
        baseline_settings.rag_embedding_max_seq_length,
    )

    corpus_path = _resolve_corpus_path(manifest, args)
    if not corpus_path or not corpus_path.exists():
        log.error(
            "Baseline corpus not found at %s (resolved from manifest). "
            "An offline run needs the corpus jsonl to rebuild the FAISS "
            "index in-memory.",
            corpus_path,
        )
        return 2

    try:
        retriever, generator, corpus_info, cache_status = _build_offline_retriever(
            settings=baseline_settings,
            corpus_path=corpus_path,
            cache_root=(args.cache_dir or _DEFAULT_CACHE_ROOT).resolve(),
            use_cache=not args.no_cache,
        )
    except Exception as ex:
        log.error(
            "Failed to build offline retriever from baseline corpus "
            "(%s: %s). Corpus path: %s",
            type(ex).__name__, ex, corpus_path,
        )
        return 3
    log.info(
        "Offline retriever ready: docs=%d chunks=%d index_version=%s "
        "embedder=%s reranker=%s cache=%s",
        corpus_info.document_count,
        corpus_info.chunk_count,
        corpus_info.index_version,
        corpus_info.embedding_model,
        baseline_settings.rag_reranker,
        cache_status,
    )

    parser = RegexQueryParser()
    critic = RuleCritic()
    rewriter = NoOpQueryRewriter()
    answer_generator = generator if args.use_baseline_generator else ExtractiveGenerator()
    synthesizer = (
        AgentSynthesizer(answer_generator) if args.synthesize else None
    )

    agent_cfg = (selected.get("agentLoop") or manifest.get("agentLoop") or {})
    budget = LoopBudget(
        max_iter=int(agent_cfg.get("maxIter", 3)),
        max_total_ms=int(agent_cfg.get("maxTotalMs", 15_000)),
        max_llm_tokens=int(agent_cfg.get("maxLlmTokens", 4_000)),
        min_confidence_to_stop=float(agent_cfg.get("minStopConfidence", 0.75)),
    )
    log.info(
        "Agent loop budget: max_iter=%d max_total_ms=%d max_llm_tokens=%d "
        "min_confidence_to_stop=%.2f",
        budget.max_iter, budget.max_total_ms, budget.max_llm_tokens,
        budget.min_confidence_to_stop,
    )

    legacy_runner = AgentLoopController(
        critic=critic,
        rewriter=rewriter,
        parser=parser,
        budget=budget,
    )
    try:
        graph_runner = AgentLoopGraph(
            critic=critic,
            rewriter=rewriter,
            parser=parser,
            budget=budget,
        )
    except Exception as ex:
        log.error(
            "AgentLoopGraph could not be instantiated (%s: %s). Verify "
            "langgraph is installed (pip install -r requirements.txt).",
            type(ex).__name__, ex,
        )
        return 3

    executor_builder = make_default_executor_builder(
        retriever=retriever, generator=answer_generator,
    )

    run_name = args.run_name or _default_run_name()
    output_dir = (args.output_dir or _DEFAULT_OUTPUT_ROOT) / run_name

    started = time.perf_counter()
    rows, summary = run_ab_eval(
        queries=queries,
        legacy_runner=legacy_runner,
        graph_runner=graph_runner,
        parser=parser,
        executor_builder=executor_builder,
        legacy_synthesizer=synthesizer,
        graph_synthesizer=synthesizer,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    metadata = {
        "runName": run_name,
        "baselineDir": str(baseline_dir),
        "baselineCommit": manifest.get("commitHash"),
        "baselineQuerySetPath": manifest.get("querySetPath"),
        "querySetPath": str(queries_path),
        "corpusPath": str(corpus_path),
        "queryCount": len(queries),
        "indexVersion": corpus_info.index_version,
        "documentCount": corpus_info.document_count,
        "chunkCount": corpus_info.chunk_count,
        "embeddingModel": corpus_info.embedding_model,
        "embeddingDimension": corpus_info.dimension,
        "retriever": {
            "denseTopN": baseline_settings.rag_top_k,
            "candidateK": baseline_settings.rag_candidate_k,
            "finalTopK": baseline_settings.rag_top_k,
            "useMmr": baseline_settings.rag_use_mmr,
            "mmrLambda": (
                baseline_settings.rag_mmr_lambda
                if baseline_settings.rag_use_mmr else None
            ),
            "embeddingModel": baseline_settings.rag_embedding_model,
            "embeddingMaxSeqLength": baseline_settings.rag_embedding_max_seq_length,
            "embeddingBatchSize": baseline_settings.rag_embedding_batch_size,
            "indexBackend": "faiss-in-memory",
        },
        "reranker": {
            "name": baseline_settings.rag_reranker,
            "batchSize": baseline_settings.rag_rerank_batch,
        },
        "agentLoop": {
            "backend": "legacy_vs_graph",
            "critic": "rule",
            "rewriter": "noop",
            "parser": "regex",
            "maxIter": budget.max_iter,
            "maxTotalMs": budget.max_total_ms,
            "maxLlmTokens": budget.max_llm_tokens,
            "minStopConfidence": budget.min_confidence_to_stop,
        },
        "synthesize": args.synthesize,
        "useBaselineGenerator": args.use_baseline_generator,
        "indexCache": cache_status,
        "elapsedMs": elapsed_ms,
        "evaluatedAt": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    paths = write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
        metadata=metadata,
    )

    report_path = output_dir / "comparison_report.md"
    report_path.write_text(
        _render_report(
            metadata=metadata,
            summary=summary,
            rows=rows,
        ),
        encoding="utf-8",
    )
    paths["comparison_report"] = report_path

    print(json.dumps(
        {
            "queryCount": len(queries),
            "elapsedMs": elapsed_ms,
            "summary": summary,
            "paths": {k: str(v) for k, v in paths.items()},
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline legacy-vs-graph A/B harness using the locked "
            "legacy baseline config (eval/reports/legacy-baseline-final/)."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=_DEFAULT_BASELINE_DIR,
        help=(
            "Directory holding baseline_manifest.json + selected_config.json. "
            f"Defaults to {_DEFAULT_BASELINE_DIR}."
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help=(
            "Override the query set. Defaults to the path recorded in the "
            "baseline manifest (querySetPath)."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Override the corpus jsonl. Defaults to the path recorded in "
            "the baseline manifest (corpusInfo.path). The corpus is chunked "
            "and embedded into an in-memory FAISS index per run."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached offline FAISS indexes. Defaults to "
            f"{_DEFAULT_CACHE_ROOT}. Cache keys derive from corpus path + "
            "size + mtime + embedding model + max_seq_length, so a corpus "
            "refresh or a model swap forces a rebuild automatically."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Skip the cached-index lookup and rebuild the offline FAISS "
            "index every run (matches the original phase2a-latency-sweep "
            "behaviour). The 47K-chunk bge-m3 embedding step takes 30-60 "
            "min on RTX 5080; default behaviour caches the result."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Sub-directory name under --output-dir. Defaults to "
            "'legacy-vs-graph-<YYYYMMDD-HHMM>' so successive invocations "
            "don't clobber."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output root directory. Defaults to {_DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap the loaded query set (smoke / debug aid).",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help=(
            "Run AgentSynthesizer over outcome.aggregated_chunks. Off by "
            "default — keeping it off makes the comparison apples-to-apples "
            "across backends and avoids extra LLM calls."
        ),
    )
    parser.add_argument(
        "--use-baseline-generator",
        action="store_true",
        help=(
            "Use the rag_generator the baseline shared bundle returns "
            "(extractive by default in the registry). Off by default — "
            "the harness pins ExtractiveGenerator so a Claude/Ollama "
            "outage can't poison the comparison."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit DEBUG-level logging from the harness.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Settings patching
# ---------------------------------------------------------------------------


def _patch_settings_for_baseline(
    settings: WorkerSettings,
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> WorkerSettings:
    """Return a copy of ``settings`` with baseline retriever knobs applied.

    The process-global ``get_settings()`` cache is *not* mutated — the
    returned object is only used by this run's call to
    ``_get_shared_retriever_bundle``. Production worker invocations
    continue to honour the env defaults (``rag_top_k=5``,
    ``rag_reranker='off'``, etc.).
    """
    retriever_cfg = (
        selected.get("retriever") or manifest.get("retrieverConfig") or {}
    )
    reranker_cfg = (
        selected.get("reranker") or manifest.get("rerankerConfig") or {}
    )
    selected_mode = manifest.get("selectedMode") or {}

    final_top_k = int(
        retriever_cfg.get("finalTopK")
        or selected_mode.get("finalTopK")
        or settings.rag_top_k
    )
    candidate_k = int(
        retriever_cfg.get("candidateK")
        or retriever_cfg.get("denseTopN")
        or selected_mode.get("denseTopN")
        or final_top_k
    )

    update: Dict[str, Any] = {
        "rag_top_k": final_top_k,
        "rag_candidate_k": candidate_k,
        "rag_use_mmr": bool(retriever_cfg.get("useMmr", False)),
        "rag_mmr_lambda": float(retriever_cfg.get("mmrLambda") or 0.7),
    }

    embed_model = retriever_cfg.get("embeddingModel") or manifest.get("embeddingModel")
    if embed_model:
        update["rag_embedding_model"] = embed_model

    max_seq = retriever_cfg.get("embeddingMaxSeqLength")
    if max_seq is not None:
        update["rag_embedding_max_seq_length"] = int(max_seq)

    embed_batch = retriever_cfg.get("embeddingBatchSize")
    if embed_batch is not None:
        update["rag_embedding_batch_size"] = int(embed_batch)

    index_dir = retriever_cfg.get("indexDir")
    if index_dir:
        update["rag_index_dir"] = index_dir

    rerank_name = (reranker_cfg.get("name") or "").strip().lower()
    if rerank_name:
        update["rag_reranker"] = rerank_name
    rerank_batch = reranker_cfg.get("batchSize")
    if rerank_batch is not None:
        update["rag_rerank_batch"] = int(rerank_batch)

    # Pin the query parser to the value used by the baseline (manifest
    # doesn't record it explicitly; baseline used the registry default
    # ``off`` because the LLM backend was noop). Force noop here so
    # multi-query RRF stays dead and matches the baseline retrieval
    # exactly.
    update["rag_query_parser"] = "off"
    # Pin the LLM backend to noop so the rewriter / critic / parser
    # never reach for Ollama / Claude during the offline A/B run. The
    # graph backend's rewrite path will still execute; NoOpQueryRewriter
    # just hands back the parser's view of the original query.
    update["llm_backend"] = "noop"

    return settings.model_copy(update=update)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_query_path(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    if args.queries:
        return args.queries.resolve()
    raw = manifest.get("querySetPath") or ""
    # Manifest stores backslash-separated paths on Windows. Normalise so
    # the path is portable and matches the working tree on either OS.
    p = Path(str(raw).replace("\\", "/"))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _resolve_corpus_path(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Path]:
    if args.corpus:
        return args.corpus.resolve()
    raw = (manifest.get("corpusInfo") or {}).get("path")
    if not raw:
        return None
    p = Path(str(raw).replace("\\", "/"))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _build_offline_retriever(
    *,
    settings: WorkerSettings,
    corpus_path: Path,
    cache_root: Path,
    use_cache: bool,
) -> Tuple[Any, Any, Any, str]:
    """Build an in-memory FAISS retriever from the baseline corpus.

    Same retrieval shape as ``eval/run_eval.py phase2a-latency-sweep``,
    but with an opt-in disk cache: when ``use_cache=True`` (the default
    for the baseline A/B), the FAISS index + chunk metadata are
    persisted under ``cache_root/<key>/`` keyed on (corpus path + size
    + mtime + embedding model + max_seq_length), so the second run on
    the same corpus skips the 30-60-minute embedding step and just
    loads the index off disk.

    The reranker is constructed inline from the baseline reranker
    config; on init failure we fall back to ``NoOpReranker`` and log a
    warning — the run still completes but loses cross-encoder signal.

    Returns ``(retriever, generator, corpus_info, cache_status)`` where
    ``cache_status`` is one of ``"build"``, ``"hit"``, or ``"disabled"``
    so the caller can surface the cache-hit signal in the run metadata.
    """
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=resolve_max_seq_length(settings.rag_embedding_max_seq_length),
        batch_size=int(settings.rag_embedding_batch_size),
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )

    reranker = _build_reranker_or_noop(settings)

    if use_cache:
        cache_dir = _resolve_cache_dir(
            corpus_path=corpus_path,
            cache_root=cache_root,
            embedding_model=settings.rag_embedding_model,
            max_seq_length=int(settings.rag_embedding_max_seq_length),
        )
        if _cache_is_valid(cache_dir):
            try:
                retriever, generator, info = _load_offline_stack_from_cache(
                    cache_dir=cache_dir,
                    embedder=embedder,
                    reranker=reranker,
                    top_k=settings.rag_top_k,
                    candidate_k=settings.rag_candidate_k,
                )
                log.info(
                    "Loaded cached offline FAISS index from %s "
                    "(skipping corpus embedding — saved ~30-60 min).",
                    cache_dir,
                )
                return retriever, generator, info, "hit"
            except Exception as ex:
                log.warning(
                    "Cached offline index at %s could not be loaded "
                    "(%s: %s); rebuilding.",
                    cache_dir, type(ex).__name__, ex,
                )
        log.info("No cached index for this corpus/model — building at %s", cache_dir)
        retriever, generator, info = _build_and_persist_offline_stack(
            corpus_path=corpus_path,
            embedder=embedder,
            reranker=reranker,
            top_k=settings.rag_top_k,
            candidate_k=settings.rag_candidate_k,
            cache_dir=cache_dir,
        )
        return retriever, generator, info, "build"

    # Cache disabled: behave like phase2a-latency-sweep — fresh tmp dir
    # every run, cleaned up implicitly when the OS reclaims tempdir.
    import tempfile
    from eval.harness.offline_corpus import build_offline_rag_stack

    tmp_dir = Path(tempfile.mkdtemp(prefix="agent-loop-ab-baseline-"))
    log.info("Cache disabled — building offline FAISS index in %s", tmp_dir)
    retriever, generator, info = build_offline_rag_stack(
        corpus_path=corpus_path,
        embedder=embedder,
        index_dir=tmp_dir,
        top_k=settings.rag_top_k,
        reranker=reranker,
        candidate_k=settings.rag_candidate_k,
    )
    return retriever, generator, info, "disabled"


def _build_reranker_or_noop(settings: WorkerSettings) -> Optional[RerankerProvider]:
    rerank_name = (settings.rag_reranker or "off").strip().lower()
    if rerank_name not in ("cross_encoder", "cross-encoder", "ce"):
        return None
    try:
        reranker: RerankerProvider = CrossEncoderReranker(
            batch_size=settings.rag_rerank_batch,
        )
        log.info(
            "Reranker active: %s batch_size=%d",
            reranker.name, settings.rag_rerank_batch,
        )
        return reranker
    except Exception as ex:
        log.warning(
            "CrossEncoderReranker init failed (%s: %s). Falling back "
            "to NoOpReranker — A/B comparison loses cross-encoder "
            "signal but still runs.",
            type(ex).__name__, ex,
        )
        return NoOpReranker()


def _resolve_cache_dir(
    *,
    corpus_path: Path,
    cache_root: Path,
    embedding_model: str,
    max_seq_length: int,
) -> Path:
    """Stable per-(corpus, model, max_seq) cache directory.

    Includes the corpus file ``size`` + ``mtime_ns`` in the key so a
    silent corpus refresh (chunker rebuild, manual edit) invalidates
    the cache automatically — failing to detect that would corrupt
    retrieval comparisons.
    """
    stat = corpus_path.stat()
    payload = "|".join([
        str(corpus_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        embedding_model,
        str(max_seq_length),
    ]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    safe_model = embedding_model.replace("/", "_")
    return cache_root / f"{safe_model}-mseq{max_seq_length}-{digest}"


def _cache_is_valid(cache_dir: Path) -> bool:
    return (
        cache_dir.exists()
        and (cache_dir / "build.json").exists()
        and (cache_dir / "faiss.index").exists()
        and (cache_dir / "chunks.jsonl").exists()
    )


def _load_offline_stack_from_cache(
    *,
    cache_dir: Path,
    embedder: Any,
    reranker: Optional[RerankerProvider],
    top_k: int,
    candidate_k: int,
) -> Tuple[Any, Any, Any]:
    """Reconstitute the offline retrieval stack from a cached directory.

    Loads the FAISS index in place + replays ``chunks.jsonl`` into a
    fresh ``_InMemoryMetadataStore`` so the Retriever surface is
    identical to a freshly-built one. ``OfflineCorpusInfo`` is
    reconstructed from ``build.json`` and the chunk-line count.
    """
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.generation import ExtractiveGenerator
    from app.capabilities.rag.metadata_store import ChunkLookupResult
    from app.capabilities.rag.retriever import Retriever
    from eval.harness.offline_corpus import OfflineCorpusInfo, _InMemoryMetadataStore

    index = FaissIndex(cache_dir)
    rows: List[ChunkLookupResult] = []
    doc_ids: set[str] = set()
    with (cache_dir / "chunks.jsonl").open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                ChunkLookupResult(
                    chunk_id=obj["chunk_id"],
                    doc_id=obj["doc_id"],
                    section=obj.get("section") or "",
                    text=obj["text"],
                    faiss_row_id=int(obj["faiss_row_id"]),
                )
            )
            doc_ids.add(obj["doc_id"])

    build = json.loads((cache_dir / "build.json").read_text(encoding="utf-8"))
    info = OfflineCorpusInfo(
        corpus_path=str(build.get("corpus_path") or ""),
        document_count=len(doc_ids),
        chunk_count=len(rows),
        index_version=build["index_version"],
        embedding_model=build["embedding_model"],
        dimension=int(build["dimension"]),
    )
    store = _InMemoryMetadataStore(info.index_version, rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=top_k,
        reranker=reranker,
        candidate_k=candidate_k,
    )
    retriever.ensure_ready()
    return retriever, ExtractiveGenerator(), info


def _build_and_persist_offline_stack(
    *,
    corpus_path: Path,
    embedder: Any,
    reranker: Optional[RerankerProvider],
    top_k: int,
    candidate_k: int,
    cache_dir: Path,
) -> Tuple[Any, Any, Any]:
    """Build the offline stack normally, then snapshot it into the cache.

    Calls ``build_offline_rag_stack`` with ``cache_dir`` as the index
    directory so FAISS already writes ``faiss.index`` + ``build.json``
    in place. The chunker output that ``build_offline_rag_stack``
    keeps in-memory is re-derived here and saved as ``chunks.jsonl``
    so the next run can rehydrate the metadata store without re-running
    the chunker — the chunker is much cheaper than embedding but
    keeping the chunk list pinned protects against chunker code drift
    in between runs.
    """
    from app.capabilities.rag.ingest import (
        _chunks_from_section,
        _iter_documents,
        _stable_chunk_id,
    )
    from eval.harness.offline_corpus import build_offline_rag_stack

    cache_dir.mkdir(parents=True, exist_ok=True)

    retriever, generator, info = build_offline_rag_stack(
        corpus_path=corpus_path,
        embedder=embedder,
        index_dir=cache_dir,
        top_k=top_k,
        reranker=reranker,
        candidate_k=candidate_k,
    )

    chunks_path = cache_dir / "chunks.jsonl"
    faiss_row_id = 0
    with chunks_path.open("w", encoding="utf-8") as fp:
        for raw in _iter_documents(corpus_path):
            doc_id = str(raw.get("doc_id") or raw.get("seed") or raw.get("title") or "").strip()
            if not doc_id:
                continue
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
                    chunk_id = _stable_chunk_id(doc_id, section_name, order, chunk_text)
                    fp.write(json.dumps({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "section": section_name,
                        "text": chunk_text,
                        "faiss_row_id": faiss_row_id,
                    }, ensure_ascii=False) + "\n")
                    faiss_row_id += 1

    # Stash the corpus path inside build.json so the cache loader can
    # surface it back through OfflineCorpusInfo. FaissIndex.build()
    # writes build.json with index_version + embedding_model + dim;
    # we annotate it without rewriting the FAISS bytes.
    build_path = cache_dir / "build.json"
    try:
        build = json.loads(build_path.read_text(encoding="utf-8"))
    except Exception:
        build = {}
    build["corpus_path"] = str(corpus_path)
    build["chunk_count_cached"] = faiss_row_id
    build_path.write_text(
        json.dumps(build, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info(
        "Persisted offline index cache to %s (index_version=%s "
        "chunk_count=%d)",
        cache_dir, info.index_version, info.chunk_count,
    )
    return retriever, generator, info


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(data).__name__}")
    return data


def _default_run_name() -> str:
    return "legacy-vs-graph-" + dt.datetime.now().strftime("%Y%m%d-%H%M")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


# ---------------------------------------------------------------------------
# Comparison report renderer
# ---------------------------------------------------------------------------


def _render_report(
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[AgentLoopABComparisonRow],
) -> str:
    """Render the comparison_report.md the spec asks for.

    Sections are kept in the spec's order so a reader can scan top-down:
    quality (hit@k / MRR / keyword-hit), latency, iteration / rewrite /
    retrieval cost, verdict counts with representative examples for
    each side, and the recommendation block keyed off the harness's
    aggregate verdict.
    """
    lines: List[str] = []
    run_name = metadata.get("runName") or "legacy-vs-graph"
    lines.append(f"# Legacy vs Graph A/B comparison — {run_name}")
    lines.append("")
    lines.append(_render_metadata_block(metadata))
    lines.append("")
    lines.append("## Quality (legacy vs graph)")
    lines.append(_render_quality_table(summary))
    lines.append("")
    lines.append("## Latency")
    lines.append(_render_latency_table(summary))
    lines.append("")
    lines.append("## Iteration / rewrite / retrieval cost")
    lines.append(_render_cost_table(summary))
    lines.append("")
    lines.append("## Verdicts")
    lines.append(_render_verdicts(summary))
    lines.append("")
    lines.append("### Representative graph wins")
    lines.append(_render_examples(rows, VERDICT_GRAPH_WIN, kind="graph_win"))
    lines.append("")
    lines.append("### Representative legacy wins")
    lines.append(_render_examples(rows, VERDICT_LEGACY_WIN, kind="legacy_win"))
    lines.append("")
    lines.append("### Graph regressions")
    lines.append(_render_examples(rows, VERDICT_REGRESSION, kind="regression"))
    lines.append("")
    lines.append("## Recommendation")
    lines.append(_render_recommendation(summary))
    lines.append("")
    return "\n".join(lines)


def _render_metadata_block(metadata: Mapping[str, Any]) -> str:
    rt = metadata.get("retriever") or {}
    rr = metadata.get("reranker") or {}
    al = metadata.get("agentLoop") or {}
    rows = [
        ("baseline", metadata.get("baselineDir")),
        ("baseline commit", metadata.get("baselineCommit")),
        ("query set", metadata.get("querySetPath")),
        ("query count", metadata.get("queryCount")),
        ("index version", metadata.get("indexVersion")),
        ("embedding model", metadata.get("embeddingModel")),
        ("retriever", _format_dict(rt)),
        ("reranker", _format_dict(rr)),
        ("agent loop", _format_dict(al)),
        ("evaluated at", metadata.get("evaluatedAt")),
        ("elapsed ms", metadata.get("elapsedMs")),
    ]
    body = ["| Field | Value |", "| --- | --- |"]
    for k, v in rows:
        body.append(f"| {k} | {_md_cell(v)} |")
    return "\n".join(body)


def _format_dict(d: Mapping[str, Any]) -> str:
    if not d:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _render_quality_table(summary: Mapping[str, Any]) -> str:
    metric_rows: List[Tuple[str, Any, Any, Any]] = []
    for label, key in (
        ("hit@1", "HitAt1"),
        ("hit@3", "HitAt3"),
        ("hit@5", "HitAt5"),
        ("MRR (across rows with expected_doc_id)", "MRR"),
        ("keyword hit (across rows with expected_keywords)", "KeywordHitRate"),
        ("success rate", "SuccessRate"),
    ):
        legacy = summary.get(f"legacy{key}")
        graph = summary.get(f"graph{key}")
        delta = _delta(legacy, graph)
        metric_rows.append((label, legacy, graph, delta))

    body = [
        "| Metric | Legacy | Graph | Δ (graph − legacy) |",
        "| --- | --- | --- | --- |",
    ]
    for label, legacy, graph, delta in metric_rows:
        body.append(
            f"| {label} | {_fmt(legacy)} | {_fmt(graph)} | {_fmt(delta, signed=True)} |"
        )

    expected_count = summary.get("expectedDocRowCount")
    keyword_count = summary.get("expectedKeywordRowCount")
    notes: List[str] = []
    if expected_count is not None:
        notes.append(f"hit@k / MRR computed over {expected_count} rows with `expected_doc_id`.")
    if keyword_count is not None:
        notes.append(f"keyword hit computed over {keyword_count} rows with `expected_keywords`.")
    if notes:
        body.append("")
        body.append("> " + " ".join(notes))
    return "\n".join(body)


def _render_latency_table(summary: Mapping[str, Any]) -> str:
    rows = [
        ("p50 latency (ms)", "LatencyP50"),
        ("p95 latency (ms)", "LatencyP95"),
    ]
    body = [
        "| Metric | Legacy | Graph | Δ (graph − legacy) | Ratio (graph / legacy) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, key in rows:
        legacy = summary.get(f"legacy{key}")
        graph = summary.get(f"graph{key}")
        delta = _delta(legacy, graph)
        ratio = _ratio(legacy, graph)
        body.append(
            f"| {label} | {_fmt(legacy)} | {_fmt(graph)} | {_fmt(delta, signed=True)} | {_fmt(ratio)} |"
        )
    return "\n".join(body)


def _render_cost_table(summary: Mapping[str, Any]) -> str:
    rows = [
        ("avg loop iterations", "AvgIterations"),
        ("avg rewrite count", "AvgRewriteCount"),
        ("avg retrieval calls", "AvgRetrievalCalls"),
        ("avg LLM calls", "AvgLlmCalls"),
    ]
    body = [
        "| Metric | Legacy | Graph | Δ (graph − legacy) |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in rows:
        legacy = summary.get(f"legacy{key}")
        graph = summary.get(f"graph{key}")
        delta = _delta(legacy, graph)
        body.append(
            f"| {label} | {_fmt(legacy)} | {_fmt(graph)} | {_fmt(delta, signed=True)} |"
        )
    return "\n".join(body)


def _render_verdicts(summary: Mapping[str, Any]) -> str:
    counts = [
        ("graph wins", summary.get("graphWins", 0)),
        ("legacy wins", summary.get("legacyWins", 0)),
        ("ties", summary.get("ties", 0)),
        ("regressions", summary.get("regressions", 0)),
    ]
    body = ["| Verdict | Count |", "| --- | --- |"]
    for label, n in counts:
        body.append(f"| {label} | {int(n)} |")
    return "\n".join(body)


def _render_examples(
    rows: Sequence[AgentLoopABComparisonRow],
    verdict: str,
    *,
    kind: str,
    limit: int = 5,
) -> str:
    matching = [r for r in rows if r.verdict == verdict]
    if not matching:
        return "_(none)_"
    matching = matching[:limit]
    body = [
        "| query_id | query | legacy | graph | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in matching:
        notes = _example_notes(r, kind=kind)
        body.append(
            f"| {_md_cell(r.query_id)} | {_md_cell(_clip(r.query, 80))} | "
            f"{_one_liner(r.legacy)} | {_one_liner(r.graph)} | {_md_cell(notes)} |"
        )
    return "\n".join(body)


def _one_liner(metrics: Mapping[str, Any]) -> str:
    """Compact per-backend metric summary used inside the example tables."""
    bits = [
        f"hit@1={_bool_flag(metrics.get('expected_doc_hit_at_1'))}",
        f"hit@3={_bool_flag(metrics.get('expected_doc_hit_at_3'))}",
        f"hit@5={_bool_flag(metrics.get('expected_doc_hit_at_5'))}",
        f"mrr={_fmt(metrics.get('mrr_contribution'))}",
        f"latMs={_fmt(metrics.get('total_latency_ms'))}",
        f"iters={metrics.get('loop_iterations')}",
    ]
    if metrics.get("error_code"):
        bits.append(f"err={metrics.get('error_code')}")
    if metrics.get("success") is False:
        bits.append("success=false")
    return " · ".join(bits)


def _example_notes(row: AgentLoopABComparisonRow, *, kind: str) -> str:
    legacy = row.legacy
    graph = row.graph
    if kind == "graph_win":
        if not legacy.get("success") and graph.get("success"):
            return "graph recovered from legacy failure"
        deltas = []
        for k_label, k in (
            ("hit@1", "expected_doc_hit_at_1"),
            ("hit@3", "expected_doc_hit_at_3"),
            ("hit@5", "expected_doc_hit_at_5"),
        ):
            if graph.get(k) and not legacy.get(k):
                deltas.append(f"graph gained {k_label}")
        gm = graph.get("mrr_contribution") or 0.0
        lm = legacy.get("mrr_contribution") or 0.0
        if gm > lm + 1e-6:
            deltas.append(f"MRR +{gm - lm:.4f}")
        if not deltas and graph.get("expected_keyword_hit") and not legacy.get("expected_keyword_hit"):
            deltas.append("graph hit keyword")
        return "; ".join(deltas) or "graph improved without doc/keyword swap"
    if kind == "legacy_win":
        legacy_lat = legacy.get("total_latency_ms") or 0.0
        graph_lat = graph.get("total_latency_ms") or 0.0
        ratio = (graph_lat / legacy_lat) if legacy_lat else float("nan")
        return f"graph spent {_fmt(graph_lat)} ms vs legacy {_fmt(legacy_lat)} ms ({_fmt(ratio)}x)"
    if kind == "regression":
        if not graph.get("success") and legacy.get("success"):
            return f"graph errored: {graph.get('error_code') or 'unknown'}"
        deltas = []
        for k_label, k in (
            ("hit@1", "expected_doc_hit_at_1"),
            ("hit@3", "expected_doc_hit_at_3"),
            ("hit@5", "expected_doc_hit_at_5"),
        ):
            if legacy.get(k) and not graph.get(k):
                deltas.append(f"graph lost {k_label}")
        if not deltas:
            gm = graph.get("mrr_contribution") or 0.0
            lm = legacy.get("mrr_contribution") or 0.0
            if lm > gm + 1e-6:
                deltas.append(f"MRR −{lm - gm:.4f}")
        return "; ".join(deltas) or "graph latency blew up without quality gain"
    return ""


def _render_recommendation(summary: Mapping[str, Any]) -> str:
    label = (summary.get("recommendation") or "").strip()
    legacy_p95 = summary.get("legacyLatencyP95") or 0.0
    graph_p95 = summary.get("graphLatencyP95") or 0.0
    ratio = (graph_p95 / legacy_p95) if legacy_p95 > 0 else float("nan")
    quality_lift = has_quality_lift(summary)
    regressions = int(summary.get("regressions", 0) or 0)
    legacy_avg_llm = summary.get("legacyAvgLlmCalls") or 0.0
    graph_avg_llm = summary.get("graphAvgLlmCalls") or 0.0
    legacy_avg_retr = summary.get("legacyAvgRetrievalCalls") or 0.0
    graph_avg_retr = summary.get("graphAvgRetrievalCalls") or 0.0
    legacy_succ = summary.get("legacySuccessRate")
    graph_succ = summary.get("graphSuccessRate")

    bullets = [
        f"- harness recommendation: **{label or 'unknown'}**",
        (
            f"- p95 latency ratio (graph / legacy): {_fmt(ratio)} "
            f"(threshold ≤ {LATENCY_RATIO_ADOPT_CEILING})"
        ),
        f"- regressions: {regressions} (threshold = 0)",
        (
            "- quality lift on hit@k / MRR / keyword: "
            + ("yes" if quality_lift else "no")
        ),
        (
            f"- success rate: legacy={_fmt(legacy_succ)} graph={_fmt(graph_succ)}"
        ),
        (
            f"- avg LLM calls: legacy={_fmt(legacy_avg_llm)} "
            f"graph={_fmt(graph_avg_llm)} (Δ={_fmt(graph_avg_llm - legacy_avg_llm, signed=True)})"
        ),
        (
            f"- avg retrieval calls: legacy={_fmt(legacy_avg_retr)} "
            f"graph={_fmt(graph_avg_retr)} (Δ={_fmt(graph_avg_retr - legacy_avg_retr, signed=True)})"
        ),
    ]

    if label == RECOMMENDATION_ADOPT:
        bullets.append(
            "- decision: **adopt graph backend as candidate.** Quality lifts "
            "without breaking the latency/regression bars; graph backend "
            "remains experimental until promoted by a follow-up config "
            "change. agent_loop_backend default stays `legacy`."
        )
    elif label == RECOMMENDATION_HOLD_NO_QUALITY_GAIN:
        bullets.append(
            "- decision: **hold — no quality gain.** Graph adds LLM/retrieval "
            "cost without measurable improvement. Keep `agent_loop_backend=legacy`."
        )
    elif label == RECOMMENDATION_HOLD_EXPERIMENTAL:
        bullets.append(
            "- decision: **hold — experimental backend only.** Trace / "
            "debuggability or cold-start latency may improve, but retrieval "
            "quality (hit@k / MRR / keyword) is flat. Keep "
            "`agent_loop_backend=legacy`; revisit if a graph-only feature "
            "(branching plans, parallel retrieves) lands."
        )
    elif label == RECOMMENDATION_HOLD_REVIEW_REGRESSIONS:
        bullets.append(
            "- decision: **hold — investigate regressions.** Graph either "
            "regressed on at least one query or its quality lift came at a "
            "p95 latency cost that exceeds the adoption ceiling. Triage the "
            "regression cases above before any promotion."
        )
    elif label == RECOMMENDATION_HOLD_REVIEW_ERRORS:
        bullets.append(
            "- decision: **hold — graph error rate is higher than legacy.** "
            "Inspect graph_build_failed / graph_invoke_failed rows in "
            "raw_results.jsonl before promoting."
        )
    else:
        bullets.append(
            "- decision: **hold — see verdict counters and regression "
            "examples above.** Default `agent_loop_backend=legacy` "
            "remains in effect."
        )

    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Tiny formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _md_cell(value)
    if f != f:  # NaN
        return "—"
    if abs(f) >= 100:
        formatted = f"{f:.1f}"
    elif abs(f) >= 1:
        formatted = f"{f:.3f}"
    else:
        formatted = f"{f:.4f}"
    if signed and f >= 0 and not formatted.startswith("+"):
        formatted = "+" + formatted
    return formatted


def _delta(legacy: Any, graph: Any) -> Optional[float]:
    try:
        if legacy is None or graph is None:
            return None
        return float(graph) - float(legacy)
    except (TypeError, ValueError):
        return None


def _ratio(legacy: Any, graph: Any) -> Optional[float]:
    try:
        if legacy is None or graph is None:
            return None
        legacy_f = float(legacy)
        graph_f = float(graph)
        if legacy_f == 0.0:
            return None
        return graph_f / legacy_f
    except (TypeError, ValueError):
        return None


def _bool_flag(value: Any) -> str:
    if value is None:
        return "—"
    return "Y" if bool(value) else "N"


def _clip(text: Any, length: int) -> str:
    s = str(text or "")
    if len(s) <= length:
        return s
    return s[: max(0, length - 1)] + "…"


def _md_cell(value: Any) -> str:
    if value is None:
        return "—"
    s = str(value)
    # Markdown table cells can't have raw pipes / newlines.
    return s.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())
