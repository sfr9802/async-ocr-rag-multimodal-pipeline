"""Offline ``evaluate(params)`` for the optuna-round-refinement skill.

Sister to ``tune_eval.py`` — that module builds the production retriever
bundle (psycopg2 + ragmeta + ingest pipeline) which is not available
in offline / CI envs. This module reuses the same cached FAISS index
the wide-MMR-titlecap sweep operates against:

    eval/agent_loop_ab/_indexes/BAAI_bge-m3-mseq1024-30fc1cc1cd8c319a/

…and the silver-200 query set, so an Optuna trial directly probes the
*same* search space the wide-MMR sweep just diagnosed: candidate_k,
MMR lambda, title cap, rerank_in, final_top_k.

Tunable params accepted via ``params: dict``. Unknown keys are ignored
so the same evaluator handles iterative round refinements without
needing a dispatch table.

Param contract (skill maps these to env vars in the production case;
here we read them out of the dict directly):

    candidate_k          : int  in [50, 200]
    final_top_k          : int  in [5, 10]
    rerank_in            : int  in [16, 40]
    use_mmr              : bool / "true"|"false"
    mmr_lambda           : float in [0.5, 0.8]
    mmr_k                : int  in [32, 96]
    title_cap_rerank_input : int|null in [1, 4]
    title_cap_final      : int|null in [1, 4]

Returned shape matches the round-runner's expected
``{primary: float, secondary: dict}``.

Performance posture:
  * Embedder + reranker + FAISS load happen ONCE at import (module-
    level globals) — Optuna trials reuse the same instances. No model
    reload per trial. This is the critical difference from the
    production ``tune_eval``: each trial here is just an adapter
    construction + 50 query evaluations.
  * Default subsample: first ``OFFLINE_TUNE_QUERY_LIMIT`` (=50) rows
    of ``anime_silver_200.jsonl``. Override via the env var.
  * Single-trial wall-clock on a warm RTX 5080: ~60s for 50 rows.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Module-level lazy bundle (built once across all Optuna trials).
# --------------------------------------------------------------------------


_DATASET_PATH = Path(
    os.environ.get(
        "OFFLINE_TUNE_DATASET",
        "eval/eval_queries/anime_silver_200.jsonl",
    )
)
_CORPUS_PATH = Path(
    os.environ.get(
        "OFFLINE_TUNE_CORPUS",
        "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl",
    )
)
_CACHE_DIR = Path(
    os.environ.get(
        "OFFLINE_TUNE_CACHE_DIR",
        "eval/agent_loop_ab/_indexes/BAAI_bge-m3-mseq1024-30fc1cc1cd8c319a",
    )
)
_QUERY_LIMIT = int(os.environ.get("OFFLINE_TUNE_QUERY_LIMIT", "50"))


_RETRIEVER = None  # type: Any
_RERANKER = None  # type: Any
_TITLE_PROVIDER = None  # type: Any
_DATASET = None  # type: Any
_BUNDLE_BUILD_LOCK = False


def _build_offline_bundle():
    """One-shot construction of the offline RAG stack.

    Subsequent calls reuse the module globals — Optuna trials only
    pay the construction cost once per python process.
    """
    global _RETRIEVER, _RERANKER, _TITLE_PROVIDER, _DATASET, _BUNDLE_BUILD_LOCK
    if _RETRIEVER is not None:
        return
    if _BUNDLE_BUILD_LOCK:
        raise RuntimeError(
            "tune_eval_offline bundle build re-entered before completion"
        )
    _BUNDLE_BUILD_LOCK = True

    try:
        from app.capabilities.rag.embeddings import (
            SentenceTransformerEmbedder,
        )
        from app.capabilities.rag.faiss_index import FaissIndex
        from app.capabilities.rag.metadata_store import ChunkLookupResult
        from app.capabilities.rag.reranker import CrossEncoderReranker
        from app.capabilities.rag.retriever import Retriever
        from app.core.config import get_settings
        from eval.harness.io_utils import load_jsonl
        from eval.harness.offline_corpus import _InMemoryMetadataStore
        from eval.harness.wide_retrieval_helpers import DocTitleResolver

        for required in ("faiss.index", "build.json", "chunks.jsonl"):
            if not (_CACHE_DIR / required).exists():
                raise FileNotFoundError(
                    f"Cache dir {_CACHE_DIR} missing {required}; run "
                    "scripts.eval_full_silver_minimal_sweep first."
                )

        settings = get_settings()
        embedder = SentenceTransformerEmbedder(
            model_name=settings.rag_embedding_model,
            query_prefix=settings.rag_embedding_prefix_query,
            passage_prefix=settings.rag_embedding_prefix_passage,
            max_seq_length=int(
                os.environ.get("OFFLINE_TUNE_MAX_SEQ", "1024")
            ),
            batch_size=int(
                os.environ.get("OFFLINE_TUNE_EMBED_BATCH", "32")
            ),
            show_progress_bar=False,
            cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
        )
        reranker = CrossEncoderReranker(
            model_name=os.environ.get(
                "OFFLINE_TUNE_RERANKER",
                "BAAI/bge-reranker-v2-m3",
            ),
            max_length=int(
                os.environ.get("OFFLINE_TUNE_RERANKER_MAX", "512")
            ),
            batch_size=int(
                os.environ.get("OFFLINE_TUNE_RERANKER_BATCH", "16")
            ),
            text_max_chars=int(
                os.environ.get("OFFLINE_TUNE_RERANKER_TEXT_MAX", "800")
            ),
            device=os.environ.get("OFFLINE_TUNE_RERANKER_DEVICE") or None,
            collect_stage_timings=False,
        )

        index = FaissIndex(_CACHE_DIR)
        info = index.load()
        rows: List[ChunkLookupResult] = []
        with (_CACHE_DIR / "chunks.jsonl").open("r", encoding="utf-8") as fp:
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

        title_resolver = DocTitleResolver.from_corpus(_CORPUS_PATH)
        title_provider = title_resolver.title_provider()

        dataset = list(load_jsonl(_DATASET_PATH))
        if _QUERY_LIMIT > 0:
            dataset = dataset[:_QUERY_LIMIT]
        log.info(
            "Built offline tune bundle: chunks=%d, queries=%d, cache=%s",
            len(rows), len(dataset), _CACHE_DIR,
        )

        _RETRIEVER = retriever
        _RERANKER = reranker
        _TITLE_PROVIDER = title_provider
        _DATASET = dataset
    finally:
        _BUNDLE_BUILD_LOCK = False


# --------------------------------------------------------------------------
# Param coercion
# --------------------------------------------------------------------------


def _bool_param(raw: Any, default: bool = True) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def _int_or_none(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("none", "null", ""):
            return None
        try:
            return int(s)
        except ValueError:
            return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Public evaluator
# --------------------------------------------------------------------------


def _resolve_primary_metric() -> str:
    """Re-read on every call so a round config can override per-run.

    The round_runner doesn't pass the config's ``objective_name`` into
    evaluate(); the convention is that evaluate returns ``primary`` and
    the runner uses that as the optimization target. To make a single
    evaluate function support multiple round-level objectives we look
    at OFFLINE_TUNE_PRIMARY at call-time, defaulting to mean_hit_at_5.
    """
    return os.environ.get("OFFLINE_TUNE_PRIMARY", "mean_hit_at_5")


def evaluate(params: dict) -> dict:
    """Run a wide-MMR-titlecap eval on the offline silver subset.

    Returns ``{primary, secondary}``. ``primary`` is the metric named
    by ``OFFLINE_TUNE_PRIMARY`` (default ``mean_hit_at_5``). ``secondary``
    is a small grab-bag of correlated metrics so the round bundle has
    enough breadcrumbs for the analyst to drift-check.

    Errors are caught and surfaced as ``primary=-inf`` with the
    exception class name in ``secondary["error"]`` — matches the
    contract the round runner expects (no exception propagation).
    """
    try:
        _build_offline_bundle()
        from eval.harness.retrieval_eval import (
            DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS, run_retrieval_eval,
        )
        from eval.harness.wide_retrieval_adapter import (
            WideRetrievalConfig, WideRetrievalEvalAdapter,
        )

        cfg = WideRetrievalConfig(
            candidate_k=int(params.get("candidate_k", 200)),
            final_top_k=int(params.get("final_top_k", 8)),
            rerank_in=int(params.get("rerank_in", 32)),
            use_mmr=_bool_param(params.get("use_mmr", True)),
            mmr_lambda=float(params.get("mmr_lambda", 0.65)),
            mmr_k=int(params.get("mmr_k", 64)),
            title_cap_rerank_input=_int_or_none(
                params.get("title_cap_rerank_input")
            ),
            title_cap_final=_int_or_none(
                params.get("title_cap_final")
            ),
        )
        adapter = WideRetrievalEvalAdapter(
            _RETRIEVER,
            config=cfg,
            final_reranker=_RERANKER,
            title_provider=_TITLE_PROVIDER,
            name="optuna-offline",
        )
        candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))
        t0 = time.perf_counter()
        summary, _, _, _ = run_retrieval_eval(
            list(_DATASET),
            retriever=adapter,
            top_k=cfg.final_top_k,
            mrr_k=10,
            ndcg_k=10,
            candidate_ks=candidate_ks,
            diversity_ks=DEFAULT_DIVERSITY_KS,
            dataset_path=str(_DATASET_PATH),
            corpus_path=str(_CORPUS_PATH),
        )
        elapsed = round(time.perf_counter() - t0, 3)

        primary_metric = _resolve_primary_metric()
        primary_value = getattr(summary, primary_metric, None)
        if primary_value is None:
            return {
                "primary": float("-inf"),
                "secondary": {
                    "error": "primary_metric_undefined",
                    "metric": primary_metric,
                },
            }

        cand_hits = summary.candidate_hit_rates or {}
        dup_ratios = summary.duplicate_doc_ratios or {}
        unique_counts = summary.unique_doc_counts or {}
        secondary = {
            "mean_hit_at_1": summary.mean_hit_at_1,
            "mean_hit_at_3": summary.mean_hit_at_3,
            "mean_hit_at_5": summary.mean_hit_at_5,
            "mean_mrr_at_10": summary.mean_mrr_at_10,
            "mean_ndcg_at_10": summary.mean_ndcg_at_10,
            "candidate_hit_at_50": cand_hits.get("50"),
            "candidate_hit_at_100": cand_hits.get("100"),
            "duplicate_doc_ratio_at_10": dup_ratios.get("10"),
            "unique_doc_count_at_10": unique_counts.get("10"),
            "p95_total_retrieval_ms": (
                summary.p95_total_retrieval_ms or summary.p95_retrieval_ms
            ),
            "mean_dense_retrieval_ms": summary.mean_dense_retrieval_ms,
            "mean_rerank_ms": summary.mean_rerank_ms,
            "trial_eval_seconds": elapsed,
            "row_count": summary.row_count,
        }
        return {
            "primary": float(primary_value),
            "secondary": secondary,
        }
    except Exception as exc:  # pragma: no cover — defensive
        log.exception("offline tune evaluate failed: %s", exc)
        return {
            "primary": float("-inf"),
            "secondary": {
                "error": type(exc).__name__,
                "message": str(exc)[:200],
            },
        }
