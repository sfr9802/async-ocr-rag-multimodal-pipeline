"""Bridge between optuna-round-refinement and the project's RAG eval harness.

The skill imports `evaluate(params)` via the dotted pointer
`eval.tune_eval:evaluate` and calls it once per Optuna trial. This module
owns nothing Optuna-specific — sampler / pruner / bundle export all live
in the skill package. Here we only:

  * load the combined anime + enterprise KR fixtures once at import,
  * patch AIPIPELINE_WORKER_* env vars from the trial params,
  * build the production retriever bundle from the patched settings,
  * call `run_rag_eval`, then slice the per-row metrics by domain.

The dataset rows carry an internal `_domain` tag; `run_rag_eval` ignores
unknown keys, so it is safe to thread domain metadata through the same
list it consumes.
"""

from __future__ import annotations

import os
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

from eval.harness.io_utils import load_jsonl
from eval.harness.rag_eval import run_rag_eval, summary_to_dict

_DATASET_DIR = Path(__file__).resolve().parent / "datasets"

# Honour the literal name from the spec; fall back to the actual file in
# the repo (rag_anime_extended_kr.jsonl) so tuning is not blocked on a
# rename.
_ANIME_CANDIDATES = (
    _DATASET_DIR / "rag_anime_kr.jsonl",
    _DATASET_DIR / "rag_anime_extended_kr.jsonl",
)
_ENTERPRISE_PATH = _DATASET_DIR / "rag_enterprise_kr.jsonl"

# Skill param name -> WorkerSettings field name where they differ.
_KEY_TO_SETTING: Dict[str, str] = {
    "agent_min_stop_conf": "agent_min_stop_confidence",
}


def _load_combined() -> List[Dict[str, Any]]:
    anime_path = next((p for p in _ANIME_CANDIDATES if p.exists()), None)
    if anime_path is None:
        raise FileNotFoundError(
            "No anime RAG dataset found; tried "
            f"{[str(p) for p in _ANIME_CANDIDATES]}"
        )
    if not _ENTERPRISE_PATH.exists():
        raise FileNotFoundError(
            f"Enterprise RAG dataset not found at {_ENTERPRISE_PATH}"
        )
    rows: List[Dict[str, Any]] = []
    for raw in load_jsonl(anime_path):
        rows.append({**raw, "_domain": "anime"})
    for raw in load_jsonl(_ENTERPRISE_PATH):
        rows.append({**raw, "_domain": "enterprise"})
    return rows


_DATASET = _load_combined()


def _params_to_env(params: Mapping[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for name, value in params.items():
        setting = _KEY_TO_SETTING.get(name, name)
        env_key = f"AIPIPELINE_WORKER_{setting.upper()}"
        if isinstance(value, bool):
            env[env_key] = "true" if value else "false"
        else:
            env[env_key] = str(value)
    return env


@contextmanager
def _patched_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply env overrides for the duration of the with-block.

    Saves the prior value (or absence) of every key it touches and
    restores it on exit, even if the body raises.
    """
    sentinel = object()
    saved: Dict[str, Any] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key, sentinel)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _build_rag_stack():
    """Construct retriever + generator under the currently-patched env.

    `WorkerSettings()` is instantiated fresh to bypass `get_settings()`'s
    process-global singleton. The retriever-bundle cache inside the
    registry is cleared first so a new (settings → bundle) entry is
    actually built — otherwise a prior trial's cached bundle would
    shadow the new env values.
    """
    from app.capabilities import registry as _registry
    from app.core.config import WorkerSettings

    _registry._shared_component_cache.clear()
    settings = WorkerSettings()
    retriever, generator = _registry._get_shared_retriever_bundle(settings)
    return retriever, generator, int(settings.rag_top_k)


def _domain_recall(rows: List[Any], dataset: List[Dict[str, Any]], domain: str) -> float:
    values = [
        r.recall_at_k
        for r, src in zip(rows, dataset)
        if src.get("_domain") == domain and r.recall_at_k is not None
    ]
    return round(statistics.fmean(values), 4) if values else 0.0


def evaluate(params: dict) -> dict:
    try:
        with _patched_env(_params_to_env(params)):
            retriever, generator, top_k = _build_rag_stack()
            summary, rows = run_rag_eval(
                _DATASET,
                retriever=retriever,
                generator=generator,
                top_k=top_k,
                dataset_path="combined_anime_enterprise_kr",
            )
        agg = summary_to_dict(summary)
        primary = agg["mean_recall_at_k"]
        # The round runner rejects a non-numeric primary, so coerce the
        # "no rows had expected_doc_ids" sentinel (None) into a failed-trial
        # marker rather than letting it bubble up as a runner-level crash.
        if primary is None:
            return {
                "primary": float("-inf"),
                "secondary": {"error": "mean_recall_at_k_undefined"},
            }
        return {
            "primary": primary,
            "secondary": {
                "mrr": agg["mrr"],
                "mean_dup_rate": agg["mean_dup_rate"],
                "mean_topk_gap": agg["mean_topk_gap"],
                "p50_retrieval_ms": agg["p50_retrieval_ms"],
                "p95_retrieval_ms": agg["p95_retrieval_ms"],
                "mean_hit_at_k": agg["mean_hit_at_k"],
                "mean_keyword_coverage": agg["mean_keyword_coverage"],
                "recall_at_k_anime": _domain_recall(rows, _DATASET, "anime"),
                "recall_at_k_enterprise": _domain_recall(rows, _DATASET, "enterprise"),
            },
        }
    except Exception as exc:
        return {
            "primary": float("-inf"),
            "secondary": {"error": type(exc).__name__},
        }
