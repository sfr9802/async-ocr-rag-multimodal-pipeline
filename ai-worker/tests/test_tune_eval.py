"""Tests for `eval.tune_eval.evaluate` — the Optuna round adapter.

Fully offline: the real RAG stack and `run_rag_eval` are monkey-patched
so no FAISS index, no sentence-transformer download, and no live
generator are required. The dataset is loaded once at module import
from the real JSONL fixtures (verifying the comment-line skipping path)
but is not evaluated against.

The tests cover the contract the `optuna-round-refinement` round runner
relies on:

  * evaluate(params) returns a dict with numeric ``primary`` and a
    ``secondary`` dict containing the documented keys.
  * Env vars patched from ``params`` are removed again when the call
    finishes so a prior trial cannot leak into the next.
  * Any exception inside the call is turned into a failed-trial
    sentinel (``primary == -inf`` + ``secondary.error``) so the runner
    doesn't crash the whole round.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from eval import tune_eval
from eval.harness.rag_eval import RagEvalSummary


def _fake_summary(**overrides: Any) -> RagEvalSummary:
    base = dict(
        dataset_path="fake",
        row_count=2,
        rows_with_expected_doc_ids=2,
        rows_with_expected_keywords=0,
        top_k=5,
        mean_hit_at_k=0.5,
        mean_recall_at_k=0.75,
        mrr=0.6,
        mean_keyword_coverage=None,
        mean_dup_rate=0.0,
        mean_topk_gap=0.1,
        mean_retrieval_ms=12.0,
        p50_retrieval_ms=10.0,
        p95_retrieval_ms=30.0,
        max_retrieval_ms=40.0,
        mean_generation_ms=5.0,
        mean_total_ms=15.0,
        error_count=0,
    )
    base.update(overrides)
    return RagEvalSummary(**base)


class _FakeRow:
    def __init__(self, recall: float) -> None:
        self.recall_at_k = recall


def _fake_rows() -> list[_FakeRow]:
    return [_FakeRow(0.8), _FakeRow(0.7)]


def _install_successful_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tune_eval,
        "_build_rag_stack",
        lambda: (object(), object(), 5),
    )
    monkeypatch.setattr(
        tune_eval,
        "run_rag_eval",
        lambda *a, **kw: (_fake_summary(), _fake_rows()),
    )


def test_evaluate_returns_primary_and_expected_secondary_keys(monkeypatch):
    _install_successful_stubs(monkeypatch)

    result = tune_eval.evaluate({"rag_top_k": 5})

    assert isinstance(result, dict)
    assert "primary" in result
    assert "secondary" in result
    assert result["primary"] == pytest.approx(0.75)
    assert isinstance(result["primary"], (int, float))

    secondary = result["secondary"]
    assert isinstance(secondary, dict)

    expected_keys = {
        "mrr",
        "mean_dup_rate",
        "mean_topk_gap",
        "p50_retrieval_ms",
        "p95_retrieval_ms",
        "mean_hit_at_k",
        "mean_keyword_coverage",
        "recall_at_k_anime",
        "recall_at_k_enterprise",
    }
    missing = expected_keys - set(secondary)
    assert not missing, f"secondary is missing keys: {sorted(missing)}"


def test_evaluate_restores_env_after_call(monkeypatch):
    _install_successful_stubs(monkeypatch)

    key_top_k = "AIPIPELINE_WORKER_RAG_TOP_K"
    key_stop_conf = "AIPIPELINE_WORKER_AGENT_MIN_STOP_CONFIDENCE"
    monkeypatch.delenv(key_top_k, raising=False)
    monkeypatch.delenv(key_stop_conf, raising=False)
    assert key_top_k not in os.environ
    assert key_stop_conf not in os.environ

    tune_eval.evaluate({"rag_top_k": 42, "agent_min_stop_conf": 0.7})

    # Both env keys the call would have injected must be gone again so a
    # subsequent trial's WorkerSettings() does not inherit stale values.
    assert key_top_k not in os.environ
    assert key_stop_conf not in os.environ


def test_evaluate_restores_prior_env_value(monkeypatch):
    _install_successful_stubs(monkeypatch)

    key_top_k = "AIPIPELINE_WORKER_RAG_TOP_K"
    monkeypatch.setenv(key_top_k, "9")

    tune_eval.evaluate({"rag_top_k": 3})

    assert os.environ[key_top_k] == "9"


def test_evaluate_exception_path_returns_minus_inf(monkeypatch):
    def _boom() -> Any:
        raise RuntimeError("simulated stack build failure")

    monkeypatch.setattr(tune_eval, "_build_rag_stack", _boom)

    result = tune_eval.evaluate({"rag_top_k": 5})

    assert result["primary"] == float("-inf")
    assert isinstance(result["secondary"], dict)
    assert result["secondary"].get("error") == "RuntimeError"


def test_evaluate_none_primary_becomes_failed_trial(monkeypatch):
    monkeypatch.setattr(
        tune_eval,
        "_build_rag_stack",
        lambda: (object(), object(), 5),
    )
    monkeypatch.setattr(
        tune_eval,
        "run_rag_eval",
        lambda *a, **kw: (_fake_summary(mean_recall_at_k=None), _fake_rows()),
    )

    result = tune_eval.evaluate({"rag_top_k": 5})

    assert result["primary"] == float("-inf")
    assert result["secondary"].get("error") == "mean_recall_at_k_undefined"
