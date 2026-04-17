"""Tests for `scripts.tune` — the Optuna tuning driver.

Four layers, all offline (no eval subprocess actually runs):

  1. active.yaml parsing: required fields, type coercion, bad inputs.
  2. params_to_env + config_hash: deterministic mapping / hashing.
  3. Objective wrapper: we monkeypatch `run_one_trial` to return a
     canned summary dict, then drive a tiny Optuna study over the
     real `_run_single_trial` + `suggest_params` code path. Verifies
     that primary/secondary metrics, config_hash, latency, and cost
     land on `trial.user_attrs`.
  4. Full `run_study` on a one-param YAML: confirms sqlite storage
     + frozen config.yaml + the --resume guard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from scripts import tune as tune_mod


# ---------------------------------------------------------------------------
# 1. active.yaml parsing.
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "active.yaml"
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return p


def _minimal_active(**overrides) -> dict:
    base = {
        "experiment_id": "smoke-v1",
        "objective": {
            "mode": "rag",
            "dataset": "eval/datasets/rag_sample.jsonl",
            "primary_metric": "mean_hit_at_k",
            "secondary_metrics": ["mrr"],
        },
        "search_space": {
            "rag_top_k": {"type": "int", "low": 1, "high": 5},
        },
        "optuna": {
            "sampler": "tpe",
            "n_trials": 3,
            "seed": 7,
            "direction": "maximize",
        },
        "_meta": {"created_by": "human"},
    }
    base.update(overrides)
    return base


class TestLoadActiveConfig:
    def test_happy_path(self, tmp_path: Path):
        path = _write_yaml(tmp_path, _minimal_active())
        config = tune_mod.load_active_config(path)
        assert config.experiment_id == "smoke-v1"
        assert config.mode == "rag"
        assert config.primary_metric == "mean_hit_at_k"
        assert config.secondary_metrics == ["mrr"]
        assert config.sampler == "tpe"
        assert config.n_trials == 3
        assert config.seed == 7
        assert config.direction == "maximize"
        assert "rag_top_k" in config.search_space
        spec = config.search_space["rag_top_k"]
        assert spec.type == "int"
        assert spec.low == 1 and spec.high == 5
        assert config.meta.get("created_by") == "human"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            tune_mod.load_active_config(tmp_path / "nope.yaml")

    def test_missing_experiment_id(self, tmp_path: Path):
        payload = _minimal_active()
        del payload["experiment_id"]
        path = _write_yaml(tmp_path, payload)
        with pytest.raises(ValueError, match="experiment_id"):
            tune_mod.load_active_config(path)

    def test_invalid_mode(self, tmp_path: Path):
        payload = _minimal_active()
        payload["objective"]["mode"] = "bogus"
        path = _write_yaml(tmp_path, payload)
        with pytest.raises(ValueError, match="mode"):
            tune_mod.load_active_config(path)

    def test_invalid_direction(self, tmp_path: Path):
        payload = _minimal_active()
        payload["optuna"]["direction"] = "sideways"
        path = _write_yaml(tmp_path, payload)
        with pytest.raises(ValueError, match="direction"):
            tune_mod.load_active_config(path)

    def test_param_without_low_high(self, tmp_path: Path):
        payload = _minimal_active()
        payload["search_space"] = {"bad": {"type": "int"}}
        path = _write_yaml(tmp_path, payload)
        with pytest.raises(ValueError, match="low.*high|high"):
            tune_mod.load_active_config(path)

    def test_categorical_requires_choices(self, tmp_path: Path):
        payload = _minimal_active()
        payload["search_space"] = {
            "pick": {"type": "categorical"},
        }
        path = _write_yaml(tmp_path, payload)
        with pytest.raises(ValueError, match="choices"):
            tune_mod.load_active_config(path)

    def test_categorical_with_choices_loads(self, tmp_path: Path):
        payload = _minimal_active()
        payload["search_space"] = {
            "pick": {"type": "categorical", "choices": ["a", "b", "c"]},
        }
        path = _write_yaml(tmp_path, payload)
        config = tune_mod.load_active_config(path)
        assert config.search_space["pick"].choices == ["a", "b", "c"]

    def test_loguniform_sets_log_flag(self, tmp_path: Path):
        payload = _minimal_active()
        payload["search_space"] = {
            "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
        }
        path = _write_yaml(tmp_path, payload)
        config = tune_mod.load_active_config(path)
        assert config.search_space["lr"].log is True
        assert config.search_space["lr"].type == "loguniform"


# ---------------------------------------------------------------------------
# 2. params_to_env + config_hash.
# ---------------------------------------------------------------------------


class TestParamsToEnv:
    def test_simple_int(self):
        env = tune_mod.params_to_env({"rag_top_k": 7})
        assert env == {"AIPIPELINE_WORKER_RAG_TOP_K": "7"}

    def test_preserves_float_repr(self):
        env = tune_mod.params_to_env({"ocr_min_confidence_warn": 42.5})
        assert env == {"AIPIPELINE_WORKER_OCR_MIN_CONFIDENCE_WARN": "42.5"}

    def test_bool_becomes_true_false(self):
        env = tune_mod.params_to_env({"cross_modal_enabled": True})
        assert env == {"AIPIPELINE_WORKER_CROSS_MODAL_ENABLED": "true"}

    def test_multiple_params(self):
        env = tune_mod.params_to_env({"a": 1, "b": 2})
        assert set(env) == {"AIPIPELINE_WORKER_A", "AIPIPELINE_WORKER_B"}


class TestConfigHash:
    def test_stable_across_order(self):
        h1 = tune_mod.config_hash({"a": 1, "b": 2})
        h2 = tune_mod.config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_changes_with_values(self):
        h1 = tune_mod.config_hash({"a": 1})
        h2 = tune_mod.config_hash({"a": 2})
        assert h1 != h2

    def test_twelve_hex_chars(self):
        h = tune_mod.config_hash({"x": 1})
        assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# 3. Objective wrapper + run_study driver (with monkeypatched subprocess).
# ---------------------------------------------------------------------------


def _fake_trial_runner(response_map: Dict[str, Any]):
    """Return a drop-in for `tune_mod.run_one_trial` that returns a
    canned summary dict. `response_map` is checked against env_overrides
    so tests can assert the right AIPIPELINE_WORKER_* got set."""

    calls = []

    def _run(*, mode, dataset, env_overrides, rag_top_k, ai_worker_root,
             subprocess_timeout):
        calls.append({
            "mode": mode,
            "dataset": str(dataset),
            "env_overrides": dict(env_overrides),
            "rag_top_k": rag_top_k,
        })
        summary = dict(response_map)
        return summary, 42.0

    _run.calls = calls
    return _run


class TestObjectiveWrapper:
    def test_primary_and_secondary_land_on_trial(self, tmp_path, monkeypatch):
        import optuna

        path = _write_yaml(tmp_path, _minimal_active())
        config = tune_mod.load_active_config(path)

        summary = {"mean_hit_at_k": 0.42, "mrr": 0.33, "mean_total_ms": 17.0}
        fake = _fake_trial_runner(summary)
        monkeypatch.setattr(tune_mod, "run_one_trial", fake)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(
            lambda t: tune_mod._run_single_trial(
                t, config=config,
                ai_worker_root=Path("/fake"),
                subprocess_timeout=None,
            ),
            n_trials=1,
        )
        best = study.best_trial
        assert best.value == pytest.approx(0.42)
        assert best.user_attrs["config_hash"]
        assert best.user_attrs["secondary_metric_values"]["mrr"] == pytest.approx(0.33)
        assert best.user_attrs["cost_usd"] == 0.0
        # rag_top_k is in the env_overrides and also passed as --top-k.
        call = fake.calls[0]
        assert "AIPIPELINE_WORKER_RAG_TOP_K" in call["env_overrides"]
        assert call["rag_top_k"] is not None

    def test_none_primary_returns_negative_infinity_for_maximize(
        self, tmp_path, monkeypatch
    ):
        import optuna

        path = _write_yaml(tmp_path, _minimal_active())
        config = tune_mod.load_active_config(path)

        summary = {"mean_hit_at_k": None, "mrr": None, "mean_total_ms": 1.0}
        monkeypatch.setattr(
            tune_mod, "run_one_trial", _fake_trial_runner(summary),
        )

        class _Trial:
            # Minimal stand-in so we don't need a live study for this
            # path. suggest_int / set_user_attr are the only surfaces
            # _run_single_trial touches.
            def __init__(self):
                self.number = 0
                self.user_attrs: Dict[str, Any] = {}

            def suggest_int(self, name, low, high, step=1):
                return (low + high) // 2

            def suggest_float(self, name, low, high, step=None, log=False):
                return (low + high) / 2.0

            def suggest_categorical(self, name, choices):
                return choices[0]

            def set_user_attr(self, name, value):
                self.user_attrs[name] = value

        trial = _Trial()
        value = tune_mod._run_single_trial(
            trial, config=config,
            ai_worker_root=Path("/fake"),
            subprocess_timeout=None,
        )
        assert value == float("-inf")

    def test_missing_primary_metric_raises_runtime_error(
        self, tmp_path, monkeypatch
    ):
        path = _write_yaml(tmp_path, _minimal_active())
        config = tune_mod.load_active_config(path)

        # Summary doesn't contain the requested primary metric name.
        summary = {"some_other_metric": 1.0}
        monkeypatch.setattr(
            tune_mod, "run_one_trial", _fake_trial_runner(summary),
        )

        class _Trial:
            def __init__(self):
                self.number = 0
                self.user_attrs = {}

            def suggest_int(self, name, low, high, step=1):
                return low

            def suggest_float(self, name, low, high, step=None, log=False):
                return low

            def suggest_categorical(self, name, choices):
                return choices[0]

            def set_user_attr(self, name, value):
                self.user_attrs[name] = value

        with pytest.raises(RuntimeError, match="mean_hit_at_k"):
            tune_mod._run_single_trial(
                _Trial(), config=config,
                ai_worker_root=Path("/fake"),
                subprocess_timeout=None,
            )


# ---------------------------------------------------------------------------
# 4. End-to-end run_study on a monkeypatched subprocess.
# ---------------------------------------------------------------------------


class TestRunStudy:
    def test_creates_db_and_freezes_config(self, tmp_path, monkeypatch):
        yaml_path = _write_yaml(tmp_path, _minimal_active())
        studies_root = tmp_path / "studies"

        summary = {"mean_hit_at_k": 0.5, "mrr": 0.4, "mean_total_ms": 20.0}
        monkeypatch.setattr(
            tune_mod, "run_one_trial", _fake_trial_runner(summary),
        )

        config = tune_mod.load_active_config(yaml_path)
        study = tune_mod.run_study(
            config=config,
            active_yaml_path=yaml_path,
            studies_root=studies_root,
            ai_worker_root=tmp_path,
            resume=False,
            sampler_override=None,
            subprocess_timeout=None,
            n_trials_override=2,
        )

        study_dir = studies_root / "smoke-v1"
        assert (study_dir / "study.db").exists()
        assert (study_dir / "config.yaml").exists()
        # config.yaml is a bit-identical copy of the active.yaml that
        # drove this study.
        assert (
            (study_dir / "config.yaml").read_text(encoding="utf-8")
            == yaml_path.read_text(encoding="utf-8")
        )
        # Both trials completed with the canned metric value.
        assert len([t for t in study.trials if t.value == 0.5]) == 2
        assert study.user_attrs["mode"] == "rag"
        assert study.user_attrs["primary_metric"] == "mean_hit_at_k"

    def test_refuses_to_overwrite_without_resume(
        self, tmp_path, monkeypatch,
    ):
        yaml_path = _write_yaml(tmp_path, _minimal_active())
        studies_root = tmp_path / "studies"
        (studies_root / "smoke-v1").mkdir(parents=True)
        (studies_root / "smoke-v1" / "study.db").write_bytes(b"")

        monkeypatch.setattr(
            tune_mod, "run_one_trial",
            _fake_trial_runner({"mean_hit_at_k": 0.1, "mrr": 0.1,
                                "mean_total_ms": 1.0}),
        )
        config = tune_mod.load_active_config(yaml_path)
        with pytest.raises(SystemExit, match="Refusing to overwrite"):
            tune_mod.run_study(
                config=config,
                active_yaml_path=yaml_path,
                studies_root=studies_root,
                ai_worker_root=tmp_path,
                resume=False,
                sampler_override=None,
                subprocess_timeout=None,
                n_trials_override=1,
            )

    def test_random_sampler_override(self, tmp_path, monkeypatch):
        yaml_path = _write_yaml(tmp_path, _minimal_active())
        monkeypatch.setattr(
            tune_mod, "run_one_trial",
            _fake_trial_runner({"mean_hit_at_k": 0.9, "mrr": 0.8,
                                "mean_total_ms": 1.0}),
        )
        config = tune_mod.load_active_config(yaml_path)
        study = tune_mod.run_study(
            config=config,
            active_yaml_path=yaml_path,
            studies_root=tmp_path / "studies",
            ai_worker_root=tmp_path,
            resume=False,
            sampler_override="random",
            subprocess_timeout=None,
            n_trials_override=1,
        )
        assert study.user_attrs["sampler"] == "random"


# ---------------------------------------------------------------------------
# 5. Shipped active.yaml parses.
# ---------------------------------------------------------------------------


def test_shipped_active_yaml_parses():
    path = Path("eval/experiments/active.yaml")
    if not path.exists():  # pragma: no cover — defensive in case layout moves
        pytest.skip("active.yaml not present (run from ai-worker/)")
    config = tune_mod.load_active_config(path)
    assert config.experiment_id
    assert config.search_space  # at least one param declared
