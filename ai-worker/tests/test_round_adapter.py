"""Tests for `scripts.round_adapter` — the optuna-round-refinement adapter.

Strategy: every test that needs the skill's jsonschemas uses `skill_dir`
fixture which skips gracefully when the skill isn't installed. Bundle
export uses an in-memory Optuna study so it never touches the shipped
rag-cheap-sweep-v3 study.db.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from scripts import round_adapter as ra
from scripts import tune as tune_mod


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def skill_dir() -> Path:
    try:
        return ra.find_skill_dir()
    except FileNotFoundError:
        pytest.skip("optuna-round-refinement skill not installed")


def _build_tiny_study(tmp_path: Path, experiment: str) -> Path:
    """Create a study.db + config.yaml pair with 4 completed trials over
    one int axis + one categorical axis. Returns studies_root."""
    import optuna

    studies_root = tmp_path / "studies"
    study_dir = studies_root / experiment
    study_dir.mkdir(parents=True)

    # Minimal frozen active.yaml snapshot — mirrors load_active_config's
    # expected shape.
    cfg = {
        "experiment_id": experiment,
        "objective": {
            "mode": "rag",
            "dataset": "eval/datasets/rag_sample.jsonl",
            "primary_metric": "mean_hit_at_k",
            "secondary_metrics": ["mrr"],
        },
        "search_space": {
            "rag_top_k": {"type": "int", "low": 3, "high": 15},
            "rag_use_mmr": {"type": "categorical", "choices": [False, True]},
        },
        "optuna": {"sampler": "tpe", "n_trials": 4, "seed": 1, "direction": "maximize"},
        "_meta": {"created_by": "human", "notes": "tiny test study"},
    }
    (study_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    storage = f"sqlite:///{(study_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=experiment, storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=1),
        direction="maximize",
    )

    # Four synthetic trials: two plateau'd at top, two lower. One hits
    # the low boundary (rag_top_k=3) so boundary_hits shows up.
    fixtures = [
        ({"rag_top_k": 7,  "rag_use_mmr": False}, 0.85),
        ({"rag_top_k": 10, "rag_use_mmr": True},  0.85),
        ({"rag_top_k": 3,  "rag_use_mmr": False}, 0.70),
        ({"rag_top_k": 5,  "rag_use_mmr": True},  0.75),
    ]
    for params, value in fixtures:
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                "rag_top_k": optuna.distributions.IntDistribution(3, 15),
                "rag_use_mmr": optuna.distributions.CategoricalDistribution([False, True]),
            },
            value=value,
            user_attrs={"config_hash": "deadbeef", "latency_ms": 12.3, "cost_usd": 0.0},
        )
        study.add_trial(trial)
    return studies_root


# ---------------------------------------------------------------------------
# find_skill_dir / _sha256_hex / bump_experiment_id.
# ---------------------------------------------------------------------------


class TestFindSkillDir:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch):
        fake = tmp_path / "fake-skill"
        fake.mkdir()
        (fake / "SKILL.md").write_text("x")
        monkeypatch.setenv("OPTUNA_ROUND_REFINEMENT_HOME", str(fake))
        assert ra.find_skill_dir() == fake

    def test_missing_raises(self, tmp_path: Path, monkeypatch):
        # Point the env var at a dir with no SKILL.md, and clobber the
        # user-home fallback to somewhere empty.
        empty = tmp_path / "nope"
        empty.mkdir()
        monkeypatch.setenv("OPTUNA_ROUND_REFINEMENT_HOME", str(empty))
        monkeypatch.setenv("HOME", str(tmp_path))           # posix
        monkeypatch.setenv("USERPROFILE", str(tmp_path))    # windows
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            ra.find_skill_dir()


class TestBumpExperimentId:
    @pytest.mark.parametrize("current,expected", [
        ("rag-cheap-sweep-v3", "rag-cheap-sweep-v4"),
        ("foo-v1",             "foo-v2"),
        ("no-version",         "no-version-v2"),
        # "v9" has no hyphen-v prefix structure, so we treat it as an
        # unversioned name and just append -v2. That's the safer
        # behaviour than interpreting the whole string as the version
        # counter.
        ("v9",                 "v9-v2"),
        ("ocr-eng-v42",        "ocr-eng-v43"),
    ])
    def test_various(self, current: str, expected: str):
        assert ra.bump_experiment_id(current) == expected


class TestCanonicalJsonAndHash:
    def test_key_order_irrelevant(self):
        a = ra._canonical_json({"b": 1, "a": 2})
        b = ra._canonical_json({"a": 2, "b": 1})
        assert a == b
        assert ra._sha256_hex(a) == ra._sha256_hex(b)

    def test_sha_length_and_alphabet(self):
        h = ra._sha256_hex("hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# export_study_bundle.
# ---------------------------------------------------------------------------


class TestExportStudyBundle:
    def test_happy_path(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        assert bundle["schema_version"] == "1.0"
        assert bundle["round_id"] == "round_01"
        assert bundle["study_id"] == "exp-v1"
        assert bundle["n_trials"] == 4
        assert bundle["statistics"]["n_complete"] == 4
        assert bundle["statistics"]["best_value"] == pytest.approx(0.85)
        # rag_top_k=3 is the low boundary; one trial hits it.
        assert bundle["statistics"]["boundary_hits"]["rag_top_k"]["low"] == 1
        # rag_top_k=15 never sampled; high boundary hit count is 0.
        assert bundle["statistics"]["boundary_hits"]["rag_top_k"]["high"] == 0

    def test_bool_params_preserved(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        # Booleans must survive the round-trip — they are valid JSON primitives.
        for t in bundle["trials"]:
            assert isinstance(t["params"]["rag_use_mmr"], bool)

    def test_autocluster_finds_plateau(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        # Two trials tied at 0.85 should form a cluster; singletons at
        # 0.70 and 0.75 should not.
        clusters = bundle.get("clusters", [])
        assert len(clusters) == 1
        assert set(clusters[0]["trial_numbers"]) == {0, 1}
        assert clusters[0]["center"]["value"] == pytest.approx(0.85)

    def test_parent_config_hash_is_sha256_of_yaml(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        raw = (studies_root / "exp-v1" / "config.yaml").read_bytes()
        assert bundle["parent_config_hash"] == ra._sha256_hex(raw)

    def test_missing_study_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="study.db"):
            ra.export_study_bundle(
                experiment_id="nope", studies_root=tmp_path / "nowhere",
            )

    def test_validates_against_skill_schema(
        self, tmp_path: Path, skill_dir: Path,
    ):
        import jsonschema

        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        schema = ra._load_schema(skill_dir, "study_bundle")
        jsonschema.validate(bundle, schema)  # raises on failure


# ---------------------------------------------------------------------------
# render_llm_input.
# ---------------------------------------------------------------------------


class TestRenderLlmInput:
    def test_sections_present(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        out = tmp_path / "brief.md"
        ra.render_llm_input(bundle, out)
        text = out.read_text(encoding="utf-8")
        # The prompts key off these exact section headers, so they MUST
        # be present after rendering.
        for header in (
            "## 1. Frozen search space",
            "## 2. Headline statistics",
            "## 3. Param importances",
            "## 4. Boundary hits",
            "## 5. Best trial",
            "## 6. Top-k trials",
            "## 7. Clusters",
        ):
            assert header in text

    def test_bundle_hash_in_brief(self, tmp_path: Path):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        out = tmp_path / "brief.md"
        ra.render_llm_input(bundle, out)
        expected_hash = ra._sha256_hex(ra._canonical_json(bundle))
        assert expected_hash in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# validate_and_hash.
# ---------------------------------------------------------------------------


class TestValidateAndHash:
    def test_valid_bundle_round_trip(
        self, tmp_path: Path, skill_dir: Path,
    ):
        studies_root = _build_tiny_study(tmp_path, "exp-v1")
        bundle = ra.export_study_bundle(
            experiment_id="exp-v1", studies_root=studies_root,
        )
        path = tmp_path / "bundle.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        digest = ra.validate_and_hash(path, kind="study_bundle", skill_dir=skill_dir)
        assert len(digest) == 64

    def test_invalid_payload_raises(
        self, tmp_path: Path, skill_dir: Path,
    ):
        import jsonschema

        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        with pytest.raises(jsonschema.ValidationError):
            ra.validate_and_hash(path, kind="study_bundle", skill_dir=skill_dir)


# ---------------------------------------------------------------------------
# apply_next_round_config.
# ---------------------------------------------------------------------------


def _write_active(tmp_path: Path, payload: Dict[str, Any]) -> Path:
    p = tmp_path / "active.yaml"
    p.write_text(yaml.safe_dump(payload, sort_keys=False))
    return p


def _minimal_next_round_config() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "round_id": "round_02",
        "n_trials": 10,
        "sampler": {"type": "TPESampler", "params": {"multivariate": True}, "seed": 42},
        "pruner": {"type": "NopPruner", "params": {}},
        "search_space": {
            "rag_top_k": {"type": "int", "low": 7, "high": 15},
        },
        "fixed_params": {"rag_use_mmr": False, "rag_query_parser": "off"},
        "provenance": {
            "kind": "llm_proposed",
            "source_round_id": "round_01",
            "source_bundle_hash": "a" * 64,
            "parent_config_hash": "b" * 64,
            "generated_at": "2026-04-24T00:00:00Z",
            "generated_by": {"tool": "claude_code", "model": "test",
                             "prompt_version": "0.1.0", "prompt_path": "x"},
            "rationale": "test rationale",
        },
    }


class TestApplyNextRoundConfig:
    def test_preserves_objective_and_bumps_experiment(
        self, tmp_path: Path, skill_dir: Path,
    ):
        active = _write_active(tmp_path, {
            "experiment_id": "rag-cheap-sweep-v3",
            "objective": {
                "mode": "rag",
                "dataset": "eval/datasets/x.jsonl",
                "primary_metric": "mrr",
                "secondary_metrics": ["mean_hit_at_k"],
            },
            "search_space": {"rag_top_k": {"type": "int", "low": 3, "high": 15}},
            "optuna": {"sampler": "tpe", "n_trials": 18, "seed": 42,
                       "direction": "maximize"},
            "_meta": {"created_by": "human"},
        })
        cfg_path = tmp_path / "round_02_config.json"
        cfg_path.write_text(json.dumps(_minimal_next_round_config()),
                            encoding="utf-8")

        new = ra.apply_next_round_config(
            config_path=cfg_path, active_yaml_path=active,
            skill_dir=skill_dir,
        )
        assert new["experiment_id"] == "rag-cheap-sweep-v4"
        # objective preserved verbatim
        assert new["objective"]["primary_metric"] == "mrr"
        assert new["objective"]["mode"] == "rag"
        # search_space replaced
        assert list(new["search_space"].keys()) == ["rag_top_k"]
        assert new["search_space"]["rag_top_k"]["low"] == 7
        # fixed_params applied
        assert new["fixed_params"]["rag_use_mmr"] is False
        assert new["fixed_params"]["rag_query_parser"] == "off"
        # optuna section rebuilt
        assert new["optuna"]["n_trials"] == 10
        assert new["optuna"]["sampler"] == "tpe"
        assert new["optuna"]["direction"] == "maximize"  # preserved
        # provenance migrated into _meta
        assert new["_meta"]["created_by"] == "claude-proposed"
        assert new["_meta"]["parent_experiment_id"] == "rag-cheap-sweep-v3"
        assert new["_meta"]["source_bundle_hash"] == "a" * 64
        assert new["_meta"]["parent_config_hash"] == "b" * 64

    def test_rejects_unsupported_sampler(
        self, tmp_path: Path, skill_dir: Path,
    ):
        active = _write_active(tmp_path, {
            "experiment_id": "x", "objective": {"mode": "rag", "dataset": "y",
            "primary_metric": "mrr", "secondary_metrics": []},
            "search_space": {"rag_top_k": {"type": "int", "low": 1, "high": 2}},
            "optuna": {"sampler": "tpe", "n_trials": 1, "seed": 0,
                       "direction": "maximize"},
            "_meta": {"created_by": "human"},
        })
        cfg = _minimal_next_round_config()
        cfg["sampler"]["type"] = "CmaEsSampler"
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported sampler"):
            ra.apply_next_round_config(
                config_path=cfg_path, active_yaml_path=active,
                skill_dir=skill_dir,
            )

    def test_rejects_mismatched_bundle_hash(
        self, tmp_path: Path, skill_dir: Path,
    ):
        """SKILL.md §8.1: config.source_bundle_hash must match on-disk bundle."""
        active = _write_active(tmp_path, {
            "experiment_id": "exp-v1",
            "objective": {"mode": "rag", "dataset": "y", "primary_metric": "mrr",
                          "secondary_metrics": []},
            "search_space": {"rag_top_k": {"type": "int", "low": 1, "high": 2}},
            "optuna": {"sampler": "tpe", "n_trials": 1, "seed": 0,
                       "direction": "maximize"},
            "_meta": {"created_by": "human"},
        })
        # Write a bundle file next to the config. Its real hash will NOT
        # match the one we put in provenance, so apply must reject.
        study_dir = tmp_path
        (study_dir / "round_01_bundle.json").write_text(
            json.dumps({"schema_version": "1.0", "round_id": "round_01",
                        "study_id": "x", "parent_config_hash": None,
                        "optuna": {"version": "x",
                                   "sampler": {"type": "RandomSampler", "params": {}},
                                   "pruner": {"type": "NopPruner", "params": {}}},
                        "objective": {"name": "m", "direction": "maximize"},
                        "search_space": {"a": {"type": "int", "low": 1, "high": 2}},
                        "n_trials": 0, "trials": [],
                        "best_trial": {"number": 0, "state": "COMPLETE",
                                       "value": 1.0, "params": {}},
                        "statistics": {"n_complete": 0, "n_pruned": 0,
                                       "n_failed": 0}}),
            encoding="utf-8",
        )
        cfg = _minimal_next_round_config()
        # claimed hash is all a's — bundle real hash is something else.
        cfg_path = tmp_path / "round_02_config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        with pytest.raises(ValueError, match="source_bundle_hash"):
            ra.apply_next_round_config(
                config_path=cfg_path, active_yaml_path=active,
                skill_dir=skill_dir,
            )

    def test_dataset_migrates_from_fixed_params_to_objective(
        self, tmp_path: Path, skill_dir: Path,
    ):
        active = _write_active(tmp_path, {
            "experiment_id": "old-v1",
            "objective": {"mode": "rag", "dataset": "old.jsonl",
                          "primary_metric": "mrr", "secondary_metrics": []},
            "search_space": {"rag_top_k": {"type": "int", "low": 1, "high": 2}},
            "optuna": {"sampler": "tpe", "n_trials": 1, "seed": 0,
                       "direction": "maximize"},
            "_meta": {"created_by": "human"},
        })
        cfg = _minimal_next_round_config()
        cfg["fixed_params"]["dataset"] = "new.jsonl"
        cfg_path = tmp_path / "c.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        new = ra.apply_next_round_config(
            config_path=cfg_path, active_yaml_path=active,
            skill_dir=skill_dir,
        )
        assert new["objective"]["dataset"] == "new.jsonl"
        assert "dataset" not in new["fixed_params"]


# ---------------------------------------------------------------------------
# tune.py `fixed_params` integration.
# ---------------------------------------------------------------------------


class TestTuneFixedParams:
    def test_fixed_params_loaded(self, tmp_path: Path):
        path = tmp_path / "active.yaml"
        path.write_text(yaml.safe_dump({
            "experiment_id": "x",
            "objective": {"mode": "rag", "dataset": "d",
                          "primary_metric": "mrr", "secondary_metrics": []},
            "search_space": {"rag_top_k": {"type": "int", "low": 1, "high": 5}},
            "fixed_params": {"rag_use_mmr": False, "rag_query_parser": "off"},
            "optuna": {"sampler": "tpe", "n_trials": 2, "seed": 1,
                       "direction": "maximize"},
        }, sort_keys=False), encoding="utf-8")
        cfg = tune_mod.load_active_config(path)
        assert cfg.fixed_params == {"rag_use_mmr": False, "rag_query_parser": "off"}

    def test_collision_rejected(self, tmp_path: Path):
        path = tmp_path / "active.yaml"
        path.write_text(yaml.safe_dump({
            "experiment_id": "x",
            "objective": {"mode": "rag", "dataset": "d",
                          "primary_metric": "mrr", "secondary_metrics": []},
            "search_space": {"rag_top_k": {"type": "int", "low": 1, "high": 5}},
            "fixed_params": {"rag_top_k": 7},
            "optuna": {"sampler": "tpe", "n_trials": 2, "seed": 1,
                       "direction": "maximize"},
        }, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="share keys"):
            tune_mod.load_active_config(path)

    def test_fixed_params_flow_into_env(self, tmp_path: Path, monkeypatch):
        """run_one_trial receives fixed_params in env_overrides alongside
        sampled params."""
        calls = []

        def _fake(*, mode, dataset, env_overrides, rag_top_k,
                  ai_worker_root, subprocess_timeout):
            calls.append(dict(env_overrides))
            return {"mean_hit_at_k": 0.9, "mrr": 0.8, "mean_total_ms": 1.0}, 1.0

        monkeypatch.setattr(tune_mod, "run_one_trial", _fake)

        path = tmp_path / "active.yaml"
        path.write_text(yaml.safe_dump({
            "experiment_id": "x",
            "objective": {"mode": "rag", "dataset": "d.jsonl",
                          "primary_metric": "mean_hit_at_k",
                          "secondary_metrics": ["mrr"]},
            "search_space": {"rag_top_k": {"type": "int", "low": 3, "high": 3}},
            "fixed_params": {"rag_use_mmr": False, "rag_query_parser": "off"},
            "optuna": {"sampler": "tpe", "n_trials": 1, "seed": 1,
                       "direction": "maximize"},
        }, sort_keys=False), encoding="utf-8")
        cfg = tune_mod.load_active_config(path)

        import optuna
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1),
        )
        study.optimize(
            lambda t: tune_mod._run_single_trial(
                t, config=cfg, ai_worker_root=Path("/fake"),
                subprocess_timeout=None,
            ),
            n_trials=1,
        )

        assert calls, "run_one_trial was never called"
        env = calls[0]
        assert env["AIPIPELINE_WORKER_RAG_TOP_K"] == "3"       # from search_space
        assert env["AIPIPELINE_WORKER_RAG_USE_MMR"] == "false"  # from fixed_params
        assert env["AIPIPELINE_WORKER_RAG_QUERY_PARSER"] == "off"  # from fixed_params
