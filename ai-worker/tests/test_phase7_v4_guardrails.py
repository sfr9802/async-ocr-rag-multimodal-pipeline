"""Guardrails for the Phase 7 v4 retrieval workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from eval.harness.answerability_v4_adapter import V4AdapterError
from scripts import tune as tune_mod
from scripts.export_answerability_audit import (
    _validate_v4_production_chunks_path,
)


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_YAML = AI_WORKER_ROOT / "eval" / "experiments" / "active.yaml"
PHASE7_V4_ARTIFACTS_ENV = "PHASE7_V4_ARTIFACTS_REQUIRED"
PHASE7_V4_ARTIFACTS_SKIP_REASON = (
    "v4 artifacts are not present; set "
    "PHASE7_V4_ARTIFACTS_REQUIRED=1 to require local Phase 7 v4 artifact "
    "preflight"
)

FORBIDDEN_ACTIVE_STRINGS = (
    "v3",
    "anime_namu_v3",
    "rag-cheap-sweep-v3",
    "bge-m3-anime-namu-v3",
)

LEGACY_V3_SCRIPT_PATHS = (
    Path("eval/tune_eval_offline.py"),
    Path("scripts/eval_full_silver_minimal_sweep.py"),
    Path("scripts/eval_wide_mmr_titlecap_sweep.py"),
    Path("scripts/confirm_wide_mmr_best_configs.py"),
    Path("scripts/confirm_embedding_text_variant.py"),
    Path("scripts/confirm_reranker_input_format.py"),
    Path("scripts/confirm_rerank_input_cap_policy.py"),
    Path("scripts/build_legacy_baseline_final.py"),
    Path("scripts/eval_agent_loop_ab_baseline.py"),
)


def _active_yaml() -> dict:
    return yaml.safe_load(ACTIVE_YAML.read_text(encoding="utf-8"))


def _write_fail_closed_yaml(tmp_path: Path, meta: dict) -> Path:
    payload = {
        "experiment_id": "phase7-v4-fail-closed-fixture",
        "objective": {
            "mode": "rag",
            "dataset": (
                "eval/reports/phase7/seeds/llm_silver_focus_50/"
                "queries_v4_llm_silver_focus_50.jsonl"
            ),
            "primary_metric": "mrr",
            "secondary_metrics": ["mean_hit_at_k"],
        },
        "search_space": {
            "rag_top_k": {"type": "categorical", "choices": [10]},
        },
        "optuna": {
            "sampler": "random",
            "n_trials": 1,
            "seed": 42,
            "direction": "maximize",
        },
        "_meta": meta,
    }
    path = tmp_path / "fail_closed_active.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _resolve_ai_worker_path(relative: str) -> Path:
    return AI_WORKER_ROOT / relative


def _canonical_v4_artifact_paths() -> dict[str, Path]:
    artifacts = _active_yaml()["_meta"]["canonical_v4_artifacts"]
    expected = {
        "pages_v4",
        "chunks_v4",
        "rag_chunks",
        "split_manifest",
        "split_manifest_report",
        "validation_report",
    }
    assert set(artifacts) == expected
    return {name: _resolve_ai_worker_path(rel) for name, rel in artifacts.items()}


def _phase7_v4_artifacts_are_required() -> bool:
    return os.environ.get(PHASE7_V4_ARTIFACTS_ENV) == "1"


def _require_or_skip_v4_artifacts(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing and not _phase7_v4_artifacts_are_required():
        pytest.skip(PHASE7_V4_ARTIFACTS_SKIP_REASON)
    assert not missing, (
        "missing Phase 7 v4 artifacts; set up "
        "eval/corpora/namu-v4-structured-combined/ or unset "
        f"{PHASE7_V4_ARTIFACTS_ENV} for non-artifact guardrails only: "
        + "; ".join(missing)
    )


def test_active_yaml_is_phase7_v4_active_template_without_legacy_defaults():
    text = ACTIVE_YAML.read_text(encoding="utf-8").lower()
    for forbidden in FORBIDDEN_ACTIVE_STRINGS:
        assert forbidden not in text

    payload = _active_yaml()
    assert payload["experiment_id"].startswith("phase7-v4-")
    assert "v4" in payload["experiment_id"]

    meta = payload["_meta"]
    assert meta["phase7_v4_guardrail"] is True
    assert meta.get("fail_closed") in (False, None)
    assert str(meta.get("status") or "").lower() not in {
        "fail_closed",
        "fail-closed",
        "disabled",
    }

    dataset = payload["objective"]["dataset"]
    assert "phase7" in dataset
    assert "v4" in dataset

    config = tune_mod.load_active_config(ACTIVE_YAML)
    tune_mod.assert_active_config_runnable(config)


@pytest.mark.parametrize(
    "meta",
    (
        {"fail_closed": True, "reason": "fixture guardrail"},
        {"fail_closed": False, "status": "fail_closed"},
    ),
)
def test_tune_runner_refuses_fail_closed_fixture(tmp_path: Path, meta: dict):
    config = tune_mod.load_active_config(_write_fail_closed_yaml(tmp_path, meta))
    with pytest.raises(SystemExit, match="fail-closed"):
        tune_mod.assert_active_config_runnable(config)


def test_active_yaml_preserves_canonical_artifacts_and_join_policy():
    paths = _canonical_v4_artifact_paths()
    assert {
        "pages_v4",
        "chunks_v4",
        "rag_chunks",
        "split_manifest",
        "split_manifest_report",
        "validation_report",
    } == set(paths)

    audit_cfg = _active_yaml()["_meta"]["answerability_audit"]
    production_join = audit_cfg["production_join_chunks"]
    forbidden_join = audit_cfg["forbidden_production_join_chunks"]

    assert production_join.endswith("rag_chunks.jsonl")
    assert forbidden_join.endswith("chunks_v4.jsonl")
    assert production_join != forbidden_join
    assert audit_cfg["chunks_v4_is_production_retrieval_join_fixture"] is False


def test_phase7_v4_canonical_artifacts_are_present_and_identifiable():
    """Opt-in local artifact preflight.

    Default guardrails:
        python -m pytest tests/test_phase7_v4_guardrails.py

    Force local v4 artifact validation:
        PHASE7_V4_ARTIFACTS_REQUIRED=1 python -m pytest tests/test_phase7_v4_guardrails.py
    """
    paths = _canonical_v4_artifact_paths()
    _require_or_skip_v4_artifacts(paths)

    for name, path in paths.items():
        assert path.stat().st_size > 0, f"empty Phase 7 v4 artifact {name}: {path}"

    first_page = json.loads(paths["pages_v4"].open(encoding="utf-8").readline())
    assert first_page["schema_version"] == "namu_anime_v4_page", paths["pages_v4"]
    assert first_page["page_id"]
    assert first_page["retrieval_title"]

    first_chunk = json.loads(paths["chunks_v4"].open(encoding="utf-8").readline())
    assert first_chunk["schema_version"] == "namu_anime_v4_chunk", paths["chunks_v4"]
    assert first_chunk["chunk_id"]
    assert first_chunk["page_id"]
    assert "text_for_embedding" in first_chunk

    first_rag_chunk = json.loads(
        paths["rag_chunks"].open(encoding="utf-8").readline()
    )
    assert first_rag_chunk["schema_version"] == "namu_anime_v4_rag_chunk", paths[
        "rag_chunks"
    ]
    assert first_rag_chunk["chunk_id"]
    assert first_rag_chunk["doc_id"]
    assert "chunk_text" in first_rag_chunk
    assert "embedding_text" in first_rag_chunk

    split_manifest = json.loads(
        paths["split_manifest"].read_text(encoding="utf-8")
    )
    assert split_manifest["counts"]["docs"]["total"] == 4314, paths["split_manifest"]

    split_report = json.loads(
        paths["split_manifest_report"].read_text(encoding="utf-8")
    )
    assert split_report["schema_version"] == "namu_anime_v4_split_report", paths[
        "split_manifest_report"
    ]
    assert split_report["total_docs"] == 4314, paths["split_manifest_report"]
    assert split_report["leakage"]["doc_id_overlap"] == [], paths[
        "split_manifest_report"
    ]
    assert split_report["leakage"]["group_id_overlap"] == [], paths[
        "split_manifest_report"
    ]

    validation_report = json.loads(
        paths["validation_report"].read_text(encoding="utf-8")
    )
    assert validation_report["pages_count"] == 4314, paths["validation_report"]
    assert validation_report["duplicate_page_id_count"] == 0, paths[
        "validation_report"
    ]
    assert validation_report["duplicate_chunk_id_count"] == 0, paths[
        "validation_report"
    ]
    assert validation_report["schema_version_mismatch_pages"] == 0, paths[
        "validation_report"
    ]
    assert validation_report["schema_version_mismatch_chunks"] == 0, paths[
        "validation_report"
    ]

    audit_cfg = _active_yaml()["_meta"]["answerability_audit"]
    production_join = _resolve_ai_worker_path(audit_cfg["production_join_chunks"])
    assert production_join == paths["rag_chunks"]
    _validate_v4_production_chunks_path(production_join)


def test_answerability_production_join_requires_rag_chunks_namespace():
    audit_cfg = _active_yaml()["_meta"]["answerability_audit"]
    production_join = audit_cfg["production_join_chunks"]
    forbidden_join = audit_cfg["forbidden_production_join_chunks"]

    assert production_join.endswith("rag_chunks.jsonl")
    assert forbidden_join.endswith("chunks_v4.jsonl")

    _validate_v4_production_chunks_path(_resolve_ai_worker_path(production_join))
    with pytest.raises(V4AdapterError, match="rag_chunks.jsonl"):
        _validate_v4_production_chunks_path(_resolve_ai_worker_path(forbidden_join))


def test_v3_default_scripts_are_marked_legacy_only():
    for relative_path in LEGACY_V3_SCRIPT_PATHS:
        path = AI_WORKER_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        assert "LEGACY V3 ONLY" in text


def _parser_help_for(args: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    from eval.run_eval import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(args)
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_run_eval_help_separates_legacy_v3_from_phase7_v4(capsys):
    main_help = _parser_help_for(["--help"], capsys)
    assert "Phase 7 active eval/tuning" in main_help
    assert "namu-v4-structured-combined" in main_help
    assert "LEGACY V3 ONLY" in main_help
    assert "rag-cheap-sweep-v3" not in main_help
    assert "bge-m3-anime-namu-v3" not in main_help

    retrieval_help = _parser_help_for(["retrieval", "--help"], capsys)
    assert "eval/corpora/anime_namu_v3/corpus.jsonl" in retrieval_help
    assert "LEGACY V3 ONLY" in retrieval_help
    assert "namu-v4-structured-combined" in retrieval_help

    phase2_modes = (
        "retrieval-rerank",
        "phase2a-reranker-comparison",
        "phase2a-reranker-failure-analysis",
        "phase2a-latency-sweep",
        "emit-preprocessed-corpus",
    )
    for mode in phase2_modes:
        help_text = _parser_help_for([mode, "--help"], capsys)
        assert "LEGACY V3 ONLY" in help_text

    preprocess_help = _parser_help_for(
        ["emit-preprocessed-corpus", "--help"],
        capsys,
    )
    assert "eval/corpora/anime_namu_v3_preprocessed" in preprocess_help
    assert "LEGACY V3 ONLY default" in preprocess_help
