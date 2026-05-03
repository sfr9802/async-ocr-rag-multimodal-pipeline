"""Optuna-driven hyperparameter tuning over the eval harness.

Reads `eval/experiments/active.yaml` (cwd=ai-worker/), drives a TPE study by
shelling out to `python -m eval.run_eval <mode> ...` once per trial,
parses the JSON report, and records the primary / secondary metrics
plus latency + cost onto each Optuna trial.

Why shell out instead of calling `run_eval` in-process: trial parameters
map to `AIPIPELINE_WORKER_*` env vars consumed by pydantic-settings at
WorkerSettings() construction. A subprocess picks up the mutated env
cleanly without mucking with the `get_settings()` singleton cache,
and it mirrors how the worker itself reads config at boot.

Usage (from ai-worker/):

    python -m scripts.tune --experiment rag-cheap-sweep-v1 --n-trials 50
    python -m scripts.tune --experiment rag-cheap-sweep-v1 --resume
    python -m scripts.tune --experiment rag-cheap-sweep-v1 \
        --active-yaml eval/experiments/active.yaml --random-sampler
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

log = logging.getLogger("tune")


# ---------------------------------------------------------------------------
# active.yaml schema (pydantic-settings is for env vars; yaml.safe_load is
# the right tool for this — see the "Notes" section of the kickoff prompt).
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """One entry inside `search_space`.

    Supported `type` values: `int`, `float`, `categorical`, `loguniform`
    (alias for `float` with `log=True`). Extra fields like `step` pass
    through to `trial.suggest_*`.
    """

    name: str
    type: str
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None


@dataclass
class ActiveConfig:
    experiment_id: str
    mode: str
    dataset: Path
    primary_metric: str
    secondary_metrics: List[str]
    search_space: Dict[str, ParamSpec]
    sampler: str
    n_trials: int
    seed: Optional[int]
    direction: str
    meta: Dict[str, Any] = field(default_factory=dict)
    # Raw snapshot as loaded — used verbatim when freezing the config
    # copy into the study directory so replay is bit-identical.
    raw: Dict[str, Any] = field(default_factory=dict)


# Keys in `search_space` whose name does not match a WorkerSettings
# field directly. Left empty for Phase 1 but the indirection keeps the
# env-var mapping one edit away if a YAML alias ever differs from the
# canonical config field.
_KEY_TO_SETTING: Dict[str, str] = {}


# Keys that Phase 1 YAMLs may name but that are NOT env-var-backed in
# WorkerSettings yet. tune.py sets the env var regardless (a no-op so
# the YAML remains self-documenting), but logs a warning once per run
# so the operator knows the knob isn't moving the eval yet.
_PHASE2_KNOWN_UNWIRED = frozenset({
    "max_query_chars",
    "short_query_words",
    "max_fused_chunk_chars",
    "excerpt_chars",
})


def load_active_config(path: Path) -> ActiveConfig:
    """Parse `active.yaml` into a typed dataclass.

    Raises ValueError with a clear message on any missing required
    section rather than letting a cryptic KeyError bubble up.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"active.yaml not found at {path}. Create one from the "
            f"template in docs/tuning.md."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"active.yaml at {path} must be a mapping at top level, "
            f"got {type(raw).__name__}"
        )

    experiment_id = _require(raw, "experiment_id", str, where=str(path))
    objective = _require(raw, "objective", dict, where=str(path))
    search_space_raw = _require(raw, "search_space", dict, where=str(path))
    optuna_cfg = _require(raw, "optuna", dict, where=str(path))

    mode = _require(objective, "mode", str, where="objective")
    if mode not in ("rag", "ocr", "multimodal"):
        raise ValueError(
            f"objective.mode must be one of rag|ocr|multimodal, got {mode!r}"
        )
    dataset = Path(_require(objective, "dataset", str, where="objective"))
    primary_metric = _require(objective, "primary_metric", str, where="objective")
    secondary_metrics = objective.get("secondary_metrics") or []
    if not isinstance(secondary_metrics, list):
        raise ValueError("objective.secondary_metrics must be a list")
    secondary_metrics = [str(m) for m in secondary_metrics]

    search_space: Dict[str, ParamSpec] = {}
    for name, spec in search_space_raw.items():
        search_space[name] = _parse_param(name, spec)

    sampler = str(optuna_cfg.get("sampler", "tpe")).lower()
    if sampler not in ("tpe", "random"):
        raise ValueError(
            f"optuna.sampler must be tpe or random, got {sampler!r}"
        )
    n_trials = int(optuna_cfg.get("n_trials", 50))
    seed = optuna_cfg.get("seed")
    if seed is not None:
        seed = int(seed)
    direction = str(optuna_cfg.get("direction", "maximize")).lower()
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"optuna.direction must be maximize or minimize, got {direction!r}"
        )

    meta = raw.get("_meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("_meta must be a mapping")

    return ActiveConfig(
        experiment_id=experiment_id,
        mode=mode,
        dataset=dataset,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        search_space=search_space,
        sampler=sampler,
        n_trials=n_trials,
        seed=seed,
        direction=direction,
        meta=meta,
        raw=raw,
    )


def _require(raw: Mapping[str, Any], key: str, typ: type, *, where: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field {key!r} in {where}")
    value = raw[key]
    if not isinstance(value, typ):
        raise ValueError(
            f"Field {key!r} in {where} must be {typ.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _parse_param(name: str, spec: Any) -> ParamSpec:
    if not isinstance(spec, dict):
        raise ValueError(
            f"search_space.{name} must be a mapping, got {type(spec).__name__}"
        )
    ptype = str(spec.get("type", "")).lower()
    if ptype not in ("int", "float", "categorical", "loguniform"):
        raise ValueError(
            f"search_space.{name}.type must be int|float|categorical|loguniform, "
            f"got {ptype!r}"
        )
    if ptype in ("int", "float", "loguniform"):
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"search_space.{name} ({ptype}) requires `low` and `high`"
            )
    choices = spec.get("choices")
    if ptype == "categorical":
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                f"search_space.{name} (categorical) requires a non-empty `choices` list"
            )
    return ParamSpec(
        name=name,
        type=ptype,
        low=spec.get("low"),
        high=spec.get("high"),
        step=spec.get("step"),
        log=bool(spec.get("log", False)) or ptype == "loguniform",
        choices=choices,
    )


# ---------------------------------------------------------------------------
# Suggesting + mapping params onto env vars.
# ---------------------------------------------------------------------------


def suggest_params(trial: "optuna.trial.Trial", config: ActiveConfig) -> Dict[str, Any]:
    """Ask Optuna for one concrete point in the search space."""
    params: Dict[str, Any] = {}
    for name, spec in config.search_space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(
                name, int(spec.low), int(spec.high),
                step=int(spec.step) if spec.step else 1,
            )
        elif spec.type in ("float", "loguniform"):
            params[name] = trial.suggest_float(
                name, float(spec.low), float(spec.high),
                step=float(spec.step) if spec.step else None,
                log=spec.log,
            )
        elif spec.type == "categorical":
            params[name] = trial.suggest_categorical(name, spec.choices)
        else:  # pragma: no cover — _parse_param already validated
            raise RuntimeError(f"unknown param type: {spec.type}")
    return params


def params_to_env(params: Mapping[str, Any]) -> Dict[str, str]:
    """Turn {search_space key: value} into {env var name: stringified}.

    Each key maps to `AIPIPELINE_WORKER_<KEY_UPPER>` unless the optional
    alias table `_KEY_TO_SETTING` overrides it. Values are stringified
    for the subprocess environment; booleans become `"true"/"false"`.
    """
    env: Dict[str, str] = {}
    for name, value in params.items():
        setting = _KEY_TO_SETTING.get(name, name)
        env_key = f"AIPIPELINE_WORKER_{setting.upper()}"
        if isinstance(value, bool):
            env[env_key] = "true" if value else "false"
        else:
            env[env_key] = str(value)
    return env


def config_hash(params: Mapping[str, Any]) -> str:
    """Deterministic 12-hex-char hash for one trial's param set.

    Stored as a user_attr so downstream analysis can cluster trials
    that sampled identical-after-rounding parameter combinations.
    """
    payload = json.dumps(dict(sorted(params.items())), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Running one trial: shell out to `python -m eval.run_eval ...` and parse
# the JSON report.
# ---------------------------------------------------------------------------


def build_eval_cmd(
    mode: str,
    dataset: Path,
    out_json: Path,
    *,
    rag_top_k: Optional[int] = None,
) -> List[str]:
    """Assemble the `python -m eval.run_eval <mode>` command."""
    cmd = [
        sys.executable, "-m", "eval.run_eval", mode,
        "--dataset", str(dataset),
        "--out-json", str(out_json),
        "--no-csv",
    ]
    if mode == "rag" and rag_top_k is not None:
        # --top-k is redundant with AIPIPELINE_WORKER_RAG_TOP_K but the
        # explicit flag shows up in process listings which is handy
        # when you eyeball `ps` during a long study.
        cmd.extend(["--top-k", str(int(rag_top_k))])
    return cmd


def run_one_trial(
    *,
    mode: str,
    dataset: Path,
    env_overrides: Mapping[str, str],
    rag_top_k: Optional[int],
    ai_worker_root: Path,
    subprocess_timeout: Optional[float],
) -> Tuple[Dict[str, Any], float]:
    """Run one eval subprocess, return (summary_dict, wall_ms).

    Raises RuntimeError on subprocess non-zero exit or malformed JSON.
    Timeout (if set) is a hard wall — caller decides how to classify it.
    """
    env = os.environ.copy()
    env.update(env_overrides)

    with tempfile.TemporaryDirectory(prefix="tune_") as tmp:
        out_json = Path(tmp) / "report.json"
        cmd = build_eval_cmd(mode, dataset, out_json, rag_top_k=rag_top_k)
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(ai_worker_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=subprocess_timeout,
            )
        except subprocess.TimeoutExpired as ex:
            raise RuntimeError(
                f"eval subprocess timed out after {subprocess_timeout}s"
            ) from ex
        wall_ms = (time.perf_counter() - t0) * 1000.0

        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-10:]
            raise RuntimeError(
                f"eval subprocess exited {result.returncode}. "
                f"stderr tail: {tail}"
            )
        if not out_json.exists():
            raise RuntimeError(
                f"eval subprocess finished without writing {out_json}"
            )
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        if "summary" not in payload:
            raise RuntimeError(
                f"eval report missing `summary` key: keys={list(payload)}"
            )
        return payload["summary"], wall_ms


def extract_metric(summary: Mapping[str, Any], name: str) -> Optional[float]:
    """Pull a metric out of the summary dict with a helpful error."""
    if name not in summary:
        raise RuntimeError(
            f"Metric {name!r} not in eval summary. Available keys: "
            f"{sorted(summary)}"
        )
    value = summary[name]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as ex:
        raise RuntimeError(
            f"Metric {name!r} in summary is not numeric: {value!r}"
        ) from ex


# ---------------------------------------------------------------------------
# Top-level study driver.
# ---------------------------------------------------------------------------


def run_study(
    *,
    config: ActiveConfig,
    active_yaml_path: Path,
    studies_root: Path,
    ai_worker_root: Path,
    resume: bool,
    sampler_override: Optional[str],
    subprocess_timeout: Optional[float],
    n_trials_override: Optional[int],
) -> "optuna.Study":
    import optuna  # local import so `--help` works without optuna installed

    assert_active_config_runnable(config)

    study_dir = studies_root / config.experiment_id
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "study.db"

    if db_path.exists() and not resume:
        raise SystemExit(
            f"Refusing to overwrite existing study at {db_path}. "
            f"Pass --resume to keep appending trials to it, or delete "
            f"the file first if you really want a fresh study."
        )

    # Freeze the active.yaml so later runs / Claude sessions can replay
    # exactly what was searched, even if active.yaml rotates.
    frozen_config = study_dir / "config.yaml"
    if not frozen_config.exists() or not resume:
        shutil.copyfile(active_yaml_path, frozen_config)
    else:
        log.info(
            "Resume mode: keeping existing frozen config at %s",
            frozen_config,
        )

    # Warn once about YAML keys we know aren't wired yet.
    unwired = set(config.search_space) & _PHASE2_KNOWN_UNWIRED
    if unwired:
        log.warning(
            "Search-space keys %s are not env-var-backed in WorkerSettings "
            "yet — env vars will be set but the eval will ignore them. "
            "Wire them through config.py before relying on their effect.",
            sorted(unwired),
        )

    sampler_name = (sampler_override or config.sampler).lower()
    sampler = _build_sampler(sampler_name, config.seed)
    direction = config.direction

    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name=config.experiment_id,
        storage=storage_url,
        sampler=sampler,
        direction=direction,
        load_if_exists=True,
    )
    # Stash the active metadata on the study so optuna-dashboard shows
    # the dataset / metric at a glance.
    study.set_user_attr("mode", config.mode)
    study.set_user_attr("dataset", str(config.dataset))
    study.set_user_attr("primary_metric", config.primary_metric)
    study.set_user_attr("secondary_metrics", list(config.secondary_metrics))
    study.set_user_attr("sampler", sampler_name)
    if config.seed is not None:
        study.set_user_attr("seed", int(config.seed))

    n_trials = n_trials_override if n_trials_override is not None else config.n_trials

    def _objective(trial: "optuna.trial.Trial") -> float:
        return _run_single_trial(
            trial,
            config=config,
            ai_worker_root=ai_worker_root,
            subprocess_timeout=subprocess_timeout,
        )

    log.info(
        "Starting study %s: mode=%s dataset=%s metric=%s sampler=%s "
        "direction=%s trials=%d",
        config.experiment_id, config.mode, config.dataset,
        config.primary_metric, sampler_name, direction, n_trials,
    )
    study.optimize(_objective, n_trials=n_trials, gc_after_trial=True)
    log.info(
        "Study %s finished. Best value=%s params=%s",
        config.experiment_id,
        study.best_value if study.best_trial else "n/a",
        study.best_params if study.best_trial else {},
    )
    return study


def active_config_fail_closed_reason(config: ActiveConfig) -> Optional[str]:
    """Return a human-readable reason when ``active.yaml`` is disabled.

    Phase 7 keeps ``eval/experiments/active.yaml`` as a schema-valid
    placeholder so tests and docs can still load it, but the file must
    not accidentally drive the older generic Optuna loop. The explicit
    ``_meta.fail_closed`` switch is the guardrail: parsing is allowed;
    starting a study is not.
    """
    if bool(config.meta.get("fail_closed")):
        return str(
            config.meta.get("reason")
            or "active.yaml is marked _meta.fail_closed=true"
        )
    status = str(config.meta.get("status") or "").strip().lower()
    if status in {"fail_closed", "fail-closed", "disabled"}:
        return f"active.yaml _meta.status={status!r}"
    return None


def assert_active_config_runnable(config: ActiveConfig) -> None:
    """Raise ``SystemExit`` if the active config is intentionally disabled."""
    reason = active_config_fail_closed_reason(config)
    if reason:
        raise SystemExit(
            "active.yaml is fail-closed and must not be used for a "
            f"tune run: {reason}"
        )


def _build_sampler(name: str, seed: Optional[int]) -> "optuna.samplers.BaseSampler":
    import optuna

    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)


# Extracted so tests can drive the objective with a fake trial.
def _run_single_trial(
    trial: "optuna.trial.Trial",
    *,
    config: ActiveConfig,
    ai_worker_root: Path,
    subprocess_timeout: Optional[float],
) -> float:
    params = suggest_params(trial, config)
    env_overrides = params_to_env(params)
    rag_top_k = int(params["rag_top_k"]) if "rag_top_k" in params else None

    t0 = time.perf_counter()
    summary, wall_ms = run_one_trial(
        mode=config.mode,
        dataset=config.dataset,
        env_overrides=env_overrides,
        rag_top_k=rag_top_k,
        ai_worker_root=ai_worker_root,
        subprocess_timeout=subprocess_timeout,
    )
    trial_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    primary = extract_metric(summary, config.primary_metric)
    secondaries: Dict[str, Optional[float]] = {}
    for metric in config.secondary_metrics:
        try:
            secondaries[metric] = extract_metric(summary, metric)
        except RuntimeError as ex:
            log.warning("Secondary metric %r unavailable: %s", metric, ex)
            secondaries[metric] = None

    trial.set_user_attr("config_hash", config_hash(params))
    trial.set_user_attr("latency_ms", round(trial_elapsed_ms, 3))
    trial.set_user_attr("eval_wall_ms", round(wall_ms, 3))
    trial.set_user_attr("secondary_metric_values", secondaries)
    # Cost accounting stays loose in Phase 1 — the harness doesn't
    # surface API spend yet, so 0.0 when Claude providers are off.
    # A later phase can thread usage stats through the eval report
    # and populate this attr with a real USD figure.
    trial.set_user_attr("cost_usd", _estimate_cost_usd(config, summary))

    if primary is None:
        # Optuna requires a finite float. Treat a None metric as
        # "worst possible" so the sampler moves away from it without
        # marking the trial as failed.
        worst = float("-inf") if config.direction == "maximize" else float("inf")
        log.warning(
            "Trial %d produced primary_metric=%s (None). Returning %s to "
            "steer the sampler away.",
            trial.number, config.primary_metric, worst,
        )
        return worst
    return primary


def _estimate_cost_usd(config: ActiveConfig, summary: Mapping[str, Any]) -> float:
    """Hook for Phase 2 cost tracking. Returns 0.0 today.

    The eval harness does not yet surface API usage; wiring Anthropic
    usage stats into the eval report is a follow-up task tracked in
    docs/optuna-tuning-plan.md Phase 5.
    """
    # TODO(phase-2): when the eval harness starts emitting
    # `usage.input_tokens` / `usage.output_tokens`, multiply by the
    # model's published price here.
    return 0.0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.tune",
        description=(
            "Drive an Optuna study over the eval harness. Reads "
            "active.yaml, runs eval.run_eval in a subprocess per trial, "
            "records metrics + latency onto each trial."
        ),
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Expected experiment_id in active.yaml. Runner fails if "
             "active.yaml's experiment_id does not match, to prevent "
             "accidental overwrites of the wrong study.",
    )
    parser.add_argument(
        "--active-yaml", type=Path,
        default=Path("eval/experiments/active.yaml"),
        help="Path to active.yaml (default: eval/experiments/active.yaml).",
    )
    parser.add_argument(
        "--studies-root", type=Path,
        default=Path("eval/experiments/studies"),
        help="Parent directory for studies/<experiment_id>/ output.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override active.yaml's optuna.n_trials (handy for smoke).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append trials to an existing study.db instead of failing.",
    )
    parser.add_argument(
        "--random-sampler", action="store_true",
        help="Force the RandomSampler for this run. The docs suggest "
             "scheduling a wide-random round every 3rd tune-round to "
             "escape TPE local optima.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Optional per-trial timeout (seconds) for the eval subprocess.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    ai_worker_root = Path.cwd()
    active_yaml = args.active_yaml
    if not active_yaml.is_absolute():
        active_yaml = ai_worker_root / active_yaml

    config = load_active_config(active_yaml)
    if config.experiment_id != args.experiment:
        raise SystemExit(
            f"active.yaml experiment_id={config.experiment_id!r} does "
            f"not match --experiment {args.experiment!r}. Refusing to "
            f"run against the wrong study."
        )

    studies_root = args.studies_root
    if not studies_root.is_absolute():
        studies_root = ai_worker_root / studies_root
    studies_root.mkdir(parents=True, exist_ok=True)

    sampler_override = "random" if args.random_sampler else None

    run_study(
        config=config,
        active_yaml_path=active_yaml,
        studies_root=studies_root,
        ai_worker_root=ai_worker_root,
        resume=args.resume,
        sampler_override=sampler_override,
        subprocess_timeout=args.timeout,
        n_trials_override=args.n_trials,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
