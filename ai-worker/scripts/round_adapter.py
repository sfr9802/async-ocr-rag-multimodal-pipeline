"""Project-side adapter for the `optuna-round-refinement` skill (v0.1.0).

The skill defines schemas, prompts, and templates for a round-level
LLM-in-the-outer-loop workflow. It explicitly delegates four functions to
the project (§7 of the skill's SKILL.md):

    1. export_study_bundle(study)        -> dict  (this module)
    2. apply_next_round_config(cfg)      -> new active.yaml contents
    3. render_llm_input(bundle, out_path) -> markdown brief for the LLM
    4. validate_and_hash(cfg_path)       -> sha256 hex (jsonschema-validated)

This module implements those four functions as a single file + a CLI with
four subcommands. It does NOT replace `scripts.tune`; it sits next to it
and bridges `studies/<experiment_id>/study.db` (owned by tune) and
`studies/<experiment_id>/round_NN_config.json` (owned by the skill).

Usage (from ai-worker/):

    # Export bundle from a completed study.
    python -m scripts.round_adapter export-bundle \\
        --experiment rag-cheap-sweep-v3

    # Render the bundle as a markdown brief for the LLM analyst.
    python -m scripts.round_adapter render-input \\
        --bundle eval/experiments/studies/rag-cheap-sweep-v3/round_01_bundle.json

    # jsonschema + sha256 of a config or bundle.
    python -m scripts.round_adapter validate \\
        --kind next_round_config \\
        --path eval/experiments/studies/rag-cheap-sweep-v3/round_02_config.json

    # Preview the active.yaml mutation a round_NN_config.json would cause.
    python -m scripts.round_adapter apply \\
        --config eval/experiments/studies/rag-cheap-sweep-v3/round_02_config.json
    # add --write to actually overwrite eval/experiments/active.yaml
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import statistics as stats
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

log = logging.getLogger("round_adapter")


# ---------------------------------------------------------------------------
# Skill-dir discovery
# ---------------------------------------------------------------------------


_SKILL_NAME = "optuna-round-refinement"
_SKILL_ENV_VAR = "OPTUNA_ROUND_REFINEMENT_HOME"


def find_skill_dir() -> Path:
    """Locate the installed optuna-round-refinement skill.

    Search order:
      1. $OPTUNA_ROUND_REFINEMENT_HOME (explicit override)
      2. ~/.claude/skills/optuna-round-refinement  (user-level)
      3. .claude/skills/optuna-round-refinement    (project-local)

    Raises FileNotFoundError with all attempted paths on miss.
    """
    candidates: List[Path] = []
    override = os.environ.get(_SKILL_ENV_VAR)
    if override:
        candidates.append(Path(override))
    candidates.append(Path.home() / ".claude" / "skills" / _SKILL_NAME)
    candidates.append(Path(".claude") / "skills" / _SKILL_NAME)

    for c in candidates:
        if c and (c / "SKILL.md").exists():
            return c
    raise FileNotFoundError(
        f"{_SKILL_NAME} skill not found. Checked: "
        + ", ".join(str(p) for p in candidates)
        + f". Install via `git clone https://github.com/sfr9802/{_SKILL_NAME} "
          f"~/.claude/skills/{_SKILL_NAME}` or set ${_SKILL_ENV_VAR}."
    )


def _load_schema(skill_dir: Path, kind: str) -> Dict[str, Any]:
    fname = {
        "study_bundle": "study_bundle.schema.json",
        "next_round_config": "next_round_config.schema.json",
    }[kind]
    return json.loads((skill_dir / "schemas" / fname).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Sorted-keys, minimal-separator JSON for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# (1) export_study_bundle
# ---------------------------------------------------------------------------


def export_study_bundle(
    *,
    experiment_id: str,
    studies_root: Path,
    round_number: int = 1,
    parent_config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Read an Optuna study.db + frozen config.yaml and produce a bundle
    dict conforming to `schemas/study_bundle.schema.json`.

    The caller is responsible for persisting the bundle + computing its
    hash via `_canonical_json` + `_sha256_hex`. `parent_config_hash` is
    sha256 of the frozen config.yaml if not supplied.
    """
    import optuna

    study_dir = studies_root / experiment_id
    db_path = study_dir / "study.db"
    cfg_path = study_dir / "config.yaml"
    if not db_path.exists():
        raise FileNotFoundError(f"study.db not found at {db_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=experiment_id, storage=storage_url)

    frozen_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(frozen_cfg, dict):
        raise ValueError(f"config.yaml at {cfg_path} is not a mapping")

    if parent_config_hash is None:
        parent_config_hash = _sha256_hex(cfg_path.read_bytes())

    trials = [_trial_summary(t) for t in study.trials]
    completed = [
        t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None
    ]

    search_space_out = _search_space_from_config(frozen_cfg)
    fixed_params = dict(frozen_cfg.get("fixed_params") or {})
    boundary_hits = _boundary_hits(completed, search_space_out)
    values = [float(t["value"]) for t in completed]

    try:
        importances = optuna.importance.get_param_importances(study)
        importances_out = {k: round(float(v), 6) for k, v in importances.items()}
    except Exception as ex:  # pragma: no cover — param-importance can fail on degenerate studies
        log.info("param-importance unavailable: %s", ex)
        importances_out = {}

    clusters = _autocluster_by_value(completed)

    bundle: Dict[str, Any] = {
        "schema_version": "1.0",
        "round_id": f"round_{round_number:02d}",
        "study_id": experiment_id,
        "parent_config_hash": parent_config_hash,
        "optuna": {
            "version": optuna.__version__,
            "sampler": {
                "type": _sampler_type(frozen_cfg),
                "params": {"multivariate": True},
                "seed": frozen_cfg.get("optuna", {}).get("seed"),
            },
            "pruner": {"type": "NopPruner", "params": {}},
        },
        "objective": {
            "name": frozen_cfg.get("objective", {}).get("primary_metric", "?"),
            "direction": frozen_cfg.get("optuna", {}).get("direction", "maximize"),
        },
        "search_space": search_space_out,
        "n_trials": len(trials),
        "trials": trials,
        "best_trial": _trial_summary(study.best_trial) if study.best_trial else None,
        "statistics": {
            "n_complete": len(completed),
            "n_pruned": sum(1 for t in trials if t["state"] == "PRUNED"),
            "n_failed": sum(1 for t in trials if t["state"] == "FAIL"),
            "best_value": study.best_value if study.best_trial else None,
            "median_value": stats.median(values) if values else None,
            "mean_value": stats.mean(values) if values else None,
            "std_value": stats.pstdev(values) if len(values) > 1 else 0.0,
            "quantiles": _quantiles(values),
            "boundary_hits": boundary_hits,
        },
    }
    if fixed_params:
        bundle["fixed_params"] = fixed_params
    if importances_out:
        bundle["param_importances"] = importances_out
    if clusters:
        bundle["clusters"] = clusters

    notes = (frozen_cfg.get("_meta") or {}).get("notes")
    if isinstance(notes, str) and notes.strip():
        # Strip to one line per the schema's "no raw data" guidance.
        bundle["notes"] = notes.strip().splitlines()[0][:240]

    return bundle


def _trial_summary(t: "optuna.trial.FrozenTrial") -> Dict[str, Any]:
    params = {k: (bool(v) if isinstance(v, bool) else v) for k, v in t.params.items()}
    user_attrs = {
        k: v
        for k, v in t.user_attrs.items()
        if k in {"config_hash", "latency_ms", "eval_wall_ms",
                 "secondary_metric_values", "cost_usd"}
    }
    return {
        "number": t.number,
        "state": t.state.name,
        "value": float(t.value) if t.value is not None else None,
        "params": params,
        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        "user_attrs": user_attrs,
    }


def _search_space_from_config(frozen_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw_space = frozen_cfg.get("search_space", {}) or {}
    out: Dict[str, Any] = {}
    for name, spec in raw_space.items():
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("type", "")).lower()
        if kind in ("int", "float"):
            out[name] = {"type": kind, "low": spec["low"], "high": spec["high"]}
            if spec.get("step") is not None:
                out[name]["step"] = spec["step"]
            if spec.get("log"):
                out[name]["log"] = True
        elif kind == "loguniform":
            out[name] = {
                "type": "float",
                "low": spec["low"],
                "high": spec["high"],
                "log": True,
            }
        elif kind == "categorical":
            out[name] = {"type": "categorical", "choices": list(spec["choices"])}
    return out


def _boundary_hits(
    completed: List[Dict[str, Any]],
    search_space: Mapping[str, Any],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for name, spec in search_space.items():
        if spec.get("type") not in ("int", "float"):
            continue
        low = spec["low"]
        high = spec["high"]
        out[name] = {
            "low":  sum(1 for t in completed if t["params"].get(name) == low),
            "high": sum(1 for t in completed if t["params"].get(name) == high),
        }
    return out


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _q(q: float) -> float:
        idx = min(int(q * (n - 1)), n - 1)
        return sorted_vals[idx]

    return {
        "p10": _q(0.10),
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "p90": _q(0.90),
    }


def _sampler_type(frozen_cfg: Mapping[str, Any]) -> str:
    name = str(frozen_cfg.get("optuna", {}).get("sampler", "tpe")).lower()
    if name == "random":
        return "RandomSampler"
    return "TPESampler"


def _autocluster_by_value(completed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group completed trials by identical value.

    When a study plateaus (as in rag-cheap-sweep-v3), the flat-value
    cluster is the single most useful structural signal — it tells the
    LLM analyst which axes actually move the metric. We don't do proper
    k-means here; identical-value grouping is honest about what we know.
    """
    if len(completed) < 2:
        return []
    by_value: Dict[float, List[int]] = {}
    for t in completed:
        by_value.setdefault(t["value"], []).append(t["number"])
    # Only emit clusters that have >= 2 members; singletons aren't useful.
    clusters: List[Dict[str, Any]] = []
    for value, numbers in sorted(by_value.items(), reverse=True):
        if len(numbers) < 2:
            continue
        clusters.append({
            "label": f"value={value:.4f} plateau ({len(numbers)} trials)",
            "trial_numbers": sorted(numbers),
            "center": {"value": value, "size": len(numbers)},
        })
    return clusters


# ---------------------------------------------------------------------------
# (2) render_llm_input — lightweight direct renderer (no Mustache engine)
# ---------------------------------------------------------------------------


def render_llm_input(bundle: Mapping[str, Any], out_path: Path) -> Path:
    """Render the bundle as a markdown brief for the LLM analyst.

    The skill's `templates/llm_input.md` is Mustache-shaped; implementing
    a full engine is overkill for three placeholders. We emit markdown
    with the same section structure the template prescribes, so LLM
    prompts keyed on those section headers still work.
    """
    bundle_hash = _sha256_hex(_canonical_json(bundle))
    stats_section = bundle.get("statistics", {}) or {}
    quantiles = stats_section.get("quantiles", {}) or {}
    search_space = bundle.get("search_space", {}) or {}
    fixed_params = bundle.get("fixed_params", {}) or {}
    importances = bundle.get("param_importances", {}) or {}
    boundary_hits = stats_section.get("boundary_hits", {}) or {}
    clusters = bundle.get("clusters", []) or []
    completed = [t for t in bundle.get("trials", []) if t.get("state") == "COMPLETE"]
    top_k = 10
    reverse = bundle.get("objective", {}).get("direction") == "maximize"
    top_trials = sorted(
        completed,
        key=lambda t: (t.get("value") if t.get("value") is not None else float("-inf")),
        reverse=reverse,
    )[:top_k]

    lines: List[str] = []
    lines.append(f"# Round {bundle['round_id']} — study bundle\n")
    lines.append(f"**Study:** `{bundle['study_id']}`")
    lines.append(
        f"**Objective:** `{bundle['objective']['name']}` "
        f"({bundle['objective']['direction']})"
    )
    lines.append(
        f"**Optuna:** `{bundle['optuna']['version']}` | "
        f"sampler `{bundle['optuna']['sampler']['type']}` | "
        f"pruner `{bundle['optuna']['pruner']['type']}`"
    )
    lines.append(
        f"**Parent config hash:** `{bundle.get('parent_config_hash') or '(none — initial round)'}`"
    )
    lines.append(f"**Bundle hash:** `{bundle_hash}`\n")
    lines.append("---\n")

    lines.append("## 1. Frozen search space (this round)\n")
    lines.append("| Param | Type | Range / choices |")
    lines.append("|-------|------|-----------------|")
    for name, spec in search_space.items():
        if spec.get("type") == "categorical":
            rng = ", ".join(repr(c) for c in spec["choices"])
        else:
            lo = spec.get("low")
            hi = spec.get("high")
            extras = []
            if spec.get("step") is not None:
                extras.append(f"step={spec['step']}")
            if spec.get("log"):
                extras.append("log")
            extra = f" ({', '.join(extras)})" if extras else ""
            rng = f"[{lo}, {hi}]{extra}"
        lines.append(f"| `{name}` | {spec.get('type')} | {rng} |")
    lines.append("")
    if fixed_params:
        lines.append(f"Fixed params: `{json.dumps(fixed_params, ensure_ascii=False)}`\n")

    lines.append("## 2. Headline statistics\n")
    lines.append(
        f"- Trials: **{bundle['n_trials']}** "
        f"(complete {stats_section.get('n_complete', 0)}, "
        f"pruned {stats_section.get('n_pruned', 0)}, "
        f"failed {stats_section.get('n_failed', 0)})"
    )
    lines.append(f"- Best value: **{stats_section.get('best_value')}**")
    if quantiles:
        lines.append(
            f"- Quantiles: p10 `{quantiles.get('p10')}`, "
            f"p50 `{quantiles.get('p50')}`, p90 `{quantiles.get('p90')}`"
        )
    lines.append(
        f"- Mean ± std: `{stats_section.get('mean_value')} ± {stats_section.get('std_value')}`\n"
    )

    lines.append("## 3. Param importances\n")
    if importances:
        lines.append("| Param | Importance |")
        lines.append("|-------|-----------|")
        for name, value in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"| `{name}` | {value:.4f} |")
    else:
        lines.append("_No importances computed this round._")
    lines.append("")

    lines.append("## 4. Boundary hits\n")
    if boundary_hits:
        for name, hits in boundary_hits.items():
            lines.append(f"- `{name}`: low={hits.get('low', 0)}, high={hits.get('high', 0)}")
    else:
        lines.append("_No boundary hits recorded (no int/float axes, or space is all categorical)._")
    lines.append("")

    lines.append("## 5. Best trial\n")
    if bundle.get("best_trial"):
        lines.append("```json")
        lines.append(json.dumps(bundle["best_trial"], indent=2, ensure_ascii=False))
        lines.append("```\n")
    else:
        lines.append("_No best trial (no completed trials)._\n")

    lines.append(f"## 6. Top-k trials (k={len(top_trials)})\n")
    lines.append("| # | value | params |")
    lines.append("|---|-------|--------|")
    for t in top_trials:
        lines.append(
            f"| {t['number']} | {t.get('value')} | `{json.dumps(t['params'], ensure_ascii=False)}` |"
        )
    lines.append("")

    lines.append("## 7. Clusters\n")
    if clusters:
        for c in clusters:
            numbers = ", ".join(str(n) for n in c["trial_numbers"])
            lines.append(f"- **{c['label']}** — trials {numbers}")
    else:
        lines.append("_No clusters provided._")
    lines.append("")

    lines.append("## 8. Operator notes\n")
    lines.append(bundle.get("notes") or "(none)")
    lines.append("\n---\n")
    lines.append("## Your task\n")
    lines.append(
        "You are the outer-loop analyst. Produce a round report and a next-round "
        "config JSON per the skill's `prompts/claude_code/propose_next_round.md`. "
        "Every search_space change MUST cite a field of this bundle."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# (3) validate_and_hash
# ---------------------------------------------------------------------------


def validate_and_hash(path: Path, *, kind: str, skill_dir: Optional[Path] = None) -> str:
    """jsonschema-validate a bundle or next-round-config and return the
    sha256 of its canonical JSON.

    `kind` is one of 'study_bundle' | 'next_round_config'. Raises
    `jsonschema.ValidationError` on failure; the CLI converts to a
    human-readable exit.
    """
    import jsonschema

    skill_dir = skill_dir or find_skill_dir()
    schema = _load_schema(skill_dir, kind)
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    return _sha256_hex(_canonical_json(payload))


# ---------------------------------------------------------------------------
# (4) apply_next_round_config
# ---------------------------------------------------------------------------


_EXPERIMENT_VERSION_RE = re.compile(r"^(?P<prefix>.*?)(?:-v(?P<n>\d+))?$")


def bump_experiment_id(current: str) -> str:
    """'rag-cheap-sweep-v3' -> 'rag-cheap-sweep-v4'.

    If the current id has no trailing `-v<int>`, append `-v2`.
    """
    m = _EXPERIMENT_VERSION_RE.match(current)
    if not m:
        return f"{current}-v2"
    prefix = m.group("prefix")
    n = m.group("n")
    if n is None:
        return f"{prefix}-v2"
    return f"{prefix}-v{int(n) + 1}"


def apply_next_round_config(
    *,
    config_path: Path,
    active_yaml_path: Path,
    skill_dir: Optional[Path] = None,
    bump_experiment: bool = True,
) -> Dict[str, Any]:
    """Convert a `round_NN_config.json` into the new `active.yaml` body.

    Does NOT write to disk — returns the new yaml content as a dict so
    the CLI / caller can preview first. The returned dict preserves
    keys the skill config doesn't own (objective, direction) from the
    current active.yaml.

    Raises ValueError on incompatible configs (e.g. non-TPE/Random
    samplers the project does not support, or non-NopPruner pruners).
    """
    skill_dir = skill_dir or find_skill_dir()

    # Schema-validate both sides before merging so we never produce a
    # half-applied active.yaml on a broken input.
    validate_and_hash(config_path, kind="next_round_config", skill_dir=skill_dir)
    next_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    current = yaml.safe_load(active_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        raise ValueError(f"active.yaml at {active_yaml_path} is not a mapping")

    # SKILL.md §8.1: reject any config whose provenance.source_bundle_hash
    # does not match the on-disk bundle it claims to derive from. Path
    # convention: the config sits next to its source bundle in
    # studies/<source_exp>/round_NN_{bundle,config}.json.
    src_round = next_cfg["provenance"].get("source_round_id")
    claimed_hash = next_cfg["provenance"].get("source_bundle_hash")
    if src_round and claimed_hash:
        bundle_path = config_path.parent / f"{src_round}_bundle.json"
        if bundle_path.exists():
            on_disk = json.loads(bundle_path.read_text(encoding="utf-8"))
            actual = _sha256_hex(_canonical_json(on_disk))
            if actual != claimed_hash:
                raise ValueError(
                    f"provenance.source_bundle_hash={claimed_hash[:12]}... "
                    f"does not match on-disk {bundle_path.name} "
                    f"hash {actual[:12]}... (SKILL.md §8.1)"
                )
        else:
            log.warning(
                "source bundle %s not on disk; skipping hash cross-check. "
                "This should only happen outside the adapter-produced flow.",
                bundle_path,
            )

    sampler_type = next_cfg["sampler"]["type"]
    if sampler_type not in ("TPESampler", "RandomSampler"):
        raise ValueError(
            f"Unsupported sampler {sampler_type!r}. tune.py only wires "
            f"TPESampler and RandomSampler today."
        )
    pruner_type = next_cfg["pruner"]["type"]
    if pruner_type != "NopPruner":
        log.warning(
            "pruner=%s is not wired in tune.py; it will be ignored at run time.",
            pruner_type,
        )

    new: Dict[str, Any] = copy.deepcopy(current)
    if bump_experiment:
        new["experiment_id"] = bump_experiment_id(str(current.get("experiment_id", "")))
    # `objective` block is preserved verbatim — the skill config doesn't
    # own it. primary_metric / secondary_metrics / mode / dataset all
    # stay the same round-to-round unless a human edits them by hand.
    new["search_space"] = _search_space_to_active_yaml(next_cfg["search_space"])
    new["fixed_params"] = dict(next_cfg.get("fixed_params") or {})
    new["optuna"] = {
        "sampler": "random" if sampler_type == "RandomSampler" else "tpe",
        "n_trials": int(next_cfg["n_trials"]),
        "seed": next_cfg["sampler"].get("seed"),
        "direction": current.get("optuna", {}).get("direction", "maximize"),
    }
    # Strip the search_space keys the new round froze.
    if "dataset" in new["fixed_params"]:
        # The dataset belongs in objective, not fixed_params — the
        # skill's fixed_params schema permits it but tune.py reads it
        # from objective.dataset. Migrate silently.
        new.setdefault("objective", {})["dataset"] = new["fixed_params"].pop("dataset")

    provenance = next_cfg["provenance"]
    new["_meta"] = {
        "created_by": "claude-proposed",
        "parent_experiment_id": current.get("experiment_id"),
        "source_bundle_hash": provenance.get("source_bundle_hash"),
        "parent_config_hash": provenance.get("parent_config_hash"),
        "generated_at": provenance.get("generated_at"),
        "generated_by": dict(provenance.get("generated_by", {})),
        "rationale": provenance.get("rationale"),
        "notes": (
            "Auto-generated from "
            f"{config_path.name}. Flip created_by to 'claude-approved' "
            "after reviewing diff_summary entries against the bundle."
        ),
    }
    return new


def _search_space_to_active_yaml(
    schema_space: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in schema_space.items():
        if spec.get("type") == "categorical":
            out[name] = {"type": "categorical", "choices": list(spec["choices"])}
        else:
            entry: Dict[str, Any] = {
                "type": spec["type"],
                "low": spec["low"],
                "high": spec["high"],
            }
            if spec.get("step") is not None:
                entry["step"] = spec["step"]
            if spec.get("log"):
                entry["log"] = True
            out[name] = entry
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    studies_root = args.studies_root.resolve()
    bundle = export_study_bundle(
        experiment_id=args.experiment,
        studies_root=studies_root,
        round_number=args.round,
    )
    out_path = studies_root / args.experiment / f"round_{args.round:02d}_bundle.json"
    out_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    bundle_hash = _sha256_hex(_canonical_json(bundle))
    print(f"bundle:      {out_path}")
    print(f"bundle_hash: {bundle_hash}")
    print(
        f"trials:      {bundle['n_trials']} "
        f"(complete={bundle['statistics']['n_complete']})"
    )
    return 0


def _cmd_render_input(args: argparse.Namespace) -> int:
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    out = args.out or args.bundle.with_name(args.bundle.stem + "_llm_input.md")
    render_llm_input(bundle, out)
    print(f"wrote: {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import jsonschema

    try:
        digest = validate_and_hash(args.path, kind=args.kind)
    except jsonschema.ValidationError as ex:
        print(f"INVALID: {ex.message}", file=sys.stderr)
        if ex.absolute_path:
            print(f"  at: {'/'.join(str(p) for p in ex.absolute_path)}", file=sys.stderr)
        return 2
    print(f"VALID: {args.path}")
    print(f"sha256: {digest}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    new = apply_next_round_config(
        config_path=args.config,
        active_yaml_path=args.active_yaml,
        bump_experiment=not args.keep_experiment_id,
    )
    as_yaml = yaml.safe_dump(
        new, sort_keys=False, allow_unicode=True, default_flow_style=False,
    )
    if args.write:
        args.active_yaml.write_text(as_yaml, encoding="utf-8")
        print(f"WROTE: {args.active_yaml}")
        print(f"new experiment_id: {new['experiment_id']}")
    else:
        print("PREVIEW (no write - pass --write to apply):")
        print("-" * 72)
        print(as_yaml.rstrip())
        print("-" * 72)
        print(f"target experiment_id: {new['experiment_id']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Windows consoles default to cp949 (ko-KR) and blow up on the
    # em-dashes / Korean notes that frequently appear in yaml output.
    # Force UTF-8 when we can — stdout.reconfigure is Python 3.7+.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass  # pragma: no cover — pipe / redirect cases

    parser = argparse.ArgumentParser(
        prog="scripts.round_adapter",
        description=(
            "Project-side adapter for the optuna-round-refinement skill. "
            "Bridges studies/<exp>/study.db and studies/<exp>/round_NN_config.json."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG logs.",
    )

    subs = parser.add_subparsers(dest="cmd", required=True)

    p_export = subs.add_parser("export-bundle", help="study.db -> round_NN_bundle.json")
    p_export.add_argument("--experiment", required=True)
    p_export.add_argument("--round", type=int, default=1)
    p_export.add_argument(
        "--studies-root", type=Path, default=Path("eval/experiments/studies"),
    )
    p_export.set_defaults(func=_cmd_export_bundle)

    p_render = subs.add_parser("render-input", help="round_NN_bundle.json -> llm_input.md")
    p_render.add_argument("--bundle", type=Path, required=True)
    p_render.add_argument("--out", type=Path, default=None)
    p_render.set_defaults(func=_cmd_render_input)

    p_validate = subs.add_parser(
        "validate",
        help="jsonschema-validate a bundle or next-round-config + print sha256",
    )
    p_validate.add_argument("--path", type=Path, required=True)
    p_validate.add_argument(
        "--kind", choices=("study_bundle", "next_round_config"), required=True,
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_apply = subs.add_parser(
        "apply",
        help="Preview or write the active.yaml mutation from a round_NN_config.json",
    )
    p_apply.add_argument("--config", type=Path, required=True)
    p_apply.add_argument(
        "--active-yaml", type=Path, default=Path("eval/experiments/active.yaml"),
    )
    p_apply.add_argument(
        "--write", action="store_true",
        help="Actually overwrite active.yaml (default is preview-only).",
    )
    p_apply.add_argument(
        "--keep-experiment-id", action="store_true",
        help="Don't bump experiment_id (use with care — breaks study.db isolation).",
    )
    p_apply.set_defaults(func=_cmd_apply)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
