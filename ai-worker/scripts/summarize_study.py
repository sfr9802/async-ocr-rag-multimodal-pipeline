"""Render PNGs + a markdown summary for a finished Optuna study.

Loads the study SQLite produced by `scripts.tune`, emits matplotlib
visualizations under `plots/`, and writes `summary.md` with a metrics
table and narrative-placeholder comments that a later Claude session
fills in via the `/analyze-study` slash command.

Usage (from ai-worker/):

    python -m scripts.summarize_study --experiment rag-cheap-sweep-v1
    python -m scripts.summarize_study --experiment rag-cheap-sweep-v1 \
        --top-n 15
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

log = logging.getLogger("summarize_study")


_NARRATIVE_PLACEHOLDERS = (
    "top-trial-pattern",
    "param-importances",
    "next-direction",
)


@dataclass
class _TrialRow:
    number: int
    value: Optional[float]
    params: Dict[str, Any]
    user_attrs: Dict[str, Any]
    state: str
    duration_s: Optional[float]


def _collect_trials(study: "optuna.Study") -> List[_TrialRow]:
    rows: List[_TrialRow] = []
    for t in study.trials:
        duration = None
        if t.datetime_start and t.datetime_complete:
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        rows.append(
            _TrialRow(
                number=t.number,
                value=t.value,
                params=dict(t.params),
                user_attrs=dict(t.user_attrs),
                state=t.state.name,
                duration_s=duration,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Plot generation.
# ---------------------------------------------------------------------------


def generate_plots(
    study: "optuna.Study",
    plots_dir: Path,
) -> List[Path]:
    """Emit optimization-history, param-importances, per-param slice,
    and top-2-importance contour PNGs. Returns the paths written.

    Any single plot that raises is logged and skipped — a tiny study
    with too few trials for importance estimation should not block
    the rest of the summary.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_contour,
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    written.extend(_safe_plot(
        plots_dir / "optimization_history.png",
        lambda: plot_optimization_history(study),
    ))

    importances = _best_importances(study)
    if importances:
        written.extend(_safe_plot(
            plots_dir / "param_importances.png",
            lambda: plot_param_importances(study),
        ))

    param_names = _completed_param_names(study)
    for param in param_names:
        written.extend(_safe_plot(
            plots_dir / f"slice_{_safe_name(param)}.png",
            lambda p=param: plot_slice(study, params=[p]),
        ))

    if len(param_names) >= 2 and importances:
        top_two = [p for p, _ in importances[:2] if p in param_names]
        if len(top_two) == 2:
            written.extend(_safe_plot(
                plots_dir / f"contour_{_safe_name(top_two[0])}_{_safe_name(top_two[1])}.png",
                lambda: plot_contour(study, params=top_two),
            ))

    plt.close("all")
    return written


def _safe_plot(out_path: Path, make_ax) -> List[Path]:
    import matplotlib.pyplot as plt

    try:
        ax = make_ax()
    except Exception as ex:
        log.warning("Plot %s skipped: %s", out_path.name, ex)
        return []

    fig = _axes_figure(ax)
    if fig is None:
        log.warning("Plot %s produced no figure, skipping", out_path.name)
        return []
    try:
        fig.tight_layout()
    except Exception:
        pass  # tight_layout can fail on multi-panel figures; savefig still works
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def _axes_figure(ax: Any) -> Optional[Any]:
    """plot_slice returns ndarray; plot_optimization_history returns Axes."""
    import numpy as np

    if isinstance(ax, np.ndarray):
        flat = ax.flatten()
        if flat.size == 0:
            return None
        return flat[0].figure
    if hasattr(ax, "figure"):
        return ax.figure
    return None


def _best_importances(study: "optuna.Study") -> List[Tuple[str, float]]:
    """Try fANOVA importances; fall back to empty on failure (needs
    >= 2 completed trials and at least one moved parameter)."""
    try:
        import optuna

        imps = optuna.importance.get_param_importances(study)
        return sorted(imps.items(), key=lambda kv: kv[1], reverse=True)
    except Exception as ex:
        log.info("Param importance unavailable: %s", ex)
        return []


def _completed_param_names(study: "optuna.Study") -> List[str]:
    import optuna

    names: List[str] = []
    seen: set[str] = set()
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        for name in t.params:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


# ---------------------------------------------------------------------------
# summary.md rendering.
# ---------------------------------------------------------------------------


def render_summary_md(
    *,
    study: "optuna.Study",
    trials: Sequence[_TrialRow],
    config: Mapping[str, Any],
    plots: Sequence[Path],
    plots_dir_name: str,
    top_n: int,
) -> str:
    import optuna

    completed = [t for t in trials if t.state == "COMPLETE" and t.value is not None]
    direction = study.direction.name.lower()  # "maximize" / "minimize"
    reverse = direction == "maximize"
    completed_sorted = sorted(
        completed, key=lambda t: t.value, reverse=reverse,
    )
    top_rows = completed_sorted[:top_n]

    primary_metric = study.user_attrs.get("primary_metric", "?")
    secondary_metrics = list(study.user_attrs.get("secondary_metrics", []))

    out: List[str] = []
    out.append(f"# Study summary — `{study.study_name}`\n")
    out.append(_header_table(study, trials, config))
    out.append("")

    importances = _best_importances(study)
    if importances:
        out.append("## Parameter importances\n")
        out.append("| parameter | importance |")
        out.append("| --- | ---: |")
        for name, value in importances:
            out.append(f"| `{name}` | {value:.4f} |")
        out.append("")

    out.append(f"## Top {len(top_rows)} trials (by `{primary_metric}`)\n")
    out.append(_top_trials_table(top_rows, primary_metric, secondary_metrics))
    out.append("")

    out.append("## Best trial\n")
    out.append(_best_trial_card(study, primary_metric, secondary_metrics))
    out.append("")

    if plots:
        out.append("## Plots\n")
        for p in plots:
            rel = f"{plots_dir_name}/{p.name}"
            out.append(f"![{p.stem}]({rel})")
        out.append("")

    out.append("## Narrative (filled in by `/analyze-study`)\n")
    out.append(
        "<!-- claude-narrative:top-trial-pattern -->\n"
        "_(Claude fills: what do top trials have in common?)_\n"
    )
    out.append(
        "<!-- claude-narrative:param-importances -->\n"
        "_(Claude fills: which params mattered, which didn't?)_\n"
    )
    out.append(
        "<!-- claude-narrative:next-direction -->\n"
        "_(Claude fills: where should the next round search?)_\n"
    )
    return "\n".join(out).rstrip() + "\n"


def _header_table(
    study: "optuna.Study",
    trials: Sequence[_TrialRow],
    config: Mapping[str, Any],
) -> str:
    completed = [t for t in trials if t.state == "COMPLETE"]
    failed = [t for t in trials if t.state in ("FAIL", "PRUNED")]
    total_s = sum((t.duration_s or 0.0) for t in trials)

    started = min((t.datetime_start for t in study.trials if t.datetime_start),
                  default=None)
    finished = max((t.datetime_complete for t in study.trials if t.datetime_complete),
                   default=None)
    objective = config.get("objective", {}) if isinstance(config, dict) else {}

    lines = [
        "| field | value |",
        "| --- | --- |",
        f"| experiment_id | `{study.study_name}` |",
        f"| mode | `{objective.get('mode', '?')}` |",
        f"| dataset | `{objective.get('dataset', '?')}` |",
        f"| primary_metric | `{study.user_attrs.get('primary_metric', '?')}` |",
        f"| direction | `{study.direction.name.lower()}` |",
        f"| sampler | `{study.user_attrs.get('sampler', '?')}` |",
        f"| seed | `{study.user_attrs.get('seed', 'n/a')}` |",
        f"| trials (total / complete / failed) | {len(trials)} / {len(completed)} / {len(failed)} |",
        f"| wall-time (sum of trial durations) | {total_s:.1f} s |",
        f"| started_at | {started.isoformat(timespec='seconds') if started else 'n/a'} |",
        f"| finished_at | {finished.isoformat(timespec='seconds') if finished else 'n/a'} |",
    ]
    return "\n".join(lines)


def _top_trials_table(
    top_rows: Sequence[_TrialRow],
    primary: str,
    secondary: Sequence[str],
) -> str:
    if not top_rows:
        return "_(no completed trials yet)_"
    param_keys = sorted({k for t in top_rows for k in t.params})
    header = (
        ["#", primary]
        + [f"sec:{m}" for m in secondary]
        + ["cost_usd", "latency_ms"]
        + [f"p:{k}" for k in param_keys]
    )
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for t in top_rows:
        row: List[str] = [str(t.number), _fmt(t.value)]
        sec_values = t.user_attrs.get("secondary_metric_values") or {}
        for m in secondary:
            row.append(_fmt(sec_values.get(m)))
        row.append(_fmt(t.user_attrs.get("cost_usd")))
        row.append(_fmt(t.user_attrs.get("latency_ms")))
        for k in param_keys:
            row.append(_fmt(t.params.get(k)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _best_trial_card(
    study: "optuna.Study",
    primary: str,
    secondary: Sequence[str],
) -> str:
    if not study.best_trial:
        return "_(no best trial — study has no completed trials)_"
    best = study.best_trial
    sec_values = best.user_attrs.get("secondary_metric_values") or {}
    lines = [
        f"- trial #{best.number}",
        f"- `{primary}` = **{_fmt(best.value)}**",
    ]
    for m in secondary:
        lines.append(f"- `{m}` = {_fmt(sec_values.get(m))}")
    lines.append(f"- `config_hash` = `{best.user_attrs.get('config_hash', 'n/a')}`")
    lines.append(f"- `latency_ms` = {_fmt(best.user_attrs.get('latency_ms'))}")
    lines.append("- params:")
    for k, v in sorted(best.params.items()):
        lines.append(f"  - `{k}` = `{v}`")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.4f}" if abs(value) < 1_000 else f"{value:.1f}"
    return str(value)


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------


def summarize(
    *,
    experiment_id: str,
    studies_root: Path,
    top_n: int = 10,
) -> Path:
    import optuna

    study_dir = studies_root / experiment_id
    db_path = study_dir / "study.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"No study.db at {db_path}. Run scripts.tune first."
        )
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=experiment_id, storage=storage_url)

    frozen_config_path = study_dir / "config.yaml"
    frozen_config: Dict[str, Any] = {}
    if frozen_config_path.exists():
        loaded = yaml.safe_load(frozen_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            frozen_config = loaded

    plots_dir = study_dir / "plots"
    plots = generate_plots(study, plots_dir)
    trials = _collect_trials(study)
    md = render_summary_md(
        study=study,
        trials=trials,
        config=frozen_config,
        plots=plots,
        plots_dir_name="plots",
        top_n=top_n,
    )
    summary_path = study_dir / "summary.md"
    summary_path.write_text(md, encoding="utf-8")
    log.info(
        "Wrote %s (%d trials, %d plots)",
        summary_path, len(trials), len(plots),
    )
    return summary_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.summarize_study",
        description=(
            "Render PNGs + summary.md for an Optuna study at "
            "eval/experiments/studies/<experiment_id>/study.db."
        ),
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--studies-root", type=Path,
        default=Path("eval/experiments/studies"),
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    studies_root = args.studies_root
    if not studies_root.is_absolute():
        studies_root = Path.cwd() / studies_root

    summarize(
        experiment_id=args.experiment,
        studies_root=studies_root,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
