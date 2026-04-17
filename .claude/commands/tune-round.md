---
name: tune-round
description: Run one Optuna tuning round against the active.yaml search space, then render summary.md + plots.
---

# /tune-round

Run one Optuna tuning round. This is the **numerical** step of the
loop — Optuna samples, `eval.run_eval` scores, metrics + latency land
on each trial. Narrative analysis is done separately via `/analyze-study`.

## Arguments (parse from `$ARGUMENTS`, fall back to prompting)

- `experiment` (required) — must match `experiment_id` in `ai-worker/eval/experiments/active.yaml`.
- `n_trials` (optional, default from `active.yaml`).
- `random_sampler` (optional boolean) — pass `--random-sampler` to force `RandomSampler` for this round. **Rule: every third round MUST be a random-sampler round** to escape TPE local optima. Before starting, count existing studies under `ai-worker/eval/experiments/studies/<experiment>/` + completed rounds — if the count implies this is round 3, 6, 9, …, inject `--random-sampler` automatically and say so in your kickoff message.

## Steps

1. **Sanity-check active.yaml.**
   - Read `ai-worker/eval/experiments/active.yaml`.
   - Confirm `experiment_id` matches the arg.
   - Echo `search_space`, `objective.mode`, `objective.primary_metric`, and `_meta.created_by` back to the user in a short summary before starting.

2. **Run the tune script.**
   - Working directory: `ai-worker/`.
   - Command:
     ```
     python -m scripts.tune \
         --experiment <experiment> \
         [--n-trials <n>] \
         [--random-sampler]
     ```
   - Stream the output. If Optuna refuses to overwrite an existing `study.db` and the user clearly wants to resume, re-invoke with `--resume` after confirming.

3. **Render the summary.**
   - ```
     python -m scripts.summarize_study --experiment <experiment>
     ```
   - This produces `eval/experiments/studies/<experiment>/plots/*.png` and `summary.md` with Claude narrative placeholders left unfilled.

4. **Commit the durable artifacts.**
   - Commit: `eval/experiments/studies/<experiment>/summary.md` + `config.yaml`.
   - DO NOT commit: `study.db` or `plots/` — both are `.gitignore`d. Deterministic replay only needs the YAML + seed + dataset.
   - Commit message template:
     ```
     Tune round for <experiment> (<n> trials, sampler=<tpe|random>)

     Best <primary_metric>=<value> at trial #<n>.
     ```

5. **Hand off.**
   - Tell the user: next step is `/analyze-study <experiment>` to fill in the narrative, then `/propose-next` to draft the next `active.yaml` diff.

## Guardrails

- NEVER `Edit` `active.yaml` inside this command. This is the numerical round only.
- If the eval subprocess fails repeatedly (>3 trials in a row), stop, print the stderr tail, and ask the user before continuing.
- If `--random-sampler` is injected because of the "every 3rd round" rule, state that explicitly at kickoff so the user can override.
