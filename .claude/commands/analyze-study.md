---
name: analyze-study
description: Fill in the Claude narrative placeholders in summary.md for a tuned study.
---

# /analyze-study

Interpret one finished Optuna study and fill the narrative placeholders
in its `summary.md`. **This is the LLM-as-interpreter step.** Optuna
already did the numerical work — your job here is to read the study +
metrics table and write paragraphs a human can skim.

## Arguments

- `experiment` (required) — the `experiment_id`. Loads `ai-worker/eval/experiments/studies/<experiment>/`.

## Steps

1. **Load everything.**
   - Read `eval/experiments/studies/<experiment>/summary.md`.
   - Read `eval/experiments/studies/<experiment>/config.yaml` (frozen snapshot of active.yaml).
   - Load the study programmatically if you need richer detail:
     ```python
     import optuna
     study = optuna.load_study(
         study_name="<experiment>",
         storage=f"sqlite:///eval/experiments/studies/<experiment>/study.db",
     )
     ```
     `study.trials` is the raw record. Prefer the summary table first; drop into the DB only if the table is insufficient.

2. **Confirm the placeholders exist.**
   - Grep the summary.md for the three narrative markers:
     - `<!-- claude-narrative:top-trial-pattern -->`
     - `<!-- claude-narrative:param-importances -->`
     - `<!-- claude-narrative:next-direction -->`
   - If any are missing or already filled, ask the user whether to overwrite or append.

3. **Fill each placeholder in turn.**

   **top-trial-pattern** — 2-4 sentences:
   - What do the top-5 trials have in common? Parameter ranges, any obvious cluster?
   - Is the best metric meaningfully above the median of completed trials, or is it noise?
   - Mention any completed trial that hit a clear ceiling (primary metric = 1.0 with no tie-breaker).

   **param-importances** — 2-4 sentences:
   - Which params have importance > 0.3? Which are effectively zero?
   - Are there interactions visible in the contour plot? (Top-2 contour PNG is linked in the summary.)
   - Call out any param whose sampled range was so narrow that importance is uninformative.

   **next-direction** — 2-4 sentences:
   - Which bound should narrow (importance high, best region concentrated)?
   - Which bound should expand (best value lands at the edge)?
   - Which param should be dropped (importance ~0, wasted search)?
   - Mention anything the current search space canNOT answer (e.g. "chunk size is out of scope until Phase 5").

4. **Edit summary.md.**
   - Replace each `_(Claude fills: …)_` italic stub with your prose. Keep the HTML comment markers intact so future `/analyze-study` calls can locate them.
   - NEVER touch the metrics tables, plots section, or header table. Only the narrative blocks.

5. **Commit.**
   - Commit only `summary.md` with message:
     ```
     Analyze study <experiment>: narrative for round <n>
     ```
   - DO NOT touch `study.db`, `plots/`, or `active.yaml` in this command.

## Guardrails

- Base every claim on the actual study. If `param_importances` is empty (too few trials), say so explicitly instead of fabricating importances.
- If the primary metric is 1.0 across the top 5 trials, point out that the metric is saturated and a harder dataset is the right next step, not further tuning.
- This command NEVER emits a diff against `active.yaml` — that is `/propose-next`'s job.
