---
name: propose-next
description: Emit a proposed new active.yaml as a chat diff (never auto-applied).
---

# /propose-next

Draft the next `active.yaml` based on a filled `summary.md`. **Output
ONLY a unified diff in the chat. NEVER call `Edit`, `Write`, or any
file-mutating tool against `active.yaml` in this command.** The human
reviews, edits, and applies the diff manually.

## Arguments

- `experiment` (required) — the experiment whose `summary.md` you are basing the proposal on.

## Steps

1. **Load inputs.**
   - Read `ai-worker/eval/experiments/active.yaml` (current state).
   - Read `ai-worker/eval/experiments/studies/<experiment>/summary.md` (must already have filled narrative — if the placeholders are still stubs, stop and ask the user to run `/analyze-study` first).
   - Optionally load the study.db for trial-level detail:
     ```python
     import optuna
     study = optuna.load_study(
         study_name="<experiment>",
         storage=f"sqlite:///eval/experiments/studies/<experiment>/study.db",
     )
     ```

2. **Draft the new search space.**

   Apply **one change per parameter** unless two are tightly coupled:
   - **Narrow** a bound when importance is high AND the best region is clearly concentrated in one half of the current range.
   - **Expand** a bound when the best trial lands at the extreme edge.
   - **Drop** a param whose importance is effectively zero.
   - **Promote** a param from TODO comments to active search when the narrative calls out that it blocked progress.
   - Bump `experiment_id` to the next version (`rag-cheap-sweep-v1` → `rag-cheap-sweep-v2`).
   - Bump `_meta.created_by` to `claude-proposed`. The human changes it to `claude-approved` on accept.

3. **Emit a unified diff.**
   - Format: fenced `diff` block showing the before/after of `active.yaml`. Include 3 lines of context per hunk.
   - DO NOT include the trailing newline changes or formatting-only churn.
   - Example:
     ```diff
     --- a/ai-worker/eval/experiments/active.yaml
     +++ b/ai-worker/eval/experiments/active.yaml
     @@ ...
     -experiment_id: rag-cheap-sweep-v1
     +experiment_id: rag-cheap-sweep-v2
     @@ ...
     -  rag_top_k:
     -    type: int
     -    low: 3
     -    high: 15
     +  rag_top_k:
     +    type: int
     +    low: 5
     +    high: 10
     @@ ...
     -  created_by: human
     +  created_by: claude-proposed
     ```

4. **Justify every change.**
   - Below the diff, emit a bullet list, one bullet per modified parameter:
     - `rag_top_k`: narrowed [3,15] → [5,10]. Top-5 trials all landed in this band; param_importance=0.62, best value at rag_top_k=7.
     - `cross_modal_rrf_k`: dropped. Importance=0.02 over 50 trials; no visible effect on hit@5.
   - If you are expanding a bound, explicitly name which current trial sat at the edge and what metric it achieved.

5. **Offer next commands.**
   - Tell the user: "Review the diff, apply it manually, flip `_meta.created_by` to `claude-approved`, then run `/tune-round <new-experiment-id>`."

## Guardrails

- **Output a diff in chat. NEVER call Edit or Write against active.yaml.** The human applies it.
- Do not bump `optuna.n_trials` unless the narrative explicitly asks to shrink the budget.
- Do not remove the Phase 2 TODO comments in active.yaml — they are the backlog for expensive-param support.
- Every numeric change must cite either a trial number or a param importance. No vibes-based bounds.
- If the narrative says the metric is saturated, respond with "recommend changing the dataset, not the search space" and do NOT emit a diff.
