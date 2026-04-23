# Optuna + Claude-interpreter Tuning Loop — Plan

Implementation roadmap for a local hyperparameter tuning pipeline where
Optuna drives numerical search and Claude (in subsequent Claude Code
sessions) acts as the interpreter — reading study results, writing
narrative summaries, and proposing next-round search spaces as YAML
diffs for human review.

**Core principle:** LLM is an interpreter, not a sampler. Optuna picks
trial parameters; Claude analyzes outcomes and proposes refinements.
Every mutation of `active.yaml` passes through human approval.

## Phase 1 — Core Optuna driver (cheap params only)

- `ai-worker/scripts/tune.py` — YAML-driven Optuna runner wrapping the
  existing eval harness (`eval.run_eval`).
- `ai-worker/eval/experiments/active.yaml` — search space declaration
  with `experiment_id`, `objective`, `search_space`, `optuna`, and
  `_meta` sections.
- Target parameters (no re-embedding required):
  - `rag_top_k`
  - `rrf_k` (cross-modal)
  - `max_query_chars`, `short_query_words`, `max_fused_chunk_chars`
  - `excerpt_chars`
  - `ocr_min_confidence_warn`
  - `multimodal_max_vision_pages`
- Per-trial `trial.set_user_attr` for `cost_usd`, `latency_ms`,
  `secondary_metric_values`, `config_hash`.
- SQLite study storage at
  `ai-worker/eval/experiments/studies/<exp-id>/study.db`.

## Phase 2 — Visualization + narrative stub

- `ai-worker/scripts/summarize_study.py` — matplotlib PNG generation
  via `optuna.visualization.matplotlib.*` (`optimization_history`,
  `param_importances`, per-parameter `slice`, top-2 `contour`).
- `summary.md` stub with explicit Claude-narrative placeholders
  (`<!-- claude-narrative:top-trial-pattern -->`,
  `param-importances`, `next-direction`).
- `optuna-dashboard` for live browsing:
  `optuna-dashboard sqlite:///ai-worker/eval/experiments/studies/<exp>/study.db`.
- `docs/tuning.md` authored covering: first study launch, dashboard
  usage, cheap-vs-expensive parameter distinction, human-approval gate.

## Phase 3 — Claude Code integration

- `.claude/commands/tune-round.md` — wraps `scripts/tune.py`, auto-calls
  summarize, commits `studies/<id>/` minus gitignored `study.db`.
- `.claude/commands/analyze-study.md` — reads study.db + `summary.md`,
  fills narrative placeholders, commits filled summary.
- `.claude/commands/propose-next.md` — emits new `active.yaml` as a
  **chat diff only** (no direct `Edit` on `active.yaml`). Each change
  must carry justification.

## Phase 4 — Optional automation

- `loop` skill to schedule recurring tune → analyze cycles.
- Every 3rd round auto-injects a `RandomSampler` wide-random round to
  avoid local-optimum traps.
- Human approval still gates `active.yaml` mutations.

## Phase 5 — Deferred: expensive parameters

- Adds `embedding_model`, chunk `MIN_CH/MAX_CH/OVERLAP`, query/passage
  prefixes to the search space.
- Requires an index cache keyed by
  `(embedding_model, chunk_size, overlap)` hash at
  `rag-data/<hash>/{faiss.index, build.json}` so trials reuse rather
  than rebuild.
- Two-phase study pattern: outer expensive sweep selects Pareto
  candidates, inner cheap sweep refines each.

## Guardrails (encoded in code + command docs)

- `scripts/tune.py` refuses to overwrite an existing `study.db` unless
  `--resume` is passed.
- `propose-next` command explicitly instructs Claude to output a diff,
  never to call `Edit` on `active.yaml` directly.
- `active.yaml._meta.created_by` tracks provenance
  (`human` | `claude-proposed` | `claude-approved`).
- `.gitignore` excludes `study.db` + `plots/` but keeps `summary.md`
  and `config.yaml` (deterministic replay requires the YAML, not the
  binary study file).

## Out of scope (explicit)

- Agent SDK daemon — slash commands plus `loop` skill are sufficient.
- Weights & Biases / MLflow / any cloud integration — all local.
- GCS storage adapter — unrelated to tuning.
- Index caching for expensive params — Phase 5, not in the kickoff
  task.

## Priority ordering

Phase 1 → Phase 2 → Phase 3 as a single kickoff task. Phase 4 and
Phase 5 land in follow-up sessions once the baseline loop is exercised
against real data.
