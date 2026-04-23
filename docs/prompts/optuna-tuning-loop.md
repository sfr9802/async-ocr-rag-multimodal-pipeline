# Task — Build an Optuna + Claude-interpreter tuning loop

Self-contained kickoff prompt for a fresh Claude Code session. Paste
the entire body below into the new session; it does not rely on any
prior conversation.

---

## Project context

Working directory: `D:\async-ocr-rag-multimodal-pipeline`. Async AI job
platform:

- Java Spring Boot 4.0.3 `core-api/` — job lifecycle, PostgreSQL,
  Redis dispatch, local / S3 storage.
- Python 3.12 `ai-worker/` — three capabilities:
  - **RAG**: FAISS `IndexFlatIP` + bge-m3 multilingual embedder
    (1024-dim) + extractive / Claude generator.
  - **OCR**: Tesseract + PyMuPDF.
  - **MULTIMODAL v1**: OCR + vision + text-RAG fusion
    (heuristic / Claude vision).

Eval harness already exists at `ai-worker/eval/run_eval.py` with
subcommands `rag` / `ocr` / `multimodal` producing JSON / CSV reports.
Fixtures live in `ai-worker/eval/datasets/*.jsonl`. Runtime config is
driven by `AIPIPELINE_WORKER_*` env vars via
`ai-worker/app/core/config.py::WorkerSettings`.

**Read these first to orient:**

- `docs/architecture.md` — capability design + stage trace.
- `docs/optuna-tuning-plan.md` — the phased plan this task implements.
- `ai-worker/eval/run_eval.py` — existing CLI.
- `ai-worker/eval/harness/{rag_eval,ocr_eval,multimodal_eval}.py` —
  metric producers.
- `ai-worker/app/core/config.py` — the tunable parameter surface.

Verify the existing 249 unit tests are green before touching anything:
`cd ai-worker && pytest tests/ -q`.

## Goal

Build a hyperparameter tuning loop in which:

- **Optuna** drives the numerical search (TPE sampler).
- **Claude (in subsequent sessions)** reads study results, writes
  narrative summaries, and proposes next search spaces as YAML diffs.
- **Human approval** gates every mutation of `active.yaml`.

Core principle: **LLM is an interpreter, not a sampler.** Optuna picks
trial parameters; Claude analyzes outcomes and proposes search-space
refinements.

## Architecture

Directory layout to create:

```
ai-worker/eval/experiments/
├── active.yaml                  # current search space; git-tracked
├── studies/<exp-id>/
│   ├── study.db                 # Optuna SQLite; gitignored
│   ├── plots/*.png              # matplotlib visualizations
│   ├── summary.md               # Claude narrative + auto tables
│   └── config.yaml              # frozen search space snapshot
└── archive/                     # past active.yaml versions
```

## Deliverables (in priority order)

### 1. `ai-worker/scripts/tune.py`

- CLI: `python -m scripts.tune --experiment <name> --n-trials 50 [--resume]`
- `active.yaml` schema:
  ```yaml
  experiment_id: rag-cheap-sweep-v1
  objective:
    mode: rag                           # rag | ocr | multimodal
    dataset: eval/datasets/rag_sample_kr.jsonl
    primary_metric: hit_at_5            # maximize
    secondary_metrics: [mrr, p50_latency_ms]
  search_space:
    rag_top_k: {type: int, low: 3, high: 15}
    rrf_k:     {type: int, low: 10, high: 120}
  optuna:
    sampler: tpe
    n_trials: 50
    seed: 42
    direction: maximize
  _meta:
    created_by: human                   # human | claude-proposed | claude-approved
  ```
- Per trial: set sampled params into env, shell out to `eval.run_eval`,
  parse the JSON report, return the primary metric.
- Record `trial.set_user_attr` for: `cost_usd` (when Claude providers
  are active), `latency_ms`, `secondary_metric_values`, `config_hash`.
- Study storage: SQLite at
  `ai-worker/eval/experiments/studies/<experiment_id>/study.db`.
- On start: freeze the current `active.yaml` into
  `studies/<id>/config.yaml`.
- **Phase 1 scope — cheap params only** (no re-embedding): `rag_top_k`,
  `rrf_k`, `max_query_chars`, `short_query_words`,
  `max_fused_chunk_chars`, `excerpt_chars`, `ocr_min_confidence_warn`,
  `multimodal_max_vision_pages`. Leave a TODO comment for Phase 2
  expensive params (`embedding_model`, chunk size) behind an index
  cache.

### 2. `ai-worker/scripts/summarize_study.py`

- CLI: `python -m scripts.summarize_study --experiment <name>`.
- Reads the latest `studies/<id>/study.db`.
- Generates PNGs under `plots/` using
  `optuna.visualization.matplotlib`:
  - `optimization_history.png`
  - `param_importances.png`
  - `slice_<param>.png` per parameter
  - `contour_<a>_<b>.png` for the top-2 important parameters
- Generates `summary.md` with:
  - Header: experiment ID, dataset, start/end time, `n_trials`.
  - Metrics table: top-10 trials with all params + primary / secondary
    metrics + cost.
  - Best trial card: params, metric values, runtime.
  - Narrative placeholders for Claude to fill in a later session:
    ```markdown
    <!-- claude-narrative:top-trial-pattern -->
    (Claude fills: what do top trials have in common?)

    <!-- claude-narrative:param-importances -->
    (Claude fills: which params mattered, which didn't?)

    <!-- claude-narrative:next-direction -->
    (Claude fills: where should the next round search?)
    ```
- Add `optuna`, `optuna-dashboard`, `matplotlib`, `plotly` to
  `ai-worker/requirements-dev.txt` (create the file if it does not
  exist).

### 3. `.claude/commands/` slash commands

Create three markdown files:

- **`tune-round.md`** — invokes `scripts/tune.py` with validated args,
  tails the log, runs `summarize_study.py` afterward, and commits
  `studies/<id>/` (the gitignored `study.db` stays local).
- **`analyze-study.md`** — opens the latest `summary.md`, loads
  `study.db` via `optuna.load_study`, fills the narrative placeholders
  by analyzing top trials / parameter importances / failure patterns,
  and commits the filled summary.
- **`propose-next.md`** — reads the filled `summary.md` plus
  `study.db` and emits a proposed new `active.yaml` as a **diff in the
  chat** (NEVER auto-applied). The user reviews, edits, and commits
  manually. The proposal must state: which bounds narrowed, which
  expanded, and justification per change.

### 4. Guardrails (encode in code + command docs)

- `scripts/tune.py` refuses to overwrite an existing `study.db` unless
  `--resume` is passed.
- `propose-next.md` explicitly instructs Claude to output a diff —
  never to invoke `Edit` on `active.yaml` directly.
- `active.yaml._meta.created_by` tracks provenance
  (`human` / `claude-proposed` / `claude-approved`).
- Every third `tune-round` auto-injects a `RandomSampler` wide-random
  round to avoid local-optimum traps. Document this in
  `docs/tuning.md`.

### 5. `docs/tuning.md`

- How to kick off the first study.
- How to run `optuna-dashboard` locally:
  `optuna-dashboard sqlite:///ai-worker/eval/experiments/studies/<exp>/study.db`.
- When to use cheap vs expensive (Phase 2) params.
- How the human-approval gate on `active.yaml` works.

### 6. `.gitignore` updates

- Add `ai-worker/eval/experiments/studies/**/study.db`.
- Add `ai-worker/eval/experiments/studies/**/plots/`.
- Keep `summary.md` and `config.yaml` tracked (deterministic replay
  needs YAML + seed, not the binary study file).

## Out of scope (explicitly NOT in this task)

- Index caching for expensive params (`embedding_model`, chunk size) —
  Phase 5, leave a TODO only.
- Agent SDK daemon — slash commands plus the `loop` skill are
  sufficient.
- Cloud integrations (Weights & Biases, MLflow) — all local.
- GCS storage adapter — unrelated.

## Verification before declaring done

- `python -m scripts.tune --experiment smoke --n-trials 5` runs
  end-to-end on a subset of `rag_sample_kr.jsonl` (or using the
  `HashingEmbedder` fallback) without network access.
- `python -m scripts.summarize_study --experiment smoke` produces PNGs
  plus `summary.md` with visible narrative placeholders.
- `ai-worker/tests/test_tune.py` covers `active.yaml` parsing and
  objective wrapping.
- All 249 existing tests still pass.
- `optuna-dashboard` against the generated study renders correctly in
  a local browser.

## Notes

- `active.yaml` parsing with `yaml.safe_load` is sufficient —
  pydantic-settings is for env vars, not YAML files, so don't force
  the fit.
- Keep `tune.py` simple; resist adding features not listed. The YAML
  schema can grow in later phases.
- The existing eval harness already emits JSON with `primary_metric`
  and `secondary_metrics` — reuse that structure, do not re-invent.
