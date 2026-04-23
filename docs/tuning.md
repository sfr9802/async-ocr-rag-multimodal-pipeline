# Hyperparameter tuning — Optuna + Claude interpreter

This pipeline tunes the eval harness via an **Optuna TPE sweep + Claude
narrative interpreter** loop. The division of labor is deliberate:

- **Optuna** drives the numerical search. TPE sampler, seeded,
  reproducible.
- **Claude** (in `/analyze-study` and `/propose-next` sessions) reads
  the study, writes human-readable summaries, and proposes the next
  search space as a YAML diff.
- **Human approval** gates every mutation of `active.yaml`. Claude
  never edits it in place.

Core principle: **the LLM is an interpreter, not a sampler.**

## Directory layout

```
ai-worker/eval/experiments/
├── active.yaml                          # current search space — git-tracked
├── studies/<experiment_id>/
│   ├── study.db                         # Optuna SQLite — gitignored
│   ├── plots/*.png                      # matplotlib visualizations — gitignored
│   ├── config.yaml                      # frozen search-space snapshot — tracked
│   ├── summary.md                       # metrics + /analyze-study narrative — tracked
│   ├── round_NN_bundle.json             # skill adapter export — tracked
│   ├── round_NN_bundle_llm_input.md     # rendered brief for the LLM — tracked
│   ├── round_NN_analysis.md             # analyze_round output — tracked
│   ├── round_NN_report.md               # round report — tracked
│   └── round_NN+1_config.json           # proposed next-round config — tracked
└── archive/                             # past active.yaml versions
```

`summary.md` + `config.yaml` give a deterministic replay record for the
slash-command flow; `round_NN_bundle.json` + `round_NN+1_config.json`
play the same role for the skill-adapter flow. The binary `study.db` is
not required to reproduce the finding — if someone deletes it,
rerunning `scripts.tune` against the frozen `config.yaml` recreates the
same trials.

## Kick off the first study

From `ai-worker/` (activate your venv first):

```bash
# 0. Dev deps — one-time
pip install -r requirements.txt -r requirements-dev.txt

# 1. Confirm existing tests still pass
pytest tests/ -q

# 2. Review the initial search space
cat eval/experiments/active.yaml

# 3. Smoke test with a tiny budget
python -m scripts.tune --experiment rag-cheap-sweep-v1 --n-trials 5

# 4. Render the summary + plots
python -m scripts.summarize_study --experiment rag-cheap-sweep-v1

# 5. Open the study in your browser for interactive browsing
optuna-dashboard sqlite:///eval/experiments/studies/rag-cheap-sweep-v1/study.db
```

For a real round, drop `--n-trials 5` and let `active.yaml`'s
`optuna.n_trials` (default 50) drive it.

## The round-by-round loop

One tuning round is three Claude Code slash commands, invoked in order:

1. **`/tune-round <experiment-id>`** — runs `scripts.tune` then
   `scripts.summarize_study`, commits `summary.md` + `config.yaml`.
   No narrative yet.

2. **`/analyze-study <experiment-id>`** — Claude reads the study +
   summary, fills the three narrative placeholders in `summary.md`,
   commits the filled summary.

3. **`/propose-next <experiment-id>`** — Claude emits a proposed new
   `active.yaml` as a unified **diff in chat only**, with a
   justification bullet per change. The human reviews, edits if
   needed, applies manually, flips `_meta.created_by` to
   `claude-approved`, then loops back to `/tune-round` for the next
   experiment ID.

### Every third round should be a random-sampler round

TPE biases toward whatever region looks good after 20-30 trials. To
keep it from settling into a local optimum, **every third round
must use the `RandomSampler`** for wide exploration:

```bash
python -m scripts.tune --experiment <exp> --random-sampler
```

The `/tune-round` slash command counts past rounds and auto-injects
`--random-sampler` on rounds 3, 6, 9, … — you can override with an
explicit argument if you need to hammer a narrow region. The same
rule applies to the skill-adapter flow described below.

## Alternative: round refinement via the `optuna-round-refinement` skill

Two round-transition flows coexist; pick one per round:

1. **Slash-command flow** (documented above): `/tune-round` → `/analyze-study`
   fills `summary.md` narrative placeholders → `/propose-next` emits a chat diff.
   Optimised for human reading.
2. **Skill-adapter flow** (this section): `/tune-round` → `round_adapter
   export-bundle` → LLM analyst applies the skill's prompts → `round_adapter
   apply`. Schema-validated JSON + hash-attested artifacts end-to-end.

### Install the skill once

```bash
git clone https://github.com/sfr9802/optuna-round-refinement \
    ~/.claude/skills/optuna-round-refinement
```

Point `$OPTUNA_ROUND_REFINEMENT_HOME` at a local checkout if you want an
override; otherwise `scripts.round_adapter` auto-discovers the install
under `~/.claude/skills/` or `.claude/skills/`.

### Five steps per round

```bash
# 1. Numerical round (same as the slash-command flow).
python -m scripts.tune --experiment rag-cheap-sweep-v3 --n-trials 18

# 2. Canonical bundle export from study.db.
python -m scripts.round_adapter export-bundle --experiment rag-cheap-sweep-v3

# 3. Human-readable brief for the LLM analyst (optional if your LLM
#    can read the JSON bundle directly).
python -m scripts.round_adapter render-input \
    --bundle eval/experiments/studies/rag-cheap-sweep-v3/round_01_bundle.json

# 4. Hand the brief + the skill's prompts/claude_code/{analyze_round,
#    propose_next_round}.md to the LLM. Save its outputs as
#    round_NN_analysis.md, round_NN_report.md, round_NN+1_config.json.

# 5. Preview (default) or write the active.yaml mutation.
python -m scripts.round_adapter apply \
    --config eval/experiments/studies/rag-cheap-sweep-v3/round_02_config.json
# --write actually overwrites active.yaml (bumps experiment_id).
```

### What `apply` guarantees

- `jsonschema.validate` against the skill's
  `schemas/next_round_config.schema.json`.
- Cross-checks `provenance.source_bundle_hash` against the on-disk
  bundle's canonical sha256 (SKILL.md §8.1 enforcement — apply
  refuses to run on a mismatch).
- Bumps `experiment_id` (v3 → v4) so `study.db` isolation is kept.
- Preserves `objective` (mode / dataset / primary_metric /
  secondary_metrics); the skill config doesn't own these.
- Translates schema `search_space` + `fixed_params` into the
  active.yaml shape `scripts.tune` expects.
- Migrates `dataset` from `fixed_params` into `objective.dataset` if
  the LLM put it there (both locations are schema-valid; tune.py only
  reads from objective).
- Populates `_meta` with full provenance
  (`created_by: claude-proposed`, `parent_experiment_id`,
  `source_bundle_hash`, `parent_config_hash`, `generated_at`,
  `generated_by`, `rationale`).

Flip `_meta.created_by` to `claude-approved` by hand before the next
`/tune-round` — the string `claude-approved` is the human signature.

### `fixed_params` — non-sweeped axes

The skill's `freeze` action lands at a top-level `fixed_params: {}`
block in active.yaml (alongside `search_space`). `scripts.tune`
injects these as `AIPIPELINE_WORKER_<KEY>` env vars on every trial
subprocess — same code path as `search_space` values, just not
sampled. Keys that appear in both blocks are rejected at config-load
time to prevent the search from shadowing a freeze decision.

### Windows note

`round_adapter` forces stdout to UTF-8 in the CLI entry so yaml
previews carrying em-dashes or Korean operator notes don't fall over
on CP949 consoles.

## When to use cheap vs. expensive params

### Phase 1 — cheap params (the current sweep)

These do NOT trigger re-embedding or index rebuild. A trial is
bounded by the eval run itself (seconds on the Korean sample):

- `rag_top_k`
- `cross_modal_rrf_k` (MULTIMODAL mode, cross-modal only)
- `ocr_min_confidence_warn`
- `multimodal_max_vision_pages`

Phase 2 knobs that live in the capability config but are NOT yet
environment-variable-backed in `WorkerSettings`:

- `max_query_chars`, `short_query_words` (fusion)
- `max_fused_chunk_chars` (multimodal capability)
- `excerpt_chars` (extractive generator)

`tune.py` sets an env var for each of these so the YAML stays
self-documenting, but the eval subprocess will ignore them until
someone wires them into `app/core/config.py`. See the TODOs in
`active.yaml`.

### Phase 2 — expensive params (DEFERRED)

These rebuild the FAISS index per trial. Don't enable them until the
index cache is built (`rag-data/<hash>/{faiss.index, build.json}`
keyed on `(embedding_model, chunk_size, overlap)`):

- `rag_embedding_model` — switching bge-m3 ↔ multilingual-e5-small
  means a full re-embed.
- `rag_chunk_min_chars` / `rag_chunk_max_chars` / `rag_chunk_overlap`
  — re-chunk + re-embed.
- `rag_embedding_prefix_query` / `rag_embedding_prefix_passage`.

When we hit this phase, the pattern is:

1. Outer "expensive" study with the large-rebuild params in its
   search space. Small `n_trials` (≤ 20) and Pareto-selects a handful
   of index candidates.
2. Per candidate, an inner "cheap" study tunes the Phase 1 params.

See `docs/optuna-tuning-plan.md` Phase 5 for the full plan.

## Human-approval gate on `active.yaml`

`active.yaml._meta.created_by` carries provenance:

| value | meaning |
| --- | --- |
| `human` | Initial state — written by a human. |
| `claude-proposed` | `/propose-next` drafted this (should never hit disk unless the human applied the diff). |
| `claude-approved` | A human reviewed a Claude proposal and is running the next round with it. |

If you see a commit changing `active.yaml` with `_meta.created_by:
claude-proposed`, that means someone accepted a Claude diff without
flipping the flag — reject the commit and ask for the flag to be
updated. The point of the gate is that the string `claude-approved`
is proof a human saw the diff.

## Monitoring a running study

### optuna-dashboard (interactive, local)

```bash
optuna-dashboard sqlite:///eval/experiments/studies/<exp>/study.db
```

Auto-refreshes. Gives you the same plots `summarize_study.py` emits,
plus per-trial inspection and hyperparameter-relationship views.

### Quick tail of `study.db`

```bash
python -c "
import optuna
s = optuna.load_study(
    study_name='<exp>',
    storage='sqlite:///eval/experiments/studies/<exp>/study.db',
)
for t in s.trials[-5:]:
    print(t.number, t.state.name, t.value, t.params)
"
```

## Resuming vs. restarting

- `python -m scripts.tune --experiment <exp>` without `--resume`
  **refuses** to overwrite an existing `study.db`. This is on purpose
  — a study is cheap to accumulate but painful to lose.
- Pass `--resume` to append more trials to the same study. The
  frozen `config.yaml` is preserved; don't expect a changed
  `active.yaml` to take effect on resume (it wouldn't be
  reproducible).
- To truly restart: delete `study.db` (and optionally `plots/` +
  `summary.md`) for that experiment, then rerun without `--resume`.

## Troubleshooting

- **"active.yaml experiment_id does not match --experiment"**: the
  file was bumped to the next version but the CLI flag still names
  the old one. Update the CLI flag to match.
- **"No study.db at …"** from `summarize_study`: `scripts.tune` never
  ran for that experiment, or the studies root is wrong (CWD must be
  `ai-worker/`).
- **Param-importances plot missing**: too few completed trials. Needs
  at least a handful of trials over a param that actually varied.
- **"eval subprocess exited N"**: the underlying eval harness
  failed — check the stderr tail printed by `scripts.tune`, then
  debug with `python -m eval.run_eval <mode> --dataset <...>`
  directly.
