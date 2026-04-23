# Round 01 analysis — `rag-cheap-sweep-v3`

**Bundle:** `round_01_bundle.json` (sha256 `b7612a2956a4…7f78`)
**Prompt:** `prompts/claude_code/analyze_round.md` v0.1.0

## 1. Summary

- Best value `statistics.best_value = 0.8778` at `best_trial.number = 0` — never improved across the remaining 17 trials (10 of them tied at the same value).
- All 18 trials `COMPLETE`; `statistics.n_pruned = 0`, `statistics.n_failed = 0`.
- Signal is **single-dimensional**: `param_importances.rag_top_k = 1.0`, both categoricals at `0.0`. Two clean clusters — `clusters[0]` (13 trials, mrr 0.8778, `rag_top_k >= 7`) vs `clusters[1]` (5 trials, mrr 0.8667, `rag_top_k in {3, 5}`).

## 2. Per-param findings

**`rag_top_k`** (`int`, `[3, 15]`). Dominant axis — `param_importances.rag_top_k = 1.0`. Top cluster spans sampled values {7, 8, 9, 10, 11, 13, 14}; every trial in `clusters[1]` has `rag_top_k in {3, 5}`. `statistics.boundary_hits.rag_top_k = {low: 2, high: 0}` — TPE avoided the upper edge entirely, and the 2 low hits both sit in the weaker cluster. Nothing in `top_trials` lands above 14, so no evidence that raising `high` would help.

**`rag_use_mmr`** (`categorical`, `[false, true]`). `param_importances.rag_use_mmr = 0.0`. Splits the top plateau 5/5 between `false` and `true` (see `trials[*].params.rag_use_mmr` for trial numbers 0,6,7,8,14 vs 4,5,11,12,13). Irrelevant on this fixture; the 0.032 importance reported at the 8-trial interim was TPE noise.

**`rag_query_parser`** (`categorical`, `["off", "regex"]`). `param_importances.rag_query_parser = 0.0`. Plateau split 6 `off` / 4 `regex`. Offline regex normalization produces the same embedder input as passthrough for these 30 well-formed Korean queries — the axis is structurally inert on this corpus.

## 3. Cross-param patterns

`clusters[0]` groups the 13 plateau trials at `value = 0.8778`; `clusters[1]` the 5 worse trials at `0.8667`. No interaction is visible — the plateau is reached as soon as `rag_top_k >= 7` regardless of `rag_use_mmr` or `rag_query_parser`. No multi-objective / Pareto structure in the bundle.

## 4. Pruning and failure analysis

None. `statistics.n_pruned = 0`, `statistics.n_failed = 0`. The round used `optuna.pruner.type = NopPruner` (no pruning configured) and the eval subprocess never errored.

## 5. Open questions

- `cost_usd` is a hard-coded `0.0` in every `user_attrs` payload — the adapter has no LLM-usage hook yet, so Pareto-on-cost is impossible this round.
- `latency_ms` in `user_attrs` is the **subprocess wall-clock** (~13 s cold-loaded bge-m3 per trial). The bundle lacks a separate field for the *eval-internal* retrieval latency; only `secondary_metric_values.mean_total_ms` (~39 ms) reflects the real serving cost.
- No per-query miss breakdown in the bundle. The 3 fixed misses on `rag_anime_extended_kr` (observed in the standalone eval) are opaque at this layer — surfacing which doc-IDs consistently miss would tell us whether the remaining headroom is retrieval or generation.
- Primary metric saturates at **0.8778** on this fixture no matter what cheap param is touched. The bundle lacks a field declaring "metric ceiling at current index/reranker" — useful for the plateau decision.
