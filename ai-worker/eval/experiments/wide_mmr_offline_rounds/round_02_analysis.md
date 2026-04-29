# Round 02 — `rag-wide-mmr-offline-v1`

## TL;DR

`mean_mrr_at_10` swap surfaced a **clean monotone signal on `rerank_in`** that hit@5 had been hiding. Every trial's MRR is determined by rerank_in alone in this offline subset:

| rerank_in | trials | every trial scored |
|---|---:|---|
| 16 | 10 | 0.6717 |
| 24 | 4 | 0.6673 |
| 32 | 2 | 0.664 |

Param importance: `rerank_in` = 0.9827; every other param < 0.006. Once `rerank_in` is fixed, the masked signals on other axes should surface in round 03.

## Decision for round 03

**Narrow `rerank_in` from `[16, 24, 32]` to `[16]`** (single-choice). This is well-evidenced — A11 boundary evidence on a categorical axis that perfectly partitions the trial set into three deterministic value buckets. Other axes stay frozen because their signals are within 0.005 spread, dominated by the rerank_in effect; only when rerank_in is fixed will TPE be able to read them.

`mmr_lambda` axis_coverage still flags both edges as UNSAMPLED (sampled range 0.554-0.746); A10 forbids narrowing.

## Anti-pattern audit

- **A10 (narrow against unsampled edge):** `mmr_lambda` edges still UNSAMPLED — kept full [0.55, 0.75].
- **A11 (narrow with axis evidence):** rerank_in narrowed because every trial bucket is deterministic.
- **A12 (search-space change without evidence):** other axes have no decisive evidence — kept unchanged.

## Headline diff (round_02 → round_03)

| field | round_02 | round_03 | reason |
|---|---|---|---|
| `search_space.rerank_in` | `[16, 24, 32]` | `[16]` | importance=0.9827; deterministic value buckets in trials |
| `n_trials` | 16 | 16 | unchanged |
| `sampler.seed` | 43 | 44 | reseed since search space changed |
