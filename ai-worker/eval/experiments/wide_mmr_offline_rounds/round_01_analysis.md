# Round 01 — `rag-wide-mmr-offline-v1`

## TL;DR

Primary metric `mean_hit_at_5` saturated at exactly **0.76** across **all 12 trials**. Secondary metrics (`mean_mrr_at_10`, `mean_ndcg_at_10`, duplicate ratio, latency) DO vary but were never the optimization target. Param importances flat at 0.125 (no signal). `mmr_lambda` edges both UNSAMPLED (per axis_coverage). **Cannot narrow.** Recommendation: switch primary to `mean_mrr_at_10` (rank-sensitive, varies in this exact dataset slice) and re-run identical search space.

## Headline numbers

- Trials complete: 12 / 0 pruned / 0 failed
- Best value: 0.76 (trial 0); std ≈ 0 (numerical noise only)
- Best params: `{candidate_k=200, final_top_k=5, rerank_in=32, use_mmr=true, mmr_lambda=0.670, mmr_k=64, title_cap_rerank_input=1, title_cap_final=2}`

## Param importance reading

All 8 params importance = 0.125 → **fANOVA collapse**: when the objective has zero variance, importance defaults to uniform. This is *not* "all params equally useful"; it's "the model has no gradient to learn from".

## Boundary / coverage notes

| param | sampled range | edges sampled? |
|---|---|---|
| `mmr_lambda` | [0.583, 0.726] | both edges (0.55, 0.75) UNSAMPLED |
| `candidate_k` | {100, 200} | both choices sampled |
| `final_top_k` | {5, 8, 10} | all sampled |
| `rerank_in` | {16, 24, 32} | all sampled |
| `mmr_k` | {32, 48, 64, 80} | all sampled |
| `title_cap_rerank_input` | {1, 2, 3} | all sampled |
| `title_cap_final` | {1, 2, 3} | all sampled |

Per skill's anti-pattern A10: do NOT narrow `mmr_lambda` because both edges are unsampled (cannot prove they are bad).

## Why mean_hit_at_5 saturated

The 50-row subset of `anime_silver_200` has only ~0.02 hit@5 of headroom around the wide-MMR-titlecap reading (200-row sweep showed hit@5 0.72 baseline → 0.74 MMR). Compressing to 50 rows effectively snaps every MMR config onto a single value (38/50 = 0.76, every trial got the same 38 right). hit@5 is also a thresholded indicator metric; rank position within top-5 is invisible to it.

Secondary metrics that DO vary (per best-trial bundle samples):
- `mean_mrr_at_10`: 0.664 in best trial, varied 0.65–0.67 across trials → 1–2 pp swing.
- `mean_ndcg_at_10`: 0.688 best, ~0.683–0.688 → 0.5 pp swing.
- `duplicate_doc_ratio_at_10`: best 0.50, range across configs 0.20–0.51 → meaningful diversity-by-config swing.
- `mean_rerank_ms`: best 1183, range 700–1450 ms → reflects rerank_in directly.

## Decision for round 02

**Switch primary metric to `mean_mrr_at_10`.** Rationale:

1. The skill anti-pattern docs explicitly advise switching the indicator-metric when it's flat across rounds (parallel to `active.yaml` v3 which moved off `mean_hit_at_k` to `mrr` for the same reason on a different dataset).
2. `mean_mrr_at_10` has measurable variance in trial bundle: TPE will see signal.
3. We do NOT change the dataset or search space — only the metric. This keeps round 2 a *clean* "is the bottleneck the metric or the params" experiment.

**Do NOT narrow** any axis. `mmr_lambda` edges are unsampled (A10 violation if narrowed). Categoricals all sampled but objective variance was zero — narrowing categoricals on importance=0.125 is not evidence-based.

**Increase n_trials** modestly to 16 so TPE has more data for the new metric (n_startup_trials stays 4).

## Anti-patterns audit

- A10 (narrow against unsampled edge): would fire if I tried to narrow `mmr_lambda`. **Avoided.**
- A07 (silently swap primary metric without flagging): the metric swap is the explicit headline of this analysis and must be reflected in `provenance.diff_summary` of round_02 config.
- A12 (search-space change without evidence): no search_space changes in round_02 — only metric + n_trials.

## Headline diff (round_01 → round_02)

| field | round_01 | round_02 | reason |
|---|---|---|---|
| `objective_name` | `mean_hit_at_5` | `mean_mrr_at_10` | round_01 saw 0 variance on hit@5 over the 50-row subset; secondary MRR has 1-2 pp variance. |
| `n_trials` | 12 | 16 | give TPE more data on the new metric. |
| search_space | (8 dims) | unchanged | A10 + A12: no evidence to narrow. |
