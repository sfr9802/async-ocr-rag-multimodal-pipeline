# Round 05 — `rag-wide-mmr-offline-v1`

## TL;DR

All 16 trials at the (candidate_k=100, title_cap_rerank_input=1) recipe scored mean_mrr_at_10 = **0.6745** exactly (std_value = 0.0). With both winner axes fixed, the four remaining open axes (`final_top_k`, `mmr_k`, `mmr_lambda`, `title_cap_final`) had **zero effect** on the MRR ceiling. The 16 trials covered:

- `final_top_k`: 5/8/10 — all 0.6745
- `mmr_k`: 32/48/64/80 — all 0.6745
- `mmr_lambda`: 0.568–0.748 (continuous) — all 0.6745, edges still UNSAMPLED
- `title_cap_final`: 1/2/3 — all 0.6745

**Optuna study converged on the offline 100-row subset.** No further single-round narrowing on this subset can move the metric.

## Final winner config

```json
{
  "candidate_k": 100,
  "rerank_in": 16,
  "use_mmr": "true",
  "title_cap_rerank_input": 1,
  "title_cap_final": "any of 1/2/3",
  "final_top_k": "any of 5/8/10",
  "mmr_k": "any of 32/48/64/80",
  "mmr_lambda": "any value in [0.55, 0.75]"
}
```

with `mean_mrr_at_10 = 0.6745` on the 100-row subset of `anime_silver_200`.

## Anti-pattern audit

- **A10 (narrow against unsampled edge):** mmr_lambda edges still UNSAMPLED in round_05 axis_coverage; would forbid narrowing — but the round did NOT narrow it.
- **A14 (terminate prematurely):** Saturation at 0.6745 across 16 trials with std=0 is sufficient evidence that the open axes carry no residual signal at this dataset size. Further rounds on the same 100-row subset would be wasteful.
- **A07 (silent objective swap):** the metric swap to MRR happened in round_02 with explicit provenance and rationale; round_05 maintained it.

## Recommendation: study terminated

Within this 5-round budget the study is converged. Further work would have to change a fixed_params field, not the search space:

1. **Confirm on the full 200-row silver set.** The offline tune_eval at QUERY_LIMIT=200 would tell us whether (cand=100, cap_rr=1) MRR uplift transfers from the 100-row slice to the full set, or whether the slice happens to favour cap_rr=1 and the full set actually prefers cap_rr=2 (which the wide-MMR-titlecap diagnostic highlighted).
2. **Probe the embedding text variant axis** (title vs raw vs title_section). The round_04+05 study held this fixed; the wide-MMR diagnostic flagged candidate-recall@50=0.80 as the embedding-side ceiling.
3. **Manually tag query_type** and re-run with byQueryType primary so per-bucket optimization can identify whether character_relation / plot_event have different optimal recipes.

None of those are part of this round budget.
