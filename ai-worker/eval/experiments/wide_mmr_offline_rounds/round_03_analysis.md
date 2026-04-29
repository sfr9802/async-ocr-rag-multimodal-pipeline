# Round 03 — `rag-wide-mmr-offline-v1`

## TL;DR

With `rerank_in` fixed at 16, **every other axis is invisible to MRR@10 on the 50-row subset**. All 16 trials returned MRR=0.6717 exactly (std=0). Param importance fell back to a default uniform-1.0-on-mmr_lambda assignment (Optuna's default when there is no objective variance). The 50-row subset has hit a **dataset-side ceiling** — TPE cannot read any signal from candidate_k / mmr_lambda / mmr_k / cap_* / final_top_k because they all map to the same 16 docs at the rerank stage.

## Decision for round 04

**Expand the dataset subset from 50 → 100 rows.** The wide-MMR-titlecap diagnostic at 200 rows showed MRR varied 0.6587 → 0.6699 across MMR variants. The 50-row first-prefix slice happens to be ranked identically by every config we tried. Doubling to 100 rows brings in different gold docs whose ranks should expose the signal that this slice masks.

`OFFLINE_TUNE_QUERY_LIMIT=100` becomes the round_04 fixed_param. Search space stays unchanged; no axis can be narrowed when no axis has produced any evidence.

## Anti-pattern audit

- **A10 (narrow against unsampled edge):** mmr_lambda edges still UNSAMPLED — kept full [0.55, 0.75].
- **A12 (search-space change without evidence):** no search_space changes — only the fixed_params subset size.
- **A13 (objective swap masking the issue):** would fire if I tried to swap to NDCG instead of fixing the real problem (dataset slice too small). Subset expansion is the right move.

## Headline diff (round_03 → round_04)

| field | round_03 | round_04 | reason |
|---|---|---|---|
| `fixed_params.dataset_subset` | `anime_silver_200_first_50` | `anime_silver_200_first_100` | std_value=0 across 16 trials at 50 rows; need bigger slice for variance |
| `sampler.seed` | 44 | 45 | reseed after fixed_params change |
| search_space | unchanged | unchanged | no axis has produced narrowing evidence yet |
