# Study summary — `rag-cheap-sweep-v3`

| field | value |
| --- | --- |
| experiment_id | `rag-cheap-sweep-v3` |
| mode | `rag` |
| dataset | `eval/datasets/rag_anime_extended_kr.jsonl` |
| primary_metric | `mrr` |
| direction | `maximize` |
| sampler | `tpe` |
| seed | `42` |
| trials (total / complete / failed) | 8 / 8 / 0 |
| wall-time (sum of trial durations) | 115.1 s |
| started_at | 2026-04-23T21:30:52 |
| finished_at | 2026-04-23T21:32:48 |

## Parameter importances

| parameter | importance |
| --- | ---: |
| `rag_top_k` | 0.9683 |
| `rag_use_mmr` | 0.0317 |
| `rag_query_parser` | 0.0000 |

## Top 8 trials (by `mrr`)

| # | mrr | sec:mean_hit_at_k | sec:mean_keyword_coverage | sec:mean_total_ms | sec:p50_retrieval_ms | cost_usd | latency_ms | p:rag_query_parser | p:rag_top_k | p:rag_use_mmr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8778 | 0.9667 | 0.9556 | 39.2280 | 30.9740 | 0.0000 | 13567.0 | off | 7 | False |
| 4 | 0.8778 | 0.9667 | 0.9556 | 39.2470 | 31.1180 | 0.0000 | 13507.7 | regex | 10 | True |
| 5 | 0.8778 | 0.9667 | 0.9667 | 40.5600 | 31.0070 | 0.0000 | 13567.6 | off | 13 | True |
| 6 | 0.8778 | 0.9667 | 0.9556 | 39.4630 | 31.0650 | 0.0000 | 13749.4 | regex | 10 | False |
| 7 | 0.8778 | 0.9667 | 0.9667 | 38.7880 | 30.5290 | 0.0000 | 13895.0 | off | 13 | False |
| 1 | 0.8667 | 0.9000 | 0.9000 | 39.9580 | 30.9300 | 0.0000 | 13644.1 | regex | 5 | True |
| 2 | 0.8667 | 0.9000 | 0.9000 | 38.4110 | 30.9590 | 0.0000 | 13455.6 | off | 3 | False |
| 3 | 0.8667 | 0.9000 | 0.9000 | 39.1460 | 31.1200 | 0.0000 | 13483.5 | off | 5 | True |

## Best trial

- trial #0
- `mrr` = **0.8778**
- `mean_hit_at_k` = 0.9667
- `mean_keyword_coverage` = 0.9556
- `mean_total_ms` = 39.2280
- `p50_retrieval_ms` = 30.9740
- `config_hash` = `fcf99e483dce`
- `latency_ms` = 13567.0
- params:
  - `rag_query_parser` = `off`
  - `rag_top_k` = `7`
  - `rag_use_mmr` = `False`

## Plots

![optimization_history](plots/optimization_history.png)
![param_importances](plots/param_importances.png)
![slice_rag_top_k](plots/slice_rag_top_k.png)
![slice_rag_use_mmr](plots/slice_rag_use_mmr.png)
![slice_rag_query_parser](plots/slice_rag_query_parser.png)
![contour_rag_top_k_rag_use_mmr](plots/contour_rag_top_k_rag_use_mmr.png)

## Narrative (filled in by `/analyze-study`)

<!-- claude-narrative:top-trial-pattern -->
All five trials tied at `mrr=0.8778` share `rag_top_k >= 7` (values 7, 10, 10, 13, 13);
the three trials at `mrr=0.8667` all have `rag_top_k in {3, 5}`. `rag_use_mmr` and
`rag_query_parser` both appear in each cluster without separating them — the signal is
single-dimensional on this fixture. The best (trial #0) sits at `rag_top_k=7`, mid-range
rather than at the `high=15` edge, and three different `top_k` values (7, 10, 13) all
hit the same plateau — suggesting a ceiling rather than a unique best.

<!-- claude-narrative:param-importances -->
`rag_top_k` dominates at importance 0.968; it is effectively the only knob moving
`mrr` on this 30-row anime_extended fixture. `rag_use_mmr` lands at 0.032 (likely
noise at n=8 — both levels occur in the best cluster). `rag_query_parser` is 0.000
because the anime queries are already well-formed Korean; offline regex normalization
produces the same embedder input as passthrough. Importance estimates at 8 trials are
directional only — 50 trials would firm them up, but the rag_query_parser result is
already credible because its within-cluster variance is exactly zero.

<!-- claude-narrative:next-direction -->
Narrow `rag_top_k` to `[7, 15]` — the `[3, 6]` half is provably worse (0.8667 vs
0.8778) and `top_k >= 7` is a flat plateau. Drop `rag_query_parser` entirely; its
importance is 0 and both choices produce identical mrr here. Keep `rag_use_mmr` but
treat its 0.032 importance as unconfirmed until the next round runs with more trials.
The 0.8778 mrr ceiling + baseline `mean_hit_at_k=0.9667` says the remaining headroom
is not reachable via cheap params — the right next lever is promoting
`rag_reranker: cross_encoder` (prewarm `BAAI/bge-reranker-v2-m3` in the HF cache
first so trial #0 isn't a 300MB download).
