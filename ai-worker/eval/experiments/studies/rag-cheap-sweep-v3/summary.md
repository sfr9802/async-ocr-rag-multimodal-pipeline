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
| trials (total / complete / failed) | 18 / 18 / 0 |
| wall-time (sum of trial durations) | 253.6 s |
| started_at | 2026-04-23T21:30:52 |
| finished_at | 2026-04-23T23:49:43 |

## Parameter importances

| parameter | importance |
| --- | ---: |
| `rag_top_k` | 1.0000 |
| `rag_use_mmr` | 0.0000 |
| `rag_query_parser` | 0.0000 |

## Top 10 trials (by `mrr`)

| # | mrr | sec:mean_hit_at_k | sec:mean_keyword_coverage | sec:mean_total_ms | sec:p50_retrieval_ms | cost_usd | latency_ms | p:rag_query_parser | p:rag_top_k | p:rag_use_mmr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8778 | 0.9667 | 0.9556 | 39.2280 | 30.9740 | 0.0000 | 13567.0 | off | 7 | False |
| 4 | 0.8778 | 0.9667 | 0.9556 | 39.2470 | 31.1180 | 0.0000 | 13507.7 | regex | 10 | True |
| 5 | 0.8778 | 0.9667 | 0.9667 | 40.5600 | 31.0070 | 0.0000 | 13567.6 | off | 13 | True |
| 6 | 0.8778 | 0.9667 | 0.9556 | 39.4630 | 31.0650 | 0.0000 | 13749.4 | regex | 10 | False |
| 7 | 0.8778 | 0.9667 | 0.9667 | 38.7880 | 30.5290 | 0.0000 | 13895.0 | off | 13 | False |
| 8 | 0.8778 | 0.9667 | 0.9556 | 41.1080 | 31.1240 | 0.0000 | 13678.7 | off | 7 | False |
| 11 | 0.8778 | 0.9667 | 0.9667 | 38.5230 | 30.7670 | 0.0000 | 12937.1 | regex | 11 | True |
| 12 | 0.8778 | 0.9667 | 0.9556 | 38.8050 | 30.9010 | 0.0000 | 13057.5 | off | 9 | True |
| 13 | 0.8778 | 0.9667 | 0.9667 | 39.8730 | 30.9040 | 0.0000 | 13084.8 | regex | 14 | True |
| 14 | 0.8778 | 0.9667 | 0.9556 | 38.4550 | 30.8770 | 0.0000 | 12918.2 | off | 9 | False |

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
_(Claude fills: what do top trials have in common?)_

<!-- claude-narrative:param-importances -->
_(Claude fills: which params mattered, which didn't?)_

<!-- claude-narrative:next-direction -->
_(Claude fills: where should the next round search?)_
