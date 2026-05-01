# Phase 7.5 — latency smoke check

> Smoke compares pre-promotion baseline (no MMR), the Phase 7.x first-pass best (candidate_k=30, MMR, λ=0.70), and the Phase 7.5 production-recommended (candidate_k=40, MMR, λ=0.70) on the same query set.

## Honest scope

- **Pool size**: 40
- **Pool note**: candidate_gen_ms reflects the cached pool's elapsed_ms — the confirm sweep ran the live retriever at pool_size=40 (no MMR). For candidate_k less than the pool size the live retriever would do slightly less work, so candidate_gen_ms here is a small upper bound.
- **Rerank note**: Reranker stage NOT measured in this smoke check. The eval environment runs with the NoOp reranker — any timing produced here would be a fixture artefact, not a production-reranker number.

## Configs under test

| name | role | top_k | candidate_k | use_mmr | mmr_lambda | description |
|---|---|---:|---:|:---:|---:|---|
| `baseline_top10` | baseline | 10 | 10 | — | 0.70 | pre-promotion baseline (use_mmr=false, top_k=10). Mirrors the shipping retrieval path. |
| `previous_best_candk30_lambda070` | previous_best | 10 | 30 | ✓ | 0.70 | Phase 7.x first-pass best (candidate_k=30, MMR on, λ=0.70). Intermediate fallback config. |
| `recommended_candk40_lambda070` | production_recommended | 10 | 40 | ✓ | 0.70 | Phase 7.5 production-recommended (candidate_k=40, MMR on, λ=0.70 from the plateau-aware policy). |

## Per-set latency (ms)

### gold-50

| config | role | n | candidate_gen p50/p90/p99 | mmr_post p50/p90/p99 | total p50/p90/p99 |
|---|---|---:|---:|---:|---:|
| `baseline_top10` | baseline | 50 | 21.210 / 23.065 / 123.506 | 0.007 / 0.007 / 0.009 | 21.218 / 23.072 / 123.514 |
| `previous_best_candk30_lambda070` | previous_best | 50 | 21.210 / 23.065 / 123.506 | 0.034 / 0.035 / 0.041 | 21.245 / 23.098 / 123.541 |
| `recommended_candk40_lambda070` | production_recommended | 50 | 21.210 / 23.065 / 123.506 | 0.042 / 0.044 / 0.047 | 21.253 / 23.107 / 123.549 |

### silver-500

| config | role | n | candidate_gen p50/p90/p99 | mmr_post p50/p90/p99 | total p50/p90/p99 |
|---|---|---:|---:|---:|---:|
| `baseline_top10` | baseline | 500 | 20.517 / 22.637 / 30.805 | 0.007 / 0.007 / 0.008 | 20.524 / 22.644 / 30.813 |
| `previous_best_candk30_lambda070` | previous_best | 500 | 20.517 / 22.637 / 30.805 | 0.034 / 0.035 / 0.039 | 20.552 / 22.671 / 30.839 |
| `recommended_candk40_lambda070` | production_recommended | 500 | 20.517 / 22.637 / 30.805 | 0.043 / 0.044 / 0.047 | 20.559 / 22.680 / 30.847 |

### combined-550

| config | role | n | candidate_gen p50/p90/p99 | mmr_post p50/p90/p99 | total p50/p90/p99 |
|---|---|---:|---:|---:|---:|
| `baseline_top10` | baseline | 550 | 20.606 / 22.736 / 32.235 | 0.007 / 0.007 / 0.008 | 20.614 / 22.744 / 32.242 |
| `previous_best_candk30_lambda070` | previous_best | 550 | 20.606 / 22.736 / 32.235 | 0.034 / 0.034 / 0.038 | 20.640 / 22.775 / 32.268 |
| `recommended_candk40_lambda070` | production_recommended | 550 | 20.606 / 22.736 / 32.235 | 0.043 / 0.044 / 0.049 | 20.649 / 22.780 / 32.277 |

## Verdict

- **gold-50**: recommended total_p90=23.107ms vs previous-best 23.098ms (Δ=+0.0%).
  - vs baseline (no-MMR) total_p99=123.514ms → recommended 123.549ms (Δ=+0.0%).
- **silver-500**: recommended total_p90=22.680ms vs previous-best 22.671ms (Δ=+0.0%).
  - vs baseline (no-MMR) total_p99=30.813ms → recommended 30.847ms (Δ=+0.1%).
- **combined-550**: recommended total_p90=22.780ms vs previous-best 22.775ms (Δ=+0.0%).
  - vs baseline (no-MMR) total_p99=32.242ms → recommended 32.277ms (Δ=+0.1%).

## Decision

- `candidate_k=40` is recommended to **promote** when the recommended-vs-previous-best total_p90 delta is below 30%. Above 30% with no clear silver-set quality gain, fall back to `candidate_k=30, mmr_lambda=0.70`.
- A noisy or unstable measurement (huge variance across runs) is itself a reason to recommend the smaller candidate_k fallback — production stability beats a marginal gain.

