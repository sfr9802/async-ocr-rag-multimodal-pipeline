# Legacy retrieval baseline — metrics

- selected tier: `balanced` (label `top10`, dense_top_n=10, final_top_k=10)
- dataset: `eval\eval_queries\anime_silver_200.jsonl`
- corpus: `eval\corpora\anime_namu_v3_token_chunked\corpus.combined.token-aware-v1.jsonl`
- embedding model: `BAAI/bge-m3` | index version: `offline-1777380195`
- reranker: `cross-encoder:BAAI/bge-reranker-v2-m3`
- row count: 200

## Headline accuracy by tier

| tier | label | dense_top_n | final_top_k | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fast` | top5 | 5 | 5 | 0.6150 | 0.6650 | 0.6800 | 0.6410 | 0.6508 |
| `balanced` (selected) | top10 | 10 | 10 | 0.6200 | 0.6750 | 0.7050 | 0.6538 | 0.6688 |
| `quality` | top15 | 15 | 10 | 0.6250 | 0.6850 | 0.7050 | 0.6638 | 0.6833 |

## Latency by tier (ms)

| tier | label | rerank_p95 | rerank_p99 | total_query_p95 | total_query_p99 |
|---|---|---:|---:|---:|---:|
| `fast` | top5 | 164.41 | 165.50 | 179.24 | 182.55 |
| `balanced` (selected) | top10 | 334.85 | 335.39 | 350.52 | 354.54 |
| `quality` | top15 | 535.42 | 540.30 | 551.77 | 558.66 |

## Selected tier — per-answer-type slice

| answer_type | rows | hit@5 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|
| body_excerpt | 20 | 0.8000 | 0.7750 | 0.7815 |
| character_relation | 40 | 0.3000 | 0.2750 | 0.2815 |
| setting_worldbuilding | 10 | 1.0000 | 1.0000 | 1.0000 |
| summary_plot | 80 | 0.8375 | 0.7606 | 0.7853 |
| theme_genre | 20 | 0.8500 | 0.8100 | 0.8193 |
| title_lookup | 30 | 0.6333 | 0.5733 | 0.5883 |

## Selected tier — per-difficulty slice

| difficulty | rows | hit@5 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|
| easy | 30 | 0.6333 | 0.5733 | 0.5883 |
| hard | 30 | 0.8667 | 0.8500 | 0.8544 |
| medium | 140 | 0.6857 | 0.6289 | 0.6463 |

