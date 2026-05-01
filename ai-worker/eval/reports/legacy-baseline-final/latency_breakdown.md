# Phase 2A-L reranker latency breakdown

Stage-level latency profile for one retrieval-rerank run. Pure post-processing — never re-runs retrieval. CUDA synchronize is applied around the cross-encoder forward pass when the run was on a GPU; CPU-only runs see approximate forward times since async semantics don't apply.

- label: `top10`
- report: `eval\reports\phase2\2a_latency\rerank-top10\retrieval_eval_report.json`
- corpus: `eval\corpora\anime_namu_v3_token_chunked\corpus.combined.token-aware-v1.jsonl`
- reranker_name: cross-encoder:BAAI/bge-reranker-v2-m3
- reranker_model: `BAAI/bge-reranker-v2-m3`
- dense_top_n: 10 | final_top_k: 10 | reranker_batch_size: 16
- rows: 200
- rows_with_rerank_breakdown: 200
- rows_with_dense_retrieval_ms: 200

## Per-stage latency (ms)

| stage | n | avg | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense_retrieval_ms` | 200 | 15.14 | 14.99 | 15.83 | 16.68 | 18.64 | 22.23 |
| `pair_build_ms` | 200 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `tokenize_ms` | 200 | 2.45 | 2.44 | 2.68 | 2.76 | 2.95 | 3.12 |
| `forward_ms` | 200 | 326.35 | 328.05 | 329.05 | 329.51 | 330.07 | 330.21 |
| `postprocess_ms` | 200 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 |
| `total_rerank_ms` | 200 | 328.84 | 330.60 | 331.61 | 332.00 | 332.52 | 332.79 |
| `total_query_ms` | 200 | 346.73 | 348.34 | 350.13 | 350.52 | 354.54 | 355.03 |

