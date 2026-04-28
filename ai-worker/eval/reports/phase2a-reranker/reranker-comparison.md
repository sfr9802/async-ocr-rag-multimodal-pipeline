# Phase 2A reranker comparison

- slices: 5

## Caveats

- reranker는 candidate set 안의 순서만 바꾼다 — candidate에 정답이 없으면 회복할 수 없다.
- candidate_recall@N (= dense-only top-N hit@N) 이 reranker 성능 상한이다.
- dense top-N을 키우면 latency / GPU memory 비용이 증가한다 — 회복 가능한 query 수와 비교해서 결정할 것.
- B1 (combined-old) 와 B2 (combined-token-aware-v1) 는 chunk granularity가 다르다 — candidate population이 동일하지 않다.
- rerank_latency 는 cross-encoder predict 만의 wall-clock — bi-encoder + FAISS 부분은 mean_retrieval_ms 에 별도로 잡혀 있다.
- 이 리포트는 어떤 설정도 production default 로 승격하지 않는다 — 결과는 evidence, 결정은 별도.

## Headline metrics

| label | corpus | reranker | dense_top_n | final_top_k | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1 dense (combined-old) | corpus.prefix-v1.inline-edit-v1.jsonl | noop | 30 | 10 | 0.5600 | 0.6700 | 0.6850 | 0.6167 | 0.6428 |
| B2 dense (token-aware-v1) | corpus.combined.token-aware-v1.jsonl | noop | 30 | 10 | 0.5400 | 0.6650 | 0.6800 | 0.6044 | 0.6314 |
| B2 dense top50 (candidate-recall) | corpus.combined.token-aware-v1.jsonl | noop | 30 | 50 | 0.5400 | 0.6650 | 0.6800 | 0.6044 | 0.6314 |
| B2 rerank top20 | corpus.combined.token-aware-v1.jsonl | cross-encoder:BAAI/bge-reranker-v2-m3 | 20 | 10 | 0.6050 | 0.6800 | 0.7000 | 0.6526 | 0.6748 |
| B2 rerank top50 | corpus.combined.token-aware-v1.jsonl | cross-encoder:BAAI/bge-reranker-v2-m3 | 50 | 10 | 0.6150 | 0.7000 | 0.7150 | 0.6657 | 0.6885 |

## Candidate / extra hit cutoffs

| label | hit@10 | hit@20 | hit@50 |
|---|---:|---:|---:|
| B1 dense (combined-old) | - | - | - |
| B2 dense (token-aware-v1) | - | - | - |
| B2 dense top50 (candidate-recall) | 0.7150 | 0.7700 | 0.8000 |
| B2 rerank top20 | - | - | - |
| B2 rerank top50 | - | - | - |

## Latency + cost

| label | reranker_batch | retrieval_p95_ms | rerank_p95_ms | mean_avg_ctx_tokens | dup_rate (top-k) |
|---|---:|---:|---:|---:|---:|
| B1 dense (combined-old) | - | 14.00 | - | 531.0490 | 0.2740 |
| B2 dense (token-aware-v1) | - | 16.53 | - | 292.7680 | 0.2460 |
| B2 dense top50 (candidate-recall) | - | 20.52 | - | 315.0784 | 0.2956 |
| B2 rerank top20 | 16 | 722.98 | 706.01 | 294.2020 | 0.2600 |
| B2 rerank top50 | 16 | 1854.70 | 1839.56 | 295.0295 | 0.2675 |

