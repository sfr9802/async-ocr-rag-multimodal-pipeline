# Study `rag-wide-mmr-offline-v1` — 5-round summary

5 round의 Optuna round-refinement 결과 종합. wide-MMR-titlecap diagnostic sweep
(`eval/reports/retrieval-wide-mmr-titlecap-20260429-1353/`)에서 도출된 narrow
search space를 갖고 출발해, retrieval pipeline의 가장 effective한 hyperparam
조합을 찾는 것을 목표로 했음.

## Final best

| 항목 | 값 |
|---|---|
| Best objective (mean_mrr_at_10) | **0.6745** |
| Recipe | `candidate_k=100, rerank_in=16, use_mmr=true, title_cap_rerank_input=1` |
| Free axes | `final_top_k ∈ {5,8,10}`, `mmr_k ∈ {32,48,64,80}`, `mmr_lambda ∈ [0.55, 0.75]`, `title_cap_final ∈ {1,2,3}` (모두 무관) |
| Subset | first 100 rows of `eval/eval_queries/anime_silver_200.jsonl` |
| Total trials | 76 (12 + 16 + 16 + 16 + 16) |

## Round-by-round narrative

| round | n_trials | 변경 | best_value | std | 핵심 발견 |
|---|---:|---|---:|---:|---|
| 01 | 12 | initial | 0.7600 (hit@5) | ~0 | hit@5 saturate. metric 교체 필요 |
| 02 | 16 | objective → MRR | 0.6717 | 0.0023 | **rerank_in 0.98 importance** (16>24>32 monotone) |
| 03 | 16 | rerank_in 고정 16 | 0.6717 | 0.0 | 50-row subset에서 rest axis 신호 없음 |
| 04 | 16 | subset 50→100 | 0.6745 | 0.0025 | **(cand_k=100, cap_rr=1) winner combo** 발견 |
| 05 | 16 | cand_k=100, cap_rr=1 narrow | 0.6745 | 0.0 | residual axes 모두 무신호 → converged |

## Round 02 핵심 표 — rerank_in의 결정성

| rerank_in | 시도된 trial | 모든 trial이 도달한 값 |
|---|---:|---|
| 16 | 10 | 0.6717 |
| 24 | 4  | 0.6673 |
| 32 | 2  | 0.6640 |

해석: rerank_in 단일 값으로 MRR이 deterministic하게 결정됨. importance 0.9827.

## Round 04 핵심 표 — (cand_k × cap_rr) 인터랙션

| candidate_k | cap_rr=1 | cap_rr=2 | cap_rr=3 |
|---|---|---|---|
| 100 | **0.6745** (9 trial) | 0.6695 (1) | 0.6695 (3) |
| 200 | 0.6695 (1) | 0.6695 (1) | 0.6695 (1) |

해석: winner combo는 `(cand_k=100, cap_rr=1)`. 둘 중 하나만 만족하면 0.6695 plateau.

## Round 05 — 잔여 axis 무신호 확인

cand_k=100, cap_rr=1, rerank_in=16 고정 후 16 trial. std_value = 0.0 (모두 0.6745). 
- final_top_k 5/8/10: 차이 없음
- mmr_k 32-80: 차이 없음
- mmr_lambda 0.568-0.748: 차이 없음
- title_cap_final 1/2/3: 차이 없음

이 axes는 100-row subset에서 의미 없음. 더 큰 subset에서는 다를 수 있으나 5-round budget 안에서는 측정 불가.

## Anti-pattern 적용 정리

- **A10 (narrow against unsampled edge)**: mmr_lambda 양 edge가 5 round 내내 UNSAMPLED — 항상 narrow 거부.
- **A11 (narrow with axis evidence)**: rerank_in (round 2→3), candidate_k + cap_rr (round 4→5)에 적용. 모두 deterministic value bucket 증거 보유.
- **A12 (no narrowing without evidence)**: importance가 미미한 axes (mmr_k, final_top_k, cap_final)는 narrow 안 함.
- **A13 (objective swap masking issue)**: round 01→02 metric 교체는 saturate가 명백할 때만 수행, provenance에 명시.
- **A14 (premature termination)**: round 05 std=0에서 study terminate — convergence 증거 충분.

## Wide-MMR diagnostic과의 비교

이 Optuna study는 wide-MMR-titlecap diagnostic의 "narrow Optuna round" 제안을 그대로 실행. diagnostic 200-row 결과와 study 100-row 결과 비교:

| metric | diagnostic (200 row, cap2_top8) | study best (100 row, cand=100/cap=1) |
|---|---|---|
| hit@5 | 0.7400 | 0.7600 (round_01에서 측정) |
| MRR@10 | 0.6699 | 0.6745 |

**Caveat**: subset이 100 row vs 200 row로 다르므로 직접 비교 불가. study의 winner가 200-row 전체에서도 우월한지는 별도 confirm run 필요.

## 다음 단계 제안 (이번 5-round 외)

1. **Confirm round** — `(cand=100, cap_rr=1)` 조합을 200-row 전체에서 평가해, 100-row subset 효과인지 일반 효과인지 판정.
2. **Embedding text variant** axis 추가 — 현재 raw 고정. wide-MMR diagnostic은 cand@50=0.80 ceiling이 embedding 측 병목임을 시사.
3. **Per-query-type optimization** — heuristic tagging draft (eval/eval_queries/anime_silver_200.query_type_draft.jsonl) manual review 후 byQueryType별 best recipe 탐색.
4. **Larger silver dataset** — 200-row silver를 500/1000으로 확장하면 mmr_lambda / final_top_k / mmr_k 각각의 신호가 surface될 가능성.

## 산출물

- `eval/experiments/wide_mmr_offline_rounds/round_NN_config.json` (NN=01..05)
- `eval/experiments/wide_mmr_offline_rounds/round_NN_analysis.md` (NN=01..05)
- `eval/experiments/run_output/wide_mmr_round_NN_bundle.json` (NN=01..05)
- `eval/experiments/run_output/wide_mmr_round_NN_llm_input.md` (NN=01..05)
- `eval/experiments/wide_mmr_offline_rounds/STUDY_SUMMARY.md` (이 파일)

## evaluator

- `ai-worker/eval/tune_eval_offline.py` (신규) — production DB에 의존하지 않는 offline evaluate. wide-MMR adapter + cached FAISS + silver_200 사용. 환경변수로 metric/subset/rerank/embed 파라미터 제어.
