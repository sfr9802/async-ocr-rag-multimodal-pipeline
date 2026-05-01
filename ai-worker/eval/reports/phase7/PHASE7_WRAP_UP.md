# Phase 7 — wrap-up

> Phase 7.0–7.5 production recommendation 마무리 문서. Phase 7.6
> (section-aware reranking)은 후속 실험으로 분리되어 있고, 이 문서의
> production recommendation은 Phase 7.6 결과에 의존하지 않는다.

## Phase 7 시퀀스 요약

| phase | 결정                                                               | 산출물 위치                                            |
|-------|--------------------------------------------------------------------|--------------------------------------------------------|
| 7.0   | embedding-text variant `retrieval_title_section` A/B               | `7.0_retrieval_title_ab/`                              |
| 7.1   | reranker A/B (Phase 7.0 dense 위)                                  | `7.1_reranker_ab/`                                     |
| 7.2   | `retrieval_title_section`을 production 기본값으로 promote          | `7.2_production_embedding/`                            |
| 7.3   | retrieval confidence detector / failure classifier                 | `7.3_confidence_eval/`                                 |
| 7.4   | controlled recovery loop (hybrid + query rewrite)                  | `7.4_controlled_recovery/`                             |
| 7.5   | MMR confirm sweep + production promotion (이 PR의 핵심)            | `seeds/human_gold_seed_50_tuning/confirm_sweep/`       |
| 7.6   | section-aware reranking — **scaffold only, 후속 실험으로 분리**    | `phase7_6_section_aware_reranking_plan.md`, `seeds/.../section_rerank/` |

## Production recommended config (확정)

```
AIPIPELINE_WORKER_RAG_TOP_K=10
AIPIPELINE_WORKER_RAG_CANDIDATE_K=40
AIPIPELINE_WORKER_RAG_USE_MMR=true
AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000
```

- index variant cache: `namu-v4-2008-2026-04-retrieval-title-section-mseq512`
  (Phase 7.2에서 이미 promote, 이번 PR에서 변경 없음)
- selected_lambda_policy: `PLATEAU_TIE_BREAK_TO_PREVIOUS_BEST`

근거 요약:
- λ-grid 전체가 같은 plateau (5개 λ 모두 primary_score=0.8130)
- metric-best λ=0.60은 lexicographic tie-break 산물
- 이전 best와 일관된 λ=0.70으로 promote → PR이 단일-knob 변경
  (`USE_MMR=true` + `CANDIDATE_K=40`)이 되어 운영상 더 안정적
- candidate_k=40은 silver hit@5 +0.0547로 확장된 pool 효과 확인
- gold-50 focus set primary_score: 0.7327 → 0.8130 (+0.0804)
- subpage_named weighted_hit@5: 0.7107 → 0.9371 (+0.2264)

전체 sweep grid + 가드레일 + per-row breakdown은
`seeds/human_gold_seed_50_tuning/confirm_sweep/confirm_sweep_report.md`.

## Latency caveat (운영 latency 확정 근거 아님)

`seeds/human_gold_seed_50_tuning/confirm_sweep/latency_smoke_report.md`의
숫자는 **cached candidate pool elapsed_ms + post-hoc MMR 측정 기반**이다.

- `candidate_gen_ms`는 confirm sweep이 pool_size=40에서 측정한 live FAISS+embed
  값을 그대로 재사용한다. `candidate_k<40` 구성에서는 live 시간이 약간 더
  적게 나오므로 여기 숫자는 작은 upper bound로 봐야 한다.
- `mmr_post_ms`는 smoke run에서 live timing.
- **Reranker stage는 측정하지 않았고, 환경은 NoOp reranker로 돌고 있다.**

따라서 본 PR의 latency 문구는:

> "promotion을 막을 만한 smoke regression 없음
> (gold-50 / silver-500 / combined-550 모든 set에서 recommended-vs-previous-best
> total_p90 delta ≈ 0%, 30% regression veto threshold보다 충분히 아래)."

production end-to-end p90 / p99는 **canary monitoring 대상**이고, 이 smoke
숫자로 확정해서는 안 된다.

## Rollback plan

1. config flag 두/세 개를 baseline으로 되돌리면 끝:
   ```
   AIPIPELINE_WORKER_RAG_USE_MMR=false
   # AIPIPELINE_WORKER_RAG_CANDIDATE_K   unset (default 30)
   # AIPIPELINE_WORKER_RAG_MMR_LAMBDA    unset (default 0.7)
   ```
2. 인덱스 캐시(`namu-v4-2008-2026-04-retrieval-title-section-mseq512`)는
   Phase 7.2에서 promote된 상태로 유지된다. 본 PR rollback에서는
   인덱스 재빌드가 **필요하지 않다**.
3. 중간 fallback(rollback 직전 단계): `candidate_k=30, MMR=true,
   lambda=0.70`. Phase 7.x first-pass best 구성과 동일.

## Post-promotion monitoring checklist

canary 첫 24시간 동안 확인:

- [ ] **Retrieval p90 / p99 latency.** 직전 주간 baseline 대비 p90이
      +30% 이상 튀면 즉시 fallback (`candidate_k=30`) 또는 rollback.
- [ ] **First-hit rank 분포.** rank=1 비율이 갑자기 떨어졌는데
      page / section coverage가 같이 늘지 않으면 MMR이 잘못된
      후보를 reorder하고 있을 가능성 → 게이트 검토.
- [ ] **subpage_named hit 비율.** gold-50에서 +22pp 개선이 있었음.
      운영에서도 같은 방향(혹은 최소한 비-회귀)이어야 한다.
- [ ] **silver hit@5 회귀 감시.** −3pp 이상이면 silver overfitting
      veto 트리거.
- [ ] **section_hit 후속.** Phase 7.6 section-aware reranking이
      section_hit를 회복하지 못하면, chunk-level generation audit으로
      escalate. promoted config는 *page-level* retrieval에 한해
      이미 충분히 검증되었지만, section-level grounding은 아직
      open question임을 잊지 말 것.

## Test results

- Phase 7 신규/수정 테스트 **141 pass**:
  - `tests/test_phase7_human_gold_tune.py`
  - `tests/test_phase7_mmr_confirm_sweep.py`
  - `tests/test_phase7_latency_smoke.py`
  - `tests/test_phase7_section_aware_rerank.py`
  - `tests/test_finalize_phase7_5_production_recommendation.py`
- 전체 ai-worker test suite **1818 pass** (이번 변경으로 인한 회귀 없음).
- `scripts/finalize_phase7_5_production_recommendation.py` 재실행 시
  `best_config.production_recommended.{env,json}` + `confirm_sweep_report.md`의
  `## Production recommendation` splice가 **byte-identical** (idempotent).

## Phase 7.6 진입 조건

Phase 7.6 (section-aware reranking) 작업을 본격 가동하기 전에 아래가 모두
참이어야 한다:

- [ ] Phase 7.5 production recommendation PR이 main에 머지되어
      운영 config가 promoted 상태가 되었다.
- [ ] canary 24h 모니터링 체크리스트 5개 항목 모두 OK.
- [ ] section_hit 회복 vs metric brittleness 가설 중 하나를 우선 가설로
      선택했다 (Phase 7.6 grid가 두 가설을 모두 나누어 검증하도록 설계됨).
- [ ] `run_phase7_section_aware_rerank.py`의 `--score`가 미구현임을
      확인. 본격 sweep 실행은 별도 PR.

Phase 7.6 산출물(scaffold만, 본 PR에 같이 land):

- 계획: `phase7_6_section_aware_reranking_plan.md`
- 그리드: `seeds/human_gold_seed_50_tuning/section_rerank/section_rerank_grid.{json,md}`
- 하네스: `eval/harness/phase7_section_aware_rerank.py`
- CLI: `scripts/run_phase7_section_aware_rerank.py` (`--score` 미구현,
  호출 시 명확한 에러 메시지와 exit code 2 반환)
- 단위 테스트: `tests/test_phase7_section_aware_rerank.py`

## 산출물 인덱스 (이 PR이 다루는 범위)

- 본 wrap-up: `eval/reports/phase7/PHASE7_WRAP_UP.md`
- PR description paste-target:
  `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/promotion_pr_summary.md`
- Confirm sweep:
  `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/confirm_sweep_report.md`
- Production-recommended config:
  `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/best_config.production_recommended.{env,json}`
- Latency smoke:
  `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/latency_smoke_report.md`
- Phase 7.6 plan + grid scaffold (이 PR 외 후속 실험의 사전 작업):
  `eval/reports/phase7/phase7_6_section_aware_reranking_plan.md`,
  `eval/reports/phase7/seeds/human_gold_seed_50_tuning/section_rerank/`
