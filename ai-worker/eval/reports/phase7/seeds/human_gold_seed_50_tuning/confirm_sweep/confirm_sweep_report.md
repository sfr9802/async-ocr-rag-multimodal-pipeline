# Phase 7.5 — MMR confirm sweep

> This evaluation is NOT a representative retrieval-quality benchmark. It is a human-weighted focus set drawn from queries_v4_llm_silver_500, designed to surface v4 subpage / section-level retrieval failures. primary_score improvements only mean 'we got better at the gold-50 subpage / named-subpage failures this set was curated to expose'.

## Promotion target clarification

Promotion target: this evaluation tests retrieval CONFIG changes (candidate_k, use_mmr, mmr_lambda) on top of the production-default retrieval_title_section index. It does NOT test another embedding-text variant promotion. The rejected `cand_title_section_top10` candidate is the *previous* (Phase 7.0) embedding-text variant and is included only as a sanity check that retrieval_title_section is still the right index choice — its large regression on both gold and silver confirms that decision and is not part of this promotion proposal.

- baseline: **baseline_retrieval_title_section_top10** — primary_score=0.7327
- previous best (Phase 7.x first pass): **cand_top10_mmr_lambda07**
  - primary_score=0.7948 (+0.0622)
- confirmed best: **cand_candk40_mmr_lambda060** — primary_score=0.8130 (+0.0804 vs baseline)
- promotion recommended: **YES** — confirmed_best=cand_candk40_mmr_lambda060, primary_score=0.813022 (+0.080353 vs baseline). Plateau: PLATEAU_OK. All hard guardrails (silver hit@5, silver subpage_named, gold subpage_named hold, gold main_work) passed.

## Headline comparison

| metric | baseline | previous best | confirmed best |
|---|---:|---:|---:|
| gold primary_score | 0.7327 | 0.7948 | 0.8130 |
| gold weighted_hit@5 | 0.7858 | 0.8922 | 0.9195 |
| gold weighted_MRR@10 | 0.6697 | 0.6949 | 0.7040 |
| gold weighted_nDCG@10 | 0.7232 | 0.7506 | 0.7642 |
| silver hit@5 | 0.7811 | 0.8337 | 0.8358 |
| silver MRR@10 | 0.6544 | — | — |
| subpage_named weighted_hit@5 | 0.7107 | 0.9371 | 0.9371 |
| subpage_generic weighted_hit@5 | 0.9301 | 0.9301 | 1.0000 |
| main_work weighted_hit@5 | 0.6512 | 0.6977 | 0.6977 |
| section_hit@5 | 0.0455 | 0.0227 | 0.0227 |

## Sweep grid (15 candidates)

| variant | candidate_k | λ | primary | wh@5 | wMRR@10 | silver_h@5 | subpage_named_wh@5 | main_work_wh@5 | section_h@5 | accepted | rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| cand_candk20_mmr_lambda060 | 20 | 0.60 | 0.7916 (+0.0590) | 0.8922 | 0.6910 | 0.8316 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk20_mmr_lambda065 | 20 | 0.65 | 0.7916 (+0.0590) | 0.8922 | 0.6910 | 0.8316 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk20_mmr_lambda070 | 20 | 0.70 | 0.7916 (+0.0590) | 0.8922 | 0.6910 | 0.8316 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk20_mmr_lambda075 | 20 | 0.75 | 0.7916 (+0.0590) | 0.8922 | 0.6910 | 0.8316 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk20_mmr_lambda080 | 20 | 0.80 | 0.7916 (+0.0590) | 0.8922 | 0.6910 | 0.8316 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk30_mmr_lambda060 | 30 | 0.60 | 0.7948 (+0.0622) | 0.8922 | 0.6949 | 0.8337 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk30_mmr_lambda065 | 30 | 0.65 | 0.7948 (+0.0622) | 0.8922 | 0.6949 | 0.8337 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk30_mmr_lambda070 | 30 | 0.70 | 0.7948 (+0.0622) | 0.8922 | 0.6949 | 0.8337 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk30_mmr_lambda075 | 30 | 0.75 | 0.7948 (+0.0622) | 0.8922 | 0.6949 | 0.8337 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk30_mmr_lambda080 | 30 | 0.80 | 0.7948 (+0.0622) | 0.8922 | 0.6949 | 0.8337 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk40_mmr_lambda060 | 40 | 0.60 | 0.8130 (+0.0804) | 0.9195 | 0.7040 | 0.8358 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk40_mmr_lambda065 | 40 | 0.65 | 0.8130 (+0.0804) | 0.9195 | 0.7040 | 0.8358 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk40_mmr_lambda070 | 40 | 0.70 | 0.8130 (+0.0804) | 0.9195 | 0.7040 | 0.8358 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk40_mmr_lambda075 | 40 | 0.75 | 0.8130 (+0.0804) | 0.9195 | 0.7040 | 0.8358 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |
| cand_candk40_mmr_lambda080 | 40 | 0.80 | 0.8130 (+0.0804) | 0.9195 | 0.7040 | 0.8358 | 0.9371 | 0.6977 | 0.0227 | ✓ | — |

## Guardrail warnings

- **cand_candk20_mmr_lambda060**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk20_mmr_lambda065**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk20_mmr_lambda070**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk20_mmr_lambda075**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk20_mmr_lambda080**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk30_mmr_lambda060**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk30_mmr_lambda065**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk30_mmr_lambda070**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk30_mmr_lambda075**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk30_mmr_lambda080**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk40_mmr_lambda060**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk40_mmr_lambda065**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk40_mmr_lambda070**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk40_mmr_lambda075**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.
- **cand_candk40_mmr_lambda080**:
  - `SECTION_RETRIEVAL_WARNING` (section_hit_at_5, bucket=None): baseline=0.0455 → candidate=0.0227 (Δ=-0.0227; threshold 0.5). gold section_hit@5 fell from 0.0455 to 0.0227 (<= 50% of baseline). Documented as a caveat — page_hit improved, section-level exact match regressed on a tiny base. Not a promotion blocker; needs section-aware reranking or chunk-level audit to validate.

## Plateau analysis

- status: **PLATEAU_OK**
- best variant: `cand_candk40_mmr_lambda060` (candidate_k=40, λ=0.60)
- λ-neighbours (same candidate_k row):
  - λ=0.65: primary_score=0.8130
- λ=0.60 winner is within 0.0200 primary_score of all immediate neighbours (1 compared); treated as a stable plateau.

## Section_hit@5 caveat

Section_hit@5 caveat: page-level retrieval is meaningfully better under MMR; section-level exact-match fell relative to baseline. Baseline section_hit@5 is tiny (≈4.5pp on a 22-row defined-only subset), which makes the metric brittle: any reordering of neighbouring chunks within the same page can flip section_hit without affecting answer quality. We therefore document the drop but do NOT treat it as a promotion blocker. Section-aware reranking or chunk-level generation audit is the right follow-up to validate this assumption.

## Recommendation

- **Promote** retrieval config `top_k=10, candidate_k=40, use_mmr=true, mmr_lambda=0.60` on the production-default `namu-v4-2008-2026-04-retrieval-title-section-mseq512` index.

### Rollback

- Set `AIPIPELINE_WORKER_RAG_USE_MMR=false` and remove the `AIPIPELINE_WORKER_RAG_CANDIDATE_K` override to restore the exact pre-promotion retrieval behaviour. The index variant itself was promoted in Phase 7.2 and is NOT touched by this change.

## Production recommendation

- confirmed best by metric: `cand_candk40_mmr_lambda060` (candidate_k=40, MMR enabled, λ-plateau)
- production recommended lambda: **0.7000**
- production recommended variant: `cand_candk40_mmr_lambda070`
- selected_lambda_policy: `PLATEAU_TIE_BREAK_TO_PREVIOUS_BEST`
- reason: λ-row at candidate_k=40 plateaus within epsilon=0.0200 of the metric-best primary_score; recommended λ=0.7000 matches the prior best, lowering the PR explanation cost vs the lexicographic tie-break value λ=0.6000.
- plateau set (candidate_k=40): λ=0.60 → 0.8130, λ=0.65 → 0.8130, λ=0.70 → 0.8130, λ=0.75 → 0.8130, λ=0.80 → 0.8130

### Why not the metric-best λ?

- the lexicographic tie-break value is λ=0.6000
- λ=0.7000 sits on the same plateau (within epsilon)
- λ=0.70 (the prior Phase 7.x best) is more familiar to a reviewer; promoting at that value keeps the PR a single-knob change (`USE_MMR=true`, widen `CANDIDATE_K`) instead of also moving λ

### Improvement summary (vs baseline)

- gold primary_score: 0.7327 → 0.8130 (+0.0804)
- gold weighted_hit@5: 0.7858 → 0.9195 (+0.1337)
- silver hit@5: 0.7811 → 0.8358 (+0.0547)
- subpage_named weighted_hit@5: 0.7107 → 0.9371 (+0.2264)
- main_work weighted_hit@5: 0.6512 → 0.6977 (+0.0465)

### Known caveat: section_hit@5

- baseline 0.0455 → recommended 0.0227 (Δ=-0.0227)
- production blocker? **NO** — caveat documented in the main report; deferred to Phase 7.6 section-aware reranking.

### Recommended env snippet

```
AIPIPELINE_WORKER_RAG_TOP_K=10
AIPIPELINE_WORKER_RAG_CANDIDATE_K=40
AIPIPELINE_WORKER_RAG_USE_MMR=true
AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000
```

### Fallback configs

- intermediate fallback (smaller candidate_k pool, same λ): `candidate_k=30, mmr_lambda=0.7000, use_mmr=true` — matches the Phase 7.x first-pass best.
- full rollback (restore baseline retrieval path): set `AIPIPELINE_WORKER_RAG_USE_MMR=false` and unset / drop `AIPIPELINE_WORKER_RAG_CANDIDATE_K` + `AIPIPELINE_WORKER_RAG_MMR_LAMBDA` overrides.


## Reminders

- gold-50 is a *human-weighted focus set* drawn from queries_v4_llm_silver_500. Improvements only mean we got better at the subpage / named-subpage failures the gold-50 set was curated to expose. NOT a generic retrieval benchmark.
- silver-500 is LLM-generated and serves as the **overfitting guardrail / sanity check**, NOT the primary tuning objective.
- production retrieval config MUST NOT be changed off this report alone — promote via the standard config-change PR.

