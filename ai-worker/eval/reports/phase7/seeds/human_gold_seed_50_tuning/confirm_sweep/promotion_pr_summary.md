# Phase 7.5 — production promotion PR summary

> Draft summary for the config-change PR that promotes the Phase 7.5
> retrieval config to production. **DO NOT auto-apply.** This file is
> a paste-target for the PR description; the actual rollout still goes
> through the standard config-change PR review.

## What this PR proposes

Flip the production retrieval config to MMR-enabled with a wider
candidate pool, on the *same* embedding-text variant the production
index already uses (`namu-v4-2008-2026-04-retrieval-title-section-mseq512`).

Effective env diff:

```diff
- AIPIPELINE_WORKER_RAG_USE_MMR=false   (default)
- AIPIPELINE_WORKER_RAG_CANDIDATE_K     (unset → default 30)
+ AIPIPELINE_WORKER_RAG_USE_MMR=true
+ AIPIPELINE_WORKER_RAG_CANDIDATE_K=40
+ AIPIPELINE_WORKER_RAG_MMR_LAMBDA=0.7000
  AIPIPELINE_WORKER_RAG_TOP_K=10        (unchanged)
```

This is **not** another embedding-text variant promotion. Phase 7.2
already promoted `retrieval_title_section` to the production default;
this PR only changes retrieval-config knobs (candidate_k, MMR(λ)) on
top of that index.

## Why this change

Phase 7.5's MMR confirm sweep evaluated 15 candidates over a
(candidate_k × λ) grid against a human-weighted gold-50 focus set
(50 queries curated to surface v4 subpage / section-level retrieval
failures) and a 500-query LLM-generated silver guardrail.

The metric-best winner (`cand_candk40_mmr_lambda060`,
primary_score=0.8130, +0.0804 vs baseline) cleared every hard
guardrail (silver hit@5, silver subpage_named, gold subpage_named,
gold main_work). The full λ-row at `candidate_k=40` plateaus at the
same primary_score (within ε=0.0200), so the lexicographic tie-break
to λ=0.60 is essentially arbitrary on this evaluation; we recommend
λ=0.70 instead because:

1. **Plateau equivalence.** Every λ ∈ {0.60, 0.65, 0.70, 0.75, 0.80}
   scores 0.8130 on the same `candidate_k=40` row.
2. **Continuity with prior work.** λ=0.70 was the Phase 7.x first-
   pass best; staying at 0.70 keeps the PR a single-knob flip ("turn
   MMR on, widen candidate_k") rather than a "and λ moved 0.10 too"
   change.
3. **Lower explanation cost.** Reviewers familiar with the Phase 7.x
   sweep do not need to re-build their intuition for a different λ
   when the metric is indifferent.

## Improvement summary (gold-50, vs `baseline_retrieval_title_section_top10`)

| metric                              | baseline | promoted | Δ           |
|-------------------------------------|---------:|---------:|------------:|
| gold primary_score                  |   0.7327 |   0.8130 |   +0.0804   |
| gold weighted_hit@5                 |   0.7858 |   0.9195 |   +0.1337   |
| gold weighted_MRR@10                |   0.6697 |   0.7040 |   +0.0343   |
| gold weighted_nDCG@10               |   0.7232 |   0.7642 |   +0.0410   |
| subpage_named weighted_hit@5        |   0.7107 |   0.9371 |   +0.2264   |
| subpage_generic weighted_hit@5      |   0.9301 |   1.0000 |   +0.0699   |
| main_work weighted_hit@5            |   0.6512 |   0.6977 |   +0.0465   |
| section_hit@5 (when defined)        |   0.0455 |   0.0227 |   −0.0227   |

Reframe these gains correctly: this is **not** a generic retrieval-
quality improvement claim. The gold-50 focus set was curated to
expose v4 subpage / named-subpage / section-level failures; these
deltas mean *we got better at the failures the set was curated for*.

## Silver guardrail (500 queries, LLM-generated overfitting check)

| metric              | baseline | promoted | Δ          |
|---------------------|---------:|---------:|-----------:|
| silver hit@5        |   0.7811 |   0.8358 |  +0.0547   |
| silver MRR@10       |   0.6544 |    —     |  (in-spec) |

Silver-500 is the *overfitting guardrail / sanity check*. The
promoted config improves silver hit@5 by +0.0547, well above the
−3pp regression veto threshold.

## Rejected candidates (sweep grid)

All 15 candidates in the (candidate_k × λ) grid passed every hard
guardrail. The metric-best lexicographic tie-break selected
`cand_candk40_mmr_lambda060` (λ=0.60); the production-promotion
record at `best_config.production_recommended.{env,json}` overrides
λ to 0.70 via the plateau-aware policy described above. The rejected
`cand_title_section_top10` candidate that appears in some logs is
the *previous* (Phase 7.0) embedding-text variant, present only as a
sanity check that `retrieval_title_section` is still the right index
choice — its large regression on both gold and silver confirms that
choice and is **not** part of this promotion proposal.

## Section_hit@5 caveat (NOT a blocker)

Page-level retrieval is meaningfully better under MMR; section-level
exact-match fell from 0.0455 → 0.0227 on a 22-row defined-only
subset. The metric base is tiny (≈4.5pp), which makes the metric
brittle — any reordering of neighbouring chunks within the same
page can flip section_hit without affecting answer quality. We
document the drop but do **NOT** treat it as a promotion blocker.
Section-aware reranking is the right follow-up; see
`eval/reports/phase7/phase7_6_section_aware_reranking_plan.md` for
the Phase 7.6 plan.

## Latency smoke check

Smoke check (replay over the cached candidate pool, 50 gold + 500
silver = 550 queries, MMR pass timed live) at
`latency_smoke_report.md`:

| set          | baseline (no-MMR)                | previous-best (k=30, λ=0.70)        | recommended (k=40, λ=0.70)           |
|--------------|---------------------------------:|------------------------------------:|-------------------------------------:|
| gold-50      | total p90 ≈ 23.07ms              | total p90 ≈ 23.10ms                 | total p90 ≈ 23.11ms (Δ vs prev ≈ 0%) |
| silver-500   | total p90 ≈ 22.64ms              | total p90 ≈ 22.67ms                 | total p90 ≈ 22.68ms (Δ vs prev ≈ 0%) |
| combined-550 | total p90 ≈ 22.74ms              | total p90 ≈ 22.78ms                 | total p90 ≈ 22.78ms (Δ vs prev ≈ 0%) |

**Verdict:** the post-hoc MMR cost grows from ~0.007ms (baseline) to
~0.044ms (k=40), which is well below the 30% p90 regression threshold
relative to the previous best. **Promote `candidate_k=40`.**

Honest scope of these numbers (re-stated from the smoke report):

  * `candidate_gen_ms` is the live FAISS+embed time the confirm sweep
    recorded at `pool_size=40`. For `candidate_k<40` configs it is a
    small upper bound (FAISS NN lookup is `O(log N + k)`).
  * `mmr_post_ms` is timed live in the smoke run.
  * **Reranker time is NOT measured** — the eval environment runs
    with the NoOp reranker. Confirm rerank cost separately before
    pasting these numbers into a production-latency dashboard.

## Rollback

Set `AIPIPELINE_WORKER_RAG_USE_MMR=false` and unset (or remove) the
`AIPIPELINE_WORKER_RAG_CANDIDATE_K` and `AIPIPELINE_WORKER_RAG_MMR_LAMBDA`
overrides:

```
AIPIPELINE_WORKER_RAG_USE_MMR=false
# unset AIPIPELINE_WORKER_RAG_CANDIDATE_K   (default 30)
# unset AIPIPELINE_WORKER_RAG_MMR_LAMBDA    (default 0.7)
```

The index variant cache (`namu-v4-2008-2026-04-retrieval-title-section-mseq512`)
itself was promoted in Phase 7.2 and is **NOT** touched by this PR;
rolling back this PR does not require an index rebuild.

## Post-promotion monitoring

Watch (on the first day of canary):

  1. **Retrieval p90/p99 latency** on the live ai-worker. If p90 jumps
     more than +30% vs the previous-week baseline, roll back.
  2. **First-hit-rank distribution** in the worker's RAG telemetry.
     The MMR pass mostly diversifies *within* the candidate window;
     a sudden shift in rank=1 rate without a corresponding gain in
     section / page coverage probably means MMR is firing on
     candidates it should not be reordering.
  3. **Section_hit caveat follow-up.** If Phase 7.6 section-aware
     reranking does not recover section_hit, escalate to a chunk-
     level generation audit. The promoted config is fine for *page-
     level* retrieval; section-level grounding is the open question.

## Pointers

  * Confirm sweep report:
    `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/confirm_sweep_report.md`
  * Production-recommended config:
    `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/best_config.production_recommended.{env,json}`
  * Latency smoke:
    `eval/reports/phase7/seeds/human_gold_seed_50_tuning/confirm_sweep/latency_smoke_report.md`
  * Phase 7.6 plan:
    `eval/reports/phase7/phase7_6_section_aware_reranking_plan.md`
