# Phase 7.6 — section-aware reranking plan

> Follow-up to Phase 7.5's MMR confirm sweep. Phase 7.5 proposes
> promoting `candidate_k=40 + use_mmr=true + λ=0.70` on the
> production-default `retrieval_title_section` index. The page-level
> retrieval improvements were significant; the section-level
> exact-match (section_hit@5) regressed from 0.0455 → 0.0227 on a
> small defined-only subset. Phase 7.6 is the experiment that figures
> out whether that section-level regression is real or an artefact
> of a brittle small-base metric.

## Problem statement

Phase 7.5 produced large page-level retrieval gains:

  * gold weighted_hit@5: 0.7858 → 0.9195 (+0.1337)
  * subpage_named weighted_hit@5: 0.7107 → 0.9371 (+0.2264)
  * silver hit@5: 0.7811 → 0.8358 (+0.0547)

But section_hit@5 (the rate at which the retrieved chunks include
one whose `section_path` matches the gold row's `expected_section_
path`) fell from 0.0455 → 0.0227. The defined-only subset of the
gold-50 set is 22 rows, so the metric base is tiny — flipping one
row is ≈4.5pp.

We need to answer: **does the page-level gain come at the cost of
real section-level grounding, or is section_hit@5 just brittle on
this base?** The Phase 7.6 experiment is designed to surface that
distinction explicitly: we want to keep the page-level wins from
Phase 7.5 while also improving (or at least restoring) section
coverage on the defined-only subset.

## Goals

1. Maintain the Phase 7.5 page-level retrieval improvements
   (gold primary_score, weighted_hit@5, silver hit@5,
   subpage_named weighted_hit@5).
2. Recover or improve section_hit@5 / section_hit@10 on the
   defined-only subset.
3. Validate whether MMR's diversification has any negative effect
   on RAG generation answerability — separate from the section
   metric, and distinct from page-level recall.
4. Surface candidate strategies that are *production-deployable*
   (i.e. do not require oracle expected_section_path / supporting
   chunk id at inference time) vs *diagnostic-only* strategies (we
   only want to know whether the metric *can* be improved, even if
   the improvement requires oracle access).

## Candidate strategies

Each candidate is documented with its production deployability.
Diagnostic-only candidates are explicitly NOT for production rollout
— their job is to set an upper-bound and validate that the metric
is improvable at all.

### 1. Section prefix / path bonus  (DIAGNOSTIC ONLY)

**Mechanism.** For each retrieved doc, compute a similarity score
against the gold row's `expected_section_path` (prefix match,
substring match, or token overlap). Add a small bonus to the base
relevance score and re-rank.

**Why diagnostic only.** The strategy depends on knowing the
expected_section_path at inference time. In production we do not
have that — we have a query and the candidate pool. So this
candidate's job is to set the upper bound: *if* we had perfect
section knowledge, *how much* could section_hit@5 be recovered?

**Variants in the harness:**
  * `section_bonus_0.05` — small bonus, shouldn't perturb page-level
  * `section_bonus_0.10` — medium
  * `section_bonus_0.15` — large; tests whether overshoot harms
    page-level metrics

**Risk.** Overfitting to the gold-50 expected_section_path. Mitigate
by also reporting the metric on the silver-500 set (the silver rows
have an expected_section_path most of the time) and watching for
silver guardrail regressions.

### 2. Supporting chunk proximity bonus  (DIAGNOSTIC ONLY)

**Mechanism.** When the gold row carries a
`human_supporting_chunk_id` annotation, give a bonus to candidates
in the retrieved pool that share the same page AND lie within a
small chunk-id distance of the supporting chunk. This is a stronger
signal than section_path because it operates at chunk-id level.

**Why diagnostic only.** The annotation is not available in
production retrieval. Same as (1), the goal is to validate that
the metric is recoverable in principle.

**Caveat.** Gold-50 has only ≈22 rows with a defined supporting
chunk id, so this candidate over-indexes on a small set. Use it as
a sanity-check upper bound, not a variant ranking.

### 3. Page-first then section rerank  (PRODUCTION-DEPLOYABLE)

**Mechanism.** Two-pass:

  1. First pass: page-level candidate diversification (the Phase
     7.5 promoted config — `candidate_k=40, MMR(λ=0.70)`,
     deduplicated to one chunk per page).
  2. Second pass: for each page that landed in the top-k, re-fetch
     the top-N chunks within that page and rerank them by either
     a) section-name token overlap with the query, or
     b) the cross-encoder reranker score recomputed on the
        within-page candidate set.

**Why production-deployable.** The second-pass scoring uses only
the query and the chunk metadata — no oracle. The cross-encoder
path costs an extra rerank pass per top-k page; the section-name
overlap path is essentially free.

**Variants in the harness:**
  * `page_first_section_rerank_overlap` — section-name token overlap
  * `page_first_section_rerank_xenc` — cross-encoder rerank within
    page (gated on the reranker variant being available)

**Risk.** The second pass can re-introduce same-page duplicates that
the Phase 7.5 MMR pass removed. Mitigate with a page-level cap
(N≤2 chunks per page in the final top-k).

### 4. Same-page chunk rerank  (PRODUCTION-DEPLOYABLE)

**Mechanism.** Lightweight variant of (3): take the Phase 7.5
top-k as-is, then for any page that contributes a chunk to the
top-k, query the page's full chunk list and pick the *best* chunk
by section-name overlap with the query (or by mean token overlap).

**Trade-off vs (3).** (3) re-fetches and reranks the per-page
candidate window; (4) only swaps the *picked* chunk within the
already-included page. (4) is cheaper but cannot re-rank pages that
should have been in the top-k but weren't.

### 5. Generation-grounded answerability audit prep  (Phase 7.7 hook)

**Mechanism.** Not a reranking strategy. For each gold-50 row,
freeze the top-5 retrieval result of the Phase 7.5 promoted config
and the section-aware reranked variants. Hand the (query, top-5)
pair to a generation prompt and audit whether the produced answer
is correctly grounded in retrieved context. Section-level metrics
are a *proxy* for answerability; this audit measures answerability
directly.

**Output.** A small (≈50-row) gold-grounded answerability dataset
with one row per (query, variant) pair. Used as the input to
Phase 7.7. Not part of the Phase 7.6 promotion decision itself.

## Metrics to record per variant

  * `page_hit@5`, `page_hit@10`
  * `weighted_page_hit@5`, `weighted_page_hit@10`
  * `MRR@10`, `nDCG@10`
  * `section_hit@5`, `section_hit@10`
  * `chunk_hit@10` (when defined)
  * answerability proxy score (Phase 7.7 prep — populated only if a
    grounded-answerability judge is available; otherwise None)
  * Bucket breakdown: main_work / subpage_named / subpage_generic
  * Group breakdown: STRICT / SOFT / AMBIGUOUS / ABSTAIN

The headline objective remains the existing
`primary_score = 0.45 * weighted_hit@5 + 0.35 * weighted_MRR@10 +
0.20 * weighted_nDCG@10`. section_hit is auxiliary.

## Guardrails

A variant is **rejected** when:

  * gold `primary_score` drops more than 2pp below the Phase 7.5
    production-recommended config (`candidate_k=40, λ=0.70`).
  * silver `hit@5` drops more than 3pp below the Phase 7.5
    production-recommended config (matches the existing silver
    overfitting threshold).
  * `subpage_named weighted_hit@5` drops more than 5pp below the
    Phase 7.5 promoted level (subpage_named is what gold-50 was
    curated to surface — a regression is non-negotiable).
  * page-level `weighted_hit@5` drops below baseline. A variant
    that "recovers section_hit but breaks page recall" is solving
    the wrong problem.

## Experimental procedure

  1. Replay the cached candidate pool from Phase 7.5 (no FAISS
     re-run needed for sections 1, 2, 3a, 4 — the pool is already
     wide enough). Section 3b (cross-encoder within-page rerank)
     does need a live reranker pass over the in-page candidate
     window.
  2. For each candidate strategy, run the gold-50 + silver-500
     evaluation through the Phase 7 harness with the strategy's
     scoring function plugged into the post-hoc rerank step.
  3. Score against the Phase 7.5 production-recommended config as
     the *new baseline* (not the original Phase 7.4 baseline). The
     question being asked is "can section-aware rerank improve on
     Phase 7.5?", not "is Phase 7.5 itself an improvement?".
  4. Run plateau analysis on any winner — same shape as Phase 7.5,
     same epsilon, to keep the promotion signal stable.
  5. Generate a Phase 7.6 confirm sweep report and a Phase 7.6
     production recommendation following the same artefact contract
     as Phase 7.5 (`best_config.confirmed.{env,json}` +
     `best_config.production_recommended.{env,json}`).

## Out of scope for Phase 7.6

  * Phase 7.7 answerability judge: hooked but not run. Phase 7.6
    only prepares the (query, top-5) frozen dataset.
  * Embedding-text variant changes: the production index stays at
    `retrieval_title_section`. Phase 7.6 is purely a rerank-policy
    experiment.
  * Cross-encoder reranker model swap: any candidate that involves
    a different reranker model is out of scope; section 3b uses the
    *same* `bge-reranker-v2-m3` checkpoint Phase 7.4 promoted, just
    invoked at a different stage.
  * Production rollout. Phase 7.6 lands an *eval harness* + report;
    it does not propose a config change. A production rollout is a
    separate Phase 7.6.x PR if a winner survives the guardrails.

## Deliverables

  * `eval/harness/phase7_section_aware_rerank.py` — strategy specs,
    grid generation, scoring, renderers (lands in this commit as a
    scaffold; full strategy logic populated in the Phase 7.6 work).
  * `scripts/run_phase7_section_aware_rerank.py` — CLI entry point.
  * `tests/test_phase7_section_aware_rerank.py` — unit tests for the
    strategy specs, scoring functions, and renderer.
  * `eval/reports/phase7/seeds/human_gold_seed_50_tuning/
    section_rerank/` — output directory for the experiment artefacts.
