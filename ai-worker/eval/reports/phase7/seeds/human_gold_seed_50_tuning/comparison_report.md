# Phase 7.x — human-weighted gold seed 50 + silver 500 tuning

> This evaluation is NOT a representative retrieval-quality benchmark. It is a human-weighted focus set drawn from queries_v4_llm_silver_500, designed to surface v4 subpage / section-level retrieval failures. primary_score improvements only mean 'we got better at the gold-50 subpage / named-subpage failures this set was curated to expose'.

## Promotion target clarification

Promotion target framing: this evaluation tests retrieval CONFIG changes (candidate_k, use_mmr, mmr_lambda) on top of the production-default retrieval_title_section index. It does NOT test another embedding-text variant promotion. The `cand_title_section_top10` candidate, when present, is the *previous* (Phase 7.0) embedding-text variant — included only as a sanity check that retrieval_title_section is still the right choice. Any regression on that candidate confirms the Phase 7.2 embedding-text decision and is NOT a justification for promoting it.

Reminder: gold-50 is a *human-weighted focus set* drawn from queries_v4_llm_silver_500, NOT a generic retrieval benchmark — primary_score improvements only mean we got better at the subpage / named-subpage failures the set was curated to expose. silver-500 is LLM-generated and acts as the **overfitting guardrail / sanity check**, NOT the primary tuning objective.

- baseline variant: **baseline_retrieval_title_section_top10**
- gold-50 distribution: STRICT=30 / SOFT=14 / AMBIGUOUS_PROBE=3 / ABSTAIN_TEST=3
- best candidate: **cand_top10_mmr_lambda07** — primary_score=0.794832 (+0.062163 vs baseline) with no blocking guardrail.

## Headline (gold-50, weighted)

| metric | baseline | cand_top10_candk30 | cand_top10_mmr_lambda07 | cand_title_section_top10 |
|---|---:|---:|---:|---:|
| primary_score | 0.7327 | 0.7327 (+0.0000) | 0.7948 (+0.0622) | 0.5654 (-0.1672) |
| weighted_hit@1 | 0.5675 | 0.5675 (+0.0000) | 0.5675 (+0.0000) | 0.4147 (-0.1528) |
| weighted_hit@3 | 0.7312 | 0.7312 (+0.0000) | 0.8022 (+0.0709) | 0.6057 (-0.1255) |
| weighted_hit@5 | 0.7858 | 0.7858 (+0.0000) | 0.8922 (+0.1064) | 0.6057 (-0.1801) |
| weighted_hit@10 | 0.8922 | 0.8922 (+0.0000) | 0.9195 (+0.0273) | 0.7231 (-0.1692) |
| weighted_MRR@10 | 0.6697 | 0.6697 (+0.0000) | 0.6949 (+0.0252) | 0.5142 (-0.1555) |
| weighted_nDCG@10 | 0.7232 | 0.7232 (+0.0000) | 0.7506 (+0.0274) | 0.5644 (-0.1588) |
| strict_hit@5 | 0.8333 | 0.8333 (+0.0000) | 0.9333 (+0.1000) | 0.7000 (-0.1333) |
| strict_MRR@10 | 0.7125 | 0.7125 (+0.0000) | 0.7381 (+0.0256) | 0.5833 (-0.1292) |
| hit@1 (positive, unweighted) | 0.5455 | 0.5455 (+0.0000) | 0.5455 (+0.0000) | 0.3864 (-0.1591) |
| hit@5 (positive, unweighted) | 0.7500 | 0.7500 (+0.0000) | 0.8636 (+0.1136) | 0.5455 (-0.2045) |
| MRR@10 (positive, unweighted) | 0.6394 | 0.6394 (+0.0000) | 0.6642 (+0.0248) | 0.4740 (-0.1655) |
| nDCG@10 (positive, unweighted) | 0.6929 | 0.6929 (+0.0000) | 0.7191 (+0.0262) | 0.5231 (-0.1699) |

## Auxiliary (section / chunk hits — defined-only mean)

| metric | baseline | cand_top10_candk30 | cand_top10_mmr_lambda07 | cand_title_section_top10 |
|---|---:|---:|---:|---:|
| section_hit@5 | 0.0455 | 0.0455 | 0.0227 | 0.0455 |
| section_hit@10 | 0.0909 | 0.0909 | 0.0455 | 0.0909 |
| chunk_hit@10 | — | — | — | — |

## Bucket breakdown — gold (weighted_hit@5)

| bucket | n_pos | baseline | cand_top10_candk30 | cand_top10_mmr_lambda07 | cand_title_section_top10 |
|---|---:|---:|---:|---:|---:|
| main_work | 11 | 0.6512 | 0.6512 (+0.0000) | 0.6977 (+0.0465) | 0.6512 (+0.0000) |
| not_in_corpus | 0 | 0.0000 | 0.0000 (+0.0000) | 0.0000 (+0.0000) | 0.0000 (+0.0000) |
| subpage_generic | 16 | 0.9301 | 0.9301 (+0.0000) | 0.9301 (+0.0000) | 0.4895 (-0.4406) |
| subpage_named | 17 | 0.7107 | 0.7107 (+0.0000) | 0.9371 (+0.2264) | 0.6918 (-0.0189) |

## Query-type breakdown — gold (weighted_hit@5)

| query_type | n_pos | baseline | cand_top10_candk30 | cand_top10_mmr_lambda07 | cand_title_section_top10 |
|---|---:|---:|---:|---:|---:|
| alias_variant | 5 | 0.6818 | 0.6818 (+0.0000) | 0.6818 (+0.0000) | 0.6818 (+0.0000) |
| ambiguous | 3 | 0.3333 | 0.3333 (+0.0000) | 0.6667 (+0.3333) | 0.0000 (-0.3333) |
| direct_title | 5 | 1.0000 | 1.0000 (+0.0000) | 1.0000 (+0.0000) | 0.6818 (-0.3182) |
| indirect_entity | 10 | 0.7368 | 0.7368 (+0.0000) | 0.8538 (+0.1170) | 0.7018 (-0.0351) |
| paraphrase_semantic | 11 | 0.9286 | 0.9286 (+0.0000) | 1.0000 (+0.0714) | 0.7381 (-0.1905) |
| section_intent | 10 | 0.7000 | 0.7000 (+0.0000) | 0.9000 (+0.2000) | 0.4000 (-0.3000) |
| unanswerable_or_not_in_corpus | 0 | 0.0000 | 0.0000 (+0.0000) | 0.0000 (+0.0000) | 0.0000 (+0.0000) |

## Silver guardrail (sanity, NOT primary objective)

| metric | baseline | cand_top10_candk30 | cand_top10_mmr_lambda07 | cand_title_section_top10 |
|---|---:|---:|---:|---:|
| hit_at_1 | 0.5537 | 0.5537 (+0.0000) | 0.5537 (+0.0000) | 0.4168 (-0.1368) |
| hit_at_3 | 0.7347 | 0.7347 (+0.0000) | 0.7726 (+0.0379) | 0.5684 (-0.1663) |
| hit_at_5 | 0.7811 | 0.7811 (+0.0000) | 0.8337 (+0.0526) | 0.6105 (-0.1705) |
| hit_at_10 | 0.8337 | 0.8337 (+0.0000) | 0.8737 (+0.0400) | 0.7158 (-0.1179) |
| mrr_at_10 | 0.6544 | 0.6544 (+0.0000) | 0.6718 (+0.0175) | 0.5089 (-0.1454) |

### Silver guardrail warnings

- **cand_title_section_top10**:
  - `SILVER_REGRESSION_WARNING` (hit_at_5, bucket=None): baseline=0.7811 → candidate=0.6105 (Δ=-0.1705; threshold 3.0pp). silver hit@5 dropped 17.05pp (>= 3.0pp threshold) — gold primary_score gain may not generalize.
  - `BUCKET_REGRESSION_WARNING` (hit_at_5, bucket=subpage_named): baseline=0.8500 → candidate=0.7800 (Δ=-0.0700; threshold 5.0pp). silver bucket='subpage_named' hit@5 dropped 7.00pp (>= 5.0pp threshold) — the named-subpage retrieval that the gold-50 set was curated to fix is regressing on the broad silver set; do NOT promote without a deeper audit.

## Recommended next action

- **Adopt `cand_top10_mmr_lambda07`** — primary_score 0.794832 (+0.062163 vs baseline).
- The gain comes from gold-50 weighted hit/MRR/nDCG; verify the silver guardrail tables above before promoting to production. The diagnostic only proves we got better at the subpage / named-subpage failures the gold-50 set was curated to expose.

## Reminders

- AMBIGUOUS_PROBE / ABSTAIN_TEST rows are excluded from the primary objective and only reported separately.
- silver-500 is LLM-generated. Treat its metrics as a regression sanity check, NOT a primary target.
- production retrieval code MUST NOT be changed off this report alone — promote via the standard config-change PR review.

