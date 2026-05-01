# Phase 7.6 — section-aware rerank candidate grid

> Scaffolding artefact. Phase 7.6 lands an eval harness; production promotion is a separate Phase 7.6.x PR if a winner survives the guardrails. The Phase 7.5 production-recommended config is the **new baseline** for this sweep.

## Grid

| name | strategy | deployable | base candidate_k | base λ | base top_k | section_bonus | inner_top_k | description |
|---|---|:---:|---:|---:|---:|---:|---:|---|
| `baseline_phase7_5_recommended` | `baseline_no_section_rerank` | ✓ | 40 | 0.70 | 10 | 0.00 | 5 | Phase 7.5 production-recommended: candidate_k=40, MMR λ=0.70. No section-aware rerank. New baseline for Phase 7.6. |
| `section_bonus_005` | `section_bonus` | diagnostic | 40 | 0.70 | 10 | 0.05 | 5 | Add a +0.05 bonus to candidates whose section_path matches the gold expected_section_path (prefix or substring). DIAGNOSTIC ONLY — needs oracle access, not production-deployable. |
| `section_bonus_010` | `section_bonus` | diagnostic | 40 | 0.70 | 10 | 0.10 | 5 | Add a +0.10 bonus to candidates whose section_path matches the gold expected_section_path (prefix or substring). DIAGNOSTIC ONLY — needs oracle access, not production-deployable. |
| `section_bonus_015` | `section_bonus` | diagnostic | 40 | 0.70 | 10 | 0.15 | 5 | Add a +0.15 bonus to candidates whose section_path matches the gold expected_section_path (prefix or substring). DIAGNOSTIC ONLY — needs oracle access, not production-deployable. |
| `supporting_chunk_proximity` | `supporting_chunk_proximity` | diagnostic | 40 | 0.70 | 10 | 0.00 | 5 | Boost candidates that share the gold row's human_supporting_chunk_id page AND a small section-path edit distance. DIAGNOSTIC ONLY — needs oracle supporting-chunk annotation. |
| `page_first_section_rerank_overlap` | `page_first_section_rerank` | ✓ | 40 | 0.70 | 10 | 0.00 | 5 | Two-pass rerank: page-level diversification then section-name token overlap with the query inside each retained page. PRODUCTION-DEPLOYABLE. |
| `same_page_chunk_rerank` | `same_page_chunk_rerank` | ✓ | 40 | 0.70 | 10 | 0.00 | 5 | Within each page in the Phase 7.5 top-k, swap the represented chunk for the page's best section-name overlap chunk. PRODUCTION-DEPLOYABLE. |

## Guardrails (vs Phase 7.5 production-recommended)

- gold primary_score may not drop more than 2pp
- silver hit@5 may not drop more than 3pp
- subpage_named weighted_hit@5 may not drop more than 5pp
- page-level `weighted_hit@5` may not drop below the original Phase 7.4 baseline.

## Notes

- Strategies marked DIAGNOSTIC ONLY require oracle access (expected_section_path or supporting_chunk_id) at inference time. They cannot be promoted to production; their job is to set an upper bound on the section_hit@5 metric.
- Production-deployable strategies use only the query and the candidate metadata.

