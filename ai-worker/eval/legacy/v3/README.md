# Legacy v3 Evaluation Area

This area documents legacy v3-only evaluation material that is preserved for
historical reproduction and migration provenance. It is not the active Phase 7
workflow.

## Current Rule

- Canonical Phase 7 data is dataset v4 under
  `eval/corpora/namu-v4-structured-combined/`.
- Active eval/tuning must not default to `anime_namu_v3`,
  `rag-cheap-sweep-v3`, or `bge-m3-anime-namu-v3-*` caches.
- Phase 7.7 answerability joins production retrieval emits through
  `rag_chunks.jsonl`; `chunks_v4.jsonl` is a different chunk-id namespace and
  must not be used as that join source.

## Preserved Legacy Script Entry Points

The following scripts remain in their original locations to avoid import and
report-reproduction churn, but each is marked `LEGACY V3 ONLY`:

- `eval/tune_eval_offline.py`
- `scripts/eval_full_silver_minimal_sweep.py`
- `scripts/eval_wide_mmr_titlecap_sweep.py`
- `scripts/confirm_wide_mmr_best_configs.py`
- `scripts/confirm_embedding_text_variant.py`
- `scripts/confirm_reranker_input_format.py`
- `scripts/confirm_rerank_input_cap_policy.py`
- `scripts/build_legacy_baseline_final.py`
- `scripts/eval_agent_loop_ab_baseline.py`

Use these only to reproduce archived Phase 0-2 / v3 reports. Do not use them
to choose Phase 7 v4 retrieval defaults.

## Historical Reports

Reports under `eval/reports/phase0/`, `eval/reports/phase1/`,
`eval/reports/phase2/`, and `eval/reports/legacy-baseline-final/` are preserved
as historical evidence. They may mention v3 paths by design.

## Migration Provenance

v3 mentions inside v4 conversion, validation, or crawl-audit material can be
valid provenance. Do not remove those references unless the migration record
has a replacement source of truth.
