# Legacy v3 Cleanup Candidates

This report lists possible cleanup targets after Phase 7 v4 guardrails are in
place. It is intentionally non-destructive: no v3 artifact, legacy report, or
migration source was deleted.

## Likely Deletion Candidates After Verification

- Ignored v3 raw/preprocessed/token-chunked corpora such as
  `eval/corpora/anime_namu_v3*` copies, if an external archive exists and no
  active test/script uses them.
- Legacy FAISS/cache directories matching
  `eval/agent_loop_ab/_indexes/bge-m3-anime-namu-v3-*`, if archived reports no
  longer need local reruns.
- Generated `_archive/confirm-runs/` outputs that duplicate committed summary
  reports, after checking manifest digests and preserving the final Markdown /
  JSON evidence.

## Preserve

- `eval/reports/phase0/`, `eval/reports/phase1/`, `eval/reports/phase2/`, and
  `eval/reports/legacy-baseline-final/` until the migration story is finalized.
- v4 conversion, validation, split-manifest, and crawl-audit reports that cite
  v3 input records as provenance.
- Tests that assert production embedding text builder behavior or v4 adapter
  safety, even if they mention legacy variants for regression coverage.

## Still Ambiguous

- `eval/run_eval.py` contains many Phase 0-2 subcommands and v3 examples. Some
  are historical CLI docs, but the file is also the shared eval CLI, so deletion
  or moving is risky without a separate command-level ownership review.
- `eval/harness/embedding_text_reindex.py` and
  `eval/harness/generate_eval_queries.py` can reproduce legacy v3 artifacts,
  while newer v4 harness code still imports or references helper behavior from
  nearby modules.
- Standalone confirmation scripts under `scripts/confirm_*` may still be useful
  for report replay, but should not remain in active docs.

## Required Verification Before Deletion

- Run `rg` for every candidate path and confirm no active Phase 7 config,
  script, or test imports it.
- Run the focused guardrail, answerability adapter/audit, and production
  embedding text tests.
- Confirm final report artifacts have committed summaries and enough manifest
  metadata to make local cache deletion reversible from external storage.
- Keep v3-to-v4 migration provenance until an explicit archive location and
  checksum list exists.

## Expected Risks

- Deleting a local cache may make a historical report harder to reproduce.
- Moving old scripts can break imports in tests or report builders even when
  those scripts are not part of active Phase 7 work.
- Removing provenance too early can make v4 validation claims harder to audit.
