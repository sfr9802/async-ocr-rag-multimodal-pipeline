# Phase 7.7 — Answerability Audit (operator notes)

This directory holds the **inputs** and (eventually) the **labelled outputs** of the
human answerability audit on the Phase 7.5 production_recommended retrieval config.
It is not a production report directory — there are no scores, no winners, no
promotion decisions here. The score CLI (`scripts.score_answerability_audit`) is
NOT run until a reviewer has filled in label cells.

> **Query set framing.** The current export joins against the
> **LLM-silver** query set
> (`v4-llm-silver-XXX`, from
> [`seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl`](../seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl)).
> Treat the audit's outputs as "answerability over an LLM-silver
> sampled query set" — **not** "answerability over the human gold
> set". The Phase 7.5 human gold-50 (`v4-silver-500-XXXX` namespace)
> would need its own retrieval emit + audit to make a human-gold
> claim.

## What lives here

| File pattern | Meaning | Tracked by git? |
|---|---|---|
| `*_unlabeled_export.csv` / `*_unlabeled_export.jsonl` | Raw export from `scripts.export_answerability_audit --input-format v4-production`. Regenerable from inputs. Label cells are empty by contract. | **No** — gitignored (`.gitignore` patterns at repo root). |
| `*_sample{N}q_unlabeled_export.csv` | Pilot sample — N distinct query_ids deterministically pulled from the parent unlabeled export by `--mode row-sample`. Label cells stay empty. | **No** — gitignored (matches the `*_unlabeled_export` pattern). |
| `*_human_labeled.csv` / `*_human_labeled.jsonl` | Reviewer-filled labels. The score CLI consumes these. | **Yes** — commit when a labelling pass is finalised. |
| `*_score_report.md` / `*_score_report.json` | Output of `scripts.score_answerability_audit` over a labelled file. | Yes (after labelling). |
| `README.md` (this file) | Operator / reviewer notes. | Yes. |
| `ANSWERABILITY_LABELING_GUIDE.md` | Reviewer-facing labeling guide for the row-level pilot (label scale, flag definitions, OOC policy, common pitfalls). | Yes. |

## v4 input contract

`scripts.export_answerability_audit --input-format v4-production` is the only
supported entry point for production v4 retrieval emits. It is wired in
[`eval/harness/answerability_v4_adapter.py`](../../../harness/answerability_v4_adapter.py).
Three v4 canonical artifacts must be supplied:

1. **`--retrieval-results-path`** — production v4 retrieval emit JSONL. Each
   record has `{variant, query_id, query, elapsed_ms, docs:[{rank, chunk_id,
   page_id, title, section_path:list, score}]}`. The Phase 7.5 promoted
   config (top_k=10, candidate_k=40, MMR λ=0.70) lives at
   [`seeds/human_gold_seed_50_tuning/confirm_sweep/retrieval_cand_candk40_mmr_lambda070_gold.jsonl`](../seeds/human_gold_seed_50_tuning/confirm_sweep/retrieval_cand_candk40_mmr_lambda070_gold.jsonl).

2. **`--chunks-path`** — chunk fixture for raw text resolution. **Use
   `rag_chunks.jsonl`**, not `chunks_v4.jsonl`. The two files use disjoint
   `chunk_id` namespaces (135,602 vs 48,675 chunk_ids, intersection = 0):
   `rag_chunks.jsonl` is what the production retrieval index ingests, so its
   `chunk_id`s match a retrieval emit; `chunks_v4.jsonl` is the structured
   canonical source whose chunk_ids do NOT line up with the retrieval index.
   Path:
   [`eval/corpora/namu-v4-structured-combined/rag_chunks.jsonl`](../../../corpora/namu-v4-structured-combined/rag_chunks.jsonl).

3. **`--gold-path`** — gold / silver query metadata. The retrieval emit's
   `query_id` namespace decides which file to use:
   - `v4-llm-silver-XXX` ⇒
     [`seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl`](../seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl).
     This is the file the Phase 7.5 retrieval emit joins against.
   - `v4-silver-500-XXXX` ⇒ `silver500/queries/queries_v4_silver_500.jsonl`
     or [`seeds/human_seed_50/phase7_human_gold_seed_50.csv`](../seeds/human_seed_50/phase7_human_gold_seed_50.csv).
     **Note:** the gold-50 CSV's `query_id`s do NOT overlap with the
     `v4-llm-silver-XXX` retrieval emits — its `human_correct_*` columns
     (currently 0/50 filled) cannot be reflected in those exports. It would
     only kick in if a separate retrieval emit is generated against the
     gold-50 CSV's silver-500 ID space.

## chunk_text rule (read this before changing the resolver)

Answerability is graded by a human reading the retrieved chunk's text. The
adapter therefore enforces three invariants:

- **Source must match the retrieval index.** `chunk_text` is resolved from the
  fixture supplied via `--chunks-path` by `chunk_id` lookup. The retrieval emit
  itself never carries `chunk_text` and must not be relied on for it.
- **Raw text only, never embedding text.** The resolver reads `chunk_text`
  (rag_chunks form) and `text` (chunks_v4 form). It explicitly does NOT read
  `embedding_text` or `text_for_embedding` — those fields prepend the title /
  section header / section type and would make a reviewer think the retriever
  returned context that it did not.
- **`--on-missing-chunk` policy.** The default `error` mode raises immediately
  on a chunk_id that is absent from the fixture — the right behaviour for a
  production labelling export. The `collect` mode is for triage runs only:
  it fills `chunk_text` with empty string and surfaces the missing tuples
  in the adapter's log; the audit's import validator will then reject those
  rows at scoring time anyway. Do NOT use `collect` for labelling exports.

## Forbidden in this directory

- Synthetic / illustrative labels rendered as if they were a production audit
  result (the previous `sample_report.*` artifacts were deleted for exactly
  this reason).
- LLM-judge labels masquerading as human labels.
- Score reports built on a partially-labelled file.
- `*_unlabeled_export.*` files committed to the repo (gitignored).
- Framing the audit's outputs as "human gold answerability". The current
  exports are joined against the LLM-silver query set; treat all numbers as
  *LLM-silver query-set answerability* and call them that in any downstream
  report.

## Workflow

1. Generate or locate a v4 retrieval emit (Phase 7.5 production_recommended is
   the canonical input today).
2. Run `python -m scripts.export_answerability_audit --mode row --input-format
   v4-production --retrieval-results-path … --chunks-path … --gold-path …
   --variant-name … --top-k 5 --on-missing-chunk error --out-path
   <variant>_<query_set>_top<k>_unlabeled_export.csv`. Verify the smoke
   checklist in `tests/test_answerability_v4_adapter.py` (row count, missing
   chunk = 0, no embedding-prelude leak, page/section_hit correctness).
3. **Pilot subset.** Run
   `python -m scripts.export_answerability_audit --mode row-sample
   --input-path <variant>_<query_set>_top<k>_unlabeled_export.csv
   --variant-name production_recommended
   --sample-query-count 10 --seed 42
   --out-path <variant>_<query_set>_top<k>_sample10q_unlabeled_export.csv`.
   This pulls 10 distinct query_ids × every rank into a smaller pilot CSV
   (50 rows for top-5). The sampler never fills label cells.
4. Make a copy under the `*_human_labeled.csv` name and have a reviewer fill
   in `label_answerability` / `flags` / `notes`. The unlabeled file remains
   gitignored so the reviewer has a clean baseline to start from. Reviewer
   instructions live in
   [`ANSWERABILITY_LABELING_GUIDE.md`](ANSWERABILITY_LABELING_GUIDE.md).
5. Run `python -m scripts.score_answerability_audit --mode row --labeled-path
   <variant>_<query_set>_top<k>_human_labeled.csv --report-path
   <variant>_<query_set>_top<k>_score_report.md` to render the audit report.

Until a reviewer has actually labelled a file, no score report is produced.
