# Phase 7.7 Answerability Audit Labeling Guide

This is the reviewer-facing instructions for the row-level answerability
labeling pilot on the Phase 7.5 `production_recommended` retrieval config
(top_k=10, candidate_k=40, MMR λ=0.70, variant
`cand_candk40_mmr_lambda070`).

## Scope and framing — read this first

- This labeling pass operates on an **LLM-silver sampled query set**
  (`v4-llm-silver-XXX`, drawn from
  [`seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl`](../seeds/llm_silver_500/queries_v4_llm_silver_500.jsonl)).
  It is **not** a human-gold labeling pass. The question being answered
  is "given these LLM-authored silver queries and the retrieved chunks
  the production config returned, how often is the evidence
  answerable?" — not "what is ground-truth answerability over the
  human gold set."
- This is a **row-level audit**: each row is one retrieved chunk for
  one query at one rank. The reviewer judges that single chunk on its
  own ("does *this* chunk carry evidence to answer the query?"). The
  bundle-level (top-k context as a whole) audit is a separate
  follow-up step (Phase 7.7.1) and is **out of scope here**.
- The answerability metric is a **complementary signal**, not a
  replacement for hit@k or any retrieval-side metric. Promotion
  decisions stay gated on the existing Phase 7.5 / 7.6 retrieval
  guardrails.
- All judgments must come from a human reading the chunk. No LLM
  judge, no auto-fill, no generation eval. The exported CSV ships
  with `label_answerability` / `flags` / `notes` cells **empty by
  contract**; the score CLI refuses to run on a partially-labeled
  file.

## Pilot input

| File | Purpose |
|---|---|
| [`production_recommended_v4_gold50_top5_sample10q_unlabeled_export.csv`](production_recommended_v4_gold50_top5_sample10q_unlabeled_export.csv) | Pilot input. 10 distinct query_ids × top-5 = 50 rows, sampled deterministically (`--seed 42`) from the full 50-query export. Gitignored — regenerate with the CLI command in the README. |
| [`production_recommended_v4_gold50_top5_unlabeled_export.csv`](production_recommended_v4_gold50_top5_unlabeled_export.csv) | Source of the pilot (50 query × top-5 = 250 rows). Gitignored. |
| `production_recommended_v4_gold50_top5_sample10q_human_labeled.csv` | **The reviewer fills this in.** Make a copy of the pilot CSV under this name and edit the three label columns. Tracked by git. |

The 10 pilot query_ids (deterministic at seed=42):

`v4-llm-silver-013`, `v4-llm-silver-074`, `v4-llm-silver-075`,
`v4-llm-silver-076`, `v4-llm-silver-109`, `v4-llm-silver-166`,
`v4-llm-silver-169`, `v4-llm-silver-204`, `v4-llm-silver-383`,
`v4-llm-silver-437`.

None of the pilot queries have `expected_not_in_corpus=true` in the
underlying silver set — the three OOC queries in the parent 50-set
(silver-510 / 518 / 520) did not land in this seed=42 sample, so the
pilot has no OOC contamination to worry about. (See
*Out-of-corpus queries* below for the policy when a future pilot does
include OOC queries.)

## Label scale

The `label_answerability` cell takes one of four integers. Read the
chunk text and the query, then pick the highest tier that holds:

| Value | Name | Definition |
|---|---|---|
| 0 | `NOT_RELEVANT` | The chunk is about the wrong work / wrong page / wrong context. There is no plausible reading where this chunk could answer the query. |
| 1 | `RELATED_BUT_NOT_ANSWERABLE` | The chunk is about the same work / character / setting / section family the query is about, but it does not carry direct evidence for the query's specific question. |
| 2 | `PARTIALLY_ANSWERABLE` | The chunk gives some direction or partial evidence toward the answer, but a key piece of information is missing. The query might become fully answerable when this chunk is combined with another. |
| 3 | `FULLY_ANSWERABLE` | Reading this single chunk is enough to answer the core of the query directly. Almost no outside knowledge or strong inference is required. |

If you sit between two tiers, prefer the lower tier and explain in
`notes`. The score CLI applies a single `ANSWERABLE_MIN_LEVEL`
threshold (currently `PARTIALLY_ANSWERABLE`) to compute
`answerable@k`; conservative-when-uncertain keeps that signal honest.

## Flags

`flags` is a comma-separated list (empty allowed). Apply only the
flags that actually fire on the row — flags are diagnostic, not
exhaustive.

| Flag | When to apply |
|---|---|
| `wrong_page` | The retrieved chunk is from a different page (different work / character / topic) than the query is about. Often co-occurs with label 0. |
| `right_page_wrong_section` | The retrieved chunk is from the right page but the wrong section of that page (e.g. query asks about ending music but chunk is about opening music). |
| `evidence_too_noisy` | The right neighbourhood, but the chunk text is too cluttered (template lines, list artifacts, OCR-like noise, irrelevant filler) for the evidence to be usable. |
| `needs_cross_section` | This chunk alone is not enough, but combined with another section of the same page it would be answerable. Use with label 2. |
| `needs_subpage` | The query points to a sub-work / sub-character / sub-topic that lives on its own subpage, and this chunk is from the parent page (or vice versa). |
| `ambiguous_query` | The query itself is ambiguous enough that "answerable" cannot be cleanly judged regardless of the chunk. Pair with a `notes` cell explaining the ambiguity. |

## Out-of-corpus queries

For queries flagged `expected_not_in_corpus=true` in the silver query
set, `gold_page_id` is empty in the export. **This pilot has zero
such queries** (see above), so the policy below is for future pilots:

- The reviewer judges the row strictly on the query text vs.
  `chunk_text`. Do not infer answerability from the (empty) gold
  fields.
- Most chunks for an OOC query will be `NOT_RELEVANT` or
  `RELATED_BUT_NOT_ANSWERABLE`, but **do not auto-assign** that
  label — the retriever may have surfaced something genuinely close
  by accident, and that signal is worth recording.
- If the query itself looks like it cannot be unambiguously answered
  from any v4 corpus chunk (because the topic is genuinely outside
  the corpus), set `ambiguous_query` *and* annotate `notes` with
  "OOC query — silver set marks expected_not_in_corpus".

## Labeling order (per query)

1. Read the query.
2. Skim ranks 1 → 5 in order to get a sense of what the retriever
   thinks the topic is.
3. Go back to rank 1 and label `label_answerability` for **that
   single chunk**, on its own merits, against the query.
4. Add `flags` if any apply.
5. If you hesitated, drop the reasoning into `notes`.
6. Repeat for ranks 2–5.
7. After labeling all 5 ranks of a query, glance back over them: did
   your bar drift? Re-calibrate and adjust if needed.

## Common pitfalls

- **Do not let `page_hit` / `section_hit` colour your judgment.**
  Those columns are computed from `(gold_page_id, gold_section_path)`
  and are diagnostic only. A row with `page_hit=true` can still be
  `NOT_RELEVANT` if the chunk text doesn't actually carry the
  evidence; a row with `page_hit=false` can still be
  `FULLY_ANSWERABLE` if the chunk happens to contain the answer
  through a different page on the same topic.
- **Do not let `retrieved_page_title` substitute for evidence.**
  A correct-looking title doesn't mean the *chunk text* answers the
  question — only the chunk text does.
- **Do not lean on outside knowledge.** Even if you know the work
  well, you must judge only what the chunk text says. The audit is a
  measure of *what the retriever returned*, not of what you happen
  to know.
- **Cross-section queries.** When the query needs evidence from
  multiple sections / chunks, the per-row label should reflect that
  *this single row* is `PARTIALLY_ANSWERABLE` and `flags` should
  carry `needs_cross_section`. Do **not** label every cross-section
  row `FULLY_ANSWERABLE` because the union of rows is — that breaks
  row-level vs. bundle-level diagnosis.

## When you are done

The pilot is "done" when all 50 rows in
`production_recommended_v4_gold50_top5_sample10q_human_labeled.csv`
have a `label_answerability` value (0 / 1 / 2 / 3) and any flags /
notes you want to record. Then the score CLI can be run; until that
point it stays unrun.

## Score CLI (post-labeling)

```
python -m scripts.score_answerability_audit \
  --mode row \
  --labeled-path eval/reports/phase7/7.7_answerability_audit/production_recommended_v4_gold50_top5_sample10q_human_labeled.csv \
  --report-path  eval/reports/phase7/7.7_answerability_audit/production_recommended_v4_gold50_top5_sample10q_score_report.md
```

The score report stays out of git until a labelled file actually
exists.

## Forbidden in this pilot (recap)

- Auto-filling labels, flags, or notes.
- Running the score CLI before labels are filled.
- Substituting an LLM judge for the human reviewer.
- Treating this set as if it were human-gold-derived (it is
  LLM-silver-derived).
- Promoting / blocking a production config based on this pilot
  alone — Phase 7.5 / 7.6 retrieval guardrails are still the
  promotion gate.
