# Eval queries — retrieval-quality datasets

JSONL files in this directory are **eval query sets** used by the
`retrieval` mode of the eval CLI. They are scored against a corpus
(see [`../corpora/`](../corpora/)) — the corpus is the haystack, the
queries are the needles + the gold answer keys.

> ⚠️ **Strict corpus / queries split**: do not put corpus documents
> here. Corpus rows live under `eval/corpora/<name>/corpus.jsonl`. A
> query row's `expected_doc_ids` cite ids in some corpus.

## Three quality tiers

| File                           | Source              | Rows | Quality | Trust as ground truth? |
|--------------------------------|---------------------|------|---------|------------------------|
| `anime_smoke_6.jsonl`          | hand-authored       | 6    | gold-style smoke | yes — used for harness sanity-check only |
| `anime_silver_200.jsonl`       | deterministic gen   | 200  | **silver** | **NO** — synthetic, best-effort phrasing |
| `anime_silver_200_llm.jsonl`   | LLM-backed gen      | 200  | silver   | NO — same caveat as above + non-determinism |
| `anime_gold_20.jsonl`          | hand-authored       | 20   | gold     | yes — manually curated against corpus content |

### Silver vs gold semantics

* **silver** — programmatically generated from corpus fields (titles,
  summaries, character section excerpts). Good for tracking *trends* in
  retrieval quality across config changes, but the queries themselves
  may be awkwardly phrased (deterministic) or model-flavored (LLM), and
  the expected keywords are heuristically extracted, not curated.
  **Never gate a release on silver numbers.**

* **gold** — every row is hand-written by a human who has read the
  source doc. Queries phrase a real question; `expected_doc_ids`
  point at the doc that actually contains the answer; keywords are the
  terms a correct answer would mention. This is the set you trust as
  the bar for "did the retriever find the right thing".

## Row schema

```jsonc
{
  "id":                        "anime-{tier}-{seq}",
  "query":                     "Korean question",
  "language":                  "ko",
  "expected_doc_ids":          ["<doc_id from the target corpus>"],
  "expected_section_keywords": ["substring", ...],
  "answer_type":               "summary_plot" | "title_lookup" |
                               "character_relation" | "body_excerpt" |
                               "theme_genre" | "setting_worldbuilding",
  "difficulty":                "easy" | "medium" | "hard",
  "tags":                      ["anime", "<tier>", ...]
}
```

* `expected_doc_ids` — list because a question can have more than one
  correct doc; the retrieval metrics treat any of them in top-k as a hit.
* `expected_section_keywords` — used by `expected_keyword_match_rate`
  (fraction present in any retrieved chunk text). Not the same as the
  legacy `expected_keywords` field used by the older `rag` harness
  (which scored against the generated answer string).
* `answer_type` — drives per-type metric breakdowns in the report.
* `tags` — `gold` / `silver` / `synthetic` / `manual` / `deterministic` /
  `llm:<backend>` are all conventional. The harness does not parse
  tags; they are for human filtering and downstream tooling.

## Generating silver

Run from `ai-worker/`:

```bash
# Deterministic — reproducible, zero LLM cost, ~50ms per 1764-doc corpus
python -m eval.harness.generate_eval_queries \
    --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
    --out    eval/eval_queries/anime_silver_200.jsonl \
    --target 200 \
    --generator deterministic \
    --seed 42

# LLM-backed — wraps scripts.dataset.generate_anime_queries
# (requires ANTHROPIC_API_KEY or a running Ollama daemon)
python -m eval.harness.generate_eval_queries \
    --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
    --out    eval/eval_queries/anime_silver_200_llm.jsonl \
    --target 200 \
    --generator llm \
    --llm-backend claude:haiku
```

The deterministic backend stratification is **calibrated to actual
field coverage** in the namu_anime_v3 corpus:

| answer_type            | source field            | corpus coverage | target % |
|------------------------|-------------------------|-----------------|----------|
| `summary_plot`         | `summary`               | 100%            | 40%      |
| `character_relation`   | `sections.등장인물`       | 27.8%           | 20%      |
| `title_lookup`         | `title`                 | 100%            | 15%      |
| `body_excerpt`         | `sections.본문`          | 99.9%           | 10%      |
| `theme_genre`          | `summary_bullets`       | 100%            | 10%      |
| `setting_worldbuilding`| `sections.설정`/`세계관` | 1.6%            | 5%       |

Coverage-poor types are capped at the docs that actually carry the
source signal — the generator never fabricates section content.

## Extending the gold set

`anime_gold_20.jsonl` is intentionally small (20 rows) because hand
curation takes real time. To extend toward gold-50 or gold-100:

1. Pick a doc in the corpus you have personal context for. Verify with:
   ```bash
   python - <<'PY'
   import json
   for line in open('eval/corpora/anime_namu_v3/corpus.jsonl', encoding='utf-8'):
       d = json.loads(line)
       if 'YOUR_TITLE_QUERY' in (d.get('title') or ''):
           print(d['doc_id'], d['title'])
           print((d.get('summary') or '')[:300])
   PY
   ```
2. Phrase a question whose answer is **in the doc's `summary` or
   `summary_bullets`**. Avoid trivia that requires external knowledge.
3. Set `expected_doc_ids` to the verified `doc_id`. Set
   `expected_section_keywords` to 2-4 substrings the chunk text would
   reasonably contain (these become the `expected_keyword_match_rate`
   denominator).
4. Pick `answer_type` from the same six values as silver. Keep the
   tier balance roughly: 30% easy / 50% medium / 20% hard.
5. Tag with `["anime", "gold", "manual", "<answer_type>"]`.

Run the smoke harness on every batch you add to catch typos in
expected_doc_ids:

```bash
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3/corpus.jsonl \
    --dataset eval/eval_queries/anime_gold_20.jsonl \
    --top-k 10
```

Any row reporting `hit@10 == 0` is either a typo (wrong doc_id) or a
genuinely hard query — read the top-k dump for that row to decide.

## What this directory does NOT cover

* **Generator-quality eval.** This package only scores retrieval. The
  legacy `rag` mode (separate, untouched) is what scores the generated
  answer against `expected_keywords` / refusal phrases. If you need
  end-to-end answer quality, use that mode against
  `eval/datasets/rag_*.jsonl`.
* **Cross-modal queries.** Image+text fusion lives under MULTIMODAL.
* **OCR-driven retrieval.** Use the OCR harness, then feed extracted
  text into the retrieval harness as the query.
