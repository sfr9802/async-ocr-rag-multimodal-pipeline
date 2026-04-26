# anime_namu_v3 — namu-wiki anime corpus (1,764 titles)

A retrieval **corpus** (not an eval query set). Used as the indexed
document store the retrieval-eval harness scores against. Eval queries
that target this corpus live under
[`ai-worker/eval/eval_queries/`](../../eval_queries/).

## Source

| Field          | Value                                                                |
|----------------|----------------------------------------------------------------------|
| Source path    | `D:/port/rag/app/scripts/namu_anime_v3.jsonl`                         |
| Size           | ~261 MB (gitignored — too large to commit)                            |
| Document count | 1,764 anime titles                                                    |
| Origin         | namu-wiki crawl (port/rag pipeline)                                   |
| Language       | ko                                                                    |
| License        | Same as namu-wiki source (CC-BY-NC-SA 2.0 KR; see project LICENSE)    |

## Re-stage on a fresh clone

The corpus file is intentionally NOT committed. To stage it locally:

```bash
cp 'D:/port/rag/app/scripts/namu_anime_v3.jsonl' \
   ai-worker/eval/corpora/anime_namu_v3/corpus.jsonl
```

If the source is no longer available at the path above, regenerate it
from the namu-wiki dump using `D:/port/rag/app/scripts/build_with_subpages.fixed.py`,
or use the smaller committed sample at
[`ai-worker/fixtures/anime_corpus_kr.jsonl`](../../../fixtures/anime_corpus_kr.jsonl)
(300 titles, sufficient for harness smoke tests but underpowered as a
retrieval baseline).

## Schema (per row)

```jsonc
{
  "doc_id":          "string — stable id (slug + hash)",
  "seed":            "string — original namu-wiki seed page",
  "title":           "string — display title",
  "summary":         "string — 1-2 sentence Claude-generated summary",
  "summary_bullets": ["string", ...],   // 3-5 bullet summary
  "sum_bullets":     ["string", ...],   // duplicate of summary_bullets (legacy)
  "sections": {
    "요약": {"text": "...", "chunks": ["...", ...]},
    "본문": {"text": "...", "chunks": ["...", ...]},
    "등장인물": {"text": "...", "chunks": ["...", ...]}   // ~28% of docs
    // optionally: "설정", "줄거리", "평가", ...
  },
  "section_order":   ["요약", "본문", ...],
  "meta":            {"seed_title": "...", "depth": 0, "fetched_at": "ISO-8601"},
  "subpages":        []   // empty in v3
}
```

### Field availability across the 1,764 docs

| Field                | Coverage  | Notes                                          |
|----------------------|-----------|------------------------------------------------|
| `summary`            | 100%      | Always populated, ~1-2 sentences               |
| `summary_bullets`    | 100%      | 3-5 bullets per doc                            |
| `sections.요약`       | 100%      | Always present                                 |
| `sections.본문`       | 99.9%     | 1763/1764                                      |
| `sections.등장인물`    | **27.8%** | 491 docs — the only character signal we have   |
| `sections.설정`       | 1.6%      | 28 docs                                        |
| `sections.줄거리`     | 1.0%      | 18 docs                                        |
| `aliases`            | 0%        | Field exists, always None                      |
| `keywords`           | 0%        | Field exists, always None                      |

These distributions drive the synthetic-query stratification in
[`ai-worker/eval/harness/generate_eval_queries.py`](../../harness/generate_eval_queries.py)
— we cannot honestly target 20% character queries on a corpus where
72% of docs have no character section, so the deterministic generator
weights query types by what the corpus actually supports.

## Indexing

The retrieval-eval CLI builds an in-memory FAISS index from this file
on the fly via the existing offline-corpus path:

```bash
cd ai-worker
python -m eval.run_eval retrieval \
  --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
  --dataset eval/eval_queries/anime_smoke_6.jsonl \
  --top-k 5
```

No Postgres `ragmeta` schema and no on-disk FAISS index are required —
the corpus is re-chunked + re-embedded into a tempdir per run. This
matches the `--offline-corpus` flag the existing `rag` subcommand
already supports.
