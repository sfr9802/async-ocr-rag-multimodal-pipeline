# Korean Enterprise Corpus — License and Provenance

This notice covers every file under
``ai-worker/fixtures/enterprise_corpus_kr/`` plus the derived eval
dataset at ``ai-worker/eval/datasets/rag_enterprise_kr.jsonl``.

## License

MIT License (see ``LICENSE`` at the project root).

## Nature of the content

The corpus is **fully synthetic**. Every document, clause, figure,
named role, internal policy reference, and product name was generated
by a large language model against the prompt templates committed under
``ai-worker/scripts/dataset/prompts/enterprise/``. No content was
scraped, copied, or derived from any real company's private documents.

Any resemblance between the figures, policies, or product names in
this corpus and those of a real Korean company is **coincidental**.
The corpus is intended exclusively as an evaluation fixture for a
Korean-language RAG system; it must not be used to make claims about,
or decisions concerning, any real-world person, company, statute, or
court decision.

## Generation pipeline

- **Corpus synthesis.** Claude Sonnet 4.6 (``claude-sonnet-4-6``)
  produced each document via
  ``python -m scripts.dataset.build_enterprise_corpus``. Each row in
  ``enterprise_corpus_kr/index.jsonl`` carries the concrete model id in
  ``source`` plus the per-doc ``seed`` and ``generated_ts`` timestamp
  so the generation is reproducible.
- **Query synthesis.** Claude Haiku (``claude-haiku-4-5-20251001``)
  produced the evaluation queries via
  ``python -m scripts.dataset.generate_enterprise_queries`` from the
  corpus above. ``rag_enterprise_kr_raw.jsonl`` is the full raw
  output; ``rag_enterprise_kr.jsonl`` is the stratified 200-row subset
  produced by ``validate_enterprise_dataset``.
- **Diversity guard.** After generation, each category's documents
  are embedded with ``BAAI/bge-m3`` and pairwise cosine is computed.
  Pairs above 0.88 are flagged and the younger doc is regenerated
  (up to 3 passes). Residuals, if any, are logged in
  ``enterprise_corpus_kr/corpus_summary.json``.
- **Audit trail.** Every Claude call — success, validation rejection,
  or API error — is appended to
  ``enterprise_corpus_kr/generation_log.jsonl`` with model id, token
  counts, latency, and the original seed. The log is the cost ledger.

## Reproducibility

All prompts live under version control at
``ai-worker/scripts/dataset/prompts/enterprise/*.md``. Rebuilding the
corpus from scratch uses the default CLI invocation:

```
python -m scripts.dataset.build_enterprise_corpus \
    --out ai-worker/fixtures/enterprise_corpus_kr/ \
    --categories hr,finance,it,product,legal \
    --per-category 25 --seed 42 \
    --generator claude:sonnet-4-6
```

Given the same committed prompts, seed, and model version, re-running
produces comparable but not byte-identical content (Claude is sampled,
not deterministic). The ``seed`` and ``generated_ts`` fields in
``index.jsonl`` preserve the exact generation context.

## Budget

Expected Claude Sonnet 4.6 cost for a full 125-doc generation is
approximately **$1.80** (≈ 150 docs × 800 output tokens at $15 / 1M
output tokens). The actual cost is logged in
``enterprise_corpus_kr/corpus_summary.json`` under
``estimated_cost_usd`` once a real run completes.
