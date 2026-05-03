# Golden Retrieval Eval

Small SearchUnit-focused retrieval eval. It calls the retriever only and does
not run RAG answer generation.

Run from `ai-worker/`:

```bash
python -m ai_worker.evals.golden_retrieval.run \
  --queries evals/golden_retrieval/golden_queries.jsonl \
  --manifest evals/golden_retrieval/source_manifest.json \
  --top-k 5 \
  --out evals/golden_retrieval/eval_report.json
```

Primary matching uses `sourceFileName + unitType + unitKey`; fallbacks include
`sourceFileId + unitType + unitKey`, `searchUnitId`, and page range.
