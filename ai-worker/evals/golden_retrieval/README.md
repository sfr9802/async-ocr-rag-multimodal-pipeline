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
`sourceFileName + pageStart/pageEnd`, `sourceFileId + unitType + unitKey`,
and `searchUnitId` when a fixture mapped id is available. Metrics include
`hit@1/3/5`, `mrr`, `ndcg@5`, `source_file_accuracy@5`,
`page_accuracy@5`, and `citation_match@5`.

## KoViDoRe economic adapter

The KoViDoRe adapter builds local/dev fixtures only. It does not run OCR or
operating ingestion; `corpus.markdown` and `elements` are treated as an
already-extracted `OCR_RESULT_JSON` artifact.

```bash
python -m ai_worker.evals.golden_retrieval.adapters.kovidore \
  --dataset-path ../datasets/golden/kovidore-economic/raw \
  --limit-docs 20 \
  --limit-queries 163 \
  --out-dir evals/golden_retrieval/fixtures/kovidore-economic
```

Explicit local files are also supported via `--corpus`, `--queries`, `--qrels`,
and optional `--document-metadata`. If no local input is provided, the adapter
tries the Hugging Face `datasets` loader for
`whybe-choi/kovidore-v2-economic-beir`.

It writes:

- `manifest.json`: dataset id, source, license note, counts, and matching identity.
- `source_files.jsonl`: one READY pdf source fixture per `doc_id`.
- `extracted_artifacts.jsonl`: one `OCR_RESULT_JSON` fixture per `doc_id`.
- `search_units.jsonl`: PAGE SearchUnits keyed as `page:{page_number_in_doc}`.
- `source_manifest.json`: stable matching helper for the eval runner.
- `golden_queries.jsonl`: qrels joined to PAGE SearchUnit identities.

Qrels are graded: score `2` becomes `expected`, score `1` becomes
`acceptable`, and score `1` is promoted to `expected` when a query has no
score `2` page in the fixture.

Import the generated fixture into a local Spring dev DB with the direct seed
command below. The importer is idempotent and does not call the OCR callback
path.

```bash
python -m ai_worker.evals.golden_retrieval.import_kovidore \
  --fixture-dir evals/golden_retrieval/fixtures/kovidore-economic \
  --core-api-url http://localhost:8080 \
  --batch-size 100
```

Then index and run retrieval eval:

```bash
python -m ai_worker.search_unit_indexing --once --batch-size 100

python -m ai_worker.evals.golden_retrieval.run \
  --queries evals/golden_retrieval/fixtures/kovidore-economic/golden_queries.jsonl \
  --manifest evals/golden_retrieval/fixtures/kovidore-economic/source_manifest.json \
  --top-k 5 \
  --out evals/golden_retrieval/fixtures/kovidore-economic/eval_report.json
```
