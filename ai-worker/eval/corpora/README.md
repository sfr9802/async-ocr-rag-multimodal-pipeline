# Retrieval corpora

Each subdirectory is one **retrieval corpus** — a body of documents the
retrieval-eval harness indexes and scores eval queries against. Corpora
are kept separate from eval queries (see
[`../eval_queries/`](../eval_queries/)) so the same corpus can be tested
by multiple query sets and the same query set can later be re-pointed
at a different corpus.

## Layout

```
corpora/
├── anime_namu_v3/
│   ├── README.md               source path, schema, re-stage instructions
│   └── corpus.jsonl            (gitignored — 261 MB; staged on demand)
└── README.md                   this file
```

## Conventions

| Rule | Why |
|---|---|
| `corpus.jsonl` is the canonical filename inside each corpus dir | Tooling can find it without a manifest lookup |
| The `.jsonl` itself is gitignored; `README.md` is committed | Corpora are large (>100 MB common) and have non-redistributable upstream sources |
| Every corpus dir has a README explaining how to re-stage from source | A fresh clone can rebuild the eval setup deterministically |
| Schema must be the production-ingest shape (`doc_id`, `sections: {<name>: {text, chunks}}`) | Reuses [`app.capabilities.rag.ingest._chunks_from_section`](../../../app/capabilities/rag/ingest.py) and [`eval.harness.offline_corpus.build_offline_rag_stack`](../harness/offline_corpus.py) without conversion |
