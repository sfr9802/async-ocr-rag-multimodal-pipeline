# async-ocr-rag-multimodal-pipeline

Asynchronous AI processing platform.

## Benchmarks (measured on this branch)

| Dataset            | Capability | hit@5 | MRR  | CER (OCR) | p50 latency |
|--------------------|------------|-------|------|-----------|-------------|
| anime (en, 8 docs) | RAG        | not measured | not measured | n/a       | not measured |
| kr_sample (kr, 10) | RAG        | not measured | not measured | n/a       | not measured |
| ocr_sample (en)    | OCR        | n/a   | n/a  | not measured | not measured |
| ocr_sample (kr)    | OCR        | n/a   | n/a  | not measured | not measured |

> Measured with claude-sonnet-4-6 vision + generation, bge-m3 embedder,
> FAISS IndexFlatIP, Tesseract kor+eng.
> Reproduce: `python -m eval.run_eval rag --dataset eval/datasets/rag_sample_kr.jsonl`
>
> **Note:** Benchmarks are marked "not measured" because this environment
> does not have a running PostgreSQL ragmeta schema, FAISS index, or
> Tesseract binary available to execute the eval harness. To produce real
> numbers: build the index (`python -m scripts.build_rag_index --fixture both`),
> ensure Tesseract is installed with `eng+kor` language packs, then run
> `python -m eval.run_eval rag` and `python -m eval.run_eval ocr`.

- **Phase 1**: async skeleton (job lifecycle, artifact model, claim/callback,
  Redis dispatch, local-filesystem storage, `MOCK` capability).
- **Phase 1.1**: stabilized onto the target stack (Spring Boot 4.0.3, Java 21,
  PostgreSQL 18, Python 3.12, Redis) and verified end-to-end.
- **Phase 2 (this commit)**: adds a real **text-RAG capability** — FAISS
  vector index, sentence-transformers embedding, extractive grounded
  generation, RAG metadata persisted in PostgreSQL (separate `ragmeta`
  schema — no MongoDB). The `MOCK` capability stays alongside it.

Real OCR, file/PDF parsing, and multimodal capabilities land in later phases.

## Stack

| Component      | Version                                |
|----------------|----------------------------------------|
| Java           | 21 (LTS)                               |
| Spring Boot    | 4.0.3                                  |
| Maven          | 3.9+                                   |
| Python         | 3.12 (3.13 also works)                 |
| PostgreSQL     | 18                                     |
| Redis          | latest                                 |

## Layout

```
.
├── core-api/                   Spring Boot — jobs / artifacts / claim / callback
│   └── src/main/resources/db/migration/
│       ├── V1__init.sql           pipeline schema (job, artifact)
│       └── V2__ragmeta_schema.sql ragmeta schema (documents, chunks, index_builds)
├── ai-worker/                  Python worker — Redis BRPOP consumer
│   ├── app/capabilities/
│   │   ├── mock_processor.py   MOCK capability (phase 1)
│   │   └── rag/                RAG capability (phase 2):
│   │       ├── capability.py      entry point
│   │       ├── chunker.py         ported from port/rag (greedy + window_by_chars)
│   │       ├── embeddings.py      EmbeddingProvider + sentence-transformers impl
│   │       ├── generation.py      GenerationProvider + extractive impl
│   │       ├── faiss_index.py     IndexFlatIP wrapper
│   │       ├── metadata_store.py  psycopg2 DAO for ragmeta.*
│   │       ├── ingest.py          JSONL -> chunks -> FAISS + metadata
│   │       └── retriever.py       query -> top-k RetrievedChunk list
│   ├── scripts/build_rag_index.py indexing CLI
│   └── fixtures/anime_sample.jsonl small committed fixture
├── frontend/                   Tiny HTML test client
├── scripts/e2e_smoke.py        End-to-end MOCK smoke test
├── docker-compose.yml          Infra-only (redis default, postgres in "independent" profile, minio in "minio" profile)
├── .env.example                Full env-var reference
└── docs/
    ├── architecture.md
    ├── local-run.md
    └── api-summary.md
```

## One-minute tour

- **core-api** follows DDD + hexagonal: `domain` / `application/port` /
  `application/service` / `adapter/in/web` / `adapter/out/persistence`
  plus separate `queue` and `storage` outbound adapters.
- **ai-worker** is a straight-line execution engine: Redis BRPOP → claim
  → fetch input → run capability → upload output bytes → callback.
- **PostgreSQL owns job state**. Redis is just the dispatch signal, not a
  source of truth.
- **Storage is abstracted behind a port**; phase 1 ships a
  local-filesystem adapter. MinIO and S3 slot in later without touching
  the application layer.
- **Capabilities are pluggable**: `MOCK` lives in
  `ai-worker/app/capabilities/mock_processor.py`. `rag/`, `ocr/`, and
  `multimodal/` are empty packages that will light up in phase 2+.

## Two database execution modes

Phase 1.1 supports two ways of getting a Postgres you can point at.
Pick exactly one.

- **Mode A (default): reuse an existing Postgres container** (e.g. one
  you already run for another project like RIA). You create a dedicated
  `aipipeline` database and user inside it, so there is zero collision
  with whatever else lives in that container. This compose starts only
  redis.
- **Mode B: independent infra compose.** Brings up its own `postgres:18`
  on host port `5433` so it cannot collide with anything on `5432`.
  Requires running core-api with the `independent` Spring profile.

See [`docs/local-run.md`](docs/local-run.md) for the exact bootstrap
commands, the troubleshooting list, and the Windows-specific tips.

## Run it

Short version (Mode A, assuming you already have a Postgres container on
:5432 and you've bootstrapped the `aipipeline` DB + user as documented):

```bash
docker compose up -d redis                                                  # terminal 1
(cd core-api && mvn spring-boot:run)                                        # terminal 2
(cd ai-worker && pip install -r requirements.txt && python -m app.main)     # terminal 3
python scripts/e2e_smoke.py                                                 # terminal 4
```

### Internal endpoint authentication

All `/api/internal/**` endpoints (claim, callback, artifact upload) are
gated by a shared-secret header (`X-Internal-Secret`). Core-api and the
worker use **different env-var names** but the actual secret value **must
be identical**:

| Service   | Env var                             |
|-----------|-------------------------------------|
| core-api  | `AIPIPELINE_INTERNAL_SECRET`        |
| ai-worker | `AIPIPELINE_WORKER_INTERNAL_SECRET` |

When neither is set, core-api runs in **dev pass-through mode** (a WARN
is logged once at startup and all internal requests are accepted without
a header). This keeps local development and CI friction-free.

```bash
# Production-style launch with auth enforced:
export AIPIPELINE_INTERNAL_SECRET=change-me-in-production
export AIPIPELINE_WORKER_INTERNAL_SECRET=change-me-in-production
(cd core-api && mvn spring-boot:run)      # terminal 2
(cd ai-worker && python -m app.main)      # terminal 3
```

### S3 / MinIO storage backend

By default artifacts are stored on the local filesystem (`backend=local`).
To use MinIO (S3-compatible) instead:

```bash
# 1. Start MinIO + auto-create the bucket
docker compose --profile minio up -d minio minio-bootstrap

# 2. Set env vars (or add to .env)
export AIPIPELINE_STORAGE_BACKEND=s3
export AIPIPELINE_STORAGE_S3_ENDPOINT=http://localhost:9000
export AIPIPELINE_STORAGE_S3_BUCKET=aipipeline-artifacts
export AIPIPELINE_WORKER_S3_ENDPOINT=http://localhost:9000
export AIPIPELINE_WORKER_S3_ACCESS_KEY=aipipeline
export AIPIPELINE_WORKER_S3_SECRET_KEY=aipipeline_secret

# 3. Start core-api and worker as usual
```

The MinIO console is available at `http://localhost:9001` (user:
`aipipeline`, password: `aipipeline_secret`).

Expected smoke output:
```
[1/4] submitting text job ...
...
[4/4] downloading output artifact content ...
     parsed JSON keys = ['jobId', 'capability', ...]
OK - pipeline survived the round trip.
```

## One-command demo

A richer demo script that uploads a sample PDF, runs the full
MULTIMODAL pipeline, and pretty-prints results:

```bash
pip install reportlab                # one-time: PDF generation
pip install rich                     # optional: coloured output

python scripts/demo.py               # default: MULTIMODAL
python scripts/demo.py --capability OCR
python scripts/demo.py --question "What metrics are shown?"
```

The script auto-generates a 2-page sample PDF in `scripts/assets/`
(gitignored) on first run. Pass `--self-test` to verify imports and PDF
generation without a running backend:

```bash
python scripts/demo.py --self-test
```

Sample output (with `rich`):
```
[1/5] health check (http://localhost:8080) ...
[2/5] preparing sample PDF ...
  .../scripts/assets/demo_sample.pdf (12,345 bytes)
[3/5] submitting MULTIMODAL job ...
  jobId = abc-123  status = QUEUED
[4/5] polling for completion ...
  ... RUNNING
  SUCCEEDED
[5/5] fetching results ...
+----- Job Summary -----+
| Job ID: abc-123       |
| Capability: MULTIMODAL|
| Status: SUCCEEDED     |
+-----------------------+
--- FINAL_RESPONSE ---
(grounded markdown answer)
Demo complete.
```

## Verified on this machine (phase 2)

Phase 1.1 results still hold. New phase-2 verification:

- **Flyway V2** migration: `Successfully applied 1 migration to schema "aipipeline", now at version v2`, `ragmeta` schema created alongside pipeline tables.
- **Worker startup with RAG**: `Active capabilities: ['MOCK', 'RAG']` after loading FAISS index + sentence-transformers model on cuda:0.
- **Index build**: `scripts/build_rag_index.py --fixture` ingests the 8-doc anime fixture → 24 chunks → 384-dim vectors → FAISS `IndexFlatIP` + ragmeta rows in 10.4s.
- **RAG query 1** ("anime about an old fisherman feeding stray harbor cats") → top hit `anime-005#overview` score **0.845**, correct answer ("The Harbor Cats"), 5 grounded citations.
- **RAG query 2** ("anime about weather researchers stranded in a mountain observatory during a long storm") → top hit `anime-004#plot` score **0.515**, correct answer ("Signal Fires").
- **MOCK regression**: still succeeds alongside RAG, returns FINAL_RESPONSE unchanged.
- **Artifact shape**: each RAG job produces exactly one INPUT_TEXT + one RETRIEVAL_RESULT + one FINAL_RESPONSE row (no duplicates).
- **Unit tests**: `10 passed in 0.26s` (chunker edge cases + mock capability + full RAG capability with hashing fallback embedder).

## Next phases

- **Phase 2.1 (OCR + file input)** — add an OCR capability, expand the
  `INPUT_FILE` path to run OCR → `OCR_TEXT` → chunk → index. The
  chunker, embedding provider, generation provider, and retriever from
  phase 2 all extend without rewrites — OCR is a new capability that
  produces an OCR_TEXT artifact which the RAG capability can then consume.
- **Phase 3+** — multimodal capability, real LLM generation provider,
  MinIO/S3 storage adapter, retry orchestration, Kubernetes manifests,
  per-capability Redis lanes, real frontend.

See [`docs/architecture.md`](docs/architecture.md) for design decisions
and [`docs/api-summary.md`](docs/api-summary.md) for the concrete endpoint
contracts.
