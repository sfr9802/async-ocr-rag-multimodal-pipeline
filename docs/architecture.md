# Architecture — AI Processing Platform (phase 1 / 1.1)

## Purpose

A generic AI job processing platform. Clients submit work (text or files),
the platform schedules it onto long-lived workers, workers execute a
capability (OCR / RAG / multimodal / mock), and results are returned as
artifacts the client can fetch later.

Phase 1 delivers **only the skeleton**: the job lifecycle, artifact model,
claim / callback contract, Redis-based dispatch, a local-filesystem storage
backend, and a single `MOCK` capability that echoes its input. Real engines
land in phase 2+.

## Target stack (phase 2)

- Java 21, Spring Boot **4.0.3**
- Python 3.12 (3.13 also works)
- PostgreSQL 18
- Redis (latest)
- Maven 3.9+
- **FAISS (faiss-cpu)**, **sentence-transformers**, **psycopg2** (phase 2 RAG capability)

Phase 1.1 aligned the Spring Boot version to 4.0.3 and verified the
async skeleton. Phase 2 adds the first real capability on top of that
skeleton without touching the pipeline structure.

## Phase 2 addition: text-RAG capability

Phase 2 replaces "there is only a MOCK capability" with "there is a MOCK
capability and a real text-RAG capability, and both live behind the same
`Capability` interface".

The RAG capability has two logically distinct paths:

- **Indexing path** (offline / one-shot) — `scripts/build_rag_index.py`
  reads a JSONL dataset, chunks documents, embeds each chunk with
  sentence-transformers, builds a FAISS `IndexFlatIP`, and persists
  document + chunk metadata into the `ragmeta` schema in PostgreSQL.
  This runs as a CLI, not inside the worker process.
- **Serving path** (online) — the worker at startup loads the FAISS
  index and the sentence-transformers model into memory, pings the DB,
  and registers the RAG capability. Per-job, the task runner passes the
  user query through the retriever → generation provider → output
  artifacts (`RETRIEVAL_RESULT` JSON + `FINAL_RESPONSE` markdown).

Both paths share the same chunker, embedding provider, FAISS wrapper,
and metadata store classes — they just use them in different orders.

### Storage split (intentional, enforced at the DB level)

| Store              | Purpose                                  | Phase |
|--------------------|------------------------------------------|-------|
| `aipipeline` schema (PostgreSQL)  | pipeline state: job, artifact | 1 |
| `ragmeta` schema (PostgreSQL)     | RAG metadata: documents, chunks, index_builds | 2 |
| FAISS index file on disk          | vectors + FAISS metadata     | 2 |
| Local filesystem (`local-storage/`) | artifact content blobs      | 1 |

No MongoDB. No vector DB service. The two PostgreSQL schemas live in the
same database but are on different sides of a deliberate namespace
boundary: Spring's JPA never maps ragmeta, and the worker's psycopg2
calls never touch `aipipeline`.

### RAG capability file layout

```
ai-worker/app/capabilities/rag/
├── capability.py       Capability interface impl — orchestrates retrieve + generate
├── chunker.py          greedy_chunk + window_by_chars (ported from port/rag)
├── embeddings.py       EmbeddingProvider abstract + SentenceTransformerEmbedder + HashingEmbedder (test fallback)
├── generation.py       GenerationProvider abstract + ExtractiveGenerator (grounded, cited, non-mock)
├── faiss_index.py      Thin FaissIndex wrapper over IndexFlatIP with a build.json sidecar
├── metadata_store.py   psycopg2 DAO for ragmeta.documents / chunks / index_builds
├── ingest.py           Ingestion service (JSONL -> chunks -> vectors -> FAISS + DB)
└── retriever.py        Retriever (query text -> embeddings -> FAISS search -> DB lookup)
```

### End-to-end RAG flow (per job)

1. Client `POST /api/v1/jobs` with `{"capability": "RAG", "text": "..."}`.
2. Core-api stages the text as an `INPUT_TEXT` artifact and dispatches via Redis (unchanged).
3. Worker BRPOPs the message, claims the job, receives the INPUT_TEXT URI (unchanged).
4. Task runner hands the artifact to the `RAG` capability instead of `MOCK`.
5. `RagCapability.run()`:
   - decodes the INPUT_TEXT bytes into a query string
   - calls `Retriever.retrieve(query)`:
     - embeds the query via `EmbeddingProvider.embed_queries`
     - runs FAISS `IndexFlatIP.search(top_k)`
     - resolves the returned row ids to chunks via `ragmeta.chunks` JOIN `ragmeta.documents` (one SQL query per job)
   - calls `GenerationProvider.generate(query, chunks)` to produce markdown
6. Capability emits two output artifacts:
   - `RETRIEVAL_RESULT` — JSON of the full retrieval report (query, top-k, index_version, embedding_model, ranked chunks with scores + text)
   - `FINAL_RESPONSE` — the grounded markdown answer
7. Task runner uploads both artifacts and posts the SUCCEEDED callback (unchanged).

The pipeline (steps 1-3, 6-7) is identical to phase 1. Only steps 4-5 are new.

## Phase 2 addition: OCR capability

Phase 2 also adds a first practical OCR capability, sharing the same
`Capability` interface and the same worker pipeline as MOCK and RAG.
OCR is deliberately scoped to **stable artifact generation** — no
multimodal reasoning, no image embedding, no VLM answer generation,
no OCR→RAG chaining in this phase. The contract is:

```
INPUT_FILE  ──► OcrCapability ──►  OCR_TEXT   (plain UTF-8 text)
                              ──►  OCR_RESULT (JSON envelope)
```

The `OCR_TEXT` artifact is shaped identically to what a later RAG job
would consume as `INPUT_TEXT`, which is how OCR→RAG chaining is meant
to compose in a future phase without rewriting OCR internals.

### OCR capability file layout

```
ai-worker/app/capabilities/ocr/
├── capability.py           OcrCapability — mime dispatch, artifact envelope, warning rollup
├── provider.py             OcrProvider abstract + OcrPageResult / OcrDocumentResult / OcrError
└── tesseract_provider.py   TesseractOcrProvider — pytesseract + PyMuPDF, lazy-imported
```

The provider seam is intentionally thin:

- `ocr_image(image_bytes) -> OcrPageResult` for PNG/JPEG inputs
- `ocr_pdf(pdf_bytes) -> OcrDocumentResult` for PDFs (the provider
  owns page iteration so it can fall back to the native text layer
  for born-digital PDFs before rasterizing)

Swapping Tesseract for EasyOCR, PaddleOCR, or a cloud API is a single
new file implementing `OcrProvider` and a one-line change in
`registry._build_ocr_capability` — everything upstream stays put.

### End-to-end OCR flow (per job)

1. Client `POST /api/v1/jobs` multipart with `capability=OCR` and a
   `file` field. Core-api stages the bytes as an `INPUT_FILE` artifact
   via the storage port and enqueues the job — **unchanged from
   phase 1**.
2. Worker BRPOPs, claims the job, resolves the `INPUT_FILE` storage
   URI to raw bytes via the local filesystem resolver — **unchanged**.
3. `TaskRunner` builds a `CapabilityInputArtifact` with the raw bytes,
   content type, and (new) a best-effort `filename` recovered from the
   storage URI's trailing `{uuid}-{filename}` segment.
4. `OcrCapability.run()`:
   - Picks the first `INPUT_FILE` artifact (rejects everything else
     with `UNSUPPORTED_INPUT_TYPE`).
   - Classifies the input by content type → filename extension → magic
     bytes, producing `(mime_type, "image" | "pdf")`. Unsupported
     inputs raise `CapabilityError("UNSUPPORTED_INPUT_TYPE")`.
   - Dispatches to `provider.ocr_image` or `provider.ocr_pdf`.
   - Collects per-page warnings, adds a document-level warning for
     zero-text extraction and for below-threshold average confidence,
     and rolls everything up into the envelope.
5. Capability emits two output artifacts:
   - `OCR_TEXT` — plain UTF-8 concatenation of per-page text in
     document order.
   - `OCR_RESULT` — JSON with `filename`, `mimeType`, `kind`,
     `engineName`, `pageCount`, `textLength`, `avgConfidence`,
     per-page rollups, and a flat `warnings` array.
6. Task runner uploads both artifacts and posts the SUCCEEDED callback
   — **unchanged**.

Steps 1, 2, 3 (except the `filename` backfill), and 6 are identical to
phase 1 and to RAG. Only steps 4 and 5 are new.

### PDF page handling

`TesseractOcrProvider.ocr_pdf` opens the PDF with PyMuPDF (no poppler,
no external binary, handled in-process) and iterates pages in order.
For each page it tries the native text layer first via `page.get_text()`
— born-digital PDFs return clean text with zero OCR cost. If the text
layer is empty or under 8 characters, the page is treated as scanned:
rasterize with `page.get_pixmap(dpi=ocr_pdf_dpi)` → hand the PNG bytes
to Tesseract → record `page N: no text layer, ran OCR at N dpi` as a
per-page warning. Document-level `avgConfidence` is the mean of
per-page confidences; pages handled via the text layer report `None`
for confidence (there is no OCR to measure), so a mixed document
yields a confidence averaged over only the OCR'd pages.

A hard page cap (`ocr_max_pages`, default 100) prevents runaway jobs
on multi-thousand-page scans. Exceeding the cap produces
`OCR_TOO_MANY_PAGES`, visible in the job's `errorCode`.

### OCR registry resilience

Like RAG, OCR registers opportunistically in `build_default_registry`.
`TesseractOcrProvider.ensure_ready()` probes the Tesseract binary and
the configured language packs at worker startup; any failure raises
`OcrError`, which the registry catches and turns into a clean
`OCR capability NOT registered (...)` warning. MOCK and RAG remain
unaffected. Unit tests pin this behaviour in
`tests/test_ocr_capability.py::test_ocr_failure_*`.

## High-level topology

```
                   ┌──────────────────┐
                   │   frontend /     │
                   │   test client    │
                   └────────┬─────────┘
                            │ HTTP (public)
                            ▼
                   ┌──────────────────┐
                   │    core-api      │  ← Spring Boot, DDD + hexagonal
                   │  (Java 21)       │
                   └──┬───────┬───┬───┘
                      │       │   │
            JPA       │       │   │ HTTP (claim / callback / upload)
                      ▼       ▼   │
               ┌──────────┐  LPUSH│
               │PostgreSQL│       │
               │  (SoT)   │       │
               └──────────┘       │
                            ┌─────┴──────┐
                            │   Redis    │
                            │  pending   │
                            │   list     │
                            └─────┬──────┘
                                  │ BRPOP
                                  ▼
                       ┌──────────────────────┐
                       │     ai-worker        │  ← FastAPI deps,
                       │   (Python, long-     │    long-lived,
                       │    lived)            │    capability registry
                       └──────────┬───────────┘
                                  │ read/write
                                  ▼
                       ┌──────────────────────┐
                       │  storage backend     │
                       │  local FS (phase 1)  │
                       │  MinIO / S3 (later)  │
                       └──────────────────────┘
```

## Why Redis + long-lived workers (not Cloud Tasks / serverless)

The platform is expected to run GPU, OCR, PDF, and FAISS-in-memory
workloads. Those are ill-suited to serverless: cold starts murder GPU
initialization, FAISS indexes want to live in worker RAM across many
requests, and per-invocation billing makes long processing expensive.

So: the worker is a regular OS process that you run as many copies of as
you have GPUs / cores, and Redis is just the "wake up, a new job is in the
queue" signal. The **state of truth stays in PostgreSQL**. Redis can be
lost and restored (at worst jobs sit in QUEUED until you re-dispatch) and
the platform still knows what every job is doing.

## Source of truth

| Concern                  | Owner             | Notes                                        |
|--------------------------|-------------------|----------------------------------------------|
| Job lifecycle / status   | PostgreSQL (SoT)  | Written by core-api only                     |
| Dispatch signaling       | Redis             | Ephemeral, reconstructable                   |
| Artifact bytes           | Storage backend   | Local FS now, MinIO/S3 later                 |
| Artifact metadata        | PostgreSQL        | Linked to jobs                               |
| Claim lease              | PostgreSQL        | Atomic conditional UPDATE                    |
| Callback idempotency     | PostgreSQL        | `last_callback_id` on job row                |

## Core-api layout (DDD + hexagonal)

```
core-api/src/main/java/com/aipipeline/coreapi/
├── CoreApiApplication.java
├── common/                          ← cross-cutting (properties, clock)
├── job/
│   ├── domain/                      ← pure Java, no framework imports
│   │   ├── Job.java                 ← aggregate root with state transitions
│   │   ├── JobId.java               ← value object
│   │   ├── JobCapability.java       ← enum
│   │   ├── JobStatus.java           ← state machine
│   │   └── JobStateTransitionException.java
│   ├── application/
│   │   ├── port/in/                 ← primary ports (use cases)
│   │   │   ├── JobManagementUseCase.java
│   │   │   └── JobExecutionUseCase.java
│   │   ├── port/out/                ← secondary ports
│   │   │   ├── JobRepository.java
│   │   │   └── JobDispatchPort.java
│   │   └── service/                 ← use case implementations
│   │       ├── JobCommandService.java
│   │       └── JobExecutionService.java
│   └── adapter/
│       ├── in/web/                  ← inbound: REST controllers + DTOs
│       │   ├── JobController.java
│       │   ├── InternalWorkerController.java
│       │   └── dto/*
│       └── out/persistence/         ← outbound: Spring Data JPA
│           ├── JobJpaEntity.java
│           ├── JobJpaRepository.java
│           └── JobPersistenceAdapter.java
├── artifact/                        ← same three-layer shape
│   ├── domain/
│   ├── application/
│   └── adapter/
├── queue/adapter/out/redis/         ← Redis dispatch adapter
└── storage/adapter/out/local/       ← Local-filesystem storage adapter
```

**Rule**: the domain layer imports nothing from Spring, JPA, Redis, or
Jackson. The application layer imports only its own ports and the domain.
Adapters are where framework dependencies live.

## Job state machine

```
    ┌─────────┐ enqueue  ┌────────┐ claim   ┌─────────┐ success  ┌──────────┐
    │ PENDING ├─────────▶│ QUEUED ├────────▶│ RUNNING ├─────────▶│SUCCEEDED │
    └─────────┘          └────────┘         └────┬────┘          └──────────┘
                                                 │ failure
                                                 ▼
                                             ┌────────┐
                                             │ FAILED │
                                             └────────┘
```

Transitions are encoded in `JobStatus.canTransitionTo(...)` — every service
call that changes state asks the enum first, so an illegal transition
throws a domain exception rather than silently corrupting state.

- **enqueue**: `JobCommandService.createAndEnqueue` creates the job, marks
  it QUEUED, and registers an afterCommit hook that hands the job to the
  `JobDispatchPort`. The database commit happens first so the worker can
  never see a row it cannot claim.
- **claim**: `JobExecutionService.claim` calls
  `JobRepository.tryAtomicClaim`, which runs a conditional UPDATE that
  only succeeds when the row is PENDING or QUEUED and no live lease is
  held by another worker. Atomicity lives at the SQL level so multiple
  workers racing on the same job never both win.
- **callback**: `JobExecutionService.handleCallback` validates the claim
  token the worker presents, checks the callback id against
  `last_callback_id` for idempotent replays, and only then transitions
  the job to SUCCEEDED or FAILED. Output artifacts are persisted in the
  same transaction.

## Redis dispatch contract

- **Pending list**: `aipipeline:jobs:pending` (LPUSH from core-api, BRPOP
  from worker). Single key in phase 1; per-capability lanes are a future
  refinement.
- **Message shape** (JSON):
  ```json
  {
    "jobId": "...",
    "capability": "MOCK",
    "attemptNo": 1,
    "enqueuedAtEpochMilli": 1744723200000,
    "callbackBaseUrl": "http://localhost:8080"
  }
  ```
- Messages carry only enough info for the worker to phone home. Input
  artifacts come back via the claim response, so we never duplicate state
  between Redis and Postgres.

## Storage contract

`ArtifactStoragePort` exposes four operations:

- `store(jobId, type, filename, contentType, content, length)` — write
  bytes, return a `StoredObject(storageUri, sizeBytes, checksumSha256)`.
- `openForRead(storageUri)` — resolve the opaque URI and return an
  `InputStream`. Phase 1 only handles `local://...`; S3/MinIO add their
  own schemes later.
- `generateDownloadUrl(artifactId)` — phase 1 returns a core-api route;
  phase 2 returns a real presigned URL.
- The worker in phase 1 reads `local://` directly off shared disk for
  speed, and falls back to HTTP downloads for any scheme it can't resolve
  locally.

## Worker layout

```
ai-worker/
└── app/
    ├── main.py                ← entrypoint: build registry, start consumer
    ├── core/
    │   ├── config.py          ← pydantic-settings (env-driven)
    │   └── logging.py
    ├── queue/
    │   ├── redis_consumer.py  ← BRPOP loop
    │   └── messages.py        ← QueueMessage wire shape
    ├── clients/
    │   ├── core_api_client.py ← claim / callback / upload HTTP
    │   └── schemas.py
    ├── storage/
    │   └── resolver.py        ← local://  → filesystem path
    ├── capabilities/
    │   ├── base.py            ← Capability interface + data classes
    │   ├── registry.py        ← name → instance map
    │   ├── mock_processor.py  ← phase 1 echo capability
    │   ├── rag/               ← placeholder for phase 2 (FAISS lives here)
    │   ├── ocr/               ← placeholder for phase 2
    │   └── multimodal/        ← placeholder for later
    └── services/
        └── task_runner.py     ← claim → fetch → run → upload → callback
```

The worker deliberately does **not** use DDD. It's a procedural execution
engine: one class per responsibility, straight-line orchestration in
`TaskRunner`. Capabilities are the only extension point, and they're a
plain interface.

## End-to-end flow

1. Client `POST /api/v1/jobs` (text) or multipart upload.
2. `JobController` stages the bytes through `ArtifactStoragePort.store`.
3. `JobCommandService.createAndEnqueue`:
   - creates `Job` (PENDING) → marks QUEUED,
   - persists input artifacts,
   - registers an afterCommit hook,
   - transaction commits,
   - hook fires `JobDispatchPort.dispatch` → `RedisJobDispatchAdapter`
     LPUSHes a `QueueMessage` onto `aipipeline:jobs:pending`.
4. Worker's `RedisQueueConsumer` BRPOPs the message and hands it to
   `TaskRunner.handle`.
5. `TaskRunner` calls `POST /api/internal/jobs/claim`. Core-api runs the
   atomic conditional UPDATE; if it wins, status becomes RUNNING and the
   response carries the list of input artifacts.
6. Worker reads input bytes via the storage resolver (local filesystem).
7. Worker runs the matching `Capability` (phase 1: `MockProcessor`).
8. Worker uploads each output artifact via
   `POST /api/internal/artifacts`. Core-api writes the bytes through the
   same `ArtifactStoragePort.store` and returns only
   `(storageUri, sizeBytes, checksumSha256)` — **it does NOT create an
   Artifact row at this point**. This matches the semantics of a real
   presigned-upload flow, where the upload lands directly on object
   storage and the database hears about the new object only from the
   callback.
9. Worker posts the final callback:
   `POST /api/internal/jobs/callback` with outcome SUCCEEDED / FAILED and
   the list of output artifact metadata (including the storage URIs it
   just received).
10. `JobExecutionService.handleCallback` applies the status transition AND
    persists the output Artifact rows (single authoritative write path).
    If the same `callbackId` arrives twice it's detected via
    `last_callback_id` on the job row and the second call becomes a
    no-op — no duplicate artifacts.
11. Client polls `GET /api/v1/jobs/{id}` / `GET /api/v1/jobs/{id}/result`
    and downloads the output via the returned `accessUrl`.

## What's deferred

| Concern                             | Phase |
|-------------------------------------|-------|
| Real OCR engine                     | 2     |
| FAISS-based RAG capability          | 2     |
| Multimodal model                    | 3     |
| MinIO / S3 storage adapter          | 2     |
| Auth on `/api/internal/*`           | 2     |
| Retry orchestration                 | 2     |
| Per-capability Redis lanes          | 2     |
| Kubernetes manifests                | 2+    |
| Frontend beyond a test form         | 2+    |

See `docs/local-run.md` for how to actually run it, and
`docs/api-summary.md` for the concrete endpoint contracts.
