# Local run guide

This walks through running the full pipeline on a single developer machine.
You need four things alive at the same time: PostgreSQL, Redis, core-api,
and ai-worker. A fifth optional actor is the test frontend (a static HTML
file you open in a browser).

## Versions

| Component      | Version                                |
|----------------|----------------------------------------|
| Java           | 21 (LTS)                               |
| Spring Boot    | 4.0.3                                  |
| Maven          | 3.9+                                   |
| Python         | 3.12 (3.13 also works)                 |
| PostgreSQL     | 18                                     |
| Redis          | latest (7.x in phase 1, no hard floor) |

## Prerequisites

- **Java 21** on `PATH`. Verify: `java -version` shows `21.x.x`.
- **Maven 3.9+**: `mvn -v` shows Maven 3.9 or newer and reports Java 21.
  If Maven is missing and you have miniconda:
  `conda install -c conda-forge maven -y`. Otherwise download from
  https://maven.apache.org/download.cgi and put `bin/` on PATH.
- **Python 3.12** (or 3.13): `python --version`.
- **Docker Desktop** or a compatible engine with `docker compose` v2.
- About 1 GB free disk for the Maven local repo on first build.

## Two DB execution modes

Pick exactly one — they are mutually exclusive and both are supported.

### Mode A — reuse an existing postgres container (default)

Recommended when you already run another postgres container on the host
(for example, RIA's `ria-postgres` on port 5432). This mode puts the
AI-platform data in a **separate `aipipeline` database** inside that
container so there is zero schema or table collision with the host
project — the DBs don't even share a logical namespace.

One-time bootstrap (creates the isolated user + database):

```bash
# Replace 'ria-postgres' / -U ria with your actual container and superuser
docker exec ria-postgres psql -U ria -d ria_core -c \
  "CREATE USER aipipeline WITH PASSWORD 'aipipeline_pw';"

docker exec ria-postgres psql -U ria -d ria_core -c \
  "CREATE DATABASE aipipeline OWNER aipipeline;"

docker exec ria-postgres psql -U ria -d ria_core -c \
  "GRANT ALL PRIVILEGES ON DATABASE aipipeline TO aipipeline;"
```

Verify:

```bash
docker exec ria-postgres psql -U aipipeline -d aipipeline \
  -c "SELECT current_database(), current_user;"
#   current_database | current_user
#  ------------------+--------------
#   aipipeline       | aipipeline
```

Then just start redis from our compose (we don't own postgres in this
mode):

```bash
cd <repo-root>
docker compose up -d redis
```

### Mode B — independent infra compose (no RIA, no host postgres)

Use this on a clean machine, or when you want hard isolation from any
other infra. This brings up **postgres:18 on host port 5433** so it
cannot collide with a 5432 already in use.

```bash
cd <repo-root>
docker compose --profile independent up -d
docker compose ps
# should list aipipeline-redis and aipipeline-postgres both healthy
```

If you use Mode B you must also tell core-api to point at port 5433 —
easiest way is to run it with the `independent` Spring profile:

```bash
cd core-api
mvn spring-boot:run -Dspring-boot.run.profiles=independent
```

Or set the env var `SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5433/aipipeline`
before starting core-api. See `.env.example`.

## Starting the whole stack (Mode A shown)

Four terminals or four tmux panes. Order matters for the first time
only: redis must exist before core-api tries to publish, and core-api
must exist before the worker tries to claim.

### Terminal 1 — infra

```bash
cd <repo-root>
docker compose up -d redis
# Mode A: rely on existing ria-postgres (already running)
# Mode B: docker compose --profile independent up -d
```

### Terminal 2 — core-api

```bash
cd core-api
mvn spring-boot:run
```

Expected log tail:

```
INFO  o.f.core.internal.command.DbMigrate  : Successfully applied 1 migration
      to schema "aipipeline", now at version v1
INFO  .c.s.a.o.l.LocalFilesystemStorageAdapter : Local storage root: .../local-storage
INFO  o.s.boot.tomcat.TomcatWebServer    : Tomcat started on port 8080 (http)
INFO  c.aipipeline.coreapi.CoreApiApplication : Started CoreApiApplication in ~3 s
```

Confirm health from another shell: `curl -s http://localhost:8080/actuator/health`
should print `{"status":"UP", ...}`.

### Terminal 3 — ai-worker

```bash
cd ai-worker
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m app.main
```

Expected log tail:

```
INFO [__main__] Starting worker id=worker-local-1 core_api=http://localhost:8080
INFO [app.queue.redis_consumer] Queue consumer started key=aipipeline:jobs:pending block_timeout=5s
```

### Terminal 4 — smoke test

```bash
cd <repo-root>
python scripts/e2e_smoke.py
```

Expected:

```
[1/4] submitting text job ...
     jobId = ...  status = QUEUED
[2/4] polling job status ...
     status = QUEUED
     status = SUCCEEDED
[3/4] fetching job result ...
     output artifactId = ...  type = FINAL_RESPONSE
[4/4] downloading output artifact content ...
     parsed JSON keys = ['jobId', 'capability', ...]
OK - pipeline survived the round trip.
```

Exit code 0 on success.

## Optional: browser frontend

Open `frontend/index.html` directly in a browser. Phase 1 has no CORS
configured, so if your browser blocks the cross-origin call from a
`file://` page you can either:

- host the file via a local static server (e.g. `python -m http.server 8000`
  inside `frontend/` and open `http://localhost:8000/`), or
- run it as an HTML artifact from an IDE.

## Verifying the data actually landed

After the smoke test:

```bash
docker exec ria-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT id, capability, status FROM job ORDER BY updated_at DESC LIMIT 5;"

docker exec ria-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT job_id, role, type, size_bytes FROM artifact ORDER BY created_at DESC LIMIT 10;"
```

Each successful job should have **exactly** one INPUT and one OUTPUT row.
Two OUTPUT rows per job = double-write regression — file a bug.

## Shutting down

- **Terminals 2 and 3**: `Ctrl+C`.
- **Redis**: `docker compose down` (or leave it — it's harmless).
- **Mode B postgres**: `docker compose --profile independent down`.
  Add `-v` to wipe the volume for a truly fresh start.
- **Wiping only aipipeline data without touching RIA** (Mode A):
  ```bash
  docker exec ria-postgres psql -U aipipeline -d aipipeline -c \
    "TRUNCATE TABLE artifact, job CASCADE;"
  ```

## Troubleshooting

### core-api

- **`connection refused` against localhost:5432** — in Mode A, your host
  postgres is not running. Start it (or switch to Mode B). In Mode B,
  you forgot `--profile independent` and nothing owns 5432.
- **Flyway error `Schema validation: missing table [artifact]`** — Flyway
  did not run before Hibernate. In Spring Boot 4.0 you need
  `spring-boot-starter-flyway` in the POM (not just `flyway-core`). The
  shipped pom.xml already has this; if you customized the POM, make
  sure the starter is present.
- **`No qualifying bean of type ObjectMapper`** — Spring Boot 4.0 no longer
  auto-publishes a singleton ObjectMapper from `starter-web` alone. The
  shipped code provides one explicitly in
  `core-api/.../common/web/JacksonConfig.java`. Don't delete it.
- **Port 8080 already in use** — set `SERVER_PORT=8081` (or any free
  port) and update `AIPIPELINE_WORKER_CALLBACK_BASE_URL` on the worker
  to match.

### ai-worker

- **`Redis ping failed`** at startup — Redis isn't running or the URL is
  wrong. `docker compose up -d redis` and try again.
- **Worker processes a job but callback returns 500 with
  `ValidationError: UploadResponse missing field artifactId`** — you're
  running an older worker against a newer core-api (or vice versa).
  Restart both after pulling. The upload endpoint intentionally returns
  only `storageUri`, `sizeBytes`, `checksumSha256` in phase 1 — no
  artifact id.
- **`FileNotFoundError: Artifact missing on shared disk`** — core-api
  and the worker are looking at different local-storage roots. Set
  `AIPIPELINE_STORAGE_LOCAL_ROOT_DIR` (core-api) and
  `AIPIPELINE_WORKER_LOCAL_STORAGE_ROOT` (worker) to the same absolute
  path. The shipped defaults resolve to `{repo-root}/local-storage`
  when each process is run from its own directory.

### Smoke test

- **`UnicodeEncodeError: cp949 codec can't encode character`** — fixed in
  phase 1.1; the script no longer uses non-ASCII characters. If you see
  this, you're on an old checkout.
- **Job stays `QUEUED` forever** — the worker isn't running, or it's
  pointed at the wrong Redis / wrong queue key. Check worker logs;
  verify `AIPIPELINE_WORKER_REDIS_URL` and `AIPIPELINE_WORKER_QUEUE_PENDING_KEY`
  match the core-api side.

## Windows-specific notes

- Use git-bash or PowerShell. `cmd.exe` also works but some of the
  one-liners in this doc assume POSIX shell.
- Filesystem paths in .env use forward slashes even on Windows (Spring
  and Python both handle them).
- `taskkill //F //PID <pid>` to force-kill a hung core-api or worker
  from git-bash (note the doubled `//` for git-bash argument escaping).
- Port listings: `netstat -an | findstr LISTEN` (Windows), or
  `netstat -an | grep LISTEN` (git-bash).

## Phase 2: RAG capability

Phase 2 adds a real text-RAG capability to the worker. The pipeline
itself is unchanged — it's a second entry in the worker's capability
registry, selected by the job's `capability` field.

### Prerequisites

The RAG capability loads a FAISS index and a sentence-transformers
model into the worker process at startup. Before it can do that you
must:

1. **Let core-api run the V2 Flyway migration.** Just restart core-api
   after pulling phase 2 code — Flyway applies `V2__ragmeta_schema.sql`
   automatically, creating the `ragmeta` schema inside the existing
   `aipipeline` database.
2. **Build the FAISS index once.** The worker refuses to register the
   RAG capability if the index isn't on disk yet. It still registers
   MOCK, so phase-1 flows keep working:
   ```bash
   cd ai-worker
   python -m scripts.build_rag_index --fixture
   ```
   This uses the committed 8-document anime fixture under
   `fixtures/anime_sample.jsonl`. The first run downloads the default
   embedding model (`BAAI/bge-m3`, ~2.3 GB, multilingual, 1024-dim)
   into the HuggingFace cache. Subsequent runs are fast. If disk or
   bandwidth is tight, override with a smaller model — see
   [Changing the embedding model](#changing-the-embedding-model) below.
3. **Restart the worker.** The worker loads the FAISS index + the
   embedding model into memory at startup, so any rebuild requires a
   worker restart. You should see something like:
   ```
   RAG init: configured_model=BAAI/bge-m3 query_prefix='' passage_prefix='' index_dir=../rag-data top_k=5
   Loaded FAISS index v-... (model=BAAI/bge-m3, dim=1024, vectors=...)
   Retriever readiness check: configured_model='BAAI/bge-m3' index_model='BAAI/bge-m3' configured_dim=1024 index_dim=1024 index_version=v-... chunk_count=...
   Retriever ready: model=BAAI/bge-m3 dim=1024 index_version=v-...
   RAG capability registered.
   Active capabilities: ['MOCK', 'RAG']
   ```
   If the worker prints `RAG capability NOT registered (RuntimeError:
   Embedding MODEL mismatch ...)` it means the configured model does
   not match the one that built the index. Rebuild + restart; see
   [Migrating the embedding model](#migrating-the-embedding-model).

### Submitting a RAG job

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":"which anime is about an old fisherman feeding stray harbor cats"}'
```

Poll `/api/v1/jobs/{id}` until status is `SUCCEEDED`, then fetch
`/api/v1/jobs/{id}/result`. The `outputs` array contains two artifacts:

- `RETRIEVAL_RESULT` — JSON payload with the query, topK, index version,
  embedding model, and every retrieved chunk's score + text.
- `FINAL_RESPONSE` — markdown answer, grounded in the retrieved chunks
  and citing source doc ids.

### Switching capabilities

Phase 2 leaves MOCK registered. To test MOCK, change `capability` to
`"MOCK"` in the request body. Both capabilities share the pipeline and
are distinguished only by the worker-side dispatch.

### Indexing a real dataset

The CLI accepts `--input <path>` for any JSONL file in the port/rag
schema:

```bash
python -m scripts.build_rag_index \
  --input D:/port/rag/app/scripts/namu_anime_v3.jsonl \
  --index-version prod-2026-04 \
  --notes "full namu_anime_v3 dataset rebuild"
```

Larger datasets will take longer to embed but the worker-side serving
path is agnostic to dataset size at phase-2 scale (IndexFlatIP is exact
over any number of vectors).

### Changing the embedding model

The default is `BAAI/bge-m3` — multilingual, 1024-dim, trained without
query/passage prefixes so both prefix env vars default to empty. To
switch models, set `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` (and the
prefix vars, if the new model needs them) **before** running both the
indexing CLI and the worker. The index sidecar records which model
built it, and the worker now does a **strict** runtime check: if the
configured model name is not byte-for-byte identical to the one in
`build.json`, RAG capability registration fails with a clear
`Embedding MODEL mismatch` RuntimeError. MOCK still registers. This is
intentional — two different models can share a vector dimension and
silently corrupt retrieval quality.

Supported options:

| Model                                | Dim  | Prefixes                     | Notes                                                    |
|--------------------------------------|------|------------------------------|----------------------------------------------------------|
| `BAAI/bge-m3` (default)              | 1024 | none (leave empty)           | Multilingual, ~2.3 GB download, strongest out of the box |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | none (leave empty)           | English-only, ~80 MB, fastest                            |
| `intfloat/multilingual-e5-small`     | 384  | `query: ` / `passage: `      | Multilingual, ~120 MB, requires prefixes                 |

E5-family example:

```
AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
AIPIPELINE_WORKER_RAG_EMBEDDING_PREFIX_QUERY="query: "
AIPIPELINE_WORKER_RAG_EMBEDDING_PREFIX_PASSAGE="passage: "
```

### Migrating the embedding model

Changing `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` is a **two-step
mandatory sequence**. Missing either step leaves the worker refusing
to register RAG on startup.

1. **Rebuild the FAISS index with the new model.**
   The old index is overwritten in place; the new `build.json` records
   the exact model name the CLI was run with. You will see it echoed
   in the final `Ingest done ...` log line.
   ```bash
   cd ai-worker
   # fixture rebuild (fast, for smoke testing)
   python -m scripts.build_rag_index --fixture

   # real dataset rebuild
   python -m scripts.build_rag_index \
     --input D:/port/rag/app/scripts/namu_anime_v3.jsonl \
     --index-version prod-bge-m3 \
     --notes "bge-m3 migration"
   ```

2. **Restart the worker.**
   The long-lived `RagCapability` holds the old FAISS index and
   embedder in memory; a rebuild is not hot-reloaded. Stop the worker
   (`Ctrl-C`) and restart it:
   ```bash
   cd ai-worker
   python -m app.main
   ```
   On a clean startup you should see `RAG capability registered.` and
   the readiness log line with the new model/dim. If you instead see
   `Embedding MODEL mismatch` it means the indexing CLI and the worker
   were run with different `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL`
   values — fix the env var and redo step 1.

### ragmeta schema

```
ragmeta.documents      (doc_id, title, source, category, metadata_json, created_at, updated_at)
ragmeta.chunks         (chunk_id, doc_id, section, chunk_order, text, token_count,
                        faiss_row_id, index_version, extra_json, created_at)
ragmeta.index_builds   (index_version, embedding_model, embedding_dim, chunk_count,
                        document_count, faiss_index_path, notes, built_at)
```

Peek at contents:

```bash
docker exec ria-postgres psql -U aipipeline -d aipipeline -c "\dt ragmeta.*"
docker exec ria-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT index_version, embedding_model, chunk_count FROM ragmeta.index_builds ORDER BY built_at DESC;"
```

## Phase 2: OCR capability

The OCR capability accepts an **INPUT_FILE** artifact (PNG, JPEG, or
PDF) and produces two output artifacts:

- **OCR_TEXT** — plain UTF-8 extracted text. Same shape a later RAG job
  can consume as INPUT_TEXT, which is how OCR→RAG chaining is meant to
  compose in a future phase.
- **OCR_RESULT** — JSON envelope with `filename`, `mimeType`, `kind`,
  `engineName`, `pageCount`, `textLength`, `avgConfidence`, per-page
  rollups, and a flat `warnings` array. The contract is stable enough
  to carry forward without schema churn.

Phase 1 OCR deliberately stops at "stable artifact generation". There
is **no** multimodal reasoning, image embedding, VLM answer generation,
or OCR+RAG chaining yet — those are explicit non-goals.

### Prerequisites

The default provider is **Tesseract** via **pytesseract**, with
**PyMuPDF** handling PDF page iteration. Setup is three steps, all
local:

1. **Install the Tesseract binary.**
   - **Windows**: grab the installer from
     https://github.com/UB-Mannheim/tesseract/wiki and let it install
     to `C:\Program Files\Tesseract-OCR\`. Then either add that folder
     to `PATH` or set `AIPIPELINE_WORKER_OCR_TESSERACT_CMD` to the
     absolute path of `tesseract.exe`.
   - **Linux**: `sudo apt install tesseract-ocr` (or your distro
     equivalent). The `tesseract` binary ends up on PATH automatically.
   - **macOS**: `brew install tesseract`.

2. **Install the Python wrappers.**
   ```bash
   cd ai-worker
   pip install -r requirements.txt
   ```
   This brings in `pytesseract`, `pymupdf`, and pins `Pillow`. None of
   these pull torch/transformers, so the install is fast even on a
   cold venv.

3. **(Optional) Install extra language packs.** Tesseract's `eng`
   pack ships with the default install; anything else needs an
   additional download. For Korean, Japanese, Chinese, etc. install
   the corresponding traineddata:
   - **Windows installer**: pick the languages in the installer UI.
   - **Linux**: `sudo apt install tesseract-ocr-kor tesseract-ocr-jpn`
     (etc.)
   - **macOS**: `brew install tesseract-lang` (installs everything).

   Then set `AIPIPELINE_WORKER_OCR_LANGUAGES` to a `+`-joined string:
   ```
   AIPIPELINE_WORKER_OCR_LANGUAGES=eng+kor
   ```
   The worker **refuses to register** the OCR capability at startup
   if any requested language pack is missing, and logs exactly which
   ones. MOCK and RAG are unaffected.

### Starting the worker with OCR

Once Tesseract + the Python wrappers are installed, just start the
worker normally:

```bash
cd ai-worker
python -m app.main
```

On a healthy startup you should see:

```
OCR init: languages=eng pdf_dpi=200 tesseract_cmd=<PATH> min_conf_warn=40.0 max_pages=100
TesseractOcrProvider ready: tesseract=5.3.3 languages=eng pdf_dpi=200
OCR capability registered.
Active capabilities: ['MOCK', 'OCR', 'RAG']
```

If Tesseract is not installed or the language packs are missing,
you'll instead see a clean warning like:

```
OCR capability NOT registered (OcrError: Could not execute the tesseract binary. ...)
Active capabilities: ['MOCK', 'RAG']
```

MOCK and RAG still register; OCR jobs submitted while in this state
fail with `UNKNOWN_CAPABILITY` until Tesseract is installed and the
worker is restarted.

### Submitting an OCR job

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/receipt.png"
```

Poll `/api/v1/jobs/{id}` until `SUCCEEDED`, then fetch
`/api/v1/jobs/{id}/result`. The `outputs` array holds two artifacts
(`OCR_TEXT` and `OCR_RESULT`).

### OCR configuration knobs (env vars)

| Env var                                         | Default | Notes                                                         |
|--------------------------------------------------|---------|---------------------------------------------------------------|
| `AIPIPELINE_WORKER_OCR_ENABLED`                  | `true`  | Set to `false` to skip OCR registration entirely.             |
| `AIPIPELINE_WORKER_OCR_LANGUAGES`                | `eng`   | `+`-joined Tesseract language codes (`eng+kor`, `eng+jpn`). |
| `AIPIPELINE_WORKER_OCR_TESSERACT_CMD`            | *unset* | Absolute path to `tesseract.exe` if not on PATH.              |
| `AIPIPELINE_WORKER_OCR_PDF_DPI`                  | `200`   | DPI used when rasterizing scanned PDF pages.                  |
| `AIPIPELINE_WORKER_OCR_MIN_CONFIDENCE_WARN`      | `40.0`  | Add a warning to `OCR_RESULT` when avg confidence is below this. Does **not** fail the job. Set to `0` to disable. |
| `AIPIPELINE_WORKER_OCR_MAX_PAGES`                | `100`   | Hard cap on PDF page count. Exceeding the cap produces `OCR_TOO_MANY_PAGES`. |

### Supported input types

| Kind  | Content types accepted          | Extensions accepted | Notes                                                                 |
|-------|---------------------------------|---------------------|-----------------------------------------------------------------------|
| Image | `image/png`, `image/jpeg`       | `.png`, `.jpg`, `.jpeg` | Single-frame. Provider reports per-image avg confidence.         |
| PDF   | `application/pdf`, `application/x-pdf` | `.pdf`          | Page-by-page. Born-digital pages use the PDF text layer (no OCR, no confidence); scanned pages are rasterized at `ocr_pdf_dpi` and OCR'd. |

Everything else fails with a typed `UNSUPPORTED_INPUT_TYPE` error
visible in the job's `errorCode` field.

### Known limitations (phase 1 OCR)

- **No image preprocessing.** We hand raw bytes to Tesseract. For
  low-contrast or rotated scans, quality will be poor. A future
  Pillow-based deskew/threshold pass is a natural next step.
- **No layout / bounding boxes.** `OCR_RESULT` reports text + avg
  confidence only. Tesseract's word-level bounding boxes are not
  surfaced.
- **No handwriting.** Tesseract is optimized for typeset text.
- **PDF page cap** (`ocr_max_pages`, default 100) is a safety gate,
  not a fundamental limit. Raise via env var if you need more.
- **No chaining yet.** OCR output is two independent artifacts. A
  later phase will add an orchestration step that takes the `OCR_TEXT`
  output of job A and feeds it as `INPUT_TEXT` to a RAG job B; phase
  1 OCR deliberately stops short of that.

## Running tests

Worker-side unit tests (no infra needed):

```bash
cd ai-worker
python -m pytest tests/ -q
```

Expected: `29 passed` — 2 mock capability, 6 chunker, 2 RAG happy-path,
6 RAG validation, and 13 OCR (5 happy-path variants, 3 unsupported-type
rejections, 2 empty/low-confidence edge cases, 3 registry resilience
cases). All OCR tests use a fake provider, so they run without
Tesseract or PyMuPDF installed; all RAG tests use the hashing fallback
embedder, so no model download and no Postgres needed.

## Configuration knobs (env vars)

See `.env.example` at the repo root for the full list. Most commonly
touched:

| Env var                                    | Default                         | Notes                               |
|--------------------------------------------|---------------------------------|-------------------------------------|
| `SPRING_DATASOURCE_URL`                    | jdbc:postgresql://localhost:5432/aipipeline | Mode A default; Mode B uses :5433  |
| `SPRING_DATASOURCE_USERNAME` / `_PASSWORD` | aipipeline / aipipeline_pw      |                                     |
| `SPRING_DATA_REDIS_HOST` / `_PORT`         | localhost / 6379                |                                     |
| `SERVER_PORT`                              | 8080                            | core-api listen port                |
| `AIPIPELINE_STORAGE_LOCAL_ROOT_DIR`        | `${user.dir}/../local-storage`  | shared disk for artifacts           |
| `AIPIPELINE_WORKER_WORKER_ID`              | worker-local-1                  | claim token                         |
| `AIPIPELINE_WORKER_CORE_API_BASE_URL`      | http://localhost:8080           |                                     |
| `AIPIPELINE_WORKER_REDIS_URL`              | redis://localhost:6379/0        |                                     |
| `AIPIPELINE_WORKER_LOCAL_STORAGE_ROOT`     | `../local-storage`              | must resolve to the same folder as core-api |
