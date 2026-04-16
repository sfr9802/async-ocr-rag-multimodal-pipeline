# API summary

Concrete shapes for every HTTP endpoint the platform exposes. All paths
are relative to `http://localhost:8080`.

Phase 1 established the pipeline (MOCK capability). Phase 2 adds the
`RAG` capability without changing any endpoint shapes — the job submission,
status, result, and artifact download endpoints are identical. Only the
`capability` value in the request body and the artifact `type` values in
the result change.

## Public (client-facing)

### `POST /api/v1/jobs`

Submit a text job.

Request (JSON):
```json
{ "capability": "MOCK", "text": "hello" }
```

or, to run the text-RAG capability (phase 2):
```json
{ "capability": "RAG", "text": "which anime is about an old fisherman feeding stray harbor cats" }
```

Accepted capability values:

| Capability   | Status  | Output artifact types                                                          |
|--------------|---------|---------------------------------------------------------------------------------|
| `MOCK`       | phase 1 | `FINAL_RESPONSE` (JSON echo)                                                    |
| `RAG`        | phase 2 | `RETRIEVAL_RESULT` + `FINAL_RESPONSE`                                           |
| `OCR`        | phase 2 | `OCR_TEXT` + `OCR_RESULT`                                                       |
| `MULTIMODAL` | phase 2 v1 | `OCR_TEXT` + `VISION_RESULT` + `RETRIEVAL_RESULT` + `FINAL_RESPONSE` (+ optional `MULTIMODAL_TRACE`) |

Before submitting a RAG or MULTIMODAL job the worker-side FAISS
index must be built with `python -m scripts.build_rag_index
--fixture` and the worker must be restarted. Before submitting an
OCR or MULTIMODAL job Tesseract and PyMuPDF must be available to
the worker. See `docs/local-run.md`.

**MULTIMODAL v1 definition.** "Multimodal" here means an
INPUT_FILE (PNG/JPEG/PDF) gets run through OCR + a visual-
description provider, and the two signals are fused into a
retrieval query + grounding context that feed the existing text-
RAG retriever and generator. This is explicitly **not** true
multimodal retrieval — see `docs/architecture.md` "Multimodal
v1 limitations" for the full list of deferred items.

Response `202 Accepted`:
```json
{
  "jobId": "5d31e42e-...",
  "status": "QUEUED",
  "capability": "MOCK",
  "inputs": [
    {
      "id": "art-...",
      "role": "INPUT",
      "type": "INPUT_TEXT",
      "contentType": "text/plain; charset=utf-8",
      "sizeBytes": 5,
      "checksumSha256": "...",
      "accessUrl": "/api/v1/artifacts/art-.../content"
    }
  ]
}
```

### `POST /api/v1/jobs` (multipart)

Submit a file job (same path, different content type).

Form fields:
- `capability` — e.g. `OCR`, `MULTIMODAL`
- `file` — the binary upload
- `text` — **optional** accompanying user question/prompt. When
  present, core-api stages it as a second `INPUT_TEXT` artifact
  alongside the `INPUT_FILE`. The `MULTIMODAL` capability uses it
  as the user question in its fusion step. `OCR` and `MOCK` ignore
  it — they only ever pick up one of `INPUT_FILE` / `INPUT_TEXT`
  respectively. Omitting the field yields a single-artifact job
  identical to the pre-multimodal contract, so existing OCR smoke
  tests are unaffected.

Example — MULTIMODAL job with an image + user question:
```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/invoice.png" \
  -F "text=what is the total amount on this invoice?"
```

Response `202 Accepted`: same shape as the JSON variant. When `text`
is present the response's `inputs` array contains two entries
(`INPUT_FILE` + `INPUT_TEXT`), otherwise just `INPUT_FILE`.

### Submission contract (capability / input matrix)

Both job-creation endpoints validate the submitted `capability` value
against the shape of the request BEFORE any bytes are staged in
storage and BEFORE any job row is persisted. Invalid submissions are
rejected with a stable error code and the async pipeline is left in a
pristine state — no `artifact` row, no `job` row, no Redis dispatch.

| Capability    | Endpoint   | `text` field                | `file` field                      | Allowed file types        |
|---------------|------------|-----------------------------|-----------------------------------|----------------------------|
| `MOCK`        | JSON       | required (may be empty)     | —                                 | —                          |
| `MOCK`        | multipart  | optional                    | **required, non-empty**           | any                        |
| `RAG`         | JSON       | **required, non-blank**     | —                                 | —                          |
| `RAG`         | multipart  | **required, non-blank**     | required, non-empty (ignored by worker) | any                  |
| `OCR`         | JSON       | — (rejected: FILE_REQUIRED) | **required via multipart**        | PNG, JPEG, PDF             |
| `OCR`         | multipart  | optional (ignored by worker)| **required, non-empty**           | PNG, JPEG, PDF             |
| `MULTIMODAL`  | JSON       | — (rejected: FILE_REQUIRED) | **required via multipart**        | PNG, JPEG, PDF             |
| `MULTIMODAL`  | multipart  | optional (user question)    | **required, non-empty**           | PNG, JPEG, PDF             |

Validation rules enforced at the API boundary:

1. **Capability is required** and must match the `JobCapability` enum
   (`MOCK`, `RAG`, `OCR`, `MULTIMODAL`). Blank → `CAPABILITY_REQUIRED`;
   unknown value → `UNKNOWN_CAPABILITY`.
2. **File-based capabilities on the JSON endpoint are rejected** with
   `FILE_REQUIRED`. OCR / MULTIMODAL must use the multipart endpoint.
3. **RAG requires a non-blank `text` field** on both endpoints —
   rejected with `TEXT_REQUIRED` when missing, empty, or whitespace-only.
4. **The multipart endpoint always requires a real file** on every
   capability. Missing file → `FILE_REQUIRED`. Zero-byte file →
   `FILE_EMPTY`.
5. **OCR and MULTIMODAL files must be PNG, JPEG, or PDF.** The
   validator accepts a match on EITHER the content-type header OR the
   filename extension (real-world clients sometimes send
   `application/octet-stream` for known file types, or drop the
   filename entirely). Anything outside the allowed set →
   `UNSUPPORTED_FILE_TYPE`.
6. **MULTIMODAL text is optional.** A blank or missing `text` field
   is accepted; the worker's fusion layer falls back to a neutral
   default retrieval query.
7. **MOCK preserves phase-1 compatibility.** Any file type is
   accepted for MOCK on multipart (no type gate); MOCK on the JSON
   endpoint accepts an empty (but non-null) `text` field.

### Submission examples

Valid — MOCK text job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"MOCK","text":"hello platform"}'
# → 202 Accepted, jobId=..., status=QUEUED
```

Valid — RAG text job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":"who feeds the harbor cats?"}'
# → 202 Accepted
```

Valid — OCR file job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/receipt.png"
# → 202 Accepted
```

Valid — MULTIMODAL with file + optional text:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/invoice.png" \
  -F "text=what is the total amount on this invoice?"
# → 202 Accepted
```

Valid — MULTIMODAL with file only (no question):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/diagram.pdf"
# → 202 Accepted
```

**Invalid** — RAG without text:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":""}'
# → 400 Bad Request
# { "code": "TEXT_REQUIRED",
#   "message": "RAG jobs require a non-blank 'text' field in the JSON body." }
```

**Invalid** — OCR on the JSON endpoint (wrong endpoint for a file job):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"OCR","text":"please extract"}'
# → 400 Bad Request
# { "code": "FILE_REQUIRED",
#   "message": "OCR jobs require a file upload. Use the multipart endpoint ..." }
```

**Invalid** — MULTIMODAL multipart without a file:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "text=what is this?"
# → 400 Bad Request
# { "code": "FILE_REQUIRED",
#   "message": "MULTIMODAL job requires a 'file' form field on the multipart endpoint." }
```

**Invalid** — OCR with a zero-byte file:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/dev/null;filename=blank.png;type=image/png"
# → 400 Bad Request
# { "code": "FILE_EMPTY",
#   "message": "Uploaded file is empty (0 bytes) ..." }
```

**Invalid** — OCR with an unsupported file type:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/cat.gif"
# → 400 Bad Request
# { "code": "UNSUPPORTED_FILE_TYPE",
#   "message": "Unsupported file type for OCR. Received contentType='image/gif' filename='cat.gif'. Supported types: PNG, JPEG, PDF ..." }
```

**Invalid** — unknown capability:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"SUMMARIZE","text":"hi"}'
# → 400 Bad Request
# { "code": "UNKNOWN_CAPABILITY",
#   "message": "Unknown capability: SUMMARIZE. Accepted values: MOCK, RAG, OCR, MULTIMODAL." }
```

### `GET /api/v1/jobs/{jobId}`

Status view. Returns:
```json
{
  "jobId": "...",
  "capability": "MOCK",
  "status": "SUCCEEDED",
  "attemptNo": 1,
  "errorCode": null,
  "errorMessage": null,
  "createdAt": "2026-04-15T12:34:56Z",
  "updatedAt": "2026-04-15T12:34:58Z"
}
```

404 if no job with that id exists.

### `GET /api/v1/jobs/{jobId}/result`

Full result view with every artifact.
```json
{
  "jobId": "...",
  "status": "SUCCEEDED",
  "inputs":  [ { "id": "...", "type": "INPUT_TEXT",     "accessUrl": "..." } ],
  "outputs": [ { "id": "...", "type": "FINAL_RESPONSE", "accessUrl": "..." } ],
  "errorCode": null,
  "errorMessage": null
}
```

For a RAG job, `outputs` contains two entries:

```json
{
  "jobId": "...",
  "status": "SUCCEEDED",
  "inputs":  [ { "id": "...", "type": "INPUT_TEXT", "accessUrl": "..." } ],
  "outputs": [
    {
      "id": "...",
      "type": "RETRIEVAL_RESULT",
      "contentType": "application/json",
      "accessUrl": "/api/v1/artifacts/.../content"
    },
    {
      "id": "...",
      "type": "FINAL_RESPONSE",
      "contentType": "text/markdown; charset=utf-8",
      "accessUrl": "/api/v1/artifacts/.../content"
    }
  ],
  "errorCode": null,
  "errorMessage": null
}
```

The `RETRIEVAL_RESULT` artifact downloads as JSON with this shape
(produced by `RagCapability._retrieval_payload`):

```json
{
  "query": "which anime is about an old fisherman feeding stray harbor cats",
  "topK": 5,
  "indexVersion": "v-1776253724",
  "embeddingModel": "sentence-transformers/all-MiniLM-L6-v2",
  "hitCount": 5,
  "results": [
    {
      "rank": 1,
      "chunkId": "1ddaaea67b0b094b82523744_0",
      "docId": "anime-005",
      "section": "overview",
      "score": 0.844866,
      "text": "..."
    }
  ]
}
```

The `FINAL_RESPONSE` artifact downloads as markdown produced by the
extractive generator: a short grounded answer plus a numbered list of
supporting passages with citations.

For a MULTIMODAL job, `outputs` contains four entries (five when
`AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true`):

```json
{
  "jobId": "...",
  "status": "SUCCEEDED",
  "inputs":  [
    { "id": "...", "type": "INPUT_FILE", "contentType": "image/png", "accessUrl": "..." },
    { "id": "...", "type": "INPUT_TEXT", "contentType": "text/plain; charset=utf-8", "accessUrl": "..." }
  ],
  "outputs": [
    { "id": "...", "type": "OCR_TEXT",         "contentType": "text/plain; charset=utf-8", "accessUrl": "..." },
    { "id": "...", "type": "VISION_RESULT",    "contentType": "application/json",          "accessUrl": "..." },
    { "id": "...", "type": "RETRIEVAL_RESULT", "contentType": "application/json",          "accessUrl": "..." },
    { "id": "...", "type": "FINAL_RESPONSE",   "contentType": "text/markdown; charset=utf-8", "accessUrl": "..." }
  ],
  "errorCode": null,
  "errorMessage": null
}
```

The `VISION_RESULT` JSON has this shape (produced by
`MultimodalCapability._vision_result_json`):

```json
{
  "filename": "invoice.png",
  "mimeType": "image/png",
  "kind": "image",
  "provider": "heuristic-vision-v1",
  "caption": "A portrait light-toned image (800x1200 pixels) dominated by white tones with moderate contrast.",
  "details": [
    "dimensions: 800x1200 pixels (portrait)",
    "mean brightness: 220.5/255 (light)",
    "contrast (stddev): 32.1 (moderate)",
    "dominant channel: neutral",
    "source mode: RGB"
  ],
  "pageNumber": 1,
  "latencyMs": 3.214,
  "warnings": [],
  "available": true
}
```

When the vision stage fails, `available` is `false`, `provider` /
`caption` / `details` / `pageNumber` / `latencyMs` are null, and
`warnings` records the reason.

The MULTIMODAL `RETRIEVAL_RESULT` schema is identical to the RAG
capability's — downstream consumers can treat them identically, the
only difference is the job's `capability` value in the status endpoint.

### `GET /api/v1/artifacts/{id}/content`

Streams the artifact bytes. Content-Type and Content-Length are set from
the stored metadata. 404 if unknown.

## Internal (worker-facing)

These live under `/api/internal/*`. Phase 1 leaves them unauthenticated;
production deployments must gate them behind a shared secret or mTLS.

### `POST /api/internal/jobs/claim`

Worker asks to take ownership of a job.

Request:
```json
{ "jobId": "...", "workerClaimToken": "worker-local-1", "attemptNo": 1 }
```

Response (granted):
```json
{
  "granted": true,
  "currentStatus": "RUNNING",
  "reason": null,
  "capability": "MOCK",
  "attemptNo": 1,
  "inputs": [
    {
      "artifactId": "...",
      "type": "INPUT_TEXT",
      "storageUri": "local://.../prompt.txt",
      "contentType": "text/plain; charset=utf-8",
      "sizeBytes": 5
    }
  ]
}
```

Response (denied):
```json
{
  "granted": false,
  "currentStatus": "RUNNING",
  "reason": "ALREADY_CLAIMED",
  "capability": null,
  "attemptNo": 0,
  "inputs": []
}
```

Reasons: `JOB_NOT_FOUND`, `JOB_TERMINAL`, `ALREADY_CLAIMED`, `CLAIM_RACE`.

### `POST /api/internal/jobs/callback`

Worker reports the terminal outcome.

Request:
```json
{
  "jobId": "...",
  "callbackId": "f1aa...",
  "workerClaimToken": "worker-local-1",
  "outcome": "SUCCEEDED",
  "errorCode": null,
  "errorMessage": null,
  "outputArtifacts": [
    {
      "type": "FINAL_RESPONSE",
      "storageUri": "local://.../mock-response.json",
      "contentType": "application/json",
      "sizeBytes": 187,
      "checksumSha256": "..."
    }
  ]
}
```

Response:
```json
{ "applied": true, "duplicate": false, "currentStatus": "SUCCEEDED" }
```

If the same `callbackId` arrives twice on the same job, the second call
returns `{ "applied": false, "duplicate": true }` and the job state is
left untouched.

### `POST /api/internal/artifacts` (multipart)

Phase 1 substitute for a presigned PUT URL. The worker uploads result
bytes to core-api and gets back a storage URI it can echo into the
subsequent callback.

Form fields:
- `jobId` — job this output belongs to
- `type` — e.g. `FINAL_RESPONSE`
- `file` — the binary output

Response:
```json
{
  "storageUri": "local://...",
  "sizeBytes": 187,
  "checksumSha256": "..."
}
```

**Important**: this endpoint writes **bytes only**. It does NOT create an
`artifact` row. The row is created by the subsequent `callback`, which
references the returned `storageUri`. This matches the semantics of a
real presigned-upload flow (where the upload lands directly on object
storage and the DB hears about it only from the callback) and prevents
the double-write that would otherwise happen when the callback echoes
the same bytes.

## Error envelope

All 4xx and 5xx responses follow a single shape:
```json
{ "code": "TEXT_REQUIRED", "message": "RAG jobs require a non-blank 'text' field in the JSON body." }
```

### Submission error codes (`POST /api/v1/jobs`)

These are raised by `JobSubmissionValidator` at the API boundary BEFORE
any storage write or job-row persistence. A 400 response with one of
these codes guarantees that the async pipeline has not been touched —
no `artifact` row was created, no `job` row was inserted, and no Redis
dispatch was issued. Clients can safely retry with a fixed request.

| Code                     | HTTP | Meaning                                                                                |
|---------------------------|------|----------------------------------------------------------------------------------------|
| `CAPABILITY_REQUIRED`     | 400  | `capability` field is missing / blank / whitespace-only.                               |
| `UNKNOWN_CAPABILITY`      | 400  | `capability` value is not one of `MOCK`, `RAG`, `OCR`, `MULTIMODAL`.                  |
| `TEXT_REQUIRED`           | 400  | Capability requires a non-blank `text` field but none was supplied (RAG; MOCK-JSON null). |
| `FILE_REQUIRED`           | 400  | Capability requires a file upload but none was supplied, or the capability was submitted on the wrong endpoint. |
| `FILE_EMPTY`              | 400  | A `file` form field was present but carried zero bytes.                                |
| `UNSUPPORTED_FILE_TYPE`   | 400  | File's content-type and filename extension both fall outside the allowed set for the capability (OCR / MULTIMODAL require PNG, JPEG, or PDF). |

### Other known error codes

| Code                        | Source                                                 |
|------------------------------|--------------------------------------------------------|
| `INVALID_ARGUMENT`          | Generic `IllegalArgumentException` from the application layer. |
| `VALIDATION_ERROR`          | Jakarta Validation constraint failure on a DTO field.  |
| `INVALID_STATE_TRANSITION`  | Domain state machine refused the transition.           |
| `CONFLICT`                  | Application-level conflict (idempotency, claim race).  |
| `INTERNAL_ERROR`            | Fallback for unhandled exceptions. 500 status.         |
