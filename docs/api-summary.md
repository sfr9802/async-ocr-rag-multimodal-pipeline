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

| Capability | Status       | Output artifact types                   |
|------------|--------------|------------------------------------------|
| `MOCK`     | phase 1      | `FINAL_RESPONSE` (JSON echo)             |
| `RAG`      | phase 2      | `RETRIEVAL_RESULT` + `FINAL_RESPONSE`    |
| `OCR`      | phase 2.1+   | (reserved — not yet implemented)         |
| `MULTIMODAL` | later phase | (reserved — not yet implemented)         |

Before submitting a RAG job the worker-side FAISS index must be built
with `python -m scripts.build_rag_index --fixture` and the worker must
be restarted. See `docs/local-run.md`.

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
- `capability` — e.g. `MOCK`
- `file` — the binary upload

Response `202 Accepted`: same shape as the JSON variant, but the input
artifact is `INPUT_FILE`.

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
{ "code": "INVALID_ARGUMENT", "message": "capability must not be blank" }
```

Known codes: `INVALID_ARGUMENT`, `VALIDATION_ERROR`, `INVALID_STATE_TRANSITION`,
`CONFLICT`, `INTERNAL_ERROR`.
