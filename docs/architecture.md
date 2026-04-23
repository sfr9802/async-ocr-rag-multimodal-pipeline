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

The default embedding model is multilingual (BAAI/bge-m3, 1024-dim),
covering Korean and English retrieval in a single index. The generation
provider is selectable: `extractive` (default, deterministic heuristic)
or `claude` (Claude LLM via Anthropic API, with automatic extractive
fallback on API failure). Set `AIPIPELINE_WORKER_RAG_GENERATOR=claude`
to use Claude generation.

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

## Phase 2 addition: MULTIMODAL capability (v1)

Phase 2 also ships a v1 multimodal capability that opens the pipeline
to image / PDF inputs. The product-level definition of "multimodal" at
this stage is deliberately narrow:

> MULTIMODAL v1 means an INPUT_FILE (image or PDF) gets run through
> OCR + a visual-description provider, and the two signals are fused
> into a retrieval query + grounding context that feed the existing
> text-RAG retriever and generator.

This is **not** true multimodal retrieval. There is no dedicated image
embedding index, no cross-modal nearest-neighbour search, no VLM
generation. Those are explicit non-goals for v1 — see
"Multimodal v1 limitations" below.

### MULTIMODAL capability file layout

```
ai-worker/app/capabilities/multimodal/
├── capability.py          MultimodalCapability — orchestrator for the five-stage pipeline
├── vision_provider.py     VisionDescriptionProvider abstract + VisionDescriptionResult / VisionError
├── heuristic_vision.py    HeuristicVisionProvider — deterministic Pillow-based fallback
└── fusion.py              build_fusion() + FusionResult — structured context builder
```

### End-to-end MULTIMODAL flow (per job)

```
INPUT_FILE  ──►  MultimodalCapability  ──►  OCR_TEXT         (plain UTF-8)
(+ optional                             ──►  VISION_RESULT    (JSON envelope)
 INPUT_TEXT                             ──►  RETRIEVAL_RESULT (same schema as RAG)
 question)                              ──►  FINAL_RESPONSE   (grounded markdown)
                                        ──►  MULTIMODAL_TRACE (optional JSON — gated
                                                               behind emit_trace)
```

Stages:

1. **Stage A — OCR extraction.** Reuses the same `OcrProvider` that
   backs the standalone OCR capability. Image inputs go through
   `ocr_image`; PDF inputs go through `ocr_pdf` (which handles
   born-digital text layers + rasterization per page). OCR failures
   are downgraded to non-fatal warnings — the capability continues
   with the vision-only signal.

2. **Stage B — vision description.** A `VisionDescriptionProvider`
   produces a caption + structured details from the file. For image
   inputs the raw bytes are handed straight to the provider. For PDF
   inputs the capability rasterizes PDF page 1 via PyMuPDF (already a
   worker dep for the OCR stack) and hands that PNG to the provider.
   v1 only captions page 1 — captioning every page would double cost
   for negligible product gain at this stage. Vision failures are
   downgraded to non-fatal warnings; the pipeline continues with the
   OCR-only signal.

3. **Stage C — fusion.** `build_fusion()` takes the optional user
   question, the OCR text, and the vision result, and produces a
   deterministic `FusionResult` with two fields:

   - `retrieval_query` — a short query string (capped at
     `max_query_chars`, default 400) handed to the existing
     `Retriever.retrieve(...)`. When the user question is short
     (< 5 whitespace-tokens) and OCR text is available, the fusion
     helper enriches the query with the first OCR keywords so the
     embedder has material to work with.
   - `fused_context` — a long, structured markdown block with three
     always-present sections: `### User question`, `### Extracted
     text (OCR)`, `### Visual description`. This is the block that
     actually grounds the final answer.

   The helper is intentionally pure and deterministic — given the
   same inputs it produces byte-identical output, so ops can diff
   `MULTIMODAL_TRACE` artifacts across runs and tests can reason
   about fusion behaviour without any stochasticity.

4. **Stage D — retrieval.** Exactly the same `Retriever.retrieve(...)`
   call as the standalone RAG capability, against the same FAISS
   index with the same embedding model. v1 deliberately reuses the
   RAG retriever so MULTIMODAL jobs see identical retrieval shape to
   pure RAG jobs.

5. **Stage E — generation.** The fused context is prepended as a
   synthetic rank-0 `RetrievedChunk` (doc_id `input:multimodal`,
   section `fused_context`, score 1.0) to the retrieval results
   before calling the existing `GenerationProvider.generate(...)`.
   The extractive generator picks its "short answer" sentence from
   the top-scoring chunk, which is now the fused context by
   construction — so the final answer is grounded in the job's
   actual OCR + vision signal, with the retrieved chunks following
   as supporting passages.

Steps 1–3 (core-api staging + dispatch) and step 6 (callback + artifact
persistence) are identical to phase 1 / RAG / OCR. Only the capability
internals are new.

### Provider seams

Two abstract classes pin the v1 extension points:

- `OcrProvider` — already present for the standalone OCR capability.
  MULTIMODAL reuses whatever instance the registry built.
- `VisionDescriptionProvider` — new in phase 2. Implementations must
  expose a single `describe_image(image_bytes, mime_type, hint,
  page_number) -> VisionDescriptionResult` method and a stable `name`
  property. v1 ships only `HeuristicVisionProvider`, which uses
  Pillow's `ImageStat` to emit a deterministic structural description
  (orientation, brightness, contrast, dominant channel). Swapping in
  a real VLM (BLIP-2, LLaVA, GPT-4V, Claude Vision, Gemini) is a
  single-file change + a one-line switch in
  `registry._build_vision_provider`.

### MULTIMODAL registry dependencies

MULTIMODAL is a **dependent capability**: it needs both the OCR
provider and the RAG retriever to have registered successfully.
`build_default_registry` enforces this explicitly:

1. RAG registration is attempted — `rag_registered = True` on success.
2. OCR registration is attempted — `ocr_registered = True` on success.
3. MULTIMODAL registration is attempted only if BOTH parents
   registered. Otherwise it is skipped with a warning that names the
   missing parent. MOCK / RAG / OCR remain unaffected in either case.
4. MULTIMODAL init failures inside the builder (e.g. unknown vision
   provider, Pillow import failure) produce a clean
   `MULTIMODAL capability NOT registered (...)` warning that leaves
   MOCK / RAG / OCR alone.

The OCR provider + RAG retriever are built once and cached in a
module-level dict (`_shared_component_cache`) so a worker that
registers RAG + OCR + MULTIMODAL does NOT end up with two
sentence-transformers models in memory or two Tesseract probes at
startup. The cache is cleared at the top of every
`build_default_registry` call so repeat invocations (tests,
in-process restarts) get fresh instances.

Unit tests pin this behaviour in
`tests/test_multimodal_capability.py::test_multimodal_failure_does_not_affect_mock_rag_or_ocr`
and the two parent-missing variants.

### Multimodal v1 limitations

The architecture is deliberately small for a reason. The following
are explicit non-goals for v1 and are **not** implemented:

- **Not true multimodal retrieval.** v1 uses OCR text + visual
  description to feed the existing **text** RAG retriever. There is
  no multimodal vector DB, no CLIP / BLIP-2 / VLM embedding index,
  no cross-modal nearest-neighbour search.
- **Dedicated image embeddings are deferred.** Image inputs do not
  produce their own vectors; they only contribute through OCR text
  + the (currently heuristic) vision description.
- **Default vision provider is claude-sonnet-4-6; heuristic remains as
  offline/CI fallback.** Set `AIPIPELINE_WORKER_MULTIMODAL_VISION_PROVIDER=claude`
  and provide an Anthropic API key to use Claude Vision. The
  `HeuristicVisionProvider` (Pillow-based structural description) is
  still the default for environments without an API key.
- **Per-page captions are single-page in v1.** Multi-page PDFs get
  OCR on every page but only page 1 goes through the vision stage.
  Extending this is a config knob change + a loop in the capability.
- **Multimodal evaluation is mostly manual.** The eval schema stub
  in `ai-worker/eval/datasets/multimodal_sample.jsonl` is a
  forward-looking placeholder. There is no multimodal harness yet —
  that's the next-phase deliverable.
- **No VLM answer generation.** `FINAL_RESPONSE` is produced by the
  existing extractive generator, which consumes the fused OCR + vision
  context as a synthetic retrieval chunk. Plugging in a real LLM
  generator is an independent change behind the existing
  `GenerationProvider` interface.

## Phase 3: AUTO capability

Phase 3 adds a single-pass **AUTO** capability — a dispatcher that
inspects the submitted `(text, file)` pair and routes the job to one of
the existing RAG / OCR / MULTIMODAL capabilities internally, so clients
no longer have to pick the capability up-front. Phase 6 upgrades the
same seam into a real agent loop; Phase 3 keeps it deliberately
simple — one router call, one sub-capability call, one callback.

### Why this lives in the worker, not core-api

Routing is a capability decision (what engine to run, which index to
hit, how to frame the question), not a submission decision (is the
payload valid). Putting it in the worker keeps core-api free of
capability-specific logic and means the router has access to the same
`LlmChatProvider` the RAG query parser already depends on.

### Five-stage flow

```
         ┌─────────────────┐
(text,   │ route_classify  │  decode INPUT_TEXT / INPUT_FILE into
 file) ─►│                 │──► text, file_bytes, file_mime
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ route_decide    │  AgentRouterProvider -> AgentDecision
         │ (rule | llm)    │  (action, confidence, reason,
         └────────┬────────┘   router_name, parsed_query?)
                  ▼
          ┌─────────────────┐
          │    dispatch     │ ─► MULTIMODAL.run(input)
          │                 │ ─► OCR.run(input)
          │                 │ ─► RAG.run(input)
          │                 │ ─► direct_answer (inline LLM call)
          │                 │ ─► clarify (inline Korean message)
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │   AGENT_DECISION│  JSON artifact with the routing metadata;
          │   + sub outputs │  always the FIRST output artifact.
          └─────────────────┘
```

### Rule-based router (default)

`RuleBasedAgentRouter` is the deterministic fallback and the default
when `AIPIPELINE_WORKER_AGENT_ROUTER=rule` (also the default value).
Its decision tree handles the ~95% of cases where routing is obvious
from the input shape alone:

| Input                                         | Action       | Confidence |
|-----------------------------------------------|--------------|------------|
| `text>6ch + file in {png,jpeg,pdf}`           | `multimodal` | 0.95       |
| `file in {png,jpeg,pdf}` only                 | `ocr`        | 0.90       |
| `text>6ch` only                               | `rag`        | 0.70       |
| `text<=6ch` only                              | `clarify`    | 0.50       |
| neither                                       | `clarify`    | 0.00       |

"File in {png,jpeg,pdf}" is by mime-type — unsupported types
(`image/gif`, `application/zip`, etc.) collapse to the "no-file"
row so a `text>6ch + gif` input still routes to RAG rather than
failing inside MULTIMODAL. Empty files (`file_size=0`) are also
treated as "no-file" at this layer.

### LLM router (opt-in)

`LlmAgentRouter` wraps the shared `LlmChatProvider` (same instance the
query parser uses) and asks it to pick between the five actions. On
backends that advertise `function_calling` (Ollama with gemma4, Claude)
it routes through `chat_tools` with a `route_job` tool spec; otherwise
it falls back to `chat_json` with a schema hint. Thinking mode is
enabled iff the backend advertises it.

The LLM router **degrades to the rule router** (never raises) on ANY of:

- `LlmChatError` from the underlying provider (network, timeout,
  invalid JSON, empty response).
- Schema violation at our layer (missing field, wrong type, action
  not in the 5-enum).
- `confidence < AIPIPELINE_WORKER_AGENT_CONFIDENCE_THRESHOLD` (default
  0.55) — the model answered but isn't sure enough to act on.

The fallback decision's `router_name` becomes
`f"llm-{chat.name}-fallback-rule"` so operators can diff clean LLM runs
from degraded runs in the `AGENT_DECISION` artifact without reading
logs.

### Registry wiring

`build_default_registry` registers AUTO opportunistically:

1. RAG, OCR, MULTIMODAL all attempt registration as usual.
2. If **at least one** of the three succeeded, AUTO is registered with
   references to whichever subs are live. Missing subs become `None`
   on the AutoCapability — the router can still route to them, but the
   capability raises a typed `AUTO_<sub>_UNAVAILABLE` error code at
   dispatch time rather than crashing with `AttributeError`.
3. If **none** of the three are live, AUTO is skipped with a warning —
   there is nothing to dispatch to.

Requesting the LLM router while `llm_backend=noop` (or the backend
downgraded to NoOp at worker startup) causes the registry to
auto-downgrade AUTO to the rule router with a warning. AUTO never
goes down because of a missing LLM.

### Single terminal callback

AUTO is a normal `Capability` — it takes one `CapabilityInput` and
returns one `CapabilityOutput`. The sub-capability's artifacts are
passed through to the outputs list unchanged (with `AGENT_DECISION`
prepended), so the TaskRunner issues exactly one terminal callback per
AUTO job, matching the shape it already handles for RAG / OCR /
MULTIMODAL. The sub-capability's internal trace (OCR_RESULT.trace or
MULTIMODAL_TRACE) is preserved verbatim — AUTO adds the routing
metadata as a sibling artifact rather than nesting its own trace.

### Environment variables

| Variable                                             | Default | Meaning                                                                 |
|-------------------------------------------------------|---------|-------------------------------------------------------------------------|
| `AIPIPELINE_WORKER_AGENT_ROUTER`                      | `rule`  | Router implementation — `rule` or `llm`.                                |
| `AIPIPELINE_WORKER_AGENT_CONFIDENCE_THRESHOLD`        | `0.55`  | LLM decisions below this confidence fall back to the rule router.       |
| `AIPIPELINE_WORKER_AGENT_DIRECT_ANSWER_MAX_TOKENS`    | `512`   | Max tokens requested when the LLM router selects `direct_answer`.       |

All three default to safe values, so an unset env produces the same
behaviour the deterministic rule router has shipped with since Phase 3.

## Pipeline trace and failure reporting

### Goal

When a MULTIMODAL or OCR job falls back, partially succeeds, or fails
outright, operators and developers need to see **where** and **why**
without reading the worker's structured log and correlating it with a
job id. The capability layer carries a **normalized stage trace** that
captures stage flow, timing, and typed error codes in a single compact
shape. Consumers get the same field names regardless of which
capability produced the job.

The trace is **reporting-only** — it does not change the pipeline's
success or fallback behavior. A MULTIMODAL job that would have
succeeded without the trace still succeeds with the trace, and an OCR
job that would have failed with a specific error code still fails with
that code. The trace adds observability, not new failure modes.

### Where the trace lives

| Capability  | Trace carrier             | Gate                         |
|-------------|---------------------------|-------------------------------|
| `OCR`       | `OCR_RESULT.trace`        | Always emitted (additive to existing OCR_RESULT envelope) |
| `MULTIMODAL`| `MULTIMODAL_TRACE` artifact | Opt-in via `AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true` |

Both carriers hold the **same schema** (`trace.v1`) so downstream
consumers can treat them identically. `OCR_RESULT`'s existing extraction
fields (`filename`, `mimeType`, `kind`, `engineName`, `pageCount`,
`textLength`, `avgConfidence`, `pages`, `warnings`) are unchanged —
the `trace` key is additive and consumers that don't parse it see zero
behavior change.

On any **failure path** that doesn't emit a trace artifact, the stage
flow summary is folded into the `CapabilityError.message` so operators
can still read it from the job's `errorMessage` field in
`GET /api/v1/jobs/{id}`. The summary line looks like:

```
classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms) vision:fail(VISION_VLM_TIMEOUT,5ms,fallback) fusion:skipped retrieve:skipped generate:skipped
```

### Schema (trace.v1)

The payload is flat — no nested maps other than per-stage `details`:

```jsonc
{
  "schemaVersion": "trace.v1",   // string — bump on breaking changes
  "capability": "MULTIMODAL",    // "OCR" | "MULTIMODAL"
  "inputKind":  "image",         // "image" | "pdf" | "text" | "unknown"
  "finalStatus":"ok",            // "ok" | "partial" | "failed"
  "stages": [
    {
      "stage":        "ocr",     // canonical stage name (see vocabulary)
      "provider":     "tesseract-5.3.3", // null for intrinsic stages like classify
      "status":       "ok",      // "ok" | "warn" | "fail" | "skipped"
      "code":         null,      // stable error/warning code on non-ok
      "message":      null,      // clipped to ~200 chars
      "retryable":    null,      // true | false | null (unknown)
      "fallbackUsed": false,     // did the pipeline continue past this?
      "durationMs":   42.1,      // wall-clock for the stage
      "details": {               // small per-stage metadata (no payload dumps)
        "pageCount":      1,
        "textLength":     28,
        "avgConfidence":  91.0,
        "engineName":     "tesseract-5.3.3"
      }
    }
    // ... more stage records ...
  ],
  "warnings": [],                // pipeline-level warnings not tied to a stage
  "summary":  "classify:ok(0ms) ocr:ok(42ms) ..."
}
```

### Stage name vocabulary

Canonical values for `stage`. The capability layer emits the first six;
the last four are reserved for the TaskRunner / outer orchestrator
(not currently populated inside the capability, but documented here so
the vocabulary is unified).

| Stage       | Emitted by                    | Meaning                                          |
|-------------|--------------------------------|--------------------------------------------------|
| `classify`  | OCR + MULTIMODAL               | Input file type / mime classification           |
| `ocr`       | OCR + MULTIMODAL               | Text extraction from image or PDF                |
| `vision`    | MULTIMODAL                     | Visual description (heuristic or VLM)            |
| `fusion`    | MULTIMODAL                     | OCR + vision + question fusion                   |
| `retrieve`  | MULTIMODAL                     | Text-RAG retrieval against FAISS                 |
| `generate`  | MULTIMODAL                     | Grounded answer generation                       |
| `fetch`     | *reserved (TaskRunner)*        | Downloading input artifact bytes                 |
| `decode`    | *reserved (TaskRunner)*        | Decoding artifact content                        |
| `upload`    | *reserved (TaskRunner)*        | Uploading output artifact bytes                  |
| `callback`  | *reserved (TaskRunner)*        | Reporting terminal state to core-api             |

### Status vocabulary

| Status      | When                                                                      |
|-------------|---------------------------------------------------------------------------|
| `ok`        | Stage completed cleanly.                                                  |
| `warn`      | Stage completed but with a non-fatal issue (e.g. empty OCR, low confidence). Often paired with `fallbackUsed=true` to mark a partial fallback. |
| `fail`      | Stage raised an exception. When `fallbackUsed=true`, the pipeline continued past this stage via a fallback path; when `false`, this failure is terminal. |
| `skipped`   | Stage was never executed (typically because an earlier stage failed).     |

### `finalStatus` values

| Value     | Meaning                                                          |
|-----------|------------------------------------------------------------------|
| `ok`      | Every stage was `ok` — clean success with no caveats.            |
| `partial` | At least one stage was `warn` or `fail(fallbackUsed=true)`, but the capability still produced the required output artifacts. |
| `failed`  | Terminal failure — a stage failed and no fallback was possible.  |

### Stable error / warning codes

The codes below are the authoritative registry recorded in
`ai-worker/app/capabilities/trace.py::STABLE_ERROR_CODES`. Clients
that key off error codes should match on this set; unknown codes from
underlying providers are preserved with a capability prefix (e.g. a
VLM returning `PROVIDER_TIMEOUT` surfaces as `VISION_PROVIDER_TIMEOUT`
in the stage record).

| Code                               | Emitted on                                           | Typical `retryable` |
|------------------------------------|------------------------------------------------------|---------------------|
| `UNSUPPORTED_INPUT_TYPE`           | Classify stage — file is not PNG / JPEG / PDF        | `false`             |
| `NO_INPUT`                         | Capability received zero input artifacts            | `false`             |
| `OCR_IMAGE_DECODE_FAILED`          | Pillow could not decode the image bytes              | `false`             |
| `OCR_PDF_OPEN_FAILED`              | PyMuPDF could not open the PDF (corrupt/encrypted)   | `false`             |
| `OCR_PDF_EMPTY`                    | Zero-page PDF                                        | `false`             |
| `OCR_TESSERACT_RUN_FAILED`         | Tesseract invocation failed mid-extraction           | `true`              |
| `OCR_TOO_MANY_PAGES`               | PDF exceeds configured `ocr_max_pages`               | `false`             |
| `OCR_EMPTY_TEXT` (warn)            | OCR ran cleanly but produced zero characters         | —                   |
| `OCR_LOW_CONFIDENCE` (warn)        | Avg confidence below `ocr_min_confidence_warn`       | —                   |
| `VISION_<provider-code>`           | Vision provider raised a VisionError                 | usually `true`      |
| `VISION_PDF_RASTERIZATION_FAILED`  | PyMuPDF first-page rasterization failed              | `false`             |
| `VISION_PDF_EMPTY`                 | Rasterization returned zero bytes                    | `false`             |
| `MULTIMODAL_ALL_PROVIDERS_FAILED`  | Both OCR and vision produced no usable signal        | `false`             |
| `MULTIMODAL_RETRIEVAL_FAILED`      | Retriever raised after OCR / vision succeeded        | `true`              |
| `MULTIMODAL_GENERATION_FAILED`     | Generator raised after retrieval succeeded           | `true`              |

### Example: success

Happy-path MULTIMODAL job, all six stages `ok`, `finalStatus=ok`:

```json
{
  "schemaVersion": "trace.v1",
  "capability": "MULTIMODAL",
  "inputKind": "image",
  "finalStatus": "ok",
  "stages": [
    {
      "stage": "classify", "provider": null, "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 0.12,
      "details": {"mimeType": "image/png", "filename": "invoice.png", "kind": "image", "hasQuestion": true, "sizeBytes": 4096}
    },
    {
      "stage": "ocr", "provider": "tesseract-5.3.3", "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 42.1,
      "details": {"pageCount": 1, "textLength": 28, "avgConfidence": 91.0, "engineName": "tesseract-5.3.3"}
    },
    {
      "stage": "vision", "provider": "heuristic-vision-v1", "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 3.4,
      "details": {"pageNumber": 1, "captionPreview": "A portrait light-toned image", "latencyMs": 2.8, "detailCount": 5}
    },
    {
      "stage": "fusion", "provider": null, "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 0.3,
      "details": {"sources": ["user_question", "ocr_text", "vision_description"], "retrievalQueryLength": 24, "fusedContextLength": 410, "fusionWarnings": 0}
    },
    {
      "stage": "retrieve", "provider": "Retriever", "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 11.7,
      "details": {"hitCount": 5, "indexVersion": "v-1776253724", "embeddingModel": "sentence-transformers/all-MiniLM-L6-v2", "topK": 5}
    },
    {
      "stage": "generate", "provider": "extractive-generator-1", "status": "ok",
      "code": null, "message": null, "retryable": null,
      "fallbackUsed": false, "durationMs": 2.0,
      "details": {"answerLength": 620, "chunkCount": 6}
    }
  ],
  "warnings": [],
  "summary": "classify:ok(0ms) ocr:ok(42ms) vision:ok(3ms) fusion:ok(0ms) retrieve:ok(12ms) generate:ok(2ms)"
}
```

### Example: partial fallback (OCR ok, vision fail)

The vision provider timed out; OCR and everything downstream still
completed. `finalStatus=partial` tells consumers "completed, but with
caveats worth surfacing":

```json
{
  "schemaVersion": "trace.v1",
  "capability": "MULTIMODAL",
  "inputKind": "image",
  "finalStatus": "partial",
  "stages": [
    {"stage": "classify", "provider": null, "status": "ok", "code": null, "message": null, "retryable": null, "fallbackUsed": false, "durationMs": 0.1, "details": {}},
    {"stage": "ocr", "provider": "tesseract-5.3.3", "status": "ok", "code": null, "message": null, "retryable": null, "fallbackUsed": false, "durationMs": 41.2, "details": {"pageCount": 1, "textLength": 44}},
    {"stage": "vision", "provider": "fake-vision-1.0", "status": "fail", "code": "VISION_VLM_TIMEOUT", "message": "provider timed out after 3s", "retryable": true, "fallbackUsed": true, "durationMs": 3002.0, "details": {}},
    {"stage": "fusion", "provider": null, "status": "ok", "code": null, "message": null, "retryable": null, "fallbackUsed": false, "durationMs": 0.2, "details": {"sources": ["ocr_text"], "retrievalQueryLength": 44, "fusedContextLength": 380}},
    {"stage": "retrieve", "provider": "Retriever", "status": "ok", "code": null, "message": null, "retryable": null, "fallbackUsed": false, "durationMs": 9.8, "details": {"hitCount": 5}},
    {"stage": "generate", "provider": "extractive-generator-1", "status": "ok", "code": null, "message": null, "retryable": null, "fallbackUsed": false, "durationMs": 1.9, "details": {"answerLength": 540}}
  ],
  "warnings": [],
  "summary": "classify:ok(0ms) ocr:ok(41ms) vision:fail(VISION_VLM_TIMEOUT,3002ms,fallback) fusion:ok(0ms) retrieve:ok(10ms) generate:ok(2ms)"
}
```

### Example: terminal failure (both providers failed)

Both OCR and vision raised. No artifacts are produced; the worker
returns `MULTIMODAL_ALL_PROVIDERS_FAILED` with the stage-flow summary
embedded in the error message so operators see the full picture from
just `GET /api/v1/jobs/{id}`:

```
errorCode    = "MULTIMODAL_ALL_PROVIDERS_FAILED"
errorMessage = "Multimodal pipeline could not extract any signal from the input — OCR returned no text AND the vision provider returned no description. Upstream diagnostics: ocr stage failed (IMAGE_DECODE_FAILED): corrupt png | trace: classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms,fallback) vision:fail(VISION_VLM_TIMEOUT,5ms,fallback) fusion:skipped retrieve:skipped generate:skipped"
```

When `emit_trace=true` is set, the same stage list above is also
downloadable as a `MULTIMODAL_TRACE` artifact — but the default gate
stays off to preserve the documented 4-artifact MULTIMODAL response.

### Example: retrieval failure after OCR / vision succeeded

A classic "downstream broke after upstream worked" scenario. The new
`MULTIMODAL_RETRIEVAL_FAILED` code surfaces this cleanly with earlier
success context preserved:

```
errorCode    = "MULTIMODAL_RETRIEVAL_FAILED"
errorMessage = "Retrieval stage failed after OCR / vision already produced usable signal. Upstream error: RuntimeError: FAISS search blew up: index corrupt | trace: classify:ok(0ms) ocr:ok(3ms) vision:ok(2ms) fusion:ok(0ms) retrieve:fail(MULTIMODAL_RETRIEVAL_FAILED,12ms) generate:skipped"
```

Operators reading this can immediately diagnose:

1. OCR and vision are healthy (`ocr:ok`, `vision:ok`).
2. The FAISS index is the actual problem (`retrieve:fail` plus the
   `FAISS search blew up: index corrupt` cause line).
3. Generation was never attempted (`generate:skipped`).

The analogous `MULTIMODAL_GENERATION_FAILED` code covers the rare
case where retrieval completed but the generator raised.

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
    │   ├── registry.py        ← name → instance map + shared-component cache
    │   ├── mock_processor.py  ← phase 1 echo capability
    │   ├── rag/               ← phase 2: FAISS retriever + extractive generator
    │   ├── ocr/               ← phase 2: Tesseract + PyMuPDF provider
    │   └── multimodal/        ← phase 2 v1: OCR + vision + fusion + text RAG
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

| Concern                                   | Phase |
|-------------------------------------------|-------|
| Real OCR engine                           | 2 (shipped) |
| FAISS-based RAG capability                | 2 (shipped) |
| Multimodal v1 (OCR + vision + text RAG)   | 2 (shipped) |
| AUTO capability (single-pass dispatcher)  | 3 (shipped) |
| AGENT capability (loop + critic + retry)  | 6     |
| ~~True multimodal retrieval (image embeddings, cross-modal search)~~ | shipped (CLIP + RRF, opt-in) |
| Real VLM provider (BLIP-2 / Claude Vision / GPT-4V / Gemini) | 3+    |
| Multi-page vision captioning for PDFs     | 3+    |
| Multimodal automated eval harness         | 3+    |
| ~~MinIO / S3 storage adapter~~            | shipped (backend=s3, AWS SDK v2) |
| ~~Auth on `/api/internal/*`~~             | shipped (shared-secret header) |
| Retry orchestration                       | 2     |
| Per-capability Redis lanes                | 2     |
| Kubernetes manifests                      | 2+    |
| Frontend beyond a test form               | 2+    |

See `docs/local-run.md` for how to actually run it, and
`docs/api-summary.md` for the concrete endpoint contracts.
