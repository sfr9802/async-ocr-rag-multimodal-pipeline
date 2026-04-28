# API 요약

플랫폼이 노출하는 모든 HTTP 엔드포인트의 구체적 모양. 모든 경로는
`http://localhost:8080` 기준입니다.

Phase 1 에서 파이프라인 (MOCK capability) 을 확립했습니다. Phase 2 는
엔드포인트 모양을 바꾸지 않고 `RAG` capability 를 추가합니다 — job 제출,
상태, 결과, artifact 다운로드 엔드포인트는 모두 동일합니다. 요청 본문의
`capability` 값과 결과의 artifact `type` 값만 달라집니다.

## Public (클라이언트용)

### `POST /api/v1/jobs`

텍스트 job 제출.

요청 (JSON):
```json
{ "capability": "MOCK", "text": "hello" }
```

또는 text-RAG capability (phase 2) 실행:
```json
{ "capability": "RAG", "text": "which anime is about an old fisherman feeding stray harbor cats" }
```

허용되는 capability 값:

| Capability   | 상태       | 출력 artifact 타입                                                              |
|--------------|------------|---------------------------------------------------------------------------------|
| `MOCK`       | phase 1    | `FINAL_RESPONSE` (JSON echo)                                                    |
| `RAG`        | phase 2    | `RETRIEVAL_RESULT` + `FINAL_RESPONSE`                                           |
| `OCR`        | phase 2    | `OCR_TEXT` + `OCR_RESULT`                                                       |
| `MULTIMODAL` | phase 2 v1 | `OCR_TEXT` + `VISION_RESULT` + `RETRIEVAL_RESULT` + `FINAL_RESPONSE` (+ 선택적 `MULTIMODAL_TRACE`) |
| `AUTO`       | phase 3    | `AGENT_DECISION` + 디스패치된 서브-capability 가 emit 한 것 (RAG / OCR / MULTIMODAL artifact, 또는 clarify / direct_answer 의 인라인 `FINAL_RESPONSE`) |

RAG 또는 MULTIMODAL job 을 제출하기 전에 worker 측 FAISS 인덱스를
`python -m scripts.build_rag_index --fixture` 로 빌드해야 하고 worker
재시작이 필요. OCR 또는 MULTIMODAL job 을 제출하기 전에 Tesseract 와
PyMuPDF 가 worker 에서 사용 가능해야 함. `docs/local-run.md` 참조.

**MULTIMODAL v1 정의.** 여기서 "Multimodal" 은 INPUT_FILE
(PNG/JPEG/PDF) 이 OCR + visual-description provider 를 통과하고, 두
신호가 retrieval query + grounding context 로 융합되어 기존 text-RAG
retriever 와 generator 를 먹인다는 의미입니다. 이는 명시적으로
진정한 multimodal retrieval 이 **아님** — 보류된 항목 전체 목록은
`docs/architecture.md` 의 "Multimodal v1 한계" 참조.

응답 `202 Accepted`:
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

파일 job 제출 (같은 경로, 다른 content type).

폼 필드:
- `capability` — 예: `OCR`, `MULTIMODAL`
- `file` — 바이너리 업로드
- `text` — **선택적** 사용자 질문/프롬프트. 존재하면 core-api 가 이를
  `INPUT_FILE` 옆에 두 번째 `INPUT_TEXT` artifact 로 stage. `MULTIMODAL`
  capability 는 fusion 단계에서 사용자 질문으로 사용. `OCR` 와 `MOCK`
  은 무시 — 둘은 각각 `INPUT_FILE` / `INPUT_TEXT` 중 하나만 사용.
  필드를 생략하면 multimodal 이전 계약과 동일한 단일-artifact job 이
  되므로 기존 OCR 스모크 테스트는 영향받지 않음.

예시 — 이미지 + 사용자 질문이 있는 MULTIMODAL job:
```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/invoice.png" \
  -F "text=what is the total amount on this invoice?"
```

응답 `202 Accepted`: JSON 변형과 같은 모양. `text` 가 있으면 응답의
`inputs` 배열에 두 항목 (`INPUT_FILE` + `INPUT_TEXT`) 이, 없으면
`INPUT_FILE` 만 있음.

### 제출 계약 (capability / 입력 매트릭스)

두 job-생성 엔드포인트는 모두 바이트가 스토리지에 stage 되기 전에 그리고
어떤 job row 가 영속화되기 전에 제출된 `capability` 값을 요청 모양에
대해 검증합니다. 잘못된 제출은 안정적인 에러 코드로 거부되고 비동기
파이프라인은 깨끗한 상태로 남겨집니다 — `artifact` row 없음, `job`
row 없음, Redis 디스패치 없음.

| Capability    | 엔드포인트 | `text` 필드                  | `file` 필드                           | 허용되는 파일 타입         |
|---------------|------------|------------------------------|---------------------------------------|----------------------------|
| `MOCK`        | JSON       | 필수 (빈 값 가능)            | —                                     | —                          |
| `MOCK`        | multipart  | 선택                         | **필수, non-empty**                   | 모두                       |
| `RAG`         | JSON       | **필수, non-blank**          | —                                     | —                          |
| `RAG`         | multipart  | **필수, non-blank**          | 필수, non-empty (worker 가 무시)      | 모두                       |
| `OCR`         | JSON       | — (거부됨: FILE_REQUIRED)    | **multipart 로 필수**                 | PNG, JPEG, PDF             |
| `OCR`         | multipart  | 선택 (worker 가 무시)        | **필수, non-empty**                   | PNG, JPEG, PDF             |
| `MULTIMODAL`  | JSON       | — (거부됨: FILE_REQUIRED)    | **multipart 로 필수**                 | PNG, JPEG, PDF             |
| `MULTIMODAL`  | multipart  | 선택 (사용자 질문)           | **필수, non-empty**                   | PNG, JPEG, PDF             |
| `AUTO`        | JSON       | 필수 (blank 가능)            | —                                     | —                          |
| `AUTO`        | multipart  | 선택 (blank 가능)            | 선택, non-empty; text/file 중 최소 하나 필수 | 있을 경우 PNG, JPEG, PDF |

API 경계에서 강제되는 검증 규칙:

1. **Capability 는 필수** 이며 `JobCapability` enum (`MOCK`, `RAG`,
   `OCR`, `MULTIMODAL`) 과 일치해야 합니다. Blank → `CAPABILITY_REQUIRED`;
   알 수 없는 값 → `UNKNOWN_CAPABILITY`.
2. **JSON 엔드포인트의 파일 기반 capability 는 거부됨** —
   `FILE_REQUIRED`. OCR / MULTIMODAL 은 multipart 엔드포인트를
   사용해야 함.
3. **RAG 는 두 엔드포인트 모두에서 non-blank `text` 필드 필요** —
   누락, 빈 문자열, 공백만 있으면 `TEXT_REQUIRED` 로 거부.
4. **multipart 엔드포인트는 모든 capability 에서 항상 실제 파일 필요.**
   파일 누락 → `FILE_REQUIRED`. 0바이트 파일 → `FILE_EMPTY`.
5. **OCR 와 MULTIMODAL 파일은 PNG, JPEG, PDF 이어야 함.** 검증기는
   content-type 헤더 또는 파일명 확장자 중 하나만 매치되면 통과
   (실제 클라이언트는 알려진 파일 타입에 대해 가끔
   `application/octet-stream` 을 보내거나 파일명을 완전히 빠뜨림).
   허용 집합 밖의 어떤 것도 → `UNSUPPORTED_FILE_TYPE`.
6. **MULTIMODAL text 는 선택적.** Blank 또는 누락된 `text` 필드는
   허용; worker 의 fusion 레이어가 중립적인 기본 retrieval query 로
   폴백.
7. **MOCK 은 phase-1 호환성 유지.** multipart 의 MOCK 은 어떤 파일
   타입도 허용 (타입 게이트 없음); JSON 엔드포인트의 MOCK 은 빈
   (그러나 non-null) `text` 필드 허용.
8. **AUTO 는 multipart 에서 text 또는 file 중 최소 하나 필요.**
   worker 측 라우터는 text 도 file 도 없는 job 을 라우팅할 수 없음 —
   `AUTO_NO_INPUT` 으로 빠르게 거부. 제공된 파일은 여전히 PNG/JPEG/PDF
   여야 함 (MULTIMODAL 과 같은 규칙). JSON 엔드포인트의 AUTO 는 MOCK
   과 미러링: text 가 (non-null 로) 있어야 하지만 blank 가능, file
   필드 없음. 출력은 항상 첫 번째로 `AGENT_DECISION` artifact 포함.

### 제출 예시

유효 — MOCK 텍스트 job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"MOCK","text":"hello platform"}'
# → 202 Accepted, jobId=..., status=QUEUED
```

유효 — RAG 텍스트 job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":"who feeds the harbor cats?"}'
# → 202 Accepted
```

유효 — OCR 파일 job:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/receipt.png"
# → 202 Accepted
```

유효 — 파일 + 선택적 text 가 있는 MULTIMODAL:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/invoice.png" \
  -F "text=what is the total amount on this invoice?"
# → 202 Accepted
```

유효 — 파일만 있는 MULTIMODAL (질문 없음):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/diagram.pdf"
# → 202 Accepted
```

**무효** — text 없는 RAG:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":""}'
# → 400 Bad Request
# { "code": "TEXT_REQUIRED",
#   "message": "RAG jobs require a non-blank 'text' field in the JSON body." }
```

**무효** — JSON 엔드포인트의 OCR (파일 job 에 잘못된 엔드포인트):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"OCR","text":"please extract"}'
# → 400 Bad Request
# { "code": "FILE_REQUIRED",
#   "message": "OCR jobs require a file upload. Use the multipart endpoint ..." }
```

**무효** — 파일 없는 MULTIMODAL multipart:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "text=what is this?"
# → 400 Bad Request
# { "code": "FILE_REQUIRED",
#   "message": "MULTIMODAL job requires a 'file' form field on the multipart endpoint." }
```

**무효** — 0바이트 파일의 OCR:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/dev/null;filename=blank.png;type=image/png"
# → 400 Bad Request
# { "code": "FILE_EMPTY",
#   "message": "Uploaded file is empty (0 bytes) ..." }
```

**무효** — 지원되지 않는 파일 타입의 OCR:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/cat.gif"
# → 400 Bad Request
# { "code": "UNSUPPORTED_FILE_TYPE",
#   "message": "Unsupported file type for OCR. Received contentType='image/gif' filename='cat.gif'. Supported types: PNG, JPEG, PDF ..." }
```

**무효** — 알 수 없는 capability:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"SUMMARIZE","text":"hi"}'
# → 400 Bad Request
# { "code": "UNKNOWN_CAPABILITY",
#   "message": "Unknown capability: SUMMARIZE. Accepted values: MOCK, RAG, OCR, MULTIMODAL, AUTO." }
```

유효 — AUTO 텍스트만 있는 job (text 가 충분히 길면 라우터가 `rag` emit):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"AUTO","text":"which anime features harbor cats?"}'
# → 202 Accepted, jobId=..., status=QUEUED
# 출력: AGENT_DECISION + RETRIEVAL_RESULT + FINAL_RESPONSE
```

유효 — text + PDF 가 있는 AUTO multipart (라우터가 `multimodal` emit):

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=AUTO" \
  -F "file=@/path/to/invoice.pdf" \
  -F "text=what is the total amount"
# → 202 Accepted
# 출력: AGENT_DECISION + OCR_TEXT + VISION_RESULT + RETRIEVAL_RESULT + FINAL_RESPONSE
```

**무효** — text 도 file 도 없는 AUTO multipart:

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=AUTO"
# → 400 Bad Request
# { "code": "AUTO_NO_INPUT",
#   "message": "AUTO jobs on the multipart endpoint require AT LEAST ONE of ..." }
```

### `GET /api/v1/jobs/{jobId}`

상태 뷰. 반환:
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

해당 id 의 job 이 없으면 404.

### `GET /api/v1/jobs/{jobId}/result`

모든 artifact 가 포함된 풀 결과 뷰.
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

RAG job 의 경우 `outputs` 에 두 항목 포함:

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

`RETRIEVAL_RESULT` artifact 는 다음 모양의 JSON 으로 다운로드됩니다
(`RagCapability._retrieval_payload` 가 생성):

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

`FINAL_RESPONSE` artifact 는 extractive generator 가 만든 markdown 으로
다운로드됩니다: 짧은 grounded 답변 + 인용이 포함된 보조 passage 의 번호
매겨진 리스트.

MULTIMODAL job 의 경우 `outputs` 에 네 항목 포함
(`AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true` 면 다섯):

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

`VISION_RESULT` JSON 은 다음 모양 (`MultimodalCapability._vision_result_json`
이 생성):

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

Vision stage 가 실패하면 `available` 이 `false`, `provider` /
`caption` / `details` / `pageNumber` / `latencyMs` 는 null, 그리고
`warnings` 가 이유를 기록.

MULTIMODAL `RETRIEVAL_RESULT` 스키마는 RAG capability 와 동일 — 다운스트림
컨슈머는 동일하게 처리할 수 있고, 유일한 차이는 상태 엔드포인트에서의
job `capability` 값.

AUTO job 의 경우 `outputs` 는 항상 라우팅 결정을 기록한 `AGENT_DECISION`
artifact 로 시작하고, 디스패치된 서브-capability 가 emit 한 artifact 들이
뒤따릅니다. 모양은 라우터가 어떤 action 을 선택했는지에 따라 달라짐:

```json
{
  "jobId": "...",
  "capability": "AUTO",
  "status": "SUCCEEDED",
  "outputs": [
    { "id": "...", "type": "AGENT_DECISION",   "contentType": "application/json",             "accessUrl": "..." },
    { "id": "...", "type": "OCR_TEXT",         "contentType": "text/plain; charset=utf-8",    "accessUrl": "..." },
    { "id": "...", "type": "VISION_RESULT",    "contentType": "application/json",             "accessUrl": "..." },
    { "id": "...", "type": "RETRIEVAL_RESULT", "contentType": "application/json",             "accessUrl": "..." },
    { "id": "...", "type": "FINAL_RESPONSE",   "contentType": "text/markdown; charset=utf-8", "accessUrl": "..." }
  ]
}
```

`AGENT_DECISION` artifact 는 다음 모양 (`AutoCapability._serialize_decision`
이 생성):

```json
{
  "action": "multimodal",
  "reason": "text (32 chars) + supported file (application/pdf, 20480B) -> multimodal",
  "confidence": 0.95,
  "routerName": "rule",
  "parsedQuery": null
}
```

`action` 은 `rag`, `ocr`, `multimodal`, `direct_answer`, `clarify` 중
하나. `routerName` 은 결정적 라우터일 때 `rule`, 깨끗한 LLM 결정에서는
`llm-<backend>`, LLM 라우터가 룰 경로로 강등된 경우 (낮은 confidence,
schema 위반, provider 실패) `llm-<backend>-fallback-rule`. `parsedQuery`
는 LLM 라우터가 `rag` 결정에 첨부했을 때 RAG `parsedQuery` 구조와 같은
모양; 그 외에는 `null`.

Clarify 와 direct_answer action 은 서브-capability 호출 대신 인라인으로
`FINAL_RESPONSE` 를 emit 하므로, outputs 리스트는 `AGENT_DECISION +
FINAL_RESPONSE` 만.

### `GET /api/v1/artifacts/{id}/content`

Artifact 바이트를 스트리밍. Content-Type 과 Content-Length 는 저장된
메타데이터에서 설정. 알 수 없으면 404.

## Internal (worker 용)

이 엔드포인트들은 `/api/internal/*` 아래에 있습니다. Phase 1 은 이를
인증 없이 두지만, 프로덕션 배포는 공유 시크릿 또는 mTLS 뒤에 게이팅해야
합니다.

### `POST /api/internal/jobs/claim`

Worker 가 job 의 소유권을 가져가겠다고 요청.

요청:
```json
{ "jobId": "...", "workerClaimToken": "worker-local-1", "attemptNo": 1 }
```

응답 (granted):
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

응답 (denied):
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

거부 사유: `JOB_NOT_FOUND`, `JOB_TERMINAL`, `ALREADY_CLAIMED`, `CLAIM_RACE`.

### `POST /api/internal/jobs/callback`

Worker 가 종단 결과를 보고.

요청:
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

응답:
```json
{ "applied": true, "duplicate": false, "currentStatus": "SUCCEEDED" }
```

같은 job 에 같은 `callbackId` 가 두 번 도착하면, 두 번째 호출은
`{ "applied": false, "duplicate": true }` 를 반환하고 job 상태는
변경되지 않은 채로 남습니다.

### `POST /api/internal/artifacts` (multipart)

Phase 1 의 presigned PUT URL 대체. Worker 가 결과 바이트를 core-api 에
업로드하고 후속 callback 에 echo 할 storage URI 를 받음.

폼 필드:
- `jobId` — 이 출력이 속한 job
- `type` — 예: `FINAL_RESPONSE`
- `file` — 바이너리 출력

응답:
```json
{
  "storageUri": "local://...",
  "sizeBytes": 187,
  "checksumSha256": "..."
}
```

**중요**: 이 엔드포인트는 **바이트만** 씁니다. `artifact` row 를 만들지
않습니다. row 는 후속 `callback` 에서 만들어지고, 반환된 `storageUri`
를 참조합니다. 이는 진짜 presigned-upload 흐름의 의미와 일치하며 (업로드는
object 스토리지에 직접 도착하고 DB 는 callback 으로부터만 그것에 대해
들음), 그렇지 않으면 callback 이 같은 바이트를 echo 할 때 발생할 더블
쓰기를 방지합니다.

## 에러 envelope

모든 4xx 와 5xx 응답은 단일 모양을 따릅니다:
```json
{ "code": "TEXT_REQUIRED", "message": "RAG jobs require a non-blank 'text' field in the JSON body." }
```

### 제출 에러 코드 (`POST /api/v1/jobs`)

이 코드들은 `JobSubmissionValidator` 가 API 경계에서 어떤 스토리지 쓰기
또는 job-row 영속화 전에 raise. 이 코드 중 하나의 400 응답은 비동기
파이프라인이 건드려지지 않았음을 보장합니다 — `artifact` row 가 만들어지지
않았고, `job` row 가 삽입되지 않았고, Redis 디스패치가 발행되지 않았음.
클라이언트는 수정된 요청으로 안전하게 재시도할 수 있습니다.

| 코드                       | HTTP | 의미                                                                                    |
|----------------------------|------|-----------------------------------------------------------------------------------------|
| `CAPABILITY_REQUIRED`      | 400  | `capability` 필드가 누락 / blank / 공백만.                                              |
| `UNKNOWN_CAPABILITY`       | 400  | `capability` 값이 `MOCK`, `RAG`, `OCR`, `MULTIMODAL`, `AUTO` 중 하나가 아님.            |
| `TEXT_REQUIRED`            | 400  | Capability 가 non-blank `text` 필드를 요구하지만 제공되지 않음 (RAG; MOCK-JSON / AUTO-JSON 의 null). |
| `FILE_REQUIRED`            | 400  | Capability 가 파일 업로드를 요구하지만 제공되지 않았거나, capability 가 잘못된 엔드포인트로 제출됨. |
| `FILE_EMPTY`               | 400  | `file` 폼 필드는 있었지만 0 바이트.                                                     |
| `UNSUPPORTED_FILE_TYPE`    | 400  | 파일의 content-type 과 파일명 확장자가 모두 capability 의 허용 집합 밖 (OCR / MULTIMODAL / AUTO 는 파일 제공 시 PNG, JPEG, PDF 필요). |
| `AUTO_NO_INPUT`            | 400  | AUTO multipart job 이 non-blank `text` 도 non-empty `file` 도 제공 안 함.               |

### 기타 알려진 에러 코드

| 코드                          | 출처                                                       |
|-------------------------------|------------------------------------------------------------|
| `INVALID_ARGUMENT`            | 애플리케이션 레이어의 일반적인 `IllegalArgumentException`. |
| `VALIDATION_ERROR`            | DTO 필드의 Jakarta Validation 제약 위반.                   |
| `INVALID_STATE_TRANSITION`    | 도메인 상태 머신이 전이를 거부.                             |
| `CONFLICT`                    | 애플리케이션 레벨 충돌 (멱등성, claim race).                |
| `INTERNAL_ERROR`              | 처리되지 않은 예외에 대한 폴백. 500 상태.                  |
