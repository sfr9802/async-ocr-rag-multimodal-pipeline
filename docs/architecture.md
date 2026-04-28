# 아키텍처 — AI 처리 플랫폼 (phase 1 / 1.1)

## 목적

범용 AI job 처리 플랫폼. 클라이언트가 작업 (텍스트 또는 파일) 을 제출하면,
플랫폼은 이를 장수명 worker에 스케줄링하고, worker는 capability (OCR /
RAG / multimodal / mock) 를 실행하며, 결과는 클라이언트가 나중에 가져갈
수 있는 artifact 로 반환됩니다.

Phase 1 은 **스켈레톤만** 제공합니다: job 생명주기, artifact 모델,
claim / callback 계약, Redis 기반 디스패치, 로컬 파일시스템 스토리지
백엔드, 그리고 입력을 그대로 echo 하는 단일 `MOCK` capability. 진짜 엔진은
phase 2+ 에서 들어옵니다.

## 타깃 스택 (phase 2)

- Java 21, Spring Boot **4.0.3**
- Python 3.12 (3.13 도 동작)
- PostgreSQL 18
- Redis (latest)
- Maven 3.9+
- **FAISS (faiss-cpu)**, **sentence-transformers**, **psycopg2** (phase 2 RAG capability)

Phase 1.1 에서 Spring Boot 버전을 4.0.3 에 맞추고 비동기 스켈레톤을
검증했습니다. Phase 2 는 그 스켈레톤 위에 첫 진짜 capability 를 추가하면서도
파이프라인 구조는 건드리지 않습니다.

## Phase 2 추가 사항: text-RAG capability

Phase 2 는 "MOCK capability 만 있다" 를 "MOCK capability 와 진짜 text-RAG
capability 가 둘 다 있고, 둘이 같은 `Capability` 인터페이스 뒤에 산다" 로
바꿉니다.

RAG capability 는 두 개의 논리적으로 구분되는 경로를 갖습니다:

- **Indexing 경로** (오프라인 / 1회성) — `scripts/build_rag_index.py` 가
  JSONL 데이터셋을 읽어 문서를 chunking 하고, sentence-transformers 로
  각 chunk 를 임베딩한 뒤, FAISS `IndexFlatIP` 를 빌드하고, 문서 + chunk
  메타데이터를 PostgreSQL 의 `ragmeta` 스키마에 적재합니다. 이 작업은
  worker 프로세스 안이 아니라 CLI 로 실행됩니다.
- **Serving 경로** (온라인) — worker 가 시작 시 FAISS 인덱스와
  sentence-transformers 모델을 메모리에 로드하고 DB ping 후 RAG
  capability 를 등록합니다. job 단위로, task runner 는 사용자 query 를
  retriever → generation provider → 출력 artifact (`RETRIEVAL_RESULT`
  JSON + `FINAL_RESPONSE` 마크다운) 흐름으로 통과시킵니다.

두 경로는 같은 chunker, embedding provider, FAISS 래퍼, metadata store
클래스를 공유하며, 다만 사용 순서가 다를 뿐입니다.

기본 임베딩 모델은 다국어 (BAAI/bge-m3, 1024-dim) 로, 한국어와 영어
검색을 단일 인덱스로 커버합니다. Generation provider 는 선택 가능:
`extractive` (기본, 결정적 휴리스틱) 또는 `claude` (Anthropic API
경유 Claude LLM, API 실패 시 자동 extractive 폴백). Claude generation 을
쓰려면 `AIPIPELINE_WORKER_RAG_GENERATOR=claude` 를 설정하세요.

### 스토리지 분리 (의도적이며, DB 레벨에서 강제됨)

| 스토어                              | 용도                                          | Phase |
|-------------------------------------|-----------------------------------------------|-------|
| `aipipeline` 스키마 (PostgreSQL)    | 파이프라인 상태: job, artifact                | 1     |
| `ragmeta` 스키마 (PostgreSQL)       | RAG 메타데이터: documents, chunks, index_builds | 2   |
| 디스크 상의 FAISS 인덱스 파일       | 벡터 + FAISS 메타데이터                       | 2     |
| 로컬 파일시스템 (`local-storage/`)  | artifact 콘텐츠 blob                          | 1     |

MongoDB 사용 안 함. Vector DB 서비스 없음. 두 PostgreSQL 스키마는 같은
데이터베이스에 있지만 의도적인 namespace 경계의 양쪽에 있습니다 — Spring
의 JPA 는 ragmeta 를 매핑하지 않으며, worker 의 psycopg2 호출은
`aipipeline` 을 건드리지 않습니다.

### RAG capability 파일 구조

```
ai-worker/app/capabilities/rag/
├── capability.py       Capability 인터페이스 구현 — retrieve + generate 오케스트레이션
├── chunker.py          greedy_chunk + window_by_chars (port/rag 에서 이식)
├── embeddings.py       EmbeddingProvider 추상 + SentenceTransformerEmbedder + HashingEmbedder (테스트 폴백)
├── generation.py       GenerationProvider 추상 + ExtractiveGenerator (grounded, cited, non-mock)
├── faiss_index.py      build.json 사이드카가 붙은 IndexFlatIP 의 얇은 FaissIndex 래퍼
├── metadata_store.py   ragmeta.documents / chunks / index_builds 용 psycopg2 DAO
├── ingest.py           Ingestion 서비스 (JSONL -> chunks -> vectors -> FAISS + DB)
└── retriever.py        Retriever (query text -> 임베딩 -> FAISS search -> DB lookup)
```

### End-to-end RAG 흐름 (job 단위)

1. 클라이언트가 `POST /api/v1/jobs` 에 `{"capability": "RAG", "text": "..."}` 전송.
2. core-api 가 텍스트를 `INPUT_TEXT` artifact 로 stage 하고 Redis 로 디스패치 (변경 없음).
3. Worker 가 메시지를 BRPOP 으로 받아 job 을 claim 하고 INPUT_TEXT URI 를 받음 (변경 없음).
4. Task runner 가 artifact 를 `MOCK` 대신 `RAG` capability 에 넘김.
5. `RagCapability.run()` 동작:
   - INPUT_TEXT 바이트를 query 문자열로 디코딩
   - `Retriever.retrieve(query)` 호출:
     - `EmbeddingProvider.embed_queries` 로 query 임베딩
     - FAISS `IndexFlatIP.search(top_k)` 실행
     - 반환된 row id 를 `ragmeta.chunks` JOIN `ragmeta.documents` 로
       chunk 에 매핑 (job 당 SQL 1쿼리)
   - `GenerationProvider.generate(query, chunks)` 호출하여 markdown 생성
6. Capability 가 두 개의 출력 artifact 를 발행:
   - `RETRIEVAL_RESULT` — 전체 retrieval 리포트 JSON (query, top-k, index_version, embedding_model, score 와 텍스트가 포함된 ranked chunk)
   - `FINAL_RESPONSE` — grounded markdown 답변
7. Task runner 가 두 artifact 를 업로드하고 SUCCEEDED callback 을 게시 (변경 없음).

파이프라인 (1-3, 6-7 단계) 은 phase 1 과 동일. 4-5 단계만 새로 추가됨.

## Phase 2 추가 사항: OCR capability

Phase 2 는 첫 실용 OCR capability 도 추가합니다. MOCK / RAG 와 동일한
`Capability` 인터페이스와 동일한 worker 파이프라인을 공유합니다. OCR 은
의도적으로 **안정적인 artifact 생성**에 범위를 좁혔습니다 — multimodal
추론, 이미지 임베딩, VLM 답변 생성, OCR→RAG 체이닝은 이 phase 에서 다루지
않습니다. 계약은 다음과 같습니다:

```
INPUT_FILE  ──► OcrCapability ──►  OCR_TEXT   (plain UTF-8 text)
                              ──►  OCR_RESULT (JSON envelope)
```

`OCR_TEXT` artifact 는 나중에 RAG job 이 `INPUT_TEXT` 로 소비할 모양과
동일하게 설계되어 있습니다 — OCR 내부를 재작성하지 않고도 미래 phase 에서
OCR→RAG 체이닝이 자연스럽게 연결되게 하기 위함입니다.

### OCR capability 파일 구조

```
ai-worker/app/capabilities/ocr/
├── capability.py           OcrCapability — mime 디스패치, artifact envelope, warning 롤업
├── provider.py             OcrProvider 추상 + OcrPageResult / OcrDocumentResult / OcrError
└── tesseract_provider.py   TesseractOcrProvider — pytesseract + PyMuPDF, 지연 import
```

Provider seam 은 의도적으로 얇습니다:

- PNG/JPEG 입력용 `ocr_image(image_bytes) -> OcrPageResult`
- PDF 용 `ocr_pdf(pdf_bytes) -> OcrDocumentResult` (provider 가 페이지
  순회를 소유하므로, born-digital PDF 의 경우 rasterize 전에 native text
  layer 로 폴백할 수 있음)

Tesseract 를 EasyOCR, PaddleOCR, 또는 클라우드 API 로 바꾸려면 새
`OcrProvider` 구현 파일 하나와 `registry._build_ocr_capability` 의
한 줄 수정만 필요 — 그 위쪽은 그대로 둡니다.

### End-to-end OCR 흐름 (job 단위)

1. 클라이언트가 `POST /api/v1/jobs` 에 `capability=OCR` 와 `file` 필드를
   포함한 multipart 로 전송. core-api 는 바이트를 storage port 로
   `INPUT_FILE` artifact 로 stage 하고 job 을 enqueue — **phase 1 과
   동일.**
2. Worker 가 BRPOP 으로 받아 job 을 claim, `INPUT_FILE` storage URI 를
   로컬 파일시스템 resolver 로 raw bytes 까지 해소 — **변경 없음.**
3. `TaskRunner` 가 raw bytes, content type, 그리고 (신규) storage URI
   끝의 `{uuid}-{filename}` 세그먼트에서 best-effort 로 복원한 `filename`
   을 담은 `CapabilityInputArtifact` 를 만듦.
4. `OcrCapability.run()` 동작:
   - 첫 번째 `INPUT_FILE` artifact 를 선택 (그 외는 모두
     `UNSUPPORTED_INPUT_TYPE` 으로 거부).
   - 입력을 content type → 파일명 확장자 → magic byte 순서로 분류해
     `(mime_type, "image" | "pdf")` 를 생성. 지원되지 않는 입력은
     `CapabilityError("UNSUPPORTED_INPUT_TYPE")` 발생.
   - `provider.ocr_image` 또는 `provider.ocr_pdf` 로 디스패치.
   - 페이지별 warning 을 모으고, 텍스트가 0 이거나 평균 신뢰도가 임계값
     이하인 경우 문서 단위 warning 을 추가, 모두를 envelope 에 롤업.
5. Capability 가 두 개의 출력 artifact 를 발행:
   - `OCR_TEXT` — 페이지별 텍스트를 문서 순서대로 이은 plain UTF-8.
   - `OCR_RESULT` — `filename`, `mimeType`, `kind`, `engineName`,
     `pageCount`, `textLength`, `avgConfidence`, 페이지별 롤업, 평탄한
     `warnings` 배열을 담은 JSON.
6. Task runner 가 두 artifact 를 업로드하고 SUCCEEDED callback 을 게시
   — **변경 없음.**

1, 2, 3 (단 `filename` 백필 제외), 6 단계는 phase 1 / RAG 와 동일.
4와 5 단계만 새 부분입니다.

### PDF 페이지 처리

`TesseractOcrProvider.ocr_pdf` 는 PyMuPDF 로 PDF 를 엽니다 (poppler 필요
없음, 외부 바이너리 없음, 인-프로세스로 처리) 그리고 페이지를 순서대로
순회합니다. 각 페이지에서 먼저 `page.get_text()` 로 native text layer 를
시도 — born-digital PDF 는 OCR 비용 0 으로 깨끗한 텍스트를 반환합니다.
text layer 가 비어있거나 8 글자 미만이면 그 페이지는 스캔으로 간주:
`page.get_pixmap(dpi=ocr_pdf_dpi)` 로 rasterize → PNG 바이트를 Tesseract
에 넘김 → `page N: no text layer, ran OCR at N dpi` 를 페이지별 warning
으로 기록. 문서 단위 `avgConfidence` 는 페이지별 신뢰도의 평균. text
layer 로 처리된 페이지는 신뢰도 `None` (측정할 OCR 이 없음) 이라, mixed
문서는 OCR 된 페이지들에 대해서만 평균이 계산됩니다.

하드 페이지 cap (`ocr_max_pages`, 기본 100) 이 수천 페이지짜리 스캔에서
runaway job 을 막아줍니다. cap 초과 시 `OCR_TOO_MANY_PAGES` 가 발생하며,
job 의 `errorCode` 에서 확인 가능합니다.

### OCR 레지스트리 회복력

RAG 와 마찬가지로 OCR 도 `build_default_registry` 에서 기회주의적으로
등록됩니다. `TesseractOcrProvider.ensure_ready()` 가 worker 시작 시
Tesseract 바이너리와 설정된 언어팩을 probe; 어떤 실패든 `OcrError` 를
발생시키고, 레지스트리가 이를 잡아 깨끗한 `OCR capability NOT registered
(...)` warning 으로 변환합니다. MOCK 과 RAG 는 영향 없음. 단위 테스트가
이 동작을 `tests/test_ocr_capability.py::test_ocr_failure_*` 에 고정해
둡니다.

## Phase 2 추가 사항: MULTIMODAL capability (v1)

Phase 2 는 이미지 / PDF 입력으로 파이프라인을 여는 v1 multimodal
capability 도 출시합니다. 이 단계의 "multimodal" 의 제품 수준 정의는
의도적으로 좁습니다:

> MULTIMODAL v1 은 INPUT_FILE (이미지 또는 PDF) 을 OCR + visual-description
> provider 로 통과시켜 두 신호를 융합한 retrieval query + grounding
> context 를 만들고, 이를 기존 text-RAG retriever 와 generator 에
> 공급한다는 의미입니다.

이건 진정한 multimodal retrieval 이 **아닙니다**. 전용 이미지 임베딩
인덱스도, 크로스모달 nearest-neighbour 검색도, VLM 생성도 없습니다.
이것들은 v1 에서 명시적인 non-goal 입니다 — 아래 "Multimodal v1 한계"
참조.

### MULTIMODAL capability 파일 구조

```
ai-worker/app/capabilities/multimodal/
├── capability.py          MultimodalCapability — 5단계 파이프라인 오케스트레이터
├── vision_provider.py     VisionDescriptionProvider 추상 + VisionDescriptionResult / VisionError
├── heuristic_vision.py    HeuristicVisionProvider — 결정적 Pillow 기반 폴백
└── fusion.py              build_fusion() + FusionResult — 구조화된 컨텍스트 빌더
```

### End-to-end MULTIMODAL 흐름 (job 단위)

```
INPUT_FILE  ──►  MultimodalCapability  ──►  OCR_TEXT         (plain UTF-8)
(+ 선택적                              ──►  VISION_RESULT    (JSON envelope)
 INPUT_TEXT                            ──►  RETRIEVAL_RESULT (RAG 와 같은 스키마)
 question)                             ──►  FINAL_RESPONSE   (grounded markdown)
                                       ──►  MULTIMODAL_TRACE (선택 JSON — emit_trace
                                                              뒤로 게이팅)
```

단계:

1. **Stage A — OCR 추출.** 단독 OCR capability 를 받쳐주는 동일한
   `OcrProvider` 를 재사용. 이미지 입력은 `ocr_image`, PDF 입력은
   `ocr_pdf` (born-digital text layer + 페이지별 rasterization 처리)
   를 통과. OCR 실패는 non-fatal warning 으로 강등됨 — capability 는
   vision-only 신호로 계속 진행.

2. **Stage B — vision description.** `VisionDescriptionProvider` 가
   파일에서 caption + 구조화된 details 를 생성. 이미지 입력은 raw
   bytes 가 그대로 provider 에 넘어감. PDF 입력은 capability 가 PDF
   1페이지를 PyMuPDF (이미 OCR 스택의 worker dep) 로 rasterize 한 뒤
   PNG 를 provider 에 넘김. v1 은 1페이지만 captioning — 모든 페이지
   captioning 은 이 단계의 제품 가치 대비 비용이 두 배가 되므로
   생략. Vision 실패도 non-fatal warning 으로 강등됨; 파이프라인은
   OCR-only 신호로 계속 진행.

3. **Stage C — fusion.** `build_fusion()` 이 선택적 사용자 질문, OCR
   텍스트, vision 결과를 받아 두 필드를 가진 결정적 `FusionResult` 를
   생성:

   - `retrieval_query` — 짧은 query 문자열 (`max_query_chars`, 기본
     400 으로 cap), 기존 `Retriever.retrieve(...)` 에 넘김. 사용자
     질문이 짧고 (whitespace 토큰 < 5) OCR 텍스트가 있으면, fusion
     helper 가 첫 OCR 키워드들로 query 를 보강해 임베더가 다룰 자료를
     줌.
   - `fused_context` — 길고 구조화된 markdown 블록. 항상 포함되는 세
     섹션: `### User question`, `### Extracted text (OCR)`, `### Visual
     description`. 이 블록이 최종 답변을 grounding 합니다.

   helper 는 의도적으로 순수하고 결정적입니다 — 같은 입력은 byte 단위로
   동일한 출력을 생성하므로, ops 는 여러 실행에 걸쳐 `MULTIMODAL_TRACE`
   artifact 를 diff 할 수 있고, 테스트는 어떤 확률성도 없이 fusion
   동작에 대해 추론할 수 있습니다.

4. **Stage D — retrieval.** 단독 RAG capability 와 정확히 같은
   `Retriever.retrieve(...)` 호출, 동일한 FAISS 인덱스 + 동일한
   임베딩 모델. v1 은 의도적으로 RAG retriever 를 재사용해서
   MULTIMODAL job 이 순수 RAG job 과 동일한 retrieval shape 을 보게 함.

5. **Stage E — generation.** Fused context 를 합성된 rank-0
   `RetrievedChunk` (doc_id `input:multimodal`, section
   `fused_context`, score 1.0) 로 retrieval 결과 앞에 prepend 한 뒤
   기존 `GenerationProvider.generate(...)` 호출. Extractive generator
   는 최상위 score chunk 에서 "짧은 답변" 문장을 고르는데, 이제
   구조상 그게 fused context 가 됨 — 따라서 최종 답변은 job 의 실제
   OCR + vision 신호에 grounding 되고, 검색된 chunk 들은 보조 passage
   로 따라옴.

1-3 단계 (core-api staging + dispatch) 와 6 단계 (callback + artifact
영속화) 는 phase 1 / RAG / OCR 와 동일. capability 내부만 새 부분.

### Provider seam

두 개의 추상 클래스가 v1 확장 지점을 고정합니다:

- `OcrProvider` — 단독 OCR capability 에서 이미 존재. MULTIMODAL 은
  레지스트리가 빌드한 인스턴스를 그대로 재사용.
- `VisionDescriptionProvider` — phase 2 신규. 구현은 단일
  `describe_image(image_bytes, mime_type, hint, page_number) ->
  VisionDescriptionResult` 메서드와 안정적인 `name` 프로퍼티를 노출해야
  함. v1 은 `HeuristicVisionProvider` 만 출시 — Pillow 의 `ImageStat`
  으로 결정적 구조 description (orientation, brightness, contrast,
  dominant channel) 을 emit. 진짜 VLM (BLIP-2, LLaVA, GPT-4V, Claude
  Vision, Gemini) 으로 교체하는 건 단일 파일 변경 +
  `registry._build_vision_provider` 의 한 줄 변경.

### MULTIMODAL 레지스트리 의존성

MULTIMODAL 은 **종속 capability** 입니다: OCR provider 와 RAG retriever
가 모두 성공적으로 등록되어 있어야 함. `build_default_registry` 가 이를
명시적으로 강제:

1. RAG 등록 시도 — 성공 시 `rag_registered = True`.
2. OCR 등록 시도 — 성공 시 `ocr_registered = True`.
3. MULTIMODAL 등록은 두 부모가 모두 등록된 경우에만 시도. 그렇지 않으면
   누락된 부모 이름을 담은 warning 과 함께 skip. MOCK / RAG / OCR 은
   어느 경우든 영향 없음.
4. 빌더 안의 MULTIMODAL 초기화 실패 (예: 모르는 vision provider, Pillow
   import 실패) 는 깨끗한 `MULTIMODAL capability NOT registered (...)`
   warning 을 만들고 MOCK / RAG / OCR 은 그대로 둠.

OCR provider + RAG retriever 는 한 번 빌드되어 모듈 레벨 dict
(`_shared_component_cache`) 에 캐시됩니다 — RAG + OCR + MULTIMODAL 을
모두 등록하는 worker 가 메모리에 두 개의 sentence-transformers 모델이나
시작 시 두 번의 Tesseract probe 로 끝나지 않게 하기 위함. 캐시는 모든
`build_default_registry` 호출 맨 위에서 비워지므로 반복 호출 (테스트,
인-프로세스 재시작) 은 fresh 인스턴스를 받습니다.

단위 테스트가 이 동작을
`tests/test_multimodal_capability.py::test_multimodal_failure_does_not_affect_mock_rag_or_ocr`
와 두 개의 parent-missing 변형에서 고정합니다.

### Multimodal v1 한계

아키텍처가 의도적으로 작은 데에는 이유가 있습니다. 다음은 v1 의 명시적
non-goal 이며 **구현되지 않습니다**:

- **진정한 multimodal retrieval 아님.** v1 은 OCR 텍스트 + visual
  description 으로 기존 **text** RAG retriever 를 먹입니다. multimodal
  vector DB 도, CLIP / BLIP-2 / VLM 임베딩 인덱스도, 크로스모달
  nearest-neighbour 검색도 없습니다.
- **전용 이미지 임베딩은 보류.** 이미지 입력은 자체 벡터를 만들지 않고,
  OCR 텍스트 + (지금은 휴리스틱인) vision description 으로만 기여.
- **기본 vision provider 는 claude-sonnet-4-6, heuristic 은
  오프라인/CI 폴백으로 남음.** Claude Vision 을 쓰려면
  `AIPIPELINE_WORKER_MULTIMODAL_VISION_PROVIDER=claude` 설정 + Anthropic
  API 키 제공. API 키 없는 환경에서는 `HeuristicVisionProvider`
  (Pillow 기반 구조 description) 가 여전히 기본.
- **v1 은 페이지별 caption 이 단일 페이지만.** 다중 페이지 PDF 는 모든
  페이지에 OCR 이 돌지만 vision stage 는 1페이지만 통과. 확장은 config
  knob 변경 + capability 안의 loop 추가.
- **Multimodal eval 은 RAG/OCR 와 같은 harness shape 으로 실행.**
  `ai-worker/eval/harness/multimodal_eval.py` 가
  `multimodal_sample.jsonl` (mixed EN/KR + Phase 9 anime 포스터) 과
  Phase 9 `multimodal_anime_kr.jsonl` 에 대해 실행. 두 파일 경로와 각
  row 의 `domain` 필드로 리포트가 도메인별 메트릭을 분리할 수 있음.
- **VLM 답변 생성 없음.** `FINAL_RESPONSE` 는 fused OCR + vision
  context 를 합성된 retrieval chunk 로 소비하는 기존 extractive
  generator 가 생성. 진짜 LLM generator 를 꽂는 것은 기존
  `GenerationProvider` 인터페이스 뒤의 독립 변경.

## Phase 3: AUTO capability

Phase 3 는 single-pass **AUTO** capability 를 추가합니다 — 제출된
`(text, file)` 쌍을 검사해 내부적으로 RAG / OCR / MULTIMODAL 중 하나로
job 을 라우팅하는 디스패처. 클라이언트가 capability 를 미리 고를 필요가
없게 됩니다. Phase 6 가 같은 seam 을 진짜 agent loop 로 업그레이드하는데,
Phase 3 는 의도적으로 단순하게 유지 — 라우터 호출 1번, 서브-capability
호출 1번, callback 1번.

### 왜 core-api 가 아니라 worker 에 두는가

라우팅은 capability 결정 (어느 엔진을 돌릴지, 어느 인덱스를 칠지, 질문을
어떻게 frame 할지) 이지 제출 결정 (payload 가 valid 한가) 이 아닙니다.
worker 에 두면 core-api 가 capability 별 로직에서 자유로워지고, 라우터는
RAG query parser 가 이미 의존하는 `LlmChatProvider` 에 접근할 수 있습니다.

### 5단계 흐름

```
         ┌─────────────────┐
(text,   │ route_classify  │  INPUT_TEXT / INPUT_FILE 을 디코드해
 file) ─►│                 │──► text, file_bytes, file_mime 으로
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
          │                 │ ─► direct_answer (인라인 LLM 호출)
          │                 │ ─► clarify (인라인 한국어 메시지)
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │   AGENT_DECISION│  라우팅 메타데이터를 담은 JSON artifact;
          │   + sub outputs │  항상 첫 번째 출력 artifact.
          └─────────────────┘
```

### 룰 기반 라우터 (기본)

`RuleBasedAgentRouter` 는 결정적 폴백이자 `AIPIPELINE_WORKER_AGENT_ROUTER=rule`
(기본값) 일 때의 기본 라우터. 입력 모양만으로 라우팅이 명백한 ~95% 의
경우를 처리하는 결정 트리:

| 입력                                           | Action       | Confidence |
|------------------------------------------------|--------------|------------|
| `text>6ch + file in {png,jpeg,pdf}`            | `multimodal` | 0.95       |
| `file in {png,jpeg,pdf}` 만                    | `ocr`        | 0.90       |
| `text>6ch` 만                                  | `rag`        | 0.70       |
| `text<=6ch` 만                                 | `clarify`    | 0.50       |
| 둘 다 없음                                     | `clarify`    | 0.00       |

"file in {png,jpeg,pdf}" 는 mime-type 기준 — 지원되지 않는 타입
(`image/gif`, `application/zip` 등) 은 "no-file" 행으로 collapse 되므로
`text>6ch + gif` 입력은 MULTIMODAL 안에서 실패하지 않고 RAG 로 라우팅
됩니다. 빈 파일 (`file_size=0`) 도 이 레이어에서 "no-file" 로 취급.

### LLM 라우터 (opt-in)

`LlmAgentRouter` 는 공유 `LlmChatProvider` (query parser 가 쓰는 동일
인스턴스) 를 감싸 5개 action 중 하나를 고르도록 요청. `function_calling`
을 광고하는 백엔드 (gemma4 가 있는 Ollama, Claude) 에서는 `route_job` tool
spec 으로 `chat_tools` 를 통과; 그 외에는 schema 힌트와 함께 `chat_json`
으로 폴백. 백엔드가 thinking mode 를 광고하면 활성화됨.

LLM 라우터는 다음 중 어떤 경우에도 **룰 라우터로 강등** (절대 raise 하지
않음):

- 하부 provider 의 `LlmChatError` (network, timeout, invalid JSON,
  empty response).
- 우리 레이어의 schema 위반 (필드 누락, 타입 오류, action 이 5-enum
  밖).
- `confidence < AIPIPELINE_WORKER_AGENT_CONFIDENCE_THRESHOLD` (기본
  0.55) — 모델이 답하긴 했지만 행동할 만큼 확신이 없음.

폴백 결정의 `router_name` 은 `f"llm-{chat.name}-fallback-rule"` 이 되어,
운영자가 `AGENT_DECISION` artifact 만으로 깨끗한 LLM 실행과 강등된 실행을
diff 할 수 있게 (로그를 읽지 않아도 됨).

### 레지스트리 와이어링

`build_default_registry` 가 AUTO 를 기회주의적으로 등록:

1. RAG, OCR, MULTIMODAL 모두 평소대로 등록 시도.
2. 셋 중 **하나라도** 성공하면, 살아있는 sub 들의 참조와 함께 AUTO
   등록. 누락된 sub 은 AutoCapability 의 `None` 이 됨 — 라우터는 여전히
   거기로 라우팅할 수 있지만, capability 가 `AttributeError` 로 죽지
   않고 dispatch 시점에 typed `AUTO_<sub>_UNAVAILABLE` 에러 코드를
   raise.
3. 셋 다 살아있지 않으면 AUTO 는 warning 과 함께 skip — 디스패치할 곳이
   없음.

`llm_backend=noop` (또는 worker 시작 시 NoOp 으로 강등된 백엔드) 일 때
LLM 라우터를 요청하면 레지스트리가 AUTO 를 룰 라우터로 자동 강등하면서
warning. AUTO 는 LLM 부재로 다운되지 않습니다.

### 단일 종단 callback

AUTO 는 평범한 `Capability` 입니다 — `CapabilityInput` 1개를 받아
`CapabilityOutput` 1개를 반환. 서브-capability 의 artifact 는 출력
리스트에 변경 없이 통과 (`AGENT_DECISION` 이 prepend 됨), 따라서
TaskRunner 는 RAG / OCR / MULTIMODAL 에 이미 처리하는 모양과 똑같이
AUTO job 당 정확히 1번의 종단 callback 을 발행. 서브-capability 의 내부
trace (OCR_RESULT.trace 또는 MULTIMODAL_TRACE) 는 그대로 보존 — AUTO 는
자기 trace 를 nesting 하지 않고 sibling artifact 로 라우팅 메타데이터를
추가합니다.

### 환경변수

| 변수                                                  | 기본값  | 의미                                                                    |
|-------------------------------------------------------|---------|-------------------------------------------------------------------------|
| `AIPIPELINE_WORKER_AGENT_ROUTER`                      | `rule`  | 라우터 구현 — `rule` 또는 `llm`.                                        |
| `AIPIPELINE_WORKER_AGENT_CONFIDENCE_THRESHOLD`        | `0.55`  | 이 confidence 미만의 LLM 결정은 룰 라우터로 폴백.                       |
| `AIPIPELINE_WORKER_AGENT_DIRECT_ANSWER_MAX_TOKENS`    | `512`   | LLM 라우터가 `direct_answer` 를 고를 때 요청하는 max token.              |

세 변수 모두 안전한 기본값으로 설정되어 있어, 환경변수가 비어있으면
Phase 3 부터 출시된 결정적 룰 라우터와 같은 동작을 얻습니다.

## Phase 3: AGENT capability — loop 기반 self-refinement

Phase 6 는 Phase 3/5 의 single-pass 디스패처를 진짜 agent loop 로
업그레이드하는 반복적 **AGENT** capability 를 추가합니다. AUTO capability
는 변경 없이 그대로 출시 (single-pass dispatch). AGENT 는 동일한
라우터 / 서브-capability 와이어링을 재사용하면서 그 위에 critic /
rewriter / synthesizer 레이어를 얹는 새 capability 이름.

이 loop 는 Phase 8 가 `kr_sample` 데이터셋에 대해 `loop_recovery_rate`
를 측정할 때까지 **opt-in** (`AIPIPELINE_WORKER_AGENT_LOOP=off` 가
기본). `agent_loop=off` 일 때 AGENT 는 AUTO 와 비트 단위로 동일한 동작
— 통합 테스트 (`tests/test_auto_capability.py`) 가 그 가드를 유지합니다.

### Loop 흐름

```
INPUT ─▶ classify ─▶ route_decide ─▶ ┌─[iter 0: delegate -> critique]
                                     │    critic 이 "sufficient" -> stop
                                     ├─[iter 1: rewrite -> retrieve -> critique]
                                     │    critic 이 "sufficient" -> stop
                                     ├─[iter N 또는 converged 또는 budget breach]
                                     └─▶ synthesize (union chunks) -> FINAL_RESPONSE
```

iter 0 에서 loop 의 `execute_fn` 동작:

* `action=rag` — 공유 `retriever.retrieve` + `generator.generate` 쌍을
  직접 실행 (loop 가 나중에 덮어쓸 중간 `RETRIEVAL_RESULT` 발행을
  피하기 위해 `RagCapability` 우회).
* `action=multimodal` — 풀 `MultimodalCapability` 를 실행해 OCR + vision
  + fusion 이 정확히 한 번만 발생. 이후 iter 에서는 loop 가 다시 작성된
  query 로 `retriever.retrieve` + `generator.generate` 로 폴백 (OCR/Vision
  은 재실행 안 함). MULTIMODAL 의 사이드 출력 artifact (`OCR_TEXT`,
  `VISION_RESULT`, `MULTIMODAL_TRACE`) 는 loop 자체 출력과 함께 보존.

Critic 이 충분성 판단:

* `NoOpCritic` — 항상 sufficient. Loop 가 single-pass 로 퇴화;
  `agent_loop=off` 또는 `agent_critic=noop` 일 때 선택.
* `RuleCritic` — 결정적 휴리스틱. 40자 미만이거나 한국어/영어 "I don't
  know" 마커를 포함한 답변을 `missing_facts` 로 표시.
* `LlmCritic` — Gemma 4 E2B function calling (`judge_answer` tool) +
  thinking mode. 5 글자 중 하나를 선택:
  `A=sufficient`, `B=missing_facts`, `C=ambiguous`,
  `D=off_topic`, `E=unanswerable`. Provider 실패나 invalid 글자에
  `RuleCritic` 으로 강등하면서 `critic_name="llm-fallback-rule"`.

Rewriter 가 다음 반복의 query 를 제안:

* `LlmQueryRewriter` — 이전 iter 의 전체 retrieved chunk 와 함께
  `chat_json` (`max_context_chars`, 기본 10K 로 clip — Gemma 4 의 128K
  context 안에 충분). Prompt 는 이미 검색된 것과 다른 정보를 명시적으로
  요구. `LlmChatError` 또는 invalid payload 시
  `parser.parse(gap_reason + " " + original)` 로 폴백,
  `parser_name="rewriter-fallback"`.

Synthesizer 가 매 iter 의 검색된 chunk 를 합집합 (중복 제거는
`chunk_id`, 첫 등장 우선) 으로 모아 최종 답변을 작성. 이게 핵심 품질
이득 — 두 다른 doc 의 보완 정보가 필요한 질문에서, 어느 단일 iter 의
top-k 도 둘 다 가지지 않더라도 양쪽 모두 표면화될 수 있음.

### Budget

`LoopBudget` 가 loop 를 묶음. 어떤 위반이든 typed `stop_reason` 으로
종료:

| 필드                       | 기본값   | 위반 시 stop reason       |
|----------------------------|---------:|---------------------------|
| `max_iter`                 | 3        | `iter_cap`                |
| `max_total_ms`             | 15 000   | `time_cap`                |
| `max_llm_tokens`           | 4 000    | `token_cap`               |
| `min_confidence_to_stop`   | 0.75     | (`converged` 결정에 사용) |

Critic verdict `gap_type='unanswerable'` 는 남은 budget 과 관계 없이
loop 를 `stop_reason='unanswerable'` 로 short-circuit — 답이 없는 코퍼스
에 대해 rewriting 해봐야 도움 안 됨.

### Artifact

Loop 가 동작할 때 (`agent_loop=on` + action 이 `{rag, multimodal}` 안):

| Artifact                | 내용                                                          |
|-------------------------|---------------------------------------------------------------|
| `AGENT_DECISION`        | 라우팅 메타데이터 (AUTO 와 동일).                             |
| `AGENT_TRACE`           | 풀 `LoopOutcome` JSON (budget + step 별 critique).            |
| `RETRIEVAL_RESULT_AGG`  | 매 iter 의 검색된 chunk 합집합, dedup 됨.                     |
| `FINAL_RESPONSE`        | 집계된 chunk 위에서 synthesizer 의 답변.                      |
| (MULTIMODAL 만)         | `OCR_TEXT`, `VISION_RESULT`, `MULTIMODAL_TRACE` 보존.         |

Loop 가 꺼졌거나 action 이 `ocr` / `direct_answer` / `clarify` 면
AGENT 는 Phase 5 의 artifact shape (`AGENT_DECISION` +
서브-capability 출력) 을 발행.

### Stop reason

| 값             | 의미                                                            |
|----------------|-----------------------------------------------------------------|
| `converged`    | Critic 이 sufficient 라고 했고 confidence >= `min_confidence_to_stop`. |
| `iter_cap`     | 수렴 없이 `max_iter` 반복 실행.                                 |
| `time_cap`     | Wall-clock 이 `max_total_ms` 초과.                              |
| `token_cap`    | 누적 LLM 토큰 (critic + executor) cap 초과.                     |
| `unanswerable` | Critic 이 질문을 코퍼스에서 답할 수 없다고 판단.                |

### 환경변수

| 변수                                                 | 기본값  | 의미                                                            |
|------------------------------------------------------|---------|-----------------------------------------------------------------|
| `AIPIPELINE_WORKER_AGENT_LOOP`                       | `off`   | 마스터 스위치. `on` 이면 AGENT 에서 반복 loop 활성화.            |
| `AIPIPELINE_WORKER_AGENT_CRITIC`                     | `rule`  | Critic provider. `llm`, `rule`, 또는 `noop`.                    |
| `AIPIPELINE_WORKER_AGENT_MAX_ITER`                   | `3`     | Loop 반복 hard cap.                                             |
| `AIPIPELINE_WORKER_AGENT_MAX_TOTAL_MS`               | `15000` | 전체 loop 의 wall-clock cap (ms).                               |
| `AIPIPELINE_WORKER_AGENT_MAX_LLM_TOKENS`             | `4000`  | 모든 iter 에 걸친 누적 LLM 토큰 cap.                            |
| `AIPIPELINE_WORKER_AGENT_MIN_STOP_CONF`              | `0.75`  | Early-stop `converged` 에 필요한 최소 critic confidence.        |

`agent_loop=off` 일 때 AGENT capability 는 등록되지만 AUTO 와 동일하게
동작. Loop 관련 환경변수는 default-off 플래그 옆에 함께 설정해도 안전 —
loop 가 켜졌을 때만 효과가 나타남.

### 실패 격리

Loop 는 capability 의 실패 표면을 확장해서는 **안 됩니다**:

* Critic / rewriter 실패는 룰 폴백으로 강등되고 `critic_name` /
  `parser_name` 접미사로 표면화 (`AGENT_TRACE` 와 Phase 8 metric layer
  에 보임).
* Iter 0 에서 `execute_fn` 이 raise 하면 빈 outcome 으로 폴백
  (`stop_reason=iter_cap`), capability 는 여전히 4개의 loop artifact 를
  발행 — 클라이언트가 일관된 shape 을 받음.
* Loop-wide 에러는 best-so-far 답변을 반환. job 이 loop 에러로 절대
  실패하지 않음.

## 파이프라인 trace 와 실패 리포팅

### 목적

MULTIMODAL 또는 OCR job 이 폴백하거나, 부분 성공하거나, 완전히 실패할 때,
운영자와 개발자는 worker 의 구조화된 로그를 읽고 job id 와 상관시키지
않고도 **어디서** 그리고 **왜** 그랬는지 봐야 합니다. Capability 레이어가
**정규화된 stage trace** 를 들고 다닙니다 — stage 흐름, 타이밍, typed
에러 코드를 단일 컴팩트 shape 에 캡쳐. 어느 capability 가 만든 job 이든
컨슈머는 같은 필드 이름을 봅니다.

Trace 는 **리포팅 전용** 입니다 — 파이프라인의 성공/폴백 동작을 바꾸지
않습니다. trace 없이도 성공했을 MULTIMODAL job 은 trace 와 함께도 여전히
성공하고, 특정 에러 코드로 실패했을 OCR job 은 같은 코드로 여전히 실패.
Trace 는 새로운 실패 모드가 아니라 관찰 가능성을 추가합니다.

### Trace 가 어디 사는가

| Capability  | Trace 캐리어              | 게이트                       |
|-------------|---------------------------|------------------------------|
| `OCR`       | `OCR_RESULT.trace`        | 항상 발행 (기존 OCR_RESULT envelope 에 추가) |
| `MULTIMODAL`| `MULTIMODAL_TRACE` artifact | `AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true` 로 opt-in |

두 캐리어 모두 **같은 스키마** (`trace.v1`) 를 갖습니다 — 다운스트림
컨슈머는 동일하게 처리할 수 있음. `OCR_RESULT` 의 기존 추출 필드
(`filename`, `mimeType`, `kind`, `engineName`, `pageCount`,
`textLength`, `avgConfidence`, `pages`, `warnings`) 는 변경 없음 —
`trace` 키는 추가이고, 파싱하지 않는 컨슈머는 동작 변화 0.

Trace artifact 를 발행하지 않는 어떤 **실패 경로**에서도 stage 흐름
요약은 `CapabilityError.message` 로 접혀 들어가므로, 운영자는 여전히
`GET /api/v1/jobs/{id}` 의 `errorMessage` 필드에서 읽을 수 있습니다.
요약 줄은 다음과 같이 보입니다:

```
classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms) vision:fail(VISION_VLM_TIMEOUT,5ms,fallback) fusion:skipped retrieve:skipped generate:skipped
```

### 스키마 (trace.v1)

페이로드는 평탄 — stage 별 `details` 외에는 중첩 map 없음:

```jsonc
{
  "schemaVersion": "trace.v1",   // string — 호환성 깨질 때 bump
  "capability": "MULTIMODAL",    // "OCR" | "MULTIMODAL"
  "inputKind":  "image",         // "image" | "pdf" | "text" | "unknown"
  "finalStatus":"ok",            // "ok" | "partial" | "failed"
  "stages": [
    {
      "stage":        "ocr",     // 정규 stage 이름 (어휘 참조)
      "provider":     "tesseract-5.3.3", // classify 같은 본질적 stage 는 null
      "status":       "ok",      // "ok" | "warn" | "fail" | "skipped"
      "code":         null,      // non-ok 에서 안정적인 에러/경고 코드
      "message":      null,      // ~200자로 클립
      "retryable":    null,      // true | false | null (unknown)
      "fallbackUsed": false,     // 파이프라인이 이 stage 를 지나 계속 갔는가?
      "durationMs":   42.1,      // stage wall-clock
      "details": {               // 작은 stage 별 메타데이터 (payload dump 안 함)
        "pageCount":      1,
        "textLength":     28,
        "avgConfidence":  91.0,
        "engineName":     "tesseract-5.3.3"
      }
    }
    // ... 더 많은 stage 레코드 ...
  ],
  "warnings": [],                // stage 와 묶이지 않은 파이프라인 수준 warning
  "summary":  "classify:ok(0ms) ocr:ok(42ms) ..."
}
```

### Stage 이름 어휘

`stage` 의 정규 값. Capability 레이어가 처음 6개를 발행. 마지막 4개는
TaskRunner / 외부 오케스트레이터용 예약 (현재 capability 안에서
populate 되지 않지만 어휘를 통일해두기 위해 문서화).

| Stage       | 발행자                         | 의미                                              |
|-------------|--------------------------------|---------------------------------------------------|
| `classify`  | OCR + MULTIMODAL               | 입력 파일 타입 / mime 분류                        |
| `ocr`       | OCR + MULTIMODAL               | 이미지 또는 PDF 에서 텍스트 추출                  |
| `vision`    | MULTIMODAL                     | 시각 description (heuristic 또는 VLM)             |
| `fusion`    | MULTIMODAL                     | OCR + vision + question 융합                      |
| `retrieve`  | MULTIMODAL                     | FAISS 에 대한 text-RAG retrieval                  |
| `generate`  | MULTIMODAL                     | Grounded 답변 생성                                |
| `fetch`     | *예약 (TaskRunner)*            | 입력 artifact 바이트 다운로드                     |
| `decode`    | *예약 (TaskRunner)*            | Artifact 콘텐츠 디코드                            |
| `upload`    | *예약 (TaskRunner)*            | 출력 artifact 바이트 업로드                       |
| `callback`  | *예약 (TaskRunner)*            | 종단 상태를 core-api 로 리포트                    |

### Status 어휘

| Status      | 시점                                                                       |
|-------------|----------------------------------------------------------------------------|
| `ok`        | Stage 가 깨끗하게 완료.                                                    |
| `warn`      | Stage 가 완료되었지만 non-fatal 이슈 있음 (예: 빈 OCR, 낮은 신뢰도). 부분 폴백 표시로 종종 `fallbackUsed=true` 와 짝지어짐. |
| `fail`      | Stage 가 예외를 raise. `fallbackUsed=true` 면 파이프라인이 폴백 경로로 이 stage 를 지나 계속 진행, `false` 면 이 실패가 종단. |
| `skipped`   | Stage 가 실행되지 않음 (보통 더 앞 stage 가 실패해서).                     |

### `finalStatus` 값

| 값         | 의미                                                              |
|------------|-------------------------------------------------------------------|
| `ok`       | 모든 stage 가 `ok` — 깨끗한 성공, caveat 없음.                    |
| `partial`  | 하나 이상의 stage 가 `warn` 또는 `fail(fallbackUsed=true)` 였지만 capability 는 여전히 필요한 출력 artifact 를 생성. |
| `failed`   | 종단 실패 — stage 가 실패했고 폴백 불가능.                        |

### 안정적 에러 / 경고 코드

아래 코드는
`ai-worker/app/capabilities/trace.py::STABLE_ERROR_CODES` 에 기록된
공식 레지스트리. 에러 코드로 분기하는 클라이언트는 이 집합으로
매치해야 합니다. 하부 provider 의 알려지지 않은 코드는 capability
prefix 와 함께 보존됩니다 (예: VLM 이 `PROVIDER_TIMEOUT` 을 반환하면
stage 레코드에서 `VISION_PROVIDER_TIMEOUT` 으로 표면화).

| 코드                                | 발행 시점                                              | 일반 `retryable` |
|------------------------------------|--------------------------------------------------------|------------------|
| `UNSUPPORTED_INPUT_TYPE`           | Classify stage — 파일이 PNG / JPEG / PDF 가 아님       | `false`          |
| `NO_INPUT`                         | Capability 가 입력 artifact 0건을 받음                | `false`          |
| `OCR_IMAGE_DECODE_FAILED`          | Pillow 가 이미지 바이트 디코드 실패                    | `false`          |
| `OCR_PDF_OPEN_FAILED`              | PyMuPDF 가 PDF 열기 실패 (corrupt/encrypted)           | `false`          |
| `OCR_PDF_EMPTY`                    | 0페이지 PDF                                            | `false`          |
| `OCR_TESSERACT_RUN_FAILED`         | Tesseract 호출이 추출 중 실패                          | `true`           |
| `OCR_TOO_MANY_PAGES`               | PDF 가 설정된 `ocr_max_pages` 초과                     | `false`          |
| `OCR_EMPTY_TEXT` (warn)            | OCR 가 깨끗하게 돌았으나 0글자 생성                    | —                |
| `OCR_LOW_CONFIDENCE` (warn)        | 평균 신뢰도가 `ocr_min_confidence_warn` 미만           | —                |
| `VISION_<provider-code>`           | Vision provider 가 VisionError raise                  | 보통 `true`      |
| `VISION_PDF_RASTERIZATION_FAILED`  | PyMuPDF 1페이지 rasterization 실패                     | `false`          |
| `VISION_PDF_EMPTY`                 | Rasterization 이 0바이트 반환                          | `false`          |
| `MULTIMODAL_ALL_PROVIDERS_FAILED`  | OCR 와 vision 모두 사용 가능한 신호를 못 만듦          | `false`          |
| `MULTIMODAL_RETRIEVAL_FAILED`      | OCR / vision 성공 후 Retriever 가 raise                | `true`           |
| `MULTIMODAL_GENERATION_FAILED`     | Retrieval 성공 후 Generator 가 raise                   | `true`           |

### 예시: 성공

Happy-path MULTIMODAL job, 6 stage 모두 `ok`, `finalStatus=ok`:

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

### 예시: 부분 폴백 (OCR ok, vision fail)

Vision provider 가 timeout. OCR 와 다운스트림 모두 여전히 완료.
`finalStatus=partial` 이 컨슈머에게 "완료, 단 표면화할 가치 있는 caveat
있음" 을 알림:

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

### 예시: 종단 실패 (두 provider 모두 실패)

OCR 와 vision 모두 raise. Artifact 생성되지 않음. Worker 는
`MULTIMODAL_ALL_PROVIDERS_FAILED` 를 반환하면서 stage 흐름 요약을 에러
메시지에 임베딩하므로 운영자는 `GET /api/v1/jobs/{id}` 만으로 전체
그림을 봄:

```
errorCode    = "MULTIMODAL_ALL_PROVIDERS_FAILED"
errorMessage = "Multimodal pipeline could not extract any signal from the input — OCR returned no text AND the vision provider returned no description. Upstream diagnostics: ocr stage failed (IMAGE_DECODE_FAILED): corrupt png | trace: classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms,fallback) vision:fail(VISION_VLM_TIMEOUT,5ms,fallback) fusion:skipped retrieve:skipped generate:skipped"
```

`emit_trace=true` 일 때는 위와 같은 stage 리스트가 `MULTIMODAL_TRACE`
artifact 로도 다운로드 가능 — 단, 기본 게이트는 off 로 유지하여 문서화된
4-artifact MULTIMODAL 응답을 보존합니다.

### 예시: OCR / vision 성공 후 retrieval 실패

전형적인 "upstream 동작했으나 downstream 깨짐" 시나리오. 새
`MULTIMODAL_RETRIEVAL_FAILED` 코드가 앞선 성공 컨텍스트를 보존한 채
이 상황을 깔끔하게 표면화:

```
errorCode    = "MULTIMODAL_RETRIEVAL_FAILED"
errorMessage = "Retrieval stage failed after OCR / vision already produced usable signal. Upstream error: RuntimeError: FAISS search blew up: index corrupt | trace: classify:ok(0ms) ocr:ok(3ms) vision:ok(2ms) fusion:ok(0ms) retrieve:fail(MULTIMODAL_RETRIEVAL_FAILED,12ms) generate:skipped"
```

이를 읽는 운영자는 즉시 진단 가능:

1. OCR 와 vision 은 건강 (`ocr:ok`, `vision:ok`).
2. FAISS 인덱스가 실제 문제 (`retrieve:fail` + `FAISS search blew up:
   index corrupt` 원인 라인).
3. Generation 은 시도되지 않음 (`generate:skipped`).

유사한 `MULTIMODAL_GENERATION_FAILED` 코드는 retrieval 은 완료됐는데
generator 가 raise 한 드문 경우를 커버합니다.

## 고수준 토폴로지

```
                   ┌──────────────────┐
                   │   frontend /     │
                   │   test client    │
                   └────────┬─────────┘
                            │ HTTP (public)
                            ▼
                   ┌──────────────────┐
                   │    core-api      │  ← Spring Boot, DDD + 헥사고날
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
                       │   (Python, 장수명)    │    long-lived,
                       │                      │    capability registry
                       └──────────┬───────────┘
                                  │ read/write
                                  ▼
                       ┌──────────────────────┐
                       │  스토리지 백엔드     │
                       │  로컬 FS (phase 1)   │
                       │  MinIO / S3 (이후)   │
                       └──────────────────────┘
```

## 왜 Redis + 장수명 worker 인가 (Cloud Tasks / serverless 가 아니라)

이 플랫폼은 GPU, OCR, PDF, FAISS-in-memory 워크로드를 돌릴 것으로
예상됩니다. 그것들은 serverless 와 잘 맞지 않습니다: 콜드 스타트가 GPU
초기화를 죽이고, FAISS 인덱스는 여러 요청에 걸쳐 worker RAM 에서 살고
싶어하며, 호출당 과금은 긴 처리를 비싸게 만듭니다.

그래서: worker 는 평범한 OS 프로세스로, 가지고 있는 GPU / 코어 수만큼
복사본을 띄우고, Redis 는 단순히 "일어나, 새 job 이 큐에 있어" 신호일
뿐입니다. **진실의 상태는 PostgreSQL 에 머무릅니다.** Redis 는 잃었다가
복원해도 (최악의 경우 job 들이 재디스패치 전까지 QUEUED 에 머무름)
플랫폼은 모든 job 이 무엇을 하는지 여전히 압니다.

## 진실의 출처

| 관심사                       | 소유자             | 비고                                          |
|------------------------------|--------------------|-----------------------------------------------|
| Job 생명주기 / 상태          | PostgreSQL (SoT)   | core-api 만 작성                              |
| 디스패치 신호                | Redis              | Ephemeral, 재구성 가능                        |
| Artifact 바이트              | 스토리지 백엔드    | 지금은 로컬 FS, 나중에 MinIO/S3              |
| Artifact 메타데이터          | PostgreSQL         | Job 에 연결                                   |
| Claim lease                  | PostgreSQL         | 원자적 conditional UPDATE                     |
| Callback 멱등성              | PostgreSQL         | Job row 의 `last_callback_id`                 |

## Core-api 구조 (DDD + 헥사고날)

```
core-api/src/main/java/com/aipipeline/coreapi/
├── CoreApiApplication.java
├── common/                          ← 횡단 관심사 (properties, clock)
├── job/
│   ├── domain/                      ← 순수 Java, framework import 없음
│   │   ├── Job.java                 ← 상태 전이를 가진 aggregate root
│   │   ├── JobId.java               ← value object
│   │   ├── JobCapability.java       ← enum
│   │   ├── JobStatus.java           ← state machine
│   │   └── JobStateTransitionException.java
│   ├── application/
│   │   ├── port/in/                 ← primary port (use case)
│   │   │   ├── JobManagementUseCase.java
│   │   │   └── JobExecutionUseCase.java
│   │   ├── port/out/                ← secondary port
│   │   │   ├── JobRepository.java
│   │   │   └── JobDispatchPort.java
│   │   └── service/                 ← use case 구현
│   │       ├── JobCommandService.java
│   │       └── JobExecutionService.java
│   └── adapter/
│       ├── in/web/                  ← inbound: REST 컨트롤러 + DTO
│       │   ├── JobController.java
│       │   ├── InternalWorkerController.java
│       │   └── dto/*
│       └── out/persistence/         ← outbound: Spring Data JPA
│           ├── JobJpaEntity.java
│           ├── JobJpaRepository.java
│           └── JobPersistenceAdapter.java
├── artifact/                        ← 같은 3-레이어 모양
│   ├── domain/
│   ├── application/
│   └── adapter/
├── queue/adapter/out/redis/         ← Redis 디스패치 어댑터
└── storage/adapter/out/local/       ← 로컬 파일시스템 스토리지 어댑터
```

**규칙**: domain 레이어는 Spring, JPA, Redis, Jackson 의 그 무엇도
import 하지 않습니다. application 레이어는 자기 port 와 domain 만
import. Framework 의존성은 어댑터에 삽니다.

## Job 상태 머신

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

전이는 `JobStatus.canTransitionTo(...)` 에 인코딩되어 있습니다 — 상태를
바꾸는 모든 service 호출은 enum 에게 먼저 묻기 때문에, illegal transition
은 상태를 조용히 손상시키는 대신 domain 예외를 발생시킵니다.

- **enqueue**: `JobCommandService.createAndEnqueue` 가 job 을 생성하고,
  QUEUED 로 마크하고, job 을 `JobDispatchPort` 에 넘기는 afterCommit
  hook 을 등록. 데이터베이스 commit 이 먼저 일어나므로 worker 는 claim
  할 수 없는 row 를 절대 보지 않습니다.
- **claim**: `JobExecutionService.claim` 이
  `JobRepository.tryAtomicClaim` 호출, row 가 PENDING 또는 QUEUED 이고
  다른 worker 가 live lease 를 가지지 않을 때만 성공하는 conditional
  UPDATE 를 실행. 원자성이 SQL 레벨에 있어 같은 job 에 경합하는 여러
  worker 가 둘 다 이기는 일은 없습니다.
- **callback**: `JobExecutionService.handleCallback` 이 worker 가
  제시한 claim token 을 검증하고, 멱등 재생을 위해 callback id 를
  `last_callback_id` 와 대조한 뒤에야 job 을 SUCCEEDED 또는 FAILED
  로 전이. 출력 artifact 는 같은 트랜잭션에서 영속화됩니다.

## Redis 디스패치 계약

- **Pending list**: `aipipeline:jobs:pending` (core-api 에서 LPUSH,
  worker 에서 BRPOP). Phase 1 에서는 단일 키. capability 별 lane 은
  미래 정제 사항.
- **메시지 모양** (JSON):
  ```json
  {
    "jobId": "...",
    "capability": "MOCK",
    "attemptNo": 1,
    "enqueuedAtEpochMilli": 1744723200000,
    "callbackBaseUrl": "http://localhost:8080"
  }
  ```
- 메시지는 worker 가 home 에 phone 할 정도의 정보만 들고 다님. 입력
  artifact 는 claim 응답으로 돌아오므로, Redis 와 Postgres 간 상태를
  중복시키지 않습니다.

## 스토리지 계약

`ArtifactStoragePort` 가 4개 연산을 노출:

- `store(jobId, type, filename, contentType, content, length)` — 바이트
  를 쓰고 `StoredObject(storageUri, sizeBytes, checksumSha256)` 반환.
- `openForRead(storageUri)` — opaque URI 를 해소하고 `InputStream` 반환.
  Phase 1 은 `local://...` 만 처리. S3/MinIO 는 자체 scheme 을 나중에
  추가.
- `generateDownloadUrl(artifactId)` — phase 1 은 core-api route 반환.
  Phase 2 는 진짜 presigned URL 반환.
- Phase 1 에서 worker 는 속도를 위해 공유 디스크에서 `local://` 를 직접
  읽고, 로컬에서 해소할 수 없는 어떤 scheme 이든 HTTP 다운로드로 폴백.

## Worker 구조

```
ai-worker/
└── app/
    ├── main.py                ← 엔트리포인트: registry 빌드, consumer 시작
    ├── core/
    │   ├── config.py          ← pydantic-settings (env 기반)
    │   └── logging.py
    ├── queue/
    │   ├── redis_consumer.py  ← BRPOP 루프
    │   └── messages.py        ← QueueMessage wire shape
    ├── clients/
    │   ├── core_api_client.py ← claim / callback / upload HTTP
    │   └── schemas.py
    ├── storage/
    │   └── resolver.py        ← local://  → 파일시스템 경로
    ├── capabilities/
    │   ├── base.py            ← Capability 인터페이스 + 데이터 클래스
    │   ├── registry.py        ← 이름 → 인스턴스 map + 공유 컴포넌트 캐시
    │   ├── mock_processor.py  ← phase 1 echo capability
    │   ├── rag/               ← phase 2: FAISS retriever + extractive generator
    │   ├── ocr/               ← phase 2: Tesseract + PyMuPDF provider
    │   └── multimodal/        ← phase 2 v1: OCR + vision + fusion + text RAG
    └── services/
        └── task_runner.py     ← claim → fetch → run → upload → callback
```

Worker 는 의도적으로 DDD 를 사용하지 **않습니다**. 절차적 실행 엔진:
책임당 클래스 하나, `TaskRunner` 안의 직선형 오케스트레이션. 확장
지점은 capability 뿐이고, 그건 평범한 인터페이스입니다.

## End-to-end 흐름

1. 클라이언트가 `POST /api/v1/jobs` (텍스트) 또는 multipart 업로드.
2. `JobController` 가 `ArtifactStoragePort.store` 로 바이트 stage.
3. `JobCommandService.createAndEnqueue`:
   - `Job` (PENDING) 생성 → QUEUED 로 마크,
   - 입력 artifact 영속화,
   - afterCommit hook 등록,
   - 트랜잭션 commit,
   - hook 이 `JobDispatchPort.dispatch` → `RedisJobDispatchAdapter`
     발화, `aipipeline:jobs:pending` 에 `QueueMessage` 를 LPUSH.
4. Worker 의 `RedisQueueConsumer` 가 메시지를 BRPOP 으로 받아
   `TaskRunner.handle` 에 넘김.
5. `TaskRunner` 가 `POST /api/internal/jobs/claim` 호출. core-api 가
   원자적 conditional UPDATE 실행; 이기면 상태가 RUNNING 이 되고
   응답에 입력 artifact 리스트가 실려옴.
6. Worker 가 storage resolver (로컬 파일시스템) 로 입력 바이트 읽음.
7. Worker 가 매칭되는 `Capability` 실행 (phase 1: `MockProcessor`).
8. Worker 가 `POST /api/internal/artifacts` 로 각 출력 artifact 업로드.
   core-api 가 같은 `ArtifactStoragePort.store` 로 바이트를 쓰고
   `(storageUri, sizeBytes, checksumSha256)` 만 반환 — **이 시점에
   Artifact row 를 만들지 않습니다.** 이는 진짜 presigned-upload 흐름
   의미와 일치합니다 — 업로드는 object 스토리지에 직접 도착하고,
   데이터베이스는 callback 으로부터만 새 object 에 대해 듣습니다.
9. Worker 가 최종 callback 게시:
   `POST /api/internal/jobs/callback` 에 결과 SUCCEEDED / FAILED 와
   출력 artifact 메타데이터 (방금 받은 storage URI 포함) 리스트 전송.
10. `JobExecutionService.handleCallback` 이 상태 전이를 적용 AND 출력
    Artifact row 를 영속화 (단일 권위 있는 쓰기 경로). 같은
    `callbackId` 가 두 번 도착하면 job row 의 `last_callback_id` 로
    감지되어 두 번째 호출은 no-op 이 됩니다 — 중복 artifact 없음.
11. 클라이언트가 `GET /api/v1/jobs/{id}` /
    `GET /api/v1/jobs/{id}/result` 로 폴링하고 반환된 `accessUrl` 로
    출력 다운로드.

## 보류된 항목

| 관심사                                                          | Phase |
|-----------------------------------------------------------------|-------|
| 진짜 OCR 엔진                                                   | 2 (출시) |
| FAISS 기반 RAG capability                                       | 2 (출시) |
| Multimodal v1 (OCR + vision + text RAG)                         | 2 (출시) |
| AUTO capability (single-pass dispatcher)                        | 3 (출시) |
| AGENT capability (loop + critic + retry)                        | 6     |
| ~~진정한 multimodal retrieval (이미지 임베딩, 크로스모달 검색)~~ | 출시 (CLIP + RRF, opt-in) |
| 진짜 VLM provider (BLIP-2 / Claude Vision / GPT-4V / Gemini)    | 3+    |
| PDF 다중 페이지 vision captioning                               | 3+    |
| Multimodal 자동 eval harness                                    | 3+    |
| ~~MinIO / S3 스토리지 어댑터~~                                  | 출시 (backend=s3, AWS SDK v2) |
| ~~`/api/internal/*` 인증~~                                      | 출시 (공유 시크릿 헤더) |
| Retry 오케스트레이션                                            | 2     |
| Capability 별 Redis lane                                        | 2     |
| Kubernetes 매니페스트                                           | 2+    |
| 테스트 폼 이상의 frontend                                       | 2+    |

실제 실행 방법은 `docs/local-run.md`, 구체적 엔드포인트 계약은
`docs/api-summary.md` 참조.
