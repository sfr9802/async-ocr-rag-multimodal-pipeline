# Tier 2 — 멀티모달 Eval + Internal Auth + Cross-modal Retrieval

> 이 파일은 새 Claude Code 세션에 그대로 붙여넣기 위한 self-contained 프롬프트입니다.
> 작업 루트: `D:\async-ocr-rag-multimodal-pipeline`
> **선행 조건:** Tier 1 (`docs/prompts/tier1-vlm-llm-korean.md`)이 먼저 머지된 상태를 가정합니다.

---

# 작업: 멀티모달 eval 하네스 구현, /api/internal/* shared-secret 인증, CLIP 이미지 임베딩 인덱스 + RRF fusion 추가

## 배경
프로젝트 루트: `D:\async-ocr-rag-multimodal-pipeline`.
이 세 작업은 독립적이지만 Tier 2 "프로덕션 스토리 완성" 묶음이다.

현재 상태:
1. `ai-worker/eval/datasets/multimodal_sample.jsonl`는 스키마 플레이스홀더만 있고 harness가 없다
   (`eval/README.md` 87-92줄에 "placeholder only, harness deferred"로 명시되어 있음)
2. `core-api`의 `/api/internal/*` 엔드포인트 (claim / callback / artifact upload)에 인증이 없다.
   `docs/architecture.md` 623줄에 deferred로 기록되어 있음
3. MULTIMODAL capability는 "OCR 텍스트 + 구조적 vision description → 기존 text RAG" 구조다.
   진짜 cross-modal retrieval (이미지 임베딩 인덱스)이 없다

Tier 1(프롬프트 1)이 먼저 머지된 상태를 가정한다. 아니라면 먼저 Tier 1을 실행할 것.

## 먼저 읽어야 할 파일
1. `ai-worker/eval/README.md`, `eval/run_eval.py`, `eval/harness/rag_eval.py`, `eval/harness/ocr_eval.py`, `eval/harness/metrics.py`
2. `ai-worker/eval/datasets/multimodal_sample.jsonl` — 기존 스키마 스텁
3. `core-api/src/main/java/com/aipipeline/coreapi/job/adapter/in/web/InternalWorkerController.java`
4. `core-api/src/main/java/com/aipipeline/coreapi/artifact/...` — internal artifact upload 컨트롤러
5. `core-api/src/main/java/com/aipipeline/coreapi/common/` — properties, config 패턴
6. `ai-worker/app/clients/core_api_client.py` — HTTP 클라이언트
7. `ai-worker/app/capabilities/rag/faiss_index.py`, `retriever.py`, `embeddings.py`, `ingest.py`
8. `ai-worker/app/capabilities/multimodal/capability.py` — 현 cross-modal 부재 구조
9. `docs/architecture.md` 섹션: Phase 2 multimodal limitations, Storage contract

## 목표
1. `eval/harness/multimodal_eval.py` 구현 및 `run_eval.py multimodal` 서브커맨드 추가
2. `/api/internal/*` 전 경로에 공유 시크릿 헤더 인증 (+ worker 클라이언트 업데이트)
3. 이미지 임베딩 FAISS 인덱스 추가 + RRF score fusion → MULTIMODAL이 cross-modal retrieval 사용

## 작업 A — 멀티모달 eval 하네스

### A.1 데이터셋 확장
`ai-worker/eval/datasets/multimodal_sample.jsonl`:
- 최소 6개 row
- 이미지는 `eval/datasets/samples/multimodal/` 하위에 생성
  (kr/en 혼합 3개 이상, OCR+vision 조합 커버)
- 스키마는 기존 placeholder 필드 그대로 사용 (`image`, `question`, `expected_answer`,
  `expected_keywords`, `expected_labels`, `requires_ocr`, `language`, `notes`)
- 생성 스크립트: `ai-worker/scripts/make_multimodal_sample_fixtures.py` (새 파일)
  - Pillow로 문자+간단한 도형 포함 PNG 생성 (scale chart, receipt mock, etc.)
  - OCR만으로도 답이 나오는 row, 이미지 context 필요한 row, 두 가지 모두 필요한 row 구성
- `eval/datasets/samples/multimodal/`는 `.gitignore`에 추가

### A.2 Harness 구현
`ai-worker/eval/harness/multimodal_eval.py` (new):
- `rag_eval.py`의 구조와 로깅/보고 포맷을 따른다
- 입력: (retriever, generator, ocr_provider, vision_provider) 또는 MultimodalCapability 인스턴스
- 각 row에 대해 해당 capability를 실행하고 FINAL_RESPONSE 텍스트를 수집
- 메트릭:
  - `exact_match` (비어있지 않은 `expected_answer`에 대해서만)
  - `substring_match` (`expected_answer`가 답변에 substring으로 포함)
  - `keyword_coverage` (기존 `metrics.py::keyword_coverage` 재사용)
  - `label_recall` / `label_precision` (`expected_labels` 기준)
  - latency p50/mean/max (OCR 단계 / Vision 단계 / Retrieval+Generation 단계 분리)
- 단일 row 실패를 전체 실패로 만들지 말고 row-level error로 report에 포함
- 출력: JSON + CSV, 기존 `io_utils.py` 재사용

### A.3 CLI 통합
`ai-worker/eval/run_eval.py`:
- `multimodal` 서브커맨드 추가 — `rag`/`ocr` 기존 서브커맨드 패턴 그대로 복사
- 기본 report 경로: `eval/reports/multimodal-{timestamp}.{json,csv}`
- `--require-ocr-only` 플래그: `requires_ocr=True` row만 실행
- `--vision-provider heuristic|claude` 플래그로 오버라이드

### A.4 gate 문서화
`eval/README.md`의 "Recommended evaluation sequence" 섹션 4번("Multimodal (future)")을
실제 gate로 교체:
- `mean_keyword_coverage ≥ 0.60` AND `substring_match ≥ 0.50`을 baseline으로 제시

### A.5 단위 테스트
`ai-worker/tests/test_multimodal_eval.py` (new) — harness 동작 검증 (stub providers 주입)

## 작업 B — /api/internal/* shared-secret 인증

### B.1 core-api 쪽
- `core-api/src/main/java/com/aipipeline/coreapi/common/security/InternalSecretProperties.java` (new)
  - `@ConfigurationProperties(prefix = "aipipeline.internal")` / `secret: String`
- `InternalSecretAuthInterceptor.java` (new)
  - `HandlerInterceptor` 구현
  - `preHandle`에서 `X-Internal-Secret` 헤더 검증
  - 시크릿 미설정(개발 모드)이면 pass + WARN 로그 1회
  - 불일치/누락이면 401 + `{ "error": "internal_auth_failed" }`
- `WebConfig` 또는 새 `InternalSecurityConfig.java`에서 interceptor 등록,
  pathPattern: `/api/internal/**`
- `application.yml`에 `aipipeline.internal.secret: ${AIPIPELINE_INTERNAL_SECRET:}` 추가
- `application-test.yml` / `application-independent.yml` 있으면 동일 키 추가 (빈 값 OK)

### B.2 worker 쪽
- `ai-worker/app/core/config.py`에 `internal_secret: SecretStr | None = None`
  (env: `AIPIPELINE_WORKER_INTERNAL_SECRET`) 추가
- `ai-worker/app/clients/core_api_client.py`:
  - 모든 요청(claim / callback / artifact upload)에 `X-Internal-Secret` 헤더 자동 삽입
  - 시크릿 미설정 시 헤더 생략 (dev 호환)
- `.env.example`에 `AIPIPELINE_WORKER_INTERNAL_SECRET=` 추가

### B.3 E2E smoke 업데이트
`scripts/e2e_smoke.py`:
- 존재할 경우 동일 환경변수로부터 시크릿 읽어서 core-api 호출 시 주입
- dev 모드(시크릿 미설정)면 기존 동작 유지

### B.4 테스트
- core-api 쪽: `InternalSecretAuthInterceptorTest` (MockMvc로 401/200 검증)
- worker 쪽: `test_core_api_client.py`에 헤더 주입 케이스 추가

### B.5 문서
- `docs/architecture.md`의 "What's deferred" 테이블에서 "Auth on /api/internal/*" 제거
  또는 "shipped" 로 마크
- `README.md` "Run it" 섹션에 시크릿 생성/주입 스니펫 추가

## 작업 C — CLIP 이미지 임베딩 인덱스 + RRF Fusion

### C.1 스택 결정
- 임베더: **`sentence-transformers/clip-ViT-B-32`** (sentence-transformers가 이미 종속성에 있음,
  추가 설치 없음). 대안 `open-clip-torch`는 채택하지 말 것 (종속성 폭증)
- 이미지 FAISS는 기존 `IndexFlatIP` 재사용 (정규화된 512-dim 벡터)
- 텍스트 쿼리로 이미지 검색은 CLIP 텍스트 인코더 사용

### C.2 신규 모듈
`ai-worker/app/capabilities/rag/image_embeddings.py`:
- `ImageEmbedder` (abstract) + `ClipImageEmbedder` 구현
- `encode_images(bytes_list) -> np.ndarray` (L2 normalized)
- `encode_texts(text_list) -> np.ndarray` (L2 normalized, cross-modal 쿼리용)
- Lazy import (클래스 생성자에서 `from sentence_transformers import SentenceTransformer`)

`ai-worker/app/capabilities/rag/image_index.py`:
- `ImageFaissIndex` — 기존 `FaissIndex`와 동일 패턴, 사이드카 `image_build.json`
- 빌드 시 `doc_id`, `page_number`, `image_hash` 메타 트래킹

### C.3 Metadata store 확장
`ai-worker/app/capabilities/rag/metadata_store.py`:
- `ragmeta.images` 테이블 + migration V3
  - 컬럼: `image_id (PK)`, `doc_id`, `page_number`, `source_uri`, `width`, `height`, `sha256`, `indexed_at`
- DAO 메서드: `insert_images`, `resolve_image_rows(ids)`

Core-api 쪽에는 손대지 말 것 — ragmeta는 worker 전용 psycopg2 경계.
Flyway migration은 `core-api/src/main/resources/db/migration/V3__ragmeta_images.sql` 로 추가.

### C.4 Ingest 확장
`ai-worker/scripts/build_rag_index.py`:
- `--with-images DIR` 플래그 추가
- 해당 디렉터리 아래 PNG/JPG를 순회하며 CLIP으로 임베드 → `ImageFaissIndex`에 저장
- 텍스트 인덱스와 별개 파일 (`index_dir/image.faiss`, `image_build.json`)

### C.5 Retriever 확장
`ai-worker/app/capabilities/rag/retriever.py`:
- `Retriever.retrieve()` 시그니처는 그대로, 내부에서 fusion 옵션 추가
- 새 메서드 `retrieve_multimodal(query_text, image_bytes=None, top_k=...)`:
  - text index 검색 → top-k 텍스트 후보
  - CLIP 텍스트 인코더로 `query_text`를 벡터화 → 이미지 인덱스 검색 → top-k 이미지 후보
  - **RRF fusion** (k=60): `score = Σ 1/(k + rank)` — 동일 `doc_id`의 텍스트/이미지 후보를 합산
  - 최종 top-k를 `RetrievedChunk` 리스트로 반환 (이미지 후보는 `section="image"` 로 표시, text는 CLIP이 설명한 caption으로 대체)
- 기존 `retrieve()`는 text-only 경로 그대로 유지 — 호환성 보장

### C.6 MULTIMODAL capability 배선
`ai-worker/app/capabilities/multimodal/capability.py`:
- 새 config 필드: `use_cross_modal_retrieval: bool = False` (안전한 off 기본값)
- True일 때 `retriever.retrieve_multimodal(fusion.retrieval_query, image_bytes=artifact.content)` 사용
- False면 기존 `retriever.retrieve(...)` 유지

### C.7 Registry
`_get_shared_retriever_bundle`:
- 이미지 인덱스가 존재하면 CLIP embedder 로드 후 retriever에 주입
- 존재하지 않으면 None 주입 → cross-modal 요청은 text-only로 fallback
- Settings: `rag_image_index_dir`, `rag_clip_model` (`sentence-transformers/clip-ViT-B-32`),
  `multimodal_use_cross_modal_retrieval: bool = False`

### C.8 Eval 확장
`eval/run_eval.py multimodal` 서브커맨드에 `--cross-modal` 플래그 추가 — off로 돌린 결과와
on으로 돌린 결과의 메트릭 차이를 README 벤치마크 테이블에 새 row로 추가.

### C.9 테스트
- `test_image_embeddings.py` — CLIP lazy load + L2 normalization 검증
- `test_image_index.py` — build/save/load roundtrip
- `test_retriever_cross_modal.py` — stub embedder + 더미 인덱스로 RRF fusion 검증
- 기존 `test_rag_*`, `test_multimodal_*` 회귀 금지

## 기존 패턴 준수
- `registry.py` 의 graceful degradation 패턴을 이미지 인덱스/CLIP 로딩 실패에도 적용
- `_shared_component_cache`에 CLIP embedder + image index 추가 (더블 로딩 금지)
- core-api 쪽 변경은 domain 레이어를 건드리지 않는다 — interceptor는 adapter/web에 위치
- ragmeta 스키마 migration은 Flyway V3로 추가 (V1 pipeline, V2 ragmeta가 이미 있음)

## 수용 기준
- `pytest ai-worker/tests/ -q` 전부 통과 + 신규 테스트 포함
- `cd core-api && mvn test` 전부 통과 + 인증 인터셉터 테스트 포함
- `AIPIPELINE_INTERNAL_SECRET=test` 환경에서 worker와 core-api를 모두 기동해 e2e smoke 통과
- Cross-modal retrieval off/on 두 경로 모두 수동 smoke:
  ```
  python scripts/e2e_smoke.py  # 기존
  python scripts/e2e_smoke.py --multimodal-cross-modal  # 신규
  ```
- `eval/run_eval.py multimodal` 실행 후 report JSON + CSV 생성
- README 벤치마크 테이블에 multimodal 섹션 추가
- `docs/architecture.md`에서 "Multimodal v1 limitations"의 "Not true multimodal retrieval" 항목을
  "opt-in cross-modal retrieval via CLIP + RRF fusion (off by default)" 로 수정

## 비목표
- 새 VLM/LLM 추가 — 프롬프트 1에서 이미 처리
- 이미지 인덱스 migration 자동화 (수동 `--with-images` CLI 실행으로 충분)
- 멀티모달 retrieval을 RAG capability의 기본 경로로 변경 — opt-in만
- CLIP fine-tuning
- BLIP-2 / ImageBind / BGE-M3 multimodal 등 다른 임베더
- Internal auth에 JWT / OAuth — shared secret으로 충분
- Rate limiting / IP allowlist
