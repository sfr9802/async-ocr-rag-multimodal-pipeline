# Tier 3 — Multi-page Vision + S3 Adapter + Demo Script

> 이 파일은 새 Claude Code 세션에 그대로 붙여넣기 위한 self-contained 프롬프트입니다.
> 작업 루트: `D:\async-ocr-rag-multimodal-pipeline`
> **선행 조건:** Tier 1 + Tier 2 (`docs/prompts/tier1-*.md`, `docs/prompts/tier2-*.md`)가 먼저 머지된 상태를 가정합니다.

---

# 작업: Multimodal에 다페이지 vision 캡션 지원, S3/MinIO 스토리지 어댑터 추가, end-to-end 데모 스크립트 작성

## 배경
프로젝트 루트: `D:\async-ocr-rag-multimodal-pipeline`.
Tier 1 + Tier 2가 머지된 상태를 가정한다. 이 프롬프트는 프로덕션 감각 마무리 단계다.

현재 상태:
1. `ai-worker/app/capabilities/multimodal/capability.py`는 PDF의 1페이지만 vision stage에
   보낸다. 코드 주석(capability docstring)과 `docs/architecture.md` 49-50줄에 명시되어 있음.
2. `ArtifactStoragePort` (core-api)는 포트가 잘 잡혀있지만 로컬 파일시스템 어댑터 하나뿐이다.
   `docs/architecture.md` "What's deferred" 테이블에 MinIO/S3 phase 2로 예정됨.
3. 처음 보는 사람을 위한 원커맨드 데모가 없다. `scripts/e2e_smoke.py`는 smoke일 뿐
   서사/아웃풋이 화려하지 않다.

## 먼저 읽어야 할 파일
1. `ai-worker/app/capabilities/multimodal/capability.py` — 특히 `_run_vision_stage` 와 `_default_rasterize_pdf_first_page`
2. `ai-worker/app/capabilities/ocr/tesseract_provider.py::ocr_pdf` — 페이지 순회 패턴
3. `core-api/src/main/java/com/aipipeline/coreapi/storage/` 전체 — port + local adapter
4. `core-api/src/main/java/com/aipipeline/coreapi/artifact/...` — StoredObject 반환 shape
5. `docker-compose.yml` 의 minio profile 설정
6. `README.md` 의 "Run it" 섹션
7. `scripts/e2e_smoke.py`

## 목표
1. MULTIMODAL capability가 설정된 `max_vision_pages`까지 PDF 페이지를 캡션한다
2. `S3ArtifactStorageAdapter` 구현 — MinIO와 AWS S3 호환, 기존 port 규약 준수
3. `scripts/demo.py` — 샘플 PDF를 업로드하고 multimodal 쿼리를 돌려 결과를 예쁘게 출력

## 작업 A — 다페이지 Vision

### A.1 설정
`ai-worker/app/core/config.py`:
- `multimodal_max_vision_pages: int = 3` (0 또는 음수면 모든 페이지 — 단 `ocr_max_pages` 상한 준수)

### A.2 PDF 다페이지 래스터라이저
`ai-worker/app/capabilities/multimodal/capability.py`:
- 기존 `_default_rasterize_pdf_first_page`를 삭제하지 말고 유지 (이미지 입력 경로에서
  여전히 유효). 옆에 `_default_rasterize_pdf_pages(pdf_bytes, dpi, max_pages) -> list[(page_no, png_bytes)]`
  신규 함수 추가
- Capability 생성자의 `pdf_rasterizer` 주입 파라미터 시그니처를 확장:
  `Callable[[bytes, int, int], List[Tuple[int, bytes]]]` (max_pages 인자 추가)
- 구버전 테스트 호환을 위해 old signature도 accept 하는 shim은 만들지 말 것 —
  기존 테스트가 주입하는 `pdf_rasterizer`만 업데이트

### A.3 Stage B 루프
`_run_vision_stage`:
- PDF 경로에서 `_default_rasterize_pdf_pages(..., max_pages=config.max_vision_pages)` 호출
- 각 페이지에 대해 `vision_provider.describe_image(png, page_number=n)` 실행
- 실패 페이지는 warning으로 수집하되 전체 실패로 만들지 말 것
- 반환 타입을 `list[VisionDescriptionResult]`로 확장 — 기존 단일 결과는 첫 element
- 이미지 입력은 기존과 동일하게 단일 페이지 결과 리스트 반환

### A.4 Fusion 확장
`ai-worker/app/capabilities/multimodal/fusion.py::build_fusion`:
- `vision: VisionDescriptionResult | None` → `vision_pages: list[VisionDescriptionResult]`로 변경
- `fused_context` 의 "Visual description" 섹션은 페이지별 bullet으로 렌더:
  ```
  ### Visual description (3 pages)
  - [page 1] {caption}
    - {detail1}
  - [page 2] ...
  ```
- 빈 리스트 = vision 없음으로 취급 (기존 None 경로 재현)

### A.5 Artifact 스키마
`VISION_RESULT` JSON envelope 구조 변경:
```json
{
  "filename": "...",
  "mimeType": "application/pdf",
  "kind": "pdf",
  "pageCount": 3,
  "pages": [
    { "pageNumber": 1, "provider": "...", "caption": "...", "details": [...], "latencyMs": ... },
    ...
  ],
  "warnings": [...],
  "available": true
}
```
- 이미지 입력도 `pages: [{ pageNumber: 1, ... }]` 단일 요소 리스트로 통일

### A.6 테스트
- `test_multimodal_capability.py` 기존 케이스 업데이트 (새 스키마 반영)
- 신규 케이스: 3페이지 PDF → VISION_RESULT.pages 3개 확인
- 신규 케이스: 1/3 페이지 vision 실패 → 전체는 success, warnings에 실패 기록
- 기존 이미지 입력 경로 회귀 금지
- 기존 `test_fusion.py` 있다면 업데이트

## 작업 B — S3/MinIO Artifact Storage Adapter

### B.1 의존성
`core-api/pom.xml`:
- `software.amazon.awssdk:s3` 추가 (AWS SDK v2). 버전은 Spring Boot BOM과 충돌하지 않는 최신 안정판

### B.2 새 패키지
`core-api/src/main/java/com/aipipeline/coreapi/storage/adapter/out/s3/`:
- `S3StorageProperties.java` — `@ConfigurationProperties(prefix="aipipeline.storage.s3")`
  필드: `endpoint` (optional, MinIO용), `region`, `bucket`, `accessKey`, `secretKey`, `forcePathStyle: boolean`
- `S3ArtifactStorageAdapter.java` — `ArtifactStoragePort` 구현
  - `store(...)`: `PutObjectRequest` + SHA-256 체크섬 계산 (업로드 전), storageUri: `s3://{bucket}/{key}`
  - `openForRead(storageUri)`: scheme 파싱 → `GetObjectRequest`, `InputStream` 반환
  - `generateDownloadUrl(artifactId)`: `S3Presigner`로 15분 유효 presigned URL 반환
  - Local adapter와 동일한 `StoredObject(storageUri, sizeBytes, checksumSha256)` 반환 shape
- `S3StorageConfiguration.java` — `@ConditionalOnProperty("aipipeline.storage.backend", havingValue="s3")`
  - 빈 등록: `S3Client`, `S3Presigner`, `S3ArtifactStorageAdapter`
- Local adapter의 기존 `@ConditionalOnProperty` 또는 default bean 설정을 `backend=local` 조건으로 가드

### B.3 application 설정
`application.yml` 기본값:
```yaml
aipipeline:
  storage:
    backend: local  # local | s3
    s3:
      endpoint: ${AIPIPELINE_S3_ENDPOINT:}
      region: ${AIPIPELINE_S3_REGION:us-east-1}
      bucket: ${AIPIPELINE_S3_BUCKET:aipipeline-artifacts}
      access-key: ${AIPIPELINE_S3_ACCESS_KEY:}
      secret-key: ${AIPIPELINE_S3_SECRET_KEY:}
      force-path-style: ${AIPIPELINE_S3_FORCE_PATH_STYLE:true}
```

### B.4 Worker 측 resolver
`ai-worker/app/storage/resolver.py`:
- `s3://` scheme 처리 추가 — core-api가 넘겨준 presigned URL이 있을 때는 그걸 우선 사용,
  없으면 boto3로 직접 GET (dev 편의)
- `boto3` 를 optional dep로 lazy import. 미설치 시 에러 메시지는 친절하게
- Local 경로 기존 동작 회귀 금지

### B.5 docker-compose / bootstrap
`docker-compose.yml`:
- `minio` profile이 이미 존재한다면 버킷 자동 생성 init 컨테이너 추가 (`mc mb`)
- 없다면 새로 추가:
  ```yaml
  minio:
    image: minio/minio:latest
    profiles: ["minio"]
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio-data:/data"]
  minio-bootstrap:
    image: minio/mc:latest
    profiles: ["minio"]
    depends_on: [minio]
    entrypoint: >
      sh -c "sleep 3 && mc alias set local http://minio:9000 minioadmin minioadmin &&
             mc mb -p local/aipipeline-artifacts"
  ```
- 기존 `redis`/`postgres` profile 절대 건드리지 말 것

### B.6 테스트
- `S3ArtifactStorageAdapterTest` — LocalStack 또는 Testcontainers MinIO 사용
  (가능하면 Testcontainers `minio/minio:latest`). 환경이 없으면 `@Disabled`로 마크하고 이유 기록
- `StoredObject` 반환 shape 동일성 검증
- `generateDownloadUrl` → HTTP GET으로 바이트 다운로드 검증

### B.7 문서
- `docs/architecture.md` "Storage contract" 섹션에 S3 scheme 및 backend 선택 방법 추가
- `README.md` "Run it"에 minio profile 사용 예시 추가:
  ```bash
  docker compose --profile minio up -d minio minio-bootstrap
  AIPIPELINE_STORAGE_BACKEND=s3 \
  AIPIPELINE_S3_ENDPOINT=http://localhost:9000 \
  AIPIPELINE_S3_ACCESS_KEY=minioadmin AIPIPELINE_S3_SECRET_KEY=minioadmin \
  mvn -pl core-api spring-boot:run
  ```

## 작업 C — Demo Script

### C.1 신규 파일
`scripts/demo.py`:
- 순수 Python 단일 스크립트, 외부 의존성 최소 (`httpx` + `rich` 정도는 OK,
  없으면 `requests`)
- 단계:
  1. 헬스 체크 — core-api 기동 여부 확인, 실패 시 친절한 설명 출력
  2. 샘플 PDF 준비 — `scripts/assets/demo_sample.pdf`가 없으면 reportlab으로 즉석 생성
     (한국어+영어 혼용, 표 하나, 도형 하나 포함)
  3. `POST /api/v1/jobs` 멀티파트 업로드 (capability=MULTIMODAL, text=샘플 한국어 질문)
  4. 폴링: 1초 간격으로 `GET /api/v1/jobs/{id}` — 최대 60초
  5. 완료 시 `GET /api/v1/jobs/{id}/result`로 artifact 목록 수신
  6. FINAL_RESPONSE, VISION_RESULT(페이지 캡션), RETRIEVAL_RESULT(top-3) 를
     섹션별로 예쁘게 print (rich 설치 시 Panel/Table 사용)
- Internal secret은 worker만 쓰므로 여기선 불필요 (public API 사용)
- `--vision-provider {heuristic|claude}` 오버라이드는 core-api에 전달 불가 (worker 설정) —
  스크립트에서 "현재 active한 provider를 보여달라"는 헬스 엔드포인트가 없다면 만들지 말 것,
  대신 출력에서 VISION_RESULT.provider 필드를 표시

### C.2 헬스 엔드포인트 (optional, skip 가능)
이미 `GET /api/v1/system/capabilities` 같은 게 있으면 그걸 사용. 없어도 데모는 동작해야 함.
새 엔드포인트 추가는 하지 말 것 (scope 벗어남).

### C.3 샘플 asset
`scripts/assets/demo_sample.pdf`:
- 저장소에 커밋하지 말고 `.gitignore`에 추가 + 스크립트가 없으면 생성
- 없으면 reportlab으로 2페이지 생성 (1페이지: 한국어 본문, 2페이지: 표)

### C.4 README 업데이트
`README.md` 최상단 (Benchmarks 아래)에 새 섹션:
```markdown
## One-command demo

```bash
# (백엔드 기동 후)
python scripts/demo.py
```

Sample output:
```
━━━━ Multimodal demo ━━━━
[1/5] health check: core-api alive on :8080
[2/5] generating sample PDF at scripts/assets/demo_sample.pdf
[3/5] submitted job e7c2... (MULTIMODAL)
[4/5] job SUCCEEDED in 4.8s
[5/5] result:
  Vision (page 1): "A bilingual report cover page..."
  Retrieval top-3:
    1. [doc-xxx#section] score=0.821 ...
  Answer:
    이 문서는 ...
```
```

### C.5 테스트
- `ai-worker/tests/` 에 넣지 말 것. `scripts/test_demo.py` 또는 skip.
- 대신 `scripts/demo.py --self-test` 플래그로 dry-run 모드 구현 (임포트 + PDF 생성까지만)

## 기존 패턴 준수
- Storage port는 core-api 의 기존 `ArtifactStoragePort` 정확히 구현. 새 인터페이스 만들지 말 것
- Worker 측 `resolver.py`는 기존 scheme switch 패턴 유지
- Multi-page vision 확장은 기존 VisionDescriptionResult dataclass 수정하지 않음 —
  리스트로 감싸는 방식
- 새 bean은 `@ConditionalOnProperty`로 backend 선택 (런타임에 둘 다 안 뜨게)

## 수용 기준
- `pytest ai-worker/tests/ -q` 전부 통과
- `cd core-api && mvn test` 전부 통과 (S3 테스트가 Testcontainers 못 찾으면 @Disabled + 이유 로그)
- `docker compose --profile minio up -d minio minio-bootstrap` 후 backend=s3로 기동 → e2e smoke 통과
- `python scripts/demo.py --self-test` 성공
- backend=local로 기동 → 기존 동작 완전 회귀 없음 (`scripts/e2e_smoke.py` 통과)
- `docs/architecture.md` 의 "What's deferred" 테이블에서 "MinIO / S3 storage adapter" 항목 "shipped" 로 마크
- README에 minio profile 사용 예시 + demo 출력 예시 커밋

## 비목표
- 다른 VLM/LLM 추가
- Presigned URL에 권한 세분화
- 멀티 버킷 / 버킷별 IAM
- S3 lifecycle policy 관리
- Worker 측에서 S3로 직접 upload (현재는 core-api가 유일한 스토리지 게이트웨이 — port 원칙 유지)
- 데모 스크립트를 TUI/웹 UI로 확장
- 페이지별 vision를 multimodal RAG retrieval에 직접 배선 (프롬프트 2의 cross-modal work로 끝)
