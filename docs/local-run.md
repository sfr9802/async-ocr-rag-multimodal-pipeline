# 로컬 실행 가이드

이 문서는 단일 개발자 머신에서 전체 파이프라인을 실행하는 과정을
다룹니다. 동시에 4개가 살아있어야 합니다: PostgreSQL, Redis, core-api,
그리고 ai-worker. 다섯 번째 선택적 액터는 테스트 frontend (브라우저에서
여는 정적 HTML 파일).

## 버전

| 컴포넌트       | 버전                                       |
|----------------|--------------------------------------------|
| Java           | 21 (LTS)                                   |
| Spring Boot    | 4.0.3                                      |
| Maven          | 3.9+                                       |
| Python         | 3.12 (3.13 도 동작)                        |
| PostgreSQL     | 18                                         |
| Redis          | latest (phase 1 에서 7.x, hard floor 없음) |

## 사전 준비

- **Java 21** 이 `PATH` 에. 확인: `java -version` 이 `21.x.x` 표시.
- **Maven 3.9+**: `mvn -v` 가 Maven 3.9 이상 + Java 21 보고. Maven 이
  없고 miniconda 가 있다면: `conda install -c conda-forge maven -y`.
  그 외에는 https://maven.apache.org/download.cgi 에서 받아 `bin/` 을
  PATH 에 추가.
- **Python 3.12** (또는 3.13): `python --version`.
- **Docker Desktop** 또는 `docker compose` v2 가 호환되는 엔진.
- 첫 빌드 시 Maven 로컬 저장소용 약 1 GB 여유 디스크.

## 로컬 인프라

기본 로컬 실행은 repo의 compose가 PostgreSQL과 Redis를 함께 소유합니다.
PostgreSQL은 호스트 `5433`에 노출해서 이미 `5432`를 쓰는 다른 로컬 DB와
충돌하지 않게 둡니다.

```bash
cd <repo-root>
docker compose up -d
docker compose ps
# aipipeline-postgres 와 aipipeline-redis 가 healthy 로 표시되어야 함
```

기본 core-api/worker 설정도 이 compose PostgreSQL을 바라봅니다:

- core-api: `jdbc:postgresql://localhost:5433/aipipeline`
- ai-worker RAG DSN: `host=localhost port=5433 dbname=aipipeline user=aipipeline password=aipipeline_pw`

### 기존 PostgreSQL 재사용이 필요한 경우

호스트에 이미 다른 postgres 컨테이너(예: RIA 의 `ria-postgres`가 5432
포트에)가 있고 그 컨테이너만 쓰고 싶다면, compose에서는 Redis만 띄우고
앱 DB 설정을 외부 PostgreSQL로 override 합니다. 이 모드는 AI 플랫폼
데이터를 그 컨테이너 안의 **별도 `aipipeline` 데이터베이스**에 둡니다.

1회 부트스트랩 (격리된 유저 + 데이터베이스 생성):

```bash
# 'ria-postgres' / -U ria 를 실제 컨테이너와 superuser 로 교체
docker exec ria-postgres psql -U ria -d ria_core -c \
  "CREATE USER aipipeline WITH PASSWORD 'aipipeline_pw';"

docker exec ria-postgres psql -U ria -d ria_core -c \
  "CREATE DATABASE aipipeline OWNER aipipeline;"

docker exec ria-postgres psql -U ria -d ria_core -c \
  "GRANT ALL PRIVILEGES ON DATABASE aipipeline TO aipipeline;"
```

확인:

```bash
docker exec ria-postgres psql -U aipipeline -d aipipeline \
  -c "SELECT current_database(), current_user;"
#   current_database | current_user
#  ------------------+--------------
#   aipipeline       | aipipeline
```

이후 우리 compose 에서 Redis만 시작:

```bash
cd <repo-root>
docker compose up -d redis
```

그리고 core-api/worker 시작 전에 다음처럼 외부 DB를 가리키도록 설정:

```bash
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/aipipeline
AIPIPELINE_WORKER_RAG_DB_DSN="host=localhost port=5432 dbname=aipipeline user=aipipeline password=aipipeline_pw"
```

## 전체 스택 시작

4개의 터미널 또는 4개의 tmux 페인. 처음만 순서가 중요합니다: redis 가
core-api 가 publish 시도하기 전에 존재해야 하고, core-api 가 worker 가
claim 시도하기 전에 존재해야 합니다.

### 터미널 1 — 인프라

```bash
cd <repo-root>
docker compose up -d
```

### 터미널 2 — core-api

```bash
cd core-api
mvn spring-boot:run
```

기대되는 로그 끝부분:

```
INFO  o.f.core.internal.command.DbMigrate  : Successfully applied 1 migration
      to schema "aipipeline", now at version v1
INFO  .c.s.a.o.l.LocalFilesystemStorageAdapter : Local storage root: .../local-storage
INFO  o.s.boot.tomcat.TomcatWebServer    : Tomcat started on port 8080 (http)
INFO  c.aipipeline.coreapi.CoreApiApplication : Started CoreApiApplication in ~3 s
```

다른 셸에서 health 확인: `curl -s http://localhost:8080/actuator/health`
가 `{"status":"UP", ...}` 를 출력해야 함.

### 터미널 3 — ai-worker

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

기대되는 로그 끝부분:

```
INFO [__main__] Starting worker id=worker-local-1 core_api=http://localhost:8080
INFO [app.queue.redis_consumer] Queue consumer started key=aipipeline:jobs:pending block_timeout=5s
```

### 터미널 4 — 스모크 테스트

```bash
cd <repo-root>
python scripts/e2e_smoke.py
```

기대 출력:

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

성공 시 exit code 0.

## 선택: 브라우저 frontend

`frontend/index.html` 을 브라우저에서 직접 엽니다. Phase 1 은 CORS 가
설정되어 있지 않아, 브라우저가 `file://` 페이지에서의 cross-origin
호출을 차단하면 다음 중 하나로 우회 가능:

- 로컬 정적 서버로 호스팅 (예: `frontend/` 안에서
  `python -m http.server 8000` 후 `http://localhost:8000/` 열기), 또는
- IDE 에서 HTML artifact 로 실행.

## 데이터가 실제로 들어왔는지 확인

스모크 테스트 후:

```bash
docker exec aipipeline-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT id, capability, status FROM job ORDER BY updated_at DESC LIMIT 5;"

docker exec aipipeline-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT job_id, role, type, size_bytes FROM artifact ORDER BY created_at DESC LIMIT 10;"
```

각 성공한 job 은 INPUT 행 1개 + OUTPUT 행 1개 **정확히** 가져야 합니다.
job 당 OUTPUT 행 2개는 더블 쓰기 회귀 — 버그 등록.

## 종료

- **터미널 2와 3**: `Ctrl+C`.
- **compose 인프라**: `docker compose down` (또는 그대로 두기 — 무해).
  PostgreSQL 볼륨까지 깨끗하게 지우려면 `docker compose down -v`.
- **외부 PostgreSQL을 재사용한 경우**, 그 DB 안의 aipipeline 데이터만 삭제:
  ```bash
  docker exec ria-postgres psql -U aipipeline -d aipipeline -c \
    "TRUNCATE TABLE artifact, job CASCADE;"
  ```

## 트러블슈팅

### core-api

- **`connection refused` localhost:5433 에 대해** — compose PostgreSQL이
  실행 중이 아님. `docker compose up -d` 후 재시도. 외부 PostgreSQL을
  재사용한다면 `SPRING_DATASOURCE_URL`이 의도한 포트를 가리키는지 확인.
- **Flyway 에러 `Schema validation: missing table [artifact]`** —
  Hibernate 전에 Flyway 가 돌지 않았음. Spring Boot 4.0 에서는 POM 에
  `spring-boot-starter-flyway` 가 필요 (`flyway-core` 만으로는 부족).
  배포된 pom.xml 에는 이미 들어 있음; POM 을 커스터마이즈했다면 starter
  가 있는지 확인.
- **`No qualifying bean of type ObjectMapper`** — Spring Boot 4.0 은
  `starter-web` 만으로는 더 이상 singleton ObjectMapper 를 자동으로
  publish 하지 않음. 배포된 코드는 이를
  `core-api/.../common/web/JacksonConfig.java` 에 명시적으로 제공. 삭제
  하지 마세요.
- **Port 8080 이미 사용 중** — `SERVER_PORT=8081` (또는 사용 가능한 다른
  포트) 설정하고 worker 의 `AIPIPELINE_WORKER_CALLBACK_BASE_URL` 도
  맞춤.

### ai-worker

- **시작 시 `Redis ping failed`** — Redis 가 실행 중이 아니거나 URL 이
  잘못됨. `docker compose up -d` 후 재시도.
- **Worker 가 job 을 처리하지만 callback 이 500 + `ValidationError:
  UploadResponse missing field artifactId`** — 더 오래된 worker 를 더 새
  core-api 에 (또는 그 반대). pull 후 둘 다 재시작. 업로드 엔드포인트는
  phase 1 에서 의도적으로 `storageUri`, `sizeBytes`, `checksumSha256` 만
  반환 — artifact id 없음.
- **`FileNotFoundError: Artifact missing on shared disk`** — core-api 와
  worker 가 다른 local-storage 루트를 보고 있음.
  `AIPIPELINE_STORAGE_LOCAL_ROOT_DIR` (core-api) 와
  `AIPIPELINE_WORKER_LOCAL_STORAGE_ROOT` (worker) 를 같은 절대 경로로
  설정. 배포된 기본값은 각 프로세스가 자기 디렉토리에서 실행될 때
  `{repo-root}/local-storage` 로 해소됨.

### 스모크 테스트

- **`UnicodeEncodeError: cp949 codec can't encode character`** — phase
  1.1 에서 수정됨. 스크립트는 더 이상 비-ASCII 문자를 사용하지 않음.
  이게 보이면 오래된 checkout 사용 중.
- **Job 이 영원히 `QUEUED` 에 머무름** — worker 가 실행 중이 아니거나
  잘못된 Redis / 잘못된 큐 키를 가리키고 있음. worker 로그 확인;
  `AIPIPELINE_WORKER_REDIS_URL` 과
  `AIPIPELINE_WORKER_QUEUE_PENDING_KEY` 가 core-api 측과 일치하는지
  확인.

### 파이프라인 trace artifact 검사

OCR 와 MULTIMODAL job 은 정확히 어떤 stage 가 돌았는지, 어떤 게 폴백
했는지, 어떤 게 실패했는지 알려주는 정규화된 **stage 단위 trace** 를
들고 있습니다. 풀 스키마, 에러 코드 표, 예시 페이로드는
`docs/architecture.md` 의 "파이프라인 trace 와 실패 리포팅" 참조.

**FAILED job (MULTIMODAL) 에서**, trace 요약은 job 의 `errorMessage`
필드에 직접 접혀 들어가므로 아무것도 다운로드하지 않고도 볼 수 있음:

```bash
curl -s http://localhost:8080/api/v1/jobs/$JOB_ID | python -c \
  "import json, sys; j=json.load(sys.stdin); print(j.get('errorCode'), '|', j.get('errorMessage'))"
# → MULTIMODAL_ALL_PROVIDERS_FAILED | ... | trace: classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms,fallback) vision:fail(VISION_VLM_TIMEOUT,5ms,fallback) fusion:skipped retrieve:skipped generate:skipped
```

**SUCCEEDED OCR job 에서**, trace 는 `OCR_RESULT` artifact 안에 있음:

```bash
# 1. OCR_RESULT 의 access URL 가져오기
OCR_URL=$(curl -s http://localhost:8080/api/v1/jobs/$JOB_ID/result \
  | python -c "import json, sys; outs = json.load(sys.stdin)['outputs']; \
    print(next(o['accessUrl'] for o in outs if o['type']=='OCR_RESULT'))")

# 2. trace 다운로드 + pretty-print
curl -s "http://localhost:8080$OCR_URL" | python -m json.tool | grep -A 30 '"trace"'
```

**SUCCEEDED MULTIMODAL job 에서** (`emit_trace=true` 일 때), trace 는
별도의 `MULTIMODAL_TRACE` artifact:

```bash
# worker 시작 전에 trace 발행 활성화:
export AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true

# MULTIMODAL job 제출 후 SUCCEEDED 까지 polling:
TRACE_URL=$(curl -s http://localhost:8080/api/v1/jobs/$JOB_ID/result \
  | python -c "import json, sys; outs = json.load(sys.stdin)['outputs']; \
    print(next(o['accessUrl'] for o in outs if o['type']=='MULTIMODAL_TRACE'))")
curl -s "http://localhost:8080$TRACE_URL" | python -m json.tool
```

**trace `summary` 필드로 빠른 진단:**

| 요약 패턴                                         | 진단                                                    |
|---------------------------------------------------|---------------------------------------------------------|
| `ocr:ok vision:ok ... generate:ok`                | 깨끗한 성공, 이슈 없음                                  |
| `vision:fail(VISION_...,fallback)`                | Vision provider 다운; 파이프라인이 OCR 만으로 진행      |
| `ocr:warn(OCR_EMPTY_TEXT,...,fallback)`           | 입력에 읽을 텍스트 없음; vision caption 만으로 grounding |
| `ocr:fail(...) vision:fail(...) fusion:skipped`   | 두 provider 모두 실패 — 종단 `MULTIMODAL_ALL_PROVIDERS_FAILED` |
| `retrieve:fail(MULTIMODAL_RETRIEVAL_FAILED,...)`  | FAISS 인덱스 손상 또는 도달 불가; OCR/vision 은 정상    |
| `generate:fail(MULTIMODAL_GENERATION_FAILED,...)` | Retrieval 성공 후 generator 가 raise — 드문 경우        |

## Windows 전용 노트

- git-bash 또는 PowerShell 사용. `cmd.exe` 도 작동하지만 이 문서의 일부
  one-liner 는 POSIX shell 가정.
- .env 의 파일시스템 경로는 Windows 에서도 forward slash 사용 (Spring 과
  Python 모두 처리 가능).
- git-bash 에서 멈춘 core-api 또는 worker 를 강제 종료:
  `taskkill //F //PID <pid>` (git-bash 인자 escape 를 위한 `//` 두 개에
  주의).
- 포트 listing: `netstat -an | findstr LISTEN` (Windows), 또는
  `netstat -an | grep LISTEN` (git-bash).

## Phase 2: RAG capability

Phase 2 는 worker 에 진짜 text-RAG capability 를 추가합니다. 파이프라인
자체는 변경 없음 — worker 의 capability 레지스트리에 두 번째 항목이
추가되어 job 의 `capability` 필드로 선택될 뿐.

### 사전 준비

RAG capability 는 시작 시 FAISS 인덱스와 sentence-transformers 모델을
worker 프로세스에 로드합니다. 그 전에 다음이 필요:

1. **core-api 가 V2 Flyway 마이그레이션을 돌리도록.** phase 2 코드를
   pull 한 뒤 core-api 를 재시작하면 됨 — Flyway 가 자동으로
   `V2__ragmeta_schema.sql` 을 적용해 기존 `aipipeline` 데이터베이스 안에
   `ragmeta` 스키마를 생성.
2. **FAISS 인덱스를 1회 빌드.** worker 는 인덱스가 디스크에 없으면 RAG
   capability 등록을 거부합니다. MOCK 은 여전히 등록되므로 phase-1 흐름
   은 계속 동작:
   ```bash
   cd ai-worker
   python -m scripts.build_rag_index --fixture
   ```
   이는 `fixtures/anime_sample.jsonl` 아래 커밋된 8문서 anime 픽스처를
   사용. 첫 실행은 기본 임베딩 모델 (`BAAI/bge-m3`, ~2.3 GB, 다국어,
   1024 차원) 을 HuggingFace 캐시에 다운로드. 이후 실행은 빠름. 디스크
   나 대역폭이 부족하면 더 작은 모델로 override — 아래
   [임베딩 모델 변경](#임베딩-모델-변경) 참조.
3. **Worker 재시작.** Worker 는 시작 시 FAISS 인덱스 + 임베딩 모델을
   메모리에 로드하므로, rebuild 는 worker 재시작이 필요. 다음과 같은
   출력이 보여야 함:
   ```
   RAG init: configured_model=BAAI/bge-m3 query_prefix='' passage_prefix='' index_dir=../rag-data top_k=5
   Loaded FAISS index v-... (model=BAAI/bge-m3, dim=1024, vectors=...)
   Retriever readiness check: configured_model='BAAI/bge-m3' index_model='BAAI/bge-m3' configured_dim=1024 index_dim=1024 index_version=v-... chunk_count=...
   Retriever ready: model=BAAI/bge-m3 dim=1024 index_version=v-...
   RAG capability registered.
   Active capabilities: ['MOCK', 'RAG']
   ```
   Worker 가 `RAG capability NOT registered (RuntimeError:
   Embedding MODEL mismatch ...)` 를 출력하면 설정된 모델이 인덱스를
   빌드한 모델과 일치하지 않는다는 뜻. 재빌드 + 재시작;
   [임베딩 모델 마이그레이션](#임베딩-모델-마이그레이션) 참조.

### RAG job 제출

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"capability":"RAG","text":"which anime is about an old fisherman feeding stray harbor cats"}'
```

`/api/v1/jobs/{id}` 가 `SUCCEEDED` 가 될 때까지 polling, 그 다음
`/api/v1/jobs/{id}/result` 가져오기. `outputs` 배열에 두 artifact:

- `RETRIEVAL_RESULT` — query, topK, 인덱스 버전, 임베딩 모델, 검색된 모든
  chunk 의 score + text 를 담은 JSON payload.
- `FINAL_RESPONSE` — 검색된 chunk 에 grounding 되고 source doc id 를
  인용한 markdown 답변.

### Capability 전환

Phase 2 는 MOCK 등록을 유지. MOCK 을 테스트하려면 요청 본문의
`capability` 를 `"MOCK"` 으로 변경. 두 capability 는 파이프라인을
공유하고 worker 측 디스패치로만 구분됩니다.

### 실제 데이터셋 인덱싱

CLI 는 port/rag 스키마의 어떤 JSONL 파일이든 `--input <path>` 로
받습니다:

```bash
python -m scripts.build_rag_index \
  --input D:/port/rag/app/scripts/namu_anime_v3.jsonl \
  --index-version prod-2026-04 \
  --notes "full namu_anime_v3 dataset rebuild"
```

큰 데이터셋은 임베딩에 더 오래 걸리지만 worker 측 serving 경로는 phase-2
규모에서 데이터셋 크기에 무관 (IndexFlatIP 는 어떤 수의 벡터에서도 정확).

### 임베딩 모델 변경

기본은 `BAAI/bge-m3` — 다국어, 1024 차원, query/passage prefix 없이
훈련되어 둘 다 빈 값이 기본. 모델을 바꾸려면 인덱싱 CLI 와 worker 를
실행하기 **전에** `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` (그리고 새
모델이 prefix 가 필요하다면 prefix 변수도) 설정. 인덱스 사이드카는 어떤
모델이 빌드했는지 기록하고, worker 는 이제 **엄격한** 런타임 체크를
수행: 설정된 모델 이름이 `build.json` 의 것과 byte 단위로 동일하지
않으면 RAG capability 등록이 명확한 `Embedding MODEL mismatch`
RuntimeError 로 실패. MOCK 은 여전히 등록됨. 의도적입니다 — 두 다른
모델이 vector dimension 을 공유하면서 retrieval 품질을 조용히 손상시킬
수 있기 때문.

지원 옵션:

| 모델                                     | 차원 | Prefix                       | 비고                                                  |
|------------------------------------------|------|------------------------------|-------------------------------------------------------|
| `BAAI/bge-m3` (기본)                     | 1024 | 없음 (빈 값)                 | 다국어, ~2.3 GB 다운로드, 박스에서 가장 강함          |
| `sentence-transformers/all-MiniLM-L6-v2` | 384  | 없음 (빈 값)                 | 영어 전용, ~80 MB, 가장 빠름                          |
| `intfloat/multilingual-e5-small`         | 384  | `query: ` / `passage: `      | 다국어, ~120 MB, prefix 필요                          |

E5 패밀리 예시:

```
AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
AIPIPELINE_WORKER_RAG_EMBEDDING_PREFIX_QUERY="query: "
AIPIPELINE_WORKER_RAG_EMBEDDING_PREFIX_PASSAGE="passage: "
```

### 임베딩 모델 마이그레이션

`AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` 변경은 **2단계 필수 시퀀스**.
어느 단계든 빠뜨리면 worker 가 시작 시 RAG 등록을 거부합니다.

1. **새 모델로 FAISS 인덱스 재빌드.**
   기존 인덱스는 in-place 로 덮어쓰여지고; 새 `build.json` 이 CLI 가
   실행된 정확한 모델 이름을 기록. 이는 마지막 `Ingest done ...` 로그
   라인에 echo 됨.
   ```bash
   cd ai-worker
   # fixture 재빌드 (빠름, 스모크 테스트용)
   python -m scripts.build_rag_index --fixture

   # 실제 데이터셋 재빌드
   python -m scripts.build_rag_index \
     --input D:/port/rag/app/scripts/namu_anime_v3.jsonl \
     --index-version prod-bge-m3 \
     --notes "bge-m3 migration"
   ```

2. **Worker 재시작.**
   장수명 `RagCapability` 는 메모리에 옛 FAISS 인덱스와 embedder 를
   잡고 있고, rebuild 는 hot-reload 되지 않음. Worker 종료 (`Ctrl-C`)
   후 재시작:
   ```bash
   cd ai-worker
   python -m app.main
   ```
   깨끗한 시작에서 `RAG capability registered.` 와 새 model/dim 의
   readiness 로그 라인이 보여야 함. 대신 `Embedding MODEL mismatch` 가
   보이면 인덱싱 CLI 와 worker 가 다른
   `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` 값으로 실행됐다는 뜻 — 환경
   변수 수정 후 1단계 재실행.

### ragmeta 스키마

```
ragmeta.documents      (doc_id, title, source, category, metadata_json, created_at, updated_at)
ragmeta.chunks         (chunk_id, doc_id, section, chunk_order, text, token_count,
                        faiss_row_id, index_version, extra_json, created_at)
ragmeta.index_builds   (index_version, embedding_model, embedding_dim, chunk_count,
                        document_count, faiss_index_path, notes, built_at)
```

내용 들여다보기:

```bash
docker exec aipipeline-postgres psql -U aipipeline -d aipipeline -c "\dt ragmeta.*"
docker exec aipipeline-postgres psql -U aipipeline -d aipipeline -c \
  "SELECT index_version, embedding_model, chunk_count FROM ragmeta.index_builds ORDER BY built_at DESC;"
```

## Phase 2: OCR capability

OCR capability 는 **INPUT_FILE** artifact (PNG, JPEG, PDF) 를 받아 두
출력 artifact 를 생성합니다:

- **OCR_TEXT** — plain UTF-8 추출 텍스트. 나중의 RAG job 이 INPUT_TEXT 로
  소비할 수 있는 동일한 모양 — 미래 phase 에서 OCR→RAG 체이닝이 자연
  스럽게 구성되도록 설계됨.
- **OCR_RESULT** — `filename`, `mimeType`, `kind`, `engineName`,
  `pageCount`, `textLength`, `avgConfidence`, 페이지별 롤업, 평탄한
  `warnings` 배열을 담은 JSON envelope. 계약은 스키마 churn 없이 앞으로
  들고 갈 만큼 안정적.

Phase 1 OCR 은 의도적으로 "안정적 artifact 생성" 에서 멈춤. multimodal
추론, 이미지 임베딩, VLM 답변 생성, OCR+RAG 체이닝 등은 **없음** —
명시적인 non-goal.

### 사전 준비

기본 provider 는 **pytesseract** 경유 **Tesseract**, **PyMuPDF** 가
PDF 페이지 순회 처리. 셋업은 3단계, 모두 로컬:

1. **Tesseract 바이너리 설치.**
   - **Windows**: https://github.com/UB-Mannheim/tesseract/wiki 에서
     인스톨러를 받아 `C:\Program Files\Tesseract-OCR\` 에 설치되도록.
     그 폴더를 `PATH` 에 추가하거나
     `AIPIPELINE_WORKER_OCR_TESSERACT_CMD` 를 `tesseract.exe` 의 절대
     경로로 설정.
   - **Linux**: `sudo apt install tesseract-ocr` (또는 배포판 동등).
     `tesseract` 바이너리가 자동으로 PATH 에 있게 됨.
   - **macOS**: `brew install tesseract`.

2. **Python 래퍼 설치.**
   ```bash
   cd ai-worker
   pip install -r requirements.txt
   ```
   `pytesseract`, `pymupdf` 를 가져오고 `Pillow` 를 pin. 어떤 것도
   torch/transformers 를 pull 하지 않으므로 cold venv 에서도 설치는
   빠름.

3. **(선택) 추가 언어팩 설치.** Tesseract 의 `eng` 팩은 기본 설치와
   함께 옴; 그 외에는 추가 다운로드 필요. 한국어, 일본어, 중국어 등은
   해당하는 traineddata 설치:
   - **Windows 인스톨러**: 인스톨러 UI 에서 언어 선택.
   - **Linux**: `sudo apt install tesseract-ocr-kor tesseract-ocr-jpn`
     등.
   - **macOS**: `brew install tesseract-lang` (모두 설치).

   그런 다음 `AIPIPELINE_WORKER_OCR_LANGUAGES` 를 `+` 로 연결한 문자열
   로 설정:
   ```
   AIPIPELINE_WORKER_OCR_LANGUAGES=eng+kor
   ```
   요청한 언어팩이 누락되면 worker 는 시작 시 OCR capability **등록을
   거부** 하고, 어떤 게 누락됐는지 정확히 로그. MOCK 과 RAG 은 영향
   없음.

### OCR 와 함께 worker 시작

Tesseract + Python 래퍼가 설치되면 worker 를 평소대로 시작:

```bash
cd ai-worker
python -m app.main
```

건강한 시작에서 다음과 같은 출력이 보임:

```
OCR init: languages=eng pdf_dpi=200 tesseract_cmd=<PATH> min_conf_warn=40.0 max_pages=100
TesseractOcrProvider ready: tesseract=5.3.3 languages=eng pdf_dpi=200
OCR capability registered.
Active capabilities: ['MOCK', 'OCR', 'RAG']
```

Tesseract 가 설치되지 않았거나 언어팩이 누락되면 다음과 같은 깨끗한
warning 이 보임:

```
OCR capability NOT registered (OcrError: Could not execute the tesseract binary. ...)
Active capabilities: ['MOCK', 'RAG']
```

MOCK 과 RAG 은 여전히 등록되고; 이 상태에서 제출된 OCR job 은 Tesseract
를 설치하고 worker 를 재시작할 때까지 `UNKNOWN_CAPABILITY` 로 실패.

### OCR job 제출

```bash
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=OCR" \
  -F "file=@/path/to/receipt.png"
```

`/api/v1/jobs/{id}` 가 `SUCCEEDED` 가 될 때까지 polling, 그 다음
`/api/v1/jobs/{id}/result` 가져오기. `outputs` 배열에 두 artifact
(`OCR_TEXT` 와 `OCR_RESULT`).

### OCR 설정 knob (환경변수)

| 환경변수                                          | 기본값  | 비고                                                              |
|---------------------------------------------------|---------|-------------------------------------------------------------------|
| `AIPIPELINE_WORKER_OCR_ENABLED`                   | `true`  | `false` 로 설정 시 OCR 등록을 완전히 skip.                        |
| `AIPIPELINE_WORKER_OCR_LANGUAGES`                 | `eng`   | `+` 로 연결한 Tesseract 언어 코드 (`eng+kor`, `eng+jpn`).         |
| `AIPIPELINE_WORKER_OCR_TESSERACT_CMD`             | *unset* | PATH 에 없으면 `tesseract.exe` 의 절대 경로.                      |
| `AIPIPELINE_WORKER_OCR_PDF_DPI`                   | `200`   | 스캔된 PDF 페이지 rasterize 시 사용 DPI.                          |
| `AIPIPELINE_WORKER_OCR_MIN_CONFIDENCE_WARN`       | `40.0`  | 평균 신뢰도가 이 미만일 때 `OCR_RESULT` 에 warning 추가. job 을 **실패시키지 않음**. `0` 으로 설정 시 비활성화. |
| `AIPIPELINE_WORKER_OCR_MAX_PAGES`                 | `100`   | PDF 페이지 수 hard cap. 초과 시 `OCR_TOO_MANY_PAGES` 발생.        |

### 지원 입력 타입

| 종류  | 허용 content type                       | 허용 확장자             | 비고                                                              |
|-------|-----------------------------------------|-------------------------|-------------------------------------------------------------------|
| Image | `image/png`, `image/jpeg`               | `.png`, `.jpg`, `.jpeg` | 단일 프레임. Provider 가 이미지별 평균 신뢰도 보고.               |
| PDF   | `application/pdf`, `application/x-pdf`  | `.pdf`                  | 페이지별. Born-digital 페이지는 PDF text layer 사용 (OCR 없음, 신뢰도 없음); 스캔된 페이지는 `ocr_pdf_dpi` 로 rasterize 후 OCR. |

그 외는 모두 typed `UNSUPPORTED_INPUT_TYPE` 에러로 실패하고 job 의
`errorCode` 필드에 표시.

### 알려진 한계 (phase 1 OCR)

- **이미지 전처리 없음.** raw 바이트를 Tesseract 에 그대로 넘김. low-
  contrast 또는 회전된 스캔에서는 품질이 나쁨. Pillow 기반 deskew/
  threshold 패스가 자연스러운 다음 단계.
- **레이아웃 / bounding box 없음.** `OCR_RESULT` 는 텍스트 + 평균
  신뢰도만 보고. Tesseract 의 word 단위 bounding box 는 표면화되지
  않음.
- **손글씨 없음.** Tesseract 는 활자체에 최적화.
- **PDF 페이지 cap** (`ocr_max_pages`, 기본 100) 은 안전 게이트지 근본
  적인 한계가 아님. 더 필요하면 환경변수로 올림.
- **체이닝 없음.** OCR 출력은 두 독립 artifact. 나중 phase 에서 job A
  의 `OCR_TEXT` 출력을 job B 에 `INPUT_TEXT` 로 먹이는 오케스트레이션
  단계 추가; phase 1 OCR 은 의도적으로 거기서 멈춤.

## Phase 2: MULTIMODAL capability (v1)

MULTIMODAL capability 는 **INPUT_FILE** artifact (PNG, JPEG, PDF) +
**선택적 INPUT_TEXT 사용자 질문**을 받아, OCR + visual-description
provider 를 통과시키고, 두 신호를 retrieval query + grounding context 로
융합한 뒤, 그 context 를 기존 text-RAG retriever + generator 에
넘깁니다. 결과는 4개의 출력 artifact:

- **OCR_TEXT** — OCR stage 의 plain UTF-8 텍스트.
- **VISION_RESULT** — provider 이름, caption, 구조화된 details, 페이지
  메타데이터, warnings, 그리고 vision stage 가 실제로 결과를 생성했는지
  다운스트림 컨슈머가 알 수 있는 `available` 플래그를 담은 JSON
  envelope.
- **RETRIEVAL_RESULT** — 단독 RAG capability 가 이미 emit 하는 것과
  동일한 JSON 스키마 — 기존 컨슈머가 변경 없이 동작.
- **FINAL_RESPONSE** — extractive generator 가 만든 grounded markdown
  답변, fused OCR + vision 신호가 합성된 rank-0 chunk 로 주입됨.

선택적 (기본 off): **MULTIMODAL_TRACE** — 어떤 stage 가 기여했는지,
fusion 메타데이터, stage 별 warning 을 기록하는 JSON trace.
`AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true` 로 활성화.

> **v1 범위 환기.** MULTIMODAL v1 은 진정한 multimodal retrieval 이
> **아닙니다**. 기존 text-RAG retriever 를 재사용; 전용 이미지 임베딩
> 인덱스도, VLM 답변 generator 도 없음. 보류된 항목 전체 목록은
> `docs/architecture.md` 의 "Multimodal v1 한계" 참조.

### 사전 준비

MULTIMODAL 은 **종속** capability. Worker 시작 시:

1. **RAG 가 성공적으로 등록되어야 함.** MULTIMODAL 은 RAG retriever +
   generator 를 재사용. 위의 "Phase 2: RAG capability" 섹션을 따라 FAISS
   인덱스를 빌드하고 RAG 를 먼저 띄움.
2. **OCR 이 성공적으로 등록되어야 함.** MULTIMODAL 은 텍스트 추출에 OCR
   provider 를 재사용. 위의 "Phase 2: OCR capability" 섹션을 따라
   Tesseract 와 PyMuPDF 설치.
3. v1 vision provider 는 `HeuristicVisionProvider` 로, Pillow 외의
   의존성이 없음 (sentence-transformers 가 transitive 로 이미 가져옴).
   추가 설치 단계 없음.

RAG 또는 OCR 중 어느 것이라도 등록 실패하면 MULTIMODAL 은 누락된 부모를
이름으로 명시한 명확한 warning 과 함께 자동 skip. MOCK, RAG, OCR 는 두
경우 모두 영향 없음.

### MULTIMODAL 와 함께 worker 시작

Worker 를 평소대로 시작 — 등록은 자동:

```bash
cd ai-worker
python -m app.main
```

모든 의존성이 갖춰진 건강한 시작에서 다음과 같은 출력이 보임:

```
RAG init: configured_model=BAAI/bge-m3 query_prefix='' passage_prefix='' index_dir=../rag-data top_k=5
...
RAG capability registered.
OCR init: languages=eng pdf_dpi=200 tesseract_cmd=<PATH> min_conf_warn=40.0 max_pages=100
...
OCR capability registered.
MULTIMODAL init: vision_provider=heuristic pdf_vision_dpi=150 emit_trace=False default_question='' (retrieval top-k inherited from RAG=5)
MULTIMODAL capability registered.
Active capabilities: ['MOCK', 'MULTIMODAL', 'OCR', 'RAG']
```

RAG 또는 OCR 등록이 실패하면 MULTIMODAL 은 다음 중 하나의 warning 과
함께 skip:

```
MULTIMODAL capability NOT registered: OCR capability is unavailable. MULTIMODAL v1 reuses the OCR provider — enable and fix OCR first, then restart the worker. MOCK, RAG remain registered.
```

또는

```
MULTIMODAL capability NOT registered: RAG capability is unavailable. MULTIMODAL v1 reuses the RAG retriever + generator to feed the fused OCR + vision context into the existing text-RAG path — enable and fix RAG first, then restart the worker. MOCK, OCR remain registered.
```

### MULTIMODAL job 제출

OCR 에 이미 사용 중인 동일한 multipart 엔드포인트로 파일 + 선택적 text:

```bash
# 이미지 + 사용자 질문
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/invoice.png" \
  -F "text=what is the total amount on this invoice?"

# PDF + 사용자 질문
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/report.pdf" \
  -F "text=summarize the main findings"

# 질문 없는 이미지 — fusion helper 가 중립 기본 retrieval query 선택
curl -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/diagram.jpg"
```

`GET /api/v1/jobs/{id}` 가 `SUCCEEDED` 까지 polling, 그 다음
`GET /api/v1/jobs/{id}/result` 가져오기. `outputs` 배열에 4개 artifact
(`OCR_TEXT`, `VISION_RESULT`, `RETRIEVAL_RESULT`, `FINAL_RESPONSE`),
`AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE=true` 일 때 다섯 번째
`MULTIMODAL_TRACE` 추가.

### 샘플 end-to-end 테스트 흐름

신규 설치가 end-to-end 로 동작하는지 검증하는 정확한 시퀀스:

```bash
# 1. (1회만) 위 RAG 와 OCR 섹션에서 설명한 대로 RAG 인덱스 빌드 + Tesseract 설치.

# 2. 세 터미널에서 인프라 + core-api + worker 시작.
docker compose up -d
cd core-api && mvn spring-boot:run
cd ai-worker && python -m app.main

# 3. 임의의 PNG 또는 PDF 와 선택적 질문으로 MULTIMODAL job 제출.
JOB_ID=$(curl -s -X POST http://localhost:8080/api/v1/jobs \
  -F "capability=MULTIMODAL" \
  -F "file=@/path/to/some_document.png" \
  -F "text=what is written in this image?" \
  | python -c "import json, sys; print(json.load(sys.stdin)['jobId'])")
echo "jobId=$JOB_ID"

# 4. SUCCEEDED 까지 polling.
for i in 1 2 3 4 5; do
  STATUS=$(curl -s http://localhost:8080/api/v1/jobs/$JOB_ID \
    | python -c "import json, sys; print(json.load(sys.stdin)['status'])")
  echo "status=$STATUS"
  [ "$STATUS" = "SUCCEEDED" ] && break
  sleep 1
done

# 5. 결과 가져오기 + artifact 타입 나열.
curl -s http://localhost:8080/api/v1/jobs/$JOB_ID/result \
  | python -c "import json, sys; print([o['type'] for o in json.load(sys.stdin)['outputs']])"
# → ['OCR_TEXT', 'VISION_RESULT', 'RETRIEVAL_RESULT', 'FINAL_RESPONSE']

# 6. grounded 답변을 보기 위해 FINAL_RESPONSE 다운로드.
OUTPUT_URL=$(curl -s http://localhost:8080/api/v1/jobs/$JOB_ID/result \
  | python -c "import json, sys; outs = json.load(sys.stdin)['outputs']; \
    print(next(o['accessUrl'] for o in outs if o['type']=='FINAL_RESPONSE'))")
curl -s "http://localhost:8080$OUTPUT_URL"
```

### MULTIMODAL 설정 knob (환경변수)

| 환경변수                                           | 기본값      | 비고                                                                |
|----------------------------------------------------|-------------|---------------------------------------------------------------------|
| `AIPIPELINE_WORKER_MULTIMODAL_ENABLED`             | `true`      | `false` 로 설정 시 MULTIMODAL 등록을 완전히 skip.                   |
| `AIPIPELINE_WORKER_MULTIMODAL_VISION_PROVIDER`     | `heuristic` | v1 은 `heuristic` 만 지원 — 미래 VLM 용으로 예약.                   |
| `AIPIPELINE_WORKER_MULTIMODAL_PDF_VISION_DPI`      | `150`       | vision provider 용 PDF 1페이지 rasterize 시 DPI.                    |
| `AIPIPELINE_WORKER_MULTIMODAL_EMIT_TRACE`          | `false`     | true 일 때 `MULTIMODAL_TRACE` JSON artifact 도 발행.                |
| `AIPIPELINE_WORKER_MULTIMODAL_DEFAULT_QUESTION`    | (빈 값)     | job 이 INPUT_TEXT 를 제출하지 않을 때의 폴백 사용자 질문.           |

참고: MULTIMODAL retrieval top-k 는 의도적으로
**`AIPIPELINE_WORKER_RAG_TOP_K` 에서 상속**. 동일 FAISS 인덱스에 대한
MULTIMODAL job 과 RAG job 은 같은 retrieval shape 을 봄.

### 알려진 한계 (MULTIMODAL v1)

이것들은 설계상 보류 — v1 의 목표는 파이프라인을 image/PDF 입력에
**여는** 것이지 아직 품질을 쫓는 게 아닙니다:

- **v1 은 진정한 multimodal retrieval 이 아님.** Retriever 는 기존
  text-RAG retriever. 이미지 임베딩 인덱스도, 크로스모달 검색도,
  multimodal FAISS partition 도 없음.
- **v1 은 OCR 텍스트 + visual description 으로 기존 text RAG 을 먹임.**
  모든 multimodal 신호가 fused context 블록에서 텍스트로 retrieval
  stage 에 들어옴.
- **전용 이미지 임베딩 / 크로스모달 retrieval 은 보류.** "수동
  multimodal eval set 큐레이션 + OCR/RAG/MM delta 측정" 다음 phase.
- **기본 vision provider 는 결정적 Pillow 휴리스틱.** orientation,
  brightness, contrast, dominant-channel descriptor 를 emit. 이는 진짜
  VLM (BLIP-2, GPT-4V, Claude Vision, Gemini) 을 꽂을 seam 이지 품질
  바가 아님.
- **다중 페이지 PDF 는 모든 페이지에 OCR 이 돌지만 1페이지만 caption
  됨.** Vision stage 를 페이지별 caption 으로 확장하는 건 단일 config
  knob + loop 만 추가하면 됨.
- **multimodal 품질 평가는 이 단계에서 여전히 대부분 수동.**
  `ai-worker/eval/datasets/multimodal_sample.jsonl` 의 eval stub 은
  forward-looking 한 스키마 제안이지 라이브 harness 가 아님.

### 권장 다음 단계

진정한 multimodal retrieval 또는 진짜 VLM 에 투자하기 전에, 수동
multimodal eval set (30-50 샘플) 을 큐레이션하고 그 위에서 OCR-only /
RAG-only / MULTIMODAL 답변 품질 delta 를 측정하세요. 그 측정이 다음
한계 개선을 (a) 더 나은 OCR 전처리, (b) `VisionDescriptionProvider`
seam 뒤의 진짜 VLM, (c) 전용 이미지 임베딩, 또는 (d) 진짜 LLM
generator 중 어느 쪽으로 해야 할지 알려주고 — 그 결정을 개발자 직관이
아니라 실제 제품 품질에 고정합니다.

## LLM 스택 (Ollama)

Phase 5/6 — 에이전트 라우팅, critique, query rewriting — 은 공유
`LlmChatProvider` seam (`ai-worker/app/clients/llm_chat.py`) 뒤의 chat
스타일 LLM 에 의존. 기본 백엔드는 `gemma4:e2b` 를 실행하는 로컬 Ollama
서비스로, opt-in `llm` 프로파일 아래 compose 파일에 와이어드 됨 (그래야
CPU-only 개발 박스가 완전히 skip 가능).

### LLM 스택 시작

```bash
docker compose --profile llm up -d ollama ollama-bootstrap
```

첫 실행에서 `ollama-bootstrap` 이 `gemma4:e2b` (~5 GB) 를 named 볼륨
`ollama_data` 로 pull 하고 깨끗하게 종료. 이후 실행은 no-op. 확인:

```bash
curl http://localhost:11434/api/tags
# → {"models": [{"name": "gemma4:e2b", ...}]}
```

서비스에 `OLLAMA_KEEP_ALIVE=30m` 이 설정되어 있어 모델이 job 사이에서
resident; worker 도 `/api/chat` 호출마다 `keep_alive=30m` 을 forward
(`AIPIPELINE_WORKER_LLM_OLLAMA_KEEP_ALIVE` 로 설정 가능).

### Worker 가 Ollama 를 가리키게 설정

두 환경변수가 오늘 `LlmQueryParser` 가 사용하고 나중 phase 에서 agent
router / critic / rewriter 가 사용할 공유 `LlmChatProvider` 를 와이어
링:

```
AIPIPELINE_WORKER_LLM_BACKEND=ollama
AIPIPELINE_WORKER_LLM_OLLAMA_BASE_URL=http://localhost:11434
AIPIPELINE_WORKER_LLM_OLLAMA_MODEL=gemma4:e2b
```

worker 가 ollama 컨테이너와 같은 compose 네트워크 안에서 실행될 때는
`http://ollama:11434` 사용. worker 가 호스트에서 실행될 때는
`http://localhost:11434` 사용.

백엔드 위에 LLM 기반 query parser 를 활성화:

```
AIPIPELINE_WORKER_RAG_QUERY_PARSER=llm
```

Parser 는 어떤 provider 실패 (네트워크 다운, timeout, invalid JSON,
schema 위반) 에도 자동으로 regex parser 로 폴백, `ParsedQuery` 에
`parser_name='llm-fallback-regex'` 를 stamp 하여 강등이 `retrieval.json`
에서 보이도록.

### CPU 전용 호스트 (NVIDIA GPU 없음)

Ollama 서비스는 기본적으로 GPU 예약을 선언. NVIDIA 드라이버가 없는
머신에서 compose 는 서비스 시작을 거부. 두 가지 우회 방법:

**Option A — compose override 추가.** 메인 compose 파일 옆에
`docker-compose.cpu.yml` 생성:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices: []
```

그 다음 두 파일 모두로 시작:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml \
  --profile llm up -d ollama ollama-bootstrap
```

**Option B — `docker-compose.yml` 의 deploy 블록을 로컬에서 주석 처리**
(커밋하지 않음).

CPU 에서의 `gemma4:e2b` 추론은 modest GPU 보다 10-30배 느릴 것을 예상.
라우팅 / 분류 / rewriting 에는 여전히 사용 가능; long-form 생성에는
부적합.

### 같은 seam 뒤로 Claude 교체

이미 Anthropic API 키가 있고 로컬 LLM 을 실행하지 않으려면:

```
AIPIPELINE_WORKER_LLM_BACKEND=claude
AIPIPELINE_WORKER_ANTHROPIC_API_KEY=sk-ant-...
AIPIPELINE_WORKER_LLM_CLAUDE_MODEL=claude-haiku-4-5-20251001
```

`LlmQueryParser`, agent router, 그 외 모든 `LlmChatProvider` 컨슈머는
정확히 같은 인터페이스를 봄 — 백엔드 교체에 재컴파일이나 다운스트림
코드 변경이 필요 없음.

### LLM 설정 knob (환경변수)

| 환경변수                                          | 기본값                              | 비고                                                                |
|---------------------------------------------------|-------------------------------------|---------------------------------------------------------------------|
| `AIPIPELINE_WORKER_LLM_BACKEND`                   | `noop`                              | `noop` / `ollama` / `claude`. `noop` → 모든 chat 호출이 `LlmChatError` raise 하고 컨슈머가 폴백. |
| `AIPIPELINE_WORKER_LLM_TIMEOUT_SECONDS`           | `15.0`                              | 호출당 기본 timeout (s).                                            |
| `AIPIPELINE_WORKER_LLM_OLLAMA_BASE_URL`           | `http://localhost:11434`            | compose 안에서는 `http://ollama:11434` 사용.                        |
| `AIPIPELINE_WORKER_LLM_OLLAMA_MODEL`              | `gemma4:e2b`                        | Ollama 가 서빙하는 모델 태그. pull 되어 있어야 함.                 |
| `AIPIPELINE_WORKER_LLM_OLLAMA_KEEP_ALIVE`         | `30m`                               | 호출 사이의 모델 residency. 짧을수록 VRAM 회수가 빠름.              |
| `AIPIPELINE_WORKER_LLM_CLAUDE_MODEL`              | `claude-haiku-4-5-20251001`         | backend=claude 일 때 Anthropic 모델 id.                             |
| `AIPIPELINE_WORKER_RAG_QUERY_PARSER`              | `off`                               | `off` / `regex` / `llm`. `llm` 옵션은 공유 chat provider 를 통과.   |

## 파이프라인 closeout 도구

개발자와 운영자 워크플로용으로 `ai-worker/scripts/` 에 두 개의 로컬
우선 도구가 함께 출시. 어느 것도 품질을 튜닝하지 않음 — 사람이 4개의
터미널에 걸친 로그 파일을 쫓는 대신 "내 스택이 제대로 와이어링됐다" 를
초 단위로 확인할 수 있도록 존재.

### `python -m scripts.doctor` — 준비 상태 확인

Worker 가 진짜 job 을 서비스하기 전에 필요한 모든 런타임 사전 조건을
가벼운 오프라인 우선으로 probe. capability 가 등록을 거부할 이유를
진단하기 위해 worker 시작 전에 실행.

```bash
cd ai-worker
python -m scripts.doctor
```

각 체크는 PASS / FAIL / WARN 을 구체적인 remediation 힌트와 함께 반환.
모든 체크가 통과하면 exit code `0`, 그 외에는 `1`. green 실행이 커버:

- `AIPIPELINE_WORKER_REDIS_URL` 에서 Redis 도달 가능
- `AIPIPELINE_WORKER_RAG_DB_DSN` 에서 PostgreSQL 도달 가능
- `aipipeline.job`, `aipipeline.artifact`, `ragmeta.*` 테이블 존재
  (Flyway V1 + V2 실행됨)
- 설정된 인덱스 디렉토리에 `faiss.index` 와 `build.json` 존재
- `build.json` 이 필요한 필드가 있는 valid JSON
- 런타임 `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` 이 인덱스를 빌드한
  모델과 일치 — worker 의 `Retriever.ensure_ready()` 가 시작 시 수행
  하는 것과 같은 엄격한 체크
- Tesseract 바이너리가 PATH 에 있음 (또는
  `AIPIPELINE_WORKER_OCR_TESSERACT_CMD` 에) 그리고
  `AIPIPELINE_WORKER_OCR_LANGUAGES` 의 모든 언어가 설치됨
- Pillow + PyMuPDF (`fitz`) 가 import 가능
- worker registry 를 미러링하는 롤업된 capability readiness 요약:
  `MOCK` 은 항상 ready; `RAG` 는 모든 DB + 인덱스 체크가 통과한 경우에
  만 ready; `OCR` 은 Tesseract + image dep 이 통과한 경우에만 ready;
  `MULTIMODAL` 은 `RAG` 와 `OCR` 둘 다 ready 인 경우에만 ready.

플래그:

- `--json` — 머신 읽기용 JSON 리포트를 stdout 에 출력
- `--only redis,postgres,...` — 부분 실행 (이미 다른 절반을 확인한
  경우 유용)

완전히 와이어드된 개발 머신에서의 기대 출력:

```
== ai-worker doctor ==
[PASS] redis                  Redis reachable at redis://localhost:6379/0 (2 ms)
[PASS] postgres               PostgreSQL reachable. (12 ms)
[PASS] schemas                aipipeline + ragmeta schemas / tables present. (8 ms)
[PASS] faiss_index            FAISS index files present in ../rag-data (0 ms)
[PASS] build_json             build.json parseable (version=v-1776253724). (0 ms)
[PASS] runtime_model_match    Runtime model matches index (BAAI/bge-m3). (0 ms)
[PASS] tesseract              Tesseract 5.3.3 available (languages: ['eng']). (40 ms)
[PASS] image_deps             Pillow + PyMuPDF importable. (2 ms)
[PASS] capability_readiness   MOCK:ready, RAG:ready, OCR:ready, MULTIMODAL:ready
```

표면화하는 일반적인 실패 모드:

| 실패 체크                  | 의미                                                                       |
|----------------------------|----------------------------------------------------------------------------|
| `redis` FAIL               | `docker compose up -d` 잊음, 또는 잘못된 `REDIS_URL` 환경변수              |
| `postgres` FAIL            | `docker compose up -d` 잊음, 또는 DB DSN 이 잘못된 포트를 가리킴          |
| `schemas` FAIL             | core-api 가 아직 부팅 안 함 → Flyway V1/V2 가 돌지 않음                  |
| `faiss_index` FAIL         | 인덱스가 아직 안 빌드됨 → `python -m scripts.build_rag_index --fixture`   |
| `build_json` FAIL          | 인덱스 dir 가 수동 편집되거나 부분적으로 덮어써짐 → 재빌드                 |
| `runtime_model_match` FAIL | 환경변수 `AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL` 이 인덱스 모델에서 표류  |
| `tesseract` FAIL           | PATH 에 바이너리 누락 또는 언어팩 누락                                     |
| `image_deps` FAIL          | `pip install -r requirements.txt` 없는 fresh venv                          |

### `python -m scripts.smoke_runner` — 풀 스택 스모크 러너

진짜 Spring API 를 통한 진짜 비동기 파이프라인의 원커맨드 end-to-end
스모크. capability (`MOCK`, `RAG`, `OCR`, `MULTIMODAL`) 마다 job 1개를
제출하고, 각 job 을 종단까지 polling 하고, 결과를 가져오고, 기대된
artifact 타입이 도착했는지 assert.

사전 준비 (위 doctor 체크가 모두 커버):

1. Redis, PostgreSQL, core-api, worker 모두 실행 중
2. FAISS 인덱스 존재 (`python -m scripts.build_rag_index --fixture`)
3. Tesseract + PyMuPDF + Pillow 사용 가능
4. OCR 샘플 픽스처가 1회 생성됨:
   `python -m scripts.make_ocr_sample_fixtures` (재실행 안전)

로컬 기본값을 위해 `ai-worker/` 안에서 실행:

```bash
cd ai-worker
python -m scripts.smoke_runner
```

또는 repo 루트에서 얇은 wrapper 경유:

```bash
python scripts/smoke_all.py
```

플래그:

- `--base-url http://host:port` — non-default core-api 가리키기
- `--timeout 120` — job 당 polling 데드라인 올리기 (기본 60s)
- `--poll-interval 0.25` — 더 빠른 피드백을 위해 polling 강화
- `--only MOCK,RAG` — 부분 실행
- `--fixture path/to/file.png` — OCR/MULTIMODAL 입력 override
- `--report smoke-report.json` — 머신 읽기용 리포트 작성
- `--verbose` — runner 자체의 DEBUG 로깅

건강한 파이프라인에서의 기대 콘솔 출력:

```
== smoke runner (http://localhost:8080) ==
started: 2026-04-15T12:34:56+0900  duration: 4123 ms  pass=4 fail=0 skip=0
[OK]   MOCK        job=5d31e42e-... final=SUCCEEDED (145 ms)
         outputs: ['FINAL_RESPONSE']
[OK]   RAG         job=6e42f53f-... final=SUCCEEDED (612 ms)
         outputs: ['RETRIEVAL_RESULT', 'FINAL_RESPONSE']
[OK]   OCR         job=7f53064a-... final=SUCCEEDED (1842 ms)
         outputs: ['OCR_TEXT', 'OCR_RESULT']
[OK]   MULTIMODAL  job=8a64175b-... final=SUCCEEDED (1524 ms)
         outputs: ['OCR_TEXT', 'VISION_RESULT', 'RETRIEVAL_RESULT', 'FINAL_RESPONSE']
```

대응되는 JSON 리포트 (`smoke-report.json`):

```json
{
  "baseUrl": "http://localhost:8080",
  "startedAt": "2026-04-15T12:34:56+0900",
  "durationMs": 4123.0,
  "summary": { "passed": 4, "failed": 0, "skipped": 0, "total": 4 },
  "cases": [
    {
      "capability": "MOCK",
      "status": "SUCCESS",
      "jobId": "5d31e42e-...",
      "submitHttpStatus": 202,
      "finalJobStatus": "SUCCEEDED",
      "durationMs": 145.0,
      "outputTypes": ["FINAL_RESPONSE"],
      "missingArtifacts": [],
      "unexpectedArtifacts": [],
      "errorCode": null,
      "errorMessage": null,
      "failureReason": null
    },
    {
      "capability": "MULTIMODAL",
      "status": "SUCCESS",
      "jobId": "8a64175b-...",
      "submitHttpStatus": 202,
      "finalJobStatus": "SUCCEEDED",
      "durationMs": 1524.0,
      "outputTypes": ["OCR_TEXT", "VISION_RESULT", "RETRIEVAL_RESULT", "FINAL_RESPONSE"],
      "missingArtifacts": [],
      "unexpectedArtifacts": [],
      "errorCode": null,
      "errorMessage": null,
      "failureReason": null
    }
  ]
}
```

(JSON 리포트의 필드명은 `dataclasses.asdict` 를 따릅니다 —
camelCase 가 아닌 `snake_case` python 속성. 최상위 summary 키만이
직접 작성된 camelCase 필드.)

표면화하는 일반적인 실패 모드:

| 실패 라인                                                     | 가능한 원인                                                |
|---------------------------------------------------------------|------------------------------------------------------------|
| `HTTP error: ConnectError`                                    | core-api 실행 안 함                                        |
| `Timed out ... waiting for job ... to reach a terminal state` | worker 실행 안 함, 잘못된 Redis 큐 키                      |
| `Job ended in 'FAILED' (errorCode='UNKNOWN_CAPABILITY' ...)`  | worker 가 capability 등록 거부 (doctor 참조)              |
| `missing required artifact type(s) ['RETRIEVAL_RESULT']`      | RAG capability 등록됐지만 retrieval stage 깨짐            |
| `carries unexpected artifact type(s) [...]`                   | capability 가 `REQUIRED_OUTPUTS` 업데이트 없이 새 출력 emit 시작 |
| `POST /jobs returned status='RUNNING'`                        | core-api 회귀 (항상 `QUEUED` 로 받아야 함)                 |
| `fixture load failed`                                         | `python -m scripts.make_ocr_sample_fixtures` 가 안 돌았음  |

**이 도구가 증명하는 것:** 파이프라인 연결성과 계약 무결성. green 실행은
core-api 가 모든 capability 를 받아들이고, worker 가 Redis 큐를 소비
하고, 모든 capability 가 API 문서가 나열한 artifact 타입을 생성하고,
`/result` 엔드포인트가 각각에 대해 다운로드 가능한 access URL 을 들고
있음을 보장.

**이 도구가 증명하지 않는 것:** OCR, RAG, MULTIMODAL 답변 품질. Runner
는 artifact *타입* 에 대해 assert 하지 답변 내용에 대해서가 아님. 샘플
입력 ("which anime is about an old fisherman feeding stray harbor
cats", SMOKE TEST placeholder PNG) 은 plumbing 을 운동시키기 위해
선택된 것이지 의미 있는 eval 이 아님. 품질 측정은 큐레이션된 데이터셋에
대해 `eval/run_eval.py` 사용 — harness 계약은 `eval/README.md` 참조.

### 파이프라인 closeout 체크리스트

파이프라인 변경을 close out 하거나 fresh clone 을 검증할 때 이 시퀀스
사용:

1. `docker compose up -d`
2. `cd core-api && mvn spring-boot:run` — Tomcat 배너까지 대기
3. `cd ai-worker && python -m scripts.build_rag_index --fixture` —
   1회 또는 모델/데이터셋 변경 후
4. `python -m scripts.make_ocr_sample_fixtures` — 1회,
   `eval/datasets/samples/` 아래 커밋된 PNG 생성
5. `python -m scripts.doctor` — 진행 전 모든 행이 green 이어야 함
6. `python -m app.main` — fresh 터미널에서 worker 시작; `Active
   capabilities: ['MOCK', 'MULTIMODAL', 'OCR', 'RAG']` 확인
7. `python -m scripts.smoke_runner --report smoke-report.json` —
   `pass=4 fail=0 skip=0` 와 4개의 SUCCESS 행 기대
8. `python -m pytest tests/ -q` — 만지는 동안 worker 테스트 슈트가
   회귀하지 않았는지 sanity-check

5단계는 스모크 실행을 낭비하기 전에 "env drift" 부류의 버그 (누락된
바이너리, 잘못된 모델, 누락된 스키마) 를 잡아냄. 7단계는 비동기 파이프
라인이 실제 round trip 을 견디는지 end to end 로 증명. 함께 "UI 를
poke 해서 뭔가 깨지는지 보기" 를 결정적이고 머신 읽기용인 무엇이
동작하고 무엇이 동작 안 하는지의 기록으로 대체.

## 테스트 실행

Worker 측 단위 테스트 (인프라 필요 없음):

```bash
cd ai-worker
python -m pytest tests/ -q
```

기대: 154개 통과:
- mock capability 2건
- RAG chunker 6건
- RAG happy-path 2건
- RAG validation 6건
- OCR 13건 (happy-path 변형, unsupported-type 거부, empty/low-confidence
  엣지 케이스, registry 회복력)
- MULTIMODAL 16건 (image + PDF happy path, OCR/vision 실패 격리, both-
  fail hard error, unsupported type 거부, fusion 결정성, heuristic
  vision provider 운동, registry parent-dependency 체크)
- doctor 22건 (체크 함수 happy path + 모든 remediation 분기, capability
  롤업, DSN redaction, text/JSON reporter)
- smoke runner 32건 (제출 / 최종 상태 / 결과-출력 표류 모드 전반의
  shape assertion 단위 커버리지, 리포트 builder counter, parse_only arg
  parser, 픽스처 loader 폴백)
- 나머지는 eval harness 단위 테스트에서

모든 OCR + MULTIMODAL 테스트는 fake provider 사용으로 Tesseract 또는
PyMuPDF 설치 없이 실행. 모든 RAG 테스트는 hashing 폴백 embedder 사용
으로 모델 다운로드 없고 Postgres 도 필요 없음. 한 MULTIMODAL 테스트는
실제 Pillow 생성 이미지에 `HeuristicVisionProvider` 를 운동; Pillow 는
sentence-transformers 와 함께 출시되므로 비정상적으로 깎아낸 venv 에서
만 자동 skip.

## 설정 knob (환경변수)

전체 리스트는 repo 루트의 `.env.example` 참조. 가장 자주 만지는 것:

| 환경변수                                   | 기본값                                      | 비고                                |
|--------------------------------------------|---------------------------------------------|-------------------------------------|
| `SPRING_DATASOURCE_URL`                    | jdbc:postgresql://localhost:5433/aipipeline | compose PostgreSQL 기본             |
| `SPRING_DATASOURCE_USERNAME` / `_PASSWORD` | aipipeline / aipipeline_pw                  |                                     |
| `SPRING_DATA_REDIS_HOST` / `_PORT`         | localhost / 6379                            |                                     |
| `SERVER_PORT`                              | 8080                                        | core-api listen 포트                |
| `AIPIPELINE_STORAGE_LOCAL_ROOT_DIR`        | `${user.dir}/../local-storage`              | artifact 용 공유 디스크             |
| `AIPIPELINE_WORKER_WORKER_ID`              | worker-local-1                              | claim token                         |
| `AIPIPELINE_WORKER_CORE_API_BASE_URL`      | http://localhost:8080                       |                                     |
| `AIPIPELINE_WORKER_REDIS_URL`              | redis://localhost:6379/0                    |                                     |
| `AIPIPELINE_WORKER_LOCAL_STORAGE_ROOT`     | `../local-storage`                          | core-api 와 같은 폴더로 해소되어야 함 |
