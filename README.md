# async-ocr-rag-multimodal-pipeline

비동기 AI 처리 플랫폼.

## 벤치마크 (현재 브랜치 기준 측정값)

Phase-0 RAG 베이스라인은 2026-04-23에 bge-m3 + FAISS IndexFlatIP 조합으로
커밋된 픽스처에 대해 측정했습니다. 리포트는
[`ai-worker/eval/reports/baseline-phase0.json`](ai-worker/eval/reports/baseline-phase0.json)
(kr_sample) 와 `baseline-phase0-anime.json` 에 있습니다.

| 데이터셋             | 능력(Capability) | hit@5 | recall@5 | MRR    | dup_rate | topk_gap | CER (OCR) | p50 / p95 retrieval ms |
|----------------------|------------------|-------|----------|--------|----------|----------|-----------|------------------------|
| anime (en, 8 docs)   | RAG              | 1.000 | 1.000    | 1.000  | 0.433    | 0.236    | n/a       | 8.6 / 24.9             |
| kr_sample (kr, 10)   | RAG              | 1.000 | 1.000    | 1.000  | 0.340    | 0.233    | n/a       | 8.7 / 27.4             |
| ocr_sample (en)      | OCR              | n/a   | n/a      | n/a    | n/a      | n/a      | not measured | not measured        |
| ocr_sample (kr)      | OCR              | n/a   | n/a      | n/a    | n/a      | n/a      | not measured | not measured        |

> bge-m3 임베더, FAISS IndexFlatIP, ExtractiveGenerator,
> CUDA 백엔드 sentence-transformers 조합으로 측정. 두 픽스처 모두 hit@5 /
> recall@5 / MRR이 만점이라 — 이 작은 코퍼스 위에서 retriever는 더 끌어올릴
> 여지가 없습니다. 다음 단계에서 움직일 신호는 **dup_rate** (reranker 도입 시
> 떨어져야 함), **topk_gap** (clean hit에서 더 벌어져야 함), 그리고
> **p95 retrieval** (agent/배칭 도입 후에도 안정적이어야 함) 입니다.
>
> 재현 (ragmeta 없이, 모델이 캐시된 개발 머신 기준):
> `python -m eval.run_eval rag --dataset eval/datasets/rag_sample_kr.jsonl --offline-corpus fixtures/kr_sample.jsonl --out-json eval/reports/baseline-phase0.json`.
>
> 라이브 프로덕션 경로 (`build_rag_index` → bge-m3 → ragmeta →
> run_eval) 는 그대로 동작하지만, 이 환경에는 Tesseract 와 eng/kor
> 언어팩이 설치되어 있지 않아 OCR 행은 "not measured" 로 남아 있습니다.

- **Phase 1**: 비동기 스켈레톤 (job 생명주기, artifact 모델, claim/callback,
  Redis 디스패치, 로컬 파일시스템 스토리지, `MOCK` capability).
- **Phase 1.1**: 타깃 스택 (Spring Boot 4.0.3, Java 21, PostgreSQL 18,
  Python 3.12, Redis) 으로 안정화하고 end-to-end 검증.
- **Phase 2 (이 커밋)**: 진짜 **text-RAG capability** 를 추가 — FAISS
  벡터 인덱스, sentence-transformers 임베딩, extractive grounded
  generation, RAG 메타데이터를 PostgreSQL의 별도 `ragmeta` 스키마에 저장
  (MongoDB 사용 안 함). `MOCK` capability 도 그대로 유지.

실제 OCR, 파일/PDF 파싱, multimodal capability는 이후 phase 에서 도입됩니다.

## 스택

| 컴포넌트       | 버전                                   |
|----------------|----------------------------------------|
| Java           | 21 (LTS)                               |
| Spring Boot    | 4.0.3                                  |
| Maven          | 3.9+                                   |
| Python         | 3.12 (3.13 도 동작)                    |
| PostgreSQL     | 18                                     |
| Redis          | latest                                 |

## 디렉토리 구조

```
.
├── core-api/                   Spring Boot — jobs / artifacts / claim / callback
│   └── src/main/resources/db/migration/
│       ├── V1__init.sql           pipeline 스키마 (job, artifact)
│       └── V2__ragmeta_schema.sql ragmeta 스키마 (documents, chunks, index_builds)
├── ai-worker/                  Python worker — Redis BRPOP 컨슈머
│   ├── app/capabilities/
│   │   ├── mock_processor.py   MOCK capability (phase 1)
│   │   └── rag/                RAG capability (phase 2):
│   │       ├── capability.py      엔트리 포인트
│   │       ├── chunker.py         port/rag 에서 이식 (greedy + window_by_chars)
│   │       ├── embeddings.py      EmbeddingProvider + sentence-transformers 구현
│   │       ├── generation.py      GenerationProvider + extractive 구현
│   │       ├── faiss_index.py     IndexFlatIP 래퍼
│   │       ├── metadata_store.py  ragmeta.* 용 psycopg2 DAO
│   │       ├── ingest.py          JSONL -> chunks -> FAISS + 메타데이터
│   │       └── retriever.py       query -> top-k RetrievedChunk 리스트
│   ├── scripts/build_rag_index.py 인덱싱 CLI
│   └── fixtures/anime_sample.jsonl 작은 커밋된 픽스처
├── frontend/                   작은 HTML 테스트 클라이언트
├── scripts/e2e_smoke.py        End-to-end MOCK 스모크 테스트
├── docker-compose.yml          인프라 전용 (redis 기본, postgres는 "independent" 프로파일, minio는 "minio" 프로파일)
├── .env.example                전체 환경변수 레퍼런스
└── docs/
    ├── architecture.md
    ├── local-run.md
    └── api-summary.md
```

## 1분 투어

- **core-api** 는 DDD + 헥사고날 구조: `domain` / `application/port` /
  `application/service` / `adapter/in/web` / `adapter/out/persistence`
  에 더해 별도의 `queue` / `storage` outbound 어댑터.
- **ai-worker** 는 직선형 실행 엔진: Redis BRPOP → claim → 입력 가져오기 →
  capability 실행 → 출력 바이트 업로드 → callback.
- **PostgreSQL이 job 상태를 소유합니다.** Redis는 디스패치 신호일 뿐,
  진실의 출처가 아닙니다.
- **스토리지는 port 뒤에 추상화**되어 있습니다. phase 1 에서는
  로컬 파일시스템 어댑터를 제공하고, MinIO/S3는 application 레이어를
  건드리지 않고 나중에 끼워넣을 수 있습니다.
- **Capability 는 플러그인 구조**: `MOCK` 은
  `ai-worker/app/capabilities/mock_processor.py` 에 살고, `rag/` ,
  `ocr/`, `multimodal/` 은 phase 2+ 에서 켜질 빈 패키지입니다.

## 두 가지 데이터베이스 실행 모드

Phase 1.1 은 Postgres 를 띄우는 두 가지 방법을 지원합니다. 정확히 하나만
고르세요.

- **Mode A (기본): 기존 Postgres 컨테이너 재사용** (예: RIA 같은 다른
  프로젝트에서 이미 띄워둔 것). 그 안에 전용 `aipipeline` 데이터베이스와
  유저를 만들어 다른 무엇과도 충돌하지 않게 합니다. 이 compose 는
  redis 만 띄웁니다.
- **Mode B: 독립 인프라 compose**. 자체 `postgres:18` 을 호스트 포트
  `5433` 으로 띄워서 `5432` 위의 어떤 것과도 충돌하지 않습니다. core-api는
  `independent` Spring 프로파일로 실행해야 합니다.

정확한 부트스트랩 명령, 트러블슈팅 목록, Windows 전용 팁은
[`docs/local-run.md`](docs/local-run.md) 참조.

## 실행

짧은 버전 (Mode A. `:5432` 에 이미 Postgres 컨테이너가 있고, 문서대로
`aipipeline` DB + 유저가 부트스트랩되어 있다고 가정):

```bash
docker compose up -d redis                                                  # 터미널 1
(cd core-api && mvn spring-boot:run)                                        # 터미널 2
(cd ai-worker && pip install -r requirements.txt && python -m app.main)     # 터미널 3
python scripts/e2e_smoke.py                                                 # 터미널 4
```

### 내부 엔드포인트 인증

모든 `/api/internal/**` 엔드포인트 (claim, callback, artifact 업로드) 는
공유 시크릿 헤더 (`X-Internal-Secret`) 로 게이팅됩니다. core-api 와
worker 가 사용하는 환경변수 이름은 **다르지만** 실제 시크릿 값은
**반드시 같아야** 합니다:

| 서비스    | 환경변수                            |
|-----------|-------------------------------------|
| core-api  | `AIPIPELINE_INTERNAL_SECRET`        |
| ai-worker | `AIPIPELINE_WORKER_INTERNAL_SECRET` |

둘 다 미설정이면 core-api 는 **개발용 패스스루 모드**로 동작합니다
(시작 시 WARN 한 번 로그가 찍히고, 모든 internal 요청이 헤더 없이
허용됨). 로컬 개발과 CI 의 마찰을 줄이기 위함입니다.

```bash
# 인증을 강제하는 프로덕션 스타일 실행:
export AIPIPELINE_INTERNAL_SECRET=change-me-in-production
export AIPIPELINE_WORKER_INTERNAL_SECRET=change-me-in-production
(cd core-api && mvn spring-boot:run)      # 터미널 2
(cd ai-worker && python -m app.main)      # 터미널 3
```

### S3 / MinIO 스토리지 백엔드

기본적으로 artifact 는 로컬 파일시스템에 저장됩니다 (`backend=local`).
대신 MinIO (S3 호환) 를 쓰려면:

```bash
# 1. MinIO 시작 + 버킷 자동 생성
docker compose --profile minio up -d minio minio-bootstrap

# 2. 환경변수 설정 (또는 .env 에 추가)
export AIPIPELINE_STORAGE_BACKEND=s3
export AIPIPELINE_STORAGE_S3_ENDPOINT=http://localhost:9000
export AIPIPELINE_STORAGE_S3_BUCKET=aipipeline-artifacts
export AIPIPELINE_WORKER_S3_ENDPOINT=http://localhost:9000
export AIPIPELINE_WORKER_S3_ACCESS_KEY=aipipeline
export AIPIPELINE_WORKER_S3_SECRET_KEY=aipipeline_secret

# 3. core-api 와 worker 를 평소대로 시작
```

MinIO 콘솔은 `http://localhost:9001` 에서 접근 가능 (user:
`aipipeline`, password: `aipipeline_secret`).

기대되는 스모크 출력:
```
[1/4] submitting text job ...
...
[4/4] downloading output artifact content ...
     parsed JSON keys = ['jobId', 'capability', ...]
OK - pipeline survived the round trip.
```

## 원클릭 데모

샘플 PDF 를 업로드하고, 전체 MULTIMODAL 파이프라인을 돌리고, 결과를
보기좋게 출력해주는 더 풍부한 데모 스크립트:

```bash
pip install reportlab                # 1회: PDF 생성용
pip install rich                     # 선택: 컬러 출력

python scripts/demo.py               # 기본: MULTIMODAL
python scripts/demo.py --capability OCR
python scripts/demo.py --question "What metrics are shown?"
```

스크립트는 첫 실행 시 `scripts/assets/` 아래에 2페이지 샘플 PDF 를 자동
생성합니다 (gitignore 됨). 백엔드 없이 import 와 PDF 생성만 검증하려면
`--self-test` 옵션을 사용하세요:

```bash
python scripts/demo.py --self-test
```

샘플 출력 (`rich` 사용 시):
```
[1/5] health check (http://localhost:8080) ...
[2/5] preparing sample PDF ...
  .../scripts/assets/demo_sample.pdf (12,345 bytes)
[3/5] submitting MULTIMODAL job ...
  jobId = abc-123  status = QUEUED
[4/5] polling for completion ...
  ... RUNNING
  SUCCEEDED
[5/5] fetching results ...
+----- Job Summary -----+
| Job ID: abc-123       |
| Capability: MULTIMODAL|
| Status: SUCCEEDED     |
+-----------------------+
--- FINAL_RESPONSE ---
(grounded markdown answer)
Demo complete.
```

## 이 머신에서 검증됨 (phase 2)

Phase 1.1 결과는 그대로 유효. phase 2 신규 검증:

- **Flyway V2** 마이그레이션: `Successfully applied 1 migration to schema "aipipeline", now at version v2`, `ragmeta` 스키마가 pipeline 테이블 옆에 생성됨.
- **RAG 포함 Worker 시작**: cuda:0 에서 FAISS 인덱스 + sentence-transformers 모델 로드 후 `Active capabilities: ['MOCK', 'RAG']`.
- **인덱스 빌드**: `scripts/build_rag_index.py --fixture` 가 8문서 anime 픽스처를 → 24 chunks → 384 차원 벡터 → FAISS `IndexFlatIP` + ragmeta 행으로 10.4초 만에 적재.
- **RAG query 1** ("anime about an old fisherman feeding stray harbor cats") → 최상위 hit `anime-005#overview` score **0.845**, 정답 ("The Harbor Cats"), grounded citation 5 건.
- **RAG query 2** ("anime about weather researchers stranded in a mountain observatory during a long storm") → 최상위 hit `anime-004#plot` score **0.515**, 정답 ("Signal Fires").
- **MOCK 회귀**: RAG 와 함께 여전히 성공, FINAL_RESPONSE 변경 없음.
- **Artifact 모양**: 각 RAG job 은 INPUT_TEXT 1건 + RETRIEVAL_RESULT 1건 + FINAL_RESPONSE 1건 정확히 생성 (중복 없음).
- **단위 테스트**: `10 passed in 0.26s` (chunker 엣지 케이스 + mock capability + 해싱 폴백 임베더로 풀 RAG capability).

## 다음 phase

- **Phase 2.1 (OCR + 파일 입력)** — OCR capability 추가, `INPUT_FILE`
  경로를 확장하여 OCR → `OCR_TEXT` → chunk → 인덱스로 흐르게 함. phase 2의
  chunker, embedding provider, generation provider, retriever 는 모두
  재작성 없이 확장 가능 — OCR 은 `OCR_TEXT` artifact 를 만들어내는 새
  capability 이고, 그 결과를 RAG capability 가 다시 소비할 수 있습니다.
- **Phase 3+** — multimodal capability, 진짜 LLM generation provider,
  MinIO/S3 스토리지 어댑터, retry 오케스트레이션, Kubernetes 매니페스트,
  capability 별 Redis lane, 진짜 frontend.

## 문서

**설계 / 운영 / API**

- [`docs/architecture.md`](docs/architecture.md) — 시스템 아키텍처, 설계
  결정, capability 별 흐름과 trace 스키마
- [`docs/api-summary.md`](docs/api-summary.md) — 모든 HTTP 엔드포인트의
  요청/응답 모양과 에러 코드
- [`docs/local-run.md`](docs/local-run.md) — 로컬 부트스트랩, 트러블
  슈팅, capability 활성화 가이드

**평가 (ai-worker/eval)**

- [`ai-worker/eval/README.md`](ai-worker/eval/README.md) — eval harness
  개요, 메트릭 정의, 권장 평가 시퀀스
- [`ai-worker/eval/datasets/README.md`](ai-worker/eval/datasets/README.md)
  — eval 데이터셋 카탈로그 (anime / enterprise / cross-domain)
- [`ai-worker/eval/corpora/README.md`](ai-worker/eval/corpora/README.md)
  — retrieval 코퍼스 디렉토리 규약
- [`ai-worker/eval/corpora/anime_namu_v3/README.md`](ai-worker/eval/corpora/anime_namu_v3/README.md)
  — namu-wiki anime 코퍼스 (1,764 작품) 스키마와 re-stage 지침
- [`ai-worker/eval/eval_queries/README.md`](ai-worker/eval/eval_queries/README.md)
  — retrieval-eval query 셋 (silver / gold / smoke)

**튜닝**

- [`docs/tuning.md`](docs/tuning.md) — Optuna + Claude interpreter loop
  운영 가이드
- [`docs/optuna-tuning-plan.md`](docs/optuna-tuning-plan.md) — 튜닝
  파이프라인 단계별 로드맵
- [`ai-worker/eval/experiments/README.md`](ai-worker/eval/experiments/README.md)
  — round-refinement 워크스페이스 구조와 round 진행 방법

**Frontend**

- [`frontend/app/README.md`](frontend/app/README.md) — Vite + React +
  TypeScript 셋업 노트
