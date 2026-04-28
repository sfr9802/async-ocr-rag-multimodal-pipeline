# async-ocr-rag-multimodal-pipeline

비동기 AI 처리 플랫폼.

## 벤치마크

`anime_silver_200` (1,764 docs, 200 deterministic synthetic queries) +
커밋된 fixture 두 개로 측정. 환경: RTX 5080 (16 GB), bge-m3 임베더,
FAISS `IndexFlatIP`. 상세 리포트는
[`ai-worker/eval/reports/`](ai-worker/eval/reports/) 아래 — phase별
하위 디렉토리에 JSON + Markdown 모두 커밋되어 있음.

**Phase 0 — fixture baseline** (`baseline-phase0*.json`)

| dataset            | hit@5 | recall@5 | MRR  | dup_rate | topk_gap | p50/p95 ret (ms) |
|--------------------|------:|---------:|-----:|---------:|---------:|-----------------:|
| anime (en, 8 docs) | 1.000 | 1.000    | 1.000 | 0.433   | 0.236    | 8.6 / 24.9       |
| kr_sample (kr, 10) | 1.000 | 1.000    | 1.000 | 0.340   | 0.233    | 8.7 / 27.4       |

OCR 행은 이 환경에 Tesseract / 언어팩이 없어 not measured.

**Phase 2A — silver-200 cross-encoder reranker progression**
(`phase2a-reranker/`)

| run                                 | hit@1 | hit@3 | hit@5 | MRR@10 | NDCG@10 | rerank p95 ms |
|-------------------------------------|------:|------:|------:|-------:|--------:|--------------:|
| B1 dense (combined-old)             | 0.560 | 0.670 | 0.685 | 0.617  | 0.643   | –             |
| B2 dense (token-aware-v1)           | 0.540 | 0.665 | 0.680 | 0.604  | 0.631   | –             |
| **B2 + rerank top20**               | 0.605 | 0.680 | 0.700 | 0.653  | 0.675   | 706           |
| **B2 + rerank top50**               | **0.615** | **0.700** | **0.715** | **0.666** | **0.689** | 1840 |

Reranker: `BAAI/bge-reranker-v2-m3`, batch=16, max_len=512, final_top_k=10.
B1 ↔ B2는 chunk granularity가 달라 hit@k 차이는 chunker + reranker 합산
효과로 봐야 함. Production default는 `rag_reranker="off"`; reranker는
eval CLI(`retrieval-rerank`)에서만 활성화.

**Candidate-recall ceiling (B2 dense top-50)** — reranker는 candidate
순서만 바꾸므로 recall@N이 hit@k 상한.

| metric | value |
|--------|------:|
| hit@10 | 0.715 |
| hit@20 | 0.770 |
| hit@50 | 0.800 |

**Phase 2A-L — selected_baseline (top10)** (`legacy-baseline-final/`):
hit@1=0.620, hit@3=0.675, hit@5=0.705, MRR@10=0.654, total query p95
350 ms. Agent loop legacy backend는 이 manifest를 reference로 사용.

**재현**

```bash
cd ai-worker

# Phase 2A — candidate-recall + rerank sweep
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --top-k 50 --extra-hit-k 10 --extra-hit-k 20 --extra-hit-k 50 \
    --out-dir eval/reports/phase2a-reranker/candidate-recall-b2

for N in 20 50; do
  python -m eval.run_eval retrieval-rerank \
      --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
      --dataset eval/eval_queries/anime_silver_200.jsonl \
      --dense-top-n $N --final-top-k 10 \
      --reranker-model BAAI/bge-reranker-v2-m3 --reranker-batch-size 16 \
      --out-dir eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top$N
done

# Phase 0 — fixture baseline (ragmeta 없이)
python -m eval.run_eval rag \
    --dataset eval/datasets/rag_sample_kr.jsonl \
    --offline-corpus fixtures/kr_sample.jsonl \
    --out-json eval/reports/baseline-phase0.json
```

## Phase timeline

- **1** — 비동기 스켈레톤 (job/artifact, claim/callback, Redis 디스패치, MOCK).
- **1.1** — 타깃 스택(Spring Boot 4.0.3 / Java 21 / PG 18 / Py 3.12 / Redis) 안정화 + E2E 검증.
- **2** — text-RAG capability (FAISS + sentence-transformers + extractive generator + ragmeta 스키마).
- **1B** — namu-wiki 코퍼스 prefix / inline-edit-marker strip.
- **1C** — token-aware chunker (1024-token hard cap, avg ctx tokens 531 → 293).
- **2A** — bge-reranker-v2-m3 cross-encoder 후처리, eval CLI 전용. Production default는 그대로 NoOp.

OCR / 파일·PDF 파싱 / multimodal은 이후 phase에서.

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

## 실행

Mode A (기본 — 기존 Postgres `:5432` 재사용, `aipipeline` DB/유저 부트스트랩
완료 가정). Mode B (독립 `postgres:18` on `:5433`, core-api는
`independent` 프로파일) 는 [`docs/local-run.md`](docs/local-run.md) 참조.

```bash
docker compose up -d redis                                              # 1
(cd core-api && mvn spring-boot:run)                                    # 2
(cd ai-worker && pip install -r requirements.txt && python -m app.main) # 3
python scripts/e2e_smoke.py                                             # 4
```

**내부 엔드포인트 인증** — `/api/internal/**` 는 `X-Internal-Secret` 으로
게이팅. 두 서비스 시크릿 값은 같아야 함; 둘 다 미설정이면 core-api는
WARN 후 패스스루(개발 편의).

| 서비스    | 환경변수                            |
|-----------|-------------------------------------|
| core-api  | `AIPIPELINE_INTERNAL_SECRET`        |
| ai-worker | `AIPIPELINE_WORKER_INTERNAL_SECRET` |

**S3 / MinIO 스토리지** — 기본 `backend=local`. MinIO로 전환하려면
`docker compose --profile minio up -d minio minio-bootstrap` 후
`AIPIPELINE_STORAGE_BACKEND=s3` + `AIPIPELINE_STORAGE_S3_ENDPOINT` /
`AIPIPELINE_WORKER_S3_*` 설정. 콘솔 `http://localhost:9001`
(`aipipeline` / `aipipeline_secret`). 자세한 env 목록은 `.env.example`.

## 원클릭 데모

`scripts/demo.py` 가 샘플 PDF를 만들어 MULTIMODAL 파이프라인을 풀로
돌리고 결과를 출력합니다. `--self-test` 로 백엔드 없이 PDF 생성만
검증 가능.

```bash
pip install reportlab rich           # 1회 — rich 는 선택
python scripts/demo.py               # 기본: MULTIMODAL
python scripts/demo.py --capability OCR --question "What metrics are shown?"
python scripts/demo.py --self-test
```

## 다음 phase

- **2.1 (OCR + 파일 입력)** — OCR capability + `INPUT_FILE` →
  `OCR_TEXT` → RAG 재소비 흐름.
- **3+** — multimodal capability, real LLM generation, MinIO/S3 어댑터,
  retry 오케스트레이션, K8s 매니페스트, capability별 Redis lane,
  실 frontend.

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
