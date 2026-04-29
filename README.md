# async-ocr-rag-multimodal-pipeline

Spring Boot API 서버와 FastAPI Worker를 분리하여, RAG 검색 및 AI 처리 작업을 비동기 파이프라인으로 실행하는 포트폴리오 프로젝트입니다.

이 프로젝트는 단순한 LLM API 호출 예제가 아니라, **AI 작업을 백엔드 서비스 구조 안에서 안정적으로 처리하는 방식**을 실험하는 데 목적이 있습니다.  
작업 요청은 PostgreSQL 기반 Job 상태로 관리하고, Worker는 queue 신호만으로 바로 실행하지 않고 `claim-before-execute` 방식으로 작업 소유권을 확보한 뒤 실행합니다.

현재 구현 범위는 **비동기 작업 파이프라인 + Text RAG + 검색 평가/튜닝**입니다.  
OCR, file parsing, multimodal capability는 이후 phase에서 확장할 예정입니다.

---

## What this project demonstrates

이 저장소는 다음 질문에 답하기 위해 만들었습니다.

> RAG, OCR, multimodal 같은 AI 작업을 단순 함수 호출이 아니라  
> 운영 가능한 백엔드 작업 흐름으로 설계하려면 어떤 구조가 필요한가?

핵심 설계 포인트는 다음과 같습니다.

- Spring Boot 기반 `core-api`와 Python/FastAPI 기반 `ai-worker` 분리
- PostgreSQL을 Job 상태의 Source of Truth로 사용
- Redis는 상태 저장소가 아니라 dispatch signal 역할로 제한
- Worker 실행 전 `claim`을 통한 작업 소유권 확보
- callback 기반 결과 반영
- artifact 기반 입력/출력 관리
- RAG capability를 독립 모듈로 구성
- FAISS + sentence-transformers 기반 vector retrieval
- cross-encoder reranker 평가
- hit@k, MRR, NDCG, latency 기반 검색 품질 측정
- Optuna 기반 retrieval parameter tuning 실험

---

## Current status

| Area | Status | Notes |
|---|---|---|
| Async job pipeline | Done | job / artifact / claim / callback |
| Local worker dispatch | Done | Redis BRPOP 기반 dispatch |
| Job state management | Done | PostgreSQL이 Source of Truth |
| Text RAG | Done | FAISS + sentence-transformers |
| Reranker evaluation | Done | `BAAI/bge-reranker-v2-m3` |
| Retrieval evaluation | Done | hit@k, MRR, NDCG, latency |
| Optuna tuning | In progress | retrieval parameter tuning 실험 |
| OCR input | Planned | file / PDF → OCR_TEXT |
| Multimodal capability | Planned | image / document understanding |
| Real LLM generation | Planned | local/API provider abstraction |

---

## Architecture

```mermaid
flowchart LR
    Client[Client / Frontend] --> Core[Spring Boot Core API]

    Core -->|create job / update state| DB[(PostgreSQL)]
    Core -->|dispatch signal| Redis[(Redis Queue)]

    Worker[FastAPI AI Worker] -->|BRPOP signal| Redis
    Worker -->|claim job| Core
    Worker -->|fetch input / metadata| Core

    Worker --> RAG[RAG Capability]
    RAG --> Embed[Embedding Provider]
    RAG --> FAISS[(FAISS Index)]
    RAG --> Reranker[Cross-Encoder Reranker]
    RAG --> DB

    Worker -->|upload artifact| Core
    Worker -->|callback result| Core
    Core -->|persist result state| DB
```

---

## Design principles

### 1. PostgreSQL owns job state

Redis는 queue 신호를 전달할 뿐, 작업 상태를 소유하지 않습니다.  
실제 Job 상태는 PostgreSQL에 저장되며, Worker는 실행 전 core-api에 claim을 요청합니다.

이 구조를 통해 다음 문제를 줄입니다.

- 같은 작업이 여러 Worker에서 중복 실행되는 문제
- queue 재전송으로 인해 결과가 중복 반영되는 문제
- callback 실패 후 재시도 시 상태가 꼬이는 문제
- Worker 장애 이후 작업 상태를 추적하기 어려운 문제

---

### 2. Claim before execute

Worker는 Redis에서 작업 신호를 받더라도 즉시 실행하지 않습니다.  
먼저 core-api에 claim을 요청하고, claim에 성공한 작업만 실행합니다.

```text
Redis signal
  → Worker receives job id
  → Worker requests claim
  → Core API checks current job state
  → Claim succeeds or fails
  → Worker executes only if claim succeeds
```

이 방식은 queue가 at-least-once delivery 성격을 가지더라도, 실제 실행 소유권은 DB 상태로 제어할 수 있게 합니다.

---

### 3. Queue is replaceable

현재 구현은 로컬 재현성과 GPU 비용 절감을 위해 Redis 기반 dispatch를 사용합니다.

다만 Redis는 작업 상태를 소유하지 않고 dispatch signal 역할만 수행하므로, 구조적으로는 Cloud Tasks, Pub/Sub, SQS 같은 managed queue로 교체할 수 있습니다.

```text
Current:
Spring Core API → Redis → FastAPI Worker

Possible production adapter:
Spring Core API → Cloud Tasks / Pub/Sub / SQS → FastAPI Worker
```

이 저장소는 실제 클라우드 GPU 배포보다, **로컬/단일 머신 환경에서 전체 비동기 AI 처리 흐름을 재현하는 것**을 우선했습니다.

---

### 4. Capability is pluggable

Worker는 capability 단위로 AI 작업을 실행합니다.

현재 구현된 capability:

- `MOCK`
- `RAG`

확장 예정 capability:

- `OCR`
- `MULTIMODAL`
- real LLM generation

Capability 구조를 분리해두었기 때문에, 향후 OCR이나 multimodal processing을 추가하더라도 job / artifact / claim / callback 흐름은 유지할 수 있습니다.

---

## Core components

### core-api

Spring Boot 기반 백엔드 API 서버입니다.

역할:

- Job 생성
- Job 상태 관리
- Worker claim 처리
- artifact metadata 관리
- callback 수신
- 내부 API 인증
- RAG metadata schema 관리

구조:

```text
domain
application
  ├── port
  └── service
adapter
  ├── in/web
  ├── out/persistence
  ├── out/queue
  └── out/storage
```

---

### ai-worker

Python/FastAPI 기반 AI Worker입니다.

역할:

- Redis queue 신호 수신
- core-api에 claim 요청
- 입력 artifact 로드
- capability 실행
- 결과 artifact 업로드
- callback 전송

기본 실행 흐름:

```text
Redis BRPOP
  → claim
  → fetch input
  → execute capability
  → upload output artifact
  → callback
```

---

### RAG capability

현재 가장 많이 구현된 capability입니다.

구성 요소:

- JSONL corpus ingestion
- token-aware chunking
- sentence-transformers embedding
- FAISS `IndexFlatIP`
- vector retrieval
- optional cross-encoder reranking
- extractive generation provider
- retrieval evaluation harness
- latency breakdown
- Optuna tuning workflow

---

## Tech stack

| Area | Stack |
|---|---|
| Backend | Java 21, Spring Boot 4.0.3, Maven |
| Worker | Python 3.12, FastAPI |
| Database | PostgreSQL 18 |
| Queue / Dispatch | Redis |
| Vector Search | FAISS `IndexFlatIP` |
| Embedding | `bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Evaluation | hit@k, recall@k, MRR, NDCG, latency |
| Infra | Docker Compose |
| Storage | Local filesystem, S3/MinIO adapter-ready |

---

## Repository structure

```text
.
├── core-api/                        Spring Boot — jobs / artifacts / claim / callback
│   └── src/main/resources/db/migration/
│       ├── V1__init.sql             pipeline schema
│       └── V2__ragmeta_schema.sql   RAG metadata schema
│
├── ai-worker/                       Python worker
│   ├── app/capabilities/
│   │   ├── mock_processor.py         MOCK capability
│   │   └── rag/                      RAG capability
│   │       ├── capability.py         entry point
│   │       ├── chunker.py            chunking logic
│   │       ├── embeddings.py         embedding provider
│   │       ├── generation.py         generation provider
│   │       ├── faiss_index.py        FAISS wrapper
│   │       ├── metadata_store.py     ragmeta DAO
│   │       ├── ingest.py             JSONL → chunks → FAISS
│   │       └── retriever.py          query → retrieved chunks
│   │
│   ├── scripts/build_rag_index.py    indexing CLI
│   ├── eval/                         evaluation harness
│   └── fixtures/                     small committed fixtures
│
├── frontend/                         minimal HTML test client
├── scripts/e2e_smoke.py              end-to-end smoke test
├── docker-compose.yml                Redis / PostgreSQL / MinIO profiles
├── .env.example                      environment variable reference
└── docs/
    ├── architecture.md
    ├── local-run.md
    ├── api-summary.md
    ├── tuning.md
    └── optuna-tuning-plan.md
```

---

## RAG evaluation results

검색 성능은 `anime_silver_200` 데이터셋을 기준으로 평가했습니다.

- Corpus: 1,764 documents
- Queries: 200 deterministic synthetic queries
- Environment: RTX 5080 16GB
- Embedder: `bge-m3`
- Vector index: FAISS `IndexFlatIP`
- Reranker: `BAAI/bge-reranker-v2-m3`

상세 리포트는 `ai-worker/eval/reports/` 아래 phase별 하위 디렉토리에 JSON과 Markdown으로 커밋되어 있습니다.

---

### Phase 0 — fixture baseline

Small committed fixtures 기준 baseline입니다.

| dataset | hit@5 | recall@5 | MRR | dup_rate | topk_gap | p50/p95 ret (ms) |
|---|---:|---:|---:|---:|---:|---:|
| anime (en, 8 docs) | 1.000 | 1.000 | 1.000 | 0.433 | 0.236 | 8.6 / 24.9 |
| kr_sample (kr, 10) | 1.000 | 1.000 | 1.000 | 0.340 | 0.233 | 8.7 / 27.4 |

OCR row는 현재 환경에 Tesseract / 언어팩이 없어 측정하지 않았습니다.

---

### Phase 2A — silver-200 cross-encoder reranker progression

| run | hit@1 | hit@3 | hit@5 | MRR@10 | NDCG@10 | rerank p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| B1 dense (combined-old) | 0.560 | 0.670 | 0.685 | 0.617 | 0.643 | – |
| B2 dense (token-aware-v1) | 0.540 | 0.665 | 0.680 | 0.604 | 0.631 | – |
| **B2 + rerank top20** | 0.605 | 0.680 | 0.700 | 0.653 | 0.675 | 706 |
| **B2 + rerank top50** | **0.615** | **0.700** | **0.715** | **0.666** | **0.689** | 1840 |

Reranker 적용 결과 dense baseline 대비 hit@1, hit@5, MRR@10, NDCG@10이 개선되었습니다.  
다만 rerank top50은 품질은 가장 좋지만 p95 latency가 커지므로, 실제 서비스 적용 시에는 품질과 지연 시간의 trade-off를 고려해야 합니다.

---

### Candidate-recall ceiling

Reranker는 후보 문서의 순서를 재배치할 뿐, dense retrieval 단계에서 후보에 들어오지 않은 문서는 복구할 수 없습니다.  
따라서 dense top-N의 recall은 reranker 성능의 상한선입니다.

B2 dense top-50 기준:

| metric | value |
|---|---:|
| hit@10 | 0.715 |
| hit@20 | 0.770 |
| hit@50 | 0.800 |

---

### Selected baseline

`legacy-baseline-final/` 기준 selected baseline:

| metric | value |
|---|---:|
| hit@1 | 0.620 |
| hit@3 | 0.675 |
| hit@5 | 0.705 |
| MRR@10 | 0.654 |
| total query p95 | 350 ms |

이 baseline은 agent loop legacy backend의 reference manifest로 사용됩니다.

---

## Reproduce evaluation

```bash
cd ai-worker
```

### Phase 2A — candidate recall

```bash
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --top-k 50 \
    --extra-hit-k 10 \
    --extra-hit-k 20 \
    --extra-hit-k 50 \
    --out-dir eval/reports/phase2a-reranker/candidate-recall-b2
```

### Phase 2A — rerank sweep

```bash
for N in 20 50; do
  python -m eval.run_eval retrieval-rerank \
      --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
      --dataset eval/eval_queries/anime_silver_200.jsonl \
      --dense-top-n $N \
      --final-top-k 10 \
      --reranker-model BAAI/bge-reranker-v2-m3 \
      --reranker-batch-size 16 \
      --out-dir eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top$N
done
```

### Phase 0 — fixture baseline

```bash
python -m eval.run_eval rag \
    --dataset eval/datasets/rag_sample_kr.jsonl \
    --offline-corpus fixtures/kr_sample.jsonl \
    --out-json eval/reports/baseline-phase0.json
```

---

## Local run

Mode A는 기존 PostgreSQL `:5432`를 재사용하는 기본 실행 방식입니다.  
Mode B는 독립 `postgres:18`을 `:5433`에서 실행하는 방식이며, 자세한 내용은 `docs/local-run.md`를 참고합니다.

```bash
# 1. Start Redis
docker compose up -d redis

# 2. Start core-api
cd core-api
mvn spring-boot:run

# 3. Start ai-worker
cd ../ai-worker
pip install -r requirements.txt
python -m app.main

# 4. Run smoke test
cd ..
python scripts/e2e_smoke.py
```

---

## Internal endpoint authentication

`/api/internal/**` 엔드포인트는 `X-Internal-Secret`으로 게이팅합니다.

| Service | Environment variable |
|---|---|
| core-api | `AIPIPELINE_INTERNAL_SECRET` |
| ai-worker | `AIPIPELINE_WORKER_INTERNAL_SECRET` |

두 서비스의 secret 값은 동일해야 합니다.  
둘 다 미설정인 경우 core-api는 개발 편의를 위해 WARN 로그를 남긴 뒤 pass-through합니다.

---

## Storage

기본 storage backend는 local filesystem입니다.

S3 / MinIO 전환을 위해 storage port가 분리되어 있습니다.

```bash
docker compose --profile minio up -d minio minio-bootstrap
```

MinIO 사용 시 주요 환경변수:

```bash
AIPIPELINE_STORAGE_BACKEND=s3
AIPIPELINE_STORAGE_S3_ENDPOINT=...
AIPIPELINE_WORKER_S3_ENDPOINT=...
AIPIPELINE_WORKER_S3_ACCESS_KEY=...
AIPIPELINE_WORKER_S3_SECRET_KEY=...
```

자세한 환경변수 목록은 `.env.example`을 참고합니다.

---

## Demo

`scripts/demo.py`는 샘플 PDF를 생성하고 capability pipeline을 실행하는 데모 스크립트입니다.

```bash
pip install reportlab rich

python scripts/demo.py
python scripts/demo.py --capability OCR --question "What metrics are shown?"
python scripts/demo.py --self-test
```

현재 OCR / multimodal capability는 planned 상태이므로, 실제 사용 가능 범위는 현재 구현 상태에 맞춰 확인해야 합니다.

---

## Development phases

| Phase | Description |
|---|---|
| Phase 1 | 비동기 skeleton: job / artifact / claim / callback / Redis dispatch / MOCK |
| Phase 1.1 | target stack 안정화: Spring Boot 4.0.3 / Java 21 / PostgreSQL 18 / Python 3.12 / Redis |
| Phase 2 | text-RAG capability: FAISS + sentence-transformers + extractive generator + ragmeta schema |
| Phase 1B | namu-wiki corpus prefix / inline-edit-marker strip |
| Phase 1C | token-aware chunker: 1024-token hard cap, avg ctx tokens 531 → 293 |
| Phase 2A | `bge-reranker-v2-m3` cross-encoder reranker evaluation |
| Phase 2.1 | OCR + file input flow |
| Phase 3+ | multimodal capability, real LLM generation, MinIO/S3 adapter, retry orchestration, K8s manifests |

---

## Roadmap

### Near-term

- OCR capability 추가
- file / PDF input pipeline 구성
- `INPUT_FILE → OCR_TEXT → RAG` 재소비 흐름 구현
- real LLM generation provider 추가
- RAG eval harness 정리
- tuning workflow 문서 개선

### Mid-term

- multimodal capability 추가
- S3 / MinIO artifact storage 안정화
- capability별 dispatch lane 분리
- retry orchestration 고도화
- frontend 개선

### Later

- Cloud Tasks / Pub/Sub / SQS adapter 검토
- Kubernetes manifest 추가
- GPU worker deployment 전략 정리
- production-like observability 추가

---

## Documentation

### Design / Operations / API

- `docs/architecture.md` — system architecture, design decisions, capability flows, trace schema
- `docs/api-summary.md` — HTTP endpoints, request/response shape, error codes
- `docs/local-run.md` — local bootstrap, troubleshooting, capability activation guide

### Evaluation

- `ai-worker/eval/README.md` — eval harness overview, metric definitions, recommended sequence
- `ai-worker/eval/datasets/README.md` — eval dataset catalog
- `ai-worker/eval/corpora/README.md` — retrieval corpus directory convention
- `ai-worker/eval/corpora/anime_namu_v3/README.md` — namu-wiki anime corpus schema
- `ai-worker/eval/eval_queries/README.md` — retrieval eval query sets

### Tuning

- `docs/tuning.md` — Optuna + Claude interpreter loop guide
- `docs/optuna-tuning-plan.md` — tuning pipeline roadmap
- `ai-worker/eval/experiments/README.md` — round-refinement workspace

### Frontend

- `frontend/app/README.md` — Vite + React + TypeScript setup notes

---

## Notes

이 프로젝트의 현재 초점은 “모든 AI capability를 완성하는 것”이 아니라,  
AI 작업을 안정적으로 실행할 수 있는 **백엔드 파이프라인 구조**와  
RAG 검색 품질을 측정하고 개선할 수 있는 **평가/튜닝 기반**을 만드는 것입니다.

따라서 OCR / multimodal / real LLM generation은 roadmap으로 남겨두고,  
현재는 다음 두 가지를 우선 검증했습니다.

1. 비동기 작업 처리 구조가 안정적으로 동작하는가
2. RAG 검색 파이프라인을 평가하고 개선할 수 있는가
