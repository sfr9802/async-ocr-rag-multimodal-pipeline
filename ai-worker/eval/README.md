# Evaluation harness

ai-worker capability 를 위한 가벼운 로컬 우선 eval harness. 프로덕션
파이프라인이 eval 코드를 절대 import 하지 않도록 (그 반대도) `app/`
과 의도적으로 분리되어 있습니다.

## 무엇이며 무엇이 아닌가

**무엇:** 한 줌의 순수 Python 메트릭, JSONL-in / JSON+CSV-out 리포트
writer, 그리고 한 명의 개발자가 트래킹 플랫폼, 클라우드 서비스, 모델
서버 없이 커맨드 라인에서 반복할 수 있는 두 개의 harness 함수
(`run_rag_eval`, `run_ocr_eval`).

**무엇이 아닌가:** 진짜 모델-품질 파이프라인의 대체. 실험 데이터베이스도,
리더보드도, 하이퍼파라미터 sweep driver 도, CI 게이팅도 없음. 프로젝트가
flat 파일을 능가할 때 교체 — `app/` 안의 어느 것도 여기 어떤 것에도
의존하지 않음.

## 디렉토리 구조

```
eval/
├── __init__.py
├── README.md                       ← 이 파일
├── run_eval.py                     ← CLI: `python -m eval.run_eval ...`
├── datasets/                       ← legacy `rag` / `ocr` / `multimodal` 모드
│   ├── rag_sample.jsonl            ← 커밋된 RAG 픽스처에 대한 6개 query
│   ├── ocr_sample.jsonl            ← 3개 OCR 행; 이미지는 helper 가 생성
│   └── multimodal_sample.jsonl     ← PLACEHOLDER 스키마 (아직 harness 없음)
├── corpora/                        ← `retrieval` 모드용 retrieval 코퍼스
│   └── anime_namu_v3/              ← gitignore 된 261-MB namu-wiki anime 샘플
├── eval_queries/                   ← retrieval-eval query 셋 (smoke / silver / gold)
├── reports/                        ← 생성됨 — 커밋하지 마세요
└── harness/
    ├── __init__.py                 ← public API 재 export
    ├── metrics.py                  ← CER/WER/hit@k/MRR/keyword coverage + retrieval 진단
    ├── io_utils.py                 ← JSONL loader + JSON/CSV writer
    ├── rag_eval.py                 ← text RAG harness (legacy `rag` 모드)
    ├── ocr_eval.py                 ← OCR harness
    ├── retrieval_eval.py           ← retrieval-quality harness (`retrieval` 모드)
    ├── miss_analysis.py            ← retrieval 실행에 대한 doc/keyword cross-tab
    ├── baseline_comparison.py      ← 3-slice 비교 (det vs det-without-X vs opus)
    ├── analyze_corpus_lengths.py   ← tokenizer 기반 char/token 길이 분석기
    └── generate_eval_queries.py    ← 결정적 + LLM 합성-query 생성기
```

## 데이터셋 스키마

### A. Text RAG eval — `eval/datasets/rag_sample.jsonl`

라인당 JSON 객체 하나. `#` 로 시작하는 라인과 빈 라인은 skip 되므로
파일 안에 인라인 코멘트 유지 가능.

| 필드                | 타입            | 필수 | 비고                                                                  |
|---------------------|-----------------|------|-----------------------------------------------------------------------|
| `query`             | string          | yes  | 테스트할 사용자 query.                                                |
| `expected_doc_ids`  | list<string>    | no   | top-k 에 기대하는 doc id. 이 필드 없는 행은 hit@k 와 MRR 집계에서 제외. |
| `expected_keywords` | list<string>    | no   | 생성된 답변에 포함되어야 하는 substring (대소문자 무시 매치).         |
| `notes`             | string          | no   | 작성자용 자유 형식 주석. 채점 안 함.                                  |

예시:

```jsonl
{"query": "who runs the bookshop at the end of the railway line?", "expected_doc_ids": ["anime-003"], "expected_keywords": ["bookshop", "translator"], "notes": "cozy mystery"}
```

### B. OCR eval — `eval/datasets/ocr_sample.jsonl`

| 필드           | 타입    | 필수 | 비고                                                                                  |
|----------------|---------|------|---------------------------------------------------------------------------------------|
| `file`         | string  | yes  | 이 JSONL 파일의 디렉토리에 **상대적인** 경로. 지원: `.png`, `.jpg`, `.jpeg`, `.pdf`. |
| `ground_truth` | string  | yes  | 정확한 기대 추출 텍스트 (UTF-8). CER/WER 패스에서 공백은 정규화됨.                    |
| `language`     | string  | no   | Tesseract 언어 코드 — `eng`, `eng+kor`, `kor`, `jpn`, `chi_sim` 등. CJK 코드의 경우 harness 가 WER 을 `None` 으로 보고하고 행을 WER 집계에서 제외 (whitespace-split WER 이 거기서는 의미 없음). |
| `notes`        | string  | no   | 자유 형식 주석.                                                                       |

예시:

```jsonl
{"file": "samples/hello_world.png", "ground_truth": "HELLO WORLD", "language": "eng", "notes": "bare minimum smoke test"}
```

### C. Multimodal eval — `eval/datasets/multimodal_sample.jsonl` (스키마만, 아직 harness 없음)

| 필드                | 타입         | 필수 | 비고                                                                       |
|---------------------|--------------|------|----------------------------------------------------------------------------|
| `image`             | string       | yes  | JSONL 파일에 상대적인 경로 (OCR 와 같은 규칙).                             |
| `question`          | string       | yes  | 이미지에 대한 자연어 질문.                                                 |
| `expected_answer`   | string       | no   | 정확/substring 매치를 위한 표준 짧은 답변.                                 |
| `expected_keywords` | list<string> | no   | 답변이 포함해야 하는 substring.                                            |
| `expected_labels`   | list<string> | no   | 모델이 식별해야 하는 시각적 레이블 (오브젝트 이름, 색상 등).                |
| `requires_ocr`      | bool         | no   | task 가 답변 전 텍스트 추출을 요구하면 True.                               |
| `language`          | string       | no   | `requires_ocr` 가 true 일 때 OCR/VLM 언어 코드.                            |
| `notes`             | string       | no   | 자유 형식 주석.                                                            |

**상태: placeholder 만.** 아직 multimodal harness 없음. 미래 phase 가
다시 데이터셋 디자인 라운드 없이 scorer 를 wire 할 수 있도록 스키마는
커밋되어 있음. 권장 미래 메트릭 (모두 `eval.harness.metrics` 에서
재사용 가능): exact match, substring match, `keyword_coverage`, label
recall/precision, 그리고 어떤 VLM-보고 OCR 서브필드에 대한 CER.

## 메트릭 (각각의 의미)

- **`hit@k`** — retriever 의 top-k 에 ANY `expected_doc_id` 가 등장하면
  1.0, 그 외 0.0. 행에 `expected_doc_ids` 가 없으면 `None`. "유용한
  것을 회상했는가" 에 대한 단순한 binary 게이트.
- **`reciprocal_rank` / `MRR`** — ranked 리스트에서 첫 매칭 expected
  id 의 1/rank, 없으면 0.0, `expected_doc_ids` 없는 행은 `None`. MRR
  은 non-None 행의 평균. 정답을 1번 위치 대신 5번 위치에 묻는 것을
  벌점.
- **`keyword_coverage`** — 생성된 답변에 substring (대소문자 무시) 으로
  존재하는 `expected_keywords` 의 비율. "generator 가 실제로 그것을
  언급했는가" 에 대한 가장 싼 합리적 신호.
- **`CER`** — 공백 정규화 후 character edit distance / 참조 character
  count. 0.0 이 완벽. ~0.1 이상이면 보통 눈에 띄게 저하된 OCR; ~0.3
  이상이면 출력이 다운스트림 retrieval 에 사용 불가.
- **`WER`** — 공백 분할 단어 레벨에서의 같은 것. CJK 언어에는 의미
  없음 — harness 가 그것들에 대해 `None` 보고.
- **`empty_rate`** — OCR 엔진이 정규화된 character 0 을 생성한 행의
  비율. 높은 empty_rate 는 보통 잘못 설정된 언어팩 또는 읽을 수 없는
  스캔.
- **latency (ms)** — 각 provider 호출의 wall-clock. p50, mean, max
  보고. 새 모델의 회귀 검사에 유용.

## 권장 평가 시퀀스

이 순서로 반복 — 각 단계의 메트릭이 다음을 게이팅:

### 1. Text RAG baseline

**목표:** retriever 가 OCR 을 건드리기 전에 "대부분 맞는 doc, 대부분
맞는 순서" 에 도달하는지 확인.

**실행:**
```bash
cd ai-worker
python -m scripts.build_rag_index --fixture     # 1회
python -m eval.run_eval rag \
  --dataset eval/datasets/rag_sample.jsonl \
  --top-k 5
```

**게이트:** 픽스처에서 `mean_hit_at_k ≥ 0.80` **그리고**
`MRR ≥ 0.50`. 이 아래로 떨어지면 문제는 임베딩/chunking/config 에 있음
— 아직 OCR 로 넘어가지 마세요. `top_k` 올리기 (`--top-k 10`) 는 진단
용으로만 사용하고 fix 로 사용하지 마세요.

### 2. OCR 추출 품질

**목표:** OCR provider 가 *retrieval 입력으로 충분히 깨끗한* 텍스트를
생성하는지 확인. 이는 결합 단계의 사전 조건.

**실행:**
```bash
python -m scripts.make_ocr_sample_fixtures       # 1회
python -m eval.run_eval ocr \
  --dataset eval/datasets/ocr_sample.jsonl
```

**게이트:** 샘플 픽스처에서 `mean_cer ≤ 0.10` 와
`empty_rate == 0.0`. 진짜 큐레이션된 스캔에서는 `mean_cer ≤ 0.20` 이
보통 RAG 다운스트림이 여전히 정상적으로 동작하는 임계값.

CER 이 0.25 위면 진행하지 마세요 — provider 부터 디버그. 흔한 원인:
누락된 언어팩, 너무 낮은 `ocr_pdf_dpi`, low-contrast 스캔에 누락된
전처리.

### 3. OCR + RAG 결합 (미래, 아직 자동화 안 됨)

**목표:** OCR 출력이 진짜 문서에 대해 RAG 입력으로 실제로 사용 가능
한지 end-to-end 테스트.

**거기 도달했을 때:** `{file, ground_truth_text, query,
expected_doc_ids}` 의 작은 결합 데이터셋 빌드, OCR harness 를 돌려
추출된 텍스트 생성, 추출된 텍스트를 `query` 로 RAG harness 에 먹임,
두 hit@k 숫자 비교:

- RAG-on-ground-truth: 상한, OCR 을 완벽하다고 취급.
- RAG-on-OCR-output: 실제 동작.

delta 가 OCR 이 너에게 비용으로 부과하는 것. **게이트:** 두 숫자가
서로 0.10 이내여야 함; 그보다 나쁘면 OCR 품질이 진짜 입력에 대한
retrieval 을 막고 있다는 뜻. **이 단계는 현재 수동** — 이 문서 하단의
"여전히 수동인 것" 참조.

### 4. Multimodal

**목표:** 풀 multimodal 파이프라인 (OCR + vision + fusion + retrieval +
generation) 이 올바른 키워드를 언급하고 올바른 시각적 레이블을 표면화
하는 답변을 생성하는지 확인.

**실행:**
```bash
python -m scripts.make_multimodal_sample_fixtures    # 1회
python -m eval.run_eval multimodal \
  --dataset eval/datasets/multimodal_sample.jsonl
```

**게이트:** 픽스처에서 `mean_keyword_coverage >= 0.60` **그리고**
`mean_substring_match >= 0.50`. 이 아래로 떨어지면 문제는 vision
provider 또는 fusion 단계일 가능성 — stage 단위 진단을 위해
MULTIMODAL_TRACE artifact 확인.

**필터링:** OCR 의존 행만 평가하려면 `--require-ocr-only` 사용. A/B
비교를 위해 vision provider override 하려면
`--vision-provider heuristic|claude` 사용.

**Stage 단위 latency 분석:** `emit_trace=True` (eval 실행 중 자동
활성화) 일 때 harness 가 OCR, vision, retrieval+generation latency 를
별도로 보고. 어느 stage 가 병목인지 식별하는 데 사용.

## `retrieval` 모드 — dense-retrieval baseline 측정

`rag` 모드 (generator 출력도 점수 매김) 와 별도로, `retrieval` 모드는
dense-retrieval 단계 **만** 측정. generator 가 호출되지 않고, 출력은
실행당 4개의 artifact:

```
eval/reports/retrieval-<timestamp>/
├── retrieval_eval_report.json     summary + 행별 메트릭
├── retrieval_eval_report.md       사람이 읽을 수 있는 요약
├── top_k_dump.jsonl               (query, rank) 쌍당 1개 레코드
└── duplicate_analysis.json        query 별 + 집계 dup 통계
```

**행별 메트릭**: hit@1, hit@3, hit@5, mrr@10, ndcg@10 (binary
relevance), dup_rate, unique_doc_coverage, top1_score_margin,
avg_context_token_count, expected_keyword_match_rate.

오프라인 anime 코퍼스에 대해 **실행** (Postgres 필요 없음):

```bash
cd ai-worker

# 1. 코퍼스 stage (1회, eval/corpora/anime_namu_v3/README.md 참조)
cp 'D:/port/rag/app/scripts/namu_anime_v3.jsonl' \
   eval/corpora/anime_namu_v3/corpus.jsonl

# 2. 6개 수작업 query 에 대한 스모크 테스트
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3/corpus.jsonl \
    --dataset eval/eval_queries/anime_smoke_6.jsonl \
    --top-k 10

# 3. 풀 silver baseline (200개 결정적 합성 query)
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3/corpus.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --top-k 10

# 4. Gold baseline (20개 수작업 큐레이션 query — ground truth 로 신뢰)
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3/corpus.jsonl \
    --dataset eval/eval_queries/anime_gold_20.jsonl \
    --top-k 10
```

`retrieval` 모드는 라이브 ragmeta/FAISS 경로도 받아들임 (`--corpus`
생략) — `--offline-corpus` 플래그 없는 legacy `rag` 모드와 같은 동작.

silver vs. gold 의미, 결정적 generator 의 stratification, gold-set
확장 레시피는 [`eval_queries/README.md`](eval_queries/README.md) 참조.
코퍼스 스키마와 re-stage 지침은
[`corpora/anime_namu_v3/README.md`](corpora/anime_namu_v3/README.md)
참조.

### Phase 0 baseline 도구 (post-retrieval)

세 개의 동반 서브커맨드가 기존 retrieval 실행에 대해 동작. 어느 것도
재임베딩하지 않으므로 비용이 쌈.

```bash
# 발행하지 않은 retrieval 실행에 doc/keyword cross-tab 추가
python -m eval.run_eval retrieval-miss-analysis \
    --report-dir eval/reports/retrieval-silver200-baseline \
    --top-k 10

# 두 개의 retrieval 실행을 나란히 비교 (deterministic vs opus).
# .md 에 Caveat 블록 + slice 별 retriever_config 자동 발행.
python -m eval.run_eval retrieval-compare \
    --deterministic-report eval/reports/retrieval-silver200-baseline/retrieval_eval_report.json \
    --opus-report          eval/reports/retrieval-silver200-opus-baseline/retrieval_eval_report.json \
    --deterministic-max-seq-length 8192 \
    --opus-max-seq-length 1024 \
    --out-json eval/reports/retrieval-baseline-comparison.json \
    --out-md   eval/reports/retrieval-baseline-comparison.md

# 같은 비교지만 deterministic 측에 hyperparameter-tuned 변형 포함.
# 튜닝된 slice (와 그 진단) 는 자체 headline-metrics 표에 렌더링되어
# baseline 숫자와 행을 공유하지 않음.
python -m eval.run_eval retrieval-compare \
    --deterministic-report eval/reports/retrieval-silver200-tuned/retrieval_eval_report.json \
    --opus-report          eval/reports/retrieval-silver200-opus-baseline/retrieval_eval_report.json \
    --deterministic-kind tuned \
    --opus-kind baseline \
    --out-json eval/reports/retrieval-tuned-vs-baseline.json \
    --out-md   eval/reports/retrieval-tuned-vs-baseline.md

# 코퍼스에 대한 tokenizer 기반 char/token 길이 분포.
# char 유추 추측 대신 측정된 숫자로 max_seq_length cap 사이즈 결정에 사용.
python -m eval.run_eval analyze-corpus-lengths \
    --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
    --tokenizer BAAI/bge-m3 \
    --out-json eval/reports/corpus-length-analysis.json \
    --out-md   eval/reports/corpus-length-analysis.md
```

전체 Phase 0 trade-off 로그는
[`reports/phase0-baseline-tradeoffs.md`](reports/phase0-baseline-tradeoffs.md)
에 있음.

### Phase 2A — cross-encoder reranker (`retrieval-rerank`)

`retrieval-rerank` 서브커맨드는 dense retriever 위에 cross-encoder
reranker 를 후처리로 끼움. dense top-N candidate 를 가져와 cross-encoder
로 재정렬한 뒤 final top-K 를 점수. corpus / chunker / preprocessor 는
건드리지 않음 — 순수 retrieval 후처리.

```bash
# 1. Candidate-recall 진단 — reranker 성능 상한 측정.
#    NoOp reranker + top-k=50 + extra-hit-k 로 hit@1/3/5/10/20/50 계산.
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --top-k 50 \
    --extra-hit-k 10 --extra-hit-k 20 --extra-hit-k 50 \
    --out-dir eval/reports/phase2a-reranker/candidate-recall-b2

# 2. dense top-20 → cross-encoder rerank → top-10
python -m eval.run_eval retrieval-rerank \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --dense-top-n 20 \
    --final-top-k 10 \
    --reranker-model BAAI/bge-reranker-v2-m3 \
    --reranker-batch-size 16 \
    --out-dir eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top20

# 3. dense top-50 → cross-encoder rerank → top-10
python -m eval.run_eval retrieval-rerank \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --dense-top-n 50 \
    --final-top-k 10 \
    --reranker-model BAAI/bge-reranker-v2-m3 \
    --reranker-batch-size 16 \
    --out-dir eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top50

# 4. 5-slice 비교 (B1 dense / B2 dense / candidate-recall / rerank top20 / rerank top50)
python -m eval.run_eval phase2a-reranker-comparison \
    --slice "B1 dense (combined-old):eval/reports/retrieval-silver200-combined-old-chunker/retrieval_eval_report.json" \
    --slice "B2 dense (token-aware-v1):eval/reports/retrieval-silver200-combined-token-aware-v1/retrieval_eval_report.json" \
    --slice "B2 dense top50 (candidate-recall):eval/reports/phase2a-reranker/candidate-recall-b2/retrieval_eval_report.json" \
    --slice "B2 rerank top20:eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top20/retrieval_eval_report.json" \
    --slice "B2 rerank top50:eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top50/retrieval_eval_report.json" \
    --out-json eval/reports/phase2a-reranker/reranker-comparison.json \
    --out-md   eval/reports/phase2a-reranker/reranker-comparison.md

# 5. Failure analysis (dense top-10 vs rerank top-20 cross-tab)
python -m eval.run_eval phase2a-reranker-failure-analysis \
    --dense-report-dir  eval/reports/retrieval-silver200-combined-token-aware-v1 \
    --rerank-report-dir eval/reports/retrieval-silver200-combined-token-aware-v1-rerank-top20 \
    --out-dir eval/reports/phase2a-reranker \
    --k-preview 5 --sample-cap 10
```

**Phase 2A silver-200 결과** (RTX 5080 / bge-m3 + bge-reranker-v2-m3):

| run                              | hit@1 | hit@3 | hit@5 | MRR@10 | NDCG@10 | rerank p95 (ms) |
|----------------------------------|------:|------:|------:|-------:|--------:|----------------:|
| B1 dense (combined-old)          | 0.5600 | 0.6700 | 0.6850 | 0.6167 | 0.6428 |               – |
| B2 dense (token-aware-v1)        | 0.5400 | 0.6650 | 0.6800 | 0.6044 | 0.6314 |               – |
| **B2 + rerank top20**            | 0.6050 | 0.6800 | 0.7000 | 0.6526 | 0.6748 |             706 |
| **B2 + rerank top50**            | **0.6150** | **0.7000** | **0.7150** | **0.6657** | **0.6885** |       1840 |

Candidate recall ceiling (B2 dense top-50): hit@10=0.7150, hit@20=0.7700,
hit@50=0.8000. reranker 는 candidate set 안의 순서만 바꿀 수 있으므로 이
값들이 reranker hit@k 의 이론적 상한.

**Caveat**

- rerank latency 는 cross-encoder predict 만의 wall-clock — bi-encoder +
  FAISS 부분은 `mean_retrieval_ms` 에 별도로 잡힘. p95 한 번에 700–1800ms 는
  query-time UX 에 무거우므로 production default 로 승격하기 전에 batch
  처리 / async 호출 설계 필요.
- B1 (combined-old) 와 B2 (combined-token-aware-v1) 는 chunk granularity 가
  다르므로 candidate population 자체가 동일하지 않음 — 직접 비교 시 chunker
  효과 + reranker 효과가 섞여 있음.
- production default 는 여전히 `rag_reranker="off"` (NoOp). reranker 는
  eval CLI 에서만 활성화하며, registry 의 `cross_encoder` 분기를 production
  으로 켤지는 별도 결정.

### Phase 2A-L — reranker latency profiling (`phase2a-latency-sweep`)

`phase2a-latency-sweep` 는 Phase 2A 의 정확도 결과(top20 / top50) 위에서
**latency 분해 + accuracy ↔ latency Pareto frontier + 운영 모드 추천** 을
한 번의 명령으로 만들어내는 evaluation 모드. 정확도 개선이 아니라 reranker
의 latency budget 을 정량화하기 위한 도구.

```bash
# silver-200 + B2 token-aware corpus 에서 6 개 dense_top_n sweep.
# corpus 는 한 번만 빌드, 6 번의 retrieval-rerank 가 동일 인덱스 위에서 돌고,
# 추가로 dense-only candidate-recall sibling 이 hit@10/20/50 상한을 잡는다.
python -m eval.run_eval phase2a-latency-sweep \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --corpus  eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --out-dir eval/reports/phase2a-latency \
    --final-top-k 10 \
    --dense-top-n 5  --dense-top-n 10 --dense-top-n 15 \
    --dense-top-n 20 --dense-top-n 30 --dense-top-n 50 \
    --breakdown-anchor-dense-top-n 20 \
    --candidate-recall-extra-hit-k 10 --candidate-recall-extra-hit-k 20 --candidate-recall-extra-hit-k 50 \
    --metric mean_hit_at_1 \
    --latency rerank_p95_ms
```

산출물 (`eval/reports/phase2a-latency/`):

- `rerank-top{N}/retrieval_eval_report.{json,md}` — 각 config 의 표준
  retrieval-rerank 산출물 (rerank_breakdown_ms 포함).
- `candidate-recall/retrieval_eval_report.{json,md}` — dense-only sibling
  (hit@10/20/50 상한).
- `reranker-latency-breakdown.{json,md}` — anchor config(기본 top20)
  에서의 stage breakdown: `pair_build_ms`, `tokenize_ms`, `forward_ms`,
  `postprocess_ms`, `total_rerank_ms`, `dense_retrieval_ms`,
  `total_query_ms`. 각 stage 마다 avg/p50/p90/p95/p99/max.
- `topn-sweep.{json,md}` — 6 config × accuracy/latency 표.
- `accuracy-latency-frontier.{json,md}` — Pareto frontier (정확도 ↑ / latency ↓
  로 dominance 판정, dominated 항목은 dominator 라벨 명시).
- `recommended-modes.{json,md}` — fast/balanced/quality 후보. budget 옵션
  (`--fast-p95-budget-ms`, `--balanced-p95-budget-ms`,
  `--quality-target-metric`) 으로 선정 기준 조정 가능.

Post-processing 만 다시 돌리고 싶으면 분리 모드 사용:

```bash
# 단일 retrieval-rerank report 의 latency breakdown.
python -m eval.run_eval phase2a-latency-breakdown \
    --report eval/reports/phase2a-latency/rerank-top20/retrieval_eval_report.json \
    --out-json eval/reports/phase2a-latency/reranker-latency-breakdown.json \
    --out-md   eval/reports/phase2a-latency/reranker-latency-breakdown.md

# N 개 retrieval-rerank report → topN sweep.
python -m eval.run_eval phase2a-topn-sweep \
    --slice "top10:eval/reports/phase2a-latency/rerank-top10/retrieval_eval_report.json" \
    --slice "top20:eval/reports/phase2a-latency/rerank-top20/retrieval_eval_report.json" \
    --slice "top50:eval/reports/phase2a-latency/rerank-top50/retrieval_eval_report.json" \
    --candidate-recall-report eval/reports/phase2a-latency/candidate-recall/retrieval_eval_report.json \
    --out-json eval/reports/phase2a-latency/topn-sweep.json \
    --out-md   eval/reports/phase2a-latency/topn-sweep.md

# topn-sweep.json → Pareto frontier + recommended modes.
python -m eval.run_eval phase2a-recommended-modes \
    --sweep-json   eval/reports/phase2a-latency/topn-sweep.json \
    --out-md       eval/reports/phase2a-latency/recommended-modes.md \
    --out-modes-json eval/reports/phase2a-latency/recommended-modes.json \
    --out-frontier-json eval/reports/phase2a-latency/accuracy-latency-frontier.json \
    --out-frontier-md   eval/reports/phase2a-latency/accuracy-latency-frontier.md
```

**Stage breakdown 측정 메모**

- `CrossEncoderReranker(collect_stage_timings=True)` 일 때만 stage 별 timing
  이 잡힌다. Production default 는 `False` 이므로 retriever / RAG capability /
  registry 경로는 byte-identical.
- `tokenize_ms` 는 host-side; `forward_ms` 는 host→device 전송 + model
  forward + activation 로, GPU 위에서는 `torch.cuda.synchronize()` 로 양쪽
  경계를 잡고 측정. CPU-only 실행에서는 sync 가 no-op 이므로 forward_ms 가
  실측치보다 약간 흐려질 수 있다.
- OOM-fallback path (CUDA OOM 발생 후 half-batch 재시도) 에서는 stage
  breakdown 이 None — 두 batch_size 의 측정을 섞어 보고하는 것을 의도적으로
  피한 결과.
- 이 mode 는 production default 를 변경하지 않는다. `recommended-modes.md`
  는 의사결정 근거이지 자동 적용되는 config 가 아니다.

## eval CLI 실행

두 서브커맨드 모두 stdout 에 짧은 사람용 요약을 출력하고 JSON 리포트
(그리고 기본적으로 CSV) 를 작성. `--out-json` / `--out-csv` 가
전달되지 않으면 리포트는 `eval/reports/{mode}-{timestamp}.{json,csv}`
로 갑니다.

```bash
# Text RAG — 진짜 프로덕션 스택 빌드 (bge-m3 + FAISS + ragmeta)
python -m eval.run_eval rag \
    --dataset eval/datasets/rag_sample.jsonl \
    --out-json eval/reports/rag-latest.json \
    --out-csv  eval/reports/rag-latest.csv \
    --top-k 5

# OCR — 진짜 Tesseract + PyMuPDF provider 빌드
python -m eval.run_eval ocr \
    --dataset eval/datasets/ocr_sample.jsonl \
    --out-json eval/reports/ocr-latest.json \
    --out-csv  eval/reports/ocr-latest.csv

# Multimodal — 풀 MULTIMODAL capability 빌드 (OCR + vision + RAG)
python -m eval.run_eval multimodal \
    --dataset eval/datasets/multimodal_sample.jsonl \
    --out-json eval/reports/multimodal-latest.json \
    --out-csv  eval/reports/multimodal-latest.csv

# Multimodal — Claude vision provider 로 OCR 전용 행
python -m eval.run_eval multimodal \
    --dataset eval/datasets/multimodal_sample.jsonl \
    --require-ocr-only \
    --vision-provider claude
```

CLI 플래그:

| 플래그              | 적용 대상 | 기본값                                                    |
|---------------------|-----------|-----------------------------------------------------------|
| `--dataset PATH`    | 양쪽      | (필수)                                                    |
| `--out-json PATH`   | 양쪽      | `eval/reports/{mode}-<timestamp>.json`                    |
| `--out-csv PATH`    | 양쪽      | `eval/reports/{mode}-<timestamp>.csv`                     |
| `--no-csv`          | 양쪽      | JSON 만 발행                                              |
| `-v` / `--verbose`  | 양쪽      | DEBUG 로깅                                                |
| `--top-k N`         | rag       | worker 의 `AIPIPELINE_WORKER_RAG_TOP_K` (기본 `5`)        |
| `--fail-missing`    | ocr       | 누락된 픽스처 파일을 skip 대신 에러로 취급                |

## 프로그래매틱 사용

Harness 는 단위 테스트 또는 커스텀 runner 용으로 import 가능:

```python
from eval.harness import (
    run_rag_eval, run_ocr_eval,
    cer, wer, hit_at_k, reciprocal_rank, keyword_coverage,
)
```

`run_rag_eval` 와 `run_ocr_eval` 모두 이미 생성된 retriever/generator/
provider 객체를 받음 — config 결합 없음 — 그래서 테스트는 fake 를
넘길 수 있고 커스텀 runner 는 이 패키지를 건드리지 않고 GPU 백엔드
provider 를 넘길 수 있음.

## 무시 규칙 / `.gitignore`

리포트는 생성되며 커밋되어서는 안 됩니다. 로컬에서 git 을 사용한다면
다음 추가:

```
ai-worker/eval/reports/
ai-worker/eval/datasets/samples/
```

`samples/` 폴더는 `scripts/make_ocr_sample_fixtures.py` 가 만든 합성
OCR 픽스처 이미지를 보유; 재생성은 비용이 싸고 폰트에 따라 표류함.

## 여전히 수동 / 아직 자동화되지 않은 것

이것들은 의도적인 phase-범위 라인이지 간과가 아닙니다:

1. **데이터셋 큐레이션.** JSONL 파일을 손으로 작성. 스크래핑 없음, LLM
   생성 질문 없음. 한 명의 개발자의 반복 loop 에서는 작은 큐레이션 셋이
   큰 노이즈 셋을 이김.
2. **OCR → RAG chaining eval.** OCR 출력을 RAG 입력으로 받는 harness
   가 없음. 지금은 OCR harness 실행, OCR_TEXT artifact 를 새 RAG
   데이터셋의 `query` 필드에 복사, RAG harness 재실행, 그 다음 ground-
   truth RAG 실행과 눈으로 비교. 아키텍처가 이 흐름을 지원 —
   [architecture.md](../../docs/architecture.md) 참조 — 그러나 자동화는
   나중 phase 의 몫.
3. **Multimodal 스코어링.** 스키마 커밋, harness 보류.
4. **CI 의 회귀 게이팅.** 메트릭 품질에 관계없이 성공한 실행에서
   harness 는 0 으로 종료. CI 게이팅을 원할 때 CLI 를 JSON 리포트를
   읽고 예: `mean_cer > 0.15` 면 실패하는 후속 실행 체크로 감싸세요.
   임계값을 harness 자체에 baking 하지 마세요 — 그것들은 도구가 아니라
   프로젝트의 release 기준 옆에 살아야 함.
5. **머신 전반의 latency baseline.** 리포트의 latency 는 raw wall-
   clock 숫자. 한 머신 / 한 실행 안에서만 비교 가능. 머신 전반 비교는
   고정 하드웨어 harness 가 필요하고, 이는 여기 범위 밖.
