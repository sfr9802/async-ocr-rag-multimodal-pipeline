# Tier 1 — 실전 VLM + LLM + 한국어 지원

> 이 파일은 새 Claude Code 세션에 그대로 붙여넣기 위한 self-contained 프롬프트입니다.
> 작업 루트: `D:\async-ocr-rag-multimodal-pipeline`

---

# 작업: Multimodal/RAG capability에 실제 Claude Vision + Claude LLM generation을 연결하고 한국어 end-to-end 증명을 추가한다

## 배경
이 프로젝트(`D:\async-ocr-rag-multimodal-pipeline`)는 async OCR/RAG/multimodal 파이프라인이며,
core-api(Spring Boot 4.0.3 / Java 21)와 ai-worker(Python 3.12, Redis BRPOP 소비자)로 구성되어 있다.
현재 상태에서 두 가지 치명적인 갭이 있다:

1. `ai-worker/app/capabilities/multimodal/heuristic_vision.py`는 Pillow.ImageStat 기반
   (brightness/contrast/dominant_channel) **구조 통계**일 뿐, 실제 VLM이 아니다.
2. `ai-worker/app/capabilities/rag/generation.py::ExtractiveGenerator`는 쿼리-문장
   단어 겹침(set intersection) heuristic이다. LLM generation이 아니다.
3. 한국어로 동작하는 증거가 없다. 픽스처는 영어 anime 데이터이고 eval도 영어다.
4. `VisionDescriptionProvider`와 `GenerationProvider` 시임은 이미 깔끔하게 추상화되어 있다.

## 먼저 읽어야 할 파일 (reading list — 순서대로)
1. `README.md` — 전체 구조
2. `docs/architecture.md` — phase 2 multimodal 설명 및 "Multimodal v1 limitations"
3. `ai-worker/app/capabilities/multimodal/vision_provider.py` — VisionDescriptionProvider ABC
4. `ai-worker/app/capabilities/multimodal/heuristic_vision.py` — 유지되는 fallback 구현
5. `ai-worker/app/capabilities/multimodal/capability.py` — VLM provider 사용 지점
6. `ai-worker/app/capabilities/rag/generation.py` — GenerationProvider ABC + Extractive
7. `ai-worker/app/capabilities/registry.py` — graceful degradation 패턴, `_build_vision_provider`, `_get_shared_retriever_bundle`
8. `ai-worker/app/core/config.py` — pydantic-settings 방식
9. `ai-worker/fixtures/anime_sample.jsonl` — 픽스처 스키마
10. `ai-worker/eval/README.md` + `eval/datasets/rag_sample.jsonl` + `eval/harness/rag_eval.py`
11. `ai-worker/tests/test_multimodal_capability.py` — 테스트 패턴

## 목표
1. `ClaudeVisionProvider` 신규 구현 — 실제 Claude Vision API 호출, 기본 운영 기본값으로 전환
2. `ClaudeGenerationProvider` 신규 구현 — grounded LLM 답변 생성, extractive는 fallback 유지
3. 한국어 픽스처, 한국어 OCR/RAG eval 데이터셋, README 상단 실측 벤치마크 테이블
4. Heuristic vision / Extractive generator는 **제거하지 말고** CI/오프라인/테스트용 fallback으로 유지

## 작업 A — ClaudeVisionProvider

### A.1 신규 파일
`ai-worker/app/capabilities/multimodal/claude_vision.py`

### A.2 구현 요구사항
- `VisionDescriptionProvider` 상속
- `anthropic` Python SDK 사용 (requirements.txt에 `anthropic>=0.40.0` 추가)
- 기본 모델: `claude-sonnet-4-6` (2026년 기준 Sonnet 4.6)
- System prompt (한국어+영어 혼용 문서 대응):
  > "You are a document-aware vision assistant. Analyze the image and produce:
  > (1) a single-sentence factual caption of the main subject,
  > (2) 3-5 bullet points of salient visual details,
  > (3) verbatim transcription of any visible text.
  > Respond in Korean if the image contains Korean text OR if the hint is in Korean; otherwise respond in English.
  > Never speculate beyond what is visible."
- User message: image (base64) + optional hint
- Parameters: `temperature=0`, `max_tokens=512`
- Timeout 30초, 재시도 최대 2회 (`httpx` transient errors 또는 `anthropic.APIStatusError` 5xx)
- 실패 시 `VisionError` 로 감싸되 코드 분리: `VLM_API_FAILED` / `VLM_TIMEOUT` / `VLM_RATE_LIMIT` / `VLM_BAD_RESPONSE`
- 응답 파싱: 모델 응답 텍스트에서 caption 첫 문장 추출, 나머지는 bullet details 리스트로 split
- `VisionDescriptionResult`에 `provider_name="claude-vision-v1"`, `latency_ms`, `page_number` 채우기

### A.3 Settings 확장
`ai-worker/app/core/config.py`의 `WorkerSettings`에 필드 추가:
- `anthropic_api_key: SecretStr | None = None` (env: `AIPIPELINE_WORKER_ANTHROPIC_API_KEY`)
- `multimodal_vision_provider` 기존 필드의 허용값 확장: `"heuristic" | "claude"`
- `multimodal_claude_vision_model: str = "claude-sonnet-4-6"`
- `multimodal_claude_timeout_seconds: float = 30.0`

### A.4 Registry 배선
`registry.py::_build_vision_provider`에 `elif provider_name == "claude":` 분기 추가.
- API key가 None이면 `RuntimeError("ANTHROPIC_API_KEY missing")` 발생 → 상위 try/except가 "MULTIMODAL NOT registered" 경고로 graceful 처리 (기존 패턴 그대로 활용)
- heuristic 분기는 기본값으로 유지

## 작업 B — ClaudeGenerationProvider

### B.1 신규 파일
`ai-worker/app/capabilities/rag/claude_generation.py`

### B.2 구현 요구사항
- `GenerationProvider` 상속, `name = "claude-generation-v1"`
- `anthropic` SDK 재사용 (작업 A와 동일 client)
- 모델 기본값: `claude-sonnet-4-6`
- System prompt 핵심 규칙:
  > "You are a retrieval-augmented answer generator.
  > Answer ONLY from the supplied passages. Cite every claim as `[doc_id#section]`.
  > If the passages don't contain the answer, respond exactly:
  > 한국어 쿼리: '제공된 자료에서 답을 찾을 수 없습니다.'
  > English query: 'The provided sources do not contain an answer.'
  > Never use outside knowledge. Never hallucinate document IDs."
- User message 구성:
  ```
  질문: {query}

  관련 자료:
  [1] doc_id#section (score=0.845)
  {chunk text}

  [2] doc_id#section (score=0.720)
  ...
  ```
- 출력 형식: extractive와 **동일한 3-part markdown** (Short answer / Supporting passages / Sources)
  유지 → 다운스트림 FINAL_RESPONSE 소비자 호환
- Temperature 0, max_tokens 1024
- Fallback 정책: API 호출 실패 시 `rag_generator_fallback_on_error=True`이면 extractive로 자동 폴백 + 경고 로그,
  False이면 `CapabilityError`로 전파

### B.3 Settings 확장
- `rag_generator: Literal["extractive","claude"] = "extractive"` (CI/테스트 기본값 유지)
- `rag_claude_generation_model: str = "claude-sonnet-4-6"`
- `rag_generator_fallback_on_error: bool = True`
- `rag_claude_timeout_seconds: float = 60.0`

### B.4 Registry 배선
`_get_shared_retriever_bundle` 수정:
- 기존 `generator = ExtractiveGenerator()` 부분을 설정값 분기로 변경
- `claude` 선택 시 API key 없으면 `RuntimeError` → 상위가 RAG NOT registered 처리
- Cache key에 `settings.rag_generator`, `settings.rag_claude_generation_model` 추가

## 작업 C — 한국어 end-to-end 증명

### C.1 신규 픽스처
`ai-worker/fixtures/kr_sample.jsonl` — 10~15개 한국어 문서. 스키마는 `anime_sample.jsonl`과 동일.
도메인은 일반적인 것: IT 뉴스, 사내 위키, FAQ 스타일. 다음 필드를 반드시 포함:
`doc_id`, `title`, `sections` (list of `{section, text}`)

### C.2 신규 eval 데이터셋
- `ai-worker/eval/datasets/rag_sample_kr.jsonl` — 8-10개 한국어 쿼리
  필드: `query`, `expected_doc_ids`, `expected_keywords`, `notes` (한국어)
- `ai-worker/eval/datasets/ocr_sample_kr.jsonl` — 3-5개 row
  필드: `file`, `ground_truth` (한국어), `language: "kor"`, `notes`

### C.3 한국어 OCR 샘플 생성기 확장
`ai-worker/scripts/make_ocr_sample_fixtures.py` 수정:
- 한국어 폰트로 PNG 생성 (NanumGothic / Malgun Gothic / D2Coding 중 시스템에 있는 것 자동 탐지)
- Windows면 `C:\Windows\Fonts\malgun.ttf` fallback
- 폰트를 찾지 못하면 스킵 + 명시적 경고 (harness 실패시키지 말 것)

### C.4 임베딩 모델 한국어 호환성 확인 + 기본값 변경
`ai-worker/app/core/config.py`에서 `rag_embedding_model` 기본값을 한국어/영어 모두 커버하는
**`BAAI/bge-m3`**로 변경. 기존에 이미 bge-m3이면 그대로 둔다.
`docs/architecture.md`에 "default embedder is multilingual (bge-m3)" 문장 추가.

### C.5 Index builder 확장
`ai-worker/scripts/build_rag_index.py`:
- `--fixture` 인자에 `en | kr | both` 허용 (기본 `en`, 호환성 유지)
- `both`는 두 픽스처를 병합하여 단일 인덱스로 빌드
- `build.json`에 `languages: ["en","kr"]` 필드 추가

### C.6 README 벤치마크 테이블
`README.md` 최상단 (프로젝트 한 줄 설명 바로 아래)에 다음 섹션 추가:

```markdown
## Benchmarks (measured on this branch)

| Dataset            | Capability | hit@5 | MRR  | CER (OCR) | p50 latency |
|--------------------|------------|-------|------|-----------|-------------|
| anime (en, 8 docs) | RAG        | 0.XX  | 0.XX | n/a       | XXms        |
| kr_sample (kr, N)  | RAG        | 0.XX  | 0.XX | n/a       | XXms        |
| ocr_sample (en)    | OCR        | n/a   | n/a  | 0.XX      | XXms        |
| ocr_sample (kr)    | OCR        | n/a   | n/a  | 0.XX      | XXms        |

> Measured with claude-sonnet-4-6 vision + generation, bge-m3 embedder,
> FAISS IndexFlatIP, Tesseract kor+eng.
> Reproduce: `python -m eval.run_eval rag --dataset eval/datasets/rag_sample_kr.jsonl`
```

실제 숫자는 eval을 돌려서 채운다 — 플레이스홀더 금지. eval을 돌릴 수 없는 환경이라면
**해당 셀을 "not measured"로 표기**하고 그 이유를 각주로 적는다.

### C.7 Architecture 문서 업데이트
`docs/architecture.md`에서:
- "Multimodal v1 limitations" 섹션의 "Vision quality is intentionally mock-grade" 항목을
  "Default vision provider is claude-sonnet-4-6; heuristic remains as offline/CI fallback" 로 수정
- "Phase 2 addition: text-RAG capability" 섹션에 generator 선택지 설명 문장 추가

## 기존 패턴 준수
- 임포트 실패(`anthropic` 미설치) 시 VisionError가 아닌 ImportError는 registry 상위에서
  "NOT registered" 경고로 graceful 처리 — 로컬 임포트를 builder 함수 안에 유지할 것
- 모든 새 Provider는 기존 ABC의 `name` property와 예외 계층(`VisionError`, `GenerationError`)을 준수
- 새 설정값은 반드시 `.env.example`에 추가
- 주석은 기존 파일 톤(Why+How, 영어)에 맞춘다

## 테스트 요구사항
새 테스트 파일:
1. `ai-worker/tests/test_claude_vision_provider.py`
   - anthropic client를 `monkeypatch`로 mock, 정상 응답 파싱 검증
   - 5xx → retry 2회 후 `VLM_API_FAILED` VisionError
   - timeout → `VLM_TIMEOUT`
   - 빈 응답 → `VLM_BAD_RESPONSE`
2. `ai-worker/tests/test_claude_generation.py`
   - mock client로 정상 답변 생성, 3-part markdown 구조 검증
   - 인용 형식 `[doc_id#section]` 포함 검증
   - API 실패 + fallback=True → extractive 결과 반환 + 경고 로그
   - API 실패 + fallback=False → CapabilityError 전파
3. `ai-worker/tests/test_rag_kr_fixture.py`
   - `build_rag_index.py --fixture kr` smoke
   - 한국어 쿼리에 대해 기대 doc_id가 top-5에 들어오는지

기존 테스트 회귀 금지:
- `test_multimodal_capability.py` 전부 pass 유지 (heuristic vision provider로 돌리는 기존 픽스처)
- `test_rag_capability.py`, `test_ocr_capability.py`, `test_mock_capability.py` 전부 pass

## 수용 기준
- `cd ai-worker && pytest -q` 모두 통과
- `.env.example`에 신규 변수 documented
- API key 없이 기본 `heuristic` / `extractive`로 기동 시 기존 로그/동작 불변
- `AIPIPELINE_WORKER_MULTIMODAL_VISION_PROVIDER=claude` + `AIPIPELINE_WORKER_RAG_GENERATOR=claude` +
  `AIPIPELINE_WORKER_ANTHROPIC_API_KEY=sk-...` 로 기동 시:
  ```
  RAG init: ... generator=claude-generation-v1
  MULTIMODAL init: vision_provider=claude-vision-v1 ...
  Active capabilities: ['MOCK', 'OCR', 'RAG', 'MULTIMODAL']
  ```
- README 벤치마크 테이블에 **실측 숫자** (not measured 각주 허용)
- `docs/architecture.md` 해당 섹션 업데이트
- 커밋 메시지는 3개로 분리: `feat(vision): claude-sonnet-4-6 provider`,
  `feat(rag): claude generation provider with extractive fallback`,
  `feat(eval): korean fixture + eval dataset + benchmark table`

## 비목표 (하지 말 것)
- GPT-4o / Gemini / 로컬 VLM(LLaVA 등) 추가 — 이번 phase는 Claude만
- Streaming 응답 — single-shot만
- Prompt caching (`cache_control`) — 이번 phase 비포함
- Function calling / tool use
- Heuristic vision / Extractive generator 제거
- 임베딩 모델 fine-tuning
- 멀티모달 retrieval index 신설 (작업 6번에서 할 예정)
- 기존 `anime_sample.jsonl` 변경
