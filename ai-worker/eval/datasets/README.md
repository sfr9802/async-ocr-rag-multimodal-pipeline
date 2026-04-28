# Eval 데이터셋 — 카탈로그

이 디렉토리의 모든 JSONL 파일은 `ai-worker/eval/run_eval.py` 의 CLI
harness 가 소비하는 eval 데이터셋입니다. 두 축으로 깔끔하게 분리:
**도메인** (retrieval 품질 회귀를 위해 유지된 `anime` 픽스처 vs. Phase
9 에서 production-shape 타깃으로 추가된 `enterprise` 픽스처) 과
**harness** (`rag` / `ocr` / `multimodal` / `routing` / `cross-domain`).

## 카탈로그

| 파일 | 도메인 | 언어 | 행 수 | 목적 | 라이선스 |
|------|--------|------|------:|------|----------|
| `rag_sample.jsonl` | anime | en | 6 | Phase 0 EN anime RAG baseline — 원래 8문서 anime 픽스처에서 FAISS+bge-m3 retrieval 회귀 테스트 | Apache-2.0 |
| `rag_sample_kr.jsonl` | enterprise (KR placeholder) | ko | 10 | legacy `kr_sample.jsonl` 10문서 픽스처에 대한 Phase 0 KR baseline. Phase 9 baseline 이 Phase 0 숫자에 대한 무회귀를 증명할 수 있도록 유지 | Apache-2.0 |
| `rag_anime_extended_kr.jsonl` | anime | ko | 30 | `anime_kr.jsonl` 에 대한 Phase 9 확장 한국어 anime eval (10 문서 × 3 query, 난이도 easy:10 / medium:15 / hard:5) | Apache-2.0 |
| `rag_enterprise_kr.jsonl` | enterprise | ko | 20 | Phase 9 primary enterprise RAG eval — 5개 카테고리 전반에 시드된 대표 셋. `scripts.dataset.generate_queries` 로 ~200 까지 확장 가능 | Apache-2.0 |
| `rag_cross_domain_kr.jsonl` | cross-domain | ko | 20 | Phase 9 unanswerable 셋 — 모든 행이 검색을 잘못된 도메인으로 제한하는 필터를 pin. harness 게이트는 `cross_domain_refusal_rate >= 0.85`; relevance-gated `ExtractiveGenerator` 가 top score < 0.48 일 때 "문서에서 관련 정보를 찾을 수 없습니다" emit | Apache-2.0 |
| `rag_agent_fixture_kr.jsonl` | enterprise (KR placeholder) | ko | ~50 | `kr_sample.jsonl` 에 대한 Phase 8 agent-loop 결정 게이트 픽스처 — 난이도 stratified (easy / hard / impossible) query | Apache-2.0 |
| `ocr_sample.jsonl` | (general) | en | 3 | Phase 2 EN OCR 픽스처 — 합성 invoice/pangram; `scripts.make_ocr_sample_fixtures` 로 재생성 | Apache-2.0 |
| `ocr_sample_kr.jsonl` | (general) | ko | 2 | Phase 2 KR OCR 픽스처 — 합성 공지; `scripts.make_ocr_sample_fixtures` 로 재생성 | Apache-2.0 |
| `ocr_enterprise_kr.jsonl` | enterprise | ko | 5 | Phase 9 enterprise OCR eval — 각 코퍼스 doc 을 PNG 페이지로 렌더; `scripts.dataset.synthesize_ocr_pages --count 50` 으로 50 까지 확장 | Apache-2.0 |
| `multimodal_sample.jsonl` | mixed | en+ko | 9 | OCR-only / visual-only / OCR+visual 행 + Phase 9 anime 포스터에 걸친 multimodal eval | Apache-2.0 |
| `multimodal_anime_kr.jsonl` | anime | ko | 6 | Phase 9 anime-poster multimodal 데모 셋 (6개 프로그래매틱 포스터); 이미지는 `fixtures/posters/` 아래 (CC0) | Apache-2.0 |
| `multimodal_enterprise_kr.jsonl` | enterprise | ko | 5 | Phase 9 enterprise multimodal eval — `ocr_enterprise/` 렌더링된 페이지 재사용; `scripts.dataset.generate_multimodal --count 50` 으로 50 까지 확장 | Apache-2.0 |
| `routing_enterprise_kr.jsonl` | enterprise | ko | 15 | Phase 9 AUTO-routing seed eval — 5개 action (rag / ocr / multimodal / direct_answer / clarify); `scripts.dataset.generate_routing_cases --total 80` 으로 80 까지 확장 | Apache-2.0 |

## 데이터셋들이 어떻게 맞물리는가

1. **Phase 0 회귀 락.** `rag_sample.jsonl` 와 `rag_sample_kr.jsonl` 은
   Phase 0 픽스처로 변경 없이 유지 — 통합 Phase 9 인덱스
   (`--fixture all`) 에서 그들의 recall@5 숫자가 Phase 0 baseline 의
   ±0.03 안에 있어야 함, 그렇지 않으면 enterprise 코퍼스가 anime query
   로 새고 있다는 뜻.

2. **Enterprise 코퍼스 재생성.** `fixtures/corpus_kr/` 아래 5개의
   수작업 시드 doc 이 커밋된 시작점; `python -m
   scripts.dataset.build_corpus --out ai-worker/fixtures/corpus_kr
   --per-category 25` 로 125 문서까지 확장. `python -m
   scripts.dataset.rebuild_corpus_index --out
   ai-worker/fixtures/corpus_kr` 로 인덱스 행 재빌드.

3. **Eval-셋 확장.** 각 enterprise eval 파일은 수작업 시드 + 풀 N 으로
   가는 스크립트 경로를 가짐. API 토큰 쓰기 전에 사이즈/비용을 계획
   하려면 generator 스크립트에서 먼저 `--dry-run` 사용.

4. **Cross-domain 게이트.** `rag_cross_domain_kr.jsonl` 은
   `ParsedQuery.filters` 가 실제로 검색 공간을 좁히는지의 결정적 테스트.
   relevance-gated `ExtractiveGenerator` 가 top-chunk-score-0.48-미만
   retrieval 을 거절로 변환; LLM 백엔드 generator 는 의미적 추론으로
   같은 거절에 도달.

## Phase 9 baseline 리포트 재빌드

`ai-worker/` 에서:

```
python -m scripts.build_rag_index --fixture all
python -m eval.run_eval rag --dataset eval/datasets/rag_sample.jsonl         --out-json eval/reports/phase9-baseline/rag_sample.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_sample_kr.jsonl      --out-json eval/reports/phase9-baseline/rag_sample_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_anime_extended_kr.jsonl --out-json eval/reports/phase9-baseline/rag_anime_extended_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_enterprise_kr.jsonl  --out-json eval/reports/phase9-baseline/rag_enterprise_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_cross_domain_kr.jsonl --cross-domain \
    --out-json eval/reports/phase9-baseline/rag_cross_domain_kr.json --no-csv
```

`eval/reports/phase9-baseline/` 아래 리포트는 Phase 10 (Optuna) 이
명확한 비교점을 가지도록 커밋. CUDA 사용 가능 시 bge-m3 모델은 GPU
가속; CPU 빌드는 ~3-4배 더 오래 걸리지만 byte 단위 동일한 출력 생성.
