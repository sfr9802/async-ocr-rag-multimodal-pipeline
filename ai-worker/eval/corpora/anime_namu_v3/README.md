# anime_namu_v3 — namu-wiki anime 코퍼스 (1,764개 작품)

> **LEGACY V3 ONLY.** 이 코퍼스는 historical reproduction 과 v3 -> v4
> migration provenance 용도로만 유지합니다. Phase 7 이후 active
> eval/tuning 은 `eval/corpora/namu-v4-structured-combined/` 를 기준으로
> 시작하세요.

Retrieval **코퍼스** (eval query 셋이 아님). retrieval-eval harness 가
점수 매기는 인덱싱된 문서 저장소로 사용. 이 코퍼스를 타깃으로 하는 eval
query 는 [`ai-worker/eval/eval_queries/`](../../eval_queries/) 에 있음.

## 출처

| 필드            | 값                                                                    |
|-----------------|-----------------------------------------------------------------------|
| 출처 경로       | `D:/port/rag/app/scripts/namu_anime_v3.jsonl`                         |
| 크기            | ~261 MB (gitignore — 커밋하기엔 너무 큼)                              |
| 문서 수         | 1,764개 anime 작품                                                    |
| 기원            | namu-wiki 크롤 (port/rag 파이프라인)                                  |
| 언어            | ko                                                                    |
| 라이선스        | namu-wiki source 와 동일 (CC-BY-NC-SA 2.0 KR; 프로젝트 LICENSE 참조)  |

## Fresh clone 에서 re-stage

코퍼스 파일은 의도적으로 커밋되지 **않음**. 로컬에 stage 하려면:

```bash
cp 'D:/port/rag/app/scripts/namu_anime_v3.jsonl' \
   ai-worker/eval/corpora/anime_namu_v3/corpus.jsonl
```

위 경로에서 출처를 더 이상 사용할 수 없으면, namu-wiki dump 에서
`D:/port/rag/app/scripts/build_with_subpages.fixed.py` 로 재생성하거나,
[`ai-worker/fixtures/anime_corpus_kr.jsonl`](../../../fixtures/anime_corpus_kr.jsonl)
에 있는 더 작은 커밋된 샘플 (300개 작품, harness 스모크 테스트에는
충분하지만 retrieval baseline 으로는 부족) 을 사용하세요.

## 스키마 (행별)

```jsonc
{
  "doc_id":          "string — 안정적인 id (slug + hash)",
  "seed":            "string — 원본 namu-wiki seed page",
  "title":           "string — 표시 제목",
  "summary":         "string — 1-2 문장의 Claude 생성 요약",
  "summary_bullets": ["string", ...],   // 3-5 bullet 요약
  "sum_bullets":     ["string", ...],   // summary_bullets 의 중복 (legacy)
  "sections": {
    "요약": {"text": "...", "chunks": ["...", ...]},
    "본문": {"text": "...", "chunks": ["...", ...]},
    "등장인물": {"text": "...", "chunks": ["...", ...]}   // 문서의 ~28%
    // 선택적: "설정", "줄거리", "평가", ...
  },
  "section_order":   ["요약", "본문", ...],
  "meta":            {"seed_title": "...", "depth": 0, "fetched_at": "ISO-8601"},
  "subpages":        []   // v3 에서는 비어있음
}
```

### 1,764 문서 전반의 필드 가용성

| 필드                  | 커버리지   | 비고                                            |
|-----------------------|------------|-------------------------------------------------|
| `summary`             | 100%       | 항상 채워짐, ~1-2 문장                          |
| `summary_bullets`     | 100%       | 문서당 3-5 bullet                               |
| `sections.요약`       | 100%       | 항상 존재                                       |
| `sections.본문`       | 99.9%      | 1763/1764                                       |
| `sections.등장인물`   | **27.8%**  | 491개 문서 — 우리가 가진 유일한 character 신호 |
| `sections.설정`       | 1.6%       | 28개 문서                                       |
| `sections.줄거리`     | 1.0%       | 18개 문서                                       |
| `aliases`             | 0%         | 필드 존재, 항상 None                            |
| `keywords`            | 0%         | 필드 존재, 항상 None                            |

이 분포는
[`ai-worker/eval/harness/generate_eval_queries.py`](../../harness/generate_eval_queries.py)
의 합성 query stratification 을 주도 — 72% 문서가 character section 이
없는 코퍼스에서 character query 20% 를 정직하게 타깃할 수는 없으므로,
결정적 generator 는 코퍼스가 실제로 지원하는 것에 따라 query 타입을
가중함.

## 인덱싱

retrieval-eval CLI 가 기존 offline-corpus 경로로 이 파일에서 즉석으로
in-memory FAISS 인덱스를 빌드:

```bash
cd ai-worker
python -m eval.run_eval retrieval \
  --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
  --dataset eval/eval_queries/anime_smoke_6.jsonl \
  --top-k 5
```

Postgres `ragmeta` 스키마도 디스크 FAISS 인덱스도 필요 없음 — 코퍼스가
실행마다 tempdir 에 재 chunk + 재임베딩 됨. 이는 기존 `rag` 서브커맨드가
이미 지원하는 `--offline-corpus` 플래그와 일치.
