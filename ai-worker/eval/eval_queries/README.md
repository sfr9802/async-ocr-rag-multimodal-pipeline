# Eval query — retrieval 품질 데이터셋

이 디렉토리의 JSONL 파일은 eval CLI 의 `retrieval` 모드가 사용하는
**eval query 셋** 입니다. 이는 코퍼스
([`../corpora/`](../corpora/) 참조) 에 대해 점수 매겨짐 — 코퍼스가
haystack, query 가 needle + 정답 키.

> ⚠️ **엄격한 코퍼스 / query 분리**: 코퍼스 문서를 여기 두지 마세요.
> 코퍼스 행은 `eval/corpora/<name>/corpus.jsonl` 아래에 살고, query
> 행의 `expected_doc_ids` 는 어느 코퍼스의 id 를 인용함.

## 세 품질 등급

| 파일                            | 출처                  | 행 수 | 품질             | Ground truth 로 신뢰? |
|---------------------------------|-----------------------|-------|------------------|------------------------|
| `anime_smoke_6.jsonl`           | 수작업                | 6     | gold-style smoke | yes — harness 정상성 체크용으로만 사용 |
| `anime_silver_200.jsonl`        | 결정적 생성           | 200   | **silver**       | **NO** — 합성, best-effort 표현 |
| `anime_silver_200_llm.jsonl`    | LLM 백엔드 생성       | 200   | silver           | NO — 위와 같은 caveat + 비결정성 |
| `anime_gold_20.jsonl`           | 수작업                | 20    | gold             | yes — 코퍼스 콘텐츠에 대해 수동 큐레이션 |

### Silver vs gold 의미

* **silver** — 코퍼스 필드 (제목, 요약, 캐릭터 섹션 발췌) 에서 프로그래
  매틱하게 생성됨. config 변경에 걸쳐 retrieval 품질의 *추세* 를 추적
  하는 데 좋지만, query 자체는 어색하게 표현되거나 (결정적) 모델
  스타일이 들어갈 수 있고 (LLM), 기대 키워드는 큐레이션이 아니라
  휴리스틱하게 추출됨. **silver 숫자에 release 를 게이팅하지 마세요.**

* **gold** — 모든 행이 source doc 을 읽은 사람이 수작업으로 작성. Query
  가 진짜 질문을 표현; `expected_doc_ids` 가 답을 실제로 포함하는 doc
  을 가리킴; 키워드는 올바른 답변이 언급할 용어. "retriever 가 올바른
  것을 찾았는가" 의 바로 그 셋.

## 행 스키마

```jsonc
{
  "id":                        "anime-{tier}-{seq}",
  "query":                     "한국어 질문",
  "language":                  "ko",
  "expected_doc_ids":          ["<타깃 코퍼스의 doc_id>"],
  "expected_section_keywords": ["substring", ...],
  "answer_type":               "summary_plot" | "title_lookup" |
                               "character_relation" | "body_excerpt" |
                               "theme_genre" | "setting_worldbuilding",
  "difficulty":                "easy" | "medium" | "hard",
  "tags":                      ["anime", "<tier>", ...]
}
```

* `expected_doc_ids` — 한 질문이 한 개 이상의 정답 doc 을 가질 수
  있어 리스트; retrieval 메트릭은 top-k 의 어느 것이든 hit 으로 취급.
* `expected_section_keywords` — `expected_keyword_match_rate` (검색된
  어떤 chunk 텍스트에든 존재하는 비율) 가 사용. 더 오래된 `rag` harness
  에서 사용하는 (생성된 답변 문자열에 대해 점수 매기는) legacy
  `expected_keywords` 필드와 다름.
* `answer_type` — 리포트의 타입별 메트릭 분리를 주도.
* `tags` — `gold` / `silver` / `synthetic` / `manual` / `deterministic`
  / `llm:<backend>` 모두 관례적. harness 는 tag 를 파싱하지 않고; 사람
  필터링과 다운스트림 도구용.

## Silver 생성

`ai-worker/` 에서 실행:

```bash
# 결정적 — 재현 가능, LLM 비용 0, 1764문서 코퍼스에서 ~50ms
python -m eval.harness.generate_eval_queries \
    --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
    --out    eval/eval_queries/anime_silver_200.jsonl \
    --target 200 \
    --generator deterministic \
    --seed 42

# LLM 백엔드 — scripts.dataset.generate_anime_queries 를 감쌈
# (ANTHROPIC_API_KEY 또는 실행 중인 Ollama 데몬 필요)
python -m eval.harness.generate_eval_queries \
    --corpus eval/corpora/anime_namu_v3/corpus.jsonl \
    --out    eval/eval_queries/anime_silver_200_llm.jsonl \
    --target 200 \
    --generator llm \
    --llm-backend claude:haiku
```

결정적 백엔드 stratification 은 namu_anime_v3 코퍼스의 **실제 필드
커버리지에 보정됨**:

| answer_type             | source 필드             | 코퍼스 커버리지 | 타깃 % |
|-------------------------|-------------------------|-----------------|--------|
| `summary_plot`          | `summary`               | 100%            | 40%    |
| `character_relation`    | `sections.등장인물`     | 27.8%           | 20%    |
| `title_lookup`          | `title`                 | 100%            | 15%    |
| `body_excerpt`          | `sections.본문`         | 99.9%           | 10%    |
| `theme_genre`           | `summary_bullets`       | 100%            | 10%    |
| `setting_worldbuilding` | `sections.설정`/`세계관` | 1.6%            | 5%     |

커버리지 빈약한 타입은 source 신호를 실제로 들고 있는 doc 으로 cap —
generator 는 절대 섹션 콘텐츠를 조작하지 않음.

## Gold 셋 확장

`anime_gold_20.jsonl` 은 수동 큐레이션이 진짜 시간이 걸리기 때문에
의도적으로 작음 (20 행). gold-50 또는 gold-100 으로 확장하려면:

1. 개인적 컨텍스트가 있는 코퍼스의 doc 을 선택. 다음으로 검증:
   ```bash
   python - <<'PY'
   import json
   for line in open('eval/corpora/anime_namu_v3/corpus.jsonl', encoding='utf-8'):
       d = json.loads(line)
       if 'YOUR_TITLE_QUERY' in (d.get('title') or ''):
           print(d['doc_id'], d['title'])
           print((d.get('summary') or '')[:300])
   PY
   ```
2. 답이 doc 의 **`summary` 또는 `summary_bullets` 안에 있는** 질문을
   표현. 외부 지식이 필요한 trivia 는 피하기.
3. `expected_doc_ids` 를 검증된 `doc_id` 로 설정.
   `expected_section_keywords` 를 chunk 텍스트가 합리적으로 포함할 2-4
   개 substring 으로 설정 (이것들이 `expected_keyword_match_rate` 의
   분모가 됨).
4. silver 와 같은 6개 값에서 `answer_type` 선택. 등급 균형은 대략
   30% easy / 50% medium / 20% hard 유지.
5. `["anime", "gold", "manual", "<answer_type>"]` 로 태깅.

추가하는 모든 배치에 스모크 harness 실행하여 expected_doc_ids 의 오타
잡기:

```bash
python -m eval.run_eval retrieval \
    --corpus  eval/corpora/anime_namu_v3/corpus.jsonl \
    --dataset eval/eval_queries/anime_gold_20.jsonl \
    --top-k 10
```

`hit@10 == 0` 을 보고하는 행은 오타 (잘못된 doc_id) 이거나 진정으로
어려운 query — 그 행의 top-k dump 를 읽고 결정.

## 이 디렉토리가 다루지 **않는** 것

* **Generator 품질 eval.** 이 패키지는 retrieval 만 점수 매김.
  legacy `rag` 모드 (별도, 손대지 않음) 가 `expected_keywords` /
  거절 문구에 대해 생성된 답변을 점수 매김. End-to-end 답변 품질이
  필요하면 `eval/datasets/rag_*.jsonl` 에 대해 그 모드 사용.
* **Cross-modal query.** 이미지+텍스트 fusion 은 MULTIMODAL 아래.
* **OCR 주도 retrieval.** OCR harness 사용 후, 추출된 텍스트를
  retrieval harness 에 query 로 넣음.
