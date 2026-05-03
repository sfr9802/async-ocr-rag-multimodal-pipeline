# namu_anime v4 migration — validation report

## Overview

- input v3 records: **4314**
- pages_v4 produced: **4314**
- chunks_v4 produced: **48675**
- promoted subpages (depth >= 1): **0**

## Page type distribution

- `character`: 414
- `episode`: 67
- `other`: 315
- `plot`: 56
- `review`: 98
- `setting`: 72
- `work`: 3292

## Relation distribution

- `character`: 414
- `episode`: 67
- `main`: 3292
- `other`: 315
- `plot`: 56
- `review`: 98
- `setting`: 72

## Chunk length stats (chars)

- min=60, p50=460, p90=2333, p95=3410, max=38753
- short (<60 chars): 0 (0.0%)
- empty: 0 (0.0%)

## Quality issues

- duplicate page_id: 0
- duplicate chunk_id: 0
- duplicate canonical_url: 0
- missing work_title: 0
- missing page_title: 0
- missing canonical_url (source): 0
- aliases missing: 0 (0.0%)
- generic page_title (still pollutes): 831
- empty section in pages: 0
- schema mismatches: pages=0, chunks=0
- episode-keyword subpages routed to `other`: 0

## Section type distribution

- `summary`: 13732
- `character`: 10326
- `music`: 5791
- `evaluation`: 4535
- `episode`: 3846
- `synopsis`: 2936
- `trivia`: 2936
- `other`: 1861
- `setting`: 1822
- `production`: 780
- `worldview`: 95
- `concept`: 15

## Title & alias quality (Phase 6.3)

- generic page_title: 831 (19.3%)
  - sample: '등장인물', '평가', '설정', '줄거리', '음악', '회차 목록', '에피소드 가이드', '주제가', '미디어 믹스', '기타 등장인물'
- alias_source: html=0, mixed=0, fallback=4314 (fallback ratio 100.0%), none=0
- title_source distribution: seed=3719, canonical_url=595
- retrieval_title collisions (extra dups): 0
- pages whose display_title differs from page_title: 818

## Top generic page_titles still present

- `등장인물`: 373
- `평가`: 94
- `설정`: 59
- `줄거리`: 50
- `음악`: 42
- `회차 목록`: 39
- `에피소드 가이드`: 34
- `주제가`: 28
- `미디어 믹스`: 18
- `기타 등장인물`: 17

## Warnings sample (first 30 of 1190)

- line 3293: work_title recovered from URL (https://namu.wiki/w/%EA%B0%80%EB%82%9C%EB%B1%85%EC%9D%B4%20%EC%8B%A0%EC%9D%B4%21/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '가난뱅이 신이!'
- line 3293: v3 title '등장인물' is generic; using work_title '가난뱅이 신이!' for root page_title
- line 3294: work_title recovered from URL (https://namu.wiki/w/%EA%B0%80%EC%A0%95%EA%B5%90%EC%82%AC%20%ED%9E%88%ED%8A%B8%EB%A7%A8%20REBORN%21/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '가정교사 히트맨 REBORN!'
- line 3294: v3 title '등장인물' is generic; using work_title '가정교사 히트맨 REBORN!' for root page_title
- line 3295: work_title recovered from URL (https://namu.wiki/w/%EA%B0%80%EC%A0%95%EA%B5%90%EC%82%AC%20%ED%9E%88%ED%8A%B8%EB%A7%A8%20REBORN%21/%EC%84%A4%EC%A0%95) -> '가정교사 히트맨 REBORN!'
- line 3295: v3 title '설정' is generic; using work_title '가정교사 히트맨 REBORN!' for root page_title
- line 3297: work_title recovered from URL (https://namu.wiki/w/%EA%B0%91%EC%B2%A0%EC%84%B1%EC%9D%98%20%EC%B9%B4%EB%B0%94%EB%84%A4%EB%A6%AC/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '갑철성의 카바네리'
- line 3297: v3 title '등장인물' is generic; using work_title '갑철성의 카바네리' for root page_title
- line 3299: work_title recovered from URL (https://namu.wiki/w/%EA%B0%91%EC%B2%A0%EC%84%B1%EC%9D%98%20%EC%B9%B4%EB%B0%94%EB%84%A4%EB%A6%AC/%EC%84%A4%EC%A0%95) -> '갑철성의 카바네리'
- line 3299: v3 title '설정' is generic; using work_title '갑철성의 카바네리' for root page_title
- line 3301: work_title recovered from URL (https://namu.wiki/w/%EA%B0%93%20%EC%98%A4%EB%B8%8C%20%ED%95%98%EC%9D%B4%EC%8A%A4%EC%BF%A8%28TVA%29/%ED%8F%89%EA%B0%80) -> '갓 오브 하이스쿨(TVA)'
- line 3301: v3 title '평가' is generic; using work_title '갓 오브 하이스쿨(TVA)' for root page_title
- line 3305: work_title recovered from URL (https://namu.wiki/w/%EA%B0%9C%EA%B7%B8%EB%A7%8C%ED%99%94%20%EB%B3%B4%EA%B8%B0%20%EC%A2%8B%EC%9D%80%20%EB%82%A0/%EC%97%90%ED%94%BC%EC%86%8C%EB%93%9C) -> '개그만화 보기 좋은 날'
- line 3305: v3 title '에피소드' is generic; using work_title '개그만화 보기 좋은 날' for root page_title
- line 3307: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20%EB%B9%8C%EB%93%9C%20%EB%8B%A4%EC%9D%B4%EB%B2%84%EC%A6%88/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '건담 빌드 다이버즈'
- line 3307: v3 title '등장인물' is generic; using work_title '건담 빌드 다이버즈' for root page_title
- line 3308: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20%EB%B9%8C%EB%93%9C%20%EB%8B%A4%EC%9D%B4%EB%B2%84%EC%A6%88/%EC%84%A4%EC%A0%95) -> '건담 빌드 다이버즈'
- line 3308: v3 title '설정' is generic; using work_title '건담 빌드 다이버즈' for root page_title
- line 3309: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20%EB%B9%8C%EB%93%9C%20%ED%8C%8C%EC%9D%B4%ED%84%B0%EC%A6%88/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '건담 빌드 파이터즈'
- line 3309: v3 title '등장인물' is generic; using work_title '건담 빌드 파이터즈' for root page_title
- line 3310: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20%EB%B9%8C%EB%93%9C%20%ED%8C%8C%EC%9D%B4%ED%84%B0%EC%A6%88%20%ED%8A%B8%EB%9D%BC%EC%9D%B4/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '건담 빌드 파이터즈 트라이'
- line 3310: v3 title '등장인물' is generic; using work_title '건담 빌드 파이터즈 트라이' for root page_title
- line 3311: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20%EB%B9%8C%EB%93%9C%20%ED%8C%8C%EC%9D%B4%ED%84%B0%EC%A6%88%20%ED%8A%B8%EB%9D%BC%EC%9D%B4/%ED%8F%89%EA%B0%80) -> '건담 빌드 파이터즈 트라이'
- line 3311: v3 title '평가' is generic; using work_title '건담 빌드 파이터즈 트라이' for root page_title
- line 3312: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20G%EC%9D%98%20%EB%A0%88%EC%BD%98%EA%B8%B0%EC%8A%A4%ED%83%80/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '건담 G의 레콘기스타'
- line 3312: v3 title '등장인물' is generic; using work_title '건담 G의 레콘기스타' for root page_title
- line 3313: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B4%EB%8B%B4%20G%EC%9D%98%20%EB%A0%88%EC%BD%98%EA%B8%B0%EC%8A%A4%ED%83%80/%EC%84%A4%EC%A0%95) -> '건담 G의 레콘기스타'
- line 3313: v3 title '설정' is generic; using work_title '건담 G의 레콘기스타' for root page_title
- line 3314: work_title recovered from URL (https://namu.wiki/w/%EA%B1%B8%EB%A6%AC%20%EC%97%90%EC%96%B4%ED%8F%AC%EC%8A%A4/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC) -> '걸리 에어포스'
- line 3314: v3 title '등장인물' is generic; using work_title '걸리 에어포스' for root page_title

## Suggested next steps

- Re-crawl pages whose `page_title` is still generic — they were rescued from URL/seed but the source title was unreliable.
- No subpages were promoted; verify v3 input actually has `subpages` populated.
- Index `chunks_v4.jsonl` with the `text_for_embedding` field, not raw `text`.
