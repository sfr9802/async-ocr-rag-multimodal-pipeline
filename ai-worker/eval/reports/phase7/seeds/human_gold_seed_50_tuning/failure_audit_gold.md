# Failure audit — gold-50 (baseline=baseline_retrieval_title_section_top10). Each row is one human-reviewed query.

> This evaluation is NOT a representative retrieval-quality benchmark. It is a human-weighted focus set drawn from queries_v4_llm_silver_500, designed to surface v4 subpage / section-level retrieval failures. primary_score improvements only mean 'we got better at the gold-50 subpage / named-subpage failures this set was curated to expose'.

| query_id | group | wt | bucket | qtype | reason | hit@1/5/10 | expected | top1 |
|---|---|---:|---|---|---|---|---|---|
| v4-llm-silver-324 | SOFT_POSITIVE | 0.25 | main_work | indirect_entity | OVER_BROAD_QUERY | 0/0/0 | 소드 아트 온라인(애니메이션 1기) [68666a36] §개요 | 극장판 소드 아트 온라인 -프로그레시브- 짙은 어둠의 스케르초 [57a658a7] |
| v4-llm-silver-412 | SOFT_POSITIVE | 0.70 | main_work | alias_variant | TITLE_MISS | 0/0/0 | 소드 아트 온라인(애니메이션 1기) [68666a36] §개요 | 사이키 쿠스오의 재난(애니메이션 1기) [9930c936] |
| v4-llm-silver-413 | SOFT_POSITIVE | 0.70 | main_work | alias_variant | TITLE_MISS | 0/0/0 | 소드 아트 온라인 Ⅱ [047e43cb] §개요 | 기동전사 건담 수성의 마녀 [ad2229ca] |
| v4-llm-silver-478 | AMBIGUOUS_PROBE | 0.00 | main_work | ambiguous | OVER_BROAD_QUERY | 0/0/0 | 강철의 연금술사 FULLMETAL ALCHEMIST [fca4f946] §개요 | 학원 핸섬 [7bac0c7f] |
| v4-llm-silver-480 | AMBIGUOUS_PROBE | 0.00 | main_work | ambiguous | OVER_BROAD_QUERY | 0/0/0 | 강철의 연금술사 FULLMETAL ALCHEMIST [fca4f946] §개요 | 마법과고교의 열등생 [0dda048d] |
| v4-llm-silver-485 | SOFT_POSITIVE | 0.30 | main_work | ambiguous | OVER_BROAD_QUERY | 0/0/0 | Go! 프린세스 프리큐어 [8157d9e3] §개요 | 프리큐어 시리즈 [bd7e8b49] |
| v4-llm-silver-109 | AMBIGUOUS_PROBE | 0.20 | subpage_generic | paraphrase_semantic | SUBPAGE_MISS | 0/0/0 | 원피스(애니메이션)/주제가 [06cd0747] §주제가 | 해골기사님은 지금 이세계 모험 중(애니메이션 1기) [d27cb963] |
| v4-llm-silver-221 | STRICT_POSITIVE | 1.00 | subpage_generic | section_intent | WRONG_SERIES | 0/0/0 | 달이 아름답다/에피소드 가이드 [bd17f9db] §에피소드 가이드 | 달이 아름답다 [60cabf11] |
| v4-llm-silver-375 | STRICT_POSITIVE | 1.00 | subpage_named | indirect_entity | WRONG_SERIES | 0/0/0 | 기동전사 건담 수성의 마녀/설정/기체 [6c5cd8d2] §설정 | 기체 | 기동전사 건담 수성의 마녀 [ad2229ca] |
| v4-llm-silver-510 | ABSTAIN_TEST | 0.00 | not_in_corpus | unanswerable_or_not_in_corpus | NOT_IN_CORPUS_CASE | 0/0/0 | — | 명탐정 코난 [6b42c9d8] |
| v4-llm-silver-518 | ABSTAIN_TEST | 0.00 | not_in_corpus | unanswerable_or_not_in_corpus | NOT_IN_CORPUS_CASE | 0/0/0 | — | 알바 뛰는 마왕님!! [14661507] |
| v4-llm-silver-520 | ABSTAIN_TEST | 0.00 | not_in_corpus | unanswerable_or_not_in_corpus | NOT_IN_CORPUS_CASE | 0/0/0 | — | 미래소년 코난 [f39d3a1e] |

