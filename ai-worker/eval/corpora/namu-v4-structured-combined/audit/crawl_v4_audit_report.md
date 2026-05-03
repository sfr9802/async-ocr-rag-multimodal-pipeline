# Crawl V4 Audit Report

- schema_version: `namu_anime_v4_crawl_audit`
- warnings: **5**

## Pages
- total_pages: **4314**
- missing work_title: 0
- missing aliases: 0
- missing source.fetched_at: 0
- duplicate page_titles (23): ['OST', '기타', '기타 등장인물', '등장인물', '미디어 믹스']
- section_count: count=4314 mean=11.28 median=10 min=1 max=183
- unsupported_section_type_ratio: 3.82% (unknown=0, other=1861)
- sections_with_blank_title: 0, sections_with_generic_title: 9465
- section_depth distribution: 2=20927, 3=20141, 4=7607

### page_type distribution
- work: 3292
- character: 414
- other: 315
- review: 98
- setting: 72
- episode: 67
- plot: 56

### section_type distribution
- summary: 13732
- character: 10326
- music: 5791
- evaluation: 4535
- episode: 3846
- synopsis: 2936
- trivia: 2936
- other: 1861
- setting: 1822
- production: 780
- worldview: 95
- concept: 15

### top section titles
- 개요: 3097
- 기타: 1822
- 줄거리: 1646
- 회차 목록: 1527
- 평가: 1450
- 등장인물: 891
- 공개 정보: 850
- 해외 공개 > 대한민국: 724
- 주제가 > OP: 651
- 주제가 > ED: 595

### title & alias quality (Phase 6.3)
- generic page_title: 831 (19.26%)
  - sample: ['등장인물', '평가', '설정', '줄거리', '음악', '회차 목록', '에피소드 가이드', '주제가', '미디어 믹스', '기타 등장인물']
- alias_source: html=0, mixed=0, fallback=4314 (100.00%), none=0
- title_source distribution: seed=3719, canonical_url=595
- retrieval_title collisions: 0

## Chunks
- total_chunks: **135602**
- text_length: count=135602 mean=334.8 median=203 p25=107 p75=473 min=60 max=2993
- too_short (0.00%): 0, too_long (0.00%): 0
- is_stub: 0 (0.00%), is_table_like: 0 (0.00%), is_list_like: 31 (0.02%)
- chunks_with_blank_section_title: 0

### top work skew
- 애니메이션: 2625 chunks (1.94%)
- 에피소드 가이드: 2311 chunks (1.70%)
- 회차 목록: 1571 chunks (1.16%)
- 주제가: 1163 chunks (0.86%)
- 음악: 1053 chunks (0.78%)
- 기타 등장인물: 735 chunks (0.54%)
- 일일외출록 반장: 732 chunks (0.54%)
- 던전에서 만남을 추구하면 안 되는 걸까: 675 chunks (0.50%)
- 릭 앤 모티: 654 chunks (0.48%)
- 닌자 슬레이어: 478 chunks (0.35%)

### chunks by section_type
- character: 31073
- summary: 28861
- trivia: 19601
- evaluation: 13340
- episode: 12861
- music: 8731
- synopsis: 8082
- other: 5410
- setting: 4888
- production: 2450
- worldview: 242
- concept: 63

### chunks by section_title (top)
- 기타: 14077
- 회차 목록: 5624
- 평가: 4346
- 개요: 4039
- 등장인물: 3304
- 줄거리: 2783
- 여담: 1539
- 특징: 1523
- 공개 정보: 1395
- 해외 공개 > 대한민국: 1302

## Manifest
- doc_counts: train=3513 valid=411 test=390
- chunk_counts: train=111588 valid=12706 test=11308
- valid missing section_types: ['concept']
- test missing section_types: ['concept']

## Warnings
- pages: 23 duplicated page_title(s) detected — sample=['OST', '기타', '기타 등장인물']
- pages: generic page_title ratio 19.26% > 10% (831/4314) — consider using retrieval_title for embedding text builders. sample=['등장인물', '평가', '설정', '줄거리', '음악']
- pages: alias_source='fallback' on 4314/4314 pages (100.00% > 95%) — aliases are backfilled from seed/title rather than HTML; a future recrawl would populate richer variants.
- manifest: split 'valid' is missing section_types ['concept'] — coverage gap vs. other splits
- manifest: split 'test' is missing section_types ['concept'] — coverage gap vs. other splits
