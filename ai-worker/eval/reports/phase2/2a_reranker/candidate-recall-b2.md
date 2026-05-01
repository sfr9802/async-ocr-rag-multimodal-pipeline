# Retrieval eval report

- dataset: `eval\eval_queries\anime_silver_200.jsonl`
- corpus:  `eval\corpora\anime_namu_v3_token_chunked\corpus.combined.token-aware-v1.jsonl`
- rows:    200 (errors: 0)
- top_k:   50 (mrr@10, ndcg@10)
- model:   BAAI/bge-m3
- index:   offline-1777368182
- reranker: noop
- started: 2026-04-28T19:10:27
- duration_ms: 4254.7

## Headline metrics

| metric | value |
|---|---|
| hit@1 | 0.5400 |
| hit@3 | 0.6650 |
| hit@5 | 0.6800 |
| hit@10 | 0.7150 |
| hit@20 | 0.7700 |
| hit@50 | 0.8000 |
| mrr@10 | 0.6044 |
| ndcg@10 | 0.6314 |
| dup_rate (top-50) | 0.2956 |
| unique_doc_coverage | 0.7044 |
| top1_score_margin | 0.0290 |
| avg_context_token_count | 315.0784 |
| expected_keyword_match_rate | 0.9233 |

## Latency (ms)

- mean: 15.21
- p50:  13.51
- p95:  20.52
- max:  31.61

## Per answer_type

| answer_type | n | hit@5 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|
| body_excerpt | 20 | 0.7000 | 0.5246 | 0.5910 |
| character_relation | 40 | 0.2750 | 0.2650 | 0.2730 |
| setting_worldbuilding | 10 | 1.0000 | 1.0000 | 1.0000 |
| summary_plot | 80 | 0.8250 | 0.7322 | 0.7641 |
| theme_genre | 20 | 0.8500 | 0.7292 | 0.7596 |
| title_lookup | 30 | 0.6000 | 0.5542 | 0.5736 |

## Per difficulty

| difficulty | n | hit@5 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|
| easy | 30 | 0.6000 | 0.5542 | 0.5736 |
| hard | 30 | 0.8000 | 0.6831 | 0.7273 |
| medium | 140 | 0.6714 | 0.5983 | 0.6232 |

## Duplicate analysis

- queries with doc dup:     200 (1.000)
- queries with section dup: 199 (0.995)
- queries with text dup:    29 (0.145)

### Most-duplicated doc_ids (extra copies summed across runs)

- `70ab496518e106ac8dd5c9c5` — 68 extra copies
- `46ba8240585f26a8eebb5fbf` — 58 extra copies
- `7bc38ee682c29e93146b55ad` — 58 extra copies
- `7485834b0a97e63196d41f67` — 51 extra copies
- `03efd55b414a1d48b8406751` — 46 extra copies
- `af65062fb752f0f83e0d2175` — 43 extra copies
- `ef97086a15c60e5ce65b3f0e` — 42 extra copies
- `e7a0c720cb266e8ca51614bd` — 41 extra copies
- `17af1055498fd8036612470f` — 40 extra copies
- `5b703b719249a314fd3d7036` — 36 extra copies

## First misses (hit@5 == 0)

- `anime-silver-0006` (character_relation/medium): 등장인물에서 그들과(와) 한국에서의 관계를 설명해 주세요. → expected ['6d300d9338438eb2f9122b3f'] got [ae0d47addc09f88cf373dd76@0.544, 5d61cda44fc6247adb1e8e62@0.541, 2c27f6809a37a7f4b746c007@0.538]
- `anime-silver-0007` (summary_plot/medium): BanG와(과) Dream이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['83a98a20cff1d499ad507448'] got [553309e56edc1a4f234a31de@0.623, c89c8e74f23c546e44843c06@0.582, c89c8e74f23c546e44843c06@0.574]
- `anime-silver-0008` (summary_plot/medium): 평범한와(과) 고등학생이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['1e506f227ed3cf16058d2010'] got [cbb7b831a9781d48fe77d16a@0.607, 9dcff1ce8460f001db8636ff@0.596, df3f86368b376676c01c3051@0.593]
- `anime-silver-0010` (character_relation/medium): 등장인물에서 배경과(와) 작가의 관계를 설명해 주세요. → expected ['19e9bf4715cce8fce4984f32'] got [58dee7ffe43190362a2b0922@0.588, ec26b239199920dc9e502237@0.574, 9de93f2f8fe9845f85b9edd8@0.565]
- `anime-silver-0012` (body_excerpt/hard): DIVE!! 본문에서 요이치와는와(과) 친분이이(가) 어떻게 다루어지나요? → expected ['7b6284e7ebfcf0d5955651c3'] got [46ba8240585f26a8eebb5fbf@0.589, 46ba8240585f26a8eebb5fbf@0.587, 46ba8240585f26a8eebb5fbf@0.584]
- `anime-silver-0021` (summary_plot/medium): 일본의와(과) 로컬이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['d769687d49b681e436d8534a'] got [7f3a0088839bf373035486b5@0.547, 730b95893eafe7eaa39993df@0.545, 8a1b0735daa7e9316a0292a9@0.544]
- `anime-silver-0022` (character_relation/medium): 등장인물에서 펼쳐지과(와) ADK의 관계를 설명해 주세요. → expected ['4c4855df513ea8ab567248b4'] got [250aa39bb621ae8c94a410fa@0.530, 250aa39bb621ae8c94a410fa@0.530, 9611b772ccc2084b2bb221d3@0.528]
- `anime-silver-0024` (character_relation/medium): 등장인물에서 스토리과(와) 카오루코의 관계를 설명해 주세요. → expected ['bfbaa3fc6a7069ddb7259e20'] got [0849e15dc3d03631807eebd5@0.569, 928487eec1d5355b8184f8a1@0.567, 17af1055498fd8036612470f@0.561]
- `anime-silver-0025` (character_relation/medium): 등장인물에서 감독과(와) 여성향의 관계를 설명해 주세요. → expected ['8c42446ee0478f11c2d45fa0'] got [9de93f2f8fe9845f85b9edd8@0.568, d62ff763466a5b832b4f1b7e@0.559, c2d5757207e3383905cd030f@0.555]
- `anime-silver-0027` (character_relation/medium): 등장인물에서 야키시오과(와) 토카의 관계를 설명해 주세요. → expected ['8e220af327e191cd5c437134'] got [c6f77412c5c55fe3cd302034@0.629, ae0d47addc09f88cf373dd76@0.627, de89603ba70548d3b816b64c@0.625]
- `anime-silver-0028` (character_relation/medium): 등장인물에서 자신과(와) 호시미야의 관계를 설명해 주세요. → expected ['488378060796af2d6781f242'] got [047c06a9e2847f5fc36cf7d1@0.668, 047c06a9e2847f5fc36cf7d1@0.647, 8e220af327e191cd5c437134@0.629]
- `anime-silver-0032` (character_relation/medium): 등장인물에서 먼나라과(와) 박시후가의 관계를 설명해 주세요. → expected ['be3c8485d51c9ae98deae76b'] got [df121bead8d087f21881935b@0.563, 8de472dcd2c5a83c277d18f3@0.531, 2f9207da8ab0877cbf5ca2a5@0.526]
- `anime-silver-0034` (character_relation/medium): 등장인물에서 실제로과(와) 작가의 관계를 설명해 주세요. → expected ['69474fc45342e281dbaaec39'] got [9de93f2f8fe9845f85b9edd8@0.565, 70ab496518e106ac8dd5c9c5@0.564, 9d616de94b32503471e9ac92@0.557]
- `anime-silver-0044` (summary_plot/medium): 전생와(과) 귀족이이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['830ddc2ee4b1810df53f5c8c'] got [23d9bdd82ae4f6b1845b1491@0.587, 523c78cde97215c39474b38c@0.583, 3875e74363999987bab637ed@0.574]
- `anime-silver-0047` (title_lookup/easy): 등장인물이(가) 어떤 작품인가요? → expected ['f5528fab349568f423d566f0'] got [58dee7ffe43190362a2b0922@0.621, ec26b239199920dc9e502237@0.575, 7b6db618b3b273e9a90440bb@0.562]
- `anime-silver-0048` (summary_plot/medium): 아이돌와(과) 마스터이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['56d0537fd932b39d448cbc0a'] got [ef97086a15c60e5ce65b3f0e@0.631, 7b6db618b3b273e9a90440bb@0.625, ef97086a15c60e5ce65b3f0e@0.620]
- `anime-silver-0056` (character_relation/medium): 등장인물에서 서적화과(와) 작가의 관계를 설명해 주세요. → expected ['fd9e9dc06234e5f17d9dad2d'] got [ec26b239199920dc9e502237@0.583, 58dee7ffe43190362a2b0922@0.548, 9de93f2f8fe9845f85b9edd8@0.547]
- `anime-silver-0060` (title_lookup/easy): 영검산 예지의 자격이(가) 어떤 작품인가요? → expected ['c81fcda937385e99ce8e459b'] got [6b2104ae97550029886cded4@0.450, ab7d104cfe2cbb319ee38bc3@0.450, 8e02f4091225d97eef954e31@0.449]
- `anime-silver-0062` (summary_plot/medium): 드림페스와(과) 에서이(가) 등장하는 작품의 줄거리를 알려주세요. → expected ['14d03563b66d25fec5521a9e'] got [a87ce8532f4393119b575fdb@0.544, bc2dea5c3c421a7e1cafcc7d@0.537, 566cddd99047d697c9d7cd56@0.534]
- `anime-silver-0067` (character_relation/medium): 등장인물에서 애니메과(와) 박시후가의 관계를 설명해 주세요. → expected ['449b7b5db32202d1e6fe8373'] got [f55ca4856bc27aa1e678fc52@0.629, 1b01baa047c6eb1c1ee025c0@0.618, e7a0c720cb266e8ca51614bd@0.603]

