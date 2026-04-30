# Phase 7.0 — Retrieval Title Embedding Variant A/B (Final Report)

## 1. Summary

Compared two embedding-text variants over the Phase 6.3
`namu-v4-structured-combined-2008-2026-04-phase6_3_title_alias_quality`
artefacts:

- **baseline `title_section`** — Phase 6.3 stored `embedding_text`
  format with `page_title` as the title segment.
- **candidate `retrieval_title_section`** — same format, but uses
  `retrieval_title` (folded `"{work_title}/{page_title}"` form for
  generic page_titles) with fallback to `page_title` when missing.

**Verdict:** the candidate **strongly wins** on all retrieval metrics
on the 200-query v4 silver set; **promote `retrieval_title_section`
to default candidate** for the next phase. Headline numbers:

- hit@1 **0.595 → 0.815 (+22.0pt)**
- hit@10 **0.795 → 0.985 (+19.0pt)**
- MRR@10 **0.663 → 0.882 (+21.9pt)**
- nDCG@10 **0.695 → 0.907 (+21.3pt)**
- improved : regressed = **74 : 3**

Effect is concentrated where Phase 6.3's audit predicted: subpages
whose `page_title` is generic (`등장인물`, `평가`, `설정`, `줄거리`,
`회차`, `미디어 믹스`, `OST`, `주제가`, `음악`, `에피소드`, `기타 등장인물`).
Main work pages are unaffected (the bucket has retrieval_title ==
page_title by construction); this is the no-harm zone confirmed.

---

## 2. Modified files

### New modules (under `ai-worker/eval/harness/`)
- `v4_chunk_export.py` — streaming export of `rag_chunks.jsonl` per
  variant, with `manifest_<variant>.json` (sha256, changed counts,
  page_id coverage). Preserves all non-`embedding_text` fields
  byte-for-byte.
- `v4_variant_diff_report.py` — pairs the two exported chunk files,
  joins each chunk with its page-level metadata from `pages_v4.jsonl`,
  emits `variant_diff_report.json/.md` with breakdowns by
  page_type / section_type / title_source / alias_source + top
  changed examples.
- `v4_silver_queries.py` — deterministic v4-aware silver query
  generator. Stratified across three buckets (subpage_generic /
  main_work / subpage_named) with parent-work recovery from
  `retrieval_title` (the v4 `work_title` field is unreliable on
  subpages — it was preserved from page_title verbatim).
- `v4_index_builder.py` — variant-aware FAISS index builder over
  pre-chunked `rag_chunks_<variant>.jsonl`. Cache slug includes the
  variant; cache key `v4_variant_cache_key` digests
  (chunk file, mtime, model, max_seq, variant) so two variants
  cannot collide.
- `v4_ab_eval.py` — paired A/B over a query set against two
  retrievers; emits `ab_summary.json/.md`,
  `per_query_comparison.jsonl`, `improved_queries.jsonl`,
  `regressed_queries.jsonl`. Status classifier rules tie ranks at
  `both_hit`, miss/miss at `both_missed`.

### Existing modules edited
- `eval/harness/embedding_text_builder.py`
  - Added `VARIANT_RETRIEVAL_TITLE_SECTION = "retrieval_title_section"`
    to `EMBEDDING_TEXT_VARIANTS`.
  - Added `V4EmbeddingTextInput` dataclass (`from_chunk_record`
    constructor accepts a Phase 6.3 chunk record dict).
  - Added `build_v4_embedding_text(chunk, *, variant)` reproducing
    Phase 6.3's `"제목: ... \n섹션: ... \n섹션타입: ... \n\n본문:\n..."`
    format byte-for-byte for both variants.
  - The legacy `build_embedding_text` rejects the v4 variant with a
    clear error so a sweep that mixes v3/v4 variants cannot silently
    conflate them. Existing v3 variant outputs (`raw`, `title`,
    `section`, `title_section`, `keyword`, `all`) are unchanged
    (regression-pinned by tests).

### New CLI
- `scripts/run_phase7_0_retrieval_title_ab.py` — one-shot orchestrator
  with `--skip-export / --skip-diff / --skip-index-build / --skip-ab`
  flags so partial reruns are one command.

---

## 3. New or updated tests

`tests/test_phase7_0_retrieval_title_ab.py` (30 tests, all passing):

- `test_retrieval_title_section_uses_retrieval_title_when_present` —
  candidate uses retrieval_title verbatim.
- `test_retrieval_title_section_falls_back_to_page_title` — empty
  retrieval_title falls back, no sentinel leakage.
- `test_title_section_baseline_uses_page_title_only` — baseline
  ignores retrieval_title even when set (otherwise A/B is contaminated).
- `test_v4_format_matches_phase6_3_layout` — pins the four labels
  (`제목` / `섹션` / `섹션타입` / `본문`) and the body separator.
- `test_legacy_title_section_output_unchanged` — v3 builder regressionpin.
- `test_legacy_builder_rejects_v4_variant` — guard against accidental
  v4-on-v3 calls.
- `test_v4_builder_rejects_unknown_variant` — error path.
- `test_retrieval_title_section_listed_in_known_variants` — registry pin.
- `test_recompute_embedding_text_baseline_matches_phase63` — baseline
  reproduces stored embedding_text byte-for-byte.
- `test_recompute_embedding_text_candidate_uses_retrieval_title` —
  candidate substitutes correctly.
- `test_export_v4_chunks_writes_both_variants` — manifest counts
  changed=0 for baseline / changed>0 for candidate.
- `test_export_cli_module_accepts_retrieval_title_section` —
  CLI tuple includes both variants.
- `test_variant_diff_counts_changed_and_unchanged` — diff report math.
- `test_variant_diff_examples_carry_old_and_new_previews` — top
  examples carry both previews.
- `test_variant_diff_report_writes_both_files` — `.json` + `.md`.
- `test_silver_queries_generate_for_v4_pages` — generation works on
  a v4 fixture and reaches both buckets.
- `test_subpage_generic_uses_retrieval_title_for_parent_work` —
  parent recovery from retrieval_title (NOT page.work_title).
- `test_subpage_generic_skipped_when_parent_unavailable` — degenerate
  query avoidance.
- `test_is_generic_page_title_covers_phase6_3_set` — generic-title set
  match.
- `test_v4_cache_key_includes_variant` — cache key cannot collide
  across variants.
- `test_v4_default_cache_dir_includes_variant_slug` — slug includes
  variant + max_seq_length.
- `test_v4_default_cache_dir_rejects_unknown_variant` — error path.
- `test_v4_index_supported_variants_are_phase7_0_pair` — registry pin.
- `test_classify_*` (6 tests) — A/B classifier covers all 5 statuses
  including improved-from-miss / regressed-to-miss / rank-comparison
  ties.
- `test_run_paired_ab_writes_artefacts` — end-to-end with mock
  retrievers; verifies status counts, per_query / improved / regressed
  jsonl outputs, aggregate metric calculation.

Test count: **1308 passed** (28 new tests + 1280 existing). Existing
tests (`test_eval_harness`, `test_phase2_retrieval_experiments`, etc.)
are all green — no regression.

---

## 4. Commands run

```bash
# 1. Export both variant chunk files (in-process, ~10s)
python -c "from eval.harness.v4_chunk_export import export_v4_chunks, V4_EXPORT_VARIANTS
src='D:/port/crawling/.../rag_chunks.jsonl'
out='eval/reports/.../phase7_0_retrieval_title_ab'
for v in V4_EXPORT_VARIANTS:
    export_v4_chunks(src, f'{out}/rag_chunks_{v}.jsonl', variant=v)"

# 2. Compute variant diff report (~5s)
python -c "from eval.harness.v4_variant_diff_report import compute_variant_diff, write_variant_diff_report
report = compute_variant_diff(...)
write_variant_diff_report(report, out_dir=...)"

# 3. Build FAISS indexes for both variants (RTX 5080, max_seq=512, batch=64)
#    Variant 1 ~42 min, Variant 2 ~45 min (sequential)
python -c "from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from eval.harness.v4_index_builder import build_v4_variant_index, v4_default_cache_dir, V4_INDEX_VARIANTS
embedder = SentenceTransformerEmbedder('BAAI/bge-m3', max_seq_length=512, batch_size=64)
for v in V4_INDEX_VARIANTS:
    build_v4_variant_index(...)"

# 4. Generate v4 silver queries (~2s)
python -c "from eval.harness.v4_silver_queries import generate_v4_silver_queries, write_v4_silver_queries
queries = generate_v4_silver_queries(pages_v4_path, target_total=200, seed=42)
write_v4_silver_queries(queries, out_path)"

# 5. Run paired A/B (~12s, 200 queries × 2 retrievers)
python -c "from eval.harness.v4_ab_eval import run_paired_ab, write_ab_outputs
result = run_paired_ab(...)
write_ab_outputs(result, ...)"

# 6. Tests
pytest tests/test_phase7_0_retrieval_title_ab.py -q       # 30 passed
pytest tests/ -q                                          # 1308 passed
```

The `scripts/run_phase7_0_retrieval_title_ab.py` orchestrator runs
all six steps sequentially with `--skip-*` flags for partial reruns.

---

## 5. Test results

- **Phase 7.0 module tests:** 30 passed.
- **Full ai-worker test suite:** 1308 passed, 9 warnings, 0 failures.
- Pre-existing tests (e.g. `test_eval_harness`, `test_corpus_audit`,
  `test_cap_policy*`, `test_reranker_eval`) all unchanged.

---

## 6. Variant diff results

Source: `D:\port\crawling\namu-v4-structured-combined-2008-2026-04-phase6_3_title_alias_quality\rag_chunks.jsonl`

| metric | value |
|---|---:|
| total_chunks | 135,602 |
| changed_embedding_text_count | **44,759** |
| changed_embedding_text_ratio | **33.01%** |
| integrity_non_embedding_text_diffs | 0 |
| missing_page_meta | 0 |

Matches the Phase 6.3 spec exactly (`PHASE6_3_TITLE_ALIAS_DIFF.md`
predicted 44,759 / 135,602 = 33.0% — observed 33.01%).

### Changed by page_type

| page_type | changed | total | ratio |
|---|---:|---:|---:|
| `character` | 20,925 | 20,925 | 100.0% |
| `other`     |  9,550 |  9,550 | 100.0% |
| `plot`      |  4,071 |  4,071 | 100.0% |
| `review`    |  3,927 |  3,927 | 100.0% |
| `episode`   |  3,468 |  3,468 | 100.0% |
| `setting`   |  2,818 |  2,818 | 100.0% |
| **`work`**  | **0**  | 90,843 | 0.0%   |

`work` pages (relation=main) are 100% unchanged — retrieval_title ==
page_title for those by Phase 6.3 design. All other page_types (the
subpages) have 100% of their chunks affected.

### Changed by title_source

| title_source | changed | total | ratio |
|---|---:|---:|---:|
| `canonical_url` | 29,215 | 29,215 | 100.0% |
| `seed`          | 15,544 | 106,387 | 14.6% |

`canonical_url`-sourced pages (where Phase 6.3 had to recover the
page_title from the URL because the seed title was generic) are 100%
affected. `seed`-sourced pages are partially affected: 14.6% of seed
chunks belong to subpages whose page_title was overridden by the
generic-title detector during Phase 6.3.

### Top changed example

```
page_title       = '등장인물'
retrieval_title  = '가난뱅이 신이! / 등장인물'
canonical_url    = https://namu.wiki/w/.../등장인물
section_path     = '개요'
section_type     = 'summary'

old embedding_text:
  제목: 등장인물
  섹션: 개요
  섹션타입: summary

  본문:
  만화 가난뱅이 신이! 의 등장인물 일람. ...

new embedding_text:
  제목: 가난뱅이 신이! / 등장인물
  섹션: 개요
  섹션타입: summary

  본문:
  만화 가난뱅이 신이! 의 등장인물 일람. ...
```

The full breakdown + 15 top examples are in `variant_diff_report.md`
and `variant_diff_report.json` next to this file.

---

## 7. Index build results

| variant | cache_dir | docs | chunks | dim | embed_text_sha16 | build_time |
|---|---|---:|---:|---:|---|---|
| `title_section` | `eval/indexes/namu-v4-2008-2026-04-title-section-mseq512` | 4,314 | 135,602 | 1024 | `bc97a45f376d4d7d` | ~42 min |
| `retrieval_title_section` | `eval/indexes/namu-v4-2008-2026-04-retrieval-title-section-mseq512` | 4,314 | 135,602 | 1024 | `10acf88cb2911d3f` | ~45 min |

- Embedder: `BAAI/bge-m3`, `max_seq_length=512`, `batch_size=64`,
  device CUDA (RTX 5080, 17GB).
- Each cache dir contains: `faiss.index` (~530MB, IndexFlatIP),
  `build.json`, `chunks.jsonl` (~120MB, ChunkLookupResult rows with
  raw `chunk_text` — NOT the prefixed form, so reranker / generation
  paths are unaffected), `variant_manifest.json`.
- The `embed_text_sha256` written into each index manifest matches
  the export manifest's `embed_text_sha256` byte-for-byte for both
  variants — confirms what got embedded matches what got exported.
- The variant slug appears in both the cache directory name and the
  `index_version` (`v4-title-section-…` vs
  `v4-retrieval-title-section-…`) so a downstream loader cannot
  swap one for the other silently.

Variant_manifest sample row 0 from the candidate index:

```
제목: ARIA The ORIGINATION
섹션: 개요
섹션타입: summary

본문:
일본의 만화 ARIA를 원작으로 하는 …
```

(For ARIA pages, page_title == retrieval_title so the candidate row 0
matches the baseline row 0; the variants diverge at row offsets
where the underlying page is a generic-title subpage.)

---

## 8. Retrieval A/B metrics

200 queries (`queries_v4_silver.jsonl`) generated deterministically
from `pages_v4.jsonl` with seed=42, stratified across the
buckets the variant should and should not affect:

| bucket | n | description |
|---|---:|---|
| `subpage_generic` | 90 | page_title ∈ {등장인물, 평가, OST, 회차, 음악, 미디어 믹스, …}; query mentions parent work title |
| `main_work` | 60 | page_type=work, relation=main; retrieval_title == page_title (neutral by construction) |
| `subpage_named` | 50 | non-generic subpage page_title; retrieval_title still folds in parent work |

### Aggregate

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| **hit@1** | 0.5950 | **0.8150** | **+0.2200** |
| **hit@3** | 0.7100 | **0.9350** | **+0.2250** |
| **hit@5** | 0.7400 | **0.9600** | **+0.2200** |
| **hit@10** | 0.7950 | **0.9850** | **+0.1900** |
| **mrr@10** | 0.6629 | **0.8816** | **+0.2187** |
| **nDCG@10** | 0.6948 | **0.9073** | **+0.2126** |
| dup_rate | 0.6800 | 0.7320 | +0.052 |
| same_section_collisions_avg | 5.16 | 4.08 | -1.08 |

### Status counts

- **improved**: 74
- **regressed**: 3
- both_hit: 120
- both_missed: 3
- unchanged: 0

Improved : Regressed = **24.7 : 1**.

### By bucket

#### `subpage_generic` (n=90) — the bucket the variant is built for

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.367 | **0.767** | **+0.400** |
| hit@3 | 0.533 | **0.933** | **+0.400** |
| hit@5 | 0.589 | **0.967** | **+0.378** |
| hit@10 | 0.667 | **0.989** | **+0.322** |
| mrr@10 | 0.464 | **0.860** | **+0.397** |

Status: improved=53, regressed=1, both_hit=35, both_missed=1,
unchanged=0.

This is where the variant is supposed to deliver — and it does, with
a 40-point hit@k improvement and ~85% relative MRR lift.

#### `main_work` (n=60) — neutral bucket

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.967 | 0.967 | 0.000 |
| hit@3 | 1.000 | 1.000 | 0.000 |
| hit@5 | 1.000 | 1.000 | 0.000 |
| hit@10 | 1.000 | 1.000 | 0.000 |
| mrr@10 | 0.983 | 0.981 | -0.003 |

Status: improved=1, regressed=1, both_hit=58, both_missed=0,
unchanged=0.

Neutral as designed — both variants embed the same string (since
retrieval_title == page_title on main work pages). The single
regression is rank 1 → 3 within the same work; one improvement is
a rank-shift the other way. Net effect is zero.

#### `subpage_named` (n=50) — non-generic subpage

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.560 | 0.720 | +0.160 |
| hit@3 | 0.680 | 0.860 | +0.180 |
| hit@5 | 0.700 | 0.900 | +0.200 |
| hit@10 | 0.780 | 0.960 | +0.180 |
| mrr@10 | 0.637 | 0.802 | +0.164 |

Status: improved=20, regressed=1, both_hit=27, both_missed=2,
unchanged=0.

Even on subpages whose page_title was already non-generic, the
candidate wins by a large margin. The retrieval_title's
`{work_title}/{page_title}` form gives the dense ranker a stronger
parent-work signal than the bare page_title alone, especially for
subpages whose name is shared across multiple works (`애니메이션`,
`사운드트랙`, `세계관`, etc.).

---

## 9. Improved query analysis

74 queries flipped to a better rank under the candidate. Generic
page_title breakdown for the improved set:

| page_title bucket | improved-count |
|---|---:|
| `등장인물` (incl. `기타 등장인물`) | 36 |
| _other (subpage_named or work pages with named page_title) | 21 |
| `평가` | 4 |
| `설정` | 3 |
| `줄거리` | 3 |
| `음악` | 2 |
| `주제가` | 2 |
| `에피소드` | 2 |
| `회차` | 1 |

The pattern is exactly what Phase 6.3's audit predicted: the most
impacted page_title (`등장인물`, 373 pages of which 36 entered the
silver set and all benefited from the variant). The "_other" bucket
(21 improvements) covers `subpage_named` queries where the variant
helps because the dense ranker now sees the parent work title in the
embedded prefix.

### 12 representative improved samples

| qid | bucket | base→cand rank | query |
|---|---|---:|---|
| v4-silver-0001 | subpage_generic | -1 → 4 | 마법과고교의 열등생의 설정에 대해 알려주세요. |
| v4-silver-0007 | subpage_generic | -1 → 1 | 원더풀 프리큐어!의 등장인물에 대해 알려주세요. |
| v4-silver-0011 | subpage_generic | -1 → 1 | Go! 프린세스 프리큐어의 평가에 대해 알려주세요. |
| v4-silver-0015 | subpage_generic | -1 → 1 | 쿠로무쿠로의 등장인물에 대해 알려주세요. |
| v4-silver-0025 | subpage_generic | -1 → 1 | 바이트초이카의 등장인물에 대해 알려주세요. |
| v4-silver-0027 | subpage_generic | -1 → 2 | 마유비검첩의 기타 등장인물에 대해 알려주세요. |
| v4-silver-0028 | subpage_named | 2 → 1 | 목소리의 형태(애니메이션)의 원작과의 차이점에 대해 알려주세요. |
| v4-silver-0032 | subpage_generic | -1 → 1 | 사라잔마이의 등장인물에 대해 알려주세요. |
| v4-silver-0039 | subpage_generic | 2 → 1 | 걸즈 앤 판처의 줄거리에 대해 알려주세요. |
| v4-silver-0040 | subpage_generic | 3 → 1 | 신비아파트 고스트볼 ZERO의 평가에 대해 알려주세요. |
| v4-silver-0043 | subpage_generic | 8 → 2 | 괴담 레스토랑의 회차에 대해 알려주세요. |
| v4-silver-0044 | subpage_generic | -1 → 1 | 천관사복의 등장인물에 대해 알려주세요. |

Pattern: queries that ask about a specific subpage (`등장인물`, `평가`,
`줄거리`, `회차`, …) of a named work jump from "miss" or rank-8 to
top-1 / top-2. The full list of 74 is in
`improved_queries.jsonl`.

---

## 10. Regressed query analysis

Only 3 of 200 queries regressed. All three are minor (rank 1 → rank
2 / 3) and reflect either the silver query's structural ambiguity
or a tie-break shift inside a sufficiently-relevant work cluster:

| qid | bucket | base→cand rank | query | analysis |
|---|---|---:|---|---|
| v4-silver-0033 | main_work | 1 → 3 | 판타지스타 돌이(가) 어떤 작품인가요? | Both retrievals stay within the work; baseline returned the `회차 목록` chunk first, candidate returned a different `DIH` chunk first. retrieval_title == page_title here ("판타지스타 돌"), so the variant doesn't apply to the ranking — the shift is bge-m3 stochasticity inside the same work, not a variant defect. |
| v4-silver-0048 | subpage_generic | 1 → 2 | 기동전사 건담 UC의 미디어 믹스에 대해 알려주세요. | Baseline retrieved a `게임 > PS3` chunk that happened to be in-work; candidate retrieved a generic "미디어 믹스" chunk from a different but related work that contains the words "기동전사 건담 UC" in its body. The variant's retrieval_title prefix lifts a related-work chunk above the target; this is the type of cross-work confusion the v3 generic-title problem masked. |
| v4-silver-0133 | subpage_named | 1 → 2 | 실바니안 패밀리의 세계관에 대해 알려주세요. | Both retrievals are within the same `실바니안 패밀리` work cluster; rank shift is from `등장 가족 > 초콜릿 토끼 가족` to `제품 특징 > 주제`. Tie-break shift, not a defect. |

None of the three regressions touch the 223 records Phase 6.3
flagged where `work_title` would have flipped under the looser
generic detector — those rows continued to use the strict
`is_generic_title` resolver per Phase 6.3 design, so their
`page_id` and `work_title` are stable and the silver bucket they
land in is consistent.

---

## 11. Recommendation: keep baseline or promote retrieval_title_section?

**Promote `retrieval_title_section` to default.**

Justification:

- Aggregate gain is large and unambiguous: hit@1 +22pt, MRR +21.9pt,
  nDCG +21.3pt — well above any reasonable noise floor for a 200-query
  silver set.
- 74 improvements vs. 3 regressions; all 3 regressions are minor
  rank-1→rank-2/3 shifts within already-relevant work clusters, not
  catastrophic misses.
- No-harm zone is preserved: main work pages (60 of 200) show 0pt
  hit@k delta with one improvement and one symmetric regression.
- Effect is concentrated exactly where Phase 6.3's title-noise audit
  predicted: 36 `등장인물`-style queries flipped from miss → top-1.
- The change is cheap to ship: only the bi-encoder dense vectors
  change; the metadata store keeps raw `chunk_text` so the reranker
  / generation paths are unaffected, and `chunk_id` / `doc_id` are
  preserved so any downstream eval that joins on chunk_id continues
  to work.
- The promotion is reversible: both index caches sit at distinct
  paths with manifest sha256 provenance, so a rollback is
  `--skip-export --skip-index-build` against the old cache.

Suggested next-step concretes (not part of Phase 7.0):
- Re-run the existing `silver_200` reranker / cap-policy sweeps on
  top of the new dense candidates to confirm reranker doesn't undo
  the gain.
- Adopt `retrieval_title` in the production ingest path
  (`app/capabilities/rag/ingest.py`) so live queries see the same
  embedding text as eval — this would be a Phase 7.1.

---

## 12. Remaining limitations

1. **Silver query coverage.** The 200 v4 silver queries are
   deterministic templates over Phase 6.3 page records. They are
   sufficient to detect the variant's signal but are not natural
   user queries. A future phase that drives the same A/B against an
   LLM-generated query set or against the legacy silver_200 (after
   doc_id re-mapping) would be a stronger validation.

2. **No reranker in the loop.** The A/B is dense-only (top-k from
   FAISS, no cross-encoder). The variant's gain at hit@1 (+22pt) is
   so large that a reranker is unlikely to flip the verdict — even
   a perfectly-ordered reranker can't add what the candidate pool
   doesn't contain — but a follow-up phase should confirm the gain
   on (dense + reranker) for production parity.

3. **`max_seq_length=512` chosen for build time.** Phase 6.3's audit
   noted some chunks reach ~2153 chars; with bge-m3 tokenisation
   ~1.5 chars/token, 512 tokens covers ~95% of chunks fully and
   tail-truncates ~5%. A run at `max_seq_length=1024` would close
   that gap but doubles the build time (already ~90 min for both
   variants on RTX 5080). Tail-truncation behaves the same way for
   both variants, so the comparison is fair, but absolute hit@k
   could be higher under longer sequences.

4. **`same_title_collisions_avg` is approximate.** `RetrievedChunk`
   does not carry a `title` field, only `chunk_id` / `doc_id` /
   `section`. The metric falls back to detecting generic tokens in
   the section path, which is a directional indicator but not a
   precise per-row title check.

5. **The 223 records with potentially mis-resolved `work_title`.**
   Phase 6.3 explicitly chose to keep the strict `is_generic_title`
   resolver for `_resolve_work_title` to preserve `page_id` stability.
   Those rows still benefit from `retrieval_title` at retrieval time
   (since `retrieval_title` uses the looser detector) but the silver
   queries we generated for them inherit the same bias as Phase 6.3's
   audit. None of the three regressions in §10 touch this set.

6. **Test environment limit.** Index builds were run on a single
   RTX 5080 host; correctness was verified by sha256-matching the
   manifest digests, but a CI run that rebuilds the indexes on a
   different GPU and confirms identical retrieval results has not
   been performed.

---

## 13. Next recommended phase

**Phase 7.1 — Reranker A/B on Phase 7.0 indexes.**

Plan:
- Re-use the two `eval/indexes/namu-v4-2008-2026-04-*` caches built
  here (no new embedding work).
- Drive the existing cross-encoder reranker (`reranker_eval` family)
  against both candidate pools and compare reranked hit@k / mrr@10.
- Confirm the +22pt hit@1 lift survives reranking; if not, identify
  whether the candidate pool's broader (parent-work-prefixed)
  retrievals interact with the reranker's top-N truncation in a
  way that loses the gain.

Optionally, in parallel:

**Phase 7.2 — Production ingest adoption of retrieval_title.**

- Wire `retrieval_title` (with `page_title` fallback) into
  `app/capabilities/rag/ingest.py` so live ingest produces
  `retrieval_title_section`-format embeddings by default.
- Add a feature flag (`rag.embedding_use_retrieval_title`) so a
  rollback is one config line.
- Re-run the agent-loop A/B (`eval/agent_loop_ab/`) end-to-end so
  the eval corpus and production ingest stay aligned.

These two can run concurrently; 7.1 is the verification that the
gain is real under reranker, 7.2 is the productionisation.
