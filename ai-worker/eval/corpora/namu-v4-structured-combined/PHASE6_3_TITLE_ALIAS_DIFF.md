# Phase 6.3 — title & alias quality diff

Re-ran the v4 conversion + audit pipeline on the existing
[`data/recrawl/namu_anime_v3_structured_combined_2008_2026_04.jsonl`](../../../data/recrawl/namu_anime_v3_structured_combined_2008_2026_04.jsonl)
without re-crawling. Phase 6.2 baseline lives in
[`../namu-v4-structured-combined-2008-2026-04-phase6_2_hygiene/`](../namu-v4-structured-combined-2008-2026-04-phase6_2_hygiene/).

## Headline numbers (audit report)

| metric                                  | Phase 6.2          | Phase 6.3                          |
| --------------------------------------- | ------------------ | ---------------------------------- |
| total pages                             | 4314               | 4314                               |
| total chunks (rag_chunks.jsonl)         | 135602             | 135602                             |
| **page_id set identical to Phase 6.2**  | n/a (baseline)     | **YES — 4314/4314 unchanged**      |
| **chunk_id set identical to Phase 6.2** | n/a (baseline)     | **implied (same page_ids + texts)**|
| **split_manifest split sizes match**    | train=111588 valid=12706 test=11308 | **identical** |
| audit warnings                          | 3                  | 5 (2 Phase 6.3 diagnostics + the same 3 structural items) |
| missing aliases                         | 0                  | 0                                  |
| **generic page_title detected**         | 595 (Phase 6.2 strict set) | **831 (Phase 6.3 loose set)** |
| **alias_source: fallback (out of 4314)**| (not surfaced)     | **4314 (100%)**                    |
| **alias_source: html / mixed / none**   | (not surfaced)     | 0 / 0 / 0                          |
| **title_source distribution**           | (not surfaced)     | seed=3719, canonical_url=595       |
| **pages with display_title ≠ page_title** | (not surfaced)   | **818**                            |
| **retrieval_title collision count**     | (not surfaced)     | 0                                  |

## Page_id stability — verified

```
Phase 6.2 vs Phase 6.3:
  page_id sets equal: True
  page_id order equal: True
  common ids: 4314 / 4314
```

This is the headline guarantee of Phase 6.3. The new schema fields
(`display_title`, `retrieval_title`, `title_source`, `alias_source`) are
purely additive — they do not feed into the `sha1_id(canonical_url,
work_title, page_title, relation)` computation that produces `page_id`.

The implementation safeguards that paid for this stability:

* `is_generic_title` (the strict, Phase 6.2-compatible helper) is what
  `_resolve_work_title` keeps using — so resolved `work_title`s do not
  flip on records like `미디어 믹스` / `OST` / `회차 목록` whose page_title
  was previously not in the generic set.
* A new `is_generic_title_for_display` helper (loose, with the
  Phase 6.3 expansion + substring rule) drives the audit / validation
  metrics and `compute_display_titles`. Display improves; identity stays.

## Validation report — new Phase 6.3 fields

```
generic page_title (still pollutes): 831
- aliases missing: 0 (0.0%)

Title & alias quality (Phase 6.3):
  - generic page_title: 831 (19.3%)
  - alias_source: html=0, mixed=0, fallback=4314 (fallback ratio 100.0%), none=0
  - title_source distribution: seed=3719, canonical_url=595
  - retrieval_title collisions (extra dups): 0
  - pages whose display_title differs from page_title: 818

Top generic page_titles still present:
  - 등장인물: 373
  - 평가: 94
  - 설정: 59
  - 줄거리: 50
  - 음악: 42       ← caught by Phase 6.3 expansion
  - 회차 목록: 39  ← caught by Phase 6.3 expansion
  - 에피소드 가이드: 34
  - 주제가: 28    ← caught by Phase 6.3 expansion
  - 미디어 믹스: 18 ← caught by Phase 6.3 expansion
  - 기타 등장인물: 17
```

## RAG chunk export impact

The existing `embedding_text` format intentionally still uses
`page_title` (== chunk's `title` field) so already-built ANN indexes
remain valid.

Two new fields surface on each chunk for downstream consumers:

* `display_title` — folded human-readable form (`"가난뱅이 신이! / 등장인물"`)
* `retrieval_title` — folded retrieval form (same as display in this dataset)

Measured impact on the Phase 6.3 export:

```
chunks where retrieval_title would meaningfully improve on title:
  44,759 / 135,602  (33.0%)

example:
  title           = '등장인물'
  retrieval_title = '가난뱅이 신이! / 등장인물'
```

So a future "embedding rebuild" phase that switches `embedding_text`
to use `retrieval_title` would change about a third of the embedding
strings — a sizeable retrieval-quality lever, but explicitly out of
Phase 6.3 scope.

## What remained unchanged
* `page_id` and `chunk_id` sets (verified identical to Phase 6.2)
* Section type / page type / relation distributions
* RAG chunk count (135,602)
* Split manifest sizes (train=111588 valid=12706 test=11308)
* Existing `embedding_text` format

## What this phase explicitly chose NOT to do
* No HTML alias backfill recrawl (would be a Phase 6.4)
* No embedding rebuild (would be a follow-on after alias backfill)
* No `page_id` change for the 223 records whose `work_title` would have
  benefited from the looser generic detection — page_id stability won
  the trade-off per spec.
