# Eval datasets — catalogue

Every JSONL file in this directory is an eval dataset consumed by the
CLI harness at `ai-worker/eval/run_eval.py`. They split cleanly across
two axes: **domain** (`anime` fixtures kept for retrieval-quality
regression vs. `enterprise` fixtures added in Phase 9 as the
production-shaped target) and **harness** (`rag` / `ocr` / `multimodal` /
`routing` / `cross-domain`).

## Catalogue

| File | Domain | Language | Rows | Purpose | License |
|------|--------|----------|-----:|---------|---------|
| `rag_sample.jsonl` | anime | en | 6 | Phase 0 EN anime RAG baseline — regression-test the FAISS+bge-m3 retrieval on the original 8-doc anime fixture | Apache-2.0 |
| `rag_sample_kr.jsonl` | enterprise (KR placeholder) | ko | 10 | Phase 0 KR baseline against the legacy `kr_sample.jsonl` 10-doc fixture. Retained so Phase 9 baselines can prove no regression vs. Phase 0 numbers | Apache-2.0 |
| `rag_anime_extended_kr.jsonl` | anime | ko | 30 | Phase 9 extended Korean anime eval against `anime_kr.jsonl` (10 docs × 3 queries, difficulty easy:10 / medium:15 / hard:5) | Apache-2.0 |
| `rag_enterprise_kr.jsonl` | enterprise | ko | 20 | Phase 9 primary enterprise RAG eval — seeded representative set across all 5 categories. Scales to ~200 via `scripts.dataset.generate_queries` | Apache-2.0 |
| `rag_cross_domain_kr.jsonl` | cross-domain | ko | 20 | Phase 9 unanswerable set — every row pins a filter that restricts search to the WRONG domain. The harness gate is `cross_domain_refusal_rate >= 0.85`; the relevance-gated `ExtractiveGenerator` emits "문서에서 관련 정보를 찾을 수 없습니다" when top score < 0.48 | Apache-2.0 |
| `rag_agent_fixture_kr.jsonl` | enterprise (KR placeholder) | ko | ~50 | Phase 8 agent-loop decision-gate fixture — difficulty-stratified (easy / hard / impossible) queries against `kr_sample.jsonl` | Apache-2.0 |
| `ocr_sample.jsonl` | (general) | en | 3 | Phase 2 EN OCR fixtures — synthetic invoices/pangrams; regenerate with `scripts.make_ocr_sample_fixtures` | Apache-2.0 |
| `ocr_sample_kr.jsonl` | (general) | ko | 2 | Phase 2 KR OCR fixtures — synthetic notices; regenerate with `scripts.make_ocr_sample_fixtures` | Apache-2.0 |
| `ocr_enterprise_kr.jsonl` | enterprise | ko | 5 | Phase 9 enterprise OCR eval — renders each corpus doc into a PNG page; scales to 50 via `scripts.dataset.synthesize_ocr_pages --count 50` | Apache-2.0 |
| `multimodal_sample.jsonl` | mixed | en+ko | 9 | Multimodal eval spanning OCR-only / visual-only / OCR+visual rows plus Phase 9 anime posters | Apache-2.0 |
| `multimodal_anime_kr.jsonl` | anime | ko | 6 | Phase 9 anime-poster multimodal demo set (6 programmatic posters); images under `fixtures/posters/` (CC0) | Apache-2.0 |
| `multimodal_enterprise_kr.jsonl` | enterprise | ko | 5 | Phase 9 enterprise multimodal eval — reuses the `ocr_enterprise/` rendered pages; scales to 50 via `scripts.dataset.generate_multimodal --count 50` | Apache-2.0 |
| `routing_enterprise_kr.jsonl` | enterprise | ko | 15 | Phase 9 AUTO-routing seed eval — 5 actions (rag / ocr / multimodal / direct_answer / clarify); scales to 80 via `scripts.dataset.generate_routing_cases --total 80` | Apache-2.0 |

## How the datasets fit together

1. **Phase 0 regression lock.** `rag_sample.jsonl` and
   `rag_sample_kr.jsonl` are Phase 0 fixtures retained unchanged —
   their recall@5 numbers on the unified Phase 9 index (`--fixture all`)
   must stay within ±0.03 of their Phase 0 baselines, otherwise the
   enterprise corpus is bleeding into anime queries.

2. **Enterprise corpus regeneration.** The 5 hand-authored seed docs
   under `fixtures/corpus_kr/` are the committed starting point; scale
   to 125 docs with `python -m scripts.dataset.build_corpus --out
   ai-worker/fixtures/corpus_kr --per-category 25`. Rebuild the index
   row with `python -m scripts.dataset.rebuild_corpus_index --out
   ai-worker/fixtures/corpus_kr`.

3. **Eval-set scaling.** Each enterprise eval file has a hand-authored
   seed + a scripted path to the full N. Use `--dry-run` on the
   generator scripts first to plan sizes/cost before spending API
   tokens.

4. **Cross-domain gate.** `rag_cross_domain_kr.jsonl` is the
   definitive test that `ParsedQuery.filters` actually narrows the
   search space. The relevance-gated `ExtractiveGenerator` turns a
   top-chunk-score-below-0.48 retrieval into a refusal; an LLM-backed
   generator would reach the same refusal through semantic reasoning.

## Rebuild the Phase 9 baseline reports

From `ai-worker/`:

```
python -m scripts.build_rag_index --fixture all
python -m eval.run_eval rag --dataset eval/datasets/rag_sample.jsonl         --out-json eval/reports/phase9-baseline/rag_sample.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_sample_kr.jsonl      --out-json eval/reports/phase9-baseline/rag_sample_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_anime_extended_kr.jsonl --out-json eval/reports/phase9-baseline/rag_anime_extended_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_enterprise_kr.jsonl  --out-json eval/reports/phase9-baseline/rag_enterprise_kr.json --no-csv
python -m eval.run_eval rag --dataset eval/datasets/rag_cross_domain_kr.jsonl --cross-domain \
    --out-json eval/reports/phase9-baseline/rag_cross_domain_kr.json --no-csv
```

The reports under `eval/reports/phase9-baseline/` are committed so
Phase 10 (Optuna) has a clear comparison point. The bge-m3 model is
GPU-accelerated when CUDA is available; CPU builds take ~3-4x longer
but produce byte-identical output.
