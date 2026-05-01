# Agent loop decision gate — 2026-04-23

Honest A/B over the 30-row difficulty-stratified Korean fixture
(`eval/datasets/rag_agent_fixture_kr.jsonl`) comparing the Phase-5
single-pass AGENT (loop off) against the Phase-6 iterative loop
(loop on).

**Decision: keep `AIPIPELINE_WORKER_AGENT_LOOP` default `off`.**

The loop adds cost (4× tokens, +7.4 ms p95) without measurable quality
gain in the offline configuration the registry falls back to when the
LLM backend is unavailable.

## Setup

| Field | Value |
|---|---|
| Fixture | `eval/datasets/rag_agent_fixture_kr.jsonl` (10 easy / 10 hard / 10 impossible) |
| Corpus | `fixtures/kr_sample.jsonl` (10 docs, 24 chunks) |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Retriever | offline FAISS (in-memory), top_k = 5, reranker = none |
| Generator | `ExtractiveGenerator` (deterministic, extractive) |
| Critic | `RuleCritic` (offline heuristic — LLM backend unavailable) |
| Rewriter | `NoOpQueryRewriter` (registry default when backend is `noop`) |
| Budget | `max_iter=3`, `max_total_ms=15000`, `min_confidence_to_stop=0.75` |
| Report | `eval/reports/phase8/agent_loop_decision.json` + `.csv` (this file's siblings) |

Run command:

```
AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
AIPIPELINE_WORKER_RAG_RERANKER=none \
AIPIPELINE_WORKER_LLM_BACKEND=noop \
python -m eval.run_eval rag \
  --dataset eval/datasets/rag_agent_fixture_kr.jsonl \
  --offline-corpus fixtures/kr_sample.jsonl \
  --agent-mode compare \
  --top-k 5 \
  --out-json eval/reports/phase8/agent_loop_decision.json \
  --out-csv  eval/reports/phase8/agent_loop_decision.csv
```

## Results

### Overall

| Metric | loop=off | loop=on | Δ |
|---|--:|--:|--:|
| rows | 30 | 30 | — |
| mean recall@5 | 0.8500 | 0.8500 | 0.0000 |
| MRR | 0.8750 | 0.8750 | 0.0000 |
| mean keyword_coverage | 0.8667 | 0.8667 | 0.0000 |
| mean latency (ms) | 4.47 | 12.74 | +8.27 |
| p50 latency (ms) | 3.90 | 12.50 | +8.60 |
| p95 latency (ms) | 7.28 | 14.68 | +7.40 |
| mean tokens | 265.3 | 1061.1 | ×4.00 |

### Per difficulty

| Difficulty | mode | n | recall@5 | MRR | kw_cov | p50 ms | p95 ms |
|---|---|--:|--:|--:|--:|--:|--:|
| easy | off | 10 | 0.9000 | 0.8000 | 0.8000 | 3.90 | 23.18 |
| easy | on  | 10 | 0.9000 | 0.8000 | 0.8000 | 12.50 | 14.60 |
| hard | off | 10 | 0.8000 | 0.9500 | 0.8000 | 3.24 | 4.22 |
| hard | on  | 10 | 0.8000 | 0.9500 | 0.8000 | 12.19 | 14.77 |
| impossible | off | 10 | n/a | n/a | 1.0000 | 3.90 | 4.42 |
| impossible | on  | 10 | n/a | n/a | 1.0000 | 11.92 | 14.68 |

*(recall@5 / MRR are `n/a` on the impossible slice because those rows
carry no `expected_doc_ids` — nothing in the corpus is correct.)*

### Agent metrics

| Metric | Value |
|---|--:|
| loop_recovery_rate (overall, threshold=0.5) | 0.0000 |
| loop_recovery_rate (hard, threshold=0.5) | n/a |
| avg_cost_multiplier (total_tokens / iter0_tokens) | 4.0000 |
| iter_count_mean | 3.0000 |
| answer_recall_delta (final_kw_cov − iter0_kw_cov) | 0.0000 |

### Stop-reason distribution (loop=on)

| stop_reason | count | fraction |
|---|--:|--:|
| `iter_cap` | 30 | 1.000 |

All 30 runs hit the iter cap. The `RuleCritic`'s default "sufficient"
verdict emits `confidence=0.65`, which is below
`LoopBudget.min_confidence_to_stop=0.75`, so the controller never takes
the `converged` branch and runs every row to `max_iter=3`.

## Decision gate

Rule:

> `loop_recovery_rate(hard) >= 0.15`
> AND `avg_cost_multiplier <= 3.0`
> AND `p95 latency increase <= 8s`
> → flip default to `on` and update `docs/architecture.md`.
> Else → keep default `off` and document why.

Against the measured numbers:

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| `loop_recovery_rate(hard)` | ≥ 0.15 | **n/a** (zero hard rows below 0.5 threshold) | **FAIL** — insufficient evidence of recovery |
| `avg_cost_multiplier` | ≤ 3.0 | **4.00** | **FAIL** — 33 % over budget |
| `p95` latency increase | ≤ 8000 ms | **7.4 ms** | pass |

Two of the three gates fail. **Keep default `off`.**

## Why the loop adds no value here

The offline configuration used in this run matches what the registry
constructs when no LLM backend is reachable:

1. **Rule critic**: deterministic heuristic (length ≥ 40 chars + no
   "I don't know" markers ⇒ `sufficient=True, confidence=0.65`). Every
   non-empty extractive answer passes, so the critic never points to
   an actionable gap.
2. **NoOp rewriter**: the registry swaps in `NoOpQueryRewriter` when
   the chat backend is noop. Iteration N just re-runs the same query,
   so the retriever returns the same chunks and the synthesizer — which
   grounds on the *union* of every iteration's chunks — sees exactly
   the iter-0 chunk set. The final answer is byte-identical to the
   iter-0 answer.
3. **Confidence vs. min-stop-confidence mismatch**: rule critic
   confidence 0.65 < `min_confidence_to_stop` 0.75, so a "sufficient"
   verdict never triggers `converged`. The loop always pays the full
   `iter_cap` (3 iterations) before stopping.

Net effect: loop on = loop off on answer quality, loop on = 4× cost
and +7.4 ms p95 latency. The latency overhead is below the 8 s gate,
but the cost multiplier and the (undefined) hard-set recovery rate are
enough to kill the flip.

### Metric caveats worth recording

* On the `impossible` slice, `mean_keyword_coverage=1.0` is an
  artifact of the metric: `ExtractiveGenerator` prints the query
  verbatim ("**Query:** …") so any keyword that appears in the query
  text is trivially "covered" regardless of retrieval quality.
  Coverage on impossible rows is not a signal here.
* Because every hard row scored ≥ 0.5 keyword coverage at iter 0, the
  `loop_recovery_rate(hard)` denominator was 0 and the metric is
  returned as `None`. The fixture honestly reflects retrieval
  headroom (hard queries are multi-fact across docs), but MiniLM +
  top-5 was strong enough on this corpus that no hard row was a
  "needing recovery" candidate for the 0.5 threshold. A tighter
  threshold or a harder corpus would give the loop a shot at proving
  itself; neither is appropriate to introduce here without writing
  code to cherry-pick the gate.

## Next steps (post-commit)

The failed gate is the evidence. Only after committing this result:

1. Lower `agent_min_stop_confidence` so the rule critic can early-stop
   and the loop degrades cleanly to single-pass when the LLM backend
   is unavailable. Current default 0.75 guarantees `iter_cap=3` under
   the rule critic — that is a pure cost, not a quality knob.
2. Re-run this gate with a live LLM backend (Ollama + `gemma4:e2b`),
   where the rewriter is the real `LlmQueryRewriter` and the critic
   can see retrieved chunks. The current result says nothing about
   loop value under that configuration — only that it is worse than
   off without an LLM.
3. Consider a harder fixture slice where iter-0 single-pass recall is
   genuinely below 0.5 on hard queries (e.g. cross-doc queries where
   the gold docs don't share embedding-space vocabulary). The current
   fixture satisfies the Phase 8 acceptance criteria (≥30 rows,
   stratified, honest labels) but does not stress the retriever
   enough to create recovery opportunities.
