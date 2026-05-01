# Legacy retrieval baseline (locked)

This directory holds the **locked legacy retrieval / rerank baseline** that LangGraph A/B testing measures against. It is consolidated from the ``phase2a-latency-sweep`` run referenced by ``sourceSweepDir`` in ``baseline_manifest.json`` (``eval/reports/phase2/2a_latency``). The sweep was executed against the same commit recorded in ``baseline_manifest.json -> commitHash`` and the same dataset / corpus / model the A/B test will use, so re-running the consolidation step against the same sweep dir is byte-stable (modulo the ``manifestGeneratedAt`` timestamp).

## Why this exists

The LangGraph backend (``AgentLoopGraph``) is the experimental alternative to the legacy ``AgentLoopController``. Before we promote it, we need a stable reference for the legacy stack — this directory is that reference. ``baseline_manifest.json`` records the corpus / index / model / metrics / latency snapshot; ``selected_config.json`` records the canonical config the A/B compares against.

## File layout

| file | purpose |
|---|---|
| `metrics.json` | machine-readable accuracy + latency metrics for fast/balanced/quality tiers, plus per-answer-type / per-difficulty slices for the selected tier. |
| `metrics.md` | human-readable summary of `metrics.json`. |
| `latency_breakdown.json` | stage-level latency breakdown (dense_retrieval / tokenize / forward / postprocess / total) anchored on the selected tier. Schema `phase2a-latency-breakdown.v1`. |
| `latency_breakdown.md` | rendered version of `latency_breakdown.json`. |
| `selected_config.json` | canonical config — retriever + reranker + agent-loop knobs — for the A/B legacy arm. Includes the exact CLI to reproduce. |
| `baseline_manifest.json` | provenance + metrics + latency manifest. The single doc the A/B harness should read to anchor its legacy reference. |
| `../phase2/2a_latency/` | upstream `phase2a-latency-sweep` output this baseline was consolidated from. Do not delete — the manifest's `sourceSweepDir` field points back to it. |

`baseline_manifest.json -> sourceSweepDir = "eval/reports/phase2/2a_latency"`. `evaluatedAt` records when that sweep ran (`2026-04-28T22:33:42`); `manifestGeneratedAt` records when this directory was assembled.

## Tier selection

Three tiers were considered (Pareto frontier of ``mean_hit_at_1`` ↑ vs ``rerank_p95_ms`` ↓):

- `fast`: dense_top_n=5, final_top_k=5, hit@1=0.6150, rerank_p95_ms=164.41. lowest-latency on-frontier entry.
- `balanced` — **selected**: dense_top_n=10, final_top_k=10, hit@1=0.6200, rerank_p95_ms=334.85. median-latency on-frontier entry (2 of 3 sorted by latency).
- `quality`: dense_top_n=15, final_top_k=10, hit@1=0.6250, rerank_p95_ms=535.42. on-frontier entry with the highest mean_hit_at_1.

`balanced` is the default A/B reference because it sits on the Pareto frontier with the best latency/quality balance — the median-latency entry among the frontier-eligible configs.

## Reproducing this baseline

```bash
# 1. Re-run the legacy sweep across the full topN axis.
#    (~50 minutes on RTX 5080 — embedding 47k chunks dominates.)
python -m eval.run_eval phase2a-latency-sweep \
    --dataset eval/eval_queries/anime_silver_200.jsonl \
    --corpus eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \
    --out-dir eval/reports/phase2/2a_latency \
    --final-top-k 10 \
    --dense-top-n 5 --dense-top-n 10 --dense-top-n 15 \
    --dense-top-n 20 --dense-top-n 30 --dense-top-n 50 \
    --breakdown-anchor-dense-top-n 10 \
    --candidate-recall-extra-hit-k 10 \
    --candidate-recall-extra-hit-k 20 \
    --candidate-recall-extra-hit-k 50 \
    --metric mean_hit_at_1 --latency rerank_p95_ms

# 2. Re-build the consolidated baseline files.
python -m scripts.build_legacy_baseline_final \
    --sweep-dir eval/reports/phase2/2a_latency \
    --out-dir eval/reports/legacy-baseline-final \
    --selected-tier balanced
```

Step 1 takes ~50 min on a single RTX 5080 (47k-chunk bge-m3 embedding dominates, then 6 rerank passes run in serial). Step 2 is pure post-processing and finishes in milliseconds. The two together produce identical metrics within rerank latency noise as long as the corpus, query set, embedding model, and reranker model are unchanged.

## Next step — LangGraph A/B

Once this baseline is locked, run the offline A/B harness to compare the legacy ``AgentLoopController`` against the experimental ``AgentLoopGraph`` on the same query set:

```bash
# Stub-mode smoke (no FAISS / no GPU needed)
python -m scripts.eval_agent_loop_ab \
    --queries eval/agent_loop_ab/smoke.jsonl \
    --mode stub --run-name smoke

# Live registry-mode A/B against the silver-200 set
# (requires the FAISS index + same WorkerSettings as the worker;
# does NOT touch Redis / DB / callbacks)
python -m scripts.eval_agent_loop_ab \
    --queries eval/eval_queries/anime_silver_200.jsonl \
    --mode registry \
    --run-name legacy-vs-graph-anime_silver_200 \
    --critic rule --parser regex \
    --max-iter 3 --max-total-ms 15000 \
    --max-llm-tokens 4000 --min-confidence 0.75
```

Outputs land in `eval/agent_loop_ab/<run-name>/` (`raw_results.jsonl`, `summary.csv`, `comparison_summary.json`). The legacy arm of the comparison must reproduce the metrics in this directory; if it doesn't, treat that as a regression — re-anchor before reading the graph results.

**silver_200 schema:** the loader in ``eval/harness/agent_loop_ab.py`` accepts both the singular (``expected_doc_id`` / ``expected_keywords``) and the silver_200 plural (``expected_doc_ids[0]`` / ``expected_section_keywords``) shapes, so ``anime_silver_200.jsonl`` plugs in directly with full per-row hit@k / keyword metrics — no projection step needed. When both shapes are present the singular fields win, so an operator can override the fallback inline.

## Hard rules for this baseline

1. **Do not edit the LangGraph backend** to make A/B numbers look better. The legacy reference must stay frozen.
2. **`agent_loop_backend` default stays `legacy`**. The graph backend is opt-in only.
3. **No Redis / DB / callback / Spring repo / infra mutations** from this directory's tooling. Everything here is post-processing.
4. **Re-runs go through the upstream sweep dir** (``eval/reports/phase2/2a_latency``) and then through `build_legacy_baseline_final.py`. Don't hand-edit `metrics.*` or `baseline_manifest.json` — re-run the script.
