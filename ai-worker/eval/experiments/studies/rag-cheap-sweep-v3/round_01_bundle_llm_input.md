# Round round_01 — study bundle

**Study:** `rag-cheap-sweep-v3`
**Objective:** `mrr` (maximize)
**Optuna:** `4.5.0` | sampler `TPESampler` | pruner `NopPruner`
**Parent config hash:** `0e073814a046c6ebe348110fe1949d4ad66f2880753aec3786c2920cd03615db`
**Bundle hash:** `1827a27b9055c65b24aa0706dac8905f8f0f2c49b711894090c7d13bd8b4dbfc`

---

## 1. Frozen search space (this round)

| Param | Type | Range / choices |
|-------|------|-----------------|
| `rag_top_k` | int | [3, 15] |
| `rag_use_mmr` | categorical | False, True |
| `rag_query_parser` | categorical | 'off', 'regex' |

## 2. Headline statistics

- Trials: **18** (complete 18, pruned 0, failed 0)
- Best value: **0.8778**
- Quantiles: p10 `0.8667`, p50 `0.8778`, p90 `0.8778`
- Mean ± std: `0.8747166666666667 ± 0.004971725611450772`

## 3. Param importances

| Param | Importance |
|-------|-----------|
| `rag_top_k` | 1.0000 |
| `rag_use_mmr` | 0.0000 |
| `rag_query_parser` | 0.0000 |

## 4. Boundary hits

- `rag_top_k`: low=2, high=0

## 5. Best trial

```json
{
  "number": 0,
  "state": "COMPLETE",
  "value": 0.8778,
  "params": {
    "rag_top_k": 7,
    "rag_use_mmr": false,
    "rag_query_parser": "off"
  },
  "datetime_start": "2026-04-23T21:30:52.959761",
  "datetime_complete": "2026-04-23T21:31:07.226876",
  "user_attrs": {
    "config_hash": "fcf99e483dce",
    "cost_usd": 0.0,
    "eval_wall_ms": 13555.637,
    "latency_ms": 13566.954,
    "secondary_metric_values": {
      "mean_hit_at_k": 0.9667,
      "mean_keyword_coverage": 0.9556,
      "mean_total_ms": 39.228,
      "p50_retrieval_ms": 30.974
    }
  }
}
```

## 6. Top-k trials (k=10)

| # | value | params |
|---|-------|--------|
| 0 | 0.8778 | `{"rag_top_k": 7, "rag_use_mmr": false, "rag_query_parser": "off"}` |
| 4 | 0.8778 | `{"rag_top_k": 10, "rag_use_mmr": true, "rag_query_parser": "regex"}` |
| 5 | 0.8778 | `{"rag_top_k": 13, "rag_use_mmr": true, "rag_query_parser": "off"}` |
| 6 | 0.8778 | `{"rag_top_k": 10, "rag_use_mmr": false, "rag_query_parser": "regex"}` |
| 7 | 0.8778 | `{"rag_top_k": 13, "rag_use_mmr": false, "rag_query_parser": "off"}` |
| 8 | 0.8778 | `{"rag_top_k": 7, "rag_use_mmr": false, "rag_query_parser": "off"}` |
| 11 | 0.8778 | `{"rag_top_k": 11, "rag_use_mmr": true, "rag_query_parser": "regex"}` |
| 12 | 0.8778 | `{"rag_top_k": 9, "rag_use_mmr": true, "rag_query_parser": "off"}` |
| 13 | 0.8778 | `{"rag_top_k": 14, "rag_use_mmr": true, "rag_query_parser": "regex"}` |
| 14 | 0.8778 | `{"rag_top_k": 9, "rag_use_mmr": false, "rag_query_parser": "off"}` |

## 7. Clusters

- **value=0.8778 plateau (13 trials)** — trials 0, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17
- **value=0.8667 plateau (5 trials)** — trials 1, 2, 3, 9, 10

## 8. Operator notes

v3 — first round wired against a workload with actual headroom.

---

## Your task

You are the outer-loop analyst. Produce a round report and a next-round config JSON per the skill's `prompts/claude_code/propose_next_round.md`. Every search_space change MUST cite a field of this bundle.
