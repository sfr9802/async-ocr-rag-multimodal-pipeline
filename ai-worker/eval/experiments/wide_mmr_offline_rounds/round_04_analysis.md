# Round 04 — `rag-wide-mmr-offline-v1`

## TL;DR

Subset expansion 50→100 restored objective variance (std=0.0025 vs 0 in round_03). The signal **fully resolves to a 2-axis interaction**:

| candidate_k | cap_rr=1 | cap_rr=2 | cap_rr=3 |
|---|---|---|---|
| 100 | **0.6745** (9/9) | 0.6695 (1/1) | 0.6695 (3/3) |
| 200 | 0.6695 (1/1) | 0.6695 (1/1) | 0.6695 (1/1) |

Importance:
- `title_cap_rerank_input`: 0.6389
- `candidate_k`: 0.2418
- `mmr_lambda`: 0.0986 (with both edges still UNSAMPLED)
- `mmr_k`: 0.0137
- everything else: ~0

The winning recipe is `candidate_k=100, title_cap_rerank_input=1` — both must be set together. **NO single axis alone determines the win**: cand=200 with cap=1 stays at 0.6695, just like cand=100 with cap=2 or 3.

## Decision for round 05

Narrow both winner axes to their winning value:
- `candidate_k` → `[100]`
- `title_cap_rerank_input` → `[1]`

This is well-evidenced:
- 9 of 9 cand_k=100 + cap_rr=1 trials scored exactly 0.6745.
- 5 of 5 trials at the OTHER (cand_k, cap_rr) combinations scored exactly 0.6695.
- The 0.005 MRR gap is consistent and the cluster boundary is sharp.

Keep `final_top_k`, `mmr_k`, `mmr_lambda`, `title_cap_final` open: importance < 0.10 each, no axis-coverage evidence to narrow them. Round 5 with cand+cap_rr fixed should expose what (if anything) carries residual signal among them.

## Anti-pattern audit

- **A10 (narrow against unsampled edge):** `mmr_lambda` edges still UNSAMPLED — kept full range.
- **A11 (narrow with axis evidence):** `candidate_k` and `title_cap_rerank_input` have clear deterministic value buckets — narrowing is justified.
- **A12 (no narrowing without evidence):** `title_cap_final` importance 0.0; final_top_k 0.007; mmr_k 0.0137 — all kept because the 50-100 row subset can't yet distinguish them.

## Headline diff (round_04 → round_05)

| field | round_04 | round_05 | reason |
|---|---|---|---|
| `search_space.candidate_k` | `[100, 200]` | `[100]` | 9/9 winners use cand=100; 3/3 cand=200 trials → 0.6695 |
| `search_space.title_cap_rerank_input` | `[1, 2, 3]` | `[1]` | 9/10 cap=1 trials → 0.6745 (only the cand=200 cap=1 trial dropped); cap=2/3 → 0.6695 always |
| `n_trials` | 16 | 16 | unchanged |
| `sampler.seed` | 45 | 46 | reseed after narrowing |
