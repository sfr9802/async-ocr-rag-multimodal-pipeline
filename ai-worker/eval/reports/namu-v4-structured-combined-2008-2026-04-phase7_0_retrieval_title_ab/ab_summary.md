# Phase 7.0 A/B — title_section vs retrieval_title_section

- n_queries: **200**
- k_values: [1, 3, 5, 10]

## Aggregate metrics

| metric | baseline | candidate | Δ (cand − base) |
|---|---:|---:|---:|
| hit@1 | 0.5950 | 0.8150 | +0.2200 |
| hit@3 | 0.7100 | 0.9350 | +0.2250 |
| hit@5 | 0.7400 | 0.9600 | +0.2200 |
| hit@10 | 0.7950 | 0.9850 | +0.1900 |
| mrr_at_10 | 0.6629 | 0.8816 | +0.2187 |
| ndcg_at_10 | 0.6948 | 0.9073 | +0.2126 |
| dup_rate | 0.6800 | 0.7320 | +0.0520 |
| same_title_collisions_avg | 5.1600 | 4.0800 | -1.0800 |

## Status counts

- both_hit: **120**
- both_missed: **3**
- improved: **74**
- regressed: **3**
- unchanged: **0**

## By bucket

### main_work (n=60)

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.9667 | 0.9667 | +0.0000 |
| hit@3 | 1.0000 | 1.0000 | +0.0000 |
| hit@5 | 1.0000 | 1.0000 | +0.0000 |
| hit@10 | 1.0000 | 1.0000 | +0.0000 |
| mrr_at_10 | 0.9833 | 0.9806 | -0.0028 |
| ndcg_at_10 | 0.9877 | 0.9855 | -0.0022 |

- status: both_hit=58, both_missed=0, improved=1, regressed=1, unchanged=0

### subpage_generic (n=90)

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.3667 | 0.7667 | +0.4000 |
| hit@3 | 0.5333 | 0.9333 | +0.4000 |
| hit@5 | 0.5889 | 0.9667 | +0.3778 |
| hit@10 | 0.6667 | 0.9889 | +0.3222 |
| mrr_at_10 | 0.4637 | 0.8602 | +0.3965 |
| ndcg_at_10 | 0.5124 | 0.8926 | +0.3802 |

- status: both_hit=35, both_missed=1, improved=53, regressed=1, unchanged=0

### subpage_named (n=50)

| metric | baseline | candidate | Δ |
|---|---:|---:|---:|
| hit@1 | 0.5600 | 0.7200 | +0.1600 |
| hit@3 | 0.6800 | 0.8600 | +0.1800 |
| hit@5 | 0.7000 | 0.9000 | +0.2000 |
| hit@10 | 0.7800 | 0.9600 | +0.1800 |
| mrr_at_10 | 0.6370 | 0.8015 | +0.1645 |
| ndcg_at_10 | 0.6715 | 0.8400 | +0.1685 |

- status: both_hit=27, both_missed=2, improved=20, regressed=1, unchanged=0

