# Phase 1C — token-aware corpus emit summary

- chunker_version: `token-aware-v1`
- source: `eval\corpora\anime_namu_v3_preprocessed\corpus.prefix-v1.inline-edit-v1.jsonl`
- output: `eval\corpora\anime_namu_v3_token_chunked\corpus.combined.token-aware-v1.jsonl`
- config: target=512 · soft_max=768 · hard_max=1024 · overlap=80
- documents: 1764
- sections: 4609

## Accounting

- **input payload units**: 221458 _(non-empty entries from ``chunks`` / ``list`` / ``text``, before splitting)_
- **output retrievable chunks**: 47634 _(what the FAISS index would contain)_
- output / input ratio: **0.215× **

## Strategy breakdown

| split_strategy | chunks |
|---|---:|
| `paragraph` | 45300 |
| `short` | 2305 |
| `hard_token` | 20 |
| `sentence` | 9 |

## Fallback / overflow

- chunks needing hard-token / hard-char fallback: 20
- chunks > hard_max_tokens (should be 0 in production emit): 0
- sections emitting zero chunks: 0

