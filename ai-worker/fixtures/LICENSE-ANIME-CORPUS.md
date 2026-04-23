# LICENSE-ANIME-CORPUS

The fixture file `anime_corpus_kr.jsonl` committed alongside this
document is a derivative, 300-title reservoir sample produced by
`scripts/dataset/sample_anime_corpus.py` from the source JSONL
`namu_anime_v3.fixed.jsonl` maintained in the sibling project
`port/rag`.

## Provenance

- Upstream source: port/rag preprocessing pipeline over publicly
  reachable namu-wiki pages (Korean-language community wiki covering
  animation titles).
- Sampling: deterministic reservoir sample (Algorithm R, Vitter 1985)
  with seed `42` and `--sample-size 300`. Dedup by `seed_title` during
  streaming.
- Transformations: list-of-sections -> dict-of-sections with
  `section_order`, duplicate section names disambiguated with `#N`
  suffix, plus `domain`, `language`, `source`, and `source_ts` metadata
  fields added for downstream filtering. No content was rewritten.

## License statement

Sampled from namu-wiki content via the port/rag preprocessing pipeline.
Derivative summaries only; the original namu-wiki content is subject to
CC BY-NC-SA 2.0 Korea terms, which may apply to the raw text included
in this derivative. Downstream users must verify the upstream license
before redistribution — in particular:

- Non-commercial use only under CC BY-NC-SA 2.0.
- Share-alike: derivative works must be released under the same license.
- Attribution: credit to namu-wiki contributors.

The code that produces this fixture (the `scripts/dataset/*.py` files)
is MIT-licensed and belongs to this repository.

## Regenerating

From `ai-worker/`:

```
python -m scripts.dataset.sample_anime_corpus \
    --source 'D:/port/rag/app/scripts/namu_anime_v3.fixed.jsonl' \
    --out    fixtures/anime_corpus_kr.jsonl \
    --sample-size 300 \
    --seed   42
```

The sampler is deterministic: the same source file + seed + size will
produce the same set of 300 titles every time. Changing the seed (or
the `--sample-size`) produces a different sample and should be
accompanied by a new commit that updates both `anime_corpus_kr.jsonl`
and `anime_corpus_kr.meta.json`.
