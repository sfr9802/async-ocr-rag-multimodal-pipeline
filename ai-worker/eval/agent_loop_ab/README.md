# agent_loop_ab — offline A/B harness output

Outputs from `scripts/eval_agent_loop_ab.py` land in
`eval/agent_loop_ab/<run-name>/`. The harness compares the legacy
`AgentLoopController` against the experimental `AgentLoopGraph`
(LangGraph backend) on the same query set, **without** going through
Redis, the TaskRunner, or core-api callbacks.

## Output files

* `raw_results.jsonl` — one JSON object per input query. Each row carries
  the full per-backend metric block under `legacy.*` and `graph.*` plus
  the per-query verdict (`graph_win` | `legacy_win` | `tie` |
  `regression`).
* `summary.csv` — flat per-(backend, query) row. Two rows per query
  (legacy + graph). Column order is stable (`AGENT_LOOP_AB_METRIC_COLUMNS`)
  so successive runs diff cleanly.
* `comparison_summary.json` — aggregate statistics: success rate, p50 /
  p95 latency, average iteration / rewrite counts, hit@k aggregates
  (when the dataset carries `expected_doc_id`), and a verdict counter.
  The `recommendation` field encodes the spec's adoption rules:
  * `adopt_candidate` — graph improves quality without a latency blow-up.
  * `hold_no_quality_gain` — graph adds LLM/retrieval cost with no
    quality movement.
  * `hold_experimental_backend_only` — only trace/debuggability improves.
  * `hold_review_regressions` — quality lift is real but regressions
    > 0; investigate before adoption.

## Input schema

JSONL or CSV; the loader sniffs the suffix. Required field: `query`.
Optional fields: `expected_doc_id`, `expected_keywords`, `input_kind`,
`capability` (`rag` | `multimodal`), `metadata`. CSV stores list
fields with `|` (or `,`) separators; JSON-shaped metadata can be
embedded as a `metadata` column with stringified JSON.

## Sample fixture

`smoke.jsonl` is a tiny offline-safe fixture (uses the stub retriever):

    python -m scripts.eval_agent_loop_ab \
        --queries eval/agent_loop_ab/smoke.jsonl \
        --mode stub --run-name smoke

The smoke run takes a few hundred ms and exercises every output path
without touching FAISS / Ollama / Claude / Postgres.

## Side-effect contract

The harness **never**:

* publishes a Redis job,
* writes to the ragmeta / ragchunks tables,
* sends a job-status callback to core-api,
* edits the Spring repo,
* mutates production trace artifacts (the AGENT_TRACE bytes the
  in-process runners produce live in memory only).

Operators are safe to run it against the same WorkerSettings as the
production worker as long as `--mode=registry` only reads from the
existing FAISS index.
