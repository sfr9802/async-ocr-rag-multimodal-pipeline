"""Evaluation harness for the ai-worker capabilities.

Kept intentionally separate from the serving path (`app/`) so that

  - the production pipeline never imports anything from here,
  - eval code can evolve freely without threading changes through
    worker contracts, and
  - one developer can run a local iteration loop without needing
    experiment-tracking infra, model servers, or external services.

Layout:

    eval/
    ├── __init__.py              (this file)
    ├── README.md                eval guide: what/when/how
    ├── run_eval.py              CLI entry point: `python -m eval.run_eval ...`
    ├── datasets/                small committed JSONL fixtures
    ├── reports/                 generated reports (gitignored)
    └── harness/
        ├── metrics.py           pure-Python CER/WER/hit@k/MRR/keyword coverage
        ├── io_utils.py          JSONL loading + JSON/CSV report writers
        ├── rag_eval.py          text RAG harness (pluggable retriever+generator)
        └── ocr_eval.py          OCR harness (pluggable provider/capability)

Naming note: `eval` is a Python builtin *function*, not a *module*, so
using it as a package name does not shadow anything — the builtin
remains accessible as `builtins.eval`.
"""
