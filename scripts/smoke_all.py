"""Top-level entry point for the full-capability smoke runner.

Delegates to `ai-worker/scripts/smoke_runner.py`, which holds the real
implementation. This wrapper exists so developers running from the repo
root can invoke the smoke runner without `cd ai-worker/` first:

    python scripts/smoke_all.py
    python scripts/smoke_all.py --only MOCK,RAG
    python scripts/smoke_all.py --report smoke-report.json

All CLI flags forward verbatim. See docs/local-run.md "Pipeline closeout
checklist" for the full workflow.

Kept deliberately thin — the real logic (HTTP orchestration, shape
assertions, report building) lives in the worker package so the
ai-worker tests can import it directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    worker_dir = here / "ai-worker"
    if not worker_dir.is_dir():
        print(
            f"ERROR: ai-worker directory not found at {worker_dir}. "
            "Run from the repo root.",
            file=sys.stderr,
        )
        return 2

    # Add ai-worker to sys.path so `scripts.smoke_runner` (which imports
    # `app.*`) resolves the same way `python -m scripts.smoke_runner`
    # does when invoked from inside ai-worker/.
    sys.path.insert(0, str(worker_dir))
    # chdir so relative default paths (eval/datasets/samples/...) resolve.
    os.chdir(worker_dir)

    from scripts.smoke_runner import main as runner_main

    return runner_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
