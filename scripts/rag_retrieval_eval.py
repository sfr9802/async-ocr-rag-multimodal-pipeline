"""CLI wrapper for the xlsx/pdf RAG ingestion retrieval eval harness."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AI_WORKER = ROOT / "ai-worker"
if str(AI_WORKER) not in sys.path:
    sys.path.insert(0, str(AI_WORKER))

from eval.harness.rag_ingestion_retrieval_eval import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
