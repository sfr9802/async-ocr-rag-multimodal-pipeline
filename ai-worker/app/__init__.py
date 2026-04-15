"""ai-worker package.

Long-lived worker that:
  1. Consumes job dispatch messages from a Redis list
  2. Claims the job via the core-api internal endpoint
  3. Executes the matching capability
  4. Uploads result artifacts
  5. Posts a callback to core-api

Phase 1 only ships the MOCK capability — OCR / RAG / multimodal land in later phases.
"""

__version__ = "0.1.0"
