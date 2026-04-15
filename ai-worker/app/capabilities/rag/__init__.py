"""RAG capability.

Phase 2 implementation: text query → FAISS retrieval → grounded extractive
answer. Metadata lives in PostgreSQL (ragmeta schema), vectors live in a
single FAISS index file, embeddings come from sentence-transformers (or an
optional deterministic fallback), and generation uses retrieved chunks to
build a cited answer — no mock output.

Entry point: `RagCapability` in `capability.py`.
"""

from app.capabilities.rag.capability import RagCapability

__all__ = ["RagCapability"]
