"""Capabilities — the "what the worker actually does" layer.

Each capability implements the same minimal interface and is registered
via `app.capabilities.registry`. Phase 1 ships only MOCK; OCR / RAG /
MULTIMODAL live as placeholder packages so imports don't break when the
real implementations land.
"""
