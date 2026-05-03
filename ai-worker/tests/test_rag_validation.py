"""Strict model/index validation tests.

These tests cover the three hard requirements of the bge-m3 migration:

  1. Retriever.ensure_ready() refuses to serve when the runtime embedder's
     model name differs from the build.json embedding_model (even when
     dimensions happen to agree).
  2. Retriever.ensure_ready() still fails on dimension mismatch (as a
     belt-and-suspenders guard against a tampered build.json).
  3. When the RAG capability fails to initialize for any reason, the
     MOCK capability is still registered and the worker boots.

All tests run fully offline — no sentence-transformers download, no
Postgres connection, no FAISS rebuild beyond a tiny in-memory index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pytest

from app.capabilities.rag.embedding_text_builder import (
    EMBEDDING_TEXT_BUILDER_VERSION,
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_TITLE_SECTION,
)
from app.capabilities.rag.embeddings import EmbeddingProvider, HashingEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Tiny test doubles so we don't need Postgres or a real sentence-transformer.
# ---------------------------------------------------------------------------


class _RenamingEmbedder(EmbeddingProvider):
    """Wraps another embedder but reports a custom model_name.

    Only used by tests that need to force a (model_name, dimension) tuple
    the HashingEmbedder wouldn't naturally produce on its own.
    """

    def __init__(self, inner: EmbeddingProvider, *, name: str) -> None:
        self._inner = inner
        self._name = name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_name(self) -> str:
        return self._name

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return self._inner.embed_passages(texts)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._inner.embed_queries(texts)


class _FakeMetadataStore:
    """Minimal stand-in so ensure_ready() doesn't need Postgres."""

    def __init__(self, index_version: str) -> None:
        self._version = index_version

    def lookup_chunks_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ChunkLookupResult]:  # pragma: no cover — ensure_ready doesn't call this
        assert index_version == self._version
        return []


def _build_index_on_disk(
    tmp_path: Path,
    *,
    recorded_model_name: str,
    dim: int,
    index_version: str = "test-v1",
) -> FaissIndex:
    """Build a minimal FAISS index whose build.json records the given
    model name and dimension, regardless of whatever embedder actually
    produced the vectors."""
    embedder = HashingEmbedder(dim=dim)
    vectors = embedder.embed_passages(["seed passage one", "seed passage two"])
    assert vectors.shape == (2, dim)

    index = FaissIndex(tmp_path / "idx")
    index.build(
        vectors,
        index_version=index_version,
        embedding_model=recorded_model_name,
    )
    # Return a fresh FaissIndex pointing at the same dir so the caller
    # exercises the load() path instead of reusing the in-memory handle.
    return FaissIndex(tmp_path / "idx")


def _write_manifest(
    tmp_path: Path,
    *,
    variant: str = VARIANT_RETRIEVAL_TITLE_SECTION,
    model: str = "hashing-embedder-dim64",
    dim: int = 64,
    chunks: int = 2,
    max_seq_length: int | None = None,
) -> None:
    (tmp_path / "idx" / "ingest_manifest.json").write_text(
        """{
          "embedding_text_variant": "%s",
          "embedding_text_builder_version": "%s",
          "embedding_model": "%s",
          "max_seq_length": %s,
          "chunk_count": %d,
          "document_count": 1,
          "dimension": %d,
          "index_version": "test-v1",
          "corpus_path": "fixture.jsonl",
          "embed_text_sha256": "%s",
          "embed_text_samples": []
        }"""
        % (
            variant,
            EMBEDDING_TEXT_BUILDER_VERSION,
            model,
            "null" if max_seq_length is None else str(max_seq_length),
            chunks,
            dim,
            "0" * 64,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Model name mismatch must fail hard.
# ---------------------------------------------------------------------------


def test_ensure_ready_rejects_model_name_mismatch(tmp_path):
    """build.json says 'model-A', runtime embedder reports 'model-B' →
    strict RuntimeError, not a silent warning."""
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="model-A", dim=64
    )
    retriever = Retriever(
        embedder=_RenamingEmbedder(HashingEmbedder(dim=64), name="model-B"),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
    )

    with pytest.raises(RuntimeError) as exc_info:
        retriever.ensure_ready()

    message = str(exc_info.value)
    assert "MODEL mismatch" in message
    assert "model-A" in message
    assert "model-B" in message
    # Operational guidance must be surfaced so ops know how to recover.
    assert "rebuild" in message.lower() or "build_rag_index" in message.lower()


def test_ensure_ready_rejects_model_name_mismatch_even_with_same_dim(tmp_path):
    """Same dimension, different model → still a hard failure.

    This is the exact silent-corruption scenario the strict check exists
    to prevent: two different embedding models can emit the same vector
    dimension and produce plausible-looking nearest neighbours while
    quietly destroying retrieval quality.
    """
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="sentence-transformers/all-MiniLM-L6-v2", dim=128
    )
    retriever = Retriever(
        embedder=_RenamingEmbedder(HashingEmbedder(dim=128), name="BAAI/bge-m3"),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
    )

    with pytest.raises(RuntimeError, match="MODEL mismatch"):
        retriever.ensure_ready()


# ---------------------------------------------------------------------------
# 2. Dimension mismatch still fails even when model names agree.
# ---------------------------------------------------------------------------


def test_ensure_ready_rejects_dimension_mismatch(tmp_path):
    """build.json dim != runtime embedder dim → RuntimeError.

    Kept as a belt-and-suspenders guard. In normal operation a matching
    model name implies a matching dimension, but a hand-edited or
    corrupted build.json could trip this path.
    """
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="matched-model", dim=64
    )
    retriever = Retriever(
        embedder=_RenamingEmbedder(HashingEmbedder(dim=128), name="matched-model"),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
    )

    with pytest.raises(RuntimeError) as exc_info:
        retriever.ensure_ready()

    message = str(exc_info.value)
    assert "DIMENSION mismatch" in message
    assert "64" in message
    assert "128" in message


# ---------------------------------------------------------------------------
# 3. MOCK capability survives a failing RAG registration.
# ---------------------------------------------------------------------------


def test_mock_capability_still_registers_when_rag_fails(monkeypatch):
    """If _build_rag_capability raises (e.g. model-mismatch RuntimeError
    from ensure_ready), the registry must still serve MOCK and the
    worker must still boot."""
    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    def _failing_build(_settings):
        raise RuntimeError(
            "simulated RAG init failure (e.g. model-name mismatch)"
        )

    monkeypatch.setattr(
        registry_module, "_build_rag_capability", _failing_build
    )

    settings = WorkerSettings(rag_enabled=True)
    result = registry_module.build_default_registry(settings)

    assert "MOCK" in result.available()
    assert "RAG" not in result.available()


def test_rag_disabled_leaves_mock_only(monkeypatch):
    """Sanity check on the opt-out path: with every real capability
    disabled, the registry never tries to build RAG/OCR/MULTIMODAL,
    AUTO has no sub to dispatch to, and MOCK is the sole capability."""
    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    # If _build_rag_capability were called, this would raise — so reaching
    # the assertions below proves the rag_enabled gate short-circuits it.
    def _should_not_be_called(_settings):
        raise AssertionError(
            "rag_enabled=False must skip RAG capability construction"
        )

    monkeypatch.setattr(
        registry_module, "_build_rag_capability", _should_not_be_called
    )

    settings = WorkerSettings(
        rag_enabled=False,
        ocr_enabled=False,
        multimodal_enabled=False,
        ocr_extract_enabled=False,
        xlsx_extract_enabled=False,
    )
    result = registry_module.build_default_registry(settings)

    assert result.available() == ["MOCK"]


# ---------------------------------------------------------------------------
# 4. Happy path: matching model name + matching dim → ensure_ready() OK.
# ---------------------------------------------------------------------------


def test_ensure_ready_accepts_matching_model_and_dim(tmp_path):
    """Positive control: when everything matches, ensure_ready() returns
    cleanly and the retriever becomes usable."""
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="hashing-embedder-dim64", dim=64
    )
    retriever = Retriever(
        embedder=HashingEmbedder(dim=64),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
    )

    # Must not raise.
    retriever.ensure_ready()


# ---------------------------------------------------------------------------
# 5. Serving path validates ingest_manifest when configured by registry.
# ---------------------------------------------------------------------------


def test_ensure_ready_rejects_missing_manifest_when_variant_expected(tmp_path):
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="hashing-embedder-dim64", dim=64
    )
    retriever = Retriever(
        embedder=HashingEmbedder(dim=64),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
        embedding_text_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )

    with pytest.raises(RuntimeError, match="Ingest manifest missing"):
        retriever.ensure_ready()


def test_ensure_ready_rejects_manifest_variant_mismatch(tmp_path):
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="hashing-embedder-dim64", dim=64
    )
    _write_manifest(tmp_path, variant=VARIANT_TITLE_SECTION)
    retriever = Retriever(
        embedder=HashingEmbedder(dim=64),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
        embedding_text_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )

    with pytest.raises(RuntimeError) as exc_info:
        retriever.ensure_ready()

    assert "Ingest manifest mismatch" in str(exc_info.value)
    assert "variant" in str(exc_info.value)


def test_ensure_ready_accepts_matching_manifest(tmp_path):
    index = _build_index_on_disk(
        tmp_path, recorded_model_name="hashing-embedder-dim64", dim=64
    )
    _write_manifest(tmp_path, max_seq_length=1024)
    retriever = Retriever(
        embedder=HashingEmbedder(dim=64),
        index=index,
        metadata=_FakeMetadataStore("test-v1"),
        top_k=3,
        embedding_text_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
        embedding_max_seq_length=1024,
    )

    retriever.ensure_ready()
