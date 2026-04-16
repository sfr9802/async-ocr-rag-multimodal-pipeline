"""Tests for ImageFaissIndex — build, persist, load, search round-trip."""

import json

import numpy as np

from app.capabilities.rag.image_index import ImageFaissIndex


class TestImageFaissIndex:
    def test_build_and_load(self, tmp_path):
        idx = ImageFaissIndex(tmp_path)
        vectors = np.random.randn(5, 64).astype(np.float32)
        # L2 normalize
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        info = idx.build(vectors, index_version="v-test", embedding_model="clip-test")
        assert info.chunk_count == 5
        assert info.dimension == 64

        # Load in a fresh instance
        idx2 = ImageFaissIndex(tmp_path)
        loaded = idx2.load()
        assert loaded.index_version == "v-test"
        assert loaded.embedding_model == "clip-test"
        assert loaded.chunk_count == 5
        assert loaded.dimension == 64

    def test_search_returns_hits(self, tmp_path):
        idx = ImageFaissIndex(tmp_path)
        vectors = np.eye(4, dtype=np.float32)
        idx.build(vectors, index_version="v-test", embedding_model="test")

        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = idx.search(query, top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2
        # Row 0 should be the best match (exact vector)
        row_id, score = results[0][0]
        assert row_id == 0
        assert score > 0.99

    def test_empty_index(self, tmp_path):
        idx = ImageFaissIndex(tmp_path)
        vectors = np.empty((0, 32), dtype=np.float32)
        idx.build(vectors, index_version="v-empty", embedding_model="test")

        query = np.random.randn(1, 32).astype(np.float32)
        results = idx.search(query, top_k=3)
        assert results == [[]]

    def test_build_json_sidecar(self, tmp_path):
        idx = ImageFaissIndex(tmp_path)
        vectors = np.random.randn(3, 16).astype(np.float32)
        idx.build(vectors, index_version="v-sidecar", embedding_model="clip-model")

        build_json = tmp_path / "image" / "build.json"
        assert build_json.exists()
        meta = json.loads(build_json.read_text())
        assert meta["index_version"] == "v-sidecar"
        assert meta["embedding_model"] == "clip-model"
        assert meta["dimension"] == 16
        assert meta["chunk_count"] == 3
