"""Tests for HashingImageEmbedder (deterministic test-only embedder)."""

import numpy as np

from app.capabilities.rag.image_embeddings import HashingImageEmbedder


class TestHashingImageEmbedder:
    def test_dimension(self):
        e = HashingImageEmbedder(dim=64)
        assert e.dimension == 64

    def test_model_name(self):
        e = HashingImageEmbedder(dim=64)
        assert "64" in e.model_name

    def test_encode_images_shape(self):
        e = HashingImageEmbedder(dim=128)
        images = [b"fake-png-bytes-1", b"fake-png-bytes-2"]
        out = e.encode_images(images)
        assert out.shape == (2, 128)
        assert out.dtype == np.float32

    def test_encode_texts_shape(self):
        e = HashingImageEmbedder(dim=128)
        texts = ["a cat sitting on a mat", "diagram of a pipeline"]
        out = e.encode_texts(texts)
        assert out.shape == (2, 128)
        assert out.dtype == np.float32

    def test_encode_images_empty(self):
        e = HashingImageEmbedder(dim=64)
        out = e.encode_images([])
        assert out.shape == (0, 64)

    def test_encode_texts_empty(self):
        e = HashingImageEmbedder(dim=64)
        out = e.encode_texts([])
        assert out.shape == (0, 64)

    def test_vectors_are_l2_normalized(self):
        e = HashingImageEmbedder(dim=128)
        images = [b"some image data"]
        out = e.encode_images(images)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_texts_are_l2_normalized(self):
        e = HashingImageEmbedder(dim=128)
        out = e.encode_texts(["hello world"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self):
        e = HashingImageEmbedder(dim=64)
        a = e.encode_images([b"same-bytes"])
        b = e.encode_images([b"same-bytes"])
        np.testing.assert_array_equal(a, b)
