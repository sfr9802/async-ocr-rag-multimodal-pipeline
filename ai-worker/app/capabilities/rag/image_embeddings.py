"""ImageEmbedder contract + CLIP implementation.

Maps both images and text into the same vector space so that text queries
can retrieve visually relevant images from a FAISS index.

The CLIP provider uses ``sentence-transformers/clip-ViT-B-32`` (512-dim)
via the sentence-transformers library, which can encode both PIL Images
and plain strings with the same ``model.encode()`` call.

A deterministic hashing fallback lives alongside the real implementation
so unit tests can exercise the retriever plumbing without downloading a
600 MB model.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


class ImageEmbedder(ABC):
    """Produces L2-normalized float32 vectors for images and text queries."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def encode_images(self, images: List[bytes]) -> np.ndarray:
        """Encode raw image bytes into L2-normalized vectors."""
        ...

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text queries into the same vector space as images."""
        ...


class ClipImageEmbedder(ImageEmbedder):
    """Real CLIP embedder backed by sentence-transformers.

    Loads the model lazily on first call. Both ``encode_images`` and
    ``encode_texts`` produce vectors in the same 512-dim space so that
    cosine similarity between a text query and an image embedding is
    meaningful.
    """

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32") -> None:
        self._model_name = model_name
        self._model = None  # lazy
        self._dim: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_images(self, images: List[bytes]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        from PIL import Image

        pil_images = [Image.open(BytesIO(b)).convert("RGB") for b in images]
        assert self._model is not None
        vectors = self._model.encode(
            pil_images,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        log.info("Loading CLIP model: %s", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        log.info("CLIP model ready (dim=%d)", self._dim)


class HashingImageEmbedder(ImageEmbedder):
    """Deterministic test-only embedder.

    Produces L2-normalized vectors via content hashing. Images are hashed
    from their raw bytes; texts are hashed from tokens — same approach as
    ``HashingEmbedder`` in embeddings.py but adapted for the dual-encoder
    (image + text) contract.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hashing-image-embedder-dim{self._dim}"

    def encode_images(self, images: List[bytes]) -> np.ndarray:
        out = np.zeros((len(images), self._dim), dtype=np.float32)
        for i, img_bytes in enumerate(images):
            v = np.zeros(self._dim, dtype=np.float32)
            # Hash bytes in 64-byte segments to fill the vector
            for seg in range(0, self._dim, 64):
                chunk_size = min(64, self._dim - seg)
                h = hashlib.blake2b(
                    img_bytes, digest_size=chunk_size, key=seg.to_bytes(2, "little")
                ).digest()
                arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
                v[seg : seg + chunk_size] = arr - 128.0
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            out[i] = v
        return out

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = [t for t in text.lower().split() if t]
            if not tokens:
                continue
            v = np.zeros(self._dim, dtype=np.float32)
            for tok in tokens:
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if (h[4] & 1) else -1.0
                v[idx] += sign
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            out[i] = v
        return out
