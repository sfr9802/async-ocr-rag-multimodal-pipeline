"""FAISS index wrapper.

Uses `IndexFlatIP` over L2-normalized vectors so inner-product search is
equivalent to cosine similarity. Phase 2 scale is small (thousands of
chunks at most for the included dataset), so an exact flat index is fine
and keeps the mental model simple: no training, no retraining, no tuning.

Index files live on disk alongside a small JSON sidecar describing the
embedding model and dimension used to build them, so a retriever loading
an index can refuse to serve if its configured embedder doesn't match.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss  # type: ignore
import numpy as np

log = logging.getLogger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "build.json"


@dataclass(frozen=True)
class IndexBuildInfo:
    index_version: str
    embedding_model: str
    dimension: int
    chunk_count: int


class FaissIndex:
    """Thin, explicit wrapper around a single IndexFlatIP on disk."""

    def __init__(self, index_dir: Path) -> None:
        self._dir = Path(index_dir)
        self._index: Optional[faiss.Index] = None
        self._info: Optional[IndexBuildInfo] = None

    # ---- build -----------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,
        *,
        index_version: str,
        embedding_model: str,
    ) -> IndexBuildInfo:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        n, d = vectors.shape
        index = faiss.IndexFlatIP(d)
        if n > 0:
            index.add(vectors)

        self._dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._dir / _INDEX_FILE))

        info = IndexBuildInfo(
            index_version=index_version,
            embedding_model=embedding_model,
            dimension=d,
            chunk_count=n,
        )
        (self._dir / _META_FILE).write_text(
            json.dumps(
                {
                    "index_version": info.index_version,
                    "embedding_model": info.embedding_model,
                    "dimension": info.dimension,
                    "chunk_count": info.chunk_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._index = index
        self._info = info
        log.info(
            "FAISS index built at %s (vectors=%d, dim=%d, version=%s)",
            self._dir, n, d, index_version,
        )
        return info

    # ---- load ------------------------------------------------------

    def load(self) -> IndexBuildInfo:
        index_path = self._dir / _INDEX_FILE
        meta_path = self._dir / _META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._dir}. "
                f"Run scripts/build_rag_index.py first."
            )
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self._info = IndexBuildInfo(
            index_version=raw["index_version"],
            embedding_model=raw["embedding_model"],
            dimension=int(raw["dimension"]),
            chunk_count=int(raw["chunk_count"]),
        )
        log.info(
            "Loaded FAISS index %s (model=%s, dim=%d, vectors=%d)",
            self._info.index_version,
            self._info.embedding_model,
            self._info.dimension,
            self._info.chunk_count,
        )
        return self._info

    # ---- query -----------------------------------------------------

    @property
    def info(self) -> IndexBuildInfo:
        if self._info is None:
            raise RuntimeError("FAISS index is not loaded")
        return self._info

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[Tuple[int, float]]]:
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded")
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32, copy=False)
        if self._index.ntotal == 0:
            return [[] for _ in range(query_vectors.shape[0])]
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_vectors, k)
        results: List[List[Tuple[int, float]]] = []
        for row_scores, row_ids in zip(scores, ids):
            row: list[tuple[int, float]] = []
            for row_id, score in zip(row_ids, row_scores):
                if row_id < 0:
                    continue
                row.append((int(row_id), float(score)))
            results.append(row)
        return results
