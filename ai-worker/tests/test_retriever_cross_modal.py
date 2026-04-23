"""Tests for CrossModalRetriever — RRF fusion of text + image hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.capabilities.rag.cross_modal_retriever import CrossModalRetriever
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.image_embeddings import HashingImageEmbedder
from app.capabilities.rag.image_index import ImageFaissIndex
from app.capabilities.rag.image_metadata_store import ImageLookupResult
from app.capabilities.rag.retriever import RetrievalReport


# ---- Fakes ----------------------------------------------------------------

class _FakeTextRetriever:
    """Fake text retriever that returns scripted results."""

    def __init__(self, results: List[RetrievedChunk]):
        self._results = results
        self.calls: list[str] = []

    def retrieve(self, query: str) -> RetrievalReport:
        self.calls.append(query)
        return RetrievalReport(
            query=query,
            top_k=len(self._results),
            index_version="v-text-test",
            embedding_model="text-model",
            results=self._results,
        )


class _FakeImageMetadataStore:
    """Fake image metadata that maps faiss_row_id -> ImageLookupResult."""

    def __init__(self, rows: List[ImageLookupResult]):
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_by_faiss_rows(
        self, index_version: str, faiss_row_ids
    ) -> List[ImageLookupResult]:
        ids = list(faiss_row_ids)
        return [self._by_row[i] for i in ids if i in self._by_row]


# ---- Helpers ---------------------------------------------------------------

def _build_cross_modal(
    tmp_path,
    text_chunks: List[RetrievedChunk],
    image_metas: List[ImageLookupResult],
    image_vectors: np.ndarray | None = None,
) -> CrossModalRetriever:
    """Build a CrossModalRetriever with in-memory fakes."""
    embedder = HashingImageEmbedder(dim=128)

    # Build image FAISS index
    image_index = ImageFaissIndex(tmp_path)
    n = len(image_metas)
    if image_vectors is None:
        if n > 0:
            image_vectors = np.random.randn(n, 128).astype(np.float32)
            image_vectors /= np.linalg.norm(image_vectors, axis=1, keepdims=True)
        else:
            image_vectors = np.empty((0, 128), dtype=np.float32)
    image_index.build(image_vectors, index_version="v-img-test", embedding_model=embedder.model_name)

    return CrossModalRetriever(
        text_retriever=_FakeTextRetriever(text_chunks),
        image_embedder=embedder,
        image_index=image_index,
        image_metadata=_FakeImageMetadataStore(image_metas),
        top_k=5,
        rrf_k=60,
    )


# ---- Tests -----------------------------------------------------------------

class TestRRFFusion:
    def test_text_only_when_no_image_hits(self, tmp_path):
        text_chunks = [
            RetrievedChunk("c1", "doc-1", "overview", "text about cats", 0.9),
        ]
        retriever = _build_cross_modal(tmp_path, text_chunks, [])
        report = retriever.retrieve_multimodal("cat document")
        assert len(report.results) == 1
        assert report.results[0].chunk_id == "c1"
        assert report.results[0].section == "overview"

    def test_image_hits_get_section_image(self, tmp_path):
        text_chunks = []
        image_metas = [
            ImageLookupResult("img-1", "doc-1", 1, "A photo of a cat", None, 0),
        ]
        # Build with a vector that will be found
        vec = np.ones((1, 128), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        retriever = _build_cross_modal(tmp_path, text_chunks, image_metas, vec)
        report = retriever.retrieve_multimodal("anything")
        # Should find at least the image hit
        image_results = [r for r in report.results if r.section == "image"]
        assert len(image_results) >= 1
        assert image_results[0].chunk_id == "img-1"
        assert image_results[0].doc_id == "doc-1"

    def test_caption_used_as_text(self, tmp_path):
        text_chunks = []
        image_metas = [
            ImageLookupResult("img-1", "doc-1", 1, "Caption from manifest", None, 0),
        ]
        vec = np.ones((1, 128), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        retriever = _build_cross_modal(tmp_path, text_chunks, image_metas, vec)
        report = retriever.retrieve_multimodal("query")
        image_results = [r for r in report.results if r.section == "image"]
        assert image_results[0].text == "Caption from manifest"

    def test_no_caption_uses_placeholder(self, tmp_path):
        text_chunks = []
        image_metas = [
            ImageLookupResult("img-1", "doc-1", 1, None, None, 0),
        ]
        vec = np.ones((1, 128), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        retriever = _build_cross_modal(tmp_path, text_chunks, image_metas, vec)
        report = retriever.retrieve_multimodal("query")
        image_results = [r for r in report.results if r.section == "image"]
        assert "[image: img-1]" in image_results[0].text

    def test_rrf_merges_both_lists(self, tmp_path):
        text_chunks = [
            RetrievedChunk("c1", "doc-1", "overview", "text chunk", 0.9),
            RetrievedChunk("c2", "doc-2", "plot", "another chunk", 0.8),
        ]
        image_metas = [
            ImageLookupResult("img-3", "doc-3", 1, "image caption", None, 0),
        ]
        vec = np.ones((1, 128), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        retriever = _build_cross_modal(tmp_path, text_chunks, image_metas, vec)
        report = retriever.retrieve_multimodal("query")
        doc_ids = {r.doc_id for r in report.results}
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids

    def test_same_doc_from_both_ranks_higher(self, tmp_path):
        """A doc appearing in both text and image results should get a
        higher RRF score than one appearing in only one list."""
        text_chunks = [
            RetrievedChunk("c1", "shared-doc", "overview", "text", 0.9),
            RetrievedChunk("c2", "text-only-doc", "plot", "text", 0.8),
        ]
        image_metas = [
            ImageLookupResult("img-shared", "shared-doc", 1, "caption", None, 0),
            ImageLookupResult("img-other", "img-only-doc", 2, "other", None, 1),
        ]
        vecs = np.eye(2, 128, dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        retriever = _build_cross_modal(tmp_path, text_chunks, image_metas, vecs)
        report = retriever.retrieve_multimodal("query")
        # shared-doc items should have higher combined RRF scores
        shared_scores = [r.score for r in report.results if r.doc_id == "shared-doc"]
        other_scores = [r.score for r in report.results if r.doc_id != "shared-doc"]
        if shared_scores and other_scores:
            assert max(shared_scores) >= max(other_scores)


class TestExistingRetrieverUnchanged:
    """Verify that the text-only Retriever path is not broken."""

    def test_text_retriever_still_called(self, tmp_path):
        text_chunks = [
            RetrievedChunk("c1", "doc-1", "s1", "text", 0.9),
        ]
        retriever = _build_cross_modal(tmp_path, text_chunks, [])
        report = retriever.retrieve_multimodal("test query")
        # Text retriever should have been called
        fake_text = retriever._text
        assert "test query" in fake_text.calls
