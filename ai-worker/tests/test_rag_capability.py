"""Unit test for the RAG capability using the deterministic hashing
embedder + an in-memory fake metadata store so that the test can run
without touching a real model or Postgres.

This intentionally does NOT exercise the sentence-transformers embedder
(that path is covered by the full E2E smoke test with core-api + worker).
The goal here is to catch wiring bugs between the retriever, the FAISS
index, and the generation provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from app.capabilities.base import CapabilityInput, CapabilityInputArtifact
from app.capabilities.rag.capability import RagCapability, RagCapabilityConfig
from app.capabilities.rag.embeddings import HashingEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.generation import ExtractiveGenerator
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.retriever import Retriever


class _FakeMetadataStore:
    """In-memory stand-in for RagMetadataStore.

    The RAG capability talks to the metadata store through a very narrow
    interface — ping() (not called here) and lookup_chunks_by_faiss_rows.
    We build the store from a list of (chunk_id, doc_id, section, text)
    tuples and let FAISS own the row ids.
    """

    def __init__(self, index_version: str, rows: List[ChunkLookupResult]) -> None:
        self._version = index_version
        self._rows = rows
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ChunkLookupResult]:
        assert index_version == self._version, "version mismatch in fake store"
        return [self._by_row[i] for i in faiss_row_ids if i in self._by_row]


def _build_in_memory_rag(tmp_path: Path) -> RagCapability:
    passages = [
        ("chunk-1", "doc-aoi",  "overview",  "Aoi Tsukishiro is a new arrival who tends luminescent gardens suspended above the clouds."),
        ("chunk-2", "doc-aoi",  "characters","Grandmother Rin is the village's oldest gardener, stern about pruning."),
        ("chunk-3", "doc-mech", "plot",      "Ironclad Academy students pilot construction mechs to reinforce a coastal dam before a typhoon."),
        ("chunk-4", "doc-book", "overview",  "A retired translator runs a secondhand bookshop at the last station on a dying railway line."),
        ("chunk-5", "doc-cats", "overview",  "An elderly fisherman feeds the stray cats of a small harbor every morning without fail."),
    ]

    embedder = HashingEmbedder(dim=64)
    texts = [p[3] for p in passages]
    vectors = embedder.embed_passages(texts)

    index = FaissIndex(tmp_path / "idx")
    index.build(vectors, index_version="test-v1", embedding_model=embedder.model_name)

    rows = [
        ChunkLookupResult(
            chunk_id=p[0],
            doc_id=p[1],
            section=p[2],
            text=p[3],
            faiss_row_id=i,
        )
        for i, p in enumerate(passages)
    ]
    metadata = _FakeMetadataStore(index_version="test-v1", rows=rows)

    retriever = Retriever(embedder=embedder, index=index, metadata=metadata, top_k=3)
    retriever.ensure_ready()

    return RagCapability(
        retriever=retriever,
        generator=ExtractiveGenerator(),
        config=RagCapabilityConfig(top_k=3),
    )


def _make_input(query: str) -> CapabilityInput:
    return CapabilityInput(
        job_id="job-rag-test",
        capability="RAG",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="art-query",
                type="INPUT_TEXT",
                content=query.encode("utf-8"),
                content_type="text/plain",
            )
        ],
    )


def test_rag_capability_emits_retrieval_and_answer_artifacts(tmp_path):
    cap = _build_in_memory_rag(tmp_path)
    result = cap.run(_make_input("who runs the bookshop?"))

    assert len(result.outputs) == 2
    types = {a.type for a in result.outputs}
    assert types == {"RETRIEVAL_RESULT", "FINAL_RESPONSE"}

    retrieval = next(a for a in result.outputs if a.type == "RETRIEVAL_RESULT")
    body = json.loads(retrieval.content.decode("utf-8"))
    assert body["query"] == "who runs the bookshop?"
    assert body["topK"] == 3
    assert body["indexVersion"] == "test-v1"
    assert body["hitCount"] >= 1
    # At least one of the hits should mention the bookshop chunk.
    texts = " ".join(r["text"] for r in body["results"]).lower()
    assert "bookshop" in texts or "translator" in texts

    final = next(a for a in result.outputs if a.type == "FINAL_RESPONSE")
    answer = final.content.decode("utf-8")
    # The generator is grounded: the answer must cite at least one doc id
    # from the retrieval step.
    doc_ids_in_answer = [d for d in ("doc-aoi", "doc-mech", "doc-book", "doc-cats") if d in answer]
    assert doc_ids_in_answer, f"answer should cite retrieved doc ids: {answer}"


def test_rag_capability_rejects_empty_input(tmp_path):
    from app.capabilities.base import CapabilityError

    cap = _build_in_memory_rag(tmp_path)
    empty_input = CapabilityInput(
        job_id="job-empty",
        capability="RAG",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="art-empty",
                type="INPUT_TEXT",
                content=b"   ",
                content_type="text/plain",
            )
        ],
    )
    try:
        cap.run(empty_input)
    except CapabilityError as ex:
        assert ex.code == "EMPTY_QUERY"
        return
    raise AssertionError("expected a CapabilityError on empty query")
