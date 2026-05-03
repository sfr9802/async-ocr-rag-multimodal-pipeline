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
from app.capabilities.rag.generation import ExtractiveGenerator, RetrievedChunk
from app.capabilities.rag.metadata_store import ChunkLookupResult
from app.capabilities.rag.retriever import RetrievalReport, Retriever


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


def _build_in_memory_rag(tmp_path: Path, audit_store=None) -> RagCapability:
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
        audit_store=audit_store,
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


def test_retrieval_payload_includes_search_unit_citation_contract():
    report = RetrievalReport(
        query="diagram",
        top_k=1,
        index_version="test-v1",
        embedding_model="hashing",
        reranker_name="noop",
        results=[
            RetrievedChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                section="architecture",
                text="architecture diagram caption",
                score=0.88,
                search_unit_id="unit-image-1",
                source_file_id="source-file-1",
                source_file_name="design.pdf",
                extracted_artifact_id="artifact-1",
                artifact_type="IMAGE_CAPTION_JSON",
                unit_type="IMAGE",
                unit_key="image:fig-7",
                title="architecture diagram",
                section_path="Architecture",
                page_start=3,
                page_end=3,
                metadata_json={"bbox": [1, 2, 3, 4]},
            )
        ],
    )

    body = json.loads(RagCapability._retrieval_payload(report))
    hit = body["results"][0]

    assert hit["searchUnitId"] == "unit-image-1"
    assert hit["unitType"] == "IMAGE"
    assert hit["pageStart"] == 3
    assert hit["sectionPath"] == "Architecture"
    assert hit["artifactType"] == "IMAGE_CAPTION_JSON"
    assert hit["citation"]["searchUnitId"] == "unit-image-1"
    assert hit["citation"]["unitId"] == "unit-image-1"
    assert hit["citation"]["unitType"] == "IMAGE"
    assert hit["citation"]["unitKey"] == "image:fig-7"
    assert hit["citation"]["title"] == "architecture diagram"
    assert hit["citation"]["imageId"] == "fig-7"
    assert hit["citation"]["bbox"] == [1, 2, 3, 4]
    assert hit["citation"]["artifactId"] == "artifact-1"
    assert hit["citation"]["artifactType"] == "IMAGE_CAPTION_JSON"
    assert hit["grounding"]["hasCitation"] is True
    assert hit["grounding"]["hasSearchUnitId"] is True
    assert hit["grounding"]["hasPageRange"] is True


def test_rag_capability_records_retrieval_audit_best_effort(tmp_path):
    class _AuditStore:
        def __init__(self) -> None:
            self.calls = []

        def record_retrieval(self, report, *, request_id, user_id):
            self.calls.append((report, request_id, user_id))

    audit = _AuditStore()
    cap = _build_in_memory_rag(tmp_path, audit_store=audit)
    result = cap.run(_make_input("who runs the bookshop?"))

    assert len(result.outputs) == 2
    assert len(audit.calls) == 1
    assert audit.calls[0][1] == "job-rag-test"
    assert audit.calls[0][0].results


def test_rag_capability_ignores_audit_store_failure(tmp_path):
    class _FailingAuditStore:
        def record_retrieval(self, report, *, request_id, user_id):
            raise RuntimeError("audit unavailable")

    cap = _build_in_memory_rag(tmp_path, audit_store=_FailingAuditStore())
    result = cap.run(_make_input("who runs the bookshop?"))

    assert {a.type for a in result.outputs} == {"RETRIEVAL_RESULT", "FINAL_RESPONSE"}


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
