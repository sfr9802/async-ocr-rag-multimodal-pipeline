"""MMR (Maximal Marginal Relevance) diversity-pass tests.

The MMR selector composes after the reranker: the reranker produces a
candidate list ordered by relevance, and MMR then picks the final top-k
by penalising candidates that share a doc_id with an already-selected
chunk. Four scenarios, all fully offline:

  1. use_mmr=False -> reranker's top-k is returned unchanged. This is
     the Phase 1 reproducibility guarantee (env unset must be a no-op).
  2. use_mmr=True with rank 1..4 all sharing doc_id 'A': rank 1 is
     preserved, but rank 2+ are pulled from other docs because the
     doc_id penalty dominates the lower-relevance deltas. Verifies the
     core "stop crowding k with the same doc" behaviour.
  3. use_mmr=True with mmr_lambda=1.0 degenerates to relevance-only —
     the penalty term vanishes entirely and the final top-k matches the
     no-MMR ordering bit-for-bit.
  4. RetrievalReport surfaces ``use_mmr``, ``mmr_lambda``, and
     ``dup_rate``, and the ``dup_rate`` value matches what the eval
     harness metric computes from the same doc_id list.

The tests drive the real ``Retriever`` against a fake FAISS index +
fake metadata store, mirroring the pattern already used in
``test_rag_reranker.py`` so the MMR path exercises the same integration
seams as the reranker path.
"""

from __future__ import annotations

from typing import List, Sequence

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import NoOpReranker, RerankerProvider
from app.capabilities.rag.retriever import Retriever, _mmr_select
from eval.harness.metrics import dup_rate as harness_dup_rate


# ---------------------------------------------------------------------------
# Fakes shared by all tests
# ---------------------------------------------------------------------------


class _StaticReranker(RerankerProvider):
    """Reranker that returns a pre-baked list of chunks with rerank_scores.

    The test case owns the relevance order — which lets us construct
    pathological "all top chunks are the same doc" scenarios and check
    that MMR actually pulls diversity out of the tail.
    """

    def __init__(self, ranked: List[RetrievedChunk]) -> None:
        self._ranked = ranked

    @property
    def name(self) -> str:
        return "static-test"

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        k = max(0, int(k))
        return list(self._ranked[:k])


class _FakeFaissIndex:
    def __init__(self, info, chunk_count: int) -> None:
        self._info = info
        self._chunk_count = chunk_count
        self.last_search_k: int = -1

    def load(self):
        return self._info

    def search(self, query_vectors, top_k: int):
        self.last_search_k = int(top_k)
        k = min(int(top_k), self._chunk_count)
        return [[(i, 1.0 - 0.01 * i) for i in range(k)]]


class _FakeStore:
    def __init__(self, rows) -> None:
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(self, index_version, ids):
        return [self._by_row[i] for i in ids if i in self._by_row]


def _fake_info(chunk_count: int, model_name: str, dim: int):
    from app.capabilities.rag.faiss_index import IndexBuildInfo

    return IndexBuildInfo(
        index_version="mmr-test-v1",
        embedding_model=model_name,
        dimension=dim,
        chunk_count=chunk_count,
    )


def _build_rows(n: int):
    """Build n ChunkLookupResults with unique doc_ids (doc-0 .. doc-{n-1}).

    Actual doc_id assignment is overridden per test via ``doc_ids`` —
    this helper exists only so the fake store can resolve row_ids back
    to _something_ and the Retriever's candidate-construction loop has
    stable inputs.
    """
    from app.capabilities.rag.metadata_store import ChunkLookupResult

    return [
        ChunkLookupResult(
            chunk_id=f"c{i}",
            doc_id=f"doc-{i}",
            section="overview",
            text=f"passage {i}",
            faiss_row_id=i,
        )
        for i in range(n)
    ]


def _chunk(
    chunk_id: str,
    doc_id: str,
    *,
    score: float,
    rerank_score: float | None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section="overview",
        text=f"text for {chunk_id}",
        score=score,
        rerank_score=rerank_score,
    )


# ---------------------------------------------------------------------------
# 1. use_mmr=False preserves the reranker's ordering exactly.
# ---------------------------------------------------------------------------


def test_mmr_off_preserves_reranker_order():
    """use_mmr=False + env-unset-equivalent: the retriever must return
    the reranker's top-k in its original order. This is the "Phase 1
    byte-for-byte" contract the spec calls out."""
    from app.capabilities.rag.embeddings import HashingEmbedder

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=5,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info, chunk_count=5)
    store = _FakeStore(_build_rows(5))

    # The reranker returns chunks in relevance order. A doc_id
    # collision on rank 1/2 exists; with MMR OFF the retriever must
    # NOT unshuffle it.
    ranked = [
        _chunk("r1", "doc-A", score=0.9, rerank_score=9.0),
        _chunk("r2", "doc-A", score=0.8, rerank_score=8.0),
        _chunk("r3", "doc-B", score=0.7, rerank_score=7.0),
        _chunk("r4", "doc-C", score=0.6, rerank_score=6.0),
    ]
    reranker = _StaticReranker(ranked)

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=reranker,
        candidate_k=4,
        use_mmr=False,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("q")

    assert [c.chunk_id for c in report.results] == ["r1", "r2", "r3"]
    assert report.use_mmr is False
    assert report.mmr_lambda is None  # off path surfaces None


# ---------------------------------------------------------------------------
# 2. use_mmr=True pulls diversity out of a doc-crowded candidate list.
# ---------------------------------------------------------------------------


def test_mmr_on_preserves_rank1_and_pulls_other_docs_into_k():
    """Ranks 1..4 all share doc_id 'doc-A'. With MMR on, rank 1 still
    wins (no selected set yet), but ranks 2 and 3 must come from other
    docs because the 0.6 doc_id penalty outweighs the small relevance
    gap to the lower-ranked rank-5/6 candidates."""
    from app.capabilities.rag.embeddings import HashingEmbedder

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=6,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info, chunk_count=6)
    store = _FakeStore(_build_rows(6))

    # Rank 1..4 all share doc-A; rank 5 is doc-B, rank 6 is doc-C.
    # Math at lambda=0.7, penalty=0.6 after r1 is selected:
    #   value(r2 doc-A, 0.60) = 0.7*0.60 - 0.3*0.6 = 0.240
    #   value(r5 doc-B, 0.55) = 0.7*0.55           = 0.385  <- wins
    #   value(r6 doc-C, 0.53) = 0.7*0.53           = 0.371
    # After r5 is selected, r6 still beats every doc-A tail chunk:
    #   value(r2 doc-A, 0.60) = 0.240  (still penalised, doc-A in set)
    #   value(r6 doc-C, 0.53) = 0.371  <- wins
    # So top-3 = r1 (doc-A), r5 (doc-B), r6 (doc-C).
    ranked = [
        _chunk("r1", "doc-A", score=0.95, rerank_score=0.95),
        _chunk("r2", "doc-A", score=0.60, rerank_score=0.60),
        _chunk("r3", "doc-A", score=0.58, rerank_score=0.58),
        _chunk("r4", "doc-A", score=0.56, rerank_score=0.56),
        _chunk("r5", "doc-B", score=0.55, rerank_score=0.55),
        _chunk("r6", "doc-C", score=0.53, rerank_score=0.53),
    ]
    reranker = _StaticReranker(ranked)

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=reranker,
        candidate_k=6,
        use_mmr=True,
        mmr_lambda=0.7,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("q")

    ids = [c.chunk_id for c in report.results]
    doc_ids = [c.doc_id for c in report.results]

    assert len(ids) == 3
    # Rank 1 preserved.
    assert ids[0] == "r1"
    # Ranks 2 and 3 must come from other docs, not another doc-A chunk.
    assert doc_ids[1] != "doc-A"
    assert doc_ids[2] != "doc-A"
    # Specifically: the two other-doc candidates (r5, r6) should both
    # be selected, since every doc-A chunk incurs the full penalty
    # once doc-A is in the selected set.
    assert set(ids[1:]) == {"r5", "r6"}

    # And the report surfaces the knobs.
    assert report.use_mmr is True
    assert report.mmr_lambda == 0.7


# ---------------------------------------------------------------------------
# 3. lambda = 1.0 degenerates to relevance-only ordering.
# ---------------------------------------------------------------------------


def test_mmr_lambda_one_degenerates_to_relevance_only():
    """At lambda=1.0 the penalty term is multiplied by zero; the MMR
    selector picks strictly by relevance and the result must equal the
    reranker's top-k even when doc_id collisions exist."""
    from app.capabilities.rag.embeddings import HashingEmbedder

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=6,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info, chunk_count=6)
    store = _FakeStore(_build_rows(6))

    ranked = [
        _chunk("r1", "doc-A", score=0.9, rerank_score=0.90),
        _chunk("r2", "doc-A", score=0.8, rerank_score=0.80),
        _chunk("r3", "doc-A", score=0.7, rerank_score=0.70),
        _chunk("r4", "doc-B", score=0.6, rerank_score=0.60),
        _chunk("r5", "doc-C", score=0.5, rerank_score=0.50),
        _chunk("r6", "doc-D", score=0.4, rerank_score=0.40),
    ]
    reranker = _StaticReranker(ranked)

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=reranker,
        candidate_k=6,
        use_mmr=True,
        mmr_lambda=1.0,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("q")

    assert [c.chunk_id for c in report.results] == ["r1", "r2", "r3"]
    assert report.use_mmr is True
    assert report.mmr_lambda == 1.0


# ---------------------------------------------------------------------------
# 4. RetrievalReport.dup_rate matches the harness metric.
# ---------------------------------------------------------------------------


def test_retrieval_report_dup_rate_matches_computed_value():
    """The dup_rate field on RetrievalReport must equal what the eval
    harness computes from the same list of doc_ids. This is the single
    source of truth the rag_eval harness now reads — drift here means
    the eval summary would silently disagree with the retriever."""
    from app.capabilities.rag.embeddings import HashingEmbedder

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=5,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info, chunk_count=5)
    store = _FakeStore(_build_rows(5))

    # Force a duplicate-heavy top-k: 3 out of 4 share doc-A.
    # Expected dup_rate = 1 - (2 / 4) = 0.5
    ranked = [
        _chunk("r1", "doc-A", score=0.9, rerank_score=0.9),
        _chunk("r2", "doc-A", score=0.8, rerank_score=0.8),
        _chunk("r3", "doc-A", score=0.7, rerank_score=0.7),
        _chunk("r4", "doc-B", score=0.6, rerank_score=0.6),
    ]
    reranker = _StaticReranker(ranked)

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=4,
        reranker=reranker,
        candidate_k=4,
        use_mmr=False,  # keep ordering so the dup_rate expectation is fixed
    )
    retriever.ensure_ready()
    report = retriever.retrieve("q")

    doc_ids = [c.doc_id for c in report.results]
    expected = round(harness_dup_rate(doc_ids), 4)

    assert report.dup_rate == expected
    assert report.dup_rate == 0.5


# ---------------------------------------------------------------------------
# 5. _mmr_select unit-level sanity: empty input + k=0 short-circuits.
# ---------------------------------------------------------------------------


def test_mmr_select_empty_and_zero_k():
    assert _mmr_select([], top_k=3, mmr_lambda=0.7, doc_id_penalty=0.6) == []

    chunks = [
        _chunk("r1", "doc-A", score=0.9, rerank_score=0.9),
        _chunk("r2", "doc-B", score=0.8, rerank_score=0.8),
    ]
    assert _mmr_select(chunks, top_k=0, mmr_lambda=0.7, doc_id_penalty=0.6) == []


def test_mmr_select_falls_back_to_bi_encoder_score_when_rerank_missing():
    """When the reranker is NoOp, candidates carry no rerank_score.
    _mmr_select must use the bi-encoder ``score`` as the relevance
    signal so NoOp + MMR is still a meaningful combination.

    Math at lambda=0.7, penalty=0.6:
      After picking r1, value(r2 doc-A) = 0.7*0.55 - 0.3*0.6 = 0.205
                     value(r3 doc-B) = 0.7*0.50           = 0.350
      So r3 wins rank 2 — the penalty dominates r2's small relevance
      edge (0.55 vs 0.50).
    """
    chunks = [
        _chunk("r1", "doc-A", score=0.90, rerank_score=None),
        _chunk("r2", "doc-A", score=0.55, rerank_score=None),
        _chunk("r3", "doc-B", score=0.50, rerank_score=None),
    ]
    selected = _mmr_select(
        chunks, top_k=2, mmr_lambda=0.7, doc_id_penalty=0.6
    )
    ids = [c.chunk_id for c in selected]
    assert ids == ["r1", "r3"]
