"""Multi-query RRF tests for the Retriever.

Three scenario groups, all fully offline:

  1. Empty rewrites (NoOp / Regex parser in phase 3) -> single-query
     path only, no RRF pass, behaviour bit-for-bit identical to the
     pre-parser Retriever. This is the "env unset is a no-op" contract
     the registry default depends on.

  2. A manually crafted parser emitting 2 rewrites forces 3 FAISS
     searches (one per rewrite + the normalized primary). The
     candidate pool the reranker sees is the RRF-merged fusion of the
     three per-query lists; reranking still composes on top.

  3. The ``parsed_query`` field of RetrievalReport round-trips through
     JSON (to_dict -> json.dumps) so the RETRIEVAL_RESULT artifact
     stays valid JSON when the parser is active.

Drives the real ``Retriever`` against a fake FAISS index + fake
metadata store, mirroring the shape of ``test_rag_mmr.py`` /
``test_rag_reranker.py`` so the RRF path exercises the same
integration seams as the reranker + MMR paths.
"""

from __future__ import annotations

import json
from typing import Dict, List

from app.capabilities.rag.embeddings import HashingEmbedder
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)
from app.capabilities.rag.reranker import NoOpReranker, RerankerProvider
from app.capabilities.rag.retriever import Retriever, _rrf_merge


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFaissIndex:
    """Returns per-query hit lists drawn from ``hits_by_query``.

    Every call to ``search`` is logged into ``search_queries_seen`` so
    tests can assert how many queries the Retriever actually embedded +
    searched. Falls back to a default descending hit list when a query
    isn't mapped, so simple single-query tests don't need to specify
    every query variant explicitly.
    """

    def __init__(
        self,
        info,
        chunk_count: int,
        hits_by_embedding: Dict[tuple, List[tuple]] | None = None,
    ) -> None:
        self._info = info
        self._chunk_count = chunk_count
        self._hits_by_embedding = hits_by_embedding or {}
        self.search_call_count = 0

    def load(self):
        return self._info

    def search(self, query_vectors, top_k: int):
        self.search_call_count += 1
        # Default: descending row_ids with decreasing scores.
        default = [(i, 1.0 - 0.01 * i) for i in range(min(int(top_k), self._chunk_count))]
        key = tuple(query_vectors.tolist()[0]) if len(query_vectors) else ()
        hits = self._hits_by_embedding.get(key, default)
        return [hits[:int(top_k)]]


class _StubEmbedder(HashingEmbedder):
    """HashingEmbedder with a recorder so tests can assert which text
    strings were embedded for retrieval."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.queries_seen: List[str] = []

    def embed_queries(self, texts):  # type: ignore[override]
        self.queries_seen.extend(texts)
        return super().embed_queries(texts)


class _FakeStore:
    def __init__(self, rows) -> None:
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(self, index_version, ids):
        return [self._by_row[i] for i in ids if i in self._by_row]


class _StaticParser(QueryParserProvider):
    """Parser that returns a fixed ParsedQuery regardless of input.

    Lets a test force the multi-query path without implementing an
    actual LLM-backed parser — exactly the seam phase 4 will replace.
    """

    def __init__(self, parsed: ParsedQuery) -> None:
        self._parsed = parsed

    @property
    def name(self) -> str:
        return "static-test"

    def parse(self, query: str) -> ParsedQuery:
        # Swap the original in so RetrievalReport.query stays the raw text.
        return ParsedQuery(
            original=query,
            normalized=self._parsed.normalized,
            keywords=list(self._parsed.keywords),
            intent=self._parsed.intent,
            rewrites=list(self._parsed.rewrites),
            filters=dict(self._parsed.filters),
            parser_name=self._parsed.parser_name,
        )


def _fake_info(chunk_count: int, model_name: str, dim: int):
    from app.capabilities.rag.faiss_index import IndexBuildInfo

    return IndexBuildInfo(
        index_version="rrf-test-v1",
        embedding_model=model_name,
        dimension=dim,
        chunk_count=chunk_count,
    )


def _build_rows(n: int):
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
    rerank_score: float | None = None,
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
# 1. Empty rewrites -> single-query path, no RRF.
# ---------------------------------------------------------------------------


def test_no_rewrites_runs_single_faiss_search():
    """NoOpQueryParser (default) -> no rewrites -> exactly one FAISS
    search. This is the "env unset is a no-op" acceptance test."""
    embedder = _StubEmbedder(dim=32)
    info = _fake_info(5, embedder.model_name, embedder.dimension)
    index = _FakeFaissIndex(info, chunk_count=5)
    store = _FakeStore(_build_rows(5))

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=NoOpReranker(),
        candidate_k=5,
        query_parser=NoOpQueryParser(),
    )
    retriever.ensure_ready()
    report = retriever.retrieve("what is vector search")

    assert index.search_call_count == 1
    # Single embed call -- ``embed_queries`` is called once with one
    # element, not once per rewrite.
    assert len(embedder.queries_seen) == 1
    # ParsedQuery is surfaced on the report even when it's the no-op,
    # so downstream artifacts always get a consistent shape.
    assert report.parsed_query is not None
    assert report.parsed_query.parser_name == "noop"
    assert report.parsed_query.rewrites == []


def test_regex_parser_also_skips_rrf_because_rewrites_empty():
    """RegexQueryParser always returns rewrites=[] in phase 3 — the
    Retriever must treat that the same as NoOp for search-call-count
    purposes."""
    embedder = _StubEmbedder(dim=32)
    info = _fake_info(5, embedder.model_name, embedder.dimension)
    index = _FakeFaissIndex(info, chunk_count=5)
    store = _FakeStore(_build_rows(5))

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=NoOpReranker(),
        candidate_k=5,
        query_parser=RegexQueryParser(),
    )
    retriever.ensure_ready()
    report = retriever.retrieve("  RAG reranker 성능  ")

    assert index.search_call_count == 1
    # Regex normalized query is what the embedder saw (whitespace
    # collapsed), not the raw padded form.
    assert embedder.queries_seen == ["RAG reranker 성능"]
    # Keywords are surfaced on the report.
    assert report.parsed_query is not None
    assert "reranker" in report.parsed_query.keywords
    assert report.parsed_query.rewrites == []


# ---------------------------------------------------------------------------
# 2. Manually crafted parsed query with 2 rewrites -> 3 searches, RRF.
# ---------------------------------------------------------------------------


def test_rewrites_trigger_three_faiss_searches_and_rrf_merges_results():
    """Parser emits normalized + 2 rewrites -> Retriever runs exactly
    3 FAISS searches. The reranker is NoOp so the final list reflects
    the RRF merge order, not a reranker reshuffle."""
    embedder = _StubEmbedder(dim=32)
    info = _fake_info(6, embedder.model_name, embedder.dimension)

    # Three query variants -> three distinct hit lists. RRF should
    # surface row 2 near the top because it's in the top-3 of every
    # list, even though no single list has it at rank 1.
    per_query_hits: Dict[tuple, List[tuple]] = {}
    variants = [
        "normalized text",
        "rewrite one alpha",
        "rewrite two beta",
    ]
    # Pre-compute embeddings to key the fake index by.
    for i, text in enumerate(variants):
        vec = embedder._embed([text])
        key = tuple(vec.tolist()[0])
        if i == 0:
            hits = [(0, 0.9), (2, 0.8), (4, 0.7), (5, 0.6)]
        elif i == 1:
            hits = [(1, 0.95), (2, 0.85), (3, 0.75), (5, 0.65)]
        else:
            hits = [(3, 0.92), (2, 0.82), (1, 0.72), (4, 0.62)]
        per_query_hits[key] = hits

    index = _FakeFaissIndex(info, chunk_count=6, hits_by_embedding=per_query_hits)
    store = _FakeStore(_build_rows(6))

    parsed = ParsedQuery(
        original="normalized text",
        normalized="normalized text",
        keywords=["normalized", "text"],
        intent="other",
        rewrites=["rewrite one alpha", "rewrite two beta"],
        filters={},
        parser_name="static-test",
    )

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=4,
        reranker=NoOpReranker(),
        candidate_k=6,
        query_parser=_StaticParser(parsed),
        multi_query_rrf_k=60,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("any user query")

    # Exactly 3 FAISS searches — one per variant.
    assert index.search_call_count == 3
    # Each variant was embedded exactly once.
    assert sorted(embedder.queries_seen) == sorted(variants)

    # Chunk c2 is the RRF winner because it appears at rank 2 in EVERY
    # variant, while no other chunk appears in all three. Full math at
    # k_rrf=60:
    #   c0: 1/61                       ≈ 0.01639  (variant 0 rank 1)
    #   c1: 1/61 + 1/63                ≈ 0.03226  (v1 r1, v2 r3)
    #   c2: 1/62 + 1/62 + 1/62 = 3/62  ≈ 0.04839  (rank 2 everywhere)
    #   c3: 1/61 + 1/63                ≈ 0.03226  (v1 r3, v2 r1)
    #   c4: 1/63 + 1/64                ≈ 0.03149  (v0 r3, v2 r4)
    #   c5: 1/64 + 1/64                ≈ 0.03125  (v0 r4, v1 r4)
    # Descending: c2 > c1 == c3 > c4 > c5 > c0. c1/c3 tie is broken by
    # dict-insertion order (c1 inserted during variant 1 iter, c3 during
    # variant 2) -> c1 before c3 in the final list.
    result_ids = [c.chunk_id for c in report.results]
    assert result_ids[0] == "c2"  # appears in all three variants -> wins
    # The tied c1/c3 pair must both appear in the top-4 pool.
    assert "c1" in result_ids
    assert "c3" in result_ids
    # Top-4 must be drawn from the six candidates that any variant saw.
    assert set(result_ids).issubset({f"c{i}" for i in range(6)})

    # Parsed query is surfaced on the report.
    assert report.parsed_query is not None
    assert report.parsed_query.rewrites == [
        "rewrite one alpha", "rewrite two beta",
    ]


def test_rrf_composes_with_reranker():
    """With a reranker in front, the RRF merge supplies the candidate
    pool and the reranker reorders on its own score signal. The seam
    is the same one MMR uses — reranker gets a pool, emits top-k."""

    class _ScriptedReranker(RerankerProvider):
        @property
        def name(self) -> str:
            return "scripted"

        def rerank(self, query, chunks, k):
            # Force a stable order: c5 > c4 > c3 > c2 > c1 > c0, ignoring
            # whatever the RRF merge actually produced. The Retriever is
            # expected to trust the reranker.
            by_id = {c.chunk_id: c for c in chunks}
            desired = ["c5", "c4", "c3", "c2", "c1", "c0"]
            return [by_id[cid] for cid in desired if cid in by_id][:k]

    embedder = _StubEmbedder(dim=32)
    info = _fake_info(6, embedder.model_name, embedder.dimension)
    index = _FakeFaissIndex(info, chunk_count=6)
    store = _FakeStore(_build_rows(6))

    parsed = ParsedQuery(
        original="q", normalized="q", keywords=["q"],
        intent="other", rewrites=["r1", "r2"],
        filters={}, parser_name="static-test",
    )

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=_ScriptedReranker(),
        candidate_k=6,
        query_parser=_StaticParser(parsed),
    )
    retriever.ensure_ready()
    report = retriever.retrieve("whatever")

    # 3 searches triggered by the 2 rewrites + normalized.
    assert index.search_call_count == 3
    # Reranker-imposed order wins over RRF order.
    assert [c.chunk_id for c in report.results] == ["c5", "c4", "c3"]


# ---------------------------------------------------------------------------
# 3. parsed_query is JSON-serializable on the report.
# ---------------------------------------------------------------------------


def test_retrieval_report_parsed_query_round_trips_through_json():
    """RetrievalReport.parsed_query.to_dict() must survive json.dumps.
    The capability layer already goes through this path — we pin it
    here so a future ParsedQuery field with a non-serializable type
    fails at CI time, not in prod."""
    embedder = _StubEmbedder(dim=32)
    info = _fake_info(4, embedder.model_name, embedder.dimension)
    index = _FakeFaissIndex(info, chunk_count=4)
    store = _FakeStore(_build_rows(4))

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=2,
        reranker=NoOpReranker(),
        candidate_k=4,
        query_parser=RegexQueryParser(),
    )
    retriever.ensure_ready()
    report = retriever.retrieve("reranker vs cross-encoder 비교")

    assert report.parsed_query is not None
    rendered = json.dumps(
        report.parsed_query.to_dict(), ensure_ascii=False
    )
    round_tripped = json.loads(rendered)
    assert round_tripped["parserName"] == "regex"
    assert "reranker" in round_tripped["keywords"]
    assert round_tripped["intent"] == "other"


# ---------------------------------------------------------------------------
# 4. _rrf_merge direct unit sanity.
# ---------------------------------------------------------------------------


def test_rrf_merge_empty_input_returns_empty():
    assert _rrf_merge([], k_rrf=60, pool_size=5) == []
    assert _rrf_merge([[]], k_rrf=60, pool_size=5) == []


def test_rrf_merge_ranks_shared_chunks_higher_than_singletons():
    """A chunk appearing in every list beats chunks that only show up
    once, even when the singletons ranked higher in their own list."""
    shared = _chunk("shared", "doc-A", score=0.5)
    only_a = _chunk("only_a", "doc-B", score=0.9)
    only_b = _chunk("only_b", "doc-C", score=0.9)
    only_c = _chunk("only_c", "doc-D", score=0.9)

    merged = _rrf_merge(
        [
            [only_a, shared],
            [only_b, shared],
            [only_c, shared],
        ],
        k_rrf=60,
        pool_size=4,
    )

    # 'shared' accumulates three 1/(60+2) contributions = 3/62 ~= 0.0484,
    # each 'only_X' gets a single 1/(60+1) = 1/61 ~= 0.0164.
    assert merged[0].chunk_id == "shared"
    shared_score = merged[0].score
    singleton_scores = [c.score for c in merged if c.chunk_id != "shared"]
    assert all(shared_score > s for s in singleton_scores)


def test_rrf_merge_deduplicates_by_search_unit_id_and_preserves_metadata():
    first = RetrievedChunk(
        chunk_id="legacy-a",
        doc_id="doc-a",
        section="overview",
        text="same unit first",
        score=0.5,
        search_unit_id="unit-1",
        source_file_id="source-1",
        unit_type="PAGE",
        unit_key="page:1",
        page_start=1,
        page_end=1,
    )
    second = RetrievedChunk(
        chunk_id="legacy-b",
        doc_id="doc-a",
        section="overview",
        text="same unit second",
        score=0.4,
        search_unit_id="unit-1",
        source_file_id="source-1",
        unit_type="PAGE",
        unit_key="page:1",
        page_start=1,
        page_end=1,
    )

    merged = _rrf_merge([[first], [second]], k_rrf=60, pool_size=5)

    assert len(merged) == 1
    assert merged[0].chunk_id == "legacy-a"
    assert merged[0].search_unit_id == "unit-1"
    assert merged[0].unit_type == "PAGE"
    assert merged[0].page_start == 1
    assert merged[0].score == (1 / 61) + (1 / 61)


def test_rrf_merge_respects_pool_size_cap():
    chunks = [_chunk(f"c{i}", f"doc-{i}", score=0.5) for i in range(10)]
    merged = _rrf_merge([chunks], k_rrf=60, pool_size=3)
    assert len(merged) == 3
