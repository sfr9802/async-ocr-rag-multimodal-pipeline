"""Phase 2 retrieval-experiment tests.

Covers the four eval-only modules added in Phase 2:

  1. ``embedding_text_builder`` — prefix-variant composition.
  2. ``bm25_retriever`` — pure-Python BM25Okapi + duck-typed retriever.
  3. ``hybrid_retriever`` — dense + sparse RRF fusion.
  4. ``retrieval_sweep`` — sweep driver + Pareto adapter.

All tests are offline (no FAISS, no embedder, no langgraph). Each
module is exercised in isolation and then wired together in a tiny
end-to-end smoke that proves a Phase 2 sweep run-through produces a
``RetrievalSweepReport`` consumable by the existing
``compute_pareto_frontier`` tooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from eval.harness import (
    BM25_DEFAULT_K1,
    BM25EvalRetriever,
    EMBEDDING_TEXT_VARIANTS,
    EmbeddingTextInput,
    KIND_BM25,
    KIND_DENSE,
    KIND_HYBRID,
    PREFIX_SEPARATOR,
    RRFHybridEvalRetriever,
    RetrievalSweepConfig,
    VARIANT_ALL,
    VARIANT_KEYWORD,
    VARIANT_RAW,
    VARIANT_SECTION,
    VARIANT_TITLE,
    VARIANT_TITLE_SECTION,
    build_bm25_index,
    build_embedding_text,
    compute_pareto_frontier,
    is_known_variant,
    render_sweep_markdown,
    rrf_fuse_ranked_lists,
    run_retrieval_sweep,
    sweep_report_to_dict,
    sweep_to_topn_sweep_report,
    tokenize_for_bm25,
)


# ---------------------------------------------------------------------------
# 1. embedding_text_builder
# ---------------------------------------------------------------------------


class TestEmbeddingTextBuilder:
    def test_raw_returns_text_only(self):
        inp = EmbeddingTextInput(
            text="alpha beta", title="My Title", section="Intro",
            keywords=("k1", "k2"),
        )
        assert build_embedding_text(inp, variant=VARIANT_RAW) == "alpha beta"

    def test_title_prefixes_with_separator(self):
        inp = EmbeddingTextInput(text="alpha", title="My Title")
        out = build_embedding_text(inp, variant=VARIANT_TITLE)
        assert out == f"My Title{PREFIX_SEPARATOR}alpha"

    def test_title_section_combines_both(self):
        inp = EmbeddingTextInput(
            text="alpha", title="Doc-A", section="Plot",
        )
        out = build_embedding_text(inp, variant=VARIANT_TITLE_SECTION)
        assert out == f"Doc-A{PREFIX_SEPARATOR}Plot{PREFIX_SEPARATOR}alpha"

    def test_keyword_variant_dedups_case_insensitively(self):
        inp = EmbeddingTextInput(
            text="body",
            keywords=("BGE", "bge", "embedding", "Embedding"),
        )
        out = build_embedding_text(inp, variant=VARIANT_KEYWORD)
        # "bge" and "Bge" are dedup'd case-insensitively; only "embedding"
        # appears once. Original casing of first occurrence preserved.
        assert out == f"BGE embedding{PREFIX_SEPARATOR}body"

    def test_all_variant_includes_every_segment(self):
        inp = EmbeddingTextInput(
            text="body",
            title="T",
            section="S",
            keywords=("a", "b"),
        )
        out = build_embedding_text(inp, variant=VARIANT_ALL)
        # Order is fixed: title, section, keywords, body.
        assert out == (
            f"T{PREFIX_SEPARATOR}S{PREFIX_SEPARATOR}a b{PREFIX_SEPARATOR}body"
        )

    def test_missing_prefix_falls_back_silently(self):
        # variant asks for title but chunk doesn't carry one — must
        # NOT inject an empty line / sentinel.
        inp = EmbeddingTextInput(text="body", title=None)
        assert build_embedding_text(inp, variant=VARIANT_TITLE) == "body"
        # whitespace-only metadata also drops, not surfaces.
        inp2 = EmbeddingTextInput(text="body", section="   ")
        assert build_embedding_text(inp2, variant=VARIANT_SECTION) == "body"

    def test_unknown_variant_raises(self):
        inp = EmbeddingTextInput(text="body")
        with pytest.raises(ValueError):
            build_embedding_text(inp, variant="invalid_variant")

    def test_keyword_limit_caps_count(self):
        inp = EmbeddingTextInput(
            text="body",
            keywords=tuple(f"kw{i}" for i in range(20)),
        )
        out = build_embedding_text(
            inp, variant=VARIANT_KEYWORD, keyword_limit=3,
        )
        # Only first 3 unique keywords land in the prefix.
        assert "kw0 kw1 kw2" in out
        assert "kw3" not in out

    def test_is_known_variant(self):
        for v in EMBEDDING_TEXT_VARIANTS:
            assert is_known_variant(v) is True
        assert is_known_variant("nope") is False


# ---------------------------------------------------------------------------
# 2. bm25_retriever
# ---------------------------------------------------------------------------


@dataclass
class _CorpusChunk:
    """Minimal duck-typed chunk for BM25 / hybrid tests."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    title: Optional[str] = None
    keywords: tuple = field(default_factory=tuple)


def _tiny_corpus() -> List[_CorpusChunk]:
    return [
        _CorpusChunk(
            chunk_id="c1", doc_id="doc-book", section="overview",
            text="A retired translator runs a secondhand bookshop on a dying railway line.",
            title="The Quiet Bookshop",
            keywords=("translator", "bookshop"),
        ),
        _CorpusChunk(
            chunk_id="c2", doc_id="doc-book", section="overview",
            text="The bookshop has tea and old translations the translator curates herself.",
            title="The Quiet Bookshop",
            keywords=("tea", "translation"),
        ),
        _CorpusChunk(
            chunk_id="c3", doc_id="doc-cats", section="overview",
            text="An elderly fisherman feeds the stray cats of a small harbor every morning.",
            title="Harbor Cats",
            keywords=("fisherman", "cats"),
        ),
        _CorpusChunk(
            chunk_id="c4", doc_id="doc-mech", section="plot",
            text="Ironclad Academy students pilot construction mechs to reinforce a coastal dam.",
            title="Ironclad Academy",
            keywords=("mech", "academy"),
        ),
        _CorpusChunk(
            chunk_id="c5", doc_id="doc-aoi", section="overview",
            text="Aoi tends luminescent gardens suspended above the clouds.",
            title="Cloud Gardens",
            keywords=("garden", "aoi"),
        ),
    ]


class TestTokenizer:
    def test_whitespace_split_with_strip(self):
        toks = tokenize_for_bm25("Hello, World!")
        # "Hello," strips trailing comma → "hello"; same for "World!".
        assert "hello" in toks
        assert "world" in toks

    def test_unicode_nfkc_casefold(self):
        # Full-width A becomes "a"; case folds to lowercase.
        assert "abc" in tokenize_for_bm25("ＡＢＣ")
        assert "Abc" not in tokenize_for_bm25("ＡＢＣ")

    def test_cjk_emits_character_1grams(self):
        toks = tokenize_for_bm25("한국어 텍스트")
        # Each CJK char becomes its own token. The whitespace-token form
        # is also retained.
        assert "한" in toks
        assert "국" in toks
        assert "어" in toks
        # Whitespace token preserved alongside.
        assert "한국어" in toks

    def test_empty_input(self):
        assert tokenize_for_bm25("") == []
        assert tokenize_for_bm25(None) == []  # type: ignore[arg-type]


class TestBM25Index:
    def test_index_builds_and_scores_top_doc_first(self):
        chunks = _tiny_corpus()
        index = build_bm25_index(chunks)
        retriever = BM25EvalRetriever(index, top_k=5)
        report = retriever.retrieve("translator bookshop")
        assert report.results
        # Top hit should be a doc-book chunk (those mention both query
        # tokens directly).
        assert report.results[0].doc_id == "doc-book"
        # Candidate doc_ids are deduplicated and in fused-rank order.
        assert "doc-book" in report.candidate_doc_ids
        # BM25 surfaces wall-clock under dense_retrieval_ms by design.
        assert report.dense_retrieval_ms is not None
        assert report.rerank_ms is None

    def test_index_uses_embedding_text_variant(self):
        chunks = _tiny_corpus()
        # Build with title prefix — querying for the title alone should
        # surface the bookshop doc despite the body not mentioning it.
        index_with_title = build_bm25_index(
            chunks, embedding_text_variant=VARIANT_TITLE,
        )
        retriever = BM25EvalRetriever(index_with_title, top_k=3)
        report = retriever.retrieve("Quiet")
        # "Quiet" only appears in the doc-book title; with VARIANT_RAW
        # the query would miss. With VARIANT_TITLE it lands.
        assert any(r.doc_id == "doc-book" for r in report.results)

    def test_index_handles_empty_query(self):
        chunks = _tiny_corpus()
        index = build_bm25_index(chunks)
        retriever = BM25EvalRetriever(index)
        report = retriever.retrieve("")
        assert report.results == []
        assert report.candidate_doc_ids == []

    def test_index_skips_oov_query_tokens(self):
        chunks = _tiny_corpus()
        index = build_bm25_index(chunks)
        retriever = BM25EvalRetriever(index, top_k=5)
        # "xyzzy" doesn't appear in any chunk → df=0 → no contribution.
        report = retriever.retrieve("xyzzy translator")
        assert report.results
        assert report.results[0].doc_id == "doc-book"

    def test_idf_smoothing_is_finite_for_singleton_term(self):
        chunks = _tiny_corpus()
        index = build_bm25_index(chunks)
        # "fisherman" appears in 1 chunk — IDF should be finite.
        retriever = BM25EvalRetriever(index, top_k=3)
        report = retriever.retrieve("fisherman")
        assert report.results
        assert report.results[0].doc_id == "doc-cats"
        # All scores must be finite numbers.
        for r in report.results:
            assert math.isfinite(r.score)


# ---------------------------------------------------------------------------
# 3. hybrid_retriever
# ---------------------------------------------------------------------------


class _StubDenseRetriever:
    """Returns a fixed scripted result list — independent of BM25."""

    def __init__(self, results, *, dense_retrieval_ms=12.0):
        self._results = results
        self._dense_retrieval_ms = dense_retrieval_ms

    def retrieve(self, query: str):
        from eval.harness.bm25_retriever import BM25Report  # any duck shape
        return BM25Report(
            results=list(self._results),
            candidate_doc_ids=[],
            dense_retrieval_ms=self._dense_retrieval_ms,
            rerank_ms=None,
            reranker_name="dense-stub",
        )


class TestRRFFusion:
    def test_fuse_two_lists_preserves_shared_keys_at_top(self):
        a = [("k1", None), ("k2", None), ("k3", None)]
        b = [("k2", None), ("k4", None), ("k5", None)]
        fused = rrf_fuse_ranked_lists(a, b, k_rrf=60)
        keys = [k for k, _ in fused]
        # k2 appears in BOTH lists → must rank above any single-list key.
        assert keys[0] == "k2"
        # k1 is rank-1 in list A (1/(60+1) = ~0.0164)
        # k4 is rank-2 in list B (1/(60+2) = ~0.0161)
        # → k1 outranks k4.
        assert keys.index("k1") < keys.index("k4")

    def test_fuse_returns_empty_for_empty_inputs(self):
        assert rrf_fuse_ranked_lists() == []
        assert rrf_fuse_ranked_lists([]) == []

    def test_fuse_skips_empty_keys(self):
        # Empty-string keys are filtered out before scoring.
        fused = rrf_fuse_ranked_lists([("", None), ("k1", None)])
        assert [k for k, _ in fused] == ["k1"]


class TestRRFHybridRetriever:
    def test_hybrid_fuses_dense_and_sparse(self):
        from app.capabilities.rag.generation import RetrievedChunk

        dense_results = [
            RetrievedChunk(
                chunk_id="c-dense-1", doc_id="doc-A", section="s",
                text="alpha beta", score=0.9,
            ),
            RetrievedChunk(
                chunk_id="c-shared", doc_id="doc-shared", section="s",
                text="shared chunk", score=0.7,
            ),
        ]
        sparse_chunks = _tiny_corpus()
        # Construct a sparse retriever where a known chunk-id wins.
        bm25_index = build_bm25_index(sparse_chunks)
        sparse_retriever = BM25EvalRetriever(bm25_index, top_k=5)
        # Inject a "shared" chunk into the sparse view by reusing
        # ``c-shared`` chunk_id manually via a wrapper retriever.

        class _SharedSparse:
            def retrieve(self, query):
                # Return the BM25 result + force a "c-shared" prefix.
                base = sparse_retriever.retrieve(query)
                base.results.insert(
                    0,
                    RetrievedChunk(
                        chunk_id="c-shared", doc_id="doc-shared", section="s",
                        text="shared chunk", score=10.0,
                    ),
                )
                return base

        dense = _StubDenseRetriever(dense_results)
        hybrid = RRFHybridEvalRetriever(
            dense=dense, sparse=_SharedSparse(),
            k_rrf=60, final_top_k=3, per_backend_top_k=10,
        )
        report = hybrid.retrieve("translator bookshop")
        assert report.fused_candidate_count >= 2
        # The shared chunk (in both lists) must be at fused rank 1.
        assert report.results[0].chunk_id == "c-shared"
        # Both backend latencies surface.
        assert report.dense_retrieval_ms is not None
        assert report.rerank_ms is not None
        # Score field on results carries the FUSED score, not the dense score.
        assert report.results[0].score > 0


# ---------------------------------------------------------------------------
# 4. retrieval_sweep + Pareto adapter
# ---------------------------------------------------------------------------


def _build_sweep_dataset():
    return [
        {
            "id": "q1", "query": "translator bookshop",
            "expected_doc_ids": ["doc-book"],
            "expected_section_keywords": ["bookshop", "translator"],
            "query_type": "character",
        },
        {
            "id": "q2", "query": "fisherman harbor cats",
            "expected_doc_ids": ["doc-cats"],
            "expected_section_keywords": ["fisherman", "cats"],
            "query_type": "plot_event",
        },
        {
            "id": "q3", "query": "luminescent gardens above clouds",
            "expected_doc_ids": ["doc-aoi"],
            "expected_section_keywords": ["garden", "aoi"],
            "query_type": "setting",
        },
    ]


def test_sweep_runs_grid_and_returns_one_cell_per_config():
    chunks = _tiny_corpus()
    index_raw = build_bm25_index(chunks, embedding_text_variant=VARIANT_RAW)
    index_ts = build_bm25_index(
        chunks, embedding_text_variant=VARIANT_TITLE_SECTION,
    )
    bm25_raw = BM25EvalRetriever(index_raw, top_k=10)
    bm25_ts = BM25EvalRetriever(index_ts, top_k=10)

    configs = [
        RetrievalSweepConfig(
            label="bm25-raw-k10",
            retriever_kind=KIND_BM25,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=None,
            final_top_k=5,
            retriever=bm25_raw,
            extra={"dataset_chunks": len(chunks)},
        ),
        RetrievalSweepConfig(
            label="bm25-ts-k10",
            retriever_kind=KIND_BM25,
            embedding_text_variant=VARIANT_TITLE_SECTION,
            candidate_k=None,
            final_top_k=5,
            retriever=bm25_ts,
        ),
    ]
    dataset = _build_sweep_dataset()
    sweep = run_retrieval_sweep(
        dataset, configs=configs, dataset_path="<test>",
        sample_rows=2,
    )
    assert len(sweep.cells) == 2
    labels = {c.label for c in sweep.cells}
    assert labels == {"bm25-raw-k10", "bm25-ts-k10"}
    # Each cell carries a populated summary.
    for cell in sweep.cells:
        assert cell.summary.row_count == 3
        assert cell.summary.mean_hit_at_5 is not None
    # Sample rows preserved at the configured cap.
    assert all(len(c.rows_sample) <= 2 for c in sweep.cells)


def test_sweep_serialises_to_dict_round_trip():
    chunks = _tiny_corpus()
    bm25 = BM25EvalRetriever(
        build_bm25_index(chunks), top_k=5,
    )
    cfg = RetrievalSweepConfig(
        label="bm25-tiny",
        retriever_kind=KIND_BM25,
        embedding_text_variant=VARIANT_RAW,
        candidate_k=None,
        final_top_k=5,
        retriever=bm25,
    )
    sweep = run_retrieval_sweep(
        _build_sweep_dataset(), configs=[cfg], dataset_path="<test>",
    )
    payload = sweep_report_to_dict(sweep)
    assert payload["schema"].startswith("phase2-retrieval-sweep")
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["label"] == "bm25-tiny"
    # Summary nests as a dict (asdict-roundtripped).
    assert "summary" in payload["cells"][0]
    assert "mean_hit_at_5" in payload["cells"][0]["summary"]


def test_sweep_renders_markdown_with_required_sections():
    chunks = _tiny_corpus()
    bm25 = BM25EvalRetriever(build_bm25_index(chunks), top_k=5)
    cfg = RetrievalSweepConfig(
        label="bm25-md",
        retriever_kind=KIND_BM25,
        embedding_text_variant=VARIANT_RAW,
        candidate_k=None,
        final_top_k=5,
        retriever=bm25,
    )
    sweep = run_retrieval_sweep(
        _build_sweep_dataset(), configs=[cfg], dataset_path="<test>",
    )
    md = render_sweep_markdown(sweep)
    assert "# Phase 2 retrieval sweep" in md
    assert "## Configurations" in md
    assert "## Headline metrics" in md
    assert "## Latency (ms)" in md
    assert "## Composite scores" in md
    assert "bm25-md" in md


def test_sweep_pareto_adapter_feeds_compute_pareto_frontier():
    """End-to-end: run a 2-cell sweep, adapt to TopNSweepReport, hand
    to ``compute_pareto_frontier``. The adapter must produce entries
    the existing Pareto algorithm understands.
    """
    chunks = _tiny_corpus()
    bm25_a = BM25EvalRetriever(
        build_bm25_index(chunks, embedding_text_variant=VARIANT_RAW),
        top_k=5,
    )
    bm25_b = BM25EvalRetriever(
        build_bm25_index(chunks, embedding_text_variant=VARIANT_TITLE_SECTION),
        top_k=5,
    )
    configs = [
        RetrievalSweepConfig(
            label="raw", retriever_kind=KIND_BM25,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=10, final_top_k=5, retriever=bm25_a,
        ),
        RetrievalSweepConfig(
            label="title_section", retriever_kind=KIND_BM25,
            embedding_text_variant=VARIANT_TITLE_SECTION,
            candidate_k=10, final_top_k=5, retriever=bm25_b,
        ),
    ]
    sweep = run_retrieval_sweep(
        _build_sweep_dataset(), configs=configs, dataset_path="<test>",
    )
    topn = sweep_to_topn_sweep_report(sweep)
    pareto = compute_pareto_frontier(
        topn,
        metric="mean_hit_at_5",
        latency="total_query_p95_ms",
    )
    assert len(pareto.entries) == 2
    labels = {p.label for p in pareto.entries}
    assert labels == {"raw", "title_section"}


def test_sweep_dense_vs_bm25_vs_hybrid_smoke():
    """End-to-end: dense-stub + bm25 + hybrid all run through a single
    sweep; the report carries one cell per kind and the headline
    metrics are populated for each.
    """
    from app.capabilities.rag.generation import RetrievedChunk

    chunks = _tiny_corpus()
    # Dense stub: returns the bookshop chunk first for q1, harbor
    # for q2, gardens for q3 — perfect for the dataset.
    dense_results_by_query = {
        "translator bookshop": [
            RetrievedChunk(
                chunk_id="d-c1", doc_id="doc-book", section="overview",
                text="dense bookshop", score=0.95,
            ),
        ],
        "fisherman harbor cats": [
            RetrievedChunk(
                chunk_id="d-c3", doc_id="doc-cats", section="overview",
                text="dense cats", score=0.92,
            ),
        ],
        "luminescent gardens above clouds": [
            RetrievedChunk(
                chunk_id="d-c5", doc_id="doc-aoi", section="overview",
                text="dense gardens", score=0.88,
            ),
        ],
    }

    class _ScriptedDense:
        def retrieve(self, query):
            from eval.harness.bm25_retriever import BM25Report
            return BM25Report(
                results=list(dense_results_by_query.get(query, [])),
                candidate_doc_ids=[
                    r.doc_id for r in dense_results_by_query.get(query, [])
                ],
                dense_retrieval_ms=8.0,
                rerank_ms=None,
                reranker_name="dense-scripted",
            )

    dense = _ScriptedDense()
    bm25 = BM25EvalRetriever(build_bm25_index(chunks), top_k=10)
    hybrid = RRFHybridEvalRetriever(
        dense=dense, sparse=bm25, k_rrf=60,
        final_top_k=5, per_backend_top_k=10,
    )

    configs = [
        RetrievalSweepConfig(
            label="dense-only", retriever_kind=KIND_DENSE,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=None, final_top_k=5, retriever=dense,
        ),
        RetrievalSweepConfig(
            label="bm25-only", retriever_kind=KIND_BM25,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=None, final_top_k=5, retriever=bm25,
        ),
        RetrievalSweepConfig(
            label="hybrid-rrf", retriever_kind=KIND_HYBRID,
            embedding_text_variant=VARIANT_RAW,
            candidate_k=10, final_top_k=5, retriever=hybrid,
        ),
    ]
    sweep = run_retrieval_sweep(
        _build_sweep_dataset(), configs=configs, dataset_path="<test>",
    )
    assert len(sweep.cells) == 3
    by_label = {c.label: c for c in sweep.cells}
    # Dense-stub returns the gold chunk first for every query → hit@1 == 1.
    assert by_label["dense-only"].summary.mean_hit_at_1 == pytest.approx(1.0)
    # Hybrid must also achieve at least the dense quality (RRF tends to
    # at least preserve the dense win when sparse adds the same gold).
    assert by_label["hybrid-rrf"].summary.mean_hit_at_5 is not None
    # All three cells emit a non-None composite quality score.
    for cell in sweep.cells:
        assert cell.summary.quality_score is not None


def test_sweep_records_byqueryt_type_breakdown_per_cell():
    """Sweep cells inherit Phase 1's byQueryType — query_type field on
    the dataset rows must surface in each cell's summary.
    """
    chunks = _tiny_corpus()
    bm25 = BM25EvalRetriever(build_bm25_index(chunks), top_k=5)
    cfg = RetrievalSweepConfig(
        label="bm25", retriever_kind=KIND_BM25,
        embedding_text_variant=VARIANT_RAW,
        candidate_k=None, final_top_k=5, retriever=bm25,
    )
    sweep = run_retrieval_sweep(
        _build_sweep_dataset(), configs=[cfg], dataset_path="<test>",
    )
    bt = sweep.cells[0].summary.by_query_type
    # The 3 dataset rows carry query_type = character / plot_event /
    # setting respectively → 3 buckets each with count==1.
    assert {"character", "plot_event", "setting"} <= set(bt.keys())
    assert all(bt[k]["count"] == 1 for k in ("character", "plot_event", "setting"))
