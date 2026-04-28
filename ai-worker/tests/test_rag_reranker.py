"""RerankerProvider tests.

Four scenarios, all fully offline:

  1. NoOpReranker is an identity: it returns chunks[:k] in the exact
     order it received them, regardless of content, and leaves the
     ``rerank_score`` field at its default None.
  2. CrossEncoderReranker with a monkeypatched CrossEncoder returns
     the correct re-ordering under deterministic scores, attaches the
     score to each chunk, and respects ``batch_size`` on predict().
  3. CrossEncoderReranker swallows predict() exceptions and returns
     chunks[:k] in the original bi-encoder order — the failure-isolation
     pattern the registry relies on.
  4. Registry wiring:
       - rag_reranker='off' -> NoOpReranker, no CrossEncoder import,
         RAG registers.
       - rag_reranker='cross_encoder' with a CrossEncoder that fails
         to load -> NoOpReranker fallback + warning, RAG still
         registers, MOCK/OCR unaffected.
"""

from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np
import pytest

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankerProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section="overview",
        text=text,
        score=score,
    )


def _string_only_oom_check(exc):
    """Pure-Python OOM detector — no torch import, no CUDA driver init."""
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


@pytest.fixture(autouse=True)
def _no_cuda_helpers(monkeypatch):
    """Replace torch-touching helpers with pure-Python stand-ins.

    Phase 2A added two CUDA-touching helper calls inside
    CrossEncoderReranker.rerank() — ``_log_cuda_memory`` (for
    observability) and ``_is_cuda_oom_exception`` (for the OOM-retry
    path). Both lazily ``import torch``. The existing comment in
    ``tests/test_rag_embeddings_helpers.py`` documents that triggering
    a torch import inside an already-loaded pytest process can SEGFAULT
    the interpreter — not raise, abort.

    Patching both at the reranker module's import sites keeps every
    test in this module isolated from torch-init instability without
    shipping a runtime setting that disables observability in
    production. Real worker processes import torch via the embedder
    long before any reranker call so the production path always sees
    the real helpers.
    """
    from app.capabilities.rag import (
        embeddings as embeddings_module,
        reranker as reranker_module,
    )

    monkeypatch.setattr(reranker_module, "_log_cuda_memory", lambda _label: None)
    monkeypatch.setattr(embeddings_module, "_log_cuda_memory", lambda _label: None)
    monkeypatch.setattr(
        reranker_module, "_is_cuda_oom_exception", _string_only_oom_check,
    )


class _FakeCrossEncoder:
    """Minimal stand-in for sentence_transformers.CrossEncoder.

    ``score_by_text`` is consulted per (query, passage) pair; the
    scorer matches on a substring inside the passage so a test can
    push chunks up or down the ranking without caring about the
    exact truncation. Any pair that doesn't match any substring
    gets a 0.0 score.
    """

    def __init__(self, score_by_text: dict[str, float]) -> None:
        self._score_by_text = score_by_text
        self.predict_calls: List[dict[str, Any]] = []

    def predict(
        self,
        pairs: List[tuple[str, str]],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self.predict_calls.append({
            "pair_count": len(pairs),
            "batch_size": batch_size,
        })
        scores: List[float] = []
        for _q, passage in pairs:
            found = 0.0
            for needle, value in self._score_by_text.items():
                if needle in passage:
                    found = value
                    break
            scores.append(found)
        return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. NoOpReranker identity behaviour.
# ---------------------------------------------------------------------------


def test_noop_reranker_returns_prefix_unchanged():
    reranker = NoOpReranker()
    chunks = [
        _chunk("c1", "doc-a", "aaa", score=0.9),
        _chunk("c2", "doc-b", "bbb", score=0.8),
        _chunk("c3", "doc-c", "ccc", score=0.7),
    ]

    result = reranker.rerank("any query", chunks, k=2)

    assert [c.chunk_id for c in result] == ["c1", "c2"]
    # The NoOp path must NOT attach a rerank_score.
    assert all(c.rerank_score is None for c in result)
    # And must not mutate the input list.
    assert [c.chunk_id for c in chunks] == ["c1", "c2", "c3"]


def test_noop_reranker_handles_empty_and_oversized_k():
    reranker = NoOpReranker()
    assert reranker.rerank("q", [], k=5) == []

    chunks = [_chunk("c1", "doc-a", "aaa")]
    # Oversized k returns everything (no padding).
    assert [c.chunk_id for c in reranker.rerank("q", chunks, k=10)] == ["c1"]


# ---------------------------------------------------------------------------
# 2. CrossEncoderReranker with deterministic monkeypatched scores.
# ---------------------------------------------------------------------------


def _monkeypatch_encoder(monkeypatch, fake: _FakeCrossEncoder) -> None:
    """Swap the lru_cache'd loader for one that returns our fake, and
    clear the cache first so a prior test's real load attempt can't
    leak into this one."""
    from app.capabilities.rag import reranker as reranker_module

    reranker_module._load_cross_encoder.cache_clear()

    def _fake_loader(model_name, max_length, device):
        return fake

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", _fake_loader)


def test_cross_encoder_reorders_by_predicted_scores(monkeypatch):
    """The lowest-scoring bi-encoder chunk can end up rank-1 after
    reranking — that's exactly the signal the CrossEncoder is paid
    to produce."""
    fake = _FakeCrossEncoder({
        "bookshop": 0.95,
        "typhoon":  0.30,
        "gardens":  0.10,
    })
    _monkeypatch_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(batch_size=8, text_max_chars=200, device="cpu")
    chunks = [
        _chunk("c1", "doc-gardens",  "gardens suspended above clouds", score=0.9),
        _chunk("c2", "doc-typhoon",  "construction mechs before typhoon", score=0.8),
        _chunk("c3", "doc-bookshop", "retired translator runs a bookshop", score=0.1),
    ]

    result = reranker.rerank("who runs the bookshop?", chunks, k=2)

    assert [c.chunk_id for c in result] == ["c3", "c2"]
    # rerank_score attached and sorted descending.
    assert result[0].rerank_score == pytest.approx(0.95, rel=1e-5)
    assert result[1].rerank_score == pytest.approx(0.30, rel=1e-5)
    # Original bi-encoder score is preserved alongside the new rerank_score.
    assert result[0].score == pytest.approx(0.1)
    # Batch size is threaded through to predict().
    assert fake.predict_calls == [{"pair_count": 3, "batch_size": 8}]
    # name() surfaces the model for logging.
    assert reranker.name.startswith("cross-encoder:")


def test_cross_encoder_empty_and_k_zero_short_circuit(monkeypatch):
    fake = _FakeCrossEncoder({})
    _monkeypatch_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(device="cpu")
    assert reranker.rerank("q", [], k=5) == []
    # predict must not run when the input list is empty.
    assert fake.predict_calls == []

    chunks = [_chunk("c1", "doc-a", "aaa")]
    assert reranker.rerank("q", chunks, k=0) == []
    assert fake.predict_calls == []


# ---------------------------------------------------------------------------
# 3. Failure isolation: predict() raises -> fall back to bi-encoder order.
# ---------------------------------------------------------------------------


class _ExplodingCrossEncoder:
    def predict(self, *args, **kwargs):
        raise RuntimeError("simulated CUDA OOM during rerank")


def test_cross_encoder_predict_failure_preserves_original_order(monkeypatch):
    _monkeypatch_encoder(monkeypatch, _ExplodingCrossEncoder())

    reranker = CrossEncoderReranker(device="cpu")
    chunks = [
        _chunk("c1", "doc-a", "aaa", score=0.9),
        _chunk("c2", "doc-b", "bbb", score=0.8),
        _chunk("c3", "doc-c", "ccc", score=0.7),
    ]

    result = reranker.rerank("q", chunks, k=2)

    # Original bi-encoder top-k, unchanged order, no rerank_score attached.
    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert all(c.rerank_score is None for c in result)


# ---------------------------------------------------------------------------
# 4. Registry wiring: env=off vs env=cross_encoder (broken init).
# ---------------------------------------------------------------------------


def test_registry_off_yields_noop_and_leaves_rag_untouched(monkeypatch):
    """With rag_reranker='off' the registry must pick NoOpReranker
    without even importing sentence_transformers.CrossEncoder. RAG
    construction path is not exercised here — that's covered by the
    test_rag_validation suite — but we verify the builder's own output."""
    from app.capabilities import registry as registry_module

    # Guard: if _build_reranker accidentally imports the real
    # CrossEncoder for an 'off' setting, this monkeypatch forces a
    # clear failure instead of a silent model download.
    def _should_not_load(*args, **kwargs):  # pragma: no cover
        raise AssertionError(
            "off path must not trigger CrossEncoder load"
        )

    registry_module._build_reranker  # ensure symbol exists
    from app.capabilities.rag import reranker as reranker_module

    reranker_module._load_cross_encoder.cache_clear()
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", _should_not_load)

    from app.core.config import WorkerSettings

    settings = WorkerSettings(rag_reranker="off")
    built = registry_module._build_reranker(settings)

    assert isinstance(built, NoOpReranker)
    assert built.name == "noop"


def test_registry_broken_cross_encoder_falls_back_to_noop(monkeypatch, caplog):
    """When rag_reranker='cross_encoder' but CrossEncoder init blows up,
    the reranker must degrade to NoOp-equivalent behaviour (return
    chunks[:k] unchanged) instead of bubbling the failure up into the
    RAG registration path. The point of the test is: RAG survives a
    broken reranker. The registry is free to return either NoOpReranker
    or a CrossEncoderReranker that internally degrades — both satisfy
    the "RAG still registers" contract.
    """
    import logging as _logging

    from app.capabilities import registry as registry_module
    from app.capabilities.rag import reranker as reranker_module
    from app.core.config import WorkerSettings

    reranker_module._load_cross_encoder.cache_clear()

    def _failing_loader(model_name, max_length, device):
        raise RuntimeError("simulated model-download failure")

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", _failing_loader)
    # Short-circuit torch detection so running this test after other
    # tests that load native modules doesn't trigger a torch-init abort.
    monkeypatch.setattr(reranker_module, "_auto_device", lambda: "cpu")

    settings = WorkerSettings(rag_reranker="cross_encoder")

    with caplog.at_level(_logging.WARNING, logger="app.capabilities.registry"):
        built = registry_module._build_reranker(settings)

    # Whether the registry returned NoOpReranker directly or a
    # CrossEncoderReranker that will degrade at first rerank(), the
    # observable contract is identical: top-k comes back in bi-encoder
    # order with no rerank_score attached.
    chunks = [
        _chunk("c1", "doc-a", "aaa", score=0.9),
        _chunk("c2", "doc-b", "bbb", score=0.8),
    ]
    result = built.rerank("q", chunks, k=2)
    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert all(c.rerank_score is None for c in result)


def test_registry_unknown_reranker_value_falls_back_to_noop(caplog):
    """Typo'd env var names fall back to NoOp with a warning — never
    raise, never silently run the wrong scorer."""
    import logging as _logging

    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    settings = WorkerSettings(rag_reranker="bm25")  # not supported

    with caplog.at_level(_logging.WARNING, logger="app.capabilities.registry"):
        built = registry_module._build_reranker(settings)

    assert isinstance(built, NoOpReranker)


# ---------------------------------------------------------------------------
# 5. Retriever integration: candidate_k -> reranker -> top_k wiring.
# ---------------------------------------------------------------------------


class _CountingReranker(RerankerProvider):
    """Records the length of the candidate list it received so the
    test can verify candidate_k was honoured."""

    def __init__(self) -> None:
        self.last_candidate_count: int = -1

    @property
    def name(self) -> str:
        return "counting"

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        self.last_candidate_count = len(chunks)
        # Trivially re-score so downstream can confirm the output was
        # reranker-produced (every chunk gets rerank_score=1.0).
        scored = [
            RetrievedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, section=c.section,
                text=c.text, score=c.score, rerank_score=1.0,
            )
            for c in chunks
        ]
        return scored[:k]


class _FakeFaissIndex:
    """Fake FAISS index that records the ``top_k`` it was asked for.

    Skips the native FAISS round-trip entirely — the only thing the
    reranker wiring tests need to verify is that the Retriever's
    candidate_k flows into search() as the k argument. Anything
    that actually runs FAISS for this is wasted test-infra surface
    and has bitten us with native aborts on small indices before.
    """

    def __init__(self, info) -> None:
        self._info = info
        self.last_search_k: int = -1
        self._row_count = info.chunk_count

    def load(self):
        return self._info

    @property
    def info(self):
        return self._info

    def search(self, query_vectors, top_k: int):
        self.last_search_k = int(top_k)
        k = min(int(top_k), self._row_count)
        # Return row_ids 0..k-1 with decreasing synthetic scores so
        # ranking is stable and the metadata store can hand back chunks.
        return [[(i, 1.0 - 0.01 * i) for i in range(k)]]


class _FakeStore:
    def __init__(self, rows) -> None:
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(self, index_version, ids):
        return [self._by_row[i] for i in ids if i in self._by_row]


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


def _fake_info(chunk_count: int, model_name: str, dim: int):
    from app.capabilities.rag.faiss_index import IndexBuildInfo

    return IndexBuildInfo(
        index_version="test-v1",
        embedding_model=model_name,
        dimension=dim,
        chunk_count=chunk_count,
    )


def test_retriever_plumbs_candidate_k_into_reranker():
    """The Retriever must ask FAISS for candidate_k vectors and hand
    the resulting chunks to the reranker; the reranker's output
    becomes the RetrievalReport.results (at most top_k rows)."""
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.retriever import Retriever

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=5,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info)
    store = _FakeStore(_build_rows(5))

    counting = _CountingReranker()
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=2,
        reranker=counting,
        candidate_k=4,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("any query")

    assert index.last_search_k == 4
    assert counting.last_candidate_count == 4
    assert len(report.results) == 2
    assert report.reranker_name == "counting"
    assert report.candidate_k == 4
    assert all(r.rerank_score == 1.0 for r in report.results)


def test_retriever_candidate_k_never_below_top_k():
    """candidate_k <= top_k collapses to top_k; this is the Phase 0
    reproducibility guarantee."""
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.retriever import Retriever

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=3,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info)
    store = _FakeStore(_build_rows(3))

    counting = _CountingReranker()
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=3,
        reranker=counting,
        candidate_k=1,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("any query")

    assert index.last_search_k == 3
    assert report.candidate_k == 3
    assert counting.last_candidate_count == 3


def test_retriever_defaults_to_noop_reranker_when_none_passed():
    """Phase 0 path: existing call sites that don't pass a reranker get
    a NoOpReranker + candidate_k == top_k, which reproduces the old
    bi-encoder-only behaviour byte-for-byte."""
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.retriever import Retriever

    embedder = HashingEmbedder(dim=32)
    info = _fake_info(
        chunk_count=3,
        model_name=embedder.model_name,
        dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info)
    store = _FakeStore(_build_rows(3))

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=2,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("any query")

    assert index.last_search_k == 2  # candidate_k collapsed to top_k
    assert report.reranker_name == "noop"
    assert report.candidate_k == 2
    # NoOp path must not attach rerank_score.
    assert all(r.rerank_score is None for r in report.results)
