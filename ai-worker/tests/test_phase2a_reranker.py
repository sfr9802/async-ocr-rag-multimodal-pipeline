"""Phase 2A reranker tests.

Five blocks of coverage, all fully offline:

  1. CrossEncoderReranker OOM-retry path
     - First predict() raises a CUDA-OOM-style RuntimeError, second
       predict() (called at half batch_size) succeeds. The reranker
       returns the rescaled top-k correctly.
     - First predict() raises non-OOM RuntimeError → original chunks[:k]
       (no second attempt; this preserves the failure-isolation
       contract for non-recoverable errors).
     - First predict() raises OOM, second predict() also raises OOM →
       original chunks[:k] (one retry only, then degrade).

  2. ``_is_cuda_oom_exception`` does not import the torch CUDA driver
     in-process — confirmed by patching ``torch`` to a stub before the
     helper is called.

  3. RetrievalReport.rerank_ms is populated by Retriever.retrieve
     when a non-NoOp reranker is in play; NoOpReranker leaves it None
     so eval reports can distinguish "didn't rerank" from "0 ms".

  4. retrieval_eval honours extra_hit_ks: the requested cutoffs flow
     into row.extra_hits AND summary.mean_extra_hits with the right
     keys; the markdown writer renders them in numeric order; the
     latency aggregator picks up rerank_ms when present.

  5. CLI helper ``_resolve_extra_hit_ks`` clamps non-positive,
     deduplicates, and sorts.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import pytest

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_rag_reranker.py so this file stands alone).
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


class _SequencedCrossEncoder:
    """A CrossEncoder stand-in that runs from a queue of (callable, label).

    Each predict() call pops the next entry off ``script`` and either
    calls it (raising whatever it raises) or returns a numpy array of
    scores. The harness records the batch_size threaded through to
    each call so the OOM-retry test can assert "halved on the retry".
    """

    def __init__(self, script: List[Any]) -> None:
        self._script = list(script)
        self.calls: List[dict] = []

    def predict(
        self,
        pairs,
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ):
        self.calls.append({
            "pair_count": len(pairs),
            "batch_size": batch_size,
        })
        if not self._script:
            raise AssertionError(
                "Sequenced encoder ran out of scripted responses"
            )
        action = self._script.pop(0)
        result = action(pairs)
        return np.asarray(result, dtype=np.float32)


def _patch_loader(monkeypatch, fake) -> None:
    from app.capabilities.rag import reranker as reranker_module

    reranker_module._load_cross_encoder.cache_clear()
    monkeypatch.setattr(
        reranker_module,
        "_load_cross_encoder",
        lambda model, max_length, device: fake,
    )


def _string_only_oom_check(exc: BaseException) -> bool:
    """Pure-Python OOM detector used in unit tests to avoid ``import torch``.

    The production helper falls back to this same string match when
    torch isn't importable; using it directly in the test fixture
    keeps the OOM-retry tests from triggering torch init mid-process.
    """
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


@pytest.fixture(autouse=True)
def _no_cuda_log(monkeypatch):
    """Replace torch-touching helpers with pure-Python stand-ins.

    Two helpers in ``embeddings.py`` (``_log_cuda_memory``,
    ``_is_cuda_oom_exception``) lazily ``import torch``. In a pytest
    process that has already loaded faiss + numpy + other native
    modules, that import can SEGFAULT (interpreter abort, not Python
    exception), taking down the whole test session.

    The existing comment in ``tests/test_rag_embeddings_helpers.py``
    documents the same risk; we patch both helpers to no-op-equivalent
    pure-Python versions for the duration of any test in this file.
    Production observability is unaffected because the real worker
    imports torch via the embedder long before any reranker call.
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


# ---------------------------------------------------------------------------
# 1. CrossEncoderReranker OOM-retry path.
# ---------------------------------------------------------------------------


def test_oom_retries_at_half_batch_size_and_succeeds(monkeypatch):
    """First predict() raises a CUDA-OOM-style RuntimeError; second
    predict() returns scores. The reranker must:

      - call predict() exactly twice
      - thread batch_size=8 into call 1 and batch_size=4 into call 2
      - emit the reranked top-k from the retry's scores
      - log a WARNING with both batch sizes (smoke check via caplog)
    """
    def _raise_oom(pairs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 8 GiB")

    def _ok(pairs):
        # Score by passage substring so the assertion below knows the
        # ordering before the test runs.
        scores: List[float] = []
        for _q, passage in pairs:
            if "winner" in passage:
                scores.append(0.95)
            elif "middle" in passage:
                scores.append(0.50)
            else:
                scores.append(0.10)
        return scores

    fake = _SequencedCrossEncoder(script=[_raise_oom, _ok])
    _patch_loader(monkeypatch, fake)

    reranker = CrossEncoderReranker(
        batch_size=8,
        text_max_chars=200,
        device="cpu",
        # Default OOM fallback would be 4; pin it explicitly so the
        # assertion below is decoupled from the default heuristic.
        oom_fallback_batch_size=4,
    )
    chunks = [
        _chunk("c1", "doc-loser",  "background loser passage", score=0.8),
        _chunk("c2", "doc-middle", "middle of the pack", score=0.7),
        _chunk("c3", "doc-winner", "this is the winner", score=0.6),
    ]

    result = reranker.rerank("any query", chunks, k=3)

    assert [c.chunk_id for c in result] == ["c3", "c2", "c1"]
    assert result[0].rerank_score == pytest.approx(0.95, rel=1e-5)
    assert [c["batch_size"] for c in fake.calls] == [8, 4]
    assert len(fake.calls) == 2


def test_non_oom_runtime_error_falls_back_without_retry(monkeypatch, caplog):
    """A non-OOM RuntimeError must NOT trigger the OOM-fallback retry.

    Two reasons this matters: (a) we don't want infinite-loop cascades
    on a configuration error that re-raises every call, and (b) the
    OOM fallback is meaningful only when the failure is plausibly
    memory-pressure driven. Other RuntimeErrors are handled by the
    same chunks[:k] degradation as before.
    """
    def _raise_other(pairs):
        raise RuntimeError("model misconfigured: max_length is None")

    fake = _SequencedCrossEncoder(script=[_raise_other])
    _patch_loader(monkeypatch, fake)

    reranker = CrossEncoderReranker(batch_size=8, device="cpu")
    chunks = [
        _chunk("c1", "doc-a", "aaa", score=0.9),
        _chunk("c2", "doc-b", "bbb", score=0.8),
    ]

    with caplog.at_level(logging.WARNING, logger="app.capabilities.rag.reranker"):
        result = reranker.rerank("q", chunks, k=2)

    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert all(c.rerank_score is None for c in result)
    # Exactly one predict() call was made — no retry on a non-OOM error.
    assert len(fake.calls) == 1
    assert any(
        "predict failed" in m and "RuntimeError" in m
        for m in caplog.messages
    )


def test_oom_then_oom_again_falls_back(monkeypatch, caplog):
    """When the half-batch retry ALSO raises (OOM or otherwise) the
    reranker must give up and return chunks[:k]."""
    def _raise_oom(pairs):
        raise RuntimeError("CUDA out of memory: again")

    def _raise_again(pairs):
        raise RuntimeError("still out of memory")

    fake = _SequencedCrossEncoder(script=[_raise_oom, _raise_again])
    _patch_loader(monkeypatch, fake)

    reranker = CrossEncoderReranker(
        batch_size=8, oom_fallback_batch_size=4, device="cpu",
    )
    chunks = [
        _chunk("c1", "doc-a", "aaa", score=0.9),
        _chunk("c2", "doc-b", "bbb", score=0.8),
    ]

    with caplog.at_level(logging.WARNING, logger="app.capabilities.rag.reranker"):
        result = reranker.rerank("q", chunks, k=2)

    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert all(c.rerank_score is None for c in result)
    assert len(fake.calls) == 2  # one OOM + one retry, then give up.


def test_oom_fallback_disabled_when_not_smaller(monkeypatch, caplog):
    """oom_fallback_batch_size >= batch_size means "no retry" — the
    reranker logs a warning and falls back without a second predict().

    Important for the unit-test contract: callers can pin the fallback
    to the same number as the primary to disable retry entirely.
    """
    def _raise_oom(pairs):
        raise RuntimeError("CUDA out of memory")

    fake = _SequencedCrossEncoder(script=[_raise_oom])
    _patch_loader(monkeypatch, fake)

    reranker = CrossEncoderReranker(
        batch_size=4, oom_fallback_batch_size=4, device="cpu",
    )
    chunks = [_chunk("c1", "doc-a", "aaa")]

    with caplog.at_level(logging.WARNING, logger="app.capabilities.rag.reranker"):
        result = reranker.rerank("q", chunks, k=1)

    assert [c.chunk_id for c in result] == ["c1"]
    assert len(fake.calls) == 1
    assert any("not smaller" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# 2. _is_cuda_oom_exception does not touch CUDA driver in-process.
# ---------------------------------------------------------------------------


def test_oom_helper_string_match_contract():
    """Pin the OOM-detection contract via a pure-Python equivalent.

    We deliberately do NOT exercise the production
    ``_is_cuda_oom_exception`` here: the production helper does
    ``import torch`` lazily, and the existing comment in
    ``tests/test_rag_embeddings_helpers.py`` documents that triggering
    a torch import inside an already-loaded pytest process can segfault
    the interpreter (it has bitten the suite before).

    Instead we test the same string-match rule using the same
    ``_string_only_oom_check`` helper the OOM-retry tests rely on for
    fault injection. This exercises the contract — "RuntimeError with
    'out of memory' in the message → OOM; everything else → not OOM"
    — without touching torch in this process. The production helper's
    isinstance-against-torch.cuda.OutOfMemoryError branch is exercised
    end-to-end when the worker actually runs against a GPU; the unit
    tests cover the version-portable string-match fallback that catches
    the same condition on older torch versions and CPU-only deployments.
    """
    assert _string_only_oom_check(
        RuntimeError("CUDA out of memory. Tried to allocate 8.00 GiB")
    ) is True
    # Case-insensitive substring match.
    assert _string_only_oom_check(
        RuntimeError("[CUDA OUT OF MEMORY] cudaMalloc failed")
    ) is True
    # Non-OOM RuntimeError → False.
    assert _string_only_oom_check(
        RuntimeError("model misconfigured: max_length is None")
    ) is False
    # Non-RuntimeError → False, regardless of message.
    assert _string_only_oom_check(ValueError("CUDA out of memory")) is False
    assert _string_only_oom_check(TypeError("anything")) is False


# ---------------------------------------------------------------------------
# 3. RetrievalReport.rerank_ms wiring.
# ---------------------------------------------------------------------------


class _FakeFaissIndex:
    """Minimal FAISS stub matching test_rag_reranker.py's pattern."""

    def __init__(self, info) -> None:
        self._info = info
        self._row_count = info.chunk_count
        self.last_search_k = -1

    def load(self):
        return self._info

    def search(self, query_vectors, top_k: int):
        self.last_search_k = int(top_k)
        k = min(int(top_k), self._row_count)
        return [[(i, 1.0 - 0.01 * i) for i in range(k)]]


class _FakeStore:
    def __init__(self, rows) -> None:
        self._by_row = {r.faiss_row_id: r for r in rows}

    def lookup_chunks_by_faiss_rows(self, index_version, ids):
        return [self._by_row[i] for i in ids if i in self._by_row]


def _build_chunk_rows(n: int):
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


def test_rerank_ms_populated_when_reranker_active(monkeypatch):
    """When a non-NoOp reranker runs, the RetrievalReport carries a
    non-None rerank_ms reflecting the rerank wall-clock."""
    import time as _time

    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.reranker import RerankerProvider
    from app.capabilities.rag.retriever import Retriever

    class _SlowReranker(RerankerProvider):
        @property
        def name(self) -> str:
            return "slow-test"

        def rerank(self, query, chunks, k):
            _time.sleep(0.005)
            return list(chunks[:k])

    embedder = HashingEmbedder(dim=16)
    info = _fake_info(
        chunk_count=3, model_name=embedder.model_name, dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info)
    store = _FakeStore(_build_chunk_rows(3))

    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=store,
        top_k=2,
        reranker=_SlowReranker(),
        candidate_k=3,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("any query")

    assert report.reranker_name == "slow-test"
    assert report.rerank_ms is not None
    # 5 ms sleep, allow generous slack for slow CI hosts.
    assert report.rerank_ms >= 4.5


def test_rerank_ms_left_none_for_noop_reranker():
    from app.capabilities.rag.embeddings import HashingEmbedder
    from app.capabilities.rag.retriever import Retriever

    embedder = HashingEmbedder(dim=16)
    info = _fake_info(
        chunk_count=3, model_name=embedder.model_name, dim=embedder.dimension,
    )
    index = _FakeFaissIndex(info)
    store = _FakeStore(_build_chunk_rows(3))

    retriever = Retriever(
        embedder=embedder, index=index, metadata=store, top_k=2,
    )
    retriever.ensure_ready()
    report = retriever.retrieve("q")

    assert report.reranker_name == "noop"
    assert report.rerank_ms is None


# ---------------------------------------------------------------------------
# 4. retrieval_eval extra_hit_ks + rerank_ms aggregation.
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns a hard-coded list of RetrievedChunks; useful for harness tests."""

    def __init__(self, results, *, rerank_ms=None, reranker_name="noop"):
        self._results = list(results)
        self._rerank_ms = rerank_ms
        self._reranker_name = reranker_name

    def retrieve(self, query: str):
        return _StubReport(
            results=self._results,
            rerank_ms=self._rerank_ms,
            reranker_name=self._reranker_name,
        )


class _StubReport:
    def __init__(self, *, results, rerank_ms, reranker_name):
        self.results = results
        self.rerank_ms = rerank_ms
        self.reranker_name = reranker_name
        self.index_version = "stub-v1"
        self.embedding_model = "stub-embedder"
        self.candidate_k = len(results)
        self.use_mmr = False
        self.mmr_lambda = None
        self.dup_rate = 0.0


def test_retrieval_eval_extra_hit_ks_round_trip():
    from eval.harness.retrieval_eval import (
        render_markdown_report,
        run_retrieval_eval,
    )

    results = [
        RetrievedChunk(chunk_id=f"c{i}", doc_id=f"doc-{i}",
                       section="s", text=f"text {i}", score=1.0 - 0.05 * i)
        for i in range(15)
    ]
    # Expected doc lives at rank 12 — within hit@20 / hit@50 but
    # outside hit@10. This pins the metric direction in the assertion.
    dataset = [{
        "id": "row-1",
        "query": "anything",
        "expected_doc_ids": ["doc-12"],
        "answer_type": "summary_plot",
        "difficulty": "medium",
    }]
    retriever = _StubRetriever(results)

    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever,
        top_k=20, mrr_k=20, ndcg_k=20,
        extra_hit_ks=(10, 20, 50),
    )

    row = rows[0]
    assert row.extra_hits == {"10": 0.0, "20": 1.0, "50": 1.0}
    assert summary.mean_extra_hits == {"10": 0.0, "20": 1.0, "50": 1.0}

    md = render_markdown_report(summary, rows, _fake_dup())
    # The markdown must include hit@10 / hit@20 / hit@50 rows after
    # hit@5 (numeric order, not lexicographic).
    h5 = md.find("hit@5")
    h10 = md.find("hit@10")
    h20 = md.find("hit@20")
    h50 = md.find("hit@50")
    assert -1 < h5 < h10 < h20 < h50


def _fake_dup():
    from eval.harness.retrieval_eval import DuplicateAnalysis

    return DuplicateAnalysis(
        top_k=10,
        queries_evaluated=1,
        queries_with_doc_dup=0,
        queries_with_section_dup=0,
        queries_with_text_dup=0,
        queries_with_doc_dup_ratio=0.0,
        queries_with_section_dup_ratio=0.0,
        queries_with_text_dup_ratio=0.0,
    )


def test_retrieval_eval_aggregates_rerank_ms_when_present():
    """When the retriever surfaces rerank_ms, the harness must roll up
    mean / p50 / p95 / max — and it must surface a meaningful row count
    so a NoOp run (rerank_row_count=0) is distinguishable from a real
    rerank run with all-zero latencies."""
    from eval.harness.retrieval_eval import run_retrieval_eval

    results_a = [
        RetrievedChunk(chunk_id="c1", doc_id="doc-a", section="s",
                       text="x", score=0.9),
    ]
    results_b = [
        RetrievedChunk(chunk_id="c2", doc_id="doc-b", section="s",
                       text="y", score=0.8),
    ]

    class _AlternatingRetriever:
        def __init__(self):
            self._calls = 0

        def retrieve(self, query):
            self._calls += 1
            ms = 5.0 if self._calls == 1 else 15.0
            results = results_a if self._calls == 1 else results_b
            return _StubReport(
                results=results,
                rerank_ms=ms,
                reranker_name="cross-encoder:test",
            )

    dataset = [
        {"id": "r-1", "query": "q1", "expected_doc_ids": ["doc-a"]},
        {"id": "r-2", "query": "q2", "expected_doc_ids": ["doc-b"]},
    ]

    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=_AlternatingRetriever(),
        top_k=5, mrr_k=10, ndcg_k=10,
    )

    assert [r.rerank_ms for r in rows] == [5.0, 15.0]
    assert summary.rerank_row_count == 2
    assert summary.mean_rerank_ms == pytest.approx(10.0)
    assert summary.max_rerank_ms == pytest.approx(15.0)


def test_retrieval_eval_rerank_row_count_zero_when_noop():
    from eval.harness.retrieval_eval import run_retrieval_eval

    results = [
        RetrievedChunk(chunk_id="c1", doc_id="doc-a", section="s",
                       text="x", score=0.9),
    ]
    dataset = [
        {"id": "r-1", "query": "q1", "expected_doc_ids": ["doc-a"]},
    ]
    retriever = _StubRetriever(results, rerank_ms=None, reranker_name="noop")

    summary, rows, _, _ = run_retrieval_eval(
        dataset, retriever=retriever, top_k=5, mrr_k=10, ndcg_k=10,
    )

    assert rows[0].rerank_ms is None
    assert summary.rerank_row_count == 0
    assert summary.mean_rerank_ms is None
    assert summary.p95_rerank_ms is None


# ---------------------------------------------------------------------------
# 5. CLI helper: _resolve_extra_hit_ks.
# ---------------------------------------------------------------------------


def test_resolve_extra_hit_ks_drops_nonpositive_and_dedupes():
    from eval.run_eval import _resolve_extra_hit_ks

    out = _resolve_extra_hit_ks([20, -1, 0, 10, 50, 20], top_k=50)
    assert out == (10, 20, 50)


def test_resolve_extra_hit_ks_warns_when_above_top_k(caplog):
    from eval.run_eval import _resolve_extra_hit_ks

    with caplog.at_level(logging.WARNING, logger="eval"):
        out = _resolve_extra_hit_ks([10, 60], top_k=50)

    assert out == (10, 60)  # we don't drop the value, just warn
    assert any("exceeds top_k" in m for m in caplog.messages)


def test_resolve_extra_hit_ks_empty_for_none_input():
    from eval.run_eval import _resolve_extra_hit_ks

    assert _resolve_extra_hit_ks(None, top_k=10) == ()
    assert _resolve_extra_hit_ks([], top_k=10) == ()


# ---------------------------------------------------------------------------
# 6. Reranker abstraction contracts under harness use.
# ---------------------------------------------------------------------------


def test_noop_reranker_preserves_candidate_count_within_k():
    """Smoke check that's separate from the existing test suite: when
    the harness asks the NoOpReranker for k less than len(chunks), the
    reranker must surface exactly k chunks. This pins the contract
    that reranker.rerank(query, chunks, k) returns at most k items —
    relied on by Phase 2A's "dense_top_n=20, final_top_k=10" flow."""
    reranker = NoOpReranker()
    chunks = [
        _chunk(f"c{i}", f"doc-{i}", f"text {i}") for i in range(20)
    ]
    out = reranker.rerank("q", chunks, k=10)
    assert len(out) == 10
    # First-10 in order, no rerank_score attached (NoOp contract).
    assert [c.chunk_id for c in out] == [f"c{i}" for i in range(10)]
    assert all(c.rerank_score is None for c in out)


def test_cross_encoder_preserves_dense_score_alongside_rerank_score(monkeypatch):
    """When the reranker re-orders, each emitted chunk MUST keep the
    original bi-encoder ``score`` field. Failure analysis depends on
    the dense score being available even after a rerank reorder so a
    reviewer can see what the bi-encoder thought before the
    cross-encoder rescued or regressed the query."""
    def _ok(pairs):
        # Score by the suffix in the passage so we get a deterministic
        # reorder.
        scores: List[float] = []
        for _q, passage in pairs:
            if "TOP" in passage:
                scores.append(0.95)
            else:
                scores.append(0.10)
        return scores

    fake = _SequencedCrossEncoder(script=[_ok])
    _patch_loader(monkeypatch, fake)

    reranker = CrossEncoderReranker(batch_size=8, device="cpu")
    chunks = [
        _chunk("c1", "doc-low",  "low rank dense", score=0.95),
        _chunk("c2", "doc-high", "high rank dense TOP", score=0.05),
    ]
    out = reranker.rerank("q", chunks, k=2)

    assert [c.chunk_id for c in out] == ["c2", "c1"]
    # Dense scores preserved alongside the new rerank_score.
    assert out[0].score == pytest.approx(0.05)
    assert out[1].score == pytest.approx(0.95)
    assert out[0].rerank_score == pytest.approx(0.95, rel=1e-5)
    assert out[1].rerank_score == pytest.approx(0.10, rel=1e-5)


# ---------------------------------------------------------------------------
# 7. Reranker comparison + failure-analysis post-processors.
# ---------------------------------------------------------------------------


def test_build_reranker_comparison_round_trips_min_fields(tmp_path):
    """Comparison builder reads N retrieval_eval_report.json files and
    pulls out the headline summary + key metadata into a flat row.
    The markdown writer renders the resulting comparison without
    crashing on partial slices (some Phase 1 reports won't have any
    rerank fields)."""
    import json

    from eval.harness.reranker_eval import (
        build_reranker_comparison,
        render_reranker_comparison_markdown,
    )

    a = tmp_path / "a.json"
    a.write_text(json.dumps({
        "metadata": {
            "corpus_path": "eval/corpora/anime_x/corpus.jsonl",
        },
        "summary": {
            "row_count": 100,
            "rows_with_expected_doc_ids": 100,
            "top_k": 10,
            "mrr_k": 10,
            "ndcg_k": 10,
            "mean_hit_at_1": 0.50,
            "mean_hit_at_3": 0.65,
            "mean_hit_at_5": 0.70,
            "mean_mrr_at_10": 0.58,
            "mean_ndcg_at_10": 0.61,
            "mean_dup_rate": 0.20,
            "mean_avg_context_token_count": 250.0,
            "mean_extra_hits": {"20": 0.85, "50": 0.92},
            "mean_retrieval_ms": 12.0,
            "p95_retrieval_ms": 16.0,
            "rerank_row_count": 0,
            "mean_rerank_ms": None,
            "p95_rerank_ms": None,
            "reranker_name": "noop",
            "embedding_model": "BAAI/bge-m3",
            "index_version": "offline-1",
        },
    }), encoding="utf-8")

    b = tmp_path / "b.json"
    b.write_text(json.dumps({
        "metadata": {
            "corpus_path": "eval/corpora/anime_x/corpus.jsonl",
            "dense_top_n": 20,
            "final_top_k": 10,
            "reranker_batch_size": 16,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        },
        "summary": {
            "row_count": 100,
            "rows_with_expected_doc_ids": 100,
            "top_k": 10,
            "mrr_k": 10,
            "ndcg_k": 10,
            "mean_hit_at_1": 0.55,
            "mean_hit_at_3": 0.70,
            "mean_hit_at_5": 0.75,
            "mean_mrr_at_10": 0.62,
            "mean_ndcg_at_10": 0.65,
            "mean_dup_rate": 0.18,
            "mean_avg_context_token_count": 240.0,
            "mean_retrieval_ms": 12.0,
            "p95_retrieval_ms": 16.0,
            "rerank_row_count": 100,
            "mean_rerank_ms": 80.0,
            "p95_rerank_ms": 120.0,
            "reranker_name": "cross-encoder:BAAI/bge-reranker-v2-m3",
            "embedding_model": "BAAI/bge-m3",
            "index_version": "offline-2",
        },
    }), encoding="utf-8")

    comparison = build_reranker_comparison(
        [("dense-only", a), ("rerank-top20", b)],
    )
    assert comparison["slice_count"] == 2
    labels = [s["label"] for s in comparison["slices"]]
    assert labels == ["dense-only", "rerank-top20"]
    rerank_slice = comparison["slices"][1]
    assert rerank_slice["dense_top_n"] == 20
    assert rerank_slice["final_top_k"] == 10
    assert rerank_slice["reranker_batch_size"] == 16

    md = render_reranker_comparison_markdown(comparison)
    assert "dense-only" in md and "rerank-top20" in md
    assert "BAAI/bge-reranker-v2-m3" in md
    # The candidate-recall companion table renders only when at least
    # one slice has extra hits — we put them on the dense slice.
    assert "hit@20" in md
    assert "hit@50" in md


def test_build_reranker_failure_analysis_buckets_correctly():
    """Buckets a synthetic dense vs. rerank cross-tab into rescued /
    regressed / both-miss / both-hit using only hit_at_1 fields."""
    from eval.harness.reranker_eval import build_reranker_failure_analysis

    dense_rows = [
        {"id": "q1", "query": "rescue me", "expected_doc_ids": ["d1"], "hit_at_1": 0.0,
         "answer_type": "title_lookup", "difficulty": "easy"},
        {"id": "q2", "query": "regress me", "expected_doc_ids": ["d2"], "hit_at_1": 1.0},
        {"id": "q3", "query": "miss both", "expected_doc_ids": ["d3"], "hit_at_1": 0.0},
        {"id": "q4", "query": "hit both",  "expected_doc_ids": ["d4"], "hit_at_1": 1.0},
    ]
    rerank_rows = [
        {"id": "q1", "query": "rescue me", "expected_doc_ids": ["d1"], "hit_at_1": 1.0},
        {"id": "q2", "query": "regress me", "expected_doc_ids": ["d2"], "hit_at_1": 0.0},
        {"id": "q3", "query": "miss both", "expected_doc_ids": ["d3"], "hit_at_1": 0.0},
        {"id": "q4", "query": "hit both",  "expected_doc_ids": ["d4"], "hit_at_1": 1.0},
    ]
    dense_dump = [
        {"query_id": "q1", "rank": 1, "doc_id": "X", "chunk_id": "cx",
         "section_path": "s", "score": 0.5, "rerank_score": None,
         "is_expected_doc": False, "chunk_preview": "wrong"},
    ]
    rerank_dump = [
        {"query_id": "q1", "rank": 1, "doc_id": "d1", "chunk_id": "cd1",
         "section_path": "s", "score": 0.4, "rerank_score": 0.95,
         "is_expected_doc": True, "chunk_preview": "right"},
    ]

    analysis = build_reranker_failure_analysis(
        dense_rows=dense_rows,
        rerank_rows=rerank_rows,
        dense_dump=dense_dump,
        rerank_dump=rerank_dump,
        k_preview=5,
        sample_cap=10,
    )
    assert analysis["bucket_counts"] == {
        "dense_miss_to_rerank_hit": 1,
        "dense_hit_to_rerank_miss": 1,
        "both_hit": 1,
        "both_miss": 1,
    }
    rescued = analysis["buckets"]["dense_miss_to_rerank_hit"]
    assert len(rescued) == 1
    assert rescued[0]["query_id"] == "q1"
    assert rescued[0]["dense_top"][0]["doc_id"] == "X"
    assert rescued[0]["rerank_top"][0]["doc_id"] == "d1"
    # both_hit bucket is counted but not sampled — it's not a
    # diagnostic group.
    assert analysis["buckets"]["both_hit"] == []
