"""Phase 7.1 — tests for the reranker A/B harness.

Fully fixture-driven: every test composes a fake retriever (returning
prescribed dense candidate pools per query) and a fake reranker (with a
prescribed score table) so the harness can be exercised end-to-end with
zero FAISS / CrossEncoder dependency. The tests pin:

  - dense baseline ordering preservation when the reranker is no-op
  - reranker ordering applied when it has a non-trivial score table
  - weighted_dense_rerank arithmetic (min-max normalised blend)
  - improved / regressed / both_hit / both_missed classification
  - latency_summary aggregation (count + percentile boundaries)
  - bucket aggregation: every bucket the input set carries appears in
    by_bucket; counts add up to n_queries
  - missing-gold handling: a query with no expected_doc_ids stays in
    the bookkeeping but is never marked "improved" / "regressed"
  - JSONL output schema stability for per_query / improved / regressed
    / latency files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import pytest

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import NoOpReranker, RerankerProvider

from eval.harness.v4_ab_eval import QueryRecord
from eval.harness.v4_rerank_ab import (
    LatencySummary,
    RerankerAbConfig,
    RerankerAbResult,
    SCORE_MODE_RERANKER_ONLY,
    SCORE_MODE_WEIGHTED,
    SCORE_MODES,
    blend_scores,
    run_reranker_ab,
    run_reranker_candidate,
    summarize_latency,
    write_ab_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str, doc_id: str, *, score: float = 0.5,
    section: str = "개요", text: str = "본문",
    rerank_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=doc_id, section=section,
        text=text, score=score, rerank_score=rerank_score,
    )


class _FakeRetrievalReport:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = list(results)


class _ProgrammedRetriever:
    """Retriever stand-in that returns a prescribed pool per query.

    The pool is whatever the test wires into the lookup table. The test
    is responsible for ensuring the pool has at least ``candidate_k``
    entries when the harness consumes it as the candidate side.
    """

    def __init__(self, table: Dict[str, List[RetrievedChunk]]) -> None:
        self._table = table

    def retrieve(self, query: str) -> _FakeRetrievalReport:
        return _FakeRetrievalReport(self._table.get(query, []))


class _ScoreTableReranker(RerankerProvider):
    """Reranker that orders by a (chunk_id -> score) table.

    Anything not in the table gets a default score (-inf by default,
    so it sinks to the bottom). The reranker writes the score onto
    each chunk as ``rerank_score`` so the harness's weighted-blend mode
    has something to operate on.
    """

    def __init__(
        self,
        scores: Dict[str, float],
        *,
        default: float = float("-inf"),
        latency_log: List[Dict[str, Any]] | None = None,
        name_str: str = "fake-rerank",
    ) -> None:
        self._scores = dict(scores)
        self._default = default
        self._calls: List[Dict[str, Any]] = []
        self._latency_log = latency_log if latency_log is not None else self._calls
        self._name = name_str

    @property
    def name(self) -> str:
        return self._name

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return list(self._calls)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        self._calls.append({"query": query, "n_chunks": len(chunks), "k": k})
        if not chunks:
            return []
        scored: List[RetrievedChunk] = []
        for c in chunks:
            s = self._scores.get(c.chunk_id, self._default)
            scored.append(RetrievedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, section=c.section,
                text=c.text, score=c.score, rerank_score=float(s),
            ))
        scored.sort(key=lambda x: x.rerank_score or float("-inf"), reverse=True)
        return scored[: max(0, int(k))]


def _q(
    qid: str, query: str, expected: tuple[str, ...] = (),
    bucket: str = "main_work",
) -> QueryRecord:
    return QueryRecord(
        qid=qid, query=query, expected_doc_ids=expected,
        answer_type="title_lookup", difficulty="easy",
        bucket=bucket, v4_meta={"bucket": bucket},
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_validation_rejects_candidate_smaller_than_final():
    cfg = RerankerAbConfig(candidate_k=5, final_k=10)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_validation_rejects_unknown_score_mode():
    cfg = RerankerAbConfig(score_mode="bogus")
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_score_modes_registry_pinned():
    assert set(SCORE_MODES) == {SCORE_MODE_RERANKER_ONLY, SCORE_MODE_WEIGHTED}


def test_config_validation_rejects_negative_weights():
    cfg = RerankerAbConfig(
        score_mode=SCORE_MODE_WEIGHTED,
        dense_weight=-0.1, rerank_weight=0.5,
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_validation_rejects_zero_total_weights():
    cfg = RerankerAbConfig(
        score_mode=SCORE_MODE_WEIGHTED,
        dense_weight=0.0, rerank_weight=0.0,
    )
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------


def test_blend_scores_minmax_normalises_inside_pool():
    a = _chunk("a", "d1", score=0.1, rerank_score=0.0)
    b = _chunk("b", "d2", score=0.9, rerank_score=10.0)
    out = blend_scores([a, b], dense_weight=0.5, rerank_weight=0.5)
    # Both columns min-max → 0 / 1; weighted sum / total = 0 / 1.
    assert out[0][1] == pytest.approx(0.0, abs=1e-9)
    assert out[1][1] == pytest.approx(1.0, abs=1e-9)


def test_blend_scores_collapses_constant_to_half():
    a = _chunk("a", "d1", score=0.5, rerank_score=2.0)
    b = _chunk("b", "d2", score=0.5, rerank_score=2.0)
    out = blend_scores([a, b], dense_weight=1.0, rerank_weight=1.0)
    # Constant on both columns → minmax floor = 0.5; weighted = 0.5.
    for _, blended in out:
        assert blended == pytest.approx(0.5, abs=1e-9)


def test_blend_scores_respects_weights_when_columns_differ():
    a = _chunk("a", "d1", score=0.0, rerank_score=10.0)  # bad dense, good rerank
    b = _chunk("b", "d2", score=10.0, rerank_score=0.0)  # good dense, bad rerank
    # Weight rerank 100% → "a" wins.
    rerank_only = blend_scores([a, b], dense_weight=0.0, rerank_weight=1.0)
    rerank_only.sort(key=lambda t: t[1], reverse=True)
    assert rerank_only[0][0].chunk_id == "a"
    # Weight dense 100% → "b" wins.
    dense_only = blend_scores([a, b], dense_weight=1.0, rerank_weight=0.0)
    dense_only.sort(key=lambda t: t[1], reverse=True)
    assert dense_only[0][0].chunk_id == "b"


def test_blend_scores_handles_empty_input():
    assert blend_scores([], dense_weight=0.5, rerank_weight=0.5) == []


# ---------------------------------------------------------------------------
# run_reranker_candidate — diagnostics
# ---------------------------------------------------------------------------


def test_run_reranker_candidate_records_rank_before_and_after():
    """Gold sits at dense rank 3; rerank promotes it to rank 1."""
    pool = [
        _chunk("c1", "d-other"),
        _chunk("c2", "d-other"),
        _chunk("c3", "d-target"),
        _chunk("c4", "d-other"),
    ]
    reranker = _ScoreTableReranker({"c3": 1.0})
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.rank_before_rerank == 3
    assert out.rank_after_rerank == 1
    assert out.gold_in_input is True
    assert out.gold_was_demoted is False
    assert out.results[0].chunk_id == "c3"


def test_run_reranker_candidate_marks_demotion_when_gold_drops_past_final_k():
    """Gold at dense rank 1 (inside final_k=2); rerank pushes it to rank 4."""
    pool = [
        _chunk("c1", "d-target"),
        _chunk("c2", "d-other"),
        _chunk("c3", "d-other"),
        _chunk("c4", "d-other"),
    ]
    # Reranker punishes c1 (gold) — gives it the lowest score so it sinks.
    reranker = _ScoreTableReranker({
        "c1": 0.0, "c2": 0.9, "c3": 0.8, "c4": 0.7,
    })
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.rank_before_rerank == 1
    assert out.rank_after_rerank == -1  # gold dropped past final_k
    assert out.gold_in_input is True
    assert out.gold_was_demoted is True


def test_run_reranker_candidate_does_not_flag_demotion_when_gold_was_outside_final_k():
    """Gold at dense rank 4 (already outside final_k=2); reranker leaving it
    out is a recall ceiling, not a rerank defect."""
    pool = [
        _chunk("c1", "d-other"),
        _chunk("c2", "d-other"),
        _chunk("c3", "d-other"),
        _chunk("c4", "d-target"),
    ]
    reranker = _ScoreTableReranker({})  # all default → original order
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.rank_before_rerank == 4
    assert out.rank_after_rerank == -1
    assert out.gold_in_input is True
    assert out.gold_was_demoted is False  # not a demotion — ceiling


def test_run_reranker_candidate_handles_gold_missing_from_pool():
    pool = [_chunk("c1", "d-other"), _chunk("c2", "d-other")]
    cfg = RerankerAbConfig(candidate_k=2, final_k=2).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=_ScoreTableReranker({}),
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.rank_before_rerank == -1
    assert out.rank_after_rerank == -1
    assert out.gold_in_input is False
    assert out.gold_was_demoted is False


def test_run_reranker_candidate_empty_pool():
    cfg = RerankerAbConfig(candidate_k=2, final_k=2).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=[], reranker=_ScoreTableReranker({}),
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.results == []
    assert out.rank_before_rerank == -1
    assert out.rank_after_rerank == -1
    assert out.gold_in_input is False


def test_run_reranker_candidate_records_latency():
    pool = [_chunk("c1", "d-target")]
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    # Hand a fake monotonic clock so latency is deterministic.
    times = iter([100.0, 100.025])  # 25 ms
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=_ScoreTableReranker({}),
        config=cfg, expected_doc_ids=("d-target",),
        clock=lambda: next(times),
    )
    assert out.rerank_latency_ms == 25.0


# ---------------------------------------------------------------------------
# Score-mode behaviour
# ---------------------------------------------------------------------------


def test_score_mode_reranker_only_uses_rerank_score():
    """When dense puts gold last but rerank promotes it, reranker_only wins."""
    pool = [
        _chunk("c1", "d-other", score=0.9),
        _chunk("c2", "d-target", score=0.1),  # bad dense, good rerank
    ]
    reranker = _ScoreTableReranker({"c1": 0.0, "c2": 1.0})
    cfg = RerankerAbConfig(candidate_k=2, final_k=1,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.results[0].doc_id == "d-target"


def test_score_mode_weighted_dense_can_outrank_rerank():
    """With dense_weight=1, rerank_weight=0 the blend ignores rerank."""
    pool = [
        _chunk("c1", "d-target", score=0.9),
        _chunk("c2", "d-other", score=0.1),
    ]
    # Rerank wants to swap them; weight pins dense → no swap.
    reranker = _ScoreTableReranker({"c1": 0.0, "c2": 1.0})
    cfg = RerankerAbConfig(
        candidate_k=2, final_k=2,
        score_mode=SCORE_MODE_WEIGHTED,
        dense_weight=1.0, rerank_weight=0.0,
    ).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    assert out.results[0].chunk_id == "c1"
    assert out.results[1].chunk_id == "c2"


def test_score_mode_weighted_combines_signals():
    """Equal weights — both signals contribute. Rerank's clear winner takes top."""
    pool = [
        _chunk("c1", "d-other", score=0.5, rerank_score=None),
        _chunk("c2", "d-target", score=0.5, rerank_score=None),
        _chunk("c3", "d-other", score=0.5, rerank_score=None),
    ]
    reranker = _ScoreTableReranker({"c1": 0.0, "c2": 1.0, "c3": 0.5})
    cfg = RerankerAbConfig(
        candidate_k=3, final_k=3,
        score_mode=SCORE_MODE_WEIGHTED,
        dense_weight=0.5, rerank_weight=0.5,
    ).validate()
    out = run_reranker_candidate(
        "q", candidate_pool=pool, reranker=reranker,
        config=cfg, expected_doc_ids=("d-target",),
    )
    # Dense is constant → minmax = 0.5 each, so rerank decides ordering.
    assert [c.chunk_id for c in out.results] == ["c2", "c3", "c1"]


# ---------------------------------------------------------------------------
# summarize_latency
# ---------------------------------------------------------------------------


def test_summarize_latency_empty():
    s = summarize_latency([])
    assert s == LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_summarize_latency_singleton_collapses():
    s = summarize_latency([42.0])
    assert s.count == 1
    assert s.mean_ms == 42.0
    assert s.p50_ms == 42.0
    assert s.p90_ms == 42.0
    assert s.p99_ms == 42.0
    assert s.max_ms == 42.0


def test_summarize_latency_quantiles_reasonable():
    samples = list(range(1, 101))  # 1..100 ms
    s = summarize_latency(samples)
    assert s.count == 100
    # p50 should fall around 50 (linear interpolation midpoint of 50..51).
    assert 50.0 <= s.p50_ms <= 51.0
    # p90 should fall in the upper decile.
    assert 89.0 <= s.p90_ms <= 92.0
    assert 98.0 <= s.p99_ms <= 100.0
    assert s.max_ms == 100.0
    assert abs(s.mean_ms - 50.5) < 1e-3


# ---------------------------------------------------------------------------
# End-to-end run_reranker_ab
# ---------------------------------------------------------------------------


def test_run_reranker_ab_dense_baseline_preserved_with_noop_reranker():
    """With NoOp reranker, dense ordering survives → status stays both_hit/missed.

    This is the harness self-check the spec asks for: when the reranker
    is identity, no query can be misclassified as improved/regressed.
    """
    queries = [
        _q("q1", "q1-text", expected=("d-target",), bucket="main_work"),
        _q("q2", "q2-text", expected=("d-target",), bucket="subpage_generic"),
    ]
    base_pool = {
        "q1-text": [_chunk("c1", "d-target"), _chunk("c2", "d-other")],
        "q2-text": [_chunk("c3", "d-other"), _chunk("c4", "d-other")],
    }
    cand_pool = {
        # Candidate side gets a wider pool but the relative top-final_k
        # ordering of doc_ids must match base for noop semantics.
        "q1-text": [_chunk("c1", "d-target"), _chunk("c2", "d-other"),
                    _chunk("c5", "d-other")],
        "q2-text": [_chunk("c3", "d-other"), _chunk("c4", "d-other"),
                    _chunk("c6", "d-other")],
    }
    cfg = RerankerAbConfig(candidate_k=3, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    sc = result.aggregate["status_counts"]
    # No reranker ordering can make a query swap status under NoOp.
    assert sc["improved"] == 0
    assert sc["regressed"] == 0
    assert sc["both_hit"] + sc["both_missed"] + sc["unchanged"] == 2


def test_run_reranker_ab_reranker_can_improve():
    """A reranker that promotes gold flips a query from miss to hit."""
    queries = [
        _q("q1", "q1-text", expected=("d-target",), bucket="subpage_generic"),
    ]
    base_pool = {
        # Dense baseline: gold at rank 3 → outside final_k=2 (miss).
        "q1-text": [
            _chunk("c1", "d-other"),
            _chunk("c2", "d-other"),
        ],
    }
    cand_pool = {
        # Candidate side: wider pool, gold at rank 3 → reranker promotes.
        "q1-text": [
            _chunk("c1", "d-other"),
            _chunk("c2", "d-other"),
            _chunk("c3", "d-target"),
            _chunk("c4", "d-other"),
        ],
    }
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    reranker = _ScoreTableReranker({"c3": 1.0})
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool),
        reranker=reranker,
        config=cfg,
    )
    sc = result.aggregate["status_counts"]
    assert sc["improved"] == 1
    assert sc["regressed"] == 0


def test_run_reranker_ab_reranker_can_regress():
    """Reranker that demotes gold flips hit → miss; bookkeeping records it."""
    queries = [
        _q("q1", "q1-text", expected=("d-target",), bucket="main_work"),
    ]
    base_pool = {
        "q1-text": [_chunk("c1", "d-target"), _chunk("c2", "d-other")],
    }
    cand_pool = {
        "q1-text": [
            _chunk("c1", "d-target"),
            _chunk("c2", "d-other"),
            _chunk("c3", "d-other"),
            _chunk("c4", "d-other"),
        ],
    }
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    # Reranker wants other chunks at the top, sinks c1.
    reranker = _ScoreTableReranker({
        "c1": -10.0, "c2": 0.5, "c3": 0.4, "c4": 0.3,
    })
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool),
        reranker=reranker,
        config=cfg,
    )
    sc = result.aggregate["status_counts"]
    assert sc["regressed"] == 1
    # Per-query row should carry a non-None severity for the regression.
    row = result.per_query[0]
    assert row["status"] == "regressed"
    assert row["regression_severity"] is not None
    # gold_was_demoted: dense had c1 at rank 1 (within final_k); reranker dropped.
    assert row["candidate"]["gold_was_demoted"] is True


def test_run_reranker_ab_aggregates_across_buckets():
    """by_bucket should appear for every bucket present in the input set."""
    queries = [
        _q("q1", "q1-text", expected=("d-target",), bucket="main_work"),
        _q("q2", "q2-text", expected=("d-target",), bucket="subpage_generic"),
        _q("q3", "q3-text", expected=("d-target",), bucket="subpage_named"),
    ]
    pool = {q.query: [_chunk("c1", "d-target")] for q in queries}
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(pool),
        candidate_retriever=_ProgrammedRetriever(pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    by_bucket = result.aggregate["by_bucket"]
    assert set(by_bucket.keys()) == {
        "main_work", "subpage_generic", "subpage_named",
    }
    assert sum(b["count"] for b in by_bucket.values()) == 3


def test_run_reranker_ab_handles_query_with_no_expected_doc_ids():
    """A query with no gold can never be improved or regressed."""
    q = QueryRecord(
        qid="q1", query="q-orphan", expected_doc_ids=(),
        answer_type="?", difficulty="?", bucket="main_work",
        v4_meta={"bucket": "main_work"},
    )
    pool = {"q-orphan": [_chunk("c1", "d-other")]}
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    result = run_reranker_ab(
        [q],
        baseline_retriever=_ProgrammedRetriever(pool),
        candidate_retriever=_ProgrammedRetriever(pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    sc = result.aggregate["status_counts"]
    assert sc["improved"] == 0
    assert sc["regressed"] == 0
    # Both sides get rank=-1; the classifier's both_missed branch fires.
    assert sc["both_missed"] == 1


def test_run_reranker_ab_aggregate_carries_gold_input_counts():
    """gold_in_input / gold_was_demoted aggregate counts surface."""
    queries = [
        _q("q1", "q1-text", expected=("d-target",), bucket="main_work"),
        _q("q2", "q2-text", expected=("d-target",), bucket="main_work"),
    ]
    base_pool = {
        "q1-text": [_chunk("c1", "d-target")],  # in input
        "q2-text": [_chunk("c2", "d-other")],   # gold absent
    }
    cand_pool = {
        "q1-text": [_chunk("c1", "d-target")],
        "q2-text": [_chunk("c2", "d-other")],
    }
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    gic = result.aggregate["gold_input_counts"]
    assert gic["gold_in_input"] == 1
    assert gic["gold_was_demoted"] == 0


def test_run_reranker_ab_records_latency_summary():
    queries = [_q("q1", "q1-text", expected=("d-target",))]
    pool = {"q1-text": [_chunk("c1", "d-target")]}
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(pool),
        candidate_retriever=_ProgrammedRetriever(pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    lat = result.latency_summary
    assert lat["n_queries"] == 1
    assert lat["mean_ms"] >= 0.0
    assert lat["p50_ms"] >= 0.0
    assert lat["p90_ms"] >= 0.0
    assert lat["p99_ms"] >= 0.0
    assert lat["max_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Output writers — schema stability
# ---------------------------------------------------------------------------


def test_write_ab_outputs_emits_expected_artefacts(tmp_path: Path):
    queries = [
        _q("q-improved", "q1-text", expected=("d-target",),
           bucket="subpage_generic"),
        _q("q-regressed", "q2-text", expected=("d-target",),
           bucket="main_work"),
        _q("q-both-hit", "q3-text", expected=("d-target",),
           bucket="subpage_named"),
        _q("q-both-missed", "q4-text", expected=("d-target",),
           bucket="main_work"),
    ]
    base_pool = {
        # q-improved: dense rank 3 (miss with final_k=2)
        "q1-text": [_chunk("c1", "d-other"), _chunk("c2", "d-other")],
        # q-regressed: dense rank 1
        "q2-text": [_chunk("c-tgt2", "d-target"), _chunk("c2", "d-other")],
        # q-both-hit: dense rank 1
        "q3-text": [_chunk("c-tgt3", "d-target")],
        # q-both-missed: dense never has gold
        "q4-text": [_chunk("c-x1", "d-other")],
    }
    cand_pool = {
        # q-improved: gold appears at rank 3, reranker promotes
        "q1-text": [
            _chunk("c1", "d-other"),
            _chunk("c2", "d-other"),
            _chunk("c-tgt", "d-target"),
            _chunk("c4", "d-other"),
        ],
        # q-regressed: same pool but reranker punishes c-tgt2
        "q2-text": [
            _chunk("c-tgt2", "d-target"),
            _chunk("c2", "d-other"),
            _chunk("c3", "d-other"),
        ],
        # q-both-hit: identical
        "q3-text": [_chunk("c-tgt3", "d-target")],
        # q-both-missed: identical
        "q4-text": [_chunk("c-x1", "d-other")],
    }
    cfg = RerankerAbConfig(candidate_k=4, final_k=2,
                           score_mode=SCORE_MODE_RERANKER_ONLY).validate()
    reranker = _ScoreTableReranker({
        # q-improved promotion
        "c-tgt": 1.0,
        # q-regressed demotion
        "c-tgt2": -10.0, "c2": 0.5, "c3": 0.4,
    })
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool),
        reranker=reranker,
        config=cfg,
    )
    paths = write_ab_outputs(result, out_dir=tmp_path)

    # All expected files exist.
    for role in ("summary_json", "summary_md", "per_query",
                 "improved", "regressed", "latency"):
        assert paths[role].exists(), f"missing {role}"

    # summary_json schema: must carry config + status_counts + by_bucket +
    # gold_input_counts.
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert "config" in summary
    assert "status_counts" in summary
    assert "by_bucket" in summary
    assert "gold_input_counts" in summary
    assert summary["status_counts"]["improved"] == 1
    assert summary["status_counts"]["regressed"] == 1

    # per_query: each row carries baseline + candidate + diagnostics.
    rows = [
        json.loads(line) for line in paths["per_query"].read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    assert len(rows) == 4
    for r in rows:
        assert {"qid", "query", "expected_doc_ids", "bucket",
                "status", "baseline", "candidate",
                "candidate_pool_preview"} <= set(r.keys())
        for col in ("rank", "hit_at", "mrr_at_10",
                    "ndcg_at_10", "dup_rate"):
            assert col in r["baseline"]
            assert col in r["candidate"]
        # Phase 7.1 diagnostic columns on candidate.
        for col in ("rank_before_rerank", "rank_after_rerank",
                    "gold_in_input", "gold_was_demoted",
                    "rerank_latency_ms"):
            assert col in r["candidate"]

    # improved JSONL contains exactly 1 row
    improved = [
        json.loads(line) for line in paths["improved"].read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    assert len(improved) == 1
    assert improved[0]["qid"] == "q-improved"

    # regressed JSONL contains exactly 1 row
    regressed = [
        json.loads(line) for line in paths["regressed"].read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    assert len(regressed) == 1
    assert regressed[0]["qid"] == "q-regressed"
    # Regressed row carries the candidate pool preview for diagnosis.
    assert regressed[0]["candidate_pool_preview"]
    assert regressed[0]["regression_severity"] is not None

    # latency JSON schema
    latency = json.loads(paths["latency"].read_text(encoding="utf-8"))
    for key in ("config", "n_queries", "mean_ms", "p50_ms",
                "p90_ms", "p99_ms", "max_ms"):
        assert key in latency, f"latency missing {key}"

    # summary_md is non-empty and mentions the metric tables and the
    # bucket section so the human-readable report is intact.
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "Phase 7.1 reranker A/B" in md
    assert "Aggregate metrics" in md
    assert "By bucket" in md
    assert "Latency" in md


def test_write_ab_outputs_creates_dir(tmp_path: Path):
    nested = tmp_path / "nested" / "ab_out"
    queries = [_q("q1", "q1-text", expected=("d-target",))]
    pool = {"q1-text": [_chunk("c1", "d-target")]}
    cfg = RerankerAbConfig(candidate_k=1, final_k=1).validate()
    result = run_reranker_ab(
        queries,
        baseline_retriever=_ProgrammedRetriever(pool),
        candidate_retriever=_ProgrammedRetriever(pool),
        reranker=NoOpReranker(),
        config=cfg,
    )
    paths = write_ab_outputs(result, out_dir=nested)
    assert nested.exists()
    assert paths["summary_json"].exists()


# ---------------------------------------------------------------------------
# Integration: candidate_k probing
# ---------------------------------------------------------------------------


def test_candidate_k_widening_changes_gold_in_input_count():
    """At candidate_k=2 gold is missing; at candidate_k=4 gold is present.

    Pin the diagnostic so a future regression in the rank_before_rerank
    bookkeeping (e.g. accidentally truncating the pool to final_k before
    measuring) can't slip through.
    """
    q = _q("q1", "q1-text", expected=("d-target",))
    # Gold sits at dense rank 3 — visible only when candidate_k >= 3.
    full_pool = [
        _chunk("c1", "d-other"),
        _chunk("c2", "d-other"),
        _chunk("c3", "d-target"),
        _chunk("c4", "d-other"),
    ]
    base_pool = {"q1-text": full_pool[:2]}  # final_k=2 sees only top-2
    # Narrow candidate side
    cand_pool_narrow = {"q1-text": full_pool[:2]}
    # Wide candidate side
    cand_pool_wide = {"q1-text": full_pool[:4]}

    narrow_cfg = RerankerAbConfig(candidate_k=2, final_k=2).validate()
    wide_cfg = RerankerAbConfig(candidate_k=4, final_k=2).validate()
    reranker = _ScoreTableReranker({"c3": 1.0})  # boost gold if visible

    narrow_result = run_reranker_ab(
        [q],
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool_narrow),
        reranker=reranker,
        config=narrow_cfg,
    )
    wide_result = run_reranker_ab(
        [q],
        baseline_retriever=_ProgrammedRetriever(base_pool),
        candidate_retriever=_ProgrammedRetriever(cand_pool_wide),
        reranker=reranker,
        config=wide_cfg,
    )

    assert narrow_result.aggregate["gold_input_counts"]["gold_in_input"] == 0
    assert wide_result.aggregate["gold_input_counts"]["gold_in_input"] == 1
    # And the wide side improves, narrow doesn't.
    assert wide_result.aggregate["status_counts"]["improved"] == 1
    assert narrow_result.aggregate["status_counts"]["improved"] == 0
