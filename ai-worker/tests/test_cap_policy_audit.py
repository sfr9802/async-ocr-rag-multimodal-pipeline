"""Tests for the cap-policy audit adapter + score_audit_event helper.

Covers:
  - CapPolicyAuditAdapter records per-stage chunk lists on last_audit.
  - score_audit_event computes 1-based gold ranks at each stage.
  - gold_was_capped_out fires when gold is in the MMR pool but the cap
    drops it before the rerank-input slice.
  - Different cap policies surface different cap-out behaviour on the
    same input.
  - Retriever state restored after retrieve() — same finally-block
    contract as WideRetrievalEvalAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

from eval.harness.cap_policy import (
    DocIdCapPolicy,
    NoCapPolicy,
    SectionPathCapPolicy,
    TitleCapPolicy,
)
from eval.harness.cap_policy_audit import (
    CapPolicyAuditAdapter,
    CapPolicyAuditConfig,
    score_audit_event,
)


@dataclass
class _StubChunk:
    chunk_id: str
    doc_id: str
    section: str = "overview"
    text: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None
    title: Optional[str] = None


def _title_of(chunk: Any) -> Optional[str]:
    return getattr(chunk, "title", None)


class _StubRetriever:
    """Minimal stub matching the production Retriever mutator surface."""

    def __init__(self, pool: List[_StubChunk]):
        self._pool = list(pool)
        self._top_k = 10
        self._candidate_k = 10
        self._reranker = None
        self._use_mmr = False
        self._mmr_lambda = 0.7

    def retrieve(self, query: str):
        return SimpleNamespace(
            results=self._pool[: self._top_k],
            index_version="stub-v1",
            embedding_model="stub-model",
            reranker_name="noop",
            rerank_ms=None,
            dense_retrieval_ms=12.0,
            rerank_breakdown_ms=None,
            candidate_doc_ids=[],
        )


class _StubReranker:
    """Identity reranker — returns chunks as supplied (no reorder)."""

    name = "stub-identity"
    last_breakdown_ms = None

    def rerank(self, query, chunks, k):
        return list(chunks)[:k]


# ---------------------------------------------------------------------------
# Pool / MMR / cap / rerank pipeline records
# ---------------------------------------------------------------------------


def _pool() -> List[_StubChunk]:
    # 12 chunks across 4 doc_ids with 2 titles, decreasing relevance.
    return [
        _StubChunk("c01", "doc-A", title="Series-1", score=1.00),
        _StubChunk("c02", "doc-A", title="Series-1", score=0.99),
        _StubChunk("c03", "doc-A", title="Series-1", score=0.98),
        _StubChunk("c04", "doc-B", title="Series-1", score=0.97),
        _StubChunk("c05", "doc-B", title="Series-1", score=0.96),
        _StubChunk("c06", "doc-C", title="Series-2", score=0.95),
        _StubChunk("c07", "doc-C", title="Series-2", score=0.94),
        _StubChunk("c08", "doc-D", title="Series-3", score=0.93),
        _StubChunk("c09", "doc-D", title="Series-3", score=0.92),
        _StubChunk("c10", "doc-D", title="Series-3", score=0.91),
        _StubChunk("c11", "doc-A", title="Series-1", score=0.90),
        _StubChunk("c12", "doc-B", title="Series-1", score=0.89),
    ]


class TestAuditAdapterPipeline:
    def test_audit_records_each_stage(self):
        retriever = _StubRetriever(_pool())
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=TitleCapPolicy(
                    1, title_provider=_title_of,
                ),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        report = adapter.retrieve("q")
        audit = adapter.last_audit
        assert audit is not None
        assert len(audit.pool) == 12
        # MMR off → mmr_pool == pool snapshot.
        assert len(audit.mmr_pool) == 12
        # title_cap=1 over Series-1/2/3 → 3 chunks (c01, c06, c08).
        assert {c.chunk_id for c in audit.after_cap_rerank_input} == {
            "c01", "c06", "c08",
        }
        # rerank_in=8 doesn't bite; rerank_input == after_cap.
        assert audit.rerank_input == audit.after_cap_rerank_input
        # Identity reranker preserves order.
        assert [c.chunk_id for c in audit.reranked] == [
            c.chunk_id for c in audit.rerank_input
        ]
        # Final has cap_policy_final=None → after_cap_final == reranked.
        # final_top_k=5 truncates to 3 in this case (only 3 survived).
        assert len(audit.final) == 3
        assert [c.chunk_id for c in report.results] == [
            c.chunk_id for c in audit.final
        ]

    def test_doc_id_cap_does_not_collapse_franchise(self):
        retriever = _StubRetriever(_pool())
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=8,
                rerank_in=12,
                cap_policy_rerank_input=DocIdCapPolicy(1),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        adapter.retrieve("q")
        audit = adapter.last_audit
        # cap=1 per doc_id with 4 docs → 4 chunks (one per doc).
        assert len(audit.after_cap_rerank_input) == 4
        ids = {c.chunk_id for c in audit.after_cap_rerank_input}
        # Order-preserving: first chunk per doc_id wins.
        assert ids == {"c01", "c04", "c06", "c08"}

    def test_no_cap_lets_all_through(self):
        retriever = _StubRetriever(_pool())
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=8,
                rerank_in=12,
                cap_policy_rerank_input=NoCapPolicy(),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        adapter.retrieve("q")
        audit = adapter.last_audit
        # NoCap → after_cap == mmr_pool (12 chunks).
        assert len(audit.after_cap_rerank_input) == 12

    def test_section_path_cap_groups_by_doc_section(self):
        chunks = [
            _StubChunk("c1", "doc-A", section="요약"),
            _StubChunk("c2", "doc-A", section="요약"),
            _StubChunk("c3", "doc-A", section="본문"),
            _StubChunk("c4", "doc-A", section="본문"),
            _StubChunk("c5", "doc-B", section="요약"),
        ]
        retriever = _StubRetriever(chunks)
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=10,
                final_top_k=8,
                rerank_in=8,
                cap_policy_rerank_input=SectionPathCapPolicy(1),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
        )
        adapter.retrieve("q")
        audit = adapter.last_audit
        # cap=1 per (doc, section) → 3 chunks (c1, c3, c5).
        assert [c.chunk_id for c in audit.after_cap_rerank_input] == [
            "c1", "c3", "c5",
        ]

    def test_retriever_state_restored(self):
        retriever = _StubRetriever(_pool())
        retriever._top_k = 7
        retriever._candidate_k = 7
        retriever._use_mmr = True
        retriever._mmr_lambda = 0.55
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=NoCapPolicy(),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
        )
        adapter.retrieve("q")
        # All four mutated attributes restored.
        assert retriever._top_k == 7
        assert retriever._candidate_k == 7
        assert retriever._use_mmr is True
        assert retriever._mmr_lambda == 0.55


# ---------------------------------------------------------------------------
# score_audit_event
# ---------------------------------------------------------------------------


class TestScoreAuditEvent:
    def test_gold_present_throughout(self):
        retriever = _StubRetriever(_pool())
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=NoCapPolicy(),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
        )
        adapter.retrieve("q")
        scored = score_audit_event(
            adapter.last_audit,
            expected_doc_ids=["doc-A"],
            cap_policy=NoCapPolicy(),
        )
        # doc-A appears at rank 1 in pool / mmr_pool / rerank_input / final.
        assert scored["gold_dense_rank"] == 1
        assert scored["gold_after_mmr_rank"] == 1
        assert scored["gold_rerank_input_rank"] == 1
        assert scored["gold_final_rank"] == 1
        assert scored["gold_was_in_dense_pool"] is True
        assert scored["gold_was_capped_out"] is False
        assert scored["capped_out_by_group_key"] == ""

    def test_gold_was_capped_out_when_cap_drops(self):
        # Build a pool where gold is at rank 2 within doc-A; title cap=1
        # keeps c01 (also doc-A), drops c11. Gold = c11 (still doc-A so
        # gold doc IS in cap-kept, but we want a case where gold doc is
        # not the first in its group). Simulate by making the gold's
        # doc_id only present in deep ranks of the pool.
        chunks = [
            _StubChunk("c01", "doc-A", title="Series-1", score=1.00),
            _StubChunk("c02", "doc-A", title="Series-1", score=0.99),
            _StubChunk("c03", "doc-A", title="Series-1", score=0.98),
            # gold is doc-X, but doc-X only has chunks beyond cap=2 of
            # title=Series-1: position 4 (still title=Series-1 because
            # of franchise grouping by title_provider).
            _StubChunk("c04", "doc-X", title="Series-1", score=0.97),
        ]
        retriever = _StubRetriever(chunks)
        cap_policy = TitleCapPolicy(2, title_provider=_title_of)
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=10,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=cap_policy,
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        adapter.retrieve("q")
        scored = score_audit_event(
            adapter.last_audit,
            expected_doc_ids=["doc-X"],
            cap_policy=cap_policy,
        )
        # title cap=2 over Series-1 keeps c01 and c02 (both doc-A);
        # c03 / c04 are dropped. Gold doc-X is at pool rank 4; cap drops
        # it → gold_was_capped_out=True.
        assert scored["gold_dense_rank"] == 4
        assert scored["gold_after_mmr_rank"] == 4
        assert scored["gold_rerank_input_rank"] is None
        assert scored["gold_was_capped_out"] is True
        # Group key is the title (casefolded).
        assert scored["capped_out_by_group_key"] == "series-1"
        assert scored["capped_out_group_size"] == 4

    def test_gold_not_in_pool_at_all(self):
        retriever = _StubRetriever(_pool())
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=12,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=NoCapPolicy(),
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
        )
        adapter.retrieve("q")
        scored = score_audit_event(
            adapter.last_audit,
            expected_doc_ids=["doc-NONE"],
            cap_policy=NoCapPolicy(),
        )
        assert scored["gold_was_in_dense_pool"] is False
        assert scored["gold_was_in_mmr_pool"] is False
        assert scored["gold_was_in_rerank_input"] is False
        # cap_out=False since gold was never in MMR pool either.
        assert scored["gold_was_capped_out"] is False

    def test_doc_id_cap_does_not_capped_out_for_franchise_sibling(self):
        # Doc-X exists separately from same-titled doc-A. DocIdCapPolicy
        # treats them independently → gold_was_capped_out stays False.
        chunks = [
            _StubChunk("c01", "doc-A", title="Series-1", score=1.00),
            _StubChunk("c02", "doc-A", title="Series-1", score=0.99),
            _StubChunk("c03", "doc-A", title="Series-1", score=0.98),
            _StubChunk("c04", "doc-X", title="Series-1", score=0.97),
        ]
        retriever = _StubRetriever(chunks)
        cap_policy = DocIdCapPolicy(2)
        adapter = CapPolicyAuditAdapter(
            retriever,
            config=CapPolicyAuditConfig(
                candidate_k=10,
                final_top_k=5,
                rerank_in=8,
                cap_policy_rerank_input=cap_policy,
                use_mmr=False,
            ),
            final_reranker=_StubReranker(),
            title_provider=_title_of,
        )
        adapter.retrieve("q")
        scored = score_audit_event(
            adapter.last_audit,
            expected_doc_ids=["doc-X"],
            cap_policy=cap_policy,
        )
        # doc-A capped at 2 (c01, c02). doc-X has 1 chunk → kept.
        assert scored["gold_after_cap_rerank_input_rank"] is not None
        assert scored["gold_was_capped_out"] is False
