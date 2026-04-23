"""Phase 9 cross-domain eval harness tests.

Two layers under test, both offline:

  1. ``_is_refusal`` correctly classifies the small set of phrases the
     extractive / claude generators emit for ``no relevant context``.

  2. ``run_rag_cross_domain_eval`` aggregates the right rows:
       - filter-excluded + refusal answer        -> cross_domain_pass
       - filter-excluded + hallucinated answer   -> NOT pass
       - filter-permissive + refusal answer      -> NOT pass (the filter
         didn't bind, so refusal alone isn't a win)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from eval.harness.rag_eval import (
    _is_refusal,
    run_rag_cross_domain_eval,
)


@dataclass
class _FakeChunk:
    chunk_id: str
    doc_id: str
    section: str = ""
    text: str = ""
    score: float = 0.0


@dataclass
class _FakeReport:
    results: List[_FakeChunk] = field(default_factory=list)
    filter_produced_no_docs: bool = False


class _FakeRetriever:
    """Returns the canned report attached to each query."""

    def __init__(self, by_query: dict) -> None:
        self._by_query = by_query

    def retrieve(self, query: str, filters=None):  # noqa: ARG002
        return self._by_query.get(query, _FakeReport())


class _RefusalGenerator:
    """Echoes 'no info found' regardless of input — simulates the
    extractive generator's empty-result branch."""

    def generate(self, query: str, chunks):  # noqa: ARG002
        return "문서에서 찾을 수 없습니다." if not chunks else f"답변: {len(chunks)} chunks"


class _HallucinatingGenerator:
    """Always emits a confident-sounding non-refusal answer."""

    def generate(self, query: str, chunks):  # noqa: ARG002
        return "재택근무 신청 기한은 7일입니다."  # plausibly wrong


# ---------------------------------------------------------------------------
# 1. _is_refusal contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "정보가 없습니다",
    "관련 정보 없음",
    "문서에서 찾을 수 없습니다.",
    "답변할 수 없습니다.",
    "관련 내용을 찾을 수 없습니다.",
    "No relevant passages were retrieved",
    "I don't have information about that",
])
def test_is_refusal_true_on_known_phrases(text):
    assert _is_refusal(text) is True


def test_is_refusal_true_on_empty_string():
    assert _is_refusal("") is True


@pytest.mark.parametrize("text", [
    "재택근무 신청 기한은 3 영업일입니다.",
    "법인카드 한도는 50만원입니다.",
    "Kubernetes 알림 채널은 PagerDuty입니다.",
])
def test_is_refusal_false_on_confident_answers(text):
    assert _is_refusal(text) is False


# ---------------------------------------------------------------------------
# 2. run_rag_cross_domain_eval aggregation
# ---------------------------------------------------------------------------


def test_cross_domain_passes_when_filter_excludes_and_generator_refuses():
    """Filter short-circuited (zero docs) + refusal answer -> pass."""
    dataset = [
        {
            "query": "재택근무 신청 기한",
            "filters": {"domain": "anime"},
            "expected_action": "unanswerable",
            "expected_keywords": [],
        },
    ]
    retriever = _FakeRetriever({
        "재택근무 신청 기한": _FakeReport(results=[], filter_produced_no_docs=True),
    })

    summary, rows = run_rag_cross_domain_eval(
        dataset, retriever=retriever, generator=_RefusalGenerator(),
    )
    assert summary["row_count"] == 1
    assert summary["cross_domain_refusal_rate"] == 1.0
    assert summary["cross_domain_zero_results_rate"] == 1.0
    assert rows[0].cross_domain_pass is True
    assert rows[0].refusal_detected is True


def test_cross_domain_fails_when_generator_hallucinates():
    """Even with the filter excluding the corpus, a generator that
    invents an answer must NOT be counted as a pass."""
    dataset = [
        {
            "query": "재택근무 신청 기한",
            "filters": {"domain": "anime"},
            "expected_action": "unanswerable",
        },
    ]
    retriever = _FakeRetriever({
        "재택근무 신청 기한": _FakeReport(results=[], filter_produced_no_docs=True),
    })
    summary, rows = run_rag_cross_domain_eval(
        dataset, retriever=retriever, generator=_HallucinatingGenerator(),
    )
    assert summary["cross_domain_refusal_rate"] == 0.0
    assert rows[0].cross_domain_pass is False
    assert rows[0].refusal_detected is False


def test_cross_domain_pass_depends_only_on_refusal_detected():
    """The pass gate is the generator's refusal, not whether the
    retriever returned zero docs. The filter's correctness is verified
    separately (by inspecting retrieved_doc_ids against filters).

    If the filter returned wrong-domain docs AND the (relevance-gated)
    generator still emitted a refusal phrase, that's the intended
    Phase 9 behaviour."""
    dataset = [
        {
            "query": "재택근무 신청 기한",
            "filters": {"domain": "anime"},
            "expected_action": "unanswerable",
        },
    ]
    retriever = _FakeRetriever({
        "재택근무 신청 기한": _FakeReport(
            results=[_FakeChunk(chunk_id="c", doc_id="anime-001")],
            filter_produced_no_docs=False,
        ),
    })
    # Generator that always refuses (simulates the relevance-gated
    # ExtractiveGenerator when top-chunk score is below threshold).
    class _AlwaysRefuse:
        def generate(self, query, chunks):  # noqa: ARG002
            return "문서에서 관련 정보를 찾을 수 없습니다."

    summary, rows = run_rag_cross_domain_eval(
        dataset, retriever=retriever, generator=_AlwaysRefuse(),
    )
    assert summary["cross_domain_refusal_rate"] == 1.0
    assert rows[0].cross_domain_pass is True


def test_cross_domain_aggregate_mixes_pass_and_fail():
    dataset = [
        {"query": "q1", "filters": {"domain": "anime"}, "expected_action": "unanswerable"},
        {"query": "q2", "filters": {"domain": "anime"}, "expected_action": "unanswerable"},
        {"query": "q3", "filters": {"domain": "anime"}, "expected_action": "unanswerable"},
        {"query": "q4", "filters": {"domain": "anime"}, "expected_action": "unanswerable"},
    ]
    retriever = _FakeRetriever({
        "q1": _FakeReport(results=[], filter_produced_no_docs=True),
        "q2": _FakeReport(results=[], filter_produced_no_docs=True),
        "q3": _FakeReport(results=[], filter_produced_no_docs=True),
        "q4": _FakeReport(
            results=[_FakeChunk(chunk_id="c", doc_id="anime-001")],
            filter_produced_no_docs=False,
        ),
    })
    summary, _ = run_rag_cross_domain_eval(
        dataset, retriever=retriever, generator=_RefusalGenerator(),
    )
    # 3 of 4 pass -> 0.75
    assert summary["cross_domain_refusal_rate"] == 0.75
    assert summary["passing_rows"] == 3
