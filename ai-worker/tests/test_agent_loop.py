"""AgentLoopController tests.

Covers every ``stop_reason`` branch via deterministic fakes so no
network or model access is required:

  * ``converged`` at iter=0 — critic says sufficient with confidence >=
    min_stop_conf, rewriter is never consulted.
  * ``iter_cap`` — critic never converges, loop runs max_iter passes.
  * ``time_cap`` — a slow fake execute pushes total_ms past
    max_total_ms.
  * ``token_cap`` — executor reports large token counts each step.
  * ``unanswerable`` — critic emits gap_type='unanswerable' at iter=0,
    loop short-circuits.
  * aggregated_chunks dedup across iters verified when the executor
    returns overlapping chunk_ids.
  * AGENT_TRACE JSON-serializable (ensure_ascii=False for Korean).
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple

import pytest

from app.capabilities.agent.critic import (
    AgentCriticProvider,
    CritiqueResult,
    NoOpCritic,
    RuleCritic,
)
from app.capabilities.agent.loop import (
    AgentLoopController,
    LoopBudget,
    LoopOutcome,
    LoopStep,
)
from app.capabilities.agent.rewriter import (
    NoOpQueryRewriter,
    QueryRewriterProvider,
)
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StaticCritic(AgentCriticProvider):
    """Critic that returns a canned verdict per iter from a queue."""

    def __init__(self, verdicts: List[CritiqueResult]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    @property
    def name(self) -> str:
        return "static"

    def evaluate(self, *, question, answer, retrieved):
        idx = min(self.calls, len(self._verdicts) - 1)
        self.calls += 1
        return self._verdicts[idx]


class _CountingRewriter(QueryRewriterProvider):
    """Rewriter that returns a new ParsedQuery per call; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting"

    def rewrite(
        self,
        *,
        original,
        prev_answer,
        gap_reason,
        already_retrieved_chunks,
        parser,
    ):
        self.calls += 1
        return parser.parse(f"{original} iter{self.calls}")


def _mk_chunk(cid: str, doc: str = "doc-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc, section="", text=f"text-{cid}", score=0.9,
    )


def _verdict(
    *,
    sufficient: bool,
    gap_type: str = "none",
    confidence: float = 1.0,
    reason: str = "r",
    tokens: int = 0,
) -> CritiqueResult:
    if sufficient and gap_type == "none":
        return CritiqueResult(
            sufficient=True,
            gap_type="none",
            gap_reason=reason,
            confidence=confidence,
            critic_name="static",
            llm_tokens_used=tokens,
        )
    return CritiqueResult(
        sufficient=False,
        gap_type=gap_type if gap_type != "none" else "missing_facts",
        gap_reason=reason,
        confidence=confidence,
        critic_name="static",
        llm_tokens_used=tokens,
    )


# ---------------------------------------------------------------------------
# 1. converged at iter=0
# ---------------------------------------------------------------------------


def test_loop_converges_at_iter_0_when_critic_sufficient():
    critic = _StaticCritic([
        _verdict(sufficient=True, confidence=0.9),
    ])
    rewriter = _CountingRewriter()
    parser = NoOpQueryParser()
    budget = LoopBudget(max_iter=3, min_confidence_to_stop=0.75)

    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    calls = []

    def execute_fn(pq):
        calls.append(pq)
        return "a sufficient answer", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )

    assert outcome.stop_reason == "converged"
    assert len(outcome.steps) == 1
    assert len(calls) == 1
    assert rewriter.calls == 0  # never asked for a rewrite


def test_loop_does_not_converge_below_min_confidence():
    """Critic says sufficient but confidence below threshold -> keep looping."""
    critic = _StaticCritic([
        _verdict(sufficient=True, confidence=0.5),   # below 0.75 threshold
        _verdict(sufficient=True, confidence=0.9),   # passes threshold
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    budget = LoopBudget(max_iter=3, min_confidence_to_stop=0.75)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "answer", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "converged"
    assert len(outcome.steps) == 2
    assert rewriter.calls == 1


# ---------------------------------------------------------------------------
# 2. iter_cap
# ---------------------------------------------------------------------------


def test_loop_hits_iter_cap_when_critic_never_converges():
    critic = _StaticCritic([
        _verdict(sufficient=False, gap_type="missing_facts"),
        _verdict(sufficient=False, gap_type="missing_facts"),
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    budget = LoopBudget(max_iter=2)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "answer", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "iter_cap"
    assert len(outcome.steps) == 2
    # Rewriter is called between iter 0 and iter 1 — once.
    assert rewriter.calls == 1


# ---------------------------------------------------------------------------
# 3. time_cap
# ---------------------------------------------------------------------------


def test_loop_hits_time_cap_when_wall_clock_exceeds_budget():
    critic = _StaticCritic([
        _verdict(sufficient=False, gap_type="missing_facts") for _ in range(5)
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    # Tight wall-clock budget.
    budget = LoopBudget(max_iter=10, max_total_ms=50)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def slow_execute(pq):
        time.sleep(0.08)  # 80ms per iter — blows the 50ms cap after iter 0.
        return "answer", [_mk_chunk(f"c-{pq.normalized}")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=slow_execute,
    )
    assert outcome.stop_reason == "time_cap"
    assert len(outcome.steps) >= 1


# ---------------------------------------------------------------------------
# 4. token_cap
# ---------------------------------------------------------------------------


def test_loop_hits_token_cap_when_tokens_cross_budget():
    critic = _StaticCritic([
        _verdict(sufficient=False, gap_type="missing_facts", tokens=100)
        for _ in range(5)
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    # Cap below one iter's execute cost so iter 0 trips the check.
    budget = LoopBudget(max_iter=5, max_llm_tokens=150)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "answer", [_mk_chunk("c1")], 1000  # blow the cap

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "token_cap"
    assert len(outcome.steps) == 1


# ---------------------------------------------------------------------------
# 5. unanswerable
# ---------------------------------------------------------------------------


def test_loop_short_circuits_on_unanswerable_gap_type():
    critic = _StaticCritic([
        _verdict(sufficient=False, gap_type="unanswerable",
                 reason="corpus has no data on X"),
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    budget = LoopBudget(max_iter=3)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "I don't know", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "unanswerable"
    assert len(outcome.steps) == 1
    assert rewriter.calls == 0


# ---------------------------------------------------------------------------
# 6. aggregated_chunks dedup across iters
# ---------------------------------------------------------------------------


def test_aggregated_chunks_dedupe_across_iters_by_chunk_id():
    """Overlapping chunk_ids across iters collapse to one entry each."""
    critic = _StaticCritic([
        _verdict(sufficient=False, gap_type="missing_facts"),
        _verdict(sufficient=False, gap_type="missing_facts"),
        _verdict(sufficient=True, confidence=0.9),  # converge on iter 2
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    budget = LoopBudget(max_iter=5)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    iter_counter = {"n": 0}

    def execute_fn(pq):
        iter_counter["n"] += 1
        if iter_counter["n"] == 1:
            chunks = [_mk_chunk("c1"), _mk_chunk("c2")]
        elif iter_counter["n"] == 2:
            # Overlaps c2 (already seen), brings fresh c3.
            chunks = [_mk_chunk("c2"), _mk_chunk("c3")]
        else:
            # Brings c4 only.
            chunks = [_mk_chunk("c4")]
        return "answer", chunks, 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "converged"
    ids = [c.chunk_id for c in outcome.aggregated_chunks]
    assert ids == ["c1", "c2", "c3", "c4"]


# ---------------------------------------------------------------------------
# 7. AGENT_TRACE JSON-serializable, ensure_ascii=False for Korean.
# ---------------------------------------------------------------------------


def test_loop_outcome_json_serializable_with_korean():
    critic = _StaticCritic([
        CritiqueResult(
            sufficient=False,
            gap_type="missing_facts",
            gap_reason="정보가 부족합니다",
            confidence=0.6,
            critic_name="static",
            llm_tokens_used=12,
        ),
        CritiqueResult(
            sufficient=True,
            gap_type="none",
            gap_reason="충분합니다",
            confidence=0.95,
            critic_name="static",
            llm_tokens_used=10,
        ),
    ])
    rewriter = _CountingRewriter()
    parser = RegexQueryParser()
    budget = LoopBudget(max_iter=3)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "한국어 답변입니다", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="질문이 있습니다",
        initial_parsed_query=parser.parse("질문이 있습니다"),
        execute_fn=execute_fn,
    )
    body = outcome.to_dict()
    serialized = json.dumps(body, ensure_ascii=False)
    assert "정보가 부족합니다" in serialized
    assert "충분합니다" in serialized
    # Default ensure_ascii=True would escape the Hangul; fail loudly
    # if that escape shows up in the ensure_ascii=False serialization.
    assert "\\u" not in serialized


# ---------------------------------------------------------------------------
# Budget value-object contract
# ---------------------------------------------------------------------------


def test_loop_budget_rejects_bad_values():
    with pytest.raises(ValueError):
        LoopBudget(max_iter=0)
    with pytest.raises(ValueError):
        LoopBudget(max_total_ms=0)
    with pytest.raises(ValueError):
        LoopBudget(max_llm_tokens=-1)
    with pytest.raises(ValueError):
        LoopBudget(min_confidence_to_stop=1.5)


def test_loop_step_to_dict_round_trips_answer_preview():
    step = LoopStep(
        iter=0,
        query=RegexQueryParser().parse("hello world"),
        retrieved_chunk_ids=["c1", "c2"],
        answer_preview="preview",
        critique=_verdict(sufficient=True, confidence=0.9),
        step_ms=1.234,
        llm_tokens_used=5,
    )
    body = step.to_dict()
    assert body["iter"] == 0
    assert body["retrievedChunkIds"] == ["c1", "c2"]
    assert body["answerPreview"] == "preview"
    assert body["critique"]["sufficient"] is True
    assert body["stepMs"] == pytest.approx(1.234)


# ---------------------------------------------------------------------------
# Composition with NoOpCritic => single-pass
# ---------------------------------------------------------------------------


def test_noop_critic_makes_loop_single_pass():
    """The loop with a NoOpCritic converges at iter=0 unconditionally."""
    critic = NoOpCritic()
    rewriter = NoOpQueryRewriter()
    parser = NoOpQueryParser()
    budget = LoopBudget(max_iter=5, min_confidence_to_stop=0.75)
    ctrl = AgentLoopController(
        critic=critic, rewriter=rewriter, parser=parser, budget=budget
    )

    def execute_fn(pq):
        return "any answer", [_mk_chunk("c1")], 0

    outcome = ctrl.run(
        question="q",
        initial_parsed_query=parser.parse("q"),
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "converged"
    assert len(outcome.steps) == 1
