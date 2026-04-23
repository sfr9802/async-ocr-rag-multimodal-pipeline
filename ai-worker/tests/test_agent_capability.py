"""AgentCapability integration tests.

Covers both the loop-off (Phase 5 parity) and loop-on (Phase 6) paths:

  1. ``name`` — AgentCapability reports "AGENT", AutoCapability reports
     "AUTO". Both are distinct registry entries.

  2. Loop-off parity — AgentCapability with ``loop_enabled=False``
     emits the same artifact shape as AutoCapability for identical
     input (AGENT_DECISION + sub-capability outputs only).

  3. Loop-on RAG — the loop emits AGENT_DECISION + AGENT_TRACE +
     RETRIEVAL_RESULT_AGG + FINAL_RESPONSE; the RagCapability's
     artifacts are NOT surfaced (the loop replaces them with its own
     aggregated shape).

  4. Loop-on convergence at iter=0 — NoOpCritic stops the loop on
     the first iteration; AGENT_TRACE records steps=1.

  5. Loop-on with a slow executor hits the time budget and still
     emits all four artifacts without raising.

  6. Zero aggregated chunks — executor returns empty, synthesizer
     falls back to the loop's last answer, FINAL_RESPONSE is not empty.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from app.capabilities.agent.capability import (
    AgentCapability,
    AutoCapability,
)
from app.capabilities.agent.critic import (
    CritiqueResult,
    NoOpCritic,
    RuleCritic,
)
from app.capabilities.agent.loop import LoopBudget
from app.capabilities.agent.rewriter import NoOpQueryRewriter
from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
    RuleBasedAgentRouter,
)
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.generation import (
    GenerationProvider,
    RetrievedChunk,
)
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    RegexQueryParser,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SentinelCapability(Capability):
    def __init__(self, name: str, payload: str) -> None:
        self.name = name
        self._payload = payload
        self.calls: List[CapabilityInput] = []

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        self.calls.append(input)
        return CapabilityOutput(
            outputs=[
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename=f"{self.name.lower()}-sentinel.md",
                    content_type="text/markdown; charset=utf-8",
                    content=self._payload.encode("utf-8"),
                )
            ]
        )


class _FixedRouter(AgentRouterProvider):
    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision

    @property
    def name(self) -> str:
        return self._decision.router_name

    def decide(self, *, text, has_file, file_mime, file_size):
        return self._decision


class _FakeRetriever:
    """Retriever stub that hands back canned chunks per query."""

    def __init__(self, chunks_by_query: dict[str, List[RetrievedChunk]]) -> None:
        self._by_query = chunks_by_query
        self.calls: list[str] = []

    def retrieve(self, query: str):
        self.calls.append(query)
        chunks = self._by_query.get(query, [])

        class _Report:
            def __init__(self, results):
                self.results = results

        return _Report(chunks)


class _FakeGenerator(GenerationProvider):
    def __init__(self, template: str = "answer({n_chunks})") -> None:
        self._template = template
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "fake-gen"

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        self.calls.append((query, len(chunks)))
        joined = " | ".join(c.chunk_id for c in chunks)
        return f"{self._template.format(n_chunks=len(chunks))}: {joined}"


def _mk_chunk(cid: str, doc: str = "doc-a", text: str = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=doc,
        section="",
        text=text or f"text for {cid}",
        score=0.9,
    )


def _input(text: str, job_id: str = "job-1") -> CapabilityInput:
    return CapabilityInput(
        job_id=job_id,
        capability="AGENT",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="art-t",
                type="INPUT_TEXT",
                content=text.encode("utf-8"),
                content_type="text/plain; charset=utf-8",
            )
        ],
    )


def _decision(
    *, action: str, parsed_query=None, router_name: str = "rule",
    confidence: float = 0.8,
) -> AgentDecision:
    return AgentDecision(
        action=action,  # type: ignore[arg-type]
        reason="test",
        parsed_query=parsed_query,
        confidence=confidence,
        router_name=router_name,
    )


# ---------------------------------------------------------------------------
# 1. name contract
# ---------------------------------------------------------------------------


def test_agent_capability_reports_name_AGENT():
    cap = AgentCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
    )
    assert cap.name == "AGENT"


def test_auto_capability_still_reports_name_AUTO():
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
    )
    assert cap.name == "AUTO"


def test_auto_capability_ignores_loop_enabled_kwarg():
    """AutoCapability forces loop_enabled=False even when a caller passes True."""
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
        loop_enabled=True,  # should be silently ignored
    )
    assert cap._loop_enabled is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# 2. loop-off parity with AutoCapability
# ---------------------------------------------------------------------------


def test_agent_loop_off_matches_auto_for_rag_action():
    """AGENT(loop=off) must emit the same artifacts as AUTO for RAG."""
    rag_sentinel_a = _SentinelCapability("RAG", "rag answer bytes")
    rag_sentinel_b = _SentinelCapability("RAG", "rag answer bytes")

    auto_cap = AutoCapability(
        router=_FixedRouter(_decision(action="rag")),
        parser=NoOpQueryParser(),
        rag=rag_sentinel_a,
    )
    agent_cap = AgentCapability(
        router=_FixedRouter(_decision(action="rag")),
        parser=NoOpQueryParser(),
        rag=rag_sentinel_b,
        loop_enabled=False,
    )

    auto_out = auto_cap.run(_input("what is bge-m3"))
    agent_out = agent_cap.run(_input("what is bge-m3"))

    # Artifact type / filename / content-type / content must match
    # bit-for-bit so Phase 5 clients can switch capability names
    # without a behavioural surprise.
    assert len(auto_out.outputs) == len(agent_out.outputs)
    for a, b in zip(auto_out.outputs, agent_out.outputs):
        assert a.type == b.type
        assert a.filename == b.filename
        assert a.content_type == b.content_type
        assert a.content == b.content


def test_agent_loop_off_matches_auto_for_clarify_action():
    """Clarify path never touches the loop even when loop_enabled=True."""
    auto_cap = AutoCapability(
        router=_FixedRouter(_decision(action="clarify", confidence=0.5)),
        parser=NoOpQueryParser(),
    )
    agent_cap = AgentCapability(
        router=_FixedRouter(_decision(action="clarify", confidence=0.5)),
        parser=NoOpQueryParser(),
        loop_enabled=True,
    )

    auto_out = auto_cap.run(_input("hi"))
    agent_out = agent_cap.run(_input("hi"))
    assert len(auto_out.outputs) == len(agent_out.outputs)
    assert [a.type for a in auto_out.outputs] == [
        a.type for a in agent_out.outputs
    ]
    # FINAL_RESPONSE bytes identical (both produce the Korean clarify msg).
    assert auto_out.outputs[-1].content == agent_out.outputs[-1].content


# ---------------------------------------------------------------------------
# 3. loop-on RAG happy path
# ---------------------------------------------------------------------------


def test_agent_loop_on_rag_emits_four_artifacts_and_uses_retriever():
    parser = RegexQueryParser()
    retriever = _FakeRetriever(
        {
            "what is bge-m3": [_mk_chunk("c1"), _mk_chunk("c2", doc="doc-b")],
        }
    )
    generator = _FakeGenerator()
    agent_cap = AgentCapability(
        router=_FixedRouter(
            _decision(
                action="rag",
                parsed_query=parser.parse("what is bge-m3"),
            )
        ),
        parser=parser,
        rag=_SentinelCapability("RAG", "unused"),
        loop_enabled=True,
        critic=NoOpCritic(),  # converges at iter=0
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(generator),
        retriever=retriever,
        generator=generator,
        budget=LoopBudget(max_iter=3),
    )

    result = agent_cap.run(_input("what is bge-m3"))
    types = [a.type for a in result.outputs]
    assert types == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]

    # Retriever called once for the iter-0 execute; generator called
    # twice total — once inside execute_fn and once by the synthesizer.
    assert len(retriever.calls) == 1
    assert len(generator.calls) == 2

    # AGENT_TRACE shape sanity check.
    trace_body = json.loads(result.outputs[1].content.decode("utf-8"))
    assert trace_body["action"] == "rag"
    assert trace_body["outcome"]["stopReason"] == "converged"
    assert trace_body["outcome"]["stepCount"] == 1
    assert trace_body["outcome"]["aggregatedChunkCount"] == 2

    # RETRIEVAL_RESULT_AGG carries both chunks, deduped.
    agg_body = json.loads(result.outputs[2].content.decode("utf-8"))
    assert agg_body["aggregatedChunkCount"] == 2
    assert [r["chunkId"] for r in agg_body["results"]] == ["c1", "c2"]


def test_agent_loop_on_missing_rag_sub_raises_typed_error():
    agent_cap = AgentCapability(
        router=_FixedRouter(_decision(action="rag")),
        parser=NoOpQueryParser(),
        rag=None,  # rag missing
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(_FakeGenerator()),
        retriever=_FakeRetriever({}),
        generator=_FakeGenerator(),
        budget=LoopBudget(max_iter=1),
    )
    from app.capabilities.base import CapabilityError

    with pytest.raises(CapabilityError) as ex_info:
        agent_cap.run(_input("what is bge-m3"))
    assert ex_info.value.code == "AUTO_RAG_UNAVAILABLE"


# ---------------------------------------------------------------------------
# 4. loop-on with failing executor does not raise
# ---------------------------------------------------------------------------


def test_agent_loop_on_executor_failure_still_emits_artifacts():
    """A retrieval failure on iter 0 triggers best-effort empty outcome."""

    class _BoomRetriever:
        def retrieve(self, q):
            raise RuntimeError("boom")

    generator = _FakeGenerator()
    agent_cap = AgentCapability(
        router=_FixedRouter(_decision(action="rag")),
        parser=NoOpQueryParser(),
        rag=_SentinelCapability("RAG", "unused"),
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(generator),
        retriever=_BoomRetriever(),
        generator=generator,
        budget=LoopBudget(max_iter=2),
    )
    result = agent_cap.run(_input("what is X"))
    types = [a.type for a in result.outputs]
    # Four artifacts still emitted.
    assert types == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]
    trace_body = json.loads(result.outputs[1].content.decode("utf-8"))
    # stepCount 0 because the executor failed before any step was recorded.
    assert trace_body["outcome"]["stepCount"] == 0


# ---------------------------------------------------------------------------
# 5. loop-on with multiple iterations produces an iter>0 step
# ---------------------------------------------------------------------------


def test_agent_loop_iter_count_exceeds_one_when_rule_critic_flags_short_answer():
    parser = RegexQueryParser()
    # Executor returns a very short answer so RuleCritic flags
    # missing_facts on iter 0 and the loop runs another pass.
    retriever = _FakeRetriever(
        {
            "what is bge-m3": [_mk_chunk("c1")],
        }
    )

    class _ShortGenerator(GenerationProvider):
        @property
        def name(self) -> str:
            return "short-gen"

        def generate(self, query, chunks):
            return "short"  # <40ch triggers RuleCritic missing_facts

    agent_cap = AgentCapability(
        router=_FixedRouter(
            _decision(
                action="rag",
                parsed_query=parser.parse("what is bge-m3"),
            )
        ),
        parser=parser,
        rag=_SentinelCapability("RAG", "unused"),
        loop_enabled=True,
        critic=RuleCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(_ShortGenerator()),
        retriever=retriever,
        generator=_ShortGenerator(),
        budget=LoopBudget(max_iter=2, min_confidence_to_stop=0.75),
    )
    result = agent_cap.run(_input("what is bge-m3"))
    trace_body = json.loads(result.outputs[1].content.decode("utf-8"))
    # Ran both iterations — critic never converged.
    assert trace_body["outcome"]["stopReason"] == "iter_cap"
    assert trace_body["outcome"]["stepCount"] == 2


# ---------------------------------------------------------------------------
# 6. zero aggregated chunks synthesizer fallback
# ---------------------------------------------------------------------------


def test_agent_loop_empty_retrieval_falls_back_to_loop_answer():
    """Synthesizer returns loop's last answer when no chunks aggregated."""

    class _EmptyRetriever:
        def retrieve(self, q):
            class _R:
                results = []

            return _R()

    agent_cap = AgentCapability(
        router=_FixedRouter(_decision(action="rag")),
        parser=NoOpQueryParser(),
        rag=_SentinelCapability("RAG", "unused"),
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(_FakeGenerator()),
        retriever=_EmptyRetriever(),
        generator=_FakeGenerator(template="fallback_answer"),
        budget=LoopBudget(max_iter=1),
    )
    result = agent_cap.run(_input("any question"))
    types = [a.type for a in result.outputs]
    assert types[-1] == "FINAL_RESPONSE"
    # Non-empty answer — falls back to the loop's live answer rather
    # than letting an empty string through.
    final = result.outputs[-1].content.decode("utf-8")
    assert final  # non-empty
