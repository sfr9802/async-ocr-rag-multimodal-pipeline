"""AgentLoopGraph (LangGraph backend) tests.

Covers the experimental graph-loop runner introduced as a sibling of
``AgentLoopController``. The legacy controller stays the production
default — these tests verify:

  1. ``settings.agent_loop_backend`` defaults to ``'legacy'`` and only
     ``'legacy' | 'graph'`` is accepted at the settings boundary.
  2. AgentCapability without an injected runner still uses the legacy
     controller (Phase 6 default unchanged).
  3. AgentCapability with an AgentLoopGraph runner emits the same
     four-artifact set (AGENT_DECISION, AGENT_TRACE, RETRIEVAL_RESULT_AGG,
     FINAL_RESPONSE).
  4. Direct AgentLoopGraph.run smoke: NoOpCritic + NoOpQueryRewriter +
     a deterministic executor converges at iter=0.
  5. LoopOutcome.to_dict() schema matches between legacy and graph
     backends (the artifact-byte-stability invariant the design doc
     calls out).
  6. Budget exhaustion (RuleCritic flagging short answers) emits a
     non-empty FINAL_RESPONSE artifact and stop_reason='iter_cap'.
  7. The graph runner never produces more steps than ``max_iter``.
  8. An executor failure on iter 0 falls back to a synthetic empty
     outcome — the four-artifact contract still holds.

The module is skipped wholesale when langgraph is not installed in the
dev environment; the production code path also handles that gracefully
via the registry's try/except, but the unit tests need the real graph
to assert behaviour.
"""

from __future__ import annotations

import json
from typing import List

import pytest

# Skip the whole module if langgraph is not installed in the dev env.
pytest.importorskip("langgraph")

from app.capabilities.agent.capability import AgentCapability
from app.capabilities.agent.critic import (
    AgentCriticProvider,
    CritiqueResult,
    NoOpCritic,
    RuleCritic,
)
from app.capabilities.agent.graph_loop import AgentLoopGraph
from app.capabilities.agent.graph_loop.adapters import _empty_outcome
from app.capabilities.agent.loop import (
    AgentLoopController,
    LoopBudget,
    LoopOutcome,
)
from app.capabilities.agent.rewriter import NoOpQueryRewriter
from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
)
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.generation import GenerationProvider, RetrievedChunk
from app.capabilities.rag.query_parser import RegexQueryParser
from app.core.config import WorkerSettings


# ---------------------------------------------------------------------------
# Test doubles — kept parallel to test_agent_capability.py so an existing
# reader recognises the shape immediately.
# ---------------------------------------------------------------------------


class _FixedRouter(AgentRouterProvider):
    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision

    @property
    def name(self) -> str:
        return self._decision.router_name

    def decide(self, *, text, has_file, file_mime, file_size):
        return self._decision


class _FakeRetriever:
    def __init__(self, chunks_by_query: dict) -> None:
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


class _ShortGenerator(GenerationProvider):
    """Always returns a sub-40ch answer so RuleCritic flags missing_facts."""

    @property
    def name(self) -> str:
        return "short-gen"

    def generate(self, query, chunks):
        return "short"


class _NeverSufficientCritic(AgentCriticProvider):
    """Critic that always votes missing_facts so the loop runs to iter_cap."""

    @property
    def name(self) -> str:
        return "never-sufficient"

    def evaluate(self, *, question, answer, retrieved):
        return CritiqueResult(
            sufficient=False,
            gap_type="missing_facts",
            gap_reason="never sufficient",
            confidence=0.4,
            critic_name=self.name,
            llm_tokens_used=0,
        )


class _SentinelCapability(Capability):
    def __init__(self, name: str, payload: str) -> None:
        self.name = name
        self._payload = payload

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        return CapabilityOutput(
            outputs=[
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename="sentinel.md",
                    content_type="text/markdown; charset=utf-8",
                    content=self._payload.encode("utf-8"),
                )
            ]
        )


def _mk_chunk(cid: str, doc: str = "doc-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc, section="", text=f"text for {cid}", score=0.9,
    )


def _input(text: str) -> CapabilityInput:
    return CapabilityInput(
        job_id="job-graph",
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


def _decision(*, action: str = "rag", parsed_query=None) -> AgentDecision:
    return AgentDecision(
        action=action,  # type: ignore[arg-type]
        reason="test",
        parsed_query=parsed_query,
        confidence=0.8,
        router_name="rule",
    )


# ---------------------------------------------------------------------------
# 1. Settings default + boundary
# ---------------------------------------------------------------------------


def test_settings_default_agent_loop_backend_is_legacy(monkeypatch):
    """A fresh WorkerSettings with no env override defaults to legacy."""
    # Strip any externally-set value so the test reflects the literal
    # default the production worker sees on a clean install.
    monkeypatch.delenv("AIPIPELINE_WORKER_AGENT_LOOP_BACKEND", raising=False)
    s = WorkerSettings()
    assert s.agent_loop_backend == "legacy"


def test_settings_accepts_explicit_graph_backend(monkeypatch):
    """The 'graph' value round-trips through env loading."""
    monkeypatch.setenv("AIPIPELINE_WORKER_AGENT_LOOP_BACKEND", "graph")
    s = WorkerSettings()
    assert s.agent_loop_backend == "graph"


def test_settings_rejects_unknown_backend_at_parse_time(monkeypatch):
    """Literal['legacy','graph'] makes pydantic fail-fast on typos.

    Catching a typo at settings parse time is preferable to a runtime
    surprise in the registry — the worker simply refuses to start with
    a misconfigured AGENT_LOOP_BACKEND, and the operator sees a typed
    pydantic ValidationError.
    """
    from pydantic import ValidationError

    monkeypatch.setenv("AIPIPELINE_WORKER_AGENT_LOOP_BACKEND", "lang")
    with pytest.raises(ValidationError):
        WorkerSettings()


# ---------------------------------------------------------------------------
# 2. AgentCapability dispatch — legacy default vs explicit graph injection
# ---------------------------------------------------------------------------


def test_capability_without_runner_uses_legacy_controller():
    """No loop_runner injected -> capability falls back to AgentLoopController.

    The behavioural proof: the four-artifact set is emitted, AGENT_TRACE
    has the legacy LoopOutcome shape, and the test relies on no
    langgraph nodes whatsoever.
    """
    parser = RegexQueryParser()
    retriever = _FakeRetriever({"q": [_mk_chunk("c1")]})
    generator = _FakeGenerator()
    cap = AgentCapability(
        router=_FixedRouter(_decision(parsed_query=parser.parse("q"))),
        parser=parser,
        rag=_SentinelCapability("RAG", "x"),
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(generator),
        retriever=retriever,
        generator=generator,
        budget=LoopBudget(max_iter=2),
        # loop_runner left None -> legacy default
    )
    out = cap.run(_input("q"))
    assert [a.type for a in out.outputs] == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]


def test_capability_with_graph_runner_uses_graph_backend(monkeypatch):
    """An injected AgentLoopGraph drives the loop instead of the controller.

    Asserts via a runner spy: legacy AgentLoopController.run is never
    called when the graph backend is wired in.
    """
    parser = RegexQueryParser()
    retriever = _FakeRetriever({"q": [_mk_chunk("c1")]})
    generator = _FakeGenerator()

    legacy_calls: list[str] = []
    real_legacy_run = AgentLoopController.run

    def _spy_legacy_run(self, *args, **kwargs):  # pragma: no cover - spy
        legacy_calls.append("legacy")
        return real_legacy_run(self, *args, **kwargs)

    monkeypatch.setattr(AgentLoopController, "run", _spy_legacy_run)

    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )
    cap = AgentCapability(
        router=_FixedRouter(_decision(parsed_query=parser.parse("q"))),
        parser=parser,
        rag=_SentinelCapability("RAG", "x"),
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(generator),
        retriever=retriever,
        generator=generator,
        budget=LoopBudget(max_iter=2),
        loop_runner=runner,
    )
    out = cap.run(_input("q"))

    assert legacy_calls == [], "graph backend must not delegate to legacy"
    assert [a.type for a in out.outputs] == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]
    trace = json.loads(out.outputs[1].content.decode("utf-8"))
    assert trace["outcome"]["stopReason"] == "converged"
    assert trace["outcome"]["stepCount"] == 1


# ---------------------------------------------------------------------------
# 3. Direct AgentLoopGraph.run smoke
# ---------------------------------------------------------------------------


def test_graph_runner_converges_at_iter_zero_with_noop_critic():
    parser = RegexQueryParser()
    pq = parser.parse("hello")
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=3),
    )

    def execute_fn(q):
        return "an answer", [_mk_chunk("c1")], 0

    outcome = runner.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    assert outcome.stop_reason == "converged"
    assert len(outcome.steps) == 1
    assert outcome.aggregated_chunks[0].chunk_id == "c1"
    assert outcome.final_answer == "an answer"


# ---------------------------------------------------------------------------
# 4. Schema parity: graph and legacy produce the same LoopOutcome dict shape
# ---------------------------------------------------------------------------


def test_graph_and_legacy_share_loopoutcome_dict_schema():
    """Both backends MUST agree on the LoopOutcome.to_dict() key set.

    AGENT_TRACE is a JSON dump of LoopOutcome.to_dict() keyed under
    'outcome'; downstream consumers (Spring callback recorder + eval
    harness) tolerate field additions but not key-renames or removals.
    """
    parser = RegexQueryParser()
    pq = parser.parse("hello")
    legacy = AgentLoopController(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )
    graph = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )

    def execute_fn(q):
        return "an answer with enough length", [_mk_chunk("c1")], 0

    legacy_outcome = legacy.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    graph_outcome = graph.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    assert (
        legacy_outcome.to_dict().keys() == graph_outcome.to_dict().keys()
    )
    assert legacy_outcome.stop_reason == graph_outcome.stop_reason
    assert len(legacy_outcome.steps) == len(graph_outcome.steps)
    assert (
        [c.chunk_id for c in legacy_outcome.aggregated_chunks]
        == [c.chunk_id for c in graph_outcome.aggregated_chunks]
    )
    # LoopStep.to_dict() schemas must agree too — the AGENT_TRACE
    # body inlines them under outcome["steps"].
    assert (
        legacy_outcome.steps[0].to_dict().keys()
        == graph_outcome.steps[0].to_dict().keys()
    )


# ---------------------------------------------------------------------------
# 5. Budget exhaustion still emits FINAL_RESPONSE
# ---------------------------------------------------------------------------


def test_graph_iter_cap_emits_final_response_artifact():
    """RuleCritic flagging short answers triggers iter_cap with FINAL_RESPONSE."""
    parser = RegexQueryParser()
    retriever = _FakeRetriever({"q": [_mk_chunk("c1")]})
    runner = AgentLoopGraph(
        critic=RuleCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2, min_confidence_to_stop=0.75),
    )
    cap = AgentCapability(
        router=_FixedRouter(_decision(parsed_query=parser.parse("q"))),
        parser=parser,
        rag=_SentinelCapability("RAG", "x"),
        loop_enabled=True,
        critic=RuleCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(_ShortGenerator()),
        retriever=retriever,
        generator=_ShortGenerator(),
        budget=LoopBudget(max_iter=2, min_confidence_to_stop=0.75),
        loop_runner=runner,
    )
    out = cap.run(_input("q"))
    types = [a.type for a in out.outputs]
    assert types == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]
    final = out.outputs[-1]
    assert final.content, "FINAL_RESPONSE must have non-empty bytes"
    trace = json.loads(out.outputs[1].content.decode("utf-8"))
    assert trace["outcome"]["stopReason"] == "iter_cap"


# ---------------------------------------------------------------------------
# 6. max_iter respected by graph runner
# ---------------------------------------------------------------------------


def test_graph_runner_does_not_exceed_max_iter_steps():
    parser = RegexQueryParser()
    pq = parser.parse("hello")
    runner = AgentLoopGraph(
        critic=_NeverSufficientCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )

    def execute_fn(q):
        return "short", [_mk_chunk("c1")], 0

    outcome = runner.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    assert len(outcome.steps) <= 2
    assert outcome.stop_reason == "iter_cap"


def test_graph_runner_max_iter_three_caps_steps_at_three():
    """A larger budget still hard-caps the step count at max_iter."""
    parser = RegexQueryParser()
    pq = parser.parse("hello")
    runner = AgentLoopGraph(
        critic=_NeverSufficientCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=3),
    )

    def execute_fn(q):
        return "short", [_mk_chunk(f"c-{q.normalized}")], 0

    outcome = runner.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    assert len(outcome.steps) == 3
    assert outcome.stop_reason == "iter_cap"


# ---------------------------------------------------------------------------
# 7. Graph executor failure on iter 0 -> empty outcome (legacy parity)
# ---------------------------------------------------------------------------


def test_graph_runner_executor_failure_returns_zero_step_outcome():
    parser = RegexQueryParser()
    pq = parser.parse("hello")
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )

    def execute_fn(q):
        raise RuntimeError("boom")

    outcome = runner.run(
        question="hello",
        initial_parsed_query=pq,
        execute_fn=execute_fn,
    )
    # Initial-retrieve error path stamps stop_reason='iter_cap', skips
    # the critic (no LoopStep recorded), and routes through synthesize.
    # That mirrors the legacy fallback the capability already handles.
    assert outcome.stop_reason == "iter_cap"
    assert outcome.steps == []
    assert outcome.aggregated_chunks == []
    assert outcome.final_answer == ""


def test_graph_capability_executor_failure_still_emits_four_artifacts():
    """Even when the executor explodes inside a graph run, the capability
    must still emit the four-artifact set — same contract as legacy."""

    class _BoomRetriever:
        def retrieve(self, q):
            raise RuntimeError("boom")

    parser = RegexQueryParser()
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )
    cap = AgentCapability(
        router=_FixedRouter(_decision(parsed_query=parser.parse("q"))),
        parser=parser,
        rag=_SentinelCapability("RAG", "x"),
        loop_enabled=True,
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        synthesizer=AgentSynthesizer(_FakeGenerator()),
        retriever=_BoomRetriever(),
        generator=_FakeGenerator(),
        budget=LoopBudget(max_iter=2),
        loop_runner=runner,
    )
    out = cap.run(_input("q"))
    assert [a.type for a in out.outputs] == [
        "AGENT_DECISION",
        "AGENT_TRACE",
        "RETRIEVAL_RESULT_AGG",
        "FINAL_RESPONSE",
    ]
    trace = json.loads(out.outputs[1].content.decode("utf-8"))
    assert trace["outcome"]["stepCount"] == 0


# ---------------------------------------------------------------------------
# 8. _empty_outcome shape parity
# ---------------------------------------------------------------------------


def test_empty_outcome_helper_returns_loopoutcome_zero_step():
    """The graph adapter's empty fallback must be a real LoopOutcome.

    Guards the "graph build / invoke failure -> empty outcome" branch
    that the registry implicitly relies on when langgraph is absent.
    """
    out = _empty_outcome()
    assert isinstance(out, LoopOutcome)
    assert out.stop_reason == "iter_cap"
    assert out.steps == []
    assert out.aggregated_chunks == []
    assert out.total_ms == 0.0
    assert out.total_llm_tokens == 0


# ---------------------------------------------------------------------------
# 9. last_failure tracking (A/B harness contract)
# ---------------------------------------------------------------------------


def test_agent_loop_graph_initial_last_failure_is_none():
    """A fresh runner has not run yet, so ``last_failure`` is None."""
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=RegexQueryParser(),
        budget=LoopBudget(max_iter=1),
    )
    assert runner.last_failure is None


def test_agent_loop_graph_last_failure_set_when_build_raises(monkeypatch):
    """``build_agent_loop_graph`` raising must mark last_failure='build_failed'.

    The runner returns the empty fallback outcome so the production
    capability stays alive, but the A/B harness reads ``last_failure`` to
    surface the degraded run as success=False instead of letting it
    masquerade as a clean empty result.
    """
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=RegexQueryParser(),
        budget=LoopBudget(max_iter=1),
    )

    def _boom(**_kwargs):
        raise RuntimeError("simulated build failure")

    # Patch the symbol the adapter imported, not the source module —
    # the adapter binds ``build_agent_loop_graph`` at import time.
    import app.capabilities.agent.graph_loop.adapters as adapters_module
    monkeypatch.setattr(adapters_module, "build_agent_loop_graph", _boom)

    parser = RegexQueryParser()
    initial_pq = parser.parse("anything")
    outcome = runner.run(
        question="anything",
        initial_parsed_query=initial_pq,
        execute_fn=lambda pq: ("", [], 0),
    )
    assert outcome.steps == []
    assert outcome.aggregated_chunks == []
    assert runner.last_failure == "build_failed"


def test_agent_loop_graph_last_failure_resets_on_successful_run():
    """After a clean run, ``last_failure`` returns to None."""
    runner = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=RegexQueryParser(),
        budget=LoopBudget(max_iter=1),
    )
    parser = RegexQueryParser()
    initial_pq = parser.parse("anything")
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="d1", section="", text="t", score=0.9,
    )
    runner.run(
        question="anything",
        initial_parsed_query=initial_pq,
        execute_fn=lambda pq: ("answer", [chunk], 0),
    )
    assert runner.last_failure is None
