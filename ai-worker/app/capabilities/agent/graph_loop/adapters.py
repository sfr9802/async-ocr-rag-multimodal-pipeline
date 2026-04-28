"""Public adapter exposing a controller-compatible interface over the graph backend.

``AgentLoopGraph`` mirrors ``AgentLoopController.run`` exactly so the
calling code (``AgentCapability._run_loop_and_synthesize``) can swap
the runner without any other change. The adapter:

  1. Builds a per-run ``StateGraph`` closing over the supplied
     ``execute_fn``.
  2. Invokes the graph with an initial state populated from the run-time
     arguments + safe empties for every optional graph-side field.
  3. Extracts a ``LoopOutcome`` from the terminal state — schema is
     identical to what the legacy controller emits, so AGENT_TRACE,
     RETRIEVAL_RESULT_AGG, and FINAL_RESPONSE artifacts stay byte-stable
     for downstream consumers.

Failure modes that fall back to an empty outcome instead of raising:

  * langgraph import / build error (e.g. dependency missing).
  * ``graph.invoke`` raising for any reason (recursion limit, internal
    LangGraph bug).

Either case logs a warning and returns the same empty
``LoopOutcome(stop_reason='iter_cap', steps=[], ...)`` shape the
capability already handles for the legacy ``_ExecuteFailure`` path —
so the four-artifact contract holds even when the graph backend
itself misbehaves.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.capabilities.agent.critic import AgentCriticProvider
from app.capabilities.agent.graph_loop.builder import build_agent_loop_graph
from app.capabilities.agent.loop import (
    ExecuteFn,
    LoopBudget,
    LoopOutcome,
    LoopStep,
    StopReason,
)
from app.capabilities.agent.rewriter import QueryRewriterProvider
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider

log = logging.getLogger(__name__)


# Per-run recursion budget for LangGraph. Each iteration walks roughly
# 6 nodes (retrieve → aggregate → score → critic → decide → rewrite);
# we multiply by ``max_iter`` and add headroom for the synthesize /
# budget_exhausted tail. LangGraph defaults to 25, which is enough for
# max_iter=3 but not for sweeps that bump max_iter higher.
_NODES_PER_ITER = 6
_RECURSION_HEADROOM = 16


class AgentLoopGraph:
    """LangGraph-backed agent loop runner.

    Drop-in compatible with ``AgentLoopController`` — same constructor
    kwargs, same ``run`` signature, same ``LoopOutcome`` return shape.
    Selected via ``settings.agent_loop_backend = 'graph'``; the default
    backend stays ``'legacy'``.
    """

    name = "graph"

    def __init__(
        self,
        *,
        critic: AgentCriticProvider,
        rewriter: QueryRewriterProvider,
        parser: QueryParserProvider,
        budget: LoopBudget,
    ) -> None:
        self._critic = critic
        self._rewriter = rewriter
        self._parser = parser
        self._budget = budget
        # Failure marker for the most recent ``run`` invocation. ``None``
        # means the graph completed normally (the empty outcome may still
        # be returned by a real run that had nothing to retrieve). The A/B
        # harness reads this attribute to mark graph build/invoke faults
        # as success=False so degraded runs don't masquerade as healthy
        # ones in the comparison summary.
        self._last_failure: Optional[str] = None

    @property
    def budget(self) -> LoopBudget:
        return self._budget

    @property
    def last_failure(self) -> Optional[str]:
        """Reason of the last graph build/invoke failure, or ``None``.

        Set to ``"build_failed"`` when ``build_agent_loop_graph`` raised,
        ``"invoke_failed"`` when ``graph.invoke`` raised. Reset to
        ``None`` at the top of every ``run`` so callers see the status
        of the most recent invocation.
        """
        return self._last_failure

    # ------------------------------------------------------------------

    def run(
        self,
        *,
        question: str,
        initial_parsed_query: ParsedQuery,
        execute_fn: ExecuteFn,
    ) -> LoopOutcome:
        started_at = time.monotonic()
        self._last_failure = None

        try:
            graph = build_agent_loop_graph(
                critic=self._critic,
                rewriter=self._rewriter,
                parser=self._parser,
                budget=self._budget,
                execute_fn=execute_fn,
            )
        except Exception as ex:
            self._last_failure = "build_failed"
            log.warning(
                "AgentLoopGraph: build failed (%s: %s); returning "
                "empty outcome.",
                type(ex).__name__,
                ex,
            )
            return _empty_outcome()

        initial_state = {
            "job_id": "",
            "capability": "AGENT",
            "action": "rag",
            "original_query": question,
            "current_query": initial_parsed_query,
            "iteration": 0,
            "max_iterations": int(self._budget.max_iter),
            "started_at": started_at,
            "total_tokens": 0,
            "last_answer": "",
            "last_chunks": [],
            "last_exec_tokens": 0,
            "critic_decision": None,
            "retrieval_reports": [],
            "candidate_pool": [],
            "seen_chunk_ids": [],
            "quality_history": [],
            "rewrite_history": [],
            "steps": [],
            "stop_reason": None,
            "final_answer": "",
            "trace": [],
            "errors": [],
        }

        recursion_limit = (
            int(self._budget.max_iter) * _NODES_PER_ITER + _RECURSION_HEADROOM
        )

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
        except Exception as ex:
            self._last_failure = "invoke_failed"
            log.warning(
                "AgentLoopGraph: graph invocation failed (%s: %s); "
                "returning empty outcome.",
                type(ex).__name__,
                ex,
            )
            return _empty_outcome()

        return _extract_outcome(final_state, started_at)


def _extract_outcome(
    final_state: dict,
    started_at: float,
) -> LoopOutcome:
    """Build a LoopOutcome from the graph's terminal state.

    Schema must match what ``AgentLoopController`` emits — that's the
    invariant AGENT_TRACE / RETRIEVAL_RESULT_AGG / FINAL_RESPONSE
    artifact byte-stability rests on.
    """
    steps: List[LoopStep] = list(final_state.get("steps") or [])
    aggregated: List[RetrievedChunk] = list(final_state.get("candidate_pool") or [])
    stop_reason: StopReason = (
        final_state.get("stop_reason") or "iter_cap"  # type: ignore[assignment]
    )
    final_answer = final_state.get("final_answer") or ""
    total_ms = round((time.monotonic() - started_at) * 1000.0, 3)
    total_tokens = int(final_state.get("total_tokens") or 0)
    return LoopOutcome(
        steps=steps,
        stop_reason=stop_reason,
        final_answer=final_answer,
        aggregated_chunks=aggregated,
        total_ms=total_ms,
        total_llm_tokens=total_tokens,
    )


def _empty_outcome() -> LoopOutcome:
    """Synthetic zero-step outcome used when the graph itself can't run.

    Mirrors the fallback the capability builds when ``_ExecuteFailure``
    propagates out of the legacy controller. Returning this shape keeps
    the four-artifact set intact even on graph-side failures.
    """
    return LoopOutcome(
        steps=[],
        stop_reason="iter_cap",
        final_answer="",
        aggregated_chunks=[],
        total_ms=0.0,
        total_llm_tokens=0,
    )
