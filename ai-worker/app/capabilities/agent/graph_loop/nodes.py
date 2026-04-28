"""LangGraph node factories for the agent loop graph backend.

Each ``make_*_node`` returns a callable suitable for
``StateGraph.add_node``. The factories close over the shared
collaborators (critic / rewriter / parser / budget / execute_fn) so
node bodies stay clean of constructor wiring.

Stop-reason priority MUST match ``AgentLoopController`` exactly:

    unanswerable > converged > time_cap > token_cap > iter_cap.

That ladder lives in ``make_decide_next_action_node`` — every other
node either records breadcrumbs or advances state. The graph backend
is a structural reorganisation, not a behavioural change.

Error handling: any exception escaping a node-side collaborator
(executor / critic / rewriter) is caught here, recorded onto
``state['errors']``, and converted into an ``iter_cap`` stop. The
graph adapter promises the same "loop never raises" contract the
legacy controller offers — the conditional edges route the error
state straight to ``budget_exhausted_node`` so the synthesise tail
still runs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Tuple

from app.capabilities.agent.critic import AgentCriticProvider, CritiqueResult
from app.capabilities.agent.graph_loop.state import (
    ROUTE_ERROR,
    ROUTE_EXPAND_CANDIDATES,
    ROUTE_REWRITE_QUERY,
    ROUTE_STOP_BUDGET_EXHAUSTED,
    ROUTE_STOP_SUFFICIENT,
    AgentLoopGraphState,
)
from app.capabilities.agent.loop import LoopBudget, LoopStep
from app.capabilities.agent.rewriter import QueryRewriterProvider
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider

log = logging.getLogger(__name__)


# Same answer-preview clip the legacy controller uses (loop.py). Keeps
# LoopStep bytes identical between backends so AGENT_TRACE consumers
# don't see a length drift on the answerPreview field.
_ANSWER_PREVIEW_CHARS = 300


# Type alias mirrors ``app.capabilities.agent.loop.ExecuteFn``. Restated
# locally so this module doesn't reach into the legacy package's
# internals beyond the shared value objects.
GraphExecuteFn = Callable[
    [ParsedQuery], Tuple[str, List[RetrievedChunk], int]
]


# --------------------------------------------------------------------------
# Retrieve nodes (initial + re-run after rewrite / expand)
# --------------------------------------------------------------------------


def make_initial_retrieve_node(
    execute_fn: GraphExecuteFn,
) -> Callable[[AgentLoopGraphState], Dict[str, Any]]:
    """First retrieve call — feeds iter 0 with the parsed query the caller built.

    Mirrors the legacy controller's first ``execute_fn(current_query)``.
    Failure stamps both ``errors`` and ``stop_reason='iter_cap'`` so the
    initial-retrieve conditional edge routes to ``budget_exhausted`` and
    skips the critic/decide chain — that keeps step count and aggregated
    chunks at zero, matching the legacy ``_ExecuteFailure`` fallback.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        try:
            answer, chunks, tokens = execute_fn(state["current_query"])
        except Exception as ex:
            log.warning(
                "graph_loop initial_retrieve failed (%s: %s); routing "
                "to budget_exhausted",
                type(ex).__name__,
                ex,
            )
            return {
                "errors": list(state.get("errors") or []) + [
                    {
                        "node": "initial_retrieve",
                        "type": type(ex).__name__,
                        "message": str(ex),
                    }
                ],
                "last_answer": "",
                "last_chunks": [],
                "last_exec_tokens": 0,
                "stop_reason": "iter_cap",
            }

        exec_tokens = max(0, int(tokens or 0))
        return {
            "last_answer": answer or "",
            "last_chunks": list(chunks or []),
            "last_exec_tokens": exec_tokens,
            "total_tokens": int(state.get("total_tokens") or 0) + exec_tokens,
            "retrieval_reports": list(state.get("retrieval_reports") or []) + [
                {
                    "iter": int(state.get("iteration") or 0),
                    "query": state["current_query"].normalized
                    or state["current_query"].original,
                    "chunkCount": len(chunks or []),
                }
            ],
        }

    return _node


def make_retrieve_again_node(
    execute_fn: GraphExecuteFn,
) -> Callable[[AgentLoopGraphState], Dict[str, Any]]:
    """Re-run the executor with the current query (set by rewrite / expand).

    Same shape as ``make_initial_retrieve_node`` but tagged for log
    breadcrumbs and routed differently in the graph: a failure here
    short-circuits to ``budget_exhausted`` while the steps already
    recorded for earlier iters stay on state.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        try:
            answer, chunks, tokens = execute_fn(state["current_query"])
        except Exception as ex:
            log.warning(
                "graph_loop retrieve_again failed at iter=%d (%s: %s); "
                "routing to budget_exhausted",
                int(state.get("iteration") or 0),
                type(ex).__name__,
                ex,
            )
            return {
                "errors": list(state.get("errors") or []) + [
                    {
                        "node": "retrieve_again",
                        "type": type(ex).__name__,
                        "message": str(ex),
                    }
                ],
                "last_answer": state.get("last_answer") or "",
                "last_chunks": [],
                "last_exec_tokens": 0,
                "stop_reason": "iter_cap",
            }

        exec_tokens = max(0, int(tokens or 0))
        return {
            "last_answer": answer or "",
            "last_chunks": list(chunks or []),
            "last_exec_tokens": exec_tokens,
            "total_tokens": int(state.get("total_tokens") or 0) + exec_tokens,
            "retrieval_reports": list(state.get("retrieval_reports") or []) + [
                {
                    "iter": int(state.get("iteration") or 0),
                    "query": state["current_query"].normalized
                    or state["current_query"].original,
                    "chunkCount": len(chunks or []),
                }
            ],
        }

    return _node


# --------------------------------------------------------------------------
# Aggregation + retrieval-quality breadcrumbs
# --------------------------------------------------------------------------


def make_aggregate_candidates_node() -> Callable[
    [AgentLoopGraphState], Dict[str, Any]
]:
    """Dedup ``last_chunks`` into ``candidate_pool`` by chunk_id.

    First-occurrence-wins: the earliest iteration that surfaces a
    chunk_id wins the slot, preserving its original score / rank. This
    is exactly the rule ``AgentLoopController`` uses (loop.py:239-243),
    so the aggregated chunk list stays byte-for-byte stable across
    backends.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        seen_list = list(state.get("seen_chunk_ids") or [])
        seen_set = set(seen_list)
        pool = list(state.get("candidate_pool") or [])
        for chunk in state.get("last_chunks") or []:
            if chunk.chunk_id in seen_set:
                continue
            seen_set.add(chunk.chunk_id)
            seen_list.append(chunk.chunk_id)
            pool.append(chunk)
        return {
            "candidate_pool": pool,
            "seen_chunk_ids": seen_list,
        }

    return _node


def make_score_quality_node() -> Callable[
    [AgentLoopGraphState], Dict[str, Any]
]:
    """Record a lightweight retrieval-quality breadcrumb per iter.

    Side-effect-only — does NOT influence routing or the critic verdict.
    The legacy controller has no equivalent stage, so anything that
    affects loop decisions here would break legacy parity. Keep this
    node observability-only; heavier judgement belongs on the critic.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        chunks = state.get("last_chunks") or []
        n = len(chunks)
        if n >= 2:
            top = float(chunks[0].score)
            bot = float(chunks[-1].score)
            gap: float | None = round(top - bot, 4)
        else:
            gap = None
        entry = {
            "iter": int(state.get("iteration") or 0),
            "chunkCount": n,
            "topkGap": gap,
            "answerLen": len(state.get("last_answer") or ""),
        }
        return {
            "quality_history": list(state.get("quality_history") or []) + [entry],
        }

    return _node


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------


def make_critic_node(
    critic: AgentCriticProvider,
) -> Callable[[AgentLoopGraphState], Dict[str, Any]]:
    """Call the critic over (question, last_answer, last_chunks).

    Records a ``LoopStep`` for the current iteration as soon as the
    critic returns — that's the latest point at which we have all the
    fields ``LoopStep`` needs (iter index, query, chunk ids, answer
    preview, critique, exec tokens). Aligns with ``loop.py:253-263``
    where the legacy controller assembles the same step inline.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        try:
            verdict: CritiqueResult = critic.evaluate(
                question=state["original_query"],
                answer=state.get("last_answer") or "",
                retrieved=state.get("last_chunks") or [],
            )
        except Exception as ex:
            # The critic implementations all promise no-raise, but a
            # buggy custom critic must not break the loop's contract.
            log.warning(
                "graph_loop critic raised (%s: %s); routing to "
                "budget_exhausted",
                type(ex).__name__,
                ex,
            )
            return {
                "errors": list(state.get("errors") or []) + [
                    {
                        "node": "critic",
                        "type": type(ex).__name__,
                        "message": str(ex),
                    }
                ],
                "stop_reason": "iter_cap",
            }

        critic_tokens = max(0, int(verdict.llm_tokens_used or 0))
        step = LoopStep(
            iter=int(state.get("iteration") or 0),
            query=state["current_query"],
            retrieved_chunk_ids=[
                c.chunk_id for c in (state.get("last_chunks") or [])
            ],
            answer_preview=_clip(
                state.get("last_answer") or "", _ANSWER_PREVIEW_CHARS
            ),
            critique=verdict,
            step_ms=0.0,  # graph backend doesn't time per-iter; legacy still does
            llm_tokens_used=int(state.get("last_exec_tokens") or 0)
            + critic_tokens,
        )
        return {
            "critic_decision": verdict.to_dict(),
            "total_tokens": int(state.get("total_tokens") or 0) + critic_tokens,
            "steps": list(state.get("steps") or []) + [step],
        }

    return _node


# --------------------------------------------------------------------------
# Stop-reason judge + rewrite + tail nodes
# --------------------------------------------------------------------------


def make_decide_next_action_node(
    budget: LoopBudget,
) -> Callable[[AgentLoopGraphState], Dict[str, Any]]:
    """Translate the critic verdict + budget caps into a ``stop_reason``.

    Stop priority is identical to ``AgentLoopController.run`` (loop.py:
    279-302):

        unanswerable > converged > time_cap > token_cap > iter_cap.

    The node only records the verdict on ``state['stop_reason']``; the
    conditional edge ``decide_route`` reads that field to route. If a
    prior node already stamped ``stop_reason`` (e.g. critic crashed),
    we leave it in place — that error state must propagate.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        if state.get("stop_reason"):
            return {}

        decision = state.get("critic_decision") or {}
        gap_type = decision.get("gapType")
        sufficient = bool(decision.get("sufficient"))
        confidence = float(decision.get("confidence") or 0.0)
        iteration = int(state.get("iteration") or 0)
        elapsed_ms = (time.monotonic() - float(state["started_at"])) * 1000.0

        if gap_type == "unanswerable":
            return {"stop_reason": "unanswerable"}
        if sufficient and confidence >= float(budget.min_confidence_to_stop):
            return {"stop_reason": "converged"}
        if elapsed_ms >= float(budget.max_total_ms):
            return {"stop_reason": "time_cap"}
        if int(state.get("total_tokens") or 0) >= int(budget.max_llm_tokens):
            return {"stop_reason": "token_cap"}
        if iteration + 1 >= int(budget.max_iter):
            return {"stop_reason": "iter_cap"}
        return {}  # keep going — REWRITE_QUERY

    return _node


def make_rewrite_node(
    rewriter: QueryRewriterProvider,
    parser: QueryParserProvider,
) -> Callable[[AgentLoopGraphState], Dict[str, Any]]:
    """Ask the rewriter for a fresh query and advance the iteration counter.

    Mirrors ``loop.py:303-310`` — same arguments, same parser handed
    in so ``parser_name='rewriter-fallback'`` shows up in the trace if
    the LlmQueryRewriter falls through to its deterministic path.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        try:
            new_query: ParsedQuery = rewriter.rewrite(
                original=state["original_query"],
                prev_answer=state.get("last_answer") or "",
                gap_reason=str(
                    (state.get("critic_decision") or {}).get("gapReason") or ""
                ),
                already_retrieved_chunks=list(
                    state.get("candidate_pool") or []
                ),
                parser=parser,
            )
        except Exception as ex:
            # Production rewriters always fall back — but a buggy
            # custom one must not crash the loop.
            log.warning(
                "graph_loop rewrite raised (%s: %s); routing to "
                "budget_exhausted",
                type(ex).__name__,
                ex,
            )
            return {
                "errors": list(state.get("errors") or []) + [
                    {
                        "node": "rewrite",
                        "type": type(ex).__name__,
                        "message": str(ex),
                    }
                ],
                "stop_reason": "iter_cap",
            }
        return {
            "current_query": new_query,
            "rewrite_history": list(state.get("rewrite_history") or []) + [new_query],
            "iteration": int(state.get("iteration") or 0) + 1,
        }

    return _node


def make_synthesize_node() -> Callable[
    [AgentLoopGraphState], Dict[str, Any]
]:
    """Mark ``final_answer = last_answer``.

    Final-answer composition over ``aggregated_chunks`` runs OUTSIDE the
    graph (in ``AgentCapability._run_loop_and_synthesize``) — exactly
    the same boundary the legacy controller uses. This node only commits
    the last live answer onto state so the adapter has a single field
    to read when building the LoopOutcome.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        return {
            "final_answer": state.get("last_answer") or "",
        }

    return _node


def make_budget_exhausted_node() -> Callable[
    [AgentLoopGraphState], Dict[str, Any]
]:
    """Stamp ``iter_cap`` when an upstream error left ``stop_reason`` blank.

    Most error paths set ``stop_reason`` themselves; this node is the
    catch-all when the graph routes to ERROR but the relevant node
    forgot to. Always followed by ``synthesize`` so the adapter still
    reads a finalised state.
    """

    def _node(state: AgentLoopGraphState) -> Dict[str, Any]:
        if state.get("stop_reason"):
            return {}
        return {"stop_reason": "iter_cap"}

    return _node


# --------------------------------------------------------------------------
# Conditional-edge routers
# --------------------------------------------------------------------------


def post_retrieve_route(state: AgentLoopGraphState) -> str:
    """Route after initial_retrieve / retrieve_again.

    Errors short-circuit to ``budget_exhausted``; otherwise fall through
    to aggregation. Used by both retrieve nodes so a failure on either
    end of the loop produces the same fall-back shape.
    """
    if state.get("errors") or state.get("stop_reason"):
        return ROUTE_ERROR
    return "OK"


def decide_route(state: AgentLoopGraphState) -> str:
    """Route after decide_next_action_node.

    Reads the stop_reason / errors fields the upstream nodes set and
    picks the next graph hop. The default policy here is legacy parity:
    when no stop reason is set we always go to REWRITE_QUERY, matching
    ``AgentLoopController``'s rewrite-every-iter behaviour.
    EXPAND_CANDIDATES is reachable by future policies that set its
    sentinel on state directly — included in the route table so the
    edge graph compiles, but not selected by the default decide logic.
    """
    sr = state.get("stop_reason")
    if sr in ("unanswerable", "converged"):
        return ROUTE_STOP_SUFFICIENT
    if sr in ("time_cap", "token_cap", "iter_cap"):
        return ROUTE_STOP_BUDGET_EXHAUSTED
    if state.get("errors"):
        return ROUTE_ERROR
    return ROUTE_REWRITE_QUERY


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
