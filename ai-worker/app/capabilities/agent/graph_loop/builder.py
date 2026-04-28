"""StateGraph builder for the agent loop graph backend.

The graph is rebuilt per ``run`` invocation because each call binds a
fresh ``execute_fn`` closure — LangGraph nodes cannot share executor
state across runs without serialising callables, which we deliberately
avoid (the legacy contract uses a closure too). Compile cost is a few
milliseconds per run and is well below ``LoopBudget.max_total_ms``
even with a 100x safety factor.

Edge layout:

    START
      │
      ▼
    initial_retrieve ──errors? ─► budget_exhausted ─► synthesize ─► END
      │
      ▼
    aggregate_candidates
      │
      ▼
    score_quality
      │
      ▼
    critic
      │
      ▼
    decide_next_action ──┬──STOP_SUFFICIENT───────► synthesize ─► END
                         ├──STOP_BUDGET_EXHAUSTED─► synthesize ─► END
                         ├──REWRITE_QUERY ──► rewrite ──► retrieve_again
                         ├──EXPAND_CANDIDATES ──────────► retrieve_again
                         └──ERROR ──────────► budget_exhausted ─► synthesize

    retrieve_again ──errors? ─► budget_exhausted ─► synthesize ─► END
      │
      ▼
    aggregate_candidates  (back-edge — re-enters score_quality / critic)

The default ``decide_route`` policy never selects EXPAND_CANDIDATES;
the edge is included so a future policy can set the sentinel on state
without rebuilding the graph topology.
"""

from __future__ import annotations

from typing import Any

from app.capabilities.agent.critic import AgentCriticProvider
from app.capabilities.agent.graph_loop.nodes import (
    GraphExecuteFn,
    decide_route,
    make_aggregate_candidates_node,
    make_budget_exhausted_node,
    make_critic_node,
    make_decide_next_action_node,
    make_initial_retrieve_node,
    make_retrieve_again_node,
    make_rewrite_node,
    make_score_quality_node,
    make_synthesize_node,
    post_retrieve_route,
)
from app.capabilities.agent.graph_loop.state import (
    ROUTE_ERROR,
    ROUTE_EXPAND_CANDIDATES,
    ROUTE_REWRITE_QUERY,
    ROUTE_STOP_BUDGET_EXHAUSTED,
    ROUTE_STOP_SUFFICIENT,
    AgentLoopGraphState,
)
from app.capabilities.agent.loop import LoopBudget
from app.capabilities.agent.rewriter import QueryRewriterProvider
from app.capabilities.rag.query_parser import QueryParserProvider


def build_agent_loop_graph(
    *,
    critic: AgentCriticProvider,
    rewriter: QueryRewriterProvider,
    parser: QueryParserProvider,
    budget: LoopBudget,
    execute_fn: GraphExecuteFn,
) -> Any:
    """Compile a StateGraph that drives one agent-loop run.

    Returns a compiled graph ready for ``invoke``. The caller (the
    ``AgentLoopGraph`` adapter) builds the initial state and extracts
    the ``LoopOutcome`` from the terminal state — that boundary is what
    keeps the public ``run`` signature compatible with the legacy
    controller.
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentLoopGraphState)

    g.add_node("initial_retrieve", make_initial_retrieve_node(execute_fn))
    g.add_node("aggregate_candidates", make_aggregate_candidates_node())
    g.add_node("score_quality", make_score_quality_node())
    g.add_node("critic", make_critic_node(critic))
    g.add_node("decide_next_action", make_decide_next_action_node(budget))
    g.add_node("rewrite", make_rewrite_node(rewriter, parser))
    g.add_node("retrieve_again", make_retrieve_again_node(execute_fn))
    g.add_node("synthesize", make_synthesize_node())
    g.add_node("budget_exhausted", make_budget_exhausted_node())

    g.set_entry_point("initial_retrieve")

    # Initial retrieve: failure short-circuits to budget_exhausted so
    # the critic chain never sees a half-formed iter 0.
    g.add_conditional_edges(
        "initial_retrieve",
        post_retrieve_route,
        {
            "OK": "aggregate_candidates",
            ROUTE_ERROR: "budget_exhausted",
        },
    )

    g.add_edge("aggregate_candidates", "score_quality")
    g.add_edge("score_quality", "critic")
    g.add_edge("critic", "decide_next_action")

    g.add_conditional_edges(
        "decide_next_action",
        decide_route,
        {
            ROUTE_STOP_SUFFICIENT: "synthesize",
            ROUTE_STOP_BUDGET_EXHAUSTED: "synthesize",
            ROUTE_REWRITE_QUERY: "rewrite",
            ROUTE_EXPAND_CANDIDATES: "retrieve_again",
            ROUTE_ERROR: "budget_exhausted",
        },
    )

    g.add_edge("rewrite", "retrieve_again")

    # Retrieve-again: same error short-circuit as initial_retrieve so
    # earlier-iter steps stay on state but no half-iter slips into the
    # critic chain.
    g.add_conditional_edges(
        "retrieve_again",
        post_retrieve_route,
        {
            "OK": "aggregate_candidates",
            ROUTE_ERROR: "budget_exhausted",
        },
    )

    g.add_edge("budget_exhausted", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()
