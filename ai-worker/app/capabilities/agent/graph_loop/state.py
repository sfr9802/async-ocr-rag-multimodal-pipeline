"""LangGraph state for the experimental agent loop backend.

The state is a TypedDict because LangGraph natively merges partial-state
returns from nodes — every node returns a dict of the fields it
touched, LangGraph applies them on top of the running state. Frozen
dataclass instances (``LoopStep``, ``ParsedQuery``, ``RetrievedChunk``)
flow through unchanged: TypedDict only constrains keys, not values.

The shape is intentionally a superset of what ``AgentLoopController``
tracks in local variables. The legacy controller owns just enough
state to drive the iter loop; the graph backend additionally records
optional caller context (``job_id`` / ``capability`` / ``action``) and
per-stage breadcrumbs (``retrieval_reports`` / ``quality_history`` /
``rewrite_history`` / ``trace`` / ``errors``) so the graph nodes can
log and decide without reaching back into the calling capability. All
the extras default to safe empties — every existing test that exercises
only the legacy-parity surface keeps working without populating them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from app.capabilities.agent.loop import LoopStep
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery


# Conditional-edge labels emitted by ``decide_route`` and the
# initial-retrieve / retrieve-again post-routes. Stable strings —
# tests + log consumers grep for them, so renames need a deliberate
# bump.
ROUTE_STOP_SUFFICIENT = "STOP_SUFFICIENT"
ROUTE_STOP_BUDGET_EXHAUSTED = "STOP_BUDGET_EXHAUSTED"
ROUTE_REWRITE_QUERY = "REWRITE_QUERY"
ROUTE_EXPAND_CANDIDATES = "EXPAND_CANDIDATES"
ROUTE_ERROR = "ERROR"


GraphRoute = Literal[
    "STOP_SUFFICIENT",
    "STOP_BUDGET_EXHAUSTED",
    "REWRITE_QUERY",
    "EXPAND_CANDIDATES",
    "ERROR",
]


class AgentLoopGraphState(TypedDict, total=False):
    """Per-run state for the agent loop graph backend.

    ``total=False`` so node handlers can return only the slice they
    touched — LangGraph shallow-merges partial returns onto the running
    state. Every field has an obvious default the adapter pre-populates;
    nodes never need to guard against missing keys with ``KeyError``.
    """

    # --- caller / capability context (populated when known, otherwise empty) ---
    job_id: str
    capability: str
    action: str

    # --- query state ---
    original_query: str
    current_query: ParsedQuery

    # --- counters / budget anchors ---
    iteration: int
    max_iterations: int
    started_at: float        # time.monotonic() at graph entry
    total_tokens: int

    # --- per-iter snapshots (overwritten on each retrieve call) ---
    last_answer: str
    last_chunks: List[RetrievedChunk]
    last_exec_tokens: int
    critic_decision: Optional[Dict[str, Any]]

    # --- aggregations across iterations ---
    retrieval_reports: List[Dict[str, Any]]   # one entry per retrieve call
    candidate_pool: List[RetrievedChunk]      # UNION dedup by chunk_id
    seen_chunk_ids: List[str]                 # ordered (first-occurrence-wins)
    quality_history: List[Dict[str, Any]]     # score_quality_node output
    rewrite_history: List[ParsedQuery]        # rewrite_node output
    steps: List[LoopStep]                     # one per critic verdict

    # --- terminal ---
    stop_reason: Optional[str]
    final_answer: str

    # --- breadcrumbs for graph-side debugging ---
    trace: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
