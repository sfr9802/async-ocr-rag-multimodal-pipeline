"""Experimental LangGraph-based agent loop backend.

Selected by ``settings.agent_loop_backend = 'graph'`` (default is
``'legacy'`` — the in-process AgentLoopController shipped since Phase 6
stays the production runner). Both backends share the critic /
rewriter / parser / budget seams and emit identical AGENT_TRACE /
RETRIEVAL_RESULT_AGG / FINAL_RESPONSE artifact shapes, so downstream
consumers (TaskRunner callbacks, Spring repo, eval harness) see no
schema drift across the toggle.

Public surface is the ``AgentLoopGraph`` adapter in ``adapters.py``;
finer-grained pieces (state schema, node factories, builder) live in
sibling modules and exist primarily for tests + future extension.
"""

from app.capabilities.agent.graph_loop.adapters import AgentLoopGraph

__all__ = ["AgentLoopGraph"]
