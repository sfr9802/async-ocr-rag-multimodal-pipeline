"""Agent synthesizer — final answer grounded on the UNION of all iters.

The loop gathers retrieved chunks across multiple iterations. The
synthesizer hands the UNION of those chunks to a standard
``GenerationProvider`` and returns its answer — this is the core
quality win of the loop.

There is no LLM knowledge inside the synthesizer: it reuses the same
generator the RAG / MULTIMODAL capabilities use (``ExtractiveGenerator``
by default, ``ClaudeGenerationProvider`` when the registry wires the
Claude backend). That keeps the AGENT capability a drop-in upgrade —
the generator sees the same chunk shape it already handles.

If the loop somehow produced zero aggregated chunks (every iteration's
execute_fn returned an empty list), the synthesizer falls back to
returning the last iteration's answer unchanged so the client never
sees an empty FINAL_RESPONSE. Zero-chunk aggregation is unusual in
production but easy to hit in tests, and the generator would just
produce its "no passages" stub text anyway — returning the live loop
answer preserves more information.
"""

from __future__ import annotations

import logging

from app.capabilities.agent.loop import LoopOutcome
from app.capabilities.rag.generation import GenerationProvider

log = logging.getLogger(__name__)


class AgentSynthesizer:
    """Final-answer composer for the agent loop.

    Wraps a ``GenerationProvider`` so the synthesis step can be swapped
    independently of the critic / rewriter / loop controller.
    """

    def __init__(self, generator: GenerationProvider) -> None:
        self._generator = generator

    @property
    def generator_name(self) -> str:
        return getattr(self._generator, "name", type(self._generator).__name__)

    def synthesize(self, question: str, outcome: LoopOutcome) -> str:
        """Return the final grounded answer for ``outcome``.

        Uses ``outcome.aggregated_chunks`` — the deduped union of every
        iteration's retrievals — as the grounding pool. On the rare
        zero-chunks case the last iteration's answer is returned as-is
        (better signal than the generator's "no passages" stub).
        """
        if not outcome.aggregated_chunks:
            log.info(
                "AgentSynthesizer: outcome has zero aggregated chunks "
                "(stop_reason=%s steps=%d). Returning last loop answer.",
                outcome.stop_reason, len(outcome.steps),
            )
            return outcome.final_answer or ""

        try:
            return self._generator.generate(
                question, list(outcome.aggregated_chunks)
            )
        except Exception as ex:
            # The generator is a boundary to a potentially remote LLM
            # (Claude) — if it fails here we must still deliver
            # something. Falling back to the loop's last live answer
            # keeps the loop's promise that a failure never surfaces as
            # a capability-level error.
            log.warning(
                "AgentSynthesizer: generator %s failed (%s: %s). "
                "Returning last loop answer.",
                self.generator_name,
                type(ex).__name__, ex,
            )
            return outcome.final_answer or ""
