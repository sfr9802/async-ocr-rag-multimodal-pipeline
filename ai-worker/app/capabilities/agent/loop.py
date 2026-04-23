"""Agent loop controller.

Phase 6 upgrades AGENT from a single-pass dispatcher into an iterative
agent. The controller drives this cycle::

    iter=0:  execute(initial_query) -> (answer, chunks, tokens)
             critic.evaluate(question, answer, chunks)
             if sufficient OR budget breached -> stop
    iter=N:  rewriter.rewrite(...) -> new_parsed_query
             execute(new_parsed_query) -> (answer, chunks, tokens)
             critic.evaluate(...)
             if sufficient OR budget breached -> stop

    final_answer := synthesized over UNION of all iters' chunks
                    (see synthesizer.py).

The controller is pure glue — it depends on three provider seams
(``AgentCriticProvider``, ``QueryRewriterProvider``,
``QueryParserProvider``) and one callback (``execute_fn``). It holds no
LLM state of its own. This lets tests exercise every stop_reason branch
without any network or model calls.

Stop reasons (stable strings, referenced by tests + trace consumers):

  ``converged``    - critic said sufficient AND confidence >= min_conf_to_stop.
  ``iter_cap``     - ran ``max_iter`` iterations without converging.
  ``time_cap``     - ``max_total_ms`` elapsed before convergence.
  ``token_cap``    - accumulated LLM tokens crossed ``max_llm_tokens``.
  ``unanswerable`` - critic said ``gap_type='unanswerable'`` at any iter.

The loop never raises. Any exception from ``execute_fn`` aborts the
iteration with a best-effort stop_reason (``iter_cap`` if we already
had at least one step, else re-raises so the capability can fall back
to Phase 5 single-pass behaviour). Critic / rewriter errors are
already swallowed inside their own implementations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Tuple

from app.capabilities.agent.critic import AgentCriticProvider, CritiqueResult
from app.capabilities.agent.rewriter import QueryRewriterProvider
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider

log = logging.getLogger(__name__)


StopReason = Literal[
    "converged", "iter_cap", "time_cap", "token_cap", "unanswerable"
]


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopBudget:
    """Hard limits that keep the loop bounded.

    Every limit is inclusive of the budget (``<=`` checks). The defaults
    are tuned for the Phase 6 acceptance criteria — three iterations is
    enough to recover from a weak initial retrieval without paying
    runaway tokens, 15 s wall-clock is below the TaskRunner's default
    job-level deadline, and 4 K LLM tokens covers one critic + one
    rewriter call per iter plus slack.
    """

    max_iter: int = 3
    max_total_ms: int = 15_000
    max_llm_tokens: int = 4_000
    min_confidence_to_stop: float = 0.75

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError("LoopBudget.max_iter must be >= 1")
        if self.max_total_ms < 1:
            raise ValueError("LoopBudget.max_total_ms must be >= 1")
        if self.max_llm_tokens < 0:
            raise ValueError("LoopBudget.max_llm_tokens must be >= 0")
        if not (0.0 <= self.min_confidence_to_stop <= 1.0):
            raise ValueError(
                "LoopBudget.min_confidence_to_stop must be in [0.0, 1.0]"
            )

    def to_dict(self) -> dict:
        return {
            "maxIter": int(self.max_iter),
            "maxTotalMs": int(self.max_total_ms),
            "maxLlmTokens": int(self.max_llm_tokens),
            "minConfidenceToStop": round(self.min_confidence_to_stop, 4),
        }


@dataclass(frozen=True)
class LoopStep:
    """Record of a single iteration, suitable for AGENT_TRACE embedding."""

    iter: int
    query: ParsedQuery
    retrieved_chunk_ids: List[str]
    answer_preview: str  # clipped to 300ch
    critique: CritiqueResult
    step_ms: float
    llm_tokens_used: int

    def to_dict(self) -> dict:
        return {
            "iter": int(self.iter),
            "query": self.query.to_dict(),
            "retrievedChunkIds": list(self.retrieved_chunk_ids),
            "answerPreview": self.answer_preview,
            "critique": self.critique.to_dict(),
            "stepMs": round(float(self.step_ms), 3),
            "llmTokensUsed": int(self.llm_tokens_used),
        }


@dataclass(frozen=True)
class LoopOutcome:
    """Final result surfaced back to ``AgentCapability``.

    ``aggregated_chunks`` is the UNION of retrieved_chunks across every
    step, deduped by ``chunk_id`` (first occurrence wins so the
    highest-ranked variant of a chunk keeps its original score /
    rerank_score). The synthesizer grounds the final answer on this
    pool — that is the core quality win of the loop.
    """

    steps: List[LoopStep]
    stop_reason: StopReason
    final_answer: str
    aggregated_chunks: List[RetrievedChunk]
    total_ms: float
    total_llm_tokens: int

    def to_dict(self) -> dict:
        return {
            "stopReason": self.stop_reason,
            "stepCount": len(self.steps),
            "totalMs": round(float(self.total_ms), 3),
            "totalLlmTokens": int(self.total_llm_tokens),
            "aggregatedChunkCount": len(self.aggregated_chunks),
            "finalAnswerLength": len(self.final_answer or ""),
            "steps": [s.to_dict() for s in self.steps],
        }


# --------------------------------------------------------------------------
# ExecuteFn signature
# --------------------------------------------------------------------------


# Every call should return ``(answer, retrieved_chunks, llm_tokens_used)``.
# The capability wires this to a function that calls the routed-to
# sub-capability with a query embedded in its CapabilityInput so the
# existing RAG / MULTIMODAL pipelines keep running untouched.
ExecuteFn = Callable[
    [ParsedQuery], Tuple[str, List[RetrievedChunk], int]
]


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


_ANSWER_PREVIEW_CHARS = 300


class AgentLoopController:
    """Drives the critic / rewriter / execute cycle under a fixed budget.

    The controller holds no live model state — it takes the critic,
    rewriter, parser, and budget at construction and delegates every
    outside call through ``execute_fn`` so tests and integration code
    inject their own sub-capability wiring.
    """

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

    @property
    def budget(self) -> LoopBudget:
        return self._budget

    # ------------------------------------------------------------------

    def run(
        self,
        *,
        question: str,
        initial_parsed_query: ParsedQuery,
        execute_fn: ExecuteFn,
    ) -> LoopOutcome:
        """Execute the loop up to the budget and return the outcome.

        ``initial_parsed_query`` is the query the first iteration runs.
        The caller supplies it because the surrounding capability may
        already have parsed the user's text (e.g. via a router decision)
        and re-parsing would throw away that work.
        """
        started_at = time.monotonic()
        steps: List[LoopStep] = []
        aggregated: List[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        total_tokens = 0
        stop_reason: Optional[StopReason] = None
        current_query = initial_parsed_query
        last_answer = ""

        for iter_index in range(self._budget.max_iter):
            step_started = time.monotonic()
            answer, chunks, exec_tokens = execute_fn(current_query)
            exec_tokens = max(0, int(exec_tokens or 0))
            total_tokens += exec_tokens
            last_answer = answer or ""

            # Aggregate chunks by chunk_id so the synthesizer sees every
            # unique passage gathered across iterations. Preserves the
            # order in which each chunk was first seen — the earliest
            # iteration's rank wins on ties.
            for chunk in chunks or []:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                aggregated.append(chunk)

            critique = self._critic.evaluate(
                question=question,
                answer=last_answer,
                retrieved=chunks or [],
            )
            total_tokens += max(0, int(critique.llm_tokens_used or 0))

            step_ms = _elapsed_ms(step_started)
            step = LoopStep(
                iter=iter_index,
                query=current_query,
                retrieved_chunk_ids=[c.chunk_id for c in (chunks or [])],
                answer_preview=_clip(last_answer, _ANSWER_PREVIEW_CHARS),
                critique=critique,
                step_ms=step_ms,
                llm_tokens_used=exec_tokens
                + max(0, int(critique.llm_tokens_used or 0)),
            )
            steps.append(step)

            log.info(
                "AGENT iter=%d critic=%s sufficient=%s gap=%s conf=%.2f "
                "answer_len=%d chunks=%d exec_tokens=%d step_ms=%.1f",
                iter_index,
                critique.critic_name,
                critique.sufficient,
                critique.gap_type,
                critique.confidence,
                len(last_answer),
                len(chunks or []),
                exec_tokens,
                step_ms,
            )

            # --- stop checks: unanswerable, converged, budget caps ---
            if critique.gap_type == "unanswerable":
                stop_reason = "unanswerable"
                break

            if (
                critique.sufficient
                and critique.confidence
                >= self._budget.min_confidence_to_stop
            ):
                stop_reason = "converged"
                break

            elapsed_total_ms = _elapsed_ms(started_at)
            if elapsed_total_ms >= self._budget.max_total_ms:
                stop_reason = "time_cap"
                break
            if total_tokens >= self._budget.max_llm_tokens:
                stop_reason = "token_cap"
                break
            if iter_index + 1 >= self._budget.max_iter:
                stop_reason = "iter_cap"
                break

            # --- prepare the next iteration's query via the rewriter ---
            current_query = self._rewriter.rewrite(
                original=question,
                prev_answer=last_answer,
                gap_reason=critique.gap_reason,
                already_retrieved_chunks=list(aggregated),
                parser=self._parser,
            )

        if stop_reason is None:
            # Defensive: _budget.max_iter >= 1 so the loop body ran at
            # least once and at least one branch above set stop_reason.
            # Keep the fallback for future edits that add a ``continue``
            # path without updating the break vocabulary.
            stop_reason = "iter_cap"

        total_ms = _elapsed_ms(started_at)
        return LoopOutcome(
            steps=steps,
            stop_reason=stop_reason,
            final_answer=last_answer,
            aggregated_chunks=aggregated,
            total_ms=total_ms,
            total_llm_tokens=total_tokens,
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _elapsed_ms(started_monotonic: float) -> float:
    return round((time.monotonic() - started_monotonic) * 1000.0, 3)


def _clip(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
