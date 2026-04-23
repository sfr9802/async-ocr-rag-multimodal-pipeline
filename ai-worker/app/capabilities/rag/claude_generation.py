"""Claude-backed grounded generation provider.

Replaces the extractive heuristic with actual LLM-generated answers
that cite retrieved passages. The extractive generator stays as the
CI / offline / test fallback and as the automatic fallback when the
Claude API is unreachable (controlled by fallback_on_error).

Activation:

    AIPIPELINE_WORKER_RAG_GENERATOR=claude
    AIPIPELINE_WORKER_ANTHROPIC_API_KEY=sk-ant-...

Output format: identical 3-part markdown structure to ExtractiveGenerator
(Short answer / Supporting passages / Sources) so downstream
FINAL_RESPONSE consumers need zero changes.
"""

from __future__ import annotations

import logging
import time
from typing import List

from app.capabilities.rag.generation import (
    ExtractiveGenerator,
    GenerationProvider,
    RetrievedChunk,
)

log = logging.getLogger(__name__)


class GenerationError(Exception):
    """Typed generation failure — separates Claude API errors from general
    exceptions so the fallback policy can distinguish retryable failures."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_SYSTEM_PROMPT = (
    "You are a retrieval-augmented answer generator.\n"
    "Answer ONLY from the supplied passages. Cite every claim as "
    "`[doc_id#section]`.\n"
    "If the passages don't contain the answer, respond exactly:\n"
    "Korean query: '제공된 자료에서 답을 찾을 수 없습니다.'\n"
    "English query: 'The provided sources do not contain an answer.'\n"
    "Never use outside knowledge. Never hallucinate document IDs.\n\n"
    "Format your answer in three sections:\n"
    "1. **Short answer:** — a concise 1-3 sentence answer\n"
    "2. **Supporting passages:** — numbered list citing each passage used\n"
    "3. **Sources:** — comma-separated list of unique doc_ids"
)


class ClaudeGenerationProvider(GenerationProvider):
    """Claude LLM-backed generation with extractive fallback.

    Dependencies: `anthropic` SDK (pip install anthropic>=0.40.0).
    Requires AIPIPELINE_WORKER_ANTHROPIC_API_KEY at runtime.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        timeout_seconds: float = 60.0,
        fallback_on_error: bool = True,
    ) -> None:
        import anthropic  # local import — registry catches ImportError

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
        )
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._fallback_on_error = fallback_on_error
        self._extractive_fallback = ExtractiveGenerator()

    @property
    def name(self) -> str:
        return "claude-generation-v1"

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "No relevant passages were retrieved for your query.\n\n"
                f"> {query}"
            )

        try:
            return self._call_claude(query, chunks)
        except Exception as ex:
            if self._fallback_on_error:
                log.warning(
                    "ClaudeGenerationProvider failed, falling back to "
                    "extractive: %s: %s",
                    type(ex).__name__, ex,
                )
                return self._extractive_fallback.generate(query, chunks)
            # Re-raise as a CapabilityError-compatible exception.
            from app.capabilities.base import CapabilityError

            raise CapabilityError(
                "GENERATION_API_FAILED",
                f"Claude generation failed and fallback is disabled: "
                f"{type(ex).__name__}: {ex}",
            ) from ex

    def _call_claude(self, query: str, chunks: List[RetrievedChunk]) -> str:
        import anthropic

        user_message = _build_user_message(query, chunks)
        started_at = time.perf_counter()

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APITimeoutError as ex:
            raise GenerationError(
                "GENERATION_TIMEOUT",
                f"Claude generation timed out after {self._timeout_seconds}s: {ex}",
            ) from ex
        except anthropic.RateLimitError as ex:
            raise GenerationError(
                "GENERATION_RATE_LIMIT",
                f"Claude generation rate-limited: {ex}",
            ) from ex
        except anthropic.APIStatusError as ex:
            raise GenerationError(
                "GENERATION_API_ERROR",
                f"Claude generation API error {ex.status_code}: {ex}",
            ) from ex

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        if not raw_text.strip():
            raise GenerationError(
                "GENERATION_EMPTY_RESPONSE",
                "Claude generation returned an empty response.",
            )

        log.info(
            "ClaudeGenerationProvider generated answer: model=%s "
            "latency_ms=%.2f answer_len=%d chunk_count=%d",
            self._model, elapsed_ms, len(raw_text), len(chunks),
        )

        return raw_text.strip()


def _build_user_message(query: str, chunks: List[RetrievedChunk]) -> str:
    """Build the user message with query + numbered passage context."""
    lines = [f"질문: {query}", "", "관련 자료:"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] {chunk.doc_id}#{chunk.section} (score={chunk.score:.3f})"
        )
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)
