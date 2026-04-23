"""AgentCriticProvider seam + three concrete implementations.

Phase 6 upgrades the AUTO capability from a single-pass dispatcher into
a real agent loop. The critic is the judgment seam: given the current
iteration's ``(question, answer, retrieved_chunks)`` triple, it decides
whether the answer already addresses the question (loop terminates) or
whether some gap remains that a rewritten query could fill (loop runs
another pass).

Three providers ship here:

  * ``NoOpCritic`` — always returns ``sufficient=True``. The default
    when the loop is disabled (``agent_loop=off``); the
    ``AgentLoopController`` degenerates into a single-pass call, so the
    Phase 5 AUTO behaviour is reproduced bit-for-bit.

  * ``RuleCritic`` — deterministic heuristic. Marks an answer
    ``missing_facts`` when it is obviously empty / shorter than 40
    characters, or when it contains a canonical "I don't know" phrase
    in Korean or English. Serves as the LLM fallback and the offline
    default.

  * ``LlmCritic`` — wraps the shared ``LlmChatProvider`` and asks the
    model to pick one of five letters (A–E) via a ``judge_answer`` tool
    call. Uses thinking mode + function calling when the backend
    advertises them. On any provider failure, invalid letter, or
    schema violation, the critic falls back to ``RuleCritic`` and
    stamps ``critic_name='llm-fallback-rule'`` so downgrades show up
    in the AGENT_TRACE.

The critic NEVER raises — any failure degrades to the rule fallback.
This matches the "loop must not expand failure surface" constraint in
the Phase 6 plan: a broken LLM judge can make the loop too
conservative, but it cannot take the capability down.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence

from app.capabilities.rag.generation import RetrievedChunk

log = logging.getLogger(__name__)


# Literal of every gap_type the critic may emit. Kept stable — the
# AGENT_TRACE JSON leans on these, and the loop controller uses
# ``unanswerable`` to short-circuit.
GapType = Literal[
    "none", "missing_facts", "ambiguous", "off_topic", "unanswerable"
]


_VALID_GAP_TYPES = frozenset(
    {"none", "missing_facts", "ambiguous", "off_topic", "unanswerable"}
)

# Character floor for the rule critic. Answers this short almost never
# carry enough substance to be "sufficient" — the rule critic flags
# them even when they don't trip the explicit "don't know" markers.
_RULE_ANSWER_MIN_CHARS = 40

# Hardcoded "I don't know" phrases in the languages the project ships
# user-facing strings in. Substring match, case-insensitive. Keep the
# list small + explicit — a full open-domain refusal detector belongs in
# the LLM critic, not the regex fallback.
_RULE_INSUFFICIENT_MARKERS = (
    "정보가 없습니다",
    "모르겠습니다",
    "알 수 없습니다",
    "no information",
    "cannot answer",
    "i don't know",
    "i do not know",
)

# Max length of gap_reason surfaced through CritiqueResult.
_REASON_MAX_CHARS = 160

# How many retrieved chunks the LLM critic sees, and how much of each
# it gets. The critic prompt stays short — the heavy "read more of the
# corpus" context lives on the rewriter, not the critic.
_LLM_CONTEXT_MAX_CHUNKS = 3
_LLM_CONTEXT_CHUNK_CHARS = 400

# Max tokens the critic may spend on its tool call. The critic is a
# one-shot letter-picker; 256 tokens is plenty even with thinking mode
# reserving part of the budget.
_LLM_CRITIC_MAX_TOKENS = 256


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CritiqueResult:
    """Structured judgment emitted by an ``AgentCriticProvider``.

    Frozen so the AGENT_TRACE payload can serialize it without defensive
    copies. All fields are JSON-serializable primitives.

    Fields:
      sufficient: when True the loop terminates with the current answer.
      gap_type: categorical reason the answer isn't sufficient, or
        ``'none'`` when it is. ``'unanswerable'`` short-circuits the
        loop because rewriting a query against a corpus that doesn't
        contain the answer never helps.
      gap_reason: short human-readable reason (<=160 chars). Surfaced in
        logs + AGENT_TRACE + the rewriter prompt.
      confidence: the critic's self-reported confidence in [0.0, 1.0].
        The loop's ``min_confidence_to_stop`` budget compares against
        this when ``sufficient=True`` — low-confidence "sufficient"
        judgments can still trigger another iteration.
      critic_name: stable identifier of the provider that produced the
        verdict. Includes a ``-fallback-rule`` suffix when the LLM
        critic degraded to the rule critic.
      llm_tokens_used: total input + output tokens the critic spent on
        the LLM call, or 0 for offline critics. The loop's token budget
        accumulates this.
    """

    sufficient: bool
    gap_type: GapType
    gap_reason: str
    confidence: float
    critic_name: str
    llm_tokens_used: int

    def __post_init__(self) -> None:
        if self.gap_type not in _VALID_GAP_TYPES:
            raise ValueError(
                f"CritiqueResult.gap_type must be one of "
                f"{sorted(_VALID_GAP_TYPES)}; got {self.gap_type!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"CritiqueResult.confidence must be in [0.0, 1.0]; "
                f"got {self.confidence!r}"
            )
        if self.llm_tokens_used < 0:
            raise ValueError(
                "CritiqueResult.llm_tokens_used must be non-negative; "
                f"got {self.llm_tokens_used!r}"
            )
        # A "sufficient" verdict with a non-none gap_type, or vice versa,
        # would confuse downstream consumers of the trace. Guard here
        # rather than smearing the check over every provider.
        if self.sufficient and self.gap_type != "none":
            raise ValueError(
                "CritiqueResult: sufficient=True requires gap_type='none'; "
                f"got gap_type={self.gap_type!r}"
            )
        if not self.sufficient and self.gap_type == "none":
            raise ValueError(
                "CritiqueResult: sufficient=False requires a non-'none' "
                "gap_type"
            )

    def to_dict(self) -> dict:
        """JSON-serializable dict used by AGENT_TRACE payloads."""
        return {
            "sufficient": bool(self.sufficient),
            "gapType": self.gap_type,
            "gapReason": self.gap_reason,
            "confidence": round(self.confidence, 4),
            "criticName": self.critic_name,
            "llmTokensUsed": int(self.llm_tokens_used),
        }


# --------------------------------------------------------------------------
# Provider contract
# --------------------------------------------------------------------------


class AgentCriticProvider(ABC):
    """Converts an ``(answer, retrieved)`` pair into a ``CritiqueResult``."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> CritiqueResult:
        ...


# --------------------------------------------------------------------------
# NoOp critic — loop disabled, Phase 5 single-pass reproduction
# --------------------------------------------------------------------------


class NoOpCritic(AgentCriticProvider):
    """Always returns ``sufficient=True``.

    Picked by the registry when ``AIPIPELINE_WORKER_AGENT_LOOP=off``
    (the default until Phase 8 measures ``loop_recovery_rate``). The
    ``AgentLoopController`` terminates at iter=0 under this critic, so
    the overall AGENT output matches Phase 5 AUTO bit-for-bit.
    """

    @property
    def name(self) -> str:
        return "noop"

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> CritiqueResult:
        return CritiqueResult(
            sufficient=True,
            gap_type="none",
            gap_reason="loop disabled",
            confidence=1.0,
            critic_name=self.name,
            llm_tokens_used=0,
        )


# --------------------------------------------------------------------------
# Rule critic — offline heuristic
# --------------------------------------------------------------------------


class RuleCritic(AgentCriticProvider):
    """Deterministic offline critic.

    Flags an answer as ``missing_facts`` when any of these hold:

      * ``len(answer.strip()) < 40`` — unsophisticated length gate. The
        extractive generator routinely produces >=200-char answers even
        on weak retrievals, so anything shorter is almost always a
        fallback ("No relevant passages were retrieved ...") or an
        empty-string bug.
      * A canonical "I don't know" marker appears as a substring. The
        match is case-insensitive and covers the Korean + English
        phrases the project's user-facing strings use.

    Everything else is declared sufficient with a medium-confidence
    stop signal (0.65). The rule critic deliberately does NOT try to
    detect ambiguity or off-topic answers — that requires semantic
    judgment and belongs in the LLM critic.
    """

    @property
    def name(self) -> str:
        return "rule"

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> CritiqueResult:
        stripped = (answer or "").strip()
        if not stripped:
            return CritiqueResult(
                sufficient=False,
                gap_type="missing_facts",
                gap_reason="answer is empty",
                confidence=0.9,
                critic_name=self.name,
                llm_tokens_used=0,
            )
        if len(stripped) < _RULE_ANSWER_MIN_CHARS:
            return CritiqueResult(
                sufficient=False,
                gap_type="missing_facts",
                gap_reason=(
                    f"answer length {len(stripped)} < {_RULE_ANSWER_MIN_CHARS} "
                    "chars"
                )[:_REASON_MAX_CHARS],
                confidence=0.75,
                critic_name=self.name,
                llm_tokens_used=0,
            )
        lowered = stripped.lower()
        for marker in _RULE_INSUFFICIENT_MARKERS:
            if marker in lowered:
                return CritiqueResult(
                    sufficient=False,
                    gap_type="missing_facts",
                    gap_reason=(
                        f"answer contains insufficient marker {marker!r}"
                    )[:_REASON_MAX_CHARS],
                    confidence=0.8,
                    critic_name=self.name,
                    llm_tokens_used=0,
                )
        return CritiqueResult(
            sufficient=True,
            gap_type="none",
            gap_reason="answer passes rule heuristics",
            confidence=0.65,
            critic_name=self.name,
            llm_tokens_used=0,
        )


# --------------------------------------------------------------------------
# LLM critic — function calling + thinking mode when available
# --------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = (
    "You judge whether an answer fully addresses the question, grounded "
    "in the retrieved passages. Respond via the 'judge_answer' tool "
    "with exactly one of: A=sufficient, B=missing key facts, "
    "C=ambiguous/weakly grounded, D=off-topic, E=unanswerable from corpus."
)

_LLM_TOOL_NAME = "judge_answer"

_LLM_TOOL_DESCRIPTION = (
    "Emit the sufficiency judgment for the answer. Always call this tool."
)

_LLM_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "letter": {
            "type": "string",
            "enum": ["A", "B", "C", "D", "E"],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["letter", "reason", "confidence"],
    "additionalProperties": False,
}

_LLM_SCHEMA_HINT = (
    '{"letter": "A"|"B"|"C"|"D"|"E", '
    '"reason": string (<=160 chars), "confidence": number in [0,1]}'
)

# Map letter -> (sufficient, gap_type). A is the only sufficient letter;
# B/C/D/E are all failure modes the loop may react to differently.
_LETTER_TO_VERDICT: dict[str, tuple[bool, GapType]] = {
    "A": (True, "none"),
    "B": (False, "missing_facts"),
    "C": (False, "ambiguous"),
    "D": (False, "off_topic"),
    "E": (False, "unanswerable"),
}


class LlmCritic(AgentCriticProvider):
    """LLM-backed critic that falls back to the rule critic on any failure.

    Uses ``chat_tools`` with the ``judge_answer`` tool when the backend
    advertises ``function_calling``; otherwise falls back to
    ``chat_json`` with a stern schema hint. Thinking mode is enabled
    iff the backend advertises it AND the caller passed
    ``enable_thinking=True`` at construction.

    Failure modes that fall back to rule instead of raising:
      * ``LlmChatError`` from the underlying provider (network, timeout,
        invalid JSON, empty response).
      * Schema violation at our layer (missing letter, letter not in
        A-E, wrong types).

    The fallback's ``critic_name`` is ``"llm-fallback-rule"`` — distinct
    from both ``"llm"`` (clean LLM path) and ``"rule"`` (configured rule
    critic) so the metrics layer can measure LLM downgrade rate
    separately from baseline rule usage.
    """

    def __init__(
        self,
        chat: Any,
        *,
        enable_thinking: bool = True,
        max_tokens: int = _LLM_CRITIC_MAX_TOKENS,
    ) -> None:
        from app.clients.llm_chat import LlmChatProvider  # local — avoids cycles

        if not isinstance(chat, LlmChatProvider):
            raise TypeError(
                "LlmCritic requires an LlmChatProvider instance; "
                f"got {type(chat).__name__}"
            )
        self._chat = chat
        self._enable_thinking = bool(enable_thinking)
        self._max_tokens = int(max_tokens)
        self._rule_fallback = RuleCritic()

    @property
    def name(self) -> str:
        return f"llm-{self._chat.name}"

    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> CritiqueResult:
        from app.clients.llm_chat import (
            ChatMessage,
            ChatToolSpec,
            LlmChatError,
        )

        user_content = _build_critic_user_content(
            question=question,
            answer=answer,
            retrieved=retrieved,
        )
        messages = [
            ChatMessage(role="system", content=_LLM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]
        caps = self._chat.capabilities
        enable_thinking = self._enable_thinking and bool(caps.get("thinking"))
        uses_tools = bool(caps.get("function_calling"))

        tokens_used = 0
        try:
            if uses_tools:
                tool = ChatToolSpec(
                    name=_LLM_TOOL_NAME,
                    description=_LLM_TOOL_DESCRIPTION,
                    parameters=_LLM_TOOL_SCHEMA,
                )
                result = self._chat.chat_tools(
                    messages,
                    [tool],
                    max_tokens=self._max_tokens,
                    enable_thinking=enable_thinking,
                )
                tokens_used = int(result.tokens_in or 0) + int(
                    result.tokens_out or 0
                )
                data = _extract_tool_arguments(result)
            else:
                data = self._chat.chat_json(
                    messages,
                    schema_hint=_LLM_SCHEMA_HINT,
                    max_tokens=self._max_tokens,
                    enable_thinking=enable_thinking,
                )
        except LlmChatError as ex:
            log.warning(
                "LlmCritic: provider failure, falling back to rule (%s)",
                ex,
            )
            return self._fallback(
                question=question, answer=answer, retrieved=retrieved
            )

        try:
            letter, reason, confidence = _materialize_verdict(data)
        except ValueError as ex:
            log.warning(
                "LlmCritic: invalid response, falling back to rule (%s)",
                ex,
            )
            return self._fallback(
                question=question, answer=answer, retrieved=retrieved
            )

        sufficient, gap_type = _LETTER_TO_VERDICT[letter]
        return CritiqueResult(
            sufficient=sufficient,
            gap_type=gap_type,
            gap_reason=reason[:_REASON_MAX_CHARS],
            confidence=confidence,
            critic_name=self.name,
            llm_tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------

    def _fallback(
        self,
        *,
        question: str,
        answer: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> CritiqueResult:
        base = self._rule_fallback.evaluate(
            question=question, answer=answer, retrieved=retrieved
        )
        # Re-wrap so critic_name records the downgrade for metrics.
        return CritiqueResult(
            sufficient=base.sufficient,
            gap_type=base.gap_type,
            gap_reason=base.gap_reason,
            confidence=base.confidence,
            critic_name="llm-fallback-rule",
            llm_tokens_used=base.llm_tokens_used,
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_critic_user_content(
    *,
    question: str,
    answer: str,
    retrieved: Sequence[RetrievedChunk],
) -> str:
    """Render the user turn for the LLM critic.

    Keeps the payload compact so the critic is cheap even on 128K-context
    models: the three top chunks are clipped to 400 chars each, and the
    answer is passed through verbatim so phrases like "정보가 없습니다"
    stay visible to the judge.
    """
    chunk_blocks: List[str] = []
    for i, chunk in enumerate(retrieved[:_LLM_CONTEXT_MAX_CHUNKS], start=1):
        text = (chunk.text or "").strip()
        if len(text) > _LLM_CONTEXT_CHUNK_CHARS:
            text = text[: _LLM_CONTEXT_CHUNK_CHARS - 3] + "..."
        chunk_blocks.append(
            f"[{i}] doc={chunk.doc_id} section={chunk.section}\n{text}"
        )
    chunks_rendered = (
        "\n\n".join(chunk_blocks) if chunk_blocks else "(no retrieved passages)"
    )
    return (
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Retrieved passages (top {len(chunk_blocks)}):\n{chunks_rendered}"
    )


def _extract_tool_arguments(result: Any) -> dict:
    """Pull the judge arguments out of a ChatResult from chat_tools."""
    from app.clients.llm_chat import LlmChatError

    tool_call = getattr(result, "tool_call", None)
    if tool_call:
        args = tool_call.get("arguments")
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError as ex:
                raise LlmChatError(
                    f"tool_call arguments were non-JSON: {ex}"
                ) from ex
            if isinstance(parsed, dict):
                return parsed
            raise LlmChatError(
                f"tool_call arguments must be a JSON object; got "
                f"{type(parsed).__name__}"
            )
        raise LlmChatError("tool_call arguments were missing or malformed.")

    text = getattr(result, "text", None)
    if text and text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as ex:
            raise LlmChatError(
                f"chat_tools returned non-JSON text: {ex}"
            ) from ex
        if isinstance(parsed, dict):
            return parsed
        raise LlmChatError(
            f"chat_tools text must be a JSON object; got {type(parsed).__name__}"
        )
    raise LlmChatError("chat_tools returned neither tool_call nor text.")


def _materialize_verdict(data: dict) -> tuple[str, str, float]:
    """Validate the LLM payload and coerce to (letter, reason, confidence)."""
    if not isinstance(data, dict):
        raise ValueError(
            f"LLM response must be a JSON object; got {type(data).__name__}"
        )

    raw_letter = data.get("letter")
    if not isinstance(raw_letter, str):
        raise ValueError("'letter' must be a string")
    letter = raw_letter.strip().upper()
    if letter not in _LETTER_TO_VERDICT:
        raise ValueError(
            f"'letter' must be one of {sorted(_LETTER_TO_VERDICT)}; "
            f"got {raw_letter!r}"
        )

    raw_reason = data.get("reason", "")
    if not isinstance(raw_reason, str):
        raise ValueError("'reason' must be a string")
    reason = raw_reason.strip() or "no reason given"

    raw_conf = data.get("confidence")
    if isinstance(raw_conf, bool):
        # bool is a subclass of int; reject it so a stray "true" doesn't
        # silently become 1.0.
        raise ValueError("'confidence' must be a number, not bool")
    if not isinstance(raw_conf, (int, float)):
        raise ValueError("'confidence' must be a number in [0.0, 1.0]")
    confidence = float(raw_conf)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"'confidence' must be in [0.0, 1.0]; got {confidence!r}"
        )
    return letter, reason, confidence
