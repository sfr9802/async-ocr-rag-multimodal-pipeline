"""AgentRouterProvider seam + two concrete implementations.

The agent router is the pre-capability stage that inspects a submitted
``(text, file)`` pair and decides which downstream capability should
actually run. Phase 3 ships a single-pass dispatcher: one decision per
job, no loop, no critic, no retry. Phase 6 upgrades the same seam into
a real agent loop without rewriting the callers.

Two concrete implementations ship here:

  * ``RuleBasedAgentRouter`` — deterministic five-branch decision tree
    over the input shape. Handles the ~95% of cases where routing is
    obvious from the (text-length, file-type) pair alone. Always safe,
    zero external dependencies, serves as the fallback for the LLM
    router below.

  * ``LlmAgentRouter`` — wraps a shared ``LlmChatProvider`` and asks
    the model to pick between the five actions. Uses native function
    calling when the backend advertises it; otherwise falls back to
    ``chat_json`` with a stern schema hint. On provider failure,
    low-confidence output, or an invalid action, it degrades to the
    rule-based router and stamps ``router_name='{llm}-fallback-rule'``
    so the downgrade is visible in traces / metrics.

The router NEVER executes a capability itself — it only returns an
``AgentDecision``. The ``AutoCapability`` is the component that turns a
decision into a delegated ``.run()`` call.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider

log = logging.getLogger(__name__)


# Literal of every action the router may return. "direct_answer" is
# reserved for LlmAgentRouter — RuleBasedAgentRouter never emits it.
AgentAction = Literal["rag", "ocr", "multimodal", "direct_answer", "clarify"]


_VALID_ACTIONS = frozenset({"rag", "ocr", "multimodal", "direct_answer", "clarify"})


# File-type gate used by both routers. Mirrors the worker-side OCR /
# MULTIMODAL classifier lists + the core-api validator's allow-list so
# the three stay in lockstep. Anything outside this set is treated by
# the rule router as "no routable file".
_SUPPORTED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/jpg"})
_SUPPORTED_PDF_MIMES = frozenset({"application/pdf", "application/x-pdf"})
_SUPPORTED_FILE_MIMES = _SUPPORTED_IMAGE_MIMES | _SUPPORTED_PDF_MIMES


# Short-vs-long text threshold. Six characters is enough to catch
# pathological 1-token queries ("hi", "ok?", "test") and push them to
# clarify, while still letting real retrieval queries like "faiss?" or
# "bge-m3" through. Counted on the stripped text.
_MIN_ROUTABLE_TEXT_CHARS = 6


# Max characters of text preview sent to the LLM router. Full prompts
# would blow the tool-call budget on long OCR dumps; a 400-char preview
# is plenty for routing intent extraction.
_LLM_TEXT_PREVIEW_CHARS = 400

# Max length of the human-readable reason string on an AgentDecision.
_REASON_MAX_CHARS = 160


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentDecision:
    """Structured routing decision emitted by an AgentRouterProvider.

    Frozen so the AGENT_DECISION artifact payload is a value the
    capability can serialize once without worrying about later mutation.

    Fields:

      action: which downstream capability to dispatch to, or
        ``'direct_answer'`` (answered inline by a cheap LLM call) or
        ``'clarify'`` (ask the user for more input).
      reason: short human-readable justification, capped at 160 chars
        so it folds cleanly into log lines and error messages.
      parsed_query: optional pre-parsed structure of the retrieval
        query. Attached by ``LlmAgentRouter`` when ``action == 'rag'``
        and ``text`` is non-empty; None otherwise.
      confidence: the router's self-reported confidence in [0.0, 1.0].
        Used by ``LlmAgentRouter`` to gate fallback to the rule router.
      router_name: stable identifier of the provider that produced
        this decision. Includes a ``-fallback-rule`` suffix when an
        LLM router degraded to the rule path so observability can
        diff clean-LLM runs from fallback runs.
    """

    action: AgentAction
    reason: str
    parsed_query: Optional[ParsedQuery]
    confidence: float
    router_name: str

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"AgentDecision.action must be one of {sorted(_VALID_ACTIONS)}; "
                f"got {self.action!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"AgentDecision.confidence must be in [0.0, 1.0]; "
                f"got {self.confidence!r}"
            )

    def to_dict(self) -> dict:
        """JSON-serializable dict used by AGENT_DECISION artifact payloads."""
        body: dict[str, Any] = {
            "action": self.action,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "routerName": self.router_name,
        }
        if self.parsed_query is not None:
            body["parsedQuery"] = self.parsed_query.to_dict()
        else:
            body["parsedQuery"] = None
        return body


# --------------------------------------------------------------------------
# Provider contract
# --------------------------------------------------------------------------


class AgentRouterProvider(ABC):
    """Converts an ``(text, file)`` pair into an ``AgentDecision``."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def decide(
        self,
        *,
        text: Optional[str],
        has_file: bool,
        file_mime: Optional[str],
        file_size: int,
    ) -> AgentDecision:
        ...


# --------------------------------------------------------------------------
# Rule-based router — deterministic, dependency-free
# --------------------------------------------------------------------------


class RuleBasedAgentRouter(AgentRouterProvider):
    """Deterministic five-branch router.

    The matrix over (routable-text, routable-file) is::

        text>6ch + file in {png,jpeg,pdf}  -> multimodal, conf=0.95
        file in {png,jpeg,pdf} only         -> ocr,        conf=0.90
        text>6ch only                       -> rag,        conf=0.70
        text<=6ch only                      -> clarify,    conf=0.50
        neither                             -> clarify,    conf=0.00

    "File in {png,jpeg,pdf}" is by mime-type only — the ``has_file``
    flag alone is not enough because core-api's MOCK path accepts any
    mime on multipart, and an unsupported type (gif, docx, ...) cannot
    be routed to OCR / MULTIMODAL without an immediate capability-side
    ``UNSUPPORTED_INPUT_TYPE``. Unsupported mimes collapse to the
    "no-file" row, so ``text>6ch`` still routes to RAG and an empty
    text stays in ``clarify``.
    """

    name = "rule"

    def decide(
        self,
        *,
        text: Optional[str],
        has_file: bool,
        file_mime: Optional[str],
        file_size: int,
    ) -> AgentDecision:
        text_stripped = (text or "").strip()
        has_text = len(text_stripped) > _MIN_ROUTABLE_TEXT_CHARS
        has_short_text = bool(text_stripped) and not has_text
        routable_file = has_file and _mime_is_supported(file_mime) and file_size > 0

        if has_text and routable_file:
            return AgentDecision(
                action="multimodal",
                reason=(
                    f"text ({len(text_stripped)} chars) + supported file "
                    f"({file_mime}, {file_size}B) -> multimodal"
                )[:_REASON_MAX_CHARS],
                parsed_query=None,
                confidence=0.95,
                router_name=self.name,
            )
        if routable_file:
            return AgentDecision(
                action="ocr",
                reason=(
                    f"supported file only ({file_mime}, {file_size}B) -> ocr"
                )[:_REASON_MAX_CHARS],
                parsed_query=None,
                confidence=0.90,
                router_name=self.name,
            )
        if has_text:
            return AgentDecision(
                action="rag",
                reason=(
                    f"text only ({len(text_stripped)} chars) -> rag"
                )[:_REASON_MAX_CHARS],
                parsed_query=None,
                confidence=0.70,
                router_name=self.name,
            )
        if has_short_text:
            return AgentDecision(
                action="clarify",
                reason=(
                    f"text is too short ({len(text_stripped)} <= "
                    f"{_MIN_ROUTABLE_TEXT_CHARS}) to route"
                )[:_REASON_MAX_CHARS],
                parsed_query=None,
                confidence=0.50,
                router_name=self.name,
            )
        return AgentDecision(
            action="clarify",
            reason="no usable text, no routable file -> clarify",
            parsed_query=None,
            confidence=0.0,
            router_name=self.name,
        )


def _mime_is_supported(file_mime: Optional[str]) -> bool:
    if not file_mime:
        return False
    normalized = file_mime.split(";")[0].strip().lower()
    return normalized in _SUPPORTED_FILE_MIMES


# --------------------------------------------------------------------------
# LLM router — wraps LlmChatProvider with rule-based fallback
# --------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = (
    "You route user jobs to one of five actions.\n"
    "  rag            - text-only retrieval question over the knowledge base.\n"
    "  ocr            - extract text from a supplied image / PDF.\n"
    "  multimodal     - text question grounded in the supplied image / PDF.\n"
    "  direct_answer  - trivially answerable without retrieval or the file.\n"
    "  clarify        - ambiguous / missing input, ask the user for more.\n"
    "Pick the single best action. Return your confidence in [0.0, 1.0].\n"
    "Keep the reason under 160 characters."
)

_LLM_TOOL_NAME = "route_job"

_LLM_TOOL_DESCRIPTION = (
    "Emit the routing decision for the job. Always call this tool."
)

_LLM_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["rag", "ocr", "multimodal", "direct_answer", "clarify"],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["action", "reason", "confidence"],
    "additionalProperties": False,
}

_LLM_SCHEMA_HINT = (
    '{"action": "rag"|"ocr"|"multimodal"|"direct_answer"|"clarify", '
    '"reason": string (<=160 chars), "confidence": number in [0,1]}'
)


class LlmAgentRouter(AgentRouterProvider):
    """LLM-backed router that degrades to a rule-based fallback.

    Uses the shared ``LlmChatProvider`` to pick a routing action. When
    the backend advertises ``function_calling``, the router goes through
    ``chat_tools`` with a ``route_job`` tool spec; otherwise it falls
    back to ``chat_json`` with a schema hint. Thinking mode is enabled
    iff the backend advertises ``thinking``.

    Fallback to the rule router fires on ANY of:

      * ``LlmChatError`` from the underlying provider (network, timeout,
        invalid JSON, empty response).
      * Schema violation at our layer (missing field, wrong type,
        action not in the 5-enum).
      * ``confidence < confidence_threshold`` — the model answered but
        isn't sure enough to act on.

    The fallback decision's ``router_name`` becomes
    ``f"{self.name}-fallback-rule"`` so operators can distinguish clean
    LLM runs from degradation in the trace layer.
    """

    def __init__(
        self,
        chat: Any,
        parser: QueryParserProvider,
        *,
        confidence_threshold: float = 0.55,
        fallback: Optional[AgentRouterProvider] = None,
    ) -> None:
        from app.clients.llm_chat import LlmChatProvider  # local import — avoids cycles

        if not isinstance(chat, LlmChatProvider):
            raise TypeError(
                "LlmAgentRouter requires an LlmChatProvider instance; "
                f"got {type(chat).__name__}"
            )
        if not isinstance(parser, QueryParserProvider):
            raise TypeError(
                "LlmAgentRouter requires a QueryParserProvider instance; "
                f"got {type(parser).__name__}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0]; "
                f"got {confidence_threshold!r}"
            )
        self._chat = chat
        self._parser = parser
        self._confidence_threshold = float(confidence_threshold)
        self._fallback: AgentRouterProvider = fallback or RuleBasedAgentRouter()

    @property
    def name(self) -> str:
        return f"llm-{self._chat.name}"

    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        text: Optional[str],
        has_file: bool,
        file_mime: Optional[str],
        file_size: int,
    ) -> AgentDecision:
        from app.clients.llm_chat import ChatMessage, ChatToolSpec, LlmChatError

        preview = (text or "")[:_LLM_TEXT_PREVIEW_CHARS]
        user_content = json.dumps(
            {
                "text_preview": preview,
                "text_length": len(text or ""),
                "has_file": bool(has_file),
                "file_mime": file_mime,
                "file_size": int(file_size),
            },
            ensure_ascii=False,
        )
        messages = [
            ChatMessage(role="system", content=_LLM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]
        enable_thinking = bool(self._chat.capabilities.get("thinking"))
        uses_function_calling = bool(
            self._chat.capabilities.get("function_calling")
        )

        try:
            if uses_function_calling:
                tool = ChatToolSpec(
                    name=_LLM_TOOL_NAME,
                    description=_LLM_TOOL_DESCRIPTION,
                    parameters=_LLM_ACTION_SCHEMA,
                )
                result = self._chat.chat_tools(
                    messages,
                    [tool],
                    enable_thinking=enable_thinking,
                )
                data = _extract_tool_arguments(result)
            else:
                data = self._chat.chat_json(
                    messages,
                    schema_hint=_LLM_SCHEMA_HINT,
                    enable_thinking=enable_thinking,
                )
        except LlmChatError as ex:
            log.warning(
                "LlmAgentRouter: provider failure, falling back to rule (%s)",
                ex,
            )
            return self._run_fallback(
                text=text,
                has_file=has_file,
                file_mime=file_mime,
                file_size=file_size,
                downgrade_reason=f"llm provider error: {ex}",
            )

        try:
            action, reason, confidence = _materialize_decision(data)
        except ValueError as ex:
            log.warning(
                "LlmAgentRouter: invalid response, falling back to rule (%s)",
                ex,
            )
            return self._run_fallback(
                text=text,
                has_file=has_file,
                file_mime=file_mime,
                file_size=file_size,
                downgrade_reason=f"llm invalid response: {ex}",
            )

        if confidence < self._confidence_threshold:
            log.info(
                "LlmAgentRouter: confidence %.2f below threshold %.2f, "
                "falling back to rule",
                confidence,
                self._confidence_threshold,
            )
            return self._run_fallback(
                text=text,
                has_file=has_file,
                file_mime=file_mime,
                file_size=file_size,
                downgrade_reason=(
                    f"llm confidence {confidence:.2f} "
                    f"< threshold {self._confidence_threshold:.2f}"
                ),
            )

        parsed_query: Optional[ParsedQuery] = None
        if action == "rag" and (text or "").strip():
            try:
                parsed_query = self._parser.parse(text or "")
            except Exception as ex:  # parser errors must never block routing
                log.warning(
                    "LlmAgentRouter: parser.parse failed (%s); proceeding "
                    "with parsed_query=None",
                    ex,
                )
                parsed_query = None

        return AgentDecision(
            action=action,
            reason=reason[:_REASON_MAX_CHARS],
            parsed_query=parsed_query,
            confidence=confidence,
            router_name=self.name,
        )

    # ------------------------------------------------------------------

    def _run_fallback(
        self,
        *,
        text: Optional[str],
        has_file: bool,
        file_mime: Optional[str],
        file_size: int,
        downgrade_reason: str,
    ) -> AgentDecision:
        base = self._fallback.decide(
            text=text,
            has_file=has_file,
            file_mime=file_mime,
            file_size=file_size,
        )
        suffix = f" (after {downgrade_reason})"
        combined_reason = (base.reason + suffix)[:_REASON_MAX_CHARS]
        return AgentDecision(
            action=base.action,
            reason=combined_reason,
            parsed_query=base.parsed_query,
            confidence=base.confidence,
            router_name=f"{self.name}-fallback-rule",
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_tool_arguments(result: Any) -> dict:
    """Pull the routing arguments out of a ChatResult from chat_tools.

    Accepts two shapes:
      * ``result.tool_call`` is a dict with an ``arguments`` key
        containing the model's emitted JSON object.
      * ``result.text`` is a JSON string — fallback path for backends
        that didn't fire the native tool API.
    """
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


def _materialize_decision(data: dict) -> tuple[AgentAction, str, float]:
    """Validate the router-call payload and coerce to (action, reason, conf).

    Raises ``ValueError`` on any schema violation — the caller translates
    that into a fallback with a ``-fallback-rule`` router_name.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"LLM response must be a JSON object; got {type(data).__name__}"
        )

    raw_action = data.get("action")
    if not isinstance(raw_action, str):
        raise ValueError("'action' must be a string")
    action = raw_action.strip().lower()
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"'action' must be one of {sorted(_VALID_ACTIONS)}; got {raw_action!r}"
        )

    raw_reason = data.get("reason", "")
    if not isinstance(raw_reason, str):
        raise ValueError("'reason' must be a string")
    reason = raw_reason.strip() or "no reason given"

    raw_conf = data.get("confidence")
    if isinstance(raw_conf, bool):
        # bool is a subclass of int in Python; reject it explicitly so a
        # stray "true" doesn't silently become 1.0.
        raise ValueError("'confidence' must be a number, not bool")
    if not isinstance(raw_conf, (int, float)):
        raise ValueError("'confidence' must be a number in [0.0, 1.0]")
    confidence = float(raw_conf)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"'confidence' must be in [0.0, 1.0]; got {confidence!r}"
        )

    return action, reason, confidence  # type: ignore[return-value]
