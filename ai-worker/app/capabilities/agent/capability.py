"""AUTO capability — single-pass dispatcher over RAG / OCR / MULTIMODAL.

Given an unknown ``(text, file)`` pair, this capability:

  1. Classifies the job inputs into ``(text, file_bytes, file_mime,
     filename)``.
  2. Asks an ``AgentRouterProvider`` for an ``AgentDecision``.
  3. Delegates to the matching sub-capability via the standard
     ``Capability.run()`` interface, or produces an inline
     clarify / direct-answer response when the router chooses one
     of those terminal actions.
  4. Emits an ``AGENT_DECISION`` JSON artifact with the routing
     metadata, followed by whatever artifacts the delegated
     sub-capability returned.

Phase 3 is deliberately single-pass — one router call, one sub call,
one terminal callback. Phase 6 upgrades the same AutoCapability seam
into a real agent loop (critic, retry, rewrite) without re-wiring
callers.

The capability delegates through the ``Capability`` interface only; it
never reaches into a sub-capability's internals. Missing sub-capabilities
degrade gracefully — when the router picks an action whose sub is not
registered, the job fails with a typed ``AUTO_<sub>_UNAVAILABLE`` error
rather than a cryptic AttributeError.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
)
from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.query_parser import QueryParserProvider

log = logging.getLogger(__name__)


# Default Korean clarify response for ambiguous inputs. The user
# population for this project is Korean-first; the message mirrors the
# tone of other worker-side user-facing strings.
_CLARIFY_MESSAGE = (
    "질문이 모호합니다. 어떤 문서나 주제에 대해 알고 싶으신가요?"
)

# Stable error codes exposed by AutoCapability. Matched against in
# tests + observability layers, so don't rename without bumping them
# in the architecture doc too.
_ERR_AUTO_NO_INPUT = "AUTO_NO_INPUT"
_ERR_AUTO_RAG_UNAVAILABLE = "AUTO_RAG_UNAVAILABLE"
_ERR_AUTO_OCR_UNAVAILABLE = "AUTO_OCR_UNAVAILABLE"
_ERR_AUTO_MULTIMODAL_UNAVAILABLE = "AUTO_MULTIMODAL_UNAVAILABLE"

# Max length of the text preview shown in route_classify log lines.
_LOG_PREVIEW_CHARS = 120


# Short system prompt used by ``_direct_answer`` when the router
# selects ``direct_answer`` and a chat backend is configured.
_DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You answer brief questions concisely. Keep the answer under 120 words. "
    "If the question cannot be answered without retrieval or external data, "
    "explain that in one sentence."
)


class AutoCapability(Capability):
    """AUTO capability.

    The router is injected rather than baked in so the registry can
    swap between rule-based, LLM-backed, and (later) agent-loop routers
    without touching this class. Sub-capabilities are optional — a
    registry with OCR/MULTIMODAL missing but RAG present still serves
    AUTO for text-only jobs, and the router's missing-sub branches
    degrade to ``clarify``.
    """

    name = "AUTO"

    def __init__(
        self,
        *,
        router: AgentRouterProvider,
        parser: QueryParserProvider,
        rag: Optional[Capability] = None,
        ocr: Optional[Capability] = None,
        multimodal: Optional[Capability] = None,
        chat: Optional[Any] = None,
        direct_answer_max_tokens: int = 512,
    ) -> None:
        self._router = router
        self._parser = parser
        self._rag = rag
        self._ocr = ocr
        self._multimodal = multimodal
        self._chat = chat
        self._direct_answer_max_tokens = int(direct_answer_max_tokens)

    # ------------------------------------------------------------------

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        started = time.monotonic()
        text, file_bytes, file_mime, filename = _classify_input(input.inputs)

        if not text and not file_bytes:
            raise CapabilityError(
                _ERR_AUTO_NO_INPUT,
                "AUTO job has no usable text and no file — submit either a "
                "non-blank text field, a supported file (PNG/JPEG/PDF), or "
                "both.",
            )

        classify_ms = _elapsed_ms(started)
        log.info(
            "AUTO route_classify jobId=%s textLen=%d textPreview=%r "
            "hasFile=%s fileMime=%s fileSize=%d filename=%s classifyMs=%.3f",
            input.job_id,
            len(text or ""),
            (text or "")[:_LOG_PREVIEW_CHARS],
            bool(file_bytes),
            file_mime,
            len(file_bytes or b""),
            filename,
            classify_ms,
        )

        started = time.monotonic()
        decision = self._router.decide(
            text=text,
            has_file=bool(file_bytes),
            file_mime=file_mime,
            file_size=len(file_bytes or b""),
        )
        decide_ms = _elapsed_ms(started)
        log.info(
            "AUTO route_decide jobId=%s action=%s router=%s confidence=%.3f "
            "reason=%r decideMs=%.3f",
            input.job_id,
            decision.action,
            decision.router_name,
            decision.confidence,
            decision.reason,
            decide_ms,
        )

        sub_outputs: list[CapabilityOutputArtifact]
        if decision.action == "multimodal":
            self._require(
                self._multimodal,
                _ERR_AUTO_MULTIMODAL_UNAVAILABLE,
                "AUTO routed to MULTIMODAL but that capability is not "
                "registered on this worker. Enable it via "
                "AIPIPELINE_WORKER_MULTIMODAL_ENABLED=true and ensure its "
                "OCR + RAG dependencies are healthy, then restart the worker.",
            )
            sub_outputs = list(self._multimodal.run(input).outputs)
        elif decision.action == "ocr":
            self._require(
                self._ocr,
                _ERR_AUTO_OCR_UNAVAILABLE,
                "AUTO routed to OCR but that capability is not registered "
                "on this worker. Enable it via "
                "AIPIPELINE_WORKER_OCR_ENABLED=true and ensure Tesseract is "
                "installed, then restart the worker.",
            )
            sub_outputs = list(self._ocr.run(input).outputs)
        elif decision.action == "rag":
            self._require(
                self._rag,
                _ERR_AUTO_RAG_UNAVAILABLE,
                "AUTO routed to RAG but that capability is not registered "
                "on this worker. Enable it via "
                "AIPIPELINE_WORKER_RAG_ENABLED=true and ensure the FAISS "
                "index is built, then restart the worker.",
            )
            sub_outputs = list(self._rag.run(input).outputs)
        elif decision.action == "direct_answer":
            answer = self._direct_answer(text)
            sub_outputs = [
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename="auto-answer.md",
                    content_type="text/markdown; charset=utf-8",
                    content=answer.encode("utf-8"),
                )
            ]
        else:
            # clarify — inline short response; no sub-capability is invoked.
            sub_outputs = [
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename="auto-clarify.md",
                    content_type="text/markdown; charset=utf-8",
                    content=_CLARIFY_MESSAGE.encode("utf-8"),
                )
            ]

        agent_decision_artifact = CapabilityOutputArtifact(
            type="AGENT_DECISION",
            filename="agent-decision.json",
            content_type="application/json",
            content=_serialize_decision(decision).encode("utf-8"),
        )

        outputs = [agent_decision_artifact, *sub_outputs]
        log.info(
            "AUTO done jobId=%s action=%s router=%s outputs=%d",
            input.job_id, decision.action, decision.router_name, len(outputs),
        )
        return CapabilityOutput(outputs=outputs)

    # ------------------------------------------------------------------

    @staticmethod
    def _require(
        sub: Optional[Capability],
        code: str,
        message: str,
    ) -> None:
        """Raise a typed error when a routed-to sub-capability is missing.

        Used at dispatch time so the failure is a clean, named error
        rather than an ``AttributeError`` on ``None.run()``."""
        if sub is None:
            raise CapabilityError(code, message)

    # ------------------------------------------------------------------

    def _direct_answer(self, text: Optional[str]) -> str:
        """Produce an inline direct answer via the shared chat backend.

        Only the LLM router ever selects ``direct_answer``. If the chat
        provider is unavailable or raises, we degrade to the clarify
        message so the client always gets a FINAL_RESPONSE.
        """
        from app.clients.llm_chat import ChatMessage, LlmChatError

        if self._chat is None or not (text or "").strip():
            return _CLARIFY_MESSAGE

        try:
            messages = [
                ChatMessage(role="system", content=_DIRECT_ANSWER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=text or ""),
            ]
            enable_thinking = bool(self._chat.capabilities.get("thinking"))
            data = self._chat.chat_json(
                messages,
                schema_hint='{"answer": string}',
                max_tokens=self._direct_answer_max_tokens,
                enable_thinking=enable_thinking,
            )
        except LlmChatError as ex:
            log.warning(
                "AUTO direct_answer: chat backend failed (%s) — "
                "falling back to clarify message",
                ex,
            )
            return _CLARIFY_MESSAGE

        raw_answer = data.get("answer") if isinstance(data, dict) else None
        if isinstance(raw_answer, str) and raw_answer.strip():
            return raw_answer.strip()
        log.warning(
            "AUTO direct_answer: chat returned no 'answer' field — "
            "falling back to clarify message"
        )
        return _CLARIFY_MESSAGE


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClassifiedInput:
    """Internal tuple-like helper — _classify_input returns the four
    fields directly rather than exposing this type."""

    text: Optional[str]
    file_bytes: Optional[bytes]
    file_mime: Optional[str]
    filename: Optional[str]


def _classify_input(
    artifacts: list[CapabilityInputArtifact],
) -> tuple[Optional[str], Optional[bytes], Optional[str], Optional[str]]:
    """Split a list of CapabilityInputArtifacts into (text, file_bytes,
    file_mime, filename).

    The first INPUT_TEXT wins; likewise the first INPUT_FILE. Any other
    artifact types are ignored. UTF-8 decoding failures on INPUT_TEXT
    are silently dropped (the next INPUT_TEXT artifact, or None).
    """
    text: Optional[str] = None
    file_bytes: Optional[bytes] = None
    file_mime: Optional[str] = None
    filename: Optional[str] = None

    for artifact in artifacts:
        if artifact.type == "INPUT_TEXT" and text is None:
            try:
                text = artifact.content.decode("utf-8").strip() or None
            except UnicodeDecodeError:
                text = None
            continue
        if artifact.type == "INPUT_FILE" and file_bytes is None:
            file_bytes = artifact.content or None
            file_mime = _normalize_mime(artifact.content_type)
            filename = artifact.filename

    return text, file_bytes, file_mime, filename


def _normalize_mime(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    head = raw.split(";")[0].strip().lower()
    return head or None


def _serialize_decision(decision: AgentDecision) -> str:
    """Render the AgentDecision as pretty-printed JSON for the artifact
    payload. Stable key order is a nice-to-have for tests that parse
    the body back out."""
    return json.dumps(decision.to_dict(), ensure_ascii=False, indent=2)


def _elapsed_ms(started_monotonic: float) -> float:
    return round((time.monotonic() - started_monotonic) * 1000.0, 3)
