"""AgentRouterProvider tests.

Three scenario groups, all fully offline:

  1. ``RuleBasedAgentRouter`` — deterministic decision tree. All five
     branches are exercised individually, plus the "unsupported mime
     collapses to no-file" edge case.

  2. ``LlmAgentRouter`` happy paths — native tool-calling, JSON mode,
     ``enable_thinking`` gating.

  3. ``LlmAgentRouter`` fallback paths — provider failure, invalid
     action, confidence-below-threshold. The router_name suffix stamps
     the downgrade so metrics / trace can diff clean LLM runs from
     degraded ones.

Zero new deps; everything runs under the stdlib plus the module under
test.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
    LlmAgentRouter,
    RuleBasedAgentRouter,
)
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)
from app.clients.llm_chat import (
    ChatMessage,
    ChatResult,
    ChatToolSpec,
    LlmChatError,
    LlmChatProvider,
)


# ---------------------------------------------------------------------------
# 1. RuleBasedAgentRouter — five-branch decision tree.
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_router() -> RuleBasedAgentRouter:
    return RuleBasedAgentRouter()


def test_rule_router_text_and_pdf_routes_to_multimodal(rule_router):
    decision = rule_router.decide(
        text="What is the total on this invoice?",
        has_file=True,
        file_mime="application/pdf",
        file_size=2048,
    )
    assert decision.action == "multimodal"
    assert decision.confidence == pytest.approx(0.95)
    assert decision.router_name == "rule"
    assert decision.parsed_query is None
    assert "multimodal" in decision.reason


def test_rule_router_text_and_image_routes_to_multimodal(rule_router):
    decision = rule_router.decide(
        text="extract the handwritten note",
        has_file=True,
        file_mime="image/jpeg",
        file_size=4096,
    )
    assert decision.action == "multimodal"
    assert decision.confidence == pytest.approx(0.95)


def test_rule_router_file_only_routes_to_ocr(rule_router):
    decision = rule_router.decide(
        text="",
        has_file=True,
        file_mime="image/png",
        file_size=1024,
    )
    assert decision.action == "ocr"
    assert decision.confidence == pytest.approx(0.90)
    assert decision.router_name == "rule"


def test_rule_router_long_text_only_routes_to_rag(rule_router):
    decision = rule_router.decide(
        text="which anime is about harbor cats",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    assert decision.action == "rag"
    assert decision.confidence == pytest.approx(0.70)


def test_rule_router_short_text_only_routes_to_clarify(rule_router):
    decision = rule_router.decide(
        text="hello",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    assert decision.action == "clarify"
    assert decision.confidence == pytest.approx(0.50)
    assert "too short" in decision.reason


def test_rule_router_neither_routes_to_clarify_with_zero_confidence(rule_router):
    decision = rule_router.decide(
        text="",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    assert decision.action == "clarify"
    assert decision.confidence == pytest.approx(0.0)


def test_rule_router_unsupported_mime_collapses_to_no_file(rule_router):
    # GIF is not in the supported set — rule router treats has_file as
    # noise and routes by text alone.
    decision = rule_router.decide(
        text="describe this animation please",
        has_file=True,
        file_mime="image/gif",
        file_size=4096,
    )
    assert decision.action == "rag"

    decision_no_text = rule_router.decide(
        text=None,
        has_file=True,
        file_mime="image/gif",
        file_size=4096,
    )
    assert decision_no_text.action == "clarify"


def test_rule_router_empty_file_is_treated_as_no_file(rule_router):
    decision = rule_router.decide(
        text="long enough text to route on",
        has_file=True,
        file_mime="image/png",
        file_size=0,
    )
    assert decision.action == "rag"


def test_rule_router_normalizes_mime_with_charset_parameters(rule_router):
    decision = rule_router.decide(
        text="long enough text content",
        has_file=True,
        file_mime="IMAGE/PNG; charset=binary",
        file_size=128,
    )
    assert decision.action == "multimodal"


# ---------------------------------------------------------------------------
# 2. LlmAgentRouter — shared fake + happy-path tests.
# ---------------------------------------------------------------------------


class _FakeChat(LlmChatProvider):
    """Tiny LlmChatProvider stub used for LlmAgentRouter unit tests."""

    def __init__(
        self,
        *,
        tool_call: Optional[dict] = None,
        tool_text: Optional[str] = None,
        json_return: Optional[dict] = None,
        error: Optional[Exception] = None,
        capabilities: Optional[dict] = None,
        name: str = "fake-chat",
    ) -> None:
        self._tool_call = tool_call
        self._tool_text = tool_text
        self._json_return = json_return
        self._error = error
        self._capabilities = capabilities or {
            "function_calling": True,
            "thinking": True,
            "json_mode": True,
            "vision": False,
            "audio": False,
        }
        self._name = name
        self.chat_json_calls: list[tuple[tuple, dict]] = []
        self.chat_tools_calls: list[tuple[tuple, dict]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> dict:
        return dict(self._capabilities)

    def chat_json(
        self,
        messages,
        *,
        schema_hint,
        max_tokens=512,
        temperature=0.0,
        timeout_s=15.0,
        enable_thinking=False,
    ) -> dict:
        self.chat_json_calls.append(
            (
                (list(messages),),
                {
                    "schema_hint": schema_hint,
                    "enable_thinking": enable_thinking,
                },
            )
        )
        if self._error is not None:
            raise self._error
        assert self._json_return is not None, "json_return must be set"
        return self._json_return

    def chat_tools(
        self,
        messages,
        tools,
        *,
        max_tokens=512,
        temperature=0.0,
        timeout_s=15.0,
        enable_thinking=False,
    ) -> ChatResult:
        self.chat_tools_calls.append(
            (
                (list(messages), list(tools)),
                {"enable_thinking": enable_thinking},
            )
        )
        if self._error is not None:
            raise self._error
        return ChatResult(
            text=self._tool_text,
            tool_call=self._tool_call,
            raw={},
            tokens_in=0,
            tokens_out=0,
            latency_ms=0.0,
        )


def test_llm_router_happy_path_via_function_calling():
    chat = _FakeChat(
        tool_call={
            "name": "route_job",
            "arguments": {
                "action": "multimodal",
                "reason": "has text + pdf",
                "confidence": 0.91,
            },
        },
    )
    router = LlmAgentRouter(chat, NoOpQueryParser())

    decision = router.decide(
        text="invoice total?",
        has_file=True,
        file_mime="application/pdf",
        file_size=2048,
    )
    assert decision.action == "multimodal"
    assert decision.confidence == pytest.approx(0.91)
    assert decision.router_name == "llm-fake-chat"
    # Tool calling path was exercised; JSON path was not.
    assert len(chat.chat_tools_calls) == 1
    assert len(chat.chat_json_calls) == 0
    # Thinking was enabled because the fake advertises it.
    assert chat.chat_tools_calls[0][1]["enable_thinking"] is True


def test_llm_router_happy_path_via_json_mode():
    chat = _FakeChat(
        json_return={
            "action": "rag",
            "reason": "text only question",
            "confidence": 0.80,
        },
        capabilities={
            "function_calling": False,
            "thinking": False,
            "json_mode": True,
            "vision": False,
            "audio": False,
        },
    )
    router = LlmAgentRouter(chat, NoOpQueryParser())

    decision = router.decide(
        text="what is the retriever",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    assert decision.action == "rag"
    assert decision.confidence == pytest.approx(0.80)
    assert decision.router_name == "llm-fake-chat"
    assert len(chat.chat_json_calls) == 1
    assert chat.chat_json_calls[0][1]["enable_thinking"] is False
    assert len(chat.chat_tools_calls) == 0


def test_llm_router_attaches_parsed_query_on_rag_action():
    chat = _FakeChat(
        tool_call={
            "name": "route_job",
            "arguments": {
                "action": "rag",
                "reason": "text retrieval question",
                "confidence": 0.82,
            },
        },
    )
    parser = RegexQueryParser()
    router = LlmAgentRouter(chat, parser)

    decision = router.decide(
        text="how does FAISS index search work",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    assert decision.action == "rag"
    assert isinstance(decision.parsed_query, ParsedQuery)
    assert "faiss" in decision.parsed_query.keywords
    assert "search" in decision.parsed_query.keywords


def test_llm_router_skips_parser_when_action_is_not_rag():
    chat = _FakeChat(
        tool_call={
            "name": "route_job",
            "arguments": {
                "action": "ocr",
                "reason": "file-only",
                "confidence": 0.92,
            },
        },
    )
    # Parser that would raise if called — it shouldn't be.
    class _ExplodingParser(QueryParserProvider):
        @property
        def name(self) -> str:  # pragma: no cover - trivial
            return "explode"

        def parse(self, query: str) -> ParsedQuery:  # pragma: no cover
            raise AssertionError("parser must not be called for non-rag actions")

    router = LlmAgentRouter(chat, _ExplodingParser())
    decision = router.decide(
        text="",
        has_file=True,
        file_mime="image/png",
        file_size=1024,
    )
    assert decision.action == "ocr"
    assert decision.parsed_query is None


# ---------------------------------------------------------------------------
# 3. LlmAgentRouter fallback paths — degrades visibly.
# ---------------------------------------------------------------------------


def test_llm_router_falls_back_to_rule_on_provider_error(caplog):
    chat = _FakeChat(error=LlmChatError("network timeout"))
    router = LlmAgentRouter(chat, NoOpQueryParser())

    with caplog.at_level("WARNING"):
        decision = router.decide(
            text="describe this diagram in detail",
            has_file=True,
            file_mime="image/png",
            file_size=1024,
        )
    # Rule path for (long text + png) is multimodal.
    assert decision.action == "multimodal"
    assert decision.confidence == pytest.approx(0.95)
    assert decision.router_name == "llm-fake-chat-fallback-rule"
    assert any("falling back to rule" in rec.message for rec in caplog.records)


def test_llm_router_falls_back_on_low_confidence(caplog):
    chat = _FakeChat(
        tool_call={
            "name": "route_job",
            "arguments": {
                "action": "rag",
                "reason": "not sure",
                "confidence": 0.30,
            },
        },
    )
    router = LlmAgentRouter(chat, NoOpQueryParser(), confidence_threshold=0.55)

    with caplog.at_level("INFO"):
        decision = router.decide(
            text="what is this even",
            has_file=False,
            file_mime=None,
            file_size=0,
        )
    assert decision.router_name == "llm-fake-chat-fallback-rule"
    # Rule path for text-only > 6 chars is rag.
    assert decision.action == "rag"
    assert decision.confidence == pytest.approx(0.70)
    assert any("below threshold" in rec.message for rec in caplog.records)


def test_llm_router_falls_back_on_invalid_action(caplog):
    chat = _FakeChat(
        tool_call={
            "name": "route_job",
            "arguments": {
                "action": "summarize",  # not in the 5-enum
                "reason": "bad action",
                "confidence": 0.9,
            },
        },
    )
    router = LlmAgentRouter(chat, NoOpQueryParser())

    with caplog.at_level("WARNING"):
        decision = router.decide(
            text="short",
            has_file=False,
            file_mime=None,
            file_size=0,
        )
    assert decision.router_name == "llm-fake-chat-fallback-rule"
    # Rule path for short-text-only is clarify with 0.5.
    assert decision.action == "clarify"
    assert any("invalid response" in rec.message for rec in caplog.records)


def test_llm_router_falls_back_on_missing_tool_call_and_no_text():
    chat = _FakeChat(tool_call=None, tool_text=None)
    router = LlmAgentRouter(chat, NoOpQueryParser())

    decision = router.decide(
        text="this has enough chars to route",
        has_file=False,
        file_mime=None,
        file_size=0,
    )
    # Rule path for long-text-only is rag.
    assert decision.action == "rag"
    assert decision.router_name == "llm-fake-chat-fallback-rule"


def test_llm_router_accepts_tool_text_when_backend_skipped_tool_call():
    # Some backends return JSON in .text when they don't fire the
    # native tool API — the router should accept that path too.
    chat = _FakeChat(
        tool_call=None,
        tool_text='{"action": "ocr", "reason": "file only", "confidence": 0.88}',
    )
    router = LlmAgentRouter(chat, NoOpQueryParser())

    decision = router.decide(
        text="",
        has_file=True,
        file_mime="image/png",
        file_size=2048,
    )
    assert decision.action == "ocr"
    assert decision.router_name == "llm-fake-chat"


def test_llm_router_rejects_non_provider_chat():
    with pytest.raises(TypeError):
        LlmAgentRouter(object(), NoOpQueryParser())  # type: ignore[arg-type]


def test_llm_router_rejects_non_parser():
    chat = _FakeChat(json_return={"action": "rag", "reason": "ok", "confidence": 0.9})
    with pytest.raises(TypeError):
        LlmAgentRouter(chat, object())  # type: ignore[arg-type]


def test_llm_router_rejects_out_of_range_threshold():
    chat = _FakeChat(json_return={"action": "rag", "reason": "ok", "confidence": 0.9})
    with pytest.raises(ValueError):
        LlmAgentRouter(chat, NoOpQueryParser(), confidence_threshold=1.5)


# ---------------------------------------------------------------------------
# AgentDecision value-object contracts.
# ---------------------------------------------------------------------------


def test_agent_decision_is_frozen():
    decision = AgentDecision(
        action="rag",
        reason="ok",
        parsed_query=None,
        confidence=0.7,
        router_name="rule",
    )
    with pytest.raises(Exception):
        decision.action = "ocr"  # type: ignore[misc]


def test_agent_decision_rejects_unknown_action():
    with pytest.raises(ValueError):
        AgentDecision(
            action="summarize",  # type: ignore[arg-type]
            reason="",
            parsed_query=None,
            confidence=0.5,
            router_name="rule",
        )


def test_agent_decision_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        AgentDecision(
            action="rag",
            reason="",
            parsed_query=None,
            confidence=1.5,
            router_name="rule",
        )


def test_agent_decision_to_dict_is_json_shape():
    pq = ParsedQuery(
        original="q",
        normalized="q",
        keywords=["x"],
        intent="other",
        rewrites=[],
        filters={},
        parser_name="regex",
    )
    decision = AgentDecision(
        action="rag",
        reason="long text only",
        parsed_query=pq,
        confidence=0.705,
        router_name="rule",
    )
    body = decision.to_dict()
    assert body["action"] == "rag"
    assert body["routerName"] == "rule"
    assert body["confidence"] == pytest.approx(0.705)
    assert body["parsedQuery"]["parserName"] == "regex"


def test_agent_router_provider_subclass_contract():
    assert issubclass(RuleBasedAgentRouter, AgentRouterProvider)
    assert issubclass(LlmAgentRouter, AgentRouterProvider)
