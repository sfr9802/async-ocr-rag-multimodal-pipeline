"""AutoCapability tests.

All tests are hermetic — they build an AutoCapability with a fixed
``AgentRouterProvider`` (or a fake chat-backed one) and fake RAG / OCR /
MULTIMODAL sub-capabilities that just echo a sentinel artifact so the
dispatch path is observable without running real retrievers or OCR
engines.

Covered scenarios:

  1. Each of the router's action branches dispatches to the right
     sub-capability, and the AGENT_DECISION artifact is always the
     FIRST output (so ops can jq-parse it without walking past the
     sub-capability's artifacts).

  2. Missing sub-capability + the router selecting that action raises
     the typed AUTO_<sub>_UNAVAILABLE error code. MOCK / others are
     unaffected — the failure is isolated to the AUTO dispatch path.

  3. Short text-only input routes to clarify without invoking any sub,
     and the inline FINAL_RESPONSE carries the Korean clarify message.

  4. No input at all (no text, no file) raises AUTO_NO_INPUT before
     the router is even consulted — this fail-fast matches the core-api
     boundary's AUTO_NO_INPUT rule.

  5. direct_answer via a fake chat provider populates a FINAL_RESPONSE
     with the answer text; on chat failure it degrades to the clarify
     message rather than raising.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest

from app.capabilities.agent.capability import AutoCapability
from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
    RuleBasedAgentRouter,
)
from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
)
from app.clients.llm_chat import ChatResult, LlmChatError, LlmChatProvider


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SentinelCapability(Capability):
    """Records the CapabilityInput it was called with and returns a
    sentinel FINAL_RESPONSE so the dispatch is observable."""

    def __init__(self, name: str, payload: str) -> None:
        self.name = name
        self._payload = payload
        self.calls: List[CapabilityInput] = []

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        self.calls.append(input)
        return CapabilityOutput(
            outputs=[
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename=f"{self.name.lower()}-sentinel.md",
                    content_type="text/markdown; charset=utf-8",
                    content=self._payload.encode("utf-8"),
                )
            ]
        )


class _FixedRouter(AgentRouterProvider):
    """Router that always returns the same pre-baked decision."""

    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision
        self.calls: List[dict] = []

    @property
    def name(self) -> str:
        return self._decision.router_name

    def decide(
        self,
        *,
        text: Optional[str],
        has_file: bool,
        file_mime: Optional[str],
        file_size: int,
    ) -> AgentDecision:
        self.calls.append(
            {
                "text": text,
                "has_file": has_file,
                "file_mime": file_mime,
                "file_size": file_size,
            }
        )
        return self._decision


class _FakeChat(LlmChatProvider):
    def __init__(
        self,
        *,
        json_return: Optional[dict] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._json_return = json_return
        self._error = error
        self.chat_json_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fake-chat"

    @property
    def capabilities(self) -> dict:
        return {
            "function_calling": True,
            "thinking": False,
            "json_mode": True,
            "vision": False,
            "audio": False,
        }

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
            {"schema_hint": schema_hint, "max_tokens": max_tokens}
        )
        if self._error is not None:
            raise self._error
        assert self._json_return is not None
        return self._json_return

    def chat_tools(self, messages, tools, **kwargs):  # pragma: no cover - unused
        raise LlmChatError("not used in direct_answer tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input(
    *,
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_mime: Optional[str] = None,
    filename: Optional[str] = None,
    job_id: str = "job-auto-1",
) -> CapabilityInput:
    artifacts: List[CapabilityInputArtifact] = []
    if text is not None:
        artifacts.append(
            CapabilityInputArtifact(
                artifact_id="art-text",
                type="INPUT_TEXT",
                content=text.encode("utf-8"),
                content_type="text/plain; charset=utf-8",
            )
        )
    if file_bytes is not None:
        artifacts.append(
            CapabilityInputArtifact(
                artifact_id="art-file",
                type="INPUT_FILE",
                content=file_bytes,
                content_type=file_mime,
                filename=filename,
            )
        )
    return CapabilityInput(
        job_id=job_id,
        capability="AUTO",
        attempt_no=1,
        inputs=artifacts,
    )


def _decision(
    *,
    action: str,
    confidence: float = 0.9,
    router_name: str = "rule",
    parsed_query: Optional[ParsedQuery] = None,
    reason: str = "test decision",
) -> AgentDecision:
    return AgentDecision(
        action=action,  # type: ignore[arg-type]
        reason=reason,
        parsed_query=parsed_query,
        confidence=confidence,
        router_name=router_name,
    )


def _build_auto(
    *,
    decision: AgentDecision,
    rag: Optional[Capability] = None,
    ocr: Optional[Capability] = None,
    multimodal: Optional[Capability] = None,
    chat: Optional[LlmChatProvider] = None,
    parser: Optional[QueryParserProvider] = None,
) -> tuple[AutoCapability, _FixedRouter]:
    router = _FixedRouter(decision)
    cap = AutoCapability(
        router=router,
        parser=parser or NoOpQueryParser(),
        rag=rag,
        ocr=ocr,
        multimodal=multimodal,
        chat=chat,
    )
    return cap, router


# ---------------------------------------------------------------------------
# 1. Each action branch dispatches correctly.
# ---------------------------------------------------------------------------


def test_auto_dispatches_to_rag_when_router_picks_rag():
    rag = _SentinelCapability("RAG", "rag answer")
    ocr = _SentinelCapability("OCR", "ocr answer")
    multimodal = _SentinelCapability("MULTIMODAL", "mm answer")
    cap, router = _build_auto(
        decision=_decision(action="rag", router_name="rule", confidence=0.7),
        rag=rag,
        ocr=ocr,
        multimodal=multimodal,
    )

    result = cap.run(_input(text="what is bge-m3"))
    types = [a.type for a in result.outputs]
    assert types[0] == "AGENT_DECISION"
    assert types[1] == "FINAL_RESPONSE"
    assert result.outputs[1].content == b"rag answer"
    assert len(rag.calls) == 1
    assert len(ocr.calls) == 0
    assert len(multimodal.calls) == 0
    assert router.calls[0]["has_file"] is False


def test_auto_dispatches_to_ocr_when_router_picks_ocr():
    ocr = _SentinelCapability("OCR", "ocr answer")
    cap, _ = _build_auto(
        decision=_decision(action="ocr", confidence=0.9),
        ocr=ocr,
    )
    result = cap.run(
        _input(file_bytes=b"\x89PNG\r\n\x1a\n...", file_mime="image/png"),
    )
    assert [a.type for a in result.outputs] == ["AGENT_DECISION", "FINAL_RESPONSE"]
    assert result.outputs[1].content == b"ocr answer"
    assert len(ocr.calls) == 1


def test_auto_dispatches_to_multimodal_when_router_picks_multimodal():
    mm = _SentinelCapability("MULTIMODAL", "mm answer")
    cap, router = _build_auto(
        decision=_decision(action="multimodal", confidence=0.95),
        multimodal=mm,
    )
    result = cap.run(
        _input(
            text="what is the total on this invoice",
            file_bytes=b"\x89PNG\r\n\x1a\n...",
            file_mime="image/png",
        ),
    )
    assert [a.type for a in result.outputs] == ["AGENT_DECISION", "FINAL_RESPONSE"]
    assert result.outputs[1].content == b"mm answer"
    # The sub-capability receives the SAME CapabilityInput — AutoCapability
    # does not mutate it.
    assert len(mm.calls) == 1
    assert mm.calls[0].inputs[0].type == "INPUT_TEXT"


def test_agent_decision_artifact_is_always_first_and_parseable():
    cap, _ = _build_auto(
        decision=_decision(
            action="rag", confidence=0.77, reason="long text only"
        ),
        rag=_SentinelCapability("RAG", "rag"),
    )
    result = cap.run(_input(text="what is an index build"))

    assert result.outputs[0].type == "AGENT_DECISION"
    assert result.outputs[0].content_type == "application/json"

    body = json.loads(result.outputs[0].content.decode("utf-8"))
    assert body["action"] == "rag"
    assert body["confidence"] == pytest.approx(0.77)
    assert body["routerName"] == "rule"
    assert body["reason"] == "long text only"
    assert body["parsedQuery"] is None


def test_agent_decision_includes_parsed_query_when_router_emitted_one():
    pq = ParsedQuery(
        original="what is bge-m3",
        normalized="what is bge-m3",
        keywords=["bge-m3"],
        intent="definition",
        rewrites=[],
        filters={},
        parser_name="regex",
    )
    cap, _ = _build_auto(
        decision=_decision(action="rag", parsed_query=pq, confidence=0.8),
        rag=_SentinelCapability("RAG", "rag"),
    )
    result = cap.run(_input(text="what is bge-m3"))
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    assert body["parsedQuery"]["parserName"] == "regex"
    assert body["parsedQuery"]["intent"] == "definition"


# ---------------------------------------------------------------------------
# 2. Missing sub-capability raises typed AUTO_<sub>_UNAVAILABLE.
# ---------------------------------------------------------------------------


def test_auto_raises_when_router_picks_rag_but_rag_missing():
    cap, _ = _build_auto(
        decision=_decision(action="rag", confidence=0.7),
        # rag=None intentionally
        ocr=_SentinelCapability("OCR", "ocr"),
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input(text="what is a retriever"))
    assert ex_info.value.code == "AUTO_RAG_UNAVAILABLE"


def test_auto_raises_when_router_picks_ocr_but_ocr_missing():
    cap, _ = _build_auto(
        decision=_decision(action="ocr", confidence=0.9),
        rag=_SentinelCapability("RAG", "rag"),
        # ocr=None intentionally
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input(file_bytes=b"PNGDATA", file_mime="image/png"))
    assert ex_info.value.code == "AUTO_OCR_UNAVAILABLE"


def test_auto_raises_when_router_picks_multimodal_but_multimodal_missing():
    cap, _ = _build_auto(
        decision=_decision(action="multimodal", confidence=0.95),
        rag=_SentinelCapability("RAG", "rag"),
        ocr=_SentinelCapability("OCR", "ocr"),
        # multimodal=None intentionally
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(
            _input(
                text="what is on this",
                file_bytes=b"%PDF-1.5",
                file_mime="application/pdf",
            )
        )
    assert ex_info.value.code == "AUTO_MULTIMODAL_UNAVAILABLE"


# ---------------------------------------------------------------------------
# 3. Clarify short-text path.
# ---------------------------------------------------------------------------


def test_auto_short_text_only_emits_clarify_without_invoking_subs():
    rag = _SentinelCapability("RAG", "rag")
    ocr = _SentinelCapability("OCR", "ocr")
    cap, _ = _build_auto(
        decision=_decision(action="clarify", confidence=0.5, router_name="rule"),
        rag=rag,
        ocr=ocr,
    )
    result = cap.run(_input(text="hi"))
    assert [a.type for a in result.outputs] == ["AGENT_DECISION", "FINAL_RESPONSE"]
    assert "모호" in result.outputs[1].content.decode("utf-8")
    assert len(rag.calls) == 0
    assert len(ocr.calls) == 0


def test_auto_clarify_response_is_markdown():
    cap, _ = _build_auto(
        decision=_decision(action="clarify", confidence=0.0),
    )
    result = cap.run(_input(text="x"))
    final = result.outputs[1]
    assert final.type == "FINAL_RESPONSE"
    assert final.content_type.startswith("text/markdown")


# ---------------------------------------------------------------------------
# 4. No input at all fails fast.
# ---------------------------------------------------------------------------


def test_auto_no_text_and_no_file_raises_auto_no_input():
    cap, router = _build_auto(
        decision=_decision(action="clarify", confidence=0.0),
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input())  # no text, no file
    assert ex_info.value.code == "AUTO_NO_INPUT"
    # The router should never have been called — fail fast.
    assert router.calls == []


def test_auto_blank_text_and_no_file_raises_auto_no_input():
    cap, router = _build_auto(
        decision=_decision(action="clarify", confidence=0.0),
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input(text="   \n  "))  # whitespace only
    assert ex_info.value.code == "AUTO_NO_INPUT"
    assert router.calls == []


# ---------------------------------------------------------------------------
# 5. direct_answer routes through chat, falls back on error.
# ---------------------------------------------------------------------------


def test_auto_direct_answer_uses_chat_when_configured():
    chat = _FakeChat(json_return={"answer": "The answer is 42."})
    cap, _ = _build_auto(
        decision=_decision(
            action="direct_answer", confidence=0.9, router_name="llm-fake"
        ),
        chat=chat,
    )
    result = cap.run(_input(text="what is the meaning of life"))
    final = result.outputs[1]
    assert final.type == "FINAL_RESPONSE"
    assert final.content.decode("utf-8") == "The answer is 42."
    assert len(chat.chat_json_calls) == 1


def test_auto_direct_answer_falls_back_to_clarify_on_chat_error(caplog):
    chat = _FakeChat(error=LlmChatError("timeout"))
    cap, _ = _build_auto(
        decision=_decision(action="direct_answer", confidence=0.9),
        chat=chat,
    )
    with caplog.at_level("WARNING"):
        result = cap.run(_input(text="some question"))
    final = result.outputs[1]
    assert "모호" in final.content.decode("utf-8")
    assert any("chat backend failed" in rec.message for rec in caplog.records)


def test_auto_direct_answer_uses_clarify_when_chat_missing():
    cap, _ = _build_auto(
        decision=_decision(action="direct_answer", confidence=0.9),
        chat=None,
    )
    result = cap.run(_input(text="anything"))
    assert "모호" in result.outputs[1].content.decode("utf-8")


def test_auto_direct_answer_uses_clarify_when_chat_returns_blank_answer(caplog):
    chat = _FakeChat(json_return={"answer": "   "})
    cap, _ = _build_auto(
        decision=_decision(action="direct_answer", confidence=0.9),
        chat=chat,
    )
    with caplog.at_level("WARNING"):
        result = cap.run(_input(text="x"))
    assert "모호" in result.outputs[1].content.decode("utf-8")
    assert any("no 'answer' field" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end with the real rule router + fake subs.
# ---------------------------------------------------------------------------


def test_auto_with_rule_router_and_text_pdf_dispatches_to_multimodal():
    rag = _SentinelCapability("RAG", "rag")
    ocr = _SentinelCapability("OCR", "ocr")
    mm = _SentinelCapability("MULTIMODAL", "mm answer from fused context")
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
        rag=rag,
        ocr=ocr,
        multimodal=mm,
    )
    result = cap.run(
        _input(
            text="what is the total on this invoice",
            file_bytes=b"%PDF-1.5 content",
            file_mime="application/pdf",
        )
    )
    assert [a.type for a in result.outputs] == [
        "AGENT_DECISION",
        "FINAL_RESPONSE",
    ]
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    assert body["action"] == "multimodal"
    assert body["routerName"] == "rule"
    assert result.outputs[1].content == b"mm answer from fused context"
    assert len(mm.calls) == 1


def test_auto_with_rule_router_and_text_only_dispatches_to_rag():
    rag = _SentinelCapability("RAG", "rag grounded answer")
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
        rag=rag,
        ocr=None,
        multimodal=None,
    )
    result = cap.run(_input(text="which anime features harbor cats"))
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    assert body["action"] == "rag"
    assert result.outputs[1].content == b"rag grounded answer"


def test_auto_with_rule_router_and_unsupported_file_plus_long_text_routes_to_rag():
    rag = _SentinelCapability("RAG", "rag")
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
        rag=rag,
        ocr=None,
        multimodal=None,
    )
    result = cap.run(
        _input(
            text="describe this gif for me please",
            file_bytes=b"GIF89a",
            file_mime="image/gif",
        )
    )
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    # Rule router ignores unsupported files → routes on text alone.
    assert body["action"] == "rag"


def test_auto_name_is_auto():
    cap = AutoCapability(
        router=RuleBasedAgentRouter(),
        parser=NoOpQueryParser(),
    )
    assert cap.name == "AUTO"


# ---------------------------------------------------------------------------
# AGENT_DECISION trace observability — additive, must not change existing
# keys or the outputArtifacts type set.
# ---------------------------------------------------------------------------


def test_auto_decision_payload_carries_trace_with_classify_route_dispatch():
    rag = _SentinelCapability("RAG", "rag answer")
    cap, _ = _build_auto(
        decision=_decision(action="rag", confidence=0.7, router_name="rule"),
        rag=rag,
    )
    result = cap.run(_input(text="what is bge-m3"))

    body = json.loads(result.outputs[0].content.decode("utf-8"))

    # Existing keys must remain untouched.
    for legacy_key in ("action", "reason", "confidence", "routerName", "parsedQuery"):
        assert legacy_key in body

    # The new additive trace key carries the classify / route / dispatch flow.
    assert "trace" in body
    trace = body["trace"]
    assert trace["capability"] == "AUTO"
    assert trace["finalStatus"] == "ok"
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["classify", "route", "dispatch"]

    # Stage details record the routing decision the user can read back.
    route_stage = trace["stages"][1]
    assert route_stage["provider"] == "rule"
    assert route_stage["details"]["action"] == "rag"
    dispatch_stage = trace["stages"][2]
    assert dispatch_stage["provider"] == "rag"
    assert dispatch_stage["details"]["subOutputCount"] == 1

    # Output artifact types must stay [AGENT_DECISION, sub outputs] — no
    # new artifact type was introduced.
    assert [a.type for a in result.outputs] == ["AGENT_DECISION", "FINAL_RESPONSE"]


def test_auto_decision_trace_summary_is_human_readable():
    cap, _ = _build_auto(
        decision=_decision(action="clarify", confidence=0.5),
    )
    result = cap.run(_input(text="hi there bot"))
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    summary = body["trace"]["summary"]
    assert "classify:ok" in summary
    assert "route:ok" in summary
    assert "dispatch:ok" in summary


def test_auto_rag_unavailable_error_message_carries_trace_summary():
    cap, _ = _build_auto(
        decision=_decision(action="rag", confidence=0.7),
        # rag=None intentionally — _dispatch_single_pass will _require it
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input(text="what is a retriever"))
    assert ex_info.value.code == "AUTO_RAG_UNAVAILABLE"
    msg = ex_info.value.message
    # The fold-in must include the literal "trace:" tag and the
    # classify/route stage summaries so operators can read the
    # progression from the FAILED callback errorMessage alone.
    assert "trace:" in msg
    assert "classify:ok" in msg
    assert "route:ok" in msg


def test_auto_no_input_error_message_carries_trace_summary():
    cap, _ = _build_auto(
        decision=_decision(action="clarify", confidence=0.0),
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input())  # no text, no file
    assert ex_info.value.code == "AUTO_NO_INPUT"
    msg = ex_info.value.message
    # AUTO_NO_INPUT raises before the router is invoked, so only the
    # classify stage is in the summary — and it should be marked fail.
    assert "trace:" in msg
    assert "classify:fail" in msg


def test_auto_dispatch_failure_in_sub_capability_records_dispatch_fail_and_reraises():
    """A sub-capability raising preserves its own errorMessage (no AUTO
    fold-in) but the trace records a dispatch:fail. The TaskRunner's
    existing FAILED callback flow takes the original CapabilityError
    unchanged."""

    class _BoomCapability(Capability):
        name = "RAG"

        def run(self, input):
            raise CapabilityError("EMPTY_QUERY", "RAG saw no query")

    cap, _ = _build_auto(
        decision=_decision(action="rag", confidence=0.7),
        rag=_BoomCapability(),
    )
    with pytest.raises(CapabilityError) as ex_info:
        cap.run(_input(text="what is bge-m3"))
    # Sub-capability error code is preserved, not rewrapped.
    assert ex_info.value.code == "EMPTY_QUERY"
    # And the AUTO fold-in tag is NOT appended to a non-AUTO_TYPED code.
    assert "| trace:" not in ex_info.value.message


def test_auto_outputartifact_type_set_unchanged_by_trace_observability():
    """The trace boost must remain additive — single-pass dispatch still
    emits exactly [AGENT_DECISION, <sub artifact types>] in order."""
    rag = _SentinelCapability("RAG", "rag answer")
    cap, _ = _build_auto(
        decision=_decision(action="rag", confidence=0.7),
        rag=rag,
    )
    result = cap.run(_input(text="some long enough question"))
    types = [a.type for a in result.outputs]
    assert types == ["AGENT_DECISION", "FINAL_RESPONSE"]
    # No new artifact type was introduced.
    assert "AGENT_TRACE" not in types
    assert "AGENT_DECISION_TRACE" not in types
