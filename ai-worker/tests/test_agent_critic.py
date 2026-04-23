"""AgentCriticProvider tests.

Three scenario groups, all fully offline:

  1. ``NoOpCritic`` — always sufficient, zero tokens. Used when the
     loop is disabled.

  2. ``RuleCritic`` — deterministic heuristic. Covers the empty-answer,
     short-answer, and "insufficient marker" branches in Korean and
     English.

  3. ``LlmCritic`` — mocked chat backend. Exercises all five letters
     (A-E), the ``LlmChatError`` -> rule fallback path, the invalid
     payload -> rule fallback path, and the chat_json non-tools path.

Zero new deps; everything runs under the stdlib plus the modules under
test.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from app.capabilities.agent.critic import (
    AgentCriticProvider,
    CritiqueResult,
    LlmCritic,
    NoOpCritic,
    RuleCritic,
)
from app.capabilities.rag.generation import RetrievedChunk
from app.clients.llm_chat import (
    ChatMessage,
    ChatResult,
    ChatToolSpec,
    LlmChatError,
    LlmChatProvider,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _chunk(cid: str, text: str, doc_id: str = "doc-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc_id, section="", text=text, score=0.9,
    )


class _FakeChat(LlmChatProvider):
    """Minimal LlmChatProvider stub for LlmCritic unit tests."""

    def __init__(
        self,
        *,
        tool_call: Optional[dict] = None,
        tool_text: Optional[str] = None,
        json_return: Optional[dict] = None,
        error: Optional[Exception] = None,
        capabilities: Optional[dict] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
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
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._name = name
        self.chat_json_calls: list[dict] = []
        self.chat_tools_calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> dict:
        return dict(self._capabilities)

    def chat_json(
        self, messages, *, schema_hint, max_tokens=512, temperature=0.0,
        timeout_s=15.0, enable_thinking=False,
    ) -> dict:
        self.chat_json_calls.append(
            {"schema_hint": schema_hint, "enable_thinking": enable_thinking,
             "messages": list(messages)}
        )
        if self._error is not None:
            raise self._error
        assert self._json_return is not None
        return self._json_return

    def chat_tools(
        self, messages, tools, *, max_tokens=512, temperature=0.0,
        timeout_s=15.0, enable_thinking=False,
    ) -> ChatResult:
        self.chat_tools_calls.append(
            {"enable_thinking": enable_thinking,
             "messages": list(messages), "tools": list(tools)}
        )
        if self._error is not None:
            raise self._error
        return ChatResult(
            text=self._tool_text,
            tool_call=self._tool_call,
            raw={},
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# 1. NoOpCritic
# ---------------------------------------------------------------------------


def test_noop_critic_always_sufficient():
    critic = NoOpCritic()
    result = critic.evaluate(
        question="what is bge-m3",
        answer="",
        retrieved=[],
    )
    assert result.sufficient is True
    assert result.gap_type == "none"
    assert result.critic_name == "noop"
    assert result.llm_tokens_used == 0
    # Confidence is 1.0 so the loop's min_confidence_to_stop passes
    # even at the default 0.75.
    assert result.confidence == pytest.approx(1.0)


def test_noop_critic_ignores_rule_markers():
    # Even with an insufficient marker in the answer, NoOp returns
    # sufficient — the whole point is that the loop effectively
    # degenerates to single-pass under this critic.
    critic = NoOpCritic()
    result = critic.evaluate(
        question="what is X",
        answer="정보가 없습니다",
        retrieved=[_chunk("c1", "something")],
    )
    assert result.sufficient is True


# ---------------------------------------------------------------------------
# 2. RuleCritic
# ---------------------------------------------------------------------------


def test_rule_critic_empty_answer_missing_facts():
    critic = RuleCritic()
    result = critic.evaluate(
        question="what is a retriever",
        answer="",
        retrieved=[_chunk("c1", "foo")],
    )
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"
    assert "empty" in result.gap_reason.lower()
    assert result.critic_name == "rule"


def test_rule_critic_whitespace_only_missing_facts():
    critic = RuleCritic()
    result = critic.evaluate(
        question="what is X",
        answer="   \n\t  ",
        retrieved=[],
    )
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"


def test_rule_critic_short_answer_missing_facts():
    critic = RuleCritic()
    result = critic.evaluate(
        question="what is bge-m3",
        answer="bge-m3 is.",  # 10 chars < 40
        retrieved=[],
    )
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"
    assert "10" in result.gap_reason


def test_rule_critic_long_answer_passes_heuristics():
    critic = RuleCritic()
    answer = (
        "A retriever embeds queries and passages into a common vector "
        "space and returns the nearest neighbours of the query for "
        "downstream generation."
    )
    result = critic.evaluate(
        question="what is a retriever",
        answer=answer,
        retrieved=[_chunk("c1", "text")],
    )
    assert result.sufficient is True
    assert result.gap_type == "none"
    assert 0.0 <= result.confidence <= 1.0


def test_rule_critic_korean_insufficient_marker():
    # Long enough to clear the 40-char floor; the marker check must still fire.
    critic = RuleCritic()
    answer = (
        "이 질문에 대한 관련 문서가 없어 정보가 없습니다. 추가로 "
        "참고할 수 있는 문서가 현재 인덱스에 포함되어 있지 않습니다."
    )
    assert len(answer.strip()) >= 40
    result = critic.evaluate(
        question="해안 고양이에 관한 문서",
        answer=answer,
        retrieved=[],
    )
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"
    # The marker branch emits "insufficient marker ..." in the reason.
    assert "marker" in result.gap_reason.lower()


def test_rule_critic_english_insufficient_marker():
    critic = RuleCritic()
    answer = (
        "Unfortunately I cannot answer this question because the "
        "retrieved passages do not contain the relevant information."
    )
    result = critic.evaluate(
        question="what is X",
        answer=answer,
        retrieved=[],
    )
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"


def test_rule_critic_tokens_are_zero():
    critic = RuleCritic()
    result = critic.evaluate(
        question="q", answer="short", retrieved=[]
    )
    assert result.llm_tokens_used == 0


# ---------------------------------------------------------------------------
# 3. LlmCritic — all five letters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "letter,expected_sufficient,expected_gap_type",
    [
        ("A", True, "none"),
        ("B", False, "missing_facts"),
        ("C", False, "ambiguous"),
        ("D", False, "off_topic"),
        ("E", False, "unanswerable"),
    ],
)
def test_llm_critic_maps_each_letter(
    letter, expected_sufficient, expected_gap_type
):
    chat = _FakeChat(
        tool_call={
            "name": "judge_answer",
            "arguments": {
                "letter": letter,
                "reason": f"letter={letter}",
                "confidence": 0.88,
            },
        },
        tokens_in=42,
        tokens_out=7,
    )
    critic = LlmCritic(chat)
    result = critic.evaluate(
        question="q",
        answer="an answer long enough to pass rule heuristic as well",
        retrieved=[_chunk("c1", "supporting passage")],
    )
    assert result.sufficient is expected_sufficient
    assert result.gap_type == expected_gap_type
    assert result.critic_name == "llm-fake-chat"
    assert result.confidence == pytest.approx(0.88)
    assert result.llm_tokens_used == 49


def test_llm_critic_uses_tool_when_available():
    chat = _FakeChat(
        tool_call={
            "name": "judge_answer",
            "arguments": {
                "letter": "A", "reason": "ok", "confidence": 0.9,
            },
        },
    )
    critic = LlmCritic(chat, enable_thinking=True)
    critic.evaluate(question="q", answer="x" * 50, retrieved=[])
    # chat_tools path must fire when function_calling is advertised.
    assert len(chat.chat_tools_calls) == 1
    assert len(chat.chat_json_calls) == 0
    # Thinking mode propagates through because the fake advertises it.
    assert chat.chat_tools_calls[0]["enable_thinking"] is True


def test_llm_critic_fallback_when_provider_raises(caplog):
    chat = _FakeChat(error=LlmChatError("timeout"))
    critic = LlmCritic(chat)

    with caplog.at_level("WARNING"):
        result = critic.evaluate(
            question="what is X",
            answer="",  # empty -> rule will say missing_facts
            retrieved=[],
        )

    # Fallback to rule => critic_name is "llm-fallback-rule".
    assert result.critic_name == "llm-fallback-rule"
    assert result.sufficient is False
    assert result.gap_type == "missing_facts"
    assert any("falling back" in rec.message.lower() for rec in caplog.records)


def test_llm_critic_fallback_when_letter_is_invalid(caplog):
    chat = _FakeChat(
        tool_call={
            "name": "judge_answer",
            "arguments": {
                "letter": "Z",  # not in A-E
                "reason": "nope",
                "confidence": 0.7,
            },
        },
    )
    critic = LlmCritic(chat)

    with caplog.at_level("WARNING"):
        result = critic.evaluate(
            question="q",
            answer="x" * 50,  # long enough => rule says sufficient
            retrieved=[],
        )
    assert result.critic_name == "llm-fallback-rule"
    assert result.sufficient is True  # rule's verdict on a long answer


def test_llm_critic_fallback_when_tool_args_missing(caplog):
    chat = _FakeChat(tool_call=None, tool_text=None)
    critic = LlmCritic(chat)
    with caplog.at_level("WARNING"):
        result = critic.evaluate(
            question="q", answer="", retrieved=[]
        )
    assert result.critic_name == "llm-fallback-rule"


def test_llm_critic_uses_json_mode_when_no_function_calling():
    chat = _FakeChat(
        capabilities={
            "function_calling": False,
            "thinking": False,
            "json_mode": True,
            "vision": False,
            "audio": False,
        },
        json_return={"letter": "B", "reason": "weak", "confidence": 0.6},
    )
    critic = LlmCritic(chat, enable_thinking=True)
    result = critic.evaluate(
        question="q", answer="x" * 50, retrieved=[]
    )
    assert len(chat.chat_json_calls) == 1
    assert len(chat.chat_tools_calls) == 0
    # Thinking was asked for, but backend doesn't advertise it -> not forwarded.
    assert chat.chat_json_calls[0]["enable_thinking"] is False
    assert result.critic_name == "llm-fake-chat"
    assert result.gap_type == "missing_facts"


def test_llm_critic_confidence_out_of_range_falls_back():
    chat = _FakeChat(
        tool_call={
            "name": "judge_answer",
            "arguments": {
                "letter": "A", "reason": "ok", "confidence": 1.5,
            },
        },
    )
    critic = LlmCritic(chat)
    result = critic.evaluate(
        question="q", answer="x" * 50, retrieved=[]
    )
    assert result.critic_name == "llm-fallback-rule"


def test_llm_critic_rejects_non_chat_provider():
    with pytest.raises(TypeError):
        LlmCritic(object())


def test_critique_result_rejects_sufficient_with_non_none_gap():
    with pytest.raises(ValueError):
        CritiqueResult(
            sufficient=True,
            gap_type="missing_facts",
            gap_reason="x",
            confidence=0.5,
            critic_name="rule",
            llm_tokens_used=0,
        )


def test_critique_result_rejects_insufficient_with_none_gap():
    with pytest.raises(ValueError):
        CritiqueResult(
            sufficient=False,
            gap_type="none",
            gap_reason="x",
            confidence=0.5,
            critic_name="rule",
            llm_tokens_used=0,
        )


def test_critique_result_rejects_confidence_out_of_range():
    with pytest.raises(ValueError):
        CritiqueResult(
            sufficient=True,
            gap_type="none",
            gap_reason="x",
            confidence=1.5,
            critic_name="rule",
            llm_tokens_used=0,
        )


def test_critique_result_to_dict_ensures_ascii_false_capable():
    """CritiqueResult.to_dict() output must JSON-serialize with
    ensure_ascii=False for Korean to survive round-trip."""
    import json

    result = CritiqueResult(
        sufficient=False,
        gap_type="missing_facts",
        gap_reason="정보가 부족합니다",
        confidence=0.8,
        critic_name="rule",
        llm_tokens_used=0,
    )
    body = result.to_dict()
    serialized = json.dumps(body, ensure_ascii=False)
    assert "정보가 부족합니다" in serialized
