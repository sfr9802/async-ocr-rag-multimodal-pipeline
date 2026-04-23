"""Tests for the LlmChatProvider seam + NoOp / Ollama / Claude providers.

Everything runs offline. The Ollama tests monkeypatch ``httpx.Client``
(inside the llm_chat module) with a fake session that returns canned
responses. The Claude tests inject a MagicMock for the anthropic client
and a fake anthropic module into ``sys.modules`` so import works even
on a fresh venv without the SDK installed.

Scenario coverage per backend:

  NoOp
    - both methods raise LlmChatError
    - capabilities() is the all-false matrix

  Ollama
    - chat_json happy path (dict parsed from "message.content")
    - chat_json raises on HTTP 5xx
    - chat_json raises on timeout
    - chat_json raises on non-JSON content
    - chat_json raises on empty content
    - chat_json raises when response is not an object (e.g. list)
    - chat_tools native-tool path returns ChatResult with tool_call
    - chat_tools falls back to text when backend emits no tool_call
    - enable_thinking=True prepends "<|think|>" to the first system message
    - enable_thinking is silently ignored when capabilities.thinking=False
    - schema_hint reminder is appended to the last system message
    - capabilities() reports thinking=True only for gemma4-family models

  Claude
    - chat_json happy path (dict parsed from response text block)
    - chat_json raises when SDK throws any exception
    - chat_json raises on non-JSON content
    - chat_json raises on empty content
    - chat_tools returns ChatResult with tool_call when tool_use present
    - chat_tools returns ChatResult.text when no tool_use block
    - system messages are collapsed and moved to the 'system' kwarg
    - enable_thinking=True threads the thinking kwarg through
    - capabilities() advertises function_calling/thinking/json_mode/vision
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

# Install a fake anthropic module BEFORE importing ClaudeChatProvider so
# the local import inside the module (if any) doesn't fail on CI.
_fake_anthropic_module = MagicMock()
sys.modules.setdefault("anthropic", _fake_anthropic_module)

from app.clients.llm_chat import (
    ChatMessage,
    ChatResult,
    ChatToolSpec,
    ClaudeChatProvider,
    LlmChatError,
    NoOpChatProvider,
    OllamaChatProvider,
)


# ---------------------------------------------------------------------------
# Fake httpx primitives for Ollama tests.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, body: Optional[dict] = None):
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            # Import here so the real httpx is used for the exception type.
            import httpx

            request = httpx.Request("POST", "http://fake/api/chat")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request, text=self.text),
            )

    def json(self) -> dict:
        return self._body


class _FakeClient:
    """Drop-in replacement for httpx.Client used by OllamaChatProvider._post."""

    def __init__(
        self,
        response: Optional[_FakeResponse] = None,
        *,
        raise_on_post: Optional[BaseException] = None,
    ):
        self._response = response
        self._raise_on_post = raise_on_post
        self.recorded_payloads: List[dict] = []
        self.recorded_urls: List[str] = []

    # The ``with httpx.Client(timeout=...) as client:`` usage pattern
    # means we need the context-manager protocol AND the constructor
    # signature. Monkeypatch-side we install a callable that returns
    # self.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):  # noqa: A002 - matches httpx signature
        self.recorded_urls.append(url)
        self.recorded_payloads.append(json)
        if self._raise_on_post is not None:
            raise self._raise_on_post
        assert self._response is not None
        return self._response


def _install_fake_httpx(monkeypatch, client_factory):
    """Monkeypatch ``httpx.Client`` inside the llm_chat module's namespace.

    The provider imports httpx lazily inside ``_post`` so we target the
    ``httpx`` module attribute directly.
    """
    import httpx

    monkeypatch.setattr(httpx, "Client", client_factory)


# ---------------------------------------------------------------------------
# NoOp provider
# ---------------------------------------------------------------------------


def test_noop_capabilities_are_all_false():
    noop = NoOpChatProvider()
    assert noop.name == "noop"
    caps = noop.capabilities
    assert caps == {
        "function_calling": False,
        "thinking": False,
        "json_mode": False,
        "vision": False,
        "audio": False,
    }


def test_noop_chat_json_raises():
    noop = NoOpChatProvider()
    with pytest.raises(LlmChatError):
        noop.chat_json(
            [ChatMessage(role="user", content="hi")], schema_hint="{}"
        )


def test_noop_chat_tools_raises():
    noop = NoOpChatProvider()
    with pytest.raises(LlmChatError):
        noop.chat_tools(
            [ChatMessage(role="user", content="hi")], tools=[],
        )


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


def _ollama_provider(model: str = "gemma4:e2b") -> OllamaChatProvider:
    return OllamaChatProvider(
        base_url="http://localhost:11434",
        model=model,
        timeout_s=5.0,
        keep_alive="30m",
    )


def test_ollama_capabilities_gemma4():
    provider = _ollama_provider("gemma4:e2b")
    caps = provider.capabilities
    assert caps["function_calling"] is True
    assert caps["thinking"] is True
    assert caps["json_mode"] is True
    assert caps["vision"] is True
    assert caps["audio"] is False
    assert provider.name == "ollama-gemma4:e2b"


def test_ollama_capabilities_non_gemma4_drops_thinking_and_vision():
    provider = _ollama_provider("llama3:8b")
    caps = provider.capabilities
    assert caps["thinking"] is False
    assert caps["vision"] is False
    assert caps["function_calling"] is True
    assert caps["json_mode"] is True


def test_ollama_chat_json_happy_path(monkeypatch):
    body = {
        "message": {"role": "assistant", "content": json.dumps({"ok": True, "k": ["a"]})},
        "prompt_eval_count": 10,
        "eval_count": 7,
    }
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    result = provider.chat_json(
        [
            ChatMessage(role="system", content="You extract intent."),
            ChatMessage(role="user", content="query X"),
        ],
        schema_hint='{"ok": bool}',
    )

    assert result == {"ok": True, "k": ["a"]}
    # The provider forwards the required knobs.
    payload = client.recorded_payloads[0]
    assert payload["model"] == "gemma4:e2b"
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["options"]["temperature"] == 0.0
    # Schema hint appended to last system message.
    system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
    assert any('{"ok": bool}' in m["content"] for m in system_msgs)


def test_ollama_chat_json_raises_on_5xx(monkeypatch):
    client = _FakeClient(_FakeResponse(status_code=500, body={"error": "boom"}))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "500" in str(exc_info.value)


def test_ollama_chat_json_raises_on_timeout(monkeypatch):
    import httpx

    client = _FakeClient(raise_on_post=httpx.TimeoutException("deadline"))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
            timeout_s=1.0,
        )
    assert "timed out" in str(exc_info.value).lower()


def test_ollama_chat_json_raises_on_invalid_json(monkeypatch):
    body = {"message": {"role": "assistant", "content": "not valid json {"}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "non-JSON" in str(exc_info.value)


def test_ollama_chat_json_raises_on_empty_content(monkeypatch):
    body = {"message": {"role": "assistant", "content": "   "}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError):
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )


def test_ollama_chat_json_raises_when_response_is_list(monkeypatch):
    body = {"message": {"role": "assistant", "content": json.dumps([1, 2, 3])}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "object" in str(exc_info.value)


def test_ollama_chat_tools_native_tool_call(monkeypatch):
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": {"q": "cats"},
                    }
                }
            ],
        },
        "prompt_eval_count": 22,
        "eval_count": 3,
    }
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    tools = [
        ChatToolSpec(
            name="search",
            description="Search the corpus",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
    ]
    result = provider.chat_tools(
        [ChatMessage(role="user", content="find cats")],
        tools=tools,
    )
    assert isinstance(result, ChatResult)
    assert result.text is None
    assert result.tool_call == {"name": "search", "arguments": {"q": "cats"}}
    assert result.tokens_in == 22
    assert result.tokens_out == 3
    assert result.latency_ms >= 0.0
    # Ollama tool payload translation.
    payload = client.recorded_payloads[0]
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "search"


def test_ollama_chat_tools_falls_back_to_text_when_no_tool_call(monkeypatch):
    body = {
        "message": {"role": "assistant", "content": "just some text", "tool_calls": []},
    }
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    result = provider.chat_tools(
        [ChatMessage(role="user", content="x")],
        tools=[
            ChatToolSpec(name="n", description="d", parameters={"type": "object"})
        ],
    )
    assert result.text == "just some text"
    assert result.tool_call is None


def test_ollama_chat_tools_raises_when_empty(monkeypatch):
    body = {"message": {"role": "assistant", "content": "", "tool_calls": []}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    with pytest.raises(LlmChatError):
        provider.chat_tools(
            [ChatMessage(role="user", content="x")],
            tools=[ChatToolSpec(name="n", description="d", parameters={})],
        )


def test_ollama_enable_thinking_prepends_marker_for_gemma4(monkeypatch):
    body = {"message": {"role": "assistant", "content": json.dumps({"ok": True})}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider("gemma4:e2b")
    provider.chat_json(
        [
            ChatMessage(role="system", content="base system"),
            ChatMessage(role="user", content="q"),
        ],
        schema_hint="{}",
        enable_thinking=True,
    )
    payload = client.recorded_payloads[0]
    first_system = next(m for m in payload["messages"] if m["role"] == "system")
    assert first_system["content"].startswith("<|think|>")
    assert "base system" in first_system["content"]


def test_ollama_enable_thinking_ignored_on_non_thinking_model(monkeypatch):
    body = {"message": {"role": "assistant", "content": json.dumps({"ok": True})}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider("llama3:8b")
    provider.chat_json(
        [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q"),
        ],
        schema_hint="{}",
        enable_thinking=True,
    )
    payload = client.recorded_payloads[0]
    first_system = next(m for m in payload["messages"] if m["role"] == "system")
    assert "<|think|>" not in first_system["content"]


def test_ollama_enable_thinking_inserts_system_message_when_none_present(monkeypatch):
    body = {"message": {"role": "assistant", "content": json.dumps({"ok": True})}}
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider("gemma4:e2b")
    provider.chat_json(
        [ChatMessage(role="user", content="q")],
        schema_hint="{}",
        enable_thinking=True,
    )
    payload = client.recorded_payloads[0]
    # Schema hint creates a system message first; thinking marker prepends to it.
    first_system = next(m for m in payload["messages"] if m["role"] == "system")
    assert first_system["content"].startswith("<|think|>")


def test_ollama_tool_call_arguments_may_be_json_string(monkeypatch):
    """Some Ollama builds emit the arguments as a JSON-encoded string
    instead of a dict. The provider normalises both shapes."""
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "n", "arguments": json.dumps({"k": "v"})}}
            ],
        }
    }
    client = _FakeClient(_FakeResponse(body=body))
    _install_fake_httpx(monkeypatch, lambda timeout=None: client)

    provider = _ollama_provider()
    result = provider.chat_tools(
        [ChatMessage(role="user", content="x")],
        tools=[ChatToolSpec(name="n", description="d", parameters={})],
    )
    assert result.tool_call == {"name": "n", "arguments": {"k": "v"}}


def test_ollama_empty_messages_raises():
    provider = _ollama_provider()
    with pytest.raises(LlmChatError):
        provider.chat_json([], schema_hint="{}")


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeClaudeResponse:
    content: List[object] = field(default_factory=list)
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    model: str = "claude-haiku-4-5-20251001"


def _claude_provider(client_mock: MagicMock) -> ClaudeChatProvider:
    return ClaudeChatProvider(
        anthropic_client=client_mock,
        model="claude-haiku-4-5-20251001",
    )


def test_claude_capabilities_all_advertised():
    provider = _claude_provider(MagicMock())
    caps = provider.capabilities
    assert caps["function_calling"] is True
    assert caps["thinking"] is True
    assert caps["json_mode"] is True
    assert caps["vision"] is True
    assert caps["audio"] is False
    assert provider.name.startswith("claude-")


def test_claude_chat_json_happy_path():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[_FakeTextBlock(text=json.dumps({"intent": "factoid"}))],
        usage=_FakeUsage(input_tokens=5, output_tokens=4),
    )

    provider = _claude_provider(client)
    result = provider.chat_json(
        [
            ChatMessage(role="system", content="You extract intent."),
            ChatMessage(role="user", content="What is FAISS?"),
        ],
        schema_hint='{"intent": str}',
    )
    assert result == {"intent": "factoid"}

    # System prompts are collapsed and passed via the kwarg.
    call = client.messages.create.call_args
    assert "system" in call.kwargs
    assert "You extract intent." in call.kwargs["system"]
    assert '{"intent": str}' in call.kwargs["system"]
    assert call.kwargs["messages"][0]["role"] == "user"
    # No "system" role survives in messages=.
    for m in call.kwargs["messages"]:
        assert m["role"] != "system"


def test_claude_chat_json_raises_when_sdk_throws():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("upstream boom")
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "Claude chat_json failed" in str(exc_info.value)


def test_claude_chat_json_raises_on_invalid_json():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[_FakeTextBlock(text="not json {{{")],
    )
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "non-JSON" in str(exc_info.value)


def test_claude_chat_json_raises_on_empty_text():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(content=[])
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError):
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )


def test_claude_chat_json_raises_when_response_is_list():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[_FakeTextBlock(text=json.dumps([1, 2, 3]))],
    )
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError) as exc_info:
        provider.chat_json(
            [ChatMessage(role="user", content="q")],
            schema_hint="{}",
        )
    assert "object" in str(exc_info.value)


def test_claude_chat_tools_with_tool_use_block():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[
            _FakeToolUseBlock(name="search", input={"q": "cats"}),
        ],
        usage=_FakeUsage(input_tokens=20, output_tokens=9),
    )
    provider = _claude_provider(client)
    tools = [
        ChatToolSpec(
            name="search",
            description="Search the corpus",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
    ]
    result = provider.chat_tools(
        [ChatMessage(role="user", content="find cats")],
        tools=tools,
    )
    assert isinstance(result, ChatResult)
    assert result.tool_call == {"name": "search", "arguments": {"q": "cats"}}
    assert result.tokens_in == 20
    assert result.tokens_out == 9

    # input_schema translation — Anthropic calls it input_schema, we
    # accept parameters in our ChatToolSpec.
    call = client.messages.create.call_args
    sent_tools = call.kwargs["tools"]
    assert sent_tools[0]["name"] == "search"
    assert "input_schema" in sent_tools[0]


def test_claude_chat_tools_falls_back_to_text_when_no_tool_use():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[_FakeTextBlock(text="just a sentence")],
    )
    provider = _claude_provider(client)
    result = provider.chat_tools(
        [ChatMessage(role="user", content="x")],
        tools=[ChatToolSpec(name="n", description="d", parameters={})],
    )
    assert result.tool_call is None
    assert result.text == "just a sentence"


def test_claude_chat_tools_raises_when_empty():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(content=[])
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError):
        provider.chat_tools(
            [ChatMessage(role="user", content="x")],
            tools=[ChatToolSpec(name="n", description="d", parameters={})],
        )


def test_claude_enable_thinking_threads_kwarg():
    client = MagicMock()
    client.messages.create.return_value = _FakeClaudeResponse(
        content=[_FakeTextBlock(text=json.dumps({"ok": True}))],
    )
    provider = _claude_provider(client)
    provider.chat_json(
        [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q"),
        ],
        schema_hint="{}",
        enable_thinking=True,
    )
    call = client.messages.create.call_args
    assert "thinking" in call.kwargs
    assert call.kwargs["thinking"]["type"] == "enabled"


def test_claude_messages_must_contain_user_turn():
    client = MagicMock()
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError):
        provider.chat_json(
            [ChatMessage(role="system", content="system-only")],
            schema_hint="{}",
        )
    # No call should have reached the client.
    client.messages.create.assert_not_called()


def test_claude_empty_messages_raises():
    client = MagicMock()
    provider = _claude_provider(client)
    with pytest.raises(LlmChatError):
        provider.chat_json([], schema_hint="{}")
