"""LlmChatProvider seam + concrete backends.

A single chat-style LLM interface every phase-5/6 component depends on.
The agent router picks a capability branch, the critic scores an answer,
the query parser rewrites a query — all three want the same primitive:
"give me a structured JSON response for these messages, and if the
backend supports tool-calling, route through that path instead."

This file is provider-shaped, not use-case-shaped. Two call modes:

  * chat_json(messages, schema_hint=...) -> dict
      Plain JSON-mode completion. The caller owns the schema; the
      provider guarantees the returned value is a Python dict parsed
      from model output. On any failure (network, timeout, invalid
      JSON, schema violation at the provider layer) it raises
      LlmChatError so callers can fall back cleanly.

  * chat_tools(messages, tools, ...) -> ChatResult
      Function-calling path. The provider decides whether to use the
      backend's native tools API or a prompt-engineered JSON contract.
      Returns a ChatResult with either `text` or `tool_call` populated.

Three backends ship here:

  * NoOpChatProvider — the default. Every method raises LlmChatError.
    CI and env-unset deployments hit this — downstream consumers are
    expected to fall back (regex parser, noop router) without dragging
    RAG down.

  * OllamaChatProvider — local gemma4:e2b by default, talks to the
    Ollama HTTP API. Supports thinking mode (<|think|> prefix) when the
    model advertises it, and Ollama's native tools parameter for
    function calls.

  * ClaudeChatProvider — remote Anthropic API. Uses the same anthropic
    SDK already shipped for ClaudeGenerationProvider / ClaudeVisionProvider.
    Native tool use, native extended thinking.

The interface is deliberately small: no streaming, no multi-turn agent
state, no vision/audio input. Those arrive when a concrete phase needs
them — this seam ships only what the query parser + agent router / critic
actually call on day one.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a chat completion request."""

    role: ChatRole
    content: str


@dataclass(frozen=True)
class ChatToolSpec:
    """Declarative description of a callable the model may invoke.

    ``parameters`` is a JSON-schema object. The provider is responsible
    for translating it to the backend's native tool format (Ollama
    accepts JSON-schema directly; Anthropic wraps it under
    ``input_schema``).
    """

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ChatResult:
    """Normalized chat completion result.

    Exactly one of ``text`` or ``tool_call`` is populated when the call
    succeeded. ``tool_call`` is ``{"name": str, "arguments": dict}`` —
    the same shape across Ollama and Anthropic so downstream code does
    not branch on backend.
    """

    text: Optional[str]
    tool_call: Optional[dict]
    raw: dict
    tokens_in: int
    tokens_out: int
    latency_ms: float


class LlmChatError(Exception):
    """Typed failure — network, timeout, invalid JSON, empty response,
    provider-level validation. Callers catch this to trigger their
    fallback path instead of propagating the raw network/SDK exception.
    """


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class LlmChatProvider(ABC):
    """Shared contract for any chat-style LLM backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> dict:
        """Feature matrix consumed by callers that conditionally enable
        thinking, function calling, or multimodal paths. Keys:
        ``function_calling``, ``thinking``, ``json_mode``, ``vision``,
        ``audio`` — all booleans."""

    @abstractmethod
    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> dict:
        ...

    @abstractmethod
    def chat_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatToolSpec],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> ChatResult:
        ...

    def chat_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        max_tokens: int = 280,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> str:
        """Send a single-image multimodal prompt, return raw text.

        Default raises ``LlmChatError`` — vision-capable backends
        override. Callers must check ``self.capabilities['vision']``
        before invoking; the default here exists so non-vision backends
        (NoOp, plain-text-only Ollama tags, future providers) compose
        cleanly without forcing every subclass to implement vision.
        """
        raise LlmChatError(
            f"chat_vision not supported by {self.name!r} "
            "(capabilities['vision'] is False)."
        )


# ---------------------------------------------------------------------------
# NoOp — CI / env-unset default. Every call raises LlmChatError so callers
# fall back to their offline path (regex parser, noop router, etc).
# ---------------------------------------------------------------------------


class NoOpChatProvider(LlmChatProvider):
    """Placeholder provider. Always raises so fallback paths fire.

    The registry installs this when no LLM backend is configured or when
    backend init fails; it is never the answer itself, only the contract
    that keeps dependent components callable.
    """

    @property
    def name(self) -> str:
        return "noop"

    @property
    def capabilities(self) -> dict:
        return {
            "function_calling": False,
            "thinking": False,
            "json_mode": False,
            "vision": False,
            "audio": False,
        }

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> dict:
        raise LlmChatError("noop")

    def chat_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatToolSpec],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> ChatResult:
        raise LlmChatError("noop")

    def chat_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        max_tokens: int = 280,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> str:
        raise LlmChatError("noop")


# ---------------------------------------------------------------------------
# Ollama — local gemma4:e2b (or any pulled model)
# ---------------------------------------------------------------------------


_THINKING_PREFIX = "<|think|>"


class OllamaChatProvider(LlmChatProvider):
    """Ollama HTTP-API chat provider.

    Talks to ``POST {base_url}/api/chat`` with ``format="json"`` in
    chat_json and ``tools=[...]`` in chat_tools. ``keep_alive`` is
    forwarded so Ollama keeps the model resident between calls — the
    default 30m matches the compose file so short bursts of inference
    don't pay the cold-load penalty every time.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 15.0,
        keep_alive: str = "30m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._default_timeout_s = timeout_s
        self._keep_alive = keep_alive

    @property
    def name(self) -> str:
        return f"ollama-{self._model}"

    @property
    def capabilities(self) -> dict:
        # Gemma 4 supports thinking + native tool use + vision inputs.
        # Older models hit via the same seam advertise the subset they
        # actually handle so callers can gate features off.
        is_gemma4 = self._model.startswith("gemma4")
        return {
            "function_calling": True,
            "thinking": is_gemma4,
            "json_mode": True,
            "vision": is_gemma4,
            "audio": False,
        }

    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> dict:
        prepared = self._prepare_messages(
            messages,
            schema_hint=schema_hint,
            enable_thinking=enable_thinking,
        )
        payload = {
            "model": self._model,
            "messages": prepared,
            "format": "json",
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        body = self._post("/api/chat", payload, timeout_s=timeout_s)
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LlmChatError("Ollama returned an empty message content.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as ex:
            raise LlmChatError(
                f"Ollama returned non-JSON content: {ex}"
            ) from ex
        if not isinstance(parsed, dict):
            raise LlmChatError(
                f"Ollama JSON response must be an object; got {type(parsed).__name__}."
            )
        return parsed

    # ------------------------------------------------------------------

    def chat_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatToolSpec],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> ChatResult:
        prepared = self._prepare_messages(
            messages,
            schema_hint=None,
            enable_thinking=enable_thinking,
        )
        payload = {
            "model": self._model,
            "messages": prepared,
            "stream": False,
            "keep_alive": self._keep_alive,
            "tools": [_tool_spec_to_ollama(t) for t in tools],
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        started_at = time.perf_counter()
        body = self._post("/api/chat", payload, timeout_s=timeout_s)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        message = body.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            first = tool_calls[0] or {}
            fn = first.get("function") or {}
            tool_call = {
                "name": fn.get("name", ""),
                "arguments": _coerce_arguments(fn.get("arguments")),
            }
            return ChatResult(
                text=None,
                tool_call=tool_call,
                raw=body,
                tokens_in=int(body.get("prompt_eval_count") or 0),
                tokens_out=int(body.get("eval_count") or 0),
                latency_ms=elapsed_ms,
            )

        # No native tool_call — fall back to JSON-mode and emit the
        # result as text so callers can still parse something.
        text = message.get("content") or ""
        if not text.strip():
            raise LlmChatError(
                "Ollama tools call returned neither tool_calls nor content."
            )
        return ChatResult(
            text=text,
            tool_call=None,
            raw=body,
            tokens_in=int(body.get("prompt_eval_count") or 0),
            tokens_out=int(body.get("eval_count") or 0),
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------

    def chat_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        max_tokens: int = 280,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> str:
        """Vision path for Ollama multimodal tags (gemma4, llava, …).

        Talks to ``POST /api/chat`` with ``messages=[{role:user,
        content:<prompt>, images:[<base64>]}]``. The response text is
        returned as-is; the caller (GemmaVisionProvider) parses it into
        caption + details.
        """
        if not self.capabilities.get("vision"):
            raise LlmChatError(
                f"Ollama model {self._model!r} does not advertise vision "
                "capability. Pull a multimodal tag (e.g. gemma4:e2b, "
                "llava:latest) or pick a different backend."
            )
        import base64 as _b64

        b64 = _b64.standard_b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        body = self._post("/api/chat", payload, timeout_s=timeout_s)
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LlmChatError("Ollama vision returned an empty message content.")
        return content

    # ------------------------------------------------------------------
    # helpers

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: Optional[str],
        enable_thinking: bool,
    ) -> list[dict]:
        """Mutate messages for Ollama: enforce a system message, thread
        the schema reminder into the last system turn, and prepend the
        ``<|think|>`` marker when the caller asked for thinking mode."""
        out: list[dict] = [{"role": m.role, "content": m.content} for m in messages]
        if not out:
            raise LlmChatError("messages must not be empty")

        if schema_hint:
            reminder = (
                f"Respond ONLY with a JSON object matching: {schema_hint}"
            )
            last_system_idx = _last_system_index(out)
            if last_system_idx is None:
                out.insert(0, {"role": "system", "content": reminder})
            else:
                out[last_system_idx]["content"] = (
                    out[last_system_idx]["content"].rstrip() + "\n" + reminder
                )

        if enable_thinking and self.capabilities.get("thinking"):
            first_system_idx = _first_system_index(out)
            if first_system_idx is None:
                out.insert(0, {"role": "system", "content": _THINKING_PREFIX})
            else:
                out[first_system_idx]["content"] = (
                    _THINKING_PREFIX + out[first_system_idx]["content"]
                )

        return out

    def _post(self, path: str, payload: dict, *, timeout_s: float) -> dict:
        """POST to Ollama and return the decoded body, wrapping every
        known failure class in LlmChatError."""
        import httpx

        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as ex:
            raise LlmChatError(
                f"Ollama timed out after {timeout_s}s at {url}: {ex}"
            ) from ex
        except httpx.HTTPStatusError as ex:
            raise LlmChatError(
                f"Ollama HTTP {ex.response.status_code} at {url}: "
                f"{ex.response.text[:256]}"
            ) from ex
        except httpx.HTTPError as ex:
            raise LlmChatError(
                f"Ollama network error at {url}: {ex}"
            ) from ex
        except json.JSONDecodeError as ex:
            raise LlmChatError(
                f"Ollama returned non-JSON body at {url}: {ex}"
            ) from ex


def _first_system_index(messages: list[dict]) -> Optional[int]:
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            return i
    return None


def _last_system_index(messages: list[dict]) -> Optional[int]:
    idx: Optional[int] = None
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            idx = i
    return idx


def _tool_spec_to_ollama(tool: ChatToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _coerce_arguments(raw: Any) -> dict:
    """Ollama usually returns tool-call arguments as a dict already but
    some builds emit a JSON string — accept both. Anything else becomes
    an empty dict (caller treats missing args as a no-op call)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


# ---------------------------------------------------------------------------
# Claude — remote Anthropic backend
# ---------------------------------------------------------------------------


_CLAUDE_JSON_SYSTEM_SUFFIX = (
    "You MUST respond with a single JSON object and nothing else. "
    "Do not wrap the JSON in markdown fences. Do not add commentary "
    "before or after the JSON."
)


class ClaudeChatProvider(LlmChatProvider):
    """Anthropic Claude chat provider.

    Wraps an already-constructed ``anthropic.Anthropic`` client so tests
    can inject a MagicMock without touching network. Native tool use and
    native extended thinking are both advertised; JSON mode is simulated
    via a stern system prompt because Anthropic does not (yet) expose a
    ``response_format=json`` knob.
    """

    def __init__(
        self,
        *,
        anthropic_client: Any,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = anthropic_client
        self._model = model

    @property
    def name(self) -> str:
        return f"claude-{self._model}"

    @property
    def capabilities(self) -> dict:
        return {
            "function_calling": True,
            "thinking": True,
            "json_mode": True,
            "vision": True,
            "audio": False,
        }

    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> dict:
        system_prompt, user_messages = self._split_system_and_messages(
            messages, schema_hint=schema_hint,
        )
        kwargs: dict = {
            "model": self._model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system": system_prompt,
            "messages": user_messages,
        }
        if enable_thinking and self.capabilities.get("thinking"):
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": max(512, max_tokens)}

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as ex:
            raise LlmChatError(
                f"Claude chat_json failed: {type(ex).__name__}: {ex}"
            ) from ex

        text = _extract_claude_text(response)
        if not text.strip():
            raise LlmChatError("Claude chat_json returned empty content.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as ex:
            raise LlmChatError(
                f"Claude chat_json returned non-JSON content: {ex}"
            ) from ex
        if not isinstance(parsed, dict):
            raise LlmChatError(
                f"Claude chat_json response must be an object; got {type(parsed).__name__}."
            )
        return parsed

    # ------------------------------------------------------------------

    def chat_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatToolSpec],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 15.0,
        enable_thinking: bool = False,
    ) -> ChatResult:
        system_prompt, user_messages = self._split_system_and_messages(
            messages, schema_hint=None,
        )
        kwargs: dict = {
            "model": self._model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system": system_prompt,
            "messages": user_messages,
            "tools": [_tool_spec_to_claude(t) for t in tools],
        }
        if enable_thinking and self.capabilities.get("thinking"):
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": max(512, max_tokens)}

        started_at = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as ex:
            raise LlmChatError(
                f"Claude chat_tools failed: {type(ex).__name__}: {ex}"
            ) from ex
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        tool_use = _extract_claude_tool_use(response)
        text = _extract_claude_text(response)
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        raw = _response_to_dict(response)

        if tool_use is not None:
            return ChatResult(
                text=text or None,
                tool_call={
                    "name": tool_use.get("name", ""),
                    "arguments": tool_use.get("input") or {},
                },
                raw=raw,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=elapsed_ms,
            )

        if not text.strip():
            raise LlmChatError(
                "Claude chat_tools returned neither tool_use nor text."
            )
        return ChatResult(
            text=text,
            tool_call=None,
            raw=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # helpers

    def _split_system_and_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_hint: Optional[str],
    ) -> tuple[str, list[dict]]:
        """Anthropic takes the system prompt as a top-level arg, not as
        a role='system' turn — collapse every system message into one
        string and drop it from the turns list."""
        if not messages:
            raise LlmChatError("messages must not be empty")
        system_parts: list[str] = []
        user_turns: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                user_turns.append({"role": m.role, "content": m.content})

        if schema_hint:
            system_parts.append(
                _CLAUDE_JSON_SYSTEM_SUFFIX
                + f"\nSchema: {schema_hint}"
            )

        system_prompt = "\n\n".join(p for p in system_parts if p)
        if not user_turns:
            raise LlmChatError(
                "messages must contain at least one user/assistant turn"
            )
        return system_prompt, user_turns


def _tool_spec_to_claude(tool: ChatToolSpec) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _extract_claude_text(response: Any) -> str:
    text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if t:
                text_parts.append(t)
        elif hasattr(block, "text"):
            t = getattr(block, "text", "")
            if t:
                text_parts.append(t)
    return "".join(text_parts).strip()


def _extract_claude_tool_use(response: Any) -> Optional[dict]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return {
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {},
            }
    return None


def _response_to_dict(response: Any) -> dict:
    """Best-effort dict view of an Anthropic response for ChatResult.raw.
    Never raises — ChatResult.raw is diagnostic, not load-bearing."""
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump()
    except Exception:  # pragma: no cover - SDK-version-specific
        pass
    try:
        if hasattr(response, "to_dict"):
            return response.to_dict()
    except Exception:  # pragma: no cover - SDK-version-specific
        pass
    return {"repr": repr(response)}
