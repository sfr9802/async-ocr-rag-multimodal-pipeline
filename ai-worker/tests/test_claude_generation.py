"""ClaudeGenerationProvider unit tests.

All tests mock the anthropic client — no real API calls are made.
Scenarios:
  1. Normal answer — 3-part markdown structure with citations
  2. Citation format [doc_id#section] present
  3. API failure + fallback=True → extractive result + warning log
  4. API failure + fallback=False → CapabilityError propagation
  5. Empty chunks → "no passages" message
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.rag.generation import RetrievedChunk


# ---------------------------------------------------------------------------
# Fake Anthropic SDK types.
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: List[_FakeTextBlock]
    model: str = "claude-sonnet-4-6"
    stop_reason: str = "end_turn"


class _FakeAPITimeoutError(Exception):
    pass


class _FakeRateLimitError(Exception):
    pass


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# Install fake anthropic module for the entire test module's lifetime.
_fake_anthropic_module = MagicMock()
_fake_anthropic_module.APITimeoutError = _FakeAPITimeoutError
_fake_anthropic_module.RateLimitError = _FakeRateLimitError
_fake_anthropic_module.APIStatusError = _FakeAPIStatusError
sys.modules["anthropic"] = _fake_anthropic_module


def _make_provider(*, client_mock=None, fallback_on_error=True):
    """Build a ClaudeGenerationProvider with a mocked anthropic client."""
    from app.capabilities.rag.claude_generation import ClaudeGenerationProvider

    provider = ClaudeGenerationProvider.__new__(ClaudeGenerationProvider)
    provider._model = "claude-sonnet-4-6"
    provider._timeout_seconds = 60.0
    provider._fallback_on_error = fallback_on_error
    if client_mock is None:
        client_mock = MagicMock()
    provider._client = client_mock

    # Build the extractive fallback.
    from app.capabilities.rag.generation import ExtractiveGenerator
    provider._extractive_fallback = ExtractiveGenerator()

    return provider


def _sample_chunks() -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            doc_id="anime-005",
            section="overview",
            text=(
                "The Harbor Cats is a lighthearted daily-life anime about "
                "the stray cats of a small fishing port."
            ),
            score=0.845,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            doc_id="anime-003",
            section="themes",
            text="Recurring themes include memory, loss, and the comfort of precise language.",
            score=0.720,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Normal answer — 3-part markdown structure.
# ---------------------------------------------------------------------------


def test_normal_response_produces_answer():
    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(
        content=[
            _FakeTextBlock(
                text=(
                    "**Short answer:** The Harbor Cats is a lighthearted "
                    "anime about stray cats in a fishing port [anime-005#overview].\n\n"
                    "**Supporting passages:**\n"
                    "1. [anime-005#overview] (score=0.845) The show follows "
                    "stray cats in a small fishing port.\n"
                    "2. [anime-003#themes] (score=0.720) Themes of memory and "
                    "language.\n\n"
                    "**Sources:** anime-003, anime-005"
                )
            )
        ]
    )

    provider = _make_provider(client_mock=client)
    result = provider.generate("what is the harbor cats about?", _sample_chunks())

    assert "anime-005" in result
    assert "harbor" in result.lower() or "Harbor" in result
    assert provider.name == "claude-generation-v1"


# ---------------------------------------------------------------------------
# 2. Citation format [doc_id#section] present.
# ---------------------------------------------------------------------------


def test_citations_present_in_response():
    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(
        content=[
            _FakeTextBlock(
                text=(
                    "**Short answer:** Based on [anime-005#overview], the anime "
                    "features stray cats.\n\n"
                    "**Supporting passages:**\n"
                    "1. [anime-005#overview] description of cats\n\n"
                    "**Sources:** anime-005"
                )
            )
        ]
    )

    provider = _make_provider(client_mock=client)
    result = provider.generate("cats?", _sample_chunks())

    assert "[anime-005#overview]" in result


# ---------------------------------------------------------------------------
# 3. API failure + fallback=True → extractive result + warning log.
# ---------------------------------------------------------------------------


def test_api_failure_with_fallback_returns_extractive_result(caplog):
    client = MagicMock()
    client.messages.create.side_effect = _FakeAPITimeoutError("timed out")

    provider = _make_provider(client_mock=client, fallback_on_error=True)

    with caplog.at_level(logging.WARNING):
        result = provider.generate("what is the harbor cats?", _sample_chunks())

    # Should get an extractive-style result (not empty).
    assert "anime-005" in result
    assert "**Query:**" in result or "**Short answer:**" in result
    # Warning was logged.
    assert any("falling back to extractive" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. API failure + fallback=False → CapabilityError propagation.
# ---------------------------------------------------------------------------


def test_api_failure_without_fallback_raises_capability_error():
    from app.capabilities.base import CapabilityError

    client = MagicMock()
    client.messages.create.side_effect = _FakeAPITimeoutError("timed out")

    provider = _make_provider(client_mock=client, fallback_on_error=False)

    with pytest.raises(CapabilityError) as exc_info:
        provider.generate("query?", _sample_chunks())

    assert exc_info.value.code == "GENERATION_API_FAILED"
    assert "fallback is disabled" in exc_info.value.message


# ---------------------------------------------------------------------------
# 5. Empty chunks → "no passages" message.
# ---------------------------------------------------------------------------


def test_empty_chunks_returns_no_passages_message():
    provider = _make_provider()
    result = provider.generate("anything?", [])

    assert "No relevant passages" in result


# ---------------------------------------------------------------------------
# 6. Korean query formats correctly.
# ---------------------------------------------------------------------------


def test_korean_query_formats_user_message_correctly():
    from app.capabilities.rag.claude_generation import _build_user_message

    chunks = [
        RetrievedChunk(
            chunk_id="c-1", doc_id="kr-002", section="password_policy",
            text="비밀번호는 최소 12자 이상이며...", score=0.9,
        ),
    ]
    msg = _build_user_message("비밀번호 정책은?", chunks)

    assert "질문: 비밀번호 정책은?" in msg
    assert "관련 자료:" in msg
    assert "kr-002#password_policy" in msg
    assert "score=0.900" in msg
