"""ClaudeVisionProvider unit tests.

All tests mock the anthropic client — no real API calls are made.
Scenarios:
  1. Normal response parsing (caption + details extraction)
  2. 5xx retry (2 retries) then VLM_API_FAILED
  3. Timeout → VLM_TIMEOUT
  4. Empty response → VLM_BAD_RESPONSE
  5. Rate limit → VLM_RATE_LIMIT
  6. Korean response parsing
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake Anthropic SDK types — enough to exercise the provider without
# importing the real SDK (which may not be installed in CI).
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


# Build a fake anthropic module and install it for the entire test module.
# This must survive across all test functions so `import anthropic` inside
# the provider code picks up the fake.
_fake_anthropic_module = MagicMock()
_fake_anthropic_module.APITimeoutError = _FakeAPITimeoutError
_fake_anthropic_module.RateLimitError = _FakeRateLimitError
_fake_anthropic_module.APIStatusError = _FakeAPIStatusError

# Install permanently for this test module's lifetime.
_had_anthropic = "anthropic" in sys.modules
_orig_anthropic = sys.modules.get("anthropic")
sys.modules["anthropic"] = _fake_anthropic_module


def _make_provider(*, client_mock=None, model="claude-sonnet-4-6"):
    """Build a ClaudeVisionProvider with a mocked anthropic client."""
    from app.capabilities.multimodal.claude_vision import ClaudeVisionProvider

    provider = ClaudeVisionProvider.__new__(ClaudeVisionProvider)
    provider._model = model
    provider._timeout_seconds = 30.0
    if client_mock is None:
        client_mock = MagicMock()
    provider._client = client_mock
    return provider


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 1. Normal response — caption + bullet parsing.
# ---------------------------------------------------------------------------


def test_normal_response_parses_caption_and_details():
    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(
        content=[
            _FakeTextBlock(
                text=(
                    "A scanned invoice document showing line items and totals.\n"
                    "- The header reads 'Invoice #1024'\n"
                    "- Total amount: $129.95\n"
                    "- Date: 2026-04-15\n"
                    "- Logo in the top-left corner\n"
                    "(3) Visible text: 'Invoice #1024 Total: $129.95 Due: 2026-04-15'"
                )
            )
        ]
    )

    provider = _make_provider(client_mock=client)
    result = provider.describe_image(
        _png_bytes(),
        mime_type="image/png",
        hint="what is on this invoice?",
        page_number=1,
    )

    assert result.provider_name == "claude-vision-v1"
    assert result.caption  # non-empty
    assert "invoice" in result.caption.lower() or "scanned" in result.caption.lower()
    assert len(result.details) >= 3
    assert result.page_number == 1
    assert result.latency_ms >= 0
    assert result.warnings == []

    # Verify the API was called correctly.
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs.kwargs["temperature"] == 0
    assert call_kwargs.kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# 2. 5xx retry then failure.
# ---------------------------------------------------------------------------


def test_5xx_retries_twice_then_raises_vlm_api_failed():
    from app.capabilities.multimodal.vision_provider import VisionError

    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIStatusError(
        "Internal Server Error", status_code=500
    )

    provider = _make_provider(client_mock=client)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes(), mime_type="image/png")

    assert exc_info.value.code == "VLM_API_FAILED"
    # Should have been called 3 times (1 original + 2 retries).
    assert client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# 3. Timeout → VLM_TIMEOUT.
# ---------------------------------------------------------------------------


def test_timeout_raises_vlm_timeout():
    from app.capabilities.multimodal.vision_provider import VisionError

    client = MagicMock()
    client.messages.create.side_effect = _FakeAPITimeoutError("timed out")

    provider = _make_provider(client_mock=client)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes(), mime_type="image/png")

    assert exc_info.value.code == "VLM_TIMEOUT"
    assert client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# 4. Empty response → VLM_BAD_RESPONSE.
# ---------------------------------------------------------------------------


def test_empty_response_raises_vlm_bad_response():
    from app.capabilities.multimodal.vision_provider import VisionError

    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock(text="   ")]
    )

    provider = _make_provider(client_mock=client)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes(), mime_type="image/png")

    assert exc_info.value.code == "VLM_BAD_RESPONSE"


# ---------------------------------------------------------------------------
# 5. Rate limit → VLM_RATE_LIMIT.
# ---------------------------------------------------------------------------


def test_rate_limit_raises_vlm_rate_limit():
    from app.capabilities.multimodal.vision_provider import VisionError

    client = MagicMock()
    client.messages.create.side_effect = _FakeRateLimitError("rate limited")

    provider = _make_provider(client_mock=client)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes(), mime_type="image/png")

    assert exc_info.value.code == "VLM_RATE_LIMIT"
    # Rate limit is not retried — fails immediately.
    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# 6. Korean response parsing.
# ---------------------------------------------------------------------------


def test_korean_response_parses_correctly():
    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(
        content=[
            _FakeTextBlock(
                text=(
                    "한국어 텍스트가 포함된 문서 스캔 이미지입니다.\n"
                    "- 상단에 '공지사항' 제목이 표시되어 있습니다\n"
                    "- 본문에 시스템 점검 일정이 기재되어 있습니다\n"
                    "- 서명란이 하단에 위치합니다\n"
                    "(3) 추출된 텍스트: '공지사항 시스템 점검 안내'"
                )
            )
        ]
    )

    provider = _make_provider(client_mock=client)
    result = provider.describe_image(
        _png_bytes(),
        mime_type="image/png",
        hint="이 문서에 무엇이 있나요?",
        page_number=1,
    )

    assert result.provider_name == "claude-vision-v1"
    assert result.caption  # non-empty
    assert len(result.details) >= 2
    assert result.page_number == 1


# ---------------------------------------------------------------------------
# 7. 4xx error is not retried.
# ---------------------------------------------------------------------------


def test_4xx_error_not_retried():
    from app.capabilities.multimodal.vision_provider import VisionError

    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIStatusError(
        "Bad Request", status_code=400
    )

    provider = _make_provider(client_mock=client)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes(), mime_type="image/png")

    assert exc_info.value.code == "VLM_API_FAILED"
    # 4xx errors should NOT be retried.
    assert client.messages.create.call_count == 1
