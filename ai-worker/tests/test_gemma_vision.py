"""GemmaVisionProvider unit tests.

Everything runs offline. The chat provider is a MagicMock standing in
for a live LlmChatProvider; no Ollama server is contacted.

Scenario coverage:

  1. Happy path — mocked chat returns a multi-line response, provider
     splits it into caption + details and returns a
     VisionDescriptionResult with expected metadata.
  2. init raises when the chat provider does not advertise
     capabilities['vision']=True.
  3. Registry auto-downgrades to HeuristicVisionProvider (no raise)
     when vision_provider='gemma' but the LLM backend is NoOp.
  4. token_budget kwarg overrides the constructor default; the max_tokens
     passed to chat.chat_vision reflects the override.
  5. LlmChatError from chat.chat_vision becomes VisionError with code
     VLM_API_FAILED and a clean message.
  6. Korean response parses (caption + details extraction works on
     multi-byte text).
  7. Empty response → VisionError VLM_BAD_RESPONSE.
  8. hint='generic captioning' threads into the prompt as the subject.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.capabilities.multimodal.gemma_vision import GemmaVisionProvider
from app.capabilities.multimodal.vision_provider import VisionError
from app.clients.llm_chat import LlmChatError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _vision_chat(
    *,
    name: str = "ollama-gemma4:e2b",
    vision: bool = True,
    response: str = "A scanned invoice.",
) -> MagicMock:
    """Build a MagicMock LlmChatProvider with the required surface."""
    chat = MagicMock()
    chat.name = name
    chat.capabilities = {
        "function_calling": True,
        "thinking": True,
        "json_mode": True,
        "vision": vision,
        "audio": False,
    }
    chat.chat_vision.return_value = response
    return chat


def _png_bytes() -> bytes:
    # Minimal PNG-ish prefix — the provider does NOT decode bytes; the
    # chat backend owns that, and in tests the chat is mocked.
    return b"\x89PNG\r\n\x1a\n<not-a-real-png>"


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_parses_caption_and_details():
    chat = _vision_chat(
        response=(
            "A scanned invoice document with Korean and English text.\n"
            "- Header reads 'Invoice #1024'\n"
            "- Total: $129.95\n"
            "- Date stamp in top-right corner\n"
            "(3) Visible text: 'Invoice #1024 Total $129.95'"
        )
    )
    provider = GemmaVisionProvider(chat, default_token_budget=280)

    result = provider.describe_image(
        _png_bytes(),
        mime_type="image/png",
        hint="what is on this invoice?",
        page_number=2,
    )

    assert result.provider_name == "gemma-ollama-gemma4:e2b"
    assert result.page_number == 2
    assert "invoice" in result.caption.lower()
    assert len(result.details) >= 3
    assert result.warnings == []
    assert result.latency_ms >= 0

    # Verify chat.chat_vision was called with the right wiring.
    chat.chat_vision.assert_called_once()
    kwargs = chat.chat_vision.call_args.kwargs
    assert kwargs["image_bytes"] == _png_bytes()
    assert kwargs["mime_type"] == "image/png"
    assert kwargs["max_tokens"] == 280
    # Hint is embedded into the prompt as the subject.
    assert "invoice" in kwargs["prompt"].lower()


# ---------------------------------------------------------------------------
# 2. init rejects chat without vision capability
# ---------------------------------------------------------------------------


def test_init_rejects_chat_without_vision_capability():
    chat = _vision_chat(vision=False)
    with pytest.raises(ValueError) as exc_info:
        GemmaVisionProvider(chat)
    assert "vision" in str(exc_info.value).lower()


def test_init_rejects_non_positive_budget():
    chat = _vision_chat()
    with pytest.raises(ValueError):
        GemmaVisionProvider(chat, default_token_budget=0)
    with pytest.raises(ValueError):
        GemmaVisionProvider(chat, default_token_budget=-10)


# ---------------------------------------------------------------------------
# 3. Registry downgrades when chat backend is noop
# ---------------------------------------------------------------------------


def test_registry_downgrades_to_heuristic_when_chat_has_no_vision(monkeypatch):
    """vision_provider='gemma' + noop llm backend → HeuristicVisionProvider
    with a warning, instead of a registry-level crash."""
    from app.capabilities import registry as registry_mod
    from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider
    from app.core.config import WorkerSettings

    # Reset shared cache so our stubbed chat actually lands in the registry.
    registry_mod._shared_component_cache.clear()

    settings = WorkerSettings(
        multimodal_vision_provider="gemma",
        llm_backend="noop",
    )
    provider = registry_mod._build_vision_provider(settings)
    assert isinstance(provider, HeuristicVisionProvider)


def test_registry_builds_gemma_when_chat_has_vision(monkeypatch):
    from app.capabilities import registry as registry_mod
    from app.core.config import WorkerSettings

    registry_mod._shared_component_cache.clear()

    # Stub _get_shared_llm_chat to return a vision-capable mock so we don't
    # need a live Ollama.
    chat = _vision_chat()
    monkeypatch.setattr(
        registry_mod, "_get_shared_llm_chat", lambda _settings: chat
    )

    settings = WorkerSettings(
        multimodal_vision_provider="gemma",
        llm_backend="ollama",
        multimodal_gemma_token_budget=560,
    )
    provider = registry_mod._build_vision_provider(settings)

    assert isinstance(provider, GemmaVisionProvider)
    assert provider._default_token_budget == 560
    assert provider.name == "gemma-ollama-gemma4:e2b"


# ---------------------------------------------------------------------------
# 4. token_budget kwarg override
# ---------------------------------------------------------------------------


def test_token_budget_kwarg_overrides_default():
    chat = _vision_chat(response="A dense table scan.\n- cell A1\n- cell A2")
    provider = GemmaVisionProvider(chat, default_token_budget=140)

    provider.describe_image(_png_bytes(), token_budget=1120)

    kwargs = chat.chat_vision.call_args.kwargs
    assert kwargs["max_tokens"] == 1120


def test_token_budget_default_used_when_kwarg_absent():
    chat = _vision_chat(response="A photo.")
    provider = GemmaVisionProvider(chat, default_token_budget=560)

    provider.describe_image(_png_bytes())

    kwargs = chat.chat_vision.call_args.kwargs
    assert kwargs["max_tokens"] == 560


def test_token_budget_zero_kwarg_falls_through_to_default():
    chat = _vision_chat(response="A photo.")
    provider = GemmaVisionProvider(chat, default_token_budget=280)

    # Zero/None kwarg means "use default" — explicit budget must be
    # positive, but the "no override" signal passes through cleanly.
    provider.describe_image(_png_bytes(), token_budget=None)
    assert chat.chat_vision.call_args.kwargs["max_tokens"] == 280


# ---------------------------------------------------------------------------
# 5. LlmChatError → VisionError
# ---------------------------------------------------------------------------


def test_llm_chat_error_becomes_vision_error_api_failed():
    chat = _vision_chat()
    chat.chat_vision.side_effect = LlmChatError(
        "Ollama HTTP 503 at http://localhost:11434/api/chat: unavailable"
    )
    provider = GemmaVisionProvider(chat)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes())

    assert exc_info.value.code == "VLM_API_FAILED"
    # Clean, non-empty message that mentions the backend.
    assert "ollama-gemma4:e2b" in exc_info.value.message
    assert "503" in exc_info.value.message


# ---------------------------------------------------------------------------
# 6. Korean response
# ---------------------------------------------------------------------------


def test_korean_response_parses_correctly():
    chat = _vision_chat(
        response=(
            "한국어 텍스트가 포함된 문서 스캔 이미지입니다.\n"
            "- 상단에 '공지사항' 제목\n"
            "- 본문에 시스템 점검 일정\n"
            "(3) 추출된 텍스트: '공지사항 시스템 점검 안내'"
        )
    )
    provider = GemmaVisionProvider(chat)

    result = provider.describe_image(
        _png_bytes(), hint="이 문서에 무엇이 있나요?", page_number=1,
    )

    assert result.caption  # non-empty
    assert "한국어" in result.caption or "문서" in result.caption
    assert len(result.details) >= 2


# ---------------------------------------------------------------------------
# 7. Empty response
# ---------------------------------------------------------------------------


def test_empty_response_raises_vlm_bad_response():
    chat = _vision_chat(response="   \n  \n")
    provider = GemmaVisionProvider(chat)

    with pytest.raises(VisionError) as exc_info:
        provider.describe_image(_png_bytes())

    assert exc_info.value.code == "VLM_BAD_RESPONSE"


# ---------------------------------------------------------------------------
# 8. Hint threads into the prompt
# ---------------------------------------------------------------------------


def test_hint_threads_into_prompt_as_subject():
    chat = _vision_chat(response="A bar chart.\n- axes labeled Q1-Q4")
    provider = GemmaVisionProvider(chat)

    provider.describe_image(_png_bytes(), hint="revenue chart")

    prompt = chat.chat_vision.call_args.kwargs["prompt"]
    assert "revenue chart" in prompt


def test_hint_missing_uses_document_page_default():
    chat = _vision_chat(response="A page of text.")
    provider = GemmaVisionProvider(chat)

    provider.describe_image(_png_bytes(), hint=None)

    prompt = chat.chat_vision.call_args.kwargs["prompt"]
    assert "document page" in prompt


# ---------------------------------------------------------------------------
# 9. mime_type forwarded to chat_vision
# ---------------------------------------------------------------------------


def test_mime_type_forwarded_to_chat_vision():
    chat = _vision_chat(response="An image.")
    provider = GemmaVisionProvider(chat)

    provider.describe_image(_png_bytes(), mime_type="image/jpeg")
    assert chat.chat_vision.call_args.kwargs["mime_type"] == "image/jpeg"


def test_mime_type_missing_defaults_to_png():
    chat = _vision_chat(response="An image.")
    provider = GemmaVisionProvider(chat)

    provider.describe_image(_png_bytes(), mime_type=None)
    assert chat.chat_vision.call_args.kwargs["mime_type"] == "image/png"
