"""QueryRewriterProvider tests.

Focus on ``LlmQueryRewriter`` — the LLM-backed rewriter that exploits
Gemma 4 E2B's 128K context window to show the previous iter's full
chunks. Covered:

  1. Happy path — mocked chat returns a new query; prompt contains the
     previous chunks clipped under max_context_chars.

  2. Fallback on ``LlmChatError`` — fallback prepends gap_reason and
     runs the parser; parser_name reflects the downgrade.

  3. Fallback on invalid payload — empty / missing / oversized
     ``query`` field all degrade.

Also a few smaller contract checks: NoOpQueryRewriter returns the
original query unchanged, bad constructor arguments raise.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from app.capabilities.agent.rewriter import (
    LlmQueryRewriter,
    NoOpQueryRewriter,
    QueryRewriterProvider,
)
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import (
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)
from app.clients.llm_chat import (
    ChatMessage,
    ChatResult,
    LlmChatError,
    LlmChatProvider,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _chunk(cid: str, text: str, doc_id: str = "doc-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc_id, section="intro", text=text, score=0.8,
    )


class _FakeChat(LlmChatProvider):
    """LlmChatProvider stub for LlmQueryRewriter tests."""

    def __init__(
        self,
        *,
        json_return: Optional[dict] = None,
        error: Optional[Exception] = None,
        capabilities: Optional[dict] = None,
        name: str = "fake-chat",
    ) -> None:
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
        self.chat_json_calls: list[dict] = []
        self.last_user_content: Optional[str] = None

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
        for m in messages:
            if m.role == "user":
                self.last_user_content = m.content
        self.chat_json_calls.append(
            {"schema_hint": schema_hint, "enable_thinking": enable_thinking}
        )
        if self._error is not None:
            raise self._error
        assert self._json_return is not None
        return self._json_return

    def chat_tools(self, *a, **kw) -> ChatResult:  # pragma: no cover - unused
        raise LlmChatError("unused in rewriter tests")


# ---------------------------------------------------------------------------
# 1. NoOpQueryRewriter — placeholder, returns the original verbatim.
# ---------------------------------------------------------------------------


def test_noop_rewriter_returns_parser_result_for_original():
    parser = RegexQueryParser()
    rewriter = NoOpQueryRewriter()
    pq = rewriter.rewrite(
        original="what is bge-m3",
        prev_answer="",
        gap_reason="something",
        already_retrieved_chunks=[],
        parser=parser,
    )
    assert pq.original == "what is bge-m3"
    assert pq.parser_name == "regex"


# ---------------------------------------------------------------------------
# 2. LlmQueryRewriter — happy path + chunk context windowing.
# ---------------------------------------------------------------------------


def test_llm_rewriter_happy_path_parses_new_query():
    chat = _FakeChat(json_return={"query": "what does FAISS index do"})
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)

    pq = rewriter.rewrite(
        original="tell me about the index",
        prev_answer="It is a library",
        gap_reason="missing facts about FAISS specifically",
        already_retrieved_chunks=[_chunk("c1", "index stores embeddings")],
        parser=parser,
    )
    # Parser runs on the LLM's new query, so original reflects that.
    assert pq.original == "what does FAISS index do"
    assert pq.parser_name == "regex"
    assert len(chat.chat_json_calls) == 1


def test_llm_rewriter_prompt_includes_previous_chunks():
    chat = _FakeChat(json_return={"query": "new query"})
    rewriter = LlmQueryRewriter(chat)
    parser = RegexQueryParser()
    chunks = [
        _chunk("c1", "Chunk one talks about A."),
        _chunk("c2", "Chunk two talks about B."),
    ]
    rewriter.rewrite(
        original="question",
        prev_answer="prev",
        gap_reason="need more info on Z",
        already_retrieved_chunks=chunks,
        parser=parser,
    )
    prompt = chat.last_user_content or ""
    # The prompt must reference every chunk and the gap reason.
    assert "Chunk one" in prompt
    assert "Chunk two" in prompt
    assert "need more info on Z" in prompt
    # And it must mention "DIFFERENT" so the model is pushed off the
    # already-retrieved information.
    assert "DIFFERENT" in prompt


def test_llm_rewriter_clips_chunks_to_max_context_chars():
    chat = _FakeChat(json_return={"query": "x"})
    parser = RegexQueryParser()
    # Tiny budget — only the first chunk can fit and must be clipped.
    rewriter = LlmQueryRewriter(chat, max_context_chars=80)
    chunks = [
        _chunk("c1", "A" * 200),  # way over budget
        _chunk("c2", "B" * 200),  # dropped
    ]
    rewriter.rewrite(
        original="q",
        prev_answer="",
        gap_reason="x",
        already_retrieved_chunks=chunks,
        parser=parser,
    )
    prompt = chat.last_user_content or ""
    # The prompt stays bounded: chunk 1's text is truncated; chunk 2
    # never appears at all because no budget remained.
    assert "c1" in prompt or "doc=doc-a" in prompt
    # chunk 2's "B" run must be absent — we trim before it lands.
    assert "B" * 200 not in prompt


def test_llm_rewriter_clips_prev_answer_to_preview():
    chat = _FakeChat(json_return={"query": "x"})
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat, prev_answer_preview_chars=20)
    rewriter.rewrite(
        original="q",
        prev_answer="A" * 500,
        gap_reason="z",
        already_retrieved_chunks=[],
        parser=parser,
    )
    prompt = chat.last_user_content or ""
    # 500-char prev_answer gets clipped to ~20 chars + ellipsis — the
    # prompt must NOT contain the untrimmed blob.
    assert "A" * 500 not in prompt
    assert prompt.count("A") < 500


# ---------------------------------------------------------------------------
# 3. LlmQueryRewriter — fallback paths.
# ---------------------------------------------------------------------------


def test_llm_rewriter_fallback_on_provider_error(caplog):
    chat = _FakeChat(error=LlmChatError("timeout"))
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)

    with caplog.at_level("WARNING"):
        pq = rewriter.rewrite(
            original="what is FAISS",
            prev_answer="",
            gap_reason="need more context",
            already_retrieved_chunks=[],
            parser=parser,
        )

    # Fallback prepends gap_reason to original and parses that. The
    # parser_name records the downgrade so the metrics layer can diff
    # clean runs from degraded ones.
    assert pq.parser_name == "rewriter-fallback"
    assert "need more context" in pq.original or "faiss" in pq.normalized.lower()
    assert any(
        "falling back" in rec.message.lower() for rec in caplog.records
    )


def test_llm_rewriter_fallback_when_query_missing():
    chat = _FakeChat(json_return={})  # no "query" field
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)

    pq = rewriter.rewrite(
        original="original q",
        prev_answer="",
        gap_reason="gap",
        already_retrieved_chunks=[],
        parser=parser,
    )
    assert pq.parser_name == "rewriter-fallback"


def test_llm_rewriter_fallback_when_query_blank():
    chat = _FakeChat(json_return={"query": "   "})
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)
    pq = rewriter.rewrite(
        original="o", prev_answer="", gap_reason="g",
        already_retrieved_chunks=[], parser=parser,
    )
    assert pq.parser_name == "rewriter-fallback"


def test_llm_rewriter_fallback_when_query_oversized():
    chat = _FakeChat(json_return={"query": "a" * 500})  # > 200 char cap
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)
    pq = rewriter.rewrite(
        original="o", prev_answer="", gap_reason="g",
        already_retrieved_chunks=[], parser=parser,
    )
    assert pq.parser_name == "rewriter-fallback"


def test_llm_rewriter_fallback_when_query_wrong_type():
    chat = _FakeChat(json_return={"query": 12345})
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)
    pq = rewriter.rewrite(
        original="o", prev_answer="", gap_reason="g",
        already_retrieved_chunks=[], parser=parser,
    )
    assert pq.parser_name == "rewriter-fallback"


# ---------------------------------------------------------------------------
# 4. Constructor guards
# ---------------------------------------------------------------------------


def test_llm_rewriter_rejects_non_chat_provider():
    with pytest.raises(TypeError):
        LlmQueryRewriter(object())


def test_llm_rewriter_rejects_bad_max_context_chars():
    chat = _FakeChat(json_return={"query": "x"})
    with pytest.raises(ValueError):
        LlmQueryRewriter(chat, max_context_chars=0)


def test_llm_rewriter_rejects_negative_preview_chars():
    chat = _FakeChat(json_return={"query": "x"})
    with pytest.raises(ValueError):
        LlmQueryRewriter(chat, prev_answer_preview_chars=-1)


def test_llm_rewriter_thinking_mode_propagates_when_backend_supports_it():
    chat = _FakeChat(json_return={"query": "new q"})
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)
    rewriter.rewrite(
        original="q", prev_answer="", gap_reason="g",
        already_retrieved_chunks=[], parser=parser,
    )
    # Default fake advertises thinking, so the rewriter passes it through.
    assert chat.chat_json_calls[0]["enable_thinking"] is True


def test_llm_rewriter_thinking_disabled_when_backend_lacks_it():
    chat = _FakeChat(
        json_return={"query": "new q"},
        capabilities={
            "function_calling": False,
            "thinking": False,
            "json_mode": True,
            "vision": False,
            "audio": False,
        },
    )
    parser = RegexQueryParser()
    rewriter = LlmQueryRewriter(chat)
    rewriter.rewrite(
        original="q", prev_answer="", gap_reason="g",
        already_retrieved_chunks=[], parser=parser,
    )
    assert chat.chat_json_calls[0]["enable_thinking"] is False
