"""QueryRewriterProvider seam + LLM implementation.

Given the original question, the previous iteration's answer preview,
the critic's gap_reason, and the chunks that were already retrieved,
the rewriter proposes a new retrieval query that should fetch
COMPLEMENTARY passages — the ones the previous query missed.

This seam is used by ``AgentLoopController.run()`` once the critic
declares the current answer insufficient. The rewriter returns a
``ParsedQuery`` (not just a string) so the retriever can feed it
straight into its usual ``normalized`` / ``keywords`` / ``rewrites``
pipeline without re-parsing.

Two providers ship here:

  * ``LlmQueryRewriter`` — wraps the shared ``LlmChatProvider`` and
    exploits Gemma 4 E2B's 128K context window to show the model the
    full text of every chunk retrieved in the previous iteration. The
    prompt explicitly asks for a query that finds DIFFERENT information.
    On ``LlmChatError``, the rewriter degrades to a deterministic
    fallback (prepend gap_reason, hand to parser) and stamps
    ``parser_name='rewriter-fallback'`` so the AGENT_TRACE surfaces
    the downgrade.

  * ``NoOpQueryRewriter`` — deterministic rewriter used when the loop
    is disabled or the LLM chat provider is unavailable. Returns
    ``parser.parse(original)`` unchanged, so the next iteration would
    re-run the same query — useful only as a contract-compatible
    placeholder. Production configurations use ``LlmQueryRewriter``.

The rewriter never raises — any failure falls back to the deterministic
parser-only path so the loop continues to make progress.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider

log = logging.getLogger(__name__)


# Defaults sized for the loop's budget: 300ch of prev-answer context +
# 10K chars of chunk text is plenty for the LLM to see what was already
# retrieved without paying for a full 128K prompt.
_DEFAULT_PREV_ANSWER_CHARS = 300
_DEFAULT_MAX_CONTEXT_CHARS = 10_000

# Max tokens the rewriter is allowed to spend on a single call. Higher
# than the critic because the rewriter actually has to write a fresh
# query instead of picking a letter.
_LLM_REWRITER_MAX_TOKENS = 384


# --------------------------------------------------------------------------
# Provider contract
# --------------------------------------------------------------------------


class QueryRewriterProvider(ABC):
    """Produces a new ``ParsedQuery`` aimed at filling a critic-identified gap."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def rewrite(
        self,
        *,
        original: str,
        prev_answer: str,
        gap_reason: str,
        already_retrieved_chunks: Sequence[RetrievedChunk],
        parser: QueryParserProvider,
    ) -> ParsedQuery:
        ...


# --------------------------------------------------------------------------
# NoOp rewriter — placeholder for loop-disabled / offline configs.
# --------------------------------------------------------------------------


class NoOpQueryRewriter(QueryRewriterProvider):
    """Rewriter that returns the original query, unchanged.

    Used only as a safety net — when the loop is disabled this
    rewriter is never invoked (the critic says sufficient at iter=0),
    and when the LLM chat backend is noop the registry picks the LLM
    rewriter anyway and its per-call fallback handles the downgrade.
    Keep this here so integration tests have a zero-side-effect
    rewriter to exercise the controller's "sufficient at iter=0"
    path without mocking an LLM.
    """

    @property
    def name(self) -> str:
        return "noop"

    def rewrite(
        self,
        *,
        original: str,
        prev_answer: str,
        gap_reason: str,
        already_retrieved_chunks: Sequence[RetrievedChunk],
        parser: QueryParserProvider,
    ) -> ParsedQuery:
        return parser.parse(original)


# --------------------------------------------------------------------------
# LLM rewriter — exploits 128K context + JSON mode.
# --------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = (
    "You propose an improved retrieval query when the previous attempt "
    "did not answer the user's question. The previous query's retrieved "
    "passages are shown; propose a new query that finds DIFFERENT "
    "information from what's shown above. Focus on the specific gap "
    "the critic identified. Respond ONLY with a JSON object."
)

_LLM_SCHEMA_HINT = '{"query": string (1-200 chars)}'

_LLM_QUERY_MIN_CHARS = 1
_LLM_QUERY_MAX_CHARS = 200


class LlmQueryRewriter(QueryRewriterProvider):
    """LLM-backed rewriter with a deterministic fallback.

    Uses ``chat_json`` (not tools) because the response is a single
    string and JSON mode is the cheaper path on every backend. Thinking
    mode is enabled iff the backend advertises it.

    Failure modes that fall back to the parser-only path instead of
    raising:

      * ``LlmChatError`` from the underlying provider (network, timeout,
        invalid JSON, empty response).
      * Empty / non-string / out-of-range ``query`` in the LLM's JSON
        response.

    The fallback path runs ``parser.parse(gap_reason + " " + original)``
    and returns a ``ParsedQuery`` with ``parser_name='rewriter-fallback'``
    so the trace distinguishes a clean-rewrite iteration from a
    degraded one.
    """

    def __init__(
        self,
        chat: Any,
        *,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
        prev_answer_preview_chars: int = _DEFAULT_PREV_ANSWER_CHARS,
        max_tokens: int = _LLM_REWRITER_MAX_TOKENS,
    ) -> None:
        from app.clients.llm_chat import LlmChatProvider  # local — avoids cycles

        if not isinstance(chat, LlmChatProvider):
            raise TypeError(
                "LlmQueryRewriter requires an LlmChatProvider instance; "
                f"got {type(chat).__name__}"
            )
        if max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be positive; "
                f"got {max_context_chars!r}"
            )
        if prev_answer_preview_chars < 0:
            raise ValueError(
                "prev_answer_preview_chars must be non-negative; "
                f"got {prev_answer_preview_chars!r}"
            )
        self._chat = chat
        self._max_context_chars = int(max_context_chars)
        self._prev_answer_preview_chars = int(prev_answer_preview_chars)
        self._max_tokens = int(max_tokens)

    @property
    def name(self) -> str:
        return f"llm-{self._chat.name}"

    # ------------------------------------------------------------------

    def rewrite(
        self,
        *,
        original: str,
        prev_answer: str,
        gap_reason: str,
        already_retrieved_chunks: Sequence[RetrievedChunk],
        parser: QueryParserProvider,
    ) -> ParsedQuery:
        from app.clients.llm_chat import ChatMessage, LlmChatError

        user_content = _build_rewriter_user_content(
            original=original,
            prev_answer=prev_answer,
            gap_reason=gap_reason,
            already_retrieved_chunks=already_retrieved_chunks,
            max_context_chars=self._max_context_chars,
            prev_answer_preview_chars=self._prev_answer_preview_chars,
        )
        messages = [
            ChatMessage(role="system", content=_LLM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]
        enable_thinking = bool(self._chat.capabilities.get("thinking"))

        try:
            data = self._chat.chat_json(
                messages,
                schema_hint=_LLM_SCHEMA_HINT,
                max_tokens=self._max_tokens,
                enable_thinking=enable_thinking,
            )
        except LlmChatError as ex:
            log.warning(
                "LlmQueryRewriter: provider failure, falling back (%s)",
                ex,
            )
            return _fallback_parse(
                parser=parser,
                original=original,
                gap_reason=gap_reason,
            )

        raw_query = data.get("query") if isinstance(data, dict) else None
        if not isinstance(raw_query, str):
            log.warning(
                "LlmQueryRewriter: response missing 'query' string, "
                "falling back (got %r)",
                type(raw_query).__name__,
            )
            return _fallback_parse(
                parser=parser,
                original=original,
                gap_reason=gap_reason,
            )
        query = raw_query.strip()
        if len(query) < _LLM_QUERY_MIN_CHARS or len(query) > _LLM_QUERY_MAX_CHARS:
            log.warning(
                "LlmQueryRewriter: 'query' out of range (%d chars), "
                "falling back",
                len(query),
            )
            return _fallback_parse(
                parser=parser,
                original=original,
                gap_reason=gap_reason,
            )

        # Feed the LLM-proposed query through the same parser the
        # retriever uses so ``normalized`` / ``keywords`` / ``rewrites``
        # stay consistent with the non-rewritten path.
        return parser.parse(query)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_rewriter_user_content(
    *,
    original: str,
    prev_answer: str,
    gap_reason: str,
    already_retrieved_chunks: Sequence[RetrievedChunk],
    max_context_chars: int,
    prev_answer_preview_chars: int,
) -> str:
    """Render the user turn for the LLM rewriter.

    Clips the prev_answer preview and packs as many of the
    already-retrieved chunks as fit inside ``max_context_chars`` in
    order. A chunk that would push the running total over the budget is
    truncated; subsequent chunks are dropped. This keeps the prompt
    bounded even when the retriever returned 30 chunks.
    """
    prev_answer_preview = (prev_answer or "").strip()
    if (
        prev_answer_preview_chars > 0
        and len(prev_answer_preview) > prev_answer_preview_chars
    ):
        prev_answer_preview = (
            prev_answer_preview[: prev_answer_preview_chars - 3] + "..."
        )

    chunk_blocks: List[str] = []
    remaining = max_context_chars
    for i, chunk in enumerate(already_retrieved_chunks, start=1):
        if remaining <= 0:
            break
        header = f"[{i}] doc={chunk.doc_id} section={chunk.section}\n"
        header_cost = len(header)
        if header_cost >= remaining:
            break
        body_budget = remaining - header_cost
        text = (chunk.text or "").strip()
        if len(text) > body_budget:
            if body_budget < 10:
                break
            text = text[: body_budget - 3] + "..."
        chunk_blocks.append(header + text)
        remaining -= header_cost + len(text)

    chunks_rendered = (
        "\n\n".join(chunk_blocks)
        if chunk_blocks
        else "(no passages were retrieved in the previous iteration)"
    )

    return (
        f"Original question:\n{original}\n\n"
        f"Previous answer preview:\n{prev_answer_preview or '(empty)'}\n\n"
        f"Critic gap reason:\n{gap_reason}\n\n"
        f"Passages ALREADY retrieved (propose a query that finds "
        f"DIFFERENT information from these):\n{chunks_rendered}"
    )


def _fallback_parse(
    *,
    parser: QueryParserProvider,
    original: str,
    gap_reason: str,
) -> ParsedQuery:
    """Deterministic rewriter fallback.

    Prepends the gap_reason so the retriever has SOMETHING extra to
    grip onto, then runs the standard parser. The returned
    ``ParsedQuery`` is re-wrapped with ``parser_name='rewriter-fallback'``
    so the trace distinguishes this degraded path from a clean
    LLM-written rewrite.
    """
    combined = f"{gap_reason.strip()} {original.strip()}".strip() or original
    base = parser.parse(combined)
    return ParsedQuery(
        original=base.original,
        normalized=base.normalized,
        keywords=base.keywords,
        intent=base.intent,
        rewrites=base.rewrites,
        filters=base.filters,
        parser_name="rewriter-fallback",
    )
