"""AGENT / AUTO capability — single-pass dispatcher + iterative agent loop.

Given an unknown ``(text, file)`` pair, this capability:

  1. Classifies the job inputs into ``(text, file_bytes, file_mime,
     filename)``.
  2. Asks an ``AgentRouterProvider`` for an ``AgentDecision``.
  3. Either:
     * delegates to the matching sub-capability via the standard
       ``Capability.run()`` interface (Phase 5 single-pass behaviour,
       preserved bit-for-bit when ``loop_enabled=False``), OR
     * for retrieval-grounded actions (``rag`` / ``multimodal``) with
       the loop enabled, drives an iterative critic / rewriter /
       retriever cycle via ``AgentLoopController`` and synthesizes the
       final answer on the UNION of every iteration's retrieved chunks.
  4. Emits either the Phase-5 artifact shape (``AGENT_DECISION`` + sub
     outputs) or the Phase-6 loop shape (``AGENT_DECISION`` +
     ``AGENT_TRACE`` + ``RETRIEVAL_RESULT_AGG`` + ``FINAL_RESPONSE``).

Two public classes ship here:

  * ``AgentCapability`` — registered under the name ``AGENT``. The loop
    runs only when ``loop_enabled=True``; otherwise it degenerates to
    the same dispatcher ``AutoCapability`` shipped in Phase 5.

  * ``AutoCapability`` — registered under the name ``AUTO``. A thin
    subclass that forces ``loop_enabled=False`` so the Phase 5
    contract keeps running unchanged while the loop feature rolls
    out under a separate capability name.

Loop safety: any error inside the loop (critic misbehaviour, rewriter
hiccup, sub-capability raise) is caught at the capability boundary and
degrades to the best-so-far answer with a ``stop_reason`` that names
the failure — the job never fails because of a loop error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from app.capabilities.agent.critic import AgentCriticProvider, NoOpCritic
from app.capabilities.agent.loop import (
    AgentLoopController,
    ExecuteFn,
    LoopBudget,
    LoopOutcome,
)
from app.capabilities.agent.rewriter import (
    NoOpQueryRewriter,
    QueryRewriterProvider,
)
from app.capabilities.agent.router import (
    AgentDecision,
    AgentRouterProvider,
)
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.rag.generation import GenerationProvider, RetrievedChunk
from app.capabilities.rag.query_parser import ParsedQuery, QueryParserProvider
from app.capabilities.rag.retriever import RetrievalReport, Retriever

log = logging.getLogger(__name__)


# Default Korean clarify response for ambiguous inputs. The user
# population for this project is Korean-first; the message mirrors the
# tone of other worker-side user-facing strings.
_CLARIFY_MESSAGE = (
    "질문이 모호합니다. 어떤 문서나 주제에 대해 알고 싶으신가요?"
)

# Stable error codes exposed by Agent/AutoCapability. Matched against in
# tests + observability layers, so don't rename without bumping them
# in the architecture doc too.
_ERR_AUTO_NO_INPUT = "AUTO_NO_INPUT"
_ERR_AUTO_RAG_UNAVAILABLE = "AUTO_RAG_UNAVAILABLE"
_ERR_AUTO_OCR_UNAVAILABLE = "AUTO_OCR_UNAVAILABLE"
_ERR_AUTO_MULTIMODAL_UNAVAILABLE = "AUTO_MULTIMODAL_UNAVAILABLE"

# Max length of the text preview shown in route_classify log lines.
_LOG_PREVIEW_CHARS = 120


# Short system prompt used by ``_direct_answer`` when the router
# selects ``direct_answer`` and a chat backend is configured.
_DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You answer brief questions concisely. Keep the answer under 120 words. "
    "If the question cannot be answered without retrieval or external data, "
    "explain that in one sentence."
)


class AgentCapability(Capability):
    """AGENT / AUTO capability.

    The router, parser, critic, rewriter, synthesizer, and
    sub-capabilities are all injected so the registry controls every
    seam. Missing retrievers or generators only matter when the loop
    is actually used — ``loop_enabled=False`` keeps the Phase 5
    single-pass behaviour and never touches the loop seams.
    """

    name = "AGENT"

    def __init__(
        self,
        *,
        router: AgentRouterProvider,
        parser: QueryParserProvider,
        rag: Optional[Capability] = None,
        ocr: Optional[Capability] = None,
        multimodal: Optional[Capability] = None,
        chat: Optional[Any] = None,
        direct_answer_max_tokens: int = 512,
        loop_enabled: bool = False,
        critic: Optional[AgentCriticProvider] = None,
        rewriter: Optional[QueryRewriterProvider] = None,
        synthesizer: Optional[AgentSynthesizer] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[GenerationProvider] = None,
        budget: Optional[LoopBudget] = None,
    ) -> None:
        self._router = router
        self._parser = parser
        self._rag = rag
        self._ocr = ocr
        self._multimodal = multimodal
        self._chat = chat
        self._direct_answer_max_tokens = int(direct_answer_max_tokens)
        self._loop_enabled = bool(loop_enabled)
        self._critic = critic or NoOpCritic()
        self._rewriter = rewriter or NoOpQueryRewriter()
        self._synthesizer = synthesizer
        self._retriever = retriever
        self._generator = generator
        self._budget = budget or LoopBudget()

    # ------------------------------------------------------------------

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        started = time.monotonic()
        text, file_bytes, file_mime, filename = _classify_input(input.inputs)

        if not text and not file_bytes:
            raise CapabilityError(
                _ERR_AUTO_NO_INPUT,
                "AUTO job has no usable text and no file — submit either a "
                "non-blank text field, a supported file (PNG/JPEG/PDF), or "
                "both.",
            )

        classify_ms = _elapsed_ms(started)
        log.info(
            "%s route_classify jobId=%s textLen=%d textPreview=%r "
            "hasFile=%s fileMime=%s fileSize=%d filename=%s classifyMs=%.3f",
            self.name,
            input.job_id,
            len(text or ""),
            (text or "")[:_LOG_PREVIEW_CHARS],
            bool(file_bytes),
            file_mime,
            len(file_bytes or b""),
            filename,
            classify_ms,
        )

        started = time.monotonic()
        decision = self._router.decide(
            text=text,
            has_file=bool(file_bytes),
            file_mime=file_mime,
            file_size=len(file_bytes or b""),
        )
        decide_ms = _elapsed_ms(started)
        log.info(
            "%s route_decide jobId=%s action=%s router=%s confidence=%.3f "
            "reason=%r decideMs=%.3f",
            self.name,
            input.job_id,
            decision.action,
            decision.router_name,
            decision.confidence,
            decision.reason,
            decide_ms,
        )

        agent_decision_artifact = CapabilityOutputArtifact(
            type="AGENT_DECISION",
            filename="agent-decision.json",
            content_type="application/json",
            content=_serialize_decision(decision).encode("utf-8"),
        )

        # -- Loop path: rag/multimodal with loop enabled -----------------
        if (
            self._loop_enabled
            and decision.action in ("rag", "multimodal")
            and self._retriever is not None
            and self._generator is not None
            and self._synthesizer is not None
        ):
            # Make sure the sub-capability for the chosen action exists.
            # Loop path reuses the multimodal cap for iter 0 so the OCR
            # + vision stages run once; missing sub still fails cleanly.
            if decision.action == "rag":
                self._require(
                    self._rag,
                    _ERR_AUTO_RAG_UNAVAILABLE,
                    "AGENT routed to RAG but that capability is not "
                    "registered on this worker. Enable it via "
                    "AIPIPELINE_WORKER_RAG_ENABLED=true and ensure the "
                    "FAISS index is built, then restart the worker.",
                )
            elif decision.action == "multimodal":
                self._require(
                    self._multimodal,
                    _ERR_AUTO_MULTIMODAL_UNAVAILABLE,
                    "AGENT routed to MULTIMODAL but that capability is not "
                    "registered on this worker. Enable it via "
                    "AIPIPELINE_WORKER_MULTIMODAL_ENABLED=true and ensure "
                    "its OCR + RAG dependencies are healthy, then restart "
                    "the worker.",
                )

            loop_outputs = self._run_loop_and_synthesize(
                decision=decision,
                input=input,
                text=text or "",
            )
            outputs = [agent_decision_artifact, *loop_outputs]
            log.info(
                "%s done jobId=%s action=%s router=%s loop=on outputs=%d",
                self.name, input.job_id, decision.action,
                decision.router_name, len(outputs),
            )
            return CapabilityOutput(outputs=outputs)

        # -- Phase 5 single-pass path ------------------------------------
        sub_outputs = self._dispatch_single_pass(
            decision=decision, input=input, text=text
        )

        outputs = [agent_decision_artifact, *sub_outputs]
        log.info(
            "%s done jobId=%s action=%s router=%s loop=off outputs=%d",
            self.name, input.job_id, decision.action,
            decision.router_name, len(outputs),
        )
        return CapabilityOutput(outputs=outputs)

    # ------------------------------------------------------------------
    # Phase 5 single-pass dispatcher — unchanged from Phase 3/5.
    # ------------------------------------------------------------------

    def _dispatch_single_pass(
        self,
        *,
        decision: AgentDecision,
        input: CapabilityInput,
        text: Optional[str],
    ) -> List[CapabilityOutputArtifact]:
        if decision.action == "multimodal":
            self._require(
                self._multimodal,
                _ERR_AUTO_MULTIMODAL_UNAVAILABLE,
                "AUTO routed to MULTIMODAL but that capability is not "
                "registered on this worker. Enable it via "
                "AIPIPELINE_WORKER_MULTIMODAL_ENABLED=true and ensure its "
                "OCR + RAG dependencies are healthy, then restart the worker.",
            )
            return list(self._multimodal.run(input).outputs)
        if decision.action == "ocr":
            self._require(
                self._ocr,
                _ERR_AUTO_OCR_UNAVAILABLE,
                "AUTO routed to OCR but that capability is not registered "
                "on this worker. Enable it via "
                "AIPIPELINE_WORKER_OCR_ENABLED=true and ensure Tesseract is "
                "installed, then restart the worker.",
            )
            return list(self._ocr.run(input).outputs)
        if decision.action == "rag":
            self._require(
                self._rag,
                _ERR_AUTO_RAG_UNAVAILABLE,
                "AUTO routed to RAG but that capability is not registered "
                "on this worker. Enable it via "
                "AIPIPELINE_WORKER_RAG_ENABLED=true and ensure the FAISS "
                "index is built, then restart the worker.",
            )
            return list(self._rag.run(input).outputs)
        if decision.action == "direct_answer":
            answer = self._direct_answer(text)
            return [
                CapabilityOutputArtifact(
                    type="FINAL_RESPONSE",
                    filename="auto-answer.md",
                    content_type="text/markdown; charset=utf-8",
                    content=answer.encode("utf-8"),
                )
            ]
        # clarify — inline short response; no sub-capability is invoked.
        return [
            CapabilityOutputArtifact(
                type="FINAL_RESPONSE",
                filename="auto-clarify.md",
                content_type="text/markdown; charset=utf-8",
                content=_CLARIFY_MESSAGE.encode("utf-8"),
            )
        ]

    # ------------------------------------------------------------------
    # Loop path
    # ------------------------------------------------------------------

    def _run_loop_and_synthesize(
        self,
        *,
        decision: AgentDecision,
        input: CapabilityInput,
        text: str,
    ) -> List[CapabilityOutputArtifact]:
        """Drive the critic / rewriter / retrieve loop + synthesize final answer.

        The loop's ``execute_fn`` is a closure that:
          * on iter 0, runs the MULTIMODAL sub-capability when the
            routed action is ``multimodal`` so OCR + vision + fusion
            happen exactly once, and otherwise runs retriever + generator
            directly;
          * on later iters, always runs retriever + generator directly
            with the rewriter's new query.

        Any unrecoverable execute error short-circuits the loop with a
        best-so-far outcome (iter_cap / time_cap / token_cap semantics
        still apply). The capability NEVER raises out of this method —
        a loop that couldn't make progress still returns AGENT_DECISION
        + AGENT_TRACE + RETRIEVAL_RESULT_AGG + FINAL_RESPONSE artifacts.
        """
        initial_parsed_query = (
            decision.parsed_query
            if decision.parsed_query is not None
            else self._parser.parse(text or "")
        )

        # Closure state: capture the multimodal sub's output artifacts
        # on iter 0 so we can still surface OCR_TEXT / VISION_RESULT /
        # MULTIMODAL_TRACE alongside the loop artifacts.
        mm_side_output: dict[str, Any] = {
            "outputs": [],
            "first_iter_seen": False,
        }

        def execute_fn(pq: ParsedQuery) -> Tuple[str, List[RetrievedChunk], int]:
            first_iter = not mm_side_output["first_iter_seen"]
            mm_side_output["first_iter_seen"] = True

            if decision.action == "multimodal" and first_iter:
                try:
                    mm_output = self._multimodal.run(input)  # type: ignore[union-attr]
                except Exception as ex:
                    # A hard multimodal failure at iter 0 stops the loop
                    # immediately — the loop controller treats this as a
                    # zero-step outcome and the capability falls back to
                    # the single-pass path.
                    raise _ExecuteFailure(
                        "multimodal iter 0 failed",
                        ex,
                    ) from ex
                mm_side_output["outputs"] = list(mm_output.outputs)
                answer = _extract_final_response(mm_output.outputs)
                chunks = _extract_retrieval_chunks(mm_output.outputs)
                return answer, chunks, 0

            # RAG path — or MULTIMODAL after iter 0. Retrieval + generation
            # directly, no sub-capability wrapping.
            if self._retriever is None or self._generator is None:
                raise _ExecuteFailure(
                    "loop missing retriever/generator",
                    RuntimeError(
                        "AGENT loop requires retriever + generator; "
                        "none configured"
                    ),
                )
            query_text = pq.normalized or pq.original or text
            try:
                report = self._retriever.retrieve(query_text)
                answer = self._generator.generate(text, list(report.results))
            except Exception as ex:
                raise _ExecuteFailure(
                    f"retrieve/generate failed at query={query_text!r}",
                    ex,
                ) from ex
            return answer, list(report.results), 0

        controller = AgentLoopController(
            critic=self._critic,
            rewriter=self._rewriter,
            parser=self._parser,
            budget=self._budget,
        )

        try:
            outcome = controller.run(
                question=text,
                initial_parsed_query=initial_parsed_query,
                execute_fn=execute_fn,
            )
        except _ExecuteFailure as fail:
            # Total wipe-out: iter 0 raised before any step was recorded.
            # Fall back to a synthetic outcome that stops immediately and
            # still emits all four loop artifacts so the client sees a
            # consistent shape.
            log.warning(
                "%s loop execute failed at iter 0 (%s: %s). "
                "Falling back to best-effort artifacts.",
                self.name,
                type(fail.cause).__name__,
                fail.cause,
            )
            outcome = LoopOutcome(
                steps=[],
                stop_reason="iter_cap",
                final_answer="",
                aggregated_chunks=[],
                total_ms=0.0,
                total_llm_tokens=0,
            )

        log.info(
            "%s loop stopReason=%s steps=%d aggregatedChunks=%d "
            "totalMs=%.1f totalTokens=%d",
            self.name,
            outcome.stop_reason,
            len(outcome.steps),
            len(outcome.aggregated_chunks),
            outcome.total_ms,
            outcome.total_llm_tokens,
        )

        # Synthesize the final answer across the union of aggregated chunks.
        final_answer = (
            self._synthesizer.synthesize(text, outcome)
            if self._synthesizer is not None
            else outcome.final_answer
        )

        outputs: List[CapabilityOutputArtifact] = []
        # Preserve mm-specific artifacts (OCR_TEXT, VISION_RESULT,
        # MULTIMODAL_TRACE). Drop RETRIEVAL_RESULT + FINAL_RESPONSE —
        # the loop replaces them with RETRIEVAL_RESULT_AGG + its own
        # FINAL_RESPONSE.
        for art in mm_side_output["outputs"]:
            if art.type in ("RETRIEVAL_RESULT", "FINAL_RESPONSE"):
                continue
            outputs.append(art)

        outputs.append(
            CapabilityOutputArtifact(
                type="AGENT_TRACE",
                filename="agent-trace.json",
                content_type="application/json",
                content=json.dumps(
                    {
                        "action": decision.action,
                        "budget": self._budget.to_dict(),
                        "outcome": outcome.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
        )
        outputs.append(
            CapabilityOutputArtifact(
                type="RETRIEVAL_RESULT_AGG",
                filename="agent-retrieval-agg.json",
                content_type="application/json",
                content=_aggregated_retrieval_payload(
                    outcome.aggregated_chunks
                ).encode("utf-8"),
            )
        )
        outputs.append(
            CapabilityOutputArtifact(
                type="FINAL_RESPONSE",
                filename="agent-answer.md",
                content_type="text/markdown; charset=utf-8",
                content=(final_answer or _CLARIFY_MESSAGE).encode("utf-8"),
            )
        )
        return outputs

    # ------------------------------------------------------------------

    @staticmethod
    def _require(
        sub: Optional[Capability],
        code: str,
        message: str,
    ) -> None:
        """Raise a typed error when a routed-to sub-capability is missing.

        Used at dispatch time so the failure is a clean, named error
        rather than an ``AttributeError`` on ``None.run()``."""
        if sub is None:
            raise CapabilityError(code, message)

    # ------------------------------------------------------------------

    def _direct_answer(self, text: Optional[str]) -> str:
        """Produce an inline direct answer via the shared chat backend.

        Only the LLM router ever selects ``direct_answer``. If the chat
        provider is unavailable or raises, we degrade to the clarify
        message so the client always gets a FINAL_RESPONSE.
        """
        from app.clients.llm_chat import ChatMessage, LlmChatError

        if self._chat is None or not (text or "").strip():
            return _CLARIFY_MESSAGE

        try:
            messages = [
                ChatMessage(role="system", content=_DIRECT_ANSWER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=text or ""),
            ]
            enable_thinking = bool(self._chat.capabilities.get("thinking"))
            data = self._chat.chat_json(
                messages,
                schema_hint='{"answer": string}',
                max_tokens=self._direct_answer_max_tokens,
                enable_thinking=enable_thinking,
            )
        except LlmChatError as ex:
            log.warning(
                "AUTO direct_answer: chat backend failed (%s) — "
                "falling back to clarify message",
                ex,
            )
            return _CLARIFY_MESSAGE

        raw_answer = data.get("answer") if isinstance(data, dict) else None
        if isinstance(raw_answer, str) and raw_answer.strip():
            return raw_answer.strip()
        log.warning(
            "AUTO direct_answer: chat returned no 'answer' field — "
            "falling back to clarify message"
        )
        return _CLARIFY_MESSAGE


class AutoCapability(AgentCapability):
    """Phase-5 ``AUTO`` capability — alias that forces ``loop_enabled=False``.

    Exists so the AUTO contract the Phase 5 clients rely on keeps
    running unchanged while the loop feature rolls out under the AGENT
    name. Any ``loop_enabled=True`` passed into this constructor is
    silently ignored — the whole point of AUTO is to be the single-pass
    fallback.
    """

    name = "AUTO"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("loop_enabled", None)
        super().__init__(loop_enabled=False, **kwargs)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _ExecuteFailure(Exception):
    """Internal marker raised when ``execute_fn`` hits an unrecoverable error."""

    def __init__(self, message: str, cause: BaseException) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class _ClassifiedInput:
    """Internal tuple-like helper — _classify_input returns the four
    fields directly rather than exposing this type."""

    text: Optional[str]
    file_bytes: Optional[bytes]
    file_mime: Optional[str]
    filename: Optional[str]


def _classify_input(
    artifacts: list[CapabilityInputArtifact],
) -> tuple[Optional[str], Optional[bytes], Optional[str], Optional[str]]:
    """Split a list of CapabilityInputArtifacts into (text, file_bytes,
    file_mime, filename).

    The first INPUT_TEXT wins; likewise the first INPUT_FILE. Any other
    artifact types are ignored. UTF-8 decoding failures on INPUT_TEXT
    are silently dropped (the next INPUT_TEXT artifact, or None).
    """
    text: Optional[str] = None
    file_bytes: Optional[bytes] = None
    file_mime: Optional[str] = None
    filename: Optional[str] = None

    for artifact in artifacts:
        if artifact.type == "INPUT_TEXT" and text is None:
            try:
                text = artifact.content.decode("utf-8").strip() or None
            except UnicodeDecodeError:
                text = None
            continue
        if artifact.type == "INPUT_FILE" and file_bytes is None:
            file_bytes = artifact.content or None
            file_mime = _normalize_mime(artifact.content_type)
            filename = artifact.filename

    return text, file_bytes, file_mime, filename


def _normalize_mime(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    head = raw.split(";")[0].strip().lower()
    return head or None


def _serialize_decision(decision: AgentDecision) -> str:
    """Render the AgentDecision as pretty-printed JSON for the artifact
    payload. Stable key order is a nice-to-have for tests that parse
    the body back out."""
    return json.dumps(decision.to_dict(), ensure_ascii=False, indent=2)


def _elapsed_ms(started_monotonic: float) -> float:
    return round((time.monotonic() - started_monotonic) * 1000.0, 3)


def _extract_final_response(
    outputs: List[CapabilityOutputArtifact],
) -> str:
    """Find the FINAL_RESPONSE artifact body as UTF-8 text.

    Returns an empty string when no FINAL_RESPONSE artifact is
    present — the caller handles the empty-answer case.
    """
    for art in outputs:
        if art.type == "FINAL_RESPONSE":
            try:
                return art.content.decode("utf-8")
            except UnicodeDecodeError:
                return ""
    return ""


def _extract_retrieval_chunks(
    outputs: List[CapabilityOutputArtifact],
) -> List[RetrievedChunk]:
    """Reconstruct ``RetrievedChunk`` objects from a RETRIEVAL_RESULT artifact.

    Used to pipe the multimodal first-iter output back into the loop's
    aggregated-chunk pool. A missing or malformed artifact simply yields
    an empty list — the loop continues with zero chunks for that iter.
    """
    for art in outputs:
        if art.type != "RETRIEVAL_RESULT":
            continue
        try:
            body = json.loads(art.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            return []
        chunks: List[RetrievedChunk] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            chunk_id = entry.get("chunkId") or ""
            doc_id = entry.get("docId") or ""
            if not chunk_id:
                continue
            try:
                score = float(entry.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    doc_id=str(doc_id),
                    section=str(entry.get("section") or ""),
                    text=str(entry.get("text") or ""),
                    score=score,
                )
            )
        return chunks
    return []


def _aggregated_retrieval_payload(
    chunks: List[RetrievedChunk],
) -> str:
    """Render the union-of-iters chunk list as RETRIEVAL_RESULT_AGG JSON.

    Mirrors the RAG capability's RETRIEVAL_RESULT schema so downstream
    consumers (clients, analytics) can reuse the same parsing path —
    the ``type`` field is the only thing that distinguishes the two.
    """
    body = {
        "aggregatedChunkCount": len(chunks),
        "results": [
            {
                "rank": i + 1,
                "chunkId": c.chunk_id,
                "docId": c.doc_id,
                "section": c.section,
                "score": round(float(c.score), 6),
                "text": c.text,
            }
            for i, c in enumerate(chunks)
        ],
    }
    return json.dumps(body, ensure_ascii=False, indent=2)
