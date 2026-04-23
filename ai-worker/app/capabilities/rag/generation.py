"""GenerationProvider contract + extractive implementation.

Phase 2 ships a single, deliberately non-mock generator: the extractive
provider. Given a query and a list of retrieved chunks, it produces a
structured grounded answer whose content is directly derived from the
retrieved text. Different queries + different retrieval results produce
different answers — this is NOT a mock that echoes its input.

When a real LLM is wired in a later phase, swap this module out behind
the same interface and nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    score: float
    rerank_score: Optional[float] = None


class GenerationProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        ...


def _chunk_relevance(chunk: RetrievedChunk) -> Optional[float]:
    """Prefer rerank_score when present; fall back to bi-encoder score."""
    if chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    if chunk.score is not None:
        return float(chunk.score)
    return None


class ExtractiveGenerator(GenerationProvider):
    """Builds a grounded markdown answer from the top retrieved chunks.

    The output has three parts:
      1. A short answer derived from the highest-scoring chunk's most
         query-relevant sentence.
      2. A numbered "Supporting passages" list with each chunk's score,
         document id, section, and a truncated excerpt.
      3. A "Sources" footer listing distinct doc ids.

    The "short answer" sentence is picked by a simple query-overlap
    heuristic: among the sentences in the top chunk, pick the one whose
    lowercased-token set maximally overlaps with the query's lowercased
    token set. This is deterministic, query-aware, and requires no model
    inference — but it DOES actually consume the retrieval output, which
    is the phase-2 acceptance bar.

    When ``low_relevance_threshold`` is set (default ``None`` = disabled),
    the generator checks the rerank_score or bi-encoder score of the top
    chunk. If it falls below the threshold the generator emits a Korean
    + English refusal text rather than dumping a low-confidence top chunk
    as if it were the answer. Phase 9 uses this to make the extractive
    path pass the cross-domain unanswerable gate — a query whose correct
    answer is "문서에서 찾을 수 없습니다" must not produce a confident-
    looking summary of whatever the filter allowed through.
    """

    def __init__(
        self,
        *,
        excerpt_chars: int = 400,
        low_relevance_threshold: Optional[float] = None,
    ) -> None:
        self._excerpt_chars = int(excerpt_chars)
        self._low_relevance_threshold = (
            float(low_relevance_threshold)
            if low_relevance_threshold is not None
            else None
        )

    @property
    def name(self) -> str:
        return "extractive-v1"

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "문서에서 찾을 수 없습니다.\n"
                "No relevant passages were retrieved for your query.\n\n"
                f"> {query}"
            )

        # Relevance gate: if the top chunk's score sits below the
        # threshold every chunk below it is also too weak to ground an
        # answer, so we emit a refusal instead of pretending.
        if self._low_relevance_threshold is not None:
            top_score = _chunk_relevance(chunks[0])
            if top_score is not None and top_score < self._low_relevance_threshold:
                return (
                    "문서에서 관련 정보를 찾을 수 없습니다.\n"
                    "No relevant passages were retrieved for your query.\n\n"
                    f"> {query}\n"
                    f"(top score={top_score:.3f} < threshold="
                    f"{self._low_relevance_threshold:.3f})"
                )

        short_answer = self._pick_answer_sentence(query, chunks[0].text)

        lines: list[str] = []
        lines.append(f"**Query:** {query}")
        lines.append("")
        lines.append(f"**Short answer:** {short_answer}")
        lines.append("")
        lines.append("**Supporting passages:**")
        for i, c in enumerate(chunks, start=1):
            excerpt = c.text.strip().replace("\n", " ")
            if len(excerpt) > self._excerpt_chars:
                excerpt = excerpt[: self._excerpt_chars - 3] + "..."
            lines.append(
                f"{i}. [{c.doc_id}#{c.section}] (score={c.score:.3f}) {excerpt}"
            )
        lines.append("")
        lines.append("**Sources:** " + ", ".join(sorted({c.doc_id for c in chunks})))
        return "\n".join(lines)

    # --------------------------------------------------------------

    @staticmethod
    def _pick_answer_sentence(query: str, passage: str) -> str:
        if not passage.strip():
            return "(no passage text)"
        q_tokens = {t for t in query.lower().split() if len(t) > 2}
        sentences = [
            s.strip()
            for s in passage.replace("\n", " ").split(".")
            if s.strip()
        ]
        if not sentences:
            return passage.strip()[:240]
        if not q_tokens:
            return sentences[0][:240] + ("." if not sentences[0].endswith(".") else "")
        best, best_score = sentences[0], -1
        for s in sentences:
            s_tokens = {t for t in s.lower().split() if len(t) > 2}
            overlap = len(q_tokens & s_tokens)
            if overlap > best_score:
                best_score = overlap
                best = s
        answer = best.strip()
        if len(answer) > 240:
            answer = answer[:237] + "..."
        if not answer.endswith("."):
            answer += "."
        return answer
