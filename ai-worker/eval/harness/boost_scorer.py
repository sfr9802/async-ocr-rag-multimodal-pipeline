"""Metadata-based candidate boost scorer for Phase 2B.

Adds a small, configurable boost to a chunk's dense bi-encoder score
when the chunk's document title or section name matches the query.
The intent is to pull good candidates higher within the existing
top-N candidate window without growing the candidate pool, so the
small final-top-K (5/10/15) windows recover more correct answers
without paying extra reranker latency.

Three weights (all ≥ 0; zero disables the corresponding signal):

  - ``title_exact_boost``    Doc title appears verbatim in the query.
  - ``title_partial_boost``  Any title token (≥ ``title_min_len``
                             chars) appears in the query.
  - ``section_keyword_boost``  Chunk's section name appears in the
                               query, AND that section name is not in
                               the excluded list.
  - ``section_path_boost``     Any of the doc's other section names
                               appears in the query (a weaker proxy
                               for section-level intent).

Safety:

  - ``max_boost`` clamps the per-chunk total so a maxed-out boost
    can't overwhelm a high-confidence dense score.
  - ``excluded_sections`` blocks boost on broad section names like
    "본문" or "요약" that appear on every document.
  - When all weights are zero (``BoostConfig.disabled()``), the
    scorer's contract is byte-identical to dense ordering: the
    reranker returns the input list capped at ``k`` with the original
    score field unchanged.

The reranker is also a ``RerankerProvider`` so it can plug into the
existing ``Retriever`` reranker slot. The eval harness uses it
standalone (not chained) so the Phase 2A topN sweep stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.reranker import RerankerProvider
from eval.harness.boost_metadata import DocBoostMetadata
from eval.harness.query_normalizer import (
    NormalizedQuery,
    normalize_for_match,
    normalize_query,
)


# Section names too broad to use as a boost signal. Both anime-namu
# corpora put a "본문" / "요약" on every doc, so matching either
# would promote essentially every doc whose query contains the word
# "summary" or "body" — which would defeat the point of the boost.
DEFAULT_EXCLUDED_SECTIONS: Tuple[str, ...] = (
    "본문",
    "요약",
)


@dataclass(frozen=True)
class BoostConfig:
    """Per-run boost knobs. All weights are non-negative additive boosts."""

    title_exact_boost: float = 0.0
    title_partial_boost: float = 0.0
    section_keyword_boost: float = 0.0
    section_path_boost: float = 0.0
    max_boost: float = 0.30
    title_min_len: int = 2
    excluded_sections: Tuple[str, ...] = DEFAULT_EXCLUDED_SECTIONS

    @classmethod
    def disabled(cls) -> "BoostConfig":
        """All-zero config — the scorer becomes a no-op."""
        return cls(
            title_exact_boost=0.0,
            title_partial_boost=0.0,
            section_keyword_boost=0.0,
            section_path_boost=0.0,
            max_boost=0.0,
            excluded_sections=DEFAULT_EXCLUDED_SECTIONS,
        )

    def is_disabled(self) -> bool:
        return (
            self.title_exact_boost == 0.0
            and self.title_partial_boost == 0.0
            and self.section_keyword_boost == 0.0
            and self.section_path_boost == 0.0
        )

    def validate(self) -> List[str]:
        """Return the list of validation errors; empty when valid."""
        errors: List[str] = []
        for name, value in (
            ("title_exact_boost", self.title_exact_boost),
            ("title_partial_boost", self.title_partial_boost),
            ("section_keyword_boost", self.section_keyword_boost),
            ("section_path_boost", self.section_path_boost),
            ("max_boost", self.max_boost),
        ):
            if value < 0:
                errors.append(f"{name} must be >= 0 (got {value})")
        if self.title_min_len < 1:
            errors.append(
                f"title_min_len must be >= 1 (got {self.title_min_len})"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title_exact_boost": self.title_exact_boost,
            "title_partial_boost": self.title_partial_boost,
            "section_keyword_boost": self.section_keyword_boost,
            "section_path_boost": self.section_path_boost,
            "max_boost": self.max_boost,
            "title_min_len": self.title_min_len,
            "excluded_sections": list(self.excluded_sections),
        }


@dataclass(frozen=True)
class BoostScore:
    """Per-chunk boost breakdown.

    ``total`` is the post-clamp value that's actually added to the
    dense score. The four component fields sum to the PRE-clamp
    raw total; the difference between ``sum(components)`` and
    ``total`` is the amount the ``max_boost`` clamp shaved off.
    """

    title_exact: float = 0.0
    title_partial: float = 0.0
    section_keyword: float = 0.0
    section_path: float = 0.0
    total: float = 0.0
    title_match_kind: Optional[str] = None
    matched_title: Optional[str] = None
    matched_section: Optional[str] = None

    @classmethod
    def empty(cls) -> "BoostScore":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title_exact": round(self.title_exact, 6),
            "title_partial": round(self.title_partial, 6),
            "section_keyword": round(self.section_keyword, 6),
            "section_path": round(self.section_path, 6),
            "total": round(self.total, 6),
            "title_match_kind": self.title_match_kind,
            "matched_title": self.matched_title,
            "matched_section": self.matched_section,
        }

    def has_any_match(self) -> bool:
        return (
            self.title_match_kind is not None
            or self.matched_section is not None
        )


def _excluded_set(excluded_sections: Tuple[str, ...]) -> set:
    """Pre-normalized set of excluded section names for cheap membership."""
    return {normalize_for_match(s) for s in excluded_sections if s}


def compute_boost_score(
    *,
    query_normalized: str,
    section: str,
    doc_meta: Optional[DocBoostMetadata],
    config: BoostConfig,
    excluded: Optional[set] = None,
) -> BoostScore:
    """Pure-function boost computation for a single chunk.

    The caller pre-normalizes the query once per query (since one
    query is scored against many chunks) and may pre-build the
    excluded-section set for the same reason.

    Returns ``BoostScore.empty()`` when the config is disabled or
    when ``doc_meta`` is missing — defensive against drift between
    the boost-metadata map and what the retriever surfaced.
    """
    if config.is_disabled():
        return BoostScore.empty()
    if doc_meta is None:
        return BoostScore.empty()
    if excluded is None:
        excluded = _excluded_set(config.excluded_sections)

    title_exact = 0.0
    title_partial = 0.0
    title_match_kind: Optional[str] = None
    matched_title: Optional[str] = None

    norm_title = doc_meta.normalized_title
    norm_seed = doc_meta.normalized_seed
    candidates = [t for t in (norm_title, norm_seed) if t]

    if config.title_exact_boost > 0:
        for cand in candidates:
            if cand and cand in query_normalized:
                title_exact = config.title_exact_boost
                title_match_kind = "exact"
                matched_title = cand
                break

    if title_match_kind is None and config.title_partial_boost > 0:
        partial_hit: Optional[str] = None
        for cand in candidates:
            for token in cand.split():
                if len(token) < config.title_min_len:
                    continue
                if token in query_normalized:
                    partial_hit = token
                    break
            if partial_hit:
                break
        if partial_hit:
            title_partial = config.title_partial_boost
            title_match_kind = "partial"
            matched_title = partial_hit

    section_keyword = 0.0
    section_path = 0.0
    matched_section: Optional[str] = None

    if config.section_keyword_boost > 0:
        norm_section = normalize_for_match(section)
        if (
            norm_section
            and norm_section not in excluded
            and norm_section in query_normalized
        ):
            section_keyword = config.section_keyword_boost
            matched_section = section

    if config.section_path_boost > 0 and doc_meta.normalized_section_names:
        for sname in doc_meta.normalized_section_names:
            if not sname or sname in excluded:
                continue
            if sname in query_normalized:
                section_path = config.section_path_boost
                if matched_section is None:
                    matched_section = sname
                break

    raw_total = title_exact + title_partial + section_keyword + section_path
    total = (
        min(raw_total, config.max_boost)
        if config.max_boost > 0
        else raw_total
    )
    return BoostScore(
        title_exact=title_exact,
        title_partial=title_partial,
        section_keyword=section_keyword,
        section_path=section_path,
        total=total,
        title_match_kind=title_match_kind,
        matched_title=matched_title,
        matched_section=matched_section,
    )


class MetadataBoostReranker(RerankerProvider):
    """Re-orders candidates by ``dense_score + metadata_boost``.

    Implements the ``RerankerProvider`` contract so it could plug into
    the production ``Retriever``'s reranker slot, but Phase 2B uses it
    standalone via the boost CLI / harness rather than mounting it on
    the live worker (we never modify production defaults in this phase).

    The boost stage is a pure REORDER — it never drops candidates
    beyond the requested ``k`` — and the rewritten ``RetrievedChunk``
    objects carry the composite ``final = dense + boost`` value in
    the ``score`` slot so downstream metrics treat the new ordering
    as the active ranking signal. The original dense score is
    recoverable from ``last_boost_breakdown[chunk_id]`` since
    ``dense = final - boost.total``.

    Off-mode contract: when ``config.is_disabled()`` is true the
    reranker returns the input list capped at ``k`` with chunks
    unchanged — score field, doc/section/text — so the boost-off
    path is byte-identical to the dense path.
    """

    def __init__(
        self,
        *,
        config: BoostConfig,
        doc_metadata: Mapping[str, DocBoostMetadata],
        title_token_extraction: bool = False,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError(
                "Invalid BoostConfig: " + "; ".join(errors)
            )
        self._config = config
        self._doc_meta = dict(doc_metadata)
        self._extract_titles = bool(title_token_extraction)
        self._excluded = _excluded_set(config.excluded_sections)
        self._last_boosts: Dict[str, BoostScore] = {}
        self._last_query: Optional[str] = None
        self._last_normalized: Optional[NormalizedQuery] = None

    @property
    def name(self) -> str:
        return "metadata_boost"

    @property
    def config(self) -> BoostConfig:
        return self._config

    @property
    def last_boost_breakdown(self) -> Dict[str, BoostScore]:
        """Per-chunk boost breakdown from the most recent ``rerank`` call."""
        return dict(self._last_boosts)

    @property
    def last_normalized_query(self) -> Optional[NormalizedQuery]:
        """The NormalizedQuery used for the most recent ``rerank`` call."""
        return self._last_normalized

    @property
    def doc_metadata(self) -> Mapping[str, DocBoostMetadata]:
        return self._doc_meta

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        norm = normalize_query(query, extract_titles=self._extract_titles)
        self._last_query = query
        self._last_normalized = norm
        self._last_boosts = {}

        if not chunks:
            return []

        kk = max(0, int(k))
        if self._config.is_disabled():
            # Off-mode: identical ordering to input. We still record an
            # empty BoostScore for each chunk so the eval dump shows
            # an explicit "no boost applied" entry rather than a
            # missing key.
            for chunk in chunks[:kk]:
                self._last_boosts[chunk.chunk_id] = BoostScore.empty()
            return chunks[:kk]

        scored: List[Tuple[RetrievedChunk, BoostScore, float]] = []
        for chunk in chunks:
            doc_meta = self._doc_meta.get(chunk.doc_id)
            boost = compute_boost_score(
                query_normalized=norm.normalized,
                section=chunk.section,
                doc_meta=doc_meta,
                config=self._config,
                excluded=self._excluded,
            )
            self._last_boosts[chunk.chunk_id] = boost
            final = float(chunk.score) + float(boost.total)
            scored.append((chunk, boost, final))

        # Stable sort by descending final score so ties preserve input
        # ordering — keeps the boost-zero path byte-identical to dense.
        scored.sort(key=lambda x: x[2], reverse=True)

        out: List[RetrievedChunk] = []
        for chunk, _boost, final in scored[:kk]:
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    text=chunk.text,
                    score=final,
                    rerank_score=chunk.rerank_score,
                )
            )
        return out
