"""Retrieval service — query text → top-k chunks.

Holds four live objects:

  1. An embedding provider (to vectorize the query)
  2. A loaded FAISS index (to do nearest-neighbour search)
  3. A ragmeta store handle (to look up text + metadata for each hit)
  4. A reranker provider (cross-encoder by default; NoOp in CI/offline)

The retriever is built once at worker startup and used for every job.
The bi-encoder fetches ``candidate_k`` candidates from FAISS; the
reranker then re-scores those candidates with a cross-encoder and
returns the top ``top_k``. When the reranker is ``NoOpReranker``, the
bi-encoder's top-k is returned unchanged — bit-for-bit identical to the
pre-reranker Phase 0 baseline.

An optional MMR (Maximal Marginal Relevance) diversity pass composes
after the reranker. When ``use_mmr=True`` the reranker is asked for
its full candidate list (not truncated to top-k) so MMR has something
to diversify across; the MMR selector then picks the final top-k using
``value = lambda * relevance - (1 - lambda) * doc_id_penalty``, where
the penalty is 0.6 for candidates sharing a doc_id with any already-
selected chunk. ``use_mmr=False`` (default) reproduces Phase 1 exactly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.capabilities.rag.embeddings import EmbeddingProvider
from app.capabilities.rag.embedding_text_builder import (
    EMBEDDING_TEXT_BUILDER_VERSION,
    is_known_production_variant,
)
from app.capabilities.rag.faiss_index import FaissIndex, IndexBuildInfo
from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.ingest import load_ingest_manifest
from app.capabilities.rag.metadata_store import RagMetadataStore
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
)
from app.capabilities.rag.reranker import NoOpReranker, RerankerProvider

log = logging.getLogger(__name__)


_MMR_DOC_ID_PENALTY = 0.6
_DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RetrievalReport:
    query: str
    top_k: int
    index_version: str
    embedding_model: str
    results: List[RetrievedChunk]
    reranker_name: str = "noop"
    candidate_k: int = 0
    topk_gap: Optional[float] = None
    topk_rel_gap: Optional[float] = None
    use_mmr: bool = False
    mmr_lambda: Optional[float] = None
    dup_rate: float = 0.0
    parsed_query: Optional[ParsedQuery] = None
    filters: dict = field(default_factory=dict)
    filter_produced_no_docs: bool = False
    # Wall-clock latency (ms) of the rerank step in isolation. ``None``
    # for the NoOpReranker path so eval reports can distinguish "didn't
    # rerank" from "reranked in ~0 ms". Phase 2A retrieval-rerank eval
    # reads this to compute rerank_latency_avg_ms / p95.
    rerank_ms: Optional[float] = None
    # Phase 2A-L: wall-clock of the dense-retrieval part of retrieve()
    # (parser + filter resolution + FAISS search + metadata lookup +
    # any RRF merge), measured separately from the rerank step. ``None``
    # is unused — even the noop reranker path records the number; the
    # eval harness rolls it into a separate aggregate so the topN sweep
    # can attribute total query latency to FAISS-side vs rerank-side.
    dense_retrieval_ms: Optional[float] = None
    # Per-stage breakdown the CrossEncoderReranker recorded for this
    # call (pair_build / tokenize / forward / postprocess /
    # total_rerank). ``None`` for the NoOp path or when the reranker
    # ran into the OOM-retry fallback (see CrossEncoderReranker.last_
    # breakdown_ms for the contract). Phase 2A-L's per-row eval row
    # serialises this dict as-is so the aggregator can compute
    # percentile stats per stage without re-running retrieval.
    rerank_breakdown_ms: Optional[Dict[str, float]] = None


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        index: FaissIndex,
        metadata: RagMetadataStore,
        top_k: int,
        reranker: Optional[RerankerProvider] = None,
        candidate_k: Optional[int] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        query_parser: Optional[QueryParserProvider] = None,
        multi_query_rrf_k: int = _DEFAULT_RRF_K,
        embedding_text_variant: Optional[str] = None,
        embedding_text_builder_version: str = EMBEDDING_TEXT_BUILDER_VERSION,
        embedding_max_seq_length: Optional[int] = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._metadata = metadata
        self._top_k = int(top_k)
        self._reranker: RerankerProvider = reranker or NoOpReranker()
        # ``candidate_k`` controls how many bi-encoder candidates flow
        # into the reranker. None / non-positive collapses to top_k so a
        # NoOpReranker path reproduces the exact Phase 0 behaviour.
        if candidate_k is None or int(candidate_k) <= 0:
            self._candidate_k = self._top_k
        else:
            self._candidate_k = max(self._top_k, int(candidate_k))
        self._use_mmr = bool(use_mmr)
        # Clamp lambda into [0.0, 1.0]; values outside that range have
        # no meaningful interpretation and would make the selector pick
        # unstable winners on tied relevance.
        self._mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))
        self._parser: QueryParserProvider = query_parser or NoOpQueryParser()
        # RRF constant. Cormack et al. use 60; we expose it for sweeps.
        # Non-positive values would divide-by-zero / flip the sign, so
        # clamp to a sane floor the same way we clamp mmr_lambda.
        self._rrf_k = max(1, int(multi_query_rrf_k))
        self._info: IndexBuildInfo | None = None
        if embedding_text_variant is not None and not is_known_production_variant(
            embedding_text_variant,
        ):
            raise ValueError(
                f"Unknown embedding_text_variant {embedding_text_variant!r}"
            )
        self._embedding_text_variant = embedding_text_variant
        self._embedding_text_builder_version = embedding_text_builder_version
        self._embedding_max_seq_length = (
            int(embedding_max_seq_length)
            if embedding_max_seq_length is not None
            else None
        )

    def ensure_ready(self) -> None:
        """Load the index and strictly verify runtime model == build model.

        Two checks, in order:

          1. Model name equality. Mixing two different embedding models
             over the same FAISS index silently corrupts retrieval even
             when dimensions happen to agree, so we fail hard on any
             mismatch — no warning fallback.
          2. Dimension equality. Redundant in practice when (1) passes,
             but kept as a belt-and-suspenders guard against tampered
             build.json files or misbehaving embedder implementations.

        Both failures are raised as RuntimeError so the registry wraps
        them into a clean "RAG capability NOT registered" startup log
        without taking down the MOCK capability.
        """
        self._info = self._index.load()
        log.info(
            "Retriever readiness check: configured_model=%r index_model=%r "
            "configured_dim=%d index_dim=%d index_version=%s chunk_count=%d "
            "reranker=%s candidate_k=%d top_k=%d parser=%s rrf_k=%d",
            self._embedder.model_name,
            self._info.embedding_model,
            self._embedder.dimension,
            self._info.dimension,
            self._info.index_version,
            self._info.chunk_count,
            self._reranker.name,
            self._candidate_k,
            self._top_k,
            self._parser.name,
            self._rrf_k,
        )
        if self._embedder.model_name != self._info.embedding_model:
            raise RuntimeError(
                "Embedding MODEL mismatch: runtime embedder="
                f"{self._embedder.model_name!r} vs index build="
                f"{self._info.embedding_model!r}. "
                "When the embedding model changes, the FAISS index must be "
                "rebuilt AND the worker restarted — mixing models over the "
                "same index silently corrupts retrieval quality. "
                "Fix: either rebuild the index with the runtime model "
                "(`python -m scripts.build_rag_index --fixture`) or set "
                "AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL back to "
                f"{self._info.embedding_model!r}, then restart the worker."
            )
        if self._embedder.dimension != self._info.dimension:
            raise RuntimeError(
                "Embedding DIMENSION mismatch: runtime embedder dim="
                f"{self._embedder.dimension} vs index dim="
                f"{self._info.dimension}. Model names matched "
                f"({self._embedder.model_name!r}) so this almost always "
                "means build.json is stale or hand-edited. "
                "Rebuild the index (`python -m scripts.build_rag_index "
                "--fixture`) and restart the worker."
            )
        self._verify_ingest_manifest()
        log.info(
            "Retriever ready: model=%s dim=%d index_version=%s reranker=%s",
            self._embedder.model_name,
            self._embedder.dimension,
            self._info.index_version,
            self._reranker.name,
        )

    def _verify_ingest_manifest(self) -> None:
        if self._embedding_text_variant is None:
            return

        manifest = load_ingest_manifest(self._index.index_dir)
        if manifest is None:
            raise RuntimeError(
                "Ingest manifest missing: runtime embedding_text_variant="
                f"{self._embedding_text_variant!r}, but no "
                "ingest_manifest.json was found next to build.json. "
                "Rebuild the index with the current ingest path before "
                "serving RAG."
            )
        mismatches: list[str] = []
        if manifest.embedding_text_variant != self._embedding_text_variant:
            mismatches.append(
                "variant runtime="
                f"{self._embedding_text_variant!r} build="
                f"{manifest.embedding_text_variant!r}"
            )
        if (
            manifest.embedding_text_builder_version
            != self._embedding_text_builder_version
        ):
            mismatches.append(
                "builder_version runtime="
                f"{self._embedding_text_builder_version!r} build="
                f"{manifest.embedding_text_builder_version!r}"
            )
        if manifest.embedding_model != self._info.embedding_model:
            mismatches.append(
                "manifest_model="
                f"{manifest.embedding_model!r} build_json="
                f"{self._info.embedding_model!r}"
            )
        if manifest.dimension != self._info.dimension:
            mismatches.append(
                f"manifest_dim={manifest.dimension} build_json_dim={self._info.dimension}"
            )
        if manifest.chunk_count != self._info.chunk_count:
            mismatches.append(
                "manifest_chunks="
                f"{manifest.chunk_count} build_json_chunks={self._info.chunk_count}"
            )
        if (
            self._embedding_max_seq_length is not None
            and manifest.max_seq_length != self._embedding_max_seq_length
        ):
            mismatches.append(
                "max_seq_length runtime="
                f"{self._embedding_max_seq_length} build={manifest.max_seq_length}"
            )

        if mismatches:
            raise RuntimeError(
                "Ingest manifest mismatch: "
                + "; ".join(mismatches)
                + ". Rebuild the FAISS index and restart the worker."
            )

    def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> RetrievalReport:
        if self._info is None:
            raise RuntimeError("Retriever is not ready — call ensure_ready() first")
        # Dense-retrieval wall-clock (Phase 2A-L). Stops just before the
        # reranker call so the eval harness can attribute total query
        # latency between FAISS-side and reranker-side. Includes parser
        # + filter resolution + embed + FAISS + RRF merge — i.e. all the
        # work the bi-encoder path does even when reranking is off.
        dense_t0 = time.perf_counter()
        parsed = self._parser.parse(query)
        # Normalized form is what the embedder actually sees. The RegexQueryParser
        # collapses whitespace and strips unicode quotes here; NoOp passes through.
        embed_query = parsed.normalized or query

        # Compose effective filters: the caller's explicit dict wins over
        # whatever the parser inferred, which matches the usual
        # "caller-knows-best" ordering.
        parsed_filters = dict(parsed.filters or {})
        override = dict(filters or {})
        effective_filters: dict = {**parsed_filters, **override}

        # Resolve the doc_id allowlist BEFORE the FAISS search so we can
        # short-circuit on a filter that matches no documents. At current
        # corpus scale (<10k docs) the lookup is a millisecond or two.
        # TODO: once chunk count exceeds ~100k, swap this post-filter for
        # faiss.IDSelectorArray against chunk ids so we don't waste the
        # candidate budget on docs that are about to be filtered out.
        allowed_doc_ids: Optional[set[str]] = None
        filter_produced_no_docs = False
        if effective_filters:
            try:
                matched = self._metadata.doc_ids_matching(effective_filters)
            except ValueError:
                log.warning(
                    "Retriever: rejecting unknown filter keys %s",
                    sorted(effective_filters.keys()),
                )
                raise
            allowed_doc_ids = {str(d) for d in matched}
            if not allowed_doc_ids:
                filter_produced_no_docs = True
                log.info(
                    "Retriever: filters=%s matched zero docs — short-circuit",
                    effective_filters,
                )
                return RetrievalReport(
                    query=query,
                    top_k=self._top_k,
                    index_version=self._info.index_version,
                    embedding_model=self._info.embedding_model,
                    results=[],
                    reranker_name=self._reranker.name,
                    candidate_k=self._candidate_k,
                    topk_gap=None,
                    topk_rel_gap=None,
                    use_mmr=self._use_mmr,
                    mmr_lambda=self._mmr_lambda if self._use_mmr else None,
                    dup_rate=0.0,
                    parsed_query=parsed,
                    filters=effective_filters,
                    filter_produced_no_docs=True,
                )

        candidates = self._retrieve_candidates(
            embed_query, allowed_doc_ids=allowed_doc_ids,
        )

        # Multi-query RRF: when the parser produced rewrites, run one
        # FAISS search per rewrite, merge their candidate lists with the
        # primary list via Reciprocal Rank Fusion, and hand the merged
        # pool to the reranker. Guarded on ``parsed.rewrites`` so the
        # NoOp / Regex parsers (both empty) skip this path entirely —
        # single-query behaviour stays bit-for-bit identical.
        if parsed.rewrites:
            rewrite_candidates = [
                self._retrieve_candidates(r, allowed_doc_ids=allowed_doc_ids)
                for r in parsed.rewrites
            ]
            candidates = _rrf_merge(
                [candidates] + rewrite_candidates,
                k_rrf=self._rrf_k,
                pool_size=self._candidate_k,
            )

        # Mark the end of the dense-retrieval phase. The remaining wall-
        # clock (rerank + MMR) is attributed to the reranker side. Even
        # the noop path passes this point so dense_retrieval_ms always
        # has a value when the request reached candidate selection.
        dense_retrieval_ms = round((time.perf_counter() - dense_t0) * 1000.0, 3)

        if not candidates:
            return RetrievalReport(
                query=query,
                top_k=self._top_k,
                index_version=self._info.index_version,
                embedding_model=self._info.embedding_model,
                results=[],
                reranker_name=self._reranker.name,
                candidate_k=self._candidate_k,
                topk_gap=None,
                topk_rel_gap=None,
                use_mmr=self._use_mmr,
                mmr_lambda=self._mmr_lambda if self._use_mmr else None,
                dup_rate=0.0,
                parsed_query=parsed,
                filters=effective_filters,
                filter_produced_no_docs=filter_produced_no_docs,
                dense_retrieval_ms=dense_retrieval_ms,
            )

        # When MMR is on, ask the reranker for its FULL candidate pool so
        # the diversity selector has somewhere to reach for lower-ranked
        # chunks from other docs. The MMR selector then trims to top_k.
        # When MMR is off, trim at the reranker as before — bit-for-bit
        # Phase 1 behaviour.
        rerank_k = len(candidates) if self._use_mmr else self._top_k
        # Time the rerank step in isolation so the Phase 2A retrieval
        # eval can attribute latency to the reranker (separate from the
        # bi-encoder + FAISS round-trip that dominates the NoOp path).
        # NoOpReranker still gets timed — its measured cost is a
        # microsecond, the harness reports None when reranker_name is
        # "noop" so the metric stays meaningful.
        rerank_t0 = time.perf_counter()
        reranked = self._reranker.rerank(embed_query, candidates, k=rerank_k)
        rerank_ms = round((time.perf_counter() - rerank_t0) * 1000.0, 3)
        report_rerank_ms: Optional[float] = (
            None if self._reranker.name == "noop" else rerank_ms
        )
        # Pull the per-stage breakdown the reranker recorded for this
        # call — only present when the reranker provider is the cross-
        # encoder with stage-timing on. ``getattr`` keeps the contract
        # forward-compatible for future RerankerProvider implementations
        # that don't expose the hook (e.g. NoOpReranker, a hypothetical
        # API-backed reranker, or a stub in tests).
        breakdown_attr = getattr(self._reranker, "last_breakdown_ms", None)
        rerank_breakdown_ms: Optional[Dict[str, float]] = (
            dict(breakdown_attr)
            if isinstance(breakdown_attr, dict) and breakdown_attr
            else None
        )

        if self._use_mmr:
            results = _mmr_select(
                reranked,
                top_k=self._top_k,
                mmr_lambda=self._mmr_lambda,
                doc_id_penalty=_MMR_DOC_ID_PENALTY,
            )
        else:
            results = reranked

        topk_gap, topk_rel_gap = _compute_topk_gap(results)
        dup_rate_value = _compute_dup_rate(results)

        return RetrievalReport(
            query=query,
            top_k=self._top_k,
            index_version=self._info.index_version,
            embedding_model=self._info.embedding_model,
            results=results,
            reranker_name=self._reranker.name,
            candidate_k=self._candidate_k,
            topk_gap=topk_gap,
            topk_rel_gap=topk_rel_gap,
            use_mmr=self._use_mmr,
            mmr_lambda=self._mmr_lambda if self._use_mmr else None,
            dup_rate=dup_rate_value,
            parsed_query=parsed,
            filters=effective_filters,
            filter_produced_no_docs=filter_produced_no_docs,
            rerank_ms=report_rerank_ms,
            dense_retrieval_ms=dense_retrieval_ms,
            rerank_breakdown_ms=rerank_breakdown_ms,
        )

    def _retrieve_candidates(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> List[RetrievedChunk]:
        """Run one FAISS search + metadata lookup, return candidate chunks.

        Factored out of ``retrieve`` so the multi-query RRF path can reuse
        it per rewrite without duplicating the embed/search/lookup glue.
        The returned list is ordered by bi-encoder score (descending) and
        capped at ``candidate_k``.

        When ``allowed_doc_ids`` is provided, FAISS is asked for
        ``candidate_k * 2`` candidates and the result is post-filtered
        down to ``candidate_k`` matches against the allowlist. The 2x
        over-fetch is a cheap hedge against the filter eliminating most
        of the top-k; we could tune it later but at current scale the
        extra lookup is negligible.
        """
        overfetch = self._candidate_k * 2 if allowed_doc_ids is not None else self._candidate_k
        vectors = self._embedder.embed_queries([query])
        hits = self._index.search(vectors, top_k=overfetch)
        if not hits or not hits[0]:
            return []
        row_ids = [row_id for row_id, _score in hits[0]]
        looked_up = self._metadata.lookup_chunks_by_faiss_rows(
            self._info.index_version, row_ids
        )
        score_by_row = {row_id: score for row_id, score in hits[0]}
        candidates: List[RetrievedChunk] = []
        for hit in looked_up:
            if allowed_doc_ids is not None and hit.doc_id not in allowed_doc_ids:
                continue
            candidates.append(RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                section=hit.section or "",
                text=hit.text,
                score=float(score_by_row.get(hit.faiss_row_id, 0.0)),
            ))
            if len(candidates) >= self._candidate_k:
                break
        return candidates


def _compute_topk_gap(
    results: List[RetrievedChunk],
) -> tuple[Optional[float], Optional[float]]:
    """Compute absolute and relative gap between rank-1 and rank-k scores.

    Prefers ``rerank_score`` when present (the reranker is the live signal
    driving ordering); falls back to the bi-encoder ``score``. Returns
    ``(None, None)`` when the list has fewer than two items or both
    scores are missing — the caller surfaces None so ops can distinguish
    "no gap data" from "gap is 0".
    """
    if len(results) < 2:
        return None, None

    def _pick(c: RetrievedChunk) -> Optional[float]:
        return c.rerank_score if c.rerank_score is not None else c.score

    s1 = _pick(results[0])
    s2 = _pick(results[-1])
    if s1 is None or s2 is None:
        return None, None
    abs_gap = round(float(s1) - float(s2), 4)
    rel_gap = round(abs_gap / float(s1), 4) if abs(s1) > 1e-9 else None
    return abs_gap, rel_gap


def _compute_dup_rate(results: List[RetrievedChunk]) -> float:
    """Fraction of duplicate doc_ids in the result list: 1 - unique/len.

    Same definition as the eval harness metric so the per-call value
    the Retriever surfaces in RetrievalReport matches what the offline
    harness computes post-hoc. Rounded to 4 dp for stable report output.
    """
    n = len(results)
    if n <= 1:
        return 0.0
    unique = len({c.doc_id for c in results})
    return round(1.0 - unique / float(n), 4)


def _mmr_select(
    candidates: List[RetrievedChunk],
    *,
    top_k: int,
    mmr_lambda: float,
    doc_id_penalty: float,
) -> List[RetrievedChunk]:
    """Pick top_k chunks using MMR with a doc_id-based diversity penalty.

    For each un-selected candidate we compute::

        value = mmr_lambda * relevance - (1 - mmr_lambda) * max_penalty

    where ``relevance`` is the candidate's ``rerank_score`` when present
    and falls back to the bi-encoder ``score``; ``max_penalty`` is the
    maximum doc_id-overlap penalty against the already-selected set
    (``doc_id_penalty`` when the candidate's doc_id matches any selected
    chunk, 0.0 otherwise).

    The first pick is always the highest-relevance candidate (nothing
    is selected yet, so max_penalty is 0 for every candidate). At
    ``mmr_lambda == 1.0`` the penalty term vanishes entirely and the
    selector degenerates to relevance-only — ordering matches the
    no-MMR path exactly, which is what the "lambda=1.0" contract test
    exercises.

    Pure-Python, O(top_k * len(candidates)); candidate lists are bounded
    by ``candidate_k`` (default 30) so the quadratic factor is fine.
    """
    k = max(0, int(top_k))
    if k == 0 or not candidates:
        return []

    def _relevance(c: RetrievedChunk) -> float:
        if c.rerank_score is not None:
            return float(c.rerank_score)
        return float(c.score)

    remaining: List[RetrievedChunk] = list(candidates)
    selected: List[RetrievedChunk] = []
    selected_doc_ids: set[str] = set()

    while remaining and len(selected) < k:
        best_idx = 0
        best_value = float("-inf")
        for i, cand in enumerate(remaining):
            relevance = _relevance(cand)
            max_penalty = doc_id_penalty if cand.doc_id in selected_doc_ids else 0.0
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_penalty
            if value > best_value:
                best_value = value
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_doc_ids.add(chosen.doc_id)

    return selected


def _rrf_merge(
    candidate_lists: List[List[RetrievedChunk]],
    *,
    k_rrf: int,
    pool_size: int,
) -> List[RetrievedChunk]:
    """Merge per-query candidate lists via Reciprocal Rank Fusion.

    For each chunk_id appearing in any of the per-query result lists we
    sum ``1 / (k_rrf + rank_i)`` across the lists that contain it (rank
    is 1-based). The chunks are then sorted by the fused score in
    descending order and capped at ``pool_size`` so the reranker sees a
    stable-size pool regardless of how many rewrites the parser emits.

    Each chunk's ``score`` field is overwritten with the fused RRF
    score so downstream code (reranker fallback, MMR relevance signal,
    topk_gap computation) has a single consistent ordering signal. The
    original bi-encoder score is preserved on the variant that wins —
    we keep the first occurrence of each chunk_id as the "representative"
    so doc_id / section / text stay intact.

    k_rrf=60 is the Cormack et al. default; the value is exposed on the
    Retriever so evaluation sweeps can tune it without code changes.
    """
    if not candidate_lists:
        return []

    fused_score: dict[str, float] = {}
    representative: dict[str, RetrievedChunk] = {}
    for variant in candidate_lists:
        for rank, chunk in enumerate(variant, start=1):
            contribution = 1.0 / float(k_rrf + rank)
            fused_score[chunk.chunk_id] = (
                fused_score.get(chunk.chunk_id, 0.0) + contribution
            )
            representative.setdefault(chunk.chunk_id, chunk)

    merged: List[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=rep.chunk_id,
            doc_id=rep.doc_id,
            section=rep.section,
            text=rep.text,
            score=fused_score[rep.chunk_id],
            rerank_score=rep.rerank_score,
        )
        for rep in representative.values()
    ]
    merged.sort(key=lambda c: c.score, reverse=True)
    cap = max(1, int(pool_size))
    return merged[:cap]
