"""Retrieval-quality eval harness (no generator).

Scores a Retriever-like object against an eval-query JSONL by computing
the diagnostic metrics defined in ``eval.harness.metrics``. Deliberately
does not invoke a GenerationProvider — the goal is to measure dense
retrieval quality in isolation, before any reranker / MMR / hybrid
work touches the pipeline.

Inputs
------
- ``dataset``: list of dicts shaped like the rows in
  ``eval/eval_queries/<file>.jsonl`` (see eval_queries/README.md).
- ``retriever``: anything with ``retrieve(query: str) -> Report`` where
  the Report has ``.results`` (list of chunks with ``doc_id``,
  ``chunk_id``, ``section``, ``text``, ``score``, ``rerank_score``)
  and a few optional metadata fields the harness picks up if present
  (``index_version``, ``embedding_model``, ``reranker_name``).

Outputs
-------
``run_retrieval_eval`` returns a 4-tuple:

  (summary, rows, top_k_dump, duplicate_analysis)

The CLI persists each as one of the four artifacts:

  - ``retrieval_eval_report.json``   — summary + rows
  - ``retrieval_eval_report.md``     — human-readable report
  - ``top_k_dump.jsonl``             — one line per (query, rank) pair
  - ``duplicate_analysis.json``      — per-query + aggregate dup stats

Per-row fields measured
-----------------------
- hit@1, hit@3, hit@5
- mrr@10, ndcg@10
- dup_rate, unique_doc_coverage @ k
- top1_score_margin
- avg_context_token_count over top-k chunk texts
- expected_keyword_match_rate over top-k chunk texts

Failure mode
------------
A retrieve() failure on any single row becomes ``row.error`` + an
``error_count`` increment in the summary; the harness never raises
mid-run. This matches the older ``rag_eval.py`` contract so a busted
query doesn't poison a 200-row sweep.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from eval.harness.metrics import (
    count_whitespace_tokens,
    duplicate_doc_ratio_at_k,
    dup_rate,
    efficiency_score,
    expected_keyword_match_rate,
    hit_at_k,
    ndcg_at_k,
    normalized_text_hash,
    p_percentile,
    quality_score,
    recall_at_k,
    reciprocal_rank_at_k,
    section_diversity_at_k,
    top1_score_margin,
    unique_doc_count_at_k,
    unique_doc_coverage,
)

log = logging.getLogger(__name__)


# Defaults match the user-facing spec for this eval mode. Overridable
# via the CLI / programmatic API.
DEFAULT_TOP_K = 10
DEFAULT_MRR_K = 10
DEFAULT_NDCG_K = 10
DEFAULT_HIT_KS: Tuple[int, ...] = (1, 3, 5)
# Extra hit-cutoffs surfaced for the Phase 2A candidate-recall report.
# Anything in this tuple beyond {1, 3, 5} flows into row.extra_hits and
# summary.mean_extra_hits — the markdown writer renders the union as
# additional headline-metrics rows. Pass an empty tuple to disable.
DEFAULT_EXTRA_HIT_KS: Tuple[int, ...] = ()

# Top-k dump preview cap — chunk_preview is meant to be eyeball-able
# in jq / less, not the full chunk text (which is in the live store).
PREVIEW_CHARS = 160

# Duplicate analysis — most-common reporting cap.
DUP_TOP_N = 10

# Phase 1 retrieval-eval extensions ------------------------------------------
# K cutoffs for candidate hit / recall (retrieval pre-rerank step).
DEFAULT_CANDIDATE_KS: Tuple[int, ...] = (10, 20, 50, 100)
# K cutoffs for diversity / duplicate diagnostics over the final top-k.
DEFAULT_DIVERSITY_KS: Tuple[int, ...] = (5, 10)
# Sample-size threshold below which a query_type breakdown is flagged
# as low-confidence in the markdown report.
DEFAULT_LOW_QUERY_TYPE_SAMPLE = 5
# Default fallback bucket name for rows with no explicit query_type.
DEFAULT_QUERY_TYPE_UNKNOWN = "unknown"
# Diagnostic thresholds — surfaced as module-level constants so the
# unit tests pin them and the report renderer can quote the values.
DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50 = 0.80
DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5 = 0.01
DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5 = -0.005
DIAG_RERANKER_NEGATIVE_UPLIFT_MRR_AT_10 = -0.005
DIAG_HIGH_DUPLICATE_RATIO_AT_10 = 0.50


# ---------------------------------------------------------------------------
# Protocols (mirror the Retriever contract without importing from app/).
# ---------------------------------------------------------------------------


class _RetrieverLike(Protocol):
    def retrieve(self, query: str) -> Any: ...


# ---------------------------------------------------------------------------
# Dataclasses (1:1 with on-disk JSON shape via dataclasses.asdict).
# ---------------------------------------------------------------------------


@dataclass
class RetrievalEvalRow:
    id: str
    query: str
    language: Optional[str] = None
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_section_keywords: List[str] = field(default_factory=list)
    answer_type: Optional[str] = None
    difficulty: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    # Retrieval output
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    # Per-row metrics
    hit_at_1: Optional[float] = None
    hit_at_3: Optional[float] = None
    hit_at_5: Optional[float] = None
    mrr_at_10: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    dup_rate: Optional[float] = None
    unique_doc_coverage: Optional[float] = None
    top1_score_margin: Optional[float] = None
    avg_context_token_count: Optional[float] = None
    expected_keyword_match_rate: Optional[float] = None
    # Latency + provenance
    retrieval_ms: float = 0.0
    rerank_ms: Optional[float] = None
    # Phase 2A-L: dense-retrieval wall-clock (FAISS + embedder + parser
    # + RRF), measured separately from the rerank step. ``None`` when
    # the retriever doesn't surface it (older RetrievalReport without
    # the field, or a stub used in tests).
    dense_retrieval_ms: Optional[float] = None
    # Phase 2A-L: per-stage rerank breakdown when the cross-encoder
    # collects stage timings. Keys: ``pair_build_ms``, ``tokenize_ms``,
    # ``forward_ms``, ``postprocess_ms``, ``total_rerank_ms``. ``None``
    # for noop / OOM-fallback paths and any reranker that doesn't
    # expose ``last_breakdown_ms``.
    rerank_breakdown_ms: Optional[Dict[str, float]] = None
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    error: Optional[str] = None
    # Extra hit@k cutoffs requested by the caller (Phase 2A
    # candidate-recall: typically {10, 20, 50}). Keys are stringified
    # ints so the JSON shape remains stable across writers; values are
    # ``None`` for rows without expected_doc_ids, matching the contract
    # of hit_at_1 / hit_at_3 / hit_at_5.
    extra_hits: Dict[str, Optional[float]] = field(default_factory=dict)
    # Phase 1 retrieval-eval extensions ----------------------------------
    # Query-type bucket. Pulled from raw row's ``query_type`` field; rows
    # without one are bucketed under ``DEFAULT_QUERY_TYPE_UNKNOWN`` for
    # the byQueryType breakdown (left ``None`` here to preserve the
    # source-of-truth: "the dataset didn't say").
    query_type: Optional[str] = None
    # Section paths captured from ``chunk.section`` for diversity / dup
    # accounting. Same length as ``retrieved_doc_ids``.
    section_paths: List[str] = field(default_factory=list)
    # Pre-rerank ranking — derived by sorting the retriever's results by
    # raw dense score in descending order. When the report doesn't
    # surface rerank_scores at all, pre_rerank == final and the row's
    # rerank-uplift metrics are 0.
    pre_rerank_doc_ids: List[str] = field(default_factory=list)
    pre_rerank_scores: List[float] = field(default_factory=list)
    # Candidate-pool ranking. When the retriever surfaces a wider pre-
    # rerank candidate pool (e.g. ``report.candidate_doc_ids`` in the
    # ``BoostingRetrievalReport`` shape), the harness records it here
    # so candidate hit@10 / hit@50 / hit@100 can score what the
    # reranker started from rather than what it returned. Empty list
    # when no candidate pool was surfaced.
    candidate_doc_ids: List[str] = field(default_factory=list)
    candidate_count: Optional[int] = None
    final_context_count: Optional[int] = None
    # Sum of whitespace tokens across the final top-k chunks; coexists
    # with ``avg_context_token_count`` (mean) so the aggregator can
    # surface both per-chunk and per-query views.
    context_tokens: Optional[int] = None
    # Candidate hit / recall at the wider cutoffs (10/20/50/100). Values
    # are ``None`` when no candidate pool was surfaced AND the row's
    # final top-k is shorter than the cutoff *or* the row has no
    # expected_doc_ids. Keys: stringified ints for stable JSON shape.
    candidate_hits: Dict[str, Optional[float]] = field(default_factory=dict)
    candidate_recalls: Dict[str, Optional[float]] = field(default_factory=dict)
    # Pre-rerank quality at the same headline cutoffs as the final
    # metrics. ``None`` when the row has no expected_doc_ids.
    pre_rerank_hit_at_1: Optional[float] = None
    pre_rerank_hit_at_3: Optional[float] = None
    pre_rerank_hit_at_5: Optional[float] = None
    pre_rerank_mrr_at_10: Optional[float] = None
    pre_rerank_ndcg_at_10: Optional[float] = None
    # Diversity / duplicate diagnostics over the final top-k.
    # ``duplicate_doc_ratio_at_k`` complements ``unique_doc_count_at_k``
    # for the K cutoffs in ``DEFAULT_DIVERSITY_KS`` (5, 10). Keys are
    # the same stringified-int convention.
    duplicate_doc_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    unique_doc_counts: Dict[str, Optional[int]] = field(default_factory=dict)
    # ``section_diversity_at_k`` is ``None`` when no section info was
    # populated on the retrieved chunks (the dataset / retriever didn't
    # surface section paths) — this is "metric not measurable", not 0.
    section_diversities: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class RetrievalEvalSummary:
    dataset_path: str
    corpus_path: Optional[str]
    row_count: int
    rows_with_expected_doc_ids: int
    rows_with_expected_keywords: int
    top_k: int
    mrr_k: int
    ndcg_k: int
    # Aggregates (None when no row contributed)
    mean_hit_at_1: Optional[float]
    mean_hit_at_3: Optional[float]
    mean_hit_at_5: Optional[float]
    mean_mrr_at_10: Optional[float]
    mean_ndcg_at_10: Optional[float]
    mean_dup_rate: Optional[float]
    mean_unique_doc_coverage: Optional[float]
    mean_top1_score_margin: Optional[float]
    mean_avg_context_token_count: Optional[float]
    mean_expected_keyword_match_rate: Optional[float]
    # Latency
    mean_retrieval_ms: float
    p50_retrieval_ms: float
    p95_retrieval_ms: float
    max_retrieval_ms: float
    # Rerank-specific latency (None when no row reported a rerank_ms).
    # Phase 2A retrieval-rerank reports populate these; the
    # NoOpReranker path leaves them None.
    rerank_row_count: int = 0
    mean_rerank_ms: Optional[float] = None
    p50_rerank_ms: Optional[float] = None
    p95_rerank_ms: Optional[float] = None
    max_rerank_ms: Optional[float] = None
    # Phase 2A-L extended latency aggregates: p90 + p99 on the headline
    # retrieval_ms + rerank_ms series, plus dense_retrieval_ms summary
    # and rerank_breakdown summary. All Optional so older retrieval
    # reports (without these fields populated by the retriever) round-
    # trip through asdict() with the same shape.
    p90_retrieval_ms: Optional[float] = None
    p99_retrieval_ms: Optional[float] = None
    p90_rerank_ms: Optional[float] = None
    p99_rerank_ms: Optional[float] = None
    dense_retrieval_row_count: int = 0
    mean_dense_retrieval_ms: Optional[float] = None
    p50_dense_retrieval_ms: Optional[float] = None
    p90_dense_retrieval_ms: Optional[float] = None
    p95_dense_retrieval_ms: Optional[float] = None
    p99_dense_retrieval_ms: Optional[float] = None
    max_dense_retrieval_ms: Optional[float] = None
    # Per-stage rerank breakdown summary. Each value is a dict with
    # keys ``avg``, ``p50``, ``p90``, ``p95``, ``p99``, ``max``,
    # ``count`` (rows that contributed). The outer dict's keys are the
    # stage names emitted by ``CrossEncoderReranker.last_breakdown_ms``
    # (``pair_build_ms``, ``tokenize_ms``, ``forward_ms``,
    # ``postprocess_ms``, ``total_rerank_ms``). Empty dict when no row
    # surfaced a breakdown — preserves the older shape for noop runs.
    rerank_breakdown_stats: Dict[str, Dict[str, Optional[float]]] = field(
        default_factory=dict,
    )
    # Extra hit-cutoff aggregates surfaced for the candidate-recall
    # report (Phase 2A). Keys are the same stringified-int form used on
    # ``RetrievalEvalRow.extra_hits`` so dict round-tripping through
    # ``asdict`` is stable.
    mean_extra_hits: Dict[str, Optional[float]] = field(default_factory=dict)
    # Provenance
    index_version: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_name: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    error_count: int = 0
    # Per-answer-type breakdown (mean hit@5, mean ndcg@10, count)
    per_answer_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # ----------------------------------------------------------------
    # Phase 1 retrieval-eval extensions. All fields are Optional /
    # default-empty so older readers diff cleanly: no key disappeared,
    # only new ones added. Aggregation helpers (``_mean_or_none`` etc.)
    # leave the field at ``None`` when no row contributed a value.
    # ----------------------------------------------------------------
    # Candidate-pool aggregates. Keys are the same stringified ints used
    # on the per-row ``candidate_hits`` / ``candidate_recalls`` dicts.
    candidate_hit_rates: Dict[str, Optional[float]] = field(default_factory=dict)
    candidate_recalls: Dict[str, Optional[float]] = field(default_factory=dict)
    # Pre-rerank aggregates at the headline cutoffs.
    mean_pre_rerank_hit_at_1: Optional[float] = None
    mean_pre_rerank_hit_at_3: Optional[float] = None
    mean_pre_rerank_hit_at_5: Optional[float] = None
    mean_pre_rerank_mrr_at_10: Optional[float] = None
    mean_pre_rerank_ndcg_at_10: Optional[float] = None
    # Reranker uplift = final - pre_rerank. ``None`` when either side is
    # missing. The aggregator only computes uplift when both sides have
    # populated rows so a 0-uplift run looks identical to a no-rerank
    # run only when the reranker truly didn't move the order.
    rerank_uplift_hit_at_1: Optional[float] = None
    rerank_uplift_hit_at_3: Optional[float] = None
    rerank_uplift_hit_at_5: Optional[float] = None
    rerank_uplift_mrr_at_10: Optional[float] = None
    rerank_uplift_ndcg_at_10: Optional[float] = None
    # Diversity / duplicate aggregates at the K cutoffs in
    # ``DEFAULT_DIVERSITY_KS`` (5, 10). Keys: stringified ints.
    duplicate_doc_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    unique_doc_counts: Dict[str, Optional[float]] = field(default_factory=dict)
    section_diversities: Dict[str, Optional[float]] = field(default_factory=dict)
    # Latency / cost aggregates kept side-by-side with quality so the
    # report can say "what did this query cost". The ``*_total_*`` fields
    # are spec-preferred aliases of mean_retrieval_ms / p95_retrieval_ms;
    # ``avg_dense_retrieval_ms`` mirrors mean_dense_retrieval_ms and
    # ``avg_rerank_ms`` mirrors mean_rerank_ms so consumers can read the
    # spec's field names. The original ``mean_*`` / ``p95_*`` fields are
    # left untouched for backward compat — both names round-trip.
    avg_total_retrieval_ms: Optional[float] = None
    p95_total_retrieval_ms: Optional[float] = None
    avg_dense_retrieval_ms: Optional[float] = None
    avg_rerank_ms: Optional[float] = None
    avg_candidate_count: Optional[float] = None
    avg_final_context_count: Optional[float] = None
    avg_context_tokens: Optional[float] = None
    # Composite scores. Both are *comparison* aids, not adoption
    # decisions; the markdown writer flags them accordingly.
    quality_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    # By-query-type breakdown — same shape as ``per_answer_type`` but
    # over the spec's query_type field, with the row-count threshold
    # surfaced for the markdown writer's "low sample count" callout.
    by_query_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Diagnostic flags. Each value is a 3-tuple of (bool/None, threshold
    # value, observed value) so the markdown writer can quote both. We
    # represent it as a dict per flag with stable keys so the JSON
    # round-trip stays clean.
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopKDumpRow:
    query_id: str
    query: str
    rank: int
    doc_id: str
    chunk_id: str
    section_path: Optional[str]
    score: float
    normalized_score: Optional[float]
    chunk_preview: str
    is_expected_doc: bool
    matched_expected_keyword: List[str] = field(default_factory=list)


@dataclass
class DuplicateAnalysis:
    top_k: int
    queries_evaluated: int
    queries_with_doc_dup: int
    queries_with_section_dup: int
    queries_with_text_dup: int
    queries_with_doc_dup_ratio: float
    queries_with_section_dup_ratio: float
    queries_with_text_dup_ratio: float
    most_common_duplicate_doc_ids: List[Dict[str, Any]] = field(default_factory=list)
    most_common_duplicate_text_hashes: List[Dict[str, Any]] = field(default_factory=list)
    per_query: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def run_retrieval_eval(
    dataset: List[Mapping[str, Any]],
    *,
    retriever: _RetrieverLike,
    top_k: int = DEFAULT_TOP_K,
    mrr_k: int = DEFAULT_MRR_K,
    ndcg_k: int = DEFAULT_NDCG_K,
    extra_hit_ks: Tuple[int, ...] = DEFAULT_EXTRA_HIT_KS,
    candidate_ks: Tuple[int, ...] = DEFAULT_CANDIDATE_KS,
    diversity_ks: Tuple[int, ...] = DEFAULT_DIVERSITY_KS,
    dataset_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
) -> Tuple[
    RetrievalEvalSummary,
    List[RetrievalEvalRow],
    List[TopKDumpRow],
    DuplicateAnalysis,
]:
    """Score ``dataset`` against ``retriever`` and emit the four reports.

    The retrieval logic itself is untouched — this function only reads
    what the Retriever already returns. Knobs like top_k pass through to
    the Retriever, but the retriever's internal candidate_k / use_mmr /
    reranker config is taken as-is.

    ``extra_hit_ks`` is the Phase 2A candidate-recall hook: pass
    ``(10, 20, 50)`` to get hit@10/@20/@50 in row.extra_hits and the
    corresponding aggregates in summary.mean_extra_hits. The default
    empty tuple preserves byte-identical Phase 1 reports.
    """
    started_at = _now_iso()
    run_start = time.perf_counter()

    rows: List[RetrievalEvalRow] = []
    top_k_dump: List[TopKDumpRow] = []
    errors = 0

    # Per-query duplicate accounting.
    per_query_dup: List[Dict[str, Any]] = []
    doc_dup_counter: Counter = Counter()
    text_dup_counter: Counter = Counter()
    queries_with_doc_dup = 0
    queries_with_section_dup = 0
    queries_with_text_dup = 0

    for idx, raw in enumerate(dataset, start=1):
        query_id = str(raw.get("id") or f"row-{idx:04d}").strip()
        query_text = str(raw.get("query") or "").strip()
        if not query_text:
            log.warning("Row %d (id=%s) has no query — skipping", idx, query_id)
            continue

        # Phase 1: ``expected_doc_id`` (singular) is normalized into the
        # plural ``expected_doc_ids`` list so older datasets work
        # without a separate projection pass. Plural wins when both are
        # present.
        expected_doc_ids = _list_of_str(raw.get("expected_doc_ids"))
        if not expected_doc_ids:
            singular = raw.get("expected_doc_id")
            if singular:
                expected_doc_ids = [str(singular).strip()]
        expected_keywords = _list_of_str(raw.get("expected_section_keywords"))
        tags = _list_of_str(raw.get("tags"))
        query_type_raw = raw.get("query_type")
        query_type = (
            str(query_type_raw).strip()
            if query_type_raw is not None and str(query_type_raw).strip()
            else None
        )

        row = RetrievalEvalRow(
            id=query_id,
            query=query_text,
            language=str(raw["language"]) if raw.get("language") else None,
            expected_doc_ids=expected_doc_ids,
            expected_section_keywords=expected_keywords,
            answer_type=str(raw["answer_type"]) if raw.get("answer_type") else None,
            difficulty=str(raw["difficulty"]) if raw.get("difficulty") else None,
            tags=tags,
            query_type=query_type,
        )

        try:
            t0 = time.perf_counter()
            report = retriever.retrieve(query_text)
            t1 = time.perf_counter()
            row.retrieval_ms = round((t1 - t0) * 1000.0, 3)

            results = list(getattr(report, "results", []) or [])[:top_k]
            doc_ids = [getattr(r, "doc_id", "") for r in results]
            chunk_ids = [getattr(r, "chunk_id", "") for r in results]
            scores = [float(getattr(r, "score", 0.0)) for r in results]
            chunk_texts = [str(getattr(r, "text", "") or "") for r in results]
            sections = [
                str(getattr(r, "section", "") or "") for r in results
            ]
            rerank_scores = [getattr(r, "rerank_score", None) for r in results]

            row.retrieved_doc_ids = doc_ids
            row.retrieved_chunk_ids = chunk_ids
            row.retrieval_scores = [round(s, 6) for s in scores]
            row.section_paths = [str(s) for s in sections]
            row.index_version = getattr(report, "index_version", None)
            row.embedding_model = getattr(report, "embedding_model", None)
            row.reranker_name = getattr(report, "reranker_name", None)

            # Phase 1: derive pre-rerank ranking by sorting by raw dense
            # score in descending order. When the report carries no
            # rerank_scores at all (NoOp reranker path), the dense order
            # IS the final order — sorted_by_dense is identical to the
            # incoming list and uplift collapses to 0.
            has_rerank_score = any(rs is not None for rs in rerank_scores)
            paired = list(zip(doc_ids, scores))
            if has_rerank_score:
                pre_pairs = sorted(paired, key=lambda p: -float(p[1]))
            else:
                pre_pairs = paired
            row.pre_rerank_doc_ids = [p[0] for p in pre_pairs]
            row.pre_rerank_scores = [round(float(p[1]), 6) for p in pre_pairs]

            # Pre-rerank candidate pool — surfaced by the retriever's
            # report when it wraps its own reranker (BoostingRetrieval-
            # Report.dense_candidates / candidate_doc_ids). Falling back
            # to the final result's doc_ids would conflate candidate@K
            # with hit@K, so we only populate when the report explicitly
            # exposes a wider pool.
            cand_doc_ids: List[str] = []
            cand_attr = getattr(report, "candidate_doc_ids", None)
            if isinstance(cand_attr, (list, tuple)):
                cand_doc_ids = [str(d) for d in cand_attr if d]
            else:
                dense_attr = getattr(report, "dense_candidates", None)
                if isinstance(dense_attr, (list, tuple)):
                    cand_doc_ids = [
                        str(getattr(c, "doc_id", "") or "")
                        for c in dense_attr
                        if getattr(c, "doc_id", None)
                    ]
                else:
                    cand_results = getattr(report, "candidate_results", None)
                    if isinstance(cand_results, (list, tuple)):
                        cand_doc_ids = [
                            str(getattr(c, "doc_id", "") or "")
                            for c in cand_results
                            if getattr(c, "doc_id", None)
                        ]
            row.candidate_doc_ids = cand_doc_ids
            row.candidate_count = len(cand_doc_ids) if cand_doc_ids else None
            row.final_context_count = len(doc_ids)
            # Pull rerank latency off the RetrievalReport when present.
            # ``None`` means the retriever is on the NoOpReranker path,
            # which downstream aggregation distinguishes from a real
            # 0 ms measurement.
            rerank_ms_value = getattr(report, "rerank_ms", None)
            if rerank_ms_value is not None:
                try:
                    row.rerank_ms = float(rerank_ms_value)
                except (TypeError, ValueError):
                    row.rerank_ms = None
            # Phase 2A-L: dense_retrieval_ms + per-stage rerank breakdown.
            # Both are optional on the RetrievalReport — older stubs and
            # third-party retrievers may not populate them, in which
            # case the row stays at the dataclass default (None).
            dense_ms_value = getattr(report, "dense_retrieval_ms", None)
            if dense_ms_value is not None:
                try:
                    row.dense_retrieval_ms = float(dense_ms_value)
                except (TypeError, ValueError):
                    row.dense_retrieval_ms = None
            breakdown_value = getattr(report, "rerank_breakdown_ms", None)
            if isinstance(breakdown_value, dict) and breakdown_value:
                try:
                    row.rerank_breakdown_ms = {
                        str(k): float(v)
                        for k, v in breakdown_value.items()
                        if v is not None
                    }
                except (TypeError, ValueError):
                    row.rerank_breakdown_ms = None

            # Per-row metrics.
            if expected_doc_ids:
                row.hit_at_1 = hit_at_k(doc_ids, expected_doc_ids, k=1)
                row.hit_at_3 = hit_at_k(doc_ids, expected_doc_ids, k=3)
                row.hit_at_5 = hit_at_k(doc_ids, expected_doc_ids, k=5)
                row.mrr_at_10 = reciprocal_rank_at_k(
                    doc_ids, expected_doc_ids, k=mrr_k
                )
                row.ndcg_at_10 = ndcg_at_k(
                    doc_ids, expected_doc_ids, k=ndcg_k
                )
                # Extra hit cutoffs (Phase 2A candidate-recall report).
                # We compute over ``doc_ids`` which is already capped at
                # ``top_k`` — the caller is expected to pass top_k >=
                # max(extra_hit_ks) for the metric to be meaningful;
                # values where k > top_k still resolve correctly (just
                # against the available list) so we don't error.
                row.extra_hits = {
                    str(k): hit_at_k(doc_ids, expected_doc_ids, k=k)
                    for k in extra_hit_ks
                    if k > 0
                }
            row.dup_rate = round(dup_rate(doc_ids), 4)
            row.unique_doc_coverage = unique_doc_coverage(doc_ids, k=top_k)
            row.top1_score_margin = (
                round(top1_score_margin(scores), 6)
                if top1_score_margin(scores) is not None
                else None
            )
            if chunk_texts:
                token_counts = [count_whitespace_tokens(t) for t in chunk_texts]
                row.avg_context_token_count = round(
                    sum(token_counts) / len(token_counts), 2
                )
                row.context_tokens = sum(token_counts)
            row.expected_keyword_match_rate = expected_keyword_match_rate(
                chunk_texts, expected_keywords
            )

            # Phase 1 — pre-rerank quality (over the dense-sorted order).
            # Only computed when expected_doc_ids is populated, matching
            # the contract of the final-rerank metrics above.
            if expected_doc_ids:
                pre_doc_ids = row.pre_rerank_doc_ids
                row.pre_rerank_hit_at_1 = hit_at_k(pre_doc_ids, expected_doc_ids, k=1)
                row.pre_rerank_hit_at_3 = hit_at_k(pre_doc_ids, expected_doc_ids, k=3)
                row.pre_rerank_hit_at_5 = hit_at_k(pre_doc_ids, expected_doc_ids, k=5)
                row.pre_rerank_mrr_at_10 = reciprocal_rank_at_k(
                    pre_doc_ids, expected_doc_ids, k=mrr_k
                )
                row.pre_rerank_ndcg_at_10 = ndcg_at_k(
                    pre_doc_ids, expected_doc_ids, k=ndcg_k
                )

            # Phase 1 — candidate hit / recall over the wider candidate
            # pool. When the retriever didn't surface a separate pool,
            # row.candidate_doc_ids is empty and the candidate metrics
            # stay None (skipped in aggregation) — backward-compatible
            # with reports from retrievers that don't expose candidates.
            if expected_doc_ids and row.candidate_doc_ids:
                cand_hits: Dict[str, Optional[float]] = {}
                cand_recalls: Dict[str, Optional[float]] = {}
                for k in candidate_ks:
                    if k <= 0:
                        continue
                    key = str(k)
                    cand_hits[key] = hit_at_k(
                        row.candidate_doc_ids, expected_doc_ids, k=k
                    )
                    cand_recalls[key] = recall_at_k(
                        row.candidate_doc_ids, expected_doc_ids, k=k
                    )
                row.candidate_hits = cand_hits
                row.candidate_recalls = cand_recalls

            # Phase 1 — diversity / duplicate diagnostics over the final
            # top-k (per-K). Section-diversity stays None when the
            # retriever didn't populate section paths.
            dup_ratios: Dict[str, Optional[float]] = {}
            unique_counts: Dict[str, Optional[int]] = {}
            section_divs: Dict[str, Optional[float]] = {}
            for k in diversity_ks:
                if k <= 0:
                    continue
                key = str(k)
                dup_ratios[key] = duplicate_doc_ratio_at_k(doc_ids, k=k)
                unique_counts[key] = unique_doc_count_at_k(doc_ids, k=k)
                section_divs[key] = section_diversity_at_k(
                    row.section_paths, k=k
                )
            row.duplicate_doc_ratios = dup_ratios
            row.unique_doc_counts = unique_counts
            row.section_diversities = section_divs

            # Top-k dump rows.
            expected_set = set(expected_doc_ids)
            for rank, (doc_id, chunk_id, section, score, text, rerank) in enumerate(
                zip(doc_ids, chunk_ids, sections, scores, chunk_texts, rerank_scores),
                start=1,
            ):
                # bge-m3 IndexFlatIP returns cosine-similarity in [-1, 1];
                # normalize to [0, 1] for an at-a-glance "is this a strong hit"
                # signal in the dump. When a reranker is in play, we report
                # that score (already on its own scale; normalization is a
                # best-effort affine clip and may not be meaningful).
                base_score = float(rerank) if rerank is not None else float(score)
                normalized = max(0.0, min(1.0, (base_score + 1.0) / 2.0))
                preview = _truncate(text, PREVIEW_CHARS)
                matched = [
                    kw for kw in expected_keywords
                    if kw and (kw.lower() in text.lower() or kw.lower() in section.lower())
                ]
                top_k_dump.append(
                    TopKDumpRow(
                        query_id=query_id,
                        query=query_text,
                        rank=rank,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        section_path=section or None,
                        score=round(score, 6),
                        normalized_score=round(normalized, 6),
                        chunk_preview=preview,
                        is_expected_doc=(doc_id in expected_set),
                        matched_expected_keyword=matched,
                    )
                )

            # Duplicate analysis (per-query).
            doc_count = Counter(doc_ids)
            section_count = Counter(zip(doc_ids, sections))
            hashes = [normalized_text_hash(t) for t in chunk_texts]
            text_count = Counter(h for h in hashes if h)

            doc_dup_pairs = [
                (d, c) for d, c in doc_count.items() if c > 1 and d
            ]
            section_dup_pairs = [
                (f"{d}::{s or '<no-section>'}", c)
                for (d, s), c in section_count.items()
                if c > 1 and d
            ]
            text_dup_pairs = [
                (h, c) for h, c in text_count.items() if c > 1
            ]

            had_doc_dup = bool(doc_dup_pairs)
            had_section_dup = bool(section_dup_pairs)
            had_text_dup = bool(text_dup_pairs)
            if had_doc_dup:
                queries_with_doc_dup += 1
            if had_section_dup:
                queries_with_section_dup += 1
            if had_text_dup:
                queries_with_text_dup += 1
            for d, c in doc_dup_pairs:
                doc_dup_counter[d] += c - 1  # weight = #extra copies
            for h, c in text_dup_pairs:
                text_dup_counter[h] += c - 1

            per_query_dup.append(
                {
                    "query_id": query_id,
                    "doc_duplicates": [
                        {"doc_id": d, "count": c} for d, c in doc_dup_pairs
                    ],
                    "section_duplicates": [
                        {"doc_section": ds, "count": c} for ds, c in section_dup_pairs
                    ],
                    "text_hash_duplicates": [
                        {"text_hash": h, "count": c} for h, c in text_dup_pairs
                    ],
                }
            )
        except Exception as ex:
            errors += 1
            row.error = f"{type(ex).__name__}: {ex}"
            log.exception("retrieval eval row %d (id=%s) failed", idx, query_id)

        rows.append(row)

    run_end = time.perf_counter()

    summary = _aggregate(
        rows,
        top_k=top_k,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
        extra_hit_ks=extra_hit_ks,
        candidate_ks=candidate_ks,
        diversity_ks=diversity_ks,
        dataset_path=dataset_path or "<inline>",
        corpus_path=corpus_path,
        errors=errors,
    )
    summary.started_at = started_at
    summary.finished_at = _now_iso()
    summary.duration_ms = round((run_end - run_start) * 1000.0, 3)

    # Duplicate analysis aggregate.
    queries_evaluated = sum(1 for r in rows if r.error is None)
    safe_div = lambda n, d: round(n / d, 4) if d > 0 else 0.0
    dup_analysis = DuplicateAnalysis(
        top_k=top_k,
        queries_evaluated=queries_evaluated,
        queries_with_doc_dup=queries_with_doc_dup,
        queries_with_section_dup=queries_with_section_dup,
        queries_with_text_dup=queries_with_text_dup,
        queries_with_doc_dup_ratio=safe_div(queries_with_doc_dup, queries_evaluated),
        queries_with_section_dup_ratio=safe_div(queries_with_section_dup, queries_evaluated),
        queries_with_text_dup_ratio=safe_div(queries_with_text_dup, queries_evaluated),
        most_common_duplicate_doc_ids=[
            {"doc_id": d, "extra_copies_total": c}
            for d, c in doc_dup_counter.most_common(DUP_TOP_N)
        ],
        most_common_duplicate_text_hashes=[
            {"text_hash": h, "extra_copies_total": c}
            for h, c in text_dup_counter.most_common(DUP_TOP_N)
        ],
        per_query=per_query_dup,
    )

    _log_summary(summary)
    return summary, rows, top_k_dump, dup_analysis


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate(
    rows: List[RetrievalEvalRow],
    *,
    top_k: int,
    mrr_k: int,
    ndcg_k: int,
    extra_hit_ks: Tuple[int, ...],
    candidate_ks: Tuple[int, ...] = DEFAULT_CANDIDATE_KS,
    diversity_ks: Tuple[int, ...] = DEFAULT_DIVERSITY_KS,
    dataset_path: str,
    corpus_path: Optional[str],
    errors: int,
) -> RetrievalEvalSummary:
    h1 = [r.hit_at_1 for r in rows if r.hit_at_1 is not None]
    h3 = [r.hit_at_3 for r in rows if r.hit_at_3 is not None]
    h5 = [r.hit_at_5 for r in rows if r.hit_at_5 is not None]
    mrr = [r.mrr_at_10 for r in rows if r.mrr_at_10 is not None]
    ndcg = [r.ndcg_at_10 for r in rows if r.ndcg_at_10 is not None]
    dup = [r.dup_rate for r in rows if r.dup_rate is not None]
    udc = [r.unique_doc_coverage for r in rows if r.unique_doc_coverage is not None]
    margin = [r.top1_score_margin for r in rows if r.top1_score_margin is not None]
    ctx = [r.avg_context_token_count for r in rows if r.avg_context_token_count is not None]
    kwm = [
        r.expected_keyword_match_rate for r in rows
        if r.expected_keyword_match_rate is not None
    ]
    latencies = [r.retrieval_ms for r in rows if r.error is None]
    rerank_latencies = [
        float(r.rerank_ms) for r in rows
        if r.error is None and r.rerank_ms is not None
    ]

    # Extra hit aggregates: per cutoff k, mean over rows that emitted a
    # value (i.e. rows with non-empty expected_doc_ids).
    mean_extra_hits: Dict[str, Optional[float]] = {}
    for k in extra_hit_ks:
        if k <= 0:
            continue
        key = str(k)
        values = [
            r.extra_hits[key] for r in rows
            if isinstance(r.extra_hits, dict)
            and r.extra_hits.get(key) is not None
        ]
        mean_extra_hits[key] = _mean_or_none(values)

    index_version = next((r.index_version for r in rows if r.index_version), None)
    embedding_model = next((r.embedding_model for r in rows if r.embedding_model), None)
    reranker_name = next((r.reranker_name for r in rows if r.reranker_name), None)

    if rerank_latencies:
        mean_rerank_ms = round(statistics.fmean(rerank_latencies), 3)
        p50_rerank_ms = round(statistics.median(rerank_latencies), 3)
        p90_rerank_ms = round(p_percentile(rerank_latencies, 90.0), 3)
        p95_rerank_ms = round(p_percentile(rerank_latencies, 95.0), 3)
        p99_rerank_ms = round(p_percentile(rerank_latencies, 99.0), 3)
        max_rerank_ms = round(max(rerank_latencies), 3)
    else:
        mean_rerank_ms = None
        p50_rerank_ms = None
        p90_rerank_ms = None
        p95_rerank_ms = None
        p99_rerank_ms = None
        max_rerank_ms = None

    # Phase 2A-L: dense_retrieval_ms aggregate. Distinct from
    # mean_retrieval_ms (full retrieve() wall-clock) so the topN sweep
    # can attribute total query time between FAISS-side and reranker-
    # side. Rows that didn't surface the field (older stub retrievers)
    # contribute nothing — the aggregate falls to None which the writer
    # serialises as an explicit absence.
    dense_latencies = [
        float(r.dense_retrieval_ms) for r in rows
        if r.error is None and r.dense_retrieval_ms is not None
    ]
    if dense_latencies:
        mean_dense_ms = round(statistics.fmean(dense_latencies), 3)
        p50_dense_ms = round(statistics.median(dense_latencies), 3)
        p90_dense_ms = round(p_percentile(dense_latencies, 90.0), 3)
        p95_dense_ms = round(p_percentile(dense_latencies, 95.0), 3)
        p99_dense_ms = round(p_percentile(dense_latencies, 99.0), 3)
        max_dense_ms = round(max(dense_latencies), 3)
    else:
        mean_dense_ms = None
        p50_dense_ms = None
        p90_dense_ms = None
        p95_dense_ms = None
        p99_dense_ms = None
        max_dense_ms = None

    # Per-stage rerank breakdown stats (Phase 2A-L). Walk every row's
    # ``rerank_breakdown_ms`` dict and group values by stage key, then
    # compute avg / p50 / p90 / p95 / p99 / max / count for each stage.
    # Stage keys never seen in any row do not appear in the output —
    # the JSON shape is "what we measured", not "all possible stages".
    breakdown_buckets: Dict[str, List[float]] = {}
    for r in rows:
        if r.error is not None:
            continue
        if not isinstance(r.rerank_breakdown_ms, dict):
            continue
        for stage, value in r.rerank_breakdown_ms.items():
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            breakdown_buckets.setdefault(stage, []).append(value_f)

    rerank_breakdown_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for stage, values in breakdown_buckets.items():
        if not values:
            continue
        rerank_breakdown_stats[stage] = {
            "avg": round(statistics.fmean(values), 3),
            "p50": round(statistics.median(values), 3),
            "p90": round(p_percentile(values, 90.0), 3),
            "p95": round(p_percentile(values, 95.0), 3),
            "p99": round(p_percentile(values, 99.0), 3),
            "max": round(max(values), 3),
            "count": float(len(values)),
        }

    # Phase 1 ----------------------------------------------------------
    # Candidate-pool hit / recall aggregates per cutoff.
    candidate_hit_rates: Dict[str, Optional[float]] = {}
    candidate_recalls: Dict[str, Optional[float]] = {}
    for k in candidate_ks:
        if k <= 0:
            continue
        key = str(k)
        hits_vals = [
            r.candidate_hits[key] for r in rows
            if isinstance(r.candidate_hits, dict)
            and r.candidate_hits.get(key) is not None
        ]
        recall_vals = [
            r.candidate_recalls[key] for r in rows
            if isinstance(r.candidate_recalls, dict)
            and r.candidate_recalls.get(key) is not None
        ]
        candidate_hit_rates[key] = _mean_or_none(hits_vals)
        candidate_recalls[key] = _mean_or_none(recall_vals)

    # Pre-rerank metric aggregates.
    pre_h1 = [r.pre_rerank_hit_at_1 for r in rows if r.pre_rerank_hit_at_1 is not None]
    pre_h3 = [r.pre_rerank_hit_at_3 for r in rows if r.pre_rerank_hit_at_3 is not None]
    pre_h5 = [r.pre_rerank_hit_at_5 for r in rows if r.pre_rerank_hit_at_5 is not None]
    pre_mrr = [
        r.pre_rerank_mrr_at_10 for r in rows if r.pre_rerank_mrr_at_10 is not None
    ]
    pre_ndcg = [
        r.pre_rerank_ndcg_at_10 for r in rows if r.pre_rerank_ndcg_at_10 is not None
    ]
    mean_pre_h1 = _mean_or_none(pre_h1)
    mean_pre_h3 = _mean_or_none(pre_h3)
    mean_pre_h5 = _mean_or_none(pre_h5)
    mean_pre_mrr = _mean_or_none(pre_mrr)
    mean_pre_ndcg = _mean_or_none(pre_ndcg)

    final_h1 = _mean_or_none(h1)
    final_h3 = _mean_or_none(h3)
    final_h5 = _mean_or_none(h5)
    final_mrr = _mean_or_none(mrr)
    final_ndcg = _mean_or_none(ndcg)

    rerank_uplift_h1 = _delta_or_none(final_h1, mean_pre_h1)
    rerank_uplift_h3 = _delta_or_none(final_h3, mean_pre_h3)
    rerank_uplift_h5 = _delta_or_none(final_h5, mean_pre_h5)
    rerank_uplift_mrr = _delta_or_none(final_mrr, mean_pre_mrr)
    rerank_uplift_ndcg = _delta_or_none(final_ndcg, mean_pre_ndcg)

    # Diversity / duplicate aggregates per cutoff.
    duplicate_doc_ratios_agg: Dict[str, Optional[float]] = {}
    unique_doc_counts_agg: Dict[str, Optional[float]] = {}
    section_diversities_agg: Dict[str, Optional[float]] = {}
    for k in diversity_ks:
        if k <= 0:
            continue
        key = str(k)
        dup_vals = [
            r.duplicate_doc_ratios[key] for r in rows
            if isinstance(r.duplicate_doc_ratios, dict)
            and r.duplicate_doc_ratios.get(key) is not None
        ]
        unique_vals = [
            float(r.unique_doc_counts[key]) for r in rows
            if isinstance(r.unique_doc_counts, dict)
            and r.unique_doc_counts.get(key) is not None
        ]
        section_vals = [
            r.section_diversities[key] for r in rows
            if isinstance(r.section_diversities, dict)
            and r.section_diversities.get(key) is not None
        ]
        duplicate_doc_ratios_agg[key] = _mean_or_none(dup_vals)
        unique_doc_counts_agg[key] = _mean_or_none(unique_vals)
        section_diversities_agg[key] = _mean_or_none(section_vals)

    # Latency aliases + extra cost aggregates.
    avg_total_ms = _mean_or_none(latencies) if latencies else None
    p95_total_ms = (
        round(p_percentile(latencies, 95.0), 3) if latencies else None
    )
    cand_count_vals = [
        float(r.candidate_count) for r in rows if r.candidate_count is not None
    ]
    final_count_vals = [
        float(r.final_context_count) for r in rows
        if r.final_context_count is not None
    ]
    context_token_vals = [
        float(r.context_tokens) for r in rows if r.context_tokens is not None
    ]
    avg_candidate_count = _mean_or_none(cand_count_vals)
    avg_final_context_count = _mean_or_none(final_count_vals)
    avg_context_tokens = _mean_or_none(context_token_vals)

    composite_quality = quality_score(
        hit_at_1=final_h1,
        hit_at_5=final_h5,
        mrr_at_10=final_mrr,
        ndcg_at_10=final_ndcg,
    )
    composite_efficiency = efficiency_score(composite_quality, p95_total_ms)

    by_query_type = _query_type_breakdown(
        rows,
        candidate_ks=candidate_ks,
        diversity_ks=diversity_ks,
    )

    summary = RetrievalEvalSummary(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        row_count=len(rows),
        rows_with_expected_doc_ids=len(h5),
        rows_with_expected_keywords=len(kwm),
        top_k=top_k,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
        mean_hit_at_1=final_h1,
        mean_hit_at_3=_mean_or_none(h3),
        mean_hit_at_5=final_h5,
        mean_mrr_at_10=final_mrr,
        mean_ndcg_at_10=final_ndcg,
        mean_dup_rate=_mean_or_none(dup),
        mean_unique_doc_coverage=_mean_or_none(udc),
        mean_top1_score_margin=_mean_or_none(margin),
        mean_avg_context_token_count=_mean_or_none(ctx),
        mean_expected_keyword_match_rate=_mean_or_none(kwm),
        mean_retrieval_ms=_mean_or_zero(latencies),
        p50_retrieval_ms=_p50_or_zero(latencies),
        p95_retrieval_ms=round(p_percentile(latencies, 95.0), 3),
        max_retrieval_ms=round(max(latencies), 3) if latencies else 0.0,
        p90_retrieval_ms=(
            round(p_percentile(latencies, 90.0), 3) if latencies else None
        ),
        p99_retrieval_ms=(
            round(p_percentile(latencies, 99.0), 3) if latencies else None
        ),
        rerank_row_count=len(rerank_latencies),
        mean_rerank_ms=mean_rerank_ms,
        p50_rerank_ms=p50_rerank_ms,
        p90_rerank_ms=p90_rerank_ms,
        p95_rerank_ms=p95_rerank_ms,
        p99_rerank_ms=p99_rerank_ms,
        max_rerank_ms=max_rerank_ms,
        dense_retrieval_row_count=len(dense_latencies),
        mean_dense_retrieval_ms=mean_dense_ms,
        p50_dense_retrieval_ms=p50_dense_ms,
        p90_dense_retrieval_ms=p90_dense_ms,
        p95_dense_retrieval_ms=p95_dense_ms,
        p99_dense_retrieval_ms=p99_dense_ms,
        max_dense_retrieval_ms=max_dense_ms,
        rerank_breakdown_stats=rerank_breakdown_stats,
        mean_extra_hits=mean_extra_hits,
        index_version=index_version,
        embedding_model=embedding_model,
        reranker_name=reranker_name,
        started_at=None,
        finished_at=None,
        duration_ms=0.0,
        error_count=errors,
        per_answer_type=_breakdown(rows, key=lambda r: r.answer_type),
        per_difficulty=_breakdown(rows, key=lambda r: r.difficulty),
        candidate_hit_rates=candidate_hit_rates,
        candidate_recalls=candidate_recalls,
        mean_pre_rerank_hit_at_1=mean_pre_h1,
        mean_pre_rerank_hit_at_3=mean_pre_h3,
        mean_pre_rerank_hit_at_5=mean_pre_h5,
        mean_pre_rerank_mrr_at_10=mean_pre_mrr,
        mean_pre_rerank_ndcg_at_10=mean_pre_ndcg,
        rerank_uplift_hit_at_1=rerank_uplift_h1,
        rerank_uplift_hit_at_3=rerank_uplift_h3,
        rerank_uplift_hit_at_5=rerank_uplift_h5,
        rerank_uplift_mrr_at_10=rerank_uplift_mrr,
        rerank_uplift_ndcg_at_10=rerank_uplift_ndcg,
        duplicate_doc_ratios=duplicate_doc_ratios_agg,
        unique_doc_counts=unique_doc_counts_agg,
        section_diversities=section_diversities_agg,
        avg_total_retrieval_ms=avg_total_ms,
        p95_total_retrieval_ms=p95_total_ms,
        avg_dense_retrieval_ms=mean_dense_ms,
        avg_rerank_ms=mean_rerank_ms,
        avg_candidate_count=avg_candidate_count,
        avg_final_context_count=avg_final_context_count,
        avg_context_tokens=avg_context_tokens,
        quality_score=composite_quality,
        efficiency_score=composite_efficiency,
        by_query_type=by_query_type,
    )
    summary.diagnostics = compute_retrieval_diagnostics(summary)
    return summary


def _breakdown(
    rows: List[RetrievalEvalRow],
    *,
    key,
) -> Dict[str, Dict[str, Any]]:
    """Group rows by ``key(row)`` and aggregate the headline metrics.

    Buckets with key=None are skipped — they're the rows where the
    dataset author left the field off, and lumping them together
    under a fake "unknown" bucket would just hide the gap.
    """
    buckets: Dict[str, List[RetrievalEvalRow]] = {}
    for r in rows:
        k = key(r)
        if not k:
            continue
        buckets.setdefault(k, []).append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for k, brows in sorted(buckets.items()):
        out[k] = {
            "row_count": len(brows),
            "mean_hit_at_5": _mean_or_none(
                [r.hit_at_5 for r in brows if r.hit_at_5 is not None]
            ),
            "mean_mrr_at_10": _mean_or_none(
                [r.mrr_at_10 for r in brows if r.mrr_at_10 is not None]
            ),
            "mean_ndcg_at_10": _mean_or_none(
                [r.ndcg_at_10 for r in brows if r.ndcg_at_10 is not None]
            ),
        }
    return out


def _query_type_breakdown(
    rows: List[RetrievalEvalRow],
    *,
    candidate_ks: Tuple[int, ...],
    diversity_ks: Tuple[int, ...],
) -> Dict[str, Dict[str, Any]]:
    """Per-query-type aggregate following the Phase 1 spec.

    Rows with no ``query_type`` land in the ``DEFAULT_QUERY_TYPE_UNKNOWN``
    bucket so the breakdown still surfaces every row — the spec wants
    visibility on rows that the dataset author hadn't tagged. Each bucket
    carries: ``count``, ``hit_at_1/3/5``, ``mrr_at_10``, ``ndcg_at_10``,
    ``candidate_hit_at_50``, ``candidate_recall_at_50``,
    ``avg_total_retrieval_ms``, ``p95_total_retrieval_ms``,
    ``duplicate_doc_ratio_at_10``. Sub-threshold sample counts are
    flagged by the markdown writer (not here — keep aggregation pure).
    """
    candidate_breakdown_k = 50 if 50 in candidate_ks else (
        max(candidate_ks) if candidate_ks else 50
    )
    diversity_breakdown_k = 10 if 10 in diversity_ks else (
        max(diversity_ks) if diversity_ks else 10
    )
    cand_key = str(candidate_breakdown_k)
    div_key = str(diversity_breakdown_k)

    buckets: Dict[str, List[RetrievalEvalRow]] = {}
    for r in rows:
        bucket = r.query_type if r.query_type else DEFAULT_QUERY_TYPE_UNKNOWN
        buckets.setdefault(bucket, []).append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for k, brows in sorted(buckets.items()):
        latency_vals = [r.retrieval_ms for r in brows if r.error is None]
        cand_hit_vals: List[float] = []
        cand_recall_vals: List[float] = []
        for r in brows:
            if isinstance(r.candidate_hits, dict):
                v = r.candidate_hits.get(cand_key)
                if v is not None:
                    cand_hit_vals.append(float(v))
            if isinstance(r.candidate_recalls, dict):
                v = r.candidate_recalls.get(cand_key)
                if v is not None:
                    cand_recall_vals.append(float(v))
        dup_at_k_vals = [
            float(r.duplicate_doc_ratios[div_key]) for r in brows
            if isinstance(r.duplicate_doc_ratios, dict)
            and r.duplicate_doc_ratios.get(div_key) is not None
        ]
        out[k] = {
            "count": len(brows),
            "hit_at_1": _mean_or_none(
                [r.hit_at_1 for r in brows if r.hit_at_1 is not None]
            ),
            "hit_at_3": _mean_or_none(
                [r.hit_at_3 for r in brows if r.hit_at_3 is not None]
            ),
            "hit_at_5": _mean_or_none(
                [r.hit_at_5 for r in brows if r.hit_at_5 is not None]
            ),
            "mrr_at_10": _mean_or_none(
                [r.mrr_at_10 for r in brows if r.mrr_at_10 is not None]
            ),
            "ndcg_at_10": _mean_or_none(
                [r.ndcg_at_10 for r in brows if r.ndcg_at_10 is not None]
            ),
            f"candidate_hit_at_{candidate_breakdown_k}": _mean_or_none(
                cand_hit_vals
            ),
            f"candidate_recall_at_{candidate_breakdown_k}": _mean_or_none(
                cand_recall_vals
            ),
            "avg_total_retrieval_ms": (
                round(statistics.fmean(latency_vals), 3)
                if latency_vals else None
            ),
            "p95_total_retrieval_ms": (
                round(p_percentile(latency_vals, 95.0), 3)
                if latency_vals else None
            ),
            f"duplicate_doc_ratio_at_{diversity_breakdown_k}": _mean_or_none(
                dup_at_k_vals
            ),
        }
    return out


def _delta_or_none(
    final_value: Optional[float],
    pre_value: Optional[float],
) -> Optional[float]:
    """final - pre when both are populated, else None.

    Used to compute reranker uplift on metrics like hit@k / MRR / NDCG.
    Rounding stays at 6 dp so the signed delta is precise enough for
    the diagnostic threshold (``-0.005``) to compare faithfully.
    """
    if final_value is None or pre_value is None:
        return None
    return round(float(final_value) - float(pre_value), 6)


def compute_retrieval_diagnostics(
    summary: RetrievalEvalSummary,
) -> Dict[str, Any]:
    """Map summary-level signals to the four Phase 1 diagnostic flags.

    Each flag entry carries ``flagged`` (True/False/None — None when the
    underlying metric isn't measurable, so the writer can render
    ``"n/a"`` instead of a falsy "looks fine"), ``threshold`` (the
    constant the comparison used), and ``observed`` (the value the
    summary reported). This 3-tuple shape lets the markdown writer
    quote the threshold without re-typing the constants.

    Diagnostics:
      * ``candidateRecallBottleneck`` — graph candidate-stage missed
        the ``DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50`` floor on
        candidate hit@50.
      * ``rerankerUpliftLow`` — candidate hit@50 cleared the floor but
        rerank uplift on hit@5 is below
        ``DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5``.
      * ``rerankerNegativeUplift`` — uplift on hit@5 OR MRR went
        negative beyond the configured tolerance (small-magnitude
        regressions only count when one of the two metrics moved
        below its threshold).
      * ``highDuplicateRatio`` — the duplicate doc ratio at the top-10
        cutoff is at or above ``DIAG_HIGH_DUPLICATE_RATIO_AT_10``.
    """
    out: Dict[str, Any] = {}

    cand_at_50 = (summary.candidate_hit_rates or {}).get("50")
    out["candidateRecallBottleneck"] = {
        "flagged": (
            None if cand_at_50 is None
            else bool(cand_at_50 < DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50)
        ),
        "threshold": DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50,
        "observed": cand_at_50,
    }

    uplift_h5 = summary.rerank_uplift_hit_at_5
    rerank_low_flag: Optional[bool]
    if cand_at_50 is None or uplift_h5 is None:
        rerank_low_flag = None
    else:
        rerank_low_flag = bool(
            cand_at_50 >= DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50
            and uplift_h5 < DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5
        )
    out["rerankerUpliftLow"] = {
        "flagged": rerank_low_flag,
        "threshold": DIAG_RERANKER_UPLIFT_LOW_HIT_AT_5,
        "observed": uplift_h5,
    }

    uplift_mrr = summary.rerank_uplift_mrr_at_10
    if uplift_h5 is None and uplift_mrr is None:
        rerank_neg_flag: Optional[bool] = None
    else:
        rerank_neg_flag = bool(
            (uplift_h5 is not None
             and uplift_h5 < DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5)
            or (uplift_mrr is not None
                and uplift_mrr < DIAG_RERANKER_NEGATIVE_UPLIFT_MRR_AT_10)
        )
    out["rerankerNegativeUplift"] = {
        "flagged": rerank_neg_flag,
        "threshold": DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5,
        "observed_hit_at_5": uplift_h5,
        "observed_mrr_at_10": uplift_mrr,
    }

    dup_at_10 = (summary.duplicate_doc_ratios or {}).get("10")
    out["highDuplicateRatio"] = {
        "flagged": (
            None if dup_at_10 is None
            else bool(dup_at_10 >= DIAG_HIGH_DUPLICATE_RATIO_AT_10)
        ),
        "threshold": DIAG_HIGH_DUPLICATE_RATIO_AT_10,
        "observed": dup_at_10,
    }
    return out


# ---------------------------------------------------------------------------
# Markdown report writer.
# ---------------------------------------------------------------------------


def render_markdown_report(
    summary: RetrievalEvalSummary,
    rows: List[RetrievalEvalRow],
    duplicate_analysis: DuplicateAnalysis,
) -> str:
    """Compose the human-readable retrieval_eval_report.md.

    Stays concise — the JSON report is the source of truth for any
    automated downstream tooling. The .md is for human eyeballs at
    review time and so a contributor can paste a pasteable summary
    into a PR description.
    """
    lines: List[str] = []
    lines.append("# Retrieval eval report")
    lines.append("")
    lines.append(f"- dataset: `{summary.dataset_path}`")
    if summary.corpus_path:
        lines.append(f"- corpus:  `{summary.corpus_path}`")
    lines.append(f"- rows:    {summary.row_count} (errors: {summary.error_count})")
    lines.append(f"- top_k:   {summary.top_k} (mrr@{summary.mrr_k}, ndcg@{summary.ndcg_k})")
    if summary.embedding_model:
        lines.append(f"- model:   {summary.embedding_model}")
    if summary.index_version:
        lines.append(f"- index:   {summary.index_version}")
    if summary.reranker_name:
        lines.append(f"- reranker: {summary.reranker_name}")
    lines.append(f"- started: {summary.started_at}")
    lines.append(f"- duration_ms: {summary.duration_ms:.1f}")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| hit@1 | {_fmt(summary.mean_hit_at_1)} |")
    lines.append(f"| hit@3 | {_fmt(summary.mean_hit_at_3)} |")
    lines.append(f"| hit@5 | {_fmt(summary.mean_hit_at_5)} |")
    # Extra hit cutoffs (Phase 2A candidate-recall) inserted between
    # hit@5 and MRR — sorted by cutoff so the human-readable order
    # matches the ascending k convention of the rest of the table.
    if summary.mean_extra_hits:
        for key in sorted(summary.mean_extra_hits.keys(), key=_safe_int):
            lines.append(
                f"| hit@{key} | {_fmt(summary.mean_extra_hits[key])} |"
            )
    lines.append(f"| mrr@{summary.mrr_k} | {_fmt(summary.mean_mrr_at_10)} |")
    lines.append(f"| ndcg@{summary.ndcg_k} | {_fmt(summary.mean_ndcg_at_10)} |")
    lines.append(f"| dup_rate (top-{summary.top_k}) | {_fmt(summary.mean_dup_rate)} |")
    lines.append(f"| unique_doc_coverage | {_fmt(summary.mean_unique_doc_coverage)} |")
    lines.append(f"| top1_score_margin | {_fmt(summary.mean_top1_score_margin)} |")
    lines.append(f"| avg_context_token_count | {_fmt(summary.mean_avg_context_token_count)} |")
    lines.append(f"| expected_keyword_match_rate | {_fmt(summary.mean_expected_keyword_match_rate)} |")
    lines.append("")

    lines.append("## Latency (ms)")
    lines.append("")
    lines.append(f"- mean: {summary.mean_retrieval_ms:.2f}")
    lines.append(f"- p50:  {summary.p50_retrieval_ms:.2f}")
    if summary.p90_retrieval_ms is not None:
        lines.append(f"- p90:  {summary.p90_retrieval_ms:.2f}")
    lines.append(f"- p95:  {summary.p95_retrieval_ms:.2f}")
    if summary.p99_retrieval_ms is not None:
        lines.append(f"- p99:  {summary.p99_retrieval_ms:.2f}")
    lines.append(f"- max:  {summary.max_retrieval_ms:.2f}")
    lines.append("")

    if summary.dense_retrieval_row_count > 0:
        lines.append("## Dense retrieval latency (ms)")
        lines.append("")
        lines.append(
            f"- rows with dense_retrieval_ms: {summary.dense_retrieval_row_count}"
        )
        lines.append(f"- mean: {_fmt_ms(summary.mean_dense_retrieval_ms)}")
        lines.append(f"- p50:  {_fmt_ms(summary.p50_dense_retrieval_ms)}")
        lines.append(f"- p90:  {_fmt_ms(summary.p90_dense_retrieval_ms)}")
        lines.append(f"- p95:  {_fmt_ms(summary.p95_dense_retrieval_ms)}")
        lines.append(f"- p99:  {_fmt_ms(summary.p99_dense_retrieval_ms)}")
        lines.append(f"- max:  {_fmt_ms(summary.max_dense_retrieval_ms)}")
        lines.append("")

    if summary.rerank_row_count > 0:
        lines.append("## Rerank latency (ms)")
        lines.append("")
        lines.append(f"- rows with rerank: {summary.rerank_row_count}")
        lines.append(f"- mean: {_fmt_ms(summary.mean_rerank_ms)}")
        lines.append(f"- p50:  {_fmt_ms(summary.p50_rerank_ms)}")
        if summary.p90_rerank_ms is not None:
            lines.append(f"- p90:  {_fmt_ms(summary.p90_rerank_ms)}")
        lines.append(f"- p95:  {_fmt_ms(summary.p95_rerank_ms)}")
        if summary.p99_rerank_ms is not None:
            lines.append(f"- p99:  {_fmt_ms(summary.p99_rerank_ms)}")
        lines.append(f"- max:  {_fmt_ms(summary.max_rerank_ms)}")
        lines.append("")

    if summary.rerank_breakdown_stats:
        lines.append("## Rerank latency breakdown per stage (ms)")
        lines.append("")
        lines.append(
            "| stage | avg | p50 | p90 | p95 | p99 | max | n |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        # Render a stable order so the report diff stays small across
        # runs even when dict iteration order shifts. The known stages
        # come first in their natural pipeline order; any unexpected
        # stage falls to the bottom alphabetically.
        known_order = [
            "pair_build_ms",
            "tokenize_ms",
            "forward_ms",
            "postprocess_ms",
            "total_rerank_ms",
        ]
        ordered_stages = [
            s for s in known_order if s in summary.rerank_breakdown_stats
        ]
        ordered_stages.extend(
            sorted(
                k for k in summary.rerank_breakdown_stats.keys()
                if k not in known_order
            )
        )
        for stage in ordered_stages:
            row = summary.rerank_breakdown_stats[stage]
            n_value = row.get("count")
            try:
                n_int = int(n_value) if n_value is not None else 0
            except (TypeError, ValueError):
                n_int = 0
            lines.append(
                f"| {stage} | "
                f"{_fmt_ms(row.get('avg'))} | "
                f"{_fmt_ms(row.get('p50'))} | "
                f"{_fmt_ms(row.get('p90'))} | "
                f"{_fmt_ms(row.get('p95'))} | "
                f"{_fmt_ms(row.get('p99'))} | "
                f"{_fmt_ms(row.get('max'))} | "
                f"{n_int} |"
            )
        lines.append("")

    if summary.per_answer_type:
        lines.append("## Per answer_type")
        lines.append("")
        lines.append("| answer_type | n | hit@5 | mrr@10 | ndcg@10 |")
        lines.append("|---|---:|---:|---:|---:|")
        for atype, agg in summary.per_answer_type.items():
            lines.append(
                f"| {atype} | {agg['row_count']} | "
                f"{_fmt(agg['mean_hit_at_5'])} | "
                f"{_fmt(agg['mean_mrr_at_10'])} | "
                f"{_fmt(agg['mean_ndcg_at_10'])} |"
            )
        lines.append("")

    if summary.per_difficulty:
        lines.append("## Per difficulty")
        lines.append("")
        lines.append("| difficulty | n | hit@5 | mrr@10 | ndcg@10 |")
        lines.append("|---|---:|---:|---:|---:|")
        for diff, agg in summary.per_difficulty.items():
            lines.append(
                f"| {diff} | {agg['row_count']} | "
                f"{_fmt(agg['mean_hit_at_5'])} | "
                f"{_fmt(agg['mean_mrr_at_10'])} | "
                f"{_fmt(agg['mean_ndcg_at_10'])} |"
            )
        lines.append("")

    # Phase 1 sections: rendered before the (legacy) duplicate-analysis
    # block. They surface only when there's actually data to show — for
    # an older run with no candidate pool / pre-rerank data, the writer
    # silently skips them and produces a byte-identical pre-Phase-1
    # report.
    cand_section = _render_candidate_quality_section(summary)
    if cand_section:
        lines.append(cand_section)
        lines.append("")
    rerank_section = _render_reranker_uplift_section(summary)
    if rerank_section:
        lines.append(rerank_section)
        lines.append("")
    qt_section = _render_query_type_section(summary)
    if qt_section:
        lines.append(qt_section)
        lines.append("")
    div_section = _render_diversity_section(summary)
    if div_section:
        lines.append(div_section)
        lines.append("")
    qe_section = _render_quality_efficiency_section(summary)
    if qe_section:
        lines.append(qe_section)
        lines.append("")
    diag_section = _render_diagnostics_section(summary)
    if diag_section:
        lines.append(diag_section)
        lines.append("")

    lines.append("## Duplicate analysis")
    lines.append("")
    lines.append(
        f"- queries with doc dup:     {duplicate_analysis.queries_with_doc_dup} "
        f"({duplicate_analysis.queries_with_doc_dup_ratio:.3f})"
    )
    lines.append(
        f"- queries with section dup: {duplicate_analysis.queries_with_section_dup} "
        f"({duplicate_analysis.queries_with_section_dup_ratio:.3f})"
    )
    lines.append(
        f"- queries with text dup:    {duplicate_analysis.queries_with_text_dup} "
        f"({duplicate_analysis.queries_with_text_dup_ratio:.3f})"
    )
    if duplicate_analysis.most_common_duplicate_doc_ids:
        lines.append("")
        lines.append("### Most-duplicated doc_ids (extra copies summed across runs)")
        lines.append("")
        for entry in duplicate_analysis.most_common_duplicate_doc_ids:
            lines.append(
                f"- `{entry['doc_id']}` — {entry['extra_copies_total']} extra copies"
            )
    lines.append("")

    # Misses (rows with hit@5 == 0, capped at 20).
    misses = [r for r in rows if r.hit_at_5 == 0.0][:20]
    if misses:
        lines.append("## First misses (hit@5 == 0)")
        lines.append("")
        for r in misses:
            top3 = [
                f"{d}@{s:.3f}" for d, s in zip(r.retrieved_doc_ids[:3], r.retrieval_scores[:3])
            ]
            lines.append(
                f"- `{r.id}` ({r.answer_type or '?'}/{r.difficulty or '?'}): "
                f"{r.query[:80]} → expected {r.expected_doc_ids} "
                f"got [{', '.join(top3)}]"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers (private).
# ---------------------------------------------------------------------------


def _list_of_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.fmean(values), 3)


def _p50_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.median(values), 3)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_ms(value: Optional[float]) -> str:
    """Format an optional millisecond value with 2 dp.

    ``None`` becomes ``"n/a"`` so the markdown writer stays consistent
    with the headline-metrics formatter when a row contributed nothing
    (NoOpReranker path leaves rerank_ms unset).
    """
    return "n/a" if value is None else f"{value:.2f}"


def _safe_int(value: str) -> int:
    """Best-effort int parse for sort keys; falls back to a large sentinel.

    ``mean_extra_hits`` keys are stringified ints (``"10"``, ``"20"``,
    ``"50"``) so the markdown writer needs to sort them numerically
    rather than lexicographically (otherwise ``"100"`` would sort
    between ``"10"`` and ``"20"``). Values that don't parse fall to
    the bottom of the sort — they shouldn't exist in practice but
    silent failure is better than crashing the report writer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1_000_000


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().replace(microsecond=0).isoformat()


def _log_summary(summary: RetrievalEvalSummary) -> None:
    log.info(
        "retrieval eval done: rows=%d errors=%d hit@1=%s hit@5=%s mrr@%d=%s "
        "ndcg@%d=%s dup=%s p95_ms=%.1f",
        summary.row_count, summary.error_count,
        _fmt(summary.mean_hit_at_1), _fmt(summary.mean_hit_at_5),
        summary.mrr_k, _fmt(summary.mean_mrr_at_10),
        summary.ndcg_k, _fmt(summary.mean_ndcg_at_10),
        _fmt(summary.mean_dup_rate), summary.p95_retrieval_ms,
    )


# ---------------------------------------------------------------------------
# Phase 1 markdown report sections.
# ---------------------------------------------------------------------------


def _render_candidate_quality_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Render the Candidate Retrieval Quality table or skip entirely.

    Returns ``None`` when no row in the summary contributed candidate-
    pool data — no key in ``candidate_hit_rates`` is non-None. That
    keeps older runs (no candidate pool surfaced) from getting empty
    sections in the rendered report.
    """
    rates = summary.candidate_hit_rates or {}
    recalls = summary.candidate_recalls or {}
    if not any(v is not None for v in rates.values()):
        if not any(v is not None for v in recalls.values()):
            return None
    lines = ["## Candidate Retrieval Quality", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key in sorted(rates.keys() | recalls.keys(), key=_safe_int):
        lines.append(f"| candidate_hit@{key} | {_fmt(rates.get(key))} |")
        lines.append(f"| candidate_recall@{key} | {_fmt(recalls.get(key))} |")
    cand50 = rates.get("50")
    if cand50 is not None:
        if cand50 < DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50:
            lines.append("")
            lines.append(
                "> _Candidate hit@50 below "
                f"{DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50:.2f} — the "
                "reranker can only ever reorder what the candidate stage "
                "surfaced, so this is a candidate-stage / chunking / "
                "embedding bottleneck, not a reranker problem._"
            )
        else:
            lines.append("")
            lines.append(
                "> _Candidate hit@50 cleared "
                f"{DIAG_CANDIDATE_RECALL_BOTTLENECK_HIT_AT_50:.2f}; if final "
                "hit@5 is still low, that points to the reranker._"
            )
    return "\n".join(lines)


def _render_reranker_uplift_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Render the pre-rerank vs final comparison table or skip.

    Skipped when the run has no rows with expected_doc_ids — uplift is
    undefined in that case.
    """
    if summary.mean_pre_rerank_hit_at_5 is None and summary.mean_hit_at_5 is None:
        return None
    rows = [
        ("hit@1", summary.mean_pre_rerank_hit_at_1, summary.mean_hit_at_1,
         summary.rerank_uplift_hit_at_1),
        ("hit@3", summary.mean_pre_rerank_hit_at_3, summary.mean_hit_at_3,
         summary.rerank_uplift_hit_at_3),
        ("hit@5", summary.mean_pre_rerank_hit_at_5, summary.mean_hit_at_5,
         summary.rerank_uplift_hit_at_5),
        (f"mrr@{summary.mrr_k}", summary.mean_pre_rerank_mrr_at_10,
         summary.mean_mrr_at_10, summary.rerank_uplift_mrr_at_10),
        (f"ndcg@{summary.ndcg_k}", summary.mean_pre_rerank_ndcg_at_10,
         summary.mean_ndcg_at_10, summary.rerank_uplift_ndcg_at_10),
    ]
    lines = ["## Reranker Uplift", ""]
    lines.append("| metric | pre_rerank | final | uplift (final − pre) |")
    lines.append("|---|---|---|---|")
    for label, pre, final, uplift in rows:
        lines.append(
            f"| {label} | {_fmt(pre)} | {_fmt(final)} | {_fmt_signed(uplift)} |"
        )
    neg_h5 = summary.rerank_uplift_hit_at_5
    neg_mrr = summary.rerank_uplift_mrr_at_10
    if (
        (neg_h5 is not None and neg_h5 < DIAG_RERANKER_NEGATIVE_UPLIFT_HIT_AT_5)
        or (neg_mrr is not None
            and neg_mrr < DIAG_RERANKER_NEGATIVE_UPLIFT_MRR_AT_10)
    ):
        lines.append("")
        lines.append(
            "> ⚠️ _Negative reranker uplift detected. The reranker may be "
            "out-of-domain or fed mismatched input formatting._"
        )
    return "\n".join(lines)


def _render_query_type_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Per-query-type breakdown with low-sample-count callouts.

    Skipped when no row contributed a query_type (the breakdown would
    be a single ``unknown`` row that adds no information).
    """
    by_type = summary.by_query_type or {}
    if not by_type:
        return None
    if (
        len(by_type) == 1
        and DEFAULT_QUERY_TYPE_UNKNOWN in by_type
        and by_type[DEFAULT_QUERY_TYPE_UNKNOWN].get("count", 0) == 0
    ):
        return None
    lines = ["## Query Type Breakdown", ""]
    lines.append(
        "| query_type | n | hit@5 | mrr@10 | candidate_hit@50 | "
        "p95_total_ms | dup_ratio@10 | note |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for k in sorted(by_type.keys()):
        agg = by_type[k]
        n = int(agg.get("count", 0) or 0)
        cand_h50 = agg.get("candidate_hit_at_50")
        dup_at_10 = agg.get("duplicate_doc_ratio_at_10")
        note = (
            "low sample count" if n < DEFAULT_LOW_QUERY_TYPE_SAMPLE else ""
        )
        lines.append(
            f"| {k} | {n} | {_fmt(agg.get('hit_at_5'))} | "
            f"{_fmt(agg.get('mrr_at_10'))} | {_fmt(cand_h50)} | "
            f"{_fmt(agg.get('p95_total_retrieval_ms'))} | "
            f"{_fmt(dup_at_10)} | {note} |"
        )
    lines.append("")
    lines.append(
        f"> _Per-query-type metrics with `n < {DEFAULT_LOW_QUERY_TYPE_SAMPLE}` "
        "are noted as low-sample; treat their numbers as anecdotal._"
    )
    return "\n".join(lines)


def _render_diversity_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Diversity / duplication diagnostics block.

    Section is rendered only when at least one diversity / duplicate
    aggregate is populated — older runs without diversity_ks fall
    through cleanly.
    """
    dup = summary.duplicate_doc_ratios or {}
    uniq = summary.unique_doc_counts or {}
    sect = summary.section_diversities or {}
    if (
        not any(v is not None for v in dup.values())
        and not any(v is not None for v in uniq.values())
        and not any(v is not None for v in sect.values())
    ):
        return None
    lines = ["## Diversity / Duplication", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    keys = sorted({*dup.keys(), *uniq.keys(), *sect.keys()}, key=_safe_int)
    for key in keys:
        lines.append(f"| duplicate_doc_ratio@{key} | {_fmt(dup.get(key))} |")
        lines.append(f"| unique_doc_count@{key} | {_fmt(uniq.get(key))} |")
        lines.append(f"| section_diversity@{key} | {_fmt(sect.get(key))} |")
    lines.append("")
    lines.append(
        "> _Diagnostic metrics — high duplicate ratios are not always bad "
        "(e.g. queries that legitimately need multiple chunks of the same "
        "document). Cross-reference with the candidate / reranker section "
        "before concluding._"
    )
    return "\n".join(lines)


def _render_quality_efficiency_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Composite quality / efficiency summary.

    Skipped when ``quality_score`` is None (the run had no rows with
    expected_doc_ids) — the composite is undefined.
    """
    if summary.quality_score is None:
        return None
    lines = ["## Quality / Efficiency Summary", ""]
    lines.append(
        f"- quality_score: {_fmt(summary.quality_score)}  "
        "(0.30·hit@1 + 0.25·hit@5 + 0.25·MRR@10 + 0.20·NDCG@10)"
    )
    lines.append(
        f"- efficiency_score: {_fmt(summary.efficiency_score)}  "
        "(quality_score / log(1 + p95_total_retrieval_ms))"
    )
    lines.append(
        f"- p95_total_retrieval_ms: {_fmt(summary.p95_total_retrieval_ms)}"
    )
    lines.append("")
    lines.append(
        "> _quality_score / efficiency_score are **comparison aids**, not "
        "adoption rules. Adoption decisions still require the conservative "
        "per-metric checks (see eval.harness.agent_loop_ab)._"
    )
    return "\n".join(lines)


def _render_diagnostics_section(
    summary: RetrievalEvalSummary,
) -> Optional[str]:
    """Diagnostics block: which Phase 1 flags fired + next-experiment hints."""
    diag = summary.diagnostics or {}
    if not diag:
        return None
    lines = ["## Diagnostics", ""]
    lines.append("| flag | flagged | threshold | observed |")
    lines.append("|---|:--:|---:|---|")
    flag_rows = [
        ("candidateRecallBottleneck", "candidate hit@50 below floor"),
        ("rerankerUpliftLow", "rerank uplift on hit@5 below floor"),
        ("rerankerNegativeUplift", "rerank uplift on hit@5 / MRR negative"),
        ("highDuplicateRatio", "top-10 duplicate doc ratio above ceiling"),
    ]
    for key, _label in flag_rows:
        entry = diag.get(key) or {}
        flagged = entry.get("flagged")
        threshold = entry.get("threshold")
        if key == "rerankerNegativeUplift":
            obs_h5 = entry.get("observed_hit_at_5")
            obs_mrr = entry.get("observed_mrr_at_10")
            observed = f"hit@5={_fmt(obs_h5)} mrr@10={_fmt(obs_mrr)}"
        else:
            observed = _fmt(entry.get("observed"))
        flag_cell = (
            "n/a" if flagged is None
            else ("YES" if flagged else "no")
        )
        lines.append(
            f"| {key} | {flag_cell} | {_fmt(threshold)} | {observed} |"
        )

    suggestions: List[str] = []
    if diag.get("candidateRecallBottleneck", {}).get("flagged") is True:
        suggestions.append(
            "- candidateRecallBottleneck → "
            "**Next experiment:** section-aware chunking, title/section "
            "prefix, BM25 hybrid retrieval."
        )
    if diag.get("rerankerUpliftLow", {}).get("flagged") is True:
        suggestions.append(
            "- rerankerUpliftLow → "
            "**Next experiment:** reranker input formatting, candidateK "
            "sweep, max_seq_length sweep."
        )
    if diag.get("rerankerNegativeUplift", {}).get("flagged") is True:
        suggestions.append(
            "- rerankerNegativeUplift → "
            "**Next experiment:** swap to a domain-tuned reranker, audit "
            "input formatting (title/section prefix, query/passage "
            "delimiters)."
        )
    if diag.get("highDuplicateRatio", {}).get("flagged") is True:
        suggestions.append(
            "- highDuplicateRatio → "
            "**Next experiment:** MMR or per-doc cap on the top-k."
        )
    by_type = summary.by_query_type or {}
    weak_types = [
        k for k, v in by_type.items()
        if (v.get("count") or 0) >= DEFAULT_LOW_QUERY_TYPE_SAMPLE
        and v.get("hit_at_5") is not None
        and float(v["hit_at_5"]) < 0.50
    ]
    if weak_types:
        suggestions.append(
            "- weak query_type(s) " + ", ".join(sorted(weak_types))
            + " → **Next experiment:** query-type-specific preprocessing "
            "or metadata boost."
        )
    if suggestions:
        lines.append("")
        lines.append("### Next-experiment suggestions")
        lines.append("")
        lines.extend(suggestions)
    return "\n".join(lines)


def _fmt_signed(value: Optional[float]) -> str:
    """Format a signed delta with explicit ``+`` / ``-`` for clarity.

    Used by the rerank-uplift table — a positive uplift renders with a
    leading ``+`` so a quick scan distinguishes "no movement" (``0``)
    from "small lift" (``+0.001``) without re-reading the column.
    """
    if value is None:
        return "n/a"
    if value > 0:
        return f"+{value:.4f}"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Public asdict helpers (for the CLI / writer).
# ---------------------------------------------------------------------------


def summary_to_dict(summary: RetrievalEvalSummary) -> Dict[str, Any]:
    return asdict(summary)


def row_to_dict(row: RetrievalEvalRow) -> Dict[str, Any]:
    return asdict(row)


def dump_row_to_dict(row: TopKDumpRow) -> Dict[str, Any]:
    return asdict(row)


def duplicate_analysis_to_dict(da: DuplicateAnalysis) -> Dict[str, Any]:
    return asdict(da)
