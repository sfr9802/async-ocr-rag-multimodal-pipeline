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
    dup_rate,
    expected_keyword_match_rate,
    hit_at_k,
    ndcg_at_k,
    normalized_text_hash,
    p_percentile,
    reciprocal_rank_at_k,
    top1_score_margin,
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

        expected_doc_ids = _list_of_str(raw.get("expected_doc_ids"))
        expected_keywords = _list_of_str(raw.get("expected_section_keywords"))
        tags = _list_of_str(raw.get("tags"))

        row = RetrievalEvalRow(
            id=query_id,
            query=query_text,
            language=str(raw["language"]) if raw.get("language") else None,
            expected_doc_ids=expected_doc_ids,
            expected_section_keywords=expected_keywords,
            answer_type=str(raw["answer_type"]) if raw.get("answer_type") else None,
            difficulty=str(raw["difficulty"]) if raw.get("difficulty") else None,
            tags=tags,
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
            row.index_version = getattr(report, "index_version", None)
            row.embedding_model = getattr(report, "embedding_model", None)
            row.reranker_name = getattr(report, "reranker_name", None)
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
            row.expected_keyword_match_rate = expected_keyword_match_rate(
                chunk_texts, expected_keywords
            )

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

    return RetrievalEvalSummary(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        row_count=len(rows),
        rows_with_expected_doc_ids=len(h5),
        rows_with_expected_keywords=len(kwm),
        top_k=top_k,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
        mean_hit_at_1=_mean_or_none(h1),
        mean_hit_at_3=_mean_or_none(h3),
        mean_hit_at_5=_mean_or_none(h5),
        mean_mrr_at_10=_mean_or_none(mrr),
        mean_ndcg_at_10=_mean_or_none(ndcg),
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
    )


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
