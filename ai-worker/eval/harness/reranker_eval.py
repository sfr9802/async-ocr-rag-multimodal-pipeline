"""Phase 2A reranker evaluation helpers.

This module owns the bookkeeping that lives *around* a retrieval-eval
run when a cross-encoder reranker is in play: building the comparison
table that joins multiple ``retrieval_eval_report.json`` artifacts
together, and the failure-analysis pass that buckets queries by how
they moved when the reranker was applied.

The actual retrieval scoring is still done by ``retrieval_eval.py``
— Phase 2A reranker runs are just ordinary retrieval-eval runs with
``rerank_ms`` populated and the ``reranker_name`` field set to
something other than ``noop``. Splitting the post-hoc analysis out
into this module keeps ``retrieval_eval.py`` focused on per-row
metrics over a single retriever and avoids growing it into a
multi-run aggregator.

What lives here:

  - load_retrieval_report(path) -> dict
    Reads + lightly normalises a retrieval_eval_report.json so the
    comparison and failure-analysis passes can take dicts of the
    same shape regardless of which run produced them.

  - build_reranker_comparison(slices) -> dict
    Assembles a side-by-side table of headline metrics + provenance
    across N labelled retrieval reports. Pure post-processing —
    never re-runs retrieval, never re-embeds.

  - render_reranker_comparison_markdown(comparison) -> str
    Renders the comparison dict as a one-page Markdown report with
    the caveats Phase 2A wants surfaced (candidate-recall ceiling,
    candidate-population mismatch between B1 and B2, etc.).

  - build_reranker_failure_analysis(dense_dump, rerank_dump,
        rows_by_query, k_preview) -> dict
    Cross-tabs hit@1 dense vs. hit@1 rerank to pick out the three
    diagnostic groups asked for in the spec:
      - dense-miss → rerank-hit (rerank rescued the query)
      - dense-hit  → rerank-miss (rerank regressed the query)
      - both miss               (candidate set never had the answer)
    For each bucket, dumps capped per-query samples with the
    expected docs, dense top-N, reranked top-N, and a short chunk
    preview so humans can eyeball what's happening without
    re-running retrieval.

  - render_reranker_failure_markdown(analysis) -> str
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison report.
# ---------------------------------------------------------------------------


@dataclass
class _ReportSlice:
    """One row in the comparison table.

    Splitting this out from ``Dict[str, Any]`` keeps the writer honest
    about which fields it expects — Phase 2A wants exactly these on
    every row, and any new field has to land in this dataclass before
    the markdown writer can read it.
    """

    label: str
    report_path: str
    corpus_path: Optional[str]
    reranker_name: Optional[str]
    embedding_model: Optional[str]
    index_version: Optional[str]
    dense_top_n: Optional[int]
    final_top_k: Optional[int]
    reranker_batch_size: Optional[int]
    reranker_model: Optional[str]
    row_count: int
    rows_with_expected_doc_ids: int
    mean_hit_at_1: Optional[float]
    mean_hit_at_3: Optional[float]
    mean_hit_at_5: Optional[float]
    mean_mrr_at_10: Optional[float]
    mean_ndcg_at_10: Optional[float]
    mean_dup_rate: Optional[float]
    mean_avg_context_token_count: Optional[float]
    mean_extra_hits: Dict[str, Optional[float]]
    candidate_recall_at_20: Optional[float]
    candidate_recall_at_50: Optional[float]
    mean_retrieval_ms: float
    p95_retrieval_ms: float
    rerank_row_count: int
    mean_rerank_ms: Optional[float]
    p95_rerank_ms: Optional[float]


def load_retrieval_report(path: Path) -> Dict[str, Any]:
    """Read a retrieval_eval_report.json and return the dict as-is.

    Tiny wrapper purely so the comparison writer doesn't have to know
    about file IO. Raises FileNotFoundError with a helpful message if
    the path doesn't exist — this is a CLI-driven flow and silent
    failures are worse than loud ones.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Retrieval report not found: {p}. Did the retrieval-eval "
            f"or retrieval-rerank run finish? Expected the standard "
            f"retrieval_eval_report.json artifact in the run's out-dir."
        )
    with p.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _slice_from_report(
    label: str,
    path: Path,
    report: Mapping[str, Any],
) -> _ReportSlice:
    """Project the report dict down to the comparison row shape.

    Pulls a few extra knobs out of the run's ``metadata`` block when
    available — those are written by the retrieval-rerank CLI and are
    the only place dense_top_n / reranker_batch_size live (they are
    reranker-config, not summary stats).
    """
    summary = dict(report.get("summary") or {})
    metadata = dict(report.get("metadata") or {})

    extra = dict(summary.get("mean_extra_hits") or {})
    candidate_recall_at_20 = extra.get("20") if extra else None
    candidate_recall_at_50 = extra.get("50") if extra else None

    return _ReportSlice(
        label=label,
        report_path=str(path),
        corpus_path=summary.get("corpus_path") or metadata.get("corpus_path"),
        reranker_name=summary.get("reranker_name"),
        embedding_model=summary.get("embedding_model"),
        index_version=summary.get("index_version"),
        dense_top_n=metadata.get("dense_top_n") or metadata.get("candidate_k"),
        final_top_k=metadata.get("final_top_k") or summary.get("top_k"),
        reranker_batch_size=metadata.get("reranker_batch_size"),
        reranker_model=metadata.get("reranker_model"),
        row_count=int(summary.get("row_count") or 0),
        rows_with_expected_doc_ids=int(summary.get("rows_with_expected_doc_ids") or 0),
        mean_hit_at_1=summary.get("mean_hit_at_1"),
        mean_hit_at_3=summary.get("mean_hit_at_3"),
        mean_hit_at_5=summary.get("mean_hit_at_5"),
        mean_mrr_at_10=summary.get("mean_mrr_at_10"),
        mean_ndcg_at_10=summary.get("mean_ndcg_at_10"),
        mean_dup_rate=summary.get("mean_dup_rate"),
        mean_avg_context_token_count=summary.get("mean_avg_context_token_count"),
        mean_extra_hits=extra,
        candidate_recall_at_20=candidate_recall_at_20,
        candidate_recall_at_50=candidate_recall_at_50,
        mean_retrieval_ms=float(summary.get("mean_retrieval_ms") or 0.0),
        p95_retrieval_ms=float(summary.get("p95_retrieval_ms") or 0.0),
        rerank_row_count=int(summary.get("rerank_row_count") or 0),
        mean_rerank_ms=summary.get("mean_rerank_ms"),
        p95_rerank_ms=summary.get("p95_rerank_ms"),
    )


def build_reranker_comparison(
    slices: List[Tuple[str, Path]],
    *,
    caveats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble a comparison dict from N labelled retrieval reports.

    The caller passes ``[(label, path), ...]`` — labels become the
    first column of the comparison table. Order is preserved so the
    user-facing report reads in dataset/run order rather than
    alphabetical-by-label order.
    """
    loaded: List[_ReportSlice] = []
    for label, path in slices:
        report = load_retrieval_report(Path(path))
        loaded.append(_slice_from_report(label, Path(path), report))

    return {
        "schema": "phase2a-reranker-comparison.v1",
        "slice_count": len(loaded),
        "slices": [asdict(s) for s in loaded],
        "caveats": list(caveats or _default_caveats()),
    }


def _default_caveats() -> List[str]:
    return [
        "reranker는 candidate set 안의 순서만 바꾼다 — candidate에 정답이 없으면 회복할 수 없다.",
        "candidate_recall@N (= dense-only top-N hit@N) 이 reranker 성능 상한이다.",
        "dense top-N을 키우면 latency / GPU memory 비용이 증가한다 — 회복 가능한 query 수와 비교해서 결정할 것.",
        "B1 (combined-old) 와 B2 (combined-token-aware-v1) 는 chunk granularity가 다르다 — candidate population이 동일하지 않다.",
        "rerank_latency 는 cross-encoder predict 만의 wall-clock — bi-encoder + FAISS 부분은 mean_retrieval_ms 에 별도로 잡혀 있다.",
        "이 리포트는 어떤 설정도 production default 로 승격하지 않는다 — 결과는 evidence, 결정은 별도.",
    ]


def render_reranker_comparison_markdown(
    comparison: Mapping[str, Any],
) -> str:
    """Render the comparison dict as a one-page Markdown report.

    Concise on purpose — the JSON is the source of truth for any
    downstream tooling. The .md is for eyeballing the run set in a
    PR description or review notebook.
    """
    lines: List[str] = []
    lines.append("# Phase 2A reranker comparison")
    lines.append("")
    lines.append(f"- slices: {comparison.get('slice_count', 0)}")
    lines.append("")

    caveats = list(comparison.get("caveats") or [])
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    slices = list(comparison.get("slices") or [])
    if not slices:
        lines.append("_No slices loaded._")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Headline metrics")
    lines.append("")
    lines.append(
        "| label | corpus | reranker | dense_top_n | final_top_k | "
        "hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for s in slices:
        lines.append(
            f"| {s['label']} | "
            f"{_short_corpus(s.get('corpus_path'))} | "
            f"{s.get('reranker_name') or 'noop'} | "
            f"{_int_or_dash(s.get('dense_top_n'))} | "
            f"{_int_or_dash(s.get('final_top_k'))} | "
            f"{_fmt_or_dash(s.get('mean_hit_at_1'))} | "
            f"{_fmt_or_dash(s.get('mean_hit_at_3'))} | "
            f"{_fmt_or_dash(s.get('mean_hit_at_5'))} | "
            f"{_fmt_or_dash(s.get('mean_mrr_at_10'))} | "
            f"{_fmt_or_dash(s.get('mean_ndcg_at_10'))} |"
        )
    lines.append("")

    # Candidate-recall companion table — only meaningful when at least
    # one slice has extra hit cutoffs populated, which the
    # retrieval-rerank CLI does for the dense-only baseline.
    has_extra = any(
        bool(s.get("mean_extra_hits")) for s in slices
    )
    if has_extra:
        cutoffs = sorted(
            {
                int(k) for s in slices
                for k in (s.get("mean_extra_hits") or {}).keys()
            }
        )
        if cutoffs:
            lines.append("## Candidate / extra hit cutoffs")
            lines.append("")
            lines.append(
                "| label | "
                + " | ".join(f"hit@{k}" for k in cutoffs)
                + " |"
            )
            lines.append(
                "|---" + "|---:" * len(cutoffs) + "|"
            )
            for s in slices:
                cells = []
                extras = s.get("mean_extra_hits") or {}
                for k in cutoffs:
                    cells.append(_fmt_or_dash(extras.get(str(k))))
                lines.append(f"| {s['label']} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("## Latency + cost")
    lines.append("")
    lines.append(
        "| label | reranker_batch | retrieval_p95_ms | "
        "rerank_p95_ms | mean_avg_ctx_tokens | dup_rate (top-k) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )
    for s in slices:
        lines.append(
            f"| {s['label']} | "
            f"{_int_or_dash(s.get('reranker_batch_size'))} | "
            f"{s.get('p95_retrieval_ms', 0.0):.2f} | "
            f"{_fmt_ms_or_dash(s.get('p95_rerank_ms'))} | "
            f"{_fmt_or_dash(s.get('mean_avg_context_token_count'))} | "
            f"{_fmt_or_dash(s.get('mean_dup_rate'))} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Failure analysis.
# ---------------------------------------------------------------------------


@dataclass
class _FailureSample:
    query_id: str
    query: str
    expected_doc_ids: List[str]
    expected_section_keywords: List[str]
    answer_type: Optional[str]
    difficulty: Optional[str]
    dense_top: List[Dict[str, Any]] = field(default_factory=list)
    rerank_top: List[Dict[str, Any]] = field(default_factory=list)


def _index_dump_by_query(
    dump_rows: Iterable[Mapping[str, Any]],
    *,
    k_preview: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group top_k_dump rows by query_id, keeping the first k_preview ranks.

    The retrieval CLI writes dump rows pre-sorted by (query, rank); we
    rely on that order rather than re-sorting because the rerank-only
    metric ``rerank_score`` is already on the row at write time.
    """
    by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in dump_rows:
        qid = str(row.get("query_id") or "").strip()
        if not qid:
            continue
        if len(by_query[qid]) >= k_preview:
            continue
        by_query[qid].append({
            "rank": row.get("rank"),
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "section_path": row.get("section_path"),
            "score": row.get("score"),
            "rerank_score": row.get("rerank_score"),
            "normalized_score": row.get("normalized_score"),
            "is_expected_doc": bool(row.get("is_expected_doc")),
            "chunk_preview": row.get("chunk_preview"),
        })
    return by_query


def build_reranker_failure_analysis(
    *,
    dense_rows: List[Mapping[str, Any]],
    rerank_rows: List[Mapping[str, Any]],
    dense_dump: List[Mapping[str, Any]],
    rerank_dump: List[Mapping[str, Any]],
    k_preview: int = 5,
    sample_cap: int = 10,
) -> Dict[str, Any]:
    """Cross-tab dense hit@1 vs. rerank hit@1 and capture per-bucket samples.

    Inputs are the raw dict rows from ``retrieval_eval_report.json``
    plus the matching ``top_k_dump.jsonl`` records for each run. The
    function:

      1. Joins rows by ``id`` (the eval-query id, which is stable
         across the two runs because the dataset is the same).
      2. Buckets each query into one of:
         - ``dense_miss_to_rerank_hit``
         - ``dense_hit_to_rerank_miss``
         - ``both_hit``
         - ``both_miss``
      3. For the three diagnostic buckets (i.e. excluding both_hit),
         emits up to ``sample_cap`` examples each, with the dense
         top-``k_preview`` and rerank top-``k_preview`` lifted off
         the dump files.

    No retrieval is re-run. Latency-free, dependency-free, takes a few
    hundred ms on the silver-200 dataset.
    """
    dense_by_id = {str(r.get("id")): r for r in dense_rows if r.get("id")}
    rerank_by_id = {str(r.get("id")): r for r in rerank_rows if r.get("id")}
    dense_dump_by_q = _index_dump_by_query(dense_dump, k_preview=k_preview)
    rerank_dump_by_q = _index_dump_by_query(rerank_dump, k_preview=k_preview)

    buckets: Dict[str, List[_FailureSample]] = {
        "dense_miss_to_rerank_hit": [],
        "dense_hit_to_rerank_miss": [],
        "both_hit": [],
        "both_miss": [],
    }
    counts: Dict[str, int] = {k: 0 for k in buckets.keys()}

    for qid, drow in dense_by_id.items():
        rrow = rerank_by_id.get(qid)
        if rrow is None:
            # Different dataset coverage between runs — flag in the
            # summary but skip from buckets so we don't mis-attribute
            # a no-overlap case to "both miss".
            continue
        dense_h1 = drow.get("hit_at_1")
        rerank_h1 = rrow.get("hit_at_1")
        if dense_h1 is None or rerank_h1 is None:
            continue
        bucket: Optional[str] = None
        if dense_h1 == 0.0 and rerank_h1 == 1.0:
            bucket = "dense_miss_to_rerank_hit"
        elif dense_h1 == 1.0 and rerank_h1 == 0.0:
            bucket = "dense_hit_to_rerank_miss"
        elif dense_h1 == 1.0 and rerank_h1 == 1.0:
            bucket = "both_hit"
        else:
            bucket = "both_miss"
        counts[bucket] += 1
        if bucket == "both_hit":
            continue  # Sampled bucket coverage is for diagnostic groups only.
        if len(buckets[bucket]) >= sample_cap:
            continue
        buckets[bucket].append(
            _FailureSample(
                query_id=qid,
                query=str(drow.get("query") or rrow.get("query") or ""),
                expected_doc_ids=list(drow.get("expected_doc_ids") or []),
                expected_section_keywords=list(
                    drow.get("expected_section_keywords") or []
                ),
                answer_type=drow.get("answer_type"),
                difficulty=drow.get("difficulty"),
                dense_top=list(dense_dump_by_q.get(qid, [])),
                rerank_top=list(rerank_dump_by_q.get(qid, [])),
            )
        )

    return {
        "schema": "phase2a-reranker-failure-analysis.v1",
        "k_preview": k_preview,
        "sample_cap_per_bucket": sample_cap,
        "row_overlap_count": len(set(dense_by_id.keys()) & set(rerank_by_id.keys())),
        "bucket_counts": counts,
        "buckets": {
            name: [asdict(s) for s in samples]
            for name, samples in buckets.items()
        },
    }


def render_reranker_failure_markdown(analysis: Mapping[str, Any]) -> str:
    """Render the failure-analysis dict as Markdown.

    The dump groups top-N chunks per query so a reviewer can scan the
    rescued-vs-regressed cases without opening the JSONL files. We
    cap chunk previews at the value the dump itself uses (already
    ``PREVIEW_CHARS`` in retrieval_eval) so this never blows up the
    Markdown size.
    """
    lines: List[str] = []
    lines.append("# Phase 2A reranker failure analysis")
    lines.append("")
    lines.append(f"- query overlap (dense ∩ rerank): {analysis.get('row_overlap_count', 0)}")
    counts = dict(analysis.get("bucket_counts") or {})
    if counts:
        lines.append("- bucket counts:")
        for k, v in counts.items():
            lines.append(f"  - {k}: {v}")
    lines.append(f"- sample cap per bucket: {analysis.get('sample_cap_per_bucket', 0)}")
    lines.append(f"- top-{analysis.get('k_preview', 5)} previews per sample")
    lines.append("")

    bucket_titles = {
        "dense_miss_to_rerank_hit":
            "## dense-miss → rerank-hit (rerank rescued)",
        "dense_hit_to_rerank_miss":
            "## dense-hit → rerank-miss (rerank regressed)",
        "both_miss":
            "## both miss (candidate set never had answer)",
    }
    buckets = analysis.get("buckets") or {}
    for name, title in bucket_titles.items():
        samples = list(buckets.get(name) or [])
        lines.append(title)
        lines.append("")
        if not samples:
            lines.append("_No samples in this bucket._")
            lines.append("")
            continue
        for s in samples:
            lines.extend(_render_failure_sample(s))
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_failure_sample(sample: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"### `{sample.get('query_id')}` "
        f"({sample.get('answer_type') or '?'} / "
        f"{sample.get('difficulty') or '?'})"
    )
    lines.append("")
    lines.append(f"- query: {sample.get('query') or ''}")
    expected = sample.get("expected_doc_ids") or []
    if expected:
        lines.append(f"- expected_doc_ids: {expected}")
    expected_kw = sample.get("expected_section_keywords") or []
    if expected_kw:
        lines.append(f"- expected_section_keywords: {expected_kw}")
    lines.append("")
    lines.append("**Dense top-N**")
    lines.append("")
    lines.extend(_render_dump_table(sample.get("dense_top") or []))
    lines.append("")
    lines.append("**Reranked top-N**")
    lines.append("")
    lines.extend(_render_dump_table(sample.get("rerank_top") or []))
    lines.append("")
    return lines


def _render_dump_table(rows: List[Mapping[str, Any]]) -> List[str]:
    if not rows:
        return ["_(empty)_"]
    out = [
        "| rank | doc_id | section | dense_score | rerank_score | match | preview |",
        "|---:|---|---|---:|---:|:-:|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.get('rank') or '?'} | "
            f"`{r.get('doc_id') or '?'}` | "
            f"{_short_section(r.get('section_path'))} | "
            f"{_fmt_or_dash(r.get('score'))} | "
            f"{_fmt_or_dash(r.get('rerank_score'))} | "
            f"{'✓' if r.get('is_expected_doc') else ''} | "
            f"{_truncate_md(r.get('chunk_preview') or '')} |"
        )
    return out


# ---------------------------------------------------------------------------
# Tiny formatting helpers (private).
# ---------------------------------------------------------------------------


def _fmt_or_dash(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_ms_or_dash(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _int_or_dash(value: Optional[int]) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _short_corpus(path: Optional[str]) -> str:
    """Shorten a corpus_path to its filename for the table column.

    Full paths blow the column width up. The leaf filename is enough
    to disambiguate (B1 vs B2 vs raw) at this surface level — the
    full path is in the JSON for anyone who needs it.
    """
    if not path:
        return "-"
    return Path(path).name


def _short_section(section: Optional[str]) -> str:
    if not section:
        return "-"
    s = str(section)
    return s if len(s) <= 30 else s[:27] + "…"


def _truncate_md(text: str, *, limit: int = 120) -> str:
    """Truncate + escape pipe characters so chunk previews don't break tables.

    Markdown tables are pipe-delimited, so an unescaped pipe inside a
    chunk preview breaks the row layout. We escape with a backslash
    and cap to the per-cell limit so the failure-analysis report
    stays readable in a side scroll.
    """
    cleaned = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: max(0, limit - 1)].rstrip() + "…"
    return cleaned
