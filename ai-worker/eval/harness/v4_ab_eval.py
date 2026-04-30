"""Phase 7.0 — paired A/B retrieval evaluation between two v4 indexes.

Drives both variant indexes through the same query set and scores
each query with hit@k / mrr@10 / ndcg@10. Then classifies every query
as one of {improved, regressed, unchanged, both_hit, both_missed} on
top-k membership and best rank, and emits the bundle Phase 7.0 asks
for:

  - ab_summary.json + ab_summary.md     — aggregate metrics + verdict
  - per_query_comparison.jsonl          — every query, both variants,
                                          metrics + status
  - improved_queries.jsonl              — status == "improved"
  - regressed_queries.jsonl             — status == "regressed"

Same-title collision is included as a diagnostic: when the top-k
contains multiple chunks whose ``page_title`` collides on the generic
strings Phase 6.3 audits (등장인물 / 평가 / OST / …), the dense ranker
is being asked to disambiguate by section text alone — exactly the
case retrieval_title is supposed to fix.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from app.capabilities.rag.generation import RetrievedChunk
from app.capabilities.rag.retriever import RetrievalReport, Retriever

log = logging.getLogger(__name__)


_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
_DEFAULT_TOP_K = 10
_RANK_MISS = -1


_GENERIC_TITLE_KEYWORDS: frozenset = frozenset({
    "등장인물", "평가", "OST", "기타", "회차", "에피소드", "주제가",
    "음악", "회차 목록", "에피소드 가이드", "미디어 믹스", "기타 등장인물",
    "설정", "줄거리", "스태프", "성우진",
})


@dataclass(frozen=True)
class QueryRecord:
    """Subset of the silver-query schema used during evaluation."""

    qid: str
    query: str
    expected_doc_ids: Tuple[str, ...]
    answer_type: str
    difficulty: str
    bucket: str
    v4_meta: Dict[str, Any] = field(default_factory=dict)


def load_queries(query_path: Path) -> List[QueryRecord]:
    """Load silver queries from JSONL into :class:`QueryRecord`."""
    out: List[QueryRecord] = []
    with Path(query_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            v4_meta = rec.get("v4_meta") or {}
            out.append(QueryRecord(
                qid=str(rec.get("id") or ""),
                query=str(rec.get("query") or ""),
                expected_doc_ids=tuple(rec.get("expected_doc_ids") or []),
                answer_type=str(rec.get("answer_type") or ""),
                difficulty=str(rec.get("difficulty") or ""),
                bucket=str(v4_meta.get("bucket") or ""),
                v4_meta=v4_meta,
            ))
    return out


def _best_rank(
    results: Sequence[RetrievedChunk], expected_doc_ids: Sequence[str],
) -> int:
    """Return 1-indexed rank of the first matching doc, or :data:`_RANK_MISS`."""
    expected = set(expected_doc_ids)
    if not expected or not results:
        return _RANK_MISS
    for i, c in enumerate(results, start=1):
        if c.doc_id in expected:
            return i
    return _RANK_MISS


def _compute_dup_rate(results: Sequence[RetrievedChunk]) -> float:
    if not results:
        return 0.0
    seen: set = set()
    dups = 0
    for c in results:
        if c.doc_id in seen:
            dups += 1
        else:
            seen.add(c.doc_id)
    return dups / len(results)


def _generic_collision_count(sections: Sequence[str]) -> int:
    """Count top-k chunks whose section_path contains a generic token.

    The :class:`RetrievedChunk` type carries ``section`` (the section
    path joined with " > ") but not ``title``, so we approximate the
    "would have been a generic-title collision" signal by checking
    whether the section path starts with one of the Phase 6.3 generic
    tokens (등장인물 / 평가 / OST / …) — these correspond to the
    sub-page sections whose chunks would have ``title="등장인물"``
    style at index time. The metric is a directional indicator of
    how often the retriever is asked to disambiguate Phase 6.3's
    canonical noisy buckets, not a precise per-row title match.
    """
    return sum(
        1 for s in sections
        if any(kw in (s or "") for kw in _GENERIC_TITLE_KEYWORDS)
    )


@dataclass(frozen=True)
class PerQueryMetrics:
    """One side of the A/B for a single query."""

    rank: int
    hit_at: Dict[int, int]
    mrr_at_10: float
    ndcg_at_10: float
    dup_rate: float
    same_title_collisions: int
    top_results: List[Dict[str, Any]] = field(default_factory=list)


def _ndcg_at_10(rank: int) -> float:
    """DCG-style nDCG@10 for binary relevance with one expected doc.

    With a single relevant doc, ideal DCG = 1.0 (relevant at position 1),
    so nDCG@10 reduces to 1 / log2(rank + 1) when 1 ≤ rank ≤ 10, else 0.
    """
    if rank <= 0 or rank > 10:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _per_query_metrics(
    report: RetrievalReport, expected_doc_ids: Sequence[str],
) -> PerQueryMetrics:
    rank = _best_rank(report.results, expected_doc_ids)
    hit_at = {k: int(0 < rank <= k) for k in _K_VALUES}
    mrr10 = (1.0 / rank) if 0 < rank <= 10 else 0.0
    ndcg10 = _ndcg_at_10(rank)
    sections = [c.section for c in report.results]
    coll = _generic_collision_count(sections)
    top_results: List[Dict[str, Any]] = []
    for c in report.results[:_DEFAULT_TOP_K]:
        top_results.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "section": c.section,
            "score": float(c.score) if c.score is not None else None,
        })
    return PerQueryMetrics(
        rank=rank,
        hit_at=hit_at,
        mrr_at_10=mrr10,
        ndcg_at_10=ndcg10,
        dup_rate=_compute_dup_rate(report.results),
        same_title_collisions=coll,
        top_results=top_results,
    )


def _classify(
    baseline: PerQueryMetrics, candidate: PerQueryMetrics,
) -> str:
    """Map (baseline_rank, candidate_rank) → A/B status string."""
    b_hit = 0 < baseline.rank <= _DEFAULT_TOP_K
    c_hit = 0 < candidate.rank <= _DEFAULT_TOP_K
    if not b_hit and not c_hit:
        return "both_missed"
    if not b_hit and c_hit:
        return "improved"
    if b_hit and not c_hit:
        return "regressed"
    # Both hit — compare ranks; equal ranks ⇒ unchanged.
    if candidate.rank < baseline.rank:
        return "improved"
    if candidate.rank > baseline.rank:
        return "regressed"
    return "both_hit"


def _aggregate(rows: Sequence[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """Compute mean hit@k / MRR / nDCG for one side over ``rows``."""
    if not rows:
        return {
            "count": 0,
            **{f"hit_at_{k}": 0.0 for k in _K_VALUES},
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "dup_rate": 0.0,
            "same_title_collisions_avg": 0.0,
        }
    n = len(rows)
    hit_means = {
        f"hit_at_{k}": sum(r[side]["hit_at"][str(k)] for r in rows) / n
        for k in _K_VALUES
    }
    return {
        "count": n,
        **hit_means,
        "mrr_at_10": sum(r[side]["mrr_at_10"] for r in rows) / n,
        "ndcg_at_10": sum(r[side]["ndcg_at_10"] for r in rows) / n,
        "dup_rate": sum(r[side]["dup_rate"] for r in rows) / n,
        "same_title_collisions_avg":
            sum(r[side]["same_title_collisions"] for r in rows) / n,
    }


def _serialise_per_query(
    q: QueryRecord,
    baseline: PerQueryMetrics,
    candidate: PerQueryMetrics,
    status: str,
) -> Dict[str, Any]:
    def side(m: PerQueryMetrics) -> Dict[str, Any]:
        return {
            "rank": m.rank,
            "hit_at": {str(k): m.hit_at[k] for k in _K_VALUES},
            "mrr_at_10": m.mrr_at_10,
            "ndcg_at_10": m.ndcg_at_10,
            "dup_rate": m.dup_rate,
            "same_title_collisions": m.same_title_collisions,
            "top_results": m.top_results,
        }
    return {
        "qid": q.qid,
        "query": q.query,
        "expected_doc_ids": list(q.expected_doc_ids),
        "answer_type": q.answer_type,
        "difficulty": q.difficulty,
        "bucket": q.bucket,
        "v4_meta": q.v4_meta,
        "status": status,
        "baseline": side(baseline),
        "candidate": side(candidate),
    }


def _aggregate_by_bucket(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["bucket"] or "<unbucketed>", []).append(r)
    return {
        bucket: {
            "count": len(rs),
            "baseline": _aggregate(rs, "baseline"),
            "candidate": _aggregate(rs, "candidate"),
            "status_counts": _status_counts(rs),
        }
        for bucket, rs in sorted(by.items())
    }


def _status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {
        "improved": 0, "regressed": 0, "both_hit": 0,
        "both_missed": 0, "unchanged": 0,
    }
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


@dataclass
class AbResult:
    """Container the CLI hands off to the writer."""

    per_query: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)


def run_paired_ab(
    queries: Sequence[QueryRecord],
    *,
    baseline_retriever: Retriever,
    candidate_retriever: Retriever,
    baseline_variant: str,
    candidate_variant: str,
    progress_log_every: int = 25,
) -> AbResult:
    """Drive the same query set through both retrievers and pair results.

    Both retrievers must already be ``ensure_ready``-ed and configured
    with matching ``top_k`` so the per-query metrics are apples-to-apples.
    No reranker is required — the function does not interact with the
    reranker contract directly; whatever the retriever returns is what
    we score.
    """
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        b_report = baseline_retriever.retrieve(q.query)
        c_report = candidate_retriever.retrieve(q.query)
        b_metrics = _per_query_metrics(b_report, q.expected_doc_ids)
        c_metrics = _per_query_metrics(c_report, q.expected_doc_ids)
        status = _classify(b_metrics, c_metrics)
        rows.append(_serialise_per_query(q, b_metrics, c_metrics, status))
        if progress_log_every and (i % progress_log_every == 0):
            log.info("ab progress: %d/%d", i, len(queries))

    aggregate = {
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "n_queries": len(rows),
        "k_values": list(_K_VALUES),
        "baseline": _aggregate(rows, "baseline"),
        "candidate": _aggregate(rows, "candidate"),
        "status_counts": _status_counts(rows),
        "by_bucket": _aggregate_by_bucket(rows),
    }
    return AbResult(per_query=rows, aggregate=aggregate)


def write_ab_outputs(
    result: AbResult,
    *,
    out_dir: Path,
    baseline_variant: str,
    candidate_variant: str,
    summary_json_name: str = "ab_summary.json",
    summary_md_name: str = "ab_summary.md",
    per_query_name: str = "per_query_comparison.jsonl",
    improved_name: str = "improved_queries.jsonl",
    regressed_name: str = "regressed_queries.jsonl",
) -> Dict[str, Path]:
    """Persist the four standard A/B artefacts under ``out_dir``.

    Returns a dict mapping output role → resolved path so the CLI can
    log a tidy table at the end.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / summary_json_name
    summary_md = out_dir / summary_md_name
    per_query = out_dir / per_query_name
    improved = out_dir / improved_name
    regressed = out_dir / regressed_name

    summary_json.write_text(
        json.dumps(result.aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(
        _render_summary_md(result.aggregate),
        encoding="utf-8",
    )
    with per_query.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with improved.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            if r["status"] == "improved":
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with regressed.open("w", encoding="utf-8") as fp:
        for r in result.per_query:
            if r["status"] == "regressed":
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "per_query": per_query,
        "improved": improved,
        "regressed": regressed,
    }


def _render_summary_md(agg: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(
        f"# Phase 7.0 A/B — "
        f"{agg['baseline_variant']} vs {agg['candidate_variant']}"
    )
    lines.append("")
    lines.append(f"- n_queries: **{agg['n_queries']}**")
    lines.append(f"- k_values: {agg['k_values']}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append(
        "| metric | baseline | candidate | Δ (cand − base) |"
    )
    lines.append("|---|---:|---:|---:|")
    base = agg["baseline"]
    cand = agg["candidate"]
    for k in agg["k_values"]:
        b = base[f"hit_at_{k}"]
        c = cand[f"hit_at_{k}"]
        lines.append(
            f"| hit@{k} | {b:.4f} | {c:.4f} | {c-b:+.4f} |"
        )
    for key in ("mrr_at_10", "ndcg_at_10", "dup_rate",
                "same_title_collisions_avg"):
        b = base[key]
        c = cand[key]
        lines.append(
            f"| {key} | {b:.4f} | {c:.4f} | {c-b:+.4f} |"
        )
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    for k, v in sorted(agg["status_counts"].items()):
        lines.append(f"- {k}: **{v}**")
    lines.append("")
    lines.append("## By bucket")
    lines.append("")
    for bucket, payload in agg.get("by_bucket", {}).items():
        lines.append(f"### {bucket} (n={payload['count']})")
        lines.append("")
        lines.append(
            "| metric | baseline | candidate | Δ |"
        )
        lines.append("|---|---:|---:|---:|")
        for k in agg["k_values"]:
            b = payload["baseline"][f"hit_at_{k}"]
            c = payload["candidate"][f"hit_at_{k}"]
            lines.append(
                f"| hit@{k} | {b:.4f} | {c:.4f} | {c-b:+.4f} |"
            )
        for key in ("mrr_at_10", "ndcg_at_10"):
            b = payload["baseline"][key]
            c = payload["candidate"][key]
            lines.append(
                f"| {key} | {b:.4f} | {c:.4f} | {c-b:+.4f} |"
            )
        lines.append("")
        lines.append(
            "- status: " +
            ", ".join(f"{k}={v}" for k, v in sorted(
                payload["status_counts"].items()
            ))
        )
        lines.append("")
    return "\n".join(lines) + "\n"
