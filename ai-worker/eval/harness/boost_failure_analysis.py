"""Phase 2B boost-vs-baseline failure analysis.

Cross-tabulates per-query outcomes across three pipelines (dense,
boost, boost+rerank) into five labelled groups so a reviewer can
inspect exactly where the boost helped or hurt.

The five groups (a query lands in exactly one):

  - ``dense_miss_to_boost_hit``      Dense top-K missed; boost top-K
                                     hit. The "rescue" success case.
  - ``dense_hit_to_boost_miss``      Dense top-K hit; boost top-K
                                     missed. The "regression" case
                                     boost introduced.
  - ``boost_hit_rerank_hit``         Boost top-K hit AND post-rerank
                                     top-K hit. Pipeline kept the
                                     correct doc through the second
                                     stage.
  - ``boost_hit_rerank_miss``        Boost top-K hit but the cross-
                                     encoder pushed the correct doc
                                     out — a boost-stage win that
                                     the reranker failed to honor.
  - ``both_miss``                    Both stages missed. The hardest
                                     queries — typically a corpus
                                     coverage gap.

The analyzer reads three sources, each shaped like the existing
retrieval-eval row dicts so any of the Phase 2A or 2B reports is
acceptable as input. The reranker stage is optional: when omitted,
the report only emits the rescue / regression buckets.

Inputs are query-id-keyed maps to avoid order assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eval.harness.metrics import hit_at_k


GROUP_DENSE_MISS_BOOST_HIT = "dense_miss_to_boost_hit"
GROUP_DENSE_HIT_BOOST_MISS = "dense_hit_to_boost_miss"
GROUP_BOOST_HIT_RERANK_HIT = "boost_hit_rerank_hit"
GROUP_BOOST_HIT_RERANK_MISS = "boost_hit_rerank_miss"
GROUP_BOTH_MISS = "both_miss"
GROUP_NEUTRAL = "neutral"  # both dense + boost agree on hit/miss

GROUP_ORDER: Tuple[str, ...] = (
    GROUP_DENSE_MISS_BOOST_HIT,
    GROUP_DENSE_HIT_BOOST_MISS,
    GROUP_BOOST_HIT_RERANK_HIT,
    GROUP_BOOST_HIT_RERANK_MISS,
    GROUP_BOTH_MISS,
    GROUP_NEUTRAL,
)

DEFAULT_SAMPLE_LIMIT = 30


@dataclass
class FailureSampleEntry:
    """Per-query sample with all three pipelines' top-K signals.

    ``dense_top5`` / ``boost_top5`` / ``rerank_top5`` keep the doc
    ids and best-effort scores so a reviewer can see at a glance how
    the query moved through the pipeline. We cap at top-5 in the
    sample regardless of the ``top_k`` cutoff used to bucketise.
    """

    query_id: str
    query: str
    expected_doc_ids: List[str]
    expected_section_keywords: List[str]
    answer_type: Optional[str]
    difficulty: Optional[str]
    dense_top5_doc_ids: List[str] = field(default_factory=list)
    dense_top5_scores: List[float] = field(default_factory=list)
    boost_top5_doc_ids: List[str] = field(default_factory=list)
    boost_top5_scores: List[float] = field(default_factory=list)
    boost_top5_boost_scores: List[float] = field(default_factory=list)
    rerank_top5_doc_ids: List[str] = field(default_factory=list)
    rerank_top5_scores: List[float] = field(default_factory=list)
    sections_top5: List[str] = field(default_factory=list)
    chunk_previews_top5: List[str] = field(default_factory=list)


@dataclass
class FailureGroupStats:
    name: str
    count: int
    ratio: float


@dataclass
class BoostFailureAnalysis:
    schema: str = "phase2b-boost-failure-analysis.v1"
    top_k: int = 0
    queries_evaluated: int = 0
    queries_skipped: int = 0
    skip_reason: str = ""
    groups: List[FailureGroupStats] = field(default_factory=list)
    samples: Dict[str, List[FailureSampleEntry]] = field(default_factory=dict)


def _doc_ids(row: Mapping[str, Any], k: int) -> List[str]:
    return [str(d) for d in (row.get("retrieved_doc_ids") or [])[:k]]


def _scores(row: Mapping[str, Any], k: int) -> List[float]:
    out: List[float] = []
    for s in (row.get("retrieval_scores") or [])[:k]:
        try:
            out.append(round(float(s), 6))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _row_hit(row: Mapping[str, Any], expected: Sequence[str], k: int) -> bool:
    if not expected:
        return False
    score = hit_at_k(_doc_ids(row, k), list(expected), k=k)
    return bool(score and score > 0.0)


def _expected_for(row: Mapping[str, Any]) -> List[str]:
    return [str(d) for d in (row.get("expected_doc_ids") or []) if d]


def _build_sample(
    *,
    query_id: str,
    base_row: Mapping[str, Any],
    dense_row: Optional[Mapping[str, Any]],
    boost_row: Optional[Mapping[str, Any]],
    rerank_row: Optional[Mapping[str, Any]],
    boost_dump_for_query: Sequence[Mapping[str, Any]] = (),
) -> FailureSampleEntry:
    """Compose a sample with top-5 snapshots from each pipeline."""
    boost_breakdown_by_chunk: Dict[str, float] = {}
    for entry in boost_dump_for_query:
        if entry.get("stage") in ("boosted", "final"):
            cid = entry.get("chunk_id") or ""
            if cid and cid not in boost_breakdown_by_chunk:
                try:
                    boost_breakdown_by_chunk[cid] = round(
                        float(entry.get("boost_score") or 0.0), 6
                    )
                except (TypeError, ValueError):
                    boost_breakdown_by_chunk[cid] = 0.0

    boost_top5_boost_scores: List[float] = []
    if boost_row is not None:
        chunk_ids = (boost_row.get("retrieved_chunk_ids") or [])[:5]
        for cid in chunk_ids:
            boost_top5_boost_scores.append(
                round(float(boost_breakdown_by_chunk.get(str(cid), 0.0)), 6)
            )

    sections_src = boost_row or rerank_row or dense_row or base_row
    sections = [
        str(s) for s in
        (sections_src.get("retrieved_chunk_ids") or [])[:5]  # placeholder
    ]

    return FailureSampleEntry(
        query_id=query_id,
        query=str(base_row.get("query") or ""),
        expected_doc_ids=_expected_for(base_row),
        expected_section_keywords=[
            str(k) for k in (base_row.get("expected_section_keywords") or [])
        ],
        answer_type=base_row.get("answer_type"),
        difficulty=base_row.get("difficulty"),
        dense_top5_doc_ids=_doc_ids(dense_row or {}, 5),
        dense_top5_scores=_scores(dense_row or {}, 5),
        boost_top5_doc_ids=_doc_ids(boost_row or {}, 5),
        boost_top5_scores=_scores(boost_row or {}, 5),
        boost_top5_boost_scores=boost_top5_boost_scores,
        rerank_top5_doc_ids=_doc_ids(rerank_row or {}, 5),
        rerank_top5_scores=_scores(rerank_row or {}, 5),
        sections_top5=sections,
        chunk_previews_top5=[],
    )


def classify_boost_failures(
    *,
    dense_rows: Sequence[Mapping[str, Any]],
    boost_rows: Sequence[Mapping[str, Any]],
    rerank_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    top_k: int = 10,
    boost_dump: Optional[Sequence[Mapping[str, Any]]] = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> BoostFailureAnalysis:
    """Group queries by their dense → boost → rerank trajectory.

    Each input row is keyed by ``id`` so we can join across the three
    pipelines without assuming ordered alignment. Queries missing
    from any required pipeline (``dense`` and ``boost`` are both
    required) are counted in ``queries_skipped``.

    ``boost_dump`` is optional; when provided, sample entries carry
    per-chunk boost scores so a reviewer can see how much boost
    each candidate received.
    """
    dense_by_id = {
        str(r.get("id") or ""): r for r in dense_rows if r.get("id")
    }
    boost_by_id = {
        str(r.get("id") or ""): r for r in boost_rows if r.get("id")
    }
    rerank_by_id: Dict[str, Mapping[str, Any]] = {}
    if rerank_rows is not None:
        rerank_by_id = {
            str(r.get("id") or ""): r for r in rerank_rows if r.get("id")
        }

    boost_dump_by_query: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in boost_dump or ():
        qid = str(entry.get("query_id") or "")
        if not qid:
            continue
        boost_dump_by_query.setdefault(qid, []).append(entry)

    counts: Dict[str, int] = {g: 0 for g in GROUP_ORDER}
    samples: Dict[str, List[FailureSampleEntry]] = {g: [] for g in GROUP_ORDER}
    queries_evaluated = 0
    queries_skipped = 0

    qids = sorted(set(dense_by_id.keys()) | set(boost_by_id.keys()))
    for qid in qids:
        dense_row = dense_by_id.get(qid)
        boost_row = boost_by_id.get(qid)
        if dense_row is None or boost_row is None:
            queries_skipped += 1
            continue
        expected = _expected_for(boost_row) or _expected_for(dense_row)
        if not expected:
            queries_skipped += 1
            continue

        queries_evaluated += 1
        rerank_row = rerank_by_id.get(qid)

        dense_hit = _row_hit(dense_row, expected, top_k)
        boost_hit = _row_hit(boost_row, expected, top_k)
        rerank_hit = (
            _row_hit(rerank_row, expected, top_k)
            if rerank_row is not None
            else None
        )

        # Group selection. The boost_hit_rerank_* groups are only
        # populated when a rerank pipeline was provided.
        if not dense_hit and boost_hit:
            group = GROUP_DENSE_MISS_BOOST_HIT
        elif dense_hit and not boost_hit:
            group = GROUP_DENSE_HIT_BOOST_MISS
        elif boost_hit and rerank_hit is False:
            group = GROUP_BOOST_HIT_RERANK_MISS
        elif boost_hit and rerank_hit is True:
            group = GROUP_BOOST_HIT_RERANK_HIT
        elif not dense_hit and not boost_hit:
            group = GROUP_BOTH_MISS
        else:
            group = GROUP_NEUTRAL

        counts[group] += 1
        if len(samples[group]) < sample_limit:
            samples[group].append(
                _build_sample(
                    query_id=qid,
                    base_row=boost_row,
                    dense_row=dense_row,
                    boost_row=boost_row,
                    rerank_row=rerank_row,
                    boost_dump_for_query=boost_dump_by_query.get(qid, ()),
                )
            )

    group_stats = [
        FailureGroupStats(
            name=name,
            count=counts[name],
            ratio=(
                round(counts[name] / queries_evaluated, 4)
                if queries_evaluated
                else 0.0
            ),
        )
        for name in GROUP_ORDER
    ]
    return BoostFailureAnalysis(
        top_k=top_k,
        queries_evaluated=queries_evaluated,
        queries_skipped=queries_skipped,
        skip_reason=(
            "queries missing from either the dense or boost pipeline, "
            "or carrying no expected_doc_ids, are excluded"
        ),
        groups=group_stats,
        samples=samples,
    )


def boost_failure_analysis_to_dict(
    analysis: BoostFailureAnalysis,
) -> Dict[str, Any]:
    return asdict(analysis)


def render_boost_failure_markdown(
    analysis: BoostFailureAnalysis,
    *,
    sample_limit: int = 10,
) -> str:
    lines: List[str] = []
    lines.append("# Phase 2B boost failure analysis")
    lines.append("")
    lines.append(f"- top_k:               {analysis.top_k}")
    lines.append(f"- queries_evaluated:   {analysis.queries_evaluated}")
    lines.append(f"- queries_skipped:     {analysis.queries_skipped}")
    lines.append("")
    lines.append("## Group counts")
    lines.append("")
    lines.append("| group | count | ratio |")
    lines.append("|---|---:|---:|")
    for stat in analysis.groups:
        lines.append(
            f"| {stat.name} | {stat.count} | {stat.ratio:.4f} |"
        )
    lines.append("")
    for stat in analysis.groups:
        if stat.count == 0:
            continue
        sample_list = analysis.samples.get(stat.name, [])
        if not sample_list:
            continue
        lines.append(f"## Samples — `{stat.name}` (first {sample_limit})")
        lines.append("")
        for s in sample_list[:sample_limit]:
            lines.append(f"### `{s.query_id}` — {s.query[:80]}")
            lines.append("")
            lines.append(f"- expected: `{s.expected_doc_ids}`")
            lines.append(f"- dense_top5: `{s.dense_top5_doc_ids}`")
            lines.append(f"- boost_top5: `{s.boost_top5_doc_ids}`")
            if s.rerank_top5_doc_ids:
                lines.append(f"- rerank_top5: `{s.rerank_top5_doc_ids}`")
            lines.append("")
    return "\n".join(lines) + "\n"
