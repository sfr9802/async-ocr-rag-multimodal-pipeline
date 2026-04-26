"""Miss-bucket analysis for the retrieval eval harness.

Cross-tabulates each query into one of four buckets so a Phase-0
baseline can separate "the retriever missed the doc" from "the
retriever found a doc but no chunk matched the eval row's keywords".

Buckets
-------
- ``doc_hit_keyword_hit``   doc in top-k AND any expected keyword
                            matched some retrieved chunk
- ``doc_hit_keyword_miss``  doc in top-k but no keyword matched
- ``doc_miss_keyword_hit``  doc NOT in top-k but some keyword still
                            matched a retrieved chunk — usually a
                            sign of dataset-level keyword leakage
                            into a generic / off-topic doc
- ``doc_miss_keyword_miss`` neither — the hard miss class

The analyzer reads the existing ``RetrievalEvalRow`` + ``TopKDumpRow``
dataclasses, so it never re-runs retrieval. It emits a JSON / Markdown
report with per-bucket counts, ratios, and capped samples for the two
``doc_miss_*`` buckets so a reviewer can spot why a row blew up.

Definitions
-----------
- ``doc_hit`` is a hit_at_k(top_k, expected_doc_ids) over the same k
  the retrieval harness used (default 10). Doc-id normalization is
  the same as ``hit_at_k`` so unicode-width / casing drift doesn't
  fake a miss.
- ``keyword_hit`` is a single substring match (case-insensitive) of
  ANY expected_section_keyword in the union of the top-k chunk texts
  and section paths. This matches what the top-k dump's
  ``matched_expected_keyword`` field already records, so the analyzer
  reuses that field when the dump rows are passed in. When dump rows
  aren't available, it falls back to the per-row chunk_texts (the
  eval harness does not retain those), which is why callers must pass
  the dump alongside the rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from eval.harness.metrics import hit_at_k


BUCKET_DOC_HIT_KW_HIT = "doc_hit_keyword_hit"
BUCKET_DOC_HIT_KW_MISS = "doc_hit_keyword_miss"
BUCKET_DOC_MISS_KW_HIT = "doc_miss_keyword_hit"
BUCKET_DOC_MISS_KW_MISS = "doc_miss_keyword_miss"

BUCKET_ORDER = (
    BUCKET_DOC_HIT_KW_HIT,
    BUCKET_DOC_HIT_KW_MISS,
    BUCKET_DOC_MISS_KW_HIT,
    BUCKET_DOC_MISS_KW_MISS,
)

# Sample cap for the two "doc_miss_*" buckets — enough to eyeball
# patterns, small enough to fit in a markdown review file.
DEFAULT_SAMPLE_LIMIT = 25


@dataclass
class MissTopKEntry:
    rank: int
    doc_id: str
    section_path: Optional[str]
    score: float
    chunk_preview: str
    matched_expected_keyword: List[str] = field(default_factory=list)


@dataclass
class MissBucketSample:
    query_id: str
    query: str
    answer_type: Optional[str]
    difficulty: Optional[str]
    language: Optional[str]
    expected_doc_ids: List[str]
    expected_section_keywords: List[str]
    matched_expected_keyword: List[str]
    top_5: List[MissTopKEntry] = field(default_factory=list)


@dataclass
class MissBucketStats:
    name: str
    count: int
    ratio: float


@dataclass
class MissAnalysis:
    top_k: int
    rows_evaluated: int
    rows_skipped: int
    skip_reason: str
    buckets: List[MissBucketStats]
    per_answer_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict[str, int]] = field(default_factory=dict)
    samples: Dict[str, List[MissBucketSample]] = field(default_factory=dict)


def _normalize_keyword(value: str) -> str:
    return (value or "").strip().lower()


def _row_keyword_hit(
    expected_keywords: Sequence[str],
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[bool, List[str]]:
    """Return (any_match, matched_keywords) for a single row.

    Matches are substring, case-insensitive — same shape as
    ``expected_keyword_match_rate`` and the dump-row matched field.
    A keyword counts if it appears in either the chunk text OR the
    section path of any retrieved chunk.
    """
    needles = [_normalize_keyword(k) for k in expected_keywords if k]
    needles = [n for n in needles if n]
    if not needles:
        return False, []
    haystack_pieces: List[str] = []
    for chunk in chunks:
        text = str(chunk.get("text") or chunk.get("chunk_preview") or "").lower()
        section = str(chunk.get("section_path") or chunk.get("section") or "").lower()
        if text:
            haystack_pieces.append(text)
        if section:
            haystack_pieces.append(section)
    haystack = "\n".join(haystack_pieces)
    if not haystack:
        return False, []
    matched: List[str] = []
    seen: set[str] = set()
    for raw, needle in zip(expected_keywords, needles):
        if needle in seen:
            continue
        if needle in haystack:
            matched.append(raw)
            seen.add(needle)
    return bool(matched), matched


def classify_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dump_rows: Sequence[Mapping[str, Any]] = (),
    top_k: int = 10,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> MissAnalysis:
    """Classify each query into the 4 buckets.

    Inputs accept either raw dicts (e.g. straight off
    ``retrieval_eval_report.json``'s ``rows``) or dataclass instances
    of ``RetrievalEvalRow`` / ``TopKDumpRow`` — both shapes are read
    via ``.get``/``getattr`` so the analyzer can be called from the
    in-process harness or from a CLI that loaded the JSON back.

    A row is skipped (with ``rows_skipped`` incremented) when it has
    no ``expected_doc_ids`` — the cross-tab needs both axes defined.
    The skip is reported once in ``skip_reason`` so the report is
    explicit about what was excluded.
    """
    rows_dicts = [_coerce_row(r) for r in rows]
    dump_dicts = [_coerce_row(d) for d in dump_rows]

    # Index dump rows by query_id once.
    dump_by_query: Dict[str, List[Mapping[str, Any]]] = {}
    for d in dump_dicts:
        qid = str(d.get("query_id") or "")
        if not qid:
            continue
        dump_by_query.setdefault(qid, []).append(d)
    for qid, lst in dump_by_query.items():
        lst.sort(key=lambda r: int(r.get("rank") or 0))

    counts: Dict[str, int] = {b: 0 for b in BUCKET_ORDER}
    samples: Dict[str, List[MissBucketSample]] = {b: [] for b in BUCKET_ORDER}
    per_answer_type: Dict[str, Dict[str, int]] = {}
    per_difficulty: Dict[str, Dict[str, int]] = {}

    rows_evaluated = 0
    rows_skipped = 0

    for row in rows_dicts:
        qid = str(row.get("id") or "")
        expected_doc_ids = list(row.get("expected_doc_ids") or [])
        expected_keywords = list(row.get("expected_section_keywords") or [])

        if not expected_doc_ids:
            rows_skipped += 1
            continue
        rows_evaluated += 1

        retrieved_doc_ids = list(row.get("retrieved_doc_ids") or [])

        # Doc hit uses the harness's own normalization (handled by hit_at_k).
        doc_hit_score = hit_at_k(retrieved_doc_ids, expected_doc_ids, k=top_k)
        doc_hit = bool(doc_hit_score and doc_hit_score > 0.0)

        # Keyword hit: prefer dump rows so we get section_path coverage
        # exactly the same way the dump did. Fall back to whatever
        # chunk_texts the eval row carried (most rows don't, by design).
        chunks_for_kw: List[Mapping[str, Any]] = []
        dump_for_query = dump_by_query.get(qid, [])
        if dump_for_query:
            chunks_for_kw = [
                {
                    "text": d.get("chunk_preview") or "",
                    "section_path": d.get("section_path") or "",
                    "matched_expected_keyword": d.get("matched_expected_keyword") or [],
                }
                for d in dump_for_query[:top_k]
            ]
            # When dump already pre-computed matched_expected_keyword,
            # union them — that's the canonical answer.
            matched_from_dump: List[str] = []
            seen_low: set[str] = set()
            for d in dump_for_query[:top_k]:
                for kw in d.get("matched_expected_keyword") or []:
                    low = _normalize_keyword(kw)
                    if low and low not in seen_low:
                        matched_from_dump.append(kw)
                        seen_low.add(low)
            if matched_from_dump:
                kw_hit = True
                matched_kw = matched_from_dump
            else:
                kw_hit, matched_kw = _row_keyword_hit(expected_keywords, chunks_for_kw)
        else:
            kw_hit, matched_kw = _row_keyword_hit(expected_keywords, chunks_for_kw)

        bucket = _pick_bucket(doc_hit=doc_hit, kw_hit=kw_hit)
        counts[bucket] += 1

        # Per-axis breakdowns (count-only — keep the report compact).
        atype = row.get("answer_type") or "unknown"
        diff = row.get("difficulty") or "unknown"
        per_answer_type.setdefault(atype, {b: 0 for b in BUCKET_ORDER})[bucket] += 1
        per_difficulty.setdefault(diff, {b: 0 for b in BUCKET_ORDER})[bucket] += 1

        # Build a sample for the two failure buckets.
        if bucket in (BUCKET_DOC_MISS_KW_HIT, BUCKET_DOC_MISS_KW_MISS):
            if len(samples[bucket]) < sample_limit:
                samples[bucket].append(
                    _build_sample(
                        row=row,
                        dump_for_query=dump_for_query,
                        matched_kw=matched_kw,
                    )
                )

    bucket_stats = [
        MissBucketStats(
            name=name,
            count=counts[name],
            ratio=round(counts[name] / rows_evaluated, 4) if rows_evaluated else 0.0,
        )
        for name in BUCKET_ORDER
    ]

    return MissAnalysis(
        top_k=top_k,
        rows_evaluated=rows_evaluated,
        rows_skipped=rows_skipped,
        skip_reason=(
            "rows without expected_doc_ids are excluded — bucket assignment "
            "requires both doc-hit and keyword-hit axes to be defined"
        ),
        buckets=bucket_stats,
        per_answer_type=per_answer_type,
        per_difficulty=per_difficulty,
        samples=samples,
    )


def _pick_bucket(*, doc_hit: bool, kw_hit: bool) -> str:
    if doc_hit and kw_hit:
        return BUCKET_DOC_HIT_KW_HIT
    if doc_hit and not kw_hit:
        return BUCKET_DOC_HIT_KW_MISS
    if not doc_hit and kw_hit:
        return BUCKET_DOC_MISS_KW_HIT
    return BUCKET_DOC_MISS_KW_MISS


def _coerce_row(row: Any) -> Mapping[str, Any]:
    """Accept either a dataclass or a dict — return a dict-like view."""
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "__dataclass_fields__"):
        return asdict(row)
    # Best-effort: object with attributes.
    return {
        k: getattr(row, k, None)
        for k in (
            "id", "query", "language", "expected_doc_ids",
            "expected_section_keywords", "answer_type", "difficulty",
            "retrieved_doc_ids", "query_id", "rank", "doc_id",
            "section_path", "score", "chunk_preview",
            "matched_expected_keyword", "is_expected_doc",
        )
    }


def _build_sample(
    *,
    row: Mapping[str, Any],
    dump_for_query: Sequence[Mapping[str, Any]],
    matched_kw: List[str],
) -> MissBucketSample:
    top5 = []
    for d in dump_for_query[:5]:
        try:
            score = float(d.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        top5.append(
            MissTopKEntry(
                rank=int(d.get("rank") or 0),
                doc_id=str(d.get("doc_id") or ""),
                section_path=d.get("section_path"),
                score=round(score, 6),
                chunk_preview=str(d.get("chunk_preview") or ""),
                matched_expected_keyword=list(d.get("matched_expected_keyword") or []),
            )
        )
    return MissBucketSample(
        query_id=str(row.get("id") or ""),
        query=str(row.get("query") or ""),
        answer_type=row.get("answer_type"),
        difficulty=row.get("difficulty"),
        language=row.get("language"),
        expected_doc_ids=list(row.get("expected_doc_ids") or []),
        expected_section_keywords=list(row.get("expected_section_keywords") or []),
        matched_expected_keyword=matched_kw,
        top_5=top5,
    )


# ---------------------------------------------------------------------------
# Serialization helpers (used by the CLI writer + tests).
# ---------------------------------------------------------------------------


def miss_analysis_to_dict(analysis: MissAnalysis) -> Dict[str, Any]:
    return {
        "top_k": analysis.top_k,
        "rows_evaluated": analysis.rows_evaluated,
        "rows_skipped": analysis.rows_skipped,
        "skip_reason": analysis.skip_reason,
        "buckets": [asdict(b) for b in analysis.buckets],
        "per_answer_type": analysis.per_answer_type,
        "per_difficulty": analysis.per_difficulty,
        "samples": {
            name: [asdict(s) for s in sample_list]
            for name, sample_list in analysis.samples.items()
        },
    }


def render_miss_analysis_markdown(analysis: MissAnalysis) -> str:
    """Compose the human-readable miss_analysis.md.

    Lays out the 2x2 cross-tab first, then the per-axis breakdowns,
    then the capped samples for the two ``doc_miss_*`` buckets. The
    samples include the top-5 dump rows so a reviewer can tell at a
    glance whether the retriever surfaced a topical-but-wrong doc
    (keyword leakage) or something genuinely off-topic.
    """
    lines: List[str] = []
    lines.append("# Retrieval miss-bucket analysis")
    lines.append("")
    lines.append(f"- top_k:          {analysis.top_k}")
    lines.append(f"- rows_evaluated: {analysis.rows_evaluated}")
    lines.append(f"- rows_skipped:   {analysis.rows_skipped} ({analysis.skip_reason})")
    lines.append("")

    lines.append("## Buckets")
    lines.append("")
    lines.append("| bucket | count | ratio |")
    lines.append("|---|---:|---:|")
    for b in analysis.buckets:
        lines.append(f"| {b.name} | {b.count} | {b.ratio:.4f} |")
    lines.append("")

    if analysis.per_answer_type:
        lines.append("## Per answer_type")
        lines.append("")
        header = ["answer_type", "n", *BUCKET_ORDER]
        lines.append("| " + " | ".join(header) + " |")
        lines.append(
            "|---|" + "---:|" * (len(header) - 1)
        )
        for atype, counts in sorted(analysis.per_answer_type.items()):
            n = sum(counts.values())
            row = [atype, str(n)] + [str(counts.get(b, 0)) for b in BUCKET_ORDER]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if analysis.per_difficulty:
        lines.append("## Per difficulty")
        lines.append("")
        header = ["difficulty", "n", *BUCKET_ORDER]
        lines.append("| " + " | ".join(header) + " |")
        lines.append(
            "|---|" + "---:|" * (len(header) - 1)
        )
        for diff, counts in sorted(analysis.per_difficulty.items()):
            n = sum(counts.values())
            row = [diff, str(n)] + [str(counts.get(b, 0)) for b in BUCKET_ORDER]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    for bucket in (BUCKET_DOC_MISS_KW_HIT, BUCKET_DOC_MISS_KW_MISS):
        sample_list = analysis.samples.get(bucket, [])
        if not sample_list:
            continue
        lines.append(f"## Samples — {bucket} (capped at {len(sample_list)})")
        lines.append("")
        for s in sample_list:
            lines.append(
                f"### `{s.query_id}` — {s.answer_type or '?'}/{s.difficulty or '?'} "
                f"[{s.language or '?'}]"
            )
            lines.append("")
            lines.append(f"- query: {s.query}")
            lines.append(f"- expected_doc_ids: {s.expected_doc_ids}")
            lines.append(f"- expected_section_keywords: {s.expected_section_keywords}")
            lines.append(f"- matched_expected_keyword: {s.matched_expected_keyword}")
            if s.top_5:
                lines.append("- top_5:")
                for entry in s.top_5:
                    sect = entry.section_path or "<no-section>"
                    matched = (
                        f" matched={entry.matched_expected_keyword}"
                        if entry.matched_expected_keyword else ""
                    )
                    lines.append(
                        f"  - [{entry.rank}] `{entry.doc_id}` "
                        f"({sect}) score={entry.score:.4f}{matched}"
                    )
                    if entry.chunk_preview:
                        lines.append(f"    > {entry.chunk_preview}")
            lines.append("")

    return "\n".join(lines) + "\n"
