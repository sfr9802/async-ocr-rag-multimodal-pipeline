"""Phase 2B candidate-miss analysis at top-K cutoffs.

Re-tabulates the existing Phase 2A retrieval reports at the top-K
windows that survived the latency Pareto frontier (top5 / top10 /
top15) and groups every miss into a small set of failure types so
the Phase 2B boost design can target the bucket that's causing the
most pain.

Reads existing Phase 2A artifacts — never re-runs retrieval. The
candidate-recall report (top_k = 50, dense-only) is the most useful
input because it surfaces the doc rank for every gold query within
the candidate window the boost can plausibly rescue from.

Bucket rules (first match wins):

  ``corpus_missing``      Expected doc never appears in dense top-50.
                          The boost can't rescue this — the answer
                          isn't even in the candidate pool.
  ``title_mismatch``      Doc title shares a prefix with the query
                          BUT the dense top-K has a sibling doc that
                          collides on a single shared keyword. Most
                          common when a series has multiple seasons
                          and the query addresses the original.
  ``character_mismatch``  Expected keywords look like character /
                          person names (Latin / katakana proper noun
                          markers) and the dense top-K's matched
                          keywords list contains a different name.
  ``section_mismatch``    Doc is in dense top-K but on a different
                          section than the one the query asks about.
  ``lexical_keyword_mismatch``  Top-K chunk text shares few of the
                                expected keywords (overlap < 0.5).
  ``alias_or_synonym``    Expected keyword reads like a romanization
                          or partial transliteration of a hangul title
                          (or vice-versa) — common in
                          enterprise / brand corpora.
  ``overly_broad_query``  Query is short (<= 4 tokens) and the chunk
                          population in the candidate window is very
                          dispersed (many docs, no doc dominates).
  ``ambiguous_label``     Expected keywords are too generic to score
                          (e.g. all of them are common Korean function
                          words) — flags a dataset issue, not a
                          retriever issue.

Multiple buckets can arise from the same failure mode; we use a
deterministic priority order so each query maps to exactly one bucket
the report can summarize over.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Bucket constants — keep in priority order so callers can iterate.
BUCKET_CORPUS_MISSING = "corpus_missing"
BUCKET_TITLE_MISMATCH = "title_mismatch"
BUCKET_CHARACTER_MISMATCH = "character_mismatch"
BUCKET_SECTION_MISMATCH = "section_mismatch"
BUCKET_LEXICAL_MISMATCH = "lexical_keyword_mismatch"
BUCKET_ALIAS_SYNONYM = "alias_or_synonym"
BUCKET_OVERLY_BROAD = "overly_broad_query"
BUCKET_AMBIGUOUS_LABEL = "ambiguous_label"

BUCKET_ORDER: Tuple[str, ...] = (
    BUCKET_CORPUS_MISSING,
    BUCKET_TITLE_MISMATCH,
    BUCKET_CHARACTER_MISMATCH,
    BUCKET_SECTION_MISMATCH,
    BUCKET_LEXICAL_MISMATCH,
    BUCKET_ALIAS_SYNONYM,
    BUCKET_OVERLY_BROAD,
    BUCKET_AMBIGUOUS_LABEL,
)

DEFAULT_TOP_KS: Tuple[int, ...] = (5, 10, 15)
DEFAULT_DEEP_K = 50  # corpus_missing cutoff — the candidate pool
DEFAULT_SAMPLE_LIMIT = 25
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_KATAKANA_RE = re.compile(r"[゠-ヿ]+")


@dataclass
class CandidateMissBucketStats:
    """Per-(top_k, bucket) counts and ratios."""

    name: str
    count: int
    ratio: float


@dataclass
class CandidateMissSample:
    """One sample row in a bucket — enough to eyeball the failure."""

    query_id: str
    query: str
    answer_type: Optional[str]
    difficulty: Optional[str]
    expected_doc_ids: List[str]
    expected_section_keywords: List[str]
    retrieved_doc_ids_top_k: List[str]
    retrieved_doc_id_at_first_hit: Optional[str]
    expected_doc_rank_in_pool: Optional[int]


@dataclass
class CandidateMissTopKResult:
    """Bucket counts + samples at one top-K cutoff."""

    top_k: int
    queries_evaluated: int
    queries_missed: int
    miss_rate: float
    buckets: List[CandidateMissBucketStats] = field(default_factory=list)
    samples: Dict[str, List[CandidateMissSample]] = field(default_factory=dict)


@dataclass
class CandidateMissReport:
    """Top-level Phase 2B miss report — per top-K + dataset-level rollup."""

    schema: str = "phase2b-candidate-miss-analysis.v1"
    deep_k: int = DEFAULT_DEEP_K
    rows_evaluated: int = 0
    rows_skipped: int = 0
    skip_reason: str = ""
    per_top_k: List[CandidateMissTopKResult] = field(default_factory=list)
    bucket_definitions: Dict[str, str] = field(default_factory=dict)


_BUCKET_DEFINITIONS = {
    BUCKET_CORPUS_MISSING: (
        "Expected doc never appears in the candidate pool (top-deep_k); "
        "boost cannot rescue this, the answer isn't even retrieved."
    ),
    BUCKET_TITLE_MISMATCH: (
        "Doc title shares a token with a sibling doc that took the slot."
    ),
    BUCKET_CHARACTER_MISMATCH: (
        "Expected keywords look like character / proper-noun names but "
        "the top-K chunks matched a different name."
    ),
    BUCKET_SECTION_MISMATCH: (
        "Doc is in top-K but the chunk lives in a different section "
        "than the one the query asks about."
    ),
    BUCKET_LEXICAL_MISMATCH: (
        "Few of the expected keywords appear in the top-K chunk text; "
        "the bi-encoder is matching on weak signal."
    ),
    BUCKET_ALIAS_SYNONYM: (
        "Expected keywords look like an aliased / transliterated form "
        "of a title; pure substring boost won't fire."
    ),
    BUCKET_OVERLY_BROAD: (
        "Query is too short and the candidate window is dispersed; "
        "boost has no clear signal to amplify."
    ),
    BUCKET_AMBIGUOUS_LABEL: (
        "Expected keywords are generic; the dataset row may be too "
        "weakly grounded to score."
    ),
}


def _doc_ids_at_k(retrieved: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    return [str(d) for d in list(retrieved)[:k]]


def _expected_rank_in_pool(
    retrieved: Sequence[str],
    expected: Sequence[str],
) -> Optional[int]:
    """Return the 1-based rank of the first expected doc in ``retrieved``."""
    expected_set = {str(e) for e in expected if e}
    for idx, doc_id in enumerate(retrieved, start=1):
        if str(doc_id) in expected_set:
            return idx
    return None


def _has_title_overlap(query: str, doc_titles: Iterable[str]) -> bool:
    """Heuristic: query overlaps with at least one of ``doc_titles`` token-wise.

    Used by the title_mismatch bucket so we only fire when there's at
    least a token-level overlap — otherwise random misses would all
    end up in this bucket.
    """
    if not query:
        return False
    qtokens = {tok for tok in re.findall(r"\S+", query) if len(tok) >= 2}
    if not qtokens:
        return False
    for title in doc_titles:
        for tok in re.findall(r"\S+", title or ""):
            if len(tok) >= 2 and tok in qtokens:
                return True
    return False


def _looks_like_proper_noun(text: str) -> bool:
    """Heuristic for character / brand markers."""
    if not text:
        return False
    if _LATIN_RE.search(text):
        return True
    if _KATAKANA_RE.search(text):
        return True
    return False


def _keyword_coverage(
    expected_keywords: Sequence[str],
    chunk_texts: Sequence[str],
) -> float:
    """Fraction of expected keywords that appear in any chunk text."""
    if not expected_keywords:
        return 0.0
    haystack = "\n".join((t or "").lower() for t in chunk_texts)
    if not haystack:
        return 0.0
    matched = 0
    seen: set = set()
    for kw in expected_keywords:
        low = (kw or "").lower().strip()
        if not low or low in seen:
            continue
        seen.add(low)
        if low in haystack:
            matched += 1
    n = len(seen) or 1
    return matched / n


def _looks_like_alias(
    expected_keywords: Sequence[str],
    expected_doc_titles: Sequence[str],
) -> bool:
    """Pretty crude: expected_keywords carry a Latin token but the doc
    title is hangul-only (or vice-versa). The boost can't bridge that
    gap with substring matching.
    """
    kw_has_latin = any(_LATIN_RE.search(kw or "") for kw in expected_keywords)
    kw_has_hangul = any(
        re.search(r"[가-힯]", kw or "") for kw in expected_keywords
    )
    title_has_latin = any(
        _LATIN_RE.search(t or "") for t in expected_doc_titles
    )
    title_has_hangul = any(
        re.search(r"[가-힯]", t or "") for t in expected_doc_titles
    )
    return (
        (kw_has_latin and not title_has_latin and title_has_hangul)
        or (kw_has_hangul and not title_has_hangul and title_has_latin)
    )


def _classify_one(
    *,
    row: Mapping[str, Any],
    top_k_doc_ids: Sequence[str],
    deep_pool_doc_ids: Sequence[str],
    section_paths_at_k: Sequence[str],
    chunk_texts_at_k: Sequence[str],
    expected_doc_titles: Sequence[str],
    expected_doc_sections: Sequence[str],
    matched_keywords_at_k: Sequence[str],
) -> str:
    expected_doc_ids = [
        str(d) for d in (row.get("expected_doc_ids") or []) if d
    ]
    expected_keywords = [
        str(k) for k in (row.get("expected_section_keywords") or []) if k
    ]
    query_text = str(row.get("query") or "")

    expected_set = set(expected_doc_ids)

    # Priority 1: corpus_missing — expected doc not in deep pool at all.
    deep_set = set(str(d) for d in deep_pool_doc_ids)
    if not (expected_set & deep_set):
        return BUCKET_CORPUS_MISSING

    top_k_set = set(top_k_doc_ids)

    # Priority 2: section_mismatch — expected doc IS in top-K but the
    # retrieved chunk's section doesn't match what the query asks about.
    if (expected_set & top_k_set) and expected_doc_sections:
        chunks_for_doc = [
            (sec or "").strip()
            for did, sec in zip(top_k_doc_ids, section_paths_at_k)
            if did in expected_set
        ]
        if chunks_for_doc and not any(
            sec in expected_doc_sections for sec in chunks_for_doc
        ):
            return BUCKET_SECTION_MISMATCH

    # Priority 3: alias_or_synonym — keyword script ≠ title script.
    if _looks_like_alias(expected_keywords, expected_doc_titles):
        return BUCKET_ALIAS_SYNONYM

    # Priority 4: character_mismatch — expected keywords look like a
    # proper noun and the chunk's matched keywords don't include them.
    if expected_keywords and any(
        _looks_like_proper_noun(kw) for kw in expected_keywords
    ):
        # Did we match any of the expected proper-noun-y keywords?
        kw_low = {(kw or "").lower() for kw in matched_keywords_at_k}
        unmatched = [
            kw for kw in expected_keywords
            if _looks_like_proper_noun(kw)
            and (kw or "").lower() not in kw_low
        ]
        if unmatched:
            return BUCKET_CHARACTER_MISMATCH

    # Priority 5: title_mismatch — query has token-overlap with the
    # expected title AND a sibling doc occupies a top-K slot.
    if expected_doc_titles and _has_title_overlap(query_text, expected_doc_titles):
        # Is there at least one non-expected doc in top-K?
        if any(d not in expected_set for d in top_k_doc_ids):
            return BUCKET_TITLE_MISMATCH

    # Priority 6: lexical_keyword_mismatch — coverage < 0.5.
    coverage = _keyword_coverage(expected_keywords, chunk_texts_at_k)
    if expected_keywords and coverage < 0.5:
        return BUCKET_LEXICAL_MISMATCH

    # Priority 7: overly_broad_query.
    qtokens = [t for t in re.split(r"\s+", query_text) if t]
    if len(qtokens) <= 4:
        return BUCKET_OVERLY_BROAD

    # Priority 8: ambiguous_label fallthrough.
    return BUCKET_AMBIGUOUS_LABEL


def classify_candidate_misses(
    rows: Sequence[Mapping[str, Any]],
    *,
    dump_rows: Sequence[Mapping[str, Any]] = (),
    doc_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    deep_k: int = DEFAULT_DEEP_K,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CandidateMissReport:
    """Cross-tab a Phase 2A retrieval report over multiple top-K cutoffs.

    Inputs:

      - ``rows``: list of rows from a candidate-recall (or any other
        retrieval) ``retrieval_eval_report.json``. Each row must carry
        ``retrieved_doc_ids`` capped at AT LEAST ``deep_k`` so the
        ``corpus_missing`` bucket is meaningful.
      - ``dump_rows``: optional ``top_k_dump.jsonl`` lines for
        section-path / chunk-text coverage. When omitted, only doc-id
        based buckets are computed (section_mismatch and
        lexical_keyword_mismatch will not fire).
      - ``doc_metadata``: optional ``doc_id → {title, section_names}``
        map. When omitted, title_mismatch / alias_or_synonym fall
        through to less-specific buckets.

    Skips rows without ``expected_doc_ids`` (no ground truth → no
    bucket assignment), counted in ``rows_skipped``.
    """
    rows = [r for r in rows]
    dump_rows = [d for d in dump_rows]

    dump_by_query: Dict[str, List[Mapping[str, Any]]] = {}
    for d in dump_rows:
        qid = str(d.get("query_id") or "")
        if not qid:
            continue
        dump_by_query.setdefault(qid, []).append(d)
    for qid, lst in dump_by_query.items():
        lst.sort(key=lambda r: int(r.get("rank") or 0))

    doc_meta_map = dict(doc_metadata or {})

    rows_evaluated = 0
    rows_skipped = 0
    per_top_k_counts: Dict[int, Dict[str, int]] = {
        k: {b: 0 for b in BUCKET_ORDER} for k in top_ks
    }
    per_top_k_samples: Dict[int, Dict[str, List[CandidateMissSample]]] = {
        k: {b: [] for b in BUCKET_ORDER} for k in top_ks
    }
    per_top_k_missed: Dict[int, int] = {k: 0 for k in top_ks}

    for row in rows:
        expected = [
            str(d) for d in (row.get("expected_doc_ids") or []) if d
        ]
        if not expected:
            rows_skipped += 1
            continue
        rows_evaluated += 1
        retrieved = [str(d) for d in (row.get("retrieved_doc_ids") or [])]
        deep_pool = retrieved[: max(deep_k, 1)]
        expected_set = set(expected)

        # Pull doc-meta context for the expected docs.
        expected_doc_titles = [
            str(doc_meta_map.get(did, {}).get("title") or "")
            for did in expected
        ]
        expected_doc_sections: List[str] = []
        for did in expected:
            sections = doc_meta_map.get(did, {}).get("section_names") or ()
            for s in sections:
                if s:
                    expected_doc_sections.append(s)

        qid = str(row.get("id") or "")
        dump = dump_by_query.get(qid, [])
        section_paths = [str(d.get("section_path") or "") for d in dump]
        chunk_texts = [str(d.get("chunk_preview") or "") for d in dump]
        matched_kw = []
        for d in dump:
            for kw in d.get("matched_expected_keyword") or []:
                if kw and kw not in matched_kw:
                    matched_kw.append(kw)

        for top_k in top_ks:
            top_k_ids = retrieved[:top_k]
            top_k_set = set(top_k_ids)
            hit = bool(expected_set & top_k_set)
            if hit:
                continue
            per_top_k_missed[top_k] += 1
            bucket = _classify_one(
                row=row,
                top_k_doc_ids=top_k_ids,
                deep_pool_doc_ids=deep_pool,
                section_paths_at_k=section_paths[:top_k],
                chunk_texts_at_k=chunk_texts[:top_k],
                expected_doc_titles=expected_doc_titles,
                expected_doc_sections=expected_doc_sections,
                matched_keywords_at_k=matched_kw[:top_k],
            )
            per_top_k_counts[top_k][bucket] += 1

            if len(per_top_k_samples[top_k][bucket]) < sample_limit:
                per_top_k_samples[top_k][bucket].append(
                    CandidateMissSample(
                        query_id=qid,
                        query=str(row.get("query") or ""),
                        answer_type=row.get("answer_type"),
                        difficulty=row.get("difficulty"),
                        expected_doc_ids=expected,
                        expected_section_keywords=[
                            str(k) for k in row.get("expected_section_keywords") or []
                        ],
                        retrieved_doc_ids_top_k=top_k_ids,
                        retrieved_doc_id_at_first_hit=(
                            top_k_ids[0] if top_k_ids else None
                        ),
                        expected_doc_rank_in_pool=_expected_rank_in_pool(
                            retrieved, expected
                        ),
                    )
                )

    per_top_k_results: List[CandidateMissTopKResult] = []
    for k in top_ks:
        miss = per_top_k_missed[k]
        miss_rate = (
            round(miss / rows_evaluated, 4) if rows_evaluated else 0.0
        )
        bucket_stats = [
            CandidateMissBucketStats(
                name=name,
                count=per_top_k_counts[k][name],
                ratio=(
                    round(per_top_k_counts[k][name] / miss, 4)
                    if miss
                    else 0.0
                ),
            )
            for name in BUCKET_ORDER
        ]
        per_top_k_results.append(
            CandidateMissTopKResult(
                top_k=k,
                queries_evaluated=rows_evaluated,
                queries_missed=miss,
                miss_rate=miss_rate,
                buckets=bucket_stats,
                samples=per_top_k_samples[k],
            )
        )

    return CandidateMissReport(
        deep_k=deep_k,
        rows_evaluated=rows_evaluated,
        rows_skipped=rows_skipped,
        skip_reason=(
            "rows without expected_doc_ids excluded — bucketing requires "
            "at least one ground-truth doc id"
        ),
        per_top_k=per_top_k_results,
        bucket_definitions=dict(_BUCKET_DEFINITIONS),
    )


def candidate_miss_report_to_dict(report: CandidateMissReport) -> Dict[str, Any]:
    return asdict(report)


def render_candidate_miss_markdown(report: CandidateMissReport) -> str:
    """Compose a top-level summary for the analysis report."""
    lines: List[str] = []
    lines.append("# Phase 2B candidate miss analysis")
    lines.append("")
    lines.append(f"- rows evaluated: {report.rows_evaluated}")
    lines.append(f"- rows skipped:   {report.rows_skipped}")
    lines.append(f"- deep_k:         {report.deep_k}")
    lines.append("")
    lines.append("## Bucket counts per top-K")
    lines.append("")
    headers = ["bucket"] + [f"top-{k}" for k in (r.top_k for r in report.per_top_k)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for bucket in BUCKET_ORDER:
        cells = [bucket]
        for tkr in report.per_top_k:
            stat = next((b for b in tkr.buckets if b.name == bucket), None)
            if stat is None:
                cells.append("-")
            else:
                cells.append(f"{stat.count} ({stat.ratio:.2%})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Miss rates")
    lines.append("")
    lines.append("| top-K | queries_missed | miss_rate |")
    lines.append("|---:|---:|---:|")
    for tkr in report.per_top_k:
        lines.append(
            f"| {tkr.top_k} | {tkr.queries_missed} | {tkr.miss_rate:.4f} |"
        )
    lines.append("")
    lines.append("## Bucket definitions")
    lines.append("")
    for bucket in BUCKET_ORDER:
        lines.append(f"- `{bucket}` — {report.bucket_definitions.get(bucket, '')}")
    lines.append("")
    return "\n".join(lines) + "\n"
