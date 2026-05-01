"""Phase 7 silver-500 query generator (v2).

The Phase 7.0 generator (``v4_silver_queries.generate_v4_silver_queries``)
emits ~200 queries with one template per bucket. Coverage works for the
retrieval_title A/B but is too narrow for confidence + recovery analysis
once the corpus has grown — every bucket looks the same and the easy
title-match path dominates.

This module raises coverage to ~500 queries with a richer per-bucket
template family while preserving the existing JSONL schema so the rest
of the harness (Phase 7.0/7.1/7.3/7.4 readers) can consume the file
unchanged.

Terminology guarantee — every produced row is **silver**, not human-
verified gold:

  - ``id`` prefix is ``v4-silver-500-NNNN``.
  - ``tags`` always include ``"silver"`` and ``"synthetic"``; never
    ``"gold"`` and never ``"human_verified"``.
  - ``v4_meta`` carries an explicit ``silver_label_source`` describing
    *why* the expected_doc_id was chosen (page ↔ template), and
    ``silver_label_confidence`` (a coarse self-rating from the template).
  - The summary report (``queries_v4_silver_500.summary.md``) leads with
    a "silver, not human-verified gold" disclaimer.

Buckets — distribution targets:

  - ``main_work``        ≈ 150 (30%)
  - ``subpage_generic``  ≈ 200 (40%)
  - ``subpage_named``    ≈ 150 (30%)

Best-effort fill: when a bucket has fewer eligible pages × templates
than its target, the deficit is reported in the summary rather than
being silently filled from another bucket. The generator never produces
duplicate (page_id, template) pairs and shuffles deterministically off
``seed``.

Template families per bucket:

  main_work:
    - ``title_lookup``      "X이(가) 어떤 작품인가요?"
    - ``plot_summary``      "X의 줄거리에 대해 알려주세요."
    - ``evaluation``        "X의 평가는 어떤가요?"
    - ``alias_lookup``      "<alias>이(가) 무슨 작품이야?"  (only when
                            ``aliases`` carries a non-page-title alias)
    - ``ambiguous_short``   "X에 대해 알려주세요."  (only when
                            page_title is short / generic-prone, biased
                            toward title collisions)

  subpage_generic:
    - ``section_lookup``    "<work>의 <section>에 대해 알려주세요."
    - ``section_detail``    "<work>의 <section> 정보를 자세히 알려줘."
    - ``section_question``  "<work> <section>은 어떤 내용인가요?"

  subpage_named:
    - ``named_lookup``      "<work>의 <named>에 대해 알려주세요."
    - ``named_question``    "<named>는 <work>에서 어떤 등장인물/내용인가요?"
    - ``named_alias``       "<alias>는 <work>에서 어떤 인물인가요?"  (only
                            when the page carries a real alias)

Each template emits one ``QueryTemplate`` and is deterministic given
the page record + seed. The orchestrator (:func:`generate_silver_500`)
is responsible for stratified selection.
"""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from eval.harness.v4_silver_queries import (
    V4Page,
    _clean_text,
    _extract_nouns,
    _GENERIC_PAGE_TITLES,
    _is_generic_page_title,
    _parent_work_from_retrieval_title,
    iter_v4_pages,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen taxonomies
# ---------------------------------------------------------------------------


BUCKET_MAIN_WORK = "main_work"
BUCKET_SUBPAGE_GENERIC = "subpage_generic"
BUCKET_SUBPAGE_NAMED = "subpage_named"

BUCKETS: Tuple[str, ...] = (
    BUCKET_MAIN_WORK,
    BUCKET_SUBPAGE_GENERIC,
    BUCKET_SUBPAGE_NAMED,
)


TEMPLATE_TITLE_LOOKUP = "title_lookup"
TEMPLATE_PLOT_SUMMARY = "plot_summary"
TEMPLATE_EVALUATION = "evaluation"
TEMPLATE_ALIAS_LOOKUP = "alias_lookup"
TEMPLATE_AMBIGUOUS_SHORT = "ambiguous_short"
TEMPLATE_SECTION_LOOKUP = "section_lookup"
TEMPLATE_SECTION_DETAIL = "section_detail"
TEMPLATE_SECTION_QUESTION = "section_question"
TEMPLATE_NAMED_LOOKUP = "named_lookup"
TEMPLATE_NAMED_QUESTION = "named_question"
TEMPLATE_NAMED_ALIAS = "named_alias"

ALL_TEMPLATES: Tuple[str, ...] = (
    TEMPLATE_TITLE_LOOKUP,
    TEMPLATE_PLOT_SUMMARY,
    TEMPLATE_EVALUATION,
    TEMPLATE_ALIAS_LOOKUP,
    TEMPLATE_AMBIGUOUS_SHORT,
    TEMPLATE_SECTION_LOOKUP,
    TEMPLATE_SECTION_DETAIL,
    TEMPLATE_SECTION_QUESTION,
    TEMPLATE_NAMED_LOOKUP,
    TEMPLATE_NAMED_QUESTION,
    TEMPLATE_NAMED_ALIAS,
)


DEFAULT_BUCKET_TARGETS: Dict[str, int] = {
    BUCKET_MAIN_WORK: 150,
    BUCKET_SUBPAGE_GENERIC: 200,
    BUCKET_SUBPAGE_NAMED: 150,
}


# Strings the disclaimer/report must carry verbatim (asserted by tests).
SILVER_DISCLAIMER_LINE = (
    "This 500-query set is **silver** (synthetic, deterministic), "
    "NOT human-verified gold."
)
SILVER_AUDIT_LINE = (
    "No precision/recall/accuracy claim should be made until human "
    "labels are filled."
)


# ---------------------------------------------------------------------------
# Template result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryTemplate:
    """One rendered query candidate (template-side, before sampling).

    ``label_source`` records how the silver label (page_id) was derived:
    "page_lookup" means the template targets the current page directly,
    "subpage_pivot" means the work + section pair was used to anchor on
    a non-main page. ``label_confidence`` is a coarse self-rating
    ("high" for direct title match, "medium" for section/parent-anchored
    queries, "low" for ambiguous_short / alias-only).
    """

    text: str
    keywords: Tuple[str, ...]
    answer_type: str
    difficulty: str
    bucket: str
    template_kind: str
    label_source: str
    label_confidence: str
    extra_meta: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


_SHORT_TITLE_LIMIT = 5  # length below which a title is "short / generic-prone"
_NUM_RUN_RE = re.compile(r"^\W+|\W+$")


def _trim_punct(s: str) -> str:
    """Strip leading/trailing punctuation/whitespace."""
    return _NUM_RUN_RE.sub("", s) if s else s


def _real_aliases(page: V4Page) -> Tuple[str, ...]:
    """Aliases that aren't just the page_title verbatim.

    Phase 6.3's ``aliases`` always includes the page_title itself; we
    surface only the *additional* aliases so an alias-anchored query
    materially differs from the page-title lookup.
    """
    if not page.aliases:
        return ()
    pt_norm = _norm(page.page_title)
    out: List[str] = []
    for a in page.aliases:
        if not a:
            continue
        if _norm(a) == pt_norm:
            continue
        out.append(a)
    return tuple(out)


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return unicodedata.normalize("NFKC", s).strip().lower()


def _has_alias_distinct_from_work(page: V4Page) -> bool:
    """True when the page carries an alias that isn't the work title.

    Used by ``named_alias`` to decide whether the alias-anchored variant
    has a useful term to anchor on.
    """
    work_norm = _norm(page.work_title)
    parent_norm = _norm(
        _parent_work_from_retrieval_title(
            page.retrieval_title, page.page_title,
        )
    )
    page_norm = _norm(page.page_title)
    for a in _real_aliases(page):
        an = _norm(a)
        if an in (work_norm, parent_norm, page_norm):
            continue
        return True
    return False


def _first_distinct_alias(page: V4Page) -> Optional[str]:
    """First alias that's distinct from page_title and work_title.

    Returns ``None`` when no usable alias exists. Trims surrounding
    punctuation so an alias like ``"(애니메이션)"`` renders cleanly
    inside the query string.
    """
    work_norm = _norm(page.work_title)
    parent_norm = _norm(
        _parent_work_from_retrieval_title(
            page.retrieval_title, page.page_title,
        )
    )
    page_norm = _norm(page.page_title)
    for a in _real_aliases(page):
        an = _norm(a)
        if an in (work_norm, parent_norm, page_norm):
            continue
        trimmed = _trim_punct(a)
        if trimmed:
            return trimmed
    return None


def _is_short_or_generic_prone(title: str) -> bool:
    """Heuristic: is the title short enough / common enough to collide?

    Used by the ``ambiguous_short`` template to flag queries that are
    plausibly susceptible to generic-title collision in retrieval —
    short Korean titles, romanised English titles under 5 chars, and
    titles that contain a page_title generic token (등장인물 / 평가 / …).
    """
    if not title:
        return False
    if len(title) <= _SHORT_TITLE_LIMIT:
        return True
    return _is_generic_page_title(title)


# ---------------------------------------------------------------------------
# Template implementations
# ---------------------------------------------------------------------------


def _t_main_title_lookup(page: V4Page) -> Optional[QueryTemplate]:
    if page.page_type != "work" or page.relation != "main":
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = (title,) + tuple(n for n in nouns if n and n != title)
    return QueryTemplate(
        text=f"{title}이(가) 어떤 작품인가요?",
        keywords=keywords[:4],
        answer_type="title_lookup",
        difficulty="easy",
        bucket=BUCKET_MAIN_WORK,
        template_kind=TEMPLATE_TITLE_LOOKUP,
        label_source="page_lookup",
        label_confidence="high",
    )


def _t_main_plot_summary(page: V4Page) -> Optional[QueryTemplate]:
    if page.page_type != "work" or page.relation != "main":
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = (title, "줄거리") + tuple(n for n in nouns if n and n != title)
    return QueryTemplate(
        text=f"{title}의 줄거리에 대해 알려주세요.",
        keywords=keywords[:5],
        answer_type="plot_summary",
        difficulty="easy",
        bucket=BUCKET_MAIN_WORK,
        template_kind=TEMPLATE_PLOT_SUMMARY,
        label_source="page_lookup",
        label_confidence="high",
    )


def _t_main_evaluation(page: V4Page) -> Optional[QueryTemplate]:
    if page.page_type != "work" or page.relation != "main":
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    keywords = (title, "평가", "리뷰", "감상")
    return QueryTemplate(
        text=f"{title}의 평가는 어떤가요?",
        keywords=keywords,
        answer_type="evaluation",
        difficulty="medium",
        bucket=BUCKET_MAIN_WORK,
        template_kind=TEMPLATE_EVALUATION,
        label_source="page_lookup",
        label_confidence="medium",
        extra_meta={"prone_to_review_subpage": True},
    )


def _t_main_alias_lookup(page: V4Page) -> Optional[QueryTemplate]:
    if page.page_type != "work" or page.relation != "main":
        return None
    alias = _first_distinct_alias(page)
    if not alias:
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    keywords = (alias, title)
    return QueryTemplate(
        text=f"{alias}이(가) 무슨 작품이야?",
        keywords=keywords,
        answer_type="alias_lookup",
        difficulty="medium",
        bucket=BUCKET_MAIN_WORK,
        template_kind=TEMPLATE_ALIAS_LOOKUP,
        label_source="alias_pivot",
        label_confidence="medium",
        extra_meta={"alias_used": alias},
    )


def _t_main_ambiguous_short(page: V4Page) -> Optional[QueryTemplate]:
    if page.page_type != "work" or page.relation != "main":
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    if not _is_short_or_generic_prone(title):
        return None
    keywords = (title,) + tuple(_extract_nouns(page.first_section_text, limit=2))
    return QueryTemplate(
        text=f"{title}에 대해 알려주세요.",
        keywords=keywords[:3],
        answer_type="ambiguous_lookup",
        difficulty="hard",
        bucket=BUCKET_MAIN_WORK,
        template_kind=TEMPLATE_AMBIGUOUS_SHORT,
        label_source="page_lookup",
        label_confidence="low",
        extra_meta={"generic_prone": True},
    )


def _t_subpage_generic_section_lookup(
    page: V4Page,
) -> Optional[QueryTemplate]:
    if not page.page_title or not _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = (parent, page.page_title) + tuple(
        n for n in nouns if n and n not in (parent, page.page_title)
    )
    return QueryTemplate(
        text=f"{parent}의 {page.page_title}에 대해 알려주세요.",
        keywords=keywords[:5],
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket=BUCKET_SUBPAGE_GENERIC,
        template_kind=TEMPLATE_SECTION_LOOKUP,
        label_source="subpage_pivot",
        label_confidence="high",
    )


def _t_subpage_generic_section_detail(
    page: V4Page,
) -> Optional[QueryTemplate]:
    if not page.page_title or not _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = (parent, page.page_title, "정보") + tuple(
        n for n in nouns if n and n not in (parent, page.page_title)
    )
    return QueryTemplate(
        text=f"{parent}의 {page.page_title} 정보를 자세히 알려줘.",
        keywords=keywords[:5],
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket=BUCKET_SUBPAGE_GENERIC,
        template_kind=TEMPLATE_SECTION_DETAIL,
        label_source="subpage_pivot",
        label_confidence="medium",
    )


def _t_subpage_generic_section_question(
    page: V4Page,
) -> Optional[QueryTemplate]:
    if not page.page_title or not _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    keywords = (parent, page.page_title, "내용")
    return QueryTemplate(
        text=f"{parent} {page.page_title}은 어떤 내용인가요?",
        keywords=keywords,
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket=BUCKET_SUBPAGE_GENERIC,
        template_kind=TEMPLATE_SECTION_QUESTION,
        label_source="subpage_pivot",
        label_confidence="medium",
        extra_meta={"section_anchor": page.page_title},
    )


def _t_subpage_named_lookup(page: V4Page) -> Optional[QueryTemplate]:
    if page.relation == "main":
        return None
    if not page.page_title or _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=2)
    keywords = (page.page_title, parent) + tuple(nouns)
    return QueryTemplate(
        text=f"{parent}의 {page.page_title}에 대해 알려주세요.",
        keywords=keywords[:4],
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket=BUCKET_SUBPAGE_NAMED,
        template_kind=TEMPLATE_NAMED_LOOKUP,
        label_source="subpage_pivot",
        label_confidence="high",
    )


def _t_subpage_named_question(page: V4Page) -> Optional[QueryTemplate]:
    if page.relation == "main":
        return None
    if not page.page_title or _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    keywords = (page.page_title, parent, "등장인물", "내용")
    return QueryTemplate(
        text=(
            f"{page.page_title}는(은) {parent}에서 어떤 인물/내용인가요?"
        ),
        keywords=keywords,
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket=BUCKET_SUBPAGE_NAMED,
        template_kind=TEMPLATE_NAMED_QUESTION,
        label_source="subpage_pivot",
        label_confidence="medium",
    )


def _t_subpage_named_alias(page: V4Page) -> Optional[QueryTemplate]:
    if page.relation == "main":
        return None
    if not page.page_title or _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    if not _has_alias_distinct_from_work(page):
        return None
    alias = _first_distinct_alias(page)
    if not alias:
        return None
    keywords = (alias, parent, page.page_title)
    return QueryTemplate(
        text=f"{alias}는(은) {parent}에서 어떤 인물인가요?",
        keywords=keywords,
        answer_type="alias_lookup",
        difficulty="hard",
        bucket=BUCKET_SUBPAGE_NAMED,
        template_kind=TEMPLATE_NAMED_ALIAS,
        label_source="alias_pivot",
        label_confidence="low",
        extra_meta={"alias_used": alias},
    )


# Ordered (bucket, template_kind, fn) for deterministic application.
_TEMPLATES: Tuple[Tuple[str, str, Callable[[V4Page], Optional[QueryTemplate]]], ...] = (
    (BUCKET_MAIN_WORK, TEMPLATE_TITLE_LOOKUP, _t_main_title_lookup),
    (BUCKET_MAIN_WORK, TEMPLATE_PLOT_SUMMARY, _t_main_plot_summary),
    (BUCKET_MAIN_WORK, TEMPLATE_EVALUATION, _t_main_evaluation),
    (BUCKET_MAIN_WORK, TEMPLATE_ALIAS_LOOKUP, _t_main_alias_lookup),
    (BUCKET_MAIN_WORK, TEMPLATE_AMBIGUOUS_SHORT, _t_main_ambiguous_short),
    (BUCKET_SUBPAGE_GENERIC, TEMPLATE_SECTION_LOOKUP, _t_subpage_generic_section_lookup),
    (BUCKET_SUBPAGE_GENERIC, TEMPLATE_SECTION_DETAIL, _t_subpage_generic_section_detail),
    (BUCKET_SUBPAGE_GENERIC, TEMPLATE_SECTION_QUESTION, _t_subpage_generic_section_question),
    (BUCKET_SUBPAGE_NAMED, TEMPLATE_NAMED_LOOKUP, _t_subpage_named_lookup),
    (BUCKET_SUBPAGE_NAMED, TEMPLATE_NAMED_QUESTION, _t_subpage_named_question),
    (BUCKET_SUBPAGE_NAMED, TEMPLATE_NAMED_ALIAS, _t_subpage_named_alias),
)


# ---------------------------------------------------------------------------
# Sampling pool builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One (page, template) candidate before stratified selection."""

    page: V4Page
    template: QueryTemplate


def _build_candidate_pool(
    pages: Iterable[V4Page],
) -> Dict[str, List[_Candidate]]:
    """Apply every template to every page → bucket-keyed candidate lists.

    A page can produce multiple candidates (one per matching template);
    the orchestrator de-duplicates by ``page_id`` per bucket so the same
    page can't dominate a bucket.
    """
    pool: Dict[str, List[_Candidate]] = {b: [] for b in BUCKETS}
    for page in pages:
        for bucket, _kind, fn in _TEMPLATES:
            tmpl = fn(page)
            if tmpl is None:
                continue
            if tmpl.bucket != bucket:
                # Defensive: every fn already pins the bucket but the
                # contract reads cleaner if we double-check.
                continue
            pool[bucket].append(_Candidate(page=page, template=tmpl))
    return pool


def _stratified_sample(
    pool: Dict[str, List[_Candidate]],
    *,
    targets: Mapping[str, int],
    seed: int,
) -> Tuple[List[_Candidate], Dict[str, int], Dict[str, int]]:
    """Pick ``targets[bucket]`` candidates per bucket without replacement.

    Returns (selected, actual_counts, deficits). ``actual_counts`` is
    how many were taken from each bucket; ``deficits`` is the
    target − actual gap (always ≥ 0). The selection is deterministic:
    we shuffle each bucket's candidate list with the seed and slice off
    the head, with template-rotation to avoid one template dominating
    a bucket when the page count is much larger than the target.
    """
    rng = random.Random(seed)
    actual: Dict[str, int] = {}
    deficits: Dict[str, int] = {}
    selected: List[_Candidate] = []

    for bucket in BUCKETS:
        target = max(0, int(targets.get(bucket, 0)))
        bucket_pool = list(pool.get(bucket, ()))
        # Stable sort by (page_id, template_kind) so the rotation below
        # is reproducible regardless of source order. The actual order
        # selected inside each (page_id, template_kind) group is
        # determined by the rng shuffle below.
        bucket_pool.sort(key=lambda c: (c.page.page_id, c.template.template_kind))

        # Group by template_kind so we can rotate templates rather than
        # taking 50 of the first kind before any of the next.
        by_kind: Dict[str, List[_Candidate]] = {}
        for c in bucket_pool:
            by_kind.setdefault(c.template.template_kind, []).append(c)
        for kind in list(by_kind.keys()):
            rng.shuffle(by_kind[kind])

        # Keep template kinds in their bucket-defined order so the
        # rotation is stable across runs.
        ordered_kinds: List[str] = [
            kind for _b, kind, _fn in _TEMPLATES if _b == bucket and kind in by_kind
        ]

        # Round-robin draw — ensures coverage of all template kinds
        # before any kind is exhausted.
        seen_pages: set = set()
        picks: List[_Candidate] = []
        kind_idx = 0
        empty_passes = 0
        while len(picks) < target and ordered_kinds:
            kind = ordered_kinds[kind_idx % len(ordered_kinds)]
            kind_idx += 1
            queue = by_kind.get(kind) or []
            if not queue:
                empty_passes += 1
                if empty_passes >= len(ordered_kinds):
                    break
                continue
            empty_passes = 0  # reset — we found something
            cand = queue.pop()
            if cand.page.page_id in seen_pages:
                # Drop and try again — same page can't appear twice in
                # the same bucket regardless of template kind.
                continue
            seen_pages.add(cand.page.page_id)
            picks.append(cand)

        actual[bucket] = len(picks)
        deficits[bucket] = max(0, target - len(picks))
        selected.extend(picks)

    # Final shuffle so consumer files don't read in bucket-blocks; the
    # rng state already mixed in the bucket-level shuffles above so this
    # remains deterministic.
    rng.shuffle(selected)
    return selected, actual, deficits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class SilverGenerationResult:
    """Container for the silver-500 emission."""

    queries: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def generate_silver_500(
    pages_v4_path: Path,
    *,
    target_total: int = 500,
    bucket_targets: Optional[Mapping[str, int]] = None,
    seed: int = 42,
    restrict_doc_ids: Optional[Sequence[str]] = None,
) -> SilverGenerationResult:
    """Render the silver-500 set + a summary block.

    ``bucket_targets`` defaults to (main_work=150, subpage_generic=200,
    subpage_named=150). When ``target_total`` differs from the sum of
    ``bucket_targets`` the sum wins — the orchestrator does *not*
    re-scale targets to match ``target_total`` because the per-bucket
    distribution is the contract, not the total. ``target_total`` is
    used only by the summary report.
    """
    targets = dict(bucket_targets) if bucket_targets else dict(DEFAULT_BUCKET_TARGETS)
    for b in BUCKETS:
        targets.setdefault(b, 0)
    requested_total = int(sum(targets.values()))

    restrict = set(restrict_doc_ids) if restrict_doc_ids else None

    eligible_pages: List[V4Page] = []
    page_type_counts: Counter = Counter()
    for page in iter_v4_pages(Path(pages_v4_path)):
        if restrict and page.page_id not in restrict:
            continue
        eligible_pages.append(page)
        page_type_counts[page.page_type] += 1

    pool = _build_candidate_pool(eligible_pages)
    pool_counts = {b: len(pool[b]) for b in BUCKETS}
    pool_template_counts: Dict[str, Dict[str, int]] = {b: {} for b in BUCKETS}
    for bucket, cands in pool.items():
        kinds: Counter = Counter(c.template.template_kind for c in cands)
        pool_template_counts[bucket] = dict(kinds)

    selected, actual_counts, deficits = _stratified_sample(
        pool, targets=targets, seed=seed,
    )

    queries: List[Dict[str, Any]] = []
    template_kind_counts: Dict[str, int] = {k: 0 for k in ALL_TEMPLATES}
    bucket_template_counts: Dict[str, Dict[str, int]] = {b: {} for b in BUCKETS}
    label_confidence_counts: Counter = Counter()
    has_alias_count = 0
    for i, cand in enumerate(selected, start=1):
        page = cand.page
        tmpl = cand.template
        template_kind_counts[tmpl.template_kind] = (
            template_kind_counts.get(tmpl.template_kind, 0) + 1
        )
        bt = bucket_template_counts.setdefault(tmpl.bucket, {})
        bt[tmpl.template_kind] = bt.get(tmpl.template_kind, 0) + 1
        label_confidence_counts[tmpl.label_confidence] += 1
        if _has_alias_distinct_from_work(page):
            has_alias_count += 1

        record: Dict[str, Any] = {
            "id": f"v4-silver-500-{i:04d}",
            "query": tmpl.text,
            "language": "ko",
            "expected_doc_ids": [page.page_id],
            "expected_section_keywords": list(tmpl.keywords),
            "answer_type": tmpl.answer_type,
            "difficulty": tmpl.difficulty,
            "tags": [
                "anime",
                "v4-silver-500",
                "synthetic",
                "silver",
                tmpl.bucket,
                tmpl.answer_type,
                tmpl.template_kind,
                "deterministic",
            ],
            "v4_meta": {
                "bucket": tmpl.bucket,
                "template_kind": tmpl.template_kind,
                "page_type": page.page_type,
                "relation": page.relation,
                "page_title": page.page_title,
                "work_title": page.work_title,
                "retrieval_title": page.retrieval_title,
                "title_source": page.title_source,
                "alias_source": page.alias_source,
                "is_generic_page_title": _is_generic_page_title(page.page_title),
                "silver_label_source": tmpl.label_source,
                "silver_label_confidence": tmpl.label_confidence,
                "is_silver_not_gold": True,
                "extra": dict(tmpl.extra_meta),
            },
        }
        queries.append(record)

    summary: Dict[str, Any] = {
        "schema": "queries-v4-silver-500.summary.v1",
        "is_silver_not_gold": True,
        "disclaimer": SILVER_DISCLAIMER_LINE,
        "audit_disclaimer": SILVER_AUDIT_LINE,
        "seed": int(seed),
        "target_total": int(target_total),
        "requested_total": requested_total,
        "actual_total": len(queries),
        "bucket_targets": dict(targets),
        "bucket_actual_counts": dict(actual_counts),
        "bucket_deficits": dict(deficits),
        "candidate_pool_counts": pool_counts,
        "candidate_pool_template_counts": pool_template_counts,
        "template_kind_counts": dict(template_kind_counts),
        "bucket_template_counts": bucket_template_counts,
        "label_confidence_counts": dict(label_confidence_counts),
        "queries_with_distinct_alias": has_alias_count,
        "page_type_counts_eligible": dict(page_type_counts),
        "all_buckets": list(BUCKETS),
        "all_templates": list(ALL_TEMPLATES),
    }

    return SilverGenerationResult(queries=queries, summary=summary)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_silver_500_jsonl(
    queries: Sequence[Mapping[str, Any]], out_path: Path,
) -> Path:
    """Persist the silver-500 set as JSONL (one record per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for q in queries:
            fp.write(json.dumps(q, ensure_ascii=False) + "\n")
    return out_path


def write_silver_500_summary_json(
    summary: Mapping[str, Any], out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def render_silver_500_summary_md(summary: Mapping[str, Any]) -> str:
    """Render a human-readable summary report.

    The report leads with the silver-vs-gold disclaimer and the audit
    disclaimer asked for by the spec — both lines are spelled exactly
    so a downstream test can grep for them. Buckets, template kinds,
    and label-confidence counts are tabulated for quick eyeballing.
    """
    lines: List[str] = []
    lines.append("# Phase 7 silver-500 query set summary")
    lines.append("")
    lines.append(f"> {SILVER_DISCLAIMER_LINE}")
    lines.append(">")
    lines.append(f"> {SILVER_AUDIT_LINE}")
    lines.append("")
    lines.append(
        f"- seed: **{summary.get('seed')}**"
    )
    lines.append(
        f"- target_total (requested by caller): **{summary.get('target_total')}**"
    )
    lines.append(
        f"- requested_total (sum of bucket targets): "
        f"**{summary.get('requested_total')}**"
    )
    lines.append(
        f"- actual_total (rows emitted): "
        f"**{summary.get('actual_total')}**"
    )
    lines.append("")

    lines.append("## Bucket distribution")
    lines.append("")
    lines.append("| bucket | target | actual | deficit | pool |")
    lines.append("|---|---:|---:|---:|---:|")
    targets = summary.get("bucket_targets") or {}
    actual = summary.get("bucket_actual_counts") or {}
    deficits = summary.get("bucket_deficits") or {}
    pool = summary.get("candidate_pool_counts") or {}
    for b in BUCKETS:
        lines.append(
            f"| {b} | {int(targets.get(b, 0))} | {int(actual.get(b, 0))} | "
            f"{int(deficits.get(b, 0))} | {int(pool.get(b, 0))} |"
        )
    lines.append("")

    lines.append("## Template-kind distribution")
    lines.append("")
    lines.append("| bucket | template_kind | count |")
    lines.append("|---|---|---:|")
    bucket_templates = summary.get("bucket_template_counts") or {}
    for b in BUCKETS:
        kinds = bucket_templates.get(b) or {}
        ordered = [
            k for _b, k, _fn in _TEMPLATES if _b == b
        ]
        for k in ordered:
            count = int(kinds.get(k, 0))
            if count > 0 or _b_eq(k, b):
                lines.append(f"| {b} | {k} | {count} |")
    lines.append("")

    lines.append("## Silver-label confidence distribution")
    lines.append("")
    lines.append("| confidence | count |")
    lines.append("|---|---:|")
    conf = summary.get("label_confidence_counts") or {}
    for k in ("high", "medium", "low"):
        lines.append(f"| {k} | {int(conf.get(k, 0))} |")
    lines.append("")

    lines.append("## Alias coverage")
    lines.append("")
    lines.append(
        f"- queries with a distinct alias on the silver-target page: "
        f"**{int(summary.get('queries_with_distinct_alias', 0))}**"
    )
    lines.append("")

    lines.append("## Eligible page types")
    lines.append("")
    pt = summary.get("page_type_counts_eligible") or {}
    if pt:
        lines.append("| page_type | count |")
        lines.append("|---|---:|")
        for k in sorted(pt.keys()):
            lines.append(f"| {k} | {int(pt[k])} |")
        lines.append("")

    lines.append("## Reminder")
    lines.append("")
    lines.append(
        "- This set is **silver / synthetic / deterministic**. "
        "Targets and label sources are derived from page metadata; "
        "no human reviewer has confirmed any (query, expected_doc_id) "
        "pair."
    )
    lines.append(
        "- Use the `human_gold_seed_export` exporter to draw a "
        "stratified audit sample from this set + the Phase 7.3/7.4 "
        "outputs, then have a human fill the `human_label` column "
        "before reporting any precision/recall/accuracy number."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _b_eq(_kind: str, _bucket: str) -> bool:
    """Tiny helper — keep template rows present even at zero count.

    The summary table is more useful when readers see "this template
    *could* fire here but didn't". The check is trivially True, but it
    keeps the conditional explicit so a future contributor sees why
    we're emitting a zero row.
    """
    return True


def write_silver_500_summary_md(
    summary: Mapping[str, Any], out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_silver_500_summary_md(summary), encoding="utf-8")
    return out_path


def write_silver_500_outputs(
    result: SilverGenerationResult,
    *,
    out_dir: Path,
    jsonl_name: str = "queries_v4_silver_500.jsonl",
    summary_json_name: str = "queries_v4_silver_500.summary.json",
    summary_md_name: str = "queries_v4_silver_500.summary.md",
) -> Dict[str, Path]:
    """Persist the three artefacts the spec asks for in one call."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = write_silver_500_jsonl(result.queries, out_dir / jsonl_name)
    summary_json_path = write_silver_500_summary_json(
        result.summary, out_dir / summary_json_name,
    )
    summary_md_path = write_silver_500_summary_md(
        result.summary, out_dir / summary_md_name,
    )
    return {
        "jsonl": jsonl_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
    }
