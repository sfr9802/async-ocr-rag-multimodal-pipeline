"""Phase 7.0 — deterministic silver query generator over v4 pages.

The legacy ``anime_silver_200.jsonl`` query set was generated against the
v3 corpus and uses 24-char doc_ids that don't map cleanly to Phase 6.3's
16-char ``page_id`` (only 50/200 unambiguous matches by page_title; 92
unmatched outright). Rather than fudge a join, we build a small
v4-native silver set targeted at the A/B question: does
``retrieval_title``-prefixed embedding help retrieve the *correct*
sub-page when the query refers to a work via its parent title?

Strategy is deterministic (zero LLM calls, byte-identical reruns under
the seed) and stratified along the page-type axis Phase 6.3 makes
explicit:

  1. **Subpage with generic page_title** (e.g. page_title=="등장인물",
     work_title=="가난뱅이 신이!", retrieval_title=="가난뱅이 신이! / 등장인물").
     Query mentions the work_title + the section name; the correct
     answer is the *subpage's* page_id, not the main page. These are
     the rows the variant should help.
  2. **Main work page** (page_type=="work", relation=="main",
     page_title == work_title). Query asks for the plot/genre using
     the title. Neutral by construction (retrieval_title == page_title)
     so any variant-driven swing here is signal we did NOT want.
  3. **Subpage with non-generic page_title** (e.g. character page that
     has its own name). Query mentions the character name + work_title.

Output schema mirrors ``anime_silver_200.jsonl`` so the existing
``run_eval`` and ``baseline_comparison`` machinery can consume the file
without changes (``id``, ``query``, ``language``, ``expected_doc_ids``,
``expected_section_keywords``, ``answer_type``, ``difficulty``, ``tags``).
"""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


# Generic page_titles Phase 6.3's audit calls out; these are the rows
# where retrieval_title materially diverges from page_title and where
# the variant has a chance to move the needle.
_GENERIC_PAGE_TITLES: frozenset = frozenset({
    "등장인물", "평가", "OST", "기타", "회차", "에피소드", "주제가",
    "음악", "회차 목록", "에피소드 가이드", "미디어 믹스", "기타 등장인물",
    "설정", "줄거리", "스태프", "성우진",
})

_NOUN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9]{2,}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_NOUN_STOPLIST: frozenset = frozenset({
    "이야기", "작품", "내용", "주인공", "사람", "이상", "정도",
    "이번", "관련", "다음", "사이", "일이", "상황", "사실", "경우",
    "그것", "어느", "처럼", "하지만", "그러나", "그래서",
    "anime", "manga", "story", "the", "and", "for", "with", "from",
})


@dataclass(frozen=True)
class V4Page:
    """Compact view of a Phase 6.3 page record for query generation.

    Carries only the fields the deterministic templates touch. Loaded
    once per pages_v4 pass; we then keep the dataclass list rather than
    the full ~366MB dict to bound peak memory.
    """

    page_id: str
    work_id: str
    work_title: str
    page_title: str
    page_type: str
    relation: str
    canonical_url: str
    aliases: Tuple[str, ...]
    display_title: str
    retrieval_title: str
    title_source: str
    alias_source: str
    first_section_text: str
    first_section_heading: str
    section_count: int


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = _CONTROL_RE.sub(" ", s).strip()
    return re.sub(r"\s+", " ", s)


def _extract_nouns(text: str, *, limit: int = 6) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    seen: List[str] = []
    seen_set: set = set()
    for m in _NOUN_RE.finditer(text):
        token = m.group(0)
        norm = unicodedata.normalize("NFKC", token).strip()
        if not norm or norm.lower() in _NOUN_STOPLIST:
            continue
        if norm in seen_set:
            continue
        seen.append(norm)
        seen_set.add(norm)
        if len(seen) >= limit:
            break
    return seen


def iter_v4_pages(pages_v4_path: Path) -> Iterator[V4Page]:
    """Yield :class:`V4Page` for each record in ``pages_v4.jsonl``."""
    with Path(pages_v4_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sections = rec.get("sections") or []
            first_text = ""
            first_heading = ""
            if isinstance(sections, list) and sections:
                first = sections[0]
                if isinstance(first, dict):
                    first_text = _clean_text(
                        first.get("clean_text") or first.get("text") or ""
                    )
                    hp = first.get("heading_path") or []
                    if isinstance(hp, list) and hp:
                        first_heading = " > ".join(str(h) for h in hp if h)
            yield V4Page(
                page_id=str(rec.get("page_id") or ""),
                work_id=str(rec.get("work_id") or ""),
                work_title=str(rec.get("work_title") or ""),
                page_title=str(rec.get("page_title") or ""),
                page_type=str(rec.get("page_type") or ""),
                relation=str(rec.get("relation") or ""),
                canonical_url=str(rec.get("canonical_url") or ""),
                aliases=tuple(str(a) for a in (rec.get("aliases") or [])),
                display_title=str(rec.get("display_title") or ""),
                retrieval_title=str(rec.get("retrieval_title") or ""),
                title_source=str(rec.get("title_source") or ""),
                alias_source=str(rec.get("alias_source") or ""),
                first_section_text=first_text[:1200],
                first_section_heading=first_heading,
                section_count=(
                    len(sections) if isinstance(sections, list) else 0
                ),
            )


@dataclass(frozen=True)
class QueryTemplate:
    """Internal — one rendered query candidate for a page."""

    text: str
    keywords: List[str]
    answer_type: str
    difficulty: str
    bucket: str  # "subpage_generic" | "subpage_named" | "main_work"


def _is_generic_page_title(page_title: str) -> bool:
    if not page_title:
        return False
    if page_title in _GENERIC_PAGE_TITLES:
        return True
    # Substring rule for compound titles like "기타 등장인물" / "OST 모음" —
    # mirrors Phase 6.3's `is_generic_title_for_display`.
    for kw in _GENERIC_PAGE_TITLES:
        if kw in page_title:
            return True
    return False


def _parent_work_from_retrieval_title(
    retrieval_title: str, page_title: str,
) -> str:
    """Extract the parent work portion of a Phase 6.3 retrieval_title.

    For subpages whose page_title is generic, Phase 6.3 stores the
    retrieval_title as ``"{work_title}/{page_title}"`` (or with a
    space-padded ``" / "``). The page-level ``work_title`` field is
    *not* reliable here — for many subpages it was preserved as the
    page_title verbatim by the v4 conversion, so the only reliable
    way to recover the parent work is to split the retrieval_title.

    Returns ``""`` when retrieval_title carries no parent prefix
    (main work pages, or pages where retrieval_title == page_title).
    """
    if not retrieval_title:
        return ""
    page_title = (page_title or "").strip()
    rt = retrieval_title.strip()
    if not rt:
        return ""
    # Phase 6.3 emits both " / " and "/" as separators across the
    # corpus; we accept either.
    for sep in (" / ", "/"):
        if sep in rt:
            head, sep_, tail = rt.partition(sep)
            head = head.strip()
            tail = tail.strip()
            if head and tail and head != page_title:
                return head
    return ""


def _build_subpage_generic(page: V4Page) -> Optional[QueryTemplate]:
    """Query for a subpage whose page_title is generic.

    The retrieval target is the *subpage's* page_id, but the query only
    has the work title to anchor on (because the page_title alone —
    "등장인물" — is non-discriminating across thousands of pages). This
    is the bucket where retrieval_title should bring the most lift.

    Parent work title is recovered from ``retrieval_title`` (the
    Phase 6.3-folded form) rather than from ``page.work_title``, which
    on subpages was often preserved verbatim from page_title and is
    therefore not a reliable parent signal.
    """
    if not page.page_title:
        return None
    if not _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        # Without a parent prefix we'd produce a degenerate query
        # ("등장인물의 등장인물에 대해 알려주세요"). Skip — these rows
        # have no signal for retrieval_title to help with anyway.
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = [parent, page.page_title] + [
        n for n in nouns if n and n not in (parent, page.page_title)
    ]
    query = (
        f"{parent}의 {page.page_title}에 대해 알려주세요."
    )
    return QueryTemplate(
        text=query,
        keywords=keywords[:5],
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket="subpage_generic",
    )


def _build_main_work(page: V4Page) -> Optional[QueryTemplate]:
    """Query for a main work page (neutral baseline bucket)."""
    if page.page_type != "work" or page.relation != "main":
        return None
    title = page.page_title or page.work_title
    if not title:
        return None
    nouns = _extract_nouns(page.first_section_text, limit=3)
    keywords = [title] + [n for n in nouns if n and n != title]
    query = f"{title}이(가) 어떤 작품인가요?"
    return QueryTemplate(
        text=query,
        keywords=keywords[:4],
        answer_type="title_lookup",
        difficulty="easy",
        bucket="main_work",
    )


def _build_subpage_named(page: V4Page) -> Optional[QueryTemplate]:
    """Query for a subpage with a non-generic, work-specific page_title.

    These pages already have a discriminative ``page_title`` so the
    baseline ``title_section`` variant ought to retrieve them well.
    Including them in the silver set keeps the A/B honest: if
    retrieval_title degrades performance for non-generic subpages,
    the regression list will surface it.

    Like :func:`_build_subpage_generic` we recover the parent from
    ``retrieval_title``; if it disagrees with ``page_title`` (true for
    real subpages) we use that as the work anchor. Pages without a
    parent prefix are skipped — their query would degenerate.
    """
    if page.relation == "main":
        return None
    if not page.page_title or _is_generic_page_title(page.page_title):
        return None
    parent = _parent_work_from_retrieval_title(
        page.retrieval_title, page.page_title,
    )
    if not parent:
        return None
    keywords = [page.page_title, parent] + _extract_nouns(
        page.first_section_text, limit=2,
    )
    query = (
        f"{parent}의 {page.page_title}에 대해 알려주세요."
    )
    return QueryTemplate(
        text=query,
        keywords=keywords[:4],
        answer_type="subpage_lookup",
        difficulty="medium",
        bucket="subpage_named",
    )


_BUILDERS = (
    ("subpage_generic", _build_subpage_generic),
    ("main_work", _build_main_work),
    ("subpage_named", _build_subpage_named),
)


def _build_for_page(page: V4Page) -> Optional[QueryTemplate]:
    """Try each template in priority order; first match wins."""
    for _, builder in _BUILDERS:
        out = builder(page)
        if out is not None:
            return out
    return None


def generate_v4_silver_queries(
    pages_v4_path: Path,
    *,
    target_total: int = 200,
    seed: int = 42,
    bucket_ratios: Optional[Dict[str, float]] = None,
    restrict_doc_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Render a deterministic stratified v4 silver set.

    ``bucket_ratios`` defaults to a 0.45 / 0.30 / 0.25 split over
    (subpage_generic / main_work / subpage_named). The split was chosen
    to over-sample the bucket where retrieval_title is supposed to
    help so the A/B has signal density, while still leaving room for
    neutral and adverse buckets so a regression in either is visible.
    """
    ratios = dict(bucket_ratios or {
        "subpage_generic": 0.45,
        "main_work": 0.30,
        "subpage_named": 0.25,
    })
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"bucket_ratios must sum to 1.0; got {ratios}")

    restrict = set(restrict_doc_ids) if restrict_doc_ids else None

    by_bucket: Dict[str, List[Tuple[V4Page, QueryTemplate]]] = {
        "subpage_generic": [], "main_work": [], "subpage_named": [],
    }
    for page in iter_v4_pages(pages_v4_path):
        if restrict and page.page_id not in restrict:
            continue
        tmpl = _build_for_page(page)
        if tmpl is None:
            continue
        by_bucket[tmpl.bucket].append((page, tmpl))

    rng = random.Random(seed)
    for entries in by_bucket.values():
        rng.shuffle(entries)

    targets = {
        b: max(0, int(round(target_total * ratios.get(b, 0.0))))
        for b in by_bucket
    }
    # Adjust rounding so the buckets sum to target_total exactly.
    deficit = target_total - sum(targets.values())
    if deficit:
        # Order by how many candidates a bucket actually has; fill the
        # bucket with the most headroom first so we don't over-truncate
        # an undersupplied bucket.
        order = sorted(
            by_bucket.keys(),
            key=lambda b: len(by_bucket[b]) - targets[b],
            reverse=(deficit > 0),
        )
        sign = 1 if deficit > 0 else -1
        for _ in range(abs(deficit)):
            for b in order:
                if sign > 0 and targets[b] < len(by_bucket[b]):
                    targets[b] += 1
                    break
                if sign < 0 and targets[b] > 0:
                    targets[b] -= 1
                    break

    selected: List[Tuple[V4Page, QueryTemplate]] = []
    for b in ("subpage_generic", "main_work", "subpage_named"):
        selected.extend(by_bucket[b][: targets[b]])
    rng.shuffle(selected)

    out: List[Dict[str, Any]] = []
    for i, (page, tmpl) in enumerate(selected, start=1):
        out.append({
            "id": f"v4-silver-{i:04d}",
            "query": tmpl.text,
            "language": "ko",
            "expected_doc_ids": [page.page_id],
            "expected_section_keywords": tmpl.keywords,
            "answer_type": tmpl.answer_type,
            "difficulty": tmpl.difficulty,
            "tags": [
                "anime", "v4-silver", "synthetic", tmpl.bucket,
                tmpl.answer_type, "deterministic",
            ],
            # Phase 7.0-only diagnostic so per-query analysis can stratify.
            "v4_meta": {
                "bucket": tmpl.bucket,
                "page_type": page.page_type,
                "relation": page.relation,
                "page_title": page.page_title,
                "work_title": page.work_title,
                "retrieval_title": page.retrieval_title,
                "title_source": page.title_source,
                "alias_source": page.alias_source,
                "is_generic_page_title": _is_generic_page_title(page.page_title),
            },
        })
    return out


def write_v4_silver_queries(
    queries: List[Dict[str, Any]], out_path: Path,
) -> None:
    """Write rendered queries as JSONL (one record per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for q in queries:
            fp.write(json.dumps(q, ensure_ascii=False) + "\n")
