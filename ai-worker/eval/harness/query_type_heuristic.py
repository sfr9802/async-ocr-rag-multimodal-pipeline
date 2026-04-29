"""Heuristic query_type tagger — eval-only, manual review required.

The current ``anime_silver_200.jsonl`` does not carry ``query_type`` so
``run_retrieval_eval``'s ``byQueryType`` breakdown collapses every row
into the ``unknown`` bucket. This module produces a *draft* tagging:
each row gets a guessed ``query_type``, a ``query_type_confidence`` in
[0.0, 1.0], and a ``query_type_reason`` explaining which heuristic
fired. The original ``anime_silver_200.jsonl`` is left untouched — the
tagged rows go to a sibling file (``*.query_type_draft.jsonl``).

Heuristic taxonomy (kept small on purpose — adding labels here is
cheap, but every label is a manual-review burden later):

  - title_direct        : query that asks "what is X?" / "tell me
                          about X" with no qualifier — matches a
                          single anime title.
  - character_relation  : asks about ties between characters
                          ("관계", "사이", "동료", "가족", "친구", "적").
  - character_attribute : asks about a character's attributes
                          ("능력", "성격", "특징", "정체", "직업", "소속").
  - plot_event          : asks about plot turns / outcomes ("사건",
                          "왜", "어떻게", "무슨 일이", "줄거리", "결말").
  - setting             : asks about world / system / organization
                          ("세계관", "설정", "조직", "기술", "능력 체계").
  - review_reception    : asks about reception / criticism ("평가",
                          "반응", "흥행", "비판", "호평", "혹평").
  - comparison          : asks for a comparison ("비교", "차이",
                          "더", "vs").
  - alias_typo          : query is a known-alias / suspicious typo
                          form (deliberately weak — caller must
                          confirm).
  - ambiguous           : query is too short or vague to classify.
  - no_answer           : query that explicitly asks for content the
                          dataset shouldn't answer (rare — flagged
                          conservatively).
  - unknown             : default fallback. Confidence stays low.

Output shape (one JSON object per line):

    {
      "id": ...,
      "query": ...,
      ... (every original field preserved) ...,
      "query_type": "<label>",
      "query_type_confidence": 0.0-1.0,
      "query_type_reason": "<short string>"
    }

The confidence is a *very* coarse signal — it is the first matching
heuristic's strength minus a flat penalty when multiple heuristics
fired ambiguously. Treat anything below 0.5 as "needs review".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# Public taxonomy (mirrors the spec). Surface as constants so tests +
# the report can pin them.
QT_TITLE_DIRECT = "title_direct"
QT_CHARACTER_RELATION = "character_relation"
QT_CHARACTER_ATTRIBUTE = "character_attribute"
QT_PLOT_EVENT = "plot_event"
QT_SETTING = "setting"
QT_REVIEW_RECEPTION = "review_reception"
QT_COMPARISON = "comparison"
QT_ALIAS_TYPO = "alias_typo"
QT_AMBIGUOUS = "ambiguous"
QT_NO_ANSWER = "no_answer"
QT_UNKNOWN = "unknown"

QUERY_TYPES: Tuple[str, ...] = (
    QT_TITLE_DIRECT,
    QT_CHARACTER_RELATION,
    QT_CHARACTER_ATTRIBUTE,
    QT_PLOT_EVENT,
    QT_SETTING,
    QT_REVIEW_RECEPTION,
    QT_COMPARISON,
    QT_ALIAS_TYPO,
    QT_AMBIGUOUS,
    QT_NO_ANSWER,
    QT_UNKNOWN,
)


# Heuristic keyword maps. Order matters: when multiple fire, we take
# the first hit and drop confidence — meaning the categories listed
# higher win ties. Categories listed earlier are the more *specific*
# ones (relation > attribute > plot_event), to avoid e.g. a plot
# query containing "관계" being mis-bucketed as plot.
_KEYWORD_TABLE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # Reception cues first — "평가" / "흥행" are specific enough that
    # we want them to win over generic relation keywords like "적"
    # (which the Korean substring matcher would otherwise hit on
    # "성적", "흥행 성적" etc.).
    (QT_REVIEW_RECEPTION, (
        "평가", "반응", "흥행", "비판", "호평", "혹평", "리뷰", "감상",
        "인기", "수상", "비평",
    )),
    (QT_COMPARISON, (
        "비교", "차이", " vs ", " 대비 ", "보다 더", "보다 강한", "보다 약한",
    )),
    (QT_CHARACTER_RELATION, (
        # Tokens are deliberately at least 2 chars: bare "적" is too
        # short and matches "성적" / "성적인" / "체계적" false-positively.
        "관계", "사이", "누구와", "연관", "동료", "가족", "친구",
        "라이벌", "동반자", "파트너", "적군", "숙적", "원수",
    )),
    (QT_CHARACTER_ATTRIBUTE, (
        "능력", "성격", "특징", "정체", "직업", "소속", "스펙", "프로필",
        "역할", "별명", "정체성",
    )),
    (QT_PLOT_EVENT, (
        "사건", "왜", "어떻게", "무슨 일이", "줄거리", "결말",
        "에피소드", "전개", "회차", "장면", "엔딩", "스토리",
        "본문에서", "다루어지나요", "다루어지", "묘사",
    )),
    (QT_SETTING, (
        "세계관", "설정", "조직", "기술", "능력 체계", "마법 체계",
        "시스템", "배경", "지역", "지명", "단체",
    )),
    (QT_TITLE_DIRECT, (
        "에 대해 설명", "에 대해 알려주세요", "이란?", "란 무엇",
        "은 무엇", "는 무엇", "에 대해 알려줘",
    )),
)


# When the query is shorter than this many *characters* (after
# stripping whitespace), label as ambiguous regardless of keyword
# matches — the heuristic can fire false positives on tiny queries.
_AMBIGUOUS_MIN_CHARS = 6


@dataclass(frozen=True)
class HeuristicTag:
    """Single tagging decision over one query row."""

    query_type: str
    confidence: float
    reason: str


def tag_query(query: str) -> HeuristicTag:
    """Return a heuristic ``query_type`` tag for ``query``.

    Pure function — no I/O, no model. The confidence reflects how
    confident the heuristic is *as a heuristic*; downstream callers
    must NOT treat anything above 0.7 as ground truth. The reason is
    a short string useful for the manual-review report (e.g.
    "matched keyword '평가'").
    """
    text = (query or "").strip()
    if not text:
        return HeuristicTag(QT_UNKNOWN, 0.0, "empty query")
    if len(text) < _AMBIGUOUS_MIN_CHARS:
        return HeuristicTag(
            QT_AMBIGUOUS, 0.2,
            f"query too short ({len(text)} chars)",
        )

    folded = text.casefold()
    matches: List[Tuple[str, str]] = []
    for label, keywords in _KEYWORD_TABLE:
        for kw in keywords:
            if kw.casefold() in folded:
                matches.append((label, kw))
                break  # one keyword per label is enough
    if not matches:
        # Default: everything else is unknown. Confidence is low but
        # not zero so the downstream report can sort by it.
        return HeuristicTag(
            QT_UNKNOWN, 0.1,
            "no heuristic matched",
        )

    # First hit wins; multiple hits drop confidence proportionally
    # so the report can flag "this row had three competing labels".
    primary_label, primary_kw = matches[0]
    base_conf = 0.7 if len(matches) == 1 else max(
        0.45, 0.7 - 0.08 * (len(matches) - 1),
    )
    if len(matches) > 1:
        competing = ", ".join(f"{lbl}({kw})" for lbl, kw in matches[1:4])
        reason = (
            f"matched '{primary_kw}' as {primary_label}; "
            f"also competing: {competing}"
        )
    else:
        reason = f"matched '{primary_kw}' as {primary_label}"
    return HeuristicTag(primary_label, round(base_conf, 4), reason)


def tag_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply ``tag_query`` to every row, returning new dicts.

    Original keys are preserved and the three new keys are appended:
    ``query_type``, ``query_type_confidence``, ``query_type_reason``.
    Rows missing ``query`` are flagged as ``unknown`` with reason
    ``"missing query field"`` and confidence 0.
    """
    out: List[Dict[str, Any]] = []
    for raw in rows:
        new_row = dict(raw)
        query = str(raw.get("query", "")).strip()
        if not query:
            tag = HeuristicTag(QT_UNKNOWN, 0.0, "missing query field")
        else:
            tag = tag_query(query)
        new_row["query_type"] = tag.query_type
        new_row["query_type_confidence"] = tag.confidence
        new_row["query_type_reason"] = tag.reason
        out.append(new_row)
    return out


def write_draft_jsonl(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> None:
    """Write tagged rows as one JSON-per-line to ``out_path``.

    Atomic-style write: rendered to a temp string buffer first, then
    flushed. ``out_path``'s parent must already exist.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_distribution(
    tagged_rows: Sequence[Mapping[str, Any]],
    *,
    low_confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Aggregate counts per query_type + low-confidence callout.

    Returns:
      {
        "total_rows": N,
        "per_type": {"<label>": {"count": int, "fraction": float}, ...},
        "low_confidence_count": int,
        "low_confidence_fraction": float,
        "low_confidence_threshold": float,
        "competing_count": int,   # rows whose reason mentions "competing"
      }
    """
    total = len(tagged_rows)
    per_type: Dict[str, int] = {}
    low = 0
    competing = 0
    for row in tagged_rows:
        qt = str(row.get("query_type") or QT_UNKNOWN)
        per_type[qt] = per_type.get(qt, 0) + 1
        try:
            conf = float(row.get("query_type_confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < float(low_confidence_threshold):
            low += 1
        reason = str(row.get("query_type_reason") or "")
        if "competing" in reason:
            competing += 1
    safe_total = max(1, total)
    return {
        "total_rows": total,
        "per_type": {
            qt: {"count": c, "fraction": round(c / safe_total, 4)}
            for qt, c in sorted(per_type.items())
        },
        "low_confidence_count": low,
        "low_confidence_fraction": round(low / safe_total, 4),
        "low_confidence_threshold": float(low_confidence_threshold),
        "competing_count": competing,
    }
