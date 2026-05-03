"""Balanced gold-seed candidate sampler for v4 silver RAG queries.

This module is intentionally offline and artifact-only. It validates a
silver JSONL set, filters unstable rows, selects a balanced review seed, and
renders the human-labeling JSONL/CSV/report bundle. It does not run retrieval,
indexing, tuning, or modify the silver generator.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "expected_title",
    "expected_section_path",
    "expected_chunk_ids",
    "query_type",
    "title_mention_level",
    "entity_mention_level",
    "difficulty",
    "answerability",
    "source_evidence",
    "generation_note",
)

OUTPUT_FIELDS: tuple[str, ...] = (
    "seed_id",
    "source_query_id",
    "query",
    "expected_doc_id",
    "expected_doc_ids",
    "expected_title",
    "expected_section_path",
    "expected_chunk_ids",
    "query_type",
    "title_mention_level",
    "entity_mention_level",
    "difficulty",
    "answerability",
    "source_evidence",
    "generation_note",
    "human_label_status",
    "human_expected_doc_ids",
    "human_expected_chunk_ids",
    "human_answerability",
    "human_difficulty",
    "human_notes",
    "reject_reason",
)

DEFAULT_TARGET_QUERY_TYPE_COUNTS: dict[str, int] = {
    "title_direct": 5,
    "title_partial": 6,
    "alias": 4,
    "character_question": 8,
    "setting_question": 8,
    "plot_memory": 8,
    "theme_question": 4,
    "vague_recall": 4,
    "risk_probe": 3,
}

RISK_QUERY_TYPES: tuple[str, ...] = (
    "wrong_assumption",
    "ambiguous",
    "unanswerable",
)

DEFAULT_DISTRIBUTION_TARGETS: dict[str, Any] = {
    "difficulty": {"easy_max": 10, "medium_target": 27, "hard_target": 23},
    "title_mention_level": {
        "exact_title_max": 10,
        "partial_title_target": 12,
        "alias_target": 5,
        "none_min": 25,
    },
    "answerability": {
        "answerable_target": 47,
        "partially_answerable_target": 3,
        "risk_probe_target": 3,
    },
}

UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "uncertainty",
    "weak evidence",
    "guessed",
    "ambiguous evidence",
    "low confidence",
    "불확실",
    "약한 근거",
    "추정",
)

PROBLEM_LIKE_PATTERNS: tuple[str, ...] = (
    "무엇인가",
    "서술하시오",
    "고르시오",
    "다음 중",
    "정답을",
    "작품 X",
)

TITLE_COPY_TYPES: tuple[str, ...] = ("title_direct", "title_partial", "alias")
ANIME_EVIDENCE_MARKERS: tuple[str, ...] = (
    "애니",
    "애니메이션",
    "TVA",
    "극장판",
    "만화",
    "라이트 노벨",
    "원작",
    "감독",
    "방영",
    "제작사",
    "성우",
    "시리즈",
)
NON_WORK_TITLE_MARKERS: tuple[str, ...] = (
    "사태",
    "논란",
    "사건",
    "올림픽",
    "월드컵",
    "대회",
)
NON_SETTING_TARGET_MARKERS: tuple[str, ...] = (
    "주제가",
    "음악",
    "OST",
    "평가",
    "줄거리",
    "회차",
    "에피소드",
)

EVIDENCE_LINKED_TYPES: tuple[str, ...] = (
    "character_question",
    "setting_question",
    "plot_memory",
    "theme_question",
    "comparison_like",
    "vague_recall",
)

TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        "애니",
        "애니메이션",
        "문서",
        "찾아줘",
        "찾고",
        "싶어",
        "나오는",
        "작품",
        "정리",
        "어디",
        "어느",
        "뭐였지",
        "뭐였더라",
        "이런",
        "내용",
        "설명",
        "설정",
        "성우진",
        "같이",
        "기본",
        "정보",
        "확인",
        "방영",
        "시기",
        "감독",
        "원작",
        "분위기",
        "주제",
        "얘기",
        "맞아",
    }
)

QUERY_TYPE_ORDER: tuple[str, ...] = (
    "title_direct",
    "title_partial",
    "alias",
    "character_question",
    "setting_question",
    "plot_memory",
    "theme_question",
    "vague_recall",
    "wrong_assumption",
    "ambiguous",
    "unanswerable",
)


@dataclass(frozen=True)
class CandidateIssue:
    query_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SelectionPolicy:
    target_count: int = 50
    seed: int = 42
    query_type_targets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TARGET_QUERY_TYPE_COUNTS)
    )
    exact_title_cap: int = 10
    none_title_min: int = 25
    alias_soft_target: int = 5
    partial_title_soft_target: int = 12
    medium_soft_target: int = 27
    hard_soft_target: int = 23


@dataclass(frozen=True)
class SamplingResult:
    input_path: str
    total_silver_count: int
    schema_valid_rows: list[dict[str, Any]]
    invalid_issues: list[CandidateIssue]
    eligible_rows: list[dict[str, Any]]
    rejected_issues: list[CandidateIssue]
    selected_source_rows: list[dict[str, Any]]
    selected_gold_rows: list[dict[str, Any]]
    policy: SelectionPolicy


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", _clean_text(value).lower())


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def expected_doc_ids_for_row(row: Mapping[str, Any]) -> list[str]:
    ids = [str(x) for x in _as_list(row.get("expected_doc_ids")) if str(x)]
    single = str(row.get("expected_doc_id") or "")
    if single and single not in ids:
        ids.insert(0, single)
    return ids


def expected_doc_id_for_row(row: Mapping[str, Any]) -> str:
    ids = expected_doc_ids_for_row(row)
    return ids[0] if ids else ""


def contains_expected_title_copy(row: Mapping[str, Any]) -> bool:
    title = _norm(row.get("expected_title"))
    query = _norm(row.get("query"))
    return bool(title and len(title) >= 4 and title in query)


def validate_silver_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[CandidateIssue]]:
    valid: list[dict[str, Any]] = []
    issues: list[CandidateIssue] = []
    seen: set[str] = set()
    for idx, raw in enumerate(rows, start=1):
        row = dict(raw)
        qid = _clean_text(row.get("query_id")) or f"row_{idx}"
        row_issues: list[str] = []
        for field_name in REQUIRED_FIELDS:
            if field_name not in row:
                row_issues.append(f"missing:{field_name}")
        if "expected_doc_id" not in row and "expected_doc_ids" not in row:
            row_issues.append("missing:expected_doc_id_or_expected_doc_ids")
        if qid in seen:
            row_issues.append("duplicate_query_id")
        seen.add(qid)
        if not _clean_text(row.get("query")):
            row_issues.append("empty_query")
        if not _clean_text(row.get("source_evidence")):
            row_issues.append("empty_source_evidence")
        if row.get("answerability") == "answerable" and not expected_doc_id_for_row(row):
            row_issues.append("answerable_without_expected_doc_id")
        if row_issues:
            issues.append(CandidateIssue(qid, "schema_invalid", ",".join(row_issues)))
            continue
        valid.append(row)
    return valid, issues


def _tokens(text: Any) -> set[str]:
    out: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9+.-]{1,}|[가-힣]{2,}", _clean_text(text)):
        token = match.group(0).lower()
        if token in TOKEN_STOPWORDS:
            continue
        out.add(token)
    return out


def evidence_overlap_count(row: Mapping[str, Any]) -> int:
    return len(_tokens(row.get("query")) & _tokens(row.get("source_evidence")))


def rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    query = _clean_text(row.get("query"))
    evidence = _clean_text(row.get("source_evidence"))
    note = _clean_text(row.get("generation_note")).lower()
    qtype = str(row.get("query_type") or "")
    answerability = str(row.get("answerability") or "")

    if answerability == "unanswerable" and expected_doc_id_for_row(row):
        reasons.append("unanswerable_has_single_expected_doc")
    if len(query) < 12:
        reasons.append("query_too_short")
    if len(query) > 180:
        reasons.append("query_too_long_for_human_seed")
    if len(evidence) < 25:
        reasons.append("source_evidence_too_short")
    if any(marker in note for marker in UNCERTAINTY_MARKERS):
        reasons.append("generation_note_uncertain")
    if any(pattern in query for pattern in PROBLEM_LIKE_PATTERNS):
        reasons.append("problem_like_query_surface")
    if re.search(
        r"(년대|대한민국|이러한|관련 단서|주요 캐릭터|등장인물 및 주요 성우진) "
        r"(얘기가 나오는 평가 문서|관련 캐릭터 정리)",
        query,
    ):
        reasons.append("weak_manual_query_surface")
    if "<" in query or ">" in query or "이 작품라고" in query:
        reasons.append("artifact_like_query_surface")
    if qtype in TITLE_COPY_TYPES and re.match(r"^[가-힣A-Za-z!?.]{1,3}\s+극장판\b", query):
        reasons.append("short_theatrical_title_surface")
    if qtype in TITLE_COPY_TYPES and any(
        marker in str(row.get("expected_title") or "")
        for marker in NON_WORK_TITLE_MARKERS
    ):
        reasons.append("non_work_title_for_title_surface")
    if qtype == "setting_question" and "랑" in query:
        reasons.append("compound_setting_surface_needs_review")
    if qtype == "setting_question":
        setting_target_text = " ".join(
            [
                str(row.get("expected_title") or ""),
                " ".join(str(x) for x in _as_list(row.get("expected_section_path"))),
            ]
        )
        if any(marker in setting_target_text for marker in NON_SETTING_TARGET_MARKERS):
            reasons.append("setting_query_targets_non_setting_section")
    if qtype == "character_question" and re.search(
        r"(은하고|는하고|로하고|종족은|바보짓|성격 성우진|겉도는|한마디로|▶)",
        query,
    ):
        reasons.append("character_surface_not_person_like")
    if "애니" in query:
        title_and_evidence = f"{row.get('expected_title') or ''} {evidence}"
        if not any(marker in title_and_evidence for marker in ANIME_EVIDENCE_MARKERS):
            reasons.append("anime_query_without_anime_evidence")
    if qtype == "ambiguous" and len(_tokens(query)) < 3:
        reasons.append("ambiguous_query_too_low_information")
    if qtype in EVIDENCE_LINKED_TYPES and evidence_overlap_count(row) == 0:
        reasons.append("weak_source_evidence_link")
    return reasons


def split_eligible_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[CandidateIssue]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[CandidateIssue] = []
    for raw in rows:
        row = dict(raw)
        qid = str(row.get("query_id") or "")
        reasons = rejection_reasons(row)
        if reasons:
            rejected.append(CandidateIssue(qid, "policy_rejected", ",".join(reasons)))
            continue
        eligible.append(row)
    return eligible, rejected


def distribution(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows))


def selected_query_type_distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return distribution(rows, "query_type")


def _query_type_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return QUERY_TYPE_ORDER.index(qtype), qtype
    except ValueError:
        return len(QUERY_TYPE_ORDER), qtype


def _quality_score(row: Mapping[str, Any]) -> float:
    score = 100.0
    query = _clean_text(row.get("query"))
    evidence = _clean_text(row.get("source_evidence"))
    qlen = len(query)
    elen = len(evidence)
    qtype = str(row.get("query_type") or "")

    if 24 <= qlen <= 125:
        score += 8.0
    elif qlen < 18:
        score -= 18.0
    elif qlen > 150:
        score -= 12.0

    if 45 <= elen <= 220:
        score += 6.0
    elif elen < 35:
        score -= 15.0

    overlap = evidence_overlap_count(row)
    if qtype in EVIDENCE_LINKED_TYPES:
        score += min(overlap, 5) * 2.0
    elif overlap:
        score += min(overlap, 3) * 0.8

    if contains_expected_title_copy(row):
        score -= 18.0
    if qtype in {"wrong_assumption", "ambiguous", "unanswerable"}:
        score -= 4.0
    if "뭐였" in query or "찾고 있어" in query or "찾아줘" in query:
        score += 2.0
    return score


def _balance_score(
    row: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    policy: SelectionPolicy,
) -> float:
    score = _quality_score(row)
    title_counts = Counter(str(x.get("title_mention_level") or "") for x in selected)
    difficulty_counts = Counter(str(x.get("difficulty") or "") for x in selected)
    answer_counts = Counter(str(x.get("answerability") or "") for x in selected)
    selected_docs = {expected_doc_id_for_row(x) for x in selected}

    level = str(row.get("title_mention_level") or "")
    if level == "exact_title":
        if title_counts["exact_title"] >= policy.exact_title_cap:
            return -1_000_000.0
        score -= 10.0
    elif level == "none":
        if title_counts["none"] < policy.none_title_min:
            score += 12.0
    elif level == "alias":
        if title_counts["alias"] >= policy.alias_soft_target:
            score -= 8.0
        else:
            score += 2.0
    elif level == "partial_title":
        if title_counts["partial_title"] >= policy.partial_title_soft_target:
            score -= 6.0

    difficulty = str(row.get("difficulty") or "")
    if difficulty == "medium":
        if difficulty_counts["medium"] < policy.medium_soft_target:
            score += 6.0
        else:
            score -= 5.0
    elif difficulty == "hard":
        if difficulty_counts["hard"] < policy.hard_soft_target:
            score += 2.0
        else:
            score -= 6.0
    elif difficulty == "easy" and difficulty_counts["easy"] >= 10:
        score -= 8.0

    answerability = str(row.get("answerability") or "")
    if answerability == "partially_answerable":
        if answer_counts["partially_answerable"] < 3:
            score += 6.0
        elif answer_counts["partially_answerable"] >= 4:
            score -= 15.0

    doc_id = expected_doc_id_for_row(row)
    if doc_id and doc_id in selected_docs:
        score -= 16.0

    return score


def _risk_group_split_target(eligible: Sequence[Mapping[str, Any]], target: int) -> dict[str, int]:
    available = Counter(str(row.get("query_type") or "") for row in eligible)
    out = {qtype: 0 for qtype in RISK_QUERY_TYPES}
    preferred = ("wrong_assumption", "ambiguous", "unanswerable")
    remaining = target
    for qtype in preferred:
        if remaining <= 0:
            break
        if available[qtype] <= 0:
            continue
        take = 1
        if qtype == "wrong_assumption" and target >= 3 and available[qtype] >= 2:
            take = 2
        take = min(take, available[qtype], remaining)
        out[qtype] += take
        remaining -= take
    if remaining > 0:
        for qtype in preferred:
            if remaining <= 0:
                break
            extra = min(available[qtype] - out[qtype], remaining)
            if extra > 0:
                out[qtype] += extra
                remaining -= extra
    return {k: v for k, v in out.items() if v > 0}


def expanded_query_type_targets(
    eligible: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
) -> dict[str, int]:
    expanded: dict[str, int] = {}
    for qtype, count in targets.items():
        if qtype == "risk_probe":
            expanded.update(_risk_group_split_target(eligible, count))
        else:
            expanded[qtype] = count
    return expanded


def select_rows(
    eligible: Sequence[Mapping[str, Any]],
    policy: SelectionPolicy,
) -> list[dict[str, Any]]:
    rng = random.Random(policy.seed)
    random_tiebreak = {
        str(row.get("query_id") or idx): rng.random()
        for idx, row in enumerate(eligible)
    }
    targets = expanded_query_type_targets(eligible, policy.query_type_targets)
    if sum(targets.values()) != policy.target_count:
        raise ValueError(
            f"query_type targets sum to {sum(targets.values())}, "
            f"expected {policy.target_count}"
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_type[str(row.get("query_type") or "")].append(dict(row))

    selected: list[dict[str, Any]] = []
    selected_query_ids: set[str] = set()
    for qtype in sorted(targets, key=_query_type_sort_key):
        need = targets[qtype]
        pool = [row for row in by_type.get(qtype, []) if row.get("query_id") not in selected_query_ids]
        for _ in range(need):
            if not pool:
                raise ValueError(f"not enough eligible candidates for {qtype}: need {need}")
            ranked = sorted(
                pool,
                key=lambda row: (
                    _balance_score(row, selected, policy),
                    random_tiebreak.get(str(row.get("query_id") or ""), 0.0),
                    str(row.get("query_id") or ""),
                ),
                reverse=True,
            )
            choice = ranked[0]
            if _balance_score(choice, selected, policy) < -999_999:
                raise ValueError(f"exact_title cap prevents selecting enough {qtype} rows")
            selected.append(dict(choice))
            selected_query_ids.add(str(choice.get("query_id") or ""))
            pool = [row for row in pool if row.get("query_id") != choice.get("query_id")]

    return sorted(
        selected,
        key=lambda row: (
            _query_type_sort_key(str(row.get("query_type") or "")),
            str(row.get("difficulty") or ""),
            str(row.get("query_id") or ""),
        ),
    )


def make_gold_seed_row(row: Mapping[str, Any], seed_index: int) -> dict[str, Any]:
    expected_doc_ids = expected_doc_ids_for_row(row)
    expected_doc_id = expected_doc_ids[0] if expected_doc_ids else ""
    return {
        "seed_id": f"gold_seed_{seed_index:04d}",
        "source_query_id": str(row.get("query_id") or ""),
        "query": _clean_text(row.get("query")),
        "expected_doc_id": expected_doc_id,
        "expected_doc_ids": expected_doc_ids,
        "expected_title": _clean_text(row.get("expected_title")),
        "expected_section_path": _as_list(row.get("expected_section_path")),
        "expected_chunk_ids": [str(x) for x in _as_list(row.get("expected_chunk_ids"))],
        "query_type": str(row.get("query_type") or ""),
        "title_mention_level": str(row.get("title_mention_level") or ""),
        "entity_mention_level": str(row.get("entity_mention_level") or ""),
        "difficulty": str(row.get("difficulty") or ""),
        "answerability": str(row.get("answerability") or ""),
        "source_evidence": _clean_text(row.get("source_evidence")),
        "generation_note": _clean_text(row.get("generation_note")),
        "human_label_status": "pending",
        "human_expected_doc_ids": [],
        "human_expected_chunk_ids": [],
        "human_answerability": "",
        "human_difficulty": "",
        "human_notes": "",
        "reject_reason": "",
    }


def select_gold_seed_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_path: str = "",
    target_count: int = 50,
    seed: int = 42,
    query_type_targets: Mapping[str, int] | None = None,
) -> SamplingResult:
    policy = SelectionPolicy(
        target_count=target_count,
        seed=seed,
        query_type_targets=dict(query_type_targets or DEFAULT_TARGET_QUERY_TYPE_COUNTS),
    )
    schema_valid, invalid = validate_silver_rows(rows)
    eligible, rejected = split_eligible_rows(schema_valid)
    selected = select_rows(eligible, policy)
    gold_rows = [make_gold_seed_row(row, idx) for idx, row in enumerate(selected, start=1)]
    return SamplingResult(
        input_path=input_path,
        total_silver_count=len(rows),
        schema_valid_rows=schema_valid,
        invalid_issues=invalid,
        eligible_rows=eligible,
        rejected_issues=rejected,
        selected_source_rows=selected,
        selected_gold_rows=gold_rows,
        policy=policy,
    )


def issue_distribution(issues: Iterable[CandidateIssue]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for issue in issues:
        details = [x for x in issue.detail.split(",") if x] or [issue.reason]
        for detail in details:
            counts[detail] += 1
    return dict(counts)


def _json_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_cell(row.get(field, "")) for field in OUTPUT_FIELDS})


def manifest_dict(
    result: SamplingResult,
    *,
    output_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected = result.selected_gold_rows
    invalid_rejected = len(result.invalid_issues) + len(result.rejected_issues)
    return {
        "input_file_path": result.input_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_silver_count": result.total_silver_count,
        "schema_valid_count": len(result.schema_valid_rows),
        "valid_candidate_count": len(result.eligible_rows),
        "invalid_rejected_candidate_count": invalid_rejected,
        "schema_invalid_count": len(result.invalid_issues),
        "policy_rejected_count": len(result.rejected_issues),
        "selected_count": len(selected),
        "query_type_distribution": distribution(selected, "query_type"),
        "difficulty_distribution": distribution(selected, "difficulty"),
        "title_mention_level_distribution": distribution(selected, "title_mention_level"),
        "answerability_distribution": distribution(selected, "answerability"),
        "random_seed": result.policy.seed,
        "target_query_type_distribution": dict(result.policy.query_type_targets),
        "expanded_query_type_distribution": expanded_query_type_targets(
            result.eligible_rows, result.policy.query_type_targets
        ),
        "target_distribution_notes": DEFAULT_DISTRIBUTION_TARGETS,
        "invalid_rejected_reason_counts": issue_distribution(
            [*result.invalid_issues, *result.rejected_issues]
        ),
        "selection_policy_summary": (
            "Schema-valid rows are filtered for answerability/doc-id conflicts, "
            "empty or weak evidence, uncertainty notes, very short/long queries, "
            "and problem-like wording. Query_type quotas are selected first. "
            "Within each quota, deterministic seeded ranking prefers stronger "
            "query/evidence links, distinct docs, title_mention_level none, "
            "medium/hard balance, and enforces exact_title <= 10."
        ),
        "outputs": dict(output_paths or {}),
    }


def risk_review_rows(rows: Sequence[Mapping[str, Any]], *, limit: int = 5) -> list[dict[str, str]]:
    scored: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        reasons: list[str] = []
        risk = 0
        qtype = str(row.get("query_type") or "")
        if qtype in {"ambiguous", "wrong_assumption", "unanswerable"}:
            risk += 5
            reasons.append(f"risk query_type={qtype}")
        if row.get("answerability") != "answerable":
            risk += 4
            reasons.append(f"answerability={row.get('answerability')}")
        if row.get("title_mention_level") == "exact_title":
            risk += 3
            reasons.append("exact title mention")
        overlap = evidence_overlap_count(row)
        if overlap <= 1:
            risk += 2
            reasons.append(f"low query/evidence token overlap={overlap}")
        if qtype == "setting_question" and "단체 설정" in _clean_text(row.get("query")):
            risk += 3
            reasons.append("generic setting surface")
        if qtype == "theme_question" and overlap <= 1:
            risk += 3
            reasons.append("theme query has weak surface/evidence link")
        if len(_clean_text(row.get("query"))) > 135:
            risk += 2
            reasons.append("long query surface")
        if qtype in {"theme_question", "vague_recall", "comparison_like"}:
            risk += 1
            reasons.append("semantic-memory style query")
        if risk <= 0:
            continue
        scored.append(
            (
                risk,
                {
                    "seed_id": str(row.get("seed_id") or ""),
                    "source_query_id": str(row.get("source_query_id") or row.get("query_id") or ""),
                    "query": _clean_text(row.get("query")),
                    "expected_title": _clean_text(row.get("expected_title")),
                    "reason": "; ".join(reasons),
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]["seed_id"]))
    return [item[1] for item in scored[:limit]]


def render_sampling_report(result: SamplingResult, manifest: Mapping[str, Any]) -> str:
    selected = result.selected_gold_rows
    lines: list[str] = []
    lines.append("# Gold Seed 50 Sampling Report")
    lines.append("")
    lines.append("## 전체 요약")
    lines.append(f"- input: `{result.input_path}`")
    lines.append(f"- silver rows: {result.total_silver_count}")
    lines.append(f"- schema valid rows: {len(result.schema_valid_rows)}")
    lines.append(f"- eligible candidates after policy filter: {len(result.eligible_rows)}")
    lines.append(f"- invalid/rejected rows: {len(result.invalid_issues) + len(result.rejected_issues)}")
    lines.append(f"- selected rows: {len(selected)}")
    lines.append(f"- random seed: {result.policy.seed}")
    lines.append("")

    lines.append("## 목표 분포 vs 실제 분포")
    lines.append("")
    lines.append("### query_type")
    expanded = expanded_query_type_targets(result.eligible_rows, result.policy.query_type_targets)
    actual_qtype = distribution(selected, "query_type")
    for qtype in sorted(set(expanded) | set(actual_qtype), key=_query_type_sort_key):
        lines.append(f"- {qtype}: target {expanded.get(qtype, 0)} / actual {actual_qtype.get(qtype, 0)}")
    lines.append("")
    lines.append("### difficulty")
    actual_difficulty = distribution(selected, "difficulty")
    lines.append("- target: easy <= 10, medium around 27, hard around 23")
    for key in ("easy", "medium", "hard"):
        lines.append(f"- {key}: actual {actual_difficulty.get(key, 0)}")
    lines.append("")
    lines.append("### title_mention_level")
    actual_title = distribution(selected, "title_mention_level")
    lines.append("- target: exact_title <= 10, partial_title around 12, alias around 5, none >= 25")
    for key in ("exact_title", "partial_title", "alias", "none"):
        lines.append(f"- {key}: actual {actual_title.get(key, 0)}")
    lines.append("")
    lines.append("### answerability")
    actual_answer = distribution(selected, "answerability")
    lines.append("- target: answerable 47, partially_answerable 3, risk probes 3")
    for key in ("answerable", "partially_answerable", "unanswerable"):
        lines.append(f"- {key}: actual {actual_answer.get(key, 0)}")
    lines.append("")

    lines.append("## 제외된 항목 주요 사유")
    reason_counts = issue_distribution([*result.invalid_issues, *result.rejected_issues])
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Selected 50")
    for row in selected:
        lines.append("")
        lines.append(f"### {row['seed_id']} / {row['source_query_id']}")
        lines.append(f"- query: {row['query']}")
        lines.append(f"- expected_title: {row['expected_title']}")
        lines.append(f"- query_type: {row['query_type']}")
        lines.append(f"- difficulty: {row['difficulty']}")
        lines.append(f"- answerability: {row['answerability']}")
        lines.append(f"- source_evidence: {row['source_evidence']}")
    lines.append("")

    lines.append("## 사람이 직접 봐야 할 위험 샘플")
    risks = risk_review_rows(selected, limit=5)
    if risks:
        for row in risks:
            lines.append(
                f"- {row['seed_id']} / {row['source_query_id']}: "
                f"{row['expected_title']} | {row['reason']} | {row['query']}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## 검수 체크리스트")
    checklist = [
        "query가 실제 사용자가 입력할 법한 자연스러운 문장인가?",
        "query가 너무 정답 제목을 노출하고 있지는 않은가?",
        "expected_doc_id / expected_title이 query의 의도와 맞는가?",
        "source_evidence만 보고도 정답 근거가 납득되는가?",
        "다른 작품도 정답이 될 수 있을 정도로 모호하지 않은가?",
        "answerability 라벨이 적절한가?",
        "difficulty 라벨이 적절한가?",
        "gold로 승격할지, 수정 후 승격할지, reject할지 판단한다.",
    ]
    for item in checklist:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Output Paths")
    for name, path in (manifest.get("outputs") or {}).items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_sampling_bundle(result: SamplingResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": out_dir / "gold_seed_50_candidates.jsonl",
        "csv": out_dir / "gold_seed_50_candidates.csv",
        "manifest": out_dir / "gold_seed_50_sampling_manifest.json",
        "report": out_dir / "gold_seed_50_sampling_report.md",
    }
    write_jsonl(result.selected_gold_rows, paths["jsonl"])
    write_csv(result.selected_gold_rows, paths["csv"])
    manifest = manifest_dict(result, output_paths={k: str(v) for k, v in paths.items()})
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        render_sampling_report(result, manifest),
        encoding="utf-8",
    )
    return paths
