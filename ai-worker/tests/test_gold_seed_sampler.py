from __future__ import annotations

from eval.harness.gold_seed_sampler import (
    OUTPUT_FIELDS,
    distribution,
    expanded_query_type_targets,
    make_gold_seed_row,
    select_gold_seed_candidates,
    split_eligible_rows,
    validate_silver_rows,
)
from scripts import select_gold_seed_50_from_silver as gold_seed_cli


def _row(
    qid: str,
    *,
    query_type: str = "title_direct",
    title_mention_level: str = "partial_title",
    difficulty: str = "medium",
    answerability: str = "answerable",
    expected_doc_id: str | None = None,
    expected_title: str | None = None,
    query: str | None = None,
    source_evidence: str | None = None,
    generation_note: str = "stable test fixture",
) -> dict:
    title = expected_title or f"테스트 작품 {qid}"
    q = query or f"{title.split()[0]} 애니 기본 정보 찾아줘"
    evidence = (
        source_evidence
        if source_evidence is not None
        else f"{title} 애니메이션 관련 근거 문장으로 테스트 후보를 검증한다."
    )
    row = {
        "query_id": qid,
        "query": q,
        "expected_title": title,
        "expected_section_path": ["개요"],
        "expected_chunk_ids": [f"chunk-{qid}"],
        "query_type": query_type,
        "title_mention_level": title_mention_level,
        "entity_mention_level": "explicit",
        "difficulty": difficulty,
        "answerability": answerability,
        "source_evidence": evidence,
        "generation_note": generation_note,
    }
    if expected_doc_id is not None:
        row["expected_doc_id"] = expected_doc_id
    else:
        row["expected_doc_id"] = f"doc-{qid}"
    return row


def test_schema_validation_collects_missing_doc_and_evidence() -> None:
    rows = [
        _row("ok"),
        _row("bad-doc", expected_doc_id=""),
        _row("bad-evidence", source_evidence=""),
    ]

    valid, issues = validate_silver_rows(rows)

    assert [row["query_id"] for row in valid] == ["ok"]
    details = " ".join(issue.detail for issue in issues)
    assert "answerable_without_expected_doc_id" in details
    assert "empty_source_evidence" in details


def test_invalid_rows_are_excluded_before_selection() -> None:
    rows = [
        _row("ok-1", query_type="alias", title_mention_level="alias"),
        _row("ok-2", query_type="alias", title_mention_level="alias"),
        _row("bad", query_type="alias", title_mention_level="alias", expected_doc_id=""),
    ]
    valid, invalid = validate_silver_rows(rows)
    eligible, rejected = split_eligible_rows(valid)

    assert len(invalid) == 1
    assert not rejected
    assert [row["query_id"] for row in eligible] == ["ok-1", "ok-2"]


def test_target_distribution_calculation_expands_risk_probe() -> None:
    rows = [
        _row("w1", query_type="wrong_assumption", answerability="partially_answerable", difficulty="hard"),
        _row("w2", query_type="wrong_assumption", answerability="partially_answerable", difficulty="hard"),
        _row("a1", query_type="ambiguous", answerability="partially_answerable", difficulty="hard"),
    ]

    targets = expanded_query_type_targets(rows, {"title_direct": 2, "risk_probe": 3})

    assert targets == {"title_direct": 2, "wrong_assumption": 2, "ambiguous": 1}


def test_exact_title_cap_is_enforced() -> None:
    rows = []
    for idx in range(14):
        title = f"정확한제목{idx}"
        rows.append(
            _row(
                f"exact-{idx}",
                query_type="title_direct",
                title_mention_level="exact_title",
                expected_title=title,
                query=f"{title} 애니 기본 정보 찾아줘",
            )
        )
    for idx in range(4):
        rows.append(
            _row(
                f"partial-{idx}",
                query_type="title_direct",
                title_mention_level="partial_title",
                query=f"부분제목{idx} 애니 기본 정보 찾아줘",
            )
        )

    result = select_gold_seed_candidates(
        rows,
        target_count=12,
        seed=7,
        query_type_targets={"title_direct": 12},
    )

    assert len(result.selected_gold_rows) == 12
    assert distribution(result.selected_gold_rows, "title_mention_level")["exact_title"] <= 10


def test_sampling_is_deterministic_by_seed() -> None:
    rows = [
        _row(
            f"alias-{idx}",
            query_type="alias",
            title_mention_level="alias",
            query=f"별칭{idx}, 이 이름으로 찾으면 어느 애니 문서가 맞아?",
        )
        for idx in range(12)
    ]

    a = select_gold_seed_candidates(
        rows,
        target_count=5,
        seed=42,
        query_type_targets={"alias": 5},
    )
    b = select_gold_seed_candidates(
        rows,
        target_count=5,
        seed=42,
        query_type_targets={"alias": 5},
    )

    assert [row["source_query_id"] for row in a.selected_gold_rows] == [
        row["source_query_id"] for row in b.selected_gold_rows
    ]


def test_output_schema_preserves_source_and_human_review_fields() -> None:
    source = _row(
        "q1",
        expected_doc_id="doc-1",
        query_type="plot_memory",
        title_mention_level="none",
        difficulty="hard",
        query="주인공이 과거 기억을 떠올리는 장면 나오는 애니 뭐였더라",
        source_evidence="주인공이 과거 기억을 떠올리는 장면이 후반부 전개에서 중요하게 다뤄진다.",
    )

    out = make_gold_seed_row(source, 1)

    assert tuple(out.keys()) == OUTPUT_FIELDS
    assert out["seed_id"] == "gold_seed_0001"
    assert out["source_query_id"] == "q1"
    assert out["expected_doc_id"] == "doc-1"
    assert out["expected_doc_ids"] == ["doc-1"]
    assert out["human_label_status"] == "pending"
    assert out["human_expected_doc_ids"] == []
    assert out["reject_reason"] == ""


def test_gold_seed_cli_defaults_use_manual_curated_silver() -> None:
    assert "7.12_silver_manual_curated" in str(gold_seed_cli.DEFAULT_SILVER_PATH)
    assert "queries_v4_silver_manual_curated_500.jsonl" in str(
        gold_seed_cli.DEFAULT_SILVER_PATH
    )
    assert "gold_seed_50_manual_curated" in str(gold_seed_cli.DEFAULT_OUT_DIR)
