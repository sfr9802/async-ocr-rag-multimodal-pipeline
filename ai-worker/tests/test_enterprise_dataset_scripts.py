"""Tests for Phase 9b enterprise dataset synthesis scripts.

These exercise the pure-Python pieces of the pipeline:

  * prompt loading / rendering
  * response validation + index row shaping
  * diversity-guard cosine check (with an injected stub embedder)
  * type assignment + unanswerable target math
  * TF-IDF vocabulary building
  * difficulty assignment rules
  * dedup / per-doc cap / stratification
  * ResumableJsonlWriter.drop (the helper added for the diversity guard)

They do NOT call the Claude API or download the bge-m3 model — the
retrieval-behaviour tests inject a tiny deterministic embedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dataset._common import ResumableJsonlWriter
from scripts.dataset.build_enterprise_corpus import (
    _DuplicatePair,
    _load_prompt,
    _render_prompt,
    _to_index_row,
    _validate_doc,
    find_duplicate_pairs,
    resolve_generator,
)
from scripts.dataset.generate_enterprise_queries import (
    ALL_TYPES,
    ANSWERABLE_TYPES,
    _assign_doc_types,
    build_tfidf,
    parse_type_ratio,
    unanswerable_total,
)
from scripts.dataset.validate_enterprise_dataset import (
    DIFFICULTIES,
    DifficultyThresholds,
    _apply_per_doc_cap,
    _assign_difficulty,
    _dedup_rows,
    _parse_target,
    _RetrievalHit,
    _stratify,
)


# ---------------------------------------------------------------------------
# Prompt loading + rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["hr", "finance", "it", "product", "legal"])
def test_prompt_files_load_and_render(category: str) -> None:
    template = _load_prompt(category)
    assert "{doc_id}" in template
    assert "{seed}" in template
    assert "{min_chars}" in template
    assert "{max_chars}" in template
    rendered = _render_prompt(
        template,
        doc_id="kr-hr-007",
        seed=12345,
        min_chars=400,
        max_chars=1500,
    )
    assert "{doc_id}" not in rendered
    assert "kr-hr-007" in rendered
    assert "12345" in rendered


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude:sonnet-4-6", "claude-sonnet-4-6"),
        ("claude:sonnet", "claude-sonnet-4-6"),
        ("sonnet", "claude-sonnet-4-6"),
        ("claude:haiku", "claude-haiku-4-5-20251001"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ],
)
def test_resolve_generator_known_aliases(raw: str, expected: str) -> None:
    provider, model = resolve_generator(raw)
    assert provider == "claude"
    assert model == expected


def test_resolve_generator_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        resolve_generator("openai:gpt-4o")


# ---------------------------------------------------------------------------
# Doc validation + index row shaping
# ---------------------------------------------------------------------------


def _good_doc_payload(doc_id: str = "kr-hr-001") -> dict:
    return {
        "doc_id": doc_id,
        "title": "연차 휴가 운영 지침",
        "sections": [
            {"heading": "목적", "text": "본 지침은 사내 연차 휴가의 운영 방식을 정의하며, 적용 대상은 모든 임직원입니다. 관리 주관은 인사팀이며, 연차는 근속 연수에 따라 연 15일에서 최대 25일까지 차등 부여됩니다. CHRO는 본 지침의 최종 결재권자입니다."},
            {"heading": "신청 절차", "text": "직원은 희망 일자 7일 전까지 사내 포털을 통해 팀장에게 신청하고, 승인 완료 시 인사팀 시스템에 자동 반영됩니다. 긴급 상황으로 사전 신청이 어려운 경우에는 익일까지 사후 결재가 허용되며, 팀장이 반려한 경우 본부장에게 재심 요청이 가능합니다."},
            {"heading": "예외 사항", "text": "병가는 본 지침과 별도의 규정을 따르며 의료기관의 진단서를 증빙으로 요구합니다. 연속 10일을 초과하는 장기 휴가는 CHRO 승인을 받아야 하며, 프로젝트 마일스톤 전 2주 이내 신청은 본부장 협의를 선행합니다."},
        ],
        "exception_clauses": ["병가는 본 지침 적용 대상에서 제외한다."],
        "related_docs": ["취업규칙"],
    }


def test_validate_doc_happy_path() -> None:
    clean = _validate_doc(_good_doc_payload(), expected_doc_id="kr-hr-001",
                          min_chars=400, max_chars=1500)
    assert clean["doc_id"] == "kr-hr-001"
    assert len(clean["sections"]) == 3
    assert clean["related_docs"] == ["취업규칙"]


def test_validate_doc_rejects_mismatched_id() -> None:
    from scripts.dataset._common import ClaudeResponseError

    with pytest.raises(ClaudeResponseError):
        _validate_doc(_good_doc_payload("kr-hr-999"), expected_doc_id="kr-hr-001",
                      min_chars=400, max_chars=1500)


def test_validate_doc_rejects_two_sections() -> None:
    from scripts.dataset._common import ClaudeResponseError

    payload = _good_doc_payload()
    payload["sections"] = payload["sections"][:2]
    with pytest.raises(ClaudeResponseError):
        _validate_doc(payload, expected_doc_id="kr-hr-001",
                      min_chars=400, max_chars=1500)


def test_to_index_row_disambiguates_duplicate_headings() -> None:
    doc = {
        "doc_id": "kr-hr-010",
        "title": "테스트",
        "sections": [
            {"heading": "개요", "text": "A"},
            {"heading": "개요", "text": "B"},
        ],
        "exception_clauses": [],
        "related_docs": [],
    }
    row = _to_index_row(
        doc, category="hr", seed=42,
        generated_ts="2026-04-23T00:00:00+00:00",
        source_label="synthesized-by-test",
    )
    assert list(row["sections"]) == ["개요", "개요#2"]
    assert row["section_order"] == ["개요", "개요#2"]
    assert row["domain"] == "enterprise"
    assert row["language"] == "ko"
    assert row["category"] == "hr"
    assert row["source"] == "synthesized-by-test"


# ---------------------------------------------------------------------------
# Diversity guard
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Returns pre-computed rows so the diversity test stays offline."""

    def __init__(self, vectors) -> None:
        import numpy as np

        self._vectors = np.asarray(vectors, dtype="float32")

    def embed_passages(self, texts):
        import numpy as np

        assert len(texts) == len(self._vectors), (
            f"Stub expected {len(self._vectors)} texts, got {len(texts)}"
        )
        return self._vectors


def test_find_duplicate_pairs_flags_younger_doc() -> None:
    docs = [
        {"doc_id": "kr-hr-001", "title": "A", "sections": [],
         "generated_ts": "2026-04-23T00:00:00+00:00"},
        {"doc_id": "kr-hr-002", "title": "B", "sections": [],
         "generated_ts": "2026-04-23T00:01:00+00:00"},
        {"doc_id": "kr-hr-003", "title": "C", "sections": [],
         "generated_ts": "2026-04-23T00:02:00+00:00"},
    ]
    # doc 1 and 2 very similar (cosine ≈ 1), doc 3 different
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [0.0, 0.0, 1.0],
    ]
    import numpy as np

    vectors = np.asarray(vectors, dtype="float32")
    # L2-normalize rows to mirror sentence-transformers behaviour
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = find_duplicate_pairs(
        docs, category="hr", threshold=0.88, embedder=_StubEmbedder(vectors),
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert {pair.doc_a, pair.doc_b} == {"kr-hr-001", "kr-hr-002"}
    assert pair.younger == "kr-hr-002"  # higher index -> younger


def test_find_duplicate_pairs_empty_when_single_doc() -> None:
    docs = [{"doc_id": "kr-hr-001", "title": "A", "sections": []}]
    import numpy as np

    pairs = find_duplicate_pairs(
        docs, category="hr", threshold=0.88,
        embedder=_StubEmbedder(np.zeros((1, 3), dtype="float32")),
    )
    assert pairs == []


# ---------------------------------------------------------------------------
# Type ratio + assignment
# ---------------------------------------------------------------------------


def test_parse_type_ratio_normalizes_to_one() -> None:
    r = parse_type_ratio("factoid:0.55,procedural:0.25,comparison:0.15,unanswerable:0.05")
    assert set(r) == set(ALL_TYPES)
    assert abs(sum(r.values()) - 1.0) < 1e-6


def test_parse_type_ratio_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        parse_type_ratio("factoid:1,jazz:0.5")


def test_assign_doc_types_matches_ratios() -> None:
    from collections import Counter

    ratio = parse_type_ratio("factoid:0.5,procedural:0.3,comparison:0.2,unanswerable:0.0")
    doc_ids = [f"doc-{i}" for i in range(100)]
    assigned = _assign_doc_types(doc_ids, ratio, queries_per_doc=3, seed=42)
    flat = [t for types in assigned.values() for t in types]
    counts = Counter(flat)
    # 300 slots total. Expect ~150 factoid, ~90 procedural, ~60 comparison.
    assert counts["factoid"] == pytest.approx(150, abs=2)
    assert counts["procedural"] == pytest.approx(90, abs=2)
    assert counts["comparison"] == pytest.approx(60, abs=2)
    assert all(len(v) == 3 for v in assigned.values())


def test_unanswerable_total_matches_task_spec() -> None:
    ratio = parse_type_ratio(
        "factoid:0.55,procedural:0.25,comparison:0.15,unanswerable:0.05"
    )
    # 125 docs × 3 queries = 375 answerable ≈ 95% => ~20 unanswerable
    assert unanswerable_total(ratio, doc_count=125, queries_per_doc=3) == pytest.approx(20, abs=2)


def test_unanswerable_total_zero_when_ratio_zero() -> None:
    ratio = parse_type_ratio("factoid:0.8,procedural:0.1,comparison:0.1")
    assert unanswerable_total(ratio, doc_count=10, queries_per_doc=3) == 0


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


def test_build_tfidf_bucket_docs_by_category_and_score_terms() -> None:
    corpus = [
        {
            "doc_id": "kr-hr-001",
            "title": "연차 운영",
            "category": "hr",
            "sections": {
                "s1": {"text": "연차 신청 절차는 팀장 승인 후 인사팀 제출입니다."},
                "s2": {"text": "연간 15일 기준으로 운영합니다."},
            },
        },
        {
            "doc_id": "kr-finance-001",
            "title": "법인카드",
            "category": "finance",
            "sections": {
                "s1": {"text": "법인카드 사용 한도는 500만원입니다. 재무팀 승인 필요."},
            },
        },
    ]
    vocab = build_tfidf(corpus)
    assert "hr" in vocab.docs_by_category
    assert "finance" in vocab.docs_by_category
    assert vocab.contains("연차")
    assert not vocab.contains("전혀없는단어입니다")


# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------


def _hits(*pairs):
    return [_RetrievalHit(doc_id=d, score=s, rank=i + 1)
            for i, (d, s) in enumerate(pairs)]


def test_assign_difficulty_easy_when_gold_is_rank_one_high_score() -> None:
    diff, meta = _assign_difficulty(
        question_type="factoid",
        expected_doc_ids=["doc-A"],
        hits=_hits(("doc-A", 0.82), ("doc-B", 0.40)),
        thresholds=DifficultyThresholds(),
    )
    assert diff == "easy"
    assert meta["gold_rank"] == 1


def test_assign_difficulty_medium_when_gold_is_rank_two() -> None:
    diff, _ = _assign_difficulty(
        question_type="factoid",
        expected_doc_ids=["doc-A"],
        hits=_hits(("doc-B", 0.80), ("doc-A", 0.65)),
        thresholds=DifficultyThresholds(),
    )
    assert diff == "medium"


def test_assign_difficulty_hard_when_gold_is_rank_eight() -> None:
    hits = [("doc-X", 0.82)]
    for i in range(6):
        hits.append((f"doc-filler-{i}", 0.70 - i * 0.01))
    hits.append(("doc-A", 0.35))
    diff, _ = _assign_difficulty(
        question_type="factoid",
        expected_doc_ids=["doc-A"],
        hits=_hits(*hits),
        thresholds=DifficultyThresholds(),
    )
    assert diff == "hard"


def test_assign_difficulty_drops_when_gold_missing() -> None:
    diff, meta = _assign_difficulty(
        question_type="factoid",
        expected_doc_ids=["doc-A"],
        hits=_hits(*[(f"doc-X-{i}", 0.6 - i * 0.01) for i in range(12)]),
        thresholds=DifficultyThresholds(),
    )
    assert diff is None
    assert meta["drop_reason"] == "gold_not_in_topk"


def test_assign_difficulty_impossible_when_unanswerable_and_low_score() -> None:
    diff, _ = _assign_difficulty(
        question_type="unanswerable",
        expected_doc_ids=[],
        hits=_hits(("doc-A", 0.42)),
        thresholds=DifficultyThresholds(),
    )
    assert diff == "impossible"


def test_assign_difficulty_downgrade_when_unanswerable_but_retriever_matches() -> None:
    diff, meta = _assign_difficulty(
        question_type="unanswerable",
        expected_doc_ids=[],
        hits=_hits(("doc-A", 0.85)),
        thresholds=DifficultyThresholds(),
    )
    assert diff == "hard"
    assert meta.get("note") == "unanswerable-but-retriever-matched"


# ---------------------------------------------------------------------------
# Dedup + per-doc cap + stratification
# ---------------------------------------------------------------------------


def test_dedup_rows_preserves_first_seen() -> None:
    rows = [
        {"query": "연차 신청 방법"},
        {"query": "연차 신청 방법"},   # exact dupe
        {"query": " 연차  신청  방법 "},  # whitespace variant
        {"query": "법인카드 한도"},
    ]
    kept, dropped = _dedup_rows(rows)
    assert len(kept) == 2
    assert dropped == 2


def test_apply_per_doc_cap_leaves_unanswerable_alone() -> None:
    rows = [
        {"expected_doc_ids": ["doc-A"], "question_type": "factoid", "query": "q1"},
        {"expected_doc_ids": ["doc-A"], "question_type": "factoid", "query": "q2"},
        {"expected_doc_ids": ["doc-A"], "question_type": "factoid", "query": "q3"},
        {"expected_doc_ids": ["doc-A"], "question_type": "factoid", "query": "q4"},
        {"expected_doc_ids": ["doc-A"], "question_type": "factoid", "query": "q5"},
        {"expected_doc_ids": [], "question_type": "unanswerable", "query": "u1"},
        {"expected_doc_ids": [], "question_type": "unanswerable", "query": "u2"},
    ]
    kept, dropped = _apply_per_doc_cap(rows, cap=4, seed=42)
    assert dropped == 1
    types = [r["question_type"] for r in kept]
    assert types.count("unanswerable") == 2
    assert types.count("factoid") == 4


def test_parse_target_defaults_missing_buckets_to_zero() -> None:
    t = _parse_target("easy:70,medium:80,hard:40,impossible:10")
    assert t == {"easy": 70, "medium": 80, "hard": 40, "impossible": 10}
    t = _parse_target("easy:5")
    assert t == {"easy": 5, "medium": 0, "hard": 0, "impossible": 0}


def test_stratify_respects_target_and_reports_shortfalls() -> None:
    rows = []
    for bucket, n in [("easy", 100), ("medium", 50), ("hard", 10), ("impossible", 0)]:
        for i in range(n):
            rows.append({"query": f"{bucket}-{i}", "difficulty": bucket})
    target = {"easy": 70, "medium": 80, "hard": 40, "impossible": 10}
    kept, achieved, shortfalls = _stratify(rows, target=target, seed=7)
    assert achieved["easy"] == 70
    assert achieved["medium"] == 50  # pool only had 50
    assert achieved["hard"] == 10  # pool only had 10
    assert achieved["impossible"] == 0
    assert shortfalls["medium"] == 30
    assert shortfalls["hard"] == 30
    assert shortfalls["impossible"] == 10
    assert len(kept) == 130


# ---------------------------------------------------------------------------
# ResumableJsonlWriter.drop
# ---------------------------------------------------------------------------


def test_resumable_writer_drop_removes_row(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    writer = ResumableJsonlWriter(path, key_fn=lambda r: str(r["doc_id"]))
    writer.append({"doc_id": "a", "v": 1})
    writer.append({"doc_id": "b", "v": 2})
    writer.append({"doc_id": "c", "v": 3})
    removed = writer.drop("b")
    assert removed is True
    remaining = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [r["doc_id"] for r in remaining] == ["a", "c"]
    # Dropping again is a no-op.
    assert writer.drop("b") is False
    # After dropping, append can reintroduce the key with fresh content.
    writer.append({"doc_id": "b", "v": 99})
    again = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert any(r["doc_id"] == "b" and r["v"] == 99 for r in again)
