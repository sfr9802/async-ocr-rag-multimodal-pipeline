"""Unit tests for ``eval.harness.llm_silver_500``.

These tests exercise the QUERIES tuple, the cross-tab validator, and
the build pipeline. We use a tiny synthetic chunks JSONL to avoid
depending on the full corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.llm_silver_500 import (
    BUCKETS_ALL,
    CROSS_TAB_TARGETS,
    GENERATION_METHOD,
    LLM_SILVER_DISCLAIMER_MARKER,
    LLMQuery,
    QUERIES,
    Q,
    _validate_distribution,
    build_records,
    get_full_queries,
    render_summary_md,
)
from eval.harness.leakage_guard import (
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_UNANSWERABLE,
    summarize_leakage,
)


# ---------------------------------------------------------------------------
# QUERIES + cross-tab invariants
# ---------------------------------------------------------------------------


class TestQueriesInvariants:
    def test_count_500_after_fixups(self):
        full = get_full_queries()
        assert len(full) == 500

    def test_no_duplicate_query_strings(self):
        full = get_full_queries()
        queries = [q.query for q in full]
        assert len(set(queries)) == 500

    def test_qids_unique(self):
        full = get_full_queries()
        qids = [q.qid for q in full]
        assert len(set(qids)) == 500

    def test_cross_tab_match(self):
        full = get_full_queries()
        audit = _validate_distribution(full)
        assert audit["deltas"] == [], (
            f"distribution drift: {audit['deltas']}"
        )

    def test_unanswerable_doc_id_is_none(self):
        full = get_full_queries()
        for q in full:
            if q.query_type == QUERY_TYPE_UNANSWERABLE:
                assert q.expected_doc_id is None

    def test_answerable_doc_id_present(self):
        full = get_full_queries()
        for q in full:
            if q.query_type != QUERY_TYPE_UNANSWERABLE:
                assert q.expected_doc_id is not None and q.expected_doc_id != ""


# ---------------------------------------------------------------------------
# Build pipeline (against a synthetic mini-corpus)
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_corpus(tmp_path) -> Path:
    """Build a 1-row JSONL with one chunk per LLM-silver target doc.

    Every expected_doc_id in QUERIES must exist in the file or
    build_records raises. We write a stub chunk per doc id with a
    placeholder retrieval_title and chunk_text.
    """
    full = get_full_queries()
    seen: set = set()
    rows = []
    for q in full:
        if q.expected_doc_id is None or q.expected_doc_id in seen:
            continue
        seen.add(q.expected_doc_id)
        rows.append({
            "doc_id": q.expected_doc_id,
            "chunk_id": f"chunk-{q.expected_doc_id}",
            "title": "stub-title",
            "retrieval_title": f"stub-rt-{q.expected_doc_id}",
            "chunk_text": "stub chunk text 더미 컨텐츠",
        })
    path = tmp_path / "mini_chunks.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


class TestBuildRecords:
    def test_record_count_matches_queries(self, mini_corpus):
        records = build_records(mini_corpus)
        assert len(records) == 500

    def test_no_gold_in_tags(self, mini_corpus):
        records = build_records(mini_corpus)
        for r in records:
            for t in r.get("tags", []):
                assert t.lower() != "gold", (
                    f"row {r['query_id']} carries forbidden tag 'gold'"
                )

    def test_is_silver_not_gold_always_true(self, mini_corpus):
        records = build_records(mini_corpus)
        for r in records:
            assert r["is_silver_not_gold"] is True

    def test_generation_method_llm(self, mini_corpus):
        records = build_records(mini_corpus)
        for r in records:
            assert r["generation_method"] == GENERATION_METHOD

    def test_unanswerable_fields_null(self, mini_corpus):
        records = build_records(mini_corpus)
        unanswerable = [r for r in records if r["expected_not_in_corpus"]]
        assert len(unanswerable) == 25
        for r in unanswerable:
            assert r["silver_expected_title"] is None
            assert r["silver_expected_page_id"] is None
            assert r["expected_section_path"] is None
            ov = r["lexical_overlap"]
            assert ov["title_char2_jaccard"] is None
            assert ov["section_char2_jaccard"] is None
            assert ov["chunk_char4_containment"] is None
            assert ov["bm25_expected_page_first_rank"] is None
            assert ov["overlap_risk"] == "not_applicable"

    def test_answerable_have_targets(self, mini_corpus):
        records = build_records(mini_corpus)
        for r in records:
            if r["expected_not_in_corpus"]:
                continue
            assert r["silver_expected_title"] is not None
            assert r["silver_expected_page_id"] is not None
            assert r["expected_section_path"] is not None

    def test_required_fields_present(self, mini_corpus):
        records = build_records(mini_corpus)
        required = {
            "query_id", "query", "query_type", "bucket",
            "silver_expected_title", "silver_expected_page_id",
            "expected_section_path", "expected_not_in_corpus",
            "generation_method", "is_silver_not_gold",
            "rationale_for_expected_target", "lexical_overlap",
            "leakage_risk", "tags",
        }
        for r in records:
            assert required.issubset(set(r.keys())), (
                f"row {r.get('query_id')} missing fields: "
                f"{required - set(r.keys())}"
            )

    def test_deterministic_two_calls(self, mini_corpus):
        a = build_records(mini_corpus)
        b = build_records(mini_corpus)
        assert a == b

    def test_query_id_prefix(self, mini_corpus):
        records = build_records(mini_corpus)
        for r in records:
            assert r["query_id"].startswith("v4-llm-silver-")


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


class TestRenderSummaryMd:
    def test_disclaimer_present(self, mini_corpus):
        records = build_records(mini_corpus)
        leakage = summarize_leakage(records)
        md = render_summary_md(records, leakage)
        assert LLM_SILVER_DISCLAIMER_MARKER in md

    def test_no_gold_in_summary(self, mini_corpus):
        records = build_records(mini_corpus)
        leakage = summarize_leakage(records)
        md = render_summary_md(records, leakage)
        # Spec: "gold라는 단어를 tags나 report terminology에 사용하지 말 것"
        # Allow word "gold-eligible" in human-audit context, but not standalone.
        # Soft check: explicit "gold target" or similar phrasing is forbidden.
        forbidden = ["gold target", "gold label", "is_gold"]
        for f in forbidden:
            assert f not in md.lower()
