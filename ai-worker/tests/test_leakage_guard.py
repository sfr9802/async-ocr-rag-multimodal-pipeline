"""Unit tests for ``eval.harness.leakage_guard``."""

from __future__ import annotations

from eval.harness.leakage_guard import (
    annotate_leakage,
    classify_leakage_risk,
    LEAKAGE_BENIGN_TYPES,
    LEAKAGE_SENSITIVE_TYPES,
    QUERY_TYPE_ALIAS_VARIANT,
    QUERY_TYPE_AMBIGUOUS,
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_INDIRECT_ENTITY,
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_SECTION_INTENT,
    QUERY_TYPE_UNANSWERABLE,
    render_leakage_md,
    summarize_leakage,
)


class TestClassifyLeakageRisk:
    def test_unanswerable_always_na(self):
        for orisk in ("low", "medium", "high", "not_applicable"):
            assert classify_leakage_risk(
                query_type=QUERY_TYPE_UNANSWERABLE, overlap_risk=orisk,
            ) == "not_applicable"

    def test_overlap_na_always_na(self):
        for qt in (QUERY_TYPE_PARAPHRASE_SEMANTIC, QUERY_TYPE_DIRECT_TITLE):
            assert classify_leakage_risk(
                query_type=qt, overlap_risk="not_applicable",
            ) == "not_applicable"

    def test_benign_high_overlap_low(self):
        # direct_title and alias_variant get LOW even at high overlap.
        for qt in LEAKAGE_BENIGN_TYPES:
            assert classify_leakage_risk(
                query_type=qt, overlap_risk="high",
            ) == "low"

    def test_sensitive_high_overlap_high(self):
        for qt in LEAKAGE_SENSITIVE_TYPES:
            assert classify_leakage_risk(
                query_type=qt, overlap_risk="high",
            ) == "high"

    def test_sensitive_medium_overlap_medium(self):
        for qt in LEAKAGE_SENSITIVE_TYPES:
            assert classify_leakage_risk(
                query_type=qt, overlap_risk="medium",
            ) == "medium"

    def test_sensitive_low_overlap_low(self):
        for qt in LEAKAGE_SENSITIVE_TYPES:
            assert classify_leakage_risk(
                query_type=qt, overlap_risk="low",
            ) == "low"

    def test_ambiguous_high_downgrades_to_medium(self):
        # Ambiguous + high overlap → medium (downgrade per matrix).
        assert classify_leakage_risk(
            query_type=QUERY_TYPE_AMBIGUOUS, overlap_risk="high",
        ) == "medium"

    def test_ambiguous_low_overlap_low(self):
        assert classify_leakage_risk(
            query_type=QUERY_TYPE_AMBIGUOUS, overlap_risk="medium",
        ) == "low"


class TestAnnotateLeakage:
    def test_sets_leakage_risk_field(self):
        rec = {
            "query_type": QUERY_TYPE_PARAPHRASE_SEMANTIC,
            "lexical_overlap": {"overlap_risk": "high"},
        }
        out = annotate_leakage(rec)
        assert out["leakage_risk"] == "high"

    def test_idempotent(self):
        rec = {
            "query_type": QUERY_TYPE_DIRECT_TITLE,
            "lexical_overlap": {"overlap_risk": "high"},
        }
        annotate_leakage(rec)
        before = rec["leakage_risk"]
        annotate_leakage(rec)
        assert rec["leakage_risk"] == before


class TestSummarizeLeakage:
    def test_empty_input(self):
        out = summarize_leakage([])
        assert out["high_risk_query_ids"] == []
        assert out["benign_high_overlap_count"] == 0
        assert out["ambiguous_high_overlap_count"] == 0

    def test_high_risk_collected(self):
        rec = {
            "query_id": "q1",
            "query_type": QUERY_TYPE_PARAPHRASE_SEMANTIC,
            "bucket": "main_work",
            "lexical_overlap": {"overlap_risk": "high"},
            "leakage_risk": "high",
        }
        out = summarize_leakage([rec])
        assert out["high_risk_query_ids"] == ["q1"]

    def test_benign_high_counted(self):
        rec = {
            "query_id": "q1",
            "query_type": QUERY_TYPE_DIRECT_TITLE,
            "bucket": "main_work",
            "lexical_overlap": {"overlap_risk": "high"},
            "leakage_risk": "low",
        }
        out = summarize_leakage([rec])
        assert out["benign_high_overlap_count"] == 1
        assert out["high_risk_query_ids"] == []

    def test_ambiguous_high_counted_separately(self):
        rec = {
            "query_id": "q1",
            "query_type": QUERY_TYPE_AMBIGUOUS,
            "bucket": "main_work",
            "lexical_overlap": {"overlap_risk": "high"},
            "leakage_risk": "medium",
        }
        out = summarize_leakage([rec])
        assert out["ambiguous_high_overlap_count"] == 1
        assert out["high_risk_query_ids"] == []


class TestRenderLeakageMd:
    def test_renders_section_headers(self):
        block = summarize_leakage([])
        md = render_leakage_md(block)
        assert "## Leakage" in md
        assert "Per-query-type leakage_risk" in md
        assert "Per-bucket leakage_risk" in md

    def test_high_risk_listed(self):
        recs = [
            {
                "query_id": f"q{i}",
                "query_type": QUERY_TYPE_PARAPHRASE_SEMANTIC,
                "bucket": "main_work",
                "lexical_overlap": {"overlap_risk": "high"},
                "leakage_risk": "high",
            }
            for i in range(3)
        ]
        block = summarize_leakage(recs)
        md = render_leakage_md(block)
        for i in range(3):
            assert f"`q{i}`" in md
