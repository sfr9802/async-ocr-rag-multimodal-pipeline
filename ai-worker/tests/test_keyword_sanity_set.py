"""Unit tests for ``eval.harness.keyword_sanity_set``."""

from __future__ import annotations

from eval.harness.keyword_sanity_set import (
    FORBIDDEN_TAGS,
    REQUIRED_TAGS,
    SANITY_DISCLAIMER_MARKER,
    SANITY_ID_PREFIX,
    SANITY_LABEL_SOURCE,
    SANITY_PURPOSE,
    SANITY_REPLACEMENT_FILE,
    STRIPPED_TAGS,
    TAG_KEYWORD_DERIVED,
    TAG_NOT_MAIN_EVAL,
    TAG_SANITY_SET,
    TAG_SILVER,
    TAG_SMOKE_TEST,
    render_summary_md,
    retag_record,
    retag_records,
)


def _legacy_row(qid: int = 1) -> dict:
    """Build a minimal silver-500-shaped row for testing."""
    return {
        "id": f"v4-silver-500-{qid:04d}",
        "query": "원피스의 줄거리에 대해 알려주세요.",
        "language": "ko",
        "expected_doc_ids": ["abc123"],
        "tags": [
            "anime", "v4-silver-500", "synthetic", "silver",
            "main_work", "title_lookup", "deterministic",
        ],
        "v4_meta": {
            "bucket": "main_work",
            "page_title": "원피스",
            "retrieval_title": "원피스(애니메이션)",
            "is_silver_not_gold": True,
        },
    }


class TestRetagRecord:
    def test_id_prefix_changed(self):
        out = retag_record(_legacy_row(qid=42), idx=42)
        assert out["id"].startswith(SANITY_ID_PREFIX)
        # Numeric tail preserved.
        assert out["id"].endswith("0042")

    def test_required_tags_added(self):
        out = retag_record(_legacy_row(), idx=1)
        for t in REQUIRED_TAGS:
            assert t in out["tags"]
        assert TAG_KEYWORD_DERIVED in out["tags"]
        assert TAG_SANITY_SET in out["tags"]
        assert TAG_SMOKE_TEST in out["tags"]
        assert TAG_NOT_MAIN_EVAL in out["tags"]
        assert TAG_SILVER in out["tags"]

    def test_stripped_tags_removed(self):
        out = retag_record(_legacy_row(), idx=1)
        for t in STRIPPED_TAGS:
            assert t not in out["tags"]

    def test_forbidden_tags_dropped(self):
        # Even if the source has "gold", it should be stripped.
        row = _legacy_row()
        row["tags"].append("gold")
        out = retag_record(row, idx=1)
        for t in FORBIDDEN_TAGS:
            assert t not in out["tags"]

    def test_v4_meta_purpose_set(self):
        out = retag_record(_legacy_row(), idx=1)
        assert out["v4_meta"]["purpose"] == SANITY_PURPOSE
        assert out["v4_meta"]["replaced_by"] == SANITY_REPLACEMENT_FILE
        assert out["v4_meta"]["silver_label_source"] == SANITY_LABEL_SOURCE
        assert out["v4_meta"]["is_silver_not_gold"] is True

    def test_query_preserved(self):
        out = retag_record(_legacy_row(), idx=1)
        assert out["query"] == "원피스의 줄거리에 대해 알려주세요."

    def test_expected_doc_ids_preserved(self):
        out = retag_record(_legacy_row(), idx=1)
        assert out["expected_doc_ids"] == ["abc123"]


class TestRetagRecords:
    def test_counts(self):
        rows = [_legacy_row(qid=i) for i in range(1, 6)]
        out, stats = retag_records(rows)
        assert stats.rows_in == 5
        assert stats.rows_out == 5
        assert stats.ids_rewritten == 5
        assert len(out) == 5

    def test_forbidden_tags_logged(self):
        row = _legacy_row()
        row["tags"].append("gold")
        out, stats = retag_records([row])
        assert len(stats.forbidden_tags_seen) == 1


class TestRenderSummaryMd:
    def test_disclaimer_present(self):
        rows = [retag_record(_legacy_row(qid=i), idx=i) for i in range(1, 4)]
        out, stats = retag_records([_legacy_row(qid=i) for i in range(1, 4)])
        md = render_summary_md(out, stats)
        assert SANITY_DISCLAIMER_MARKER in md

    def test_main_eval_warning(self):
        out, stats = retag_records([_legacy_row()])
        md = render_summary_md(out, stats)
        assert "NOT a use case" in md or "main eval" in md.lower()
