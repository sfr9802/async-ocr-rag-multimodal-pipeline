"""Tests for the Phase 7.7 / 7.7.1 answerability audit harness.

Exercises four row-level layers (Phase 7.7) plus the bundle-level
layer added in Phase 7.7.1:

  * :func:`parse_label` / :func:`parse_flags` — input coercion
    (positive cases, negative cases, mixed enum/int round-trip).
  * :func:`parse_labeled_rows` / :func:`read_labeled_csv` —
    duplicate detection and ``chunk_text`` empty rejection.
  * :func:`compute_variant_metrics` /
    :func:`compute_all_variants` — metric definitions across
    representative top-5 fixtures (answerable@k, fully@k, page-hit
    confusion, section-miss confusion, flag counts).
  * :func:`render_markdown_report` /
    :func:`build_json_summary` — section presence, empty-input
    handling, structural JSON shape.
  * :func:`build_bundle_export_rows` / :func:`render_bundle_text` /
    :func:`compute_bundle_variant_metrics` /
    :func:`sample_bundle_records` — Phase 7.7.1 bundle-level data
    model, scoring, report integration, and sampling helper.

The CLI smoke tests at the bottom use ``main()`` directly so failures
are caught on the same stack as the test, rather than via subprocess.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from eval.harness.answerability_audit import (
    ANSWERABLE_MIN_LEVEL,
    AnswerabilityExportRow,
    AnswerabilityFlag,
    AnswerabilityLabel,
    AnswerabilityLabeledRow,
    AnswerabilityValidationError,
    EXPORT_COLUMNS,
    GoldRef,
    PREFERRED_VARIANT_ORDER,
    RetrievedRef,
    VARIANT_BASELINE,
    VARIANT_PRODUCTION_RECOMMENDED,
    VARIANT_SECTION_AWARE_CANDIDATE,
    VariantMetrics,
    build_export_rows,
    build_json_summary,
    compute_all_variants,
    compute_page_hit,
    compute_section_hit,
    compute_variant_metrics,
    parse_flags,
    parse_label,
    parse_labeled_row,
    parse_labeled_rows,
    read_labeled_csv,
    read_labeled_jsonl,
    render_markdown_report,
    write_export_csv,
    write_export_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    *,
    query_id: str,
    variant_name: str,
    rank: int,
    label: AnswerabilityLabel,
    page_hit: bool = False,
    section_hit: bool = False,
    flags: Tuple[AnswerabilityFlag, ...] = (),
    chunk_text: str = "stub chunk",
    gold_page_id: str = "p_gold",
    gold_section_path: str = "줄거리",
    retrieved_page_id: Optional[str] = None,
    retrieved_section_path: str = "줄거리",
) -> AnswerabilityLabeledRow:
    """Convenience builder so the per-test fixtures stay readable."""
    return AnswerabilityLabeledRow(
        query_id=query_id,
        query=f"q for {query_id}",
        variant_name=variant_name,
        rank=rank,
        gold_page_id=gold_page_id,
        gold_page_title="gold-title",
        gold_section_id="",
        gold_section_path=gold_section_path,
        retrieved_page_id=(
            retrieved_page_id
            if retrieved_page_id is not None
            else (gold_page_id if page_hit else "p_other")
        ),
        retrieved_page_title="r-title",
        retrieved_section_id="",
        retrieved_section_path=retrieved_section_path,
        chunk_id=f"c-{query_id}-{rank}",
        chunk_text=chunk_text,
        page_hit=page_hit,
        section_hit=section_hit,
        label=label,
        flags=flags,
        notes="",
    )


# ---------------------------------------------------------------------------
# Label / flag parsing
# ---------------------------------------------------------------------------


class TestParseLabel:
    """``parse_label`` accepts ints, stringified ints, and enum names;
    rejects everything else with :class:`AnswerabilityValidationError`."""

    @pytest.mark.parametrize("value, expected", [
        (0, AnswerabilityLabel.NOT_RELEVANT),
        (1, AnswerabilityLabel.RELATED_BUT_NOT_ANSWERABLE),
        (2, AnswerabilityLabel.PARTIALLY_ANSWERABLE),
        (3, AnswerabilityLabel.FULLY_ANSWERABLE),
        ("0", AnswerabilityLabel.NOT_RELEVANT),
        ("3", AnswerabilityLabel.FULLY_ANSWERABLE),
        (" 2 ", AnswerabilityLabel.PARTIALLY_ANSWERABLE),
        ("NOT_RELEVANT", AnswerabilityLabel.NOT_RELEVANT),
        ("not_relevant", AnswerabilityLabel.NOT_RELEVANT),
        ("FULLY_ANSWERABLE", AnswerabilityLabel.FULLY_ANSWERABLE),
        (AnswerabilityLabel.PARTIALLY_ANSWERABLE,
         AnswerabilityLabel.PARTIALLY_ANSWERABLE),
    ])
    def test_accepts_canonical_values(self, value, expected):
        assert parse_label(value) is expected

    @pytest.mark.parametrize("value", [
        4, -1, 99, "4", "PARTIAL", "MOSTLY_ANSWERABLE",
        "", "   ", None, 1.5, True, False,
    ])
    def test_rejects_invalid(self, value):
        with pytest.raises(AnswerabilityValidationError):
            parse_label(value)


class TestParseFlags:
    """``parse_flags`` accepts pipe-strings, JSON arrays, and enums;
    rejects unknown flags."""

    def test_empty_returns_tuple(self):
        assert parse_flags("") == ()
        assert parse_flags("   ") == ()
        assert parse_flags(None) == ()
        assert parse_flags([]) == ()

    def test_single_pipe_string(self):
        flags = parse_flags("wrong_page")
        assert flags == (AnswerabilityFlag.WRONG_PAGE,)

    def test_multi_pipe_string(self):
        flags = parse_flags("wrong_page|evidence_too_noisy")
        assert flags == (
            AnswerabilityFlag.WRONG_PAGE,
            AnswerabilityFlag.EVIDENCE_TOO_NOISY,
        )

    def test_list_input(self):
        flags = parse_flags([
            "needs_subpage", "needs_cross_section",
        ])
        assert flags == (
            AnswerabilityFlag.NEEDS_SUBPAGE,
            AnswerabilityFlag.NEEDS_CROSS_SECTION,
        )

    def test_dedups(self):
        flags = parse_flags(
            "wrong_page|wrong_page|evidence_too_noisy",
        )
        assert flags == (
            AnswerabilityFlag.WRONG_PAGE,
            AnswerabilityFlag.EVIDENCE_TOO_NOISY,
        )

    def test_case_insensitive(self):
        flags = parse_flags("WRONG_PAGE|Right_Page_Wrong_Section")
        assert flags == (
            AnswerabilityFlag.WRONG_PAGE,
            AnswerabilityFlag.RIGHT_PAGE_WRONG_SECTION,
        )

    def test_rejects_unknown_flag(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_flags("not_a_real_flag")

    def test_rejects_unknown_in_list(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_flags(["wrong_page", "totally_made_up"])

    def test_rejects_bad_type(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_flags(42)


# ---------------------------------------------------------------------------
# Labelled-row parsing + duplicate detection
# ---------------------------------------------------------------------------


def _csv_record(**overrides: Any) -> Dict[str, Any]:
    """Build a CSV-shaped dict for ``parse_labeled_row``.

    Defaults satisfy validation; tests override only the field they
    are exercising so failure messages stay readable.
    """
    base = {
        "query_id": "q1",
        "query": "what about X",
        "variant_name": "baseline",
        "rank": "1",
        "gold_page_id": "p_gold",
        "gold_page_title": "Gold Page",
        "gold_section_id": "",
        "gold_section_path": "줄거리",
        "retrieved_page_id": "p_gold",
        "retrieved_page_title": "Gold Page",
        "retrieved_section_id": "",
        "retrieved_section_path": "줄거리",
        "chunk_id": "c1",
        "chunk_text": "X happens at the start of the work...",
        "page_hit": "true",
        "section_hit": "true",
        "label_answerability": "FULLY_ANSWERABLE",
        "flags": "",
        "notes": "",
    }
    base.update(overrides)
    return base


class TestParseLabeledRow:
    def test_round_trips_int_label(self):
        row = parse_labeled_row(_csv_record(label_answerability="2"))
        assert row.label is AnswerabilityLabel.PARTIALLY_ANSWERABLE
        assert row.page_hit is True
        assert row.section_hit is True

    def test_mixed_int_and_enum_in_same_dataset(self):
        # Two records — one as int, one as enum name — must produce
        # the same canonical type so downstream comparisons work.
        rec_int = _csv_record(query_id="qA", label_answerability="0")
        rec_enum = _csv_record(
            query_id="qB", label_answerability="NOT_RELEVANT",
        )
        rows = parse_labeled_rows([rec_int, rec_enum])
        assert all(
            r.label is AnswerabilityLabel.NOT_RELEVANT for r in rows
        )

    def test_rejects_empty_chunk_text(self):
        with pytest.raises(AnswerabilityValidationError) as ex:
            parse_labeled_row(_csv_record(chunk_text=""))
        assert "chunk_text" in str(ex.value)

    def test_rejects_whitespace_chunk_text(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_labeled_row(_csv_record(chunk_text="    "))

    def test_rejects_invalid_label(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_labeled_row(_csv_record(label_answerability="WAT"))

    def test_rejects_invalid_flag(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_labeled_row(_csv_record(flags="not_a_flag"))

    def test_rejects_missing_required_column(self):
        rec = _csv_record()
        rec.pop("retrieved_page_id")
        with pytest.raises(AnswerabilityValidationError):
            parse_labeled_row(rec)

    def test_rejects_non_int_rank(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_labeled_row(_csv_record(rank="not_a_number"))

    def test_parses_pipe_flags(self):
        row = parse_labeled_row(_csv_record(
            flags="wrong_page|evidence_too_noisy",
        ))
        assert row.flags == (
            AnswerabilityFlag.WRONG_PAGE,
            AnswerabilityFlag.EVIDENCE_TOO_NOISY,
        )

    def test_parses_json_list_flags(self):
        row = parse_labeled_row(_csv_record(
            flags=["needs_subpage", "ambiguous_query"],
        ))
        assert AnswerabilityFlag.NEEDS_SUBPAGE in row.flags
        assert AnswerabilityFlag.AMBIGUOUS_QUERY in row.flags


class TestParseLabeledRowsDuplicates:
    """``parse_labeled_rows`` rejects same (qid,variant,rank) duplicates."""

    def test_accepts_unique_keys(self):
        rows = parse_labeled_rows([
            _csv_record(query_id="q1", rank="1"),
            _csv_record(query_id="q1", rank="2"),
            _csv_record(query_id="q2", rank="1"),
        ])
        assert len(rows) == 3

    def test_rejects_duplicate(self):
        with pytest.raises(AnswerabilityValidationError) as ex:
            parse_labeled_rows([
                _csv_record(query_id="q1", rank="1"),
                _csv_record(query_id="q1", rank="1"),
            ])
        assert "Duplicate" in str(ex.value)

    def test_distinct_variants_share_keys(self):
        # Same query_id + rank are fine when variant differs.
        rows = parse_labeled_rows([
            _csv_record(query_id="q1", variant_name="baseline", rank="1"),
            _csv_record(
                query_id="q1",
                variant_name="production_recommended",
                rank="1",
            ),
        ])
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Hit helpers
# ---------------------------------------------------------------------------


class TestHitHelpers:
    def test_page_hit_exact(self):
        assert compute_page_hit("pid", "pid") is True

    def test_page_hit_mismatch(self):
        assert compute_page_hit("pid", "other") is False

    def test_page_hit_empty_gold(self):
        assert compute_page_hit("", "pid") is False

    def test_section_hit_requires_page_hit(self):
        assert compute_section_hit(
            "p1", "줄거리", "p2", "줄거리",
        ) is False

    def test_section_hit_prefix_match(self):
        assert compute_section_hit(
            "p1", "등장인물", "p1", "등장인물 > 주인공",
        ) is True

    def test_section_hit_does_not_match_sibling_prefix(self):
        assert compute_section_hit(
            "p1", "등장인물", "p1", "등장인물상",
        ) is False

    def test_section_hit_no_prefix(self):
        assert compute_section_hit(
            "p1", "등장인물", "p1", "줄거리",
        ) is False

    def test_section_hit_empty_gold_section(self):
        # Empty gold section path returns False — section hit is
        # undefined in that case, callers shouldn't conflate with True.
        assert compute_section_hit("p1", "", "p1", "anything") is False


# ---------------------------------------------------------------------------
# Export builder + CSV / JSONL round-trip
# ---------------------------------------------------------------------------


class TestBuildExportRows:
    def test_builds_one_row_per_chunk(self):
        gold = GoldRef(
            page_id="p1", page_title="P1",
            section_id="", section_path="줄거리",
        )
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1", page_id="p1",
                page_title="P1", section_path="줄거리",
                chunk_text="...",
            ),
            RetrievedRef(
                rank=2, chunk_id="c2", page_id="p2",
                page_title="P2", section_path="설정",
                chunk_text="...",
            ),
        ]
        rows = build_export_rows(
            query_id="q1", query="...",
            variant_name="baseline", gold=gold,
            retrieved=retrieved, top_k=5,
        )
        assert len(rows) == 2
        assert rows[0].page_hit is True
        assert rows[0].section_hit is True
        assert rows[1].page_hit is False
        assert rows[1].section_hit is False

    def test_top_k_truncates(self):
        gold = GoldRef()
        retrieved = [
            RetrievedRef(rank=i, chunk_text="t") for i in range(1, 11)
        ]
        rows = build_export_rows(
            query_id="q1", query="...", variant_name="baseline",
            gold=gold, retrieved=retrieved, top_k=3,
        )
        assert len(rows) == 3

    def test_top_k_zero_keeps_all(self):
        gold = GoldRef()
        retrieved = [
            RetrievedRef(rank=i, chunk_text="t") for i in range(1, 8)
        ]
        rows = build_export_rows(
            query_id="q1", query="...", variant_name="baseline",
            gold=gold, retrieved=retrieved, top_k=0,
        )
        assert len(rows) == 7

    def test_export_row_has_blank_label_columns_in_csv_dict(self):
        gold = GoldRef()
        retrieved = [RetrievedRef(rank=1, chunk_text="t")]
        rows = build_export_rows(
            query_id="q1", query="...", variant_name="baseline",
            gold=gold, retrieved=retrieved, top_k=1,
        )
        d = rows[0].to_csv_dict()
        assert d["label_answerability"] == ""
        assert d["flags"] == ""
        assert d["notes"] == ""
        # And every export column is present.
        assert set(d.keys()) == set(EXPORT_COLUMNS)


class TestExportCsvRoundTrip:
    def test_csv_write_then_label_then_read(self, tmp_path: Path):
        gold = GoldRef(page_id="p1", section_path="줄거리")
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1", page_id="p1",
                section_path="줄거리",
                chunk_text="text-1",
            ),
            RetrievedRef(
                rank=2, chunk_id="c2", page_id="p_other",
                section_path="설정",
                chunk_text="text-2",
            ),
        ]
        rows = build_export_rows(
            query_id="q1", query="hello", variant_name="baseline",
            gold=gold, retrieved=retrieved, top_k=5,
        )
        csv_path = tmp_path / "out.csv"
        write_export_csv(csv_path, rows)

        # Simulate a reviewer filling in label cells.
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        records[0]["label_answerability"] = "FULLY_ANSWERABLE"
        records[1]["label_answerability"] = "0"
        records[1]["flags"] = "wrong_page"

        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(EXPORT_COLUMNS))
            writer.writeheader()
            writer.writerows(records)

        labelled = read_labeled_csv(csv_path)
        assert len(labelled) == 2
        assert labelled[0].label is AnswerabilityLabel.FULLY_ANSWERABLE
        assert labelled[1].label is AnswerabilityLabel.NOT_RELEVANT
        assert labelled[1].flags == (AnswerabilityFlag.WRONG_PAGE,)


class TestExportJsonlRoundTrip:
    def test_jsonl_write_then_label_then_read(self, tmp_path: Path):
        gold = GoldRef(page_id="p1", section_path="줄거리")
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1", page_id="p1",
                section_path="줄거리", chunk_text="text",
            ),
        ]
        rows = build_export_rows(
            query_id="q1", query="hi",
            variant_name="baseline", gold=gold,
            retrieved=retrieved, top_k=1,
        )
        jsonl_path = tmp_path / "out.jsonl"
        write_export_jsonl(jsonl_path, rows)

        # Reviewer fills the label inline.
        out_lines = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            obj["label_answerability"] = 2
            obj["flags"] = ["needs_subpage"]
            out_lines.append(json.dumps(obj, ensure_ascii=False))
        jsonl_path.write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8",
        )

        labelled = read_labeled_jsonl(jsonl_path)
        assert len(labelled) == 1
        assert labelled[0].label is AnswerabilityLabel.PARTIALLY_ANSWERABLE
        assert labelled[0].flags == (AnswerabilityFlag.NEEDS_SUBPAGE,)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


class TestAnswerableAtK:
    """``answerable@k`` rolls up to the query level — at least one row
    at rank ≤ k with label >= PARTIALLY_ANSWERABLE."""

    def test_zero_when_all_below_threshold(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
            _row(
                query_id="q1", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.RELATED_BUT_NOT_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.answerable_at_1 == 0.0
        assert m.answerable_at_3 == 0.0
        assert m.answerable_at_5 == 0.0

    def test_one_when_first_rank_is_answerable(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.answerable_at_1 == 1.0

    def test_partial_counts_as_answerable(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=3,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.answerable_at_1 == 0.0
        assert m.answerable_at_3 == 1.0
        assert m.answerable_at_5 == 1.0

    def test_partial_or_better_at_5_alias(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        # By construction in the harness, this is an alias of
        # answerable@5 — a future regression that splits them apart
        # would silently break the report tables.
        assert m.partial_or_better_at_5 == m.answerable_at_5

    def test_query_level_aggregation(self):
        # Two queries: q1 has answerable@5; q2 does not.
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
            _row(
                query_id="q2", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.answerable_at_5 == 0.5


class TestFullyAnswerableAtK:
    def test_partial_does_not_count_as_fully(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.fully_answerable_at_1 == 0.0
        assert m.answerable_at_1 == 1.0

    def test_fully_at_each_k(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
            _row(
                query_id="q1", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
            _row(
                query_id="q1", variant_name="baseline", rank=4,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.fully_answerable_at_1 == 0.0
        assert m.fully_answerable_at_3 == 0.0
        assert m.fully_answerable_at_5 == 1.0


class TestPageHitButNotAnswerable:
    def test_counts_query_with_page_hit_but_no_answerable(self):
        # Right page found at rank 2, but every label says
        # not-answerable.
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT, page_hit=False,
            ),
            _row(
                query_id="q1", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.NOT_RELEVANT, page_hit=True,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.page_hit_but_not_answerable_count == 1

    def test_does_not_count_when_page_hit_and_answerable(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE, page_hit=True,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.page_hit_but_not_answerable_count == 0


class TestSectionMissButAnswerable:
    def test_counts_when_no_section_hit_but_answerable(self):
        # Page hit but section path mismatches; reviewer says it
        # answers anyway (cross-section synthesis).
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=False,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.section_miss_but_answerable_count == 1

    def test_does_not_count_when_section_hits(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.section_miss_but_answerable_count == 0


class TestFlagCounts:
    def test_counts_each_flag_type(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
                flags=(
                    AnswerabilityFlag.WRONG_PAGE,
                    AnswerabilityFlag.EVIDENCE_TOO_NOISY,
                ),
            ),
            _row(
                query_id="q2", variant_name="baseline", rank=2,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                flags=(AnswerabilityFlag.RIGHT_PAGE_WRONG_SECTION,),
                page_hit=True,
            ),
            _row(
                query_id="q3", variant_name="baseline", rank=4,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                flags=(AnswerabilityFlag.NEEDS_SUBPAGE,),
            ),
            # Rank > 5 is excluded from flag counts.
            _row(
                query_id="q4", variant_name="baseline", rank=8,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                flags=(AnswerabilityFlag.WRONG_PAGE,),
            ),
        ]
        m = compute_variant_metrics(rows, "baseline")
        assert m.wrong_page_count == 1
        assert m.right_page_wrong_section_count == 1
        assert m.evidence_too_noisy_count == 1
        assert m.needs_cross_section_count == 0
        assert m.needs_subpage_count == 1


# ---------------------------------------------------------------------------
# Variant comparison
# ---------------------------------------------------------------------------


class TestVariantComparison:
    def test_compute_all_variants_orders_known_first(self):
        rows: List[AnswerabilityLabeledRow] = []
        # Use all three canonical variants plus a custom one. Each
        # gets a single FULLY_ANSWERABLE row so metrics are non-zero
        # and obviously per-variant.
        for v in [
            "z_extra_variant",
            VARIANT_SECTION_AWARE_CANDIDATE,
            VARIANT_BASELINE,
            VARIANT_PRODUCTION_RECOMMENDED,
        ]:
            rows.append(
                _row(
                    query_id=f"q-{v}", variant_name=v, rank=1,
                    label=AnswerabilityLabel.FULLY_ANSWERABLE,
                    page_hit=True, section_hit=True,
                )
            )
        result = compute_all_variants(rows)
        names = [v.variant_name for v in result]
        assert names == [
            VARIANT_BASELINE,
            VARIANT_PRODUCTION_RECOMMENDED,
            VARIANT_SECTION_AWARE_CANDIDATE,
            "z_extra_variant",
        ]

    def test_two_variant_diff_signal(self):
        # Baseline: low answerability.
        # Production_recommended: high.
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
            _row(
                query_id="q1",
                variant_name="production_recommended",
                rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        per = {v.variant_name: v for v in compute_all_variants(rows)}
        assert per["baseline"].answerable_at_5 == 0.0
        assert per["production_recommended"].answerable_at_5 == 1.0


# ---------------------------------------------------------------------------
# Markdown rendering + JSON summary
# ---------------------------------------------------------------------------


class TestRenderMarkdownReport:
    def _representative_rows(self) -> List[AnswerabilityLabeledRow]:
        return [
            # Baseline: page hit but not answerable.
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.RELATED_BUT_NOT_ANSWERABLE,
                page_hit=True, section_hit=False,
                flags=(AnswerabilityFlag.RIGHT_PAGE_WRONG_SECTION,),
            ),
            _row(
                query_id="q2", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
                page_hit=False, section_hit=False,
                flags=(AnswerabilityFlag.WRONG_PAGE,),
            ),
            # Production_recommended: better.
            _row(
                query_id="q1",
                variant_name="production_recommended",
                rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
            _row(
                query_id="q2",
                variant_name="production_recommended",
                rank=2,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                page_hit=True, section_hit=False,
                flags=(AnswerabilityFlag.NEEDS_SUBPAGE,),
            ),
        ]

    def test_pinned_sections_present(self):
        md = render_markdown_report(self._representative_rows())
        for header in [
            "## Summary",
            "## Label distribution",
            "## Answerability metrics by variant",
            "## Page hit vs answerability confusion",
            "## Section hit vs answerability confusion",
            "## Failure buckets",
            "## Top failure examples",
            "## Interpretation guide",
            "## Next action recommendation",
        ]:
            assert header in md, f"missing pinned section: {header}"

    def test_interpretation_guide_warns_about_hit_replacement(self):
        md = render_markdown_report(self._representative_rows())
        assert "additive" in md.lower()

    def test_renders_with_known_variants_first(self):
        md = render_markdown_report(self._representative_rows())
        baseline_idx = md.find("| baseline")
        prod_idx = md.find("| production_recommended")
        # Both variants are rendered, baseline appears first.
        assert baseline_idx >= 0
        assert prod_idx > baseline_idx

    def test_empty_input_renders_no_data_blocks(self):
        md = render_markdown_report([])
        # Pinned sections still appear; tables show "no variants" /
        # "no failure rows" placeholders rather than crashing.
        assert "## Answerability metrics by variant" in md
        assert "no variants" in md
        assert "no page-hit failure rows" in md


class TestBuildJsonSummary:
    def test_keys_are_pinned(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
        ]
        summary = build_json_summary(rows)
        assert set(summary) == {
            "summary",
            "label_distribution",
            "variants",
            "page_hit_confusion",
            "section_hit_confusion",
        }
        assert summary["summary"]["n_rows"] == 1
        assert summary["summary"]["n_variants"] == 1
        assert summary["summary"]["n_queries"] == 1

    def test_label_distribution_contains_all_labels(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        summary = build_json_summary(rows)
        # Each of the four enum names is present (zero-counts allowed).
        for label in AnswerabilityLabel:
            assert label.name in summary["label_distribution"]

    def test_empty_input(self):
        summary = build_json_summary([])
        assert summary["summary"]["n_rows"] == 0
        assert summary["variants"] == []
        assert summary["page_hit_confusion"] == {}
        assert summary["section_hit_confusion"] == {}


# ---------------------------------------------------------------------------
# CLI smoke tests — exercise both scripts in-process so failures surface
# on the same stack.
# ---------------------------------------------------------------------------


class TestExportScriptCli:
    def test_export_csv_end_to_end(self, tmp_path: Path):
        # The export script reads a JSONL with the canonical schema
        # documented at the top of scripts.export_answerability_audit.
        from scripts.export_answerability_audit import main as export_main

        in_path = tmp_path / "ret.jsonl"
        in_payload = {
            "query_id": "q1",
            "query": "what about X",
            "gold": {
                "page_id": "p1", "page_title": "P1",
                "section_id": "", "section_path": "줄거리",
            },
            "results": [
                {
                    "rank": 1, "chunk_id": "c1", "page_id": "p1",
                    "page_title": "P1", "section_id": "",
                    "section_path": "줄거리",
                    "chunk_text": "X happens at the start...",
                },
                {
                    "rank": 2, "chunk_id": "c2", "page_id": "p_other",
                    "page_title": "Other", "section_id": "",
                    "section_path": "설정",
                    "chunk_text": "Unrelated context",
                },
            ],
        }
        in_path.write_text(
            json.dumps(in_payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        out_path = tmp_path / "labels.csv"
        rc = export_main([
            "--retrieval-results-path", str(in_path),
            "--out-path", str(out_path),
            "--variant-name", "baseline",
            "--top-k", "5",
        ])
        assert rc == 0
        assert out_path.exists()

        with out_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        assert len(records) == 2
        assert records[0]["label_answerability"] == ""
        assert records[0]["page_hit"] == "true"
        assert records[1]["page_hit"] == "false"


class TestScoreScriptCli:
    def test_score_csv_end_to_end(self, tmp_path: Path):
        from scripts.score_answerability_audit import main as score_main

        # Hand-write a small labelled CSV, then score.
        labelled_csv = tmp_path / "labels.csv"
        records = [
            _csv_record(
                query_id="q1", variant_name="baseline", rank="1",
                label_answerability="NOT_RELEVANT",
                page_hit="true", section_hit="false",
                flags="right_page_wrong_section",
            ),
            _csv_record(
                query_id="q1",
                variant_name="production_recommended", rank="1",
                label_answerability="3",
                page_hit="true", section_hit="true",
                flags="",
            ),
        ]
        with labelled_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(EXPORT_COLUMNS))
            writer.writeheader()
            writer.writerows(records)

        report_md = tmp_path / "report.md"
        json_out = tmp_path / "report.json"
        rc = score_main([
            "--labeled-path", str(labelled_csv),
            "--report-path", str(report_md),
            "--json-path", str(json_out),
        ])
        assert rc == 0
        assert report_md.exists()
        assert json_out.exists()

        text = report_md.read_text(encoding="utf-8")
        assert "## Answerability metrics by variant" in text
        assert "baseline" in text
        assert "production_recommended" in text

        summary = json.loads(json_out.read_text(encoding="utf-8"))
        # Both variants present in the structured summary; the
        # production_recommended variant has answerable@5 == 1.0.
        per = {v["variant_name"]: v for v in summary["variants"]}
        assert per["baseline"]["answerable_at_5"] == 0.0
        assert per["production_recommended"]["answerable_at_5"] == 1.0

    def test_score_rejects_invalid_label(self, tmp_path: Path):
        from scripts.score_answerability_audit import main as score_main

        labelled_csv = tmp_path / "labels.csv"
        with labelled_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(EXPORT_COLUMNS))
            writer.writeheader()
            writer.writerow(_csv_record(
                label_answerability="this_is_not_a_label",
            ))

        report_md = tmp_path / "report.md"
        with pytest.raises(SystemExit):
            score_main([
                "--labeled-path", str(labelled_csv),
                "--report-path", str(report_md),
            ])


# ===========================================================================
# Phase 7.7.1 — bundle-level (context-set) audit tests
# ===========================================================================


from eval.harness.answerability_audit import (  # noqa: E402
    BUNDLE_EXPORT_COLUMNS,
    BundleAnswerabilityMetrics,
    ContextBundleAuditRow,
    ContextBundleExportRow,
    DEFAULT_BUNDLE_TRUNCATE_CHARS,
    DEFAULT_TOP_K_SET,
    build_bundle_export_rows,
    compute_all_bundle_variants,
    compute_bundle_page_hit,
    compute_bundle_section_hit,
    compute_bundle_variant_metrics,
    parse_bundle_labeled_row,
    parse_bundle_labeled_rows,
    read_bundle_labeled_csv,
    read_bundle_labeled_jsonl,
    render_bundle_text,
    sample_bundle_records,
    write_bundle_export_csv,
    write_bundle_export_jsonl,
)


# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


def _bundle_row(
    *,
    query_id: str,
    variant_name: str,
    top_k: int,
    label: AnswerabilityLabel,
    page_hit: bool = False,
    section_hit: bool = False,
    flags: Tuple[AnswerabilityFlag, ...] = (),
    bundle_text: str = "default bundle text",
    chunk_ids: Tuple[str, ...] = (),
    retrieved_page_ids: Tuple[str, ...] = (),
    retrieved_section_paths: Tuple[str, ...] = (),
) -> ContextBundleAuditRow:
    """Convenience builder for bundle audit rows."""
    return ContextBundleAuditRow(
        query_id=query_id,
        query=f"q for {query_id}",
        variant_name=variant_name,
        top_k=top_k,
        gold_page_id="p_gold",
        gold_page_title="GoldTitle",
        gold_section_id="",
        gold_section_path="줄거리",
        retrieved_page_ids=retrieved_page_ids,
        retrieved_page_titles=(),
        retrieved_section_ids=(),
        retrieved_section_paths=retrieved_section_paths,
        chunk_ids=chunk_ids,
        context_bundle_text=bundle_text,
        page_hit_within_k=page_hit,
        section_hit_within_k=section_hit,
        label=label,
        flags=flags,
        notes="",
    )


def _bundle_csv_record(**overrides: Any) -> Dict[str, Any]:
    """Build a CSV-shaped dict for ``parse_bundle_labeled_row``.

    Defaults satisfy validation; tests override only the field they
    are exercising.
    """
    base = {
        "query_id": "qb1",
        "query": "what about X",
        "variant_name": "baseline",
        "top_k": "5",
        "gold_page_id": "p_gold",
        "gold_page_title": "Gold Page",
        "gold_section_id": "",
        "gold_section_path": "줄거리",
        "context_bundle_text": (
            "[Rank 1]\npage: P (p_gold)\nsection: 줄거리\n"
            "chunk_id: c1\ntext:\nfacts and evidence ..."
        ),
        "retrieved_page_ids": "p_gold|p_other",
        "retrieved_page_titles": "P|O",
        "retrieved_section_ids": "|",
        "retrieved_section_paths": "줄거리|기타",
        "chunk_ids": "c1|c2",
        "page_hit_within_k": "true",
        "section_hit_within_k": "true",
        "label_context_answerability": "FULLY_ANSWERABLE",
        "context_flags": "",
        "notes": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Bundle hit helpers + text rendering
# ---------------------------------------------------------------------------


class TestBundleHitHelpers:
    def test_bundle_page_hit_when_any_chunk_matches(self):
        retrieved = [
            RetrievedRef(rank=1, page_id="other", chunk_text="t"),
            RetrievedRef(rank=2, page_id="p_gold", chunk_text="t"),
        ]
        assert compute_bundle_page_hit("p_gold", retrieved) is True

    def test_bundle_page_hit_false_when_none_match(self):
        retrieved = [
            RetrievedRef(rank=1, page_id="other", chunk_text="t"),
        ]
        assert compute_bundle_page_hit("p_gold", retrieved) is False

    def test_bundle_section_hit_requires_page_hit(self):
        retrieved = [
            RetrievedRef(
                rank=1, page_id="other", section_path="줄거리",
                chunk_text="t",
            ),
        ]
        assert compute_bundle_section_hit(
            "p_gold", "줄거리", retrieved,
        ) is False

    def test_bundle_section_hit_prefix_match(self):
        retrieved = [
            RetrievedRef(
                rank=1, page_id="p_gold",
                section_path="등장인물 > 주인공",
                chunk_text="t",
            ),
        ]
        assert compute_bundle_section_hit(
            "p_gold", "등장인물", retrieved,
        ) is True

    def test_bundle_section_hit_does_not_match_sibling_prefix(self):
        retrieved = [
            RetrievedRef(
                rank=1, page_id="p_gold",
                section_path="등장인물상",
                chunk_text="t",
            ),
        ]
        assert compute_bundle_section_hit(
            "p_gold", "등장인물", retrieved,
        ) is False


class TestRenderBundleText:
    def test_renders_blocks_in_rank_order(self):
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1",
                page_id="p1", page_title="P1",
                section_path="줄거리",
                chunk_text="줄거리 본문 내용",
            ),
            RetrievedRef(
                rank=2, chunk_id="c2",
                page_id="p2", page_title="P2",
                section_path="설정",
                chunk_text="설정 본문 내용",
            ),
        ]
        text = render_bundle_text(retrieved)
        # Pinned layout markers — the labelling format is part of the
        # human contract and a regression here would silently change
        # how reviewers see chunks.
        assert "[Rank 1]" in text
        assert "[Rank 2]" in text
        assert "page: P1 (p1)" in text
        assert "page: P2 (p2)" in text
        assert "section: 줄거리" in text
        assert "section: 설정" in text
        assert "chunk_id: c1" in text
        assert "chunk_id: c2" in text
        assert "줄거리 본문 내용" in text
        assert "설정 본문 내용" in text

    def test_truncation_marker_appears(self):
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1",
                chunk_text="A" * 1500,
            ),
        ]
        text = render_bundle_text(retrieved, truncate_chars=1200)
        # The 1200 cap leaves "...[truncated]" appended.
        assert text.count("A") == 1200
        assert "...[truncated]" in text

    def test_no_truncation_when_disabled(self):
        retrieved = [
            RetrievedRef(rank=1, chunk_text="A" * 200),
        ]
        text = render_bundle_text(retrieved, truncate_chars=0)
        assert "...[truncated]" not in text
        assert text.count("A") == 200

    def test_no_truncation_when_under_cap(self):
        retrieved = [
            RetrievedRef(rank=1, chunk_text="short text"),
        ]
        text = render_bundle_text(retrieved, truncate_chars=1200)
        assert "...[truncated]" not in text
        assert "short text" in text

    def test_empty_retrieved_returns_empty_string(self):
        assert render_bundle_text([]) == ""


# ---------------------------------------------------------------------------
# Bundle export builder (top_k=1,3,5)
# ---------------------------------------------------------------------------


class TestBuildBundleExportRows:
    def test_builds_one_row_per_top_k(self):
        gold = GoldRef(page_id="p1", section_path="줄거리")
        retrieved = [
            RetrievedRef(
                rank=i, chunk_id=f"c{i}", page_id="p1",
                section_path="줄거리", chunk_text=f"text-{i}",
            )
            for i in range(1, 8)
        ]
        rows = build_bundle_export_rows(
            query_id="q1", query="hello",
            variant_name="baseline", gold=gold,
            retrieved=retrieved,
        )
        # Default top_k_set = (1, 3, 5).
        assert sorted(r.top_k for r in rows) == [1, 3, 5]
        assert all(r.variant_name == "baseline" for r in rows)
        # The k=1 bundle has 1 chunk; k=3 has 3; k=5 has 5.
        for r in rows:
            assert len(r.retrieved) == r.top_k

    def test_custom_top_k_set(self):
        gold = GoldRef()
        retrieved = [RetrievedRef(rank=i, chunk_text="t") for i in range(1, 11)]
        rows = build_bundle_export_rows(
            query_id="q1", query="hi",
            variant_name="v1", gold=gold,
            retrieved=retrieved,
            top_k_set=(2, 7),
        )
        assert sorted(r.top_k for r in rows) == [2, 7]

    def test_rejects_non_positive_top_k(self):
        gold = GoldRef()
        retrieved = [RetrievedRef(rank=1, chunk_text="t")]
        with pytest.raises(ValueError):
            build_bundle_export_rows(
                query_id="q1", query="hi",
                variant_name="v", gold=gold,
                retrieved=retrieved,
                top_k_set=(0, 5),
            )
        with pytest.raises(ValueError):
            build_bundle_export_rows(
                query_id="q1", query="hi",
                variant_name="v", gold=gold,
                retrieved=retrieved,
                top_k_set=(-1, 5),
            )

    def test_hit_within_k_is_set_correctly(self):
        gold = GoldRef(page_id="p_gold", section_path="줄거리")
        retrieved = [
            RetrievedRef(rank=1, page_id="other", section_path="기타", chunk_text="t"),
            RetrievedRef(
                rank=2, page_id="p_gold", section_path="줄거리",
                chunk_text="t",
            ),
        ]
        rows = build_bundle_export_rows(
            query_id="q1", query="hi",
            variant_name="v", gold=gold,
            retrieved=retrieved,
            top_k_set=(1, 2),
        )
        by_k = {r.top_k: r for r in rows}
        # k=1 bundle only sees the wrong page.
        assert by_k[1].page_hit_within_k is False
        assert by_k[1].section_hit_within_k is False
        # k=2 bundle includes the gold-hit chunk.
        assert by_k[2].page_hit_within_k is True
        assert by_k[2].section_hit_within_k is True

    def test_export_row_has_blank_label_columns_in_csv_dict(self):
        gold = GoldRef()
        retrieved = [RetrievedRef(rank=1, chunk_text="t")]
        rows = build_bundle_export_rows(
            query_id="q1", query="hi",
            variant_name="v", gold=gold,
            retrieved=retrieved,
            top_k_set=(1,),
        )
        d = rows[0].to_csv_dict()
        assert d["label_context_answerability"] == ""
        assert d["context_flags"] == ""
        assert d["notes"] == ""
        # Column set matches the pinned schema.
        assert set(d.keys()) == set(BUNDLE_EXPORT_COLUMNS)


class TestBundleExportRoundTrip:
    def test_csv_write_then_label_then_read(self, tmp_path: Path):
        gold = GoldRef(page_id="p1", section_path="줄거리")
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1", page_id="p1",
                section_path="줄거리", chunk_text="text-1",
            ),
            RetrievedRef(
                rank=2, chunk_id="c2", page_id="p2",
                section_path="설정", chunk_text="text-2",
            ),
        ]
        rows = build_bundle_export_rows(
            query_id="q1", query="hello",
            variant_name="baseline", gold=gold,
            retrieved=retrieved, top_k_set=(1, 3, 5),
        )
        csv_path = tmp_path / "bundle.csv"
        write_bundle_export_csv(csv_path, rows)

        # Reviewer fills label cells.
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        for r in records:
            r["label_context_answerability"] = "PARTIALLY_ANSWERABLE"
            r["context_flags"] = "needs_subpage"
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            )
            writer.writeheader()
            writer.writerows(records)

        labelled = read_bundle_labeled_csv(csv_path)
        assert len(labelled) == 3
        assert all(
            r.label is AnswerabilityLabel.PARTIALLY_ANSWERABLE
            for r in labelled
        )
        assert all(
            r.flags == (AnswerabilityFlag.NEEDS_SUBPAGE,)
            for r in labelled
        )

    def test_jsonl_round_trip(self, tmp_path: Path):
        gold = GoldRef(page_id="p1", section_path="줄거리")
        retrieved = [
            RetrievedRef(
                rank=1, chunk_id="c1", page_id="p1",
                section_path="줄거리", chunk_text="text",
            ),
        ]
        rows = build_bundle_export_rows(
            query_id="q1", query="hi",
            variant_name="baseline", gold=gold,
            retrieved=retrieved, top_k_set=(1,),
        )
        jsonl_path = tmp_path / "bundle.jsonl"
        write_bundle_export_jsonl(jsonl_path, rows)

        # Reviewer fills the label inline (mix int and list flag form).
        out = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            obj["label_context_answerability"] = 3
            obj["context_flags"] = ["evidence_too_noisy"]
            out.append(json.dumps(obj, ensure_ascii=False))
        jsonl_path.write_text("\n".join(out) + "\n", encoding="utf-8")

        labelled = read_bundle_labeled_jsonl(jsonl_path)
        assert len(labelled) == 1
        assert labelled[0].label is AnswerabilityLabel.FULLY_ANSWERABLE
        assert labelled[0].flags == (
            AnswerabilityFlag.EVIDENCE_TOO_NOISY,
        )


# ---------------------------------------------------------------------------
# Bundle import + validation
# ---------------------------------------------------------------------------


class TestParseBundleLabeledRow:
    def test_round_trips_int_label(self):
        row = parse_bundle_labeled_row(
            _bundle_csv_record(label_context_answerability="2"),
        )
        assert row.label is AnswerabilityLabel.PARTIALLY_ANSWERABLE
        assert row.top_k == 5

    def test_string_list_columns_split_on_pipe(self):
        row = parse_bundle_labeled_row(
            _bundle_csv_record(
                retrieved_page_ids="a|b|c",
                chunk_ids="c1|c2|c3",
            ),
        )
        assert row.retrieved_page_ids == ("a", "b", "c")
        assert row.chunk_ids == ("c1", "c2", "c3")

    def test_empty_string_list_returns_empty_tuple(self):
        row = parse_bundle_labeled_row(
            _bundle_csv_record(
                retrieved_page_ids="",
                chunk_ids="",
            ),
        )
        assert row.retrieved_page_ids == ()
        assert row.chunk_ids == ()

    def test_rejects_empty_bundle_text(self):
        with pytest.raises(AnswerabilityValidationError) as ex:
            parse_bundle_labeled_row(
                _bundle_csv_record(context_bundle_text=""),
            )
        assert "context_bundle_text" in str(ex.value)

    def test_rejects_zero_top_k(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(_bundle_csv_record(top_k="0"))

    def test_rejects_negative_top_k(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(_bundle_csv_record(top_k="-3"))

    def test_rejects_non_int_top_k(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(
                _bundle_csv_record(top_k="not_a_number"),
            )

    def test_rejects_bool_top_k(self):
        # bool is an int subclass; explicit guard prevents True/False
        # silently masquerading as top_k=1/0.
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(_bundle_csv_record(top_k=True))

    def test_rejects_invalid_context_label(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(
                _bundle_csv_record(label_context_answerability="WAT"),
            )

    def test_rejects_invalid_context_flag(self):
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(
                _bundle_csv_record(context_flags="not_a_real_flag"),
            )

    def test_rejects_missing_required_column(self):
        rec = _bundle_csv_record()
        rec.pop("variant_name")
        with pytest.raises(AnswerabilityValidationError):
            parse_bundle_labeled_row(rec)


class TestParseBundleLabeledRowsDuplicates:
    def test_accepts_unique_keys(self):
        rows = parse_bundle_labeled_rows([
            _bundle_csv_record(query_id="q1", top_k="1"),
            _bundle_csv_record(query_id="q1", top_k="3"),
            _bundle_csv_record(query_id="q1", top_k="5"),
            _bundle_csv_record(query_id="q2", top_k="5"),
        ])
        assert len(rows) == 4

    def test_rejects_duplicate(self):
        with pytest.raises(AnswerabilityValidationError) as ex:
            parse_bundle_labeled_rows([
                _bundle_csv_record(query_id="q1", top_k="5"),
                _bundle_csv_record(query_id="q1", top_k="5"),
            ])
        assert "Duplicate" in str(ex.value)

    def test_distinct_variants_share_keys(self):
        rows = parse_bundle_labeled_rows([
            _bundle_csv_record(query_id="q1", variant_name="v1", top_k="5"),
            _bundle_csv_record(query_id="q1", variant_name="v2", top_k="5"),
        ])
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Bundle scoring
# ---------------------------------------------------------------------------


class TestContextAnswerableAtK:
    def test_zero_when_no_top_k_row(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        # No top_k=3 or top_k=5 row → those at_k are 0.
        assert m.context_answerable_at_1 == 0.0
        assert m.context_answerable_at_3 == 0.0
        assert m.context_answerable_at_5 == 0.0

    def test_partial_counts_as_answerable(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=3,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_answerable_at_1 == 0.0
        assert m.context_answerable_at_3 == 1.0
        # No top_k=5 row → 0 at @5 (matches "no top-k row means no hit").
        assert m.context_answerable_at_5 == 0.0

    def test_query_level_aggregation(self):
        # Two queries; q1 answers at top_k=5, q2 does not.
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
            _bundle_row(
                query_id="q2", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_answerable_at_5 == 0.5

    def test_only_top_k_5_row_counts_for_at_5(self):
        # If a query only has top_k=1 rows, context_answerable_at_5
        # is 0 — the "@5" question is genuinely unanswered.
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_answerable_at_1 == 1.0
        assert m.context_answerable_at_5 == 0.0


class TestContextFullyAnswerableAtK:
    def test_partial_does_not_count_as_fully(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_fully_answerable_at_5 == 0.0
        assert m.context_answerable_at_5 == 1.0

    def test_fully_at_each_k(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
            ),
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=3,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_fully_answerable_at_1 == 0.0
        assert m.context_fully_answerable_at_3 == 0.0
        assert m.context_fully_answerable_at_5 == 1.0


class TestPageHitButContextNotAnswerableAt5:
    def test_counts_query_with_page_hit_but_no_answerable_at_5(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.NOT_RELEVANT,
                page_hit=True, section_hit=False,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.page_hit_but_context_not_answerable_at_5 == 1

    def test_does_not_count_when_answerable(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.page_hit_but_context_not_answerable_at_5 == 0


class TestSectionMissButContextAnswerableAt5:
    def test_counts_when_no_section_hit_but_answerable(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=False,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.section_miss_but_context_answerable_at_5 == 1

    def test_does_not_count_when_section_hits(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.section_miss_but_context_answerable_at_5 == 0


class TestBundleFlagCounts:
    def test_only_top_k_5_rows_contribute(self):
        rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                flags=(AnswerabilityFlag.NEEDS_CROSS_SECTION,),
            ),
            _bundle_row(
                query_id="q2", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                flags=(AnswerabilityFlag.NEEDS_SUBPAGE,),
            ),
            # top_k=3 rows are excluded from flag counts.
            _bundle_row(
                query_id="q3", variant_name="baseline", top_k=3,
                label=AnswerabilityLabel.NOT_RELEVANT,
                flags=(AnswerabilityFlag.WRONG_PAGE,),
            ),
            _bundle_row(
                query_id="q4", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.NOT_RELEVANT,
                flags=(
                    AnswerabilityFlag.EVIDENCE_TOO_NOISY,
                    AnswerabilityFlag.WRONG_PAGE,
                ),
            ),
        ]
        m = compute_bundle_variant_metrics(rows, "baseline")
        assert m.context_needs_cross_section_count == 1
        assert m.context_needs_subpage_count == 1
        assert m.context_evidence_too_noisy_count == 1
        # WRONG_PAGE on the top_k=3 row is excluded; only the top_k=5
        # one counts.
        assert m.context_wrong_page_count == 1
        assert m.context_right_page_wrong_section_count == 0


class TestBundleVariantComparison:
    def test_compute_all_bundle_variants_orders_known_first(self):
        rows = []
        for v in [
            "z_extra",
            VARIANT_SECTION_AWARE_CANDIDATE,
            VARIANT_BASELINE,
            VARIANT_PRODUCTION_RECOMMENDED,
        ]:
            rows.append(_bundle_row(
                query_id=f"q-{v}", variant_name=v, top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ))
        result = compute_all_bundle_variants(rows)
        names = [v.variant_name for v in result]
        assert names == [
            VARIANT_BASELINE,
            VARIANT_PRODUCTION_RECOMMENDED,
            VARIANT_SECTION_AWARE_CANDIDATE,
            "z_extra",
        ]


# ---------------------------------------------------------------------------
# Markdown report — bundle sections appear when bundle_rows non-empty
# ---------------------------------------------------------------------------


class TestRenderMarkdownReportWithBundle:
    def _representative_rows(self) -> List[AnswerabilityLabeledRow]:
        return [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.NOT_RELEVANT,
                page_hit=False, section_hit=False,
                flags=(AnswerabilityFlag.WRONG_PAGE,),
            ),
            _row(
                query_id="q1",
                variant_name="production_recommended", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
        ]

    def _representative_bundle_rows(self) -> List[ContextBundleAuditRow]:
        return [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.NOT_RELEVANT,
                page_hit=False, section_hit=False,
                flags=(AnswerabilityFlag.WRONG_PAGE,),
            ),
            _bundle_row(
                query_id="q1",
                variant_name="production_recommended", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
                page_hit=True, section_hit=True,
            ),
            _bundle_row(
                query_id="q1",
                variant_name="production_recommended", top_k=1,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
                page_hit=True, section_hit=False,
                flags=(AnswerabilityFlag.NEEDS_SUBPAGE,),
            ),
        ]

    def test_no_bundle_sections_when_bundle_rows_empty(self):
        md = render_markdown_report(self._representative_rows())
        assert "## Context bundle answerability" not in md
        assert "## Multi-chunk answerability caveat" not in md
        assert "## Recommended labeling workflow" not in md

    def test_bundle_sections_inserted_when_bundle_rows_supplied(self):
        md = render_markdown_report(
            self._representative_rows(),
            bundle_rows=self._representative_bundle_rows(),
        )
        for header in [
            "## Context bundle answerability",
            "## Row evidence vs context answerability",
            "## Multi-chunk answerability caveat",
            "## Recommended labeling workflow",
        ]:
            assert header in md

    def test_caveat_section_states_complementary_not_replacement(self):
        md = render_markdown_report(
            self._representative_rows(),
            bundle_rows=self._representative_bundle_rows(),
        )
        # Pinned wording — both must appear so reviewers cannot miss
        # the "additive, not a replacement" framing.
        assert "complementary" in md.lower()
        assert "hit@k" in md.lower()
        assert "not a hit@k replacement" in md.lower()

    def test_workflow_lists_six_steps(self):
        md = render_markdown_report(
            self._representative_rows(),
            bundle_rows=self._representative_bundle_rows(),
        )
        # Pinned step markers.
        for step in ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"]:
            assert step in md

    def test_existing_row_sections_still_present(self):
        # Regression guard: the row-level sections must not vanish
        # when bundle rows are added.
        md = render_markdown_report(
            self._representative_rows(),
            bundle_rows=self._representative_bundle_rows(),
        )
        for header in [
            "## Summary",
            "## Label distribution",
            "## Answerability metrics by variant",
            "## Page hit vs answerability confusion",
            "## Section hit vs answerability confusion",
            "## Failure buckets",
            "## Top failure examples",
            "## Interpretation guide",
            "## Next action recommendation",
        ]:
            assert header in md

    def test_bundle_only_input_renders_row_placeholders(self):
        md = render_markdown_report(
            [],
            bundle_rows=self._representative_bundle_rows(),
        )
        # Row sections degrade to "no variants" placeholders rather
        # than crashing.
        assert "## Answerability metrics by variant" in md
        assert "no variants" in md
        # Bundle sections present.
        assert "## Context bundle answerability" in md


class TestBuildJsonSummaryWithBundle:
    def test_no_bundle_keys_when_bundle_rows_empty(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        summary = build_json_summary(rows)
        assert "bundle_summary" not in summary
        assert "bundle_variants" not in summary

    def test_bundle_keys_added_when_supplied(self):
        rows = [
            _row(
                query_id="q1", variant_name="baseline", rank=1,
                label=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            ),
        ]
        bundle_rows = [
            _bundle_row(
                query_id="q1", variant_name="baseline", top_k=5,
                label=AnswerabilityLabel.FULLY_ANSWERABLE,
            ),
        ]
        summary = build_json_summary(rows, bundle_rows=bundle_rows)
        assert "bundle_summary" in summary
        assert "bundle_label_distribution" in summary
        assert "bundle_variants" in summary
        assert "row_vs_bundle_at_5" in summary
        assert summary["bundle_summary"]["n_rows"] == 1
        assert summary["bundle_summary"]["top_k_set"] == [5]


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _sample_records(
    n_queries: int, top_k: int = 5, variant: str = "baseline",
) -> List[Dict[str, Any]]:
    """Deterministic synthetic record generator for sampler tests."""
    return [
        {
            "query_id": f"q{i:03d}",
            "query": f"q for {i}",
            "variant_name": variant,
            "top_k": str(top_k),
            "context_bundle_text": f"bundle text {i}",
            "label_context_answerability": "",  # unlabelled!
            "context_flags": "",
            "notes": "",
        }
        for i in range(n_queries)
    ]


class TestSampleBundleRecords:
    def test_samples_n_distinct_query_ids(self):
        records = _sample_records(50)
        sampled = sample_bundle_records(records, n_queries=10, seed=42)
        # 10 distinct query_ids selected; all sampled rows belong to
        # those 10 queries.
        qids = {r["query_id"] for r in sampled}
        assert len(qids) == 10
        # Each query had exactly 1 row in the input → 10 sampled rows.
        assert len(sampled) == 10

    def test_deterministic_with_same_seed(self):
        records = _sample_records(50)
        a = sample_bundle_records(records, n_queries=10, seed=42)
        b = sample_bundle_records(records, n_queries=10, seed=42)
        assert [r["query_id"] for r in a] == [
            r["query_id"] for r in b
        ]

    def test_different_seed_gives_different_sample(self):
        records = _sample_records(50)
        a = sample_bundle_records(records, n_queries=10, seed=42)
        b = sample_bundle_records(records, n_queries=10, seed=99)
        # Likely different — the sets may overlap but cannot be
        # identical with high probability over 50/10 selection.
        a_qids = {r["query_id"] for r in a}
        b_qids = {r["query_id"] for r in b}
        assert a_qids != b_qids

    def test_filter_by_variant(self):
        records = _sample_records(20, variant="baseline")
        records += _sample_records(20, variant="other")
        # Re-key the second batch's query ids so they don't clash.
        for i, r in enumerate(records[20:], start=100):
            r["query_id"] = f"o{i:03d}"
        sampled = sample_bundle_records(
            records, n_queries=5, seed=42,
            variant_name="baseline",
        )
        assert all(r["variant_name"] == "baseline" for r in sampled)

    def test_filter_by_top_k(self):
        records = _sample_records(10, top_k=1)
        records += _sample_records(10, top_k=5)
        # Re-key second batch to keep query_ids unique.
        for i, r in enumerate(records[10:], start=200):
            r["query_id"] = f"k5_{i:03d}"
        sampled = sample_bundle_records(
            records, n_queries=5, seed=42, top_k=5,
        )
        assert all(int(r["top_k"]) == 5 for r in sampled)

    def test_returns_all_top_k_rows_for_sampled_queries(self):
        # Each query has 3 rows (top_k=1,3,5). Without --top-k filter,
        # the sampler should return every row for every sampled query.
        records = []
        for i in range(20):
            qid = f"q{i:03d}"
            for k in (1, 3, 5):
                records.append({
                    "query_id": qid,
                    "query": "q",
                    "variant_name": "baseline",
                    "top_k": str(k),
                    "context_bundle_text": f"bundle {qid} k{k}",
                    "label_context_answerability": "",
                    "context_flags": "",
                    "notes": "",
                })
        sampled = sample_bundle_records(records, n_queries=5, seed=42)
        sampled_qids = {r["query_id"] for r in sampled}
        # 5 queries × 3 rows each = 15 rows.
        assert len(sampled_qids) == 5
        assert len(sampled) == 15

    def test_sampler_does_not_fill_labels(self):
        # Critical contract: sampler must never modify label cells.
        records = _sample_records(20)
        sampled = sample_bundle_records(records, n_queries=5, seed=42)
        assert all(
            r.get("label_context_answerability", "") == ""
            for r in sampled
        )
        assert all(
            r.get("context_flags", "") == "" for r in sampled
        )
        assert all(
            r.get("notes", "") == "" for r in sampled
        )

    def test_n_larger_than_population(self):
        # Sampling more queries than exist returns the full set.
        records = _sample_records(3)
        sampled = sample_bundle_records(records, n_queries=99, seed=42)
        assert len({r["query_id"] for r in sampled}) == 3

    def test_zero_n_returns_empty(self):
        records = _sample_records(10)
        assert sample_bundle_records(
            records, n_queries=0, seed=42,
        ) == []

    def test_input_order_independence(self):
        # Shuffle input → same sampled query_ids (selection is on
        # the sorted set of unique qids, not input order).
        records_a = _sample_records(50)
        records_b = list(reversed(records_a))
        a = sample_bundle_records(records_a, n_queries=10, seed=42)
        b = sample_bundle_records(records_b, n_queries=10, seed=42)
        assert (
            sorted(r["query_id"] for r in a)
            == sorted(r["query_id"] for r in b)
        )


# ---------------------------------------------------------------------------
# CLI tests — bundle export + bundle-sample + combined score
# ---------------------------------------------------------------------------


class TestExportBundleCli:
    def test_export_bundle_csv_end_to_end(self, tmp_path: Path):
        from scripts.export_answerability_audit import main as export_main

        in_path = tmp_path / "ret.jsonl"
        in_payload = {
            "query_id": "q1",
            "query": "q?",
            "gold": {
                "page_id": "p1", "page_title": "P1",
                "section_id": "", "section_path": "줄거리",
            },
            "results": [
                {
                    "rank": i, "chunk_id": f"c{i}",
                    "page_id": "p1", "section_path": "줄거리",
                    "page_title": "P1",
                    "chunk_text": f"text-{i}",
                }
                for i in range(1, 8)
            ],
        }
        in_path.write_text(
            json.dumps(in_payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        out_path = tmp_path / "bundle.csv"
        rc = export_main([
            "--mode", "bundle",
            "--retrieval-results-path", str(in_path),
            "--out-path", str(out_path),
            "--variant-name", "baseline",
            "--top-k-set", "1,3,5",
        ])
        assert rc == 0
        assert out_path.exists()
        with out_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        # 3 bundle rows for one query (k=1,3,5).
        assert len(records) == 3
        assert sorted(int(r["top_k"]) for r in records) == [1, 3, 5]
        # All label cells are blank.
        assert all(
            r["label_context_answerability"] == "" for r in records
        )

    def test_export_bundle_default_top_k_set(self, tmp_path: Path):
        from scripts.export_answerability_audit import main as export_main

        in_path = tmp_path / "ret.jsonl"
        in_path.write_text(
            json.dumps({
                "query_id": "q1", "query": "q",
                "gold": {"page_id": "p1", "section_path": "x"},
                "results": [
                    {"rank": i, "chunk_text": f"t{i}"} for i in range(1, 6)
                ],
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "bundle.csv"
        rc = export_main([
            "--mode", "bundle",
            "--retrieval-results-path", str(in_path),
            "--out-path", str(out_path),
            "--variant-name", "baseline",
        ])
        assert rc == 0
        with out_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        assert sorted(int(r["top_k"]) for r in records) == [1, 3, 5]


class TestExportBundleSampleCli:
    def test_bundle_sample_picks_n_queries(self, tmp_path: Path):
        from scripts.export_answerability_audit import main as export_main

        # Build a synthetic bundle CSV with 50 queries × 1 row each.
        bundle_csv = tmp_path / "bundle.csv"
        with bundle_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            )
            writer.writeheader()
            for i in range(50):
                writer.writerow({
                    col: "" for col in BUNDLE_EXPORT_COLUMNS
                } | {
                    "query_id": f"q{i:03d}",
                    "query": "q",
                    "variant_name": "production_recommended",
                    "top_k": "5",
                    "context_bundle_text": f"text {i}",
                    "page_hit_within_k": "true",
                    "section_hit_within_k": "false",
                })

        out_path = tmp_path / "sample.csv"
        rc = export_main([
            "--mode", "bundle-sample",
            "--input-path", str(bundle_csv),
            "--out-path", str(out_path),
            "--variant-name", "production_recommended",
            "--sample-query-count", "10",
            "--seed", "42",
        ])
        assert rc == 0
        assert out_path.exists()

        with out_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        assert len({r["query_id"] for r in records}) == 10
        # Sample CLI must NOT inject label cells.
        assert all(
            r["label_context_answerability"] == "" for r in records
        )

    def test_bundle_sample_seed_deterministic(self, tmp_path: Path):
        from scripts.export_answerability_audit import main as export_main

        bundle_csv = tmp_path / "bundle.csv"
        with bundle_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            )
            writer.writeheader()
            for i in range(30):
                writer.writerow({
                    col: "" for col in BUNDLE_EXPORT_COLUMNS
                } | {
                    "query_id": f"q{i:03d}",
                    "query": "q",
                    "variant_name": "v1",
                    "top_k": "5",
                    "context_bundle_text": f"text {i}",
                })

        out_a = tmp_path / "a.csv"
        out_b = tmp_path / "b.csv"
        for out in (out_a, out_b):
            rc = export_main([
                "--mode", "bundle-sample",
                "--input-path", str(bundle_csv),
                "--out-path", str(out),
                "--sample-query-count", "5",
                "--seed", "42",
            ])
            assert rc == 0
        assert out_a.read_text(encoding="utf-8") == (
            out_b.read_text(encoding="utf-8")
        )


class TestExportRowSampleCli:
    """Phase 7.7 row-level sampling CLI smoke tests.

    Mirrors :class:`TestExportBundleSampleCli` but covers the row-mode
    sampler used by the pilot labelling step. Critical contract: the
    sampler keeps every rank for each chosen query_id, never injects
    label / flag / notes values, and is deterministic at fixed seed.
    """

    def _write_row_csv(
        self, path: Path, *, n_queries: int, top_k: int = 5,
        variant_name: str = "production_recommended",
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(EXPORT_COLUMNS),
            )
            writer.writeheader()
            for q in range(n_queries):
                qid = f"q{q:03d}"
                for rank in range(1, top_k + 1):
                    writer.writerow({
                        col: "" for col in EXPORT_COLUMNS
                    } | {
                        "query_id": qid,
                        "query": "q",
                        "variant_name": variant_name,
                        "rank": str(rank),
                        "retrieved_page_id": f"page_{q}",
                        "retrieved_page_title": f"title {q}",
                        "retrieved_section_path": "개요",
                        "chunk_id": f"c_{q}_{rank}",
                        "chunk_text": f"chunk text {q}/{rank}",
                        "page_hit": "true",
                        "section_hit": "false",
                    })

    def test_row_sample_picks_n_queries_and_keeps_every_rank(
        self, tmp_path: Path,
    ):
        from scripts.export_answerability_audit import main as export_main

        row_csv = tmp_path / "row.csv"
        self._write_row_csv(row_csv, n_queries=50, top_k=5)

        out_path = tmp_path / "sample.csv"
        rc = export_main([
            "--mode", "row-sample",
            "--input-path", str(row_csv),
            "--out-path", str(out_path),
            "--variant-name", "production_recommended",
            "--sample-query-count", "10",
            "--seed", "42",
        ])
        assert rc == 0
        assert out_path.exists()

        with out_path.open("r", encoding="utf-8", newline="") as fp:
            records = list(csv.DictReader(fp))
        # 10 queries × 5 ranks = 50 rows; every chosen qid must keep
        # rank 1..5.
        assert len(records) == 50
        per_q_ranks: Dict[str, List[int]] = {}
        for r in records:
            per_q_ranks.setdefault(r["query_id"], []).append(
                int(r["rank"]),
            )
        assert len(per_q_ranks) == 10
        assert all(
            sorted(v) == [1, 2, 3, 4, 5] for v in per_q_ranks.values()
        )
        # Sampler MUST NOT fill label cells.
        assert all(r["label_answerability"] == "" for r in records)
        assert all(r["flags"] == "" for r in records)
        assert all(r["notes"] == "" for r in records)
        # CSV header must be the row schema, not the bundle schema.
        with out_path.open("r", encoding="utf-8") as fp:
            header = fp.readline().rstrip("\r\n").split(",")
        assert header == list(EXPORT_COLUMNS)

    def test_row_sample_seed_deterministic(self, tmp_path: Path):
        from scripts.export_answerability_audit import main as export_main

        row_csv = tmp_path / "row.csv"
        self._write_row_csv(row_csv, n_queries=30, top_k=3)

        out_a = tmp_path / "a.csv"
        out_b = tmp_path / "b.csv"
        for out in (out_a, out_b):
            rc = export_main([
                "--mode", "row-sample",
                "--input-path", str(row_csv),
                "--out-path", str(out),
                "--sample-query-count", "5",
                "--seed", "42",
            ])
            assert rc == 0
        assert out_a.read_text(encoding="utf-8") == (
            out_b.read_text(encoding="utf-8")
        )


class TestScoreCombinedCli:
    def test_combined_mode_renders_both_tracks(self, tmp_path: Path):
        from scripts.score_answerability_audit import main as score_main

        # Row CSV.
        row_csv = tmp_path / "row.csv"
        with row_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(EXPORT_COLUMNS),
            )
            writer.writeheader()
            writer.writerow(_csv_record(
                query_id="q1", variant_name="baseline", rank="1",
                label_answerability="FULLY_ANSWERABLE",
                page_hit="true", section_hit="true",
            ))

        # Bundle CSV.
        bundle_csv = tmp_path / "bundle.csv"
        with bundle_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            )
            writer.writeheader()
            writer.writerow(_bundle_csv_record(
                query_id="q1", variant_name="baseline", top_k="5",
                label_context_answerability="FULLY_ANSWERABLE",
            ))

        report_md = tmp_path / "combined.md"
        json_out = tmp_path / "combined.json"
        rc = score_main([
            "--mode", "combined",
            "--labeled-row-path", str(row_csv),
            "--labeled-bundle-path", str(bundle_csv),
            "--report-path", str(report_md),
            "--json-path", str(json_out),
        ])
        assert rc == 0
        text = report_md.read_text(encoding="utf-8")
        # Both tracks rendered.
        assert "## Answerability metrics by variant" in text
        assert "## Context bundle answerability" in text
        assert "## Recommended labeling workflow" in text

        summary = json.loads(json_out.read_text(encoding="utf-8"))
        assert "variants" in summary
        assert "bundle_variants" in summary
        assert "row_vs_bundle_at_5" in summary

    def test_score_bundle_only_mode(self, tmp_path: Path):
        from scripts.score_answerability_audit import main as score_main

        bundle_csv = tmp_path / "bundle.csv"
        with bundle_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            )
            writer.writeheader()
            writer.writerow(_bundle_csv_record(
                query_id="q1", variant_name="baseline", top_k="5",
                label_context_answerability="PARTIALLY_ANSWERABLE",
            ))

        report_md = tmp_path / "bundle.md"
        rc = score_main([
            "--mode", "bundle",
            "--labeled-bundle-path", str(bundle_csv),
            "--report-path", str(report_md),
        ])
        assert rc == 0
        text = report_md.read_text(encoding="utf-8")
        assert "## Context bundle answerability" in text

    def test_score_combined_requires_both_paths(self, tmp_path: Path):
        from scripts.score_answerability_audit import main as score_main

        report_md = tmp_path / "x.md"
        with pytest.raises(SystemExit):
            score_main([
                "--mode", "combined",
                "--report-path", str(report_md),
                # neither row nor bundle path given
            ])


# ---------------------------------------------------------------------------
# v4 canonical schema alignment for the export ingestion helpers
# ---------------------------------------------------------------------------


class TestV4InputAlignment:
    """v4 ``chunks_v4.jsonl`` / ``rag_chunks.jsonl`` and the production
    retrieval emitter all carry ``section_path`` as a list of segments
    and store the raw chunk text under either ``chunk_text`` (rag_chunks
    form) or ``text`` (chunks_v4 form). The export ingestion helpers
    must accept both shapes so a labelling file built from any v4
    fixture renders a single human-readable string per row."""

    def test_section_path_list_is_joined(self):
        from scripts.export_answerability_audit import (
            SECTION_PATH_JOINER,
            _gold_from_record,
            _retrieved_from_record,
        )

        gold = _gold_from_record({
            "gold": {
                "page_id": "p1",
                "section_path": ["음악", "주제가", "OP"],
            },
        })
        assert gold.section_path == SECTION_PATH_JOINER.join(
            ["음악", "주제가", "OP"]
        )

        retrieved = _retrieved_from_record({
            "rank": 1,
            "page_id": "p1",
            "section_path": ["음악", "주제가", "OP"],
            "chunk_text": "raw",
        })
        assert retrieved.section_path == "음악 > 주제가 > OP"

    def test_section_path_string_passes_through(self):
        from scripts.export_answerability_audit import (
            _gold_from_record,
            _retrieved_from_record,
        )

        gold = _gold_from_record({
            "gold": {
                "page_id": "p1",
                "section_path": "음악 > 주제가",
            },
        })
        assert gold.section_path == "음악 > 주제가"

        retrieved = _retrieved_from_record({
            "rank": 1,
            "page_id": "p1",
            "section_path": "음악 > 주제가",
            "chunk_text": "raw",
        })
        assert retrieved.section_path == "음악 > 주제가"

    def test_section_path_empty_or_missing(self):
        from scripts.export_answerability_audit import (
            _gold_from_record,
            _retrieved_from_record,
        )

        gold = _gold_from_record({"gold": {}})
        assert gold.section_path == ""

        gold_list_empty = _gold_from_record({
            "gold": {"section_path": []},
        })
        assert gold_list_empty.section_path == ""

        retrieved = _retrieved_from_record({
            "rank": 2,
            "chunk_text": "raw",
        })
        assert retrieved.section_path == ""

    def test_chunk_text_falls_back_to_text_field(self):
        """chunks_v4 stores the raw chunk under ``text``; rag_chunks
        stores it under ``chunk_text``. Either MUST be accepted."""
        from scripts.export_answerability_audit import (
            _resolve_chunk_text,
            _retrieved_from_record,
        )

        # rag_chunks form.
        assert _resolve_chunk_text({"chunk_text": "raw"}) == "raw"
        # chunks_v4 form.
        assert _resolve_chunk_text({"text": "raw"}) == "raw"
        # rag_chunks wins when both present.
        assert _resolve_chunk_text({
            "chunk_text": "rag", "text": "v4",
        }) == "rag"
        # Both empty → empty string (validator catches it later).
        assert _resolve_chunk_text({}) == ""

        retrieved = _retrieved_from_record({
            "rank": 1,
            "page_id": "p1",
            "section_path": ["개요"],
            "text": "from chunks_v4",
        })
        assert retrieved.chunk_text == "from chunks_v4"

    def test_chunk_text_does_not_pick_embedding_text(self):
        """Embedding-side fields would mislead the reviewer about what
        the retriever returned — they must NOT be substituted."""
        from scripts.export_answerability_audit import _resolve_chunk_text

        # Only embedding-side keys present → fall through to empty.
        assert _resolve_chunk_text({
            "text_for_embedding": "title: X\nsection: Y\nbody: ...",
        }) == ""
        assert _resolve_chunk_text({
            "embedding_text": "제목: X\n섹션: Y\n본문: ...",
        }) == ""
