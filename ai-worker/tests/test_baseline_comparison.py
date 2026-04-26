"""Tests for the retrieval baseline comparison helper.

Reads plain dict rows shaped like ``retrieval_eval_report.json``'s
``rows`` field — same shape the CLI emits — and produces a 3-slice
compare:

  - deterministic_all
  - deterministic_without_<excluded answer_type>
  - opus_all

Coverage:
  - excluded answer_type slice drops the right rows
  - per-axis breakdowns (answer_type / difficulty / language) are
    grouped consistently
  - empty rows produce None metrics rather than raising
  - metric formula matches a hand-computed mean
  - markdown renders all three slice headers
"""

from __future__ import annotations

import json
import math

import pytest

from eval.harness.baseline_comparison import (
    METRIC_KEYS,
    BaselineComparison,
    BaselineSlice,
    compute_baseline_slice,
    comparison_to_dict,
    render_comparison_markdown,
    run_comparison,
)


def _row(
    *,
    rid="r",
    answer_type="title_lookup",
    difficulty="medium",
    language="ko",
    hit5=1.0,
    hit1=1.0,
    hit3=1.0,
    mrr=1.0,
    ndcg=1.0,
    dup=0.0,
    udc=1.0,
    kwm=1.0,
    ctx=100.0,
    margin=0.05,
):
    return {
        "id": rid,
        "answer_type": answer_type,
        "difficulty": difficulty,
        "language": language,
        "expected_doc_ids": ["gold"],
        "expected_section_keywords": ["bookshop"],
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "mrr_at_10": mrr,
        "ndcg_at_10": ndcg,
        "dup_rate": dup,
        "unique_doc_coverage": udc,
        "expected_keyword_match_rate": kwm,
        "avg_context_token_count": ctx,
        "top1_score_margin": margin,
    }


class TestComputeBaselineSlice:
    def test_empty_rows_metrics_all_none(self):
        slc = compute_baseline_slice(
            label="empty", description="empty",
            dataset_path="x", rows=[],
        )
        assert slc.row_count == 0
        for k in METRIC_KEYS:
            assert slc.metrics[k] is None

    def test_mean_matches_hand_computation(self):
        rows = [
            _row(rid="a", hit5=1.0, ndcg=1.0),
            _row(rid="b", hit5=0.0, ndcg=0.0),
        ]
        slc = compute_baseline_slice(
            label="x", description="x",
            dataset_path=None, rows=rows,
        )
        assert slc.row_count == 2
        assert slc.metrics["hit_at_5"] == pytest.approx(0.5)
        assert slc.metrics["ndcg_at_10"] == pytest.approx(0.5)

    def test_exclude_answer_types_drops_those_rows(self):
        rows = [
            _row(rid="a", answer_type="title_lookup", hit5=1.0),
            _row(rid="b", answer_type="character_relation", hit5=0.0),
            _row(rid="c", answer_type="character_relation", hit5=0.0),
            _row(rid="d", answer_type="summary_plot", hit5=1.0),
        ]
        slc_all = compute_baseline_slice(
            label="all", description="all", dataset_path=None, rows=rows,
        )
        slc_no_cr = compute_baseline_slice(
            label="no-cr", description="no-cr",
            dataset_path=None, rows=rows,
            exclude_answer_types=("character_relation",),
        )
        assert slc_all.row_count == 4
        assert slc_no_cr.row_count == 2
        # all = mean(1,0,0,1) = 0.5
        assert slc_all.metrics["hit_at_5"] == pytest.approx(0.5)
        # no_cr = mean(1,1) = 1.0
        assert slc_no_cr.metrics["hit_at_5"] == pytest.approx(1.0)

    def test_per_axis_breakdown_groups_by_field(self):
        rows = [
            _row(rid="a", answer_type="title_lookup", hit5=1.0),
            _row(rid="b", answer_type="title_lookup", hit5=0.0),
            _row(rid="c", answer_type="summary_plot", hit5=1.0),
        ]
        slc = compute_baseline_slice(
            label="x", description="x", dataset_path=None, rows=rows,
        )
        assert "title_lookup" in slc.per_answer_type
        assert "summary_plot" in slc.per_answer_type
        assert slc.per_answer_type["title_lookup"]["row_count"] == 2
        assert slc.per_answer_type["title_lookup"]["mean_hit_at_5"] == pytest.approx(0.5)
        assert slc.per_answer_type["summary_plot"]["row_count"] == 1

    def test_skips_rows_with_none_field_for_mean(self):
        rows = [
            _row(rid="a", hit5=1.0),
            {**_row(rid="b"), "hit_at_5": None},
        ]
        slc = compute_baseline_slice(
            label="x", description="x", dataset_path=None, rows=rows,
        )
        # Only the one row with hit_at_5=1.0 contributes.
        assert slc.metrics["hit_at_5"] == pytest.approx(1.0)


class TestRunComparison:
    def test_three_slices_emitted(self):
        det = [
            _row(rid=f"d{i}", answer_type="title_lookup", hit5=1.0)
            for i in range(3)
        ] + [
            _row(rid=f"cr{i}", answer_type="character_relation", hit5=0.0)
            for i in range(2)
        ]
        opus = [_row(rid=f"o{i}", hit5=1.0) for i in range(4)]

        comparison = run_comparison(
            deterministic_rows=det,
            deterministic_dataset_path="det.jsonl",
            opus_rows=opus,
            opus_dataset_path="opus.jsonl",
        )
        labels = [s.label for s in comparison.slices]
        assert labels == [
            "deterministic_all",
            "deterministic_without_character_relation",
            "opus_all",
        ]
        # det_all hit_at_5 = mean(1,1,1,0,0) = 0.6
        det_all = comparison.slices[0]
        assert det_all.row_count == 5
        assert det_all.metrics["hit_at_5"] == pytest.approx(0.6)
        # det_no_cr = 1.0
        det_no_cr = comparison.slices[1]
        assert det_no_cr.row_count == 3
        assert det_no_cr.metrics["hit_at_5"] == pytest.approx(1.0)
        # opus_all = 1.0
        assert comparison.slices[2].metrics["hit_at_5"] == pytest.approx(1.0)

    def test_excluded_answer_type_overridable(self):
        det = [
            _row(rid="a", answer_type="title_lookup", hit5=1.0),
            _row(rid="b", answer_type="theme_genre", hit5=0.0),
        ]
        opus = [_row(rid="o", hit5=1.0)]
        comparison = run_comparison(
            deterministic_rows=det,
            deterministic_dataset_path=None,
            opus_rows=opus,
            opus_dataset_path=None,
            excluded_answer_type="theme_genre",
        )
        # Slice label reflects override.
        assert comparison.slices[1].label == "deterministic_without_theme_genre"
        assert comparison.slices[1].row_count == 1


class TestSerialization:
    def test_to_dict_round_trips_through_json(self):
        det = [_row(rid="a", hit5=1.0)]
        opus = [_row(rid="b", hit5=0.0)]
        comparison = run_comparison(
            deterministic_rows=det,
            deterministic_dataset_path=None,
            opus_rows=opus,
            opus_dataset_path=None,
        )
        payload = comparison_to_dict(comparison)
        json_str = json.dumps(payload, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded["metric_keys"]
        assert len(loaded["slices"]) == 3
        for slc in loaded["slices"]:
            assert "metrics" in slc
            assert "per_answer_type" in slc
            assert "per_difficulty" in slc
            assert "per_language" in slc

    def test_markdown_renders_three_slice_headers(self):
        det = [_row(rid="a", hit5=1.0)]
        opus = [_row(rid="b", hit5=0.0)]
        comparison = run_comparison(
            deterministic_rows=det,
            deterministic_dataset_path=None,
            opus_rows=opus,
            opus_dataset_path=None,
        )
        md = render_comparison_markdown(comparison)
        assert "# Retrieval baseline comparison" in md
        assert "deterministic_all" in md
        assert "deterministic_without_character_relation" in md
        assert "opus_all" in md
        assert "## Headline metrics" in md
        # All three per-axis sections — empty per_language buckets are
        # filled in by the per-row data.
        assert "## Per answer_type" in md
        assert "## Per difficulty" in md
        assert "## Per language" in md
