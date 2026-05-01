"""Unit tests for ``eval.harness.llm_silver_human_seed``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.llm_silver_human_seed import (
    ALLOWED_HUMAN_LABELS,
    DEFAULT_BUCKET_TARGETS,
    DEFAULT_CROSS_TAB,
    DEFAULT_QUERY_TYPE_TARGETS,
    HUMAN_GOLD_DISCLAIMER_MARKER,
    HumanSeedConfig,
    build_human_seed,
    write_csv,
    write_jsonl,
    write_md,
    write_outputs,
)


# Synthetic LLM silver-500 record builder.
def _silver_record(qid: int, bucket: str, query_type: str, leakage="low") -> dict:
    is_unans = query_type == "unanswerable_or_not_in_corpus"
    return {
        "query_id": f"v4-llm-silver-{qid:03d}",
        "query": f"sample query {qid}",
        "query_type": query_type,
        "bucket": bucket,
        "silver_expected_title": None if is_unans else f"title-{qid}",
        "silver_expected_page_id": None if is_unans else f"doc-{qid}",
        "expected_section_path": None if is_unans else ["개요"],
        "expected_not_in_corpus": is_unans,
        "rationale_for_expected_target": "test",
        "lexical_overlap": {
            "title_char2_jaccard": None if is_unans else 0.1,
            "section_char2_jaccard": None if is_unans else 0.1,
            "chunk_char4_containment": None if is_unans else 0.1,
            "bm25_expected_page_first_rank": None,
            "overlap_risk": "not_applicable" if is_unans else "low",
        },
        "leakage_risk": "not_applicable" if is_unans else leakage,
        "tags": ["anime", "v4-llm-silver-500", "silver"],
    }


def _build_silver_500_jsonl(tmp_path: Path) -> Path:
    """Build a synthetic 500-row LLM silver JSONL matching the cross-tab."""
    rows = []
    qid = 1
    # Use the same cross-tab as the real LLM silver-500.
    cross = {
        ("main_work", "direct_title"): 35,
        ("main_work", "paraphrase_semantic"): 25,
        ("main_work", "indirect_entity"): 10,
        ("main_work", "alias_variant"): 35,
        ("main_work", "ambiguous"): 45,
        ("subpage_generic", "direct_title"): 20,
        ("subpage_generic", "paraphrase_semantic"): 65,
        ("subpage_generic", "section_intent"): 105,
        ("subpage_generic", "indirect_entity"): 25,
        ("subpage_generic", "alias_variant"): 5,
        ("subpage_generic", "ambiguous"): 5,
        ("subpage_named", "direct_title"): 5,
        ("subpage_named", "paraphrase_semantic"): 35,
        ("subpage_named", "section_intent"): 5,
        ("subpage_named", "indirect_entity"): 50,
        ("subpage_named", "alias_variant"): 5,
        ("not_in_corpus", "unanswerable_or_not_in_corpus"): 25,
    }
    for (b, qt), n in cross.items():
        for _ in range(n):
            rows.append(_silver_record(qid, b, qt))
            qid += 1
    path = tmp_path / "synthetic_silver.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


class TestHumanSeedConfig:
    def test_default_config_validates(self):
        HumanSeedConfig().validate()

    def test_bucket_target_sum_must_match_total(self):
        with pytest.raises(ValueError):
            HumanSeedConfig(
                target_total=100,
                bucket_targets={"main_work": 99},  # wrong sum
                cross_tab=None,
            ).validate()

    def test_cross_tab_marginals_validated(self):
        # Bad cross-tab: sums don't match bucket targets.
        bad_ctab = {
            ("main_work", "direct_title"): 100,  # row sum != bucket_target
        }
        with pytest.raises(ValueError):
            HumanSeedConfig(
                target_total=100,
                bucket_targets={"main_work": 25, "subpage_generic": 35,
                                "subpage_named": 35, "not_in_corpus": 5},
                cross_tab=bad_ctab,
            ).validate()


class TestBuildHumanSeed:
    def test_actual_total_matches_target(self, tmp_path):
        path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(path)
        assert export.audit["actual_total"] == 100

    def test_bucket_distribution_exact(self, tmp_path):
        path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(path)
        for b, tgt in DEFAULT_BUCKET_TARGETS.items():
            assert export.audit["bucket_actual"][b] == tgt
            assert export.audit["bucket_deficits"][b] == 0

    def test_query_type_distribution_exact_with_cross_tab(self, tmp_path):
        path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(path)
        for qt, tgt in DEFAULT_QUERY_TYPE_TARGETS.items():
            assert export.audit["query_type_actual"].get(qt, 0) == tgt
            assert export.audit["query_type_deficits"].get(qt, 0) == 0

    def test_deterministic(self, tmp_path):
        path = _build_silver_500_jsonl(tmp_path)
        a = build_human_seed(path)
        b = build_human_seed(path)
        assert [r.query_id for r in a.rows] == [r.query_id for r in b.rows]

    def test_audit_fields_empty_in_rows(self, tmp_path):
        path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(path)
        for r in export.rows:
            assert r.human_label == ""
            assert r.human_correct_title == ""
            assert r.human_correct_page_id == ""
            assert r.human_supporting_chunk_id == ""
            assert r.human_notes == ""


class TestWriteOutputs:
    def test_writes_three_files(self, tmp_path):
        silver_path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(silver_path)
        out_dir = tmp_path / "out"
        paths = write_outputs(
            export, out_dir=out_dir, base_name="seed_test", target_total=100,
        )
        assert paths["jsonl"].exists()
        assert paths["csv"].exists()
        assert paths["md"].exists()

    def test_md_carries_disclaimer(self, tmp_path):
        silver_path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(silver_path)
        out_dir = tmp_path / "out"
        paths = write_outputs(
            export, out_dir=out_dir, base_name="seed_test", target_total=100,
        )
        md = paths["md"].read_text(encoding="utf-8")
        assert HUMAN_GOLD_DISCLAIMER_MARKER in md

    def test_md_lists_all_human_labels(self, tmp_path):
        silver_path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(silver_path)
        out_dir = tmp_path / "out"
        paths = write_outputs(
            export, out_dir=out_dir, base_name="seed_test", target_total=100,
        )
        md = paths["md"].read_text(encoding="utf-8")
        for lab in ALLOWED_HUMAN_LABELS:
            assert lab in md

    def test_csv_has_audit_columns(self, tmp_path):
        silver_path = _build_silver_500_jsonl(tmp_path)
        export = build_human_seed(silver_path)
        out_dir = tmp_path / "out"
        paths = write_outputs(
            export, out_dir=out_dir, base_name="seed_test", target_total=100,
        )
        csv_text = paths["csv"].read_text(encoding="utf-8")
        header = csv_text.splitlines()[0]
        assert "human_label" in header
        assert "human_correct_title" in header
        assert "human_correct_page_id" in header
        assert "human_supporting_chunk_id" in header
        assert "human_notes" in header
