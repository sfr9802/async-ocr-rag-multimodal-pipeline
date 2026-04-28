"""Tests for the Phase 2B boost-metadata extractor.

The extractor reads the namu-wiki corpus shape and builds a doc_id
→ DocBoostMetadata map. The scorer relies on the normalized fields
being pre-folded so its hot loop can do plain substring checks; these
tests pin that contract end-to-end on tiny in-memory fixtures so we
never need a live corpus to validate the wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.boost_metadata import (
    DocBoostMetadata,
    doc_metadata_from_records,
    load_doc_metadata,
)


def _doc(doc_id: str, title: str, sections=("요약", "본문")):
    return {
        "doc_id": doc_id,
        "title": title,
        "seed": title,
        "sections": {name: {"chunks": [f"chunk for {name}"]} for name in sections},
    }


class TestDocMetadataFromRecords:
    def test_normalized_title_is_lowercase_for_latin(self):
        meta = doc_metadata_from_records([_doc("d1", "MUSASHI")])
        assert meta["d1"].title == "MUSASHI"
        assert meta["d1"].normalized_title == "musashi"

    def test_korean_title_unchanged_in_normalized_form(self):
        meta = doc_metadata_from_records([_doc("d1", "템플")])
        assert meta["d1"].normalized_title == "템플"

    def test_section_names_order_preserved(self):
        meta = doc_metadata_from_records([
            _doc("d1", "X", sections=("요약", "본문", "등장인물")),
        ])
        assert meta["d1"].section_names == ("요약", "본문", "등장인물")
        assert meta["d1"].normalized_section_names == (
            "요약", "본문", "등장인물",
        )

    def test_seed_falls_back_to_title_via_doc(self):
        # The fixture sets seed == title; we just confirm it shows up.
        meta = doc_metadata_from_records([_doc("d1", "Title-A")])
        assert meta["d1"].seed == "Title-A"
        assert meta["d1"].normalized_seed == "title-a"

    def test_skips_records_without_doc_id(self):
        meta = doc_metadata_from_records([
            {"title": "no id"},
            _doc("d2", "real"),
        ])
        assert "d2" in meta
        assert len(meta) == 1

    def test_missing_title_seed_sections_default_to_empty(self):
        meta = doc_metadata_from_records([{"doc_id": "d1"}])
        m = meta["d1"]
        assert m.title == ""
        assert m.normalized_title == ""
        assert m.seed == ""
        assert m.section_names == ()

    def test_non_dict_sections_skipped(self):
        meta = doc_metadata_from_records([
            {"doc_id": "d1", "title": "X", "sections": ["요약"]},
        ])
        assert meta["d1"].section_names == ()


class TestLoadDocMetadataFromFile:
    def test_round_trips_through_jsonl(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        records = [
            _doc("d1", "MUSASHI"),
            _doc("d2", "템플"),
            _doc("d3", "X", sections=("요약",)),
        ]
        with path.open("w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta = load_doc_metadata(path)
        assert set(meta.keys()) == {"d1", "d2", "d3"}
        assert meta["d1"].normalized_title == "musashi"
        assert meta["d2"].normalized_title == "템플"

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_doc_metadata(tmp_path / "nope.jsonl")

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            fp.write("\n")
            fp.write(json.dumps(_doc("d1", "X"), ensure_ascii=False) + "\n")
            fp.write("\n\n")
        meta = load_doc_metadata(path)
        assert list(meta.keys()) == ["d1"]
