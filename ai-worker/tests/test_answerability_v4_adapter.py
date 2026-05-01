"""Tests for the Phase 7.7 v4 production retrieval emit → audit input adapter.

The adapter sits between two formats that disagree on container key,
title key, section_path shape, and chunk_text source. These tests
exercise every disagreement explicitly:

  * **section_path normalisation** — list / tuple → joined string,
    string passthrough, empty / None handling.
  * **chunk_text resolution** — chunks_v4 ``text`` form and rag_chunks
    ``chunk_text`` form both accepted; embedding-side fields
    (``embedding_text`` / ``text_for_embedding``) explicitly NOT
    substituted, even if no other text key is present.
  * **gold loaders** — gold-50 csv (with human-vs-silver fallback),
    llm-silver-500 jsonl (``query_id`` / ``silver_expected_*`` keys),
    silver-500 jsonl (``id`` / ``expected_doc_ids`` / keyword list).
  * **per-record adaptation** — docs → results, title → page_title,
    chunk_text resolved by chunk_id lookup, missing-chunk policy
    (error vs collect), missing gold ⇒ empty gold block.
  * **JSONL driver** — round-trips all of the above against an
    on-disk fixture, covers the lookup-vs-path overload, surfaces
    JSON parse errors at line number.
  * **CLI integration** — ``scripts.export_answerability_audit``
    with ``--input-format v4-production`` produces an audit-shape
    CSV that the existing labelled-row reader does not reject for
    structural reasons (label cells are blank, as expected — that
    is a labelling-step concern, not an adapter concern).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from eval.harness.answerability_v4_adapter import (
    SECTION_PATH_JOINER,
    V4AdapterError,
    V4ChunkRef,
    V4GoldRef,
    _gold_from_csv_row,
    _gold_from_jsonl_record,
    _resolve_chunk_text,
    adapt_v4_retrieval_jsonl,
    adapt_v4_retrieval_record,
    load_v4_chunk_lookup,
    load_v4_gold_lookup,
    normalise_v4_section_path,
)


# ---------------------------------------------------------------------------
# section_path normalisation
# ---------------------------------------------------------------------------


class TestNormaliseV4SectionPath:
    """v4 section_path arrives as list / tuple / string / None;
    the audit row carries a single human-readable string."""

    def test_list_joined(self):
        assert normalise_v4_section_path(
            ["음악", "주제가", "OP"]
        ) == "음악 > 주제가 > OP"

    def test_tuple_joined(self):
        assert normalise_v4_section_path(
            ("개요", "등장인물")
        ) == "개요 > 등장인물"

    def test_string_passthrough(self):
        # Pre-joined strings (older fixtures) must round-trip unchanged.
        assert normalise_v4_section_path(
            "음악 > 주제가"
        ) == "음악 > 주제가"

    def test_none_empty(self):
        assert normalise_v4_section_path(None) == ""
        assert normalise_v4_section_path([]) == ""
        assert normalise_v4_section_path(()) == ""
        assert normalise_v4_section_path("") == ""

    def test_skips_empty_segments(self):
        assert normalise_v4_section_path(
            ["개요", "", "등장인물"]
        ) == "개요 > 등장인물"

    def test_uses_module_joiner_constant(self):
        # Renaming the joiner would silently break every existing
        # labelling file; pin the constant here.
        assert SECTION_PATH_JOINER == " > "


# ---------------------------------------------------------------------------
# chunk_text resolution — raw text only, embedding fields forbidden
# ---------------------------------------------------------------------------


class TestResolveChunkText:
    """chunks_v4 stores raw text under ``text``; rag_chunks stores it
    under ``chunk_text``. Either MUST be accepted. Embedding-side
    fields MUST NOT be substituted."""

    def test_rag_chunks_form(self):
        assert _resolve_chunk_text({"chunk_text": "raw"}) == "raw"

    def test_chunks_v4_form(self):
        assert _resolve_chunk_text({"text": "raw"}) == "raw"

    def test_chunk_text_wins_over_text(self):
        assert _resolve_chunk_text(
            {"chunk_text": "rag", "text": "v4"},
        ) == "rag"

    def test_falls_through_when_chunk_text_empty(self):
        assert _resolve_chunk_text(
            {"chunk_text": "", "text": "v4"},
        ) == "v4"

    def test_empty_when_neither_present(self):
        assert _resolve_chunk_text({}) == ""

    def test_does_not_pick_embedding_fields(self):
        """The headline contract: embedding-augmented text would
        mislead a reviewer. These keys MUST NOT be substituted."""
        assert _resolve_chunk_text({
            "embedding_text": "title: X\nsection: Y\nbody: ...",
        }) == ""
        assert _resolve_chunk_text({
            "text_for_embedding": "제목: X\n섹션: Y\n본문: ...",
        }) == ""
        # Even when embedding fields coexist with empty raw fields:
        assert _resolve_chunk_text({
            "chunk_text": "",
            "embedding_text": "should not be used",
        }) == ""


# ---------------------------------------------------------------------------
# Chunk lookup loader
# ---------------------------------------------------------------------------


class TestLoadV4ChunkLookup:
    """chunks_v4 / rag_chunks must both load through the same loader,
    indexed by chunk_id, with raw text only."""

    def _write_jsonl(self, path: Path, records: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False))
                fp.write("\n")

    def test_loads_chunks_v4_form(self, tmp_path: Path):
        p = tmp_path / "chunks_v4.jsonl"
        self._write_jsonl(p, [
            {
                "chunk_id": "c1",
                "page_id": "p1",
                "page_title": "P1",
                "section_path": ["개요"],
                "text": "raw v4 text",
                # chunks_v4 also carries this — must be ignored.
                "text_for_embedding": "title:P1\nsection:개요\nbody:...",
            },
        ])
        lookup = load_v4_chunk_lookup(p)
        ref = lookup["c1"]
        assert isinstance(ref, V4ChunkRef)
        assert ref.page_id == "p1"
        assert ref.page_title == "P1"
        # chunks_v4 has no section_id; loader fills empty.
        assert ref.section_id == ""
        assert ref.section_path == "개요"
        assert ref.chunk_text == "raw v4 text"

    def test_loads_rag_chunks_form(self, tmp_path: Path):
        p = tmp_path / "rag_chunks.jsonl"
        self._write_jsonl(p, [
            {
                "chunk_id": "c2",
                "doc_id": "p2",
                "title": "P2",
                "section_id": "s2",
                "section_path": ["음악", "주제가", "OP"],
                "chunk_text": "raw rag text",
                "embedding_text": "제목:P2\n섹션:음악>주제가>OP\n본문:...",
            },
        ])
        lookup = load_v4_chunk_lookup(p)
        ref = lookup["c2"]
        assert ref.page_id == "p2"
        assert ref.page_title == "P2"
        assert ref.section_id == "s2"
        assert ref.section_path == "음악 > 주제가 > OP"
        assert ref.chunk_text == "raw rag text"

    def test_skips_blank_and_comment_lines(self, tmp_path: Path):
        p = tmp_path / "mixed.jsonl"
        with p.open("w", encoding="utf-8") as fp:
            fp.write("\n")
            fp.write("# header comment\n")
            fp.write(json.dumps({
                "chunk_id": "c1", "doc_id": "p1",
                "title": "T1", "chunk_text": "x",
            }) + "\n")
            fp.write("   \n")
        lookup = load_v4_chunk_lookup(p)
        assert set(lookup.keys()) == {"c1"}

    def test_raises_on_missing_chunk_id(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        self._write_jsonl(p, [
            {"page_id": "p1", "text": "no chunk_id here"},
        ])
        with pytest.raises(V4AdapterError, match="chunk_id"):
            load_v4_chunk_lookup(p)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(V4AdapterError, match="not found"):
            load_v4_chunk_lookup(tmp_path / "nope.jsonl")

    def test_raises_on_invalid_json(self, tmp_path: Path):
        p = tmp_path / "broken.jsonl"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(V4AdapterError, match="invalid JSON"):
            load_v4_chunk_lookup(p)


# ---------------------------------------------------------------------------
# Gold loader — three v4 schemas
# ---------------------------------------------------------------------------


class TestLoadV4GoldLookup:

    def test_csv_human_correct_wins(self, tmp_path: Path):
        """gold-50 CSV: ``human_correct_*`` wins over silver_expected_*
        when populated."""
        p = tmp_path / "gold.csv"
        with p.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "query_id",
                "silver_expected_page_id", "silver_expected_title",
                "human_correct_page_id", "human_correct_title",
            ])
            writer.writeheader()
            writer.writerow({
                "query_id": "q1",
                "silver_expected_page_id": "p_silver",
                "silver_expected_title": "T_silver",
                "human_correct_page_id": "p_human",
                "human_correct_title": "T_human",
            })
            writer.writerow({
                "query_id": "q2",
                "silver_expected_page_id": "p_silver_2",
                "silver_expected_title": "T_silver_2",
                "human_correct_page_id": "",
                "human_correct_title": "",
            })
        lookup = load_v4_gold_lookup(p)
        assert lookup["q1"].page_id == "p_human"
        assert lookup["q1"].page_title == "T_human"
        # Falls back to silver when human cells are empty.
        assert lookup["q2"].page_id == "p_silver_2"
        assert lookup["q2"].page_title == "T_silver_2"
        # gold-50 CSV has no section_path / section_id.
        assert lookup["q1"].section_path == ""
        assert lookup["q1"].section_id == ""

    def test_jsonl_llm_silver_form(self, tmp_path: Path):
        """``query_id`` key + ``silver_expected_*`` ⇒ llm-silver-500 form."""
        p = tmp_path / "llm_silver.jsonl"
        with p.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "query_id": "v4-llm-silver-001",
                "query": "...",
                "silver_expected_page_id": "p1",
                "silver_expected_title": "원피스",
                "expected_section_path": ["개요"],
            }, ensure_ascii=False) + "\n")
        lookup = load_v4_gold_lookup(p)
        ref = lookup["v4-llm-silver-001"]
        assert ref.page_id == "p1"
        assert ref.page_title == "원피스"
        assert ref.section_path == "개요"

    def test_jsonl_silver_500_form(self, tmp_path: Path):
        """``id`` key + ``expected_doc_ids`` ⇒ silver-500 form."""
        p = tmp_path / "silver_500.jsonl"
        with p.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "id": "v4-silver-500-0001",
                "query": "...",
                "expected_doc_ids": ["p1", "p2"],
                "expected_section_keywords": ["등장인물", "내용"],
            }, ensure_ascii=False) + "\n")
        lookup = load_v4_gold_lookup(p)
        ref = lookup["v4-silver-500-0001"]
        # First expected_doc_id wins.
        assert ref.page_id == "p1"
        # silver-500 carries no canonical title.
        assert ref.page_title == ""
        # keyword list joined as section_path proxy.
        assert ref.section_path == "등장인물 > 내용"

    def test_jsonl_unknown_key_skipped(self, tmp_path: Path):
        """Records lacking both query_id and id keys are skipped."""
        p = tmp_path / "weird.jsonl"
        with p.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({"foo": "bar"}) + "\n")
        lookup = load_v4_gold_lookup(p)
        assert lookup == {}

    def test_unsupported_suffix(self, tmp_path: Path):
        p = tmp_path / "gold.txt"
        p.write_text("nope", encoding="utf-8")
        with pytest.raises(V4AdapterError, match="unsupported"):
            load_v4_gold_lookup(p)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(V4AdapterError, match="not found"):
            load_v4_gold_lookup(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# Per-record adaptation
# ---------------------------------------------------------------------------


class TestAdaptV4RetrievalRecord:

    def _stub_chunk(
        self,
        chunk_id: str = "c1",
        *,
        page_id: str = "p1",
        page_title: str = "P1",
        section_id: str = "s1",
        section_path: str = "개요",
        chunk_text: str = "raw text",
    ) -> V4ChunkRef:
        return V4ChunkRef(
            chunk_id=chunk_id, page_id=page_id, page_title=page_title,
            section_id=section_id, section_path=section_path,
            chunk_text=chunk_text,
        )

    def _stub_gold(self, qid: str = "q1") -> V4GoldRef:
        return V4GoldRef(
            query_id=qid, page_id="p1", page_title="P1",
            section_id="", section_path="개요",
        )

    def test_renames_docs_to_results(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1", "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk()},
            gold_lookup={"q1": self._stub_gold()},
        )
        assert "results" in rec
        assert "docs" not in rec
        assert len(rec["results"]) == 1

    def test_renames_title_to_page_title(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "page-title-from-emit",
                     "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk(
                page_title="page-title-from-emit",
            )},
            gold_lookup={"q1": self._stub_gold()},
        )
        r0 = rec["results"][0]
        assert r0["page_title"] == "page-title-from-emit"
        assert "title" not in r0

    def test_resolves_chunk_text_from_lookup(self):
        """retrieval emit does not carry chunk_text — adapter must
        fill it from the chunk fixture by chunk_id."""
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1", "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk(
                chunk_text="raw chunk text from fixture",
            )},
            gold_lookup={"q1": self._stub_gold()},
        )
        assert (
            rec["results"][0]["chunk_text"]
            == "raw chunk text from fixture"
        )

    def test_section_path_list_joined(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1",
                     "section_path": ["음악", "주제가", "OP"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk(
                section_path="음악 > 주제가 > OP",
            )},
            gold_lookup={"q1": self._stub_gold()},
        )
        assert (
            rec["results"][0]["section_path"]
            == "음악 > 주제가 > OP"
        )

    def test_gold_block_populated(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1", "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk()},
            gold_lookup={"q1": self._stub_gold()},
        )
        assert rec["gold"] == {
            "page_id": "p1", "page_title": "P1",
            "section_id": "", "section_path": "개요",
        }

    def test_missing_gold_yields_empty_gold_block(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q_unknown", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1", "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk()},
            gold_lookup={},  # nothing for q_unknown
        )
        assert rec["gold"] == {
            "page_id": "", "page_title": "",
            "section_id": "", "section_path": "",
        }

    def test_missing_chunk_error_default(self):
        with pytest.raises(V4AdapterError, match="not found"):
            adapt_v4_retrieval_record(
                {
                    "query_id": "q1", "query": "...",
                    "docs": [
                        {"rank": 1, "chunk_id": "c_missing",
                         "page_id": "p1", "title": "P1",
                         "section_path": ["개요"]},
                    ],
                },
                chunk_lookup={"c1": self._stub_chunk()},  # no c_missing
                gold_lookup={"q1": self._stub_gold()},
            )

    def test_metadata_mismatch_errors_by_default(self):
        with pytest.raises(V4AdapterError, match="metadata mismatch"):
            adapt_v4_retrieval_record(
                {
                    "query_id": "q1", "query": "...",
                    "docs": [
                        {"rank": 1, "chunk_id": "c1",
                         "page_id": "stale-page", "title": "P1",
                         "section_path": ["개요"]},
                    ],
                },
                chunk_lookup={"c1": self._stub_chunk(page_id="p1")},
                gold_lookup={"q1": self._stub_gold()},
            )

    def test_metadata_mismatch_collect_uses_fixture_metadata(self):
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1",
                     "page_id": "stale-page", "title": "Stale title",
                     "section_path": ["오래된", "섹션"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk(
                page_id="p1",
                page_title="P1",
                section_path="개요",
            )},
            gold_lookup={"q1": self._stub_gold()},
            on_metadata_mismatch="collect",
        )

        r0 = rec["results"][0]
        assert r0["page_id"] == "p1"
        assert r0["page_title"] == "P1"
        assert r0["section_path"] == "개요"
        assert rec["_metadata_mismatches"][0][0:3] == ("q1", "c1", 1)

    def test_missing_chunk_collect_mode(self):
        """``collect`` records the missing tuple and continues with
        an empty chunk_text cell; the audit's import validator will
        reject it later, which is the correct fail-closed behaviour."""
        rec = adapt_v4_retrieval_record(
            {
                "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c_missing",
                     "page_id": "p1", "title": "P1",
                     "section_path": ["개요"]},
                ],
            },
            chunk_lookup={"c1": self._stub_chunk()},
            gold_lookup={"q1": self._stub_gold()},
            on_missing_chunk="collect",
        )
        assert rec["results"][0]["chunk_text"] == ""
        assert rec["_missing_chunks"] == [("q1", "c_missing", 1)]

    def test_invalid_on_missing_chunk_raises(self):
        with pytest.raises(V4AdapterError, match="on_missing_chunk"):
            adapt_v4_retrieval_record(
                {"query_id": "q1", "query": "...", "docs": []},
                chunk_lookup={}, gold_lookup={},
                on_missing_chunk="ignore",
            )

    def test_invalid_on_metadata_mismatch_raises(self):
        with pytest.raises(V4AdapterError, match="on_metadata_mismatch"):
            adapt_v4_retrieval_record(
                {"query_id": "q1", "query": "...", "docs": []},
                chunk_lookup={}, gold_lookup={},
                on_metadata_mismatch="ignore",
            )

    def test_record_missing_query_id_raises(self):
        with pytest.raises(V4AdapterError, match="query_id"):
            adapt_v4_retrieval_record(
                {"query": "...", "docs": []},
                chunk_lookup={}, gold_lookup={},
            )

    def test_record_missing_docs_raises(self):
        with pytest.raises(V4AdapterError, match="docs"):
            adapt_v4_retrieval_record(
                {"query_id": "q1", "query": "..."},
                chunk_lookup={}, gold_lookup={},
            )

    def test_doc_missing_rank_raises(self):
        with pytest.raises(V4AdapterError, match="rank"):
            adapt_v4_retrieval_record(
                {
                    "query_id": "q1", "query": "...",
                    "docs": [
                        {"chunk_id": "c1", "page_id": "p1",
                         "title": "P1"},
                    ],
                },
                chunk_lookup={"c1": self._stub_chunk()},
                gold_lookup={"q1": self._stub_gold()},
            )


# ---------------------------------------------------------------------------
# JSONL driver
# ---------------------------------------------------------------------------


class TestAdaptV4RetrievalJsonl:

    def _write(self, path: Path, records: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False))
                fp.write("\n")

    def test_round_trip_with_paths(self, tmp_path: Path):
        chunks_path = tmp_path / "rag_chunks.jsonl"
        gold_path = tmp_path / "gold.jsonl"
        retrieval_path = tmp_path / "retrieval.jsonl"

        self._write(chunks_path, [
            {"chunk_id": "c1", "doc_id": "p1", "title": "P1",
             "section_id": "s1", "section_path": ["개요"],
             "chunk_text": "raw"},
        ])
        self._write(gold_path, [
            {"query_id": "q1",
             "silver_expected_page_id": "p1",
             "silver_expected_title": "P1",
             "expected_section_path": ["개요"]},
        ])
        self._write(retrieval_path, [
            {"variant": "baseline", "query_id": "q1", "query": "...",
             "elapsed_ms": 10.0,
             "docs": [
                 {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                  "title": "P1", "section_path": ["개요"],
                  "score": 0.9},
             ]},
        ])

        records = list(adapt_v4_retrieval_jsonl(
            retrieval_path,
            chunks_path=chunks_path,
            gold_path=gold_path,
        ))
        assert len(records) == 1
        rec = records[0]
        assert rec["query_id"] == "q1"
        assert rec["gold"]["page_id"] == "p1"
        assert rec["results"][0]["chunk_text"] == "raw"
        assert rec["results"][0]["page_title"] == "P1"
        assert rec["results"][0]["section_path"] == "개요"

    def test_accepts_pre_loaded_lookups(self, tmp_path: Path):
        retrieval_path = tmp_path / "retrieval.jsonl"
        self._write(retrieval_path, [
            {"variant": "x", "query_id": "q1", "query": "...",
             "docs": [
                 {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                  "title": "P1", "section_path": ["개요"]},
             ]},
        ])
        chunk_lookup = {"c1": V4ChunkRef(
            chunk_id="c1", page_id="p1", page_title="P1",
            section_id="s1", section_path="개요",
            chunk_text="from in-memory lookup",
        )}
        gold_lookup = {"q1": V4GoldRef(
            query_id="q1", page_id="p1", page_title="P1",
            section_id="", section_path="개요",
        )}
        records = list(adapt_v4_retrieval_jsonl(
            retrieval_path,
            chunk_lookup=chunk_lookup,
            gold_lookup=gold_lookup,
        ))
        assert records[0]["results"][0]["chunk_text"] == (
            "from in-memory lookup"
        )

    def test_requires_lookup_or_path(self, tmp_path: Path):
        retrieval_path = tmp_path / "retrieval.jsonl"
        retrieval_path.write_text("", encoding="utf-8")
        with pytest.raises(V4AdapterError, match="chunk_lookup"):
            list(adapt_v4_retrieval_jsonl(retrieval_path))

    def test_skips_blank_and_comment_lines(self, tmp_path: Path):
        retrieval_path = tmp_path / "retrieval.jsonl"
        with retrieval_path.open("w", encoding="utf-8") as fp:
            fp.write("\n# header\n")
            fp.write(json.dumps({
                "variant": "x", "query_id": "q1", "query": "...",
                "docs": [],
            }) + "\n")
        records = list(adapt_v4_retrieval_jsonl(
            retrieval_path,
            chunk_lookup={}, gold_lookup={},
        ))
        assert len(records) == 1

    def test_invalid_json_surfaces_line_no(self, tmp_path: Path):
        retrieval_path = tmp_path / "retrieval.jsonl"
        retrieval_path.write_text("{broken", encoding="utf-8")
        with pytest.raises(V4AdapterError, match="line 1"):
            list(adapt_v4_retrieval_jsonl(
                retrieval_path,
                chunk_lookup={}, gold_lookup={},
            ))


# ---------------------------------------------------------------------------
# CLI integration — export with --input-format v4-production
# ---------------------------------------------------------------------------


class TestExportV4ProductionMode:
    """End-to-end smoke through the export CLI: write three tiny
    fixtures, run ``main([...])``, and verify the resulting CSV is
    a valid audit-shape file (no labels yet — that is the human's
    job)."""

    def _build_fixtures(self, tmp_path: Path):
        chunks_path = tmp_path / "rag_chunks.jsonl"
        gold_path = tmp_path / "gold.jsonl"
        retrieval_path = tmp_path / "retrieval.jsonl"
        out_path = tmp_path / "export.csv"

        with chunks_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "chunk_id": "c1", "doc_id": "p1",
                "title": "P1", "section_id": "s1",
                "section_path": ["개요"],
                "chunk_text": "raw chunk one",
                # MUST be ignored by the resolver:
                "embedding_text": "title: P1\nbody: ...",
            }, ensure_ascii=False) + "\n")
            fp.write(json.dumps({
                "chunk_id": "c2", "doc_id": "p1",
                "title": "P1", "section_id": "s2",
                "section_path": ["등장인물"],
                "chunk_text": "raw chunk two",
            }, ensure_ascii=False) + "\n")

        with gold_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "query_id": "q1",
                "silver_expected_page_id": "p1",
                "silver_expected_title": "P1",
                "expected_section_path": ["개요"],
            }, ensure_ascii=False) + "\n")

        with retrieval_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "variant": "baseline_retrieval_title_section_top10",
                "query_id": "q1", "query": "테스트 쿼리",
                "elapsed_ms": 12.3,
                "docs": [
                    {"rank": 1, "chunk_id": "c1", "page_id": "p1",
                     "title": "P1", "section_path": ["개요"],
                     "score": 0.9},
                    {"rank": 2, "chunk_id": "c2", "page_id": "p1",
                     "title": "P1", "section_path": ["등장인물"],
                     "score": 0.7},
                ],
            }, ensure_ascii=False) + "\n")
        return chunks_path, gold_path, retrieval_path, out_path

    def test_v4_production_smoke_writes_audit_shape_csv(
        self, tmp_path: Path,
    ):
        from scripts.export_answerability_audit import main as export_main

        (
            chunks_path, gold_path, retrieval_path, out_path,
        ) = self._build_fixtures(tmp_path)

        rc = export_main([
            "--mode", "row",
            "--input-format", "v4-production",
            "--retrieval-results-path", str(retrieval_path),
            "--chunks-path", str(chunks_path),
            "--gold-path", str(gold_path),
            "--variant-name", "baseline_v4_smoke",
            "--top-k", "5",
            "--out-path", str(out_path),
        ])
        assert rc == 0
        assert out_path.exists()

        with out_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        assert len(rows) == 2

        r0, r1 = rows
        # docs[0] → results[0] with title → page_title.
        assert r0["page_hit"] == "true"
        assert r0["section_hit"] == "true"
        assert r0["retrieved_page_title"] == "P1"
        assert r0["retrieved_section_path"] == "개요"
        # chunk_text resolved from rag_chunks (raw, not embedding text).
        assert r0["chunk_text"] == "raw chunk one"
        # Gold block populated from llm-silver-style record.
        assert r0["gold_page_id"] == "p1"
        assert r0["gold_section_path"] == "개요"
        # Label cells empty — the human's job.
        assert r0["label_answerability"] == ""
        assert r0["flags"] == ""
        assert r0["notes"] == ""

        # docs[1] (different section) — page hits, section misses.
        assert r1["page_hit"] == "true"
        assert r1["section_hit"] == "false"
        assert r1["chunk_text"] == "raw chunk two"

    def test_v4_production_requires_chunks_path(
        self, tmp_path: Path,
    ):
        from scripts.export_answerability_audit import main as export_main
        _, gold_path, retrieval_path, out_path = self._build_fixtures(
            tmp_path,
        )
        with pytest.raises(SystemExit, match="chunks-path"):
            export_main([
                "--mode", "row",
                "--input-format", "v4-production",
                "--retrieval-results-path", str(retrieval_path),
                "--gold-path", str(gold_path),
                "--variant-name", "x",
                "--out-path", str(out_path),
            ])

    def test_v4_production_no_gold_requires_explicit_triage_flag(
        self, tmp_path: Path,
    ):
        from scripts.export_answerability_audit import main as export_main

        chunks_path, gold_path, retrieval_path, out_path = (
            self._build_no_gold_fixtures(tmp_path)
        )

        with pytest.raises(SystemExit, match="allow-missing-gold"):
            export_main([
                "--mode", "row",
                "--input-format", "v4-production",
                "--retrieval-results-path", str(retrieval_path),
                "--chunks-path", str(chunks_path),
                "--gold-path", str(gold_path),
                "--variant-name", "x",
                "--top-k", "5",
                "--out-path", str(out_path),
            ])

    def test_v4_production_no_gold_yields_empty_gold_block_with_triage_flag(
        self, tmp_path: Path, caplog,
    ):
        """A retrieval record whose query_id has no matching gold
        entry may still export in explicit triage mode. ``page_hit``
        will be all-False for those rows, and the driver logs a warning
        so a reviewer notices the coverage gap before labelling."""
        import logging
        from scripts.export_answerability_audit import main as export_main

        chunks_path, gold_path, retrieval_path, out_path = (
            self._build_no_gold_fixtures(tmp_path)
        )

        with caplog.at_level(
            logging.WARNING,
            logger="scripts.export_answerability_audit",
        ):
            rc = export_main([
                "--mode", "row",
                "--input-format", "v4-production",
                "--retrieval-results-path", str(retrieval_path),
                "--chunks-path", str(chunks_path),
                "--gold-path", str(gold_path),
                "--variant-name", "x",
                "--top-k", "5",
                "--allow-missing-gold",
                "--out-path", str(out_path),
            ])
        assert rc == 0
        assert out_path.exists()

        # Row written with empty gold cells — page_hit / section_hit
        # both False (gold page_id is empty so cannot match).
        with out_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        assert len(rows) == 1
        assert rows[0]["gold_page_id"] == ""
        assert rows[0]["page_hit"] == "false"
        assert rows[0]["section_hit"] == "false"

        # Driver surfaced the no-gold case in a warning so the
        # reviewer sees the coverage gap before labelling.
        warning_text = " ".join(
            rec.message for rec in caplog.records
            if rec.levelno >= logging.WARNING
        )
        assert "no gold page_id" in warning_text

    def _build_no_gold_fixtures(self, tmp_path: Path):
        chunks_path = tmp_path / "rag_chunks.jsonl"
        gold_path = tmp_path / "gold.jsonl"
        retrieval_path = tmp_path / "retrieval.jsonl"
        out_path = tmp_path / "export.csv"

        with chunks_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "chunk_id": "c1", "doc_id": "p1", "title": "P1",
                "chunk_text": "raw",
            }) + "\n")
        # Gold lookup intentionally has only q_other, not q_no_gold.
        with gold_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "query_id": "q_other",
                "silver_expected_page_id": "p_other",
                "silver_expected_title": "P_other",
            }) + "\n")
        with retrieval_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "variant": "x", "query_id": "q_no_gold", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "c1",
                     "page_id": "p1", "title": "P1",
                     "section_path": ["개요"]},
                ],
            }) + "\n")
        return chunks_path, gold_path, retrieval_path, out_path

    def test_v4_production_missing_chunk_errors_by_default(
        self, tmp_path: Path,
    ):
        """A retrieval doc referencing a chunk_id absent from the
        fixture must blow up the export by default."""
        from scripts.export_answerability_audit import main as export_main

        chunks_path = tmp_path / "rag_chunks.jsonl"
        gold_path = tmp_path / "gold.jsonl"
        retrieval_path = tmp_path / "retrieval.jsonl"
        out_path = tmp_path / "export.csv"

        with chunks_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "chunk_id": "c1", "doc_id": "p1", "title": "P1",
                "chunk_text": "x",
            }) + "\n")
        with gold_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "query_id": "q1",
                "silver_expected_page_id": "p1",
                "silver_expected_title": "P1",
            }) + "\n")
        with retrieval_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "variant": "x", "query_id": "q1", "query": "...",
                "docs": [
                    {"rank": 1, "chunk_id": "MISSING",
                     "page_id": "p1", "title": "P1",
                     "section_path": ["개요"]},
                ],
            }) + "\n")
        with pytest.raises(SystemExit, match="not found"):
            export_main([
                "--mode", "row",
                "--input-format", "v4-production",
                "--retrieval-results-path", str(retrieval_path),
                "--chunks-path", str(chunks_path),
                "--gold-path", str(gold_path),
                "--variant-name", "x",
                "--out-path", str(out_path),
            ])
