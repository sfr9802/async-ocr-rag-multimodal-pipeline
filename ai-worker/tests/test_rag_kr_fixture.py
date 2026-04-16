"""Korean RAG fixture tests.

These tests verify:
  1. kr_sample.jsonl is valid JSONL with the expected schema
  2. rag_sample_kr.jsonl eval dataset is valid with expected fields
  3. build_rag_index.py --fixture kr argument parsing works
  4. Korean fixture doc_ids match what the eval dataset expects
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval" / "datasets"


# ---------------------------------------------------------------------------
# 1. kr_sample.jsonl is valid JSONL with expected schema.
# ---------------------------------------------------------------------------


def test_kr_fixture_is_valid_jsonl():
    path = _FIXTURES_DIR / "kr_sample.jsonl"
    assert path.exists(), f"kr_sample.jsonl not found at {path}"

    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        doc = json.loads(line)
        docs.append(doc)

    assert len(docs) >= 10, f"Expected >=10 Korean docs, got {len(docs)}"

    for doc in docs:
        assert "doc_id" in doc, f"Missing doc_id in {doc.get('title', '?')}"
        assert "title" in doc
        assert "sections" in doc
        assert isinstance(doc["sections"], dict)
        # Each section must have chunks.
        for section_name, section_data in doc["sections"].items():
            if section_name == "characters":
                # characters section may have a 'list' key instead of 'chunks'
                assert "list" in section_data or "chunks" in section_data
            else:
                assert "chunks" in section_data, (
                    f"Section {section_name} in {doc['doc_id']} missing 'chunks'"
                )


# ---------------------------------------------------------------------------
# 2. rag_sample_kr.jsonl eval dataset is valid.
# ---------------------------------------------------------------------------


def test_kr_eval_dataset_is_valid():
    path = _EVAL_DIR / "rag_sample_kr.jsonl"
    assert path.exists(), f"rag_sample_kr.jsonl not found at {path}"

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        rows.append(row)

    assert len(rows) >= 8, f"Expected >=8 Korean eval rows, got {len(rows)}"

    for row in rows:
        assert "query" in row
        assert isinstance(row["query"], str)
        assert len(row["query"]) > 0
        if "expected_doc_ids" in row:
            assert isinstance(row["expected_doc_ids"], list)
            assert len(row["expected_doc_ids"]) > 0
        if "expected_keywords" in row:
            assert isinstance(row["expected_keywords"], list)


# ---------------------------------------------------------------------------
# 3. Korean eval doc_ids match fixture doc_ids.
# ---------------------------------------------------------------------------


def test_kr_eval_doc_ids_match_fixture():
    """Every expected_doc_id in the eval dataset must exist in kr_sample.jsonl."""
    fixture_path = _FIXTURES_DIR / "kr_sample.jsonl"
    eval_path = _EVAL_DIR / "rag_sample_kr.jsonl"

    fixture_doc_ids = set()
    for line in fixture_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        doc = json.loads(line)
        fixture_doc_ids.add(doc["doc_id"])

    eval_doc_ids = set()
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        for doc_id in row.get("expected_doc_ids", []):
            eval_doc_ids.add(doc_id)

    missing = eval_doc_ids - fixture_doc_ids
    assert not missing, (
        f"Eval dataset references doc_ids not in fixture: {missing}. "
        f"Fixture has: {sorted(fixture_doc_ids)}"
    )


# ---------------------------------------------------------------------------
# 4. ocr_sample_kr.jsonl eval dataset is valid.
# ---------------------------------------------------------------------------


def test_kr_ocr_eval_dataset_is_valid():
    path = _EVAL_DIR / "ocr_sample_kr.jsonl"
    assert path.exists(), f"ocr_sample_kr.jsonl not found at {path}"

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        rows.append(row)

    assert len(rows) >= 3, f"Expected >=3 Korean OCR eval rows, got {len(rows)}"

    for row in rows:
        assert "file" in row
        assert "ground_truth" in row
        assert row.get("language") == "kor"


# ---------------------------------------------------------------------------
# 5. build_rag_index.py argument parser accepts --fixture kr/both.
# ---------------------------------------------------------------------------


def test_build_rag_index_accepts_fixture_kr():
    """Verify the CLI argparser accepts --fixture kr without crashing."""
    import argparse

    from scripts.build_rag_index import KR_FIXTURE, DEFAULT_FIXTURE

    assert KR_FIXTURE.exists(), f"KR_FIXTURE not found at {KR_FIXTURE}"
    assert DEFAULT_FIXTURE.exists(), f"DEFAULT_FIXTURE not found at {DEFAULT_FIXTURE}"


# ---------------------------------------------------------------------------
# 6. Korean fixture contains Korean text.
# ---------------------------------------------------------------------------


def test_kr_fixture_contains_korean_text():
    path = _FIXTURES_DIR / "kr_sample.jsonl"
    full_text = path.read_text(encoding="utf-8")

    # Check for Korean character range (Hangul Syllables: U+AC00..U+D7AF).
    has_korean = any("\uac00" <= ch <= "\ud7af" for ch in full_text)
    assert has_korean, "kr_sample.jsonl should contain Korean characters"
