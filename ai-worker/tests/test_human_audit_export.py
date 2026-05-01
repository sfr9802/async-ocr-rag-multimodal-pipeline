"""Tests for the Phase 7.x human audit seed exporter.

Covers:
  - audit row construction (snippets, edge-case tagging, recovery merge)
  - stratified sampling by bucket and edge case
  - dedupe across overlapping strats
  - determinism across re-runs
  - audit row shape (every required evidence field present)
  - human_label / human_notes blank by default
  - JSONL / CSV / MD writers consistent in row count
  - silver disclaimer present in markdown + summary JSON
  - end-to-end orchestrator
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pytest

from eval.harness.eval_terminology import (
    EDGE_CASE_QUERY_REWRITE,
    EDGE_CASE_TARGET_NOT_IN_CANDIDATES,
    SILVER_DISCLAIMER_MARKER,
)
from eval.harness.human_audit_export import (
    AuditExportConfig,
    AuditRow,
    DEFAULT_BUCKET_QUOTA,
    DEFAULT_EDGE_CASE_QUOTA_PER_TAG,
    HUMAN_LABEL_CHOICES,
    _default_edge_case_quotas,
    _pick_evenly_spaced,
    build_audit_rows,
    export_audit_bundle,
    index_chunks_by_id,
    index_recovery_attempts_by_qid,
    load_jsonl,
    render_audit_md,
    sample_audit_rows,
    write_audit_csv,
    write_audit_jsonl,
    write_audit_md,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_confidence_row(
    qid: str,
    *,
    bucket: str = "main_work",
    confidence_label: str = "AMBIGUOUS",
    failure_reasons: Sequence[str] = ("LOW_MARGIN",),
    recommended_action: str = "ANSWER_WITH_CAUTION",
    expected_title: str = "T",
    gold_doc_id: str = "doc-A",
    gold_in_top_k: bool = True,
    candidates: Sequence[Dict] = (),
    query: str = "테스트",
) -> Dict:
    cands_default = [
        {
            "rank": 1, "chunk_id": "c1", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["개요"], "section_type": "summary",
            "dense_score": 0.85, "rerank_score": None, "final_score": None,
        },
        {
            "rank": 2, "chunk_id": "c2", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["기타"], "section_type": "trivia",
            "dense_score": 0.78, "rerank_score": None, "final_score": None,
        },
        {
            "rank": 3, "chunk_id": "c3", "doc_id": "doc-B",
            "title": "U", "retrieval_title": "U",
            "section_path": ["등장인물"], "section_type": "character",
            "dense_score": 0.74, "rerank_score": None, "final_score": None,
        },
        {
            "rank": 4, "chunk_id": "c4", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["줄거리"], "section_type": "summary",
            "dense_score": 0.71, "rerank_score": None, "final_score": None,
        },
        {
            "rank": 5, "chunk_id": "c5", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["설정"], "section_type": "setting",
            "dense_score": 0.69, "rerank_score": None, "final_score": None,
        },
    ]
    used_cands = list(candidates) if candidates else cands_default
    return {
        "query_id": qid,
        "bucket": bucket,
        "confidence_label": confidence_label,
        "failure_reasons": list(failure_reasons),
        "recommended_action": recommended_action,
        "signals": {
            "top1_score": 0.85,
            "top1_top2_margin": 0.07,
            "page_id_consistency": 0.8,
            "same_page_top_k_count": 8,
            "candidate_count": len(used_cands),
            "title_match": True,
            "retrieval_title_match": True,
            "section_type_match": None,
            "generic_collision_count": 1,
            "duplicate_rate": 0.4,
            "gold_in_top_k": gold_in_top_k,
            "gold_rank": 1 if gold_in_top_k else -1,
            "rerank_demoted_gold": None,
        },
        "input": {
            "query_text": query,
            "gold_doc_id": gold_doc_id,
            "gold_page_id": None,
            "expected_title": expected_title,
            "expected_section_type": None,
            "candidate_count": len(used_cands),
            "top_candidates_preview": used_cands,
        },
    }


def _make_recovery_attempt(
    qid: str,
    *,
    recovery_action: str = "ATTEMPT_HYBRID",
    rewrite_mode: str = "",
    rewritten_query: str = "",
    oracle_upper_bound: bool = False,
    after_rank: int = 1,
    before_rank: int = -1,
) -> Dict:
    return {
        "decision": {
            "query_id": qid,
            "bucket": "main_work",
            "original_action": "HYBRID_RECOVERY",
            "recovery_action": recovery_action,
            "skip_reason": None,
            "rewrite_mode": rewrite_mode or None,
            "oracle_upper_bound": oracle_upper_bound,
            "query_text": "q",
            "rewritten_query": rewritten_query or None,
            "rewrite_terms": [],
            "rewrite_source": "test",
            "expected_title": "T",
            "gold_doc_id": "doc-A",
            "gold_page_id": None,
            "notes": [],
        },
        "before_rank": before_rank,
        "before_top_doc_ids": ["doc-X"],
        "before_top_chunk_ids": ["cX"],
        "before_in_top_k": before_rank > 0,
        "before_top1_score": 0.5,
        "after_rank": after_rank,
        "after_top_doc_ids": ["doc-A"],
        "after_top_chunk_ids": ["c1"],
        "after_in_top_k": after_rank > 0,
        "after_top1_score": 0.9,
        "final_k": 10,
        "latency_ms": 12.3,
        "error": None,
    }


def _make_chunks_jsonl(tmp_path: Path) -> Path:
    """Two chunks that the build_audit_rows test will look up by id."""
    p = tmp_path / "chunks.jsonl"
    rows = [
        {
            "chunk_id": "c1", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["개요"], "section_type": "summary",
            "chunk_text": "이 작품의 개요는 주인공이 마법학교에 입학하는 이야기로 시작한다.",
        },
        {
            "chunk_id": "c2", "doc_id": "doc-A",
            "title": "T", "retrieval_title": "T",
            "section_path": ["기타"], "section_type": "trivia",
            "chunk_text": "기타 잡학 정보가 여기에 들어갑니다.",
        },
    ]
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_pick_evenly_spaced_returns_input_when_smaller_than_n():
    items = ["a", "b"]
    assert _pick_evenly_spaced(items, 5) == ["a", "b"]


def test_pick_evenly_spaced_handles_empty_or_zero():
    assert _pick_evenly_spaced([], 5) == []
    assert _pick_evenly_spaced(["a"], 0) == []


def test_pick_evenly_spaced_evenly_spreads():
    items = list(range(10))
    out = _pick_evenly_spaced(items, 5)
    # Expect a strictly-increasing subsequence of indices.
    assert sorted(out) == out
    assert len(out) == 5
    assert out[0] == 0  # always anchored at the first index.
    assert out[-1] in (8, 9)


def test_default_edge_case_quotas_includes_required_tags():
    quotas = _default_edge_case_quotas()
    for required in (
        "CONFIDENT", "AMBIGUOUS", "LOW_CONFIDENCE", "FAILED",
        "TITLE_ALIAS_MISMATCH", "GENERIC_COLLISION",
        EDGE_CASE_TARGET_NOT_IN_CANDIDATES, EDGE_CASE_QUERY_REWRITE,
    ):
        assert required in quotas


# ---------------------------------------------------------------------------
# Audit row builder
# ---------------------------------------------------------------------------


def test_build_audit_rows_carries_all_required_evidence_fields():
    confidence_rows = [_make_confidence_row("q1")]
    audit_rows = build_audit_rows(confidence_rows)
    assert len(audit_rows) == 1
    row = audit_rows[0]
    # Required fields (from the brief).
    assert row.query_id == "q1"
    assert row.query == "테스트"
    assert row.bucket == "main_work"
    assert row.expected_title == "T"
    assert row.silver_target["doc_id"] == "doc-A"
    assert row.confidence_label == "AMBIGUOUS"
    assert row.failure_reasons == ["LOW_MARGIN"]
    assert row.recommended_action == "ANSWER_WITH_CAUTION"
    assert row.top1["title"] == "T"
    assert len(row.top3_titles) == 3
    assert len(row.top5_chunks) == 5
    # Recovery is None until a Phase 7.4 recovery JSONL is supplied.
    assert row.recovery is None


def test_build_audit_rows_human_label_blank_by_default():
    audit_rows = build_audit_rows([_make_confidence_row("q1")])
    row = audit_rows[0]
    assert row.human_label is None
    assert row.human_notes is None
    # Serialised dict must surface them as null / blank, not absent.
    d = row.to_dict()
    assert d["human_label"] is None
    assert d["human_notes"] is None


def test_build_audit_rows_attaches_chunk_snippets_when_chunks_provided(tmp_path):
    chunks_index = index_chunks_by_id(_make_chunks_jsonl(tmp_path))
    audit_rows = build_audit_rows(
        [_make_confidence_row("q1")], chunks_index=chunks_index,
    )
    snippets = [c.get("snippet") for c in audit_rows[0].top5_chunks]
    # c1 / c2 are in the chunks index → real text snippets.
    # c3 / c4 / c5 are not → empty strings (handled gracefully).
    assert snippets[0]  # c1 snippet present
    assert snippets[1]  # c2 snippet present
    assert "마법학교" in snippets[0]


def test_build_audit_rows_falls_back_to_empty_snippet_without_chunks_index():
    audit_rows = build_audit_rows([_make_confidence_row("q1")])
    snippets = [c.get("snippet") for c in audit_rows[0].top5_chunks]
    assert all(s == "" for s in snippets)


def test_build_audit_rows_tags_target_not_in_candidates_when_silver_missing():
    row = _make_confidence_row(
        "qx", confidence_label="FAILED",
        failure_reasons=["GOLD_NOT_IN_CANDIDATES", "LOW_MARGIN"],
        gold_in_top_k=False,
    )
    audit_rows = build_audit_rows([row])
    assert EDGE_CASE_TARGET_NOT_IN_CANDIDATES in audit_rows[0].edge_case_tags
    assert "FAILED" in audit_rows[0].edge_case_tags


def test_build_audit_rows_tags_query_rewrite_when_recovery_present():
    base = _make_confidence_row("qrw")
    attempt = _make_recovery_attempt(
        "qrw",
        recovery_action="ATTEMPT_REWRITE",
        rewrite_mode="oracle",
        oracle_upper_bound=True,
        rewritten_query="rewritten",
        before_rank=8, after_rank=2,
    )
    recovery_by_qid = {"qrw": [attempt]}
    audit_rows = build_audit_rows([base], recovery_by_qid=recovery_by_qid)
    row = audit_rows[0]
    assert EDGE_CASE_QUERY_REWRITE in row.edge_case_tags
    assert row.recovery is not None
    assert row.recovery["recovery_action"] == "ATTEMPT_REWRITE"
    assert row.recovery["rewrite_mode"] == "oracle"
    assert row.recovery["oracle_upper_bound"] is True
    assert row.recovery["rewritten_query"] == "rewritten"
    assert row.recovery["after_rank"] == 2
    assert row.recovery["rank_delta"] == -6


def test_build_audit_rows_picks_oracle_attempt_when_both_present():
    base = _make_confidence_row("qrw")
    oracle = _make_recovery_attempt(
        "qrw", recovery_action="ATTEMPT_REWRITE",
        rewrite_mode="oracle", oracle_upper_bound=True,
        rewritten_query="oracle-rw", before_rank=5, after_rank=2,
    )
    prod = _make_recovery_attempt(
        "qrw", recovery_action="ATTEMPT_REWRITE",
        rewrite_mode="production_like", oracle_upper_bound=False,
        rewritten_query="prod-rw", before_rank=5, after_rank=4,
    )
    recovery_by_qid = {"qrw": [prod, oracle]}  # order should not matter
    audit_rows = build_audit_rows([base], recovery_by_qid=recovery_by_qid)
    row = audit_rows[0]
    assert row.recovery["rewrite_mode"] == "oracle"
    assert row.recovery["rewritten_query"] == "oracle-rw"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _bulk_confidence_rows(n_per_bucket: int = 30) -> List[Dict]:
    rows: List[Dict] = []
    serial = 0
    for bucket in ("main_work", "subpage_generic", "subpage_named"):
        for i in range(n_per_bucket):
            serial += 1
            qid = f"v4-silver-{serial:04d}"
            label = (
                "FAILED" if i == 0 else
                "LOW_CONFIDENCE" if i == 1 else
                "CONFIDENT" if i == 2 else
                "AMBIGUOUS"
            )
            reasons = (
                ["GOLD_NOT_IN_CANDIDATES", "TITLE_ALIAS_MISMATCH"] if label == "FAILED"
                else ["TITLE_ALIAS_MISMATCH"] if i % 7 == 0
                else ["GENERIC_COLLISION"] if i % 5 == 0
                else ["LOW_MARGIN"]
            )
            gold_in_topk = label != "FAILED"
            rows.append(_make_confidence_row(
                qid,
                bucket=bucket,
                confidence_label=label,
                failure_reasons=reasons,
                gold_in_top_k=gold_in_topk,
                expected_title=f"silver-title-{serial}",
            ))
    return rows


def test_sample_audit_rows_respects_bucket_quota():
    rows = build_audit_rows(_bulk_confidence_rows(20))
    selected, manifest = sample_audit_rows(
        rows,
        bucket_quota={"main_work": 10, "subpage_generic": 10,
                      "subpage_named": 10},
        edge_case_quota={},
    )
    by_bucket: Dict[str, int] = {}
    for r in selected:
        by_bucket[r.bucket] = by_bucket.get(r.bucket, 0) + 1
    assert by_bucket["main_work"] == 10
    assert by_bucket["subpage_generic"] == 10
    assert by_bucket["subpage_named"] == 10


def test_sample_audit_rows_includes_edge_cases():
    rows = build_audit_rows(_bulk_confidence_rows(20))
    # Disable bucket quota so the only rows are edge-case-driven.
    selected, manifest = sample_audit_rows(
        rows,
        bucket_quota={},
        edge_case_quota={
            "FAILED": 3,
            "TITLE_ALIAS_MISMATCH": 3,
            "GENERIC_COLLISION": 3,
        },
    )
    tags_seen: set = set()
    for r in selected:
        tags_seen.update(r.edge_case_tags)
    assert "FAILED" in tags_seen
    assert "TITLE_ALIAS_MISMATCH" in tags_seen
    assert "GENERIC_COLLISION" in tags_seen
    # Manifest reports per-tag picks.
    assert manifest["strat_groups"]["edge:FAILED"]["picked"] >= 1


def test_sample_audit_rows_dedupe_by_qid():
    rows = build_audit_rows(_bulk_confidence_rows(20))
    selected, manifest = sample_audit_rows(
        rows,
        bucket_quota={"main_work": 10, "subpage_generic": 10, "subpage_named": 10},
        edge_case_quota={
            "FAILED": 5, "AMBIGUOUS": 5, "TITLE_ALIAS_MISMATCH": 5,
        },
    )
    qids = [r.query_id for r in selected]
    assert len(qids) == len(set(qids))
    # n_audit_rows must match the deduped count.
    assert manifest["n_audit_rows"] == len(selected)
    assert manifest["n_unique_qids"] == len(set(qids))


def test_sample_audit_rows_assigns_sequential_audit_ids():
    rows = build_audit_rows(_bulk_confidence_rows(15))
    selected, _ = sample_audit_rows(
        rows,
        bucket_quota={"main_work": 5, "subpage_generic": 5, "subpage_named": 5},
        edge_case_quota={},
    )
    # IDs must be audit-0001, audit-0002, ... in qid-sorted order.
    expected = [f"audit-{i:04d}" for i in range(1, len(selected) + 1)]
    assert [r.audit_id for r in selected] == expected
    # And rows are sorted by qid.
    qids = [r.query_id for r in selected]
    assert qids == sorted(qids)


def test_sample_audit_rows_is_deterministic_across_runs():
    rows = build_audit_rows(_bulk_confidence_rows(20))
    selected_a, _ = sample_audit_rows(rows)
    selected_b, _ = sample_audit_rows(rows)
    a_ids = [r.audit_id for r in selected_a]
    b_ids = [r.audit_id for r in selected_b]
    a_qids = [r.query_id for r in selected_a]
    b_qids = [r.query_id for r in selected_b]
    assert a_ids == b_ids
    assert a_qids == b_qids


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_write_audit_jsonl_csv_md_have_same_row_count(tmp_path):
    rows = build_audit_rows(_bulk_confidence_rows(15))
    selected, manifest = sample_audit_rows(rows)

    jsonl_path = tmp_path / "x.jsonl"
    csv_path = tmp_path / "x.csv"
    md_path = tmp_path / "x.md"

    write_audit_jsonl(selected, jsonl_path)
    write_audit_csv(selected, csv_path)
    write_audit_md(selected, md_path, manifest=manifest)

    jsonl_rows = load_jsonl(jsonl_path)
    csv_rows = list(csv.DictReader(io.StringIO(
        csv_path.read_text(encoding="utf-8")
    )))
    md_text = md_path.read_text(encoding="utf-8")

    assert len(jsonl_rows) == len(selected)
    assert len(csv_rows) == len(selected)
    # Each audit_id appears in the markdown once as a heading.
    for row in selected:
        assert f"### {row.audit_id} — `{row.query_id}`" in md_text


def test_audit_md_carries_silver_disclaimer():
    rows = build_audit_rows(_bulk_confidence_rows(5))
    selected, manifest = sample_audit_rows(rows)
    text = render_audit_md(selected, manifest=manifest)
    assert SILVER_DISCLAIMER_MARKER in text


def test_audit_md_lists_human_label_choices():
    rows = build_audit_rows(_bulk_confidence_rows(5))
    selected, manifest = sample_audit_rows(rows)
    text = render_audit_md(selected, manifest=manifest)
    for choice in HUMAN_LABEL_CHOICES:
        assert choice in text


def test_audit_csv_columns_use_silver_terminology():
    """No column header should claim 'gold' / 'correct' / 'wrong'."""
    rows = build_audit_rows(_bulk_confidence_rows(5))
    selected, _ = sample_audit_rows(rows)
    out_path = Path("__test__.csv")  # not actually written; use string buffer
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=[
            "audit_id", "query_id", "query", "bucket",
            "silver_target_doc_id", "silver_target_page_id",
            "silver_expected_title", "confidence_label",
            "failure_reasons", "recommended_action",
            "top1_title", "top1_retrieval_title", "top1_doc_id",
            "top3_titles", "top5_chunk_snippets",
            "recovery_action", "rewrite_mode", "rewritten_query",
            "after_rank", "after_in_top_k", "edge_case_tags",
            "human_label", "human_notes",
        ],
    )
    writer.writeheader()
    headers = buf.getvalue().splitlines()[0]
    # No "gold" / "correct" / "wrong" in column names.
    assert "gold" not in headers.lower()
    assert "correct" not in headers.lower()
    assert "wrong" not in headers.lower()
    # "silver" appears in the silver-prefixed columns.
    assert "silver" in headers.lower()


def test_audit_jsonl_csv_md_are_byte_identical_across_runs(tmp_path):
    rows = build_audit_rows(_bulk_confidence_rows(20))

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    for out_dir in (out_a, out_b):
        selected, manifest = sample_audit_rows(rows)
        write_audit_jsonl(selected, out_dir / "x.jsonl")
        write_audit_csv(selected, out_dir / "x.csv")
        write_audit_md(selected, out_dir / "x.md", manifest=manifest)

    for filename in ("x.jsonl", "x.csv", "x.md"):
        a = (out_a / filename).read_bytes()
        b = (out_b / filename).read_bytes()
        assert a == b, f"{filename} non-deterministic"


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Sequence[Mapping]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def test_export_audit_bundle_e2e(tmp_path):
    confidence_rows = _bulk_confidence_rows(20)
    confidence_jsonl = tmp_path / "confidence.jsonl"
    _write_jsonl(confidence_jsonl, confidence_rows)

    # One recovery attempt for the second qid.
    rec_jsonl = tmp_path / "recovery.jsonl"
    rec_attempt = _make_recovery_attempt(
        confidence_rows[1]["query_id"],
        recovery_action="ATTEMPT_REWRITE",
        rewrite_mode="production_like",
        rewritten_query="x",
    )
    _write_jsonl(rec_jsonl, [rec_attempt])

    chunks_jsonl = _make_chunks_jsonl(tmp_path)

    out_dir = tmp_path / "audit_out"
    paths = export_audit_bundle(
        confidence_jsonl=confidence_jsonl,
        out_dir=out_dir,
        recovery_attempts_jsonl=rec_jsonl,
        chunks_jsonl=chunks_jsonl,
    )
    for role in ("jsonl", "csv", "md", "summary"):
        assert role in paths
        assert paths[role].exists()
    # Summary JSON carries the silver disclaimer block.
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "silver_disclaimer" in summary
    assert "silver_terminology_aliases" in summary
    assert summary["n_audit_rows"] >= 1


def test_export_audit_bundle_is_byte_deterministic(tmp_path):
    confidence_rows = _bulk_confidence_rows(20)
    confidence_jsonl = tmp_path / "confidence.jsonl"
    _write_jsonl(confidence_jsonl, confidence_rows)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    paths_a = export_audit_bundle(
        confidence_jsonl=confidence_jsonl, out_dir=out_a,
    )
    paths_b = export_audit_bundle(
        confidence_jsonl=confidence_jsonl, out_dir=out_b,
    )
    for role in ("jsonl", "csv", "md", "summary"):
        assert paths_a[role].read_bytes() == paths_b[role].read_bytes()


def test_export_audit_bundle_stratifies_by_bucket(tmp_path):
    confidence_rows = _bulk_confidence_rows(20)
    confidence_jsonl = tmp_path / "confidence.jsonl"
    _write_jsonl(confidence_jsonl, confidence_rows)

    out_dir = tmp_path / "out"
    paths = export_audit_bundle(
        confidence_jsonl=confidence_jsonl,
        out_dir=out_dir,
        config=AuditExportConfig(
            bucket_quota={"main_work": 5, "subpage_generic": 5,
                          "subpage_named": 5},
            edge_case_quota={},
            snippet_max_chars=200,
        ),
    )
    rows = load_jsonl(paths["jsonl"])
    by_bucket: Dict[str, int] = {}
    for r in rows:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    assert by_bucket["main_work"] == 5
    assert by_bucket["subpage_generic"] == 5
    assert by_bucket["subpage_named"] == 5
