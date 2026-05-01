"""Phase 7 — tests for the human-gold-seed audit exporter.

Fixture-driven: each test composes a small silver / per_query /
confidence / recovery / chunks bundle in tmp_path and runs the
exporter end-to-end. Inputs are kept tiny because the picker's
contracts are about ordering and stratification, not scale.

Acceptance bar (per the spec):

  - Output is deterministic under a fixed seed.
  - Exported set is stratified across buckets.
  - Edge cases (CONFIDENT / AMBIGUOUS / LOW_CONFIDENCE / FAILED /
    LOW_MARGIN / GENERIC_COLLISION / TITLE_ALIAS_MISMATCH /
    expected_target_not_in_candidates / QUERY_REWRITE / HYBRID_RECOVERY)
    are prioritised.
  - Every row's human_label / human_correct_title / etc fields are
    blank by default.
  - The Markdown report carries the silver/gold disclaimer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from eval.harness.human_gold_seed_export import (
    ALLOWED_HUMAN_LABELS,
    DEFAULT_BUCKET_TARGETS,
    EDGE_AMBIGUOUS,
    EDGE_CASES,
    EDGE_CONFIDENT,
    EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES,
    EDGE_FAILED,
    EDGE_GENERIC_COLLISION,
    EDGE_HYBRID_RECOVERY,
    EDGE_LOW_CONFIDENCE,
    EDGE_LOW_MARGIN,
    EDGE_QUERY_REWRITE,
    EDGE_TITLE_ALIAS_MISMATCH,
    HUMAN_GOLD_DISCLAIMER,
    HumanGoldSeedConfig,
    SeedRow,
    build_human_gold_seed,
    render_md,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _silver_record(
    qid: str,
    *,
    bucket: str,
    page_id: str,
    work_title: str = "테스트작품",
    page_title: Optional[str] = None,
    retrieval_title: Optional[str] = None,
    query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose one silver query record (silver-500 schema)."""
    pt = page_title or work_title
    rt = retrieval_title or pt
    q = query_text or f"{work_title}에 대해 알려주세요."
    return {
        "id": qid,
        "query": q,
        "language": "ko",
        "expected_doc_ids": [page_id],
        "expected_section_keywords": [work_title],
        "answer_type": "title_lookup",
        "difficulty": "easy",
        "tags": ["anime", "v4-silver-500", "silver", bucket],
        "v4_meta": {
            "bucket": bucket,
            "page_type": "work",
            "relation": "main",
            "page_title": pt,
            "work_title": work_title,
            "retrieval_title": rt,
            "title_source": "seed",
            "alias_source": "fallback",
            "is_generic_page_title": False,
            "silver_label_source": "page_lookup",
            "silver_label_confidence": "high",
            "is_silver_not_gold": True,
        },
    }


def _confidence_record(
    qid: str,
    *,
    bucket: str,
    label: str = "CONFIDENT",
    reasons: Optional[List[str]] = None,
    action: str = "ANSWER",
    top_chunks: Optional[List[Dict[str, Any]]] = None,
    expected_title: str = "테스트작품",
    gold_doc_id: str = "page-1",
) -> Dict[str, Any]:
    """Compose one Phase 7.3 confidence verdict row."""
    if top_chunks is None:
        top_chunks = [
            {
                "rank": 1, "chunk_id": "c1", "doc_id": "page-1",
                "title": "테스트작품", "retrieval_title": "테스트작품",
                "section_path": ["개요"], "section_type": "summary",
                "dense_score": 0.85,
            },
        ]
    return {
        "query_id": qid,
        "bucket": bucket,
        "confidence_label": label,
        "failure_reasons": reasons or [],
        "recommended_action": action,
        "signals": {},
        "debug_summary": "",
        "input": {
            "query_text": f"{expected_title} 쿼리",
            "gold_doc_id": gold_doc_id,
            "gold_page_id": None,
            "expected_title": expected_title,
            "expected_section_type": None,
            "candidate_count": len(top_chunks),
            "top_candidates_preview": top_chunks,
        },
    }


def _per_query_record(
    qid: str,
    *,
    bucket: str,
    expected_doc_ids: List[str],
    candidate_top_results: List[Dict[str, Any]],
    query_text: str = "쿼리",
) -> Dict[str, Any]:
    """Compose one Phase 7.0 per_query_comparison row."""
    return {
        "qid": qid,
        "query": query_text,
        "expected_doc_ids": expected_doc_ids,
        "bucket": bucket,
        "v4_meta": {"bucket": bucket},
        "status": "improved",
        "baseline": {"top_results": []},
        "candidate": {
            "rank": 1,
            "top_results": candidate_top_results,
        },
    }


def _recovery_record(
    qid: str,
    *,
    bucket: str,
    original_action: str,
    recovery_action: str,
    rewrite_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose one Phase 7.4 recovery_attempts row."""
    return {
        "decision": {
            "query_id": qid,
            "bucket": bucket,
            "original_action": original_action,
            "recovery_action": recovery_action,
            "skip_reason": None,
            "rewrite_mode": rewrite_mode,
            "oracle_upper_bound": (rewrite_mode == "oracle"),
            "query_text": f"{qid} text",
            "rewritten_query": None,
            "rewrite_terms": [],
            "rewrite_source": None,
            "expected_title": "테스트작품",
            "gold_doc_id": "page-1",
            "gold_page_id": None,
            "notes": [],
        },
        "before_rank": 5,
        "before_top_doc_ids": ["page-1"],
        "before_top_chunk_ids": ["c1"],
        "before_in_top_k": True,
        "before_top1_score": 0.6,
        "after_rank": 1,
        "after_top_doc_ids": ["page-1"],
        "after_top_chunk_ids": ["c1"],
        "after_in_top_k": True,
        "after_top1_score": 0.65,
        "final_k": 10,
        "latency_ms": 10.0,
        "error": None,
    }


def _chunks_record(chunk_id: str, *, page_id: str, title: str) -> Dict[str, Any]:
    """Compose one rag_chunks_*.jsonl record."""
    return {
        "chunk_id": chunk_id,
        "doc_id": page_id,
        "page_id": page_id,
        "title": title,
        "retrieval_title": title,
        "display_title": title,
        "section_path": ["개요"],
        "section_type": "summary",
        "chunk_text": f"{title}의 본문 텍스트가 여기에 들어갑니다.",
        "embedding_text": f"제목: {title}\n섹션: 개요\n섹션타입: summary\n\n본문:\n{title}",
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _build_full_bundle(tmp_path: Path) -> Dict[str, Path]:
    """Build a comprehensive fixture set covering every edge case.

    Each query is constructed to fire exactly one or two edge flags so
    the picker has a deterministic candidate for every edge in the
    canonical edge order. The bucket ratio is roughly proportional to
    the default config (4:8:8 here) so a target_total of 20 has enough
    headroom to satisfy targets + edge top-up.
    """
    silver: List[Dict[str, Any]] = []
    confidence: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []
    recovery: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    def _add_chunk(cid: str, page_id: str, title: str) -> None:
        chunks.append(_chunks_record(cid, page_id=page_id, title=title))

    # main_work pool — enough rows to cover edges + bucket target.
    for i in range(8):
        qid = f"q-main-{i:02d}"
        page_id = f"page-main-{i:02d}"
        cid = f"c-main-{i:02d}"
        title = f"메인작품{i:02d}"
        silver.append(_silver_record(qid, bucket="main_work", page_id=page_id,
                                     work_title=title))
        _add_chunk(cid, page_id, title)
        # Even index → CONFIDENT, odd → AMBIGUOUS with LOW_MARGIN.
        if i % 2 == 0:
            confidence.append(_confidence_record(
                qid, bucket="main_work", label="CONFIDENT",
                action="ANSWER",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": title, "retrieval_title": title,
                    "section_path": ["개요"], "section_type": "summary",
                    "dense_score": 0.90,
                }],
                expected_title=title, gold_doc_id=page_id,
            ))
        else:
            confidence.append(_confidence_record(
                qid, bucket="main_work", label="AMBIGUOUS",
                reasons=["LOW_MARGIN"],
                action="ANSWER_WITH_CAUTION",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": title, "retrieval_title": title,
                    "section_path": ["개요"], "section_type": "summary",
                    "dense_score": 0.80,
                }],
                expected_title=title, gold_doc_id=page_id,
            ))
        per_query.append(_per_query_record(
            qid, bucket="main_work",
            expected_doc_ids=[page_id],
            candidate_top_results=[{
                "chunk_id": cid, "doc_id": page_id, "section": "개요", "score": 0.85,
            }],
        ))

    # subpage_generic pool — fire LOW_CONFIDENCE / GENERIC_COLLISION /
    # HYBRID_RECOVERY edges.
    for i in range(8):
        qid = f"q-gen-{i:02d}"
        page_id = f"page-gen-{i:02d}"
        cid = f"c-gen-{i:02d}"
        title = f"서브작품{i:02d}"
        silver.append(_silver_record(
            qid, bucket="subpage_generic", page_id=page_id,
            work_title=title, page_title="등장인물",
            retrieval_title=f"{title} / 등장인물",
        ))
        _add_chunk(cid, page_id, "등장인물")
        if i == 0:
            # LOW_CONFIDENCE with GENERIC_COLLISION + HYBRID_RECOVERY action.
            confidence.append(_confidence_record(
                qid, bucket="subpage_generic", label="LOW_CONFIDENCE",
                reasons=["LOW_TOP1_SCORE", "GENERIC_COLLISION"],
                action="HYBRID_RECOVERY",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": "등장인물",
                    "retrieval_title": f"{title} / 등장인물",
                    "section_path": ["등장인물"], "section_type": "character",
                    "dense_score": 0.40,
                }],
                expected_title=f"{title} / 등장인물", gold_doc_id=page_id,
            ))
            recovery.append(_recovery_record(
                qid, bucket="subpage_generic",
                original_action="HYBRID_RECOVERY",
                recovery_action="ATTEMPT_HYBRID",
            ))
        elif i == 1:
            # FAILED with expected_target_not_in_candidates.
            confidence.append(_confidence_record(
                qid, bucket="subpage_generic", label="FAILED",
                reasons=["GOLD_NOT_IN_CANDIDATES"],
                action="INSUFFICIENT_EVIDENCE",
                top_chunks=[{
                    "rank": 1, "chunk_id": "c-other-1", "doc_id": "page-other-1",
                    "title": "다른 작품", "retrieval_title": "다른 작품",
                    "section_path": ["개요"], "section_type": "summary",
                    "dense_score": 0.50,
                }],
                expected_title=f"{title} / 등장인물", gold_doc_id=page_id,
            ))
            _add_chunk("c-other-1", "page-other-1", "다른 작품")
        else:
            confidence.append(_confidence_record(
                qid, bucket="subpage_generic", label="CONFIDENT",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": "등장인물",
                    "retrieval_title": f"{title} / 등장인물",
                    "section_path": ["등장인물"], "section_type": "character",
                    "dense_score": 0.85,
                }],
                expected_title=f"{title} / 등장인물", gold_doc_id=page_id,
            ))
        per_query.append(_per_query_record(
            qid, bucket="subpage_generic",
            expected_doc_ids=[page_id],
            candidate_top_results=[{
                "chunk_id": cid, "doc_id": page_id,
                "section": "등장인물", "score": 0.50,
            }],
        ))

    # subpage_named pool — fire TITLE_ALIAS_MISMATCH / QUERY_REWRITE.
    for i in range(8):
        qid = f"q-named-{i:02d}"
        page_id = f"page-named-{i:02d}"
        cid = f"c-named-{i:02d}"
        title = f"네임드{i:02d}"
        silver.append(_silver_record(
            qid, bucket="subpage_named", page_id=page_id,
            work_title=f"네임드작품{i:02d}", page_title=title,
            retrieval_title=f"네임드작품{i:02d} / {title}",
        ))
        _add_chunk(cid, page_id, title)
        if i == 0:
            # AMBIGUOUS with TITLE_ALIAS_MISMATCH + QUERY_REWRITE action.
            confidence.append(_confidence_record(
                qid, bucket="subpage_named", label="LOW_CONFIDENCE",
                reasons=["TITLE_ALIAS_MISMATCH"],
                action="QUERY_REWRITE",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": title, "retrieval_title": "잘못된 제목",
                    "section_path": ["주인공"], "section_type": "character",
                    "dense_score": 0.55,
                }],
                expected_title=f"네임드작품{i:02d} / {title}",
                gold_doc_id=page_id,
            ))
            recovery.append(_recovery_record(
                qid, bucket="subpage_named",
                original_action="QUERY_REWRITE",
                recovery_action="ATTEMPT_REWRITE",
                rewrite_mode="production_like",
            ))
        else:
            confidence.append(_confidence_record(
                qid, bucket="subpage_named", label="CONFIDENT",
                top_chunks=[{
                    "rank": 1, "chunk_id": cid, "doc_id": page_id,
                    "title": title,
                    "retrieval_title": f"네임드작품{i:02d} / {title}",
                    "section_path": ["주인공"], "section_type": "character",
                    "dense_score": 0.85,
                }],
                expected_title=f"네임드작품{i:02d} / {title}",
                gold_doc_id=page_id,
            ))
        per_query.append(_per_query_record(
            qid, bucket="subpage_named",
            expected_doc_ids=[page_id],
            candidate_top_results=[{
                "chunk_id": cid, "doc_id": page_id,
                "section": "주인공", "score": 0.6,
            }],
        ))

    return {
        "silver": _write_jsonl(tmp_path / "silver.jsonl", silver),
        "confidence": _write_jsonl(tmp_path / "confidence.jsonl", confidence),
        "per_query": _write_jsonl(tmp_path / "per_query.jsonl", per_query),
        "recovery": _write_jsonl(tmp_path / "recovery.jsonl", recovery),
        "chunks": _write_jsonl(tmp_path / "chunks.jsonl", chunks),
    }


# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------


def test_default_bucket_targets_match_spec():
    """Spec-mandated default split: main=10, generic=20, named=20."""
    assert DEFAULT_BUCKET_TARGETS["main_work"] == 10
    assert DEFAULT_BUCKET_TARGETS["subpage_generic"] == 20
    assert DEFAULT_BUCKET_TARGETS["subpage_named"] == 20
    assert sum(DEFAULT_BUCKET_TARGETS.values()) == 50


def test_allowed_human_labels_match_spec():
    expected = {
        "SUPPORTED", "PARTIALLY_SUPPORTED", "WRONG_TARGET",
        "AMBIGUOUS_QUERY", "NOT_IN_CORPUS", "BAD_SILVER_LABEL",
    }
    assert set(ALLOWED_HUMAN_LABELS) == expected


def test_edge_cases_cover_spec_list():
    """Every edge case the spec asks for is in the EDGE_CASES tuple."""
    must_have = {
        "CONFIDENT", "AMBIGUOUS", "LOW_CONFIDENCE", "FAILED",
        "LOW_MARGIN", "GENERIC_COLLISION", "TITLE_ALIAS_MISMATCH",
        "expected_target_not_in_candidates",
        "QUERY_REWRITE", "HYBRID_RECOVERY",
    }
    assert must_have.issubset(set(EDGE_CASES))


def test_config_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        HumanGoldSeedConfig(target_total=0).validate()
    with pytest.raises(ValueError):
        HumanGoldSeedConfig(side="bogus").validate()
    with pytest.raises(ValueError):
        HumanGoldSeedConfig(
            bucket_targets={"main_work": -1, "subpage_generic": 0,
                             "subpage_named": 0}
        ).validate()


# ---------------------------------------------------------------------------
# End-to-end determinism + stratification
# ---------------------------------------------------------------------------


def test_human_gold_seed_export_is_deterministic(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=20,
        bucket_targets={"main_work": 4, "subpage_generic": 8,
                        "subpage_named": 8},
        seed=42,
    ).validate()
    e1 = build_human_gold_seed(
        silver_path=fx["silver"],
        confidence_path=fx["confidence"],
        per_query_path=fx["per_query"],
        recovery_path=fx["recovery"],
        chunks_path=fx["chunks"],
        config=cfg,
    )
    e2 = build_human_gold_seed(
        silver_path=fx["silver"],
        confidence_path=fx["confidence"],
        per_query_path=fx["per_query"],
        recovery_path=fx["recovery"],
        chunks_path=fx["chunks"],
        config=cfg,
    )
    assert [r.query_id for r in e1.rows] == [r.query_id for r in e2.rows]
    assert e1.audit_summary == e2.audit_summary


def test_human_gold_seed_export_jsonl_is_byte_stable(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=20,
        bucket_targets={"main_work": 4, "subpage_generic": 8,
                        "subpage_named": 8},
        seed=42,
    ).validate()
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    e1 = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    e2 = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    p1 = write_outputs(e1, out_dir=out_a, target_total=cfg.target_total)
    p2 = write_outputs(e2, out_dir=out_b, target_total=cfg.target_total)
    assert p1["jsonl"].read_bytes() == p2["jsonl"].read_bytes()
    assert p1["csv"].read_bytes() == p2["csv"].read_bytes()
    assert p1["md"].read_bytes() == p2["md"].read_bytes()


def test_human_gold_seed_export_is_stratified_across_buckets(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=20,
        bucket_targets={"main_work": 4, "subpage_generic": 8,
                        "subpage_named": 8},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    counts: Dict[str, int] = {}
    for r in e.rows:
        counts[r.bucket] = counts.get(r.bucket, 0) + 1
    # Pool has 8 candidates per bucket, so all targets are reachable.
    assert counts.get("main_work", 0) == 4
    assert counts.get("subpage_generic", 0) == 8
    assert counts.get("subpage_named", 0) == 8


def test_human_gold_seed_export_prioritises_edge_cases(tmp_path: Path):
    """Every spec edge case must be present at least once when any
    candidate exhibits it."""
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=20,
        bucket_targets={"main_work": 4, "subpage_generic": 8,
                        "subpage_named": 8},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    flags: set = set()
    for r in e.rows:
        flags.update(r.edge_flags)
    expected_flags = {
        EDGE_CONFIDENT, EDGE_AMBIGUOUS, EDGE_LOW_CONFIDENCE, EDGE_FAILED,
        EDGE_LOW_MARGIN, EDGE_GENERIC_COLLISION, EDGE_TITLE_ALIAS_MISMATCH,
        EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES,
        EDGE_QUERY_REWRITE, EDGE_HYBRID_RECOVERY,
    }
    assert expected_flags.issubset(flags), (
        f"missing edges: {expected_flags - flags}"
    )


# ---------------------------------------------------------------------------
# Row schema + blank human fields
# ---------------------------------------------------------------------------


def test_human_label_fields_are_blank_by_default(tmp_path: Path):
    """All five human_* fields must be blank in the exported rows."""
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    for r in e.rows:
        assert r.human_label == ""
        assert r.human_correct_title == ""
        assert r.human_correct_page_id == ""
        assert r.human_supporting_chunk_id == ""
        assert r.human_notes == ""


def test_human_gold_seed_jsonl_carries_required_fields(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    paths = write_outputs(e, out_dir=tmp_path / "out",
                          target_total=cfg.target_total)
    rows: List[Dict[str, Any]] = []
    for line in paths["jsonl"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    required = {
        "query_id", "query", "bucket", "silver_expected_title",
        "silver_expected_page_id", "top1_title", "top1_page_id",
        "top3_titles", "top5_titles", "top5_snippets",
        "confidence_label", "failure_reasons", "recommended_action",
        "recovery_action", "rewrite_mode", "notes_for_reviewer",
        "human_label", "human_correct_title", "human_correct_page_id",
        "human_supporting_chunk_id", "human_notes",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"missing fields in row {r['query_id']}: {missing}"
        # Blank human fields:
        for k in (
            "human_label", "human_correct_title", "human_correct_page_id",
            "human_supporting_chunk_id", "human_notes",
        ):
            assert r[k] == ""


def test_human_gold_seed_csv_has_header_and_blank_human_columns(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    paths = write_outputs(e, out_dir=tmp_path / "out",
                          target_total=cfg.target_total)
    with paths["csv"].open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        header = list(reader.fieldnames or ())
        for col in (
            "query_id", "query", "bucket", "silver_expected_title",
            "silver_expected_page_id", "top1_title", "top1_page_id",
            "human_label", "human_correct_title", "human_correct_page_id",
            "human_supporting_chunk_id", "human_notes",
        ):
            assert col in header, f"missing CSV column: {col}"
        for row in reader:
            for k in (
                "human_label", "human_correct_title",
                "human_correct_page_id",
                "human_supporting_chunk_id", "human_notes",
            ):
                assert row[k] == ""


def test_human_gold_seed_md_carries_silver_disclaimer(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    md = render_md(e, target_total=cfg.target_total)
    assert HUMAN_GOLD_DISCLAIMER in md
    # The disclaimer references "human-verified gold" — must say it's not.
    assert "NOT human-verified gold" in md
    # Allowed-label list must surface for the reviewer.
    for lab in ALLOWED_HUMAN_LABELS:
        assert lab in md


def test_human_gold_seed_export_filenames_match_spec(tmp_path: Path):
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"], confidence_path=fx["confidence"],
        per_query_path=fx["per_query"], recovery_path=fx["recovery"],
        chunks_path=fx["chunks"], config=cfg,
    )
    paths = write_outputs(
        e, out_dir=tmp_path / "out",
        base_name="phase7_human_gold_seed_50",
        target_total=cfg.target_total,
    )
    assert paths["jsonl"].name == "phase7_human_gold_seed_50.jsonl"
    assert paths["csv"].name == "phase7_human_gold_seed_50.csv"
    assert paths["md"].name == "phase7_human_gold_seed_50.md"


# ---------------------------------------------------------------------------
# Optional inputs
# ---------------------------------------------------------------------------


def test_export_works_without_per_query_or_recovery(tmp_path: Path):
    """When per_query / recovery / chunks are missing, the exporter still
    produces a valid seed — the corresponding fields just stay blank."""
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=10,
        bucket_targets={"main_work": 4, "subpage_generic": 3,
                        "subpage_named": 3},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"],
        confidence_path=fx["confidence"],
        per_query_path=None,
        recovery_path=None,
        chunks_path=None,
        config=cfg,
    )
    assert e.rows
    # No recovery file → every row has recovery_action / rewrite_mode None.
    for r in e.rows:
        assert r.recovery_action is None
        assert r.rewrite_mode is None


def test_export_works_with_only_silver(tmp_path: Path):
    """Most degenerate: silver only. Edge cases drop to just
    expected_target_not_in_candidates (when expected page_id is set
    but no candidates are loaded)."""
    fx = _build_full_bundle(tmp_path)
    cfg = HumanGoldSeedConfig(
        target_total=5,
        bucket_targets={"main_work": 2, "subpage_generic": 2,
                        "subpage_named": 1},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=fx["silver"],
        per_query_path=None, confidence_path=None,
        recovery_path=None, chunks_path=None,
        config=cfg,
    )
    assert len(e.rows) == 5
    for r in e.rows:
        assert r.confidence_label is None
        assert r.failure_reasons == []
        assert r.recovery_action is None
        # silver expected target was never seen in candidates (none loaded)
        assert EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES in r.edge_flags


# ---------------------------------------------------------------------------
# Edge fall-back when targets exceed pool
# ---------------------------------------------------------------------------


def test_export_reports_deficit_when_pool_is_empty(tmp_path: Path):
    """No silver records → empty seed and reported deficit."""
    silver_path = _write_jsonl(tmp_path / "silver.jsonl", [])
    cfg = HumanGoldSeedConfig(
        target_total=5,
        bucket_targets={"main_work": 2, "subpage_generic": 2,
                        "subpage_named": 1},
        seed=42,
    ).validate()
    e = build_human_gold_seed(
        silver_path=silver_path, config=cfg,
    )
    assert e.rows == []
    audit = e.audit_summary
    assert audit["actual_total"] == 0
    assert all(d > 0 for d in audit["bucket_deficits"].values())
