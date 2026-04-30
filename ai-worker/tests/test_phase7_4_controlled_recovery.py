"""Phase 7.4 — tests for the controlled recovery loop and metrics.

Tests build a tiny in-memory BM25 index over a synthetic chunk list,
hand-craft a frozen dense state per qid, and feed the loop a small
verdict batch covering every Phase 7.3 action class.

Coverage:

  - HYBRID_RECOVERY runs BM25 + dense RRF fusion.
  - QUERY_REWRITE oracle / production_like branches both produce
    rewritten queries, with the oracle row marked oracle_upper_bound.
  - INSUFFICIENT_EVIDENCE / ANSWER_WITH_CAUTION rows skip without
    running BM25.
  - 'both' mode fans out QUERY_REWRITE rows into two attempts.
  - Recovery metrics correctly count recovered, regressed,
    newly-entered, skipped.
  - JSONL / md writers are deterministic across two runs.
  - oracle_rewrite_upper_bound.jsonl is only emitted when oracle ran.
  - When BM25 retriever is None, ATTEMPT rows record an error and
    do not crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from eval.harness.bm25_retriever import (
    BM25EvalRetriever,
    build_bm25_index,
)
from eval.harness.controlled_recovery_loop import (
    ControlledRecoveryConfig,
    ControlledRecoveryResult,
    FrozenDenseRow,
    fuse_dense_and_bm25,
    load_chunks_for_bm25,
    load_frozen_dense_state,
    load_verdict_rows,
    run_controlled_recovery,
)
from eval.harness.embedding_text_builder import VARIANT_RAW
from eval.harness.recovery_metrics import (
    aggregate_results,
    render_final_report_md,
    render_summary_md,
    write_outputs,
)
from eval.harness.recovery_policy import (
    RECOVERY_ACTION_ATTEMPT_HYBRID,
    RECOVERY_ACTION_ATTEMPT_REWRITE,
    RECOVERY_ACTION_SKIP_CAUTION,
    RECOVERY_ACTION_SKIP_REFUSE,
    REWRITE_MODE_BOTH,
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    *,
    text: str,
    title: str = "",
    section: str = "",
    keywords: Sequence[str] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section=section,
        text=text,
        title=title,
        keywords=tuple(keywords),
    )


def _make_bm25_retriever() -> BM25EvalRetriever:
    """A tiny BM25 corpus where 'recovery_target' lives on doc-RECOVER.

    The dense top-N (frozen) deliberately excludes doc-RECOVER, so a
    BM25 query for "recovery_target" should recover gold via the RRF
    fuse.
    """
    chunks = [
        _make_chunk(
            "c1", "doc-A",
            text="alpha bravo charlie",
            title="title A",
        ),
        _make_chunk(
            "c2", "doc-A",
            text="alpha bravo delta",
            title="title A",
        ),
        _make_chunk(
            "c3", "doc-B",
            text="echo foxtrot",
            title="title B",
        ),
        _make_chunk(
            "c-recover", "doc-RECOVER",
            text="recovery_target term needs surfacing",
            title="title RECOVER",
        ),
        _make_chunk(
            "c-other", "doc-OTHER",
            text="other unrelated terms",
            title="title OTHER",
        ),
    ]
    index = build_bm25_index(chunks, embedding_text_variant=VARIANT_RAW)
    return BM25EvalRetriever(index, top_k=20, name="bm25-test")


def _frozen_dense_for(qid: str, *, doc_ids: Sequence[str]) -> FrozenDenseRow:
    return FrozenDenseRow(
        qid=qid,
        query_text="placeholder",
        expected_doc_ids=("doc-RECOVER",),
        top_chunk_ids=tuple(f"c-dense-{i}" for i, _ in enumerate(doc_ids)),
        top_doc_ids=tuple(doc_ids),
        top1_score=0.6,
    )


def _verdict(
    qid: str,
    action: str,
    *,
    expected_title: str | None = None,
    gold_doc_id: str | None = "doc-RECOVER",
    candidates: Sequence[Dict[str, Any]] | None = None,
    query_text: str = "recovery_target",
) -> Dict[str, Any]:
    if candidates is None:
        candidates = [
            {
                "rank": 1,
                "chunk_id": "c1",
                "doc_id": "doc-A",
                "title": "title A",
                "retrieval_title": "title A",
                "section_path": ["개요"],
                "section_type": "summary",
            },
            {
                "rank": 2,
                "chunk_id": "c3",
                "doc_id": "doc-B",
                "title": "title B",
                "retrieval_title": "title B",
                "section_path": ["등장인물"],
                "section_type": "character",
            },
        ]
    return {
        "query_id": qid,
        "bucket": "main_work",
        "confidence_label": "AMBIGUOUS",
        "failure_reasons": ["LOW_MARGIN"],
        "recommended_action": action,
        "signals": {},
        "input": {
            "query_text": query_text,
            "gold_doc_id": gold_doc_id,
            "gold_page_id": None,
            "expected_title": expected_title,
            "expected_section_type": None,
            "candidate_count": len(list(candidates)),
            "top_candidates_preview": list(candidates),
        },
    }


def _config(
    *,
    rewrite_mode: str = REWRITE_MODE_PRODUCTION_LIKE,
    strict_label_leakage: bool = False,
    final_k: int = 10,
    bm25_pool_size: int = 50,
) -> ControlledRecoveryConfig:
    return ControlledRecoveryConfig(
        rewrite_mode=rewrite_mode,
        final_k=final_k,
        hybrid_top_k=final_k,
        bm25_pool_size=bm25_pool_size,
        strict_label_leakage=strict_label_leakage,
    ).validate()


# ---------------------------------------------------------------------------
# Configuration sanity
# ---------------------------------------------------------------------------


def test_config_validates():
    cfg = ControlledRecoveryConfig(rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert cfg.validate() is cfg


def test_config_rejects_unknown_rewrite_mode():
    with pytest.raises(ValueError):
        ControlledRecoveryConfig(rewrite_mode="WHAT").validate()


def test_config_rejects_negative_pool_size():
    with pytest.raises(ValueError):
        ControlledRecoveryConfig(bm25_pool_size=0).validate()


def test_config_rejects_invalid_side():
    with pytest.raises(ValueError):
        ControlledRecoveryConfig(side="middle").validate()


# ---------------------------------------------------------------------------
# Hybrid recovery surfaces gold via BM25
# ---------------------------------------------------------------------------


def test_hybrid_recovery_brings_gold_into_top_k_via_rrf():
    """BM25 finds 'recovery_target' on doc-RECOVER; RRF surfaces it."""
    bm25 = _make_bm25_retriever()
    rows = [_verdict("q1", "HYBRID_RECOVERY")]
    frozen = {
        "q1": _frozen_dense_for(
            "q1",
            doc_ids=("doc-A", "doc-A", "doc-B"),
        ),
    }
    cfg = _config(rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    assert len(result.attempts) == 1
    a = result.attempts[0]
    assert a.decision.recovery_action == RECOVERY_ACTION_ATTEMPT_HYBRID
    # before: dense had no doc-RECOVER → rank -1
    assert a.before_rank == -1
    # after: doc-RECOVER must show up in the fused list
    assert "doc-RECOVER" in a.after_top_doc_ids
    # The corresponding RecoveryResult flags this as recovered.
    r = result.results[0]
    assert r.recovered is True
    assert r.gold_newly_entered_candidates is True


def test_hybrid_recovery_preserves_existing_gold_position():
    """If gold was already at rank 1, recovery should not regress it."""
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "HYBRID_RECOVERY",
        gold_doc_id="doc-A",  # gold IS in dense already
    )]
    frozen = {
        "q1": _frozen_dense_for(
            "q1", doc_ids=("doc-A", "doc-B", "doc-OTHER"),
        ),
    }
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    a = result.attempts[0]
    assert a.before_rank == 1
    # The fuse should keep doc-A near the top.
    assert a.after_rank > 0
    r = result.results[0]
    assert r.regression is False


# ---------------------------------------------------------------------------
# Query rewrite: oracle vs production-like
# ---------------------------------------------------------------------------


def test_query_rewrite_oracle_is_marked_oracle_upper_bound():
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title="title RECOVER",
        candidates=[
            {"title": "title A", "retrieval_title": "title A",
             "section_path": ["s"], "chunk_id": "c1", "doc_id": "doc-A"},
        ],
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(rewrite_mode=REWRITE_MODE_ORACLE)
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    assert len(result.attempts) == 1
    a = result.attempts[0]
    assert a.decision.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
    assert a.decision.oracle_upper_bound is True
    assert a.decision.rewrite_mode == REWRITE_MODE_ORACLE
    # Oracle's rewritten query must include the expected_title token.
    assert a.decision.rewritten_query is not None
    assert "title RECOVER" in a.decision.rewritten_query


def test_query_rewrite_production_like_does_not_leak_expected_title():
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title="title RECOVER",
        candidates=[
            {"title": "title B", "retrieval_title": "title B",
             "section_path": ["s"], "chunk_id": "c3", "doc_id": "doc-B"},
        ],
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(
        rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE,
        strict_label_leakage=False,  # for the gather to run
    )
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    a = result.attempts[0]
    assert a.decision.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
    assert a.decision.oracle_upper_bound is False
    if a.decision.rewritten_query is not None:
        # Production-like is not allowed to include expected_title.
        assert "title RECOVER" not in a.decision.rewritten_query


def test_query_rewrite_strict_leakage_skips_to_defer():
    """Strict mode + expected_title leak → loop catches and emits SKIP_DEFER."""
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title="title RECOVER",
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(
        rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE,
        strict_label_leakage=True,
    )
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    a = result.attempts[0]
    assert a.decision.skip_reason == "LABEL_LEAKAGE_REFUSED"
    # Skip rows have no latency.
    assert a.latency_ms is None
    r = result.results[0]
    assert r.skipped is True
    assert r.recovered is False


# ---------------------------------------------------------------------------
# 'both' mode fan-out
# ---------------------------------------------------------------------------


def test_both_mode_fans_out_query_rewrite_into_two_attempts():
    """A single QUERY_REWRITE row should produce two attempts under 'both'."""
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title=None,  # avoid leakage refusal
        candidates=[
            {"title": "title B", "retrieval_title": "title B",
             "chunk_id": "c3", "doc_id": "doc-B"},
        ],
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(rewrite_mode=REWRITE_MODE_BOTH)
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    assert len(result.attempts) == 2
    modes = {a.decision.rewrite_mode for a in result.attempts}
    assert modes == {REWRITE_MODE_ORACLE, REWRITE_MODE_PRODUCTION_LIKE}


def test_both_mode_does_not_fan_out_non_rewrite_actions():
    """HYBRID rows in 'both' mode emit one attempt only."""
    bm25 = _make_bm25_retriever()
    rows = [_verdict("q1", "HYBRID_RECOVERY")]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(rewrite_mode=REWRITE_MODE_BOTH)
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    assert len(result.attempts) == 1


# ---------------------------------------------------------------------------
# Skip semantics
# ---------------------------------------------------------------------------


def test_insufficient_evidence_does_not_invoke_bm25():
    """A SKIP_REFUSE attempt must not run any BM25 retrieval."""
    bm25_calls = []

    class _StubBM25:
        @property
        def top_k(self) -> int:
            return 10

        def retrieve(self, query: str):
            bm25_calls.append(query)
            return SimpleNamespace(results=[])

    rows = [_verdict("q1", "INSUFFICIENT_EVIDENCE")]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=_StubBM25(),  # type: ignore[arg-type]
        config=cfg,
    )
    assert bm25_calls == []
    a = result.attempts[0]
    assert a.decision.recovery_action == RECOVERY_ACTION_SKIP_REFUSE


def test_answer_with_caution_does_not_invoke_bm25():
    bm25_calls = []

    class _StubBM25:
        @property
        def top_k(self) -> int:
            return 10

        def retrieve(self, query: str):
            bm25_calls.append(query)
            return SimpleNamespace(results=[])

    rows = [_verdict("q1", "ANSWER_WITH_CAUTION")]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=_StubBM25(),  # type: ignore[arg-type]
        config=cfg,
    )
    assert bm25_calls == []
    assert result.attempts[0].decision.recovery_action == RECOVERY_ACTION_SKIP_CAUTION


# ---------------------------------------------------------------------------
# Missing BM25 retriever
# ---------------------------------------------------------------------------


def test_attempt_without_bm25_records_error():
    rows = [_verdict("q1", "HYBRID_RECOVERY")]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=None,
        config=cfg,
    )
    a = result.attempts[0]
    assert a.error == "bm25_retriever_unavailable"
    assert a.after_rank == a.before_rank  # unchanged


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def test_aggregate_counts_match_inputs():
    bm25 = _make_bm25_retriever()
    rows = [
        _verdict("q1", "HYBRID_RECOVERY"),
        _verdict("q2", "INSUFFICIENT_EVIDENCE"),
        _verdict("q3", "ANSWER_WITH_CAUTION"),
        _verdict("q4", "ANSWER"),
    ]
    frozen = {q: _frozen_dense_for(q, doc_ids=("doc-A",)) for q in
              ("q1", "q2", "q3", "q4")}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    agg = aggregate_results(result)
    totals = agg["totals"]
    assert totals["n_decisions"] == 4
    assert totals["skipped"] == 3  # INSUFFICIENT, CAUTION, ANSWER
    assert totals["attempted"] == 1  # only HYBRID

    invariants = agg["invariants"]
    assert invariants["insufficient_evidence_recovered"] is False
    assert invariants["insufficient_evidence_refused_count"] == 1
    assert invariants["answer_with_caution_recovered"] is False
    assert invariants["answer_with_caution_skip_count"] == 1


def test_aggregate_oracle_vs_production_like_block_only_when_both_present():
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title=None,
        candidates=[
            {"title": "title B", "retrieval_title": "title B",
             "chunk_id": "c3", "doc_id": "doc-B"},
        ],
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    # Single mode: no comparison block.
    cfg_single = _config(rewrite_mode=REWRITE_MODE_ORACLE)
    res_single = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg_single,
    )
    agg_single = aggregate_results(res_single)
    assert agg_single["oracle_vs_production_like"] is None

    # Both: comparison block populated.
    cfg_both = _config(rewrite_mode=REWRITE_MODE_BOTH)
    res_both = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg_both,
    )
    agg_both = aggregate_results(res_both)
    assert agg_both["oracle_vs_production_like"] is not None
    delta = agg_both["oracle_vs_production_like"]["delta"]
    assert "recovered_oracle_minus_production_like" in delta


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_jsonl_output_is_deterministic_across_runs(tmp_path):
    """Two runs of the same loop should produce identical artefact bytes."""
    bm25 = _make_bm25_retriever()
    rows = [
        _verdict("q1", "HYBRID_RECOVERY"),
        _verdict("q2", "INSUFFICIENT_EVIDENCE"),
        _verdict("q3", "QUERY_REWRITE", expected_title=None,
                 candidates=[{"title": "title B", "chunk_id": "c3",
                              "doc_id": "doc-B"}]),
    ]
    frozen = {q: _frozen_dense_for(q, doc_ids=("doc-A",))
              for q in ("q1", "q2", "q3")}
    cfg = _config(rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    for out_dir in (out_a, out_b):
        result = run_controlled_recovery(
            verdict_rows=rows,
            frozen_dense_by_qid=frozen,
            bm25_retriever=bm25,
            config=cfg,
            # Use a fixed clock so latency does not perturb determinism;
            # the writer rounds to 3 decimals so a constant clock makes
            # latency_ms identical across runs.
            clock=lambda: 0.0,
        )
        write_outputs(result, out_dir=out_dir)

    for filename in (
        "recovery_attempts.jsonl",
        "recovery_summary.json",
        "recovered_queries.jsonl",
        "unrecovered_queries.jsonl",
        "recovery_summary.md",
        "PHASE7_4_FINAL_REPORT.md",
    ):
        a_text = (out_a / filename).read_text(encoding="utf-8")
        b_text = (out_b / filename).read_text(encoding="utf-8")
        assert a_text == b_text, f"{filename} is non-deterministic"


def test_oracle_jsonl_only_emitted_when_oracle_ran(tmp_path):
    bm25 = _make_bm25_retriever()
    rows = [_verdict("q1", "HYBRID_RECOVERY")]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)

    result = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg,
    )
    paths = write_outputs(result, out_dir=tmp_path)
    assert "oracle" not in paths
    assert not (tmp_path / "oracle_rewrite_upper_bound.jsonl").exists()


def test_oracle_jsonl_emitted_when_oracle_attempt_present(tmp_path):
    bm25 = _make_bm25_retriever()
    rows = [_verdict(
        "q1", "QUERY_REWRITE",
        expected_title="title RECOVER",
        candidates=[{"title": "title B", "chunk_id": "c3",
                     "doc_id": "doc-B"}],
    )]
    frozen = {"q1": _frozen_dense_for("q1", doc_ids=("doc-A",))}
    cfg = _config(rewrite_mode=REWRITE_MODE_ORACLE)

    result = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg,
    )
    paths = write_outputs(result, out_dir=tmp_path)
    assert "oracle" in paths
    assert paths["oracle"].exists()
    rows_out = [
        json.loads(line) for line in
        paths["oracle"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows_out) == 1
    assert rows_out[0]["decision"]["oracle_upper_bound"] is True


# ---------------------------------------------------------------------------
# Output file shape
# ---------------------------------------------------------------------------


def test_write_outputs_produces_required_files(tmp_path):
    bm25 = _make_bm25_retriever()
    rows = [
        _verdict("q1", "HYBRID_RECOVERY"),
        _verdict("q2", "INSUFFICIENT_EVIDENCE"),
    ]
    frozen = {q: _frozen_dense_for(q, doc_ids=("doc-A",)) for q in ("q1", "q2")}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg,
    )
    paths = write_outputs(result, out_dir=tmp_path)
    for role in ("attempts", "summary_json", "summary_md",
                 "recovered", "unrecovered", "final_report"):
        assert role in paths
        assert paths[role].exists()


def test_summary_md_contains_required_sections(tmp_path):
    bm25 = _make_bm25_retriever()
    rows = [
        _verdict("q1", "HYBRID_RECOVERY"),
        _verdict("q2", "INSUFFICIENT_EVIDENCE"),
    ]
    frozen = {q: _frozen_dense_for(q, doc_ids=("doc-A",)) for q in ("q1", "q2")}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg,
    )
    paths = write_outputs(result, out_dir=tmp_path)
    md = paths["summary_md"].read_text(encoding="utf-8")
    for marker in (
        "Phase 7.4", "Totals", "By recovery action", "By bucket", "Invariants",
    ):
        assert marker in md


def test_final_report_contains_invariant_section(tmp_path):
    bm25 = _make_bm25_retriever()
    rows = [
        _verdict("q1", "HYBRID_RECOVERY"),
        _verdict("q2", "INSUFFICIENT_EVIDENCE"),
        _verdict("q3", "ANSWER_WITH_CAUTION"),
    ]
    frozen = {q: _frozen_dense_for(q, doc_ids=("doc-A",))
              for q in ("q1", "q2", "q3")}
    cfg = _config()
    result = run_controlled_recovery(
        verdict_rows=rows, frozen_dense_by_qid=frozen,
        bm25_retriever=bm25, config=cfg,
    )
    paths = write_outputs(result, out_dir=tmp_path)
    md = paths["final_report"].read_text(encoding="utf-8")
    assert "Invariants confirmed" in md
    assert "INSUFFICIENT_EVIDENCE" in md
    assert "ANSWER_WITH_CAUTION" in md


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_verdict_rows_skips_empty_lines(tmp_path):
    p = tmp_path / "vr.jsonl"
    p.write_text(
        '{"query_id": "q1", "recommended_action": "ANSWER"}\n'
        '\n'
        '{"query_id": "q2", "recommended_action": "HYBRID_RECOVERY"}\n',
        encoding="utf-8",
    )
    rows = load_verdict_rows(p)
    assert len(rows) == 2
    assert rows[0]["query_id"] == "q1"


def test_load_frozen_dense_state_reads_candidate_side(tmp_path):
    p = tmp_path / "pq.jsonl"
    p.write_text(
        json.dumps({
            "qid": "q1",
            "query": "테스트",
            "expected_doc_ids": ["doc-A"],
            "candidate": {
                "top_results": [
                    {"chunk_id": "c1", "doc_id": "doc-A", "score": 0.85},
                    {"chunk_id": "c2", "doc_id": "doc-B", "score": 0.70},
                ],
            },
        }) + "\n",
        encoding="utf-8",
    )
    state = load_frozen_dense_state(p, side="candidate", final_k=10)
    assert "q1" in state
    row = state["q1"]
    assert row.top_chunk_ids == ("c1", "c2")
    assert row.top_doc_ids == ("doc-A", "doc-B")
    assert row.top1_score == pytest.approx(0.85)


def test_load_chunks_for_bm25_handles_chunk_text_fallback(tmp_path):
    p = tmp_path / "ch.jsonl"
    p.write_text(
        json.dumps({
            "chunk_id": "c1", "doc_id": "doc-A",
            "title": "T", "chunk_text": "alpha beta",
            "section_path": ["개요"],
            "aliases": ["alias1"],
        }) + "\n"
        + json.dumps({
            "chunk_id": "c2", "doc_id": "doc-B",
            "embedding_text": "title prefix\n\ngamma delta",
            "title": "T2",
        }) + "\n",
        encoding="utf-8",
    )
    chunks = load_chunks_for_bm25(p)
    assert len(chunks) == 2
    # First chunk uses chunk_text since no embedding_text.
    assert chunks[0].text == "alpha beta"
    assert chunks[0].keywords == ("alias1",)
    assert chunks[0].section == "개요"
    # Second chunk prefers embedding_text.
    assert "gamma delta" in chunks[1].text


# ---------------------------------------------------------------------------
# Lower-level helpers
# ---------------------------------------------------------------------------


def test_fuse_dense_and_bm25_preserves_dense_order_on_ties():
    """When fused scores are exactly equal, dense list order wins.

    Two-item case: dense has [c1, c2], BM25 has [c2, c1] → both items
    get the same RRF score (1/61 + 1/62) so the tie-break is
    first_list_seen (dense first, list_idx=0) plus stable-sort over the
    dict's insertion order, which is the dense iteration order.
    """
    dense_chunk_ids = ["c1", "c2"]
    dense_doc_ids = ["doc-A", "doc-B"]

    from app.capabilities.rag.generation import RetrievedChunk
    bm25_results = [
        RetrievedChunk(chunk_id="c2", doc_id="doc-B", section="", text="",
                       score=0.9, rerank_score=None),
        RetrievedChunk(chunk_id="c1", doc_id="doc-A", section="", text="",
                       score=0.7, rerank_score=None),
    ]
    fused = fuse_dense_and_bm25(
        dense_chunk_ids=dense_chunk_ids,
        dense_doc_ids=dense_doc_ids,
        bm25_results=bm25_results,
        final_k=2,
    )
    # Both chunks score 1/61 + 1/62 — tie. Dense order wins.
    assert fused.chunk_ids == ("c1", "c2")


def test_fuse_dense_and_bm25_brings_in_bm25_only_chunks():
    dense_chunk_ids = ["c1"]
    dense_doc_ids = ["doc-A"]
    from app.capabilities.rag.generation import RetrievedChunk
    bm25_results = [
        RetrievedChunk(chunk_id="c-NEW", doc_id="doc-NEW",
                       section="", text="", score=0.5, rerank_score=None),
    ]
    fused = fuse_dense_and_bm25(
        dense_chunk_ids=dense_chunk_ids,
        dense_doc_ids=dense_doc_ids,
        bm25_results=bm25_results,
        final_k=10,
    )
    # c1 from dense wins rank 1, c-NEW from BM25 takes rank 2.
    assert fused.chunk_ids == ("c1", "c-NEW")
    assert fused.doc_ids == ("doc-A", "doc-NEW")
