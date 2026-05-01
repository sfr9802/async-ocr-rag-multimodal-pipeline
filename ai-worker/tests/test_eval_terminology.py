"""Tests for the eval-only silver-set terminology aliasing layer.

Verifies that Phase 7.3 / 7.4 markdown summaries now carry the
silver-set disclaimer, that the misleading "gold/correct/wrong"
phrases have been rewritten to silver-aware names, and that the
helpers themselves are idempotent.
"""

from __future__ import annotations

import json

import pytest

from eval.harness.eval_terminology import (
    SILVER_DISCLAIMER_LINES,
    SILVER_DISCLAIMER_MARKER,
    SILVER_DISCLAIMER_MD,
    SILVER_DISCLAIMER_TEXT,
    apply_silver_terminology,
    prepend_silver_disclaimer,
    silver_aliases,
    silver_disclaimer_block,
    translate_markdown_terms,
)
from eval.harness.recovery_metrics import (
    aggregate_results,
    render_final_report_md,
    render_summary_md as render_recovery_summary_md,
    write_outputs as write_recovery_outputs,
)
from eval.harness.controlled_recovery_loop import (
    ControlledRecoveryConfig,
    run_controlled_recovery,
    FrozenDenseRow,
)
from eval.harness.bm25_retriever import build_bm25_index, BM25EvalRetriever
from eval.harness.embedding_text_builder import VARIANT_RAW
from eval.harness.recovery_policy import REWRITE_MODE_PRODUCTION_LIKE
from eval.harness.v4_confidence_detector import (
    CandidateChunk,
    ConfidenceConfig,
    ConfidenceEvalResult,
    ConfidenceQueryInput,
    aggregate_verdicts,
    decide,
    render_summary_md as render_confidence_summary_md,
    write_outputs as write_confidence_outputs,
)


# ---------------------------------------------------------------------------
# Direct helper tests
# ---------------------------------------------------------------------------


def test_silver_disclaimer_lines_form_md_block():
    """The disclaimer constant should be the joined version of the line list."""
    rebuilt = "\n".join(SILVER_DISCLAIMER_LINES)
    assert SILVER_DISCLAIMER_MD == rebuilt
    assert SILVER_DISCLAIMER_MARKER in SILVER_DISCLAIMER_MD


def test_silver_disclaimer_text_mentions_silver_and_human_audit():
    """The plain-text version must explicitly say 'silver' and 'human audit'."""
    assert "silver" in SILVER_DISCLAIMER_TEXT.lower()
    assert "human audit" in SILVER_DISCLAIMER_TEXT.lower()
    assert "gold" in SILVER_DISCLAIMER_TEXT.lower()


def test_silver_aliases_covers_calibration_terms():
    """Every renamed cross-tab name must appear in the alias map."""
    aliases = silver_aliases()
    for old in (
        "confident_but_wrong",
        "low_confidence_but_correct",
        "confident-but-wrong",
        "low-confidence-but-correct",
        "gold_not_in_candidates",
        "recovered@1",
        "recovered@3",
        "recovered@5",
        "rec@1",
    ):
        assert old in aliases, f"alias map missing entry for {old!r}"
        new = aliases[old]
        # Each new name must include either "silver" or
        # "expected_target" so the aliasing intent is visible.
        assert "silver" in new or "expected_target" in new


def test_translate_markdown_terms_rewrites_compound_names():
    text = (
        "Sample confident-but-wrong (≤10)\n"
        "- confident_but_wrong: 0\n"
        "- low_confidence_but_correct: 4\n"
    )
    out = translate_markdown_terms(text)
    assert "confident_but_silver_mismatch" in out
    assert "low_confidence_but_silver_match" in out
    # Old names must be entirely gone.
    assert "confident_but_wrong" not in out
    assert "confident-but-wrong" not in out
    assert "low_confidence_but_correct" not in out


def test_translate_markdown_terms_is_idempotent():
    """Applying the translator twice yields the same output as once."""
    text = (
        "## Calibration cross-tabs\n\n"
        "- confident_but_wrong: 0\n"
        "- low_confidence_but_correct: 4\n"
        "- recovered@1: 0.143\n"
    )
    once = translate_markdown_terms(text)
    twice = translate_markdown_terms(once)
    assert once == twice


def test_translate_markdown_terms_handles_empty_input():
    assert translate_markdown_terms("") == ""
    assert translate_markdown_terms(None) is None


def test_prepend_silver_disclaimer_inserts_after_first_heading():
    body = "# Phase X — Title\n\n## Subhead\nbody text\n"
    out = prepend_silver_disclaimer(body)
    assert SILVER_DISCLAIMER_MARKER in out
    # The disclaimer must come AFTER the leading heading.
    heading_idx = out.find("# Phase X")
    disclaimer_idx = out.find(SILVER_DISCLAIMER_MARKER)
    assert heading_idx < disclaimer_idx


def test_prepend_silver_disclaimer_idempotent():
    body = "# Phase X — Title\n\nbody\n"
    once = prepend_silver_disclaimer(body)
    twice = prepend_silver_disclaimer(once)
    assert once == twice
    # Disclaimer marker appears exactly once.
    assert once.count(SILVER_DISCLAIMER_MARKER) == 1


def test_apply_silver_terminology_runs_both_passes():
    body = "# Phase 7.3\n\n## Calibration cross-tabs\n- confident_but_wrong: 0\n"
    out = apply_silver_terminology(body)
    assert SILVER_DISCLAIMER_MARKER in out
    assert "confident_but_silver_mismatch" in out
    assert "confident_but_wrong" not in out


def test_silver_disclaimer_block_carries_required_fields():
    block = silver_disclaimer_block()
    assert "silver_disclaimer" in block
    assert "silver_terminology_aliases" in block
    assert isinstance(block["silver_terminology_aliases"], dict)
    assert "human audit" in block["silver_disclaimer"].lower()


# ---------------------------------------------------------------------------
# Phase 7.3 renderer integration
# ---------------------------------------------------------------------------


def _phase73_query(qid: str = "q1") -> ConfidenceQueryInput:
    cands = (
        CandidateChunk(
            rank=1, chunk_id="c1", doc_id="doc-A",
            title="T", retrieval_title="T",
            section_path=("개요",), section_type="summary",
            section="개요", dense_score=0.90,
        ),
        CandidateChunk(
            rank=2, chunk_id="c2", doc_id="doc-A",
            title="T", retrieval_title="T",
            section_path=("기타",), section_type="trivia",
            section="기타", dense_score=0.70,
        ),
    )
    return ConfidenceQueryInput(
        query_id=qid, query_text="q",
        bucket="main_work",
        gold_doc_id="doc-A", expected_title="T",
        top_candidates=cands,
    )


def test_phase73_summary_md_carries_silver_disclaimer():
    verdicts = [decide(_phase73_query("q1"))]
    inputs = [_phase73_query("q1")]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(verdicts=verdicts, inputs=inputs, aggregate=agg)
    md = render_confidence_summary_md(result)
    assert SILVER_DISCLAIMER_MARKER in md
    # The phase header must remain — the disclaimer is appended, not
    # injected in place of any existing content.
    assert "Phase 7.3" in md


def test_phase73_summary_md_uses_silver_terminology():
    """No occurrence of 'confident_but_wrong' / 'low_confidence_but_correct'."""
    verdicts = [decide(_phase73_query("q1"))]
    inputs = [_phase73_query("q1")]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(verdicts=verdicts, inputs=inputs, aggregate=agg)
    md = render_confidence_summary_md(result)
    # Old misleading terms must be absent from the rendered summary.
    assert "confident_but_wrong" not in md
    assert "low_confidence_but_correct" not in md
    assert "confident-but-wrong" not in md
    assert "low-confidence-but-correct" not in md
    # The Calibration section now carries the silver framing.
    assert "Calibration cross-tabs (silver-set)" in md


def test_phase73_aggregate_json_carries_silver_disclaimer_block():
    verdicts = [decide(_phase73_query("q1"))]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    assert "silver_disclaimer" in agg
    assert "silver_terminology_aliases" in agg
    assert "human audit" in agg["silver_disclaimer"].lower()


def test_phase73_summary_md_writes_silver_disclaimer_to_disk(tmp_path):
    """Ensure write_outputs() also emits the disclaimer in the on-disk md."""
    verdicts = [decide(_phase73_query("q1"))]
    inputs = [_phase73_query("q1")]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(verdicts=verdicts, inputs=inputs, aggregate=agg)
    paths = write_confidence_outputs(result, out_dir=tmp_path)
    md_text = paths["summary_md"].read_text(encoding="utf-8")
    assert SILVER_DISCLAIMER_MARKER in md_text
    assert "confident_but_wrong" not in md_text


# ---------------------------------------------------------------------------
# Phase 7.4 renderer integration
# ---------------------------------------------------------------------------


def _phase74_run_minimal():
    """Produce a tiny ControlledRecoveryResult so the writers have something."""
    from types import SimpleNamespace
    chunks = [
        SimpleNamespace(
            chunk_id="c1", doc_id="doc-A",
            section="개요", text="alpha bravo charlie", title="T", keywords=(),
        ),
        SimpleNamespace(
            chunk_id="c2", doc_id="doc-B",
            section="개요", text="delta echo", title="U", keywords=(),
        ),
    ]
    bm25 = BM25EvalRetriever(
        build_bm25_index(chunks, embedding_text_variant=VARIANT_RAW),
        top_k=10,
        name="bm25",
    )
    rows = [
        {
            "query_id": "q1",
            "bucket": "main_work",
            "confidence_label": "AMBIGUOUS",
            "failure_reasons": ["LOW_MARGIN"],
            "recommended_action": "HYBRID_RECOVERY",
            "signals": {},
            "input": {
                "query_text": "alpha",
                "gold_doc_id": "doc-A",
                "gold_page_id": None,
                "expected_title": None,
                "expected_section_type": None,
                "candidate_count": 1,
                "top_candidates_preview": [
                    {
                        "rank": 1, "chunk_id": "c1", "doc_id": "doc-A",
                        "title": "T", "retrieval_title": "T",
                        "section_path": ["개요"], "section_type": "summary",
                    },
                ],
            },
        },
        {
            "query_id": "q2",
            "bucket": "main_work",
            "confidence_label": "AMBIGUOUS",
            "failure_reasons": ["LOW_MARGIN"],
            "recommended_action": "ANSWER_WITH_CAUTION",
            "signals": {},
            "input": {
                "query_text": "alpha",
                "gold_doc_id": "doc-A",
                "gold_page_id": None,
                "expected_title": None,
                "expected_section_type": None,
                "candidate_count": 0,
                "top_candidates_preview": [],
            },
        },
    ]
    frozen = {
        "q1": FrozenDenseRow(
            qid="q1", query_text="alpha",
            expected_doc_ids=("doc-A",),
            top_chunk_ids=("c1",), top_doc_ids=("doc-A",),
            top1_score=0.7,
        ),
        "q2": FrozenDenseRow(
            qid="q2", query_text="alpha",
            expected_doc_ids=("doc-A",),
            top_chunk_ids=("c1",), top_doc_ids=("doc-A",),
            top1_score=0.7,
        ),
    }
    cfg = ControlledRecoveryConfig(
        rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE,
        strict_label_leakage=False,
    ).validate()
    result = run_controlled_recovery(
        verdict_rows=rows,
        frozen_dense_by_qid=frozen,
        bm25_retriever=bm25,
        config=cfg,
    )
    return result


def test_phase74_summary_md_carries_silver_disclaimer():
    result = _phase74_run_minimal()
    agg = aggregate_results(result)
    md = render_recovery_summary_md(agg)
    assert SILVER_DISCLAIMER_MARKER in md
    assert "Phase 7.4" in md


def test_phase74_summary_md_uses_silver_terminology():
    """The Phase 7.4 summary must rewrite recovered@k → silver_target_recovered@k."""
    result = _phase74_run_minimal()
    agg = aggregate_results(result)
    md = render_recovery_summary_md(agg)
    # Old labels gone; silver labels present (or the columns aren't
    # rendered in this minimal run — at minimum the rewrites must hold
    # whenever they appear).
    assert "rec@1" not in md or "silver_target_rec@1" in md
    assert "rec@3" not in md or "silver_target_rec@3" in md


def test_phase74_final_report_carries_silver_disclaimer():
    result = _phase74_run_minimal()
    agg = aggregate_results(result)
    md = render_final_report_md(agg)
    assert SILVER_DISCLAIMER_MARKER in md
    # The Invariants section the existing tests grep for is still here.
    assert "Invariants confirmed" in md


def test_phase74_aggregate_json_carries_silver_disclaimer_block():
    result = _phase74_run_minimal()
    agg = aggregate_results(result)
    assert "silver_disclaimer" in agg
    assert "silver_terminology_aliases" in agg


def test_phase74_writer_emits_silver_disclaimer_to_disk(tmp_path):
    result = _phase74_run_minimal()
    paths = write_recovery_outputs(result, out_dir=tmp_path)
    summary_md = paths["summary_md"].read_text(encoding="utf-8")
    final_md = paths["final_report"].read_text(encoding="utf-8")
    assert SILVER_DISCLAIMER_MARKER in summary_md
    assert SILVER_DISCLAIMER_MARKER in final_md
    summary_json = json.loads(
        paths["summary_json"].read_text(encoding="utf-8")
    )
    assert "silver_disclaimer" in summary_json
    assert "silver_terminology_aliases" in summary_json
