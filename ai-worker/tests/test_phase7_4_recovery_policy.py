"""Phase 7.4 — tests for the recovery_policy module.

The policy is a pure planning function over a Phase 7.3 verdict row.
Every test composes a tiny verdict dict (the same shape Phase 7.3
emits via ``_verdict_to_dict``) and asserts the routing decision plus
the rewriter output.

The contracts under test:

  - INSUFFICIENT_EVIDENCE never triggers recovery.
  - ANSWER_WITH_CAUTION never triggers recovery (calibration only).
  - HYBRID_RECOVERY produces an ATTEMPT_HYBRID decision.
  - QUERY_REWRITE oracle mode is explicitly marked oracle_upper_bound.
  - QUERY_REWRITE production_like mode does not use expected_title.
  - expected_title usage is rejected (LabelLeakageError) when
    production_like + strict_label_leakage=True.
  - classify_attempt computes recovered / regression / newly_entered
    correctly given before/after ranks.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import pytest

from eval.harness.recovery_policy import (
    LabelLeakageError,
    RECOVERY_ACTION_ATTEMPT_HYBRID,
    RECOVERY_ACTION_ATTEMPT_REWRITE,
    RECOVERY_ACTION_NOOP,
    RECOVERY_ACTION_SKIP_CAUTION,
    RECOVERY_ACTION_SKIP_DEFER,
    RECOVERY_ACTION_SKIP_REFUSE,
    RECOVERY_ACTIONS,
    REWRITE_MODE_BOTH,
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
    REWRITE_MODES,
    RecoveryAttempt,
    RecoveryDecision,
    SKIP_REASON_CAUTION_NOT_RECOVERED,
    SKIP_REASON_CLARIFICATION_DEFERRED,
    SKIP_REASON_REFUSED_INSUFFICIENT,
    build_rewritten_query,
    classify_attempt,
    decide_recovery,
)


# ---------------------------------------------------------------------------
# Verdict-row factory
# ---------------------------------------------------------------------------


def _verdict(
    *,
    qid: str = "v4-silver-test",
    bucket: str = "main_work",
    action: str = "QUERY_REWRITE",
    expected_title: str | None = "마법과고교의 열등생",
    gold_doc_id: str | None = "doc-A",
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Compose a Phase-7.3-shaped verdict row for tests."""
    if candidates is None:
        candidates = (
            {
                "rank": 1,
                "chunk_id": "c1",
                "doc_id": "doc-A",
                "title": "마법과고교의 열등생",
                "retrieval_title": "마법과고교의 열등생 / 설정",
                "section_path": ["개요"],
                "section_type": "summary",
            },
            {
                "rank": 2,
                "chunk_id": "c2",
                "doc_id": "doc-B",
                "title": "다른 작품",
                "retrieval_title": "다른 작품 / 설정",
                "section_path": ["등장인물"],
                "section_type": "character",
            },
        )
    return {
        "query_id": qid,
        "bucket": bucket,
        "confidence_label": "AMBIGUOUS",
        "failure_reasons": ["LOW_MARGIN"],
        "recommended_action": action,
        "signals": {},
        "input": {
            "query_text": "테스트 쿼리",
            "gold_doc_id": gold_doc_id,
            "gold_page_id": None,
            "expected_title": expected_title,
            "expected_section_type": None,
            "candidate_count": len(list(candidates)),
            "top_candidates_preview": list(candidates),
        },
    }


# ---------------------------------------------------------------------------
# Action routing
# ---------------------------------------------------------------------------


def test_insufficient_evidence_never_triggers_recovery():
    row = _verdict(action="INSUFFICIENT_EVIDENCE")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_SKIP_REFUSE
    assert d.skip_reason == SKIP_REASON_REFUSED_INSUFFICIENT
    assert d.rewritten_query is None


def test_answer_with_caution_does_not_trigger_recovery():
    row = _verdict(action="ANSWER_WITH_CAUTION")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_SKIP_CAUTION
    assert d.skip_reason == SKIP_REASON_CAUTION_NOT_RECOVERED
    assert d.rewritten_query is None


def test_ask_clarification_is_deferred():
    row = _verdict(action="ASK_CLARIFICATION")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_SKIP_DEFER
    assert d.skip_reason == SKIP_REASON_CLARIFICATION_DEFERRED


def test_answer_action_is_noop():
    row = _verdict(action="ANSWER")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_NOOP
    assert d.rewritten_query is None


def test_hybrid_recovery_creates_hybrid_attempt():
    row = _verdict(action="HYBRID_RECOVERY")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_ATTEMPT_HYBRID
    # Hybrid does NOT carry a rewritten query — BM25 just runs the
    # original query.
    assert d.rewritten_query is None
    assert d.oracle_upper_bound is False


def test_query_rewrite_oracle_mode_is_marked_oracle():
    row = _verdict(action="QUERY_REWRITE")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_ORACLE)
    assert d.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
    assert d.rewrite_mode == REWRITE_MODE_ORACLE
    assert d.oracle_upper_bound is True
    # Oracle includes expected_title in the rewritten query.
    assert d.rewritten_query is not None
    assert "마법과고교의 열등생" in d.rewritten_query
    assert d.rewrite_source.startswith("oracle:")


def test_query_rewrite_production_like_does_not_use_expected_title():
    row = _verdict(action="QUERY_REWRITE")
    d = decide_recovery(
        row,
        rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE,
        strict_label_leakage=False,  # allow non-strict for this test
    )
    assert d.recovery_action == RECOVERY_ACTION_ATTEMPT_REWRITE
    assert d.rewrite_mode == REWRITE_MODE_PRODUCTION_LIKE
    assert d.oracle_upper_bound is False
    # The candidate preview's first title HAPPENS to equal expected_title;
    # the rewriter must filter it out, but candidate #2's "다른 작품"
    # should still be present.
    if d.rewritten_query is not None:
        assert "다른 작품" in d.rewritten_query
    # In every case, we never see expected_title verbatim in the term
    # list (it was de-duped against expected_title).
    for term in d.rewrite_terms:
        # expected_title value
        assert term != "마법과고교의 열등생"
    assert d.rewrite_source is not None


def test_query_rewrite_production_like_strict_raises_on_label_leak():
    """Strict mode + expected_title set → LabelLeakageError.

    The intent is to prevent silver labels from leaking into a
    production-like evaluation. The error surfaces at the policy level
    so the loop never produces a misclassified row.
    """
    row = _verdict(action="QUERY_REWRITE", expected_title="마법과고교의 열등생")
    with pytest.raises(LabelLeakageError):
        decide_recovery(
            row,
            rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE,
            strict_label_leakage=True,
        )


def test_decide_recovery_rejects_both_mode():
    """rewrite_mode='both' is a CLI fan-out, not a per-call mode.

    The loop calls decide_recovery once per concrete mode. Asking the
    policy directly with 'both' should raise so a downstream caller
    can't accidentally end up with mode='both' in a JSONL row.
    """
    row = _verdict(action="QUERY_REWRITE")
    with pytest.raises(ValueError):
        decide_recovery(row, rewrite_mode=REWRITE_MODE_BOTH)


def test_decide_recovery_unknown_action_defers():
    row = _verdict(action="WAT_LO_THIS_DOES_NOT_EXIST")
    d = decide_recovery(row, rewrite_mode=REWRITE_MODE_PRODUCTION_LIKE)
    assert d.recovery_action == RECOVERY_ACTION_SKIP_DEFER


# ---------------------------------------------------------------------------
# Rewriter invariants
# ---------------------------------------------------------------------------


def test_build_rewritten_query_oracle_uses_expected_title_first():
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="테스트",
        expected_title="EXPECTED_TITLE",
        candidate_preview=[],
        mode=REWRITE_MODE_ORACLE,
    )
    assert rewritten is not None
    # expected_title appears in terms[0] and sources[0].
    assert terms[0] == "EXPECTED_TITLE"
    assert sources[0] == "expected_title"
    assert label == "oracle:expected_title"


def test_build_rewritten_query_oracle_with_no_title_returns_noop():
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="테스트",
        expected_title=None,
        candidate_preview=[],
        mode=REWRITE_MODE_ORACLE,
    )
    assert rewritten is None
    assert terms == ()
    assert label == "oracle:noop"


def test_build_rewritten_query_production_like_strict_blocks_leakage():
    with pytest.raises(LabelLeakageError):
        build_rewritten_query(
            query_text="테스트",
            expected_title="EXPECTED_TITLE",
            candidate_preview=[],
            mode=REWRITE_MODE_PRODUCTION_LIKE,
            strict_label_leakage=True,
        )


def test_build_rewritten_query_production_like_non_strict_strips_leak():
    """Non-strict mode strips expected_title from the term pool but does
    not raise. The loop should mark the row appropriately."""
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="테스트",
        expected_title="EXPECTED_TITLE",
        candidate_preview=[
            {"title": "EXPECTED_TITLE", "retrieval_title": "EXPECTED_TITLE"},
            {"title": "OTHER_TITLE", "retrieval_title": "OTHER_TITLE"},
        ],
        mode=REWRITE_MODE_PRODUCTION_LIKE,
        strict_label_leakage=False,
    )
    # EXPECTED_TITLE is filtered out; only OTHER_TITLE remains.
    for term in terms:
        assert term != "EXPECTED_TITLE"
    assert "OTHER_TITLE" in terms
    assert rewritten is not None
    assert label == "production_like:top_n_titles"


def test_build_rewritten_query_production_like_no_titles_returns_noop():
    """If candidate pool yields nothing the production_like rewrite is a no-op.

    The loop should fall back to running BM25 against the original
    query. Sources / terms are empty.
    """
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="테스트",
        expected_title=None,
        candidate_preview=[],
        mode=REWRITE_MODE_PRODUCTION_LIKE,
    )
    assert rewritten is None
    assert terms == ()
    assert sources == ()
    assert label == "production_like:noop"


def test_build_rewritten_query_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_rewritten_query(
            query_text="x",
            expected_title=None,
            candidate_preview=[],
            mode="not_a_real_mode",
        )


def test_build_rewritten_query_both_mode_is_invalid_at_call_site():
    with pytest.raises(ValueError):
        build_rewritten_query(
            query_text="x",
            expected_title=None,
            candidate_preview=[],
            mode=REWRITE_MODE_BOTH,
        )


def test_production_like_dedupes_and_orders_titles():
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="q",
        expected_title=None,
        candidate_preview=[
            {"title": "TITLE_A", "retrieval_title": "TITLE_A"},
            {"title": "TITLE_A", "retrieval_title": "TITLE_A / SUB"},
            {"title": "TITLE_B"},
        ],
        mode=REWRITE_MODE_PRODUCTION_LIKE,
        top_n_for_production=3,
    )
    # Each canonical title appears once.
    assert terms.count("TITLE_A") == 1
    # TITLE_A / SUB normalises against TITLE_A and is dropped (loose
    # title comparator treats them as the same canonical surface).
    # TITLE_B is kept because it is genuinely distinct.
    assert "TITLE_B" in terms


def test_production_like_top_n_caps_term_count():
    rewritten, terms, sources, label = build_rewritten_query(
        query_text="q",
        expected_title=None,
        candidate_preview=[
            {"title": f"TITLE_{i}"} for i in range(20)
        ],
        mode=REWRITE_MODE_PRODUCTION_LIKE,
        top_n_for_production=3,
    )
    # The top_n_for_production cap is on candidates, not terms — but the
    # gather function only walks the first N candidates, so terms cannot
    # exceed N (one title per candidate in this fixture).
    assert len(terms) <= 3


# ---------------------------------------------------------------------------
# classify_attempt
# ---------------------------------------------------------------------------


def _attempt(
    *,
    action: str,
    before: int,
    after: int,
    final_k: int = 10,
    rewrite_mode: str | None = None,
    oracle: bool = False,
) -> RecoveryAttempt:
    decision = RecoveryDecision(
        query_id="q",
        bucket="main_work",
        original_action="QUERY_REWRITE",
        recovery_action=action,
        rewrite_mode=rewrite_mode,
        oracle_upper_bound=oracle,
    )
    return RecoveryAttempt(
        decision=decision,
        before_rank=before,
        before_top_doc_ids=(),
        before_top_chunk_ids=(),
        before_in_top_k=(0 < before <= final_k),
        before_top1_score=None,
        after_rank=after,
        after_top_doc_ids=(),
        after_top_chunk_ids=(),
        after_in_top_k=(0 < after <= final_k),
        after_top1_score=None,
        final_k=final_k,
        latency_ms=1.5,
    )


def test_classify_attempt_recovered_when_gold_enters_top_k():
    """before > k, after ≤ k → recovered=True."""
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_HYBRID,
        before=15, after=3, final_k=10,
    )
    r = classify_attempt(a)
    assert r.recovered is True
    assert r.regression is False
    assert r.gold_newly_entered_candidates is False  # had-gold-already
    assert r.rank_delta == 3 - 15


def test_classify_attempt_newly_entered_when_before_was_miss():
    """before == -1, after positive → gold_newly_entered_candidates."""
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_HYBRID,
        before=-1, after=4, final_k=10,
    )
    r = classify_attempt(a)
    assert r.gold_newly_entered_candidates is True
    # also recovered (after ≤ k).
    assert r.recovered is True


def test_classify_attempt_regression_when_gold_falls_out():
    """before ≤ k, after == -1 → regression."""
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_HYBRID,
        before=3, after=-1, final_k=10,
    )
    r = classify_attempt(a)
    assert r.recovered is False
    assert r.regression is True


def test_classify_attempt_regression_when_after_rank_is_above_k():
    """before ≤ k, after > k → regression."""
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_HYBRID,
        before=3, after=12, final_k=10,
    )
    r = classify_attempt(a)
    assert r.recovered is False
    assert r.regression is True


def test_classify_attempt_unchanged_when_both_in_top_k():
    """Both before and after ≤ k, after > before but still in window."""
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_HYBRID,
        before=3, after=6, final_k=10,
    )
    r = classify_attempt(a)
    assert r.recovered is False
    assert r.regression is False


def test_classify_attempt_skip_actions_never_recover():
    for action in (
        RECOVERY_ACTION_SKIP_REFUSE,
        RECOVERY_ACTION_SKIP_CAUTION,
        RECOVERY_ACTION_SKIP_DEFER,
        RECOVERY_ACTION_NOOP,
    ):
        a = _attempt(action=action, before=-1, after=1, final_k=10)
        r = classify_attempt(a)
        # Even though after looks like an improvement, skipped attempts
        # never count as recovered/regressed — there was no work done.
        assert r.skipped is True
        assert r.recovered is False
        assert r.regression is False
        assert r.gold_newly_entered_candidates is False


def test_classify_attempt_oracle_flag_propagates():
    a = _attempt(
        action=RECOVERY_ACTION_ATTEMPT_REWRITE,
        before=-1, after=2,
        rewrite_mode=REWRITE_MODE_ORACLE,
        oracle=True,
    )
    r = classify_attempt(a)
    assert r.oracle_upper_bound is True
    assert r.rewrite_mode == REWRITE_MODE_ORACLE
