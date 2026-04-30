"""Phase 7.3 — tests for the retrieval confidence detector.

Fixture-driven: every test composes a ``ConfidenceQueryInput`` with a
small CandidateChunk list and asserts the verdict shape. The classifier
has no FAISS / model dependency, so a single ``decide`` call is enough
to exercise every rule.

The test cases were chosen from the spec in the Phase 7.3 brief:

  - high score + high margin + same page → CONFIDENT / ANSWER
  - low top1 score → LOW_CONFIDENCE / HYBRID_RECOVERY
  - low margin → AMBIGUOUS / ANSWER_WITH_CAUTION (or stronger when
    combined with another reason)
  - page_id 분산 → PAGE_ID_DISAGREEMENT
  - generic title collision → GENERIC_COLLISION
  - duplicate rate 높음 → HIGH_DUPLICATE_RATE
  - gold not in candidates → GOLD_NOT_IN_CANDIDATES + FAILED
  - rerank demoted gold → RERANK_DEMOTED_GOLD
  - missing optional gold → none of the gold-aware reasons fire
  - JSONL output schema is stable

Plus a few defence-in-depth cases:

  - title alias mismatch → TITLE_ALIAS_MISMATCH
  - section intent mismatch → SECTION_INTENT_MISMATCH
  - empty candidate list → INSUFFICIENT_EVIDENCE / FAILED-not-applicable
  - precedence: GOLD_NOT_IN_CANDIDATES wins label assignment
  - aggregate_verdicts counts add up to n_queries
  - find_confident_but_wrong / find_low_confidence_but_correct selectors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from eval.harness.v4_confidence_detector import (
    ACTION_ANSWER,
    ACTION_ANSWER_WITH_CAUTION,
    ACTION_HYBRID_RECOVERY,
    ACTION_INSUFFICIENT_EVIDENCE,
    ACTION_QUERY_REWRITE,
    CONFIDENCE_LABELS,
    CandidateChunk,
    ConfidenceConfig,
    ConfidenceEvalResult,
    ConfidenceQueryInput,
    FAILURE_REASONS,
    LABEL_AMBIGUOUS,
    LABEL_CONFIDENT,
    LABEL_FAILED,
    LABEL_LOW_CONFIDENCE,
    REASON_GENERIC_COLLISION,
    REASON_GOLD_LOW_RANK,
    REASON_GOLD_NOT_IN_CANDIDATES,
    REASON_HIGH_DUPLICATE_RATE,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_LOW_MARGIN,
    REASON_LOW_TOP1_SCORE,
    REASON_PAGE_ID_DISAGREEMENT,
    REASON_RERANK_DEMOTED_GOLD,
    REASON_SECTION_INTENT_MISMATCH,
    REASON_TITLE_ALIAS_MISMATCH,
    RECOMMENDED_ACTIONS,
    aggregate_verdicts,
    decide,
    find_confident_but_wrong,
    find_low_confidence_but_correct,
    load_inputs_from_phase7_artifacts,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cand(
    rank: int,
    chunk_id: str,
    *,
    doc_id: str = "doc-A",
    title: str = "마법과고교의 열등생",
    retrieval_title: str = "마법과고교의 열등생 / 설정",
    section_path: Sequence[str] = ("개요",),
    section_type: str = "summary",
    dense_score: float = 0.80,
    rerank_score: float | None = None,
    final_score: float | None = None,
    page_id: str | None = None,
) -> CandidateChunk:
    return CandidateChunk(
        rank=rank,
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        retrieval_title=retrieval_title,
        section_path=tuple(section_path),
        section_type=section_type,
        section=" > ".join(section_path),
        dense_score=dense_score,
        rerank_score=rerank_score,
        final_score=final_score,
        page_id=page_id,
    )


def _strong_topk(*, doc_id: str = "doc-A", n: int = 6) -> tuple[CandidateChunk, ...]:
    """A high-confidence top-k: same page, high score, big top1-top2 gap.

    The first chunk is the well-separated top, the remaining n-1 are
    the same page at slightly lower scores. Margin is 0.20 between
    rank 1 and rank 2 — comfortably above the default 0.04 floor.
    """
    cs: list[CandidateChunk] = []
    cs.append(_cand(1, "c1", doc_id=doc_id, dense_score=0.90,
                    section_path=("개요",), section_type="summary"))
    for i in range(2, n + 1):
        cs.append(_cand(
            i, f"c{i}", doc_id=doc_id,
            dense_score=0.70 - 0.01 * i,
            section_path=(f"섹션-{i}",),
            section_type="summary",
        ))
    return tuple(cs)


def _query(
    qid: str = "v4-silver-0001",
    *,
    bucket: str = "main_work",
    gold_doc_id: str | None = "doc-A",
    expected_title: str | None = "마법과고교의 열등생",
    expected_section_type: str | None = None,
    candidates: tuple[CandidateChunk, ...] | None = None,
    rerank_demoted_gold: bool | None = None,
) -> ConfidenceQueryInput:
    return ConfidenceQueryInput(
        query_id=qid,
        query_text="테스트 쿼리",
        bucket=bucket,
        gold_doc_id=gold_doc_id,
        expected_title=expected_title,
        expected_section_type=expected_section_type,
        top_candidates=candidates if candidates is not None else _strong_topk(),
        rerank_demoted_gold=rerank_demoted_gold,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_validation_rejects_negative_top1_score():
    with pytest.raises(ValueError):
        ConfidenceConfig(min_top1_score=-0.1).validate()


def test_config_validation_rejects_out_of_range_ratios():
    with pytest.raises(ValueError):
        ConfidenceConfig(min_same_page_ratio=1.5).validate()
    with pytest.raises(ValueError):
        ConfidenceConfig(max_duplicate_rate=-0.1).validate()


def test_config_validation_rejects_zero_evidence_chunks():
    with pytest.raises(ValueError):
        ConfidenceConfig(min_evidence_chunks_same_page=0).validate()


# ---------------------------------------------------------------------------
# Core decision rules
# ---------------------------------------------------------------------------


def test_high_score_high_margin_same_page_yields_confident_answer():
    """Strong topk: high top1, big margin, fully same-page → CONFIDENT/ANSWER."""
    inp = _query()
    v = decide(inp)
    assert v.failure_reasons == ()
    assert v.confidence_label == LABEL_CONFIDENT
    assert v.recommended_action == ACTION_ANSWER
    # Sanity-check the signal block.
    assert v.signals.top1_score == pytest.approx(0.90, abs=1e-9)
    assert v.signals.top1_top2_margin > 0.10
    assert v.signals.page_id_consistency == 1.0
    assert v.signals.gold_in_top_k is True
    assert v.signals.gold_rank == 1


def test_low_top1_score_yields_low_confidence_hybrid_recovery():
    """Top-1 dense at 0.40 < default 0.55 → LOW_CONFIDENCE/HYBRID_RECOVERY."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
        _cand(3, "c3", doc_id="doc-A", dense_score=0.38),
    )
    v = decide(_query(candidates=cands))
    assert REASON_LOW_TOP1_SCORE in v.failure_reasons
    assert v.confidence_label == LABEL_LOW_CONFIDENCE
    assert v.recommended_action == ACTION_HYBRID_RECOVERY


def test_low_margin_alone_yields_ambiguous_answer_with_caution():
    """Top1 0.81, top2 0.80 (margin 0.01 < 0.04) → AMBIGUOUS/ANSWER_WITH_CAUTION."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.81),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.80),
        _cand(3, "c3", doc_id="doc-A", dense_score=0.79),
    )
    v = decide(_query(candidates=cands))
    assert REASON_LOW_MARGIN in v.failure_reasons
    assert v.confidence_label == LABEL_AMBIGUOUS
    assert v.recommended_action == ACTION_ANSWER_WITH_CAUTION


def test_page_id_disagreement_emits_reason():
    """Top-k chunks evenly spread across 4 pages → PAGE_ID_DISAGREEMENT."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.80),
        _cand(2, "c2", doc_id="doc-B", dense_score=0.78),
        _cand(3, "c3", doc_id="doc-C", dense_score=0.76),
        _cand(4, "c4", doc_id="doc-D", dense_score=0.74),
    )
    v = decide(_query(candidates=cands, gold_doc_id=None))
    assert REASON_PAGE_ID_DISAGREEMENT in v.failure_reasons
    # 4 distinct pages → consistency = 0.25 < 0.30
    assert v.signals.page_id_consistency == pytest.approx(0.25, abs=1e-9)
    # PAGE_ID_DISAGREEMENT is HARD → label LOW_CONFIDENCE
    assert v.confidence_label == LABEL_LOW_CONFIDENCE


def test_generic_collision_emits_reason_when_above_threshold():
    """7+ chunks under generic page-title sections → GENERIC_COLLISION."""
    generic_sections = ("등장인물", "평가", "OST", "기타", "회차", "에피소드", "음악", "줄거리")
    cands = tuple(
        _cand(
            i + 1, f"c{i+1}", doc_id="doc-A",
            dense_score=0.80 - 0.01 * i,
            section_path=(generic_sections[i],),
            section_type="character",
        )
        for i in range(8)
    )
    v = decide(_query(candidates=cands, gold_doc_id="doc-A"))
    assert REASON_GENERIC_COLLISION in v.failure_reasons


def test_generic_collision_does_not_fire_below_threshold():
    """Default threshold = 6, so 4 generic chunks should not fire."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.85, section_path=("등장인물",), section_type="character"),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.80, section_path=("평가",), section_type="reception"),
        _cand(3, "c3", doc_id="doc-A", dense_score=0.78, section_path=("개요",), section_type="summary"),
        _cand(4, "c4", doc_id="doc-A", dense_score=0.74, section_path=("개요",), section_type="summary"),
    )
    v = decide(_query(candidates=cands, gold_doc_id="doc-A"))
    assert REASON_GENERIC_COLLISION not in v.failure_reasons


def test_high_duplicate_rate_emits_reason():
    """Pathological repetition (>0.90 dup) → HIGH_DUPLICATE_RATE.

    The default ceiling 0.90 is set so 10-of-10 same doc (dup=0.9)
    does NOT fire — that's the *desired* same-page convergence and
    flagging it would over-fire on every CONFIDENT main_work query.
    11+ same-doc chunks (dup>0.9) is the true repetition tail.
    """
    cands = tuple(
        _cand(i, f"c{i}", doc_id="doc-A", dense_score=0.85 - 0.005 * i)
        for i in range(1, 12)  # 11 chunks → dup_rate = 10/11 ≈ 0.909
    )
    v = decide(_query(candidates=cands))
    assert REASON_HIGH_DUPLICATE_RATE in v.failure_reasons
    assert v.signals.duplicate_rate > 0.90


def test_gold_not_in_candidates_yields_failed_label():
    """Gold doc-X but top-k is doc-A only → GOLD_NOT_IN_CANDIDATES/FAILED."""
    cands = _strong_topk(doc_id="doc-A")
    v = decide(_query(candidates=cands, gold_doc_id="doc-X"))
    assert REASON_GOLD_NOT_IN_CANDIDATES in v.failure_reasons
    assert v.confidence_label == LABEL_FAILED
    assert v.recommended_action == ACTION_INSUFFICIENT_EVIDENCE
    assert v.signals.gold_in_top_k is False
    assert v.signals.gold_rank == -1


def test_gold_low_rank_emits_reason_above_threshold():
    """Gold sits at rank 8 (>5 default threshold) → GOLD_LOW_RANK."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.85),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.83),
        _cand(3, "c3", doc_id="doc-A", dense_score=0.82),
        _cand(4, "c4", doc_id="doc-A", dense_score=0.81),
        _cand(5, "c5", doc_id="doc-A", dense_score=0.80),
        _cand(6, "c6", doc_id="doc-A", dense_score=0.79),
        _cand(7, "c7", doc_id="doc-A", dense_score=0.78),
        _cand(8, "c8", doc_id="doc-X", dense_score=0.77),  # gold here
        _cand(9, "c9", doc_id="doc-A", dense_score=0.76),
    )
    v = decide(_query(candidates=cands, gold_doc_id="doc-X"))
    assert REASON_GOLD_LOW_RANK in v.failure_reasons
    assert v.signals.gold_rank == 8


def test_rerank_demoted_gold_emits_reason():
    """Phase 7.1 demotion flag set → RERANK_DEMOTED_GOLD/HYBRID_RECOVERY."""
    cands = _strong_topk(doc_id="doc-A")
    v = decide(_query(candidates=cands, rerank_demoted_gold=True))
    assert REASON_RERANK_DEMOTED_GOLD in v.failure_reasons
    assert v.confidence_label == LABEL_LOW_CONFIDENCE
    assert v.recommended_action == ACTION_HYBRID_RECOVERY


def test_rerank_demoted_gold_false_does_not_emit_reason():
    """Demoted=False (rerank ran but did NOT demote) must not fire."""
    cands = _strong_topk(doc_id="doc-A")
    v = decide(_query(candidates=cands, rerank_demoted_gold=False))
    assert REASON_RERANK_DEMOTED_GOLD not in v.failure_reasons


def test_title_alias_mismatch_when_top1_retrieval_title_differs():
    """Top1's retrieval_title doesn't match expected → TITLE_ALIAS_MISMATCH."""
    cands = (
        _cand(
            1, "c1", doc_id="doc-A", dense_score=0.85,
            retrieval_title="다른 작품 / 등장인물",
            title="다른 작품",
            section_path=("등장인물",),
        ),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.80),
    )
    v = decide(_query(candidates=cands, expected_title="마법과고교의 열등생"))
    assert REASON_TITLE_ALIAS_MISMATCH in v.failure_reasons
    assert v.confidence_label == LABEL_LOW_CONFIDENCE
    assert v.recommended_action == ACTION_QUERY_REWRITE


def test_section_intent_mismatch_when_section_type_differs():
    """expected_section_type='setting', top1 section_type='character' → mismatch."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.85, section_type="character"),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.80, section_type="character"),
    )
    v = decide(_query(candidates=cands, expected_section_type="setting"))
    assert REASON_SECTION_INTENT_MISMATCH in v.failure_reasons
    # SECTION_INTENT_MISMATCH alone is SOFT → AMBIGUOUS.
    assert v.confidence_label == LABEL_AMBIGUOUS


def test_missing_optional_gold_does_not_fire_gold_reasons():
    """No gold_doc_id / gold_page_id → none of the gold-aware reasons fire."""
    cands = _strong_topk(doc_id="doc-A")
    v = decide(_query(candidates=cands, gold_doc_id=None))
    assert REASON_GOLD_NOT_IN_CANDIDATES not in v.failure_reasons
    assert REASON_GOLD_LOW_RANK not in v.failure_reasons
    assert v.signals.gold_in_top_k is None
    assert v.signals.gold_rank is None


def test_missing_expected_title_does_not_fire_title_reason():
    """No expected_title → TITLE_ALIAS_MISMATCH cannot fire."""
    cands = (
        _cand(
            1, "c1", retrieval_title="completely different",
            section_path=("개요",), dense_score=0.85,
        ),
        _cand(2, "c2", dense_score=0.80),
    )
    v = decide(_query(candidates=cands, expected_title=None))
    assert REASON_TITLE_ALIAS_MISMATCH not in v.failure_reasons


def test_empty_candidate_list_yields_insufficient_evidence():
    """No candidates at all → INSUFFICIENT_EVIDENCE only, label LOW_CONFIDENCE."""
    v = decide(_query(candidates=()))
    assert REASON_INSUFFICIENT_EVIDENCE in v.failure_reasons
    # FAILED is reserved for known-gold-missing; with empty cands and
    # a known gold the gold-missing reason also fires.
    assert v.confidence_label in (LABEL_LOW_CONFIDENCE, LABEL_FAILED)


def test_empty_candidate_list_without_gold_is_low_confidence_not_failed():
    """Empty cands, gold unknown → LOW_CONFIDENCE (FAILED is gold-only)."""
    v = decide(_query(candidates=(), gold_doc_id=None))
    assert REASON_INSUFFICIENT_EVIDENCE in v.failure_reasons
    assert v.confidence_label == LABEL_LOW_CONFIDENCE


def test_gold_not_in_candidates_takes_precedence_over_low_top1_score():
    """Both reasons fire, but FAILED label wins because gold is missing."""
    # Low top1 score AND gold not in candidates.
    cands = tuple(
        _cand(i, f"c{i}", doc_id="doc-A", dense_score=0.40 - 0.01 * i)
        for i in range(1, 5)
    )
    v = decide(_query(candidates=cands, gold_doc_id="doc-X"))
    assert REASON_LOW_TOP1_SCORE in v.failure_reasons
    assert REASON_GOLD_NOT_IN_CANDIDATES in v.failure_reasons
    # FAILED dominates LOW_CONFIDENCE when gold is provably missing.
    assert v.confidence_label == LABEL_FAILED
    assert v.recommended_action == ACTION_INSUFFICIENT_EVIDENCE


def test_combined_soft_reasons_yield_ambiguous():
    """Two SOFT reasons (low margin + section mismatch) → AMBIGUOUS still."""
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.81, section_type="character"),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.80, section_type="character"),
    )
    v = decide(_query(candidates=cands, expected_section_type="setting"))
    assert REASON_LOW_MARGIN in v.failure_reasons
    assert REASON_SECTION_INTENT_MISMATCH in v.failure_reasons
    assert v.confidence_label == LABEL_AMBIGUOUS
    # Both SOFT → ANSWER_WITH_CAUTION still wins
    assert v.recommended_action == ACTION_ANSWER_WITH_CAUTION


def test_rerank_score_overrides_dense_for_top1_threshold():
    """When rerank_score is set on top1, that's what top1_score reads."""
    cands = (
        _cand(
            1, "c1", doc_id="doc-A",
            dense_score=0.40,           # would trip LOW_TOP1_SCORE on dense
            rerank_score=0.95,           # rerank says it's strong
        ),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.39, rerank_score=0.50),
    )
    v = decide(_query(candidates=cands))
    assert v.signals.top1_score == pytest.approx(0.95, abs=1e-9)
    assert REASON_LOW_TOP1_SCORE not in v.failure_reasons


def test_final_score_overrides_rerank_for_top1_threshold():
    """When final_score is set (weighted-blend mode), it wins over rerank."""
    cands = (
        _cand(
            1, "c1", doc_id="doc-A",
            dense_score=0.40, rerank_score=0.95, final_score=0.30,
        ),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.39, final_score=0.20),
    )
    v = decide(_query(candidates=cands))
    assert v.signals.top1_score == pytest.approx(0.30, abs=1e-9)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_verdicts_counts_sum_to_n():
    """Across a small batch the per-label counts must sum to n_queries."""
    verdicts = [
        decide(_query(qid="q1", candidates=_strong_topk())),
        decide(_query(qid="q2", candidates=(
            _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
            _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
        ))),
        decide(_query(qid="q3", candidates=_strong_topk(), gold_doc_id="doc-X")),
    ]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    assert agg["n_queries"] == 3
    assert sum(agg["labels"].values()) == 3
    assert sum(agg["actions"].values()) == 3
    # Every label / action key from the registry is present in the output.
    for lab in CONFIDENCE_LABELS:
        assert lab in agg["labels"]
    for a in RECOMMENDED_ACTIONS:
        assert a in agg["actions"]
    for r in FAILURE_REASONS:
        assert r in agg["reasons"]


def test_aggregate_verdicts_by_bucket_partitions_correctly():
    verdicts = [
        decide(_query(qid="q1", bucket="main_work")),
        decide(_query(qid="q2", bucket="main_work")),
        decide(_query(qid="q3", bucket="subpage_named")),
    ]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    by_bucket = agg["by_bucket"]
    assert by_bucket["main_work"]["count"] == 2
    assert by_bucket["subpage_named"]["count"] == 1
    # Each bucket's label counts must sum to its count.
    for bucket, payload in by_bucket.items():
        assert sum(payload["labels"].values()) == payload["count"]


def test_find_confident_but_wrong_filters_correctly():
    """CONFIDENT label + gold_in_top_k=False is the bucket of interest."""
    # q1: confident but gold missing → counted
    # q2: confident and gold present → excluded
    # q3: low confidence, gold missing → excluded (label, not gold)
    cands = _strong_topk(doc_id="doc-A")
    v_conf_wrong = decide(_query(qid="q1", candidates=cands, gold_doc_id="doc-Z"))
    # decide() will mark this as FAILED, not CONFIDENT, so to actually
    # construct a CONFIDENT-but-wrong row we need to bypass: easier to
    # test that the selector handles label correctly given verdict input
    # by constructing one directly using decide() then asserting.
    # In practice CONFIDENT-but-wrong means the gold label was wrong /
    # gold not labelled correctly; test the filter logic on a synthetic
    # verdict rather than rely on it firing in this corpus.
    # Simulate: take a real CONFIDENT verdict and pretend gold_in_top_k
    # is False — direct construction not possible (frozen dataclass),
    # so we reach into the selector with a hand-built list.
    from eval.harness.v4_confidence_detector import (
        ConfidenceSignals, ConfidenceVerdict,
    )
    fake = ConfidenceVerdict(
        query_id="qsim",
        bucket="main_work",
        confidence_label=LABEL_CONFIDENT,
        failure_reasons=(),
        recommended_action=ACTION_ANSWER,
        signals=ConfidenceSignals(
            top1_score=0.9, top1_top2_margin=0.2,
            page_id_consistency=1.0, same_page_top_k_count=10,
            candidate_count=10,
            title_match=True, retrieval_title_match=True,
            section_type_match=True, generic_collision_count=0,
            duplicate_rate=0.0, gold_in_top_k=False, gold_rank=-1,
            rerank_demoted_gold=None,
        ),
        debug_summary="qid=qsim",
    )
    fake_correct = ConfidenceVerdict(
        query_id="qok",
        bucket="main_work",
        confidence_label=LABEL_CONFIDENT,
        failure_reasons=(),
        recommended_action=ACTION_ANSWER,
        signals=ConfidenceSignals(
            top1_score=0.9, top1_top2_margin=0.2,
            page_id_consistency=1.0, same_page_top_k_count=10,
            candidate_count=10,
            title_match=True, retrieval_title_match=True,
            section_type_match=True, generic_collision_count=0,
            duplicate_rate=0.0, gold_in_top_k=True, gold_rank=1,
            rerank_demoted_gold=None,
        ),
        debug_summary="qid=qok",
    )
    out = find_confident_but_wrong([fake, fake_correct, v_conf_wrong])
    qids = {v.query_id for v in out}
    assert "qsim" in qids
    assert "qok" not in qids


def test_find_low_confidence_but_correct_filters_correctly():
    """LOW_CONFIDENCE / FAILED label + gold at rank 1 → returned."""
    # Direct construct: a low-top1-score verdict whose gold somehow is at rank 1.
    cands = (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.40),  # gold-A but low score
        _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
    )
    v = decide(_query(candidates=cands, gold_doc_id="doc-A"))
    assert v.confidence_label == LABEL_LOW_CONFIDENCE
    assert v.signals.gold_rank == 1
    out = find_low_confidence_but_correct([v])
    assert len(out) == 1
    assert out[0].query_id == v.query_id


# ---------------------------------------------------------------------------
# JSONL output schema stability
# ---------------------------------------------------------------------------


def test_write_outputs_creates_all_files(tmp_path):
    verdicts = [
        decide(_query(qid="q1")),
        decide(_query(qid="q2", candidates=(
            _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
            _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
        ))),
        decide(_query(qid="q3", gold_doc_id="doc-X")),  # FAILED
    ]
    inputs = [
        _query(qid="q1"),
        _query(qid="q2", candidates=(
            _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
            _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
        )),
        _query(qid="q3", gold_doc_id="doc-X"),
    ]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(
        verdicts=verdicts, inputs=inputs, aggregate=agg,
    )
    out_paths = write_outputs(result, out_dir=tmp_path)
    for role, p in out_paths.items():
        assert p.exists(), f"missing output for role={role}"
    # Spot-check the per-query JSONL: schema keys present + parseable.
    rows = [
        json.loads(line)
        for line in (tmp_path / "per_query_confidence.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    expected_keys = {
        "query_id", "bucket", "confidence_label", "failure_reasons",
        "recommended_action", "signals", "debug_summary",
    }
    for row in rows:
        assert expected_keys <= set(row.keys())
        sig_keys = {
            "top1_score", "top1_top2_margin", "page_id_consistency",
            "same_page_top_k_count", "candidate_count", "title_match",
            "retrieval_title_match", "section_type_match",
            "generic_collision_count", "duplicate_rate",
            "gold_in_top_k", "gold_rank", "rerank_demoted_gold",
        }
        assert sig_keys <= set(row["signals"].keys())


def test_write_outputs_recovery_subset_excludes_answer(tmp_path):
    """recommended_recovery_queries.jsonl must skip rows whose action is ANSWER."""
    verdicts = [
        decide(_query(qid="q1")),  # CONFIDENT/ANSWER
        decide(_query(qid="q2", candidates=(
            _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
            _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
        ))),  # LOW_CONFIDENCE/HYBRID_RECOVERY
    ]
    inputs = [_query(qid="q1"), _query(qid="q2", candidates=verdicts[1].signals and (
        _cand(1, "c1", doc_id="doc-A", dense_score=0.40),
        _cand(2, "c2", doc_id="doc-A", dense_score=0.39),
    ))]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(
        verdicts=verdicts, inputs=inputs, aggregate=agg,
    )
    out_paths = write_outputs(result, out_dir=tmp_path)
    recovery_rows = [
        json.loads(line) for line in
        out_paths["recovery"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qids = {r["query_id"] for r in recovery_rows}
    assert "q1" not in qids
    assert "q2" in qids


def test_summary_md_contains_required_sections(tmp_path):
    """Phase 7.3 summary.md must surface labels / actions / reasons / buckets."""
    verdicts = [decide(_query(qid="q1")), decide(_query(qid="q2"))]
    inputs = [_query(qid="q1"), _query(qid="q2")]
    agg = aggregate_verdicts(verdicts, ConfidenceConfig())
    result = ConfidenceEvalResult(verdicts=verdicts, inputs=inputs, aggregate=agg)
    out_paths = write_outputs(result, out_dir=tmp_path)
    md = out_paths["summary_md"].read_text(encoding="utf-8")
    for marker in (
        "Phase 7.3", "Confidence label distribution",
        "Recommended action distribution", "Failure reason counts",
        "By bucket", "Calibration cross-tabs",
    ):
        assert marker in md, f"summary md missing section {marker!r}"


# ---------------------------------------------------------------------------
# Phase 7.0 / 7.1 artefact loader
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_loader_reads_phase7_0_per_query_with_chunk_enrichment(tmp_path):
    """End-to-end: write tiny Phase 7.0 + chunks files, load, decide."""
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks_path, [
        {
            "chunk_id": "c1", "doc_id": "doc-A",
            "title": "마법과고교의 열등생",
            "retrieval_title": "마법과고교의 열등생 / 설정",
            "section_path": ["개요"],
            "section_type": "summary",
        },
        {
            "chunk_id": "c2", "doc_id": "doc-A",
            "title": "마법과고교의 열등생",
            "retrieval_title": "마법과고교의 열등생 / 설정",
            "section_path": ["기타"],
            "section_type": "summary",
        },
    ])
    per_query_path = tmp_path / "per_query.jsonl"
    _write_jsonl(per_query_path, [
        {
            "qid": "qx",
            "query": "테스트 쿼리",
            "expected_doc_ids": ["doc-A"],
            "bucket": "main_work",
            "v4_meta": {
                "bucket": "main_work",
                "page_type": "main",
                "retrieval_title": "마법과고교의 열등생",
                "work_title": "마법과고교의 열등생",
            },
            "candidate": {
                "rank": 1,
                "top_results": [
                    {"chunk_id": "c1", "doc_id": "doc-A",
                     "section": "개요", "score": 0.85},
                    {"chunk_id": "c2", "doc_id": "doc-A",
                     "section": "기타", "score": 0.70},
                ],
            },
        },
    ])
    inputs = load_inputs_from_phase7_artifacts(
        per_query_path,
        chunks_jsonl=chunks_path,
        side="candidate",
    )
    assert len(inputs) == 1
    inp = inputs[0]
    assert inp.query_id == "qx"
    assert inp.gold_doc_id == "doc-A"
    assert inp.expected_title == "마법과고교의 열등생"
    # page_type='main' is NOT in the direct-match set, loader returns None
    # to avoid over-firing SECTION_INTENT_MISMATCH.
    assert inp.expected_section_type is None
    assert len(inp.top_candidates) == 2
    assert inp.top_candidates[0].chunk_id == "c1"
    # Enrichment from chunks_jsonl applied:
    assert inp.top_candidates[0].title == "마법과고교의 열등생"
    assert inp.top_candidates[0].retrieval_title == "마법과고교의 열등생 / 설정"
    assert inp.top_candidates[0].section_path == ("개요",)
    assert inp.top_candidates[0].section_type == "summary"
    # decide() runs end-to-end without error.
    v = decide(inp)
    assert v.query_id == "qx"
    assert v.signals.candidate_count == 2


def test_loader_merges_phase7_1_rerank_demotion_flag(tmp_path):
    """Phase 7.1 rerank input → rerank_demoted_gold and rerank score merge."""
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks_path, [
        {"chunk_id": "c1", "doc_id": "doc-A",
         "title": "T", "retrieval_title": "T",
         "section_path": ["개요"], "section_type": "summary"},
    ])
    per_query_path = tmp_path / "per_query.jsonl"
    _write_jsonl(per_query_path, [
        {
            "qid": "qy", "query": "q", "expected_doc_ids": ["doc-A"],
            "bucket": "main_work",
            "v4_meta": {"bucket": "main_work", "retrieval_title": "T"},
            "candidate": {
                "top_results": [
                    {"chunk_id": "c1", "doc_id": "doc-A",
                     "section": "개요", "score": 0.85},
                ],
            },
        },
    ])
    rerank_path = tmp_path / "rerank_per_query.jsonl"
    _write_jsonl(rerank_path, [
        {
            "qid": "qy",
            "candidate": {
                "rank": -1, "gold_was_demoted": True,
                "top_results": [
                    {"chunk_id": "c1", "doc_id": "doc-A",
                     "section": "개요", "score": 0.85},
                ],
            },
            "candidate_pool_preview": [
                {"chunk_id": "c1", "rerank_score": 0.92},
            ],
        },
    ])
    inputs = load_inputs_from_phase7_artifacts(
        per_query_path,
        chunks_jsonl=chunks_path,
        side="candidate",
        rerank_per_query_path=rerank_path,
    )
    assert len(inputs) == 1
    inp = inputs[0]
    assert inp.rerank_demoted_gold is True
    assert inp.top_candidates[0].rerank_score == pytest.approx(0.92, abs=1e-9)


def test_loader_baseline_side_uses_baseline_top_results(tmp_path):
    """side='baseline' must read from the baseline block, not candidate."""
    per_query_path = tmp_path / "per_query.jsonl"
    _write_jsonl(per_query_path, [
        {
            "qid": "qz", "query": "q",
            "expected_doc_ids": ["doc-A"],
            "bucket": "main_work",
            "v4_meta": {"bucket": "main_work"},
            "baseline": {
                "top_results": [
                    {"chunk_id": "b1", "doc_id": "doc-B",
                     "section": "개요", "score": 0.50},
                ],
            },
            "candidate": {
                "top_results": [
                    {"chunk_id": "c1", "doc_id": "doc-A",
                     "section": "개요", "score": 0.85},
                ],
            },
        },
    ])
    inputs = load_inputs_from_phase7_artifacts(per_query_path, side="baseline")
    assert inputs[0].top_candidates[0].chunk_id == "b1"
    assert inputs[0].top_candidates[0].doc_id == "doc-B"
