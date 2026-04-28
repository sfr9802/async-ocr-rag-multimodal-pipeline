"""Tests for the Phase 2B metadata boost scorer + reranker.

The scorer wraps three signals (title exact, title partial, section
keyword / path) plus a max_boost clamp and an excluded-section block.
The reranker plugs into the RerankerProvider contract with a strict
"boost off → byte-identical to dense" guarantee.

Coverage:

  - title exact / partial firing rules
  - section keyword + path firing rules
  - excluded sections block boost
  - max_boost clamp shapes the total
  - missing doc_meta returns empty
  - disabled config returns empty score AND keeps input ordering
  - re-order is by descending final = dense + boost
  - last_boost_breakdown / last_normalized_query expose call state
  - invalid config raises ValueError with all errors listed
  - title_partial respects title_min_len
"""

from __future__ import annotations

from typing import List

import pytest

from app.capabilities.rag.generation import RetrievedChunk
from eval.harness.boost_metadata import DocBoostMetadata, doc_metadata_from_records
from eval.harness.boost_scorer import (
    DEFAULT_EXCLUDED_SECTIONS,
    BoostConfig,
    BoostScore,
    MetadataBoostReranker,
    compute_boost_score,
)
from eval.harness.query_normalizer import normalize_for_match


def _meta(doc_id: str, title: str, sections=("요약", "본문")) -> DocBoostMetadata:
    metas = doc_metadata_from_records(
        [
            {
                "doc_id": doc_id,
                "title": title,
                "seed": title,
                "sections": {n: {"chunks": ["x"]} for n in sections},
            }
        ]
    )
    return metas[doc_id]


def _chunk(chunk_id: str, doc_id: str, section: str, score: float, text: str = "x"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section=section,
        text=text,
        score=score,
    )


# ---------------------------------------------------------------------------
# BoostConfig
# ---------------------------------------------------------------------------


class TestBoostConfig:
    def test_disabled_factory_is_disabled(self):
        cfg = BoostConfig.disabled()
        assert cfg.is_disabled() is True

    def test_explicit_zero_weights_is_disabled(self):
        cfg = BoostConfig(
            title_exact_boost=0.0,
            title_partial_boost=0.0,
            section_keyword_boost=0.0,
            section_path_boost=0.0,
        )
        assert cfg.is_disabled() is True

    def test_any_positive_weight_is_enabled(self):
        cfg = BoostConfig(title_exact_boost=0.1)
        assert cfg.is_disabled() is False

    def test_validate_negative_weight(self):
        cfg = BoostConfig(title_exact_boost=-0.1)
        errors = cfg.validate()
        assert any("title_exact_boost" in e for e in errors)

    def test_validate_min_len(self):
        cfg = BoostConfig(title_min_len=0)
        errors = cfg.validate()
        assert any("title_min_len" in e for e in errors)

    def test_to_dict_round_trips_excluded(self):
        cfg = BoostConfig(
            title_exact_boost=0.1,
            excluded_sections=("요약",),
        )
        out = cfg.to_dict()
        assert out["excluded_sections"] == ["요약"]


# ---------------------------------------------------------------------------
# compute_boost_score
# ---------------------------------------------------------------------------


class TestComputeBoostScore:
    def test_disabled_returns_empty(self):
        cfg = BoostConfig.disabled()
        score = compute_boost_score(
            query_normalized="템플",
            section="요약",
            doc_meta=_meta("d1", "템플"),
            config=cfg,
        )
        assert score == BoostScore.empty()

    def test_missing_doc_meta_returns_empty(self):
        cfg = BoostConfig(title_exact_boost=0.1)
        score = compute_boost_score(
            query_normalized="anything",
            section="요약",
            doc_meta=None,
            config=cfg,
        )
        assert score == BoostScore.empty()

    def test_title_exact_match_only(self):
        cfg = BoostConfig(title_exact_boost=0.1, title_partial_boost=0.05)
        score = compute_boost_score(
            query_normalized=normalize_for_match("템플의 주요 주제"),
            section="요약",
            doc_meta=_meta("d1", "템플"),
            config=cfg,
        )
        assert score.title_exact == pytest.approx(0.1)
        assert score.title_partial == 0.0
        assert score.title_match_kind == "exact"
        assert score.matched_title == "템플"

    def test_title_partial_when_no_exact(self):
        cfg = BoostConfig(
            title_exact_boost=0.10,
            title_partial_boost=0.05,
            title_min_len=2,
        )
        # Title "마법소녀 리리컬 나노하" — query has "마법소녀" but not the
        # full title verbatim. Expected: partial-match firing.
        meta = _meta("d1", "마법소녀 리리컬 나노하")
        score = compute_boost_score(
            query_normalized=normalize_for_match("마법소녀와 리리컬 작품"),
            section="요약",
            doc_meta=meta,
            config=cfg,
        )
        # Both "마법소녀" AND "리리컬" appear in the query → partial fires.
        # Exact does NOT fire because the full normalized title isn't
        # a contiguous substring of the query.
        assert score.title_match_kind == "partial"
        assert score.title_partial == pytest.approx(0.05)
        assert score.title_exact == 0.0

    def test_title_partial_min_len_filters_short_tokens(self):
        cfg = BoostConfig(title_partial_boost=0.05, title_min_len=4)
        # Title tokens "마법" / "X" / "ABC" all < 4 chars → partial blocked.
        meta = _meta("d1", "마법 X ABC")
        score = compute_boost_score(
            query_normalized="abc 마법",
            section="요약",
            doc_meta=meta,
            config=cfg,
        )
        assert score.title_match_kind is None
        assert score.title_partial == 0.0

    def test_section_keyword_fires(self):
        cfg = BoostConfig(section_keyword_boost=0.03)
        # Use a non-excluded section name; "줄거리" appears in the query.
        meta = _meta("d1", "X", sections=("줄거리", "본문"))
        score = compute_boost_score(
            query_normalized=normalize_for_match("줄거리를 알려줘"),
            section="줄거리",
            doc_meta=meta,
            config=cfg,
        )
        assert score.section_keyword == pytest.approx(0.03)
        assert score.matched_section == "줄거리"

    def test_section_keyword_blocked_by_excluded(self):
        cfg = BoostConfig(
            section_keyword_boost=0.03,
            excluded_sections=("요약",),
        )
        meta = _meta("d1", "X", sections=("요약",))
        score = compute_boost_score(
            query_normalized="요약 좀 보여줘",
            section="요약",
            doc_meta=meta,
            config=cfg,
        )
        assert score.section_keyword == 0.0
        assert score.matched_section is None

    def test_section_path_fires_when_other_section_in_query(self):
        cfg = BoostConfig(section_path_boost=0.02, section_keyword_boost=0.0)
        meta = _meta("d1", "X", sections=("줄거리", "등장인물"))
        score = compute_boost_score(
            query_normalized="등장인물 알려줘",
            section="요약",  # the chunk lives in 요약, but title pulls 등장인물
            doc_meta=meta,
            config=cfg,
        )
        assert score.section_path == pytest.approx(0.02)

    def test_max_boost_clamp(self):
        cfg = BoostConfig(
            title_exact_boost=0.20,
            section_keyword_boost=0.20,
            max_boost=0.10,
        )
        meta = _meta("d1", "줄거리", sections=("줄거리",))
        score = compute_boost_score(
            query_normalized=normalize_for_match("줄거리"),
            section="줄거리",
            doc_meta=meta,
            config=cfg,
        )
        # Raw total = 0.40 but clamp keeps it at 0.10.
        assert score.title_exact == pytest.approx(0.20)
        assert score.section_keyword == pytest.approx(0.20)
        assert score.total == pytest.approx(0.10)

    def test_no_clamp_when_max_boost_zero(self):
        cfg = BoostConfig(title_exact_boost=0.20, max_boost=0.0)
        meta = _meta("d1", "x")
        score = compute_boost_score(
            query_normalized="x",
            section="줄거리",
            doc_meta=meta,
            config=cfg,
        )
        # max_boost = 0 disables the clamp; total = raw sum.
        assert score.total == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# MetadataBoostReranker
# ---------------------------------------------------------------------------


class TestMetadataBoostReranker:
    def test_off_path_byte_identical_ordering(self):
        cfg = BoostConfig.disabled()
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "X", "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunks = [
            _chunk("c1", "d1", "요약", 0.50),
            _chunk("c2", "d1", "본문", 0.40),
            _chunk("c3", "d1", "요약", 0.30),
        ]
        out = reranker.rerank("anything", chunks, k=3)
        # Same chunk ids, same order, same score field (byte-identical).
        assert [c.chunk_id for c in out] == ["c1", "c2", "c3"]
        assert [c.score for c in out] == [0.50, 0.40, 0.30]

    def test_off_path_records_empty_breakdown(self):
        cfg = BoostConfig.disabled()
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "X", "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunks = [_chunk("c1", "d1", "요약", 0.50)]
        reranker.rerank("q", chunks, k=1)
        assert reranker.last_boost_breakdown == {"c1": BoostScore.empty()}

    def test_boost_reorders_to_match_title(self):
        cfg = BoostConfig(title_exact_boost=0.20)
        meta = doc_metadata_from_records(
            [
                {"doc_id": "d-good", "title": "템플",
                 "sections": {"요약": {"chunks": ["x"]}}},
                {"doc_id": "d-bad", "title": "다른",
                 "sections": {"요약": {"chunks": ["x"]}}},
            ]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunks = [
            _chunk("bad-1", "d-bad", "요약", 0.55),
            _chunk("good-1", "d-good", "요약", 0.40),
        ]
        out = reranker.rerank("템플의 주요 주제", chunks, k=2)
        # Boost-aware ordering: good-1 (0.40 + 0.20) > bad-1 (0.55).
        assert [c.chunk_id for c in out] == ["good-1", "bad-1"]
        # Score field carries the composite final = dense + boost.
        assert out[0].score == pytest.approx(0.60)
        assert out[1].score == pytest.approx(0.55)

    def test_boost_score_separable_from_dense(self):
        cfg = BoostConfig(title_exact_boost=0.10)
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "x",
              "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunks = [_chunk("c1", "d1", "요약", 0.50)]
        out = reranker.rerank("x query", chunks, k=1)
        breakdown = reranker.last_boost_breakdown
        # final = dense + boost.total → boost recoverable from the diff.
        assert out[0].score == pytest.approx(0.60)
        assert breakdown["c1"].total == pytest.approx(0.10)
        assert pytest.approx(out[0].score - breakdown["c1"].total) == 0.50

    def test_invalid_config_raises_with_aggregated_messages(self):
        cfg = BoostConfig(
            title_exact_boost=-0.5,
            section_keyword_boost=-1.0,
            title_min_len=0,
        )
        with pytest.raises(ValueError) as exc:
            MetadataBoostReranker(config=cfg, doc_metadata={})
        msg = str(exc.value)
        assert "title_exact_boost" in msg
        assert "section_keyword_boost" in msg
        assert "title_min_len" in msg

    def test_truncates_to_k(self):
        cfg = BoostConfig.disabled()
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "x", "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunks = [_chunk(f"c{i}", "d1", "요약", float(10 - i)) for i in range(5)]
        out = reranker.rerank("q", chunks, k=2)
        assert len(out) == 2

    def test_empty_input_returns_empty(self):
        cfg = BoostConfig(title_exact_boost=0.1)
        reranker = MetadataBoostReranker(config=cfg, doc_metadata={})
        assert reranker.rerank("q", [], k=5) == []

    def test_last_normalized_query_recorded(self):
        cfg = BoostConfig(title_exact_boost=0.1)
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "x", "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        reranker.rerank("Hello, WORLD?", [_chunk("c1", "d1", "요약", 0.5)], k=1)
        nq = reranker.last_normalized_query
        assert nq is not None
        assert nq.raw == "Hello, WORLD?"
        # Punctuation collapsed and ASCII lowercased.
        assert nq.normalized == "hello world"

    def test_chunk_with_unknown_doc_id_gets_no_boost(self):
        cfg = BoostConfig(title_exact_boost=0.1)
        meta = doc_metadata_from_records(
            [{"doc_id": "d1", "title": "x", "sections": {"요약": {"chunks": ["x"]}}}]
        )
        reranker = MetadataBoostReranker(config=cfg, doc_metadata=meta)
        chunk = _chunk("c1", "missing-doc", "요약", 0.5)
        out = reranker.rerank("x", [chunk], k=1)
        # No boost because doc_meta missing → score stays at 0.5.
        assert out[0].score == pytest.approx(0.5)
        assert reranker.last_boost_breakdown["c1"].total == 0.0
