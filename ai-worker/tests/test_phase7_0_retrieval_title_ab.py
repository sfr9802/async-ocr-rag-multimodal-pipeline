"""Phase 7.0 — tests for the retrieval_title A/B pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from eval.harness.embedding_text_builder import (
    EMBEDDING_TEXT_VARIANTS,
    EmbeddingTextInput,
    V4EmbeddingTextInput,
    VARIANT_RAW,
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_SECTION,
    VARIANT_TITLE,
    VARIANT_TITLE_SECTION,
    build_embedding_text,
    build_v4_embedding_text,
)
from eval.harness.v4_ab_eval import (
    AbResult,
    QueryRecord,
    _classify,
    _per_query_metrics,
    run_paired_ab,
    write_ab_outputs,
)
from eval.harness.v4_chunk_export import (
    V4_EXPORT_VARIANTS,
    export_v4_chunks,
    recompute_embedding_text,
)
from eval.harness.v4_index_builder import (
    V4_INDEX_VARIANTS,
    v4_default_cache_dir,
    v4_variant_cache_key,
)
from eval.harness.v4_silver_queries import (
    _GENERIC_PAGE_TITLES,
    _is_generic_page_title,
    generate_v4_silver_queries,
)
from eval.harness.v4_variant_diff_report import (
    compute_variant_diff,
    render_variant_diff_md,
    write_variant_diff_report,
)


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------


def test_retrieval_title_section_uses_retrieval_title_when_present():
    chunk = V4EmbeddingTextInput(
        chunk_text="본문 텍스트입니다.",
        page_title="등장인물",
        retrieval_title="가난뱅이 신이! / 등장인물",
        section_path=("등장인물",),
        section_type="character",
    )
    out = build_v4_embedding_text(
        chunk, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert "가난뱅이 신이! / 등장인물" in out
    assert out.startswith("제목: 가난뱅이 신이! / 등장인물")


def test_retrieval_title_section_falls_back_to_page_title():
    chunk = V4EmbeddingTextInput(
        chunk_text="본문",
        page_title="ARIA The ORIGINATION",
        retrieval_title="",   # missing → fall back
        section_path=("개요",),
        section_type="summary",
    )
    out = build_v4_embedding_text(
        chunk, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert out.startswith("제목: ARIA The ORIGINATION")


def test_title_section_baseline_uses_page_title_only():
    chunk = V4EmbeddingTextInput(
        chunk_text="본문",
        page_title="등장인물",
        retrieval_title="가난뱅이 신이! / 등장인물",  # set but ignored
        section_path=("등장인물",),
        section_type="character",
    )
    out = build_v4_embedding_text(chunk, variant=VARIANT_TITLE_SECTION)
    # Baseline must use page_title verbatim — the work-prefixed form
    # MUST NOT appear or the A/B is contaminated.
    assert out.startswith("제목: 등장인물")
    assert "가난뱅이 신이!" not in out


def test_v4_format_matches_phase6_3_layout():
    """Spot-check the four labels + body separator against the spec."""
    chunk = V4EmbeddingTextInput(
        chunk_text="본문 텍스트",
        page_title="제목값",
        retrieval_title="제목값",
        section_path=("음악", "주제가", "OP"),
        section_type="music",
    )
    out = build_v4_embedding_text(chunk, variant=VARIANT_TITLE_SECTION)
    assert out == (
        "제목: 제목값\n"
        "섹션: 음악 > 주제가 > OP\n"
        "섹션타입: music\n"
        "\n"
        "본문:\n본문 텍스트"
    )


def test_legacy_title_section_output_unchanged():
    """The v3 ``title_section`` output must be byte-identical to before."""
    chunk = EmbeddingTextInput(
        text="문단 텍스트",
        title="작품 제목",
        section="개요",
    )
    expected = "작품 제목\n개요\n문단 텍스트"
    assert build_embedding_text(chunk, variant=VARIANT_TITLE_SECTION) == expected
    # Other v3 variants also unaffected by the v4 additions.
    assert build_embedding_text(chunk, variant=VARIANT_RAW) == "문단 텍스트"
    assert build_embedding_text(chunk, variant=VARIANT_TITLE) == "작품 제목\n문단 텍스트"
    assert build_embedding_text(chunk, variant=VARIANT_SECTION) == "개요\n문단 텍스트"


def test_legacy_builder_rejects_v4_variant():
    """Calling the legacy v3 builder with the v4 variant must raise."""
    chunk = EmbeddingTextInput(text="x", title="t", section="s")
    with pytest.raises(ValueError, match="v4 schema"):
        build_embedding_text(chunk, variant=VARIANT_RETRIEVAL_TITLE_SECTION)


def test_v4_builder_rejects_unknown_variant():
    chunk = V4EmbeddingTextInput(chunk_text="x", page_title="t")
    with pytest.raises(ValueError):
        build_v4_embedding_text(chunk, variant="raw")
    with pytest.raises(ValueError):
        build_v4_embedding_text(chunk, variant="bogus")


def test_retrieval_title_section_listed_in_known_variants():
    assert VARIANT_RETRIEVAL_TITLE_SECTION in EMBEDDING_TEXT_VARIANTS


# ---------------------------------------------------------------------------
# Chunk export
# ---------------------------------------------------------------------------


def _make_phase63_chunk_record(**overrides: Any) -> Dict[str, Any]:
    """Mock chunk in Phase 6.3 schema."""
    rec = {
        "schema_version": "namu_anime_v4_rag_chunk",
        "chunk_id": "abc123",
        "doc_id": "doc-xyz",
        "title": "등장인물",
        "aliases": ["등장인물"],
        "section_id": "sec-1",
        "section_key": "key-1",
        "section_path": ["등장인물"],
        "section_type": "character",
        "chunk_text": "주인공은...",
        "embedding_text": "제목: 등장인물\n섹션: 등장인물\n섹션타입: character\n\n본문:\n주인공은...",
        "metadata": {"source_url": "https://x/", "is_stub": False},
        "display_title": "가난뱅이 신이! / 등장인물",
        "retrieval_title": "가난뱅이 신이! / 등장인물",
    }
    rec.update(overrides)
    return rec


def test_recompute_embedding_text_baseline_matches_phase63():
    rec = _make_phase63_chunk_record()
    new_text = recompute_embedding_text(
        rec, variant=VARIANT_TITLE_SECTION,
    )
    assert new_text == rec["embedding_text"]


def test_recompute_embedding_text_candidate_uses_retrieval_title():
    rec = _make_phase63_chunk_record()
    new_text = recompute_embedding_text(
        rec, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert "가난뱅이 신이! / 등장인물" in new_text
    assert new_text != rec["embedding_text"]


def test_export_v4_chunks_writes_both_variants(tmp_path: Path) -> None:
    src = tmp_path / "rag_chunks.jsonl"
    src.write_text(
        "\n".join(json.dumps(_make_phase63_chunk_record(
            chunk_id=f"c{i}", doc_id=f"d{i}",
        )) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    for variant in V4_EXPORT_VARIANTS:
        out = tmp_path / f"rag_chunks_{variant}.jsonl"
        summary = export_v4_chunks(src, out, variant=variant)
        assert out.exists()
        manifest = tmp_path / f"manifest_{variant}.json"
        assert manifest.exists()
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m["variant"] == variant
        assert m["total_chunks"] == 3
        if variant == VARIANT_TITLE_SECTION:
            # baseline reproduces stored embedding_text → no changes
            assert m["changed_embedding_text_count"] == 0
        else:
            # candidate replaces the title segment everywhere → 3 changes
            assert m["changed_embedding_text_count"] == 3


def test_export_cli_module_accepts_retrieval_title_section():
    """The Phase 7.0 export tuple must include both variants."""
    assert VARIANT_TITLE_SECTION in V4_EXPORT_VARIANTS
    assert VARIANT_RETRIEVAL_TITLE_SECTION in V4_EXPORT_VARIANTS


# ---------------------------------------------------------------------------
# Variant diff report
# ---------------------------------------------------------------------------


def _write_phase63_fixtures(tmp_path: Path) -> Dict[str, Path]:
    """Two-page fixture: one generic page_title row, one neutral row."""
    pages = [
        {
            "schema_version": "namu_anime_v4_page",
            "page_id": "p1",
            "work_id": "w1",
            "work_title": "가난뱅이 신이!",
            "page_title": "등장인물",
            "page_type": "character",
            "relation": "subpage",
            "canonical_url": "https://x/work1/char",
            "title_source": "canonical_url",
            "alias_source": "fallback",
            "aliases": ["등장인물"],
            "categories": [],
            "source": {}, "crawl": {},
            "sections": [{
                "section_id": "s1", "heading_path": ["등장인물"],
                "depth": 2, "order": 0,
                "text": "주인공은 강철수.",
                "clean_text": "주인공은 강철수.",
                "section_key": "k1", "section_type": "character",
                "summary": None, "keywords": [], "entities": [],
                "relations": [], "qa_candidates": [],
                "quality": {},
            }],
            "display_title": "가난뱅이 신이! / 등장인물",
            "retrieval_title": "가난뱅이 신이! / 등장인물",
        },
        {
            "schema_version": "namu_anime_v4_page",
            "page_id": "p2",
            "work_id": "w2",
            "work_title": "ARIA The ORIGINATION",
            "page_title": "ARIA The ORIGINATION",
            "page_type": "work",
            "relation": "main",
            "canonical_url": "https://x/work2",
            "title_source": "seed",
            "alias_source": "fallback",
            "aliases": ["ARIA The ORIGINATION"],
            "categories": [],
            "source": {}, "crawl": {},
            "sections": [{
                "section_id": "s2", "heading_path": ["개요"],
                "depth": 2, "order": 0,
                "text": "TV 애니메이션.",
                "clean_text": "TV 애니메이션.",
                "section_key": "k2", "section_type": "summary",
                "summary": None, "keywords": [], "entities": [],
                "relations": [], "qa_candidates": [],
                "quality": {},
            }],
            "display_title": "ARIA The ORIGINATION",
            "retrieval_title": "ARIA The ORIGINATION",
        },
    ]
    chunks = [
        _make_phase63_chunk_record(
            chunk_id="c1", doc_id="p1",
            title="등장인물", retrieval_title="가난뱅이 신이! / 등장인물",
            section_path=["등장인물"], section_type="character",
            embedding_text="제목: 등장인물\n섹션: 등장인물\n섹션타입: character\n\n본문:\n주인공은 강철수.",
            chunk_text="주인공은 강철수.",
        ),
        _make_phase63_chunk_record(
            chunk_id="c2", doc_id="p2",
            title="ARIA The ORIGINATION",
            retrieval_title="ARIA The ORIGINATION",
            section_path=["개요"], section_type="summary",
            embedding_text="제목: ARIA The ORIGINATION\n섹션: 개요\n섹션타입: summary\n\n본문:\nTV 애니메이션.",
            chunk_text="TV 애니메이션.",
        ),
    ]
    pages_path = tmp_path / "pages_v4.jsonl"
    chunks_path = tmp_path / "rag_chunks.jsonl"
    pages_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pages) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return {"pages": pages_path, "chunks": chunks_path}


def test_variant_diff_counts_changed_and_unchanged(tmp_path: Path) -> None:
    fx = _write_phase63_fixtures(tmp_path)
    base_path = tmp_path / "base.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    export_v4_chunks(fx["chunks"], base_path, variant=VARIANT_TITLE_SECTION)
    export_v4_chunks(
        fx["chunks"], cand_path, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )

    report = compute_variant_diff(
        base_path, cand_path,
        pages_v4_path=fx["pages"],
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert report["total_chunks"] == 2
    # c1 (page_title=등장인물) changes; c2 (page_title==retrieval_title) does not.
    assert report["changed_embedding_text_count"] == 1
    assert abs(report["changed_embedding_text_ratio"] - 0.5) < 1e-9
    assert report["integrity_non_embedding_text_diffs"] == 0

    # Breakdowns: the changed chunk lives under page_type=character / section_type=character
    page_type_breakdown = {b["key"]: b for b in report["changed_by_page_type"]}
    assert page_type_breakdown["character"]["changed"] == 1
    assert page_type_breakdown["work"]["changed"] == 0


def test_variant_diff_examples_carry_old_and_new_previews(tmp_path: Path) -> None:
    fx = _write_phase63_fixtures(tmp_path)
    base_path = tmp_path / "base.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    export_v4_chunks(fx["chunks"], base_path, variant=VARIANT_TITLE_SECTION)
    export_v4_chunks(
        fx["chunks"], cand_path, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    report = compute_variant_diff(
        base_path, cand_path,
        pages_v4_path=fx["pages"],
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert report["top_examples"]
    ex = report["top_examples"][0]
    assert ex["page_title"] == "등장인물"
    assert ex["retrieval_title"] == "가난뱅이 신이! / 등장인물"
    assert "등장인물" in ex["old_embedding_text_preview"]
    assert "가난뱅이 신이!" in ex["new_embedding_text_preview"]


def test_variant_diff_report_writes_both_files(tmp_path: Path) -> None:
    fx = _write_phase63_fixtures(tmp_path)
    base_path = tmp_path / "base.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    export_v4_chunks(fx["chunks"], base_path, variant=VARIANT_TITLE_SECTION)
    export_v4_chunks(
        fx["chunks"], cand_path, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    report = compute_variant_diff(
        base_path, cand_path,
        pages_v4_path=fx["pages"],
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    out_dir = tmp_path / "diff_out"
    json_path, md_path = write_variant_diff_report(
        report, out_dir=out_dir,
    )
    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "v4 variant diff" in md
    assert "page_type" in md


# ---------------------------------------------------------------------------
# v4 silver queries
# ---------------------------------------------------------------------------


def test_silver_queries_generate_for_v4_pages(tmp_path: Path) -> None:
    fx = _write_phase63_fixtures(tmp_path)
    queries = generate_v4_silver_queries(
        fx["pages"], target_total=2, seed=42,
        # ensure both buckets fire
        bucket_ratios={"subpage_generic": 0.5, "main_work": 0.5,
                       "subpage_named": 0.0},
    )
    assert len(queries) == 2
    buckets = {q["v4_meta"]["bucket"] for q in queries}
    assert "subpage_generic" in buckets
    assert "main_work" in buckets

    # subpage_generic must reference the work_title in the query string
    for q in queries:
        if q["v4_meta"]["bucket"] == "subpage_generic":
            assert "가난뱅이 신이!" in q["query"]
            assert q["expected_doc_ids"] == ["p1"]


def test_is_generic_page_title_covers_phase6_3_set():
    assert _is_generic_page_title("등장인물")
    assert _is_generic_page_title("기타 등장인물")
    assert _is_generic_page_title("OST")
    assert not _is_generic_page_title("ARIA The ORIGINATION")
    assert not _is_generic_page_title("")


def test_subpage_generic_uses_retrieval_title_for_parent_work(
    tmp_path: Path,
) -> None:
    """When ``page.work_title`` was preserved as the page_title verbatim
    on Phase 6.3 subpages, the parent must come from retrieval_title."""
    # Mimic the v4 reality: page_title == work_title == "기타 등장인물",
    # but retrieval_title carries the actual parent work prefix.
    page_rec: Dict[str, Any] = {
        "schema_version": "namu_anime_v4_page",
        "page_id": "p_sub", "work_id": "w_sub",
        "work_title": "기타 등장인물",
        "page_title": "기타 등장인물",
        "page_type": "character", "relation": "subpage",
        "canonical_url": "https://x/sub", "title_source": "canonical_url",
        "alias_source": "fallback", "aliases": ["기타 등장인물"],
        "categories": [], "source": {}, "crawl": {},
        "sections": [{
            "section_id": "s", "heading_path": ["등장인물"],
            "depth": 2, "order": 0,
            "text": "조연으로 등장한 인물들입니다.",
            "clean_text": "조연으로 등장한 인물들입니다.",
            "section_key": "k", "section_type": "character",
            "summary": None, "keywords": [], "entities": [],
            "relations": [], "qa_candidates": [], "quality": {},
        }],
        "display_title": "디트로이트 메탈 시티 / 기타 등장인물",
        "retrieval_title": "디트로이트 메탈 시티/기타 등장인물",
    }
    p = tmp_path / "pages.jsonl"
    p.write_text(json.dumps(page_rec, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    queries = generate_v4_silver_queries(
        p, target_total=1, seed=42,
        bucket_ratios={
            "subpage_generic": 1.0, "main_work": 0.0, "subpage_named": 0.0,
        },
    )
    assert len(queries) == 1
    q = queries[0]
    assert "디트로이트 메탈 시티" in q["query"]
    # Must NOT produce the degenerate "기타 등장인물의 기타 등장인물..." form
    assert q["query"].count("기타 등장인물") == 1


def test_subpage_generic_skipped_when_parent_unavailable(
    tmp_path: Path,
) -> None:
    """A page where retrieval_title provides no parent prefix must be
    skipped — the resulting query would have no signal to anchor on."""
    page_rec: Dict[str, Any] = {
        "schema_version": "namu_anime_v4_page",
        "page_id": "p_orphan", "work_id": "w",
        "work_title": "등장인물", "page_title": "등장인물",
        "page_type": "character", "relation": "subpage",
        "canonical_url": "https://x/orphan", "title_source": "canonical_url",
        "alias_source": "fallback", "aliases": [],
        "categories": [], "source": {}, "crawl": {},
        "sections": [{
            "section_id": "s", "heading_path": ["등장인물"],
            "depth": 2, "order": 0, "text": "x", "clean_text": "x",
            "section_key": "k", "section_type": "character",
            "summary": None, "keywords": [], "entities": [],
            "relations": [], "qa_candidates": [], "quality": {},
        }],
        "display_title": "등장인물",
        "retrieval_title": "등장인물",  # no parent prefix
    }
    p = tmp_path / "pages.jsonl"
    p.write_text(json.dumps(page_rec, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    queries = generate_v4_silver_queries(
        p, target_total=1, seed=42,
        bucket_ratios={
            "subpage_generic": 1.0, "main_work": 0.0, "subpage_named": 0.0,
        },
    )
    assert queries == []


# ---------------------------------------------------------------------------
# Index slug / cache key
# ---------------------------------------------------------------------------


def test_v4_cache_key_includes_variant(tmp_path: Path) -> None:
    chunks_path = tmp_path / "rag_chunks.jsonl"
    chunks_path.write_text("{}\n", encoding="utf-8")
    base_key = v4_variant_cache_key(
        chunks_path, "BAAI/bge-m3", 1024, VARIANT_TITLE_SECTION,
    )
    cand_key = v4_variant_cache_key(
        chunks_path, "BAAI/bge-m3", 1024, VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert base_key != cand_key
    assert len(base_key) == 16
    assert len(cand_key) == 16


def test_v4_default_cache_dir_includes_variant_slug(tmp_path: Path) -> None:
    base_dir = v4_default_cache_dir(
        cache_root=tmp_path,
        embedding_model="BAAI/bge-m3",
        max_seq_length=1024,
        variant=VARIANT_TITLE_SECTION,
    )
    cand_dir = v4_default_cache_dir(
        cache_root=tmp_path,
        embedding_model="BAAI/bge-m3",
        max_seq_length=1024,
        variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert base_dir != cand_dir
    assert "title-section" in base_dir.name
    assert "retrieval-title-section" in cand_dir.name
    assert "mseq1024" in base_dir.name
    assert "mseq1024" in cand_dir.name


def test_v4_default_cache_dir_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        v4_default_cache_dir(
            cache_root=tmp_path,
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            variant=VARIANT_RAW,
        )


def test_v4_index_supported_variants_are_phase7_0_pair():
    assert set(V4_INDEX_VARIANTS) == {
        VARIANT_TITLE_SECTION, VARIANT_RETRIEVAL_TITLE_SECTION,
    }


# ---------------------------------------------------------------------------
# A/B classification
# ---------------------------------------------------------------------------


def _make_metrics(rank: int) -> Any:
    """Synthesize a PerQueryMetrics-shaped object for the classifier."""
    from eval.harness.v4_ab_eval import PerQueryMetrics
    return PerQueryMetrics(
        rank=rank,
        hit_at={1: int(0 < rank <= 1), 3: int(0 < rank <= 3),
                5: int(0 < rank <= 5), 10: int(0 < rank <= 10)},
        mrr_at_10=(1.0 / rank) if 0 < rank <= 10 else 0.0,
        ndcg_at_10=0.0,
        dup_rate=0.0,
        same_title_collisions=0,
        top_results=[],
    )


def test_classify_improved_when_candidate_finds_missed_baseline():
    assert _classify(_make_metrics(-1), _make_metrics(3)) == "improved"


def test_classify_regressed_when_candidate_loses_baseline_hit():
    assert _classify(_make_metrics(2), _make_metrics(-1)) == "regressed"


def test_classify_improved_when_candidate_ranks_higher():
    assert _classify(_make_metrics(7), _make_metrics(2)) == "improved"


def test_classify_regressed_when_candidate_ranks_lower():
    assert _classify(_make_metrics(2), _make_metrics(7)) == "regressed"


def test_classify_both_hit_at_same_rank():
    assert _classify(_make_metrics(3), _make_metrics(3)) == "both_hit"


def test_classify_both_missed():
    assert _classify(_make_metrics(-1), _make_metrics(-1)) == "both_missed"


# ---------------------------------------------------------------------------
# A/B end-to-end with mock retrievers + writer
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, doc_id: str, title: str = "T") -> None:
        self.chunk_id = f"chunk-{doc_id}"
        self.doc_id = doc_id
        self.title = title
        self.section = "sec"
        self.score = 0.5


class _FakeReport:
    def __init__(self, doc_ids: List[str]) -> None:
        self.results = [_FakeChunk(d) for d in doc_ids]


class _FakeRetriever:
    """Minimal stand-in for :class:`Retriever`.

    Resolves to a programmed top-k per qid so a test can prescribe
    improvements / regressions without spinning up a real index.
    """

    def __init__(self, table: Dict[str, List[str]]) -> None:
        self._table = table

    def retrieve(self, query: str) -> Any:
        return _FakeReport(self._table.get(query, []))


def test_run_paired_ab_writes_artefacts(tmp_path: Path) -> None:
    queries = [
        QueryRecord(
            qid="q1", query="q1-text", expected_doc_ids=("d-target",),
            answer_type="title_lookup", difficulty="easy",
            bucket="main_work", v4_meta={"bucket": "main_work"},
        ),
        QueryRecord(
            qid="q2", query="q2-text", expected_doc_ids=("d-target",),
            answer_type="subpage_lookup", difficulty="medium",
            bucket="subpage_generic",
            v4_meta={"bucket": "subpage_generic"},
        ),
        QueryRecord(
            qid="q3", query="q3-text", expected_doc_ids=("d-target",),
            answer_type="subpage_lookup", difficulty="medium",
            bucket="subpage_generic",
            v4_meta={"bucket": "subpage_generic"},
        ),
        QueryRecord(
            qid="q4", query="q4-text", expected_doc_ids=("d-target",),
            answer_type="title_lookup", difficulty="easy",
            bucket="main_work", v4_meta={"bucket": "main_work"},
        ),
    ]
    # q1: both miss
    # q2: candidate improves rank from miss → 1
    # q3: candidate regresses from rank 1 → miss
    # q4: tied at rank 1
    base_table = {
        "q1-text": ["d-other"],
        "q2-text": ["d-other"],
        "q3-text": ["d-target", "d-other"],
        "q4-text": ["d-target"],
    }
    cand_table = {
        "q1-text": ["d-other"],
        "q2-text": ["d-target"],
        "q3-text": ["d-other"],
        "q4-text": ["d-target"],
    }
    result = run_paired_ab(
        queries,
        baseline_retriever=_FakeRetriever(base_table),
        candidate_retriever=_FakeRetriever(cand_table),
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    paths = write_ab_outputs(
        result,
        out_dir=tmp_path / "ab",
        baseline_variant=VARIANT_TITLE_SECTION,
        candidate_variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["status_counts"]["improved"] == 1
    assert summary["status_counts"]["regressed"] == 1
    assert summary["status_counts"]["both_hit"] == 1
    assert summary["status_counts"]["both_missed"] == 1
    assert paths["per_query"].exists()
    assert paths["improved"].exists()
    assert paths["regressed"].exists()

    improved_lines = [
        json.loads(l) for l in paths["improved"].read_text(
            encoding="utf-8"
        ).splitlines() if l.strip()
    ]
    assert len(improved_lines) == 1
    assert improved_lines[0]["qid"] == "q2"

    regressed_lines = [
        json.loads(l) for l in paths["regressed"].read_text(
            encoding="utf-8"
        ).splitlines() if l.strip()
    ]
    assert len(regressed_lines) == 1
    assert regressed_lines[0]["qid"] == "q3"

    # Aggregate metrics: candidate should be flat or better at hit@10
    base_hit10 = summary["baseline"]["hit_at_10"]
    cand_hit10 = summary["candidate"]["hit_at_10"]
    # Two queries hit baseline (q3, q4); two hit candidate (q2, q4) → tie.
    assert base_hit10 == cand_hit10 == 0.5
