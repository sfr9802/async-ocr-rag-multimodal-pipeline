"""Phase 7 — tests for the silver-500 query generator.

Fixture-driven: each test composes a tiny ``pages_v4.jsonl`` and runs
the generator end-to-end. The generator never imports a model or
touches FAISS, so the fixtures are small and the tests stay fast.

Acceptance bar (per the spec):

  - Output is deterministic — same seed → byte-identical jsonl.
  - Bucket distribution is approximately respected (best-effort fill
    when the corpus can't supply enough rows).
  - Generated queries carry "silver" terminology and never "gold".
  - The summary report includes the silver/gold disclaimer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from eval.harness.silver_500_generator import (
    ALL_TEMPLATES,
    BUCKETS,
    BUCKET_MAIN_WORK,
    BUCKET_SUBPAGE_GENERIC,
    BUCKET_SUBPAGE_NAMED,
    DEFAULT_BUCKET_TARGETS,
    SILVER_AUDIT_LINE,
    SILVER_DISCLAIMER_LINE,
    TEMPLATE_ALIAS_LOOKUP,
    TEMPLATE_AMBIGUOUS_SHORT,
    TEMPLATE_EVALUATION,
    TEMPLATE_NAMED_ALIAS,
    TEMPLATE_NAMED_LOOKUP,
    TEMPLATE_NAMED_QUESTION,
    TEMPLATE_PLOT_SUMMARY,
    TEMPLATE_SECTION_DETAIL,
    TEMPLATE_SECTION_LOOKUP,
    TEMPLATE_SECTION_QUESTION,
    TEMPLATE_TITLE_LOOKUP,
    generate_silver_500,
    render_silver_500_summary_md,
    write_silver_500_outputs,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _page_record(
    page_id: str,
    *,
    work_id: str = "w-default",
    work_title: str = "테스트 작품",
    page_title: str = "테스트 작품",
    page_type: str = "work",
    relation: str = "main",
    retrieval_title: str = "",
    aliases: List[str] | None = None,
    title_source: str = "seed",
    alias_source: str = "fallback",
    section_text: str = "이 작품은 테스트용 샘플입니다.",
    section_heading: List[str] | None = None,
    section_type: str = "summary",
) -> Dict[str, Any]:
    """Compose one Phase 6.3 ``pages_v4`` record."""
    rt = retrieval_title or page_title
    return {
        "schema_version": "namu_anime_v4_page",
        "page_id": page_id,
        "work_id": work_id,
        "work_title": work_title,
        "page_title": page_title,
        "page_type": page_type,
        "relation": relation,
        "canonical_url": f"https://x/{page_id}",
        "title_source": title_source,
        "alias_source": alias_source,
        "aliases": aliases or [page_title],
        "categories": [], "source": {}, "crawl": {},
        "sections": [{
            "section_id": f"sec-{page_id}",
            "heading_path": section_heading or ["개요"],
            "depth": 2, "order": 0,
            "text": section_text, "clean_text": section_text,
            "section_key": f"k-{page_id}", "section_type": section_type,
            "summary": None, "keywords": [], "entities": [],
            "relations": [], "qa_candidates": [], "quality": {},
        }],
        "display_title": rt,
        "retrieval_title": rt,
    }


def _write_pages(tmp_path: Path, records: List[Dict[str, Any]]) -> Path:
    """Serialise ``records`` to a pages_v4.jsonl file."""
    p = tmp_path / "pages_v4.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return p


def _make_corpus(tmp_path: Path, n_each: int = 6) -> Path:
    """Build a balanced corpus that exercises every template family.

    Produces ``n_each`` rows of each kind:
      - main_work pages (with and without aliases),
      - subpage_generic pages (with parent prefix + generic page_title),
      - subpage_named pages (with parent prefix + non-generic page_title,
        some carrying distinct aliases).
    """
    records: List[Dict[str, Any]] = []
    for i in range(n_each):
        records.append(_page_record(
            page_id=f"main-{i:02d}",
            work_id=f"work-{i:02d}",
            work_title=f"메인작품{i:02d}",
            page_title=f"메인작품{i:02d}",
            page_type="work",
            relation="main",
            section_text="줄거리 본문 줄거리 안내 입니다.",
            section_type="summary",
            aliases=[f"메인작품{i:02d}", f"별칭{i:02d}"],
        ))
    # main_work with short title to exercise ambiguous_short
    for i in range(n_each):
        records.append(_page_record(
            page_id=f"short-{i:02d}",
            work_id=f"work-short-{i:02d}",
            work_title=f"AAA{i:02d}",
            page_title=f"AAA{i:02d}",
            page_type="work",
            relation="main",
            section_text="짧은 제목.",
            section_type="summary",
            aliases=[f"AAA{i:02d}"],
        ))
    for i in range(n_each):
        records.append(_page_record(
            page_id=f"sub-gen-{i:02d}",
            work_id=f"work-gen-{i:02d}",
            work_title=f"서브작품{i:02d}",
            page_title="등장인물",
            page_type="character",
            relation="character",
            retrieval_title=f"서브작품{i:02d} / 등장인물",
            title_source="canonical_url",
            section_text="캐릭터 정보 본문 입니다.",
            section_heading=["등장인물"],
            section_type="character",
            aliases=["등장인물"],
        ))
    for i in range(n_each):
        records.append(_page_record(
            page_id=f"sub-named-{i:02d}",
            work_id=f"work-named-{i:02d}",
            work_title=f"네임드작품{i:02d}",
            page_title=f"주인공{i:02d}",
            page_type="character",
            relation="subpage",
            retrieval_title=f"네임드작품{i:02d} / 주인공{i:02d}",
            title_source="canonical_url",
            section_text="주인공 캐릭터 설명.",
            section_heading=["주인공"],
            section_type="character",
            aliases=[f"주인공{i:02d}", f"닉네임{i:02d}"],
        ))
    return _write_pages(tmp_path, records)


# ---------------------------------------------------------------------------
# Schema / determinism
# ---------------------------------------------------------------------------


def test_silver_500_uses_silver_terminology_not_gold(tmp_path: Path):
    pages = _make_corpus(tmp_path, n_each=6)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 4,
            BUCKET_SUBPAGE_GENERIC: 4,
            BUCKET_SUBPAGE_NAMED: 4,
        },
        seed=42,
    )
    assert result.queries
    for q in result.queries:
        # ID prefix must be silver-flavoured.
        assert q["id"].startswith("v4-silver-500-")
        # Tags carry "silver" but never "gold" / "human_verified".
        tags = q["tags"]
        assert "silver" in tags
        assert "gold" not in tags
        assert "human_verified" not in tags
        # Meta must mark the row as silver, not gold.
        meta = q["v4_meta"]
        assert meta["is_silver_not_gold"] is True
        assert meta["silver_label_source"] in {
            "page_lookup", "subpage_pivot", "alias_pivot",
        }
        assert meta["silver_label_confidence"] in {"high", "medium", "low"}


def test_silver_500_summary_includes_disclaimer_lines(tmp_path: Path):
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 3,
            BUCKET_SUBPAGE_GENERIC: 3,
            BUCKET_SUBPAGE_NAMED: 3,
        },
        seed=42,
    )
    md = render_silver_500_summary_md(result.summary)
    assert SILVER_DISCLAIMER_LINE in md
    assert SILVER_AUDIT_LINE in md
    # Negative: no "human-verified gold" boast about the silver set.
    assert "human-verified gold" in md  # disclaimer references it
    assert "Use the `human_gold_seed_export`" in md


def test_silver_500_is_deterministic_under_same_seed(tmp_path: Path):
    pages = _make_corpus(tmp_path, n_each=5)
    targets = {
        BUCKET_MAIN_WORK: 3,
        BUCKET_SUBPAGE_GENERIC: 4,
        BUCKET_SUBPAGE_NAMED: 3,
    }
    r1 = generate_silver_500(pages, bucket_targets=targets, seed=42)
    r2 = generate_silver_500(pages, bucket_targets=targets, seed=42)
    assert [q["id"] for q in r1.queries] == [q["id"] for q in r2.queries]
    assert [q["query"] for q in r1.queries] == [q["query"] for q in r2.queries]
    assert [q["expected_doc_ids"] for q in r1.queries] == [
        q["expected_doc_ids"] for q in r2.queries
    ]


def test_silver_500_seed_changes_selection(tmp_path: Path):
    """Non-deterministic across seeds — guards against accidental hard-code."""
    pages = _make_corpus(tmp_path, n_each=6)
    targets = {
        BUCKET_MAIN_WORK: 4,
        BUCKET_SUBPAGE_GENERIC: 4,
        BUCKET_SUBPAGE_NAMED: 4,
    }
    r_a = generate_silver_500(pages, bucket_targets=targets, seed=1)
    r_b = generate_silver_500(pages, bucket_targets=targets, seed=2)
    # Both runs should return the same number of rows (target sums match)
    # but with at least some difference in ordering / selection.
    assert len(r_a.queries) == len(r_b.queries)
    a_ids = [q["id"] for q in r_a.queries]
    b_ids = [q["id"] for q in r_b.queries]
    a_doc = [q["expected_doc_ids"][0] for q in r_a.queries]
    b_doc = [q["expected_doc_ids"][0] for q in r_b.queries]
    # ids share the v4-silver-500-NNNN scheme so we compare expected_doc_ids
    assert a_doc != b_doc or a_ids != b_ids


def test_silver_500_jsonl_output_is_byte_stable(tmp_path: Path):
    """write_silver_500_outputs must produce byte-identical jsonl on rerun."""
    pages = _make_corpus(tmp_path, n_each=4)
    targets = {
        BUCKET_MAIN_WORK: 3,
        BUCKET_SUBPAGE_GENERIC: 3,
        BUCKET_SUBPAGE_NAMED: 3,
    }
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    r1 = generate_silver_500(pages, bucket_targets=targets, seed=42)
    r2 = generate_silver_500(pages, bucket_targets=targets, seed=42)
    p1 = write_silver_500_outputs(r1, out_dir=out_a)
    p2 = write_silver_500_outputs(r2, out_dir=out_b)
    assert p1["jsonl"].read_bytes() == p2["jsonl"].read_bytes()
    # summary.json must also be byte-stable; the human-readable .md
    # carries no timestamps so it's also byte-stable.
    assert p1["summary_json"].read_bytes() == p2["summary_json"].read_bytes()
    assert p1["summary_md"].read_bytes() == p2["summary_md"].read_bytes()


# ---------------------------------------------------------------------------
# Bucket distribution
# ---------------------------------------------------------------------------


def test_silver_500_respects_bucket_targets_when_pool_sufficient(
    tmp_path: Path,
):
    pages = _make_corpus(tmp_path, n_each=12)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 10,
            BUCKET_SUBPAGE_GENERIC: 10,
            BUCKET_SUBPAGE_NAMED: 10,
        },
        seed=42,
    )
    summary = result.summary
    actual = summary["bucket_actual_counts"]
    deficits = summary["bucket_deficits"]
    assert actual[BUCKET_MAIN_WORK] == 10
    assert actual[BUCKET_SUBPAGE_GENERIC] == 10
    assert actual[BUCKET_SUBPAGE_NAMED] == 10
    assert all(d == 0 for d in deficits.values())
    assert summary["actual_total"] == 30


def test_silver_500_reports_deficits_when_pool_undersupplies(tmp_path: Path):
    """Best-effort: ask for more than the corpus has and report the gap."""
    pages = _make_corpus(tmp_path, n_each=2)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 50,
            BUCKET_SUBPAGE_GENERIC: 50,
            BUCKET_SUBPAGE_NAMED: 50,
        },
        seed=42,
    )
    summary = result.summary
    deficits = summary["bucket_deficits"]
    # Every bucket should report a positive deficit and the total must
    # be < requested_total.
    assert any(d > 0 for d in deficits.values())
    assert summary["actual_total"] < summary["requested_total"]


def test_silver_500_no_duplicate_page_ids_within_bucket(tmp_path: Path):
    """Same page must not be picked twice (regardless of template kind)."""
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 4,
            BUCKET_SUBPAGE_GENERIC: 4,
            BUCKET_SUBPAGE_NAMED: 4,
        },
        seed=42,
    )
    by_bucket: Dict[str, set] = {b: set() for b in BUCKETS}
    for q in result.queries:
        b = q["v4_meta"]["bucket"]
        pid = q["expected_doc_ids"][0]
        assert pid not in by_bucket[b], (
            f"Page {pid} duplicated within bucket {b}"
        )
        by_bucket[b].add(pid)


# ---------------------------------------------------------------------------
# Template diversity
# ---------------------------------------------------------------------------


def test_silver_500_main_work_emits_multiple_template_kinds(tmp_path: Path):
    """main_work should fan out across more than one template kind."""
    pages = _make_corpus(tmp_path, n_each=8)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 12,
            BUCKET_SUBPAGE_GENERIC: 0,
            BUCKET_SUBPAGE_NAMED: 0,
        },
        seed=42,
    )
    kinds = {
        q["v4_meta"]["template_kind"] for q in result.queries
        if q["v4_meta"]["bucket"] == BUCKET_MAIN_WORK
    }
    # Must include at least the title_lookup + plot_summary + evaluation
    # variants. (alias_lookup / ambiguous_short fire conditionally.)
    assert TEMPLATE_TITLE_LOOKUP in kinds
    assert TEMPLATE_PLOT_SUMMARY in kinds
    assert TEMPLATE_EVALUATION in kinds


def test_silver_500_main_work_short_title_triggers_ambiguous_short(
    tmp_path: Path,
):
    """Short / generic-prone titles must surface the ambiguous_short variant.

    The corpus fixture seeds 'AAA00'..'AAA0N' as 5-char short titles —
    the ambiguous_short template only fires on titles of length ≤ 5.
    """
    pages = _make_corpus(tmp_path, n_each=8)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 24,
            BUCKET_SUBPAGE_GENERIC: 0,
            BUCKET_SUBPAGE_NAMED: 0,
        },
        seed=42,
    )
    kinds = {q["v4_meta"]["template_kind"] for q in result.queries}
    assert TEMPLATE_AMBIGUOUS_SHORT in kinds


def test_silver_500_main_work_alias_lookup_fires_when_alias_present(
    tmp_path: Path,
):
    pages = _make_corpus(tmp_path, n_each=6)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 18,
            BUCKET_SUBPAGE_GENERIC: 0,
            BUCKET_SUBPAGE_NAMED: 0,
        },
        seed=42,
    )
    kinds = {q["v4_meta"]["template_kind"] for q in result.queries}
    # Alias-bearing main pages were seeded with '별칭NN' aliases.
    assert TEMPLATE_ALIAS_LOOKUP in kinds


def test_silver_500_subpage_generic_emits_multiple_template_kinds(
    tmp_path: Path,
):
    pages = _make_corpus(tmp_path, n_each=6)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 0,
            BUCKET_SUBPAGE_GENERIC: 6,
            BUCKET_SUBPAGE_NAMED: 0,
        },
        seed=42,
    )
    kinds = {q["v4_meta"]["template_kind"] for q in result.queries}
    # Three subpage_generic templates exist.
    assert TEMPLATE_SECTION_LOOKUP in kinds
    assert TEMPLATE_SECTION_DETAIL in kinds or TEMPLATE_SECTION_QUESTION in kinds


def test_silver_500_subpage_named_emits_lookup_and_alias_when_alias_present(
    tmp_path: Path,
):
    pages = _make_corpus(tmp_path, n_each=6)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 0,
            BUCKET_SUBPAGE_GENERIC: 0,
            BUCKET_SUBPAGE_NAMED: 18,
        },
        seed=42,
    )
    kinds = {q["v4_meta"]["template_kind"] for q in result.queries}
    assert TEMPLATE_NAMED_LOOKUP in kinds
    # Subpage-named alias variant requires the page to carry an alias
    # distinct from page_title and parent — the fixture supplies
    # '닉네임NN' for those pages.
    assert TEMPLATE_NAMED_ALIAS in kinds


def test_silver_500_section_lookup_query_anchors_on_parent_work(
    tmp_path: Path,
):
    """subpage_generic queries must mention the parent work title."""
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 0,
            BUCKET_SUBPAGE_GENERIC: 4,
            BUCKET_SUBPAGE_NAMED: 0,
        },
        seed=42,
    )
    for q in result.queries:
        meta = q["v4_meta"]
        if meta["bucket"] != BUCKET_SUBPAGE_GENERIC:
            continue
        # retrieval_title is "<work> / <section>"; the work prefix must
        # appear inside the rendered query text.
        rt = meta["retrieval_title"]
        head = rt.split("/")[0].strip()
        assert head in q["query"]


# ---------------------------------------------------------------------------
# Schema preservation
# ---------------------------------------------------------------------------


def test_silver_500_preserves_phase7_silver_schema(tmp_path: Path):
    """Existing silver schema fields must still be present."""
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 3,
            BUCKET_SUBPAGE_GENERIC: 3,
            BUCKET_SUBPAGE_NAMED: 3,
        },
        seed=42,
    )
    required_fields = (
        "id", "query", "language", "expected_doc_ids",
        "expected_section_keywords", "answer_type", "difficulty",
        "tags", "v4_meta",
    )
    for q in result.queries:
        for f in required_fields:
            assert f in q, f"Missing schema field {f!r} in {q['id']}"
        assert q["language"] == "ko"
        assert isinstance(q["expected_doc_ids"], list)
        assert q["expected_doc_ids"]
        assert isinstance(q["expected_section_keywords"], list)


def test_silver_500_summary_schema_carries_distribution_counts(
    tmp_path: Path,
):
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 3,
            BUCKET_SUBPAGE_GENERIC: 3,
            BUCKET_SUBPAGE_NAMED: 3,
        },
        seed=42,
    )
    summary = result.summary
    for key in (
        "is_silver_not_gold", "disclaimer", "audit_disclaimer",
        "bucket_targets", "bucket_actual_counts", "bucket_deficits",
        "candidate_pool_counts", "template_kind_counts",
        "bucket_template_counts", "label_confidence_counts",
        "all_buckets", "all_templates",
    ):
        assert key in summary, f"Missing summary field {key!r}"
    assert summary["is_silver_not_gold"] is True
    assert summary["disclaimer"] == SILVER_DISCLAIMER_LINE
    assert summary["audit_disclaimer"] == SILVER_AUDIT_LINE
    assert set(summary["all_buckets"]) == set(BUCKETS)
    assert set(summary["all_templates"]) == set(ALL_TEMPLATES)


def test_silver_500_default_targets_sum_to_500():
    """Default per-bucket targets sum to 500 — the core spec target."""
    assert sum(DEFAULT_BUCKET_TARGETS.values()) == 500
    assert DEFAULT_BUCKET_TARGETS[BUCKET_MAIN_WORK] == 150
    assert DEFAULT_BUCKET_TARGETS[BUCKET_SUBPAGE_GENERIC] == 200
    assert DEFAULT_BUCKET_TARGETS[BUCKET_SUBPAGE_NAMED] == 150


def test_silver_500_writers_produce_three_files(tmp_path: Path):
    pages = _make_corpus(tmp_path, n_each=4)
    result = generate_silver_500(
        pages,
        bucket_targets={
            BUCKET_MAIN_WORK: 3,
            BUCKET_SUBPAGE_GENERIC: 3,
            BUCKET_SUBPAGE_NAMED: 3,
        },
        seed=42,
    )
    paths = write_silver_500_outputs(result, out_dir=tmp_path / "out")
    assert paths["jsonl"].exists()
    assert paths["summary_json"].exists()
    assert paths["summary_md"].exists()
    # Filenames are exactly what the spec requires.
    assert paths["jsonl"].name == "queries_v4_silver_500.jsonl"
    assert paths["summary_json"].name == "queries_v4_silver_500.summary.json"
    assert paths["summary_md"].name == "queries_v4_silver_500.summary.md"
    # JSONL must be one record per line.
    text = paths["jsonl"].read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    parsed = [json.loads(l) for l in lines]
    assert len(parsed) == len(result.queries)
