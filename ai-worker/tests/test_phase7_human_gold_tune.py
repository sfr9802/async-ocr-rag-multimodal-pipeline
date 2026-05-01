"""Phase 7.x — tests for the human-weighted gold-50 + silver-500 tuning.

Targets the harness module ``eval.harness.phase7_human_gold_tune`` and
the CLI ``scripts.phase7_human_gold_tune``. The test bar is:

  - loaders raise / collect issues for every schema problem the spec
    calls out (NaN weight, blank query, dup id, NOT_IN_CORPUS row with
    expected page id, etc.).
  - eval_use → normalized_eval_group is a pure mapping pinned by the
    test set so a future contributor can't silently shift the
    boundaries between STRICT / SOFT / AMBIGUOUS_PROBE / ABSTAIN_TEST.
  - weighted metrics use ``eval_weight`` correctly: weight 0 rows are
    excluded; weight 0.4 contributes 0.4 of a hit point.
  - ABSTAIN_TEST + AMBIGUOUS_PROBE rows are excluded from the primary
    objective.
  - Silver guardrail fires the right warning code at the right
    threshold.
  - Failure audit rows render with the right heuristic.
  - The CLI replay path produces the full output bundle.

All tests are pure-Python: no FAISS, no embedder. Live retrieval is
covered indirectly by the replay-path tests, which feed synthetic
RetrievalResult JSONL into the same scoring pipeline the live mode
would.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from eval.harness.phase7_human_gold_tune import (
    FAIL_NAMED_SUBPAGE_MISS,
    FAIL_NOT_IN_CORPUS_CASE,
    FAIL_OVER_BROAD_QUERY,
    FAIL_SECTION_MISS,
    FAIL_SUBPAGE_MISS,
    FAIL_TITLE_MISS,
    FAIL_UNKNOWN,
    FAIL_WRONG_SEASON,
    FAIL_WRONG_SERIES,
    GOLD_REQUIRED_COLUMNS,
    GROUP_ABSTAIN_TEST,
    GROUP_AMBIGUOUS_PROBE,
    GROUP_SOFT_POSITIVE,
    GROUP_STRICT_POSITIVE,
    HUMAN_FOCUS_DISCLAIMER,
    PRIMARY_WEIGHTED_HIT_AT_5,
    PRIMARY_WEIGHTED_MRR_AT_10,
    PRIMARY_WEIGHTED_NDCG_AT_10,
    PROMOTION_TARGET_FRAMING,
    SILVER_BUCKET_FOR_NAMED_GUARDRAIL,
    SILVER_BUCKET_REGRESSION_THRESHOLD,
    SILVER_HIT_AT_5_REGRESSION_THRESHOLD,
    GoldRow,
    GoldSeedValidationError,
    GoldQueryEvalRow,
    GoldSeedDataset,
    RetrievedDoc,
    SilverDataset,
    SilverRow,
    build_failure_audit_row,
    classify_failure,
    compare_variants,
    comparison_to_dict,
    evaluate_gold,
    evaluate_silver,
    evaluate_silver_guardrail,
    hit_at_k,
    load_human_gold_seed_50,
    load_llm_silver_500,
    mrr_at_k,
    ndcg_at_k,
    normalize_eval_group,
    primary_score,
    render_comparison_report,
    render_failure_audit_md,
    section_hit_at_k,
    silver_summary_to_dict,
    summarize_gold,
    summarize_silver,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic gold / silver row + retrieval factories
# ---------------------------------------------------------------------------


def _gold_row(
    *,
    qid: str,
    query: str = "q",
    expected_pid: str = "P-A",
    expected_title: str = "Title A",
    expected_section_path=("개요",),
    bucket: str = "main_work",
    query_type: str = "direct_title",
    weight: float = 1.0,
    eval_use: str = "SUPPORTED",
    human_label: str = "SUPPORTED",
    expected_not_in_corpus: bool = False,
    leakage_risk: str = "low",
    human_chunk_id: str = "",
) -> GoldRow:
    group = normalize_eval_group(
        eval_use_raw=eval_use,
        eval_weight=weight,
        expected_not_in_corpus=expected_not_in_corpus,
        human_label=human_label,
    )
    return GoldRow(
        query_id=qid,
        query=query,
        query_type=query_type,
        bucket=bucket,
        silver_expected_title=expected_title,
        silver_expected_page_id=expected_pid,
        expected_section_path=tuple(expected_section_path),
        expected_not_in_corpus=expected_not_in_corpus,
        human_label=human_label,
        human_correct_title="",
        human_correct_page_id="",
        human_supporting_chunk_id=human_chunk_id,
        human_notes="",
        eval_use_raw=eval_use,
        eval_weight=weight,
        leakage_risk=leakage_risk,
        expected_title=expected_title,
        expected_page_id=expected_pid,
        normalized_eval_group=group,
    )


def _silver_row(
    *,
    qid: str,
    expected_pid: str = "P-A",
    bucket: str = "main_work",
    query_type: str = "direct_title",
    expected_not_in_corpus: bool = False,
    leakage_risk: str = "low",
    overlap_risk: str = "low",
) -> SilverRow:
    return SilverRow(
        query_id=qid,
        query=f"silver query {qid}",
        query_type=query_type,
        bucket=bucket,
        silver_expected_title="t",
        silver_expected_page_id=expected_pid,
        expected_section_path=("개요",),
        expected_not_in_corpus=expected_not_in_corpus,
        leakage_risk=leakage_risk,
        overlap_risk=overlap_risk,
    )


def _docs(*entries) -> List[RetrievedDoc]:
    """Build a list of RetrievedDocs from (page_id, title, section[, chunk_id, score]).

    Sections may be a string (single segment) or a tuple. Chunk IDs
    default to ``"{page_id}-c{rank}"``; scores default to ``1.0 / rank``.
    """
    out: List[RetrievedDoc] = []
    for rank, e in enumerate(entries, start=1):
        if len(e) == 2:
            page_id, title = e
            section = ("개요",)
            chunk_id = f"{page_id}-c{rank}"
            score = 1.0 / rank
        elif len(e) == 3:
            page_id, title, section = e
            chunk_id = f"{page_id}-c{rank}"
            score = 1.0 / rank
        elif len(e) == 4:
            page_id, title, section, chunk_id = e
            score = 1.0 / rank
        else:
            page_id, title, section, chunk_id, score = e
        if isinstance(section, str):
            section = (section,)
        out.append(RetrievedDoc(
            rank=rank, chunk_id=chunk_id, page_id=page_id, title=title,
            section_path=tuple(section), score=score,
        ))
    return out


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> Path:
    fieldnames = list(GOLD_REQUIRED_COLUMNS)
    extras = sorted({k for r in rows for k in r.keys()} - set(fieldnames))
    fieldnames = fieldnames + extras
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def _make_full_gold_row(**overrides: Any) -> Dict[str, str]:
    base: Dict[str, str] = {
        "query_id": "q1", "query": "어떤 작품?",
        "query_type": "direct_title", "bucket": "main_work",
        "silver_expected_title": "T", "silver_expected_page_id": "P-A",
        "expected_section_path": "[\"개요\"]",
        "expected_not_in_corpus": "FALSE",
        "human_label": "SUPPORTED",
        "human_correct_title": "", "human_correct_page_id": "",
        "human_supporting_chunk_id": "", "human_notes": "",
        "eval_use": "SUPPORTED", "eval_weight": "1",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return base


def test_load_gold_seed_csv_happy_path(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", eval_weight="1"),
        _make_full_gold_row(
            query_id="q2", eval_weight="0.4",
            human_label="PARTIALLY_SUPPORTED", eval_use="PARTIALLY_SUPPORTED",
        ),
        _make_full_gold_row(
            query_id="q3", eval_weight="0",
            human_label="AMBIGUOUS_QUERY", eval_use="AMBIGUOUS_QUERY",
        ),
        _make_full_gold_row(
            query_id="q4", eval_weight="0",
            human_label="NOT_IN_CORPUS", eval_use="NOT_IN_CORPUS",
            expected_not_in_corpus="TRUE",
            silver_expected_page_id="",
        ),
    ])
    ds = load_human_gold_seed_50(p)
    assert len(ds.rows) == 4
    by_id = {r.query_id: r for r in ds.rows}
    assert by_id["q1"].normalized_eval_group == GROUP_STRICT_POSITIVE
    assert by_id["q2"].normalized_eval_group == GROUP_SOFT_POSITIVE
    assert by_id["q3"].normalized_eval_group == GROUP_AMBIGUOUS_PROBE
    assert by_id["q4"].normalized_eval_group == GROUP_ABSTAIN_TEST
    # No errors expected on the happy path.
    assert not [i for i in ds.issues if i.severity == "error"]


def test_load_gold_seed_csv_blank_eval_weight_raises(tmp_path: Path) -> None:
    """Blank eval_weight on a positive row is a per-row error.

    The loader collects per-row errors and only raises when *every*
    row failed; here only one of two rows has the blank weight, so
    we expect one error issue plus one valid row.
    """
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", eval_weight=""),
        _make_full_gold_row(query_id="q2", eval_weight="1"),
    ])
    ds = load_human_gold_seed_50(p)
    errors = [i for i in ds.issues if i.severity == "error"]
    assert len(errors) == 1
    assert errors[0].query_id == "q1"
    assert errors[0].field_name == "eval_weight"
    # The valid row should still be loaded.
    assert {r.query_id for r in ds.rows} == {"q2"}


def test_load_gold_seed_csv_all_blank_weight_raises(tmp_path: Path) -> None:
    """When every row's eval_weight is blank, the loader bails entirely."""
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", eval_weight=""),
        _make_full_gold_row(query_id="q2", eval_weight=""),
    ])
    with pytest.raises(GoldSeedValidationError):
        load_human_gold_seed_50(p)


def test_load_gold_seed_csv_nan_weight_collected(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", eval_weight="nope"),
        _make_full_gold_row(query_id="q2", eval_weight="0.5"),
    ])
    ds = load_human_gold_seed_50(p)
    errors = [i for i in ds.issues if i.severity == "error"]
    assert any(i.query_id == "q1" and "not numeric" in i.message for i in errors)


def test_load_gold_seed_csv_duplicate_id(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1"),
        _make_full_gold_row(query_id="q1"),
    ])
    ds = load_human_gold_seed_50(p)
    errors = [i for i in ds.issues if i.severity == "error"]
    assert any(i.field_name == "query_id" and "duplicate" in i.message for i in errors)
    assert len(ds.rows) == 1


def test_load_gold_seed_csv_blank_query_dropped(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", query=""),
        _make_full_gold_row(query_id="q2"),
    ])
    ds = load_human_gold_seed_50(p)
    errors = [i for i in ds.issues if i.severity == "error"]
    assert any(i.query_id == "q1" and i.field_name == "query" for i in errors)
    assert len(ds.rows) == 1


def test_load_gold_seed_csv_not_in_corpus_with_pid_warning(tmp_path: Path) -> None:
    """A NOT_IN_CORPUS row that nonetheless carries a page_id is a soft warning."""
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(
            query_id="q1", eval_weight="0",
            human_label="NOT_IN_CORPUS", eval_use="NOT_IN_CORPUS",
            expected_not_in_corpus="TRUE",
            silver_expected_page_id="P-A",
        ),
    ])
    ds = load_human_gold_seed_50(p)
    warnings = [i for i in ds.issues if i.severity == "warning"]
    assert any("ABSTAIN_TEST" in i.message for i in warnings)
    assert ds.rows[0].normalized_eval_group == GROUP_ABSTAIN_TEST


def test_load_gold_seed_csv_positive_without_pid_warning(tmp_path: Path) -> None:
    """Positive row with no page id at all is kept but flagged."""
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(
            query_id="q1", eval_weight="1",
            silver_expected_page_id="", human_correct_page_id="",
        ),
    ])
    ds = load_human_gold_seed_50(p)
    warnings = [i for i in ds.issues if i.severity == "warning"]
    assert any(
        i.field_name == "expected_page_id" and "no expected page id" in i.message
        for i in warnings
    )
    assert ds.rows[0].normalized_eval_group == GROUP_STRICT_POSITIVE
    assert ds.rows[0].expected_page_id == ""


def test_load_gold_seed_csv_human_overrides_silver(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(
            query_id="q1",
            silver_expected_title="silver_t", silver_expected_page_id="silver_p",
            human_correct_title="human_t", human_correct_page_id="human_p",
        ),
    ])
    ds = load_human_gold_seed_50(p)
    assert ds.rows[0].expected_title == "human_t"
    assert ds.rows[0].expected_page_id == "human_p"


def test_load_gold_seed_csv_missing_required_column(tmp_path: Path) -> None:
    """Header missing a required column → hard error before any row is read."""
    p = tmp_path / "gold.csv"
    fieldnames = [c for c in GOLD_REQUIRED_COLUMNS if c != "eval_weight"]
    with p.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: "x" for k in fieldnames})
    with pytest.raises(GoldSeedValidationError):
        load_human_gold_seed_50(p)


def test_load_real_gold_focus_50_csv_passes() -> None:
    """The real focus_50 CSV in the repo loads without hard errors.

    Pinned because the file is small enough to live next to the
    harness — if a contributor breaks the schema the test catches it
    immediately.
    """
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (
        repo_root
        / "eval/reports/phase7/seeds/llm_silver_focus_50/"
        "phase7_human_gold_seed_50.csv"
    )
    if not csv_path.exists():
        pytest.skip(f"focus_50 CSV not present: {csv_path}")
    ds = load_human_gold_seed_50(csv_path)
    assert len(ds.rows) == 50
    groups = {r.normalized_eval_group for r in ds.rows}
    # Every group must be represented; otherwise the harness can't
    # exercise its full surface area on the real data.
    assert GROUP_STRICT_POSITIVE in groups
    assert GROUP_ABSTAIN_TEST in groups


# ---------------------------------------------------------------------------
# Silver loader tests
# ---------------------------------------------------------------------------


def _write_silver_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_load_llm_silver_500_jsonl_happy(tmp_path: Path) -> None:
    p = _write_silver_jsonl(tmp_path / "s.jsonl", [
        {
            "query_id": "s1", "query": "q1", "query_type": "direct_title",
            "bucket": "main_work", "silver_expected_title": "T",
            "silver_expected_page_id": "P-A",
            "expected_section_path": ["개요"],
            "expected_not_in_corpus": False,
            "leakage_risk": "low",
            "lexical_overlap": {"overlap_risk": "low"},
            "tags": ["anime"],
        },
        {
            "query_id": "s2", "query": "q2", "query_type": "section_intent",
            "bucket": "subpage_named", "silver_expected_title": "T2",
            "silver_expected_page_id": "P-B",
            "expected_section_path": ["등장인물", "주역"],
            "expected_not_in_corpus": False,
            "leakage_risk": "high",
        },
    ])
    ds = load_llm_silver_500(p)
    assert len(ds.rows) == 2
    by_id = {r.query_id: r for r in ds.rows}
    assert by_id["s1"].overlap_risk == "low"
    assert by_id["s2"].expected_section_path == ("등장인물", "주역")


def test_load_silver_jsonl_collects_dup_id_and_blank_query(tmp_path: Path) -> None:
    p = _write_silver_jsonl(tmp_path / "s.jsonl", [
        {"query_id": "s1", "query": "q", "silver_expected_page_id": "P-A"},
        {"query_id": "s1", "query": "dup", "silver_expected_page_id": "P-A"},
        {"query_id": "s2", "query": "", "silver_expected_page_id": "P-B"},
        {"query_id": "s3", "query": "ok", "silver_expected_page_id": ""},
    ])
    ds = load_llm_silver_500(p)
    assert {r.query_id for r in ds.rows} == {"s1", "s3"}
    issues_by_severity = {i.severity for i in ds.issues}
    # dup id + blank query are errors; missing-pid on answerable is a warning.
    assert "error" in issues_by_severity
    assert "warning" in issues_by_severity


def test_load_real_silver_500_jsonl_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    silver_path = (
        repo_root
        / "eval/reports/phase7/seeds/llm_silver_500/"
        "queries_v4_llm_silver_500.jsonl"
    )
    if not silver_path.exists():
        pytest.skip(f"silver_500 jsonl not present: {silver_path}")
    ds = load_llm_silver_500(silver_path)
    assert len(ds.rows) == 500


# ---------------------------------------------------------------------------
# eval_use → normalized_eval_group mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eval_use,weight,not_in_corpus,expected", [
    ("SUPPORTED", 1.0, False, GROUP_STRICT_POSITIVE),
    ("SUPPORTED", 0.8, False, GROUP_STRICT_POSITIVE),
    ("SUPPORTED", 0.7, False, GROUP_SOFT_POSITIVE),
    ("PARTIALLY_SUPPORTED", 0.5, False, GROUP_SOFT_POSITIVE),
    ("PARTIALLY_SUPPORTED | AMBIGUOUS_QUERY", 0.3, False, GROUP_SOFT_POSITIVE),
    ("AMBIGUOUS_QUERY", 0.0, False, GROUP_AMBIGUOUS_PROBE),
    ("AMBIGUOUS_QUERY", 0.2, False, GROUP_AMBIGUOUS_PROBE),
    ("NOT_IN_CORPUS", 0.0, True, GROUP_ABSTAIN_TEST),
    ("NOT_IN_CORPUS", 0.0, False, GROUP_ABSTAIN_TEST),  # token alone wins
    ("", 0.0, True, GROUP_ABSTAIN_TEST),
    ("", 1.0, False, GROUP_AMBIGUOUS_PROBE),  # no positive token → probe
    ("SUPPORTED", 0.0, False, GROUP_AMBIGUOUS_PROBE),  # weight wins over label
])
def test_normalize_eval_group_mapping(
    eval_use: str, weight: float, not_in_corpus: bool, expected: str,
) -> None:
    assert normalize_eval_group(
        eval_use_raw=eval_use,
        eval_weight=weight,
        expected_not_in_corpus=not_in_corpus,
    ) == expected


def test_normalize_eval_group_uses_human_label_token() -> None:
    """human_label is consulted when eval_use is empty.

    Pinned because real CSVs sometimes leave eval_use blank and only
    fill in human_label.
    """
    assert normalize_eval_group(
        eval_use_raw="", eval_weight=1.0,
        expected_not_in_corpus=False, human_label="SUPPORTED",
    ) == GROUP_STRICT_POSITIVE
    assert normalize_eval_group(
        eval_use_raw="", eval_weight=0.0,
        expected_not_in_corpus=False, human_label="NOT_IN_CORPUS",
    ) == GROUP_ABSTAIN_TEST


# ---------------------------------------------------------------------------
# Metric primitive tests
# ---------------------------------------------------------------------------


def test_hit_at_k_basic() -> None:
    docs = _docs(("P-X", "X"), ("P-A", "A"), ("P-Y", "Y"))
    assert hit_at_k(docs, "P-A", k=1) == 0
    assert hit_at_k(docs, "P-A", k=2) == 1
    assert hit_at_k(docs, "P-A", k=10) == 1
    assert hit_at_k(docs, "P-Z", k=10) == 0
    assert hit_at_k(docs, "", k=10) == 0   # no expected → not a hit


def test_mrr_at_k_returns_reciprocal() -> None:
    docs = _docs(("P-X", "X"), ("P-A", "A"))
    assert mrr_at_k(docs, "P-A", k=10) == pytest.approx(0.5)
    assert mrr_at_k(docs, "P-X", k=10) == pytest.approx(1.0)
    assert mrr_at_k(docs, "P-Z", k=10) == 0.0


def test_ndcg_at_k_matches_log2_form() -> None:
    docs = _docs(("P-X", "X"), ("P-A", "A"))
    # rank 2 → 1 / log2(3) ≈ 0.6309
    assert ndcg_at_k(docs, "P-A", k=10) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(docs, "P-X", k=10) == pytest.approx(1.0)


def test_section_hit_at_k_requires_page_match_first() -> None:
    docs = _docs(
        ("P-A", "A", ("개요",)),
        ("P-B", "B", ("등장인물", "주인공")),
    )
    # expected page is P-A but section path is on P-B → 0.
    assert section_hit_at_k(
        docs, "P-A", expected_section_path=("등장인물",), k=10,
    ) == 0
    # expected page is P-B and section prefix matches → 1.
    assert section_hit_at_k(
        docs, "P-B", expected_section_path=("등장인물",), k=10,
    ) == 1
    # No expected section → metric undefined → None.
    assert section_hit_at_k(
        docs, "P-A", expected_section_path=(), k=10,
    ) is None


# ---------------------------------------------------------------------------
# Aggregation tests — primary_score, weighting, group exclusion
# ---------------------------------------------------------------------------


def _ds_from_rows(rows: List[GoldRow]) -> GoldSeedDataset:
    return GoldSeedDataset(rows=list(rows), issues=[])


def test_primary_score_formula_pinned() -> None:
    s = primary_score(
        weighted_hit_at_5=1.0,
        weighted_mrr_at_10=1.0,
        weighted_ndcg_at_10=1.0,
    )
    assert s == pytest.approx(
        PRIMARY_WEIGHTED_HIT_AT_5
        + PRIMARY_WEIGHTED_MRR_AT_10
        + PRIMARY_WEIGHTED_NDCG_AT_10,
    )
    # Sanity: weights sum to 1.0.
    assert (
        PRIMARY_WEIGHTED_HIT_AT_5
        + PRIMARY_WEIGHTED_MRR_AT_10
        + PRIMARY_WEIGHTED_NDCG_AT_10
    ) == pytest.approx(1.0)


def test_weighted_metrics_use_eval_weight() -> None:
    """Two strict rows (weight 1) hit, one soft row (weight 0.4) misses.

    Unweighted hit@5 = 2/3.
    Weighted hit@5 = (1*1 + 1*1 + 0.4*0) / (1+1+0.4) = 2 / 2.4 ≈ 0.833.
    """
    rows = [
        _gold_row(qid="q1", expected_pid="P-A", weight=1.0),
        _gold_row(qid="q2", expected_pid="P-B", weight=1.0),
        _gold_row(
            qid="q3", expected_pid="P-C", weight=0.4,
            eval_use="PARTIALLY_SUPPORTED", human_label="PARTIALLY_SUPPORTED",
        ),
    ]
    ds = _ds_from_rows(rows)
    retrievals = {
        "q1": _docs(("P-A", "A")),
        "q2": _docs(("P-B", "B")),
        "q3": _docs(("P-X", "X")),
    }
    eval_rows = evaluate_gold(ds, retrievals)
    summary = summarize_gold(eval_rows)
    assert summary.hit_at_5 == pytest.approx(2.0 / 3.0)
    assert summary.weighted_hit_at_5 == pytest.approx(2.0 / 2.4)


def test_ambiguous_probe_excluded_from_primary() -> None:
    """AMBIGUOUS_PROBE rows must NOT contribute to primary_score."""
    rows = [
        _gold_row(qid="q1", expected_pid="P-A", weight=1.0),
        _gold_row(
            qid="q2", expected_pid="P-B", weight=0.0,
            eval_use="AMBIGUOUS_QUERY", human_label="AMBIGUOUS_QUERY",
        ),
    ]
    ds = _ds_from_rows(rows)
    retrievals = {
        "q1": _docs(("P-A", "A")),
        "q2": _docs(("P-B", "B")),  # would be a hit if counted
    }
    eval_rows = evaluate_gold(ds, retrievals)
    summary = summarize_gold(eval_rows)
    # Only q1 is positive; primary_score reflects only it.
    assert summary.n_strict_positive == 1
    assert summary.n_ambiguous_probe == 1
    assert summary.weighted_hit_at_5 == pytest.approx(1.0)
    assert summary.primary_score == pytest.approx(1.0)


def test_abstain_test_excluded_from_primary() -> None:
    """NOT_IN_CORPUS rows must NOT contribute to hit/MRR aggregates."""
    rows = [
        _gold_row(qid="q1", expected_pid="P-A", weight=1.0),
        _gold_row(
            qid="q2", expected_pid="", weight=0.0,
            eval_use="NOT_IN_CORPUS", human_label="NOT_IN_CORPUS",
            expected_not_in_corpus=True,
        ),
    ]
    ds = _ds_from_rows(rows)
    retrievals = {
        "q1": _docs(("P-X", "X")),  # miss
        "q2": _docs(("P-Z", "Z")),  # would be a confident wrong-answer
    }
    eval_rows = evaluate_gold(ds, retrievals)
    summary = summarize_gold(eval_rows)
    assert summary.n_abstain_test == 1
    # q1 missed → primary at 0; q2 must not bias the average.
    assert summary.weighted_hit_at_5 == 0.0
    assert summary.primary_score == 0.0


def test_strict_only_metric_separates_strict_from_soft() -> None:
    """strict_hit@5 only sums STRICT rows; soft don't drag it down."""
    rows = [
        _gold_row(qid="q1", expected_pid="P-A", weight=1.0),  # strict, hit
        _gold_row(
            qid="q2", expected_pid="P-B", weight=0.4,
            eval_use="PARTIALLY_SUPPORTED",
        ),  # soft, miss
    ]
    ds = _ds_from_rows(rows)
    retrievals = {
        "q1": _docs(("P-A", "A")),
        "q2": _docs(("P-X", "X")),
    }
    eval_rows = evaluate_gold(ds, retrievals)
    summary = summarize_gold(eval_rows)
    assert summary.strict_hit_at_5 == pytest.approx(1.0)
    # Mean over both positives is 0.5.
    assert summary.hit_at_5 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Silver guardrail tests
# ---------------------------------------------------------------------------


def _silver_summary_with_hit5(
    *, hit_at_5: float, named_hit_at_5: float | None = None,
):
    rows = []
    if hit_at_5 < 1.0:
        rows.append(_silver_row(qid="s_miss", expected_pid="P-X", bucket="main_work"))
    if hit_at_5 > 0.0:
        rows.append(_silver_row(qid="s_hit", expected_pid="P-A", bucket="main_work"))
    if named_hit_at_5 is not None:
        rows.append(_silver_row(qid="s_named", expected_pid="P-N", bucket="subpage_named"))
    ds = SilverDataset(rows=rows, issues=[])

    retrievals: Dict[str, List[RetrievedDoc]] = {}
    if hit_at_5 < 1.0:
        retrievals["s_miss"] = _docs(("P-Z", "Z"))
    if hit_at_5 > 0.0:
        retrievals["s_hit"] = _docs(("P-A", "A"))
    if named_hit_at_5 is not None:
        if named_hit_at_5 >= 1.0:
            retrievals["s_named"] = _docs(("P-N", "N"))
        else:
            retrievals["s_named"] = _docs(("P-Z", "Z"))

    eval_rows = evaluate_silver(ds, retrievals)
    return summarize_silver(eval_rows)


def test_silver_guardrail_fires_general_regression() -> None:
    base = _silver_summary_with_hit5(hit_at_5=1.0)
    cand = _silver_summary_with_hit5(hit_at_5=0.0)
    warns = evaluate_silver_guardrail(baseline=base, candidate=cand)
    codes = [w.code for w in warns]
    assert "SILVER_REGRESSION_WARNING" in codes
    # Threshold check: the warning's threshold must equal the configured
    # constant so a future contributor can't silently lower the bar.
    sw = next(w for w in warns if w.code == "SILVER_REGRESSION_WARNING")
    assert sw.threshold == pytest.approx(SILVER_HIT_AT_5_REGRESSION_THRESHOLD)


def test_silver_guardrail_does_not_fire_below_threshold() -> None:
    """A 1% drop on silver hit@5 should NOT fire the warning.

    Threshold is 3pp; we sit at 1pp.
    """
    base = _silver_summary_with_hit5(hit_at_5=1.0)
    cand_rows = [
        _silver_row(qid=f"s{i}", expected_pid="P-A") for i in range(99)
    ] + [_silver_row(qid="s99", expected_pid="P-X")]
    ds = SilverDataset(rows=cand_rows, issues=[])
    retrievals: Dict[str, List[RetrievedDoc]] = {}
    for i in range(99):
        retrievals[f"s{i}"] = _docs(("P-A", "A"))
    retrievals["s99"] = _docs(("P-Z", "Z"))
    cand = summarize_silver(evaluate_silver(ds, retrievals))
    warns = evaluate_silver_guardrail(baseline=base, candidate=cand)
    assert not any(w.code == "SILVER_REGRESSION_WARNING" for w in warns)


def test_silver_guardrail_fires_bucket_regression() -> None:
    """Named-subpage bucket dropping >= 5pp is its own warning code."""
    base = _silver_summary_with_hit5(hit_at_5=1.0, named_hit_at_5=1.0)
    cand = _silver_summary_with_hit5(hit_at_5=1.0, named_hit_at_5=0.0)
    warns = evaluate_silver_guardrail(baseline=base, candidate=cand)
    codes = [w.code for w in warns]
    assert "BUCKET_REGRESSION_WARNING" in codes
    bw = next(w for w in warns if w.code == "BUCKET_REGRESSION_WARNING")
    assert bw.bucket == SILVER_BUCKET_FOR_NAMED_GUARDRAIL
    assert bw.threshold == pytest.approx(SILVER_BUCKET_REGRESSION_THRESHOLD)


# ---------------------------------------------------------------------------
# Failure audit tests
# ---------------------------------------------------------------------------


def test_classify_failure_named_subpage_miss() -> None:
    eval_row = GoldQueryEvalRow(
        query_id="q", query="q", bucket="subpage_named",
        query_type="section_intent",
        normalized_eval_group=GROUP_STRICT_POSITIVE,
        eval_weight=1.0,
        expected_title="T", expected_page_id="P-A",
        expected_section_path=("등장인물",),
        leakage_risk="low",
        hit_at_10=0,
        docs=tuple(_docs(("P-Z", "Z"))),
    )
    assert classify_failure(eval_row=eval_row) == FAIL_NAMED_SUBPAGE_MISS


def test_classify_failure_subpage_generic() -> None:
    eval_row = GoldQueryEvalRow(
        query_id="q", query="q", bucket="subpage_generic",
        query_type="section_intent",
        normalized_eval_group=GROUP_STRICT_POSITIVE,
        eval_weight=1.0,
        expected_title="T", expected_page_id="P-A",
        expected_section_path=("등장인물",),
        leakage_risk="low",
        hit_at_10=0,
        docs=tuple(_docs(("P-Z", "Z"))),
    )
    assert classify_failure(eval_row=eval_row) == FAIL_SUBPAGE_MISS


def test_classify_failure_section_miss_when_page_hits() -> None:
    """top-1 page hits but section path mismatched → SECTION_MISS."""
    eval_row = GoldQueryEvalRow(
        query_id="q", query="q", bucket="subpage_generic",
        query_type="section_intent",
        normalized_eval_group=GROUP_STRICT_POSITIVE,
        eval_weight=1.0,
        expected_title="T", expected_page_id="P-A",
        expected_section_path=("등장인물",),
        leakage_risk="low",
        hit_at_10=0,
        docs=tuple(_docs(("P-A", "A", ("개요",)))),
    )
    assert classify_failure(eval_row=eval_row) == FAIL_SECTION_MISS


def test_classify_failure_wrong_season() -> None:
    eval_row = GoldQueryEvalRow(
        query_id="q", query="q", bucket="main_work",
        query_type="direct_title",
        normalized_eval_group=GROUP_STRICT_POSITIVE,
        eval_weight=1.0,
        expected_title="주술회전(애니메이션 1기)",
        expected_page_id="P-A",
        expected_section_path=(),
        leakage_risk="low",
        hit_at_10=0,
        docs=tuple(_docs(("P-B", "주술회전(애니메이션 1기)"))),
    )
    # Same title (with "기"), different page → WRONG_SEASON.
    assert classify_failure(eval_row=eval_row) == FAIL_WRONG_SEASON


def test_classify_failure_not_in_corpus() -> None:
    eval_row = GoldQueryEvalRow(
        query_id="q", query="q", bucket="not_in_corpus",
        query_type="unanswerable_or_not_in_corpus",
        normalized_eval_group=GROUP_ABSTAIN_TEST,
        eval_weight=0.0,
        expected_title="", expected_page_id="",
        expected_section_path=(),
        leakage_risk="not_applicable",
        docs=tuple(_docs(("P-X", "X"))),
    )
    assert classify_failure(eval_row=eval_row) == FAIL_NOT_IN_CORPUS_CASE


def test_failure_audit_row_renders_top_k_fields() -> None:
    eval_row = GoldQueryEvalRow(
        query_id="q1", query="q text", bucket="main_work",
        query_type="direct_title",
        normalized_eval_group=GROUP_STRICT_POSITIVE,
        eval_weight=1.0,
        expected_title="T", expected_page_id="P-A",
        expected_section_path=(),
        leakage_risk="low",
        hit_at_10=0,
        docs=tuple(_docs(
            ("P-Z", "Z", ("개요",)),
            ("P-Y", "Y", ("기타",)),
        )),
    )
    audit = build_failure_audit_row(eval_row)
    assert audit.query_id == "q1"
    assert audit.top1_page_id == "P-Z"
    assert audit.top1_title == "Z"
    assert audit.top_k_page_ids == ("P-Z", "P-Y")
    assert audit.top_k_section_paths == (("개요",), ("기타",))
    # Failure reason populated for misses.
    assert audit.failure_reason in {FAIL_TITLE_MISS, FAIL_OVER_BROAD_QUERY}


def test_failure_audit_md_carries_disclaimer_and_bucket() -> None:
    rows = [
        build_failure_audit_row(GoldQueryEvalRow(
            query_id="q1", query="q", bucket="main_work",
            query_type="direct_title",
            normalized_eval_group=GROUP_STRICT_POSITIVE,
            eval_weight=1.0, expected_title="T", expected_page_id="P-A",
            expected_section_path=(), leakage_risk="low",
            docs=tuple(_docs(("P-Z", "Z"))),
        ))
    ]
    md = render_failure_audit_md(rows, header="t")
    assert HUMAN_FOCUS_DISCLAIMER in md
    # bucket + query_type now appear in the table; pinning here so
    # future contributors don't drop them silently.
    assert "main_work" in md
    assert "direct_title" in md
    # The structured row carries the same fields.
    assert rows[0].bucket == "main_work"
    assert rows[0].query_type == "direct_title"


# ---------------------------------------------------------------------------
# Variant comparison tests
# ---------------------------------------------------------------------------


def _build_variant_result(
    *, variant: str,
    gold_rows: List[GoldRow], retrievals: Dict[str, List[RetrievedDoc]],
    silver_rows: List[SilverRow], silver_retrievals: Dict[str, List[RetrievedDoc]],
):
    from eval.harness.phase7_human_gold_tune import (
        VariantResult, build_failure_audit_row,
    )
    gold_eval = evaluate_gold(GoldSeedDataset(rows=gold_rows), retrievals)
    silver_eval = evaluate_silver(SilverDataset(rows=silver_rows), silver_retrievals)
    return VariantResult(
        variant=variant,
        gold_summary=summarize_gold(gold_eval),
        silver_summary=summarize_silver(silver_eval),
        gold_per_query=gold_eval,
        silver_per_query=silver_eval,
        failure_audit=[build_failure_audit_row(r) for r in gold_eval],
        config={"name": variant},
    )


def test_compare_variants_picks_best_when_clean_win() -> None:
    gold_rows = [
        _gold_row(qid="q1", expected_pid="P-A"),
        _gold_row(qid="q2", expected_pid="P-B"),
    ]
    silver_rows = [_silver_row(qid="s1", expected_pid="P-A")]
    base = _build_variant_result(
        variant="baseline",
        gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-X", "X")), "q2": _docs(("P-Y", "Y"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-A", "A"))},
    )
    cand = _build_variant_result(
        variant="cand",
        gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-A", "A")), "q2": _docs(("P-B", "B"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-A", "A"))},
    )
    comp = compare_variants(baseline=base, candidates=[cand])
    assert comp.best_variant == "cand"


def test_compare_variants_rejects_silver_regression() -> None:
    """Candidate that gains on gold but loses 4pp on silver hit@5 → rejected
    when strict_hit_at_5 didn't improve."""
    gold_rows = [
        _gold_row(qid="q1", expected_pid="P-A", weight=0.4,
                  eval_use="PARTIALLY_SUPPORTED")  # soft only
        for _ in range(1)
    ]
    silver_rows = [_silver_row(qid=f"s{i}", expected_pid="P-A") for i in range(100)]
    base = _build_variant_result(
        variant="baseline",
        gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-X", "X"))},
        silver_rows=silver_rows,
        silver_retrievals={f"s{i}": _docs(("P-A", "A")) for i in range(100)},
    )
    # Candidate: gold soft hits q1, but silver hit@5 drops 100% → 0%.
    cand_silver_retr = {f"s{i}": _docs(("P-Z", "Z")) for i in range(100)}
    cand = _build_variant_result(
        variant="cand",
        gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-A", "A"))},
        silver_rows=silver_rows,
        silver_retrievals=cand_silver_retr,
    )
    comp = compare_variants(baseline=base, candidates=[cand])
    # Baseline retained because silver crashed.
    assert comp.best_variant == "baseline"
    assert "no candidate cleared" in comp.best_reason or "guardrail" in comp.best_reason


def test_compare_variants_renders_md_with_disclaimer() -> None:
    gold_rows = [_gold_row(qid="q1", expected_pid="P-A")]
    silver_rows = [_silver_row(qid="s1", expected_pid="P-A")]
    base = _build_variant_result(
        variant="b", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-X", "X"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-X", "X"))},
    )
    cand = _build_variant_result(
        variant="c", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-A", "A"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-A", "A"))},
    )
    comp = compare_variants(baseline=base, candidates=[cand])
    md = render_comparison_report(comp)
    assert HUMAN_FOCUS_DISCLAIMER in md
    # Headline + bucket + silver guardrail tables present.
    assert "primary_score" in md
    assert "Silver guardrail" in md


def test_render_comparison_report_carries_promotion_target_framing() -> None:
    """The renderer must embed PROMOTION_TARGET_FRAMING verbatim so the
    reader cannot confuse a retrieval-config change with another
    embedding-text variant promotion. Pinned because a future
    contributor who edits the renderer accidentally would erase this
    framing without a test failing otherwise.
    """
    gold_rows = [_gold_row(qid="q1", expected_pid="P-A")]
    silver_rows = [_silver_row(qid="s1", expected_pid="P-A")]
    base = _build_variant_result(
        variant="b", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-X", "X"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-X", "X"))},
    )
    cand = _build_variant_result(
        variant="c", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-A", "A"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-A", "A"))},
    )
    comp = compare_variants(baseline=base, candidates=[cand])
    md = render_comparison_report(comp)
    # The full clarification block, not just the disclaimer.
    assert PROMOTION_TARGET_FRAMING in md
    # And explicitly: the words "embedding-text variant" must be
    # present so a reader skimming the table headers cannot miss the
    # framing.
    assert "embedding-text variant" in md
    # The "## Promotion target clarification" header must appear so
    # the reviewer can navigate to it without reading the disclaimer.
    assert "## Promotion target clarification" in md


def test_render_comparison_report_does_not_promote_title_section() -> None:
    """Belt-and-suspenders: the renderer must NOT produce text that
    frames a `cand_title_section_top10` candidate as the promotion
    target — even when one is among the candidates and accidentally
    gets a higher primary_score (it doesn't on the real data, but if
    it did, the framing block must keep the reader honest).
    """
    gold_rows = [_gold_row(qid="q1", expected_pid="P-A")]
    silver_rows = [_silver_row(qid="s1", expected_pid="P-A")]
    base = _build_variant_result(
        variant="b", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-X", "X"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-X", "X"))},
    )
    cand = _build_variant_result(
        variant="cand_title_section_top10", gold_rows=gold_rows,
        retrievals={"q1": _docs(("P-A", "A"))},
        silver_rows=silver_rows,
        silver_retrievals={"s1": _docs(("P-A", "A"))},
    )
    comp = compare_variants(baseline=base, candidates=[cand])
    md = render_comparison_report(comp)
    # The framing block must still tell the reader that this candidate
    # is a previous embedding-text variant, NOT the change being
    # proposed for promotion.
    assert PROMOTION_TARGET_FRAMING in md
    assert "previous" in md.lower()


# ---------------------------------------------------------------------------
# CLI replay-mode end-to-end test
# ---------------------------------------------------------------------------


def test_cli_replay_end_to_end(tmp_path: Path) -> None:
    """Drive the scripts.phase7_human_gold_tune CLI in replay mode.

    Uses synthetic gold + silver files plus a hand-crafted retrieval
    JSONL per variant; verifies every output the spec promises.
    """
    from scripts.phase7_human_gold_tune import (
        DEFAULT_VARIANTS, RetrievalResult, main, write_retrieval_jsonl,
    )

    gold_path = _write_csv(tmp_path / "gold.csv", [
        _make_full_gold_row(query_id="q1", eval_weight="1"),
        _make_full_gold_row(
            query_id="q2", eval_weight="0.4",
            human_label="PARTIALLY_SUPPORTED", eval_use="PARTIALLY_SUPPORTED",
        ),
    ])
    silver_path = _write_silver_jsonl(tmp_path / "silver.jsonl", [
        {
            "query_id": "s1", "query": "q1", "query_type": "direct_title",
            "bucket": "main_work", "silver_expected_title": "T",
            "silver_expected_page_id": "P-A",
            "expected_section_path": ["개요"], "expected_not_in_corpus": False,
            "leakage_risk": "low",
        },
    ])

    base_name = DEFAULT_VARIANTS[0].name
    cand_name = DEFAULT_VARIANTS[1].name

    base_rows = [
        RetrievalResult(
            variant=base_name, query_id="q1", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-X", "X"))),
        ),
        RetrievalResult(
            variant=base_name, query_id="q2", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-Y", "Y"))),
        ),
        RetrievalResult(
            variant=base_name, query_id="s1", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-A", "A"))),
        ),
    ]
    cand_rows = [
        RetrievalResult(
            variant=cand_name, query_id="q1", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-A", "A"))),
        ),
        RetrievalResult(
            variant=cand_name, query_id="q2", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-A", "A"))),
        ),
        RetrievalResult(
            variant=cand_name, query_id="s1", query="q",
            elapsed_ms=1.0, docs=tuple(_docs(("P-A", "A"))),
        ),
    ]
    base_jsonl = write_retrieval_jsonl(base_rows, tmp_path / "base.jsonl")
    cand_jsonl = write_retrieval_jsonl(cand_rows, tmp_path / "cand.jsonl")

    report_dir = tmp_path / "report"
    rc = main([
        "--gold-path", str(gold_path),
        "--silver-path", str(silver_path),
        "--report-dir", str(report_dir),
        "--variants", f"{base_name},{cand_name}",
        "--baseline-variant", base_name,
        "--baseline-results", str(base_jsonl),
        "--candidate-results", str(cand_jsonl),
        "--include-silver-guardrail",
    ])
    assert rc == 0

    # Every promised file exists.
    for fname in [
        "baseline_gold_summary.json",
        "baseline_silver_summary.json",
        "candidate_results.jsonl",
        "comparison_summary.json",
        "comparison_report.md",
        "failure_audit_gold.md",
        "failure_audit_gold.jsonl",
        "failure_audit_silver.md",
        "failure_audit_silver.jsonl",
        "best_config.json",
        "best_config.env",
        "manifest.json",
    ]:
        assert (report_dir / fname).exists(), f"missing: {fname}"

    # MD carries the disclaimer.
    md_text = (report_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert HUMAN_FOCUS_DISCLAIMER in md_text

    # Comparison summary names the candidate as best (clean win).
    summary = json.loads(
        (report_dir / "comparison_summary.json").read_text(encoding="utf-8"),
    )
    assert summary["best_variant"] == cand_name
    assert summary["deltas"][cand_name]["primary_score"] > 0

    # Manifest lists both variants and pins the disclaimer.
    manifest = json.loads(
        (report_dir / "manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["best_variant"] == cand_name
    assert manifest["human_focus_disclaimer"] == HUMAN_FOCUS_DISCLAIMER
    variant_names = {v["name"] for v in manifest["variants"]}
    assert {base_name, cand_name} <= variant_names
