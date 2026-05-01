"""Human-audit seed exporter for the LLM silver-500 set (100-row export).

Why this exists
---------------
The LLM silver-500 set (``llm_silver_500.py``) carries hand-authored
queries + silver target labels — they are *not* human-verified gold.
Before any precision / recall / accuracy claim can be made off the
set, a human reviewer needs to confirm the silver labels for a
stratified subset.

This module draws that subset and emits the JSONL / CSV / Markdown
trio a human reviewer can fill in. The reviewer's verdict gets stored
in ``human_label`` (one of seven allowed values) — when populated,
those rows promote from "silver" to "gold-eligible".

Schema vs the legacy ``human_gold_seed_export``
-----------------------------------------------
The legacy exporter (``eval/harness/human_gold_seed_export.py``) takes
the keyword-derived silver-500 + Phase 7.0/7.3/7.4 outputs as input.
This new exporter targets the **LLM silver-500's** flat schema
(``query_id`` / ``silver_expected_title`` / ``silver_expected_page_id``
/ ``lexical_overlap`` / ``leakage_risk``) and outputs 100 rows by
default with the spec's distribution (25/35/35/5 across buckets).

Distribution
------------
Default targets::

    main_work        25
    subpage_generic  35
    subpage_named    35
    not_in_corpus     5
                    ----
    total           100

Within each bucket the picker tries to match the spec's query_type
breakdown::

    direct_title                10
    paraphrase_semantic         25
    section_intent              20
    indirect_entity             20
    alias_variant               10
    ambiguous                   10
    unanswerable_or_not_in_corpus 5

Picks are deterministic: a fixed seed shuffles each (bucket,
query_type) candidate list, the picker takes the head until the
target is met, then a deficit-aware fallback fills any remaining
budget from the over-allocated bucket. Calling the export twice on
the same input yields byte-identical files.

Allowed ``human_label`` values
------------------------------
  - SUPPORTED          silver target is correct and well-supported.
  - PARTIALLY_SUPPORTED silver target is correct but the supporting
                       chunk is weak / partial.
  - WRONG_TARGET       silver target is wrong, the corpus has the
                       right answer somewhere else (filled into
                       ``human_correct_*``).
  - AMBIGUOUS_QUERY    the query itself is under-specified.
  - NOT_IN_CORPUS      the corpus genuinely has no matching content.
  - BAD_SILVER_LABEL   the silver label is malformed (e.g. wrong
                       bucket assignment).
  - QUERY_LEAKAGE      the query echoes the silver target verbatim,
                       so a hit would be trivial — re-author needed.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Frozen taxonomies
# ---------------------------------------------------------------------------


HUMAN_LABEL_SUPPORTED = "SUPPORTED"
HUMAN_LABEL_PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
HUMAN_LABEL_WRONG_TARGET = "WRONG_TARGET"
HUMAN_LABEL_AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
HUMAN_LABEL_NOT_IN_CORPUS = "NOT_IN_CORPUS"
HUMAN_LABEL_BAD_SILVER_LABEL = "BAD_SILVER_LABEL"
HUMAN_LABEL_QUERY_LEAKAGE = "QUERY_LEAKAGE"

ALLOWED_HUMAN_LABELS: Tuple[str, ...] = (
    HUMAN_LABEL_SUPPORTED,
    HUMAN_LABEL_PARTIALLY_SUPPORTED,
    HUMAN_LABEL_WRONG_TARGET,
    HUMAN_LABEL_AMBIGUOUS_QUERY,
    HUMAN_LABEL_NOT_IN_CORPUS,
    HUMAN_LABEL_BAD_SILVER_LABEL,
    HUMAN_LABEL_QUERY_LEAKAGE,
)


_BUCKET_MAIN_WORK = "main_work"
_BUCKET_SUBPAGE_GENERIC = "subpage_generic"
_BUCKET_SUBPAGE_NAMED = "subpage_named"
_BUCKET_NOT_IN_CORPUS = "not_in_corpus"


DEFAULT_BUCKET_TARGETS: Mapping[str, int] = {
    _BUCKET_MAIN_WORK: 25,
    _BUCKET_SUBPAGE_GENERIC: 35,
    _BUCKET_SUBPAGE_NAMED: 35,
    _BUCKET_NOT_IN_CORPUS: 5,
}

DEFAULT_QUERY_TYPE_TARGETS: Mapping[str, int] = {
    "direct_title": 10,
    "paraphrase_semantic": 25,
    "section_intent": 20,
    "indirect_entity": 20,
    "alias_variant": 10,
    "ambiguous": 10,
    "unanswerable_or_not_in_corpus": 5,
}


# Canonical (bucket × query_type) cross-tab that hits both axes' targets
# exactly. Hand-derived from the spec's bucket / query_type targets and
# the LLM silver-500 source supply per cell — see docstring for the
# feasibility check. The 28-cell ideal allocation:
#
#                  direct  para  section  indirect  alias  ambig  unans   ROW
#   main_work          3     6        0         2      6      8      0   = 25
#   subpage_generic    4    10       15         2      2      2      0   = 35
#   subpage_named      3     9        5        16      2      0      0   = 35
#   not_in_corpus      0     0        0         0      0      0      5   =  5
#                  -----  ----  ------  --------  -----  -----  -----   ---
#   COL TOTAL         10    25       20        20     10     10      5   100
#
# Both row totals and column totals match the spec exactly. Source supply
# in every non-zero cell is at least 2x the requested allocation, so the
# picker has slack on every cell.
DEFAULT_CROSS_TAB: Mapping[Tuple[str, str], int] = {
    (_BUCKET_MAIN_WORK,        "direct_title"):                  3,
    (_BUCKET_MAIN_WORK,        "paraphrase_semantic"):           6,
    (_BUCKET_MAIN_WORK,        "indirect_entity"):               2,
    (_BUCKET_MAIN_WORK,        "alias_variant"):                 6,
    (_BUCKET_MAIN_WORK,        "ambiguous"):                     8,
    (_BUCKET_SUBPAGE_GENERIC,  "direct_title"):                  4,
    (_BUCKET_SUBPAGE_GENERIC,  "paraphrase_semantic"):          10,
    (_BUCKET_SUBPAGE_GENERIC,  "section_intent"):               15,
    (_BUCKET_SUBPAGE_GENERIC,  "indirect_entity"):               2,
    (_BUCKET_SUBPAGE_GENERIC,  "alias_variant"):                 2,
    (_BUCKET_SUBPAGE_GENERIC,  "ambiguous"):                     2,
    (_BUCKET_SUBPAGE_NAMED,    "direct_title"):                  3,
    (_BUCKET_SUBPAGE_NAMED,    "paraphrase_semantic"):           9,
    (_BUCKET_SUBPAGE_NAMED,    "section_intent"):                5,
    (_BUCKET_SUBPAGE_NAMED,    "indirect_entity"):              16,
    (_BUCKET_SUBPAGE_NAMED,    "alias_variant"):                 2,
    (_BUCKET_NOT_IN_CORPUS,    "unanswerable_or_not_in_corpus"): 5,
}


# Disclaimer the .md output carries verbatim. Tests grep this marker.
HUMAN_GOLD_DISCLAIMER_MARKER = "Human-audit seed (NOT gold yet)"
HUMAN_GOLD_DISCLAIMER_MD = (
    "> **Human-audit seed (NOT gold yet).** This file is a 100-row\n"
    "> stratified sample of the LLM-authored silver-500 set. The\n"
    "> ``silver_expected_*`` columns are silver hypotheses, not\n"
    "> human-verified gold — the reviewer must fill ``human_label``\n"
    "> with one of the allowed values before any row is treated as\n"
    "> gold-eligible. Until that happens, no precision / recall /\n"
    "> accuracy claim should be made off this file."
)


# ---------------------------------------------------------------------------
# Row + sampling
# ---------------------------------------------------------------------------


@dataclass
class HumanSeedRow:
    """One audit row.

    Carries the silver record verbatim plus the five audit columns the
    human reviewer fills. ``query_type_target_count`` is filled by the
    sampler so the reviewer sees how many other rows of the same type
    landed in the seed (helps prioritise).
    """

    query_id: str
    query: str
    query_type: str
    bucket: str
    silver_expected_title: Optional[str]
    silver_expected_page_id: Optional[str]
    expected_section_path: Optional[List[str]]
    expected_not_in_corpus: bool
    rationale_for_expected_target: str
    lexical_overlap: Mapping[str, Any]
    leakage_risk: str
    tags: List[str]
    # Audit fields — empty by design.
    human_label: str = ""
    human_correct_title: str = ""
    human_correct_page_id: str = ""
    human_supporting_chunk_id: str = ""
    human_notes: str = ""


@dataclass(frozen=True)
class HumanSeedConfig:
    """Tunable knobs.

    ``cross_tab`` is the canonical (bucket, query_type) → count map. When
    set, the picker uses it directly (and ``bucket_targets`` /
    ``query_type_targets`` are derived for the audit). When None, the
    picker falls back to the legacy bucket-then-qt nested loop, which
    can leave per-qt deficits (best-effort).
    """

    target_total: int = 100
    bucket_targets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_BUCKET_TARGETS)
    )
    query_type_targets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_QUERY_TYPE_TARGETS)
    )
    cross_tab: Optional[Mapping[Tuple[str, str], int]] = field(
        default_factory=lambda: dict(DEFAULT_CROSS_TAB)
    )
    seed: int = 42

    def validate(self) -> "HumanSeedConfig":
        if self.target_total < 1:
            raise ValueError(f"target_total must be >= 1, got {self.target_total}")
        if sum(self.bucket_targets.values()) != self.target_total:
            raise ValueError(
                f"bucket_targets sum {sum(self.bucket_targets.values())} "
                f"!= target_total {self.target_total}"
            )
        if sum(self.query_type_targets.values()) != self.target_total:
            raise ValueError(
                f"query_type_targets sum "
                f"{sum(self.query_type_targets.values())} "
                f"!= target_total {self.target_total}"
            )
        if self.cross_tab is not None:
            if sum(self.cross_tab.values()) != self.target_total:
                raise ValueError(
                    f"cross_tab sum {sum(self.cross_tab.values())} "
                    f"!= target_total {self.target_total}"
                )
            # Verify cross_tab marginals match bucket/qt targets.
            from collections import Counter as _C
            row_sums: Dict[str, int] = {}
            col_sums: Dict[str, int] = {}
            for (b, qt), c in self.cross_tab.items():
                row_sums[b] = row_sums.get(b, 0) + c
                col_sums[qt] = col_sums.get(qt, 0) + c
            for b, target in self.bucket_targets.items():
                if row_sums.get(b, 0) != target:
                    raise ValueError(
                        f"cross_tab row sum for bucket {b!r} = "
                        f"{row_sums.get(b, 0)}, but bucket_target = {target}"
                    )
            for qt, target in self.query_type_targets.items():
                if col_sums.get(qt, 0) != target:
                    raise ValueError(
                        f"cross_tab col sum for qt {qt!r} = "
                        f"{col_sums.get(qt, 0)}, but qt_target = {target}"
                    )
        return self


def _record_to_row(rec: Mapping[str, Any]) -> HumanSeedRow:
    """Project a silver-500 record to the seed row schema."""
    return HumanSeedRow(
        query_id=str(rec.get("query_id") or ""),
        query=str(rec.get("query") or ""),
        query_type=str(rec.get("query_type") or ""),
        bucket=str(rec.get("bucket") or ""),
        silver_expected_title=rec.get("silver_expected_title"),
        silver_expected_page_id=rec.get("silver_expected_page_id"),
        expected_section_path=(
            list(rec["expected_section_path"])
            if rec.get("expected_section_path") else None
        ),
        expected_not_in_corpus=bool(rec.get("expected_not_in_corpus", False)),
        rationale_for_expected_target=str(rec.get("rationale_for_expected_target") or ""),
        lexical_overlap=dict(rec.get("lexical_overlap") or {}),
        leakage_risk=str(rec.get("leakage_risk") or ""),
        tags=list(rec.get("tags") or []),
    )


def _pick_stratified(
    rows: Sequence[HumanSeedRow],
    *,
    config: HumanSeedConfig,
) -> Tuple[List[HumanSeedRow], Dict[str, Any]]:
    """Pick ``target_total`` rows obeying bucket + query_type targets.

    When ``config.cross_tab`` is provided (the default for the spec's
    100-row export), we pick exactly the requested count from each
    (bucket, query_type) cell — both axes' marginals match the spec by
    construction.

    When ``cross_tab`` is None, fall back to the legacy nested-loop
    picker that hits bucket targets exactly and query_type targets on
    a best-effort basis (reports deficit in the audit).

    Returns (selected, audit_dict).
    """
    rng = random.Random(int(config.seed))

    by_bucket_qt: Dict[Tuple[str, str], List[HumanSeedRow]] = defaultdict(list)
    for r in rows:
        by_bucket_qt[(r.bucket, r.query_type)].append(r)
    for v in by_bucket_qt.values():
        v.sort(key=lambda r: r.query_id)
        rng.shuffle(v)

    selected: List[HumanSeedRow] = []
    selected_qids: set = set()
    cell_deficit: Dict[Tuple[str, str], int] = {}

    bucket_order = list(config.bucket_targets.keys())
    qt_order = list(config.query_type_targets.keys())

    if config.cross_tab is not None:
        # Cross-tab driven picker — guarantees both marginals.
        for (b, qt), need in config.cross_tab.items():
            queue = by_bucket_qt.get((b, qt), [])
            taken = 0
            for r in queue:
                if taken >= need:
                    break
                if r.query_id in selected_qids:
                    continue
                selected.append(r)
                selected_qids.add(r.query_id)
                taken += 1
            if taken < need:
                cell_deficit[(b, qt)] = need - taken
    else:
        # Legacy best-effort picker (bucket primary, qt secondary).
        bucket_budget: Dict[str, int] = dict(config.bucket_targets)
        qt_budget: Dict[str, int] = dict(config.query_type_targets)
        for b in bucket_order:
            for qt in qt_order:
                queue = by_bucket_qt.get((b, qt), [])
                for r in queue:
                    if bucket_budget.get(b, 0) <= 0:
                        break
                    if qt_budget.get(qt, 0) <= 0:
                        continue
                    if r.query_id in selected_qids:
                        continue
                    selected.append(r)
                    selected_qids.add(r.query_id)
                    bucket_budget[b] -= 1
                    qt_budget[qt] -= 1
        # Stage 2: fill remaining bucket budgets from leftover rows.
        by_bucket: Dict[str, List[HumanSeedRow]] = defaultdict(list)
        for r in rows:
            if r.query_id in selected_qids:
                continue
            by_bucket[r.bucket].append(r)
        for v in by_bucket.values():
            v.sort(key=lambda r: r.query_id)
            rng.shuffle(v)
        for b in bucket_order:
            while bucket_budget.get(b, 0) > 0 and by_bucket[b]:
                r = by_bucket[b].pop()
                selected.append(r)
                selected_qids.add(r.query_id)
                bucket_budget[b] -= 1

    # Final ordering: bucket first, then qid.
    bucket_idx = {b: i for i, b in enumerate(bucket_order)}
    selected.sort(key=lambda r: (bucket_idx.get(r.bucket, 99), r.query_id))

    bucket_actual = Counter(r.bucket for r in selected)
    qt_actual = Counter(r.query_type for r in selected)
    audit = {
        "target_total": int(config.target_total),
        "actual_total": len(selected),
        "bucket_targets": dict(config.bucket_targets),
        "bucket_actual": dict(bucket_actual),
        "bucket_deficits": {
            b: max(0, config.bucket_targets.get(b, 0) - bucket_actual.get(b, 0))
            for b in bucket_order
        },
        "query_type_targets": dict(config.query_type_targets),
        "query_type_actual": dict(qt_actual),
        "query_type_deficits": {
            qt: max(0, config.query_type_targets.get(qt, 0) - qt_actual.get(qt, 0))
            for qt in qt_order
        },
        "cross_tab_deficit": {
            f"{k[0]}|{k[1]}": v for k, v in cell_deficit.items()
        },
    }
    return selected, audit


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


@dataclass
class HumanSeedExport:
    rows: List[HumanSeedRow]
    audit: Dict[str, Any]
    candidate_count: int


def build_human_seed(
    silver_path: Path,
    *,
    config: Optional[HumanSeedConfig] = None,
) -> HumanSeedExport:
    """Read the LLM silver-500 JSONL → stratified sample → export."""
    cfg = (config or HumanSeedConfig()).validate()
    rows: List[HumanSeedRow] = []
    with Path(silver_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(_record_to_row(rec))
    selected, audit = _pick_stratified(rows, config=cfg)
    return HumanSeedExport(
        rows=selected, audit=audit, candidate_count=len(rows),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


_FIELD_ORDER: Tuple[str, ...] = (
    "query_id",
    "query",
    "query_type",
    "bucket",
    "silver_expected_title",
    "silver_expected_page_id",
    "expected_section_path",
    "expected_not_in_corpus",
    "rationale_for_expected_target",
    "lexical_overlap",
    "leakage_risk",
    "tags",
    "human_label",
    "human_correct_title",
    "human_correct_page_id",
    "human_supporting_chunk_id",
    "human_notes",
)


def _row_to_dict(row: HumanSeedRow) -> Dict[str, Any]:
    full = asdict(row)
    return {k: full.get(k) for k in _FIELD_ORDER}


def write_jsonl(rows: Sequence[HumanSeedRow], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(_row_to_dict(r), ensure_ascii=False) + "\n")
    return out_path


def _csv_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(rows: Sequence[HumanSeedRow], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(_FIELD_ORDER)
        for r in rows:
            d = _row_to_dict(r)
            writer.writerow([_csv_value(d.get(k)) for k in _FIELD_ORDER])
    return out_path


def render_md(export: HumanSeedExport, *, target_total: int) -> str:
    rows = export.rows
    audit = export.audit
    n = len(rows)

    lines: List[str] = []
    lines.append(f"# Phase 7 LLM-silver human-audit seed ({n} rows)")
    lines.append("")
    lines.append(HUMAN_GOLD_DISCLAIMER_MD)
    lines.append("")
    lines.append("Allowed `human_label` values: "
                 + ", ".join(f"`{lab}`" for lab in ALLOWED_HUMAN_LABELS) + ".")
    lines.append("")

    lines.append("## Sampling parameters")
    lines.append("")
    lines.append(f"- target_total: **{target_total}**")
    lines.append(f"- candidate_count (silver rows seen): "
                 f"**{export.candidate_count}**")
    lines.append(f"- actual_total: **{n}**")
    lines.append("")

    lines.append("### Bucket distribution")
    lines.append("")
    lines.append("| bucket | target | actual | deficit |")
    lines.append("|---|---:|---:|---:|")
    bt = audit.get("bucket_targets") or {}
    ba = audit.get("bucket_actual") or {}
    bd = audit.get("bucket_deficits") or {}
    for b in bt.keys():
        lines.append(
            f"| {b} | {int(bt.get(b, 0))} | {int(ba.get(b, 0))} | "
            f"{int(bd.get(b, 0))} |"
        )
    lines.append("")

    lines.append("### query_type distribution")
    lines.append("")
    lines.append("| query_type | target | actual | deficit |")
    lines.append("|---|---:|---:|---:|")
    qt = audit.get("query_type_targets") or {}
    qa = audit.get("query_type_actual") or {}
    qd = audit.get("query_type_deficits") or {}
    for q in qt.keys():
        lines.append(
            f"| {q} | {int(qt.get(q, 0))} | {int(qa.get(q, 0))} | "
            f"{int(qd.get(q, 0))} |"
        )
    lines.append("")

    lines.append("## Reviewer instructions")
    lines.append("")
    lines.append(
        "1. Use the JSONL or CSV file (not this Markdown) as the working copy."
    )
    lines.append(
        "2. For each row, confirm whether `silver_expected_title` / "
        "`silver_expected_page_id` is the correct answer in the corpus."
    )
    lines.append(
        "3. Pick exactly one `human_label` from the allowed list above."
    )
    lines.append(
        "4. When `human_label` ∈ {`WRONG_TARGET`, `BAD_SILVER_LABEL`, "
        "`QUERY_LEAKAGE`}, fill `human_correct_*` so the silver record "
        "can be patched."
    )
    lines.append(
        "5. Until the labels are filled, this seed is **silver, not "
        "gold** — no precision/recall/accuracy claim should be made off it."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_md(
    export: HumanSeedExport, out_path: Path, *, target_total: int,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_md(export, target_total=target_total), encoding="utf-8",
    )
    return out_path


def write_outputs(
    export: HumanSeedExport,
    *,
    out_dir: Path,
    base_name: str = "phase7_human_gold_seed_100",
    target_total: int = 100,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": write_jsonl(export.rows, out_dir / f"{base_name}.jsonl"),
        "csv": write_csv(export.rows, out_dir / f"{base_name}.csv"),
        "md": write_md(
            export, out_dir / f"{base_name}.md",
            target_total=target_total,
        ),
    }
