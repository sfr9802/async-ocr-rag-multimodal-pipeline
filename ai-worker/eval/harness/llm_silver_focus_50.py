"""Focus-50 subset selector — 50 rows hand-curated from the LLM silver-500.

Why this exists
---------------
After producing the full LLM silver-500 (``queries_v4_llm_silver_500.jsonl``),
the user asked for a 50-row focused subset. Use cases:

  - Early review / smoke check of the new LLM-authored set's *quality*
    before commissioning the 100-row human audit.
  - A quick way to eyeball every (bucket, query_type) cell at a manageable
    table size.
  - Lightweight regression A/B against the keyword-sanity-500 set.

The 50-row picker reuses ``llm_silver_human_seed._pick_stratified``
with a custom cross-tab whose marginals are exactly half the
``DEFAULT_CROSS_TAB`` of the human-seed-100 export (with the
not_in_corpus column rounded down from 5 → 3 since 50 isn't divisible
by the 100-row marginals cleanly).

Determinism: a fixed seed (``42``) shuffles each (bucket, query_type)
candidate list. Two runs over the same input yield byte-identical
files.

Schema: matches the human-audit seed schema exactly (carries the same
``human_*`` empty audit fields). The reviewer can fill them in just
like the 100-row seed — at smaller scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

from eval.harness.llm_silver_human_seed import (
    HumanSeedConfig,
    HumanSeedExport,
    _pick_stratified,
    _record_to_row,
    write_outputs,
)


# Bucket / query_type targets — half of the 100-row spec, rounded so the
# total hits 50 and every cell is feasible against the silver-500 supply.
FOCUS50_BUCKET_TARGETS: Mapping[str, int] = {
    "main_work":       13,
    "subpage_generic": 17,
    "subpage_named":   17,
    "not_in_corpus":    3,
}

FOCUS50_QUERY_TYPE_TARGETS: Mapping[str, int] = {
    "direct_title":                  5,
    "paraphrase_semantic":          12,
    "section_intent":               10,
    "indirect_entity":              10,
    "alias_variant":                 5,
    "ambiguous":                     5,
    "unanswerable_or_not_in_corpus": 3,
}


# Hand-derived 4×7 cross-tab whose row sums hit ``FOCUS50_BUCKET_TARGETS``
# and column sums hit ``FOCUS50_QUERY_TYPE_TARGETS`` simultaneously.
#
#                  direct  para  section  indirect  alias  ambig  unans   ROW
#   main_work          2     3        0         1      3      4      0   = 13
#   subpage_generic    2     5        7         1      1      1      0   = 17
#   subpage_named      1     4        3         8      1      0      0   = 17
#   not_in_corpus      0     0        0         0      0      0      3   =  3
#                  -----  ----  ------  --------  -----  -----  -----   ---
#   COL TOTAL          5    12       10        10      5      5      3   50
FOCUS50_CROSS_TAB: Mapping[Tuple[str, str], int] = {
    ("main_work",        "direct_title"):                  2,
    ("main_work",        "paraphrase_semantic"):           3,
    ("main_work",        "indirect_entity"):               1,
    ("main_work",        "alias_variant"):                 3,
    ("main_work",        "ambiguous"):                     4,
    ("subpage_generic",  "direct_title"):                  2,
    ("subpage_generic",  "paraphrase_semantic"):           5,
    ("subpage_generic",  "section_intent"):                7,
    ("subpage_generic",  "indirect_entity"):               1,
    ("subpage_generic",  "alias_variant"):                 1,
    ("subpage_generic",  "ambiguous"):                     1,
    ("subpage_named",    "direct_title"):                  1,
    ("subpage_named",    "paraphrase_semantic"):           4,
    ("subpage_named",    "section_intent"):                3,
    ("subpage_named",    "indirect_entity"):               8,
    ("subpage_named",    "alias_variant"):                 1,
    ("not_in_corpus",    "unanswerable_or_not_in_corpus"): 3,
}


def make_focus50_config(seed: int = 42) -> HumanSeedConfig:
    """Default config for the focus-50 export."""
    return HumanSeedConfig(
        target_total=50,
        bucket_targets=dict(FOCUS50_BUCKET_TARGETS),
        query_type_targets=dict(FOCUS50_QUERY_TYPE_TARGETS),
        cross_tab=dict(FOCUS50_CROSS_TAB),
        seed=int(seed),
    ).validate()


def build_focus50(silver_path: Path, *, seed: int = 42) -> HumanSeedExport:
    """Read the LLM silver-500 JSONL and return the 50-row picked subset."""
    import json
    cfg = make_focus50_config(seed=seed)
    rows = []
    with Path(silver_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(_record_to_row(json.loads(line)))
    selected, audit = _pick_stratified(rows, config=cfg)
    return HumanSeedExport(rows=selected, audit=audit, candidate_count=len(rows))
