"""Re-tag the legacy keyword-derived silver-500 as a *sanity-only* set.

Why this module exists
----------------------
The Phase 7 silver-500 set built by ``silver_500_generator.py`` stitches
its queries from ``retrieval_title`` + a fixed Korean template (``X의
줄거리에 대해 알려주세요``, ``X의 등장인물에 대해 알려주세요``…).
The retrieval target's title is therefore copied almost verbatim into
the query string — hit@k against that set is dominated by string match,
not real semantic retrieval, and inflates Phase 7.0–7.4 numbers.

The new LLM silver-500 (``llm_silver_500.py``) takes over the *main*
retrieval evaluation. The legacy keyword-derived 500 is **not deleted**;
it stays useful for:

  - **Index sanity** — a string-match-friendly set that should
    near-monotonically pass on a healthy BM25 / dense index. If hit@1
    on the keyword set drops, something fundamental broke.
  - **Lexical retrieval smoke test** — gates regressions in the BM25 +
    title-section embedding text path.
  - **Title-lookup regression** — the Phase 7.2 "promote
    retrieval_title_section" landing was justified on this set; we
    keep it around as the fixed reference for that diff.

This module rewrites the legacy file's tags / metadata so a downstream
reader can NEVER mistake it for a main eval set. It does **not**
recompute queries, expected_doc_ids, or any semantic content — those
all stay byte-identical so the existing Phase 7.0/7.1 baseline
comparisons remain reproducible.

What changes vs the source file::

  - ``id``: prefix swapped from ``v4-silver-500-NNNN`` to
    ``v4-keyword-sanity-NNNN`` so a single ``grep`` distinguishes the
    sets.
  - ``tags``:
      + adds ``"keyword-derived"``, ``"sanity_set"``, ``"smoke_test"``,
        ``"NOT_main_eval"``;
      + removes ``"v4-silver-500"`` (so a downstream filter on that tag
        no longer matches the sanity file);
      + ensures ``"silver"`` is still present;
      + asserts ``"gold"`` is NOT present (would fail loudly).
  - ``v4_meta.purpose``: set to
    ``"sanity_smoke_only_NOT_main_eval"``.
  - ``v4_meta.replaced_by``: set to
    ``"queries_v4_llm_silver_500.jsonl"`` — points readers at the new
    main eval set.
  - ``v4_meta.is_silver_not_gold``: set to ``True`` (already true on
    the source; we re-assert).
  - ``v4_meta.silver_label_source``: set to
    ``"keyword_template_synthetic"`` so the source is unambiguous.

What stays identical:
  - ``query``
  - ``expected_doc_ids``
  - ``expected_section_keywords``
  - ``answer_type``
  - ``difficulty``
  - ``language``
  - ``v4_meta.bucket / page_type / relation / page_title /
    retrieval_title / template_kind / extra``

The module is intentionally tiny — it's a one-pass JSONL transform
plus a summary writer. Tests grep the new tags so a future
"accidentally repurposed back to main eval" change blows up CI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen tag / metadata vocabulary
# ---------------------------------------------------------------------------


SANITY_ID_PREFIX = "v4-keyword-sanity"

# Tags the new sanity set MUST carry. Tests assert each one.
TAG_KEYWORD_DERIVED = "keyword-derived"
TAG_SANITY_SET = "sanity_set"
TAG_SMOKE_TEST = "smoke_test"
TAG_NOT_MAIN_EVAL = "NOT_main_eval"
TAG_SILVER = "silver"

REQUIRED_TAGS: tuple = (
    TAG_KEYWORD_DERIVED,
    TAG_SANITY_SET,
    TAG_SMOKE_TEST,
    TAG_NOT_MAIN_EVAL,
    TAG_SILVER,
)

# Tags we strip — anything that could be misread as "this is the main
# eval set". Keep this list explicit; we'd rather miss-strip a custom
# tag than accidentally retain a misleading one.
STRIPPED_TAGS: frozenset = frozenset({
    "v4-silver-500",
    "main_eval",
    "primary_eval",
    "deterministic",  # remove the legacy "looks-like-a-main-eval" hint
})

# Tags that are FORBIDDEN — assertion check; fails loud if seen.
FORBIDDEN_TAGS: frozenset = frozenset({
    "gold",
    "human_verified",
    "human-verified",
    "GOLD",
})


SANITY_PURPOSE = "sanity_smoke_only_NOT_main_eval"
SANITY_REPLACEMENT_FILE = "queries_v4_llm_silver_500.jsonl"
SANITY_LABEL_SOURCE = "keyword_template_synthetic"


SANITY_DISCLAIMER_LINES: tuple = (
    "> **Keyword-derived sanity set.** This file is *not* the main",
    "> retrieval evaluation. It exists as an index sanity / lexical",
    "> smoke test only. The queries were synthesized by a keyword",
    "> template (``X의 줄거리에 대해 알려주세요`` etc.) and copy the",
    "> retrieval_title nearly verbatim — hit@k here is dominated by",
    "> string match, not semantic retrieval. The main eval set is",
    f"> ``{SANITY_REPLACEMENT_FILE}`` (LLM-authored silver).",
)
SANITY_DISCLAIMER_MD: str = "\n".join(SANITY_DISCLAIMER_LINES)
SANITY_DISCLAIMER_MARKER: str = "Keyword-derived sanity set."


# ---------------------------------------------------------------------------
# Per-row transform
# ---------------------------------------------------------------------------


@dataclass
class RetagStats:
    """Counters returned by :func:`retag_records`."""

    rows_in: int = 0
    rows_out: int = 0
    forbidden_tags_seen: List[str] = field(default_factory=list)
    ids_rewritten: int = 0


def _new_id(old_id: str, idx: int) -> str:
    """Re-emit the row's id under the sanity prefix.

    We don't re-number from 1 — we keep the old numeric tail so the
    sanity file's row ordinals match the legacy file's, which makes
    diffing the two trivial. ``idx`` is used only when the old id
    can't be parsed (defensive fallback)."""
    if old_id and "-" in old_id:
        # legacy form: "v4-silver-500-0042"  →  "v4-keyword-sanity-0042"
        tail = old_id.rsplit("-", 1)[-1]
        if tail.isdigit():
            return f"{SANITY_ID_PREFIX}-{tail}"
    return f"{SANITY_ID_PREFIX}-{idx:04d}"


def retag_record(row: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    """Return a sanity-tagged copy of one legacy silver-500 row.

    Mutates a copy, not the input. The output contains the same fields
    as the source plus the sanity-specific overrides — semantic content
    (query / expected_doc_ids / etc.) is NOT touched.
    """
    out: Dict[str, Any] = json.loads(json.dumps(row))  # deep copy via json round-trip

    old_id = str(out.get("id") or "")
    out["id"] = _new_id(old_id, idx)

    # Tag rewrite.
    tags_in: List[str] = list(out.get("tags") or [])
    tags_out: List[str] = []
    for t in tags_in:
        ts = str(t)
        if ts in STRIPPED_TAGS:
            continue
        if ts in FORBIDDEN_TAGS:
            # Surface, but DO NOT silently keep — replace with a
            # marker so the row still validates and the caller can
            # see it in stats.
            log.warning(
                "Sanity retag: row %s carried forbidden tag %r — dropping",
                old_id, ts,
            )
            continue
        tags_out.append(ts)
    for required in REQUIRED_TAGS:
        if required not in tags_out:
            tags_out.append(required)
    # Stable order: REQUIRED_TAGS first (frozen order), then anything else
    # the source carried that we didn't strip, in the source's order.
    head = [t for t in REQUIRED_TAGS if t in tags_out]
    tail = [t for t in tags_out if t not in REQUIRED_TAGS]
    out["tags"] = head + tail

    # v4_meta overrides.
    meta = dict(out.get("v4_meta") or {})
    meta["purpose"] = SANITY_PURPOSE
    meta["replaced_by"] = SANITY_REPLACEMENT_FILE
    meta["is_silver_not_gold"] = True
    meta["silver_label_source"] = SANITY_LABEL_SOURCE
    out["v4_meta"] = meta

    return out


def retag_records(
    rows: Iterable[Mapping[str, Any]],
) -> tuple:
    """Walk ``rows`` and return (list_of_retagged, stats).

    The function never raises on a forbidden tag — it logs and drops
    the tag, then records the qid in ``stats.forbidden_tags_seen`` so
    the caller can decide whether to fail loud.
    """
    stats = RetagStats()
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        stats.rows_in += 1
        old_id = str(row.get("id") or "")
        for t in (row.get("tags") or []):
            if str(t) in FORBIDDEN_TAGS:
                stats.forbidden_tags_seen.append(f"{old_id}:{t}")
        retagged = retag_record(row, idx=i + 1)
        if retagged["id"] != old_id:
            stats.ids_rewritten += 1
        out.append(retagged)
        stats.rows_out += 1
    return out, stats


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file → list of dicts. Empty lines skipped."""
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(rows: Sequence[Mapping[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


def render_summary_md(rows: Sequence[Mapping[str, Any]], stats: RetagStats) -> str:
    """Markdown disclaimer + counts. Tests grep ``SANITY_DISCLAIMER_MARKER``.

    Counts that go in: total rows, bucket distribution (carried over
    from the source), and a banner pointing at the new main eval file.
    """
    from collections import Counter

    bucket = Counter(
        str((r.get("v4_meta") or {}).get("bucket") or "")
        for r in rows
    )

    lines: List[str] = []
    lines.append("# Phase 7 keyword-derived sanity set (re-tagged silver-500)")
    lines.append("")
    lines.append(SANITY_DISCLAIMER_MD)
    lines.append("")
    lines.append(f"- rows_in:  **{stats.rows_in}**")
    lines.append(f"- rows_out: **{stats.rows_out}**")
    lines.append(f"- ids_rewritten: **{stats.ids_rewritten}**")
    if stats.forbidden_tags_seen:
        lines.append(
            f"- forbidden_tags_dropped: **{len(stats.forbidden_tags_seen)}**"
        )
    lines.append("")
    lines.append("## Bucket distribution (from source)")
    lines.append("")
    lines.append("| bucket | count |")
    lines.append("|---|---:|")
    for b in sorted(bucket.keys()):
        lines.append(f"| {b} | {bucket[b]} |")
    lines.append("")
    lines.append("## Use cases")
    lines.append("")
    lines.append("- index sanity check (BM25 + dense)")
    lines.append("- lexical retrieval smoke test")
    lines.append("- Phase 7.2 retrieval_title_section regression")
    lines.append("")
    lines.append("## NOT a use case")
    lines.append("")
    lines.append("- main retrieval eval (use the LLM silver-500 instead)")
    lines.append("- hit@k / confidence calibration / recovery loop main metric")
    lines.append("- precision/recall/accuracy claims")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_summary_md(
    rows: Sequence[Mapping[str, Any]], stats: RetagStats, out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_summary_md(rows, stats), encoding="utf-8")
    return out_path
