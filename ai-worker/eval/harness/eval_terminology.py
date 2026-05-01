"""Eval-only terminology aliases for Phase 7.x reports.

The Phase 7.0–7.4 silver query set is *heuristic / pseudo-labelled* and
has NOT been verified by a human annotator. Calling its targets
"gold" or claiming top-1 agreement is "correct" overstates what the
metric actually measures: the harness can only show whether retrieval
agrees with the silver expected_title / expected_doc_id. True precision
or recall against ground truth would require human audit.

This module exposes the corrected terminology so report writers and the
human-audit exporter can display silver-aware names without changing
the frozen taxonomy strings (e.g. ``"GOLD_NOT_IN_CANDIDATES"``) the
eval harness uses internally for its decision rules and JSONL contract.

Two pieces are intentionally separate:

  - :data:`SILVER_DISCLAIMER_MD` and :data:`SILVER_DISCLAIMER_LINES` are
    a fixed-text disclaimer the report renderers prepend to every Phase
    7.3 / 7.4 markdown summary.
  - :func:`translate_markdown_terms` rewrites individual phrases inside
    a markdown body (e.g. ``"confident_but_wrong"`` →
    ``"confident_but_silver_mismatch"``) so the reader sees the
    silver-aware name without us having to rename the underlying frozen
    constants.

The translation is deliberately conservative: each old phrase maps to
a single new phrase, replacement is whole-word so we don't double-rewrite
the body of a sentence that happens to contain the substring, and
applying the function twice yields the same text (idempotent).

This module never mutates decision data and never changes the JSONL
contract — it only re-labels human-facing text.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Tuple


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------


SILVER_DISCLAIMER_LINES: Tuple[str, ...] = (
    "> **Silver-set disclaimer.** This evaluation uses a *silver* query",
    "> set (heuristic / pseudo-labels), **not** a human-verified gold set.",
    "> Reported metrics measure agreement with the silver",
    "> ``expected_title`` / ``expected_doc_id`` targets, **not** true answer",
    "> correctness. Human audit is required before claiming precision,",
    "> recall, or accuracy. See ``phase7_human_audit_seed.{jsonl,csv,md}``",
    "> for the audit seed export this phase ships.",
)


SILVER_DISCLAIMER_MD: str = "\n".join(SILVER_DISCLAIMER_LINES)


# A short header marker the disclaimer block carries so tests can grep for
# its presence without depending on the wording of any single sentence.
SILVER_DISCLAIMER_MARKER: str = "Silver-set disclaimer."


# ---------------------------------------------------------------------------
# Terminology map: misleading "gold/correct/wrong" labels → silver-aware names
# ---------------------------------------------------------------------------


# Each entry is (old, new). Replacements are applied in three passes so
# the longer / more-specific names win over the shorter ones, and so
# compound names get rewritten before any single-word pass can break
# them. The translator is idempotent: when ``old`` is a substring of
# ``new`` (e.g. ``recovered@1`` ⊂ ``silver_target_recovered@1``) the
# pattern compiler attaches a negative lookbehind / lookahead so a
# second run cannot re-prefix the already-rewritten phrase.
#
# Adding a new mapping: pick the pass that doesn't break a longer
# entry. If ``new`` literally contains ``old``, the compiler will
# generate the lookbehind/lookahead automatically.
_PHRASE_REPLACEMENTS_FIRST_PASS: Tuple[Tuple[str, str], ...] = (
    # Compound calibration cross-tab names — rewrite before the
    # single-word "wrong"/"correct" passes can fire.
    ("confident_but_wrong", "confident_but_silver_mismatch"),
    ("confident-but-wrong", "confident_but_silver_mismatch"),
    ("low_confidence_but_correct", "low_confidence_but_silver_match"),
    ("low-confidence-but-correct", "low_confidence_but_silver_match"),
    # Reason / failure-mode aliases (markdown only — frozen string IDs
    # remain unchanged on the JSONL contract).
    ("gold_not_in_candidates", "expected_target_not_in_candidates"),
    ("GOLD_NOT_IN_CANDIDATES", "expected_target_not_in_candidates"),
    # @k metric aliases.
    ("recovered@k", "silver_target_recovered@k"),
    ("recovered@1", "silver_target_recovered@1"),
    ("recovered@3", "silver_target_recovered@3"),
    ("recovered@5", "silver_target_recovered@5"),
    ("rec@1", "silver_target_rec@1"),
    ("rec@3", "silver_target_rec@3"),
    ("rec@5", "silver_target_rec@5"),
    # Oracle rewrite framing.
    ("oracle rewrite upper", "expected_title_oracle_upper_bound"),
    ("oracle-vs-production-like", "expected_title_oracle_vs_production_like"),
    ("Oracle vs production-like", "expected_title_oracle_vs_production_like"),
    # Phrases that explicitly use the words "gold" or "correct" in the
    # rendered narrative.
    ("(CONFIDENT label, gold not in top-k)", "(CONFIDENT label, silver target not in top-k)"),
    ("(LOW_CONFIDENCE/FAILED label, gold at rank 1)", "(LOW_CONFIDENCE/FAILED label, silver target at rank 1)"),
    ("gold not in top-k", "silver target not in top-k"),
    ("gold at rank 1", "silver target at rank 1"),
    ("gold rank", "silver target rank"),
    ("gold_rank", "silver_target_rank"),
)


# Phrases applied AFTER the multi-word pass — single-word terms only.
_PHRASE_REPLACEMENTS_SECOND_PASS: Tuple[Tuple[str, str], ...] = (
    ("expected_title", "expected_title (silver)"),
)


def _all_replacements() -> Tuple[Tuple[str, str], ...]:
    return _PHRASE_REPLACEMENTS_FIRST_PASS + _PHRASE_REPLACEMENTS_SECOND_PASS


def silver_aliases() -> Dict[str, str]:
    """Return the full old→new phrase map as a plain dict.

    Useful for emitting a ``silver_terminology_aliases`` block into JSON
    summaries so a downstream consumer can map old field names back to
    silver-aware names without re-implementing this table.
    """
    return {old: new for old, new in _all_replacements()}


# Pre-compile the substitution patterns. Most entries are exact
# substrings (so a regex with ``re.escape`` is correct). When the new
# phrase contains the old phrase as a substring (e.g. ``recovered@1`` ⊂
# ``silver_target_recovered@1``), the compiler attaches a negative
# lookbehind / lookahead derived from the literal prefix / suffix of
# ``old`` inside ``new`` — that is what makes the second pass of the
# translator a no-op.
def _compile_pattern_for(old: str, new: str) -> re.Pattern:
    if old and old in new and old != new:
        idx = new.find(old)
        prefix = new[:idx]
        suffix = new[idx + len(old):]
        parts: List[str] = []
        if prefix:
            parts.append(f"(?<!{re.escape(prefix)})")
        parts.append(re.escape(old))
        if suffix:
            parts.append(f"(?!{re.escape(suffix)})")
        return re.compile("".join(parts))
    return re.compile(re.escape(old))


def _compile_pattern(old: str) -> re.Pattern:
    """Backwards-compatible single-arg compiler; assumes no overlap."""
    return re.compile(re.escape(old))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate_markdown_terms(text: str) -> str:
    """Rewrite misleading phrases in a markdown body to the silver-aware names.

    Two-pass: multi-word compound names are rewritten first so a later
    single-word pass cannot break them. The function is idempotent —
    each pattern carries a negative lookbehind/lookahead when the new
    phrase contains the old phrase as a substring, so a re-run cannot
    re-prefix the already-rewritten phrase.
    """
    if text is None:
        return None
    if not text:
        return text
    out = text
    for old, new in _PHRASE_REPLACEMENTS_FIRST_PASS:
        out = _compile_pattern_for(old, new).sub(new, out)
    for old, new in _PHRASE_REPLACEMENTS_SECOND_PASS:
        out = _compile_pattern_for(old, new).sub(new, out)
    return out


def prepend_silver_disclaimer(body: str, *, after_first_heading: bool = True) -> str:
    """Insert :data:`SILVER_DISCLAIMER_MD` near the top of a markdown body.

    When ``after_first_heading=True`` (default) the disclaimer is placed
    immediately after the first ``#``-prefixed line so the generated
    summaries stay rooted by their phase header. When ``False`` the
    disclaimer is prepended verbatim — useful for files that have no
    leading heading.

    The function is idempotent: if :data:`SILVER_DISCLAIMER_MARKER`
    already appears in the body, the body is returned unchanged.
    """
    if SILVER_DISCLAIMER_MARKER in body:
        return body
    block = SILVER_DISCLAIMER_MD + "\n"
    if not after_first_heading:
        return block + "\n" + body
    lines = body.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("# "):
            insert_at = i + 1
            # Skip a single blank line right after the heading so we
            # render as: heading\n\n<disclaimer>\n\n<rest>.
            if insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1
            head = "".join(lines[:insert_at])
            tail = "".join(lines[insert_at:])
            sep = "" if head.endswith("\n") else "\n"
            return head + sep + block + "\n" + tail
    # No heading found — fall back to verbatim prepend.
    return block + "\n" + body


def apply_silver_terminology(body: str) -> str:
    """One-shot: prepend the disclaimer + rewrite misleading phrases.

    The intended call site for report renderers — replaces what would
    otherwise be a two-line ``prepend_silver_disclaimer(translate_markdown_terms(...))``
    pattern repeated across every renderer.
    """
    return prepend_silver_disclaimer(translate_markdown_terms(body))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


SILVER_DISCLAIMER_TEXT: str = (
    "This evaluation uses a silver query set (heuristic / pseudo-labels), "
    "not a human-verified gold set. Reported metrics measure agreement "
    "with the silver expected_title / expected_doc_id targets, not true "
    "answer correctness. Human audit is required before claiming "
    "precision, recall, or accuracy."
)


def silver_disclaimer_block() -> Dict[str, object]:
    """Block to merge into the top of any aggregate JSON dump.

    Layout::

        {
            "silver_disclaimer": "...one-line disclaimer text...",
            "silver_terminology_aliases": {old: new, ...}
        }

    Tests grep for the ``silver_disclaimer`` key to confirm the new
    summaries embed the warning. The aliases block lets a downstream
    consumer (e.g. a dashboard) translate frozen reason names into the
    silver-aware display strings without re-implementing this table.
    """
    return {
        "silver_disclaimer": SILVER_DISCLAIMER_TEXT,
        "silver_terminology_aliases": silver_aliases(),
    }


# ---------------------------------------------------------------------------
# Edge-case fingerprint helpers shared with the audit exporter
# ---------------------------------------------------------------------------


# Confidence labels worth surfacing in the audit seed. These match the
# Phase 7.3 frozen vocabulary; we re-export them here so the audit
# exporter doesn't have to import the full v4_confidence_detector module
# just to enumerate edge cases.
EDGE_CASE_CONFIDENCE_LABELS: Tuple[str, ...] = (
    "CONFIDENT",
    "AMBIGUOUS",
    "LOW_CONFIDENCE",
    "FAILED",
)


# Failure reasons singled out by the brief as audit-worthy.
EDGE_CASE_FAILURE_REASONS: Tuple[str, ...] = (
    "TITLE_ALIAS_MISMATCH",
    "GENERIC_COLLISION",
)


# Synthetic edge-case label used by the audit exporter to flag rows
# whose silver target is not in the candidate top-k.
EDGE_CASE_TARGET_NOT_IN_CANDIDATES: str = "expected_target_not_in_candidates"


# Synthetic edge-case label used to flag rows that came from a Phase 7.4
# query-rewrite attempt (oracle or production_like).
EDGE_CASE_QUERY_REWRITE: str = "query_rewrite_candidate"


def all_edge_case_tags() -> Tuple[str, ...]:
    """Every edge-case tag the audit exporter may attach to a row."""
    return (
        *EDGE_CASE_CONFIDENCE_LABELS,
        *EDGE_CASE_FAILURE_REASONS,
        EDGE_CASE_TARGET_NOT_IN_CANDIDATES,
        EDGE_CASE_QUERY_REWRITE,
    )
