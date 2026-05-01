"""Phase 7.x — human-weighted gold seed 50 + silver 500 tuning harness.

The goal of this module is *not* to be a general benchmark. It is a
focus set built on top of v4 — every gold-50 row has a human reviewer's
``eval_weight`` attached, and the metric stack here treats that weight
as the source of truth for "how confidently can a single page-id be
called the right answer for this query".

Three things make this file different from the existing
``v4_ab_eval`` / ``retrieval_eval`` aggregators:

1. ``eval_use`` is a free-form column that human reviewers fill in by
   hand (``SUPPORTED`` / ``PARTIALLY_SUPPORTED`` / ``AMBIGUOUS_QUERY`` /
   ``NOT_IN_CORPUS`` / combinations thereof). The harness folds it into
   four canonical groups — STRICT_POSITIVE, SOFT_POSITIVE,
   AMBIGUOUS_PROBE, ABSTAIN_TEST — which the downstream metric layer
   uses to decide *which queries enter the primary objective at all*.
2. The gold primary score is **weighted**: a query with weight 0.4 only
   contributes 0.4 of a hit@5 point. weight 0 (AMBIGUOUS_PROBE) /
   not-in-corpus (ABSTAIN_TEST) rows are excluded from the main
   objective and reported separately.
3. The silver-500 set is a *guardrail*, not a target. A candidate that
   posts a higher gold ``primary_score`` while collapsing on silver
   subpage_named is flagged via ``SILVER_REGRESSION_WARNING`` /
   ``BUCKET_REGRESSION_WARNING`` rather than declared the winner.

All scoring functions are pure — they take a ranked list of
``RetrievedDoc`` records and return a metric. The "actually run the
retriever" wiring lives in the CLI (`scripts.phase7_human_gold_tune`),
not here, so the test suite can exercise every path with synthetic
top-k inputs.

NOTE on the disclaimer language pinned by the report writer:
this is a **subpage/section-level focus set** drawn from
``queries_v4_llm_silver_500``, then human-weighted. It is NOT a general
retrieval benchmark — `primary_score` improvements over baseline only
mean "we got better at the gold-50 subpage / named-subpage failures
this set was hand-curated to expose".
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


GOLD_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "query",
    "query_type",
    "bucket",
    "silver_expected_title",
    "silver_expected_page_id",
    "expected_section_path",
    "expected_not_in_corpus",
    "human_label",
    "human_correct_title",
    "human_correct_page_id",
    "human_supporting_chunk_id",
    "human_notes",
    "eval_use",
    "eval_weight",
)


# Canonical eval-group labels.
GROUP_STRICT_POSITIVE = "STRICT_POSITIVE"
GROUP_SOFT_POSITIVE = "SOFT_POSITIVE"
GROUP_AMBIGUOUS_PROBE = "AMBIGUOUS_PROBE"
GROUP_ABSTAIN_TEST = "ABSTAIN_TEST"

NORMALIZED_EVAL_GROUPS: Tuple[str, ...] = (
    GROUP_STRICT_POSITIVE,
    GROUP_SOFT_POSITIVE,
    GROUP_AMBIGUOUS_PROBE,
    GROUP_ABSTAIN_TEST,
)

POSITIVE_GROUPS: Tuple[str, ...] = (GROUP_STRICT_POSITIVE, GROUP_SOFT_POSITIVE)


# Failure-audit reason heuristics.
FAIL_TITLE_MISS = "TITLE_MISS"
FAIL_SECTION_MISS = "SECTION_MISS"
FAIL_SUBPAGE_MISS = "SUBPAGE_MISS"
FAIL_NAMED_SUBPAGE_MISS = "NAMED_SUBPAGE_MISS"
FAIL_OVER_BROAD_QUERY = "OVER_BROAD_QUERY"
FAIL_WRONG_SERIES = "WRONG_SERIES"
FAIL_WRONG_SEASON = "WRONG_SEASON"
FAIL_NOT_IN_CORPUS_CASE = "NOT_IN_CORPUS_CASE"
FAIL_UNKNOWN = "UNKNOWN"


# Disclaimer pinned by the test suite — the report writer must include
# this verbatim so the reviewer can never read it as a generic benchmark.
HUMAN_FOCUS_DISCLAIMER = (
    "This evaluation is NOT a representative retrieval-quality benchmark. "
    "It is a human-weighted focus set drawn from queries_v4_llm_silver_500, "
    "designed to surface v4 subpage / section-level retrieval failures. "
    "primary_score improvements only mean 'we got better at the gold-50 "
    "subpage / named-subpage failures this set was curated to expose'."
)


# Promotion-target framing — pinned verbatim by the test suite. The
# report writer must include this *exactly* so a reviewer skimming the
# headline table can never accidentally read the comparison as another
# embedding-text variant promotion (the kind Phase 7.2 already shipped
# when retrieval_title_section was promoted to production default).
#
# What this comparison IS: a retrieval-config sweep (top_k, candidate_k,
# MMR(λ)) over the *same* embedding-text variant the production index
# already uses (``retrieval_title_section``). What it is NOT: a
# proposal to promote a different embedding-text variant — when a
# variant like ``cand_title_section_top10`` shows up in the candidate
# list, it is the *previous* embedding-text variant kept around as a
# regression anchor, not the change being proposed for promotion.
PROMOTION_TARGET_FRAMING = (
    "Promotion target framing: this comparison sweeps retrieval-config "
    "knobs (top_k / candidate_k / MMR(λ)) over the production "
    "embedding-text variant retrieval_title_section. It is NOT a "
    "proposal to promote a different embedding-text variant — Phase 7.2 "
    "already promoted retrieval_title_section to the production default. "
    "Any candidate variant whose name starts with cand_title_section_* "
    "is a previous embedding-text variant kept as a regression anchor, "
    "not the change being proposed for promotion off this report."
)


# Promotion-target framing. Pinned in the report so a reviewer cannot
# confuse "this run tests retrieval CONFIG knobs (candidate_k, MMR)"
# with "this run is another embedding-text variant promotion". The
# rejected ``cand_title_section_top10`` candidate is the *previous*
# (Phase 7.0) embedding-text variant; it appears only as a sanity check
# that retrieval_title_section is still the right index choice — its
# regression confirms that decision and is NOT the change being
# proposed for promotion.
PROMOTION_TARGET_FRAMING = (
    "Promotion target framing: this evaluation tests retrieval CONFIG "
    "changes (candidate_k, use_mmr, mmr_lambda) on top of the "
    "production-default retrieval_title_section index. It does NOT test "
    "another embedding-text variant promotion. The "
    "`cand_title_section_top10` candidate, when present, is the "
    "*previous* (Phase 7.0) embedding-text variant — included only as a "
    "sanity check that retrieval_title_section is still the right "
    "choice. Any regression on that candidate confirms the Phase 7.2 "
    "embedding-text decision and is NOT a justification for promoting "
    "it.\n\n"
    "Reminder: gold-50 is a *human-weighted focus set* drawn from "
    "queries_v4_llm_silver_500, NOT a generic retrieval benchmark — "
    "primary_score improvements only mean we got better at the "
    "subpage / named-subpage failures the set was curated to expose. "
    "silver-500 is LLM-generated and acts as the **overfitting "
    "guardrail / sanity check**, NOT the primary tuning objective."
)


# Silver guardrail thresholds.
SILVER_HIT_AT_5_REGRESSION_THRESHOLD = 0.03   # -3pp
SILVER_BUCKET_REGRESSION_THRESHOLD = 0.05     # -5pp
SILVER_BUCKET_FOR_NAMED_GUARDRAIL = "subpage_named"


# ---------------------------------------------------------------------------
# Validation errors — surfaced as a list, not raised, so the CLI can
# print every problem at once instead of bailing on the first row.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetValidationIssue:
    """One schema problem in a gold or silver dataset row.

    ``severity`` is "error" for hard schema violations (will cause the
    row to be dropped) or "warning" for soft inconsistencies (row is
    kept but flagged).
    """

    query_id: str
    field_name: str
    severity: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldSeedValidationError(ValueError):
    """Raised when the gold-50 CSV cannot be loaded at all (missing
    required columns, every row has a NaN weight, etc).

    Per-row issues are collected onto :class:`GoldSeedDataset.issues`
    instead of being raised so the report can list the entire set.
    """


# ---------------------------------------------------------------------------
# Loaded dataset records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldRow:
    """One human-reviewed query from the gold-50 CSV.

    ``expected_page_id`` is the canonical target id chosen per the
    human_correct_page_id > silver_expected_page_id precedence rule.
    ``expected_title`` follows the same human > silver rule.
    ``normalized_eval_group`` is one of :data:`NORMALIZED_EVAL_GROUPS`.
    """

    query_id: str
    query: str
    query_type: str
    bucket: str
    silver_expected_title: str
    silver_expected_page_id: str
    expected_section_path: Tuple[str, ...]
    expected_not_in_corpus: bool
    human_label: str
    human_correct_title: str
    human_correct_page_id: str
    human_supporting_chunk_id: str
    human_notes: str
    eval_use_raw: str
    eval_weight: float
    leakage_risk: str = ""
    expected_title: str = ""
    expected_page_id: str = ""
    normalized_eval_group: str = ""

    def is_positive(self) -> bool:
        return self.normalized_eval_group in POSITIVE_GROUPS


@dataclass(frozen=True)
class SilverRow:
    """One LLM-generated silver query from the silver-500 JSONL."""

    query_id: str
    query: str
    query_type: str
    bucket: str
    silver_expected_title: str
    silver_expected_page_id: str
    expected_section_path: Tuple[str, ...]
    expected_not_in_corpus: bool
    leakage_risk: str
    overlap_risk: str
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class GoldSeedDataset:
    """A loaded gold-50 set plus any validation issues raised."""

    rows: List[GoldRow]
    issues: List[DatasetValidationIssue] = field(default_factory=list)

    def positive_rows(self) -> List[GoldRow]:
        return [r for r in self.rows if r.is_positive()]

    def by_group(self) -> Dict[str, List[GoldRow]]:
        out: Dict[str, List[GoldRow]] = {g: [] for g in NORMALIZED_EVAL_GROUPS}
        for r in self.rows:
            out.setdefault(r.normalized_eval_group, []).append(r)
        return out


@dataclass
class SilverDataset:
    """A loaded silver-500 set plus validation issues."""

    rows: List[SilverRow]
    issues: List[DatasetValidationIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# eval_use normalization
# ---------------------------------------------------------------------------


_EVAL_USE_TOKEN_NORMALIZE = re.compile(r"\s+")


def _split_eval_use_tokens(raw: str) -> List[str]:
    """Split a free-form eval_use string into upper-cased tokens.

    Reviewers separate compound labels with ``|``; some rows also use
    commas. Whitespace inside a token (``"PARTIALLY_SUPPORTED  "``) is
    collapsed before upper-casing so ``" PARTIALLY_SUPPORTED "`` and
    ``"PARTIALLY_SUPPORTED"`` round-trip to the same canonical token.
    """
    if not raw:
        return []
    parts: List[str] = []
    for chunk in re.split(r"[|,]", raw):
        stripped = _EVAL_USE_TOKEN_NORMALIZE.sub(" ", chunk).strip()
        if stripped:
            parts.append(stripped.upper())
    return parts


def normalize_eval_group(
    *,
    eval_use_raw: str,
    eval_weight: float,
    expected_not_in_corpus: bool,
    human_label: str = "",
) -> str:
    """Map (eval_use, eval_weight, expected_not_in_corpus) → canonical group.

    Order of resolution (matches the spec):

      1. NOT_IN_CORPUS / expected_not_in_corpus → ABSTAIN_TEST
      2. weight == 0 OR AMBIGUOUS_QUERY without a positive companion
         → AMBIGUOUS_PROBE
      3. weight >= 0.8 AND eval_use signals a positive label
         → STRICT_POSITIVE
      4. otherwise positive but low-confidence → SOFT_POSITIVE

    A row that includes both PARTIALLY_SUPPORTED and AMBIGUOUS_QUERY is
    a SOFT_POSITIVE — the reviewer still pinned a target page id, just
    with low confidence. We honour the weight rather than dropping the
    row entirely.
    """
    use_tokens = set(_split_eval_use_tokens(eval_use_raw))
    label_tokens = set(_split_eval_use_tokens(human_label))
    all_tokens = use_tokens | label_tokens

    if expected_not_in_corpus or "NOT_IN_CORPUS" in all_tokens:
        return GROUP_ABSTAIN_TEST

    has_positive_signal = any(
        t in all_tokens for t in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    )

    try:
        weight = float(eval_weight)
    except (TypeError, ValueError):
        weight = 0.0

    # weight == 0 with no positive token, or AMBIGUOUS_QUERY-only label
    # → probe row (top-k audit only, dropped from primary).
    if not has_positive_signal:
        # AMBIGUOUS_QUERY-only or empty label → probe.
        return GROUP_AMBIGUOUS_PROBE

    if weight <= 0.0:
        return GROUP_AMBIGUOUS_PROBE

    if weight >= 0.8 and "AMBIGUOUS_QUERY" not in all_tokens \
            and "PARTIALLY_SUPPORTED" not in all_tokens:
        return GROUP_STRICT_POSITIVE

    return GROUP_SOFT_POSITIVE


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    """Loose truthiness check tolerant of CSV string casings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _parse_section_path(value: Any) -> Tuple[str, ...]:
    """Normalize the expected_section_path field into a tuple of strings.

    The CSV stores section paths as a single string (often a JSON list
    or ``"개요"``); the JSONL stores them as a real list. Both forms
    funnel through this helper so downstream code only deals with
    tuples.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(x).strip() for x in value if str(x).strip())
    s = str(value).strip()
    if not s:
        return ()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return tuple(str(x).strip() for x in arr if str(x).strip())
        except json.JSONDecodeError:
            pass
    return (s,) if s else ()


def _parse_weight_strict(raw: Any, *, query_id: str) -> float:
    """Parse eval_weight; raise if NaN/blank/non-numeric.

    The spec is explicit: missing weight is NOT silently treated as
    zero. A missing weight means the reviewer didn't fill the row in
    yet, and silently zeroing it would let those rows leak into the
    AMBIGUOUS_PROBE bucket while looking complete.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise GoldSeedValidationError(
            f"row {query_id!r}: eval_weight is blank — fill in 0.0 / 0.4 / "
            f"0.8 / 1.0 explicitly so the reviewer can tell missing apart "
            f"from intentional zero."
        )
    try:
        w = float(raw)
    except (TypeError, ValueError) as e:
        raise GoldSeedValidationError(
            f"row {query_id!r}: eval_weight={raw!r} is not numeric: {e}"
        )
    if math.isnan(w):
        raise GoldSeedValidationError(
            f"row {query_id!r}: eval_weight is NaN."
        )
    return w


def load_human_gold_seed_50(path: Path) -> GoldSeedDataset:
    """Load the human-reviewed gold-50 CSV into :class:`GoldSeedDataset`.

    Required columns are listed in :data:`GOLD_REQUIRED_COLUMNS`. Per-
    row schema problems (duplicate query_id, blank query, positive row
    without an expected page id, NOT_IN_CORPUS row WITH an expected
    page id, etc) are recorded onto :attr:`GoldSeedDataset.issues` —
    severity ``"error"`` rows are dropped from the returned ``rows``
    list, severity ``"warning"`` rows are kept but flagged.
    """
    p = Path(path)
    if not p.exists():
        raise GoldSeedValidationError(f"gold-50 CSV not found: {p}")

    issues: List[DatasetValidationIssue] = []
    rows_out: List[GoldRow] = []
    seen_ids: set[str] = set()

    with p.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise GoldSeedValidationError(
                f"gold-50 CSV has no header row: {p}"
            )
        missing = [c for c in GOLD_REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise GoldSeedValidationError(
                f"gold-50 CSV is missing required columns: {missing}. "
                f"Header was: {reader.fieldnames}"
            )

        for raw_row in reader:
            qid = (raw_row.get("query_id") or "").strip()
            if not qid:
                issues.append(DatasetValidationIssue(
                    query_id="<blank>", field_name="query_id",
                    severity="error", message="missing query_id",
                ))
                continue
            if qid in seen_ids:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="query_id",
                    severity="error", message="duplicate query_id",
                ))
                continue
            seen_ids.add(qid)

            query = (raw_row.get("query") or "").strip()
            if not query:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="query",
                    severity="error", message="blank query text",
                ))
                continue

            try:
                weight = _parse_weight_strict(
                    raw_row.get("eval_weight"), query_id=qid,
                )
            except GoldSeedValidationError as e:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="eval_weight",
                    severity="error", message=str(e),
                ))
                continue

            expected_not_in_corpus = _truthy(raw_row.get("expected_not_in_corpus"))
            human_label = (raw_row.get("human_label") or "").strip()
            eval_use_raw = (raw_row.get("eval_use") or "").strip()

            group = normalize_eval_group(
                eval_use_raw=eval_use_raw,
                eval_weight=weight,
                expected_not_in_corpus=expected_not_in_corpus,
                human_label=human_label,
            )

            silver_title = (raw_row.get("silver_expected_title") or "").strip()
            silver_pid = (raw_row.get("silver_expected_page_id") or "").strip()
            human_title = (raw_row.get("human_correct_title") or "").strip()
            human_pid = (raw_row.get("human_correct_page_id") or "").strip()

            expected_title = human_title or silver_title
            expected_pid = human_pid or silver_pid

            # Schema warnings — kept but flagged.
            if group in POSITIVE_GROUPS and not expected_pid:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="expected_page_id",
                    severity="warning",
                    message=(
                        "positive row has no expected page id — neither "
                        "human_correct_page_id nor silver_expected_page_id "
                        "is set. Drops out of weighted hit/MRR but kept "
                        "for audit."
                    ),
                ))
            if group == GROUP_ABSTAIN_TEST and expected_pid:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="expected_page_id",
                    severity="warning",
                    message=(
                        "ABSTAIN_TEST row has an expected page id — "
                        "ignored from primary metrics; kept on the "
                        "abstain report for cross-check."
                    ),
                ))
            if group == GROUP_AMBIGUOUS_PROBE and weight > 0.0:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="eval_weight",
                    severity="warning",
                    message=(
                        "AMBIGUOUS_PROBE with non-zero weight — weight is "
                        "ignored for primary objective; row stays in the "
                        "top-k audit."
                    ),
                ))

            rows_out.append(GoldRow(
                query_id=qid,
                query=query,
                query_type=(raw_row.get("query_type") or "").strip(),
                bucket=(raw_row.get("bucket") or "").strip(),
                silver_expected_title=silver_title,
                silver_expected_page_id=silver_pid,
                expected_section_path=_parse_section_path(
                    raw_row.get("expected_section_path"),
                ),
                expected_not_in_corpus=expected_not_in_corpus,
                human_label=human_label,
                human_correct_title=human_title,
                human_correct_page_id=human_pid,
                human_supporting_chunk_id=(
                    raw_row.get("human_supporting_chunk_id") or ""
                ).strip(),
                human_notes=(raw_row.get("human_notes") or "").strip(),
                eval_use_raw=eval_use_raw,
                eval_weight=weight,
                leakage_risk=(raw_row.get("leakage_risk") or "").strip(),
                expected_title=expected_title,
                expected_page_id=expected_pid,
                normalized_eval_group=group,
            ))

    if not rows_out:
        raise GoldSeedValidationError(
            f"gold-50 CSV produced 0 valid rows after schema checks: {p}. "
            f"Issues: {[i.to_dict() for i in issues]}"
        )

    return GoldSeedDataset(rows=rows_out, issues=issues)


def load_llm_silver_500(path: Path) -> SilverDataset:
    """Load the LLM-generated silver-500 JSONL into :class:`SilverDataset`.

    Silver rows do NOT have a human-attached weight; the harness only
    uses them for guardrail / breakdown reporting, so any per-row
    weight semantics come from the gold loader, not here.
    """
    p = Path(path)
    if not p.exists():
        raise GoldSeedValidationError(f"silver-500 JSONL not found: {p}")

    rows_out: List[SilverRow] = []
    issues: List[DatasetValidationIssue] = []
    seen: set[str] = set()

    with p.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(DatasetValidationIssue(
                    query_id=f"<line-{lineno}>", field_name="<json>",
                    severity="error", message=f"bad JSON: {e}",
                ))
                continue

            qid = str(rec.get("query_id") or "").strip()
            if not qid:
                issues.append(DatasetValidationIssue(
                    query_id=f"<line-{lineno}>", field_name="query_id",
                    severity="error", message="missing query_id",
                ))
                continue
            if qid in seen:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="query_id",
                    severity="error", message="duplicate query_id",
                ))
                continue
            seen.add(qid)

            query = str(rec.get("query") or "").strip()
            if not query:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="query",
                    severity="error", message="blank query text",
                ))
                continue

            expected_not_in_corpus = bool(rec.get("expected_not_in_corpus") or False)
            silver_pid = str(rec.get("silver_expected_page_id") or "").strip()

            if not expected_not_in_corpus and not silver_pid:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="silver_expected_page_id",
                    severity="warning",
                    message=(
                        "silver row marked answerable but no "
                        "silver_expected_page_id; skipped from hit/MRR."
                    ),
                ))
            if expected_not_in_corpus and silver_pid:
                issues.append(DatasetValidationIssue(
                    query_id=qid, field_name="silver_expected_page_id",
                    severity="warning",
                    message=(
                        "silver row marked not_in_corpus but carries an "
                        "expected page id; ignored for hit/MRR."
                    ),
                ))

            overlap_risk = ""
            ov = rec.get("lexical_overlap")
            if isinstance(ov, dict):
                overlap_risk = str(ov.get("overlap_risk") or "").strip()

            rows_out.append(SilverRow(
                query_id=qid,
                query=query,
                query_type=str(rec.get("query_type") or "").strip(),
                bucket=str(rec.get("bucket") or "").strip(),
                silver_expected_title=str(rec.get("silver_expected_title") or "").strip(),
                silver_expected_page_id=silver_pid,
                expected_section_path=_parse_section_path(
                    rec.get("expected_section_path"),
                ),
                expected_not_in_corpus=expected_not_in_corpus,
                leakage_risk=str(rec.get("leakage_risk") or "").strip(),
                overlap_risk=overlap_risk,
                tags=tuple(str(t) for t in (rec.get("tags") or [])),
            ))

    if not rows_out:
        raise GoldSeedValidationError(
            f"silver-500 JSONL produced 0 valid rows: {p}. "
            f"Issues: {[i.to_dict() for i in issues]}"
        )
    return SilverDataset(rows=rows_out, issues=issues)


# ---------------------------------------------------------------------------
# Retrieved-doc record + scoring primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedDoc:
    """One ranked entry returned by a retriever for a query.

    The harness deliberately does NOT depend on the ``RetrievedChunk``
    type from ``app.capabilities.rag.generation``: keeping this a tiny
    dataclass means the test suite can build top-k lists by hand
    without any FAISS / embedder infrastructure. The CLI is responsible
    for translating a real :class:`RetrievalReport` into a list of
    ``RetrievedDoc``.

    ``page_id`` is the deduplication key for hit@k. ``section_path`` is
    used by the auxiliary section-level metric and the failure-audit
    heuristic. ``chunk_id`` lets the chunk-level audit know which
    fragment scored.
    """

    rank: int
    chunk_id: str
    page_id: str
    title: str
    section_path: Tuple[str, ...]
    score: Optional[float] = None


@dataclass(frozen=True)
class QueryRetrieval:
    """A query plus the top-k retrieval result for one variant."""

    query_id: str
    query: str
    variant: str
    docs: Tuple[RetrievedDoc, ...]


def _doc_hits_page(doc: RetrievedDoc, expected_page_id: str) -> bool:
    return bool(expected_page_id) and doc.page_id == expected_page_id


def hit_at_k(docs: Sequence[RetrievedDoc], expected_page_id: str, *, k: int) -> int:
    if not expected_page_id or k <= 0:
        return 0
    for d in docs[:k]:
        if _doc_hits_page(d, expected_page_id):
            return 1
    return 0


def first_hit_rank(
    docs: Sequence[RetrievedDoc], expected_page_id: str, *, k: int,
) -> Optional[int]:
    if not expected_page_id or k <= 0:
        return None
    for d in docs[:k]:
        if _doc_hits_page(d, expected_page_id):
            return d.rank
    return None


def mrr_at_k(
    docs: Sequence[RetrievedDoc], expected_page_id: str, *, k: int,
) -> float:
    rank = first_hit_rank(docs, expected_page_id, k=k)
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def ndcg_at_k(
    docs: Sequence[RetrievedDoc], expected_page_id: str, *, k: int,
) -> float:
    """Binary nDCG@k with a single relevant page id.

    Identical to ``v4_ab_eval._ndcg_at_10`` algebraically; reproduced
    here so the harness has no dependency on that module.
    """
    rank = first_hit_rank(docs, expected_page_id, k=k)
    if rank is None or rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def section_hit_at_k(
    docs: Sequence[RetrievedDoc],
    expected_page_id: str,
    expected_section_path: Sequence[str],
    *,
    k: int,
) -> Optional[int]:
    """Auxiliary: did any top-k chunk hit the expected (page, section) pair?

    Returns ``None`` when ``expected_section_path`` is empty — section
    matching is undefined in that case. We require a *page* match
    first, then require the retrieved section path to start with the
    expected prefix; this keeps "section hit but on the wrong page"
    from being credited.
    """
    if not expected_section_path:
        return None
    if not expected_page_id or k <= 0:
        return 0
    expected = tuple(s for s in expected_section_path if s)
    if not expected:
        return None
    for d in docs[:k]:
        if not _doc_hits_page(d, expected_page_id):
            continue
        if not d.section_path:
            continue
        if d.section_path[: len(expected)] == expected:
            return 1
    return 0


def chunk_hit_at_k(
    docs: Sequence[RetrievedDoc],
    expected_chunk_id: str,
    *,
    k: int,
) -> Optional[int]:
    if not expected_chunk_id:
        return None
    if k <= 0:
        return 0
    for d in docs[:k]:
        if d.chunk_id == expected_chunk_id:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
_MRR_K = 10
_NDCG_K = 10


# Primary-score weights (pinned by the test suite + the report writer).
PRIMARY_WEIGHTED_HIT_AT_5 = 0.45
PRIMARY_WEIGHTED_MRR_AT_10 = 0.35
PRIMARY_WEIGHTED_NDCG_AT_10 = 0.20


@dataclass
class GoldQueryEvalRow:
    """Per-query metric record built by :func:`evaluate_gold`."""

    query_id: str
    query: str
    bucket: str
    query_type: str
    normalized_eval_group: str
    eval_weight: float
    expected_title: str
    expected_page_id: str
    expected_section_path: Tuple[str, ...]
    leakage_risk: str
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    section_hit_at_5: Optional[int] = None
    section_hit_at_10: Optional[int] = None
    chunk_hit_at_10: Optional[int] = None
    first_hit_rank: Optional[int] = None
    docs: Tuple[RetrievedDoc, ...] = ()


@dataclass
class SilverQueryEvalRow:
    """Per-query metric record for a silver row."""

    query_id: str
    query: str
    bucket: str
    query_type: str
    leakage_risk: str
    overlap_risk: str
    expected_page_id: str
    expected_not_in_corpus: bool
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    mrr_at_10: float = 0.0
    first_hit_rank: Optional[int] = None


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den > 0.0 else 0.0


def evaluate_gold(
    dataset: GoldSeedDataset,
    retrievals: Mapping[str, Sequence[RetrievedDoc]],
) -> List[GoldQueryEvalRow]:
    """Score every row against its variant's top-k.

    ``retrievals`` is a mapping ``{query_id: [RetrievedDoc, ...]}``. Any
    query without an entry gets an empty top-k (counts as a miss for
    positive rows).
    """
    out: List[GoldQueryEvalRow] = []
    for r in dataset.rows:
        docs = tuple(retrievals.get(r.query_id) or ())
        row = GoldQueryEvalRow(
            query_id=r.query_id,
            query=r.query,
            bucket=r.bucket,
            query_type=r.query_type,
            normalized_eval_group=r.normalized_eval_group,
            eval_weight=r.eval_weight,
            expected_title=r.expected_title,
            expected_page_id=r.expected_page_id,
            expected_section_path=r.expected_section_path,
            leakage_risk=r.leakage_risk,
            docs=docs,
        )
        if r.expected_page_id:
            row.hit_at_1 = hit_at_k(docs, r.expected_page_id, k=1)
            row.hit_at_3 = hit_at_k(docs, r.expected_page_id, k=3)
            row.hit_at_5 = hit_at_k(docs, r.expected_page_id, k=5)
            row.hit_at_10 = hit_at_k(docs, r.expected_page_id, k=10)
            row.mrr_at_10 = mrr_at_k(docs, r.expected_page_id, k=_MRR_K)
            row.ndcg_at_10 = ndcg_at_k(docs, r.expected_page_id, k=_NDCG_K)
            row.first_hit_rank = first_hit_rank(docs, r.expected_page_id, k=10)
            row.section_hit_at_5 = section_hit_at_k(
                docs, r.expected_page_id, r.expected_section_path, k=5,
            )
            row.section_hit_at_10 = section_hit_at_k(
                docs, r.expected_page_id, r.expected_section_path, k=10,
            )
            if r.human_supporting_chunk_id:
                row.chunk_hit_at_10 = chunk_hit_at_k(
                    docs, r.human_supporting_chunk_id, k=10,
                )
        out.append(row)
    return out


def evaluate_silver(
    dataset: SilverDataset,
    retrievals: Mapping[str, Sequence[RetrievedDoc]],
) -> List[SilverQueryEvalRow]:
    out: List[SilverQueryEvalRow] = []
    for r in dataset.rows:
        docs = tuple(retrievals.get(r.query_id) or ())
        row = SilverQueryEvalRow(
            query_id=r.query_id,
            query=r.query,
            bucket=r.bucket,
            query_type=r.query_type,
            leakage_risk=r.leakage_risk,
            overlap_risk=r.overlap_risk,
            expected_page_id=r.silver_expected_page_id,
            expected_not_in_corpus=r.expected_not_in_corpus,
        )
        if r.silver_expected_page_id and not r.expected_not_in_corpus:
            row.hit_at_1 = hit_at_k(docs, r.silver_expected_page_id, k=1)
            row.hit_at_3 = hit_at_k(docs, r.silver_expected_page_id, k=3)
            row.hit_at_5 = hit_at_k(docs, r.silver_expected_page_id, k=5)
            row.hit_at_10 = hit_at_k(docs, r.silver_expected_page_id, k=10)
            row.mrr_at_10 = mrr_at_k(docs, r.silver_expected_page_id, k=_MRR_K)
            row.first_hit_rank = first_hit_rank(
                docs, r.silver_expected_page_id, k=10,
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Summarization (gold)
# ---------------------------------------------------------------------------


@dataclass
class GoldSummary:
    """Aggregate gold-50 metrics for one variant."""

    n_total: int
    n_strict_positive: int
    n_soft_positive: int
    n_ambiguous_probe: int
    n_abstain_test: int

    # Unweighted positive-only means (STRICT + SOFT).
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    # eval_weight-weighted positive-only means.
    weighted_hit_at_1: float
    weighted_hit_at_3: float
    weighted_hit_at_5: float
    weighted_hit_at_10: float
    weighted_mrr_at_10: float
    weighted_ndcg_at_10: float

    # STRICT-only unweighted means (sanity for "did we trade strict
    # for soft?").
    strict_hit_at_5: float
    strict_mrr_at_10: float

    # Auxiliary, not in primary objective.
    section_hit_at_5_when_defined: Optional[float]
    section_hit_at_10_when_defined: Optional[float]
    chunk_hit_at_10_when_defined: Optional[float]

    primary_score: float

    by_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_normalized_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_leakage_risk: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _mean_int(rows: Sequence[GoldQueryEvalRow], attr: str) -> float:
    if not rows:
        return 0.0
    return sum(int(getattr(r, attr)) for r in rows) / float(len(rows))


def _mean_float(rows: Sequence[GoldQueryEvalRow], attr: str) -> float:
    if not rows:
        return 0.0
    return sum(float(getattr(r, attr)) for r in rows) / float(len(rows))


def _weighted_mean(rows: Sequence[GoldQueryEvalRow], attr: str) -> float:
    """Weight-by-eval_weight mean over rows. Empty / zero-weight → 0."""
    if not rows:
        return 0.0
    total_weight = sum(float(r.eval_weight) for r in rows)
    if total_weight <= 0.0:
        return 0.0
    weighted = sum(float(getattr(r, attr)) * float(r.eval_weight) for r in rows)
    return weighted / total_weight


def _mean_optional(
    rows: Sequence[GoldQueryEvalRow], attr: str,
) -> Optional[float]:
    """Mean over rows where ``attr`` is not None.

    Used for section_hit / chunk_hit which return None when the gold
    row didn't define the expected section path or supporting chunk
    id. Returns None when no row contributed a value so the report
    can render "not defined" rather than 0.
    """
    vals: List[float] = []
    for r in rows:
        v = getattr(r, attr)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _gold_breakdown(
    rows: Sequence[GoldQueryEvalRow], key_fn: Callable[[GoldQueryEvalRow], str],
) -> Dict[str, Dict[str, float]]:
    """Helper: per-key aggregation for the report breakdown tables.

    Each bucket / query_type / group cell carries the same metric set
    so reviewers can read them across columns without dancing between
    schemas.
    """
    by: Dict[str, List[GoldQueryEvalRow]] = {}
    for r in rows:
        by.setdefault(key_fn(r) or "<blank>", []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for key, group_rows in sorted(by.items()):
        positives = [
            r for r in group_rows if r.normalized_eval_group in POSITIVE_GROUPS
        ]
        out[key] = {
            "n_total": float(len(group_rows)),
            "n_positive": float(len(positives)),
            "hit_at_1": _mean_int(positives, "hit_at_1"),
            "hit_at_3": _mean_int(positives, "hit_at_3"),
            "hit_at_5": _mean_int(positives, "hit_at_5"),
            "hit_at_10": _mean_int(positives, "hit_at_10"),
            "mrr_at_10": _mean_float(positives, "mrr_at_10"),
            "ndcg_at_10": _mean_float(positives, "ndcg_at_10"),
            "weighted_hit_at_5": _weighted_mean(positives, "hit_at_5"),
            "weighted_mrr_at_10": _weighted_mean(positives, "mrr_at_10"),
            "weighted_ndcg_at_10": _weighted_mean(positives, "ndcg_at_10"),
        }
    return out


def primary_score(
    *,
    weighted_hit_at_5: float,
    weighted_mrr_at_10: float,
    weighted_ndcg_at_10: float,
) -> float:
    """``0.45 * w_hit@5 + 0.35 * w_MRR@10 + 0.20 * w_nDCG@10``.

    The composite that the report and best-config selector use as the
    headline scalar — see :data:`PRIMARY_WEIGHTED_HIT_AT_5` etc. for
    the weights. Inputs are expected to be the *positive-only*,
    *eval_weight*-weighted means; AMBIGUOUS_PROBE / ABSTAIN_TEST rows
    are excluded upstream by :func:`summarize_gold`.
    """
    return round(
        PRIMARY_WEIGHTED_HIT_AT_5 * float(weighted_hit_at_5)
        + PRIMARY_WEIGHTED_MRR_AT_10 * float(weighted_mrr_at_10)
        + PRIMARY_WEIGHTED_NDCG_AT_10 * float(weighted_ndcg_at_10),
        6,
    )


def summarize_gold(rows: Sequence[GoldQueryEvalRow]) -> GoldSummary:
    """Roll per-query gold rows up to the variant-level summary."""
    positives = [r for r in rows if r.normalized_eval_group in POSITIVE_GROUPS]
    strict = [r for r in rows if r.normalized_eval_group == GROUP_STRICT_POSITIVE]

    weighted_h5 = _weighted_mean(positives, "hit_at_5")
    weighted_mrr = _weighted_mean(positives, "mrr_at_10")
    weighted_ndcg = _weighted_mean(positives, "ndcg_at_10")

    return GoldSummary(
        n_total=len(rows),
        n_strict_positive=sum(
            1 for r in rows if r.normalized_eval_group == GROUP_STRICT_POSITIVE
        ),
        n_soft_positive=sum(
            1 for r in rows if r.normalized_eval_group == GROUP_SOFT_POSITIVE
        ),
        n_ambiguous_probe=sum(
            1 for r in rows if r.normalized_eval_group == GROUP_AMBIGUOUS_PROBE
        ),
        n_abstain_test=sum(
            1 for r in rows if r.normalized_eval_group == GROUP_ABSTAIN_TEST
        ),
        hit_at_1=_mean_int(positives, "hit_at_1"),
        hit_at_3=_mean_int(positives, "hit_at_3"),
        hit_at_5=_mean_int(positives, "hit_at_5"),
        hit_at_10=_mean_int(positives, "hit_at_10"),
        mrr_at_10=_mean_float(positives, "mrr_at_10"),
        ndcg_at_10=_mean_float(positives, "ndcg_at_10"),
        weighted_hit_at_1=_weighted_mean(positives, "hit_at_1"),
        weighted_hit_at_3=_weighted_mean(positives, "hit_at_3"),
        weighted_hit_at_5=weighted_h5,
        weighted_hit_at_10=_weighted_mean(positives, "hit_at_10"),
        weighted_mrr_at_10=weighted_mrr,
        weighted_ndcg_at_10=weighted_ndcg,
        strict_hit_at_5=_mean_int(strict, "hit_at_5"),
        strict_mrr_at_10=_mean_float(strict, "mrr_at_10"),
        section_hit_at_5_when_defined=_mean_optional(
            positives, "section_hit_at_5",
        ),
        section_hit_at_10_when_defined=_mean_optional(
            positives, "section_hit_at_10",
        ),
        chunk_hit_at_10_when_defined=_mean_optional(
            positives, "chunk_hit_at_10",
        ),
        primary_score=primary_score(
            weighted_hit_at_5=weighted_h5,
            weighted_mrr_at_10=weighted_mrr,
            weighted_ndcg_at_10=weighted_ndcg,
        ),
        by_bucket=_gold_breakdown(rows, lambda r: r.bucket),
        by_query_type=_gold_breakdown(rows, lambda r: r.query_type),
        by_normalized_group=_gold_breakdown(
            rows, lambda r: r.normalized_eval_group,
        ),
        by_leakage_risk=_gold_breakdown(rows, lambda r: r.leakage_risk),
    )


# ---------------------------------------------------------------------------
# Summarization (silver)
# ---------------------------------------------------------------------------


@dataclass
class SilverSummary:
    """Aggregate silver-500 metrics for one variant."""

    n_total: int
    n_scored: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    by_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_leakage_risk: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_overlap_risk: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _silver_mean(rows: Sequence[SilverQueryEvalRow], attr: str) -> float:
    if not rows:
        return 0.0
    return sum(float(getattr(r, attr)) for r in rows) / float(len(rows))


def _silver_breakdown(
    rows: Sequence[SilverQueryEvalRow],
    key_fn: Callable[[SilverQueryEvalRow], str],
) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[SilverQueryEvalRow]] = {}
    for r in rows:
        by.setdefault(key_fn(r) or "<blank>", []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for key, group_rows in sorted(by.items()):
        scored = [
            r for r in group_rows
            if not r.expected_not_in_corpus and r.expected_page_id
        ]
        out[key] = {
            "n_total": float(len(group_rows)),
            "n_scored": float(len(scored)),
            "hit_at_1": _silver_mean(scored, "hit_at_1"),
            "hit_at_3": _silver_mean(scored, "hit_at_3"),
            "hit_at_5": _silver_mean(scored, "hit_at_5"),
            "hit_at_10": _silver_mean(scored, "hit_at_10"),
            "mrr_at_10": _silver_mean(scored, "mrr_at_10"),
        }
    return out


def summarize_silver(rows: Sequence[SilverQueryEvalRow]) -> SilverSummary:
    scored = [
        r for r in rows
        if not r.expected_not_in_corpus and r.expected_page_id
    ]
    return SilverSummary(
        n_total=len(rows),
        n_scored=len(scored),
        hit_at_1=_silver_mean(scored, "hit_at_1"),
        hit_at_3=_silver_mean(scored, "hit_at_3"),
        hit_at_5=_silver_mean(scored, "hit_at_5"),
        hit_at_10=_silver_mean(scored, "hit_at_10"),
        mrr_at_10=_silver_mean(scored, "mrr_at_10"),
        by_bucket=_silver_breakdown(rows, lambda r: r.bucket),
        by_query_type=_silver_breakdown(rows, lambda r: r.query_type),
        by_leakage_risk=_silver_breakdown(rows, lambda r: r.leakage_risk),
        by_overlap_risk=_silver_breakdown(rows, lambda r: r.overlap_risk),
    )


# ---------------------------------------------------------------------------
# Silver guardrail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailWarning:
    """One regression warning fired by :func:`evaluate_silver_guardrail`.

    ``code`` is the canonical token (SILVER_REGRESSION_WARNING /
    BUCKET_REGRESSION_WARNING) the report writer keys off; ``delta``
    is the candidate-minus-baseline difference (negative for a
    regression, since the threshold is configured as a positive number).
    """

    code: str
    metric: str
    bucket: Optional[str]
    baseline: float
    candidate: float
    delta: float
    threshold: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_silver_guardrail(
    *,
    baseline: SilverSummary,
    candidate: SilverSummary,
    hit_at_5_threshold: float = SILVER_HIT_AT_5_REGRESSION_THRESHOLD,
    bucket_threshold: float = SILVER_BUCKET_REGRESSION_THRESHOLD,
    bucket_for_named_guardrail: str = SILVER_BUCKET_FOR_NAMED_GUARDRAIL,
) -> List[GuardrailWarning]:
    """Compare candidate-vs-baseline silver metrics; emit warnings.

    Two warnings are surfaced:

      - SILVER_REGRESSION_WARNING: candidate hit@5 dropped >= 3pp.
      - BUCKET_REGRESSION_WARNING: candidate hit@5 on
        ``bucket_for_named_guardrail`` (default ``subpage_named``)
        dropped >= 5pp.

    Both thresholds are passed in so the test suite can pin them; the
    defaults match :data:`SILVER_HIT_AT_5_REGRESSION_THRESHOLD` /
    :data:`SILVER_BUCKET_REGRESSION_THRESHOLD`.
    """
    warnings: List[GuardrailWarning] = []
    delta_h5 = candidate.hit_at_5 - baseline.hit_at_5
    if delta_h5 <= -float(hit_at_5_threshold):
        warnings.append(GuardrailWarning(
            code="SILVER_REGRESSION_WARNING",
            metric="hit_at_5",
            bucket=None,
            baseline=baseline.hit_at_5,
            candidate=candidate.hit_at_5,
            delta=delta_h5,
            threshold=float(hit_at_5_threshold),
            message=(
                f"silver hit@5 dropped {abs(delta_h5)*100:.2f}pp "
                f"(>= {hit_at_5_threshold*100:.1f}pp threshold) — gold "
                f"primary_score gain may not generalize."
            ),
        ))

    base_named = baseline.by_bucket.get(bucket_for_named_guardrail) or {}
    cand_named = candidate.by_bucket.get(bucket_for_named_guardrail) or {}
    if base_named and cand_named:
        delta_named = float(cand_named.get("hit_at_5", 0.0)) - float(
            base_named.get("hit_at_5", 0.0)
        )
        if delta_named <= -float(bucket_threshold):
            warnings.append(GuardrailWarning(
                code="BUCKET_REGRESSION_WARNING",
                metric="hit_at_5",
                bucket=bucket_for_named_guardrail,
                baseline=float(base_named.get("hit_at_5", 0.0)),
                candidate=float(cand_named.get("hit_at_5", 0.0)),
                delta=delta_named,
                threshold=float(bucket_threshold),
                message=(
                    f"silver bucket={bucket_for_named_guardrail!r} hit@5 "
                    f"dropped {abs(delta_named)*100:.2f}pp "
                    f"(>= {bucket_threshold*100:.1f}pp threshold) — the "
                    f"named-subpage retrieval that the gold-50 set was "
                    f"curated to fix is regressing on the broad silver "
                    f"set; do NOT promote without a deeper audit."
                ),
            ))
    return warnings


# ---------------------------------------------------------------------------
# Failure audit
# ---------------------------------------------------------------------------


def classify_failure(
    *,
    eval_row: GoldQueryEvalRow,
    expected_section_path: Sequence[str] = (),
    expected_title: str = "",
) -> str:
    """Heuristic failure-reason label for one gold row.

    Order of resolution:

      - ABSTAIN: gold row is NOT_IN_CORPUS_CASE — surface as that
        directly so the audit table doesn't pretend a hit exists.
      - hit@10 already → no failure (caller filters first; we still
        return UNKNOWN for safety).
      - top-1 page matches but title does not → TITLE_MISS (the
        retriever picked a wrong page that nonetheless rendered as the
        correct title, e.g. a different season).
      - top-1 page hits → SECTION_MISS / NAMED_SUBPAGE_MISS depending
        on whether the gold has an expected section path.
      - top-1 misses → bucket-driven: SUBPAGE_MISS / NAMED_SUBPAGE_MISS
        / OVER_BROAD_QUERY / WRONG_SERIES / WRONG_SEASON; falls back
        to UNKNOWN when nothing fires.
    """
    if eval_row.normalized_eval_group == GROUP_ABSTAIN_TEST:
        return FAIL_NOT_IN_CORPUS_CASE
    if eval_row.hit_at_10 == 1:
        return FAIL_UNKNOWN  # caller should filter; safety net
    docs = eval_row.docs
    if not docs:
        return FAIL_UNKNOWN
    top1 = docs[0]
    expected_pid = eval_row.expected_page_id
    expected_title_eff = (expected_title or eval_row.expected_title or "").strip()

    bucket = (eval_row.bucket or "").lower()
    qtype = (eval_row.query_type or "").lower()

    # Top-1 page id matches → it's a section / chunk-level miss, not a
    # title miss. Distinguish "named subpage" (subpage_named bucket
    # with a section path) from generic section misses.
    if _doc_hits_page(top1, expected_pid):
        if expected_section_path or eval_row.expected_section_path:
            if bucket == "subpage_named":
                return FAIL_NAMED_SUBPAGE_MISS
            return FAIL_SECTION_MISS
        return FAIL_SECTION_MISS

    # Top-1 page id misses. Check title-level confusion next.
    if expected_title_eff and top1.title and \
            top1.title.strip() == expected_title_eff:
        # Same title, different page id — almost always a season / variant
        # collision. Default to WRONG_SEASON when the gold mentions one.
        if "기" in expected_title_eff or "시즌" in expected_title_eff:
            return FAIL_WRONG_SEASON
        return FAIL_WRONG_SERIES

    if expected_title_eff and top1.title and \
            (expected_title_eff in top1.title or top1.title in expected_title_eff):
        return FAIL_WRONG_SERIES

    # Bucket-driven heuristics for the remaining "title miss + page
    # miss" cases.
    if bucket == "subpage_named":
        return FAIL_NAMED_SUBPAGE_MISS
    if bucket == "subpage_generic":
        return FAIL_SUBPAGE_MISS
    if bucket == "main_work" and qtype in ("ambiguous", "indirect_entity"):
        return FAIL_OVER_BROAD_QUERY
    if bucket == "main_work":
        return FAIL_TITLE_MISS
    return FAIL_UNKNOWN


@dataclass
class FailureAuditRow:
    """One row rendered into the failure-audit MD / JSONL."""

    query_id: str
    query: str
    normalized_eval_group: str
    eval_weight: float
    bucket: str
    query_type: str
    expected_title: str
    expected_page_id: str
    expected_section_path: Tuple[str, ...]
    top1_title: Optional[str]
    top1_page_id: Optional[str]
    top1_score: Optional[float]
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    hit_at_10: int
    top_k_titles: Tuple[str, ...]
    top_k_page_ids: Tuple[str, ...]
    top_k_section_paths: Tuple[Tuple[str, ...], ...]
    failure_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "normalized_eval_group": self.normalized_eval_group,
            "eval_weight": self.eval_weight,
            "bucket": self.bucket,
            "query_type": self.query_type,
            "expected_title": self.expected_title,
            "expected_page_id": self.expected_page_id,
            "expected_section_path": list(self.expected_section_path),
            "top1_title": self.top1_title,
            "top1_page_id": self.top1_page_id,
            "top1_score": self.top1_score,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "top_k_titles": list(self.top_k_titles),
            "top_k_page_ids": list(self.top_k_page_ids),
            "top_k_section_paths": [list(p) for p in self.top_k_section_paths],
            "failure_reason": self.failure_reason,
        }


def build_failure_audit_row(
    eval_row: GoldQueryEvalRow,
    *,
    audit_top_k: int = 10,
) -> FailureAuditRow:
    """Render one :class:`GoldQueryEvalRow` into a :class:`FailureAuditRow`.

    Always included in the audit even when hit@10 == 1 — the report
    can filter by ``failure_reason != "UNKNOWN"`` if the reviewer
    only wants the misses. Keeping hits in the same JSONL means the
    reviewer can grep by query_id without joining files.
    """
    docs = eval_row.docs[:audit_top_k]
    top1 = docs[0] if docs else None
    return FailureAuditRow(
        query_id=eval_row.query_id,
        query=eval_row.query,
        normalized_eval_group=eval_row.normalized_eval_group,
        eval_weight=eval_row.eval_weight,
        bucket=eval_row.bucket,
        query_type=eval_row.query_type,
        expected_title=eval_row.expected_title,
        expected_page_id=eval_row.expected_page_id,
        expected_section_path=eval_row.expected_section_path,
        top1_title=top1.title if top1 else None,
        top1_page_id=top1.page_id if top1 else None,
        top1_score=(float(top1.score) if (top1 and top1.score is not None) else None),
        hit_at_1=eval_row.hit_at_1,
        hit_at_3=eval_row.hit_at_3,
        hit_at_5=eval_row.hit_at_5,
        hit_at_10=eval_row.hit_at_10,
        top_k_titles=tuple(d.title for d in docs),
        top_k_page_ids=tuple(d.page_id for d in docs),
        top_k_section_paths=tuple(d.section_path for d in docs),
        failure_reason=(
            classify_failure(eval_row=eval_row)
            if eval_row.hit_at_10 == 0
            or eval_row.normalized_eval_group == GROUP_ABSTAIN_TEST
            else FAIL_UNKNOWN
        ),
    )


# ---------------------------------------------------------------------------
# Variant comparison
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    """Per-variant evaluation result (gold + silver + audit)."""

    variant: str
    gold_summary: GoldSummary
    silver_summary: SilverSummary
    gold_per_query: List[GoldQueryEvalRow]
    silver_per_query: List[SilverQueryEvalRow]
    failure_audit: List[FailureAuditRow]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Baseline vs candidates comparison + best-config selection."""

    baseline_variant: str
    baseline: VariantResult
    candidates: List[VariantResult]
    guardrails: Dict[str, List[GuardrailWarning]]
    best_variant: str
    best_reason: str
    deltas: Dict[str, Dict[str, float]]


def compare_variants(
    *,
    baseline: VariantResult,
    candidates: Sequence[VariantResult],
    primary_min_delta: float = 0.0005,
) -> ComparisonResult:
    """Pick the best candidate by primary_score, subject to guardrails.

    Selection rule (matches the spec's "don't pick best by aggregate
    only"):

      1. Compute candidate vs baseline ``primary_score`` delta.
      2. Drop candidates whose ``primary_score`` delta < ``primary_min_delta``
         — they are not meaningful improvements.
      3. Drop candidates with a SILVER_REGRESSION_WARNING that is *not*
         offset by a strict_hit_at_5 improvement of at least the same
         magnitude — the gold gain has to be paid for, not silver-traded.
      4. Drop candidates with a BUCKET_REGRESSION_WARNING on
         ``subpage_named`` UNLESS strict_hit_at_5 also improved.
      5. Of the survivors, pick the one with the highest
         ``primary_score``; tiebreak on ``weighted_mrr_at_10`` then on
         the variant name.
      6. If no candidate survives, fall back to baseline and record the
         reason in ``best_reason``.
    """
    guardrails: Dict[str, List[GuardrailWarning]] = {}
    deltas: Dict[str, Dict[str, float]] = {}
    survivors: List[Tuple[VariantResult, float]] = []

    for cand in candidates:
        warns = evaluate_silver_guardrail(
            baseline=baseline.silver_summary,
            candidate=cand.silver_summary,
        )
        guardrails[cand.variant] = warns

        primary_delta = cand.gold_summary.primary_score \
            - baseline.gold_summary.primary_score
        strict_h5_delta = cand.gold_summary.strict_hit_at_5 \
            - baseline.gold_summary.strict_hit_at_5
        wh5_delta = cand.gold_summary.weighted_hit_at_5 \
            - baseline.gold_summary.weighted_hit_at_5
        wmrr_delta = cand.gold_summary.weighted_mrr_at_10 \
            - baseline.gold_summary.weighted_mrr_at_10
        deltas[cand.variant] = {
            "primary_score": primary_delta,
            "strict_hit_at_5": strict_h5_delta,
            "weighted_hit_at_5": wh5_delta,
            "weighted_mrr_at_10": wmrr_delta,
            "silver_hit_at_5": (
                cand.silver_summary.hit_at_5 - baseline.silver_summary.hit_at_5
            ),
        }

        if primary_delta < primary_min_delta:
            continue

        has_named_warn = any(
            w.code == "BUCKET_REGRESSION_WARNING" for w in warns
        )
        has_general_warn = any(
            w.code == "SILVER_REGRESSION_WARNING" for w in warns
        )
        if has_general_warn and strict_h5_delta <= 0:
            # silver regressed AND strict didn't improve → reject.
            continue
        if has_named_warn and strict_h5_delta <= 0:
            continue

        survivors.append((cand, primary_delta))

    if survivors:
        # Highest primary_score with weighted_mrr tiebreak.
        survivors.sort(
            key=lambda x: (
                -x[0].gold_summary.primary_score,
                -x[0].gold_summary.weighted_mrr_at_10,
                x[0].variant,
            )
        )
        best = survivors[0][0]
        reason = (
            f"primary_score={best.gold_summary.primary_score:.6f} "
            f"(+{deltas[best.variant]['primary_score']:.6f} vs baseline) "
            f"with no blocking guardrail."
        )
        return ComparisonResult(
            baseline_variant=baseline.variant,
            baseline=baseline,
            candidates=list(candidates),
            guardrails=guardrails,
            best_variant=best.variant,
            best_reason=reason,
            deltas=deltas,
        )

    return ComparisonResult(
        baseline_variant=baseline.variant,
        baseline=baseline,
        candidates=list(candidates),
        guardrails=guardrails,
        best_variant=baseline.variant,
        best_reason=(
            "no candidate cleared the primary_score epsilon AND survived "
            "the silver guardrail; recommendation is to keep baseline."
        ),
        deltas=deltas,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _format_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_comparison_report(comp: ComparisonResult) -> str:
    """Render the human-facing comparison_report.md.

    Always carries :data:`HUMAN_FOCUS_DISCLAIMER` at the top so a
    reviewer can never read the table as a generic benchmark.
    """
    lines: List[str] = []
    lines.append("# Phase 7.x — human-weighted gold seed 50 + silver 500 tuning")
    lines.append("")
    lines.append(f"> {HUMAN_FOCUS_DISCLAIMER}")
    lines.append("")
    lines.append("## Promotion target clarification")
    lines.append("")
    lines.append(PROMOTION_TARGET_FRAMING)
    lines.append("")
    base = comp.baseline.gold_summary
    lines.append(
        f"- baseline variant: **{comp.baseline_variant}**"
    )
    lines.append(
        f"- gold-50 distribution: STRICT={base.n_strict_positive} / "
        f"SOFT={base.n_soft_positive} / "
        f"AMBIGUOUS_PROBE={base.n_ambiguous_probe} / "
        f"ABSTAIN_TEST={base.n_abstain_test}"
    )
    lines.append(
        f"- best candidate: **{comp.best_variant}** — {comp.best_reason}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Headline table
    # ------------------------------------------------------------------
    lines.append("## Headline (gold-50, weighted)")
    lines.append("")
    lines.append("| metric | baseline | " + " | ".join(
        c.variant for c in comp.candidates
    ) + " |")
    lines.append("|---|---:|" + "---:|" * len(comp.candidates))
    metric_keys = (
        ("primary_score", "primary_score"),
        ("weighted_hit@1", "weighted_hit_at_1"),
        ("weighted_hit@3", "weighted_hit_at_3"),
        ("weighted_hit@5", "weighted_hit_at_5"),
        ("weighted_hit@10", "weighted_hit_at_10"),
        ("weighted_MRR@10", "weighted_mrr_at_10"),
        ("weighted_nDCG@10", "weighted_ndcg_at_10"),
        ("strict_hit@5", "strict_hit_at_5"),
        ("strict_MRR@10", "strict_mrr_at_10"),
        ("hit@1 (positive, unweighted)", "hit_at_1"),
        ("hit@5 (positive, unweighted)", "hit_at_5"),
        ("MRR@10 (positive, unweighted)", "mrr_at_10"),
        ("nDCG@10 (positive, unweighted)", "ndcg_at_10"),
    )
    for label, key in metric_keys:
        row = [f"| {label}", f"{getattr(base, key):.4f}"]
        for c in comp.candidates:
            cv = getattr(c.gold_summary, key)
            delta = cv - getattr(base, key)
            row.append(f"{cv:.4f} ({delta:+.4f})")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Auxiliary section / chunk hits
    # ------------------------------------------------------------------
    lines.append("## Auxiliary (section / chunk hits — defined-only mean)")
    lines.append("")
    lines.append("| metric | baseline | " + " | ".join(
        c.variant for c in comp.candidates
    ) + " |")
    lines.append("|---|---:|" + "---:|" * len(comp.candidates))
    aux_keys = (
        ("section_hit@5", "section_hit_at_5_when_defined"),
        ("section_hit@10", "section_hit_at_10_when_defined"),
        ("chunk_hit@10", "chunk_hit_at_10_when_defined"),
    )
    for label, key in aux_keys:
        cells = [f"| {label}", _format_pct(getattr(base, key))]
        for c in comp.candidates:
            cells.append(_format_pct(getattr(c.gold_summary, key)))
        lines.append(" | ".join(cells) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Bucket breakdown (gold)
    # ------------------------------------------------------------------
    lines.append("## Bucket breakdown — gold (weighted_hit@5)")
    lines.append("")
    bucket_keys = sorted({
        *base.by_bucket.keys(),
        *(k for c in comp.candidates for k in c.gold_summary.by_bucket.keys()),
    })
    lines.append(
        "| bucket | n_pos | baseline | "
        + " | ".join(c.variant for c in comp.candidates)
        + " |"
    )
    lines.append("|---|---:|---:|" + "---:|" * len(comp.candidates))
    for bk in bucket_keys:
        b_cell = base.by_bucket.get(bk, {})
        n_pos = int(b_cell.get("n_positive", 0))
        b_v = float(b_cell.get("weighted_hit_at_5", 0.0))
        row = [f"| {bk}", str(n_pos), f"{b_v:.4f}"]
        for c in comp.candidates:
            cell = c.gold_summary.by_bucket.get(bk, {})
            cv = float(cell.get("weighted_hit_at_5", 0.0))
            row.append(f"{cv:.4f} ({cv - b_v:+.4f})")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Query-type breakdown (gold)
    # ------------------------------------------------------------------
    lines.append("## Query-type breakdown — gold (weighted_hit@5)")
    lines.append("")
    qtype_keys = sorted({
        *base.by_query_type.keys(),
        *(k for c in comp.candidates for k in c.gold_summary.by_query_type.keys()),
    })
    lines.append(
        "| query_type | n_pos | baseline | "
        + " | ".join(c.variant for c in comp.candidates)
        + " |"
    )
    lines.append("|---|---:|---:|" + "---:|" * len(comp.candidates))
    for qk in qtype_keys:
        b_cell = base.by_query_type.get(qk, {})
        n_pos = int(b_cell.get("n_positive", 0))
        b_v = float(b_cell.get("weighted_hit_at_5", 0.0))
        row = [f"| {qk}", str(n_pos), f"{b_v:.4f}"]
        for c in comp.candidates:
            cell = c.gold_summary.by_query_type.get(qk, {})
            cv = float(cell.get("weighted_hit_at_5", 0.0))
            row.append(f"{cv:.4f} ({cv - b_v:+.4f})")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Silver guardrail
    # ------------------------------------------------------------------
    lines.append("## Silver guardrail (sanity, NOT primary objective)")
    lines.append("")
    base_sil = comp.baseline.silver_summary
    lines.append("| metric | baseline | " + " | ".join(
        c.variant for c in comp.candidates
    ) + " |")
    lines.append("|---|---:|" + "---:|" * len(comp.candidates))
    for key in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10"):
        bv = getattr(base_sil, key)
        row = [f"| {key}", f"{bv:.4f}"]
        for c in comp.candidates:
            cv = getattr(c.silver_summary, key)
            row.append(f"{cv:.4f} ({cv - bv:+.4f})")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    lines.append("### Silver guardrail warnings")
    lines.append("")
    any_warn = False
    for c in comp.candidates:
        warns = comp.guardrails.get(c.variant) or []
        if not warns:
            continue
        any_warn = True
        lines.append(f"- **{c.variant}**:")
        for w in warns:
            lines.append(f"  - `{w.code}` ({w.metric}, "
                         f"bucket={w.bucket!s}): "
                         f"baseline={w.baseline:.4f} → "
                         f"candidate={w.candidate:.4f} "
                         f"(Δ={w.delta:+.4f}; threshold "
                         f"{w.threshold*100:.1f}pp). {w.message}")
    if not any_warn:
        lines.append("- (none)")
    lines.append("")

    # ------------------------------------------------------------------
    # Recommended next action
    # ------------------------------------------------------------------
    lines.append("## Recommended next action")
    lines.append("")
    if comp.best_variant == comp.baseline_variant:
        lines.append(
            "- **Keep baseline.** No candidate cleared the primary_score "
            "epsilon AND the silver guardrail."
        )
    else:
        best_cand = next(
            c for c in comp.candidates if c.variant == comp.best_variant
        )
        primary_delta = comp.deltas[comp.best_variant]["primary_score"]
        lines.append(
            f"- **Adopt `{comp.best_variant}`** — primary_score "
            f"{best_cand.gold_summary.primary_score:.6f} "
            f"({primary_delta:+.6f} vs baseline)."
        )
        lines.append(
            "- The gain comes from gold-50 weighted hit/MRR/nDCG; verify "
            "the silver guardrail tables above before promoting to "
            "production. The diagnostic only proves we got better at the "
            "subpage / named-subpage failures the gold-50 set was "
            "curated to expose."
        )

    lines.append("")
    lines.append("## Reminders")
    lines.append("")
    lines.append("- AMBIGUOUS_PROBE / ABSTAIN_TEST rows are excluded from "
                 "the primary objective and only reported separately.")
    lines.append("- silver-500 is LLM-generated. Treat its metrics as a "
                 "regression sanity check, NOT a primary target.")
    lines.append("- production retrieval code MUST NOT be changed off "
                 "this report alone — promote via the standard config-"
                 "change PR review.")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_failure_audit_md(
    rows: Sequence[FailureAuditRow], *, header: str,
) -> str:
    """Render a per-query failure audit MD.

    Includes ABSTAIN_TEST rows so the reviewer can sanity-check that
    not-in-corpus queries aren't accidentally getting a confident hit.
    """
    lines: List[str] = []
    lines.append(f"# {header}")
    lines.append("")
    lines.append(f"> {HUMAN_FOCUS_DISCLAIMER}")
    lines.append("")
    lines.append(
        "| query_id | group | wt | bucket | qtype | reason | hit@1/5/10 | "
        "expected | top1 |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|---|")
    for r in rows:
        bucket = r.bucket or "—"
        qtype = r.query_type or "—"
        section_str = " > ".join(r.expected_section_path) \
            if r.expected_section_path else ""
        expected_str = (
            f"{r.expected_title} [{r.expected_page_id[:8]}]"
            if r.expected_page_id else "—"
        )
        if section_str:
            expected_str += f" §{section_str}"
        if r.top1_page_id:
            top1_str = f"{r.top1_title or '?'} [{r.top1_page_id[:8]}]"
        else:
            top1_str = "—"
        lines.append(
            f"| {r.query_id} | {r.normalized_eval_group} | "
            f"{r.eval_weight:.2f} | {bucket} | {qtype} | "
            f"{r.failure_reason} | "
            f"{r.hit_at_1}/{r.hit_at_5}/{r.hit_at_10} | "
            f"{expected_str} | {top1_str} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Serialization helpers (used by the CLI)
# ---------------------------------------------------------------------------


def gold_summary_to_dict(s: GoldSummary) -> Dict[str, Any]:
    return asdict(s)


def silver_summary_to_dict(s: SilverSummary) -> Dict[str, Any]:
    return asdict(s)


def comparison_to_dict(comp: ComparisonResult) -> Dict[str, Any]:
    return {
        "baseline_variant": comp.baseline_variant,
        "best_variant": comp.best_variant,
        "best_reason": comp.best_reason,
        "deltas": comp.deltas,
        "guardrails": {
            v: [w.to_dict() for w in ws]
            for v, ws in comp.guardrails.items()
        },
        "baseline": {
            "variant": comp.baseline.variant,
            "config": comp.baseline.config,
            "gold_summary": gold_summary_to_dict(comp.baseline.gold_summary),
            "silver_summary": silver_summary_to_dict(
                comp.baseline.silver_summary,
            ),
        },
        "candidates": [
            {
                "variant": c.variant,
                "config": c.config,
                "gold_summary": gold_summary_to_dict(c.gold_summary),
                "silver_summary": silver_summary_to_dict(c.silver_summary),
            }
            for c in comp.candidates
        ],
        "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
    }


__all__ = [
    "GOLD_REQUIRED_COLUMNS",
    "GROUP_STRICT_POSITIVE",
    "GROUP_SOFT_POSITIVE",
    "GROUP_AMBIGUOUS_PROBE",
    "GROUP_ABSTAIN_TEST",
    "NORMALIZED_EVAL_GROUPS",
    "POSITIVE_GROUPS",
    "HUMAN_FOCUS_DISCLAIMER",
    "PROMOTION_TARGET_FRAMING",
    "PRIMARY_WEIGHTED_HIT_AT_5",
    "PRIMARY_WEIGHTED_MRR_AT_10",
    "PRIMARY_WEIGHTED_NDCG_AT_10",
    "SILVER_HIT_AT_5_REGRESSION_THRESHOLD",
    "SILVER_BUCKET_REGRESSION_THRESHOLD",
    "SILVER_BUCKET_FOR_NAMED_GUARDRAIL",
    "DatasetValidationIssue",
    "GoldSeedValidationError",
    "GoldRow",
    "GoldSeedDataset",
    "SilverRow",
    "SilverDataset",
    "RetrievedDoc",
    "QueryRetrieval",
    "GoldQueryEvalRow",
    "SilverQueryEvalRow",
    "GoldSummary",
    "SilverSummary",
    "GuardrailWarning",
    "VariantResult",
    "ComparisonResult",
    "FailureAuditRow",
    "FAIL_TITLE_MISS",
    "FAIL_SECTION_MISS",
    "FAIL_SUBPAGE_MISS",
    "FAIL_NAMED_SUBPAGE_MISS",
    "FAIL_OVER_BROAD_QUERY",
    "FAIL_WRONG_SERIES",
    "FAIL_WRONG_SEASON",
    "FAIL_NOT_IN_CORPUS_CASE",
    "FAIL_UNKNOWN",
    "normalize_eval_group",
    "load_human_gold_seed_50",
    "load_llm_silver_500",
    "evaluate_gold",
    "evaluate_silver",
    "summarize_gold",
    "summarize_silver",
    "evaluate_silver_guardrail",
    "classify_failure",
    "build_failure_audit_row",
    "compare_variants",
    "primary_score",
    "render_comparison_report",
    "render_failure_audit_md",
    "gold_summary_to_dict",
    "silver_summary_to_dict",
    "comparison_to_dict",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "first_hit_rank",
    "section_hit_at_k",
    "chunk_hit_at_k",
]
