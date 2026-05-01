"""Phase 7.7 — answerability audit harness (with Phase 7.7.1 bundle track).

Phase 7.5 promoted a production retrieval config; Phase 7.6 explores
section-aware reranking on top of it. Both phases score against
``page_hit@k`` and ``section_hit@k`` metrics, which are *retrieval-side*
proxies for grounding quality. They cannot tell you whether the
returned chunks actually contain enough evidence to answer the query.

Phase 7.7 adds a parallel evaluation track that reads the retrieved
chunks and asks a *human* (initially — LLM judges are deliberately
out of scope here) whether the top-k context is sufficient to
produce a faithful answer. The harness is purely the bookkeeping
layer for that audit; it covers both:

  * **row-level** — one labelled record per ``(query, variant, rank)``,
    asking *does this single chunk carry the evidence?*. Used for
    evidence-quality diagnosis and per-chunk failure attribution.
  * **bundle-level (Phase 7.7.1)** — one labelled record per
    ``(query, variant, top_k)`` that wraps the rendered top-k chunks
    as a single context block, asking *does the context as a whole
    answer the query?*. This is the metric closer to
    ``answerable@k``'s natural meaning, and it is the primary signal
    for cross-chunk synthesis cases (``needs_cross_section`` /
    ``needs_subpage``) where row-level aggregation underestimates
    answerability.

For each track the four operations are the same — only the row
shape differs:

  1. **Export.** Take retrieval results and emit a CSV / JSONL with
     three blank label columns for a human reviewer to fill in.
     Row track: one row per ``(query, variant, rank)``.
     Bundle track: one row per ``(query, variant, top_k)`` with the
     top-k chunks pre-rendered into ``context_bundle_text``.
  2. **Import / validate.** Parse the labelled file back, normalising
     int / enum-name labels, validating flag membership, detecting
     duplicate primary-key rows, and surfacing empty text cells.
  3. **Score.** Compute ``answerable@k`` / ``fully_answerable@k`` (row
     track) plus ``context_answerable@k`` / ``context_fully@k`` (bundle
     track) plus the page-hit-vs-answerability and
     section-miss-vs-answerability cross tabs.
  4. **Report.** Render a Markdown summary that shows both tracks
     side-by-side when bundle labels exist, with a multi-chunk
     answerability caveat and a fixed interpretation guide so
     reviewers don't read these numbers as a hit@k replacement.

Sampling helper (Phase 7.7.1) — ``sample_bundle_records`` /
``--mode bundle-sample`` lets a reviewer take an unlabelled bundle
export and pull N distinct query_ids deterministically (via seeded
RNG) for pilot review. The helper does NOT fill in labels; that is
strictly a human step.

Design contract:

  * **Phase 7.6 is upstream**, not downstream. The harness ingests
    the (query, top-k) frozen pairs Phase 7.6 prepares and treats
    them as opaque retrieval result records. Promotion decisions
    stay with Phase 7.6 / 7.5; this module only *audits*.
  * **Production config is unchanged.** This is an evaluation harness
    only — there is no retriever code, no reranker code, no FAISS
    interaction. All inputs are pre-computed.
  * **No LLM judge.** ``label_answerability`` /
    ``label_context_answerability`` are filled in by a human; the
    schema permits a future LLM-judge integration but the column
    semantics ("graded by a human reviewer") must not silently drift
    if a judge is later wired up.
  * **Not a hit@k replacement.** Answerability metrics are deliberately
    *additive*. The interpretation block in every rendered report
    reminds the reader of this.
  * **Row vs bundle are complementary.** Row metrics → evidence-quality
    diagnosis. Bundle metrics → top-k answerability judgment. The
    primary number for ``answerable@k``-style questions is
    bundle-level; row-level is for failure attribution.
"""

from __future__ import annotations

import csv
import json
import logging
import random as _random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label enum + flag enum
# ---------------------------------------------------------------------------


class AnswerabilityLabel(IntEnum):
    """How well does this retrieved chunk support answering the query.

    The four levels are deliberately ordinal — ``>= PARTIALLY_ANSWERABLE``
    is the cutoff for "answerable" in the metric definitions. Reviewers
    should pick the lowest level that still applies; "right page but
    only one fact useful" is ``PARTIALLY_ANSWERABLE``, not ``FULLY_``.
    """

    NOT_RELEVANT = 0
    RELATED_BUT_NOT_ANSWERABLE = 1
    PARTIALLY_ANSWERABLE = 2
    FULLY_ANSWERABLE = 3


class AnswerabilityFlag(str, Enum):
    """Optional reviewer-supplied tags on a labelled row.

    A row can carry zero or more flags. They serialise as a ``|``-joined
    string in CSV (matches the existing ``io_utils._csv_cell`` convention)
    and as a JSON list in JSONL. Membership is closed: anything that is
    not one of these values is rejected at import time so reviewers
    can't silently invent new flags that the report renderer won't know
    how to bucket.
    """

    WRONG_PAGE = "wrong_page"
    RIGHT_PAGE_WRONG_SECTION = "right_page_wrong_section"
    EVIDENCE_TOO_NOISY = "evidence_too_noisy"
    NEEDS_CROSS_SECTION = "needs_cross_section"
    NEEDS_SUBPAGE = "needs_subpage"
    AMBIGUOUS_QUERY = "ambiguous_query"


_LABEL_BY_NAME: Dict[str, AnswerabilityLabel] = {
    label.name: label for label in AnswerabilityLabel
}
_LABEL_BY_VALUE: Dict[int, AnswerabilityLabel] = {
    label.value: label for label in AnswerabilityLabel
}
_FLAG_BY_VALUE: Dict[str, AnswerabilityFlag] = {
    flag.value: flag for flag in AnswerabilityFlag
}


# Threshold used by ``answerable@k``. Pinned by the report's
# interpretation guide; a reviewer-driven change must update both.
ANSWERABLE_MIN_LEVEL: AnswerabilityLabel = (
    AnswerabilityLabel.PARTIALLY_ANSWERABLE
)


# CSV column order — reproduced verbatim by the export writer and
# enforced by the importer. Renaming any of these breaks every previously
# labelled file, so changes must roll forward every existing CSV.
EXPORT_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "query",
    "gold_page_id",
    "gold_page_title",
    "gold_section_id",
    "gold_section_path",
    "variant_name",
    "rank",
    "retrieved_page_id",
    "retrieved_page_title",
    "retrieved_section_id",
    "retrieved_section_path",
    "chunk_id",
    "chunk_text",
    "page_hit",
    "section_hit",
    "label_answerability",
    "flags",
    "notes",
)


# Standard variant labels used across the report. The fixture set here
# is deliberately small — Phase 7.7 is a label-collection harness, not
# a sweep harness — and the report renderer falls back gracefully when
# a labelled file uses different names.
VARIANT_BASELINE: str = "baseline"
VARIANT_PRODUCTION_RECOMMENDED: str = "production_recommended"
VARIANT_SECTION_AWARE_CANDIDATE: str = "section_aware_candidate"


# ---------------------------------------------------------------------------
# Source records — deliberately tiny dataclasses so the test suite can
# build them by hand without any retrieval-side dependency.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldRef:
    """The gold answer's page/section coordinates for one query.

    Empty strings are valid for any of the four fields — a query with
    no gold section path simply skips section-hit accounting. Matching
    is exact on ``page_id`` and prefix-based on ``section_path``.
    """

    page_id: str = ""
    page_title: str = ""
    section_id: str = ""
    section_path: str = ""


@dataclass(frozen=True)
class RetrievedRef:
    """One ranked retrieval result for a query.

    ``chunk_text`` is the raw text the reviewer reads when assigning
    a label. The exporter copies it through unchanged; the validator
    surfaces empty cells as a labelling error.
    """

    rank: int
    chunk_id: str = ""
    page_id: str = ""
    page_title: str = ""
    section_id: str = ""
    section_path: str = ""
    chunk_text: str = ""


# ---------------------------------------------------------------------------
# Hit helpers — simple, pure, mirror the existing harness conventions.
# ---------------------------------------------------------------------------


def compute_page_hit(gold_page_id: str, retrieved_page_id: str) -> bool:
    """Page-level exact-match hit. Empty gold ⇒ False."""
    return bool(gold_page_id) and gold_page_id == retrieved_page_id


def compute_section_hit(
    gold_page_id: str,
    gold_section_path: str,
    retrieved_page_id: str,
    retrieved_section_path: str,
) -> bool:
    """Section-level descendant hit, gated on a page hit.

    Mirrors ``phase7_human_gold_tune.section_hit_at_k``: requires a
    page-id match first, then ``retrieved_section_path`` must equal the
    gold section or be a child path separated by ``" > "``. Empty gold
    section ⇒ False (section matching is undefined in that case —
    callers that want "always return True when undefined" should not
    use this helper).
    """
    if not compute_page_hit(gold_page_id, retrieved_page_id):
        return False
    if not gold_section_path:
        return False
    return _section_path_matches_gold(gold_section_path, retrieved_section_path)


def _section_path_matches_gold(gold_section_path: str, retrieved_section_path: str) -> bool:
    gold = gold_section_path.strip()
    retrieved = retrieved_section_path.strip()
    return bool(gold) and (
        retrieved == gold or retrieved.startswith(f"{gold} > ")
    )


# ---------------------------------------------------------------------------
# Export rows — what the CSV writer / JSONL writer consume.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerabilityExportRow:
    """One CSV row in the labelling file *before* a reviewer touches it.

    ``label_answerability``, ``flags``, ``notes`` are intentionally
    absent from this dataclass — the writer emits empty cells so the
    reviewer can fill them in. The labelled-row dataclass is the
    post-review counterpart.
    """

    query_id: str
    query: str
    variant_name: str
    rank: int
    gold: GoldRef
    retrieved: RetrievedRef
    page_hit: bool
    section_hit: bool

    def to_csv_dict(self) -> Dict[str, Any]:
        """Materialise the CSV-shaped dict (with blank label fields).

        Keys match :data:`EXPORT_COLUMNS` exactly.
        """
        return {
            "query_id": self.query_id,
            "query": self.query,
            "gold_page_id": self.gold.page_id,
            "gold_page_title": self.gold.page_title,
            "gold_section_id": self.gold.section_id,
            "gold_section_path": self.gold.section_path,
            "variant_name": self.variant_name,
            "rank": self.rank,
            "retrieved_page_id": self.retrieved.page_id,
            "retrieved_page_title": self.retrieved.page_title,
            "retrieved_section_id": self.retrieved.section_id,
            "retrieved_section_path": self.retrieved.section_path,
            "chunk_id": self.retrieved.chunk_id,
            "chunk_text": self.retrieved.chunk_text,
            "page_hit": "true" if self.page_hit else "false",
            "section_hit": "true" if self.section_hit else "false",
            "label_answerability": "",
            "flags": "",
            "notes": "",
        }


def build_export_rows(
    *,
    query_id: str,
    query: str,
    variant_name: str,
    gold: GoldRef,
    retrieved: Sequence[RetrievedRef],
    top_k: int = 5,
) -> List[AnswerabilityExportRow]:
    """Build one export row per top-k retrieved chunk for a single query.

    Ranks come from the ``RetrievedRef.rank`` field, not the slice
    index — the export trusts the upstream rank ordering. Truncates
    to ``top_k`` items so the labelling burden stays bounded; pass
    ``top_k=0`` for "no truncation".
    """
    rows: List[AnswerabilityExportRow] = []
    if top_k and top_k > 0:
        slice_ = list(retrieved)[:top_k]
    else:
        slice_ = list(retrieved)
    for ref in slice_:
        page_hit = compute_page_hit(gold.page_id, ref.page_id)
        section_hit = compute_section_hit(
            gold.page_id, gold.section_path,
            ref.page_id, ref.section_path,
        )
        rows.append(
            AnswerabilityExportRow(
                query_id=str(query_id),
                query=str(query),
                variant_name=str(variant_name),
                rank=int(ref.rank),
                gold=gold,
                retrieved=ref,
                page_hit=page_hit,
                section_hit=section_hit,
            )
        )
    return rows


def write_export_csv(
    path: Path, rows: Iterable[AnswerabilityExportRow],
) -> Path:
    """Emit rows to a CSV with exactly :data:`EXPORT_COLUMNS` headers.

    Parent directory is auto-created. Re-runs overwrite. The label
    columns (`label_answerability`, `flags`, `notes`) are written as
    empty strings — that's the contract with the reviewer.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=list(EXPORT_COLUMNS), extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())
    log.info("Wrote answerability export CSV: %s", out_path)
    return out_path


def write_export_jsonl(
    path: Path, rows: Iterable[AnswerabilityExportRow],
) -> Path:
    """Emit rows to a JSONL with the same column contract as the CSV.

    Same parent-directory + overwrite behaviour as ``write_export_csv``.
    JSONL is provided alongside CSV because some reviewers prefer a
    text-editor pass over a spreadsheet pass; either file round-trips
    through the importer.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(
                json.dumps(row.to_csv_dict(), ensure_ascii=False)
            )
            fp.write("\n")
    log.info("Wrote answerability export JSONL: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Import + validation
# ---------------------------------------------------------------------------


class AnswerabilityValidationError(ValueError):
    """Raised when a labelled file violates the schema contract.

    Single source of truth for parse-time errors so CLI / harness
    callers can ``except AnswerabilityValidationError`` rather than
    catching generic ``ValueError`` and risking false positives from
    JSON parsing or downstream consumers.
    """


@dataclass(frozen=True)
class AnswerabilityLabeledRow:
    """One labelled row, ready for scoring.

    Constructed exclusively by :func:`parse_labeled_row` so the label
    / flags / hit fields are guaranteed to be canonical types. The
    scorer should not need to defensively re-coerce anything.
    """

    query_id: str
    query: str
    variant_name: str
    rank: int
    gold_page_id: str
    gold_page_title: str
    gold_section_id: str
    gold_section_path: str
    retrieved_page_id: str
    retrieved_page_title: str
    retrieved_section_id: str
    retrieved_section_path: str
    chunk_id: str
    chunk_text: str
    page_hit: bool
    section_hit: bool
    label: AnswerabilityLabel
    flags: Tuple[AnswerabilityFlag, ...]
    notes: str = ""


def parse_label(value: Any) -> AnswerabilityLabel:
    """Coerce an int / string into an :class:`AnswerabilityLabel`.

    Accepts:
      * the IntEnum value itself
      * an int 0..3
      * a stringified int "0".."3"
      * the enum name ``"NOT_RELEVANT"`` etc. (case-insensitive)

    Anything else (including float, ``None``, empty string, unknown
    name) raises :class:`AnswerabilityValidationError`.
    """
    if isinstance(value, AnswerabilityLabel):
        return value
    if isinstance(value, bool):
        # bool is an int subclass — guard explicitly so True/False
        # don't silently coerce to NOT_RELEVANT / RELATED_BUT_*.
        raise AnswerabilityValidationError(
            f"label_answerability cannot be a bool: {value!r}"
        )
    if isinstance(value, int):
        if value in _LABEL_BY_VALUE:
            return _LABEL_BY_VALUE[value]
        raise AnswerabilityValidationError(
            f"label_answerability int {value} is not in {{0,1,2,3}}"
        )
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise AnswerabilityValidationError(
                "label_answerability cannot be empty"
            )
        # Try int first (e.g. "0", "  3  ").
        try:
            ivalue = int(s)
        except ValueError:
            ivalue = None
        if ivalue is not None:
            if ivalue in _LABEL_BY_VALUE:
                return _LABEL_BY_VALUE[ivalue]
            raise AnswerabilityValidationError(
                f"label_answerability int {ivalue} is not in {{0,1,2,3}}"
            )
        upper = s.upper()
        if upper in _LABEL_BY_NAME:
            return _LABEL_BY_NAME[upper]
        raise AnswerabilityValidationError(
            f"Unknown label_answerability name: {value!r}. "
            f"Allowed: {sorted(_LABEL_BY_NAME)}"
        )
    raise AnswerabilityValidationError(
        f"label_answerability has unsupported type: {type(value).__name__}"
    )


def parse_flags(value: Any) -> Tuple[AnswerabilityFlag, ...]:
    """Coerce a CSV cell / JSON array / Python iterable into flag tuple.

    CSV cells use the ``|`` separator (matches ``io_utils._csv_cell``);
    JSONL exports may store ``["wrong_page","ambiguous_query"]`` as a
    list. Empty / whitespace cells return an empty tuple. Unknown flag
    values raise :class:`AnswerabilityValidationError`.
    """
    if value is None:
        return ()
    if isinstance(value, AnswerabilityFlag):
        return (value,)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ()
        parts = [p.strip() for p in s.split("|") if p.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(p).strip() for p in value if str(p).strip()]
    else:
        raise AnswerabilityValidationError(
            f"flags has unsupported type: {type(value).__name__}"
        )
    out: List[AnswerabilityFlag] = []
    seen: set = set()
    for part in parts:
        key = part.lower()
        if key not in _FLAG_BY_VALUE:
            raise AnswerabilityValidationError(
                f"Unknown flag {part!r}. Allowed: "
                f"{sorted(_FLAG_BY_VALUE)}"
            )
        flag = _FLAG_BY_VALUE[key]
        if flag in seen:
            continue
        seen.add(flag)
        out.append(flag)
    return tuple(out)


def _parse_bool(value: Any, *, field_name: str) -> bool:
    """Coerce CSV bool cells. Accepts true/false (case-insensitive),
    1/0, yes/no, and Python booleans. Empty string is treated as False
    so a partially-labelled file can still pass validation.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return False
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        raise AnswerabilityValidationError(
            f"{field_name} expected bool-like, got {value!r}"
        )
    raise AnswerabilityValidationError(
        f"{field_name} has unsupported type: {type(value).__name__}"
    )


def _require_field(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise AnswerabilityValidationError(
            f"Missing required column: {name!r}"
        )
    return record[name]


def parse_labeled_row(record: Mapping[str, Any]) -> AnswerabilityLabeledRow:
    """Parse one record into an :class:`AnswerabilityLabeledRow`.

    Validates required columns, coerces label / flags / hit columns to
    canonical types, and surfaces empty ``chunk_text`` cells (the most
    common labelling slip — copying a spreadsheet row but forgetting to
    paste the chunk back in).
    """
    # Required columns — anything missing here is a structural problem
    # with the file, not with one row's labelling.
    required = (
        "query_id", "query", "variant_name", "rank",
        "retrieved_page_id", "chunk_id", "chunk_text",
        "label_answerability",
    )
    for name in required:
        _require_field(record, name)

    chunk_text = str(record.get("chunk_text", "") or "")
    if not chunk_text.strip():
        raise AnswerabilityValidationError(
            f"chunk_text is empty for "
            f"query_id={record.get('query_id')!r} "
            f"variant_name={record.get('variant_name')!r} "
            f"rank={record.get('rank')!r}"
        )

    try:
        rank = int(record.get("rank"))
    except (TypeError, ValueError) as ex:
        raise AnswerabilityValidationError(
            f"rank must be an int, got {record.get('rank')!r}"
        ) from ex

    label = parse_label(record["label_answerability"])
    flags = parse_flags(record.get("flags"))
    page_hit = _parse_bool(record.get("page_hit"), field_name="page_hit")
    section_hit = _parse_bool(
        record.get("section_hit"), field_name="section_hit",
    )

    return AnswerabilityLabeledRow(
        query_id=str(record["query_id"]),
        query=str(record.get("query", "") or ""),
        variant_name=str(record["variant_name"]),
        rank=rank,
        gold_page_id=str(record.get("gold_page_id", "") or ""),
        gold_page_title=str(record.get("gold_page_title", "") or ""),
        gold_section_id=str(record.get("gold_section_id", "") or ""),
        gold_section_path=str(record.get("gold_section_path", "") or ""),
        retrieved_page_id=str(record["retrieved_page_id"]),
        retrieved_page_title=str(
            record.get("retrieved_page_title", "") or "",
        ),
        retrieved_section_id=str(
            record.get("retrieved_section_id", "") or "",
        ),
        retrieved_section_path=str(
            record.get("retrieved_section_path", "") or "",
        ),
        chunk_id=str(record["chunk_id"]),
        chunk_text=chunk_text,
        page_hit=page_hit,
        section_hit=section_hit,
        label=label,
        flags=flags,
        notes=str(record.get("notes", "") or ""),
    )


def parse_labeled_rows(
    records: Iterable[Mapping[str, Any]],
) -> List[AnswerabilityLabeledRow]:
    """Parse + validate a stream of records, surfacing duplicates.

    Two records sharing the same ``(query_id, variant_name, rank)``
    triple raise :class:`AnswerabilityValidationError` — that combination
    is the row's primary key, and a duplicate almost always means the
    reviewer pasted the same row twice or merged two labelling sessions
    without dedup.
    """
    out: List[AnswerabilityLabeledRow] = []
    seen: Dict[Tuple[str, str, int], int] = {}
    for index, record in enumerate(records, start=1):
        row = parse_labeled_row(record)
        key = (row.query_id, row.variant_name, row.rank)
        if key in seen:
            raise AnswerabilityValidationError(
                f"Duplicate labelled row at record #{index}: "
                f"query_id={row.query_id!r} "
                f"variant_name={row.variant_name!r} "
                f"rank={row.rank} "
                f"(first seen at record #{seen[key]})"
            )
        seen[key] = index
        out.append(row)
    return out


def read_labeled_csv(path: Path) -> List[AnswerabilityLabeledRow]:
    """Read a labelled CSV and return validated rows.

    Errors include the file path so the reviewer can fix the right
    file when running multiple labelling sessions.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Labelled file not found: {src}"
        )
    with src.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        try:
            return parse_labeled_rows(reader)
        except AnswerabilityValidationError as ex:
            raise AnswerabilityValidationError(
                f"{src}: {ex}"
            ) from ex


def read_labeled_jsonl(path: Path) -> List[AnswerabilityLabeledRow]:
    """Read a labelled JSONL and return validated rows.

    Same contract as :func:`read_labeled_csv`. Blank lines are
    skipped (matches ``io_utils.load_jsonl``); ``flags`` may be a JSON
    list or a ``|``-joined string.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Labelled file not found: {src}"
        )
    records: List[Mapping[str, Any]] = []
    with src.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise AnswerabilityValidationError(
                    f"{src}: invalid JSON on line {line_no}: {ex}"
                ) from ex
            if not isinstance(obj, dict):
                raise AnswerabilityValidationError(
                    f"{src}: line {line_no} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    try:
        return parse_labeled_rows(records)
    except AnswerabilityValidationError as ex:
        raise AnswerabilityValidationError(
            f"{src}: {ex}"
        ) from ex


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantMetrics:
    """Per-variant rollup of the answerability metrics.

    All ``*_at_k`` rates are query-level fractions over queries that
    have at least one row at rank ≤ k. Flag counts are row-level over
    rows at rank ≤ 5. The four query-level confusion counts are also
    bounded to rank ≤ 5.
    """

    variant_name: str
    n_queries: int
    n_rows: int
    answerable_at_1: float
    answerable_at_3: float
    answerable_at_5: float
    fully_answerable_at_1: float
    fully_answerable_at_3: float
    fully_answerable_at_5: float
    partial_or_better_at_5: float
    page_hit_but_not_answerable_count: int
    section_miss_but_answerable_count: int
    wrong_page_count: int
    right_page_wrong_section_count: int
    evidence_too_noisy_count: int
    needs_cross_section_count: int
    needs_subpage_count: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _group_by_variant(
    rows: Sequence[AnswerabilityLabeledRow],
) -> Dict[str, List[AnswerabilityLabeledRow]]:
    by_variant: Dict[str, List[AnswerabilityLabeledRow]] = defaultdict(list)
    for row in rows:
        by_variant[row.variant_name].append(row)
    return dict(by_variant)


def _group_by_query(
    rows: Sequence[AnswerabilityLabeledRow],
) -> Dict[str, List[AnswerabilityLabeledRow]]:
    by_query: Dict[str, List[AnswerabilityLabeledRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_id].append(row)
    for qid, lst in by_query.items():
        lst.sort(key=lambda r: r.rank)
    return dict(by_query)


def _rate_at_k(
    by_query: Mapping[str, Sequence[AnswerabilityLabeledRow]],
    *,
    k: int,
    min_level: AnswerabilityLabel,
) -> float:
    """Fraction of queries with at least one rank-≤k row at >= min_level.

    Queries with zero rows at rank ≤ k are still counted in the
    denominator — answering "no top-k answerable" with "skip the row"
    would silently inflate the rate. Returns 0.0 for an empty input.
    """
    if not by_query:
        return 0.0
    n_total = len(by_query)
    n_hit = 0
    for rows in by_query.values():
        for row in rows:
            if row.rank <= k and int(row.label) >= int(min_level):
                n_hit += 1
                break
    return n_hit / n_total


def _query_has_rank_at_or_below(
    rows: Sequence[AnswerabilityLabeledRow], k: int,
) -> bool:
    return any(row.rank <= k for row in rows)


def _query_any_page_hit_at_or_below(
    rows: Sequence[AnswerabilityLabeledRow], k: int,
) -> bool:
    return any(row.rank <= k and row.page_hit for row in rows)


def _query_any_section_hit_at_or_below(
    rows: Sequence[AnswerabilityLabeledRow], k: int,
) -> bool:
    return any(row.rank <= k and row.section_hit for row in rows)


def _query_any_label_at_or_above(
    rows: Sequence[AnswerabilityLabeledRow],
    *,
    k: int,
    min_level: AnswerabilityLabel,
) -> bool:
    return any(
        row.rank <= k and int(row.label) >= int(min_level)
        for row in rows
    )


def compute_variant_metrics(
    rows: Sequence[AnswerabilityLabeledRow], variant_name: str,
) -> VariantMetrics:
    """Compute per-variant metrics over the rows belonging to one variant.

    The caller should pass only rows where ``variant_name`` matches
    ``variant_name``; the function does *not* filter — it only stamps
    the variant name onto the result.
    """
    rows = list(rows)
    by_query = _group_by_query(rows)

    n_queries = len(by_query)
    n_rows = len(rows)

    answerable_at_1 = _rate_at_k(
        by_query, k=1, min_level=ANSWERABLE_MIN_LEVEL,
    )
    answerable_at_3 = _rate_at_k(
        by_query, k=3, min_level=ANSWERABLE_MIN_LEVEL,
    )
    answerable_at_5 = _rate_at_k(
        by_query, k=5, min_level=ANSWERABLE_MIN_LEVEL,
    )
    fully_answerable_at_1 = _rate_at_k(
        by_query, k=1, min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )
    fully_answerable_at_3 = _rate_at_k(
        by_query, k=3, min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )
    fully_answerable_at_5 = _rate_at_k(
        by_query, k=5, min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )
    # ``partial_or_better_at_5`` is a deliberate alias of
    # ``answerable_at_5`` — the threshold IS PARTIALLY_ANSWERABLE.
    # The spec lists both columns so reviewers can read either name
    # without having to know which one happens to be the alias.
    partial_or_better_at_5 = answerable_at_5

    # Query-level confusion counts:
    #   - page-hit-but-not-answerable: page hit somewhere in top-5 but
    #     the labelled answerability never reaches PARTIALLY_ANSWERABLE
    #   - section-miss-but-answerable: no section hit in top-5 yet the
    #     labelled answerability does reach PARTIALLY_ANSWERABLE
    page_hit_but_not_answerable = 0
    section_miss_but_answerable = 0
    for qid, qrows in by_query.items():
        any_page_hit_5 = _query_any_page_hit_at_or_below(qrows, 5)
        any_section_hit_5 = _query_any_section_hit_at_or_below(qrows, 5)
        any_answerable_5 = _query_any_label_at_or_above(
            qrows, k=5, min_level=ANSWERABLE_MIN_LEVEL,
        )
        if any_page_hit_5 and not any_answerable_5:
            page_hit_but_not_answerable += 1
        if any_answerable_5 and not any_section_hit_5:
            section_miss_but_answerable += 1

    # Flag counts: row-level over rows at rank ≤ 5.
    flag_counter: Counter = Counter()
    for row in rows:
        if row.rank > 5:
            continue
        for flag in row.flags:
            flag_counter[flag] += 1

    return VariantMetrics(
        variant_name=variant_name,
        n_queries=n_queries,
        n_rows=n_rows,
        answerable_at_1=answerable_at_1,
        answerable_at_3=answerable_at_3,
        answerable_at_5=answerable_at_5,
        fully_answerable_at_1=fully_answerable_at_1,
        fully_answerable_at_3=fully_answerable_at_3,
        fully_answerable_at_5=fully_answerable_at_5,
        partial_or_better_at_5=partial_or_better_at_5,
        page_hit_but_not_answerable_count=page_hit_but_not_answerable,
        section_miss_but_answerable_count=section_miss_but_answerable,
        wrong_page_count=int(flag_counter.get(
            AnswerabilityFlag.WRONG_PAGE, 0,
        )),
        right_page_wrong_section_count=int(flag_counter.get(
            AnswerabilityFlag.RIGHT_PAGE_WRONG_SECTION, 0,
        )),
        evidence_too_noisy_count=int(flag_counter.get(
            AnswerabilityFlag.EVIDENCE_TOO_NOISY, 0,
        )),
        needs_cross_section_count=int(flag_counter.get(
            AnswerabilityFlag.NEEDS_CROSS_SECTION, 0,
        )),
        needs_subpage_count=int(flag_counter.get(
            AnswerabilityFlag.NEEDS_SUBPAGE, 0,
        )),
    )


def compute_all_variants(
    rows: Sequence[AnswerabilityLabeledRow],
) -> List[VariantMetrics]:
    """Group rows by ``variant_name`` and compute per-variant metrics.

    The output order follows :data:`PREFERRED_VARIANT_ORDER` — known
    variants come in the canonical order, anything else is sorted
    alphabetically afterwards. Stable order keeps successive report
    runs diff-friendly.
    """
    by_variant = _group_by_variant(rows)
    out: List[VariantMetrics] = []
    seen: set = set()
    for name in PREFERRED_VARIANT_ORDER:
        if name in by_variant:
            out.append(compute_variant_metrics(by_variant[name], name))
            seen.add(name)
    for name in sorted(by_variant):
        if name in seen:
            continue
        out.append(compute_variant_metrics(by_variant[name], name))
    return out


PREFERRED_VARIANT_ORDER: Tuple[str, ...] = (
    VARIANT_BASELINE,
    VARIANT_PRODUCTION_RECOMMENDED,
    VARIANT_SECTION_AWARE_CANDIDATE,
)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


_INTERPRETATION_GUIDE: Tuple[str, ...] = (
    "Answerability scores are a *human-graded* signal of whether the "
    "retrieved top-k actually supports answering the query.",
    "They are **additive** to ``hit@k`` / ``section_hit@k``, not a "
    "replacement. A retrieval that hits the right page but fails "
    "answerability is a context-quality problem; a retrieval that "
    "answers despite missing the section is a section-metric brittleness "
    "signal.",
    "``page_hit_but_not_answerable`` counts queries where the right "
    "page is in the top-5 but the chunks do not actually carry the "
    "evidence the query asks for. High values typically mean section "
    "selection or chunking is too coarse.",
    "``section_miss_but_answerable`` counts queries where the gold "
    "section path is not in the top-5 but the retrieved chunks still "
    "answer the query. High values typically mean the gold section "
    "annotation is brittle or that cross-section synthesis is happening.",
    "Initial labels are collected from a human reviewer — LLM judges are "
    "intentionally out of scope for this phase.",
    "These metrics do NOT measure generated-answer quality. That belongs "
    "to a separate evaluation phase.",
)


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _label_distribution(
    rows: Sequence[AnswerabilityLabeledRow],
) -> Dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        counter[row.label.name] += 1
    out: Dict[str, int] = {}
    for label in AnswerabilityLabel:
        out[label.name] = int(counter.get(label.name, 0))
    return out


def _confusion_block(
    rows: Sequence[AnswerabilityLabeledRow],
    *,
    by_query: Mapping[str, Sequence[AnswerabilityLabeledRow]],
    hit_predicate,
    hit_name: str,
) -> Dict[str, int]:
    """Build a 2x2 confusion table query-level, hit vs answerable.

    ``hit_predicate(qrows) -> bool`` returns True when the query has at
    least one row at rank ≤ 5 satisfying the hit predicate; we reuse it
    for both page-hit and section-hit. Cells are query counts.
    """
    counts = {
        f"{hit_name}_and_answerable": 0,
        f"{hit_name}_and_not_answerable": 0,
        f"no_{hit_name}_and_answerable": 0,
        f"no_{hit_name}_and_not_answerable": 0,
    }
    for qid, qrows in by_query.items():
        had_hit = hit_predicate(qrows)
        was_ans = _query_any_label_at_or_above(
            qrows, k=5, min_level=ANSWERABLE_MIN_LEVEL,
        )
        if had_hit and was_ans:
            counts[f"{hit_name}_and_answerable"] += 1
        elif had_hit and not was_ans:
            counts[f"{hit_name}_and_not_answerable"] += 1
        elif not had_hit and was_ans:
            counts[f"no_{hit_name}_and_answerable"] += 1
        else:
            counts[f"no_{hit_name}_and_not_answerable"] += 1
    return counts


def _render_label_distribution(
    rows: Sequence[AnswerabilityLabeledRow],
) -> str:
    dist = _label_distribution(rows)
    total = sum(dist.values())
    lines = ["| label | count | share |", "|---|---:|---:|"]
    for label in AnswerabilityLabel:
        n = dist[label.name]
        share = (n / total) if total else 0.0
        lines.append(f"| {label.name} | {n} | {share:.4f} |")
    lines.append(f"| **total** | **{total}** | 1.0000 |")
    return "\n".join(lines)


def _render_variant_metrics_table(
    variants: Sequence[VariantMetrics],
) -> str:
    if not variants:
        return "_(no variants — empty input)_"
    header = (
        "| variant | n_queries | n_rows "
        "| answerable@1 | answerable@3 | answerable@5 "
        "| fully@1 | fully@3 | fully@5 "
        "| partial+@5 |"
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header, sep]
    for v in variants:
        rows.append(
            "| "
            f"{v.variant_name} | {v.n_queries} | {v.n_rows} | "
            f"{_format_rate(v.answerable_at_1)} | "
            f"{_format_rate(v.answerable_at_3)} | "
            f"{_format_rate(v.answerable_at_5)} | "
            f"{_format_rate(v.fully_answerable_at_1)} | "
            f"{_format_rate(v.fully_answerable_at_3)} | "
            f"{_format_rate(v.fully_answerable_at_5)} | "
            f"{_format_rate(v.partial_or_better_at_5)} |"
        )
    return "\n".join(rows)


def _render_confusion_section(
    rows: Sequence[AnswerabilityLabeledRow], *, hit_name: str,
) -> str:
    by_variant = _group_by_variant(rows)
    if not by_variant:
        return "_(no variants — empty input)_"
    if hit_name == "page_hit":
        predicate = lambda qr: _query_any_page_hit_at_or_below(qr, 5)
    elif hit_name == "section_hit":
        predicate = lambda qr: _query_any_section_hit_at_or_below(qr, 5)
    else:  # pragma: no cover — guarded against typos in caller
        raise ValueError(f"Unknown hit_name: {hit_name!r}")

    header = (
        f"| variant | {hit_name} ∧ answerable | "
        f"{hit_name} ∧ ¬answerable | ¬{hit_name} ∧ answerable | "
        f"¬{hit_name} ∧ ¬answerable |"
    )
    sep = "|---|---:|---:|---:|---:|"
    out = [header, sep]
    for name in _ordered_variant_names(by_variant):
        qrows = _group_by_query(by_variant[name])
        cells = _confusion_block(
            by_variant[name],
            by_query=qrows,
            hit_predicate=predicate,
            hit_name=hit_name,
        )
        out.append(
            "| "
            f"{name} | "
            f"{cells[f'{hit_name}_and_answerable']} | "
            f"{cells[f'{hit_name}_and_not_answerable']} | "
            f"{cells[f'no_{hit_name}_and_answerable']} | "
            f"{cells[f'no_{hit_name}_and_not_answerable']} |"
        )
    return "\n".join(out)


def _ordered_variant_names(
    by_variant: Mapping[str, Any],
) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for name in PREFERRED_VARIANT_ORDER:
        if name in by_variant:
            out.append(name)
            seen.add(name)
    for name in sorted(by_variant):
        if name in seen:
            continue
        out.append(name)
    return out


def _render_failure_buckets(
    variants: Sequence[VariantMetrics],
) -> str:
    if not variants:
        return "_(no variants — empty input)_"
    header = (
        "| variant "
        "| page_hit∧¬answerable_q | section_miss∧answerable_q "
        "| wrong_page | right_page_wrong_section "
        "| evidence_too_noisy | needs_cross_section | needs_subpage |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for v in variants:
        rows.append(
            "| "
            f"{v.variant_name} | "
            f"{v.page_hit_but_not_answerable_count} | "
            f"{v.section_miss_but_answerable_count} | "
            f"{v.wrong_page_count} | "
            f"{v.right_page_wrong_section_count} | "
            f"{v.evidence_too_noisy_count} | "
            f"{v.needs_cross_section_count} | "
            f"{v.needs_subpage_count} |"
        )
    return "\n".join(rows)


def _top_failure_examples(
    rows: Sequence[AnswerabilityLabeledRow], *, limit: int = 10,
) -> List[AnswerabilityLabeledRow]:
    """Pick top-rank rows that look like failures, capped at ``limit``.

    A failure is any row at rank ≤ 5 with label < PARTIALLY_ANSWERABLE
    where the page DID hit (i.e. the retriever found the right page
    but the chunk does not actually answer). Sort by variant, then
    query_id, then rank for diff stability.
    """
    out: List[AnswerabilityLabeledRow] = []
    for row in rows:
        if row.rank > 5:
            continue
        if int(row.label) >= int(ANSWERABLE_MIN_LEVEL):
            continue
        if not row.page_hit:
            continue
        out.append(row)
    out.sort(key=lambda r: (r.variant_name, r.query_id, r.rank))
    return out[:limit]


def _render_top_failure_examples(
    rows: Sequence[AnswerabilityLabeledRow], *, limit: int = 10,
) -> str:
    examples = _top_failure_examples(rows, limit=limit)
    if not examples:
        return "_(no page-hit failure rows in top-5)_"
    header = (
        "| variant | query_id | rank | retrieved_page_id | "
        "retrieved_section_path | label | flags |"
    )
    sep = "|---|---|---:|---|---|---|---|"
    out = [header, sep]
    for row in examples:
        flags = "|".join(f.value for f in row.flags)
        out.append(
            "| "
            f"{row.variant_name} | "
            f"{row.query_id} | "
            f"{row.rank} | "
            f"{row.retrieved_page_id} | "
            f"{row.retrieved_section_path} | "
            f"{row.label.name} | "
            f"{flags} |"
        )
    return "\n".join(out)


def _next_action_recommendation(
    variants: Sequence[VariantMetrics],
) -> str:
    """Boilerplate next-step pointer.

    Deliberately conservative — Phase 7.7 is a label-collection harness,
    not a winner-selection harness. The recommendation only points at
    the work that should follow once labels exist; it does not promote
    a variant.
    """
    if not variants:
        return (
            "_No variants in the labelled set — collect labels first, "
            "then re-run the scorer._"
        )
    bullets = [
        "Treat the per-variant metrics above as **descriptive**, "
        "not prescriptive — Phase 7.7 does not promote a config.",
        "If ``page_hit_but_not_answerable_count`` is high relative to "
        "``page_hit@5``, the next experiment should be a chunking / "
        "section-aware rerank exercise (Phase 7.6 candidate strategies).",
        "If ``section_miss_but_answerable_count`` is high, the gold "
        "``expected_section_path`` annotations are likely too strict — "
        "auditing the gold-50 section labels is the next step.",
        "Repeat the audit after Phase 7.6 lands a section-aware "
        "candidate to compare answerability deltas before promoting.",
    ]
    return "\n".join(f"- {b}" for b in bullets)


def render_markdown_report(
    rows: Sequence[AnswerabilityLabeledRow],
    *,
    title: str = "Phase 7.7 — answerability audit",
    bundle_rows: Optional[Sequence["ContextBundleAuditRow"]] = None,
) -> str:
    """Render the full Markdown report from a labelled-row set.

    Sections are pinned (the test suite checks each header is present)
    so a future report consumer can rely on the structure. Empty
    inputs render an explicit "no data" report rather than crashing.

    When ``bundle_rows`` is non-empty, four additional sections are
    inserted between "Top failure examples" and "Interpretation guide":

      * Context bundle answerability — bundle-level metric table
      * Row evidence vs context answerability — row vs bundle Δ
      * Multi-chunk answerability caveat — guidance text
      * Recommended labeling workflow — pilot workflow steps

    The report stays single-track when ``bundle_rows`` is None / empty
    so existing Phase 7.7 row-only callers keep working unchanged.
    """
    bundle_rows_list: List["ContextBundleAuditRow"] = list(bundle_rows or [])

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "> Human-graded answerability audit. Labels are collected by "
        "a human reviewer; LLM judges are explicitly out of scope. "
        "Answerability is **additive** to ``hit@k`` and ``section_hit@k`` "
        "— it does not replace either."
    )
    lines.append("")

    variants = compute_all_variants(rows)

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- n_rows: **{len(rows)}**")
    lines.append(f"- n_variants: **{len(variants)}**")
    n_queries = len({r.query_id for r in rows})
    lines.append(f"- n_queries: **{n_queries}**")
    flagged_rows = sum(1 for r in rows if r.flags)
    lines.append(f"- n_rows_with_flags: **{flagged_rows}**")
    lines.append("")

    # Label distribution
    lines.append("## Label distribution")
    lines.append("")
    lines.append(_render_label_distribution(rows))
    lines.append("")

    # Per-variant metrics
    lines.append("## Answerability metrics by variant")
    lines.append("")
    lines.append(_render_variant_metrics_table(variants))
    lines.append("")

    # Confusion blocks
    lines.append("## Page hit vs answerability confusion")
    lines.append("")
    lines.append(
        "Query-level counts at top-5. ``page_hit ∧ ¬answerable`` is "
        "the cell that quantifies the gap between retrieval-side "
        "metrics and downstream answerability."
    )
    lines.append("")
    lines.append(_render_confusion_section(rows, hit_name="page_hit"))
    lines.append("")

    lines.append("## Section hit vs answerability confusion")
    lines.append("")
    lines.append(
        "Query-level counts at top-5. ``¬section_hit ∧ answerable`` "
        "is the cell that flags brittle ``expected_section_path`` "
        "annotations or successful cross-section synthesis."
    )
    lines.append("")
    lines.append(_render_confusion_section(rows, hit_name="section_hit"))
    lines.append("")

    # Failure buckets
    lines.append("## Failure buckets")
    lines.append("")
    lines.append(_render_failure_buckets(variants))
    lines.append("")

    # Top failure examples
    lines.append("## Top failure examples")
    lines.append("")
    lines.append(
        "Up to 10 rows where the page hit at top-5 but the labelled "
        "answerability did not reach ``PARTIALLY_ANSWERABLE``. Sorted "
        "by ``(variant, query_id, rank)`` for diff stability."
    )
    lines.append("")
    lines.append(_render_top_failure_examples(rows))
    lines.append("")

    # Phase 7.7.1 bundle sections (only when bundle labels exist).
    if bundle_rows_list:
        lines.extend(_render_bundle_sections(rows, bundle_rows_list))

    # Interpretation guide
    lines.append("## Interpretation guide")
    lines.append("")
    for guide in _INTERPRETATION_GUIDE:
        lines.append(f"- {guide}")
    lines.append("")

    # Next action recommendation
    lines.append("## Next action recommendation")
    lines.append("")
    lines.append(_next_action_recommendation(variants))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON summary writer — used by the CLI when --json-path is set.
# ---------------------------------------------------------------------------


def build_json_summary(
    rows: Sequence[AnswerabilityLabeledRow],
    *,
    bundle_rows: Optional[Sequence["ContextBundleAuditRow"]] = None,
) -> Dict[str, Any]:
    """Build the structured JSON sidecar matching the Markdown report.

    Keys mirror the report's section order: ``summary`` /
    ``label_distribution`` / ``variants`` / ``page_hit_confusion`` /
    ``section_hit_confusion``. When ``bundle_rows`` is non-empty,
    four extra keys are added: ``bundle_summary`` /
    ``bundle_label_distribution`` / ``bundle_variants`` /
    ``row_vs_bundle_at_5``. Reviewers run downstream analysis off
    this file; the Markdown report is for human reading.
    """
    variants = compute_all_variants(rows)
    by_variant = _group_by_variant(rows)
    page_confusion: Dict[str, Dict[str, int]] = {}
    section_confusion: Dict[str, Dict[str, int]] = {}
    for name in _ordered_variant_names(by_variant):
        qrows = _group_by_query(by_variant[name])
        page_confusion[name] = _confusion_block(
            by_variant[name], by_query=qrows,
            hit_predicate=lambda qr: _query_any_page_hit_at_or_below(qr, 5),
            hit_name="page_hit",
        )
        section_confusion[name] = _confusion_block(
            by_variant[name], by_query=qrows,
            hit_predicate=lambda qr: _query_any_section_hit_at_or_below(qr, 5),
            hit_name="section_hit",
        )
    out: Dict[str, Any] = {
        "summary": {
            "n_rows": len(rows),
            "n_variants": len(variants),
            "n_queries": len({r.query_id for r in rows}),
            "n_rows_with_flags": sum(1 for r in rows if r.flags),
        },
        "label_distribution": _label_distribution(rows),
        "variants": [v.as_dict() for v in variants],
        "page_hit_confusion": page_confusion,
        "section_hit_confusion": section_confusion,
    }

    bundle_rows_list: List["ContextBundleAuditRow"] = list(bundle_rows or [])
    if bundle_rows_list:
        bundle_variants = compute_all_bundle_variants(bundle_rows_list)
        out["bundle_summary"] = {
            "n_rows": len(bundle_rows_list),
            "n_variants": len(bundle_variants),
            "n_queries": len({r.query_id for r in bundle_rows_list}),
            "n_rows_with_flags": sum(
                1 for r in bundle_rows_list if r.flags
            ),
            "top_k_set": sorted(
                {r.top_k for r in bundle_rows_list}
            ),
        }
        out["bundle_label_distribution"] = (
            _bundle_label_distribution(bundle_rows_list)
        )
        out["bundle_variants"] = [v.as_dict() for v in bundle_variants]
        out["row_vs_bundle_at_5"] = _row_vs_bundle_at_5_dict(
            variants, bundle_variants,
        )
    return out


# ===========================================================================
# Phase 7.7.1 — bundle-level (context-set) audit
# ===========================================================================
#
# Row-level audit answers "does *this single chunk* carry the evidence?".
# Bundle-level audit answers "does the *top-k context as a whole* answer
# the query?". The two metrics are not redundant: cross-section synthesis
# cases (``needs_cross_section`` / ``needs_subpage``) and multi-chunk
# evidence cases under-report on row-level aggregation alone, because no
# single chunk carries the full answer even when the bundle does.
#
# Bundle rows share the same label / flag enums as row-level audit but
# have a distinct schema, distinct primary key
# ``(query_id, variant_name, top_k)``, and a pre-rendered
# ``context_bundle_text`` cell that the reviewer reads in lieu of
# eyeballing the per-chunk CSV row-by-row.


# ---------------------------------------------------------------------------
# Bundle-level constants + schema
# ---------------------------------------------------------------------------


# Default top_k slices a single (query, variant) gets exported at. The
# reviewer can collapse to a single k (e.g. just k=5) during pilot if
# the labelling burden is too high; the harness imposes no restriction
# beyond "positive ints only".
DEFAULT_TOP_K_SET: Tuple[int, ...] = (1, 3, 5)

# Default per-chunk truncation budget for ``context_bundle_text``.
# 1200 chars ≈ ~250 Korean tokens — long enough to label confidently,
# short enough to keep a 5-chunk bundle under ~6000 chars (one screen).
DEFAULT_BUNDLE_TRUNCATE_CHARS: int = 1200


# CSV / JSONL column order for bundle exports. Pinned by the writer +
# importer; renaming any of these breaks every previously-labelled
# bundle file.
BUNDLE_EXPORT_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "query",
    "gold_page_id",
    "gold_page_title",
    "gold_section_id",
    "gold_section_path",
    "variant_name",
    "top_k",
    "context_bundle_text",
    "retrieved_page_ids",
    "retrieved_page_titles",
    "retrieved_section_ids",
    "retrieved_section_paths",
    "chunk_ids",
    "page_hit_within_k",
    "section_hit_within_k",
    "label_context_answerability",
    "context_flags",
    "notes",
)


# ---------------------------------------------------------------------------
# Bundle data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBundleRef:
    """One context bundle for a (query, variant, top_k) cell.

    Holds the top-k ``RetrievedRef`` slice that fed the
    ``context_bundle_text`` rendering. Kept as a separate dataclass so
    the export builder can compute hit predicates without re-deriving
    the slice each time.
    """

    top_k: int
    retrieved: Tuple[RetrievedRef, ...]


@dataclass(frozen=True)
class ContextBundleExportRow:
    """Bundle-level CSV / JSONL row *before* a reviewer touches it.

    The label columns (``label_context_answerability``,
    ``context_flags``, ``notes``) are intentionally absent — the
    writer emits them as empty cells so the reviewer fills them in.
    The labelled-row counterpart is :class:`ContextBundleAuditRow`.
    """

    query_id: str
    query: str
    variant_name: str
    top_k: int
    gold: GoldRef
    retrieved: Tuple[RetrievedRef, ...]
    context_bundle_text: str
    page_hit_within_k: bool
    section_hit_within_k: bool

    def to_csv_dict(self) -> Dict[str, Any]:
        """Materialise the CSV-shaped dict (with blank label fields).

        List-valued fields are joined with ``|`` (matches the existing
        ``io_utils._csv_cell`` convention). Keys match
        :data:`BUNDLE_EXPORT_COLUMNS` exactly.
        """
        return {
            "query_id": self.query_id,
            "query": self.query,
            "gold_page_id": self.gold.page_id,
            "gold_page_title": self.gold.page_title,
            "gold_section_id": self.gold.section_id,
            "gold_section_path": self.gold.section_path,
            "variant_name": self.variant_name,
            "top_k": self.top_k,
            "context_bundle_text": self.context_bundle_text,
            "retrieved_page_ids": "|".join(
                r.page_id for r in self.retrieved
            ),
            "retrieved_page_titles": "|".join(
                r.page_title for r in self.retrieved
            ),
            "retrieved_section_ids": "|".join(
                r.section_id for r in self.retrieved
            ),
            "retrieved_section_paths": "|".join(
                r.section_path for r in self.retrieved
            ),
            "chunk_ids": "|".join(
                r.chunk_id for r in self.retrieved
            ),
            "page_hit_within_k": (
                "true" if self.page_hit_within_k else "false"
            ),
            "section_hit_within_k": (
                "true" if self.section_hit_within_k else "false"
            ),
            "label_context_answerability": "",
            "context_flags": "",
            "notes": "",
        }


@dataclass(frozen=True)
class ContextBundleAuditRow:
    """Labelled bundle row, ready for scoring.

    Constructed exclusively by :func:`parse_bundle_labeled_row` so the
    label / flags / hit fields are guaranteed canonical types. The
    list-valued retrieval fields are tuples of strings — empty tuple
    when the export had no chunks (which would be unusual but is not
    an error per se; the validator only rejects empty
    ``context_bundle_text``).
    """

    query_id: str
    query: str
    variant_name: str
    top_k: int
    gold_page_id: str
    gold_page_title: str
    gold_section_id: str
    gold_section_path: str
    retrieved_page_ids: Tuple[str, ...]
    retrieved_page_titles: Tuple[str, ...]
    retrieved_section_ids: Tuple[str, ...]
    retrieved_section_paths: Tuple[str, ...]
    chunk_ids: Tuple[str, ...]
    context_bundle_text: str
    page_hit_within_k: bool
    section_hit_within_k: bool
    label: AnswerabilityLabel
    flags: Tuple[AnswerabilityFlag, ...]
    notes: str = ""


# ---------------------------------------------------------------------------
# Bundle hit helpers + text rendering
# ---------------------------------------------------------------------------


def compute_bundle_page_hit(
    gold_page_id: str, retrieved: Sequence[RetrievedRef],
) -> bool:
    """Any chunk in the bundle has page_id == gold_page_id."""
    if not gold_page_id:
        return False
    return any(r.page_id == gold_page_id for r in retrieved)


def compute_bundle_section_hit(
    gold_page_id: str,
    gold_section_path: str,
    retrieved: Sequence[RetrievedRef],
) -> bool:
    """Any chunk is a page hit AND its section_path matches gold/child."""
    if not gold_page_id or not gold_section_path:
        return False
    for r in retrieved:
        if (
            r.page_id == gold_page_id
            and _section_path_matches_gold(gold_section_path, r.section_path)
        ):
            return True
    return False


def render_bundle_text(
    retrieved: Sequence[RetrievedRef],
    *,
    truncate_chars: Optional[int] = DEFAULT_BUNDLE_TRUNCATE_CHARS,
) -> str:
    """Render top-k chunks as a labeller-friendly multi-section text.

    Layout::

        [Rank 1]
        page: <page_title> (<page_id>)
        section: <section_path>
        chunk_id: <chunk_id>
        text:
        <chunk_text — possibly truncated>
        ...[truncated]   <- only when truncate_chars cut the chunk

        [Rank 2]
        ...

    ``truncate_chars`` set to None or 0 disables truncation. Empty
    ``retrieved`` returns the empty string.
    """
    if not retrieved:
        return ""
    blocks: List[str] = []
    cap: Optional[int] = (
        int(truncate_chars)
        if truncate_chars is not None and int(truncate_chars) > 0
        else None
    )
    for ref in retrieved:
        text = ref.chunk_text or ""
        truncated = False
        if cap is not None and len(text) > cap:
            text = text[:cap]
            truncated = True
        block_lines: List[str] = [
            f"[Rank {ref.rank}]",
            f"page: {ref.page_title} ({ref.page_id})",
            f"section: {ref.section_path}",
            f"chunk_id: {ref.chunk_id}",
            "text:",
            text,
        ]
        if truncated:
            block_lines.append("...[truncated]")
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Bundle export builder + writers
# ---------------------------------------------------------------------------


def build_bundle_export_rows(
    *,
    query_id: str,
    query: str,
    variant_name: str,
    gold: GoldRef,
    retrieved: Sequence[RetrievedRef],
    top_k_set: Sequence[int] = DEFAULT_TOP_K_SET,
    truncate_chars: Optional[int] = DEFAULT_BUNDLE_TRUNCATE_CHARS,
) -> List[ContextBundleExportRow]:
    """Build one bundle row per top_k for a single (query, variant).

    ``retrieved`` is sorted by ``rank`` before slicing so callers don't
    need to pre-sort. Each top_k value yields one row whose
    ``retrieved`` field holds the first k chunks. ``top_k`` values
    must be positive ints; non-positive values raise ``ValueError``
    so a typo at the CLI surface fails loudly rather than silently
    producing zero-length bundles.
    """
    if not top_k_set:
        return []
    sorted_ks = sorted({int(k) for k in top_k_set})
    for k in sorted_ks:
        if k <= 0:
            raise ValueError(
                f"top_k must be positive, got {k} in {tuple(top_k_set)!r}"
            )
    sorted_retrieved = sorted(
        retrieved, key=lambda r: int(r.rank),
    )
    rows: List[ContextBundleExportRow] = []
    for k in sorted_ks:
        slice_ = tuple(sorted_retrieved[:k])
        text = render_bundle_text(slice_, truncate_chars=truncate_chars)
        page_hit = compute_bundle_page_hit(gold.page_id, slice_)
        section_hit = compute_bundle_section_hit(
            gold.page_id, gold.section_path, slice_,
        )
        rows.append(
            ContextBundleExportRow(
                query_id=str(query_id),
                query=str(query),
                variant_name=str(variant_name),
                top_k=int(k),
                gold=gold,
                retrieved=slice_,
                context_bundle_text=text,
                page_hit_within_k=page_hit,
                section_hit_within_k=section_hit,
            )
        )
    return rows


def write_bundle_export_csv(
    path: Path, rows: Iterable[ContextBundleExportRow],
) -> Path:
    """Emit bundle rows to a CSV with :data:`BUNDLE_EXPORT_COLUMNS`.

    Parent directory is auto-created. Re-runs overwrite. The label
    columns are written as empty strings — that is the contract with
    the reviewer.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(BUNDLE_EXPORT_COLUMNS),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())
    log.info("Wrote bundle answerability export CSV: %s", out_path)
    return out_path


def write_bundle_export_jsonl(
    path: Path, rows: Iterable[ContextBundleExportRow],
) -> Path:
    """Emit bundle rows to a JSONL with the same column contract."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(
                json.dumps(row.to_csv_dict(), ensure_ascii=False)
            )
            fp.write("\n")
    log.info("Wrote bundle answerability export JSONL: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Bundle import + validation
# ---------------------------------------------------------------------------


def _parse_string_list(
    value: Any, *, field_name: str,
) -> Tuple[str, ...]:
    """Coerce a CSV cell / JSON list / Python iterable into a string tuple.

    CSV cells use ``|`` separator (matches the writer); empty / None
    yields an empty tuple. Unknown types raise
    :class:`AnswerabilityValidationError`.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ()
        return tuple(p for p in s.split("|"))
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise AnswerabilityValidationError(
        f"{field_name} has unsupported type: {type(value).__name__}"
    )


def parse_bundle_labeled_row(
    record: Mapping[str, Any],
) -> ContextBundleAuditRow:
    """Parse one labelled bundle record into :class:`ContextBundleAuditRow`.

    Validates required columns, coerces label / flags / hit / top_k
    columns to canonical types, and surfaces empty
    ``context_bundle_text`` cells (the bundle analogue of the row-level
    "missing chunk_text" check).
    """
    required = (
        "query_id", "query", "variant_name", "top_k",
        "context_bundle_text", "label_context_answerability",
    )
    for name in required:
        _require_field(record, name)

    text = str(record.get("context_bundle_text", "") or "")
    if not text.strip():
        raise AnswerabilityValidationError(
            f"context_bundle_text is empty for "
            f"query_id={record.get('query_id')!r} "
            f"variant_name={record.get('variant_name')!r} "
            f"top_k={record.get('top_k')!r}"
        )

    raw_top_k = record["top_k"]
    if isinstance(raw_top_k, bool):
        # ``bool`` is an int subclass; reject it explicitly so True/False
        # do not silently masquerade as top_k=1/0.
        raise AnswerabilityValidationError(
            f"top_k cannot be a bool: {raw_top_k!r}"
        )
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError) as ex:
        raise AnswerabilityValidationError(
            f"top_k must be an int, got {raw_top_k!r}"
        ) from ex
    if top_k <= 0:
        raise AnswerabilityValidationError(
            f"top_k must be positive, got {top_k}"
        )

    label = parse_label(record["label_context_answerability"])
    flags = parse_flags(record.get("context_flags"))
    page_hit = _parse_bool(
        record.get("page_hit_within_k"),
        field_name="page_hit_within_k",
    )
    section_hit = _parse_bool(
        record.get("section_hit_within_k"),
        field_name="section_hit_within_k",
    )

    return ContextBundleAuditRow(
        query_id=str(record["query_id"]),
        query=str(record.get("query", "") or ""),
        variant_name=str(record["variant_name"]),
        top_k=top_k,
        gold_page_id=str(record.get("gold_page_id", "") or ""),
        gold_page_title=str(record.get("gold_page_title", "") or ""),
        gold_section_id=str(record.get("gold_section_id", "") or ""),
        gold_section_path=str(record.get("gold_section_path", "") or ""),
        retrieved_page_ids=_parse_string_list(
            record.get("retrieved_page_ids"),
            field_name="retrieved_page_ids",
        ),
        retrieved_page_titles=_parse_string_list(
            record.get("retrieved_page_titles"),
            field_name="retrieved_page_titles",
        ),
        retrieved_section_ids=_parse_string_list(
            record.get("retrieved_section_ids"),
            field_name="retrieved_section_ids",
        ),
        retrieved_section_paths=_parse_string_list(
            record.get("retrieved_section_paths"),
            field_name="retrieved_section_paths",
        ),
        chunk_ids=_parse_string_list(
            record.get("chunk_ids"), field_name="chunk_ids",
        ),
        context_bundle_text=text,
        page_hit_within_k=page_hit,
        section_hit_within_k=section_hit,
        label=label,
        flags=flags,
        notes=str(record.get("notes", "") or ""),
    )


def parse_bundle_labeled_rows(
    records: Iterable[Mapping[str, Any]],
) -> List[ContextBundleAuditRow]:
    """Parse + validate a stream of bundle records, surfacing duplicates.

    Bundle primary key is ``(query_id, variant_name, top_k)``. A
    duplicate raises :class:`AnswerabilityValidationError` for the
    same reasons as the row-level duplicate check.
    """
    out: List[ContextBundleAuditRow] = []
    seen: Dict[Tuple[str, str, int], int] = {}
    for index, record in enumerate(records, start=1):
        row = parse_bundle_labeled_row(record)
        key = (row.query_id, row.variant_name, row.top_k)
        if key in seen:
            raise AnswerabilityValidationError(
                f"Duplicate bundle row at record #{index}: "
                f"query_id={row.query_id!r} "
                f"variant_name={row.variant_name!r} "
                f"top_k={row.top_k} "
                f"(first seen at record #{seen[key]})"
            )
        seen[key] = index
        out.append(row)
    return out


def read_bundle_labeled_csv(path: Path) -> List[ContextBundleAuditRow]:
    """Read a labelled bundle CSV and return validated rows."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Labelled bundle file not found: {src}"
        )
    with src.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        try:
            return parse_bundle_labeled_rows(reader)
        except AnswerabilityValidationError as ex:
            raise AnswerabilityValidationError(
                f"{src}: {ex}"
            ) from ex


def read_bundle_labeled_jsonl(
    path: Path,
) -> List[ContextBundleAuditRow]:
    """Read a labelled bundle JSONL and return validated rows."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Labelled bundle file not found: {src}"
        )
    records: List[Mapping[str, Any]] = []
    with src.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise AnswerabilityValidationError(
                    f"{src}: invalid JSON on line {line_no}: {ex}"
                ) from ex
            if not isinstance(obj, dict):
                raise AnswerabilityValidationError(
                    f"{src}: line {line_no} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    try:
        return parse_bundle_labeled_rows(records)
    except AnswerabilityValidationError as ex:
        raise AnswerabilityValidationError(f"{src}: {ex}") from ex


# ---------------------------------------------------------------------------
# Bundle scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleAnswerabilityMetrics:
    """Per-variant rollup of the bundle-level answerability metrics.

    All ``context_*_at_k`` rates are query-level fractions: for each
    distinct ``query_id``, look up the bundle row at ``top_k=k`` and
    check the label. A query without a top_k=k row contributes 0 to
    the numerator and 1 to the denominator (matches the row-level
    "no top-k row means no hit" convention).

    Confusion + flag counts are scoped to ``top_k=5`` rows to mirror
    the row-level metric stack's "@5 is the headline" convention.
    """

    variant_name: str
    n_queries: int
    n_rows: int
    context_answerable_at_1: float
    context_answerable_at_3: float
    context_answerable_at_5: float
    context_fully_answerable_at_1: float
    context_fully_answerable_at_3: float
    context_fully_answerable_at_5: float
    page_hit_but_context_not_answerable_at_5: int
    section_miss_but_context_answerable_at_5: int
    context_needs_cross_section_count: int
    context_needs_subpage_count: int
    context_evidence_too_noisy_count: int
    context_wrong_page_count: int
    context_right_page_wrong_section_count: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _group_bundle_by_query(
    rows: Sequence[ContextBundleAuditRow],
) -> Dict[str, List[ContextBundleAuditRow]]:
    by_query: Dict[str, List[ContextBundleAuditRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_id].append(row)
    for qid, lst in by_query.items():
        lst.sort(key=lambda r: r.top_k)
    return dict(by_query)


def _group_bundle_by_variant(
    rows: Sequence[ContextBundleAuditRow],
) -> Dict[str, List[ContextBundleAuditRow]]:
    by_variant: Dict[str, List[ContextBundleAuditRow]] = defaultdict(list)
    for row in rows:
        by_variant[row.variant_name].append(row)
    return dict(by_variant)


def _bundle_rate_at_k(
    by_query: Mapping[str, Sequence[ContextBundleAuditRow]],
    *,
    k: int,
    min_level: AnswerabilityLabel,
) -> float:
    """Fraction of queries whose top_k=k row's label is >= min_level.

    Queries without a top_k=k row contribute 0 to the numerator.
    Returns 0.0 for an empty input.
    """
    if not by_query:
        return 0.0
    n_total = len(by_query)
    n_hit = 0
    for rows in by_query.values():
        for row in rows:
            if row.top_k == k and int(row.label) >= int(min_level):
                n_hit += 1
                break
    return n_hit / n_total


def compute_bundle_variant_metrics(
    rows: Sequence[ContextBundleAuditRow], variant_name: str,
) -> BundleAnswerabilityMetrics:
    """Compute per-variant bundle metrics over rows for one variant.

    The caller should pass only rows where ``variant_name`` matches
    ``variant_name`` (mirrors :func:`compute_variant_metrics`); the
    function does not filter, only stamps the variant name.
    """
    rows = list(rows)
    by_query = _group_bundle_by_query(rows)

    n_queries = len(by_query)
    n_rows = len(rows)

    context_answerable_at_1 = _bundle_rate_at_k(
        by_query, k=1, min_level=ANSWERABLE_MIN_LEVEL,
    )
    context_answerable_at_3 = _bundle_rate_at_k(
        by_query, k=3, min_level=ANSWERABLE_MIN_LEVEL,
    )
    context_answerable_at_5 = _bundle_rate_at_k(
        by_query, k=5, min_level=ANSWERABLE_MIN_LEVEL,
    )
    context_fully_answerable_at_1 = _bundle_rate_at_k(
        by_query, k=1,
        min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )
    context_fully_answerable_at_3 = _bundle_rate_at_k(
        by_query, k=3,
        min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )
    context_fully_answerable_at_5 = _bundle_rate_at_k(
        by_query, k=5,
        min_level=AnswerabilityLabel.FULLY_ANSWERABLE,
    )

    # Confusion counts at top_k=5: one bundle row per query.
    page_hit_but_not = 0
    section_miss_but = 0
    for qrows in by_query.values():
        row5: Optional[ContextBundleAuditRow] = None
        for r in qrows:
            if r.top_k == 5:
                row5 = r
                break
        if row5 is None:
            continue
        is_answerable = (
            int(row5.label) >= int(ANSWERABLE_MIN_LEVEL)
        )
        if row5.page_hit_within_k and not is_answerable:
            page_hit_but_not += 1
        if is_answerable and not row5.section_hit_within_k:
            section_miss_but += 1

    # Flag counts at top_k=5.
    flag_counter: Counter = Counter()
    for row in rows:
        if row.top_k != 5:
            continue
        for flag in row.flags:
            flag_counter[flag] += 1

    return BundleAnswerabilityMetrics(
        variant_name=variant_name,
        n_queries=n_queries,
        n_rows=n_rows,
        context_answerable_at_1=context_answerable_at_1,
        context_answerable_at_3=context_answerable_at_3,
        context_answerable_at_5=context_answerable_at_5,
        context_fully_answerable_at_1=context_fully_answerable_at_1,
        context_fully_answerable_at_3=context_fully_answerable_at_3,
        context_fully_answerable_at_5=context_fully_answerable_at_5,
        page_hit_but_context_not_answerable_at_5=page_hit_but_not,
        section_miss_but_context_answerable_at_5=section_miss_but,
        context_needs_cross_section_count=int(flag_counter.get(
            AnswerabilityFlag.NEEDS_CROSS_SECTION, 0,
        )),
        context_needs_subpage_count=int(flag_counter.get(
            AnswerabilityFlag.NEEDS_SUBPAGE, 0,
        )),
        context_evidence_too_noisy_count=int(flag_counter.get(
            AnswerabilityFlag.EVIDENCE_TOO_NOISY, 0,
        )),
        context_wrong_page_count=int(flag_counter.get(
            AnswerabilityFlag.WRONG_PAGE, 0,
        )),
        context_right_page_wrong_section_count=int(flag_counter.get(
            AnswerabilityFlag.RIGHT_PAGE_WRONG_SECTION, 0,
        )),
    )


def compute_all_bundle_variants(
    rows: Sequence[ContextBundleAuditRow],
) -> List[BundleAnswerabilityMetrics]:
    """Group bundle rows by variant_name and compute per-variant metrics.

    Output order follows :data:`PREFERRED_VARIANT_ORDER`, then
    alphabetical for unknown variants — matches the row-level
    :func:`compute_all_variants` so reports diff cleanly.
    """
    by_variant = _group_bundle_by_variant(rows)
    out: List[BundleAnswerabilityMetrics] = []
    seen: set = set()
    for name in PREFERRED_VARIANT_ORDER:
        if name in by_variant:
            out.append(
                compute_bundle_variant_metrics(by_variant[name], name)
            )
            seen.add(name)
    for name in sorted(by_variant):
        if name in seen:
            continue
        out.append(
            compute_bundle_variant_metrics(by_variant[name], name)
        )
    return out


# ---------------------------------------------------------------------------
# Bundle render helpers (called from render_markdown_report when
# ``bundle_rows`` is non-empty).
# ---------------------------------------------------------------------------


def _bundle_label_distribution(
    rows: Sequence[ContextBundleAuditRow],
) -> Dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        counter[row.label.name] += 1
    out: Dict[str, int] = {}
    for label in AnswerabilityLabel:
        out[label.name] = int(counter.get(label.name, 0))
    return out


def _render_bundle_metrics_table(
    variants: Sequence[BundleAnswerabilityMetrics],
) -> str:
    if not variants:
        return "_(no variants — empty bundle input)_"
    header = (
        "| variant | n_queries | n_rows "
        "| ctx_ans@1 | ctx_ans@3 | ctx_ans@5 "
        "| ctx_fully@1 | ctx_fully@3 | ctx_fully@5 |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for v in variants:
        rows.append(
            "| "
            f"{v.variant_name} | {v.n_queries} | {v.n_rows} | "
            f"{_format_rate(v.context_answerable_at_1)} | "
            f"{_format_rate(v.context_answerable_at_3)} | "
            f"{_format_rate(v.context_answerable_at_5)} | "
            f"{_format_rate(v.context_fully_answerable_at_1)} | "
            f"{_format_rate(v.context_fully_answerable_at_3)} | "
            f"{_format_rate(v.context_fully_answerable_at_5)} |"
        )
    return "\n".join(rows)


def _render_bundle_failure_buckets(
    variants: Sequence[BundleAnswerabilityMetrics],
) -> str:
    if not variants:
        return "_(no variants — empty bundle input)_"
    header = (
        "| variant "
        "| page_hit∧¬ctx_ans@5 | section_miss∧ctx_ans@5 "
        "| ctx_wrong_page | ctx_right_page_wrong_section "
        "| ctx_evidence_too_noisy | ctx_needs_cross_section "
        "| ctx_needs_subpage |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for v in variants:
        rows.append(
            "| "
            f"{v.variant_name} | "
            f"{v.page_hit_but_context_not_answerable_at_5} | "
            f"{v.section_miss_but_context_answerable_at_5} | "
            f"{v.context_wrong_page_count} | "
            f"{v.context_right_page_wrong_section_count} | "
            f"{v.context_evidence_too_noisy_count} | "
            f"{v.context_needs_cross_section_count} | "
            f"{v.context_needs_subpage_count} |"
        )
    return "\n".join(rows)


def _row_vs_bundle_at_5_pairs(
    row_variants: Sequence[VariantMetrics],
    bundle_variants: Sequence[BundleAnswerabilityMetrics],
) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Side-by-side (row_answerable@5, context_answerable@5) per variant.

    Returns ``(variant_name, row@5, ctx@5)`` where either rate is None
    when that track has no labels for the variant. Order follows
    :data:`PREFERRED_VARIANT_ORDER` then alphabetical.
    """
    rv = {v.variant_name: v.answerable_at_5 for v in row_variants}
    bv = {
        v.variant_name: v.context_answerable_at_5
        for v in bundle_variants
    }
    all_names: List[str] = []
    seen: set = set()
    for n in PREFERRED_VARIANT_ORDER:
        if n in rv or n in bv:
            all_names.append(n)
            seen.add(n)
    for n in sorted(set(rv) | set(bv)):
        if n in seen:
            continue
        all_names.append(n)
    return [
        (n, rv.get(n), bv.get(n)) for n in all_names
    ]


def _row_vs_bundle_at_5_dict(
    row_variants: Sequence[VariantMetrics],
    bundle_variants: Sequence[BundleAnswerabilityMetrics],
) -> Dict[str, Dict[str, Any]]:
    """Same data as :func:`_row_vs_bundle_at_5_pairs`, keyed for JSON."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, row_val, ctx_val in _row_vs_bundle_at_5_pairs(
        row_variants, bundle_variants,
    ):
        delta = (
            ctx_val - row_val
            if (row_val is not None and ctx_val is not None)
            else None
        )
        out[name] = {
            "row_answerable_at_5": row_val,
            "context_answerable_at_5": ctx_val,
            "delta_context_minus_row_at_5": delta,
        }
    return out


def _render_row_vs_bundle_comparison(
    row_variants: Sequence[VariantMetrics],
    bundle_variants: Sequence[BundleAnswerabilityMetrics],
) -> str:
    pairs = _row_vs_bundle_at_5_pairs(row_variants, bundle_variants)
    if not pairs:
        return "_(no variants — empty input)_"
    header = (
        "| variant "
        "| row_answerable@5 (any chunk evidence) "
        "| context_answerable@5 (top-5 bundle) "
        "| Δ (context − row) |"
    )
    sep = "|---|---:|---:|---:|"
    rows = [header, sep]
    for name, row_val, ctx_val in pairs:
        r_str = (
            _format_rate(row_val)
            if row_val is not None else "—"
        )
        b_str = (
            _format_rate(ctx_val)
            if ctx_val is not None else "—"
        )
        if row_val is not None and ctx_val is not None:
            d_str = f"{(ctx_val - row_val):+.4f}"
        else:
            d_str = "—"
        rows.append(f"| {name} | {r_str} | {b_str} | {d_str} |")
    return "\n".join(rows)


_MULTI_CHUNK_CAVEAT_LINES: Tuple[str, ...] = (
    "Row-level metrics ask: *does this single chunk carry the "
    "evidence?*. Bundle-level metrics ask: *does the top-k context as "
    "a whole answer the query?*",
    "When ``needs_cross_section`` or ``needs_subpage`` flags fire on "
    "bundle rows, row-level aggregation alone tends to "
    "**underestimate** answerability — multiple chunks together carry "
    "an answer that no single chunk does.",
    "When ``evidence_too_noisy`` or ``wrong_page`` flags fire on bundle "
    "rows alongside high row-level page_hit, the retriever found the "
    "right neighbourhood but the context lacks the specific evidence "
    "— a chunking / section-selection problem, not a retrieval problem.",
    "**Row-level metric is for evidence-quality diagnosis; bundle-level "
    "metric is for top-k answerability judgment.** They are "
    "complementary, not interchangeable — read both together.",
    "**Answerability metric is NOT a hit@k replacement.** Both row and "
    "bundle scores are *additive* signals — production-config promotion "
    "stays gated by the existing retrieval-side metrics from "
    "Phase 7.5 / 7.6.",
)


_RECOMMENDED_LABELING_WORKFLOW: Tuple[str, ...] = (
    "**Step 1.** Generate a bundle export against the "
    "production_recommended config: ``scripts/export_answerability_"
    "audit.py --mode bundle ...`` with ``--top-k-set 1,3,5`` (default).",
    "**Step 2.** Sample 10 distinct query_ids for human pilot review: "
    "``scripts/export_answerability_audit.py --mode bundle-sample "
    "--input-path bundle_export.csv --sample-query-count 10 --seed 42``.",
    "**Step 3 — human review** (NOT automated). Read the sampled "
    "bundles and confirm: are the four label definitions clear on real "
    "chunks? Are top_k=1/3/5 all useful, or is a single k sufficient? "
    "Are both row-level and bundle-level labels worth keeping? Do the "
    "``needs_cross_section`` / ``needs_subpage`` flags actually fire "
    "on real cases? Is ``context_bundle_text`` too long to label "
    "efficiently?",
    "**Step 4 — human decision** (NOT automated). Based on Step 3, fix "
    "the pilot labelling scope (top_k subset, row-vs-bundle balance, "
    "truncate budget) before any large-scale labelling.",
    "**Step 5.** Run the pilot labelling and score with "
    "``scripts/score_answerability_audit.py``.",
    "**Step 6.** Repeat the export → sample → label loop on the Phase "
    "7.6 section-aware candidate query set, then compare deltas. "
    "Promotion decisions stay with Phase 7.6's existing guardrails.",
)


def _render_bundle_sections(
    rows: Sequence[AnswerabilityLabeledRow],
    bundle_rows: Sequence[ContextBundleAuditRow],
) -> List[str]:
    """Build the four bundle-related Markdown sections.

    Returns the lines (already including blank-line separators) so
    :func:`render_markdown_report` just splices them in. Sections are
    pinned by the test suite — renaming any header is a regression.
    """
    bundle_variants = compute_all_bundle_variants(bundle_rows)
    row_variants = compute_all_variants(rows)

    out: List[str] = []

    # Bundle metric table (with brief summary line).
    out.append("## Context bundle answerability")
    out.append("")
    n_bundle_rows = len(bundle_rows)
    n_bundle_queries = len({r.query_id for r in bundle_rows})
    bundle_top_ks = sorted({r.top_k for r in bundle_rows})
    out.append(
        f"Top-k bundle-level answerability — {n_bundle_rows} labelled "
        f"row(s) over {n_bundle_queries} distinct query(ies); "
        f"top_k set = {bundle_top_ks}. Each metric column reports the "
        f"fraction of queries whose bundle row at that top_k is "
        f"labelled at the given level or above."
    )
    out.append("")
    out.append(_render_bundle_metrics_table(bundle_variants))
    out.append("")
    out.append(_render_bundle_failure_buckets(bundle_variants))
    out.append("")

    # Row vs bundle Δ.
    out.append("## Row evidence vs context answerability")
    out.append("")
    out.append(
        "Row-level metrics aggregate per-chunk labels (evidence "
        "diagnosis); bundle-level metrics use a single label over the "
        "top-k context (answerability judgment). The Δ column is "
        "context@5 − row@5 — positive Δ usually means multi-chunk "
        "synthesis is recovering answerability that single-chunk "
        "evidence alone would miss."
    )
    out.append("")
    out.append(
        _render_row_vs_bundle_comparison(row_variants, bundle_variants)
    )
    out.append("")

    # Multi-chunk caveat.
    out.append("## Multi-chunk answerability caveat")
    out.append("")
    for caveat in _MULTI_CHUNK_CAVEAT_LINES:
        out.append(f"- {caveat}")
    out.append("")

    # Recommended labelling workflow.
    out.append("## Recommended labeling workflow")
    out.append("")
    for step in _RECOMMENDED_LABELING_WORKFLOW:
        out.append(f"- {step}")
    out.append("")

    return out


# ---------------------------------------------------------------------------
# Sampling helper (Phase 7.7.1)
# ---------------------------------------------------------------------------


DEFAULT_SAMPLE_QUERY_COUNT: int = 10
DEFAULT_SAMPLE_SEED: int = 42


def sample_bundle_records(
    records: Sequence[Mapping[str, Any]],
    *,
    n_queries: int = DEFAULT_SAMPLE_QUERY_COUNT,
    seed: int = DEFAULT_SAMPLE_SEED,
    variant_name: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Sample N distinct query_ids from a bundle export, deterministic.

    Filtering happens *before* the candidate pool is built — pass
    ``variant_name`` and/or ``top_k`` to restrict the candidate
    queries to records matching those values. The returned list
    contains every record (after filtering) whose ``query_id`` was
    selected, in the same order they appeared in ``records``.

    The sample is **deterministic**: identical ``records`` + ``seed``
    + ``n_queries`` + filters always produce the same query_ids.
    Selection is over the *sorted* set of unique query_ids so input
    ordering does not perturb the result.

    The function is **read-only**: it never modifies any record and
    in particular never fills in ``label_context_answerability`` /
    ``context_flags`` / ``notes`` — that is strictly a human step.
    """
    n_queries_int = int(n_queries)
    if n_queries_int <= 0:
        return []

    filtered: List[Mapping[str, Any]] = []
    for rec in records:
        if variant_name is not None:
            if str(rec.get("variant_name")) != str(variant_name):
                continue
        if top_k is not None:
            try:
                rec_k = int(rec.get("top_k"))
            except (TypeError, ValueError):
                continue
            if rec_k != int(top_k):
                continue
        filtered.append(rec)

    unique_qids = sorted({
        str(r.get("query_id"))
        for r in filtered
        if r.get("query_id") not in (None, "")
    })
    if not unique_qids:
        return []

    rng = _random.Random(int(seed))
    n_pick = min(n_queries_int, len(unique_qids))
    chosen: List[str] = rng.sample(unique_qids, k=n_pick)
    chosen_set = set(chosen)

    return [
        rec for rec in filtered
        if str(rec.get("query_id")) in chosen_set
    ]
