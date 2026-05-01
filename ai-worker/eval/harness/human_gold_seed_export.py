"""Phase 7 human-gold-seed audit exporter.

The Phase 7.0 silver set (and the silver-500 expansion built by
:mod:`silver_500_generator`) carries silver-derived expected titles /
expected page ids. The labels are template-driven, not human-verified —
no precision/recall/accuracy claim should be made off them alone.

This exporter draws a small **stratified** audit sample (default 50
rows) across:

  - bucket            (main_work / subpage_generic / subpage_named)
  - confidence label  (CONFIDENT / AMBIGUOUS / LOW_CONFIDENCE / FAILED)
  - failure reason    (LOW_MARGIN, GENERIC_COLLISION, TITLE_ALIAS_MISMATCH)
  - recovery context  (QUERY_REWRITE, HYBRID_RECOVERY)
  - the synthetic edge ``expected_target_not_in_candidates``

…then emits three sibling files a human reviewer can fill in:

  - ``phase7_human_gold_seed_50.jsonl``   (machine-readable)
  - ``phase7_human_gold_seed_50.csv``     (spreadsheet-friendly)
  - ``phase7_human_gold_seed_50.md``      (human-readable + disclaimer)

Each row carries the silver expected target, the top-1/top-3/top-5
retrieval previews, the Phase 7.3 confidence verdict, the Phase 7.4
recovery action, and a block of empty fields for the reviewer to fill::

  - human_label              one of the allowed enum values below
  - human_correct_title      free text, the reviewer's chosen title
  - human_correct_page_id    free text, the reviewer's chosen page_id
  - human_supporting_chunk_id chunk_id the reviewer used as evidence
  - human_notes              free-text comments

Allowed ``human_label`` values (frozen vocabulary):

  - ``SUPPORTED``             silver target is correct and the retrieval
                              actually returned it.
  - ``PARTIALLY_SUPPORTED``   silver target is correct but the
                              retrieval is mixed / weak.
  - ``WRONG_TARGET``          silver target is correct but the retriever
                              returned the wrong thing.
  - ``AMBIGUOUS_QUERY``       the query itself is under-specified and
                              admits multiple targets.
  - ``NOT_IN_CORPUS``         the silver target turned out to be a
                              page that doesn't exist in the corpus.
  - ``BAD_SILVER_LABEL``      the silver target itself is wrong (silver
                              generator produced a misleading label).

Determinism: the exporter uses a fixed seed for the within-stratum
shuffling; the same set of inputs always produces byte-identical
outputs.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen taxonomies — kept in sync with the spec
# ---------------------------------------------------------------------------


HUMAN_LABEL_SUPPORTED = "SUPPORTED"
HUMAN_LABEL_PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
HUMAN_LABEL_WRONG_TARGET = "WRONG_TARGET"
HUMAN_LABEL_AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
HUMAN_LABEL_NOT_IN_CORPUS = "NOT_IN_CORPUS"
HUMAN_LABEL_BAD_SILVER_LABEL = "BAD_SILVER_LABEL"

ALLOWED_HUMAN_LABELS: Tuple[str, ...] = (
    HUMAN_LABEL_SUPPORTED,
    HUMAN_LABEL_PARTIALLY_SUPPORTED,
    HUMAN_LABEL_WRONG_TARGET,
    HUMAN_LABEL_AMBIGUOUS_QUERY,
    HUMAN_LABEL_NOT_IN_CORPUS,
    HUMAN_LABEL_BAD_SILVER_LABEL,
)


# Edge cases: each row is tagged with every applicable flag so the
# stratifier can pull at least one row matching each. The synthetic
# ``expected_target_not_in_candidates`` flag is derived from the silver
# label + the per-query top results list.
EDGE_CONFIDENT = "CONFIDENT"
EDGE_AMBIGUOUS = "AMBIGUOUS"
EDGE_LOW_CONFIDENCE = "LOW_CONFIDENCE"
EDGE_FAILED = "FAILED"
EDGE_LOW_MARGIN = "LOW_MARGIN"
EDGE_GENERIC_COLLISION = "GENERIC_COLLISION"
EDGE_TITLE_ALIAS_MISMATCH = "TITLE_ALIAS_MISMATCH"
EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES = "expected_target_not_in_candidates"
EDGE_QUERY_REWRITE = "QUERY_REWRITE"
EDGE_HYBRID_RECOVERY = "HYBRID_RECOVERY"

EDGE_CASES: Tuple[str, ...] = (
    EDGE_CONFIDENT,
    EDGE_AMBIGUOUS,
    EDGE_LOW_CONFIDENCE,
    EDGE_FAILED,
    EDGE_LOW_MARGIN,
    EDGE_GENERIC_COLLISION,
    EDGE_TITLE_ALIAS_MISMATCH,
    EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES,
    EDGE_QUERY_REWRITE,
    EDGE_HYBRID_RECOVERY,
)


_BUCKET_MAIN_WORK = "main_work"
_BUCKET_SUBPAGE_GENERIC = "subpage_generic"
_BUCKET_SUBPAGE_NAMED = "subpage_named"

_BUCKETS: Tuple[str, ...] = (
    _BUCKET_MAIN_WORK,
    _BUCKET_SUBPAGE_GENERIC,
    _BUCKET_SUBPAGE_NAMED,
)


DEFAULT_BUCKET_TARGETS: Dict[str, int] = {
    _BUCKET_MAIN_WORK: 10,
    _BUCKET_SUBPAGE_GENERIC: 20,
    _BUCKET_SUBPAGE_NAMED: 20,
}


HUMAN_GOLD_DISCLAIMER = (
    "This file is a **manual audit seed** drawn from a silver query "
    "set. The `human_label` column is intentionally blank — fill it "
    "in with one of the allowed values before reporting any "
    "precision/recall/accuracy number."
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_jsonl_lines(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Empty lines are skipped."""
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_silver_queries(silver_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the silver query JSONL into ``{qid: record}``.

    Accepts both the Phase 7.0 silver schema (``id`` + ``query``) and
    the silver-500 schema (same shape, different ``id`` prefix).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_lines(silver_path):
        qid = rec.get("id") or rec.get("qid")
        if qid:
            out[str(qid)] = rec
    return out


def load_per_query_results(
    per_query_path: Path, *, side: str = "candidate",
) -> Dict[str, Dict[str, Any]]:
    """Load Phase 7.0's per_query_comparison.jsonl by qid.

    ``side`` selects ``baseline`` or ``candidate``; the per-query
    block from the chosen side is what we surface as ``top_results``.
    """
    if side not in ("baseline", "candidate"):
        raise ValueError(f"side must be baseline / candidate, got {side!r}")
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_lines(per_query_path):
        qid = rec.get("qid") or rec.get("query_id")
        if not qid:
            continue
        side_block = rec.get(side) or {}
        out[str(qid)] = {
            "query": rec.get("query"),
            "expected_doc_ids": rec.get("expected_doc_ids") or [],
            "v4_meta": rec.get("v4_meta") or {},
            "bucket": rec.get("bucket")
                or (rec.get("v4_meta") or {}).get("bucket")
                or "",
            "top_results": list(side_block.get("top_results") or []),
            "rank": side_block.get("rank"),
            "status": rec.get("status"),
        }
    return out


def load_confidence_verdicts(
    confidence_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load Phase 7.3 per_query_confidence.jsonl by query_id.

    Returns the full row including ``input.top_candidates_preview`` so
    the exporter can read enriched titles without re-joining against
    the chunks file.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_lines(confidence_path):
        qid = rec.get("query_id") or rec.get("qid")
        if qid:
            out[str(qid)] = rec
    return out


def load_recovery_attempts(
    recovery_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load Phase 7.4 recovery_attempts.jsonl by query_id.

    When ``--rewrite-mode both`` was used in Phase 7.4, a single qid
    may carry two attempts (oracle + production_like). We keep the
    most-informative entry per qid: prefer rows where ``recovered`` /
    ``before_in_top_k != after_in_top_k`` are visible, then fall back
    to the first row for the qid.
    """
    by_qid: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_lines(recovery_path):
        decision = rec.get("decision") or {}
        qid = decision.get("query_id") or decision.get("qid")
        if not qid:
            continue
        qid = str(qid)
        existing = by_qid.get(qid)
        if existing is None:
            by_qid[qid] = rec
            continue
        # Prefer the row that has a real attempt outcome (after_rank
        # different from before_rank) over the synthetic skip rows.
        before = existing.get("before_rank")
        after = existing.get("after_rank")
        new_before = rec.get("before_rank")
        new_after = rec.get("after_rank")
        existing_diff = (before != after) if (before is not None and after is not None) else False
        new_diff = (new_before != new_after) if (new_before is not None and new_after is not None) else False
        if new_diff and not existing_diff:
            by_qid[qid] = rec
    return by_qid


def load_chunk_index(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    """Index ``rag_chunks_*.jsonl`` by chunk_id; carry only the audit fields.

    Carries ``title``, ``retrieval_title``, ``display_title``, ``page_id``,
    ``doc_id``, ``section_path``, ``section_type``, plus a *snippet* —
    the first ~240 characters of ``chunk_text`` (or ``embedding_text``
    when chunk_text is missing). Snippets are pre-truncated so the
    exporter can dump them directly to CSV without exploding row size.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_lines(chunks_path):
        cid = rec.get("chunk_id")
        if not cid:
            continue
        sp = rec.get("section_path")
        section_path: Tuple[str, ...]
        if isinstance(sp, list):
            section_path = tuple(str(x) for x in sp)
        else:
            section_path = ()
        text = rec.get("chunk_text") or rec.get("embedding_text") or ""
        snippet = _snippet(text, limit=240)
        out[str(cid)] = {
            "doc_id": str(rec.get("doc_id") or ""),
            "title": rec.get("title"),
            "retrieval_title": rec.get("retrieval_title"),
            "display_title": rec.get("display_title"),
            "section_path": section_path,
            "section_type": rec.get("section_type"),
            "page_id": rec.get("page_id"),
            "snippet": snippet,
        }
    return out


def _snippet(text: str, *, limit: int) -> str:
    """Single-line snippet — collapse internal whitespace and clip."""
    if not text:
        return ""
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    return s[: limit].rstrip() + "…"


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


@dataclass
class SeedRow:
    """One audit-seed row.

    ``edge_flags`` is the set of edge cases this row exhibits — used
    for stratified picking and reported back to the reviewer in the
    ``notes_for_reviewer`` field. ``stratum_keys`` is the list of
    stratifier hooks the row satisfies (bucket plus every edge flag),
    materialised once per row to keep the picker O(1) per pick.
    """

    query_id: str
    query: str
    bucket: str
    silver_expected_title: Optional[str]
    silver_expected_page_id: Optional[str]
    top1_title: Optional[str]
    top1_page_id: Optional[str]
    top3_titles: List[str]
    top5_titles: List[str]
    top5_snippets: List[str]
    confidence_label: Optional[str]
    failure_reasons: List[str]
    recommended_action: Optional[str]
    recovery_action: Optional[str]
    rewrite_mode: Optional[str]
    notes_for_reviewer: str
    edge_flags: List[str]

    # Empty by design — to be filled by the human reviewer.
    human_label: str = ""
    human_correct_title: str = ""
    human_correct_page_id: str = ""
    human_supporting_chunk_id: str = ""
    human_notes: str = ""


def _resolve_top_candidates(
    qid: str,
    *,
    confidence_row: Optional[Mapping[str, Any]],
    per_query_row: Optional[Mapping[str, Any]],
    chunk_index: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return up to 5 enriched candidates for ``qid``.

    Preference order: Phase 7.3's ``input.top_candidates_preview`` (carries
    enriched title + retrieval_title + section_path) → Phase 7.0's
    ``top_results`` enriched against the chunks index. When neither is
    available the result is empty.
    """
    if confidence_row is not None:
        inp = confidence_row.get("input") or {}
        prev = inp.get("top_candidates_preview")
        if isinstance(prev, list) and prev:
            out: List[Dict[str, Any]] = []
            for c in prev[:5]:
                cid = str(c.get("chunk_id") or "")
                enriched = chunk_index.get(cid) or {}
                out.append({
                    "chunk_id": cid,
                    "doc_id": str(c.get("doc_id") or ""),
                    "title": c.get("title") or enriched.get("title"),
                    "retrieval_title": c.get("retrieval_title")
                        or enriched.get("retrieval_title"),
                    "section_path": list(
                        c.get("section_path")
                        or enriched.get("section_path") or ()
                    ),
                    "page_id": enriched.get("page_id") or c.get("page_id"),
                    "snippet": enriched.get("snippet") or "",
                })
            return out
    if per_query_row is not None:
        top_results = per_query_row.get("top_results") or []
        out2: List[Dict[str, Any]] = []
        for c in top_results[:5]:
            cid = str(c.get("chunk_id") or "")
            enriched = chunk_index.get(cid) or {}
            out2.append({
                "chunk_id": cid,
                "doc_id": str(c.get("doc_id") or enriched.get("doc_id") or ""),
                "title": enriched.get("title"),
                "retrieval_title": enriched.get("retrieval_title"),
                "section_path": list(enriched.get("section_path") or ()),
                "page_id": enriched.get("page_id"),
                "snippet": enriched.get("snippet") or "",
            })
        return out2
    return []


def _silver_expected_title(silver_rec: Mapping[str, Any]) -> Optional[str]:
    """Pull the expected title off a silver record.

    Prefers ``v4_meta.retrieval_title`` (most informative for sub-pages)
    then falls back to ``v4_meta.work_title`` and finally
    ``v4_meta.page_title``.
    """
    meta = silver_rec.get("v4_meta") or {}
    rt = meta.get("retrieval_title")
    if rt:
        return str(rt)
    wt = meta.get("work_title")
    if wt:
        return str(wt)
    pt = meta.get("page_title")
    if pt:
        return str(pt)
    return None


def _silver_expected_page_id(silver_rec: Mapping[str, Any]) -> Optional[str]:
    """Return the silver expected page_id, or ``None`` if absent."""
    expected = silver_rec.get("expected_doc_ids") or []
    if isinstance(expected, list) and expected:
        return str(expected[0])
    return None


def _bucket_of(
    silver_rec: Optional[Mapping[str, Any]],
    confidence_row: Optional[Mapping[str, Any]],
    per_query_row: Optional[Mapping[str, Any]],
) -> str:
    """Best-effort bucket lookup across the three input sources."""
    if silver_rec is not None:
        meta = silver_rec.get("v4_meta") or {}
        b = meta.get("bucket")
        if b:
            return str(b)
        for tag in (silver_rec.get("tags") or []):
            if tag in _BUCKETS:
                return tag
    if confidence_row is not None:
        b = confidence_row.get("bucket")
        if b:
            return str(b)
    if per_query_row is not None:
        b = per_query_row.get("bucket")
        if b:
            return str(b)
    return ""


def _expected_target_in_top_results(
    expected_page_id: Optional[str],
    top_candidates: Sequence[Mapping[str, Any]],
) -> bool:
    """Did the silver expected page_id appear anywhere in the top-K?

    Matches against ``page_id`` first (Phase 7.0 chunks carry ``page_id``
    as the page-level grouping key) and falls back to ``doc_id`` since
    the v4 namu corpus uses doc_id == page_id at the page level.
    """
    if not expected_page_id:
        return True  # unknown gold → no signal
    for c in top_candidates:
        pid = c.get("page_id") or c.get("doc_id")
        if pid and str(pid) == str(expected_page_id):
            return True
    return False


def _compose_notes(
    *,
    confidence_row: Optional[Mapping[str, Any]],
    recovery_row: Optional[Mapping[str, Any]],
    edge_flags: Sequence[str],
    expected_in_topk: bool,
) -> str:
    """Build the reviewer-facing one-line note.

    Highlights only the most informative signals: confidence label,
    Phase 7.4 decision (when present), and any edge flags that
    distinguish this row. The note is intentionally short — the
    reviewer reads the JSONL row for full detail.
    """
    parts: List[str] = []
    if confidence_row is not None:
        lab = confidence_row.get("confidence_label")
        if lab:
            parts.append(f"label={lab}")
        reasons = confidence_row.get("failure_reasons") or []
        if reasons:
            parts.append("reasons=[" + ",".join(str(r) for r in reasons) + "]")
        action = confidence_row.get("recommended_action")
        if action:
            parts.append(f"action={action}")
    if recovery_row is not None:
        decision = recovery_row.get("decision") or {}
        ra = decision.get("recovery_action")
        if ra:
            parts.append(f"recovery={ra}")
        rm = decision.get("rewrite_mode")
        if rm:
            parts.append(f"rewrite_mode={rm}")
    if not expected_in_topk:
        parts.append("silver_target_missing_from_top_k")
    if edge_flags:
        parts.append("edges=[" + ",".join(edge_flags) + "]")
    if not parts:
        return "no Phase 7.3/7.4 signal — review directly."
    return " | ".join(parts)


def _edge_flags_for(
    *,
    confidence_row: Optional[Mapping[str, Any]],
    recovery_row: Optional[Mapping[str, Any]],
    expected_in_topk: bool,
) -> List[str]:
    """Compute the edge-case flags for one row.

    Confidence labels and failure reasons map 1:1 to edge keys; the
    Phase 7.4 ``recovery_action`` maps onto QUERY_REWRITE / HYBRID_RECOVERY
    but only when the loop actually attempted a recovery (we ignore
    SKIP_*). The ``expected_target_not_in_candidates`` flag is the
    sole edge derived purely from inputs, not from labels.
    """
    flags: List[str] = []
    if confidence_row is not None:
        lab = str(confidence_row.get("confidence_label") or "")
        if lab in (EDGE_CONFIDENT, EDGE_AMBIGUOUS, EDGE_LOW_CONFIDENCE, EDGE_FAILED):
            flags.append(lab)
        for r in (confidence_row.get("failure_reasons") or []):
            r = str(r)
            if r in (EDGE_LOW_MARGIN, EDGE_GENERIC_COLLISION, EDGE_TITLE_ALIAS_MISMATCH):
                flags.append(r)
    if recovery_row is not None:
        decision = recovery_row.get("decision") or {}
        original_action = str(decision.get("original_action") or "")
        recovery_action = str(decision.get("recovery_action") or "")
        if (
            recovery_action == "ATTEMPT_REWRITE"
            or original_action == "QUERY_REWRITE"
        ):
            flags.append(EDGE_QUERY_REWRITE)
        if (
            recovery_action == "ATTEMPT_HYBRID"
            or original_action == "HYBRID_RECOVERY"
        ):
            flags.append(EDGE_HYBRID_RECOVERY)
    if not expected_in_topk:
        flags.append(EDGE_EXPECTED_TARGET_NOT_IN_CANDIDATES)
    # Dedupe while preserving fire order.
    seen: set = set()
    out: List[str] = []
    for f in flags:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _build_seed_row(
    qid: str,
    *,
    silver_rec: Mapping[str, Any],
    per_query_row: Optional[Mapping[str, Any]],
    confidence_row: Optional[Mapping[str, Any]],
    recovery_row: Optional[Mapping[str, Any]],
    chunk_index: Mapping[str, Mapping[str, Any]],
) -> SeedRow:
    """Materialise one ``SeedRow`` for ``qid``.

    ``per_query_row`` may be ``None`` when the user passes only a
    confidence file; the row is still emitted with whatever the
    confidence row's preview block carries. The same applies to
    ``recovery_row``.
    """
    bucket = _bucket_of(silver_rec, confidence_row, per_query_row)
    expected_title = _silver_expected_title(silver_rec)
    expected_page_id = _silver_expected_page_id(silver_rec)

    candidates = _resolve_top_candidates(
        qid,
        confidence_row=confidence_row,
        per_query_row=per_query_row,
        chunk_index=chunk_index,
    )

    top1 = candidates[0] if candidates else {}
    top3_titles = [
        str(c.get("retrieval_title") or c.get("title") or "")
        for c in candidates[:3]
    ]
    top5_titles = [
        str(c.get("retrieval_title") or c.get("title") or "")
        for c in candidates[:5]
    ]
    top5_snippets = [str(c.get("snippet") or "") for c in candidates[:5]]

    expected_in_topk = _expected_target_in_top_results(
        expected_page_id, candidates,
    )
    edge_flags = _edge_flags_for(
        confidence_row=confidence_row,
        recovery_row=recovery_row,
        expected_in_topk=expected_in_topk,
    )

    confidence_label: Optional[str] = None
    failure_reasons: List[str] = []
    recommended_action: Optional[str] = None
    if confidence_row is not None:
        if confidence_row.get("confidence_label"):
            confidence_label = str(confidence_row["confidence_label"])
        failure_reasons = [str(r) for r in (confidence_row.get("failure_reasons") or [])]
        if confidence_row.get("recommended_action"):
            recommended_action = str(confidence_row["recommended_action"])

    recovery_action: Optional[str] = None
    rewrite_mode: Optional[str] = None
    if recovery_row is not None:
        decision = recovery_row.get("decision") or {}
        if decision.get("recovery_action"):
            recovery_action = str(decision["recovery_action"])
        if decision.get("rewrite_mode"):
            rewrite_mode = str(decision["rewrite_mode"])

    notes = _compose_notes(
        confidence_row=confidence_row,
        recovery_row=recovery_row,
        edge_flags=edge_flags,
        expected_in_topk=expected_in_topk,
    )

    return SeedRow(
        query_id=qid,
        query=str(silver_rec.get("query") or ""),
        bucket=bucket,
        silver_expected_title=expected_title,
        silver_expected_page_id=expected_page_id,
        top1_title=str(top1.get("retrieval_title") or top1.get("title") or "")
            or None,
        top1_page_id=(
            str(top1.get("page_id") or top1.get("doc_id") or "") or None
        ) if top1 else None,
        top3_titles=top3_titles,
        top5_titles=top5_titles,
        top5_snippets=top5_snippets,
        confidence_label=confidence_label,
        failure_reasons=failure_reasons,
        recommended_action=recommended_action,
        recovery_action=recovery_action,
        rewrite_mode=rewrite_mode,
        notes_for_reviewer=notes,
        edge_flags=edge_flags,
    )


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanGoldSeedConfig:
    """Configuration knobs for the exporter."""

    target_total: int = 50
    bucket_targets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_BUCKET_TARGETS)
    )
    seed: int = 42
    side: str = "candidate"

    def validate(self) -> "HumanGoldSeedConfig":
        if self.target_total < 1:
            raise ValueError(
                f"target_total must be >= 1, got {self.target_total}"
            )
        if self.side not in ("baseline", "candidate"):
            raise ValueError(
                f"side must be baseline / candidate, got {self.side!r}"
            )
        for b in _BUCKETS:
            if int(self.bucket_targets.get(b, 0)) < 0:
                raise ValueError(
                    f"bucket_targets[{b!r}] must be >= 0, "
                    f"got {self.bucket_targets.get(b)!r}"
                )
        return self


def _pick_stratified(
    rows: Sequence[SeedRow],
    *,
    config: HumanGoldSeedConfig,
) -> Tuple[List[SeedRow], Dict[str, Any]]:
    """Pick ``target_total`` rows via stratified sampling.

    Stage 1: per-bucket target met first (deterministic shuffle).
    Stage 2: edge-case top-up — if any edge case has zero rows in the
    selection, swap in one row that exhibits it (drawn from a bucket
    that's currently *over* its target, or unfilled budget). Stage 3:
    deterministic fallback fills the remaining budget from the global
    pool (sorted by query_id) so the total exactly meets target_total
    when enough rows exist.

    The picker never reaches across the silver/non-silver boundary —
    every row in ``rows`` came out of ``_build_seed_row`` so they're
    all silver-target audit candidates by construction.
    """
    rng = random.Random(int(config.seed))

    by_bucket: Dict[str, List[SeedRow]] = defaultdict(list)
    for r in rows:
        by_bucket[r.bucket or ""].append(r)
    for entries in by_bucket.values():
        entries.sort(key=lambda r: r.query_id)
        rng.shuffle(entries)

    selected: List[SeedRow] = []
    selected_qids: set = set()

    # Stage 1: per-bucket targets.
    bucket_actual: Dict[str, int] = {b: 0 for b in _BUCKETS}
    bucket_deficits: Dict[str, int] = {b: 0 for b in _BUCKETS}
    for b in _BUCKETS:
        target = int(config.bucket_targets.get(b, 0))
        pool = by_bucket.get(b, [])
        for r in pool:
            if len(selected) >= config.target_total:
                break
            if bucket_actual[b] >= target:
                break
            if r.query_id in selected_qids:
                continue
            selected.append(r)
            selected_qids.add(r.query_id)
            bucket_actual[b] += 1
        bucket_deficits[b] = max(0, target - bucket_actual[b])

    # Stage 2: edge-case top-up. Walk the canonical edge order, swap in
    # one row per missing edge if possible. We always try to *add*
    # rather than swap when budget remains; only swap when budget is
    # already at target_total (preserves stratum if possible).
    edge_present: Dict[str, int] = {e: 0 for e in EDGE_CASES}
    for r in selected:
        for e in r.edge_flags:
            if e in edge_present:
                edge_present[e] += 1

    # Sort all rows by query_id so the candidate ordering for the
    # top-up phase is deterministic regardless of the random shuffle.
    sorted_rows = sorted(rows, key=lambda r: r.query_id)
    for edge in EDGE_CASES:
        if edge_present.get(edge, 0) > 0:
            continue
        candidates = [r for r in sorted_rows if edge in r.edge_flags
                      and r.query_id not in selected_qids]
        if not candidates:
            continue
        # Preserve bucket distribution where possible: prefer the
        # bucket with the largest deficit, then any bucket.
        candidates.sort(key=lambda r: (
            -bucket_deficits.get(r.bucket, 0),
            r.query_id,
        ))
        new_row = candidates[0]
        if len(selected) < config.target_total:
            selected.append(new_row)
            selected_qids.add(new_row.query_id)
            if new_row.bucket in bucket_actual:
                bucket_actual[new_row.bucket] += 1
                bucket_deficits[new_row.bucket] = max(
                    0,
                    int(config.bucket_targets.get(new_row.bucket, 0))
                    - bucket_actual[new_row.bucket],
                )
        else:
            # Swap: drop the row that contributes least (no edge flags).
            swap_idx = -1
            for i, existing in enumerate(selected):
                if not existing.edge_flags:
                    swap_idx = i
                    break
            if swap_idx >= 0:
                dropped = selected.pop(swap_idx)
                selected_qids.remove(dropped.query_id)
                if dropped.bucket in bucket_actual:
                    bucket_actual[dropped.bucket] -= 1
                selected.append(new_row)
                selected_qids.add(new_row.query_id)
                if new_row.bucket in bucket_actual:
                    bucket_actual[new_row.bucket] += 1
        for e in new_row.edge_flags:
            if e in edge_present:
                edge_present[e] += 1

    # Stage 3: deterministic fill.
    if len(selected) < config.target_total:
        # Prefer rows with edge flags, then any row, sorted by qid for
        # stability.
        remaining = [r for r in sorted_rows if r.query_id not in selected_qids]
        remaining.sort(
            key=lambda r: (
                0 if r.edge_flags else 1,
                r.query_id,
            )
        )
        for r in remaining:
            if len(selected) >= config.target_total:
                break
            selected.append(r)
            selected_qids.add(r.query_id)
            if r.bucket in bucket_actual:
                bucket_actual[r.bucket] += 1

    # Final ordering: by bucket then by qid so a CSV reader scans the
    # main_work block first, then subpage_generic, then subpage_named.
    bucket_order = {b: i for i, b in enumerate(_BUCKETS)}
    selected.sort(
        key=lambda r: (bucket_order.get(r.bucket, 99), r.query_id)
    )

    edge_after = Counter()
    for r in selected:
        for e in r.edge_flags:
            edge_after[e] += 1

    audit_summary: Dict[str, Any] = {
        "target_total": int(config.target_total),
        "actual_total": len(selected),
        "bucket_targets": dict(config.bucket_targets),
        "bucket_actual_counts": dict(bucket_actual),
        "bucket_deficits": dict(bucket_deficits),
        "edge_case_counts": {e: int(edge_after.get(e, 0)) for e in EDGE_CASES},
        "edge_cases_missing": [
            e for e in EDGE_CASES if edge_after.get(e, 0) == 0
        ],
    }
    return selected, audit_summary


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


@dataclass
class HumanGoldSeedExport:
    """Container for the export's three artefacts."""

    rows: List[SeedRow] = field(default_factory=list)
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0


def build_human_gold_seed(
    *,
    silver_path: Path,
    per_query_path: Optional[Path] = None,
    confidence_path: Optional[Path] = None,
    recovery_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    config: Optional[HumanGoldSeedConfig] = None,
) -> HumanGoldSeedExport:
    """End-to-end: load inputs, build candidate rows, stratified pick.

    All inputs except ``silver_path`` are optional. When an input is
    omitted the corresponding fields stay blank (e.g. no confidence
    file → ``confidence_label`` is None for every row, the
    confidence-derived edge cases never fire). The exporter still
    produces a valid audit seed; the reviewer simply has fewer
    pre-filled signals.
    """
    cfg = (config or HumanGoldSeedConfig()).validate()

    silver = load_silver_queries(silver_path)
    per_query = (
        load_per_query_results(per_query_path, side=cfg.side)
        if per_query_path is not None else {}
    )
    confidence = (
        load_confidence_verdicts(confidence_path)
        if confidence_path is not None else {}
    )
    recovery = (
        load_recovery_attempts(recovery_path)
        if recovery_path is not None else {}
    )
    chunk_index = (
        load_chunk_index(chunks_path)
        if chunks_path is not None else {}
    )

    rows: List[SeedRow] = []
    for qid, silver_rec in silver.items():
        rows.append(_build_seed_row(
            qid,
            silver_rec=silver_rec,
            per_query_row=per_query.get(qid),
            confidence_row=confidence.get(qid),
            recovery_row=recovery.get(qid),
            chunk_index=chunk_index,
        ))

    selected, audit_summary = _pick_stratified(rows, config=cfg)
    return HumanGoldSeedExport(
        rows=selected,
        audit_summary=audit_summary,
        candidate_count=len(rows),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


_JSONL_FIELDS: Tuple[str, ...] = (
    "query_id",
    "query",
    "bucket",
    "silver_expected_title",
    "silver_expected_page_id",
    "top1_title",
    "top1_page_id",
    "top3_titles",
    "top5_titles",
    "top5_snippets",
    "confidence_label",
    "failure_reasons",
    "recommended_action",
    "recovery_action",
    "rewrite_mode",
    "notes_for_reviewer",
    "edge_flags",
    "human_label",
    "human_correct_title",
    "human_correct_page_id",
    "human_supporting_chunk_id",
    "human_notes",
)


def _row_to_dict(row: SeedRow) -> Dict[str, Any]:
    """Project a SeedRow to the on-disk dict with stable field order."""
    full = asdict(row)
    return {k: full.get(k) for k in _JSONL_FIELDS}


def write_jsonl(
    rows: Sequence[SeedRow], out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(_row_to_dict(r), ensure_ascii=False) + "\n")
    return out_path


def _csv_value(v: Any) -> str:
    """Coerce list / None values to CSV-friendly strings.

    Lists become ``" | "``-joined strings — readable in Excel and
    re-parseable by a downstream loader. ``None`` becomes empty.
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v)
    return str(v)


def write_csv(
    rows: Sequence[SeedRow], out_path: Path,
) -> Path:
    """Write the seed as a CSV with the same column order as the JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(_JSONL_FIELDS)
        for r in rows:
            d = _row_to_dict(r)
            writer.writerow([_csv_value(d.get(k)) for k in _JSONL_FIELDS])
    return out_path


def render_md(
    export: HumanGoldSeedExport,
    *,
    target_total: int,
) -> str:
    """Render the human-readable Markdown summary + per-row table.

    The header carries the silver/gold disclaimer in two paragraphs so
    the reviewer can't miss it: one banner paragraph before the row
    table, one footer paragraph after the table.
    """
    rows = export.rows
    audit = export.audit_summary
    n = len(rows)

    lines: List[str] = []
    lines.append(f"# Phase 7 human-gold-seed audit ({n} rows)")
    lines.append("")
    lines.append(f"> {HUMAN_GOLD_DISCLAIMER}")
    lines.append("")
    lines.append(
        "This file is **silver-derived** — it is **NOT human-verified "
        "gold**. The columns `silver_expected_title` / "
        "`silver_expected_page_id` come from the silver query generator "
        "and have NOT been verified by a human. Treat them as "
        "hypotheses to confirm or reject."
    )
    lines.append("")
    lines.append(
        "Allowed `human_label` values: "
        + ", ".join(f"`{lab}`" for lab in ALLOWED_HUMAN_LABELS) + "."
    )
    lines.append("")

    lines.append("## Sampling parameters")
    lines.append("")
    lines.append(f"- target_total: **{target_total}**")
    lines.append(f"- candidate_count (silver rows seen): "
                 f"**{export.candidate_count}**")
    bt = audit.get("bucket_targets") or {}
    ba = audit.get("bucket_actual_counts") or {}
    bd = audit.get("bucket_deficits") or {}
    lines.append("")
    lines.append("| bucket | target | actual | deficit |")
    lines.append("|---|---:|---:|---:|")
    for b in _BUCKETS:
        lines.append(
            f"| {b} | {int(bt.get(b, 0))} | {int(ba.get(b, 0))} | "
            f"{int(bd.get(b, 0))} |"
        )
    lines.append("")

    lines.append("## Edge-case coverage")
    lines.append("")
    lines.append("| edge | count |")
    lines.append("|---|---:|")
    edge_counts = audit.get("edge_case_counts") or {}
    for e in EDGE_CASES:
        lines.append(f"| {e} | {int(edge_counts.get(e, 0))} |")
    missing = audit.get("edge_cases_missing") or []
    if missing:
        lines.append("")
        lines.append(
            "**Missing edge cases (no candidate row matched):** "
            + ", ".join(f"`{e}`" for e in missing)
        )
    lines.append("")

    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| qid | bucket | silver_expected_title | top1_title | "
        "confidence | reasons | recovery |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        reasons = ",".join(r.failure_reasons) if r.failure_reasons else "-"
        recovery = r.recovery_action or "-"
        if r.rewrite_mode:
            recovery = f"{recovery}/{r.rewrite_mode}"
        lines.append(
            f"| {r.query_id} | {r.bucket} | "
            f"{_escape_pipe(r.silver_expected_title)} | "
            f"{_escape_pipe(r.top1_title)} | "
            f"{r.confidence_label or '-'} | "
            f"{_escape_pipe(reasons)} | "
            f"{_escape_pipe(recovery)} |"
        )
    lines.append("")

    lines.append("## Reviewer instructions")
    lines.append("")
    lines.append(
        "1. For each row, confirm whether the silver expected target "
        "matches the corpus content cited in the top-K chunks. Use "
        "the JSONL or CSV file (not this Markdown) as the working "
        "copy — they carry the snippets and chunk_ids."
    )
    lines.append(
        "2. Pick exactly one `human_label` from the allowed list."
    )
    lines.append(
        "3. When `human_label` ∈ {`WRONG_TARGET`, `BAD_SILVER_LABEL`}, "
        "fill `human_correct_title` / `human_correct_page_id` so the "
        "follow-up generator can correct the silver record."
    )
    lines.append(
        "4. Until the labels are filled, this seed is **silver, not "
        "gold** — no precision/recall/accuracy claim should be made "
        "off it."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _escape_pipe(s: Any) -> str:
    """Markdown-table-safe rendering — escape pipes and trim."""
    if s is None:
        return "-"
    text = str(s).replace("|", "\\|").replace("\n", " ")
    if len(text) > 60:
        text = text[:60].rstrip() + "…"
    return text or "-"


def write_md(
    export: HumanGoldSeedExport,
    out_path: Path,
    *,
    target_total: int,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_md(export, target_total=target_total),
        encoding="utf-8",
    )
    return out_path


def write_outputs(
    export: HumanGoldSeedExport,
    *,
    out_dir: Path,
    base_name: str = "phase7_human_gold_seed_50",
    target_total: int = 50,
) -> Dict[str, Path]:
    """Write the JSONL / CSV / Markdown artefacts in one call."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = write_jsonl(export.rows, out_dir / f"{base_name}.jsonl")
    csv_path = write_csv(export.rows, out_dir / f"{base_name}.csv")
    md_path = write_md(
        export, out_dir / f"{base_name}.md",
        target_total=target_total,
    )
    return {
        "jsonl": jsonl,
        "csv": csv_path,
        "md": md_path,
    }
