"""Phase 7.x human audit seed exporter.

Builds a small, *stratified* sample of Phase 7.3 / 7.4 queries packaged
for manual review. Output is deterministic: same inputs + config →
byte-identical bundle.

The exporter does NOT make any claim about correctness — every audit
row carries the silver target, the model's verdict, the recovery
attempt's after-state (when present), and a blank ``human_label``
field for the reviewer to fill in. The phrase "silver" is used in
every header and field name to make it obvious that the labels are
heuristic, not human-verified gold.

Inputs:

  - **Phase 7.3 per-query JSONL** — required. Carries the verdict
    (``confidence_label``, ``failure_reasons``, ``recommended_action``,
    ``signals``) plus an ``input`` block with ``query_text``,
    ``expected_title``, ``gold_doc_id`` (the silver target doc), and a
    5-row ``top_candidates_preview`` with title / retrieval_title /
    section_path / section_type / dense / rerank / final scores.

  - **Phase 7.4 recovery_attempts.jsonl** — optional. When supplied,
    each audit row gets a ``recovery`` block carrying recovery_action,
    rewrite_mode, rewritten_query, after_rank, and after_in_top_k.

  - **rag_chunks_*.jsonl** — optional. When supplied, the top-5 chunk
    rows in the audit record gain a real text snippet excerpted from
    the chunk. Without it, the audit shows section_path + section_type
    as the only chunk-level evidence.

Sampling (deterministic):

  - Stratified by ``bucket``: ``main_work`` / ``subpage_generic`` /
    ``subpage_named`` get ``bucket_quota`` rows each.
  - Edge-case quota: every ``confidence_label`` value, the
    ``TITLE_ALIAS_MISMATCH`` and ``GENERIC_COLLISION`` failure
    reasons, the synthetic ``expected_target_not_in_candidates``
    flag (for queries whose silver target was missing from the
    candidate top-k), and ``query_rewrite_candidate`` (Phase 7.4
    rewrite attempts when the recovery JSONL is loaded).

  - Within each strat group, rows are picked *evenly spaced* across
    the qid-sorted list — qid 0001/0050/0100/… instead of
    0001/0002/0003/… — so the audit covers more of the silver
    distribution.

  - The final list is deduplicated by qid (an audit row that satisfies
    multiple strat groups still appears exactly once) and sorted by
    qid. Audit IDs (``audit-0001``, ``audit-0002``, …) are assigned in
    that final order, so re-running the exporter always assigns the
    same audit_id to the same qid.

Output bundle:

  - ``phase7_human_audit_seed.jsonl`` — full record per audit row.
  - ``phase7_human_audit_seed.csv``  — flat columns; ``top3_titles`` /
    ``top5_chunk_snippets`` are pipe-joined for spreadsheet import.
  - ``phase7_human_audit_seed.md``   — human-friendly per-row layout
    with a blank ``Human label`` / ``Human notes`` field at the foot
    so the reviewer can fill them in directly.
  - ``phase7_human_audit_seed_summary.json`` — manifest: counts by
    strat group, the configuration used, and the silver disclaimer.

Production code is NOT touched — this module reads JSONL artefacts
and writes JSONL/CSV/MD outputs. There is no FAISS, no GPU, no
network in the export path.
"""

from __future__ import annotations

import csv
import json
import logging
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

from eval.harness.eval_terminology import (
    EDGE_CASE_CONFIDENCE_LABELS,
    EDGE_CASE_FAILURE_REASONS,
    EDGE_CASE_QUERY_REWRITE,
    EDGE_CASE_TARGET_NOT_IN_CANDIDATES,
    SILVER_DISCLAIMER_LINES,
    SILVER_DISCLAIMER_MARKER,
    SILVER_DISCLAIMER_MD,
    SILVER_DISCLAIMER_TEXT,
    silver_disclaimer_block,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default sampling quotas
# ---------------------------------------------------------------------------


# Bucket-stratified sample size. The brief asks for 10 rows per bucket;
# kept as a default so callers can override for narrower / wider audits.
DEFAULT_BUCKET_QUOTA: Mapping[str, int] = {
    "main_work": 10,
    "subpage_generic": 10,
    "subpage_named": 10,
}


# Edge-case quota. Each confidence_label, failure-reason, and synthetic
# tag the brief calls out gets a dedicated quota of 5. Keeping these
# separate from the bucket quota means the final bundle can grow up
# to (3 buckets × 10) + (8 edge cases × 5) = 70 rows before dedupe.
DEFAULT_EDGE_CASE_QUOTA_PER_TAG: int = 5


def _default_edge_case_quotas() -> Dict[str, int]:
    """Build the full edge-case quota dict from the constants table."""
    out: Dict[str, int] = {}
    for tag in EDGE_CASE_CONFIDENCE_LABELS:
        out[tag] = DEFAULT_EDGE_CASE_QUOTA_PER_TAG
    for tag in EDGE_CASE_FAILURE_REASONS:
        out[tag] = DEFAULT_EDGE_CASE_QUOTA_PER_TAG
    out[EDGE_CASE_TARGET_NOT_IN_CANDIDATES] = DEFAULT_EDGE_CASE_QUOTA_PER_TAG
    out[EDGE_CASE_QUERY_REWRITE] = DEFAULT_EDGE_CASE_QUOTA_PER_TAG
    return out


# Human-label vocabulary. Audit reviewers fill ``human_label`` with
# exactly one of these strings, or leave it blank for "needs more
# review". Documented here so the JSONL / CSV / MD writers all surface
# the same options.
HUMAN_LABEL_CHOICES: Tuple[str, ...] = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "WRONG_TARGET",
    "AMBIGUOUS_QUERY",
    "NOT_IN_CORPUS",
    "BAD_SILVER_LABEL",
)


# Maximum number of characters of chunk text to include in the audit
# snippet. Long enough to read the lead, short enough that the JSONL /
# CSV stay scannable.
DEFAULT_SNIPPET_MAX_CHARS: int = 240


# ---------------------------------------------------------------------------
# Audit row model
# ---------------------------------------------------------------------------


@dataclass
class AuditRow:
    """One row in the human audit seed bundle.

    The dataclass is mutable on purpose — ``human_label`` and
    ``human_notes`` are blank when the bundle is emitted; the auditor
    fills them in by editing the JSONL / CSV / MD directly. The audit
    seed is the unit of work, not a frozen contract.
    """

    audit_id: str
    query_id: str
    query: str
    bucket: str
    silver_target: Dict[str, Any]
    expected_title: Optional[str]
    confidence_label: str
    failure_reasons: List[str]
    recommended_action: str
    top1: Dict[str, Any]
    top3_titles: List[str]
    top5_chunks: List[Dict[str, Any]]
    recovery: Optional[Dict[str, Any]]
    edge_case_tags: List[str]
    human_label: Optional[str] = None
    human_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts; skips blank lines."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_chunks_by_id(chunks_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """Build a chunk_id → record index for snippet enrichment.

    Only the fields the audit row needs are kept (chunk_text /
    embedding_text, title, retrieval_title, section_path, section_type,
    page_id, doc_id) so the index doesn't bloat memory on large corpora.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with Path(chunks_jsonl).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                continue
            out[str(chunk_id)] = {
                "chunk_text": rec.get("chunk_text"),
                "embedding_text": rec.get("embedding_text"),
                "title": rec.get("title"),
                "retrieval_title": rec.get("retrieval_title"),
                "display_title": rec.get("display_title"),
                "section_path": list(rec.get("section_path") or []),
                "section_type": rec.get("section_type"),
                "page_id": rec.get("page_id"),
                "doc_id": str(rec.get("doc_id") or ""),
            }
    return out


def index_recovery_attempts_by_qid(
    attempts: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    """Group recovery_attempts.jsonl rows by query_id (one qid → many attempts).

    A single qid can have multiple attempts when ``rewrite_mode='both'``
    (oracle + production_like) or when a verdict matches several
    recovery actions across modes. The index keeps every attempt so the
    audit row can flag the qid as a query-rewrite candidate without
    losing the per-attempt detail.
    """
    by_qid: Dict[str, List[Mapping[str, Any]]] = {}
    for a in attempts:
        decision = a.get("decision") or {}
        qid = str(decision.get("query_id") or "")
        if not qid:
            continue
        by_qid.setdefault(qid, []).append(a)
    return by_qid


# ---------------------------------------------------------------------------
# Snippet helper
# ---------------------------------------------------------------------------


def _snippet(text: Optional[str], *, max_chars: int) -> str:
    """Trim ``text`` to ``max_chars`` characters; return ``""`` on empty.

    Whitespace is collapsed and trailing ellipsis ``…`` is appended only
    when truncation actually occurred. The function is intentionally
    plain — no language-aware sentence splitting, because the audit
    reviewer wants the raw lead, not a summary.
    """
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _chunk_snippet_block(
    cand: Mapping[str, Any],
    *,
    chunks_index: Optional[Mapping[str, Mapping[str, Any]]],
    max_chars: int,
) -> Dict[str, Any]:
    """Build one ``top5_chunks`` entry from a candidate preview row."""
    chunk_id = str(cand.get("chunk_id") or "")
    title = cand.get("title")
    retrieval_title = cand.get("retrieval_title")
    section_path = list(cand.get("section_path") or [])
    section_type = cand.get("section_type")
    rank = cand.get("rank")
    snippet_text = ""
    if chunks_index is not None and chunk_id:
        enrich = chunks_index.get(chunk_id) or {}
        # Prefer chunk_text (the human-readable raw text) over
        # embedding_text (the prefixed string fed to the embedder).
        snippet_text = _snippet(
            enrich.get("chunk_text") or enrich.get("embedding_text"),
            max_chars=max_chars,
        )
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "doc_id": str(cand.get("doc_id") or ""),
        "title": title,
        "retrieval_title": retrieval_title,
        "section_path": section_path,
        "section_type": section_type,
        "snippet": snippet_text,
    }


# ---------------------------------------------------------------------------
# Edge-case fingerprinting
# ---------------------------------------------------------------------------


def _gather_edge_case_tags(
    confidence_row: Mapping[str, Any],
    *,
    has_recovery_attempt: bool,
    silver_target_in_candidates: Optional[bool],
) -> List[str]:
    """Compute the edge-case tags an audit row should carry.

    Tags come from three sources:
      1. ``confidence_label`` (CONFIDENT / AMBIGUOUS / LOW_CONFIDENCE / FAILED)
      2. Failure reasons that the brief singled out (TITLE_ALIAS_MISMATCH,
         GENERIC_COLLISION).
      3. Synthetic flags: ``expected_target_not_in_candidates`` when the
         silver target is provably missing, ``query_rewrite_candidate``
         when the row has at least one Phase 7.4 rewrite attempt.
    """
    tags: List[str] = []
    label = str(confidence_row.get("confidence_label") or "").strip()
    if label and label in EDGE_CASE_CONFIDENCE_LABELS:
        tags.append(label)
    reasons = list(confidence_row.get("failure_reasons") or [])
    for reason in reasons:
        if reason in EDGE_CASE_FAILURE_REASONS:
            tags.append(str(reason))
    if silver_target_in_candidates is False:
        tags.append(EDGE_CASE_TARGET_NOT_IN_CANDIDATES)
    if has_recovery_attempt:
        tags.append(EDGE_CASE_QUERY_REWRITE)
    # Stable order, dedupe.
    seen: set = set()
    ordered: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


# ---------------------------------------------------------------------------
# Recovery extraction
# ---------------------------------------------------------------------------


def _select_primary_recovery_attempt(
    attempts: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Pick the most-informative recovery attempt for the audit row.

    Preference order: oracle rewrite (highest signal, since it shows
    what the silver-tagged title would have done) > production-like
    rewrite > hybrid > skip. Picks deterministically by sorting on
    a (priority, query_id, rewrite_mode) tuple.
    """
    if not attempts:
        return None

    def _priority(a: Mapping[str, Any]) -> Tuple[int, str, str]:
        decision = a.get("decision") or {}
        action = str(decision.get("recovery_action") or "")
        rewrite_mode = str(decision.get("rewrite_mode") or "")
        # Lower priority value = picked first.
        if action == "ATTEMPT_REWRITE" and rewrite_mode == "oracle":
            p = 0
        elif action == "ATTEMPT_REWRITE":
            p = 1
        elif action == "ATTEMPT_HYBRID":
            p = 2
        else:
            p = 3
        qid = str(decision.get("query_id") or "")
        return (p, qid, rewrite_mode)

    sorted_attempts = sorted(attempts, key=_priority)
    return sorted_attempts[0]


def _build_recovery_block(
    attempt: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compact recovery summary the audit row carries."""
    decision = attempt.get("decision") or {}
    return {
        "recovery_action": decision.get("recovery_action"),
        "rewrite_mode": decision.get("rewrite_mode"),
        "oracle_upper_bound": bool(decision.get("oracle_upper_bound") or False),
        "rewritten_query": decision.get("rewritten_query"),
        "rewrite_terms": list(decision.get("rewrite_terms") or []),
        "rewrite_source": decision.get("rewrite_source"),
        "before_rank": attempt.get("before_rank"),
        "before_in_top_k": attempt.get("before_in_top_k"),
        "after_rank": attempt.get("after_rank"),
        "after_in_top_k": attempt.get("after_in_top_k"),
        "rank_delta": (
            (attempt.get("after_rank") - attempt.get("before_rank"))
            if (
                attempt.get("after_rank") is not None
                and attempt.get("before_rank") is not None
            )
            else None
        ),
        "latency_ms": attempt.get("latency_ms"),
        "error": attempt.get("error"),
    }


# ---------------------------------------------------------------------------
# Audit row builder
# ---------------------------------------------------------------------------


def build_audit_rows(
    confidence_rows: Sequence[Mapping[str, Any]],
    *,
    recovery_by_qid: Optional[Mapping[str, List[Mapping[str, Any]]]] = None,
    chunks_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
    snippet_max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> List[AuditRow]:
    """Convert per-query confidence rows + optional Phase 7.4 attempts → audit rows.

    Audit IDs are assigned later (by the sampler) so this function just
    seeds them with the input qid; the sampler re-numbers in qid order
    before writing.
    """
    audit_rows: List[AuditRow] = []
    recovery_by_qid = recovery_by_qid or {}

    for row in confidence_rows:
        qid = str(row.get("query_id") or row.get("qid") or "")
        bucket = str(row.get("bucket") or "")
        confidence_label = str(row.get("confidence_label") or "")
        failure_reasons = list(row.get("failure_reasons") or [])
        recommended_action = str(row.get("recommended_action") or "")
        signals = row.get("signals") or {}
        input_block = row.get("input") or {}
        query_text = str(input_block.get("query_text") or row.get("query") or "")

        expected_title = input_block.get("expected_title")
        # The silver target object: doc_id + page_id + the canonical
        # silver title. Using a separate object (vs flat fields) so the
        # JSONL row is self-describing.
        silver_target = {
            "doc_id": input_block.get("gold_doc_id"),
            "page_id": input_block.get("gold_page_id"),
            "title": expected_title,
            "expected_section_type": input_block.get("expected_section_type"),
        }
        # Did the silver target survive into the candidate top-k?
        silver_in_topk = signals.get("gold_in_top_k")

        # Top candidates preview from Phase 7.3 — capped at 5 rows by
        # the writer; we don't re-cap here so an enriched phase 7.3
        # bundle that ships top-10 still works.
        candidates_preview = list(input_block.get("top_candidates_preview") or [])
        top1 = {}
        if candidates_preview:
            c0 = candidates_preview[0]
            top1 = {
                "rank": c0.get("rank", 1),
                "chunk_id": c0.get("chunk_id"),
                "doc_id": c0.get("doc_id"),
                "title": c0.get("title"),
                "retrieval_title": c0.get("retrieval_title"),
                "section_path": list(c0.get("section_path") or []),
                "section_type": c0.get("section_type"),
                "dense_score": c0.get("dense_score"),
                "rerank_score": c0.get("rerank_score"),
                "final_score": c0.get("final_score"),
            }
        top3_titles = [
            str(c.get("retrieval_title") or c.get("title") or "")
            for c in candidates_preview[:3]
        ]
        top5_chunks = [
            _chunk_snippet_block(
                c, chunks_index=chunks_index, max_chars=snippet_max_chars,
            )
            for c in candidates_preview[:5]
        ]

        attempts = recovery_by_qid.get(qid) or []
        primary_attempt = _select_primary_recovery_attempt(attempts)
        recovery_block = (
            _build_recovery_block(primary_attempt) if primary_attempt else None
        )

        edge_case_tags = _gather_edge_case_tags(
            row,
            has_recovery_attempt=bool(attempts),
            silver_target_in_candidates=silver_in_topk,
        )

        audit_rows.append(AuditRow(
            audit_id="",  # filled in by the sampler
            query_id=qid,
            query=query_text,
            bucket=bucket,
            silver_target=silver_target,
            expected_title=expected_title,
            confidence_label=confidence_label,
            failure_reasons=failure_reasons,
            recommended_action=recommended_action,
            top1=top1,
            top3_titles=top3_titles,
            top5_chunks=top5_chunks,
            recovery=recovery_block,
            edge_case_tags=edge_case_tags,
            human_label=None,
            human_notes=None,
        ))
    return audit_rows


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _pick_evenly_spaced(
    items: Sequence[AuditRow], n: int,
) -> List[AuditRow]:
    """Pick ``n`` items evenly spread across ``items``.

    Returns the items at indices ``round(step*i)`` for
    ``step = len(items)/n``. Works without random sampling so the
    selection is fully deterministic. When ``n >= len(items)`` returns
    every item; when ``n <= 0`` returns the empty list.
    """
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / float(n)
    indices: List[int] = []
    seen: set = set()
    for i in range(n):
        idx = int(round(step * i))
        if idx >= len(items):
            idx = len(items) - 1
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    indices.sort()
    return [items[i] for i in indices]


def sample_audit_rows(
    rows: Sequence[AuditRow],
    *,
    bucket_quota: Optional[Mapping[str, int]] = None,
    edge_case_quota: Optional[Mapping[str, int]] = None,
) -> Tuple[List[AuditRow], Dict[str, Any]]:
    """Stratified, deterministic sampler.

    Returns (selected_rows_sorted_by_qid, manifest_dict). The manifest
    block carries per-strat counts so the writer can include it in
    the summary JSON.
    """
    # Distinguish "explicitly disabled" (empty dict) from "not provided"
    # (None). ``{}`` means "skip this strat group entirely"; ``None``
    # means "use the default quotas".
    bucket_quota = (
        dict(bucket_quota) if bucket_quota is not None
        else dict(DEFAULT_BUCKET_QUOTA)
    )
    edge_case_quota = (
        dict(edge_case_quota) if edge_case_quota is not None
        else _default_edge_case_quotas()
    )

    # Sort canonical iteration order by qid so the evenly-spaced picker
    # is deterministic regardless of input order.
    sorted_rows = sorted(rows, key=lambda r: r.query_id)

    selected_qids: set = set()
    manifest_strat: Dict[str, Dict[str, int]] = {}

    # Bucket strat.
    for bucket, quota in bucket_quota.items():
        bucket_pool = [r for r in sorted_rows if r.bucket == bucket]
        picked = _pick_evenly_spaced(bucket_pool, quota)
        manifest_strat[f"bucket:{bucket}"] = {
            "quota": int(quota),
            "available": len(bucket_pool),
            "picked": len(picked),
        }
        for r in picked:
            selected_qids.add(r.query_id)

    # Edge-case strat. A row qualifies for an edge case if its
    # ``edge_case_tags`` contains the tag.
    for tag, quota in edge_case_quota.items():
        edge_pool = [r for r in sorted_rows if tag in r.edge_case_tags]
        picked = _pick_evenly_spaced(edge_pool, quota)
        manifest_strat[f"edge:{tag}"] = {
            "quota": int(quota),
            "available": len(edge_pool),
            "picked": len(picked),
        }
        for r in picked:
            selected_qids.add(r.query_id)

    # Materialise selection in qid order, then renumber audit_id.
    selected = [r for r in sorted_rows if r.query_id in selected_qids]
    for idx, row in enumerate(selected, start=1):
        row.audit_id = f"audit-{idx:04d}"

    manifest: Dict[str, Any] = {
        "n_audit_rows": len(selected),
        "n_unique_qids": len(selected_qids),
        "strat_groups": manifest_strat,
        "bucket_quota": dict(bucket_quota),
        "edge_case_quota": dict(edge_case_quota),
    }
    return selected, manifest


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _flatten_for_csv(row: AuditRow) -> Dict[str, str]:
    """Compact a single audit row into a CSV-friendly flat dict.

    Lists of titles / chunk snippets are pipe-joined so a spreadsheet
    can show them in one cell. The ``recovery`` block is rendered as
    a compact key=value string so the column count stays bounded.
    """
    sigtarget = row.silver_target or {}
    rec = row.recovery or {}
    top5_text: List[str] = []
    for c in row.top5_chunks:
        title = (c.get("retrieval_title") or c.get("title") or "").strip()
        section = " > ".join(c.get("section_path") or [])
        snippet = c.get("snippet") or ""
        top5_text.append(f"#{c.get('rank')} [{title} :: {section}] {snippet}")
    return {
        "audit_id": row.audit_id,
        "query_id": row.query_id,
        "query": row.query,
        "bucket": row.bucket,
        "silver_target_doc_id": str(sigtarget.get("doc_id") or ""),
        "silver_target_page_id": str(sigtarget.get("page_id") or ""),
        "silver_expected_title": str(row.expected_title or ""),
        "confidence_label": row.confidence_label,
        "failure_reasons": "|".join(row.failure_reasons),
        "recommended_action": row.recommended_action,
        "top1_title": str((row.top1 or {}).get("title") or ""),
        "top1_retrieval_title": str(
            (row.top1 or {}).get("retrieval_title") or ""
        ),
        "top1_doc_id": str((row.top1 or {}).get("doc_id") or ""),
        "top3_titles": "|".join(row.top3_titles),
        "top5_chunk_snippets": "\n".join(top5_text),
        "recovery_action": str(rec.get("recovery_action") or ""),
        "rewrite_mode": str(rec.get("rewrite_mode") or ""),
        "rewritten_query": str(rec.get("rewritten_query") or ""),
        "after_rank": str(rec.get("after_rank") if rec else ""),
        "after_in_top_k": str(rec.get("after_in_top_k") if rec else ""),
        "edge_case_tags": "|".join(row.edge_case_tags),
        "human_label": row.human_label or "",
        "human_notes": row.human_notes or "",
    }


# Column order for the CSV — frozen so reviewers can sort / pivot
# without column-numbers shifting between runs.
_CSV_COLUMNS: Tuple[str, ...] = (
    "audit_id",
    "query_id",
    "query",
    "bucket",
    "silver_target_doc_id",
    "silver_target_page_id",
    "silver_expected_title",
    "confidence_label",
    "failure_reasons",
    "recommended_action",
    "top1_title",
    "top1_retrieval_title",
    "top1_doc_id",
    "top3_titles",
    "top5_chunk_snippets",
    "recovery_action",
    "rewrite_mode",
    "rewritten_query",
    "after_rank",
    "after_in_top_k",
    "edge_case_tags",
    "human_label",
    "human_notes",
)


def write_audit_jsonl(rows: Sequence[AuditRow], path: Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            payload = row.to_dict()
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def write_audit_csv(rows: Sequence[AuditRow], path: Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=list(_CSV_COLUMNS), extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_for_csv(row))
    return path


def _md_row_block(row: AuditRow) -> List[str]:
    """Render one audit row as a self-contained markdown block."""
    sig = row.silver_target or {}
    rec = row.recovery or {}
    out: List[str] = []
    out.append(f"### {row.audit_id} — `{row.query_id}` [{row.bucket}]")
    out.append("")
    out.append(f"- **query:** {row.query}")
    out.append(
        f"- **silver target:** doc_id=`{sig.get('doc_id') or ''}` "
        f"page_id=`{sig.get('page_id') or ''}`"
    )
    out.append(f"- **expected_title (silver):** {row.expected_title or ''}")
    out.append(f"- **confidence_label:** {row.confidence_label}")
    out.append(
        f"- **failure_reasons:** {', '.join(row.failure_reasons) or '—'}"
    )
    out.append(f"- **recommended_action:** {row.recommended_action}")
    out.append(f"- **edge_case_tags:** {', '.join(row.edge_case_tags) or '—'}")
    if row.top1:
        out.append(
            f"- **top1 title:** {row.top1.get('title') or ''} "
            f"(retrieval_title=`{row.top1.get('retrieval_title') or ''}`, "
            f"doc_id=`{row.top1.get('doc_id') or ''}`)"
        )
    if row.top3_titles:
        out.append("- **top3 titles:**")
        for i, t in enumerate(row.top3_titles, start=1):
            out.append(f"    {i}. {t}")
    if row.top5_chunks:
        out.append("- **top5 chunk evidence:**")
        for c in row.top5_chunks:
            section = " > ".join(c.get("section_path") or [])
            title = c.get("retrieval_title") or c.get("title") or ""
            out.append(
                f"    - rank={c.get('rank')} title={title!r} "
                f"section=`{section}` section_type=`{c.get('section_type')}` "
                f"chunk_id=`{c.get('chunk_id')}`"
            )
            snippet = c.get("snippet") or ""
            if snippet:
                out.append(f"        > {snippet}")
    if rec:
        out.append("- **Phase 7.4 recovery attempt:**")
        out.append(
            f"    - recovery_action=`{rec.get('recovery_action')}` "
            f"rewrite_mode=`{rec.get('rewrite_mode') or ''}` "
            f"oracle_upper_bound={rec.get('oracle_upper_bound')}"
        )
        rewritten = rec.get("rewritten_query") or ""
        if rewritten:
            out.append(f"    - rewritten_query: {rewritten}")
        out.append(
            f"    - before_rank={rec.get('before_rank')} → "
            f"after_rank={rec.get('after_rank')} "
            f"(in_top_k: {rec.get('before_in_top_k')} → "
            f"{rec.get('after_in_top_k')})"
        )
    out.append("")
    out.append("**Human label** (one of: "
               + ", ".join(f"`{c}`" for c in HUMAN_LABEL_CHOICES) + "):")
    out.append("")
    out.append(f"- human_label: `{row.human_label or ''}`")
    out.append(f"- human_notes: {row.human_notes or ''}")
    out.append("")
    out.append("---")
    out.append("")
    return out


def render_audit_md(
    rows: Sequence[AuditRow],
    *,
    manifest: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render the audit seed bundle as a single markdown document.

    Layout: heading + silver disclaimer + manifest summary + one block
    per audit row + footer pointing the auditor at the JSONL/CSV.
    """
    lines: List[str] = []
    lines.append("# Phase 7.x — Human audit seed")
    lines.append("")
    lines.extend(SILVER_DISCLAIMER_LINES)
    lines.append("")
    lines.append(
        "Each row below carries the silver target, the model's verdict,"
        " and (when available) the Phase 7.4 recovery attempt. Fill the"
        " ``Human label`` field with one of:"
    )
    lines.append("")
    for choice in HUMAN_LABEL_CHOICES:
        lines.append(f"- ``{choice}``")
    lines.append("")
    lines.append(
        "Leave ``human_label`` blank for queries that need more review."
    )
    lines.append("")
    if manifest is not None:
        lines.append("## Audit manifest")
        lines.append("")
        lines.append(f"- n_audit_rows: **{manifest.get('n_audit_rows', 0)}**")
        lines.append(
            f"- n_unique_qids: **{manifest.get('n_unique_qids', 0)}**"
        )
        bq = manifest.get("bucket_quota") or {}
        if bq:
            lines.append(
                "- bucket_quota: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(bq.items())
                )
            )
        eq = manifest.get("edge_case_quota") or {}
        if eq:
            lines.append(
                "- edge_case_quota: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(eq.items())
                )
            )
        sg = manifest.get("strat_groups") or {}
        if sg:
            lines.append("- strat groups picked:")
            for k in sorted(sg.keys()):
                cell = sg[k]
                lines.append(
                    f"    - {k}: picked={cell.get('picked')} "
                    f"(quota={cell.get('quota')}, "
                    f"available={cell.get('available')})"
                )
        lines.append("")
    lines.append("## Audit rows")
    lines.append("")
    for row in rows:
        lines.extend(_md_row_block(row))
    return "\n".join(lines) + "\n"


def write_audit_md(
    rows: Sequence[AuditRow], path: Path,
    *, manifest: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.write_text(render_audit_md(rows, manifest=manifest), encoding="utf-8")
    return path


def write_audit_summary(
    manifest: Mapping[str, Any], path: Path,
) -> Path:
    """Persist the manifest + silver disclaimer block as a JSON summary."""
    path = Path(path)
    out = dict(manifest)
    out.update(silver_disclaimer_block())
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class AuditExportConfig:
    """Knobs the orchestrator reads.

    All defaults come from module-level constants so a CLI / test can
    construct the dataclass with no arguments and get the spec'd
    behaviour.
    """

    bucket_quota: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_BUCKET_QUOTA)
    )
    edge_case_quota: Mapping[str, int] = field(
        default_factory=_default_edge_case_quotas
    )
    snippet_max_chars: int = DEFAULT_SNIPPET_MAX_CHARS


def export_audit_bundle(
    *,
    confidence_jsonl: Path,
    out_dir: Path,
    recovery_attempts_jsonl: Optional[Path] = None,
    chunks_jsonl: Optional[Path] = None,
    config: Optional[AuditExportConfig] = None,
    bundle_basename: str = "phase7_human_audit_seed",
) -> Dict[str, Path]:
    """End-to-end: load → build → sample → write the audit seed bundle.

    Returns a dict of ``role → path`` so the CLI / test can verify what
    was emitted. The orchestrator is deterministic — same inputs +
    config produce byte-identical files.
    """
    cfg = config or AuditExportConfig()
    confidence_rows = load_jsonl(confidence_jsonl)

    recovery_by_qid: Dict[str, List[Mapping[str, Any]]] = {}
    if recovery_attempts_jsonl is not None:
        attempts = load_jsonl(recovery_attempts_jsonl)
        recovery_by_qid = index_recovery_attempts_by_qid(attempts)

    chunks_index: Optional[Dict[str, Dict[str, Any]]] = None
    if chunks_jsonl is not None:
        chunks_index = index_chunks_by_id(chunks_jsonl)

    rows = build_audit_rows(
        confidence_rows,
        recovery_by_qid=recovery_by_qid,
        chunks_index=chunks_index,
        snippet_max_chars=cfg.snippet_max_chars,
    )
    selected, manifest = sample_audit_rows(
        rows,
        bucket_quota=cfg.bucket_quota,
        edge_case_quota=cfg.edge_case_quota,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{bundle_basename}.jsonl"
    csv_path = out_dir / f"{bundle_basename}.csv"
    md_path = out_dir / f"{bundle_basename}.md"
    summary_path = out_dir / f"{bundle_basename}_summary.json"

    write_audit_jsonl(selected, jsonl_path)
    write_audit_csv(selected, csv_path)
    write_audit_md(selected, md_path, manifest=manifest)
    write_audit_summary(manifest, summary_path)

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "md": md_path,
        "summary": summary_path,
    }
