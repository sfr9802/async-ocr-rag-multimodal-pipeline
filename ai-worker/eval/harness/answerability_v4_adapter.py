"""Phase 7.7 — production v4 retrieval emit → answerability audit input adapter.

Bridges the production v4 retrieval pipeline's emit format with the
answerability audit's export builder. The two formats disagree on the
top-level container (``docs`` vs ``results``), the doc-level title
field (``title`` vs ``page_title``), the ``section_path`` shape
(list vs string), and on whether the raw chunk text travels with the
record at all (production retrieval emits chunk_id only; this adapter
resolves the raw text from a chunk fixture).

This is a pure transformation layer — it does not run any retrieval,
it does not score, and it does not promote any config. The output of
``adapt_v4_retrieval_record`` is the same dict shape that the
existing :mod:`scripts.export_answerability_audit` row / bundle
builders consume in non-v4 mode, so the export CLI only needs a thin
``--input-format v4-production`` switch on top of this module.

Input artifacts (v4 canonical only — Phase 7.x active path):

  * **Production retrieval emit JSONL** — for example::

        eval/reports/phase7/seeds/human_gold_seed_50_tuning/
            retrieval_baseline_retrieval_title_section_top10_*.jsonl

    Record shape::

        {
          "variant": "...",
          "query_id": "v4-llm-silver-010",
          "query": "...",
          "elapsed_ms": 24.1,
          "docs": [
            {"rank": 1, "chunk_id": "...", "page_id": "...",
             "title": "...", "section_path": [str, ...],
             "score": 0.658},
            ...
          ]
        }

  * **v4 chunk fixture** — **must be the file the production retrieval
    index ingests**, i.e. ``rag_chunks.jsonl``. The audit's raw text
    source MUST share a chunk_id namespace with the retrieval emit;
    otherwise every lookup fails. ``chunks_v4.jsonl`` is the
    structured canonical source upstream of the retrieval index but
    its chunk_ids are NOT the same namespace as ``rag_chunks.jsonl``
    (current corpus: 48,675 vs 135,602 chunk_ids, intersection = 0).
    ``chunks_v4.jsonl`` is therefore only acceptable when the
    retrieval emit itself was produced against ``chunks_v4.jsonl``
    (rare). Both shapes are auto-detected once selected:
    ``chunk_text`` is the rag_chunks raw-text field, ``text`` is the
    chunks_v4 raw-text field. Embedding-side fields
    (``embedding_text`` / ``text_for_embedding``) are *never* read —
    they include the title / section header prelude and would
    mislead a human reviewer about what the retriever actually
    returned.

  * **Gold / silver query metadata** — supports three v4 schemas
    (selected by file suffix + key probing):
      - ``queries_v4_llm_silver_500.jsonl`` (``query_id`` key,
        ``silver_expected_page_id`` / ``silver_expected_title`` /
        ``expected_section_path`` list).
      - ``queries_v4_silver_500.jsonl`` (``id`` key,
        ``expected_doc_ids`` list, ``expected_section_keywords``
        list — keywords are joined as a section_path proxy because
        the silver-500 generator does not store a single canonical
        section_path).
      - ``phase7_human_gold_seed_50.csv`` / ``.jsonl`` — gold-50
        seed; ``human_correct_*`` columns win when populated, else
        ``silver_expected_*``.

v4 chunk text rule:
  Raw chunk text comes from the chunk fixture (rag_chunks /
  chunks_v4) by chunk_id lookup. The retrieval emit does NOT carry
  chunk_text and MUST NOT be relied on for it. Embedding-augmented
  text is forbidden. ``_resolve_chunk_text`` only reads
  ``chunk_text`` / ``text`` keys; everything else is ignored.

Section-path rule:
  v4 chunks / retrieval emit / silver queries all carry section_path
  as a list of segments. The adapter joins them with
  :data:`SECTION_PATH_JOINER` (`` > ``) so the audit row always
  carries a single human-readable string. Strings pass through
  unchanged; ``None`` / empty list / missing ⇒ empty string.

Missing-chunk policy:
  ``adapt_v4_retrieval_record`` raises :class:`V4AdapterError` if a
  doc references a chunk_id absent from the fixture (default
  ``on_missing_chunk='error'``). Set ``on_missing_chunk='collect'``
  to keep going and return the missing tuples in the record's
  ``_missing_chunks`` field. **``collect`` is for debugging /
  triage only and MUST NOT be used to produce a labelling export
  for human review** — the audit's import validator rejects empty
  ``chunk_text`` cells at scoring time, so a labelling pass over a
  ``collect`` export wastes the reviewer's effort. Production
  labelling pipelines must use ``on_missing_chunk='error'`` so a
  missing chunk_id surfaces immediately instead of silently
  producing a row the validator will throw out later.

Metadata mismatch policy:
  When a retrieval doc's ``page_id`` / ``title`` / ``section_path``
  disagrees with the chunk fixture for the same ``chunk_id``, the
  default behaviour is to raise. This catches stale or malformed
  retrieval emits before they become plausible-looking but incorrect
  page_hit / section_hit rows. ``on_metadata_mismatch='collect'`` is
  available for triage and emits chunk-fixture metadata in the output.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union,
)


log = logging.getLogger(__name__)


SECTION_PATH_JOINER: str = " > "


# Embedding-side fields that MUST NOT be substituted for raw chunk
# text — listed explicitly so future maintainers see the contract.
_EMBEDDING_ONLY_KEYS: Tuple[str, ...] = (
    "embedding_text",
    "text_for_embedding",
)


class V4AdapterError(ValueError):
    """Raised when a v4 input artifact violates this adapter's contract.

    Distinct from
    :class:`eval.harness.answerability_audit.AnswerabilityValidationError`
    so a CLI caller can tell adapter / fixture problems (chunk_id
    missing, retrieval emit malformed) apart from labelling problems
    (chunk_text empty, label out of range) in the audit harness
    itself.
    """


# ---------------------------------------------------------------------------
# Source records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V4ChunkRef:
    """One chunk-fixture record, post-normalisation.

    ``page_title`` is sourced from chunks_v4's ``page_title`` key or
    rag_chunks's ``title`` key. ``section_path`` is the joined string,
    not the original list. ``chunk_text`` is raw text only.
    """

    chunk_id: str
    page_id: str
    page_title: str
    section_id: str
    section_path: str
    chunk_text: str


@dataclass(frozen=True)
class V4GoldRef:
    """One gold-side record, post-normalisation.

    Empty strings are valid for any field — the audit harness treats
    empty gold as "no gold for this field" and records all-False hit
    columns. Section_path is joined.
    """

    query_id: str
    page_id: str
    page_title: str
    section_id: str
    section_path: str


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, no global state)
# ---------------------------------------------------------------------------


def normalise_v4_section_path(value: Any) -> str:
    """Render a section_path value as a single human-readable string.

    v4 ``chunks_v4.jsonl`` / ``rag_chunks.jsonl`` and the production
    retrieval emitter store ``section_path`` as a list of segments
    (``["음악", "주제가", "OP"]``); silver queries also use list form.
    Pre-joined strings (``"음악 > 주제가 > OP"``) pass through. Any
    other type is best-effort coerced via ``str()``. ``None``,
    empty list, and empty string all collapse to ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return SECTION_PATH_JOINER.join(
            str(seg) for seg in value if str(seg)
        )
    return str(value)


def _resolve_chunk_text(record: Mapping[str, Any]) -> str:
    """Return the raw chunk text from a chunk-fixture record.

    Order of preference:
      1. ``chunk_text`` (rag_chunks.jsonl form).
      2. ``text`` (chunks_v4.jsonl form).

    Embedding-side keys (``embedding_text``, ``text_for_embedding``)
    are intentionally NOT read — substituting them would mislead a
    human reviewer about what the retriever returned. An empty result
    is allowed at this layer; the missing-chunk policy in
    :func:`adapt_v4_retrieval_record` is what enforces non-empty text
    in the labelling pipeline.
    """
    raw = record.get("chunk_text")
    if raw is None or raw == "":
        raw = record.get("text")
    return str(raw or "")


def _metadata_mismatches(
    *,
    emit_page_id: str,
    emit_page_title: str,
    emit_section_path: str,
    chunk_ref: V4ChunkRef,
) -> List[str]:
    """Return non-empty retrieval-vs-fixture metadata disagreements."""
    checks = (
        ("page_id", emit_page_id, chunk_ref.page_id),
        ("page_title", emit_page_title, chunk_ref.page_title),
        ("section_path", emit_section_path, chunk_ref.section_path),
    )
    out: List[str] = []
    for field, emit_value, fixture_value in checks:
        emit = str(emit_value or "").strip()
        fixture = str(fixture_value or "").strip()
        if emit and fixture and emit != fixture:
            out.append(
                f"{field}: emit={emit!r} fixture={fixture!r}"
            )
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_v4_chunk_lookup(path: Union[str, Path]) -> Dict[str, V4ChunkRef]:
    """Load a chunk fixture (rag_chunks.jsonl or chunks_v4.jsonl) into
    a chunk_id → :class:`V4ChunkRef` mapping.

    Both shapes are accepted in one pass:
      * ``page_id`` (chunks_v4) or ``doc_id`` (rag_chunks) ⇒ ``page_id``
      * ``page_title`` (chunks_v4) or ``title`` (rag_chunks) ⇒
        ``page_title``
      * ``section_id`` is rag_chunks-only; chunks_v4 records produce
        an empty ``section_id`` (the audit harness already tolerates
        empty section_id).
      * raw text comes from :func:`_resolve_chunk_text` —
        embedding-side fields are ignored.

    Duplicate chunk_ids overwrite (last wins) with a debug log;
    chunks_v4 / rag_chunks are deduped at corpus build time so this
    only fires on caller error.
    """
    src = Path(path)
    if not src.exists():
        raise V4AdapterError(f"chunk fixture not found: {src}")
    out: Dict[str, V4ChunkRef] = {}
    n_dup = 0
    with src.open("r", encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as ex:
                raise V4AdapterError(
                    f"{src}: invalid JSON on line {line_no}: {ex}"
                ) from ex
            if not isinstance(rec, dict):
                raise V4AdapterError(
                    f"{src}: line {line_no} must be an object, "
                    f"got {type(rec).__name__}"
                )
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                raise V4AdapterError(
                    f"{src}: line {line_no} missing 'chunk_id'"
                )
            chunk_id = str(chunk_id)
            page_id = str(rec.get("page_id") or rec.get("doc_id") or "")
            page_title = str(rec.get("page_title") or rec.get("title") or "")
            section_id = str(rec.get("section_id") or "")
            section_path = normalise_v4_section_path(rec.get("section_path"))
            chunk_text = _resolve_chunk_text(rec)
            if chunk_id in out:
                n_dup += 1
            out[chunk_id] = V4ChunkRef(
                chunk_id=chunk_id,
                page_id=page_id,
                page_title=page_title,
                section_id=section_id,
                section_path=section_path,
                chunk_text=chunk_text,
            )
    if n_dup:
        log.debug(
            "load_v4_chunk_lookup(%s): %d duplicate chunk_id(s) "
            "(last-wins)", src, n_dup,
        )
    log.info(
        "loaded %d unique chunk_id(s) from %s", len(out), src,
    )
    return out


def _gold_from_csv_row(row: Mapping[str, Any]) -> Optional[V4GoldRef]:
    """Build a :class:`V4GoldRef` from a phase7_human_gold_seed_50 CSV row.

    Prefers ``human_correct_*`` columns when populated (the reviewer's
    final answer), falls back to ``silver_expected_*`` otherwise. The
    gold-50 seed does not carry a section_path / section_id, so both
    fields are left empty — the audit's section-hit metric will read
    "no gold section" and skip section-hit accounting for these
    queries, which matches the current Phase 7.x convention.
    """
    qid = str(row.get("query_id") or "").strip()
    if not qid:
        return None
    human_pid = str(row.get("human_correct_page_id") or "").strip()
    human_title = str(row.get("human_correct_title") or "").strip()
    silver_pid = str(row.get("silver_expected_page_id") or "").strip()
    silver_title = str(row.get("silver_expected_title") or "").strip()
    return V4GoldRef(
        query_id=qid,
        page_id=human_pid or silver_pid,
        page_title=human_title or silver_title,
        section_id="",
        section_path="",
    )


def _gold_from_jsonl_record(rec: Mapping[str, Any]) -> Optional[V4GoldRef]:
    """Build a :class:`V4GoldRef` from one silver / llm-silver JSONL record.

    Supports two key conventions:

      * ``query_id`` key (llm-silver-500 form) — gold via
        ``silver_expected_page_id`` / ``silver_expected_title`` /
        ``expected_section_path`` (list).
      * ``id`` key (silver-500 form) — gold via
        ``expected_doc_ids[0]`` / ``expected_section_keywords``. The
        silver-500 generator does not record a single canonical
        section path; the keyword list is joined as a section_path
        proxy. Page title is empty (silver-500 records do not carry
        a title field at the top level).

    Returns ``None`` for records with no usable id so the caller can
    decide whether to skip or raise.
    """
    if "query_id" in rec:
        qid = str(rec.get("query_id") or "").strip()
        if not qid:
            return None
        return V4GoldRef(
            query_id=qid,
            page_id=str(rec.get("silver_expected_page_id") or ""),
            page_title=str(rec.get("silver_expected_title") or ""),
            section_id="",
            section_path=normalise_v4_section_path(
                rec.get("expected_section_path"),
            ),
        )
    if "id" in rec:
        qid = str(rec.get("id") or "").strip()
        if not qid:
            return None
        expected_docs = rec.get("expected_doc_ids") or []
        page_id = str(expected_docs[0]) if expected_docs else ""
        return V4GoldRef(
            query_id=qid,
            page_id=page_id,
            page_title="",
            section_id="",
            section_path=normalise_v4_section_path(
                rec.get("expected_section_keywords"),
            ),
        )
    return None


def load_v4_gold_lookup(path: Union[str, Path]) -> Dict[str, V4GoldRef]:
    """Load gold / silver query metadata into query_id → :class:`V4GoldRef`.

    Auto-detects format by suffix:
      * ``.csv`` ⇒ phase7_human_gold_seed_50 form.
      * ``.jsonl`` ⇒ key probing per record (query_id vs id).

    Records the file lacks a usable id for are silently skipped with
    a debug log; truly malformed JSON / CSV raises
    :class:`V4AdapterError`.
    """
    src = Path(path)
    if not src.exists():
        raise V4AdapterError(f"gold/silver fixture not found: {src}")
    suffix = src.suffix.lower()
    out: Dict[str, V4GoldRef] = {}
    skipped = 0
    if suffix == ".csv":
        with src.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                ref = _gold_from_csv_row(row)
                if ref is None:
                    skipped += 1
                    continue
                out[ref.query_id] = ref
    elif suffix == ".jsonl":
        with src.open("r", encoding="utf-8") as fp:
            for line_no, raw_line in enumerate(fp, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise V4AdapterError(
                        f"{src}: invalid JSON on line {line_no}: {ex}"
                    ) from ex
                if not isinstance(rec, dict):
                    raise V4AdapterError(
                        f"{src}: line {line_no} must be an object, "
                        f"got {type(rec).__name__}"
                    )
                ref = _gold_from_jsonl_record(rec)
                if ref is None:
                    skipped += 1
                    continue
                out[ref.query_id] = ref
    else:
        raise V4AdapterError(
            f"unsupported gold/silver suffix {suffix!r} on {src}; "
            f"expected .csv or .jsonl"
        )
    if skipped:
        log.debug(
            "load_v4_gold_lookup(%s): skipped %d record(s) with no usable id",
            src, skipped,
        )
    log.info(
        "loaded %d gold record(s) from %s", len(out), src,
    )
    return out


# ---------------------------------------------------------------------------
# Per-record adaptation
# ---------------------------------------------------------------------------


def adapt_v4_retrieval_record(
    record: Mapping[str, Any],
    *,
    chunk_lookup: Mapping[str, V4ChunkRef],
    gold_lookup: Mapping[str, V4GoldRef],
    on_missing_chunk: str = "error",
    on_metadata_mismatch: str = "error",
) -> Dict[str, Any]:
    """Convert one production retrieval emit record into audit input shape.

    Output dict matches the schema documented at the top of
    :mod:`scripts.export_answerability_audit` (the row / bundle export
    builders' expected input):

        {
          "query_id": str,
          "query": str,
          "gold": {"page_id", "page_title", "section_id", "section_path"},
          "results": [
            {"rank", "chunk_id", "page_id", "page_title",
             "section_id", "section_path", "chunk_text"},
            ...
          ],
          "_missing_chunks": [(query_id, chunk_id, rank), ...],
        }

    The export builders ignore extra keys (``DictWriter`` is built
    with ``extrasaction='ignore'``) so ``_missing_chunks`` is safe to
    pass through.

    Parameters:
      record: one decoded production retrieval emit JSONL record.
      chunk_lookup: chunk_id → V4ChunkRef.
      gold_lookup: query_id → V4GoldRef.
      on_missing_chunk: ``"error"`` (default) raises
        :class:`V4AdapterError`; ``"collect"`` records the tuple and
        emits an empty chunk_text cell so the caller can inspect
        what happened. The audit's import validator will still
        reject the empty cell at scoring time — collect mode is for
        triage only.
      on_metadata_mismatch: ``"error"`` (default) raises when
        retrieval emit metadata disagrees with the chunk fixture for
        the same chunk_id; ``"collect"`` records the mismatch and
        emits fixture metadata.
    """
    if on_missing_chunk not in {"error", "collect"}:
        raise V4AdapterError(
            f"on_missing_chunk must be 'error' or 'collect', "
            f"got {on_missing_chunk!r}"
        )
    if on_metadata_mismatch not in {"error", "collect"}:
        raise V4AdapterError(
            f"on_metadata_mismatch must be 'error' or 'collect', "
            f"got {on_metadata_mismatch!r}"
        )
    qid = record.get("query_id")
    if not qid:
        raise V4AdapterError(
            f"retrieval record missing 'query_id': {record!r}"
        )
    qid = str(qid)
    query_text = str(record.get("query") or "")
    docs = record.get("docs")
    if docs is None:
        raise V4AdapterError(
            f"retrieval record {qid!r} missing 'docs' "
            "(production v4 retrieval emit uses 'docs', not 'results')"
        )
    if not isinstance(docs, list):
        raise V4AdapterError(
            f"retrieval record {qid!r}: 'docs' must be a list, "
            f"got {type(docs).__name__}"
        )

    gold_ref = gold_lookup.get(qid)
    if gold_ref is None:
        gold_block: Dict[str, str] = {
            "page_id": "", "page_title": "",
            "section_id": "", "section_path": "",
        }
    else:
        gold_block = {
            "page_id": gold_ref.page_id,
            "page_title": gold_ref.page_title,
            "section_id": gold_ref.section_id,
            "section_path": gold_ref.section_path,
        }

    results: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str, int]] = []
    metadata_mismatches: List[Tuple[str, str, int, str]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            raise V4AdapterError(
                f"{qid}: doc must be an object, got {type(doc).__name__}"
            )
        rank_raw = doc.get("rank")
        if rank_raw is None:
            raise V4AdapterError(
                f"{qid}: doc missing 'rank': {doc!r}"
            )
        try:
            rank = int(rank_raw)
        except (TypeError, ValueError) as ex:
            raise V4AdapterError(
                f"{qid}: doc 'rank' must be int-like, got {rank_raw!r}"
            ) from ex
        chunk_id = str(doc.get("chunk_id") or "")
        page_id = str(doc.get("page_id") or "")
        # In v4 production retrieval emit, the doc-level 'title' field
        # carries the page_title (verified against pages_v4.jsonl).
        page_title = str(doc.get("title") or "")
        section_path = normalise_v4_section_path(doc.get("section_path"))

        chunk_ref = chunk_lookup.get(chunk_id) if chunk_id else None
        if chunk_ref is None:
            if on_missing_chunk == "error":
                raise V4AdapterError(
                    f"{qid} (rank={rank}): chunk_id {chunk_id!r} not "
                    "found in chunk fixture"
                )
            missing.append((qid, chunk_id, rank))
            chunk_text = ""
            section_id = ""
        else:
            mismatches = _metadata_mismatches(
                emit_page_id=page_id,
                emit_page_title=page_title,
                emit_section_path=section_path,
                chunk_ref=chunk_ref,
            )
            if mismatches:
                message = "; ".join(mismatches)
                if on_metadata_mismatch == "error":
                    raise V4AdapterError(
                        f"{qid} (rank={rank}, chunk_id={chunk_id!r}): "
                        f"retrieval metadata mismatch: {message}"
                    )
                metadata_mismatches.append((qid, chunk_id, rank, message))
            chunk_text = chunk_ref.chunk_text
            section_id = chunk_ref.section_id
            page_id = chunk_ref.page_id or page_id
            page_title = chunk_ref.page_title or page_title
            section_path = chunk_ref.section_path or section_path

        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "page_id": page_id,
            "page_title": page_title,
            "section_id": section_id,
            "section_path": section_path,
            "chunk_text": chunk_text,
        })

    out: Dict[str, Any] = {
        "query_id": qid,
        "query": query_text,
        "gold": gold_block,
        "results": results,
    }
    if missing:
        # Surface to the CLI driver but invisible to the export
        # builder (extrasaction='ignore' on the writer).
        out["_missing_chunks"] = missing
    if metadata_mismatches:
        out["_metadata_mismatches"] = metadata_mismatches
    return out


# ---------------------------------------------------------------------------
# JSONL driver
# ---------------------------------------------------------------------------


def adapt_v4_retrieval_jsonl(
    retrieval_path: Union[str, Path],
    *,
    chunk_lookup: Optional[Mapping[str, V4ChunkRef]] = None,
    gold_lookup: Optional[Mapping[str, V4GoldRef]] = None,
    chunks_path: Optional[Union[str, Path]] = None,
    gold_path: Optional[Union[str, Path]] = None,
    on_missing_chunk: str = "error",
    on_metadata_mismatch: str = "error",
) -> Iterator[Dict[str, Any]]:
    """Stream-adapt a v4 retrieval emit JSONL file.

    Either supply ``chunk_lookup`` / ``gold_lookup`` directly (already
    loaded — useful when adapting multiple retrieval files against the
    same fixtures) or ``chunks_path`` / ``gold_path`` (the loader
    reads them once before yielding). Mixing is allowed: a passed
    lookup wins over the corresponding path.

    Yields one audit-input dict per non-blank, non-comment line.
    Raises :class:`V4AdapterError` for malformed JSON or empty
    retrieval files.
    """
    if chunk_lookup is None:
        if chunks_path is None:
            raise V4AdapterError(
                "adapt_v4_retrieval_jsonl: provide chunk_lookup "
                "or chunks_path"
            )
        chunk_lookup = load_v4_chunk_lookup(chunks_path)
    if gold_lookup is None:
        if gold_path is None:
            raise V4AdapterError(
                "adapt_v4_retrieval_jsonl: provide gold_lookup "
                "or gold_path"
            )
        gold_lookup = load_v4_gold_lookup(gold_path)

    src = Path(retrieval_path)
    if not src.exists():
        raise V4AdapterError(
            f"retrieval emit file not found: {src}"
        )
    n_yielded = 0
    with src.open("r", encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as ex:
                raise V4AdapterError(
                    f"{src}: invalid JSON on line {line_no}: {ex}"
                ) from ex
            if not isinstance(rec, dict):
                raise V4AdapterError(
                    f"{src}: line {line_no} must be an object, "
                    f"got {type(rec).__name__}"
                )
            yield adapt_v4_retrieval_record(
                rec,
                chunk_lookup=chunk_lookup,
                gold_lookup=gold_lookup,
                on_missing_chunk=on_missing_chunk,
                on_metadata_mismatch=on_metadata_mismatch,
            )
            n_yielded += 1
    log.info(
        "adapt_v4_retrieval_jsonl(%s): yielded %d audit record(s)",
        src, n_yielded,
    )


__all__ = [
    "SECTION_PATH_JOINER",
    "V4AdapterError",
    "V4ChunkRef",
    "V4GoldRef",
    "adapt_v4_retrieval_jsonl",
    "adapt_v4_retrieval_record",
    "load_v4_chunk_lookup",
    "load_v4_gold_lookup",
    "normalise_v4_section_path",
]
