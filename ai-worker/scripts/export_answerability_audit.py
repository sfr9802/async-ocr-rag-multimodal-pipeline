"""Phase 7.7 — emit a labelling CSV / JSONL from a retrieval bundle.

The harness in ``eval.harness.answerability_audit`` does the heavy
lifting: it builds export rows from ``GoldRef`` + ``RetrievedRef``
records and writes them as CSV / JSONL with three blank columns
(``label_*`` / ``*_flags`` / ``notes``) for a human reviewer to fill
in. This CLI is the thin wiring that translates a generic JSONL
retrieval bundle into those records.

Three modes are supported:

  * ``--mode row`` (default) — Phase 7.7 row-level export. One row
    per ``(query, variant, rank)``; the reviewer labels each chunk
    individually for evidence-quality diagnosis.
  * ``--mode bundle`` — Phase 7.7.1 bundle-level export. One row per
    ``(query, variant, top_k)`` with the top-k chunks pre-rendered
    into ``context_bundle_text``; the reviewer labels the *bundle*
    (top-k context as a whole) for answerability judgment.
  * ``--mode bundle-sample`` — Phase 7.7.1 sampling helper. Reads a
    previously-exported bundle CSV / JSONL, samples N distinct
    query_ids deterministically, and writes a smaller file for human
    pilot review. Does NOT fill in labels.
  * ``--mode row-sample`` — Phase 7.7 row-level sampling helper.
    Mirrors ``bundle-sample`` but consumes a previously-exported
    *row* CSV / JSONL (one row per ``(query, rank)``) and writes the
    sampled rows back using :data:`EXPORT_COLUMNS`. Does NOT fill in
    labels — strictly a human step.

Expected input shape for ``row`` / ``bundle`` modes (one JSON object
per line in ``--retrieval-results-path``):

    {
      "query_id": "v4-silver-0001",
      "query": "...",
      "gold": {
        "page_id": "...",
        "page_title": "...",
        "section_id": "...",
        "section_path": "..."
      },
      "results": [
        {
          "rank": 1,
          "chunk_id": "...",
          "page_id": "...",
          "page_title": "...",
          "section_id": "...",
          "section_path": "...",
          "chunk_text": "..."
        },
        ...
      ]
    }

All gold and result fields are optional except ``query_id``, ``query``,
and ``rank``; missing fields default to empty strings.

v4 canonical alignment notes (Phase 7.7 audit, namu-v4 corpus):

  * ``section_path`` may arrive as either a string ("음악 > 주제가 > OP")
    or as a list of segments (``["음악", "주제가", "OP"]``). Lists
    and tuples are joined with `` > `` so the labelling file always
    carries a single human-readable string. v4 ``chunks_v4.jsonl`` /
    ``rag_chunks.jsonl`` and the production retrieval emitter all
    produce list-form section paths; older fixtures used pre-joined
    strings. Both round-trip through this loader.
  * ``chunk_text`` is the **raw** chunk text the reviewer reads — not
    the embedding-augmented text. Acceptable source keys, in order
    of preference: ``chunk_text`` (rag_chunks form), ``text``
    (chunks_v4 form). Embedding-side fields (``text_for_embedding`` /
    ``embedding_text``) MUST NOT be substituted, as they include the
    title / section header prelude and would mislead the reviewer.
  * The production v4 retrieval pipeline emits records keyed
    ``docs`` (not ``results``) and uses ``title`` (not
    ``page_title``); ``--input-format v4-production`` converts that
    emit format and resolves raw chunk_text from ``rag_chunks.jsonl``.
    ``chunks_v4.jsonl`` is rejected for production emits because it
    uses a different chunk_id namespace.

Output format follows the suffix of ``--out-path``: ``.csv`` writes a
spreadsheet-friendly file, ``.jsonl`` writes a text-editor-friendly
file. Both round-trip through ``score_answerability_audit``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from eval.harness.answerability_audit import (
    BUNDLE_EXPORT_COLUMNS,
    DEFAULT_BUNDLE_TRUNCATE_CHARS,
    DEFAULT_SAMPLE_QUERY_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TOP_K_SET,
    EXPORT_COLUMNS,
    GoldRef,
    RetrievedRef,
    build_bundle_export_rows,
    build_export_rows,
    sample_bundle_records,
    write_bundle_export_csv,
    write_bundle_export_jsonl,
    write_export_csv,
    write_export_jsonl,
)
from eval.harness.answerability_v4_adapter import (
    V4AdapterError,
    adapt_v4_retrieval_jsonl,
    load_v4_chunk_lookup,
    load_v4_gold_lookup,
)


log = logging.getLogger("scripts.export_answerability_audit")


SUPPORTED_MODES = ("row", "bundle", "bundle-sample", "row-sample")
SUPPORTED_INPUT_FORMATS = ("audit", "v4-production")
SUPPORTED_ON_MISSING_CHUNK = ("error", "collect")


# ---------------------------------------------------------------------------
# Common JSONL ingestion helpers (shared by row + bundle modes).
# ---------------------------------------------------------------------------


SECTION_PATH_JOINER: str = " > "


def _normalise_section_path(value: Any) -> str:
    """Render a v4 list/tuple section_path as a single string.

    v4 ``chunks_v4.jsonl`` / ``rag_chunks.jsonl`` and the production
    retrieval emitter store ``section_path`` as a list of segments
    (``["음악", "주제가", "OP"]``); older fixtures used a pre-joined
    string (``"음악 > 주제가 > OP"``). Strings pass through unchanged;
    lists / tuples are joined with :data:`SECTION_PATH_JOINER` so the
    labelling file always carries the human-readable form. Falsy
    inputs collapse to an empty string.
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
    """Return the raw chunk text from a v4 retrieval record.

    Accepts the ``rag_chunks.jsonl``-style ``chunk_text`` key first,
    falling back to the ``chunks_v4.jsonl``-style ``text`` key.
    Embedding-side fields (``text_for_embedding`` / ``embedding_text``)
    are intentionally NOT used — they include the title / section
    header prelude and would mislead a human reviewer about what the
    retriever actually returned.
    """
    raw = record.get("chunk_text")
    if raw is None or raw == "":
        raw = record.get("text")
    return str(raw or "")


def _gold_from_record(record: Dict[str, Any]) -> GoldRef:
    """Coerce the ``gold`` sub-object into :class:`GoldRef`.

    Tolerant of missing keys — the harness treats empty strings as
    "no gold for this field" when computing page / section hit, so a
    minimally-annotated input just yields all-False hit columns.
    Accepts ``section_path`` as either a string or a list/tuple of
    segments (see :func:`_normalise_section_path`).
    """
    gold_payload = record.get("gold") or {}
    return GoldRef(
        page_id=str(gold_payload.get("page_id") or ""),
        page_title=str(gold_payload.get("page_title") or ""),
        section_id=str(gold_payload.get("section_id") or ""),
        section_path=_normalise_section_path(
            gold_payload.get("section_path")
        ),
    )


def _retrieved_from_record(record: Dict[str, Any]) -> RetrievedRef:
    """Coerce one entry of ``results`` into :class:`RetrievedRef`.

    ``rank`` is required (a result without a rank is meaningless for
    a top-k labelling pass); everything else defaults to empty string.
    ``section_path`` may be a string or a list/tuple of segments;
    ``chunk_text`` falls back to ``text`` (chunks_v4 form) when absent.
    """
    if "rank" not in record:
        raise ValueError(
            f"retrieval result missing 'rank': {record!r}"
        )
    return RetrievedRef(
        rank=int(record["rank"]),
        chunk_id=str(record.get("chunk_id") or ""),
        page_id=str(record.get("page_id") or ""),
        page_title=str(record.get("page_title") or ""),
        section_id=str(record.get("section_id") or ""),
        section_path=_normalise_section_path(
            record.get("section_path")
        ),
        chunk_text=_resolve_chunk_text(record),
    )


def _load_retrieval_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read the input JSONL into a list of dicts, skipping blanks /
    comment lines (mirrors :func:`io_utils.load_jsonl`).
    """
    if not path.exists():
        raise FileNotFoundError(f"retrieval results file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(
                    f"invalid JSON on line {line_no} of {path}: {ex}"
                ) from ex
            if not isinstance(obj, dict):
                raise ValueError(
                    f"line {line_no} of {path} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


def _parse_top_k_set(value: str) -> List[int]:
    """Parse a comma-separated top_k spec like ``"1,3,5"`` into a list.

    Empty / whitespace cells are skipped. Non-positive values raise
    ``argparse.ArgumentTypeError`` so the CLI surfaces the typo
    immediately rather than producing zero-length bundles silently.
    """
    if not value:
        raise argparse.ArgumentTypeError(
            "--top-k-set cannot be empty"
        )
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            f"--top-k-set has no positive ints: {value!r}"
        )
    out: List[int] = []
    for p in parts:
        try:
            k = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--top-k-set value {p!r} is not an int"
            )
        if k <= 0:
            raise argparse.ArgumentTypeError(
                f"--top-k-set value {k} is not positive"
            )
        if k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# Input dispatch — picks ``audit`` (existing schema) or ``v4-production``
# (production retrieval emit + chunk fixture + gold/silver fixture).
# ---------------------------------------------------------------------------


def _resolve_retrieval_records(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Materialise the retrieval records the row / bundle modes consume.

    For ``--input-format audit`` (default) this is just
    :func:`_load_retrieval_jsonl` — the file is already in the
    audit-shaped schema.

    For ``--input-format v4-production`` this drives
    :mod:`eval.harness.answerability_v4_adapter`, which loads the
    chunk + gold fixtures once and adapts each retrieval emit record
    in-place. Missing-chunk policy is taken from
    ``args.on_missing_chunk``.

    Materialising up-front (rather than streaming) keeps the
    downstream code paths identical between modes and lets the row /
    bundle runners report ``len(records)`` without re-reading the file.
    """
    in_path = Path(args.retrieval_results_path)
    if not in_path.exists():
        raise SystemExit(
            f"--retrieval-results-path not found: {in_path}"
        )

    fmt = getattr(args, "input_format", "audit")
    if fmt == "audit":
        return _load_retrieval_jsonl(in_path)

    if fmt == "v4-production":
        if not args.chunks_path:
            raise SystemExit(
                "--input-format v4-production requires --chunks-path "
                "(rag_chunks.jsonl)"
            )
        if not args.gold_path:
            raise SystemExit(
                "--input-format v4-production requires --gold-path "
                "(silver / llm-silver jsonl or gold-50 csv/jsonl)"
            )
        log.info(
            "[adapter] loading chunk fixture %s + gold/silver %s",
            args.chunks_path, args.gold_path,
        )
        try:
            _validate_v4_production_chunks_path(args.chunks_path)
            chunk_lookup = load_v4_chunk_lookup(args.chunks_path)
            gold_lookup = load_v4_gold_lookup(args.gold_path)
            adapted = list(adapt_v4_retrieval_jsonl(
                in_path,
                chunk_lookup=chunk_lookup,
                gold_lookup=gold_lookup,
                on_missing_chunk=args.on_missing_chunk,
                on_metadata_mismatch=args.on_metadata_mismatch,
            ))
        except V4AdapterError as ex:
            raise SystemExit(f"v4 adapter error: {ex}")
        n_with_missing = sum(
            1 for r in adapted if r.get("_missing_chunks")
        )
        n_missing_total = sum(
            len(r.get("_missing_chunks") or []) for r in adapted
        )
        if n_missing_total:
            log.warning(
                "[adapter] %d retrieval record(s) had unresolved "
                "chunk_ids (total %d missing); on_missing_chunk=%s",
                n_with_missing, n_missing_total, args.on_missing_chunk,
            )
        n_metadata_mismatch = sum(
            len(r.get("_metadata_mismatches") or []) for r in adapted
        )
        if n_metadata_mismatch:
            log.warning(
                "[adapter] %d retrieval doc(s) had metadata mismatches; "
                "on_metadata_mismatch=%s",
                n_metadata_mismatch, args.on_metadata_mismatch,
            )
        n_no_gold = sum(
            1 for r in adapted
            if not (r.get("gold") or {}).get("page_id")
        )
        if n_no_gold:
            message = (
                f"[adapter] {n_no_gold}/{len(adapted)} retrieval record(s) "
                "had no gold page_id"
            )
            if not args.allow_missing_gold:
                raise SystemExit(
                    message
                    + "; pass --allow-missing-gold only for triage exports"
                )
            log.warning(
                "%s (page_hit will be all-False for those queries)",
                message,
            )
        return adapted

    raise SystemExit(
        f"--input-format must be one of {SUPPORTED_INPUT_FORMATS}, "
        f"got {fmt!r}"
    )


def _validate_v4_production_chunks_path(path: Path) -> None:
    """Reject structured ``chunks_v4.jsonl`` for production retrieval joins.

    The production Phase 7 retrieval index is built from
    ``rag_chunks.jsonl``. ``chunks_v4.jsonl`` is a canonical structured
    source artifact, but its chunk ids live in a different namespace.
    Letting a labelling export join production emits against it would
    either fail every lookup or, worse, make a reviewer audit the wrong
    fixture family.
    """
    if Path(path).name == "chunks_v4.jsonl":
        raise V4AdapterError(
            "--input-format v4-production must use rag_chunks.jsonl "
            "for --chunks-path. chunks_v4.jsonl is the structured "
            "source artifact and does not share the production retrieval "
            "chunk_id namespace."
        )


# ---------------------------------------------------------------------------
# Mode runners.
# ---------------------------------------------------------------------------


def _run_row_mode(args: argparse.Namespace) -> int:
    """Phase 7.7 row-level export — one CSV row per (query, variant, rank)."""
    out_path = Path(args.out_path)
    suffix = out_path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--out-path must end in .csv or .jsonl, got {out_path.name!r}"
        )

    log.info(
        "[row] loading retrieval results "
        "(input_format=%s, variant=%s, top_k=%d)",
        args.input_format, args.variant_name, args.top_k,
    )
    records = _resolve_retrieval_records(args)

    all_rows = []
    n_skipped = 0
    for record in records:
        query_id = record.get("query_id")
        query_text = record.get("query")
        if not query_id or not query_text:
            n_skipped += 1
            log.warning(
                "skipping record with missing query_id / query: %r", record,
            )
            continue
        gold = _gold_from_record(record)
        results = record.get("results") or []
        retrieved = [_retrieved_from_record(r) for r in results]
        all_rows.extend(build_export_rows(
            query_id=str(query_id),
            query=str(query_text),
            variant_name=str(args.variant_name),
            gold=gold,
            retrieved=retrieved,
            top_k=int(args.top_k),
        ))

    log.info(
        "[row] built %d rows from %d records (skipped=%d)",
        len(all_rows), len(records), n_skipped,
    )
    if suffix == ".csv":
        write_export_csv(out_path, all_rows)
    else:
        write_export_jsonl(out_path, all_rows)
    log.info("[row] done. wrote %s", out_path)
    return 0


def _run_bundle_mode(args: argparse.Namespace) -> int:
    """Phase 7.7.1 bundle-level export — one row per (query, variant, top_k)."""
    out_path = Path(args.out_path)
    suffix = out_path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--out-path must end in .csv or .jsonl, got {out_path.name!r}"
        )

    top_k_set = list(args.top_k_set)
    log.info(
        "[bundle] loading retrieval results "
        "(input_format=%s, variant=%s, top_k_set=%s, truncate_chars=%s)",
        args.input_format, args.variant_name, top_k_set, args.truncate_chars,
    )
    records = _resolve_retrieval_records(args)

    all_rows = []
    n_skipped = 0
    for record in records:
        query_id = record.get("query_id")
        query_text = record.get("query")
        if not query_id or not query_text:
            n_skipped += 1
            log.warning(
                "skipping record with missing query_id / query: %r", record,
            )
            continue
        gold = _gold_from_record(record)
        results = record.get("results") or []
        retrieved = [_retrieved_from_record(r) for r in results]
        all_rows.extend(build_bundle_export_rows(
            query_id=str(query_id),
            query=str(query_text),
            variant_name=str(args.variant_name),
            gold=gold,
            retrieved=retrieved,
            top_k_set=top_k_set,
            truncate_chars=int(args.truncate_chars),
        ))

    log.info(
        "[bundle] built %d bundle rows from %d records (skipped=%d)",
        len(all_rows), len(records), n_skipped,
    )
    if suffix == ".csv":
        write_bundle_export_csv(out_path, all_rows)
    else:
        write_bundle_export_jsonl(out_path, all_rows)
    log.info("[bundle] done. wrote %s", out_path)
    return 0


def _read_records_from_export(path: Path) -> List[Dict[str, Any]]:
    """Read previously-exported bundle rows from CSV or JSONL.

    This is *not* the labelled file reader — it intentionally bypasses
    validation so the sampler can operate on unlabelled exports
    (where ``label_context_answerability`` cells are still empty).
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fp:
            return [dict(row) for row in csv.DictReader(fp)]
    if suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise SystemExit(
                        f"non-object line in {path}: {line[:60]}..."
                    )
                out.append(obj)
        return out
    raise SystemExit(
        f"--input-path must end in .csv or .jsonl, got {path.name!r}"
    )


def _run_row_sample_mode(args: argparse.Namespace) -> int:
    """Phase 7.7 row-level sampling helper — pulls N query_ids deterministically.

    Reads an unlabelled row export, samples ``--sample-query-count``
    distinct query_ids using a seeded RNG, and writes the matching
    rows out unchanged (every rank for each chosen query_id is kept).
    Labels are NEVER filled in here — that is strictly a human step.
    """
    in_path = Path(args.input_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise SystemExit(f"--input-path not found: {in_path}")

    in_suffix = in_path.suffix.lower()
    if in_suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--input-path must end in .csv or .jsonl, got {in_path.name!r}"
        )
    out_suffix = out_path.suffix.lower()
    if out_suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--out-path must end in .csv or .jsonl, got {out_path.name!r}"
        )

    log.info(
        "[row-sample] reading %s (filter: variant=%s; n_queries=%d, seed=%d)",
        in_path, args.variant_name, args.sample_query_count, args.seed,
    )
    records = _read_records_from_export(in_path)
    sampled = sample_bundle_records(
        records,
        n_queries=int(args.sample_query_count),
        seed=int(args.seed),
        variant_name=args.variant_name,
        top_k=None,
    )
    log.info(
        "[row-sample] selected %d row(s) covering %d distinct "
        "query_id(s) out of %d input records",
        len(sampled),
        len({str(r.get("query_id")) for r in sampled}),
        len(records),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_suffix == ".csv":
        with out_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=list(EXPORT_COLUMNS),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in sampled:
                writer.writerow({
                    col: row.get(col, "")
                    for col in EXPORT_COLUMNS
                })
    else:
        with out_path.open("w", encoding="utf-8") as fp:
            for row in sampled:
                fp.write(json.dumps(dict(row), ensure_ascii=False))
                fp.write("\n")

    log.info("[row-sample] done. wrote %s", out_path)
    return 0


def _run_bundle_sample_mode(args: argparse.Namespace) -> int:
    """Phase 7.7.1 sampling helper — pulls N query_ids deterministically.

    Reads an unlabelled bundle export, samples ``--sample-query-count``
    distinct query_ids using a seeded RNG, and writes the matching
    rows out unchanged. Labels are NEVER filled in here — that is
    strictly a human step.
    """
    in_path = Path(args.input_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise SystemExit(f"--input-path not found: {in_path}")

    in_suffix = in_path.suffix.lower()
    if in_suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--input-path must end in .csv or .jsonl, got {in_path.name!r}"
        )
    out_suffix = out_path.suffix.lower()
    if out_suffix not in {".csv", ".jsonl"}:
        raise SystemExit(
            f"--out-path must end in .csv or .jsonl, got {out_path.name!r}"
        )

    log.info(
        "[bundle-sample] reading %s (filter: variant=%s, top_k=%s; "
        "n_queries=%d, seed=%d)",
        in_path, args.variant_name, args.top_k,
        args.sample_query_count, args.seed,
    )
    records = _read_records_from_export(in_path)
    sampled = sample_bundle_records(
        records,
        n_queries=int(args.sample_query_count),
        seed=int(args.seed),
        variant_name=args.variant_name,
        top_k=args.top_k,
    )
    log.info(
        "[bundle-sample] selected %d row(s) covering %d distinct "
        "query_id(s) out of %d input records",
        len(sampled),
        len({str(r.get("query_id")) for r in sampled}),
        len(records),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_suffix == ".csv":
        # Use the bundle export columns exactly so the sampled file is
        # a drop-in input to the score CLI after labelling.
        with out_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=list(BUNDLE_EXPORT_COLUMNS),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in sampled:
                # Pass-through: never inject label/flag/notes values.
                writer.writerow({
                    col: row.get(col, "")
                    for col in BUNDLE_EXPORT_COLUMNS
                })
    else:
        with out_path.open("w", encoding="utf-8") as fp:
            for row in sampled:
                fp.write(json.dumps(dict(row), ensure_ascii=False))
                fp.write("\n")

    log.info("[bundle-sample] done. wrote %s", out_path)
    return 0


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.7 / 7.7.1 — emit a labelling CSV / JSONL or a sampled "
        "subset. --mode row keeps the Phase 7.7 row-level export "
        "(default for backwards compat). --mode bundle emits the "
        "Phase 7.7.1 bundle export. --mode bundle-sample reads a "
        "previously-exported bundle file and samples N query_ids "
        "deterministically. None of these modes fills in label cells."
    ))
    p.add_argument(
        "--mode", choices=SUPPORTED_MODES, default="row",
        help=(
            "Export mode. row = per-chunk labelling (Phase 7.7). "
            "bundle = per-(query, top_k) labelling (Phase 7.7.1). "
            "bundle-sample = sample query_ids from an existing "
            "unlabelled bundle export. "
            "row-sample = sample query_ids from an existing "
            "unlabelled row export."
        ),
    )

    # Row + bundle shared inputs.
    p.add_argument(
        "--retrieval-results-path", type=Path, default=None,
        help=(
            "Used by --mode row / bundle. JSONL file with one query "
            "per line (see module docstring for the expected schema)."
        ),
    )
    p.add_argument(
        "--variant-name", type=str, default=None,
        help=(
            "Variant label for row / bundle modes. Acts as a filter "
            "in bundle-sample mode (optional)."
        ),
    )
    p.add_argument(
        "--input-format", choices=SUPPORTED_INPUT_FORMATS, default="audit",
        help=(
            "row / bundle modes: input shape of "
            "--retrieval-results-path. 'audit' (default) expects the "
            "audit-shaped schema (results/page_title/chunk_text). "
            "'v4-production' expects the production v4 retrieval emit "
            "(docs/title) and resolves chunk_text from --chunks-path "
            "by chunk_id, plus gold metadata from --gold-path."
        ),
    )
    p.add_argument(
        "--chunks-path", type=Path, default=None,
        help=(
            "v4-production input format only: chunk fixture for "
            "chunk_text resolution. Must be rag_chunks.jsonl for "
            "production retrieval emits; chunks_v4.jsonl is rejected "
            "because it uses a different chunk_id namespace."
        ),
    )
    p.add_argument(
        "--gold-path", type=Path, default=None,
        help=(
            "v4-production input format only: gold / silver query "
            "metadata. Accepts phase7_human_gold_seed_50.csv, "
            "queries_v4_llm_silver_500.jsonl, or "
            "queries_v4_silver_500.jsonl."
        ),
    )
    p.add_argument(
        "--on-missing-chunk",
        choices=SUPPORTED_ON_MISSING_CHUNK, default="error",
        help=(
            "v4-production input format only: behaviour when a "
            "retrieval doc references a chunk_id absent from the "
            "chunk fixture. 'error' (default) raises immediately; "
            "'collect' fills the cell with empty chunk_text and "
            "logs a summary — only suitable for triage runs because "
            "the audit's import validator will reject empty cells "
            "at scoring time."
        ),
    )
    p.add_argument(
        "--on-metadata-mismatch",
        choices=("error", "collect"),
        default="error",
        help=(
            "v4-production input format only: behaviour when retrieval "
            "emit metadata for a chunk_id disagrees with the chunk "
            "fixture. 'error' (default) fails closed; 'collect' logs "
            "a summary and emits fixture metadata, suitable only for "
            "triage."
        ),
    )
    p.add_argument(
        "--allow-missing-gold",
        action="store_true",
        help=(
            "v4-production input format only: allow retrieval records "
            "whose query_id has no gold page_id. Defaults to fail-closed "
            "because missing gold pollutes page_hit/section_hit metrics; "
            "use this only for triage exports."
        ),
    )

    # Row mode only.
    p.add_argument(
        "--top-k", type=int, default=None,
        help=(
            "row mode: how many top-ranked chunks per query to "
            "export (default 5; 0 means no truncation). "
            "bundle-sample mode: optional filter to a single top_k "
            "value."
        ),
    )

    # Bundle mode only.
    p.add_argument(
        "--top-k-set",
        type=_parse_top_k_set,
        default=None,
        help=(
            "bundle mode: comma-separated positive ints, e.g. "
            "\"1,3,5\" (default 1,3,5)."
        ),
    )
    p.add_argument(
        "--truncate-chars", type=int, default=DEFAULT_BUNDLE_TRUNCATE_CHARS,
        help=(
            "bundle mode: per-chunk character cap inside "
            "context_bundle_text. 0 disables truncation. "
            f"Default {DEFAULT_BUNDLE_TRUNCATE_CHARS}."
        ),
    )

    # Bundle-sample / row-sample modes.
    p.add_argument(
        "--input-path", type=Path, default=None,
        help=(
            "bundle-sample / row-sample modes: previously-exported "
            "CSV / JSONL to sample from."
        ),
    )
    p.add_argument(
        "--sample-query-count", type=int,
        default=DEFAULT_SAMPLE_QUERY_COUNT,
        help=(
            "bundle-sample / row-sample modes: number of distinct "
            f"query_ids to sample (default {DEFAULT_SAMPLE_QUERY_COUNT})."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SAMPLE_SEED,
        help=(
            "bundle-sample / row-sample modes: RNG seed for "
            f"deterministic sampling (default {DEFAULT_SAMPLE_SEED})."
        ),
    )

    # Common.
    p.add_argument(
        "--out-path", type=Path, required=True,
        help="Output path. Suffix .csv / .jsonl decides the writer.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _validate_mode_args(args: argparse.Namespace) -> None:
    """Check that the args required by ``args.mode`` are present.

    Argparse alone can't express the conditional-required pattern; we
    surface concrete error messages here so a missing ``--top-k-set``
    in bundle mode is obvious from the failure.
    """
    if args.mode in {"row", "bundle"}:
        if args.retrieval_results_path is None:
            raise SystemExit(
                f"--mode {args.mode} requires --retrieval-results-path"
            )
        if not args.variant_name:
            raise SystemExit(
                f"--mode {args.mode} requires --variant-name"
            )
    if args.mode == "row":
        if args.top_k is None:
            args.top_k = 5
    if args.mode == "bundle":
        if args.top_k_set is None:
            args.top_k_set = list(DEFAULT_TOP_K_SET)
        if args.truncate_chars is None:
            args.truncate_chars = DEFAULT_BUNDLE_TRUNCATE_CHARS
    if args.mode == "bundle-sample":
        if args.input_path is None:
            raise SystemExit(
                "--mode bundle-sample requires --input-path"
            )
        if args.sample_query_count <= 0:
            raise SystemExit(
                "--sample-query-count must be positive"
            )
    if args.mode == "row-sample":
        if args.input_path is None:
            raise SystemExit(
                "--mode row-sample requires --input-path"
            )
        if args.sample_query_count <= 0:
            raise SystemExit(
                "--sample-query-count must be positive"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _validate_mode_args(args)

    if args.mode == "row":
        return _run_row_mode(args)
    if args.mode == "bundle":
        return _run_bundle_mode(args)
    if args.mode == "bundle-sample":
        return _run_bundle_sample_mode(args)
    if args.mode == "row-sample":
        return _run_row_sample_mode(args)
    raise SystemExit(f"unsupported --mode {args.mode!r}")


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    sys.exit(main())
