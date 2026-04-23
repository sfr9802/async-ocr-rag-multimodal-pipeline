"""Sample a Korean anime corpus from port/rag into the project fixture tree.

Stream-reads the source JSONL (~195 MB, 1,764 titles) one line at a time
and produces a deterministic reservoir sample of N titles flattened into
the project's ingest-compatible shape.

Usage (from ai-worker/)::

    python -m scripts.dataset.sample_anime_corpus \\
        --source 'D:/port/rag/app/scripts/namu_anime_v3.fixed.jsonl' \\
        --out    fixtures/anime_corpus_kr.jsonl \\
        --sample-size 300 --seed 42

Shape notes
-----------
The source schema encodes ``sections`` as a LIST of {name, text, chunks,
bullets, urls, summary} objects. The production ingest pipeline
(``app.capabilities.rag.ingest``) consumes a DICT of
``{<section_name>: {"chunks": [...], "text": "..."}}``. This script
converts list -> dict at write time so the committed fixture is drop-in
compatible with ``python -m scripts.build_rag_index --input <path>`` and
the new ``--fixture anime_corpus_kr`` shortcut. Section order is preserved
in a separate ``section_order`` field and duplicate section names are
disambiguated with a ``#N`` suffix, matching how ``build_corpus.py``
handles the same collision for the enterprise KR corpus.

Safety rails
------------
* Committed-size cap: 40 MB (spec limit). Exceeding it logs a warning.
* Hard cap: 60 MB. Exceeding it deletes the output and exits non-zero —
  the caller almost certainly passed the wrong ``--sample-size``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("scripts.dataset.sample_anime_corpus")

_MAX_OUTPUT_BYTES_COMMIT = 40 * 1024 * 1024   # 40 MB — spec's committed-file cap
_MAX_OUTPUT_BYTES_HARD_FAIL = 60 * 1024 * 1024  # 60 MB — abort-if-above threshold


# ---------------------------------------------------------------------------
# Doc id helpers
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    """Korean-aware slug.

    Keeps 한글 (U+AC00..U+D7A3), ASCII letters, and digits; replaces
    everything else with a single ``-``. Lowercased and truncated to 60
    chars. Falls back to ``untitled`` if the input collapses to empty.
    """
    cleaned = re.sub(r"[^\w\uac00-\ud7a3]+", "-", title.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-").lower()
    cleaned = cleaned[:60]
    return cleaned or "untitled"


def _short_hash(seed_title: str) -> str:
    return hashlib.md5(seed_title.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Streaming + sampling
# ---------------------------------------------------------------------------


def _iter_source(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield each source record at most once per seed_title.

    The source file contains multiple records for some titles (likely
    re-crawls of the same namu-wiki page). Deduping on ``seed_title``
    at the stream stage keeps the reservoir's effective pool clean so
    ``--sample-size 300`` yields 300 distinct titles, not 300 slots with
    repeats of popular anime.
    """
    seen_seeds: set[str] = set()
    duplicates = 0
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Line %d: malformed JSON, skipping.", line_no)
                continue
            key = str(
                record.get("seed_title")
                or record.get("title")
                or ""
            ).strip()
            if not key:
                continue
            if key in seen_seeds:
                duplicates += 1
                continue
            seen_seeds.add(key)
            yield record
    if duplicates:
        log.info(
            "Stream dedup dropped %d duplicate-seed_title records "
            "(unique titles yielded: %d)",
            duplicates, len(seen_seeds),
        )


def _is_usable(record: Dict[str, Any]) -> bool:
    """True if the record has at least one section with non-empty text/chunks."""
    sections = record.get("sections")
    if not isinstance(sections, list) or not sections:
        return False
    for s in sections:
        if not isinstance(s, dict):
            continue
        text = s.get("text")
        if isinstance(text, str) and text.strip():
            return True
        chunks = s.get("chunks")
        if isinstance(chunks, list) and any(
            isinstance(c, str) and c.strip() for c in chunks
        ):
            return True
    return False


def _flatten_sections(
    sections_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Convert list-of-sections (source) to dict-of-sections (ingest)."""
    out: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for raw in sections_list:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip() or "section"
        dedup = name
        suffix = 2
        while dedup in out:
            dedup = f"{name}#{suffix}"
            suffix += 1

        body: Dict[str, Any] = {}
        text = raw.get("text")
        if isinstance(text, str) and text.strip():
            body["text"] = text.strip()
        chunks = raw.get("chunks")
        if isinstance(chunks, list):
            kept = [
                str(c).strip()
                for c in chunks
                if isinstance(c, (str, int, float)) and str(c).strip()
            ]
            if kept:
                body["chunks"] = kept
        if not body:
            continue

        out[dedup] = body
        order.append(dedup)
    return out, order


def _reservoir_sample(
    stream: Iterator[Dict[str, Any]],
    *,
    k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Algorithm R (Vitter 1985) with a seeded RNG.

    Returns ``(reservoir, total_usable_seen)``. Filters unusable records
    before sampling so reservoir positions stay deterministic across
    source files that differ only in skippable junk lines.
    """
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    total_seen = 0
    for record in stream:
        if not _is_usable(record):
            continue
        if len(reservoir) < k:
            reservoir.append(record)
        else:
            j = rng.randint(0, total_seen)
            if j < k:
                reservoir[j] = record
        total_seen += 1
    return reservoir, total_seen


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------


def _to_output_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = str(record.get("title") or "").strip()
    seed_title = str(record.get("seed_title") or title).strip()
    if not title or not seed_title:
        return None
    sections_map, order = _flatten_sections(record.get("sections") or [])
    if not sections_map:
        return None
    return {
        "doc_id": f"{_slugify(title)}-{_short_hash(seed_title)}",
        "title": title[:500],
        "seed_title": seed_title[:500],
        "sections": sections_map,
        "section_order": order,
        "domain": "anime",
        "language": "ko",
        "source": "namu-wiki-v3-fixed",
        "source_ts": None,
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False))
            fp.write("\n")
            count += 1
    return count


def _count_source_lines(path: Path) -> int:
    """Cheap line-count pass. Source is ~195 MB — a couple of seconds."""
    total = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                total += 1
    return total


def _compute_stats(
    records: List[Dict[str, Any]],
    *,
    source_total: int,
    source_unique_titles: int,
    seed: int,
) -> Dict[str, Any]:
    chunk_total = 0
    section_char_total = 0
    section_count_total = 0
    for r in records:
        sections = r.get("sections") or {}
        for body in sections.values():
            chunks = body.get("chunks") or []
            chunk_total += len(chunks)
            text = body.get("text") or ""
            section_char_total += len(text) or sum(len(c) for c in chunks)
            section_count_total += 1
    per_section_avg = (
        section_char_total / section_count_total
        if section_count_total
        else 0.0
    )
    return {
        "sampled": len(records),
        "source_total": source_total,
        "source_unique_titles": source_unique_titles,
        "chunk_count_total": chunk_total,
        "section_count_total": section_count_total,
        "per_section_avg_chars": round(per_section_avg, 1),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source JSONL (e.g. D:/port/rag/app/scripts/namu_anime_v3.fixed.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL (e.g. fixtures/anime_corpus_kr.jsonl).",
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.source.exists():
        log.error("Source file not found: %s", args.source)
        return 2
    if args.sample_size <= 0:
        log.error("--sample-size must be positive, got %d", args.sample_size)
        return 2

    log.info(
        "Reservoir-sampling %d titles from %s (seed=%d)",
        args.sample_size, args.source, args.seed,
    )
    reservoir, total_usable = _reservoir_sample(
        _iter_source(args.source),
        k=args.sample_size,
        seed=args.seed,
    )
    log.info(
        "Reservoir filled with %d records (seen %d usable source records)",
        len(reservoir), total_usable,
    )

    seen_doc_ids: set[str] = set()
    output_records: List[Dict[str, Any]] = []
    for rec in reservoir:
        flattened = _to_output_record(rec)
        if flattened is None:
            continue
        if flattened["doc_id"] in seen_doc_ids:
            log.warning(
                "doc_id collision: %s — skipping duplicate title=%r",
                flattened["doc_id"], flattened.get("title"),
            )
            continue
        seen_doc_ids.add(flattened["doc_id"])
        output_records.append(flattened)

    count = _write_jsonl(args.out, output_records)
    size_bytes = args.out.stat().st_size
    size_mb = size_bytes / 1024 / 1024
    log.info("Wrote %d records to %s (%.1f MB)", count, args.out, size_mb)

    if size_bytes > _MAX_OUTPUT_BYTES_HARD_FAIL:
        log.error(
            "Output size %.1f MB exceeds hard cap of %.0f MB — "
            "likely wrong --sample-size. Removing file.",
            size_mb, _MAX_OUTPUT_BYTES_HARD_FAIL / 1024 / 1024,
        )
        args.out.unlink(missing_ok=True)
        return 2
    if size_bytes > _MAX_OUTPUT_BYTES_COMMIT:
        log.warning(
            "Output %.1f MB exceeds %.0f MB commit cap — consider a smaller "
            "--sample-size before committing.",
            size_mb, _MAX_OUTPUT_BYTES_COMMIT / 1024 / 1024,
        )

    log.info("Counting raw source lines for provenance ...")
    source_total = _count_source_lines(args.source)
    stats = _compute_stats(
        output_records,
        source_total=source_total,
        source_unique_titles=total_usable,
        seed=args.seed,
    )
    meta_path = args.out.with_name(args.out.stem + ".meta.json")
    meta_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Meta: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
