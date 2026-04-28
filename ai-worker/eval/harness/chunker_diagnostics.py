"""Phase 1C — chunker provenance diagnostics.

Goal
----
Explain *why* the production chunker leaves long chunks behind. We
re-run the corpus through ``_chunks_from_section`` (the same path
``ingest.py`` uses) while keeping a per-section provenance map, then
sort the emitted chunks by token count and dump the top-N with their
source-payload context.

The diagnostic does not modify the corpus, the chunker, or the index.
It only reads + tokenizes.

What gets recorded per emitted chunk
------------------------------------
- ``doc_id``                — source document id
- ``title``                 — source document title (if available)
- ``section_path``          — section name within the document
- ``source_payload_type``   — which of ``chunks`` / ``list`` / ``text``
                              the section's units came from
- ``original_unit_count``   — number of source units fed to the
                              chunker for that section
- ``original_total_chars``  — sum of source-unit char lengths
- ``original_total_tokens`` — sum of source-unit token lengths
- ``emitted_chunk_index``   — index within the section's emitted list
- ``emitted_chunk_char_count``
- ``emitted_chunk_token_count``
- ``was_split``             — True iff > 1 chunk was emitted from the
                              section (i.e. window_by_chars at least
                              tried to split)
- ``split_strategy``        — "window_by_chars_solo" when only one
                              chunk was emitted, "window_by_chars_packed"
                              otherwise
- ``split_reason``          — "single_unit_oversize", "list_oversize_per_entry",
                              "merged_padding", or "unknown" — best-effort
                              attribution
- ``preview``               — first ~200 chars of the chunk text

Public surface
--------------
- ``ChunkerDiagnosisSample``       — one provenance record
- ``ChunkerDiagnosisSummary``      — corpus-level rollup
- ``diagnose_chunker_long_tail``   — main entry point
- ``chunker_diagnosis_to_dict``    — JSON-friendly serializer
- ``render_chunker_diagnosis_markdown``
- ``render_chunker_provenance_markdown``
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from app.capabilities.rag.ingest import _chunks_from_section, _iter_documents
from app.capabilities.rag.token_aware_chunker import raw_section_units


log = logging.getLogger(__name__)


PREVIEW_CHARS = 200


# Default thresholds — match the analyze-corpus-lengths convention so
# the headline cross-tab numbers line up across reports.
DEFAULT_THRESHOLDS: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
DEFAULT_TOP_N = 200


# ---------------------------------------------------------------------------
# Datatypes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkerDiagnosisSample:
    doc_id: str
    title: Optional[str]
    section_path: str
    source_payload_type: str
    original_unit_count: int
    original_total_chars: int
    original_total_tokens: int
    emitted_chunk_index: int
    emitted_chunk_char_count: int
    emitted_chunk_token_count: int
    was_split: bool
    split_strategy: str
    split_reason: str
    preview: str


@dataclass
class PayloadTypeBreakdown:
    """Per-payload-type rollup."""

    chunk_count: int = 0
    chunks_over_512: int = 0
    chunks_over_1024: int = 0
    chunks_over_2048: int = 0
    chunks_over_4096: int = 0
    chunks_over_8192: int = 0
    max_token_count: int = 0


@dataclass
class ChunkerDiagnosisSummary:
    corpus_path: str
    tokenizer: str
    document_count: int
    section_count: int
    chunk_count: int
    chunks_over_token_threshold: Dict[int, int] = field(default_factory=dict)
    chunks_over_token_threshold_ratio: Dict[int, float] = field(default_factory=dict)
    payload_type_breakdown: Dict[str, PayloadTypeBreakdown] = field(default_factory=dict)
    split_reason_counts: Dict[str, int] = field(default_factory=dict)
    long_chunk_split_reason_counts: Dict[str, int] = field(default_factory=dict)
    long_chunk_threshold_tokens: int = 1024
    section_max_token_p50: float = 0.0
    section_max_token_p90: float = 0.0
    section_max_token_p95: float = 0.0
    section_max_token_p99: float = 0.0
    sections_with_long_chunks: int = 0


# ---------------------------------------------------------------------------
# Tokenizer plumbing — same shape as analyze_corpus_lengths.
# ---------------------------------------------------------------------------


# Batch token counter. Wrapping a transformers tokenizer is cheap
# enough that we don't bother with a single-string variant here.
TokenCounter = Callable[[Sequence[str]], List[int]]


def _default_token_counter(model_name: str) -> TokenCounter:
    """Return a batched HF AutoTokenizer wrapper.

    Lazy import keeps the diagnostics module importable in unit tests
    that pass an in-memory stub.
    """
    log.info("Loading tokenizer: %s", model_name)
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _count(batch: Sequence[str]) -> List[int]:
        if not batch:
            return []
        encoded = tokenizer(
            list(batch),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    return _count


def _count_in_batches(
    texts: Sequence[str],
    *,
    counter: TokenCounter,
    batch_size: int,
) -> List[int]:
    out: List[int] = []
    n = len(texts)
    if n == 0:
        return out
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        counts = counter(batch)
        if len(counts) != len(batch):
            raise RuntimeError(
                f"token_counter returned {len(counts)} for batch of {len(batch)}"
            )
        out.extend(int(c) for c in counts)
        if (i // batch_size) % 50 == 0 and i > 0:
            log.info("Tokenized %d / %d", i + len(batch), n)
    return out


# ---------------------------------------------------------------------------
# Per-section provenance reconstruction.
# ---------------------------------------------------------------------------


def _detect_payload_type(raw_section: Mapping[str, Any]) -> str:
    """Re-derive which of ``chunks`` / ``list`` / ``text`` was used.

    Mirrors the priority in
    ``app.capabilities.rag.ingest._chunks_from_section``. Returns
    ``"chunks_list"`` if the section's pre-chunked list was non-empty,
    ``"list_entries"`` if the structured list became the unit source,
    ``"text_blob"`` if neither was present and only the text blob
    contributed, ``"empty"`` if no source was found.
    """
    pre = raw_section.get("chunks")
    if isinstance(pre, list) and any(
        isinstance(x, (str, int, float)) and str(x).strip() for x in pre
    ):
        return "chunks_list"

    list_entries = raw_section.get("list")
    if isinstance(list_entries, list):
        for entry in list_entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            if name or desc:
                return "list_entries"

    blob = raw_section.get("text")
    if isinstance(blob, str) and blob.strip():
        return "text_blob"

    return "empty"


def _split_reason(
    *,
    payload_type: str,
    original_unit_count: int,
    emitted_count: int,
    long_chunk: bool,
) -> str:
    """Best-effort attribution for why the chunk landed long.

    The heuristic uses payload type + (units in / chunks out):
      - "single_unit_oversize"  : 1 source unit, emitted as 1 chunk;
        window_by_chars never had a boundary to split on.
      - "list_oversize_per_entry": payload_type == list_entries and
        emitted_count > 1 — the list was big enough that
        window_by_chars produced multiple chunks but each is still
        oversize because individual entries can't be split.
      - "merged_padding"        : payload_type in (chunks_list, list_entries)
        with emitted_count > 1 — window_by_chars merged small fragments
        until the buffer overflowed.
      - "unknown"               : everything else.
    """
    if not long_chunk:
        return "below_threshold"
    if original_unit_count == 1 and emitted_count == 1:
        return "single_unit_oversize"
    if payload_type == "list_entries" and emitted_count > 1:
        return "list_oversize_per_entry"
    if payload_type in ("chunks_list", "list_entries") and emitted_count > 1:
        return "merged_padding"
    if payload_type == "text_blob" and emitted_count == 1:
        return "single_unit_oversize"
    return "unknown"


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def diagnose_chunker_long_tail(
    corpus_path: Path,
    *,
    token_counter: Optional[TokenCounter] = None,
    tokenizer_name: str = "BAAI/bge-m3",
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    long_chunk_threshold: int = 1024,
    top_n: int = DEFAULT_TOP_N,
    batch_size: int = 256,
) -> Tuple[ChunkerDiagnosisSummary, List[ChunkerDiagnosisSample]]:
    """Diagnose long-tail chunks emitted by the production chunker.

    Returns ``(summary, top_samples)`` where ``top_samples`` is sorted
    by emitted chunk token count descending and capped at ``top_n``.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    counter = token_counter or _default_token_counter(tokenizer_name)

    # Pass 1 — walk every (doc, section) and collect:
    #   - emitted chunk text + provenance fields (without token count yet)
    rows: List[Dict[str, Any]] = []
    doc_ids: set[str] = set()
    section_count = 0

    for raw in _iter_documents(path):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id:
            continue
        doc_ids.add(doc_id)
        title = raw.get("title") or raw.get("seed") or None
        if isinstance(title, str):
            title = title[:300]
        else:
            title = None

        sections = raw.get("sections") or {}
        if not isinstance(sections, dict):
            continue

        for section_name, section_raw in sections.items():
            if not isinstance(section_raw, dict):
                continue
            section_count += 1

            payload_type = _detect_payload_type(section_raw)
            source_units = raw_section_units(section_raw)
            emitted = _chunks_from_section(section_raw)
            if not emitted:
                continue

            original_total_chars = sum(len(u) for u in source_units)
            for idx, chunk_text in enumerate(emitted):
                rows.append({
                    "doc_id": doc_id,
                    "title": title,
                    "section_path": str(section_name),
                    "source_payload_type": payload_type,
                    "original_unit_count": len(source_units),
                    "original_total_chars": original_total_chars,
                    "_source_units": source_units,  # for token sum (pass 2)
                    "emitted_chunk_index": idx,
                    "emitted_chunk_char_count": len(chunk_text),
                    "_chunk_text": chunk_text,
                    "was_split": len(emitted) > 1,
                    "_emitted_count": len(emitted),
                })

    if not rows:
        raise RuntimeError(
            f"Corpus {path} produced zero chunks — empty or wrong schema."
        )

    log.info(
        "Diagnosing %d emitted chunks from %d sections in %d docs",
        len(rows), section_count, len(doc_ids),
    )

    # Pass 2 — tokenize emitted chunks AND source units in batches.
    chunk_texts = [r["_chunk_text"] for r in rows]
    chunk_token_counts = _count_in_batches(
        chunk_texts, counter=counter, batch_size=batch_size,
    )
    if len(chunk_token_counts) != len(rows):
        raise RuntimeError(
            "token_counter returned a mismatched length: "
            f"{len(chunk_token_counts)} vs {len(rows)}"
        )

    # Tokenize source units once per (doc_id, section_path) to avoid
    # repeating work for sections that emitted multiple chunks.
    section_unit_token_totals: Dict[Tuple[str, str], int] = {}
    pending_unit_texts: List[str] = []
    pending_unit_keys: List[Tuple[str, str]] = []
    seen_keys: set[Tuple[str, str]] = set()
    for r in rows:
        key = (r["doc_id"], r["section_path"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        for u in r["_source_units"]:
            pending_unit_texts.append(u)
            pending_unit_keys.append(key)

    if pending_unit_texts:
        unit_tokens = _count_in_batches(
            pending_unit_texts, counter=counter, batch_size=batch_size,
        )
        for key, t in zip(pending_unit_keys, unit_tokens):
            section_unit_token_totals[key] = (
                section_unit_token_totals.get(key, 0) + int(t)
            )

    # Pass 3 — assemble samples + summary.
    samples: List[ChunkerDiagnosisSample] = []
    for r, chunk_tokens in zip(rows, chunk_token_counts):
        key = (r["doc_id"], r["section_path"])
        original_total_tokens = int(section_unit_token_totals.get(key, 0))
        long_chunk = chunk_tokens > long_chunk_threshold
        split_strategy = (
            "window_by_chars_packed" if r["_emitted_count"] > 1
            else "window_by_chars_solo"
        )
        split_reason = _split_reason(
            payload_type=r["source_payload_type"],
            original_unit_count=r["original_unit_count"],
            emitted_count=r["_emitted_count"],
            long_chunk=long_chunk,
        )
        samples.append(ChunkerDiagnosisSample(
            doc_id=r["doc_id"],
            title=r["title"],
            section_path=r["section_path"],
            source_payload_type=r["source_payload_type"],
            original_unit_count=int(r["original_unit_count"]),
            original_total_chars=int(r["original_total_chars"]),
            original_total_tokens=original_total_tokens,
            emitted_chunk_index=int(r["emitted_chunk_index"]),
            emitted_chunk_char_count=int(r["emitted_chunk_char_count"]),
            emitted_chunk_token_count=int(chunk_tokens),
            was_split=bool(r["was_split"]),
            split_strategy=split_strategy,
            split_reason=split_reason,
            preview=_make_preview(r["_chunk_text"]),
        ))

    summary = _build_summary(
        path=path,
        tokenizer=tokenizer_name,
        samples=samples,
        document_count=len(doc_ids),
        section_count=section_count,
        thresholds=thresholds,
        long_chunk_threshold=long_chunk_threshold,
    )

    # Sort + cap top samples.
    top_samples = sorted(
        samples,
        key=lambda s: s.emitted_chunk_token_count,
        reverse=True,
    )[: max(0, top_n)]

    return summary, top_samples


def _make_preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _build_summary(
    *,
    path: Path,
    tokenizer: str,
    samples: Sequence[ChunkerDiagnosisSample],
    document_count: int,
    section_count: int,
    thresholds: Sequence[int],
    long_chunk_threshold: int,
) -> ChunkerDiagnosisSummary:
    summary = ChunkerDiagnosisSummary(
        corpus_path=str(path),
        tokenizer=tokenizer,
        document_count=document_count,
        section_count=section_count,
        chunk_count=len(samples),
        long_chunk_threshold_tokens=long_chunk_threshold,
    )

    # Threshold cross-tab.
    threshold_counts: Dict[int, int] = {int(t): 0 for t in thresholds}
    for s in samples:
        for t in thresholds:
            if s.emitted_chunk_token_count > t:
                threshold_counts[int(t)] += 1
    n = len(samples)
    summary.chunks_over_token_threshold = threshold_counts
    summary.chunks_over_token_threshold_ratio = {
        t: round(c / n, 6) if n else 0.0 for t, c in threshold_counts.items()
    }

    # Payload type breakdown.
    breakdown: Dict[str, PayloadTypeBreakdown] = {}
    for s in samples:
        b = breakdown.setdefault(s.source_payload_type, PayloadTypeBreakdown())
        b.chunk_count += 1
        if s.emitted_chunk_token_count > 512:
            b.chunks_over_512 += 1
        if s.emitted_chunk_token_count > 1024:
            b.chunks_over_1024 += 1
        if s.emitted_chunk_token_count > 2048:
            b.chunks_over_2048 += 1
        if s.emitted_chunk_token_count > 4096:
            b.chunks_over_4096 += 1
        if s.emitted_chunk_token_count > 8192:
            b.chunks_over_8192 += 1
        if s.emitted_chunk_token_count > b.max_token_count:
            b.max_token_count = s.emitted_chunk_token_count
    summary.payload_type_breakdown = breakdown

    # Split-reason counts (long chunks only — the diagnostic question
    # is "why is *this* chunk long").
    reason_counts: Dict[str, int] = {}
    long_reason_counts: Dict[str, int] = {}
    for s in samples:
        reason_counts[s.split_reason] = reason_counts.get(s.split_reason, 0) + 1
        if s.emitted_chunk_token_count > long_chunk_threshold:
            long_reason_counts[s.split_reason] = (
                long_reason_counts.get(s.split_reason, 0) + 1
            )
    summary.split_reason_counts = reason_counts
    summary.long_chunk_split_reason_counts = long_reason_counts

    # Per-section max-token percentile distribution.
    section_max: Dict[Tuple[str, str], int] = {}
    sections_with_long: set[Tuple[str, str]] = set()
    for s in samples:
        key = (s.doc_id, s.section_path)
        section_max[key] = max(
            section_max.get(key, 0), s.emitted_chunk_token_count
        )
        if s.emitted_chunk_token_count > long_chunk_threshold:
            sections_with_long.add(key)

    if section_max:
        sorted_max = sorted(section_max.values())
        summary.section_max_token_p50 = _percentile(sorted_max, 50)
        summary.section_max_token_p90 = _percentile(sorted_max, 90)
        summary.section_max_token_p95 = _percentile(sorted_max, 95)
        summary.section_max_token_p99 = _percentile(sorted_max, 99)
    summary.sections_with_long_chunks = len(sections_with_long)

    return summary


def _percentile(sorted_vals: Sequence[int], pct: int) -> float:
    """Nearest-rank percentile (matches eval.harness.metrics.p_percentile)."""
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    import math

    idx = int(math.ceil((pct / 100.0) * len(sorted_vals))) - 1
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


# ---------------------------------------------------------------------------
# Serializers + markdown.
# ---------------------------------------------------------------------------


def chunker_diagnosis_to_dict(
    summary: ChunkerDiagnosisSummary,
) -> Dict[str, Any]:
    """Convert summary to a JSON-friendly dict.

    ``payload_type_breakdown`` becomes a nested dict-of-dicts, which is
    what callers actually want when consuming the report.
    """
    payload_breakdown = {
        ptype: asdict(b)
        for ptype, b in summary.payload_type_breakdown.items()
    }
    return {
        "corpus_path": summary.corpus_path,
        "tokenizer": summary.tokenizer,
        "document_count": summary.document_count,
        "section_count": summary.section_count,
        "chunk_count": summary.chunk_count,
        "long_chunk_threshold_tokens": summary.long_chunk_threshold_tokens,
        "chunks_over_token_threshold": summary.chunks_over_token_threshold,
        "chunks_over_token_threshold_ratio": summary.chunks_over_token_threshold_ratio,
        "payload_type_breakdown": payload_breakdown,
        "split_reason_counts": summary.split_reason_counts,
        "long_chunk_split_reason_counts": summary.long_chunk_split_reason_counts,
        "section_max_token_p50": summary.section_max_token_p50,
        "section_max_token_p90": summary.section_max_token_p90,
        "section_max_token_p95": summary.section_max_token_p95,
        "section_max_token_p99": summary.section_max_token_p99,
        "sections_with_long_chunks": summary.sections_with_long_chunks,
    }


def render_chunker_diagnosis_markdown(
    summary: ChunkerDiagnosisSummary,
) -> str:
    """Render the corpus-level summary as markdown."""
    lines: List[str] = []
    lines.append("# Phase 1C — chunker long-tail diagnosis")
    lines.append("")
    lines.append(f"- corpus: `{summary.corpus_path}`")
    lines.append(f"- tokenizer: `{summary.tokenizer}`")
    lines.append(f"- documents: {summary.document_count}")
    lines.append(f"- sections: {summary.section_count}")
    lines.append(f"- emitted chunks: {summary.chunk_count}")
    lines.append(
        f"- long-chunk threshold: > {summary.long_chunk_threshold_tokens} tokens"
    )
    lines.append("")

    lines.append("## Chunks over token threshold")
    lines.append("")
    lines.append("| threshold | chunks > threshold | ratio |")
    lines.append("|---|---:|---:|")
    for t in sorted(summary.chunks_over_token_threshold.keys()):
        c = summary.chunks_over_token_threshold[t]
        r = summary.chunks_over_token_threshold_ratio.get(t, 0.0)
        lines.append(f"| > {t} | {c} | {r:.4f} |")
    lines.append("")

    lines.append("## Payload-type breakdown")
    lines.append("")
    lines.append(
        "| payload type | chunks | max tok | >512 | >1024 | >2048 | >4096 | >8192 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ptype in sorted(summary.payload_type_breakdown.keys()):
        b = summary.payload_type_breakdown[ptype]
        lines.append(
            f"| `{ptype}` | {b.chunk_count} | {b.max_token_count} | "
            f"{b.chunks_over_512} | {b.chunks_over_1024} | "
            f"{b.chunks_over_2048} | {b.chunks_over_4096} | {b.chunks_over_8192} |"
        )
    lines.append("")

    lines.append("## Split-reason rollup (long chunks only)")
    lines.append("")
    lines.append("| split_reason | count |")
    lines.append("|---|---:|")
    for reason in sorted(
        summary.long_chunk_split_reason_counts,
        key=lambda r: -summary.long_chunk_split_reason_counts[r],
    ):
        lines.append(
            f"| `{reason}` | {summary.long_chunk_split_reason_counts[reason]} |"
        )
    lines.append("")

    lines.append("## Per-section max-token percentiles")
    lines.append("")
    lines.append(
        f"- p50: {summary.section_max_token_p50:.0f}"
    )
    lines.append(
        f"- p90: {summary.section_max_token_p90:.0f}"
    )
    lines.append(
        f"- p95: {summary.section_max_token_p95:.0f}"
    )
    lines.append(
        f"- p99: {summary.section_max_token_p99:.0f}"
    )
    lines.append(
        f"- sections with > {summary.long_chunk_threshold_tokens} token chunks: "
        f"{summary.sections_with_long_chunks}"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def render_chunker_provenance_markdown(
    samples: Sequence[ChunkerDiagnosisSample],
    *,
    summary: Optional[ChunkerDiagnosisSummary] = None,
) -> str:
    """Render the top-N provenance samples as markdown.

    Each sample is one section: doc_id, title, section path, payload
    type, original unit shape, emitted chunk shape, split attribution,
    and a 200-char preview.
    """
    lines: List[str] = []
    lines.append("# Phase 1C — chunker provenance (top samples)")
    lines.append("")
    if summary is not None:
        lines.append(
            f"_Source corpus: `{summary.corpus_path}` · tokenizer: "
            f"`{summary.tokenizer}` · samples: {len(samples)}_"
        )
        lines.append("")
    if not samples:
        lines.append("_No samples._")
        return "\n".join(lines) + "\n"

    for i, s in enumerate(samples, start=1):
        title = s.title or "(no title)"
        lines.append(
            f"## {i}. {s.doc_id} · {title} · `{s.section_path}` (chunk #{s.emitted_chunk_index})"
        )
        lines.append("")
        lines.append(
            f"- emitted chunk: **{s.emitted_chunk_token_count} tokens** "
            f"({s.emitted_chunk_char_count} chars)"
        )
        lines.append(
            f"- source payload: `{s.source_payload_type}` "
            f"({s.original_unit_count} units, {s.original_total_chars} chars, "
            f"{s.original_total_tokens} tokens)"
        )
        lines.append(
            f"- was_split: `{s.was_split}` · split_strategy: "
            f"`{s.split_strategy}` · split_reason: `{s.split_reason}`"
        )
        lines.append("")
        lines.append("```")
        lines.append(s.preview)
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def samples_to_dict_list(
    samples: Sequence[ChunkerDiagnosisSample],
) -> List[Dict[str, Any]]:
    """Serialize a sample list for JSON output."""
    return [asdict(s) for s in samples]
