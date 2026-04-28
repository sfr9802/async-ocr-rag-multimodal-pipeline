"""Long-chunk audit + raw-vs-cleaned length comparison report.

Phase 1A surfaces the long-tail of multi-thousand token chunks the
``analyze_corpus_lengths`` Phase 0 report flagged, and pairs each long
chunk with the noise signals the conservative detector finds in it. The
result is the input we use to decide whether a downstream cleaning pass
is justified — and, if so, by how much.

Two reports
-----------
1. **Long-chunk audit** (``audit_long_chunks``)
   Top-N chunks by token count, each annotated with detected noise
   signals. Emitted via ``audit-corpus-noise`` CLI to JSON + Markdown.

2. **Length comparison** (``compare_raw_vs_cleaned``)
   Side-by-side char/token distributions for raw vs cleaned chunks,
   plus chunks-over-cap counts at the standard thresholds. Emitted via
   ``audit-corpus-noise`` CLI alongside the audit. ``clean-corpus-dry-run``
   re-uses the same routine to emit a focused cleaner-effect summary.

Tokenizer
---------
We accept the same ``token_counter`` callable that
``analyze_corpus_lengths`` uses. Tests inject a deterministic stub. The
CLI defaults to a real bge-m3 tokenizer.

Percentile policy
-----------------
We use ``eval.harness.metrics.p_percentile`` directly so this report's
percentiles are byte-identical with every other percentile in the
harness. (``analyze_corpus_lengths._percentile`` is a private clone
that is already pinned to ``p_percentile`` by its tests.)

Public surface
--------------
- ``ChunkAuditEntry``        — one row of the long-chunk audit
- ``LongChunkAuditReport``   — the full audit
- ``LengthBucket``           — char/token distribution for one corpus pass
- ``LengthComparisonReport`` — raw + cleaned + delta
- ``audit_long_chunks``      — main audit entry
- ``compare_raw_vs_cleaned`` — main length-comparison entry
- ``audit_to_dict``, ``render_audit_markdown``
- ``length_comparison_to_dict``, ``render_length_comparison_markdown``
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
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from app.capabilities.rag.ingest import _chunks_from_section, _iter_documents
from eval.harness.analyze_corpus_lengths import (
    DEFAULT_THRESHOLDS,
    DEFAULT_TOKENIZER_NAME,
    LengthDistribution,
    _count_in_batches,
    _default_token_counter,
    _make_preview,
)
from eval.harness.corpus_cleaner import (
    DROP_REASON_EMPTY_AFTER_CLEAN,
    DROP_REASON_EMPTY_INPUT,
    CleaningResult,
    clean_chunk,
)
from eval.harness.corpus_noise_signals import (
    NoiseSignal,
    aggregate_signals,
    detect_noise_signals,
    signal_to_dict,
)
from eval.harness.metrics import p_percentile

log = logging.getLogger(__name__)


DEFAULT_AUDIT_TOP_N = 200
PREVIEW_CHARS = 240


TokenCounter = Callable[[Sequence[str]], List[int]]


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class ChunkAuditEntry:
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    chunk_index: int
    char_count: int
    token_count: int
    preview: str
    detected_noise_signals: List[NoiseSignal] = field(default_factory=list)


@dataclass
class LongChunkAuditReport:
    corpus_path: str
    document_count: int
    chunk_count: int
    tokenizer: str
    top_n: int
    long_chunks: List[ChunkAuditEntry]
    noise_signal_summary: Dict[str, int]


@dataclass
class LengthBucket:
    label: str  # "raw" or "cleaned"
    char: LengthDistribution
    token: LengthDistribution
    chunks_over_token_threshold: Dict[int, int]


@dataclass
class LengthComparisonReport:
    corpus_path: str
    tokenizer: str
    thresholds: List[int]
    raw: LengthBucket
    cleaned: LengthBucket
    raw_chunk_count: int
    cleaned_chunk_count: int
    dropped_chunk_count: int
    drop_reasons: Dict[str, int]
    cleaner_total_removed_lines: int
    cleaner_total_collapsed_repeats: int


# --- Iteration over the production chunker ------------------------------


@dataclass(frozen=True)
class _RawChunk:
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    chunk_index: int
    text: str


def _iter_corpus_chunks(corpus_path: Path) -> Iterator[_RawChunk]:
    """Yield chunks via the production chunker.

    Same iteration shape as ``analyze_corpus_lengths``: skip docs with
    no ``doc_id``, skip non-dict sections, skip empty stripped chunks.
    The yielded ``chunk_id`` is informational only — not the FAISS row
    hash — but stable enough to cite in reports.
    """
    for raw in _iter_documents(corpus_path):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id:
            continue
        title = str(raw.get("title") or raw.get("seed") or "")[:200]
        sections = raw.get("sections") or {}
        if not isinstance(sections, dict):
            continue
        for section_name, section_raw in sections.items():
            if not isinstance(section_raw, dict):
                continue
            for order, text in enumerate(_chunks_from_section(section_raw)):
                chunk_text = text.strip()
                if not chunk_text:
                    continue
                yield _RawChunk(
                    chunk_id=f"{doc_id}::{section_name}::{order}",
                    doc_id=doc_id,
                    title=title,
                    section_path=str(section_name),
                    chunk_index=order,
                    text=chunk_text,
                )


# --- Long-chunk audit --------------------------------------------------


def audit_long_chunks(
    corpus_path: Path,
    *,
    top_n: int = DEFAULT_AUDIT_TOP_N,
    token_counter: Optional[TokenCounter] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    batch_size: int = 256,
) -> LongChunkAuditReport:
    """Build the long-chunk audit report.

    Reads the corpus via the production chunker, tokenizes every chunk,
    sorts by token count descending, takes the first ``top_n``, and
    annotates each with detected noise signals. The noise summary at
    the top of the report is over the *full corpus*, not just the
    long-N — so the audit also tells you whether noise is concentrated
    in the long tail or spread across the whole index.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks = list(_iter_corpus_chunks(path))
    if not chunks:
        raise RuntimeError(
            f"Corpus {path} produced zero chunks — empty or wrong schema."
        )

    doc_ids = {c.doc_id for c in chunks}

    log.info(
        "Auditing %d chunks from %d docs in %s (tokenizer=%s, top_n=%d)",
        len(chunks), len(doc_ids), path, tokenizer_name, top_n,
    )

    counter = token_counter or _default_token_counter(tokenizer_name)
    token_lengths = _count_in_batches(
        [c.text for c in chunks], counter=counter, batch_size=batch_size,
    )
    if len(token_lengths) != len(chunks):
        raise RuntimeError(
            "token_counter returned a mismatched length: "
            f"{len(token_lengths)} vs {len(chunks)} chunks"
        )

    # Detect signals across the whole corpus so the summary covers every
    # chunk, not just the top-N. The per-entry detection below only has
    # to re-run on the top-N rows so we don't pay for it twice.
    all_signal_lists: List[List[NoiseSignal]] = [
        detect_noise_signals(c.text) for c in chunks
    ]
    noise_summary = aggregate_signals(all_signal_lists)

    top_indices = sorted(
        range(len(chunks)),
        key=lambda i: token_lengths[i],
        reverse=True,
    )[: max(0, top_n)]

    long_entries: List[ChunkAuditEntry] = []
    for i in top_indices:
        c = chunks[i]
        long_entries.append(ChunkAuditEntry(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
            section_path=c.section_path,
            chunk_index=c.chunk_index,
            char_count=len(c.text),
            token_count=int(token_lengths[i]),
            preview=_make_preview(c.text, limit=PREVIEW_CHARS),
            detected_noise_signals=list(all_signal_lists[i]),
        ))

    return LongChunkAuditReport(
        corpus_path=str(path),
        document_count=len(doc_ids),
        chunk_count=len(chunks),
        tokenizer=tokenizer_name,
        top_n=top_n,
        long_chunks=long_entries,
        noise_signal_summary=noise_summary,
    )


# --- Length comparison -------------------------------------------------


def compare_raw_vs_cleaned(
    corpus_path: Path,
    *,
    token_counter: Optional[TokenCounter] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    batch_size: int = 256,
    cleaner: Callable[[str], CleaningResult] = clean_chunk,
) -> LengthComparisonReport:
    """Run the cleaner over the corpus and compare length distributions.

    No raw corpus file is mutated. The cleaner runs in-memory and we
    aggregate counters as we go. ``cleaner`` is parameterized so tests
    can swap in a stub; the default is the real conservative cleaner.

    Chunks the cleaner empties are *dropped* from the cleaned bucket
    (with their drop_reason counted) — they don't show up as zero-length
    rows in the cleaned distribution, which would otherwise drag the
    cleaned percentiles to zero.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    raw_texts: List[str] = []
    cleaned_texts: List[str] = []
    drop_reasons: Dict[str, int] = {}
    total_removed_lines = 0
    total_collapsed_repeats = 0

    for chunk in _iter_corpus_chunks(path):
        raw_texts.append(chunk.text)
        result = cleaner(chunk.text)
        total_removed_lines += result.removed_lines
        total_collapsed_repeats += result.collapsed_repeats
        if result.drop_reason is not None:
            drop_reasons[result.drop_reason] = (
                drop_reasons.get(result.drop_reason, 0) + 1
            )
            continue
        cleaned_texts.append(result.text)

    if not raw_texts:
        raise RuntimeError(
            f"Corpus {path} produced zero chunks — empty or wrong schema."
        )

    counter = token_counter or _default_token_counter(tokenizer_name)
    raw_tokens = _count_in_batches(
        raw_texts, counter=counter, batch_size=batch_size,
    )
    cleaned_tokens = (
        _count_in_batches(
            cleaned_texts, counter=counter, batch_size=batch_size,
        )
        if cleaned_texts
        else []
    )

    raw_bucket = _bucket_from_lengths(
        label="raw",
        char_lengths=[len(t) for t in raw_texts],
        token_lengths=raw_tokens,
        thresholds=thresholds,
    )
    cleaned_bucket = _bucket_from_lengths(
        label="cleaned",
        char_lengths=[len(t) for t in cleaned_texts],
        token_lengths=cleaned_tokens,
        thresholds=thresholds,
    )

    return LengthComparisonReport(
        corpus_path=str(path),
        tokenizer=tokenizer_name,
        thresholds=list(thresholds),
        raw=raw_bucket,
        cleaned=cleaned_bucket,
        raw_chunk_count=len(raw_texts),
        cleaned_chunk_count=len(cleaned_texts),
        dropped_chunk_count=sum(drop_reasons.values()),
        drop_reasons=drop_reasons,
        cleaner_total_removed_lines=total_removed_lines,
        cleaner_total_collapsed_repeats=total_collapsed_repeats,
    )


def _bucket_from_lengths(
    *,
    label: str,
    char_lengths: Sequence[int],
    token_lengths: Sequence[int],
    thresholds: Sequence[int],
) -> LengthBucket:
    over: Dict[int, int] = {}
    for t in thresholds:
        over[int(t)] = sum(1 for v in token_lengths if v > t)
    return LengthBucket(
        label=label,
        char=_distribution(char_lengths),
        token=_distribution(token_lengths),
        chunks_over_token_threshold=over,
    )


def _distribution(values: Sequence[int]) -> LengthDistribution:
    if not values:
        return LengthDistribution(
            count=0, mean=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0,
        )
    floats = [float(v) for v in values]
    return LengthDistribution(
        count=len(values),
        mean=round(statistics.fmean(floats), 2),
        p50=p_percentile(floats, 50),
        p90=p_percentile(floats, 90),
        p95=p_percentile(floats, 95),
        p99=p_percentile(floats, 99),
        max=int(max(values)),
    )


# --- Serializers --------------------------------------------------------


def audit_to_dict(report: LongChunkAuditReport) -> Dict[str, Any]:
    payload = asdict(report)
    payload["long_chunks"] = [
        {
            "chunk_id": e.chunk_id,
            "doc_id": e.doc_id,
            "title": e.title,
            "section_path": e.section_path,
            "chunk_index": e.chunk_index,
            "char_count": e.char_count,
            "token_count": e.token_count,
            "preview": e.preview,
            "detected_noise_signals": [signal_to_dict(s) for s in e.detected_noise_signals],
        }
        for e in report.long_chunks
    ]
    return payload


def length_comparison_to_dict(report: LengthComparisonReport) -> Dict[str, Any]:
    payload = asdict(report)
    # Threshold dict keys to strings for JSON friendliness.
    for bucket_key in ("raw", "cleaned"):
        bucket = payload[bucket_key]
        bucket["chunks_over_token_threshold"] = {
            str(k): v for k, v in bucket["chunks_over_token_threshold"].items()
        }
    return payload


def render_audit_markdown(report: LongChunkAuditReport) -> str:
    lines: List[str] = []
    lines.append("# Long-chunk corpus audit")
    lines.append("")
    lines.append(f"- corpus: `{report.corpus_path}`")
    lines.append(f"- documents: {report.document_count}")
    lines.append(f"- chunks: {report.chunk_count}")
    lines.append(f"- tokenizer: `{report.tokenizer}`")
    lines.append(f"- top_n reported: {report.top_n}")
    lines.append("")

    lines.append("## Noise signal summary (whole corpus)")
    lines.append("")
    if report.noise_signal_summary:
        lines.append("| signal | total occurrences |")
        lines.append("|---|---:|")
        for name in sorted(report.noise_signal_summary.keys()):
            lines.append(f"| {name} | {report.noise_signal_summary[name]} |")
    else:
        lines.append("_No noise signals detected._")
    lines.append("")

    lines.append(f"## Top {len(report.long_chunks)} longest chunks (by tokens)")
    lines.append("")
    if report.long_chunks:
        lines.append(
            "| rank | doc_id | title | section | idx | chars | tokens | "
            "noise_signals | preview |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---|---|")
        for rank, entry in enumerate(report.long_chunks, start=1):
            preview = entry.preview.replace("|", "\\|").replace("\n", " ")
            title = entry.title.replace("|", "\\|").replace("\n", " ")
            section = entry.section_path.replace("|", "\\|").replace("\n", " ")
            signals = (
                ", ".join(
                    f"{s.name}×{s.occurrences}"
                    for s in entry.detected_noise_signals
                )
                or "—"
            )
            lines.append(
                f"| {rank} | {entry.doc_id} | {title} | {section} | "
                f"{entry.chunk_index} | {entry.char_count} | "
                f"{entry.token_count} | {signals} | {preview} |"
            )
    else:
        lines.append("_No chunks reported._")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_length_comparison_markdown(
    report: LengthComparisonReport,
) -> str:
    lines: List[str] = []
    lines.append("# Corpus length comparison: raw vs cleaned")
    lines.append("")
    lines.append(f"- corpus: `{report.corpus_path}`")
    lines.append(f"- tokenizer: `{report.tokenizer}`")
    lines.append(f"- raw chunks: {report.raw_chunk_count}")
    lines.append(f"- cleaned chunks: {report.cleaned_chunk_count}")
    lines.append(f"- dropped chunks: {report.dropped_chunk_count}")
    if report.drop_reasons:
        for reason in sorted(report.drop_reasons.keys()):
            lines.append(f"  - `{reason}`: {report.drop_reasons[reason]}")
    lines.append(
        f"- cleaner removed lines (total): {report.cleaner_total_removed_lines}"
    )
    lines.append(
        f"- cleaner collapsed repeats (total): {report.cleaner_total_collapsed_repeats}"
    )
    lines.append("")

    lines.append("## Char distribution")
    lines.append("")
    lines.append(_dist_table_header())
    lines.append(_dist_row("raw", report.raw.char))
    lines.append(_dist_row("cleaned", report.cleaned.char))
    lines.append("")

    lines.append("## Token distribution")
    lines.append("")
    lines.append(_dist_table_header())
    lines.append(_dist_row("raw", report.raw.token))
    lines.append(_dist_row("cleaned", report.cleaned.token))
    lines.append("")

    lines.append("## Chunks over token threshold")
    lines.append("")
    lines.append("| threshold | raw | cleaned | delta |")
    lines.append("|---:|---:|---:|---:|")
    for t in sorted(report.raw.chunks_over_token_threshold.keys()):
        raw_n = report.raw.chunks_over_token_threshold[t]
        clean_n = report.cleaned.chunks_over_token_threshold.get(t, 0)
        lines.append(f"| > {t} | {raw_n} | {clean_n} | {clean_n - raw_n} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _dist_table_header() -> str:
    return (
        "| bucket | count | mean | p50 | p90 | p95 | p99 | max |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )


def _dist_row(label: str, dist: LengthDistribution) -> str:
    return (
        f"| {label} | {dist.count} | {dist.mean:.2f} | "
        f"{dist.p50:.0f} | {dist.p90:.0f} | {dist.p95:.0f} | "
        f"{dist.p99:.0f} | {dist.max} |"
    )
