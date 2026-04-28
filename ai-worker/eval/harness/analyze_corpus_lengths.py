"""Tokenizer-based chunk-length analyzer for an offline corpus.

Reads ``corpus.jsonl`` through the same chunker the offline retrieval
stack uses (``ingest._chunks_from_section``) and reports both
character-length and token-length distributions, plus how many chunks
the embedder would truncate at common ``max_seq_length`` caps.

Why this exists
---------------
The Phase 0 baseline tradeoff document originally reported a
back-of-the-envelope "p95 chunk = 3,817 chars ≈ 954 tokens" estimate
that does not survive contact with the actual bge-m3 tokenizer:
Korean text on namu-wiki has a highly variable bytes-per-token ratio
(SentencePiece BPE), so char-derived token estimates are off by a
factor of 2-4× depending on the chunk. We need a measured distribution
before we can claim the 1024-cap impact is "small".

Public surface
--------------
- ``ChunkLengthSample``                — one of the longest chunks
- ``LengthAnalysis``                   — frozen-shaped dataclass for the report
- ``analyze_corpus_lengths``           — main entry point
- ``length_analysis_to_dict``          — JSON-friendly serializer
- ``render_length_analysis_markdown``  — markdown for eyeball review

Tokenization
------------
By default the analyzer loads ``AutoTokenizer.from_pretrained(
"BAAI/bge-m3")`` and tokenizes each chunk with ``add_special_tokens=
True``. That count includes the model's ``<s>`` + ``</s>`` markers and
matches what ``max_seq_length`` truncates against — so a chunk
reported here as 1,024 tokens is exactly the boundary at which the
encoder starts dropping content.

Tests inject a ``token_counter`` callable so they don't need the real
tokenizer; the CLI uses the default loader.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from app.capabilities.rag.ingest import _chunks_from_section, _iter_documents

log = logging.getLogger(__name__)


# Buckets we report ``chunks_over_<N>_tokens`` for. 512/1024/2048/4096
# cover the practical caps for bge-m3 + a 16 GB GPU; 8192 is the
# model's own default window and the implicit "no truncation" mark.
DEFAULT_THRESHOLDS: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
DEFAULT_TOP_LONGEST = 20
PREVIEW_CHARS = 160
DEFAULT_TOKENIZER_NAME = "BAAI/bge-m3"

# Percentile points we report for char and token lengths.
PERCENTILES: Tuple[int, ...] = (50, 90, 95, 99)


TokenCounter = Callable[[Sequence[str]], List[int]]


@dataclass(frozen=True)
class ChunkLengthSample:
    chunk_id: str
    doc_id: str
    section: str
    order: int
    char_length: int
    token_length: int
    preview: str


@dataclass
class LengthDistribution:
    """Single-axis (chars or tokens) summary stats."""

    count: int
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: int


@dataclass
class LengthAnalysis:
    corpus_path: str
    document_count: int
    chunk_count: int
    tokenizer: str
    char_length: LengthDistribution
    token_length: LengthDistribution
    chunks_over_token_threshold: Dict[int, int]
    chunks_over_token_threshold_ratio: Dict[int, float]
    longest_chunks: List[ChunkLengthSample] = field(default_factory=list)


def analyze_corpus_lengths(
    corpus_path: Path,
    *,
    token_counter: Optional[TokenCounter] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    top_longest: int = DEFAULT_TOP_LONGEST,
    batch_size: int = 256,
) -> LengthAnalysis:
    """Run the full analysis over ``corpus_path``.

    The chunker is the same one the offline retrieval stack uses, so
    the chunk count here matches the FAISS index built by
    ``offline_corpus.py``. ``token_counter`` defaults to a real
    bge-m3 tokenizer; tests override it with a deterministic stub.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    if top_longest < 0:
        raise ValueError("top_longest must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks: List[Tuple[str, str, str, int, str]] = []
    doc_ids: set[str] = set()

    for raw in _iter_documents(path):
        doc_id = str(
            raw.get("doc_id") or raw.get("seed") or raw.get("title") or ""
        ).strip()
        if not doc_id:
            continue
        doc_ids.add(doc_id)
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
                # chunk_id is informational only — not the same hash
                # the FAISS row uses, but stable enough for the report.
                chunk_id = f"{doc_id}::{section_name}::{order}"
                chunks.append(
                    (chunk_id, doc_id, section_name, order, chunk_text)
                )

    if not chunks:
        raise RuntimeError(
            f"Corpus {path} produced zero chunks — empty or wrong schema."
        )

    log.info(
        "Analyzing %d chunks from %d docs in %s (tokenizer=%s)",
        len(chunks), len(doc_ids), path, tokenizer_name,
    )

    counter = token_counter or _default_token_counter(tokenizer_name)
    char_lengths = [len(c[4]) for c in chunks]
    token_lengths = _count_in_batches(
        [c[4] for c in chunks], counter=counter, batch_size=batch_size,
    )
    if len(token_lengths) != len(chunks):
        raise RuntimeError(
            "token_counter returned a mismatched length: "
            f"{len(token_lengths)} vs {len(chunks)} chunks"
        )

    threshold_counts: Dict[int, int] = {}
    threshold_ratios: Dict[int, float] = {}
    n = len(token_lengths)
    for t in thresholds:
        c = sum(1 for v in token_lengths if v > t)
        threshold_counts[int(t)] = c
        threshold_ratios[int(t)] = round(c / n, 6) if n else 0.0

    top_indices = sorted(
        range(len(chunks)),
        key=lambda i: token_lengths[i],
        reverse=True,
    )[: max(0, top_longest)]
    longest = [
        ChunkLengthSample(
            chunk_id=chunks[i][0],
            doc_id=chunks[i][1],
            section=chunks[i][2],
            order=chunks[i][3],
            char_length=char_lengths[i],
            token_length=token_lengths[i],
            preview=_make_preview(chunks[i][4]),
        )
        for i in top_indices
    ]

    return LengthAnalysis(
        corpus_path=str(path),
        document_count=len(doc_ids),
        chunk_count=len(chunks),
        tokenizer=tokenizer_name,
        char_length=_distribution(char_lengths),
        token_length=_distribution(token_lengths),
        chunks_over_token_threshold=threshold_counts,
        chunks_over_token_threshold_ratio=threshold_ratios,
        longest_chunks=longest,
    )


# ---------------------------------------------------------------------------
# Tokenizer plumbing.
# ---------------------------------------------------------------------------


def _default_token_counter(model_name: str) -> TokenCounter:
    """Return a callable that batches strings through HF AutoTokenizer.

    Loaded lazily so importing this module does not pull in the
    transformers stack (which is heavy on cold start). The returned
    counter calls ``tokenizer(batch, add_special_tokens=True,
    truncation=False)`` so the count reflects the model's full
    decoded length — including ``<s>`` / ``</s>`` — i.e. exactly what
    ``max_seq_length`` truncates against.
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
            log.info("Tokenized %d / %d chunks", i + len(batch), n)
    return out


# ---------------------------------------------------------------------------
# Stats helpers.
# ---------------------------------------------------------------------------


def _distribution(values: Sequence[int]) -> LengthDistribution:
    if not values:
        return LengthDistribution(
            count=0, mean=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0,
        )
    sorted_vals = sorted(values)
    return LengthDistribution(
        count=len(values),
        mean=round(statistics.fmean(values), 2),
        p50=_percentile(sorted_vals, 50),
        p90=_percentile(sorted_vals, 90),
        p95=_percentile(sorted_vals, 95),
        p99=_percentile(sorted_vals, 99),
        max=int(sorted_vals[-1]),
    )


def _percentile(sorted_vals: Sequence[int], pct: int) -> float:
    """Nearest-rank percentile.

    Uses ``ceil(pct/100 * n) - 1`` on the sorted list — byte-identical
    to ``eval.harness.metrics.p_percentile`` so corpus length stats
    here line up with the latency percentile stats elsewhere in the
    harness.
    """
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    idx = int(math.ceil(pct / 100.0 * len(sorted_vals))) - 1
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def _make_preview(text: str, *, limit: int = PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# ---------------------------------------------------------------------------
# Serializers.
# ---------------------------------------------------------------------------


def length_analysis_to_dict(analysis: LengthAnalysis) -> Dict[str, Any]:
    payload = asdict(analysis)
    # Keep threshold keys as strings in JSON (json.dumps converts int
    # keys to strings anyway, but being explicit avoids surprises for
    # downstream readers).
    payload["chunks_over_token_threshold"] = {
        str(k): v for k, v in analysis.chunks_over_token_threshold.items()
    }
    payload["chunks_over_token_threshold_ratio"] = {
        str(k): v for k, v in analysis.chunks_over_token_threshold_ratio.items()
    }
    return payload


def render_length_analysis_markdown(analysis: LengthAnalysis) -> str:
    lines: List[str] = []
    lines.append("# Corpus chunk-length analysis")
    lines.append("")
    lines.append(
        f"- corpus: `{analysis.corpus_path}`"
    )
    lines.append(f"- documents: {analysis.document_count}")
    lines.append(f"- chunks: {analysis.chunk_count}")
    lines.append(f"- tokenizer: `{analysis.tokenizer}`")
    lines.append("")
    lines.append(
        "Token counts include the model's special tokens "
        "(``<s>`` / ``</s>``), so they line up with what "
        "``max_seq_length`` truncates against."
    )
    lines.append("")

    lines.append("## Length distribution")
    lines.append("")
    lines.append("| axis | count | mean | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, dist in (
        ("char_length", analysis.char_length),
        ("token_length", analysis.token_length),
    ):
        lines.append(
            f"| {label} | {dist.count} | "
            f"{dist.mean:.2f} | {dist.p50:.0f} | {dist.p90:.0f} | "
            f"{dist.p95:.0f} | {dist.p99:.0f} | {dist.max} |"
        )
    lines.append("")

    lines.append("## Chunks above max_seq_length cap")
    lines.append("")
    lines.append(
        "How many chunks would have their tail truncated at each cap. "
        "Phase 0 baselines were embedded with `max_seq_length=1024` to "
        "stay inside a 16 GB GPU; the row marked `> 1024` is the "
        "directly load-bearing one for the existing baselines."
    )
    lines.append("")
    lines.append("| threshold (tokens) | chunks over threshold | ratio |")
    lines.append("|---|---:|---:|")
    for t in sorted(analysis.chunks_over_token_threshold.keys()):
        count = analysis.chunks_over_token_threshold[t]
        ratio = analysis.chunks_over_token_threshold_ratio.get(t, 0.0)
        lines.append(
            f"| > {t} | {count} | {ratio * 100:.2f}% |"
        )
    lines.append("")

    if analysis.longest_chunks:
        lines.append(
            f"## Top {len(analysis.longest_chunks)} longest chunks (by tokens)"
        )
        lines.append("")
        lines.append(
            "| rank | doc_id | section | order | chars | tokens | preview |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for rank, sample in enumerate(analysis.longest_chunks, start=1):
            preview = sample.preview.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {rank} | {sample.doc_id} | {sample.section} | "
                f"{sample.order} | {sample.char_length} | {sample.token_length} | "
                f"{preview} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
