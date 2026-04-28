"""Phase 1C — emit a token-aware-chunked corpus artifact.

Reads a source ``corpus.jsonl`` and rewrites every section's
``chunks`` list with the output of the token-aware chunker. The
schema of the emitted file is identical to the source so the
downstream offline retrieval stack reads it without changes:

    {
      "doc_id": ...,
      "title": ...,
      "sections": {
        "<section_name>": {
          "chunks": [ "...", "...", ... ],   # rewritten
          "text":   ...,                     # left in place (ignored by ingest
                                             # when ``chunks`` is non-empty)
          "list":   [...],                   # same
          ...other metadata kept verbatim
        },
        ...
      },
      ...
    }

We also dump per-emitted-chunk provenance into ``chunks_provenance.jsonl``
so downstream analysis (length comparison, retrieval drift
investigation) can trace each retrievable chunk back to its source
section + split strategy.

Public surface
--------------
- ``EmitConfig``                   — knobs + token-aware config bundle
- ``EmitSummary``                  — corpus-level rollup
- ``emit_token_aware_corpus``      — main entry point (writes files)
- ``emit_summary_to_dict``         — JSON-friendly serializer
- ``render_emit_summary_markdown``
"""

from __future__ import annotations

import json
import logging
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

from app.capabilities.rag.ingest import _iter_documents
from app.capabilities.rag.token_aware_chunker import (
    CHUNKER_VERSION,
    TokenAwareChunk,
    TokenAwareConfig,
    TokenizerDecodeFn,
    TokenizerEncodeFn,
    SingleTokenCounter,
    raw_section_units,
    token_aware_chunks_from_section,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datatypes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmitConfig:
    """Bundle of token-aware config + emit options."""

    chunker: TokenAwareConfig
    write_provenance: bool = True


@dataclass
class EmitSummary:
    source_corpus: str
    output_corpus: str
    chunker_version: str = CHUNKER_VERSION
    target_tokens: int = 0
    soft_max_tokens: int = 0
    hard_max_tokens: int = 0
    overlap_tokens: int = 0
    document_count: int = 0
    section_count: int = 0

    # Input payload accounting (mirrors corpus_preprocessor's
    # chunks_processed). One source unit = one entry from
    # ``chunks`` / ``list`` / ``text`` *before* token-aware splitting.
    input_payload_unit_count: int = 0

    # Output retrievable chunks (what the offline retrieval stack ends
    # up indexing).
    output_chunk_count: int = 0

    # Per-strategy + fallback rollups.
    chunks_by_strategy: Dict[str, int] = field(default_factory=dict)
    fallback_used_count: int = 0
    chunks_over_hard_max: int = 0  # populated when allow_hard_max_overflow
    sections_emitting_zero_chunks: int = 0


# ---------------------------------------------------------------------------
# Tokenizer plumbing.
# ---------------------------------------------------------------------------


def build_default_tokenizer_callables(
    model_name: str,
) -> Tuple[SingleTokenCounter, TokenizerEncodeFn, TokenizerDecodeFn]:
    """Build (counter, encode, decode) bound to a HF AutoTokenizer.

    All three are needed so the hard-token fallback can slice on token
    ids and the overlap-builder can decode trailing tokens. The
    counter wraps ``tokenizer(text, add_special_tokens=True)`` so the
    counts match what ``max_seq_length`` truncates against.
    """
    log.info("Loading tokenizer: %s", model_name)
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _count(text: str) -> int:
        if not text:
            return 0
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def _encode(text: str) -> List[int]:
        if not text:
            return []
        encoded = tokenizer(
            text,
            add_special_tokens=False,  # raw stream — no <s>/</s>
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return list(encoded["input_ids"])

    def _decode(ids: Sequence[int]) -> str:
        if not ids:
            return ""
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    return _count, _encode, _decode


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def emit_token_aware_corpus(
    source_corpus: Path,
    out_corpus: Path,
    *,
    config: EmitConfig,
    token_counter: SingleTokenCounter,
    encode_fn: Optional[TokenizerEncodeFn] = None,
    decode_fn: Optional[TokenizerDecodeFn] = None,
    provenance_path: Optional[Path] = None,
) -> EmitSummary:
    """Stream a corpus through the token-aware chunker and write artifacts.

    Returns an ``EmitSummary``. Writes:
      - ``out_corpus`` — the rewritten corpus jsonl (same schema as source)
      - ``provenance_path`` — per-chunk provenance jsonl (when set)

    The source corpus is not modified.
    """
    if not source_corpus.exists():
        raise FileNotFoundError(f"Source corpus not found: {source_corpus}")
    if out_corpus.resolve() == source_corpus.resolve():
        raise ValueError(
            "Refusing to overwrite the source corpus — pick a different "
            "output path."
        )

    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    if provenance_path is not None:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)

    summary = EmitSummary(
        source_corpus=str(source_corpus),
        output_corpus=str(out_corpus),
        chunker_version=CHUNKER_VERSION,
        target_tokens=config.chunker.target_tokens,
        soft_max_tokens=config.chunker.soft_max_tokens,
        hard_max_tokens=config.chunker.hard_max_tokens,
        overlap_tokens=config.chunker.overlap_tokens,
    )

    prov_fp = None
    if provenance_path is not None and config.write_provenance:
        prov_fp = provenance_path.open("w", encoding="utf-8")

    try:
        with out_corpus.open("w", encoding="utf-8") as out_fp:
            for raw_doc in _iter_documents(source_corpus):
                summary.document_count += 1
                doc_id = str(
                    raw_doc.get("doc_id")
                    or raw_doc.get("seed")
                    or raw_doc.get("title")
                    or ""
                ).strip() or "<unknown>"

                new_doc = dict(raw_doc)
                sections = raw_doc.get("sections")
                if not isinstance(sections, dict):
                    out_fp.write(json.dumps(new_doc, ensure_ascii=False))
                    out_fp.write("\n")
                    continue

                new_sections: Dict[str, Any] = {}
                for sname, payload in sections.items():
                    if not isinstance(payload, dict):
                        new_sections[sname] = payload
                        continue
                    summary.section_count += 1
                    summary.input_payload_unit_count += _count_input_units(payload)

                    chunks = token_aware_chunks_from_section(
                        payload,
                        config=config.chunker,
                        token_counter=token_counter,
                        encode_fn=encode_fn,
                        decode_fn=decode_fn,
                    )

                    new_payload = _rewrite_section(payload, chunks)
                    new_sections[sname] = new_payload

                    if not chunks:
                        summary.sections_emitting_zero_chunks += 1

                    for ch in chunks:
                        summary.output_chunk_count += 1
                        summary.chunks_by_strategy[ch.split_strategy] = (
                            summary.chunks_by_strategy.get(ch.split_strategy, 0) + 1
                        )
                        if ch.fallback_used:
                            summary.fallback_used_count += 1
                        if ch.token_count > config.chunker.hard_max_tokens:
                            summary.chunks_over_hard_max += 1
                        if prov_fp is not None:
                            prov_fp.write(json.dumps({
                                "doc_id": doc_id,
                                "section_path": str(sname),
                                "chunk_index": ch.chunk_index,
                                "char_count": ch.char_count,
                                "token_count": ch.token_count,
                                "split_strategy": ch.split_strategy,
                                "fallback_used": ch.fallback_used,
                                "chunker_version": CHUNKER_VERSION,
                            }, ensure_ascii=False))
                            prov_fp.write("\n")

                new_doc["sections"] = new_sections
                out_fp.write(json.dumps(new_doc, ensure_ascii=False))
                out_fp.write("\n")

                if summary.document_count % 100 == 0:
                    log.info(
                        "Emitted %d docs · %d output chunks so far",
                        summary.document_count, summary.output_chunk_count,
                    )
    finally:
        if prov_fp is not None:
            prov_fp.close()

    log.info(
        "Emit complete: %d docs · %d sections · %d input units → %d output chunks",
        summary.document_count, summary.section_count,
        summary.input_payload_unit_count, summary.output_chunk_count,
    )
    return summary


def _count_input_units(payload: Mapping[str, Any]) -> int:
    """Count ``chunks`` / ``list`` / ``text`` payload units before splitting.

    Lines up with ``corpus_preprocessor.chunks_processed`` so the
    accounting note in the report can compare apples-to-apples
    between transformed-payload-units and final-retrievable-chunks.
    """
    pre = payload.get("chunks")
    if isinstance(pre, list) and any(
        isinstance(x, (str, int, float)) and str(x).strip() for x in pre
    ):
        # Count the actual non-empty payload entries, not just the
        # list length, so blank entries don't inflate the count.
        return sum(
            1 for x in pre
            if isinstance(x, (str, int, float)) and str(x).strip()
        )

    list_entries = payload.get("list")
    if isinstance(list_entries, list):
        count = 0
        for entry in list_entries:
            if isinstance(entry, dict):
                if str(entry.get("name", "")).strip() or str(entry.get("desc", "")).strip():
                    count += 1
        if count > 0:
            return count

    blob = payload.get("text")
    if isinstance(blob, str) and blob.strip():
        return 1

    return 0


def _rewrite_section(
    payload: Mapping[str, Any],
    chunks: Sequence[TokenAwareChunk],
) -> Dict[str, Any]:
    """Replace ``chunks`` with token-aware text + clear the other
    chunker-reachable fields (``text``, ``list``).

    Critical: ``ingest._chunks_from_section`` *concatenates*
    ``chunks`` + ``list`` (only ``text`` is gated on emptiness). If we
    leave ``list`` populated, the offline retrieval stack will feed
    the original raw list entries back through ``window_by_chars`` on
    top of our token-aware chunks — defeating the entire hard-cap
    invariant. Same goes for ``text`` to a lesser extent (it only
    fires when ``chunks`` is empty, but we still drop it to shrink
    the artifact).

    We preserve the original sizes in metadata fields
    (``text_chars_original``, ``list_entries_original``) so debugging
    + provenance traces still have the source-shape signal.

    Other metadata (``urls``, ``model``, ``ts``, etc.) is left
    untouched.
    """
    new_payload = dict(payload)
    new_payload["chunks"] = [c.text for c in chunks]

    if "text" in new_payload:
        original_blob = new_payload.pop("text")
        if isinstance(original_blob, str):
            new_payload["text_chars_original"] = len(original_blob)

    if "list" in new_payload:
        original_list = new_payload.pop("list")
        if isinstance(original_list, list):
            new_payload["list_entries_original"] = len(original_list)
            # Replace with an empty list so downstream code that
            # iterates ``list`` keeps working without re-running over
            # raw entries.
            new_payload["list"] = []

    return new_payload


# ---------------------------------------------------------------------------
# Serializers.
# ---------------------------------------------------------------------------


def emit_summary_to_dict(summary: EmitSummary) -> Dict[str, Any]:
    return {
        "chunker_version": summary.chunker_version,
        "source_corpus": summary.source_corpus,
        "output_corpus": summary.output_corpus,
        "config": {
            "target_tokens": summary.target_tokens,
            "soft_max_tokens": summary.soft_max_tokens,
            "hard_max_tokens": summary.hard_max_tokens,
            "overlap_tokens": summary.overlap_tokens,
        },
        "document_count": summary.document_count,
        "section_count": summary.section_count,
        "input_payload_unit_count": summary.input_payload_unit_count,
        "output_chunk_count": summary.output_chunk_count,
        "chunks_by_strategy": summary.chunks_by_strategy,
        "fallback_used_count": summary.fallback_used_count,
        "chunks_over_hard_max": summary.chunks_over_hard_max,
        "sections_emitting_zero_chunks": summary.sections_emitting_zero_chunks,
    }


def render_emit_summary_markdown(summary: EmitSummary) -> str:
    lines: List[str] = []
    lines.append("# Phase 1C — token-aware corpus emit summary")
    lines.append("")
    lines.append(f"- chunker_version: `{summary.chunker_version}`")
    lines.append(f"- source: `{summary.source_corpus}`")
    lines.append(f"- output: `{summary.output_corpus}`")
    lines.append(
        f"- config: target={summary.target_tokens} · "
        f"soft_max={summary.soft_max_tokens} · "
        f"hard_max={summary.hard_max_tokens} · "
        f"overlap={summary.overlap_tokens}"
    )
    lines.append(f"- documents: {summary.document_count}")
    lines.append(f"- sections: {summary.section_count}")
    lines.append("")
    lines.append("## Accounting")
    lines.append("")
    lines.append(
        f"- **input payload units**: {summary.input_payload_unit_count} "
        "_(non-empty entries from ``chunks`` / ``list`` / ``text``, before splitting)_"
    )
    lines.append(
        f"- **output retrievable chunks**: {summary.output_chunk_count} "
        "_(what the FAISS index would contain)_"
    )
    if summary.input_payload_unit_count > 0:
        ratio = summary.output_chunk_count / summary.input_payload_unit_count
        lines.append(
            f"- output / input ratio: **{ratio:.3f}× **"
        )
    lines.append("")
    lines.append("## Strategy breakdown")
    lines.append("")
    lines.append("| split_strategy | chunks |")
    lines.append("|---|---:|")
    for strategy in sorted(
        summary.chunks_by_strategy.keys(),
        key=lambda s: -summary.chunks_by_strategy[s],
    ):
        lines.append(
            f"| `{strategy}` | {summary.chunks_by_strategy[strategy]} |"
        )
    lines.append("")
    lines.append("## Fallback / overflow")
    lines.append("")
    lines.append(
        f"- chunks needing hard-token / hard-char fallback: "
        f"{summary.fallback_used_count}"
    )
    lines.append(
        f"- chunks > hard_max_tokens (should be 0 in production emit): "
        f"{summary.chunks_over_hard_max}"
    )
    lines.append(
        f"- sections emitting zero chunks: "
        f"{summary.sections_emitting_zero_chunks}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"
