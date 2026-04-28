"""Doc-level boost metadata extracted from a corpus JSONL once.

The retriever's ``RetrievedChunk`` only carries ``(doc_id, section,
text, score)`` — the boost scorer needs the document's title (or its
``seed`` field, for namu-wiki dumps where the seed is the article
title) and its section list. Neither is on the chunk itself, so we
read the corpus once at startup and build a doc_id → DocBoostMetadata
map that the scorer queries by doc_id.

The reader is read-only; we never mutate the corpus on disk and never
touch the FAISS / embedder side. This is also why ``load_doc_metadata``
takes a corpus path rather than a Retriever — it should work just as
well from a unit test that points at a tiny fixture as from the live
B2 token-aware-v1 corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from eval.harness.query_normalizer import normalize_for_match


@dataclass(frozen=True)
class DocBoostMetadata:
    """Per-document boost-relevant fields.

    All ``normalized_*`` fields are pre-folded via
    ``query_normalizer.normalize_for_match`` so the scorer's hot loop
    only does substring checks — no per-chunk normalization cost.
    """

    doc_id: str
    title: str
    normalized_title: str
    seed: str
    normalized_seed: str
    section_names: Tuple[str, ...] = field(default_factory=tuple)
    normalized_section_names: Tuple[str, ...] = field(default_factory=tuple)


def _read_jsonl(path: Path) -> Iterable[Mapping]:
    """Stream a JSONL file line-by-line, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_section_names(sections_raw) -> List[str]:
    """Pull section names from the namu-wiki corpus shape.

    The corpus shape is::

        {"sections": {"<name>": {"chunks": [...], ...}, ...}, ...}

    Returns an ordered list of unique non-empty section names. Order
    follows the dict iteration order, which matches ``section_order``
    in practice for the B2 corpus.
    """
    out: List[str] = []
    if not isinstance(sections_raw, dict):
        return out
    seen: set = set()
    for k in sections_raw.keys():
        name = str(k).strip()
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def load_doc_metadata(corpus_path: Path) -> Dict[str, DocBoostMetadata]:
    """One-shot read of the corpus → ``doc_id → DocBoostMetadata``.

    Documents missing a ``doc_id`` are skipped (matches the offline
    corpus builder's contract). Missing ``title`` / ``seed`` /
    ``sections`` fall back to empty values so the scorer always has
    something to compare against.
    """
    if not isinstance(corpus_path, Path):
        corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    out: Dict[str, DocBoostMetadata] = {}
    for raw in _read_jsonl(corpus_path):
        doc_id = str(raw.get("doc_id") or "").strip()
        if not doc_id:
            continue
        title = str(raw.get("title") or "").strip()
        seed = str(raw.get("seed") or "").strip()
        section_names = _extract_section_names(raw.get("sections"))
        out[doc_id] = DocBoostMetadata(
            doc_id=doc_id,
            title=title,
            normalized_title=normalize_for_match(title),
            seed=seed,
            normalized_seed=normalize_for_match(seed),
            section_names=tuple(section_names),
            normalized_section_names=tuple(
                normalize_for_match(s) for s in section_names
            ),
        )
    return out


def doc_metadata_from_records(
    records: Iterable[Mapping],
) -> Dict[str, DocBoostMetadata]:
    """In-memory sibling of ``load_doc_metadata`` for tests / fixtures."""
    out: Dict[str, DocBoostMetadata] = {}
    for raw in records:
        doc_id = str(raw.get("doc_id") or "").strip()
        if not doc_id:
            continue
        title = str(raw.get("title") or "").strip()
        seed = str(raw.get("seed") or "").strip()
        section_names = _extract_section_names(raw.get("sections"))
        out[doc_id] = DocBoostMetadata(
            doc_id=doc_id,
            title=title,
            normalized_title=normalize_for_match(title),
            seed=seed,
            normalized_seed=normalize_for_match(seed),
            section_names=tuple(section_names),
            normalized_section_names=tuple(
                normalize_for_match(s) for s in section_names
            ),
        )
    return out
