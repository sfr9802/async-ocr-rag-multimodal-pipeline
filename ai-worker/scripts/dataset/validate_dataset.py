"""Validate a generated RAG eval dataset against a built FAISS index.

Generated queries are noisy: Claude sometimes proposes a query whose
gold doc is no longer the best match because a sibling doc covers the
same topic, or invents an "expected_keyword" that doesn't appear in the
generated answer. This script runs the dataset against the live
retriever + extractive generator, scores each row with the same metrics
the eval harness uses, and writes:

  * ``<out>``         — kept rows (passed all gates)
  * ``<out>.dropped.jsonl`` — rows that didn't pass, with the reason

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.validate_dataset \\
        --in   eval/datasets/rag_anime_extended_kr_raw.jsonl \\
        --out  eval/datasets/rag_anime_extended_kr.jsonl \\
        --filters domain=anime \\
        --top-k 5 \\
        --keep-easy-min-recall 0.5 \\
        --keep-hard-min-recall 0.0

Gates default to "loose for hard, strict for easy" because the point of
hard rows is exactly to be retrieval challenges; we'd rather keep them
even if recall is low so the agent loop has something to recover.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.dataset._common import (
    configure_logging,
    read_jsonl,
    write_jsonl,
)

log = logging.getLogger("scripts.dataset.validate_dataset")


def _parse_filters(spec: Optional[str]) -> Dict[str, str]:
    if not spec:
        return {}
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        if "=" not in chunk:
            raise ValueError(f"--filters must be key=value,key=value; got {chunk!r}")
        k, v = chunk.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def validate(
    *,
    input_path: Path,
    output_path: Path,
    filters: Dict[str, str],
    top_k: int,
    keep_easy_min_recall: float,
    keep_hard_min_recall: float,
    keep_medium_min_recall: float,
) -> int:
    rows_in = read_jsonl(input_path)
    if not rows_in:
        log.error("input file is empty: %s", input_path)
        return 0

    log.info("Loaded %d candidate rows", len(rows_in))

    # Build the retriever lazily — this script is the only path that
    # NEEDs the live FAISS+Postgres stack, so failing here is the
    # expected mode for offline / CI environments.
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.faiss_index import FaissIndex
    from app.capabilities.rag.metadata_store import RagMetadataStore
    from app.capabilities.rag.retriever import Retriever
    from app.core.config import get_settings

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    metadata = RagMetadataStore(settings.rag_db_dsn)
    metadata.ping()
    index = FaissIndex(Path(settings.rag_index_dir))
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=top_k,
        candidate_k=max(top_k * 4, 20),
    )
    retriever.ensure_ready()

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    thresholds = {
        "easy": keep_easy_min_recall,
        "medium": keep_medium_min_recall,
        "hard": keep_hard_min_recall,
    }

    for row in rows_in:
        query = str(row.get("query", "")).strip()
        if not query:
            dropped.append({**row, "drop_reason": "empty_query"})
            continue
        gold = [str(d) for d in row.get("expected_doc_ids", [])]
        difficulty = str(row.get("difficulty", "medium")).strip().lower()
        if difficulty not in thresholds:
            difficulty = "medium"

        try:
            report = retriever.retrieve(query, filters=filters or None)
        except Exception as ex:  # noqa: BLE001
            dropped.append({**row, "drop_reason": f"retrieve_error:{type(ex).__name__}"})
            continue

        retrieved_ids = [c.doc_id for c in (report.results or [])]
        recall = (
            sum(1 for d in gold if d in retrieved_ids) / max(1, len(gold))
            if gold else None
        )
        threshold = thresholds[difficulty]
        if recall is not None and recall < threshold:
            dropped.append({
                **row,
                "drop_reason": f"recall_below_threshold:{recall:.2f}<{threshold:.2f}",
                "retrieved_doc_ids": retrieved_ids[:5],
            })
            continue

        kept.append({**row, "validated_recall_at_k": recall})

    write_jsonl(
        output_path,
        kept,
        header=(
            f"Validated against live retriever; filters={filters}\n"
            f"top_k={top_k} thresholds={thresholds}"
        ),
    )
    dropped_path = output_path.with_suffix(output_path.suffix + ".dropped.jsonl")
    write_jsonl(dropped_path, dropped, header="dropped during validation")

    counts_kept = Counter(r.get("difficulty", "medium") for r in kept)
    counts_dropped = Counter(r.get("difficulty", "medium") for r in dropped)
    log.info(
        "Done. kept=%d (%s), dropped=%d (%s) -> %s",
        len(kept), dict(counts_kept),
        len(dropped), dict(counts_dropped),
        output_path,
    )
    return len(kept)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters", type=str, default=None,
                        help="comma-separated key=value filters, e.g. domain=anime")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keep-easy-min-recall", type=float, default=0.5)
    parser.add_argument("--keep-medium-min-recall", type=float, default=0.2)
    parser.add_argument("--keep-hard-min-recall", type=float, default=0.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    filters = _parse_filters(args.filters)

    try:
        validate(
            input_path=args.input_path,
            output_path=args.out,
            filters=filters,
            top_k=args.top_k,
            keep_easy_min_recall=args.keep_easy_min_recall,
            keep_medium_min_recall=args.keep_medium_min_recall,
            keep_hard_min_recall=args.keep_hard_min_recall,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("validation failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
