"""Rebuild ``index.jsonl`` from every per-doc JSON already on disk.

``build_corpus.py`` writes the index row as a side effect of generating
a doc, which is the right default — the index stays in sync with what
actually landed on disk. But there are two moments when you want to
rebuild the index without calling the model:

  * After a schema change (e.g. adding ``domain='enterprise'`` to the
    row in Phase 9) — the existing docs on disk are fine, but their
    index rows are stale.
  * When you've hand-authored seed docs and haven't run ``build_corpus``
    yet, so ``index.jsonl`` doesn't exist at all.

Usage (from ``ai-worker/``)::

    python -m scripts.dataset.rebuild_corpus_index \\
        --out fixtures/corpus_kr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from scripts.dataset._common import configure_logging, write_jsonl
from scripts.dataset.build_corpus import _to_index_row

log = logging.getLogger("scripts.dataset.rebuild_corpus_index")


def rebuild(out_dir: Path) -> int:
    rows = []
    if not out_dir.exists():
        log.error("Corpus directory does not exist: %s", out_dir)
        return 0
    for json_path in sorted(out_dir.rglob("*.json")):
        if json_path.name in {"index.jsonl"}:
            continue
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as ex:  # noqa: BLE001
            log.warning("Skipping %s: %s", json_path, ex)
            continue
        if not isinstance(doc, dict) or "doc_id" not in doc or "sections" not in doc:
            continue
        seed = int(doc.get("seed") or 0)
        created_at = doc.get("created_at") or datetime.now(timezone.utc).isoformat()
        rows.append(_to_index_row(doc, seed=seed, created_at=created_at))

    out_path = out_dir / "index.jsonl"
    count = write_jsonl(out_path, rows, header=(
        f"Rebuilt from per-doc JSON by rebuild_corpus_index\n"
        f"doc_count={len(rows)}"
    ))
    log.info("Wrote %d rows to %s", count, out_path)
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        rebuild(args.out)
    except Exception as ex:  # noqa: BLE001
        log.exception("rebuild failed: %s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
