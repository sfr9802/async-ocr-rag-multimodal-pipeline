"""Dataset loading + report writing.

Intentionally minimal. Datasets are JSONL, reports are JSON + CSV.
No schemas beyond "the dict has a `query` or `file` field"; the
harness-specific modules do their own validation and conversion.

The writers make their parent directory so the CLI can be pointed at
`eval/reports/ocr-2026-04-15.json` without the user having to run a
separate mkdir step.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loader.
# ---------------------------------------------------------------------------


def load_jsonl(dataset_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Blank lines and lines starting with `#` are skipped so dataset
    authors can keep inline comments in the file. A malformed line
    raises ValueError with the 1-indexed line number so the author
    can fix it quickly.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

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
                    f"Invalid JSON on line {line_no} of {path}: {ex}"
                ) from ex
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} of {path} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            rows.append(obj)

    log.info("Loaded %d rows from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Report writers.
# ---------------------------------------------------------------------------


def write_json_report(
    report_path: Path,
    *,
    summary: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Emit a single JSON document with `metadata`, `summary`, `rows`.

    Using one flat file (rather than a directory of per-row files)
    keeps the whole run inspectable with `cat` / `jq` / a text editor
    and trivially easy to diff between runs.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    payload["summary"] = dict(summary)
    payload["rows"] = [dict(row) for row in rows]

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    log.info("Wrote JSON report: %s", path)
    return path


def write_csv_report(
    report_path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Optional[List[str]] = None,
) -> Path:
    """Flatten the per-row dicts into a CSV.

    If `columns` is omitted, the writer takes the union of keys across
    all rows and sorts them alphabetically so successive runs over the
    same dataset produce diffable files. List-valued fields are
    joined with `|` so they fit on one line.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(row) for row in rows]

    if columns is None:
        key_set = set()
        for row in rows_list:
            key_set.update(row.keys())
        columns = sorted(key_set)

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow({col: _csv_cell(row.get(col)) for col in columns})

    log.info("Wrote CSV report:  %s", path)
    return path


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Fall-through encoder for values that json.dumps doesn't natively
    understand — most commonly Path objects and datetime instances."""
    if isinstance(value, Path):
        return str(value)
    try:
        # dataclass instances — the harness exposes results as dataclasses
        import dataclasses

        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
    except Exception:  # pragma: no cover — defensive only
        pass
    return str(value)


def _csv_cell(value: Any) -> str:
    """Flatten a value to a CSV-safe string.

    Lists are joined with `|` (deliberately not `,` so there's no
    collision with the field separator). None becomes empty string.
    Everything else goes through `str()`.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
