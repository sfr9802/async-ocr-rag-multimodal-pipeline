"""Extract balanced Phase 7 v4 human-gold seed-50 candidates.

Consumes the regenerated v4 silver pool and writes a labeling CSV plus
validation artifacts. This script does not run retrieval, indexing, tuning,
MMR, answerability scoring, or human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.run_phase7_v4_silver_regen_dry_run import (
    _find_forbidden_strings,
    _read_text,
)


log = logging.getLogger("scripts.run_phase7_v4_human_gold_seed50")

DEFAULT_INPUT = Path(
    "eval/reports/phase7/7.10_silver_regen_full/queries_v4_silver_500.jsonl"
)
DEFAULT_REPORT_DIR = Path("eval/reports/phase7/7.11_human_gold_seed_50")
DEFAULT_BUCKET_TARGETS = {
    "main_work": 15,
    "subpage_generic": 22,
    "subpage_named": 13,
}
CSV_FIELDS = [
    "query_id",
    "bucket",
    "query",
    "expected_doc_id",
    "expected_page_title",
    "expected_section_path",
    "answerability_label",
    "gold_doc_match",
    "gold_chunk_match",
    "labeler_note",
    "label_status",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract balanced human-gold seed-50 labeling candidates."
    )
    p.add_argument("--silver-path", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--pages-v4", type=Path, default=Path(
        "eval/corpora/namu-v4-structured-combined/pages_v4.jsonl"
    ))
    p.add_argument("--rag-chunks", type=Path, default=Path(
        "eval/corpora/namu-v4-structured-combined/rag_chunks.jsonl"
    ))
    p.add_argument(
        "--bucket-targets",
        default="main_work=15,subpage_generic=22,subpage_named=13",
        help="Comma-separated bucket=count targets.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _parse_bucket_targets(raw: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid bucket target {item!r}; expected bucket=count")
        name, value = item.split("=", 1)
        name = name.strip()
        try:
            count = int(value)
        except ValueError as ex:
            raise SystemExit(f"invalid count in bucket target {item!r}") from ex
        if count < 0:
            raise SystemExit(f"bucket target must be non-negative: {item!r}")
        targets[name] = count
    if sum(targets.values()) != 50:
        raise SystemExit(f"bucket targets must sum to 50, got {sum(targets.values())}")
    return targets


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _expected_doc_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        expected = row.get("expected_doc_ids") or []
        if expected:
            out.add(str(expected[0]))
    return out


def _scan_pages(
    pages_v4: Path, expected_doc_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with Path(pages_v4).open("r", encoding="utf-8") as fp:
        for line in fp:
            if len(found) == len(expected_doc_ids):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_id = str(rec.get("page_id") or "")
            if page_id not in expected_doc_ids:
                continue
            section_path = ""
            sections = rec.get("sections") or []
            if isinstance(sections, list) and sections:
                first = sections[0]
                if isinstance(first, dict):
                    heading_path = first.get("heading_path") or []
                    if isinstance(heading_path, list):
                        section_path = " > ".join(
                            str(x) for x in heading_path if str(x)
                        )
                    elif heading_path:
                        section_path = str(heading_path)
            found[page_id] = {
                "page_id": page_id,
                "schema_version": rec.get("schema_version"),
                "page_title": rec.get("page_title") or "",
                "retrieval_title": rec.get("retrieval_title") or "",
                "expected_section_path": section_path,
                "relation": rec.get("relation") or "",
                "page_type": rec.get("page_type") or "",
            }
    return found


def _scan_rag_chunks(
    rag_chunks: Path, expected_doc_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {
        doc_id: {"chunk_count": 0, "sample_chunk_ids": []}
        for doc_id in expected_doc_ids
    }
    with Path(rag_chunks).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = str(rec.get("doc_id") or "")
            if doc_id not in expected_doc_ids:
                continue
            info = found[doc_id]
            info["chunk_count"] += 1
            if len(info["sample_chunk_ids"]) < 3:
                info["sample_chunk_ids"].append(str(rec.get("chunk_id") or ""))
    return found


def _group_by_bucket(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = str((row.get("v4_meta") or {}).get("bucket") or "")
        grouped[bucket].append(dict(row))
    return grouped


def _spread_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise SystemExit(
            f"bucket has only {len(rows)} rows, cannot select {count}"
        )
    if count == 1:
        return [rows[len(rows) // 2]]
    # Deterministic spread over the existing generated order. This keeps the
    # sample balanced without taking the first contiguous block from each bucket.
    indexes = [
        round(i * (len(rows) - 1) / (count - 1))
        for i in range(count)
    ]
    selected = [rows[i] for i in indexes]
    if len({row["id"] for row in selected}) != count:
        # Extremely defensive: round() can collide for odd small inputs.
        selected = []
        used: set[int] = set()
        for i in indexes:
            j = i
            while j in used and j + 1 < len(rows):
                j += 1
            while j in used and j > 0:
                j -= 1
            used.add(j)
            selected.append(rows[j])
    return selected


def _select_candidates(
    rows: list[dict[str, Any]], targets: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped = _group_by_bucket(rows)
    selected: list[dict[str, Any]] = []
    for bucket, count in targets.items():
        selected.extend(_spread_select(grouped.get(bucket, []), int(count)))
    return sorted(selected, key=lambda row: str(row.get("id") or ""))


def _to_csv_row(
    row: Mapping[str, Any], page_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    expected_doc_id = str((row.get("expected_doc_ids") or [""])[0])
    page = page_lookup.get(expected_doc_id) or {}
    return {
        "query_id": str(row.get("id") or ""),
        "bucket": str((row.get("v4_meta") or {}).get("bucket") or ""),
        "query": str(row.get("query") or ""),
        "expected_doc_id": expected_doc_id,
        "expected_page_title": str(
            page.get("page_title")
            or (row.get("v4_meta") or {}).get("page_title")
            or ""
        ),
        "expected_section_path": str(page.get("expected_section_path") or ""),
        "answerability_label": "",
        "gold_doc_match": "",
        "gold_chunk_match": "",
        "labeler_note": "",
        "label_status": "pending",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_bucket_summary(
    path: Path,
    *,
    full_counts: Mapping[str, int],
    selected_counts: Mapping[str, int],
    targets: Mapping[str, int],
) -> None:
    payload = {
        "target_total": 50,
        "targets": dict(targets),
        "selected_counts": dict(sorted(selected_counts.items())),
        "full_silver_bucket_counts": dict(sorted(full_counts.items())),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_sampling_preview(
    path: Path,
    csv_rows: list[dict[str, str]],
    *,
    per_bucket: int = 5,
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows:
        grouped[row["bucket"]].append(row)
    lines = [
        "# Phase 7 v4 Human-Gold Seed 50 Preview",
        "",
        "Preview only. Labels are intentionally blank in the CSV.",
        "",
    ]
    for bucket, rows in sorted(grouped.items()):
        lines.append(f"## {bucket}")
        lines.append("")
        for row in rows[:per_bucket]:
            lines.extend([
                f"- query_id: `{row['query_id']}`",
                f"  query: {row['query']}",
                f"  expected_doc_id: `{row['expected_doc_id']}`",
                f"  expected_page_title: {row['expected_page_title']}",
                f"  expected_section_path: {row['expected_section_path']}",
                "",
            ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _validate_csv_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    ids = [row["query_id"] for row in rows]
    bucket_values = [row["bucket"] for row in rows]
    return {
        "valid": (
            len(rows) == 50
            and len(set(ids)) == len(ids)
            and all(row["query"].strip() for row in rows)
            and all(row["expected_doc_id"].strip() for row in rows)
            and all(row["expected_page_title"].strip() for row in rows)
            and all(bucket_values)
            and all(row["label_status"] == "pending" for row in rows)
        ),
        "duplicate_query_ids": sorted(
            query_id for query_id, count in Counter(ids).items() if count > 1
        ),
        "empty_query_ids": [
            row["query_id"] for row in rows if not row["query"].strip()
        ],
        "empty_expected_doc_query_ids": [
            row["query_id"] for row in rows if not row["expected_doc_id"].strip()
        ],
        "empty_page_title_query_ids": [
            row["query_id"] for row in rows if not row["expected_page_title"].strip()
        ],
        "empty_bucket_query_ids": [
            row["query_id"] for row in rows if not row["bucket"].strip()
        ],
    }


def _write_contamination_scan(paths: Iterable[Path], out_path: Path) -> dict[str, Any]:
    results = []
    contaminated = False
    for path in paths:
        hits = _find_forbidden_strings(_read_text(path))
        contaminated = contaminated or bool(hits)
        results.append({"path": str(path), "forbidden_hits": hits})
    payload = {
        "note": (
            "Scan targets are generated CSV/manifest content. This scan file "
            "may contain inspection tokens by design."
        ),
        "contaminated": contaminated,
        "results": results,
    }
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    silver_path = _resolve(args.silver_path)
    pages_v4 = _resolve(args.pages_v4)
    rag_chunks = _resolve(args.rag_chunks)
    if not silver_path.exists():
        raise SystemExit(f"missing silver input: {silver_path}")
    if not pages_v4.exists():
        raise SystemExit(f"missing pages_v4: {pages_v4}")
    if rag_chunks.name != "rag_chunks.jsonl":
        raise SystemExit("production join source must be rag_chunks.jsonl")
    if not rag_chunks.exists():
        raise SystemExit(f"missing rag_chunks: {rag_chunks}")

    targets = _parse_bucket_targets(args.bucket_targets)
    rows = _load_jsonl(silver_path)
    full_counts = Counter(
        str((row.get("v4_meta") or {}).get("bucket") or "") for row in rows
    )
    selected = _select_candidates(rows, targets)
    expected_doc_ids = _expected_doc_ids(selected)
    page_lookup = _scan_pages(pages_v4, expected_doc_ids)
    chunk_lookup = _scan_rag_chunks(rag_chunks, expected_doc_ids)
    csv_rows = [_to_csv_row(row, page_lookup) for row in selected]
    selected_counts = Counter(row["bucket"] for row in csv_rows)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "phase7_human_gold_seed_50_candidates.csv"
    manifest_path = report_dir / "manifest.json"
    bucket_summary_path = report_dir / "bucket_summary.json"
    preview_path = report_dir / "sampling_preview.md"
    contamination_path = report_dir / "legacy_contamination_scan.json"

    _write_csv(csv_path, csv_rows)
    _write_bucket_summary(
        bucket_summary_path,
        full_counts=full_counts,
        selected_counts=selected_counts,
        targets=targets,
    )
    _write_sampling_preview(preview_path, csv_rows)

    missing_pages = sorted(expected_doc_ids - set(page_lookup))
    missing_rag_doc_ids = sorted(
        doc_id
        for doc_id, info in chunk_lookup.items()
        if int(info["chunk_count"]) <= 0
    )
    csv_validation = _validate_csv_rows(csv_rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_silver": _artifact_info(silver_path),
        "source_artifacts": {
            "pages_v4": _artifact_info(pages_v4),
            "rag_chunks": _artifact_info(rag_chunks),
        },
        "selection_strategy": "deterministic_bucket_balanced_spread",
        "bucket_targets": targets,
        "full_silver_count": len(rows),
        "full_silver_bucket_counts": dict(sorted(full_counts.items())),
        "selected_total": len(csv_rows),
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
        "csv_fields": CSV_FIELDS,
        "artifacts": {
            "csv": str(csv_path),
            "manifest_json": str(manifest_path),
            "bucket_summary_json": str(bucket_summary_path),
            "sampling_preview_md": str(preview_path),
            "legacy_contamination_scan_json": str(contamination_path),
        },
        "schema_validation": csv_validation,
        "namespace_validation": {
            "expected_doc_id_count": len(expected_doc_ids),
            "missing_pages_v4_doc_ids": missing_pages,
            "missing_rag_chunks_doc_ids": missing_rag_doc_ids,
            "all_expected_doc_ids_join_pages_v4": not missing_pages,
            "all_expected_doc_ids_have_rag_chunks": not missing_rag_doc_ids,
            "chunks_v4_used_as_production_join_source": False,
            "doc_id_chunk_probe": dict(sorted(chunk_lookup.items())),
            "page_probe": dict(sorted(page_lookup.items())),
        },
        "blocked_operations": {
            "retrieval_eval": True,
            "indexing": True,
            "optuna_tuning": True,
            "mmr_sweep": True,
            "answerability_scoring": True,
            "human_labels_written": False,
        },
        "legacy_contamination": {
            "contaminated": None,
            "scanned_paths": [str(csv_path), str(manifest_path)],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contamination = _write_contamination_scan(
        [csv_path, manifest_path], contamination_path,
    )
    manifest["legacy_contamination"] = {
        "contaminated": bool(contamination["contaminated"]),
        "scanned_paths": [str(csv_path), str(manifest_path)],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contamination = _write_contamination_scan(
        [csv_path, manifest_path], contamination_path,
    )

    if contamination["contaminated"]:
        raise SystemExit("candidate artifacts contain legacy strings")
    if (
        not csv_validation["valid"]
        or selected_counts != Counter(targets)
        or missing_pages
        or missing_rag_doc_ids
    ):
        raise SystemExit("candidate validation failed; see manifest")

    log.info("wrote candidate CSV: %s", csv_path)
    log.info("wrote manifest: %s", manifest_path)
    log.info("wrote bucket summary: %s", bucket_summary_path)
    log.info("wrote sampling preview: %s", preview_path)
    log.info("wrote contamination scan: %s", contamination_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
