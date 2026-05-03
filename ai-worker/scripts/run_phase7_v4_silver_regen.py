"""Full Phase 7 v4 silver candidate-pool regeneration.

Generates a deterministic v4 silver JSONL from pages_v4.jsonl and validates
the expected doc ids against canonical pages_v4.jsonl plus rag_chunks.jsonl.
This script does not run retrieval, indexing, tuning, MMR, answerability
scoring, or human-gold sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from eval.harness.v4_silver_queries import (
    generate_v4_silver_queries,
    write_v4_silver_queries,
)
from scripts.run_phase7_v4_silver_regen_dry_run import (
    _expected_doc_ids,
    _find_forbidden_strings,
    _load_active,
    _read_text,
    _resolve,
    _scan_pages,
    _scan_rag_chunks,
    _validate_active,
    _validate_query_schema,
)


log = logging.getLogger("scripts.run_phase7_v4_silver_regen")

DEFAULT_REPORT_DIR = Path("eval/reports/phase7/7.10_silver_regen_full")
DEFAULT_TARGET_TOTAL = 500
GENERATOR_NAME = "generate_v4_silver_queries"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate the full Phase 7 v4 silver candidate pool."
    )
    p.add_argument(
        "--active-yaml",
        type=Path,
        default=Path("eval/experiments/active.yaml"),
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    p.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow --target-total > 1000.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _validate_target_total(target_total: int, *, allow_large: bool) -> int:
    target_total = int(target_total)
    if target_total < 1:
        raise SystemExit("--target-total must be positive")
    if target_total > 1000 and not allow_large:
        raise SystemExit("--target-total > 1000 requires --allow-large")
    return target_total


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_artifact_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _active_artifact_paths(active: Mapping[str, Any]) -> dict[str, Path]:
    meta = active.get("_meta") or {}
    artifacts = meta.get("canonical_v4_artifacts") or {}
    audit = meta.get("answerability_audit") or {}
    return {
        "pages_v4": _resolve(str(artifacts.get("pages_v4") or "")),
        "rag_chunks": _resolve(str(audit.get("production_join_chunks") or "")),
        "split_manifest": _resolve(str(artifacts.get("split_manifest") or "")),
        "validation_report": _resolve(str(artifacts.get("validation_report") or "")),
    }


def _query_id_to_doc_id(
    queries: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in queries:
        qid = str(row.get("id") or "")
        expected = row.get("expected_doc_ids") or [""]
        out[qid] = str(expected[0])
    return out


def _doc_to_query_ids(query_id_to_doc_id: Mapping[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in query_id_to_doc_id.items():
        out[doc_id].append(qid)
    return dict(sorted(out.items()))


def _write_bucket_summary(
    *,
    queries: list[dict[str, Any]],
    bucket_counts: Mapping[str, int],
    out_json: Path,
    out_md: Path,
) -> None:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        bucket = str((row.get("v4_meta") or {}).get("bucket") or "")
        by_bucket[bucket].append(row)

    payload = {
        "total": len(queries),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_examples": {
            bucket: [
                {
                    "id": row["id"],
                    "query": row["query"],
                    "expected_doc_id": row["expected_doc_ids"][0],
                    "page_title": (row.get("v4_meta") or {}).get("page_title"),
                    "retrieval_title": (
                        (row.get("v4_meta") or {}).get("retrieval_title")
                    ),
                }
                for row in rows[:5]
            ]
            for bucket, rows in sorted(by_bucket.items())
        },
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Phase 7 v4 Silver Bucket Summary", ""]
    lines.append(f"- total: {len(queries)}")
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- {bucket}: {count}")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sampling_preview(
    *,
    queries: list[dict[str, Any]],
    pages_found: Mapping[str, Mapping[str, Any]],
    out_path: Path,
    per_bucket: int = 5,
) -> None:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        bucket = str((row.get("v4_meta") or {}).get("bucket") or "")
        by_bucket[bucket].append(row)

    lines = [
        "# Phase 7 v4 Silver Sampling Preview",
        "",
        "Human-labeling preview only. This file intentionally shows a small "
        "sample per bucket, not the full silver pool.",
        "",
    ]
    for bucket, rows in sorted(by_bucket.items()):
        lines.append(f"## {bucket}")
        lines.append("")
        for row in rows[:per_bucket]:
            expected_doc_id = str((row.get("expected_doc_ids") or [""])[0])
            page = pages_found.get(expected_doc_id) or {}
            keywords = ", ".join(str(x) for x in row.get("expected_section_keywords") or [])
            lines.extend(
                [
                    f"- id: `{row['id']}`",
                    f"  query: {row['query']}",
                    f"  expected_doc_id: `{expected_doc_id}`",
                    f"  expected_page_title: {page.get('page_title', '')}",
                    f"  retrieval_title: {page.get('retrieval_title', '')}",
                    f"  expected_section_keywords: {keywords}",
                    "",
                ]
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_contamination_scan(
    paths: Iterable[Path], out_path: Path,
) -> dict[str, Any]:
    results = []
    contaminated = False
    for path in paths:
        hits = _find_forbidden_strings(_read_text(path))
        contaminated = contaminated or bool(hits)
        results.append({"path": str(path), "forbidden_hits": hits})
    payload = {
        "note": (
            "The scan target is generated JSONL/manifest content. This scan "
            "file may contain the literal inspection tokens by design."
        ),
        "contaminated": contaminated,
        "results": results,
    }
    out_path.write_text(
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
    target_total = _validate_target_total(
        args.target_total, allow_large=bool(args.allow_large),
    )

    active = _load_active(args.active_yaml)
    validated_paths = _validate_active(active)
    artifact_paths = _active_artifact_paths(active)
    for name, path in artifact_paths.items():
        if not path.exists():
            raise SystemExit(f"missing source artifact {name}: {path}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    silver_path = report_dir / f"queries_v4_silver_{target_total}.jsonl"
    manifest_path = report_dir / "manifest.json"
    contamination_path = report_dir / "legacy_contamination_scan.json"
    bucket_summary_json = report_dir / "bucket_summary.json"
    bucket_summary_md = report_dir / "bucket_summary.md"
    preview_path = report_dir / "sampling_preview.md"

    queries = generate_v4_silver_queries(
        validated_paths["pages_v4"],
        target_total=target_total,
        seed=int(args.seed),
    )
    write_v4_silver_queries(queries, silver_path)

    schema_errors = _validate_query_schema(queries)
    if len(queries) != target_total:
        schema_errors.append(
            f"actual_total={len(queries)} did not match target_total={target_total}"
        )
    expected_doc_ids = _expected_doc_ids(queries)
    pages_found = _scan_pages(validated_paths["pages_v4"], expected_doc_ids)
    chunk_info = _scan_rag_chunks(validated_paths["rag_chunks"], expected_doc_ids)
    missing_pages = sorted(expected_doc_ids - set(pages_found))
    missing_rag_doc_ids = sorted(
        doc_id
        for doc_id, info in chunk_info.items()
        if int(info["chunk_count"]) <= 0
    )
    empty_chunk_ids = sorted(
        doc_id
        for doc_id, info in chunk_info.items()
        if not any(info["sample_chunk_ids"])
    )
    bucket_counts = Counter(
        str((row.get("v4_meta") or {}).get("bucket") or "") for row in queries
    )
    query_id_to_doc_id = _query_id_to_doc_id(queries)
    duplicate_query_ids = sorted(
        qid for qid, count in Counter(row.get("id") for row in queries).items()
        if count > 1
    )
    empty_query_ids = sorted(
        str(row.get("id") or "")
        for row in queries
        if not str(row.get("query") or "").strip()
    )
    empty_expected_doc_query_ids = sorted(
        str(row.get("id") or "")
        for row in queries
        if not (row.get("expected_doc_ids") or [""])[0]
    )
    empty_bucket_query_ids = sorted(
        str(row.get("id") or "")
        for row in queries
        if not str((row.get("v4_meta") or {}).get("bucket") or "")
    )

    _write_bucket_summary(
        queries=queries,
        bucket_counts=bucket_counts,
        out_json=bucket_summary_json,
        out_md=bucket_summary_md,
    )
    _write_sampling_preview(
        queries=queries,
        pages_found=pages_found,
        out_path=preview_path,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "experiment_id": active["experiment_id"],
        "generated_at": generated_at,
        "generator_name": GENERATOR_NAME,
        "target_total": target_total,
        "actual_total": len(queries),
        "seed": int(args.seed),
        "source_artifacts": {
            name: _source_artifact_info(path)
            for name, path in artifact_paths.items()
        },
        "production_join_source": str(validated_paths["rag_chunks"]),
        "join_source_kind": "rag_chunks",
        "chunks_v4_used_as_production_join_source": False,
        "artifacts": {
            "silver_jsonl": str(silver_path),
            "manifest_json": str(manifest_path),
            "legacy_contamination_scan_json": str(contamination_path),
            "bucket_summary_json": str(bucket_summary_json),
            "bucket_summary_md": str(bucket_summary_md),
            "sampling_preview_md": str(preview_path),
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "schema_validation": {
            "valid": not schema_errors,
            "errors": schema_errors,
            "duplicate_query_ids": duplicate_query_ids,
            "empty_query_ids": empty_query_ids,
            "empty_expected_doc_query_ids": empty_expected_doc_query_ids,
            "empty_bucket_query_ids": empty_bucket_query_ids,
        },
        "namespace_validation": {
            "expected_doc_id_count": len(expected_doc_ids),
            "missing_pages_v4_doc_ids": missing_pages,
            "missing_rag_chunks_doc_ids": missing_rag_doc_ids,
            "empty_sample_chunk_ids": empty_chunk_ids,
            "all_expected_doc_ids_join_pages_v4": not missing_pages,
            "all_expected_doc_ids_have_rag_chunks": (
                not missing_rag_doc_ids and not empty_chunk_ids
            ),
            "doc_id_to_query_ids": _doc_to_query_ids(query_id_to_doc_id),
            "doc_id_chunk_probe": dict(sorted(chunk_info.items())),
            "page_probe": dict(sorted(pages_found.items())),
        },
        "blocked_operations": {
            "retrieval_eval": True,
            "full_indexing": True,
            "optuna_tuning": True,
            "mmr_sweep": True,
            "answerability_scoring": True,
            "human_gold_sampling": True,
        },
        "legacy_contamination": {
            "contaminated": None,
            "scanned_paths": [str(silver_path), str(manifest_path)],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contamination = _write_contamination_scan(
        [silver_path, manifest_path], contamination_path,
    )
    manifest["legacy_contamination"] = {
        "contaminated": bool(contamination["contaminated"]),
        "scanned_paths": [str(silver_path), str(manifest_path)],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contamination = _write_contamination_scan(
        [silver_path, manifest_path], contamination_path,
    )

    if contamination["contaminated"]:
        raise SystemExit("generated silver artifacts contain legacy strings")
    if (
        schema_errors
        or duplicate_query_ids
        or empty_query_ids
        or empty_expected_doc_query_ids
        or empty_bucket_query_ids
        or missing_pages
        or missing_rag_doc_ids
        or empty_chunk_ids
    ):
        raise SystemExit("silver regeneration validation failed; see manifest")

    log.info("wrote silver JSONL: %s", silver_path)
    log.info("wrote manifest: %s", manifest_path)
    log.info("wrote contamination scan: %s", contamination_path)
    log.info("wrote bucket summaries: %s / %s", bucket_summary_json, bucket_summary_md)
    log.info("wrote sampling preview: %s", preview_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
