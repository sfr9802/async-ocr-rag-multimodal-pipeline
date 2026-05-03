"""Small Phase 7 v4 silver-regeneration dry run.

This wrapper only renders a tiny deterministic v4 silver sample and validates
that its expected doc ids exist in the canonical v4 pages and rag_chunks
namespace. It does not run retrieval, indexing, tuning, MMR, or answerability
scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from eval.harness.v4_silver_queries import (
    generate_v4_silver_queries,
    write_v4_silver_queries,
)


log = logging.getLogger("scripts.run_phase7_v4_silver_regen_dry_run")

FORBIDDEN_STRINGS = (
    "v3",
    "anime_namu_v3",
    "rag-cheap-sweep-v3",
    "bge-m3-anime-namu-v3",
)
DEFAULT_REPORT_DIR = Path("eval/reports/phase7/7.9_silver_regen_dry_run")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render and validate a tiny Phase 7 v4 silver sample."
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
    p.add_argument("--target-total", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _find_forbidden_strings(text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in FORBIDDEN_STRINGS if token in lowered]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def _load_active(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    hits = _find_forbidden_strings(text)
    if hits:
        raise SystemExit(
            f"active config contains forbidden legacy strings: {hits}"
        )
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SystemExit("active config must be a mapping")
    return payload


def _validate_active(payload: Mapping[str, Any]) -> dict[str, Path]:
    experiment_id = str(payload.get("experiment_id") or "")
    if not experiment_id.startswith("phase7-v4-"):
        raise SystemExit(f"unexpected experiment_id: {experiment_id!r}")

    meta = payload.get("_meta") or {}
    if not isinstance(meta, dict):
        raise SystemExit("_meta must be a mapping")
    if meta.get("fail_closed") is True:
        raise SystemExit("active config is fail-closed")
    if str(meta.get("status") or "").lower() in {
        "fail_closed",
        "fail-closed",
        "disabled",
    }:
        raise SystemExit("active config status blocks execution")

    policy = meta.get("execution_policy") or {}
    broad_flags = [name for name, value in policy.items() if bool(value)]
    if broad_flags:
        raise SystemExit(f"broad execution flags are enabled: {broad_flags}")

    artifacts = meta.get("canonical_v4_artifacts") or {}
    audit = meta.get("answerability_audit") or {}
    pages_v4 = _resolve(str(artifacts.get("pages_v4") or ""))
    rag_chunks = _resolve(str(audit.get("production_join_chunks") or ""))
    forbidden_join = _resolve(str(audit.get("forbidden_production_join_chunks") or ""))

    if pages_v4.name != "pages_v4.jsonl":
        raise SystemExit("canonical pages artifact must be pages_v4.jsonl")
    if rag_chunks.name != "rag_chunks.jsonl":
        raise SystemExit("production join source must be rag_chunks.jsonl")
    if forbidden_join.name != "chunks_v4.jsonl":
        raise SystemExit("forbidden join source must name chunks_v4.jsonl")
    if rag_chunks == forbidden_join:
        raise SystemExit("rag_chunks and forbidden chunks path must differ")
    if audit.get("chunks_v4_is_production_retrieval_join_fixture") is not False:
        raise SystemExit("chunks_v4 must be explicitly marked non-join fixture")

    if not pages_v4.exists():
        raise SystemExit(f"missing pages_v4 artifact: {pages_v4}")
    if not rag_chunks.exists():
        raise SystemExit(f"missing rag_chunks artifact: {rag_chunks}")

    return {"pages_v4": pages_v4, "rag_chunks": rag_chunks}


def _validate_limit(target_total: int) -> int:
    target_total = int(target_total)
    if target_total < 1 or target_total > 20:
        raise SystemExit("--target-total must be between 1 and 20 for dry-run")
    return target_total


def _validate_query_schema(queries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    required = {
        "id",
        "query",
        "language",
        "expected_doc_ids",
        "expected_section_keywords",
        "answer_type",
        "difficulty",
        "tags",
        "v4_meta",
    }
    for index, row in enumerate(queries, start=1):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"row {index} missing fields: {missing}")
        qid = str(row.get("id") or "")
        if not qid:
            errors.append(f"row {index} has empty id")
        elif qid in seen_ids:
            errors.append(f"duplicate id: {qid}")
        seen_ids.add(qid)
        if not str(row.get("query") or "").strip():
            errors.append(f"row {qid or index} has empty query")
        expected = row.get("expected_doc_ids")
        if not isinstance(expected, list) or len(expected) != 1 or not expected[0]:
            errors.append(f"row {qid or index} must have one expected_doc_id")
        meta = row.get("v4_meta")
        if not isinstance(meta, dict) or not str(meta.get("bucket") or ""):
            errors.append(f"row {qid or index} missing v4_meta.bucket")
        tags = row.get("tags")
        if not isinstance(tags, list) or "v4-silver" not in tags:
            errors.append(f"row {qid or index} missing v4-silver tag")
    return errors


def _expected_doc_ids(queries: Iterable[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in queries:
        for doc_id in row.get("expected_doc_ids") or []:
            if doc_id:
                out.add(str(doc_id))
    return out


def _scan_pages(
    pages_v4: Path, expected_doc_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not expected_doc_ids:
        return found
    with pages_v4.open("r", encoding="utf-8") as fp:
        for line in fp:
            if len(found) == len(expected_doc_ids):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_id = str(rec.get("page_id") or "")
            if page_id in expected_doc_ids:
                found[page_id] = {
                    "schema_version": rec.get("schema_version"),
                    "page_id": page_id,
                    "page_title": rec.get("page_title"),
                    "retrieval_title": rec.get("retrieval_title"),
                    "relation": rec.get("relation"),
                    "page_type": rec.get("page_type"),
                }
    return found


def _scan_rag_chunks(
    rag_chunks: Path, expected_doc_ids: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        doc_id: {"chunk_count": 0, "sample_chunk_ids": []}
        for doc_id in expected_doc_ids
    }
    remaining = set(expected_doc_ids)
    with rag_chunks.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = str(rec.get("doc_id") or "")
            if doc_id not in expected_doc_ids:
                continue
            info = out[doc_id]
            info["chunk_count"] += 1
            if len(info["sample_chunk_ids"]) < 3:
                info["sample_chunk_ids"].append(str(rec.get("chunk_id") or ""))
            if info["chunk_count"] > 0 and len(info["sample_chunk_ids"]) >= 1:
                remaining.discard(doc_id)
            # Keep scanning to count all chunks for selected docs. This is a
            # streaming metadata validation pass, not retrieval or scoring.
    return out


def _write_contamination_scan(
    paths: Iterable[Path], out_path: Path,
) -> dict[str, Any]:
    results = []
    contaminated = False
    for path in paths:
        text = _read_text(path)
        hits = _find_forbidden_strings(text)
        contaminated = contaminated or bool(hits)
        results.append({"path": str(path), "forbidden_hits": hits})
    payload = {
        "forbidden_strings": list(FORBIDDEN_STRINGS),
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
    target_total = _validate_limit(args.target_total)
    active = _load_active(args.active_yaml)
    paths = _validate_active(active)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_path = report_dir / "queries_v4_silver_regen_sample.jsonl"
    manifest_path = report_dir / "manifest.json"
    contamination_path = report_dir / "legacy_contamination_scan.json"

    queries = generate_v4_silver_queries(
        paths["pages_v4"],
        target_total=target_total,
        seed=int(args.seed),
    )
    write_v4_silver_queries(queries, sample_path)

    schema_errors = _validate_query_schema(queries)
    expected_doc_ids = _expected_doc_ids(queries)
    pages_found = _scan_pages(paths["pages_v4"], expected_doc_ids)
    chunk_info = _scan_rag_chunks(paths["rag_chunks"], expected_doc_ids)

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
    query_id_to_doc_id = {
        str(row.get("id")): str((row.get("expected_doc_ids") or [""])[0])
        for row in queries
    }
    doc_to_query_ids: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in query_id_to_doc_id.items():
        doc_to_query_ids[doc_id].append(qid)

    manifest = {
        "experiment_id": active["experiment_id"],
        "dry_run_only": True,
        "generation_mode": "deterministic_v4_pages",
        "target_total": target_total,
        "actual_query_count": len(queries),
        "seed": int(args.seed),
        "pages_v4_path": str(paths["pages_v4"]),
        "production_join_source": str(paths["rag_chunks"]),
        "join_source_kind": "rag_chunks",
        "chunks_v4_used_as_join_source": False,
        "artifacts": {
            "sample_jsonl": str(sample_path),
            "manifest_json": str(manifest_path),
            "contamination_scan_json": str(contamination_path),
        },
        "schema_validation": {
            "valid": not schema_errors,
            "errors": schema_errors,
            "bucket_counts": dict(sorted(bucket_counts.items())),
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
            "doc_id_to_query_ids": dict(sorted(doc_to_query_ids.items())),
            "doc_id_chunk_probe": dict(sorted(chunk_info.items())),
            "page_probe": dict(sorted(pages_found.items())),
        },
        "blocked_operations": {
            "full_retrieval_eval": True,
            "full_indexing": True,
            "optuna_tuning": True,
            "mmr_sweep": True,
            "answerability_scoring": True,
            "silver_full_regeneration": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    contamination = _write_contamination_scan(
        [sample_path, manifest_path], contamination_path,
    )
    if contamination["contaminated"]:
        raise SystemExit("dry-run artifacts contain forbidden legacy strings")
    if schema_errors or missing_pages or missing_rag_doc_ids or empty_chunk_ids:
        raise SystemExit("silver dry-run validation failed; see manifest.json")

    log.info("wrote silver sample: %s", sample_path)
    log.info("wrote validation manifest: %s", manifest_path)
    log.info("wrote contamination scan: %s", contamination_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
