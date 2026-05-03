"""Manifest-driven smoke runner for real PDF RAG ingestion samples.

The runner uploads each PDF from a manifest, starts the pdf-extract job,
polls completion, optionally validates PostgreSQL rows, and writes a JSON
report. It exits non-zero when any sample fails or required v2 metadata is
missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_pdf_ingestion_smoke import (  # noqa: E402
    start_pdf_extract,
    upload_pdf,
    wait_job,
)


DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MANIFEST = Path("samples/rag_pdf_ingestion_manifest.json")
DEFAULT_REPORT = Path("reports/rag_pdf_ingestion_sample_batch_report.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_CONTAINER = "aipipeline-postgres"
DEFAULT_DB_USER = "aipipeline"
DEFAULT_DB_NAME = "aipipeline"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    samples = manifest.get("samples") or []
    defaults = manifest.get("default_expectations") or {}

    report: dict[str, Any] = {
        "run_id": utc_run_id(),
        "manifest": str(manifest_path),
        "total_samples": len(samples),
        "passed": 0,
        "failed": 0,
        "samples": [],
    }

    client = httpx.Client(base_url=args.base_url, timeout=args.http_timeout)
    for sample in samples:
        sample_report = run_sample(
            client=client,
            sample=sample,
            defaults=defaults,
            manifest_path=manifest_path,
            args=args,
        )
        report["samples"].append(sample_report)
        if sample_report["status"] == "PASSED":
            report["passed"] += 1
        else:
            report["failed"] += 1

    add_aggregate_metrics(report)
    write_json_report(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--extract-path", default="pdf-extract")
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--poll-timeout", type=float, default=240.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--db-container", default=DEFAULT_DB_CONTAINER)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--skip-db-check", action="store_true")
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("samples"), list):
        raise ValueError("Manifest must contain a samples array")
    return manifest


def run_sample(
    *,
    client: httpx.Client,
    sample: dict[str, Any],
    defaults: dict[str, Any],
    manifest_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "unknown")
    warnings: list[str] = []
    errors: list[str] = []
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "bucket": sample.get("bucket"),
        "status": "FAILED",
        "source_file_id": None,
        "job_id": None,
        "search_unit_count": 0,
        "missing_required_metadata_count": None,
        "warnings": warnings,
        "errors": errors,
    }
    sample_started = time.perf_counter()

    try:
        if str(sample.get("file_type") or "").lower() != "pdf":
            raise ValueError("Only pdf samples are supported by this runner")
        path = resolve_sample_path(str(sample.get("file_path") or sample.get("path") or ""), manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")

        source = upload_pdf(client, path)
        source_id = str(source["sourceFileId"])
        result["source_file_id"] = source_id
        parsing_started = time.perf_counter()
        job = start_pdf_extract(client, source_id, args.extract_path)
        job_id = str(job["jobId"])
        result["job_id"] = job_id
        final = wait_job(
            client,
            job_id,
            timeout_seconds=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
        result["parsing_latency_seconds"] = round(time.perf_counter() - parsing_started, 3)
        result["job_status"] = final.get("status")
        if final.get("status") != "SUCCEEDED":
            errors.append("pdf-extract job did not succeed")
            result["job"] = final
            return result

        if args.skip_db_check:
            warnings.append("DB validation skipped")
            result["indexing_latency_seconds"] = 0.0
            result["status"] = "PASSED"
            return result

        indexing_started = time.perf_counter()
        db_report = query_sample_db_report(
            container=args.db_container,
            user=args.db_user,
            database=args.db_name,
            source_file_id=source_id,
        )
        result["indexing_latency_seconds"] = round(time.perf_counter() - indexing_started, 3)
        result.update(db_report)
        validate_sample_db_report(result, sample=sample, defaults=defaults)
        result["status"] = "PASSED"
        return result
    except Exception as exc:
        errors.append(str(exc))
        return result
    finally:
        result["total_latency_seconds"] = round(time.perf_counter() - sample_started, 3)


def resolve_sample_path(raw_path: str, manifest_path: Path) -> Path:
    if not raw_path:
        raise ValueError("sample file_path is required")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    manifest_relative = (manifest_path.parent / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return REPO_ROOT.joinpath(path).resolve()


def query_sample_db_report(
    *,
    container: str,
    user: str,
    database: str,
    source_file_id: str,
) -> dict[str, Any]:
    sql = f"""
    select jsonb_build_object(
      'source_status', (
        select status from source_file where id = {sql_literal(source_file_id)}
      ),
      'pdf_parsed_artifact_count', (
        select count(*) from extracted_artifact
        where source_file_id = {sql_literal(source_file_id)}
          and artifact_type = 'PDF_PARSED_JSON'
      ),
      'pdf_plaintext_artifact_count', (
        select count(*) from extracted_artifact
        where source_file_id = {sql_literal(source_file_id)}
          and artifact_type = 'PDF_PLAINTEXT'
      ),
      'parsed_artifact_count', (
        select count(*) from parsed_artifact
        where source_file_id = {sql_literal(source_file_id)}
          and artifact_type = 'PDF_PARSED_JSON'
      ),
      'parsed_artifact_parser_names', (
        select coalesce(jsonb_agg(parser_name order by parser_name), '[]'::jsonb)
        from (
          select distinct parser_name from parsed_artifact
          where source_file_id = {sql_literal(source_file_id)}
            and artifact_type = 'PDF_PARSED_JSON'
        ) names
      ),
      'parsed_artifact_parser_versions', (
        select coalesce(jsonb_agg(parser_version order by parser_version), '[]'::jsonb)
        from (
          select distinct parser_version from parsed_artifact
          where source_file_id = {sql_literal(source_file_id)}
            and artifact_type = 'PDF_PARSED_JSON'
        ) versions
      ),
      'search_unit_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
      ),
      'page_count_from_location', (
        select count(distinct location_json->>'physical_page_index') from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and location_json ? 'physical_page_index'
      ),
      'parsed_pdf_page_count', (
        select coalesce(sum(jsonb_array_length(coalesce(artifact_json->'pages', '[]'::jsonb))), 0)
        from parsed_artifact
        where source_file_id = {sql_literal(source_file_id)}
          and artifact_type = 'PDF_PARSED_JSON'
      ),
      'pdf_page_metadata_count', (
        select count(*) from pdf_page_metadata
        where source_file_id = {sql_literal(source_file_id)}
      ),
      'missing_page_metadata_count', (
        with pages as (
          select pa.document_version_id,
                 (page.value->>'physical_page_index')::int as physical_page_index
          from parsed_artifact pa
          cross join lateral jsonb_array_elements(coalesce(pa.artifact_json->'pages', '[]'::jsonb)) page(value)
          where pa.source_file_id = {sql_literal(source_file_id)}
            and pa.artifact_type = 'PDF_PARSED_JSON'
            and page.value ? 'physical_page_index'
        )
        select count(*) from pages p
        left join pdf_page_metadata ppm
          on ppm.document_version_id = p.document_version_id
         and ppm.physical_page_index = p.physical_page_index
        where ppm.id is null
      ),
      'inconsistent_location_page_metadata_count', (
        select count(*) from search_unit su
        left join pdf_page_metadata ppm
          on ppm.document_version_id = su.document_version_id
         and ppm.physical_page_index = (su.location_json->>'physical_page_index')::int
        where su.source_file_id = {sql_literal(source_file_id)}
          and su.source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and su.location_json ? 'physical_page_index'
          and (
            ppm.id is null
            or ppm.page_no <> (su.location_json->>'page_no')::int
            or coalesce(ppm.page_label, '') <> coalesce(su.location_json->>'page_label', '')
          )
      ),
      'missing_required_metadata_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and (parser_version is null or location_json is null or citation_text is null)
      ),
      'invalid_pdf_location_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and (
            location_json is null
            or location_json->>'type' not in ('pdf', 'ocr')
            or location_json ? 'physical_page_index' = false
            or location_json ? 'page_no' = false
            or location_json ? 'page_label' = false
          )
      ),
      'missing_pdf_citation_text_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and (citation_text is null or citation_text not like '% > p.%')
      ),
      'missing_text_block_bbox_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and unit_type = 'CHUNK'
          and chunk_type in ('paragraph', 'text', 'text_block')
          and (location_json ? 'bbox' = false or location_json->'bbox' is null)
      ),
      'ocr_used_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and location_json->>'ocr_used' = 'true'
      ),
      'ocr_confidence_missing_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and location_json->>'ocr_used' = 'true'
          and (
            location_json ? 'ocr_confidence' = false
            or location_json->'ocr_confidence' is null
            or location_json->>'ocr_confidence' = ''
          )
      ),
      'search_unit_parser_names', (
        select coalesce(jsonb_agg(parser_name order by parser_name), '[]'::jsonb)
        from (
          select distinct parser_name from search_unit
          where source_file_id = {sql_literal(source_file_id)}
            and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
        ) names
      ),
      'search_unit_parser_versions', (
        select coalesce(jsonb_agg(parser_version order by parser_version), '[]'::jsonb)
        from (
          select distinct parser_version from search_unit
          where source_file_id = {sql_literal(source_file_id)}
            and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
        ) versions
      ),
      'sample_units', (
        select coalesce(jsonb_agg(jsonb_build_object(
          'unit_type', unit_type,
          'chunk_type', chunk_type,
          'parser_name', parser_name,
          'parser_version', parser_version,
          'location_json', location_json,
          'citation_text', citation_text
        ) order by unit_type, unit_key), '[]'::jsonb)
        from (
          select * from search_unit
          where source_file_id = {sql_literal(source_file_id)}
            and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          order by unit_type, unit_key
          limit 5
        ) sample
      )
    );
    """
    completed = subprocess.run(
        ["docker", "exec", container, "psql", "-U", user, "-d", database, "-At", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(completed.stdout.strip())


def validate_sample_db_report(
    result: dict[str, Any],
    *,
    sample: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    expected = merged_expectations(sample, defaults)

    source_status = result.get("source_status")
    if source_status != "READY":
        raise AssertionError(f"source_status must be READY, got {source_status}")

    min_units = int(expected.get("min_search_units") or 1)
    search_unit_count = int(result.get("search_unit_count") or 0)
    if search_unit_count < min_units:
        raise AssertionError(f"search_unit_count {search_unit_count} < expected {min_units}")

    min_pages = int(expected.get("min_pages") or 1)
    page_count = int(result.get("page_count_from_location") or 0)
    if page_count < min_pages:
        raise AssertionError(f"page_count_from_location {page_count} < expected {min_pages}")

    if bool(expected.get("require_page_metadata", True)):
        page_metadata_count = int(result.get("pdf_page_metadata_count") or 0)
        parsed_page_count = int(result.get("parsed_pdf_page_count") or 0)
        if page_metadata_count <= 0:
            raise AssertionError("pdf_page_metadata_count must be > 0")
        if parsed_page_count and page_metadata_count < parsed_page_count:
            raise AssertionError(
                f"pdf_page_metadata_count {page_metadata_count} < parsed_pdf_page_count {parsed_page_count}"
            )
        missing_page_metadata = int(result.get("missing_page_metadata_count") or 0)
        if missing_page_metadata != 0:
            raise AssertionError("missing_page_metadata_count must be 0")
        inconsistent_page_metadata = int(result.get("inconsistent_location_page_metadata_count") or 0)
        if inconsistent_page_metadata != 0:
            raise AssertionError("inconsistent_location_page_metadata_count must be 0")

    min_parsed = int(expected.get("min_pdf_parsed_artifacts") or 1)
    parsed_count = int(result.get("pdf_parsed_artifact_count") or 0)
    if parsed_count < min_parsed:
        raise AssertionError(f"pdf_parsed_artifact_count {parsed_count} < expected {min_parsed}")

    min_plaintext = int(expected.get("min_pdf_plaintext_artifacts") or 1)
    plaintext_count = int(result.get("pdf_plaintext_artifact_count") or 0)
    if plaintext_count < min_plaintext:
        raise AssertionError(f"pdf_plaintext_artifact_count {plaintext_count} < expected {min_plaintext}")

    missing = int(result.get("missing_required_metadata_count") or 0)
    if missing != 0:
        raise AssertionError("missing_required_metadata_count must be 0")

    invalid_location = int(result.get("invalid_pdf_location_count") or 0)
    if invalid_location != 0:
        raise AssertionError("invalid_pdf_location_count must be 0")

    missing_citation = int(result.get("missing_pdf_citation_text_count") or 0)
    if missing_citation != 0:
        raise AssertionError("missing_pdf_citation_text_count must be 0")

    if bool(expected.get("require_bbox_if_text_block")):
        missing_bbox = int(result.get("missing_text_block_bbox_count") or 0)
        if missing_bbox != 0:
            raise AssertionError("missing_text_block_bbox_count must be 0")

    expected_parser_name = expected.get("require_parser_name") or expected.get("must_have_parser_name")
    if expected_parser_name:
        parser_names = set(result.get("parsed_artifact_parser_names") or [])
        parser_names.update(result.get("search_unit_parser_names") or [])
        compatible_parser_names = {expected_parser_name}
        if expected_parser_name == "pymupdf" and int(result.get("ocr_used_count") or 0) > 0:
            compatible_parser_names.add("pymupdf+paddleocr")
        if parser_names.isdisjoint(compatible_parser_names):
            raise AssertionError(f"parser_name {expected_parser_name} not found")

    expected_parser_version = expected.get("require_parser_version") or expected.get("must_have_parser_version")
    if expected_parser_version:
        parser_versions = set(result.get("parsed_artifact_parser_versions") or [])
        parser_versions.update(result.get("search_unit_parser_versions") or [])
        compatible_parser_versions = {expected_parser_version}
        if expected_parser_version == "pdf-extract-v1" and int(result.get("ocr_used_count") or 0) > 0:
            compatible_parser_versions.add("pdf-extract-v2")
        if parser_versions.isdisjoint(compatible_parser_versions):
            raise AssertionError(f"parser_version {expected_parser_version} not found")


def merged_expectations(sample: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    nested = sample.get("expected")
    if isinstance(nested, dict):
        merged.update(nested)
    flat_aliases = {
        "expected_min_search_units": "min_search_units",
        "expected_min_pages": "min_pages",
        "must_have_parser_name": "require_parser_name",
        "must_have_parser_version": "require_parser_version",
    }
    for source_key, target_key in flat_aliases.items():
        if source_key in sample:
            merged[target_key] = sample[source_key]
    return merged


def add_aggregate_metrics(report: dict[str, Any]) -> None:
    total = int(report.get("total_samples") or 0)
    samples = list(report.get("samples") or [])
    parser_successes = sum(1 for item in samples if item.get("job_status") == "SUCCEEDED")
    zero_indexable = sum(
        1
        for item in samples
        if item.get("job_status") == "SUCCEEDED"
        and int(item.get("search_unit_count") or 0) == 0
    )
    missing_metadata = sum(
        int(item.get("missing_required_metadata_count") or 0)
        for item in samples
        if item.get("missing_required_metadata_count") is not None
    )
    invalid_locations = sum(int(item.get("invalid_pdf_location_count") or 0) for item in samples)
    missing_citations = sum(int(item.get("missing_pdf_citation_text_count") or 0) for item in samples)
    missing_page_metadata = sum(int(item.get("missing_page_metadata_count") or 0) for item in samples)
    inconsistent_page_metadata = sum(
        int(item.get("inconsistent_location_page_metadata_count") or 0) for item in samples
    )
    parsing_latencies = [
        float(item["parsing_latency_seconds"])
        for item in samples
        if item.get("parsing_latency_seconds") is not None
    ]
    indexing_latencies = [
        float(item["indexing_latency_seconds"])
        for item in samples
        if item.get("indexing_latency_seconds") is not None
    ]
    report["metrics"] = {
        "parser_success_rate": round(parser_successes / total, 4) if total else 0.0,
        "unsupported_file_rate": 0.0,
        "zero_indexable_chunk_count": zero_indexable,
        "missing_required_metadata_count": missing_metadata,
        "invalid_pdf_location_count": invalid_locations,
        "missing_pdf_citation_text_count": missing_citations,
        "missing_page_metadata_count": missing_page_metadata,
        "inconsistent_location_page_metadata_count": inconsistent_page_metadata,
        "parsing_latency_p95_seconds": percentile(parsing_latencies, 0.95),
        "indexing_latency_p95_seconds": percentile(indexing_latencies, 0.95),
        "fatal_warning_count": 0,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return round(ordered[index], 3)


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
