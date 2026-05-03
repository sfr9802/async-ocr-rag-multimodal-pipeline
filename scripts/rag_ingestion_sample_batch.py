"""Manifest-driven smoke runner for real XLSX RAG ingestion samples.

The runner uploads each XLSX from a manifest, starts the xlsx-extract job,
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


DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MANIFEST = Path("samples/rag_ingestion_manifest.json")
DEFAULT_REPORT = Path("reports/rag_ingestion_sample_batch_report.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_CONTAINER = "aipipeline-postgres"
DEFAULT_DB_USER = "aipipeline"
DEFAULT_DB_NAME = "aipipeline"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}


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
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
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
        if str(sample.get("file_type") or "").lower() != "xlsx":
            raise ValueError("Only xlsx samples are supported by this runner")
        path = resolve_sample_path(str(sample.get("file_path") or sample.get("path") or ""), manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")

        source = upload_xlsx(client, path)
        source_id = str(source["sourceFileId"])
        result["source_file_id"] = source_id
        parsing_started = time.perf_counter()
        job = start_xlsx_extract(client, source_id)
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
            errors.append("xlsx-extract job did not succeed")
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
            expected_sheets=list(sample.get("expected_sheets") or []),
            expected_citation_patterns=list(sample.get("expected_citation_patterns") or []),
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


def upload_xlsx(client: httpx.Client, path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post(
            "/api/v1/library/source-files",
            files={
                "file": (
                    path.name,
                    handle,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    response.raise_for_status()
    return response.json()


def start_xlsx_extract(client: httpx.Client, source_file_id: str) -> dict[str, Any]:
    response = client.post(f"/api/v1/library/source-files/{source_file_id}/xlsx-extract")
    response.raise_for_status()
    return response.json()


def wait_job(
    client: httpx.Client,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        body = response.json()
        if body.get("status") in TERMINAL_STATUSES:
            return body
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def query_sample_db_report(
    *,
    container: str,
    user: str,
    database: str,
    source_file_id: str,
    expected_sheets: list[str],
    expected_citation_patterns: list[str],
) -> dict[str, Any]:
    sheet_checks = ", ".join(
        f"jsonb_build_object('sheet_name', {sql_literal(sheet)}, 'count', "
        f"(select count(*) from search_unit where source_file_id = {sql_literal(source_file_id)} "
        f"and location_json->>'sheet_name' = {sql_literal(sheet)}))"
        for sheet in expected_sheets
    )
    sheet_json = f"jsonb_build_array({sheet_checks})" if sheet_checks else "'[]'::jsonb"
    pattern_checks = ", ".join(
        f"jsonb_build_object('pattern', {sql_literal(pattern)}, 'count', "
        f"(select count(*) from search_unit where source_file_id = {sql_literal(source_file_id)} "
        f"and citation_text like '%' || {sql_literal(pattern)} || '%'))"
        for pattern in expected_citation_patterns
    )
    pattern_json = f"jsonb_build_array({pattern_checks})" if pattern_checks else "'[]'::jsonb"
    sql = f"""
    select jsonb_build_object(
      'search_unit_count', (
        select count(*) from search_unit where source_file_id = {sql_literal(source_file_id)}
      ),
      'missing_required_metadata_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and lower(source_file_type) in ('spreadsheet', 'xlsx')
          and (parser_version is null or location_json is null or citation_text is null)
      ),
      'xlsx_table_search_unit_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and lower(source_file_type) in ('spreadsheet', 'xlsx')
          and (unit_type = 'TABLE' or chunk_type = 'row_group')
      ),
      'table_metadata_count', (
        select count(*) from table_metadata
        where source_file_id = {sql_literal(source_file_id)}
      ),
      'missing_table_metadata_count', (
        select greatest((
          select count(*) from search_unit
          where source_file_id = {sql_literal(source_file_id)}
            and lower(source_file_type) in ('spreadsheet', 'xlsx')
            and (unit_type = 'TABLE' or chunk_type = 'row_group')
        ) - (
          select count(*) from table_metadata
          where source_file_id = {sql_literal(source_file_id)}
        ), 0)
      ),
      'cell_metadata_count', (
        select count(*) from cell_metadata
        where source_file_id = {sql_literal(source_file_id)}
      ),
      'formula_cell_metadata_count', (
        select count(*) from cell_metadata
        where source_file_id = {sql_literal(source_file_id)}
          and formula is not null
      ),
      'formatted_cell_metadata_count', (
        select count(*) from cell_metadata
        where source_file_id = {sql_literal(source_file_id)}
          and formatted_value is not null
      ),
      'hidden_search_unit_leakage_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and lower(source_file_type) in ('spreadsheet', 'xlsx')
          and (
            location_json->>'hidden' = 'true'
            or location_json->>'hidden_sheet' = 'true'
            or coalesce(citation_text, '') like '%숨김%'
          )
      ),
      'expected_sheet_counts', {sheet_json},
      'citation_pattern_counts', {pattern_json},
      'sample_units', (
        select coalesce(jsonb_agg(jsonb_build_object(
          'unit_type', unit_type,
          'chunk_type', chunk_type,
          'citation_text', citation_text,
          'parser_version', parser_version,
          'location_json', location_json
        ) order by unit_type, unit_key), '[]'::jsonb)
        from (
          select * from search_unit
          where source_file_id = {sql_literal(source_file_id)}
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
    min_units = int(sample.get("expected_min_search_units") or defaults.get("min_search_units") or 1)
    search_unit_count = int(result.get("search_unit_count") or 0)
    if search_unit_count < min_units:
        raise AssertionError(f"search_unit_count {search_unit_count} < expected {min_units}")

    missing = int(result.get("missing_required_metadata_count") or 0)
    if missing != 0:
        raise AssertionError("missing_required_metadata_count must be 0")

    if bool(defaults.get("require_table_metadata", True)):
        table_like_units = int(result.get("xlsx_table_search_unit_count") or 0)
        table_count = int(result.get("table_metadata_count") or 0)
        if table_like_units > 0 and table_count <= 0:
            raise AssertionError("table_metadata_count must be > 0")
        missing_tables = int(result.get("missing_table_metadata_count") or 0)
        if missing_tables != 0:
            raise AssertionError("missing_table_metadata_count must be 0")

    if bool(defaults.get("require_cell_metadata", True)):
        cell_count = int(result.get("cell_metadata_count") or 0)
        if cell_count <= 0:
            raise AssertionError("cell_metadata_count must be > 0")

    hidden_leakage = int(result.get("hidden_search_unit_leakage_count") or 0)
    if hidden_leakage != 0:
        raise AssertionError("hidden_search_unit_leakage_count must be 0")

    missing_sheets = [
        item["sheet_name"]
        for item in result.get("expected_sheet_counts", [])
        if int(item.get("count") or 0) == 0
    ]
    if missing_sheets:
        raise AssertionError(f"expected sheets not found in location_json: {missing_sheets}")

    missing_patterns = [
        item["pattern"]
        for item in result.get("citation_pattern_counts", [])
        if int(item.get("count") or 0) == 0
    ]
    if missing_patterns:
        raise AssertionError(f"citation patterns not found: {missing_patterns}")


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
    missing_table_metadata = sum(int(item.get("missing_table_metadata_count") or 0) for item in samples)
    hidden_leakage = sum(int(item.get("hidden_search_unit_leakage_count") or 0) for item in samples)
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
        "missing_table_metadata_count": missing_table_metadata,
        "hidden_search_unit_leakage_count": hidden_leakage,
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
    sys.exit(main())
