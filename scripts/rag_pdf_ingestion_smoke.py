"""Smoke runner for PDF RAG ingestion v2 metadata.

The runner uploads a PDF, starts the PDF extract endpoint, polls job status,
and optionally validates that PDF search units carry parser_version,
location_json, and citation_text. If --file is omitted, it tries to generate a
minimal text-layer PDF using reportlab first, then PyMuPDF.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_DB_CONTAINER = "aipipeline-postgres"
DEFAULT_DB_USER = "aipipeline"
DEFAULT_DB_NAME = "aipipeline"
DEFAULT_REPORT = Path("reports/rag_pdf_ingestion_smoke_report.json")
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_report: dict[str, Any] = {
        "run_id": utc_run_id(),
        "status": "FAILED",
        "source_file_id": None,
        "job_id": None,
        "job_status": None,
        "generated_by": None,
        "db_report": None,
        "warnings": [],
    }
    with tempfile.TemporaryDirectory(prefix="rag-pdf-ingestion-smoke-") as tmp:
        pdf_path = Path(args.file) if args.file else Path(tmp) / "native_text_sample.pdf"
        if args.file:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
        else:
            generated_by = generate_minimal_pdf(pdf_path)
            run_report["generated_by"] = generated_by
            print(f"[0/4] generated text-layer PDF with {generated_by}: {pdf_path}")

        client = httpx.Client(base_url=args.base_url, timeout=args.http_timeout)
        source = upload_pdf(client, pdf_path)
        source_id = str(source["sourceFileId"])
        run_report["source_file_id"] = source_id
        print(f"[1/4] uploaded sourceFileId={source_id} name={source['originalFileName']}")

        job = start_pdf_extract(client, source_id, args.extract_path)
        job_id = str(job["jobId"])
        run_report["job_id"] = job_id
        print(f"[2/4] started PDF extract jobId={job_id} endpoint={args.extract_path}")

        final = wait_job(
            client,
            job_id,
            timeout_seconds=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
        run_report["job_status"] = final.get("status")
        print(f"[3/4] job status={final.get('status')}")
        if final.get("status") != "SUCCEEDED":
            run_report["job"] = final
            write_json_report(Path(args.report), run_report)
            print(json.dumps(final, ensure_ascii=False, indent=2))
            return 3

        if args.skip_db_check:
            run_report["warnings"].append("DB validation skipped")
            run_report["status"] = "PASSED"
            write_json_report(Path(args.report), run_report)
            print("[4/4] DB check skipped")
            return 0

        report = query_pdf_db_report(
            container=args.db_container,
            user=args.db_user,
            database=args.db_name,
            source_file_id=source_id,
        )
        validate_pdf_report(report)
        run_report["db_report"] = report
        run_report["status"] = "PASSED"
        write_json_report(Path(args.report), run_report)
        print("[4/4] DB PDF metadata verified")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="PDF file to upload. Generates a minimal PDF when omitted.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--extract-path", default="pdf-extract")
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--db-container", default=DEFAULT_DB_CONTAINER)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--skip-db-check", action="store_true")
    return parser.parse_args(argv)


def generate_minimal_pdf(path: Path) -> str:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path), pagesize=letter)
        c.drawString(72, 720, "Native text PDF smoke sample")
        c.drawString(72, 700, "The contract purpose is searchable on page one.")
        c.save()
        return "reportlab"
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Native text PDF smoke sample")
        page.insert_text((72, 96), "The contract purpose is searchable on page one.")
        doc.save(path)
        doc.close()
        return "pymupdf"
    except Exception as exc:
        raise RuntimeError("Install reportlab or pymupdf to generate a fallback PDF") from exc


def upload_pdf(client: httpx.Client, path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post(
            "/api/v1/library/source-files",
            files={"file": (path.name, handle, "application/pdf")},
        )
    response.raise_for_status()
    return response.json()


def start_pdf_extract(client: httpx.Client, source_file_id: str, extract_path: str) -> dict[str, Any]:
    normalized = extract_path.strip("/")
    response = client.post(f"/api/v1/library/source-files/{source_file_id}/{normalized}")
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


def query_pdf_db_report(
    *,
    container: str,
    user: str,
    database: str,
    source_file_id: str,
) -> dict[str, Any]:
    sql = f"""
    select jsonb_build_object(
      'source_file_id', {sql_literal(source_file_id)},
      'pdf_search_unit_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
      ),
      'missing_pdf_citation_metadata', (
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
      'ocr_used_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          and location_json->>'ocr_used' = 'true'
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
      'sample_units', (
        select coalesce(jsonb_agg(jsonb_build_object(
          'source_file_name', source_file_name,
          'source_file_type', source_file_type,
          'parser_version', parser_version,
          'location_json', location_json,
          'citation_text', citation_text
        ) order by id), '[]'::jsonb)
        from (
          select * from search_unit
          where source_file_id = {sql_literal(source_file_id)}
            and source_file_type in ('PDF', 'pdf', 'OCR', 'ocr')
          order by created_at desc
          limit 10
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


def validate_pdf_report(report: dict[str, Any]) -> None:
    checks = {
        "pdf_search_unit_count": int(report.get("pdf_search_unit_count") or 0) > 0,
        "missing_pdf_citation_metadata": int(report.get("missing_pdf_citation_metadata") or 0) == 0,
        "invalid_pdf_location_count": int(report.get("invalid_pdf_location_count") or 0) == 0,
        "missing_pdf_citation_text_count": int(report.get("missing_pdf_citation_text_count") or 0) == 0,
        "ocr_used_count": int(report.get("ocr_used_count") or 0) == 0,
        "pdf_page_metadata_count": int(report.get("pdf_page_metadata_count") or 0) > 0,
        "missing_page_metadata_count": int(report.get("missing_page_metadata_count") or 0) == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"PDF ingestion smoke validation failed: {failed}")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
