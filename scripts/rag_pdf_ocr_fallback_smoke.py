"""Smoke runner for PDF_EXTRACT PaddleOCR fallback.

The runner generates an image-only PDF, uploads it, starts pdf-extract, and
checks that OCR-specific location/page metadata is stored. If PaddleOCR is not
installed in the local environment, the runner writes a SKIPPED report and
exits 0 by default; pass --require-paddle to fail instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_pdf_ingestion_smoke import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_DB_CONTAINER,
    DEFAULT_DB_NAME,
    DEFAULT_DB_USER,
    start_pdf_extract,
    upload_pdf,
    wait_job,
)


DEFAULT_REPORT = Path("reports/rag_pdf_ocr_fallback_smoke_report.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "run_id": utc_run_id(),
        "status": "FAILED",
        "source_file_id": None,
        "job_id": None,
        "warnings": [],
    }

    missing_dependencies = [
        name
        for name in ("paddleocr", "paddle")
        if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        message = "missing PaddleOCR dependencies in this Python environment: " + ", ".join(missing_dependencies)
        if args.require_paddle:
            report["errors"] = [message]
            write_json_report(Path(args.report), report)
            print(f"[FAIL] {message}", file=sys.stderr)
            return 2
        report["status"] = "SKIPPED"
        report["warnings"].append(message)
        write_json_report(Path(args.report), report)
        print(f"[SKIP] {message}")
        return 0

    with tempfile.TemporaryDirectory(prefix="rag-pdf-ocr-smoke-") as tmp:
        pdf_path = Path(tmp) / "image-only-ocr.pdf"
        write_image_only_pdf(pdf_path)
        client = httpx.Client(base_url=args.base_url, timeout=args.http_timeout)
        source = upload_pdf(client, pdf_path)
        source_id = str(source["sourceFileId"])
        report["source_file_id"] = source_id
        print(f"[1/4] uploaded OCR-needed PDF sourceFileId={source_id}")

        job = start_pdf_extract(client, source_id, args.extract_path)
        job_id = str(job["jobId"])
        report["job_id"] = job_id
        print(f"[2/4] started PDF_EXTRACT jobId={job_id}")

        final = wait_job(
            client,
            job_id,
            timeout_seconds=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
        report["job_status"] = final.get("status")
        print(f"[3/4] job status={final.get('status')}")
        if final.get("status") != "SUCCEEDED":
            report["job"] = final
            write_json_report(Path(args.report), report)
            return 3

        if args.skip_db_check:
            report["warnings"].append("DB validation skipped")
            report["status"] = "PASSED"
            write_json_report(Path(args.report), report)
            return 0

        db_report = query_ocr_db_report(
            container=args.db_container,
            user=args.db_user,
            database=args.db_name,
            source_file_id=source_id,
        )
        report["db_report"] = db_report
        try:
            validate_ocr_report(db_report)
        except AssertionError as exc:
            report["errors"] = [str(exc)]
            write_json_report(Path(args.report), report)
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 4
        report["status"] = "PASSED"
        write_json_report(Path(args.report), report)
        print("[4/4] OCR fallback DB metadata verified")
        print(json.dumps(db_report, ensure_ascii=False, indent=2))
        return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--extract-path", default="pdf-extract")
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--poll-timeout", type=float, default=240.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--db-container", default=DEFAULT_DB_CONTAINER)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--skip-db-check", action="store_true")
    parser.add_argument("--require-paddle", action="store_true")
    return parser.parse_args(argv)


def write_image_only_pdf(path: Path) -> None:
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError as ex:
        raise RuntimeError("PyMuPDF and Pillow are required to generate an OCR smoke PDF") from ex

    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 90), "PaddleOCR fallback smoke amount 12345", fill="black")
    png = BytesIO()
    image.save(png, format="PNG")

    document = fitz.open()
    page = document.new_page(width=595, height=200)
    page.insert_image(fitz.Rect(36, 36, 559, 164), stream=png.getvalue())
    document.save(path)
    document.close()


def query_ocr_db_report(
    *,
    container: str,
    user: str,
    database: str,
    source_file_id: str,
) -> dict[str, Any]:
    sql = f"""
    select jsonb_build_object(
      'ocr_search_unit_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and location_json->>'ocr_used' = 'true'
      ),
      'paddle_ocr_search_unit_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and location_json->>'ocr_used' = 'true'
          and location_json->>'ocr_engine' = 'paddleocr'
      ),
      'ocr_confidence_missing_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and location_json->>'ocr_used' = 'true'
          and (
            location_json ? 'ocr_confidence' = false
            or location_json->'ocr_confidence' is null
            or location_json->>'ocr_confidence' = ''
          )
      ),
      'ocr_confidence_avg', (
        select avg((location_json->>'ocr_confidence')::double precision)
        from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and location_json->>'ocr_used' = 'true'
          and location_json ? 'ocr_confidence'
          and location_json->>'ocr_confidence' <> ''
      ),
      'ocr_page_metadata_count', (
        select count(*) from pdf_page_metadata
        where source_file_id = {sql_literal(source_file_id)}
          and ocr_used = true
          and ocr_engine = 'paddleocr'
          and ocr_confidence_avg is not null
      ),
      'low_trust_ocr_chunk_count', (
        select count(*) from search_unit
        where source_file_id = {sql_literal(source_file_id)}
          and location_json->>'ocr_used' = 'true'
          and quality_score is not null
          and confidence_score is not null
          and quality_score < confidence_score
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


def validate_ocr_report(report: dict[str, Any]) -> None:
    checks = {
        "ocr_search_unit_count": int(report.get("ocr_search_unit_count") or 0) > 0,
        "paddle_ocr_search_unit_count": int(report.get("paddle_ocr_search_unit_count") or 0) > 0,
        "ocr_confidence_missing_count": int(report.get("ocr_confidence_missing_count") or 0) == 0,
        "ocr_page_metadata_count": int(report.get("ocr_page_metadata_count") or 0) > 0,
        "low_trust_ocr_chunk_count": int(report.get("low_trust_ocr_chunk_count") or 0) > 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"OCR fallback smoke validation failed: {failed}")


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
