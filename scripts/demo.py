"""One-command demo for the AI processing platform.

    python scripts/demo.py                      # full pipeline demo
    python scripts/demo.py --capability OCR      # OCR-only demo
    python scripts/demo.py --self-test           # offline sanity check

Generates a sample PDF (Korean + English mixed content), submits it as a
MULTIMODAL job, polls until completion, and pretty-prints the results.

Prerequisites (for the full demo, not --self-test):
  - core-api running on localhost:8080
  - ai-worker running with at least OCR + RAG capabilities
  - reportlab installed (pip install reportlab)

Optional:
  - rich installed for coloured panel/table output (pip install rich)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("DEMO_BASE_URL", "http://localhost:8080")
POLL_INTERVAL = 1.0
POLL_TIMEOUT = 60.0
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLE_PDF = ASSETS_DIR / "demo_sample.pdf"


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _ensure_sample_pdf() -> Path:
    """Create a 2-page sample PDF if it does not already exist."""
    if SAMPLE_PDF.exists():
        return SAMPLE_PDF

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen.canvas import Canvas
    except ImportError:
        raise RuntimeError(
            "reportlab is required to generate the sample PDF. "
            "pip install reportlab"
        )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    width, height = A4
    c = Canvas(str(SAMPLE_PDF), pagesize=A4)

    # ---- page 1: mixed Korean + English body text ----
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, "AI Pipeline Demo Document")

    c.setFont("Helvetica", 11)
    body = [
        "This document is a sample input for the multimodal AI pipeline.",
        "The pipeline performs OCR, vision analysis, and text retrieval.",
        "",
        "Section 1: Overview",
        "The platform accepts PDF and image uploads, extracts text via",
        "Tesseract OCR, generates visual descriptions, retrieves relevant",
        "context from a FAISS vector index, and produces a grounded answer.",
        "",
        "Section 2: Technical Stack",
        "- Core API: Spring Boot 4.0.3, Java 21, PostgreSQL 18",
        "- Worker: Python 3.12, FAISS, sentence-transformers (bge-m3)",
        "- Queue: Redis (BRPOP dispatch)",
        "- Storage: Local filesystem (phase 1), S3/MinIO (phase 2+)",
        "",
        "Section 3: Capabilities",
        "MOCK  - echo capability for smoke testing",
        "RAG   - text retrieval with grounded generation",
        "OCR   - Tesseract + PyMuPDF text extraction",
        "MULTIMODAL - OCR + vision + RAG fusion pipeline",
    ]
    y = height - 90
    for line in body:
        c.drawString(40, y, line)
        y -= 16

    c.showPage()

    # ---- page 2: table with metrics ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, "Performance Summary")

    c.setFont("Helvetica", 10)
    table_data = [
        ("Metric", "Value", "Unit"),
        ("OCR throughput", "2.4", "pages/sec"),
        ("Embedding latency", "45", "ms/query"),
        ("Retrieval top-5 hit rate", "0.82", "ratio"),
        ("End-to-end p50", "1.8", "seconds"),
        ("End-to-end p99", "4.2", "seconds"),
    ]
    y = height - 90
    col_x = [40, 220, 350]
    for row in table_data:
        for i, cell in enumerate(row):
            c.drawString(col_x[i], y, cell)
        y -= 18
        if row == table_data[0]:
            c.line(40, y + 8, 450, y + 8)

    # simple rectangle as a visual element
    c.setStrokeColorRGB(0.2, 0.4, 0.8)
    c.setFillColorRGB(0.9, 0.93, 1.0)
    c.roundRect(40, y - 80, 400, 60, 8, fill=True, stroke=True)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(55, y - 50, "Note: metrics are illustrative. Run the eval")
    c.drawString(55, y - 64, "harness for real numbers on your hardware.")

    c.showPage()
    c.save()
    return SAMPLE_PDF


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    headers: dict[str, str] = {}
    secret = os.environ.get("AIPIPELINE_INTERNAL_SECRET")
    if secret:
        headers["X-Internal-Secret"] = secret
    return httpx.Client(base_url=BASE_URL, timeout=15.0, headers=headers)


def _health_check(client: httpx.Client) -> bool:
    try:
        r = client.get("/actuator/health")
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _submit_job(
    client: httpx.Client,
    pdf_path: Path,
    capability: str,
    question: str,
) -> dict:
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data: dict[str, str] = {"capability": capability}
        if question:
            data["text"] = question
        r = client.post("/api/v1/jobs", data=data, files=files)
    r.raise_for_status()
    return r.json()


def _poll_job(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        status = job["status"]
        if status in ("SUCCEEDED", "FAILED"):
            return job
        _print_status(status)
        time.sleep(POLL_INTERVAL)
    print("[TIMEOUT] job did not complete within 60 seconds")
    sys.exit(2)


def _fetch_result(client: httpx.Client, job_id: str) -> dict:
    r = client.get(f"/api/v1/jobs/{job_id}/result")
    r.raise_for_status()
    return r.json()


def _download_artifact(client: httpx.Client, access_url: str) -> bytes:
    r = client.get(access_url)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_status_printed: set[str] = set()


def _print_status(status: str) -> None:
    if status not in _status_printed:
        print(f"  ... {status}")
        _status_printed.add(status)


def _print_results_rich(
    final: str | None,
    vision: dict | None,
    retrieval: dict | None,
    job: dict,
) -> None:
    console = Console()
    console.print()

    # job summary
    console.print(Panel(
        f"[bold]Job ID:[/bold] {job['jobId']}\n"
        f"[bold]Capability:[/bold] {job.get('capability', 'N/A')}\n"
        f"[bold]Status:[/bold] [green]{job['status']}[/green]",
        title="Job Summary",
        border_style="blue",
    ))

    # final response
    if final:
        console.print(Panel(
            final.strip(),
            title="FINAL_RESPONSE",
            border_style="green",
        ))

    # vision result
    if vision and vision.get("pages"):
        table = Table(title="VISION_RESULT", show_lines=True)
        table.add_column("Page", style="cyan", width=6)
        table.add_column("Caption", ratio=1)
        for page in vision["pages"]:
            table.add_row(
                str(page.get("pageNumber", "?")),
                page.get("caption", "(no caption)"),
            )
        console.print(table)

    # retrieval result top-3
    if retrieval and retrieval.get("results"):
        table = Table(title="RETRIEVAL_RESULT (top 3)", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Score", width=8)
        table.add_column("Doc / Section", width=24)
        table.add_column("Text", ratio=1)
        for hit in retrieval["results"][:3]:
            section = hit.get("section", "")
            doc_id = hit.get("docId", "")
            label = f"{doc_id}#{section}" if section else doc_id
            text = hit.get("text", "")
            if len(text) > 120:
                text = text[:117] + "..."
            table.add_row(
                str(hit.get("rank", "?")),
                f"{hit.get('score', 0):.3f}",
                label,
                text,
            )
        console.print(table)

    console.print()


def _print_results_plain(
    final: str | None,
    vision: dict | None,
    retrieval: dict | None,
    job: dict,
) -> None:
    print()
    print("=" * 60)
    print(f"  Job ID:     {job['jobId']}")
    print(f"  Capability: {job.get('capability', 'N/A')}")
    print(f"  Status:     {job['status']}")
    print("=" * 60)

    if final:
        print()
        print("--- FINAL_RESPONSE ---")
        print(final.strip())

    if vision and vision.get("pages"):
        print()
        print("--- VISION_RESULT ---")
        for page in vision["pages"]:
            print(f"  Page {page.get('pageNumber', '?')}: "
                  f"{page.get('caption', '(no caption)')}")

    if retrieval and retrieval.get("results"):
        print()
        print("--- RETRIEVAL_RESULT (top 3) ---")
        for hit in retrieval["results"][:3]:
            section = hit.get("section", "")
            doc_id = hit.get("docId", "")
            label = f"{doc_id}#{section}" if section else doc_id
            text = hit.get("text", "")
            if len(text) > 120:
                text = text[:117] + "..."
            print(f"  #{hit.get('rank', '?')}  "
                  f"score={hit.get('score', 0):.3f}  "
                  f"{label}")
            print(f"      {text}")

    print()


def _print_results(
    final: str | None,
    vision: dict | None,
    retrieval: dict | None,
    job: dict,
) -> None:
    if _RICH:
        _print_results_rich(final, vision, retrieval, job)
    else:
        _print_results_plain(final, vision, retrieval, job)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    print("[self-test] checking imports ...")
    errors: list[str] = []

    if httpx is None:
        errors.append("httpx is not installed (pip install httpx)")

    try:
        from reportlab.pdfgen.canvas import Canvas  # noqa: F401
    except ImportError:
        errors.append("reportlab is not installed (pip install reportlab)")

    print("[self-test] generating sample PDF ...")
    try:
        path = _ensure_sample_pdf()
        size = path.stat().st_size
        print(f"  OK: {path} ({size:,} bytes)")
    except Exception as exc:
        errors.append(f"PDF generation failed: {exc}")

    print(f"[self-test] rich available: {_RICH}")
    print(f"[self-test] base URL: {BASE_URL}")

    if errors:
        print()
        for e in errors:
            print(f"  FAIL: {e}")
        return 1

    print()
    print("All self-test checks passed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command demo for the AI processing platform.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Offline check: verify imports, PDF generation, and wiring.",
    )
    parser.add_argument(
        "--capability",
        default="MULTIMODAL",
        help="Capability to demo (default: MULTIMODAL).",
    )
    parser.add_argument(
        "--question",
        default="Summarize this document and list the key metrics.",
        help="User question sent with the upload.",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if httpx is None:
        print("ERROR: httpx is required. pip install httpx", file=sys.stderr)
        return 1

    # Step 1: health check
    print(f"[1/5] health check ({BASE_URL}) ...")
    client = _make_client()
    if not _health_check(client):
        print("  WARN: /actuator/health not reachable — continuing anyway")

    # Step 2: ensure sample PDF
    print("[2/5] preparing sample PDF ...")
    try:
        pdf = _ensure_sample_pdf()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  {pdf} ({pdf.stat().st_size:,} bytes)")

    # Step 3: submit job
    print(f"[3/5] submitting {args.capability} job ...")
    created = _submit_job(client, pdf, args.capability, args.question)
    job_id = created["jobId"]
    print(f"  jobId = {job_id}  status = {created['status']}")

    # Step 4: poll
    print("[4/5] polling for completion ...")
    job = _poll_job(client, job_id)
    if job["status"] != "SUCCEEDED":
        print(f"  FAILED: {job.get('errorCode')} / {job.get('errorMessage')}")
        return 3

    print(f"  SUCCEEDED")

    # Step 5: fetch and display results
    print("[5/5] fetching results ...")
    result = _fetch_result(client, job_id)

    final_text: str | None = None
    vision_data: dict | None = None
    retrieval_data: dict | None = None

    for artifact in result.get("outputs", []):
        atype = artifact.get("type", "")
        url = artifact.get("accessUrl", "")
        if not url:
            continue
        raw = _download_artifact(client, url)

        if atype == "FINAL_RESPONSE":
            final_text = raw.decode("utf-8", errors="replace")
        elif atype == "VISION_RESULT":
            try:
                vision_data = json.loads(raw)
            except json.JSONDecodeError:
                pass
        elif atype == "RETRIEVAL_RESULT":
            try:
                retrieval_data = json.loads(raw)
            except json.JSONDecodeError:
                pass

    _print_results(final_text, vision_data, retrieval_data, job)

    print("Demo complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except httpx.HTTPError as exc:
        print(f"[FAIL] HTTP error: {exc}", file=sys.stderr)
        sys.exit(10)
