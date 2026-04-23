"""End-to-end smoke test for the AI processing platform.

Preconditions:
  - PostgreSQL running (docker compose up -d postgres)
  - Redis running       (docker compose up -d redis)
  - core-api running    (mvn -f core-api/pom.xml spring-boot:run)
  - ai-worker running   (python -m app.main  inside ai-worker/)

What it does:
  1. POSTs a text job to /api/v1/jobs
  2. Polls /api/v1/jobs/{id} until status is SUCCEEDED or FAILED
  3. Fetches /api/v1/jobs/{id}/result and downloads the first OUTPUT artifact
  4. Prints a short human-readable report

Exit code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.parse import urljoin

import httpx

BASE = "http://localhost:8080"
POLL_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.5


def main() -> int:
    # If AIPIPELINE_INTERNAL_SECRET is set, include the header on every
    # request. The current flow only hits public /api/v1/** endpoints (not
    # gated), but the header is harmless and keeps the smoke test ready for
    # future internal-endpoint assertions.
    headers: dict[str, str] = {}
    secret = os.environ.get("AIPIPELINE_INTERNAL_SECRET")
    if secret:
        headers["X-Internal-Secret"] = secret
    client = httpx.Client(base_url=BASE, timeout=10.0, headers=headers)

    print("[1/4] submitting text job ...")
    submit = client.post("/api/v1/jobs", json={
        "capability": "MOCK",
        "text": "e2e smoke test payload",
    })
    submit.raise_for_status()
    created = submit.json()
    job_id = created["jobId"]
    print(f"     jobId = {job_id}  status = {created['status']}")

    print("[2/4] polling job status ...")
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    final = None
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        print(f"     status = {job['status']}")
        if job["status"] in ("SUCCEEDED", "FAILED"):
            final = job
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if final is None:
        print("[FAIL] timed out waiting for terminal status")
        return 2
    if final["status"] != "SUCCEEDED":
        print(f"[FAIL] job ended in {final['status']}: "
              f"{final.get('errorCode')} / {final.get('errorMessage')}")
        return 3

    print("[3/4] fetching job result ...")
    r = client.get(f"/api/v1/jobs/{job_id}/result")
    r.raise_for_status()
    result = r.json()
    outputs = result.get("outputs", [])
    if not outputs:
        print("[FAIL] no output artifacts on completed job")
        return 4
    output = outputs[0]
    print(f"     output artifactId = {output['id']} type = {output['type']}")

    print("[4/4] downloading output artifact content ...")
    dl = client.get(output["accessUrl"])
    dl.raise_for_status()
    body = dl.content
    try:
        parsed = json.loads(body)
        print(f"     parsed JSON keys = {list(parsed.keys())}")
    except Exception:
        print(f"     raw bytes len = {len(body)}")

    print("OK - pipeline survived the round trip.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPError as exc:
        print(f"[FAIL] HTTP error: {exc}")
        sys.exit(10)
