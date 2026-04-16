"""Full-stack smoke runner for the async OCR/RAG/multimodal pipeline.

Exercises the **real** async pipeline through the **real** core-api:
submits one job for each registered capability, polls status until
terminal, fetches the result, and asserts that the expected artifact
shapes landed. Emits both a human-readable console summary and a
machine-readable JSON report.

Usage (from ai-worker/)::

    python -m scripts.smoke_runner
    python -m scripts.smoke_runner --report smoke-report.json
    python -m scripts.smoke_runner --base-url http://localhost:8080 --timeout 120
    python -m scripts.smoke_runner --only MOCK,OCR

Design notes:
    * The runner NEVER mocks or short-circuits core-api. It submits HTTP
      requests, polls HTTP endpoints, and downloads HTTP artifacts.
      That is the whole point — to prove that the Spring side, Redis,
      the worker, and the storage layer are wired together correctly.
    * Shape assertions are pure functions on parsed JSON, so unit tests
      can exercise them with fabricated responses without needing a
      running pipeline.
    * Every job submission carries a unique per-capability payload so
      repeat runs can be distinguished from each other in the database
      without colliding.
    * This tool only proves connectivity + contract integrity. It does
      NOT prove OCR/RAG/MULTIMODAL quality — assertions focus on
      artifact *types* and basic shape, not on answer content.

Expected artifacts per capability (kept in lockstep with the
architecture doc):
    MOCK       -> FINAL_RESPONSE
    RAG        -> RETRIEVAL_RESULT + FINAL_RESPONSE
    OCR        -> OCR_TEXT + OCR_RESULT
    MULTIMODAL -> OCR_TEXT + VISION_RESULT + RETRIEVAL_RESULT + FINAL_RESPONSE
    (MULTIMODAL_TRACE is allowed but optional; other types are unexpected.)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # httpx is a direct worker dep
    import httpx
except ImportError:  # pragma: no cover — requirements.txt pins httpx
    httpx = None  # type: ignore

log = logging.getLogger("scripts.smoke_runner")


# ---------------------------------------------------------------------------
# Capability contract — the source of truth for artifact assertions.
# ---------------------------------------------------------------------------


# Artifacts a capability MUST emit for the smoke runner to mark it green.
REQUIRED_OUTPUTS: Dict[str, frozenset] = {
    "MOCK": frozenset({"FINAL_RESPONSE"}),
    "RAG": frozenset({"RETRIEVAL_RESULT", "FINAL_RESPONSE"}),
    "OCR": frozenset({"OCR_TEXT", "OCR_RESULT"}),
    "MULTIMODAL": frozenset({
        "OCR_TEXT", "VISION_RESULT", "RETRIEVAL_RESULT", "FINAL_RESPONSE",
    }),
}

# Artifacts a capability MAY emit without being flagged as "unexpected".
# Worker config gates MULTIMODAL_TRACE behind `multimodal_emit_trace`, so
# MULTIMODAL can legitimately produce 4 or 5 outputs. OCR/RAG/MOCK have
# no optional outputs today.
OPTIONAL_OUTPUTS: Dict[str, frozenset] = {
    "MOCK": frozenset(),
    "RAG": frozenset(),
    "OCR": frozenset(),
    "MULTIMODAL": frozenset({"MULTIMODAL_TRACE"}),
}


# Fixed list of capabilities the smoke runner knows how to exercise.
KNOWN_CAPABILITIES = ("MOCK", "RAG", "OCR", "MULTIMODAL")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SmokeCaseResult:
    """Per-capability outcome — the atom of the JSON report."""

    capability: str
    status: str                         # SUCCESS | FAIL | SKIP
    job_id: Optional[str] = None
    submit_http_status: Optional[int] = None
    final_job_status: Optional[str] = None
    duration_ms: float = 0.0
    output_types: List[str] = field(default_factory=list)
    missing_artifacts: List[str] = field(default_factory=list)
    unexpected_artifacts: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    failure_reason: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == "SUCCESS"


@dataclass
class SmokeReport:
    """Top-level report written to disk and echoed to stdout."""

    base_url: str
    started_at: str
    duration_ms: float
    cases: List[SmokeCaseResult]
    passed: int
    failed: int
    skipped: int

    def to_dict(self) -> dict:
        return {
            "baseUrl": self.base_url,
            "startedAt": self.started_at,
            "durationMs": self.duration_ms,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "total": len(self.cases),
            },
            "cases": [asdict(c) for c in self.cases],
        }


# ---------------------------------------------------------------------------
# Shape assertions — pure functions so unit tests can cover them.
# ---------------------------------------------------------------------------


class SmokeAssertionError(Exception):
    """Raised when a response's shape does not match expectations.

    Carries the failure_reason string that will land in the smoke report
    so the test and the reporter share one message.
    """


def assert_submission_shape(payload: Any, *, expected_capability: str) -> str:
    """Verify the POST /jobs response and return the job id.

    The submission response is the first place a schema drift would
    surface. A hard check here catches "core-api rolled back to an
    older DTO" before we waste polling time.
    """
    if not isinstance(payload, dict):
        raise SmokeAssertionError(
            f"Expected object from POST /jobs, got {type(payload).__name__}"
        )
    for field_name in ("jobId", "status", "capability"):
        if field_name not in payload:
            raise SmokeAssertionError(
                f"POST /jobs response missing required field '{field_name}' "
                f"(keys present: {sorted(payload.keys())})"
            )
    if payload["capability"] != expected_capability:
        raise SmokeAssertionError(
            f"POST /jobs echoed capability={payload['capability']!r}, "
            f"expected {expected_capability!r}"
        )
    if payload["status"] != "QUEUED":
        raise SmokeAssertionError(
            f"POST /jobs returned status={payload['status']!r}, expected 'QUEUED'. "
            "Core-api should always enqueue before returning."
        )
    return str(payload["jobId"])


def assert_final_status(
    job_view: Any, *, expected_capability: str
) -> None:
    """Verify the final GET /jobs/{id} shape after terminal state.

    Not concerned with what the job did — only that it reached SUCCEEDED
    and still carries the correct capability. FAILED jobs raise with the
    core-api errorCode/errorMessage embedded so the runner reports them.
    """
    if not isinstance(job_view, dict):
        raise SmokeAssertionError(
            f"Expected object from GET /jobs/{{id}}, got {type(job_view).__name__}"
        )
    for field_name in ("jobId", "status", "capability"):
        if field_name not in job_view:
            raise SmokeAssertionError(
                f"GET /jobs/{{id}} response missing required field '{field_name}'"
            )
    status = job_view["status"]
    if status != "SUCCEEDED":
        code = job_view.get("errorCode")
        msg = job_view.get("errorMessage")
        raise SmokeAssertionError(
            f"Job ended in {status!r} (errorCode={code!r}, errorMessage={msg!r})"
        )
    if job_view["capability"] != expected_capability:
        raise SmokeAssertionError(
            f"Final status echoed capability={job_view['capability']!r}, "
            f"expected {expected_capability!r}"
        )


def assert_result_outputs(
    result_payload: Any, *, capability: str
) -> Dict[str, List[str]]:
    """Verify GET /jobs/{id}/result → outputs match the contract.

    Returns `{"output_types": [...], "missing": [...], "unexpected": [...]}`
    so the caller can stash the breakdown in the report even when the
    assertion passes.

    Rules:
        * `outputs` must be a non-empty list for every capability.
        * Every required artifact type (per REQUIRED_OUTPUTS) must be
          present. Missing types raise SmokeAssertionError.
        * Types that are neither required nor in OPTIONAL_OUTPUTS are
          flagged as "unexpected" and also raise — this catches
          regressions where a capability accidentally starts emitting
          a new artifact type without updating the contract.
        * Duplicate types are allowed (belt-and-suspenders: the
          contract talks in sets, not sequences).
    """
    if not isinstance(result_payload, dict):
        raise SmokeAssertionError(
            f"Expected object from GET /jobs/{{id}}/result, got "
            f"{type(result_payload).__name__}"
        )
    outputs = result_payload.get("outputs")
    if not isinstance(outputs, list):
        raise SmokeAssertionError(
            f"GET /jobs/{{id}}/result has no 'outputs' list (got {outputs!r})"
        )
    if not outputs:
        raise SmokeAssertionError(
            f"{capability} completed with zero output artifacts — every "
            "capability must produce at least one output."
        )

    observed_types: List[str] = []
    for idx, artifact in enumerate(outputs):
        if not isinstance(artifact, dict):
            raise SmokeAssertionError(
                f"outputs[{idx}] is not an object (got {type(artifact).__name__})"
            )
        type_ = artifact.get("type")
        if not isinstance(type_, str) or not type_:
            raise SmokeAssertionError(
                f"outputs[{idx}] missing 'type' field (keys: {sorted(artifact.keys())})"
            )
        if "accessUrl" not in artifact:
            raise SmokeAssertionError(
                f"outputs[{idx}] ({type_}) missing 'accessUrl' — the runner "
                "cannot fetch the artifact body without it."
            )
        observed_types.append(type_)

    required = REQUIRED_OUTPUTS.get(capability, frozenset())
    optional = OPTIONAL_OUTPUTS.get(capability, frozenset())
    allowed = required | optional

    observed_set = set(observed_types)
    missing = sorted(required - observed_set)
    unexpected = sorted(observed_set - allowed)

    if missing:
        raise SmokeAssertionError(
            f"{capability} result is missing required artifact type(s) "
            f"{missing}. Present: {sorted(observed_types)}. "
            f"Required: {sorted(required)}."
        )
    if unexpected:
        raise SmokeAssertionError(
            f"{capability} result carries unexpected artifact type(s) "
            f"{unexpected}. Allowed: {sorted(allowed)}. "
            "Update REQUIRED_OUTPUTS / OPTIONAL_OUTPUTS if the contract has "
            "changed intentionally."
        )

    return {
        "output_types": observed_types,
        "missing": missing,
        "unexpected": unexpected,
    }


# ---------------------------------------------------------------------------
# Fixture helpers — lazy so `--only MOCK` still works without Pillow.
# ---------------------------------------------------------------------------


def _repo_root_relative(p: Path) -> Path:
    """Return `p` relative to `ai-worker/` so callers can invoke the runner
    from any working directory with a sensible default."""
    return (Path(__file__).resolve().parent.parent / p).resolve()


def _default_ocr_fixture_path() -> Path:
    return _repo_root_relative(Path("eval/datasets/samples/hello_world.png"))


def load_ocr_fixture_bytes(custom_path: Optional[Path] = None) -> tuple[bytes, str]:
    """Return `(bytes, filename)` for the PNG used by OCR / MULTIMODAL jobs.

    Resolution order:
      1. `custom_path` if provided (useful for tests / ad-hoc runs).
      2. The committed eval fixture at `eval/datasets/samples/hello_world.png`,
         which `python -m scripts.make_ocr_sample_fixtures` generates.
      3. An in-memory Pillow PNG rendered on the fly. This is the
         survival path when nobody has run the fixture generator yet —
         the bytes are real and Tesseract can read them, so OCR still
         gets exercised end-to-end. Raises a clear error if Pillow is
         not available.
    """
    if custom_path is not None:
        data = Path(custom_path).read_bytes()
        return data, Path(custom_path).name

    fixture = _default_ocr_fixture_path()
    if fixture.exists():
        return fixture.read_bytes(), fixture.name

    # Fallback: render a tiny PNG in-memory.
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError as ex:
        raise RuntimeError(
            "OCR/MULTIMODAL smoke cases need an image fixture. Either "
            "run `python -m scripts.make_ocr_sample_fixtures` to generate "
            "the committed PNGs, or install Pillow so the runner can "
            f"synthesize one on the fly. Underlying error: {ex}"
        ) from ex

    image = Image.new("L", (260, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "SMOKE TEST", fill=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "smoke-fallback.png"


# ---------------------------------------------------------------------------
# HTTP-level orchestration
# ---------------------------------------------------------------------------


class SmokeRunner:
    """Submits + polls a single capability case.

    Kept as a class so that the HTTP client, timeouts, and polling
    interval can be tweaked by the CLI without threading them through
    every helper signature.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
        client: Optional["httpx.Client"] = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required but not installed. `pip install httpx` "
                "or `pip install -r requirements.txt`."
            )
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=15.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------

    def run_mock(self) -> SmokeCaseResult:
        return self._run_text_case(
            capability="MOCK",
            text=f"smoke-mock-{int(time.time())}",
        )

    def run_rag(self) -> SmokeCaseResult:
        return self._run_text_case(
            capability="RAG",
            text="which anime is about an old fisherman feeding stray harbor cats",
        )

    def run_ocr(self, fixture_bytes: bytes, fixture_name: str) -> SmokeCaseResult:
        return self._run_file_case(
            capability="OCR",
            file_bytes=fixture_bytes,
            filename=fixture_name,
            content_type="image/png",
            text=None,
        )

    def run_multimodal(
        self, fixture_bytes: bytes, fixture_name: str
    ) -> SmokeCaseResult:
        return self._run_file_case(
            capability="MULTIMODAL",
            file_bytes=fixture_bytes,
            filename=fixture_name,
            content_type="image/png",
            text="what does this smoke-test image say?",
        )

    # ------------------------------------------------------------------

    def _run_text_case(self, *, capability: str, text: str) -> SmokeCaseResult:
        case = SmokeCaseResult(capability=capability, status="FAIL")
        started = time.monotonic()
        try:
            submit = self._client.post(
                "/api/v1/jobs",
                json={"capability": capability, "text": text},
            )
            case.submit_http_status = submit.status_code
            submit.raise_for_status()
            payload = submit.json()
            case.job_id = assert_submission_shape(
                payload, expected_capability=capability
            )

            final_view = self._poll_until_terminal(case.job_id)
            case.final_job_status = final_view.get("status")
            case.error_code = final_view.get("errorCode")
            case.error_message = final_view.get("errorMessage")
            assert_final_status(final_view, expected_capability=capability)

            result = self._fetch_result(case.job_id)
            breakdown = assert_result_outputs(result, capability=capability)
            case.output_types = breakdown["output_types"]
            case.missing_artifacts = breakdown["missing"]
            case.unexpected_artifacts = breakdown["unexpected"]
            case.status = "SUCCESS"
        except SmokeAssertionError as ex:
            case.failure_reason = str(ex)
        except httpx.HTTPError as ex:
            case.failure_reason = f"HTTP error: {ex}"
        except Exception as ex:  # defensive: never let a case crash the whole report
            case.failure_reason = f"unexpected {type(ex).__name__}: {ex}"
        finally:
            case.duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        return case

    def _run_file_case(
        self,
        *,
        capability: str,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        text: Optional[str],
    ) -> SmokeCaseResult:
        case = SmokeCaseResult(capability=capability, status="FAIL")
        started = time.monotonic()
        try:
            files = {"file": (filename, file_bytes, content_type)}
            data = {"capability": capability}
            if text is not None:
                data["text"] = text

            submit = self._client.post("/api/v1/jobs", files=files, data=data)
            case.submit_http_status = submit.status_code
            submit.raise_for_status()
            payload = submit.json()
            case.job_id = assert_submission_shape(
                payload, expected_capability=capability
            )

            final_view = self._poll_until_terminal(case.job_id)
            case.final_job_status = final_view.get("status")
            case.error_code = final_view.get("errorCode")
            case.error_message = final_view.get("errorMessage")
            assert_final_status(final_view, expected_capability=capability)

            result = self._fetch_result(case.job_id)
            breakdown = assert_result_outputs(result, capability=capability)
            case.output_types = breakdown["output_types"]
            case.missing_artifacts = breakdown["missing"]
            case.unexpected_artifacts = breakdown["unexpected"]
            case.status = "SUCCESS"
        except SmokeAssertionError as ex:
            case.failure_reason = str(ex)
        except httpx.HTTPError as ex:
            case.failure_reason = f"HTTP error: {ex}"
        except Exception as ex:
            case.failure_reason = f"unexpected {type(ex).__name__}: {ex}"
        finally:
            case.duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        return case

    # ------------------------------------------------------------------

    def _poll_until_terminal(self, job_id: str) -> dict:
        deadline = time.monotonic() + self._timeout
        last_status: Optional[str] = None
        while time.monotonic() < deadline:
            response = self._client.get(f"/api/v1/jobs/{job_id}")
            response.raise_for_status()
            body = response.json()
            status = body.get("status")
            if status != last_status:
                log.debug("job %s status=%s", job_id, status)
                last_status = status
            if status in ("SUCCEEDED", "FAILED"):
                return body
            time.sleep(self._poll_interval)
        raise SmokeAssertionError(
            f"Timed out after {self._timeout:.0f}s waiting for job {job_id} "
            f"to reach a terminal state (last status={last_status!r}). "
            "Is the worker running and consuming the Redis queue?"
        )

    def _fetch_result(self, job_id: str) -> dict:
        response = self._client.get(f"/api/v1/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def build_report(
    *,
    base_url: str,
    started_at: str,
    duration_ms: float,
    cases: List[SmokeCaseResult],
) -> SmokeReport:
    passed = sum(1 for c in cases if c.status == "SUCCESS")
    failed = sum(1 for c in cases if c.status == "FAIL")
    skipped = sum(1 for c in cases if c.status == "SKIP")
    return SmokeReport(
        base_url=base_url,
        started_at=started_at,
        duration_ms=duration_ms,
        cases=cases,
        passed=passed,
        failed=failed,
        skipped=skipped,
    )


def format_console_summary(report: SmokeReport) -> str:
    """Compact status table suitable for terminal output."""
    lines: List[str] = []
    lines.append(f"== smoke runner ({report.base_url}) ==")
    lines.append(
        f"started: {report.started_at}  "
        f"duration: {report.duration_ms:.0f} ms  "
        f"pass={report.passed} fail={report.failed} skip={report.skipped}"
    )
    for case in report.cases:
        marker = {
            "SUCCESS": "[OK]  ",
            "FAIL": "[FAIL]",
            "SKIP": "[SKIP]",
        }.get(case.status, "[????]")
        job = case.job_id or "-"
        dur = f"{case.duration_ms:.0f} ms"
        lines.append(
            f"{marker} {case.capability:<11} job={job} "
            f"final={case.final_job_status or '-'} ({dur})"
        )
        if case.output_types:
            lines.append(f"         outputs: {case.output_types}")
        if case.failure_reason:
            lines.append(f"         reason : {case.failure_reason}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_only_arg(value: Optional[str]) -> List[str]:
    if not value:
        return list(KNOWN_CAPABILITIES)
    raw = [v.strip().upper() for v in value.split(",") if v.strip()]
    unknown = [v for v in raw if v not in KNOWN_CAPABILITIES]
    if unknown:
        raise SystemExit(
            f"Unknown capability in --only: {unknown}. "
            f"Valid: {','.join(KNOWN_CAPABILITIES)}"
        )
    return raw


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.smoke_runner",
        description="End-to-end smoke runner for the async pipeline.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="core-api base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-job polling timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between GET /jobs/{id} polls (default: 0.5).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated capability subset (default: MOCK,RAG,OCR,MULTIMODAL).",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional explicit path to a PNG fixture for OCR/MULTIMODAL.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the JSON report to this path in addition to stdout.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging for the runner (keeps 3rd-party libs quiet).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verbose:
        log.setLevel(logging.DEBUG)

    selected = parse_only_arg(args.only)

    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_mono = time.monotonic()

    runner = SmokeRunner(
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
    )

    cases: List[SmokeCaseResult] = []
    needs_fixture = any(c in selected for c in ("OCR", "MULTIMODAL"))
    fixture_bytes = b""
    fixture_name = ""
    if needs_fixture:
        try:
            fixture_bytes, fixture_name = load_ocr_fixture_bytes(args.fixture)
        except Exception as ex:
            log.error("Fixture load failed: %s", ex)
            for cap in selected:
                if cap in ("OCR", "MULTIMODAL"):
                    cases.append(SmokeCaseResult(
                        capability=cap,
                        status="FAIL",
                        failure_reason=f"fixture load failed: {ex}",
                    ))

    try:
        if "MOCK" in selected:
            cases.append(runner.run_mock())
        if "RAG" in selected:
            cases.append(runner.run_rag())
        if "OCR" in selected and fixture_bytes:
            cases.append(runner.run_ocr(fixture_bytes, fixture_name))
        if "MULTIMODAL" in selected and fixture_bytes:
            cases.append(runner.run_multimodal(fixture_bytes, fixture_name))
    finally:
        runner.close()

    duration_ms = round((time.monotonic() - started_mono) * 1000.0, 2)
    report = build_report(
        base_url=args.base_url,
        started_at=started_wall,
        duration_ms=duration_ms,
        cases=cases,
    )

    sys.stdout.write(format_console_summary(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sys.stdout.write(f"JSON report written to {args.report}\n")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
