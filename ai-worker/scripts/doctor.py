"""Local readiness checker for the async pipeline worker.

This is the developer/operator "doctor" CLI — a fast, local-first smoke
of every runtime prerequisite the worker needs before it can serve real
jobs. It never talks to core-api and never dispatches a job: the goal is
"tell me why my worker would refuse to register capability X" and
"confirm my infrastructure is reachable" in one place, with concrete
remediation hints.

Usage (from ai-worker/)::

    python -m scripts.doctor
    python -m scripts.doctor --json          # machine-readable output
    python -m scripts.doctor --only rag,ocr  # run a subset

Exit codes:
    0  every check PASSed (or was skipped with a reason)
    1  at least one check FAILed

Design notes:
    * Checks are pure functions that return a `CheckResult`. The CLI
      runs them, formats them, and computes an exit code. Tests call
      the functions directly with stubbed inputs — no subprocess, no
      network.
    * Every check catches its own exceptions and turns them into a
      FAIL result with a remediation hint. A misbehaving check must
      not take down the whole report.
    * This tool is NEVER authoritative for production readiness. It
      catches the 90% of local-dev foot-guns (Redis not running,
      tesseract not on PATH, build.json stale after an index rebuild,
      embedding dim mismatch, etc.). A green doctor run does not prove
      worker quality; it proves worker connectivity and contract
      integrity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

log = logging.getLogger("scripts.doctor")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"


@dataclass
class CheckResult:
    """Outcome of a single readiness check.

    Attributes:
        name:        Stable short identifier (used in JSON output keys).
        status:      One of PASS / FAIL / SKIP / WARN.
        summary:     One-line human-readable summary.
        details:     Optional dict of structured context (e.g. model names,
                     counts, dimensions). Serialized verbatim into the JSON
                     report.
        remediation: Hint on how to fix a FAIL / WARN. Omitted for PASS.
        duration_ms: Wall-clock time of the check, useful for spotting
                     cold-start latencies (e.g. sentence-transformers
                     model load).
    """

    name: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)
    remediation: Optional[str] = None
    duration_ms: float = 0.0

    def is_failure(self) -> bool:
        return self.status == FAIL


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_redis(redis_url: str) -> CheckResult:
    """Probe Redis with a single PING. No queue inspection — just reachability."""
    started = time.monotonic()
    try:
        import redis  # type: ignore
    except ImportError as ex:
        return CheckResult(
            name="redis",
            status=FAIL,
            summary="redis-py is not installed in the worker environment.",
            remediation="pip install -r requirements.txt",
            duration_ms=_elapsed_ms(started),
        )

    try:
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=2.0)
        if not client.ping():
            raise RuntimeError("PING returned a falsy value")
    except Exception as ex:
        return CheckResult(
            name="redis",
            status=FAIL,
            summary=f"Redis PING failed at {redis_url}",
            details={"error": f"{type(ex).__name__}: {ex}"},
            remediation=(
                "Start redis (`docker compose up -d redis`) and check "
                "AIPIPELINE_WORKER_REDIS_URL."
            ),
            duration_ms=_elapsed_ms(started),
        )

    return CheckResult(
        name="redis",
        status=PASS,
        summary=f"Redis reachable at {redis_url}",
        duration_ms=_elapsed_ms(started),
    )


def check_postgres(dsn: str) -> CheckResult:
    """Open a psycopg2 connection and run `SELECT 1`."""
    started = time.monotonic()
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return CheckResult(
            name="postgres",
            status=FAIL,
            summary="psycopg2 is not installed in the worker environment.",
            remediation="pip install -r requirements.txt",
            duration_ms=_elapsed_ms(started),
        )

    try:
        conn = psycopg2.connect(dsn)
    except Exception as ex:
        return CheckResult(
            name="postgres",
            status=FAIL,
            summary="PostgreSQL is not reachable.",
            details={"error": f"{type(ex).__name__}: {ex}", "dsn": _redact_dsn(dsn)},
            remediation=(
                "Check that PostgreSQL is running and AIPIPELINE_WORKER_RAG_DB_DSN "
                "(or the core-api spring.datasource.url) points at the right host "
                "and port. See docs/local-run.md for Mode A / Mode B bootstrap."
            ),
            duration_ms=_elapsed_ms(started),
        )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    finally:
        conn.close()
    return CheckResult(
        name="postgres",
        status=PASS,
        summary="PostgreSQL reachable.",
        details={"dsn": _redact_dsn(dsn)},
        duration_ms=_elapsed_ms(started),
    )


def check_schemas(dsn: str) -> CheckResult:
    """Verify that the schemas / tables the worker reads from are present.

    We check three things, in order:
        - aipipeline.job            (owned by core-api Flyway V1)
        - aipipeline.artifact       (owned by core-api Flyway V1)
        - ragmeta schema + ragmeta.chunks (owned by Flyway V2)

    A missing aipipeline table points at "core-api hasn't migrated yet".
    A missing ragmeta schema means Flyway V2 didn't run — usually because
    core-api was started before the migration was added.
    """
    started = time.monotonic()
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return CheckResult(
            name="schemas",
            status=FAIL,
            summary="psycopg2 is not installed in the worker environment.",
            remediation="pip install -r requirements.txt",
            duration_ms=_elapsed_ms(started),
        )

    try:
        conn = psycopg2.connect(dsn)
    except Exception as ex:
        return CheckResult(
            name="schemas",
            status=FAIL,
            summary="Cannot check schemas — PostgreSQL is unreachable.",
            details={"error": f"{type(ex).__name__}: {ex}"},
            remediation="Fix the 'postgres' check first.",
            duration_ms=_elapsed_ms(started),
        )

    missing: List[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM information_schema.tables
                 WHERE table_schema = current_schema()
                   AND table_name IN ('job', 'artifact')
                """
            )
            core_tables = {row[0] for row in cur.fetchall() or []}
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                 WHERE table_schema = 'public'
                   AND table_name IN ('job', 'artifact')
                """
            )
            for row in cur.fetchall() or []:
                core_tables.add(row[0])

            for needed in ("job", "artifact"):
                if needed not in core_tables:
                    missing.append(f"public.{needed}")

            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ragmeta'"
            )
            if cur.fetchone() is None:
                missing.append("ragmeta (schema)")
            else:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                     WHERE table_schema = 'ragmeta'
                       AND table_name IN ('documents', 'chunks', 'index_builds')
                    """
                )
                have = {row[0] for row in cur.fetchall() or []}
                for needed in ("documents", "chunks", "index_builds"):
                    if needed not in have:
                        missing.append(f"ragmeta.{needed}")
    finally:
        conn.close()

    if missing:
        return CheckResult(
            name="schemas",
            status=FAIL,
            summary=f"{len(missing)} expected table(s) / schema(s) missing.",
            details={"missing": missing},
            remediation=(
                "Restart core-api so Flyway runs (V1 creates the aipipeline "
                "schema, V2 creates ragmeta). Check core-api logs for a "
                "Flyway failure."
            ),
            duration_ms=_elapsed_ms(started),
        )
    return CheckResult(
        name="schemas",
        status=PASS,
        summary="aipipeline + ragmeta schemas / tables present.",
        duration_ms=_elapsed_ms(started),
    )


def check_faiss_index(index_dir: Path) -> CheckResult:
    """Verify both the FAISS index file and build.json sidecar exist.

    Only checks presence — file-content validation lives in
    check_runtime_model_match below.
    """
    started = time.monotonic()
    index_dir = Path(index_dir)
    index_file = index_dir / "faiss.index"
    build_file = index_dir / "build.json"

    missing = [p.name for p in (index_file, build_file) if not p.exists()]
    if missing:
        return CheckResult(
            name="faiss_index",
            status=FAIL,
            summary=f"FAISS index missing under {index_dir}",
            details={"missing": missing, "index_dir": str(index_dir)},
            remediation=(
                "Build the RAG index: "
                "`cd ai-worker && python -m scripts.build_rag_index --fixture`"
            ),
            duration_ms=_elapsed_ms(started),
        )
    return CheckResult(
        name="faiss_index",
        status=PASS,
        summary=f"FAISS index files present in {index_dir}",
        details={"index_dir": str(index_dir)},
        duration_ms=_elapsed_ms(started),
    )


def check_build_json(index_dir: Path) -> CheckResult:
    """Parse build.json and surface its contents.

    A well-formed build.json is required before the runtime-model-match
    check can run. We return FAIL on "can't parse" / "missing required
    fields" and PASS on "looks like a BuildInfo record".
    """
    started = time.monotonic()
    build_file = Path(index_dir) / "build.json"
    if not build_file.exists():
        return CheckResult(
            name="build_json",
            status=FAIL,
            summary=f"build.json not found at {build_file}",
            remediation=(
                "Build the RAG index: "
                "`cd ai-worker && python -m scripts.build_rag_index --fixture`"
            ),
            duration_ms=_elapsed_ms(started),
        )

    try:
        raw = json.loads(build_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        return CheckResult(
            name="build_json",
            status=FAIL,
            summary=f"build.json is not valid JSON: {ex.msg}",
            details={"path": str(build_file)},
            remediation=(
                "Delete build.json + faiss.index and rebuild with "
                "`python -m scripts.build_rag_index --fixture`."
            ),
            duration_ms=_elapsed_ms(started),
        )

    required = ("index_version", "embedding_model", "dimension", "chunk_count")
    missing = [k for k in required if k not in raw]
    if missing:
        return CheckResult(
            name="build_json",
            status=FAIL,
            summary="build.json is missing required fields.",
            details={"missing": missing, "path": str(build_file)},
            remediation=(
                "Rebuild the index with "
                "`python -m scripts.build_rag_index --fixture`."
            ),
            duration_ms=_elapsed_ms(started),
        )

    return CheckResult(
        name="build_json",
        status=PASS,
        summary=f"build.json parseable (version={raw.get('index_version')!s}).",
        details={
            "index_version": raw.get("index_version"),
            "embedding_model": raw.get("embedding_model"),
            "dimension": raw.get("dimension"),
            "chunk_count": raw.get("chunk_count"),
        },
        duration_ms=_elapsed_ms(started),
    )


def check_runtime_model_match(
    index_dir: Path,
    configured_model: str,
    *,
    expected_dim: Optional[int] = None,
) -> CheckResult:
    """Compare the runtime RAG embedding model against build.json.

    This is the same strict check `Retriever.ensure_ready` enforces at
    worker startup. Pulling it out into its own doctor check lets
    operators diagnose a mismatch without having to read the worker's
    startup log.

    If `expected_dim` is supplied the check also verifies dimension
    equality. Leaving it None skips the dim comparison (useful for
    tests where we only want to assert the model-name logic).
    """
    started = time.monotonic()
    build_file = Path(index_dir) / "build.json"
    if not build_file.exists():
        return CheckResult(
            name="runtime_model_match",
            status=FAIL,
            summary="Cannot compare models — build.json is missing.",
            remediation=(
                "Run `python -m scripts.build_rag_index --fixture` first."
            ),
            duration_ms=_elapsed_ms(started),
        )

    try:
        raw = json.loads(build_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        return CheckResult(
            name="runtime_model_match",
            status=FAIL,
            summary="build.json is unreadable.",
            details={"error": ex.msg},
            remediation=(
                "Rebuild: `python -m scripts.build_rag_index --fixture`."
            ),
            duration_ms=_elapsed_ms(started),
        )

    index_model = raw.get("embedding_model")
    index_dim = raw.get("dimension")

    if index_model != configured_model:
        return CheckResult(
            name="runtime_model_match",
            status=FAIL,
            summary="Embedding MODEL mismatch between worker config and index.",
            details={
                "configured_model": configured_model,
                "index_model": index_model,
            },
            remediation=(
                "Rebuild the index with the configured model "
                "(`python -m scripts.build_rag_index --fixture`) OR set "
                "AIPIPELINE_WORKER_RAG_EMBEDDING_MODEL back to the indexed "
                "model, then restart the worker."
            ),
            duration_ms=_elapsed_ms(started),
        )

    if expected_dim is not None and int(expected_dim) != int(index_dim):
        return CheckResult(
            name="runtime_model_match",
            status=FAIL,
            summary="Embedding DIMENSION mismatch between worker config and index.",
            details={
                "configured_dim": int(expected_dim),
                "index_dim": int(index_dim),
                "model": configured_model,
            },
            remediation=(
                "Dimensions agree only when the same model built the index. "
                "Rebuild with `python -m scripts.build_rag_index --fixture` "
                "and restart the worker."
            ),
            duration_ms=_elapsed_ms(started),
        )

    return CheckResult(
        name="runtime_model_match",
        status=PASS,
        summary=f"Runtime model matches index ({configured_model}).",
        details={"model": configured_model, "dimension": int(index_dim)},
        duration_ms=_elapsed_ms(started),
    )


def check_tesseract(tesseract_cmd: Optional[str], languages: str) -> CheckResult:
    """Verify the Tesseract binary is on PATH (or at the configured path)
    AND that every requested language pack is installed.

    Mirrors the TesseractOcrProvider.ensure_ready flow — a worker that
    passes this check will successfully register the OCR capability.
    """
    started = time.monotonic()
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return CheckResult(
            name="tesseract",
            status=FAIL,
            summary="pytesseract is not installed.",
            remediation="pip install -r requirements.txt",
            duration_ms=_elapsed_ms(started),
        )

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as ex:
        return CheckResult(
            name="tesseract",
            status=FAIL,
            summary="Tesseract binary not found.",
            details={"error": f"{type(ex).__name__}: {ex}"},
            remediation=(
                "Install Tesseract (https://tesseract-ocr.github.io/) and add "
                "it to PATH, or set AIPIPELINE_WORKER_OCR_TESSERACT_CMD."
            ),
            duration_ms=_elapsed_ms(started),
        )

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()

    requested = [lang for lang in languages.split("+") if lang]
    missing = [lang for lang in requested if available and lang not in available]
    if missing:
        return CheckResult(
            name="tesseract",
            status=FAIL,
            summary=f"Tesseract language pack(s) missing: {missing}",
            details={"available": sorted(available), "requested": requested},
            remediation=(
                "Install the missing traineddata files, or change "
                f"AIPIPELINE_WORKER_OCR_LANGUAGES to a subset of the "
                f"installed languages ({sorted(available) or '<none>'})."
            ),
            duration_ms=_elapsed_ms(started),
        )

    return CheckResult(
        name="tesseract",
        status=PASS,
        summary=f"Tesseract {version} available (languages: {requested}).",
        details={"version": version, "languages": requested},
        duration_ms=_elapsed_ms(started),
    )


def check_image_deps() -> CheckResult:
    """Confirm Pillow + PyMuPDF import. These are the two libraries every
    image / PDF path in the worker touches."""
    started = time.monotonic()
    missing: List[str] = []
    try:
        import PIL  # type: ignore # noqa: F401
    except ImportError:
        missing.append("Pillow")
    try:
        import fitz  # type: ignore # noqa: F401
    except ImportError:
        missing.append("PyMuPDF (fitz)")

    if missing:
        return CheckResult(
            name="image_deps",
            status=FAIL,
            summary=f"Image/PDF dependencies missing: {missing}",
            remediation="pip install -r requirements.txt",
            duration_ms=_elapsed_ms(started),
        )
    return CheckResult(
        name="image_deps",
        status=PASS,
        summary="Pillow + PyMuPDF importable.",
        duration_ms=_elapsed_ms(started),
    )


def summarize_capability_readiness(results: Iterable[CheckResult]) -> CheckResult:
    """Roll the per-subsystem checks up into a single capability view.

    The logic matches the worker registry:
      MOCK is always ready.
      RAG is ready iff postgres + schemas + faiss_index + build_json +
          runtime_model_match are all PASS.
      OCR is ready iff tesseract + image_deps are all PASS.
      MULTIMODAL is ready iff BOTH RAG and OCR are ready.

    We emit this as a single check rather than mutating upstream
    statuses so tests can still assert on each sub-check in isolation.
    """
    by_name = {r.name: r for r in results}

    def ok(name: str) -> bool:
        r = by_name.get(name)
        return r is not None and r.status == PASS

    mock_ready = True  # MOCK has no infra dependency
    rag_ready = all(ok(n) for n in (
        "postgres", "schemas", "faiss_index", "build_json", "runtime_model_match"
    ))
    ocr_ready = all(ok(n) for n in ("tesseract", "image_deps"))
    mm_ready = rag_ready and ocr_ready

    details = {
        "MOCK": "ready" if mock_ready else "blocked",
        "RAG": "ready" if rag_ready else "blocked",
        "OCR": "ready" if ocr_ready else "blocked",
        "MULTIMODAL": "ready" if mm_ready else "blocked",
    }

    # If everything downstream was ready, the roll-up is PASS; if any
    # downstream failed, we flag it as WARN (not FAIL) because the
    # underlying FAIL already drove the exit code — we don't want to
    # double-count it.
    status = PASS if (mock_ready and rag_ready and ocr_ready and mm_ready) else WARN
    summary = ", ".join(f"{k}:{v}" for k, v in details.items())
    remediation = None
    if status == WARN:
        remediation = (
            "See individual check failures above. MOCK always stays ready; "
            "fix the downstream checks named in the FAIL rows and rerun."
        )
    return CheckResult(
        name="capability_readiness",
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Names of the checks the doctor knows about. `--only` accepts any subset.
ALL_CHECK_NAMES = (
    "redis",
    "postgres",
    "schemas",
    "faiss_index",
    "build_json",
    "runtime_model_match",
    "tesseract",
    "image_deps",
)


def run_all_checks(settings, only: Optional[List[str]] = None) -> List[CheckResult]:
    """Execute every check (or just `only`) and return their results.

    Kept module-level so tests can call it with a stub settings object.
    """
    selected = set(only or ALL_CHECK_NAMES)

    results: List[CheckResult] = []

    if "redis" in selected:
        results.append(check_redis(settings.redis_url))
    if "postgres" in selected:
        results.append(check_postgres(settings.rag_db_dsn))
    if "schemas" in selected:
        results.append(check_schemas(settings.rag_db_dsn))
    if "faiss_index" in selected:
        results.append(check_faiss_index(Path(settings.rag_index_dir)))
    if "build_json" in selected:
        results.append(check_build_json(Path(settings.rag_index_dir)))
    if "runtime_model_match" in selected:
        results.append(
            check_runtime_model_match(
                Path(settings.rag_index_dir),
                settings.rag_embedding_model,
            )
        )
    if "tesseract" in selected:
        results.append(
            check_tesseract(settings.ocr_tesseract_cmd, settings.ocr_languages)
        )
    if "image_deps" in selected:
        results.append(check_image_deps())

    # Always append the capability roll-up at the end so it reflects the
    # full set of checks actually run this invocation.
    results.append(summarize_capability_readiness(results))
    return results


def format_text_report(results: Iterable[CheckResult]) -> str:
    """Pretty-print a PASS/FAIL table with remediation hints indented below."""
    out_lines: List[str] = []
    out_lines.append("== ai-worker doctor ==")
    for r in results:
        marker = {
            PASS: "[PASS]",
            FAIL: "[FAIL]",
            SKIP: "[SKIP]",
            WARN: "[WARN]",
        }.get(r.status, "[????]")
        duration = f" ({r.duration_ms:.0f} ms)" if r.duration_ms else ""
        out_lines.append(f"{marker} {r.name:<22} {r.summary}{duration}")
        if r.details:
            for k, v in r.details.items():
                out_lines.append(f"          {k}: {v}")
        if r.remediation:
            out_lines.append(f"    -> {r.remediation}")
    return "\n".join(out_lines) + "\n"


def format_json_report(results: Iterable[CheckResult]) -> str:
    results_list = list(results)
    overall = (
        FAIL if any(r.is_failure() for r in results_list)
        else PASS
    )
    body = {
        "overall": overall,
        "checks": [asdict(r) for r in results_list],
    }
    return json.dumps(body, indent=2, ensure_ascii=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.doctor",
        description="Local-first readiness check for the ai-worker process.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated list of checks to run (default: all). "
            f"Valid values: {','.join(ALL_CHECK_NAMES)}"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Late import so `python -m scripts.doctor --help` works even when
    # pydantic-settings trips over a missing env file.
    from app.core.config import get_settings

    settings = get_settings()
    only = None
    if args.only:
        only = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [n for n in only if n not in ALL_CHECK_NAMES]
        if unknown:
            parser.error(
                f"Unknown check names: {unknown}. "
                f"Valid: {','.join(ALL_CHECK_NAMES)}"
            )

    results = run_all_checks(settings, only=only)

    if args.json:
        sys.stdout.write(format_json_report(results))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text_report(results))

    return 1 if any(r.is_failure() for r in results) else 0


# ---------------------------------------------------------------------------
# small helpers (kept private)
# ---------------------------------------------------------------------------


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 2)


def _redact_dsn(dsn: str) -> str:
    """Return a DSN with the password masked.

    Works for both libpq key=value strings ("host=... password=...")
    and URL-shaped DSNs ("postgresql://user:pw@host/db").
    """
    if "://" in dsn and "@" in dsn:
        scheme, rest = dsn.split("://", 1)
        creds, tail = rest.split("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:****@{tail}"
        return dsn
    parts = []
    for token in dsn.split():
        if token.lower().startswith("password="):
            parts.append("password=****")
        else:
            parts.append(token)
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
