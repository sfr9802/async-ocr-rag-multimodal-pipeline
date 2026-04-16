"""Unit coverage for the doctor CLI (scripts/doctor.py).

These tests exercise the pure check functions with synthetic inputs so
they can run fully offline — no Redis, no Postgres, no Tesseract. The
goal is to prove that:

  * a well-formed build.json yields a PASS for the runtime-model match
  * a stale / mismatched build.json yields a FAIL with a specific
    remediation hint
  * the capability roll-up interprets a constellation of check results
    the same way the worker registry does at startup
  * the `run_all_checks` dispatcher respects `only=[...]` and still
    appends the roll-up row
  * format_text_report / format_json_report serialize results without
    losing information
  * _redact_dsn hides passwords in both libpq and URL-shaped DSNs

These tests deliberately stay away from the infra-touching checks
(`check_redis`, `check_postgres`, `check_schemas`, `check_tesseract`)
because those would need running services; the doctor CLI already
catches their exceptions and returns a FAIL in that case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import doctor
from scripts.doctor import (
    CheckResult,
    FAIL,
    PASS,
    WARN,
    check_build_json,
    check_faiss_index,
    check_runtime_model_match,
    format_json_report,
    format_text_report,
    run_all_checks,
    summarize_capability_readiness,
)


# ---------------------------------------------------------------------------
# check_faiss_index
# ---------------------------------------------------------------------------


class TestCheckFaissIndex:
    def test_pass_when_both_files_present(self, tmp_path: Path):
        (tmp_path / "faiss.index").write_bytes(b"\x00" * 16)
        (tmp_path / "build.json").write_text("{}")
        result = check_faiss_index(tmp_path)
        assert result.status == PASS
        assert "FAISS index files present" in result.summary

    def test_fail_when_index_missing(self, tmp_path: Path):
        (tmp_path / "build.json").write_text("{}")
        result = check_faiss_index(tmp_path)
        assert result.status == FAIL
        assert "faiss.index" in result.details["missing"]
        assert "build_rag_index" in (result.remediation or "")

    def test_fail_when_both_missing(self, tmp_path: Path):
        result = check_faiss_index(tmp_path / "does-not-exist")
        assert result.status == FAIL
        assert set(result.details["missing"]) == {"faiss.index", "build.json"}


# ---------------------------------------------------------------------------
# check_build_json
# ---------------------------------------------------------------------------


def _write_build_json(dir_: Path, **overrides):
    payload = {
        "index_version": "v-test",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "chunk_count": 24,
    }
    payload.update(overrides)
    (dir_ / "build.json").write_text(json.dumps(payload))
    return payload


class TestCheckBuildJson:
    def test_pass_on_well_formed_file(self, tmp_path: Path):
        _write_build_json(tmp_path)
        result = check_build_json(tmp_path)
        assert result.status == PASS
        assert result.details["embedding_model"] == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_fail_when_file_missing(self, tmp_path: Path):
        result = check_build_json(tmp_path)
        assert result.status == FAIL
        assert "build.json" in result.summary

    def test_fail_on_invalid_json(self, tmp_path: Path):
        (tmp_path / "build.json").write_text("{not json")
        result = check_build_json(tmp_path)
        assert result.status == FAIL
        assert "not valid JSON" in result.summary

    def test_fail_on_missing_required_field(self, tmp_path: Path):
        # drop dimension — one of the four required keys
        (tmp_path / "build.json").write_text(json.dumps({
            "index_version": "v-1",
            "embedding_model": "m",
            "chunk_count": 1,
        }))
        result = check_build_json(tmp_path)
        assert result.status == FAIL
        assert "dimension" in result.details["missing"]


# ---------------------------------------------------------------------------
# check_runtime_model_match
# ---------------------------------------------------------------------------


class TestRuntimeModelMatch:
    def test_pass_when_model_and_dim_agree(self, tmp_path: Path):
        _write_build_json(tmp_path)
        result = check_runtime_model_match(
            tmp_path,
            "sentence-transformers/all-MiniLM-L6-v2",
            expected_dim=384,
        )
        assert result.status == PASS
        assert result.details["dimension"] == 384

    def test_fail_on_model_name_mismatch(self, tmp_path: Path):
        _write_build_json(tmp_path)
        result = check_runtime_model_match(
            tmp_path,
            "BAAI/bge-m3",  # different from what build.json records
            expected_dim=None,
        )
        assert result.status == FAIL
        assert result.details["index_model"] == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert result.details["configured_model"] == "BAAI/bge-m3"
        assert "rebuild" in (result.remediation or "").lower()

    def test_fail_on_dimension_mismatch(self, tmp_path: Path):
        _write_build_json(tmp_path, dimension=1024)
        result = check_runtime_model_match(
            tmp_path,
            "sentence-transformers/all-MiniLM-L6-v2",
            expected_dim=384,
        )
        assert result.status == FAIL
        assert result.details["configured_dim"] == 384
        assert result.details["index_dim"] == 1024

    def test_fail_when_build_json_missing(self, tmp_path: Path):
        result = check_runtime_model_match(
            tmp_path, "m", expected_dim=None
        )
        assert result.status == FAIL
        assert "missing" in result.summary.lower()


# ---------------------------------------------------------------------------
# summarize_capability_readiness
# ---------------------------------------------------------------------------


def _pass(name: str) -> CheckResult:
    return CheckResult(name=name, status=PASS, summary="ok")


def _fail(name: str) -> CheckResult:
    return CheckResult(name=name, status=FAIL, summary="down")


class TestCapabilitySummary:
    def test_all_ready_is_pass(self):
        results = [
            _pass("redis"),
            _pass("postgres"),
            _pass("schemas"),
            _pass("faiss_index"),
            _pass("build_json"),
            _pass("runtime_model_match"),
            _pass("tesseract"),
            _pass("image_deps"),
        ]
        rollup = summarize_capability_readiness(results)
        assert rollup.status == PASS
        assert rollup.details == {
            "MOCK": "ready",
            "RAG": "ready",
            "OCR": "ready",
            "MULTIMODAL": "ready",
        }

    def test_broken_postgres_blocks_rag_and_multimodal(self):
        results = [
            _pass("redis"),
            _fail("postgres"),
            _fail("schemas"),
            _pass("faiss_index"),
            _pass("build_json"),
            _pass("runtime_model_match"),
            _pass("tesseract"),
            _pass("image_deps"),
        ]
        rollup = summarize_capability_readiness(results)
        assert rollup.status == WARN
        assert rollup.details["MOCK"] == "ready"
        assert rollup.details["RAG"] == "blocked"
        assert rollup.details["OCR"] == "ready"
        assert rollup.details["MULTIMODAL"] == "blocked"

    def test_broken_tesseract_blocks_ocr_and_multimodal(self):
        results = [
            _pass("redis"),
            _pass("postgres"),
            _pass("schemas"),
            _pass("faiss_index"),
            _pass("build_json"),
            _pass("runtime_model_match"),
            _fail("tesseract"),
            _pass("image_deps"),
        ]
        rollup = summarize_capability_readiness(results)
        assert rollup.details["MOCK"] == "ready"
        assert rollup.details["RAG"] == "ready"
        assert rollup.details["OCR"] == "blocked"
        assert rollup.details["MULTIMODAL"] == "blocked"

    def test_mock_is_always_ready(self):
        rollup = summarize_capability_readiness([_fail("everything")])
        assert rollup.details["MOCK"] == "ready"


# ---------------------------------------------------------------------------
# run_all_checks dispatcher
# ---------------------------------------------------------------------------


class _StubSettings:
    """Just enough attributes for run_all_checks to build its inputs.

    The individual checks are free to fail fast on bad data — we only
    need the dispatcher to find them and attach them to the report.
    """

    def __init__(self, tmp_path: Path):
        self.redis_url = "redis://127.0.0.1:1"  # unreachable on purpose
        self.rag_db_dsn = "host=127.0.0.1 port=1 dbname=x user=x password=x"
        self.rag_index_dir = str(tmp_path)
        self.rag_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.ocr_tesseract_cmd = None
        self.ocr_languages = "eng"


class TestDispatcher:
    def test_only_subset_runs_requested_checks_plus_rollup(self, tmp_path: Path):
        _write_build_json(tmp_path)
        (tmp_path / "faiss.index").write_bytes(b"\x00" * 16)

        results = run_all_checks(
            _StubSettings(tmp_path),
            only=["build_json", "faiss_index"],
        )
        names = [r.name for r in results]
        assert "build_json" in names
        assert "faiss_index" in names
        assert "capability_readiness" in names  # always present
        # checks we did NOT ask for should not be in the list
        assert "redis" not in names
        assert "postgres" not in names

    def test_run_all_checks_default_runs_every_known_check(self, tmp_path: Path):
        _write_build_json(tmp_path)
        (tmp_path / "faiss.index").write_bytes(b"\x00" * 16)

        results = run_all_checks(_StubSettings(tmp_path))
        names = [r.name for r in results]
        for expected in doctor.ALL_CHECK_NAMES:
            assert expected in names, f"missing check {expected}"
        assert names[-1] == "capability_readiness"


# ---------------------------------------------------------------------------
# format helpers
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_text_report_includes_every_row(self):
        results = [
            CheckResult(name="redis", status=PASS, summary="reachable"),
            CheckResult(
                name="postgres",
                status=FAIL,
                summary="down",
                details={"error": "connection refused"},
                remediation="start postgres",
            ),
        ]
        text = format_text_report(results)
        assert "redis" in text
        assert "postgres" in text
        assert "connection refused" in text
        assert "start postgres" in text
        assert "[PASS]" in text
        assert "[FAIL]" in text

    def test_json_report_has_overall_fail_when_any_check_fails(self):
        results = [
            CheckResult(name="redis", status=PASS, summary="ok"),
            CheckResult(name="postgres", status=FAIL, summary="down"),
        ]
        body = json.loads(format_json_report(results))
        assert body["overall"] == FAIL
        assert len(body["checks"]) == 2
        assert body["checks"][0]["name"] == "redis"
        assert body["checks"][1]["status"] == FAIL

    def test_json_report_pass_overall_when_everything_passes(self):
        results = [
            CheckResult(name="redis", status=PASS, summary="ok"),
            CheckResult(name="postgres", status=PASS, summary="ok"),
        ]
        body = json.loads(format_json_report(results))
        assert body["overall"] == PASS


# ---------------------------------------------------------------------------
# dsn redaction
# ---------------------------------------------------------------------------


class TestRedactDsn:
    def test_redacts_libpq_password_token(self):
        from scripts.doctor import _redact_dsn

        redacted = _redact_dsn(
            "host=localhost port=5432 dbname=aipipeline "
            "user=aipipeline password=aipipeline_pw"
        )
        assert "aipipeline_pw" not in redacted
        assert "password=****" in redacted
        # Non-password tokens survive
        assert "host=localhost" in redacted
        assert "user=aipipeline" in redacted

    def test_redacts_url_shaped_dsn(self):
        from scripts.doctor import _redact_dsn

        redacted = _redact_dsn(
            "postgresql://aipipeline:aipipeline_pw@localhost:5432/aipipeline"
        )
        assert "aipipeline_pw" not in redacted
        assert "****" in redacted
        assert "aipipeline" in redacted  # user survived
