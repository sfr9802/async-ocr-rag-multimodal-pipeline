"""Multimodal eval harness tests.

All tests use stub capability/providers — no real OCR, VLM, FAISS, or
Postgres. Scenarios:

  1. Happy path: stub capability returns predictable output, metrics computed
  2. Row-level failure doesn't abort the full eval
  3. JSON + CSV reports are generated
  4. require_ocr_only filter skips non-OCR rows
  5. Missing image file is skipped (not fatal)
  6. Label precision/recall computed correctly
  7. Exact match and substring match logic
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pytest

from eval.harness.multimodal_eval import (
    MultimodalEvalRow,
    MultimodalEvalSummary,
    run_multimodal_eval,
    row_to_dict,
    summary_to_dict,
)
from eval.harness.io_utils import write_csv_report, write_json_report


# ---------------------------------------------------------------------------
# Stub capability + input builder.
# ---------------------------------------------------------------------------


@dataclass
class _StubArtifact:
    type: str
    content: bytes
    filename: str = ""
    content_type: str = ""


@dataclass
class _StubOutput:
    outputs: List[_StubArtifact]


class _StubCapability:
    """Returns a canned FINAL_RESPONSE + optional MULTIMODAL_TRACE."""

    def __init__(
        self,
        *,
        answer: str = "The answer is here.",
        trace: Optional[dict] = None,
        fail_on_image: Optional[str] = None,
    ) -> None:
        self._answer = answer
        self._trace = trace
        self._fail_on_image = fail_on_image
        self.run_calls: List[Any] = []

    def run(self, input: Any) -> _StubOutput:
        self.run_calls.append(input)

        # Simulate a failure for a specific image.
        if self._fail_on_image:
            for artifact in getattr(input, "inputs", []):
                fn = getattr(artifact, "filename", "")
                if fn and self._fail_on_image in fn:
                    raise RuntimeError(f"Simulated failure on {fn}")

        outputs = [
            _StubArtifact(
                type="FINAL_RESPONSE",
                content=self._answer.encode("utf-8"),
            )
        ]
        if self._trace:
            outputs.append(
                _StubArtifact(
                    type="MULTIMODAL_TRACE",
                    content=json.dumps(self._trace).encode("utf-8"),
                )
            )
        return _StubOutput(outputs=outputs)


@dataclass
class _StubInput:
    job_id: str
    capability: str
    attempt_no: int
    inputs: list


@dataclass
class _StubInputArtifact:
    artifact_id: str
    type: str
    content: bytes
    content_type: Optional[str] = None
    filename: Optional[str] = None


def _stub_input_builder(image_path, image_bytes, question, filename):
    artifacts = [
        _StubInputArtifact(
            artifact_id="eval-file-stub",
            type="INPUT_FILE",
            content=image_bytes,
            content_type="image/png",
            filename=filename,
        )
    ]
    if question:
        artifacts.append(
            _StubInputArtifact(
                artifact_id="eval-q-stub",
                type="INPUT_TEXT",
                content=question.encode("utf-8"),
                content_type="text/plain",
            )
        )
    return _StubInput(
        job_id="eval-test",
        capability="MULTIMODAL",
        attempt_no=1,
        inputs=artifacts,
    )


def _make_dataset_with_images(tmp_dir: Path) -> tuple[List[dict], Path]:
    """Create a small dataset with actual image files on disk."""
    img_dir = tmp_dir / "samples" / "multimodal"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Minimal PNG (1x1 white pixel).
    _write_minimal_png(img_dir / "test_ocr.png")
    _write_minimal_png(img_dir / "test_visual.png")
    _write_minimal_png(img_dir / "test_both.png")

    dataset = [
        {
            "image": "samples/multimodal/test_ocr.png",
            "question": "What text is in the image?",
            "expected_answer": "The answer is here",
            "expected_keywords": ["answer"],
            "expected_labels": [],
            "requires_ocr": True,
            "language": "eng",
            "notes": "OCR-only test row",
        },
        {
            "image": "samples/multimodal/test_visual.png",
            "question": "What shapes are in the image?",
            "expected_answer": None,
            "expected_keywords": ["answer"],
            "expected_labels": ["answer"],
            "requires_ocr": False,
            "language": "eng",
            "notes": "Visual-only test row",
        },
        {
            "image": "samples/multimodal/test_both.png",
            "question": "Describe the diagram",
            "expected_answer": None,
            "expected_keywords": ["answer", "here"],
            "expected_labels": ["answer"],
            "requires_ocr": True,
            "language": "eng",
            "notes": "OCR+visual test row",
        },
    ]
    return dataset, tmp_dir


def _write_minimal_png(path: Path) -> None:
    """Write a tiny valid PNG (Pillow not required)."""
    # 1x1 white pixel PNG
    import struct, zlib
    def _chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\xff\xff")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


# ---------------------------------------------------------------------------
# 1. Happy path — metrics computed correctly.
# ---------------------------------------------------------------------------


def test_happy_path_computes_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        dataset, base = _make_dataset_with_images(tmp_dir)

        cap = _StubCapability(answer="The answer is here.")
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=base,
        )

        assert summary.row_count == 3
        assert summary.evaluated_rows == 3
        assert summary.error_count == 0
        assert summary.skipped_rows == 0

        # Row 0: OCR-only, has expected_answer
        assert rows[0].exact_match is not None
        assert rows[0].substring_match is not None
        # "the answer is here" should substring-match "the answer is here."
        assert rows[0].substring_match == 1.0

        # All rows should have keyword_coverage computed
        for row in rows:
            assert row.keyword_coverage is not None
            assert row.keyword_coverage >= 0.0

        # Capability was called 3 times
        assert len(cap.run_calls) == 3


# ---------------------------------------------------------------------------
# 2. Row-level failure doesn't abort the eval.
# ---------------------------------------------------------------------------


def test_row_failure_does_not_abort_eval():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        dataset, base = _make_dataset_with_images(tmp_dir)

        # Will fail on the second image (test_visual.png).
        cap = _StubCapability(
            answer="The answer is here.",
            fail_on_image="test_visual",
        )

        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=base,
        )

        assert summary.row_count == 3
        assert summary.error_count == 1
        assert summary.evaluated_rows == 2

        # The failed row has an error message.
        failed = [r for r in rows if r.error is not None]
        assert len(failed) == 1
        assert "Simulated failure" in failed[0].error

        # Other rows succeeded.
        ok_rows = [r for r in rows if r.error is None]
        assert len(ok_rows) == 2
        assert all(r.answer is not None for r in ok_rows)


# ---------------------------------------------------------------------------
# 3. JSON + CSV reports are generated.
# ---------------------------------------------------------------------------


def test_reports_are_generated():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        dataset, base = _make_dataset_with_images(tmp_dir)

        cap = _StubCapability(answer="The answer is here.")
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=base,
        )

        json_path = tmp_dir / "report.json"
        csv_path = tmp_dir / "report.csv"

        write_json_report(
            json_path,
            summary=summary_to_dict(summary),
            rows=[row_to_dict(r) for r in rows],
        )
        write_csv_report(
            csv_path,
            [row_to_dict(r) for r in rows],
        )

        assert json_path.exists()
        assert csv_path.exists()

        # JSON is valid
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert "summary" in report
        assert "rows" in report
        assert len(report["rows"]) == 3

        # CSV has content
        csv_text = csv_path.read_text(encoding="utf-8")
        csv_lines = [l for l in csv_text.strip().splitlines() if l.strip()]
        assert len(csv_lines) >= 4  # header + 3 data rows


# ---------------------------------------------------------------------------
# 4. require_ocr_only filter skips non-OCR rows.
# ---------------------------------------------------------------------------


def test_require_ocr_only_filter():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        dataset, base = _make_dataset_with_images(tmp_dir)

        cap = _StubCapability(answer="The answer is here.")
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=base,
            require_ocr_only=True,
        )

        # Row 1 (visual-only) should be skipped
        assert summary.skipped_rows == 1
        assert summary.evaluated_rows == 2

        skipped = [r for r in rows if r.skipped_reason is not None]
        assert len(skipped) == 1
        assert "require" in skipped[0].skipped_reason.lower()


# ---------------------------------------------------------------------------
# 5. Missing image file is skipped.
# ---------------------------------------------------------------------------


def test_missing_image_is_skipped():
    dataset = [
        {
            "image": "samples/multimodal/nonexistent.png",
            "question": "What is this?",
            "expected_keywords": [],
        },
    ]

    cap = _StubCapability(answer="irrelevant")
    summary, rows = run_multimodal_eval(
        dataset,
        capability=cap,
        input_builder=_stub_input_builder,
        dataset_dir=Path(tempfile.gettempdir()),
        skip_missing_files=True,
    )

    assert summary.row_count == 1
    assert summary.skipped_rows == 1
    assert summary.evaluated_rows == 0
    assert rows[0].skipped_reason is not None
    assert "not found" in rows[0].skipped_reason


# ---------------------------------------------------------------------------
# 6. Label precision and recall.
# ---------------------------------------------------------------------------


def test_label_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        img_dir = tmp_dir / "img"
        img_dir.mkdir()
        _write_minimal_png(img_dir / "test.png")

        dataset = [
            {
                "image": "img/test.png",
                "question": "What labels?",
                "expected_labels": ["red", "blue", "green"],
                "expected_keywords": [],
            },
        ]

        # Answer mentions "red" and "blue" but not "green".
        cap = _StubCapability(answer="I see a Red circle and a Blue square.")
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=tmp_dir,
        )

        row = rows[0]
        assert row.label_recall is not None
        # 2 out of 3 labels found
        assert abs(row.label_recall - 2.0 / 3.0) < 0.01
        assert row.label_precision == 1.0  # all found labels are valid


# ---------------------------------------------------------------------------
# 7. Exact match and substring match.
# ---------------------------------------------------------------------------


def test_exact_and_substring_match():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        img_dir = tmp_dir / "img"
        img_dir.mkdir()
        _write_minimal_png(img_dir / "test.png")

        dataset = [
            {
                "image": "img/test.png",
                "question": "What is the total?",
                "expected_answer": "$257.50",
                "expected_keywords": ["257.50"],
            },
        ]

        # Answer contains the expected answer as a substring.
        cap = _StubCapability(answer="The total is $257.50 as shown.")
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=tmp_dir,
        )

        row = rows[0]
        assert row.substring_match == 1.0
        # Exact match should be 0 because the full answer != "$257.50"
        assert row.exact_match == 0.0


# ---------------------------------------------------------------------------
# 8. Trace-based stage latency extraction.
# ---------------------------------------------------------------------------


def test_trace_latency_extraction():
    trace = {
        "schemaVersion": "trace.v1",
        "capability": "MULTIMODAL",
        "stages": [
            {"stage": "classify", "durationMs": 0.5},
            {"stage": "ocr", "durationMs": 42.0, "provider": "tesseract-5.3.3"},
            {"stage": "vision", "durationMs": 3.2, "provider": "heuristic-vision-v1"},
            {"stage": "fusion", "durationMs": 0.3},
            {"stage": "retrieve", "durationMs": 11.0},
            {"stage": "generate", "durationMs": 2.0},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        img_dir = tmp_dir / "img"
        img_dir.mkdir()
        _write_minimal_png(img_dir / "test.png")

        dataset = [
            {
                "image": "img/test.png",
                "question": "test",
                "expected_keywords": [],
            },
        ]

        cap = _StubCapability(answer="answer", trace=trace)
        summary, rows = run_multimodal_eval(
            dataset,
            capability=cap,
            input_builder=_stub_input_builder,
            dataset_dir=tmp_dir,
        )

        row = rows[0]
        assert row.ocr_latency_ms == 42.0
        assert row.vision_latency_ms == 3.2
        assert row.rag_latency_ms == 13.0  # retrieve + generate
        assert row.vision_provider == "heuristic-vision-v1"


# ---------------------------------------------------------------------------
# 9. CLI argument parsing accepts multimodal subcommand.
# ---------------------------------------------------------------------------


def test_cli_multimodal_subcommand_parses():
    """Verify the argparser accepts multimodal with its flags."""
    from eval.run_eval import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "multimodal",
        "--dataset", "eval/datasets/multimodal_sample.jsonl",
        "--require-ocr-only",
        "--vision-provider", "heuristic",
        "--no-csv",
    ])

    assert args.mode == "multimodal"
    assert args.require_ocr_only is True
    assert args.vision_provider == "heuristic"
    assert args.no_csv is True
