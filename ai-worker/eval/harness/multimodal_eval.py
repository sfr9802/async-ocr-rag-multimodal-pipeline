"""Multimodal evaluation harness.

Takes an already-constructed MultimodalCapability (or any callable that
accepts a CapabilityInput and returns a CapabilityOutput) and runs it
over a JSONL dataset of multimodal rows.

Design follows the same pattern as rag_eval.py and ocr_eval.py:
  - Pluggable capability / provider bundle (tests inject fakes)
  - Row-level error isolation (one bad row doesn't abort the run)
  - JSON + CSV report output via shared io_utils

Per-row metrics:
  - exact_match       : 1.0 if expected_answer == answer (normalized), else 0.0
  - substring_match   : 1.0 if expected_answer appears in answer, else 0.0
  - keyword_coverage  : fraction of expected_keywords present (reuses metrics.py)
  - label_precision   : fraction of expected_labels actually found in answer
  - label_recall      : fraction of expected_labels found / total expected
  - latency_ms        : total wall-clock for the capability run
  - ocr_latency_ms    : OCR stage latency (from MULTIMODAL_TRACE if available)
  - vision_latency_ms : vision stage latency
  - rag_latency_ms    : retrieval + generation latency

Summary aggregations:
  - mean / p50 / max for each latency bucket
  - mean exact_match, substring_match, keyword_coverage
  - mean label_precision, label_recall
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from eval.harness.metrics import keyword_coverage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural protocols — the harness doesn't import app.capabilities.
# ---------------------------------------------------------------------------


class _CapabilityLike(Protocol):
    def run(self, input: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Row + summary dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class MultimodalEvalRow:
    image: str
    question: str
    expected_answer: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)
    expected_labels: List[str] = field(default_factory=list)
    requires_ocr: Optional[bool] = None
    language: Optional[str] = None
    notes: Optional[str] = None

    # computed
    answer: Optional[str] = None
    exact_match: Optional[float] = None
    substring_match: Optional[float] = None
    keyword_coverage: Optional[float] = None
    label_precision: Optional[float] = None
    label_recall: Optional[float] = None
    latency_ms: float = 0.0
    ocr_latency_ms: float = 0.0
    vision_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    vision_provider: Optional[str] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class MultimodalEvalSummary:
    dataset_path: str
    row_count: int
    evaluated_rows: int
    skipped_rows: int
    error_count: int
    mean_exact_match: Optional[float]
    mean_substring_match: Optional[float]
    mean_keyword_coverage: Optional[float]
    mean_label_precision: Optional[float]
    mean_label_recall: Optional[float]
    mean_latency_ms: float
    p50_latency_ms: float
    max_latency_ms: float
    mean_ocr_latency_ms: float
    mean_vision_latency_ms: float
    mean_rag_latency_ms: float
    vision_provider: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def run_multimodal_eval(
    dataset: List[Mapping[str, Any]],
    *,
    capability: _CapabilityLike,
    input_builder: Callable[[str, bytes, str, Optional[str]], Any],
    dataset_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    skip_missing_files: bool = True,
    answer_excerpt_chars: int = 600,
    require_ocr_only: bool = False,
) -> tuple[MultimodalEvalSummary, List[MultimodalEvalRow]]:
    """Run multimodal eval over `dataset`, returning (summary, rows).

    Parameters:
      capability    : anything with a `.run(input)` method that returns
                      a CapabilityOutput-shaped object.
      input_builder : callable(image_path, image_bytes, question, filename)
                      -> CapabilityInput. Decouples the harness from
                      the app.capabilities.base import so tests can
                      inject their own shape.
      require_ocr_only : when True, skip rows where requires_ocr is not True.
    """
    started_at = _now_iso()
    run_start = time.perf_counter()

    base_dir = Path(dataset_dir or ".").resolve()
    rows: List[MultimodalEvalRow] = []

    for idx, raw in enumerate(dataset, start=1):
        image_field = _require_str(raw, "image", row_index=idx)
        question = _require_str(raw, "question", row_index=idx)

        row = MultimodalEvalRow(
            image=image_field,
            question=question,
            expected_answer=raw.get("expected_answer"),
            expected_keywords=_list_of_str(raw.get("expected_keywords")),
            expected_labels=_list_of_str(raw.get("expected_labels")),
            requires_ocr=raw.get("requires_ocr"),
            language=raw.get("language"),
            notes=str(raw.get("notes")) if raw.get("notes") is not None else None,
        )

        # Filter by requires_ocr if flag is set.
        if require_ocr_only and not raw.get("requires_ocr"):
            row.skipped_reason = "skipped: requires_ocr filter"
            rows.append(row)
            continue

        resolved_path = (base_dir / image_field).resolve()
        if not resolved_path.exists():
            message = f"file not found: {resolved_path}"
            if skip_missing_files:
                row.skipped_reason = message
                log.warning("Row %d skipped: %s", idx, message)
                rows.append(row)
                continue
            row.error = message
            rows.append(row)
            continue

        try:
            image_bytes = resolved_path.read_bytes()
            filename = resolved_path.name
            cap_input = input_builder(str(resolved_path), image_bytes, question, filename)

            t0 = time.perf_counter()
            cap_output = capability.run(cap_input)
            t1 = time.perf_counter()

            row.latency_ms = round((t1 - t0) * 1000.0, 3)

            # Extract FINAL_RESPONSE text.
            answer = _extract_final_response(cap_output)
            if answer is not None:
                row.answer = _truncate(answer, answer_excerpt_chars)

            # Extract stage-level latencies from MULTIMODAL_TRACE if present.
            trace = _extract_trace(cap_output)
            if trace:
                row.ocr_latency_ms = _stage_latency(trace, "ocr")
                row.vision_latency_ms = _stage_latency(trace, "vision")
                row.rag_latency_ms = (
                    _stage_latency(trace, "retrieve")
                    + _stage_latency(trace, "generate")
                )
                vision_stage = _find_stage(trace, "vision")
                if vision_stage:
                    row.vision_provider = vision_stage.get("provider")

            # Compute metrics.
            if answer is not None:
                if row.expected_answer:
                    norm_answer = _normalize(answer)
                    norm_expected = _normalize(row.expected_answer)
                    row.exact_match = 1.0 if norm_expected == norm_answer else 0.0
                    row.substring_match = 1.0 if norm_expected in norm_answer else 0.0

                row.keyword_coverage = keyword_coverage(
                    answer, row.expected_keywords
                )

                if row.expected_labels:
                    row.label_precision, row.label_recall = _label_metrics(
                        answer, row.expected_labels
                    )

        except Exception as ex:
            row.error = f"{type(ex).__name__}: {ex}"
            log.exception("Multimodal eval row %d (%r) failed", idx, image_field)

        rows.append(row)

    run_end = time.perf_counter()

    summary = _aggregate(
        rows,
        dataset_path=dataset_path or "<inline>",
    )
    summary.started_at = started_at
    summary.finished_at = _now_iso()
    summary.duration_ms = round((run_end - run_start) * 1000.0, 3)

    _log_summary(summary)
    return summary, rows


# ---------------------------------------------------------------------------
# Output extraction.
# ---------------------------------------------------------------------------


def _extract_final_response(output: Any) -> Optional[str]:
    """Pull the FINAL_RESPONSE text from a CapabilityOutput-shaped object."""
    outputs = getattr(output, "outputs", []) or []
    for artifact in outputs:
        if getattr(artifact, "type", None) == "FINAL_RESPONSE":
            content = getattr(artifact, "content", b"")
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return str(content)
    return None


def _extract_trace(output: Any) -> Optional[dict]:
    """Pull and parse the MULTIMODAL_TRACE JSON if present."""
    outputs = getattr(output, "outputs", []) or []
    for artifact in outputs:
        if getattr(artifact, "type", None) == "MULTIMODAL_TRACE":
            content = getattr(artifact, "content", b"")
            try:
                text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
                return json.loads(text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
    return None


def _find_stage(trace: dict, stage_name: str) -> Optional[dict]:
    for stage in trace.get("stages", []):
        if stage.get("stage") == stage_name:
            return stage
    return None


def _stage_latency(trace: dict, stage_name: str) -> float:
    stage = _find_stage(trace, stage_name)
    if stage is None:
        return 0.0
    return round(float(stage.get("durationMs", 0.0)), 3)


# ---------------------------------------------------------------------------
# Metrics helpers.
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for comparison."""
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def _label_metrics(
    answer: str, expected_labels: List[str]
) -> tuple[float, float]:
    """Compute label precision and recall.

    Precision: of the expected labels found in the answer, how many
               were expected (always 1.0 by definition since we only
               check expected labels).
    Recall:    fraction of expected labels found in the answer.

    In practice both equal recall since we don't have a "predicted
    labels" set separate from expected_labels. The distinction exists
    for forward compatibility if the harness later extracts predicted
    labels from structured output.
    """
    if not expected_labels:
        return 0.0, 0.0

    answer_lower = answer.lower()
    found = sum(
        1 for label in expected_labels
        if label.lower() in answer_lower
    )

    recall = found / len(expected_labels)
    precision = 1.0 if found > 0 else 0.0  # all found labels are valid
    return round(precision, 4), round(recall, 4)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate(
    rows: List[MultimodalEvalRow],
    *,
    dataset_path: str,
) -> MultimodalEvalSummary:
    evaluated = [r for r in rows if r.error is None and r.skipped_reason is None]

    em_values = [r.exact_match for r in evaluated if r.exact_match is not None]
    sm_values = [r.substring_match for r in evaluated if r.substring_match is not None]
    kc_values = [r.keyword_coverage for r in evaluated if r.keyword_coverage is not None]
    lp_values = [r.label_precision for r in evaluated if r.label_precision is not None]
    lr_values = [r.label_recall for r in evaluated if r.label_recall is not None]

    latencies = [r.latency_ms for r in evaluated]
    ocr_latencies = [r.ocr_latency_ms for r in evaluated if r.ocr_latency_ms > 0]
    vision_latencies = [r.vision_latency_ms for r in evaluated if r.vision_latency_ms > 0]
    rag_latencies = [r.rag_latency_ms for r in evaluated if r.rag_latency_ms > 0]

    vision_provider = next(
        (r.vision_provider for r in evaluated if r.vision_provider), None
    )

    return MultimodalEvalSummary(
        dataset_path=dataset_path,
        row_count=len(rows),
        evaluated_rows=len(evaluated),
        skipped_rows=sum(1 for r in rows if r.skipped_reason is not None),
        error_count=sum(1 for r in rows if r.error is not None),
        mean_exact_match=_mean_or_none(em_values),
        mean_substring_match=_mean_or_none(sm_values),
        mean_keyword_coverage=_mean_or_none(kc_values),
        mean_label_precision=_mean_or_none(lp_values),
        mean_label_recall=_mean_or_none(lr_values),
        mean_latency_ms=_mean_or_zero(latencies),
        p50_latency_ms=_p50_or_zero(latencies),
        max_latency_ms=round(max(latencies), 3) if latencies else 0.0,
        mean_ocr_latency_ms=_mean_or_zero(ocr_latencies),
        mean_vision_latency_ms=_mean_or_zero(vision_latencies),
        mean_rag_latency_ms=_mean_or_zero(rag_latencies),
        vision_provider=vision_provider,
    )


def _log_summary(summary: MultimodalEvalSummary) -> None:
    log.info(
        "Multimodal eval complete: rows=%d evaluated=%d errors=%d skipped=%d "
        "exact_match=%s substring_match=%s kw_cov=%s "
        "label_prec=%s label_rec=%s mean_total_ms=%.1f",
        summary.row_count,
        summary.evaluated_rows,
        summary.error_count,
        summary.skipped_rows,
        _fmt_opt(summary.mean_exact_match),
        _fmt_opt(summary.mean_substring_match),
        _fmt_opt(summary.mean_keyword_coverage),
        _fmt_opt(summary.mean_label_precision),
        _fmt_opt(summary.mean_label_recall),
        summary.mean_latency_ms,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _require_str(raw: Mapping[str, Any], key: str, *, row_index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Row {row_index} is missing required string field {key!r}"
        )
    return value.strip()


def _list_of_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.fmean(values), 3)


def _p50_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.median(values), 3)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _fmt_opt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().replace(microsecond=0).isoformat()


def summary_to_dict(summary: MultimodalEvalSummary) -> Dict[str, Any]:
    return asdict(summary)


def row_to_dict(row: MultimodalEvalRow) -> Dict[str, Any]:
    return asdict(row)
