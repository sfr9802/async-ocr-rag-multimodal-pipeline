"""OCR evaluation harness.

Takes an already-constructed OcrProvider (tesseract, fake, or a future
engine) and runs it over a JSONL dataset of `{file, ground_truth,
language?, notes?}` rows. Intentionally narrow: the harness never
builds a provider itself, so tests can hand in a FakeOcrProvider and
the CLI can hand in a TesseractOcrProvider built from WorkerSettings.

Per-row metrics:

  - cer            : character error rate
  - wer            : word error rate (None for CJK / no-whitespace langs)
  - text_length    : characters of extracted hypothesis
  - expected_length: characters of ground truth
  - is_empty       : hypothesis has zero (normalized) characters
  - latency_ms     : wall-clock for the provider call

Summary aggregations:

  - mean_cer / median_cer / max_cer
  - mean_wer / median_wer (excluding CJK rows where WER is None)
  - empty_rate : fraction of rows that extracted zero characters
  - latency p50 / mean / max

The dataset convention for file paths is "relative to the dataset
file's directory", so a dataset at `eval/datasets/ocr_sample.jsonl`
can reference `samples/hello.png` and the harness will resolve it to
`eval/datasets/samples/hello.png` without needing absolute paths.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

from eval.harness.metrics import cer, wer

log = logging.getLogger(__name__)


# Languages for which whitespace-split WER is not meaningful. The
# harness reports `None` for WER on these rows and excludes them from
# the WER aggregation.
_CJK_LANGUAGES = frozenset({
    "zh", "zh-cn", "zh-tw", "chi_sim", "chi_tra",
    "ja", "jpn",
    "ko", "kor",
    "th", "tha",
})


class _OcrProviderLike(Protocol):
    @property
    def name(self) -> str: ...
    def ocr_image(self, image_bytes: bytes, *, mime_type: Optional[str] = None) -> Any: ...
    def ocr_pdf(self, pdf_bytes: bytes) -> Any: ...


# ---------------------------------------------------------------------------
# Row + summary dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class OcrEvalRow:
    file: str
    kind: str
    language: Optional[str] = None
    expected_length: int = 0
    text_length: int = 0
    cer: Optional[float] = None
    wer: Optional[float] = None
    is_empty: bool = False
    latency_ms: float = 0.0
    avg_confidence: Optional[float] = None
    page_count: int = 0
    engine_name: Optional[str] = None
    notes: Optional[str] = None
    hypothesis_preview: Optional[str] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class OcrEvalSummary:
    dataset_path: str
    row_count: int
    evaluated_rows: int
    skipped_rows: int
    error_count: int
    empty_rate: float
    mean_cer: Optional[float]
    median_cer: Optional[float]
    max_cer: Optional[float]
    mean_wer: Optional[float]
    median_wer: Optional[float]
    mean_latency_ms: float
    p50_latency_ms: float
    max_latency_ms: float
    engine_name: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def run_ocr_eval(
    dataset: List[Mapping[str, Any]],
    *,
    provider: _OcrProviderLike,
    dataset_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    preview_chars: int = 120,
    skip_missing_files: bool = True,
) -> tuple[OcrEvalSummary, List[OcrEvalRow]]:
    """Run OCR eval over `dataset`, returning (summary, rows).

    File paths in each row are resolved relative to `dataset_dir`
    (typically the directory containing the JSONL file). Rows that
    point at a missing file are by default skipped with a `skipped_reason`
    annotation rather than failing the whole run — set
    `skip_missing_files=False` to surface them as errors instead.
    """
    started_at = _now_iso()
    run_start = time.perf_counter()

    base_dir = Path(dataset_dir or ".").resolve()
    rows: List[OcrEvalRow] = []

    for idx, raw in enumerate(dataset, start=1):
        file_field = _require_str(raw, "file", row_index=idx)
        ground_truth = raw.get("ground_truth", "")
        if not isinstance(ground_truth, str):
            raise ValueError(
                f"Row {idx}: ground_truth must be a string (got "
                f"{type(ground_truth).__name__})"
            )
        language = raw.get("language")
        notes = raw.get("notes")

        resolved_path = (base_dir / file_field).resolve()
        kind = _classify_path(resolved_path)

        row = OcrEvalRow(
            file=file_field,
            kind=kind,
            language=language,
            expected_length=len(ground_truth),
            notes=str(notes) if notes is not None else None,
        )

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

        if kind == "unknown":
            row.error = (
                f"unsupported extension: {resolved_path.suffix!r} "
                "(expected .png, .jpg, .jpeg, .pdf)"
            )
            rows.append(row)
            continue

        try:
            t0 = time.perf_counter()
            hypothesis, page_count, avg_confidence = _run_provider(
                provider, resolved_path, kind
            )
            t1 = time.perf_counter()
        except Exception as ex:
            row.error = f"{type(ex).__name__}: {ex}"
            log.exception("OCR eval row %d (%s) failed", idx, file_field)
            rows.append(row)
            continue

        row.text_length = len(hypothesis)
        row.is_empty = len(hypothesis.strip()) == 0
        row.cer = round(cer(hypothesis, ground_truth), 6)
        row.wer = (
            None if _is_cjk(language) else round(wer(hypothesis, ground_truth), 6)
        )
        row.latency_ms = round((t1 - t0) * 1000.0, 3)
        row.avg_confidence = avg_confidence
        row.page_count = page_count
        row.engine_name = getattr(provider, "name", None)
        if preview_chars > 0:
            row.hypothesis_preview = _truncate(hypothesis, preview_chars)

        rows.append(row)

    run_end = time.perf_counter()

    summary = _aggregate(
        rows,
        dataset_path=dataset_path or "<inline>",
        engine_name=getattr(provider, "name", None),
    )
    summary.started_at = started_at
    summary.finished_at = _now_iso()
    summary.duration_ms = round((run_end - run_start) * 1000.0, 3)

    _log_summary(summary)
    return summary, rows


# ---------------------------------------------------------------------------
# Provider dispatch.
# ---------------------------------------------------------------------------


def _run_provider(
    provider: _OcrProviderLike,
    path: Path,
    kind: str,
) -> tuple[str, int, Optional[float]]:
    """Read the file, call the provider, return (text, pages, avg_conf)."""
    raw_bytes = path.read_bytes()
    if kind == "pdf":
        document = provider.ocr_pdf(raw_bytes)
        pages_attr = getattr(document, "pages", [])
        page_count = len(pages_attr)
        # Prefer the document's full_text property if it exists so we
        # match the shape OcrCapability would emit as OCR_TEXT.
        full_text = getattr(document, "full_text", None)
        if full_text is None:
            full_text = "\n\n".join(
                getattr(p, "text", "") for p in pages_attr if getattr(p, "text", "")
            )
        avg_conf = getattr(document, "avg_confidence", None)
        return full_text, page_count, _round_opt(avg_conf)

    # image
    page = provider.ocr_image(raw_bytes, mime_type=_guess_mime(path))
    text = getattr(page, "text", "")
    avg_conf = getattr(page, "avg_confidence", None)
    return text, 1, _round_opt(avg_conf)


def _classify_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        return "image"
    if suffix == ".pdf":
        return "pdf"
    return "unknown"


def _guess_mime(path: Path) -> Optional[str]:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate(
    rows: List[OcrEvalRow],
    *,
    dataset_path: str,
    engine_name: Optional[str],
) -> OcrEvalSummary:
    evaluated = [r for r in rows if r.error is None and r.skipped_reason is None]
    cer_values = [r.cer for r in evaluated if r.cer is not None]
    wer_values = [r.wer for r in evaluated if r.wer is not None]
    latencies = [r.latency_ms for r in evaluated]
    empty_count = sum(1 for r in evaluated if r.is_empty)

    return OcrEvalSummary(
        dataset_path=dataset_path,
        row_count=len(rows),
        evaluated_rows=len(evaluated),
        skipped_rows=sum(1 for r in rows if r.skipped_reason is not None),
        error_count=sum(1 for r in rows if r.error is not None),
        empty_rate=round(empty_count / len(evaluated), 4) if evaluated else 0.0,
        mean_cer=_mean_or_none(cer_values),
        median_cer=_median_or_none(cer_values),
        max_cer=round(max(cer_values), 4) if cer_values else None,
        mean_wer=_mean_or_none(wer_values),
        median_wer=_median_or_none(wer_values),
        mean_latency_ms=_mean_or_zero(latencies),
        p50_latency_ms=_median_or_zero(latencies),
        max_latency_ms=round(max(latencies), 3) if latencies else 0.0,
        engine_name=engine_name,
    )


def _log_summary(summary: OcrEvalSummary) -> None:
    log.info(
        "OCR eval complete: rows=%d evaluated=%d errors=%d skipped=%d "
        "mean_cer=%s mean_wer=%s empty_rate=%.3f mean_latency_ms=%.1f",
        summary.row_count,
        summary.evaluated_rows,
        summary.error_count,
        summary.skipped_rows,
        _fmt_opt(summary.mean_cer),
        _fmt_opt(summary.mean_wer),
        summary.empty_rate,
        summary.mean_latency_ms,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _is_cjk(language: Any) -> bool:
    if not language:
        return False
    lang = str(language).strip().lower()
    return any(lang == l or lang.startswith(l + "+") or lang.endswith("+" + l)
               for l in _CJK_LANGUAGES)


def _require_str(raw: Mapping[str, Any], key: str, *, row_index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Row {row_index} is missing required string field {key!r}"
        )
    return value.strip()


def _round_opt(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _median_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.median(values), 4)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.fmean(values), 3)


def _median_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.median(values), 3)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _fmt_opt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().replace(microsecond=0).isoformat()


def summary_to_dict(summary: OcrEvalSummary) -> Dict[str, Any]:
    return asdict(summary)


def row_to_dict(row: OcrEvalRow) -> Dict[str, Any]:
    return asdict(row)
