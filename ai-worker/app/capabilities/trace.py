"""Normalized pipeline trace + stable error-code registry.

Shared by the OCR and MULTIMODAL capabilities so that operators get a
**single, stable, compact** view of stage flow, regardless of which
capability produced the job. The goal is operational clarity on the
partial-fallback and failure paths — "exactly which stage did what,
in what order, and with what code" should be one download away.

Design principles:

* **Compact.** Stage records do NOT embed full OCR text, full fused
  context, full retrieved chunks, or caption prose longer than a
  short preview. Payloads are metadata-first; previews are bounded.

* **Stable.** The schema carries `schemaVersion = "trace.v1"` so
  forward-compatible readers can key off it. Stage names, status
  values, and the StableErrorCode constants are listed in the module
  docstring below so the set is inspectable without reading the call
  sites.

* **Uniform across capabilities.** Both OCR and MULTIMODAL populate
  the same `PipelineTrace` shape. The only difference is which
  stages appear in the list. Consumers that parse the trace from
  `OCR_RESULT.trace` vs. from a `MULTIMODAL_TRACE` artifact see
  byte-identical field names.

* **Reuses existing artifacts.** MULTIMODAL_TRACE already exists as
  an opt-in artifact type — this module formalizes its payload
  rather than introducing a new artifact type. OCR embeds the trace
  under a new top-level `trace` key inside the existing `OCR_RESULT`
  JSON, which keeps the extraction fields in charge of extraction
  semantics and the trace in charge of stage flow.

## Normalized stage names

Stages the capability layer currently records (strings below are the
literal values in `StageRecord.stage`):

    classify   — input file type / mime classification
    ocr        — text extraction from image or PDF
    vision     — visual-description provider
    fusion     — deterministic OCR + vision + question fusion
    retrieve   — text-RAG retrieval against FAISS
    generate   — grounded answer generation

Additional stage names reserved for the TaskRunner / outer orchestrator
(documented here so the vocabulary is unified, even though the worker
capability layer doesn't emit them today):

    fetch      — downloading input artifact bytes
    decode     — decoding artifact content to in-memory form
    upload     — uploading output artifact bytes to core-api
    callback   — reporting terminal state back to core-api

The `stage` field is a free-form string — tests can assert on any
value. The constants below are the canonical names consumers should
look for.

## Stable error / warning codes

The table in `STABLE_ERROR_CODES` is the authoritative registry. Every
capability is expected to use codes from the table when the failure
is a known class; free-form codes from an underlying provider are
preserved (prefixed by the capability name) without being blocked.

See `docs/architecture.md` for the full table + example payloads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


TRACE_SCHEMA_VERSION = "trace.v1"

# Canonical stage names. String values are the stable on-the-wire form.
STAGE_CLASSIFY = "classify"
STAGE_OCR = "ocr"
STAGE_VISION = "vision"
STAGE_FUSION = "fusion"
STAGE_RETRIEVE = "retrieve"
STAGE_GENERATE = "generate"
# AGENT/AUTO single-pass stages. ``route`` records the router decision
# (rule or LLM) and ``dispatch`` records the call into the chosen
# sub-capability. Both are populated only by AgentCapability.run — the
# loop path keeps emitting its own AGENT_TRACE artifact unchanged.
STAGE_ROUTE = "route"
STAGE_DISPATCH = "dispatch"
# Reserved for the TaskRunner / outer orchestrator — not currently
# populated by the in-capability trace.
STAGE_FETCH = "fetch"
STAGE_DECODE = "decode"
STAGE_UPLOAD = "upload"
STAGE_CALLBACK = "callback"

# Stage statuses. Keep the set tiny — tests and consumers enumerate it.
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"

# Final trace statuses.
FINAL_OK = "ok"
FINAL_PARTIAL = "partial"
FINAL_FAILED = "failed"

# Input kinds.
INPUT_KIND_IMAGE = "image"
INPUT_KIND_PDF = "pdf"
INPUT_KIND_TEXT = "text"
INPUT_KIND_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Stable error / warning code registry
# ---------------------------------------------------------------------------


#: Authoritative list of codes the trace / CapabilityError layer is
#: expected to emit. Values are short human-readable descriptions used
#: by the architecture doc table and by unit tests that verify the
#: registry is kept in sync with the code paths.
#:
#: Free-form codes from underlying providers (e.g. a VLM raising its
#: own `PROVIDER_TIMEOUT`) are allowed — the capability wraps them
#: with the appropriate capability prefix (e.g. "VISION_PROVIDER_TIMEOUT")
#: when recording the stage, without blocking on whether the code is
#: registered here.
STABLE_ERROR_CODES: Dict[str, str] = {
    # OCR-side typed errors. The OcrCapability wraps any OcrError.code
    # with an `OCR_` prefix before raising CapabilityError, so the
    # registered values listed here are the ones consumers will see.
    "OCR_IMAGE_DECODE_FAILED": (
        "Pillow could not decode the input image bytes."
    ),
    "OCR_PDF_OPEN_FAILED": (
        "PyMuPDF could not open the input PDF (corrupt or encrypted)."
    ),
    "OCR_PDF_EMPTY": "PDF contains zero pages.",
    "OCR_TESSERACT_RUN_FAILED": (
        "Tesseract invocation failed mid-extraction."
    ),
    "OCR_TOO_MANY_PAGES": (
        "PDF exceeds the configured OCR page cap."
    ),

    # MULTIMODAL-side typed errors, raised by MultimodalCapability
    # directly. Provider errors from OCR or vision get normalized
    # under these codes.
    "MULTIMODAL_ALL_PROVIDERS_FAILED": (
        "Both OCR and vision produced no usable signal."
    ),
    "MULTIMODAL_RETRIEVAL_FAILED": (
        "Retrieval stage raised an exception after OCR / vision succeeded."
    ),
    "MULTIMODAL_GENERATION_FAILED": (
        "Generation stage raised an exception after retrieval succeeded."
    ),
    # Shared between OCR and MULTIMODAL — raised by the classifier
    # before any trace exists. Deliberately NOT prefixed so clients
    # that already match on it keep working.
    "UNSUPPORTED_INPUT_TYPE": (
        "Input artifact is not a supported OCR/MULTIMODAL input "
        "(PNG/JPEG/PDF)."
    ),
    "NO_INPUT": (
        "Capability received zero input artifacts."
    ),

    # Trace warnings (not raised as errors but surfaced on stages).
    "OCR_EMPTY_TEXT": (
        "OCR extraction produced zero characters — input may be blank."
    ),
    "OCR_LOW_CONFIDENCE": (
        "Average OCR confidence fell below the configured warn threshold."
    ),
    "VISION_UNAVAILABLE": (
        "Vision provider did not produce a description — OCR signal only."
    ),
    "OCR_UNAVAILABLE": (
        "OCR extraction produced no text — vision signal only."
    ),
    "FUSION_DEFAULT_QUERY": (
        "No user question, OCR, or vision caption — fused on default query."
    ),

    # AGENT / AUTO single-pass dispatch errors. Raised by AgentCapability
    # when the routed-to sub-capability is not registered or no usable
    # input was supplied. Folded into the FAILED callback's errorMessage
    # alongside the AGENT trace summary.
    "AUTO_NO_INPUT": (
        "AUTO job received neither usable text nor a routable file."
    ),
    "AUTO_RAG_UNAVAILABLE": (
        "AUTO routed to RAG but the RAG capability is not registered."
    ),
    "AUTO_OCR_UNAVAILABLE": (
        "AUTO routed to OCR but the OCR capability is not registered."
    ),
    "AUTO_MULTIMODAL_UNAVAILABLE": (
        "AUTO routed to MULTIMODAL but the MULTIMODAL capability is not registered."
    ),

    # MULTIMODAL retrieval warning. Emitted when the retriever returns
    # zero hits — the pipeline still answers using the fused OCR + vision
    # context as the synthetic rank-0 chunk, but the trace flags it.
    "RETRIEVAL_EMPTY": (
        "Retrieval returned zero hits — answer grounded only on fused context."
    ),
}


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """One entry in a `PipelineTrace.stages` list.

    Field conventions:

    - `stage` — canonical stage name (`STAGE_*` constants).
    - `provider` — stable identifier of the component that ran this
      stage (e.g. "tesseract-5.3.3", "heuristic-vision-v1"). `None`
      when the stage is intrinsic to the capability (e.g. `classify`).
    - `status` — one of STATUS_OK / STATUS_WARN / STATUS_FAIL /
      STATUS_SKIPPED.
    - `code` — stable error / warning code on non-OK statuses. `None`
      on pure OK stages.
    - `message` — short, human-readable, bounded to ~200 chars by the
      builder. Not a dump of provider output.
    - `retryable` — `True` when retrying the stage would plausibly
      help (transient network hiccup, timeout), `False` when the
      failure is a hard data problem (corrupt PDF, unsupported file
      type), `None` when unknown.
    - `fallback_used` — `True` when the pipeline proceeded past this
      stage via a fallback path (e.g. OCR empty → vision-only). Does
      not replace the `status` field — a stage can be `fail` with
      `fallback_used=True` meaning "this one failed but the pipeline
      kept going".
    - `duration_ms` — wall-clock time attributable to this stage.
    - `details` — small metadata dict. The builder enforces a loose
      payload cap so this never becomes a place to smuggle full
      OCR text or chunk bodies.
    """

    stage: str
    provider: Optional[str] = None
    status: str = STATUS_OK
    code: Optional[str] = None
    message: Optional[str] = None
    retryable: Optional[bool] = None
    fallback_used: bool = False
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the on-the-wire camelCase form.

        Matches the rest of the worker's JSON artifact conventions so
        consumers that already parse RETRIEVAL_RESULT / VISION_RESULT
        don't have to switch case styles."""
        return {
            "stage": self.stage,
            "provider": self.provider,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "fallbackUsed": self.fallback_used,
            "durationMs": round(self.duration_ms, 3),
            "details": dict(self.details),
        }


@dataclass
class PipelineTrace:
    """Top-level trace payload embedded into MULTIMODAL_TRACE / OCR_RESULT.

    Kept deliberately flat — no nested maps other than per-stage
    `details`. `to_dict()` returns the canonical on-the-wire form.
    """

    capability: str
    input_kind: str
    final_status: str = FINAL_OK
    stages: List[StageRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "capability": self.capability,
            "inputKind": self.input_kind,
            "finalStatus": self.final_status,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": list(self.warnings),
            "summary": self.summary(),
        }

    def summary(self) -> str:
        """Compact one-line stage-flow summary.

        Format examples::

            classify:ok(0ms) ocr:ok(2ms) vision:ok(3ms) fusion:ok(0ms) retrieve:ok(1ms) generate:ok(0ms)
            classify:ok(0ms) ocr:ok(2ms) vision:fail(VISION_PROVIDER_TIMEOUT,5ms,fallback) fusion:ok(0ms) retrieve:ok(1ms) generate:ok(0ms)
            classify:ok(0ms) ocr:fail(OCR_IMAGE_DECODE_FAILED,2ms) vision:fail(VISION_PROVIDER_TIMEOUT,5ms) fusion:skipped retrieve:skipped generate:skipped

        Designed to be folded verbatim into an error message so
        operators can see stage progression in the `errorMessage`
        field of a FAILED job without downloading the trace artifact.
        """
        parts: List[str] = []
        for rec in self.stages:
            if rec.status == STATUS_SKIPPED:
                parts.append(f"{rec.stage}:skipped")
                continue
            if rec.status == STATUS_OK:
                parts.append(f"{rec.stage}:ok({rec.duration_ms:.0f}ms)")
                continue
            # warn / fail — include code (falling back to "-") and
            # optional ",fallback" marker.
            code_tag = rec.code or "-"
            inner = f"{code_tag},{rec.duration_ms:.0f}ms"
            if rec.fallback_used:
                inner = f"{inner},fallback"
            parts.append(f"{rec.stage}:{rec.status}({inner})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Builder — the ergonomic API capabilities use to populate a trace
# ---------------------------------------------------------------------------


# Loose cap on per-stage detail payload size to keep traces compact.
# Characters, not bytes — a generous budget so small metadata survives
# unchanged but a full OCR dump would be truncated.
_MAX_DETAIL_VALUE_CHARS = 400


class TraceBuilder:
    """Mutable trace builder threaded through a capability's run().

    Usage pattern::

        builder = TraceBuilder(capability="MULTIMODAL", input_kind="image")
        started = time.monotonic()
        # ... classify ...
        builder.record_ok("classify", provider=None,
                          duration_ms=_elapsed_ms(started),
                          details={"mimeType": mime})

        started = time.monotonic()
        try:
            doc = ocr.ocr_image(...)
        except OcrError as ex:
            builder.record_fail(
                "ocr",
                provider=ocr.name,
                code="OCR_" + ex.code,
                message=ex.message,
                duration_ms=_elapsed_ms(started),
                retryable=False,
                fallback_used=True,
            )
        else:
            builder.record_ok(
                "ocr",
                provider=ocr.name,
                duration_ms=_elapsed_ms(started),
                details={"pageCount": len(doc.pages),
                         "textLength": doc.total_text_length,
                         "avgConfidence": doc.avg_confidence},
            )

    The builder does NOT enforce uniqueness of stage names — tests
    that re-record a stage will get two entries, which is the correct
    behavior for retry loops.

    It also does NOT enforce a schema version bump when the API
    changes — bump `TRACE_SCHEMA_VERSION` explicitly.
    """

    def __init__(self, *, capability: str, input_kind: str) -> None:
        self._trace = PipelineTrace(
            capability=capability,
            input_kind=input_kind,
        )

    # --- stage recording ----------------------------------------------

    def record_ok(
        self,
        stage: str,
        *,
        provider: Optional[str] = None,
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a clean success for `stage`."""
        self._trace.stages.append(
            StageRecord(
                stage=stage,
                provider=provider,
                status=STATUS_OK,
                duration_ms=duration_ms,
                details=_clip_details(details or {}),
            )
        )

    def record_warn(
        self,
        stage: str,
        *,
        provider: Optional[str] = None,
        code: Optional[str] = None,
        message: Optional[str] = None,
        duration_ms: float = 0.0,
        fallback_used: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a non-fatal warning on `stage` (e.g. low-confidence OCR)."""
        self._trace.stages.append(
            StageRecord(
                stage=stage,
                provider=provider,
                status=STATUS_WARN,
                code=code,
                message=_clip_message(message),
                duration_ms=duration_ms,
                fallback_used=fallback_used,
                details=_clip_details(details or {}),
            )
        )

    def record_fail(
        self,
        stage: str,
        *,
        provider: Optional[str] = None,
        code: Optional[str] = None,
        message: Optional[str] = None,
        duration_ms: float = 0.0,
        retryable: Optional[bool] = None,
        fallback_used: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a hard failure on `stage`.

        `fallback_used=True` means the pipeline continued past this
        stage via a fallback — do not confuse it with "the whole
        pipeline failed", which is represented by `finalize_failed`.
        """
        self._trace.stages.append(
            StageRecord(
                stage=stage,
                provider=provider,
                status=STATUS_FAIL,
                code=code,
                message=_clip_message(message),
                retryable=retryable,
                fallback_used=fallback_used,
                duration_ms=duration_ms,
                details=_clip_details(details or {}),
            )
        )

    def record_skipped(
        self,
        stage: str,
        *,
        message: Optional[str] = None,
    ) -> None:
        """Log that a stage was intentionally skipped.

        Useful when an earlier failure means we never even tried a
        downstream stage (retrieve skipped because ocr+vision both
        failed) — the trace still carries a placeholder so the
        stage list remains complete and inspectable."""
        self._trace.stages.append(
            StageRecord(
                stage=stage,
                status=STATUS_SKIPPED,
                message=_clip_message(message),
            )
        )

    # --- pipeline-level warnings --------------------------------------

    def add_warning(self, message: str) -> None:
        """Attach a pipeline-level warning that doesn't belong to a stage."""
        if message:
            self._trace.warnings.append(message)

    # --- finalization --------------------------------------------------

    def finalize_ok(self) -> PipelineTrace:
        """Mark the trace as fully successful and return it."""
        self._trace.final_status = FINAL_OK
        return self._trace

    def finalize_partial(self) -> PipelineTrace:
        """Mark the trace as partial (some stage failed but the
        capability produced useful output via a fallback)."""
        self._trace.final_status = FINAL_PARTIAL
        return self._trace

    def finalize_failed(self) -> PipelineTrace:
        """Mark the trace as terminal failure and return it.

        Typically called right before raising a `CapabilityError` so
        the trace summary can be folded into the error message."""
        self._trace.final_status = FINAL_FAILED
        return self._trace

    @property
    def trace(self) -> PipelineTrace:
        """Read-only access to the in-progress trace."""
        return self._trace

    def summary(self) -> str:
        """Return the current stage-flow summary string (can be called
        any time, used to build error messages before finalize)."""
        return self._trace.summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_message(msg: Optional[str], *, max_chars: int = 200) -> Optional[str]:
    if msg is None:
        return None
    if len(msg) <= max_chars:
        return msg
    return msg[: max_chars - 3] + "..."


def _clip_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Bound string values inside a details dict so the trace stays compact.

    Non-string values pass through unchanged — numbers, booleans, and
    small lists are the primary payload. String fields (captions,
    filenames, error messages, previews) are capped at
    `_MAX_DETAIL_VALUE_CHARS`.
    """
    out: Dict[str, Any] = {}
    for k, v in details.items():
        if isinstance(v, str) and len(v) > _MAX_DETAIL_VALUE_CHARS:
            out[k] = v[: _MAX_DETAIL_VALUE_CHARS - 3] + "..."
        else:
            out[k] = v
    return out


def elapsed_ms(started_monotonic: float) -> float:
    """Return `(now - started)` in milliseconds, rounded to 3 decimals.

    Kept here so callers don't have to duplicate the `time.monotonic()`
    + `* 1000` arithmetic at every stage boundary."""
    return round((time.monotonic() - started_monotonic) * 1000.0, 3)
