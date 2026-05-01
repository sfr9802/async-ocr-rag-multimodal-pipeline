"""OCR capability — the thing the worker's task runner calls for OCR jobs.

Given a job's INPUT_FILE artifact, this capability:

  1. Determines whether the bytes are an image or a PDF (content_type
     header first, then filename extension, then magic-byte sniff).
  2. Dispatches to the provider's ocr_image or ocr_pdf path.
  3. Emits two output artifacts:

       - OCR_TEXT   : plain UTF-8 text of the extracted content
                      (same shape downstream RAG jobs already consume
                      as INPUT_TEXT, so an OCR→RAG chain stays a trivial
                      re-wire in a later phase)
       - OCR_RESULT : JSON envelope with input filename, mime type,
                      page count, total text length, average confidence
                      (if any), engine name, and a flat list of any
                      warnings/errors collected during extraction

Unsupported inputs fail with a typed CapabilityError("UNSUPPORTED_INPUT_TYPE")
so core-api records a clean FAILED status instead of a worker crash.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrPageResult,
    OcrProvider,
)
from app.capabilities.trace import (
    INPUT_KIND_IMAGE,
    INPUT_KIND_PDF,
    STAGE_CLASSIFY,
    STAGE_OCR,
    TraceBuilder,
    elapsed_ms,
)

log = logging.getLogger(__name__)


# MIME classifications we can handle. Image MIME types are recognised by
# prefix; PDF is an exact match. Anything else raises UNSUPPORTED_INPUT_TYPE.
_IMAGE_MIME_PREFIXES = ("image/png", "image/jpeg", "image/jpg")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_PDF_MIME_TYPES = ("application/pdf", "application/x-pdf")
_PDF_EXTENSION = ".pdf"


@dataclass(frozen=True)
class OcrCapabilityConfig:
    """Runtime knobs that affect artifact shape but not the engine.

    `min_confidence_warn` adds a document-level warning when the
    average confidence drops below the threshold; it does NOT cause
    failure. Empty extraction (zero characters) produces a warning
    too but also does not fail — downstream consumers decide whether
    that's tolerable.
    """

    min_confidence_warn: float = 40.0
    empty_text_is_warning: bool = True
    max_pages: int = 100


class OcrCapability(Capability):
    name = "OCR"

    def __init__(
        self,
        *,
        provider: OcrProvider,
        config: OcrCapabilityConfig,
    ) -> None:
        self._provider = provider
        self._config = config

    # ------------------------------------------------------------------

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        artifact = self._pick_input_artifact(input)

        # -- stage 0: classify -----------------------------------------
        # Happens before the trace builder exists, so an UNSUPPORTED
        # input raises a clean typed error with no ambiguous stage
        # bookkeeping.
        started = time.monotonic()
        mime_type, kind = self._classify(artifact)
        filename = self._derive_filename(artifact, kind)
        classify_ms = elapsed_ms(started)

        input_kind = INPUT_KIND_PDF if kind == "pdf" else INPUT_KIND_IMAGE
        builder = TraceBuilder(capability=self.name, input_kind=input_kind)
        builder.record_ok(
            STAGE_CLASSIFY,
            duration_ms=classify_ms,
            details={
                "mimeType": mime_type,
                "filename": filename,
                "kind": kind,
                "sizeBytes": len(artifact.content),
            },
        )

        log.info(
            "OCR start jobId=%s artifact=%s kind=%s mime=%s filename=%s "
            "size_bytes=%d",
            input.job_id,
            artifact.artifact_id,
            kind,
            mime_type,
            filename,
            len(artifact.content),
        )

        # -- stage A: OCR extraction -----------------------------------
        started = time.monotonic()
        try:
            if kind == "image":
                page = self._provider.ocr_image(
                    artifact.content, mime_type=mime_type
                )
                document = OcrDocumentResult(
                    pages=[page],
                    engine_name=self._provider.name,
                    warnings=[],
                )
            elif kind == "pdf":
                preflight_page_count = self._preflight_pdf_page_count(
                    artifact.content,
                )
                if (
                    preflight_page_count is not None
                    and preflight_page_count > self._config.max_pages
                ):
                    ocr_ms = elapsed_ms(started)
                    builder.record_fail(
                        STAGE_OCR,
                        provider=self._provider.name,
                        code="OCR_TOO_MANY_PAGES",
                        message=(
                            f"PDF has {preflight_page_count} pages, limit is "
                            f"{self._config.max_pages}"
                        ),
                        duration_ms=ocr_ms,
                        retryable=False,
                        details={"pageCount": preflight_page_count},
                    )
                    builder.finalize_failed()
                    raise CapabilityError(
                        "OCR_TOO_MANY_PAGES",
                        f"PDF has {preflight_page_count} pages, limit is "
                        f"{self._config.max_pages}. Raise "
                        f"AIPIPELINE_WORKER_OCR_MAX_PAGES or split the "
                        f"document before submitting. "
                        f"| trace: {builder.summary()}",
                    )
                document = self._provider.ocr_pdf(artifact.content)
                if len(document.pages) > self._config.max_pages:
                    ocr_ms = elapsed_ms(started)
                    builder.record_fail(
                        STAGE_OCR,
                        provider=self._provider.name,
                        code="OCR_TOO_MANY_PAGES",
                        message=(
                            f"PDF has {len(document.pages)} pages, limit is "
                            f"{self._config.max_pages}"
                        ),
                        duration_ms=ocr_ms,
                        retryable=False,
                        details={"pageCount": len(document.pages)},
                    )
                    builder.finalize_failed()
                    raise CapabilityError(
                        "OCR_TOO_MANY_PAGES",
                        f"PDF has {len(document.pages)} pages, limit is "
                        f"{self._config.max_pages}. Raise "
                        f"AIPIPELINE_WORKER_OCR_MAX_PAGES or split the "
                        f"document before submitting. "
                        f"| trace: {builder.summary()}",
                    )
            else:
                # _classify already raises, but belt-and-suspenders:
                raise CapabilityError(
                    "UNSUPPORTED_INPUT_TYPE",
                    f"OCR cannot process kind={kind!r}",
                )
        except OcrError as ex:
            ocr_ms = elapsed_ms(started)
            builder.record_fail(
                STAGE_OCR,
                provider=self._provider.name,
                code=f"OCR_{ex.code}",
                message=ex.message,
                duration_ms=ocr_ms,
                retryable=False,
            )
            builder.finalize_failed()
            raise CapabilityError(
                f"OCR_{ex.code}",
                f"{ex.message} | trace: {builder.summary()}",
            ) from ex

        ocr_ms = elapsed_ms(started)
        warnings = self._collect_warnings(document)

        # Classify the OCR stage as ok / warn depending on whether
        # the engine produced any usable text and whether confidence
        # fell below the configured threshold. An empty-text or
        # low-confidence run still emits the artifacts, matching the
        # existing behavior — the trace just flags it clearly.
        ocr_details = {
            "pageCount": len(document.pages),
            "textLength": document.total_text_length,
            "avgConfidence": (
                round(document.avg_confidence, 2)
                if document.avg_confidence is not None
                else None
            ),
            "engineName": document.engine_name,
        }
        if document.total_text_length == 0:
            builder.record_warn(
                STAGE_OCR,
                provider=self._provider.name,
                code="OCR_EMPTY_TEXT",
                message="OCR produced zero characters",
                duration_ms=ocr_ms,
                details=ocr_details,
            )
        elif (
            document.avg_confidence is not None
            and document.avg_confidence < self._config.min_confidence_warn
        ):
            builder.record_warn(
                STAGE_OCR,
                provider=self._provider.name,
                code="OCR_LOW_CONFIDENCE",
                message=(
                    f"avg confidence {document.avg_confidence:.1f} "
                    f"< warn threshold {self._config.min_confidence_warn:.1f}"
                ),
                duration_ms=ocr_ms,
                details=ocr_details,
            )
        else:
            builder.record_ok(
                STAGE_OCR,
                provider=self._provider.name,
                duration_ms=ocr_ms,
                details=ocr_details,
            )

        # If any OCR record was a warn, treat the overall trace as
        # partial so consumers can tell "completed with caveats" apart
        # from "clean success". Hard-fails never reach here — they
        # raise above.
        has_warn = any(rec.status == "warn" for rec in builder.trace.stages)
        trace = (
            builder.finalize_partial() if has_warn else builder.finalize_ok()
        )

        log.info(
            "OCR done jobId=%s pages=%d text_len=%d avg_conf=%s warnings=%d "
            "final_status=%s",
            input.job_id,
            len(document.pages),
            document.total_text_length,
            _fmt_conf(document.avg_confidence),
            len(warnings),
            trace.final_status,
        )

        ocr_text_artifact = CapabilityOutputArtifact(
            type="OCR_TEXT",
            filename="ocr.txt",
            content_type="text/plain; charset=utf-8",
            content=document.full_text.encode("utf-8"),
        )
        ocr_result_artifact = CapabilityOutputArtifact(
            type="OCR_RESULT",
            filename="ocr-result.json",
            content_type="application/json",
            content=self._build_result_json(
                filename=filename,
                mime_type=mime_type,
                kind=kind,
                document=document,
                warnings=warnings,
                trace=trace,
            ).encode("utf-8"),
        )
        return CapabilityOutput(outputs=[ocr_text_artifact, ocr_result_artifact])

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_input_artifact(input: CapabilityInput) -> CapabilityInputArtifact:
        """First INPUT_FILE wins; any other artifact is rejected.

        We deliberately do NOT fall back to the first artifact regardless
        of type — OCR on an INPUT_TEXT is almost certainly a bug and we'd
        rather raise than silently "OCR" the prompt string.
        """
        for candidate in input.inputs:
            if candidate.type == "INPUT_FILE":
                return candidate
        if not input.inputs:
            raise CapabilityError(
                "NO_INPUT",
                "OCR job has no input artifacts.",
            )
        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "OCR requires an INPUT_FILE artifact; got "
            + ", ".join(sorted({i.type for i in input.inputs})),
        )

    def _classify(
        self, artifact: CapabilityInputArtifact
    ) -> tuple[Optional[str], str]:
        """Return (normalized_mime_type_or_None, kind) where kind is one
        of 'image', 'pdf'. Raises CapabilityError for unsupported inputs.

        Detection order:
          1. Explicit content_type header from core-api
          2. Filename extension
          3. Magic-byte sniff on the first ~8 bytes

        Multiple signals agreeing is the happy path. A disagreement is
        non-fatal as long as at least one signal produces a supported
        classification.
        """
        mime = (artifact.content_type or "").split(";")[0].strip().lower() or None
        filename = (artifact.filename or "").lower()

        if mime:
            if mime in _PDF_MIME_TYPES:
                return mime, "pdf"
            if any(mime.startswith(prefix) for prefix in _IMAGE_MIME_PREFIXES):
                return mime, "image"

        if filename.endswith(_PDF_EXTENSION):
            return mime or "application/pdf", "pdf"
        if filename.endswith(_IMAGE_EXTENSIONS):
            # Pick a canonical MIME for each extension.
            ext = filename.rsplit(".", 1)[-1]
            guessed = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
            return mime or guessed, "image"

        # Fall back to a magic-byte sniff on the leading bytes.
        head = artifact.content[:8]
        if head.startswith(b"%PDF-"):
            return mime or "application/pdf", "pdf"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return mime or "image/png", "image"
        if head.startswith(b"\xff\xd8\xff"):
            return mime or "image/jpeg", "image"

        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "OCR supports PNG, JPEG, and PDF only. Received "
            f"content_type={artifact.content_type!r} filename={artifact.filename!r}.",
        )

    @staticmethod
    def _derive_filename(
        artifact: CapabilityInputArtifact, kind: str
    ) -> str:
        """Return a best-effort display filename for the OCR_RESULT envelope.

        In order:
          1. Explicit filename on the artifact (TaskRunner populates this
             from the storage URI's trailing segment).
          2. `{artifact_id}.{ext}` where ext is inferred from `kind`.
        """
        if artifact.filename:
            return artifact.filename
        ext = {"image": "img", "pdf": "pdf"}.get(kind, "bin")
        return f"{artifact.artifact_id}.{ext}"

    def _preflight_pdf_page_count(self, pdf_bytes: bytes) -> Optional[int]:
        counter = getattr(self._provider, "pdf_page_count", None)
        if not callable(counter):
            return None
        page_count = counter(pdf_bytes)
        if page_count is None:
            return None
        return int(page_count)

    # ------------------------------------------------------------------
    # warnings + result envelope
    # ------------------------------------------------------------------

    def _collect_warnings(self, document: OcrDocumentResult) -> List[str]:
        warnings = list(document.warnings)
        for page in document.pages:
            warnings.extend(page.warnings)

        if self._config.empty_text_is_warning and document.total_text_length == 0:
            warnings.append(
                "extraction produced zero characters — input may be blank, "
                "low-contrast, or in an unsupported script"
            )

        conf = document.avg_confidence
        if conf is not None and conf < self._config.min_confidence_warn:
            warnings.append(
                f"average OCR confidence {conf:.1f} is below "
                f"warn threshold {self._config.min_confidence_warn:.1f}"
            )

        return warnings

    @staticmethod
    def _build_result_json(
        *,
        filename: str,
        mime_type: Optional[str],
        kind: str,
        document: OcrDocumentResult,
        warnings: List[str],
        trace,  # type: PipelineTrace — typed via duck-typing to avoid circular import
    ) -> str:
        """Serialize the OCR_RESULT envelope.

        The top-level body keeps its phase-2 extraction shape unchanged
        (filename / mimeType / kind / engineName / pageCount / textLength
        / avgConfidence / pages / warnings). A new `trace` field carries
        the normalized stage flow — it is additive, so consumers that
        don't parse it see zero behavior change.
        """
        avg_conf = document.avg_confidence
        body = {
            "filename": filename,
            "mimeType": mime_type,
            "kind": kind,
            "engineName": document.engine_name,
            "pageCount": len(document.pages),
            "textLength": document.total_text_length,
            "avgConfidence": (
                round(avg_conf, 2) if avg_conf is not None else None
            ),
            "pages": [
                {
                    "pageNumber": p.page_number,
                    "textLength": len(p.text),
                    "avgConfidence": (
                        round(p.avg_confidence, 2)
                        if p.avg_confidence is not None
                        else None
                    ),
                    "warnings": list(p.warnings),
                }
                for p in document.pages
            ],
            "warnings": warnings,
            "trace": trace.to_dict(),
        }
        return json.dumps(body, ensure_ascii=False, indent=2)


def _fmt_conf(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"
