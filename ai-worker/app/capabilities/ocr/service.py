"""OCR_EXTRACT service and capability wrapper."""

from __future__ import annotations

import logging
from typing import Optional

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
)
from app.capabilities.ocr.artifact_builder import (
    OCR_LITE_PIPELINE_VERSION,
    build_output_artifacts,
)
from app.capabilities.ocr.models import OcrProviderError
from app.capabilities.ocr.provider import OcrLiteProvider

log = logging.getLogger(__name__)

_IMAGE_MIME_PREFIXES = ("image/png", "image/jpeg", "image/jpg")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_PDF_MIME_TYPES = ("application/pdf", "application/x-pdf")
_PDF_EXTENSION = ".pdf"


class OcrExtractService:
    def __init__(
        self,
        *,
        provider: OcrLiteProvider,
        pipeline_version: str = OCR_LITE_PIPELINE_VERSION,
    ) -> None:
        self._provider = provider
        self._pipeline_version = pipeline_version

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        artifact = self._pick_input_artifact(input)
        source_record_id = artifact.source_file_id or f"input-artifact:{artifact.artifact_id}"
        if not artifact.source_file_id:
            log.warning(
                "OCR-lite claim missing sourceFileId jobId=%s artifact=%s; "
                "using legacy sourceRecordId fallback.",
                input.job_id,
                artifact.artifact_id,
            )
        mime_type, kind = self._classify(artifact)
        filename = self._derive_filename(artifact, kind)

        log.info(
            "OCR-lite start jobId=%s artifact=%s kind=%s engine=%s",
            input.job_id,
            artifact.artifact_id,
            kind,
            self._provider.engine,
        )
        try:
            document = self._provider.extract(
                artifact.content,
                source_record_id=source_record_id,
                pipeline_version=self._pipeline_version,
                content_type=mime_type,
                filename=filename,
            )
        except OcrProviderError as ex:
            raise CapabilityError(f"OCR_{ex.code}", ex.message) from ex

        return CapabilityOutput(outputs=build_output_artifacts(document))

    @staticmethod
    def _pick_input_artifact(input: CapabilityInput) -> CapabilityInputArtifact:
        for candidate in input.inputs:
            if candidate.type == "INPUT_FILE":
                return candidate
        if not input.inputs:
            raise CapabilityError("NO_INPUT", "OCR_EXTRACT job has no input artifacts.")
        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "OCR_EXTRACT requires an INPUT_FILE artifact; got "
            + ", ".join(sorted({i.type for i in input.inputs})),
        )

    def _classify(
        self, artifact: CapabilityInputArtifact
    ) -> tuple[Optional[str], str]:
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
            ext = filename.rsplit(".", 1)[-1]
            guessed = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
            return mime or guessed, "image"

        head = artifact.content[:8]
        if head.startswith(b"%PDF-"):
            return mime or "application/pdf", "pdf"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return mime or "image/png", "image"
        if head.startswith(b"\xff\xd8\xff"):
            return mime or "image/jpeg", "image"

        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "OCR_EXTRACT supports PNG, JPEG, and PDF only. Received "
            f"content_type={artifact.content_type!r} filename={artifact.filename!r}.",
        )

    @staticmethod
    def _derive_filename(
        artifact: CapabilityInputArtifact, kind: str
    ) -> str:
        if artifact.filename:
            return artifact.filename
        ext = {"image": "img", "pdf": "pdf"}.get(kind, "bin")
        return f"{artifact.artifact_id}.{ext}"


class OcrExtractCapability(Capability):
    name = "OCR_EXTRACT"

    def __init__(self, service: OcrExtractService) -> None:
        self._service = service

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        return self._service.run(input)
