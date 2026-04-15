"""Task runner — the orchestration spine of the worker.

Given a QueueMessage this module:
  1. Claims the job on core-api
  2. Loads INPUT artifact bytes via the storage resolver (local) or
     falls back to core-api HTTP for schemes it can't read directly
  3. Invokes the matching capability
  4. Uploads output artifacts through core-api
  5. Posts a SUCCEEDED or FAILED callback

Everything idempotent-ish lives in core-api (claim + callback); this
runner is a straight-line orchestrator and intentionally carries no state
between jobs.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Optional

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
)
from app.capabilities.registry import CapabilityRegistry
from app.clients.core_api_client import CoreApiClient
from app.clients.schemas import (
    CallbackRequest,
    ClaimRequest,
    ClaimedInput,
    OutputArtifactPayload,
)
from app.queue.messages import QueueMessage
from app.storage.resolver import StorageResolver

log = logging.getLogger(__name__)


class TaskRunner:
    def __init__(
        self,
        core_api: CoreApiClient,
        registry: CapabilityRegistry,
        resolver: StorageResolver,
        worker_id: str,
    ) -> None:
        self._core_api = core_api
        self._registry = registry
        self._resolver = resolver
        self._worker_id = worker_id

    # ------------------------------------------------------------------

    def handle(self, message: QueueMessage) -> None:
        log.info(
            "Received dispatch jobId=%s capability=%s attempt=%d",
            message.job_id, message.capability, message.attempt_no,
        )

        claim_response = self._core_api.claim(
            ClaimRequest(
                job_id=message.job_id,
                worker_claim_token=self._worker_id,
                attempt_no=message.attempt_no,
            )
        )
        if not claim_response.granted:
            log.info(
                "Claim denied jobId=%s reason=%s status=%s",
                message.job_id, claim_response.reason, claim_response.current_status,
            )
            return

        try:
            capability = self._registry.get(claim_response.capability or message.capability)
            fetched_inputs = [self._fetch_input(i) for i in claim_response.inputs]
            cap_input = CapabilityInput(
                job_id=message.job_id,
                capability=claim_response.capability or message.capability,
                attempt_no=claim_response.attempt_no or message.attempt_no,
                inputs=fetched_inputs,
            )
            output = self._execute(capability, cap_input)
            uploaded = self._upload_outputs(message.job_id, output)
            self._send_success_callback(message.job_id, uploaded)
            log.info("Job %s SUCCEEDED", message.job_id)

        except CapabilityError as ex:
            log.warning("Job %s failed in capability: %s", message.job_id, ex)
            self._send_failure_callback(message.job_id, ex.code, ex.message)
        except Exception as ex:
            log.exception("Job %s crashed unexpectedly", message.job_id)
            self._send_failure_callback(
                message.job_id,
                "WORKER_EXCEPTION",
                f"{type(ex).__name__}: {ex}\n{traceback.format_exc(limit=3)}",
            )

    # ------------------------------------------------------------------

    def _fetch_input(self, claimed: ClaimedInput) -> CapabilityInputArtifact:
        try:
            content = self._resolver.read_bytes(claimed.storage_uri)
        except NotImplementedError:
            # scheme not directly readable — fall back to HTTP
            content = self._core_api.download_artifact_content(claimed.artifact_id)
        return CapabilityInputArtifact(
            artifact_id=claimed.artifact_id,
            type=claimed.type,
            content=content,
            content_type=claimed.content_type,
            filename=_filename_from_storage_uri(claimed.storage_uri),
        )

    def _execute(self, capability: Capability, cap_input: CapabilityInput) -> CapabilityOutput:
        return capability.run(cap_input)

    def _upload_outputs(self, job_id: str, output: CapabilityOutput) -> list[OutputArtifactPayload]:
        uploaded: list[OutputArtifactPayload] = []
        for artifact in output.outputs:
            response = self._core_api.upload_output_artifact(
                job_id=job_id,
                artifact_type=artifact.type,
                filename=artifact.filename,
                content_type=artifact.content_type,
                content=artifact.content,
            )
            uploaded.append(OutputArtifactPayload(
                type=artifact.type,
                storage_uri=response.storage_uri,
                content_type=artifact.content_type,
                size_bytes=response.size_bytes,
                checksum_sha256=response.checksum_sha256,
            ))
        return uploaded

    def _send_success_callback(
        self, job_id: str, outputs: list[OutputArtifactPayload]
    ) -> None:
        self._core_api.callback(CallbackRequest(
            job_id=job_id,
            callback_id=str(uuid.uuid4()),
            worker_claim_token=self._worker_id,
            outcome="SUCCEEDED",
            output_artifacts=outputs,
        ))

    def _send_failure_callback(self, job_id: str, code: str, message: str) -> None:
        self._core_api.callback(CallbackRequest(
            job_id=job_id,
            callback_id=str(uuid.uuid4()),
            worker_claim_token=self._worker_id,
            outcome="FAILED",
            error_code=code,
            error_message=message[:1900],
        ))


def _filename_from_storage_uri(storage_uri: Optional[str]) -> Optional[str]:
    """Recover the original filename from a core-api storage URI.

    LocalFilesystemStorageAdapter formats object keys as
        `{jobId}/{type}/{uuid}-{sanitized_filename}`
    so the trailing path segment is `{uuid}-{filename}`. UUIDs are a
    fixed 36 characters, so we strip the first 37 bytes (36 + the
    dash) off the last segment and return what's left. If anything
    looks off, we return None rather than guessing — the OCR
    capability gracefully falls back to `{artifact_id}.{ext}`.
    """
    if not storage_uri:
        return None
    last_segment = storage_uri.rsplit("/", 1)[-1]
    # UUID is 8-4-4-4-12 chars with dashes → 36 chars; +1 for the separator
    if len(last_segment) <= 37 or last_segment[36] != "-":
        return None
    return last_segment[37:] or None
