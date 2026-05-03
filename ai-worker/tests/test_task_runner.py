from __future__ import annotations

import pytest

from app.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.clients.schemas import ClaimResponse, ClaimedInput, UploadResponse
from app.capabilities.ocr.fixture_provider import FixtureOcrProvider
from app.capabilities.ocr.service import OcrExtractCapability, OcrExtractService
from app.queue.messages import QueueMessage
from app.services.task_runner import CallbackDeliveryError, TaskRunner


class _EchoCapability(Capability):
    name = "MOCK"

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        assert input.inputs[0].source_file_id == "source-1"
        return CapabilityOutput(outputs=[
            CapabilityOutputArtifact(
                type="FINAL_RESPONSE",
                filename="answer.txt",
                content_type="text/plain",
                content=b"ok",
            )
        ])


class _Registry:
    def get(self, name: str) -> Capability:
        assert name == "MOCK"
        return _EchoCapability()


class _Resolver:
    def read_bytes(self, storage_uri: str) -> bytes:
        assert storage_uri == "local://input.txt"
        return b"hello"


class _CoreApi:
    def __init__(self, *, callback_failures: int = 0) -> None:
        self.callback_failures = callback_failures
        self.callback_attempts = []

    def claim(self, request):
        return ClaimResponse(
            granted=True,
            currentStatus="RUNNING",
            capability="MOCK",
            attemptNo=1,
            inputs=[
                ClaimedInput(
                    artifactId="in-1",
                    sourceFileId="source-1",
                    type="TEXT",
                    storageUri="local://input.txt",
                    contentType="text/plain",
                    sizeBytes=5,
                )
            ],
        )

    def upload_output_artifact(
        self,
        *,
        job_id: str,
        artifact_type: str,
        filename: str,
        content_type: str,
        content: bytes,
    ):
        assert job_id == "job-1"
        assert artifact_type == "FINAL_RESPONSE"
        assert filename == "answer.txt"
        assert content_type == "text/plain"
        assert content == b"ok"
        return UploadResponse(
            storageUri="local://job-1/output/answer.txt",
            sizeBytes=len(content),
            checksumSha256="abc",
        )

    def callback(self, request):
        self.callback_attempts.append(request)
        if self.callback_failures > 0:
            self.callback_failures -= 1
            raise RuntimeError("temporary callback outage")
        return object()


def _message() -> QueueMessage:
    return QueueMessage(
        jobId="job-1",
        capability="MOCK",
        attemptNo=1,
        enqueuedAtEpochMilli=1,
        callbackBaseUrl="http://core-api",
    )


def _ocr_message() -> QueueMessage:
    return QueueMessage(
        jobId="job-ocr-1",
        capability="OCR_EXTRACT",
        taskKind="OCR_EXTRACT",
        pipelineVersion="ocr-lite-v1",
        attemptNo=1,
        enqueuedAtEpochMilli=1,
        callbackBaseUrl="http://core-api",
    )


class _OcrRegistry:
    def get(self, name: str) -> Capability:
        assert name == "OCR_EXTRACT"
        return OcrExtractCapability(
            service=OcrExtractService(provider=FixtureOcrProvider("catalog text")),
        )


class _OcrResolver:
    def read_bytes(self, storage_uri: str) -> bytes:
        assert storage_uri == "local://input.png"
        return b"\x89PNG\r\n\x1a\nfixture"


class _OcrCoreApi:
    def __init__(self, *, source_file_id: str | None) -> None:
        self.source_file_id = source_file_id
        self.uploads = []
        self.callback_attempts = []

    def claim(self, request):
        return ClaimResponse(
            granted=True,
            currentStatus="RUNNING",
            capability="OCR_EXTRACT",
            attemptNo=1,
            inputs=[
                ClaimedInput(
                    artifactId="input-artifact-1",
                    sourceFileId=self.source_file_id,
                    type="INPUT_FILE",
                    storageUri="local://input.png",
                    contentType="image/png",
                    sizeBytes=8,
                )
            ],
        )

    def upload_output_artifact(
        self,
        *,
        job_id: str,
        artifact_type: str,
        filename: str,
        content_type: str,
        content: bytes,
    ):
        self.uploads.append({
            "job_id": job_id,
            "artifact_type": artifact_type,
            "filename": filename,
            "content_type": content_type,
            "content": content,
        })
        return UploadResponse(
            storageUri=f"local://job-ocr-1/output/{filename}",
            sizeBytes=len(content),
            checksumSha256=f"sha-{artifact_type}",
        )

    def callback(self, request):
        self.callback_attempts.append(request)
        return object()


def test_success_callback_retries_without_marking_failed(monkeypatch):
    monkeypatch.setattr("app.services.task_runner.time.sleep", lambda _seconds: None)
    core = _CoreApi(callback_failures=1)
    runner = TaskRunner(
        core_api=core,
        registry=_Registry(),
        resolver=_Resolver(),
        worker_id="worker-1",
    )

    runner.handle(_message())

    assert [c.outcome for c in core.callback_attempts] == ["SUCCEEDED", "SUCCEEDED"]


def test_exhausted_success_callback_raises_for_lease_recovery(monkeypatch):
    monkeypatch.setattr("app.services.task_runner.time.sleep", lambda _seconds: None)
    core = _CoreApi(callback_failures=99)
    runner = TaskRunner(
        core_api=core,
        registry=_Registry(),
        resolver=_Resolver(),
        worker_id="worker-1",
    )

    with pytest.raises(CallbackDeliveryError):
        runner.handle(_message())

    assert [c.outcome for c in core.callback_attempts] == [
        "SUCCEEDED",
        "SUCCEEDED",
        "SUCCEEDED",
    ]


def test_ocr_extract_claim_without_source_file_id_uses_legacy_fallback():
    core = _OcrCoreApi(source_file_id=None)
    runner = TaskRunner(
        core_api=core,
        registry=_OcrRegistry(),
        resolver=_OcrResolver(),
        worker_id="worker-1",
    )

    runner.handle(_ocr_message())

    assert [upload["artifact_type"] for upload in core.uploads] == [
        "OCR_RESULT_JSON",
        "OCR_TEXT_MARKDOWN",
    ]
    assert len(core.callback_attempts) == 1
    assert core.callback_attempts[0].outcome == "SUCCEEDED"
    result_json = core.uploads[0]["content"].decode("utf-8")
    assert '"sourceRecordId": "input-artifact:input-artifact-1"' in result_json


def test_ocr_extract_uploads_result_json_with_source_file_id():
    core = _OcrCoreApi(source_file_id="source-file-1")
    runner = TaskRunner(
        core_api=core,
        registry=_OcrRegistry(),
        resolver=_OcrResolver(),
        worker_id="worker-1",
    )

    runner.handle(_ocr_message())

    assert [upload["artifact_type"] for upload in core.uploads] == [
        "OCR_RESULT_JSON",
        "OCR_TEXT_MARKDOWN",
    ]
    result_json = core.uploads[0]["content"].decode("utf-8")
    assert '"sourceRecordId": "source-file-1"' in result_json
    assert core.callback_attempts[0].outcome == "SUCCEEDED"
