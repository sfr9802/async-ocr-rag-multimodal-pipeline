from __future__ import annotations

import pytest

from app.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.clients.schemas import ClaimResponse, ClaimedInput, UploadResponse
from app.queue.messages import QueueMessage
from app.services.task_runner import CallbackDeliveryError, TaskRunner


class _EchoCapability(Capability):
    name = "MOCK"

    def run(self, input: CapabilityInput) -> CapabilityOutput:
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
