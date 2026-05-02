"""FastAPI route tests for POST /internal/tasks/ocr-extract."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app


class _FakeRunner:
    def __init__(self) -> None:
        self.messages = []

    def handle(self, message) -> None:
        self.messages.append(message)


def test_ocr_extract_route_builds_task_runner_message():
    runner = _FakeRunner()
    client = TestClient(create_app(runner=runner))

    response = client.post(
        "/internal/tasks/ocr-extract",
        json={
            "jobId": "job-1",
            "taskKind": "OCR_EXTRACT",
            "attemptNo": 2,
            "enqueuedAtEpochMilli": 123,
            "callbackBaseUrl": "http://localhost:8080",
            "pipelineVersion": "ocr-lite-v1",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "accepted": True,
        "jobId": "job-1",
        "taskKind": "OCR_EXTRACT",
        "pipelineVersion": "ocr-lite-v1",
    }

    assert len(runner.messages) == 1
    message = runner.messages[0]
    assert message.job_id == "job-1"
    assert message.capability == "OCR_EXTRACT"
    assert message.task_kind == "OCR_EXTRACT"
    assert message.pipeline_version == "ocr-lite-v1"
    assert message.attempt_no == 2


def test_ocr_extract_route_rejects_wrong_task_kind():
    client = TestClient(create_app(runner=_FakeRunner()))

    response = client.post(
        "/internal/tasks/ocr-extract",
        json={
            "jobId": "job-1",
            "taskKind": "RAG",
            "pipelineVersion": "ocr-lite-v1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "taskKind must be OCR_EXTRACT"
