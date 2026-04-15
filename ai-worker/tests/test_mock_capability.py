"""Happy-path test for the MOCK capability.

Does not touch Redis, core-api, or the filesystem. Exists so the critical
worker-side path (input → capability → output shape) is covered by at
least one fast unit test before later phases start adding real engines.
"""

from __future__ import annotations

import json

from app.capabilities.base import CapabilityInput, CapabilityInputArtifact
from app.capabilities.mock_processor import MockProcessor
from app.core.config import WorkerSettings


def test_mock_capability_echoes_input(monkeypatch):
    # remove the artificial delay so the test is instant
    import app.core.config as config_module

    monkeypatch.setattr(
        config_module,
        "_settings",
        WorkerSettings(mock_processing_delay_ms=0),
    )

    cap = MockProcessor()
    input_artifact = CapabilityInputArtifact(
        artifact_id="art-1",
        type="INPUT_TEXT",
        content=b"hello platform",
        content_type="text/plain",
    )
    result = cap.run(CapabilityInput(
        job_id="job-42",
        capability="MOCK",
        attempt_no=1,
        inputs=[input_artifact],
    ))

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.type == "FINAL_RESPONSE"
    assert output.content_type == "application/json"

    body = json.loads(output.content.decode("utf-8"))
    assert body["jobId"] == "job-42"
    assert body["capability"] == "MOCK"
    assert body["echoedArtifactId"] == "art-1"
    assert body["preview"] == "hello platform"


def test_mock_capability_handles_missing_input(monkeypatch):
    import app.core.config as config_module

    monkeypatch.setattr(
        config_module,
        "_settings",
        WorkerSettings(mock_processing_delay_ms=0),
    )

    cap = MockProcessor()
    result = cap.run(CapabilityInput(
        job_id="job-empty",
        capability="MOCK",
        attempt_no=1,
        inputs=[],
    ))

    assert len(result.outputs) == 1
    body = json.loads(result.outputs[0].content.decode("utf-8"))
    assert body["jobId"] == "job-empty"
    assert "note" in body
