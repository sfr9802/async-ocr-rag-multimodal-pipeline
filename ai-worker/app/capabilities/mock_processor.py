"""MOCK capability.

Purpose: keep the pipeline open-for-business while real capabilities are
being built. Reads the first INPUT artifact, wraps it in a small JSON
envelope, and returns it as a FINAL_RESPONSE artifact so the E2E test
can verify the bytes round-trip.
"""

from __future__ import annotations

import json
import logging
import time

from app.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.core.config import get_settings

log = logging.getLogger(__name__)


class MockProcessor(Capability):
    name = "MOCK"

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        settings = get_settings()
        if settings.mock_processing_delay_ms > 0:
            time.sleep(settings.mock_processing_delay_ms / 1000.0)

        first = input.inputs[0] if input.inputs else None
        if first is None:
            body = {
                "jobId": input.job_id,
                "capability": input.capability,
                "note": "no input artifact supplied; emitted empty mock response",
            }
        else:
            snippet = _preview(first.content)
            body = {
                "jobId": input.job_id,
                "capability": input.capability,
                "echoedArtifactId": first.artifact_id,
                "echoedArtifactType": first.type,
                "echoedBytes": len(first.content),
                "preview": snippet,
            }

        payload = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        log.info(
            "MOCK capability produced %d bytes for job %s",
            len(payload), input.job_id,
        )
        return CapabilityOutput(outputs=[
            CapabilityOutputArtifact(
                type="FINAL_RESPONSE",
                filename="mock-response.json",
                content_type="application/json",
                content=payload,
            )
        ])


def _preview(content: bytes, max_chars: int = 200) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary len={len(content)}>"
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
