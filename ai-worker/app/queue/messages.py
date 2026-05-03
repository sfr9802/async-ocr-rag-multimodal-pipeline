"""Queue message schemas.

Must stay wire-compatible with core-api's
`com.aipipeline.coreapi.queue.adapter.out.redis.QueueMessage`.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DispatchInputArtifact(BaseModel):
    """Non-secret input artifact metadata carried on the queue signal."""

    artifact_id: str = Field(alias="artifactId")
    role: str
    type: str
    storage_uri: str = Field(alias="storageUri")
    content_type: Optional[str] = Field(default=None, alias="contentType")
    size_bytes: Optional[int] = Field(default=None, alias="sizeBytes")
    checksum_sha256: Optional[str] = Field(default=None, alias="checksumSha256")

    model_config = {"populate_by_name": True}


class QueueMessage(BaseModel):
    """Dispatch message as serialized by core-api."""

    job_id: str = Field(alias="jobId")
    capability: str
    task_kind: Optional[str] = Field(default=None, alias="taskKind")
    pipeline_version: Optional[str] = Field(default=None, alias="pipelineVersion")
    attempt_no: int = Field(alias="attemptNo")
    enqueued_at_epoch_milli: int = Field(alias="enqueuedAtEpochMilli")
    callback_base_url: str = Field(alias="callbackBaseUrl")
    input_artifacts: list[DispatchInputArtifact] = Field(default_factory=list, alias="inputArtifacts")

    model_config = {"populate_by_name": True}
