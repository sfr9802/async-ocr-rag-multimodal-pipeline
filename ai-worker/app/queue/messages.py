"""Queue message schemas.

Must stay wire-compatible with core-api's
`com.aipipeline.coreapi.queue.adapter.out.redis.QueueMessage`.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueueMessage(BaseModel):
    """Dispatch message as serialized by core-api."""

    job_id: str = Field(alias="jobId")
    capability: str
    task_kind: Optional[str] = Field(default=None, alias="taskKind")
    pipeline_version: Optional[str] = Field(default=None, alias="pipelineVersion")
    attempt_no: int = Field(alias="attemptNo")
    enqueued_at_epoch_milli: int = Field(alias="enqueuedAtEpochMilli")
    callback_base_url: str = Field(alias="callbackBaseUrl")

    model_config = {"populate_by_name": True}
