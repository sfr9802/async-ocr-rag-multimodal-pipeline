"""FastAPI HTTP entrypoints for managed task dispatch."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.capabilities.ocr.artifact_builder import OCR_LITE_PIPELINE_VERSION
from app.capabilities.registry import build_default_registry
from app.clients.core_api_client import CoreApiClient
from app.core.config import WorkerSettings, get_settings
from app.queue.messages import QueueMessage
from app.services.task_runner import CallbackDeliveryError, TaskRunner
from app.storage.resolver import StorageResolver

log = logging.getLogger(__name__)


class OcrExtractTaskRequest(BaseModel):
    job_id: str = Field(alias="jobId")
    task_kind: str = Field(default="OCR_EXTRACT", alias="taskKind")
    attempt_no: int = Field(default=1, alias="attemptNo")
    enqueued_at_epoch_milli: int = Field(default=0, alias="enqueuedAtEpochMilli")
    callback_base_url: str = Field(default="", alias="callbackBaseUrl")
    pipeline_version: str = Field(
        default=OCR_LITE_PIPELINE_VERSION,
        alias="pipelineVersion",
    )

    model_config = {"populate_by_name": True}


def create_app(
    *,
    runner: Optional[TaskRunner] = None,
    settings: Optional[WorkerSettings] = None,
) -> FastAPI:
    app = FastAPI(title="ai-worker", version="0.1.0")
    app.state.runner = runner
    app.state.settings = settings

    @app.post(
        "/internal/tasks/ocr-extract",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_ocr_extract_task(body: OcrExtractTaskRequest) -> dict[str, object]:
        if body.task_kind != "OCR_EXTRACT":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="taskKind must be OCR_EXTRACT",
            )
        if body.pipeline_version != OCR_LITE_PIPELINE_VERSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"pipelineVersion must be {OCR_LITE_PIPELINE_VERSION}",
            )

        message = QueueMessage(
            jobId=body.job_id,
            capability=body.task_kind,
            taskKind=body.task_kind,
            pipelineVersion=body.pipeline_version,
            attemptNo=body.attempt_no,
            enqueuedAtEpochMilli=body.enqueued_at_epoch_milli,
            callbackBaseUrl=body.callback_base_url,
        )
        try:
            _get_runner(app).handle(message)
        except CallbackDeliveryError as ex:
            log.warning("OCR_EXTRACT callback delivery failed: %s", ex)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="callback delivery failed",
            ) from ex

        return {
            "accepted": True,
            "jobId": body.job_id,
            "taskKind": body.task_kind,
            "pipelineVersion": body.pipeline_version,
        }

    return app


def _get_runner(app: FastAPI) -> TaskRunner:
    if app.state.runner is not None:
        return app.state.runner

    settings = app.state.settings or get_settings()
    core_api = CoreApiClient(
        base_url=settings.core_api_base_url,
        timeout_seconds=settings.core_api_request_timeout_seconds,
        internal_secret=settings.internal_secret,
    )
    registry = build_default_registry(settings)
    resolver = StorageResolver(
        local_root=settings.local_storage_root,
        s3_endpoint=settings.s3_endpoint,
        s3_region=settings.s3_region,
        s3_access_key=settings.s3_access_key,
        s3_secret_key=settings.s3_secret_key,
    )
    app.state.runner = TaskRunner(
        core_api=core_api,
        registry=registry,
        resolver=resolver,
        worker_id=settings.worker_id,
    )
    app.state.core_api = core_api
    return app.state.runner
