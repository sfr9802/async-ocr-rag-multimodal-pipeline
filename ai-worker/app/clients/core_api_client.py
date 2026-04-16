"""HTTP client for core-api.

Covers the three operations the worker needs:
  - POST /api/internal/jobs/claim
  - POST /api/internal/jobs/callback
  - POST /api/internal/artifacts (multipart upload)

Kept synchronous via httpx.Client because the per-job pipeline in this
worker is sequential — there is no benefit to asyncio in phase 1, and
a sync client is easier to reason about under the blocking BRPOP loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.clients.schemas import (
    CallbackRequest,
    CallbackResponse,
    ClaimRequest,
    ClaimResponse,
    UploadResponse,
)

log = logging.getLogger(__name__)


class CoreApiClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        internal_secret: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {}
        if internal_secret:
            headers["X-Internal-Secret"] = internal_secret
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_seconds,
            headers=headers,
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------

    def claim(self, request: ClaimRequest) -> ClaimResponse:
        payload = request.model_dump(by_alias=True)
        response = self._client.post("/api/internal/jobs/claim", json=payload)
        response.raise_for_status()
        return ClaimResponse.model_validate(response.json())

    def callback(self, request: CallbackRequest) -> CallbackResponse:
        payload = request.model_dump(by_alias=True)
        response = self._client.post("/api/internal/jobs/callback", json=payload)
        response.raise_for_status()
        return CallbackResponse.model_validate(response.json())

    def upload_output_artifact(
        self,
        job_id: str,
        artifact_type: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> UploadResponse:
        files = {"file": (filename, content, content_type)}
        data = {"jobId": job_id, "type": artifact_type}
        response = self._client.post("/api/internal/artifacts", data=data, files=files)
        response.raise_for_status()
        return UploadResponse.model_validate(response.json())

    def download_artifact_content(self, artifact_id: str) -> bytes:
        """Fallback reader for INPUT artifacts.

        In phase 1 local mode the worker can reach shared disk directly
        (see StorageResolver) — this is here for the S3/MinIO path or
        whenever the worker runs on a different host than core-api.
        """
        response = self._client.get(f"/api/v1/artifacts/{artifact_id}/content")
        response.raise_for_status()
        return response.content
