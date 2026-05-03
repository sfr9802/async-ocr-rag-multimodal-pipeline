"""DTOs for the core-api internal contract.

These must stay wire-compatible with the Spring DTOs in
`com.aipipeline.coreapi.job.adapter.in.web.dto.*`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---- claim ----

class ClaimRequest(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    worker_claim_token: str = Field(serialization_alias="workerClaimToken")
    attempt_no: int = Field(serialization_alias="attemptNo")

    model_config = {"populate_by_name": True}


class ClaimedInput(BaseModel):
    artifact_id: str = Field(alias="artifactId")
    source_file_id: Optional[str] = Field(default=None, alias="sourceFileId")
    type: str
    storage_uri: str = Field(alias="storageUri")
    content_type: Optional[str] = Field(default=None, alias="contentType")
    size_bytes: Optional[int] = Field(default=None, alias="sizeBytes")

    model_config = {"populate_by_name": True}


class ClaimResponse(BaseModel):
    granted: bool
    current_status: Optional[str] = Field(default=None, alias="currentStatus")
    reason: Optional[str] = None
    capability: Optional[str] = None
    attempt_no: int = Field(default=0, alias="attemptNo")
    inputs: list[ClaimedInput] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---- callback ----

CallbackOutcome = Literal["SUCCEEDED", "FAILED"]


class OutputArtifactPayload(BaseModel):
    type: str
    storage_uri: str = Field(serialization_alias="storageUri")
    content_type: Optional[str] = Field(default=None, serialization_alias="contentType")
    size_bytes: Optional[int] = Field(default=None, serialization_alias="sizeBytes")
    checksum_sha256: Optional[str] = Field(default=None, serialization_alias="checksumSha256")

    model_config = {"populate_by_name": True}


class CallbackRequest(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    callback_id: str = Field(serialization_alias="callbackId")
    worker_claim_token: str = Field(serialization_alias="workerClaimToken")
    outcome: CallbackOutcome
    error_code: Optional[str] = Field(default=None, serialization_alias="errorCode")
    error_message: Optional[str] = Field(default=None, serialization_alias="errorMessage")
    output_artifacts: list[OutputArtifactPayload] = Field(
        default_factory=list, serialization_alias="outputArtifacts"
    )

    model_config = {"populate_by_name": True}


class CallbackResponse(BaseModel):
    applied: bool
    duplicate: bool
    current_status: Optional[str] = Field(default=None, alias="currentStatus")

    model_config = {"populate_by_name": True}


# ---- SearchUnit indexing ----

class SearchUnitIndexClaimRequest(BaseModel):
    worker_id: str = Field(serialization_alias="workerId")
    batch_size: Optional[int] = Field(default=None, serialization_alias="batchSize")
    stale_after_seconds: Optional[int] = Field(
        default=None,
        serialization_alias="staleAfterSeconds",
    )

    model_config = {"populate_by_name": True}


class SearchUnitIndexDocument(BaseModel):
    search_unit_id: str = Field(alias="searchUnitId")
    claim_token: str = Field(alias="claimToken")
    index_id: str = Field(alias="indexId")
    source_file_id: str = Field(alias="sourceFileId")
    source_file_name: Optional[str] = Field(default=None, alias="sourceFileName")
    extracted_artifact_id: Optional[str] = Field(default=None, alias="extractedArtifactId")
    artifact_type: Optional[str] = Field(default=None, alias="artifactType")
    unit_type: str = Field(alias="unitType")
    unit_key: str = Field(alias="unitKey")
    title: Optional[str] = None
    section_path: Optional[str] = Field(default=None, alias="sectionPath")
    page_start: Optional[int] = Field(default=None, alias="pageStart")
    page_end: Optional[int] = Field(default=None, alias="pageEnd")
    text_content: str = Field(alias="textContent")
    content_sha256: str = Field(alias="contentSha256")
    metadata_json: Optional[Any] = Field(default=None, alias="metadataJson")
    index_metadata: dict[str, Any] = Field(default_factory=dict, alias="indexMetadata")

    model_config = {"populate_by_name": True}


class SearchUnitIndexClaimResponse(BaseModel):
    units: list[SearchUnitIndexDocument] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class SearchUnitIndexEmbeddedRequest(BaseModel):
    claim_token: str = Field(serialization_alias="claimToken")
    content_sha256: str = Field(serialization_alias="contentSha256")
    index_id: Optional[str] = Field(default=None, serialization_alias="indexId")

    model_config = {"populate_by_name": True}


class SearchUnitIndexFailedRequest(BaseModel):
    claim_token: str = Field(serialization_alias="claimToken")
    content_sha256: str = Field(serialization_alias="contentSha256")
    detail: Optional[str] = None

    model_config = {"populate_by_name": True}


class SearchUnitIndexCompletionResponse(BaseModel):
    applied: bool
    stale: bool = False
    index_id: Optional[str] = Field(default=None, alias="indexId")
    detail: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---- artifact upload ----

class UploadResponse(BaseModel):
    """Bytes-only upload response.

    The upload endpoint writes content to storage and returns a storage URI
    that the worker echoes back in the subsequent callback. No artifact row
    is created until the callback arrives. This matches the semantics of a
    real presigned-upload flow: the upload lands on object storage, and
    only the callback teaches the database that the object now exists.
    """

    storage_uri: str = Field(alias="storageUri")
    size_bytes: Optional[int] = Field(default=None, alias="sizeBytes")
    checksum_sha256: Optional[str] = Field(default=None, alias="checksumSha256")

    model_config = {"populate_by_name": True}
