"""Capability interface and data contracts.

A Capability is a function-shaped object that takes `CapabilityInput` and
returns `CapabilityOutput`. The task runner handles everything around it
(claim, artifact I/O, callback) so capabilities can stay focused on
"given bytes / text, produce bytes / text".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CapabilityInputArtifact:
    """A single input artifact already fetched into memory.

    `filename` is a best-effort display name extracted from the storage
    URI by the TaskRunner. It is None when core-api did not embed a
    filename in the URI (which it does for every multipart upload in
    phase 1). Capabilities MAY use it for metadata envelopes but must
    never treat it as an authoritative identifier — `artifact_id` is
    the only stable handle.
    """
    artifact_id: str
    type: str                     # ArtifactType on the Spring side
    content: bytes
    content_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass(frozen=True)
class CapabilityInput:
    job_id: str
    capability: str
    attempt_no: int
    inputs: list[CapabilityInputArtifact]


@dataclass(frozen=True)
class CapabilityOutputArtifact:
    """An artifact the capability wants the runner to upload + register."""
    type: str                     # e.g. FINAL_RESPONSE
    filename: str
    content_type: str
    content: bytes


@dataclass(frozen=True)
class CapabilityOutput:
    outputs: list[CapabilityOutputArtifact] = field(default_factory=list)


class CapabilityError(Exception):
    """Raised by a capability to signal a clean FAILED callback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Capability(ABC):
    """Base interface every capability implements."""

    #: Capability name as seen by core-api (matches JobCapability enum).
    name: str = ""

    @abstractmethod
    def run(self, input: CapabilityInput) -> CapabilityOutput:
        ...
