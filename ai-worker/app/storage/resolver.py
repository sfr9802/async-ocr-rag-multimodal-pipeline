"""Storage URI resolver.

The worker needs to read INPUT artifacts the core-api staged, and (in phase
1) those bytes live on a shared local filesystem. This resolver turns a
`local://...` URI into an actual path relative to the worker's configured
storage root.

When MinIO/S3 arrives, this is the module that grows an `s3://` branch.
Everything else in the worker stays oblivious.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

LOCAL_SCHEME = "local://"


class StorageResolver:
    def __init__(self, local_root: str) -> None:
        self._local_root = Path(local_root).resolve()

    def read_bytes(self, storage_uri: str, fallback_http_reader=None) -> bytes:
        """Read the bytes behind a storage URI.

        :param storage_uri: opaque URI produced by core-api
        :param fallback_http_reader: callable(artifact_id) -> bytes, used
            when the scheme isn't locally resolvable (e.g. the worker runs
            on a different machine than core-api)
        """
        if storage_uri.startswith(LOCAL_SCHEME):
            path = self._resolve_local(storage_uri)
            return path.read_bytes()
        raise NotImplementedError(
            f"Storage scheme not supported by this worker build: {storage_uri!r}. "
            "Remote readers will be added when MinIO/S3 support lands."
        )

    def _resolve_local(self, uri: str) -> Path:
        key = uri[len(LOCAL_SCHEME):]
        path = (self._local_root / key).resolve()
        if not str(path).startswith(str(self._local_root)):
            raise ValueError(f"local URI escapes root: {uri}")
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing on shared disk: {path}")
        return path
