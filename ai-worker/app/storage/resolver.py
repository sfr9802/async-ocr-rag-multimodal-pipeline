"""Storage URI resolver.

The worker needs to read INPUT artifacts the core-api staged. This resolver
turns opaque storage URIs into actual bytes.

Supported schemes:
  - ``local://`` — shared local filesystem (phase 1 default)
  - ``s3://``    — S3/MinIO via boto3 (phase 2, requires boto3 + config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

LOCAL_SCHEME = "local://"
S3_SCHEME = "s3://"


class StorageResolver:
    def __init__(
        self,
        local_root: str,
        *,
        s3_endpoint: Optional[str] = None,
        s3_region: str = "us-east-1",
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
    ) -> None:
        self._local_root = Path(local_root).resolve()
        self._s3_endpoint = s3_endpoint
        self._s3_region = s3_region
        self._s3_access_key = s3_access_key
        self._s3_secret_key = s3_secret_key
        self._s3_client = None  # lazy

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
        if storage_uri.startswith(S3_SCHEME):
            return self._read_s3(storage_uri)
        raise NotImplementedError(
            f"Storage scheme not supported: {storage_uri!r}. "
            "Supported: local://, s3://"
        )

    def _resolve_local(self, uri: str) -> Path:
        key = uri[len(LOCAL_SCHEME):]
        path = (self._local_root / key).resolve()
        if not str(path).startswith(str(self._local_root)):
            raise ValueError(f"local URI escapes root: {uri}")
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing on shared disk: {path}")
        return path

    def _read_s3(self, uri: str) -> bytes:
        """Download bytes from S3/MinIO via boto3."""
        bucket, key = self._parse_s3_uri(uri)
        client = self._get_s3_client()
        response = client.get_object(Bucket=bucket, Key=key)
        try:
            return response["Body"].read()
        finally:
            response["Body"].close()

    def _get_s3_client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise RuntimeError(
                "boto3 is required for s3:// storage URIs. "
                "pip install boto3"
            )
        kwargs: dict = {
            "service_name": "s3",
            "region_name": self._s3_region,
            "config": BotoConfig(signature_version="s3v4"),
        }
        if self._s3_endpoint:
            kwargs["endpoint_url"] = self._s3_endpoint
        if self._s3_access_key and self._s3_secret_key:
            kwargs["aws_access_key_id"] = self._s3_access_key
            kwargs["aws_secret_access_key"] = self._s3_secret_key
        self._s3_client = boto3.client(**kwargs)
        log.info("S3 client initialized (endpoint=%s)", self._s3_endpoint or "AWS default")
        return self._s3_client

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        path = uri[len(S3_SCHEME):]
        slash = path.index("/")
        return path[:slash], path[slash + 1:]
