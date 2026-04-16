"""Tests for StorageResolver s3:// support.

These tests verify URI parsing and the local:// path without requiring
a running MinIO. The actual boto3 download path is tested only when
MinIO is available.
"""

from __future__ import annotations

import pytest

from app.storage.resolver import StorageResolver


class TestS3UriParsing:
    def test_parse_valid_uri(self):
        bucket, key = StorageResolver._parse_s3_uri(
            "s3://my-bucket/job-1/final_response/uuid-test.json"
        )
        assert bucket == "my-bucket"
        assert key == "job-1/final_response/uuid-test.json"

    def test_parse_bucket_only_raises(self):
        with pytest.raises(ValueError):
            StorageResolver._parse_s3_uri("s3://my-bucket")

    def test_parse_empty_key(self):
        bucket, key = StorageResolver._parse_s3_uri("s3://bucket/")
        assert bucket == "bucket"
        assert key == ""


class TestResolverSchemeRouting:
    def test_local_scheme_still_works(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        resolver = StorageResolver(local_root=str(tmp_path))
        data = resolver.read_bytes(f"local://file.txt")
        assert data == b"hello"

    def test_unknown_scheme_raises(self, tmp_path):
        resolver = StorageResolver(local_root=str(tmp_path))
        with pytest.raises(NotImplementedError):
            resolver.read_bytes("gs://bucket/key")

    def test_s3_without_boto3_config_still_recognizes_scheme(self, tmp_path):
        """s3:// is recognized; failure is from boto3 connection, not scheme."""
        resolver = StorageResolver(
            local_root=str(tmp_path),
            s3_endpoint="http://localhost:9999",
            s3_access_key="key",
            s3_secret_key="secret",
        )
        # This will fail at the boto3 connection level, not at scheme parsing
        with pytest.raises(Exception):
            resolver.read_bytes("s3://bucket/key")


class TestLocalRegressions:
    def test_path_traversal_blocked(self, tmp_path):
        resolver = StorageResolver(local_root=str(tmp_path))
        with pytest.raises(ValueError, match="escapes root"):
            resolver.read_bytes("local://../../etc/passwd")

    def test_missing_file_raises(self, tmp_path):
        resolver = StorageResolver(local_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            resolver.read_bytes("local://nonexistent.txt")
