"""Tests for X-Internal-Secret header injection in CoreApiClient."""

from app.clients.core_api_client import CoreApiClient


class TestInternalSecretHeader:
    def test_secret_provided_sets_default_header(self):
        client = CoreApiClient(
            base_url="http://localhost:8080",
            timeout_seconds=5.0,
            internal_secret="my-secret",
        )
        assert client._client.headers["X-Internal-Secret"] == "my-secret"
        client.close()

    def test_no_secret_omits_header(self):
        client = CoreApiClient(
            base_url="http://localhost:8080",
            timeout_seconds=5.0,
        )
        assert "X-Internal-Secret" not in client._client.headers
        client.close()

    def test_none_secret_omits_header(self):
        client = CoreApiClient(
            base_url="http://localhost:8080",
            timeout_seconds=5.0,
            internal_secret=None,
        )
        assert "X-Internal-Secret" not in client._client.headers
        client.close()

    def test_empty_string_secret_omits_header(self):
        client = CoreApiClient(
            base_url="http://localhost:8080",
            timeout_seconds=5.0,
            internal_secret="",
        )
        assert "X-Internal-Secret" not in client._client.headers
        client.close()
