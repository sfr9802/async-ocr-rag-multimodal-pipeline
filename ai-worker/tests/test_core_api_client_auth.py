"""Tests for X-Internal-Secret header injection in CoreApiClient."""

from app.clients.core_api_client import CoreApiClient
from app.clients.schemas import (
    SearchUnitIndexClaimRequest,
    SearchUnitIndexEmbeddedRequest,
    SearchUnitIndexFailedRequest,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self):
        self.posts = []
        self.headers = {}

    def post(self, path, json=None, **kwargs):  # noqa: A002 - matches httpx
        self.posts.append((path, json, kwargs))
        if path.endswith("/claim"):
            return _FakeResponse({
                "units": [
                    {
                        "searchUnitId": "unit-1",
                        "claimToken": "claim-1",
                        "indexId": "source_file:source-1:unit:PAGE:page:1",
                        "sourceFileId": "source-1",
                        "unitType": "PAGE",
                        "unitKey": "page:1",
                        "textContent": "page text",
                        "contentSha256": "hash-1",
                        "indexMetadata": {"search_unit_id": "unit-1"},
                    }
                ]
            })
        return _FakeResponse({"applied": True, "stale": False, "indexId": "idx"})

    def close(self):
        return None


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

    def test_search_unit_indexing_contract_methods_use_internal_endpoints(self):
        client = CoreApiClient(base_url="http://localhost:8080", timeout_seconds=5.0)
        fake = _FakeHttpClient()
        client._client = fake

        claim = client.claim_search_unit_indexing(
            SearchUnitIndexClaimRequest(
                worker_id="worker-1",
                batch_size=2,
                stale_after_seconds=60,
            )
        )
        embedded = client.mark_search_unit_embedded(
            "unit-1",
            SearchUnitIndexEmbeddedRequest(
                claim_token="claim-1",
                content_sha256="hash-1",
                index_id="source_file:source-1:unit:PAGE:page:1",
            ),
        )
        failed = client.mark_search_unit_indexing_failed(
            "unit-1",
            SearchUnitIndexFailedRequest(
                claim_token="claim-1",
                content_sha256="hash-1",
                detail="boom",
            ),
        )

        assert claim.units[0].search_unit_id == "unit-1"
        assert claim.units[0].index_metadata["search_unit_id"] == "unit-1"
        assert embedded.applied is True
        assert failed.applied is True
        assert fake.posts[0][0] == "/api/internal/search-units/indexing/claim"
        assert fake.posts[0][1] == {
            "workerId": "worker-1",
            "batchSize": 2,
            "staleAfterSeconds": 60,
        }
        assert fake.posts[1][0] == "/api/internal/search-units/indexing/unit-1/embedded"
        assert fake.posts[1][1]["claimToken"] == "claim-1"
        assert fake.posts[2][0] == "/api/internal/search-units/indexing/unit-1/failed"
        assert fake.posts[2][1]["detail"] == "boom"
