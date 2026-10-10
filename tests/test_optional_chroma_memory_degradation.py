from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
from backend.chroma_config import (  # noqa: E402
    CHROMA_HTTP_HOST_ENV,
    CHROMA_HTTP_PORT_ENV,
    CHROMA_HTTP_SSL_ENV,
    CHROMA_HTTP_TIMEOUT_ENV,
    ChromaDeploymentConfig,
    ChromaDeploymentMode,
    load_chroma_deployment_config,
)
from backend.chroma_http_client_factory import (  # noqa: E402
    ChromaFactoryDisabled,
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import (  # noqa: E402
    ChromaCollectionMissing,
    ChromaTransportTimeout,
    ChromaTransportUnavailable,
)
from backend.chroma_operational_reader import ChromaOperationalReader  # noqa: E402
from backend.chroma_read_client import ChromaReadClient  # noqa: E402
from backend.chroma_write_client import ChromaWriteClient  # noqa: E402
from backend.chroma_write_models import ChromaWriteRecord  # noqa: E402
from memory_store import MemoryVectorStore  # noqa: E402
from tests.chroma_http_test_support import (  # noqa: E402
    DelayedLoopbackHttpServer,
    EphemeralChromaServer,
    allocate_dynamic_loopback_endpoint,
    is_loopback_port_releasable,
    prepare_registered_collection_for_test,
)


class FailingMemoryReadClient:
    def __init__(self, error):
        self.error = error

    def read_records(self, *_args, **_kwargs):
        raise self.error

    def vector_query(self, *_args, **_kwargs):
        raise self.error


def make_store(tmp_path, *, read_client=None):
    return MemoryVectorStore(
        tmp_path / "chroma",
        tmp_path / "memory.json",
        tmp_path / "github",
        read_client=read_client,
    )


def local_http_config(endpoint, *, timeout_seconds=0.2):
    return ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        endpoint.host,
        endpoint.port,
        False,
        timeout_seconds,
    )


def profile_record():
    return ChromaWriteRecord(
        "required-profile-record",
        'Profile memory section: summary\n"required"',
        {"section": "summary", "index": 0, "is_list": 0},
        (1.0, 0.0),
    )


def test_github_scan_with_optional_memory_source_survives_disabled_chroma(monkeypatch):
    monkeypatch.setenv("CHROMA_DEPLOYMENT_MODE", "disabled")

    with TestClient(api_server.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/github/scan",
            json={"resume_source": "memory"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "repos": [],
        "token_configured": api_server.agent.github_token_is_configured(),
        "identities": api_server.agent.read_github_identities(),
        "project_name": "",
        "project_id": "",
    }
    assert "ChromaFactoryDisabled" not in response.text


def test_default_disabled_optional_read_is_semantic_and_strict_read_still_rejects(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHROMA_DEPLOYMENT_MODE", "disabled")
    for key in (
        CHROMA_HTTP_HOST_ENV,
        CHROMA_HTTP_PORT_ENV,
        CHROMA_HTTP_SSL_ENV,
        CHROMA_HTTP_TIMEOUT_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    store = make_store(tmp_path)

    result = store.read_profile_optional()

    assert result.safe_summary() == {
        "state": "unavailable",
        "reason": "chroma_disabled",
        "field_count": 0,
    }
    assert result.available is False
    with pytest.raises(ChromaFactoryDisabled, match="chroma_factory_disabled"):
        store.read_profile()
    factory = ChromaHttpClientFactory(load_chroma_deployment_config({}))
    with pytest.raises(ChromaFactoryDisabled, match="chroma_factory_disabled"):
        factory.get_transport()


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (ChromaTransportUnavailable("private endpoint"), "chroma_unavailable"),
        (ChromaTransportTimeout("private endpoint"), "chroma_timeout"),
        (ChromaCollectionMissing("private collection"), "chroma_collection_missing"),
    ),
)
def test_expected_optional_read_failures_have_bounded_semantic_results(
    tmp_path,
    error,
    reason,
):
    store = make_store(tmp_path, read_client=FailingMemoryReadClient(error))

    result = store.read_profile_optional()

    assert result.state == "unavailable"
    assert result.reason == reason
    assert dict(result.memory) == {}
    assert "private" not in repr(result)


def test_unexpected_optional_read_error_is_not_swallowed(tmp_path):
    store = make_store(
        tmp_path,
        read_client=FailingMemoryReadClient(RuntimeError("programming bug")),
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        store.read_profile_optional()


def test_unexpected_memory_error_remains_observable_at_consumers(
    monkeypatch,
    tmp_path,
):
    store = make_store(
        tmp_path,
        read_client=FailingMemoryReadClient(RuntimeError("programming bug")),
    )
    monkeypatch.setattr(api_server.agent, "MEMORY_STORE", store)

    with pytest.raises(RuntimeError, match="programming bug"):
        api_server.read_user_memory_for_skills()
    with TestClient(api_server.app, raise_server_exceptions=False) as client:
        response = client.post("/api/github/scan", json={"resume_source": "memory"})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "programming bug" not in response.text


def test_local_http_unavailable_optional_memory_degrades_and_github_scan_continues(
    monkeypatch,
    tmp_path,
):
    endpoint = allocate_dynamic_loopback_endpoint()
    deployment = local_http_config(endpoint)
    reader = ChromaReadClient(config_provider=lambda: deployment)
    store = make_store(tmp_path, read_client=reader)

    result = store.read_profile_optional()

    assert result.state == "unavailable"
    assert result.reason == "chroma_unavailable"
    assert is_loopback_port_releasable(endpoint.port)
    monkeypatch.setattr(api_server.agent, "MEMORY_STORE", store)
    with TestClient(api_server.app, raise_server_exceptions=False) as client:
        response = client.post("/api/github/scan", json={"resume_source": "memory"})
    assert response.status_code == 200
    assert response.json()["repos"] == []
    assert "ChromaTransportUnavailable" not in response.text
    assert endpoint.host not in response.text


@pytest.mark.chroma_http_integration
def test_controlled_local_http_timeout_degrades_optional_memory_and_scan_continues(
    monkeypatch,
    tmp_path,
):
    with DelayedLoopbackHttpServer(delay_seconds=1.0) as delayed:
        delayed_host = delayed.endpoint.host
        deployment = local_http_config(delayed.endpoint, timeout_seconds=0.2)
        reader = ChromaReadClient(config_provider=lambda: deployment)
        store = make_store(tmp_path, read_client=reader)

        result = store.read_profile_optional()

        assert result.state == "unavailable"
        assert result.reason == "chroma_timeout"
        monkeypatch.setattr(api_server.agent, "MEMORY_STORE", store)
        with TestClient(api_server.app, raise_server_exceptions=False) as client:
            response = client.post("/api/github/scan", json={"resume_source": "memory"})
    assert response.status_code == 200
    assert response.json()["repos"] == []
    assert "ChromaTransportTimeout" not in response.text
    assert delayed_host not in response.text


def test_required_write_and_readiness_remain_fail_closed_when_disabled(tmp_path):
    deployment = load_chroma_deployment_config({})
    writer = ChromaWriteClient(config_provider=lambda: deployment)
    status = ChromaOperationalReader(
        config_provider=lambda: deployment
    ).read_collection_status("profile_facts")

    with pytest.raises(ChromaFactoryDisabled, match="chroma_factory_disabled"):
        writer.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[profile_record()],
        )
    assert status.available is False
    assert status.safe_record_count is None
    assert status.server_state == "unavailable"


def test_required_write_fails_when_local_http_is_unavailable():
    endpoint = allocate_dynamic_loopback_endpoint()
    deployment = local_http_config(endpoint)
    writer = ChromaWriteClient(config_provider=lambda: deployment)

    with pytest.raises(ChromaTransportUnavailable, match="chroma_transport_unavailable"):
        writer.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[profile_record()],
        )
    assert is_loopback_port_releasable(endpoint.port)


@pytest.mark.chroma_http_integration
def test_healthy_ephemeral_http_memory_has_data_without_degradation(
    monkeypatch,
    tmp_path,
):
    with EphemeralChromaServer(tmp_path) as server:
        deployment = server.deployment_config(timeout_seconds=1.0)
        factory_builder = lambda config: ChromaHttpClientFactory(config, test_context=True)
        writer = ChromaWriteClient(
            config_provider=lambda: deployment,
            factory_builder=factory_builder,
        )
        with pytest.raises(ChromaCollectionMissing, match="chroma_collection_missing"):
            writer.upsert_records(
                "profile_facts",
                consumer_id="profile_memory_indexer",
                records=[profile_record()],
            )

        document = 'Profile memory section: github\n"https://github.com/owner/healthy"'
        prepare_registered_collection_for_test(
            server.endpoint,
            "profile_facts",
            ids=[MemoryVectorStore._record_id("profile", "github")],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"section": "github", "index": 0, "is_list": 0}],
            documents=[document],
        )
        reader = ChromaReadClient(
            config_provider=lambda: deployment,
            factory_builder=factory_builder,
        )
        store = make_store(tmp_path, read_client=reader)

        result = store.read_profile_optional()

        assert result.state == "ready"
        assert result.reason is None
        assert dict(result.memory) == {"github": "https://github.com/owner/healthy"}
        monkeypatch.setattr(api_server.agent, "MEMORY_STORE", store)
        rendered_memory = api_server.agent.read_memory()
        assert json.loads(rendered_memory) == dict(result.memory)
        assert "unavailable" not in rendered_memory
        with TestClient(api_server.app, raise_server_exceptions=False) as client:
            response = client.post("/api/github/scan", json={"resume_source": "memory"})
        assert response.status_code == 200
        assert response.json()["repos"] == [
            {
                "owner": "owner",
                "repo": "healthy",
                "url": "https://github.com/owner/healthy",
            }
        ]
