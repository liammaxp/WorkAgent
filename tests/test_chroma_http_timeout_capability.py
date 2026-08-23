from __future__ import annotations

import inspect
import socket
import time
from pathlib import Path

import pytest
from chromadb.config import Settings
from backend.chroma_http_client_factory import (
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import ChromaTransportTimeout
from tests.chroma_http_test_support import (
    DelayedLoopbackHttpServer,
    construct_public_http_client_for_timeout_probe,
    ephemeral_deployment_config,
    inspect_public_http_timeout_capability,
)


pytestmark = pytest.mark.chroma_http_integration
ARCHITECTURE_DOCUMENT = (
    Path(__file__).resolve().parents[1] / "docs" / "chroma_local_server_architecture.md"
)


def test_installed_public_http_timeout_capability_is_recorded_deterministically():
    capability = inspect_public_http_timeout_capability()
    assert capability.installed_version == "1.5.9"
    assert capability.http_client_parameters == (
        "host",
        "port",
        "ssl",
        "headers",
        "settings",
        "tenant",
        "database",
    )
    assert "timeout" not in capability.http_client_parameters
    assert capability.settings_http_fields == (
        "chroma_http_keepalive_secs",
        "chroma_http_max_connections",
        "chroma_http_max_keepalive_connections",
        "chroma_server_http_port",
    )
    assert capability.settings_timeout_fields == (
        "chroma_logservice_request_timeout_seconds",
        "chroma_query_request_timeout_seconds",
        "chroma_sysdb_request_timeout_seconds",
    )
    assert capability.timeout_support == "unsupported_by_current_public_client_api"
    assert capability.public_mechanism == "none"
    assert capability.production_migration_gate == "blocked"
    assert capability.required_work_item == "Bounded Chroma HTTP Transport Adapter"


def test_public_http_client_signature_and_settings_offer_no_general_request_timeout():
    import chromadb

    signature = inspect.signature(chromadb.HttpClient)
    assert "timeout" not in signature.parameters
    public_fields = Settings.model_fields
    assert "chroma_http_request_timeout_seconds" not in public_fields
    assert not {
        name
        for name in public_fields
        if "http" in name.casefold() and "timeout" in name.casefold()
    }


def test_workagent_transport_interrupts_controlled_delayed_response():
    configured_timeout = 0.2
    response_delay = 1.0
    default_socket_timeout = socket.getdefaulttimeout()
    with DelayedLoopbackHttpServer(delay_seconds=response_delay) as delayed:
        factory = ChromaHttpClientFactory(
            ephemeral_deployment_config(
                delayed.endpoint,
                timeout_seconds=configured_timeout,
            ),
            test_context=True,
        )
        factory.get_transport()
        started = time.monotonic()
        with pytest.raises(ChromaTransportTimeout, match="chroma_transport_timeout") as captured:
            factory.get_collection_handle(
                "github_evidence",
                "read",
                "central_http_collection_factory",
            )
        elapsed = time.monotonic() - started
    assert elapsed >= configured_timeout * 0.75
    assert elapsed < response_delay * 0.8
    assert socket.getdefaulttimeout() == default_socket_timeout
    assert "127.0.0.1" not in str(captured.value)
    assert factory.get_transport().get_transport_summary()["last_error_category"] == "timeout"


def test_public_query_timeout_setting_does_not_bound_http_transport_response():
    configured_query_timeout = 1
    response_delay = 1.2
    with DelayedLoopbackHttpServer(delay_seconds=response_delay) as delayed:
        started = time.monotonic()
        with pytest.raises(ValueError):
            construct_public_http_client_for_timeout_probe(
                delayed.endpoint,
                query_timeout_seconds=configured_query_timeout,
            )
        elapsed = time.monotonic() - started
    assert elapsed >= response_delay * 0.8
    assert elapsed > configured_query_timeout
    assert elapsed < 4.0


def test_timeout_investigation_source_uses_no_private_transport_patch_or_global_socket_mutation():
    source = Path(__file__).read_text(encoding="utf-8").casefold()
    forbidden = (
        "." + "_session",
        "setdefault" + "timeout",
        "monkey" + "patch",
        "httpx." + "client",
        "socket." + "defaulttimeout",
    )
    assert all(token not in source for token in forbidden)
    assert "bounded chroma http transport adapter" in source


def test_architecture_document_records_capability_outcome_and_migration_gate():
    document = ARCHITECTURE_DOCUMENT.read_text(encoding="utf-8")
    assert "Chroma `1.5.9`" in document
    assert "`unsupported_by_current_public_client_api`" in document
    assert "`Bounded Chroma HTTP Transport Adapter`" in document
    assert "`transport_timeout_support = enforced`" in document
    assert "`production_consumer_migration_gate = transport_ready`" in document
