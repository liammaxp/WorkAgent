from __future__ import annotations

import os
import time

import pytest

from backend.chroma_http_client_factory import (
    ChromaCollectionMissing,
    ChromaHttpClientFactory,
    ChromaTransportUnavailable,
)
from tests.chroma_http_test_support import (
    EphemeralChromaServer,
    allocate_dynamic_loopback_endpoint,
    ephemeral_deployment_config,
    prepare_registered_collection_for_test,
    query_collection_for_test,
    read_collection_for_test,
    wait_for_loopback_port_release,
)


pytestmark = pytest.mark.chroma_http_integration


@pytest.fixture(scope="module")
def running_chroma_server(tmp_path_factory):
    storage_parent = tmp_path_factory.mktemp("chroma-http-integration")
    server = EphemeralChromaServer(storage_parent)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="module")
def prepared_github_evidence(running_chroma_server):
    factory = ChromaHttpClientFactory(
        running_chroma_server.deployment_config(timeout_seconds=0.25),
        test_context=True,
    )
    prepare_registered_collection_for_test(
        running_chroma_server.endpoint,
        "github_evidence",
        ids=["synthetic-evidence"],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {
                "project_id": "synthetic-project",
                "repository": "synthetic/repository",
            }
        ],
    )
    return factory


def test_real_server_runs_in_a_separate_process_with_test_owned_endpoint(
    running_chroma_server,
):
    summary = running_chroma_server.safe_summary()
    assert running_chroma_server.ready is True
    assert running_chroma_server.process_id not in {None, os.getpid()}
    assert summary["process_state"] == "ready"
    assert summary["host_scope"] == "loopback"
    assert summary["test_owned"] is True
    assert summary["storage_scope"] == "temporary_test_owned"
    assert running_chroma_server.storage_path is not None
    assert running_chroma_server.storage_path.is_dir()


def test_central_factory_builds_real_http_client_lazily(prepared_github_evidence):
    factory = prepared_github_evidence
    assert factory.get_factory_summary()["client_cached"] is False
    client = factory.get_client()
    assert client is factory.get_client()
    assert factory.get_factory_summary()["factory_state"] == "ready"
    assert factory.get_factory_summary()["timeout_enforced"] is True


def test_existing_collection_read_lifecycle_succeeds_over_real_http(
    prepared_github_evidence,
):
    handle = prepared_github_evidence.get_collection_handle(
        "github_evidence",
        "read",
        "central_http_collection_factory",
    )
    result = read_collection_for_test(
        handle._collection_accessor,
        ids=["synthetic-evidence"],
    )
    assert handle.creation_allowed is False
    assert result.ids == ("synthetic-evidence",)
    assert result.metadatas == (
        {
            "project_id": "synthetic-project",
            "repository": "synthetic/repository",
        },
    )


def test_existing_collection_count_succeeds_over_real_http(prepared_github_evidence):
    handle = prepared_github_evidence.get_collection_handle(
        "github_evidence",
        "read",
        "central_http_collection_factory",
    )
    result = handle._collection_accessor.count()
    assert result.value == 1


def test_existing_collection_vector_lifecycle_succeeds_over_real_http(
    prepared_github_evidence,
):
    handle = prepared_github_evidence.get_collection_handle(
        "github_evidence",
        "vector_query",
        "github_evidence_vector_reader",
    )
    result = query_collection_for_test(
        handle._collection_accessor,
        query_embeddings=[[1.0, 0.0]],
    )
    assert result.ids == (("synthetic-evidence",),)
    assert result.distances[0][0] == pytest.approx(0.0)


def test_missing_collection_stays_missing_until_explicit_test_preparation(tmp_path):
    with EphemeralChromaServer(tmp_path) as server:
        factory = ChromaHttpClientFactory(
            server.deployment_config(timeout_seconds=0.25),
            test_context=True,
        )
        with pytest.raises(ChromaCollectionMissing, match="chroma_collection_missing"):
            factory.get_collection_handle(
                "profile_facts",
                "read",
                "central_http_collection_factory",
            )
        prepare_registered_collection_for_test(server.endpoint, "profile_facts")
        handle = factory.get_collection_handle(
            "profile_facts",
            "read",
            "central_http_collection_factory",
        )
        assert handle.collection_name == "profile_facts"
        assert handle.creation_allowed is False


def test_unavailable_test_owned_endpoint_fails_safely_and_boundedly():
    endpoint = allocate_dynamic_loopback_endpoint()
    configured_timeout = 0.1
    config = ephemeral_deployment_config(
        endpoint, timeout_seconds=configured_timeout
    )
    factory = ChromaHttpClientFactory(config, test_context=True)
    transport = factory.get_transport()
    bounded_scheduler_slack = 0.25
    elapsed_upper_bound = max(
        configured_timeout * 4.0,
        configured_timeout + bounded_scheduler_slack,
    )
    try:
        started = time.monotonic()
        with pytest.raises(
            ChromaTransportUnavailable,
            match="chroma_transport_unavailable",
        ) as captured:
            factory.get_collection_handle(
                "github_evidence",
                "read",
                "central_http_collection_factory",
            )
        elapsed = time.monotonic() - started
    finally:
        transport.close()
    assert elapsed < elapsed_upper_bound
    assert "127.0.0.1" not in str(captured.value)
    assert str(endpoint.port) not in str(captured.value)
    assert factory.get_factory_summary()["client_cached"] is True


def test_server_disappearing_after_client_creation_maps_to_safe_unavailable(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    server.start()
    endpoint = server.endpoint
    factory = ChromaHttpClientFactory(
        server.deployment_config(timeout_seconds=0.1),
        test_context=True,
    )
    prepare_registered_collection_for_test(endpoint, "github_evidence")
    factory.get_collection_handle(
        "github_evidence",
        "read",
        "central_http_collection_factory",
    )
    server.stop()
    assert wait_for_loopback_port_release(endpoint.port)
    started = time.monotonic()
    with pytest.raises(
        ChromaTransportUnavailable,
        match="chroma_transport_unavailable",
    ):
        factory.get_collection_handle(
            "github_evidence",
            "read",
            "central_http_collection_factory",
        )
    elapsed = time.monotonic() - started
    assert elapsed < 1.0


def test_real_assertion_failure_teardown_leaves_no_process_storage_or_bound_port(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    with pytest.raises(AssertionError, match="controlled integration assertion"):
        with server:
            endpoint = server.endpoint
            storage_path = server.storage_path
            raise AssertionError("controlled integration assertion")
    assert server.process_running is False
    assert storage_path is not None and not storage_path.exists()
    assert wait_for_loopback_port_release(endpoint.port)
