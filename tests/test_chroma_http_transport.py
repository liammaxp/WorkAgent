from __future__ import annotations

import ast
import builtins
import dataclasses
import inspect
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_transport import (
    BoundedChromaHttpTransport,
    ChromaCollectionMissing,
    ChromaTransportClosed,
    ChromaTransportError,
    ChromaTransportProtocolError,
    ChromaTransportResponseError,
    ChromaTransportTimeout,
    ChromaTransportUnavailable,
    InvalidChromaTransportConfiguration,
    MAX_GET_OFFSET,
    _default_httpx_client_builder,
)


ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_SOURCE = ROOT / "backend" / "chroma_http_transport.py"
PROTECTED_CHROMA_ROOT = ROOT / "information" / "chroma"
COLLECTION_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def transport_config(timeout_seconds: float = 0.2) -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.LOCAL_HTTP,
        host="127.0.0.1",
        port=18124,
        ssl=False,
        timeout_seconds=timeout_seconds,
    )


def json_response(status: int, payload: Any) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"Content-Type": "application/json"},
        json=payload,
        request=httpx.Request("GET", "http://test.invalid"),
    )


def null_json_response(status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"Content-Type": "application/json"},
        content=b"null",
        request=httpx.Request("GET", "http://test.invalid"),
    )


def collection_response(name: str = "github_evidence") -> dict[str, Any]:
    return {
        "id": COLLECTION_ID,
        "name": name,
        "configuration_json": {},
        "tenant": "default_tenant",
        "database": "default_database",
        "log_position": 0,
        "version": 0,
        "metadata": None,
        "dimension": 2,
    }


class RecordingHttpClient:
    def __init__(self, outcomes: list[Any] | None = None):
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, path: str, *, json: Any) -> httpx.Response:
        self.calls.append({"method": method, "path": path, "json": json})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


class RecordingBuilder:
    def __init__(self, client: RecordingHttpClient | None = None):
        self.client = client or RecordingHttpClient()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> RecordingHttpClient:
        self.calls.append(kwargs)
        return self.client


def build_transport(*outcomes: Any, timeout_seconds: float = 0.2):
    client = RecordingHttpClient(list(outcomes))
    builder = RecordingBuilder(client)
    adapter = BoundedChromaHttpTransport(
        transport_config(timeout_seconds), client_builder=builder
    )
    return adapter, client, builder


def call_adapter(adapter: Any, operation: str, *args: Any, **kwargs: Any) -> Any:
    """Exercise the injected fake-HTTP unit boundary without Chroma discovery."""

    return getattr(adapter, operation)(*args, **kwargs)


def open_github_collection(*following_outcomes: Any):
    adapter, client, builder = build_transport(
        json_response(200, collection_response()), *following_outcomes
    )
    binding = call_adapter(adapter, "get_collection", "github_evidence")
    return adapter, binding, client, builder


def test_transport_module_import_and_adapter_construction_are_io_free(monkeypatch):
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any):
        calls.append("forbidden")
        raise AssertionError("transport construction must not perform I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    builder = RecordingBuilder()
    adapter = BoundedChromaHttpTransport(transport_config(), client_builder=builder)
    assert calls == []
    assert builder.calls and builder.client.calls == []
    assert adapter.get_transport_summary()["last_error_category"] == "none"


def test_transport_source_has_no_embedded_private_or_environment_fallback():
    source = TRANSPORT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "Persistent" + "Client" not in source
    assert "chromadb" not in source
    assert "load_" + "dotenv" not in source
    assert "os." + "environ" not in source
    assert "information/" + "chroma" not in source
    assert not any(
        isinstance(node, ast.Attribute) and node.attr.startswith("_session")
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("timeout_seconds", (0.1, 2.5, 30.0))
def test_single_validated_timeout_maps_to_every_httpx_dimension(timeout_seconds):
    adapter, _client, builder = build_transport(timeout_seconds=timeout_seconds)
    timeout = builder.calls[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == timeout_seconds
    assert timeout.read == timeout_seconds
    assert timeout.write == timeout_seconds
    assert timeout.pool == timeout_seconds
    summary = adapter.get_transport_summary()
    assert summary["timeout_enforced"] is True
    assert summary["timeout_dimensions"] == "connect,read,write,pool"
    assert summary["retry_policy"] == "none"


def test_default_httpx_builder_disables_retries_redirects_and_environment(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeHttpTransport:
        def __init__(self, **kwargs: Any):
            captured["transport"] = kwargs

    class FakeClient:
        def __init__(self, **kwargs: Any):
            captured["client"] = kwargs

    monkeypatch.setattr(httpx, "HTTPTransport", FakeHttpTransport)
    monkeypatch.setattr(httpx, "Client", FakeClient)
    timeout = httpx.Timeout(1.0)
    _default_httpx_client_builder(
        base_url="http://127.0.0.1:18124/api/v2",
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    assert captured["transport"] == {"retries": 0, "trust_env": False}
    assert captured["client"]["follow_redirects"] is False
    assert captured["client"]["trust_env"] is False
    assert captured["client"]["timeout"] is timeout


@pytest.mark.parametrize(
    "config",
    (
        ChromaDeploymentConfig(
            ChromaDeploymentMode.DISABLED, None, None, False, 5.0
        ),
        ChromaDeploymentConfig(
            ChromaDeploymentMode.LOCAL_HTTP, None, None, False, 5.0
        ),
    ),
)
def test_disabled_or_inconsistent_transport_config_fails_closed(config):
    with pytest.raises(InvalidChromaTransportConfiguration):
        BoundedChromaHttpTransport(config, client_builder=RecordingBuilder())


@pytest.mark.parametrize("timeout", (True, math.nan, math.inf, 0.09, 30.01))
def test_manually_invalid_transport_timeout_is_rejected(timeout):
    config = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP, "127.0.0.1", 18124, False, timeout
    )
    with pytest.raises(
        InvalidChromaTransportConfiguration,
        match="invalid_chroma_transport_timeout",
    ):
        BoundedChromaHttpTransport(config, client_builder=RecordingBuilder())


def test_unknown_semantic_collection_fails_before_network_access():
    adapter, client, _ = build_transport()
    with pytest.raises(ChromaTransportProtocolError, match="unknown_chroma_collection"):
        call_adapter(adapter, "get_collection", "not_registered")
    assert client.calls == []


def test_existing_collection_count_get_and_query_use_bounded_semantic_models():
    adapter, binding, client, _ = open_github_collection(
        json_response(200, 1),
        json_response(
            200,
            {
                "ids": ["evidence-1"],
                "include": ["metadatas"],
                "metadatas": [{"project_id": "project-1"}],
                "documents": None,
                "embeddings": None,
                "uris": None,
            },
        ),
        json_response(
            200,
            {
                "ids": [["evidence-1"]],
                "include": ["distances", "metadatas"],
                "distances": [[0.0]],
                "metadatas": [[{"project_id": "project-1"}]],
                "documents": None,
                "embeddings": None,
                "uris": None,
            },
        ),
    )
    count = binding.count()
    records = binding.get(ids=["evidence-1"])
    result = binding.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert count.value == 1
    assert records.ids == ("evidence-1",)
    assert dict(records.metadatas[0]) == {"project_id": "project-1"}
    assert result.ids == (("evidence-1",),)
    assert result.distances == ((0.0,),)
    assert dict(result.metadatas[0][0]) == {"project_id": "project-1"}
    assert [call["path"].rsplit("/", 1)[-1] for call in client.calls] == [
        "github_evidence",
        "count",
        "get",
        "query",
    ]
    assert "documents" not in json.dumps(records.safe_summary())
    assert "project-1" not in repr(records)
    assert "project-1" not in repr(result)
    assert adapter.get_transport_summary()["last_error_category"] == "none"


def test_get_requires_bounded_selector_and_rejects_embedding_or_uri_includes():
    adapter, binding, client, _ = open_github_collection()
    with pytest.raises(ChromaTransportProtocolError, match="selector_required"):
        binding.get()
    for include in (["embeddings"], ["uris"]):
        with pytest.raises(ChromaTransportProtocolError, match="unsupported_chroma_transport_include"):
            binding.get(ids=["one"], include=include)


def test_authorized_document_reads_are_bounded_and_absent_from_safe_summary():
    adapter, binding, client, _ = open_github_collection(
        json_response(
            200,
            {
                "ids": ["one"],
                "include": ["documents"],
                "metadatas": None,
                "documents": ["approved business content"],
                "embeddings": None,
                "uris": None,
            },
        )
    )
    records = binding.get(ids=["one"], include=["documents"])
    assert records.documents == ("approved business content",)
    assert "approved business content" not in repr(records)
    assert "approved business content" not in json.dumps(records.safe_summary())
    adapter.close()
    assert len(client.calls) == 2


def test_get_offset_supports_bounded_multi_page_collection_reads():
    adapter, binding, client, _ = open_github_collection(
        json_response(
            200,
            {
                "ids": [],
                "include": ["metadatas"],
                "metadatas": [],
                "documents": None,
                "embeddings": None,
                "uris": None,
            },
        )
    )
    assert binding.get(limit=1, offset=MAX_GET_OFFSET).ids == ()
    assert client.calls[-1]["json"]["offset"] == MAX_GET_OFFSET
    with pytest.raises(ChromaTransportProtocolError, match="invalid_chroma_transport_offset"):
        binding.get(limit=1, offset=MAX_GET_OFFSET + 1)


def test_adapter_exposes_only_bounded_mutations_and_no_collection_creation_surface():
    adapter, binding, _client, _ = open_github_collection()
    for target in (adapter, binding):
        for operation in (
            "create_collection",
            "get_or_create_collection",
            "list_collections",
            "add",
            "update",
            "peek",
        ):
            assert not hasattr(target, operation)
    assert hasattr(adapter, "upsert_records")
    assert hasattr(adapter, "delete_records")
    assert not hasattr(adapter, "upsert")
    assert not hasattr(adapter, "delete")
    assert hasattr(binding, "upsert")
    assert hasattr(binding, "delete")


def test_bounded_upsert_and_delete_preserve_payload_and_return_safe_counts():
    adapter, binding, client, _ = open_github_collection(
        null_json_response(),
        json_response(200, {"deleted": 1}),
    )
    upserted = binding.upsert(
        ids=["evidence-1"],
        embeddings=[[1.0, 0.0]],
        documents=["approved content"],
        metadatas=[{"project_id": "project-1"}],
    )
    deleted = binding.delete(ids=["evidence-1"])
    assert upserted.safe_summary() == {
        "semantic_collection_id": "github_evidence",
        "operation": "upsert",
        "requested_count": 1,
        "affected_count": 1,
    }
    assert deleted.affected_count == 1
    assert client.calls[-2]["path"].endswith("/upsert")
    assert client.calls[-2]["json"] == {
        "ids": ["evidence-1"],
        "embeddings": [[1.0, 0.0]],
        "documents": ["approved content"],
        "metadatas": [{"project_id": "project-1"}],
        "uris": None,
    }
    assert client.calls[-1]["path"].endswith("/delete")
    assert client.calls[-1]["json"] == {
        "ids": ["evidence-1"],
        "where": None,
        "where_document": None,
    }
    assert "approved content" not in repr(upserted)
    assert "project-1" not in repr(upserted)


def test_upsert_accepts_only_the_installed_empty_object_or_null_success_shapes():
    for response in (null_json_response(), json_response(200, {})):
        adapter, binding, _client, _ = open_github_collection(response)
        assert binding.upsert(
            ids=["one"],
            embeddings=[[1.0, 0.0]],
            documents=["content"],
            metadatas=[{}],
        ).affected_count == 1
    adapter, binding, _client, _ = open_github_collection(json_response(200, {"ok": True}))
    with pytest.raises(ChromaTransportProtocolError, match="invalid_chroma_upsert_response"):
        binding.upsert(
            ids=["one"],
            embeddings=[[1.0, 0.0]],
            documents=["content"],
            metadatas=[{}],
        )


@pytest.mark.parametrize(
    "kwargs,code",
    (
        (
            {
                "ids": ["duplicate", "duplicate"],
                "embeddings": [[1.0, 0.0], [1.0, 0.0]],
                "documents": ["one", "two"],
                "metadatas": [{}, {}],
            },
            "invalid_chroma_mutation_ids",
        ),
        (
            {
                "ids": ["one"],
                "embeddings": [[math.nan, 0.0]],
                "documents": ["private document"],
                "metadatas": [{"secret": "private metadata"}],
            },
            "invalid_chroma_mutation_embeddings",
        ),
        (
            {
                "ids": ["one"],
                "embeddings": [[1.0]],
                "documents": ["private document"],
                "metadatas": [{"secret": "private metadata"}],
            },
            "chroma_mutation_embedding_dimension_mismatch",
        ),
    ),
)
def test_mutation_validation_fails_before_network_without_exposing_content(kwargs, code):
    adapter, binding, client, _ = open_github_collection()
    before = len(client.calls)
    with pytest.raises(ChromaTransportProtocolError, match=f"^{code}$") as captured:
        binding.upsert(**kwargs)
    assert len(client.calls) == before
    assert "private document" not in str(captured.value)
    assert "private metadata" not in str(captured.value)


def test_delete_requires_explicit_ids_and_mutation_has_no_retry():
    adapter, binding, client, _ = open_github_collection(
        httpx.WriteTimeout("private", request=httpx.Request("POST", "http://private"))
    )
    with pytest.raises(ChromaTransportProtocolError, match="invalid_chroma_mutation_ids"):
        binding.delete(ids=[])
    with pytest.raises(ChromaTransportTimeout, match="^chroma_transport_timeout$"):
        binding.upsert(
            ids=["one"],
            embeddings=[[1.0, 0.0]],
            documents=["private document"],
            metadatas=[{"repository": "owner/repo"}],
        )
    assert len([call for call in client.calls if call["path"].endswith("/upsert")]) == 1


@pytest.mark.parametrize(
    "error,error_type,category",
    (
        (
            httpx.ReadTimeout(
                "secret response", request=httpx.Request("GET", "http://private")
            ),
            ChromaTransportTimeout,
            "timeout",
        ),
        (
            httpx.WriteTimeout(
                "secret response", request=httpx.Request("POST", "http://private")
            ),
            ChromaTransportTimeout,
            "timeout",
        ),
        (
            httpx.PoolTimeout(
                "secret response", request=httpx.Request("GET", "http://private")
            ),
            ChromaTransportTimeout,
            "timeout",
        ),
        (
            httpx.ConnectError(
                "C:/private", request=httpx.Request("GET", "http://private")
            ),
            ChromaTransportUnavailable,
            "unavailable",
        ),
        (
            httpx.ConnectTimeout(
                "C:/private", request=httpx.Request("GET", "http://private")
            ),
            ChromaTransportUnavailable,
            "unavailable",
        ),
        (
            httpx.RemoteProtocolError(
                "password=x", request=httpx.Request("GET", "http://private")
            ),
            ChromaTransportProtocolError,
            "protocol",
        ),
    ),
)
def test_transport_exceptions_map_to_safe_distinct_semantic_errors(
    error, error_type, category
):
    adapter, _client, _ = build_transport(error)
    with pytest.raises(error_type) as captured:
        call_adapter(adapter, "heartbeat")
    assert captured.value.__cause__ is None
    assert "private" not in str(captured.value).casefold()
    assert "secret" not in str(captured.value).casefold()
    assert "password" not in str(captured.value).casefold()
    assert adapter.get_transport_summary()["last_error_category"] == category


@pytest.mark.parametrize(
    "status,error_type,code",
    (
        (401, ChromaTransportResponseError, "chroma_transport_authority_response_error"),
        (422, ChromaTransportResponseError, "chroma_transport_client_response_error"),
        (503, ChromaTransportResponseError, "chroma_transport_server_response_error"),
    ),
)
def test_http_status_errors_are_not_converted_to_empty_results(status, error_type, code):
    adapter, _client, _ = build_transport(
        json_response(status, {"message": "sensitive raw response"})
    )
    with pytest.raises(error_type) as captured:
        call_adapter(adapter, "heartbeat")
    assert str(captured.value) == code
    assert "sensitive" not in str(captured.value)


def test_collection_404_is_distinct_and_existing_only():
    adapter, client, _ = build_transport(json_response(404, {"message": "missing"}))
    with pytest.raises(ChromaCollectionMissing, match="chroma_collection_missing"):
        call_adapter(adapter, "get_collection", "github_evidence")
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "response,code",
    (
        (
            httpx.Response(
                200,
                text="not-json",
                headers={"Content-Type": "text/plain"},
                request=httpx.Request("GET", "http://test.invalid"),
            ),
            "chroma_transport_non_json_response",
        ),
        (
            httpx.Response(
                200,
                content=b"{",
                headers={"Content-Type": "application/json"},
                request=httpx.Request("GET", "http://test.invalid"),
            ),
            "chroma_transport_malformed_json",
        ),
        (json_response(200, {"unexpected": True}), "invalid_chroma_heartbeat_response"),
    ),
)
def test_malformed_json_and_unexpected_schema_fail_safely(response, code):
    adapter, _client, _ = build_transport(response)
    with pytest.raises(ChromaTransportProtocolError, match=f"^{code}$"):
        call_adapter(adapter, "heartbeat")


def test_oversized_or_raw_content_responses_fail_without_exposure():
    oversized = httpx.Response(
        200,
        content=b"x" * 2_000_001,
        headers={"Content-Type": "application/json"},
        request=httpx.Request("GET", "http://test.invalid"),
    )
    adapter, _client, _ = build_transport(oversized)
    with pytest.raises(ChromaTransportProtocolError) as captured:
        call_adapter(adapter, "heartbeat")
    assert str(captured.value) == "chroma_transport_response_too_large"
    assert "x" not in str(captured.value)

    adapter, binding, _client, _ = open_github_collection(
        json_response(
            200,
            {
                "ids": ["one"],
                "include": ["metadatas"],
                "metadatas": [{"project_id": "project-1"}],
                "documents": ["raw source body"],
                "embeddings": None,
                "uris": None,
            },
        )
    )
    with pytest.raises(ChromaTransportProtocolError) as captured:
        binding.get(ids=["one"])
    assert str(captured.value) == "unsafe_chroma_get_response"
    assert "raw source body" not in str(captured.value)


def test_semantic_models_are_frozen_and_summaries_do_not_leak_protocol_data():
    _adapter, binding, _client, _ = open_github_collection()
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.collection_name = "changed"
    summary = binding.safe_summary()
    encoded = json.dumps(summary, sort_keys=True)
    assert "127.0.0.1" not in encoded
    assert COLLECTION_ID not in encoded
    assert "tenant" not in encoded
    assert "database" not in encoded
    assert "metadata" not in encoded


def test_closed_transport_rejects_requests_without_network_access():
    adapter, client, _ = build_transport()
    adapter.close()
    assert client.closed is True
    with pytest.raises(ChromaTransportClosed, match="chroma_transport_closed"):
        call_adapter(adapter, "heartbeat")
    assert client.calls == []


def test_transport_dependency_and_public_api_assumptions_are_explicit():
    import importlib.metadata

    assert importlib.metadata.version("httpx") == "0.28.1"
    assert "/api/{CHROMA_PUBLIC_HTTP_API}" in inspect.getsource(
        BoundedChromaHttpTransport.__init__
    )
    requirements = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
    assert "httpx" in requirements.splitlines()
