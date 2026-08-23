from __future__ import annotations

import ast
import builtins
import dataclasses
import json
import os
import socket
from pathlib import Path

import pytest

from backend import chroma_http_client_factory as factory_module
from backend.chroma_access_manifest import INVENTORY
from backend.chroma_collection_literal_guard import validate_collection_name_literals
from backend.chroma_collection_registry import validate_inventory_against_collection_registry
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaCollectionAuthorityMismatch,
    ChromaCollectionCreationNotAllowed,
    ChromaCollectionHandle,
    ChromaCollectionMissing,
    ChromaConsumerAuthority,
    ChromaConsumerNotApproved,
    ChromaFactoryDisabled,
    ChromaFactoryState,
    ChromaHttpClientFactory,
    ChromaTransportError,
    ChromaTransportUnavailable,
    InvalidChromaFactoryConfiguration,
    UnknownChromaCollection,
    UnknownChromaConsumer,
    UnsupportedChromaLifecycle,
    ValidatedChromaCollectionAccess,
)
from backend.chroma_http_transport import (
    ChromaTransportMutationResult,
    ChromaTransportRecords,
)


ROOT = Path(__file__).resolve().parents[1]
FACTORY_SOURCE = ROOT / "backend" / "chroma_http_client_factory.py"
TRANSPORT_SOURCE = ROOT / "backend" / "chroma_http_transport.py"


def disabled_config() -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.DISABLED,
        host=None,
        port=None,
        ssl=False,
        timeout_seconds=5.0,
    )


def local_config() -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.LOCAL_HTTP,
        host="127.0.0.1",
        port=8100,
        ssl=False,
        timeout_seconds=2.5,
    )


def remote_config() -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.REMOTE_HTTP,
        host="chroma.internal.example",
        port=8200,
        ssl=True,
        timeout_seconds=7.5,
    )


def ephemeral_config() -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.EPHEMERAL_TEST,
        host="127.0.0.1",
        port=18123,
        ssl=False,
        timeout_seconds=1.0,
    )


class FakeCollection:
    def __init__(
        self,
        name: str,
        *,
        semantic_collection_id: str | None = None,
        schema_version: str | None = None,
        count_value: int = 0,
        page_value=None,
        mutation_affected: int = 1,
    ):
        self.name = name
        self.semantic_collection_id = semantic_collection_id or name
        self.schema_version = schema_version or f"{name}.v1"
        self.count_value = count_value
        self.page_value = page_value
        self.mutation_affected = mutation_affected
        self.mutation_calls = []

    def count(self):
        return self.count_value

    def get(self, **_kwargs):
        return self.page_value

    def upsert(self, **kwargs):
        self.mutation_calls.append(("upsert", kwargs))
        return ChromaTransportMutationResult(
            self.semantic_collection_id,
            "upsert",
            len(kwargs["ids"]),
            len(kwargs["ids"]),
        )

    def delete(self, **kwargs):
        self.mutation_calls.append(("delete", kwargs))
        return ChromaTransportMutationResult(
            self.semantic_collection_id,
            "delete",
            len(kwargs["ids"]),
            min(self.mutation_affected, len(kwargs["ids"])),
        )


class FakeClient:
    def __init__(self, *, collections=None, lookup_error: BaseException | None = None):
        self.collections = collections or {
            "github_evidence": FakeCollection("github_evidence"),
            "profile_facts": FakeCollection("profile_facts"),
        }
        self.lookup_error = lookup_error
        self.lookup_names: list[str] = []
        self.heartbeat_calls = 0

    def get_collection(self, semantic_collection_id):
        self.lookup_names.append(semantic_collection_id)
        if self.lookup_error is not None:
            raise self.lookup_error
        if semantic_collection_id not in self.collections:
            raise ChromaCollectionMissing("chroma_collection_missing")
        return self.collections[semantic_collection_id]

    def heartbeat(self):
        self.heartbeat_calls += 1
        raise AssertionError("factory must not heartbeat automatically")


class RecordingConstructor:
    def __init__(self, client=None, error: BaseException | None = None):
        self.client = client or FakeClient()
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.client


def ready_factory(
    config: ChromaDeploymentConfig | None = None,
    *,
    client: FakeClient | None = None,
    test_context: bool = False,
):
    constructor = RecordingConstructor(client=client)
    factory = ChromaHttpClientFactory(
        config or local_config(), constructor, test_context=test_context
    )
    return factory, constructor


def test_factory_module_import_boundary_is_lazy_and_side_effect_free():
    source = FACTORY_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "chromadb" not in top_level_imports
    assert not top_level_imports & {"pathlib", "socket", "requests", "httpx", "dotenv"}
    assert "os." + "environ" not in source
    assert "load_" + "dotenv" not in source
    assert "information/" + "chroma" not in source


def test_constructing_factory_performs_no_client_network_or_filesystem_io(monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("factory construction must perform no I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    constructor = RecordingConstructor()
    factory = ChromaHttpClientFactory(local_config(), constructor)
    assert calls == []
    assert constructor.calls == []
    assert factory.state is ChromaFactoryState.UNINITIALIZED


def test_constructing_factory_does_not_observe_process_environment(monkeypatch):
    monkeypatch.setenv("CHROMA_DEPLOYMENT_MODE", "disabled")
    monkeypatch.setenv("CHROMA_HTTP_HOST", "attacker.example")
    factory, constructor = ready_factory(local_config())
    assert factory.get_factory_summary()["deployment_mode"] == "local_http"
    assert constructor.calls == []


def test_factory_summary_is_bounded_safe_and_deterministic():
    factory, _ = ready_factory(remote_config())
    assert factory.get_factory_summary() == {
        "factory_state": "uninitialized",
        "deployment_mode": "remote_http",
        "transport": "http",
        "registered_collection_count": 2,
        "client_cached": False,
        "timeout_enforced": True,
    }
    encoded = json.dumps(factory.get_factory_summary(), sort_keys=True)
    assert "chroma.internal.example" not in encoded
    assert "8200" not in encoded
    assert "password" not in encoded.casefold()
    assert repr(factory) == repr(factory)


def test_invalid_config_constructor_and_test_context_fail_closed():
    with pytest.raises(InvalidChromaFactoryConfiguration, match="invalid_chroma_factory_configuration"):
        ChromaHttpClientFactory(object())
    with pytest.raises(InvalidChromaFactoryConfiguration, match="invalid_http_transport_constructor"):
        ChromaHttpClientFactory(local_config(), object())
    with pytest.raises(InvalidChromaFactoryConfiguration, match="invalid_chroma_test_context"):
        ChromaHttpClientFactory(local_config(), test_context="yes")


def test_manually_inconsistent_http_config_fails_closed():
    inconsistent = ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.LOCAL_HTTP,
        host=None,
        port=None,
        ssl=False,
        timeout_seconds=5.0,
    )
    with pytest.raises(InvalidChromaFactoryConfiguration, match="invalid_http_deployment_configuration"):
        ChromaHttpClientFactory(inconsistent)


def test_disabled_factory_summary_and_state_are_explicit():
    constructor = RecordingConstructor()
    factory = ChromaHttpClientFactory(disabled_config(), constructor)
    assert factory.state is ChromaFactoryState.DISABLED
    assert factory.get_factory_summary() == {
        "factory_state": "disabled",
        "deployment_mode": "disabled",
        "transport": "none",
        "registered_collection_count": 2,
        "client_cached": False,
        "timeout_enforced": False,
    }
    assert constructor.calls == []


def test_disabled_factory_rejects_client_without_construction_or_fallback():
    constructor = RecordingConstructor()
    factory = ChromaHttpClientFactory(disabled_config(), constructor)
    with pytest.raises(ChromaFactoryDisabled, match="chroma_factory_disabled"):
        factory.get_client()
    assert constructor.calls == []
    assert factory.state is ChromaFactoryState.DISABLED


@pytest.mark.parametrize(
    "method",
    ("validate_collection_access", "get_collection_handle"),
)
def test_disabled_factory_rejects_all_collection_access_before_lookup(method):
    constructor = RecordingConstructor()
    factory = ChromaHttpClientFactory(disabled_config(), constructor)
    with pytest.raises(ChromaFactoryDisabled, match="chroma_factory_disabled"):
        getattr(factory, method)("github_evidence", "read", "github_evidence_metadata_reader")
    assert constructor.calls == []
    assert constructor.client.lookup_names == []


@pytest.mark.parametrize(
    "config",
    (local_config(), remote_config(), ephemeral_config()),
    ids=("local", "remote", "ephemeral"),
)
def test_http_modes_construct_injected_client_lazily(config):
    factory, constructor = ready_factory(config)
    assert constructor.calls == []
    assert factory.state is ChromaFactoryState.UNINITIALIZED
    assert factory.get_client() is constructor.client
    assert len(constructor.calls) == 1
    assert factory.state is ChromaFactoryState.READY


@pytest.mark.parametrize("config", (local_config(), remote_config(), ephemeral_config()))
def test_transport_constructor_receives_validated_config_as_single_authority(config):
    factory, constructor = ready_factory(config)
    factory.get_client()
    assert constructor.calls == [{"config": config}]


def test_http_client_is_cached_once_per_factory_instance():
    factory, constructor = ready_factory()
    first = factory.get_client()
    second = factory.get_client()
    assert first is second is constructor.client
    assert len(constructor.calls) == 1
    assert factory.get_factory_summary()["client_cached"] is True


def test_client_construction_does_not_heartbeat_or_open_collection():
    factory, constructor = ready_factory()
    factory.get_client()
    assert constructor.client.heartbeat_calls == 0
    assert constructor.client.lookup_names == []


def test_production_factory_constructs_only_workagent_bounded_transport():
    source = FACTORY_SOURCE.read_text(encoding="utf-8")
    assert "BoundedChromaHttpTransport(config)" in source
    assert "chromadb." + "HttpClient" not in source
    assert "Persistent" + "Client" not in source


def test_client_construction_unavailable_and_unexpected_errors_are_distinct_and_safe():
    unavailable = RecordingConstructor(error=ConnectionError("secret=C:/private"))
    factory = ChromaHttpClientFactory(local_config(), unavailable)
    with pytest.raises(ChromaTransportUnavailable, match="^chroma_transport_unavailable$"):
        factory.get_client()
    unexpected = RecordingConstructor(error=RuntimeError("secret=C:/private"))
    factory = ChromaHttpClientFactory(local_config(), unexpected)
    with pytest.raises(ChromaTransportError, match="^chroma_http_transport_construction_failed$"):
        factory.get_client()


def test_failed_client_construction_is_not_cached_and_can_be_retried():
    constructor = RecordingConstructor(error=TimeoutError("private"))
    factory = ChromaHttpClientFactory(local_config(), constructor)
    for _ in range(2):
        with pytest.raises(ChromaTransportUnavailable):
            factory.get_client()
    assert len(constructor.calls) == 2
    assert factory.state is ChromaFactoryState.UNINITIALIZED


def test_none_client_adapter_result_is_rejected_without_caching():
    constructor = RecordingConstructor()
    constructor.client = None
    factory = ChromaHttpClientFactory(local_config(), constructor)
    with pytest.raises(ChromaTransportError, match="invalid_chroma_http_transport"):
        factory.get_client()
    assert factory.state is ChromaFactoryState.UNINITIALIZED


@pytest.mark.parametrize(
    "semantic_id,consumer,lifecycle,physical,schema",
    (
        (
            "github_evidence",
            "github_evidence_metadata_reader",
            "read",
            "github_evidence",
            "github_evidence.v1",
        ),
        (
            "profile_facts",
            "profile_memory_reader",
            ChromaAccessLifecycle.READ,
            "profile_facts",
            "profile_facts.v1",
        ),
    ),
)
def test_known_semantic_collection_resolves_canonical_registry_authority(
    semantic_id, consumer, lifecycle, physical, schema
):
    factory, constructor = ready_factory()
    access = factory.validate_collection_access(semantic_id, lifecycle, consumer)
    assert access.semantic_collection_id == semantic_id
    assert access.collection_name == physical
    assert access.schema_version == schema
    assert access.creation_allowed is False
    assert access.access_permitted is True
    assert constructor.calls == []


@pytest.mark.parametrize("semantic_id", ("", "unknown", "github_evidence_local", None))
def test_unknown_semantic_collection_fails_closed_without_client(semantic_id):
    factory, constructor = ready_factory()
    with pytest.raises(UnknownChromaCollection, match="unknown_chroma_collection"):
        factory.validate_collection_access(
            semantic_id, "read", "github_evidence_metadata_reader"
        )
    assert constructor.calls == []


def test_validated_access_is_immutable_bounded_and_safely_serializable():
    factory, _ = ready_factory()
    access = factory.validate_collection_access(
        "github_evidence", "vector_query", "github_evidence_vector_reader"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        access.consumer_id = "changed"
    assert access.safe_summary() == {
        "semantic_collection_id": "github_evidence",
        "collection_name": "github_evidence",
        "schema_version": "github_evidence.v1",
        "requested_lifecycle": "vector_query",
        "consumer_id": "github_evidence_vector_reader",
        "consumer_authority": "approved",
        "creation_allowed": False,
        "access_mode": "read_only",
        "access_permitted": True,
        "migration_target": None,
    }
    encoded = json.dumps(access.safe_summary(), sort_keys=True)
    assert "document" not in encoded and "embedding" not in encoded
    assert "metadata" not in encoded and "path" not in encoded


def test_registry_lookup_order_and_lifecycle_enum_are_deterministic():
    assert tuple(item.value for item in ChromaAccessLifecycle) == (
        "read",
        "vector_query",
        "write",
        "index",
        "migration",
        "maintenance",
        "test_only",
    )
    assert tuple(item.value for item in ChromaFactoryState) == (
        "disabled",
        "uninitialized",
        "ready",
    )


@pytest.mark.parametrize(
    "semantic_id,lifecycle,consumer,mode",
    (
        ("github_evidence", "read", "github_evidence_metadata_reader", "read_only"),
        ("github_evidence", "vector_query", "github_evidence_vector_reader", "read_only"),
        ("github_evidence", "index", "github_evidence_materializer", "controlled_index"),
        ("github_evidence", "migration", "github_evidence_migration", "controlled_migration"),
        (
            "github_evidence",
            "maintenance",
            "github_evidence_maintenance",
            "controlled_maintenance",
        ),
        ("profile_facts", "read", "profile_memory_reader", "read_only"),
        ("profile_facts", "vector_query", "profile_memory_vector_reader", "read_only"),
        ("profile_facts", "write", "profile_memory_writer", "controlled_write"),
        ("profile_facts", "index", "profile_memory_indexer", "controlled_index"),
        ("profile_facts", "migration", "profile_memory_migration", "controlled_migration"),
        (
            "profile_facts",
            "maintenance",
            "profile_memory_maintenance",
            "controlled_maintenance",
        ),
    ),
)
def test_allowed_lifecycle_and_approved_consumer_validate_without_io(
    semantic_id, lifecycle, consumer, mode
):
    factory, constructor = ready_factory()
    access = factory.validate_collection_access(semantic_id, lifecycle, consumer)
    assert access.consumer_authority is ChromaConsumerAuthority.APPROVED
    assert access.access_mode == mode
    assert constructor.calls == []


@pytest.mark.parametrize("lifecycle", ("unknown", "status", "config", "create", "", None))
def test_unknown_lifecycle_fails_closed_without_client(lifecycle):
    factory, constructor = ready_factory()
    with pytest.raises(UnsupportedChromaLifecycle, match="unsupported_chroma_lifecycle"):
        factory.validate_collection_access(
            "github_evidence", lifecycle, "github_evidence_metadata_reader"
        )
    assert constructor.calls == []


def test_github_ordinary_future_write_remains_disallowed():
    factory, constructor = ready_factory()
    with pytest.raises(UnsupportedChromaLifecycle, match="collection_lifecycle_not_permitted"):
        factory.validate_collection_access(
            "github_evidence", "write", "github_evidence_vector_reader"
        )
    assert constructor.calls == []


def test_profile_write_is_allowed_only_for_profile_writer():
    factory, _ = ready_factory()
    access = factory.validate_collection_access(
        "profile_facts", "write", "profile_memory_writer"
    )
    assert access.access_mode == "controlled_write"
    with pytest.raises(ChromaConsumerNotApproved):
        factory.validate_collection_access(
            "profile_facts", "write", "profile_memory_reader"
        )


@pytest.mark.parametrize("creation_requested", (True, 1, "yes", None))
def test_any_creation_request_fails_before_client_or_collection_lookup(creation_requested):
    factory, constructor = ready_factory()
    with pytest.raises(ChromaCollectionCreationNotAllowed, match="collection_creation_not_allowed"):
        factory.get_collection_handle(
            "profile_facts",
            "read",
            "profile_memory_reader",
            creation_requested=creation_requested,
        )
    assert constructor.calls == []
    assert constructor.client.lookup_names == []


def test_migration_maintenance_reader_writer_and_index_authorities_are_distinct():
    factory, _ = ready_factory()
    cases = (
        ("read", "profile_memory_writer"),
        ("write", "profile_memory_reader"),
        ("index", "profile_memory_writer"),
        ("migration", "profile_memory_maintenance"),
        ("maintenance", "profile_memory_migration"),
    )
    for lifecycle, consumer in cases:
        with pytest.raises(ChromaConsumerNotApproved):
            factory.validate_collection_access("profile_facts", lifecycle, consumer)


@pytest.mark.parametrize("consumer", ("", "unknown_consumer", "backend.module.Reader", None))
def test_unknown_consumer_fails_closed_without_client(consumer):
    factory, constructor = ready_factory()
    with pytest.raises(UnknownChromaConsumer, match="unknown_chroma_consumer"):
        factory.validate_collection_access("profile_facts", "read", consumer)
    assert constructor.calls == []


def test_known_consumer_in_wrong_category_is_not_approved():
    factory, _ = ready_factory()
    with pytest.raises(ChromaConsumerNotApproved, match="chroma_consumer_not_approved"):
        factory.validate_collection_access(
            "profile_facts", "read", "profile_memory_writer"
        )


@pytest.mark.parametrize(
    "semantic_id,lifecycle,consumer",
    (
        (
            "profile_facts",
            "read",
            "legacy_embedded_reader",
        ),
        (
            "github_evidence",
            "index",
            "legacy_embedded_writer",
        ),
    ),
)
def test_removed_legacy_consumers_are_unknown_without_authority(
    semantic_id, lifecycle, consumer
):
    factory, constructor = ready_factory()
    with pytest.raises(UnknownChromaConsumer, match="unknown_chroma_consumer"):
        factory.validate_collection_access(semantic_id, lifecycle, consumer)
    assert constructor.calls == []


def test_legacy_consumer_cannot_open_collection_or_construct_client():
    factory, constructor = ready_factory()
    with pytest.raises(UnknownChromaConsumer, match="unknown_chroma_consumer"):
        factory.get_collection_handle(
            "github_evidence", "index", "legacy_embedded_writer"
        )
    assert constructor.calls == []
    assert constructor.client.lookup_names == []


def test_test_consumer_requires_ephemeral_mode_and_explicit_test_context():
    no_context, _ = ready_factory(ephemeral_config(), test_context=False)
    with pytest.raises(ChromaConsumerNotApproved, match="test_consumer_context_required"):
        no_context.validate_collection_access(
            "profile_facts", "test_only", "ephemeral_test_fixture"
        )
    wrong_mode, _ = ready_factory(local_config(), test_context=True)
    with pytest.raises(ChromaConsumerNotApproved, match="test_consumer_context_required"):
        wrong_mode.validate_collection_access(
            "profile_facts", "test_only", "ephemeral_test_fixture"
        )
    valid, _ = ready_factory(ephemeral_config(), test_context=True)
    access = valid.validate_collection_access(
        "profile_facts", "test_only", "ephemeral_test_fixture"
    )
    assert access.consumer_authority is ChromaConsumerAuthority.TEST_ONLY
    assert access.access_permitted is True


def test_ordinary_consumer_cannot_use_test_only_lifecycle():
    factory, _ = ready_factory(ephemeral_config(), test_context=True)
    with pytest.raises(ChromaConsumerNotApproved):
        factory.validate_collection_access(
            "profile_facts", "test_only", "profile_memory_reader"
        )


def test_collection_handle_uses_existing_only_lookup_after_validation():
    factory, constructor = ready_factory()
    handle = factory.get_collection_handle(
        "github_evidence", "read", "github_evidence_metadata_reader"
    )
    assert isinstance(handle, ChromaCollectionHandle)
    assert constructor.calls and constructor.client.lookup_names == ["github_evidence"]
    assert handle.semantic_collection_id == "github_evidence"
    assert handle.collection_name == "github_evidence"
    assert handle.schema_version == "github_evidence.v1"
    assert handle.creation_allowed is False
    assert handle.collection_bound is True


def test_collection_handle_safe_count_returns_only_validated_cardinality():
    collection = FakeCollection("github_evidence", count_value=7)
    client = FakeClient(collections={"github_evidence": collection})
    factory, _ = ready_factory(
        ephemeral_config(), client=client, test_context=True
    )
    handle = factory.get_collection_handle(
        "github_evidence", "test_only", "ephemeral_test_fixture"
    )
    assert handle.safe_count() == 7


def test_collection_handle_safe_count_rejects_invalid_cardinality():
    collection = FakeCollection("github_evidence", count_value=-1)
    client = FakeClient(collections={"github_evidence": collection})
    factory, _ = ready_factory(
        ephemeral_config(), client=client, test_context=True
    )
    handle = factory.get_collection_handle(
        "github_evidence", "test_only", "ephemeral_test_fixture"
    )
    with pytest.raises(
        ChromaCollectionAuthorityMismatch, match="invalid_chroma_collection_count"
    ):
        handle.safe_count()


def test_collection_handle_safe_get_page_accepts_only_metadata_transport_records():
    page = ChromaTransportRecords(
        semantic_collection_id="github_evidence",
        ids=("record",),
        metadatas=({"repository": "owner/repo"},),
        included=("metadatas",),
    )
    collection = FakeCollection("github_evidence", page_value=page)
    factory, _ = ready_factory(
        ephemeral_config(),
        client=FakeClient(collections={"github_evidence": collection}),
        test_context=True,
    )
    handle = factory.get_collection_handle(
        "github_evidence", "test_only", "ephemeral_test_fixture"
    )
    assert handle.safe_get_page(limit=1, offset=0) is page
    collection.page_value = object()
    with pytest.raises(ChromaCollectionAuthorityMismatch, match="invalid_chroma_collection_page"):
        handle.safe_get_page(limit=1, offset=0)


def test_handle_summary_exposes_only_bounded_semantic_fields():
    factory, _ = ready_factory()
    handle = factory.get_collection_handle(
        "profile_facts", "write", "profile_memory_writer"
    )
    assert handle.safe_summary() == {
        "semantic_collection_id": "profile_facts",
        "collection_name": "profile_facts",
        "schema_version": "profile_facts.v1",
        "requested_lifecycle": "write",
        "consumer_id": "profile_memory_writer",
        "consumer_authority": "approved",
        "creation_allowed": False,
        "access_mode": "controlled_write",
        "access_permitted": True,
        "migration_target": None,
        "collection_bound": True,
    }
    encoded = json.dumps(handle.safe_summary(), sort_keys=True)
    assert "document" not in encoded and "embedding" not in encoded
    assert "metadata" not in encoded and "path" not in encoded
    assert "127.0.0.1" not in encoded and "8100" not in encoded


def test_collection_handle_is_immutable_and_raw_accessor_is_not_serialized():
    factory, _ = ready_factory()
    handle = factory.get_collection_handle(
        "profile_facts", "read", "profile_memory_reader"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        handle.access = None
    assert "FakeClient" not in repr(handle)
    assert "FakeCollection" not in repr(handle)
    assert "_collection_accessor" not in json.dumps(handle.safe_summary())


def test_missing_collection_is_distinct_and_safe():
    client = FakeClient(collections={"other": FakeCollection("other")})
    factory, constructor = ready_factory(client=client)
    with pytest.raises(ChromaCollectionMissing, match="^chroma_collection_missing$"):
        factory.get_collection_handle(
            "profile_facts", "read", "profile_memory_reader"
        )
    assert len(constructor.calls) == 1
    assert client.lookup_names == ["profile_facts"]


@pytest.mark.parametrize(
    "error,error_type,code",
    (
        (ConnectionError("secret=C:/private"), ChromaTransportUnavailable, "chroma_transport_unavailable"),
        (TimeoutError("secret=C:/private"), ChromaTransportUnavailable, "chroma_transport_unavailable"),
        (RuntimeError("secret=C:/private"), ChromaTransportError, "chroma_collection_lookup_failed"),
    ),
)
def test_collection_lookup_transport_errors_are_mapped_without_raw_details(
    error, error_type, code
):
    client = FakeClient(lookup_error=error)
    factory, _ = ready_factory(client=client)
    with pytest.raises(error_type) as captured:
        factory.get_collection_handle(
            "profile_facts", "read", "profile_memory_reader"
        )
    assert str(captured.value) == code
    assert captured.value.__cause__ is None
    assert "private" not in str(captured.value)
    assert "secret" not in str(captured.value)


def test_collection_authority_mismatch_fails_closed():
    client = FakeClient(
        collections={
            "profile_facts": FakeCollection(
                "wrong_collection",
                semantic_collection_id="profile_facts",
                schema_version="profile_facts.v1",
            )
        }
    )
    factory, _ = ready_factory(client=client)
    with pytest.raises(
        ChromaCollectionAuthorityMismatch,
        match="chroma_collection_authority_mismatch",
    ):
        factory.get_collection_handle(
            "profile_facts", "read", "profile_memory_reader"
        )


def test_failed_collection_lookup_keeps_only_transport_cache_not_collection_fallback():
    client = FakeClient(
        lookup_error=ChromaCollectionMissing("chroma_collection_missing")
    )
    factory, constructor = ready_factory(client=client)
    for _ in range(2):
        with pytest.raises(ChromaCollectionMissing):
            factory.get_collection_handle(
                "profile_facts", "read", "profile_memory_reader"
            )
    assert len(constructor.calls) == 1
    assert client.lookup_names == ["profile_facts", "profile_facts"]


def test_factory_source_has_one_semantic_lookup_and_only_bounded_mutation_api():
    source = FACTORY_SOURCE.read_text(encoding="utf-8")
    assert source.count(".get_" + "collection(") == 1
    assert ".get_or_create_" + "collection(" not in source
    assert ".create_" + "collection(" not in source
    assert ".delete_" + "collection(" not in source
    assert ".query(" in source
    assert "safe_vector_query" in source
    assert "safe_upsert_records" in source
    assert "safe_delete_records" in source
    assert ".add(" not in source


def test_factory_handle_gates_upsert_to_index_and_delete_to_write_or_index():
    profile = FakeCollection("profile_facts")
    client = FakeClient(collections={"profile_facts": profile})
    factory, _ = ready_factory(client=client)
    index_handle = factory.get_collection_handle(
        "profile_facts", "index", "profile_memory_indexer"
    )
    result = index_handle.safe_upsert_records(
        ids=["profile-1"],
        embeddings=[[1.0, 0.0]],
        documents=["profile content"],
        metadatas=[{"section": "summary"}],
    )
    assert result.operation == "upsert"
    index_handle.safe_delete_records(ids=["profile-1"])

    write_handle = factory.get_collection_handle(
        "profile_facts", "write", "profile_memory_writer"
    )
    write_handle.safe_delete_records(ids=["profile-1"])
    with pytest.raises(ChromaCollectionAuthorityMismatch, match="index_lifecycle_required"):
        write_handle.safe_upsert_records(
            ids=["profile-1"],
            embeddings=[[1.0, 0.0]],
            documents=["profile content"],
            metadatas=[{"section": "summary"}],
        )


def test_factory_read_handle_cannot_mutate():
    profile = FakeCollection("profile_facts")
    factory, _ = ready_factory(client=FakeClient(collections={"profile_facts": profile}))
    handle = factory.get_collection_handle(
        "profile_facts", "read", "profile_memory_reader"
    )
    with pytest.raises(ChromaCollectionAuthorityMismatch, match="index_lifecycle_required"):
        handle.safe_upsert_records(
            ids=["one"],
            embeddings=[[1.0, 0.0]],
            documents=["private"],
            metadatas=[{}],
        )
    with pytest.raises(ChromaCollectionAuthorityMismatch, match="mutation_lifecycle_required"):
        handle.safe_delete_records(ids=["one"])
    assert profile.mutation_calls == []


def test_inventory_classifies_factory_authority_and_bounded_transport_paths():
    factory_records = [
        item
        for item in INVENTORY["records"]
        if item["module"] == "backend/chroma_http_client_factory.py"
    ]
    assert len(factory_records) == 1
    assert {
        (item["operation"], item["client_type"], item["may_create_collection"])
        for item in factory_records
    } == {("get collection", "http", False)}
    assert {item["current_owner"] for item in factory_records} == {
        "central_http_client_factory"
    }
    assert {item["current_state"] for item in factory_records} == {
        "accepted_central_http_factory"
    }
    transport_records = [
        item
        for item in INVENTORY["records"]
        if item["module"] == "backend/chroma_http_transport.py"
    ]
    assert {
        (item["operation"], item["client_type"], item["may_create_collection"])
        for item in transport_records
    } == {
        ("client construction", "http", False),
        ("http request", "http", False),
    }
    assert {item["current_owner"] for item in transport_records} == {
        "bounded_chroma_http_transport"
    }


def test_inventory_registry_sync_and_literal_guard_remain_exact():
    summary = validate_inventory_against_collection_registry(INVENTORY)
    assert summary["inventory_record_count"] == len(INVENTORY["records"])
    assert summary["resolved_collection_entry_count"] == 9
    assert summary["resolved_collection_binding_count"] == 18
    assert (
        summary["resolved_collection_entry_count"]
        + summary["non_collection_entry_count"]
        == summary["inventory_record_count"]
    )
    assert summary["approved_entry_count"] == 1
    assert summary["legacy_entry_count"] == 0
    assert summary["test_entry_count"] == 8
    assert summary["validation_state"] == "valid"
    literal_report = validate_collection_name_literals(ROOT)
    assert literal_report["violation_count"] == literal_report["unknown_count"] == 0


def test_factory_files_are_backend_only_semantically_named_and_contain_no_later_work():
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (FACTORY_SOURCE, Path(__file__))
    ).casefold()
    assert "frontend/" + "src" not in sources
    assert "phase" + "6_5" not in sources
    assert "phase_" + "6_5" not in sources
    assert "step" + "4" not in sources
    assert "start_" + "server" not in sources
    assert "stop_" + "server" not in sources
    assert "backup_" + "collection" not in sources
    assert "restore_" + "collection" not in sources


def test_error_messages_and_reprs_do_not_expose_config_or_transport_values():
    constructor = RecordingConstructor(error=RuntimeError("password=x C:/private"))
    factory = ChromaHttpClientFactory(remote_config(), constructor)
    with pytest.raises(ChromaTransportError) as captured:
        factory.get_client()
    combined = f"{captured.value!s}\n{factory!r}"
    assert "password" not in combined.casefold()
    assert "C:/" not in combined
    assert "chroma.internal.example" not in combined
    assert "8200" not in combined


def test_factory_does_not_duplicate_config_parsing_or_registry_names():
    source = FACTORY_SOURCE.read_text(encoding="utf-8")
    assert "load_chroma_deployment_config" not in source
    assert "CHROMA_DEPLOYMENT_MODE" not in source
    assert '"profile_facts"' not in source
    assert '"github_evidence"' not in source
    assert "get_collection_definition" in source
