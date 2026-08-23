from __future__ import annotations

import ast
import builtins
import inspect
import math
import socket
from pathlib import Path

import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_client_factory import (
    ChromaHttpClientFactory,
    UnknownChromaCollection,
    UnknownChromaConsumer,
)
from backend.chroma_http_transport import (
    ChromaCollectionMissing,
    ChromaTransportMutationResult,
    ChromaTransportTimeout,
)
from backend.chroma_write_client import (
    ChromaSemanticWriteError,
    ChromaWriteAuthorityViolation,
    ChromaWriteClient,
    ChromaWriteLimitExceeded,
)
from backend.chroma_write_models import (
    MAX_WRITE_DOCUMENT_CHARS,
    MAX_WRITE_EMBEDDING_DIMENSIONS,
    MAX_WRITE_RECORDS,
    ChromaWriteModelError,
    ChromaWriteRecord,
)
from backend.project_repository_identity import (
    build_project_repository_identity_authority,
)


ROOT = Path(__file__).resolve().parents[1]
WRITE_CLIENT_SOURCE = ROOT / "backend" / "chroma_write_client.py"


def local_config() -> ChromaDeploymentConfig:
    return ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        "127.0.0.1",
        8100,
        False,
        2.0,
    )


def authority(*pairs: tuple[str, str]):
    return build_project_repository_identity_authority(
        project_memory={
            "projects": [
                {"project_id": project_id, "repository": repository}
                for project_id, repository in pairs
            ]
        }
    )


class FakeCollection:
    def __init__(self, semantic_id: str, *, error: BaseException | None = None):
        self.semantic_collection_id = semantic_id
        self.name = semantic_id
        self.schema_version = f"{semantic_id}.v1"
        self.error = error
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))
        if self.error is not None:
            raise self.error
        count = len(kwargs["ids"])
        return ChromaTransportMutationResult(self.semantic_collection_id, "upsert", count, count)

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if self.error is not None:
            raise self.error
        count = len(kwargs["ids"])
        return ChromaTransportMutationResult(self.semantic_collection_id, "delete", count, count)


class FakeTransport:
    def __init__(self, collections=None):
        self.collections = collections or {
            "github_evidence": FakeCollection("github_evidence"),
            "profile_facts": FakeCollection("profile_facts"),
        }
        self.lookups = []
        self.closed = False

    def get_collection(self, semantic_id):
        self.lookups.append(semantic_id)
        if semantic_id not in self.collections:
            raise ChromaCollectionMissing("chroma_collection_missing")
        return self.collections[semantic_id]

    def close(self):
        self.closed = True


class FactoryBuilder:
    def __init__(self, transport: FakeTransport):
        self.transport = transport
        self.factory_calls = 0
        self.transport_calls = 0

    def __call__(self, config):
        self.factory_calls += 1

        def construct(**_kwargs):
            self.transport_calls += 1
            return self.transport

        return ChromaHttpClientFactory(config, construct)


def client_for(transport: FakeTransport):
    builder = FactoryBuilder(transport)
    return ChromaWriteClient(
        config_provider=local_config,
        factory_builder=builder,
    ), builder


def record(
    record_id="record-1",
    *,
    document="approved content",
    metadata=None,
    embedding=(1.0, 0.0),
):
    return ChromaWriteRecord(
        record_id=record_id,
        document=document,
        metadata={} if metadata is None else metadata,
        embedding=embedding,
    )


def test_write_client_import_and_construction_are_io_free(monkeypatch):
    source = WRITE_CLIENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "chromadb" not in imports
    assert "httpx" not in imports
    assert "Persistent" + "Client" not in source
    assert "create_" + "collection" not in source
    assert "get_or_create_" + "collection" not in source

    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("io")
        raise AssertionError("construction must be I/O free")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    ChromaWriteClient(config_provider=forbidden, factory_builder=forbidden)
    assert calls == []


def test_profile_upsert_and_delete_require_central_factory_authority():
    transport = FakeTransport()
    client, builder = client_for(transport)
    upserted = client.upsert_records(
        "profile_facts",
        consumer_id="profile_memory_indexer",
        records=[record(metadata={"section": "summary"})],
    )
    deleted = client.delete_records(
        "profile_facts",
        consumer_id="profile_memory_writer",
        ids=["record-1"],
        lifecycle="write",
    )
    assert upserted.safe_summary()["accepted_count"] == 1
    assert deleted.safe_summary()["accepted_count"] == 1
    assert builder.transport_calls == 2
    assert transport.lookups == ["profile_facts", "profile_facts"]
    assert transport.collections["profile_facts"].calls[0][1]["ids"] == ["record-1"]
    assert transport.closed is True


@pytest.mark.parametrize(
    "semantic_id,consumer,error_type",
    (
        ("unknown", "profile_memory_indexer", UnknownChromaCollection),
        ("profile_facts", "unknown_consumer", UnknownChromaConsumer),
    ),
)
def test_unknown_collection_or_consumer_is_rejected_before_network(
    semantic_id, consumer, error_type
):
    transport = FakeTransport()
    client, builder = client_for(transport)
    with pytest.raises(error_type):
        client.upsert_records(
            semantic_id,
            consumer_id=consumer,
            records=[record()],
        )
    assert builder.transport_calls == 0
    assert transport.lookups == []


def test_unauthorized_lifecycle_is_rejected_before_factory_or_network():
    transport = FakeTransport()
    client, builder = client_for(transport)
    with pytest.raises(ChromaSemanticWriteError, match="index_lifecycle_required"):
        client.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[record()],
            lifecycle="write",
        )
    with pytest.raises(ChromaSemanticWriteError, match="mutation_lifecycle_required"):
        client.delete_records(
            "profile_facts",
            consumer_id="profile_memory_writer",
            ids=["one"],
            lifecycle="read",
        )
    assert builder.factory_calls == 0
    assert builder.transport_calls == 0


def test_github_authority_is_validated_without_expanding_stored_metadata():
    transport = FakeTransport()
    client, _ = client_for(transport)
    verified = authority(("project-a", "owner/repo"))
    stored_metadata = {"repository": "owner/repo", "source": "github-fetch"}
    result = client.upsert_records(
        "github_evidence",
        consumer_id="github_evidence_materializer",
        records=[record(metadata=stored_metadata)],
        authority_metadata=[
            {
                "project_id": "project-a",
                "repository": "owner/repo",
                "repository_project_id": "project-a",
            }
        ],
        repository_authority=verified,
    )
    assert result.accepted_count == 1
    call = transport.collections["github_evidence"].calls[0]
    assert call[1]["metadatas"] == [stored_metadata]
    assert "project_id" not in call[1]["metadatas"][0]


@pytest.mark.parametrize(
    "authority_metadata,verified,code",
    (
        (None, None, "github_write_authority_required"),
        (
            [{"project_id": "project-a", "repository": "owner/repo", "repository_project_id": "project-a"}],
            None,
            "github_write_authority_unavailable",
        ),
        (
            [{"project_id": "project-b", "repository": "owner/repo", "repository_project_id": "project-b"}],
            authority(("project-a", "owner/repo")),
            "github_write_authority_violation",
        ),
        (
            [{"project_id": "project-a", "repository": "owner/unknown", "repository_project_id": "project-a"}],
            authority(("project-a", "owner/repo")),
            "github_write_authority_violation",
        ),
        (
            [{"project_id": "project-a", "repository": "owner/repo", "repository_project_id": "project-b"}],
            authority(("project-a", "owner/repo")),
            "github_write_authority_violation",
        ),
    ),
)
def test_github_missing_or_conflicting_authority_fails_before_network(
    authority_metadata, verified, code
):
    transport = FakeTransport()
    client, builder = client_for(transport)
    with pytest.raises(ChromaWriteAuthorityViolation, match=f"^{code}$"):
        client.upsert_records(
            "github_evidence",
            consumer_id="github_evidence_materializer",
            records=[record(metadata={"repository": "owner/repo"})],
            authority_metadata=authority_metadata,
            repository_authority=verified,
        )
    assert builder.factory_calls == 0
    assert transport.lookups == []


def test_cross_project_batch_is_rejected_before_network():
    transport = FakeTransport()
    client, builder = client_for(transport)
    verified = authority(
        ("project-a", "owner/repo-a"),
        ("project-b", "owner/repo-b"),
    )
    with pytest.raises(ChromaWriteAuthorityViolation, match="cross_project"):
        client.upsert_records(
            "github_evidence",
            consumer_id="github_evidence_materializer",
            records=[record("one"), record("two")],
            authority_metadata=[
                {"project_id": "project-a", "repository": "owner/repo-a", "repository_project_id": "project-a"},
                {"project_id": "project-b", "repository": "owner/repo-b", "repository_project_id": "project-b"},
            ],
            repository_authority=verified,
        )
    assert builder.factory_calls == 0


def test_missing_collection_fails_without_creation_or_embedded_fallback():
    transport = FakeTransport(collections={"other": FakeCollection("other")})
    client, builder = client_for(transport)
    with pytest.raises(ChromaCollectionMissing):
        client.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[record()],
        )
    assert builder.transport_calls == 1
    assert transport.lookups == ["profile_facts"]


def test_mutation_failure_has_no_retry_and_no_content_in_error():
    secret = "private document body"
    collection = FakeCollection(
        "profile_facts",
        error=ChromaTransportTimeout("chroma_transport_timeout"),
    )
    transport = FakeTransport(collections={"profile_facts": collection})
    client, _ = client_for(transport)
    with pytest.raises(ChromaTransportTimeout) as captured:
        client.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[record(document=secret, metadata={"secret": "value"})],
        )
    assert len(collection.calls) == 1
    assert secret not in str(captured.value)
    assert "value" not in str(captured.value)


def test_model_and_request_bounds_reject_unsafe_mutations_before_network():
    with pytest.raises(ChromaWriteModelError, match="record_id"):
        record("")
    with pytest.raises(ChromaWriteModelError, match="document"):
        record(document="x" * (MAX_WRITE_DOCUMENT_CHARS + 1))
    with pytest.raises(ChromaWriteModelError, match="embedding"):
        record(embedding=[0.0] * (MAX_WRITE_EMBEDDING_DIMENSIONS + 1))
    with pytest.raises(ChromaWriteModelError, match="embedding"):
        record(embedding=[math.inf])

    transport = FakeTransport()
    client, builder = client_for(transport)
    with pytest.raises(ChromaSemanticWriteError, match="upsert_request"):
        client.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[record(str(index)) for index in range(MAX_WRITE_RECORDS + 1)],
        )
    with pytest.raises(ChromaSemanticWriteError, match="delete_request"):
        client.delete_records(
            "profile_facts",
            consumer_id="profile_memory_writer",
            ids=["same", "same"],
            lifecycle="write",
        )
    with pytest.raises(ChromaWriteLimitExceeded, match="request_limit"):
        client.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[
                record(str(index), embedding=[0.0] * MAX_WRITE_EMBEDDING_DIMENSIONS)
                for index in range(MAX_WRITE_RECORDS)
            ],
        )
    assert builder.factory_calls == 0


def test_safe_models_and_source_never_render_documents_embeddings_or_raw_metadata():
    item = record(
        document="private document",
        metadata={"token": "private metadata"},
        embedding=(0.25, 0.75),
    )
    rendered = repr(item)
    assert "private document" not in rendered
    assert "private metadata" not in rendered
    assert "0.25" not in rendered
    assert inspect.signature(ChromaWriteClient.upsert_records).parameters["lifecycle"].default.value == "index"
