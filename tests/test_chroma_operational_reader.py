from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from backend import chroma_operational_reader as operational
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_client_factory import ChromaCollectionAuthorityMismatch
from backend.chroma_http_transport import (
    ChromaCollectionMissing,
    ChromaTransportRecords,
    ChromaTransportTimeout,
    ChromaTransportUnavailable,
)
from backend.chroma_operational_models import (
    CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA,
    ChromaOperationalCollectionStatus,
    ChromaOperationalRepositorySummary,
    ChromaOperationalModelError,
)
from backend.memory_store import MemoryVectorStore


def config(mode=ChromaDeploymentMode.LOCAL_HTTP):
    return ChromaDeploymentConfig(
        mode=mode,
        host=None if mode is ChromaDeploymentMode.DISABLED else "127.0.0.1",
        port=None if mode is ChromaDeploymentMode.DISABLED else 18180,
        ssl=False,
        timeout_seconds=1.0,
    )


class FakeTransport:
    def __init__(self, *, heartbeat_error=None):
        self.heartbeat_error = heartbeat_error
        self.heartbeat_calls = 0
        self.closed = False

    def heartbeat(self):
        self.heartbeat_calls += 1
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        return 1

    def close(self):
        self.closed = True


class FakeHandle:
    def __init__(self, *, counts=(0,), pages=None, count_error=None):
        self.counts = list(counts)
        self.pages = pages or {}
        self.count_error = count_error
        self.page_calls = []

    def safe_count(self):
        if self.count_error is not None:
            raise self.count_error
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]

    def safe_get_page(self, *, limit, offset):
        self.page_calls.append((limit, offset))
        ids, metadatas = self.pages[offset]
        return ChromaTransportRecords(
            semantic_collection_id="github_evidence",
            ids=tuple(ids),
            metadatas=tuple(metadatas),
            included=("metadatas",),
        )


class FakeFactory:
    def __init__(self, *, transport=None, handle=None, handle_error=None):
        self.transport = transport or FakeTransport()
        self.handle = handle or FakeHandle()
        self.handle_error = handle_error
        self.handle_calls = []

    def get_transport(self):
        return self.transport

    def get_collection_handle(self, *args, **kwargs):
        self.handle_calls.append((args, kwargs))
        if self.handle_error is not None:
            raise self.handle_error
        return self.handle


def reader(factory, *, deployment=None):
    return operational.ChromaOperationalReader(
        config_provider=lambda: deployment or config(),
        factory_builder=lambda _config: factory,
        clock=iter((10.0, 10.012)).__next__,
    )


def valid_status(
    semantic_id: str,
    count: int,
    repositories=(),
):
    return ChromaOperationalCollectionStatus(
        schema=CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA,
        server_state="available",
        collection_semantic_id=semantic_id,
        collection_name=semantic_id,
        collection_available=True,
        safe_record_count=count,
        latency_ms=1,
        integrity_state="valid",
        detail="ready",
        repositories=tuple(repositories),
    )


def test_models_are_strict_immutable_and_safe():
    repository = ChromaOperationalRepositorySummary(
        repository="owner/repo",
        project_id="project-a",
        source_type="file",
        updated_at="2026-08-10T01:02:03Z",
    )
    status = valid_status("github_evidence", 3, (repository,))
    assert status.available is True
    assert status.schema == CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA
    with pytest.raises(FrozenInstanceError):
        status.safe_record_count = 4
    serialized = json.dumps(status.safe_summary()).casefold()
    for forbidden in (
        "document",
        "embedding",
        "raw_metadata",
        "record_id",
        "filesystem",
        "absolute_path",
    ):
        assert forbidden not in serialized
    with pytest.raises(ChromaOperationalModelError):
        ChromaOperationalRepositorySummary(repository="https://github.com/owner/repo")


def test_success_uses_heartbeat_existing_only_handle_and_safe_count():
    factory = FakeFactory(handle=FakeHandle(counts=(7,)))
    status = reader(factory).read_collection_status("profile_facts")
    assert status.safe_summary() == {
        "schema": CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA,
        "server_state": "available",
        "collection_semantic_id": "profile_facts",
        "collection_name": "profile_facts",
        "collection_available": True,
        "available": True,
        "safe_record_count": 7,
        "latency_ms": 12,
        "integrity_state": "valid",
        "detail": "ready",
        "repositories": [],
    }
    assert factory.transport.heartbeat_calls == 1
    assert factory.transport.closed is True
    args, kwargs = factory.handle_calls[0]
    assert args[0] == "profile_facts"
    assert args[2] == operational.OPERATIONAL_READER_CONSUMER_ID
    assert kwargs == {"creation_requested": False}


def test_disabled_server_and_missing_collection_are_safe_states():
    disabled_factory = FakeFactory()
    disabled = reader(
        disabled_factory, deployment=config(ChromaDeploymentMode.DISABLED)
    ).read_collection_status("profile_facts")
    assert disabled.integrity_state == "unavailable"
    assert disabled.detail == "deployment_disabled"
    assert disabled_factory.transport.heartbeat_calls == 0

    missing_factory = FakeFactory(
        handle_error=ChromaCollectionMissing("chroma_collection_missing")
    )
    missing = reader(missing_factory).read_collection_status("github_evidence")
    assert missing.server_state == "degraded"
    assert missing.integrity_state == "collection_missing"
    assert missing.safe_record_count is None


@pytest.mark.parametrize(
    "error,detail",
    [
        (ChromaTransportTimeout("chroma_transport_timeout"), "transport_timeout"),
        (ChromaTransportUnavailable("chroma_transport_unavailable"), "server_unavailable"),
    ],
)
def test_timeout_and_unavailable_server_do_not_leak_or_fallback(error, detail):
    factory = FakeFactory(transport=FakeTransport(heartbeat_error=error))
    status = reader(factory).read_collection_status("profile_facts")
    assert status.server_state == "unavailable"
    assert status.integrity_state == "unavailable"
    assert status.detail == detail
    assert status.safe_record_count is None
    assert factory.handle_calls == []


def test_integrity_failure_is_bounded_and_does_not_return_count():
    factory = FakeFactory(
        handle=FakeHandle(
            count_error=ChromaCollectionAuthorityMismatch("invalid_count_private")
        )
    )
    status = reader(factory).read_collection_status("profile_facts")
    assert status.server_state == "degraded"
    assert status.integrity_state == "integrity_failure"
    assert status.detail == "integrity_failure"
    assert "private" not in json.dumps(status.safe_summary())


def test_repository_inventory_is_bounded_deduplicated_and_contains_no_ids():
    pages = {
        0: (
            ("secret-record-a", "secret-record-b"),
            (
                {
                    "repository": "Owner/Repo",
                    "project_id": "ProjectA",
                    "source_type": "file",
                    "updated_at": "2026-08-10T01:00:00Z",
                    "raw_metadata": "must-not-propagate",
                },
                {
                    "repository": "owner/repo",
                    "repository_project_id": "ProjectA",
                    "source_type": "commit",
                    "updated_at": "2026-08-10T02:00:00Z",
                    "document": "must-not-propagate",
                },
            ),
        )
    }
    handle = FakeHandle(counts=(2, 2), pages=pages)
    status = reader(FakeFactory(handle=handle)).read_collection_status(
        "github_evidence", include_repository_inventory=True
    )
    assert status.integrity_state == "valid"
    assert len(status.repositories) == 1
    assert status.repositories[0].repository == "owner/repo"
    assert status.repositories[0].updated_at == "2026-08-10T02:00:00Z"
    safe = json.dumps(status.safe_summary())
    assert "secret-record" not in safe
    assert "must-not-propagate" not in safe
    assert handle.page_calls == [(2, 0)]


def test_repository_inventory_conflict_duplicate_and_limit_fail_integrity(monkeypatch):
    conflict = FakeHandle(
        counts=(2, 2),
        pages={
            0: (
                ("a", "b"),
                (
                    {"repository": "owner/repo", "project_id": "project-a"},
                    {"repository": "owner/repo", "project_id": "project-b"},
                ),
            )
        },
    )
    status = reader(FakeFactory(handle=conflict)).read_collection_status(
        "github_evidence", include_repository_inventory=True
    )
    assert status.integrity_state == "integrity_failure"

    duplicate = FakeHandle(
        counts=(2, 2),
        pages={0: (("a", "a"), ({"repository": "owner/repo"},) * 2)},
    )
    status = reader(FakeFactory(handle=duplicate)).read_collection_status(
        "github_evidence", include_repository_inventory=True
    )
    assert status.integrity_state == "integrity_failure"

    monkeypatch.setattr(operational, "MAX_OPERATIONAL_INVENTORY_RECORDS", 1)
    limited = reader(FakeFactory(handle=FakeHandle(counts=(2,)))).read_collection_status(
        "github_evidence", include_repository_inventory=True
    )
    assert limited.integrity_state == "integrity_failure"


def test_safe_count_returns_zero_for_every_nonvalid_state():
    factory = FakeFactory(transport=FakeTransport(heartbeat_error=RuntimeError("private")))
    assert reader(factory).safe_count("profile_facts") == 0


def test_unknown_collection_and_wrong_inventory_request_fail_before_io():
    factory = FakeFactory()
    instance = reader(factory)
    with pytest.raises(ValueError, match="unknown_chroma_operational_collection"):
        instance.read_collection_status("unknown")
    with pytest.raises(ValueError, match="unsupported_chroma_operational_inventory"):
        instance.read_collection_status(
            "profile_facts", include_repository_inventory=True
        )
    assert factory.transport.heartbeat_calls == 0


def test_memory_store_operational_counts_and_status_never_touch_embedded_collections(tmp_path):
    class FailingCollection:
        def count(self):
            raise AssertionError("operational count must not use embedded collection")

        def get(self, **_kwargs):
            raise AssertionError("operational status must not use embedded collection")

    repository = ChromaOperationalRepositorySummary(
        repository="owner/repo", updated_at="2026-08-10T02:00:00Z"
    )

    class FakeOperationalReader:
        def safe_count(self, semantic_id):
            return {"profile_facts": 4, "github_evidence": 6}[semantic_id]

        def read_collection_status(self, semantic_id, **_kwargs):
            return valid_status(semantic_id, 6, (repository,))

    store = MemoryVectorStore(
        tmp_path / "must-not-be-read",
        tmp_path / "memory",
        tmp_path / "github",
        operational_reader=FakeOperationalReader(),
    )
    store._profile = FailingCollection()
    store._github = FailingCollection()
    assert store.profile_count() == 4
    assert store.github_count() == 6
    assert store.github_metadata_status() == {
        "available": True,
        "count": 6,
        "repositories": [
            {"repository": "owner/repo", "updated_at": "2026-08-10T02:00:00Z"}
        ],
    }
    assert not store.persist_directory.exists()


def test_reader_source_has_no_embedded_client_persistence_or_raw_content_surface():
    source = inspect.getsource(operational)
    assert "import chroma" + "db" not in source
    assert "Persistent" + "Client" not in source
    assert "Http" + "Client(" not in source
    assert "create_" + "collection" not in source
    assert "get_or_create_" + "collection" not in source
    assert "information/chroma" not in source
    assert "sqlite" not in source.casefold()
    assert "document" not in source.casefold()
    assert "embedding" not in source.casefold()
    assert "openai" not in source.casefold()
    assert "restart" not in source.casefold()
    assert "Path(" not in source
