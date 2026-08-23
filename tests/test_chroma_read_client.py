from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_client_factory import ChromaAccessLifecycle
from backend.chroma_http_transport import (
    ChromaTransportQueryResult,
    ChromaTransportRecords,
)
from backend.chroma_read_client import (
    ChromaReadClient,
    ChromaReadLimitExceeded,
    ChromaReadSnapshotInconsistent,
)


CONFIG = ChromaDeploymentConfig(
    mode=ChromaDeploymentMode.LOCAL_HTTP,
    host="127.0.0.1",
    port=8100,
    ssl=False,
    timeout_seconds=1.0,
)


class FakeTransport:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeHandle:
    def __init__(self, semantic_id, records):
        self.semantic_id = semantic_id
        self.records = records
        self.get_calls = []
        self.query_calls = []
        self.count_calls = 0
        self.duplicate_page = False

    def safe_count(self):
        self.count_calls += 1
        return len(self.records)

    def safe_get_records(self, **kwargs):
        self.get_calls.append(kwargs)
        ids = kwargs.get("ids")
        where = kwargs.get("where")
        values = self.records
        if ids is not None:
            selected = [record for record in values if record[0] in ids]
        else:
            selected = values
            if where is not None:
                selected = [
                    record
                    for record in selected
                    if all(record[2].get(key) == value for key, value in where.items())
                ]
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit")
            selected = selected[offset : offset + limit]
            if self.duplicate_page and offset:
                selected = values[:1]
        include_documents = kwargs["include_documents"]
        include_metadata = kwargs["include_metadata"]
        included = tuple(
            name
            for name, enabled in (
                ("documents", include_documents),
                ("metadatas", include_metadata),
            )
            if enabled
        )
        return ChromaTransportRecords(
            semantic_collection_id=self.semantic_id,
            ids=tuple(record[0] for record in selected),
            documents=(tuple(record[1] for record in selected) if include_documents else None),
            metadatas=(tuple(record[2] for record in selected) if include_metadata else None),
            included=included,
        )

    def safe_vector_query(self, **kwargs):
        self.query_calls.append(kwargs)
        selected = self.records[: kwargs["n_results"]]
        include_documents = kwargs["include_documents"]
        include_metadata = kwargs["include_metadata"]
        include_distances = kwargs["include_distances"]
        included = tuple(
            name
            for name, enabled in (
                ("distances", include_distances),
                ("documents", include_documents),
                ("metadatas", include_metadata),
            )
            if enabled
        )
        return ChromaTransportQueryResult(
            semantic_collection_id=self.semantic_id,
            ids=(tuple(record[0] for record in selected),),
            distances=(tuple(float(index) / 10 for index, _ in enumerate(selected)),)
            if include_distances
            else None,
            documents=(tuple(record[1] for record in selected),) if include_documents else None,
            metadatas=(tuple(record[2] for record in selected),) if include_metadata else None,
            included=included,
        )


class FakeFactory:
    def __init__(self, handle, transport):
        self.handle = handle
        self.transport = transport
        self.calls = []

    def get_collection_handle(self, semantic_id, lifecycle, consumer_id, **kwargs):
        self.calls.append((semantic_id, lifecycle, consumer_id, kwargs))
        return self.handle

    def get_transport(self):
        return self.transport


def make_reader(records):
    handle = FakeHandle("github_evidence", records)
    transport = FakeTransport()
    factory = FakeFactory(handle, transport)
    config_calls = []
    factory_calls = []

    def config_provider():
        config_calls.append(True)
        return CONFIG

    def factory_builder(config):
        factory_calls.append(config)
        return factory

    return (
        ChromaReadClient(config_provider=config_provider, factory_builder=factory_builder),
        handle,
        transport,
        factory,
        config_calls,
        factory_calls,
    )


def test_import_and_construction_perform_no_network_or_factory_io():
    calls = []
    reader = ChromaReadClient(
        config_provider=lambda: calls.append("config"),
        factory_builder=lambda _config: calls.append("factory"),
    )
    assert calls == []
    assert "record_count" not in repr(reader)


def test_read_records_uses_central_factory_projects_metadata_and_closes_transport():
    reader, handle, transport, factory, config_calls, factory_calls = make_reader([
        ("one", "approved content", {"project_id": "ProjectA", "secret": "hidden"}),
    ])
    result = reader.read_records(
        "github_evidence",
        consumer_id="github_evidence_metadata_reader",
        ids=["one"],
        include_documents=True,
        metadata_fields=("project_id",),
        max_records=1,
    )

    assert config_calls == [True]
    assert factory_calls == [CONFIG]
    assert factory.calls == [(
        "github_evidence",
        ChromaAccessLifecycle.READ,
        "github_evidence_metadata_reader",
        {"creation_requested": False},
    )]
    assert handle.get_calls[0]["ids"] == ["one"]
    assert result.records[0].document == "approved content"
    assert dict(result.records[0].metadata) == {"project_id": "ProjectA"}
    assert "approved content" not in repr(result)
    assert "approved content" not in json.dumps(result.safe_summary())
    assert transport.closed


def test_paginated_read_is_ordered_bounded_and_snapshot_checked():
    records = [(f"id-{index}", None, {"index": index}) for index in range(5)]
    reader, handle, _transport, _factory, _config, _builders = make_reader(records)
    result = reader.read_records(
        "github_evidence",
        consumer_id="github_evidence_metadata_reader",
        metadata_fields=("index",),
        max_records=5,
        page_size=2,
    )
    assert [record.record_id for record in result.records] == [f"id-{index}" for index in range(5)]
    assert [call["offset"] for call in handle.get_calls] == [0, 2, 4]
    assert handle.count_calls == 2


def test_read_limit_and_duplicate_pagination_fail_safely():
    records = [(f"id-{index}", None, {}) for index in range(3)]
    reader, _handle, _transport, _factory, _config, _builders = make_reader(records)
    with pytest.raises(ChromaReadLimitExceeded, match="chroma_read_record_limit_exceeded"):
        reader.read_records(
            "github_evidence",
            consumer_id="github_evidence_metadata_reader",
            max_records=2,
        )

    reader, handle, _transport, _factory, _config, _builders = make_reader(records)
    handle.duplicate_page = True
    with pytest.raises(ChromaReadSnapshotInconsistent, match="duplicate_chroma_read_record"):
        reader.read_records(
            "github_evidence",
            consumer_id="github_evidence_metadata_reader",
            max_records=3,
            page_size=2,
        )


def test_vector_query_forwards_embedding_filter_limit_and_preserves_order_distance():
    records = [
        ("one", None, {"project_id": "ProjectA", "secret": "hidden"}),
        ("two", None, {"project_id": "ProjectA"}),
    ]
    reader, handle, transport, factory, _config, _builders = make_reader(records)
    result = reader.vector_query(
        "github_evidence",
        consumer_id="github_evidence_vector_reader",
        query_embedding=[0.0, 1.0],
        n_results=2,
        where={"project_id": "ProjectA"},
        include_documents=False,
        metadata_fields=("project_id",),
        include_distances=True,
    )
    assert factory.calls[0][1] is ChromaAccessLifecycle.VECTOR_QUERY
    assert handle.query_calls == [{
        "query_embeddings": [[0.0, 1.0]],
        "n_results": 2,
        "ids": None,
        "where": {"project_id": "ProjectA"},
        "include_documents": False,
        "include_metadata": True,
        "include_distances": True,
    }]
    assert [hit.record_id for hit in result.hits] == ["one", "two"]
    assert [hit.distance for hit in result.hits] == [0.0, 0.1]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert dict(result.hits[0].metadata) == {"project_id": "ProjectA"}
    assert transport.closed


def test_collection_lookup_failure_closes_existing_transport_without_fallback():
    transport = FakeTransport()

    class FailingFactory:
        def get_collection_handle(self, *_args, **_kwargs):
            raise RuntimeError("collection unavailable")

        def close(self):
            transport.close()

    reader = ChromaReadClient(
        config_provider=lambda: CONFIG,
        factory_builder=lambda _config: FailingFactory(),
    )
    with pytest.raises(RuntimeError, match="collection unavailable"):
        reader.read_records(
            "github_evidence",
            consumer_id="github_evidence_metadata_reader",
            ids=["one"],
        )
    assert transport.closed


def test_semantic_reader_has_no_direct_chroma_httpx_or_write_api():
    source = Path(ChromaReadClient.__module__.replace(".", "/") + ".py")
    source_text = (Path(__file__).parents[1] / source).read_text(encoding="utf-8")
    assert "import chromadb" not in source_text
    assert "import httpx" not in source_text
    assert "Persistent" + "Client" not in source_text
    assert "Http" + "Client(" not in source_text
    for forbidden in ("def add(", ".upsert(", ".update(", ".delete(", "create_collection"):
        assert forbidden not in source_text
