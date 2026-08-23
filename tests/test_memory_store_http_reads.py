from __future__ import annotations

import json

import pytest

from backend.chroma_http_transport import ChromaCollectionMissing
from backend.chroma_read_models import (
    ChromaReadRecord,
    ChromaReadResult,
    ChromaVectorHit,
    ChromaVectorResult,
)
from backend.memory_store import MemoryVectorStore


class MemoryReadClient:
    def __init__(self, profile_records=(), github_records=(), *, error=None):
        self.profile_records = tuple(profile_records)
        self.github_records = tuple(github_records)
        self.error = error
        self.read_calls = []
        self.vector_calls = []

    def _records(self, semantic_id):
        return self.profile_records if semantic_id == "profile_facts" else self.github_records

    def read_records(self, semantic_id, **kwargs):
        self.read_calls.append({"semantic_id": semantic_id, **kwargs})
        if self.error is not None:
            raise self.error
        records = self._records(semantic_id)
        ids = kwargs.get("ids")
        where = kwargs.get("where")
        if ids is not None:
            records = tuple(record for record in records if record.record_id in ids)
        if where is not None:
            records = tuple(
                record
                for record in records
                if all(record.metadata.get(key) == value for key, value in where.items())
            )
        fields = kwargs["metadata_fields"]
        return ChromaReadResult(
            semantic_id,
            tuple(
                ChromaReadRecord(
                    record.record_id,
                    record.document if kwargs["include_documents"] else None,
                    {key: record.metadata[key] for key in fields if key in record.metadata},
                )
                for record in records[: kwargs["max_records"]]
            ),
        )

    def vector_query(self, semantic_id, **kwargs):
        self.vector_calls.append({"semantic_id": semantic_id, **kwargs})
        if self.error is not None:
            raise self.error
        fields = kwargs["metadata_fields"]
        hits = tuple(
            ChromaVectorHit(
                record.record_id,
                float(index) / 10,
                record.document if kwargs["include_documents"] else None,
                {key: record.metadata[key] for key in fields if key in record.metadata},
                index + 1,
            )
            for index, record in enumerate(self._records(semantic_id)[: kwargs["n_results"]])
        )
        return ChromaVectorResult(semantic_id, hits)


def profile_record(record_id, section, value, *, index=None):
    metadata = {"section": section, "is_list": index is not None}
    if index is not None:
        metadata["index"] = index
    return ChromaReadRecord(
        record_id,
        f"Profile memory section: {section}\n{json.dumps(value)}",
        metadata,
    )


def github_record(record_id, value, **metadata):
    return ChromaReadRecord(
        record_id,
        f"Approved GitHub evidence\n{json.dumps(value)}",
        metadata,
    )


def make_store(tmp_path, reader):
    store = MemoryVectorStore(
        tmp_path / "chroma",
        tmp_path / "memory.json",
        tmp_path / "github",
        read_client=reader,
    )
    store._ensure_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("migrated reads must not initialize embedded Chroma")
    )
    return store


def test_profile_read_preserves_membership_list_order_and_scalar_shape(tmp_path):
    scalar_id = MemoryVectorStore._record_id("profile", "summary")
    reader = MemoryReadClient(profile_records=(
        profile_record("project-1", "projects", {"name": "First"}, index=1),
        profile_record("project-0", "projects", {"name": "Zero"}, index=0),
        profile_record(scalar_id, "summary", {"title": "Engineer"}),
    ))
    store = make_store(tmp_path, reader)

    assert store.read_profile() == {
        "projects": [{"name": "Zero"}, {"name": "First"}],
        "summary": {"title": "Engineer"},
    }
    assert all(call["consumer_id"] == "profile_memory_reader" for call in reader.read_calls)
    assert not (tmp_path / "chroma").exists()


def test_profile_query_preserves_selected_section_semantics(tmp_path):
    reader = MemoryReadClient(profile_records=(
        profile_record("project-0", "projects", {"name": "Zero"}, index=0),
        profile_record("project-1", "projects", {"name": "First"}, index=1),
    ))
    store = make_store(tmp_path, reader)
    result = store.read_profile(query="retrieval", limit=1)
    assert result == {"projects": [{"name": "Zero"}, {"name": "First"}]}
    assert reader.vector_calls[0]["consumer_id"] == "profile_memory_vector_reader"
    assert reader.vector_calls[0]["n_results"] == 1
    assert reader.vector_calls[0]["include_documents"] is False


def test_github_context_reads_preserve_document_order_and_query_limit(tmp_path):
    reader = MemoryReadClient(github_records=(
        github_record("one", {"repository": "owner/one"}),
        github_record("two", {"repository": "owner/two"}),
    ))
    store = make_store(tmp_path, reader)
    assert store.read_github_contexts() == [
        {"repository": "owner/one"},
        {"repository": "owner/two"},
    ]
    assert store.read_github_contexts(query="evidence", limit=1) == [
        {"repository": "owner/one"},
    ]
    assert reader.vector_calls[-1]["consumer_id"] == "github_evidence_vector_reader"
    assert reader.vector_calls[-1]["include_documents"] is True


def test_github_record_lookup_projects_metadata_and_does_not_require_local_path(tmp_path):
    reader = MemoryReadClient(github_records=(
        github_record("one", {"repository": "owner/one"}, repository="owner/one", secret="hidden"),
    ))
    store = make_store(tmp_path, reader)
    assert store.read_github_document("one") == {
        "id": "one",
        "document": 'Approved GitHub evidence\n{"repository": "owner/one"}',
        "metadata": {"repository": "owner/one"},
    }
    assert not (tmp_path / "chroma").exists()


def test_missing_collection_is_optional_empty_but_unavailable_is_not_fabricated(tmp_path):
    missing = make_store(tmp_path, MemoryReadClient(error=ChromaCollectionMissing("missing")))
    assert missing.read_profile() == {}
    assert missing.read_github_contexts() == []
    assert missing.read_github_document("one") is None

    unavailable = make_store(tmp_path, MemoryReadClient(error=RuntimeError("server unavailable")))
    with pytest.raises(RuntimeError, match="server unavailable"):
        unavailable.read_profile()
    with pytest.raises(RuntimeError, match="server unavailable"):
        unavailable.read_github_contexts()
    with pytest.raises(RuntimeError, match="server unavailable"):
        unavailable.read_github_document("one")
