from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from backend.chroma_read_models import (
    ChromaReadRecord,
    ChromaReadResult,
    ChromaVectorHit,
    ChromaVectorResult,
)
from backend.chroma_write_client import ChromaWriteAuthorityViolation
from backend.chroma_write_models import ChromaWriteResult
from backend.memory_store import MemoryVectorStore, normalized_json
from backend.project_repository_identity import build_project_repository_identity_authority


ROOT = Path(__file__).resolve().parents[1]
MEMORY_STORE_SOURCE = ROOT / "backend" / "memory_store.py"


def authority(*pairs: tuple[str, str]):
    return build_project_repository_identity_authority(
        project_memory={
            "projects": [
                {"project_id": project_id, "repository": repository}
                for project_id, repository in pairs
            ]
        }
    )


class StateReadClient:
    def __init__(self, state=None, *, distances=None):
        self.state = state or {"profile_facts": {}, "github_evidence": {}}
        self.distances = distances or {}
        self.read_calls = []
        self.vector_calls = []

    def read_records(self, semantic_id, **kwargs):
        self.read_calls.append({"semantic_id": semantic_id, **kwargs})
        records = list(self.state[semantic_id].values())
        ids = kwargs.get("ids")
        where = kwargs.get("where")
        if ids is not None:
            records = [record for record in records if record.record_id in ids]
        if where is not None:
            records = [
                record
                for record in records
                if all(record.metadata.get(key) == value for key, value in where.items())
            ]
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
        fields = kwargs["metadata_fields"]
        records = list(self.state[semantic_id].values())[: kwargs["n_results"]]
        return ChromaVectorResult(
            semantic_id,
            tuple(
                ChromaVectorHit(
                    record.record_id,
                    self.distances.get(record.record_id, 1.0),
                    record.document if kwargs["include_documents"] else None,
                    {key: record.metadata[key] for key in fields if key in record.metadata},
                    rank,
                )
                for rank, record in enumerate(records, start=1)
            ),
        )


class StateWriteClient:
    def __init__(self, state, *, error=None):
        self.state = state
        self.error = error
        self.calls = []

    def upsert_records(self, semantic_id, **kwargs):
        self.calls.append({"operation": "upsert", "semantic_id": semantic_id, **kwargs})
        if self.error is not None:
            raise self.error
        for record in kwargs["records"]:
            self.state[semantic_id][record.record_id] = ChromaReadRecord(
                record.record_id,
                record.document,
                dict(record.metadata),
            )
        count = len(kwargs["records"])
        return ChromaWriteResult(semantic_id, "upsert", count, count, "applied")

    def delete_records(self, semantic_id, **kwargs):
        self.calls.append({"operation": "delete", "semantic_id": semantic_id, **kwargs})
        if self.error is not None:
            raise self.error
        affected = 0
        for record_id in kwargs["ids"]:
            affected += int(self.state[semantic_id].pop(record_id, None) is not None)
        return ChromaWriteResult(
            semantic_id,
            "delete",
            len(kwargs["ids"]),
            affected,
            "applied",
        )


def profile_record(record_id, section, value, *, index=0, is_list=False, source="old"):
    return ChromaReadRecord(
        record_id,
        f"Profile memory section: {section}\n{normalized_json(value)}",
        {
            "section": section,
            "index": index,
            "is_list": int(is_list),
            "source": source,
            "updated_at": "old",
        },
    )


def github_record(record_id, repository, value, *, updated_at="old"):
    return ChromaReadRecord(
        record_id,
        f"Approved GitHub evidence for {repository}\n{normalized_json(value)}",
        {
            "repository": repository,
            "run_id": updated_at,
            "source": "github-fetch",
            "updated_at": updated_at,
        },
    )


def make_store(tmp_path, *, state=None, distances=None, repository_authority=None, error=None):
    state = state or {"profile_facts": {}, "github_evidence": {}}
    reader = StateReadClient(state, distances=distances)
    writer = StateWriteClient(state, error=error)
    store = MemoryVectorStore(
        tmp_path / "chroma",
        tmp_path / "memory.json",
        tmp_path / "github",
        read_client=reader,
        write_client=writer,
        repository_authority_provider=lambda: repository_authority,
    )
    store._ensure_client = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("runtime writes must not initialize embedded Chroma")
    )
    return store, reader, writer, state


def test_profile_replace_preserves_ids_documents_metadata_and_outcome_counts(tmp_path):
    summary_id = MemoryVectorStore._record_id("profile", "summary")
    stale_id = MemoryVectorStore._record_id("profile", "stale")
    state = {
        "profile_facts": {
            summary_id: profile_record(summary_id, "summary", {"title": "Old"}),
            stale_id: profile_record(stale_id, "stale", "remove"),
        },
        "github_evidence": {},
    }
    store, reader, writer, state = make_store(tmp_path, state=state)
    result = store.replace_profile(
        {"summary": {"title": "New"}, "projects": [{"name": "One"}]},
        source="profile-update",
    )
    project_id = MemoryVectorStore._record_id("profile", "projects:One")
    assert result == {
        "inserted": 1,
        "updated": 1,
        "unchanged": 0,
        "deduplicated": 0,
        "deleted": 1,
    }
    assert set(state["profile_facts"]) == {summary_id, project_id}
    assert state["profile_facts"][summary_id].document == (
        "Profile memory section: summary\n" + normalized_json({"title": "New"})
    )
    assert state["profile_facts"][project_id].metadata["section"] == "projects"
    assert writer.calls[0]["operation"] == "delete"
    assert writer.calls[0]["consumer_id"] == "profile_memory_writer"
    assert all(
        call["consumer_id"] == "profile_memory_indexer"
        for call in writer.calls
        if call["operation"] == "upsert"
    )
    assert reader.vector_calls and not (tmp_path / "chroma").exists()


def test_profile_unchanged_and_empty_replacement_behavior_are_preserved(tmp_path):
    summary_id = MemoryVectorStore._record_id("profile", "summary")
    state = {
        "profile_facts": {
            summary_id: profile_record(summary_id, "summary", {"title": "Same"})
        },
        "github_evidence": {},
    }
    store, _reader, writer, state = make_store(tmp_path, state=state)
    unchanged = store.replace_profile({"summary": {"title": "Same"}})
    assert unchanged == {
        "inserted": 0,
        "updated": 0,
        "unchanged": 1,
        "deduplicated": 0,
        "deleted": 0,
    }
    assert writer.calls == []
    emptied = store.replace_profile({})
    assert emptied["deleted"] == 1
    assert state["profile_facts"] == {}


def test_profile_delete_preserves_selected_item_and_section_semantics(tmp_path):
    first = profile_record("first", "projects", {"name": "Zero"}, index=0, is_list=True)
    second = profile_record("second", "projects", {"name": "One"}, index=1, is_list=True)
    state = {
        "profile_facts": {"first": first, "second": second},
        "github_evidence": {},
    }
    store, _reader, writer, state = make_store(tmp_path, state=state)
    result = store.delete_profile("projects", item_index=1)
    assert result == {
        "deleted": 1,
        "section": "projects",
        "item_index": 1,
        "deleted_values": [{"name": "One"}],
    }
    assert set(state["profile_facts"]) == {"first"}
    assert writer.calls[-1]["ids"] == ["second"]


def test_similarity_threshold_and_dedup_decision_remain_point_twelve(tmp_path):
    similar = profile_record("similar", "skills", {"name": "Python"})
    state = {
        "profile_facts": {"similar": similar},
        "github_evidence": {},
    }
    store, _reader, writer, state = make_store(
        tmp_path,
        state=state,
        distances={"similar": 0.12},
    )
    result = store._upsert_with_similarity(
        "profile_facts",
        [
            {
                "id": "replacement",
                "document": "Profile memory section: skills\n{}",
                "metadata": {"section": "skills", "index": 0, "is_list": 0},
            }
        ],
        read_consumer_id="profile_memory_reader",
        vector_consumer_id="profile_memory_vector_reader",
        index_consumer_id="profile_memory_indexer",
    )
    assert result["deduplicated"] == 1
    assert set(state["profile_facts"]) == {"replacement"}
    assert [call["operation"] for call in writer.calls] == ["delete", "upsert"]


def test_github_store_uses_materializer_authority_without_changing_stored_schema(tmp_path):
    verified = authority(("project-a", "owner/repo"))
    store, _reader, writer, state = make_store(
        tmp_path,
        repository_authority=verified,
    )
    context = {"repository": "owner/repo", "project_id": "project-a", "files": 2}
    result = store.store_github_contexts([context])
    record_id = MemoryVectorStore._record_id("github", "owner/repo")
    assert result["inserted"] == 1
    stored = state["github_evidence"][record_id]
    assert stored.document == (
        "Approved GitHub evidence for owner/repo\n" + normalized_json(context)
    )
    assert set(stored.metadata) == {"repository", "run_id", "source", "updated_at"}
    upsert = next(call for call in writer.calls if call["operation"] == "upsert")
    assert upsert["consumer_id"] == "github_evidence_materializer"
    assert upsert["authority_metadata"] == [
        {
            "project_id": "project-a",
            "repository": "owner/repo",
            "repository_project_id": "project-a",
        }
    ]


def test_github_cross_project_and_unknown_repository_reject_before_read_or_write(tmp_path):
    verified = authority(
        ("project-a", "owner/a"),
        ("project-b", "owner/b"),
    )
    store, reader, writer, _state = make_store(
        tmp_path,
        repository_authority=verified,
    )
    with pytest.raises(ChromaWriteAuthorityViolation, match="cross_project"):
        store.store_github_contexts(
            [
                {"repository": "owner/a", "project_id": "project-a"},
                {"repository": "owner/b", "project_id": "project-b"},
            ]
        )
    with pytest.raises(ChromaWriteAuthorityViolation, match="authority_violation"):
        store.store_github_contexts(
            [{"repository": "owner/unknown", "project_id": "project-a"}]
        )
    assert reader.read_calls == []
    assert reader.vector_calls == []
    assert writer.calls == []


def test_github_cleanup_deletes_only_duplicate_ids_with_project_isolation(tmp_path):
    verified = authority(
        ("project-a", "owner/a"),
        ("project-b", "owner/b"),
    )
    canonical_a = MemoryVectorStore._record_id("github", "owner/a")
    canonical_b = MemoryVectorStore._record_id("github", "owner/b")
    state = {
        "profile_facts": {},
        "github_evidence": {
            canonical_a: github_record(canonical_a, "owner/a", {"repository": "owner/a"}, updated_at="20260102"),
            "duplicate-a": github_record("duplicate-a", "owner/a", {"repository": "owner/a"}, updated_at="20260101"),
            canonical_b: github_record(canonical_b, "owner/b", {"repository": "owner/b"}, updated_at="20260102"),
        },
    }
    store, _reader, writer, state = make_store(
        tmp_path,
        state=state,
        repository_authority=verified,
    )
    result = store.cleanup_github_repositories()
    assert result == {"canonicalized": 0, "deleted": 1}
    assert set(state["github_evidence"]) == {canonical_a, canonical_b}
    delete = next(call for call in writer.calls if call["operation"] == "delete")
    assert delete["ids"] == ["duplicate-a"]
    assert {item["project_id"] for item in delete["authority_metadata"]} == {"project-a"}


def test_http_write_failure_has_no_embedded_fallback_or_false_success(tmp_path):
    store, _reader, writer, _state = make_store(
        tmp_path,
        error=RuntimeError("transport_unavailable"),
    )
    with pytest.raises(RuntimeError, match="transport_unavailable"):
        store.replace_profile({"summary": "private"})
    assert len(writer.calls) == 1
    assert not (tmp_path / "chroma").exists()


def test_runtime_write_methods_do_not_reach_legacy_embedded_boundary():
    source = MEMORY_STORE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in (
        "replace_profile",
        "delete_profile",
        "store_github_contexts",
        "cleanup_github_repositories",
        "_upsert_with_similarity",
    ):
        assert "_ensure_client" not in methods[name]
        method_tree = ast.parse(methods[name])
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in {"_profile", "_github"}
            for node in ast.walk(method_tree)
        )
    persistent_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PersistentClient"
    ]
    assert persistent_calls == []
    assert "_ensure_client" not in methods
