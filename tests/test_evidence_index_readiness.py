from __future__ import annotations

import json
import hashlib
import sqlite3

from backend import evidence_index_readiness as readiness
from backend import github_evidence_chunks as chunking
from backend import project_repository_identity as identity
from backend.memory_store import MemoryVectorStore


def project_memory(*projects):
    return {"projects": list(projects)}


def project(project_id, repository=None, **extra):
    value = {"project_id": project_id, **extra}
    if repository is not None:
        value["repository"] = repository
    return value


def vector_record(name, **metadata):
    return {"id": f"vec_{name}", "metadata": metadata}


def chunk(project_id, name="one"):
    return chunking.build_github_evidence_chunk_record(
        source_id=f"raw_{name}", project_id=project_id, repo="owner/repo",
        chunk_type="file_section", text="retrieval evidence", raw_hash="a" * 64,
    )


def test_authoritative_mapping_is_unique_deterministic_and_alias_safe():
    memory = project_memory(
        project("ProjectA", "https://github.com/Owner/Repo-A", repository_aliases=["owner/repo-a.git"]),
        project("ProjectB", "owner/repo-b"),
    )
    first = readiness.build_authoritative_repository_project_mapping(memory)
    second = readiness.build_authoritative_repository_project_mapping(
        project_memory(*reversed(memory["projects"]))
    )
    assert first == second
    assert first["repository_to_project"] == {
        "owner/repo-a": "ProjectA", "owner/repo-b": "ProjectB",
    }
    assert first["mapping_count"] == 2


def test_mapping_conflicts_fuzzy_names_and_missing_memory_fail_closed():
    conflict = readiness.build_authoritative_repository_project_mapping(project_memory(
        project("ProjectA", "owner/shared"), project("ProjectB", "owner/shared"),
        project("ProjectC", "not a repository"),
    ))
    assert conflict["repository_to_project"] == {}
    assert conflict["conflicts"] == ["owner/shared"]
    assert conflict["unmapped_projects"] == ["ProjectC"]
    assert readiness.build_authoritative_repository_project_mapping(None)["mapping_count"] == 0
    fuzzy = readiness.build_authoritative_repository_project_mapping(
        project_memory(project("ProjectA", "owner/repository-a"))
    )
    assert "owner/repo-a" not in fuzzy["repository_to_project"]


def test_vector_metadata_inspection_is_read_only_and_sanitized():
    class Store:
        def __init__(self): self.calls = []
        def inspect_github_vector_metadata(self, limit):
            self.calls.append(limit)
            return [vector_record(
                "one", project_id="ProjectA", repository="owner/repo-a",
                raw_text="diff --git private", body="private", token="secret",
            )]
    store = Store()
    records = readiness.inspect_github_evidence_vector_metadata(vector_store=store)
    assert store.calls == [readiness.MAX_VECTOR_METADATA_RECORDS]
    assert records == [{
        "vector_record_id": "vec_one",
        "metadata": {"project_id": "ProjectA", "repository": "owner/repo-a"},
    }]
    assert "raw_text" not in json.dumps(records)


def test_memory_store_inspector_requests_metadata_only_without_creating_collection(tmp_path):
    class Collection:
        def __init__(self): self.include = None
        def count(self): return 1
        def get(self, **kwargs):
            self.include = kwargs["include"]
            return {"ids": ["vec_one"], "metadatas": [{"project_id": "ProjectA"}]}
    store = MemoryVectorStore(tmp_path / "chroma", tmp_path / "memory", tmp_path / "github")
    store._github = Collection()
    assert store.inspect_github_vector_metadata() == [{
        "vector_record_id": "vec_one", "metadata": {"project_id": "ProjectA"},
    }]
    assert store._github.include == ["metadatas"]


def test_memory_store_disk_inspector_is_immutable_sqlite_read_only(tmp_path):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    database_path = chroma_path / "chroma.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript("""
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT NOT NULL, collection TEXT NOT NULL);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT NOT NULL, embedding_id TEXT NOT NULL);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT);
        INSERT INTO collections VALUES ('collection', 'github_evidence');
        INSERT INTO segments VALUES ('metadata-segment', 'METADATA', 'collection');
        INSERT INTO embeddings VALUES (1, 'metadata-segment', 'vector-one');
        INSERT INTO embedding_metadata VALUES (1, 'repository', 'owner/repo');
        INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'private body');
    """)
    connection.commit(); connection.close()
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    store = MemoryVectorStore(chroma_path, tmp_path / "memory", tmp_path / "github")
    assert store.inspect_github_vector_metadata() == [{
        "vector_record_id": "vector-one", "metadata": {"repository": "owner/repo"},
    }]
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def test_disk_inspector_stays_immutable_when_live_collection_is_initialized(tmp_path):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    database_path = chroma_path / "chroma.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript("""
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT NOT NULL, collection TEXT NOT NULL);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT NOT NULL, embedding_id TEXT NOT NULL);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT);
        INSERT INTO collections VALUES ('collection', 'github_evidence');
        INSERT INTO segments VALUES ('metadata-segment', 'METADATA', 'collection');
        INSERT INTO embeddings VALUES (1, 'metadata-segment', 'vector-one');
        INSERT INTO embedding_metadata VALUES (1, 'repository', 'owner/repo');
    """)
    connection.commit(); connection.close()

    class LiveCollection:
        def count(self):
            raise AssertionError("readiness inspection must not open the live collection")

    store = MemoryVectorStore(chroma_path, tmp_path / "memory", tmp_path / "github")
    store._github = LiveCollection()
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    assert store.inspect_github_vector_metadata() == [{
        "vector_record_id": "vector-one", "metadata": {"repository": "owner/repo"},
    }]
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def test_memory_store_status_reads_reuse_immutable_metadata_inspector(tmp_path):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    database_path = chroma_path / "chroma.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript("""
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT NOT NULL, collection TEXT NOT NULL);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT NOT NULL, embedding_id TEXT NOT NULL);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT);
        INSERT INTO collections VALUES ('collection', 'github_evidence');
        INSERT INTO collections VALUES ('profile-collection', 'profile_facts');
        INSERT INTO segments VALUES ('metadata-segment', 'METADATA', 'collection');
        INSERT INTO segments VALUES ('profile-segment', 'METADATA', 'profile-collection');
        INSERT INTO embeddings VALUES (1, 'metadata-segment', 'vector-one');
        INSERT INTO embeddings VALUES (2, 'metadata-segment', 'vector-two');
        INSERT INTO embeddings VALUES (3, 'profile-segment', 'profile-one');
        INSERT INTO embedding_metadata VALUES (1, 'repository', 'owner/repo');
        INSERT INTO embedding_metadata VALUES (1, 'updated_at', '2026-08-01T00:00:00Z');
        INSERT INTO embedding_metadata VALUES (1, 'source', 'commit');
        INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'private body');
        INSERT INTO embedding_metadata VALUES (2, 'repository', 'owner/repo');
        INSERT INTO embedding_metadata VALUES (2, 'updated_at', '2026-08-02T00:00:00Z');
        INSERT INTO embedding_metadata VALUES (2, 'source', 'repository');
    """)
    connection.commit(); connection.close()

    class LiveCollection:
        def count(self):
            raise AssertionError("status reads must not open the live collection")

        def get(self, **_kwargs):
            raise AssertionError("status reads must not open the live collection")

    store = MemoryVectorStore(chroma_path, tmp_path / "memory", tmp_path / "github")
    store._github = LiveCollection()
    store._profile = LiveCollection()
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    assert store.profile_count() == 1
    assert store.github_count() == 2
    assert store.list_github_repositories() == [{
        "repository": "owner/repo", "updated_at": "2026-08-02T00:00:00Z",
    }]
    assert store.github_metadata_status() == {
        "available": True,
        "count": 2,
        "repositories": [{
            "repository": "owner/repo", "updated_at": "2026-08-02T00:00:00Z",
        }],
    }
    assert store.github_preview_metadata(limit=1) == [{
        "id": "vector-one",
        "repository": "owner/repo",
        "updated_at": "2026-08-01T00:00:00Z",
        "source": "commit",
    }]
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def test_identity_categories_and_authoritative_resolution_are_counted():
    memory = project_memory(project("ProjectA", "owner/repo-a"), project("ProjectB", "owner/repo-b"))
    authority = identity.build_project_repository_identity_authority(project_memory=memory)
    records = [
        vector_record("explicit", project_id="ProjectA", repository="owner/repo-a"),
        vector_record("resolved", repository="owner/repo-b"),
        vector_record("unresolved", repository="owner/unknown"),
        vector_record("missing"),
        vector_record("conflict", project_id="ProjectA", repository="owner/repo-b"),
        vector_record("fields", project_id="ProjectA", project_name="ProjectB"),
    ]
    result = readiness.inspect_evidence_index_readiness(
        project_memory=memory, vector_records=records, chunks=[chunk("ProjectA"), chunk("ProjectB", "two")],
        raw_sources=[], intended_project_ids=["ProjectA", "ProjectB"], identity_authority=authority,
    )
    assert result["records_with_explicit_project_id"] == 1
    assert result["records_with_repository_only"] == 2
    assert result["records_resolved_by_authoritative_mapping"] == 1
    assert result["records_with_missing_identity"] == 1
    assert result["records_with_conflicting_identity"] == 2
    assert result["records_unresolved"] == 4
    assert result["vector_ready"] is False
    assert result["ready_for_hybrid_retrieval"] is False


def test_readiness_gate_ready_partial_blocked_and_empty_states():
    memory = project_memory(project("ProjectA", "owner/repo-a"))
    record = vector_record("one", project_id="ProjectA", repository="owner/repo-a")
    evidence_chunk = chunk("ProjectA")
    manifest = {
        "schema_version": "github_evidence_materialization.v1", "status": "ready",
        "raw_source_count": 0, "chunk_count": 1,
    }
    ready = readiness.inspect_evidence_index_readiness(
        project_memory=memory, vector_records=[record], raw_sources=[], chunks=[evidence_chunk],
        manifest=manifest, intended_project_ids=["ProjectA"],
    )
    assert ready["status"] == "ready"
    assert ready["vector_ready"] and ready["chunk_mapping_ready"]
    assert ready["ready_for_hybrid_retrieval"]
    no_chunks = readiness.inspect_evidence_index_readiness(
        project_memory=memory, vector_records=[record], chunks=[], intended_project_ids=["ProjectA"]
    )
    assert no_chunks["status"] == "blocked" and not no_chunks["chunk_mapping_ready"]
    no_vectors = readiness.inspect_evidence_index_readiness(
        project_memory=memory, vector_records=[], chunks=[evidence_chunk], intended_project_ids=["ProjectA"]
    )
    assert no_vectors["status"] == "blocked" and not no_vectors["vector_ready"]
    empty = readiness.inspect_evidence_index_readiness(
        project_memory=memory, vector_records=[], chunks=[]
    )
    assert empty["status"] == "empty"


def test_manifest_hash_mismatch_blocks_chunk_readiness(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    chunk_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "github_evidence_materialization.v1", "status": "ready",
        "raw_source_count": 1, "chunk_count": 1,
        "raw_artifact_hash": "0" * 64, "chunk_artifact_hash": "0" * 64,
    }
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory(project("ProjectA", "owner/repo-a")),
        vector_records=[vector_record("one", project_id="ProjectA")],
        raw_sources=[{}], chunks=[chunk("ProjectA")], manifest=manifest,
        raw_source_path=raw_path, chunk_path=chunk_path, intended_project_ids=["ProjectA"],
    )
    assert not result["chunk_mapping_ready"]
    assert "materialization_manifest_mismatch" in result["warnings"]


def test_requested_manifest_path_must_exist_and_be_ready(tmp_path):
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory(project("ProjectA", "owner/repo-a")),
        vector_records=[vector_record("one", project_id="ProjectA")],
        raw_sources=[], chunks=[chunk("ProjectA")],
        manifest_path=tmp_path / "missing-manifest.json",
        intended_project_ids=["ProjectA"],
    )
    assert result["status"] != "ready"
    assert result["chunk_mapping_ready"] is False
    assert "materialization_manifest_mismatch" in result["warnings"]


def test_readiness_never_materializes_writes_or_exposes_content(tmp_path, monkeypatch):
    forbidden = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        "backend.github_evidence_materializer.materialize_saved_github_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    result = readiness.inspect_evidence_index_readiness(
        project_memory={}, vector_records=[vector_record("unsafe", body="diff --git private")],
        chunks=[], raw_sources=[], manifest_path=forbidden,
    )
    serialized = json.dumps(result, sort_keys=True)
    assert not forbidden.exists()
    assert "diff --git" not in serialized and "body" not in serialized


def test_malformed_readiness_inputs_fail_closed():
    result = readiness.inspect_evidence_index_readiness(
        project_memory=None, vector_records=[None, [], {"metadata": "bad"}],
        chunks=[None, {"project_id": object()}], raw_sources=object(),
    )
    assert result["vector_ready"] is False
    assert result["chunk_mapping_ready"] is False
    assert result["ready_for_hybrid_retrieval"] is False
