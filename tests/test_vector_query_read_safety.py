from __future__ import annotations

import inspect
import json

from backend import evidence_vector_search as vector_search
from backend.memory_store import MemoryVectorStore


def _store(tmp_path):
    return MemoryVectorStore(
        tmp_path / "chroma",
        tmp_path / "project_memory.json",
        tmp_path / "github_context",
    )


class _ForbiddenCollection:
    def __getattr__(self, name):
        raise AssertionError(f"Chroma collection access is forbidden: {name}")


def test_production_vector_reader_fails_closed_without_storage_or_collection(tmp_path):
    store = _store(tmp_path)
    store._ensure_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("PersistentClient must not be initialized")
    )
    store._github = _ForbiddenCollection()

    assert store.search_github_vector_records("repository mapping", n_results=5) == []
    assert not (tmp_path / "chroma").exists()


def test_production_vector_reader_rejects_malformed_queries_and_limits(tmp_path):
    store = _store(tmp_path)

    for query in (None, "", "   ", 42, [], {}):
        assert store.search_github_vector_records(query, n_results=5) == []
    for limit in (None, True, False, 0, -1, 1.5, "5"):
        assert store.search_github_vector_records("evidence", n_results=limit) == []
    assert store.search_github_vector_records("evidence", n_results=10_000) == []


def test_production_vector_reader_has_no_embedded_chroma_query_call_graph():
    source = inspect.getsource(MemoryVectorStore.search_github_vector_records)

    for forbidden in (
        "_ensure_client(",
        ".get_or_create_collection(",
        ".get_collection(",
        ".query(",
        ".get(",
        ".count(",
        ".peek(",
        ".add(",
        ".upsert(",
        ".update(",
        ".delete(",
        ".reset(",
        ".persist(",
    ):
        assert forbidden not in source


def test_safe_adapter_clamps_limits_filters_metadata_and_is_deterministic():
    private = "diff --git API_KEY=fake ACCESS_TOKEN=fake PASSWORD=fake BEGIN PRIVATE KEY"

    class Store:
        def __init__(self):
            self.calls = []

        def search_github_vector_records(self, query, n_results, *, project_id):
            self.calls.append((query, n_results, project_id))
            return [
                {
                    "id": "vec_b",
                    "distance": 0.25,
                    "metadata": {
                        "project_id": "workagent",
                        "repository": "liammaxp/workagent",
                        "path": "backend/memory_store.py",
                        "raw patch": private,
                        "source body": private,
                        "embedding array": [0.1, 0.2],
                    },
                    "document": private,
                    "documents": [private],
                    "embedding": [0.1],
                    "embeddings": [[0.1]],
                },
                {
                    "id": "vec_a",
                    "distance": 0.25,
                    "metadata": {
                        "project_id": "workagent",
                        "repository": "liammaxp/workagent",
                    },
                },
                {
                    "id": "cross_project",
                    "distance": 0.0,
                    "metadata": {
                        "project_id": "another-project",
                        "repository": "liammaxp/workagent",
                    },
                },
            ]

    store = Store()
    first = vector_search.search_existing_github_evidence_vectors(
        query="repository mapping evidence",
        n_results=10_000,
        project_id="workagent",
        memory_store=store,
    )
    second = vector_search.search_existing_github_evidence_vectors(
        query="repository mapping evidence",
        n_results=10_000,
        project_id="workagent",
        memory_store=store,
    )

    assert first == second
    assert [item["vector_record_id"] for item in first] == ["vec_a", "vec_b"]
    assert all(call[1] == vector_search.MAX_VECTOR_TOP_K_PER_QUERY for call in store.calls)
    assert all(call[2] == "workagent" for call in store.calls)
    serialized = json.dumps(first, sort_keys=True)
    for forbidden in (
        "document",
        "embedding",
        "raw patch",
        "source body",
        "diff --git",
        "PRIVATE KEY",
        "API_KEY=",
        "ACCESS_TOKEN=",
        "PASSWORD=",
    ):
        assert forbidden not in serialized


def test_safe_adapter_exposes_no_raw_exception_details():
    class Store:
        def search_github_vector_records(self, query, n_results, *, project_id):
            raise RuntimeError("PASSWORD=do-not-expose")

    assert vector_search.search_existing_github_evidence_vectors(
        query="repository mapping",
        project_id="workagent",
        memory_store=Store(),
    ) == []
