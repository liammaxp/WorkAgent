from __future__ import annotations

import json

from backend import chroma_http_vector_search as http_vectors
from backend.memory_store import MemoryVectorStore


ENABLED = {
    http_vectors.VECTOR_QUERY_BACKEND_ENV: http_vectors.CHROMA_HTTP_VECTOR_QUERY_BACKEND,
    http_vectors.CHROMA_HTTP_HOST_ENV: "127.0.0.1",
    http_vectors.CHROMA_HTTP_PORT_ENV: "8100",
    http_vectors.CHROMA_HTTP_SSL_ENV: "0",
    http_vectors.CHROMA_HTTP_TIMEOUT_ENV: "2.5",
}

AUTHORITY_MAPPING = {
    "repository_to_project": {
        "liammaxp/other": "other_project",
        "liammaxp/workagent": "workagent",
    },
    "alias_to_repository": {},
    "conflicts": [],
    "unmapped_projects": [],
    "mapping_count": 2,
}


class Embedder:
    def __init__(self):
        self.queries = []

    def embed(self, query):
        self.queries.append(query)
        return [0.0] * (http_vectors.EXPECTED_EMBEDDING_DIMENSIONS - 1) + [1.0]


class Collection:
    name = http_vectors.GITHUB_EVIDENCE_COLLECTION
    metadata = {"hnsw:space": http_vectors.EXPECTED_DISTANCE_METRIC}

    def __init__(self, records=None):
        self.records = records or [
            (
                "vec_workagent",
                0.2,
                {
                    "repository": "liammaxp/WorkAgent",
                    "path": "backend/retrieval.py",
                    "raw patch": "diff --git",
                    "embedding": "not-safe",
                },
            ),
            ("vec_other", 0.1, {"repository": "liammaxp/other"}),
            ("vec_unresolved", 0.0, {"repository": "liammaxp/unresolved"}),
        ]
        self.query_kwargs = None
        self.get_kwargs = None

    def count(self):
        return len(self.records)

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        values = self.records[: kwargs["n_results"]]
        return {
            "ids": [[item[0] for item in values]],
            "distances": [[item[1] for item in values]],
            "metadatas": [[item[2] for item in values]],
            "documents": [["diff --git API_KEY=fake"]],
            "embeddings": [[[0.1]]],
        }

    def get(self, **kwargs):
        self.get_kwargs = kwargs
        return {
            "ids": [item[0] for item in self.records],
            "metadatas": [item[2] for item in self.records],
            "documents": ["must-not-be-consumed" for _ in self.records],
            "embeddings": [[0.1] for _ in self.records],
        }

    def __getattr__(self, name):
        if name in {"add", "upsert", "update", "delete", "peek", "get_or_create_collection"}:
            raise AssertionError(f"write or unsafe collection method called: {name}")
        raise AttributeError(name)


class Client:
    def __init__(self, collection=None, error=None):
        self.collection = collection or Collection()
        self.error = error
        self.requested_collection = None
        self.closed = False

    def get_collection(self, *, name):
        self.requested_collection = name
        if self.error is not None:
            raise self.error
        return self.collection

    def close(self):
        self.closed = True

    def __getattr__(self, name):
        if name in {"get_or_create_collection", "create_collection", "delete_collection", "reset"}:
            raise AssertionError(f"mutating client method called: {name}")
        raise AttributeError(name)


def _factory(client, calls):
    def create(**kwargs):
        calls.append(kwargs)
        return client
    return create


def _mapping(monkeypatch):
    monkeypatch.setattr(http_vectors, "_load_authority_mapping", lambda authority=None: AUTHORITY_MAPPING)


def test_backend_defaults_to_disabled_and_invalid_values_fail_closed():
    assert http_vectors.get_vector_query_backend_config({})["backend"] == "disabled"
    assert http_vectors.get_vector_query_backend_config({
        http_vectors.VECTOR_QUERY_BACKEND_ENV: "unknown"
    })["backend"] == "disabled"
    for key, value in (
        (http_vectors.CHROMA_HTTP_HOST_ENV, "0.0.0.0"),
        (http_vectors.CHROMA_HTTP_PORT_ENV, "0"),
        (http_vectors.CHROMA_HTTP_PORT_ENV, "not-a-port"),
        (http_vectors.CHROMA_HTTP_SSL_ENV, "maybe"),
        (http_vectors.CHROMA_HTTP_TIMEOUT_ENV, "0"),
    ):
        values = dict(ENABLED, **{key: value})
        assert http_vectors.get_vector_query_backend_config(values)["backend"] == "disabled"


def test_disabled_or_unavailable_server_returns_empty_without_client(monkeypatch):
    calls = []
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=Embedder(), environ={}, client_factory=lambda **kwargs: calls.append(kwargs),
    ) == []
    monkeypatch.setattr(http_vectors, "_server_reachable", lambda config: False)
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=Embedder(), environ=ENABLED, client_factory=lambda **kwargs: calls.append(kwargs),
    ) == []
    assert calls == []


def test_http_query_uses_exact_collection_authoritative_embedder_and_safe_fields(monkeypatch):
    _mapping(monkeypatch)
    collection = Collection()
    client = Client(collection)
    calls = []
    embedder = Embedder()
    results = http_vectors.search_github_evidence_vectors_http(
        query="repository mapping evidence", n_results=100, project_id="workagent",
        embedder=embedder, environ=ENABLED, client_factory=_factory(client, calls),
        skip_socket_preflight=True,
    )

    assert embedder.queries == ["repository mapping evidence"]
    assert client.requested_collection == http_vectors.GITHUB_EVIDENCE_COLLECTION
    assert client.closed
    assert calls == [{
        "host": "127.0.0.1", "port": 8100, "ssl": False, "timeout_seconds": 2.5,
    }]
    assert collection.query_kwargs["n_results"] == len(collection.records)
    assert collection.query_kwargs["include"] == ["distances", "metadatas"]
    assert len(collection.query_kwargs["query_embeddings"]) == 1
    assert len(collection.query_kwargs["query_embeddings"][0]) == 384
    assert results == [{
        "vector_record_id": "vec_workagent",
        "distance": 0.2,
        "metadata": {
            "path": "backend/retrieval.py",
            "project_id": "workagent",
            "repository": "liammaxp/workagent",
        },
        "rank": 1,
    }]
    serialized = json.dumps(results, sort_keys=True)
    for forbidden in ("document", "embedding", "raw patch", "diff --git", "API_KEY="):
        assert forbidden not in serialized


def test_missing_collection_wrong_metric_bad_embedding_and_raw_errors_fail_closed(monkeypatch):
    _mapping(monkeypatch)
    bad_metric = Collection()
    bad_metric.metadata = {"hnsw:space": "l2"}
    clients = [Client(error=RuntimeError("PASSWORD=private")), Client(bad_metric)]
    for client in clients:
        assert http_vectors.search_github_evidence_vectors_http(
            query="repository mapping", n_results=5, project_id="workagent",
            embedder=Embedder(), environ=ENABLED, client_factory=lambda **kwargs: client,
            skip_socket_preflight=True,
        ) == []
        assert client.closed

    class WrongDimension:
        def embed(self, query):
            return [0.0] * 10

    calls = []
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=WrongDimension(), environ=ENABLED,
        client_factory=lambda **kwargs: calls.append(kwargs), skip_socket_preflight=True,
    ) == []
    assert calls == []


def test_query_top_k_and_results_are_bounded_in_server_rank_order(monkeypatch):
    _mapping(monkeypatch)
    records = [
        (
            f"vec_{index:02d}",
            index / 100,
            {"repository": "liammaxp/workagent"},
        )
        for index in range(30)
    ]
    collection = Collection(records=records)
    result = http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=10_000, project_id="workagent",
        embedder=Embedder(), environ=ENABLED,
        client_factory=lambda **kwargs: Client(collection), skip_socket_preflight=True,
    )

    assert collection.query_kwargs["n_results"] == http_vectors.MAX_VECTOR_TOP_K
    assert len(result) == http_vectors.MAX_VECTOR_TOP_K
    assert [item["vector_record_id"] for item in result] == [
        f"vec_{index:02d}" for index in range(http_vectors.MAX_VECTOR_TOP_K)
    ]


def test_runtime_http_path_never_constructs_embedded_client(monkeypatch, tmp_path):
    _mapping(monkeypatch)
    monkeypatch.setattr(http_vectors, "_server_reachable", lambda config: True)
    client = Client(Collection(records=[
        ("vec_workagent", 0.25, {"repository": "liammaxp/workagent"})
    ]))

    class ChromaBoundary:
        @staticmethod
        def HttpClient(**kwargs):
            return client

        @staticmethod
        def PersistentClient(**kwargs):
            raise AssertionError("embedded client must not be constructed")

    monkeypatch.setattr(http_vectors, "chromadb", ChromaBoundary)
    monkeypatch.setenv(http_vectors.VECTOR_QUERY_BACKEND_ENV, "chroma_http")
    monkeypatch.setenv(http_vectors.CHROMA_HTTP_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(http_vectors.CHROMA_HTTP_PORT_ENV, "8100")
    store = MemoryVectorStore(tmp_path / "chroma", tmp_path / "memory.json", tmp_path / "github")
    store._ensure_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("embedded store initialization must not run")
    )

    result = store.search_github_vector_records(
        "repository mapping", n_results=5, project_id="workagent"
    )
    assert len(result) == 1 and result[0]["metadata"]["project_id"] == "workagent"
    assert not (tmp_path / "chroma").exists()


def test_logical_fingerprint_is_bounded_safe_and_order_independent(monkeypatch):
    _mapping(monkeypatch)
    records = [
        ("vec_b", 0.2, {"repository": "liammaxp/workagent", "source body": "private"}),
        ("vec_a", 0.1, {"repository": "liammaxp/other", "embedding": "private"}),
    ]
    first_collection = Collection(records=records)
    second_collection = Collection(records=list(reversed(records)))
    first = http_vectors.compute_github_evidence_logical_fingerprint_http(
        environ=ENABLED, client_factory=lambda **kwargs: Client(first_collection),
        skip_socket_preflight=True,
    )
    second = http_vectors.compute_github_evidence_logical_fingerprint_http(
        environ=ENABLED, client_factory=lambda **kwargs: Client(second_collection),
        skip_socket_preflight=True,
    )

    assert first == second
    assert first["status"] == "ready"
    assert first["record_count"] == 2
    assert first["record_ids"] == ["vec_a", "vec_b"]
    assert first["repositories"] == ["liammaxp/other", "liammaxp/workagent"]
    assert len(first["fingerprint"]) == 64
    assert first_collection.get_kwargs == {"limit": 2, "include": ["metadatas"]}
    serialized = json.dumps(first, sort_keys=True)
    assert "source body" not in serialized and "embedding" not in serialized


def test_logical_fingerprint_changes_for_id_or_authorized_repository_metadata(monkeypatch):
    _mapping(monkeypatch)

    def fingerprint(records):
        return http_vectors.compute_github_evidence_logical_fingerprint_http(
            environ=ENABLED, client_factory=lambda **kwargs: Client(Collection(records=records)),
            skip_socket_preflight=True,
        )["fingerprint"]

    baseline = fingerprint([("vec_a", 0.1, {"repository": "liammaxp/workagent"})])
    changed_id = fingerprint([("vec_b", 0.1, {"repository": "liammaxp/workagent"})])
    changed_repo = fingerprint([("vec_a", 0.1, {"repository": "liammaxp/other"})])
    assert len({baseline, changed_id, changed_repo}) == 3


def test_readiness_metadata_inspection_requests_only_identity_metadata():
    collection = Collection()
    client = Client(collection)
    records = http_vectors.inspect_github_evidence_vector_metadata_http(
        environ=ENABLED,
        client_factory=lambda **kwargs: client,
        skip_socket_preflight=True,
    )

    assert collection.get_kwargs == {"limit": 3, "include": ["metadatas"]}
    assert records == [
        {"vector_record_id": "vec_workagent", "metadata": {"repository": "liammaxp/WorkAgent"}},
        {"vector_record_id": "vec_other", "metadata": {"repository": "liammaxp/other"}},
        {"vector_record_id": "vec_unresolved", "metadata": {"repository": "liammaxp/unresolved"}},
    ]
    serialized = json.dumps(records, sort_keys=True)
    for forbidden in ("document", "embedding", "raw patch", "diff --git", "API_KEY="):
        assert forbidden not in serialized
