from __future__ import annotations

import json

from backend import chroma_http_vector_search as http_vectors
from backend.memory_store import MemoryVectorStore
from backend.chroma_read_models import (
    ChromaReadRecord,
    ChromaReadResult,
    ChromaVectorHit,
    ChromaVectorResult,
)


ENABLED = {
    http_vectors.VECTOR_QUERY_BACKEND_ENV: http_vectors.CHROMA_HTTP_VECTOR_QUERY_BACKEND,
    "CHROMA_DEPLOYMENT_MODE": "local_http",
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


class SemanticReader:
    def __init__(self, records=None, error=None):
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
        self.error = error
        self.vector_kwargs = None
        self.read_kwargs = None

    def vector_query(self, semantic_collection_id, **kwargs):
        self.vector_kwargs = {"semantic_collection_id": semantic_collection_id, **kwargs}
        if self.error is not None:
            raise self.error
        values = self.records[: kwargs["n_results"]]
        return ChromaVectorResult(
            semantic_collection_id,
            tuple(
                ChromaVectorHit(
                    record_id=record_id,
                    distance=distance,
                    document=None,
                    metadata=metadata,
                    rank=rank,
                )
                for rank, (record_id, distance, metadata) in enumerate(values, start=1)
            ),
        )

    def read_records(self, semantic_collection_id, **kwargs):
        self.read_kwargs = {"semantic_collection_id": semantic_collection_id, **kwargs}
        if self.error is not None:
            raise self.error
        return ChromaReadResult(
            semantic_collection_id,
            tuple(
                ChromaReadRecord(record_id, None, metadata)
                for record_id, _distance, metadata in self.records
            ),
        )


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


def test_disabled_or_unavailable_server_returns_empty_without_reader_call(monkeypatch):
    _mapping(monkeypatch)
    reader = SemanticReader(error=RuntimeError("PASSWORD=private"))
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=Embedder(), environ={}, read_client=reader,
    ) == []
    assert reader.vector_kwargs is None
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=Embedder(), environ=ENABLED, read_client=reader,
    ) == []


def test_http_query_uses_semantic_reader_authoritative_embedder_and_safe_fields(monkeypatch):
    _mapping(monkeypatch)
    reader = SemanticReader()
    embedder = Embedder()
    results = http_vectors.search_github_evidence_vectors_http(
        query="repository mapping evidence", n_results=100, project_id="workagent",
        embedder=embedder, environ=ENABLED, read_client=reader,
    )

    assert embedder.queries == ["repository mapping evidence"]
    assert reader.vector_kwargs["semantic_collection_id"] == "github_evidence"
    assert reader.vector_kwargs["consumer_id"] == "github_evidence_vector_reader"
    assert reader.vector_kwargs["n_results"] == http_vectors.MAX_VECTOR_TOP_K
    assert reader.vector_kwargs["include_documents"] is False
    assert reader.vector_kwargs["include_distances"] is True
    assert len(reader.vector_kwargs["query_embedding"]) == 384
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


def test_bad_embedding_authority_and_reader_errors_fail_closed(monkeypatch):
    _mapping(monkeypatch)

    class WrongDimension:
        def embed(self, query):
            return [0.0] * 10

    reader = SemanticReader()
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=WrongDimension(), environ=ENABLED, read_client=reader,
    ) == []
    assert reader.vector_kwargs is None
    assert http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=5, project_id="workagent",
        embedder=Embedder(), environ=ENABLED,
        read_client=SemanticReader(error=RuntimeError("raw private response")),
    ) == []


def test_query_top_k_and_results_are_bounded_in_server_rank_order(monkeypatch):
    _mapping(monkeypatch)
    records = [
        (f"vec_{index:02d}", index / 100, {"repository": "liammaxp/workagent"})
        for index in range(30)
    ]
    reader = SemanticReader(records=records)
    result = http_vectors.search_github_evidence_vectors_http(
        query="repository mapping", n_results=10_000, project_id="workagent",
        embedder=Embedder(), environ=ENABLED, read_client=reader,
    )

    assert reader.vector_kwargs["n_results"] == http_vectors.MAX_VECTOR_TOP_K
    assert len(result) == http_vectors.MAX_VECTOR_TOP_K
    assert [item["vector_record_id"] for item in result] == [
        f"vec_{index:02d}" for index in range(http_vectors.MAX_VECTOR_TOP_K)
    ]


def test_runtime_vector_path_never_initializes_embedded_store(monkeypatch, tmp_path):
    _mapping(monkeypatch)
    reader = SemanticReader(records=[
        ("vec_workagent", 0.25, {"repository": "liammaxp/workagent"})
    ])
    monkeypatch.setattr(
        http_vectors,
        "_semantic_reader",
        lambda **_kwargs: reader,
    )
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
    first_reader = SemanticReader(records=records)
    second_reader = SemanticReader(records=list(reversed(records)))
    first = http_vectors.compute_github_evidence_logical_fingerprint_http(
        environ=ENABLED, read_client=first_reader,
    )
    second = http_vectors.compute_github_evidence_logical_fingerprint_http(
        environ=ENABLED, read_client=second_reader,
    )

    assert first == second
    assert first["status"] == "ready"
    assert first["record_count"] == 2
    assert first["record_ids"] == ["vec_a", "vec_b"]
    assert first["repositories"] == ["liammaxp/other", "liammaxp/workagent"]
    assert len(first["fingerprint"]) == 64
    assert first_reader.read_kwargs["include_documents"] is False
    assert first_reader.read_kwargs["max_records"] == http_vectors.MAX_FINGERPRINT_RECORDS
    serialized = json.dumps(first, sort_keys=True)
    assert "source body" not in serialized and "embedding" not in serialized


def test_logical_fingerprint_changes_for_id_or_authorized_repository_metadata(monkeypatch):
    _mapping(monkeypatch)

    def fingerprint(records):
        return http_vectors.compute_github_evidence_logical_fingerprint_http(
            environ=ENABLED, read_client=SemanticReader(records=records),
        )["fingerprint"]

    baseline = fingerprint([("vec_a", 0.1, {"repository": "liammaxp/workagent"})])
    changed_id = fingerprint([("vec_b", 0.1, {"repository": "liammaxp/workagent"})])
    changed_repo = fingerprint([("vec_a", 0.1, {"repository": "liammaxp/other"})])
    assert len({baseline, changed_id, changed_repo}) == 3


def test_readiness_metadata_inspection_requests_only_identity_metadata():
    reader = SemanticReader()
    records = http_vectors.inspect_github_evidence_vector_metadata_http(
        environ=ENABLED,
        read_client=reader,
    )

    assert reader.read_kwargs["include_documents"] is False
    assert reader.read_kwargs["metadata_fields"] == http_vectors._FINGERPRINT_METADATA_KEYS
    assert records == [
        {"vector_record_id": "vec_workagent", "metadata": {"repository": "liammaxp/WorkAgent"}},
        {"vector_record_id": "vec_other", "metadata": {"repository": "liammaxp/other"}},
        {"vector_record_id": "vec_unresolved", "metadata": {"repository": "liammaxp/unresolved"}},
    ]
    serialized = json.dumps(records, sort_keys=True)
    for forbidden in ("document", "embedding", "raw patch", "diff --git", "API_KEY="):
        assert forbidden not in serialized


def test_vector_wrapper_has_no_direct_chroma_or_httpx_client_ownership():
    source = (http_vectors.__file__ and open(http_vectors.__file__, encoding="utf-8").read())
    assert "import chromadb" not in source
    assert "chromadb.Http" + "Client" not in source
    assert "httpx.Client" not in source
    assert "socket.create_connection" not in source
