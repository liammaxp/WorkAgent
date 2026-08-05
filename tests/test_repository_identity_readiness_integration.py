from __future__ import annotations

from backend import evidence_index_readiness as readiness
from backend import project_repository_identity as identity


def memory(*project_ids):
    return {"projects": [{"project_id": value} for value in project_ids]}


def record(name, **metadata):
    return {"id": name, "metadata": metadata}


def authority(project_memory, *links):
    return identity.build_project_repository_identity_authority(
        project_memory=project_memory, user_confirmed_links=list(links),
    )


def test_repository_only_metadata_resolves_only_through_exact_confirmed_authority():
    project_memory = memory("A")
    confirmed = authority(project_memory, {
        "project_id": "A", "repository": "owner/repository", "aliases": ["ExactAlias"], "confirmed": True,
    })
    records = [record("canonical", repository="owner/repository"), record("alias", repository="ExactAlias"), record("fuzzy", repository="repository")]
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory, identity_authority=confirmed,
        vector_records=records, raw_sources=[], chunks=[],
    )
    assert result["records_resolved_by_authoritative_mapping"] == 2
    assert result["records_unresolved"] == 1
    assert result["identity_mapping_count"] == 1
    assert result["user_confirmation_required"] is True


def test_missing_or_conflicting_authority_remains_unresolved():
    project_memory = memory("A", "B")
    records = [record(str(index), repository=f"owner/repo-{index}") for index in range(5)]
    missing = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory, vector_records=records, raw_sources=[], chunks=[],
    )
    assert missing["status"] == "blocked"
    assert missing["records_resolved_by_authoritative_mapping"] == 0
    assert missing["records_unresolved"] == 5
    assert not missing["ready_for_hybrid_retrieval"]
    conflicting = authority(
        project_memory,
        {"project_id": "A", "repository": "owner/repo-0", "confirmed": True},
        {"project_id": "B", "repository": "owner/repo-0", "confirmed": True},
    )
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory, identity_authority=conflicting,
        vector_records=records, raw_sources=[], chunks=[],
    )
    assert result["records_resolved_by_authoritative_mapping"] == 0
    assert result["identity_authority_status"] == "blocked"


def test_explicit_vector_disagreement_is_conflicting_and_no_writes_or_builds_occur(tmp_path, monkeypatch):
    project_memory = memory("A", "B")
    confirmed = authority(project_memory, {
        "project_id": "A", "repository": "owner/repo", "confirmed": True,
    })
    forbidden = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        identity, "build_project_repository_identity_authority",
        lambda **_: (_ for _ in ()).throw(AssertionError("readiness must not build")),
    )
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory, identity_authority=confirmed,
        vector_records=[record("conflict", project_id="B", repository="owner/repo")],
        raw_sources=[], chunks=[], manifest_path=forbidden,
    )
    assert result["records_with_conflicting_identity"] == 1
    assert not forbidden.exists()


def test_only_matching_repository_becomes_resolvable_when_chunks_exist():
    project_memory = memory("A")
    confirmed = authority(project_memory, {
        "project_id": "A", "repository": "owner/match", "confirmed": True,
    })
    result = readiness.inspect_evidence_index_readiness(
        project_memory=project_memory, identity_authority=confirmed,
        vector_records=[record("one", repository="owner/match"), record("two", repository="owner/other")],
        raw_sources=[], chunks=[{"chunk_id": "chunk-a", "project_id": "A"}], intended_project_ids=["A"],
    )
    assert result["records_resolved_by_authoritative_mapping"] == 1
    assert result["records_unresolved"] == 1
    assert result["ready_for_hybrid_retrieval"] is False
