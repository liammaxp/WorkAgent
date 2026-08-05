from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

from backend import project_repository_mapping_service as service
from backend import project_retrieval_v2
from backend.project_repository_identity import load_project_repository_identity_authority


def memory(*projects):
    return {"projects": list(projects)}


def vector(name, **metadata):
    return {"id": name, "metadata": metadata}


def request(project_id="A", repository="owner/repo", **extra):
    return {"project_id": project_id, "repository": repository, "confirmed": True, **extra}


def paths(tmp_path):
    return tmp_path / "confirmations.json", tmp_path / "authority.json"


def test_unresolved_list_merges_canonical_records_and_marks_bare_aliases_safely():
    result = service.list_unresolved_repository_mappings(vector_records=[
        vector("1", repository="Owner/Repo", body="private"),
        vector("2", repo="https://github.com/owner/repo.git", raw_text="diff --git"),
        vector("3", repository="WorkAgent"),
        vector("4", repository="bad alias with spaces"),
        vector("5", project_id="A", repository="owner/already-explicit"),
    ])
    assert result["unresolved_count"] == 2
    assert result["repositories"][0] == {
        "repository": "owner/repo", "repository_alias": None, "repository_aliases": [],
        "canonical": True, "requires_canonical_repository": False,
        "vector_record_count": 2, "currently_mapped": False,
        "conflicting": False, "requires_confirmation": True,
    }
    assert result["repositories"][1]["repository"] is None
    assert result["repositories"][1]["repository_alias"] == "WorkAgent"
    assert result["repositories"][1]["requires_canonical_repository"] is True
    serialized = json.dumps(result)
    assert "private" not in serialized and "diff --git" not in serialized
    assert "project" not in result["repositories"][0]


def test_project_list_is_bounded_safe_and_deterministic():
    project_memory = memory(
        {"project_id": "B", "project_name": " Beta ", "description": "private", "evidence_facts": ["x"]},
        {"project_id": "A", "project_name": "Alpha", "capability_facts": ["secret"]},
        {"project_id": "bad id", "project_name": "ignored"},
    )
    result = service.list_repository_mapping_projects(project_memory=project_memory)
    assert result == {"status": "ready", "projects": [
        {"project_id": "A", "project_name": "Alpha", "already_linked_repositories": []},
        {"project_id": "B", "project_name": "Beta", "already_linked_repositories": []},
    ], "count": 2, "warnings": [], "errors": []}
    assert "description" not in json.dumps(result)


def test_valid_confirmation_creates_pair_and_duplicate_is_idempotent(tmp_path):
    confirmation_path, authority_path = paths(tmp_path)
    project_memory = memory({"project_id": "A", "project_name": "Alpha"})
    first = service.confirm_repository_mapping(
        project_memory=project_memory, request=request(),
        vector_records=[vector("1", repository="owner/repo")], raw_sources=[], chunks=[],
        confirmation_path=confirmation_path, authority_path=authority_path,
    )
    first_pair = (confirmation_path.read_bytes(), authority_path.read_bytes())
    second = service.confirm_repository_mapping(
        project_memory=project_memory, request=request(),
        vector_records=[vector("1", repository="owner/repo")], raw_sources=[], chunks=[],
        confirmation_path=confirmation_path, authority_path=authority_path,
    )
    assert first["status"] == "created" and second["status"] == "unchanged"
    assert first_pair == (confirmation_path.read_bytes(), authority_path.read_bytes())
    assert first["records_resolved_by_authoritative_mapping"] == 1
    assert first["identity_ready"] is True
    assert first["materialization_required"] is True
    assert first["ready_for_hybrid_retrieval"] is False


def test_confirmation_validation_fails_closed_without_writes(tmp_path):
    project_memory = memory({"project_id": "A"})
    invalid = [
        {"project_id": "A", "repository": "owner/repo", "confirmed": False},
        request(project_id="unknown"), request(repository="WorkAgent"),
        request(repository="https://user:secret@github.com/owner/repo"),
        request(repository="https://gitlab.com/owner/repo"),
        request(repository="owner/../repo"), request(extra_metadata="not allowed"),
    ]
    for index, value in enumerate(invalid):
        confirmation_path, authority_path = paths(tmp_path / str(index))
        confirmation_path.parent.mkdir()
        result = service.confirm_repository_mapping(
            project_memory=project_memory, request=value,
            confirmation_path=confirmation_path, authority_path=authority_path,
        )
        assert result["status"] in {"blocked", "error"}
        assert not confirmation_path.exists() and not authority_path.exists()


def test_conflict_does_not_modify_existing_pair_and_second_repo_for_project_is_allowed(tmp_path):
    confirmation_path, authority_path = paths(tmp_path)
    project_memory = memory({"project_id": "A"}, {"project_id": "B"})
    assert service.confirm_repository_mapping(
        project_memory=project_memory, request=request(), confirmation_path=confirmation_path,
        authority_path=authority_path, vector_records=[], raw_sources=[], chunks=[],
    )["status"] == "created"
    before = (confirmation_path.read_bytes(), authority_path.read_bytes())
    blocked = service.confirm_repository_mapping(
        project_memory=project_memory, request=request(project_id="B"), confirmation_path=confirmation_path,
        authority_path=authority_path, vector_records=[], raw_sources=[], chunks=[],
    )
    assert blocked["status"] == "blocked"
    assert before == (confirmation_path.read_bytes(), authority_path.read_bytes())
    allowed = service.confirm_repository_mapping(
        project_memory=project_memory, request=request(repository="owner/second"),
        confirmation_path=confirmation_path, authority_path=authority_path,
        vector_records=[], raw_sources=[], chunks=[],
    )
    assert allowed["status"] == "updated" and allowed["identity_mapping_count"] == 2


def test_authority_write_failure_rolls_back_confirmation_pair(tmp_path, monkeypatch):
    confirmation_path, authority_path = paths(tmp_path)
    project_memory = memory({"project_id": "A"})
    service.confirm_repository_mapping(
        project_memory=project_memory, request=request(), confirmation_path=confirmation_path,
        authority_path=authority_path, vector_records=[], raw_sources=[], chunks=[],
    )
    before = (confirmation_path.read_bytes(), authority_path.read_bytes())
    monkeypatch.setattr(service, "write_project_repository_identity_authority", lambda *_args, **_kwargs: {"status": "error"})
    result = service.confirm_repository_mapping(
        project_memory=project_memory, request=request(repository="owner/two"),
        confirmation_path=confirmation_path, authority_path=authority_path,
        vector_records=[], raw_sources=[], chunks=[],
    )
    assert result["status"] == "error"
    assert before == (confirmation_path.read_bytes(), authority_path.read_bytes())


def test_malformed_existing_authority_fails_closed_without_modifying_pair(tmp_path):
    confirmation_path, authority_path = paths(tmp_path)
    authority_path.write_text("{}", encoding="utf-8")
    before = authority_path.read_bytes()
    result = service.confirm_repository_mapping(
        project_memory=memory({"project_id": "A"}), request=request(),
        confirmation_path=confirmation_path, authority_path=authority_path,
    )
    assert result["status"] == "error"
    assert result["errors"] == ["identity_authority_invalid"]
    assert authority_path.read_bytes() == before and not confirmation_path.exists()


def test_authority_build_failure_is_safe_and_writes_nothing(tmp_path, monkeypatch):
    confirmation_path, authority_path = paths(tmp_path)
    monkeypatch.setattr(
        service, "build_project_repository_identity_authority",
        lambda **_: (_ for _ in ()).throw(RuntimeError("private failure")),
    )
    result = service.confirm_repository_mapping(
        project_memory=memory({"project_id": "A"}), request=request(),
        confirmation_path=confirmation_path, authority_path=authority_path,
    )
    assert result["status"] == "error"
    assert "private failure" not in json.dumps(result)
    assert not confirmation_path.exists() and not authority_path.exists()


def test_readiness_failure_is_degraded_after_durable_commit(tmp_path):
    confirmation_path, authority_path = paths(tmp_path)
    result = service.confirm_repository_mapping(
        project_memory=memory({"project_id": "A"}), request=request(),
        confirmation_path=confirmation_path, authority_path=authority_path,
        readiness_inspector=lambda **_: (_ for _ in ()).throw(RuntimeError("private path")),
    )
    assert result["status"] == "degraded"
    assert result["errors"] == ["readiness_inspection_failed"]
    assert confirmation_path.exists() and authority_path.exists()
    assert "private path" not in json.dumps(result)


def test_concurrent_identical_and_conflicting_confirmations_are_safe(tmp_path):
    confirmation_path, authority_path = paths(tmp_path)
    project_memory = memory({"project_id": "A"}, {"project_id": "B"})
    def submit(project_id):
        return service.confirm_repository_mapping(
            project_memory=project_memory, request=request(project_id=project_id),
            confirmation_path=confirmation_path, authority_path=authority_path,
            vector_records=[], raw_sources=[], chunks=[],
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        identical = list(executor.map(lambda _: submit("A"), range(4)))
    assert {item["status"] for item in identical} <= {"created", "unchanged"}
    with ThreadPoolExecutor(max_workers=2) as executor:
        conflicting = list(executor.map(submit, ["A", "B"]))
    assert sum(item["status"] in {"created", "updated", "unchanged"} for item in conflicting) == 1
    artifact = service.load_repository_confirmations(confirmation_path)
    authority = load_project_repository_identity_authority(authority_path)
    assert artifact is not None and len(artifact["confirmations"]) == 1
    assert authority is not None and authority["mapping_count"] == 1


def test_retrieval_and_materialization_boundaries_remain_inactive(monkeypatch):
    monkeypatch.delenv("USE_GITHUB_EVIDENCE_RETRIEVAL_V2", raising=False)
    monkeypatch.delenv("USE_GITHUB_EVIDENCE_MATERIALIZATION", raising=False)
    assert project_retrieval_v2.is_github_evidence_retrieval_v2_enabled() is False
    assert project_retrieval_v2.retrieve_evidence_for_project_v2({}) == []
