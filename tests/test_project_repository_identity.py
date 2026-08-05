from __future__ import annotations

import json

from backend import project_repository_identity as identity


def memory(*projects):
    return {"projects": list(projects)}


def test_repository_normalization_accepts_explicit_github_forms_and_rejects_unsafe_values():
    expected = "owner/repo"
    accepted = [
        " https://GitHub.com/Owner/Repo ",
        "https://github.com/Owner/Repo.git?token=discarded#fragment",
        "git@github.com:Owner/Repo.git",
        "ssh://git@github.com/Owner/Repo.git",
        "github.com/Owner/Repo/",
        "Owner/Repo",
    ]
    assert {identity.normalize_repository_identity(value) for value in accepted} == {expected}
    for value in [
        "WorkAgent", "owner/", "/repo", "owner/../repo", "https://gitlab.com/owner/repo",
        "https://user:secret@github.com/owner/repo", "ssh://root@github.com/owner/repo",
        "api_key=secret", "owner/repo/extra",
    ]:
        assert identity.normalize_repository_identity(value) == ""
    assert identity.normalize_repository_identity(expected) == expected


def test_audit_uses_only_explicit_structured_cooccurrence():
    project_memory = memory(
        {"project_id": "A", "project_name": "Repo", "repository": "owner/repo"},
        {"project_id": "B", "project_name": "similar", "description": "github.com/x/similar", "readme": "owner/ignored"},
    )
    saved = [
        {"project_id": "B", "repository": "owner/second", "body": "private"},
        {"repository": "owner/unowned", "readme": "A"},
        {"project_id": "A", "nested": {"repository": "owner/not-scanned"}},
    ]
    audit = identity.audit_explicit_project_repository_links(
        project_memory=project_memory, saved_github_context=saved,
    )
    pairs = {(item["project_id"], item["repository_identity"]) for item in audit["candidates"]}
    assert pairs == {("A", "owner/repo"), ("B", "owner/second")}
    serialized = json.dumps(audit)
    assert "private" not in serialized and "owner/ignored" not in serialized


def test_confirmations_require_explicit_true_known_project_and_deduplicate():
    project_memory = memory({"project_id": "A"})
    links = [
        {"project_id": "A", "repository": "owner/repo", "aliases": ["Repo"], "confirmed": True},
        {"project_id": "A", "repository": "https://github.com/owner/repo", "aliases": ["repo"], "confirmed": True},
        {"project_id": "A", "repository": "owner/no", "confirmed": False},
        {"project_id": "Unknown", "repository": "owner/no", "confirmed": True},
        {"project_id": "A", "repository": "https://u:p@github.com/owner/no", "confirmed": True},
    ]
    result = identity.validate_user_confirmed_repository_links(project_memory=project_memory, links=links)
    assert result["accepted_count"] == 1
    assert result["rejected_count"] == 3


def test_conflicts_fail_closed_but_multiple_repositories_for_one_project_are_allowed():
    project_memory = memory({"project_id": "A"}, {"project_id": "B"})
    authority = identity.build_project_repository_identity_authority(
        project_memory=project_memory,
        user_confirmed_links=[
            {"project_id": "A", "repository": "owner/one", "aliases": ["shared"], "confirmed": True},
            {"project_id": "A", "repository": "owner/two", "confirmed": True},
            {"project_id": "B", "repository": "owner/one", "confirmed": True},
            {"project_id": "B", "repository": "owner/three", "aliases": ["shared"], "confirmed": True},
        ],
    )
    assert authority["status"] == "blocked"
    assert {item["repository_identity"] for item in authority["mappings"]} == {"owner/two", "owner/three"}
    assert {item["type"] for item in authority["conflicts"]} == {
        "repository_multiple_projects", "alias_multiple_targets",
    }
    mapping = identity.authority_to_repository_mapping(authority)
    assert mapping["repository_to_project"] == {"owner/three": "B", "owner/two": "A"}
    assert "owner/one" in mapping["conflicts"]


def test_authority_is_order_independent_bounded_and_contains_no_raw_content():
    project_memory = memory({"project_id": "A"}, {"project_id": "B"})
    links = [
        {"project_id": "A", "repository": "owner/one", "confirmed": True},
        {"project_id": "B", "repository": "owner/two", "confirmed": True},
    ]
    first = identity.build_project_repository_identity_authority(project_memory=project_memory, user_confirmed_links=links)
    second = identity.build_project_repository_identity_authority(project_memory=memory(*reversed(project_memory["projects"])), user_confirmed_links=list(reversed(links)))
    assert first == second
    assert first["status"] == "ready" and first["mapping_count"] == 2
    assert "raw_text" not in json.dumps(first)


def test_authority_persistence_is_atomic_idempotent_and_fails_closed(tmp_path, monkeypatch):
    authority = identity.build_project_repository_identity_authority(
        project_memory=memory({"project_id": "A", "repository": "owner/repo"})
    )
    path = tmp_path / "authority.json"
    assert identity.write_project_repository_identity_authority(authority, path)["status"] == "created"
    original = path.read_bytes()
    assert identity.write_project_repository_identity_authority(authority, path)["status"] == "unchanged"
    assert identity.load_project_repository_identity_authority(path) == authority
    path.write_text("{}", encoding="utf-8")
    assert identity.load_project_repository_identity_authority(path) is None
    path.write_bytes(original)
    monkeypatch.setattr(identity.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("no")))
    assert identity.write_project_repository_identity_authority(
        identity.build_project_repository_identity_authority(
            project_memory=memory({"project_id": "A", "repository": "owner/changed"})
        ), path,
    )["status"] == "error"
    assert path.read_bytes() == original


def test_missing_authority_has_safe_confirmation_report():
    report = identity.safe_repository_identity_report(None)
    assert report == {
        "unresolved_project_ids": [], "unresolved_repository_identities": [],
        "conflict_count": 0, "unresolved_count": 0, "mapping_count": 0,
        "requires_user_confirmation": True,
    }
