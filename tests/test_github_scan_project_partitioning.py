from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
from backend.chroma_write_client import ChromaWriteAuthorityViolation  # noqa: E402
from backend.project_repository_identity import (  # noqa: E402
    authority_to_repository_mapping,
    build_project_repository_identity_authority,
    normalize_repository_identity,
)


ACCEPTED_REPOSITORIES = (
    ("a_hike_through_time", "liammaxp/a-hike-through-time"),
    ("course_management_database", "liammaxp/course-management-database-system"),
    ("event_lottery_system", "liammaxp/event-lottery-system-application"),
    ("furniture_search_inventory", "liammaxp/furniture-search-and-inventory-system"),
    ("workagent", "liammaxp/workagent"),
)


def authority(*pairs: tuple[str, str]):
    return build_project_repository_identity_authority(
        project_memory={
            "projects": [
                {"project_id": project_id, "repository": repository}
                for project_id, repository in pairs
            ]
        }
    )


def contexts(*repositories: str) -> list[dict[str, object]]:
    return [
        {
            "repository": repository,
            "url": f"https://github.com/{repository}",
            "description": f"Controlled context for {repository}",
        }
        for repository in repositories
    ]


class PersistenceSpy:
    def __init__(self, identity_authority, *, fail_project_id: str = ""):
        self.mapping = authority_to_repository_mapping(identity_authority)
        self.fail_project_id = fail_project_id
        self.calls: list[dict[str, object]] = []

    def __call__(self, repo_contexts):
        context_project_ids = {
            str(context.get("project_id") or self.mapping["repository_to_project"].get(
                normalize_repository_identity(context.get("repository"))
            ) or "")
            for context in repo_contexts
        }
        expected_project_id = next(iter(context_project_ids)) if len(context_project_ids) == 1 else ""
        self.calls.append(
            {
                "expected_project_id": expected_project_id,
                "context_project_ids": context_project_ids,
                "repositories": tuple(context["repository"] for context in repo_contexts),
            }
        )
        if len(context_project_ids) != 1:
            raise ChromaWriteAuthorityViolation("github_cross_project_batch_rejected")
        if expected_project_id == self.fail_project_id:
            raise RuntimeError("private storage failure detail")
        return Path("private/chroma/path")


def persist(monkeypatch, identity_authority, repo_contexts, *, fail_project_id: str = ""):
    spy = PersistenceSpy(identity_authority, fail_project_id=fail_project_id)
    monkeypatch.setattr(api_server.agent, "save_github_context_output", spy)
    result = api_server.persist_github_contexts_by_project(
        repo_contexts,
        identity_authority=identity_authority,
    )
    return result, spy.calls


def assert_project_isolated(calls):
    assert all(
        call["context_project_ids"] == {call["expected_project_id"]}
        for call in calls
    )


def test_single_project_scan_keeps_one_existing_storage_call(monkeypatch):
    verified = authority(("project-a", "owner/repo"))
    result, calls = persist(monkeypatch, verified, contexts("owner/repo"))

    assert result["status"] == "ready"
    assert result["persisted_project_ids"] == ["project-a"]
    assert calls == [
        {
            "expected_project_id": "project-a",
            "context_project_ids": {"project-a"},
            "repositories": ("owner/repo",),
        }
    ]


def test_five_project_scan_all_uses_five_project_isolated_storage_calls(monkeypatch):
    verified = authority(*ACCEPTED_REPOSITORIES)
    result, calls = persist(
        monkeypatch,
        verified,
        contexts(*(repository for _project_id, repository in reversed(ACCEPTED_REPOSITORIES))),
    )

    assert result["status"] == "ready"
    assert len(calls) == 5
    assert [call["expected_project_id"] for call in calls] == sorted(
        project_id for project_id, _repository in ACCEPTED_REPOSITORIES
    )
    assert_project_isolated(calls)


def test_multiple_repositories_for_one_project_share_one_storage_call(monkeypatch):
    verified = authority(
        ("project-a", "owner/a-one"),
        ("project-a", "owner/a-two"),
        ("project-a", "owner/a-three"),
    )
    result, calls = persist(
        monkeypatch,
        verified,
        contexts("owner/a-three", "owner/a-one", "owner/a-two"),
    )

    assert result["persisted_project_ids"] == ["project-a"]
    assert len(calls) == 1
    assert calls[0]["repositories"] == ("owner/a-one", "owner/a-three", "owner/a-two")
    assert_project_isolated(calls)


def test_unresolved_repository_is_reported_and_never_persisted(monkeypatch):
    verified = authority(("project-a", "owner/known"))
    result, calls = persist(
        monkeypatch,
        verified,
        contexts("owner/unknown", "owner/known"),
    )

    assert result["status"] == "partial"
    assert result["unresolved_repository_identities"] == ["owner/unknown"]
    assert calls[0]["repositories"] == ("owner/known",)
    assert all("owner/unknown" not in call["repositories"] for call in calls)


def test_conflicting_mapping_fails_closed_for_only_that_repository(monkeypatch):
    conflicting = authority(
        ("project-a", "owner/conflict"),
        ("project-b", "owner/conflict"),
        ("project-c", "owner/valid"),
    )
    result, calls = persist(
        monkeypatch,
        conflicting,
        contexts("owner/conflict", "owner/valid"),
    )

    assert result["status"] == "partial"
    assert result["conflicting_repository_identities"] == ["owner/conflict"]
    assert calls == [
        {
            "expected_project_id": "project-c",
            "context_project_ids": {"project-c"},
            "repositories": ("owner/valid",),
        }
    ]


def test_one_project_failure_does_not_stop_or_contaminate_other_projects(monkeypatch):
    verified = authority(
        ("project-a", "owner/a"),
        ("project-b", "owner/b"),
        ("project-c", "owner/c"),
    )
    result, calls = persist(
        monkeypatch,
        verified,
        contexts("owner/c", "owner/b", "owner/a"),
        fail_project_id="project-b",
    )

    assert result["status"] == "partial"
    assert result["persisted_project_ids"] == ["project-a", "project-c"]
    assert result["failed_project_ids"] == ["project-b"]
    assert [call["expected_project_id"] for call in calls] == ["project-a", "project-b", "project-c"]
    assert "private storage failure detail" not in json.dumps(result)
    assert_project_isolated(calls)


def test_grouping_and_aggregate_result_are_deterministic(monkeypatch):
    verified = authority(
        ("project-b", "owner/b-two"),
        ("project-a", "owner/a"),
        ("project-b", "owner/b-one"),
    )
    first, first_calls = persist(
        monkeypatch,
        verified,
        contexts("owner/b-two", "owner/a", "owner/b-one"),
    )
    second, second_calls = persist(
        monkeypatch,
        verified,
        contexts("owner/b-one", "owner/b-two", "owner/a"),
    )

    assert first == second
    assert first_calls == second_calls


def test_similar_unknown_repository_is_not_guessed(monkeypatch):
    verified = authority(("workagent", "liammaxp/workagent"))
    result, calls = persist(
        monkeypatch,
        verified,
        contexts("liammaxp/work-agent"),
    )

    assert result["status"] == "blocked"
    assert result["persisted_project_ids"] == []
    assert result["unresolved_repository_identities"] == ["liammaxp/work-agent"]
    assert calls == []


def test_aggregate_persistence_result_is_bounded_and_contains_no_raw_context(monkeypatch):
    verified = authority(("project-a", "owner/repo"))
    repo_contexts = contexts("owner/repo")
    repo_contexts[0]["readme"] = "private raw content"
    repo_contexts[0]["patch"] = "private diff"
    repo_contexts.append(
        {"repository": "https://private.invalid/repo?token=private-api-token"}
    )
    result, _calls = persist(monkeypatch, verified, repo_contexts)

    serialized = json.dumps(result)
    assert set(result) == {
        "status",
        "project_group_count",
        "persisted_project_ids",
        "failed_project_ids",
        "unresolved_repository_identities",
        "conflicting_repository_identities",
    }
    assert "private raw content" not in serialized
    assert "private diff" not in serialized
    assert "private-api-token" not in serialized
    assert "private/chroma/path" not in serialized
    assert result["unresolved_repository_identities"] == ["invalid_repository_identity"]


def test_product_fetch_scan_all_never_sends_a_mixed_storage_batch(monkeypatch, tmp_path):
    verified = authority(*ACCEPTED_REPOSITORIES)
    spy = PersistenceSpy(verified)
    repo_text = "\n".join(
        f"https://github.com/{repository}"
        for _project_id, repository in reversed(ACCEPTED_REPOSITORIES)
    )

    monkeypatch.setattr(api_server, "read_github_repo_source", lambda *_args, **_kwargs: repo_text)
    monkeypatch.setattr(api_server.agent, "read_github_identities", lambda: {"usernames": ["tester"]})
    monkeypatch.setattr(api_server.agent, "identity_has_values", lambda _value: True)
    monkeypatch.setattr(api_server, "load_github_repo_scan_state", lambda: {"repositories": {}})
    monkeypatch.setattr(api_server, "project_memory_prompt_hash", lambda: "controlled-prompt")
    monkeypatch.setattr(
        api_server,
        "fetch_github_remote_state",
        lambda repo, _previous: {
            "changed": True,
            "change_reason": "controlled test",
            "default_branch": "main",
            "latest_commit_sha": f"sha-{repo['repo']}",
            "checked_at": "2026-08-24T00:00:00",
        },
    )
    monkeypatch.setattr(api_server.agent, "fetch_github_repo_context", lambda repo: contexts(f"{repo['owner']}/{repo['repo']}")[0])
    monkeypatch.setattr(api_server.agent, "fetch_user_commits_for_repo", lambda *_args: [])
    monkeypatch.setattr(api_server, "save_project_tech_stack", lambda _context: None)
    monkeypatch.setattr(api_server.agent, "save_github_context_output", spy)
    monkeypatch.setattr(
        api_server.project_repository_mapping_service,
        "load_project_repository_identity_authority",
        lambda _path: verified,
    )
    monkeypatch.setattr(api_server, "is_github_evidence_enabled", lambda: False)
    monkeypatch.setattr(
        api_server,
        "update_project_memory_from_repo_analysis",
        lambda *_args, **_kwargs: {"updated": True, "additions": [], "project_memory": {}},
    )
    monkeypatch.setattr(api_server, "build_project_memory_status_summary", lambda *_args, **_kwargs: {"status": "updated"})
    monkeypatch.setattr(api_server, "save_github_repo_scan_state", lambda _state: None)
    monkeypatch.setattr(api_server, "assert_agent_task_not_cancelled", lambda: None)
    monkeypatch.setattr(api_server.agent, "PROJECT_MEMORY_PATH", tmp_path / "project_memory.json")

    response = api_server.fetch_github_context_api(approved=True)

    assert response["github_context_persistence"]["status"] == "ready"
    assert len(spy.calls) == 5
    assert_project_isolated(spy.calls)
