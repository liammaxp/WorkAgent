from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend import project_retrieval_v2


PROJECT_A = {
    "project_id": "workagent",
    "project_name": "WorkAgent",
    "repo": "liammaxp/workagent",
}
PROJECT_B = {
    "project_id": "course_management_database",
    "project_name": "Course Management Database",
    "repo": "liammaxp/course-management-database-system",
}


def test_simultaneous_explicit_legacy_and_v2_modes_do_not_leak(monkeypatch):
    barrier = threading.Barrier(2)
    calls = []
    call_lock = threading.Lock()

    def legacy(project):
        barrier.wait(timeout=5)
        with call_lock:
            calls.append(("legacy", project["project_id"]))
        return [{"provider": "legacy", "project_id": project["project_id"], "values": []}]

    def v2(project, **_kwargs):
        barrier.wait(timeout=5)
        with call_lock:
            calls.append(("retrieval_v2", project["project_id"]))
        return [{"provider": "retrieval_v2", "project_id": project["project_id"], "values": []}]

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        legacy_future = executor.submit(
            api_server.retrieve_evidence_for_project_for_resume,
            PROJECT_A,
            retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY,
        )
        v2_future = executor.submit(
            api_server.retrieve_evidence_for_project_for_resume,
            PROJECT_B,
            retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
        )
        legacy_result = legacy_future.result(timeout=10)
        v2_result = v2_future.result(timeout=10)
    assert legacy_result == [{"provider": "legacy", "project_id": "workagent", "values": []}]
    assert v2_result == [{"provider": "retrieval_v2", "project_id": "course_management_database", "values": []}]
    assert set(calls) == {("legacy", "workagent"), ("retrieval_v2", "course_management_database")}
    legacy_result[0]["values"].append("changed")
    assert v2_result[0]["values"] == []


def test_interleaved_explicit_modes_ignore_later_environment_changes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda project: calls.append(("legacy", project["project_id"])) or [{"provider": "legacy"}],
    )
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda project, **_kwargs: calls.append(("retrieval_v2", project["project_id"])) or [{"provider": "retrieval_v2"}],
    )
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    first = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT_A,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY,
    )
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "0")
    second = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT_B,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
    )
    assert first == [{"provider": "legacy"}]
    assert second == [{"provider": "retrieval_v2"}]
    assert calls == [("legacy", "workagent"), ("retrieval_v2", "course_management_database")]


def test_v2_results_are_separately_allocated_for_each_request(monkeypatch):
    def v2(project, **_kwargs):
        return [{"project_id": project["project_id"], "repo": project["repo"], "labels": []}]

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)
    first = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT_A,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
    )
    second = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT_A,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
    )
    assert first == second
    assert first is not second and first[0] is not second[0] and first[0]["labels"] is not second[0]["labels"]
    first[0]["labels"].append("request-a")
    assert second[0]["labels"] == []


def test_explicit_mode_does_not_read_global_flag(monkeypatch):
    monkeypatch.setattr(
        project_retrieval_v2,
        "is_github_evidence_retrieval_v2_enabled",
        lambda: pytest.fail("explicit request mode must not re-read global configuration"),
    )
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda _project: [])
    assert api_server.retrieve_evidence_for_project_for_resume(
        PROJECT_A,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY,
    ) == []
