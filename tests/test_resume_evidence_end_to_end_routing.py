from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend import project_retrieval_v2


PROJECT = {
    "project_id": "workagent",
    "project_name": "WorkAgent",
    "repo": "liammaxp/workagent",
    "tech_stack": ["Python", "React"],
    "workflows": ["retrieval validation"],
}
JD_REQUIREMENTS = {"technologies": ["Python"], "requirements": ["API retrieval validation"]}


class WriterBoundaryReached(RuntimeError):
    pass


class ProjectLoopReached(RuntimeError):
    pass


def adapted_evidence(project=PROJECT, count=2):
    hits = []
    for index in range(count):
        hits.append(
            {
                "project_id": project["project_id"],
                "chunk_id": f"chk_{project['project_id']}_{index}",
                "source_id": f"raw_{project['project_id']}_{index}",
                "repo": project["repo"],
                "path": f"backend/module_{index}.py",
                "commit_sha": "abc123",
                "source_type": "file_snapshot",
                "chunk_type": "file_section",
                "symbol": "retrieve_evidence",
                "score": 0.9 - index * 0.01,
                "keyword_score": 0.7,
                "symbol_score": 0.0,
                "vector_score": 0.8,
                "search_sources": ["keyword", "vector"],
                "query_groups": ["project_identity", "mechanisms"],
                "match_reasons": ["keyword_exact:retrieval"],
                "summary": "Safe retrieval validation summary",
                "keywords": ["Python", "retrieval"],
                "technical_tags": ["validation"],
                "text_hash": "a" * 64,
                "text_chars": 128,
                "raw_text": "diff --git API_KEY=fake",
                "embedding": [0.1],
            }
        )
    return project_retrieval_v2.adapt_hybrid_hits_for_resume_evidence(
        hits,
        project_id=project["project_id"],
        project_name=project["project_name"],
        authorized_repositories=[project["repo"]],
        limit=count,
    )


def prepare_resume_operation(monkeypatch, projects):
    monkeypatch.setattr(api_server.agent, "read_job_description", lambda: "Python API validation role")
    monkeypatch.setattr(api_server.agent, "read_resume", lambda: "Existing resume")
    monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: json.dumps({"projects": projects}))
    monkeypatch.setattr(api_server, "resolve_saved_application_hint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_server, "agent_progress_guidance_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(api_server, "classify_role_family", lambda *_args, **_kwargs: {"role_family": "software_engineering"})
    monkeypatch.setattr(api_server, "jd_requirements_for_prompt", lambda *_args, **_kwargs: JD_REQUIREMENTS)
    monkeypatch.setattr(api_server, "enrich_jd_profile_with_tech_ontology", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(
        api_server,
        "select_staged_projects_with_ranking",
        lambda *_args, **_kwargs: (
            projects,
            {
                "selected_projects": [
                    {"project_id": project["project_id"], "rank": index + 1, "bullet_budget": 2}
                    for index, project in enumerate(projects)
                ]
            },
        ),
    )
    monkeypatch.setattr(api_server, "get_completed_resume_candidate_checkpoint", lambda *_args, **_kwargs: None)


def test_flag_off_actual_orchestration_reaches_writer_with_exact_legacy_evidence(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "0")
    prepare_resume_operation(monkeypatch, [PROJECT])
    expected = [
        {"repository": PROJECT["repo"], "description": "legacy-one", "order": 1},
        {"repository": PROJECT["repo"], "description": "legacy-two", "order": 2},
    ]
    expected_serialized = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    calls = {"legacy": 0, "v2": 0}

    def legacy(project):
        calls["legacy"] += 1
        assert project is PROJECT
        return expected

    def v2_forbidden(*_args, **_kwargs):
        calls["v2"] += 1
        pytest.fail("v2 I/O must not run in the flag-off operation")

    def capture_writer(*_args, **kwargs):
        evidence = kwargs["evidence"]
        assert evidence is expected
        assert json.dumps(evidence, ensure_ascii=False, sort_keys=True) == expected_serialized
        raise WriterBoundaryReached()

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2_forbidden)
    monkeypatch.setattr(api_server, "run_resume_bullet_writer_tool", capture_writer)
    with pytest.raises(WriterBoundaryReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=True))
    assert calls == {"legacy": 1, "v2": 0}


def test_flag_on_actual_orchestration_reaches_writer_with_safe_v2_evidence(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    prepare_resume_operation(monkeypatch, [PROJECT])
    expected = adapted_evidence(count=api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    before = deepcopy(expected)
    calls = {"legacy": 0, "v2": 0}

    def v2(project, **kwargs):
        calls["v2"] += 1
        assert project is PROJECT
        assert kwargs == {
            "jd_targets": JD_REQUIREMENTS,
            "limit": api_server.RESUME_PROJECT_EVIDENCE_LIMIT,
        }
        return expected

    def capture_writer(*_args, **kwargs):
        evidence = kwargs["evidence"]
        assert evidence is expected
        assert len(evidence) == api_server.RESUME_PROJECT_EVIDENCE_LIMIT
        assert all("vector" in item["search_sources"] for item in evidence)
        raise WriterBoundaryReached()

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("legacy fallback is forbidden"),
    )
    monkeypatch.setattr(api_server, "run_resume_bullet_writer_tool", capture_writer)
    with pytest.raises(WriterBoundaryReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=True))
    assert calls == {"legacy": 0, "v2": 1}
    assert expected == before
    serialized = json.dumps(expected, ensure_ascii=False)
    assert "raw_text" not in serialized and "embedding" not in serialized and "API_KEY" not in serialized


def test_resume_operation_resolves_mode_once_for_all_selected_projects(monkeypatch):
    projects = [
        PROJECT,
        {
            "project_id": "course_management_database",
            "project_name": "Course Management Database",
            "repo": "liammaxp/course-management-database-system",
        },
    ]
    prepare_resume_operation(monkeypatch, projects)
    enabled_calls = []
    provider_calls = []
    captured = []

    def enabled_once():
        enabled_calls.append(True)
        return len(enabled_calls) == 1

    monkeypatch.setattr(project_retrieval_v2, "is_github_evidence_retrieval_v2_enabled", enabled_once)
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda project, **_kwargs: provider_calls.append(project["project_id"]) or adapted_evidence(project, 1),
    )
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("the resolved v2 operation must not switch to legacy"),
    )

    def capture_candidate(_job, _resume, project, evidence, *_args, **_kwargs):
        captured.append((project["project_id"], evidence))
        return {"project_id": project["project_id"], "project_name": project["project_name"], "final_bullets": []}

    monkeypatch.setattr(api_server, "build_project_resume_candidate", capture_candidate)
    monkeypatch.setattr(
        api_server,
        "attach_candidate_claims_to_project_ranking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProjectLoopReached()),
    )
    with pytest.raises(ProjectLoopReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=True))
    assert len(enabled_calls) == 1
    assert provider_calls == [project["project_id"] for project in projects]
    assert [project_id for project_id, _evidence in captured] == provider_calls


def test_v2_prompt_compaction_preserves_safe_provenance_without_mutation():
    evidence = adapted_evidence(count=api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    evidence[0]["raw_text"] = "diff --git PASSWORD=fake"
    evidence[0]["documents"] = ["private"]
    before = deepcopy(evidence)
    compacted = api_server.compact_github_evidence_for_prompt(evidence)
    assert evidence == before
    assert len(compacted) == api_server.RESUME_PROJECT_EVIDENCE_LIMIT
    assert [item["chunk_id"] for item in compacted] == [item["chunk_id"] for item in evidence]
    assert all("vector" in item["search_sources"] for item in compacted)
    assert len(api_server.serialize_resume_evidence_for_budget(compacted)) <= api_server.MAX_PROMPT_EVIDENCE_CHARS
    serialized = json.dumps(compacted, ensure_ascii=False)
    for forbidden in ("raw_text", "documents", "diff --git", "PASSWORD"):
        assert forbidden not in serialized
