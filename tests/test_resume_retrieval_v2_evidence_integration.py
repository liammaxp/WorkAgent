from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend import chroma_http_vector_search
from backend import evidence_hybrid_retrieval
from backend import project_retrieval_v2


PROJECT_A = {
    "project_id": "workagent",
    "project_name": "WorkAgent",
    "repo": "liammaxp/workagent",
    "tech_stack": ["Python", "React"],
    "workflows": ["retrieval validation"],
}
PROJECT_B = {
    "project_id": "course_management_database",
    "project_name": "Course Management Database",
    "repo": "liammaxp/course-management-database-system",
    "tech_stack": ["Python", "database"],
    "workflows": ["API validation"],
}
JD_TARGETS = {"technologies": ["Python"], "requirements": ["API validation retrieval"]}


def hybrid_hit(project, name="one", *, unsafe=False):
    private = "diff --git API_KEY=fake ACCESS_TOKEN=fake PASSWORD=fake BEGIN PRIVATE KEY"
    value = {
        "project_id": project["project_id"],
        "chunk_id": f"chk_{project['project_id']}_{name}",
        "source_id": f"raw_{project['project_id']}_{name}",
        "repo": project["repo"],
        "path": f"backend/{name}.py",
        "commit_sha": "abc123",
        "source_type": "file_snapshot",
        "chunk_type": "file_section",
        "symbol": "retrieve_evidence",
        "score": 0.9,
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
    }
    if unsafe:
        value.update({
            "raw_text": private, "text": private, "patch": private,
            "readme_body": private, "document": private, "embedding": [0.1],
        })
    return value


def adapted(project, count=1, *, unsafe=False):
    return project_retrieval_v2.adapt_hybrid_hits_for_resume_evidence(
        [hybrid_hit(project, str(index), unsafe=unsafe) for index in range(count)],
        project_id=project["project_id"],
        project_name=project["project_name"],
        authorized_repositories=[project["repo"]],
        limit=count,
    )


class EvidenceBoundaryReached(RuntimeError):
    pass


def prepare_pipeline_to_evidence_boundary(monkeypatch, projects, captured):
    memory = {"projects": projects}
    monkeypatch.setattr(api_server.agent, "read_job_description", lambda: "Python API validation role")
    monkeypatch.setattr(api_server.agent, "read_resume", lambda: "Existing resume")
    monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: json.dumps(memory))
    monkeypatch.setattr(api_server, "resolve_saved_application_hint", lambda *_args, **_kwargs: {"company": "", "role": ""})
    monkeypatch.setattr(api_server, "agent_progress_guidance_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(api_server, "classify_role_family", lambda *_args, **_kwargs: {"role_family": "software_engineering"})
    monkeypatch.setattr(api_server, "jd_requirements_for_prompt", lambda *_args, **_kwargs: JD_TARGETS)
    monkeypatch.setattr(api_server, "enrich_jd_profile_with_tech_ontology", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(
        api_server,
        "select_staged_projects_with_ranking",
        lambda *_args, **_kwargs: (
            projects,
            {"selected_projects": [
                {"project_id": item["project_id"], "rank": index + 1, "bullet_budget": 2}
                for index, item in enumerate(projects)
            ]},
        ),
    )

    def capture_candidate(_job, _resume, project, evidence, *_args, **_kwargs):
        captured.append({"project": project, "evidence": evidence})
        return {
            "project_id": project["project_id"], "project_name": project["project_name"],
            "final_bullets": [], "recommended_bullets": [],
        }

    monkeypatch.setattr(api_server, "build_project_resume_candidate", capture_candidate)
    monkeypatch.setattr(
        api_server,
        "attach_candidate_claims_to_project_ranking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(EvidenceBoundaryReached()),
    )
    for name in (
        "run_resume_bullet_writer_tool", "safe_model_call", "build_skills_resume_candidate",
        "build_experience_resume_candidate", "build_summary_resume_candidate", "merge_staged_resume",
        "prepare_tailored_resume_for_save",
    ):
        monkeypatch.setattr(
            api_server,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"real downstream writer must not run: {_name}"),
        )
    monkeypatch.setattr(
        api_server.agent,
        "write_resume_bullets",
        lambda *_args, **_kwargs: pytest.fail("real bullet validation writer must not run"),
    )
    monkeypatch.setattr(
        api_server.agent,
        "save_tailored_resume",
        lambda *_args, **_kwargs: pytest.fail("resume persistence must not run"),
    )


@pytest.mark.parametrize("flag", (None, "0", "false", "invalid"))
def test_flag_off_wrapper_preserves_exact_legacy_object_and_performs_no_v2_io(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, raising=False)
    else:
        monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, flag)
    expected = [{"repository": PROJECT_A["repo"], "description": "legacy shape"}]
    calls = {"legacy": 0, "v2": 0, "readiness": 0, "hybrid": 0, "vector": 0}

    def legacy(project):
        calls["legacy"] += 1
        assert project is PROJECT_A
        return expected

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            pytest.fail(f"{name} must not run while disabled")
        return fail

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", forbidden("v2"))
    monkeypatch.setattr(project_retrieval_v2, "inspect_evidence_index_readiness", forbidden("readiness"))
    monkeypatch.setattr(evidence_hybrid_retrieval, "run_project_hybrid_retrieval", forbidden("hybrid"))
    monkeypatch.setattr(chroma_http_vector_search, "search_github_evidence_vectors_http", forbidden("vector"))

    result = api_server.retrieve_evidence_for_project_for_resume(PROJECT_A, jd_targets=JD_TARGETS)
    assert result is expected
    assert calls == {"legacy": 1, "v2": 0, "readiness": 0, "hybrid": 0, "vector": 0}


def test_flag_on_wrapper_passes_structured_jd_and_stricter_resume_limit(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    expected = adapted(PROJECT_A, api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    captured = {}

    def v2(project, **kwargs):
        captured.update(kwargs)
        assert project is PROJECT_A
        return expected

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)
    monkeypatch.setattr(
        api_server, "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("legacy fallback is forbidden"),
    )
    result = api_server.retrieve_evidence_for_project_for_resume(PROJECT_A, jd_targets=JD_TARGETS)
    assert result is expected
    assert captured == {"jd_targets": JD_TARGETS, "limit": api_server.RESUME_PROJECT_EVIDENCE_LIMIT}
    assert len(result) == 8
    assert all("vector" in item["search_sources"] for item in result)


@pytest.mark.parametrize("condition", ("empty", "blocked", "controlled_error"))
def test_flag_on_empty_blocked_or_controlled_error_never_falls_back(condition, monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    calls = {"legacy": 0, "v2": 0}
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: calls.__setitem__("legacy", calls["legacy"] + 1),
    )

    def safe_empty(*_args, **_kwargs):
        calls["v2"] += 1
        return []

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", safe_empty)
    assert api_server.retrieve_evidence_for_project_for_resume(PROJECT_A, jd_targets=JD_TARGETS) == []
    assert calls == {"legacy": 0, "v2": 1}


def test_multi_project_wrapper_keeps_overlapping_evidence_isolated(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda project, **_kwargs: adapted(project, 2),
    )
    results = {
        project["project_id"]: api_server.retrieve_evidence_for_project_for_resume(
            project, jd_targets=JD_TARGETS
        )
        for project in (PROJECT_A, PROJECT_B)
    }
    for project in (PROJECT_A, PROJECT_B):
        evidence = results[project["project_id"]]
        assert {item["project_id"] for item in evidence} == {project["project_id"]}
        assert {item["repo"] for item in evidence} == {project["repo"]}
    assert PROJECT_A["repo"] not in json.dumps(results[PROJECT_B["project_id"]])
    assert PROJECT_B["repo"] not in json.dumps(results[PROJECT_A["project_id"]])


def test_actual_staged_resume_path_routes_v2_to_writer_capture_without_real_writing(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    captured = []
    provider_calls = []
    prepare_pipeline_to_evidence_boundary(monkeypatch, [PROJECT_A, PROJECT_B], captured)

    def v2(project, **kwargs):
        provider_calls.append({"project_id": project["project_id"], **kwargs})
        return adapted(project, 2, unsafe=True)

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)
    monkeypatch.setattr(
        api_server, "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("legacy must not run while v2 is enabled"),
    )
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_staged(api_server.TailorBody())

    assert [item["project"]["project_id"] for item in captured] == [
        PROJECT_A["project_id"], PROJECT_B["project_id"],
    ]
    assert all(call["jd_targets"] == JD_TARGETS for call in provider_calls)
    assert all(call["limit"] == api_server.RESUME_PROJECT_EVIDENCE_LIMIT for call in provider_calls)
    serialized = json.dumps(captured, sort_keys=True)
    for forbidden in (
        "raw_text", "text\"", "diff --git", "patch", "readme_body", "document",
        "embedding", "API_KEY", "ACCESS_TOKEN", "PASSWORD", "PRIVATE KEY",
    ):
        assert forbidden not in serialized
    assert all(
        "vector" in evidence["search_sources"]
        for item in captured for evidence in item["evidence"]
    )


def test_actual_staged_resume_path_flag_off_preserves_legacy_shape_once_per_project(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "0")
    captured = []
    prepare_pipeline_to_evidence_boundary(monkeypatch, [PROJECT_A, PROJECT_B], captured)
    expected = {
        PROJECT_A["project_id"]: [{"repository": PROJECT_A["repo"], "description": "legacy-a"}],
        PROJECT_B["project_id"]: [{"repository": PROJECT_B["repo"], "description": "legacy-b"}],
    }
    calls = []

    def legacy(project):
        calls.append(project["project_id"])
        return expected[project["project_id"]]

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)
    monkeypatch.setattr(
        project_retrieval_v2, "retrieve_evidence_for_project_v2",
        lambda *_args, **_kwargs: pytest.fail("v2 must not run while disabled"),
    )
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_staged(api_server.TailorBody())
    assert calls == [PROJECT_A["project_id"], PROJECT_B["project_id"]]
    assert captured[0]["evidence"] is expected[PROJECT_A["project_id"]]
    assert captured[1]["evidence"] is expected[PROJECT_B["project_id"]]


def test_existing_no_evidence_behavior_remains_empty_at_writer_boundary(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "0")
    captured = []
    prepare_pipeline_to_evidence_boundary(monkeypatch, [PROJECT_A], captured)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda _project: [])
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_staged(api_server.TailorBody())
    assert captured[0]["evidence"] == []


def test_resume_evidence_count_and_prompt_context_budgets_remain_bounded():
    values = adapted(PROJECT_A, api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    compacted = api_server.compact_github_evidence_for_prompt(values)
    assert len(values) == api_server.RESUME_PROJECT_EVIDENCE_LIMIT
    assert len(compacted) == api_server.RESUME_PROJECT_EVIDENCE_LIMIT
    assert len(api_server.serialize_resume_evidence_for_budget(compacted)) <= api_server.MAX_PROMPT_EVIDENCE_CHARS
    assert all("vector" in item["search_sources"] for item in compacted)
    assert [item["chunk_id"] for item in values] == [
        f"chk_workagent_{index}" for index in range(api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    ]


def test_resume_integration_change_has_no_prompt_writer_budget_latex_pdf_ats_or_frontend_edits():
    source = Path(api_server.__file__).read_text(encoding="utf-8")
    wrapper_start = source.index("def retrieve_evidence_for_project_for_resume")
    wrapper_end = source.index("\n\n", wrapper_start)
    wrapper = source[wrapper_start:wrapper_end]
    assert "retrieve_evidence_for_project_v2" in wrapper
    assert "retrieve_evidence_for_project(project)" in wrapper
    assert "safe_model_call" not in wrapper and "write_resume" not in wrapper
