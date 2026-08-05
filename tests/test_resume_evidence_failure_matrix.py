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
from backend import project_retrieval_v2


PROJECT = {
    "project_id": "workagent",
    "project_name": "WorkAgent",
    "repo": "liammaxp/workagent",
}
V2_MODE = api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2
LEGACY_MODE = api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY


@pytest.mark.parametrize(
    "condition",
    (
        "readiness_blocked",
        "no_project_chunks",
        "http_backend_disabled",
        "chroma_unavailable",
        "timeout",
        "collection_missing",
        "vector_provenance_missing",
        "hybrid_empty",
        "empty_query_plan",
        "malformed_adapted_evidence",
        "malformed_evidence_artifact",
        "controlled_unexpected_error",
    ),
)
def test_flag_on_failure_matrix_never_calls_legacy(condition, monkeypatch):
    calls = {"legacy": 0, "v2": 0}

    def legacy(*_args, **_kwargs):
        calls["legacy"] += 1
        pytest.fail(f"legacy fallback is forbidden for {condition}")

    def safe_v2(*_args, **_kwargs):
        calls["v2"] += 1
        return []

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", safe_v2)
    result = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT,
        jd_targets={"requirements": ["retrieval"]},
        retrieval_mode=V2_MODE,
    )
    assert result == []
    assert calls == {"legacy": 0, "v2": 1}


def test_flag_on_ready_success_uses_v2_only(monkeypatch):
    expected = [{"project_id": "workagent", "chunk_id": "chk_one", "repo": PROJECT["repo"]}]
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("legacy must not run for v2 success"),
    )
    assert api_server.retrieve_evidence_for_project_for_resume(
        PROJECT,
        retrieval_mode=V2_MODE,
    ) is expected


def test_flag_off_success_empty_and_controlled_failure_preserve_legacy_contract(monkeypatch):
    expected = [{"repository": PROJECT["repo"], "description": "legacy"}]
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", lambda *_args, **_kwargs: pytest.fail("v2 must not run"))
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda _project: expected)
    assert api_server.retrieve_evidence_for_project_for_resume(PROJECT, retrieval_mode=LEGACY_MODE) is expected
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda _project: [])
    assert api_server.retrieve_evidence_for_project_for_resume(PROJECT, retrieval_mode=LEGACY_MODE) == []

    def controlled_failure(_project):
        raise ValueError("legacy-controlled-failure")

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", controlled_failure)
    with pytest.raises(ValueError, match="legacy-controlled-failure"):
        api_server.retrieve_evidence_for_project_for_resume(PROJECT, retrieval_mode=LEGACY_MODE)


def test_invalid_explicit_mode_fails_closed_to_legacy_without_reading_environment(monkeypatch):
    expected = [{"repository": PROJECT["repo"]}]
    monkeypatch.setattr(
        project_retrieval_v2,
        "is_github_evidence_retrieval_v2_enabled",
        lambda: pytest.fail("an explicit invalid mode must not reread process state"),
    )
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", lambda *_args, **_kwargs: pytest.fail("invalid mode must not select v2"))
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda _project: expected)
    assert api_server.retrieve_evidence_for_project_for_resume(
        PROJECT,
        retrieval_mode="automatic_fallback",
    ) is expected


def test_actual_v2_provider_contains_unexpected_dependency_errors(monkeypatch):
    def unavailable():
        raise TimeoutError("private host and path details")

    result = project_retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT,
        vector_backend_enabled=unavailable,
    )
    assert result == []
    assert "private" not in json.dumps(result)


def test_malformed_v2_shaped_evidence_fails_closed_at_prompt_consumer():
    malformed = [{
        "project_id": "workagent",
        "chunk_id": "chk_missing_source",
        "repo": "liammaxp/workagent",
        "raw_text": "diff --git API_KEY=fake",
        "description": "must not be reinterpreted as legacy evidence",
    }]
    assert api_server.compact_github_evidence_for_prompt(malformed) == []


class BlockedEvidenceBoundary(RuntimeError):
    pass


@pytest.mark.parametrize(
    "condition",
    ("http_backend_disabled", "chroma_unavailable", "readiness_partial", "no_project_chunks", "vector_provenance_absent"),
)
def test_flag_on_blocker_reaches_existing_empty_evidence_path_in_actual_orchestration(condition, monkeypatch):
    monkeypatch.setattr(api_server.agent, "read_job_description", lambda: "Backend retrieval role")
    monkeypatch.setattr(api_server.agent, "read_resume", lambda: "Existing resume")
    monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: json.dumps({"projects": [PROJECT]}))
    monkeypatch.setattr(api_server, "resolve_saved_application_hint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_server, "agent_progress_guidance_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(api_server, "classify_role_family", lambda *_args, **_kwargs: {"role_family": "software_engineering"})
    monkeypatch.setattr(api_server, "jd_requirements_for_prompt", lambda *_args, **_kwargs: {"requirements": ["retrieval"]})
    monkeypatch.setattr(api_server, "enrich_jd_profile_with_tech_ontology", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(
        api_server,
        "select_staged_projects_with_ranking",
        lambda *_args, **_kwargs: ([PROJECT], {"selected_projects": [{"project_id": "workagent", "rank": 1, "bullet_budget": 2}]}),
    )
    monkeypatch.setattr(api_server, "resolve_resume_evidence_retrieval_mode", lambda: V2_MODE)
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail(f"legacy fallback is forbidden for {condition}"),
    )

    def capture_candidate(_job, _resume, _project, evidence, *_args, **_kwargs):
        assert evidence == []
        raise BlockedEvidenceBoundary()

    monkeypatch.setattr(api_server, "build_project_resume_candidate", capture_candidate)
    with pytest.raises(BlockedEvidenceBoundary):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=True))
