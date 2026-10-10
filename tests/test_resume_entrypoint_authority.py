"""Resume request routing owns ranking; optional GitHub retrieval owns evidence only."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.engineering_story_matching import EngineeringStoryIdentityMap
from tests.test_resume_story_ranking_production import (
    EvidenceBoundaryReached, api_server, forbidden, prepare_boundary,
)
from tests.test_resume_story_ranking_integration import (
    Artifact, AuthoritativeStoryMemoryLoadResult, CONTEXT, EngineeringStoryMemoryReadiness,
    FLAG, HiringContextEnrichment, OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG,
    OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG, Ready, WebStatus,
    _fact, _official_ref, _signal, _snapshot, build_authoritative_engineering_story_memory,
    integration, ranking_service, ready_memory,
)


def entrypoint_boundary(monkeypatch, github):
    cards = [{"project_id": "workagent", "project_name": "WorkAgent"}]
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv(OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG, raising=False)
    monkeypatch.delenv(OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG, raising=False)
    for name in ("run_agent_task", "rank_projects_for_resume", "retrieve_evidence_for_project"):
        monkeypatch.setattr(api_server, name, forbidden)
    for name in ("ask_agent", "read_github_context", "read_project_evidence_map"):
        monkeypatch.setattr(api_server.agent, name, forbidden)
    provider = Mock(return_value=[{"project_id": "workagent", "chunk_id": "safe"}])
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project_for_resume", provider if github else forbidden)
    if not github:
        monkeypatch.setattr(api_server, "resolve_resume_evidence_retrieval_mode", forbidden)
    prepare = Mock(wraps=integration.prepare_resume_ranking)
    monkeypatch.setattr(integration, "prepare_resume_ranking", prepare)
    captured = []

    def candidate(job, resume, project, evidence, language, guidance, ranking, entry, constraints):
        captured.append((project, evidence, ranking, entry))
        return {"project_id": project["project_id"], "final_bullets": [],
                **({"ranking_generation_id": ranking["ranking_generation_id"]}
                   if "ranking_generation_id" in ranking else {})}

    monkeypatch.setattr(api_server, "build_project_resume_candidate", candidate)
    return prepare, provider, captured


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("public_route", [False, True])
def test_healthy_request_consumes_real_authority_without_agent_bypass(monkeypatch, ready_memory, github, public_route):
    prepare, provider, captured = entrypoint_boundary(monkeypatch, github)
    invoke = api_server.tailor_resume if public_route else api_server.tailor_resume_task
    with pytest.raises(EvidenceBoundaryReached):
        invoke(api_server.TailorBody(use_github_context=github))
    prepare.assert_called_once_with(normalized_job_context=CONTEXT,
                                    project_cards=[{"project_id": "workagent", "project_name": "WorkAgent"}],
                                    project_limit=3)
    assert len(captured) == 1
    project, evidence, ranking, entry = captured[0]
    assert project["project_id"] == entry["project_id"] == "workagent"
    assert entry["story_ranking"]["current_revision_ids"] == [ready_memory[0].histories[0].current_revision.revision_id]
    assert ranking["ranking_generation_id"]
    assert evidence == (provider.return_value if github else [])
    assert provider.call_count == int(github)


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("fallback", ["missing", "empty", "upstream_unavailable"])
def test_allowed_fallback_uses_existing_staged_legacy_ranker_only(monkeypatch, ready_memory, github, fallback):
    prepare, provider, captured = entrypoint_boundary(monkeypatch, github)
    memory, readiness = ready_memory
    if fallback == "missing":
        monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                            lambda _: SimpleNamespace(status=Artifact.MISSING))
    elif fallback == "empty":
        empty = replace(memory, histories=(), identity_map=EngineeringStoryIdentityMap(), logical_fingerprint="")
        monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                            lambda _: AuthoritativeStoryMemoryLoadResult(Artifact.EMPTY, empty))
        monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                            lambda **_: EngineeringStoryMemoryReadiness(Ready.READY, logical_fingerprint=empty.logical_fingerprint))
    else:
        monkeypatch.setattr(integration, "load_project_evidence_memory", lambda: SimpleNamespace(status="missing"))
        monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                            lambda **_: replace(readiness, state=Ready.STALE_OR_REVALIDATION_REQUIRED,
                                                error_codes=("upstream_not_ready",)))
    ranking = {"selected_projects": [{"project_id": "workagent", "project_name": "WorkAgent",
                                       "rank": 1, "bullet_budget": 5}], "omitted_projects": []}
    legacy = Mock(return_value=ranking)
    monkeypatch.setattr(api_server, "rank_projects_for_resume", legacy)
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=github))
    assert prepare.call_count == legacy.call_count == len(captured) == 1
    assert "ranking_generation_id" not in captured[0][2]
    assert provider.call_count == int(github)


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("failure", ["stale", "conflict", "corrupt", "generation_mismatch", "project_conflict"])
def test_integrity_failure_cannot_escape_to_agent_or_cache(monkeypatch, ready_memory, github, failure):
    prepare, provider, captured = entrypoint_boundary(monkeypatch, github)
    _, readiness = ready_memory
    if failure == "corrupt":
        monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                            lambda _: SimpleNamespace(status=Artifact.INVALID))
    elif failure == "project_conflict":
        monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: '{"projects":[{"project_id":"other"}]}')
    else:
        changed = (replace(readiness, logical_fingerprint="f" * 64) if failure == "generation_mismatch" else
                   replace(readiness, state=Ready.CONFLICTED if failure == "conflict" else Ready.STALE_OR_REVALIDATION_REQUIRED))
        monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness", lambda **_: changed)
    monkeypatch.setattr(api_server, "get_completed_resume_candidate_checkpoint", forbidden)
    with pytest.raises(api_server.HTTPException) as error:
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=github))
    assert error.value.status_code == 409
    assert error.value.detail == "Project ranking needs revalidation before generation."
    assert prepare.call_count == 1
    assert captured == []
    provider.assert_not_called()


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("status", [WebStatus.READY, WebStatus.UNAVAILABLE, WebStatus.NO_OFFICIAL_SOURCE, WebStatus.CONFLICTED])
def test_optional_web_preserves_shared_authority_for_both_evidence_modes(monkeypatch, ready_memory, github, status):
    prepare, _, captured = entrypoint_boundary(monkeypatch, github)
    memory = build_authoritative_engineering_story_memory(_snapshot(
        _fact("game", mechanism="Designed gameplay game development real-time system architecture"))).memory
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                        lambda _: AuthoritativeStoryMemoryLoadResult(Artifact.READY, memory))
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: EngineeringStoryMemoryReadiness(Ready.READY, logical_fingerprint=memory.logical_fingerprint))
    offline = integration.prepare_resume_ranking(normalized_job_context=CONTEXT,
                                                project_cards=[{"project_id": "workagent"}], project_limit=3)
    prepare.reset_mock()
    ref = _official_ref()
    acquire = Mock(side_effect=lambda profile, **_: HiringContextEnrichment(
        offline_profile_id=profile.profile_id, offline_profile_fingerprint=profile.fingerprint,
        status=status, official_source_refs=(ref,) if status is WebStatus.READY else (),
        signals=(_signal(ref),) if status is WebStatus.READY else (),
        conflicts=("domain_conflict",) if status is WebStatus.CONFLICTED else ()))
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", acquire)
    monkeypatch.setenv(OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG, "1")
    monkeypatch.setenv(OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG, "1")
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=github))
    assert prepare.call_count == acquire.call_count == len(captured) == 1
    generation = captured[0][2]["ranking_generation_id"]
    assert (generation != offline.plan.ranking_generation_id) is (status is WebStatus.READY)


@pytest.mark.parametrize("flag", [None, "0"])
def test_off_github_on_preserves_exact_staged_dispatch(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.setattr(integration, "prepare_resume_ranking", forbidden)
    monkeypatch.setattr(api_server, "run_agent_task", forbidden)
    staged = Mock(return_value={"legacy": "staged"})
    monkeypatch.setattr(api_server, "tailor_resume_staged", staged)
    body = api_server.TailorBody(use_github_context=True)
    assert api_server.tailor_resume_task(body) is staged.return_value
    staged.assert_called_once_with(body)


@pytest.mark.parametrize("flag", [None, "0"])
def test_off_github_off_preserves_legacy_general_agent_boundary(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.setattr(integration, "prepare_resume_ranking", forbidden)
    monkeypatch.setattr(api_server, "tailor_resume_staged", forbidden)
    monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: "original project memory")
    monkeypatch.setattr(api_server.agent, "file_is_ready", lambda *_: False)
    monkeypatch.setattr(api_server, "resolve_saved_application_hint", lambda *_: {})
    legacy = Mock(side_effect=EvidenceBoundaryReached)
    monkeypatch.setattr(api_server, "run_agent_task", legacy)
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_task(api_server.TailorBody(use_github_context=False))
    legacy.assert_called_once()
    prompt = legacy.call_args.args[0]
    assert api_server.RESUME_TAILOR_PROMPT in prompt
    assert "original project memory" in prompt
    assert "GitHub evidence is not requested" in prompt
    # OFF intentionally retains general-agent tool capabilities, not a no-tool promise.
    assert {"read_project_memory", "read_project_evidence_map", "read_github_context"} <= api_server.agent.TOOL_FUNCTIONS.keys()
