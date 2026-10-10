"""Deterministic production routing and immutable Story handoff regressions."""

from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend import resume_story_ranking_integration as integration
from backend import hiring_context_ranking_service as ranking_service
from backend.engineering_story_memory_service import (
    AuthoritativeStoryMemoryLoadResult, EngineeringStoryMemoryReadiness,
    StoryMemoryArtifactStatus as Artifact, StoryMemoryReadinessState as Ready,
    build_authoritative_engineering_story_memory,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_ranking_service import build_hiring_context_ranking_state
from backend.hiring_context_web_enrichment import (
    HiringContextEnrichment, HiringContextEnrichmentStatus as WebStatus,
    OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG,
)
from backend.hiring_context_web_ranking import (
    OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG, WebAwareProjectStoryRankingState,
)
from backend.project_repository_identity import build_project_repository_identity_authority
from tests.test_engineering_story_memory_service import _fact, _snapshot
from tests.test_hiring_context_cross_context_ranking import _candidate_views
from tests.test_hiring_context_web_ranking import _official_ref, _signal


FLAG = integration.STORY_AWARE_RESUME_RANKING_FLAG
ON = {FLAG: "1"}
CONTEXT = {"company": "The Coalition", "job_title": "Software Engineer"}
Mode = integration.ResumeRankingMode


@pytest.fixture
def ready_memory(monkeypatch):
    memory = build_authoritative_engineering_story_memory(_snapshot(_fact("one"))).memory
    loaded = AuthoritativeStoryMemoryLoadResult(Artifact.READY, memory)
    readiness = EngineeringStoryMemoryReadiness(Ready.READY, logical_fingerprint=memory.logical_fingerprint)
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory", Mock(return_value=loaded))
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness", Mock(return_value=readiness))
    monkeypatch.setattr(integration, "load_project_repository_identity_authority", lambda: None)
    return memory, readiness


def prepare(**kwargs):
    return integration.prepare_resume_ranking(
        normalized_job_context=CONTEXT, project_cards=[{"project_id": "workagent"}],
        project_limit=3, **kwargs,
    )


@pytest.mark.parametrize("value", (None, "", "0", "false", "off", "invalid", "2", True))
def test_off_has_no_story_web_authority_or_ranking_calls(monkeypatch, value):
    for name in ("load_authoritative_engineering_story_memory",
                 "load_project_evidence_memory",
                 "inspect_authoritative_engineering_story_memory_readiness",
                 "load_project_repository_identity_authority", "build_hiring_context_profile",
                 "build_configured_hiring_context_ranking_state"):
        monkeypatch.setattr(integration, name, Mock(side_effect=AssertionError(name)))
    result = prepare(environ={} if value is None else {FLAG: value})
    assert result.mode is Mode.LEGACY_FLAG_OFF
    assert result.plan is None


@pytest.mark.parametrize("value", ("1", "true", "yes", "on", " TRUE "))
def test_valid_flag_values(value):
    assert integration.story_aware_resume_ranking_enabled({FLAG: value})


def test_ready_real_memory_is_read_only_and_generation_bound(ready_memory):
    memory, _ = ready_memory
    before = memory.to_dict()
    result = prepare(environ=ON)
    assert result.mode is Mode.STORY_AWARE_OFFLINE
    assert result.plan.state.hiring_context_fingerprint == result.plan.hiring_context.fingerprint
    story = result.plan.projects[0].stories[0]
    assert story.canonical_story_id == memory.histories[0].canonical_story_id
    assert story.current_revision_id == memory.histories[0].current_revision.revision_id
    assert memory.to_dict() == before


def test_missing_memory_allows_explicit_legacy_fallback(monkeypatch):
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                        lambda _: SimpleNamespace(status=Artifact.MISSING))
    assert prepare(environ=ON).mode is Mode.LEGACY_STORY_MEMORY_UNAVAILABLE


@pytest.mark.parametrize("status", (Artifact.INVALID, Artifact.INTEGRITY_MISMATCH, Artifact.UNSUPPORTED_VERSION))
def test_corrupt_memory_is_never_a_legacy_fallback(monkeypatch, status):
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                        lambda _: SimpleNamespace(status=status))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


@pytest.mark.parametrize("state", (Ready.INVALID, Ready.CONFLICTED, Ready.STALE_OR_REVALIDATION_REQUIRED, Ready.MISSING))
def test_nonready_integrity_states_fail_closed(ready_memory, monkeypatch, state):
    _, readiness = ready_memory
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: replace(readiness, state=state))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


def test_upstream_temporarily_not_ready_has_bounded_fallback(ready_memory, monkeypatch):
    _, readiness = ready_memory
    monkeypatch.setattr(integration, "load_project_evidence_memory",
                        lambda: SimpleNamespace(status="missing"))
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: replace(readiness, state=Ready.STALE_OR_REVALIDATION_REQUIRED,
                                            error_codes=("upstream_not_ready",)))
    assert prepare(environ=ON).mode is Mode.LEGACY_STORY_MEMORY_UNAVAILABLE


@pytest.mark.parametrize("status", ("invalid", "hash_mismatch", "unsupported_version", "error", "ready"))
def test_upstream_not_ready_cannot_hide_corruption_or_generation_race(ready_memory, monkeypatch, status):
    _, readiness = ready_memory
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: replace(readiness, state=Ready.STALE_OR_REVALIDATION_REQUIRED,
                                            error_codes=("upstream_not_ready",)))
    monkeypatch.setattr(integration, "load_project_evidence_memory",
                        lambda: SimpleNamespace(status=status))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


@pytest.mark.parametrize("artifact", ("missing", "malformed", "hash_mismatch"))
def test_real_upstream_readiness_distinguishes_absence_from_corruption(ready_memory, monkeypatch, tmp_path, artifact):
    from backend import engineering_story_memory_service as memory_service
    from backend.project_evidence_memory import load_project_evidence_memory

    memory, _ = ready_memory
    path = tmp_path / "upstream.json"
    if artifact == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    elif artifact == "hash_mismatch":
        payload = _snapshot(_fact("one")).to_dict()
        payload["content_hash"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes() if path.exists() else None
    load_upstream = lambda *_: load_project_evidence_memory(path)
    monkeypatch.setattr(memory_service, "load_authoritative_engineering_story_memory",
                        lambda *_: AuthoritativeStoryMemoryLoadResult(Artifact.READY, memory))
    monkeypatch.setattr(memory_service, "load_project_evidence_memory", load_upstream)
    monkeypatch.setattr(integration, "load_project_evidence_memory", load_upstream)
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        memory_service.inspect_authoritative_engineering_story_memory_readiness)
    if artifact == "missing":
        assert prepare(environ=ON).mode is Mode.LEGACY_STORY_MEMORY_UNAVAILABLE
    else:
        assert load_upstream().status == ("invalid" if artifact == "malformed" else "hash_mismatch")
        with pytest.raises(integration.ResumeRankingIntegrityError):
            prepare(environ=ON)
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("field", ("stale_story_count", "requires_revalidation_count", "conflicted_story_count"))
def test_upstream_unavailable_cannot_hide_lifecycle_failure(ready_memory, monkeypatch, field):
    _, readiness = ready_memory
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: replace(readiness, state=Ready.STALE_OR_REVALIDATION_REQUIRED,
                                            error_codes=("upstream_not_ready",), **{field: 1}))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


def test_readiness_generation_race_fails_closed(ready_memory, monkeypatch):
    _, readiness = ready_memory
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: replace(readiness, logical_fingerprint="f" * 64))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


def test_no_active_stories_has_distinct_fallback(ready_memory, monkeypatch):
    monkeypatch.setattr(integration, "get_active_engineering_stories_for_project", lambda *_: ())
    assert prepare(environ=ON).mode is Mode.LEGACY_NO_ACTIVE_STORIES


def test_empty_valid_memory_has_distinct_fallback(ready_memory, monkeypatch):
    from backend.engineering_story_matching import EngineeringStoryIdentityMap
    memory, _ = ready_memory
    empty = replace(memory, histories=(), identity_map=EngineeringStoryIdentityMap(), logical_fingerprint="")
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                        lambda _: AuthoritativeStoryMemoryLoadResult(Artifact.EMPTY, empty))
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: EngineeringStoryMemoryReadiness(Ready.READY, logical_fingerprint=empty.logical_fingerprint))
    assert prepare(environ=ON).mode is Mode.LEGACY_NO_ACTIVE_STORIES


@pytest.mark.parametrize("cards", ([{"project_name": "workagent"}], [{"project_id": "other"}],
                                  [{"project_id": "workagent"}, {"project_id": "workagent"}]))
def test_project_identity_mismatch_is_not_name_matched(ready_memory, cards):
    with pytest.raises(integration.ResumeRankingIntegrityError):
        integration.prepare_resume_ranking(normalized_job_context=CONTEXT, project_cards=cards,
                                           project_limit=3, environ=ON)


@pytest.mark.parametrize("repository", ("owner/other", "malformed", ""))
def test_repository_claim_must_match_saved_authority(ready_memory, monkeypatch, repository):
    authority = build_project_repository_identity_authority(
        project_memory={"projects": [{"project_id": "other", "repo": "owner/other"}]})
    monkeypatch.setattr(integration, "load_project_repository_identity_authority", lambda: authority)
    cards = [{"project_id": "workagent", "identity": {"repository": repository or "owner/unknown"}}]
    with pytest.raises(integration.ResumeRankingIntegrityError):
        integration.prepare_resume_ranking(normalized_job_context=CONTEXT, project_cards=cards,
                                           project_limit=3, environ=ON)


@pytest.mark.parametrize("status", (WebStatus.UNAVAILABLE, WebStatus.NO_OFFICIAL_SOURCE, WebStatus.CONFLICTED))
def test_web_failures_use_offline_story_ranking(ready_memory, monkeypatch, status):
    offline = prepare(environ=ON)
    def enrichment(profile, **_):
        return HiringContextEnrichment(offline_profile_id=profile.profile_id,
                                      offline_profile_fingerprint=profile.fingerprint, status=status,
                                      conflicts=("domain_conflict",) if status is WebStatus.CONFLICTED else ())
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", enrichment)
    result = prepare(environ={**ON, OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG: "1",
                              OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG: "1"})
    assert result.mode is Mode.STORY_AWARE_OFFLINE
    assert result.plan.state == offline.plan.state


@pytest.mark.parametrize("enrich,influence", ((False, False), (False, True), (True, False)))
def test_independent_web_flags_preserve_offline_ranking(ready_memory, monkeypatch, enrich, influence):
    baseline = prepare(environ=ON)
    acquirer = Mock(side_effect=lambda profile, **_: HiringContextEnrichment(
        offline_profile_id=profile.profile_id, offline_profile_fingerprint=profile.fingerprint,
        status=WebStatus.UNAVAILABLE))
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", acquirer)
    result = prepare(environ={**ON, OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG: str(int(enrich)),
                              OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG: str(int(influence))})
    assert result.plan.state == baseline.plan.state
    assert acquirer.call_count == int(enrich)


def matrix_plan(context, limit=3, views=None, enrichment=None):
    views = _candidate_views() if views is None else views
    profile = build_hiring_context_profile(
        company=context.get("company"), team=context.get("team"), role_title=context.get("job_title"),
        normalized_job_context=context,
    )
    grouped = {pid: tuple(view for view in views if view.project_id == pid) for pid in sorted({v.project_id for v in views})}
    state = build_hiring_context_ranking_state(hiring_context=profile, views_by_project=grouped,
                                              enrichment=enrichment, web_influence=enrichment is not None)
    return integration.ResumeRankingPlan(state, profile, limit)


def test_plan_keeps_full_state_and_assigns_counts_only_after_portfolio_order():
    plan = matrix_plan(CONTEXT)
    cards = [{"project_id": p.project_id, "project_name": p.project_id} for p in reversed(plan.projects)]
    allocator = Mock(side_effect=lambda position, count: 6 - position)
    selected, ranking = integration.production_selection(plan, cards, allocator)
    assert [c["project_id"] for c in selected] == [p.project_id for p in plan.projects[:3]]
    assert [call.args for call in allocator.call_args_list] == [(1, 3), (2, 3), (3, 3)]
    assert len(plan.projects) == 5
    assert plan.state.story_count == 25
    for entry, project in zip(ranking["selected_projects"], plan.projects):
        assert entry["story_ranking"] == project.story_handoff()
        assert len(project.stories) == 5
        assert len(project.top_contributing_story_ids) <= 3
        assert not {"total_score", "relevance_score", "space_score"} & entry.keys()


def test_unranked_cards_are_explicitly_omitted_not_available_for_name_reuse():
    plan = matrix_plan(CONTEXT)
    cards = [{"project_id": p.project_id, "project_name": p.project_id} for p in plan.projects]
    cards.append({"project_id": "unranked", "project_name": plan.projects[0].project_id})
    selected, ranking = integration.production_selection(plan, cards, lambda rank, total: 1)
    assert "unranked" not in {card["project_id"] for card in selected}
    assert ranking["omitted_projects"][-1] == {
        **cards[-1], "reason_omitted": "No active ranked Stories"}
    assert ranking == integration.production_selection(plan, list(reversed(cards)), lambda rank, total: 1)[1]


@pytest.mark.parametrize("context", (
    {"job_title": "Backend Engineer", "required_qualifications": ["backend API distributed systems"]},
    CONTEXT,
    {"job_title": "Analytics Engineer", "required_qualifications": ["analytics ETL data pipeline"]},
))
def test_cross_context_handoff_is_deterministic_and_preserves_truth(context):
    views = _candidate_views()
    before = tuple(view.to_dict() for view in views)
    first = matrix_plan(context, views=views)
    second = matrix_plan(context, views=tuple(reversed(views)))
    assert first == second
    cards = [{"project_id": p.project_id} for p in first.projects]
    selected, _ = integration.production_selection(first, cards, lambda rank, total: 1)
    assert [c["project_id"] for c in selected] == [p.project_id for p in first.state.portfolio.ranked_projects[:3]]
    assert tuple(view.to_dict() for view in views) == before


def test_mixed_context_and_tampered_state_are_rejected():
    plan = matrix_plan(CONTEXT)
    other = build_hiring_context_profile(company="Other", team=None, role_title="Analyst", normalized_job_context={})
    with pytest.raises(integration.ResumeRankingIntegrityError):
        replace(plan, hiring_context=other)
    state = replace(plan.state)
    object.__setattr__(state, "state_id", "tampered")
    with pytest.raises(ValueError):
        replace(plan, state=state)


def test_web_handoff_uses_corrected_claim_safety_without_new_formula():
    from tests.test_project_story_ranking_refresh import _view
    from backend.engineering_story_models import SufficiencyLevel
    views = tuple(_view(story_key=key, revision_key=key, project_id=key,
                        text="Designed gameplay game development real-time system architecture",
                        claim_level=level) for key, level in (("safe", SufficiencyLevel.HIGH), ("risky", SufficiencyLevel.LOW)))
    base = matrix_plan(CONTEXT, views=views)
    ref = _official_ref()
    enrichment = HiringContextEnrichment(
        offline_profile_id=base.hiring_context.profile_id,
        offline_profile_fingerprint=base.hiring_context.fingerprint,
        status=WebStatus.READY, official_source_refs=(ref,), signals=(_signal(ref),),
    )
    plan = matrix_plan(CONTEXT, views=views, enrichment=enrichment)
    assert isinstance(plan.state, WebAwareProjectStoryRankingState)
    assert plan.projects[0].project_id == "safe"
    for snapshot in plan.state.project_snapshots:
        story = snapshot.story_relevance[0]
        assert story.total_relevance_score == round(story.adjusted_raw_relevance_score * story.components.evidence_claim_safety, 6)


def test_integration_has_no_retrieval_reconstruction_writes_or_new_budget_policy():
    source = inspect.getsource(integration)
    for forbidden in ("import chromadb", ".query(", "persist_", "write_", "build_authoritative_engineering_story_memory",
                      "rank_projects_for_resume", "retrieve_evidence", "space_score"):
        assert forbidden not in source
    assert "preference" not in inspect.signature(integration.prepare_resume_ranking).parameters


def test_prepare_material_web_then_unavailable_stays_in_story_mode(ready_memory, monkeypatch):
    memory = build_authoritative_engineering_story_memory(_snapshot(
        _fact("game", mechanism="Designed gameplay game development real-time system architecture")
    )).memory
    monkeypatch.setattr(integration, "load_authoritative_engineering_story_memory",
                        lambda _: AuthoritativeStoryMemoryLoadResult(Artifact.READY, memory))
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        lambda **_: EngineeringStoryMemoryReadiness(Ready.READY, logical_fingerprint=memory.logical_fingerprint))
    ref = _official_ref()
    def enrichment(profile, **_):
        return HiringContextEnrichment(
            offline_profile_id=profile.profile_id, offline_profile_fingerprint=profile.fingerprint,
            status=WebStatus.READY, official_source_refs=(ref,), signals=(_signal(ref),))
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", enrichment)
    flags = {**ON, OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG: "1", OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG: "1"}
    result = prepare(environ=flags)
    assert result.mode is Mode.STORY_AWARE_WEB
    assert isinstance(result.plan.state, WebAwareProjectStoryRankingState)
    offline = prepare(environ=ON)
    def unavailable(profile, **_):
        return HiringContextEnrichment(offline_profile_id=profile.profile_id,
                                      offline_profile_fingerprint=profile.fingerprint, status=WebStatus.UNAVAILABLE)
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", unavailable)
    fallback = prepare(environ=flags)
    assert fallback.mode is Mode.STORY_AWARE_OFFLINE
    assert fallback.plan.state == offline.plan.state


def test_wrong_ranked_context_fails_closed(ready_memory, monkeypatch):
    foreign = matrix_plan({"company": "Other", "job_title": "Analyst"})
    monkeypatch.setattr(integration, "build_configured_hiring_context_ranking_state", lambda **_: foreign.state)
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


def test_cross_project_story_reader_failure_is_not_fallback(ready_memory, monkeypatch):
    monkeypatch.setattr(integration, "get_active_engineering_stories_for_project", lambda *_: (_candidate_views()[0],))
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


def test_valid_saved_repository_authority_is_reused(ready_memory, monkeypatch):
    cards = [{"project_id": "workagent", "identity": {"github_url": "https://github.com/owner/workagent"}}]
    authority = build_project_repository_identity_authority(project_memory={"projects": cards})
    monkeypatch.setattr(integration, "load_project_repository_identity_authority", lambda: authority)
    result = integration.prepare_resume_ranking(normalized_job_context=CONTEXT,
                                                project_cards=cards, project_limit=3, environ=ON)
    assert result.mode is Mode.STORY_AWARE_OFFLINE


def test_conflicted_repository_authority_fails_closed(ready_memory, monkeypatch):
    authority = build_project_repository_identity_authority(project_memory={"projects": [
        {"project_id": "one", "repo": "owner/shared"}, {"project_id": "two", "repo": "owner/shared"}]})
    monkeypatch.setattr(integration, "load_project_repository_identity_authority", lambda: authority)
    with pytest.raises(integration.ResumeRankingIntegrityError):
        prepare(environ=ON)


@pytest.mark.parametrize("limit", (0, -1, True, "3"))
def test_invalid_downstream_project_limit_fails_closed(ready_memory, limit):
    with pytest.raises(integration.ResumeRankingIntegrityError):
        integration.prepare_resume_ranking(normalized_job_context=CONTEXT,
                                            project_cards=[{"project_id": "workagent"}], project_limit=limit, environ=ON)


def test_non_authoritative_preferences_and_review_flags_cannot_change_production(ready_memory):
    baseline = prepare(environ=ON)
    for enabled in ("0", "1"):
        result = prepare(environ={**ON, "USE_HIRING_CONTEXT_RANKING_REVIEW": enabled,
                                  "USE_RESUME_EMPHASIS_PREFERENCES": enabled})
        assert result == baseline
