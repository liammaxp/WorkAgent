"""Candidate reuse consumes the complete accepted ranking generation."""

from dataclasses import fields, replace
import hashlib

import pytest

from backend import project_story_ranking_state as state_contract
from backend import hiring_context_ranking_service as ranking_service
from backend import hiring_context_web_ranking as web_contract
from backend.hiring_context_web_enrichment import HiringContextEnrichment, HiringContextEnrichmentStatus
from tests.test_resume_retrieval_v2_evidence_integration import api_server
from tests.test_resume_story_ranking_integration import CONTEXT, _candidate_views, integration, matrix_plan
from tests.test_hiring_context_web_ranking import _enrichment, _official_ref, _signal


class CacheBoundary(RuntimeError):
    pass


def cache_hash(monkeypatch, plan):
    cards = [{"project_id": project.project_id, "project_name": project.project_id}
             for project in plan.projects]
    projects, ranking = integration.production_selection(plan, cards, lambda *_: 4)
    assert plan.ranking_generation_id == plan.state.state_id
    assert ranking["ranking_generation_id"] == plan.ranking_generation_id
    hashes = []

    def capture(*args):
        hashes.append(args[3])
        raise CacheBoundary

    monkeypatch.setattr(api_server, "get_completed_resume_candidate_checkpoint", capture)
    with pytest.raises(CacheBoundary):
        api_server.build_project_resume_candidate(
            "same job", "same resume", projects[0], [], "en",
            project_ranking=ranking, project_rank_entry=ranking["selected_projects"][0])
    assert len(hashes) == 1
    return hashes[0], ranking


def revise(views, story_id):
    return tuple(replace(view, current_revision_id="engineering_story_revision_" + "d" * 24,
                         provenance_fingerprint=hashlib.sha256(b"accepted new generation").hexdigest())
                 if view.canonical_story_id == story_id else view for view in views)


@pytest.mark.parametrize("story_index", [0, 3, 4])
def test_selected_and_hidden_story_revisions_change_actual_candidate_cache_hash(monkeypatch, story_index):
    views = _candidate_views()
    before = matrix_plan(CONTEXT, views=views)
    selected = before.projects[0]
    assert len(selected.stories) == 5
    changed = selected.stories[story_index]
    after = matrix_plan(CONTEXT, views=revise(views, changed.canonical_story_id))
    assert before.state.state_id != after.state.state_id
    assert [value.project_id for value in before.projects] == [value.project_id for value in after.projects]
    if story_index >= 3:
        assert not changed.contribution_member
        assert before.state.portfolio == after.state.portfolio
        assert before.projects[0].top_contributing_story_ids == after.projects[0].top_contributing_story_ids
    assert cache_hash(monkeypatch, before)[0] != cache_hash(monkeypatch, after)[0]


def test_unselected_project_revision_invalidates_selected_candidates(monkeypatch):
    views = _candidate_views()
    before = matrix_plan(CONTEXT, views=views)
    assert len(before.projects) > before.project_limit
    omitted = before.projects[3]
    after = matrix_plan(CONTEXT, views=revise(views, omitted.stories[-1].canonical_story_id))
    assert before.projects[:3] == after.projects[:3]
    assert before.state.state_id != after.state.state_id
    first_hash, first_ranking = cache_hash(monkeypatch, before)
    second_hash, second_ranking = cache_hash(monkeypatch, after)
    assert first_ranking["selected_projects"] == second_ranking["selected_projects"]
    assert first_hash != second_hash


def test_equivalent_reconstruction_and_source_order_keep_generation_and_cache_hash(monkeypatch):
    views = _candidate_views()
    first = matrix_plan(CONTEXT, views=views)
    second = matrix_plan(CONTEXT, views=tuple(reversed(views)))
    assert first.state == second.state
    assert first.ranking_generation_id == second.ranking_generation_id
    assert cache_hash(monkeypatch, first)[0] == cache_hash(monkeypatch, second)[0]


def test_material_hiring_context_and_portfolio_order_change_cache_hash(monkeypatch):
    first = matrix_plan({"job_title": "Backend Engineer", "required_qualifications": ["backend API distributed systems"]})
    second = matrix_plan({"job_title": "Game Developer", "required_qualifications": ["game development realtime gameplay Unity"]})
    assert first.hiring_context.fingerprint != second.hiring_context.fingerprint
    assert [item.project_id for item in first.projects] != [item.project_id for item in second.projects]
    assert cache_hash(monkeypatch, first)[0] != cache_hash(monkeypatch, second)[0]


def test_accepted_ranking_policy_generation_changes_cache_without_new_downstream_policy(monkeypatch):
    first = matrix_plan(CONTEXT)
    first_hash = cache_hash(monkeypatch, first)[0]
    next_policy = "project_story_ranking_state.complete_generation.test_policy"
    monkeypatch.setattr(state_contract, "PROJECT_STORY_RANKING_STATE_POLICY_ID", next_policy)
    second_state = replace(first.state, state_policy_id=next_policy, state_id="", state_fingerprint="")
    second = integration.ResumeRankingPlan(second_state, first.hiring_context, first.project_limit)
    assert first.projects == second.projects
    assert first.ranking_generation_id != second.ranking_generation_id
    assert first_hash != cache_hash(monkeypatch, second)[0]


def test_material_web_generation_and_semantic_signal_change_invalidate_cache(monkeypatch):
    offline = matrix_plan(CONTEXT)
    ref = _official_ref()
    first_enrichment = _enrichment(offline.hiring_context, refs=(ref,), signals=(_signal(ref),))
    second_enrichment = _enrichment(offline.hiring_context, refs=(ref,),
                                    signals=(_signal(ref), _signal(ref, "Real-time systems")))
    first = matrix_plan(CONTEXT, enrichment=first_enrichment)
    second = matrix_plan(CONTEXT, enrichment=second_enrichment)
    assert first.state.state_id != offline.state.state_id
    assert second.state.state_id != first.state.state_id
    assert len({cache_hash(monkeypatch, plan)[0] for plan in (offline, first, second)}) == 3


@pytest.mark.parametrize("status", [HiringContextEnrichmentStatus.UNAVAILABLE,
                                    HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE,
                                    HiringContextEnrichmentStatus.CONFLICTED])
def test_optional_web_failure_keeps_offline_generation_and_cache_hash(monkeypatch, status):
    offline = matrix_plan(CONTEXT)
    enrichment = _enrichment(offline.hiring_context, status=status,
                             conflicts=("domain_conflict",) if status is HiringContextEnrichmentStatus.CONFLICTED else ())
    optional = matrix_plan(CONTEXT, enrichment=enrichment)
    assert offline.state == optional.state
    assert cache_hash(monkeypatch, offline)[0] == cache_hash(monkeypatch, optional)[0]


def test_fetch_time_is_not_a_semantic_ranking_or_cache_input(monkeypatch):
    offline = matrix_plan(CONTEXT)
    ref = _official_ref()
    acquisitions = [("2026-10-01T00:00:00Z", _enrichment(offline.hiring_context, refs=(ref,), signals=(_signal(ref),))),
                    ("2026-10-02T00:00:00Z", _enrichment(offline.hiring_context, refs=(ref,), signals=(_signal(ref),)))]
    assert not {"fetched_at", "retrieved_at", "timestamp"} & {field.name for field in fields(HiringContextEnrichment)}
    plans = [matrix_plan(CONTEXT, enrichment=accepted) for _, accepted in acquisitions]
    assert plans[0].state == plans[1].state
    assert cache_hash(monkeypatch, plans[0])[0] == cache_hash(monkeypatch, plans[1])[0]


@pytest.mark.parametrize("enrich,influence", [(False, True), (True, False), (False, False)])
def test_configured_disabled_web_flags_preserve_actual_cache_identity(monkeypatch, enrich, influence):
    from unittest.mock import Mock
    offline = matrix_plan(CONTEXT)
    views = _candidate_views()
    grouped = {project: tuple(view for view in views if view.project_id == project)
               for project in {view.project_id for view in views}}
    ref = _official_ref()
    accepted = _enrichment(offline.hiring_context, refs=(ref,), signals=(_signal(ref),))
    acquire = Mock(return_value=accepted)
    monkeypatch.setattr(ranking_service, "build_hiring_context_web_enrichment", acquire)
    state = ranking_service.build_configured_hiring_context_ranking_state(
        hiring_context=offline.hiring_context, views_by_project=grouped,
        environ={"USE_OFFICIAL_HIRING_CONTEXT_ENRICHMENT": str(int(enrich)),
                 "USE_OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE": str(int(influence))})
    configured = integration.ResumeRankingPlan(state, offline.hiring_context, offline.project_limit)
    assert acquire.call_count == int(enrich)
    assert configured.state == offline.state
    assert cache_hash(monkeypatch, configured)[0] == cache_hash(monkeypatch, offline)[0]


def test_accepted_web_policy_generation_propagates_to_actual_cache_identity(monkeypatch):
    from backend.project_story_ranking import aggregate_project_story_relevance
    from backend.project_portfolio_ranking import rank_project_portfolio
    from backend.project_story_ranking_refresh import ProjectStoryRelevanceSnapshot
    offline = matrix_plan(CONTEXT)
    ref = _official_ref()
    accepted = _enrichment(offline.hiring_context, refs=(ref,), signals=(_signal(ref),))
    first = matrix_plan(CONTEXT, enrichment=accepted)
    first_hash = cache_hash(monkeypatch, first)[0]
    policy = "hiring_context_web_ranking.bounded_context_adjustment.test_policy"
    monkeypatch.setattr(web_contract, "OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID", policy)
    stories = {}
    projects = {}
    for snapshot in first.state.project_snapshots:
        adjusted = []
        for story in snapshot.story_relevance:
            adjustment = replace(story.web_context_adjustment, policy_id=policy, adjustment_id="")
            adjusted.append(replace(story, web_context_adjustment=adjustment, relevance_id=""))
        stories[snapshot.project_id] = tuple(adjusted)
        projects[snapshot.project_id] = aggregate_project_story_relevance(
            project_id=snapshot.project_id, story_relevance=tuple(adjusted))
    portfolio = rank_project_portfolio(projects=tuple(projects.values()))
    snapshots = tuple(ProjectStoryRelevanceSnapshot(
        snapshot_policy_id=old.snapshot_policy_id, source_portfolio_id=portfolio.portfolio_id,
        project_relevance=projects[old.project_id], story_relevance=stories[old.project_id])
        for old in first.state.project_snapshots)
    inner = state_contract._build_project_story_ranking_state(
        hiring_context=first.hiring_context, portfolio=portfolio, project_snapshots=snapshots,
        allow_web_adjusted=True)
    state = web_contract.WebAwareProjectStoryRankingState(
        ranking_state=inner, enrichment_fingerprint=accepted.fingerprint,
        enrichment_status=accepted.status, web_influence_policy_id=policy)
    second = integration.ResumeRankingPlan(state, first.hiring_context, first.project_limit)
    assert [project.project_id for project in first.projects] == [project.project_id for project in second.projects]
    assert first.ranking_generation_id != second.ranking_generation_id
    assert first_hash != cache_hash(monkeypatch, second)[0]
