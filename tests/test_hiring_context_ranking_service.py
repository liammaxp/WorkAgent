"""Shared ranking semantics and coherent authoritative-generation regressions."""

import ast
from dataclasses import replace
import inspect
import itertools
import json
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import engineering_story_memory_service as memory_service
from backend import hiring_context_ranking_review as review
from backend import hiring_context_ranking_service as service
from backend import resume_story_ranking_integration as production
from backend.engineering_story_memory_service import (
    AuthoritativeStoryMemoryLoadResult, EngineeringStoryMemoryReadiness,
    StoryMemoryArtifactStatus, StoryMemoryReadinessState,
    build_authoritative_engineering_story_memory,
)
from backend.hiring_context_ranking_review_api import create_hiring_context_ranking_review_router
from backend.hiring_context_web_enrichment import HiringContextEnrichment, HiringContextEnrichmentStatus
from backend.hiring_context_web_ranking import WebAwareProjectStoryRankingState
from tests.test_engineering_story_memory_service import _fact, _snapshot
from tests.test_hiring_context_web_ranking import _official_ref, _signal


CONTEXT = {"company": "The Coalition", "team": "Gameplay", "job_title": "Software Engineer"}
ENRICH = "USE_OFFICIAL_HIRING_CONTEXT_ENRICHMENT"
INFLUENCE = "USE_OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE"
ROLLOUT = "USE_STORY_AWARE_RESUME_RANKING"
REVIEW = "USE_HIRING_CONTEXT_RANKING_REVIEW"


@pytest.fixture
def coherent_memory(monkeypatch):
    memory = build_authoritative_engineering_story_memory(_snapshot(
        _fact("game", project_id="game", mechanism="Designed gameplay game development real-time system architecture"),
        _fact("backend", project_id="backend", mechanism="Implemented backend API request validation and reliable data flow"),
        _fact("recovery", project_id="backend", change_id="recovery", mechanism="Debugged reliability validation repair testing and recovery"),
        _fact("analytics", project_id="analytics", mechanism="Built analytics data pipelines and decision support"),
    )).memory
    loaded = AuthoritativeStoryMemoryLoadResult(StoryMemoryArtifactStatus.READY, memory)
    readiness = EngineeringStoryMemoryReadiness(
        StoryMemoryReadinessState.READY, logical_fingerprint=memory.logical_fingerprint)
    for consumer in (review, production):
        monkeypatch.setattr(consumer, "load_authoritative_engineering_story_memory", Mock(return_value=loaded))
        monkeypatch.setattr(consumer, "inspect_authoritative_engineering_story_memory_readiness", Mock(return_value=readiness))
    monkeypatch.setattr(production, "load_project_repository_identity_authority", lambda: None)
    return memory, readiness


def enrichment(profile, status=HiringContextEnrichmentStatus.READY, reverse=False, **_):
    refs = (_official_ref(), _official_ref(suffix="engineering"))
    signals = (_signal(refs[0]), _signal(refs[1], value="Real-time systems"))
    usable = status is HiringContextEnrichmentStatus.READY
    return HiringContextEnrichment(
        offline_profile_id=profile.profile_id, offline_profile_fingerprint=profile.fingerprint,
        status=status, official_source_refs=tuple(reversed(refs)) if reverse and usable else refs if usable else (),
        signals=tuple(reversed(signals)) if reverse and usable else signals if usable else (),
        conflicts=("domain_conflict",) if status is HiringContextEnrichmentStatus.CONFLICTED else (),
    )


def production_decision(memory, flags, context=CONTEXT):
    return production.prepare_resume_ranking(
        normalized_job_context=context,
        project_cards=[{"project_id": pid} for pid in sorted({h.project_id for h in memory.histories})],
        project_limit=3, environ={**flags, ROLLOUT: "1"},
    )


def preview(flags, context=CONTEXT):
    return review.build_hiring_context_ranking_review(
        company=context["company"], team=context.get("team"), role_title=context["job_title"],
        normalized_job_context=context, environ=flags,
    )


def capture_semantic_state(monkeypatch):
    states = []
    presenter = review.present_hiring_context_ranking_review
    def capture(**kwargs):
        states.append(kwargs["ranking_state"])
        return presenter(**kwargs)
    monkeypatch.setattr(review, "present_hiring_context_ranking_review", capture)
    return states


@pytest.mark.parametrize("enrich,influence", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("rollout", ("0", "1"))
def test_review_and_production_share_effective_flag_semantics(coherent_memory, monkeypatch, enrich, influence, rollout):
    memory, _ = coherent_memory
    before = memory.to_dict()
    acquirer = Mock(side_effect=enrichment)
    monkeypatch.setattr(service, "build_hiring_context_web_enrichment", acquirer)
    states = capture_semantic_state(monkeypatch)
    flags = {ENRICH: str(int(enrich)), INFLUENCE: str(int(influence)), ROLLOUT: rollout, REVIEW: "1"}
    result = preview(flags)
    decision = production_decision(memory, flags)
    assert result.status is review.HiringContextRankingReviewStatus.READY
    assert states == [decision.plan.state]
    assert isinstance(states[0], WebAwareProjectStoryRankingState) is (enrich and influence)
    assert acquirer.call_count == 2 * int(enrich)
    state = states[0]
    assert [p.project_id for p in result.projects] == [p.project_id for p in state.portfolio.ranked_projects]
    for project in result.projects:
        stories = state.snapshot_for_project(project.project_id).story_relevance
        assert [s.story_id for s in project.strongest_stories + project.additional_stories] == [s.canonical_story_id for s in stories]
    assert memory.to_dict() == before


@pytest.mark.parametrize("status", (
    HiringContextEnrichmentStatus.UNAVAILABLE, HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE,
    HiringContextEnrichmentStatus.CONFLICTED,
))
def test_optional_web_failure_has_exact_offline_parity(coherent_memory, monkeypatch, status):
    memory, _ = coherent_memory
    monkeypatch.setattr(service, "build_hiring_context_web_enrichment", lambda profile, **_: enrichment(profile, status))
    states = capture_semantic_state(monkeypatch)
    flags = {ENRICH: "1", INFLUENCE: "1"}
    assert preview(flags).status is review.HiringContextRankingReviewStatus.READY
    online = production_decision(memory, flags)
    offline = production_decision(memory, {})
    assert states == [online.plan.state]
    assert online.mode is production.ResumeRankingMode.STORY_AWARE_OFFLINE
    assert states[0] == offline.plan.state


def test_material_web_state_and_adjusted_identity_are_shared_and_order_independent(coherent_memory, monkeypatch):
    memory, _ = coherent_memory
    flags = {ENRICH: "1", INFLUENCE: "1"}
    states = capture_semantic_state(monkeypatch)
    offline = production_decision(memory, {}).plan.state
    for reverse in (False, True):
        monkeypatch.setattr(service, "build_hiring_context_web_enrichment",
                            lambda profile, **_: enrichment(profile, reverse=reverse))
        assert preview(flags).status is review.HiringContextRankingReviewStatus.READY
        actual = production_decision(memory, flags).plan.state
        assert states[-1] == actual
        assert states[-1].portfolio.portfolio_id == actual.portfolio.portfolio_id
        assert states[-1].state_id == actual.state_id
        assert [[s.relevance_id for s in p.story_relevance] for p in states[-1].project_snapshots] == [
            [s.relevance_id for s in p.story_relevance] for p in actual.project_snapshots]
    assert states[0] == states[1]
    assert isinstance(states[0], WebAwareProjectStoryRankingState)
    assert states[0].state_id != offline.state_id
    base = {s.canonical_story_id: s for p in offline.project_snapshots for s in p.story_relevance}
    for snapshot in states[0].project_snapshots:
        for story in snapshot.story_relevance:
            prior = base[story.canonical_story_id]
            assert story.semantic_features == prior.semantic_features
            assert story.claim_sufficiency == prior.claim_sufficiency
            assert story.story_sufficiency == prior.story_sufficiency
            assert story.components == prior.components


def test_corrected_preview_enriches_corrected_identity_without_persistence(coherent_memory, monkeypatch):
    memory, _ = coherent_memory
    acquirer = Mock(side_effect=lambda profile, **_: enrichment(profile, HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE))
    monkeypatch.setattr(service, "build_hiring_context_web_enrichment", acquirer)
    flags = {ENRICH: "1", INFLUENCE: "1"}
    saved = dict(CONTEXT)
    corrected = {**CONTEXT, "company": "Corrected Employer", "team": "Corrected Team", "job_title": "Analytics Engineer"}
    response = review.build_hiring_context_ranking_review(
        company=corrected["company"], team=corrected["team"], role_title=corrected["job_title"],
        normalized_job_context=saved, environ=flags,
    )
    production_decision(memory, flags)
    reviewed_profile, production_profile = [call.args[0] for call in acquirer.call_args_list]
    assert (reviewed_profile.company, reviewed_profile.team, reviewed_profile.role_title) == (
        corrected["company"], corrected["team"], corrected["job_title"])
    assert production_profile.company == CONTEXT["company"]
    assert production_profile.team == CONTEXT["team"]
    assert saved == CONTEXT
    assert response.corrections_persisted is False


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("web", (False, True))
def test_mid_read_generation_change_returns_no_partial_or_stale_state_and_next_request_recovers(
    coherent_memory, monkeypatch, empty, web,
):
    memory, readiness = coherent_memory
    if empty:
        memory = build_authoritative_engineering_story_memory(_snapshot()).memory
        readiness = replace(readiness, logical_fingerprint=memory.logical_fingerprint)
    loaded = AuthoritativeStoryMemoryLoadResult(
        StoryMemoryArtifactStatus.EMPTY if empty else StoryMemoryArtifactStatus.READY, memory)
    events = []
    def load(_):
        events.append("load-A")
        return loaded
    readings = iter((replace(readiness, logical_fingerprint="e" * 64), readiness))
    def inspect_generation(**kwargs):
        assert kwargs["compare_upstream"] is True
        events.append("readiness")
        return next(readings)
    monkeypatch.setattr(review, "load_authoritative_engineering_story_memory", load)
    monkeypatch.setattr(review, "inspect_authoritative_engineering_story_memory_readiness", inspect_generation)
    acquirer = Mock(side_effect=enrichment)
    monkeypatch.setattr(service, "build_hiring_context_web_enrichment", acquirer)
    ranker = Mock(wraps=service.build_configured_hiring_context_ranking_state)
    monkeypatch.setattr(review, "build_configured_hiring_context_ranking_state", ranker)
    flags = {ENRICH: str(int(web)), INFLUENCE: str(int(web))}
    before = memory.to_dict()
    rejected = preview(flags)
    assert rejected.status is review.HiringContextRankingReviewStatus.ERROR
    assert rejected.projects == () and rejected.preference_binding is None
    ranker.assert_not_called()
    acquirer.assert_not_called()
    text = json.dumps(rejected.to_dict())
    for private in (memory.logical_fingerprint, "e" * 64, "fingerprint", "policy", "artifact", "web_aware"):
        assert private not in text
    fresh = preview(flags)
    assert fresh.status is (review.HiringContextRankingReviewStatus.EMPTY if empty else review.HiringContextRankingReviewStatus.READY)
    assert ranker.call_count == int(not empty)
    assert events == ["load-A", "readiness", "load-A", "readiness"]
    assert memory.to_dict() == before


def test_http_mismatch_response_is_bounded_and_can_recover(coherent_memory, monkeypatch):
    _, readiness = coherent_memory
    current = Mock(side_effect=(replace(readiness, logical_fingerprint="e" * 64), readiness))
    monkeypatch.setattr(review, "inspect_authoritative_engineering_story_memory_readiness", current)
    for flag in (ENRICH, INFLUENCE):
        monkeypatch.delenv(flag, raising=False)
    app = FastAPI()
    app.include_router(create_hiring_context_ranking_review_router(
        read_job_description=lambda: "Saved software engineering role",
        normalize_job_context=lambda _: CONTEXT, read_project_memory=lambda: {},
        feature_enabled=lambda: True,
    ))
    with TestClient(app) as client:
        failed = client.post("/api/hiring-context/review", json={})
        recovered = client.post("/api/hiring-context/review", json={})
    assert failed.status_code == 200
    assert failed.json()["status"] == "error"
    assert failed.json()["projects"] == [] and failed.json()["preference_binding"] is None
    assert "fingerprint" not in failed.text and "e" * 64 not in failed.text
    assert recovered.json()["status"] == "ready"


@pytest.mark.parametrize("consumer", (review, production))
def test_both_consumers_use_the_same_authoritative_generation_rule(coherent_memory, monkeypatch, consumer):
    memory, _ = coherent_memory
    validator = Mock(wraps=memory_service.validate_authoritative_story_memory_generation)
    monkeypatch.setattr(consumer, "validate_authoritative_story_memory_generation", validator)
    if consumer is review:
        assert preview({}).status is review.HiringContextRankingReviewStatus.READY
    else:
        assert production_decision(memory, {}).plan is not None
    validator.assert_called_once_with(memory, coherent_memory[1])


@pytest.mark.parametrize("empty", (False, True))
def test_authoritative_validator_accepts_stable_and_rejects_changed_generation(coherent_memory, empty):
    memory, readiness = coherent_memory
    if empty:
        memory = build_authoritative_engineering_story_memory(_snapshot()).memory
        readiness = replace(readiness, logical_fingerprint=memory.logical_fingerprint)
    before = memory.to_dict()
    memory_service.validate_authoritative_story_memory_generation(memory, readiness)
    with pytest.raises(memory_service.EngineeringStoryMemoryServiceError, match="^integrity:integrity_failed$"):
        memory_service.validate_authoritative_story_memory_generation(memory, replace(readiness, logical_fingerprint="e" * 64))
    assert memory.to_dict() == before


def test_review_delegates_all_ranking_and_acquisition_policies_through_shared_service():
    source = inspect.getsource(review)
    tree = ast.parse(source)
    forbidden = {
        "rank_engineering_stories_for_hiring_context", "evaluate_engineering_story_relevance",
        "aggregate_project_story_relevance", "rank_project_portfolio",
        "calculate_story_web_context_adjustment", "apply_hiring_context_web_influence",
        "rank_projects_with_hiring_context_web_influence", "build_hiring_context_web_enrichment",
        "rank_projects_for_resume",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & forbidden
    assert "build_configured_hiring_context_ranking_state" in names
    for token in ("import httpx", "import requests", "evidence_claim_safety", "MAX_WEB_CONTEXT_ADJUSTMENT"):
        assert token not in source
    body = inspect.getsource(review.build_hiring_context_ranking_review)
    assert "logical_fingerprint" not in body
    assert ROLLOUT not in source


def test_web_presentation_keeps_existing_product_safe_shape(coherent_memory, monkeypatch):
    monkeypatch.setattr(service, "build_hiring_context_web_enrichment", lambda profile, **_: enrichment(profile))
    payload = preview({ENRICH: "1", INFLUENCE: "1"}).to_dict()
    assert payload["status"] == "ready"
    # Existing opaque preference binding stays unchanged; diagnostics never enter visible copy.
    payload.pop("preference_binding")
    text = json.dumps(payload)
    for token in ("score", "adjustment", "specificity", "fingerprint", "policy", "https://", "base_relevance"):
        assert token not in text
