from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
import itertools
from pathlib import Path

import pytest

import backend.project_story_ranking_refresh as refresh_module
from backend.engineering_story_memory_service import EngineeringStoryView
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryContextGap,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    evaluate_engineering_story_relevance,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import HiringContextProfile
from backend.project_portfolio_ranking import (
    PortfolioProjectRanking,
    RankedProjectStoryPortfolio,
    rank_project_portfolio,
)
from backend.project_story_ranking import (
    MAX_CONTRIBUTING_STORIES,
    MAX_PROJECT_STORY_INPUTS,
    ProjectHiringRelevance,
    aggregate_project_story_relevance,
)
from backend.project_story_ranking_refresh import (
    MAX_PORTFOLIO_RANKING_DELTA_CHANGES,
    PROJECT_STORY_RANKING_REFRESH_POLICY_ID,
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    PartialProjectStoryRerankResult,
    PortfolioRankingChange,
    PortfolioRankingDelta,
    ProjectStoryRankingRefreshError,
    ProjectStoryRankingRefreshErrorCode,
    ProjectStoryRankingRefreshStatus,
    ProjectStoryRelevanceSnapshot,
    refresh_project_story_ranking,
)
from backend.story_clarification_handoff import (
    build_story_clarification_handoffs,
)


MODULE_PATH = (
    Path(__file__).parents[1] / "backend" / "project_story_ranking_refresh.py"
)
PROJECT_A = "refresh_project_a"
PROJECT_B = "refresh_project_b"
PROJECT_C = "refresh_project_c"
EVIDENCE_ID = "pef_ranking_refresh"
CAPABILITY_ID = "pcf_ranking_refresh"
FIELD_NAMES = tuple(item.value for item in EngineeringStoryFieldName)


def _positive(value: str) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=value,
        evidence_state=StoryFieldEvidenceState.CONFIRMED,
        evidence_fact_ids=(EVIDENCE_ID,),
    )


def _absent() -> EngineeringStoryField:
    return EngineeringStoryField(
        value=None,
        evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
    )


def _assessment(
    cls,
    level: SufficiencyLevel,
    positive_names: tuple[EngineeringStoryFieldName, ...],
    missing_names: tuple[EngineeringStoryFieldName, ...],
):
    if level is SufficiencyLevel.UNASSESSED:
        return cls(level=level)
    return cls(
        level=level,
        supported_fields=positive_names,
        missing_fields=(
            () if level is SufficiencyLevel.HIGH else missing_names[:3]
        ),
    )


def _view(
    *,
    story_key: str,
    revision_key: str,
    project_id: str,
    text: str,
    claim_level: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_level: SufficiencyLevel = SufficiencyLevel.HIGH,
    opportunity_level: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    status: EngineeringStoryStatus = EngineeringStoryStatus.ACTIVE,
    requires_revalidation: bool = False,
    provenance_override: str | None = None,
) -> EngineeringStoryView:
    story_token = hashlib.sha256(story_key.encode("utf-8")).hexdigest()[:24]
    revision_token = hashlib.sha256(
        revision_key.encode("utf-8")
    ).hexdigest()[:24]
    canonical_id = f"engineering_story_{story_token}"
    story_fields = {
        name: _positive(text) if name == "mechanism" else _absent()
        for name in FIELD_NAMES
    }
    positive_names = (EngineeringStoryFieldName.MECHANISM,)
    missing_names = tuple(
        item
        for item in EngineeringStoryFieldName
        if item is not EngineeringStoryFieldName.MECHANISM
    )
    lifecycle = EngineeringStoryLifecycle(
        status=status,
        requires_revalidation=requires_revalidation,
        superseded_by_story_id=(
            "engineering_story_successor"
            if status is EngineeringStoryStatus.SUPERSEDED
            else None
        ),
    )
    claim = _assessment(
        ClaimSufficiency,
        claim_level,
        positive_names,
        missing_names,
    )
    story_sufficiency = _assessment(
        StorySufficiency,
        story_level,
        positive_names,
        missing_names,
    )
    opportunity = (
        StoryOpportunity(level=StoryOpportunityLevel.NONE)
        if opportunity_level is StoryOpportunityLevel.NONE
        else StoryOpportunity(
            level=opportunity_level,
            signals=(StoryOpportunitySignal.MAJOR_DESIGN_DECISION,),
            missing_context=(StoryContextGap.DECISION_REASON,),
        )
    )
    story = EngineeringStory(
        story_id=canonical_id,
        project_id=project_id,
        story_type=EngineeringStoryType.OTHER,
        **story_fields,
        evidence_fact_ids=(EVIDENCE_ID,),
        capability_fact_ids=(CAPABILITY_ID,),
        claim_boundary_ids=(),
        lifecycle=lifecycle,
        claim_sufficiency=claim,
        story_sufficiency=story_sufficiency,
        opportunity=opportunity,
    )
    provenance = provenance_override or hashlib.sha256(
        (
            f"{story_key}|{revision_key}|{text}|{claim_level.value}|"
            f"{story_level.value}|{opportunity_level.value}|{status.value}|"
            f"{requires_revalidation}"
        ).encode("utf-8")
    ).hexdigest()
    return EngineeringStoryView(
        canonical_story_id=canonical_id,
        project_id=project_id,
        current_story=story,
        claim_sufficiency=claim,
        story_sufficiency=story_sufficiency,
        opportunity=opportunity,
        lifecycle=lifecycle,
        current_revision_id=f"engineering_story_revision_{revision_token}",
        evidence_fact_ids=story.evidence_fact_ids,
        capability_fact_ids=story.capability_fact_ids,
        claim_boundary_ids=story.claim_boundary_ids,
        provenance_fingerprint=provenance,
        source_lineage_fingerprints=(provenance,),
    )


def _context(
    *,
    title: str = "Backend Engineer",
    required: tuple[str, ...] = ("Build reliable backend API systems",),
) -> HiringContextProfile:
    return build_hiring_context_profile(
        company="Unregistered Systems",
        team=None,
        parent_organization=None,
        role_title=title,
        normalized_job_context={"required_qualifications": required},
    )


def _is_rankable(view: EngineeringStoryView) -> bool:
    return (
        view.lifecycle.status is EngineeringStoryStatus.ACTIVE
        and not view.lifecycle.requires_revalidation
    )


def _clean_state(
    profile: HiringContextProfile,
    views: tuple[EngineeringStoryView, ...],
) -> tuple[
    dict[str, tuple[StoryHiringRelevance, ...]],
    dict[str, ProjectHiringRelevance],
    RankedProjectStoryPortfolio | None,
]:
    by_project: dict[str, list[StoryHiringRelevance]] = {}
    for view in views:
        if not _is_rankable(view):
            continue
        result = evaluate_engineering_story_relevance(
            hiring_context=profile,
            story_view=view,
        )
        by_project.setdefault(view.project_id, []).append(result)
    story_results: dict[str, tuple[StoryHiringRelevance, ...]] = {}
    projects: dict[str, ProjectHiringRelevance] = {}
    for project_id, values in by_project.items():
        stories = tuple(sorted(values, key=lambda item: (
            -item.total_relevance_score,
            item.canonical_story_id,
            item.current_revision_id,
            item.relevance_id,
        )))
        project = aggregate_project_story_relevance(
            project_id=project_id,
            story_relevance=stories,
        )
        assert project is not None
        story_results[project_id] = stories
        projects[project_id] = project
    portfolio = rank_project_portfolio(projects=tuple(projects.values()))
    return story_results, projects, portfolio


def _snapshot(
    *,
    portfolio: RankedProjectStoryPortfolio,
    project: ProjectHiringRelevance,
    stories: tuple[StoryHiringRelevance, ...],
) -> ProjectStoryRelevanceSnapshot:
    return ProjectStoryRelevanceSnapshot(
        snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
        source_portfolio_id=portfolio.portfolio_id,
        project_relevance=project,
        story_relevance=stories,
    )


def _prior(
    profile: HiringContextProfile,
    views: tuple[EngineeringStoryView, ...],
    *,
    affected_project: str = PROJECT_A,
) -> tuple[
    RankedProjectStoryPortfolio,
    ProjectStoryRelevanceSnapshot,
    dict[str, tuple[StoryHiringRelevance, ...]],
    dict[str, ProjectHiringRelevance],
]:
    stories, projects, portfolio = _clean_state(profile, views)
    assert portfolio is not None
    snapshot = _snapshot(
        portfolio=portfolio,
        project=projects[affected_project],
        stories=stories[affected_project],
    )
    return portfolio, snapshot, stories, projects


def _ranking(
    portfolio: RankedProjectStoryPortfolio,
    project_id: str,
) -> PortfolioProjectRanking:
    return next(
        item for item in portfolio.ranked_projects if item.project_id == project_id
    )


def _basic_views() -> tuple[EngineeringStoryView, EngineeringStoryView]:
    old = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Implemented validation",
    )
    other = _view(
        story_key="other",
        revision_key="other_r1",
        project_id=PROJECT_B,
        text="Built reliable backend API systems",
    )
    return old, other


def _updated_basic() -> EngineeringStoryView:
    return _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture reliability storage",
    )


def _refresh_basic() -> tuple[
    PartialProjectStoryRerankResult,
    HiringContextProfile,
    EngineeringStoryView,
    EngineeringStoryView,
    RankedProjectStoryPortfolio,
    ProjectStoryRelevanceSnapshot,
]:
    profile = _context()
    old, other = _basic_views()
    prior_portfolio, snapshot, _stories, _projects = _prior(
        profile,
        (old, other),
    )
    updated = _updated_basic()
    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=prior_portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    return result, profile, old, updated, prior_portfolio, snapshot


def test_revision_advance_recomputes_affected_story_relevance() -> None:
    result, profile, _old, updated, _portfolio, _snapshot_value = _refresh_basic()
    assert result.updated_project_snapshot is not None
    refreshed = next(
        item
        for item in result.updated_project_snapshot.story_relevance
        if item.canonical_story_id == updated.canonical_story_id
    )
    clean = evaluate_engineering_story_relevance(
        hiring_context=profile,
        story_view=updated,
    )

    assert refreshed == clean
    assert refreshed.current_revision_id == updated.current_revision_id
    assert result.status is ProjectStoryRankingRefreshStatus.RERANKED


def test_unaffected_story_relevance_is_reused_by_identity() -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Implemented validation",
    )
    unchanged = _view(
        story_key="unchanged",
        revision_key="unchanged_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )
    other = _view(
        story_key="other",
        revision_key="other_r1",
        project_id=PROJECT_B,
        text="Built testing automation",
    )
    portfolio, snapshot, _stories, _projects = _prior(
        profile,
        (affected, unchanged, other),
    )
    old_unchanged = next(
        item
        for item in snapshot.story_relevance
        if item.canonical_story_id == unchanged.canonical_story_id
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=_updated_basic(),
    )

    assert result.updated_project_snapshot is not None
    new_unchanged = next(
        item
        for item in result.updated_project_snapshot.story_relevance
        if item.canonical_story_id == unchanged.canonical_story_id
    )
    assert new_unchanged is old_unchanged


def test_unaffected_project_relevance_is_reused_by_identity() -> None:
    result, _profile, _old, _updated, prior, _snapshot_value = _refresh_basic()
    assert result.new_portfolio is not None
    old_project = _ranking(prior, PROJECT_B).project_relevance
    new_project = _ranking(result.new_portfolio, PROJECT_B).project_relevance

    assert new_project is old_project


def test_owning_project_is_reaggregated_from_complete_snapshot() -> None:
    result, _profile, _old, _updated, _portfolio, snapshot = _refresh_basic()
    assert result.updated_project_snapshot is not None

    assert result.updated_project_snapshot.project_relevance != (
        snapshot.project_relevance
    )
    assert result.delta.old_project_relevance_id == (
        snapshot.project_relevance.project_relevance_id
    )
    assert result.delta.new_project_relevance_id == (
        result.updated_project_snapshot.project_relevance.project_relevance_id
    )


def test_portfolio_is_reconciled_with_existing_ranking_service() -> None:
    result, _profile, _old, _updated, prior, _snapshot_value = _refresh_basic()
    assert result.new_portfolio is not None
    affected = result.updated_project_snapshot
    assert affected is not None
    clean = rank_project_portfolio(projects=(
        affected.project_relevance,
        _ranking(prior, PROJECT_B).project_relevance,
    ))

    assert result.new_portfolio == clean


def test_same_revision_and_authoritative_state_is_exact_no_change() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=old,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.NO_CHANGE
    assert result.new_portfolio is portfolio
    assert result.updated_project_snapshot is snapshot
    assert result.delta.changes == (PortfolioRankingChange.NO_CHANGE,)
    assert not result.delta.changed


def test_same_revision_no_change_skips_semantic_ranking_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    calls = {"story": 0, "project": 0, "portfolio": 0}

    def unexpected_story(**_kwargs):
        calls["story"] += 1
        raise AssertionError("Story evaluator must not run")

    def unexpected_project(**_kwargs):
        calls["project"] += 1
        raise AssertionError("project aggregation must not run")

    def unexpected_portfolio(**_kwargs):
        calls["portfolio"] += 1
        raise AssertionError("portfolio ranking must not run")

    monkeypatch.setattr(
        refresh_module,
        "evaluate_engineering_story_relevance",
        unexpected_story,
    )
    monkeypatch.setattr(
        refresh_module,
        "aggregate_project_story_relevance",
        unexpected_project,
    )
    monkeypatch.setattr(
        refresh_module,
        "rank_project_portfolio",
        unexpected_portfolio,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=old,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.NO_CHANGE
    assert calls == {"story": 0, "project": 0, "portfolio": 0}


def test_same_revision_with_conflicting_provenance_fails_closed() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    conflicting = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Conflicting content under one revision",
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=conflicting,
        )

    assert caught.value.code is (
        ProjectStoryRankingRefreshErrorCode.SAME_REVISION_CONFLICT
    )


@pytest.mark.parametrize(
    ("claim", "story", "opportunity"),
    (
        (SufficiencyLevel.LOW, SufficiencyLevel.HIGH, StoryOpportunityLevel.NONE),
        (SufficiencyLevel.HIGH, SufficiencyLevel.LOW, StoryOpportunityLevel.NONE),
        (SufficiencyLevel.HIGH, SufficiencyLevel.HIGH, StoryOpportunityLevel.HIGH),
    ),
)
def test_same_revision_with_changed_authoritative_state_fails_closed(
    claim: SufficiencyLevel,
    story: SufficiencyLevel,
    opportunity: StoryOpportunityLevel,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    conflicting = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Implemented validation",
        claim_level=claim,
        story_level=story,
        opportunity_level=opportunity,
        provenance_override=old.provenance_fingerprint,
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=conflicting,
        )

    assert caught.value.code is (
        ProjectStoryRankingRefreshErrorCode.SAME_REVISION_CONFLICT
    )


def test_wrong_project_fails_closed() -> None:
    result, profile, _old, _updated, portfolio, snapshot = _refresh_basic()
    assert result.new_portfolio is not None
    wrong = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_B,
        text="Designed distributed backend API architecture",
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=wrong,
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.IDENTITY_MISMATCH


def test_new_canonical_story_insertion_is_rejected() -> None:
    _result, profile, _old, _updated, portfolio, snapshot = _refresh_basic()
    unknown = _view(
        story_key="new_story",
        revision_key="new_story_r1",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture",
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=unknown,
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.STORY_NOT_FOUND


def test_changed_hiring_context_fails_closed() -> None:
    _result, _profile, _old, updated, portfolio, snapshot = _refresh_basic()
    changed = _context(
        title="Embedded Engineer",
        required=("Build embedded firmware",),
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=changed,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=updated,
        )

    assert caught.value.code is (
        ProjectStoryRankingRefreshErrorCode.MIXED_HIRING_CONTEXT
    )


@pytest.mark.parametrize(
    ("attribute", "wrong_value"),
    (
        ("profile_id", "hiring_context_stale"),
        ("fingerprint", "b" * 64),
    ),
)
def test_tampered_hiring_context_identity_fails_closed(
    attribute: str,
    wrong_value: str,
) -> None:
    _result, profile, _old, updated, portfolio, snapshot = _refresh_basic()
    object.__setattr__(profile, attribute, wrong_value)

    with pytest.raises(ProjectStoryRankingRefreshError):
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=updated,
        )


def test_malformed_updated_revision_identity_fails_before_ranking() -> None:
    _result, profile, _old, updated, portfolio, snapshot = _refresh_basic()
    object.__setattr__(updated, "current_revision_id", "revision_invalid")

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=updated,
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.INVALID_INPUT


def test_snapshot_source_portfolio_identity_is_enforced() -> None:
    _result, profile, _old, updated, portfolio, snapshot = _refresh_basic()
    stale = replace(
        snapshot,
        source_portfolio_id="ranked_project_story_portfolio_stale",
        snapshot_id="",
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=stale,
            updated_story_view=updated,
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO


@pytest.mark.parametrize(
    ("target", "attribute", "wrong_value"),
    (
        ("portfolio", "portfolio_id", "ranked_project_story_portfolio_stale"),
        ("project", "project_relevance_id", "project_relevance_stale"),
        ("story", "relevance_id", "story_hiring_relevance_stale"),
    ),
)
def test_tampered_prior_ranking_identity_fails_closed(
    target: str,
    attribute: str,
    wrong_value: str,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    targets = {
        "portfolio": portfolio,
        "project": snapshot.project_relevance,
        "story": snapshot.story_relevance[0],
    }
    object.__setattr__(targets[target], attribute, wrong_value)

    with pytest.raises((ProjectStoryRankingRefreshError, ValueError)):
        refresh_project_story_ranking(
            hiring_context=profile,
            prior_portfolio=portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=_updated_basic(),
        )


@pytest.mark.parametrize(
    ("claim_level", "story_level"),
    (
        (SufficiencyLevel.HIGH, SufficiencyLevel.LOW),
        (SufficiencyLevel.LOW, SufficiencyLevel.HIGH),
        (SufficiencyLevel.HIGH, SufficiencyLevel.HIGH),
    ),
)
def test_revision_advance_preserves_new_authoritative_sufficiency(
    claim_level: SufficiencyLevel,
    story_level: SufficiencyLevel,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    updated = _view(
        story_key="affected",
        revision_key=f"affected_{claim_level.value}_{story_level.value}",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture",
        claim_level=claim_level,
        story_level=story_level,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    assert result.updated_project_snapshot is not None
    refreshed = next(
        item
        for item in result.updated_project_snapshot.story_relevance
        if item.canonical_story_id == updated.canonical_story_id
    )

    assert refreshed.claim_sufficiency is claim_level
    assert refreshed.story_sufficiency is story_level


@pytest.mark.parametrize("opportunity", tuple(StoryOpportunityLevel))
def test_story_opportunity_changes_only_through_updated_view(
    opportunity: StoryOpportunityLevel,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    updated = _view(
        story_key="affected",
        revision_key=f"affected_opportunity_{opportunity.value}",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture",
        opportunity_level=opportunity,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    assert result.updated_project_snapshot is not None
    refreshed = next(
        item
        for item in result.updated_project_snapshot.story_relevance
        if item.canonical_story_id == updated.canonical_story_id
    )

    assert refreshed.story_opportunity is opportunity


def _four_story_views(
    *,
    affected_text: str,
    affected_revision: str,
) -> tuple[EngineeringStoryView, ...]:
    return (
        _view(
            story_key="story_architecture",
            revision_key="story_architecture_r1",
            project_id=PROJECT_A,
            text="Designed distributed backend API architecture",
        ),
        _view(
            story_key="story_api",
            revision_key="story_api_r1",
            project_id=PROJECT_A,
            text="Built reliable backend API systems",
        ),
        _view(
            story_key="story_testing",
            revision_key="story_testing_r1",
            project_id=PROJECT_A,
            text="Built testing automation",
        ),
        _view(
            story_key="story_affected",
            revision_key=affected_revision,
            project_id=PROJECT_A,
            text=affected_text,
        ),
    )


def test_updated_story_can_enter_top_three_contributions() -> None:
    profile = _context()
    prior_views = _four_story_views(
        affected_text="Updated documentation",
        affected_revision="story_affected_r1",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, prior_views)
    affected_id = prior_views[-1].canonical_story_id
    assert affected_id not in {
        item.canonical_story_id
        for item in snapshot.project_relevance.contributions
    }
    updated = _view(
        story_key="story_affected",
        revision_key="story_affected_r2",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture reliability storage",
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    assert result.updated_project_snapshot is not None

    assert affected_id in {
        item.canonical_story_id
        for item in result.updated_project_snapshot.project_relevance.contributions
    }


def test_updated_story_can_leave_top_three_and_fourth_enters() -> None:
    profile = _context()
    prior_views = _four_story_views(
        affected_text="Designed distributed backend API architecture reliability storage",
        affected_revision="story_affected_r1",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, prior_views)
    affected_id = prior_views[-1].canonical_story_id
    prior_contributors = {
        item.canonical_story_id
        for item in snapshot.project_relevance.contributions
    }
    assert affected_id in prior_contributors
    updated = _view(
        story_key="story_affected",
        revision_key="story_affected_r2",
        project_id=PROJECT_A,
        text="Updated documentation",
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    assert result.updated_project_snapshot is not None
    new_contributors = {
        item.canonical_story_id
        for item in result.updated_project_snapshot.project_relevance.contributions
    }

    assert affected_id not in new_contributors
    assert len(new_contributors) == MAX_CONTRIBUTING_STORIES
    assert new_contributors - prior_contributors


def test_weak_updated_story_cannot_retain_a_stale_contribution() -> None:
    profile = _context()
    prior_views = _four_story_views(
        affected_text="Designed distributed backend API architecture reliability storage",
        affected_revision="story_affected_r1",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, prior_views)
    updated = _view(
        story_key="story_affected",
        revision_key="story_affected_r2",
        project_id=PROJECT_A,
        text="Updated documentation",
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    assert result.updated_project_snapshot is not None

    assert updated.canonical_story_id not in {
        item.canonical_story_id
        for item in result.updated_project_snapshot.project_relevance.contributions
    }


@pytest.mark.parametrize(
    ("old_text", "new_text", "direction"),
    (
        (
            "Updated documentation",
            "Designed distributed backend API architecture reliability storage",
            "increase",
        ),
        (
            "Designed distributed backend API architecture reliability storage",
            "Updated documentation",
            "decrease",
        ),
    ),
)
def test_story_and_project_scores_move_in_expected_direction(
    old_text: str,
    new_text: str,
    direction: str,
) -> None:
    profile = _context()
    old = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text=old_text,
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, (old,))
    updated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text=new_text,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    compare = (lambda new, old_value: new > old_value) if direction == "increase" else (
        lambda new, old_value: new < old_value
    )

    assert result.delta.new_story_relevance is not None
    assert result.delta.new_project_relevance is not None
    assert compare(
        result.delta.new_story_relevance,
        result.delta.old_story_relevance,
    )
    assert compare(
        result.delta.new_project_relevance,
        result.delta.old_project_relevance,
    )


@pytest.mark.parametrize(
    ("old_text", "new_text", "old_position", "new_position"),
    (
        (
            "Updated documentation",
            "Designed distributed backend API architecture reliability storage",
            3,
            1,
        ),
        (
            "Designed distributed backend API architecture reliability storage",
            "Updated documentation",
            1,
            3,
        ),
    ),
)
def test_affected_project_can_move_up_or_down_portfolio(
    old_text: str,
    new_text: str,
    old_position: int,
    new_position: int,
) -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text=old_text,
    )
    project_b = _view(
        story_key="project_b",
        revision_key="project_b_r1",
        project_id=PROJECT_B,
        text="Built reliable backend API systems",
    )
    project_c = _view(
        story_key="project_c",
        revision_key="project_c_r1",
        project_id=PROJECT_C,
        text="Built testing automation",
    )
    portfolio, snapshot, _stories, _projects = _prior(
        profile,
        (affected, project_b, project_c),
    )
    updated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text=new_text,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )

    assert result.delta.old_position == old_position
    assert result.delta.new_position == new_position
    assert PortfolioRankingChange.PORTFOLIO_POSITION_CHANGED in (
        result.delta.changes
    )


@pytest.mark.parametrize(
    ("old_text", "new_text"),
    (
        (
            "Updated documentation",
            "Designed distributed backend API architecture reliability storage",
        ),
        (
            "Designed distributed backend API architecture reliability storage",
            "Updated documentation",
        ),
    ),
)
def test_partial_refresh_equals_clean_full_rerank(
    old_text: str,
    new_text: str,
) -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text=old_text,
    )
    same_project = _view(
        story_key="same_project",
        revision_key="same_project_r1",
        project_id=PROJECT_A,
        text="Built testing automation",
    )
    project_b = _view(
        story_key="project_b",
        revision_key="project_b_r1",
        project_id=PROJECT_B,
        text="Built reliable backend API systems",
    )
    project_c = _view(
        story_key="project_c",
        revision_key="project_c_r1",
        project_id=PROJECT_C,
        text="Implemented gameplay game state in a real-time frame loop",
    )
    prior_views = (affected, same_project, project_b, project_c)
    portfolio, snapshot, _stories, _projects = _prior(profile, prior_views)
    updated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text=new_text,
    )

    partial = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    clean_stories, clean_projects, clean_portfolio = _clean_state(
        profile,
        (updated, same_project, project_b, project_c),
    )

    assert partial.updated_project_snapshot is not None
    assert partial.updated_project_snapshot.story_relevance == (
        clean_stories[PROJECT_A]
    )
    assert partial.updated_project_snapshot.project_relevance == (
        clean_projects[PROJECT_A]
    )
    assert partial.new_portfolio == clean_portfolio
    assert partial.new_portfolio is not None
    assert partial.new_portfolio.to_dict() == clean_portfolio.to_dict()
    assert partial.updated_handoffs == build_story_clarification_handoffs(
        portfolio=clean_portfolio
    )


@pytest.mark.parametrize(
    ("old_text", "new_text"),
    (
        (
            "Updated documentation",
            "Designed distributed backend API architecture reliability storage",
        ),
        (
            "Designed distributed backend API architecture reliability storage",
            "Updated documentation",
        ),
    ),
)
def test_partial_full_equivalence_when_top_three_membership_changes(
    old_text: str,
    new_text: str,
) -> None:
    profile = _context()
    prior_views = _four_story_views(
        affected_text=old_text,
        affected_revision="story_affected_r1",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, prior_views)
    updated = _view(
        story_key="story_affected",
        revision_key="story_affected_r2",
        project_id=PROJECT_A,
        text=new_text,
    )

    partial = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    final_views = prior_views[:-1] + (updated,)
    clean_stories, clean_projects, clean_portfolio = _clean_state(
        profile,
        final_views,
    )

    assert partial.updated_project_snapshot is not None
    assert partial.updated_project_snapshot.story_relevance == (
        clean_stories[PROJECT_A]
    )
    assert partial.updated_project_snapshot.project_relevance == (
        clean_projects[PROJECT_A]
    )
    assert partial.new_portfolio == clean_portfolio


def test_global_redundancy_and_differentiation_match_clean_rerank() -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Worked on unrelated visual styling",
    )
    project_b = _view(
        story_key="project_b",
        revision_key="project_b_r1",
        project_id=PROJECT_B,
        text="Built reliable backend API systems",
    )
    project_c = _view(
        story_key="project_c",
        revision_key="project_c_r1",
        project_id=PROJECT_C,
        text="Designed distributed backend API architecture",
    )
    portfolio, snapshot, _stories, _projects = _prior(
        profile,
        (affected, project_b, project_c),
    )
    updated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text="Designed distributed backend API architecture reliability storage",
    )

    partial = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    _stories2, _projects2, clean = _clean_state(
        profile,
        (updated, project_b, project_c),
    )
    assert partial.new_portfolio is not None
    assert clean is not None

    assert tuple(
        (
            item.project_id,
            item.adjustment,
            item.portfolio_relevance,
            item.semantic_footprint,
        )
        for item in partial.new_portfolio.ranked_projects
    ) == tuple(
        (
            item.project_id,
            item.adjustment,
            item.portfolio_relevance,
            item.semantic_footprint,
        )
        for item in clean.ranked_projects
    )
    old_b = _ranking(portfolio, PROJECT_B)
    new_b = _ranking(partial.new_portfolio, PROJECT_B)
    old_c = _ranking(portfolio, PROJECT_C)
    new_c = _ranking(partial.new_portfolio, PROJECT_C)

    assert old_b.adjustment != new_b.adjustment
    assert old_c.adjustment != new_c.adjustment
    assert old_b.position != new_b.position
    assert old_c.position != new_c.position


@pytest.mark.parametrize(
    ("status", "requires_revalidation"),
    (
        (EngineeringStoryStatus.SUPERSEDED, False),
        (EngineeringStoryStatus.STALE, True),
        (EngineeringStoryStatus.CONFLICTED, True),
        (EngineeringStoryStatus.ACTIVE, True),
    ),
)
def test_nonrankable_updated_story_is_removed_without_evaluation(
    status: EngineeringStoryStatus,
    requires_revalidation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )
    survivor = _view(
        story_key="survivor",
        revision_key="survivor_r1",
        project_id=PROJECT_A,
        text="Built testing automation",
    )
    other = _view(
        story_key="other",
        revision_key="other_r1",
        project_id=PROJECT_B,
        text="Implemented validation",
    )
    portfolio, snapshot, _stories, _projects = _prior(
        profile,
        (affected, survivor, other),
    )
    invalidated = _view(
        story_key="affected",
        revision_key=f"affected_{status.value}",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
        status=status,
        requires_revalidation=requires_revalidation,
    )

    def unexpected_evaluation(**_kwargs):
        raise AssertionError("nonrankable Story must not be evaluated")

    monkeypatch.setattr(
        refresh_module,
        "evaluate_engineering_story_relevance",
        unexpected_evaluation,
    )
    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=invalidated,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.STORY_REMOVED
    assert result.updated_project_snapshot is not None
    assert affected.canonical_story_id not in {
        item.canonical_story_id
        for item in result.updated_project_snapshot.story_relevance
    }
    assert result.delta.new_story_relevance_id is None


def test_last_rankable_story_removes_project_and_can_empty_portfolio() -> None:
    profile = _context()
    affected = _view(
        story_key="affected",
        revision_key="affected_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, (affected,))
    invalidated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
        status=EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=invalidated,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED
    assert result.new_portfolio is None
    assert result.updated_project_snapshot is None
    assert result.updated_handoffs == ()
    assert result.delta.new_project_relevance_id is None
    assert result.delta.new_portfolio_id is None


def test_removed_project_is_excluded_while_unaffected_projects_remain() -> None:
    profile = _context()
    affected, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (affected, other))
    invalidated = _view(
        story_key="affected",
        revision_key="affected_r2",
        project_id=PROJECT_A,
        text="Implemented validation",
        status=EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )

    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=invalidated,
    )

    assert result.new_portfolio is not None
    assert tuple(item.project_id for item in result.new_portfolio.ranked_projects) == (
        PROJECT_B,
    )
    assert result.status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED


def test_new_handoffs_reference_new_revision_and_old_is_stale() -> None:
    result, _profile, old, updated, prior, _snapshot_value = _refresh_basic()
    old_handoff = next(
        item
        for item in build_story_clarification_handoffs(portfolio=prior)
        if item.canonical_story_id == old.canonical_story_id
    )
    new_handoff = next(
        item
        for item in result.updated_handoffs
        if item.canonical_story_id == updated.canonical_story_id
    )

    assert old_handoff.current_revision_id == old.current_revision_id
    assert new_handoff.current_revision_id == updated.current_revision_id
    assert old_handoff.handoff_id != new_handoff.handoff_id


def test_active_refresh_calls_each_authoritative_ranking_service_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    calls = {"story": 0, "project": 0, "portfolio": 0}
    original_story = refresh_module.evaluate_engineering_story_relevance
    original_project = refresh_module.aggregate_project_story_relevance
    original_portfolio = refresh_module.rank_project_portfolio

    def story_wrapper(**kwargs):
        calls["story"] += 1
        return original_story(**kwargs)

    def project_wrapper(**kwargs):
        calls["project"] += 1
        return original_project(**kwargs)

    def portfolio_wrapper(**kwargs):
        calls["portfolio"] += 1
        return original_portfolio(**kwargs)

    monkeypatch.setattr(
        refresh_module,
        "evaluate_engineering_story_relevance",
        story_wrapper,
    )
    monkeypatch.setattr(
        refresh_module,
        "aggregate_project_story_relevance",
        project_wrapper,
    )
    monkeypatch.setattr(
        refresh_module,
        "rank_project_portfolio",
        portfolio_wrapper,
    )

    refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=_updated_basic(),
    )

    assert calls == {"story": 1, "project": 1, "portfolio": 1}


def test_inputs_are_not_mutated() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    updated = _updated_basic()
    before = (
        repr(profile),
        portfolio.to_dict(),
        snapshot.to_dict(),
        repr(updated),
    )

    refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )

    assert before == (
        repr(profile),
        portfolio.to_dict(),
        snapshot.to_dict(),
        repr(updated),
    )


def test_snapshot_input_permutation_is_deterministic() -> None:
    profile = _context()
    views = _four_story_views(
        affected_text="Updated documentation",
        affected_revision="story_affected_r1",
    )
    portfolio, snapshot, _stories, _projects = _prior(profile, views)

    for permutation in itertools.permutations(snapshot.story_relevance):
        candidate = ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=snapshot.project_relevance,
            story_relevance=permutation,
        )
        assert candidate == snapshot


def test_exact_duplicate_snapshot_source_is_deduplicated() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    duplicate = ProjectStoryRelevanceSnapshot(
        snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
        source_portfolio_id=portfolio.portfolio_id,
        project_relevance=snapshot.project_relevance,
        story_relevance=(
            snapshot.story_relevance[0],
            snapshot.story_relevance[0],
        ),
    )

    assert duplicate.story_relevance == snapshot.story_relevance
    assert duplicate.snapshot_id == snapshot.snapshot_id


def test_conflicting_duplicate_story_snapshot_fails_closed() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))
    story = snapshot.story_relevance[0]
    conflict = replace(
        story,
        current_revision_id=(
            "engineering_story_revision_"
            + hashlib.sha256(b"conflict").hexdigest()[:24]
        ),
        relevance_id="",
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=snapshot.project_relevance,
            story_relevance=(story, conflict),
        )

    assert caught.value.code is (
        ProjectStoryRankingRefreshErrorCode.CONFLICTING_STORY_SNAPSHOT
    )


def test_conflicting_project_snapshot_fails_closed() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, stories, projects = _prior(profile, (old, other))

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=projects[PROJECT_A],
            story_relevance=stories[PROJECT_B],
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.IDENTITY_MISMATCH


def test_snapshot_bound_is_enforced_before_duplicate_normalization() -> None:
    profile = _context()
    old, other = _basic_views()
    portfolio, snapshot, _stories, _projects = _prior(profile, (old, other))

    with pytest.raises(ProjectStoryRankingRefreshError) as caught:
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=snapshot.project_relevance,
            story_relevance=(snapshot.story_relevance[0],)
            * (MAX_PROJECT_STORY_INPUTS + 1),
        )

    assert caught.value.code is ProjectStoryRankingRefreshErrorCode.BOUND_EXCEEDED


def test_output_and_delta_are_immutable_and_deterministic() -> None:
    first, profile, _old, updated, portfolio, snapshot = _refresh_basic()
    second = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )

    assert first == second
    assert first.result_id == second.result_id
    assert first.delta.delta_id == second.delta.delta_id
    with pytest.raises(FrozenInstanceError):
        first.status = ProjectStoryRankingRefreshStatus.NO_CHANGE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.delta.old_position = 99  # type: ignore[misc]


def test_delta_bound_and_order_are_enforced() -> None:
    result, _profile, _old, _updated, _portfolio, _snapshot_value = _refresh_basic()
    assert len(result.delta.changes) <= MAX_PORTFOLIO_RANKING_DELTA_CHANGES
    assert result.delta.changes == tuple(
        item for item in PortfolioRankingChange if item in result.delta.changes
    )

    with pytest.raises(ValueError):
        replace(
            result.delta,
            changes=(
                PortfolioRankingChange.STORY_RELEVANCE_CHANGED,
                PortfolioRankingChange.PROJECT_RELEVANCE_CHANGED,
                PortfolioRankingChange.PORTFOLIO_POSITION_CHANGED,
                PortfolioRankingChange.PORTFOLIO_RELEVANCE_CHANGED,
                PortfolioRankingChange.STORY_RELEVANCE_CHANGED,
            ),
        )


@pytest.mark.parametrize(
    "status",
    (
        ProjectStoryRankingRefreshStatus.STORY_REMOVED,
        ProjectStoryRankingRefreshStatus.PROJECT_REMOVED,
    ),
)
def test_delta_status_cannot_contradict_new_state(
    status: ProjectStoryRankingRefreshStatus,
) -> None:
    result, _profile, _old, _updated, _portfolio, _snapshot_value = _refresh_basic()

    if status is ProjectStoryRankingRefreshStatus.STORY_REMOVED:
        changes = (
            PortfolioRankingChange.STORY_REMOVED,
            PortfolioRankingChange.PROJECT_RELEVANCE_CHANGED,
        )
    else:
        changes = (
            PortfolioRankingChange.STORY_REMOVED,
            PortfolioRankingChange.PROJECT_REMOVED,
            PortfolioRankingChange.PORTFOLIO_POSITION_CHANGED,
            PortfolioRankingChange.PORTFOLIO_RELEVANCE_CHANGED,
        )

    with pytest.raises(ValueError):
        replace(result.delta, status=status, changes=changes)


def test_delta_preserves_all_old_and_new_freshness_identity() -> None:
    result, _profile, old, updated, prior, snapshot = _refresh_basic()
    assert result.updated_project_snapshot is not None

    assert result.delta.affected_project_id == PROJECT_A
    assert result.delta.canonical_story_id == old.canonical_story_id
    assert result.delta.old_snapshot_id == snapshot.snapshot_id
    assert result.delta.new_snapshot_id == (
        result.updated_project_snapshot.snapshot_id
    )
    assert result.delta.old_revision_id == old.current_revision_id
    assert result.delta.new_revision_id == updated.current_revision_id
    assert result.delta.old_portfolio_id == prior.portfolio_id
    assert result.delta.new_portfolio_id == result.new_portfolio.portfolio_id
    assert result.delta.old_story_relevance_id != (
        result.delta.new_story_relevance_id
    )
    assert result.delta.old_project_relevance_id != (
        result.delta.new_project_relevance_id
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("refresh_policy_id", "unsupported"),
        ("status", "unsupported"),
        ("result_id", "stale_result"),
        ("updated_handoffs", []),
    ),
)
def test_result_contract_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    result, _profile, _old, _updated, _portfolio, _snapshot_value = _refresh_basic()

    with pytest.raises((TypeError, ValueError)):
        replace(result, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("affected_project_id", " project"),
        ("canonical_story_id", ""),
        ("old_snapshot_id", None),
        ("old_story_relevance", -0.1),
        ("new_story_relevance", 1.1),
        ("old_project_relevance", float("nan")),
        ("old_position", 0),
        ("new_position", 65),
        ("changes", []),
        ("delta_id", "stale_delta"),
    ),
)
def test_delta_contract_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    result, _profile, _old, _updated, _portfolio, _snapshot_value = _refresh_basic()

    with pytest.raises((TypeError, ValueError)):
        replace(result.delta, **{field_name: invalid_value})


FORBIDDEN_OUTPUT_FIELDS = (
    "should_ask_user",
    "clarification_priority",
    "question_text",
    "question_options",
    "searchable_repository_gap",
    "human_only_gap",
    "should_retrieve",
    "clarification_friction",
    "answerability",
    "information_gain",
    "project_budget",
    "story_budget",
    "bullet_budget",
    "line_budget",
    "page_budget",
    "candidate_fact",
    "candidate_technology",
    "candidate_identity",
    "candidate_persona",
    "user_answer",
    "user_question",
)


@pytest.mark.parametrize("forbidden_field", FORBIDDEN_OUTPUT_FIELDS)
def test_output_contracts_exclude_downstream_and_candidate_truth_fields(
    forbidden_field: str,
) -> None:
    output_fields = {
        item.name
        for cls in (
            ProjectStoryRelevanceSnapshot,
            PortfolioRankingDelta,
            PartialProjectStoryRerankResult,
        )
        for item in fields(cls)
    }

    assert forbidden_field not in output_fields


def test_production_module_has_closed_pure_import_surface() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    allowed = {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "re",
        "typing",
        "backend.engineering_story_memory_service",
        "backend.engineering_story_models",
        "backend.engineering_story_relevance",
        "backend.hiring_context_models",
        "backend.project_portfolio_ranking",
        "backend.project_story_ranking",
        "backend.story_clarification_handoff",
    }

    assert modules <= allowed


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "rank_engineering_stories_for_hiring_context",
        "load_engineering_story_memory",
        "build_authoritative_engineering_story_memory",
        "persist_engineering_story_memory",
        "reconstruct_engineering_story",
        "retrieve_project_evidence",
        "search_project_evidence",
        "rank_projects_for_resume",
        "select_staged_projects_with_ranking",
        "tailor_resume_staged",
        "open",
        "getenv",
        "putenv",
    ),
)
def test_refresh_does_not_call_forbidden_services(forbidden_call: str) -> None:
    source = inspect.getsource(refresh_project_story_ranking)
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert forbidden_call not in calls


@pytest.mark.parametrize(
    "forbidden_constructor",
    (
        "StoryHiringRelevance",
        "ProjectHiringRelevance",
        "RankedProjectStoryPortfolio",
        "StoryClarificationHandoff",
    ),
)
def test_refresh_does_not_manually_construct_accepted_ranking_outputs(
    forbidden_constructor: str,
) -> None:
    source = inspect.getsource(refresh_project_story_ranking)
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert forbidden_constructor not in calls


def test_refresh_has_one_call_site_per_authoritative_ranking_layer() -> None:
    source = inspect.getsource(refresh_project_story_ranking)
    calls = [
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.count("evaluate_engineering_story_relevance") == 1
    assert calls.count("aggregate_project_story_relevance") == 1
    assert calls.count("rank_project_portfolio") == 1


def test_module_has_no_numbered_stage_terminology() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert not __import__("re").search(r"(?i)phase[ _-]?[0-9]+", source)


def test_refresh_result_is_ready_for_later_rebudget_without_budget_logic() -> None:
    result, _profile, _old, _updated, _portfolio, _snapshot_value = _refresh_basic()
    names = {item.name for item in fields(result.delta)}

    assert {
        "affected_project_id",
        "canonical_story_id",
        "old_position",
        "new_position",
        "old_project_relevance_id",
        "new_project_relevance_id",
        "old_portfolio_id",
        "new_portfolio_id",
    } <= names
    assert names.isdisjoint({
        "project_budget",
        "story_budget",
        "bullet_budget",
    })
