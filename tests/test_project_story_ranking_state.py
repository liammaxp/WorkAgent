from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
from pathlib import Path

import pytest

import backend.project_story_ranking_state as state_module
from backend.engineering_story_models import (
    EngineeringStoryStatus,
    SufficiencyLevel,
)
from backend.project_story_ranking import aggregate_project_story_relevance
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    PartialProjectStoryRerankResult,
    ProjectStoryRankingRefreshStatus,
    ProjectStoryRelevanceSnapshot,
    refresh_project_story_ranking,
)
from backend.project_story_ranking_state import (
    MAX_RANKING_STATE_PROJECTS,
    MAX_RANKING_STATE_STORIES,
    MAX_RANKING_STATE_STORIES_PER_PROJECT,
    PROJECT_STORY_RANKING_STATE_POLICY_ID,
    ProjectStoryRankingState,
    ProjectStoryRankingStateError,
    ProjectStoryRankingStateErrorCode,
    build_project_story_ranking_state,
    derive_story_clarification_handoffs,
    update_project_story_ranking_state,
)
from tests.test_project_story_ranking_refresh import (
    PROJECT_A,
    PROJECT_B,
    PROJECT_C,
    _basic_views,
    _clean_state,
    _context,
    _four_story_views,
    _snapshot,
    _updated_basic,
    _view,
)


MODULE_PATH = (
    Path(__file__).parents[1] / "backend" / "project_story_ranking_state.py"
)


def _build_state(profile, views):
    stories, projects, portfolio = _clean_state(profile, views)
    snapshots = ()
    if portfolio is not None:
        snapshots = tuple(
            _snapshot(
                portfolio=portfolio,
                project=projects[ranking.project_id],
                stories=stories[ranking.project_id],
            )
            for ranking in portfolio.ranked_projects
        )
    state = build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=portfolio,
        project_snapshots=snapshots,
    )
    return state, stories, projects


def _complete_views():
    return (
        *_four_story_views(
            affected_text="Updated documentation",
            affected_revision="ranking_state_hidden_four_r1",
        ),
        _view(
            story_key="ranking_state_hidden_five",
            revision_key="ranking_state_hidden_five_r1",
            project_id=PROJECT_A,
            text="Adjusted visual styling",
        ),
        _view(
            story_key="ranking_state_project_b",
            revision_key="ranking_state_project_b_r1",
            project_id=PROJECT_B,
            text="Built reliable backend API systems",
        ),
        _view(
            story_key="ranking_state_project_c",
            revision_key="ranking_state_project_c_r1",
            project_id=PROJECT_C,
            text="Built testing automation",
        ),
    )


def _hidden_story(snapshot, position: int = 4):
    assert len(snapshot.story_relevance) >= position
    return snapshot.story_relevance[position - 1]


def _replace_hidden_revision(snapshot, *, revision_id: str):
    hidden = _hidden_story(snapshot)
    updated = replace(
        hidden,
        current_revision_id=revision_id,
        relevance_id="",
    )
    stories = tuple(
        updated if story is hidden else story
        for story in snapshot.story_relevance
    )
    return replace(snapshot, story_relevance=stories, snapshot_id="")


def _state_with_snapshots(profile, state, snapshots):
    return build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=state.portfolio,
        project_snapshots=snapshots,
    )


def _refresh_state(profile, views, updated):
    state, _stories, _projects = _build_state(profile, views)
    assert state.portfolio is not None
    snapshot = state.snapshot_for_project(updated.project_id)
    assert snapshot is not None
    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=state.portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    return state, result, update_project_story_ranking_state(
        prior_state=state,
        refresh_result=result,
    )


def test_minimal_one_project_state_is_valid() -> None:
    profile = _context()
    view = _view(
        story_key="ranking_state_minimal",
        revision_key="ranking_state_minimal_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )

    state, _stories, _projects = _build_state(profile, (view,))

    assert state.project_count == 1
    assert state.story_count == 1
    assert state.project_snapshots[0].story_relevance[0].canonical_story_id == (
        view.canonical_story_id
    )


def test_multi_project_state_preserves_portfolio_order() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    assert state.portfolio is not None

    assert tuple(item.project_id for item in state.project_snapshots) == tuple(
        item.project_id for item in state.portfolio.ranked_projects
    )


def test_snapshot_input_permutation_normalizes_to_same_state() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    permuted = _state_with_snapshots(
        profile,
        state,
        tuple(reversed(state.project_snapshots)),
    )

    assert permuted == state
    assert permuted.state_id == state.state_id
    assert permuted.state_fingerprint == state.state_fingerprint


def test_missing_snapshot_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        _state_with_snapshots(profile, state, state.project_snapshots[:-1])

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.MISSING_PROJECT_SNAPSHOT
    )


def test_duplicate_snapshot_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        _state_with_snapshots(
            profile,
            state,
            (*state.project_snapshots, state.project_snapshots[0]),
        )

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.DUPLICATE_PROJECT_SNAPSHOT
    )


def test_unknown_snapshot_fails_closed() -> None:
    profile = _context()
    state, stories, projects = _build_state(profile, _complete_views())
    assert state.portfolio is not None
    unknown_project = "ranking_state_unknown_project"
    story = replace(
        stories[PROJECT_B][0],
        project_id=unknown_project,
        relevance_id="",
    )
    project = aggregate_project_story_relevance(
        project_id=unknown_project,
        story_relevance=(story,),
    )
    assert project is not None
    unknown = ProjectStoryRelevanceSnapshot(
        snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
        source_portfolio_id=state.portfolio.portfolio_id,
        project_relevance=project,
        story_relevance=(story,),
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        _state_with_snapshots(
            profile,
            state,
            (*state.project_snapshots, unknown),
        )

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.UNKNOWN_PROJECT_SNAPSHOT
    )
    assert projects[PROJECT_B] is not project


def test_wrong_snapshot_portfolio_binding_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    wrong = replace(
        state.project_snapshots[0],
        source_portfolio_id="ranked_project_story_portfolio_aaaaaaaaaaaaaaaaaaaaaaaa",
        snapshot_id="",
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        _state_with_snapshots(
            profile,
            state,
            (wrong, *state.project_snapshots[1:]),
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.STALE_PORTFOLIO


def test_wrong_hiring_context_id_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    other = _context(title="Data Engineer")

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        build_project_story_ranking_state(
            hiring_context=other,
            portfolio=state.portfolio,
            project_snapshots=state.project_snapshots,
        )

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.MIXED_HIRING_CONTEXT
    )


def test_wrong_hiring_context_fingerprint_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    tampered = replace(profile)
    object.__setattr__(tampered, "fingerprint", "f" * 64)

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        build_project_story_ranking_state(
            hiring_context=tampered,
            portfolio=state.portfolio,
            project_snapshots=state.project_snapshots,
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.INVALID_INPUT


def test_stale_project_relevance_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    stale = state.project_snapshots[0]
    object.__setattr__(stale, "project_relevance", state.project_snapshots[1].project_relevance)

    with pytest.raises(ProjectStoryRankingStateError):
        replace(state, state_id="", state_fingerprint="")


def test_complete_hidden_fourth_and_fifth_stories_are_preserved() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    snapshot = state.snapshot_for_project(PROJECT_A)
    assert snapshot is not None

    assert len(snapshot.story_relevance) == 5
    contributor_ids = {
        item.story_relevance_id
        for item in snapshot.project_relevance.contributions
    }
    assert snapshot.story_relevance[3].relevance_id not in contributor_ids
    assert snapshot.story_relevance[4].relevance_id not in contributor_ids


def test_complete_story_order_is_authoritative_and_deterministic() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    snapshot = state.snapshot_for_project(PROJECT_A)
    assert snapshot is not None

    assert snapshot.story_relevance == tuple(sorted(
        snapshot.story_relevance,
        key=lambda item: (
            -item.total_relevance_score,
            item.canonical_story_id,
            item.current_revision_id,
            item.relevance_id,
        ),
    ))


@pytest.mark.parametrize("hidden_position", (4, 5))
def test_hidden_story_does_not_gain_a_contribution(
    hidden_position: int,
) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    snapshot = state.snapshot_for_project(PROJECT_A)
    assert snapshot is not None
    hidden = _hidden_story(snapshot, hidden_position)

    contribution = next(
        (
            item
            for item in snapshot.project_relevance.contributions
            if item.story_relevance_id == hidden.relevance_id
        ),
        None,
    )

    assert contribution is None


@pytest.mark.parametrize(
    "attribute",
    (
        "canonical_story_id",
        "current_revision_id",
        "relevance_id",
        "total_relevance_score",
        "claim_sufficiency",
        "story_sufficiency",
        "story_opportunity",
    ),
)
def test_story_level_ranking_surface_is_available(attribute: str) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    story = state.project_snapshots[0].story_relevance[0]

    assert getattr(story, attribute) is not None


@pytest.mark.parametrize(
    "attribute",
    (
        "position",
        "project_id",
        "base_relevance",
        "portfolio_relevance",
        "ranking_id",
    ),
)
def test_project_level_ranking_surface_is_available(attribute: str) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    assert state.portfolio is not None

    assert getattr(state.portfolio.ranked_projects[0], attribute) is not None


def test_top_contribution_order_and_weight_are_preserved() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    snapshot = state.snapshot_for_project(PROJECT_A)
    assert snapshot is not None

    assert tuple(
        item.story_relevance
        for item in snapshot.project_relevance.contributions
    ) == snapshot.story_relevance[:3]
    assert all(
        item.weighted_contribution >= 0.0
        for item in snapshot.project_relevance.contributions
    )


def test_state_lookups_expose_rank_and_optional_contribution() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    snapshot = state.snapshot_for_project(PROJECT_A)
    ranking = state.ranking_for_project(PROJECT_A)
    assert snapshot is not None and ranking is not None
    top = snapshot.story_relevance[0]
    hidden = snapshot.story_relevance[3]

    assert state.contribution_for_story(
        project_id=PROJECT_A,
        story_relevance_id=top.relevance_id,
    ) is snapshot.project_relevance.contributions[0]
    assert state.contribution_for_story(
        project_id=PROJECT_A,
        story_relevance_id=hidden.relevance_id,
    ) is None
    assert ranking.position >= 1


def test_state_identity_is_deterministic_across_repeated_execution() -> None:
    profile = _context()
    first, _stories, _projects = _build_state(profile, _complete_views())
    second, _stories, _projects = _build_state(profile, _complete_views())

    assert first == second
    assert first is not second
    assert first.to_dict() == second.to_dict()


def test_runtime_timestamp_is_not_part_of_state_contract() -> None:
    assert {item.name for item in fields(ProjectStoryRankingState)}.isdisjoint({
        "timestamp",
        "created_at",
        "updated_at",
    })


def test_hidden_revision_change_changes_state_identity() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    changed = _replace_hidden_revision(
        target,
        revision_id="engineering_story_revision_aaaaaaaaaaaaaaaaaaaaaaaa",
    )
    snapshots = tuple(
        changed if item.project_id == PROJECT_A else item
        for item in state.project_snapshots
    )
    rebuilt = _state_with_snapshots(profile, state, snapshots)

    assert rebuilt.portfolio == state.portfolio
    assert rebuilt.state_id != state.state_id
    assert rebuilt.state_fingerprint != state.state_fingerprint


def test_hidden_relevance_score_change_changes_state_identity() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    hidden = target.story_relevance[-1]
    changed_story = replace(
        hidden,
        raw_relevance_score=0.01,
        evidence_risk_adjustment=0.0,
        total_relevance_score=0.01,
        relevance_id="",
    )
    changed_snapshot = replace(
        target,
        story_relevance=tuple(
            changed_story if item is hidden else item
            for item in target.story_relevance
        ),
        snapshot_id="",
    )
    snapshots = tuple(
        changed_snapshot if item.project_id == PROJECT_A else item
        for item in state.project_snapshots
    )
    rebuilt = _state_with_snapshots(profile, state, snapshots)

    assert rebuilt.portfolio == state.portfolio
    assert rebuilt.state_id != state.state_id


def test_hidden_change_with_old_state_identity_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    changed = _replace_hidden_revision(
        target,
        revision_id="engineering_story_revision_bbbbbbbbbbbbbbbbbbbbbbbb",
    )
    snapshots = tuple(
        changed if item.project_id == PROJECT_A else item
        for item in state.project_snapshots
    )

    with pytest.raises(ValueError, match="state_fingerprint"):
        ProjectStoryRankingState(
            state_policy_id=state.state_policy_id,
            hiring_context_profile_id=state.hiring_context_profile_id,
            hiring_context_fingerprint=state.hiring_context_fingerprint,
            portfolio=state.portfolio,
            project_snapshots=snapshots,
            state_fingerprint=state.state_fingerprint,
            state_id=state.state_id,
        )


def test_wrong_supplied_state_id_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    with pytest.raises(ValueError, match="state_id"):
        replace(
            state,
            state_fingerprint="",
            state_id="project_story_ranking_state_aaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_conflicting_story_relevance_identity_fails_closed() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    hidden = target.story_relevance[-1]
    tampered = replace(hidden)
    object.__setattr__(
        tampered,
        "relevance_id",
        target.story_relevance[0].relevance_id,
    )

    with pytest.raises(ValueError):
        replace(
            target,
            story_relevance=tuple(
                tampered if item is hidden else item
                for item in target.story_relevance
            ),
            snapshot_id="",
        )


def test_sufficiency_change_does_not_invent_a_second_story_order() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    hidden = target.story_relevance[-1]
    changed = replace(
        hidden,
        claim_sufficiency=SufficiencyLevel.HIGH,
        relevance_id="",
    )
    changed_snapshot = replace(
        target,
        story_relevance=tuple(
            changed if item is hidden else item
            for item in target.story_relevance
        ),
        snapshot_id="",
    )

    assert tuple(
        item.canonical_story_id for item in changed_snapshot.story_relevance
    ) == tuple(
        item.canonical_story_id for item in target.story_relevance
    )
    assert not hasattr(changed, "story_budget")


def test_hidden_story_cannot_be_promoted_without_project_reaggregation() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    target = state.snapshot_for_project(PROJECT_A)
    assert target is not None
    hidden = _hidden_story(target)
    promoted = replace(
        hidden,
        raw_relevance_score=1.0,
        evidence_risk_adjustment=0.0,
        total_relevance_score=1.0,
        relevance_id="",
    )
    stories = tuple(
        promoted if story is hidden else story
        for story in target.story_relevance
    )

    with pytest.raises(ValueError):
        replace(target, story_relevance=stories, snapshot_id="")


def test_state_identity_changes_with_material_full_rerank() -> None:
    profile = _context()
    old_views = _basic_views()
    old, _stories, _projects = _build_state(profile, old_views)
    new, _stories, _projects = _build_state(
        profile,
        (_updated_basic(), old_views[1]),
    )

    assert new.portfolio != old.portfolio
    assert new.state_id != old.state_id


def test_context_generation_changes_state_identity() -> None:
    first_profile = _context()
    second_profile = _context(title="Platform Engineer")
    views = _basic_views()
    first, _stories, _projects = _build_state(first_profile, views)
    second, _stories, _projects = _build_state(second_profile, views)

    assert second.hiring_context_profile_id != first.hiring_context_profile_id
    assert second.state_id != first.state_id


def test_conflicting_canonical_story_across_projects_fails_closed() -> None:
    profile = _context()
    stories, projects, portfolio = _clean_state(profile, _basic_views())
    assert portfolio is not None
    collision = replace(
        stories[PROJECT_B][0],
        canonical_story_id=stories[PROJECT_A][0].canonical_story_id,
        relevance_id="",
    )
    project_b = aggregate_project_story_relevance(
        project_id=PROJECT_B,
        story_relevance=(collision,),
    )
    assert project_b is not None
    from backend.project_portfolio_ranking import rank_project_portfolio

    collision_portfolio = rank_project_portfolio(
        projects=(projects[PROJECT_A], project_b)
    )
    assert collision_portfolio is not None
    snapshots = tuple(
        _snapshot(
            portfolio=collision_portfolio,
            project=(
                projects[PROJECT_A]
                if item.project_id == PROJECT_A
                else project_b
            ),
            stories=(
                stories[PROJECT_A]
                if item.project_id == PROJECT_A
                else (collision,)
            ),
        )
        for item in collision_portfolio.ranked_projects
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        build_project_story_ranking_state(
            hiring_context=profile,
            portfolio=collision_portfolio,
            project_snapshots=snapshots,
        )

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.CONFLICTING_STORY_IDENTITY
    )


def test_conflicting_revision_across_projects_fails_closed() -> None:
    profile = _context()
    stories, projects, portfolio = _clean_state(profile, _basic_views())
    assert portfolio is not None
    collision = replace(
        stories[PROJECT_B][0],
        current_revision_id=stories[PROJECT_A][0].current_revision_id,
        relevance_id="",
    )
    project_b = aggregate_project_story_relevance(
        project_id=PROJECT_B,
        story_relevance=(collision,),
    )
    assert project_b is not None
    from backend.project_portfolio_ranking import rank_project_portfolio

    collision_portfolio = rank_project_portfolio(
        projects=(projects[PROJECT_A], project_b)
    )
    assert collision_portfolio is not None
    snapshots = tuple(
        _snapshot(
            portfolio=collision_portfolio,
            project=(
                projects[PROJECT_A]
                if item.project_id == PROJECT_A
                else project_b
            ),
            stories=(
                stories[PROJECT_A]
                if item.project_id == PROJECT_A
                else (collision,)
            ),
        )
        for item in collision_portfolio.ranked_projects
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        build_project_story_ranking_state(
            hiring_context=profile,
            portfolio=collision_portfolio,
            project_snapshots=snapshots,
        )

    assert captured.value.code is (
        ProjectStoryRankingStateErrorCode.CONFLICTING_STORY_IDENTITY
    )


def test_clarification_handoffs_are_separately_deterministic() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    first = derive_story_clarification_handoffs(ranking_state=state)
    second = derive_story_clarification_handoffs(ranking_state=state)

    assert first == second
    assert state.portfolio is not None
    assert all(item.portfolio_id == state.portfolio.portfolio_id for item in first)


def test_clarification_handoffs_are_not_embedded_in_state_identity() -> None:
    names = {item.name for item in fields(ProjectStoryRankingState)}

    assert "clarification_handoffs" not in names
    assert "updated_handoffs" not in names


def test_empty_state_is_explicit_and_deterministic() -> None:
    profile = _context()
    first = build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=None,
        project_snapshots=(),
    )
    second = build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=None,
        project_snapshots=(),
    )

    assert first == second
    assert first.project_count == 0
    assert first.story_count == 0
    assert derive_story_clarification_handoffs(ranking_state=first) == ()


def test_no_change_refresh_returns_exact_prior_state() -> None:
    profile = _context()
    views = _basic_views()
    state, _stories, _projects = _build_state(profile, views)
    assert state.portfolio is not None
    snapshot = state.snapshot_for_project(PROJECT_A)
    assert snapshot is not None
    result = refresh_project_story_ranking(
        hiring_context=profile,
        prior_portfolio=state.portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=views[0],
    )

    updated = update_project_story_ranking_state(
        prior_state=state,
        refresh_result=result,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.NO_CHANGE
    assert updated is state


def test_reranked_refresh_replaces_only_affected_semantic_snapshot() -> None:
    profile = _context()
    views = _basic_views()
    old, result, new = _refresh_state(profile, views, _updated_basic())
    affected = new.snapshot_for_project(PROJECT_A)
    unaffected = new.snapshot_for_project(PROJECT_B)
    old_unaffected = old.snapshot_for_project(PROJECT_B)

    assert result.status is ProjectStoryRankingRefreshStatus.RERANKED
    assert affected is result.updated_project_snapshot
    assert unaffected is not None and old_unaffected is not None
    assert unaffected.project_relevance is old_unaffected.project_relevance
    assert unaffected.story_relevance[0] is old_unaffected.story_relevance[0]


def test_unaffected_snapshot_wrapper_rebinds_when_portfolio_changes() -> None:
    profile = _context()
    old, _result, new = _refresh_state(
        profile,
        _basic_views(),
        _updated_basic(),
    )
    old_snapshot = old.snapshot_for_project(PROJECT_B)
    new_snapshot = new.snapshot_for_project(PROJECT_B)
    assert old_snapshot is not None and new_snapshot is not None
    assert old.portfolio is not None and new.portfolio is not None

    if old.portfolio.portfolio_id != new.portfolio.portfolio_id:
        assert new_snapshot is not old_snapshot
        assert new_snapshot.source_portfolio_id == new.portfolio.portfolio_id
        assert new_snapshot.project_relevance is old_snapshot.project_relevance
        assert new_snapshot.story_relevance[0] is old_snapshot.story_relevance[0]


def test_hidden_rerank_changes_state_when_portfolio_is_unchanged() -> None:
    profile = _context()
    views = (
        *_four_story_views(
            affected_text="Updated documentation",
            affected_revision="ranking_state_hidden_refresh_r1",
        ),
        _view(
            story_key="ranking_state_hidden_other_project",
            revision_key="ranking_state_hidden_other_project_r1",
            project_id=PROJECT_B,
            text="Built reliable backend API systems",
        ),
    )
    updated = _view(
        story_key="story_affected",
        revision_key="ranking_state_hidden_refresh_r2",
        project_id=PROJECT_A,
        text="Adjusted visual styling",
    )
    old, result, new = _refresh_state(profile, views, updated)

    assert result.status is ProjectStoryRankingRefreshStatus.RERANKED
    assert new.state_id != old.state_id
    assert new.portfolio == old.portfolio
    assert new.snapshot_for_project(PROJECT_B) is old.snapshot_for_project(PROJECT_B)


def test_story_removed_refresh_removes_stale_story_candidate() -> None:
    profile = _context()
    affected = _view(
        story_key="ranking_state_remove",
        revision_key="ranking_state_remove_r1",
        project_id=PROJECT_A,
        text="Implemented validation",
    )
    retained = _view(
        story_key="ranking_state_retained",
        revision_key="ranking_state_retained_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )
    invalidated = _view(
        story_key="ranking_state_remove",
        revision_key="ranking_state_remove_r2",
        project_id=PROJECT_A,
        text="Implemented validation",
        status=EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )
    _old, result, new = _refresh_state(
        profile,
        (affected, retained),
        invalidated,
    )
    snapshot = new.snapshot_for_project(PROJECT_A)
    assert snapshot is not None

    assert result.status is ProjectStoryRankingRefreshStatus.STORY_REMOVED
    assert affected.canonical_story_id not in {
        item.canonical_story_id for item in snapshot.story_relevance
    }


def test_project_removed_refresh_removes_snapshot_and_handoff() -> None:
    profile = _context()
    affected, other = _basic_views()
    invalidated = _view(
        story_key="affected",
        revision_key="ranking_state_project_remove_r2",
        project_id=PROJECT_A,
        text="Implemented validation",
        status=EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )
    _old, result, new = _refresh_state(
        profile,
        (affected, other),
        invalidated,
    )

    assert result.status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED
    assert new.snapshot_for_project(PROJECT_A) is None
    assert all(
        item.project_id != PROJECT_A
        for item in derive_story_clarification_handoffs(ranking_state=new)
    )


def test_last_project_removal_produces_explicit_empty_state() -> None:
    profile = _context()
    affected = _view(
        story_key="ranking_state_last",
        revision_key="ranking_state_last_r1",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
    )
    invalidated = _view(
        story_key="ranking_state_last",
        revision_key="ranking_state_last_r2",
        project_id=PROJECT_A,
        text="Built reliable backend API systems",
        status=EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )
    old, result, new = _refresh_state(profile, (affected,), invalidated)

    assert result.status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED
    assert new.portfolio is None
    assert new.project_snapshots == ()
    assert new.state_id != old.state_id


def test_partial_refresh_state_equals_clean_full_final_state() -> None:
    profile = _context()
    old_views = _basic_views()
    updated = _updated_basic()
    _old, _result, partial = _refresh_state(profile, old_views, updated)
    clean, _stories, _projects = _build_state(
        profile,
        (updated, old_views[1]),
    )

    assert partial == clean
    assert partial.to_dict() == clean.to_dict()


def test_refresh_result_replay_against_another_state_fails_closed() -> None:
    profile = _context()
    views = _basic_views()
    old, result, _new = _refresh_state(profile, views, _updated_basic())
    other_profile = _context(title="Data Platform Engineer")
    other, _stories, _projects = _build_state(other_profile, views)
    assert old.state_id != other.state_id

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        update_project_story_ranking_state(
            prior_state=other,
            refresh_result=result,
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH


def test_refresh_with_stale_handoff_projection_fails_closed() -> None:
    profile = _context()
    state, result, _new = _refresh_state(
        profile,
        _basic_views(),
        _updated_basic(),
    )
    assert result.updated_handoffs
    stale = replace(
        result,
        updated_handoffs=result.updated_handoffs[:-1],
        result_id="",
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        update_project_story_ranking_state(
            prior_state=state,
            refresh_result=stale,
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH


def test_refresh_cannot_change_an_unrelated_hidden_story() -> None:
    profile = _context()
    views = (
        *_four_story_views(
            affected_text="Updated documentation",
            affected_revision="ranking_state_attack_affected_r1",
        ),
        _view(
            story_key="ranking_state_attack_unrelated_hidden",
            revision_key="ranking_state_attack_unrelated_hidden_r1",
            project_id=PROJECT_A,
            text="Adjusted documentation formatting",
        ),
    )
    updated = _view(
        story_key="story_affected",
        revision_key="ranking_state_attack_affected_r2",
        project_id=PROJECT_A,
        text="Adjusted visual styling",
    )
    state, result, _new = _refresh_state(profile, views, updated)
    snapshot = result.updated_project_snapshot
    assert snapshot is not None
    unrelated = next(
        item
        for item in snapshot.story_relevance
        if item.canonical_story_id != updated.canonical_story_id
        and item.relevance_id not in {
            contribution.story_relevance_id
            for contribution in snapshot.project_relevance.contributions
        }
    )
    changed = replace(
        unrelated,
        current_revision_id="engineering_story_revision_cccccccccccccccccccccccc",
        relevance_id="",
    )
    attacked_snapshot = replace(
        snapshot,
        story_relevance=tuple(
            changed if item is unrelated else item
            for item in snapshot.story_relevance
        ),
        snapshot_id="",
    )
    attacked_delta = replace(
        result.delta,
        new_snapshot_id=attacked_snapshot.snapshot_id,
        delta_id="",
    )
    attacked_result = replace(
        result,
        updated_project_snapshot=attacked_snapshot,
        delta=attacked_delta,
        result_id="",
    )

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        update_project_story_ranking_state(
            prior_state=state,
            refresh_result=attacked_result,
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH


def test_state_and_nested_inputs_are_not_mutated() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    before_profile = profile.to_dict()
    before_state = state.to_dict()

    rebuilt = build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=state.portfolio,
        project_snapshots=state.project_snapshots,
    )

    assert profile.to_dict() == before_profile
    assert state.to_dict() == before_state
    assert rebuilt == state


@pytest.mark.parametrize(
    "mutation",
    (
        lambda state: setattr(state, "state_id", "changed"),
        lambda state: setattr(state, "portfolio", None),
        lambda state: setattr(state, "project_snapshots", ()),
    ),
)
def test_ranking_state_is_immutable(mutation) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())

    with pytest.raises(FrozenInstanceError):
        mutation(state)


def test_project_input_bound_is_enforced_before_duplicate_validation() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        build_project_story_ranking_state(
            hiring_context=profile,
            portfolio=state.portfolio,
            project_snapshots=(
                state.project_snapshots[0],
            ) * (MAX_RANKING_STATE_PROJECTS + 1),
        )

    assert captured.value.code is ProjectStoryRankingStateErrorCode.BOUND_EXCEEDED


def test_total_story_bound_is_explicitly_enforced(monkeypatch) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())
    monkeypatch.setattr(state_module, "MAX_RANKING_STATE_STORIES", 1)

    with pytest.raises(ProjectStoryRankingStateError) as captured:
        replace(state, state_id="", state_fingerprint="")

    assert captured.value.code is ProjectStoryRankingStateErrorCode.BOUND_EXCEEDED


def test_story_snapshot_bound_is_enforced_before_deduplication() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())
    snapshot = state.project_snapshots[0]

    with pytest.raises(ValueError):
        replace(
            snapshot,
            story_relevance=(snapshot.story_relevance[0],)
            * (MAX_RANKING_STATE_STORIES_PER_PROJECT + 1),
            snapshot_id="",
        )


def test_bounds_reuse_accepted_ranking_limits() -> None:
    assert MAX_RANKING_STATE_PROJECTS == 64
    assert MAX_RANKING_STATE_STORIES_PER_PROJECT == 512
    assert MAX_RANKING_STATE_STORIES == 64 * 512


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "project_budget",
        "story_budget",
        "bullet_budget",
        "bullet_count",
        "line_count",
        "page_budget",
        "selected_for_resume",
        "selected_project_count",
        "selected_story_count",
        "should_ask_user",
        "question_text",
        "clarification_priority",
        "user_friction",
        "searchable",
        "human_only",
        "candidate_identity",
        "candidate_persona",
        "raw_job_description",
        "raw_story_prose",
        "technology_inference",
    ),
)
def test_state_models_do_not_own_downstream_or_truth_fields(
    forbidden_field: str,
) -> None:
    names = {item.name for item in fields(ProjectStoryRankingState)}

    assert forbidden_field not in names


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "evaluate_engineering_story_relevance",
        "rank_engineering_stories_for_hiring_context",
        "aggregate_project_story_relevance",
        "rank_project_portfolio",
        "rank_projects_for_resume",
        "select_staged_projects_with_ranking",
        "tailor_resume_staged",
        "project_bullet_budget",
        "load_engineering_story_memory",
        "build_authoritative_engineering_story_memory",
        "persist_engineering_story_memory",
        "reconstruct_engineering_story",
        "retrieve_project_evidence",
        "search_project_evidence",
        "load_project_evidence_memory",
        "load_project_capability_memory",
        "open",
        "getenv",
        "putenv",
        "environ",
    ),
)
def test_state_builder_and_updater_do_not_call_forbidden_services(
    forbidden_call: str,
) -> None:
    sources = "\n".join((
        inspect.getsource(build_project_story_ranking_state),
        inspect.getsource(update_project_story_ranking_state),
    ))
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(sources))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert forbidden_call not in calls


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
        "typing",
        "backend.hiring_context_models",
        "backend.project_portfolio_ranking",
        "backend.project_story_ranking",
        "backend.project_story_ranking_refresh",
        "backend.story_clarification_handoff",
    }

    assert modules <= allowed


@pytest.mark.parametrize(
    "forbidden_dependency",
    (
        "chromadb",
        "requests",
        "httpx",
        "openai",
        "pathlib",
        "sqlite3",
        "subprocess",
        "os",
        "backend.api_server",
        "backend.project_retrieval_v2",
        "backend.engineering_story_memory_service",
    ),
)
def test_production_module_has_no_io_or_runtime_dependency(
    forbidden_dependency: str,
) -> None:
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

    assert forbidden_dependency not in modules


def test_module_has_no_numbered_stage_terminology() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert not __import__("re").search(r"(?i)phase[ _-]?[0-9]+", source)


def test_state_identity_covers_complete_snapshot_ids() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())

    assert all(
        snapshot.snapshot_id in inspect.getsource(ProjectStoryRankingState)
        or snapshot.snapshot_id in str(state.to_dict())
        for snapshot in state.project_snapshots
    )


def test_state_is_ready_for_later_partial_rebudget_without_budget_logic() -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _complete_views())
    names = {item.name for item in fields(ProjectStoryRankingState)}

    assert {
        "hiring_context_profile_id",
        "hiring_context_fingerprint",
        "portfolio",
        "project_snapshots",
        "state_fingerprint",
        "state_id",
    } <= names
    assert names.isdisjoint({
        "project_budget",
        "story_budget",
        "bullet_budget",
    })
    assert state.story_count >= 5


@pytest.mark.parametrize(
    "sufficiency",
    tuple(SufficiencyLevel),
)
def test_sufficiency_is_preserved_as_input_not_budget_decision(
    sufficiency: SufficiencyLevel,
) -> None:
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())
    story = state.project_snapshots[0].story_relevance[0]
    changed = replace(
        story,
        claim_sufficiency=sufficiency,
        relevance_id="",
    )

    assert changed.claim_sufficiency is sufficiency
    assert not hasattr(changed, "bullet_budget")
