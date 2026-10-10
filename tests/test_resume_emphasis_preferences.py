"""Contract tests for the non-authoritative resume preference overlay."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import backend.resume_emphasis_preferences as preference_module
from backend.resume_emphasis_preferences import (
    EmphasisPreferenceAction,
    MAX_RESUME_PROJECT_PREFERENCES,
    ProjectEmphasisPreference,
    ResumeEmphasisPreferenceError,
    ResumeEmphasisPreferenceErrorCode,
    ResumeEmphasisPreferenceOverlay,
    ResumeEmphasisReconciliationStatus,
    StoryEmphasisPreference,
    build_resume_emphasis_preference_overlay,
    project_preference_action,
    reconcile_resume_emphasis_preferences,
    reset_all_preferences,
    reset_project_preference,
    reset_story_preference,
    story_preference_action,
    validate_overlay_against_ranking_state,
)
from tests.test_project_story_ranking_refresh import (
    PROJECT_A,
    PROJECT_B,
    _basic_views,
    _context,
    _updated_basic,
    _view,
)
from tests.test_project_story_ranking_state import _build_state, _refresh_state


def _state():
    profile = _context()
    state, _stories, _projects = _build_state(profile, _basic_views())
    return profile, state


def _identities(state):
    first_project = state.project_snapshots[0]
    first_story = first_project.story_relevance[0]
    return first_project.project_id, first_story.canonical_story_id, first_story.current_revision_id


def _project(state, action, project_id=None):
    target = project_id or state.project_snapshots[0].project_id
    return ProjectEmphasisPreference(target, action)


def _story(state, action, project_id=None, canonical_story_id=None, revision_id=None):
    project, story, revision = _identities(state)
    return StoryEmphasisPreference(
        project_id or project,
        canonical_story_id or story,
        revision_id or revision,
        action,
    )


def _overlay(state, *, projects=(), stories=()):
    return build_resume_emphasis_preference_overlay(
        ranking_state=state,
        project_preferences=projects,
        story_preferences=stories,
    )


def test_empty_overlay_means_all_preferences_default():
    _profile, state = _state()
    overlay = _overlay(state)
    project_id, story_id, _revision = _identities(state)
    assert overlay.project_preferences == ()
    assert overlay.story_preferences == ()
    assert project_preference_action(overlay, project_id) is EmphasisPreferenceAction.DEFAULT
    assert story_preference_action(
        overlay, project_id=project_id, canonical_story_id=story_id
    ) is EmphasisPreferenceAction.DEFAULT


@pytest.mark.parametrize(
    "action",
    [
        EmphasisPreferenceAction.EMPHASIZE_MORE,
        EmphasisPreferenceAction.EMPHASIZE_LESS,
        EmphasisPreferenceAction.KEEP,
        EmphasisPreferenceAction.EXCLUDE,
    ],
)
def test_each_explicit_project_action_is_preserved(action):
    _profile, state = _state()
    overlay = _overlay(state, projects=(_project(state, action),))
    assert overlay.project_preferences[0].action is action


@pytest.mark.parametrize(
    "action",
    [
        EmphasisPreferenceAction.EMPHASIZE_MORE,
        EmphasisPreferenceAction.EMPHASIZE_LESS,
        EmphasisPreferenceAction.KEEP,
        EmphasisPreferenceAction.EXCLUDE,
    ],
)
def test_each_explicit_story_action_is_preserved(action):
    _profile, state = _state()
    overlay = _overlay(state, stories=(_story(state, action),))
    assert overlay.story_preferences[0].action is action


def test_default_project_action_is_not_stored():
    _profile, state = _state()
    assert _overlay(
        state,
        projects=(_project(state, EmphasisPreferenceAction.DEFAULT),),
    ).project_preferences == ()


def test_default_story_action_is_not_stored():
    _profile, state = _state()
    assert _overlay(
        state,
        stories=(_story(state, EmphasisPreferenceAction.DEFAULT),),
    ).story_preferences == ()


def test_project_identity_is_exact_not_casefolded():
    _profile, state = _state()
    project_id, _story_id, _revision = _identities(state)
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, projects=(_project(state, "keep", project_id.upper()),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.UNKNOWN_PROJECT


def test_project_identity_rejects_whitespace_normalization():
    with pytest.raises(ValueError, match="exact"):
        ProjectEmphasisPreference(" project ", "keep")


def test_canonical_story_identity_is_exact_not_casefolded():
    _profile, state = _state()
    project_id, story_id, revision_id = _identities(state)
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, stories=(_story(
            state,
            "keep",
            project_id=project_id,
            canonical_story_id=story_id.upper(),
            revision_id=revision_id,
        ),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.UNKNOWN_STORY


def test_unknown_project_fails_closed():
    _profile, state = _state()
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, projects=(ProjectEmphasisPreference("unknown_project", "keep"),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.UNKNOWN_PROJECT


def test_unknown_story_fails_closed():
    _profile, state = _state()
    project_id, _story_id, revision_id = _identities(state)
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, stories=(StoryEmphasisPreference(
            project_id,
            "unknown_story",
            revision_id,
            "keep",
        ),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.UNKNOWN_STORY


def test_story_owned_by_another_project_fails_closed():
    _profile, state = _state()
    snapshots = {item.project_id: item for item in state.project_snapshots}
    story = snapshots[PROJECT_A].story_relevance[0]
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, stories=(StoryEmphasisPreference(
            PROJECT_B,
            story.canonical_story_id,
            story.current_revision_id,
            "keep",
        ),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.UNKNOWN_STORY


def test_stale_story_revision_fails_exact_build_validation():
    _profile, state = _state()
    project_id, story_id, _revision_id = _identities(state)
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, stories=(StoryEmphasisPreference(
            project_id,
            story_id,
            "engineering_story_revision_aaaaaaaaaaaaaaaaaaaaaaaa",
            "keep",
        ),))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.STALE_RANKING_STATE


def test_unknown_action_fails_closed():
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        ProjectEmphasisPreference(PROJECT_A, "boost")
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.INVALID_INPUT


def test_duplicate_equal_project_preference_deduplicates():
    _profile, state = _state()
    item = _project(state, "keep")
    assert len(_overlay(state, projects=(item, item)).project_preferences) == 1


def test_duplicate_equal_story_preference_deduplicates():
    _profile, state = _state()
    item = _story(state, "keep")
    assert len(_overlay(state, stories=(item, item)).story_preferences) == 1


def test_conflicting_project_duplicate_fails_closed():
    _profile, state = _state()
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, projects=(
            _project(state, "keep"),
            _project(state, "exclude"),
        ))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.CONFLICTING_DUPLICATE


def test_conflicting_story_duplicate_fails_closed():
    _profile, state = _state()
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, stories=(
            _story(state, "keep"),
            _story(state, "exclude"),
        ))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.CONFLICTING_DUPLICATE


def test_preference_input_order_does_not_change_overlay_identity():
    _profile, state = _state()
    projects = tuple(_project(state, "keep", snapshot.project_id) for snapshot in state.project_snapshots)
    left = _overlay(state, projects=projects)
    right = _overlay(state, projects=tuple(reversed(projects)))
    assert left == right
    assert left.overlay_id == right.overlay_id


def test_equivalent_overlay_is_deterministic():
    _profile, state = _state()
    first = _overlay(state, stories=(_story(state, "emphasize_more"),))
    second = _overlay(state, stories=(_story(state, EmphasisPreferenceAction.EMPHASIZE_MORE),))
    assert first.overlay_fingerprint == second.overlay_fingerprint


def test_material_preference_change_changes_identity():
    _profile, state = _state()
    keep = _overlay(state, projects=(_project(state, "keep"),))
    exclude = _overlay(state, projects=(_project(state, "exclude"),))
    assert keep.overlay_id != exclude.overlay_id
    assert keep.overlay_fingerprint != exclude.overlay_fingerprint


def test_overlay_identity_contains_no_runtime_timestamp():
    _profile, state = _state()
    first = _overlay(state)
    second = _overlay(state)
    assert first.overlay_id == second.overlay_id
    assert "timestamp" not in first.to_dict()


def test_project_exclude_clears_child_story_preferences():
    _profile, state = _state()
    project_id, _story_id, _revision = _identities(state)
    overlay = _overlay(
        state,
        projects=(_project(state, "exclude", project_id),),
        stories=(_story(state, "emphasize_more"),),
    )
    assert overlay.story_preferences == ()


def test_project_exclude_is_effective_for_child_story():
    _profile, state = _state()
    project_id, story_id, _revision = _identities(state)
    overlay = _overlay(state, projects=(_project(state, "exclude"),))
    assert story_preference_action(
        overlay, project_id=project_id, canonical_story_id=story_id
    ) is EmphasisPreferenceAction.EXCLUDE


def test_project_lower_with_child_emphasize_more_is_allowed():
    _profile, state = _state()
    project_id, story_id, _revision = _identities(state)
    overlay = _overlay(
        state,
        projects=(_project(state, "emphasize_less"),),
        stories=(_story(state, "emphasize_more"),),
    )
    assert project_preference_action(overlay, project_id) is EmphasisPreferenceAction.EMPHASIZE_LESS
    assert story_preference_action(
        overlay, project_id=project_id, canonical_story_id=story_id
    ) is EmphasisPreferenceAction.EMPHASIZE_MORE


def test_reset_project_preference_returns_project_to_default():
    _profile, state = _state()
    project_id, _story_id, _revision = _identities(state)
    overlay = _overlay(state, projects=(_project(state, "keep"),))
    reset = reset_project_preference(overlay=overlay, ranking_state=state, project_id=project_id)
    assert project_preference_action(reset, project_id) is EmphasisPreferenceAction.DEFAULT


def test_reset_project_preference_preserves_compatible_story_choice():
    _profile, state = _state()
    project_id, story_id, _revision = _identities(state)
    overlay = _overlay(
        state,
        projects=(_project(state, "emphasize_less"),),
        stories=(_story(state, "emphasize_more"),),
    )
    reset = reset_project_preference(overlay=overlay, ranking_state=state, project_id=project_id)
    assert story_preference_action(
        reset, project_id=project_id, canonical_story_id=story_id
    ) is EmphasisPreferenceAction.EMPHASIZE_MORE


def test_reset_story_preference_returns_story_to_default():
    _profile, state = _state()
    project_id, story_id, _revision = _identities(state)
    overlay = _overlay(state, stories=(_story(state, "keep"),))
    reset = reset_story_preference(
        overlay=overlay,
        ranking_state=state,
        project_id=project_id,
        canonical_story_id=story_id,
    )
    assert story_preference_action(
        reset, project_id=project_id, canonical_story_id=story_id
    ) is EmphasisPreferenceAction.DEFAULT


def test_reset_all_clears_only_preferences():
    _profile, state = _state()
    before = state.to_dict()
    overlay = _overlay(state, projects=(_project(state, "keep"),))
    reset = reset_all_preferences(overlay=overlay, ranking_state=state)
    assert reset.project_preferences == reset.story_preferences == ()
    assert state.to_dict() == before


def test_exact_state_validation_accepts_current_overlay():
    _profile, state = _state()
    validate_overlay_against_ranking_state(overlay=_overlay(state), ranking_state=state)


def test_exact_state_validation_rejects_unrelated_generation():
    profile, state = _state()
    extra = _view(
        story_key="overlay_extra",
        revision_key="overlay_extra_r1",
        project_id=PROJECT_A,
        text="Built validation automation",
    )
    other, _stories, _projects = _build_state(profile, (*_basic_views(), extra))
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        validate_overlay_against_ranking_state(overlay=_overlay(state), ranking_state=other)
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.STALE_RANKING_STATE


def test_context_change_marks_overlay_stale_without_rebinding():
    _profile, state = _state()
    changed_profile = _context(title="Game Engineer", required=("Build real-time game systems",))
    changed_state, _stories, _projects = _build_state(changed_profile, _basic_views())
    result = reconcile_resume_emphasis_preferences(
        overlay=_overlay(state, projects=(_project(state, "keep"),)),
        ranking_state=changed_state,
    )
    assert result.status is ResumeEmphasisReconciliationStatus.CONTEXT_STALE
    assert result.overlay is None


def test_exact_validation_rejects_context_mismatch():
    _profile, state = _state()
    changed_profile = _context(title="Game Engineer", required=("Build real-time game systems",))
    changed_state, _stories, _projects = _build_state(changed_profile, _basic_views())
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        validate_overlay_against_ranking_state(overlay=_overlay(state), ranking_state=changed_state)
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.CONTEXT_MISMATCH


def test_current_reconciliation_returns_same_overlay():
    _profile, state = _state()
    overlay = _overlay(state, projects=(_project(state, "keep"),))
    result = reconcile_resume_emphasis_preferences(overlay=overlay, ranking_state=state)
    assert result.status is ResumeEmphasisReconciliationStatus.CURRENT
    assert result.overlay is overlay


def test_canonical_story_preference_survives_revision_advance():
    profile = _context()
    old_state, _refresh, new_state = _refresh_state(profile, _basic_views(), _updated_basic())
    old_story = old_state.snapshot_for_project(PROJECT_A).story_relevance[0]
    overlay = _overlay(old_state, stories=(StoryEmphasisPreference(
        PROJECT_A,
        old_story.canonical_story_id,
        old_story.current_revision_id,
        "emphasize_more",
    ),))
    result = reconcile_resume_emphasis_preferences(overlay=overlay, ranking_state=new_state)
    assert result.status is ResumeEmphasisReconciliationStatus.RECONCILED
    assert result.overlay.story_preferences[0].canonical_story_id == old_story.canonical_story_id
    assert result.overlay.story_preferences[0].current_revision_id == _updated_basic().current_revision_id


def test_revision_reconciliation_does_not_treat_preference_as_claim_proof():
    profile = _context()
    old_state, _refresh, new_state = _refresh_state(profile, _basic_views(), _updated_basic())
    old_story = old_state.snapshot_for_project(PROJECT_A).story_relevance[0]
    before = new_state.to_dict()
    reconcile_resume_emphasis_preferences(
        overlay=_overlay(old_state, stories=(StoryEmphasisPreference(
            PROJECT_A,
            old_story.canonical_story_id,
            old_story.current_revision_id,
            "keep",
        ),)),
        ranking_state=new_state,
    )
    assert new_state.to_dict() == before


def test_removed_story_preference_is_dropped_without_ghost_reference():
    profile = _context()
    old, other = _basic_views()
    removed = _view(
        story_key="overlay_removed_story",
        revision_key="overlay_removed_story_r1",
        project_id=PROJECT_A,
        text="Built testing automation",
    )
    old_state, _stories, _projects = _build_state(profile, (old, removed, other))
    new_state, _stories, _projects = _build_state(profile, (old, other))
    overlay = _overlay(old_state, stories=(StoryEmphasisPreference(
        removed.project_id,
        removed.canonical_story_id,
        removed.current_revision_id,
        "exclude",
    ),))
    result = reconcile_resume_emphasis_preferences(overlay=overlay, ranking_state=new_state)
    assert result.overlay.story_preferences == ()
    assert result.removed_story_ids == ((removed.project_id, removed.canonical_story_id),)


def test_removed_project_preference_is_dropped_without_ghost_reference():
    profile = _context()
    old, other = _basic_views()
    old_state, _stories, _projects = _build_state(profile, (old, other))
    new_state, _stories, _projects = _build_state(profile, (other,))
    overlay = _overlay(old_state, projects=(ProjectEmphasisPreference(PROJECT_A, "exclude"),))
    result = reconcile_resume_emphasis_preferences(overlay=overlay, ranking_state=new_state)
    assert result.overlay.project_preferences == ()
    assert result.removed_project_ids == (PROJECT_A,)


def test_project_preference_input_bound_is_enforced_before_deduplication():
    _profile, state = _state()
    item = _project(state, "keep")
    with pytest.raises(ResumeEmphasisPreferenceError) as captured:
        _overlay(state, projects=tuple(item for _ in range(MAX_RESUME_PROJECT_PREFERENCES + 1)))
    assert captured.value.code is ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED


def test_overlay_is_frozen():
    _profile, state = _state()
    overlay = _overlay(state)
    with pytest.raises(FrozenInstanceError):
        overlay.overlay_id = "changed"


def test_project_preference_is_frozen():
    item = ProjectEmphasisPreference(PROJECT_A, "keep")
    with pytest.raises(FrozenInstanceError):
        item.action = EmphasisPreferenceAction.EXCLUDE


def test_story_preference_is_frozen():
    item = StoryEmphasisPreference(PROJECT_A, "story", "revision", "keep")
    with pytest.raises(FrozenInstanceError):
        item.action = EmphasisPreferenceAction.EXCLUDE


def test_overlay_does_not_mutate_base_ranking_state():
    _profile, state = _state()
    before = state.to_dict()
    _overlay(state, projects=(_project(state, "emphasize_more"),))
    assert state.to_dict() == before


def test_overlay_does_not_mutate_project_relevance():
    _profile, state = _state()
    project = state.portfolio.ranked_projects[0].project_relevance
    before = project.to_dict()
    _overlay(state, projects=(_project(state, "emphasize_less"),))
    assert project.to_dict() == before


def test_overlay_does_not_mutate_story_relevance():
    _profile, state = _state()
    story = state.project_snapshots[0].story_relevance[0]
    before = story.to_dict()
    _overlay(state, stories=(_story(state, "exclude"),))
    assert story.to_dict() == before


def test_overlay_does_not_change_any_ranking_score():
    _profile, state = _state()
    before = [item.total_relevance_score for item in state.project_snapshots[0].story_relevance]
    _overlay(state, stories=(_story(state, "emphasize_more"),))
    assert [item.total_relevance_score for item in state.project_snapshots[0].story_relevance] == before


def test_overlay_does_not_change_claim_sufficiency():
    _profile, state = _state()
    story = state.project_snapshots[0].story_relevance[0]
    before = story.claim_sufficiency
    _overlay(state, stories=(_story(state, "keep"),))
    assert story.claim_sufficiency == before


def test_overlay_does_not_change_story_sufficiency():
    _profile, state = _state()
    story = state.project_snapshots[0].story_relevance[0]
    before = story.story_sufficiency
    _overlay(state, stories=(_story(state, "keep"),))
    assert story.story_sufficiency == before


def test_overlay_does_not_change_story_opportunity():
    _profile, state = _state()
    story = state.project_snapshots[0].story_relevance[0]
    before = story.story_opportunity
    _overlay(state, stories=(_story(state, "keep"),))
    assert story.story_opportunity == before


def test_candidate_evidence_and_engineering_story_are_not_overlay_inputs():
    overlay_fields = {item.name for item in fields(ResumeEmphasisPreferenceOverlay)}
    assert not overlay_fields & {"candidate_evidence", "engineering_story", "evidence_facts"}


@pytest.mark.parametrize(
    "forbidden",
    [
        "candidate_identity",
        "candidate_technologies",
        "evidence_strength",
        "relevance_score",
        "claim_sufficiency",
        "story_sufficiency",
        "story_opportunity",
        "preference_score",
        "adjusted_rank",
        "project_budget",
        "story_budget",
        "bullet_count",
        "should_ask_user",
        "selected_for_resume",
    ],
)
def test_overlay_contract_has_no_truth_ranking_budget_or_question_field(forbidden):
    assert forbidden not in {item.name for item in fields(ResumeEmphasisPreferenceOverlay)}


def test_module_has_no_io_persistence_retrieval_or_generation_dependency():
    source = Path(preference_module.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "localstorage",
        "sqlite",
        "chroma",
        "openai",
        "requests.",
        "write_authoritative_engineering_story_memory",
        "tailor_resume_staged",
        "rank_projects_for_resume",
        "project_bullet_budget",
    ):
        assert forbidden not in source
