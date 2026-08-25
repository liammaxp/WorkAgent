from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import itertools
from pathlib import Path

import pytest

from backend.engineering_story_models import (
    EngineeringStoryStatus,
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    StoryRelevanceComponents,
    StoryRelevanceWeights,
)
from backend.project_story_ranking import (
    DEPTH_QUALIFICATION_FLOOR,
    MAX_CONTRIBUTING_STORIES,
    MAX_PROJECT_RELEVANCE_REASONS,
    MAX_PROJECT_STORY_INPUTS,
    PROJECT_RELEVANCE_AGGREGATION_POLICY_ID,
    PROJECT_STORY_POSITION_WEIGHTS,
    ProjectHiringRelevance,
    ProjectRelevanceAggregationError,
    ProjectRelevanceAggregationErrorCode,
    ProjectRelevanceComponents,
    ProjectRelevanceReason,
    ProjectStoryContribution,
    aggregate_project_story_relevance,
    rank_projects_from_story_relevance,
)


MODULE_PATH = Path(__file__).parents[1] / "backend" / "project_story_ranking.py"
CONTEXT_ID = "hiring_context_aaaaaaaaaaaaaaaaaaaaaaaa"
CONTEXT_FINGERPRINT = "a" * 64


def _story(
    score: float,
    *,
    project_id: str = "project_a",
    story_id: str = "story_a",
    revision_id: str | None = None,
    context_id: str = CONTEXT_ID,
    context_fingerprint: str = CONTEXT_FINGERPRINT,
    lifecycle: EngineeringStoryStatus = EngineeringStoryStatus.ACTIVE,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    risk_adjustment: float = 0.0,
    components: StoryRelevanceComponents | None = None,
) -> StoryHiringRelevance:
    revision = revision_id or f"revision_{project_id}_{story_id}"
    accepted_components = components or StoryRelevanceComponents(
        explicit_jd_relevance=0.91,
        role_family_relevance=0.72,
        organization_domain_relevance=0.63,
        transferable_engineering_relevance=0.84,
        evidence_claim_safety=1.0,
        story_completeness=1.0,
    )
    return StoryHiringRelevance(
        project_id=project_id,
        canonical_story_id=story_id,
        current_revision_id=revision,
        hiring_context_profile_id=context_id,
        hiring_context_fingerprint=context_fingerprint,
        story_provenance_fingerprint=f"provenance_{project_id}_{story_id}",
        lifecycle_status=lifecycle,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        story_opportunity=story_opportunity,
        components=accepted_components,
        weights=StoryRelevanceWeights(
            explicit_jd=0.38,
            role_family=0.22,
            organization_domain=0.20,
            transferable_engineering=0.20,
        ),
        raw_relevance_score=score + risk_adjustment,
        evidence_risk_adjustment=risk_adjustment,
        total_relevance_score=score,
        clarification_value_hint=(
            0.0
            if story_opportunity is StoryOpportunityLevel.NONE
            else 0.67
        ),
        semantic_features=(),
        reasons=(),
        hiring_context_source_refs=(),
    )


def _aggregate(*stories: StoryHiringRelevance) -> ProjectHiringRelevance:
    result = aggregate_project_story_relevance(
        project_id=stories[0].project_id,
        story_relevance=stories,
    )
    assert result is not None
    return result


@pytest.mark.parametrize(
    ("score", "expected"),
    ((0.95, 0.95), (0.90, 0.90), (0.10, 0.10), (0.0, 0.0)),
)
def test_one_story_keeps_its_full_accepted_relevance(
    score: float,
    expected: float,
) -> None:
    result = _aggregate(_story(score))

    assert result.aggregate_relevance_score == expected
    assert result.components.strongest_story_contribution == expected
    assert result.components.secondary_story_depth == 0.0


def test_strong_story_is_not_penalized_by_weak_stories() -> None:
    result = _aggregate(
        _story(0.95, story_id="story_a"),
        _story(0.30, story_id="story_b"),
        _story(0.20, story_id="story_c"),
    )

    assert result.aggregate_relevance_score == 0.95
    assert [item.weighted_contribution for item in result.contributions] == [
        0.95,
        0.0,
        0.0,
    ]


def test_multiple_strong_stories_expose_more_depth_than_one_exceptional_story() -> None:
    exceptional = _aggregate(
        _story(0.95, story_id="story_a"),
        _story(0.30, story_id="story_b"),
        _story(0.20, story_id="story_c"),
    )
    deep = _aggregate(
        _story(0.82, story_id="story_a"),
        _story(0.80, story_id="story_b"),
        _story(0.78, story_id="story_c"),
    )

    assert exceptional.aggregate_relevance_score == 0.95
    assert deep.aggregate_relevance_score == 0.897702
    assert deep.components.secondary_story_depth > (
        exceptional.components.secondary_story_depth
    )


def test_two_strong_stories_can_beat_one_moderately_stronger_story() -> None:
    one_story = _aggregate(_story(0.90))
    two_stories = _aggregate(
        _story(0.84, story_id="story_a"),
        _story(0.83, story_id="story_b"),
    )

    assert two_stories.aggregate_relevance_score == 0.90336
    assert two_stories.aggregate_relevance_score > one_story.aggregate_relevance_score
    assert two_stories.aggregate_relevance_score > max(
        item.story_total_relevance for item in two_stories.contributions
    )


def test_third_strong_story_adds_diminishing_value() -> None:
    two = _aggregate(
        _story(0.84, story_id="story_a"),
        _story(0.83, story_id="story_b"),
    )
    three = _aggregate(
        _story(0.84, story_id="story_a"),
        _story(0.83, story_id="story_b"),
        _story(0.82, story_id="story_c"),
    )

    assert three.aggregate_relevance_score == 0.91573
    assert three.aggregate_relevance_score > two.aggregate_relevance_score
    assert three.contributions[2].weighted_contribution < (
        three.contributions[1].weighted_contribution
    )


def test_fourth_story_never_changes_numeric_aggregate() -> None:
    first_three = tuple(
        _story(score, story_id=f"story_{index}")
        for index, score in enumerate((0.84, 0.83, 0.82), start=1)
    )
    three = _aggregate(*first_three)
    four = _aggregate(*first_three, _story(0.81, story_id="story_4"))

    assert four.aggregate_relevance_score == three.aggregate_relevance_score
    assert four.contributions == three.contributions
    assert four.rankable_story_count == 4
    assert ProjectRelevanceReason.ADDITIONAL_STORIES_CAPPED in four.reasons


def test_many_weak_stories_cannot_game_project_score() -> None:
    weak_flood = tuple(
        _story(
            0.55 if index == 0 else 0.10,
            project_id="project_weak",
            story_id=f"weak_story_{index:03d}",
        )
        for index in range(100)
    )

    weak_project = _aggregate(*weak_flood)
    strong_project = _aggregate(
        _story(0.90, project_id="project_strong", story_id="strong_story")
    )

    assert weak_project.aggregate_relevance_score == 0.55
    assert weak_project.aggregate_relevance_score < strong_project.aggregate_relevance_score
    assert weak_project.contributing_story_count == MAX_CONTRIBUTING_STORIES


def test_policy_constants_are_explicit_and_bounded() -> None:
    assert MAX_CONTRIBUTING_STORIES == 3
    assert PROJECT_STORY_POSITION_WEIGHTS == (1.0, 0.6, 0.2)
    assert DEPTH_QUALIFICATION_FLOOR == 0.5
    assert PROJECT_RELEVANCE_AGGREGATION_POLICY_ID.endswith("headroom_top3.v1")
    assert all(0.0 <= value <= 1.0 for value in PROJECT_STORY_POSITION_WEIGHTS)


@pytest.mark.parametrize(
    ("scores", "expected_weights", "expected_depths"),
    (
        ((0.84,), (1.0,), (0.84,)),
        ((0.84, 0.83), (1.0, 0.6), (0.84, 0.66)),
        ((0.84, 0.83, 0.82), (1.0, 0.6, 0.2), (0.84, 0.66, 0.64)),
    ),
)
def test_position_policy_for_available_story_counts(
    scores: tuple[float, ...],
    expected_weights: tuple[float, ...],
    expected_depths: tuple[float, ...],
) -> None:
    result = _aggregate(*(
        _story(score, story_id=f"story_{index}")
        for index, score in enumerate(scores)
    ))

    assert tuple(item.positional_weight for item in result.contributions) == (
        expected_weights
    )
    assert tuple(item.normalized_depth for item in result.contributions) == (
        expected_depths
    )


@pytest.mark.parametrize(
    "scores",
    ((0.0,), (1.0,), (0.99, 0.99), (1.0, 1.0, 1.0), (0.51, 0.51, 0.51)),
)
def test_aggregate_and_contribution_arithmetic_remain_bounded(
    scores: tuple[float, ...],
) -> None:
    result = _aggregate(*(
        _story(score, story_id=f"story_{index}")
        for index, score in enumerate(scores)
    ))

    assert 0.0 <= result.aggregate_relevance_score <= 1.0
    assert result.aggregate_relevance_score == pytest.approx(sum(
        item.weighted_contribution for item in result.contributions
    ))
    for contribution in result.contributions:
        assert 0.0 <= contribution.positional_weight <= 1.0
        assert 0.0 <= contribution.normalized_depth <= 1.0
        assert 0.0 <= contribution.available_headroom <= 1.0
        assert 0.0 <= contribution.weighted_contribution <= 1.0


def test_contribution_preserves_story_snapshot_and_components() -> None:
    story = _story(0.81, story_id="canonical_story", revision_id="revision_42")
    result = _aggregate(story)
    contribution = result.contributions[0]

    assert contribution.story_relevance is story
    assert contribution.canonical_story_id == "canonical_story"
    assert contribution.current_revision_id == "revision_42"
    assert contribution.story_relevance_id == story.relevance_id
    assert contribution.story_provenance_fingerprint == (
        story.story_provenance_fingerprint
    )
    assert contribution.story_total_relevance == 0.81
    assert contribution.story_components is story.components


def test_project_and_hiring_context_identity_are_preserved() -> None:
    story = _story(
        0.80,
        project_id="project_exact",
        context_id="context_exact",
        context_fingerprint="fingerprint_exact",
    )
    result = _aggregate(story)

    assert result.project_id == "project_exact"
    assert result.hiring_context_profile_id == "context_exact"
    assert result.hiring_context_fingerprint == "fingerprint_exact"
    assert result.rankable_story_count == 1


def test_mixed_project_ids_fail_closed() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        aggregate_project_story_relevance(
            project_id="project_a",
            story_relevance=(
                _story(0.80, project_id="project_a", story_id="story_a"),
                _story(0.79, project_id="project_b", story_id="story_b"),
            ),
        )

    assert exc_info.value.code is ProjectRelevanceAggregationErrorCode.MIXED_PROJECT


def test_mixed_hiring_context_ids_fail_closed() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(
            _story(0.80, story_id="story_a", context_id="context_a"),
            _story(0.79, story_id="story_b", context_id="context_b"),
        )

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.MIXED_HIRING_CONTEXT
    )


def test_mixed_hiring_context_fingerprints_fail_closed() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(
            _story(0.80, story_id="story_a", context_fingerprint="fingerprint_a"),
            _story(0.79, story_id="story_b", context_fingerprint="fingerprint_b"),
        )

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.MIXED_HIRING_CONTEXT_FINGERPRINT
    )


def test_exact_duplicate_story_result_is_deduplicated() -> None:
    story = _story(0.82)
    single = _aggregate(story)
    duplicate = _aggregate(story, replace(story))

    assert duplicate == single
    assert duplicate.rankable_story_count == 1


def test_same_canonical_story_with_conflicting_revisions_fails_closed() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(
            _story(0.82, revision_id="revision_a"),
            _story(0.82, revision_id="revision_b"),
        )

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_REVISION
    )


def test_same_story_revision_with_conflicting_results_fails_closed() -> None:
    story = _story(0.82)
    conflict = replace(
        story,
        raw_relevance_score=0.80,
        total_relevance_score=0.80,
        relevance_id="",
    )

    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(story, conflict)

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_RESULT
    )


def test_one_revision_cannot_belong_to_two_canonical_stories() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(
            _story(0.82, story_id="story_a", revision_id="shared_revision"),
            _story(0.81, story_id="story_b", revision_id="shared_revision"),
        )

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_REVISION
    )


def test_all_input_permutations_produce_identical_result() -> None:
    stories = (
        _story(0.84, story_id="story_c"),
        _story(0.83, story_id="story_a"),
        _story(0.82, story_id="story_b"),
    )

    results = {_aggregate(*permutation) for permutation in itertools.permutations(stories)}

    assert len(results) == 1


def test_equal_scores_use_canonical_story_id_as_tie_break() -> None:
    result = _aggregate(
        _story(0.80, story_id="story_z", revision_id="revision_a"),
        _story(0.80, story_id="story_a", revision_id="revision_z"),
        _story(0.80, story_id="story_m", revision_id="revision_m"),
    )

    assert [item.canonical_story_id for item in result.contributions] == [
        "story_a",
        "story_m",
        "story_z",
    ]


def test_accepted_story_total_and_components_are_not_recomputed() -> None:
    opaque_components = StoryRelevanceComponents(
        explicit_jd_relevance=0.01,
        role_family_relevance=0.02,
        organization_domain_relevance=0.03,
        transferable_engineering_relevance=0.04,
        evidence_claim_safety=0.05,
        story_completeness=0.06,
    )
    story = _story(0.84, components=opaque_components)
    result = _aggregate(story)

    assert result.aggregate_relevance_score == 0.84
    assert result.contributions[0].story_components is opaque_components
    assert result.contributions[0].story_components.explicit_jd_relevance == 0.01
    assert result.contributions[0].story_components.organization_domain_relevance == 0.03
    assert (
        result.contributions[0].story_components.transferable_engineering_relevance
        == 0.04
    )


def test_claim_risk_is_not_applied_a_second_time() -> None:
    story = _story(
        0.60,
        risk_adjustment=0.20,
        claim_sufficiency=SufficiencyLevel.LOW,
    )
    result = _aggregate(story)

    assert story.raw_relevance_score == 0.80
    assert result.aggregate_relevance_score == 0.60
    assert result.contributions[0].story_relevance.evidence_risk_adjustment == 0.20
    assert ProjectRelevanceReason.CLAIM_RISK_PRESENT in result.reasons


def test_low_story_sufficiency_does_not_zero_relevance_and_remains_visible() -> None:
    story = _story(
        0.90,
        story_sufficiency=SufficiencyLevel.LOW,
        story_opportunity=StoryOpportunityLevel.MEDIUM,
    )
    result = _aggregate(story)

    assert result.aggregate_relevance_score == 0.90
    assert result.contributions[0].story_relevance.story_sufficiency is (
        SufficiencyLevel.LOW
    )
    assert result.contributions[0].story_relevance.story_opportunity is (
        StoryOpportunityLevel.MEDIUM
    )
    assert result.contributions[0].story_relevance.clarification_value_hint == 0.67
    assert ProjectRelevanceReason.STORY_COMPLETION_NEEDED in result.reasons


def test_low_claim_safety_is_preserved_without_upgrading_score() -> None:
    components = StoryRelevanceComponents(
        explicit_jd_relevance=1.0,
        role_family_relevance=1.0,
        organization_domain_relevance=1.0,
        transferable_engineering_relevance=1.0,
        evidence_claim_safety=0.68,
        story_completeness=1.0,
    )
    story = _story(
        0.544,
        risk_adjustment=0.256,
        claim_sufficiency=SufficiencyLevel.UNASSESSED,
        components=components,
    )
    result = _aggregate(story)

    assert result.aggregate_relevance_score == 0.544
    assert result.contributions[0].story_components.evidence_claim_safety == 0.68


def test_zero_story_input_returns_no_invented_project_result() -> None:
    assert aggregate_project_story_relevance(
        project_id="project_a",
        story_relevance=(),
    ) is None
    assert rank_projects_from_story_relevance(()) == ()


@pytest.mark.parametrize("invalid", (None, 1, "story", object()))
def test_invalid_story_result_fails_closed(invalid: object) -> None:
    with pytest.raises((TypeError, ProjectRelevanceAggregationError)):
        aggregate_project_story_relevance(
            project_id="project_a",
            story_relevance=invalid if invalid == "story" else (invalid,),  # type: ignore[arg-type]
        )


def test_non_active_story_result_fails_closed() -> None:
    stale = _story(0.80, lifecycle=EngineeringStoryStatus.STALE)

    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        _aggregate(stale)

    assert exc_info.value.code is ProjectRelevanceAggregationErrorCode.INACTIVE_STORY


def test_input_bound_is_checked_before_duplicate_deduplication() -> None:
    story = _story(0.80)

    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        aggregate_project_story_relevance(
            project_id="project_a",
            story_relevance=(story,) * (MAX_PROJECT_STORY_INPUTS + 1),
        )

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.BOUND_EXCEEDED
    )


def test_contribution_and_reason_counts_are_bounded_and_stable() -> None:
    stories = tuple(
        _story(
            score,
            story_id=f"story_{index}",
            risk_adjustment=0.01,
            story_sufficiency=(
                SufficiencyLevel.LOW if index == 0 else SufficiencyLevel.HIGH
            ),
        )
        for index, score in enumerate((0.90, 0.85, 0.80, 0.75))
    )
    result = _aggregate(*stories)

    assert len(result.contributions) == MAX_CONTRIBUTING_STORIES
    assert len(result.reasons) <= MAX_PROJECT_RELEVANCE_REASONS
    assert result.reasons == tuple(
        item for item in ProjectRelevanceReason if item in set(result.reasons)
    )


def test_reason_codes_summarize_exceptional_multiple_and_limited_depth() -> None:
    exceptional = _aggregate(_story(0.90))
    multiple = _aggregate(
        _story(0.84, story_id="story_a"),
        _story(0.83, story_id="story_b"),
    )
    limited = _aggregate(
        _story(0.70, story_id="story_a"),
        _story(0.20, story_id="story_b"),
    )

    assert ProjectRelevanceReason.EXCEPTIONAL_TOP_STORY in exceptional.reasons
    assert ProjectRelevanceReason.MULTIPLE_STRONG_STORIES in multiple.reasons
    assert ProjectRelevanceReason.LIMITED_STORY_DEPTH in limited.reasons


def test_batch_groups_projects_and_sorts_by_project_relevance() -> None:
    stories = (
        _story(0.70, project_id="project_b", story_id="story_b"),
        _story(0.84, project_id="project_a", story_id="story_a1"),
        _story(0.83, project_id="project_a", story_id="story_a2"),
    )

    result = rank_projects_from_story_relevance(stories)

    assert [item.project_id for item in result] == ["project_a", "project_b"]
    assert result[0].rankable_story_count == 2
    assert result[1].rankable_story_count == 1


def test_batch_ties_sort_by_project_id() -> None:
    result = rank_projects_from_story_relevance((
        _story(0.80, project_id="project_z", story_id="story_z"),
        _story(0.80, project_id="project_a", story_id="story_a"),
    ))

    assert [item.project_id for item in result] == ["project_a", "project_z"]


def test_batch_requires_one_comparable_hiring_context() -> None:
    with pytest.raises(ProjectRelevanceAggregationError) as exc_info:
        rank_projects_from_story_relevance((
            _story(0.80, project_id="project_a", context_id="context_a"),
            _story(0.80, project_id="project_b", context_id="context_b"),
        ))

    assert exc_info.value.code is (
        ProjectRelevanceAggregationErrorCode.MIXED_HIRING_CONTEXT
    )


def test_projects_are_independent_inside_batch() -> None:
    project_a_stories = (
        _story(0.84, project_id="project_a", story_id="story_a1"),
        _story(0.83, project_id="project_a", story_id="story_a2"),
    )
    project_b_story = _story(
        0.99,
        project_id="project_b",
        story_id="story_b",
    )
    alone = _aggregate(*project_a_stories)
    batched = rank_projects_from_story_relevance(
        (*project_a_stories, project_b_story)
    )

    assert next(item for item in batched if item.project_id == "project_a") == alone


def test_batch_allows_same_canonical_identity_in_distinct_projects() -> None:
    result = rank_projects_from_story_relevance((
        _story(
            0.80,
            project_id="project_a",
            story_id="shared_canonical",
            revision_id="project_a_revision",
        ),
        _story(
            0.70,
            project_id="project_b",
            story_id="shared_canonical",
            revision_id="project_b_revision",
        ),
    ))

    assert [item.project_id for item in result] == ["project_a", "project_b"]


def test_output_models_are_immutable() -> None:
    result = _aggregate(_story(0.80))

    with pytest.raises(FrozenInstanceError):
        result.aggregate_relevance_score = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.components.secondary_story_depth = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.contributions[0].weighted_contribution = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "forbidden_module",
    (
        "engineering_story_models",
        "engineering_story_memory",
        "hiring_context",
        "candidate_evidence",
        "capability",
        "claim_boundary",
        "retrieval",
        "chroma",
        "api_server",
        "requests",
        "httpx",
        "openai",
        "sqlite",
        "subprocess",
        "pathlib",
        "os",
    ),
)
def test_production_module_has_no_forbidden_dependency(forbidden_module: str) -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert all(forbidden_module not in name.lower() for name in imports)


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "evaluate_engineering_story_relevance",
        "rank_engineering_stories_for_hiring_context",
        "rank_projects_for_resume",
        "select_staged_projects_with_ranking",
        "tailor_resume_staged",
        "open",
        "getenv",
        "load",
        "persist",
        "query",
    ),
)
def test_production_module_has_no_forbidden_runtime_call(forbidden_call: str) -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    call_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)

    assert forbidden_call not in call_names


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "candidate_id",
        "candidate_name",
        "technology_claims",
        "project_budget",
        "story_budget",
        "bullet_count",
        "resume_wording",
        "clarification_question",
        "redundancy_score",
        "differentiation_bonus",
        "selection_rank",
    ),
)
def test_output_models_exclude_deferred_or_unsafe_fields(
    forbidden_field: str,
) -> None:
    output_fields = {
        item.name
        for model in (
            ProjectStoryContribution,
            ProjectRelevanceComponents,
            ProjectHiringRelevance,
        )
        for item in fields(model)
    }

    assert forbidden_field not in output_fields


@pytest.mark.parametrize(
    "forbidden_story_input",
    (
        "raw_relevance_score",
        "evidence_risk_adjustment",
        "components",
        "weights",
        "semantic_features",
        "hiring_context_source_refs",
        "clarification_value_hint",
    ),
)
def test_numeric_aggregation_uses_only_accepted_story_total(
    forbidden_story_input: str,
) -> None:
    source = inspect.getsource(aggregate_project_story_relevance)
    accessed_attributes = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }

    assert forbidden_story_input not in accessed_attributes


@pytest.mark.parametrize(
    "forbidden_story_truth_field",
    (
        "problem_context",
        "implementation",
        "evidence_fact_ids",
        "capability_fact_ids",
        "claim_boundary_ids",
        "job_description",
        "company_name",
        "role_family",
        "technology",
    ),
)
def test_aggregation_never_reads_story_truth_or_rematches_context(
    forbidden_story_truth_field: str,
) -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert forbidden_story_truth_field not in accessed_attributes


def test_project_result_serialization_is_transparent_and_deterministic() -> None:
    result = _aggregate(
        _story(0.84, story_id="story_a"),
        _story(0.83, story_id="story_b"),
    )
    payload = result.to_dict()

    assert payload["project_relevance_id"] == result.project_relevance_id
    assert payload["aggregate_relevance_score"] == 0.90336
    assert payload["contributing_story_count"] == 2
    assert payload["contributions"][0]["canonical_story_id"] == "story_a"
    assert payload["contributions"][1]["weighted_contribution"] == 0.06336
