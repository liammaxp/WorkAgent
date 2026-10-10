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
    StoryRelevanceFeature,
    StoryRelevanceWeights,
)
from backend.project_portfolio_ranking import (
    MAX_PORTFOLIO_PROJECTS,
    RankedProjectStoryPortfolio,
    rank_project_portfolio,
)
from backend.project_story_ranking import (
    MAX_CONTRIBUTING_STORIES,
    ProjectHiringRelevance,
    aggregate_project_story_relevance,
)
from backend.story_clarification_handoff import (
    MAX_STORY_CLARIFICATION_HANDOFFS,
    MAX_STORY_CLARIFICATION_REASONS,
    STORY_CLARIFICATION_HANDOFF_POLICY_ID,
    StoryClarificationHandoff,
    StoryClarificationHandoffError,
    StoryClarificationHandoffErrorCode,
    StoryClarificationReason,
    build_story_clarification_handoffs,
)


MODULE_PATH = (
    Path(__file__).parents[1] / "backend" / "story_clarification_handoff.py"
)
CONTEXT_ID = "hiring_context_aaaaaaaaaaaaaaaaaaaaaaaa"
CONTEXT_FINGERPRINT = "a" * 64


def _story(
    score: float,
    *,
    project_id: str,
    story_id: str,
    revision_id: str | None = None,
    context_id: str = CONTEXT_ID,
    context_fingerprint: str = CONTEXT_FINGERPRINT,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    features: tuple[StoryRelevanceFeature, ...] = (),
) -> StoryHiringRelevance:
    return StoryHiringRelevance(
        project_id=project_id,
        canonical_story_id=story_id,
        current_revision_id=revision_id or f"revision_{project_id}_{story_id}",
        hiring_context_profile_id=context_id,
        hiring_context_fingerprint=context_fingerprint,
        story_provenance_fingerprint=f"provenance_{project_id}_{story_id}",
        lifecycle_status=EngineeringStoryStatus.ACTIVE,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        story_opportunity=story_opportunity,
        components=StoryRelevanceComponents(
            explicit_jd_relevance=0.80,
            role_family_relevance=0.75,
            organization_domain_relevance=0.20,
            transferable_engineering_relevance=0.85,
            evidence_claim_safety=0.70,
            story_completeness=0.65,
        ),
        weights=StoryRelevanceWeights(
            explicit_jd=0.38,
            role_family=0.22,
            organization_domain=0.20,
            transferable_engineering=0.20,
        ),
        raw_relevance_score=score,
        evidence_risk_adjustment=0.0,
        total_relevance_score=score,
        clarification_value_hint=(
            0.0
            if story_opportunity is StoryOpportunityLevel.NONE
            else 0.67
        ),
        semantic_features=features,
        reasons=(),
        hiring_context_source_refs=(),
    )


def _project(
    project_id: str,
    *stories: StoryHiringRelevance,
) -> ProjectHiringRelevance:
    project = aggregate_project_story_relevance(
        project_id=project_id,
        story_relevance=stories,
    )
    assert project is not None
    return project


def _single_story_project(
    project_id: str,
    score: float,
    *,
    story_id: str | None = None,
    revision_id: str | None = None,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    features: tuple[StoryRelevanceFeature, ...] = (),
) -> ProjectHiringRelevance:
    accepted_story_id = story_id or f"story_{project_id}"
    return _project(
        project_id,
        _story(
            score,
            project_id=project_id,
            story_id=accepted_story_id,
            revision_id=revision_id,
            claim_sufficiency=claim_sufficiency,
            story_sufficiency=story_sufficiency,
            story_opportunity=story_opportunity,
            features=features,
        ),
    )


def _portfolio(
    *projects: ProjectHiringRelevance,
) -> RankedProjectStoryPortfolio:
    portfolio = rank_project_portfolio(projects=projects)
    assert portfolio is not None
    return portfolio


def _one_handoff(
    *,
    score: float = 0.90,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
) -> StoryClarificationHandoff:
    project = _single_story_project(
        "project_a",
        score,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        story_opportunity=story_opportunity,
    )
    handoffs = build_story_clarification_handoffs(
        portfolio=_portfolio(project)
    )
    assert len(handoffs) == 1
    return handoffs[0]


@pytest.mark.parametrize(
    ("claim", "story", "claim_gap", "story_gap", "expected_reasons"),
    [
        (
            claim,
            story,
            claim is not SufficiencyLevel.HIGH,
            story is not SufficiencyLevel.HIGH,
            tuple(
                reason
                for reason, present in (
                    (
                        StoryClarificationReason.CLAIM_SAFETY_GAP,
                        claim is not SufficiencyLevel.HIGH,
                    ),
                    (
                        StoryClarificationReason.STORY_COMPLETENESS_GAP,
                        story is not SufficiencyLevel.HIGH,
                    ),
                )
                if present
            ),
        )
        for claim in SufficiencyLevel
        for story in SufficiencyLevel
    ],
)
def test_sufficiency_dimensions_remain_independent(
    claim: SufficiencyLevel,
    story: SufficiencyLevel,
    claim_gap: bool,
    story_gap: bool,
    expected_reasons: tuple[StoryClarificationReason, ...],
) -> None:
    handoff = _one_handoff(
        claim_sufficiency=claim,
        story_sufficiency=story,
    )

    assert handoff.claim_sufficiency is claim
    assert handoff.story_sufficiency is story
    assert handoff.has_claim_safety_gap is claim_gap
    assert handoff.has_story_completeness_gap is story_gap
    assert handoff.reasons == expected_reasons
    assert handoff.has_authoritative_sufficiency_signal is (
        claim_gap or story_gap
    )


@pytest.mark.parametrize("opportunity", tuple(StoryOpportunityLevel))
def test_story_opportunity_level_is_forwarded_without_changing_gaps(
    opportunity: StoryOpportunityLevel,
) -> None:
    handoff = _one_handoff(story_opportunity=opportunity)

    assert handoff.story_opportunity is opportunity
    assert handoff.reasons == ()


def test_technically_safe_story_can_still_emit_completion_handoff() -> None:
    handoff = _one_handoff(
        claim_sufficiency=SufficiencyLevel.HIGH,
        story_sufficiency=SufficiencyLevel.LOW,
    )

    assert not handoff.has_claim_safety_gap
    assert handoff.has_story_completeness_gap
    assert handoff.reasons == (
        StoryClarificationReason.STORY_COMPLETENESS_GAP,
    )


def test_narratively_complete_story_can_still_emit_claim_safety_handoff() -> None:
    handoff = _one_handoff(
        claim_sufficiency=SufficiencyLevel.LOW,
        story_sufficiency=SufficiencyLevel.HIGH,
    )

    assert handoff.has_claim_safety_gap
    assert not handoff.has_story_completeness_gap
    assert handoff.reasons == (StoryClarificationReason.CLAIM_SAFETY_GAP,)


@pytest.mark.parametrize("score", (0.0, 0.15, 0.50, 0.92, 1.0))
def test_incompleteness_does_not_manufacture_story_relevance(score: float) -> None:
    handoff = _one_handoff(
        score=score,
        claim_sufficiency=SufficiencyLevel.LOW,
        story_sufficiency=SufficiencyLevel.LOW,
    )

    assert handoff.story_relevance == score
    assert handoff.has_claim_safety_gap
    assert handoff.has_story_completeness_gap


def test_all_accepted_relevance_values_are_forwarded_exactly() -> None:
    project = _single_story_project("project_a", 0.87)
    portfolio = _portfolio(project)
    ranking = portfolio.ranked_projects[0]
    contribution = project.contributions[0]

    handoff = build_story_clarification_handoffs(portfolio=portfolio)[0]

    assert handoff.story_relevance == (
        contribution.story_relevance.total_relevance_score
    )
    assert handoff.project_story_weighted_contribution == (
        contribution.weighted_contribution
    )
    assert handoff.project_relevance == project.aggregate_relevance_score
    assert handoff.portfolio_relevance == ranking.portfolio_relevance


def test_complete_freshness_and_ranking_identity_is_forwarded() -> None:
    project = _single_story_project("project_a", 0.87)
    portfolio = _portfolio(project)
    ranking = portfolio.ranked_projects[0]
    contribution = project.contributions[0]
    story = contribution.story_relevance

    handoff = build_story_clarification_handoffs(portfolio=portfolio)[0]

    assert handoff.project_id == project.project_id
    assert handoff.canonical_story_id == story.canonical_story_id
    assert handoff.current_revision_id == story.current_revision_id
    assert handoff.story_provenance_fingerprint == (
        story.story_provenance_fingerprint
    )
    assert handoff.hiring_context_profile_id == CONTEXT_ID
    assert handoff.hiring_context_fingerprint == CONTEXT_FINGERPRINT
    assert handoff.story_relevance_id == story.relevance_id
    assert handoff.project_relevance_id == project.project_relevance_id
    assert handoff.portfolio_project_ranking_id == ranking.ranking_id
    assert handoff.portfolio_id == portfolio.portfolio_id
    assert handoff.portfolio_position == ranking.position
    assert handoff.project_story_rank_position == contribution.rank_position


def test_output_order_preserves_portfolio_then_story_ranking_topology() -> None:
    project_a = _project(
        "project_a",
        _story(0.93, project_id="project_a", story_id="story_a_1"),
        _story(0.82, project_id="project_a", story_id="story_a_2"),
        _story(0.70, project_id="project_a", story_id="story_a_3"),
    )
    project_b = _project(
        "project_b",
        _story(0.88, project_id="project_b", story_id="story_b_1"),
        _story(0.76, project_id="project_b", story_id="story_b_2"),
    )
    portfolio = _portfolio(project_a, project_b)

    handoffs = build_story_clarification_handoffs(portfolio=portfolio)

    assert [
        (item.portfolio_position, item.project_story_rank_position)
        for item in handoffs
    ] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]


@pytest.mark.parametrize(
    "project_order",
    tuple(itertools.permutations(("project_a", "project_b", "project_c"))),
)
def test_project_input_permutations_produce_identical_handoffs(
    project_order: tuple[str, ...],
) -> None:
    projects = {
        "project_a": _single_story_project("project_a", 0.91),
        "project_b": _single_story_project("project_b", 0.85),
        "project_c": _single_story_project("project_c", 0.82),
    }

    handoffs = build_story_clarification_handoffs(
        portfolio=_portfolio(*(projects[key] for key in project_order))
    )

    expected = build_story_clarification_handoffs(
        portfolio=_portfolio(*projects.values())
    )
    assert handoffs == expected


def test_story_ties_keep_upstream_stable_story_id_order() -> None:
    project = _project(
        "project_a",
        _story(0.80, project_id="project_a", story_id="story_z"),
        _story(0.80, project_id="project_a", story_id="story_a"),
    )

    handoffs = build_story_clarification_handoffs(
        portfolio=_portfolio(project)
    )

    assert tuple(item.canonical_story_id for item in handoffs) == (
        "story_a",
        "story_z",
    )


def test_repeated_execution_is_deterministic() -> None:
    portfolio = _portfolio(_single_story_project("project_a", 0.90))

    first = build_story_clarification_handoffs(portfolio=portfolio)
    second = build_story_clarification_handoffs(portfolio=portfolio)

    assert first == second
    assert first[0].handoff_id == second[0].handoff_id
    assert first[0].to_dict() == second[0].to_dict()


def test_every_preserved_contribution_gets_one_handoff_including_zero_depth() -> None:
    project = _project(
        "project_a",
        _story(0.90, project_id="project_a", story_id="story_1"),
        _story(0.40, project_id="project_a", story_id="story_2"),
        _story(0.20, project_id="project_a", story_id="story_3"),
    )
    assert project.contributions[-1].weighted_contribution == 0.0

    handoffs = build_story_clarification_handoffs(
        portfolio=_portfolio(project)
    )

    assert len(handoffs) == len(project.contributions) == 3
    assert handoffs[-1].project_story_weighted_contribution == 0.0


def test_noncontributing_fourth_story_is_not_invented() -> None:
    stories = tuple(
        _story(
            score,
            project_id="project_a",
            story_id=f"story_{index}",
        )
        for index, score in enumerate((0.95, 0.85, 0.75, 0.65), start=1)
    )
    project = _project("project_a", *stories)
    assert project.rankable_story_count == 4
    assert len(project.contributions) == MAX_CONTRIBUTING_STORIES

    handoffs = build_story_clarification_handoffs(
        portfolio=_portfolio(project)
    )

    assert len(handoffs) == MAX_CONTRIBUTING_STORIES
    assert "story_4" not in {item.canonical_story_id for item in handoffs}


def test_revision_change_is_visible_without_performing_refresh() -> None:
    old = _single_story_project(
        "project_a",
        0.90,
        revision_id="revision_r1",
    )
    new = _single_story_project(
        "project_a",
        0.90,
        revision_id="revision_r2",
    )

    old_handoff = build_story_clarification_handoffs(
        portfolio=_portfolio(old)
    )[0]
    new_handoff = build_story_clarification_handoffs(
        portfolio=_portfolio(new)
    )[0]

    assert old_handoff.current_revision_id == "revision_r1"
    assert new_handoff.current_revision_id == "revision_r2"
    assert old_handoff.story_relevance_id != new_handoff.story_relevance_id
    assert old_handoff.project_relevance_id != new_handoff.project_relevance_id
    assert old_handoff.portfolio_id != new_handoff.portfolio_id
    assert old_handoff.handoff_id != new_handoff.handoff_id


def test_output_is_frozen_and_slotted() -> None:
    handoff = _one_handoff()

    with pytest.raises(FrozenInstanceError):
        handoff.story_relevance = 0.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        handoff.new_field = "value"  # type: ignore[attr-defined]
    assert not hasattr(handoff, "__dict__")


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("handoff_policy_id", "wrong_policy"),
        ("project_id", " project_a"),
        ("canonical_story_id", ""),
        ("current_revision_id", "revision\ninvalid"),
        ("story_relevance_id", None),
        ("portfolio_position", 0),
        ("portfolio_position", MAX_PORTFOLIO_PROJECTS + 1),
        ("project_story_rank_position", 0),
        ("project_story_rank_position", MAX_CONTRIBUTING_STORIES + 1),
        ("story_relevance", -0.1),
        ("story_relevance", 1.1),
        ("project_story_weighted_contribution", float("nan")),
        ("project_relevance", True),
        ("portfolio_relevance", "high"),
        ("claim_sufficiency", "unsupported"),
        ("story_sufficiency", None),
        ("story_opportunity", "unsupported"),
        ("reasons", []),
        ("handoff_id", "stale_handoff_id"),
    ),
)
def test_handoff_contract_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    handoff = _one_handoff()

    with pytest.raises((TypeError, ValueError)):
        replace(handoff, **{field_name: invalid_value})


def test_handoff_contract_rejects_reasons_in_wrong_order() -> None:
    handoff = _one_handoff(
        claim_sufficiency=SufficiencyLevel.LOW,
        story_sufficiency=SufficiencyLevel.LOW,
    )

    with pytest.raises(ValueError):
        replace(handoff, reasons=tuple(reversed(handoff.reasons)))


def test_reason_bound_is_closed_and_enforced() -> None:
    handoff = _one_handoff(
        claim_sufficiency=SufficiencyLevel.LOW,
        story_sufficiency=SufficiencyLevel.LOW,
    )

    assert MAX_STORY_CLARIFICATION_REASONS == len(StoryClarificationReason) == 2
    assert len(handoff.reasons) == MAX_STORY_CLARIFICATION_REASONS

    with pytest.raises(ValueError):
        replace(
            handoff,
            reasons=(
                StoryClarificationReason.CLAIM_SAFETY_GAP,
                StoryClarificationReason.STORY_COMPLETENESS_GAP,
                StoryClarificationReason.CLAIM_SAFETY_GAP,
            ),
        )


def test_handoff_bound_reuses_accepted_project_and_contribution_bounds() -> None:
    assert MAX_STORY_CLARIFICATION_HANDOFFS == (
        MAX_PORTFOLIO_PROJECTS * MAX_CONTRIBUTING_STORIES
    )


def test_wrong_input_type_fails_closed() -> None:
    with pytest.raises(TypeError):
        build_story_clarification_handoffs(
            portfolio=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("attribute", "wrong_value", "expected_code"),
    (
        (
            "project_id",
            "project_wrong",
            StoryClarificationHandoffErrorCode.PROJECT_IDENTITY_MISMATCH,
        ),
        (
            "current_revision_id",
            "revision_stale",
            StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH,
        ),
        (
            "hiring_context_profile_id",
            "hiring_context_wrong",
            StoryClarificationHandoffErrorCode.MIXED_HIRING_CONTEXT,
        ),
        (
            "hiring_context_fingerprint",
            "b" * 64,
            StoryClarificationHandoffErrorCode.MIXED_HIRING_CONTEXT,
        ),
        (
            "relevance_id",
            "story_hiring_relevance_stale",
            StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH,
        ),
    ),
)
def test_tampered_story_source_fails_closed(
    attribute: str,
    wrong_value: str,
    expected_code: StoryClarificationHandoffErrorCode,
) -> None:
    project = _single_story_project("project_a", 0.90)
    portfolio = _portfolio(project)
    story = project.contributions[0].story_relevance
    object.__setattr__(story, attribute, wrong_value)

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is expected_code


@pytest.mark.parametrize(
    ("target", "attribute", "wrong_value"),
    (
        ("project", "project_relevance_id", "project_relevance_stale"),
        ("ranking", "ranking_id", "portfolio_ranking_stale"),
        ("portfolio", "portfolio_id", "portfolio_stale"),
    ),
)
def test_tampered_ranking_generation_fails_closed(
    target: str,
    attribute: str,
    wrong_value: str,
) -> None:
    project = _single_story_project("project_a", 0.90)
    portfolio = _portfolio(project)
    values = {
        "project": project,
        "ranking": portfolio.ranked_projects[0],
        "portfolio": portfolio,
    }
    object.__setattr__(values[target], attribute, wrong_value)

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is (
        StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH
    )


def test_model_valid_but_stale_footprint_relationship_fails_closed() -> None:
    project = _single_story_project("project_a", 0.90)
    portfolio = _portfolio(project)
    ranking = portfolio.ranked_projects[0]
    stale_footprint = replace(
        ranking.semantic_footprint,
        contributing_story_relevance_ids=("story_hiring_relevance_stale",),
        footprint_id="",
    )
    stale_ranking = replace(
        ranking,
        semantic_footprint=stale_footprint,
        ranking_id="",
    )
    stale_portfolio = replace(
        portfolio,
        ranked_projects=(stale_ranking,),
        portfolio_id="",
    )

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=stale_portfolio)

    assert caught.value.code is (
        StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH
    )


def test_same_canonical_story_cannot_be_owned_by_two_projects() -> None:
    portfolio = _portfolio(
        _single_story_project(
            "project_a",
            0.90,
            story_id="canonical_shared",
        ),
        _single_story_project(
            "project_b",
            0.80,
            story_id="canonical_shared",
        ),
    )

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is (
        StoryClarificationHandoffErrorCode.STORY_IDENTITY_CONFLICT
    )


def test_same_revision_cannot_be_owned_by_two_stories() -> None:
    portfolio = _portfolio(
        _single_story_project(
            "project_a",
            0.90,
            story_id="story_a",
            revision_id="revision_shared",
        ),
        _single_story_project(
            "project_b",
            0.80,
            story_id="story_b",
            revision_id="revision_shared",
        ),
    )

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is (
        StoryClarificationHandoffErrorCode.STORY_IDENTITY_CONFLICT
    )


def test_builder_rejects_tampered_project_count_before_projection() -> None:
    portfolio = _portfolio(_single_story_project("project_a", 0.90))
    object.__setattr__(
        portfolio,
        "ranked_projects",
        portfolio.ranked_projects * (MAX_PORTFOLIO_PROJECTS + 1),
    )

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is StoryClarificationHandoffErrorCode.BOUND_EXCEEDED


def test_builder_rejects_tampered_story_count_before_projection() -> None:
    project = _single_story_project("project_a", 0.90)
    portfolio = _portfolio(project)
    object.__setattr__(
        project,
        "contributions",
        project.contributions * (MAX_STORY_CLARIFICATION_HANDOFFS + 1),
    )

    with pytest.raises(StoryClarificationHandoffError) as caught:
        build_story_clarification_handoffs(portfolio=portfolio)

    assert caught.value.code is StoryClarificationHandoffErrorCode.BOUND_EXCEEDED


def test_serialization_contains_only_bounded_semantic_projection() -> None:
    handoff = _one_handoff(
        claim_sufficiency=SufficiencyLevel.LOW,
        story_sufficiency=SufficiencyLevel.MEDIUM,
        story_opportunity=StoryOpportunityLevel.HIGH,
    )

    payload = handoff.to_dict()

    assert payload["handoff_policy_id"] == STORY_CLARIFICATION_HANDOFF_POLICY_ID
    assert payload["claim_sufficiency"] == "low"
    assert payload["story_sufficiency"] == "medium"
    assert payload["story_opportunity"] == "high"
    assert payload["reasons"] == [
        "claim_safety_gap",
        "story_completeness_gap",
    ]


FORBIDDEN_OUTPUT_FIELDS = (
    "should_ask_user",
    "clarification_priority",
    "final_clarification_priority",
    "searchable_repository_gap",
    "human_only_gap",
    "should_retrieve",
    "needs_github_search",
    "question_text",
    "question_template",
    "question_options",
    "user_prompt",
    "interview_prompt",
    "user_friction",
    "friction_score",
    "answerability",
    "answerability_score",
    "information_gain",
    "information_gain_score",
    "clarification_value",
    "clarification_value_hint",
    "budget_priority",
    "project_budget_priority",
    "story_budget_priority",
    "bullet_budget",
    "story_budget",
    "resume_space_pressure",
    "candidate_fact",
    "story_fact",
    "candidate_identity",
    "candidate_technology",
    "jd_persona",
    "missing_story_fields",
    "story_field_states",
    "opportunity_missing_context",
    "opportunity_signals",
)


@pytest.mark.parametrize("forbidden_field", FORBIDDEN_OUTPUT_FIELDS)
def test_output_model_excludes_downstream_decisions_and_candidate_truth_fields(
    forbidden_field: str,
) -> None:
    model_fields = {item.name for item in fields(StoryClarificationHandoff)}
    serialized_fields = set(_one_handoff().to_dict())

    assert forbidden_field not in model_fields
    assert forbidden_field not in serialized_fields


def test_production_module_has_a_closed_pure_import_surface() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    allowed = {
        "__future__",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "typing",
        "backend.engineering_story_models",
        "backend.project_portfolio_ranking",
        "backend.project_story_ranking",
    }

    assert imported_modules <= allowed


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "evaluate_story_hiring_relevance",
        "rank_engineering_stories",
        "aggregate_project_story_relevance",
        "rank_project_portfolio",
        "load_engineering_story",
        "save_engineering_story",
        "search_project_evidence",
        "retrieve_project_evidence",
        "rank_projects_for_resume",
        "select_staged_projects_with_ranking",
        "tailor_resume_staged",
        "open",
        "getenv",
        "putenv",
    ),
)
def test_builder_does_not_call_recomputation_io_or_integration_functions(
    forbidden_call: str,
) -> None:
    source = inspect.getsource(build_story_clarification_handoffs)
    call_names = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert forbidden_call not in call_names


@pytest.mark.parametrize(
    "forbidden_story_attribute",
    (
        "components",
        "weights",
        "raw_relevance_score",
        "evidence_risk_adjustment",
        "clarification_value_hint",
        "semantic_features",
        "hiring_context_source_refs",
        "supported_fields",
        "missing_fields",
        "field_evidence_states",
        "problem_context",
        "decision",
        "ownership",
        "observable_outcome",
    ),
)
def test_builder_never_reads_fields_that_would_recompute_or_invent_truth(
    forbidden_story_attribute: str,
) -> None:
    source = inspect.getsource(build_story_clarification_handoffs)
    attributes = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }

    assert forbidden_story_attribute not in attributes


def test_module_does_not_import_story_memory_retrieval_chroma_or_frontend() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_fragments = (
        "memory_service",
        "reconstruction",
        "sufficiency",
        "opportunity",
        "retrieval",
        "chroma",
        "api_server",
        "frontend",
    )

    assert not any(
        fragment in module
        for module in modules
        for fragment in forbidden_fragments
    )


def test_contract_is_extensible_without_carrying_candidate_or_budget_truth() -> None:
    handoff = _one_handoff()
    names = {item.name for item in fields(handoff)}

    assert {
        "project_id",
        "canonical_story_id",
        "current_revision_id",
        "story_relevance_id",
        "project_relevance_id",
        "portfolio_id",
    } <= names
    assert names.isdisjoint({
        "candidate_facts",
        "project_budget",
        "story_budget",
        "clarification_priority",
    })
