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
    MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT,
    MAX_PORTFOLIO_FEATURE_REFERENCES,
    MAX_PORTFOLIO_PROJECTS,
    MAX_PORTFOLIO_RANKING_REASONS,
    MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT,
    MAX_PROJECT_SEMANTIC_TRAITS,
    PORTFOLIO_DIFFERENTIATION_BASE_FLOOR,
    PORTFOLIO_FEATURE_INFORMATION_WEIGHTS,
    PORTFOLIO_RANKING_POLICY_ID,
    PortfolioAdjustment,
    PortfolioProjectRanking,
    PortfolioRankingError,
    PortfolioRankingErrorCode,
    PortfolioRankingReason,
    ProjectSemanticFootprint,
    ProjectSemanticTrait,
    RankedProjectStoryPortfolio,
    derive_project_semantic_footprint,
    rank_project_portfolio,
)
from backend.project_story_ranking import (
    ProjectHiringRelevance,
    aggregate_project_story_relevance,
)


MODULE_PATH = Path(__file__).parents[1] / "backend" / "project_portfolio_ranking.py"
CONTEXT_ID = "hiring_context_aaaaaaaaaaaaaaaaaaaaaaaa"
CONTEXT_FINGERPRINT = "a" * 64


def _story(
    score: float,
    *,
    project_id: str,
    story_id: str,
    features: tuple[StoryRelevanceFeature, ...] = (),
    context_id: str = CONTEXT_ID,
    context_fingerprint: str = CONTEXT_FINGERPRINT,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    risk_adjustment: float = 0.0,
    components: StoryRelevanceComponents | None = None,
) -> StoryHiringRelevance:
    accepted_components = components or StoryRelevanceComponents(
        explicit_jd_relevance=0.80,
        role_family_relevance=0.80,
        organization_domain_relevance=0.0,
        transferable_engineering_relevance=0.80,
        evidence_claim_safety=1.0,
        story_completeness=1.0,
    )
    return StoryHiringRelevance(
        project_id=project_id,
        canonical_story_id=story_id,
        current_revision_id=f"revision_{project_id}_{story_id}",
        hiring_context_profile_id=context_id,
        hiring_context_fingerprint=context_fingerprint,
        story_provenance_fingerprint=f"provenance_{project_id}_{story_id}",
        lifecycle_status=EngineeringStoryStatus.ACTIVE,
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
        semantic_features=features,
        reasons=(),
        hiring_context_source_refs=(),
    )


def _project(
    project_id: str,
    score: float,
    features: tuple[StoryRelevanceFeature, ...] = (),
    *,
    context_id: str = CONTEXT_ID,
    context_fingerprint: str = CONTEXT_FINGERPRINT,
    components: StoryRelevanceComponents | None = None,
    claim_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_sufficiency: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_opportunity: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
    risk_adjustment: float = 0.0,
) -> ProjectHiringRelevance:
    story = _story(
        score,
        project_id=project_id,
        story_id=f"story_{project_id}",
        features=features,
        context_id=context_id,
        context_fingerprint=context_fingerprint,
        components=components,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        story_opportunity=story_opportunity,
        risk_adjustment=risk_adjustment,
    )
    result = aggregate_project_story_relevance(
        project_id=project_id,
        story_relevance=(story,),
    )
    assert result is not None
    return result


def _multi_story_project(
    project_id: str,
    stories: tuple[tuple[float, tuple[StoryRelevanceFeature, ...]], ...],
) -> ProjectHiringRelevance:
    results = tuple(
        _story(
            score,
            project_id=project_id,
            story_id=f"story_{project_id}_{index}",
            features=features,
        )
        for index, (score, features) in enumerate(stories)
    )
    project = aggregate_project_story_relevance(
        project_id=project_id,
        story_relevance=results,
    )
    assert project is not None
    return project


def _portfolio(*projects: ProjectHiringRelevance) -> RankedProjectStoryPortfolio:
    result = rank_project_portfolio(projects=projects)
    assert result is not None
    return result


BACKEND_A = (
    StoryRelevanceFeature.BACKEND,
    StoryRelevanceFeature.ARCHITECTURE,
    StoryRelevanceFeature.RELIABILITY,
    StoryRelevanceFeature.API_SYSTEM_DESIGN,
)
BACKEND_B = (
    StoryRelevanceFeature.BACKEND,
    StoryRelevanceFeature.RELIABILITY,
    StoryRelevanceFeature.API_SYSTEM_DESIGN,
    StoryRelevanceFeature.STORAGE,
)
GAME_COMPLEMENT = (
    StoryRelevanceFeature.GAME_DEVELOPMENT,
    StoryRelevanceFeature.STATE_MANAGEMENT,
    StoryRelevanceFeature.PERFORMANCE,
    StoryRelevanceFeature.DEBUGGING,
)


def test_one_project_returns_unchanged_base_first_ordering() -> None:
    project = _project("project_a", 0.91, BACKEND_A)
    portfolio = _portfolio(project)
    ranking = portfolio.ranked_projects[0]

    assert portfolio.project_count == 1
    assert ranking.position == 1
    assert ranking.project_relevance is project
    assert ranking.base_relevance == 0.91
    assert ranking.portfolio_relevance == 0.91
    assert ranking.adjustment.redundancy_adjustment == 0.0
    assert ranking.adjustment.differentiation_adjustment == 0.0


def test_two_unrelated_projects_preserve_clear_base_ordering() -> None:
    stronger = _project("project_a", 0.90, BACKEND_A)
    weaker = _project("project_b", 0.70, GAME_COMPLEMENT)

    portfolio = _portfolio(weaker, stronger)

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_b",
    ]
    assert portfolio.ranked_projects[1].adjustment.differentiation_adjustment > 0


def test_redundancy_is_bounded_and_does_not_mutate_base_result() -> None:
    first = _project("project_a", 0.91, BACKEND_A)
    redundant = _project("project_b", 0.85, BACKEND_B)
    before = redundant.to_dict()

    ranking = _portfolio(first, redundant).ranked_projects[1]

    assert 0.0 < ranking.adjustment.redundancy_score <= 1.0
    assert 0.0 < ranking.adjustment.redundancy_adjustment <= 0.06
    assert ranking.base_relevance == 0.85
    assert redundant.to_dict() == before


def test_unique_relevant_project_receives_bounded_differentiation() -> None:
    first = _project("project_a", 0.91, BACKEND_A)
    complement = _project("project_c", 0.82, GAME_COMPLEMENT)
    before = complement.to_dict()

    ranking = _portfolio(first, complement).ranked_projects[1]

    assert ranking.adjustment.redundancy_score == 0.0
    assert 0.0 < ranking.adjustment.differentiation_score <= 1.0
    assert 0.0 < ranking.adjustment.differentiation_adjustment <= 0.04
    assert ranking.portfolio_relevance > ranking.base_relevance
    assert complement.to_dict() == before


def test_weak_unique_project_cannot_outrank_redundant_strong_project() -> None:
    first = _project("project_a", 0.94, BACKEND_A)
    redundant = _project("project_b", 0.88, BACKEND_A)
    weak_unique = _project("project_c", 0.35, GAME_COMPLEMENT)

    portfolio = _portfolio(first, redundant, weak_unique)
    rankings = {item.project_id: item for item in portfolio.ranked_projects}

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_b",
        "project_c",
    ]
    assert rankings["project_c"].adjustment.differentiation_adjustment == 0.0
    assert rankings["project_c"].portfolio_relevance == 0.35
    assert rankings["project_b"].portfolio_relevance >= 0.82


def test_base_relevance_gap_above_combined_caps_cannot_be_inverted() -> None:
    first = _project("project_a", 0.99, BACKEND_A)
    stronger_redundant = _project("project_b", 0.80, BACKEND_A)
    weaker_unique = _project("project_c", 0.69, GAME_COMPLEMENT)

    portfolio = _portfolio(first, stronger_redundant, weaker_unique)

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_b",
        "project_c",
    ]


def test_first_project_has_no_artificial_uniqueness_or_redundancy() -> None:
    ranking = _portfolio(
        _project("project_a", 0.90, GAME_COMPLEMENT),
        _project("project_b", 0.80, BACKEND_A),
    ).ranked_projects[0]

    assert ranking.adjustment == PortfolioAdjustment(0, 0, 0, 0, (), ())
    assert ranking.reasons == (PortfolioRankingReason.BASE_RELEVANCE_FIRST,)


def test_third_project_compares_against_accumulated_portfolio() -> None:
    first = _project("project_a", 0.91, BACKEND_A)
    unrelated = _project("project_b", 0.89, GAME_COMPLEMENT)
    first_like = _project("project_c", 0.87, BACKEND_A)

    portfolio = _portfolio(first_like, unrelated, first)
    third = portfolio.ranked_projects[2]

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_b",
        "project_c",
    ]
    assert third.adjustment.redundancy_score > 0.70
    assert StoryRelevanceFeature.ARCHITECTURE in (
        third.adjustment.overlapping_features
    )


def test_backend_overlap_and_game_complement_order_a_c_b() -> None:
    portfolio = _portfolio(
        _project("project_a", 0.91, BACKEND_A),
        _project("project_b", 0.85, BACKEND_B),
        _project("project_c", 0.82, GAME_COMPLEMENT),
    )

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_c",
        "project_b",
    ]
    assert [item.portfolio_relevance for item in portfolio.ranked_projects] == [
        0.91,
        0.835779,
        0.814243,
    ]


def test_generic_backend_context_base_scores_can_preserve_a_b_c() -> None:
    portfolio = _portfolio(
        _project("project_a", 0.91, BACKEND_A),
        _project("project_b", 0.85, BACKEND_B),
        _project("project_c", 0.65, GAME_COMPLEMENT),
    )

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_b",
        "project_c",
    ]


def test_supported_game_context_scores_can_produce_a_c_b() -> None:
    portfolio = _portfolio(
        _project("workagent", 0.91, BACKEND_A),
        _project("backend_duplicate", 0.85, BACKEND_B),
        _project("game_project", 0.82, GAME_COMPLEMENT),
    )

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "workagent",
        "game_project",
        "backend_duplicate",
    ]


def test_domain_component_alone_does_not_create_differentiation() -> None:
    domain_components = StoryRelevanceComponents(
        explicit_jd_relevance=0.80,
        role_family_relevance=0.80,
        organization_domain_relevance=1.0,
        transferable_engineering_relevance=0.80,
        evidence_claim_safety=1.0,
        story_completeness=1.0,
    )
    first = _project("project_a", 0.90, ())
    candidate = _project(
        "game_project",
        0.82,
        (),
        components=domain_components,
    )

    ranking = _portfolio(first, candidate).ranked_projects[1]

    assert ranking.semantic_footprint.traits == ()
    assert ranking.adjustment.differentiation_score == 0.0


def test_supported_game_story_creates_candidate_side_differentiation() -> None:
    first = _project("project_a", 0.90, BACKEND_A)
    game = _project(
        "game_project",
        0.82,
        (
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.STATE_MANAGEMENT,
        ),
    )

    ranking = _portfolio(first, game).ranked_projects[1]

    assert ranking.adjustment.differentiation_adjustment > 0.0
    assert StoryRelevanceFeature.STATE_MANAGEMENT in (
        ranking.adjustment.differentiating_features
    )


@pytest.mark.parametrize(
    "context_id",
    (
        "the_coalition_context",
        "microsoft_context",
        "amazon_context",
        "google_context",
    ),
)
def test_company_context_name_cannot_create_candidate_features(
    context_id: str,
) -> None:
    project = _project(
        "project_a",
        0.80,
        (),
        context_id=context_id,
    )

    footprint = derive_project_semantic_footprint(project)

    assert footprint.traits == ()
    assert not any(
        token in {feature.value for feature in StoryRelevanceFeature}
        for token in ("unity", "c++", "c#", "aws", "gcp")
    )


def test_same_domain_different_engineering_is_not_highly_redundant() -> None:
    game_state = _project(
        "game_state",
        0.88,
        (
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.STATE_MANAGEMENT,
            StoryRelevanceFeature.ARCHITECTURE,
        ),
    )
    game_performance = _project(
        "game_performance",
        0.86,
        (
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.PERFORMANCE,
            StoryRelevanceFeature.DEBUGGING,
        ),
    )

    ranking = _portfolio(game_state, game_performance).ranked_projects[1]

    assert ranking.adjustment.redundancy_score < 0.20


def test_different_domains_same_engineering_can_be_highly_redundant() -> None:
    backend = _project(
        "backend_project",
        0.88,
        (
            StoryRelevanceFeature.BACKEND,
            StoryRelevanceFeature.RELIABILITY,
            StoryRelevanceFeature.API_SYSTEM_DESIGN,
            StoryRelevanceFeature.STORAGE,
        ),
    )
    game_backend = _project(
        "game_backend",
        0.84,
        (
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.RELIABILITY,
            StoryRelevanceFeature.API_SYSTEM_DESIGN,
            StoryRelevanceFeature.STORAGE,
        ),
    )

    ranking = _portfolio(backend, game_backend).ranked_projects[1]

    assert ranking.adjustment.redundancy_score > 0.70


def test_one_shared_generic_testing_feature_has_low_redundancy() -> None:
    first = _project("project_a", 0.85, (StoryRelevanceFeature.TESTING,))
    second = _project("project_b", 0.84, (StoryRelevanceFeature.TESTING,))

    ranking = _portfolio(first, second).ranked_projects[1]

    assert ranking.adjustment.redundancy_score == 0.07
    assert ranking.adjustment.redundancy_adjustment < 0.01


def test_multiple_material_features_increase_redundancy() -> None:
    features = (
        StoryRelevanceFeature.RELIABILITY,
        StoryRelevanceFeature.API_SYSTEM_DESIGN,
        StoryRelevanceFeature.STORAGE,
    )
    first = _project("project_a", 0.85, features)
    second = _project("project_b", 0.84, features)

    ranking = _portfolio(first, second).ranked_projects[1]

    assert ranking.adjustment.redundancy_score == 0.84
    assert ranking.adjustment.redundancy_score > 10 * 0.07


def test_information_weights_cover_closed_enum_and_downweight_generic_traits() -> None:
    assert MAX_PROJECT_SEMANTIC_TRAITS == len(StoryRelevanceFeature) == 30
    assert set(PORTFOLIO_FEATURE_INFORMATION_WEIGHTS) == set(StoryRelevanceFeature)
    assert PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[StoryRelevanceFeature.TESTING] == 0.25
    assert PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[StoryRelevanceFeature.DEBUGGING] == 0.25
    assert PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[StoryRelevanceFeature.BACKEND] == 0.50
    assert PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[StoryRelevanceFeature.GAME_DEVELOPMENT] == 0.50
    assert PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[StoryRelevanceFeature.RELIABILITY] == 1.0
    assert "software_engineering" not in {
        feature.value for feature in StoryRelevanceFeature
    }


def test_high_contribution_story_features_have_stronger_footprint() -> None:
    project = _multi_story_project(
        "project_a",
        (
            (0.84, (StoryRelevanceFeature.ARCHITECTURE,)),
            (0.83, (StoryRelevanceFeature.GAME_DEVELOPMENT,)),
        ),
    )
    footprint = derive_project_semantic_footprint(project)
    strengths = {item.feature: item.contribution_strength for item in footprint.traits}

    assert strengths[StoryRelevanceFeature.ARCHITECTURE] == 0.84
    assert strengths[StoryRelevanceFeature.GAME_DEVELOPMENT] == 0.06336


def test_repeated_story_feature_uses_max_not_sum() -> None:
    project = _multi_story_project(
        "project_a",
        (
            (0.84, (StoryRelevanceFeature.RELIABILITY,)),
            (0.83, (StoryRelevanceFeature.RELIABILITY,)),
            (0.82, (StoryRelevanceFeature.RELIABILITY,)),
        ),
    )
    footprint = derive_project_semantic_footprint(project)

    assert len(footprint.traits) == 1
    assert footprint.traits[0].contribution_strength == 0.84


def test_non_contributing_fourth_story_never_enters_footprint() -> None:
    project = _multi_story_project(
        "project_a",
        (
            (0.90, (StoryRelevanceFeature.ARCHITECTURE,)),
            (0.85, (StoryRelevanceFeature.RELIABILITY,)),
            (0.80, (StoryRelevanceFeature.API_SYSTEM_DESIGN,)),
            (0.75, (StoryRelevanceFeature.GAME_DEVELOPMENT,)),
        ),
    )
    footprint = derive_project_semantic_footprint(project)

    assert StoryRelevanceFeature.GAME_DEVELOPMENT not in {
        item.feature for item in footprint.traits
    }
    assert len(footprint.contributing_story_relevance_ids) == 3


@pytest.mark.parametrize(
    ("story_sufficiency", "claim_sufficiency", "risk", "opportunity"),
    (
        (SufficiencyLevel.HIGH, SufficiencyLevel.HIGH, 0.0, StoryOpportunityLevel.NONE),
        (SufficiencyLevel.LOW, SufficiencyLevel.HIGH, 0.0, StoryOpportunityLevel.HIGH),
        (SufficiencyLevel.HIGH, SufficiencyLevel.LOW, 0.20, StoryOpportunityLevel.NONE),
    ),
)
def test_sufficiency_risk_and_opportunity_are_not_multiplied_again(
    story_sufficiency: SufficiencyLevel,
    claim_sufficiency: SufficiencyLevel,
    risk: float,
    opportunity: StoryOpportunityLevel,
) -> None:
    first = _project("project_a", 0.90, BACKEND_A)
    candidate = _project(
        "project_b",
        0.60,
        GAME_COMPLEMENT,
        story_sufficiency=story_sufficiency,
        claim_sufficiency=claim_sufficiency,
        risk_adjustment=risk,
        story_opportunity=opportunity,
    )

    ranking = _portfolio(first, candidate).ranked_projects[1]

    assert ranking.base_relevance == 0.60
    assert ranking.adjustment.redundancy_score == 0.0
    assert ranking.portfolio_relevance == pytest.approx(
        0.60 + ranking.adjustment.differentiation_adjustment
    )
    assert ranking.project_relevance is candidate


def test_base_project_components_and_story_order_are_preserved() -> None:
    project = _multi_story_project(
        "project_a",
        (
            (0.84, (StoryRelevanceFeature.ARCHITECTURE,)),
            (0.83, (StoryRelevanceFeature.GAME_DEVELOPMENT,)),
            (0.82, (StoryRelevanceFeature.STORAGE,)),
        ),
    )
    before_order = tuple(
        item.story_relevance_id for item in project.contributions
    )
    ranking = _portfolio(project).ranked_projects[0]

    assert ranking.project_relevance is project
    assert ranking.project_relevance.components is project.components
    assert tuple(
        item.story_relevance_id
        for item in ranking.project_relevance.contributions
    ) == before_order


@pytest.mark.parametrize(
    "score",
    (0.0, 0.35, PORTFOLIO_DIFFERENTIATION_BASE_FLOOR, 0.99, 1.0),
)
def test_scores_and_adjustments_remain_bounded(score: float) -> None:
    first = _project("project_a", 1.0, BACKEND_A)
    candidate = _project("project_b", score, GAME_COMPLEMENT)
    ranking = _portfolio(first, candidate).ranked_projects[1]

    assert 0.0 <= ranking.adjustment.redundancy_score <= 1.0
    assert 0.0 <= ranking.adjustment.differentiation_score <= 1.0
    assert 0.0 <= ranking.adjustment.redundancy_adjustment <= (
        MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT
    )
    assert 0.0 <= ranking.adjustment.differentiation_adjustment <= (
        MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT
    )
    assert 0.0 <= ranking.portfolio_relevance <= 1.0


def test_extreme_redundancy_cannot_erase_strong_base_relevance() -> None:
    features = tuple(StoryRelevanceFeature)
    first = _project("project_a", 0.90, features[:16])
    second = _project("project_b", 0.88, features[:16])

    ranking = _portfolio(first, second).ranked_projects[1]

    assert ranking.adjustment.redundancy_score == 1.0
    assert ranking.portfolio_relevance >= 0.82


def test_many_redundant_projects_have_individually_capped_penalties() -> None:
    projects = tuple(
        _project(f"project_{index:02d}", 0.90 - index * 0.01, BACKEND_A)
        for index in range(10)
    )

    portfolio = _portfolio(*projects)

    assert portfolio.project_count == 10
    assert all(
        item.adjustment.redundancy_adjustment <= 0.06
        for item in portfolio.ranked_projects
    )


def test_many_unique_weak_projects_receive_no_bonus() -> None:
    features = tuple(StoryRelevanceFeature)
    projects = tuple(
        _project(
            f"project_{index:02d}",
            0.35,
            (features[index],),
        )
        for index in range(12)
    )

    portfolio = _portfolio(*projects)

    assert all(
        item.adjustment.differentiation_adjustment == 0.0
        for item in portfolio.ranked_projects
    )


def test_portfolio_ranks_all_eligible_projects_without_top_n_selection() -> None:
    projects = tuple(
        _project(f"project_{index}", 0.90 - index * 0.05, ())
        for index in range(7)
    )

    portfolio = _portfolio(*projects)

    assert portfolio.project_count == 7
    assert {item.project_id for item in portfolio.ranked_projects} == {
        item.project_id for item in projects
    }


def test_input_permutations_and_repeated_execution_are_deterministic() -> None:
    projects = (
        _project("project_a", 0.91, BACKEND_A),
        _project("project_b", 0.85, BACKEND_B),
        _project("project_c", 0.82, GAME_COMPLEMENT),
    )

    results = {
        _portfolio(*permutation)
        for permutation in itertools.permutations(projects)
    }

    assert len(results) == 1
    first = _portfolio(*projects)
    assert _portfolio(*projects) == first
    assert _portfolio(*projects).portfolio_id == first.portfolio_id


def test_equal_scores_use_project_id_as_stable_tie_break() -> None:
    projects = (
        _project("project_z", 0.80, ()),
        _project("project_a", 0.80, ()),
        _project("project_m", 0.80, ()),
    )

    portfolio = _portfolio(*projects)

    assert [item.project_id for item in portfolio.ranked_projects] == [
        "project_a",
        "project_m",
        "project_z",
    ]


def test_mixed_hiring_context_ids_fail_closed() -> None:
    with pytest.raises(PortfolioRankingError) as exc_info:
        _portfolio(
            _project("project_a", 0.80, context_id="context_a"),
            _project("project_b", 0.79, context_id="context_b"),
        )

    assert exc_info.value.code is PortfolioRankingErrorCode.MIXED_HIRING_CONTEXT


def test_mixed_hiring_context_fingerprints_fail_closed() -> None:
    with pytest.raises(PortfolioRankingError) as exc_info:
        _portfolio(
            _project("project_a", 0.80, context_fingerprint="fingerprint_a"),
            _project("project_b", 0.79, context_fingerprint="fingerprint_b"),
        )

    assert exc_info.value.code is (
        PortfolioRankingErrorCode.MIXED_HIRING_CONTEXT_FINGERPRINT
    )


def test_exact_duplicate_project_result_is_deduplicated() -> None:
    project = _project("project_a", 0.80, BACKEND_A)

    assert _portfolio(project, replace(project)) == _portfolio(project)


def test_conflicting_duplicate_project_result_fails_closed() -> None:
    first = _project("project_a", 0.80, BACKEND_A)
    conflict = _project("project_a", 0.79, GAME_COMPLEMENT)

    with pytest.raises(PortfolioRankingError) as exc_info:
        _portfolio(first, conflict)

    assert exc_info.value.code is (
        PortfolioRankingErrorCode.CONFLICTING_PROJECT_RESULT
    )


def test_zero_and_invalid_project_input_are_explicit() -> None:
    assert rank_project_portfolio(projects=()) is None
    with pytest.raises(TypeError):
        rank_project_portfolio(projects="project")  # type: ignore[arg-type]
    with pytest.raises(PortfolioRankingError) as exc_info:
        rank_project_portfolio(projects=(object(),))  # type: ignore[arg-type]
    assert exc_info.value.code is PortfolioRankingErrorCode.INVALID_INPUT


def test_project_input_bound_is_enforced_before_deduplication() -> None:
    project = _project("project_a", 0.80, ())

    with pytest.raises(PortfolioRankingError) as exc_info:
        rank_project_portfolio(
            projects=(project,) * (MAX_PORTFOLIO_PROJECTS + 1)
        )

    assert exc_info.value.code is PortfolioRankingErrorCode.BOUND_EXCEEDED


def test_semantic_feature_bound_is_enforced() -> None:
    project = _project("project_a", 0.80, ())
    footprint = derive_project_semantic_footprint(project)
    trait = ProjectSemanticTrait(
        StoryRelevanceFeature.ARCHITECTURE,
        0.80,
        1.0,
        0.80,
    )

    with pytest.raises(ValueError):
        replace(
            footprint,
            traits=(trait,) * (MAX_PROJECT_SEMANTIC_TRAITS + 1),
            footprint_id="",
        )


def test_overlap_and_reason_bounds_are_enforced() -> None:
    features = tuple(StoryRelevanceFeature)[:MAX_PORTFOLIO_FEATURE_REFERENCES + 1]
    with pytest.raises(ValueError):
        PortfolioAdjustment(0, 0, 0, 0, features, ())

    ranking = _portfolio(_project("project_a", 0.80, ())).ranked_projects[0]
    with pytest.raises(ValueError):
        replace(
            ranking,
            reasons=(PortfolioRankingReason.BASE_RELEVANCE_FIRST,)
            * (MAX_PORTFOLIO_RANKING_REASONS + 1),
            ranking_id="",
        )


def test_adjustment_caps_are_constructor_enforced() -> None:
    with pytest.raises(ValueError):
        PortfolioAdjustment(
            redundancy_score=1.0,
            redundancy_adjustment=(
                MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT + 0.000001
            ),
            differentiation_score=0.0,
            differentiation_adjustment=0.0,
            overlapping_features=(),
            differentiating_features=(),
        )
    with pytest.raises(ValueError):
        PortfolioAdjustment(
            redundancy_score=0.0,
            redundancy_adjustment=0.0,
            differentiation_score=1.0,
            differentiation_adjustment=(
                MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT + 0.000001
            ),
            overlapping_features=(),
            differentiating_features=(),
        )


def test_portfolio_policy_and_fingerprints_are_stable() -> None:
    projects = (
        _project("project_a", 0.90, BACKEND_A),
        _project("project_b", 0.80, GAME_COMPLEMENT),
    )
    first = _portfolio(*projects)
    second = _portfolio(*reversed(projects))

    assert first.ranking_policy_id == PORTFOLIO_RANKING_POLICY_ID
    assert first.portfolio_id == second.portfolio_id
    assert first.ranked_projects[0].semantic_footprint.footprint_id


def test_material_base_relevance_change_changes_portfolio_fingerprint() -> None:
    first = _portfolio(
        _project("project_a", 0.90, BACKEND_A),
        _project("project_b", 0.80, GAME_COMPLEMENT),
    )
    changed = _portfolio(
        _project("project_a", 0.90, BACKEND_A),
        _project("project_b", 0.79, GAME_COMPLEMENT),
    )

    assert first.portfolio_id != changed.portfolio_id


def test_semantic_footprint_change_changes_fingerprints() -> None:
    original_project = _project("project_a", 0.80, BACKEND_A)
    changed_project = _project("project_a", 0.80, GAME_COMPLEMENT)
    original = _portfolio(original_project)
    changed = _portfolio(changed_project)

    assert (
        original.ranked_projects[0].semantic_footprint.footprint_id
        != changed.ranked_projects[0].semantic_footprint.footprint_id
    )
    assert original.portfolio_id != changed.portfolio_id


def test_portfolio_outputs_are_immutable() -> None:
    portfolio = _portfolio(_project("project_a", 0.80, BACKEND_A))
    ranking = portfolio.ranked_projects[0]
    footprint = ranking.semantic_footprint

    with pytest.raises(FrozenInstanceError):
        portfolio.portfolio_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ranking.portfolio_relevance = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        footprint.traits = ()  # type: ignore[misc]


def test_serialization_preserves_project_and_story_contributions() -> None:
    project = _multi_story_project(
        "project_a",
        (
            (0.84, (StoryRelevanceFeature.ARCHITECTURE,)),
            (0.83, (StoryRelevanceFeature.GAME_DEVELOPMENT,)),
        ),
    )
    portfolio = _portfolio(project)
    payload = portfolio.to_dict()

    assert payload["portfolio_id"] == portfolio.portfolio_id
    assert payload["project_count"] == 1
    base_payload = payload["ranked_projects"][0]["base_project_relevance"]
    assert base_payload["project_relevance_id"] == project.project_relevance_id
    assert len(base_payload["contributions"]) == 2


@pytest.mark.parametrize(
    "forbidden_module",
    (
        "engineering_story_models",
        "engineering_story_memory",
        "hiring_context_models",
        "candidate_evidence",
        "project_evidence",
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
        "aggregate_project_story_relevance",
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
        "candidate_persona",
        "professional_specialization",
        "technology_claims",
        "project_budget",
        "story_budget",
        "bullet_count",
        "resume_wording",
        "clarification_question",
        "selected",
        "dropped",
        "final_project_count",
    ),
)
def test_output_models_exclude_deferred_or_unsafe_fields(
    forbidden_field: str,
) -> None:
    output_fields = {
        item.name
        for model in (
            ProjectSemanticTrait,
            ProjectSemanticFootprint,
            PortfolioAdjustment,
            PortfolioProjectRanking,
            RankedProjectStoryPortfolio,
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
        "claim_sufficiency",
        "story_sufficiency",
        "story_opportunity",
        "clarification_value_hint",
        "reasons",
        "hiring_context_source_refs",
    ),
)
def test_portfolio_scoring_excludes_safety_and_context_components(
    forbidden_story_input: str,
) -> None:
    source = "\n".join((
        inspect.getsource(derive_project_semantic_footprint),
        inspect.getsource(rank_project_portfolio),
    ))
    accessed_attributes = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }

    assert forbidden_story_input not in accessed_attributes


@pytest.mark.parametrize(
    "forbidden_truth_field",
    (
        "problem_context",
        "implementation",
        "observable_outcome",
        "evidence_fact_ids",
        "capability_fact_ids",
        "claim_boundary_ids",
        "job_description",
        "company_name",
        "technology",
        "plausible_missing",
        "unsupported",
    ),
)
def test_portfolio_never_reads_raw_truth_or_missing_suggestions(
    forbidden_truth_field: str,
) -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert forbidden_truth_field not in accessed_attributes


def test_project_hiring_relevance_schema_remains_unchanged() -> None:
    assert "portfolio_relevance" not in {
        item.name for item in fields(ProjectHiringRelevance)
    }
