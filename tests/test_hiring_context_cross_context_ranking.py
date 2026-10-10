"""Cross-context regression matrix for the complete offline ranking pipeline.

The same authoritative candidate Story set is evaluated in every scenario.
Hiring Context may change relevance artifacts, but candidate-owned truth and
candidate-derived semantic features must remain unchanged.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json
import re
from types import MappingProxyType
from typing import Mapping

import pytest

from backend.engineering_story_memory_service import EngineeringStoryView
from backend.engineering_story_models import (
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    StoryRelevanceFeature,
    StoryRelevanceReason,
    rank_engineering_stories_for_hiring_context,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignalKind,
    RankingEffect,
    RoleFamily,
)
from backend.project_portfolio_ranking import (
    MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT,
    MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT,
    PortfolioProjectRanking,
    RankedProjectStoryPortfolio,
    rank_project_portfolio,
)
from backend.project_story_ranking import (
    MAX_CONTRIBUTING_STORIES,
    ProjectHiringRelevance,
    aggregate_project_story_relevance,
)
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    ProjectStoryRankingRefreshError,
    ProjectStoryRankingRefreshErrorCode,
    ProjectStoryRelevanceSnapshot,
    refresh_project_story_ranking,
)
from backend.project_story_ranking_state import (
    ProjectStoryRankingState,
    build_project_story_ranking_state,
    derive_story_clarification_handoffs,
    update_project_story_ranking_state,
)
from backend.story_clarification_handoff import StoryClarificationReason
from tests.test_project_story_ranking_refresh import _view


WORKAGENT = "matrix_workagent"
GAME = "matrix_game"
DATABASE = "matrix_database"
ANALYTICS = "matrix_analytics"
FRONTEND = "matrix_frontend"

PROJECT_IDS = (WORKAGENT, GAME, DATABASE, ANALYTICS, FRONTEND)
STORY_KEYS = (
    "wa_architecture",
    "wa_retrieval_repair",
    "wa_operations",
    "wa_state",
    "wa_hidden_incomplete",
    "game_system",
    "game_debugging",
    "game_backend",
    "game_performance",
    "game_workflow",
    "db_design",
    "db_reliability",
    "db_pipeline",
    "db_security",
    "db_testing",
    "analytics_transform",
    "analytics_decision",
    "analytics_storage",
    "analytics_validation",
    "analytics_reporting",
    "ui_state",
    "ui_integration",
    "ui_accessibility",
    "ui_performance",
    "ui_validation",
)

BACKEND = "backend"
COALITION = "coalition_game"
DATA_ENGINEERING = "data_engineering"
DATA_ANALYTICS = "data_analytics"
DEVOPS = "devops"
FRONTEND_CONTEXT = "frontend"
FULL_STACK = "full_stack"
SECURITY = "security"
CONSULTING = "consulting"
TECHNOLOGY_RISK = "technology_risk"
PRIVACY = "privacy"
CYBER_TRANSFORMATION = "cyber_transformation"
DATA_CONTROLS = "data_controls"

CONTEXT_KEYS = (
    BACKEND,
    COALITION,
    DATA_ENGINEERING,
    DATA_ANALYTICS,
    DEVOPS,
    FRONTEND_CONTEXT,
    FULL_STACK,
    SECURITY,
    CONSULTING,
    TECHNOLOGY_RISK,
    PRIVACY,
    CYBER_TRANSFORMATION,
    DATA_CONTROLS,
)

FORBIDDEN_TECHNOLOGIES = (
    "aws",
    "azure",
    "c++",
    "directx",
    "gcp",
    "java",
    "javascript",
    "kubernetes",
    "power bi",
    "react",
    "snowflake",
    "tableau",
    "unity",
    "unreal",
)

FORBIDDEN_PERSONAS = (
    "technology risk specialist",
    "privacy engineer",
    "cyber consultant",
    "digital trust specialist",
    "risk professional",
    "data controls expert",
    "strategy specialist",
)


@dataclass(frozen=True, slots=True)
class _ContextSpec:
    company: str
    role_title: str
    required: tuple[str, ...]
    team: str | None = None
    parent_organization: str | None = None


@dataclass(frozen=True, slots=True)
class _MatrixRun:
    profile: HiringContextProfile
    views: tuple[EngineeringStoryView, ...]
    story_results: tuple[StoryHiringRelevance, ...]
    stories_by_project: Mapping[str, tuple[StoryHiringRelevance, ...]]
    projects: Mapping[str, ProjectHiringRelevance]
    portfolio: RankedProjectStoryPortfolio
    state: ProjectStoryRankingState
    handoffs: tuple[object, ...]

    def story(self, story_key: str) -> StoryHiringRelevance:
        canonical_id = _story_id(self.views, story_key)
        return next(
            item
            for item in self.story_results
            if item.canonical_story_id == canonical_id
        )

    def ranking(self, project_id: str) -> PortfolioProjectRanking:
        return next(
            item
            for item in self.portfolio.ranked_projects
            if item.project_id == project_id
        )


CONTEXT_SPECS = MappingProxyType({
    BACKEND: _ContextSpec(
        company="Unregistered Systems",
        role_title="Backend Engineer",
        required=(
            "Design reliable backend API and distributed service architecture",
            "Build database storage validation and testing",
        ),
    ),
    COALITION: _ContextSpec(
        company="The Coalition",
        role_title="Software Engineering Intern",
        required=(
            "Build reliable software systems",
            "Debug and test complex interactive features",
        ),
    ),
    DATA_ENGINEERING: _ContextSpec(
        company="Unregistered Data Systems",
        role_title="Data Engineer",
        required=(
            "Build data engineering ETL pipelines and transformations",
            "Design reliable ingestion and storage systems",
        ),
    ),
    DATA_ANALYTICS: _ContextSpec(
        company="Unregistered Analytics",
        role_title="Data Analyst",
        required=(
            "Deliver data analysis metrics and decision support",
            "Build analytics dashboards and reporting transformations",
        ),
    ),
    DEVOPS: _ContextSpec(
        company="Unregistered Infrastructure",
        role_title="Site Reliability Engineer",
        required=(
            "Build reliable infrastructure deployment and observability",
            "Automate operational recovery and validation",
        ),
    ),
    FRONTEND_CONTEXT: _ContextSpec(
        company="Unregistered Product",
        role_title="Frontend Engineer",
        required=(
            "Build React frontend user interface state",
            "Integrate browser workflows with backend APIs",
        ),
    ),
    FULL_STACK: _ContextSpec(
        company="Unregistered Product",
        role_title="Full Stack Engineer",
        required=(
            "Build frontend user interfaces and backend APIs",
            "Own end-to-end web integration and state management",
        ),
    ),
    SECURITY: _ContextSpec(
        company="Unregistered Security",
        role_title="Security Engineer",
        required=(
            "Secure backend systems with authentication and authorization",
            "Assess privacy threats and access controls",
        ),
    ),
    CONSULTING: _ContextSpec(
        company="Unregistered Advisory",
        role_title="Technology Strategy Consultant",
        required=(
            "Deliver analytics decision support for stakeholder strategy",
            "Explain transformation tradeoffs and business requirements",
        ),
    ),
    TECHNOLOGY_RISK: _ContextSpec(
        company="Unregistered Advisory",
        role_title="Technology Risk Consultant",
        required=(
            "Advise stakeholders on Technology Risk and Risk Management",
            "Assess Data Controls and Privacy requirements",
        ),
    ),
    PRIVACY: _ContextSpec(
        company="Unregistered Advisory",
        role_title="Privacy Engineer",
        required=(
            "Lead Privacy engineering and Risk Management",
            "Assess Data Controls for stakeholder systems",
        ),
    ),
    CYBER_TRANSFORMATION: _ContextSpec(
        company="Unregistered Advisory",
        role_title="Cybersecurity Consultant",
        required=(
            "Lead Cyber Transformation and Technology Risk programs",
            "Advise stakeholders on security controls",
        ),
    ),
    DATA_CONTROLS: _ContextSpec(
        company="Unregistered Advisory",
        role_title="Business Analyst",
        required=(
            "Assess Data Controls and Risk Management requirements",
            "Provide consulting decision support to stakeholders",
        ),
    ),
})


def _candidate_views() -> tuple[EngineeringStoryView, ...]:
    return (
        _view(
            story_key="wa_architecture",
            revision_key="wa_architecture_r1",
            project_id=WORKAGENT,
            text=(
                "Designed modular backend API architecture and reliable "
                "distributed service data flow"
            ),
        ),
        _view(
            story_key="wa_retrieval_repair",
            revision_key="wa_retrieval_repair_r1",
            project_id=WORKAGENT,
            text=(
                "Built validation repair testing and debugging for reliable "
                "retrieval ranking workflows"
            ),
        ),
        _view(
            story_key="wa_operations",
            revision_key="wa_operations_r1",
            project_id=WORKAGENT,
            text=(
                "Added operational hardening observability monitoring "
                "deployment automation and recovery"
            ),
        ),
        _view(
            story_key="wa_state",
            revision_key="wa_state_r1",
            project_id=WORKAGENT,
            text="Implemented state management data flow storage cache lifecycle",
        ),
        _view(
            story_key="wa_hidden_incomplete",
            revision_key="wa_hidden_incomplete_r1",
            project_id=WORKAGENT,
            text="Documented a minor workflow",
            story_level=SufficiencyLevel.LOW,
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        _view(
            story_key="game_system",
            revision_key="game_system_r1",
            project_id=GAME,
            text=(
                "Designed gameplay game state system architecture for a "
                "real-time interactive system with algorithms"
            ),
        ),
        _view(
            story_key="game_debugging",
            revision_key="game_debugging_r1",
            project_id=GAME,
            text=(
                "Debugged gameplay defects and validated game system behavior "
                "with regression tests"
            ),
        ),
        _view(
            story_key="game_backend",
            revision_key="game_backend_r1",
            project_id=GAME,
            text=(
                "Built reliable backend service API retry validation for "
                "player session state"
            ),
        ),
        _view(
            story_key="game_performance",
            revision_key="game_performance_r1",
            project_id=GAME,
            text="Optimized performance and latency in the real-time frame loop",
            story_level=SufficiencyLevel.LOW,
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        _view(
            story_key="game_workflow",
            revision_key="game_workflow_r1",
            project_id=GAME,
            text="Automated an asset workflow integration",
        ),
        _view(
            story_key="db_design",
            revision_key="db_design_r1",
            project_id=DATABASE,
            text="Designed database storage schema architecture and backend API",
        ),
        _view(
            story_key="db_reliability",
            revision_key="db_reliability_r1",
            project_id=DATABASE,
            text="Implemented reliable retry validation and recovery for persistence",
        ),
        _view(
            story_key="db_pipeline",
            revision_key="db_pipeline_r1",
            project_id=DATABASE,
            text="Built data flow transformation ingestion pipeline over storage",
        ),
        _view(
            story_key="db_security",
            revision_key="db_security_r1",
            project_id=DATABASE,
            text=(
                "Implemented authentication authorization encryption and "
                "access control for a secure API"
            ),
            claim_level=SufficiencyLevel.LOW,
        ),
        _view(
            story_key="db_testing",
            revision_key="db_testing_r1",
            project_id=DATABASE,
            text="Added integration testing and debugging for a database service",
        ),
        _view(
            story_key="analytics_transform",
            revision_key="analytics_transform_r1",
            project_id=ANALYTICS,
            text=(
                "Built data engineering ETL data pipeline transformation "
                "and analysis"
            ),
        ),
        _view(
            story_key="analytics_decision",
            revision_key="analytics_decision_r1",
            project_id=ANALYTICS,
            text=(
                "Delivered analytics decision support for stakeholder "
                "strategy transformation tradeoffs and business "
                "requirements metrics"
            ),
            story_level=SufficiencyLevel.LOW,
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        _view(
            story_key="analytics_storage",
            revision_key="analytics_storage_r1",
            project_id=ANALYTICS,
            text=(
                "Designed data warehouse ingestion storage pipeline "
                "architecture"
            ),
        ),
        _view(
            story_key="analytics_validation",
            revision_key="analytics_validation_r1",
            project_id=ANALYTICS,
            text="Validated metrics and automated data transformation tests",
        ),
        _view(
            story_key="analytics_reporting",
            revision_key="analytics_reporting_r1",
            project_id=ANALYTICS,
            text="Built analytics reporting and dashboard metrics",
        ),
        _view(
            story_key="ui_state",
            revision_key="ui_state_r1",
            project_id=FRONTEND,
            text=(
                "Designed frontend user interface state management and "
                "browser component workflow"
            ),
        ),
        _view(
            story_key="ui_integration",
            revision_key="ui_integration_r1",
            project_id=FRONTEND,
            text="Integrated frontend UI with backend API service",
        ),
        _view(
            story_key="ui_accessibility",
            revision_key="ui_accessibility_r1",
            project_id=FRONTEND,
            text="Improved accessibility and testing for user interface components",
        ),
        _view(
            story_key="ui_performance",
            revision_key="ui_performance_r1",
            project_id=FRONTEND,
            text="Optimized browser performance and latency",
        ),
        _view(
            story_key="ui_validation",
            revision_key="ui_validation_r1",
            project_id=FRONTEND,
            text="Added validation and debugging for the user workflow",
            story_level=SufficiencyLevel.LOW,
        ),
    )


def _profile(spec: _ContextSpec) -> HiringContextProfile:
    return build_hiring_context_profile(
        company=spec.company,
        team=spec.team,
        parent_organization=spec.parent_organization,
        role_title=spec.role_title,
        normalized_job_context={"required_qualifications": spec.required},
    )


def _run_pipeline(
    profile: HiringContextProfile,
    views: tuple[EngineeringStoryView, ...],
) -> _MatrixRun:
    story_results = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=views,
    )
    grouped: dict[str, list[StoryHiringRelevance]] = {}
    for story in story_results:
        grouped.setdefault(story.project_id, []).append(story)
    stories_by_project = {
        project_id: tuple(stories)
        for project_id, stories in grouped.items()
    }
    projects: dict[str, ProjectHiringRelevance] = {}
    for project_id, stories in stories_by_project.items():
        result = aggregate_project_story_relevance(
            project_id=project_id,
            story_relevance=stories,
        )
        assert result is not None
        projects[project_id] = result
    portfolio = rank_project_portfolio(projects=tuple(projects.values()))
    assert portfolio is not None
    snapshots = tuple(
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=projects[ranking.project_id],
            story_relevance=stories_by_project[ranking.project_id],
        )
        for ranking in portfolio.ranked_projects
    )
    state = build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=portfolio,
        project_snapshots=snapshots,
    )
    handoffs = derive_story_clarification_handoffs(ranking_state=state)
    return _MatrixRun(
        profile=profile,
        views=views,
        story_results=story_results,
        stories_by_project=MappingProxyType(stories_by_project),
        projects=MappingProxyType(projects),
        portfolio=portfolio,
        state=state,
        handoffs=handoffs,
    )


def _story_id(views: tuple[EngineeringStoryView, ...], story_key: str) -> str:
    return views[STORY_KEYS.index(story_key)].canonical_story_id


def _position(run: _MatrixRun, project_id: str) -> int:
    return run.ranking(project_id).position


def _candidate_projection(run: _MatrixRun) -> dict[str, object]:
    return {
        "stories": [
            {
                "project_id": item.project_id,
                "canonical_story_id": item.canonical_story_id,
                "current_revision_id": item.current_revision_id,
                "story_provenance_fingerprint": item.story_provenance_fingerprint,
                "lifecycle_status": item.lifecycle_status.value,
                "claim_sufficiency": item.claim_sufficiency.value,
                "story_sufficiency": item.story_sufficiency.value,
                "story_opportunity": item.story_opportunity.value,
                "semantic_features": [
                    feature.value for feature in item.semantic_features
                ],
            }
            for item in sorted(
                run.story_results,
                key=lambda value: (value.project_id, value.canonical_story_id),
            )
        ],
    }


def _contains_term(value: str, term: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9+#.]){re.escape(term)}(?![a-z0-9+#.])",
        value.casefold(),
    ) is not None


@pytest.fixture(scope="module")
def candidate_views() -> tuple[EngineeringStoryView, ...]:
    return _candidate_views()


@pytest.fixture(scope="module")
def contexts() -> Mapping[str, HiringContextProfile]:
    return MappingProxyType({
        key: _profile(CONTEXT_SPECS[key])
        for key in CONTEXT_KEYS
    })


@pytest.fixture(scope="module")
def matrix(
    contexts: Mapping[str, HiringContextProfile],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> Mapping[str, _MatrixRun]:
    return MappingProxyType({
        key: _run_pipeline(contexts[key], candidate_views)
        for key in CONTEXT_KEYS
    })


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_every_context_reuses_the_exact_candidate_story_set(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    run = matrix[context_key]

    assert run.views is candidate_views
    assert {
        (item.project_id, item.canonical_story_id, item.current_revision_id)
        for item in run.story_results
    } == {
        (view.project_id, view.canonical_story_id, view.current_revision_id)
        for view in candidate_views
    }


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_every_context_builds_a_complete_ranking_state(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[context_key]

    assert run.state.portfolio == run.portfolio
    assert run.state.project_count == len(PROJECT_IDS)
    assert run.state.story_count == len(run.views)
    assert {item.project_id for item in run.state.project_snapshots} == set(
        PROJECT_IDS
    )
    assert len(run.state.project_snapshots) == len(run.portfolio.ranked_projects)


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_context_identity_is_bound_through_the_complete_state(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[context_key]

    assert run.state.hiring_context_profile_id == run.profile.profile_id
    assert run.state.hiring_context_fingerprint == run.profile.fingerprint
    assert run.portfolio.hiring_context_profile_id == run.profile.profile_id
    assert run.portfolio.hiring_context_fingerprint == run.profile.fingerprint
    assert all(
        snapshot.hiring_context_profile_id == run.profile.profile_id
        and snapshot.hiring_context_fingerprint == run.profile.fingerprint
        for snapshot in run.state.project_snapshots
    )


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_hidden_stories_remain_bound_outside_top_three(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[context_key]

    for snapshot in run.state.project_snapshots:
        assert len(snapshot.story_relevance) == 5
        assert len(snapshot.project_relevance.contributions) == MAX_CONTRIBUTING_STORIES
        contributor_ids = {
            item.canonical_story_id
            for item in snapshot.project_relevance.contributions
        }
        assert len({
            item.canonical_story_id for item in snapshot.story_relevance
        } - contributor_ids) == 2


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_portfolio_adjustments_remain_bounded_in_every_context(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    rankings = matrix[context_key].portfolio.ranked_projects
    assert rankings[0].adjustment.redundancy_adjustment == 0
    assert rankings[0].adjustment.differentiation_adjustment == 0
    for ranking in rankings:
        assert (
            ranking.adjustment.redundancy_adjustment
            <= MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT
        )
        assert (
            ranking.adjustment.differentiation_adjustment
            <= MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT
        )
        assert ranking.base_relevance == (
            ranking.project_relevance.aggregate_relevance_score
        )
        assert ranking.portfolio_relevance >= max(
            0.0,
            ranking.base_relevance - MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT,
        )
        assert ranking.portfolio_relevance <= min(
            1.0,
            ranking.base_relevance + MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT,
        )


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_complete_matrix_build_is_deterministic_under_permutation(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
    contexts: Mapping[str, HiringContextProfile],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    rebuilt = _run_pipeline(contexts[context_key], tuple(reversed(candidate_views)))
    accepted = matrix[context_key]

    assert rebuilt.story_results == accepted.story_results
    assert rebuilt.portfolio == accepted.portfolio
    assert rebuilt.state == accepted.state
    assert rebuilt.handoffs == accepted.handoffs
    assert rebuilt.state.state_id == accepted.state.state_id
    assert rebuilt.state.state_fingerprint == accepted.state.state_fingerprint


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_candidate_owned_projection_is_context_independent(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    assert _candidate_projection(matrix[context_key]) == _candidate_projection(
        matrix[BACKEND]
    )


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_contextual_project_footprints_only_use_contributing_story_truth(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    for ranking in matrix[context_key].portfolio.ranked_projects:
        contributing_features = {
            feature
            for contribution in ranking.project_relevance.contributions
            for feature in contribution.story_relevance.semantic_features
        }
        footprint_features = {
            trait.feature for trait in ranking.semantic_footprint.traits
        }

        assert footprint_features <= contributing_features


def test_material_context_changes_create_distinct_profiles_and_states(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    assert len({run.profile.fingerprint for run in matrix.values()}) == len(matrix)
    assert len({run.state.state_fingerprint for run in matrix.values()}) == len(matrix)
    assert len({run.state.state_id for run in matrix.values()}) == len(matrix)


def test_equivalent_normalized_context_builds_the_identical_state(
    candidate_views: tuple[EngineeringStoryView, ...],
    matrix: Mapping[str, _MatrixRun],
) -> None:
    spec = CONTEXT_SPECS[BACKEND]
    equivalent = build_hiring_context_profile(
        company=f"  {spec.company}  ",
        team=None,
        parent_organization=None,
        role_title=f"  {spec.role_title}  ",
        normalized_job_context={
            "required_qualifications": tuple(
                f"  {value}  " for value in reversed(spec.required)
            )
        },
    )
    rebuilt = _run_pipeline(equivalent, tuple(reversed(candidate_views)))

    assert equivalent == matrix[BACKEND].profile
    assert rebuilt.state == matrix[BACKEND].state


def test_snapshot_permutation_is_semantically_set_like(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[BACKEND]
    rebuilt = build_project_story_ranking_state(
        hiring_context=run.profile,
        portfolio=run.portfolio,
        project_snapshots=tuple(reversed(run.state.project_snapshots)),
    )

    assert rebuilt == run.state
    assert rebuilt.state_id == run.state.state_id
    assert rebuilt.state_fingerprint == run.state.state_fingerprint


def test_coalition_registry_contributes_only_accepted_domain_context(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    profile = matrix[COALITION].profile
    domain_signals = tuple(
        signal
        for signal in profile.signals
        if signal.kind is HiringContextSignalKind.COMPANY_DOMAIN
    )

    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING
    assert profile.secondary_role_families == ()
    assert profile.high_value_traits == ()
    assert tuple(
        (signal.value, signal.confidence, signal.ranking_effects)
        for signal in domain_signals
    ) == (
        (
            "Game development",
            HiringContextConfidence.HIGH,
            (RankingEffect.DOMAIN_ALIGNMENT,),
        ),
        (
            "Real-time interactive software",
            HiringContextConfidence.MEDIUM,
            (RankingEffect.DOMAIN_ALIGNMENT,),
        ),
    )


def test_backend_context_values_true_backend_and_system_evidence(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    backend = matrix[BACKEND]
    analytics = matrix[DATA_ANALYTICS]

    assert backend.profile.primary_role_family is RoleFamily.BACKEND_ENGINEERING
    assert backend.story("wa_architecture").components.role_family_relevance > 0
    assert backend.story("db_design").components.role_family_relevance > 0
    assert (
        backend.story("wa_architecture").total_relevance_score
        > analytics.story("wa_architecture").total_relevance_score
    )
    assert _position(backend, WORKAGENT) < _position(backend, FRONTEND)


def test_coalition_context_moves_supported_game_evidence_upward(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    backend = matrix[BACKEND]
    game = matrix[COALITION]

    assert game.profile.company == "The Coalition"
    assert game.story("game_system").components.organization_domain_relevance > 0
    assert game.story("game_debugging").components.organization_domain_relevance > 0
    assert backend.story("game_system").components.organization_domain_relevance == 0
    assert _position(game, GAME) < _position(backend, GAME)


def test_coalition_keeps_general_backend_engineering_transferable(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    game = matrix[COALITION]
    general = game.story("wa_architecture")

    assert general.components.transferable_engineering_relevance > 0
    assert general.components.organization_domain_relevance == 0
    assert StoryRelevanceFeature.GAME_DEVELOPMENT not in general.semantic_features
    assert game.projects[WORKAGENT].aggregate_relevance_score > 0.50


@pytest.mark.parametrize("context_key", (DATA_ENGINEERING, DATA_ANALYTICS))
def test_data_contexts_raise_supported_data_or_analytics_stories(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    data = matrix[context_key]
    game = matrix[COALITION]
    story_key = (
        "analytics_transform"
        if context_key == DATA_ENGINEERING
        else "analytics_decision"
    )

    assert data.story(story_key).components.role_family_relevance > 0
    assert (
        data.story(story_key).total_relevance_score
        > game.story(story_key).total_relevance_score
    )
    assert _position(data, ANALYTICS) < _position(data, GAME)


def test_devops_context_raises_supported_operational_truth(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    devops = matrix[DEVOPS]
    analytics = matrix[DATA_ANALYTICS]

    assert devops.profile.primary_role_family is RoleFamily.DEVOPS_CLOUD
    assert StoryRelevanceFeature.DEVOPS_CLOUD in (
        devops.story("wa_operations").semantic_features
    )
    assert devops.story("wa_operations").components.role_family_relevance > 0
    assert (
        devops.story("wa_operations").total_relevance_score
        > analytics.story("wa_operations").total_relevance_score
    )


@pytest.mark.parametrize("context_key", (FRONTEND_CONTEXT, FULL_STACK))
def test_frontend_contexts_raise_supported_ui_truth(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    frontend = matrix[context_key]
    backend = matrix[BACKEND]

    assert frontend.story("ui_state").components.role_family_relevance > 0
    assert (
        frontend.story("ui_state").total_relevance_score
        > backend.story("ui_state").total_relevance_score
    )
    assert _position(frontend, FRONTEND) < _position(backend, FRONTEND)
    if context_key == FULL_STACK:
        assert frontend.story("wa_architecture").components.role_family_relevance > 0


def test_security_context_rewards_only_supported_security_semantics(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    security = matrix[SECURITY]
    supported = security.story("db_security")
    generic = security.story("db_reliability")

    assert security.profile.primary_role_family is RoleFamily.SECURITY
    assert StoryRelevanceFeature.SECURITY in supported.semantic_features
    assert supported.components.role_family_relevance > 0
    assert StoryRelevanceFeature.SECURITY not in generic.semantic_features
    assert StoryRelevanceReason.PRIMARY_ROLE_ALIGNMENT not in generic.reasons
    assert generic.components.organization_domain_relevance == 0


def test_consulting_context_values_supported_decision_support(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    consulting = matrix[CONSULTING]
    backend = matrix[BACKEND]
    decision = consulting.story("analytics_decision")

    assert consulting.profile.primary_role_family is RoleFamily.CONSULTING_STRATEGY
    assert StoryRelevanceFeature.ANALYTICS in decision.semantic_features
    assert decision.components.role_family_relevance > 0
    assert decision.total_relevance_score > (
        backend.story("analytics_decision").total_relevance_score
    )


@pytest.mark.parametrize(
    "context_key",
    (TECHNOLOGY_RISK, PRIVACY, CYBER_TRANSFORMATION, DATA_CONTROLS, CONSULTING),
)
def test_adversarial_jd_personas_never_become_candidate_output(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    serialized = json.dumps(
        _candidate_projection(matrix[context_key]),
        sort_keys=True,
    ).casefold()

    assert all(persona not in serialized for persona in FORBIDDEN_PERSONAS)
    assert _candidate_projection(matrix[context_key]) == _candidate_projection(
        matrix[BACKEND]
    )


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_context_never_manufactures_candidate_technologies(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    candidate_truth = json.dumps(
        [
            view.current_story.mechanism.value
            for view in candidate_views
        ],
        sort_keys=True,
    )
    candidate_features = {
        feature.value
        for story in matrix[context_key].story_results
        for feature in story.semantic_features
    }

    assert all(
        not _contains_term(candidate_truth, technology)
        for technology in FORBIDDEN_TECHNOLOGIES
    )
    assert candidate_features == {
        feature.value
        for story in matrix[BACKEND].story_results
        for feature in story.semantic_features
    }
    assert all(
        not hasattr(story, "technologies")
        for story in matrix[context_key].story_results
    )


def test_token_collisions_do_not_create_candidate_technology_support(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    frontend = json.dumps(_candidate_projection(matrix[FRONTEND_CONTEXT])).casefold()
    game = json.dumps(_candidate_projection(matrix[COALITION])).casefold()

    assert "react" not in frontend
    assert "java" not in frontend
    assert "javascript" not in frontend
    assert "c++" not in game
    assert "directx" not in game


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_same_story_preserves_authoritative_identity_and_sufficiency(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    run = matrix[context_key]
    result = run.story("wa_architecture")
    view = next(
        item
        for item in candidate_views
        if item.canonical_story_id == result.canonical_story_id
    )

    assert result.canonical_story_id == view.canonical_story_id
    assert result.current_revision_id == view.current_revision_id
    assert result.story_provenance_fingerprint == view.provenance_fingerprint
    assert result.claim_sufficiency == view.claim_sufficiency.level
    assert result.story_sufficiency == view.story_sufficiency.level
    assert result.story_opportunity == view.opportunity.level


def test_same_story_changes_only_contextual_relevance_artifacts(
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    before = tuple(view.to_dict() for view in candidate_views)
    backend = matrix[BACKEND].story("wa_architecture")
    game = matrix[COALITION].story("wa_architecture")
    analytics = matrix[DATA_ANALYTICS].story("wa_architecture")

    assert len({
        backend.total_relevance_score,
        game.total_relevance_score,
        analytics.total_relevance_score,
    }) >= 2
    assert len({
        backend.relevance_id,
        game.relevance_id,
        analytics.relevance_id,
    }) == 3
    assert backend.semantic_features == game.semantic_features == analytics.semantic_features
    assert tuple(view.to_dict() for view in candidate_views) == before


def test_context_change_cannot_use_partial_story_refresh(
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    backend = matrix[BACKEND]
    snapshot = backend.state.snapshot_for_project(WORKAGENT)
    assert snapshot is not None
    unchanged_view = next(
        view
        for view in candidate_views
        if view.canonical_story_id == backend.story("wa_architecture").canonical_story_id
    )

    with pytest.raises(ProjectStoryRankingRefreshError) as captured:
        refresh_project_story_ranking(
            hiring_context=matrix[COALITION].profile,
            prior_portfolio=backend.portfolio,
            prior_project_snapshot=snapshot,
            updated_story_view=unchanged_view,
        )

    assert captured.value.code is ProjectStoryRankingRefreshErrorCode.MIXED_HIRING_CONTEXT


def test_one_exceptional_story_is_not_averaged_down(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[BACKEND]
    strongest = max(
        run.stories_by_project[WORKAGENT],
        key=lambda item: item.total_relevance_score,
    )

    assert run.projects[WORKAGENT].aggregate_relevance_score >= (
        strongest.total_relevance_score
    )


def test_weak_story_flooding_does_not_increase_project_relevance(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[BACKEND]
    strong = run.story("wa_architecture")
    weak = tuple(
        item
        for item in run.stories_by_project[WORKAGENT]
        if item.total_relevance_score <= 0.50
    )
    strong_only = aggregate_project_story_relevance(
        project_id=WORKAGENT,
        story_relevance=(strong,),
    )
    flooded = aggregate_project_story_relevance(
        project_id=WORKAGENT,
        story_relevance=(strong, *weak),
    )

    assert strong_only is not None and flooded is not None
    assert len(weak) >= 2
    assert flooded.aggregate_relevance_score == strong_only.aggregate_relevance_score


def test_multiple_strong_stories_add_bounded_project_headroom(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[BACKEND]
    stories = run.stories_by_project[WORKAGENT]
    strong = tuple(item for item in stories if item.total_relevance_score > 0.50)
    assert len(strong) >= 2
    one = aggregate_project_story_relevance(
        project_id=WORKAGENT,
        story_relevance=strong[:1],
    )
    multiple = aggregate_project_story_relevance(
        project_id=WORKAGENT,
        story_relevance=strong,
    )

    assert one is not None and multiple is not None
    assert multiple.aggregate_relevance_score > one.aggregate_relevance_score
    assert multiple.aggregate_relevance_score <= 1.0


def test_context_shift_can_change_top_three_membership(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    backend_top = {
        item.canonical_story_id
        for item in matrix[BACKEND].projects[ANALYTICS].contributions
    }
    analytics_top = {
        item.canonical_story_id
        for item in matrix[DATA_ANALYTICS].projects[ANALYTICS].contributions
    }

    assert backend_top != analytics_top


def test_same_domain_different_engineering_value_retains_differentiation() -> None:
    profile = _profile(CONTEXT_SPECS[COALITION])
    views = (
        _view(
            story_key="game_arch_project",
            revision_key="game_arch_project_r1",
            project_id="game_arch_project",
            text=(
                "Designed gameplay game state system architecture for a "
                "real-time frame loop"
            ),
        ),
        _view(
            story_key="game_debug_project",
            revision_key="game_debug_project_r1",
            project_id="game_debug_project",
            text=(
                "Debugged gameplay defects and validated game system behavior "
                "with regression tests"
            ),
        ),
    )
    run = _run_pipeline(profile, views)
    second = run.portfolio.ranked_projects[1]

    assert second.adjustment.redundancy_score < 0.20
    assert second.adjustment.differentiation_score > 0
    assert second.adjustment.overlapping_features == (
        StoryRelevanceFeature.GAME_DEVELOPMENT,
    )


def test_different_domain_same_engineering_value_remains_redundant() -> None:
    profile = _profile(CONTEXT_SPECS[COALITION])
    views = (
        _view(
            story_key="game_retry_project",
            revision_key="game_retry_project_r1",
            project_id="game_retry_project",
            text="Built reliable backend API retries validation storage gameplay",
        ),
        _view(
            story_key="backend_retry_project",
            revision_key="backend_retry_project_r1",
            project_id="backend_retry_project",
            text="Built reliable backend API retries validation storage",
        ),
    )
    run = _run_pipeline(profile, views)
    second = run.portfolio.ranked_projects[1]

    assert second.adjustment.redundancy_score > 0.70
    assert StoryRelevanceFeature.RELIABILITY in second.adjustment.overlapping_features
    assert StoryRelevanceFeature.API_SYSTEM_DESIGN in second.adjustment.overlapping_features
    assert second.adjustment.redundancy_adjustment <= MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT


def test_generic_testing_debugging_overlap_is_discounted_in_redundancy(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    game = matrix[COALITION]
    rankings = game.portfolio.ranked_projects[1:]
    generic_overlaps = {
        StoryRelevanceFeature.DEBUGGING,
        StoryRelevanceFeature.TESTING,
    }

    observed = []
    for ranking in rankings:
        if generic_overlaps & set(ranking.adjustment.overlapping_features):
            observed.append(ranking)
            assert ranking.adjustment.redundancy_adjustment < (
                MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT
            )
    assert observed


@pytest.mark.parametrize("context_key", CONTEXT_KEYS)
def test_weak_unique_projects_cannot_overturn_substantial_base_advantage(
    context_key: str,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    rankings = matrix[context_key].portfolio.ranked_projects
    maximum_swing = (
        MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT
        + MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT
    )

    comparisons = 0
    for stronger in rankings:
        for weaker in rankings:
            if stronger.base_relevance - weaker.base_relevance > maximum_swing:
                comparisons += 1
                assert stronger.position < weaker.position
    assert comparisons > 0


@pytest.mark.parametrize(
    ("artifact_name", "attribute", "replacement"),
    (
        ("profile", "role_title", "Changed"),
        ("story", "total_relevance_score", 0.0),
        ("project", "aggregate_relevance_score", 0.0),
        ("state", "state_id", "changed"),
    ),
)
def test_matrix_artifacts_are_immutable(
    artifact_name: str,
    attribute: str,
    replacement: object,
    matrix: Mapping[str, _MatrixRun],
) -> None:
    run = matrix[BACKEND]
    artifact = {
        "profile": run.profile,
        "story": run.story("wa_architecture"),
        "project": run.projects[WORKAGENT],
        "state": run.state,
    }[artifact_name]

    with pytest.raises(FrozenInstanceError):
        setattr(artifact, attribute, replacement)


def _updated_view(
    original: EngineeringStoryView,
    *,
    story_key: str,
    revision_key: str,
    text: str,
    claim_level: SufficiencyLevel,
    story_level: SufficiencyLevel,
) -> EngineeringStoryView:
    return _view(
        story_key=story_key,
        revision_key=revision_key,
        project_id=original.project_id,
        text=text,
        claim_level=claim_level,
        story_level=story_level,
        opportunity_level=original.opportunity.level,
    )


@pytest.mark.parametrize(
    ("context_key", "story_key", "updated_text", "claim_level", "story_level"),
    (
        (
            COALITION,
            "game_performance",
            "Optimized gameplay performance latency in the real-time frame loop",
            SufficiencyLevel.HIGH,
            SufficiencyLevel.HIGH,
        ),
        (
            BACKEND,
            "db_security",
            "Implemented authentication authorization encryption and access control for a secure API",
            SufficiencyLevel.HIGH,
            SufficiencyLevel.HIGH,
        ),
    ),
)
def test_partial_refresh_equals_clean_full_rerank_within_context(
    context_key: str,
    story_key: str,
    updated_text: str,
    claim_level: SufficiencyLevel,
    story_level: SufficiencyLevel,
    matrix: Mapping[str, _MatrixRun],
    candidate_views: tuple[EngineeringStoryView, ...],
) -> None:
    prior = matrix[context_key]
    old = next(
        view
        for view in candidate_views
        if view.canonical_story_id == prior.story(story_key).canonical_story_id
    )
    updated = _updated_view(
        old,
        story_key=story_key,
        revision_key=f"{story_key}_r2",
        text=updated_text,
        claim_level=claim_level,
        story_level=story_level,
    )
    snapshot = prior.state.snapshot_for_project(old.project_id)
    assert snapshot is not None
    refreshed = refresh_project_story_ranking(
        hiring_context=prior.profile,
        prior_portfolio=prior.portfolio,
        prior_project_snapshot=snapshot,
        updated_story_view=updated,
    )
    updated_state = update_project_story_ranking_state(
        prior_state=prior.state,
        refresh_result=refreshed,
    )
    final_views = tuple(
        updated if view.canonical_story_id == old.canonical_story_id else view
        for view in candidate_views
    )
    clean = _run_pipeline(prior.profile, final_views)

    assert updated_state == clean.state
    assert refreshed.new_portfolio == clean.portfolio
    assert refreshed.updated_handoffs == clean.handoffs
    assert updated_state.state_id == clean.state.state_id
    assert updated_state.state_fingerprint == clean.state.state_fingerprint


def test_handoffs_preserve_claim_and_story_sufficiency_separation(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    consulting = matrix[CONSULTING]
    security = matrix[SECURITY]
    decision_id = consulting.story("analytics_decision").canonical_story_id
    security_id = security.story("db_security").canonical_story_id
    decision_handoff = next(
        item for item in consulting.handoffs if item.canonical_story_id == decision_id
    )
    security_handoff = next(
        item for item in security.handoffs if item.canonical_story_id == security_id
    )

    assert decision_handoff.claim_sufficiency is SufficiencyLevel.HIGH
    assert decision_handoff.story_sufficiency is SufficiencyLevel.LOW
    assert decision_handoff.reasons == (
        StoryClarificationReason.STORY_COMPLETENESS_GAP,
    )
    assert security_handoff.claim_sufficiency is SufficiencyLevel.LOW
    assert security_handoff.story_sufficiency is SufficiencyLevel.HIGH
    assert security_handoff.reasons == (
        StoryClarificationReason.CLAIM_SAFETY_GAP,
    )


def test_low_relevance_incomplete_story_gets_no_relevance_boost(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    weak = matrix[BACKEND].story("wa_hidden_incomplete")
    snapshot = matrix[BACKEND].state.snapshot_for_project(WORKAGENT)
    assert snapshot is not None

    assert weak.story_sufficiency is SufficiencyLevel.LOW
    assert weak.total_relevance_score <= 0.50
    assert weak.canonical_story_id not in {
        item.canonical_story_id
        for item in snapshot.project_relevance.contributions
    }
    assert all(
        handoff.canonical_story_id != weak.canonical_story_id
        for handoff in matrix[BACKEND].handoffs
    )


def test_handoffs_are_same_generation_and_have_no_final_priority_fields(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    for run in matrix.values():
        assert run.handoffs == derive_story_clarification_handoffs(
            ranking_state=run.state
        )
        assert all(item.portfolio_id == run.portfolio.portfolio_id for item in run.handoffs)
        assert all(not hasattr(item, "should_ask_user") for item in run.handoffs)
        assert all(not hasattr(item, "final_priority") for item in run.handoffs)


def test_matrix_boundary_has_no_budget_selection_or_external_io_fields(
    matrix: Mapping[str, _MatrixRun],
) -> None:
    serialized = json.dumps(
        {key: run.state.to_dict() for key, run in matrix.items()},
        sort_keys=True,
    ).casefold()

    for forbidden in (
        "project_budget",
        "story_budget",
        "bullet_count",
        "line_allocation",
        "resume_project_count",
        "should_ask_user",
        "web",
        "llm",
        "chroma",
        "retrieval_result",
        "persisted_story",
    ):
        assert forbidden not in serialized
