"""Shared, read-only ranking orchestration for review and resume production."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from backend.engineering_story_memory_service import EngineeringStoryView
from backend.hiring_context_models import HiringContextProfile
from backend.engineering_story_relevance import rank_engineering_stories_for_hiring_context
from backend.project_story_ranking import aggregate_project_story_relevance
from backend.project_portfolio_ranking import rank_project_portfolio
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID, ProjectStoryRelevanceSnapshot,
)
from backend.project_story_ranking_state import build_project_story_ranking_state
from backend.hiring_context_web_enrichment import (
    HiringContextEnrichment, build_hiring_context_web_enrichment,
    official_hiring_context_enrichment_enabled,
)
from backend.hiring_context_web_ranking import (
    WebAwareProjectStoryRankingState,
    official_hiring_context_ranking_influence_enabled,
)
from backend.project_story_ranking_state import ProjectStoryRankingState


HiringContextRankingState = ProjectStoryRankingState | WebAwareProjectStoryRankingState


def build_configured_hiring_context_ranking_state(
    *, hiring_context: HiringContextProfile,
    views_by_project: Mapping[str, Sequence[EngineeringStoryView]],
    environ: Mapping[str, str] | None = None,
) -> HiringContextRankingState:
    """Resolve the same optional official context for every ranking consumer.

    Production rollout and review availability do not select ranking semantics.
    Acquisition remains owned by the accepted enrichment boundary.
    """
    enrichment = None
    if official_hiring_context_enrichment_enabled(environ):
        enrichment = build_hiring_context_web_enrichment(hiring_context, environ=environ)
    return build_hiring_context_ranking_state(
        hiring_context=hiring_context, views_by_project=views_by_project,
        enrichment=enrichment,
        web_influence=official_hiring_context_ranking_influence_enabled(environ),
    )


def build_hiring_context_ranking_state(
    *, hiring_context: HiringContextProfile,
    views_by_project: Mapping[str, Sequence[EngineeringStoryView]],
    enrichment: HiringContextEnrichment | None = None,
    web_influence: bool = False,
) -> ProjectStoryRankingState | WebAwareProjectStoryRankingState:
    """Use accepted Story, project, portfolio and state policies without retuning."""
    relevance = {}
    for project_id, views in sorted(views_by_project.items()):
        if any(view.project_id != project_id for view in views):
            raise ValueError("cross-project Story identity")
        relevance[project_id] = rank_engineering_stories_for_hiring_context(
            hiring_context=hiring_context, story_views=views,
        )
    if enrichment is not None and web_influence:
        from backend.hiring_context_web_ranking import rank_projects_with_hiring_context_web_influence
        return rank_projects_with_hiring_context_web_influence(
            hiring_context=hiring_context, enrichment=enrichment,
            story_relevance=tuple(story for stories in relevance.values() for story in stories),
            enabled=True,
        ).state
    projects = {}
    for project_id, stories in relevance.items():
        project = aggregate_project_story_relevance(project_id=project_id, story_relevance=stories)
        if project is not None:
            projects[project_id] = project
    portfolio = rank_project_portfolio(projects=tuple(projects.values()))
    snapshots = () if portfolio is None else tuple(
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=projects[item.project_id],
            story_relevance=relevance[item.project_id],
        ) for item in portfolio.ranked_projects
    )
    return build_project_story_ranking_state(
        hiring_context=hiring_context, portfolio=portfolio, project_snapshots=snapshots,
    )
