"""Default-off resume ranking handoff, never evidence retrieval or budgeting.

The caller supplies its existing project-count limit and fixed count allocator.
That downstream structural constraint is transitional debt for the future resume
budget planner; this module neither scores space nor optimizes selection.
Session preference delivery is deferred. No client ranking/overlay is accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

from backend.engineering_story_memory_service import (
    StoryMemoryArtifactStatus, StoryMemoryReadinessState,
    get_active_engineering_stories_for_project,
    inspect_authoritative_engineering_story_memory_readiness,
    load_authoritative_engineering_story_memory,
    validate_authoritative_story_memory_generation,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import HiringContextProfile
from backend.hiring_context_ranking_service import build_configured_hiring_context_ranking_state
from backend.hiring_context_web_ranking import (
    WebAdjustedStoryHiringRelevance, WebAwareProjectStoryRankingState,
)
from backend.project_repository_identity import (
    _REPOSITORY_KEYS, authority_to_repository_mapping,
    load_project_repository_identity_authority, normalize_project_id,
    normalize_repository_identity,
)
from backend.project_story_ranking_state import ProjectStoryRankingState
from backend.project_evidence_memory import load_project_evidence_memory


STORY_AWARE_RESUME_RANKING_FLAG = "USE_STORY_AWARE_RESUME_RANKING"
logger = logging.getLogger(__name__)
RankingState = ProjectStoryRankingState | WebAwareProjectStoryRankingState


class ResumeRankingMode(str, Enum):
    LEGACY_FLAG_OFF = "legacy_flag_off"
    LEGACY_STORY_MEMORY_UNAVAILABLE = "legacy_story_memory_unavailable"
    LEGACY_NO_ACTIVE_STORIES = "legacy_no_active_stories"
    STORY_AWARE_OFFLINE = "story_aware_offline"
    STORY_AWARE_WEB = "story_aware_web"
    FAILED_INTEGRITY = "ranking_integrity_failure"


class ResumeRankingIntegrityError(ValueError):
    """Bounded diagnostic; never includes candidate data or raw exceptions."""

    def __init__(self):
        super().__init__(ResumeRankingMode.FAILED_INTEGRITY.value)


def story_aware_resume_ranking_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(STORY_AWARE_RESUME_RANKING_FLAG, "")
    return isinstance(value, str) and value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ResumeStoryRanking:
    canonical_story_id: str
    current_revision_id: str
    relevance_id: str
    contribution_member: bool
    claim_sufficiency: str
    story_sufficiency: str


@dataclass(frozen=True, slots=True)
class ResumeProjectRanking:
    project_id: str
    portfolio_position: int
    project_relevance_id: str
    stories: tuple[ResumeStoryRanking, ...]
    top_contributing_story_ids: tuple[str, ...]

    def story_handoff(self) -> dict[str, Any]:
        return {
            "ordered_story_ids": [story.canonical_story_id for story in self.stories],
            "current_revision_ids": [story.current_revision_id for story in self.stories],
            "top_contributing_story_ids": list(self.top_contributing_story_ids),
            "stories": [asdict(story) for story in self.stories],
        }


@dataclass(frozen=True, slots=True)
class ResumeRankingPlan:
    """Server-owned full generation; project truncation does not discard state."""

    state: RankingState
    hiring_context: HiringContextProfile
    project_limit: int
    projects: tuple[ResumeProjectRanking, ...] = field(init=False)

    @property
    def ranking_generation_id(self) -> str:
        """Reuse the complete upstream identity, including any effective web state."""
        return self.state.state_id

    def __post_init__(self):
        if not isinstance(self.state, (ProjectStoryRankingState, WebAwareProjectStoryRankingState)):
            raise ResumeRankingIntegrityError()
        if isinstance(self.project_limit, bool) or not isinstance(self.project_limit, int) or self.project_limit < 1:
            raise ResumeRankingIntegrityError()
        state = self.state
        if isinstance(state, WebAwareProjectStoryRankingState):
            replace(state.ranking_state)
        elif any(isinstance(story, WebAdjustedStoryHiringRelevance)
                 for snapshot in state.project_snapshots for story in snapshot.story_relevance):
            raise ResumeRankingIntegrityError()
        replace(state)
        if (state.hiring_context_profile_id != self.hiring_context.profile_id
                or state.hiring_context_fingerprint != self.hiring_context.fingerprint):
            raise ResumeRankingIntegrityError()
        snapshots = {item.project_id: item for item in state.project_snapshots}
        projects = []
        for position, ranked in enumerate(state.portfolio.ranked_projects if state.portfolio else (), 1):
            snapshot = snapshots[ranked.project_id]
            contributors = tuple(item.canonical_story_id for item in snapshot.project_relevance.contributions)
            stories = tuple(sorted(snapshot.story_relevance, key=lambda story: (
                -story.total_relevance_score, story.canonical_story_id,
                story.current_revision_id, story.relevance_id,
            )))
            projects.append(ResumeProjectRanking(
                ranked.project_id, position, snapshot.project_relevance.project_relevance_id,
                tuple(ResumeStoryRanking(
                    story.canonical_story_id, story.current_revision_id, story.relevance_id,
                    story.canonical_story_id in contributors,
                    story.claim_sufficiency.value, story.story_sufficiency.value,
                ) for story in stories), contributors,
            ))
        object.__setattr__(self, "projects", tuple(projects))


@dataclass(frozen=True, slots=True)
class ResumeRankingDecision:
    mode: ResumeRankingMode
    plan: ResumeRankingPlan | None = None


def _decision(mode: ResumeRankingMode, plan: ResumeRankingPlan | None = None) -> ResumeRankingDecision:
    policy = (plan.state.web_influence_policy_id if isinstance(plan.state, WebAwareProjectStoryRankingState)
              else plan.state.state_policy_id) if plan else "none"
    logger.info("resume_ranking mode=%s project_count=%d policy=%s", mode.value,
                min(len(plan.projects), plan.project_limit) if plan else 0, policy)
    return ResumeRankingDecision(mode, plan)


def prepare_resume_ranking(
    *, normalized_job_context: Mapping[str, Any], project_cards: Sequence[Mapping[str, Any]],
    project_limit: int, environ: Mapping[str, str] | None = None,
    story_memory_path: str | Path | None = None,
) -> ResumeRankingDecision:
    """Fallback only for absence/readiness, never integrity or stale generation.

Web acquisition handles unavailable sources internally and cannot select legacy.
Real stale/revalidation/conflicted artifacts fail closed. A temporarily missing
upstream artifact is a readiness fallback only if Story lifecycle is otherwise ready.
"""
    if not story_aware_resume_ranking_enabled(environ):
        return _decision(ResumeRankingMode.LEGACY_FLAG_OFF)
    try:
        try:
            loaded = load_authoritative_engineering_story_memory(story_memory_path)
        except OSError:
            return _decision(ResumeRankingMode.LEGACY_STORY_MEMORY_UNAVAILABLE)
        if loaded.status is StoryMemoryArtifactStatus.MISSING:
            return _decision(ResumeRankingMode.LEGACY_STORY_MEMORY_UNAVAILABLE)
        if loaded.status not in {StoryMemoryArtifactStatus.READY, StoryMemoryArtifactStatus.EMPTY}:
            raise ResumeRankingIntegrityError()
        memory = loaded.memory
        replace(memory)
        try:
            readiness = inspect_authoritative_engineering_story_memory_readiness(
                path=story_memory_path, compare_upstream=True,
            )
        except OSError:
            return _decision(ResumeRankingMode.LEGACY_STORY_MEMORY_UNAVAILABLE)
        validate_authoritative_story_memory_generation(memory, readiness)
        if (readiness.state is StoryMemoryReadinessState.STALE_OR_REVALIDATION_REQUIRED
                and readiness.error_codes == ("upstream_not_ready",)
                and readiness.stale_story_count == 0
                and readiness.requires_revalidation_count == 0
                and readiness.conflicted_story_count == 0):
            # The readiness service collapses missing and corrupt upstream
            # artifacts into one code. Only proven absence permits fallback.
            upstream = load_project_evidence_memory()
            if upstream.status == "missing":
                return _decision(ResumeRankingMode.LEGACY_STORY_MEMORY_UNAVAILABLE)
            raise ResumeRankingIntegrityError()
        if readiness.state is not StoryMemoryReadinessState.READY:
            raise ResumeRankingIntegrityError()
        views = {}
        for project_id in sorted({history.project_id for history in memory.histories}):
            active = get_active_engineering_stories_for_project(memory, project_id)
            if active:
                views[project_id] = active
        if not views:
            return _decision(ResumeRankingMode.LEGACY_NO_ACTIVE_STORIES)
        # Exact backend-owned card identities only; display-name matching is forbidden.
        cards = {}
        for card in project_cards:
            project_id = card.get("project_id")
            if not project_id or normalize_project_id(project_id) != project_id or project_id in cards:
                raise ResumeRankingIntegrityError()
            cards[project_id] = card
        if not set(views).issubset(cards):
            raise ResumeRankingIntegrityError()
        authority = load_project_repository_identity_authority()
        mapping = authority_to_repository_mapping(authority)
        if authority is not None and (authority.get("conflicts") or authority.get("warnings")):
            raise ResumeRankingIntegrityError()
        # Compare card repository claims to the saved authority; never create
        # authority from the same card being checked. Non-repository projects
        # can rank offline without inventing a repository association.
        for project_id, card in cards.items():
            containers = [card]
            if isinstance(card.get("identity"), Mapping):
                containers.append(card["identity"])
            for container in containers:
                for key in _REPOSITORY_KEYS:
                    if container.get(key):
                        repository = normalize_repository_identity(container[key])
                        canonical = mapping["alias_to_repository"].get(repository, repository)
                        if not repository or mapping["repository_to_project"].get(canonical) != project_id:
                            raise ResumeRankingIntegrityError()
        profile = build_hiring_context_profile(
            company=normalized_job_context.get("company"),
            team=normalized_job_context.get("team"),
            role_title=normalized_job_context.get("job_title"),
            normalized_job_context=normalized_job_context,
        )
        state = build_configured_hiring_context_ranking_state(
            hiring_context=profile, views_by_project=views, environ=environ,
        )
        plan = ResumeRankingPlan(state, profile, project_limit)
        return _decision(
            ResumeRankingMode.STORY_AWARE_WEB if isinstance(state, WebAwareProjectStoryRankingState)
            else ResumeRankingMode.STORY_AWARE_OFFLINE, plan,
        )
    except Exception as error:
        logger.warning("resume_ranking mode=%s", ResumeRankingMode.FAILED_INTEGRITY.value)
        raise ResumeRankingIntegrityError() from error


def production_selection(
    plan: ResumeRankingPlan, project_cards: Sequence[dict[str, Any]],
    assign_legacy_count: Callable[[int, int], int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt ordered projects to existing generation fields; never run a ranker.

The fixed bullet allocator is injected by production after selection. It sees
only position/count, never evidence, relevance, or project identity.
"""
    replace(plan)
    cards = {card["project_id"]: card for card in project_cards}
    if len(cards) != len(project_cards) or not {p.project_id for p in plan.projects}.issubset(cards):
        raise ResumeRankingIntegrityError()
    selected = plan.projects[:plan.project_limit]
    ranked_ids = {item.project_id for item in plan.projects}
    entries = [
        {
            "project_id": item.project_id,
            "project_name": str(cards[item.project_id].get("project_name")
                                or cards[item.project_id].get("name") or item.project_id),
            "rank": item.portfolio_position,
            "bullet_budget": assign_legacy_count(item.portfolio_position, len(selected)),
            "story_ranking": item.story_handoff(),
        } for item in selected
    ]
    return [cards[item.project_id] for item in selected], {
        "ranking_generation_id": plan.ranking_generation_id,
        "selected_projects": entries,
        "omitted_projects": [
            {"project_id": item.project_id,
             "project_name": str(cards[item.project_id].get("project_name")
                                 or cards[item.project_id].get("name") or item.project_id),
             "reason_omitted": "Existing project-count limit"}
            for item in plan.projects[plan.project_limit:]
        ] + [
            {"project_id": project_id,
             "project_name": str(cards[project_id].get("project_name")
                                 or cards[project_id].get("name") or project_id),
             "reason_omitted": "No active ranked Stories"}
            for project_id in sorted(set(cards) - ranked_ids)
        ],
    }
