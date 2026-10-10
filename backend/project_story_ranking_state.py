"""Coherent immutable ranking generation for downstream semantic consumers.

The state binds an accepted Hiring Context identity, portfolio, and every
complete project Story-relevance snapshot.  It does not evaluate Stories,
aggregate projects, rank portfolios, allocate resume space, or load source
truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Any, Never

from backend.hiring_context_models import HiringContextProfile
from backend.project_portfolio_ranking import (
    MAX_PORTFOLIO_PROJECTS,
    PortfolioProjectRanking,
    RankedProjectStoryPortfolio,
)
from backend.project_story_ranking import (
    MAX_PROJECT_STORY_INPUTS,
    ProjectStoryContribution,
)
from backend.project_story_ranking_refresh import (
    PartialProjectStoryRerankResult,
    ProjectStoryRankingRefreshStatus,
    ProjectStoryRelevanceSnapshot,
)
from backend.story_clarification_handoff import (
    StoryClarificationHandoff,
    StoryClarificationHandoffError,
    build_story_clarification_handoffs,
)


PROJECT_STORY_RANKING_STATE_POLICY_ID = (
    "project_story_ranking_state.complete_generation.v1"
)
MAX_RANKING_STATE_PROJECTS = MAX_PORTFOLIO_PROJECTS
MAX_RANKING_STATE_STORIES_PER_PROJECT = MAX_PROJECT_STORY_INPUTS
MAX_RANKING_STATE_STORIES = (
    MAX_RANKING_STATE_PROJECTS * MAX_RANKING_STATE_STORIES_PER_PROJECT
)


class ProjectStoryRankingStateErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    MIXED_HIRING_CONTEXT = "mixed_hiring_context"
    MISSING_PROJECT_SNAPSHOT = "missing_project_snapshot"
    UNKNOWN_PROJECT_SNAPSHOT = "unknown_project_snapshot"
    DUPLICATE_PROJECT_SNAPSHOT = "duplicate_project_snapshot"
    STALE_PORTFOLIO = "stale_portfolio"
    CONFLICTING_STORY_IDENTITY = "conflicting_story_identity"
    REFRESH_MISMATCH = "refresh_mismatch"


class ProjectStoryRankingStateError(ValueError):
    def __init__(
        self,
        code: ProjectStoryRankingStateErrorCode,
        message: str,
    ) -> None:
        self.code = ProjectStoryRankingStateErrorCode(code)
        super().__init__(message)


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _context_identity(profile_id: Any, fingerprint: Any) -> tuple[str, str]:
    normalized_id = _exact_text(profile_id, "hiring_context_profile_id")
    normalized_fingerprint = _exact_text(
        fingerprint,
        "hiring_context_fingerprint",
    )
    id_suffix = normalized_id.removeprefix("hiring_context_")
    if (
        len(id_suffix) != 24
        or normalized_id != f"hiring_context_{id_suffix}"
        or any(char not in "0123456789abcdef" for char in id_suffix)
    ):
        raise ValueError("hiring_context_profile_id has an invalid identity shape")
    if len(normalized_fingerprint) != 64 or any(
        char not in "0123456789abcdef"
        for char in normalized_fingerprint
    ):
        raise ValueError("hiring_context_fingerprint must be a SHA-256 value")
    return normalized_id, normalized_fingerprint


def _source_error(name: str, error: Exception) -> Never:
    raise ProjectStoryRankingStateError(
        ProjectStoryRankingStateErrorCode.INVALID_INPUT,
        f"{name} no longer satisfies its immutable contract",
    ) from error


def _validate_contract(value: Any, name: str) -> None:
    try:
        validated = replace(value)
    except (TypeError, ValueError) as error:
        _source_error(name, error)
    if validated != value:
        _source_error(
            name,
            ValueError("normalization changed the accepted source"),
        )


def _validate_portfolio(portfolio: RankedProjectStoryPortfolio) -> None:
    for ranking in portfolio.ranked_projects:
        project = ranking.project_relevance
        for contribution in project.contributions:
            _validate_contract(
                contribution.story_relevance.components,
                "portfolio Story relevance components",
            )
            _validate_contract(
                contribution.story_relevance.weights,
                "portfolio Story relevance weights",
            )
            _validate_contract(
                contribution.story_relevance,
                "portfolio Story relevance",
            )
            _validate_contract(contribution, "portfolio Story contribution")
        _validate_contract(project.components, "portfolio project components")
        _validate_contract(project, "portfolio project relevance")
        for trait in ranking.semantic_footprint.traits:
            _validate_contract(trait, "portfolio semantic trait")
        _validate_contract(
            ranking.semantic_footprint,
            "portfolio semantic footprint",
        )
        _validate_contract(ranking.adjustment, "portfolio adjustment")
        _validate_contract(ranking, "portfolio project ranking")
    _validate_contract(portfolio, "ranked project Story portfolio")


def _state_payload(
    *,
    policy_id: str,
    hiring_context_profile_id: str,
    hiring_context_fingerprint: str,
    portfolio: RankedProjectStoryPortfolio | None,
    project_snapshots: tuple[ProjectStoryRelevanceSnapshot, ...],
) -> dict[str, Any]:
    return {
        "state_policy_id": policy_id,
        "hiring_context_profile_id": hiring_context_profile_id,
        "hiring_context_fingerprint": hiring_context_fingerprint,
        "portfolio_id": None if portfolio is None else portfolio.portfolio_id,
        "project_snapshots": [
            {
                "project_id": snapshot.project_id,
                "snapshot_id": snapshot.snapshot_id,
            }
            for snapshot in project_snapshots
        ],
    }


def _state_identity(payload: dict[str, Any]) -> tuple[str, str]:
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return fingerprint, f"project_story_ranking_state_{fingerprint[:24]}"


@dataclass(frozen=True, slots=True)
class ProjectStoryRankingState:
    """One complete validated project/Story ranking generation."""

    state_policy_id: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    portfolio: RankedProjectStoryPortfolio | None
    project_snapshots: tuple[ProjectStoryRelevanceSnapshot, ...]
    state_fingerprint: str = ""
    state_id: str = ""

    def __post_init__(self) -> None:
        policy_id = _exact_text(self.state_policy_id, "state_policy_id")
        if policy_id != PROJECT_STORY_RANKING_STATE_POLICY_ID:
            raise ValueError("state_policy_id is not supported")
        profile_id, context_fingerprint = _context_identity(
            self.hiring_context_profile_id,
            self.hiring_context_fingerprint,
        )
        if self.portfolio is not None and not isinstance(
            self.portfolio,
            RankedProjectStoryPortfolio,
        ):
            raise TypeError("portfolio must be a ranked portfolio or None")
        if isinstance(self.project_snapshots, (str, bytes)) or not isinstance(
            self.project_snapshots,
            Sequence,
        ):
            raise TypeError("project_snapshots must be a sequence")
        if len(self.project_snapshots) > MAX_RANKING_STATE_PROJECTS:
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.BOUND_EXCEEDED,
                "project snapshots exceed the ranking-state project bound",
            )
        snapshots_by_project: dict[str, ProjectStoryRelevanceSnapshot] = {}
        for snapshot in self.project_snapshots:
            if not isinstance(snapshot, ProjectStoryRelevanceSnapshot):
                raise TypeError(
                    "project_snapshots must contain project Story snapshots"
                )
            _validate_contract(snapshot, "project Story relevance snapshot")
            if snapshot.project_id in snapshots_by_project:
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.DUPLICATE_PROJECT_SNAPSHOT,
                    "each ranked project requires exactly one snapshot",
                )
            snapshots_by_project[snapshot.project_id] = snapshot
        if self.portfolio is None:
            if snapshots_by_project:
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.UNKNOWN_PROJECT_SNAPSHOT,
                    "an empty portfolio cannot retain project snapshots",
                )
            ordered_snapshots: tuple[ProjectStoryRelevanceSnapshot, ...] = ()
        else:
            _validate_portfolio(self.portfolio)
            if (
                self.portfolio.hiring_context_profile_id != profile_id
                or self.portfolio.hiring_context_fingerprint
                != context_fingerprint
            ):
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.MIXED_HIRING_CONTEXT,
                    "portfolio does not match the ranking-state Hiring Context",
                )
            portfolio_projects = {
                item.project_id for item in self.portfolio.ranked_projects
            }
            snapshot_projects = set(snapshots_by_project)
            missing = portfolio_projects - snapshot_projects
            if missing:
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.MISSING_PROJECT_SNAPSHOT,
                    "every ranked project requires a complete Story snapshot",
                )
            extra = snapshot_projects - portfolio_projects
            if extra:
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.UNKNOWN_PROJECT_SNAPSHOT,
                    "snapshot exists for a project outside the portfolio",
                )
            ordered_snapshots = tuple(
                snapshots_by_project[item.project_id]
                for item in self.portfolio.ranked_projects
            )
            for ranking, snapshot in zip(
                self.portfolio.ranked_projects,
                ordered_snapshots,
                strict=True,
            ):
                if snapshot.source_portfolio_id != self.portfolio.portfolio_id:
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.STALE_PORTFOLIO,
                        "snapshot is bound to a different portfolio generation",
                    )
                if (
                    snapshot.hiring_context_profile_id != profile_id
                    or snapshot.hiring_context_fingerprint
                    != context_fingerprint
                ):
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.MIXED_HIRING_CONTEXT,
                        "snapshot does not match the ranking-state Hiring Context",
                    )
                if snapshot.project_relevance != ranking.project_relevance:
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.STALE_PORTFOLIO,
                        "snapshot project relevance differs from the portfolio",
                    )
        story_count = sum(
            len(snapshot.story_relevance) for snapshot in ordered_snapshots
        )
        if story_count > MAX_RANKING_STATE_STORIES:
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.BOUND_EXCEEDED,
                "complete Story state exceeds the ranking-state total bound",
            )
        canonical_owners: dict[str, str] = {}
        revision_owners: dict[str, tuple[str, str]] = {}
        relevance_owners: dict[str, tuple[str, str, str]] = {}
        for snapshot in ordered_snapshots:
            for story in snapshot.story_relevance:
                prior_project = canonical_owners.get(story.canonical_story_id)
                if prior_project is not None:
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.CONFLICTING_STORY_IDENTITY,
                        "canonical Story identity occurs in multiple snapshots",
                    )
                canonical_owners[story.canonical_story_id] = snapshot.project_id
                story_owner = (
                    snapshot.project_id,
                    story.canonical_story_id,
                )
                prior_revision = revision_owners.get(story.current_revision_id)
                if prior_revision is not None and prior_revision != story_owner:
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.CONFLICTING_STORY_IDENTITY,
                        "Story revision identity has conflicting ownership",
                    )
                revision_owners[story.current_revision_id] = story_owner
                relevance_owner = (*story_owner, story.current_revision_id)
                prior_relevance = relevance_owners.get(story.relevance_id)
                if (
                    prior_relevance is not None
                    and prior_relevance != relevance_owner
                ):
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.CONFLICTING_STORY_IDENTITY,
                        "Story relevance identity has conflicting ownership",
                    )
                relevance_owners[story.relevance_id] = relevance_owner
        payload = _state_payload(
            policy_id=policy_id,
            hiring_context_profile_id=profile_id,
            hiring_context_fingerprint=context_fingerprint,
            portfolio=self.portfolio,
            project_snapshots=ordered_snapshots,
        )
        expected_fingerprint, expected_id = _state_identity(payload)
        if self.state_fingerprint not in ("", expected_fingerprint):
            raise ValueError(
                "state_fingerprint does not match the complete ranking state"
            )
        if self.state_id not in ("", expected_id):
            raise ValueError("state_id does not match the complete ranking state")
        object.__setattr__(self, "state_policy_id", policy_id)
        object.__setattr__(self, "hiring_context_profile_id", profile_id)
        object.__setattr__(
            self,
            "hiring_context_fingerprint",
            context_fingerprint,
        )
        object.__setattr__(self, "project_snapshots", ordered_snapshots)
        object.__setattr__(self, "state_fingerprint", expected_fingerprint)
        object.__setattr__(self, "state_id", expected_id)

    @property
    def project_count(self) -> int:
        return len(self.project_snapshots)

    @property
    def story_count(self) -> int:
        return sum(
            len(snapshot.story_relevance)
            for snapshot in self.project_snapshots
        )

    def snapshot_for_project(
        self,
        project_id: str,
    ) -> ProjectStoryRelevanceSnapshot | None:
        target = _exact_text(project_id, "project_id")
        return next(
            (
                snapshot
                for snapshot in self.project_snapshots
                if snapshot.project_id == target
            ),
            None,
        )

    def ranking_for_project(
        self,
        project_id: str,
    ) -> PortfolioProjectRanking | None:
        target = _exact_text(project_id, "project_id")
        if self.portfolio is None:
            return None
        return next(
            (
                ranking
                for ranking in self.portfolio.ranked_projects
                if ranking.project_id == target
            ),
            None,
        )

    def contribution_for_story(
        self,
        *,
        project_id: str,
        story_relevance_id: str,
    ) -> ProjectStoryContribution | None:
        snapshot = self.snapshot_for_project(project_id)
        if snapshot is None:
            return None
        target = _exact_text(story_relevance_id, "story_relevance_id")
        return next(
            (
                contribution
                for contribution in snapshot.project_relevance.contributions
                if contribution.story_relevance_id == target
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_policy_id": self.state_policy_id,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "portfolio": (
                None if self.portfolio is None else self.portfolio.to_dict()
            ),
            "project_snapshots": [
                snapshot.to_dict() for snapshot in self.project_snapshots
            ],
            "project_count": self.project_count,
            "story_count": self.story_count,
            "state_fingerprint": self.state_fingerprint,
            "state_id": self.state_id,
        }


def build_project_story_ranking_state(
    *,
    hiring_context: HiringContextProfile,
    portfolio: RankedProjectStoryPortfolio | None,
    project_snapshots: Sequence[ProjectStoryRelevanceSnapshot],
) -> ProjectStoryRankingState:
    """Bind one accepted portfolio and all complete project Story snapshots."""

    if not isinstance(hiring_context, HiringContextProfile):
        raise TypeError("hiring_context must be HiringContextProfile")
    _validate_contract(hiring_context, "Hiring Context")
    if portfolio is not None and not isinstance(
        portfolio,
        RankedProjectStoryPortfolio,
    ):
        raise TypeError("portfolio must be a ranked portfolio or None")
    if isinstance(project_snapshots, (str, bytes)) or not isinstance(
        project_snapshots,
        Sequence,
    ):
        raise TypeError("project_snapshots must be a sequence")
    if len(project_snapshots) > MAX_RANKING_STATE_PROJECTS:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.BOUND_EXCEEDED,
            "project snapshots exceed the ranking-state project bound",
        )
    return ProjectStoryRankingState(
        state_policy_id=PROJECT_STORY_RANKING_STATE_POLICY_ID,
        hiring_context_profile_id=hiring_context.profile_id,
        hiring_context_fingerprint=hiring_context.fingerprint,
        portfolio=portfolio,
        project_snapshots=tuple(project_snapshots),
    )


def derive_story_clarification_handoffs(
    *,
    ranking_state: ProjectStoryRankingState,
) -> tuple[StoryClarificationHandoff, ...]:
    """Derive the existing clarification projection from one validated state."""

    if not isinstance(ranking_state, ProjectStoryRankingState):
        raise TypeError("ranking_state must be ProjectStoryRankingState")
    _validate_contract(ranking_state, "project Story ranking state")
    if ranking_state.portfolio is None:
        return ()
    try:
        return build_story_clarification_handoffs(
            portfolio=ranking_state.portfolio
        )
    except StoryClarificationHandoffError as error:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.STALE_PORTFOLIO,
            "ranking state cannot produce accepted clarification handoffs",
        ) from error


def _validate_refresh_result(
    refresh_result: PartialProjectStoryRerankResult,
) -> None:
    _validate_contract(refresh_result.delta, "partial refresh delta")
    if refresh_result.new_portfolio is not None:
        _validate_portfolio(refresh_result.new_portfolio)
    if refresh_result.updated_project_snapshot is not None:
        _validate_contract(
            refresh_result.updated_project_snapshot,
            "updated project Story snapshot",
        )
    for handoff in refresh_result.updated_handoffs:
        _validate_contract(handoff, "updated Story clarification handoff")
    _validate_contract(refresh_result, "partial project Story rerank result")
    expected_handoffs = (
        ()
        if refresh_result.new_portfolio is None
        else build_story_clarification_handoffs(
            portfolio=refresh_result.new_portfolio
        )
    )
    if refresh_result.updated_handoffs != expected_handoffs:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh handoffs do not match its portfolio generation",
        )


def _prior_refresh_sources(
    *,
    prior_state: ProjectStoryRankingState,
    refresh_result: PartialProjectStoryRerankResult,
) -> tuple[ProjectStoryRelevanceSnapshot, Any, Any]:
    delta = refresh_result.delta
    if prior_state.portfolio is None:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "an empty ranking state has no existing Story to refresh",
        )
    if delta.old_portfolio_id != prior_state.portfolio.portfolio_id:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh result does not originate from the prior ranking state",
        )
    prior_snapshot = prior_state.snapshot_for_project(delta.affected_project_id)
    if prior_snapshot is None or prior_snapshot.snapshot_id != delta.old_snapshot_id:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh result does not identify the accepted prior snapshot",
        )
    old_story = next(
        (
            story
            for story in prior_snapshot.story_relevance
            if story.canonical_story_id == delta.canonical_story_id
        ),
        None,
    )
    old_ranking = next(
        (
            ranking
            for ranking in prior_state.portfolio.ranked_projects
            if ranking.project_id == delta.affected_project_id
        ),
        None,
    )
    if old_story is None or old_ranking is None:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh result does not identify accepted prior ranking objects",
        )
    if (
        delta.old_revision_id != old_story.current_revision_id
        or delta.old_story_relevance_id != old_story.relevance_id
        or delta.old_story_relevance != old_story.total_relevance_score
        or delta.old_project_relevance_id
        != prior_snapshot.project_relevance.project_relevance_id
        or delta.old_project_relevance
        != prior_snapshot.project_relevance.aggregate_relevance_score
        or delta.old_position != old_ranking.position
        or delta.old_portfolio_relevance != old_ranking.portfolio_relevance
    ):
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh delta old state does not match the accepted prior state",
        )
    return prior_snapshot, old_story, old_ranking


def _validate_refresh_topology(
    *,
    prior_state: ProjectStoryRankingState,
    refresh_result: PartialProjectStoryRerankResult,
    prior_snapshot: ProjectStoryRelevanceSnapshot,
) -> None:
    status = refresh_result.status
    delta = refresh_result.delta
    old_projects = {snapshot.project_id for snapshot in prior_state.project_snapshots}
    new_portfolio = refresh_result.new_portfolio
    new_projects = (
        set()
        if new_portfolio is None
        else {ranking.project_id for ranking in new_portfolio.ranked_projects}
    )
    expected_projects = (
        old_projects - {delta.affected_project_id}
        if status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED
        else old_projects
    )
    if new_projects != expected_projects:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "partial refresh cannot add or alter unrelated project membership",
        )
    old_project_by_id = {
        snapshot.project_id: snapshot.project_relevance
        for snapshot in prior_state.project_snapshots
    }
    if new_portfolio is not None:
        for ranking in new_portfolio.ranked_projects:
            if ranking.project_id == delta.affected_project_id:
                continue
            if ranking.project_relevance != old_project_by_id[ranking.project_id]:
                raise ProjectStoryRankingStateError(
                    ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                    "partial refresh changed an unrelated project relevance",
                )
    old_stories = {
        story.canonical_story_id: story
        for story in prior_snapshot.story_relevance
    }
    new_snapshot = refresh_result.updated_project_snapshot
    if status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED:
        if set(old_stories) != {delta.canonical_story_id}:
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                "project removal requires the affected Story to be its last Story",
            )
        return
    if new_snapshot is None:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "a retained project requires its refreshed complete snapshot",
        )
    new_ranking = next(
        (
            ranking
            for ranking in new_portfolio.ranked_projects
            if ranking.project_id == delta.affected_project_id
        ),
        None,
    )
    if new_ranking is None:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "retained affected project is absent from the new portfolio",
        )
    if (
        delta.new_snapshot_id != new_snapshot.snapshot_id
        or delta.new_project_relevance_id
        != new_snapshot.project_relevance.project_relevance_id
        or delta.new_project_relevance
        != new_snapshot.project_relevance.aggregate_relevance_score
        or delta.new_position != new_ranking.position
        or delta.new_portfolio_relevance != new_ranking.portfolio_relevance
    ):
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "refresh delta new project state does not match its snapshot",
        )
    new_stories = {
        story.canonical_story_id: story
        for story in new_snapshot.story_relevance
    }
    expected_story_ids = (
        set(old_stories) - {delta.canonical_story_id}
        if status is ProjectStoryRankingRefreshStatus.STORY_REMOVED
        else set(old_stories)
    )
    if set(new_stories) != expected_story_ids:
        raise ProjectStoryRankingStateError(
            ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
            "partial refresh changed unrelated Story membership",
        )
    if status is ProjectStoryRankingRefreshStatus.RERANKED:
        new_story = new_stories[delta.canonical_story_id]
        if (
            delta.new_revision_id != new_story.current_revision_id
            or delta.new_story_relevance_id != new_story.relevance_id
            or delta.new_story_relevance != new_story.total_relevance_score
        ):
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                "refresh delta new Story state does not match its snapshot",
            )
    for canonical_story_id in expected_story_ids - {delta.canonical_story_id}:
        if new_stories[canonical_story_id] != old_stories[canonical_story_id]:
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                "partial refresh changed unrelated Story relevance",
            )


def _rebind_snapshot(
    snapshot: ProjectStoryRelevanceSnapshot,
    portfolio_id: str,
) -> ProjectStoryRelevanceSnapshot:
    if snapshot.source_portfolio_id == portfolio_id:
        return snapshot
    return replace(
        snapshot,
        source_portfolio_id=portfolio_id,
        snapshot_id="",
    )


def update_project_story_ranking_state(
    *,
    prior_state: ProjectStoryRankingState,
    refresh_result: PartialProjectStoryRerankResult,
) -> ProjectStoryRankingState:
    """Bind an accepted partial refresh without performing semantic ranking."""

    if not isinstance(prior_state, ProjectStoryRankingState):
        raise TypeError("prior_state must be ProjectStoryRankingState")
    if not isinstance(refresh_result, PartialProjectStoryRerankResult):
        raise TypeError(
            "refresh_result must be PartialProjectStoryRerankResult"
        )
    _validate_contract(prior_state, "prior project Story ranking state")
    _validate_refresh_result(refresh_result)
    prior_snapshot, _, _ = _prior_refresh_sources(
        prior_state=prior_state,
        refresh_result=refresh_result,
    )
    if refresh_result.status is ProjectStoryRankingRefreshStatus.NO_CHANGE:
        if (
            refresh_result.new_portfolio != prior_state.portfolio
            or refresh_result.updated_project_snapshot != prior_snapshot
        ):
            raise ProjectStoryRankingStateError(
                ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                "no-change refresh must preserve prior ranking state",
            )
        return prior_state
    _validate_refresh_topology(
        prior_state=prior_state,
        refresh_result=refresh_result,
        prior_snapshot=prior_snapshot,
    )
    new_portfolio = refresh_result.new_portfolio
    if new_portfolio is None:
        new_snapshots: tuple[ProjectStoryRelevanceSnapshot, ...] = ()
    else:
        prior_by_project = {
            snapshot.project_id: snapshot
            for snapshot in prior_state.project_snapshots
        }
        new_snapshots_list: list[ProjectStoryRelevanceSnapshot] = []
        for ranking in new_portfolio.ranked_projects:
            if ranking.project_id == refresh_result.delta.affected_project_id:
                snapshot = refresh_result.updated_project_snapshot
                if snapshot is None:
                    raise ProjectStoryRankingStateError(
                        ProjectStoryRankingStateErrorCode.REFRESH_MISMATCH,
                        "retained affected project has no refreshed snapshot",
                    )
            else:
                snapshot = prior_by_project[ranking.project_id]
            new_snapshots_list.append(
                _rebind_snapshot(snapshot, new_portfolio.portfolio_id)
            )
        new_snapshots = tuple(new_snapshots_list)
    return ProjectStoryRankingState(
        state_policy_id=PROJECT_STORY_RANKING_STATE_POLICY_ID,
        hiring_context_profile_id=prior_state.hiring_context_profile_id,
        hiring_context_fingerprint=prior_state.hiring_context_fingerprint,
        portfolio=new_portfolio,
        project_snapshots=new_snapshots,
    )


__all__ = [
    "MAX_RANKING_STATE_PROJECTS",
    "MAX_RANKING_STATE_STORIES",
    "MAX_RANKING_STATE_STORIES_PER_PROJECT",
    "PROJECT_STORY_RANKING_STATE_POLICY_ID",
    "ProjectStoryRankingState",
    "ProjectStoryRankingStateError",
    "ProjectStoryRankingStateErrorCode",
    "build_project_story_ranking_state",
    "derive_story_clarification_handoffs",
    "update_project_story_ranking_state",
]
