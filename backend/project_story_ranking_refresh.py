"""Pure incremental refresh of accepted Story, project, and portfolio ranking.

The caller supplies one authoritative updated Story View and the complete
accepted Story-relevance snapshot for its existing project.  This module does
not read or write Story Memory, reconstruct Story truth, or make resume,
clarification, retrieval, or budgeting decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Never

from backend.engineering_story_memory_service import EngineeringStoryView
from backend.engineering_story_models import EngineeringStoryStatus
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    evaluate_engineering_story_relevance,
)
from backend.hiring_context_models import HiringContextProfile
from backend.project_portfolio_ranking import (
    MAX_PORTFOLIO_PROJECTS,
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
from backend.story_clarification_handoff import (
    MAX_STORY_CLARIFICATION_HANDOFFS,
    StoryClarificationHandoff,
    StoryClarificationHandoffError,
    build_story_clarification_handoffs,
)


PROJECT_STORY_RANKING_REFRESH_POLICY_ID = (
    "project_story_ranking_refresh.single_story.v1"
)
PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID = (
    "project_story_relevance_snapshot.complete_project.v1"
)
MAX_PORTFOLIO_RANKING_DELTA_CHANGES = 4
_REVISION_ID_RE = re.compile(r"engineering_story_revision_[0-9a-f]{24}")


class ProjectStoryRankingRefreshStatus(str, Enum):
    NO_CHANGE = "no_change"
    RERANKED = "reranked"
    STORY_REMOVED = "story_removed"
    PROJECT_REMOVED = "project_removed"


class PortfolioRankingChange(str, Enum):
    NO_CHANGE = "no_change"
    STORY_RELEVANCE_CHANGED = "story_relevance_changed"
    STORY_REMOVED = "story_removed"
    PROJECT_RELEVANCE_CHANGED = "project_relevance_changed"
    PROJECT_REMOVED = "project_removed"
    PORTFOLIO_POSITION_CHANGED = "portfolio_position_changed"
    PORTFOLIO_RELEVANCE_CHANGED = "portfolio_relevance_changed"


class ProjectStoryRankingRefreshErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    MIXED_HIRING_CONTEXT = "mixed_hiring_context"
    PROJECT_NOT_FOUND = "project_not_found"
    STORY_NOT_FOUND = "story_not_found"
    IDENTITY_MISMATCH = "identity_mismatch"
    SAME_REVISION_CONFLICT = "same_revision_conflict"
    STALE_PORTFOLIO = "stale_portfolio"
    CONFLICTING_STORY_SNAPSHOT = "conflicting_story_snapshot"


class ProjectStoryRankingRefreshError(ValueError):
    def __init__(
        self,
        code: ProjectStoryRankingRefreshErrorCode,
        message: str,
    ) -> None:
        self.code = ProjectStoryRankingRefreshErrorCode(code)
        super().__init__(message)


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _exact_text(value, name)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite score from 0 to 1")
    return normalized


def _optional_score(value: Any, name: str) -> float | None:
    return None if value is None else _score(value, name)


def _position(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_PORTFOLIO_PROJECTS
    ):
        raise ValueError(f"{name} must be within the portfolio project bound")
    return value


def _optional_position(value: Any, name: str) -> int | None:
    return None if value is None else _position(value, name)


def _digest(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}{digest[:24]}"


def _raise_source_error(name: str, error: Exception) -> Never:
    raise ProjectStoryRankingRefreshError(
        ProjectStoryRankingRefreshErrorCode.INVALID_INPUT,
        f"{name} no longer satisfies its immutable contract",
    ) from error


def _validate_contract(value: Any, name: str) -> None:
    try:
        validated = replace(value)
    except (TypeError, ValueError) as error:
        _raise_source_error(name, error)
    if validated != value:
        _raise_source_error(
            name,
            ValueError("normalization changed the accepted source"),
        )


def _story_order(story: StoryHiringRelevance) -> tuple[Any, ...]:
    return (
        -story.total_relevance_score,
        story.canonical_story_id,
        story.current_revision_id,
        story.relevance_id,
    )


@dataclass(frozen=True, slots=True)
class ProjectStoryRelevanceSnapshot:
    """Complete accepted Story relevance state for one ranked project."""

    snapshot_policy_id: str
    source_portfolio_id: str
    project_relevance: ProjectHiringRelevance
    story_relevance: tuple[StoryHiringRelevance, ...]
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        policy_id = _exact_text(self.snapshot_policy_id, "snapshot_policy_id")
        if policy_id != PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID:
            raise ValueError("snapshot_policy_id is not supported")
        source_portfolio_id = _exact_text(
            self.source_portfolio_id,
            "source_portfolio_id",
        )
        if not isinstance(self.project_relevance, ProjectHiringRelevance):
            raise TypeError("project_relevance must be ProjectHiringRelevance")
        if isinstance(self.story_relevance, (str, bytes)) or not isinstance(
            self.story_relevance,
            Sequence,
        ):
            raise TypeError("story_relevance must be a sequence")
        if not 1 <= len(self.story_relevance) <= MAX_PROJECT_STORY_INPUTS:
            raise ProjectStoryRankingRefreshError(
                ProjectStoryRankingRefreshErrorCode.BOUND_EXCEEDED,
                "project Story relevance snapshot exceeds its accepted bound",
            )
        project = self.project_relevance
        for contribution in project.contributions:
            _validate_contract(
                contribution.story_relevance,
                "snapshot contributing Story relevance",
            )
            _validate_contract(
                contribution,
                "snapshot project Story contribution",
            )
        _validate_contract(project, "snapshot project relevance")
        unique: dict[str, StoryHiringRelevance] = {}
        revision_owners: dict[str, str] = {}
        relevance_owners: dict[str, tuple[str, str]] = {}
        for story in self.story_relevance:
            if not isinstance(story, StoryHiringRelevance):
                raise TypeError(
                    "story_relevance must contain StoryHiringRelevance values"
                )
            _validate_contract(story, "snapshot Story relevance")
            if story.lifecycle_status is not EngineeringStoryStatus.ACTIVE:
                raise ValueError("snapshot Stories must be active relevance results")
            if story.project_id != project.project_id:
                raise ProjectStoryRankingRefreshError(
                    ProjectStoryRankingRefreshErrorCode.IDENTITY_MISMATCH,
                    "snapshot Story project does not match project relevance",
                )
            if (
                story.hiring_context_profile_id
                != project.hiring_context_profile_id
                or story.hiring_context_fingerprint
                != project.hiring_context_fingerprint
            ):
                raise ProjectStoryRankingRefreshError(
                    ProjectStoryRankingRefreshErrorCode.MIXED_HIRING_CONTEXT,
                    "snapshot Story Hiring Context does not match project relevance",
                )
            revision_owner = revision_owners.get(story.current_revision_id)
            if (
                revision_owner is not None
                and revision_owner != story.canonical_story_id
            ):
                raise ProjectStoryRankingRefreshError(
                    ProjectStoryRankingRefreshErrorCode.CONFLICTING_STORY_SNAPSHOT,
                    "one revision cannot belong to multiple canonical Stories",
                )
            revision_owners[story.current_revision_id] = story.canonical_story_id
            relevance_owner = relevance_owners.get(story.relevance_id)
            identity = (story.canonical_story_id, story.current_revision_id)
            if relevance_owner is not None and relevance_owner != identity:
                raise ProjectStoryRankingRefreshError(
                    ProjectStoryRankingRefreshErrorCode.CONFLICTING_STORY_SNAPSHOT,
                    "one relevance ID cannot describe multiple Story identities",
                )
            relevance_owners[story.relevance_id] = identity
            prior = unique.get(story.canonical_story_id)
            if prior is None:
                unique[story.canonical_story_id] = story
            elif prior != story:
                raise ProjectStoryRankingRefreshError(
                    ProjectStoryRankingRefreshErrorCode.CONFLICTING_STORY_SNAPSHOT,
                    "one canonical Story has conflicting accepted relevance",
                )
        stories = tuple(sorted(unique.values(), key=_story_order))
        if project.rankable_story_count != len(stories):
            raise ProjectStoryRankingRefreshError(
                ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO,
                "snapshot count does not match project relevance",
            )
        expected_contributors = stories[:MAX_CONTRIBUTING_STORIES]
        actual_contributors = tuple(
            item.story_relevance for item in project.contributions
        )
        if actual_contributors != expected_contributors:
            raise ProjectStoryRankingRefreshError(
                ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO,
                "snapshot top Stories do not match project contributions",
            )
        payload = {
            "snapshot_policy_id": policy_id,
            "source_portfolio_id": source_portfolio_id,
            "project_relevance_id": project.project_relevance_id,
            "story_relevance_ids": [item.relevance_id for item in stories],
        }
        expected_id = _digest("project_story_relevance_snapshot_", payload)
        if self.snapshot_id not in ("", expected_id):
            raise ValueError("snapshot_id does not match normalized snapshot")
        object.__setattr__(self, "snapshot_policy_id", policy_id)
        object.__setattr__(self, "source_portfolio_id", source_portfolio_id)
        object.__setattr__(self, "story_relevance", stories)
        object.__setattr__(self, "snapshot_id", expected_id)

    @property
    def project_id(self) -> str:
        return self.project_relevance.project_id

    @property
    def hiring_context_profile_id(self) -> str:
        return self.project_relevance.hiring_context_profile_id

    @property
    def hiring_context_fingerprint(self) -> str:
        return self.project_relevance.hiring_context_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_policy_id": self.snapshot_policy_id,
            "source_portfolio_id": self.source_portfolio_id,
            "project_id": self.project_id,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "project_relevance": self.project_relevance.to_dict(),
            "story_relevance": [item.to_dict() for item in self.story_relevance],
            "snapshot_id": self.snapshot_id,
        }


def _expected_changes(
    *,
    status: ProjectStoryRankingRefreshStatus,
    old_story_relevance_id: str,
    new_story_relevance_id: str | None,
    old_project_relevance_id: str,
    new_project_relevance_id: str | None,
    old_position: int,
    new_position: int | None,
    old_portfolio_relevance: float,
    new_portfolio_relevance: float | None,
) -> tuple[PortfolioRankingChange, ...]:
    if status is ProjectStoryRankingRefreshStatus.NO_CHANGE:
        return (PortfolioRankingChange.NO_CHANGE,)
    changes: set[PortfolioRankingChange] = set()
    if new_story_relevance_id is None:
        changes.add(PortfolioRankingChange.STORY_REMOVED)
    elif new_story_relevance_id != old_story_relevance_id:
        changes.add(PortfolioRankingChange.STORY_RELEVANCE_CHANGED)
    if new_project_relevance_id is None:
        changes.add(PortfolioRankingChange.PROJECT_REMOVED)
    elif new_project_relevance_id != old_project_relevance_id:
        changes.add(PortfolioRankingChange.PROJECT_RELEVANCE_CHANGED)
    if new_position != old_position:
        changes.add(PortfolioRankingChange.PORTFOLIO_POSITION_CHANGED)
    if new_portfolio_relevance != old_portfolio_relevance:
        changes.add(PortfolioRankingChange.PORTFOLIO_RELEVANCE_CHANGED)
    return tuple(item for item in PortfolioRankingChange if item in changes)


@dataclass(frozen=True, slots=True)
class PortfolioRankingDelta:
    status: ProjectStoryRankingRefreshStatus
    affected_project_id: str
    canonical_story_id: str
    old_snapshot_id: str
    new_snapshot_id: str | None
    old_revision_id: str
    new_revision_id: str
    old_story_relevance_id: str
    new_story_relevance_id: str | None
    old_story_relevance: float
    new_story_relevance: float | None
    old_project_relevance_id: str
    new_project_relevance_id: str | None
    old_project_relevance: float
    new_project_relevance: float | None
    old_portfolio_id: str
    new_portfolio_id: str | None
    old_position: int
    new_position: int | None
    old_portfolio_relevance: float
    new_portfolio_relevance: float | None
    changes: tuple[PortfolioRankingChange, ...]
    delta_id: str = ""

    def __post_init__(self) -> None:
        status = ProjectStoryRankingRefreshStatus(self.status)
        text_fields = {
            "affected_project_id": self.affected_project_id,
            "canonical_story_id": self.canonical_story_id,
            "old_snapshot_id": self.old_snapshot_id,
            "old_revision_id": self.old_revision_id,
            "new_revision_id": self.new_revision_id,
            "old_story_relevance_id": self.old_story_relevance_id,
            "old_project_relevance_id": self.old_project_relevance_id,
            "old_portfolio_id": self.old_portfolio_id,
        }
        normalized_text = {
            name: _exact_text(value, name)
            for name, value in text_fields.items()
        }
        optional_text_fields = {
            "new_snapshot_id": self.new_snapshot_id,
            "new_story_relevance_id": self.new_story_relevance_id,
            "new_project_relevance_id": self.new_project_relevance_id,
            "new_portfolio_id": self.new_portfolio_id,
        }
        normalized_optional_text = {
            name: _optional_text(value, name)
            for name, value in optional_text_fields.items()
        }
        score_fields = {
            "old_story_relevance": self.old_story_relevance,
            "old_project_relevance": self.old_project_relevance,
            "old_portfolio_relevance": self.old_portfolio_relevance,
        }
        normalized_scores = {
            name: _score(value, name)
            for name, value in score_fields.items()
        }
        optional_score_fields = {
            "new_story_relevance": self.new_story_relevance,
            "new_project_relevance": self.new_project_relevance,
            "new_portfolio_relevance": self.new_portfolio_relevance,
        }
        normalized_optional_scores = {
            name: _optional_score(value, name)
            for name, value in optional_score_fields.items()
        }
        old_position = _position(self.old_position, "old_position")
        new_position = _optional_position(self.new_position, "new_position")
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be a tuple")
        if len(self.changes) > MAX_PORTFOLIO_RANKING_DELTA_CHANGES:
            raise ValueError("changes exceeds the portfolio delta bound")
        if any(not isinstance(item, PortfolioRankingChange) for item in self.changes):
            raise TypeError("changes must contain PortfolioRankingChange values")
        changes = _expected_changes(
            status=status,
            old_story_relevance_id=normalized_text["old_story_relevance_id"],
            new_story_relevance_id=(
                normalized_optional_text["new_story_relevance_id"]
            ),
            old_project_relevance_id=normalized_text["old_project_relevance_id"],
            new_project_relevance_id=(
                normalized_optional_text["new_project_relevance_id"]
            ),
            old_position=old_position,
            new_position=new_position,
            old_portfolio_relevance=(
                normalized_scores["old_portfolio_relevance"]
            ),
            new_portfolio_relevance=(
                normalized_optional_scores["new_portfolio_relevance"]
            ),
        )
        if self.changes != changes:
            raise ValueError("changes must match the old and new ranking state")
        if status is ProjectStoryRankingRefreshStatus.NO_CHANGE:
            if (
                normalized_text["old_revision_id"]
                != normalized_text["new_revision_id"]
                or normalized_optional_text["new_snapshot_id"]
                != normalized_text["old_snapshot_id"]
                or normalized_optional_text["new_story_relevance_id"]
                != normalized_text["old_story_relevance_id"]
                or normalized_optional_text["new_project_relevance_id"]
                != normalized_text["old_project_relevance_id"]
                or normalized_optional_text["new_portfolio_id"]
                != normalized_text["old_portfolio_id"]
                or new_position != old_position
                or normalized_optional_scores["new_story_relevance"]
                != normalized_scores["old_story_relevance"]
                or normalized_optional_scores["new_project_relevance"]
                != normalized_scores["old_project_relevance"]
                or normalized_optional_scores["new_portfolio_relevance"]
                != normalized_scores["old_portfolio_relevance"]
            ):
                raise ValueError("no-change delta must preserve every identity")
        else:
            if normalized_text["old_revision_id"] == normalized_text["new_revision_id"]:
                raise ValueError("a changed refresh requires a new revision identity")
            new_snapshot_id = normalized_optional_text["new_snapshot_id"]
            new_story_id = normalized_optional_text["new_story_relevance_id"]
            new_project_id = normalized_optional_text["new_project_relevance_id"]
            new_portfolio_id = normalized_optional_text["new_portfolio_id"]
            new_story_score = normalized_optional_scores["new_story_relevance"]
            new_project_score = normalized_optional_scores["new_project_relevance"]
            new_portfolio_score = normalized_optional_scores[
                "new_portfolio_relevance"
            ]
            if status is ProjectStoryRankingRefreshStatus.RERANKED and any(
                value is None
                for value in (
                    new_snapshot_id,
                    new_story_id,
                    new_project_id,
                    new_portfolio_id,
                    new_story_score,
                    new_project_score,
                    new_position,
                    new_portfolio_score,
                )
            ):
                raise ValueError("reranked delta requires complete new state")
            if status is ProjectStoryRankingRefreshStatus.STORY_REMOVED and (
                new_story_id is not None
                or new_story_score is not None
                or any(
                    value is None
                    for value in (
                        new_snapshot_id,
                        new_project_id,
                        new_portfolio_id,
                        new_project_score,
                        new_position,
                        new_portfolio_score,
                    )
                )
            ):
                raise ValueError(
                    "Story-removed delta must retain only project ranking state"
                )
            if status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED and any(
                value is not None
                for value in (
                    new_snapshot_id,
                    new_story_id,
                    new_project_id,
                    new_story_score,
                    new_project_score,
                    new_position,
                    new_portfolio_score,
                )
            ):
                raise ValueError("project-removed delta cannot retain project state")
            if not changes:
                raise ValueError("a changed refresh requires at least one delta change")
        payload = {
            "status": status.value,
            **normalized_text,
            **normalized_optional_text,
            **normalized_scores,
            **normalized_optional_scores,
            "old_position": old_position,
            "new_position": new_position,
            "changes": [item.value for item in changes],
        }
        expected_id = _digest("portfolio_ranking_delta_", payload)
        if self.delta_id not in ("", expected_id):
            raise ValueError("delta_id does not match normalized ranking delta")
        object.__setattr__(self, "status", status)
        for name, value in normalized_text.items():
            object.__setattr__(self, name, value)
        for name, value in normalized_optional_text.items():
            object.__setattr__(self, name, value)
        for name, value in normalized_scores.items():
            object.__setattr__(self, name, value)
        for name, value in normalized_optional_scores.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "old_position", old_position)
        object.__setattr__(self, "new_position", new_position)
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "delta_id", expected_id)

    @property
    def changed(self) -> bool:
        return self.status is not ProjectStoryRankingRefreshStatus.NO_CHANGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "affected_project_id": self.affected_project_id,
            "canonical_story_id": self.canonical_story_id,
            "old_snapshot_id": self.old_snapshot_id,
            "new_snapshot_id": self.new_snapshot_id,
            "old_revision_id": self.old_revision_id,
            "new_revision_id": self.new_revision_id,
            "old_story_relevance_id": self.old_story_relevance_id,
            "new_story_relevance_id": self.new_story_relevance_id,
            "old_story_relevance": self.old_story_relevance,
            "new_story_relevance": self.new_story_relevance,
            "old_project_relevance_id": self.old_project_relevance_id,
            "new_project_relevance_id": self.new_project_relevance_id,
            "old_project_relevance": self.old_project_relevance,
            "new_project_relevance": self.new_project_relevance,
            "old_portfolio_id": self.old_portfolio_id,
            "new_portfolio_id": self.new_portfolio_id,
            "old_position": self.old_position,
            "new_position": self.new_position,
            "old_portfolio_relevance": self.old_portfolio_relevance,
            "new_portfolio_relevance": self.new_portfolio_relevance,
            "changed": self.changed,
            "changes": [item.value for item in self.changes],
            "delta_id": self.delta_id,
        }


@dataclass(frozen=True, slots=True)
class PartialProjectStoryRerankResult:
    refresh_policy_id: str
    status: ProjectStoryRankingRefreshStatus
    new_portfolio: RankedProjectStoryPortfolio | None
    updated_project_snapshot: ProjectStoryRelevanceSnapshot | None
    delta: PortfolioRankingDelta
    updated_handoffs: tuple[StoryClarificationHandoff, ...]
    result_id: str = ""

    def __post_init__(self) -> None:
        policy_id = _exact_text(self.refresh_policy_id, "refresh_policy_id")
        if policy_id != PROJECT_STORY_RANKING_REFRESH_POLICY_ID:
            raise ValueError("refresh_policy_id is not supported")
        status = ProjectStoryRankingRefreshStatus(self.status)
        if not isinstance(self.delta, PortfolioRankingDelta):
            raise TypeError("delta must be PortfolioRankingDelta")
        if self.delta.status is not status:
            raise ValueError("result status must match delta status")
        if self.new_portfolio is not None and not isinstance(
            self.new_portfolio,
            RankedProjectStoryPortfolio,
        ):
            raise TypeError("new_portfolio must be a ranked portfolio or None")
        if self.updated_project_snapshot is not None and not isinstance(
            self.updated_project_snapshot,
            ProjectStoryRelevanceSnapshot,
        ):
            raise TypeError(
                "updated_project_snapshot must be a project snapshot or None"
            )
        if not isinstance(self.updated_handoffs, tuple):
            raise TypeError("updated_handoffs must be a tuple")
        if len(self.updated_handoffs) > MAX_STORY_CLARIFICATION_HANDOFFS:
            raise ValueError("updated_handoffs exceeds the accepted handoff bound")
        if any(
            not isinstance(item, StoryClarificationHandoff)
            for item in self.updated_handoffs
        ):
            raise TypeError(
                "updated_handoffs must contain StoryClarificationHandoff values"
            )
        portfolio_id = (
            None if self.new_portfolio is None else self.new_portfolio.portfolio_id
        )
        if portfolio_id != self.delta.new_portfolio_id:
            raise ValueError("new portfolio identity must match the delta")
        if self.new_portfolio is None:
            if self.updated_handoffs:
                raise ValueError("an empty portfolio cannot have Story handoffs")
        elif any(
            item.portfolio_id != portfolio_id for item in self.updated_handoffs
        ):
            raise ValueError("updated handoffs must match the new portfolio")
        if status is ProjectStoryRankingRefreshStatus.PROJECT_REMOVED:
            if self.updated_project_snapshot is not None:
                raise ValueError("a removed project cannot retain a snapshot")
        elif self.updated_project_snapshot is None:
            raise ValueError("a retained project requires an updated snapshot")
        if self.updated_project_snapshot is not None:
            if (
                self.updated_project_snapshot.project_id
                != self.delta.affected_project_id
                or self.updated_project_snapshot.snapshot_id
                != self.delta.new_snapshot_id
            ):
                raise ValueError("updated snapshot identity must match the delta")
        payload = {
            "refresh_policy_id": policy_id,
            "status": status.value,
            "new_portfolio_id": portfolio_id,
            "updated_project_snapshot_id": (
                None
                if self.updated_project_snapshot is None
                else self.updated_project_snapshot.snapshot_id
            ),
            "delta_id": self.delta.delta_id,
            "updated_handoff_ids": [
                item.handoff_id for item in self.updated_handoffs
            ],
        }
        expected_id = _digest("partial_project_story_rerank_", payload)
        if self.result_id not in ("", expected_id):
            raise ValueError("result_id does not match normalized refresh result")
        object.__setattr__(self, "refresh_policy_id", policy_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "result_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "refresh_policy_id": self.refresh_policy_id,
            "status": self.status.value,
            "new_portfolio": (
                None if self.new_portfolio is None else self.new_portfolio.to_dict()
            ),
            "updated_project_snapshot": (
                None
                if self.updated_project_snapshot is None
                else self.updated_project_snapshot.to_dict()
            ),
            "delta": self.delta.to_dict(),
            "updated_handoffs": [item.to_dict() for item in self.updated_handoffs],
            "result_id": self.result_id,
        }


def _find_project_ranking(
    portfolio: RankedProjectStoryPortfolio | None,
    project_id: str,
) -> PortfolioProjectRanking | None:
    if portfolio is None:
        return None
    return next(
        (
            item
            for item in portfolio.ranked_projects
            if item.project_id == project_id
        ),
        None,
    )


def _validate_updated_view(story_view: EngineeringStoryView) -> None:
    if not isinstance(story_view, EngineeringStoryView):
        raise TypeError("updated_story_view must be EngineeringStoryView")
    if not _REVISION_ID_RE.fullmatch(story_view.current_revision_id):
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.INVALID_INPUT,
            "updated Story View has an invalid current revision identity",
        )
    story = story_view.current_story
    if (
        story_view.evidence_fact_ids != story.evidence_fact_ids
        or story_view.capability_fact_ids != story.capability_fact_ids
        or story_view.claim_boundary_ids != story.claim_boundary_ids
    ):
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.INVALID_INPUT,
            "updated Story View authority references do not match its Story",
        )
    _validate_contract(story, "updated Engineering Story")
    _validate_contract(story_view, "updated Engineering Story View")


def _no_change_conflicts(
    old_story: StoryHiringRelevance,
    updated_story_view: EngineeringStoryView,
) -> bool:
    return (
        updated_story_view.provenance_fingerprint
        != old_story.story_provenance_fingerprint
        or updated_story_view.lifecycle.status is not old_story.lifecycle_status
        or updated_story_view.lifecycle.requires_revalidation
        or updated_story_view.claim_sufficiency.level
        is not old_story.claim_sufficiency
        or updated_story_view.story_sufficiency.level
        is not old_story.story_sufficiency
        or updated_story_view.opportunity.level is not old_story.story_opportunity
    )


def _build_delta(
    *,
    status: ProjectStoryRankingRefreshStatus,
    prior_portfolio: RankedProjectStoryPortfolio,
    new_portfolio: RankedProjectStoryPortfolio | None,
    prior_snapshot: ProjectStoryRelevanceSnapshot,
    new_snapshot: ProjectStoryRelevanceSnapshot | None,
    old_story: StoryHiringRelevance,
    new_story: StoryHiringRelevance | None,
    new_revision_id: str,
) -> PortfolioRankingDelta:
    old_ranking = _find_project_ranking(prior_portfolio, prior_snapshot.project_id)
    if old_ranking is None:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.PROJECT_NOT_FOUND,
            "affected project is absent from the prior portfolio",
        )
    new_ranking = _find_project_ranking(new_portfolio, prior_snapshot.project_id)
    new_project = None if new_snapshot is None else new_snapshot.project_relevance
    changes = _expected_changes(
        status=status,
        old_story_relevance_id=old_story.relevance_id,
        new_story_relevance_id=(None if new_story is None else new_story.relevance_id),
        old_project_relevance_id=(
            prior_snapshot.project_relevance.project_relevance_id
        ),
        new_project_relevance_id=(
            None if new_project is None else new_project.project_relevance_id
        ),
        old_position=old_ranking.position,
        new_position=None if new_ranking is None else new_ranking.position,
        old_portfolio_relevance=old_ranking.portfolio_relevance,
        new_portfolio_relevance=(
            None if new_ranking is None else new_ranking.portfolio_relevance
        ),
    )
    return PortfolioRankingDelta(
        status=status,
        affected_project_id=prior_snapshot.project_id,
        canonical_story_id=old_story.canonical_story_id,
        old_snapshot_id=prior_snapshot.snapshot_id,
        new_snapshot_id=(None if new_snapshot is None else new_snapshot.snapshot_id),
        old_revision_id=old_story.current_revision_id,
        new_revision_id=new_revision_id,
        old_story_relevance_id=old_story.relevance_id,
        new_story_relevance_id=(None if new_story is None else new_story.relevance_id),
        old_story_relevance=old_story.total_relevance_score,
        new_story_relevance=(
            None if new_story is None else new_story.total_relevance_score
        ),
        old_project_relevance_id=(
            prior_snapshot.project_relevance.project_relevance_id
        ),
        new_project_relevance_id=(
            None if new_project is None else new_project.project_relevance_id
        ),
        old_project_relevance=(
            prior_snapshot.project_relevance.aggregate_relevance_score
        ),
        new_project_relevance=(
            None if new_project is None else new_project.aggregate_relevance_score
        ),
        old_portfolio_id=prior_portfolio.portfolio_id,
        new_portfolio_id=(
            None if new_portfolio is None else new_portfolio.portfolio_id
        ),
        old_position=old_ranking.position,
        new_position=None if new_ranking is None else new_ranking.position,
        old_portfolio_relevance=old_ranking.portfolio_relevance,
        new_portfolio_relevance=(
            None if new_ranking is None else new_ranking.portfolio_relevance
        ),
        changes=changes,
    )


def refresh_project_story_ranking(
    *,
    hiring_context: HiringContextProfile,
    prior_portfolio: RankedProjectStoryPortfolio,
    prior_project_snapshot: ProjectStoryRelevanceSnapshot,
    updated_story_view: EngineeringStoryView,
) -> PartialProjectStoryRerankResult:
    """Refresh one existing canonical Story and reconcile its full portfolio."""

    if not isinstance(hiring_context, HiringContextProfile):
        raise TypeError("hiring_context must be HiringContextProfile")
    if not isinstance(prior_portfolio, RankedProjectStoryPortfolio):
        raise TypeError("prior_portfolio must be RankedProjectStoryPortfolio")
    if not isinstance(prior_project_snapshot, ProjectStoryRelevanceSnapshot):
        raise TypeError(
            "prior_project_snapshot must be ProjectStoryRelevanceSnapshot"
        )
    _validate_contract(hiring_context, "Hiring Context")
    _validate_contract(prior_project_snapshot, "prior project Story snapshot")
    _validate_updated_view(updated_story_view)
    try:
        prior_handoffs = build_story_clarification_handoffs(
            portfolio=prior_portfolio
        )
    except StoryClarificationHandoffError as error:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO,
            "prior portfolio failed accepted handoff identity validation",
        ) from error
    if (
        hiring_context.profile_id != prior_portfolio.hiring_context_profile_id
        or hiring_context.fingerprint
        != prior_portfolio.hiring_context_fingerprint
    ):
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.MIXED_HIRING_CONTEXT,
            "partial Story refresh requires the unchanged Hiring Context",
        )
    if (
        prior_project_snapshot.hiring_context_profile_id
        != hiring_context.profile_id
        or prior_project_snapshot.hiring_context_fingerprint
        != hiring_context.fingerprint
    ):
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.MIXED_HIRING_CONTEXT,
            "project Story snapshot does not match the Hiring Context",
        )
    old_ranking = _find_project_ranking(
        prior_portfolio,
        prior_project_snapshot.project_id,
    )
    if old_ranking is None:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.PROJECT_NOT_FOUND,
            "snapshot project is absent from the prior portfolio",
        )
    if old_ranking.project_relevance != prior_project_snapshot.project_relevance:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO,
            "snapshot project relevance does not match the prior portfolio",
        )
    if prior_project_snapshot.source_portfolio_id != prior_portfolio.portfolio_id:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.STALE_PORTFOLIO,
            "project Story snapshot does not match the prior portfolio identity",
        )
    if updated_story_view.project_id != prior_project_snapshot.project_id:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.IDENTITY_MISMATCH,
            "updated Story does not belong to the snapshot project",
        )
    old_by_id = {
        item.canonical_story_id: item
        for item in prior_project_snapshot.story_relevance
    }
    old_story = old_by_id.get(updated_story_view.canonical_story_id)
    if old_story is None:
        raise ProjectStoryRankingRefreshError(
            ProjectStoryRankingRefreshErrorCode.STORY_NOT_FOUND,
            "updated canonical Story is absent from the prior snapshot",
        )
    if updated_story_view.current_revision_id == old_story.current_revision_id:
        if _no_change_conflicts(old_story, updated_story_view):
            raise ProjectStoryRankingRefreshError(
                ProjectStoryRankingRefreshErrorCode.SAME_REVISION_CONFLICT,
                "one Story revision cannot carry conflicting authoritative state",
            )
        delta = _build_delta(
            status=ProjectStoryRankingRefreshStatus.NO_CHANGE,
            prior_portfolio=prior_portfolio,
            new_portfolio=prior_portfolio,
            prior_snapshot=prior_project_snapshot,
            new_snapshot=prior_project_snapshot,
            old_story=old_story,
            new_story=old_story,
            new_revision_id=old_story.current_revision_id,
        )
        return PartialProjectStoryRerankResult(
            refresh_policy_id=PROJECT_STORY_RANKING_REFRESH_POLICY_ID,
            status=ProjectStoryRankingRefreshStatus.NO_CHANGE,
            new_portfolio=prior_portfolio,
            updated_project_snapshot=prior_project_snapshot,
            delta=delta,
            updated_handoffs=prior_handoffs,
        )
    rankable = (
        updated_story_view.lifecycle.status is EngineeringStoryStatus.ACTIVE
        and not updated_story_view.lifecycle.requires_revalidation
    )
    new_story = (
        evaluate_engineering_story_relevance(
            hiring_context=hiring_context,
            story_view=updated_story_view,
        )
        if rankable
        else None
    )
    updated_stories = tuple(
        item
        for item in prior_project_snapshot.story_relevance
        if item.canonical_story_id != old_story.canonical_story_id
    ) + (() if new_story is None else (new_story,))
    candidate_project = aggregate_project_story_relevance(
        project_id=prior_project_snapshot.project_id,
        story_relevance=updated_stories,
    )
    if candidate_project == prior_project_snapshot.project_relevance:
        candidate_project = prior_project_snapshot.project_relevance
    projects = tuple(
        item.project_relevance
        for item in prior_portfolio.ranked_projects
        if item.project_id != prior_project_snapshot.project_id
    ) + (() if candidate_project is None else (candidate_project,))
    reconciled_portfolio = rank_project_portfolio(projects=projects)
    if reconciled_portfolio == prior_portfolio:
        reconciled_portfolio = prior_portfolio
    new_snapshot = (
        None
        if candidate_project is None
        else ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=reconciled_portfolio.portfolio_id,
            project_relevance=candidate_project,
            story_relevance=updated_stories,
        )
    )
    updated_handoffs = (
        ()
        if reconciled_portfolio is None
        else build_story_clarification_handoffs(portfolio=reconciled_portfolio)
    )
    if candidate_project is None:
        status = ProjectStoryRankingRefreshStatus.PROJECT_REMOVED
    elif new_story is None:
        status = ProjectStoryRankingRefreshStatus.STORY_REMOVED
    else:
        status = ProjectStoryRankingRefreshStatus.RERANKED
    delta = _build_delta(
        status=status,
        prior_portfolio=prior_portfolio,
        new_portfolio=reconciled_portfolio,
        prior_snapshot=prior_project_snapshot,
        new_snapshot=new_snapshot,
        old_story=old_story,
        new_story=new_story,
        new_revision_id=updated_story_view.current_revision_id,
    )
    return PartialProjectStoryRerankResult(
        refresh_policy_id=PROJECT_STORY_RANKING_REFRESH_POLICY_ID,
        status=status,
        new_portfolio=reconciled_portfolio,
        updated_project_snapshot=new_snapshot,
        delta=delta,
        updated_handoffs=updated_handoffs,
    )


__all__ = [
    "MAX_PORTFOLIO_RANKING_DELTA_CHANGES",
    "PROJECT_STORY_RANKING_REFRESH_POLICY_ID",
    "PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID",
    "PartialProjectStoryRerankResult",
    "PortfolioRankingChange",
    "PortfolioRankingDelta",
    "ProjectStoryRankingRefreshError",
    "ProjectStoryRankingRefreshErrorCode",
    "ProjectStoryRankingRefreshStatus",
    "ProjectStoryRelevanceSnapshot",
    "refresh_project_story_ranking",
]
