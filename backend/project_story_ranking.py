"""Pure aggregation of accepted Story relevance into project relevance.

This module consumes immutable ``StoryHiringRelevance`` results.  It does not
load or reinterpret Engineering Story truth, evaluate hiring-context signals,
or perform portfolio differentiation, selection, budgeting, or runtime I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

from backend.engineering_story_relevance import (
    MAX_STORY_RELEVANCE_BATCH,
    StoryHiringRelevance,
    StoryRelevanceComponents,
)


PROJECT_RELEVANCE_SCORE_DECIMALS = 6
MAX_PROJECT_STORY_INPUTS = MAX_STORY_RELEVANCE_BATCH
MAX_CONTRIBUTING_STORIES = 3
MAX_PROJECT_RELEVANCE_REASONS = 6
DEPTH_QUALIFICATION_FLOOR = 0.50
PROJECT_STORY_POSITION_WEIGHTS = (1.00, 0.60, 0.20)
PROJECT_RELEVANCE_AGGREGATION_POLICY_ID = (
    "project_story_relevance.headroom_top3.v1"
)

_EXCEPTIONAL_STORY_THRESHOLD = 0.90
_STRONG_STORY_THRESHOLD = 0.75


class ProjectRelevanceReason(str, Enum):
    """Bounded, stable summaries of one project's aggregation result."""

    EXCEPTIONAL_TOP_STORY = "exceptional_top_story"
    MULTIPLE_STRONG_STORIES = "multiple_strong_stories"
    LIMITED_STORY_DEPTH = "limited_story_depth"
    STORY_COMPLETION_NEEDED = "story_completion_needed"
    CLAIM_RISK_PRESENT = "claim_risk_present"
    ADDITIONAL_STORIES_CAPPED = "additional_stories_capped"


class ProjectRelevanceAggregationErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    MIXED_PROJECT = "mixed_project"
    MIXED_HIRING_CONTEXT = "mixed_hiring_context"
    MIXED_HIRING_CONTEXT_FINGERPRINT = "mixed_hiring_context_fingerprint"
    INACTIVE_STORY = "inactive_story"
    CONFLICTING_STORY_REVISION = "conflicting_story_revision"
    CONFLICTING_STORY_RESULT = "conflicting_story_result"


class ProjectRelevanceAggregationError(ValueError):
    def __init__(
        self,
        code: ProjectRelevanceAggregationErrorCode,
        message: str,
    ) -> None:
        self.code = ProjectRelevanceAggregationErrorCode(code)
        super().__init__(message)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite score from 0 to 1")
    return round(normalized, PROJECT_RELEVANCE_SCORE_DECIMALS)


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _depth_factor(story_total_relevance: float) -> float:
    """Map only accepted relevance above the weak-Story floor into depth."""

    if story_total_relevance <= DEPTH_QUALIFICATION_FLOOR:
        return 0.0
    return _score(
        (story_total_relevance - DEPTH_QUALIFICATION_FLOOR)
        / (1.0 - DEPTH_QUALIFICATION_FLOOR),
        "normalized_depth",
    )


@dataclass(frozen=True, slots=True)
class ProjectStoryContribution:
    """One accepted Story result and its inspectable project contribution."""

    rank_position: int
    story_relevance: StoryHiringRelevance
    positional_weight: float
    normalized_depth: float
    available_headroom: float
    weighted_contribution: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank_position, bool)
            or not isinstance(self.rank_position, int)
            or not 1 <= self.rank_position <= MAX_CONTRIBUTING_STORIES
        ):
            raise ValueError(
                f"rank_position must be from 1 to {MAX_CONTRIBUTING_STORIES}"
            )
        if not isinstance(self.story_relevance, StoryHiringRelevance):
            raise TypeError("story_relevance must be StoryHiringRelevance")
        weight = _score(self.positional_weight, "positional_weight")
        normalized_depth = _score(self.normalized_depth, "normalized_depth")
        headroom = _score(self.available_headroom, "available_headroom")
        contribution = _score(
            self.weighted_contribution,
            "weighted_contribution",
        )
        expected_weight = PROJECT_STORY_POSITION_WEIGHTS[self.rank_position - 1]
        if weight != expected_weight:
            raise ValueError("positional_weight does not match aggregation policy")
        if self.rank_position == 1:
            expected_depth = self.story_relevance.total_relevance_score
            expected_headroom = 1.0
        else:
            expected_depth = _depth_factor(
                self.story_relevance.total_relevance_score
            )
            expected_headroom = headroom
        expected_contribution = _score(
            expected_headroom * weight * expected_depth,
            "expected weighted contribution",
        )
        if normalized_depth != expected_depth:
            raise ValueError("normalized_depth does not match aggregation policy")
        if headroom != expected_headroom and self.rank_position == 1:
            raise ValueError("the first Story must receive all available headroom")
        if contribution != expected_contribution:
            raise ValueError(
                "weighted_contribution does not match aggregation policy"
            )
        object.__setattr__(self, "positional_weight", weight)
        object.__setattr__(self, "normalized_depth", normalized_depth)
        object.__setattr__(self, "available_headroom", headroom)
        object.__setattr__(self, "weighted_contribution", contribution)

    @property
    def canonical_story_id(self) -> str:
        return self.story_relevance.canonical_story_id

    @property
    def current_revision_id(self) -> str:
        return self.story_relevance.current_revision_id

    @property
    def story_relevance_id(self) -> str:
        return self.story_relevance.relevance_id

    @property
    def story_provenance_fingerprint(self) -> str:
        return self.story_relevance.story_provenance_fingerprint

    @property
    def story_total_relevance(self) -> float:
        return self.story_relevance.total_relevance_score

    @property
    def story_components(self) -> StoryRelevanceComponents:
        return self.story_relevance.components

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_position": self.rank_position,
            "canonical_story_id": self.canonical_story_id,
            "current_revision_id": self.current_revision_id,
            "story_relevance_id": self.story_relevance_id,
            "story_provenance_fingerprint": self.story_provenance_fingerprint,
            "story_total_relevance": self.story_total_relevance,
            "story_components": self.story_components.to_dict(),
            "claim_sufficiency": self.story_relevance.claim_sufficiency.value,
            "story_sufficiency": self.story_relevance.story_sufficiency.value,
            "story_opportunity": self.story_relevance.story_opportunity.value,
            "clarification_value_hint": (
                self.story_relevance.clarification_value_hint
            ),
            "story_semantic_features": [
                item.value for item in self.story_relevance.semantic_features
            ],
            "story_reasons": [
                item.value for item in self.story_relevance.reasons
            ],
            "story_evidence_risk_adjustment": (
                self.story_relevance.evidence_risk_adjustment
            ),
            "positional_weight": self.positional_weight,
            "normalized_depth": self.normalized_depth,
            "available_headroom": self.available_headroom,
            "weighted_contribution": self.weighted_contribution,
        }


@dataclass(frozen=True, slots=True)
class ProjectRelevanceComponents:
    strongest_story_contribution: float
    secondary_story_depth: float
    aggregate_relevance: float

    def __post_init__(self) -> None:
        strongest = _score(
            self.strongest_story_contribution,
            "strongest_story_contribution",
        )
        secondary = _score(self.secondary_story_depth, "secondary_story_depth")
        aggregate = _score(self.aggregate_relevance, "aggregate_relevance")
        if aggregate != _score(
            strongest + secondary,
            "component aggregate relevance",
        ):
            raise ValueError(
                "aggregate_relevance must equal strongest plus secondary depth"
            )
        object.__setattr__(self, "strongest_story_contribution", strongest)
        object.__setattr__(self, "secondary_story_depth", secondary)
        object.__setattr__(self, "aggregate_relevance", aggregate)

    def to_dict(self) -> dict[str, float]:
        return {
            "strongest_story_contribution": self.strongest_story_contribution,
            "secondary_story_depth": self.secondary_story_depth,
            "aggregate_relevance": self.aggregate_relevance,
        }


@dataclass(frozen=True, slots=True)
class ProjectHiringRelevance:
    project_id: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    aggregation_policy_id: str
    rankable_story_count: int
    components: ProjectRelevanceComponents
    aggregate_relevance_score: float
    contributions: tuple[ProjectStoryContribution, ...]
    reasons: tuple[ProjectRelevanceReason, ...]
    project_relevance_id: str = ""

    def __post_init__(self) -> None:
        project_id = _exact_text(self.project_id, "project_id")
        profile_id = _exact_text(
            self.hiring_context_profile_id,
            "hiring_context_profile_id",
        )
        context_fingerprint = _exact_text(
            self.hiring_context_fingerprint,
            "hiring_context_fingerprint",
        )
        policy_id = _exact_text(self.aggregation_policy_id, "aggregation_policy_id")
        if policy_id != PROJECT_RELEVANCE_AGGREGATION_POLICY_ID:
            raise ValueError("aggregation_policy_id is not supported")
        if (
            isinstance(self.rankable_story_count, bool)
            or not isinstance(self.rankable_story_count, int)
            or not 1 <= self.rankable_story_count <= MAX_PROJECT_STORY_INPUTS
        ):
            raise ValueError(
                "rankable_story_count must be within the project Story bound"
            )
        if not isinstance(self.components, ProjectRelevanceComponents):
            raise TypeError("components must be ProjectRelevanceComponents")
        aggregate = _score(
            self.aggregate_relevance_score,
            "aggregate_relevance_score",
        )
        if aggregate != self.components.aggregate_relevance:
            raise ValueError(
                "aggregate_relevance_score must match project components"
            )
        if not isinstance(self.contributions, tuple) or not self.contributions:
            raise ValueError("contributions must be a non-empty tuple")
        if len(self.contributions) > MAX_CONTRIBUTING_STORIES:
            raise ValueError("contributions exceeds the aggregation cap")
        if self.rankable_story_count < len(self.contributions):
            raise ValueError("rankable_story_count cannot be below contributions")
        if any(
            not isinstance(item, ProjectStoryContribution)
            for item in self.contributions
        ):
            raise TypeError(
                "contributions must contain ProjectStoryContribution values"
            )
        if tuple(item.rank_position for item in self.contributions) != tuple(
            range(1, len(self.contributions) + 1)
        ):
            raise ValueError("contribution positions must be consecutive")
        if len({item.canonical_story_id for item in self.contributions}) != len(
            self.contributions
        ):
            raise ValueError("contributions must have unique canonical Story IDs")
        for item in self.contributions:
            story = item.story_relevance
            if story.project_id != project_id:
                raise ValueError("contribution project does not match project result")
            if story.hiring_context_profile_id != profile_id:
                raise ValueError("contribution context does not match project result")
            if story.hiring_context_fingerprint != context_fingerprint:
                raise ValueError(
                    "contribution context fingerprint does not match project result"
                )
        expected_order = tuple(sorted(
            self.contributions,
            key=lambda item: (
                -item.story_total_relevance,
                item.canonical_story_id,
                item.current_revision_id,
                item.story_relevance_id,
            ),
        ))
        if self.contributions != expected_order:
            raise ValueError("contributions are not in deterministic Story order")
        running_total = 0.0
        for item in self.contributions:
            if item.available_headroom != _score(
                1.0 - running_total,
                "expected available headroom",
            ):
                raise ValueError(
                    "available_headroom does not match prior contributions"
                )
            running_total = _score(
                running_total + item.weighted_contribution,
                "contribution total",
            )
        if running_total != aggregate:
            raise ValueError("contributions must sum to aggregate relevance")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        if len(self.reasons) > MAX_PROJECT_RELEVANCE_REASONS:
            raise ValueError("reasons exceeds the project reason bound")
        if any(not isinstance(item, ProjectRelevanceReason) for item in self.reasons):
            raise TypeError("reasons must contain ProjectRelevanceReason values")
        stable_reasons = tuple(
            item for item in ProjectRelevanceReason if item in set(self.reasons)
        )
        if self.reasons != stable_reasons:
            raise ValueError("reasons must be unique and in deterministic order")
        payload = {
            "project_id": project_id,
            "hiring_context_profile_id": profile_id,
            "hiring_context_fingerprint": context_fingerprint,
            "aggregation_policy_id": policy_id,
            "rankable_story_count": self.rankable_story_count,
            "components": self.components.to_dict(),
            "aggregate_relevance_score": aggregate,
            "contributions": [item.to_dict() for item in self.contributions],
            "reasons": [item.value for item in stable_reasons],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        expected_id = f"project_hiring_relevance_{digest[:24]}"
        if self.project_relevance_id not in ("", expected_id):
            raise ValueError(
                "project_relevance_id does not match normalized project result"
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "hiring_context_profile_id", profile_id)
        object.__setattr__(
            self,
            "hiring_context_fingerprint",
            context_fingerprint,
        )
        object.__setattr__(self, "aggregation_policy_id", policy_id)
        object.__setattr__(self, "aggregate_relevance_score", aggregate)
        object.__setattr__(self, "reasons", stable_reasons)
        object.__setattr__(self, "project_relevance_id", expected_id)

    @property
    def contributing_story_count(self) -> int:
        return len(self.contributions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "aggregation_policy_id": self.aggregation_policy_id,
            "rankable_story_count": self.rankable_story_count,
            "contributing_story_count": self.contributing_story_count,
            "components": self.components.to_dict(),
            "aggregate_relevance_score": self.aggregate_relevance_score,
            "contributions": [item.to_dict() for item in self.contributions],
            "reasons": [item.value for item in self.reasons],
            "project_relevance_id": self.project_relevance_id,
        }


def _validated_unique_stories(
    *,
    project_id: str | None,
    story_relevance: Sequence[StoryHiringRelevance],
) -> tuple[StoryHiringRelevance, ...]:
    if isinstance(story_relevance, (str, bytes)) or not isinstance(
        story_relevance,
        Sequence,
    ):
        raise TypeError("story_relevance must be a sequence")
    if len(story_relevance) > MAX_PROJECT_STORY_INPUTS:
        raise ProjectRelevanceAggregationError(
            ProjectRelevanceAggregationErrorCode.BOUND_EXCEEDED,
            "story_relevance exceeds maximum item count "
            f"{MAX_PROJECT_STORY_INPUTS}",
        )
    target_project = (
        _exact_text(project_id, "project_id") if project_id is not None else None
    )
    unique: dict[tuple[str, str], StoryHiringRelevance] = {}
    context_profile_id: str | None = None
    context_fingerprint: str | None = None
    revision_owners: dict[str, str] = {}
    for story in story_relevance:
        if not isinstance(story, StoryHiringRelevance):
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.INVALID_INPUT,
                "story_relevance must contain StoryHiringRelevance values",
            )
        if story.lifecycle_status.value != "active":
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.INACTIVE_STORY,
                "only active accepted Story relevance results are rankable",
            )
        if target_project is not None and story.project_id != target_project:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.MIXED_PROJECT,
                "all Story relevance results must match the exact project_id",
            )
        if context_profile_id is None:
            context_profile_id = story.hiring_context_profile_id
            context_fingerprint = story.hiring_context_fingerprint
        elif story.hiring_context_profile_id != context_profile_id:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.MIXED_HIRING_CONTEXT,
                "Story relevance results cannot mix Hiring Context profiles",
            )
        elif story.hiring_context_fingerprint != context_fingerprint:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.MIXED_HIRING_CONTEXT_FINGERPRINT,
                "Story relevance results cannot mix Hiring Context fingerprints",
            )
        revision_owner = revision_owners.get(story.current_revision_id)
        if revision_owner is not None and revision_owner != story.canonical_story_id:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_REVISION,
                "one Story revision cannot belong to multiple canonical Stories",
            )
        revision_owners[story.current_revision_id] = story.canonical_story_id
        identity = (story.project_id, story.canonical_story_id)
        prior = unique.get(identity)
        if prior is None:
            unique[identity] = story
        elif prior.current_revision_id != story.current_revision_id:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_REVISION,
                "one canonical Story cannot have conflicting current revisions",
            )
        elif prior != story:
            raise ProjectRelevanceAggregationError(
                ProjectRelevanceAggregationErrorCode.CONFLICTING_STORY_RESULT,
                "one Story revision cannot have conflicting relevance results",
            )
    return tuple(sorted(
        unique.values(),
        key=lambda item: (
            -item.total_relevance_score,
            item.canonical_story_id,
            item.current_revision_id,
            item.relevance_id,
        ),
    ))


def _project_reasons(
    *,
    stories: tuple[StoryHiringRelevance, ...],
    contributions: tuple[ProjectStoryContribution, ...],
) -> tuple[ProjectRelevanceReason, ...]:
    reasons: set[ProjectRelevanceReason] = set()
    strongest = contributions[0]
    if strongest.story_total_relevance >= _EXCEPTIONAL_STORY_THRESHOLD:
        reasons.add(ProjectRelevanceReason.EXCEPTIONAL_TOP_STORY)
    if (
        len(contributions) >= 2
        and contributions[1].story_total_relevance >= _STRONG_STORY_THRESHOLD
    ):
        reasons.add(ProjectRelevanceReason.MULTIPLE_STRONG_STORIES)
    if len(contributions) == 1 or contributions[1].weighted_contribution == 0.0:
        reasons.add(ProjectRelevanceReason.LIMITED_STORY_DEPTH)
    if strongest.story_relevance.story_sufficiency.value in {
        "low",
        "unassessed",
    }:
        reasons.add(ProjectRelevanceReason.STORY_COMPLETION_NEEDED)
    if any(
        item.story_relevance.evidence_risk_adjustment > 0.0
        for item in contributions
    ):
        reasons.add(ProjectRelevanceReason.CLAIM_RISK_PRESENT)
    if len(stories) > MAX_CONTRIBUTING_STORIES:
        reasons.add(ProjectRelevanceReason.ADDITIONAL_STORIES_CAPPED)
    return tuple(item for item in ProjectRelevanceReason if item in reasons)


def aggregate_project_story_relevance(
    *,
    project_id: str,
    story_relevance: Sequence[StoryHiringRelevance],
) -> ProjectHiringRelevance | None:
    """Aggregate one exact project's accepted Story relevance results."""

    target_project = _exact_text(project_id, "project_id")
    stories = _validated_unique_stories(
        project_id=target_project,
        story_relevance=story_relevance,
    )
    if not stories:
        return None

    running_total = 0.0
    contributions: list[ProjectStoryContribution] = []
    for rank_position, story in enumerate(
        stories[:MAX_CONTRIBUTING_STORIES],
        start=1,
    ):
        weight = PROJECT_STORY_POSITION_WEIGHTS[rank_position - 1]
        available_headroom = _score(
            1.0 - running_total,
            "available_headroom",
        )
        normalized_depth = (
            story.total_relevance_score
            if rank_position == 1
            else _depth_factor(story.total_relevance_score)
        )
        weighted_contribution = _score(
            available_headroom * weight * normalized_depth,
            "weighted_contribution",
        )
        contribution = ProjectStoryContribution(
            rank_position=rank_position,
            story_relevance=story,
            positional_weight=weight,
            normalized_depth=normalized_depth,
            available_headroom=available_headroom,
            weighted_contribution=weighted_contribution,
        )
        contributions.append(contribution)
        running_total = _score(
            running_total + weighted_contribution,
            "aggregate_relevance_score",
        )

    stable_contributions = tuple(contributions)
    secondary_depth = _score(
        sum(item.weighted_contribution for item in stable_contributions[1:]),
        "secondary_story_depth",
    )
    components = ProjectRelevanceComponents(
        strongest_story_contribution=(
            stable_contributions[0].weighted_contribution
        ),
        secondary_story_depth=secondary_depth,
        aggregate_relevance=running_total,
    )
    reasons = _project_reasons(
        stories=stories,
        contributions=stable_contributions,
    )
    return ProjectHiringRelevance(
        project_id=target_project,
        hiring_context_profile_id=stories[0].hiring_context_profile_id,
        hiring_context_fingerprint=stories[0].hiring_context_fingerprint,
        aggregation_policy_id=PROJECT_RELEVANCE_AGGREGATION_POLICY_ID,
        rankable_story_count=len(stories),
        components=components,
        aggregate_relevance_score=running_total,
        contributions=stable_contributions,
        reasons=reasons,
    )


def rank_projects_from_story_relevance(
    story_relevance: Sequence[StoryHiringRelevance],
) -> tuple[ProjectHiringRelevance, ...]:
    """Group, independently aggregate, and deterministically sort projects."""

    stories = _validated_unique_stories(
        project_id=None,
        story_relevance=story_relevance,
    )
    if not stories:
        return ()
    by_project: dict[str, list[StoryHiringRelevance]] = {}
    for story in stories:
        by_project.setdefault(story.project_id, []).append(story)
    results = tuple(
        aggregate_project_story_relevance(
            project_id=project_id,
            story_relevance=by_project[project_id],
        )
        for project_id in sorted(by_project)
    )
    return tuple(sorted(
        (item for item in results if item is not None),
        key=lambda item: (
            -item.aggregate_relevance_score,
            item.project_id,
            item.project_relevance_id,
        ),
    ))


__all__ = [
    "DEPTH_QUALIFICATION_FLOOR",
    "MAX_CONTRIBUTING_STORIES",
    "MAX_PROJECT_RELEVANCE_REASONS",
    "MAX_PROJECT_STORY_INPUTS",
    "PROJECT_RELEVANCE_AGGREGATION_POLICY_ID",
    "PROJECT_RELEVANCE_SCORE_DECIMALS",
    "PROJECT_STORY_POSITION_WEIGHTS",
    "ProjectHiringRelevance",
    "ProjectRelevanceAggregationError",
    "ProjectRelevanceAggregationErrorCode",
    "ProjectRelevanceComponents",
    "ProjectRelevanceReason",
    "ProjectStoryContribution",
    "aggregate_project_story_relevance",
    "rank_projects_from_story_relevance",
]
