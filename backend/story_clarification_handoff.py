"""Pure Story-level relevance and sufficiency handoff projection.

The contracts in this module summarize accepted contextual ranking outputs for
a later clarification layer.  They do not load Stories, evaluate sufficiency,
rerank projects, generate questions, classify information sources, or perform
runtime I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Never

from backend.engineering_story_models import (
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.project_portfolio_ranking import (
    MAX_PORTFOLIO_PROJECTS,
    PortfolioProjectRanking,
    RankedProjectStoryPortfolio,
)
from backend.project_story_ranking import MAX_CONTRIBUTING_STORIES


STORY_CLARIFICATION_HANDOFF_POLICY_ID = (
    "story_clarification_handoff.authoritative_projection.v1"
)
MAX_STORY_CLARIFICATION_HANDOFFS = (
    MAX_PORTFOLIO_PROJECTS * MAX_CONTRIBUTING_STORIES
)
MAX_STORY_CLARIFICATION_REASONS = 2


class StoryClarificationReason(str, Enum):
    """Direct descriptions of authoritative non-high sufficiency states."""

    CLAIM_SAFETY_GAP = "claim_safety_gap"
    STORY_COMPLETENESS_GAP = "story_completeness_gap"


class StoryClarificationHandoffErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    SOURCE_INTEGRITY_MISMATCH = "source_integrity_mismatch"
    MIXED_HIRING_CONTEXT = "mixed_hiring_context"
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORY_IDENTITY_CONFLICT = "story_identity_conflict"


class StoryClarificationHandoffError(ValueError):
    def __init__(
        self,
        code: StoryClarificationHandoffErrorCode,
        message: str,
    ) -> None:
        self.code = StoryClarificationHandoffErrorCode(code)
        super().__init__(message)


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite score from 0 to 1")
    return normalized


def _positive_position(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _expected_reasons(
    claim_sufficiency: SufficiencyLevel,
    story_sufficiency: SufficiencyLevel,
) -> tuple[StoryClarificationReason, ...]:
    reasons: list[StoryClarificationReason] = []
    if claim_sufficiency is not SufficiencyLevel.HIGH:
        reasons.append(StoryClarificationReason.CLAIM_SAFETY_GAP)
    if story_sufficiency is not SufficiencyLevel.HIGH:
        reasons.append(StoryClarificationReason.STORY_COMPLETENESS_GAP)
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class StoryClarificationHandoff:
    handoff_policy_id: str
    project_id: str
    canonical_story_id: str
    current_revision_id: str
    story_provenance_fingerprint: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    story_relevance_id: str
    project_relevance_id: str
    portfolio_project_ranking_id: str
    portfolio_id: str
    portfolio_position: int
    project_story_rank_position: int
    story_relevance: float
    project_story_weighted_contribution: float
    project_relevance: float
    portfolio_relevance: float
    claim_sufficiency: SufficiencyLevel
    story_sufficiency: SufficiencyLevel
    story_opportunity: StoryOpportunityLevel
    reasons: tuple[StoryClarificationReason, ...]
    handoff_id: str = ""

    def __post_init__(self) -> None:
        policy_id = _exact_text(self.handoff_policy_id, "handoff_policy_id")
        if policy_id != STORY_CLARIFICATION_HANDOFF_POLICY_ID:
            raise ValueError("handoff_policy_id is not supported")
        text_values = {
            "project_id": self.project_id,
            "canonical_story_id": self.canonical_story_id,
            "current_revision_id": self.current_revision_id,
            "story_provenance_fingerprint": self.story_provenance_fingerprint,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "story_relevance_id": self.story_relevance_id,
            "project_relevance_id": self.project_relevance_id,
            "portfolio_project_ranking_id": self.portfolio_project_ranking_id,
            "portfolio_id": self.portfolio_id,
        }
        normalized_text = {
            name: _exact_text(value, name)
            for name, value in text_values.items()
        }
        portfolio_position = _positive_position(
            self.portfolio_position,
            "portfolio_position",
            MAX_PORTFOLIO_PROJECTS,
        )
        story_position = _positive_position(
            self.project_story_rank_position,
            "project_story_rank_position",
            MAX_CONTRIBUTING_STORIES,
        )
        score_values = {
            "story_relevance": self.story_relevance,
            "project_story_weighted_contribution": (
                self.project_story_weighted_contribution
            ),
            "project_relevance": self.project_relevance,
            "portfolio_relevance": self.portfolio_relevance,
        }
        normalized_scores = {
            name: _score(value, name)
            for name, value in score_values.items()
        }
        claim = SufficiencyLevel(self.claim_sufficiency)
        story = SufficiencyLevel(self.story_sufficiency)
        opportunity = StoryOpportunityLevel(self.story_opportunity)
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        if len(self.reasons) > MAX_STORY_CLARIFICATION_REASONS:
            raise ValueError("reasons exceeds the handoff reason bound")
        if any(not isinstance(item, StoryClarificationReason) for item in self.reasons):
            raise TypeError("reasons must contain StoryClarificationReason values")
        reasons = _expected_reasons(claim, story)
        if self.reasons != reasons:
            raise ValueError(
                "reasons must match authoritative sufficiency in enum order"
            )
        payload = {
            "handoff_policy_id": policy_id,
            **normalized_text,
            "portfolio_position": portfolio_position,
            "project_story_rank_position": story_position,
            **normalized_scores,
            "claim_sufficiency": claim.value,
            "story_sufficiency": story.value,
            "story_opportunity": opportunity.value,
            "reasons": [item.value for item in reasons],
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
        expected_id = f"story_clarification_handoff_{digest[:24]}"
        if self.handoff_id not in ("", expected_id):
            raise ValueError("handoff_id does not match normalized handoff")
        for name, value in normalized_text.items():
            object.__setattr__(self, name, value)
        for name, value in normalized_scores.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "portfolio_position", portfolio_position)
        object.__setattr__(self, "project_story_rank_position", story_position)
        object.__setattr__(self, "claim_sufficiency", claim)
        object.__setattr__(self, "story_sufficiency", story)
        object.__setattr__(self, "story_opportunity", opportunity)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "handoff_id", expected_id)

    @property
    def has_claim_safety_gap(self) -> bool:
        return StoryClarificationReason.CLAIM_SAFETY_GAP in self.reasons

    @property
    def has_story_completeness_gap(self) -> bool:
        return StoryClarificationReason.STORY_COMPLETENESS_GAP in self.reasons

    @property
    def has_authoritative_sufficiency_signal(self) -> bool:
        return self.has_claim_safety_gap or self.has_story_completeness_gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_policy_id": self.handoff_policy_id,
            "project_id": self.project_id,
            "canonical_story_id": self.canonical_story_id,
            "current_revision_id": self.current_revision_id,
            "story_provenance_fingerprint": self.story_provenance_fingerprint,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "story_relevance_id": self.story_relevance_id,
            "project_relevance_id": self.project_relevance_id,
            "portfolio_project_ranking_id": self.portfolio_project_ranking_id,
            "portfolio_id": self.portfolio_id,
            "portfolio_position": self.portfolio_position,
            "project_story_rank_position": self.project_story_rank_position,
            "story_relevance": self.story_relevance,
            "project_story_weighted_contribution": (
                self.project_story_weighted_contribution
            ),
            "project_relevance": self.project_relevance,
            "portfolio_relevance": self.portfolio_relevance,
            "claim_sufficiency": self.claim_sufficiency.value,
            "story_sufficiency": self.story_sufficiency.value,
            "story_opportunity": self.story_opportunity.value,
            "has_claim_safety_gap": self.has_claim_safety_gap,
            "has_story_completeness_gap": self.has_story_completeness_gap,
            "has_authoritative_sufficiency_signal": (
                self.has_authoritative_sufficiency_signal
            ),
            "reasons": [item.value for item in self.reasons],
            "handoff_id": self.handoff_id,
        }


def _source_integrity_error(name: str, error: Exception) -> Never:
    raise StoryClarificationHandoffError(
        StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH,
        f"{name} no longer matches its immutable ranking identity",
    ) from error


def _validate_contract(value: Any, name: str) -> None:
    """Re-run only immutable contract validation; do not evaluate semantics."""

    try:
        validated = replace(value)
    except (TypeError, ValueError) as error:
        _source_integrity_error(name, error)
    if validated != value:
        _source_integrity_error(
            name,
            ValueError("normalization changed the accepted source"),
        )


def _validate_project_source(
    *,
    ranking: PortfolioProjectRanking,
    portfolio: RankedProjectStoryPortfolio,
) -> None:
    project = ranking.project_relevance
    footprint = ranking.semantic_footprint
    if (
        project.hiring_context_profile_id != portfolio.hiring_context_profile_id
        or project.hiring_context_fingerprint
        != portfolio.hiring_context_fingerprint
    ):
        raise StoryClarificationHandoffError(
            StoryClarificationHandoffErrorCode.MIXED_HIRING_CONTEXT,
            "ranked project does not match the portfolio Hiring Context",
        )
    if footprint.project_id != project.project_id:
        raise StoryClarificationHandoffError(
            StoryClarificationHandoffErrorCode.PROJECT_IDENTITY_MISMATCH,
            "semantic footprint project does not match project relevance",
        )
    expected_story_ids = tuple(
        item.story_relevance_id for item in project.contributions
    )
    if footprint.contributing_story_relevance_ids != expected_story_ids:
        raise StoryClarificationHandoffError(
            StoryClarificationHandoffErrorCode.SOURCE_INTEGRITY_MISMATCH,
            "semantic footprint does not preserve the project contributions",
        )
    _validate_contract(footprint, "project semantic footprint")
    _validate_contract(ranking.adjustment, "portfolio adjustment")
    for contribution in project.contributions:
        story = contribution.story_relevance
        if story.project_id != project.project_id:
            raise StoryClarificationHandoffError(
                StoryClarificationHandoffErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Story contribution does not match its project",
            )
        if (
            story.hiring_context_profile_id
            != portfolio.hiring_context_profile_id
            or story.hiring_context_fingerprint
            != portfolio.hiring_context_fingerprint
        ):
            raise StoryClarificationHandoffError(
                StoryClarificationHandoffErrorCode.MIXED_HIRING_CONTEXT,
                "Story contribution does not match the portfolio Hiring Context",
            )
        _validate_contract(story, "Story relevance")
        _validate_contract(contribution, "project Story contribution")
    _validate_contract(project, "project relevance")
    _validate_contract(ranking, "portfolio project ranking")


def build_story_clarification_handoffs(
    *,
    portfolio: RankedProjectStoryPortfolio,
) -> tuple[StoryClarificationHandoff, ...]:
    """Project accepted ranking and sufficiency state into bounded handoffs.

    Output order is the existing portfolio position followed by the existing
    project Story contribution position.  It is not a final clarification
    priority.
    """

    if not isinstance(portfolio, RankedProjectStoryPortfolio):
        raise TypeError("portfolio must be RankedProjectStoryPortfolio")
    project_count = len(portfolio.ranked_projects)
    if not 1 <= project_count <= MAX_PORTFOLIO_PROJECTS:
        raise StoryClarificationHandoffError(
            StoryClarificationHandoffErrorCode.BOUND_EXCEEDED,
            "portfolio project count exceeds the handoff bound",
        )
    source_count = sum(
        len(item.project_relevance.contributions)
        for item in portfolio.ranked_projects
    )
    if source_count > MAX_STORY_CLARIFICATION_HANDOFFS:
        raise StoryClarificationHandoffError(
            StoryClarificationHandoffErrorCode.BOUND_EXCEEDED,
            "Story handoff source count exceeds the handoff bound",
        )
    _validate_contract(portfolio, "ranked Story project portfolio")
    handoffs: list[StoryClarificationHandoff] = []
    story_owners: dict[str, str] = {}
    revision_owners: dict[str, tuple[str, str]] = {}
    relevance_owners: dict[str, tuple[str, str, str]] = {}
    for ranking in portfolio.ranked_projects:
        _validate_project_source(ranking=ranking, portfolio=portfolio)
        project = ranking.project_relevance
        for contribution in project.contributions:
            story = contribution.story_relevance
            prior_project = story_owners.get(story.canonical_story_id)
            if prior_project is not None and prior_project != project.project_id:
                raise StoryClarificationHandoffError(
                    StoryClarificationHandoffErrorCode.STORY_IDENTITY_CONFLICT,
                    "one canonical Story cannot belong to multiple projects",
                )
            story_owners[story.canonical_story_id] = project.project_id
            revision_owner = (
                project.project_id,
                story.canonical_story_id,
            )
            prior_revision_owner = revision_owners.get(story.current_revision_id)
            if (
                prior_revision_owner is not None
                and prior_revision_owner != revision_owner
            ):
                raise StoryClarificationHandoffError(
                    StoryClarificationHandoffErrorCode.STORY_IDENTITY_CONFLICT,
                    "one Story revision cannot describe conflicting identity",
                )
            revision_owners[story.current_revision_id] = revision_owner
            freshness = (
                project.project_id,
                story.canonical_story_id,
                story.current_revision_id,
            )
            prior_freshness = relevance_owners.get(story.relevance_id)
            if prior_freshness is not None and prior_freshness != freshness:
                raise StoryClarificationHandoffError(
                    StoryClarificationHandoffErrorCode.STORY_IDENTITY_CONFLICT,
                    "one Story relevance ID cannot describe conflicting identity",
                )
            relevance_owners[story.relevance_id] = freshness
            reasons = _expected_reasons(
                story.claim_sufficiency,
                story.story_sufficiency,
            )
            handoffs.append(StoryClarificationHandoff(
                handoff_policy_id=STORY_CLARIFICATION_HANDOFF_POLICY_ID,
                project_id=project.project_id,
                canonical_story_id=story.canonical_story_id,
                current_revision_id=story.current_revision_id,
                story_provenance_fingerprint=(
                    story.story_provenance_fingerprint
                ),
                hiring_context_profile_id=portfolio.hiring_context_profile_id,
                hiring_context_fingerprint=portfolio.hiring_context_fingerprint,
                story_relevance_id=story.relevance_id,
                project_relevance_id=project.project_relevance_id,
                portfolio_project_ranking_id=ranking.ranking_id,
                portfolio_id=portfolio.portfolio_id,
                portfolio_position=ranking.position,
                project_story_rank_position=contribution.rank_position,
                story_relevance=story.total_relevance_score,
                project_story_weighted_contribution=(
                    contribution.weighted_contribution
                ),
                project_relevance=project.aggregate_relevance_score,
                portfolio_relevance=ranking.portfolio_relevance,
                claim_sufficiency=story.claim_sufficiency,
                story_sufficiency=story.story_sufficiency,
                story_opportunity=story.story_opportunity,
                reasons=reasons,
            ))
    return tuple(handoffs)


__all__ = [
    "MAX_STORY_CLARIFICATION_HANDOFFS",
    "MAX_STORY_CLARIFICATION_REASONS",
    "STORY_CLARIFICATION_HANDOFF_POLICY_ID",
    "StoryClarificationHandoff",
    "StoryClarificationHandoffError",
    "StoryClarificationHandoffErrorCode",
    "StoryClarificationReason",
    "build_story_clarification_handoffs",
]
