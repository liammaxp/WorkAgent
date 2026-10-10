"""Pure portfolio ordering over accepted project relevance results.

The portfolio layer compares only bounded candidate-side semantic features on
accepted Story contributions.  It never reconstructs Story truth, re-evaluates
hiring context, mutates evidence, selects a final project count, allocates
resume space, or performs runtime I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from backend.engineering_story_relevance import StoryRelevanceFeature
from backend.project_story_ranking import ProjectHiringRelevance


PORTFOLIO_SCORE_DECIMALS = 6
MAX_PORTFOLIO_PROJECTS = 64
MAX_PROJECT_SEMANTIC_TRAITS = len(StoryRelevanceFeature)
MAX_PORTFOLIO_FEATURE_REFERENCES = 8
MAX_PORTFOLIO_RANKING_REASONS = 4
PORTFOLIO_REDUNDANCY_MASS_FLOOR = 3.0
PORTFOLIO_DIFFERENTIATION_MASS_FLOOR = 3.0
PORTFOLIO_DIFFERENTIATION_BASE_FLOOR = 0.50
MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT = 0.06
MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT = 0.04
PORTFOLIO_RANKING_POLICY_ID = (
    "project_portfolio.candidate_semantic_marginal.v1"
)

_GENERIC_COMMON_FEATURES = frozenset({
    StoryRelevanceFeature.DEBUGGING,
    StoryRelevanceFeature.TESTING,
    StoryRelevanceFeature.INTEGRATION,
    StoryRelevanceFeature.AUTOMATION,
})
_BROAD_CATEGORY_FEATURES = frozenset({
    StoryRelevanceFeature.BACKEND,
    StoryRelevanceFeature.FRONTEND,
    StoryRelevanceFeature.DATA_ENGINEERING,
    StoryRelevanceFeature.ANALYTICS,
    StoryRelevanceFeature.GAME_DEVELOPMENT,
    StoryRelevanceFeature.DEVOPS_CLOUD,
    StoryRelevanceFeature.SECURITY,
    StoryRelevanceFeature.MACHINE_LEARNING,
    StoryRelevanceFeature.MOBILE,
    StoryRelevanceFeature.EMBEDDED,
    StoryRelevanceFeature.PLATFORM_ENGINEERING,
})

PORTFOLIO_FEATURE_INFORMATION_WEIGHTS = MappingProxyType({
    feature: (
        0.25
        if feature in _GENERIC_COMMON_FEATURES
        else 0.50
        if feature in _BROAD_CATEGORY_FEATURES
        else 1.00
    )
    for feature in StoryRelevanceFeature
})

_FEATURE_ORDER = {
    feature: index for index, feature in enumerate(StoryRelevanceFeature)
}


class PortfolioRankingReason(str, Enum):
    BASE_RELEVANCE_FIRST = "base_relevance_first"
    CANDIDATE_TRAIT_REDUNDANCY = "candidate_trait_redundancy"
    COMPLEMENTARY_CANDIDATE_TRAITS = "complementary_candidate_traits"
    NO_SEMANTIC_ADJUSTMENT = "no_semantic_adjustment"


class PortfolioRankingErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    MIXED_HIRING_CONTEXT = "mixed_hiring_context"
    MIXED_HIRING_CONTEXT_FINGERPRINT = "mixed_hiring_context_fingerprint"
    CONFLICTING_PROJECT_RESULT = "conflicting_project_result"


class PortfolioRankingError(ValueError):
    def __init__(
        self,
        code: PortfolioRankingErrorCode,
        message: str,
    ) -> None:
        self.code = PortfolioRankingErrorCode(code)
        super().__init__(message)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite score from 0 to 1")
    return round(normalized, PORTFOLIO_SCORE_DECIMALS)


def _bounded_adjustment(value: Any, name: str, maximum: float) -> float:
    normalized = _score(value, name)
    if normalized > maximum:
        raise ValueError(f"{name} exceeds policy cap {maximum}")
    return normalized


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _stable_feature_references(
    values: tuple[StoryRelevanceFeature, ...],
    name: str,
) -> tuple[StoryRelevanceFeature, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > MAX_PORTFOLIO_FEATURE_REFERENCES:
        raise ValueError(
            f"{name} exceeds maximum item count "
            f"{MAX_PORTFOLIO_FEATURE_REFERENCES}"
        )
    normalized = tuple(StoryRelevanceFeature(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique features")
    return normalized


def _base_qualification(base_relevance: float) -> float:
    if base_relevance <= PORTFOLIO_DIFFERENTIATION_BASE_FLOOR:
        return 0.0
    return _score(
        (base_relevance - PORTFOLIO_DIFFERENTIATION_BASE_FLOOR)
        / (1.0 - PORTFOLIO_DIFFERENTIATION_BASE_FLOOR),
        "base differentiation qualification",
    )


@dataclass(frozen=True, slots=True)
class ProjectSemanticTrait:
    feature: StoryRelevanceFeature
    contribution_strength: float
    information_weight: float
    weighted_strength: float

    def __post_init__(self) -> None:
        feature = StoryRelevanceFeature(self.feature)
        contribution = _score(
            self.contribution_strength,
            "contribution_strength",
        )
        information = _score(self.information_weight, "information_weight")
        weighted = _score(self.weighted_strength, "weighted_strength")
        if contribution == 0.0:
            raise ValueError("semantic traits must have positive contribution strength")
        expected_information = PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[feature]
        if information != expected_information:
            raise ValueError("information_weight does not match portfolio policy")
        if weighted != _score(
            contribution * information,
            "expected weighted trait strength",
        ):
            raise ValueError("weighted_strength does not match portfolio policy")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "contribution_strength", contribution)
        object.__setattr__(self, "information_weight", information)
        object.__setattr__(self, "weighted_strength", weighted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature.value,
            "contribution_strength": self.contribution_strength,
            "information_weight": self.information_weight,
            "weighted_strength": self.weighted_strength,
        }


@dataclass(frozen=True, slots=True)
class ProjectSemanticFootprint:
    project_id: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    source_project_relevance_id: str
    traits: tuple[ProjectSemanticTrait, ...]
    contributing_story_relevance_ids: tuple[str, ...]
    footprint_id: str = ""

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
        source_id = _exact_text(
            self.source_project_relevance_id,
            "source_project_relevance_id",
        )
        if not isinstance(self.traits, tuple):
            raise TypeError("traits must be a tuple")
        if len(self.traits) > MAX_PROJECT_SEMANTIC_TRAITS:
            raise ValueError("traits exceeds the closed semantic feature bound")
        if any(not isinstance(item, ProjectSemanticTrait) for item in self.traits):
            raise TypeError("traits must contain ProjectSemanticTrait values")
        if len({item.feature for item in self.traits}) != len(self.traits):
            raise ValueError("traits must contain unique semantic features")
        expected_traits = tuple(sorted(
            self.traits,
            key=lambda item: _FEATURE_ORDER[item.feature],
        ))
        if self.traits != expected_traits:
            raise ValueError("traits must follow semantic feature enum order")
        if not isinstance(self.contributing_story_relevance_ids, tuple):
            raise TypeError("contributing_story_relevance_ids must be a tuple")
        if not 1 <= len(self.contributing_story_relevance_ids) <= 3:
            raise ValueError(
                "contributing_story_relevance_ids must preserve one to three Stories"
            )
        story_ids = tuple(
            _exact_text(value, "contributing Story relevance ID")
            for value in self.contributing_story_relevance_ids
        )
        if len(set(story_ids)) != len(story_ids):
            raise ValueError("contributing Story relevance IDs must be unique")
        payload = {
            "project_id": project_id,
            "hiring_context_profile_id": profile_id,
            "hiring_context_fingerprint": context_fingerprint,
            "source_project_relevance_id": source_id,
            "traits": [item.to_dict() for item in expected_traits],
            "contributing_story_relevance_ids": list(story_ids),
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
        expected_id = f"project_semantic_footprint_{digest[:24]}"
        if self.footprint_id not in ("", expected_id):
            raise ValueError("footprint_id does not match normalized footprint")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "hiring_context_profile_id", profile_id)
        object.__setattr__(
            self,
            "hiring_context_fingerprint",
            context_fingerprint,
        )
        object.__setattr__(self, "source_project_relevance_id", source_id)
        object.__setattr__(self, "traits", expected_traits)
        object.__setattr__(self, "contributing_story_relevance_ids", story_ids)
        object.__setattr__(self, "footprint_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "source_project_relevance_id": self.source_project_relevance_id,
            "traits": [item.to_dict() for item in self.traits],
            "contributing_story_relevance_ids": list(
                self.contributing_story_relevance_ids
            ),
            "footprint_id": self.footprint_id,
        }


@dataclass(frozen=True, slots=True)
class PortfolioAdjustment:
    redundancy_score: float
    redundancy_adjustment: float
    differentiation_score: float
    differentiation_adjustment: float
    overlapping_features: tuple[StoryRelevanceFeature, ...]
    differentiating_features: tuple[StoryRelevanceFeature, ...]

    def __post_init__(self) -> None:
        redundancy = _score(self.redundancy_score, "redundancy_score")
        redundancy_adjustment = _bounded_adjustment(
            self.redundancy_adjustment,
            "redundancy_adjustment",
            MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT,
        )
        differentiation = _score(
            self.differentiation_score,
            "differentiation_score",
        )
        differentiation_adjustment = _bounded_adjustment(
            self.differentiation_adjustment,
            "differentiation_adjustment",
            MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT,
        )
        overlap = _stable_feature_references(
            self.overlapping_features,
            "overlapping_features",
        )
        differentiating = _stable_feature_references(
            self.differentiating_features,
            "differentiating_features",
        )
        object.__setattr__(self, "redundancy_score", redundancy)
        object.__setattr__(
            self,
            "redundancy_adjustment",
            redundancy_adjustment,
        )
        object.__setattr__(self, "differentiation_score", differentiation)
        object.__setattr__(
            self,
            "differentiation_adjustment",
            differentiation_adjustment,
        )
        object.__setattr__(self, "overlapping_features", overlap)
        object.__setattr__(self, "differentiating_features", differentiating)

    def to_dict(self) -> dict[str, Any]:
        return {
            "redundancy_score": self.redundancy_score,
            "redundancy_adjustment": self.redundancy_adjustment,
            "differentiation_score": self.differentiation_score,
            "differentiation_adjustment": self.differentiation_adjustment,
            "overlapping_features": [
                item.value for item in self.overlapping_features
            ],
            "differentiating_features": [
                item.value for item in self.differentiating_features
            ],
        }


@dataclass(frozen=True, slots=True)
class PortfolioProjectRanking:
    position: int
    project_relevance: ProjectHiringRelevance
    semantic_footprint: ProjectSemanticFootprint
    base_relevance: float
    adjustment: PortfolioAdjustment
    portfolio_relevance: float
    reasons: tuple[PortfolioRankingReason, ...]
    ranking_id: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or not 1 <= self.position <= MAX_PORTFOLIO_PROJECTS
        ):
            raise ValueError("position must be within the portfolio project bound")
        if not isinstance(self.project_relevance, ProjectHiringRelevance):
            raise TypeError("project_relevance must be ProjectHiringRelevance")
        if not isinstance(self.semantic_footprint, ProjectSemanticFootprint):
            raise TypeError("semantic_footprint must be ProjectSemanticFootprint")
        if not isinstance(self.adjustment, PortfolioAdjustment):
            raise TypeError("adjustment must be PortfolioAdjustment")
        project = self.project_relevance
        footprint = self.semantic_footprint
        if footprint.project_id != project.project_id:
            raise ValueError("semantic footprint project does not match relevance")
        if footprint.hiring_context_profile_id != project.hiring_context_profile_id:
            raise ValueError("semantic footprint context does not match relevance")
        if footprint.hiring_context_fingerprint != project.hiring_context_fingerprint:
            raise ValueError(
                "semantic footprint context fingerprint does not match relevance"
            )
        if footprint.source_project_relevance_id != project.project_relevance_id:
            raise ValueError("semantic footprint source does not match relevance")
        base = _score(self.base_relevance, "base_relevance")
        if base != project.aggregate_relevance_score:
            raise ValueError("base_relevance must preserve project relevance exactly")
        if self.position == 1:
            expected_redundancy_adjustment = 0.0
            expected_differentiation_adjustment = 0.0
            if (
                self.adjustment.redundancy_score != 0.0
                or self.adjustment.differentiation_score != 0.0
            ):
                raise ValueError("first portfolio project cannot have adjustments")
        else:
            expected_redundancy_adjustment = _score(
                MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT
                * self.adjustment.redundancy_score,
                "expected redundancy adjustment",
            )
            expected_differentiation_adjustment = _score(
                MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT
                * self.adjustment.differentiation_score
                * _base_qualification(base),
                "expected differentiation adjustment",
            )
        if (
            self.adjustment.redundancy_adjustment
            != expected_redundancy_adjustment
        ):
            raise ValueError("redundancy_adjustment does not match portfolio policy")
        if (
            self.adjustment.differentiation_adjustment
            != expected_differentiation_adjustment
        ):
            raise ValueError(
                "differentiation_adjustment does not match portfolio policy"
            )
        expected_portfolio = _score(
            min(
                1.0,
                max(
                    0.0,
                    base
                    - expected_redundancy_adjustment
                    + expected_differentiation_adjustment,
                ),
            ),
            "expected portfolio relevance",
        )
        portfolio_relevance = _score(
            self.portfolio_relevance,
            "portfolio_relevance",
        )
        if portfolio_relevance != expected_portfolio:
            raise ValueError("portfolio_relevance does not match adjustments")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        if len(self.reasons) > MAX_PORTFOLIO_RANKING_REASONS:
            raise ValueError("reasons exceeds the portfolio ranking reason bound")
        if any(not isinstance(item, PortfolioRankingReason) for item in self.reasons):
            raise TypeError("reasons must contain PortfolioRankingReason values")
        stable_reasons = tuple(
            item for item in PortfolioRankingReason if item in set(self.reasons)
        )
        if self.reasons != stable_reasons:
            raise ValueError("reasons must be unique and in deterministic order")
        payload = {
            "position": self.position,
            "project_relevance_id": project.project_relevance_id,
            "semantic_footprint_id": footprint.footprint_id,
            "base_relevance": base,
            "adjustment": self.adjustment.to_dict(),
            "portfolio_relevance": portfolio_relevance,
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
        expected_id = f"portfolio_project_ranking_{digest[:24]}"
        if self.ranking_id not in ("", expected_id):
            raise ValueError("ranking_id does not match normalized ranking")
        object.__setattr__(self, "base_relevance", base)
        object.__setattr__(self, "portfolio_relevance", portfolio_relevance)
        object.__setattr__(self, "reasons", stable_reasons)
        object.__setattr__(self, "ranking_id", expected_id)

    @property
    def project_id(self) -> str:
        return self.project_relevance.project_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "project_id": self.project_id,
            "base_project_relevance": self.project_relevance.to_dict(),
            "semantic_footprint": self.semantic_footprint.to_dict(),
            "base_relevance": self.base_relevance,
            "adjustment": self.adjustment.to_dict(),
            "portfolio_relevance": self.portfolio_relevance,
            "reasons": [item.value for item in self.reasons],
            "ranking_id": self.ranking_id,
        }


@dataclass(frozen=True, slots=True)
class RankedProjectStoryPortfolio:
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    ranking_policy_id: str
    ranked_projects: tuple[PortfolioProjectRanking, ...]
    portfolio_id: str = ""

    def __post_init__(self) -> None:
        profile_id = _exact_text(
            self.hiring_context_profile_id,
            "hiring_context_profile_id",
        )
        context_fingerprint = _exact_text(
            self.hiring_context_fingerprint,
            "hiring_context_fingerprint",
        )
        policy_id = _exact_text(self.ranking_policy_id, "ranking_policy_id")
        if policy_id != PORTFOLIO_RANKING_POLICY_ID:
            raise ValueError("ranking_policy_id is not supported")
        if not isinstance(self.ranked_projects, tuple):
            raise TypeError("ranked_projects must be a tuple")
        if not 1 <= len(self.ranked_projects) <= MAX_PORTFOLIO_PROJECTS:
            raise ValueError("ranked_projects must be within the project bound")
        if any(
            not isinstance(item, PortfolioProjectRanking)
            for item in self.ranked_projects
        ):
            raise TypeError(
                "ranked_projects must contain PortfolioProjectRanking values"
            )
        if tuple(item.position for item in self.ranked_projects) != tuple(
            range(1, len(self.ranked_projects) + 1)
        ):
            raise ValueError("portfolio positions must be consecutive")
        if len({item.project_id for item in self.ranked_projects}) != len(
            self.ranked_projects
        ):
            raise ValueError("portfolio projects must have unique project IDs")
        for item in self.ranked_projects:
            project = item.project_relevance
            if project.hiring_context_profile_id != profile_id:
                raise ValueError("ranked project context does not match portfolio")
            if project.hiring_context_fingerprint != context_fingerprint:
                raise ValueError(
                    "ranked project context fingerprint does not match portfolio"
                )
        payload = {
            "hiring_context_profile_id": profile_id,
            "hiring_context_fingerprint": context_fingerprint,
            "ranking_policy_id": policy_id,
            "ranked_project_ids": [
                item.ranking_id for item in self.ranked_projects
            ],
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
        expected_id = f"ranked_project_story_portfolio_{digest[:24]}"
        if self.portfolio_id not in ("", expected_id):
            raise ValueError("portfolio_id does not match normalized portfolio")
        object.__setattr__(self, "hiring_context_profile_id", profile_id)
        object.__setattr__(
            self,
            "hiring_context_fingerprint",
            context_fingerprint,
        )
        object.__setattr__(self, "ranking_policy_id", policy_id)
        object.__setattr__(self, "portfolio_id", expected_id)

    @property
    def project_count(self) -> int:
        return len(self.ranked_projects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "ranking_policy_id": self.ranking_policy_id,
            "project_count": self.project_count,
            "ranked_projects": [
                item.to_dict() for item in self.ranked_projects
            ],
            "portfolio_id": self.portfolio_id,
        }


def derive_project_semantic_footprint(
    project: ProjectHiringRelevance,
) -> ProjectSemanticFootprint:
    """Derive a bounded candidate-side footprint from accepted contributions."""

    if not isinstance(project, ProjectHiringRelevance):
        raise TypeError("project must be ProjectHiringRelevance")
    strengths: dict[StoryRelevanceFeature, float] = {}
    for contribution in project.contributions:
        strength = contribution.weighted_contribution
        if strength == 0.0:
            continue
        for feature in contribution.story_relevance.semantic_features:
            strengths[feature] = max(strengths.get(feature, 0.0), strength)
    traits = tuple(
        ProjectSemanticTrait(
            feature=feature,
            contribution_strength=strengths[feature],
            information_weight=PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[feature],
            weighted_strength=(
                strengths[feature]
                * PORTFOLIO_FEATURE_INFORMATION_WEIGHTS[feature]
            ),
        )
        for feature in StoryRelevanceFeature
        if feature in strengths
    )
    return ProjectSemanticFootprint(
        project_id=project.project_id,
        hiring_context_profile_id=project.hiring_context_profile_id,
        hiring_context_fingerprint=project.hiring_context_fingerprint,
        source_project_relevance_id=project.project_relevance_id,
        traits=traits,
        contributing_story_relevance_ids=tuple(
            item.story_relevance_id for item in project.contributions
        ),
    )


def _portfolio_adjustment(
    *,
    footprint: ProjectSemanticFootprint,
    represented: dict[StoryRelevanceFeature, float],
    base_relevance: float,
) -> PortfolioAdjustment:
    candidate_mass = sum(item.weighted_strength for item in footprint.traits)
    shared_by_feature: dict[StoryRelevanceFeature, float] = {}
    unique_by_feature: dict[StoryRelevanceFeature, float] = {}
    shared_mass = 0.0
    unique_value_mass = 0.0
    for trait in footprint.traits:
        represented_strength = represented.get(trait.feature, 0.0)
        shared = trait.information_weight * min(
            trait.contribution_strength,
            represented_strength,
        )
        unique_value = (
            trait.information_weight
            * trait.contribution_strength
            * max(
                trait.contribution_strength - represented_strength,
                0.0,
            )
        )
        if shared > 0.0:
            shared_by_feature[trait.feature] = shared
            shared_mass += shared
        if unique_value > 0.0:
            unique_by_feature[trait.feature] = unique_value
            unique_value_mass += unique_value
    redundancy = (
        0.0
        if candidate_mass == 0.0
        else _score(
            min(
                1.0,
                shared_mass
                / max(PORTFOLIO_REDUNDANCY_MASS_FLOOR, candidate_mass),
            ),
            "redundancy_score",
        )
    )
    differentiation = _score(
        min(
            1.0,
            unique_value_mass / PORTFOLIO_DIFFERENTIATION_MASS_FLOOR,
        ),
        "differentiation_score",
    )
    redundancy_adjustment = _score(
        MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT * redundancy,
        "redundancy_adjustment",
    )
    differentiation_adjustment = _score(
        MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT
        * differentiation
        * _base_qualification(base_relevance),
        "differentiation_adjustment",
    )
    overlap = tuple(
        feature
        for feature, _ in sorted(
            shared_by_feature.items(),
            key=lambda item: (-item[1], _FEATURE_ORDER[item[0]]),
        )[:MAX_PORTFOLIO_FEATURE_REFERENCES]
    )
    differentiating = tuple(
        feature
        for feature, _ in sorted(
            unique_by_feature.items(),
            key=lambda item: (-item[1], _FEATURE_ORDER[item[0]]),
        )[:MAX_PORTFOLIO_FEATURE_REFERENCES]
    )
    return PortfolioAdjustment(
        redundancy_score=redundancy,
        redundancy_adjustment=redundancy_adjustment,
        differentiation_score=differentiation,
        differentiation_adjustment=differentiation_adjustment,
        overlapping_features=overlap,
        differentiating_features=differentiating,
    )


def _ranking_reasons(
    *,
    position: int,
    adjustment: PortfolioAdjustment,
) -> tuple[PortfolioRankingReason, ...]:
    reasons: set[PortfolioRankingReason] = set()
    if position == 1:
        reasons.add(PortfolioRankingReason.BASE_RELEVANCE_FIRST)
    else:
        if adjustment.redundancy_adjustment > 0.0:
            reasons.add(PortfolioRankingReason.CANDIDATE_TRAIT_REDUNDANCY)
        if adjustment.differentiation_adjustment > 0.0:
            reasons.add(PortfolioRankingReason.COMPLEMENTARY_CANDIDATE_TRAITS)
        if not reasons:
            reasons.add(PortfolioRankingReason.NO_SEMANTIC_ADJUSTMENT)
    return tuple(item for item in PortfolioRankingReason if item in reasons)


def _make_ranking(
    *,
    position: int,
    project: ProjectHiringRelevance,
    footprint: ProjectSemanticFootprint,
    adjustment: PortfolioAdjustment,
) -> PortfolioProjectRanking:
    portfolio_relevance = _score(
        min(
            1.0,
            max(
                0.0,
                project.aggregate_relevance_score
                - adjustment.redundancy_adjustment
                + adjustment.differentiation_adjustment,
            ),
        ),
        "portfolio_relevance",
    )
    return PortfolioProjectRanking(
        position=position,
        project_relevance=project,
        semantic_footprint=footprint,
        base_relevance=project.aggregate_relevance_score,
        adjustment=adjustment,
        portfolio_relevance=portfolio_relevance,
        reasons=_ranking_reasons(position=position, adjustment=adjustment),
    )


def _add_to_representation(
    represented: dict[StoryRelevanceFeature, float],
    footprint: ProjectSemanticFootprint,
) -> None:
    for trait in footprint.traits:
        represented[trait.feature] = max(
            represented.get(trait.feature, 0.0),
            trait.contribution_strength,
        )


def _validated_unique_projects(
    projects: Sequence[ProjectHiringRelevance],
) -> tuple[ProjectHiringRelevance, ...]:
    if isinstance(projects, (str, bytes)) or not isinstance(projects, Sequence):
        raise TypeError("projects must be a sequence")
    if len(projects) > MAX_PORTFOLIO_PROJECTS:
        raise PortfolioRankingError(
            PortfolioRankingErrorCode.BOUND_EXCEEDED,
            f"projects exceeds maximum item count {MAX_PORTFOLIO_PROJECTS}",
        )
    unique: dict[str, ProjectHiringRelevance] = {}
    relevance_owners: dict[str, str] = {}
    context_profile_id: str | None = None
    context_fingerprint: str | None = None
    for project in projects:
        if not isinstance(project, ProjectHiringRelevance):
            raise PortfolioRankingError(
                PortfolioRankingErrorCode.INVALID_INPUT,
                "projects must contain ProjectHiringRelevance values",
            )
        if context_profile_id is None:
            context_profile_id = project.hiring_context_profile_id
            context_fingerprint = project.hiring_context_fingerprint
        elif project.hiring_context_profile_id != context_profile_id:
            raise PortfolioRankingError(
                PortfolioRankingErrorCode.MIXED_HIRING_CONTEXT,
                "portfolio projects cannot mix Hiring Context profiles",
            )
        elif project.hiring_context_fingerprint != context_fingerprint:
            raise PortfolioRankingError(
                PortfolioRankingErrorCode.MIXED_HIRING_CONTEXT_FINGERPRINT,
                "portfolio projects cannot mix Hiring Context fingerprints",
            )
        relevance_owner = relevance_owners.get(project.project_relevance_id)
        if relevance_owner is not None and relevance_owner != project.project_id:
            raise PortfolioRankingError(
                PortfolioRankingErrorCode.CONFLICTING_PROJECT_RESULT,
                "one project relevance ID cannot belong to multiple projects",
            )
        relevance_owners[project.project_relevance_id] = project.project_id
        prior = unique.get(project.project_id)
        if prior is None:
            unique[project.project_id] = project
        elif prior != project:
            raise PortfolioRankingError(
                PortfolioRankingErrorCode.CONFLICTING_PROJECT_RESULT,
                "one project ID cannot have conflicting relevance results",
            )
    return tuple(sorted(
        unique.values(),
        key=lambda item: (
            -item.aggregate_relevance_score,
            item.project_id,
            item.project_relevance_id,
        ),
    ))


def rank_project_portfolio(
    *,
    projects: Sequence[ProjectHiringRelevance],
) -> RankedProjectStoryPortfolio | None:
    """Greedily order every eligible project against accumulated semantics."""

    accepted_projects = _validated_unique_projects(projects)
    if not accepted_projects:
        return None
    footprints = {
        project.project_id: derive_project_semantic_footprint(project)
        for project in accepted_projects
    }
    first = accepted_projects[0]
    zero_adjustment = PortfolioAdjustment(
        redundancy_score=0.0,
        redundancy_adjustment=0.0,
        differentiation_score=0.0,
        differentiation_adjustment=0.0,
        overlapping_features=(),
        differentiating_features=(),
    )
    ranked = [_make_ranking(
        position=1,
        project=first,
        footprint=footprints[first.project_id],
        adjustment=zero_adjustment,
    )]
    represented: dict[StoryRelevanceFeature, float] = {}
    _add_to_representation(represented, footprints[first.project_id])
    remaining = {
        project.project_id: project for project in accepted_projects[1:]
    }
    while remaining:
        position = len(ranked) + 1
        candidates: list[PortfolioProjectRanking] = []
        for project_id in sorted(remaining):
            project = remaining[project_id]
            footprint = footprints[project_id]
            adjustment = _portfolio_adjustment(
                footprint=footprint,
                represented=represented,
                base_relevance=project.aggregate_relevance_score,
            )
            candidates.append(_make_ranking(
                position=position,
                project=project,
                footprint=footprint,
                adjustment=adjustment,
            ))
        chosen = min(
            candidates,
            key=lambda item: (
                -item.portfolio_relevance,
                -item.base_relevance,
                item.project_id,
                item.project_relevance.project_relevance_id,
            ),
        )
        ranked.append(chosen)
        _add_to_representation(represented, chosen.semantic_footprint)
        del remaining[chosen.project_id]
    return RankedProjectStoryPortfolio(
        hiring_context_profile_id=first.hiring_context_profile_id,
        hiring_context_fingerprint=first.hiring_context_fingerprint,
        ranking_policy_id=PORTFOLIO_RANKING_POLICY_ID,
        ranked_projects=tuple(ranked),
    )


__all__ = [
    "MAX_PORTFOLIO_DIFFERENTIATION_ADJUSTMENT",
    "MAX_PORTFOLIO_FEATURE_REFERENCES",
    "MAX_PORTFOLIO_PROJECTS",
    "MAX_PORTFOLIO_RANKING_REASONS",
    "MAX_PORTFOLIO_REDUNDANCY_ADJUSTMENT",
    "MAX_PROJECT_SEMANTIC_TRAITS",
    "PORTFOLIO_DIFFERENTIATION_BASE_FLOOR",
    "PORTFOLIO_DIFFERENTIATION_MASS_FLOOR",
    "PORTFOLIO_FEATURE_INFORMATION_WEIGHTS",
    "PORTFOLIO_RANKING_POLICY_ID",
    "PORTFOLIO_REDUNDANCY_MASS_FLOOR",
    "PORTFOLIO_SCORE_DECIMALS",
    "PortfolioAdjustment",
    "PortfolioProjectRanking",
    "PortfolioRankingError",
    "PortfolioRankingErrorCode",
    "PortfolioRankingReason",
    "ProjectSemanticFootprint",
    "ProjectSemanticTrait",
    "RankedProjectStoryPortfolio",
    "derive_project_semantic_footprint",
    "rank_project_portfolio",
]
