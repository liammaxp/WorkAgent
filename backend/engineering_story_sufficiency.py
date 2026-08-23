"""Deterministic sufficiency evaluation for reconstructed engineering stories.

This module evaluates only accepted engineering-story reconstruction output and
same-project evidence authority.  It performs no retrieval, persistence,
opportunity detection, hiring-context analysis, question generation, or model
calls.  Claim sufficiency and causal-story sufficiency remain independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any

from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryContract,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryStatus,
    MAX_STORY_PROVENANCE_IDS,
    StoryContextGap,
    StoryFieldEvidenceState,
    StorySufficiency,
    SufficiencyLevel,
    validate_engineering_story_id,
)
from backend.engineering_story_reconstruction import (
    StoryFieldDecisionReason,
    StoryFieldReconstructionDecision,
    StoryReconstructionQuality,
    StoryReconstructionResult,
)
from backend.project_claim_boundaries import (
    evaluate_project_numeric_claim,
    is_project_resume_metric_claim,
    normalize_project_claim,
    validate_project_claim_boundary,
)
from backend.project_evidence_models import (
    ClaimSubjectType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)
from backend.project_repository_identity import normalize_project_id


MAX_SUFFICIENCY_DIAGNOSTICS = 12

_CLUSTER_ID_RE = re.compile(r"^story_cluster_[0-9a-f]{24}$")

_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_FIELD_INDEX = {value: index for index, value in enumerate(_FIELD_ORDER)}


class SufficiencyDimensionDomain(str, Enum):
    CLAIM = "claim"
    STORY = "story"


class SufficiencyDimension(str, Enum):
    TECHNICAL_CORE = "technical_core"
    MECHANISM_SUPPORT = "mechanism_support"
    IMPLEMENTATION_SUPPORT = "implementation_support"
    VALIDATION_SUPPORT = "validation_support"
    OUTCOME_SUPPORT = "outcome_support"
    METRIC_SUPPORT = "metric_support"
    CLAIM_BOUNDARY_SAFETY = "claim_boundary_safety"
    CONFLICT_STATE = "conflict_state"
    CONTEXT = "context"
    ENGINEERING_ACTION = "engineering_action"
    VALIDATION_STATE_CHANGE = "validation_state_change"
    HUMAN_AGENCY = "human_agency"


class SufficiencyDimensionStatus(str, Enum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    MISSING = "missing"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class SufficiencyDecisionReason(str, Enum):
    DIRECT_CONFIRMED_SUPPORT = "direct_confirmed_support"
    SUPPORTED_AUTHORITY = "supported_authority"
    PARTIAL_FIELD_SUPPORT = "partial_field_support"
    MISSING_CONTEXT = "missing_context"
    UNSUPPORTED_FIELD = "unsupported_field"
    BOUNDARY_RESTRICTED = "boundary_restricted"
    METRIC_NOT_SUPPORTED = "metric_not_supported"
    APPROXIMATE_METRIC_SUPPORT = "approximate_metric_support"
    EXPLICIT_METRIC_SUPPORT = "explicit_metric_support"
    NON_NUMERIC_CLAIMS_AVAILABLE = "non_numeric_claims_available"
    CONFLICTED_FIELD = "conflicted_field"
    AMBIGUOUS_RECONSTRUCTION = "ambiguous_reconstruction"
    BLOCKED_RECONSTRUCTION = "blocked_reconstruction"
    LIFECYCLE_REVALIDATION_REQUIRED = "lifecycle_revalidation_required"
    LIFECYCLE_INACTIVE = "lifecycle_inactive"
    OPTIONAL_CONTEXT_MISSING = "optional_context_missing"
    NO_POSITIVE_SUPPORT = "no_positive_support"
    NO_BOUNDARY_RESTRICTION = "no_boundary_restriction"
    NO_CONFLICT_DETECTED = "no_conflict_detected"


class SufficiencyDiagnosticCode(str, Enum):
    CLAIM_STORY_LEVEL_DIVERGENCE = "claim_story_level_divergence"
    RECONSTRUCTION_BLOCKED = "reconstruction_blocked"
    RECONSTRUCTION_AMBIGUOUS = "reconstruction_ambiguous"
    RECONSTRUCTION_MINIMAL = "reconstruction_minimal"
    CLAIM_BOUNDARY_RESTRICTION = "claim_boundary_restriction"
    METRIC_UNSUPPORTED = "metric_unsupported"
    FIELD_CONFLICT = "field_conflict"
    LIFECYCLE_REVALIDATION_REQUIRED = "lifecycle_revalidation_required"
    LIFECYCLE_INACTIVE = "lifecycle_inactive"


class StorySufficiencyEvaluationErrorCode(str, Enum):
    INVALID_RECONSTRUCTION = "invalid_reconstruction"
    CROSS_PROJECT_AUTHORITY = "cross_project_authority"
    MISSING_AUTHORITY = "missing_authority"
    INVALID_AUTHORITY = "invalid_authority"
    INVALID_CLAIM_BOUNDARY = "invalid_claim_boundary"
    INCONSISTENT_PROVENANCE = "inconsistent_provenance"
    BOUND_EXCEEDED = "bound_exceeded"


class StorySufficiencyEvaluationError(ValueError):
    """Bounded fail-closed error for invalid sufficiency inputs."""

    def __init__(
        self,
        code: StorySufficiencyEvaluationErrorCode | str,
        reference_id: str | None = None,
    ) -> None:
        self.code = StorySufficiencyEvaluationErrorCode(code)
        self.reference_id = _bounded_reference(reference_id)
        message = self.code.value
        if self.reference_id is not None:
            message += f":{self.reference_id}"
        super().__init__(message)


_CLAIM_DIMENSIONS = (
    SufficiencyDimension.TECHNICAL_CORE,
    SufficiencyDimension.MECHANISM_SUPPORT,
    SufficiencyDimension.IMPLEMENTATION_SUPPORT,
    SufficiencyDimension.VALIDATION_SUPPORT,
    SufficiencyDimension.OUTCOME_SUPPORT,
    SufficiencyDimension.METRIC_SUPPORT,
    SufficiencyDimension.CLAIM_BOUNDARY_SAFETY,
    SufficiencyDimension.CONFLICT_STATE,
)
_STORY_DIMENSIONS = (
    SufficiencyDimension.CONTEXT,
    SufficiencyDimension.ENGINEERING_ACTION,
    SufficiencyDimension.VALIDATION_STATE_CHANGE,
    SufficiencyDimension.HUMAN_AGENCY,
)
_DIMENSION_INDEX = {
    value: index for index, value in enumerate(_CLAIM_DIMENSIONS + _STORY_DIMENSIONS)
}
_DIAGNOSTIC_INDEX = {
    value: index for index, value in enumerate(SufficiencyDiagnosticCode)
}

_CLAIM_FIELDS = (
    EngineeringStoryFieldName.PROBLEM_CONTEXT,
    EngineeringStoryFieldName.MECHANISM,
    EngineeringStoryFieldName.IMPLEMENTATION,
    EngineeringStoryFieldName.VALIDATION,
    EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
)
_CONTEXT_FIELDS = (
    EngineeringStoryFieldName.PROBLEM_CONTEXT,
    EngineeringStoryFieldName.TRIGGER,
    EngineeringStoryFieldName.BEFORE_STATE,
)
_ACTION_FIELDS = (
    EngineeringStoryFieldName.DECISION,
    EngineeringStoryFieldName.MECHANISM,
    EngineeringStoryFieldName.IMPLEMENTATION,
    EngineeringStoryFieldName.TRADEOFF,
)
_VALIDATION_FIELDS = (
    EngineeringStoryFieldName.VALIDATION,
    EngineeringStoryFieldName.AFTER_STATE,
    EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
)
_HUMAN_FIELDS = (
    EngineeringStoryFieldName.OWNERSHIP,
    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
)


def _bounded_reference(value: Any) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    normalized = "".join(character for character in value if character.isprintable())
    return normalized[:300] or None


def _stable_enum_values(
    values: Sequence[Any],
    enum_type: type[Enum],
    *,
    maximum: int,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    index = {value: position for position, value in enumerate(enum_type)}
    normalized = {enum_type(value) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__))


def _stable_authority_ids(
    values: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{kind} must be a sequence")
    if len(values) > MAX_STORY_PROVENANCE_IDS:
        raise ValueError(
            f"{kind} exceeds maximum item count {MAX_STORY_PROVENANCE_IDS}"
        )
    normalized = tuple(sorted(set(values)))
    for value in normalized:
        kwargs: dict[str, tuple[str, ...]] = {
            "evidence_fact_ids": (),
            "capability_fact_ids": (),
            "claim_boundary_ids": (),
        }
        kwargs[kind] = (value,)
        EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            **kwargs,
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SufficiencyDimensionDecision(EngineeringStoryContract):
    domain: SufficiencyDimensionDomain
    dimension: SufficiencyDimension
    status: SufficiencyDimensionStatus
    supporting_story_fields: tuple[EngineeringStoryFieldName, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()
    capability_fact_ids: tuple[str, ...] = ()
    claim_boundary_ids: tuple[str, ...] = ()
    reason_code: SufficiencyDecisionReason = SufficiencyDecisionReason.NO_POSITIVE_SUPPORT
    metric_support: MetricSupport | None = None

    def __post_init__(self) -> None:
        domain = SufficiencyDimensionDomain(self.domain)
        dimension = SufficiencyDimension(self.dimension)
        status = SufficiencyDimensionStatus(self.status)
        reason = SufficiencyDecisionReason(self.reason_code)
        expected_domain = (
            SufficiencyDimensionDomain.CLAIM
            if dimension in _CLAIM_DIMENSIONS
            else SufficiencyDimensionDomain.STORY
        )
        if domain is not expected_domain:
            raise ValueError("dimension does not belong to the supplied domain")
        story_fields = _stable_enum_values(
            self.supporting_story_fields,
            EngineeringStoryFieldName,
            maximum=len(_FIELD_ORDER),
            name="supporting_story_fields",
        )
        evidence_ids = _stable_authority_ids(
            self.evidence_fact_ids,
            kind="evidence_fact_ids",
        )
        capability_ids = _stable_authority_ids(
            self.capability_fact_ids,
            kind="capability_fact_ids",
        )
        boundary_ids = _stable_authority_ids(
            self.claim_boundary_ids,
            kind="claim_boundary_ids",
        )
        metric_support = self.metric_support
        if dimension is SufficiencyDimension.METRIC_SUPPORT:
            if metric_support is None:
                raise ValueError("metric dimension requires an explicit MetricSupport value")
            metric_support = MetricSupport(metric_support)
        elif metric_support is not None:
            raise ValueError("only the metric dimension may carry metric_support")
        if status is SufficiencyDimensionStatus.SATISFIED and not story_fields:
            if dimension not in {
                SufficiencyDimension.CLAIM_BOUNDARY_SAFETY,
                SufficiencyDimension.CONFLICT_STATE,
            }:
                raise ValueError("satisfied dimension requires a supporting story field")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "supporting_story_fields", story_fields)
        object.__setattr__(self, "evidence_fact_ids", evidence_ids)
        object.__setattr__(self, "capability_fact_ids", capability_ids)
        object.__setattr__(self, "claim_boundary_ids", boundary_ids)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "metric_support", metric_support)


@dataclass(frozen=True, slots=True)
class EngineeringStorySufficiencyResult(EngineeringStoryContract):
    cluster_id: str
    project_id: str
    story_id: str | None
    evaluated_story: EngineeringStory | None
    reconstruction_quality: StoryReconstructionQuality
    claim_sufficiency: ClaimSufficiency
    story_sufficiency: StorySufficiency
    claim_dimension_decisions: tuple[SufficiencyDimensionDecision, ...]
    story_dimension_decisions: tuple[SufficiencyDimensionDecision, ...]
    story_context_gaps: tuple[StoryContextGap, ...]
    diagnostics: tuple[SufficiencyDiagnosticCode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_id, str) or not _CLUSTER_ID_RE.fullmatch(
            self.cluster_id
        ):
            raise ValueError("cluster_id must be a canonical story-cluster ID")
        project_id = normalize_project_id(self.project_id)
        if not project_id or project_id != self.project_id:
            raise ValueError("project_id must be an exact canonical project identifier")
        quality = StoryReconstructionQuality(self.reconstruction_quality)
        if not isinstance(self.claim_sufficiency, ClaimSufficiency):
            raise TypeError("claim_sufficiency must be a ClaimSufficiency")
        if not isinstance(self.story_sufficiency, StorySufficiency):
            raise TypeError("story_sufficiency must be a StorySufficiency")
        claim_decisions = _normalize_decisions(
            self.claim_dimension_decisions,
            expected=_CLAIM_DIMENSIONS,
            domain=SufficiencyDimensionDomain.CLAIM,
            name="claim_dimension_decisions",
        )
        story_decisions = _normalize_decisions(
            self.story_dimension_decisions,
            expected=_STORY_DIMENSIONS,
            domain=SufficiencyDimensionDomain.STORY,
            name="story_dimension_decisions",
        )
        gaps = _stable_enum_values(
            self.story_context_gaps,
            StoryContextGap,
            maximum=len(StoryContextGap),
            name="story_context_gaps",
        )
        diagnostics = _stable_enum_values(
            self.diagnostics,
            SufficiencyDiagnosticCode,
            maximum=MAX_SUFFICIENCY_DIAGNOSTICS,
            name="diagnostics",
        )
        story = self.evaluated_story
        story_id = self.story_id
        if story is None:
            if story_id is not None:
                raise ValueError("a missing evaluated story cannot carry story_id")
            if quality is not StoryReconstructionQuality.BLOCKED:
                raise ValueError("a missing evaluated story requires blocked reconstruction")
            if (
                self.claim_sufficiency.level is not SufficiencyLevel.LOW
                or self.story_sufficiency.level is not SufficiencyLevel.LOW
            ):
                raise ValueError("blocked reconstruction requires low sufficiency")
        else:
            if not isinstance(story, EngineeringStory):
                raise TypeError("evaluated_story must be an EngineeringStory or None")
            if story.project_id != project_id:
                raise ValueError("evaluated story project must match result project")
            if story_id is None or validate_engineering_story_id(story_id) != story.story_id:
                raise ValueError("story_id must match the evaluated story")
            if quality is StoryReconstructionQuality.BLOCKED:
                raise ValueError("a present evaluated story cannot be blocked")
            if story.claim_sufficiency != self.claim_sufficiency:
                raise ValueError("evaluated story ClaimSufficiency must match result")
            if story.story_sufficiency != self.story_sufficiency:
                raise ValueError("evaluated story StorySufficiency must match result")
            expected_gaps = _story_context_gaps(_story_fields(story))
            if gaps != expected_gaps:
                raise ValueError("story_context_gaps must match non-positive story fields")
            story_fields = _story_fields(story)
            expected_claim_supported = _positive(story_fields, _CLAIM_FIELDS)
            expected_claim_missing = tuple(
                name for name in _CLAIM_FIELDS if name not in expected_claim_supported
            )
            expected_story_supported = _positive(story_fields, _FIELD_ORDER)
            expected_story_missing = tuple(
                name for name in _FIELD_ORDER if name not in expected_story_supported
            )
            if (
                self.claim_sufficiency.supported_fields != expected_claim_supported
                or self.claim_sufficiency.missing_fields != expected_claim_missing
            ):
                raise ValueError("ClaimSufficiency must exactly classify claim fields")
            if (
                self.story_sufficiency.supported_fields != expected_story_supported
                or self.story_sufficiency.missing_fields != expected_story_missing
            ):
                raise ValueError("StorySufficiency must exactly classify story fields")
        all_decisions = claim_decisions + story_decisions
        if story is None:
            if any(
                item.supporting_story_fields
                or item.evidence_fact_ids
                or item.capability_fact_ids
                or item.claim_boundary_ids
                for item in all_decisions
            ):
                raise ValueError("blocked decisions cannot retain story provenance")
        else:
            for decision in all_decisions:
                if any(
                    not story_fields[field_name].has_positive_value
                    for field_name in decision.supporting_story_fields
                ):
                    raise ValueError("dimension support must reference positive story fields")
                if not set(decision.evidence_fact_ids).issubset(story.evidence_fact_ids):
                    raise ValueError("dimension evidence IDs must exist in story provenance")
                if not set(decision.capability_fact_ids).issubset(
                    story.capability_fact_ids
                ):
                    raise ValueError("dimension capability IDs must exist in story provenance")
                if not set(decision.claim_boundary_ids).issubset(
                    story.claim_boundary_ids
                ):
                    raise ValueError("dimension boundary IDs must exist in story provenance")
            claim_by_dimension = {item.dimension: item for item in claim_decisions}
            story_by_dimension = {item.dimension: item for item in story_decisions}
            if self.claim_sufficiency.level is SufficiencyLevel.HIGH and (
                claim_by_dimension[SufficiencyDimension.TECHNICAL_CORE].status
                is not SufficiencyDimensionStatus.SATISFIED
                or quality
                in {
                    StoryReconstructionQuality.AMBIGUOUS,
                    StoryReconstructionQuality.MINIMAL,
                }
                or story.lifecycle.status is not EngineeringStoryStatus.ACTIVE
                or story.lifecycle.requires_revalidation
            ):
                raise ValueError("high ClaimSufficiency conflicts with result constraints")
            if self.story_sufficiency.level is SufficiencyLevel.HIGH and (
                any(
                    story_by_dimension[dimension].status
                    is not SufficiencyDimensionStatus.SATISFIED
                    for dimension in (
                        SufficiencyDimension.CONTEXT,
                        SufficiencyDimension.ENGINEERING_ACTION,
                        SufficiencyDimension.VALIDATION_STATE_CHANGE,
                    )
                )
                or quality
                in {
                    StoryReconstructionQuality.AMBIGUOUS,
                    StoryReconstructionQuality.MINIMAL,
                }
                or story.lifecycle.status is not EngineeringStoryStatus.ACTIVE
                or story.lifecycle.requires_revalidation
            ):
                raise ValueError("high StorySufficiency conflicts with result constraints")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "reconstruction_quality", quality)
        object.__setattr__(self, "claim_dimension_decisions", claim_decisions)
        object.__setattr__(self, "story_dimension_decisions", story_decisions)
        object.__setattr__(self, "story_context_gaps", gaps)
        object.__setattr__(self, "diagnostics", diagnostics)


def _normalize_decisions(
    values: Sequence[SufficiencyDimensionDecision],
    *,
    expected: tuple[SufficiencyDimension, ...],
    domain: SufficiencyDimensionDomain,
    name: str,
) -> tuple[SufficiencyDimensionDecision, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if any(not isinstance(item, SufficiencyDimensionDecision) for item in values):
        raise TypeError(f"{name} must contain SufficiencyDimensionDecision values")
    decisions = tuple(sorted(values, key=lambda item: _DIMENSION_INDEX[item.dimension]))
    if tuple(item.dimension for item in decisions) != expected:
        raise ValueError(f"{name} must contain every expected dimension exactly once")
    if any(item.domain is not domain for item in decisions):
        raise ValueError(f"{name} contains a decision from another domain")
    return decisions


def _story_fields(story: EngineeringStory) -> Mapping[EngineeringStoryFieldName, EngineeringStoryField]:
    return {field_name: getattr(story, field_name.value) for field_name in _FIELD_ORDER}


def _decision_map(
    reconstruction: StoryReconstructionResult,
) -> Mapping[EngineeringStoryFieldName, StoryFieldReconstructionDecision]:
    return {item.field_name: item for item in reconstruction.field_decisions}


def _positive(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    names: Sequence[EngineeringStoryFieldName],
) -> tuple[EngineeringStoryFieldName, ...]:
    return tuple(name for name in names if fields[name].has_positive_value)


def _provenance(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    names: Sequence[EngineeringStoryFieldName],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    evidence_ids = tuple(sorted({
        value for name in names for value in fields[name].evidence_fact_ids
    }))
    capability_ids = tuple(sorted({
        value for name in names for value in fields[name].capability_fact_ids
    }))
    boundary_ids = tuple(sorted({
        value for name in names for value in fields[name].claim_boundary_ids
    }))
    if max(len(evidence_ids), len(capability_ids), len(boundary_ids)) > MAX_STORY_PROVENANCE_IDS:
        raise StorySufficiencyEvaluationError(
            StorySufficiencyEvaluationErrorCode.BOUND_EXCEEDED
        )
    return evidence_ids, capability_ids, boundary_ids


def _support_reason(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    positive_names: Sequence[EngineeringStoryFieldName],
) -> SufficiencyDecisionReason:
    if positive_names and all(
        fields[name].evidence_state is StoryFieldEvidenceState.CONFIRMED
        for name in positive_names
    ):
        return SufficiencyDecisionReason.DIRECT_CONFIRMED_SUPPORT
    return SufficiencyDecisionReason.SUPPORTED_AUTHORITY


def _dimension_decision(
    *,
    domain: SufficiencyDimensionDomain,
    dimension: SufficiencyDimension,
    status: SufficiencyDimensionStatus,
    reason: SufficiencyDecisionReason,
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    relevant_fields: Sequence[EngineeringStoryFieldName],
    supporting_fields: Sequence[EngineeringStoryFieldName] | None = None,
    metric_support: MetricSupport | None = None,
) -> SufficiencyDimensionDecision:
    positive_names = tuple(
        supporting_fields if supporting_fields is not None else _positive(fields, relevant_fields)
    )
    evidence_ids, capability_ids, boundary_ids = _provenance(fields, relevant_fields)
    return SufficiencyDimensionDecision(
        domain=domain,
        dimension=dimension,
        status=status,
        supporting_story_fields=positive_names,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
        reason_code=reason,
        metric_support=metric_support,
    )


def _single_field_decision(
    *,
    dimension: SufficiencyDimension,
    field_name: EngineeringStoryFieldName,
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyDimensionDecision:
    field = fields[field_name]
    reconstruction = reconstruction_decisions[field_name]
    if reconstruction.reason_code is StoryFieldDecisionReason.CONFLICTING_EVIDENCE:
        status = SufficiencyDimensionStatus.BLOCKED
        reason = SufficiencyDecisionReason.CONFLICTED_FIELD
    elif reconstruction.reason_code is StoryFieldDecisionReason.BOUNDARY_RESTRICTED:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.BOUNDARY_RESTRICTED
    elif field.has_positive_value:
        status = SufficiencyDimensionStatus.SATISFIED
        reason = _support_reason(fields, (field_name,))
    elif field.evidence_state is StoryFieldEvidenceState.UNSUPPORTED:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.UNSUPPORTED_FIELD
    else:
        status = SufficiencyDimensionStatus.MISSING
        reason = SufficiencyDecisionReason.NO_POSITIVE_SUPPORT
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.CLAIM,
        dimension=dimension,
        status=status,
        reason=reason,
        fields=fields,
        relevant_fields=(field_name,),
    )


def _technical_core_decision(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyDimensionDecision:
    names = (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    )
    positive_names = _positive(fields, names)
    reasons = {reconstruction_decisions[name].reason_code for name in names}
    if StoryFieldDecisionReason.CONFLICTING_EVIDENCE in reasons:
        status = SufficiencyDimensionStatus.BLOCKED
        reason = SufficiencyDecisionReason.CONFLICTED_FIELD
    elif StoryFieldDecisionReason.BOUNDARY_RESTRICTED in reasons:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.BOUNDARY_RESTRICTED
    elif len(positive_names) == len(names):
        status = SufficiencyDimensionStatus.SATISFIED
        reason = _support_reason(fields, positive_names)
    elif positive_names:
        status = SufficiencyDimensionStatus.PARTIAL
        reason = SufficiencyDecisionReason.PARTIAL_FIELD_SUPPORT
    elif any(fields[name].evidence_state is StoryFieldEvidenceState.UNSUPPORTED for name in names):
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.UNSUPPORTED_FIELD
    else:
        status = SufficiencyDimensionStatus.MISSING
        reason = SufficiencyDecisionReason.NO_POSITIVE_SUPPORT
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.CLAIM,
        dimension=SufficiencyDimension.TECHNICAL_CORE,
        status=status,
        reason=reason,
        fields=fields,
        relevant_fields=names,
        supporting_fields=positive_names,
    )


def _metric_segments(value: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for segment in value.split(";")
        if (normalized := normalize_project_claim(segment))
        and is_project_resume_metric_claim(normalized)
    )


def _metric_support_for_outcome(
    outcome: EngineeringStoryField,
    *,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
    capability_by_id: Mapping[str, ProjectCapabilityFact],
    boundary_by_id: Mapping[str, ProjectClaimBoundary],
) -> MetricSupport:
    if outcome.value is None:
        return MetricSupport.NONE
    segments = _metric_segments(outcome.value)
    if not segments:
        return MetricSupport.NONE
    supports: list[MetricSupport] = []
    metric_fact_ids: set[str] = set()
    for segment in segments:
        segment_key = normalize_project_claim(segment).casefold()
        matched = False
        matched_fact_ids: set[str] = set()
        for evidence_id in outcome.evidence_fact_ids:
            fact = evidence_by_id[evidence_id]
            fact_impacts = {
                normalize_project_claim(value).casefold()
                for value in fact.safe_impact
                if is_project_resume_metric_claim(value)
            }
            if segment_key not in fact_impacts:
                continue
            allowed, _codes = evaluate_project_numeric_claim(segment, fact.metric_support)
            if not allowed:
                return MetricSupport.NONE
            supports.append(fact.metric_support)
            matched_fact_ids.add(evidence_id)
            metric_fact_ids.add(evidence_id)
            matched = True
        if not matched:
            return MetricSupport.NONE
        serialized_metric = f"metric:{normalize_project_claim(segment)}".casefold()
        if not any(
            boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
            and boundary.subject_id in matched_fact_ids
            and serialized_metric
            in {
                normalize_project_claim(value).casefold()
                for value in boundary.allowed_claims
            }
            for boundary_id in outcome.claim_boundary_ids
            for boundary in (boundary_by_id[boundary_id],)
        ):
            return MetricSupport.NONE
    relevant_boundaries = tuple(
        boundary_by_id[boundary_id]
        for boundary_id in outcome.claim_boundary_ids
        if (
            boundary_by_id[boundary_id].subject_type is ClaimSubjectType.PROJECT
            or (
                boundary_by_id[boundary_id].subject_type
                is ClaimSubjectType.EVIDENCE_FACT
                and boundary_by_id[boundary_id].subject_id in metric_fact_ids
            )
            or (
                boundary_by_id[boundary_id].subject_type
                is ClaimSubjectType.CAPABILITY_FACT
                and boundary_by_id[boundary_id].subject_id in capability_by_id
                and any(
                    evidence_id in metric_fact_ids
                    for evidence_id in capability_by_id[
                        boundary_by_id[boundary_id].subject_id
                    ].source_evidence_fact_ids
                )
            )
        )
    )
    boundary_support = tuple(boundary.metric_support for boundary in relevant_boundaries)
    if not boundary_support or MetricSupport.NONE in boundary_support:
        return MetricSupport.NONE
    if MetricSupport.NONE in supports or not supports:
        return MetricSupport.NONE
    if (
        MetricSupport.APPROXIMATE in supports
        or MetricSupport.APPROXIMATE in boundary_support
    ):
        return MetricSupport.APPROXIMATE
    return MetricSupport.EXPLICIT


def _metric_decision(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
    *,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
    capability_by_id: Mapping[str, ProjectCapabilityFact],
    boundary_by_id: Mapping[str, ProjectClaimBoundary],
) -> SufficiencyDimensionDecision:
    name = EngineeringStoryFieldName.OBSERVABLE_OUTCOME
    outcome = fields[name]
    reconstruction = reconstruction_decisions[name]
    segments = _metric_segments(outcome.value) if outcome.value is not None else ()
    if reconstruction.reason_code is StoryFieldDecisionReason.UNSUPPORTED_METRIC:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.METRIC_NOT_SUPPORTED
        support = MetricSupport.NONE
    elif reconstruction.reason_code is StoryFieldDecisionReason.BOUNDARY_RESTRICTED:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.BOUNDARY_RESTRICTED
        support = MetricSupport.NONE
    elif not segments:
        status = SufficiencyDimensionStatus.NOT_APPLICABLE
        reason = SufficiencyDecisionReason.NON_NUMERIC_CLAIMS_AVAILABLE
        support = MetricSupport.NONE
    else:
        support = _metric_support_for_outcome(
            outcome,
            evidence_by_id=evidence_by_id,
            capability_by_id=capability_by_id,
            boundary_by_id=boundary_by_id,
        )
        if support is MetricSupport.EXPLICIT:
            status = SufficiencyDimensionStatus.SATISFIED
            reason = SufficiencyDecisionReason.EXPLICIT_METRIC_SUPPORT
        elif support is MetricSupport.APPROXIMATE:
            status = SufficiencyDimensionStatus.PARTIAL
            reason = SufficiencyDecisionReason.APPROXIMATE_METRIC_SUPPORT
        else:
            status = SufficiencyDimensionStatus.RESTRICTED
            reason = SufficiencyDecisionReason.METRIC_NOT_SUPPORTED
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.CLAIM,
        dimension=SufficiencyDimension.METRIC_SUPPORT,
        status=status,
        reason=reason,
        fields=fields,
        relevant_fields=(name,),
        metric_support=support,
    )


def _boundary_safety_decision(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyDimensionDecision:
    restricted = tuple(
        name
        for name in _CLAIM_FIELDS
        if reconstruction_decisions[name].reason_code
        is StoryFieldDecisionReason.BOUNDARY_RESTRICTED
    )
    positive_names = _positive(fields, _CLAIM_FIELDS)
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.CLAIM,
        dimension=SufficiencyDimension.CLAIM_BOUNDARY_SAFETY,
        status=(
            SufficiencyDimensionStatus.RESTRICTED
            if restricted
            else SufficiencyDimensionStatus.SATISFIED
        ),
        reason=(
            SufficiencyDecisionReason.BOUNDARY_RESTRICTED
            if restricted
            else SufficiencyDecisionReason.NO_BOUNDARY_RESTRICTION
        ),
        fields=fields,
        relevant_fields=_CLAIM_FIELDS,
        supporting_fields=positive_names,
    )


def _conflict_decision(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyDimensionDecision:
    conflicts = tuple(
        name
        for name in _CLAIM_FIELDS
        if reconstruction_decisions[name].reason_code
        is StoryFieldDecisionReason.CONFLICTING_EVIDENCE
    )
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.CLAIM,
        dimension=SufficiencyDimension.CONFLICT_STATE,
        status=(
            SufficiencyDimensionStatus.BLOCKED
            if conflicts
            else SufficiencyDimensionStatus.SATISFIED
        ),
        reason=(
            SufficiencyDecisionReason.CONFLICTED_FIELD
            if conflicts
            else SufficiencyDecisionReason.NO_CONFLICT_DETECTED
        ),
        fields=fields,
        relevant_fields=conflicts,
        supporting_fields=(),
    )


def _story_group_decision(
    *,
    dimension: SufficiencyDimension,
    names: tuple[EngineeringStoryFieldName, ...],
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyDimensionDecision:
    positive_names = _positive(fields, names)
    conflict_names = tuple(
        name
        for name in names
        if reconstruction_decisions[name].reason_code
        is StoryFieldDecisionReason.CONFLICTING_EVIDENCE
    )
    restricted_names = tuple(
        name
        for name in names
        if reconstruction_decisions[name].reason_code
        is StoryFieldDecisionReason.BOUNDARY_RESTRICTED
    )
    if dimension is SufficiencyDimension.CONTEXT:
        complete = fields[EngineeringStoryFieldName.PROBLEM_CONTEXT].has_positive_value
        critical_conflict = EngineeringStoryFieldName.PROBLEM_CONTEXT in conflict_names
        critical_restriction = EngineeringStoryFieldName.PROBLEM_CONTEXT in restricted_names
    elif dimension is SufficiencyDimension.ENGINEERING_ACTION:
        complete = all(
            fields[name].has_positive_value
            for name in (
                EngineeringStoryFieldName.MECHANISM,
                EngineeringStoryFieldName.IMPLEMENTATION,
            )
        )
        critical_conflict = any(
            name in conflict_names
            for name in (
                EngineeringStoryFieldName.MECHANISM,
                EngineeringStoryFieldName.IMPLEMENTATION,
            )
        )
        critical_restriction = any(
            name in restricted_names
            for name in (
                EngineeringStoryFieldName.MECHANISM,
                EngineeringStoryFieldName.IMPLEMENTATION,
            )
        )
    elif dimension is SufficiencyDimension.VALIDATION_STATE_CHANGE:
        validation = fields[EngineeringStoryFieldName.VALIDATION].has_positive_value
        state_change = any(
            fields[name].has_positive_value
            for name in (
                EngineeringStoryFieldName.AFTER_STATE,
                EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            )
        )
        complete = validation and state_change
        critical_conflict = EngineeringStoryFieldName.VALIDATION in conflict_names
        critical_restriction = EngineeringStoryFieldName.VALIDATION in restricted_names
    else:
        complete = len(positive_names) == len(names)
        critical_conflict = False
        critical_restriction = False
    if critical_conflict:
        status = SufficiencyDimensionStatus.BLOCKED
        reason = SufficiencyDecisionReason.CONFLICTED_FIELD
    elif critical_restriction:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.BOUNDARY_RESTRICTED
    elif complete:
        status = SufficiencyDimensionStatus.SATISFIED
        reason = _support_reason(fields, positive_names)
    elif positive_names:
        status = SufficiencyDimensionStatus.PARTIAL
        reason = SufficiencyDecisionReason.PARTIAL_FIELD_SUPPORT
    elif restricted_names:
        status = SufficiencyDimensionStatus.RESTRICTED
        reason = SufficiencyDecisionReason.BOUNDARY_RESTRICTED
    else:
        status = SufficiencyDimensionStatus.MISSING
        reason = (
            SufficiencyDecisionReason.OPTIONAL_CONTEXT_MISSING
            if dimension is SufficiencyDimension.HUMAN_AGENCY
            else SufficiencyDecisionReason.MISSING_CONTEXT
        )
    return _dimension_decision(
        domain=SufficiencyDimensionDomain.STORY,
        dimension=dimension,
        status=status,
        reason=reason,
        fields=fields,
        relevant_fields=names,
        supporting_fields=positive_names,
    )


def _claim_level(
    story: EngineeringStory,
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    reconstruction: StoryReconstructionResult,
    reconstruction_decisions: Mapping[
        EngineeringStoryFieldName, StoryFieldReconstructionDecision
    ],
) -> SufficiencyLevel:
    mechanism = fields[EngineeringStoryFieldName.MECHANISM].has_positive_value
    implementation = fields[EngineeringStoryFieldName.IMPLEMENTATION].has_positive_value
    supporting_context = any(
        fields[name].has_positive_value
        for name in (
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        )
    )
    critical_conflict = any(
        reconstruction_decisions[name].reason_code
        is StoryFieldDecisionReason.CONFLICTING_EVIDENCE
        for name in (
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
        )
    )
    if critical_conflict:
        level = SufficiencyLevel.LOW
    elif mechanism and implementation and supporting_context:
        level = SufficiencyLevel.HIGH
    elif mechanism or implementation:
        level = SufficiencyLevel.MEDIUM
    else:
        level = SufficiencyLevel.LOW
    if reconstruction.reconstruction_quality in {
        StoryReconstructionQuality.AMBIGUOUS,
        StoryReconstructionQuality.MINIMAL,
    } and level is SufficiencyLevel.HIGH:
        level = SufficiencyLevel.MEDIUM
    if story.lifecycle.status in {
        EngineeringStoryStatus.SUPERSEDED,
        EngineeringStoryStatus.STALE,
        EngineeringStoryStatus.CONFLICTED,
    }:
        level = SufficiencyLevel.LOW
    elif story.lifecycle.requires_revalidation and level is SufficiencyLevel.HIGH:
        level = SufficiencyLevel.MEDIUM
    return level


def _story_level(
    story: EngineeringStory,
    reconstruction: StoryReconstructionResult,
    decisions: Mapping[SufficiencyDimension, SufficiencyDimensionDecision],
) -> SufficiencyLevel:
    context = decisions[SufficiencyDimension.CONTEXT].status
    action = decisions[SufficiencyDimension.ENGINEERING_ACTION].status
    validation = decisions[SufficiencyDimension.VALIDATION_STATE_CHANGE].status
    positive_statuses = {
        SufficiencyDimensionStatus.SATISFIED,
        SufficiencyDimensionStatus.PARTIAL,
    }
    if all(
        value is SufficiencyDimensionStatus.SATISFIED
        for value in (context, action, validation)
    ):
        level = SufficiencyLevel.HIGH
    elif all(value in positive_statuses for value in (context, action, validation)):
        level = SufficiencyLevel.MEDIUM
    elif (
        context in positive_statuses
        and validation in positive_statuses
        and decisions[SufficiencyDimension.ENGINEERING_ACTION].supporting_story_fields
    ):
        level = SufficiencyLevel.MEDIUM
    elif any(value in positive_statuses for value in (context, action, validation)):
        level = SufficiencyLevel.LOW
    else:
        level = SufficiencyLevel.LOW
    if reconstruction.reconstruction_quality is StoryReconstructionQuality.MINIMAL:
        level = SufficiencyLevel.LOW
    elif (
        reconstruction.reconstruction_quality is StoryReconstructionQuality.AMBIGUOUS
        and level is SufficiencyLevel.HIGH
    ):
        level = SufficiencyLevel.MEDIUM
    if story.lifecycle.status in {
        EngineeringStoryStatus.SUPERSEDED,
        EngineeringStoryStatus.STALE,
        EngineeringStoryStatus.CONFLICTED,
    } or story.lifecycle.requires_revalidation:
        if level is SufficiencyLevel.HIGH:
            level = SufficiencyLevel.MEDIUM
    return level


def _story_context_gaps(
    fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
) -> tuple[StoryContextGap, ...]:
    pairs = (
        (EngineeringStoryFieldName.PROBLEM_CONTEXT, StoryContextGap.PROBLEM_CONTEXT),
        (EngineeringStoryFieldName.TRIGGER, StoryContextGap.TRIGGER),
        (EngineeringStoryFieldName.DECISION, StoryContextGap.DECISION_REASON),
        (EngineeringStoryFieldName.TRADEOFF, StoryContextGap.TRADEOFF),
        (EngineeringStoryFieldName.OWNERSHIP, StoryContextGap.OWNERSHIP),
        (
            EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
            StoryContextGap.STAKEHOLDER_CONTEXT,
        ),
        (
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            StoryContextGap.OBSERVABLE_OUTCOME_CONTEXT,
        ),
    )
    return tuple(gap for field_name, gap in pairs if not fields[field_name].has_positive_value)


def _authority_maps(
    memory: ProjectEvidenceMemory,
    *,
    project_id: str,
) -> tuple[
    Mapping[str, ProjectEvidenceFact],
    Mapping[str, ProjectCapabilityFact],
    Mapping[str, ProjectClaimBoundary],
]:
    if not isinstance(memory, ProjectEvidenceMemory):
        raise TypeError("project_memory must be a ProjectEvidenceMemory")
    normalized = normalize_project_id(memory.project_id)
    if not normalized or normalized != memory.project_id or normalized != project_id:
        raise StorySufficiencyEvaluationError(
            StorySufficiencyEvaluationErrorCode.CROSS_PROJECT_AUTHORITY,
            memory.project_id,
        )
    collections = (
        (memory.evidence_facts, ProjectEvidenceFact, "evidence_fact_id"),
        (memory.capability_facts, ProjectCapabilityFact, "capability_id"),
        (memory.claim_boundaries, ProjectClaimBoundary, "boundary_id"),
    )
    maps: list[dict[str, Any]] = []
    for values, expected_type, id_name in collections:
        result: dict[str, Any] = {}
        for item in values:
            if not isinstance(item, expected_type):
                raise StorySufficiencyEvaluationError(
                    StorySufficiencyEvaluationErrorCode.INVALID_AUTHORITY
                )
            if item.project_id != project_id:
                raise StorySufficiencyEvaluationError(
                    StorySufficiencyEvaluationErrorCode.CROSS_PROJECT_AUTHORITY,
                    getattr(item, id_name, None),
                )
            item_id = getattr(item, id_name)
            if item_id in result:
                raise StorySufficiencyEvaluationError(
                    StorySufficiencyEvaluationErrorCode.INVALID_AUTHORITY,
                    item_id,
                )
            result[item_id] = item
        maps.append(result)
    return maps[0], maps[1], maps[2]


def _validate_story_authority(
    story: EngineeringStory,
    *,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
    capability_by_id: Mapping[str, ProjectCapabilityFact],
    boundary_by_id: Mapping[str, ProjectClaimBoundary],
) -> None:
    for evidence_id in story.evidence_fact_ids:
        fact = evidence_by_id.get(evidence_id)
        if fact is None:
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.MISSING_AUTHORITY,
                evidence_id,
            )
        if not fact.source_refs or any(
            source_ref.project_id != story.project_id for source_ref in fact.source_refs
        ):
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.INCONSISTENT_PROVENANCE,
                evidence_id,
            )
    for capability_id in story.capability_fact_ids:
        capability = capability_by_id.get(capability_id)
        if capability is None:
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.MISSING_AUTHORITY,
                capability_id,
            )
        if not capability.present or any(
            evidence_id not in story.evidence_fact_ids
            for evidence_id in capability.source_evidence_fact_ids
        ):
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.INVALID_AUTHORITY,
                capability_id,
            )
    for boundary_id in story.claim_boundary_ids:
        boundary = boundary_by_id.get(boundary_id)
        if boundary is None:
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.MISSING_AUTHORITY,
                boundary_id,
            )
        validation = validate_project_claim_boundary(
            boundary,
            evidence_facts_by_id=evidence_by_id,
            capability_facts_by_id=capability_by_id,
        )
        if not validation.valid:
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.INVALID_CLAIM_BOUNDARY,
                boundary_id,
            )
        if (
            boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
            and boundary.subject_id not in story.evidence_fact_ids
        ) or (
            boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT
            and boundary.subject_id not in story.capability_fact_ids
        ):
            raise StorySufficiencyEvaluationError(
                StorySufficiencyEvaluationErrorCode.INCONSISTENT_PROVENANCE,
                boundary_id,
            )


def _blocked_result(
    reconstruction: StoryReconstructionResult,
) -> EngineeringStorySufficiencyResult:
    empty_fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField] = {
        name: EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        )
        for name in _FIELD_ORDER
    }
    claim_decisions = tuple(
        _dimension_decision(
            domain=SufficiencyDimensionDomain.CLAIM,
            dimension=dimension,
            status=SufficiencyDimensionStatus.BLOCKED,
            reason=SufficiencyDecisionReason.BLOCKED_RECONSTRUCTION,
            fields=empty_fields,
            relevant_fields=(),
            metric_support=(
                MetricSupport.NONE
                if dimension is SufficiencyDimension.METRIC_SUPPORT
                else None
            ),
        )
        for dimension in _CLAIM_DIMENSIONS
    )
    story_decisions = tuple(
        _dimension_decision(
            domain=SufficiencyDimensionDomain.STORY,
            dimension=dimension,
            status=SufficiencyDimensionStatus.BLOCKED,
            reason=SufficiencyDecisionReason.BLOCKED_RECONSTRUCTION,
            fields=empty_fields,
            relevant_fields=(),
        )
        for dimension in _STORY_DIMENSIONS
    )
    return EngineeringStorySufficiencyResult(
        cluster_id=reconstruction.cluster_id,
        project_id=reconstruction.project_id,
        story_id=None,
        evaluated_story=None,
        reconstruction_quality=reconstruction.reconstruction_quality,
        claim_sufficiency=ClaimSufficiency(level=SufficiencyLevel.LOW),
        story_sufficiency=StorySufficiency(level=SufficiencyLevel.LOW),
        claim_dimension_decisions=claim_decisions,
        story_dimension_decisions=story_decisions,
        story_context_gaps=tuple(StoryContextGap),
        diagnostics=(SufficiencyDiagnosticCode.RECONSTRUCTION_BLOCKED,),
    )


def evaluate_engineering_story_sufficiency(
    *,
    reconstruction_result: StoryReconstructionResult,
    project_memory: ProjectEvidenceMemory,
) -> EngineeringStorySufficiencyResult:
    """Evaluate claim and causal-story sufficiency without changing authority."""

    if not isinstance(reconstruction_result, StoryReconstructionResult):
        raise TypeError("reconstruction_result must be a StoryReconstructionResult")
    evidence_by_id, capability_by_id, boundary_by_id = _authority_maps(
        project_memory,
        project_id=reconstruction_result.project_id,
    )
    story = reconstruction_result.engineering_story
    if story is None:
        return _blocked_result(reconstruction_result)
    _validate_story_authority(
        story,
        evidence_by_id=evidence_by_id,
        capability_by_id=capability_by_id,
        boundary_by_id=boundary_by_id,
    )
    fields = _story_fields(story)
    reconstruction_decisions = _decision_map(reconstruction_result)

    claim_decisions = (
        _technical_core_decision(fields, reconstruction_decisions),
        _single_field_decision(
            dimension=SufficiencyDimension.MECHANISM_SUPPORT,
            field_name=EngineeringStoryFieldName.MECHANISM,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _single_field_decision(
            dimension=SufficiencyDimension.IMPLEMENTATION_SUPPORT,
            field_name=EngineeringStoryFieldName.IMPLEMENTATION,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _single_field_decision(
            dimension=SufficiencyDimension.VALIDATION_SUPPORT,
            field_name=EngineeringStoryFieldName.VALIDATION,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _single_field_decision(
            dimension=SufficiencyDimension.OUTCOME_SUPPORT,
            field_name=EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _metric_decision(
            fields,
            reconstruction_decisions,
            evidence_by_id=evidence_by_id,
            capability_by_id=capability_by_id,
            boundary_by_id=boundary_by_id,
        ),
        _boundary_safety_decision(fields, reconstruction_decisions),
        _conflict_decision(fields, reconstruction_decisions),
    )
    story_decisions = (
        _story_group_decision(
            dimension=SufficiencyDimension.CONTEXT,
            names=_CONTEXT_FIELDS,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _story_group_decision(
            dimension=SufficiencyDimension.ENGINEERING_ACTION,
            names=_ACTION_FIELDS,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _story_group_decision(
            dimension=SufficiencyDimension.VALIDATION_STATE_CHANGE,
            names=_VALIDATION_FIELDS,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
        _story_group_decision(
            dimension=SufficiencyDimension.HUMAN_AGENCY,
            names=_HUMAN_FIELDS,
            fields=fields,
            reconstruction_decisions=reconstruction_decisions,
        ),
    )
    story_decision_map = {item.dimension: item for item in story_decisions}
    claim_level = _claim_level(
        story,
        fields,
        reconstruction_result,
        reconstruction_decisions,
    )
    story_level = _story_level(story, reconstruction_result, story_decision_map)
    claim_supported = _positive(fields, _CLAIM_FIELDS)
    claim_missing = tuple(name for name in _CLAIM_FIELDS if name not in claim_supported)
    story_supported = _positive(fields, _FIELD_ORDER)
    story_missing = tuple(name for name in _FIELD_ORDER if name not in story_supported)
    claim_sufficiency = ClaimSufficiency(
        level=claim_level,
        supported_fields=claim_supported,
        missing_fields=claim_missing,
    )
    story_sufficiency = StorySufficiency(
        level=story_level,
        supported_fields=story_supported,
        missing_fields=story_missing,
    )
    evaluated_story = replace(
        story,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
    )
    diagnostics: set[SufficiencyDiagnosticCode] = set()
    if claim_level is not story_level:
        diagnostics.add(SufficiencyDiagnosticCode.CLAIM_STORY_LEVEL_DIVERGENCE)
    if reconstruction_result.reconstruction_quality is StoryReconstructionQuality.AMBIGUOUS:
        diagnostics.add(SufficiencyDiagnosticCode.RECONSTRUCTION_AMBIGUOUS)
    if reconstruction_result.reconstruction_quality is StoryReconstructionQuality.MINIMAL:
        diagnostics.add(SufficiencyDiagnosticCode.RECONSTRUCTION_MINIMAL)
    if any(
        item.reason_code is StoryFieldDecisionReason.BOUNDARY_RESTRICTED
        for item in reconstruction_result.field_decisions
    ):
        diagnostics.add(SufficiencyDiagnosticCode.CLAIM_BOUNDARY_RESTRICTION)
    if any(
        item.reason_code is StoryFieldDecisionReason.UNSUPPORTED_METRIC
        for item in reconstruction_result.field_decisions
    ):
        diagnostics.add(SufficiencyDiagnosticCode.METRIC_UNSUPPORTED)
    if any(
        item.reason_code is StoryFieldDecisionReason.CONFLICTING_EVIDENCE
        for item in reconstruction_result.field_decisions
    ):
        diagnostics.add(SufficiencyDiagnosticCode.FIELD_CONFLICT)
    if story.lifecycle.requires_revalidation:
        diagnostics.add(SufficiencyDiagnosticCode.LIFECYCLE_REVALIDATION_REQUIRED)
    if story.lifecycle.status is not EngineeringStoryStatus.ACTIVE:
        diagnostics.add(SufficiencyDiagnosticCode.LIFECYCLE_INACTIVE)
    return EngineeringStorySufficiencyResult(
        cluster_id=reconstruction_result.cluster_id,
        project_id=reconstruction_result.project_id,
        story_id=story.story_id,
        evaluated_story=evaluated_story,
        reconstruction_quality=reconstruction_result.reconstruction_quality,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        claim_dimension_decisions=claim_decisions,
        story_dimension_decisions=story_decisions,
        story_context_gaps=_story_context_gaps(fields),
        diagnostics=tuple(sorted(diagnostics, key=_DIAGNOSTIC_INDEX.__getitem__)),
    )


__all__ = [
    "EngineeringStorySufficiencyResult",
    "MAX_SUFFICIENCY_DIAGNOSTICS",
    "StorySufficiencyEvaluationError",
    "StorySufficiencyEvaluationErrorCode",
    "SufficiencyDecisionReason",
    "SufficiencyDiagnosticCode",
    "SufficiencyDimension",
    "SufficiencyDimensionDecision",
    "SufficiencyDimensionDomain",
    "SufficiencyDimensionStatus",
    "evaluate_engineering_story_sufficiency",
]
