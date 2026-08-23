"""Pure, evidence-bounded reconstruction of one validated engineering-story cluster.

This module resolves semantic fields from authoritative project evidence.  It does
not generate prose, score story value, perform I/O, or establish persistent story
identity.  Claim-boundary and metric policy remain owned by the upstream authority
modules and are reused here as restrictive policy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence, TypeVar

from backend.engineering_story_clustering import (
    StoryCluster,
    StoryClusterLineageState,
    StoryClusterQuality,
)
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryContract,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    MAX_STORY_FIELD_PROVENANCE_IDS,
    MAX_STORY_FIELD_VALUE_LENGTH,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StorySufficiency,
    SufficiencyLevel,
)
from backend.project_capability_extractor import (
    SOURCE_CATEGORY_DIRECT,
    SOURCE_CATEGORY_PROJECT,
    classify_project_evidence_source_category,
    get_independent_project_evidence_identity,
)
from backend.project_claim_boundaries import (
    build_project_evidence_claim_boundary,
    evaluate_project_numeric_claim,
    is_project_resume_metric_claim,
    normalize_project_claim,
    validate_project_claim_boundary,
)
from backend.project_evidence_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)
from backend.project_repository_identity import normalize_project_id


MAX_RECONSTRUCTION_AUTHORITY_RECORDS = 5_000
MAX_RECONSTRUCTION_DIAGNOSTICS = 8

_CLUSTER_ID_RE = re.compile(r"^story_cluster_([0-9a-f]{24})$")
_NUMERIC_RE = re.compile(r"\d")
_OBSERVABLE_OUTCOME_PATTERNS = (
    re.compile(
        r"\b(?:test|tests|regression|validation|check|request|state|startup|failure|error)\b"
        r".{0,120}\b(?:pass(?:es|ed)?|reject(?:s|ed)?|detect(?:s|ed)?|block(?:s|ed)?|"
        r"return(?:s|ed)?|fail(?:s|ed)?|succeed(?:s|ed)?|validat(?:e|es|ed))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:pass(?:es|ed)?|reject(?:s|ed)?|detect(?:s|ed)?|block(?:s|ed)?|"
        r"return(?:s|ed)?|fail(?:s|ed)?|succeed(?:s|ed)?|validat(?:e|es|ed))\b"
        r".{0,120}\b(?:test|tests|regression|validation|check|request|state|startup|failure|error)\b",
        re.IGNORECASE,
    ),
)

_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_FIELD_INDEX = {value: index for index, value in enumerate(_FIELD_ORDER)}


class StoryReconstructionQuality(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"


class StoryReconstructionIdentityState(str, Enum):
    PROVISIONAL_CLUSTER_DERIVED = "provisional_cluster_derived"


class StoryFieldDecisionReason(str, Enum):
    DIRECT_AUTHORITATIVE_EVIDENCE = "direct_authoritative_evidence"
    MULTI_SOURCE_SUPPORT = "multi_source_support"
    CAPABILITY_CONTEXT_SUPPORT = "capability_context_support"
    BOUNDARY_RESTRICTED = "boundary_restricted"
    MISSING_HUMAN_CONTEXT = "missing_human_context"
    NO_DIRECT_EVIDENCE = "no_direct_evidence"
    NON_OBSERVABLE_SAFE_IMPACT = "non_observable_safe_impact"
    UNSUPPORTED_METRIC = "unsupported_metric"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    WEAK_AUTHORITY = "weak_authority"


class StoryReconstructionDiagnosticCode(str, Enum):
    PROVISIONAL_IDENTITY = "provisional_identity"
    PARTIAL_FIELDS = "partial_fields"
    WEAK_CLUSTER = "weak_cluster"
    AMBIGUOUS_CLUSTER = "ambiguous_cluster"
    BOUNDARY_RESTRICTIONS = "boundary_restrictions"
    CONFLICTING_FIELDS = "conflicting_fields"
    UNSUPPORTED_METRICS = "unsupported_metrics"
    BLOCKED_NO_POSITIVE_FIELDS = "blocked_no_positive_fields"


class StoryReconstructionErrorCode(str, Enum):
    INVALID_CLUSTER = "invalid_cluster"
    CROSS_PROJECT_AUTHORITY = "cross_project_authority"
    MISSING_AUTHORITY = "missing_authority"
    CONFLICTING_AUTHORITY = "conflicting_authority"
    INVALID_CLAIM_BOUNDARY = "invalid_claim_boundary"
    BOUND_EXCEEDED = "bound_exceeded"
    STORY_CONTRACT_REJECTED = "story_contract_rejected"


class StoryReconstructionError(ValueError):
    """Bounded deterministic reconstruction failure."""

    def __init__(
        self,
        code: StoryReconstructionErrorCode | str,
        reference_id: str | None = None,
    ) -> None:
        self.code = StoryReconstructionErrorCode(code)
        self.reference_id = _diagnostic_id(reference_id)
        message = self.code.value
        if self.reference_id is not None:
            message += f":{self.reference_id}"
        super().__init__(message)


def _diagnostic_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value if character.isprintable())
    return normalized[:300] or None


@dataclass(frozen=True, slots=True)
class StoryFieldReconstructionDecision(EngineeringStoryContract):
    field_name: EngineeringStoryFieldName
    resulting_state: StoryFieldEvidenceState
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    claim_boundary_ids: tuple[str, ...]
    reason_code: StoryFieldDecisionReason

    def __post_init__(self) -> None:
        field_name = EngineeringStoryFieldName(self.field_name)
        state = StoryFieldEvidenceState(self.resulting_state)
        reason = StoryFieldDecisionReason(self.reason_code)
        # Reuse the accepted Step-1 provenance validator instead of defining a
        # second authority-ID contract in this module.
        validated = EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            evidence_fact_ids=self.evidence_fact_ids,
            capability_fact_ids=self.capability_fact_ids,
            claim_boundary_ids=self.claim_boundary_ids,
        )
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "resulting_state", state)
        object.__setattr__(self, "evidence_fact_ids", validated.evidence_fact_ids)
        object.__setattr__(self, "capability_fact_ids", validated.capability_fact_ids)
        object.__setattr__(self, "claim_boundary_ids", validated.claim_boundary_ids)
        object.__setattr__(self, "reason_code", reason)


@dataclass(frozen=True, slots=True)
class StoryReconstructionResult(EngineeringStoryContract):
    cluster_id: str
    project_id: str
    engineering_story: EngineeringStory | None
    reconstruction_quality: StoryReconstructionQuality
    identity_state: StoryReconstructionIdentityState
    field_decisions: tuple[StoryFieldReconstructionDecision, ...]
    diagnostics: tuple[StoryReconstructionDiagnosticCode, ...]
    unresolved_fields: tuple[EngineeringStoryFieldName, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_id, str) or not _CLUSTER_ID_RE.fullmatch(
            self.cluster_id
        ):
            raise ValueError("cluster_id must be a canonical story-cluster ID")
        project_id = normalize_project_id(self.project_id)
        if not project_id or project_id != self.project_id:
            raise ValueError("project_id must be an exact canonical project identifier")
        quality = StoryReconstructionQuality(self.reconstruction_quality)
        identity_state = StoryReconstructionIdentityState(self.identity_state)
        if (
            isinstance(self.field_decisions, (str, bytes))
            or not isinstance(self.field_decisions, Sequence)
            or any(
                not isinstance(item, StoryFieldReconstructionDecision)
                for item in self.field_decisions
            )
        ):
            raise TypeError("field_decisions must contain field decisions")
        decisions = tuple(sorted(
            self.field_decisions,
            key=lambda item: _FIELD_INDEX[item.field_name],
        ))
        if tuple(item.field_name for item in decisions) != _FIELD_ORDER:
            raise ValueError("field_decisions must contain every story field exactly once")
        if isinstance(self.diagnostics, (str, bytes)) or not isinstance(
            self.diagnostics, Sequence
        ):
            raise TypeError("diagnostics must be a sequence")
        if len(self.diagnostics) > MAX_RECONSTRUCTION_DIAGNOSTICS:
            raise ValueError("diagnostics exceed the reconstruction bound")
        diagnostic_index = {
            value: index for index, value in enumerate(StoryReconstructionDiagnosticCode)
        }
        diagnostics = tuple(sorted(
            {StoryReconstructionDiagnosticCode(value) for value in self.diagnostics},
            key=diagnostic_index.__getitem__,
        ))
        if isinstance(self.unresolved_fields, (str, bytes)) or not isinstance(
            self.unresolved_fields, Sequence
        ):
            raise TypeError("unresolved_fields must be a sequence")
        unresolved = tuple(sorted(
            {EngineeringStoryFieldName(value) for value in self.unresolved_fields},
            key=_FIELD_INDEX.__getitem__,
        ))
        expected_unresolved = tuple(
            item.field_name
            for item in decisions
            if item.resulting_state in {
                StoryFieldEvidenceState.PLAUSIBLE_MISSING,
                StoryFieldEvidenceState.UNSUPPORTED,
            }
        )
        if unresolved != expected_unresolved:
            raise ValueError("unresolved_fields must match non-positive field decisions")
        story = self.engineering_story
        if story is None:
            if quality is not StoryReconstructionQuality.BLOCKED:
                raise ValueError("missing engineering story requires blocked quality")
        else:
            if not isinstance(story, EngineeringStory):
                raise TypeError("engineering_story must be an EngineeringStory or None")
            if story.project_id != project_id:
                raise ValueError("engineering story project must match reconstruction project")
            if quality is StoryReconstructionQuality.BLOCKED:
                raise ValueError("a reconstructed story cannot have blocked quality")
            for decision in decisions:
                field = getattr(story, decision.field_name.value)
                if (
                    field.evidence_state is not decision.resulting_state
                    or field.evidence_fact_ids != decision.evidence_fact_ids
                    or field.capability_fact_ids != decision.capability_fact_ids
                    or field.claim_boundary_ids != decision.claim_boundary_ids
                ):
                    raise ValueError("field decisions must match the reconstructed story")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "reconstruction_quality", quality)
        object.__setattr__(self, "identity_state", identity_state)
        object.__setattr__(self, "field_decisions", decisions)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "unresolved_fields", unresolved)


@dataclass(frozen=True, slots=True)
class _Candidate:
    value: str
    serialized_claim: str
    evidence_fact_id: str
    capability_fact_ids: tuple[str, ...]
    claim_boundary_ids: tuple[str, ...]
    metric_support: MetricSupport = MetricSupport.NONE


@dataclass(frozen=True, slots=True)
class _FieldBuild:
    field: EngineeringStoryField
    decision: StoryFieldReconstructionDecision
    boundary_restricted: bool = False
    conflicting: bool = False
    unsupported_metric: bool = False


_T = TypeVar("_T")


def _canonical_project_id(value: str) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        raise StoryReconstructionError(
            StoryReconstructionErrorCode.CROSS_PROJECT_AUTHORITY,
            value,
        )
    return normalized


def _authority_index(
    records: Sequence[_T],
    *,
    expected_type: type[_T],
    id_attribute: str,
    project_id: str,
) -> dict[str, _T]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("authoritative records must be a sequence")
    if len(records) > MAX_RECONSTRUCTION_AUTHORITY_RECORDS:
        raise StoryReconstructionError(StoryReconstructionErrorCode.BOUND_EXCEEDED)
    result: dict[str, _T] = {}
    payloads: dict[str, str] = {}
    for record in records:
        if not isinstance(record, expected_type):
            raise TypeError(f"authoritative records must contain {expected_type.__name__}")
        if _canonical_project_id(record.project_id) != project_id:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.CROSS_PROJECT_AUTHORITY,
                getattr(record, id_attribute, None),
            )
        identifier = getattr(record, id_attribute)
        payload = record.to_json()
        if identifier in payloads and payloads[identifier] != payload:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.CONFLICTING_AUTHORITY,
                identifier,
            )
        payloads[identifier] = payload
        result[identifier] = record
    return result


def _normalize_serialized_claim(value: str) -> str:
    prefix, separator, claim_value = value.partition(":")
    if not separator:
        return value.casefold()
    return f"{prefix.casefold()}:{normalize_project_claim(claim_value).casefold()}"


def _claims_with_prefix(
    boundary: ProjectClaimBoundary | None,
    prefix: str,
) -> tuple[str, ...]:
    if boundary is None:
        return ()
    values = {
        normalize_project_claim(value.partition(":")[2])
        for value in boundary.allowed_claims
        if value.partition(":")[0] == prefix and value.partition(":")[2]
    }
    return tuple(sorted((value for value in values if value), key=lambda item: (item.casefold(), item)))


def _relevant_boundaries(
    fact: ProjectEvidenceFact,
    *,
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
) -> tuple[ProjectClaimBoundary, ...]:
    capability_ids = {
        capability.capability_id
        for capability in capabilities.values()
        if fact.evidence_fact_id in capability.source_evidence_fact_ids
    }
    return tuple(sorted((
        boundary
        for boundary in boundaries
        if boundary.subject_type is ClaimSubjectType.PROJECT
        or (
            boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
            and boundary.subject_id == fact.evidence_fact_id
        )
        or (
            boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT
            and boundary.subject_id in capability_ids
        )
    ), key=lambda item: item.boundary_id))


def _actual_boundaries_allow(
    fact: ProjectEvidenceFact,
    serialized_claim: str,
    *,
    numeric_metric_claim: str | None,
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
) -> tuple[bool, tuple[str, ...]]:
    relevant = _relevant_boundaries(
        fact,
        boundaries=boundaries,
        capabilities=capabilities,
    )
    boundary_ids = tuple(item.boundary_id for item in relevant)
    normalized = _normalize_serialized_claim(serialized_claim)
    metric_normalized = (
        _normalize_serialized_claim(numeric_metric_claim)
        if numeric_metric_claim is not None
        else None
    )
    for boundary in relevant:
        forbidden = {
            _normalize_serialized_claim(value) for value in boundary.forbidden_claims
        }
        if normalized in forbidden or (
            metric_normalized is not None and metric_normalized in forbidden
        ):
            return False, boundary_ids
        if metric_normalized is not None and boundary.metric_support is MetricSupport.NONE:
            return False, boundary_ids
    fact_boundaries = tuple(
        boundary
        for boundary in relevant
        if boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
        and boundary.subject_id == fact.evidence_fact_id
    )
    for boundary in fact_boundaries:
        allowed = {
            _normalize_serialized_claim(value) for value in boundary.allowed_claims
        }
        if normalized not in allowed:
            return False, boundary_ids
        if metric_normalized is not None and metric_normalized not in allowed:
            return False, boundary_ids
    return True, boundary_ids


def _exact_capability_support(
    fact: ProjectEvidenceFact,
    value: str,
    capabilities: Mapping[str, ProjectCapabilityFact],
) -> tuple[str, ...]:
    normalized = normalize_project_claim(value).casefold()
    return tuple(sorted(
        capability.capability_id
        for capability in capabilities.values()
        if capability.present
        and capability.confidence is not Confidence.LOW
        and fact.evidence_fact_id in capability.source_evidence_fact_ids
        and any(
            normalize_project_claim(mechanism).casefold() == normalized
            for mechanism in capability.mechanisms
        )
    ))


def _candidate(
    fact: ProjectEvidenceFact,
    value: str,
    prefix: str,
    *,
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
    allow_capability_support: bool = False,
    numeric_metric: bool = False,
) -> tuple[_Candidate | None, bool]:
    normalized_value = normalize_project_claim(value)
    serialized = f"{prefix}:{normalized_value}"
    metric_claim = f"metric:{normalized_value}" if numeric_metric else None
    allowed, boundary_ids = _actual_boundaries_allow(
        fact,
        serialized,
        numeric_metric_claim=metric_claim,
        boundaries=boundaries,
        capabilities=capabilities,
    )
    if not allowed:
        return None, True
    capability_ids = (
        _exact_capability_support(fact, normalized_value, capabilities)
        if allow_capability_support
        else ()
    )
    return _Candidate(
        value=normalized_value,
        serialized_claim=serialized,
        evidence_fact_id=fact.evidence_fact_id,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
        metric_support=fact.metric_support,
    ), False


def _stable_ids(values: Sequence[str], *, maximum: int) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if len(result) > maximum:
        raise StoryReconstructionError(StoryReconstructionErrorCode.BOUND_EXCEEDED)
    return result


def _joined_value(values: Sequence[str]) -> str:
    result = "; ".join(sorted(set(values), key=lambda item: (item.casefold(), item)))
    if len(result) > MAX_STORY_FIELD_VALUE_LENGTH:
        raise StoryReconstructionError(StoryReconstructionErrorCode.BOUND_EXCEEDED)
    return result


def _positive_field(
    field_name: EngineeringStoryFieldName,
    candidates: Sequence[_Candidate],
    *,
    composite: bool,
    evidence_facts: Mapping[str, ProjectEvidenceFact],
) -> _FieldBuild:
    by_value: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_value.setdefault(candidate.value.casefold(), []).append(candidate)
    if not composite and len(by_value) > 1:
        evidence_ids = _stable_ids(
            [item.evidence_fact_id for item in candidates],
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        capability_ids = _stable_ids(
            [value for item in candidates for value in item.capability_fact_ids],
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        boundary_ids = _stable_ids(
            [value for item in candidates for value in item.claim_boundary_ids],
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        field = EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            evidence_fact_ids=evidence_ids,
            capability_fact_ids=capability_ids,
            claim_boundary_ids=boundary_ids,
        )
        return _FieldBuild(
            field=field,
            decision=_decision(
                field_name,
                field,
                StoryFieldDecisionReason.CONFLICTING_EVIDENCE,
            ),
            conflicting=True,
        )
    evidence_ids = _stable_ids(
        [item.evidence_fact_id for item in candidates],
        maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
    )
    capability_ids = _stable_ids(
        [value for item in candidates for value in item.capability_fact_ids],
        maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
    )
    boundary_ids = _stable_ids(
        [value for item in candidates for value in item.claim_boundary_ids],
        maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
    )
    values = tuple(item.value for item in candidates)
    value = _joined_value(values) if composite else min(
        set(values), key=lambda item: (item.casefold(), item)
    )
    source_identities = {
        get_independent_project_evidence_identity(evidence_facts[evidence_id])
        for evidence_id in evidence_ids
    }
    if len(source_identities) > 1:
        state = StoryFieldEvidenceState.SUPPORTED
        reason = StoryFieldDecisionReason.MULTI_SOURCE_SUPPORT
    elif capability_ids:
        state = StoryFieldEvidenceState.SUPPORTED
        reason = StoryFieldDecisionReason.CAPABILITY_CONTEXT_SUPPORT
    else:
        state = StoryFieldEvidenceState.CONFIRMED
        reason = StoryFieldDecisionReason.DIRECT_AUTHORITATIVE_EVIDENCE
    field = EngineeringStoryField(
        value=value,
        evidence_state=state,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
    )
    return _FieldBuild(field=field, decision=_decision(field_name, field, reason))


def _missing_field(
    field_name: EngineeringStoryFieldName,
    reason: StoryFieldDecisionReason,
    *,
    unsupported: bool = False,
    evidence_fact_ids: Sequence[str] = (),
    capability_fact_ids: Sequence[str] = (),
    claim_boundary_ids: Sequence[str] = (),
    boundary_restricted: bool = False,
    conflicting: bool = False,
    unsupported_metric: bool = False,
) -> _FieldBuild:
    state = (
        StoryFieldEvidenceState.UNSUPPORTED
        if unsupported
        else StoryFieldEvidenceState.PLAUSIBLE_MISSING
    )
    field = EngineeringStoryField(
        value=None,
        evidence_state=state,
        evidence_fact_ids=tuple(evidence_fact_ids),
        capability_fact_ids=tuple(capability_fact_ids),
        claim_boundary_ids=tuple(claim_boundary_ids),
    )
    return _FieldBuild(
        field=field,
        decision=_decision(field_name, field, reason),
        boundary_restricted=boundary_restricted,
        conflicting=conflicting,
        unsupported_metric=unsupported_metric,
    )


def _decision(
    field_name: EngineeringStoryFieldName,
    field: EngineeringStoryField,
    reason: StoryFieldDecisionReason,
) -> StoryFieldReconstructionDecision:
    return StoryFieldReconstructionDecision(
        field_name=field_name,
        resulting_state=field.evidence_state,
        evidence_fact_ids=field.evidence_fact_ids,
        capability_fact_ids=field.capability_fact_ids,
        claim_boundary_ids=field.claim_boundary_ids,
        reason_code=reason,
    )


def _scalar_field(
    field_name: EngineeringStoryFieldName,
    prefix: str,
    *,
    direct_facts: Sequence[ProjectEvidenceFact],
    policy_boundaries: Mapping[str, ProjectClaimBoundary],
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
    evidence_facts: Mapping[str, ProjectEvidenceFact],
    allow_project_context: bool = False,
    allow_capability_support: bool = False,
) -> _FieldBuild:
    candidates: list[_Candidate] = []
    restricted_ids: set[str] = set()
    restricted_evidence_ids: set[str] = set()
    for fact in direct_facts:
        policy = policy_boundaries.get(fact.evidence_fact_id)
        for value in _claims_with_prefix(policy, prefix):
            item, restricted = _candidate(
                fact,
                value,
                prefix,
                boundaries=boundaries,
                capabilities=capabilities,
                allow_capability_support=allow_capability_support,
            )
            if item is not None:
                candidates.append(item)
            if restricted:
                restricted_evidence_ids.add(fact.evidence_fact_id)
                restricted_ids.update(
                    boundary.boundary_id
                    for boundary in _relevant_boundaries(
                        fact,
                        boundaries=boundaries,
                        capabilities=capabilities,
                    )
                )
    if allow_project_context and candidates:
        direct_values = {item.value.casefold() for item in candidates}
        for fact in evidence_facts.values():
            if (
                fact.status is not EvidenceStatus.SUPPORTING
                or classify_project_evidence_source_category(fact)
                != SOURCE_CATEGORY_PROJECT
            ):
                continue
            policy = policy_boundaries.get(fact.evidence_fact_id)
            for value in _claims_with_prefix(policy, prefix):
                if value.casefold() not in direct_values:
                    continue
                item, restricted = _candidate(
                    fact,
                    value,
                    prefix,
                    boundaries=boundaries,
                    capabilities=capabilities,
                )
                if item is not None:
                    candidates.append(item)
                if restricted:
                    restricted_evidence_ids.add(fact.evidence_fact_id)
                    restricted_ids.update(
                        boundary.boundary_id
                        for boundary in _relevant_boundaries(
                            fact,
                            boundaries=boundaries,
                            capabilities=capabilities,
                        )
                    )
    if candidates:
        return _positive_field(
            field_name,
            candidates,
            composite=False,
            evidence_facts=evidence_facts,
        )
    if restricted_ids:
        return _missing_field(
            field_name,
            StoryFieldDecisionReason.BOUNDARY_RESTRICTED,
            unsupported=True,
            evidence_fact_ids=tuple(sorted(restricted_evidence_ids)),
            claim_boundary_ids=tuple(sorted(restricted_ids)),
            boundary_restricted=True,
        )
    reason = (
        StoryFieldDecisionReason.WEAK_AUTHORITY
        if evidence_facts and not direct_facts
        else StoryFieldDecisionReason.NO_DIRECT_EVIDENCE
    )
    return _missing_field(field_name, reason)


def _composite_field(
    field_name: EngineeringStoryFieldName,
    *,
    direct_facts: Sequence[ProjectEvidenceFact],
    policy_boundaries: Mapping[str, ProjectClaimBoundary],
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
    evidence_facts: Mapping[str, ProjectEvidenceFact],
) -> _FieldBuild:
    candidates: list[_Candidate] = []
    restricted_ids: set[str] = set()
    restricted_evidence_ids: set[str] = set()
    for fact in direct_facts:
        policy = policy_boundaries.get(fact.evidence_fact_id)
        for value in _claims_with_prefix(policy, "implementation"):
            item, restricted = _candidate(
                fact,
                value,
                "implementation",
                boundaries=boundaries,
                capabilities=capabilities,
            )
            if item is not None:
                candidates.append(item)
            if restricted:
                restricted_evidence_ids.add(fact.evidence_fact_id)
                restricted_ids.update(
                    boundary.boundary_id
                    for boundary in _relevant_boundaries(
                        fact,
                        boundaries=boundaries,
                        capabilities=capabilities,
                    )
                )
    if candidates:
        return _positive_field(
            field_name,
            candidates,
            composite=True,
            evidence_facts=evidence_facts,
        )
    if restricted_ids:
        return _missing_field(
            field_name,
            StoryFieldDecisionReason.BOUNDARY_RESTRICTED,
            unsupported=True,
            evidence_fact_ids=tuple(sorted(restricted_evidence_ids)),
            claim_boundary_ids=tuple(sorted(restricted_ids)),
            boundary_restricted=True,
        )
    return _missing_field(
        field_name,
        StoryFieldDecisionReason.WEAK_AUTHORITY
        if evidence_facts and not direct_facts
        else StoryFieldDecisionReason.NO_DIRECT_EVIDENCE,
    )


def _is_observable_outcome(value: str) -> bool:
    return any(pattern.search(value) for pattern in _OBSERVABLE_OUTCOME_PATTERNS)


def _outcome_field(
    *,
    direct_facts: Sequence[ProjectEvidenceFact],
    policy_boundaries: Mapping[str, ProjectClaimBoundary],
    boundaries: Sequence[ProjectClaimBoundary],
    capabilities: Mapping[str, ProjectCapabilityFact],
    evidence_facts: Mapping[str, ProjectEvidenceFact],
) -> _FieldBuild:
    candidates: list[_Candidate] = []
    restricted_ids: set[str] = set()
    restricted_evidence_ids: set[str] = set()
    unsupported_metric_ids: set[str] = set()
    saw_safe_impact = False
    for fact in direct_facts:
        saw_safe_impact = saw_safe_impact or bool(fact.safe_impact)
        policy = policy_boundaries.get(fact.evidence_fact_id)
        policy_impacts = {
            value.casefold(): value for value in _claims_with_prefix(policy, "impact")
        }
        policy_metrics = {
            value.casefold(): value for value in _claims_with_prefix(policy, "metric")
        }
        for raw_value in fact.safe_impact:
            value = normalize_project_claim(raw_value)
            resume_metric = is_project_resume_metric_claim(value)
            if resume_metric:
                metric_allowed, _codes = evaluate_project_numeric_claim(
                    value, fact.metric_support
                )
                if not metric_allowed or value.casefold() not in policy_metrics:
                    unsupported_metric_ids.add(fact.evidence_fact_id)
                    continue
            elif _NUMERIC_RE.search(value):
                # Operational counts, identifiers, and dates are not outcome metrics.
                continue
            elif not _is_observable_outcome(value):
                continue
            if value.casefold() not in policy_impacts:
                continue
            item, restricted = _candidate(
                fact,
                policy_impacts[value.casefold()],
                "impact",
                boundaries=boundaries,
                capabilities=capabilities,
                numeric_metric=resume_metric,
            )
            if item is not None:
                candidates.append(item)
            if restricted:
                restricted_evidence_ids.add(fact.evidence_fact_id)
                restricted_ids.update(
                    boundary.boundary_id
                    for boundary in _relevant_boundaries(
                        fact,
                        boundaries=boundaries,
                        capabilities=capabilities,
                    )
                )
    if candidates:
        return _positive_field(
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            candidates,
            composite=True,
            evidence_facts=evidence_facts,
        )
    if restricted_ids:
        return _missing_field(
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            StoryFieldDecisionReason.BOUNDARY_RESTRICTED,
            unsupported=True,
            evidence_fact_ids=tuple(sorted(restricted_evidence_ids)),
            claim_boundary_ids=tuple(sorted(restricted_ids)),
            boundary_restricted=True,
        )
    if unsupported_metric_ids:
        return _missing_field(
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            StoryFieldDecisionReason.UNSUPPORTED_METRIC,
            unsupported=True,
            evidence_fact_ids=tuple(sorted(unsupported_metric_ids)),
            unsupported_metric=True,
        )
    if saw_safe_impact:
        return _missing_field(
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            StoryFieldDecisionReason.NON_OBSERVABLE_SAFE_IMPACT,
        )
    return _missing_field(
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        StoryFieldDecisionReason.NO_DIRECT_EVIDENCE,
    )


_PRIMARY_STORY_TYPES: Mapping[EvidenceType, EngineeringStoryType] = {
    EvidenceType.ARCHITECTURE: EngineeringStoryType.ARCHITECTURE_CHANGE,
    EvidenceType.BUG_FIX: EngineeringStoryType.DEBUGGING_AND_REPAIR,
    EvidenceType.RETRIEVAL: EngineeringStoryType.RETRIEVAL_REDESIGN,
    EvidenceType.DATA_PERSISTENCE: EngineeringStoryType.DATA_OR_MEMORY_SYSTEM,
    EvidenceType.WORKFLOW: EngineeringStoryType.WORKFLOW_AUTOMATION,
    EvidenceType.OPTIMIZATION: EngineeringStoryType.PERFORMANCE_OR_EFFICIENCY,
    EvidenceType.INTEGRATION: EngineeringStoryType.INTEGRATION,
}


def _story_type(direct_facts: Sequence[ProjectEvidenceFact]) -> EngineeringStoryType:
    primary = {
        _PRIMARY_STORY_TYPES[fact.evidence_type]
        for fact in direct_facts
        if fact.evidence_type in _PRIMARY_STORY_TYPES
    }
    if len(primary) == 1:
        return next(iter(primary))
    if len(primary) > 1:
        return EngineeringStoryType.OTHER
    secondary: set[EngineeringStoryType] = set()
    if any(fact.evidence_type is EvidenceType.FAILURE_RECOVERY for fact in direct_facts):
        secondary.add(EngineeringStoryType.RELIABILITY_HARDENING)
    if any(
        fact.evidence_type in {EvidenceType.VALIDATION, EvidenceType.TESTING}
        for fact in direct_facts
    ):
        secondary.add(EngineeringStoryType.VALIDATION_AND_QUALITY)
    return next(iter(secondary)) if len(secondary) == 1 else EngineeringStoryType.OTHER


def _provisional_story_id(cluster_id: str) -> str:
    match = _CLUSTER_ID_RE.fullmatch(cluster_id)
    if match is None:
        raise StoryReconstructionError(StoryReconstructionErrorCode.INVALID_CLUSTER)
    return f"engineering_story_candidate_{match.group(1)}"


def _validate_cluster_authority(
    cluster: StoryCluster,
    evidence: Mapping[str, ProjectEvidenceFact],
    capabilities: Mapping[str, ProjectCapabilityFact],
    boundaries: Mapping[str, ProjectClaimBoundary],
) -> None:
    for item in cluster.evidence_inputs:
        fact = evidence.get(item.evidence_fact_id)
        if fact is None:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.MISSING_AUTHORITY,
                item.evidence_fact_id,
            )
        if (
            fact.evidence_type is not item.evidence_type
            or fact.status is not item.evidence_status
            or fact.confidence is not item.confidence
            or fact.metric_support is not item.metric_support
            or tuple(fact.technical_tags) != item.technical_tags
        ):
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.CONFLICTING_AUTHORITY,
                item.evidence_fact_id,
            )
    cluster_evidence_ids = {item.evidence_fact_id for item in cluster.evidence_inputs}
    for lineage in cluster.capability_lineages:
        capability = capabilities.get(lineage.capability_id)
        if capability is None:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.MISSING_AUTHORITY,
                lineage.capability_id,
            )
        if (
            capability.capability_type != lineage.capability_type
            or capability.present is not lineage.present
            or capability.confidence is not lineage.confidence
            or capability.metric_support is not lineage.metric_support
            or tuple(capability.source_evidence_fact_ids)
            != lineage.source_evidence_fact_ids
            or not set(capability.source_evidence_fact_ids).issubset(cluster_evidence_ids)
        ):
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.CONFLICTING_AUTHORITY,
                lineage.capability_id,
            )
    for boundary_id in cluster.claim_boundary_ids:
        boundary = boundaries.get(boundary_id)
        if boundary is None:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.MISSING_AUTHORITY,
                boundary_id,
            )
        validation = validate_project_claim_boundary(
            boundary,
            evidence_facts_by_id=evidence,
            capability_facts_by_id=capabilities,
        )
        if not validation.valid:
            raise StoryReconstructionError(
                StoryReconstructionErrorCode.INVALID_CLAIM_BOUNDARY,
                boundary_id,
            )


def reconstruct_engineering_story(
    *,
    cluster: StoryCluster,
    evidence_facts: Sequence[ProjectEvidenceFact],
    capability_facts: Sequence[ProjectCapabilityFact] = (),
    claim_boundaries: Sequence[ProjectClaimBoundary] = (),
) -> StoryReconstructionResult:
    """Reconstruct one bounded story from a validated cluster and typed authority."""

    if not isinstance(cluster, StoryCluster):
        raise TypeError("cluster must be a StoryCluster")
    project_id = _canonical_project_id(cluster.project_id)
    evidence_index = _authority_index(
        evidence_facts,
        expected_type=ProjectEvidenceFact,
        id_attribute="evidence_fact_id",
        project_id=project_id,
    )
    capability_index = _authority_index(
        capability_facts,
        expected_type=ProjectCapabilityFact,
        id_attribute="capability_id",
        project_id=project_id,
    )
    boundary_index = _authority_index(
        claim_boundaries,
        expected_type=ProjectClaimBoundary,
        id_attribute="boundary_id",
        project_id=project_id,
    )
    _validate_cluster_authority(
        cluster,
        evidence_index,
        capability_index,
        boundary_index,
    )
    selected_evidence = {
        item.evidence_fact_id: evidence_index[item.evidence_fact_id]
        for item in cluster.evidence_inputs
    }
    selected_capabilities = {
        item.capability_id: capability_index[item.capability_id]
        for item in cluster.capability_lineages
    }
    selected_boundaries = tuple(
        boundary_index[value] for value in cluster.claim_boundary_ids
    )
    policy_boundaries = {
        identifier: boundary
        for identifier, fact in selected_evidence.items()
        if (boundary := build_project_evidence_claim_boundary(fact)) is not None
    }
    direct_facts = tuple(sorted((
        fact
        for identifier, fact in selected_evidence.items()
        if identifier in policy_boundaries
        and fact.status is EvidenceStatus.ACCEPTED
        and classify_project_evidence_source_category(fact) == SOURCE_CATEGORY_DIRECT
    ), key=lambda item: item.evidence_fact_id))

    try:
        builds: dict[EngineeringStoryFieldName, _FieldBuild] = {}
        builds[EngineeringStoryFieldName.PROBLEM_CONTEXT] = _scalar_field(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            "problem",
            direct_facts=direct_facts,
            policy_boundaries=policy_boundaries,
            boundaries=selected_boundaries,
            capabilities=selected_capabilities,
            evidence_facts=selected_evidence,
            allow_project_context=(
                cluster.lineage_state is not StoryClusterLineageState.AMBIGUOUS
            ),
        )
        builds[EngineeringStoryFieldName.MECHANISM] = _scalar_field(
            EngineeringStoryFieldName.MECHANISM,
            "mechanism",
            direct_facts=direct_facts,
            policy_boundaries=policy_boundaries,
            boundaries=selected_boundaries,
            capabilities=selected_capabilities,
            evidence_facts=selected_evidence,
            allow_capability_support=(
                cluster.lineage_state is not StoryClusterLineageState.AMBIGUOUS
            ),
        )
        builds[EngineeringStoryFieldName.IMPLEMENTATION] = _composite_field(
            EngineeringStoryFieldName.IMPLEMENTATION,
            direct_facts=direct_facts,
            policy_boundaries=policy_boundaries,
            boundaries=selected_boundaries,
            capabilities=selected_capabilities,
            evidence_facts=selected_evidence,
        )
        validation_facts = tuple(
            fact
            for fact in direct_facts
            if fact.evidence_type in {
                EvidenceType.VALIDATION,
                EvidenceType.TESTING,
                EvidenceType.FAILURE_RECOVERY,
            }
        )
        builds[EngineeringStoryFieldName.VALIDATION] = _composite_field(
            EngineeringStoryFieldName.VALIDATION,
            direct_facts=validation_facts,
            policy_boundaries=policy_boundaries,
            boundaries=selected_boundaries,
            capabilities=selected_capabilities,
            evidence_facts=selected_evidence,
        )
        builds[EngineeringStoryFieldName.OBSERVABLE_OUTCOME] = _outcome_field(
            direct_facts=direct_facts,
            policy_boundaries=policy_boundaries,
            boundaries=selected_boundaries,
            capabilities=selected_capabilities,
            evidence_facts=selected_evidence,
        )
        for field_name in (
            EngineeringStoryFieldName.TRIGGER,
            EngineeringStoryFieldName.DECISION,
            EngineeringStoryFieldName.TRADEOFF,
            EngineeringStoryFieldName.OWNERSHIP,
            EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
        ):
            builds[field_name] = _missing_field(
                field_name,
                StoryFieldDecisionReason.MISSING_HUMAN_CONTEXT,
            )
        for field_name in (
            EngineeringStoryFieldName.BEFORE_STATE,
            EngineeringStoryFieldName.AFTER_STATE,
        ):
            builds[field_name] = _missing_field(
                field_name,
                StoryFieldDecisionReason.NO_DIRECT_EVIDENCE,
            )

        ordered_builds = tuple(builds[field_name] for field_name in _FIELD_ORDER)
        positive_count = sum(item.field.has_positive_value for item in ordered_builds)
        diagnostics = {StoryReconstructionDiagnosticCode.PROVISIONAL_IDENTITY}
        if positive_count < len(_FIELD_ORDER):
            diagnostics.add(StoryReconstructionDiagnosticCode.PARTIAL_FIELDS)
        if cluster.quality is StoryClusterQuality.WEAK:
            diagnostics.add(StoryReconstructionDiagnosticCode.WEAK_CLUSTER)
        if cluster.lineage_state is StoryClusterLineageState.AMBIGUOUS:
            diagnostics.add(StoryReconstructionDiagnosticCode.AMBIGUOUS_CLUSTER)
        if any(item.boundary_restricted for item in ordered_builds):
            diagnostics.add(StoryReconstructionDiagnosticCode.BOUNDARY_RESTRICTIONS)
        if any(item.conflicting for item in ordered_builds):
            diagnostics.add(StoryReconstructionDiagnosticCode.CONFLICTING_FIELDS)
        if any(item.unsupported_metric for item in ordered_builds):
            diagnostics.add(StoryReconstructionDiagnosticCode.UNSUPPORTED_METRICS)

        story: EngineeringStory | None
        if positive_count == 0:
            story = None
            quality = StoryReconstructionQuality.BLOCKED
            diagnostics.add(
                StoryReconstructionDiagnosticCode.BLOCKED_NO_POSITIVE_FIELDS
            )
        else:
            if cluster.lineage_state is StoryClusterLineageState.AMBIGUOUS:
                quality = StoryReconstructionQuality.AMBIGUOUS
            elif positive_count == len(_FIELD_ORDER):
                quality = StoryReconstructionQuality.COMPLETE
            elif positive_count >= 3:
                quality = StoryReconstructionQuality.PARTIAL
            else:
                quality = StoryReconstructionQuality.MINIMAL
            story = EngineeringStory(
                story_id=_provisional_story_id(cluster.cluster_id),
                project_id=project_id,
                story_type=_story_type(direct_facts),
                problem_context=builds[EngineeringStoryFieldName.PROBLEM_CONTEXT].field,
                trigger=builds[EngineeringStoryFieldName.TRIGGER].field,
                before_state=builds[EngineeringStoryFieldName.BEFORE_STATE].field,
                decision=builds[EngineeringStoryFieldName.DECISION].field,
                mechanism=builds[EngineeringStoryFieldName.MECHANISM].field,
                implementation=builds[EngineeringStoryFieldName.IMPLEMENTATION].field,
                tradeoff=builds[EngineeringStoryFieldName.TRADEOFF].field,
                validation=builds[EngineeringStoryFieldName.VALIDATION].field,
                after_state=builds[EngineeringStoryFieldName.AFTER_STATE].field,
                observable_outcome=builds[
                    EngineeringStoryFieldName.OBSERVABLE_OUTCOME
                ].field,
                ownership=builds[EngineeringStoryFieldName.OWNERSHIP].field,
                stakeholder_context=builds[
                    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT
                ].field,
                evidence_fact_ids=tuple(selected_evidence),
                capability_fact_ids=tuple(selected_capabilities),
                claim_boundary_ids=tuple(
                    boundary.boundary_id for boundary in selected_boundaries
                ),
                lifecycle=EngineeringStoryLifecycle(
                    status=EngineeringStoryStatus.ACTIVE,
                    requires_revalidation=False,
                ),
                claim_sufficiency=ClaimSufficiency(
                    level=SufficiencyLevel.UNASSESSED
                ),
                story_sufficiency=StorySufficiency(
                    level=SufficiencyLevel.UNASSESSED
                ),
                opportunity=StoryOpportunity(level=StoryOpportunityLevel.NONE),
            )
        unresolved = tuple(
            item.decision.field_name
            for item in ordered_builds
            if not item.field.has_positive_value
        )
        return StoryReconstructionResult(
            cluster_id=cluster.cluster_id,
            project_id=project_id,
            engineering_story=story,
            reconstruction_quality=quality,
            identity_state=StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED,
            field_decisions=tuple(item.decision for item in ordered_builds),
            diagnostics=tuple(diagnostics),
            unresolved_fields=unresolved,
        )
    except StoryReconstructionError:
        raise
    except (TypeError, ValueError) as exc:
        raise StoryReconstructionError(
            StoryReconstructionErrorCode.STORY_CONTRACT_REJECTED,
            cluster.cluster_id,
        ) from exc


def reconstruct_engineering_story_from_memory(
    *,
    cluster: StoryCluster,
    project_memory: ProjectEvidenceMemory,
) -> StoryReconstructionResult:
    """Reconstruct from one already-loaded authoritative project memory object."""

    if not isinstance(project_memory, ProjectEvidenceMemory):
        raise TypeError("project_memory must be a ProjectEvidenceMemory")
    if not isinstance(cluster, StoryCluster):
        raise TypeError("cluster must be a StoryCluster")
    if project_memory.project_id != cluster.project_id:
        raise StoryReconstructionError(
            StoryReconstructionErrorCode.CROSS_PROJECT_AUTHORITY,
            project_memory.project_memory_id,
        )
    return reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=tuple(project_memory.evidence_facts),
        capability_facts=tuple(project_memory.capability_facts),
        claim_boundaries=tuple(project_memory.claim_boundaries),
    )


__all__ = [
    "MAX_RECONSTRUCTION_AUTHORITY_RECORDS",
    "MAX_RECONSTRUCTION_DIAGNOSTICS",
    "StoryFieldDecisionReason",
    "StoryFieldReconstructionDecision",
    "StoryReconstructionDiagnosticCode",
    "StoryReconstructionError",
    "StoryReconstructionErrorCode",
    "StoryReconstructionIdentityState",
    "StoryReconstructionQuality",
    "StoryReconstructionResult",
    "reconstruct_engineering_story",
    "reconstruct_engineering_story_from_memory",
]
