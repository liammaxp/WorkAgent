"""Inherit evidence-backed claim boundaries into capability claim policies.

This module is an in-memory policy layer.  It does not construct final
capability facts, persist artifacts, or alter resume behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_scoring import CapabilitySupportAssessment
from backend.project_capability_taxonomy import get_capability_rule
from backend.project_claim_boundaries import (
    GLOBAL_FORBIDDEN_POLICIES,
    evaluate_project_numeric_claim,
    get_project_claim_boundary_evidence_ids,
    get_project_claim_safety_blockers,
    is_project_resume_metric_claim,
    normalize_project_claim,
    normalize_project_serialized_claim,
    validate_project_claim_boundary,
)
from backend.project_evidence_models import (
    ClaimSubjectType,
    MetricSupport,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
)


CAPABILITY_CLAIM_POLICY_STATUSES = frozenset({
    "boundary_conflict",
    "eligible",
    "ineligible_support",
    "insufficient_allowed_claims",
    "metric_conflict",
    "missing_boundaries",
})

_METRIC_PRIORITY = {
    MetricSupport.NONE.value: 0,
    MetricSupport.APPROXIMATE.value: 1,
    MetricSupport.EXPLICIT.value: 2,
}
_POLICY_HANDLED_BOUNDARY_ERRORS = frozenset({
    "metric_without_supported_evidence",
    "unsupported_metric_claim",
})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_strings(values: Sequence[str]) -> tuple[str, ...]:
    by_identity: dict[str, str] = {}
    for value in values:
        normalized = normalize_project_claim(value)
        if normalized:
            identity = normalized.casefold()
            previous = by_identity.get(identity)
            if previous is None or (normalized.casefold(), normalized) < (previous.casefold(), previous):
                by_identity[identity] = normalized
    return tuple(sorted(by_identity.values(), key=lambda item: (item.casefold(), item)))


@dataclass(frozen=True)
class CapabilityClaimPolicy:
    project_id: str
    capability_type: str
    supporting_evidence_ids: tuple[str, ...]
    inherited_boundary_ids: tuple[str, ...]
    allowed_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    taxonomy_allowed_claims: tuple[str, ...]
    taxonomy_forbidden_claims: tuple[str, ...]
    metric_support: str
    boundary_count: int
    covered_evidence_count: int
    uncovered_evidence_ids: tuple[str, ...]
    has_boundary_conflict: bool
    has_metric_conflict: bool
    policy_status: str
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "supporting_evidence_ids", "inherited_boundary_ids", "allowed_claims",
            "forbidden_claims", "taxonomy_allowed_claims", "taxonomy_forbidden_claims",
            "uncovered_evidence_ids", "reasons",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.metric_support not in _METRIC_PRIORITY:
            raise ValueError("unsupported metric support")
        if self.policy_status not in CAPABILITY_CLAIM_POLICY_STATUSES:
            raise ValueError("unsupported capability claim policy status")
        for name in ("boundary_count", "covered_evidence_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "capability_type": self.capability_type,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "inherited_boundary_ids": list(self.inherited_boundary_ids),
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "taxonomy_allowed_claims": list(self.taxonomy_allowed_claims),
            "taxonomy_forbidden_claims": list(self.taxonomy_forbidden_claims),
            "metric_support": self.metric_support,
            "boundary_count": self.boundary_count,
            "covered_evidence_count": self.covered_evidence_count,
            "uncovered_evidence_ids": list(self.uncovered_evidence_ids),
            "has_boundary_conflict": self.has_boundary_conflict,
            "has_metric_conflict": self.has_metric_conflict,
            "policy_status": self.policy_status,
            "reasons": list(self.reasons),
            "diagnostics": _thaw(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _taxonomy_policy(capability_type: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    definition = get_capability_rule(capability_type)
    if definition is None:
        raise ValueError("candidate capability_type is not present in the canonical taxonomy")
    allowed = _canonical_strings(definition.safe_claim_templates)
    forbidden = _canonical_strings(tuple(f"taxonomy:{item}" for item in definition.forbidden_inferences))
    return definition.capability_type, allowed, forbidden


def _validate_candidate_assessment(
    candidate: CapabilityCandidate,
    assessment: CapabilitySupportAssessment,
) -> None:
    if not isinstance(candidate, CapabilityCandidate):
        raise TypeError("candidate must be a CapabilityCandidate")
    if not isinstance(assessment, CapabilitySupportAssessment):
        raise TypeError("assessment must be a CapabilitySupportAssessment")
    if assessment.project_id != candidate.project_id:
        raise ValueError("assessment project_id does not match candidate")
    if assessment.capability_type != candidate.capability_type:
        raise ValueError("assessment capability_type does not match candidate")
    if assessment.supporting_evidence_ids != candidate.supporting_evidence_ids:
        raise ValueError("assessment Evidence IDs do not match candidate support")
    if assessment.eligibility_status == "eligible" and not all((
        assessment.meets_evidence_minimum,
        assessment.meets_signal_group_minimum,
        assessment.meets_mechanism_minimum,
        not assessment.has_explicit_conflict,
    )):
        raise ValueError("eligible assessment has inconsistent proof state")


def _base_forbidden(taxonomy_forbidden: tuple[str, ...]) -> tuple[str, ...]:
    return _canonical_strings((*GLOBAL_FORBIDDEN_POLICIES, *taxonomy_forbidden))


def _ineligible_policy(
    candidate: CapabilityCandidate,
    capability_type: str,
    taxonomy_allowed: tuple[str, ...],
    taxonomy_forbidden: tuple[str, ...],
) -> CapabilityClaimPolicy:
    return CapabilityClaimPolicy(
        project_id=candidate.project_id,
        capability_type=capability_type,
        supporting_evidence_ids=candidate.supporting_evidence_ids,
        inherited_boundary_ids=(),
        allowed_claims=(),
        forbidden_claims=_base_forbidden(taxonomy_forbidden),
        taxonomy_allowed_claims=taxonomy_allowed,
        taxonomy_forbidden_claims=taxonomy_forbidden,
        metric_support=MetricSupport.NONE.value,
        boundary_count=0,
        covered_evidence_count=0,
        uncovered_evidence_ids=candidate.supporting_evidence_ids,
        has_boundary_conflict=False,
        has_metric_conflict=False,
        policy_status="ineligible_support",
        reasons=("support_assessment_not_eligible",),
        diagnostics={
            "allowed_claim_count": 0,
            "applicable_boundary_count": 0,
            "covered_evidence_count": 0,
            "numeric_claim_block_count": 0,
            "taxonomy_forbidden_claim_count": len(taxonomy_forbidden),
            "uncovered_evidence_count": len(candidate.supporting_evidence_ids),
        },
    )


def _evidence_index_for_candidate(
    candidate: CapabilityCandidate,
    evidence_index: Mapping[str, ProjectEvidenceFact],
) -> dict[str, ProjectEvidenceFact]:
    if not isinstance(evidence_index, Mapping):
        raise TypeError("evidence_index must be a mapping")
    normalized: dict[str, ProjectEvidenceFact] = {}
    for key, fact in evidence_index.items():
        if not isinstance(key, str) or not isinstance(fact, ProjectEvidenceFact):
            raise TypeError("evidence_index must map string IDs to ProjectEvidenceFact values")
        if key != fact.evidence_fact_id:
            raise ValueError("evidence index key does not match Evidence Fact ID")
        if fact.project_id != candidate.project_id:
            raise ValueError("cross-project Evidence Fact reference")
        normalized[key] = fact
    missing = set(candidate.supporting_evidence_ids) - set(normalized)
    if missing:
        raise ValueError("candidate contains a missing Evidence Fact reference")
    return normalized


def _dedupe_boundaries(
    boundaries: Sequence[ProjectClaimBoundary],
) -> tuple[ProjectClaimBoundary, ...]:
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise TypeError("claim_boundaries must be a sequence")
    by_id: dict[str, tuple[str, ProjectClaimBoundary]] = {}
    for boundary in boundaries:
        if not isinstance(boundary, ProjectClaimBoundary):
            raise TypeError("claim_boundaries must contain ProjectClaimBoundary values")
        payload = boundary.to_json()
        previous = by_id.get(boundary.boundary_id)
        if previous is not None and previous[0] != payload:
            raise ValueError("same Boundary ID has conflicting semantic content")
        by_id[boundary.boundary_id] = (payload, boundary)
    return tuple(by_id[key][1] for key in sorted(by_id))


def _applicable_boundaries(
    candidate: CapabilityCandidate,
    evidence_index: Mapping[str, ProjectEvidenceFact],
    boundaries: Sequence[ProjectClaimBoundary],
) -> tuple[ProjectClaimBoundary, ...]:
    support = set(candidate.supporting_evidence_ids)
    applicable: list[ProjectClaimBoundary] = []
    for boundary in _dedupe_boundaries(boundaries):
        if boundary.project_id != candidate.project_id:
            raise ValueError("cross-project Claim Boundary")
        if boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT:
            raise ValueError("unsupported Capability Fact Boundary reference")
        validation = validate_project_claim_boundary(
            boundary,
            evidence_facts_by_id=evidence_index,
        )
        fatal_errors = set(validation.errors) - _POLICY_HANDLED_BOUNDARY_ERRORS
        if fatal_errors:
            raise ValueError("invalid Claim Boundary structure")
        if boundary.subject_type is not ClaimSubjectType.EVIDENCE_FACT:
            continue
        if boundary.subject_id not in evidence_index:
            raise ValueError("Claim Boundary references a missing Evidence Fact")
        linked_ids = set(get_project_claim_boundary_evidence_ids(boundary))
        if boundary.subject_id not in linked_ids:
            raise ValueError("Claim Boundary subject is not supported by its metadata")
        if boundary.subject_id in support:
            if not linked_ids <= support:
                raise ValueError("Claim Boundary includes unsupported Evidence Fact references")
            applicable.append(boundary)
    return tuple(applicable)


def _claim_value(serialized: str) -> str:
    return serialized.partition(":")[2]


def inherit_capability_claim_policy(
    *,
    candidate: CapabilityCandidate,
    assessment: CapabilitySupportAssessment,
    evidence_index: Mapping[str, ProjectEvidenceFact],
    claim_boundaries: Sequence[ProjectClaimBoundary],
) -> CapabilityClaimPolicy:
    """Inherit one conservative claim policy without constructing a final fact."""

    _validate_candidate_assessment(candidate, assessment)
    capability_type, taxonomy_allowed, taxonomy_forbidden = _taxonomy_policy(candidate.capability_type)
    if assessment.eligibility_status != "eligible":
        return _ineligible_policy(candidate, capability_type, taxonomy_allowed, taxonomy_forbidden)

    facts = _evidence_index_for_candidate(candidate, evidence_index)
    applicable = _applicable_boundaries(candidate, facts, claim_boundaries)
    covered_ids = {boundary.subject_id for boundary in applicable}
    uncovered = tuple(sorted(set(candidate.supporting_evidence_ids) - covered_ids))
    inherited_ids = tuple(boundary.boundary_id for boundary in applicable)

    boundary_allowed: list[str] = []
    boundary_forbidden: list[str] = []
    claim_boundaries_by_identity: dict[str, list[ProjectClaimBoundary]] = {}
    for boundary in applicable:
        for value in boundary.allowed_claims:
            normalized = normalize_project_serialized_claim(value)
            boundary_allowed.append(normalized)
            claim_boundaries_by_identity.setdefault(normalized.casefold(), []).append(boundary)
        boundary_forbidden.extend(normalize_project_claim(value) for value in boundary.forbidden_claims)

    allowed = _canonical_strings(tuple(boundary_allowed))
    forbidden = _canonical_strings((*boundary_forbidden, *_base_forbidden(taxonomy_forbidden)))
    forbidden_identities = {value.casefold() for value in forbidden}
    collisions = {value.casefold() for value in allowed} & forbidden_identities
    has_boundary_conflict = bool(collisions)
    has_metric_conflict = False
    numeric_block_count = 0
    reasons: set[str] = set()
    if not applicable:
        reasons.add("no_applicable_boundaries")
    if uncovered:
        reasons.update(f"uncovered_supporting_evidence:{item}" for item in uncovered)
    if collisions:
        reasons.add("allowed_forbidden_claim_collision")

    retained: list[str] = []
    metric_levels: list[str] = []
    for claim in allowed:
        identity = claim.casefold()
        if identity in collisions:
            continue
        blockers = get_project_claim_safety_blockers(_claim_value(claim))
        if blockers:
            has_boundary_conflict = True
            reasons.add("taxonomy_forbidden_claim_collision")
            reasons.update(f"claim_safety_blocked:{item}" for item in blockers)
            continue
        if not is_project_resume_metric_claim(_claim_value(claim)):
            retained.append(claim)
            continue

        supporting_boundaries = claim_boundaries_by_identity.get(identity, [])
        supports: list[str] = []
        claim_supported = True
        for boundary in supporting_boundaries:
            support = boundary.metric_support.value
            metric_allowed, _policies = evaluate_project_numeric_claim(_claim_value(claim), support)
            subject_fact = facts[boundary.subject_id]
            if _METRIC_PRIORITY[support] > _METRIC_PRIORITY[subject_fact.metric_support.value]:
                metric_allowed = False
            supports.append(support)
            if not metric_allowed:
                claim_supported = False
        if not claim_supported or not supports or len(set(supports)) > 1:
            has_metric_conflict = True
            numeric_block_count += 1
            reasons.add("numeric_claim_without_metric_support")
            reasons.add("metric_support_conflict")
            continue
        metric_levels.append(supports[0])
        retained.append(claim)

    allowed = _canonical_strings(tuple(retained))
    metric_support = (
        min(metric_levels, key=lambda value: _METRIC_PRIORITY[value])
        if metric_levels else MetricSupport.NONE.value
    )
    if not allowed:
        reasons.add("allowed_claims_empty")
    if applicable:
        reasons.add("applicable_boundaries_inherited")
    if not uncovered:
        reasons.add("all_supporting_evidence_covered")

    if has_boundary_conflict:
        status = "boundary_conflict"
    elif has_metric_conflict:
        status = "metric_conflict"
    elif not applicable or uncovered:
        status = "missing_boundaries"
    elif not allowed:
        status = "insufficient_allowed_claims"
    else:
        status = "eligible"
        reasons.add("claim_policy_eligible")

    diagnostics = {
        "allowed_claim_count": len(allowed),
        "applicable_boundary_count": len(applicable),
        "boundary_forbidden_claim_count": len(_canonical_strings(tuple(boundary_forbidden))),
        "covered_evidence_count": len(covered_ids),
        "forbidden_claim_count": len(forbidden),
        "numeric_claim_block_count": numeric_block_count,
        "taxonomy_forbidden_claim_count": len(taxonomy_forbidden),
        "uncovered_evidence_count": len(uncovered),
    }
    return CapabilityClaimPolicy(
        project_id=candidate.project_id,
        capability_type=capability_type,
        supporting_evidence_ids=candidate.supporting_evidence_ids,
        inherited_boundary_ids=inherited_ids,
        allowed_claims=allowed,
        forbidden_claims=forbidden,
        taxonomy_allowed_claims=taxonomy_allowed,
        taxonomy_forbidden_claims=taxonomy_forbidden,
        metric_support=metric_support,
        boundary_count=len(applicable),
        covered_evidence_count=len(covered_ids),
        uncovered_evidence_ids=uncovered,
        has_boundary_conflict=has_boundary_conflict,
        has_metric_conflict=has_metric_conflict,
        policy_status=status,
        reasons=tuple(sorted(reasons)),
        diagnostics=diagnostics,
    )


def inherit_project_capability_claim_policies(
    *,
    project_id: str,
    candidates: Sequence[CapabilityCandidate],
    assessments: Sequence[CapabilitySupportAssessment],
    evidence_facts: Sequence[ProjectEvidenceFact],
    claim_boundaries: Sequence[ProjectClaimBoundary],
) -> tuple[CapabilityClaimPolicy, ...]:
    """Inherit policies only for eligible assessments in one exact project."""

    if not isinstance(project_id, str) or not " ".join(project_id.split()):
        raise ValueError("project_id must be a non-blank string")
    normalized_project_id = " ".join(project_id.split())
    if any(not isinstance(item, CapabilityCandidate) for item in candidates):
        raise TypeError("candidates must contain CapabilityCandidate values")
    if any(not isinstance(item, CapabilitySupportAssessment) for item in assessments):
        raise TypeError("assessments must contain CapabilitySupportAssessment values")
    if any(item.project_id != normalized_project_id for item in (*candidates, *assessments)):
        raise ValueError("candidate or assessment project does not match requested project")
    if any(not isinstance(item, ProjectEvidenceFact) for item in evidence_facts):
        raise TypeError("evidence_facts must contain ProjectEvidenceFact values")
    if any(item.project_id != normalized_project_id for item in evidence_facts):
        raise ValueError("Evidence Fact project does not match requested project")
    if any(not isinstance(item, ProjectClaimBoundary) for item in claim_boundaries):
        raise TypeError("claim_boundaries must contain ProjectClaimBoundary values")
    if any(item.project_id != normalized_project_id for item in claim_boundaries):
        raise ValueError("Claim Boundary project does not match requested project")

    evidence_index: dict[str, ProjectEvidenceFact] = {}
    payload_by_evidence_id: dict[str, str] = {}
    for fact in evidence_facts:
        payload = fact.to_json()
        previous = payload_by_evidence_id.get(fact.evidence_fact_id)
        if previous is not None and previous != payload:
            raise ValueError("same Evidence Fact ID has conflicting semantic content")
        payload_by_evidence_id[fact.evidence_fact_id] = payload
        evidence_index[fact.evidence_fact_id] = fact

    candidate_by_type: dict[str, CapabilityCandidate] = {}
    candidate_payloads: dict[str, str] = {}
    for candidate in candidates:
        payload = json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":"))
        previous = candidate_payloads.get(candidate.capability_type)
        if previous is not None and previous != payload:
            raise ValueError("same capability type has conflicting candidates")
        candidate_payloads[candidate.capability_type] = payload
        candidate_by_type[candidate.capability_type] = candidate

    assessment_by_type: dict[str, CapabilitySupportAssessment] = {}
    assessment_payloads: dict[str, str] = {}
    for assessment in assessments:
        payload = assessment.to_json()
        previous = assessment_payloads.get(assessment.capability_type)
        if previous is not None and previous != payload:
            raise ValueError("same capability type has conflicting assessments")
        assessment_payloads[assessment.capability_type] = payload
        assessment_by_type[assessment.capability_type] = assessment
    if set(candidate_by_type) != set(assessment_by_type):
        raise ValueError("every candidate must have exactly one assessment")

    boundaries = _dedupe_boundaries(claim_boundaries)
    output = []
    for capability_type in sorted(candidate_by_type):
        assessment = assessment_by_type[capability_type]
        if assessment.eligibility_status != "eligible":
            continue
        output.append(inherit_capability_claim_policy(
            candidate=candidate_by_type[capability_type],
            assessment=assessment,
            evidence_index=evidence_index,
            claim_boundaries=boundaries,
        ))
    return tuple(output)


__all__ = [
    "CAPABILITY_CLAIM_POLICY_STATUSES",
    "CapabilityClaimPolicy",
    "inherit_capability_claim_policy",
    "inherit_project_capability_claim_policies",
]
