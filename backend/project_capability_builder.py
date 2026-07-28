"""Build verified project capability facts from completed lifecycle records.

This module is an in-memory, fail-closed construction layer.  It reuses the
authoritative evidence-memory fact model and performs no persistence, retrieval,
generation, network access, or pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeVar

from backend.project_capability_boundaries import CapabilityClaimPolicy
from backend.project_capability_extractor import (
    derive_project_capability_confidence,
    select_project_evidence_mechanisms,
    select_project_evidence_technical_tags,
)
from backend.project_capability_memory import (
    CapabilityCandidate,
    validate_project_capability_fact,
)
from backend.project_capability_scoring import (
    CapabilitySupportAssessment,
    is_concrete_project_evidence_mechanism,
)
from backend.project_capability_taxonomy import get_capability_rule
from backend.project_claim_boundaries import (
    evaluate_project_numeric_claim,
    is_project_resume_metric_claim,
    normalize_project_claim,
    normalize_project_serialized_claim,
)
from backend.project_evidence_models import (
    Confidence,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
)


PROJECT_CAPABILITY_FACT_BUILD_STATUSES = frozenset({
    "built",
    "ineligible_support",
    "ineligible_claim_policy",
    "identity_mismatch",
    "evidence_mismatch",
    "missing_mechanisms",
    "missing_allowed_claims",
    "invalid_metric_support",
    "invalid_fact",
})

_UNSAFE_TAG_ORIGIN_RE = re.compile(
    r"(?:^(?:jd|job[_ -]?description|resume|taxonomy|user[_ -]?profile|"
    r"llm[_ -]?suggestion)\s*[:=]|\b(?:jd|job description|resume|taxonomy|"
    r"llm)[_ -]derived\b)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/[^/])")
_SENSITIVE_TAG_RE = re.compile(
    r"(?:-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"\b(?:gh[oprsu]_|sk-)[A-Za-z0-9_-]{16,})",
    re.IGNORECASE,
)
_T = TypeVar("_T")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(value[key]) for key in sorted(value)})
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


def _canonical_allowed_claims(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_project_serialized_claim(value) for value in values)
    return _canonical_strings(normalized)


@dataclass(frozen=True)
class ProjectCapabilityFactBuildResult:
    project_id: str
    capability_type: str
    build_status: str
    fact: ProjectCapabilityFact | None
    supporting_evidence_ids: tuple[str, ...]
    inherited_boundary_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.build_status not in PROJECT_CAPABILITY_FACT_BUILD_STATUSES:
            raise ValueError("unsupported project capability fact build status")
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "inherited_boundary_ids", tuple(self.inherited_boundary_ids))
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
        if self.build_status == "built" and self.fact is None:
            raise ValueError("built result requires a capability fact")
        if self.build_status != "built" and self.fact is not None:
            raise ValueError("failed result cannot contain a capability fact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "capability_type": self.capability_type,
            "build_status": self.build_status,
            "fact": self.fact.to_dict() if self.fact is not None else None,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "inherited_boundary_ids": list(self.inherited_boundary_ids),
            "reasons": list(self.reasons),
            "diagnostics": _thaw(self.diagnostics),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result(
    candidate: CapabilityCandidate,
    policy: CapabilityClaimPolicy,
    status: str,
    *reasons: str,
    fact: ProjectCapabilityFact | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> ProjectCapabilityFactBuildResult:
    return ProjectCapabilityFactBuildResult(
        project_id=candidate.project_id,
        capability_type=candidate.capability_type,
        build_status=status,
        fact=fact,
        supporting_evidence_ids=candidate.supporting_evidence_ids,
        inherited_boundary_ids=tuple(sorted(set(policy.inherited_boundary_ids))),
        reasons=tuple(reasons),
        diagnostics=diagnostics or {},
    )


def _base_diagnostics(
    assessment: CapabilitySupportAssessment,
    policy: CapabilityClaimPolicy,
) -> dict[str, Any]:
    return {
        "assessment_status": assessment.eligibility_status,
        "policy_status": policy.policy_status,
        "support_score": assessment.support_score,
        "specificity_score": assessment.specificity_score,
        "supporting_evidence_count": len(assessment.supporting_evidence_ids),
        "inherited_boundary_count": len(policy.inherited_boundary_ids),
    }


def _identity_reasons(
    candidate: CapabilityCandidate,
    assessment: CapabilitySupportAssessment,
    policy: CapabilityClaimPolicy,
) -> tuple[str, ...]:
    reasons = []
    if assessment.project_id != candidate.project_id:
        reasons.append("assessment_project_mismatch")
    if policy.project_id != candidate.project_id:
        reasons.append("policy_project_mismatch")
    if assessment.capability_type != candidate.capability_type:
        reasons.append("assessment_capability_mismatch")
    if policy.capability_type != candidate.capability_type:
        reasons.append("policy_capability_mismatch")
    return tuple(reasons)


def _is_safe_project_technical_tag(value: str) -> bool:
    return not any((
        _UNSAFE_TAG_ORIGIN_RE.search(value),
        _ABSOLUTE_PATH_RE.search(value),
        _SENSITIVE_TAG_RE.search(value),
    ))


def _resolve_supporting_evidence(
    candidate: CapabilityCandidate,
    evidence_index: Mapping[str, ProjectEvidenceFact],
) -> tuple[tuple[ProjectEvidenceFact, ...] | None, tuple[str, ...]]:
    if not isinstance(evidence_index, Mapping):
        raise TypeError("evidence_index must be a mapping")
    facts = []
    reasons = []
    for evidence_id in candidate.supporting_evidence_ids:
        fact = evidence_index.get(evidence_id)
        if fact is None:
            reasons.append("missing_evidence_reference")
            continue
        if not isinstance(fact, ProjectEvidenceFact):
            raise TypeError("evidence_index values must be ProjectEvidenceFact values")
        if fact.evidence_fact_id != evidence_id:
            reasons.append("evidence_index_identity_mismatch")
        elif fact.project_id != candidate.project_id:
            reasons.append("cross_project_evidence")
        else:
            facts.append(fact)
    if reasons or len(facts) != len(candidate.supporting_evidence_ids):
        return None, tuple(sorted(set(reasons or ("incomplete_evidence_set",))))
    return tuple(sorted(facts, key=lambda item: item.evidence_fact_id)), ()


def _assessment_is_internally_eligible(assessment: CapabilitySupportAssessment) -> bool:
    return all((
        assessment.meets_evidence_minimum,
        assessment.meets_signal_group_minimum,
        assessment.meets_mechanism_minimum,
        not assessment.has_explicit_conflict,
        assessment.evidence_count == len(assessment.supporting_evidence_ids),
    ))


def _policy_is_internally_eligible(policy: CapabilityClaimPolicy) -> bool:
    return all((
        not policy.has_boundary_conflict,
        not policy.has_metric_conflict,
        not policy.uncovered_evidence_ids,
        policy.boundary_count == len(policy.inherited_boundary_ids),
        len(policy.inherited_boundary_ids) == len(set(policy.inherited_boundary_ids)),
        policy.covered_evidence_count == len(policy.supporting_evidence_ids),
        bool(policy.inherited_boundary_ids),
        all(boundary_id.startswith("pcb_") for boundary_id in policy.inherited_boundary_ids),
    ))


def build_project_capability_fact(
    *,
    candidate: CapabilityCandidate,
    assessment: CapabilitySupportAssessment,
    policy: CapabilityClaimPolicy,
    evidence_index: Mapping[str, ProjectEvidenceFact],
) -> ProjectCapabilityFactBuildResult:
    """Build one authoritative fact only after both lifecycle gates pass."""

    if not isinstance(candidate, CapabilityCandidate):
        raise TypeError("candidate must be a CapabilityCandidate")
    if not isinstance(assessment, CapabilitySupportAssessment):
        raise TypeError("assessment must be a CapabilitySupportAssessment")
    if not isinstance(policy, CapabilityClaimPolicy):
        raise TypeError("policy must be a CapabilityClaimPolicy")

    diagnostics = _base_diagnostics(assessment, policy)
    definition = get_capability_rule(candidate.capability_type)
    if definition is None or definition.capability_type != candidate.capability_type:
        return _result(candidate, policy, "identity_mismatch", "noncanonical_capability_type", diagnostics=diagnostics)
    if candidate.candidate_score != 0.0 or candidate.metadata.get("evaluation_state") != "unscored":
        return _result(candidate, policy, "invalid_fact", "candidate_lifecycle_state_invalid", diagnostics=diagnostics)

    identity_reasons = _identity_reasons(candidate, assessment, policy)
    if identity_reasons:
        return _result(candidate, policy, "identity_mismatch", *identity_reasons, diagnostics=diagnostics)

    support_ids = candidate.supporting_evidence_ids
    if assessment.supporting_evidence_ids != support_ids or policy.supporting_evidence_ids != support_ids:
        return _result(candidate, policy, "evidence_mismatch", "lifecycle_evidence_set_mismatch", diagnostics=diagnostics)

    if assessment.eligibility_status != "eligible":
        return _result(candidate, policy, "ineligible_support", "support_assessment_not_eligible", diagnostics=diagnostics)
    if not _assessment_is_internally_eligible(assessment):
        return _result(candidate, policy, "ineligible_support", "eligible_assessment_has_inconsistent_proof_state", diagnostics=diagnostics)
    if policy.policy_status != "eligible":
        return _result(candidate, policy, "ineligible_claim_policy", "claim_policy_not_eligible", diagnostics=diagnostics)
    if not _policy_is_internally_eligible(policy):
        return _result(candidate, policy, "ineligible_claim_policy", "eligible_policy_has_inconsistent_state", diagnostics=diagnostics)

    facts, evidence_reasons = _resolve_supporting_evidence(candidate, evidence_index)
    if facts is None:
        return _result(candidate, policy, "evidence_mismatch", *evidence_reasons, diagnostics=diagnostics)

    mechanisms = tuple(
        value for value in select_project_evidence_mechanisms(facts)
        if is_concrete_project_evidence_mechanism(value)
    )
    if len(mechanisms) < definition.minimum_mechanism_count:
        diagnostics["mechanism_count"] = len(mechanisms)
        diagnostics["minimum_mechanism_count"] = definition.minimum_mechanism_count
        return _result(candidate, policy, "missing_mechanisms", "mechanism_count_below_taxonomy_minimum", diagnostics=diagnostics)

    try:
        allowed_claims = _canonical_allowed_claims(policy.allowed_claims)
    except (TypeError, ValueError):
        return _result(candidate, policy, "ineligible_claim_policy", "invalid_policy_allowed_claim", diagnostics=diagnostics)
    if not allowed_claims:
        return _result(candidate, policy, "missing_allowed_claims", "eligible_policy_has_no_allowed_claims", diagnostics=diagnostics)
    forbidden_claims = _canonical_strings(policy.forbidden_claims)
    if {item.casefold() for item in allowed_claims} & {item.casefold() for item in forbidden_claims}:
        return _result(candidate, policy, "ineligible_claim_policy", "allowed_forbidden_claim_collision", diagnostics=diagnostics)

    try:
        metric_support = MetricSupport(policy.metric_support)
    except (TypeError, ValueError):
        return _result(candidate, policy, "invalid_metric_support", "unsupported_policy_metric_support", diagnostics=diagnostics)
    numeric_conflicts = []
    for claim in allowed_claims:
        if not is_project_resume_metric_claim(claim):
            continue
        safe, reasons = evaluate_project_numeric_claim(claim, metric_support)
        if not safe:
            numeric_conflicts.extend(reasons or ("numeric_claim_metric_mismatch",))
    if numeric_conflicts:
        return _result(candidate, policy, "invalid_metric_support", *numeric_conflicts, diagnostics=diagnostics)

    technical_tags = tuple(
        tag for tag in select_project_evidence_technical_tags(facts)
        if _is_safe_project_technical_tag(tag)
    )
    confidence = derive_project_capability_confidence(definition, facts)
    if confidence is Confidence.LOW:
        return _result(candidate, policy, "invalid_fact", "verified_fact_confidence_cannot_be_low", diagnostics=diagnostics)

    diagnostics.update({
        "allowed_claim_count": len(allowed_claims),
        "confidence_derivation": "authoritative_extractor_proof_rule",
        "forbidden_claim_count": len(forbidden_claims),
        "mechanism_count": len(mechanisms),
        "technical_tag_count": len(technical_tags),
    })
    try:
        fact = ProjectCapabilityFact(
            project_id=candidate.project_id,
            capability_type=definition.capability_type,
            present=True,
            source_evidence_fact_ids=list(support_ids),
            confidence=confidence,
            mechanisms=list(mechanisms),
            allowed_resume_claims=list(allowed_claims),
            forbidden_claims=list(forbidden_claims),
            metric_support=metric_support,
            technical_tags=list(technical_tags),
        )
        validated = validate_project_capability_fact(fact)
        authoritative_identity = ProjectCapabilityFact(
            project_id=validated.project_id,
            capability_type=validated.capability_type,
            present=validated.present,
            source_evidence_fact_ids=list(validated.source_evidence_fact_ids),
            confidence=validated.confidence,
            mechanisms=list(validated.mechanisms),
            allowed_resume_claims=list(validated.allowed_resume_claims),
            forbidden_claims=list(validated.forbidden_claims),
            metric_support=validated.metric_support,
            technical_tags=list(validated.technical_tags),
        ).capability_id
        if validated.capability_id != authoritative_identity or not validated.capability_id.startswith("pcf_"):
            raise ValueError("capability ID does not match authoritative model identity")
    except (TypeError, ValueError):
        return _result(candidate, policy, "invalid_fact", "authoritative_fact_validation_failed", diagnostics=diagnostics)

    return _result(
        candidate,
        policy,
        "built",
        "verified_fact_built",
        fact=validated,
        diagnostics=diagnostics,
    )


def _semantic_payload(value: Any) -> str:
    return json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe_lifecycle(
    values: Sequence[_T],
    *,
    project_id: str,
    expected_type: type[_T],
    label: str,
) -> dict[str, _T]:
    output: dict[str, tuple[str, _T]] = {}
    for value in values:
        if not isinstance(value, expected_type):
            raise TypeError(f"{label} must contain only {expected_type.__name__} values")
        if value.project_id != project_id:
            raise ValueError(f"{label} project_id does not match requested project")
        definition = get_capability_rule(value.capability_type)
        if definition is None or definition.capability_type != value.capability_type:
            raise ValueError(f"{label} contains a noncanonical capability type")
        payload = _semantic_payload(value)
        previous = output.get(value.capability_type)
        if previous is not None and previous[0] != payload:
            raise ValueError(f"same capability type has conflicting {label}")
        output[value.capability_type] = (payload, value)
    return {key: output[key][1] for key in sorted(output)}


def build_project_capability_facts(
    *,
    project_id: str,
    candidates: Sequence[CapabilityCandidate],
    assessments: Sequence[CapabilitySupportAssessment],
    policies: Sequence[CapabilityClaimPolicy],
    evidence_facts: Sequence[ProjectEvidenceFact],
) -> tuple[ProjectCapabilityFactBuildResult, ...]:
    """Build complete Candidate/Assessment/Policy pairs for exactly one project."""

    if not isinstance(project_id, str) or not " ".join(project_id.split()):
        raise ValueError("project_id must be a non-blank string")
    normalized_project_id = " ".join(project_id.split())
    candidate_by_type = _dedupe_lifecycle(
        candidates, project_id=normalized_project_id, expected_type=CapabilityCandidate, label="candidates"
    )
    assessment_by_type = _dedupe_lifecycle(
        assessments, project_id=normalized_project_id, expected_type=CapabilitySupportAssessment, label="assessments"
    )
    policy_by_type = _dedupe_lifecycle(
        policies, project_id=normalized_project_id, expected_type=CapabilityClaimPolicy, label="policies"
    )
    capability_types = set(candidate_by_type)
    if capability_types != set(assessment_by_type) or capability_types != set(policy_by_type):
        raise ValueError("batch requires complete candidate, assessment, and policy pairs")

    evidence_index: dict[str, ProjectEvidenceFact] = {}
    evidence_payloads: dict[str, str] = {}
    for fact in evidence_facts:
        if not isinstance(fact, ProjectEvidenceFact):
            raise TypeError("evidence_facts must contain only ProjectEvidenceFact values")
        if fact.project_id != normalized_project_id:
            raise ValueError("Evidence Fact project_id does not match requested project")
        payload = fact.to_json()
        previous = evidence_payloads.get(fact.evidence_fact_id)
        if previous is not None and previous != payload:
            raise ValueError("same Evidence Fact ID has conflicting semantic content")
        evidence_payloads[fact.evidence_fact_id] = payload
        evidence_index[fact.evidence_fact_id] = fact

    results = tuple(
        build_project_capability_fact(
            candidate=candidate_by_type[capability_type],
            assessment=assessment_by_type[capability_type],
            policy=policy_by_type[capability_type],
            evidence_index=evidence_index,
        )
        for capability_type in sorted(capability_types)
    )
    built_ids = [item.fact.capability_id for item in results if item.fact is not None]
    if len(built_ids) != len(set(built_ids)):
        raise ValueError("batch produced duplicate capability IDs")
    return results


__all__ = [
    "PROJECT_CAPABILITY_FACT_BUILD_STATUSES",
    "ProjectCapabilityFactBuildResult",
    "build_project_capability_fact",
    "build_project_capability_facts",
]
