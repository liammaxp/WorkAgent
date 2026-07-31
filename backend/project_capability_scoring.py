"""Deterministically assess unscored project capability candidates.

Assessments reuse the canonical taxonomy and the evidence extractor's proof
classification.  They do not mutate candidates, inspect claim boundaries,
construct capability facts, persist data, or call external services.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend.project_capability_extractor import (
    classify_project_capability_evidence_kind,
    extract_project_evidence_fact_signals_many,
    get_independent_project_evidence_identity,
    get_project_evidence_quality_score,
    select_project_evidence_mechanisms,
)
from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_taxonomy import (
    SIGNAL_REGISTRY,
    ProjectCapabilityDefinition,
    get_capability_rule,
)
from backend.project_evidence_models import ProjectEvidenceFact


CAPABILITY_SUPPORT_ELIGIBILITY_STATUSES = frozenset({
    "eligible",
    "explicitly_conflicted",
    "insufficient_evidence",
    "insufficient_mechanism_support",
    "insufficient_signal_coverage",
})

EVIDENCE_QUALITY_WEIGHT = 0.30
SIGNAL_COVERAGE_WEIGHT = 0.30
MECHANISM_SPECIFICITY_WEIGHT = 0.30
SOURCE_DIVERSITY_WEIGHT = 0.10
CONFLICT_PENALTY = 0.25
SOURCE_DIVERSITY_TARGET = 3

_GENERIC_MECHANISM_TOKENS = frozenset({
    "a", "an", "and", "build", "built", "code", "control", "feature",
    "fastapi", "implemented", "implementation", "improve", "improved",
    "mechanism", "mongodb", "openai", "optimize", "optimized", "python",
    "react", "sqlite", "system", "the", "using",
})
_WORD_RE = re.compile(r"[a-z0-9]+")


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


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


@dataclass(frozen=True)
class CapabilitySupportAssessment:
    project_id: str
    capability_type: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_signal_ids: tuple[str, ...]
    satisfied_signal_groups: tuple[str, ...]
    missing_signal_groups: tuple[str, ...]
    evidence_count: int
    distinct_signal_group_count: int
    mechanism_count: int
    source_type_count: int
    evidence_quality_score: float
    signal_coverage_score: float
    mechanism_specificity_score: float
    source_diversity_score: float
    conflict_penalty: float
    support_score: float
    specificity_score: float
    meets_evidence_minimum: bool
    meets_signal_group_minimum: bool
    meets_mechanism_minimum: bool
    has_explicit_conflict: bool
    eligibility_status: str
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "supporting_evidence_ids", "supporting_signal_ids", "satisfied_signal_groups",
            "missing_signal_groups", "reasons",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.eligibility_status not in CAPABILITY_SUPPORT_ELIGIBILITY_STATUSES:
            raise ValueError("unsupported capability support eligibility status")
        for name in (
            "evidence_quality_score", "signal_coverage_score", "mechanism_specificity_score",
            "source_diversity_score", "conflict_penalty", "support_score", "specificity_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "capability_type": self.capability_type,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "supporting_signal_ids": list(self.supporting_signal_ids),
            "satisfied_signal_groups": list(self.satisfied_signal_groups),
            "missing_signal_groups": list(self.missing_signal_groups),
            "evidence_count": self.evidence_count,
            "distinct_signal_group_count": self.distinct_signal_group_count,
            "mechanism_count": self.mechanism_count,
            "source_type_count": self.source_type_count,
            "evidence_quality_score": self.evidence_quality_score,
            "signal_coverage_score": self.signal_coverage_score,
            "mechanism_specificity_score": self.mechanism_specificity_score,
            "source_diversity_score": self.source_diversity_score,
            "conflict_penalty": self.conflict_penalty,
            "support_score": self.support_score,
            "specificity_score": self.specificity_score,
            "meets_evidence_minimum": self.meets_evidence_minimum,
            "meets_signal_group_minimum": self.meets_signal_group_minimum,
            "meets_mechanism_minimum": self.meets_mechanism_minimum,
            "has_explicit_conflict": self.has_explicit_conflict,
            "eligibility_status": self.eligibility_status,
            "reasons": list(self.reasons),
            "diagnostics": _thaw(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _group_id(index: int) -> str:
    return f"required_group_{index + 1}"


def _is_concrete_mechanism(value: str) -> bool:
    tokens = _WORD_RE.findall(value.casefold())
    specific = [token for token in tokens if token not in _GENERIC_MECHANISM_TOKENS]
    return bool(specific) and (len(tokens) >= 2 or "_" in value or "-" in value)


def is_concrete_project_evidence_mechanism(value: str) -> bool:
    """Expose the scoring layer's existing deterministic concreteness rule."""

    if not isinstance(value, str):
        raise TypeError("mechanism must be a string")
    return _is_concrete_mechanism(value)


def _resolved_evidence(
    candidate: CapabilityCandidate,
    evidence_index: Mapping[str, ProjectEvidenceFact],
) -> tuple[ProjectEvidenceFact, ...]:
    if not isinstance(candidate, CapabilityCandidate):
        raise TypeError("candidate must be a CapabilityCandidate")
    if not isinstance(evidence_index, Mapping):
        raise TypeError("evidence_index must be a mapping")
    resolved: list[ProjectEvidenceFact] = []
    for evidence_id in candidate.supporting_evidence_ids:
        fact = evidence_index.get(evidence_id)
        if fact is None:
            raise ValueError(f"missing evidence reference: {evidence_id}")
        if not isinstance(fact, ProjectEvidenceFact):
            raise TypeError("evidence_index values must be ProjectEvidenceFact values")
        if fact.evidence_fact_id != evidence_id:
            raise ValueError("evidence index key does not match Evidence Fact ID")
        if fact.project_id != candidate.project_id:
            raise ValueError("cross-project evidence reference")
        resolved.append(fact)
    return tuple(sorted(resolved, key=lambda item: item.evidence_fact_id))


def _proof_facts(
    facts: Sequence[ProjectEvidenceFact],
    definition: ProjectCapabilityDefinition,
) -> tuple[tuple[ProjectEvidenceFact, str], ...]:
    eligible = []
    for fact in facts:
        kind = classify_project_capability_evidence_kind(fact, definition)
        if kind is not None and get_project_evidence_quality_score(fact) >= definition.minimum_quality_score:
            eligible.append((fact, kind))
    return tuple(eligible)


def assess_capability_candidate_support(
    *,
    candidate: CapabilityCandidate,
    evidence_index: Mapping[str, ProjectEvidenceFact],
) -> CapabilitySupportAssessment:
    """Assess one candidate without mutating it or producing a capability fact."""

    definition = get_capability_rule(candidate.capability_type)
    if definition is None:
        raise ValueError("candidate capability_type is not present in the canonical taxonomy")
    facts = _resolved_evidence(candidate, evidence_index)
    proof_records = _proof_facts(facts, definition)
    proof_facts = tuple(item[0] for item in proof_records)
    proof_fact_ids = {fact.evidence_fact_id for fact in proof_facts}

    extractions, _report = extract_project_evidence_fact_signals_many(facts)
    extracted_by_fact = {item.evidence_fact_id: set(item.signals) for item in extractions}
    known_signals = set(candidate.supporting_signals) & set(SIGNAL_REGISTRY)
    unknown_signals = set(candidate.supporting_signals) - set(SIGNAL_REGISTRY)
    proof_signals = {
        signal
        for evidence_id in proof_fact_ids
        for signal in extracted_by_fact[evidence_id]
        if signal in known_signals
    }

    satisfied_groups: list[str] = []
    missing_groups: list[str] = []
    group_definitions: dict[str, tuple[str, ...]] = {}
    for index, group in enumerate(definition.required_signal_groups):
        identifier = _group_id(index)
        group_definitions[identifier] = tuple(group)
        (satisfied_groups if proof_signals & set(group) else missing_groups).append(identifier)

    direct_identities = {
        get_independent_project_evidence_identity(fact)
        for fact, kind in proof_records
        if kind == "direct"
    }
    qualified_count = len(proof_facts)
    meets_evidence = (
        qualified_count >= definition.minimum_total_fact_count
        and len(direct_identities) >= definition.minimum_direct_fact_count
        and (not definition.requires_direct_provenance or bool(direct_identities))
    )
    meets_groups = len(satisfied_groups) >= definition.minimum_distinct_signal_group_count

    normalized_mechanisms = select_project_evidence_mechanisms(proof_facts)
    concrete_mechanisms = tuple(value for value in normalized_mechanisms if _is_concrete_mechanism(value))
    mechanism_count = len(concrete_mechanisms)
    meets_mechanisms = mechanism_count >= definition.minimum_mechanism_count

    evidence_quality = _bounded_score(
        sum(get_project_evidence_quality_score(fact) for fact in facts) / (100 * len(facts))
        if facts else 0.0
    )
    signal_coverage = _bounded_score(
        len(satisfied_groups) / len(definition.required_signal_groups)
        if definition.required_signal_groups else 0.0
    )
    mechanism_fact_count = sum(_is_concrete_mechanism(fact.mechanism) for fact in proof_facts)
    mechanism_minimum_ratio = min(1.0, mechanism_count / definition.minimum_mechanism_count)
    mechanism_fact_ratio = mechanism_fact_count / qualified_count if qualified_count else 0.0
    mechanism_specificity = _bounded_score(0.5 * mechanism_minimum_ratio + 0.5 * mechanism_fact_ratio)
    source_types = {ref.source_type for fact in facts for ref in fact.source_refs}
    source_diversity = _bounded_score(len(source_types) / SOURCE_DIVERSITY_TARGET)
    has_conflict = bool(candidate.conflicting_signals)
    conflict_penalty = CONFLICT_PENALTY if has_conflict else 0.0
    support_score = _bounded_score(
        evidence_quality * EVIDENCE_QUALITY_WEIGHT
        + signal_coverage * SIGNAL_COVERAGE_WEIGHT
        + mechanism_specificity * MECHANISM_SPECIFICITY_WEIGHT
        + source_diversity * SOURCE_DIVERSITY_WEIGHT
        - conflict_penalty
    )

    implementation_ratio = sum(bool(fact.implementation) for fact in facts) / len(facts) if facts else 0.0
    traceability_ratio = (
        len({signal for values in extracted_by_fact.values() for signal in values} & known_signals)
        / len(candidate.supporting_signals)
        if candidate.supporting_signals else 0.0
    )
    specificity_score = _bounded_score(
        0.5 * mechanism_specificity + 0.25 * implementation_ratio + 0.25 * traceability_ratio
    )

    reasons: list[str] = []
    if meets_evidence:
        reasons.append("meets_evidence_minimum")
    else:
        if qualified_count < definition.minimum_total_fact_count:
            reasons.append("evidence_count_below_minimum")
        if len(direct_identities) < definition.minimum_direct_fact_count:
            reasons.append("direct_evidence_count_below_minimum")
        if definition.requires_direct_provenance and not direct_identities:
            reasons.append("direct_provenance_required")
        if any(get_project_evidence_quality_score(fact) < definition.minimum_quality_score for fact in facts):
            reasons.append("evidence_quality_below_minimum")
    if meets_groups:
        reasons.append("meets_signal_group_minimum")
    else:
        reasons.extend(f"missing_required_signal_group:{item}" for item in missing_groups)
    if meets_mechanisms:
        reasons.append("meets_mechanism_minimum")
    else:
        reasons.append("mechanism_count_below_minimum")
    reasons.extend(f"explicit_conflict:{item}" for item in candidate.conflicting_signals)
    reasons.extend(f"unknown_supporting_signal:{item}" for item in sorted(unknown_signals))

    if has_conflict:
        status = "explicitly_conflicted"
    elif not meets_evidence:
        status = "insufficient_evidence"
    elif not meets_groups:
        status = "insufficient_signal_coverage"
    elif not meets_mechanisms:
        status = "insufficient_mechanism_support"
    else:
        status = "eligible"

    diagnostics = {
        "direct_evidence_count": len(direct_identities),
        "eligible_evidence_count": qualified_count,
        "minimum_direct_evidence_count": definition.minimum_direct_fact_count,
        "minimum_evidence_count": definition.minimum_total_fact_count,
        "minimum_evidence_quality_score": definition.minimum_quality_score,
        "minimum_mechanism_count": definition.minimum_mechanism_count,
        "minimum_signal_group_count": definition.minimum_distinct_signal_group_count,
        "required_signal_groups": group_definitions,
        "score_weights": {
            "evidence_quality": EVIDENCE_QUALITY_WEIGHT,
            "mechanism_specificity": MECHANISM_SPECIFICITY_WEIGHT,
            "signal_coverage": SIGNAL_COVERAGE_WEIGHT,
            "source_diversity": SOURCE_DIVERSITY_WEIGHT,
        },
        "unknown_supporting_signal_count": len(unknown_signals),
    }
    return CapabilitySupportAssessment(
        project_id=candidate.project_id,
        capability_type=definition.capability_type,
        supporting_evidence_ids=candidate.supporting_evidence_ids,
        supporting_signal_ids=tuple(sorted(known_signals)),
        satisfied_signal_groups=tuple(satisfied_groups),
        missing_signal_groups=tuple(missing_groups),
        evidence_count=len(facts),
        distinct_signal_group_count=len(satisfied_groups),
        mechanism_count=mechanism_count,
        source_type_count=len(source_types),
        evidence_quality_score=evidence_quality,
        signal_coverage_score=signal_coverage,
        mechanism_specificity_score=mechanism_specificity,
        source_diversity_score=source_diversity,
        conflict_penalty=conflict_penalty,
        support_score=support_score,
        specificity_score=specificity_score,
        meets_evidence_minimum=meets_evidence,
        meets_signal_group_minimum=meets_groups,
        meets_mechanism_minimum=meets_mechanisms,
        has_explicit_conflict=has_conflict,
        eligibility_status=status,
        reasons=tuple(sorted(set(reasons))),
        diagnostics=diagnostics,
    )


def assess_project_capability_candidates(
    *,
    project_id: str,
    candidates: Sequence[CapabilityCandidate],
    evidence_facts: Sequence[ProjectEvidenceFact],
) -> tuple[CapabilitySupportAssessment, ...]:
    """Assess a deterministic, project-isolated set of candidates in memory."""

    if not isinstance(project_id, str) or not " ".join(project_id.split()):
        raise ValueError("project_id must be a non-blank string")
    normalized_project_id = " ".join(project_id.split())
    if any(not isinstance(item, CapabilityCandidate) for item in candidates):
        raise TypeError("candidates must contain only CapabilityCandidate values")
    if any(item.project_id != normalized_project_id for item in candidates):
        raise ValueError("candidate project_id does not match requested project")
    if any(not isinstance(item, ProjectEvidenceFact) for item in evidence_facts):
        raise TypeError("evidence_facts must contain only ProjectEvidenceFact values")
    if any(item.project_id != normalized_project_id for item in evidence_facts):
        raise ValueError("Evidence Fact project_id does not match requested project")

    evidence_index: dict[str, ProjectEvidenceFact] = {}
    evidence_payloads: dict[str, str] = {}
    for fact in evidence_facts:
        payload = fact.to_json()
        previous = evidence_payloads.get(fact.evidence_fact_id)
        if previous is not None and previous != payload:
            raise ValueError("same Evidence Fact ID has conflicting semantic content")
        evidence_payloads[fact.evidence_fact_id] = payload
        evidence_index[fact.evidence_fact_id] = fact

    by_type: dict[str, tuple[str, CapabilityCandidate]] = {}
    for candidate in candidates:
        payload = json.dumps(
            candidate.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        previous = by_type.get(candidate.capability_type)
        if previous is not None and previous[0] != payload:
            raise ValueError("same capability type has conflicting candidates")
        by_type[candidate.capability_type] = (payload, candidate)
    return tuple(
        assess_capability_candidate_support(candidate=by_type[key][1], evidence_index=evidence_index)
        for key in sorted(by_type)
    )


__all__ = [
    "CAPABILITY_SUPPORT_ELIGIBILITY_STATUSES",
    "CONFLICT_PENALTY",
    "EVIDENCE_QUALITY_WEIGHT",
    "MECHANISM_SPECIFICITY_WEIGHT",
    "SIGNAL_COVERAGE_WEIGHT",
    "SOURCE_DIVERSITY_TARGET",
    "SOURCE_DIVERSITY_WEIGHT",
    "CapabilitySupportAssessment",
    "assess_capability_candidate_support",
    "assess_project_capability_candidates",
    "is_concrete_project_evidence_mechanism",
]
