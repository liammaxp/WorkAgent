"""Deterministic, explainable quality scoring for Phase 4 Evidence Facts.

The scorer evaluates one already-synthesized fact at a time. It performs no
cross-fact promotion, persistence, retrieval, JD analysis, network access, or
model calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from statistics import median
from typing import Any, Iterable, Mapping

from backend.phase4_models import (
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    Phase4EvidenceFact,
    Phase4SourceRef,
)


MAX_DECISION_SAMPLES = 100
DIMENSION_MAXIMUMS = {
    "problem_specificity": 10,
    "mechanism_specificity": 25,
    "implementation_specificity": 20,
    "provenance_strength": 15,
    "safe_impact_quality": 10,
    "technical_specificity": 10,
    "claim_safety": 10,
}
if sum(DIMENSION_MAXIMUMS.values()) != 100:  # pragma: no cover - import-time invariant
    raise RuntimeError("Phase 4 evidence scoring dimensions must total 100")

GENERIC_PHRASES = frozenset({
    "added intelligent features",
    "built ai-powered system",
    "built an ai-powered system",
    "developed backend functionality",
    "developed features",
    "enhanced functionality",
    "enhanced retrieval",
    "improved pipeline",
    "improved reliability",
    "improved system",
    "improved the pipeline",
    "optimized performance",
    "used advanced technologies",
    "used ai",
    "worked on backend",
    "worked on resume generation",
})
OPERATION_MARKERS = frozenset({
    "atomic", "bind", "block", "cache", "chunk", "constraint", "deduplicat",
    "deterministic", "fallback", "filter", "gate", "map", "merge", "normaliz",
    "parse", "persist", "rank", "recover", "retriev", "route", "schema",
    "select", "sort", "split", "state", "store", "transaction", "validat",
    "workflow", "write",
})
CONDITION_MARKERS = frozenset({
    "before", "condition", "conflict", "duplicate", "failure", "invalid",
    "limitation", "missing", "mismatch", "requirement", "risk", "unsupported",
    "when", "without",
})
GENERIC_IMPACT_PHRASES = frozenset({
    "improved performance",
    "improved reliability",
    "improved system quality",
    "optimized performance",
})
DIRECT_SOURCE_TYPES = frozenset({
    "phase2_evidence_chunk",
    "phase2_evidence_card",
    "phase2_raw_change_summary",
    "phase3_evidence_card",
    "phase3_raw_change_summary",
})
CONTEXTUAL_SOURCE_TYPES = frozenset({
    "phase2_capability_fact",
    "phase3_capability_fact",
    "project_memory",
    "project_compact_facts",
    "compact_facts",
})

_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+|::|[A-Za-z_][A-Za-z0-9_]*\(|\b[A-Z]{2,}[A-Za-z0-9]*\b|\b[a-z]+_[a-z0-9_]+\b)"
)
_ABSOLUTE_CLAIM_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\beliminated\b",
    r"\bguaranteed\b",
    r"\bfully\s+prevented\b",
    r"\bzero\s+hallucinations?\b",
    r"\breduced\s+hallucinations?\b",
    r"\bproduction[- ]scale\b",
    r"\benterprise[- ]grade\b",
    r"\bats\s+success\b",
))
_NUMERIC_CLAIM_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b\d+(?:\.\d+)?\s*(?:%|\bpercent\b)",
    r"\b\d+(?:\.\d+)?\s*(?:x|times)\s+faster\b",
    r"\b(?:reduced|increased)\s+by\s+\d+(?:\.\d+)?(?:\s*(?:%|\bpercent\b))?",
    r"\bsaved\s+\d+(?:\.\d+)?\s+hours?\b",
    r"\b100\s*%",
))
_APPROXIMATE_MARKER_RE = re.compile(
    r"(?:\bapproximately\b|\babout\b|\baround\b|\bbetween\b|\brange\b|~)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Phase4EvidenceQualityEvaluation:
    score: int
    breakdown: dict[str, Any]
    quality_band: str
    recommended_status: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    unsupported_claim_penalty: bool
    generic_content_penalty: bool


@dataclass(frozen=True)
class Phase4EvidenceScoringDecision:
    evidence_fact_id: str
    project_id: str
    original_status: str
    recommended_status: str
    quality_band: str
    score: int
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class Phase4EvidenceScoringGroupCount:
    project_id: str
    structural_status: str
    quality_band: str
    evidence_type: str
    source_category: str
    count: int


@dataclass(frozen=True)
class Phase4EvidenceScoringReport:
    input_count: int
    output_count: int
    high_value_count: int
    supporting_value_count: int
    weak_value_count: int
    rejected_value_count: int
    status_changed_count: int
    unsupported_claim_penalty_count: int
    generic_content_penalty_count: int
    missing_provenance_count: int
    missing_mechanism_count: int
    missing_implementation_count: int
    minimum_score: int
    maximum_score: int
    median_score: float
    average_score: float
    score_buckets: dict[str, int]
    decisions: tuple[Phase4EvidenceScoringDecision, ...] = ()
    grouped_counts: tuple[Phase4EvidenceScoringGroupCount, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _contains_marker(value: str, markers: frozenset[str]) -> bool:
    lowered = value.casefold()
    return any(marker in lowered for marker in markers)


def _is_generic(value: str) -> bool:
    normalized = _normalize(value).casefold().strip(" .!?:;")
    return normalized in GENERIC_PHRASES


def _has_identifier(values: Iterable[str]) -> bool:
    return any(_TECHNICAL_IDENTIFIER_RE.search(value) for value in values)


def _problem_score(problem: str) -> int:
    value = _normalize(problem)
    if not value:
        return 0
    if _is_generic(value):
        return 2
    has_condition = _contains_marker(value, CONDITION_MARKERS)
    has_identifier = _has_identifier((value,))
    if has_condition and has_identifier:
        return 10
    if has_condition:
        return 8
    if has_identifier:
        return 7
    return 5 if len(value.split()) >= 5 else 3


def _mechanism_parts(mechanism: str) -> list[str]:
    return [part.strip() for part in mechanism.split(";") if part.strip()]


def _mechanism_score(mechanism: str) -> tuple[int, bool, bool]:
    parts = _mechanism_parts(_normalize(mechanism))
    if not parts:
        return 0, False, False
    generic_flags = [_is_generic(part) for part in parts]
    concrete = [part for part, generic in zip(parts, generic_flags) if not generic]
    if not concrete:
        return 3, True, False
    operation_count = sum(_contains_marker(part, OPERATION_MARKERS) for part in concrete)
    identifier = _has_identifier(concrete)
    if len(concrete) >= 2 and operation_count >= 2 and identifier:
        score = 25
    elif len(concrete) >= 2 and operation_count >= 2:
        score = 22
    elif len(concrete) >= 2:
        score = 18
    elif operation_count and identifier:
        score = 20
    elif operation_count:
        score = 16
    elif identifier:
        score = 14
    else:
        score = 12
    return score, any(generic_flags), True


def _implementation_score(values: list[str]) -> tuple[int, bool, bool]:
    normalized = [_normalize(value) for value in values if _normalize(value)]
    if not normalized:
        return 0, False, False
    generic_flags = [_is_generic(value) for value in normalized]
    concrete = [value for value, generic in zip(normalized, generic_flags) if not generic]
    if not concrete:
        return 2, True, False
    identifiers = sum(bool(_TECHNICAL_IDENTIFIER_RE.search(value)) for value in concrete)
    if len(concrete) >= 5 and identifiers >= 2:
        score = 20
    elif len(concrete) >= 3:
        score = 17
    elif len(concrete) >= 2:
        score = 14
    elif identifiers:
        score = 12
    else:
        score = 8
    return score, any(generic_flags), True


def _ref_identity(ref: Phase4SourceRef) -> tuple[str, str, str, str, str, str, int | None, int | None]:
    return (
        ref.project_id,
        ref.source_type,
        ref.source_id,
        ref.content_hash,
        ref.commit_sha or "",
        ref.file_path or "",
        ref.start_line,
        ref.end_line,
    )


def _provenance_score(fact: Phase4EvidenceFact) -> tuple[int, bool, bool]:
    if not fact.source_refs:
        return 0, True, False
    if any(ref.project_id != fact.project_id for ref in fact.source_refs):
        return 0, True, False
    unique = {_ref_identity(ref): ref for ref in fact.source_refs}
    source_types = {ref.source_type for ref in unique.values()}
    direct_count = sum(ref.source_type in DIRECT_SOURCE_TYPES for ref in unique.values())
    if direct_count >= 2:
        return 15, False, True
    if direct_count == 1:
        return 11, False, True
    if source_types & CONTEXTUAL_SOURCE_TYPES:
        if source_types & {"project_memory", "project_compact_facts", "compact_facts"}:
            return 5, False, False
        return 6, False, False
    return 4, False, False


def _claim_values(fact: Phase4EvidenceFact) -> list[str]:
    return [
        fact.problem,
        fact.mechanism,
        *fact.implementation,
        *fact.safe_impact,
        *fact.allowed_claims,
        *fact.forbidden_claims,
    ]


def _unsupported_claim(value: str, metric_support: MetricSupport) -> bool:
    if any(pattern.search(value) for pattern in _ABSOLUTE_CLAIM_PATTERNS):
        return True
    has_numeric_claim = any(pattern.search(value) for pattern in _NUMERIC_CLAIM_PATTERNS)
    if not has_numeric_claim:
        return False
    if metric_support is MetricSupport.EXPLICIT:
        return False
    if metric_support is MetricSupport.APPROXIMATE:
        return not bool(_APPROXIMATE_MARKER_RE.search(value))
    return True


def _unsafe_claim_count(fact: Phase4EvidenceFact) -> int:
    return sum(_unsupported_claim(value, fact.metric_support) for value in _claim_values(fact))


def _impact_score(fact: Phase4EvidenceFact, unsupported_count: int) -> tuple[int, bool]:
    values = [_normalize(value) for value in fact.safe_impact if _normalize(value)]
    if not values or unsupported_count:
        return 0, any(value.casefold().strip(" .!?:;") in GENERIC_IMPACT_PHRASES for value in values)
    generic = [value for value in values if value.casefold().strip(" .!?:;") in GENERIC_IMPACT_PHRASES]
    specific = [value for value in values if value not in generic]
    if not specific:
        return 2, bool(generic)
    technical = _has_identifier(specific) or any(
        _contains_marker(value, OPERATION_MARKERS | CONDITION_MARKERS) for value in specific
    )
    if len(specific) >= 2 and technical:
        return 10, bool(generic)
    if technical:
        return 8, bool(generic)
    return (7 if len(specific) >= 2 else 6), bool(generic)


def _technical_score(fact: Phase4EvidenceFact) -> int:
    tag_count = len({tag.casefold() for tag in fact.technical_tags})
    tag_score = 0 if tag_count == 0 else 3 if tag_count == 1 else 5
    type_score = 0 if fact.evidence_type is EvidenceType.UNKNOWN else 2
    identifier_score = 3 if _has_identifier((fact.mechanism, *fact.implementation)) else 0
    return min(10, tag_score + type_score + identifier_score)


def _source_category(fact: Phase4EvidenceFact) -> str:
    types = {ref.source_type for ref in fact.source_refs}
    if types & DIRECT_SOURCE_TYPES:
        return "direct_evidence"
    if types & {"phase2_capability_fact", "phase3_capability_fact"}:
        return "capability_context"
    if types & {"project_memory", "project_compact_facts", "compact_facts"}:
        return "project_context"
    return "other"


def _quality_band(score: int) -> str:
    if score >= 80:
        return "high_value"
    if score >= 60:
        return "supporting_value"
    if score >= 40:
        return "weak_value"
    return "rejected_value"


def evaluate_phase4_evidence_quality(
    fact: Phase4EvidenceFact,
) -> Phase4EvidenceQualityEvaluation:
    if not isinstance(fact, Phase4EvidenceFact):
        raise TypeError("evaluate_phase4_evidence_quality expects Phase4EvidenceFact")
    problem = _problem_score(fact.problem)
    mechanism, generic_mechanism, concrete_mechanism = _mechanism_score(fact.mechanism)
    implementation, generic_implementation, concrete_implementation = _implementation_score(fact.implementation)
    provenance, provenance_blocked, direct_provenance = _provenance_score(fact)
    unsupported_count = _unsafe_claim_count(fact)
    impact, generic_impact = _impact_score(fact, unsupported_count)
    technical = _technical_score(fact)
    claim_safety = 3 if unsupported_count else 7 if (generic_mechanism or generic_impact) else 10

    blockers: list[str] = []
    reasons: list[str] = []
    if not fact.mechanism.strip():
        blockers.append("missing_mechanism")
    elif not concrete_mechanism and not concrete_implementation:
        blockers.append("marketing_only_content")
    if provenance_blocked:
        if not fact.source_refs:
            blockers.append("missing_provenance")
        else:
            blockers.append("project_provenance_mismatch")
    if unsupported_count:
        blockers.append("unsupported_central_claim")
    if not fact.implementation:
        reasons.append("missing_implementation")
    if not fact.problem:
        reasons.append("missing_problem_allowed")
    if not fact.safe_impact:
        reasons.append("missing_impact_allowed")
    if direct_provenance:
        reasons.append("direct_provenance")

    generic_present = generic_mechanism or generic_implementation or generic_impact
    marketing_only = not concrete_mechanism and not concrete_implementation
    generic_penalty = 15 if marketing_only else 3 if generic_present else 0
    unsupported_penalty = 20 if unsupported_count else 0
    missing_implementation_penalty = 15 if not concrete_implementation else 0
    penalties = {
        "generic_content": -generic_penalty,
        "unsupported_claim": -unsupported_penalty,
        "missing_implementation": -missing_implementation_penalty,
    }
    dimensions = {
        "problem_specificity": problem,
        "mechanism_specificity": mechanism,
        "implementation_specificity": implementation,
        "provenance_strength": provenance,
        "safe_impact_quality": impact,
        "technical_specificity": technical,
        "claim_safety": claim_safety,
    }
    base_score = sum(dimensions.values())
    pre_blocker_score = max(0, min(100, base_score + sum(penalties.values())))
    blocker_adjustment = -(pre_blocker_score - 39) if blockers and pre_blocker_score > 39 else 0
    score = max(0, min(100, pre_blocker_score + blocker_adjustment))
    if not isinstance(score, int) or not 0 <= score <= 100:  # pragma: no cover - invariant
        raise ValueError("Phase 4 evidence quality score must be an integer from 0 to 100")
    quality_band = _quality_band(score)
    recommended_status = (
        EvidenceStatus.REJECTED.value if blockers else fact.status.value
    )
    breakdown: dict[str, Any] = {
        **dimensions,
        "penalties": penalties,
        "blocker_adjustment": blocker_adjustment,
        "base_score": base_score,
        "recommended_quality_band": quality_band,
        "recommended_status": recommended_status,
        "reason_codes": sorted(set(reasons)),
        "blocker_codes": sorted(set(blockers)),
    }
    return Phase4EvidenceQualityEvaluation(
        score=score,
        breakdown=breakdown,
        quality_band=quality_band,
        recommended_status=recommended_status,
        reasons=tuple(sorted(set(reasons))),
        blockers=tuple(sorted(set(blockers))),
        unsupported_claim_penalty=bool(unsupported_penalty),
        generic_content_penalty=bool(generic_penalty),
    )


def score_phase4_evidence_fact(fact: Phase4EvidenceFact) -> Phase4EvidenceFact:
    evaluation = evaluate_phase4_evidence_quality(fact)
    payload = fact.to_dict()
    payload["quality_score"] = evaluation.score
    payload["quality_breakdown"] = evaluation.breakdown
    if evaluation.blockers:
        payload["status"] = EvidenceStatus.REJECTED.value
    return Phase4EvidenceFact.from_dict(payload)


def _sort_key(fact: Phase4EvidenceFact) -> tuple[str, str, int, str, str]:
    score = int(fact.quality_score) if fact.quality_score is not None else -1
    return (
        fact.project_id,
        fact.status.value,
        -score,
        fact.evidence_type.value,
        fact.evidence_fact_id,
    )


def score_phase4_evidence_facts(
    facts: Iterable[Phase4EvidenceFact],
) -> tuple[list[Phase4EvidenceFact], Phase4EvidenceScoringReport]:
    originals = list(facts)
    scored: list[Phase4EvidenceFact] = []
    evaluations: list[Phase4EvidenceQualityEvaluation] = []
    decisions: list[Phase4EvidenceScoringDecision] = []
    groups: dict[tuple[str, str, str, str, str], int] = {}
    status_changed = 0
    for fact in originals:
        evaluation = evaluate_phase4_evidence_quality(fact)
        current = score_phase4_evidence_fact(fact)
        evaluations.append(evaluation)
        scored.append(current)
        status_changed += int(fact.status is not current.status)
        group_key = (
            current.project_id,
            fact.status.value,
            evaluation.quality_band,
            current.evidence_type.value,
            _source_category(current),
        )
        groups[group_key] = groups.get(group_key, 0) + 1
        decisions.append(Phase4EvidenceScoringDecision(
            evidence_fact_id=current.evidence_fact_id,
            project_id=current.project_id,
            original_status=fact.status.value,
            recommended_status=evaluation.recommended_status,
            quality_band=evaluation.quality_band,
            score=evaluation.score,
            reasons=evaluation.reasons,
            blockers=evaluation.blockers,
        ))
    scored.sort(key=_sort_key)
    decisions.sort(key=lambda item: (
        item.project_id,
        item.recommended_status,
        -item.score,
        item.evidence_fact_id,
    ))
    decisions = decisions[:MAX_DECISION_SAMPLES]
    band_counts = {
        "high_value": 0,
        "supporting_value": 0,
        "weak_value": 0,
        "rejected_value": 0,
    }
    for evaluation in evaluations:
        band_counts[evaluation.quality_band] += 1
    scores = [evaluation.score for evaluation in evaluations]
    buckets = {
        "0-39": sum(score < 40 for score in scores),
        "40-59": sum(40 <= score < 60 for score in scores),
        "60-79": sum(60 <= score < 80 for score in scores),
        "80-100": sum(score >= 80 for score in scores),
    }
    group_counts = tuple(
        Phase4EvidenceScoringGroupCount(
            project_id=key[0],
            structural_status=key[1],
            quality_band=key[2],
            evidence_type=key[3],
            source_category=key[4],
            count=count,
        )
        for key, count in sorted(groups.items())
    )
    report = Phase4EvidenceScoringReport(
        input_count=len(originals),
        output_count=len(scored),
        high_value_count=band_counts["high_value"],
        supporting_value_count=band_counts["supporting_value"],
        weak_value_count=band_counts["weak_value"],
        rejected_value_count=band_counts["rejected_value"],
        status_changed_count=status_changed,
        unsupported_claim_penalty_count=sum(evaluation.unsupported_claim_penalty for evaluation in evaluations),
        generic_content_penalty_count=sum(evaluation.generic_content_penalty for evaluation in evaluations),
        missing_provenance_count=sum("missing_provenance" in evaluation.blockers for evaluation in evaluations),
        missing_mechanism_count=sum("missing_mechanism" in evaluation.blockers for evaluation in evaluations),
        missing_implementation_count=sum("missing_implementation" in evaluation.reasons for evaluation in evaluations),
        minimum_score=min(scores) if scores else 0,
        maximum_score=max(scores) if scores else 0,
        median_score=float(median(scores)) if scores else 0.0,
        average_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        score_buckets=buckets,
        decisions=tuple(decisions),
        grouped_counts=group_counts,
    )
    return scored, report


__all__ = [
    "DIMENSION_MAXIMUMS",
    "Phase4EvidenceQualityEvaluation",
    "Phase4EvidenceScoringDecision",
    "Phase4EvidenceScoringGroupCount",
    "Phase4EvidenceScoringReport",
    "evaluate_phase4_evidence_quality",
    "score_phase4_evidence_fact",
    "score_phase4_evidence_facts",
]
