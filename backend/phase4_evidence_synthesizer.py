"""Deterministic, conservative Phase 4 Evidence Fact synthesis.

Only explicit fields from normalized Phase4EvidenceInput records are selected
and restructured. This module performs no semantic inference, quality scoring,
persistence, retrieval, network access, or model calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Mapping

from backend.phase4_evidence_normalizer import normalize_phase4_inputs
from backend.phase4_models import (
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    Phase4EvidenceFact,
    Phase4EvidenceInput,
    Phase4SourceRef,
    build_phase4_stable_id,
)


MAX_DECISION_SAMPLES = 100
MAX_MECHANISM_SIGNALS = 12
MAX_MECHANISM_LENGTH = 2_000
MAX_IMPLEMENTATION_SIGNALS = 200

DIRECT_STRUCTURED_TYPES = frozenset({
    "phase2_evidence_card",
    "phase3_evidence_card",
})
DIRECT_CHANGE_TYPES = frozenset({
    "phase2_raw_change_summary",
    "phase3_raw_change_summary",
})
CAPABILITY_TYPES = frozenset({
    "phase2_capability_fact",
    "phase3_capability_fact",
})
CONTEXTUAL_TYPES = frozenset({
    "project_memory",
    "project_compact_facts",
    "compact_facts",
})
DIRECT_TYPES = DIRECT_STRUCTURED_TYPES | DIRECT_CHANGE_TYPES

GENERIC_MECHANISMS = frozenset({
    "built ai-powered system",
    "built an ai-powered system",
    "improved pipeline",
    "improved the pipeline",
    "enhanced retrieval",
    "optimized performance",
    "worked on resume generation",
    "improved reliability",
    "added intelligent features",
    "used advanced technologies",
    "developed backend functionality",
    "unknown",
})
GENERIC_IMPLEMENTATION = frozenset({
    "built ai-powered system",
    "built an ai-powered system",
    "improved pipeline",
    "improved the pipeline",
    "enhanced retrieval",
    "optimized performance",
    "worked on resume generation",
    "improved reliability",
    "added intelligent features",
    "used advanced technologies",
    "developed backend functionality",
    "unknown",
})

EXPLICIT_EVIDENCE_TYPE_MAP = {
    "bug_fix": EvidenceType.BUG_FIX,
    "bug_fix_update": EvidenceType.BUG_FIX,
    "validation": EvidenceType.VALIDATION,
    "validation_logic_update": EvidenceType.VALIDATION,
    "retrieval": EvidenceType.RETRIEVAL,
    "retrieval_logic_update": EvidenceType.RETRIEVAL,
    "testing": EvidenceType.TESTING,
    "test_update": EvidenceType.TESTING,
    "architecture": EvidenceType.ARCHITECTURE,
    "architecture_update": EvidenceType.ARCHITECTURE,
    "workflow": EvidenceType.WORKFLOW,
    "integration": EvidenceType.INTEGRATION,
    "integration_update": EvidenceType.INTEGRATION,
    "data_persistence": EvidenceType.DATA_PERSISTENCE,
    "memory_storage_update": EvidenceType.DATA_PERSISTENCE,
    "failure_recovery": EvidenceType.FAILURE_RECOVERY,
    "fallback_update": EvidenceType.FAILURE_RECOVERY,
    "optimization": EvidenceType.OPTIMIZATION,
    "configuration": EvidenceType.CONFIGURATION,
    "documentation": EvidenceType.DOCUMENTATION,
    "feature": EvidenceType.FEATURE,
}

_ALWAYS_UNSAFE_IMPACT_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\beliminated\b",
    r"\bguaranteed\b",
    r"\bfully\s+prevented\b",
    r"\bproduction[- ]scale\b",
    r"\benterprise[- ]grade\b",
    r"\breduced\s+hallucinations?\b",
    r"\bimproved\s+ats\s+success\b",
    r"\bimproved\s+(?:performance|reliability|accuracy)\b",
))
_NUMERIC_IMPACT_PATTERN = re.compile(
    r"(?:\b100\s*%|\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:x|times|hours?)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Phase4EvidenceBundle:
    project_id: str
    items: tuple[Phase4EvidenceInput, ...]
    source_refs: tuple[Phase4SourceRef, ...]
    provenance_key: str
    provenance_conflict: bool = False

    @property
    def input_ids(self) -> tuple[str, ...]:
        return tuple(item.input_id for item in self.items)

    @property
    def input_types(self) -> tuple[str, ...]:
        return tuple(item.input_type for item in self.items)


@dataclass(frozen=True)
class Phase4EvidenceSynthesisDecision:
    input_ids: tuple[str, ...]
    project_id: str
    decision: str
    reason: str
    evidence_fact_id: str | None = None


@dataclass(frozen=True)
class Phase4EvidenceSynthesisGroupCount:
    project_id: str
    input_type: str
    status: str
    evidence_type: str
    count: int


@dataclass(frozen=True)
class Phase4EvidenceSynthesisReport:
    input_count: int
    bundle_count: int
    evidence_fact_count: int
    accepted_count: int
    supporting_count: int
    weak_count: int
    rejected_count: int
    missing_mechanism_count: int
    missing_implementation_count: int
    unsafe_impact_dropped_count: int
    provenance_conflict_count: int
    decisions: tuple[Phase4EvidenceSynthesisDecision, ...] = ()
    grouped_counts: tuple[Phase4EvidenceSynthesisGroupCount, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SynthesisOutcome:
    fact: Phase4EvidenceFact | None
    status: str
    evidence_type: EvidenceType
    missing_mechanism: bool = False
    missing_implementation: bool = False
    unsafe_impacts_dropped: int = 0
    reason: str = "created"


def _canonical_ref_key(ref: Phase4SourceRef) -> str:
    return ref.to_json()


def _lineage_key(item: Phase4EvidenceInput) -> str:
    return "\n".join(sorted(_canonical_ref_key(ref) for ref in item.source_refs))


def _family(input_type: str) -> str:
    if input_type.startswith("phase2_"):
        return "phase2"
    if input_type.startswith("phase3_"):
        return "phase3"
    return input_type


def _item_priority(item: Phase4EvidenceInput) -> tuple[int, str, str]:
    if item.input_type in DIRECT_STRUCTURED_TYPES:
        priority = 0
    elif item.input_type in DIRECT_CHANGE_TYPES:
        priority = 1
    elif item.input_type in CAPABILITY_TYPES:
        priority = 2
    elif item.input_type in CONTEXTUAL_TYPES:
        priority = 3
    else:
        priority = 4
    return (priority, item.input_type, item.input_id)


def _bundle_sort_key(bundle: Phase4EvidenceBundle) -> tuple[str, str, str, str]:
    primary = bundle.items[0]
    return (bundle.project_id, primary.input_type, bundle.provenance_key, primary.input_id)


def _dedupe_refs(items: Iterable[Phase4EvidenceInput]) -> tuple[Phase4SourceRef, ...]:
    refs: list[Phase4SourceRef] = []
    seen: set[str] = set()
    for item in items:
        for ref in item.source_refs:
            key = _canonical_ref_key(ref)
            if key not in seen:
                seen.add(key)
                refs.append(Phase4SourceRef.from_dict(ref.to_dict()))
    return tuple(refs)


def _problem_conflict(items: list[Phase4EvidenceInput]) -> bool:
    problems = {item.problem_signal for item in items if item.problem_signal}
    return len(problems) > 1


def build_phase4_evidence_bundles(
    items: Iterable[Phase4EvidenceInput],
) -> list[Phase4EvidenceBundle]:
    """Bundle only one direct card and one direct change with exact lineage."""

    normalized = normalize_phase4_inputs(items)
    direct_groups: dict[tuple[str, str, str], list[Phase4EvidenceInput]] = {}
    singles: list[Phase4EvidenceInput] = []
    for item in normalized:
        if item.input_type in DIRECT_TYPES:
            key = (item.project_id, _family(item.input_type), _lineage_key(item))
            direct_groups.setdefault(key, []).append(item)
        else:
            singles.append(item)
    bundles: list[Phase4EvidenceBundle] = []
    for (project_id, _source_family, lineage), group in sorted(direct_groups.items()):
        ordered = sorted(group, key=_item_priority)
        cards = [item for item in ordered if item.input_type in DIRECT_STRUCTURED_TYPES]
        changes = [item for item in ordered if item.input_type in DIRECT_CHANGE_TYPES]
        conflict = len(cards) > 1 or len(changes) > 1 or _problem_conflict(ordered)
        if len(cards) == 1 and len(changes) == 1 and not conflict:
            pair = tuple(sorted([cards[0], changes[0]], key=_item_priority))
            bundles.append(Phase4EvidenceBundle(
                project_id=project_id,
                items=pair,
                source_refs=_dedupe_refs(pair),
                provenance_key=lineage,
            ))
        else:
            for item in ordered:
                bundles.append(Phase4EvidenceBundle(
                    project_id=project_id,
                    items=(item,),
                    source_refs=_dedupe_refs((item,)),
                    provenance_key=lineage,
                    provenance_conflict=conflict,
                ))
    for item in sorted(singles, key=_item_priority):
        bundles.append(Phase4EvidenceBundle(
            project_id=item.project_id,
            items=(item,),
            source_refs=_dedupe_refs((item,)),
            provenance_key=_lineage_key(item),
        ))
    return sorted(bundles, key=_bundle_sort_key)


def _unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            result.append(normalized)
    return result


def _bounded_mechanism(values: Iterable[str]) -> str:
    selected: list[str] = []
    current_length = 0
    for value in _unique_strings(values):
        if len(selected) >= MAX_MECHANISM_SIGNALS:
            break
        extra = len(value) + (2 if selected else 0)
        if current_length + extra > MAX_MECHANISM_LENGTH:
            break
        selected.append(value)
        current_length += extra
    return "; ".join(selected)


def _is_concrete_mechanism(mechanism: str) -> bool:
    values = [part.strip().casefold() for part in mechanism.split(";") if part.strip()]
    return bool(values) and any(value not in GENERIC_MECHANISMS for value in values)


def _has_concrete_implementation(values: Iterable[str]) -> bool:
    normalized = [" ".join(value.split()).casefold() for value in values if value.strip()]
    return bool(normalized) and any(value not in GENERIC_IMPLEMENTATION for value in normalized)


def _provenance_fallback(refs: Iterable[Phase4SourceRef]) -> list[str]:
    values: list[str] = []
    for ref in refs:
        path = ref.file_path or ""
        symbol = ref.symbol or ""
        if path and symbol:
            values.append(f"{path}::{symbol}")
        elif path:
            values.append(path)
        elif symbol:
            values.append(symbol)
    return _unique_strings(values)


def _metadata_enum(refs: Iterable[Phase4SourceRef], key: str, allowed: set[str]) -> str | None:
    values = {
        str(ref.metadata.get(key) or "").strip().lower()
        for ref in refs
        if isinstance(ref.metadata, Mapping) and str(ref.metadata.get(key) or "").strip()
    }
    return next(iter(values)) if len(values) == 1 and next(iter(values)) in allowed else None


def _evidence_type(bundle: Phase4EvidenceBundle) -> EvidenceType:
    explicit = _metadata_enum(
        bundle.source_refs,
        "evidence_type",
        {item.value for item in EvidenceType},
    )
    if explicit:
        return EvidenceType(explicit)
    mapped = {
        EXPLICIT_EVIDENCE_TYPE_MAP[signal.strip().lower()]
        for item in bundle.items
        if item.input_type in DIRECT_CHANGE_TYPES
        for signal in item.mechanism_signals
        if signal.strip().lower() in EXPLICIT_EVIDENCE_TYPE_MAP
    }
    return next(iter(mapped)) if len(mapped) == 1 else EvidenceType.UNKNOWN


def _metric_support(refs: Iterable[Phase4SourceRef]) -> MetricSupport:
    explicit = _metadata_enum(
        refs,
        "metric_support",
        {item.value for item in MetricSupport},
    )
    return MetricSupport(explicit) if explicit else MetricSupport.NONE


def _unsafe_impact(value: str, metric_support: MetricSupport) -> bool:
    if any(pattern.search(value) for pattern in _ALWAYS_UNSAFE_IMPACT_PATTERNS):
        return True
    return metric_support is MetricSupport.NONE and bool(_NUMERIC_IMPACT_PATTERN.search(value))


def _status_and_confidence(
    bundle: Phase4EvidenceBundle,
    *,
    has_implementation: bool,
) -> tuple[EvidenceStatus, Confidence]:
    input_types = set(bundle.input_types)
    is_direct = bool(input_types & DIRECT_TYPES)
    is_contextual = bool(input_types & (CAPABILITY_TYPES | CONTEXTUAL_TYPES))
    explicit_confidence = _metadata_enum(
        bundle.source_refs,
        "confidence",
        {item.value for item in Confidence},
    )
    if has_implementation and is_direct:
        confidence = Confidence.HIGH if explicit_confidence == "high" else Confidence.MEDIUM
        return EvidenceStatus.ACCEPTED, confidence
    if has_implementation and is_contextual:
        return EvidenceStatus.SUPPORTING, Confidence.LOW
    return EvidenceStatus.WEAK, Confidence.LOW


def _fact_id(
    bundle: Phase4EvidenceBundle,
    *,
    problem: str,
    mechanism: str,
    implementation: list[str],
    safe_impact: list[str],
    evidence_type: EvidenceType,
    technical_tags: list[str],
) -> str:
    return build_phase4_stable_id("p4ef_", bundle.project_id, {
        "problem": problem,
        "mechanism": mechanism,
        "implementation": implementation,
        "safe_impact": safe_impact,
        "evidence_type": evidence_type.value,
        "technical_tags": technical_tags,
        "source_input_ids": sorted(bundle.input_ids),
    })


def _synthesize_bundle(bundle: Phase4EvidenceBundle) -> _SynthesisOutcome:
    evidence_type = _evidence_type(bundle)
    if not bundle.source_refs or any(ref.project_id != bundle.project_id for ref in bundle.source_refs):
        return _SynthesisOutcome(None, "rejected", evidence_type, reason="missing_or_mismatched_provenance")
    problem_values = _unique_strings(
        item.problem_signal for item in bundle.items if item.problem_signal
    )
    if len(problem_values) > 1:
        return _SynthesisOutcome(None, "rejected", evidence_type, reason="conflicting_problem_signals")
    problem = problem_values[0] if problem_values else ""
    mechanism = _bounded_mechanism(
        signal for item in bundle.items for signal in item.mechanism_signals
    )
    if not mechanism or not _is_concrete_mechanism(mechanism):
        return _SynthesisOutcome(
            None,
            "rejected",
            evidence_type,
            missing_mechanism=True,
            reason="missing_or_generic_mechanism",
        )
    implementation = _unique_strings(
        signal for item in bundle.items for signal in item.implementation_signals
    )[:MAX_IMPLEMENTATION_SIGNALS]
    if not implementation:
        implementation = _provenance_fallback(bundle.source_refs)[:MAX_IMPLEMENTATION_SIGNALS]
    metric_support = _metric_support(bundle.source_refs)
    safe_impact: list[str] = []
    unsafe_dropped = 0
    for impact in _unique_strings(
        signal for item in bundle.items for signal in item.impact_signals
    ):
        if _unsafe_impact(impact, metric_support):
            unsafe_dropped += 1
        else:
            safe_impact.append(impact)
    technical_tags = sorted(
        _unique_strings(tag for item in bundle.items for tag in item.technical_tags),
        key=lambda value: (value.casefold(), value),
    )
    has_concrete_implementation = _has_concrete_implementation(implementation)
    status, confidence = _status_and_confidence(
        bundle,
        has_implementation=has_concrete_implementation,
    )
    fact = Phase4EvidenceFact(
        evidence_fact_id=_fact_id(
            bundle,
            problem=problem,
            mechanism=mechanism,
            implementation=implementation,
            safe_impact=safe_impact,
            evidence_type=evidence_type,
            technical_tags=technical_tags,
        ),
        project_id=bundle.project_id,
        problem=problem,
        mechanism=mechanism,
        implementation=implementation,
        safe_impact=safe_impact,
        evidence_type=evidence_type,
        source_refs=list(bundle.source_refs),
        confidence=confidence,
        metric_support=metric_support,
        allowed_claims=[],
        forbidden_claims=[],
        technical_tags=technical_tags,
        status=status,
        quality_score=None,
        quality_breakdown={},
    )
    return _SynthesisOutcome(
        fact,
        status.value,
        evidence_type,
        missing_implementation=not has_concrete_implementation,
        unsafe_impacts_dropped=unsafe_dropped,
    )


def synthesize_phase4_evidence_fact(
    item: Phase4EvidenceInput,
) -> Phase4EvidenceFact | None:
    if not isinstance(item, Phase4EvidenceInput):
        raise TypeError("synthesize_phase4_evidence_fact expects Phase4EvidenceInput")
    if not item.source_refs:
        return None
    bundle = build_phase4_evidence_bundles([item])[0]
    return _synthesize_bundle(bundle).fact


def _fact_sort_key(fact: Phase4EvidenceFact) -> tuple[str, str, str, str, str, str]:
    primary = fact.source_refs[0]
    return (
        fact.project_id,
        fact.status.value,
        fact.evidence_type.value,
        primary.source_type,
        primary.source_id,
        fact.evidence_fact_id,
    )


def synthesize_phase4_evidence_facts(
    items: Iterable[Phase4EvidenceInput],
) -> tuple[list[Phase4EvidenceFact], Phase4EvidenceSynthesisReport]:
    input_records = list(items)
    bundles = build_phase4_evidence_bundles(input_records)
    facts: list[Phase4EvidenceFact] = []
    decisions: list[Phase4EvidenceSynthesisDecision] = []
    grouped: dict[tuple[str, str, str, str], int] = {}
    status_counts = {status.value: 0 for status in EvidenceStatus}
    missing_mechanism = 0
    missing_implementation = 0
    unsafe_dropped = 0
    conflict_keys = {
        (bundle.project_id, bundle.provenance_key)
        for bundle in bundles
        if bundle.provenance_conflict
    }
    for bundle in bundles:
        outcome = _synthesize_bundle(bundle)
        status_counts[outcome.status] += 1
        missing_mechanism += int(outcome.missing_mechanism)
        missing_implementation += int(outcome.missing_implementation)
        unsafe_dropped += outcome.unsafe_impacts_dropped
        input_type = "+".join(bundle.input_types)
        group_key = (
            bundle.project_id,
            input_type,
            outcome.status,
            outcome.evidence_type.value,
        )
        grouped[group_key] = grouped.get(group_key, 0) + 1
        if outcome.fact is not None:
            facts.append(outcome.fact)
        if len(decisions) < MAX_DECISION_SAMPLES:
            decisions.append(Phase4EvidenceSynthesisDecision(
                input_ids=bundle.input_ids,
                project_id=bundle.project_id,
                decision="created" if outcome.fact is not None else "rejected",
                reason=outcome.reason,
                evidence_fact_id=outcome.fact.evidence_fact_id if outcome.fact else None,
            ))
    facts.sort(key=_fact_sort_key)
    decisions.sort(key=lambda item: (
        item.project_id,
        item.decision,
        item.reason,
        item.input_ids,
    ))
    group_counts = tuple(
        Phase4EvidenceSynthesisGroupCount(
            project_id=key[0],
            input_type=key[1],
            status=key[2],
            evidence_type=key[3],
            count=count,
        )
        for key, count in sorted(grouped.items())
    )
    report = Phase4EvidenceSynthesisReport(
        input_count=len(input_records),
        bundle_count=len(bundles),
        evidence_fact_count=len(facts),
        accepted_count=status_counts[EvidenceStatus.ACCEPTED.value],
        supporting_count=status_counts[EvidenceStatus.SUPPORTING.value],
        weak_count=status_counts[EvidenceStatus.WEAK.value],
        rejected_count=status_counts[EvidenceStatus.REJECTED.value],
        missing_mechanism_count=missing_mechanism,
        missing_implementation_count=missing_implementation,
        unsafe_impact_dropped_count=unsafe_dropped,
        provenance_conflict_count=len(conflict_keys),
        decisions=tuple(decisions),
        grouped_counts=group_counts,
    )
    return facts, report


__all__ = [
    "Phase4EvidenceBundle",
    "Phase4EvidenceSynthesisDecision",
    "Phase4EvidenceSynthesisGroupCount",
    "Phase4EvidenceSynthesisReport",
    "build_phase4_evidence_bundles",
    "synthesize_phase4_evidence_fact",
    "synthesize_phase4_evidence_facts",
]
