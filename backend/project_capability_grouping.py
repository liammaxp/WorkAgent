"""Deterministically group project evidence into unscored capability candidates.

This module reuses canonical signal extraction and taxonomy lookup.  It performs
no proof evaluation, scoring, boundary inheritance, fact generation, or I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend.project_capability_extractor import (
    ProjectFactSignalExtraction,
    extract_project_evidence_fact_signals_many,
)
from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_taxonomy import (
    get_capability_types_for_signal,
    resolve_capability_type,
)
from backend.project_evidence_models import ProjectEvidenceFact


AMBIGUOUS_CAPABILITY_LABELS = frozenset({"validation_and_repair"})


@dataclass(frozen=True)
class CapabilityGroupingResult:
    project_id: str
    candidates: tuple[CapabilityCandidate, ...]
    input_evidence_ids: tuple[str, ...]
    matched_evidence_ids: tuple[str, ...]
    unmatched_evidence_ids: tuple[str, ...]
    ambiguous_evidence_ids: tuple[str, ...]
    skipped_evidence_ids: tuple[str, ...]
    matched_signal_count: int
    unmatched_signal_count: int
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "candidates": [candidate.to_safe_dict() for candidate in self.candidates],
            "input_evidence_ids": list(self.input_evidence_ids),
            "matched_evidence_ids": list(self.matched_evidence_ids),
            "unmatched_evidence_ids": list(self.unmatched_evidence_ids),
            "ambiguous_evidence_ids": list(self.ambiguous_evidence_ids),
            "skipped_evidence_ids": list(self.skipped_evidence_ids),
            "matched_signal_count": self.matched_signal_count,
            "unmatched_signal_count": self.unmatched_signal_count,
            "diagnostics": _thaw(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _normalize_project_id(project_id: str) -> str:
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")
    normalized = " ".join(project_id.split())
    if not normalized:
        raise ValueError("project_id must not be blank")
    return normalized


def _dedupe_facts(
    project_id: str,
    evidence_facts: Sequence[ProjectEvidenceFact],
) -> tuple[ProjectEvidenceFact, ...]:
    if not isinstance(evidence_facts, Sequence) or isinstance(evidence_facts, (str, bytes)):
        raise TypeError("evidence_facts must be a sequence of ProjectEvidenceFact values")
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in evidence_facts):
        raise TypeError("evidence_facts must contain only ProjectEvidenceFact values")
    mismatched = sorted({fact.project_id for fact in evidence_facts if fact.project_id != project_id})
    if mismatched:
        raise ValueError("all evidence facts must belong to the requested project_id")

    by_id: dict[str, tuple[str, ProjectEvidenceFact]] = {}
    for fact in evidence_facts:
        payload = fact.to_json()
        previous = by_id.get(fact.evidence_fact_id)
        if previous is None:
            by_id[fact.evidence_fact_id] = (payload, fact)
        elif previous[0] != payload:
            raise ValueError("same evidence_fact_id has conflicting semantic content")
    return tuple(by_id[evidence_id][1] for evidence_id in sorted(by_id))


def _explicit_capability_labels(fact: ProjectEvidenceFact) -> tuple[str, ...]:
    labels: set[str] = set()
    for tag in fact.technical_tags:
        normalized = "_".join(tag.strip().casefold().replace("-", " ").split())
        if normalized:
            labels.add(normalized)
    return tuple(sorted(labels))


def _is_ambiguous_without_mapping(
    fact: ProjectEvidenceFact,
    extraction: ProjectFactSignalExtraction,
) -> bool:
    labels = set(_explicit_capability_labels(fact))
    if labels & AMBIGUOUS_CAPABILITY_LABELS:
        return True
    return bool(extraction.rejected_candidates) and any(
        code.startswith("ambiguous_") for code in extraction.rejected_candidates
    )


def group_evidence_fact_signals(
    evidence_fact: ProjectEvidenceFact,
) -> Mapping[str, tuple[str, ...]]:
    """Return exact canonical capability mappings for one Evidence Fact."""

    extractions, _report = extract_project_evidence_fact_signals_many((evidence_fact,))
    grouped: dict[str, set[str]] = {}
    for signal in extractions[0].signals:
        for capability_type in get_capability_types_for_signal(signal):
            grouped.setdefault(capability_type, set()).add(signal)
    return MappingProxyType({
        capability_type: tuple(sorted(signals))
        for capability_type, signals in sorted(grouped.items())
    })


def group_project_evidence_facts(
    *,
    project_id: str,
    evidence_facts: Sequence[ProjectEvidenceFact],
) -> CapabilityGroupingResult:
    """Group one project's facts without applying taxonomy proof thresholds."""

    normalized_project_id = _normalize_project_id(project_id)
    facts = _dedupe_facts(normalized_project_id, evidence_facts)
    extractions, _signal_report = extract_project_evidence_fact_signals_many(facts)
    extraction_by_id = {item.evidence_fact_id: item for item in extractions}

    evidence_by_capability: dict[str, set[str]] = {}
    signals_by_capability: dict[str, set[str]] = {}
    source_types_by_capability: dict[str, set[str]] = {}
    matched: set[str] = set()
    unmatched: set[str] = set()
    ambiguous: set[str] = set()
    matched_signal_count = 0
    unmatched_signal_count = 0

    for fact in facts:
        extraction = extraction_by_id[fact.evidence_fact_id]
        fact_mappings: dict[str, set[str]] = {}
        for signal in extraction.signals:
            capability_types = get_capability_types_for_signal(signal)
            if not capability_types:
                unmatched_signal_count += 1
                continue
            matched_signal_count += 1
            for capability_type in capability_types:
                fact_mappings.setdefault(capability_type, set()).add(signal)

        # Exact canonical or alias labels may identify an already signal-supported
        # candidate, but never create support without a canonical extracted signal.
        resolved_labels = {
            resolved
            for label in _explicit_capability_labels(fact)
            if (resolved := resolve_capability_type(label)) is not None
        }
        for capability_type in resolved_labels & set(fact_mappings):
            fact_mappings.setdefault(capability_type, set())

        if fact_mappings:
            matched.add(fact.evidence_fact_id)
            for capability_type, signals in fact_mappings.items():
                evidence_by_capability.setdefault(capability_type, set()).add(fact.evidence_fact_id)
                signals_by_capability.setdefault(capability_type, set()).update(signals)
                source_types_by_capability.setdefault(capability_type, set()).update(
                    ref.source_type for ref in fact.source_refs
                )
        elif _is_ambiguous_without_mapping(fact, extraction):
            ambiguous.add(fact.evidence_fact_id)
        else:
            unmatched.add(fact.evidence_fact_id)

    candidates = tuple(
        CapabilityCandidate(
            project_id=normalized_project_id,
            capability_type=capability_type,
            supporting_evidence_ids=tuple(sorted(evidence_by_capability[capability_type])),
            supporting_signals=tuple(sorted(signals_by_capability[capability_type])),
            conflicting_signals=(),
            candidate_score=0.0,
            metadata={
                "evaluation_state": "unscored",
                "source_type_count": len(source_types_by_capability[capability_type]),
            },
        )
        for capability_type in sorted(evidence_by_capability)
    )
    input_ids = tuple(fact.evidence_fact_id for fact in facts)
    diagnostics = MappingProxyType({
        "candidate_count": len(candidates),
        "duplicate_input_count": len(evidence_facts) - len(facts),
        "evaluation_state": "unscored",
        "fact_count": len(facts),
    })
    return CapabilityGroupingResult(
        project_id=normalized_project_id,
        candidates=candidates,
        input_evidence_ids=input_ids,
        matched_evidence_ids=tuple(sorted(matched)),
        unmatched_evidence_ids=tuple(sorted(unmatched)),
        ambiguous_evidence_ids=tuple(sorted(ambiguous)),
        skipped_evidence_ids=(),
        matched_signal_count=matched_signal_count,
        unmatched_signal_count=unmatched_signal_count,
        diagnostics=diagnostics,
    )


__all__ = [
    "AMBIGUOUS_CAPABILITY_LABELS",
    "CapabilityGroupingResult",
    "group_evidence_fact_signals",
    "group_project_evidence_facts",
]
