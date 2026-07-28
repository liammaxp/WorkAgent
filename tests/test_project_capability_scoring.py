from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

from backend.project_capability_grouping import group_project_evidence_facts
from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_scoring import (
    CAPABILITY_SUPPORT_ELIGIBILITY_STATUSES,
    EVIDENCE_QUALITY_WEIGHT,
    MECHANISM_SPECIFICITY_WEIGHT,
    SIGNAL_COVERAGE_WEIGHT,
    SOURCE_DIVERSITY_WEIGHT,
    CapabilitySupportAssessment,
    assess_capability_candidate_support,
    assess_project_capability_candidates,
)
from backend.project_evidence_memory import load_project_evidence_memory
from backend.project_evidence_models import (
    EvidenceSourceRef,
    EvidenceType,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "information" / "project_evidence_memory.json"


def _fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    signals: tuple[str, ...] = (),
    mechanism: str = "stable hash-based identity",
    quality: float = 90,
    evidence_type: EvidenceType = EvidenceType.VALIDATION,
    source_types: tuple[str, ...] = ("github_evidence_card",),
) -> ProjectEvidenceFact:
    refs = [
        EvidenceSourceRef(
            source_type=source_type,
            source_id=f"source-{evidence_id}-{index}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{evidence_id}-{index}".encode()).hexdigest(),
        )
        for index, source_type in enumerate(source_types)
    ]
    return ProjectEvidenceFact(
        project_id=project_id,
        mechanism=mechanism,
        implementation=["Applied a deterministic structured implementation detail."],
        source_refs=refs,
        evidence_type=evidence_type,
        technical_tags=list(signals),
        quality_score=quality,
        evidence_fact_id=evidence_id,
    )


def _candidate(
    capability_type: str,
    facts: tuple[ProjectEvidenceFact, ...],
    signals: tuple[str, ...],
    *,
    conflicts: tuple[str, ...] = (),
) -> CapabilityCandidate:
    return CapabilityCandidate(
        project_id=facts[0].project_id,
        capability_type=capability_type,
        supporting_evidence_ids=tuple(fact.evidence_fact_id for fact in facts),
        supporting_signals=signals,
        conflicting_signals=conflicts,
        candidate_score=0.0,
        metadata={"evaluation_state": "unscored"},
    )


def _assess(candidate: CapabilityCandidate, facts: tuple[ProjectEvidenceFact, ...]):
    return assess_capability_candidate_support(
        candidate=candidate,
        evidence_index={fact.evidence_fact_id: fact for fact in facts},
    )


def test_candidate_meeting_all_taxonomy_minimums_is_eligible():
    fact = _fact("pef_eligible", signals=("quality_dimensions",))
    candidate = _candidate("output_quality_control", (fact,), ("quality_dimensions",))
    assessment = _assess(candidate, (fact,))

    assert assessment.meets_evidence_minimum
    assert assessment.meets_signal_group_minimum
    assert assessment.meets_mechanism_minimum
    assert not assessment.has_explicit_conflict
    assert assessment.eligibility_status == "eligible"
    assert not isinstance(assessment, ProjectCapabilityFact)


def test_candidate_below_evidence_minimum_is_ineligible():
    signals = ("source_grounding", "claim_validation", "structured_output_validation")
    fact = _fact("pef_one", signals=signals)
    assessment = _assess(_candidate("llm_reliability", (fact,), signals), (fact,))
    assert assessment.distinct_signal_group_count == 3
    assert not assessment.meets_evidence_minimum
    assert assessment.eligibility_status == "insufficient_evidence"


def test_candidate_missing_required_signal_groups_is_ineligible():
    signals = ("query_operation", "retrieval")
    fact = _fact("pef_one_group", signals=signals, evidence_type=EvidenceType.RETRIEVAL)
    assessment = _assess(_candidate("retrieval_and_reranking", (fact,), signals), (fact,))
    assert assessment.distinct_signal_group_count == 1
    assert assessment.missing_signal_groups == ("required_group_2",)
    assert assessment.eligibility_status == "insufficient_signal_coverage"


def test_candidate_without_concrete_mechanisms_is_ineligible():
    fact = _fact("pef_generic", signals=("quality_dimensions",), mechanism="Implemented")
    assessment = _assess(
        _candidate("output_quality_control", (fact,), ("quality_dimensions",)), (fact,)
    )
    assert assessment.mechanism_count == 0
    assert assessment.eligibility_status == "insufficient_mechanism_support"


def test_candidate_with_explicit_conflict_is_blocked():
    fact = _fact("pef_conflict", signals=("quality_dimensions",))
    candidate = _candidate(
        "output_quality_control", (fact,), ("quality_dimensions",), conflicts=("unsupported_metric",)
    )
    assessment = _assess(candidate, (fact,))
    assert assessment.has_explicit_conflict
    assert assessment.conflict_penalty > 0
    assert assessment.eligibility_status == "explicitly_conflicted"
    assert "explicit_conflict:unsupported_metric" in assessment.reasons


def test_support_score_is_composed_from_deterministic_components():
    fact = _fact("pef_components", signals=("quality_dimensions",), quality=80)
    assessment = _assess(
        _candidate("output_quality_control", (fact,), ("quality_dimensions",)), (fact,)
    )
    expected = round(
        assessment.evidence_quality_score * EVIDENCE_QUALITY_WEIGHT
        + assessment.signal_coverage_score * SIGNAL_COVERAGE_WEIGHT
        + assessment.mechanism_specificity_score * MECHANISM_SPECIFICITY_WEIGHT
        + assessment.source_diversity_score * SOURCE_DIVERSITY_WEIGHT,
        6,
    )
    assert assessment.support_score == expected
    assert 0.0 <= assessment.support_score <= 1.0
    assert sum((EVIDENCE_QUALITY_WEIGHT, SIGNAL_COVERAGE_WEIGHT, MECHANISM_SPECIFICITY_WEIGHT, SOURCE_DIVERSITY_WEIGHT)) == 1.0


def test_high_support_score_cannot_override_missing_required_group():
    signals = ("query_operation", "retrieval")
    fact = _fact(
        "pef_high", signals=signals, quality=100, evidence_type=EvidenceType.RETRIEVAL,
        source_types=("github_evidence_card", "project_memory", "test_evidence"),
    )
    assessment = _assess(_candidate("retrieval_and_reranking", (fact,), signals), (fact,))
    assert assessment.support_score >= 0.8
    assert assessment.eligibility_status == "insufficient_signal_coverage"


def test_low_quality_evidence_does_not_gain_support_from_count_alone():
    facts = tuple(
        _fact(f"pef_low_{index}", signals=("quality_dimensions",), quality=10)
        for index in range(3)
    )
    assessment = _assess(
        _candidate("output_quality_control", facts, ("quality_dimensions",)), facts
    )
    assert assessment.evidence_count == 3
    assert assessment.diagnostics["eligible_evidence_count"] == 0
    assert not assessment.meets_evidence_minimum
    assert assessment.eligibility_status == "insufficient_evidence"


def test_multiple_signals_in_one_group_count_as_one_distinct_group():
    signals = ("query_operation", "retrieval")
    fact = _fact("pef_dedup_group", signals=signals, evidence_type=EvidenceType.RETRIEVAL)
    assessment = _assess(_candidate("retrieval_and_reranking", (fact,), signals), (fact,))
    assert assessment.distinct_signal_group_count == 1


def test_duplicate_mechanisms_do_not_inflate_specificity():
    facts = (
        _fact("pef_mechanism_a", signals=("source_grounding",), mechanism="schema-bound claim validation"),
        _fact("pef_mechanism_b", signals=("claim_validation",), mechanism="schema-bound claim validation"),
    )
    candidate = _candidate(
        "evidence_grounded_generation", facts, ("source_grounding", "claim_validation")
    )
    assessment = _assess(candidate, facts)
    assert assessment.mechanism_count == 1
    assert assessment.mechanism_specificity_score == 1.0


def test_source_diversity_is_bounded_and_does_not_replace_required_proof():
    signals = ("query_operation",)
    fact = _fact(
        "pef_sources", signals=signals, evidence_type=EvidenceType.RETRIEVAL,
        source_types=("github_evidence_card", "project_memory", "test_evidence", "documentation_evidence"),
    )
    assessment = _assess(_candidate("retrieval_and_reranking", (fact,), signals), (fact,))
    assert assessment.source_type_count == 4
    assert assessment.source_diversity_score == 1.0
    assert assessment.eligibility_status == "insufficient_signal_coverage"


def test_assessment_does_not_mutate_capability_candidate():
    fact = _fact("pef_immutable", signals=("quality_dimensions",))
    candidate = _candidate("output_quality_control", (fact,), ("quality_dimensions",))
    before = candidate.to_dict()
    _assess(candidate, (fact,))
    assert candidate.to_dict() == before
    assert candidate.candidate_score == 0.0
    assert candidate.metadata["evaluation_state"] == "unscored"
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_score = 1.0


def test_missing_evidence_reference_fails_closed():
    fact = _fact("pef_missing", signals=("quality_dimensions",))
    candidate = _candidate("output_quality_control", (fact,), ("quality_dimensions",))
    with pytest.raises(ValueError, match="missing evidence reference"):
        assess_capability_candidate_support(candidate=candidate, evidence_index={})


def test_unknown_candidate_capability_type_fails_closed():
    fact = _fact("pef_unknown_type", signals=("quality_dimensions",))
    candidate = _candidate("unknown_capability", (fact,), ("quality_dimensions",))
    with pytest.raises(ValueError, match="canonical taxonomy"):
        _assess(candidate, (fact,))


def test_cross_project_evidence_reference_fails_closed():
    fact = _fact("pef_cross", signals=("quality_dimensions",))
    candidate = _candidate("output_quality_control", (fact,), ("quality_dimensions",))
    other = _fact("pef_cross", project_id="project-b", signals=("quality_dimensions",))
    with pytest.raises(ValueError, match="cross-project"):
        assess_capability_candidate_support(candidate=candidate, evidence_index={"pef_cross": other})


def test_support_assessment_is_input_order_independent():
    facts = (
        _fact("pef_z", signals=("source_grounding",)),
        _fact("pef_a", signals=("claim_validation",)),
    )
    candidate = _candidate(
        "evidence_grounded_generation", facts, ("source_grounding", "claim_validation")
    )
    second_candidate = _candidate(
        "claim_validation", facts, ("source_grounding", "claim_validation")
    )
    first = assess_project_capability_candidates(
        project_id="project-a", candidates=[candidate, second_candidate], evidence_facts=list(facts)
    )
    second = assess_project_capability_candidates(
        project_id="project-a",
        candidates=[second_candidate, candidate],
        evidence_facts=list(reversed(facts)),
    )
    assert first == second
    assert [item.to_json() for item in first] == [item.to_json() for item in second]


def test_conflicting_duplicate_evidence_records_fail_closed():
    fact = _fact("pef_duplicate", signals=("quality_dimensions",))
    conflicting = _fact(
        "pef_duplicate", signals=("quality_dimensions",), mechanism="atomic artifact replacement"
    )
    candidate = _candidate("output_quality_control", (fact,), ("quality_dimensions",))
    with pytest.raises(ValueError, match="conflicting semantic content"):
        assess_project_capability_candidates(
            project_id="project-a",
            candidates=[candidate],
            evidence_facts=[fact, conflicting],
        )


def test_support_assessment_serialization_is_deterministic_and_safe():
    sentinel = "PRIVATE_RAW_GITHUB_PATCH_SENTINEL"
    fact = _fact("pef_safe", signals=("quality_dimensions",), mechanism=sentinel)
    assessment = _assess(
        _candidate("output_quality_control", (fact,), ("quality_dimensions",)), (fact,)
    )
    serialized = assessment.to_json().casefold()
    assert serialized == assessment.to_json().casefold()
    assert sentinel.casefold() not in serialized
    assert not {"raw", "patch", "diff", "source_code", "github_context"} & set(assessment.to_dict())


def test_support_assessment_does_not_load_or_inherit_claim_boundaries():
    source = (ROOT / "backend" / "project_capability_scoring.py").read_text(encoding="utf-8").casefold()
    assert "project_claim_boundaries" not in source
    assert "allowed_claims" not in source
    assert "forbidden_claims" not in source


def test_support_assessment_does_not_generate_project_capability_fact():
    fact = _fact("pef_no_fact", signals=("quality_dimensions",))
    assessment = _assess(
        _candidate("output_quality_control", (fact,), ("quality_dimensions",)), (fact,)
    )
    assert isinstance(assessment, CapabilitySupportAssessment)
    assert not isinstance(assessment, ProjectCapabilityFact)
    assert "capability_id" not in assessment.to_dict()
    assert CAPABILITY_SUPPORT_ELIGIBILITY_STATUSES == {
        "eligible", "insufficient_evidence", "insufficient_signal_coverage",
        "insufficient_mechanism_support", "explicitly_conflicted",
    }


def test_real_candidates_can_be_assessed_read_only():
    before = ARTIFACT.read_bytes()
    before_mtime = ARTIFACT.stat().st_mtime_ns
    loaded = load_project_evidence_memory(ARTIFACT)
    assert loaded.status == "ready" and loaded.snapshot is not None

    assessments = []
    for project in loaded.snapshot.projects:
        grouping = group_project_evidence_facts(
            project_id=project.project_id, evidence_facts=project.evidence_facts
        )
        project_assessments = assess_project_capability_candidates(
            project_id=project.project_id,
            candidates=grouping.candidates,
            evidence_facts=project.evidence_facts,
        )
        assert all(item.project_id == project.project_id for item in project_assessments)
        assessments.extend(project_assessments)
    assert len(loaded.snapshot.projects) == 11
    assert len(assessments) == 57
    assert sum(len(project.capability_facts) for project in loaded.snapshot.projects) == 0
    assert ARTIFACT.read_bytes() == before
    assert ARTIFACT.stat().st_mtime_ns == before_mtime
    assert not (ROOT / "information" / "project_capability_memory.json").exists()


def test_project_capability_scoring_uses_semantic_naming():
    source = (ROOT / "backend" / "project_capability_scoring.py").read_text(encoding="utf-8").casefold()
    forbidden = ("phase" + "5", "phase_" + "5", "project_memory_" + "phase" + "5", "use_" + "phase" + "5")
    assert not any(token in source for token in forbidden)
