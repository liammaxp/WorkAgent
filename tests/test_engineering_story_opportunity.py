from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass, replace
from hashlib import sha256
import inspect
from pathlib import Path

import pytest

from backend.engineering_story_clustering import (
    StoryCluster,
    cluster_story_evidence_bundle,
)
from backend.engineering_story_evidence import resolve_story_evidence_bundle
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryContextGap,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_opportunity import (
    StoryOpportunityDetectionError,
    StoryOpportunityDetectionErrorCode,
    StoryOpportunityDetectionResult,
    StoryOpportunityDiagnosticCode,
    StoryOpportunityProjectContext,
    StoryOpportunitySignalDecision,
    StoryOpportunitySignalStrength,
    build_story_opportunity_project_context,
    detect_story_opportunity,
)
from backend.engineering_story_reconstruction import (
    StoryFieldDecisionReason,
    StoryFieldReconstructionDecision,
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
    StoryReconstructionResult,
)
from backend.engineering_story_sufficiency import (
    EngineeringStorySufficiencyResult,
    evaluate_engineering_story_sufficiency,
)
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"
_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_VALUES = {
    EngineeringStoryFieldName.PROBLEM_CONTEXT: "Requests could survive a partial service failure",
    EngineeringStoryFieldName.TRIGGER: "A deterministic failure exposed the recovery gap",
    EngineeringStoryFieldName.BEFORE_STATE: "The workflow stopped on the first failed operation",
    EngineeringStoryFieldName.DECISION: "Use an explicit recovery boundary around the workflow",
    EngineeringStoryFieldName.MECHANISM: "Bounded recovery with validation gates",
    EngineeringStoryFieldName.IMPLEMENTATION: "Added typed retry state and deterministic checks",
    EngineeringStoryFieldName.TRADEOFF: "Rejected unbounded retries to preserve predictable behavior",
    EngineeringStoryFieldName.VALIDATION: "Failure-specific tests exercise recovery and rejection paths",
    EngineeringStoryFieldName.AFTER_STATE: "Invalid recovery states fail closed",
    EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "The invalid state is rejected before persistence",
    EngineeringStoryFieldName.OWNERSHIP: "Owned the recovery boundary and regression validation",
    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT: "Protected operators consuming the workflow",
}


@dataclass(frozen=True)
class _Case:
    cluster: StoryCluster
    reconstruction: StoryReconstructionResult
    sufficiency: EngineeringStorySufficiencyResult
    memory: ProjectEvidenceMemory


def _source_ref(
    evidence_id: str,
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change-main",
    file_path: str = "backend/recovery.py",
    symbol: str = "recover",
) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type="github_evidence_chunk",
        source_id=f"chunk_{evidence_id}",
        project_id=project_id,
        content_hash=sha256(
            f"{project_id}|{evidence_id}|{change_id}|{file_path}|{symbol}".encode()
        ).hexdigest(),
        repo="owner/workagent" if project_id == PROJECT_ID else "owner/event-lottery",
        commit_sha="aaaaaaa",
        file_path=file_path,
        symbol=symbol,
        metadata={"change_id": change_id},
    )


def _fact(
    evidence_id: str,
    evidence_type: EvidenceType,
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change-main",
    file_path: str = "backend/recovery.py",
    symbol: str = "recover",
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=evidence_id,
        problem="A bounded workflow failure required deterministic handling",
        mechanism="Bounded recovery with validation gates",
        implementation=["Added typed retry state and deterministic checks"],
        safe_impact=["Invalid recovery states are rejected before persistence"],
        source_refs=[_source_ref(
            evidence_id,
            project_id=project_id,
            change_id=change_id,
            file_path=file_path,
            symbol=symbol,
        )],
        evidence_type=evidence_type,
        status=status,
        confidence=Confidence.HIGH,
        metric_support=MetricSupport.NONE,
        technical_tags=["recovery", "validation"],
        quality_score=92,
    )


def _capability(
    facts: tuple[ProjectEvidenceFact, ...],
    *,
    capability_type: str = "failure_recovery",
    project_id: str = PROJECT_ID,
) -> ProjectCapabilityFact:
    return ProjectCapabilityFact(
        project_id=project_id,
        capability_id=f"pcf_{capability_type}",
        capability_type=capability_type,
        present=True,
        source_evidence_fact_ids=[item.evidence_fact_id for item in facts],
        mechanisms=["Bounded recovery with validation gates"],
        confidence=Confidence.HIGH,
        metric_support=MetricSupport.NONE,
    )


def _boundaries(
    facts: tuple[ProjectEvidenceFact, ...],
) -> tuple[ProjectClaimBoundary, ...]:
    values = tuple(build_project_evidence_claim_boundary(item) for item in facts)
    return tuple(item for item in values if item is not None)


def _cluster(
    evidence_types: tuple[EvidenceType, ...],
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change-main",
    file_path: str = "backend/recovery.py",
    symbol: str = "recover",
    capability_type: str | None = None,
    evidence_statuses: tuple[EvidenceStatus, ...] | None = None,
) -> tuple[
    StoryCluster,
    tuple[ProjectEvidenceFact, ...],
    tuple[ProjectCapabilityFact, ...],
    tuple[ProjectClaimBoundary, ...],
]:
    statuses = evidence_statuses or tuple(
        EvidenceStatus.ACCEPTED for _ in evidence_types
    )
    if len(statuses) != len(evidence_types):
        raise ValueError("evidence_statuses must align with evidence_types")
    facts = tuple(
        _fact(
            f"pef_{change_id.replace('-', '_')}_{index}",
            evidence_type,
            project_id=project_id,
            change_id=change_id,
            file_path=file_path,
            symbol=symbol,
            status=statuses[index],
        )
        for index, evidence_type in enumerate(evidence_types)
    )
    capabilities = (
        (_capability(facts, capability_type=capability_type, project_id=project_id),)
        if capability_type is not None
        else ()
    )
    boundaries = _boundaries(facts)
    bundle = resolve_story_evidence_bundle(
        project_id=project_id,
        evidence_fact_ids=tuple(item.evidence_fact_id for item in facts),
        evidence_facts=facts,
        capability_ids=tuple(item.capability_id for item in capabilities),
        capability_facts=capabilities,
        claim_boundary_ids=tuple(item.boundary_id for item in boundaries),
        claim_boundaries=boundaries,
    )
    result = cluster_story_evidence_bundle(bundle)
    assert len(result.clusters) == 1
    return result.clusters[0], facts, capabilities, boundaries


def _case(
    *,
    evidence_types: tuple[EvidenceType, ...] = (EvidenceType.ARCHITECTURE,),
    story_type: EngineeringStoryType = EngineeringStoryType.ARCHITECTURE_CHANGE,
    positive: tuple[EngineeringStoryFieldName, ...] = (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ),
    quality: StoryReconstructionQuality = StoryReconstructionQuality.PARTIAL,
    project_id: str = PROJECT_ID,
    change_id: str = "change-main",
    file_path: str = "backend/recovery.py",
    symbol: str = "recover",
    capability_type: str | None = None,
    evidence_statuses: tuple[EvidenceStatus, ...] | None = None,
) -> _Case:
    cluster, facts, capabilities, boundaries = _cluster(
        evidence_types,
        project_id=project_id,
        change_id=change_id,
        file_path=file_path,
        symbol=symbol,
        capability_type=capability_type,
        evidence_statuses=evidence_statuses,
    )
    primary_fact_id = facts[0].evidence_fact_id
    positive_set = set(positive)
    story_fields: dict[str, EngineeringStoryField] = {}
    decisions: list[StoryFieldReconstructionDecision] = []
    for name in _FIELD_ORDER:
        is_positive = name in positive_set
        field = EngineeringStoryField(
            value=_VALUES[name] if is_positive else None,
            evidence_state=(
                StoryFieldEvidenceState.CONFIRMED
                if is_positive
                else StoryFieldEvidenceState.PLAUSIBLE_MISSING
            ),
            evidence_fact_ids=(primary_fact_id,) if is_positive else (),
        )
        story_fields[name.value] = field
        decisions.append(StoryFieldReconstructionDecision(
            field_name=name,
            resulting_state=field.evidence_state,
            evidence_fact_ids=field.evidence_fact_ids,
            capability_fact_ids=(),
            claim_boundary_ids=(),
            reason_code=(
                StoryFieldDecisionReason.DIRECT_AUTHORITATIVE_EVIDENCE
                if is_positive
                else (
                    StoryFieldDecisionReason.MISSING_HUMAN_CONTEXT
                    if name in {
                        EngineeringStoryFieldName.TRIGGER,
                        EngineeringStoryFieldName.DECISION,
                        EngineeringStoryFieldName.TRADEOFF,
                        EngineeringStoryFieldName.OWNERSHIP,
                        EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
                    }
                    else StoryFieldDecisionReason.NO_DIRECT_EVIDENCE
                )
            ),
        ))
    story = EngineeringStory(
        story_id=f"engineering_story_candidate_{cluster.cluster_id.removeprefix('story_cluster_')}",
        project_id=project_id,
        story_type=story_type,
        **story_fields,
        evidence_fact_ids=cluster.member_evidence_fact_ids,
        capability_fact_ids=cluster.member_capability_ids,
        claim_boundary_ids=cluster.claim_boundary_ids,
        lifecycle=EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
        claim_sufficiency=ClaimSufficiency(SufficiencyLevel.UNASSESSED),
        story_sufficiency=StorySufficiency(SufficiencyLevel.UNASSESSED),
        opportunity=StoryOpportunity(StoryOpportunityLevel.NONE),
    )
    reconstruction = StoryReconstructionResult(
        cluster_id=cluster.cluster_id,
        project_id=project_id,
        engineering_story=story,
        reconstruction_quality=quality,
        identity_state=StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED,
        field_decisions=tuple(reversed(decisions)),
        diagnostics=(),
        unresolved_fields=tuple(
            item.field_name
            for item in decisions
            if item.resulting_state is StoryFieldEvidenceState.PLAUSIBLE_MISSING
        ),
    )
    memory = ProjectEvidenceMemory(
        project_id=project_id,
        project_name="WorkAgent",
        evidence_facts=list(facts),
        capability_facts=list(capabilities),
        claim_boundaries=list(boundaries),
    )
    sufficiency = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )
    return _Case(cluster, reconstruction, sufficiency, memory)


def _detect(case: _Case, *, context: StoryOpportunityProjectContext | None = None):
    return detect_story_opportunity(
        reconstruction_result=case.reconstruction,
        sufficiency_result=case.sufficiency,
        story_cluster=case.cluster,
        project_context=context,
    )


def _decision(
    result: StoryOpportunityDetectionResult,
    signal: StoryOpportunitySignal,
) -> StoryOpportunitySignalDecision:
    return next(item for item in result.signal_decisions if item.signal is signal)


def test_reuses_existing_story_opportunity_contract_without_extending_taxonomy() -> None:
    assert tuple(StoryOpportunityLevel) == (
        StoryOpportunityLevel.NONE,
        StoryOpportunityLevel.LOW,
        StoryOpportunityLevel.MEDIUM,
        StoryOpportunityLevel.HIGH,
    )
    assert tuple(StoryOpportunitySignal) == (
        StoryOpportunitySignal.ARCHITECTURE_MIGRATION,
        StoryOpportunitySignal.DEFENSIVE_ENGINEERING_CLUSTER,
        StoryOpportunitySignal.FAILURE_SPECIFIC_TEST_CLUSTER,
        StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING,
        StoryOpportunitySignal.MAJOR_DESIGN_DECISION,
        StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT,
    )


def test_strong_architecture_with_missing_decision_context_is_high_opportunity() -> None:
    case = _case(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))

    result = _detect(case)

    architecture = _decision(result, StoryOpportunitySignal.ARCHITECTURE_MIGRATION)
    assert result.story_opportunity.level is StoryOpportunityLevel.HIGH
    assert architecture.strength is StoryOpportunitySignalStrength.STRONG
    assert set(architecture.relevant_context_gaps) == {
        StoryContextGap.DECISION_REASON,
        StoryContextGap.TRADEOFF,
    }
    assert case.sufficiency.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert case.sufficiency.story_sufficiency.level is SufficiencyLevel.LOW


def test_fully_contextualized_architecture_has_lower_enrichment_opportunity() -> None:
    unresolved = _detect(_case(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    )))
    complete = _detect(_case(
        positive=_FIELD_ORDER,
        quality=StoryReconstructionQuality.COMPLETE,
    ))

    assert unresolved.story_opportunity.level is StoryOpportunityLevel.HIGH
    assert complete.story_opportunity.level is StoryOpportunityLevel.LOW
    assert complete.relevant_context_gaps == ()


def test_missing_fields_without_meaningful_event_are_not_opportunity() -> None:
    case = _case(
        evidence_types=(EvidenceType.INTEGRATION,),
        story_type=EngineeringStoryType.OTHER,
        positive=(EngineeringStoryFieldName.MECHANISM,),
    )

    result = _detect(case)

    assert result.story_opportunity == StoryOpportunity(StoryOpportunityLevel.NONE)
    assert result.signal_decisions == ()
    assert result.diagnostics == (
        StoryOpportunityDiagnosticCode.NO_MEANINGFUL_EVENT_SIGNAL,
    )


def test_weak_architecture_type_cannot_create_an_affirmative_signal() -> None:
    case = _case(
        evidence_types=(EvidenceType.INTEGRATION, EvidenceType.ARCHITECTURE),
        evidence_statuses=(EvidenceStatus.ACCEPTED, EvidenceStatus.WEAK),
        story_type=EngineeringStoryType.ARCHITECTURE_CHANGE,
    )

    result = _detect(case)

    assert StoryOpportunitySignal.ARCHITECTURE_MIGRATION not in (
        result.story_opportunity.signals
    )


def test_metric_or_outcome_gap_alone_does_not_create_opportunity() -> None:
    case = _case(
        evidence_types=(EvidenceType.OPTIMIZATION,),
        story_type=EngineeringStoryType.PERFORMANCE_OR_EFFICIENCY,
        positive=(
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
    )

    result = _detect(case)

    assert StoryContextGap.OBSERVABLE_OUTCOME_CONTEXT in case.sufficiency.story_context_gaps
    assert result.story_opportunity.level is StoryOpportunityLevel.NONE


def test_defensive_engineering_requires_structured_defensive_signal() -> None:
    plain = _case(
        evidence_types=(EvidenceType.TESTING,),
        story_type=EngineeringStoryType.RELIABILITY_HARDENING,
    )
    defensive = _case(
        evidence_types=(EvidenceType.FAILURE_RECOVERY, EvidenceType.VALIDATION),
        story_type=EngineeringStoryType.RELIABILITY_HARDENING,
        capability_type="failure_recovery",
    )

    plain_result = _detect(plain)
    result = _detect(defensive)

    assert StoryOpportunitySignal.DEFENSIVE_ENGINEERING_CLUSTER not in plain_result.story_opportunity.signals
    decision = _decision(result, StoryOpportunitySignal.DEFENSIVE_ENGINEERING_CLUSTER)
    assert decision.strength is StoryOpportunitySignalStrength.STRONG
    assert decision.capability_fact_ids == ("pcf_failure_recovery",)


def test_failure_specific_validation_requires_problem_and_failure_evidence() -> None:
    case = _case(
        evidence_types=(EvidenceType.BUG_FIX, EvidenceType.TESTING),
        story_type=EngineeringStoryType.DEBUGGING_AND_REPAIR,
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
    )

    result = _detect(case)
    decision = _decision(result, StoryOpportunitySignal.FAILURE_SPECIFIC_TEST_CLUSTER)

    assert decision.strength is StoryOpportunitySignalStrength.STRONG
    assert StoryContextGap.TRIGGER in decision.relevant_context_gaps


def test_major_design_decision_records_missing_rationale_without_inventing_it() -> None:
    case = _case()

    result = _detect(case)
    decision = _decision(result, StoryOpportunitySignal.MAJOR_DESIGN_DECISION)

    assert decision.relevant_context_gaps == (
        StoryContextGap.DECISION_REASON,
        StoryContextGap.TRADEOFF,
    )
    payload = result.to_json()
    assert _VALUES[EngineeringStoryFieldName.DECISION] not in payload
    assert "why" not in payload.casefold()


def test_human_context_signal_requires_a_meaningful_engineering_event() -> None:
    meaningful = _detect(_case())
    trivial = _detect(_case(
        evidence_types=(EvidenceType.DOCUMENTATION,),
        story_type=EngineeringStoryType.OTHER,
        positive=(EngineeringStoryFieldName.IMPLEMENTATION,),
    ))

    human = _decision(
        meaningful,
        StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT,
    )
    assert StoryContextGap.OWNERSHIP in human.relevant_context_gaps
    assert StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT not in trivial.story_opportunity.signals


def test_story_high_does_not_suppress_intrinsic_architecture_signal() -> None:
    case = _case(positive=_FIELD_ORDER, quality=StoryReconstructionQuality.COMPLETE)
    assert case.sufficiency.story_sufficiency.level is SufficiencyLevel.HIGH

    result = _detect(case)

    assert StoryOpportunitySignal.ARCHITECTURE_MIGRATION in result.story_opportunity.signals
    assert result.story_opportunity.level is StoryOpportunityLevel.LOW
    assert result.relevant_context_gaps == ()


def test_story_high_with_missing_ownership_retains_nonzero_human_opportunity() -> None:
    positive = tuple(
        name for name in _FIELD_ORDER if name is not EngineeringStoryFieldName.OWNERSHIP
    )
    case = _case(positive=positive, quality=StoryReconstructionQuality.COMPLETE)
    assert case.sufficiency.story_sufficiency.level is SufficiencyLevel.HIGH

    result = _detect(case)

    assert result.story_opportunity.level is StoryOpportunityLevel.MEDIUM
    assert StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT in (
        result.story_opportunity.signals
    )
    assert StoryContextGap.OWNERSHIP in result.relevant_context_gaps


def test_weak_claim_authority_caps_opportunity_at_low() -> None:
    case = _case(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))
    assert case.sufficiency.claim_sufficiency.level is SufficiencyLevel.HIGH
    low_sufficiency = replace(
        case.sufficiency,
        claim_sufficiency=ClaimSufficiency(
            level=SufficiencyLevel.LOW,
            supported_fields=case.sufficiency.claim_sufficiency.supported_fields,
            missing_fields=case.sufficiency.claim_sufficiency.missing_fields,
        ),
        evaluated_story=replace(
            case.sufficiency.evaluated_story,
            claim_sufficiency=ClaimSufficiency(
                level=SufficiencyLevel.LOW,
                supported_fields=case.sufficiency.claim_sufficiency.supported_fields,
                missing_fields=case.sufficiency.claim_sufficiency.missing_fields,
            ),
        ),
    )

    result = detect_story_opportunity(
        reconstruction_result=case.reconstruction,
        sufficiency_result=low_sufficiency,
        story_cluster=case.cluster,
    )

    assert result.story_opportunity.level is StoryOpportunityLevel.LOW
    assert StoryOpportunityDiagnosticCode.WEAK_CLAIM_AUTHORITY_LIMIT in result.diagnostics


@pytest.mark.parametrize(
    ("quality", "expected_level", "diagnostic"),
    (
        (
            StoryReconstructionQuality.AMBIGUOUS,
            StoryOpportunityLevel.LOW,
            StoryOpportunityDiagnosticCode.AMBIGUOUS_RECONSTRUCTION_LIMIT,
        ),
        (
            StoryReconstructionQuality.MINIMAL,
            StoryOpportunityLevel.MEDIUM,
            StoryOpportunityDiagnosticCode.MINIMAL_RECONSTRUCTION_LIMIT,
        ),
    ),
)
def test_reconstruction_quality_limits_opportunity(
    quality: StoryReconstructionQuality,
    expected_level: StoryOpportunityLevel,
    diagnostic: StoryOpportunityDiagnosticCode,
) -> None:
    case = _case(
        quality=quality,
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
    )

    result = _detect(case)

    assert result.story_opportunity.level is expected_level
    assert diagnostic in result.diagnostics


def test_blocked_reconstruction_never_produces_opportunity_from_cluster_signals() -> None:
    case = _case()
    decisions = tuple(
        StoryFieldReconstructionDecision(
            field_name=name,
            resulting_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
            evidence_fact_ids=(),
            capability_fact_ids=(),
            claim_boundary_ids=(),
            reason_code=StoryFieldDecisionReason.NO_DIRECT_EVIDENCE,
        )
        for name in _FIELD_ORDER
    )
    blocked_reconstruction = StoryReconstructionResult(
        cluster_id=case.cluster.cluster_id,
        project_id=PROJECT_ID,
        engineering_story=None,
        reconstruction_quality=StoryReconstructionQuality.BLOCKED,
        identity_state=StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED,
        field_decisions=decisions,
        diagnostics=(),
        unresolved_fields=_FIELD_ORDER,
    )
    blocked_sufficiency = evaluate_engineering_story_sufficiency(
        reconstruction_result=blocked_reconstruction,
        project_memory=case.memory,
    )

    result = detect_story_opportunity(
        reconstruction_result=blocked_reconstruction,
        sufficiency_result=blocked_sufficiency,
        story_cluster=case.cluster,
    )

    assert result.story_opportunity.level is StoryOpportunityLevel.NONE
    assert result.signal_decisions == ()
    assert result.diagnostics == (
        StoryOpportunityDiagnosticCode.BLOCKED_RECONSTRUCTION,
    )


def test_repeated_subsystem_hardening_uses_path_lineage_without_merging_clusters() -> None:
    current = _case(
        evidence_types=(EvidenceType.ARCHITECTURE,),
        change_id="change-architecture",
        file_path="backend/shared.py",
    )
    validation_cluster, *_ = _cluster(
        (EvidenceType.VALIDATION,),
        change_id="change-validation",
        file_path="backend/shared.py",
    )
    failure_cluster, *_ = _cluster(
        (EvidenceType.FAILURE_RECOVERY,),
        change_id="change-failure",
        file_path="backend/shared.py",
    )
    context = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(failure_cluster, current.cluster, validation_cluster),
    )

    result = _detect(current, context=context)
    decision = _decision(result, StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING)

    assert decision.strength is StoryOpportunitySignalStrength.STRONG
    assert decision.related_cluster_ids == tuple(sorted((
        validation_cluster.cluster_id,
        failure_cluster.cluster_id,
    )))
    assert result.cluster_id == current.cluster.cluster_id
    assert len(context.clusters) == 3


def test_same_capability_or_technology_without_structural_lineage_is_not_repeated() -> None:
    current = _case(
        evidence_types=(EvidenceType.FAILURE_RECOVERY,),
        story_type=EngineeringStoryType.RELIABILITY_HARDENING,
        change_id="change-one",
        file_path="backend/one.py",
        symbol="recover_one",
        capability_type="failure_recovery",
    )
    other, *_ = _cluster(
        (EvidenceType.FAILURE_RECOVERY,),
        change_id="change-two",
        file_path="backend/two.py",
        symbol="recover_two",
        capability_type="failure_recovery",
    )
    context = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(current.cluster, other),
    )

    result = _detect(current, context=context)

    assert StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING not in result.story_opportunity.signals


def test_project_context_is_deterministic_bounded_and_path_free() -> None:
    first, *_ = _cluster(
        (EvidenceType.ARCHITECTURE,),
        change_id="change-one",
        file_path="backend/private_component.py",
    )
    second, *_ = _cluster(
        (EvidenceType.VALIDATION,),
        change_id="change-two",
        file_path="backend/private_component.py",
    )

    left = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(first, second),
    )
    right = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(second, first),
    )

    assert left == right
    assert left.to_json() == right.to_json()
    assert "private_component.py" not in left.to_json()
    assert "file_path" not in left.to_json()
    assert all(
        key.startswith("subsystem_")
        for item in left.clusters
        for key in item.structural_subsystem_keys
    )


def test_project_context_order_does_not_change_detection_output() -> None:
    current = _case(
        change_id="change-current",
        file_path="backend/shared.py",
    )
    other, *_ = _cluster(
        (EvidenceType.FAILURE_RECOVERY,),
        change_id="change-other",
        file_path="backend/shared.py",
    )
    first_context = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(current.cluster, other),
    )
    second_context = build_story_opportunity_project_context(
        project_id=PROJECT_ID,
        clusters=(other, current.cluster),
    )

    first = _detect(current, context=first_context)
    second = _detect(current, context=second_context)

    assert first == second
    assert first.to_json() == second.to_json()


def test_cross_project_project_context_fails_closed_before_detection() -> None:
    local = _case()
    foreign, *_ = _cluster(
        (EvidenceType.VALIDATION,),
        project_id=OTHER_PROJECT_ID,
        change_id="change-foreign",
    )

    with pytest.raises(StoryOpportunityDetectionError) as exc_info:
        build_story_opportunity_project_context(
            project_id=PROJECT_ID,
            clusters=(local.cluster, foreign),
        )

    assert exc_info.value.code is StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT


def test_foreign_context_is_rejected_even_when_manually_constructed() -> None:
    case = _case()
    foreign_context = StoryOpportunityProjectContext(
        project_id=OTHER_PROJECT_ID,
        clusters=(),
    )

    with pytest.raises(StoryOpportunityDetectionError) as exc_info:
        _detect(case, context=foreign_context)

    assert exc_info.value.code is StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT


def test_mismatched_cluster_or_sufficiency_fails_closed() -> None:
    first = _case()
    second = _case(change_id="change-second", file_path="backend/second.py")

    with pytest.raises(StoryOpportunityDetectionError) as exc_info:
        detect_story_opportunity(
            reconstruction_result=first.reconstruction,
            sufficiency_result=first.sufficiency,
            story_cluster=second.cluster,
        )

    assert exc_info.value.code is StoryOpportunityDetectionErrorCode.CLUSTER_MISMATCH


def test_claim_boundaries_are_restrictions_not_opportunity_evidence() -> None:
    case = _case(
        evidence_types=(EvidenceType.INTEGRATION,),
        story_type=EngineeringStoryType.OTHER,
        positive=(EngineeringStoryFieldName.MECHANISM,),
    )
    assert case.cluster.claim_boundary_ids

    result = _detect(case)

    assert result.story_opportunity.level is StoryOpportunityLevel.NONE
    assert result.signal_decisions == ()


def test_detection_is_pure_immutable_and_round_trips() -> None:
    case = _case()
    reconstruction_snapshot = case.reconstruction.to_json()
    sufficiency_snapshot = case.sufficiency.to_json()
    cluster_snapshot = case.cluster.to_json()

    result = _detect(case)
    restored = StoryOpportunityDetectionResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.to_json() == result.to_json()
    assert case.reconstruction.to_json() == reconstruction_snapshot
    assert case.sufficiency.to_json() == sufficiency_snapshot
    assert case.cluster.to_json() == cluster_snapshot
    assert case.sufficiency.evaluated_story.opportunity.level is StoryOpportunityLevel.NONE
    assert result.evaluated_story.opportunity == result.story_opportunity
    with pytest.raises(FrozenInstanceError):
        result.project_id = OTHER_PROJECT_ID  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_unknown_nested_fields_and_inconsistent_decisions_are_rejected() -> None:
    result = _detect(_case())
    payload = result.to_dict()
    payload["unknown"] = "forbidden"
    with pytest.raises(ValueError):
        StoryOpportunityDetectionResult.from_dict(payload)

    payload = result.to_dict()
    payload["signal_decisions"][0]["unknown"] = "forbidden"
    with pytest.raises(ValueError):
        StoryOpportunityDetectionResult.from_dict(payload)


def test_serialized_contract_contains_no_action_or_raw_evidence_fields() -> None:
    result = _detect(_case())
    serialized = result.to_json().casefold()

    forbidden = (
        "question",
        "ask_value",
        "friction",
        "query",
        "retrieval",
        "resume",
        "raw_text",
        "raw_patch",
        "document",
        "file_path",
        "symbol",
        "repository",
        "job_description",
        "hiring_context",
    )
    assert all(term not in serialized for term in forbidden)


def test_detector_accepts_zero_capability_facts() -> None:
    case = _case(capability_type=None)

    result = _detect(case)

    assert result.story_opportunity.level is StoryOpportunityLevel.HIGH
    assert result.evaluated_story.capability_fact_ids == ()


def test_entrypoint_has_no_jd_action_or_retrieval_inputs() -> None:
    parameters = inspect.signature(detect_story_opportunity).parameters

    assert tuple(parameters) == (
        "reconstruction_result",
        "sufficiency_result",
        "story_cluster",
        "project_context",
    )
    assert all(
        token not in name
        for name in parameters
        for token in ("jd", "role", "company", "query", "question", "retrieval")
    )


def test_module_has_no_runtime_retrieval_persistence_or_api_dependencies() -> None:
    module_path = Path(__file__).parents[1] / "backend" / "engineering_story_opportunity.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    forbidden_imports = (
        "api_server",
        "chroma",
        "memory_store",
        "project_retrieval",
        "query_planner",
        "coverage",
        "github_raw_storage",
        "project_capability_memory",
        "os",
        "requests",
        "httpx",
    )
    assert not any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imports
        for forbidden in forbidden_imports
    )
    assert not ({"open", "getenv", "environ", "PersistentClient"} & calls)


def test_only_expected_backend_and_test_files_are_unstaged_for_this_work_item() -> None:
    root = Path(__file__).parents[1]
    assert (root / "backend" / "engineering_story_opportunity.py").is_file()
    assert (root / "tests" / "test_engineering_story_opportunity.py").is_file()
