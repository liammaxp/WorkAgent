from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from hashlib import sha256
import inspect
from pathlib import Path

import pytest

from backend.engineering_story_clustering import cluster_story_evidence_bundle
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
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_reconstruction import (
    StoryFieldDecisionReason,
    StoryFieldReconstructionDecision,
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
    StoryReconstructionResult,
    reconstruct_engineering_story,
)
from backend.engineering_story_sufficiency import (
    EngineeringStorySufficiencyResult,
    StorySufficiencyEvaluationError,
    StorySufficiencyEvaluationErrorCode,
    SufficiencyDecisionReason,
    SufficiencyDiagnosticCode,
    SufficiencyDimension,
    SufficiencyDimensionDecision,
    SufficiencyDimensionDomain,
    SufficiencyDimensionStatus,
    evaluate_engineering_story_sufficiency,
)
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


PROJECT_ID = "workagent"
CLUSTER_ID = "story_cluster_aaaaaaaaaaaaaaaaaaaaaaaa"
STORY_ID = "engineering_story_candidate_aaaaaaaaaaaaaaaaaaaaaaaa"
_FIELD_ORDER = tuple(EngineeringStoryFieldName)

_DEFAULT_VALUES = {
    EngineeringStoryFieldName.PROBLEM_CONTEXT: "Cross-project evidence could be mixed",
    EngineeringStoryFieldName.TRIGGER: "A cross-project regression exposed the risk",
    EngineeringStoryFieldName.BEFORE_STATE: "Evidence selection trusted an unscoped path",
    EngineeringStoryFieldName.DECISION: "Use exact project authority as the trust boundary",
    EngineeringStoryFieldName.MECHANISM: "Project-scoped authority isolation",
    EngineeringStoryFieldName.IMPLEMENTATION: "Validated exact project IDs before selection",
    EngineeringStoryFieldName.TRADEOFF: "Rejected ambiguous evidence instead of guessing",
    EngineeringStoryFieldName.VALIDATION: "Cross-project regression tests reject invalid paths",
    EngineeringStoryFieldName.AFTER_STATE: "Foreign project evidence is rejected",
    EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "Invalid cross-project paths are rejected",
    EngineeringStoryFieldName.OWNERSHIP: "Owned the isolation design and regression validation",
    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT: "Protected downstream resume evidence consumers",
}


def _source_ref(evidence_id: str, project_id: str = PROJECT_ID) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type="github_evidence_chunk",
        source_id=f"chunk_{evidence_id}",
        project_id=project_id,
        content_hash=sha256(f"{project_id}|{evidence_id}".encode()).hexdigest(),
        repo="owner/workagent",
        commit_sha="aaaaaaa",
        file_path="backend/isolation.py",
        symbol="validate_project",
        metadata={"change_id": "change_isolation"},
    )


def _fact(
    evidence_id: str,
    *,
    project_id: str = PROJECT_ID,
    safe_impact: tuple[str, ...] = (),
    metric_support: MetricSupport = MetricSupport.NONE,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=evidence_id,
        problem="Cross-project evidence could be mixed",
        mechanism="Project-scoped authority isolation",
        implementation=["Validated exact project IDs before selection"],
        safe_impact=list(safe_impact),
        source_refs=[_source_ref(evidence_id, project_id)],
        evidence_type=EvidenceType.VALIDATION,
        status=EvidenceStatus.ACCEPTED,
        confidence=Confidence.HIGH,
        metric_support=metric_support,
        technical_tags=["validation", "isolation"],
        quality_score=90,
    )


def _case(
    *,
    positive: tuple[EngineeringStoryFieldName, ...],
    supported: tuple[EngineeringStoryFieldName, ...] = (),
    unsupported_reasons: dict[
        EngineeringStoryFieldName, StoryFieldDecisionReason
    ] | None = None,
    values: dict[EngineeringStoryFieldName, str] | None = None,
    quality: StoryReconstructionQuality = StoryReconstructionQuality.PARTIAL,
    lifecycle: EngineeringStoryLifecycle | None = None,
    metric_support: MetricSupport = MetricSupport.NONE,
) -> tuple[StoryReconstructionResult, ProjectEvidenceMemory]:
    unsupported_reasons = {} if unsupported_reasons is None else unsupported_reasons
    values = {} if values is None else values
    positive_set = set(positive)
    supported_set = set(supported)
    facts: dict[str, ProjectEvidenceFact] = {}
    boundaries: dict[str, ProjectClaimBoundary] = {}
    story_fields: dict[str, EngineeringStoryField] = {}
    decisions: list[StoryFieldReconstructionDecision] = []
    for field_name in _FIELD_ORDER:
        reason_override = unsupported_reasons.get(field_name)
        has_authority = field_name in positive_set or reason_override is not None
        evidence_id = f"pef_{field_name.value}"
        field_value = values.get(field_name, _DEFAULT_VALUES[field_name])
        boundary_ids: tuple[str, ...] = ()
        if has_authority:
            safe_impact = (
                (field_value,)
                if field_name is EngineeringStoryFieldName.OBSERVABLE_OUTCOME
                and (isinstance(field_value, str) and any(ch.isdigit() for ch in field_value))
                else ()
            )
            fact = _fact(
                evidence_id,
                safe_impact=safe_impact,
                metric_support=(
                    metric_support
                    if field_name is EngineeringStoryFieldName.OBSERVABLE_OUTCOME
                    else MetricSupport.NONE
                ),
            )
            facts[evidence_id] = fact
            if (
                field_name is EngineeringStoryFieldName.OBSERVABLE_OUTCOME
                and field_name in positive_set
                and safe_impact
            ):
                boundary = build_project_evidence_claim_boundary(fact)
                assert boundary is not None
                boundaries[boundary.boundary_id] = boundary
                boundary_ids = (boundary.boundary_id,)
            elif reason_override is StoryFieldDecisionReason.BOUNDARY_RESTRICTED:
                prefix = (
                    "impact"
                    if field_name is EngineeringStoryFieldName.OBSERVABLE_OUTCOME
                    else field_name.value
                )
                boundary = ProjectClaimBoundary(
                    project_id=PROJECT_ID,
                    subject_type=ClaimSubjectType.PROJECT,
                    subject_id=PROJECT_ID,
                    forbidden_claims=[f"{prefix}:{field_value}"],
                    metric_support=MetricSupport.NONE,
                    boundary_id=f"pcb_{field_name.value}_restricted",
                )
                boundaries[boundary.boundary_id] = boundary
                boundary_ids = (boundary.boundary_id,)
        if field_name in positive_set:
            state = (
                StoryFieldEvidenceState.SUPPORTED
                if field_name in supported_set
                else StoryFieldEvidenceState.CONFIRMED
            )
            value = field_value
            reason = (
                StoryFieldDecisionReason.MULTI_SOURCE_SUPPORT
                if state is StoryFieldEvidenceState.SUPPORTED
                else StoryFieldDecisionReason.DIRECT_AUTHORITATIVE_EVIDENCE
            )
        elif reason_override is not None:
            state = StoryFieldEvidenceState.UNSUPPORTED
            value = None
            reason = reason_override
        else:
            state = StoryFieldEvidenceState.PLAUSIBLE_MISSING
            value = None
            reason = (
                StoryFieldDecisionReason.MISSING_HUMAN_CONTEXT
                if field_name in {
                    EngineeringStoryFieldName.TRIGGER,
                    EngineeringStoryFieldName.DECISION,
                    EngineeringStoryFieldName.TRADEOFF,
                    EngineeringStoryFieldName.OWNERSHIP,
                    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
                }
                else StoryFieldDecisionReason.NO_DIRECT_EVIDENCE
            )
        field = EngineeringStoryField(
            value=value,
            evidence_state=state,
            evidence_fact_ids=(evidence_id,) if has_authority else (),
            claim_boundary_ids=boundary_ids,
        )
        story_fields[field_name.value] = field
        decisions.append(StoryFieldReconstructionDecision(
            field_name=field_name,
            resulting_state=state,
            evidence_fact_ids=field.evidence_fact_ids,
            capability_fact_ids=field.capability_fact_ids,
            claim_boundary_ids=field.claim_boundary_ids,
            reason_code=reason,
        ))
    story = EngineeringStory(
        story_id=STORY_ID,
        project_id=PROJECT_ID,
        story_type=EngineeringStoryType.RELIABILITY_HARDENING,
        **story_fields,
        evidence_fact_ids=tuple(facts),
        capability_fact_ids=(),
        claim_boundary_ids=tuple(boundaries),
        lifecycle=lifecycle or EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
        claim_sufficiency=ClaimSufficiency(SufficiencyLevel.UNASSESSED),
        story_sufficiency=StorySufficiency(SufficiencyLevel.UNASSESSED),
        opportunity=StoryOpportunity(StoryOpportunityLevel.NONE),
    )
    result = StoryReconstructionResult(
        cluster_id=CLUSTER_ID,
        project_id=PROJECT_ID,
        engineering_story=story,
        reconstruction_quality=quality,
        identity_state=StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED,
        field_decisions=tuple(reversed(decisions)),
        diagnostics=(),
        unresolved_fields=tuple(
            decision.field_name
            for decision in decisions
            if decision.resulting_state in {
                StoryFieldEvidenceState.PLAUSIBLE_MISSING,
                StoryFieldEvidenceState.UNSUPPORTED,
            }
        ),
    )
    memory = ProjectEvidenceMemory(
        project_id=PROJECT_ID,
        project_name="WorkAgent",
        evidence_facts=list(reversed(tuple(facts.values()))),
        capability_facts=[],
        claim_boundaries=list(reversed(tuple(boundaries.values()))),
    )
    return result, memory


def _evaluate(**kwargs):
    reconstruction, memory = _case(**kwargs)
    return evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )


def _claim_decision(result, dimension: SufficiencyDimension):
    return next(
        item for item in result.claim_dimension_decisions if item.dimension is dimension
    )


def _story_decision(result, dimension: SufficiencyDimension):
    return next(
        item for item in result.story_dimension_decisions if item.dimension is dimension
    )


def test_strong_technical_claim_can_have_weak_story() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))

    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert result.story_sufficiency.level is SufficiencyLevel.LOW
    assert SufficiencyDiagnosticCode.CLAIM_STORY_LEVEL_DIVERGENCE in result.diagnostics


def test_strong_claim_and_strong_causal_story_are_independently_high() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))

    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert result.story_sufficiency.level is SufficiencyLevel.HIGH
    assert _claim_decision(
        result, SufficiencyDimension.METRIC_SUPPORT
    ).status is SufficiencyDimensionStatus.NOT_APPLICABLE


def test_no_metric_does_not_penalize_safe_non_numeric_story() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))

    metric = _claim_decision(result, SufficiencyDimension.METRIC_SUPPORT)
    assert metric.metric_support is MetricSupport.NONE
    assert metric.reason_code is SufficiencyDecisionReason.NON_NUMERIC_CLAIMS_AVAILABLE
    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert result.story_sufficiency.level is SufficiencyLevel.HIGH


def test_unsupported_metric_restricts_only_metric_dimension() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
        unsupported_reasons={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME:
                StoryFieldDecisionReason.UNSUPPORTED_METRIC,
        },
        values={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "Reduced latency by 20%",
        },
    )

    metric = _claim_decision(result, SufficiencyDimension.METRIC_SUPPORT)
    assert metric.status is SufficiencyDimensionStatus.RESTRICTED
    assert metric.metric_support is MetricSupport.NONE
    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert SufficiencyDiagnosticCode.METRIC_UNSUPPORTED in result.diagnostics


def test_approximate_metric_semantics_and_wording_are_preserved() -> None:
    value = "Reduced startup latency by approximately 20%"
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        ),
        values={EngineeringStoryFieldName.OBSERVABLE_OUTCOME: value},
        metric_support=MetricSupport.APPROXIMATE,
    )

    metric = _claim_decision(result, SufficiencyDimension.METRIC_SUPPORT)
    assert metric.status is SufficiencyDimensionStatus.PARTIAL
    assert metric.metric_support is MetricSupport.APPROXIMATE
    assert metric.reason_code is SufficiencyDecisionReason.APPROXIMATE_METRIC_SUPPORT
    assert result.evaluated_story is not None
    assert result.evaluated_story.observable_outcome.value == value


def test_explicit_metric_support_is_kept_explicit() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        ),
        values={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "Reduced startup latency by 20%",
        },
        metric_support=MetricSupport.EXPLICIT,
    )

    metric = _claim_decision(result, SufficiencyDimension.METRIC_SUPPORT)
    assert metric.status is SufficiencyDimensionStatus.SATISFIED
    assert metric.metric_support is MetricSupport.EXPLICIT


def test_missing_ownership_does_not_destroy_technical_claim_and_remains_gap() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))

    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert StoryContextGap.OWNERSHIP in result.story_context_gaps
    assert (
        EngineeringStoryFieldName.OWNERSHIP
        in result.story_sufficiency.missing_fields
    )


def test_missing_decision_and_tradeoff_are_exposed_without_invented_context() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))

    assert StoryContextGap.DECISION_REASON in result.story_context_gaps
    assert StoryContextGap.TRADEOFF in result.story_context_gaps
    assert result.evaluated_story is not None
    assert result.evaluated_story.decision.value is None
    assert result.evaluated_story.tradeoff.value is None


def test_mechanism_only_story_has_narrow_claim_and_low_story_sufficiency() -> None:
    result = _evaluate(positive=(EngineeringStoryFieldName.MECHANISM,))

    assert result.claim_sufficiency.level is SufficiencyLevel.MEDIUM
    assert result.story_sufficiency.level is SufficiencyLevel.LOW
    assert _claim_decision(
        result, SufficiencyDimension.TECHNICAL_CORE
    ).status is SufficiencyDimensionStatus.PARTIAL


def test_validation_without_business_metric_is_meaningful() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))

    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert result.story_sufficiency.level is SufficiencyLevel.MEDIUM
    assert _claim_decision(
        result, SufficiencyDimension.METRIC_SUPPORT
    ).status is SufficiencyDimensionStatus.NOT_APPLICABLE


def test_boundary_restricted_outcome_does_not_block_mechanism_claim() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
        unsupported_reasons={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME:
                StoryFieldDecisionReason.BOUNDARY_RESTRICTED,
        },
    )

    assert result.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert _claim_decision(
        result, SufficiencyDimension.OUTCOME_SUPPORT
    ).status is SufficiencyDimensionStatus.RESTRICTED
    assert _claim_decision(
        result, SufficiencyDimension.MECHANISM_SUPPORT
    ).status is SufficiencyDimensionStatus.SATISFIED
    assert SufficiencyDiagnosticCode.CLAIM_BOUNDARY_RESTRICTION in result.diagnostics


def test_conflicting_mechanism_fails_closed_while_story_remains_partial() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.DECISION,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.TRADEOFF,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
            EngineeringStoryFieldName.OWNERSHIP,
            EngineeringStoryFieldName.STAKEHOLDER_CONTEXT,
        ),
        unsupported_reasons={
            EngineeringStoryFieldName.MECHANISM:
                StoryFieldDecisionReason.CONFLICTING_EVIDENCE,
        },
    )

    assert result.claim_sufficiency.level is SufficiencyLevel.LOW
    assert result.story_sufficiency.level is SufficiencyLevel.MEDIUM
    assert _claim_decision(
        result, SufficiencyDimension.MECHANISM_SUPPORT
    ).status is SufficiencyDimensionStatus.BLOCKED
    assert _story_decision(
        result, SufficiencyDimension.ENGINEERING_ACTION
    ).status is SufficiencyDimensionStatus.BLOCKED
    assert SufficiencyDiagnosticCode.FIELD_CONFLICT in result.diagnostics


def test_ambiguous_reconstruction_cannot_become_globally_high() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        ),
        quality=StoryReconstructionQuality.AMBIGUOUS,
    )

    assert result.claim_sufficiency.level is SufficiencyLevel.MEDIUM
    assert result.story_sufficiency.level is SufficiencyLevel.MEDIUM
    assert SufficiencyDiagnosticCode.RECONSTRUCTION_AMBIGUOUS in result.diagnostics


def test_blocked_reconstruction_is_low_and_has_no_story() -> None:
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
    reconstruction = StoryReconstructionResult(
        cluster_id=CLUSTER_ID,
        project_id=PROJECT_ID,
        engineering_story=None,
        reconstruction_quality=StoryReconstructionQuality.BLOCKED,
        identity_state=StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED,
        field_decisions=decisions,
        diagnostics=(),
        unresolved_fields=_FIELD_ORDER,
    )
    memory = ProjectEvidenceMemory(project_id=PROJECT_ID, project_name="WorkAgent")

    result = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )

    assert result.evaluated_story is None
    assert result.story_id is None
    assert result.claim_sufficiency.level is SufficiencyLevel.LOW
    assert result.story_sufficiency.level is SufficiencyLevel.LOW
    assert all(
        item.status is SufficiencyDimensionStatus.BLOCKED
        for item in result.claim_dimension_decisions + result.story_dimension_decisions
    )


def test_minimal_reconstruction_keeps_narrow_claim_but_story_low() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
        ),
        quality=StoryReconstructionQuality.MINIMAL,
    )

    assert result.claim_sufficiency.level is SufficiencyLevel.MEDIUM
    assert result.story_sufficiency.level is SufficiencyLevel.LOW
    assert SufficiencyDiagnosticCode.RECONSTRUCTION_MINIMAL in result.diagnostics


@pytest.mark.parametrize(
    ("state_reason", "expected_status"),
    [
        (
            StoryFieldDecisionReason.WEAK_AUTHORITY,
            SufficiencyDimensionStatus.RESTRICTED,
        ),
        (
            StoryFieldDecisionReason.NO_DIRECT_EVIDENCE,
            SufficiencyDimensionStatus.MISSING,
        ),
    ],
)
def test_non_positive_fields_contribute_zero_support(
    state_reason: StoryFieldDecisionReason,
    expected_status: SufficiencyDimensionStatus,
) -> None:
    unsupported = (
        {EngineeringStoryFieldName.VALIDATION: state_reason}
        if state_reason is StoryFieldDecisionReason.WEAK_AUTHORITY
        else {}
    )
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
        ),
        unsupported_reasons=unsupported,
    )

    validation = _claim_decision(result, SufficiencyDimension.VALIDATION_SUPPORT)
    assert validation.status is expected_status
    assert EngineeringStoryFieldName.VALIDATION not in validation.supporting_story_fields


def test_supported_and_confirmed_states_remain_distinguishable() -> None:
    confirmed = _evaluate(positive=(EngineeringStoryFieldName.MECHANISM,))
    supported = _evaluate(
        positive=(EngineeringStoryFieldName.MECHANISM,),
        supported=(EngineeringStoryFieldName.MECHANISM,),
    )

    assert _claim_decision(
        confirmed, SufficiencyDimension.MECHANISM_SUPPORT
    ).reason_code is SufficiencyDecisionReason.DIRECT_CONFIRMED_SUPPORT
    assert _claim_decision(
        supported, SufficiencyDimension.MECHANISM_SUPPORT
    ).reason_code is SufficiencyDecisionReason.SUPPORTED_AUTHORITY


def test_core_story_can_be_high_while_optional_human_gaps_remain() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))

    assert result.story_sufficiency.level is SufficiencyLevel.HIGH
    assert StoryContextGap.OWNERSHIP in result.story_context_gaps
    assert StoryContextGap.STAKEHOLDER_CONTEXT in result.story_context_gaps
    assert StoryContextGap.DECISION_REASON in result.story_context_gaps


def test_generic_safe_impact_does_not_become_observed_outcome() -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
        unsupported_reasons={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME:
                StoryFieldDecisionReason.NON_OBSERVABLE_SAFE_IMPACT,
        },
        values={EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "Improved reliability"},
    )

    assert EngineeringStoryFieldName.OBSERVABLE_OUTCOME not in (
        result.claim_sufficiency.supported_fields
    )
    assert StoryContextGap.OBSERVABLE_OUTCOME_CONTEXT in result.story_context_gaps


@pytest.mark.parametrize(
    "lifecycle",
    [
        EngineeringStoryLifecycle(
            EngineeringStoryStatus.ACTIVE,
            requires_revalidation=True,
        ),
        EngineeringStoryLifecycle(
            EngineeringStoryStatus.STALE,
            requires_revalidation=True,
        ),
        EngineeringStoryLifecycle(
            EngineeringStoryStatus.CONFLICTED,
            requires_revalidation=True,
        ),
        EngineeringStoryLifecycle(
            EngineeringStoryStatus.SUPERSEDED,
            superseded_by_story_id="engineering_story_successor",
        ),
    ],
)
def test_lifecycle_prevents_fully_claim_ready_result(
    lifecycle: EngineeringStoryLifecycle,
) -> None:
    result = _evaluate(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        ),
        lifecycle=lifecycle,
    )

    assert result.claim_sufficiency.level is not SufficiencyLevel.HIGH
    assert SufficiencyDiagnosticCode.LIFECYCLE_REVALIDATION_REQUIRED in (
        result.diagnostics
    ) or SufficiencyDiagnosticCode.LIFECYCLE_INACTIVE in result.diagnostics


def test_zero_capability_facts_remains_valid() -> None:
    reconstruction, memory = _case(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    ))
    assert memory.capability_facts == []

    result = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )

    assert result.claim_sufficiency.level is SufficiencyLevel.MEDIUM


def test_accepted_reconstruction_output_flows_into_sufficiency_evaluator() -> None:
    fact = _fact("pef_actual_reconstruction")
    boundary = build_project_evidence_claim_boundary(fact)
    assert boundary is not None
    bundle = resolve_story_evidence_bundle(
        project_id=PROJECT_ID,
        evidence_fact_ids=(fact.evidence_fact_id,),
        evidence_facts=(fact,),
        claim_boundary_ids=(boundary.boundary_id,),
        claim_boundaries=(boundary,),
    )
    clustering = cluster_story_evidence_bundle(bundle)
    assert len(clustering.clusters) == 1
    reconstruction = reconstruct_engineering_story(
        cluster=clustering.clusters[0],
        evidence_facts=(fact,),
        claim_boundaries=(boundary,),
    )
    memory = ProjectEvidenceMemory(
        project_id=PROJECT_ID,
        project_name="WorkAgent",
        evidence_facts=[fact],
        claim_boundaries=[boundary],
    )

    result = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )

    assert result.evaluated_story is not None
    assert result.story_id == reconstruction.engineering_story.story_id
    assert result.claim_sufficiency.level in {
        SufficiencyLevel.MEDIUM,
        SufficiencyLevel.HIGH,
    }
    assert result.story_sufficiency.level in {
        SufficiencyLevel.LOW,
        SufficiencyLevel.MEDIUM,
    }


def test_foreign_or_missing_authority_fails_closed() -> None:
    reconstruction, _memory = _case(positive=(EngineeringStoryFieldName.MECHANISM,))
    foreign = ProjectEvidenceMemory(
        project_id="other-project",
        project_name="Other",
    )
    with pytest.raises(StorySufficiencyEvaluationError) as foreign_error:
        evaluate_engineering_story_sufficiency(
            reconstruction_result=reconstruction,
            project_memory=foreign,
        )
    assert foreign_error.value.code is (
        StorySufficiencyEvaluationErrorCode.CROSS_PROJECT_AUTHORITY
    )

    missing = ProjectEvidenceMemory(project_id=PROJECT_ID, project_name="WorkAgent")
    with pytest.raises(StorySufficiencyEvaluationError) as missing_error:
        evaluate_engineering_story_sufficiency(
            reconstruction_result=reconstruction,
            project_memory=missing,
        )
    assert missing_error.value.code is StorySufficiencyEvaluationErrorCode.MISSING_AUTHORITY


def test_invalid_referenced_boundary_fails_closed() -> None:
    reconstruction, memory = _case(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
        ),
        values={
            EngineeringStoryFieldName.OBSERVABLE_OUTCOME: "Reduced startup latency by 20%",
        },
        metric_support=MetricSupport.EXPLICIT,
    )
    assert memory.claim_boundaries
    memory.claim_boundaries[0].allowed_claims.append("metric:unsupported 99%")

    with pytest.raises(StorySufficiencyEvaluationError) as caught:
        evaluate_engineering_story_sufficiency(
            reconstruction_result=reconstruction,
            project_memory=memory,
        )
    assert caught.value.code is StorySufficiencyEvaluationErrorCode.INVALID_CLAIM_BOUNDARY


def test_evaluation_is_deterministic_and_does_not_mutate_inputs() -> None:
    reconstruction, memory = _case(
        positive=(
            EngineeringStoryFieldName.PROBLEM_CONTEXT,
            EngineeringStoryFieldName.MECHANISM,
            EngineeringStoryFieldName.IMPLEMENTATION,
            EngineeringStoryFieldName.VALIDATION,
        ),
        supported=(EngineeringStoryFieldName.VALIDATION,),
    )
    reconstruction_before = reconstruction.to_json()
    memory_before = memory.to_json()

    first = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )
    memory.evidence_facts.reverse()
    second = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=memory,
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert reconstruction.to_json() == reconstruction_before
    memory.evidence_facts.reverse()
    assert memory.to_json() == memory_before


def test_result_is_immutable_strict_and_round_trips() -> None:
    result = _evaluate(positive=(
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))

    assert EngineeringStorySufficiencyResult.from_dict(result.to_dict()) == result
    with pytest.raises(FrozenInstanceError):
        result.story_id = "engineering_story_changed"
    with pytest.raises(ValueError, match="unknown"):
        EngineeringStorySufficiencyResult.from_dict({
            **result.to_dict(),
            "raw_query": "find evidence",
        })
    assert not hasattr(result, "__dict__")
    assert not hasattr(result.claim_dimension_decisions[0], "__dict__")


def test_dimension_contract_rejects_domain_mismatch_and_missing_metric_support() -> None:
    with pytest.raises(ValueError, match="domain"):
        SufficiencyDimensionDecision(
            domain=SufficiencyDimensionDomain.STORY,
            dimension=SufficiencyDimension.TECHNICAL_CORE,
            status=SufficiencyDimensionStatus.MISSING,
        )
    with pytest.raises(ValueError, match="MetricSupport"):
        SufficiencyDimensionDecision(
            domain=SufficiencyDimensionDomain.CLAIM,
            dimension=SufficiencyDimension.METRIC_SUPPORT,
            status=SufficiencyDimensionStatus.NOT_APPLICABLE,
        )


def test_evaluator_has_no_hiring_context_or_runtime_parameters() -> None:
    assert tuple(inspect.signature(evaluate_engineering_story_sufficiency).parameters) == (
        "reconstruction_result",
        "project_memory",
    )
    result = _evaluate(positive=(
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    ))
    serialized = result.to_json().casefold()
    for forbidden in (
        "jd_target",
        "job_description",
        "company",
        "resume_score",
        "ats",
        "question",
        "search_query",
        "raw_patch",
        "document",
        "embedding",
    ):
        assert forbidden not in serialized
    assert result.evaluated_story is not None
    assert result.evaluated_story.opportunity.level is StoryOpportunityLevel.NONE


def test_module_has_no_retrieval_model_network_or_persistence_dependency() -> None:
    source_path = Path("backend/engineering_story_sufficiency.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_import_parts = {
        "api_server",
        "chroma",
        "memory_store",
        "project_retrieval",
        "query_planner",
        "coverage",
        "github_raw_storage",
        "requests",
        "httpx",
        "openai",
    }
    assert not any(
        part in imported
        for imported in imports
        for part in forbidden_import_parts
    )
    assert not ({"open", "urlopen", "request", "post"} & calls)
    source = source_path.read_text(encoding="utf-8").casefold()
    assert "story_complete" not in source
    assert "evidence_sufficient" not in source
