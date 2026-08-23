from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest

from backend.engineering_story_clustering import (
    StoryCluster,
    StoryClusterLineageState,
    StoryClusterQuality,
    cluster_story_evidence_bundle,
)
from backend.engineering_story_evidence import resolve_story_evidence_bundle
from backend.engineering_story_models import (
    EngineeringStoryFieldName,
    EngineeringStoryType,
    StoryFieldEvidenceState,
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.engineering_story_reconstruction import (
    StoryFieldDecisionReason,
    StoryReconstructionDiagnosticCode,
    StoryReconstructionError,
    StoryReconstructionErrorCode,
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
    reconstruct_engineering_story,
    reconstruct_engineering_story_from_memory,
)
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_models import (
    ClaimSubjectType,
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


def _ref(
    source_id: str,
    *,
    project_id: str = PROJECT_ID,
    change_id: str | None = "change_main",
    repo: str | None = "owner/workagent",
    commit_sha: str | None = "aaaaaaa",
    file_path: str | None = "backend/service.py",
    symbol: str | None = "resolve",
    content: str | None = None,
) -> EvidenceSourceRef:
    material = content or "|".join((
        source_id,
        project_id,
        change_id or "",
        repo or "",
        commit_sha or "",
        file_path or "",
        symbol or "",
    ))
    return EvidenceSourceRef(
        source_type="github_evidence_chunk",
        source_id=source_id,
        project_id=project_id,
        content_hash=sha256(material.encode("utf-8")).hexdigest(),
        repo=repo,
        commit_sha=commit_sha,
        file_path=file_path,
        symbol=symbol,
        metadata={} if change_id is None else {"change_id": change_id},
    )


def _fact(
    evidence_fact_id: str,
    *,
    project_id: str = PROJECT_ID,
    source_id: str | None = None,
    change_id: str | None = "change_main",
    refs: list[EvidenceSourceRef] | None = None,
    problem: str = "",
    mechanism: str = "Bounded readiness polling",
    implementation: list[str] | None = None,
    safe_impact: list[str] | None = None,
    evidence_type: EvidenceType = EvidenceType.ARCHITECTURE,
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    confidence: Confidence = Confidence.HIGH,
    metric_support: MetricSupport = MetricSupport.NONE,
    quality_score: float = 90,
) -> ProjectEvidenceFact:
    source_id = source_id or f"chunk_{evidence_fact_id}"
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=evidence_fact_id,
        problem=problem,
        mechanism=mechanism,
        implementation=(
            ["Implemented bounded readiness polling"]
            if implementation is None
            else implementation
        ),
        safe_impact=[] if safe_impact is None else safe_impact,
        source_refs=(
            [_ref(source_id, project_id=project_id, change_id=change_id)]
            if refs is None
            else refs
        ),
        evidence_type=evidence_type,
        status=status,
        confidence=confidence,
        metric_support=metric_support,
        technical_tags=["Python", "validation"],
        quality_score=quality_score,
    )


def _capability(
    facts: list[ProjectEvidenceFact],
    *,
    capability_id: str = "pcf_readiness",
    mechanisms: list[str] | None = None,
    project_id: str = PROJECT_ID,
    confidence: Confidence = Confidence.HIGH,
) -> ProjectCapabilityFact:
    return ProjectCapabilityFact(
        project_id=project_id,
        capability_id=capability_id,
        capability_type="validation_and_repair",
        present=True,
        source_evidence_fact_ids=[item.evidence_fact_id for item in facts],
        confidence=confidence,
        mechanisms=[] if mechanisms is None else mechanisms,
        metric_support=MetricSupport.NONE,
    )


def _fact_boundaries(facts: list[ProjectEvidenceFact]) -> list[ProjectClaimBoundary]:
    result = [build_project_evidence_claim_boundary(fact) for fact in facts]
    assert all(item is not None for item in result)
    return [item for item in result if item is not None]


def _cluster(
    facts: list[ProjectEvidenceFact],
    *,
    capabilities: list[ProjectCapabilityFact] | None = None,
    boundaries: list[ProjectClaimBoundary] | None = None,
    member_ids: tuple[str, ...] | None = None,
) -> StoryCluster:
    capabilities = [] if capabilities is None else capabilities
    boundaries = _fact_boundaries(facts) if boundaries is None else boundaries
    bundle = resolve_story_evidence_bundle(
        project_id=PROJECT_ID,
        evidence_fact_ids=tuple(item.evidence_fact_id for item in facts),
        evidence_facts=tuple(facts),
        capability_ids=tuple(item.capability_id for item in capabilities),
        capability_facts=tuple(capabilities),
        claim_boundary_ids=tuple(item.boundary_id for item in boundaries),
        claim_boundaries=tuple(boundaries),
    )
    result = cluster_story_evidence_bundle(bundle)
    if member_ids is None:
        assert len(result.clusters) == 1
        return result.clusters[0]
    return next(
        item for item in result.clusters if item.member_evidence_fact_ids == member_ids
    )


def _reconstruct(
    facts: list[ProjectEvidenceFact],
    *,
    capabilities: list[ProjectCapabilityFact] | None = None,
    boundaries: list[ProjectClaimBoundary] | None = None,
    cluster: StoryCluster | None = None,
):
    capabilities = [] if capabilities is None else capabilities
    boundaries = _fact_boundaries(facts) if boundaries is None else boundaries
    cluster = cluster or _cluster(
        facts,
        capabilities=capabilities,
        boundaries=boundaries,
    )
    return reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=tuple(facts),
        capability_facts=tuple(capabilities),
        claim_boundaries=tuple(boundaries),
    )


def _field_decision(result, field_name: EngineeringStoryFieldName):
    return next(
        item for item in result.field_decisions if item.field_name is field_name
    )


def test_mechanism_only_reconstruction_keeps_unproved_context_missing() -> None:
    fact = _fact("pef_mechanism")
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    story = result.engineering_story
    assert story.mechanism.value == "Bounded readiness polling"
    assert story.mechanism.evidence_state is StoryFieldEvidenceState.CONFIRMED
    assert story.implementation.evidence_state is StoryFieldEvidenceState.CONFIRMED
    assert story.problem_context.value is None
    assert story.observable_outcome.value is None
    assert story.decision.value is None
    assert story.tradeoff.value is None
    assert story.claim_sufficiency.level is SufficiencyLevel.UNASSESSED
    assert story.story_sufficiency.level is SufficiencyLevel.UNASSESSED
    assert story.opportunity.level is StoryOpportunityLevel.NONE


def test_direct_technical_problem_is_bounded_without_frequency_or_user_impact() -> None:
    fact = _fact(
        "pef_problem",
        problem="Failed startup could be classified as ready",
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    assert result.engineering_story.problem_context.value == (
        "Failed startup could be classified as ready"
    )
    serialized = result.to_json().casefold()
    assert "frequently" not in serialized
    assert "user frustration" not in serialized


def test_architecture_migration_does_not_invent_decision_rationale_or_tradeoff() -> None:
    fact = _fact(
        "pef_architecture",
        mechanism="Central HTTP access path",
        implementation=["Removed embedded access", "Added an ownership guard"],
        evidence_type=EvidenceType.ARCHITECTURE,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    story = result.engineering_story
    assert story.story_type is EngineeringStoryType.ARCHITECTURE_CHANGE
    assert story.mechanism.value == "Central HTTP access path"
    assert story.decision.value is None
    assert story.tradeoff.value is None
    assert _field_decision(
        result, EngineeringStoryFieldName.DECISION
    ).reason_code is StoryFieldDecisionReason.MISSING_HUMAN_CONTEXT


def test_validation_fact_reconstructs_specific_validation_and_observed_result() -> None:
    fact = _fact(
        "pef_validation",
        problem="Failed startup could be treated as ready",
        mechanism="Readiness state validation",
        implementation=["Added a failed-startup regression test"],
        safe_impact=["Regression test now passes for failed startup"],
        evidence_type=EvidenceType.TESTING,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    story = result.engineering_story
    assert story.validation.value == "Added a failed-startup regression test"
    assert story.observable_outcome.value == (
        "Regression test now passes for failed startup"
    )
    assert "improved reliability" not in result.to_json().casefold()


def test_numeric_outcome_is_unsupported_when_metric_support_is_none() -> None:
    fact = _fact(
        "pef_metric_none",
        safe_impact=["Reduced startup latency by 40%"],
        metric_support=MetricSupport.NONE,
        evidence_type=EvidenceType.OPTIMIZATION,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    outcome = result.engineering_story.observable_outcome
    assert outcome.value is None
    assert outcome.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
    decision = _field_decision(
        result, EngineeringStoryFieldName.OBSERVABLE_OUTCOME
    )
    assert decision.reason_code is StoryFieldDecisionReason.UNSUPPORTED_METRIC
    assert StoryReconstructionDiagnosticCode.UNSUPPORTED_METRICS in result.diagnostics


def test_approximate_metric_preserves_approximation_wording() -> None:
    fact = _fact(
        "pef_metric_approx",
        safe_impact=["Reduced startup latency by approximately 20%"],
        metric_support=MetricSupport.APPROXIMATE,
        evidence_type=EvidenceType.OPTIMIZATION,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    outcome = result.engineering_story.observable_outcome
    assert outcome.value == "Reduced startup latency by approximately 20%"
    assert "approximately" in outcome.value.casefold()
    assert "Reduced startup latency by 20%" not in result.to_json()


def test_explicit_metric_is_preserved_only_with_explicit_support() -> None:
    fact = _fact(
        "pef_metric_explicit",
        safe_impact=["Reduced startup latency by 20%"],
        metric_support=MetricSupport.EXPLICIT,
        evidence_type=EvidenceType.OPTIMIZATION,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    outcome = result.engineering_story.observable_outcome
    assert outcome.value == "Reduced startup latency by 20%"
    assert outcome.evidence_state is StoryFieldEvidenceState.CONFIRMED


def test_safe_impact_is_not_automatically_promoted_to_observable_outcome() -> None:
    fact = _fact(
        "pef_safe_impact",
        safe_impact=["Improved service reliability"],
        evidence_type=EvidenceType.FAILURE_RECOVERY,
    )
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    assert result.engineering_story.observable_outcome.value is None
    assert _field_decision(
        result, EngineeringStoryFieldName.OBSERVABLE_OUTCOME
    ).reason_code is StoryFieldDecisionReason.NON_OBSERVABLE_SAFE_IMPACT


def test_commit_authorship_does_not_create_ownership_claim() -> None:
    fact = _fact("pef_authored")
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    assert result.engineering_story.ownership.value is None
    assert result.engineering_story.ownership.evidence_state is (
        StoryFieldEvidenceState.PLAUSIBLE_MISSING
    )
    serialized = result.to_json().casefold()
    assert '"value":"led' not in serialized
    assert '"value":"owned' not in serialized
    assert '"value":"architected' not in serialized
    assert '"value":"drove' not in serialized


def test_capability_can_support_exact_mechanism_but_cannot_invent_history() -> None:
    fact = _fact("pef_capability")
    capability = _capability([fact], mechanisms=[fact.mechanism])
    result = _reconstruct([fact], capabilities=[capability])

    assert result.engineering_story is not None
    story = result.engineering_story
    assert story.mechanism.evidence_state is StoryFieldEvidenceState.SUPPORTED
    assert story.mechanism.capability_fact_ids == (capability.capability_id,)
    assert story.problem_context.value is None
    assert story.observable_outcome.value is None
    assert story.decision.value is None


def test_capability_without_exact_mechanism_match_does_not_upgrade_field() -> None:
    fact = _fact("pef_capability_mismatch")
    capability = _capability([fact], mechanisms=["Different capability mechanism"])
    result = _reconstruct([fact], capabilities=[capability])

    assert result.engineering_story is not None
    assert result.engineering_story.mechanism.evidence_state is (
        StoryFieldEvidenceState.CONFIRMED
    )
    assert not result.engineering_story.mechanism.capability_fact_ids


def test_low_confidence_capability_does_not_upgrade_direct_field() -> None:
    fact = _fact("pef_capability_low")
    capability = _capability(
        [fact],
        mechanisms=[fact.mechanism],
        confidence=Confidence.LOW,
    )
    result = _reconstruct([fact], capabilities=[capability])

    assert result.engineering_story is not None
    assert result.engineering_story.mechanism.evidence_state is (
        StoryFieldEvidenceState.CONFIRMED
    )
    assert not result.engineering_story.mechanism.capability_fact_ids


def test_exact_claim_boundary_restriction_rejects_positive_candidate() -> None:
    fact = _fact("pef_restricted", implementation=["Added guarded access"])
    boundary = ProjectClaimBoundary(
        project_id=PROJECT_ID,
        subject_type=ClaimSubjectType.PROJECT,
        subject_id=PROJECT_ID,
        forbidden_claims=[f"mechanism:{fact.mechanism}"],
        metric_support=MetricSupport.NONE,
        boundary_id="pcb_restricted_mechanism",
    )
    result = _reconstruct([fact], boundaries=[boundary])

    assert result.engineering_story is not None
    mechanism = result.engineering_story.mechanism
    assert mechanism.value is None
    assert mechanism.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
    decision = _field_decision(result, EngineeringStoryFieldName.MECHANISM)
    assert decision.reason_code is StoryFieldDecisionReason.BOUNDARY_RESTRICTED
    assert decision.evidence_fact_ids == (fact.evidence_fact_id,)
    assert decision.claim_boundary_ids == (boundary.boundary_id,)
    assert StoryReconstructionDiagnosticCode.BOUNDARY_RESTRICTIONS in (
        result.diagnostics
    )


def test_project_boundary_with_no_metric_support_blocks_explicit_metric() -> None:
    fact = _fact(
        "pef_metric_boundary",
        safe_impact=["Reduced startup latency by 20%"],
        metric_support=MetricSupport.EXPLICIT,
        evidence_type=EvidenceType.OPTIMIZATION,
    )
    fact_boundary = _fact_boundaries([fact])[0]
    project_boundary = ProjectClaimBoundary(
        project_id=PROJECT_ID,
        subject_type=ClaimSubjectType.PROJECT,
        subject_id=PROJECT_ID,
        metric_support=MetricSupport.NONE,
        boundary_id="pcb_no_project_metric",
    )
    result = _reconstruct(
        [fact],
        boundaries=[fact_boundary, project_boundary],
    )

    assert result.engineering_story is not None
    outcome = result.engineering_story.observable_outcome
    assert outcome.value is None
    assert outcome.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
    assert _field_decision(
        result, EngineeringStoryFieldName.OBSERVABLE_OUTCOME
    ).reason_code is StoryFieldDecisionReason.BOUNDARY_RESTRICTED


def test_multi_source_same_mechanism_is_supported_with_all_provenance() -> None:
    facts = [
        _fact("pef_multi_a", source_id="chunk_a"),
        _fact("pef_multi_b", source_id="chunk_b"),
    ]
    result = _reconstruct(facts)

    assert result.engineering_story is not None
    mechanism = result.engineering_story.mechanism
    assert mechanism.evidence_state is StoryFieldEvidenceState.SUPPORTED
    assert mechanism.evidence_fact_ids == ("pef_multi_a", "pef_multi_b")
    assert _field_decision(
        result, EngineeringStoryFieldName.MECHANISM
    ).reason_code is StoryFieldDecisionReason.MULTI_SOURCE_SUPPORT


def test_duplicate_lineage_does_not_claim_multi_source_support() -> None:
    shared_ref = _ref("chunk_shared", content="same evidence lineage")
    facts = [
        _fact("pef_lineage_a", refs=[shared_ref]),
        _fact("pef_lineage_b", refs=[shared_ref]),
    ]
    result = _reconstruct(facts)

    assert result.engineering_story is not None
    mechanism = result.engineering_story.mechanism
    assert mechanism.evidence_state is StoryFieldEvidenceState.CONFIRMED
    assert mechanism.evidence_fact_ids == ("pef_lineage_a", "pef_lineage_b")


def test_conflicting_mechanisms_fail_closed_without_input_order_choice() -> None:
    facts = [
        _fact("pef_conflict_a", mechanism="Central HTTP access path"),
        _fact("pef_conflict_b", mechanism="Embedded local access path"),
    ]
    first = _reconstruct(facts)
    second = _reconstruct(list(reversed(facts)))

    assert first.to_json() == second.to_json()
    assert first.engineering_story is not None
    mechanism = first.engineering_story.mechanism
    assert mechanism.value is None
    assert mechanism.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
    assert _field_decision(
        first, EngineeringStoryFieldName.MECHANISM
    ).reason_code is StoryFieldDecisionReason.CONFLICTING_EVIDENCE


def test_weak_cluster_keeps_direct_fields_and_reports_weak_structure() -> None:
    fact = _fact(
        "pef_weak_cluster",
        refs=[
            _ref(
                "chunk_weak",
                change_id=None,
                repo=None,
                commit_sha=None,
                file_path=None,
                symbol=None,
            )
        ],
    )
    cluster = _cluster([fact])
    assert cluster.quality is StoryClusterQuality.WEAK
    result = _reconstruct([fact], cluster=cluster)

    assert result.engineering_story is not None
    assert result.engineering_story.mechanism.evidence_state is (
        StoryFieldEvidenceState.CONFIRMED
    )
    assert StoryReconstructionDiagnosticCode.WEAK_CLUSTER in result.diagnostics


def test_ambiguous_cluster_reconstructs_only_its_direct_singleton() -> None:
    shared_one = _ref("shared_one", change_id=None, content="same-one")
    shared_two = _ref(
        "shared_two",
        change_id=None,
        commit_sha="bbbbbbb",
        file_path="backend/two.py",
        symbol="two",
        content="same-two",
    )
    first = _fact("pef_ambiguous_a", refs=[shared_one])
    bridge = _fact(
        "pef_ambiguous_bridge",
        refs=[shared_one, shared_two],
        mechanism="Bounded bridge validation",
    )
    second = _fact("pef_ambiguous_c", refs=[shared_two])
    all_facts = [first, bridge, second]
    all_boundaries = _fact_boundaries(all_facts)
    cluster = _cluster(
        all_facts,
        boundaries=all_boundaries,
        member_ids=(bridge.evidence_fact_id,),
    )
    assert cluster.lineage_state is StoryClusterLineageState.AMBIGUOUS
    result = reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=tuple(all_facts),
        claim_boundaries=tuple(all_boundaries),
    )

    assert result.reconstruction_quality is StoryReconstructionQuality.AMBIGUOUS
    assert result.engineering_story is not None
    assert result.engineering_story.evidence_fact_ids == (bridge.evidence_fact_id,)
    assert result.engineering_story.mechanism.value == "Bounded bridge validation"
    assert StoryReconstructionDiagnosticCode.AMBIGUOUS_CLUSTER in result.diagnostics


def test_strong_singleton_is_a_valid_minimal_or_partial_story() -> None:
    fact = _fact("pef_singleton")
    result = _reconstruct([fact])

    assert result.engineering_story is not None
    assert result.reconstruction_quality in {
        StoryReconstructionQuality.MINIMAL,
        StoryReconstructionQuality.PARTIAL,
    }
    assert result.engineering_story.story_id.startswith(
        "engineering_story_candidate_"
    )
    assert result.identity_state is (
        StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
    )


def test_weak_or_rejected_fact_cannot_create_positive_story() -> None:
    fact = _fact(
        "pef_rejected",
        status=EvidenceStatus.REJECTED,
        implementation=[],
    )
    cluster = _cluster([fact], boundaries=[])
    result = _reconstruct([fact], boundaries=[], cluster=cluster)

    assert result.engineering_story is None
    assert result.reconstruction_quality is StoryReconstructionQuality.BLOCKED
    assert StoryReconstructionDiagnosticCode.BLOCKED_NO_POSITIVE_FIELDS in (
        result.diagnostics
    )


def test_cross_project_authority_fails_before_reconstruction() -> None:
    fact = _fact("pef_local")
    foreign = _fact(
        "pef_foreign",
        project_id=OTHER_PROJECT_ID,
        refs=[_ref("foreign", project_id=OTHER_PROJECT_ID)],
    )
    cluster = _cluster([fact])
    boundaries = _fact_boundaries([fact])

    with pytest.raises(StoryReconstructionError) as exc_info:
        reconstruct_engineering_story(
            cluster=cluster,
            evidence_facts=(fact, foreign),
            claim_boundaries=tuple(boundaries),
        )
    assert exc_info.value.code is StoryReconstructionErrorCode.CROSS_PROJECT_AUTHORITY


def test_external_same_project_fact_cannot_contribute_to_cluster_story() -> None:
    local = _fact("pef_event", mechanism="Event-local guard")
    external = _fact(
        "pef_external",
        change_id="different_change",
        mechanism="Unrelated external mechanism",
        problem="Unrelated external problem",
    )
    cluster = _cluster([local])
    boundaries = _fact_boundaries([local, external])
    result = reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=(external, local),
        claim_boundaries=tuple(boundaries),
    )

    assert result.engineering_story is not None
    assert result.engineering_story.evidence_fact_ids == (local.evidence_fact_id,)
    assert "Unrelated" not in result.to_json()


@pytest.mark.parametrize(
    ("evidence_type", "expected"),
    [
        (EvidenceType.ARCHITECTURE, EngineeringStoryType.ARCHITECTURE_CHANGE),
        (EvidenceType.FAILURE_RECOVERY, EngineeringStoryType.RELIABILITY_HARDENING),
        (EvidenceType.BUG_FIX, EngineeringStoryType.DEBUGGING_AND_REPAIR),
        (EvidenceType.RETRIEVAL, EngineeringStoryType.RETRIEVAL_REDESIGN),
        (EvidenceType.VALIDATION, EngineeringStoryType.VALIDATION_AND_QUALITY),
        (EvidenceType.DATA_PERSISTENCE, EngineeringStoryType.DATA_OR_MEMORY_SYSTEM),
        (EvidenceType.WORKFLOW, EngineeringStoryType.WORKFLOW_AUTOMATION),
        (EvidenceType.OPTIMIZATION, EngineeringStoryType.PERFORMANCE_OR_EFFICIENCY),
        (EvidenceType.INTEGRATION, EngineeringStoryType.INTEGRATION),
        (EvidenceType.FEATURE, EngineeringStoryType.OTHER),
    ],
)
def test_story_type_uses_only_fixed_authoritative_evidence_mapping(
    evidence_type: EvidenceType,
    expected: EngineeringStoryType,
) -> None:
    fact = _fact("pef_story_type", evidence_type=evidence_type)
    result = _reconstruct([fact])
    assert result.engineering_story is not None
    assert result.engineering_story.story_type is expected


def test_multiple_primary_story_types_fail_closed_to_other() -> None:
    facts = [
        _fact("pef_type_arch", evidence_type=EvidenceType.ARCHITECTURE),
        _fact("pef_type_retrieval", evidence_type=EvidenceType.RETRIEVAL),
    ]
    result = _reconstruct(facts)
    assert result.engineering_story is not None
    assert result.engineering_story.story_type is EngineeringStoryType.OTHER


def test_validation_evidence_does_not_override_primary_architecture_type() -> None:
    facts = [
        _fact("pef_primary_arch", evidence_type=EvidenceType.ARCHITECTURE),
        _fact(
            "pef_support_test",
            evidence_type=EvidenceType.TESTING,
            implementation=["Added architecture migration regression test"],
        ),
    ]
    result = _reconstruct(facts)
    assert result.engineering_story is not None
    assert result.engineering_story.story_type is (
        EngineeringStoryType.ARCHITECTURE_CHANGE
    )


def test_missing_authority_and_conflicting_cluster_snapshot_fail_closed() -> None:
    fact = _fact("pef_snapshot")
    boundaries = _fact_boundaries([fact])
    cluster = _cluster([fact], boundaries=boundaries)

    with pytest.raises(StoryReconstructionError) as missing:
        reconstruct_engineering_story(
            cluster=cluster,
            evidence_facts=(),
            claim_boundaries=tuple(boundaries),
        )
    assert missing.value.code is StoryReconstructionErrorCode.MISSING_AUTHORITY

    fact.technical_tags.append("mutated-after-clustering")
    with pytest.raises(StoryReconstructionError) as conflict:
        reconstruct_engineering_story(
            cluster=cluster,
            evidence_facts=(fact,),
            claim_boundaries=tuple(boundaries),
        )
    assert conflict.value.code is StoryReconstructionErrorCode.CONFLICTING_AUTHORITY


def test_memory_adapter_matches_direct_reconstruction() -> None:
    fact = _fact("pef_memory")
    boundaries = _fact_boundaries([fact])
    cluster = _cluster([fact], boundaries=boundaries)
    memory = ProjectEvidenceMemory(
        project_id=PROJECT_ID,
        project_name="WorkAgent",
        evidence_facts=[fact],
        capability_facts=[],
        claim_boundaries=boundaries,
        quality_summary={
            "accepted_count": 1,
            "supporting_count": 0,
            "weak_count": 0,
            "rejected_count": 0,
        },
    )

    direct = _reconstruct([fact], boundaries=boundaries, cluster=cluster)
    adapted = reconstruct_engineering_story_from_memory(
        cluster=cluster,
        project_memory=memory,
    )
    assert adapted == direct
    assert adapted.to_json() == direct.to_json()


def test_reconstruction_is_order_independent_immutable_and_does_not_mutate_inputs() -> None:
    facts = [
        _fact(
            "pef_order_b",
            source_id="chunk_b",
            implementation=["Added timeout handling"],
            evidence_type=EvidenceType.FAILURE_RECOVERY,
        ),
        _fact(
            "pef_order_a",
            source_id="chunk_a",
            implementation=["Added readiness polling"],
            evidence_type=EvidenceType.FAILURE_RECOVERY,
        ),
    ]
    boundaries = _fact_boundaries(facts)
    cluster = _cluster(facts, boundaries=boundaries)
    before_facts = [item.to_json() for item in facts]
    before_boundaries = [item.to_json() for item in boundaries]

    first = reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=tuple(facts),
        claim_boundaries=tuple(boundaries),
    )
    shuffled_facts = list(facts)
    shuffled_boundaries = list(boundaries)
    random.Random(17).shuffle(shuffled_facts)
    random.Random(31).shuffle(shuffled_boundaries)
    second = reconstruct_engineering_story(
        cluster=cluster,
        evidence_facts=tuple(shuffled_facts),
        claim_boundaries=tuple(shuffled_boundaries),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert [item.to_json() for item in facts] == before_facts
    assert [item.to_json() for item in boundaries] == before_boundaries
    with pytest.raises(FrozenInstanceError):
        first.cluster_id = "story_cluster_000000000000000000000000"
    assert not hasattr(first, "__dict__")


def test_result_round_trip_is_strict_and_preserves_tuples() -> None:
    fact = _fact("pef_round_trip")
    result = _reconstruct([fact])
    payload = json.loads(result.to_json())

    restored = type(result).from_dict(payload)
    assert restored == result
    assert isinstance(restored.field_decisions, tuple)
    assert isinstance(restored.unresolved_fields, tuple)
    payload["raw_query"] = "forbidden"
    with pytest.raises(ValueError):
        type(result).from_dict(payload)


def test_field_decisions_are_complete_bounded_and_ids_only() -> None:
    fact = _fact("pef_decisions")
    result = _reconstruct([fact])

    assert tuple(item.field_name for item in result.field_decisions) == tuple(
        EngineeringStoryFieldName
    )
    assert result.unresolved_fields == tuple(
        item.field_name
        for item in result.field_decisions
        if item.resulting_state in {
            StoryFieldEvidenceState.PLAUSIBLE_MISSING,
            StoryFieldEvidenceState.UNSUPPORTED,
        }
    )
    forbidden_keys = {
        "raw_text",
        "raw_patch",
        "document",
        "source_document",
        "query",
        "prompt",
        "embedding",
        "path",
        "file_path",
        "repo",
        "score",
    }
    assert not (forbidden_keys & set().union(*(
        set(item.to_dict()) for item in result.field_decisions
    )))


def test_reconstruction_has_no_llm_network_storage_or_downstream_dependencies() -> None:
    path = Path("backend/engineering_story_reconstruction.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
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

    forbidden_imports = {
        "openai",
        "anthropic",
        "google.generativeai",
        "chromadb",
        "requests",
        "httpx",
        "backend.api_server",
        "backend.memory_store",
        "backend.project_retrieval_v2",
        "backend.project_query_planner",
        "backend.resume_budget_planner",
    }
    assert not (imports & forbidden_imports)
    assert not ({"open", "urlopen", "PersistentClient", "HttpClient"} & calls)
    source = path.read_text(encoding="utf-8").casefold()
    for token in (
        "phase" + "6_75",
        "phase" + "675",
        "rank_story",
        "generate_bullet",
        "clarification_question",
        "story_memory",
    ):
        assert token not in source
