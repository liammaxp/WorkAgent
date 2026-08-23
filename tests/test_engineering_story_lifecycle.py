from __future__ import annotations

import ast
from copy import copy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

import backend.engineering_story_lifecycle as lifecycle_module
from backend.engineering_story_clustering import StoryClusterIdentityState
from backend.engineering_story_lifecycle import (
    MAX_ENGINEERING_STORY_REVISIONS_PER_CANONICAL,
    EngineeringStoryLifecycleError,
    EngineeringStoryLifecycleErrorCode,
    EngineeringStoryLifecycleMemory,
    EngineeringStoryRevision,
    EngineeringStoryRevisionHistory,
    StoryLifecycleDecision,
    StoryLifecycleDiagnosticCode,
    StoryLifecycleReasonCode,
    StoryRevalidationOutcome,
    StoryRevisionChangeType,
    StoryRevisionProvenanceDelta,
    apply_engineering_story_revisions,
    build_engineering_story_revision_id,
)
from backend.engineering_story_matching import (
    StoryCanonicalizationResult,
    StoryMergeOutcome,
    StorySplitOutcome,
    canonicalize_engineering_story_memory,
    resolve_canonical_story_alias,
)
from backend.engineering_story_memory import (
    EngineeringStoryIdentity,
    EngineeringStoryIdentityState,
    EngineeringStoryMemoryProvenance,
    EngineeringStoryMemoryRecord,
    build_engineering_story_memory,
)
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_opportunity import (
    StoryOpportunityReasonCode,
    StoryOpportunitySignalDecision,
    StoryOpportunitySignalStrength,
)
from backend.engineering_story_reconstruction import (
    StoryReconstructionDiagnosticCode,
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
)
from backend.engineering_story_sufficiency import SufficiencyDiagnosticCode


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"
_FIELD_ORDER = tuple(EngineeringStoryFieldName)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _positive_field(
    value: str,
    evidence_ids: tuple[str, ...],
    *,
    state: StoryFieldEvidenceState = StoryFieldEvidenceState.CONFIRMED,
    boundary_ids: tuple[str, ...] = (),
) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=value,
        evidence_state=state,
        evidence_fact_ids=evidence_ids,
        claim_boundary_ids=boundary_ids,
    )


def _missing_field(
    *,
    state: StoryFieldEvidenceState = StoryFieldEvidenceState.PLAUSIBLE_MISSING,
    evidence_ids: tuple[str, ...] = (),
    boundary_ids: tuple[str, ...] = (),
) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=None,
        evidence_state=state,
        evidence_fact_ids=evidence_ids,
        claim_boundary_ids=boundary_ids,
    )


def _record(
    token: str,
    *,
    event: str,
    evidence: tuple[str, ...],
    sources: tuple[str, ...] | None = None,
    project_id: str = PROJECT_ID,
    wording: str = "Centralized HTTP ownership behind a bounded client",
    mechanism_state: StoryFieldEvidenceState = StoryFieldEvidenceState.CONFIRMED,
    mechanism_evidence: tuple[str, ...] | None = None,
    implementation: bool = False,
    validation_state: StoryFieldEvidenceState = StoryFieldEvidenceState.PLAUSIBLE_MISSING,
    validation_evidence: tuple[str, ...] = (),
    boundary_ids: tuple[str, ...] = (),
    mechanism_boundary_ids: tuple[str, ...] = (),
    reconstruction_diagnostics: tuple[StoryReconstructionDiagnosticCode, ...] = (),
    sufficiency_diagnostics: tuple[SufficiencyDiagnosticCode, ...] = (),
    claim_level: SufficiencyLevel = SufficiencyLevel.UNASSESSED,
    story_level: SufficiencyLevel = SufficiencyLevel.UNASSESSED,
    opportunity_level: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
) -> EngineeringStoryMemoryRecord:
    if len(token) != 24 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("token must be 24 lowercase hex characters")
    evidence_ids = tuple(sorted(f"pef_{value}" for value in evidence))
    boundary_authority_ids = tuple(sorted(f"pcb_{value}" for value in boundary_ids))
    mechanism_boundary_authority_ids = tuple(
        sorted(f"pcb_{value}" for value in mechanism_boundary_ids)
    )
    selected_mechanism_evidence = (
        mechanism_evidence if mechanism_evidence is not None else (evidence[0],)
    )
    mechanism_ids = tuple(sorted(f"pef_{value}" for value in selected_mechanism_evidence))
    validation_ids = tuple(sorted(f"pef_{value}" for value in validation_evidence))
    candidate_id = f"engineering_story_candidate_{token}"
    cluster_id = f"story_cluster_{token}"
    event_fingerprint = _digest(f"event:{event}")
    fields = {
        name.value: _missing_field() for name in _FIELD_ORDER
    }
    if mechanism_state in {
        StoryFieldEvidenceState.CONFIRMED,
        StoryFieldEvidenceState.SUPPORTED,
    }:
        fields[EngineeringStoryFieldName.MECHANISM.value] = _positive_field(
            wording,
            mechanism_ids,
            state=mechanism_state,
            boundary_ids=mechanism_boundary_authority_ids,
        )
    else:
        fields[EngineeringStoryFieldName.MECHANISM.value] = _missing_field(
            state=mechanism_state,
            evidence_ids=mechanism_ids if mechanism_state is StoryFieldEvidenceState.UNSUPPORTED else (),
            boundary_ids=mechanism_boundary_authority_ids,
        )
    if implementation:
        fields[EngineeringStoryFieldName.IMPLEMENTATION.value] = _positive_field(
            "Implemented deterministic request ownership and bounded failure handling",
            (evidence_ids[0],),
        )
    if validation_state in {
        StoryFieldEvidenceState.CONFIRMED,
        StoryFieldEvidenceState.SUPPORTED,
    }:
        fields[EngineeringStoryFieldName.VALIDATION.value] = _positive_field(
            "Validated deterministic retries and failure recovery",
            validation_ids,
            state=validation_state,
        )
    elif validation_state is StoryFieldEvidenceState.UNSUPPORTED:
        fields[EngineeringStoryFieldName.VALIDATION.value] = _missing_field(
            state=validation_state,
            evidence_ids=validation_ids,
        )
    supported = tuple(
        name for name in _FIELD_ORDER if fields[name.value].has_positive_value
    )
    missing = tuple(name for name in _FIELD_ORDER if name not in supported)
    claim = (
        ClaimSufficiency(claim_level)
        if claim_level is SufficiencyLevel.UNASSESSED
        else ClaimSufficiency(claim_level, supported_fields=supported, missing_fields=missing)
    )
    story_sufficiency = (
        StorySufficiency(story_level)
        if story_level is SufficiencyLevel.UNASSESSED
        else StorySufficiency(story_level, supported_fields=supported, missing_fields=missing)
    )
    if opportunity_level is StoryOpportunityLevel.NONE:
        opportunity = StoryOpportunity(StoryOpportunityLevel.NONE)
        opportunity_decisions: tuple[StoryOpportunitySignalDecision, ...] = ()
    else:
        opportunity = StoryOpportunity(
            opportunity_level,
            signals=(StoryOpportunitySignal.ARCHITECTURE_MIGRATION,),
        )
        opportunity_decisions = (StoryOpportunitySignalDecision(
            signal=StoryOpportunitySignal.ARCHITECTURE_MIGRATION,
            strength=StoryOpportunitySignalStrength.STRONG,
            supporting_story_fields=(
                EngineeringStoryFieldName.IMPLEMENTATION
                if implementation and not fields[EngineeringStoryFieldName.MECHANISM.value].has_positive_value
                else EngineeringStoryFieldName.MECHANISM,
            ),
            evidence_fact_ids=(evidence_ids[0],),
            capability_fact_ids=(),
            relevant_context_gaps=(),
            related_cluster_ids=(),
            reason_code=StoryOpportunityReasonCode.ARCHITECTURE_EVENT_RECONSTRUCTED,
        ),)
    story = EngineeringStory(
        story_id=candidate_id,
        project_id=project_id,
        story_type=EngineeringStoryType.ARCHITECTURE_CHANGE,
        **fields,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=(),
        claim_boundary_ids=boundary_authority_ids,
        lifecycle=EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
        claim_sufficiency=claim,
        story_sufficiency=story_sufficiency,
        opportunity=opportunity,
    )
    identity = EngineeringStoryIdentity(
        project_id=project_id,
        candidate_story_id=candidate_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        identity_state=EngineeringStoryIdentityState.PROVISIONAL,
        cluster_identity_state=StoryClusterIdentityState.STABLE_EVENT_CORE,
        reconstruction_identity_state=(
            StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
        ),
        canonical_story_id=None,
    )
    source_labels = sources if sources is not None else evidence
    provenance = EngineeringStoryMemoryProvenance(
        project_id=project_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=(),
        claim_boundary_ids=boundary_authority_ids,
        source_lineage_fingerprints=tuple(
            sorted(_digest(f"lineage:{value}") for value in source_labels)
        ),
    )
    return EngineeringStoryMemoryRecord(
        identity=identity,
        engineering_story=story,
        provenance=provenance,
        reconstruction_quality=StoryReconstructionQuality.PARTIAL,
        reconstruction_diagnostics=reconstruction_diagnostics,
        reconstruction_unresolved_fields=tuple(
            name for name in _FIELD_ORDER if not fields[name.value].has_positive_value
        ),
        sufficiency_diagnostics=sufficiency_diagnostics,
        opportunity_signal_decisions=opportunity_decisions,
        opportunity_diagnostics=(),
        related_project_cluster_ids=(),
    )


def _canonicalize_initial(
    *records: EngineeringStoryMemoryRecord,
) -> StoryCanonicalizationResult:
    return canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory(records)
    )


def _canonicalize_update(
    previous: tuple[EngineeringStoryMemoryRecord, ...],
    incoming: tuple[EngineeringStoryMemoryRecord, ...],
    prior_result: StoryCanonicalizationResult,
) -> StoryCanonicalizationResult:
    return canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory(previous),
        new_records=incoming,
        existing_identity_map=prior_result.identity_map,
    )


def _initialize(
    *records: EngineeringStoryMemoryRecord,
):
    canonical = _canonicalize_initial(*records)
    applied = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        new_records=records,
    )
    return canonical, applied


def _advance(
    old: EngineeringStoryMemoryRecord,
    new: EngineeringStoryMemoryRecord,
    canonical: StoryCanonicalizationResult,
    memory: EngineeringStoryLifecycleMemory,
    *,
    revalidate: bool = False,
):
    updated_canonical = _canonicalize_update((old,), (new,), canonical)
    canonical_id = updated_canonical.identity_map.candidate_links[0].canonical_story_id
    result = apply_engineering_story_revisions(
        canonicalization_result=updated_canonical,
        new_records=(new,),
        existing_lifecycle_memory=memory,
        revalidate_canonical_story_ids=(canonical_id,) if revalidate else (),
    )
    return updated_canonical, result


def _only_history(memory: EngineeringStoryLifecycleMemory) -> EngineeringStoryRevisionHistory:
    assert len(memory.histories) == 1
    return memory.histories[0]


def test_initial_revision_separates_canonical_and_revision_identity() -> None:
    record = _record("1" * 24, event="http-migration", evidence=("a",))
    _, applied = _initialize(record)
    history = _only_history(applied.memory)
    revision = history.current_revision

    assert history.canonical_story_id.startswith("engineering_story_")
    assert revision.revision_id.startswith("engineering_story_revision_")
    assert revision.revision_id != history.canonical_story_id
    assert history.current_revision_id == revision.revision_id
    assert history.current_revision_number == 1
    assert revision.parent_revision_id is None


def test_identical_rescan_is_idempotent_and_creates_no_duplicate_revision() -> None:
    record = _record("2" * 24, event="same", evidence=("a",))
    canonical, initial = _initialize(record)
    _, repeated = _advance(record, record, canonical, initial.memory)

    assert _only_history(repeated.memory).revisions == _only_history(initial.memory).revisions
    assert repeated.revalidation_results[0].outcome is StoryRevalidationOutcome.UNCHANGED
    assert not repeated.revalidation_results[0].revision_created


def test_supporting_evidence_added_creates_active_revision_with_same_canonical_id() -> None:
    old = _record("3" * 24, event="event", evidence=("a",))
    new = _record("3" * 24, event="event", evidence=("a", "b"))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)

    assert history.current_revision_number == 2
    assert history.canonical_story_id == _only_history(initial.memory).canonical_story_id
    assert StoryRevisionChangeType.PROVENANCE_ADDED in history.current_revision.change_types
    assert history.current_revision.lifecycle.status is EngineeringStoryStatus.ACTIVE
    assert not history.current_revision.lifecycle.requires_revalidation


def test_irrelevant_removed_evidence_does_not_make_story_stale() -> None:
    old = _record("4" * 24, event="event", evidence=("a", "extra"), mechanism_evidence=("a",))
    new = _record("4" * 24, event="event", evidence=("a",), mechanism_evidence=("a",))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert StoryRevisionChangeType.PROVENANCE_REMOVED in revision.change_types
    assert revision.lifecycle == EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)


def test_validation_support_removed_requires_revalidation_without_staling_mechanism() -> None:
    old = _record(
        "5" * 24,
        event="event",
        evidence=("a", "validation"),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    new = _record("5" * 24, event="event", evidence=("a",))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert revision.lifecycle.status is EngineeringStoryStatus.ACTIVE
    assert revision.lifecycle.requires_revalidation
    assert EngineeringStoryFieldName.VALIDATION in revision.lifecycle_decision.affected_fields
    assert StoryRevisionChangeType.FIELD_SUPPORT_DOWNGRADED in revision.change_types


def test_mechanism_support_removed_without_remaining_core_becomes_stale() -> None:
    old = _record(
        "6" * 24,
        event="event",
        evidence=("mechanism", "validation"),
        mechanism_evidence=("mechanism",),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    new = _record(
        "6" * 24,
        event="event",
        evidence=("validation",),
        mechanism_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        mechanism_evidence=(),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)

    assert _only_history(updated.memory).current_revision.lifecycle.status is EngineeringStoryStatus.STALE


def test_provenance_backed_contradiction_introduces_conflicted_state() -> None:
    old = _record("7" * 24, event="event", evidence=("a",))
    conflicting = _record(
        "7" * 24,
        event="event",
        evidence=("a", "b"),
        mechanism_state=StoryFieldEvidenceState.UNSUPPORTED,
        mechanism_evidence=("a", "b"),
        implementation=True,
        reconstruction_diagnostics=(StoryReconstructionDiagnosticCode.CONFLICTING_FIELDS,),
        sufficiency_diagnostics=(SufficiencyDiagnosticCode.FIELD_CONFLICT,),
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, conflicting, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert revision.lifecycle.status is EngineeringStoryStatus.CONFLICTED
    assert StoryRevisionChangeType.CONFLICT_INTRODUCED in revision.change_types


def test_uncertainty_without_provenance_backed_conflict_is_not_conflicted() -> None:
    old = _record("8" * 24, event="event", evidence=("a",), implementation=True)
    uncertain = _record(
        "8" * 24,
        event="event",
        evidence=("a",),
        mechanism_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        mechanism_evidence=(),
        implementation=True,
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, uncertain, canonical, initial.memory)

    assert _only_history(updated.memory).current_revision.lifecycle.status is not EngineeringStoryStatus.CONFLICTED


def test_conflict_cannot_return_to_active_without_explicit_revalidation() -> None:
    old = _record("9" * 24, event="event", evidence=("a",))
    conflicting = _record(
        "9" * 24,
        event="event",
        evidence=("a", "b"),
        mechanism_state=StoryFieldEvidenceState.UNSUPPORTED,
        mechanism_evidence=("a", "b"),
        implementation=True,
        reconstruction_diagnostics=(StoryReconstructionDiagnosticCode.CONFLICTING_FIELDS,),
    )
    corrected = _record("9" * 24, event="event", evidence=("a",), implementation=True)
    canonical, initial = _initialize(old)
    canonical, conflicted = _advance(old, conflicting, canonical, initial.memory)
    _, unresolved = _advance(conflicting, corrected, canonical, conflicted.memory)

    assert _only_history(unresolved.memory).current_revision.lifecycle.status is EngineeringStoryStatus.CONFLICTED


def test_explicit_revalidation_resolves_conflict_and_preserves_history() -> None:
    old = _record("a" * 24, event="event", evidence=("a",))
    conflicting = _record(
        "a" * 24,
        event="event",
        evidence=("a", "b"),
        mechanism_state=StoryFieldEvidenceState.UNSUPPORTED,
        mechanism_evidence=("a", "b"),
        implementation=True,
        reconstruction_diagnostics=(StoryReconstructionDiagnosticCode.CONFLICTING_FIELDS,),
    )
    corrected = _record("a" * 24, event="event", evidence=("a",), implementation=True)
    canonical, initial = _initialize(old)
    canonical, conflicted = _advance(old, conflicting, canonical, initial.memory)
    _, resolved = _advance(
        conflicting,
        corrected,
        canonical,
        conflicted.memory,
        revalidate=True,
    )
    history = _only_history(resolved.memory)

    assert history.current_revision.lifecycle.status is EngineeringStoryStatus.ACTIVE
    assert StoryRevisionChangeType.CONFLICT_RESOLVED in history.current_revision.change_types
    assert StoryRevisionChangeType.REVALIDATION_SUCCEEDED in history.current_revision.change_types
    assert len(history.revisions) == 3


@pytest.mark.parametrize(
    ("old_level", "new_level"),
    ((SufficiencyLevel.HIGH, SufficiencyLevel.MEDIUM), (SufficiencyLevel.LOW, SufficiencyLevel.HIGH)),
)
def test_sufficiency_changes_create_revision_without_changing_canonical_identity(
    old_level: SufficiencyLevel,
    new_level: SufficiencyLevel,
) -> None:
    old = _record("b" * 24, event="event", evidence=("a",), claim_level=old_level)
    new = _record("b" * 24, event="event", evidence=("a",), claim_level=new_level)
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)

    assert StoryRevisionChangeType.CLAIM_SUFFICIENCY_CHANGED in history.current_revision.change_types
    assert history.canonical_story_id == _only_history(initial.memory).canonical_story_id


def test_story_sufficiency_change_is_revision_not_invalidation() -> None:
    old = _record("c" * 24, event="event", evidence=("a",), story_level=SufficiencyLevel.LOW)
    new = _record("c" * 24, event="event", evidence=("a",), story_level=SufficiencyLevel.HIGH)
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert StoryRevisionChangeType.STORY_SUFFICIENCY_CHANGED in revision.change_types
    assert revision.lifecycle == EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)


def test_opportunity_change_alone_never_requires_revalidation() -> None:
    old = _record("d" * 24, event="event", evidence=("a",), opportunity_level=StoryOpportunityLevel.HIGH)
    new = _record("d" * 24, event="event", evidence=("a",), opportunity_level=StoryOpportunityLevel.LOW)
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert StoryRevisionChangeType.OPPORTUNITY_CHANGED in revision.change_types
    assert not revision.lifecycle.requires_revalidation
    assert revision.lifecycle.status is EngineeringStoryStatus.ACTIVE


def test_story_wording_change_creates_revision_but_not_new_canonical_story() -> None:
    old = _record("e" * 24, event="event", evidence=("a",))
    new = _record("e" * 24, event="event", evidence=("a",), wording="Reworded bounded ownership mechanism")
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)

    assert StoryRevisionChangeType.FIELD_CHANGED in history.current_revision.change_types
    assert history.canonical_story_id == _only_history(initial.memory).canonical_story_id


def test_confirmed_to_supported_is_a_support_upgrade_revision() -> None:
    old = _record("f" * 24, event="event", evidence=("a",))
    new = _record(
        "f" * 24,
        event="event",
        evidence=("a", "b"),
        mechanism_state=StoryFieldEvidenceState.SUPPORTED,
        mechanism_evidence=("a", "b"),
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)

    assert StoryRevisionChangeType.FIELD_SUPPORT_UPGRADED in _only_history(updated.memory).current_revision.change_types


def test_supported_to_unsupported_is_a_support_downgrade_revision() -> None:
    old = _record(
        "0" * 24,
        event="event",
        evidence=("a", "b"),
        mechanism_state=StoryFieldEvidenceState.SUPPORTED,
        mechanism_evidence=("a", "b"),
        implementation=True,
    )
    new = _record(
        "0" * 24,
        event="event",
        evidence=("a",),
        mechanism_state=StoryFieldEvidenceState.UNSUPPORTED,
        mechanism_evidence=("a",),
        implementation=True,
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert StoryRevisionChangeType.FIELD_SUPPORT_DOWNGRADED in revision.change_types
    assert StoryRevisionChangeType.FIELD_REMOVED in revision.change_types


def test_claim_boundary_restriction_of_prior_positive_field_is_conflict() -> None:
    old = _record("1a" * 12, event="event", evidence=("a",))
    new = _record(
        "1a" * 12,
        event="event",
        evidence=("a",),
        mechanism_state=StoryFieldEvidenceState.UNSUPPORTED,
        mechanism_evidence=("a",),
        implementation=True,
        boundary_ids=("restriction",),
        mechanism_boundary_ids=("restriction",),
        reconstruction_diagnostics=(StoryReconstructionDiagnosticCode.BOUNDARY_RESTRICTIONS,),
    )
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    revision = _only_history(updated.memory).current_revision

    assert StoryRevisionChangeType.CLAIM_BOUNDARY_CHANGED in revision.change_types
    assert revision.lifecycle.status is EngineeringStoryStatus.CONFLICTED


def test_requires_revalidation_can_return_active_only_through_explicit_request() -> None:
    old = _record(
        "2a" * 12,
        event="event",
        evidence=("a", "validation"),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    weaker = _record("2a" * 12, event="event", evidence=("a",))
    canonical, initial = _initialize(old)
    canonical, pending = _advance(old, weaker, canonical, initial.memory)
    _, revalidated = _advance(
        weaker,
        weaker,
        canonical,
        pending.memory,
        revalidate=True,
    )

    assert _only_history(revalidated.memory).current_revision.lifecycle == EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)
    assert StoryRevisionChangeType.REVALIDATION_SUCCEEDED in _only_history(revalidated.memory).current_revision.change_types


def test_revalidation_failure_without_conflict_can_remain_stale() -> None:
    old = _record(
        "3a" * 12,
        event="event",
        evidence=("mechanism", "validation"),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    weaker = _record(
        "3a" * 12,
        event="event",
        evidence=("validation",),
        mechanism_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        mechanism_evidence=(),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    canonical, initial = _initialize(old)
    _, failed = _advance(old, weaker, canonical, initial.memory, revalidate=True)
    revision = _only_history(failed.memory).current_revision

    assert revision.lifecycle.status is EngineeringStoryStatus.STALE
    assert StoryRevisionChangeType.REVALIDATION_FAILED in revision.change_types
    assert revision.lifecycle.status is not EngineeringStoryStatus.CONFLICTED


def test_repeated_identical_failed_revalidation_is_idempotent() -> None:
    old = _record(
        "3b" * 12,
        event="event",
        evidence=("mechanism", "validation"),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    weaker = _record(
        "3b" * 12,
        event="event",
        evidence=("validation",),
        mechanism_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        mechanism_evidence=(),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    canonical, initial = _initialize(old)
    canonical, failed = _advance(
        old,
        weaker,
        canonical,
        initial.memory,
        revalidate=True,
    )
    canonical_id = _only_history(failed.memory).canonical_story_id
    repeated = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        new_records=(weaker,),
        existing_lifecycle_memory=failed.memory,
        revalidate_canonical_story_ids=(canonical_id,),
    )

    assert repeated.memory == failed.memory
    assert repeated.revalidation_results[0].outcome is StoryRevalidationOutcome.UNCHANGED


def test_explicit_missing_authoritative_candidate_becomes_stale_idempotently() -> None:
    record = _record(
        "1a" * 12,
        event="missing-authority",
        evidence=("missing-authority",),
        implementation=True,
    )
    canonical, initial = _initialize(record)
    canonical_id = _only_history(initial.memory).canonical_story_id

    first = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        existing_lifecycle_memory=initial.memory,
        missing_canonical_story_ids=(canonical_id,),
    )
    history = _only_history(first.memory)
    assert history.current_revision.lifecycle.status is EngineeringStoryStatus.STALE
    assert history.current_revision.lifecycle.requires_revalidation is True
    assert history.current_revision.revalidation_outcome is StoryRevalidationOutcome.STALE
    assert (
        StoryLifecycleDiagnosticCode.MISSING_AUTHORITATIVE_CANDIDATE_MARKED_STALE
        in first.diagnostics
    )

    second = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        existing_lifecycle_memory=first.memory,
        missing_canonical_story_ids=(canonical_id,),
    )
    assert second.memory == first.memory
    assert second.revalidation_results == ()


def test_missing_authoritative_candidate_signal_is_default_empty_and_validated() -> None:
    record = _record(
        "1b" * 12,
        event="missing-authority-validation",
        evidence=("missing-authority-validation",),
    )
    canonical, initial = _initialize(record)
    canonical_id = _only_history(initial.memory).canonical_story_id

    unchanged = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        existing_lifecycle_memory=initial.memory,
    )
    assert unchanged.memory == initial.memory

    with pytest.raises(EngineeringStoryLifecycleError) as exc_info:
        apply_engineering_story_revisions(
            canonicalization_result=canonical,
            new_records=(record,),
            existing_lifecycle_memory=initial.memory,
            missing_canonical_story_ids=(canonical_id,),
        )
    assert exc_info.value.code is EngineeringStoryLifecycleErrorCode.INVALID_INPUT


def test_stale_recovery_requires_explicit_revalidation() -> None:
    old = _record(
        "4a" * 12,
        event="event",
        evidence=("mechanism", "validation"),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    stale_record = _record(
        "4a" * 12,
        event="event",
        evidence=("validation",),
        mechanism_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
        mechanism_evidence=(),
        validation_state=StoryFieldEvidenceState.CONFIRMED,
        validation_evidence=("validation",),
    )
    restored = _record("4a" * 12, event="event", evidence=("mechanism",))
    canonical, initial = _initialize(old)
    canonical, stale = _advance(old, stale_record, canonical, initial.memory)
    canonical, still_stale = _advance(stale_record, restored, canonical, stale.memory)
    assert _only_history(still_stale.memory).current_revision.lifecycle.status is EngineeringStoryStatus.STALE
    _, recovered = _advance(
        restored,
        restored,
        canonical,
        still_stale.memory,
        revalidate=True,
    )
    assert _only_history(recovered.memory).current_revision.lifecycle.status is EngineeringStoryStatus.ACTIVE


def test_merge_supersedes_loser_preserves_history_and_alias() -> None:
    first = _record("5a" * 12, event="first", evidence=("a",))
    second = _record("6a" * 12, event="second", evidence=("b",))
    combined = _record("7a" * 12, event="combined", evidence=("a", "b"), sources=("a", "b"))
    canonical, initial = _initialize(first, second)
    merged_canonical = _canonicalize_update((first, second), (combined,), canonical)
    assert merged_canonical.merge_decisions[0].outcome is StoryMergeOutcome.MERGED
    merged = apply_engineering_story_revisions(
        canonicalization_result=merged_canonical,
        new_records=(combined,),
        existing_lifecycle_memory=initial.memory,
    )
    decision = merged_canonical.merge_decisions[0]
    loser = merged.memory.history_for(decision.aliased_canonical_story_ids[0])

    assert loser is not None
    assert len(loser.revisions) == 2
    assert loser.current_revision.lifecycle.status is EngineeringStoryStatus.SUPERSEDED
    assert loser.current_revision.lifecycle.superseded_by_story_id == decision.survivor_canonical_story_id
    assert resolve_canonical_story_alias(merged.memory.identity_map, loser.canonical_story_id) == decision.survivor_canonical_story_id


def test_clear_split_continues_parent_and_starts_independent_child_history() -> None:
    parent = _record("8a" * 12, event="parent", evidence=("a", "b", "c"))
    retained = _record("9a" * 12, event="parent", evidence=("a", "b"))
    child = _record("aa" * 12, event="child", evidence=("c",))
    canonical, initial = _initialize(parent)
    split_canonical = _canonicalize_update((parent,), (child, retained), canonical)
    assert split_canonical.split_decisions[0].outcome is StorySplitOutcome.SPLIT
    split = apply_engineering_story_revisions(
        canonicalization_result=split_canonical,
        new_records=(child, retained),
        existing_lifecycle_memory=initial.memory,
    )
    relationship = split.memory.identity_map.split_relationships[0]
    parent_history = split.memory.history_for(relationship.parent_canonical_story_id)

    assert parent_history is not None and len(parent_history.revisions) == 2
    new_child_ids = set(relationship.child_canonical_story_ids) - {relationship.parent_canonical_story_id}
    child_history = split.memory.history_for(new_child_ids.pop())
    assert child_history is not None and len(child_history.revisions) == 1
    assert split.memory.identity_map.aliases == ()


def test_repeated_split_relationship_does_not_duplicate_revisions() -> None:
    parent = _record("8b" * 12, event="parent", evidence=("a", "b", "c"))
    retained = _record("9b" * 12, event="parent", evidence=("a", "b"))
    child = _record("ac" * 12, event="child", evidence=("c",))
    canonical, initial = _initialize(parent)
    split_canonical = _canonicalize_update((parent,), (child, retained), canonical)
    first = apply_engineering_story_revisions(
        canonicalization_result=split_canonical,
        new_records=(child, retained),
        existing_lifecycle_memory=initial.memory,
    )
    repeated = apply_engineering_story_revisions(
        canonicalization_result=split_canonical,
        new_records=(retained, child),
        existing_lifecycle_memory=first.memory,
    )

    assert repeated.memory == first.memory
    assert all(not item.revision_created for item in repeated.revalidation_results)


def test_ambiguous_split_does_not_fabricate_lifecycle_resolution() -> None:
    parent = _record("ba" * 12, event="parent", evidence=("a", "b", "c"))
    first = _record("ca" * 12, event="first", evidence=("a", "b"))
    second = _record("da" * 12, event="second", evidence=("c",))
    canonical, initial = _initialize(parent)
    ambiguous_canonical = _canonicalize_update((parent,), (first, second), canonical)
    assert ambiguous_canonical.split_decisions[0].outcome is StorySplitOutcome.AMBIGUOUS
    result = apply_engineering_story_revisions(
        canonicalization_result=ambiguous_canonical,
        new_records=(first, second),
        existing_lifecycle_memory=initial.memory,
    )

    assert result.memory.histories == initial.memory.histories
    assert StoryLifecycleDiagnosticCode.AMBIGUOUS_SPLIT_PRESERVED in result.diagnostics


def test_revision_id_is_deterministic_and_changes_with_provenance_not_canonical_id() -> None:
    old = _record("ea" * 12, event="event", evidence=("a",))
    new = _record("ea" * 12, event="event", evidence=("a", "b"))
    canonical, initial = _initialize(old)
    _, first = _advance(old, new, canonical, initial.memory)
    _, second = _advance(old, new, canonical, initial.memory)

    first_history = _only_history(first.memory)
    second_history = _only_history(second.memory)
    assert first_history == second_history
    assert first_history.current_revision_id == second_history.current_revision_id
    assert first_history.canonical_story_id == _only_history(initial.memory).canonical_story_id


def test_revision_history_round_trip_restores_tuples_and_current_pointer() -> None:
    record = _record("fa" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    restored = EngineeringStoryLifecycleMemory.from_dict(initial.memory.to_dict())

    assert restored == initial.memory
    assert isinstance(restored.histories, tuple)
    assert restored.histories[0].current_revision_id == initial.memory.histories[0].current_revision_id


def _revision_with_parent(
    revision: EngineeringStoryRevision,
    parent_id: str | None,
) -> EngineeringStoryRevision:
    revision_id = build_engineering_story_revision_id(
        canonical_story_id=revision.canonical_story_id,
        project_id=revision.project_id,
        record=revision.record,
        parent_revision_id=parent_id,
        change_types=revision.change_types,
        provenance_delta=revision.provenance_delta,
        field_changes=revision.field_changes,
        lifecycle_decision=revision.lifecycle_decision,
        revalidation_outcome=revision.revalidation_outcome,
    )
    return replace(revision, revision_id=revision_id, parent_revision_id=parent_id)


def test_missing_parent_revision_fails_closed() -> None:
    old = _record("ab" * 12, event="event", evidence=("a",))
    new = _record("ab" * 12, event="event", evidence=("a", "b"))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)
    orphan = _revision_with_parent(
        history.current_revision,
        "engineering_story_revision_" + "f" * 24,
    )

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            (history.current_revision, orphan),
            orphan.revision_id,
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.MISSING_PARENT_REVISION


def test_revision_cycle_tampering_fails_closed() -> None:
    old = _record("bc" * 12, event="event", evidence=("a",))
    new = _record("bc" * 12, event="event", evidence=("a", "b"))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)
    root = copy(history.revisions[0])
    object.__setattr__(root, "parent_revision_id", history.revisions[1].revision_id)

    with pytest.raises(EngineeringStoryLifecycleError):
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            (root, history.revisions[1]),
            history.current_revision_id,
        )


def test_invalid_current_revision_pointer_fails_closed() -> None:
    record = _record("cd" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    history = _only_history(initial.memory)

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            history.revisions,
            "engineering_story_revision_" + "f" * 24,
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_CURRENT_REVISION


def test_invalid_conflicted_to_active_transition_without_revalidation_fails() -> None:
    conflicted = EngineeringStoryLifecycle(
        EngineeringStoryStatus.CONFLICTED,
        requires_revalidation=True,
    )

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        StoryLifecycleDecision(
            project_id=PROJECT_ID,
            canonical_story_id="engineering_story_" + "a" * 24,
            previous_lifecycle=conflicted,
            resulting_lifecycle=EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
            reason_codes=(StoryLifecycleReasonCode.SEMANTIC_STATE_UPDATED,),
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION


def test_conflicted_to_stale_is_not_an_unvalidated_escape_path() -> None:
    conflicted = EngineeringStoryLifecycle(
        EngineeringStoryStatus.CONFLICTED,
        requires_revalidation=True,
    )

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        StoryLifecycleDecision(
            project_id=PROJECT_ID,
            canonical_story_id="engineering_story_" + "b" * 24,
            previous_lifecycle=conflicted,
            resulting_lifecycle=EngineeringStoryLifecycle(
                EngineeringStoryStatus.STALE,
                requires_revalidation=True,
            ),
            reason_codes=(StoryLifecycleReasonCode.SEMANTIC_STATE_UPDATED,),
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION


def test_requires_revalidation_cannot_be_cleared_without_validated_success() -> None:
    pending = EngineeringStoryLifecycle(
        EngineeringStoryStatus.ACTIVE,
        requires_revalidation=True,
    )

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        StoryLifecycleDecision(
            project_id=PROJECT_ID,
            canonical_story_id="engineering_story_" + "c" * 24,
            previous_lifecycle=pending,
            resulting_lifecycle=EngineeringStoryLifecycle(
                EngineeringStoryStatus.ACTIVE
            ),
            reason_codes=(StoryLifecycleReasonCode.SEMANTIC_STATE_UPDATED,),
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION


def test_cross_project_revision_fails_closed() -> None:
    record = _record("de" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    revision = _only_history(initial.memory).current_revision

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        replace(revision, project_id=OTHER_PROJECT_ID)
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT


def test_caller_cannot_promote_or_inject_lifecycle_state() -> None:
    record = _record("ef" * 12, event="event", evidence=("a",))
    canonical = _canonicalize_initial(record)
    unsafe_story = replace(
        record.engineering_story,
        lifecycle=EngineeringStoryLifecycle(
            EngineeringStoryStatus.STALE,
            requires_revalidation=True,
        ),
    )
    unsafe_record = replace(
        record,
        engineering_story=unsafe_story,
        record_fingerprint="",
    )

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        apply_engineering_story_revisions(
            canonicalization_result=canonical,
            new_records=(unsafe_record,),
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.UNSAFE_CALLER_LIFECYCLE


def test_conflicting_duplicate_revision_id_fails_closed() -> None:
    record = _record("12" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    history = _only_history(initial.memory)
    conflicting = copy(history.current_revision)
    object.__setattr__(conflicting, "record_fingerprint", "f" * 64)

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            (history.current_revision, conflicting),
            history.current_revision_id,
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.CONFLICTING_REVISION_ID


def test_revision_history_rejects_tampered_change_classification() -> None:
    old = _record("13" * 12, event="event", evidence=("a",))
    new = _record("13" * 12, event="event", evidence=("a", "b"))
    canonical, initial = _initialize(old)
    _, updated = _advance(old, new, canonical, initial.memory)
    history = _only_history(updated.memory)
    tampered = copy(history.current_revision)
    object.__setattr__(tampered, "provenance_delta", StoryRevisionProvenanceDelta())

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            (history.revisions[0], tampered),
            tampered.revision_id,
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION


def test_revision_history_bound_fails_explicitly_without_truncation() -> None:
    record = _record("23" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    history = _only_history(initial.memory)

    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        EngineeringStoryRevisionHistory(
            history.founding_identity,
            history.revisions * (MAX_ENGINEERING_STORY_REVISIONS_PER_CANONICAL + 1),
            history.current_revision_id,
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED


def test_outputs_are_immutable_and_inputs_are_not_mutated() -> None:
    record = _record("34" * 12, event="event", evidence=("a",))
    canonical = _canonicalize_initial(record)
    records = [record]
    before_record = record.to_json()
    before_canonical = canonical.to_json()
    applied = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        new_records=records,
    )

    assert record.to_json() == before_record
    assert canonical.to_json() == before_canonical
    assert records == [record]
    assert isinstance(applied.memory.histories, tuple)
    with pytest.raises(FrozenInstanceError):
        applied.memory.logical_fingerprint = "f" * 64


def test_input_order_does_not_affect_memory_or_results() -> None:
    first = _record("45" * 12, event="first", evidence=("a",))
    second = _record("56" * 12, event="second", evidence=("b",))
    canonical = _canonicalize_initial(first, second)
    forward = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        new_records=(first, second),
    )
    reverse = apply_engineering_story_revisions(
        canonicalization_result=canonical,
        new_records=(second, first),
    )

    assert forward == reverse
    assert forward.to_json() == reverse.to_json()


def test_revision_identity_has_no_timestamp_or_random_component() -> None:
    source = Path(lifecycle_module.__file__).read_text(encoding="utf-8")
    lowered = source.casefold()

    assert "datetime" not in lowered
    assert "timestamp" not in lowered
    assert "uuid" not in lowered
    assert "random" not in lowered
    assert "mtime" not in lowered


def test_lifecycle_module_has_no_retrieval_model_network_or_persistence_dependencies() -> None:
    source = Path(lifecycle_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden_fragments = (
        "api_server",
        "chroma",
        "retrieval",
        "query_planner",
        "memory_store",
        "github_raw_storage",
        "requests",
        "httpx",
        "openai",
        "socket",
        "pathlib",
        "tempfile",
    )

    assert not any(
        any(part in imported.casefold() for part in forbidden_fragments)
        for imported in imports
    )
    assert "os" not in imports
    assert "open(" not in source
    assert "os.environ" not in source


def test_serialized_revision_contract_contains_no_raw_or_path_fields() -> None:
    record = _record("67" * 12, event="event", evidence=("a",))
    _, initial = _initialize(record)
    payload = initial.to_json().casefold()
    forbidden_keys = (
        '"raw_patch"',
        '"raw_text"',
        '"document"',
        '"embedding"',
        '"token"',
        '"credential"',
        '"private_key"',
        '"file_path"',
        '"query"',
        '"jd"',
        '"company"',
    )

    assert all(value not in payload for value in forbidden_keys)


def test_provenance_delta_rejects_same_id_as_added_and_removed() -> None:
    with pytest.raises(EngineeringStoryLifecycleError) as exc:
        StoryRevisionProvenanceDelta(
            added_evidence_fact_ids=("pef_same",),
            removed_evidence_fact_ids=("pef_same",),
        )
    assert exc.value.code is EngineeringStoryLifecycleErrorCode.INVALID_INPUT
