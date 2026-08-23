from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

import backend.engineering_story_matching as matching_module
from backend.engineering_story_clustering import StoryClusterIdentityState
from backend.engineering_story_matching import (
    CandidateCanonicalStoryLink,
    CanonicalEngineeringStoryIdentity,
    CanonicalStorySeedKind,
    EngineeringStoryIdentityMap,
    EngineeringStoryMatchingError,
    EngineeringStoryMatchingErrorCode,
    StoryCanonicalizationDiagnosticCode,
    StoryCanonicalizationResult,
    StoryIdentityAlias,
    StoryMatchOutcome,
    StoryMatchReasonCode,
    StoryMatchSignal,
    StoryMatchSignalCategory,
    StoryMergeOutcome,
    StorySplitOutcome,
    build_canonical_engineering_story_id,
    canonicalize_engineering_story_memory,
    classify_story_match_signal,
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
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
)


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"
_FIELD_ORDER = tuple(EngineeringStoryFieldName)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _canonical_id(
    label: str,
    *,
    project_id: str = PROJECT_ID,
) -> CanonicalEngineeringStoryIdentity:
    seed = _digest(f"canonical-seed:{label}")
    return CanonicalEngineeringStoryIdentity(
        project_id=project_id,
        canonical_story_id=build_canonical_engineering_story_id(
            project_id=project_id,
            seed_kind=CanonicalStorySeedKind.STABLE_EVENT_CORE,
            seed_fingerprint=seed,
        ),
        founding_seed_kind=CanonicalStorySeedKind.STABLE_EVENT_CORE,
        founding_seed_fingerprint=seed,
    )


def _record(
    token: str,
    *,
    event: str,
    evidence: tuple[str, ...],
    sources: tuple[str, ...] | None = None,
    project_id: str = PROJECT_ID,
    story_type: EngineeringStoryType = EngineeringStoryType.ARCHITECTURE_CHANGE,
    wording: str = "Centralized HTTP ownership behind a bounded client",
    cluster_state: StoryClusterIdentityState = (
        StoryClusterIdentityState.STABLE_EVENT_CORE
    ),
    claim_level: SufficiencyLevel = SufficiencyLevel.UNASSESSED,
    story_level: SufficiencyLevel = SufficiencyLevel.UNASSESSED,
    opportunity_level: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
) -> EngineeringStoryMemoryRecord:
    if len(token) != 24 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("token must be 24 lowercase hex characters")
    evidence_ids = tuple(sorted(f"pef_{value}" for value in evidence))
    source_labels = sources if sources is not None else evidence
    source_fingerprints = tuple(sorted(_digest(f"lineage:{value}") for value in source_labels))
    candidate_id = f"engineering_story_candidate_{token}"
    cluster_id = f"story_cluster_{token}"
    event_fingerprint = _digest(f"event:{event}")
    positive_field = EngineeringStoryField(
        value=wording,
        evidence_state=StoryFieldEvidenceState.CONFIRMED,
        evidence_fact_ids=(evidence_ids[0],),
    )
    missing_field = EngineeringStoryField(
        value=None,
        evidence_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
    )
    fields = {
        name.value: positive_field if name is EngineeringStoryFieldName.MECHANISM else missing_field
        for name in _FIELD_ORDER
    }
    claim_sufficiency = (
        ClaimSufficiency(claim_level)
        if claim_level is SufficiencyLevel.UNASSESSED
        else ClaimSufficiency(
            claim_level,
            supported_fields=(EngineeringStoryFieldName.MECHANISM,),
        )
    )
    story_sufficiency = (
        StorySufficiency(story_level)
        if story_level is SufficiencyLevel.UNASSESSED
        else StorySufficiency(
            story_level,
            supported_fields=(EngineeringStoryFieldName.MECHANISM,),
        )
    )
    if opportunity_level is StoryOpportunityLevel.NONE:
        opportunity = StoryOpportunity(StoryOpportunityLevel.NONE)
        signal_decisions: tuple[StoryOpportunitySignalDecision, ...] = ()
    else:
        opportunity = StoryOpportunity(
            opportunity_level,
            signals=(StoryOpportunitySignal.ARCHITECTURE_MIGRATION,),
        )
        signal_decisions = (StoryOpportunitySignalDecision(
            signal=StoryOpportunitySignal.ARCHITECTURE_MIGRATION,
            strength=StoryOpportunitySignalStrength.STRONG,
            supporting_story_fields=(EngineeringStoryFieldName.MECHANISM,),
            evidence_fact_ids=(evidence_ids[0],),
            capability_fact_ids=(),
            relevant_context_gaps=(),
            related_cluster_ids=(),
            reason_code=StoryOpportunityReasonCode.ARCHITECTURE_EVENT_RECONSTRUCTED,
        ),)
    story = EngineeringStory(
        story_id=candidate_id,
        project_id=project_id,
        story_type=story_type,
        **fields,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=(),
        claim_boundary_ids=(),
        lifecycle=EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        opportunity=opportunity,
    )
    identity = EngineeringStoryIdentity(
        project_id=project_id,
        candidate_story_id=candidate_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        identity_state=EngineeringStoryIdentityState.PROVISIONAL,
        cluster_identity_state=cluster_state,
        reconstruction_identity_state=(
            StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
        ),
        canonical_story_id=None,
    )
    provenance = EngineeringStoryMemoryProvenance(
        project_id=project_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=(),
        claim_boundary_ids=(),
        source_lineage_fingerprints=source_fingerprints,
    )
    return EngineeringStoryMemoryRecord(
        identity=identity,
        engineering_story=story,
        provenance=provenance,
        reconstruction_quality=StoryReconstructionQuality.PARTIAL,
        reconstruction_diagnostics=(),
        reconstruction_unresolved_fields=tuple(
            name for name in _FIELD_ORDER if name is not EngineeringStoryFieldName.MECHANISM
        ),
        sufficiency_diagnostics=(),
        opportunity_signal_decisions=signal_decisions,
        opportunity_diagnostics=(),
        related_project_cluster_ids=(),
    )


def _initial_result(
    *records: EngineeringStoryMemoryRecord,
) -> StoryCanonicalizationResult:
    return canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory(records),
    )


def _link_for(
    result: StoryCanonicalizationResult,
    candidate_id: str,
    *,
    project_id: str = PROJECT_ID,
) -> CandidateCanonicalStoryLink | None:
    return next(
        (
            item
            for item in result.identity_map.candidate_links
            if item.project_id == project_id
            and item.candidate_story_id == candidate_id
        ),
        None,
    )


def test_signal_categories_are_explicit_and_context_cannot_establish_identity() -> None:
    assert classify_story_match_signal(
        StoryMatchSignal.EXACT_CANDIDATE_IDENTITY
    ) is StoryMatchSignalCategory.IDENTITY_ESTABLISHING
    assert classify_story_match_signal(
        StoryMatchSignal.SAME_DURABLE_EVENT_CORE
    ) is StoryMatchSignalCategory.IDENTITY_ESTABLISHING
    assert classify_story_match_signal(
        StoryMatchSignal.EVIDENCE_AUTHORITY_CONTINUITY
    ) is StoryMatchSignalCategory.IDENTITY_SUPPORTING
    assert classify_story_match_signal(
        StoryMatchSignal.SOURCE_LINEAGE_CONTINUITY
    ) is StoryMatchSignalCategory.IDENTITY_SUPPORTING
    assert classify_story_match_signal(
        StoryMatchSignal.SAME_STORY_TYPE
    ) is StoryMatchSignalCategory.CONTEXT_ONLY


def test_exact_repeated_candidate_keeps_canonical_identity() -> None:
    record = _record("1" * 24, event="http-migration", evidence=("a",))
    initial = _initial_result(record)
    repeated = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((record,)),
        new_records=(record,),
        existing_identity_map=initial.identity_map,
    )

    first_link = _link_for(initial, record.candidate_story_id)
    repeated_link = _link_for(repeated, record.candidate_story_id)
    assert first_link is not None and repeated_link is not None
    assert repeated_link.canonical_story_id == first_link.canonical_story_id
    assert repeated.match_decisions[0].outcome is StoryMatchOutcome.EXACT_MATCH


def test_content_enrichment_same_candidate_does_not_change_identity() -> None:
    original = _record("1" * 24, event="http-migration", evidence=("a",))
    enriched = _record(
        "1" * 24,
        event="http-migration",
        evidence=("a", "validation"),
    )
    initial = _initial_result(original)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((original,)),
        new_records=(enriched,),
        existing_identity_map=initial.identity_map,
    )

    assert enriched.record_fingerprint != original.record_fingerprint
    assert _link_for(result, enriched.candidate_story_id) == _link_for(
        initial,
        original.candidate_story_id,
    )


@pytest.mark.parametrize(
    "change",
    ("wording", "claim", "story", "opportunity"),
)
def test_mutable_story_assessments_and_wording_do_not_change_identity(change: str) -> None:
    original = _record("2" * 24, event="owner-guard", evidence=("a",))
    kwargs: dict[str, object] = {}
    if change == "wording":
        kwargs["wording"] = "Introduced a deterministic single-owner boundary"
    elif change == "claim":
        kwargs["claim_level"] = SufficiencyLevel.HIGH
    elif change == "story":
        kwargs["story_level"] = SufficiencyLevel.LOW
    else:
        kwargs["opportunity_level"] = StoryOpportunityLevel.HIGH
    changed = _record(
        "2" * 24,
        event="owner-guard",
        evidence=("a",),
        **kwargs,
    )
    initial = _initial_result(original)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((original,)),
        new_records=(changed,),
        existing_identity_map=initial.identity_map,
    )

    assert _link_for(result, changed.candidate_story_id) == _link_for(
        initial,
        original.candidate_story_id,
    )


def test_same_story_type_and_wording_different_events_stay_separate() -> None:
    migration = _record("3" * 24, event="migration", evidence=("migration",))
    readiness = _record("4" * 24, event="readiness", evidence=("readiness",))
    initial = _initial_result(migration)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((migration,)),
        new_records=(readiness,),
        existing_identity_map=initial.identity_map,
    )

    first = _link_for(initial, migration.candidate_story_id)
    second = _link_for(result, readiness.candidate_story_id)
    assert first is not None and second is not None
    assert first.canonical_story_id != second.canonical_story_id
    decision = next(
        item for item in result.match_decisions
        if item.candidate_story_id == readiness.candidate_story_id
    )
    assert decision.reason_code is StoryMatchReasonCode.WEAK_CONTEXT_ONLY
    assert StoryMatchSignal.SAME_STORY_TYPE in decision.signals


def test_same_technology_and_subsystem_language_does_not_merge() -> None:
    first = _record(
        "5" * 24,
        event="chroma-migration",
        evidence=("migration",),
        wording="Chroma service migration with bounded readiness checks",
    )
    second = _record(
        "6" * 24,
        event="chroma-readiness",
        evidence=("readiness",),
        wording="Chroma service readiness hardening with bounded checks",
    )
    initial = _initial_result(first)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first,)),
        new_records=(second,),
        existing_identity_map=initial.identity_map,
    )

    assert len(result.identity_map.canonical_identities) == 2
    assert result.merge_decisions == ()


def test_different_candidate_ids_same_durable_core_strongly_match() -> None:
    first = _record("7" * 24, event="shared-event", evidence=("a",))
    second = _record("8" * 24, event="shared-event", evidence=("b",))
    initial = _initial_result(first)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first,)),
        new_records=(second,),
        existing_identity_map=initial.identity_map,
    )

    first_link = _link_for(initial, first.candidate_story_id)
    second_link = _link_for(result, second.candidate_story_id)
    assert first_link is not None and second_link is not None
    assert second_link.canonical_story_id == first_link.canonical_story_id
    decision = next(
        item for item in result.match_decisions
        if item.candidate_story_id == second.candidate_story_id
    )
    assert decision.outcome is StoryMatchOutcome.STRONG_MATCH
    assert StoryMatchSignal.SAME_DURABLE_EVENT_CORE in decision.signals


def test_evidence_and_source_lineage_together_can_support_strong_continuity() -> None:
    first = _record("9" * 24, event="old-core", evidence=("shared", "old"))
    evolved = _record(
        "a" * 24,
        event="new-core",
        evidence=("shared", "new"),
        sources=("shared", "new"),
    )
    initial = _initial_result(first)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first,)),
        new_records=(evolved,),
        existing_identity_map=initial.identity_map,
    )

    decision = result.match_decisions[0]
    assert decision.outcome is StoryMatchOutcome.STRONG_MATCH
    assert set(decision.signals) >= {
        StoryMatchSignal.EVIDENCE_AUTHORITY_CONTINUITY,
        StoryMatchSignal.SOURCE_LINEAGE_CONTINUITY,
    }


def test_evidence_overlap_without_source_continuity_is_not_strong() -> None:
    first = _record(
        "b" * 24,
        event="old-core",
        evidence=("shared",),
        sources=("old-source",),
    )
    second = _record(
        "c" * 24,
        event="new-core",
        evidence=("shared",),
        sources=("new-source",),
    )
    initial = _initial_result(first)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first,)),
        new_records=(second,),
        existing_identity_map=initial.identity_map,
    )

    assert len(result.identity_map.canonical_identities) == 2


def test_cluster_member_addition_and_removal_keep_stable_event_identity() -> None:
    original = _record(
        "d" * 24,
        event="stable-core",
        evidence=("a", "b"),
    )
    added = _record(
        "e" * 24,
        event="stable-core",
        evidence=("a", "b", "c"),
    )
    removed = _record(
        "f" * 24,
        event="stable-core",
        evidence=("a",),
    )
    initial = _initial_result(original)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((original,)),
        new_records=(added, removed),
        existing_identity_map=initial.identity_map,
    )

    expected = _link_for(initial, original.candidate_story_id)
    assert expected is not None
    assert _link_for(result, added.candidate_story_id).canonical_story_id == (
        expected.canonical_story_id
    )
    assert _link_for(result, removed.candidate_story_id).canonical_story_id == (
        expected.canonical_story_id
    )


def test_cross_project_structural_equality_never_shares_identity() -> None:
    first = _record("1" * 24, event="same-core", evidence=("a",))
    foreign = _record(
        "2" * 24,
        event="same-core",
        evidence=("a",),
        project_id=OTHER_PROJECT_ID,
    )
    initial = _initial_result(first)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first,)),
        new_records=(foreign,),
        existing_identity_map=initial.identity_map,
    )

    local_link = _link_for(initial, first.candidate_story_id)
    foreign_link = _link_for(
        result,
        foreign.candidate_story_id,
        project_id=OTHER_PROJECT_ID,
    )
    assert local_link is not None and foreign_link is not None
    assert local_link.canonical_story_id != foreign_link.canonical_story_id
    assert StoryCanonicalizationDiagnosticCode.CROSS_PROJECT_CANDIDATE_ISOLATED in (
        result.diagnostics
    )


def test_multiple_new_candidates_can_attach_to_one_canonical_identity() -> None:
    original = _record("3" * 24, event="same-core", evidence=("a", "b"))
    first = _record("4" * 24, event="same-core", evidence=("a",))
    second = _record("5" * 24, event="same-core", evidence=("b",))
    initial = _initial_result(original)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((original,)),
        new_records=(first, second),
        existing_identity_map=initial.identity_map,
    )

    targets = {
        _link_for(result, first.candidate_story_id).canonical_story_id,
        _link_for(result, second.candidate_story_id).canonical_story_id,
    }
    assert len(targets) == 1


def test_ambiguous_candidate_matching_two_canonicals_is_not_assigned() -> None:
    first = _record(
        "6" * 24,
        event="first-core",
        evidence=("shared", "first"),
        sources=("shared", "first"),
    )
    second = _record(
        "7" * 24,
        event="second-core",
        evidence=("shared", "second"),
        sources=("shared", "second"),
    )
    ambiguous = _record(
        "8" * 24,
        event="third-core",
        evidence=("shared",),
        sources=("shared",),
    )
    initial = _initial_result(first, second)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first, second)),
        new_records=(ambiguous,),
        existing_identity_map=initial.identity_map,
    )

    assert _link_for(result, ambiguous.candidate_story_id) is None
    assert result.match_decisions[0].outcome is StoryMatchOutcome.AMBIGUOUS
    assert result.merge_decisions[0].outcome is StoryMergeOutcome.AMBIGUOUS


def test_strong_common_core_merge_keeps_deterministic_survivor_and_alias() -> None:
    first = _record("9" * 24, event="first-core", evidence=("a",))
    second = _record("a" * 24, event="second-core", evidence=("b",))
    combined = _record(
        "b" * 24,
        event="proven-common-core",
        evidence=("a", "b"),
        sources=("a", "b"),
    )
    initial = _initial_result(first, second)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first, second)),
        new_records=(combined,),
        existing_identity_map=initial.identity_map,
    )

    merge = result.merge_decisions[0]
    assert merge.outcome is StoryMergeOutcome.MERGED
    assert merge.survivor_canonical_story_id == min(
        merge.participant_canonical_story_ids
    )
    assert merge.aliased_canonical_story_ids == tuple(
        item
        for item in merge.participant_canonical_story_ids
        if item != merge.survivor_canonical_story_id
    )
    assert len(result.identity_map.aliases) == 1
    assert _link_for(result, combined.candidate_story_id).canonical_story_id == (
        merge.survivor_canonical_story_id
    )


def test_weak_common_context_never_creates_merge_or_alias() -> None:
    first = _record("c" * 24, event="first", evidence=("a",))
    second = _record("d" * 24, event="second", evidence=("b",))
    context_only = _record("e" * 24, event="third", evidence=("c",))
    initial = _initial_result(first, second)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((first, second)),
        new_records=(context_only,),
        existing_identity_map=initial.identity_map,
    )

    assert result.merge_decisions == ()
    assert result.identity_map.aliases == ()
    assert len(result.identity_map.canonical_identities) == 3


def test_aliases_resolve_transitively_and_are_flattened() -> None:
    first, second, third = (_canonical_id(label) for label in ("a", "b", "c"))
    identity_map = EngineeringStoryIdentityMap(
        canonical_identities=(first, second, third),
        aliases=(
            StoryIdentityAlias(PROJECT_ID, second.canonical_story_id, first.canonical_story_id),
            StoryIdentityAlias(PROJECT_ID, third.canonical_story_id, second.canonical_story_id),
        ),
    )

    assert resolve_canonical_story_alias(
        identity_map,
        third.canonical_story_id,
    ) == first.canonical_story_id
    assert {item.canonical_story_id for item in identity_map.aliases} == {
        first.canonical_story_id
    }


def test_alias_cycle_fails_closed() -> None:
    first, second = (_canonical_id(label) for label in ("a", "b"))

    with pytest.raises(EngineeringStoryMatchingError) as exc:
        EngineeringStoryIdentityMap(
            canonical_identities=(first, second),
            aliases=(
                StoryIdentityAlias(PROJECT_ID, first.canonical_story_id, second.canonical_story_id),
                StoryIdentityAlias(PROJECT_ID, second.canonical_story_id, first.canonical_story_id),
            ),
        )
    assert exc.value.code is EngineeringStoryMatchingErrorCode.ALIAS_CYCLE


def test_cross_project_alias_fails_closed() -> None:
    local = _canonical_id("local")
    foreign = _canonical_id("foreign", project_id=OTHER_PROJECT_ID)

    with pytest.raises(EngineeringStoryMatchingError) as exc:
        EngineeringStoryIdentityMap(
            canonical_identities=(local, foreign),
            aliases=(StoryIdentityAlias(
                PROJECT_ID,
                local.canonical_story_id,
                foreign.canonical_story_id,
            ),),
        )
    assert exc.value.code is EngineeringStoryMatchingErrorCode.CROSS_PROJECT_ALIAS


def test_clear_split_retains_parent_only_for_original_core_child() -> None:
    parent = _record(
        "f" * 24,
        event="original-core",
        evidence=("a", "b", "c"),
    )
    original_child = _record(
        "1" * 24,
        event="original-core",
        evidence=("a", "b"),
    )
    new_child = _record(
        "2" * 24,
        event="separate-core",
        evidence=("c",),
    )
    initial = _initial_result(parent)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((parent,)),
        new_records=(new_child, original_child),
        existing_identity_map=initial.identity_map,
    )

    parent_id = _link_for(initial, parent.candidate_story_id).canonical_story_id
    assert result.split_decisions[0].outcome is StorySplitOutcome.SPLIT
    assert result.split_decisions[0].retained_parent_candidate_story_id == (
        original_child.candidate_story_id
    )
    assert _link_for(result, original_child.candidate_story_id).canonical_story_id == parent_id
    assert _link_for(result, new_child.candidate_story_id).canonical_story_id != parent_id
    assert result.identity_map.aliases == ()
    relationship = result.identity_map.split_relationships[0]
    assert relationship.parent_canonical_story_id in (
        relationship.child_canonical_story_ids
    )


def test_ambiguous_split_assigns_neither_child_and_creates_no_alias() -> None:
    parent = _record(
        "3" * 24,
        event="old-mega-core",
        evidence=("a", "b", "c"),
    )
    first = _record("4" * 24, event="first-child", evidence=("a", "b"))
    second = _record("5" * 24, event="second-child", evidence=("c",))
    initial = _initial_result(parent)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((parent,)),
        new_records=(first, second),
        existing_identity_map=initial.identity_map,
    )

    assert result.split_decisions[0].outcome is StorySplitOutcome.AMBIGUOUS
    assert _link_for(result, first.candidate_story_id) is None
    assert _link_for(result, second.candidate_story_id) is None
    assert result.identity_map.aliases == ()
    assert result.identity_map.split_relationships[0].child_canonical_story_ids == ()


def test_split_lineage_is_not_an_alias_mapping() -> None:
    parent = _record("6" * 24, event="old", evidence=("a", "b"))
    retained = _record("7" * 24, event="old", evidence=("a",))
    separate = _record("8" * 24, event="new", evidence=("b",))
    initial = _initial_result(parent)
    result = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((parent,)),
        new_records=(retained, separate),
        existing_identity_map=initial.identity_map,
    )

    assert result.identity_map.split_relationships
    assert result.identity_map.aliases == ()


def test_canonical_ids_use_only_project_and_founding_structural_seed() -> None:
    seed = _digest("founding-event")
    first = build_canonical_engineering_story_id(
        project_id=PROJECT_ID,
        seed_kind=CanonicalStorySeedKind.STABLE_EVENT_CORE,
        seed_fingerprint=seed,
    )
    repeated = build_canonical_engineering_story_id(
        project_id=PROJECT_ID,
        seed_kind=CanonicalStorySeedKind.STABLE_EVENT_CORE,
        seed_fingerprint=seed,
    )
    foreign = build_canonical_engineering_story_id(
        project_id=OTHER_PROJECT_ID,
        seed_kind=CanonicalStorySeedKind.STABLE_EVENT_CORE,
        seed_fingerprint=seed,
    )

    assert first == repeated
    assert first.startswith("engineering_story_")
    assert "candidate" not in first
    assert first != foreign


def test_provisional_cluster_uses_founding_identity_not_story_content() -> None:
    record = _record(
        "9" * 24,
        event="candidate-core",
        evidence=("a",),
        cluster_state=StoryClusterIdentityState.CANDIDATE,
    )
    changed = _record(
        "9" * 24,
        event="candidate-core",
        evidence=("a",),
        wording="Reworded reconstruction without changing its founding identity",
        cluster_state=StoryClusterIdentityState.CANDIDATE,
    )
    first = _initial_result(record)
    second = canonicalize_engineering_story_memory(
        existing_memory=build_engineering_story_memory((record,)),
        new_records=(changed,),
        existing_identity_map=first.identity_map,
    )

    identity = first.identity_map.canonical_identities[0]
    assert identity.founding_seed_kind is (
        CanonicalStorySeedKind.PROVISIONAL_FOUNDING_IDENTITY
    )
    assert _link_for(second, changed.candidate_story_id).canonical_story_id == (
        identity.canonical_story_id
    )


def test_input_permutations_produce_identical_results() -> None:
    existing_a = _record("a" * 24, event="a", evidence=("a",))
    existing_b = _record("b" * 24, event="b", evidence=("b",))
    new_a = _record("c" * 24, event="a", evidence=("a", "extra"))
    new_b = _record("d" * 24, event="distinct", evidence=("d",))
    memory = build_engineering_story_memory((existing_b, existing_a))
    initial = canonicalize_engineering_story_memory(existing_memory=memory)

    forward = canonicalize_engineering_story_memory(
        existing_memory=memory,
        new_records=(new_a, new_b),
        existing_identity_map=initial.identity_map,
    )
    reverse = canonicalize_engineering_story_memory(
        existing_memory=memory,
        new_records=(new_b, new_a),
        existing_identity_map=initial.identity_map,
    )
    assert forward == reverse
    assert forward.to_json() == reverse.to_json()


def test_result_round_trip_is_strict_deterministic_and_immutable() -> None:
    record = _record("e" * 24, event="a", evidence=("a",))
    result = _initial_result(record)

    assert StoryCanonicalizationResult.from_dict(result.to_dict()) == result
    assert not hasattr(result, "__dict__")
    assert not hasattr(result.identity_map, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.result_fingerprint = "0" * 64  # type: ignore[misc]
    payload = result.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        StoryCanonicalizationResult.from_dict(payload)


def test_conflicting_duplicate_new_candidate_fails_closed() -> None:
    first = _record("f" * 24, event="a", evidence=("a",))
    changed = _record(
        "f" * 24,
        event="a",
        evidence=("a", "b"),
    )

    with pytest.raises(EngineeringStoryMatchingError) as exc:
        canonicalize_engineering_story_memory(
            existing_memory=build_engineering_story_memory(),
            new_records=(first, changed),
        )
    assert exc.value.code is (
        EngineeringStoryMatchingErrorCode.CONFLICTING_CANDIDATE_LINK
    )


def test_comparison_and_candidate_bounds_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _record("1" * 24, event="a", evidence=("a",))
    second = _record("2" * 24, event="b", evidence=("b",))
    initial = _initial_result(first)
    monkeypatch.setattr(matching_module, "MAX_CANONICAL_COMPARISONS", 0)
    with pytest.raises(EngineeringStoryMatchingError) as comparison_exc:
        canonicalize_engineering_story_memory(
            existing_memory=build_engineering_story_memory((first,)),
            new_records=(second,),
            existing_identity_map=initial.identity_map,
        )
    assert comparison_exc.value.code is (
        EngineeringStoryMatchingErrorCode.COMPARISON_BOUND_EXCEEDED
    )

    monkeypatch.setattr(matching_module, "MAX_CANONICALIZATION_CANDIDATES", 1)
    with pytest.raises(EngineeringStoryMatchingError) as candidate_exc:
        canonicalize_engineering_story_memory(
            existing_memory=build_engineering_story_memory(),
            new_records=(first, second),
        )
    assert candidate_exc.value.code is EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED


def test_canonicalization_does_not_merge_or_rewrite_story_content() -> None:
    original = _record("3" * 24, event="same", evidence=("a",))
    reworded = _record(
        "4" * 24,
        event="same",
        evidence=("b",),
        wording="A distinct bounded reconstruction snapshot",
    )
    memory = build_engineering_story_memory((original,))
    before = memory.to_json()

    result = canonicalize_engineering_story_memory(
        existing_memory=memory,
        new_records=(reworded,),
    )

    assert memory.to_json() == before
    serialized = result.to_json()
    assert original.engineering_story.mechanism.value not in serialized
    assert reworded.engineering_story.mechanism.value not in serialized
    assert not hasattr(result, "merged_story")


def test_information_artifacts_are_not_touched() -> None:
    information = Path("information")
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in information.rglob("*")
        if path.is_file()
    }
    _initial_result(_record("5" * 24, event="a", evidence=("a",)))
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in information.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_module_has_no_jd_ranking_retrieval_chroma_network_or_model_dependency() -> None:
    source_path = Path(matching_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = {
        "api_server",
        "chromadb",
        "chroma_",
        "project_query_planner",
        "project_retrieval",
        "evidence_hybrid_retrieval",
        "hiring",
        "resume",
        "clarification",
        "requests",
        "httpx",
        "openai",
        "anthropic",
    }
    assert not {
        item
        for item in imports
        if any(fragment in item.casefold() for fragment in forbidden)
    }


def test_no_versioning_invalidation_or_lifecycle_transition_api_is_added() -> None:
    names = set(dir(matching_module))
    assert not names.intersection({
        "increment_story_version",
        "invalidate_engineering_story",
        "revalidate_engineering_story",
        "transition_engineering_story_lifecycle",
    })


def test_semantic_names_do_not_embed_roadmap_numbers() -> None:
    source = Path(matching_module.__file__).read_text(encoding="utf-8").casefold()
    test_source = Path(__file__).read_text(encoding="utf-8").casefold()
    combined = source + test_source
    assert "phase" + "6_75" not in combined
    assert "phase" + "675" not in combined
    assert "phase" + "8_story_matcher" not in combined
