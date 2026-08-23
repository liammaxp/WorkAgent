from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest

import backend.engineering_story_memory as story_memory_module
from backend.engineering_story_clustering import (
    StoryCluster,
    cluster_story_evidence_bundle,
)
from backend.engineering_story_evidence import resolve_story_evidence_bundle
from backend.engineering_story_memory import (
    ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
    MAX_ENGINEERING_STORY_MEMORY_RECORDS,
    EngineeringStoryIdentity,
    EngineeringStoryIdentityState,
    EngineeringStoryMemory,
    EngineeringStoryMemoryError,
    EngineeringStoryMemoryErrorCode,
    EngineeringStoryMemoryLoadStatus,
    EngineeringStoryMemoryRecord,
    EngineeringStoryMemoryWriteStatus,
    add_engineering_story_memory_record,
    build_engineering_story_memory,
    build_engineering_story_memory_record,
    get_engineering_story_memory_record,
    load_engineering_story_memory,
    replace_engineering_story_memory_record,
    serialize_engineering_story_memory,
    write_engineering_story_memory,
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
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_opportunity import (
    StoryOpportunityDetectionResult,
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
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"
_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_VALUES = {
    EngineeringStoryFieldName.PROBLEM_CONTEXT: (
        "Requests could survive a partial service failure"
    ),
    EngineeringStoryFieldName.TRIGGER: (
        "A deterministic failure exposed the recovery gap"
    ),
    EngineeringStoryFieldName.BEFORE_STATE: (
        "The workflow stopped on the first failed operation"
    ),
    EngineeringStoryFieldName.DECISION: (
        "Use an explicit recovery boundary around the workflow"
    ),
    EngineeringStoryFieldName.MECHANISM: (
        "Bounded recovery with validation gates"
    ),
    EngineeringStoryFieldName.IMPLEMENTATION: (
        "Added typed retry state and deterministic checks"
    ),
    EngineeringStoryFieldName.TRADEOFF: (
        "Rejected unbounded retries to preserve predictable behavior"
    ),
    EngineeringStoryFieldName.VALIDATION: (
        "Failure-specific tests exercise recovery and rejection paths"
    ),
    EngineeringStoryFieldName.AFTER_STATE: (
        "Invalid recovery states fail closed"
    ),
    EngineeringStoryFieldName.OBSERVABLE_OUTCOME: (
        "The invalid state is rejected before persistence"
    ),
    EngineeringStoryFieldName.OWNERSHIP: (
        "Owned the recovery boundary and regression validation"
    ),
    EngineeringStoryFieldName.STAKEHOLDER_CONTEXT: (
        "Protected operators consuming the workflow"
    ),
}


@dataclass(frozen=True)
class _PipelineCase:
    cluster: StoryCluster
    reconstruction: StoryReconstructionResult
    sufficiency: EngineeringStorySufficiencyResult
    opportunity: StoryOpportunityDetectionResult


def _source_ref(
    evidence_id: str,
    *,
    project_id: str,
    change_id: str,
) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type="github_evidence_chunk",
        source_id=f"chunk_{evidence_id}",
        project_id=project_id,
        content_hash=sha256(
            f"{project_id}|{evidence_id}|{change_id}".encode()
        ).hexdigest(),
        repo="owner/workagent",
        commit_sha="aaaaaaa",
        file_path="backend/recovery.py",
        symbol="recover",
        metadata={"change_id": change_id},
    )


def _fact(
    evidence_id: str,
    *,
    project_id: str,
    change_id: str,
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
        )],
        evidence_type=EvidenceType.ARCHITECTURE,
        status=EvidenceStatus.ACCEPTED,
        confidence=Confidence.HIGH,
        metric_support=MetricSupport.NONE,
        technical_tags=["recovery", "validation"],
        quality_score=92,
    )


def _pipeline_case(
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change-main",
    quality: StoryReconstructionQuality = StoryReconstructionQuality.PARTIAL,
) -> _PipelineCase:
    fact = _fact(
        f"pef_{change_id.replace('-', '_')}",
        project_id=project_id,
        change_id=change_id,
    )
    built_boundary = build_project_evidence_claim_boundary(fact)
    assert built_boundary is not None
    boundary: ProjectClaimBoundary = built_boundary
    bundle = resolve_story_evidence_bundle(
        project_id=project_id,
        evidence_fact_ids=(fact.evidence_fact_id,),
        evidence_facts=(fact,),
        claim_boundary_ids=(boundary.boundary_id,),
        claim_boundaries=(boundary,),
    )
    clustering = cluster_story_evidence_bundle(bundle)
    assert len(clustering.clusters) == 1
    cluster = clustering.clusters[0]
    positive = {
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    }
    story_fields: dict[str, EngineeringStoryField] = {}
    decisions: list[StoryFieldReconstructionDecision] = []
    for name in _FIELD_ORDER:
        is_positive = name in positive
        field = EngineeringStoryField(
            value=_VALUES[name] if is_positive else None,
            evidence_state=(
                StoryFieldEvidenceState.CONFIRMED
                if is_positive
                else StoryFieldEvidenceState.PLAUSIBLE_MISSING
            ),
            evidence_fact_ids=(fact.evidence_fact_id,) if is_positive else (),
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
                else StoryFieldDecisionReason.MISSING_HUMAN_CONTEXT
            ),
        ))
    story = EngineeringStory(
        story_id=(
            "engineering_story_candidate_"
            + cluster.cluster_id.removeprefix("story_cluster_")
        ),
        project_id=project_id,
        story_type=EngineeringStoryType.ARCHITECTURE_CHANGE,
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
        identity_state=(
            StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
        ),
        field_decisions=tuple(reversed(decisions)),
        diagnostics=(),
        unresolved_fields=tuple(
            name for name in _FIELD_ORDER if name not in positive
        ),
    )
    project_memory = ProjectEvidenceMemory(
        project_id=project_id,
        project_name="WorkAgent",
        evidence_facts=[fact],
        capability_facts=[],
        claim_boundaries=[boundary],
    )
    sufficiency = evaluate_engineering_story_sufficiency(
        reconstruction_result=reconstruction,
        project_memory=project_memory,
    )
    opportunity = detect_story_opportunity(
        reconstruction_result=reconstruction,
        sufficiency_result=sufficiency,
        story_cluster=cluster,
    )
    return _PipelineCase(cluster, reconstruction, sufficiency, opportunity)


def _record(case: _PipelineCase | None = None) -> EngineeringStoryMemoryRecord:
    selected = case or _pipeline_case()
    return build_engineering_story_memory_record(
        story_cluster=selected.cluster,
        reconstruction_result=selected.reconstruction,
        sufficiency_result=selected.sufficiency,
        opportunity_result=selected.opportunity,
    )


def _replace_story(
    record: EngineeringStoryMemoryRecord,
    story: EngineeringStory,
    **changes: object,
) -> EngineeringStoryMemoryRecord:
    return replace(
        record,
        engineering_story=story,
        record_fingerprint="",
        **changes,
    )


def test_builds_valid_record_from_the_accepted_result_chain() -> None:
    case = _pipeline_case()
    record = _record(case)

    assert record.engineering_story == case.opportunity.evaluated_story
    assert record.claim_sufficiency == case.sufficiency.claim_sufficiency
    assert record.story_sufficiency == case.sufficiency.story_sufficiency
    assert record.story_opportunity == case.opportunity.story_opportunity
    assert record.provenance.evidence_fact_ids == case.cluster.member_evidence_fact_ids
    assert record.provenance.source_lineage_fingerprints


def test_cross_story_result_chain_fails_closed() -> None:
    first = _pipeline_case(change_id="change-first")
    second = _pipeline_case(change_id="change-second")

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory_record(
            story_cluster=first.cluster,
            reconstruction_result=first.reconstruction,
            sufficiency_result=second.sufficiency,
            opportunity_result=first.opportunity,
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CLUSTER_MISMATCH


def test_cross_project_result_chain_fails_closed() -> None:
    first = _pipeline_case()
    foreign = _pipeline_case(
        project_id=OTHER_PROJECT_ID,
        change_id="change-foreign",
    )

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory_record(
            story_cluster=first.cluster,
            reconstruction_result=first.reconstruction,
            sufficiency_result=foreign.sufficiency,
            opportunity_result=first.opportunity,
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CROSS_PROJECT_INPUT


def test_cluster_mismatch_fails_closed() -> None:
    first = _pipeline_case(change_id="change-first")
    second = _pipeline_case(change_id="change-second")

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory_record(
            story_cluster=second.cluster,
            reconstruction_result=first.reconstruction,
            sufficiency_result=first.sufficiency,
            opportunity_result=first.opportunity,
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CLUSTER_MISMATCH


def test_reconstruction_quality_mismatch_fails_closed() -> None:
    case = _pipeline_case()
    mismatched = replace(
        case.sufficiency,
        reconstruction_quality=StoryReconstructionQuality.COMPLETE,
    )

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory_record(
            story_cluster=case.cluster,
            reconstruction_result=case.reconstruction,
            sufficiency_result=mismatched,
            opportunity_result=case.opportunity,
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH


def test_candidate_identity_is_explicitly_provisional_and_canonical_is_unresolved() -> None:
    record = _record()

    assert record.identity.identity_state is EngineeringStoryIdentityState.PROVISIONAL
    assert record.identity.reconstruction_identity_state is (
        StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
    )
    assert record.candidate_story_id.startswith("engineering_story_candidate_")
    assert record.canonical_story_id is None
    assert record.identity.canonical_story_id is None


def test_canonical_identity_cannot_be_promoted_by_a_caller() -> None:
    identity = _record().identity

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        replace(identity, canonical_story_id="engineering_story_not_canonical")
    assert exc.value.code is (
        EngineeringStoryMemoryErrorCode.CANONICAL_IDENTITY_UNRESOLVED
    )


def test_candidate_and_cluster_digests_must_match() -> None:
    identity = _record().identity

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        replace(
            identity,
            candidate_story_id="engineering_story_candidate_0123456789abcdef01234567",
            identity_basis_fingerprint="",
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CLUSTER_MISMATCH


def test_wording_changes_content_not_identity() -> None:
    record = _record()
    changed_field = replace(
        record.engineering_story.mechanism,
        value="Bounded typed recovery coordinated by explicit validation gates",
    )
    changed_story = replace(record.engineering_story, mechanism=changed_field)
    changed = _replace_story(record, changed_story)

    assert changed.candidate_story_id == record.candidate_story_id
    assert changed.identity.identity_basis_fingerprint == (
        record.identity.identity_basis_fingerprint
    )
    assert changed.record_fingerprint != record.record_fingerprint


def test_sufficiency_changes_content_not_identity() -> None:
    record = _record()
    claim = replace(record.claim_sufficiency, level=SufficiencyLevel.MEDIUM)
    changed_story = replace(record.engineering_story, claim_sufficiency=claim)
    changed = _replace_story(record, changed_story)

    assert changed.identity == record.identity
    assert changed.record_fingerprint != record.record_fingerprint


def test_opportunity_changes_content_not_identity() -> None:
    record = _record()
    opportunity = replace(
        record.story_opportunity,
        level=StoryOpportunityLevel.MEDIUM,
    )
    changed_story = replace(record.engineering_story, opportunity=opportunity)
    changed = _replace_story(record, changed_story)

    assert changed.identity == record.identity
    assert changed.record_fingerprint != record.record_fingerprint


def test_reconstruction_quality_changes_content_not_identity() -> None:
    record = _record()
    changed = replace(
        record,
        reconstruction_quality=StoryReconstructionQuality.MINIMAL,
        record_fingerprint="",
    )

    assert changed.identity == record.identity
    assert changed.record_fingerprint != record.record_fingerprint


def test_lifecycle_is_a_stored_snapshot_without_transition_logic() -> None:
    record = _record()
    stale_story = replace(
        record.engineering_story,
        lifecycle=EngineeringStoryLifecycle(
            status=EngineeringStoryStatus.STALE,
            requires_revalidation=True,
        ),
    )
    stale = _replace_story(record, stale_story)

    assert stale.engineering_story.lifecycle.status is EngineeringStoryStatus.STALE
    assert stale.identity == record.identity
    assert not hasattr(story_memory_module, "transition_engineering_story")


def test_envelope_order_and_fingerprint_are_input_order_independent() -> None:
    first = _record(_pipeline_case(change_id="change-first"))
    second = _record(_pipeline_case(change_id="change-second"))

    forward = build_engineering_story_memory((first, second))
    reverse = build_engineering_story_memory((second, first))

    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert forward.logical_fingerprint == reverse.logical_fingerprint


def test_exact_duplicate_candidate_is_idempotent() -> None:
    record = _record()
    memory = build_engineering_story_memory((record, record))

    assert memory.records == (record,)
    assert add_engineering_story_memory_record(memory, record) == memory


def test_conflicting_duplicate_candidate_fails_closed() -> None:
    record = _record()
    changed_field = replace(
        record.engineering_story.implementation,
        value="Added typed recovery state and bounded failure checks",
    )
    changed = _replace_story(
        record,
        replace(record.engineering_story, implementation=changed_field),
    )

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory((record, changed))
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CONFLICTING_CANDIDATE_ID


def test_different_candidates_are_not_semantically_merged() -> None:
    architecture = _record(_pipeline_case(change_id="architecture-migration"))
    readiness = _record(_pipeline_case(change_id="readiness-hardening"))
    memory = build_engineering_story_memory((architecture, readiness))

    assert len(memory.records) == 2
    assert {item.candidate_story_id for item in memory.records} == {
        architecture.candidate_story_id,
        readiness.candidate_story_id,
    }


def test_memory_schema_and_json_round_trip_are_strict_and_deterministic() -> None:
    memory = build_engineering_story_memory((_record(),))
    payload = memory.to_dict()

    assert memory.schema_version == ENGINEERING_STORY_MEMORY_SCHEMA_VERSION
    assert EngineeringStoryMemory.from_dict(payload) == memory
    assert json.loads(memory.to_json()) == payload
    assert serialize_engineering_story_memory(memory) == (
        serialize_engineering_story_memory(
            EngineeringStoryMemory.from_dict(payload)
        )
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("unknown",), True),
        (("records", 0, "unknown"), True),
        (("records", 0, "identity", "unknown"), True),
        (("records", 0, "provenance", "unknown"), True),
    ),
)
def test_unknown_fields_fail_closed(
    path: tuple[object, ...],
    value: object,
) -> None:
    payload = build_engineering_story_memory((_record(),)).to_dict()
    target: object = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown"):
        EngineeringStoryMemory.from_dict(payload)


def test_unknown_schema_version_fails_closed() -> None:
    payload = build_engineering_story_memory().to_dict()
    payload["schema_version"] = "engineering_story_memory.v999"

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        EngineeringStoryMemory.from_dict(payload)
    assert exc.value.code is EngineeringStoryMemoryErrorCode.UNSUPPORTED_SCHEMA_VERSION


@pytest.mark.parametrize(
    "fingerprint_path",
    (
        ("logical_fingerprint",),
        ("records", 0, "record_fingerprint"),
        ("records", 0, "identity", "identity_basis_fingerprint"),
        ("records", 0, "provenance", "provenance_fingerprint"),
    ),
)
def test_integrity_fingerprint_mismatch_fails_closed(
    fingerprint_path: tuple[object, ...],
) -> None:
    payload = build_engineering_story_memory((_record(),)).to_dict()
    target: object = payload
    for part in fingerprint_path[:-1]:
        target = target[part]  # type: ignore[index]
    target[fingerprint_path[-1]] = "0" * 64  # type: ignore[index]

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        EngineeringStoryMemory.from_dict(payload)
    assert exc.value.code is (
        EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
    )


def test_explicit_path_write_and_load_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "story-memory.json"
    memory = build_engineering_story_memory((_record(),))

    written = write_engineering_story_memory(destination, memory)
    loaded = load_engineering_story_memory(destination)
    unchanged = write_engineering_story_memory(destination, memory)

    assert written.status is EngineeringStoryMemoryWriteStatus.CREATED
    assert written.round_trip_validated is True
    assert loaded.status is EngineeringStoryMemoryLoadStatus.READY
    assert loaded.memory == memory
    assert unchanged.status is EngineeringStoryMemoryWriteStatus.UNCHANGED


def test_empty_memory_write_and_load_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "empty-story-memory.json"
    memory = build_engineering_story_memory()

    assert write_engineering_story_memory(destination, memory).status is (
        EngineeringStoryMemoryWriteStatus.CREATED
    )
    loaded = load_engineering_story_memory(destination)
    assert loaded.status is EngineeringStoryMemoryLoadStatus.EMPTY
    assert loaded.memory == memory


def test_missing_and_malformed_artifacts_fail_closed(tmp_path: Path) -> None:
    missing = load_engineering_story_memory(tmp_path / "missing.json")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    malformed = load_engineering_story_memory(malformed_path)

    assert missing.status is EngineeringStoryMemoryLoadStatus.MISSING
    assert missing.error_code is EngineeringStoryMemoryErrorCode.ARTIFACT_MISSING
    assert malformed.status is EngineeringStoryMemoryLoadStatus.INVALID
    assert malformed.memory is None


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate-key.json"
    destination.write_text(
        '{"schema_version":"engineering_story_memory.v1",'
        '"schema_version":"engineering_story_memory.v1",'
        '"records":[],"logical_fingerprint":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )

    loaded = load_engineering_story_memory(destination)
    assert loaded.status is EngineeringStoryMemoryLoadStatus.INVALID
    assert loaded.memory is None


def test_unknown_version_load_has_distinct_status(tmp_path: Path) -> None:
    destination = tmp_path / "future.json"
    payload = build_engineering_story_memory().to_dict()
    payload["schema_version"] = "engineering_story_memory.v2"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_engineering_story_memory(destination)
    assert loaded.status is EngineeringStoryMemoryLoadStatus.UNSUPPORTED_VERSION
    assert loaded.error_code is (
        EngineeringStoryMemoryErrorCode.UNSUPPORTED_SCHEMA_VERSION
    )


def test_integrity_mismatch_load_has_distinct_status(tmp_path: Path) -> None:
    destination = tmp_path / "tampered.json"
    payload = build_engineering_story_memory((_record(),)).to_dict()
    payload["logical_fingerprint"] = "0" * 64
    destination.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_engineering_story_memory(destination)
    assert loaded.status is EngineeringStoryMemoryLoadStatus.INTEGRITY_MISMATCH
    assert loaded.memory is None


def test_failed_atomic_replace_preserves_previous_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "story-memory.json"
    first = build_engineering_story_memory((
        _record(_pipeline_case(change_id="change-first")),
    ))
    second = build_engineering_story_memory((
        _record(_pipeline_case(change_id="change-second")),
    ))
    assert write_engineering_story_memory(destination, first).status is (
        EngineeringStoryMemoryWriteStatus.CREATED
    )
    before = destination.read_bytes()

    def _fail_replace(staged: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        story_memory_module,
        "_replace_staged_engineering_story_memory",
        _fail_replace,
    )
    result = write_engineering_story_memory(destination, second)

    assert result.status is EngineeringStoryMemoryWriteStatus.FAILED
    assert result.previous_artifact_preserved is True
    assert destination.read_bytes() == before
    assert load_engineering_story_memory(destination).memory == first
    assert not tuple(tmp_path.glob("*.stage"))


def test_persistence_requires_an_explicit_caller_path() -> None:
    load_path = inspect.signature(load_engineering_story_memory).parameters["path"]
    write_path = inspect.signature(write_engineering_story_memory).parameters["path"]

    assert load_path.default is inspect.Parameter.empty
    assert write_path.default is inspect.Parameter.empty
    assert not hasattr(story_memory_module, "DEFAULT_ENGINEERING_STORY_MEMORY_PATH")


def test_information_artifacts_are_not_touched_by_construction() -> None:
    information = Path("information")
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in information.rglob("*")
        if path.is_file()
    }

    build_engineering_story_memory((_record(),))

    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in information.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "unsafe_value",
    (
        "diff --git a/secret.py b/secret.py + access_token=topsecret",
        "-----BEGIN PRIVATE KEY-----",
        "password: topsecretvalue",
        "Authorization: Bearer abcdefghijklmnop",
    ),
)
def test_raw_or_secret_story_values_cannot_enter_memory(unsafe_value: str) -> None:
    record = _record()

    with pytest.raises(ValueError, match="raw or sensitive"):
        replace(record.engineering_story.mechanism, value=unsafe_value)


def test_serialized_memory_contains_only_safe_hashed_source_lineage() -> None:
    serialized = _record().to_json()

    assert "source_lineage_fingerprints" in serialized
    assert "backend/recovery.py" not in serialized
    assert "owner/workagent" not in serialized
    assert "raw_patch" not in serialized
    assert "embedding" not in serialized
    assert "file_path" not in serialized


def test_records_and_envelope_are_frozen_slotted_and_do_not_mutate_inputs() -> None:
    record = _record()
    supplied = [record]
    memory = build_engineering_story_memory(supplied)
    supplied.clear()

    assert memory.records == (record,)
    assert not hasattr(memory, "__dict__")
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        memory.records = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        record.record_fingerprint = "0" * 64  # type: ignore[misc]


def test_bounds_fail_explicitly_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    with pytest.raises(EngineeringStoryMemoryError) as exc:
        build_engineering_story_memory(
            (record,) * (MAX_ENGINEERING_STORY_MEMORY_RECORDS + 1)
        )
    assert exc.value.code is (
        EngineeringStoryMemoryErrorCode.MAXIMUM_RECORD_COUNT_EXCEEDED
    )

    monkeypatch.setattr(
        story_memory_module,
        "MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE",
        1,
    )
    with pytest.raises(EngineeringStoryMemoryError) as size_exc:
        build_engineering_story_memory((record,))
    assert size_exc.value.code is (
        EngineeringStoryMemoryErrorCode.MAXIMUM_SERIALIZED_SIZE_EXCEEDED
    )


def test_pure_add_get_and_replace_helpers_have_explicit_collision_semantics() -> None:
    first = _record(_pipeline_case(change_id="change-first"))
    second = _record(_pipeline_case(change_id="change-second"))
    memory = add_engineering_story_memory_record(
        build_engineering_story_memory((first,)),
        second,
    )

    assert get_engineering_story_memory_record(memory, first.candidate_story_id) == first
    assert get_engineering_story_memory_record(memory, second.candidate_story_id) == second
    changed_field = replace(
        first.engineering_story.mechanism,
        value="Bounded recovery coordinated by typed validation gates",
    )
    changed = _replace_story(
        first,
        replace(first.engineering_story, mechanism=changed_field),
    )
    replaced = replace_engineering_story_memory_record(memory, changed)
    assert get_engineering_story_memory_record(
        replaced,
        first.candidate_story_id,
    ) == changed
    assert memory != replaced


def test_replace_unknown_candidate_fails_closed() -> None:
    first = _record(_pipeline_case(change_id="change-first"))
    second = _record(_pipeline_case(change_id="change-second"))

    with pytest.raises(EngineeringStoryMemoryError) as exc:
        replace_engineering_story_memory_record(
            build_engineering_story_memory((first,)),
            second,
        )
    assert exc.value.code is EngineeringStoryMemoryErrorCode.CANDIDATE_NOT_FOUND


def test_identity_basis_excludes_all_downstream_and_presentation_state() -> None:
    keys = set(_record().identity.to_dict())

    assert keys == {
        "project_id",
        "candidate_story_id",
        "cluster_id",
        "event_core_fingerprint",
        "identity_state",
        "cluster_identity_state",
        "reconstruction_identity_state",
        "canonical_story_id",
        "identity_basis_fingerprint",
    }
    assert not keys.intersection({
        "story_title",
        "story_wording",
        "claim_sufficiency",
        "story_sufficiency",
        "opportunity",
        "resume_relevance",
        "jd",
        "company",
        "reconstruction_quality",
    })


def test_module_has_no_downstream_retrieval_chroma_network_or_model_dependency() -> None:
    source_path = Path(story_memory_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden_fragments = {
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
        imported
        for imported in imports
        if any(fragment in imported.casefold() for fragment in forbidden_fragments)
    }


def test_semantic_names_do_not_embed_roadmap_numbers() -> None:
    source = Path(story_memory_module.__file__).read_text(encoding="utf-8").casefold()
    test_source = Path(__file__).read_text(encoding="utf-8").casefold()

    combined = source + test_source
    assert "phase" + "6_75" not in combined
    assert "phase" + "675" not in combined
    assert "phase" + "7_story_memory" not in combined
