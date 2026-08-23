from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import backend.engineering_story_memory_service as service
from backend.engineering_story_clustering import StoryClusterIdentityState
from backend.engineering_story_lifecycle import (
    EngineeringStoryLifecycleMemory,
    apply_engineering_story_revisions,
)
from backend.engineering_story_matching import (
    EngineeringStoryIdentityMap,
    canonicalize_engineering_story_memory,
)
from backend.engineering_story_memory import (
    EngineeringStoryIdentity,
    EngineeringStoryIdentityState,
    EngineeringStoryMemoryProvenance,
    EngineeringStoryMemoryRecord,
    build_engineering_story_memory,
)
from backend.engineering_story_memory_service import (
    AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
    AuthoritativeEngineeringStoryMemory,
    EngineeringStoryMemoryServiceError,
    EngineeringStorySourceArtifact,
    StoryMemoryArtifactStatus,
    StoryMemoryDiagnosticCode,
    StoryMemoryOperation,
    StoryMemoryReadinessState,
    StoryMemoryServiceErrorCode,
    StoryMemoryWriteStatus,
    build_and_materialize_authoritative_engineering_story_memory,
    build_authoritative_engineering_story_memory,
    get_current_engineering_story_revision,
    get_engineering_stories_for_project,
    get_engineering_story_by_id,
    inspect_authoritative_engineering_story_memory_readiness,
    load_authoritative_engineering_story_memory,
    refresh_and_materialize_authoritative_engineering_story_memory,
    refresh_authoritative_engineering_story_memory,
    resolve_engineering_story_id,
    serialize_authoritative_engineering_story_memory,
    validate_authoritative_engineering_story_memory,
    write_authoritative_engineering_story_memory,
)
from backend.engineering_story_models import EngineeringStoryStatus
from backend.engineering_story_reconstruction import StoryReconstructionIdentityState
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_memory import (
    ProjectEvidenceMemoryDiagnostics,
    build_project_evidence_memories,
    build_project_evidence_memory_snapshot,
    persist_project_evidence_memory,
)
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectEvidenceFact,
)


PROJECT_ID = "workagent"


def _ref(
    source_id: str,
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change_main",
    symbol: str = "build_memory",
) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type="github_evidence_chunk",
        source_id=source_id,
        project_id=project_id,
        content_hash=sha256(
            f"{source_id}|{project_id}|{change_id}|{symbol}".encode()
        ).hexdigest(),
        repo="owner/workagent",
        commit_sha="a" * 40,
        file_path="backend/story_service.py",
        symbol=symbol,
        metadata={"change_id": change_id},
    )


def _fact(
    token: str,
    *,
    project_id: str = PROJECT_ID,
    change_id: str = "change_main",
    mechanism: str = "Materialized bounded semantic story memory",
    evidence_type: EvidenceType = EvidenceType.ARCHITECTURE,
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=f"pef_{token}",
        mechanism=mechanism,
        implementation=[
            "Implemented deterministic canonical story memory materialization"
        ],
        safe_impact=[],
        source_refs=[
            _ref(
                f"source_{token}",
                project_id=project_id,
                change_id=change_id,
                symbol=f"symbol_{token}",
            )
        ],
        evidence_type=evidence_type,
        status=status,
        confidence=Confidence.HIGH,
        metric_support=MetricSupport.NONE,
        technical_tags=["Python", "memory"],
        quality_score=90,
    )


def _snapshot(*facts: ProjectEvidenceFact):
    boundaries = tuple(
        boundary
        for fact in facts
        if (boundary := build_project_evidence_claim_boundary(fact)) is not None
    )
    memories, report = build_project_evidence_memories(
        facts,
        (),
        boundaries,
    )
    diagnostics = ProjectEvidenceMemoryDiagnostics(
        evidence_fact_count=report.evidence_fact_count,
        capability_fact_count=report.capability_fact_count,
        claim_boundary_count=report.claim_boundary_count,
        allowed_claim_count=sum(len(item.allowed_claims) for item in boundaries),
        forbidden_claim_count=sum(len(item.forbidden_claims) for item in boundaries),
    )
    return build_project_evidence_memory_snapshot(
        memories,
        diagnostics=diagnostics,
    )


def _write_upstream(path: Path, *facts: ProjectEvidenceFact) -> None:
    result = persist_project_evidence_memory(_snapshot(*facts), path=path)
    assert result.status in {"created", "updated", "unchanged"}


def _single_story_memory(*facts: ProjectEvidenceFact):
    result = build_authoritative_engineering_story_memory(_snapshot(*facts))
    assert result.canonical_story_count == 1
    return result


def _source_descriptor(snapshot) -> EngineeringStorySourceArtifact:
    return EngineeringStorySourceArtifact(
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        project_count=len(snapshot.projects),
        evidence_fact_count=sum(
            len(item.evidence_facts) for item in snapshot.projects
        ),
        capability_fact_count=sum(
            len(item.capability_facts) for item in snapshot.projects
        ),
        claim_boundary_count=sum(
            len(item.claim_boundaries) for item in snapshot.projects
        ),
    )


def _combined_record(
    left: EngineeringStoryMemoryRecord,
    right: EngineeringStoryMemoryRecord,
    *,
    token: str,
    event_fingerprint: str,
) -> EngineeringStoryMemoryRecord:
    candidate_id = f"engineering_story_candidate_{token}"
    cluster_id = f"story_cluster_{token}"
    evidence_ids = tuple(sorted({
        *left.provenance.evidence_fact_ids,
        *right.provenance.evidence_fact_ids,
    }))
    capability_ids = tuple(sorted({
        *left.provenance.capability_fact_ids,
        *right.provenance.capability_fact_ids,
    }))
    boundary_ids = tuple(sorted({
        *left.provenance.claim_boundary_ids,
        *right.provenance.claim_boundary_ids,
    }))
    lineages = tuple(sorted({
        *left.provenance.source_lineage_fingerprints,
        *right.provenance.source_lineage_fingerprints,
    }))
    identity = EngineeringStoryIdentity(
        project_id=left.project_id,
        candidate_story_id=candidate_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        identity_state=EngineeringStoryIdentityState.PROVISIONAL,
        cluster_identity_state=StoryClusterIdentityState.STABLE_EVENT_CORE,
        reconstruction_identity_state=(
            StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
        ),
    )
    story = replace(
        left.engineering_story,
        story_id=candidate_id,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
    )
    provenance = EngineeringStoryMemoryProvenance(
        project_id=left.project_id,
        cluster_id=cluster_id,
        event_core_fingerprint=event_fingerprint,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
        source_lineage_fingerprints=lineages,
    )
    return EngineeringStoryMemoryRecord(
        identity=identity,
        engineering_story=story,
        provenance=provenance,
        reconstruction_quality=left.reconstruction_quality,
        reconstruction_diagnostics=left.reconstruction_diagnostics,
        reconstruction_unresolved_fields=left.reconstruction_unresolved_fields,
        sufficiency_diagnostics=left.sufficiency_diagnostics,
        opportunity_signal_decisions=left.opportunity_signal_decisions,
        opportunity_diagnostics=left.opportunity_diagnostics,
        related_project_cluster_ids=left.related_project_cluster_ids,
    )


def test_empty_and_zero_capability_build_is_valid_deterministic_v2() -> None:
    snapshot = _snapshot()
    first = build_authoritative_engineering_story_memory(snapshot)
    second = build_authoritative_engineering_story_memory(snapshot)

    assert first.memory == second.memory
    assert first.memory.schema_version == AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION
    assert first.memory.source_artifact.capability_fact_count == 0
    assert first.canonical_story_count == 0
    assert first.memory.logical_fingerprint == second.memory.logical_fingerprint
    assert validate_authoritative_engineering_story_memory(first.memory).valid


def test_build_orchestrates_story_and_exposes_safe_default_view() -> None:
    result = _single_story_memory(_fact("one"))
    history = result.memory.histories[0]
    canonical_id = history.canonical_story_id

    view = get_engineering_story_by_id(result.memory, canonical_id)
    assert view is not None
    assert view.canonical_story_id == canonical_id
    assert view.current_story.story_id == canonical_id
    assert view.lifecycle.status is EngineeringStoryStatus.ACTIVE
    assert view.lifecycle.requires_revalidation is False
    assert get_engineering_stories_for_project(result.memory, PROJECT_ID) == (view,)
    assert get_current_engineering_story_revision(result.memory, canonical_id) == history.current_revision
    serialized = view.to_json()
    for forbidden in ("raw_patch", "document", "file_path", "embedding", "jd_score"):
        assert forbidden not in serialized


def test_identical_refresh_is_idempotent() -> None:
    snapshot = _snapshot(_fact("one"))
    built = build_authoritative_engineering_story_memory(snapshot)
    refreshed = refresh_authoritative_engineering_story_memory(
        existing_memory=built.memory,
        snapshot=snapshot,
    )

    assert refreshed.memory == built.memory
    assert refreshed.memory.logical_fingerprint == built.memory.logical_fingerprint
    assert refreshed.new_revision_count == 0
    assert refreshed.new_canonical_count == 0
    assert refreshed.updated_canonical_count == 0
    assert refreshed.unchanged_canonical_count == 1


def test_supporting_evidence_enrichment_preserves_canonical_identity() -> None:
    initial_snapshot = _snapshot(_fact("one"))
    initial = build_authoritative_engineering_story_memory(initial_snapshot)
    enriched_snapshot = _snapshot(
        _fact("one"),
        _fact(
            "two",
            mechanism="Validated atomic reload before authoritative replacement",
            evidence_type=EvidenceType.VALIDATION,
        ),
    )
    refreshed = refresh_authoritative_engineering_story_memory(
        existing_memory=initial.memory,
        snapshot=enriched_snapshot,
    )

    assert len(refreshed.memory.histories) == 1
    assert (
        refreshed.memory.histories[0].canonical_story_id
        == initial.memory.histories[0].canonical_story_id
    )
    assert refreshed.new_revision_count == 1
    assert len(refreshed.memory.histories[0].revisions) == 2


def test_removed_authoritative_candidate_becomes_stale_and_hidden_by_default() -> None:
    initial = _single_story_memory(_fact("one"))
    refreshed = refresh_authoritative_engineering_story_memory(
        existing_memory=initial.memory,
        snapshot=_snapshot(),
    )
    history = refreshed.memory.histories[0]

    assert history.current_revision.lifecycle.status is EngineeringStoryStatus.STALE
    assert history.current_revision.lifecycle.requires_revalidation is True
    assert get_engineering_story_by_id(
        refreshed.memory, history.canonical_story_id
    ) is None
    assert get_engineering_story_by_id(
        refreshed.memory,
        history.canonical_story_id,
        include_non_active=True,
    ) is not None
    assert get_engineering_stories_for_project(refreshed.memory, PROJECT_ID) == ()


def test_new_distinct_event_adds_canonical_story_without_rewriting_existing() -> None:
    initial = _single_story_memory(_fact("one", change_id="change_one"))
    refreshed = refresh_authoritative_engineering_story_memory(
        existing_memory=initial.memory,
        snapshot=_snapshot(
            _fact("one", change_id="change_one"),
            _fact("two", change_id="change_two"),
        ),
    )

    assert refreshed.canonical_story_count == 2
    assert refreshed.new_canonical_count == 1
    assert initial.memory.histories[0].canonical_story_id in {
        item.canonical_story_id for item in refreshed.memory.histories
    }


def test_authoritative_envelope_preserves_strong_merge_alias_and_alias_reader() -> None:
    left = _single_story_memory(
        _fact("left", change_id="change_left")
    ).memory.histories[0].current_revision.record
    right = _single_story_memory(
        _fact("right", change_id="change_right")
    ).memory.histories[0].current_revision.record
    initial_candidates = build_engineering_story_memory((left, right))
    initial_canonical = canonicalize_engineering_story_memory(
        existing_memory=initial_candidates
    )
    initial_lifecycle = apply_engineering_story_revisions(
        canonicalization_result=initial_canonical,
        new_records=(left, right),
    )
    merged_record = _combined_record(
        left,
        right,
        token="ab" * 12,
        event_fingerprint=sha256(b"merged-event").hexdigest(),
    )
    merged_canonical = canonicalize_engineering_story_memory(
        existing_memory=initial_candidates,
        new_records=(merged_record,),
        existing_identity_map=initial_lifecycle.memory.identity_map,
    )
    merged_lifecycle = apply_engineering_story_revisions(
        canonicalization_result=merged_canonical,
        new_records=(merged_record,),
        existing_lifecycle_memory=initial_lifecycle.memory,
    )
    snapshot = _snapshot(
        _fact("left", change_id="change_left"),
        _fact("right", change_id="change_right"),
    )
    memory = AuthoritativeEngineeringStoryMemory(
        schema_version=AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
        source_artifact=_source_descriptor(snapshot),
        identity_map=merged_lifecycle.memory.identity_map,
        histories=merged_lifecycle.memory.histories,
    )

    assert len(memory.identity_map.aliases) == 1
    alias = memory.identity_map.aliases[0]
    assert resolve_engineering_story_id(memory, alias.alias_story_id) == alias.canonical_story_id
    assert get_engineering_story_by_id(memory, alias.alias_story_id) is not None
    losing_history = memory.history_for(alias.alias_story_id)
    assert losing_history is not None
    assert losing_history.current_revision.lifecycle.status is EngineeringStoryStatus.SUPERSEDED


def test_authoritative_envelope_preserves_clear_split_lineage() -> None:
    left = _single_story_memory(
        _fact("left", change_id="change_left")
    ).memory.histories[0].current_revision.record
    right = _single_story_memory(
        _fact("right", change_id="change_right")
    ).memory.histories[0].current_revision.record
    parent_event = sha256(b"parent-event").hexdigest()
    parent = _combined_record(
        left,
        right,
        token="ac" * 12,
        event_fingerprint=parent_event,
    )
    initial_candidates = build_engineering_story_memory((parent,))
    initial_canonical = canonicalize_engineering_story_memory(
        existing_memory=initial_candidates
    )
    initial_lifecycle = apply_engineering_story_revisions(
        canonicalization_result=initial_canonical,
        new_records=(parent,),
    )
    owner = _combined_record(
        left,
        left,
        token="ad" * 12,
        event_fingerprint=parent_event,
    )
    child = _combined_record(
        right,
        right,
        token="ae" * 12,
        event_fingerprint=sha256(b"child-event").hexdigest(),
    )
    split_canonical = canonicalize_engineering_story_memory(
        existing_memory=initial_candidates,
        new_records=(owner, child),
        existing_identity_map=initial_lifecycle.memory.identity_map,
    )
    split_lifecycle = apply_engineering_story_revisions(
        canonicalization_result=split_canonical,
        new_records=(owner, child),
        existing_lifecycle_memory=initial_lifecycle.memory,
    )
    snapshot = _snapshot(
        _fact("left", change_id="change_left"),
        _fact("right", change_id="change_right"),
    )
    memory = AuthoritativeEngineeringStoryMemory(
        schema_version=AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
        source_artifact=_source_descriptor(snapshot),
        identity_map=split_lifecycle.memory.identity_map,
        histories=split_lifecycle.memory.histories,
    )

    assert len(memory.identity_map.split_relationships) == 1
    split = memory.identity_map.split_relationships[0]
    assert len(split.child_canonical_story_ids) == 2
    assert len(memory.histories) == 2
    assert set(split.child_canonical_story_ids) == {
        item.canonical_story_id for item in memory.histories
    }


def test_noncanonical_upstream_project_is_not_guessed_or_materialized() -> None:
    result = build_authoritative_engineering_story_memory(
        _snapshot(_fact("legacy", project_id="owner/workagent"))
    )
    assert result.canonical_story_count == 0
    assert result.skipped_noncanonical_project_count == 1
    assert (
        StoryMemoryDiagnosticCode.NONCANONICAL_UPSTREAM_PROJECT_SKIPPED
        in result.diagnostics
    )


def test_canonical_round_trip_and_strict_corruption_detection(tmp_path: Path) -> None:
    memory = _single_story_memory(_fact("one")).memory
    path = tmp_path / "story.json"
    write = write_authoritative_engineering_story_memory(
        memory,
        path=path,
        operation=StoryMemoryOperation.BUILD,
    )
    assert write.status is StoryMemoryWriteStatus.CREATED
    assert write.round_trip_validated is True
    loaded = load_authoritative_engineering_story_memory(path)
    assert loaded.status is StoryMemoryArtifactStatus.READY
    assert loaded.memory == memory
    assert path.read_bytes() == serialize_authoritative_engineering_story_memory(memory)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["logical_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_authoritative_engineering_story_memory(path).status is StoryMemoryArtifactStatus.INTEGRITY_MISMATCH

    payload["schema_version"] = "engineering_story_memory.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_authoritative_engineering_story_memory(path).status is StoryMemoryArtifactStatus.UNSUPPORTED_VERSION

    path.write_text("{not-json", encoding="utf-8")
    assert load_authoritative_engineering_story_memory(path).status is StoryMemoryArtifactStatus.INVALID


@pytest.mark.parametrize(
    "mutation",
    (
        "alias_cycle",
        "orphan_alias",
        "cross_project_alias",
        "orphan_split_child",
        "missing_current_revision",
        "revision_cycle",
    ),
)
def test_corrupt_identity_and_revision_graphs_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    memory = build_authoritative_engineering_story_memory(
        _snapshot(
            _fact("one", change_id="change_one"),
            _fact("two", change_id="change_two"),
        )
    ).memory
    assert len(memory.histories) == 2
    payload = memory.to_dict()
    identities = payload["identity_map"]["canonical_identities"]
    histories = payload["histories"]
    first_id = identities[0]["canonical_story_id"]
    second_id = identities[1]["canonical_story_id"]
    first_candidate = histories[0]["revisions"][0]["record"]["identity"][
        "candidate_story_id"
    ]
    second_candidate = histories[1]["revisions"][0]["record"]["identity"][
        "candidate_story_id"
    ]
    if mutation == "alias_cycle":
        payload["identity_map"]["aliases"] = [
            {
                "project_id": PROJECT_ID,
                "alias_story_id": first_id,
                "canonical_story_id": second_id,
            },
            {
                "project_id": PROJECT_ID,
                "alias_story_id": second_id,
                "canonical_story_id": first_id,
            },
        ]
    elif mutation == "orphan_alias":
        payload["identity_map"]["aliases"] = [
            {
                "project_id": PROJECT_ID,
                "alias_story_id": first_id,
                "canonical_story_id": "engineering_story_" + "f" * 24,
            }
        ]
    elif mutation == "cross_project_alias":
        payload["identity_map"]["aliases"] = [
            {
                "project_id": "other",
                "alias_story_id": first_id,
                "canonical_story_id": second_id,
            }
        ]
    elif mutation == "orphan_split_child":
        payload["identity_map"]["split_relationships"] = [
            {
                "project_id": PROJECT_ID,
                "parent_canonical_story_id": first_id,
                "child_candidate_story_ids": [first_candidate, second_candidate],
                "child_canonical_story_ids": [
                    first_id,
                    "engineering_story_" + "f" * 24,
                ],
                "retained_parent_candidate_story_id": first_candidate,
                "outcome": "split",
            }
        ]
    elif mutation == "missing_current_revision":
        histories[0]["current_revision_id"] = (
            "engineering_story_revision_" + "f" * 24
        )
    else:
        revision = histories[0]["revisions"][0]
        revision["parent_revision_id"] = revision["revision_id"]
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_authoritative_engineering_story_memory(path)
    assert loaded.memory is None
    assert loaded.status in {
        StoryMemoryArtifactStatus.INVALID,
        StoryMemoryArtifactStatus.INTEGRITY_MISMATCH,
    }


def test_build_refuses_existing_and_refresh_refuses_invalid_target(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    output = tmp_path / "story.json"
    _write_upstream(upstream, _fact("one"))
    first = build_and_materialize_authoritative_engineering_story_memory(
        upstream_path=upstream,
        output_path=output,
    )
    assert first.write_status is StoryMemoryWriteStatus.CREATED
    original = output.read_bytes()

    with pytest.raises(EngineeringStoryMemoryServiceError) as exc_info:
        build_and_materialize_authoritative_engineering_story_memory(
            upstream_path=upstream,
            output_path=output,
        )
    assert exc_info.value.code is StoryMemoryServiceErrorCode.ARTIFACT_ALREADY_EXISTS
    assert output.read_bytes() == original

    output.write_text("invalid", encoding="utf-8")
    with pytest.raises(EngineeringStoryMemoryServiceError) as exc_info:
        refresh_and_materialize_authoritative_engineering_story_memory(
            upstream_path=upstream,
            output_path=output,
        )
    assert exc_info.value.code is StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY


def test_atomic_refresh_failure_restores_previous_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "story.json"
    initial = _single_story_memory(_fact("one")).memory
    updated = refresh_authoritative_engineering_story_memory(
        existing_memory=initial,
        snapshot=_snapshot(_fact("one"), _fact("two")),
    ).memory
    assert write_authoritative_engineering_story_memory(
        initial, path=path, operation=StoryMemoryOperation.BUILD
    ).status is StoryMemoryWriteStatus.CREATED
    previous = path.read_bytes()
    real_replace = service._replace_staged_authoritative_story_memory

    def replace_then_fail(staged: Path, destination: Path) -> None:
        real_replace(staged, destination)
        raise OSError("simulated post-replace failure")

    monkeypatch.setattr(
        service,
        "_replace_staged_authoritative_story_memory",
        replace_then_fail,
    )
    write = write_authoritative_engineering_story_memory(
        updated,
        path=path,
        operation=StoryMemoryOperation.REFRESH,
    )
    assert write.status is StoryMemoryWriteStatus.FAILED
    assert write.previous_artifact_preserved is True
    assert path.read_bytes() == previous


def test_readiness_distinguishes_missing_ready_stale_and_invalid(tmp_path: Path) -> None:
    story_path = tmp_path / "story.json"
    upstream_path = tmp_path / "upstream.json"
    assert inspect_authoritative_engineering_story_memory_readiness(
        path=story_path
    ).state is StoryMemoryReadinessState.MISSING

    _write_upstream(upstream_path, _fact("one"))
    result = build_and_materialize_authoritative_engineering_story_memory(
        upstream_path=upstream_path,
        output_path=story_path,
    )
    ready = inspect_authoritative_engineering_story_memory_readiness(
        path=story_path,
        upstream_path=upstream_path,
        compare_upstream=True,
    )
    assert ready.state is StoryMemoryReadinessState.READY
    assert ready.logical_fingerprint == result.memory.logical_fingerprint

    _write_upstream(upstream_path, _fact("one"), _fact("two"))
    stale = inspect_authoritative_engineering_story_memory_readiness(
        path=story_path,
        upstream_path=upstream_path,
        compare_upstream=True,
    )
    assert stale.state is StoryMemoryReadinessState.STALE_OR_REVALIDATION_REQUIRED
    assert stale.error_codes == ("upstream_content_hash_changed",)

    story_path.write_text("invalid", encoding="utf-8")
    assert inspect_authoritative_engineering_story_memory_readiness(
        path=story_path
    ).state is StoryMemoryReadinessState.INVALID


def test_cross_project_lookup_fails_closed_and_unknown_alias_is_not_guessed() -> None:
    memory = _single_story_memory(_fact("one")).memory
    story_id = memory.histories[0].canonical_story_id
    assert resolve_engineering_story_id(memory, story_id, project_id="other") is None
    assert get_engineering_story_by_id(memory, story_id, project_id="other") is None
    assert get_engineering_stories_for_project(memory, "owner/workagent") == ()
    assert resolve_engineering_story_id(
        memory, "engineering_story_" + "f" * 24
    ) is None


def test_final_envelope_rejects_orphan_identity_and_is_immutable() -> None:
    source = EngineeringStorySourceArtifact(
        schema_version="project_evidence_memory.v1",
        content_hash="0" * 64,
        project_count=0,
        evidence_fact_count=0,
        capability_fact_count=0,
        claim_boundary_count=0,
    )
    empty = AuthoritativeEngineeringStoryMemory(
        schema_version=AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
        source_artifact=source,
        identity_map=EngineeringStoryIdentityMap(),
    )
    with pytest.raises(FrozenInstanceError):
        empty.schema_version = "changed"  # type: ignore[misc]

    populated = _single_story_memory(_fact("one")).memory
    with pytest.raises(ValueError, match="identity/history"):
        AuthoritativeEngineeringStoryMemory(
            schema_version=AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
            source_artifact=populated.source_artifact,
            identity_map=populated.identity_map,
            histories=(),
        )


def test_service_imports_have_no_runtime_retrieval_model_network_or_api_dependency() -> None:
    source_path = Path(service.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = (
        "chromadb",
        "backend.api_server",
        "backend.project_retrieval",
        "backend.project_retrieval_v2",
        "backend.evidence_hybrid_retrieval",
        "backend.project_query_planner",
        "backend.memory_store",
        "requests",
        "httpx",
        "openai",
    )
    assert not any(
        imported == item or imported.startswith(f"{item}.")
        for imported in imports
        for item in forbidden
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"getenv", "environ"}
        for node in ast.walk(tree)
    )


def test_reader_does_not_rebuild_or_perform_provider_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "missing.json"
    monkeypatch.setattr(
        service,
        "build_authoritative_engineering_story_memory",
        lambda *_args, **_kwargs: pytest.fail("reader rebuilt memory"),
    )
    monkeypatch.setattr(
        service,
        "load_project_evidence_memory",
        lambda *_args, **_kwargs: pytest.fail("reader loaded upstream"),
    )
    loaded = load_authoritative_engineering_story_memory(path)
    assert loaded.status is StoryMemoryArtifactStatus.MISSING


def test_real_shape_report_contains_counts_only_not_story_bodies() -> None:
    result = _single_story_memory(_fact("one"))
    report = result.to_dict()
    serialized = json.dumps(report, sort_keys=True)
    assert "memory" not in report
    assert "current_story" not in serialized
    assert "evidence_fact_ids" not in serialized
    assert "Materialized bounded" not in serialized
    assert report["canonical_story_count"] == 1
    assert report["source_schema_version"] == "project_evidence_memory.v1"
