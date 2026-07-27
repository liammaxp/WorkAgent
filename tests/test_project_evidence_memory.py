from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend.project_claim_boundaries import (
    build_project_capability_claim_boundary,
    build_project_claim_boundaries_by_project,
    build_project_evidence_claim_boundary,
    build_project_claim_boundary,
)
from backend.project_evidence_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
    EvidenceSourceRef,
)
import backend.project_evidence_memory as project_memory
from backend.project_evidence_memory import (
    MAX_SERIALIZED_SIZE,
    ProjectEvidenceMemoryDiagnostics,
    ProjectEvidenceMemoryIntegrityError,
    ProjectEvidenceMemoryLoadResult,
    ProjectEvidenceMemorySnapshot,
    build_project_evidence_memories,
    build_project_evidence_memory_snapshot,
    load_project_evidence_memory,
    persist_project_evidence_memory,
    serialize_project_evidence_memory_snapshot,
    validate_project_evidence_memory_snapshot,
)


def source_ref(project_id: str = "alpha", **changes) -> EvidenceSourceRef:
    values = {
        "source_type": "project_change_raw_change_summary",
        "source_id": f"src-{project_id}",
        "project_id": project_id,
        "content_hash": "a" * 64,
        "file_path": "backend/memory.py",
        "metadata": {"source_category": "direct_evidence"},
    }
    values.update(changes)
    return EvidenceSourceRef(**values)


def fact(project_id: str = "alpha", **changes) -> ProjectEvidenceFact:
    values = {
        "project_id": project_id,
        "problem": "Project records needed integrity checks.",
        "mechanism": "atomic persistence with schema validation",
        "implementation": ["write a same-directory temporary file", "validate then replace"],
        "safe_impact": ["Keeps structured memory replacement atomic."],
        "evidence_type": EvidenceType.DATA_PERSISTENCE,
        "confidence": Confidence.HIGH,
        "metric_support": MetricSupport.NONE,
        "technical_tags": ["Python", "persistence"],
        "source_refs": [source_ref(project_id)],
        "status": EvidenceStatus.ACCEPTED,
        "quality_score": 90,
        "quality_breakdown": {"provenance": 20},
    }
    values.update(changes)
    return ProjectEvidenceFact(**values)


def capability(item: ProjectEvidenceFact, **changes) -> ProjectCapabilityFact:
    values = {
        "project_id": item.project_id,
        "capability_type": "data_persistence",
        "present": True,
        "source_evidence_fact_ids": [item.evidence_fact_id],
        "confidence": Confidence.HIGH,
        "mechanisms": ["write", "fsync", "replace"],
        "allowed_resume_claims": ["Persisted validated structured artifacts."],
        "forbidden_claims": ["Do not claim a durability guarantee."],
        "technical_tags": ["persistence"],
    }
    values.update(changes)
    return ProjectCapabilityFact(**values)


def boundaries_for(items, capabilities=()):
    evidence_boundaries = [
        boundary for item in items
        if (boundary := build_project_evidence_claim_boundary(item)) is not None
    ]
    capability_boundaries = [
        boundary for item in capabilities
        if (boundary := build_project_capability_claim_boundary(
            item, evidence_facts_by_id={fact.evidence_fact_id: fact for fact in items}
        )) is not None
    ]
    project_ids = sorted({item.project_id for item in [*items, *capabilities]})
    project_boundaries = [
        boundary for project_id in project_ids
        if (boundary := build_project_claim_boundary(
            project_id,
            [item for item in items if item.project_id == project_id],
            [item for item in capabilities if item.project_id == project_id],
        )) is not None
    ]
    return [*evidence_boundaries, *capability_boundaries, *project_boundaries]


def built_snapshot(items=None, capabilities=None, boundaries=None, diagnostics=None):
    items = list(items if items is not None else [fact()])
    capabilities = list(capabilities or [])
    boundaries = list(boundaries if boundaries is not None else boundaries_for(items, capabilities))
    memories, report = build_project_evidence_memories(
        items, capabilities, boundaries, diagnostics=diagnostics
    )
    return build_project_evidence_memory_snapshot(memories, diagnostics=diagnostics), report


def test_one_project_builds_valid_memory_and_zero_capabilities_remain_empty():
    item = fact()
    memories, report = build_project_evidence_memories([item], [], boundaries_for([item]))
    assert report.project_count == 1
    assert memories[0].project_id == "alpha"
    assert memories[0].project_name == "alpha"
    assert memories[0].evidence_facts == [item]
    assert memories[0].capability_facts == []
    assert memories[0].project_memory_id.startswith("pem_")
    assert [warning.code for warning in memories[0].warnings] == ["capability_facts_empty"]


def test_multiple_exact_project_ids_are_separate_and_aliases_are_not_inferred():
    items = [fact("workagent"), fact("example/workagent"), fact("liammaxp/WorkAgent")]
    memories, _ = build_project_evidence_memories(items, [], boundaries_for(items))
    assert [item.project_id for item in memories] == [
        "example/workagent", "liammaxp/WorkAgent", "workagent"
    ]


def test_evidence_capability_and_all_boundary_subject_types_are_preserved():
    item = fact()
    cap = capability(item)
    boundaries = boundaries_for([item], [cap])
    memories, _ = build_project_evidence_memories([item], [cap], boundaries)
    memory = memories[0]
    assert memory.capability_facts == [cap]
    assert {boundary.subject_type for boundary in memory.claim_boundaries} == {
        ClaimSubjectType.EVIDENCE_FACT,
        ClaimSubjectType.CAPABILITY_FACT,
        ClaimSubjectType.PROJECT,
    }


def test_context_only_project_boundary_builds_documented_safe_memory():
    boundary = ProjectClaimBoundary(
        project_id="context", subject_type=ClaimSubjectType.PROJECT, subject_id="context"
    )
    memories, report = build_project_evidence_memories([], [], [boundary])
    assert memories[0].evidence_facts == []
    assert memories[0].capability_facts == []
    assert report.context_only_project_count == 1
    assert [warning.code for warning in memories[0].warnings] == [
        "capability_facts_empty", "context_only_project"
    ]


def test_empty_collections_and_input_order_serialize_deterministically():
    empty_boundary = ProjectClaimBoundary(
        project_id="empty", subject_type=ClaimSubjectType.PROJECT, subject_id="empty"
    )
    first, _ = built_snapshot([], [], [empty_boundary])
    second, _ = built_snapshot([], [], reversed([empty_boundary]))
    assert first.content_hash == second.content_hash
    assert serialize_project_evidence_memory_snapshot(first) == serialize_project_evidence_memory_snapshot(second)


def test_build_does_not_mutate_input_objects():
    item = fact()
    cap = capability(item)
    boundaries = boundaries_for([item], [cap])
    before = ([item.to_dict()], [cap.to_dict()], [value.to_dict() for value in boundaries])
    build_project_evidence_memories([item], [cap], boundaries)
    assert before == ([item.to_dict()], [cap.to_dict()], [value.to_dict() for value in boundaries])


@pytest.mark.parametrize(
    ("kind", "id_field"),
    [
        ("evidence", "evidence_fact_id"),
        ("capability", "capability_id"),
        ("boundary", "boundary_id"),
    ],
)
def test_exact_duplicates_collapse_without_inflating_counts(kind, id_field):
    item = fact()
    cap = capability(item)
    boundary = boundaries_for([item], [cap])[0]
    inputs = {
        "evidence": ([item, item], [cap], [boundary]),
        "capability": ([item], [cap, cap], [boundary]),
        "boundary": ([item], [cap], [boundary, boundary]),
    }[kind]
    memories, report = build_project_evidence_memories(*inputs)
    collection = {
        "evidence": memories[0].evidence_facts,
        "capability": memories[0].capability_facts,
        "boundary": memories[0].claim_boundaries,
    }[kind]
    assert len(collection) == 1
    assert getattr(report, f"duplicate_{kind}_fact_count" if kind != "boundary" else "duplicate_claim_boundary_count") == 1


def test_same_evidence_id_with_different_payload_fails_without_selecting_one():
    first = fact(evidence_fact_id="pef_conflict")
    second = fact(mechanism="more impressive but conflicting", evidence_fact_id="pef_conflict")
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="same_evidence_fact_id"):
        build_project_evidence_memories([first, second], [], [])


def test_same_capability_id_with_different_payload_fails():
    item = fact()
    first = capability(item, capability_id="pcf_conflict")
    second = capability(item, mechanisms=["different"], capability_id="pcf_conflict")
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="same_capability_id"):
        build_project_evidence_memories([item], [first, second], [])


def test_same_boundary_id_with_different_payload_fails():
    first = ProjectClaimBoundary(
        project_id="alpha", subject_type=ClaimSubjectType.PROJECT,
        subject_id="alpha", boundary_id="pcb_conflict",
    )
    second = ProjectClaimBoundary(
        project_id="alpha", subject_type=ClaimSubjectType.PROJECT,
        subject_id="wrong", boundary_id="pcb_conflict",
    )
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="same_boundary_id"):
        build_project_evidence_memories([], [], [first, second])


def test_conflicting_project_memory_ids_fail():
    one = ProjectEvidenceMemory(project_id="one", project_name="one", project_memory_id="pem_same")
    two = ProjectEvidenceMemory(project_id="two", project_name="two", project_memory_id="pem_same")
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="same_project_memory_id"):
        build_project_evidence_memory_snapshot([one, two])


def test_capability_source_must_resolve_in_same_project():
    item = fact()
    cap = capability(item, source_evidence_fact_ids=["pef_unknown"])
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="unknown_capability_evidence"):
        build_project_evidence_memories([item], [cap], [])


@pytest.mark.parametrize("subject_type", [ClaimSubjectType.EVIDENCE_FACT, ClaimSubjectType.CAPABILITY_FACT])
def test_record_boundary_subject_must_resolve(subject_type):
    boundary = ProjectClaimBoundary(
        project_id="alpha", subject_type=subject_type, subject_id="pe_unknown"
    )
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="unknown_"):
        build_project_evidence_memories([], [], [boundary])


def test_project_boundary_subject_must_match_exact_project_id():
    boundary = ProjectClaimBoundary(
        project_id="alpha", subject_type=ClaimSubjectType.PROJECT, subject_id="Alpha"
    )
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="project_boundary_subject"):
        build_project_evidence_memories([], [], [boundary])


def test_boundary_support_ids_must_resolve():
    item = fact()
    boundary = build_project_evidence_claim_boundary(item)
    assert boundary is not None
    bad_notes = [note.replace(item.evidence_fact_id, "pef_unknown") for note in boundary.notes]
    bad = ProjectClaimBoundary(
        project_id=boundary.project_id,
        subject_type=boundary.subject_type,
        subject_id=boundary.subject_id,
        allowed_claims=boundary.allowed_claims,
        forbidden_claims=boundary.forbidden_claims,
        notes=bad_notes,
    )
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="unknown_evidence_fact_id"):
        build_project_evidence_memories([item], [], [bad])


def test_present_false_capability_is_not_persisted():
    item = fact()
    cap = capability(item, present=False, source_evidence_fact_ids=[])
    memories, report = build_project_evidence_memories([item], [cap], [])
    assert memories[0].capability_facts == []
    assert report.not_present_capability_count == 1


def test_diagnostics_preserve_truncation_blockers_and_safe_sorted_warning_codes():
    diagnostics = ProjectEvidenceMemoryDiagnostics(
        claim_truncation_count=501,
        projects_with_truncation=11,
        claim_type_truncation_counts={"technology": 4, "mechanism": 7},
        weak_fact_blocked_count=3,
        rejected_fact_blocked_count=2,
        warnings=("low_quality_facts_excluded", "claim_budget_truncated", "claim_budget_truncated"),
    )
    snapshot, _ = built_snapshot(diagnostics=diagnostics)
    assert snapshot.diagnostics.claim_truncation_count == 501
    assert snapshot.diagnostics.weak_fact_blocked_count == 3
    assert snapshot.diagnostics.warnings == (
        "claim_budget_truncated", "low_quality_facts_excluded"
    )
    text = serialize_project_evidence_memory_snapshot(snapshot).decode("utf-8")
    assert "decision" not in text
    assert "Project records needed" in text  # structured accepted fact, not diagnostic leakage


def test_diagnostics_do_not_change_project_identity_but_change_snapshot_hash():
    first, _ = built_snapshot(diagnostics=ProjectEvidenceMemoryDiagnostics(claim_truncation_count=1))
    second, _ = built_snapshot(diagnostics=ProjectEvidenceMemoryDiagnostics(claim_truncation_count=2))
    assert first.projects[0].project_memory_id == second.projects[0].project_memory_id
    assert first.content_hash != second.content_hash


def test_schema_hash_content_change_and_destination_independence(tmp_path):
    first, _ = built_snapshot()
    changed, _ = built_snapshot([fact(mechanism="atomic persistence with reload validation")])
    assert first.schema_version == "project_evidence_memory.v1"
    assert validate_project_evidence_memory_snapshot(first).valid
    assert first.content_hash != changed.content_hash
    one = persist_project_evidence_memory(first, tmp_path / "one.json")
    two = persist_project_evidence_memory(first, tmp_path / "two.json")
    assert one.content_hash == two.content_hash == first.content_hash


def test_unsupported_missing_schema_and_invalid_hash_are_rejected():
    snapshot, _ = built_snapshot()
    unsupported = replace(snapshot, schema_version="project_evidence_memory.v2")
    missing = replace(snapshot, schema_version="")
    bad_hash = replace(snapshot, content_hash="0" * 64)
    assert "unsupported_schema_version" in validate_project_evidence_memory_snapshot(unsupported).errors
    assert "missing_schema_version" in validate_project_evidence_memory_snapshot(missing).errors
    assert "content_hash_mismatch" in validate_project_evidence_memory_snapshot(bad_hash).errors


def test_canonical_serialization_utf8_sorted_keys_newline_and_round_trip(tmp_path):
    snapshot, _ = built_snapshot([fact(problem="非 ASCII 项目记录")])
    encoded = serialize_project_evidence_memory_snapshot(snapshot)
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert "非 ASCII" in encoded.decode("utf-8")
    assert encoded.lstrip().startswith(b"{")
    path = tmp_path / "memory.json"
    assert persist_project_evidence_memory(snapshot, path).status == "created"
    assert load_project_evidence_memory(path).snapshot == snapshot


def test_sequence_sensitive_mechanism_order_changes_identity_while_path_separator_does_not():
    item = fact()
    one = capability(item, mechanisms=["retrieve", "validate"])
    two = capability(item, mechanisms=["validate", "retrieve"])
    first, _ = built_snapshot([item], [one], [])
    second, _ = built_snapshot([item], [two], [])
    assert first.content_hash != second.content_hash
    slash = fact(source_refs=[source_ref(file_path="backend/memory.py")])
    backslash = fact(source_refs=[source_ref(file_path="backend\\memory.py")])
    assert slash.evidence_fact_id == backslash.evidence_fact_id


@pytest.mark.parametrize(
    "field_name",
    ["raw_text", "raw_content", "patch", "full_patch", "source_code", "api_key", "access_token", "authorization"],
)
def test_prohibited_storage_fields_are_rejected(field_name):
    item = fact()
    item.source_refs[0].metadata[field_name] = "sensitive value"
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="prohibited_storage_field"):
        build_project_evidence_memories([item], [], [])


@pytest.mark.parametrize("secret", ["ghp_" + "a" * 30, "Bearer " + "x" * 30, "-----BEGIN PRIVATE KEY-----"])
def test_secret_like_values_are_rejected(secret):
    item = fact()
    item.source_refs[0].metadata["credential_value"] = secret
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="secret|private_key"):
        build_project_evidence_memories([item], [], [])


def test_safe_token_and_secret_detection_discussion_is_not_rejected():
    item = fact(
        mechanism="token efficiency validation and secret detection",
        implementation=["detect secret-shaped fields", "measure token handling"],
    )
    snapshot, _ = built_snapshot([item])
    assert validate_project_evidence_memory_snapshot(snapshot).valid


def test_absolute_source_path_is_rejected():
    item = fact(source_refs=[source_ref(file_path="C:\\Users\\private\\source.py")])
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match="absolute_source_path"):
        build_project_evidence_memories([item], [], [])


def test_missing_artifact_created_atomically_and_temp_removed(tmp_path):
    snapshot, _ = built_snapshot()
    path = tmp_path / "nested" / "memory.json"
    report = persist_project_evidence_memory(snapshot, path)
    assert report.status == "created"
    assert report.round_trip_validated
    assert path.exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_existing_valid_artifact_updates_atomically(tmp_path):
    path = tmp_path / "memory.json"
    first, _ = built_snapshot()
    second, _ = built_snapshot([fact(mechanism="atomic persistence and deterministic reload")])
    assert persist_project_evidence_memory(first, path).status == "created"
    report = persist_project_evidence_memory(second, path)
    assert report.status == "updated"
    assert load_project_evidence_memory(path).snapshot == second


def test_simulated_write_failure_preserves_previous_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    first, _ = built_snapshot()
    second, _ = built_snapshot([fact(mechanism="changed")])
    persist_project_evidence_memory(first, path)
    before = path.read_bytes()
    monkeypatch.setattr(project_memory, "_write_temp_bytes", lambda *_: (_ for _ in ()).throw(OSError()))
    report = persist_project_evidence_memory(second, path)
    assert report.status == "failed"
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_simulated_replace_failure_preserves_previous_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    first, _ = built_snapshot()
    second, _ = built_snapshot([fact(mechanism="changed")])
    persist_project_evidence_memory(first, path)
    before = path.read_bytes()
    monkeypatch.setattr(project_memory.os, "replace", lambda *_: (_ for _ in ()).throw(OSError()))
    report = persist_project_evidence_memory(second, path)
    assert report.status == "failed"
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_snapshot_is_never_written_and_directory_destination_fails(tmp_path):
    snapshot, _ = built_snapshot()
    invalid = replace(snapshot, content_hash="0" * 64)
    path = tmp_path / "new.json"
    assert persist_project_evidence_memory(invalid, path).status == "failed"
    assert not path.exists()
    assert persist_project_evidence_memory(snapshot, tmp_path).status == "failed"


def test_repeated_identical_write_is_unchanged_and_preserves_bytes_and_mtime(tmp_path):
    snapshot, _ = built_snapshot()
    path = tmp_path / "memory.json"
    assert persist_project_evidence_memory(snapshot, path).status == "created"
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    report = persist_project_evidence_memory(snapshot, path)
    assert report.status == "unchanged"
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime
    assert report.bytes_written == 0


def test_formatting_difference_does_not_force_update(tmp_path):
    snapshot, _ = built_snapshot()
    path = tmp_path / "memory.json"
    persist_project_evidence_memory(snapshot, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    compact = path.read_bytes()
    assert persist_project_evidence_memory(snapshot, path).status == "unchanged"
    assert path.read_bytes() == compact


def test_missing_valid_malformed_unsupported_and_hash_mismatch_load_statuses(tmp_path):
    path = tmp_path / "memory.json"
    assert load_project_evidence_memory(path).status == "missing"
    snapshot, _ = built_snapshot()
    persist_project_evidence_memory(snapshot, path)
    assert load_project_evidence_memory(path).status == "ready"
    path.write_text("{bad", encoding="utf-8")
    assert load_project_evidence_memory(path).status == "invalid"
    path.write_text(json.dumps({"schema_version": "project_evidence_memory.v2"}), encoding="utf-8")
    assert load_project_evidence_memory(path).status == "unsupported_version"
    payload = snapshot.to_dict()
    payload["content_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_project_evidence_memory(path).status == "hash_mismatch"
    assert load_project_evidence_memory(path).snapshot is None


def test_invalid_existing_artifact_not_overwritten_without_explicit_option(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("invalid", encoding="utf-8")
    snapshot, _ = built_snapshot()
    before = path.read_bytes()
    report = persist_project_evidence_memory(snapshot, path)
    assert report.status == "failed"
    assert report.previous_artifact_preserved
    assert path.read_bytes() == before
    replaced = persist_project_evidence_memory(snapshot, path, replace_invalid=True)
    assert replaced.status == "updated"
    assert load_project_evidence_memory(path).status == "ready"


def test_round_trip_ids_and_hashes_match(tmp_path):
    item = fact()
    cap = capability(item)
    snapshot, _ = built_snapshot([item], [cap])
    path = tmp_path / "memory.json"
    report = persist_project_evidence_memory(snapshot, path)
    loaded = load_project_evidence_memory(path)
    assert report.round_trip_validated
    assert loaded.snapshot == snapshot
    assert loaded.snapshot.content_hash == snapshot.content_hash
    assert [p.project_memory_id for p in loaded.snapshot.projects] == [
        p.project_memory_id for p in snapshot.projects
    ]
    assert [f.evidence_fact_id for f in loaded.snapshot.projects[0].evidence_facts] == [item.evidence_fact_id]
    assert [c.capability_id for c in loaded.snapshot.projects[0].capability_facts] == [cap.capability_id]


@pytest.mark.parametrize(
    ("limit_name", "error"),
    [
        ("MAX_PROJECTS", "maximum_project_count"),
        ("MAX_EVIDENCE_FACTS", "maximum_evidence_fact_count"),
        ("MAX_CAPABILITY_FACTS", "maximum_capability_fact_count"),
        ("MAX_CLAIM_BOUNDARIES", "maximum_claim_boundary_count"),
    ],
)
def test_record_safety_limits_are_enforced(monkeypatch, limit_name, error):
    monkeypatch.setattr(project_memory, limit_name, 0)
    item = fact()
    cap = capability(item)
    values = {
        "MAX_PROJECTS": ([item], [], []),
        "MAX_EVIDENCE_FACTS": ([item], [], []),
        "MAX_CAPABILITY_FACTS": ([item], [cap], []),
        "MAX_CLAIM_BOUNDARIES": ([], [], [ProjectClaimBoundary(
            project_id="alpha", subject_type=ClaimSubjectType.PROJECT, subject_id="alpha"
        )]),
    }[limit_name]
    with pytest.raises(ProjectEvidenceMemoryIntegrityError, match=error):
        build_project_evidence_memories(*values)


def test_maximum_serialized_size_fails_without_truncating_or_writing(tmp_path, monkeypatch):
    snapshot, _ = built_snapshot()
    monkeypatch.setattr(project_memory, "MAX_SERIALIZED_SIZE", 1)
    validation = validate_project_evidence_memory_snapshot(snapshot)
    assert "maximum_serialized_size_exceeded" in validation.errors
    path = tmp_path / "memory.json"
    assert persist_project_evidence_memory(snapshot, path).status == "failed"
    assert not path.exists()


def test_exceeding_size_limit_preserves_previous_valid_artifact(tmp_path, monkeypatch):
    snapshot, _ = built_snapshot()
    path = tmp_path / "memory.json"
    persist_project_evidence_memory(snapshot, path)
    before = path.read_bytes()
    monkeypatch.setattr(project_memory, "MAX_SERIALIZED_SIZE", 1)
    report = persist_project_evidence_memory(snapshot, path)
    assert report.status == "failed"
    assert report.previous_artifact_preserved
    assert path.read_bytes() == before


def test_warning_list_is_bounded_and_core_records_are_not_truncated():
    diagnostics = ProjectEvidenceMemoryDiagnostics(
        warnings=tuple(f"warning_{index:03d}" for index in range(150))
    )
    snapshot, _ = built_snapshot(diagnostics=diagnostics)
    assert len(snapshot.diagnostics.warnings) == project_memory.MAX_WARNINGS
    assert len(snapshot.projects[0].evidence_facts) == 1


def test_build_validate_serialize_and_load_have_documented_file_purity(tmp_path):
    item = fact()
    before = set(tmp_path.iterdir())
    memories, _ = build_project_evidence_memories([item], [], boundaries_for([item]))
    snapshot = build_project_evidence_memory_snapshot(memories)
    validate_project_evidence_memory_snapshot(snapshot)
    serialize_project_evidence_memory_snapshot(snapshot)
    assert load_project_evidence_memory(tmp_path / "missing.json").status == "missing"
    assert set(tmp_path.iterdir()) == before


def test_file_flush_and_same_directory_temp_are_exercised(tmp_path, monkeypatch):
    snapshot, _ = built_snapshot()
    path = tmp_path / "memory.json"
    observed = {"fsync": 0, "temp_parent": None}
    original_fsync = project_memory.os.fsync
    original_temp = project_memory._write_temp_bytes

    def recording_fsync(descriptor):
        observed["fsync"] += 1
        return original_fsync(descriptor)

    def recording_temp(destination, serialized):
        result = original_temp(destination, serialized)
        observed["temp_parent"] = result.parent
        return result

    monkeypatch.setattr(project_memory.os, "fsync", recording_fsync)
    monkeypatch.setattr(project_memory, "_write_temp_bytes", recording_temp)
    assert persist_project_evidence_memory(snapshot, path).status == "created"
    assert observed["fsync"] >= 1
    assert observed["temp_parent"] == path.parent


def test_round_trip_mismatch_reports_failure_and_restores_previous(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    first, _ = built_snapshot()
    second, _ = built_snapshot([fact(mechanism="changed after validation")])
    persist_project_evidence_memory(first, path)
    before = path.read_bytes()
    original_load = project_memory.load_project_evidence_memory
    calls = {"destination": 0}

    def mismatching_load(candidate=None):
        candidate_path = Path(candidate) if candidate is not None else project_memory.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH
        if candidate_path == path:
            calls["destination"] += 1
            if calls["destination"] == 2:
                return ProjectEvidenceMemoryLoadResult(
                    status="invalid",
                    snapshot=None,
                    validation=project_memory.ProjectEvidenceMemoryValidationReport(
                        valid=False, errors=("simulated_round_trip_mismatch",)
                    ),
                )
        return original_load(candidate)

    monkeypatch.setattr(project_memory, "load_project_evidence_memory", mismatching_load)
    report = persist_project_evidence_memory(second, path)
    assert report.status == "failed"
    assert not report.round_trip_validated
    assert report.previous_artifact_preserved
    assert path.read_bytes() == before


def test_module_import_has_no_file_network_database_or_model_side_effects(monkeypatch):
    code = (
        "from pathlib import Path; import sys; "
        "Path.read_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected read')); "
        "Path.write_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected write')); "
        "import backend.project_evidence_memory; "
        "assert 'backend.api_server' not in sys.modules; "
        "assert not any(n.startswith(('chromadb','openai','requests','sqlite3')) for n in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_step2_through_step9_snapshot_and_temp_round_trip(tmp_path):
    from backend.project_capability_extractor import extract_project_evidence_capabilities_by_project
    from backend.project_evidence_normalizer import dedupe_project_evidence_inputs
    from backend.project_evidence_scoring import score_project_evidence_facts
    from backend.project_evidence_synthesizer import synthesize_project_evidence_facts
    from backend.project_evidence_input import load_project_evidence_inputs

    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    synthesized, _ = synthesize_project_evidence_facts(normalized)
    scored, _ = score_project_evidence_facts(synthesized)
    grouped, _ = extract_project_evidence_capabilities_by_project(scored)
    capabilities = [item for values in grouped.values() for item in values]
    project_boundaries, step8_report = build_project_claim_boundaries_by_project(scored, capabilities)
    evidence_boundaries = [
        boundary for item in scored
        if (boundary := build_project_evidence_claim_boundary(item)) is not None
    ]
    evidence_by_id = {item.evidence_fact_id: item for item in scored}
    capability_boundaries = [
        boundary for item in capabilities
        if (boundary := build_project_capability_claim_boundary(
            item, evidence_facts_by_id=evidence_by_id
        )) is not None
    ]
    all_boundaries = [*evidence_boundaries, *capability_boundaries, *project_boundaries.values()]
    diagnostics = ProjectEvidenceMemoryDiagnostics.from_claim_boundary_report(step8_report)
    memories, report = build_project_evidence_memories(
        scored, capabilities, all_boundaries, diagnostics=diagnostics
    )
    snapshot = build_project_evidence_memory_snapshot(memories, diagnostics=diagnostics)
    assert report.project_count == len({item.project_id for item in scored})
    assert sum(len(memory.evidence_facts) for memory in memories) == len(scored)
    assert sum(len(memory.capability_facts) for memory in memories) == len(capabilities) == 0
    assert sum(len(memory.claim_boundaries) for memory in memories) == len(all_boundaries)
    assert snapshot.diagnostics.claim_truncation_count == step8_report.truncated_claim_count
    assert validate_project_evidence_memory_snapshot(snapshot).valid
    serialized = serialize_project_evidence_memory_snapshot(snapshot)
    assert len(serialized) < project_memory.MAX_SERIALIZED_SIZE
    assert b'"raw_text"' not in serialized and b'"full_patch"' not in serialized
    path = tmp_path / "real.json"
    assert persist_project_evidence_memory(snapshot, path).status == "created"
    assert load_project_evidence_memory(path).snapshot == snapshot
    assert persist_project_evidence_memory(snapshot, path).status == "unchanged"


def test_serialized_snapshot_contains_no_raw_github_context_or_runtime_metadata():
    snapshot, _ = built_snapshot()
    payload = serialize_project_evidence_memory_snapshot(snapshot).decode("utf-8")
    for forbidden in ("raw_text", "raw_content", "full_patch", "github_raw", "updated_at", "timestamp"):
        assert forbidden not in payload
    assert "object at 0x" not in payload
    assert len(payload.encode("utf-8")) < MAX_SERIALIZED_SIZE
