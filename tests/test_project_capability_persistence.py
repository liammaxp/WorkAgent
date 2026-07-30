from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

import backend.project_capability_memory as capability_memory_module
from backend.project_capability_memory import (
    PROJECT_CAPABILITY_MEMORY_PATH,
    PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION,
    ProjectCapabilityMemory,
    ProjectCapabilityMemoryDiagnostics,
    ProjectCapabilityMemoryIntegrityError,
    ProjectCapabilityProjectSummary,
    ProjectCapabilitySourceArtifact,
    build_project_capability_memory,
    compute_project_capability_memory_content_hash,
    load_project_capability_memory,
    persist_project_capability_memory,
    validate_project_capability_memory,
)
from backend.project_evidence_memory import DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH
from backend.project_evidence_models import (
    Confidence,
    MetricSupport,
    ProjectCapabilityFact,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_CAPABILITY_ARTIFACT = ROOT / "information" / "project_capability_memory.json"
EVIDENCE_ARTIFACT = ROOT / "information" / "project_evidence_memory.json"
EXPECTED_EVIDENCE_FILE_HASH = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"


def _source(
    *,
    project_count: int = 3,
    content_hash: str = "a" * 64,
    file_sha256: str | None = "b" * 64,
) -> ProjectCapabilitySourceArtifact:
    return ProjectCapabilitySourceArtifact(
        schema_version="project_evidence_memory.v1",
        content_hash=content_hash,
        file_sha256=file_sha256,
        project_count=project_count,
        evidence_fact_count=12,
        claim_boundary_count=8,
    )


def _fact(
    *,
    project_id: str = "project-a",
    capability_type: str = "output_quality_control",
    evidence_ids: tuple[str, ...] = ("pef_one",),
    mechanisms: tuple[str, ...] = ("schema-bound output validation",),
    confidence: Confidence = Confidence.MEDIUM,
    metric_support: MetricSupport = MetricSupport.NONE,
    technical_tags: tuple[str, ...] = ("FastAPI",),
    allowed_claims: tuple[str, ...] = ("mechanism:schema-bound output validation",),
    forbidden_claims: tuple[str, ...] = ("unsupported metric claim",),
    capability_id: str = "",
) -> ProjectCapabilityFact:
    return ProjectCapabilityFact(
        project_id=project_id,
        capability_type=capability_type,
        present=True,
        source_evidence_fact_ids=list(evidence_ids),
        confidence=confidence,
        mechanisms=list(mechanisms),
        allowed_resume_claims=list(allowed_claims),
        forbidden_claims=list(forbidden_claims),
        metric_support=metric_support,
        technical_tags=list(technical_tags),
        capability_id=capability_id,
    )


def _populated_memory() -> ProjectCapabilityMemory:
    facts = (
        _fact(),
        _fact(
            project_id="project-b",
            capability_type="failure_recovery",
            evidence_ids=("pef_two",),
            mechanisms=("deterministic fallback recovery",),
            confidence=Confidence.HIGH,
            metric_support=MetricSupport.APPROXIMATE,
            technical_tags=("Python",),
            allowed_claims=("mechanism:deterministic fallback recovery",),
        ),
    )
    return build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-a", "project-b", "project-c"),
        capability_facts=facts,
    )


def test_builds_valid_empty_project_capability_memory(tmp_path):
    project_ids = tuple(f"project-{index:02d}" for index in range(11))
    memory = build_project_capability_memory(
        source_artifact=_source(project_count=11),
        source_project_ids=project_ids,
        capability_facts=(),
    )
    assert memory.schema_version == PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION
    assert len(memory.projects) == 11
    assert memory.capability_facts == ()
    assert memory.diagnostics.projects_with_capabilities == 0
    assert memory.diagnostics.projects_without_capabilities == 11
    assert "capability_facts_empty" in memory.diagnostics.warnings
    assert len(memory.content_hash) == 64
    assert validate_project_capability_memory(memory).valid
    path = tmp_path / "empty-project-capability-memory.json"
    assert persist_project_capability_memory(memory, path).status == "created"
    loaded = load_project_capability_memory(path)
    assert loaded.status == "empty" and loaded.memory == memory


def test_builds_memory_from_authoritative_project_capability_facts():
    memory = _populated_memory()
    assert len(memory.capability_facts) == 2
    assert all(type(fact) is ProjectCapabilityFact for fact in memory.capability_facts)
    assert [project.project_id for project in memory.projects] == ["project-a", "project-b", "project-c"]
    assert memory.projects[0].capability_fact_ids == (memory.capability_facts[0].capability_id,)
    assert memory.projects[2].capability_fact_count == 0
    assert memory.diagnostics.capability_fact_count == 2
    assert memory.diagnostics.projects_with_capabilities == 2
    assert memory.diagnostics.projects_without_capabilities == 1


def test_memory_build_is_input_order_independent():
    first = _fact(
        evidence_ids=("pef_b", "pef_a"),
        mechanisms=("schema-bound validation", "atomic artifact replacement"),
        technical_tags=("Python", "FastAPI"),
        allowed_claims=("implementation:bounded validation", "mechanism:atomic artifact replacement"),
        forbidden_claims=("unsupported metric claim", "absolute guarantee"),
    )
    reordered = _fact(
        evidence_ids=("pef_a", "pef_b"),
        mechanisms=("atomic artifact replacement", "schema-bound validation"),
        technical_tags=("FastAPI", "Python"),
        allowed_claims=("mechanism:atomic artifact replacement", "implementation:bounded validation"),
        forbidden_claims=("absolute guarantee", "unsupported metric claim"),
    )
    one = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-c", "project-a", "project-b"),
        capability_facts=(first,),
    )
    two = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-b", "project-a", "project-c"),
        capability_facts=(reordered,),
    )
    assert one == two
    assert one.to_json() == two.to_json()
    assert one.content_hash == two.content_hash


def test_memory_content_hash_is_deterministic_and_semantic():
    base = _populated_memory()
    same = _populated_memory()
    changed_source = build_project_capability_memory(
        source_artifact=_source(content_hash="c" * 64),
        source_project_ids=("project-a", "project-b", "project-c"),
        capability_facts=base.capability_facts,
    )
    changed_fact = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-a", "project-b", "project-c"),
        capability_facts=(
            _fact(mechanisms=("atomic artifact replacement",)),
            base.capability_facts[1],
        ),
    )
    altered_hash_field = replace(base, content_hash="0" * 64)
    assert base.content_hash == same.content_hash
    assert changed_source.content_hash != base.content_hash
    assert changed_fact.content_hash != base.content_hash
    assert compute_project_capability_memory_content_hash(altered_hash_field) == base.content_hash


def test_memory_artifact_contains_no_volatile_runtime_fields():
    payload = json.dumps(_populated_memory().to_dict(), sort_keys=True).casefold()
    forbidden = (
        "generated_at", "written_at", "updated_at", "build_timestamp", "runtime_duration",
        "host_name", "working_directory", "branch_name", "commit_time", "temporary_path",
    )
    assert not any(value in payload for value in forbidden)


def test_memory_rejects_wrong_schema_version():
    memory = replace(_populated_memory(), schema_version="project_capability_memory")
    report = validate_project_capability_memory(memory)
    assert not report.valid and "unsupported_schema_version" in report.errors
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="unsupported_schema_version"):
        ProjectCapabilityMemory.from_dict(memory.to_dict())


def test_memory_rejects_content_hash_mismatch(tmp_path):
    memory = replace(_populated_memory(), content_hash="0" * 64)
    report = validate_project_capability_memory(memory)
    assert not report.valid and "content_hash_mismatch" in report.errors
    path = tmp_path / "bad-hash.json"
    path.write_text(json.dumps(memory.to_dict()), encoding="utf-8")
    loaded = load_project_capability_memory(path)
    assert loaded.status == "hash_mismatch" and loaded.memory is None


def test_memory_rejects_dangling_or_cross_project_fact_references():
    memory = _populated_memory()
    dangling_summary = replace(
        memory.projects[0], capability_fact_ids=("pcf_missing",)
    )
    dangling = replace(memory, projects=(dangling_summary, *memory.projects[1:]))
    assert "project_fact_referential_integrity_failure" in validate_project_capability_memory(dangling).errors

    cross_summary = replace(
        memory.projects[2],
        capability_fact_ids=(memory.capability_facts[0].capability_id,),
        capability_types=(memory.capability_facts[0].capability_type,),
        capability_fact_count=1,
        confirmed_capability_count=1,
    )
    cross = replace(memory, projects=(*memory.projects[:2], cross_summary))
    assert "project_fact_referential_integrity_failure" in validate_project_capability_memory(cross).errors


def test_memory_deduplicates_identical_facts_and_rejects_conflicts():
    fact = _fact()
    memory = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-a", "project-b", "project-c"),
        capability_facts=(fact, fact),
    )
    assert len(memory.capability_facts) == 1

    conflicting_payload = _fact(mechanisms=("atomic artifact replacement",))
    assert conflicting_payload.capability_id == fact.capability_id
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="conflicting_capability_id"):
        build_project_capability_memory(
            source_artifact=_source(),
            source_project_ids=("project-a", "project-b", "project-c"),
            capability_facts=(fact, conflicting_payload),
        )

    conflicting_identity = _fact(evidence_ids=("pef_different",))
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="conflicting_project_capability_identity"):
        build_project_capability_memory(
            source_artifact=_source(),
            source_project_ids=("project-a", "project-b", "project-c"),
            capability_facts=(fact, conflicting_identity),
        )


def test_memory_diagnostics_match_stored_facts():
    memory = _populated_memory()
    bad_diagnostics = replace(memory.diagnostics, capability_fact_count=3)
    invalid = replace(memory, diagnostics=bad_diagnostics)
    assert "diagnostics_mismatch" in validate_project_capability_memory(invalid).errors
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="diagnostics_mismatch"):
        build_project_capability_memory(
            source_artifact=_source(),
            source_project_ids=("project-a", "project-b", "project-c"),
            capability_facts=memory.capability_facts,
            diagnostics=bad_diagnostics,
        )


def test_memory_validates_source_artifact_lineage():
    with pytest.raises(ValueError, match="SHA-256"):
        _source(content_hash="bad")
    with pytest.raises(ValueError, match="non-negative"):
        ProjectCapabilitySourceArtifact(
            schema_version="project_evidence_memory.v1",
            content_hash="a" * 64,
            file_sha256=None,
            project_count=-1,
            evidence_fact_count=0,
            claim_boundary_count=0,
        )
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="duplicate_source_project_id"):
        build_project_capability_memory(
            source_artifact=_source(project_count=2),
            source_project_ids=("project-a", "project-a"),
            capability_facts=(),
        )
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="source_project_count_mismatch"):
        build_project_capability_memory(
            source_artifact=_source(project_count=2),
            source_project_ids=("project-a",),
            capability_facts=(),
        )


def test_project_capability_memory_round_trips_through_dict():
    memory = _populated_memory()
    restored = ProjectCapabilityMemory.from_dict(memory.to_dict())
    assert restored == memory
    assert restored.to_dict() == memory.to_dict()
    assert restored.content_hash == memory.content_hash


def test_project_capability_memory_rejects_unknown_fields():
    payload = _populated_memory().to_dict()
    payload["generated_at"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="invalid project_capability_memory fields"):
        ProjectCapabilityMemory.from_dict(payload)


def test_project_capability_memory_does_not_share_mutable_input_state():
    project_ids = ["project-a", "project-b", "project-c"]
    fact = _fact(mechanisms=("schema-bound validation",))
    memory = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=project_ids,
        capability_facts=[fact],
    )
    project_ids.append("project-d")
    fact.mechanisms.append("mutated external mechanism")
    assert len(memory.projects) == 3
    assert memory.capability_facts[0].mechanisms == ["schema-bound validation"]
    with pytest.raises(FrozenInstanceError):
        memory.content_hash = "0" * 64
    with pytest.raises(TypeError):
        memory.diagnostics.confidence_counts["high"] = 99


def test_persisted_project_capability_memory_loads_identically(tmp_path):
    memory = _populated_memory()
    path = tmp_path / "project_capability_memory.json"
    report = persist_project_capability_memory(memory, path)
    loaded = load_project_capability_memory(path)
    assert report.status == "created" and report.round_trip_validated
    assert loaded.status == "ready" and loaded.memory == memory
    assert path.read_bytes().endswith(b"\n")
    assert json.loads(path.read_text(encoding="utf-8")) == memory.to_dict()
    assert persist_project_capability_memory(memory, path).status == "unchanged"


def test_repeated_persistence_produces_identical_bytes(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    one = _populated_memory()
    two = build_project_capability_memory(
        source_artifact=_source(),
        source_project_ids=("project-c", "project-b", "project-a"),
        capability_facts=tuple(reversed(one.capability_facts)),
    )
    assert persist_project_capability_memory(one, first).status == "created"
    assert persist_project_capability_memory(two, second).status == "created"
    assert first.read_bytes() == second.read_bytes()


def test_loader_handles_missing_capability_memory(tmp_path):
    loaded = load_project_capability_memory(tmp_path / "missing.json")
    assert loaded.status == "missing" and loaded.memory is None
    assert not (tmp_path / "missing.json").exists()


def test_loader_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")
    loaded = load_project_capability_memory(path)
    assert loaded.status == "invalid" and loaded.memory is None
    assert loaded.validation.errors == ("malformed_artifact",)


def test_atomic_persistence_failure_preserves_existing_artifact(tmp_path, monkeypatch):
    path = tmp_path / "project_capability_memory.json"
    original = _populated_memory()
    assert persist_project_capability_memory(original, path).status == "created"
    before = path.read_bytes()
    changed = build_project_capability_memory(
        source_artifact=_source(content_hash="c" * 64),
        source_project_ids=("project-a", "project-b", "project-c"),
        capability_facts=original.capability_facts,
    )

    def replacement_fails(_staged, _destination):
        raise OSError("synthetic pre-replace failure")

    monkeypatch.setattr(
        capability_memory_module, "_replace_staged_project_capability_artifact", replacement_fails
    )
    report = persist_project_capability_memory(changed, path)
    assert report.status == "failed" and report.previous_artifact_preserved
    assert path.read_bytes() == before
    assert load_project_capability_memory(path).memory == original


def test_persistence_cannot_overwrite_project_evidence_memory():
    before = EVIDENCE_ARTIFACT.read_bytes()
    report = persist_project_capability_memory(_populated_memory(), DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH)
    assert report.status == "failed"
    assert report.warnings == ("upstream_artifact_path_forbidden",)
    assert EVIDENCE_ARTIFACT.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == EXPECTED_EVIDENCE_FILE_HASH


def test_project_capability_memory_contains_no_raw_or_sensitive_content():
    serialized = _populated_memory().to_json()
    forbidden = (
        "raw_text", "raw_patch", "complete_diff", "source_code", "github_context",
        "credential", "authorization_header", "chain_of_thought", "F:\\Agent Develop",
    )
    assert not any(value.casefold() in serialized.casefold() for value in forbidden)
    unsafe = _fact(technical_tags=(r"F:\Agent Develop\private.py",))
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="unsafe_artifact_value"):
        build_project_capability_memory(
            source_artifact=_source(),
            source_project_ids=("project-a", "project-b", "project-c"),
            capability_facts=(unsafe,),
        )


def test_step_does_not_create_real_project_capability_memory_artifact():
    assert PROJECT_CAPABILITY_MEMORY_PATH == REAL_CAPABILITY_ARTIFACT
    assert not REAL_CAPABILITY_ARTIFACT.exists()


def test_memory_stores_only_authoritative_project_capability_fact():
    memory = _populated_memory()
    assert all(type(fact) is ProjectCapabilityFact for fact in memory.capability_facts)
    assert not hasattr(capability_memory_module, "CapabilityFact")
    malformed = _fact(capability_id="pcf_not_authoritative")
    with pytest.raises(ProjectCapabilityMemoryIntegrityError, match="invalid_capability_id"):
        build_project_capability_memory(
            source_artifact=_source(),
            source_project_ids=("project-a", "project-b", "project-c"),
            capability_facts=(malformed,),
        )


def test_project_capability_persistence_uses_semantic_naming():
    source = (ROOT / "backend" / "project_capability_memory.py").read_text(encoding="utf-8").casefold()
    forbidden = (
        "phase" + "5", "phase_" + "5", "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5", "phase" + "5.v1", "use_" + "phase" + "5",
    )
    assert not any(value in source for value in forbidden)
    assert PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION == "project_capability_memory.v1"
