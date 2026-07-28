"""Read-only baseline contract for future Project Capability Memory work."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend import project_evidence_memory
from backend.project_evidence_models import PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "information" / "project_evidence_memory.json"
EXPECTED_BASELINE = {
    "project_count": 11,
    "evidence_fact_count": 283,
    "claim_boundary_count": 184,
    "capability_fact_count": 0,
}


def load_baseline():
    loaded = project_evidence_memory.load_project_evidence_memory(ARTIFACT)
    assert loaded.status == "ready"
    assert loaded.validation.valid
    assert loaded.snapshot is not None
    return loaded


def snapshot_counts(snapshot) -> dict[str, int]:
    return {
        "project_count": len(snapshot.projects),
        "evidence_fact_count": sum(len(project.evidence_facts) for project in snapshot.projects),
        "claim_boundary_count": sum(len(project.claim_boundaries) for project in snapshot.projects),
        "capability_fact_count": sum(len(project.capability_facts) for project in snapshot.projects),
    }


def test_project_evidence_memory_is_loadable():
    loaded = load_baseline()
    root = loaded.snapshot.to_dict()
    assert isinstance(root, dict)
    assert root["schema_version"] == PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION
    assert isinstance(root["projects"], list)
    assert ARTIFACT == project_evidence_memory.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH


def test_project_evidence_memory_uses_current_semantic_schema():
    loaded = load_baseline()
    assert loaded.snapshot.schema_version == "project_evidence_memory.v1"
    assert ARTIFACT.name == "project_evidence_memory.json"
    assert project_evidence_memory.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH.name == ARTIFACT.name


def test_project_evidence_memory_has_expected_sections_and_consistent_diagnostics():
    snapshot = load_baseline().snapshot
    counts = snapshot_counts(snapshot)
    assert all(project.project_id for project in snapshot.projects)
    assert all(isinstance(project.evidence_facts, list) for project in snapshot.projects)
    assert all(isinstance(project.claim_boundaries, list) for project in snapshot.projects)
    assert all(isinstance(project.capability_facts, list) for project in snapshot.projects)
    assert counts["project_count"] == len(snapshot.projects)
    assert counts["evidence_fact_count"] == snapshot.diagnostics.evidence_fact_count
    assert counts["claim_boundary_count"] == snapshot.diagnostics.claim_boundary_count
    assert counts["capability_fact_count"] == snapshot.diagnostics.capability_fact_count


def test_current_real_project_evidence_statistics_are_locked():
    snapshot = load_baseline().snapshot
    assert snapshot_counts(snapshot) == EXPECTED_BASELINE

    evidence_ids = [
        fact.evidence_fact_id for project in snapshot.projects for fact in project.evidence_facts
    ]
    boundary_ids = [
        boundary.boundary_id for project in snapshot.projects for boundary in project.claim_boundaries
    ]
    capability_ids = [
        fact.capability_id for project in snapshot.projects for fact in project.capability_facts
    ]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert len(boundary_ids) == len(set(boundary_ids))
    assert len(capability_ids) == len(set(capability_ids))


def test_project_evidence_memory_is_not_modified_by_baseline_checks(monkeypatch):
    before_bytes = ARTIFACT.read_bytes()
    before_digest = hashlib.sha256(before_bytes).hexdigest()
    before_mtime = ARTIFACT.stat().st_mtime_ns

    def persistence_is_forbidden(*_args, **_kwargs):
        raise AssertionError("baseline validation must not persist project evidence memory")

    monkeypatch.setattr(
        project_evidence_memory,
        "persist_project_evidence_memory",
        persistence_is_forbidden,
    )
    loaded = load_baseline()
    snapshot_counts(loaded.snapshot)

    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == before_digest
    assert ARTIFACT.read_bytes() == before_bytes
    assert ARTIFACT.stat().st_mtime_ns == before_mtime


@pytest.mark.parametrize(
    ("payload", "status", "error"),
    [
        (b"not json", "invalid", "malformed_artifact"),
        (b"{}", "invalid", "missing_schema_version"),
        (
            b'{"schema_version":"project_evidence_memory.v1","content_hash":"","project_count":0,"projects":{},"diagnostics":{}}',
            "invalid",
            "project_count_mismatch",
        ),
    ],
)
def test_authoritative_loader_fails_closed_for_invalid_baselines(tmp_path, payload, status, error):
    artifact = tmp_path / "project_evidence_memory.json"
    artifact.write_bytes(payload)
    loaded = project_evidence_memory.load_project_evidence_memory(artifact)
    assert loaded.status == status
    assert loaded.snapshot is None
    assert error in loaded.validation.errors


def test_authoritative_loader_reports_a_missing_baseline(tmp_path):
    loaded = project_evidence_memory.load_project_evidence_memory(
        tmp_path / "project_evidence_memory.json"
    )
    assert loaded.status == "missing"
    assert loaded.snapshot is None
    assert loaded.validation.errors == ("artifact_missing",)


def test_project_capability_baseline_does_not_use_numbered_artifacts():
    forbidden = set()
    for number in (4, 5):
        label = "phase" + str(number)
        forbidden.update({label, label + ".v1", "project_memory_" + label + ".json"})
    source_paths = [
        *sorted((ROOT / "backend").glob("*.py")),
        *sorted(
            path
            for path in (ROOT / "script").glob("*.py")
            if not path.name.startswith("migrate_legacy_")
        ),
        Path(__file__),
    ]
    violations = {
        str(path.relative_to(ROOT)): sorted(token for token in forbidden if token in path.read_text(encoding="utf-8").casefold())
        for path in source_paths
    }
    assert not {path: tokens for path, tokens in violations.items() if tokens}
