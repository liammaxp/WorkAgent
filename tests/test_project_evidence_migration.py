import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from script import migrate_legacy_project_evidence_memory as migration


def legacy_payload(*, content_hash: str = "") -> dict:
    payload = {
        "schema_version": migration.LEGACY_SCHEMA_VERSION,
        "project_count": 1,
        "projects": [{
            "project_id": "sample",
            "evidence_facts": [{}],
            "capability_facts": [],
            "claim_boundaries": [{}],
        }],
        "diagnostics": {},
    }
    payload["content_hash"] = content_hash or migration.project_evidence_memory._payload_hash(payload)
    return payload


def write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_loader_accepts_only_valid_exact_schema_and_hash(tmp_path):
    source = tmp_path / "legacy.json"
    write_payload(source, legacy_payload())
    summary = migration.load_and_validate_legacy_artifact(source)
    assert summary.totals == migration.ArtifactTotals(1, 1, 0, 1)

    bad_hash = legacy_payload(content_hash="0" * 64)
    write_payload(source, bad_hash)
    with pytest.raises(ValueError, match="legacy_content_hash_mismatch"):
        migration.load_and_validate_legacy_artifact(source)

    unsupported = legacy_payload()
    unsupported["schema_version"] = "unsupported.v1"
    write_payload(source, unsupported)
    with pytest.raises(ValueError, match="legacy_schema_unsupported"):
        migration.load_and_validate_legacy_artifact(source)


def test_migration_removes_old_artifact_only_after_valid_equal_semantic_totals(
    tmp_path, monkeypatch
):
    source = tmp_path / "legacy.json"
    target = tmp_path / "semantic.json"
    write_payload(source, legacy_payload())
    snapshot = SimpleNamespace(
        content_hash="a" * 64,
        projects=(SimpleNamespace(
            evidence_facts=({},),
            capability_facts=(),
            claim_boundaries=({},),
        ),),
    )
    monkeypatch.setattr(
        migration.project_evidence_pipeline,
        "run_project_evidence_pipeline",
        lambda **_kwargs: SimpleNamespace(status="ready", persistence_status="created"),
    )
    monkeypatch.setattr(
        migration.project_evidence_memory,
        "load_project_evidence_memory",
        lambda _path: SimpleNamespace(
            status="ready", snapshot=snapshot, validation=SimpleNamespace(valid=True)
        ),
    )
    result = migration.migrate_legacy_project_evidence_memory(
        legacy_path=source, destination=target
    )
    assert result.status == "created"
    assert result.totals == migration.ArtifactTotals(1, 1, 0, 1)
    assert result.old_artifact_removed and not source.exists()


def test_migration_preserves_old_artifact_when_semantic_totals_differ(tmp_path, monkeypatch):
    source = tmp_path / "legacy.json"
    write_payload(source, legacy_payload())
    snapshot = SimpleNamespace(
        content_hash="b" * 64,
        projects=(SimpleNamespace(
            evidence_facts=(), capability_facts=(), claim_boundaries=()
        ),),
    )
    monkeypatch.setattr(
        migration.project_evidence_pipeline,
        "run_project_evidence_pipeline",
        lambda **_kwargs: SimpleNamespace(status="ready", persistence_status="created"),
    )
    monkeypatch.setattr(
        migration.project_evidence_memory,
        "load_project_evidence_memory",
        lambda _path: SimpleNamespace(
            status="ready", snapshot=snapshot, validation=SimpleNamespace(valid=True)
        ),
    )
    with pytest.raises(ValueError, match="semantic_record_totals_changed"):
        migration.migrate_legacy_project_evidence_memory(
            legacy_path=source, destination=tmp_path / "semantic.json"
        )
    assert source.exists()
