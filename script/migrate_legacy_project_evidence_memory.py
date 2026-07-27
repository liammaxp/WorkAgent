"""One-time, fail-closed migration for the legacy project-evidence artifact.

The legacy schema and path constants below are intentionally isolated here.
Normal production readers and writers accept only semantic schema names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import project_evidence_memory  # noqa: E402
from backend import project_evidence_pipeline  # noqa: E402


LEGACY_SCHEMA_VERSION = "phase4.v1"
LEGACY_ARTIFACT_PATH = ROOT_DIR / "information" / "project_memory_phase4.json"
SEMANTIC_ARTIFACT_PATH = ROOT_DIR / "information" / "project_evidence_memory.json"


@dataclass(frozen=True)
class ArtifactTotals:
    project_count: int
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int


@dataclass(frozen=True)
class LegacyArtifactSummary:
    content_hash: str
    totals: ArtifactTotals


@dataclass(frozen=True)
class MigrationResult:
    status: str
    old_content_hash: str
    new_content_hash: str
    totals: ArtifactTotals
    old_artifact_removed: bool


def _legacy_snapshot_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_hash"}


def _artifact_totals(projects: list[Any]) -> ArtifactTotals:
    evidence_count = 0
    capability_count = 0
    boundary_count = 0
    for project in projects:
        if not isinstance(project, Mapping):
            raise ValueError("legacy_project_shape_invalid")
        evidence = project.get("evidence_facts")
        capabilities = project.get("capability_facts")
        boundaries = project.get("claim_boundaries")
        if not all(isinstance(items, list) for items in (evidence, capabilities, boundaries)):
            raise ValueError("legacy_record_collections_invalid")
        evidence_count += len(evidence)
        capability_count += len(capabilities)
        boundary_count += len(boundaries)
    return ArtifactTotals(
        project_count=len(projects),
        evidence_fact_count=evidence_count,
        capability_fact_count=capability_count,
        claim_boundary_count=boundary_count,
    )


def load_and_validate_legacy_artifact(
    path: str | Path = LEGACY_ARTIFACT_PATH,
) -> LegacyArtifactSummary:
    """Validate the one supported legacy schema without accepting it normally."""

    source = Path(path)
    if not source.is_file():
        raise ValueError("legacy_artifact_missing")
    if source.stat().st_size > project_evidence_memory.MAX_SERIALIZED_SIZE:
        raise ValueError("legacy_artifact_too_large")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=project_evidence_memory._reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("legacy_artifact_malformed") from error
    if not isinstance(payload, dict):
        raise ValueError("legacy_artifact_shape_invalid")
    required = {"schema_version", "content_hash", "project_count", "projects", "diagnostics"}
    if set(payload) != required or payload.get("schema_version") != LEGACY_SCHEMA_VERSION:
        raise ValueError("legacy_schema_unsupported")
    projects = payload.get("projects")
    project_count = payload.get("project_count")
    if (
        not isinstance(projects, list)
        or isinstance(project_count, bool)
        or not isinstance(project_count, int)
        or project_count != len(projects)
    ):
        raise ValueError("legacy_project_count_invalid")
    project_evidence_memory._reject_prohibited_content(payload)
    content_hash = payload.get("content_hash")
    expected_hash = project_evidence_memory._payload_hash(_legacy_snapshot_payload(payload))
    if content_hash != expected_hash:
        raise ValueError("legacy_content_hash_mismatch")
    return LegacyArtifactSummary(
        content_hash=content_hash,
        totals=_artifact_totals(projects),
    )


def migrate_legacy_project_evidence_memory(
    *,
    legacy_path: str | Path = LEGACY_ARTIFACT_PATH,
    destination: str | Path = SEMANTIC_ARTIFACT_PATH,
) -> MigrationResult:
    """Rebuild semantic memory, validate equal totals, then remove the old file."""

    source = Path(legacy_path)
    target = Path(destination)
    legacy = load_and_validate_legacy_artifact(source)
    pipeline_result = project_evidence_pipeline.run_project_evidence_pipeline(
        output_path=target,
        persist=True,
        environ={project_evidence_pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: "1"},
    )
    if pipeline_result.status in {"disabled", "empty", "error"}:
        raise ValueError("semantic_rebuild_failed")
    loaded = project_evidence_memory.load_project_evidence_memory(target)
    if loaded.status != "ready" or loaded.snapshot is None or not loaded.validation.valid:
        raise ValueError("semantic_artifact_validation_failed")
    snapshot = loaded.snapshot
    semantic_totals = ArtifactTotals(
        project_count=len(snapshot.projects),
        evidence_fact_count=sum(len(project.evidence_facts) for project in snapshot.projects),
        capability_fact_count=sum(len(project.capability_facts) for project in snapshot.projects),
        claim_boundary_count=sum(len(project.claim_boundaries) for project in snapshot.projects),
    )
    if semantic_totals != legacy.totals:
        raise ValueError("semantic_record_totals_changed")
    source.unlink()
    return MigrationResult(
        status=pipeline_result.persistence_status or pipeline_result.status,
        old_content_hash=legacy.content_hash,
        new_content_hash=snapshot.content_hash,
        totals=semantic_totals,
        old_artifact_removed=not source.exists(),
    )


if __name__ == "__main__":
    result = migrate_legacy_project_evidence_memory()
    print(json.dumps({
        "status": result.status,
        "old_content_hash": result.old_content_hash,
        "new_content_hash": result.new_content_hash,
        "project_count": result.totals.project_count,
        "evidence_fact_count": result.totals.evidence_fact_count,
        "capability_fact_count": result.totals.capability_fact_count,
        "claim_boundary_count": result.totals.claim_boundary_count,
        "old_artifact_removed": result.old_artifact_removed,
    }, sort_keys=True))
