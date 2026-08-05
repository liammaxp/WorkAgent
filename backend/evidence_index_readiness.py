"""Read-only readiness checks for vector identity and materialized evidence chunks."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from backend.github_evidence_chunks import load_github_evidence_chunk_records
from backend.github_raw_storage import load_github_raw_source_records
from backend.project_repository_identity import (
    RepositoryAuthorityMapping as AuthoritativeRepositoryMapping,
    authority_to_repository_mapping,
    build_authoritative_repository_project_mapping,
    load_project_repository_identity_authority,
    normalize_project_id,
    normalize_repository_identity,
)


MAX_VECTOR_METADATA_RECORDS = 10000
MAX_READINESS_PROJECTS = 256
MAX_READINESS_CODES = 32
_VECTOR_IDENTITY_KEYS = frozenset({
    "github_repository", "project_id", "project_name", "repo", "repository", "repository_url",
})


class EvidenceIndexReadinessResult(TypedDict):
    status: str
    vector_record_count: int
    records_with_explicit_project_id: int
    records_with_repository_only: int
    records_with_missing_identity: int
    records_with_conflicting_identity: int
    records_resolved_by_authoritative_mapping: int
    records_unresolved: int
    usable_vector_record_count: int
    authoritative_mapping_count: int
    mapping_conflict_count: int
    identity_authority_status: str
    identity_mapping_count: int
    identity_conflict_count: int
    identity_unresolved_count: int
    user_confirmation_required: bool
    raw_source_count: int
    chunk_count: int
    projects_with_chunks: list[str]
    projects_without_chunks: list[str]
    vector_ready: bool
    chunk_mapping_ready: bool
    ready_for_hybrid_retrieval: bool
    warnings: list[str]
    errors: list[str]


def inspect_github_evidence_vector_metadata(
    *,
    vector_store: Any = None,
    vector_records: Any = None,
    limit: int = MAX_VECTOR_METADATA_RECORDS,
) -> list[dict[str, Any]]:
    """Read only vector IDs and metadata through the existing store or injected records."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return []
    if vector_records is not None:
        records = vector_records
    else:
        reader = getattr(vector_store, "inspect_github_vector_metadata", None)
        if not callable(reader):
            return []
        records = reader(limit=min(limit, MAX_VECTOR_METADATA_RECORDS))
    if not isinstance(records, (list, tuple)):
        return []
    safe: list[dict[str, Any]] = []
    for record in records[: min(limit, MAX_VECTOR_METADATA_RECORDS)]:
        if not isinstance(record, Mapping):
            continue
        metadata = record.get("metadata")
        safe_metadata = {
            str(key).casefold(): str(value).strip()[:300]
            for key, value in metadata.items()
            if isinstance(metadata, Mapping)
            and str(key).casefold() in _VECTOR_IDENTITY_KEYS
            and isinstance(value, str)
            and value.strip()
        } if isinstance(metadata, Mapping) else {}
        record_id = str(record.get("vector_record_id") or record.get("id") or "")[:180]
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", record_id):
            record_id = ""
        safe.append({
            "vector_record_id": record_id,
            "metadata": safe_metadata,
        })
    return safe


def _record_identity(
    record: Mapping[str, Any], mapping: AuthoritativeRepositoryMapping,
) -> tuple[str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return "missing_identity", ""
    explicit_values = {
        value.casefold(): value
        for key in ("project_id", "project_name")
        if (value := normalize_project_id(metadata.get(key)))
    }
    repository_values = set()
    for key in ("repo", "repository", "repository_url", "github_repository"):
        raw = metadata.get(key)
        repository = normalize_repository_identity(raw)
        if not repository and isinstance(raw, str):
            repository = mapping.get("alias_to_repository", {}).get(raw.strip().casefold(), "")
        if repository:
            repository_values.add(repository)
    if len(explicit_values) > 1 or len(repository_values) > 1:
        return "conflicting_identity", ""
    explicit = next(iter(explicit_values.values()), "")
    repository = next(iter(repository_values), "")
    if explicit:
        mapped = mapping["repository_to_project"].get(repository) if repository else None
        if repository in mapping["conflicts"] or (mapped and mapped.casefold() != explicit.casefold()):
            return "conflicting_identity", ""
        return "explicit_project_id", explicit
    if repository:
        if repository in mapping["conflicts"]:
            return "conflicting_identity", ""
        mapped = mapping["repository_to_project"].get(repository)
        if mapped:
            return "resolved_by_authoritative_mapping", mapped
        return "repository_only", ""
    return "missing_identity", ""


def _load_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def inspect_evidence_index_readiness(
    *,
    project_memory: Any,
    vector_store: Any = None,
    vector_records: Any = None,
    raw_sources: Any = None,
    chunks: Any = None,
    raw_source_path: str | Path | None = None,
    chunk_path: str | Path | None = None,
    manifest: Any = None,
    manifest_path: str | Path | None = None,
    intended_project_ids: Sequence[str] | None = None,
    identity_authority: Any = None,
    identity_authority_path: str | Path | None = None,
) -> EvidenceIndexReadinessResult:
    """Evaluate independent vector-identity and chunk-index readiness without building either."""

    warnings: list[str] = []
    errors: list[str] = []
    try:
        if identity_authority is None and identity_authority_path is not None:
            identity_authority = load_project_repository_identity_authority(identity_authority_path)
        mapping = authority_to_repository_mapping(identity_authority)
        records = inspect_github_evidence_vector_metadata(
            vector_store=vector_store, vector_records=vector_records
        )
        if raw_sources is None:
            raw_sources = load_github_raw_source_records(raw_source_path) if raw_source_path else []
        if chunks is None:
            chunks = load_github_evidence_chunk_records(chunk_path) if chunk_path else []
        if manifest is None:
            manifest = _load_manifest(manifest_path)
    except Exception:
        return _error_readiness("readiness_inspection_failed")
    raw_values = raw_sources if isinstance(raw_sources, (list, tuple)) else []
    chunk_values = chunks if isinstance(chunks, (list, tuple)) else []
    counts = {
        "explicit_project_id": 0,
        "repository_only": 0,
        "missing_identity": 0,
        "conflicting_identity": 0,
        "resolved_by_authoritative_mapping": 0,
    }
    usable_projects: set[str] = set()
    for record in records:
        classification, project_id = _record_identity(record, mapping)
        counts[classification] += 1
        if project_id:
            usable_projects.add(project_id)
    unresolved = counts["repository_only"] + counts["missing_identity"] + counts["conflicting_identity"]
    chunk_projects = {
        project_id
        for item in chunk_values
        if isinstance(item, Mapping)
        and (project_id := normalize_project_id(item.get("project_id")))
        and normalize_project_id(item.get("chunk_id"))
    }
    intended = {
        project_id
        for value in (intended_project_ids or [])
        if (project_id := normalize_project_id(value))
    }
    if not intended:
        intended = set(mapping["repository_to_project"].values()) or usable_projects or chunk_projects
    missing_chunks = sorted(intended - chunk_projects)[:MAX_READINESS_PROJECTS]
    conflicts = len(mapping["conflicts"]) + counts["conflicting_identity"]
    authority_status = identity_authority.get("status", "missing") if isinstance(identity_authority, Mapping) else "missing"
    authority_unresolved = (
        len(identity_authority.get("unresolved_projects", [])) + len(identity_authority.get("unresolved_repositories", []))
        if isinstance(identity_authority, Mapping) else 0
    )
    vector_ready = bool(
        records and usable_projects and not unresolved and not conflicts and intended <= usable_projects
    )
    manifest_required = manifest_path is not None
    manifest_ready = (
        not manifest_required and manifest is None
    ) or (
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == "github_evidence_materialization.v1"
        and manifest.get("status") == "ready"
        and manifest.get("raw_source_count") == len(raw_values)
        and manifest.get("chunk_count") == len(chunk_values)
    )
    if manifest_ready and isinstance(manifest, Mapping):
        for artifact_path, hash_key in (
            (raw_source_path, "raw_artifact_hash"), (chunk_path, "chunk_artifact_hash"),
        ):
            if artifact_path is None:
                continue
            candidate = Path(artifact_path)
            expected_hash = manifest.get(hash_key)
            if (
                not candidate.exists() or not isinstance(expected_hash, str)
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_hash
            ):
                manifest_ready = False
                break
    if not manifest_ready:
        warnings.append("materialization_manifest_mismatch")
    chunk_ready = bool(chunk_values and chunk_projects and not missing_chunks and manifest_ready)
    hybrid_ready = vector_ready and chunk_ready
    if not records and not chunk_values:
        status = "empty"
    elif hybrid_ready and not unresolved:
        status = "ready"
    elif conflicts or not usable_projects or not chunk_projects:
        status = "blocked"
    else:
        status = "partial"
    return {
        "status": status,
        "vector_record_count": len(records),
        "records_with_explicit_project_id": counts["explicit_project_id"],
        "records_with_repository_only": counts["repository_only"] + counts["resolved_by_authoritative_mapping"],
        "records_with_missing_identity": counts["missing_identity"],
        "records_with_conflicting_identity": counts["conflicting_identity"],
        "records_resolved_by_authoritative_mapping": counts["resolved_by_authoritative_mapping"],
        "records_unresolved": unresolved,
        "usable_vector_record_count": len(records) - unresolved,
        "authoritative_mapping_count": mapping["mapping_count"],
        "mapping_conflict_count": len(mapping["conflicts"]),
        "identity_authority_status": str(authority_status)[:32],
        "identity_mapping_count": mapping["mapping_count"],
        "identity_conflict_count": len(mapping["conflicts"]),
        "identity_unresolved_count": authority_unresolved,
        "user_confirmation_required": bool(unresolved or conflicts or not mapping["mapping_count"]),
        "raw_source_count": len(raw_values),
        "chunk_count": len(chunk_values),
        "projects_with_chunks": sorted(chunk_projects)[:MAX_READINESS_PROJECTS],
        "projects_without_chunks": missing_chunks,
        "vector_ready": vector_ready,
        "chunk_mapping_ready": chunk_ready,
        "ready_for_hybrid_retrieval": hybrid_ready,
        "warnings": sorted(set(warnings))[:MAX_READINESS_CODES],
        "errors": errors,
    }


def _error_readiness(code: str) -> EvidenceIndexReadinessResult:
    return {
        "status": "error", "vector_record_count": 0,
        "records_with_explicit_project_id": 0, "records_with_repository_only": 0,
        "records_with_missing_identity": 0, "records_with_conflicting_identity": 0,
        "records_resolved_by_authoritative_mapping": 0, "records_unresolved": 0,
        "usable_vector_record_count": 0, "authoritative_mapping_count": 0,
        "mapping_conflict_count": 0, "raw_source_count": 0, "chunk_count": 0,
        "identity_authority_status": "error", "identity_mapping_count": 0,
        "identity_conflict_count": 0, "identity_unresolved_count": 0,
        "user_confirmation_required": True,
        "projects_with_chunks": [], "projects_without_chunks": [], "vector_ready": False,
        "chunk_mapping_ready": False, "ready_for_hybrid_retrieval": False,
        "warnings": [], "errors": [code],
    }
