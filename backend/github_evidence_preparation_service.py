"""Explicit product workflow for preparing saved GitHub project evidence."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from backend.evidence_index_readiness import inspect_evidence_index_readiness
from backend.github_evidence_chunks import DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH
from backend.github_evidence_materializer import (
    DEFAULT_MATERIALIZATION_MANIFEST_PATH,
    DEFAULT_PROJECT_MEMORY_PATH,
    DEFAULT_SAVED_GITHUB_CONTEXT_PATH,
    MATERIALIZATION_SCHEMA_VERSION,
    inspect_saved_github_context_content_hash,
    is_github_evidence_materialization_enabled,
    list_saved_github_context_repositories,
    materialize_saved_github_evidence,
)
from backend.github_raw_storage import DEFAULT_GITHUB_RAW_SOURCES_PATH
from backend.project_repository_identity import (
    DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    authority_to_repository_mapping,
    load_project_repository_identity_authority,
)
from backend.project_repository_mapping_service import (
    DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
    list_unresolved_repository_mappings,
    load_repository_confirmations,
)


MAX_PREPARATION_WARNINGS = 16
MAX_PREPARATION_ERRORS = 8
MAX_PRODUCT_MESSAGE_CHARS = 240
MAX_SAFE_PROJECT_COUNT = 128
MAX_SAFE_REPOSITORY_COUNT = 128
MAX_SAVED_CONTEXT_BYTES = 10_000_000
MAX_PROJECT_MEMORY_BYTES = 5_000_000
MAX_MANIFEST_BYTES = 1_000_000
_PREPARATION_LOCK = threading.Lock()

_MESSAGES = {
    "disabled": "Evidence preparation is currently unavailable.",
    "mapping_required": "Connect all detected GitHub repositories to projects before preparing evidence.",
    "ready_to_prepare": "GitHub project evidence is ready to prepare.",
    "prepared": "GitHub project evidence is prepared.",
    "partial": "Some GitHub project evidence could not be prepared.",
    "blocked": "GitHub project evidence cannot be prepared until its saved information is valid.",
    "error": "GitHub project evidence status could not be checked.",
    "created": "GitHub project evidence was prepared.",
    "updated": "GitHub project evidence was refreshed.",
    "unchanged": "GitHub project evidence is already up to date.",
    "empty": "No saved GitHub evidence is available to prepare.",
    "busy": "Evidence preparation is already in progress.",
    "degraded": "GitHub project evidence was prepared, but its status could not be refreshed.",
}


def _message(status: str) -> str:
    return _MESSAGES.get(status, _MESSAGES["error"])[:MAX_PRODUCT_MESSAGE_CHARS]


def _load_json(path: str | Path, max_bytes: int) -> Any:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size > max_bytes:
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _feature_enabled(value: bool | None) -> bool:
    if type(value) is bool:
        return value
    if value is not None:
        return False
    return is_github_evidence_materialization_enabled()


def _safe_response(
    status: str, *, enabled: bool, can_prepare: bool = False,
    repository_mapping_complete: bool = False, requires_repository_mapping: bool = True,
    saved_github_context_available: bool = False, evidence_prepared: bool = False,
    preparation_complete: bool = False, preparation_incomplete: bool = False,
    prepared_project_count: int = 0, remaining_repository_count: int = 0,
    conflict_count: int = 0, ready_for_retrieval_setup: bool = False,
    warnings: list[str] | None = None, errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status, "enabled": enabled, "can_prepare": can_prepare,
        "repository_mapping_complete": repository_mapping_complete,
        "requires_repository_mapping": requires_repository_mapping,
        "saved_github_context_available": saved_github_context_available,
        "evidence_prepared": evidence_prepared,
        "preparation_complete": preparation_complete,
        "preparation_incomplete": preparation_incomplete,
        "prepared_project_count": min(max(int(prepared_project_count), 0), MAX_SAFE_PROJECT_COUNT),
        "remaining_repository_count": min(max(int(remaining_repository_count), 0), MAX_SAFE_REPOSITORY_COUNT),
        "conflict_count": min(max(int(conflict_count), 0), MAX_SAFE_REPOSITORY_COUNT),
        "ready_for_retrieval_setup": ready_for_retrieval_setup,
        "message": _message(status),
        "warnings": list(warnings or [])[:MAX_PREPARATION_WARNINGS],
        "errors": list(errors or [])[:MAX_PREPARATION_ERRORS],
    }


def _inspect_preparation_state(
    *, feature_enabled: bool | None,
    saved_context: Any, project_memory: Any, identity_authority: Any,
    confirmations: Any, vector_store: Any, vector_records: Any,
    raw_source_path: str | Path, chunk_path: str | Path, manifest_path: str | Path,
    readiness_inspector: Callable[..., Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    enabled = _feature_enabled(feature_enabled)
    saved_repositories = list_saved_github_context_repositories(saved_context)
    saved_available = bool(saved_repositories)
    authority_mapping = authority_to_repository_mapping(identity_authority)
    unresolved = list_unresolved_repository_mappings(
        vector_store=vector_store, vector_records=vector_records,
        identity_authority=identity_authority,
    )
    mapped_repositories = authority_mapping["repository_to_project"]
    saved_unmapped = [repository for repository in saved_repositories if repository not in mapped_repositories]
    conflict_count = len(authority_mapping["conflicts"]) + int(unresolved.get("conflict_count", 0))
    remaining = max(int(unresolved.get("unresolved_count", 0)), len(saved_unmapped))
    authority_valid = isinstance(identity_authority, Mapping) and bool(mapped_repositories)
    confirmations_valid = isinstance(confirmations, Mapping)
    mapping_complete = authority_valid and confirmations_valid and not remaining and not conflict_count
    internal = {
        "enabled": enabled, "saved_repositories": saved_repositories,
        "authority_mapping": authority_mapping, "mapping_complete": mapping_complete,
    }
    common = {
        "enabled": enabled, "repository_mapping_complete": mapping_complete,
        "requires_repository_mapping": not mapping_complete,
        "saved_github_context_available": saved_available,
        "remaining_repository_count": remaining, "conflict_count": conflict_count,
    }
    if not enabled:
        return _safe_response("disabled", **common), internal
    if not confirmations_valid:
        return _safe_response("blocked", errors=["repository_confirmation_state_invalid"], **common), internal
    if conflict_count:
        return _safe_response("blocked", errors=["repository_association_conflict"], **common), internal
    if not mapping_complete:
        return _safe_response("mapping_required", **common), internal
    if not saved_available:
        return _safe_response("blocked", errors=["saved_evidence_unavailable"], **common), internal

    paths = [Path(raw_source_path), Path(chunk_path), Path(manifest_path)]
    existence = [path.is_file() for path in paths]
    if any(existence) and not all(existence):
        return _safe_response("blocked", errors=["prepared_evidence_state_incomplete"], **common), internal
    if not any(existence):
        return _safe_response("ready_to_prepare", can_prepare=True, **common), internal
    manifest = _load_json(manifest_path, MAX_MANIFEST_BYTES) if all(existence) else None
    if all(existence) and not isinstance(manifest, Mapping):
        return _safe_response("blocked", errors=["prepared_evidence_state_invalid"], **common), internal
    current_hash, input_status = inspect_saved_github_context_content_hash(
        saved_context=saved_context, authoritative_mapping=authority_mapping,
    )
    if input_status == "partial":
        return _safe_response(
            "partial", can_prepare=True, preparation_incomplete=True,
            errors=["saved_evidence_limit_reached"], **common,
        ), internal
    if input_status != "ready" or not current_hash:
        return _safe_response("blocked", errors=["saved_evidence_invalid"], **common), internal
    if manifest.get("source_content_hash") != current_hash:
        return _safe_response("ready_to_prepare", can_prepare=True, **common), internal
    readiness = readiness_inspector(
        project_memory=project_memory, vector_store=vector_store, vector_records=vector_records,
        raw_source_path=raw_source_path, chunk_path=chunk_path, manifest=manifest,
        manifest_path=manifest_path, identity_authority=identity_authority,
    )
    internal["readiness"] = readiness
    if readiness.get("status") == "error":
        return _safe_response("error", errors=["preparation_status_unavailable"], **common), internal
    if manifest is not None and manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        return _safe_response("blocked", errors=["prepared_evidence_state_invalid"], **common), internal
    if manifest is not None and manifest.get("status") == "partial":
        return _safe_response(
            "partial", can_prepare=True, evidence_prepared=True,
            preparation_incomplete=True,
            prepared_project_count=len(readiness.get("projects_with_chunks", [])), **common,
        ), internal
    if manifest is not None and manifest.get("status") == "ready" and readiness.get("chunk_mapping_ready"):
        return _safe_response(
            "prepared", can_prepare=True, evidence_prepared=True, preparation_complete=True,
            prepared_project_count=len(readiness.get("projects_with_chunks", [])),
            ready_for_retrieval_setup=not bool(readiness.get("records_unresolved")), **common,
        ), internal
    if manifest is not None:
        return _safe_response("blocked", errors=["prepared_evidence_state_invalid"], **common), internal
    return _safe_response("blocked", errors=["prepared_evidence_state_invalid"], **common), internal


def get_github_evidence_preparation_status(
    *, feature_enabled: bool | None = None,
    saved_context: Any = None, project_memory: Any = None,
    identity_authority: Any = None, confirmations: Any = None,
    vector_store: Any = None, vector_records: Any = None,
    saved_context_path: str | Path = DEFAULT_SAVED_GITHUB_CONTEXT_PATH,
    project_memory_path: str | Path = DEFAULT_PROJECT_MEMORY_PATH,
    identity_authority_path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    confirmation_path: str | Path = DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
    raw_source_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
    chunk_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    manifest_path: str | Path = DEFAULT_MATERIALIZATION_MANIFEST_PATH,
    readiness_inspector: Callable[..., Mapping[str, Any]] = inspect_evidence_index_readiness,
) -> dict[str, Any]:
    try:
        if saved_context is None:
            saved_context = _load_json(saved_context_path, MAX_SAVED_CONTEXT_BYTES)
        if project_memory is None:
            project_memory = _load_json(project_memory_path, MAX_PROJECT_MEMORY_BYTES)
        if identity_authority is None:
            identity_authority = load_project_repository_identity_authority(identity_authority_path)
        if confirmations is None:
            confirmations = load_repository_confirmations(confirmation_path)
        if not isinstance(saved_context, Mapping) or not isinstance(project_memory, Mapping):
            return _safe_response(
                "disabled" if not _feature_enabled(feature_enabled) else "blocked",
                enabled=_feature_enabled(feature_enabled), errors=["saved_evidence_unavailable"],
            )
        result, _internal = _inspect_preparation_state(
            feature_enabled=feature_enabled, saved_context=saved_context,
            project_memory=project_memory, identity_authority=identity_authority,
            confirmations=confirmations, vector_store=vector_store, vector_records=vector_records,
            raw_source_path=raw_source_path, chunk_path=chunk_path, manifest_path=manifest_path,
            readiness_inspector=readiness_inspector,
        )
        return result
    except Exception:
        return _safe_response("error", enabled=_feature_enabled(feature_enabled), errors=["preparation_status_unavailable"])


@contextmanager
def _preparation_file_lock(manifest_path: Path):
    lock_path = manifest_path.with_name(f".{manifest_path.name}.preparation.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def run_github_evidence_preparation(
    *, confirmed: Any, feature_enabled: bool | None = None,
    saved_context: Any = None, project_memory: Any = None,
    identity_authority: Any = None, confirmations: Any = None,
    vector_store: Any = None, vector_records: Any = None,
    saved_context_path: str | Path = DEFAULT_SAVED_GITHUB_CONTEXT_PATH,
    project_memory_path: str | Path = DEFAULT_PROJECT_MEMORY_PATH,
    identity_authority_path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    confirmation_path: str | Path = DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
    raw_source_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
    chunk_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    manifest_path: str | Path = DEFAULT_MATERIALIZATION_MANIFEST_PATH,
    materializer: Callable[..., Mapping[str, Any]] = materialize_saved_github_evidence,
    readiness_inspector: Callable[..., Mapping[str, Any]] = inspect_evidence_index_readiness,
) -> dict[str, Any]:
    if confirmed is not True:
        return _safe_response("blocked", enabled=_feature_enabled(feature_enabled), errors=["explicit_confirmation_required"])
    if not _PREPARATION_LOCK.acquire(blocking=False):
        return _safe_response("busy", enabled=_feature_enabled(feature_enabled))
    try:
        try:
            if saved_context is None:
                saved_context = _load_json(saved_context_path, MAX_SAVED_CONTEXT_BYTES)
            if project_memory is None:
                project_memory = _load_json(project_memory_path, MAX_PROJECT_MEMORY_BYTES)
            if identity_authority is None:
                identity_authority = load_project_repository_identity_authority(identity_authority_path)
            if confirmations is None:
                confirmations = load_repository_confirmations(confirmation_path)
            if not isinstance(saved_context, Mapping) or not isinstance(project_memory, Mapping):
                return _safe_response("blocked", enabled=_feature_enabled(feature_enabled), errors=["saved_evidence_unavailable"])
            preflight, internal = _inspect_preparation_state(
                feature_enabled=feature_enabled, saved_context=saved_context,
                project_memory=project_memory, identity_authority=identity_authority,
                confirmations=confirmations, vector_store=vector_store, vector_records=vector_records,
                raw_source_path=raw_source_path, chunk_path=chunk_path, manifest_path=manifest_path,
                readiness_inspector=readiness_inspector,
            )
            if preflight["status"] not in {"ready_to_prepare", "prepared", "partial"}:
                return preflight
            output_paths = [Path(raw_source_path), Path(chunk_path), Path(manifest_path)]
            if any(not path.parent.is_dir() for path in output_paths):
                return _safe_response("blocked", enabled=True, errors=["evidence_storage_unavailable"])
            try:
                with _preparation_file_lock(Path(manifest_path)):
                    result = materializer(
                        saved_context=saved_context, project_memory=project_memory,
                        authoritative_mapping=internal["authority_mapping"],
                        saved_context_path=saved_context_path, project_memory_path=project_memory_path,
                        raw_source_path=raw_source_path, chunk_path=chunk_path,
                        manifest_path=manifest_path, feature_enabled=True,
                    )
            except FileExistsError:
                return _safe_response("busy", enabled=True)
            except Exception:
                return _safe_response("error", enabled=True, errors=["evidence_preparation_failed"])
            materialization_status = str(result.get("status", "error"))
            if materialization_status not in {"created", "updated", "unchanged", "partial"}:
                translated = materialization_status if materialization_status in {"disabled", "empty", "blocked"} else "error"
                return _safe_response(
                    translated, enabled=True,
                    repository_mapping_complete=True, requires_repository_mapping=False,
                    saved_github_context_available=True,
                    errors=[] if translated in {"disabled", "empty", "blocked"} else ["evidence_preparation_failed"],
                )
            try:
                readiness = readiness_inspector(
                    project_memory=project_memory, vector_store=vector_store, vector_records=vector_records,
                    raw_source_path=raw_source_path, chunk_path=chunk_path,
                    manifest_path=manifest_path, identity_authority=identity_authority,
                )
            except Exception:
                readiness = {"status": "error"}
            if readiness.get("status") == "error":
                return _safe_response(
                    "degraded", enabled=True, repository_mapping_complete=True,
                    requires_repository_mapping=False, saved_github_context_available=True,
                    evidence_prepared=True, preparation_complete=materialization_status != "partial",
                    preparation_incomplete=materialization_status == "partial",
                    errors=["preparation_status_unavailable"],
                )
            complete = materialization_status != "partial" and bool(readiness.get("chunk_mapping_ready"))
            return _safe_response(
                materialization_status, enabled=True, can_prepare=True,
                repository_mapping_complete=True, requires_repository_mapping=False,
                saved_github_context_available=True, evidence_prepared=True,
                preparation_complete=complete, preparation_incomplete=not complete,
                prepared_project_count=len(readiness.get("projects_with_chunks", [])),
                ready_for_retrieval_setup=complete and not bool(readiness.get("records_unresolved")),
                warnings=["preparation_incomplete"] if not complete else [],
            )
        except Exception:
            return _safe_response("error", enabled=_feature_enabled(feature_enabled), errors=["evidence_preparation_failed"])
    finally:
        _PREPARATION_LOCK.release()
