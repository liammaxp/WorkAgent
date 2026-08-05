"""Backend workflow for explicit project and GitHub repository associations."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping

from backend.evidence_index_readiness import (
    inspect_evidence_index_readiness,
    inspect_github_evidence_vector_metadata,
)
from backend.project_repository_identity import (
    DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
    DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    authority_to_repository_mapping,
    build_project_repository_identity_authority,
    load_project_repository_identity_authority,
    normalize_project_id,
    normalize_repository_identity,
    write_project_repository_identity_authority,
)


CONFIRMATION_SCHEMA_VERSION = "project_repository_confirmations.v1"
MAX_UNRESOLVED_REPOSITORIES = 128
MAX_PROJECT_OPTIONS = 128
MAX_CONFIRMATIONS = 500
MAX_REPOSITORY_INPUT_CHARS = 500
MAX_PROJECT_LABEL_CHARS = 160
MAX_ALIASES_PER_CONFIRMATION = 16
MAX_WARNINGS = 32
MAX_ERRORS = 16
_SAFE_ALIAS_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_CONFIRMATION_KEYS = frozenset({"project_id", "repository", "aliases", "confirmed"})
_TRANSACTION_LOCK = threading.RLock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _empty_confirmation_artifact() -> dict[str, Any]:
    payload = {"schema_version": CONFIRMATION_SCHEMA_VERSION, "confirmations": []}
    payload["content_hash"] = _hash(payload)
    return payload


def _valid_confirmation_record(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "confirmation_id", "project_id", "repository", "aliases", "confirmed",
    }:
        return False
    project_id = normalize_project_id(value.get("project_id"))
    repository = normalize_repository_identity(value.get("repository"))
    aliases = value.get("aliases")
    if not project_id or not repository or value.get("confirmed") is not True:
        return False
    if not isinstance(aliases, list) or len(aliases) > MAX_ALIASES_PER_CONFIRMATION:
        return False
    if any(not isinstance(alias, str) or len(alias) > MAX_REPOSITORY_INPUT_CHARS for alias in aliases):
        return False
    normalized_aliases = sorted({
        normalize_repository_identity(alias) or _bare_repository_alias(alias).casefold()
        for alias in aliases
    })
    if not all(normalized_aliases) or aliases != normalized_aliases:
        return False
    core = {"project_id": project_id, "repository": repository, "aliases": aliases, "confirmed": True}
    return value.get("confirmation_id") == f"prc_{_hash(core)[:24]}"


def _valid_confirmation_artifact(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "content_hash", "confirmations"}:
        return False
    values = value.get("confirmations")
    if value.get("schema_version") != CONFIRMATION_SCHEMA_VERSION or not isinstance(values, list) or len(values) > MAX_CONFIRMATIONS:
        return False
    if not all(_valid_confirmation_record(item) for item in values):
        return False
    if values != sorted(values, key=lambda item: item["confirmation_id"]):
        return False
    unsigned = dict(value); unsigned.pop("content_hash", None)
    return value.get("content_hash") == _hash(unsigned)


def load_repository_confirmations(
    path: str | Path = DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return _empty_confirmation_artifact()
    try:
        if not candidate.is_file() or candidate.stat().st_size > 1_000_000:
            return None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return dict(payload) if _valid_confirmation_artifact(payload) else None
    except (OSError, ValueError, TypeError):
        return None


def _confirmation_record(project_id: str, repository: str, aliases: list[str]) -> dict[str, Any]:
    core = {
        "project_id": project_id, "repository": repository,
        "aliases": sorted(set(aliases))[:MAX_ALIASES_PER_CONFIRMATION], "confirmed": True,
    }
    return {"confirmation_id": f"prc_{_hash(core)[:24]}", **core}


def _confirmation_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "confirmations": sorted(records, key=lambda item: item["confirmation_id"])[:MAX_CONFIRMATIONS],
    }
    payload["content_hash"] = _hash(payload)
    return payload


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _restore_path(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
    else:
        _atomic_replace_bytes(path, previous)


@contextmanager
def _mapping_transaction_lock(confirmation_path: Path):
    lock_path = confirmation_path.with_name(f".{confirmation_path.name}.transaction.lock")
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


def _project_records(project_memory: Any) -> list[Mapping[str, Any]]:
    projects = project_memory.get("projects") if isinstance(project_memory, Mapping) else None
    return [item for item in projects[:MAX_PROJECT_OPTIONS] if isinstance(item, Mapping)] if isinstance(projects, (list, tuple)) else []


def _known_projects(project_memory: Any) -> set[str]:
    return {project_id for item in _project_records(project_memory) if (project_id := normalize_project_id(item.get("project_id")))}


def load_project_memory_for_repository_mapping(path: str | Path) -> dict[str, Any] | None:
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size > 5_000_000:
            return None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, Mapping) else None
    except (OSError, ValueError, TypeError):
        return None


def list_repository_mapping_projects(*, project_memory: Any, identity_authority: Any = None) -> dict[str, Any]:
    mapping = authority_to_repository_mapping(identity_authority)
    linked: dict[str, list[str]] = {}
    for repository, project_id in mapping["repository_to_project"].items():
        linked.setdefault(project_id.casefold(), []).append(repository)
    raw_projects = project_memory.get("projects") if isinstance(project_memory, Mapping) else None
    limited = isinstance(raw_projects, (list, tuple)) and len(raw_projects) > MAX_PROJECT_OPTIONS
    projects = []
    for item in _project_records(project_memory):
        project_id = normalize_project_id(item.get("project_id"))
        if not project_id:
            continue
        raw_name = item.get("project_name")
        name = "".join(char for char in raw_name.strip() if ord(char) >= 32 and ord(char) != 127)[:MAX_PROJECT_LABEL_CHARS] if isinstance(raw_name, str) else ""
        projects.append({
            "project_id": project_id, "project_name": name or project_id,
            "already_linked_repositories": sorted(linked.get(project_id.casefold(), [])),
        })
    projects.sort(key=lambda item: (item["project_name"].casefold(), item["project_id"].casefold()))
    selected = projects[:MAX_PROJECT_OPTIONS]
    return {
        "status": "partial" if limited else "ready" if selected else "empty",
        "projects": selected, "count": len(selected),
        "warnings": ["project_output_limit_reached"] if limited else [], "errors": [],
    }


def _bare_repository_alias(value: Any) -> str:
    if not isinstance(value, str) or len(value) > MAX_REPOSITORY_INPUT_CHARS:
        return ""
    alias = "".join(char for char in value.strip() if ord(char) >= 32 and ord(char) != 127)
    return alias if _SAFE_ALIAS_RE.fullmatch(alias) else ""


def list_unresolved_repository_mappings(
    *, vector_store: Any = None, vector_records: Any = None, identity_authority: Any = None,
) -> dict[str, Any]:
    records = inspect_github_evidence_vector_metadata(vector_store=vector_store, vector_records=vector_records)
    mapping = authority_to_repository_mapping(identity_authority)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or normalize_project_id(metadata.get("project_id")) or normalize_project_id(metadata.get("project_name")):
            continue
        raw_values = [metadata.get(key) for key in ("repository", "repo", "repository_url", "github_repository") if metadata.get(key)]
        for raw in raw_values:
            canonical = normalize_repository_identity(raw)
            if canonical:
                if canonical in mapping["repository_to_project"] and canonical not in mapping["conflicts"]:
                    continue
                key = ("canonical", canonical)
                item = grouped.setdefault(key, {
                    "repository": canonical, "repository_alias": None, "repository_aliases": [],
                    "canonical": True, "requires_canonical_repository": False,
                    "vector_record_count": 0, "currently_mapped": False,
                    "conflicting": canonical in mapping["conflicts"], "requires_confirmation": True,
                })
            else:
                alias = _bare_repository_alias(raw)
                if not alias:
                    continue
                resolved_repository = mapping["alias_to_repository"].get(alias.casefold())
                if resolved_repository and resolved_repository not in mapping["conflicts"]:
                    continue
                key = ("alias", alias.casefold())
                item = grouped.setdefault(key, {
                    "repository": None, "repository_alias": alias, "repository_aliases": [],
                    "canonical": False, "requires_canonical_repository": True,
                    "vector_record_count": 0, "currently_mapped": False,
                    "conflicting": alias.casefold() in mapping["conflicts"], "requires_confirmation": True,
                })
            item["vector_record_count"] += 1
    repositories = sorted(grouped.values(), key=lambda item: (not item["canonical"], (item["repository"] or item["repository_alias"]).casefold()))
    limited = len(repositories) > MAX_UNRESOLVED_REPOSITORIES
    repositories = repositories[:MAX_UNRESOLVED_REPOSITORIES]
    conflicts = sum(1 for item in repositories if item["conflicting"])
    warnings = ["repository_output_limit_reached"] if limited else []
    return {
        "status": "partial" if limited else "blocked" if repositories else "empty",
        "repositories": repositories, "unresolved_count": len(repositories),
        "conflict_count": conflicts, "confirmation_required": bool(repositories),
        "warnings": warnings, "errors": [],
    }


def _validate_confirmation_request(project_memory: Any, request: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(request, Mapping) or set(request) - _CONFIRMATION_KEYS:
        return None, "invalid_confirmation_request"
    if request.get("confirmed") is not True:
        return None, "explicit_confirmation_required"
    project_id = normalize_project_id(request.get("project_id"))
    if not project_id or project_id not in _known_projects(project_memory):
        return None, "unknown_project"
    repository_input = request.get("repository")
    if not isinstance(repository_input, str) or len(repository_input) > MAX_REPOSITORY_INPUT_CHARS:
        return None, "invalid_repository"
    repository = normalize_repository_identity(repository_input)
    if not repository:
        return None, "canonical_repository_required"
    raw_aliases = request.get("aliases", [])
    if not isinstance(raw_aliases, (list, tuple)) or len(raw_aliases) > MAX_ALIASES_PER_CONFIRMATION:
        return None, "invalid_aliases"
    aliases = []
    for raw in raw_aliases:
        canonical_alias = normalize_repository_identity(raw)
        bare_alias = _bare_repository_alias(raw)
        alias = canonical_alias or bare_alias.casefold()
        if not alias:
            return None, "invalid_aliases"
        aliases.append(alias)
    return _confirmation_record(project_id, repository, aliases), ""


def _safe_error(code: str) -> dict[str, Any]:
    return {
        "status": "error", "mapping": None, "authority_status": "error",
        "identity_mapping_count": 0, "records_resolved_by_authoritative_mapping": 0,
        "records_unresolved": 0, "identity_ready": False,
        "materialization_required": True, "ready_for_hybrid_retrieval": False,
        "warnings": [], "errors": [code][:MAX_ERRORS],
    }


def confirm_repository_mapping(
    *, project_memory: Any, request: Any, vector_store: Any = None, vector_records: Any = None,
    raw_sources: Any = None, chunks: Any = None,
    raw_source_path: str | Path | None = None, chunk_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    confirmation_path: str | Path = DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH,
    authority_path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    readiness_inspector=inspect_evidence_index_readiness,
) -> dict[str, Any]:
    record, error = _validate_confirmation_request(project_memory, request)
    if error or record is None:
        result = _safe_error(error or "invalid_confirmation_request"); result["status"] = "blocked"; return result
    confirmation_target = Path(confirmation_path); authority_target = Path(authority_path)
    if not confirmation_target.parent.exists() or not authority_target.parent.exists():
        return _safe_error("artifact_parent_missing")
    with _TRANSACTION_LOCK:
        try:
            with _mapping_transaction_lock(confirmation_target):
                existing = load_repository_confirmations(confirmation_target)
                if existing is None:
                    return _safe_error("confirmation_artifact_invalid")
                prior_confirmation = confirmation_target.read_bytes() if confirmation_target.exists() else None
                prior_authority = authority_target.read_bytes() if authority_target.exists() else None
                existing_authority = load_project_repository_identity_authority(authority_target)
                if prior_authority is not None and existing_authority is None:
                    return _safe_error("identity_authority_invalid")
                records = list(existing["confirmations"])
                duplicate = any(item == record for item in records)
                if not duplicate:
                    records.append(record)
                if len(records) > MAX_CONFIRMATIONS:
                    result = _safe_error("confirmation_limit_reached"); result["status"] = "blocked"; return result
                confirmations = _confirmation_artifact(records)
                authority = build_project_repository_identity_authority(
                    project_memory=project_memory, user_confirmed_links=confirmations["confirmations"],
                )
                if authority["conflicts"] or authority["status"] == "blocked":
                    result = _safe_error("repository_mapping_conflict"); result["status"] = "blocked"; return result
                confirmation_bytes = _artifact_bytes(confirmations)
                confirmation_changed = prior_confirmation != confirmation_bytes
                try:
                    if confirmation_changed:
                        _atomic_replace_bytes(confirmation_target, confirmation_bytes)
                    authority_write = write_project_repository_identity_authority(authority, authority_target)
                    if authority_write.get("status") == "error":
                        raise OSError("authority_write_failed")
                except Exception:
                    try:
                        _restore_path(confirmation_target, prior_confirmation)
                        _restore_path(authority_target, prior_authority)
                    except OSError:
                        pass
                    return _safe_error("repository_mapping_persistence_failed")
        except OSError:
            return _safe_error("repository_mapping_busy")
        except Exception:
            return _safe_error("repository_mapping_workflow_failed")
    persistence_status = "unchanged" if duplicate and not confirmation_changed else "updated" if prior_confirmation is not None else "created"
    try:
        readiness = readiness_inspector(
            project_memory=project_memory, vector_store=vector_store, vector_records=vector_records,
            raw_sources=raw_sources, chunks=chunks, raw_source_path=raw_source_path,
            chunk_path=chunk_path, manifest_path=manifest_path, identity_authority=authority,
        )
    except Exception:
        result = _safe_error("readiness_inspection_failed")
        result.update({"status": "degraded", "mapping": {"project_id": record["project_id"], "repository": record["repository"]},
                       "authority_status": authority["status"], "identity_mapping_count": authority["mapping_count"]})
        return result
    return {
        "status": persistence_status,
        "mapping": {"project_id": record["project_id"], "repository": record["repository"]},
        "authority_status": authority["status"], "identity_mapping_count": authority["mapping_count"],
        "records_resolved_by_authoritative_mapping": readiness["records_resolved_by_authoritative_mapping"],
        "records_unresolved": readiness["records_unresolved"],
        "identity_ready": bool(readiness["vector_ready"]),
        "materialization_required": not bool(readiness["chunk_mapping_ready"]),
        "ready_for_hybrid_retrieval": bool(readiness["ready_for_hybrid_retrieval"]),
        "warnings": list(readiness.get("warnings", []))[:MAX_WARNINGS],
        "errors": list(readiness.get("errors", []))[:MAX_ERRORS],
    }
