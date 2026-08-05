"""Explicit default-off materialization of saved GitHub context into evidence chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Mapping, TypedDict

from backend.evidence_index_readiness import (
    AuthoritativeRepositoryMapping,
    build_authoritative_repository_project_mapping,
    normalize_project_id,
    normalize_repository_identity,
)
from backend.github_evidence_chunks import (
    DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    build_github_evidence_chunks_from_raw_source,
)
from backend.github_raw_storage import (
    DEFAULT_GITHUB_RAW_SOURCES_PATH,
    build_github_raw_source_record,
)


GITHUB_EVIDENCE_MATERIALIZATION_FLAG = "USE_GITHUB_EVIDENCE_MATERIALIZATION"
DEFAULT_SAVED_GITHUB_CONTEXT_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "github_repo_scan_state.json"
)
DEFAULT_PROJECT_MEMORY_PATH = Path(__file__).resolve().parents[1] / "information" / "project_memory.json"
DEFAULT_MATERIALIZATION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "github_evidence_materialization.json"
)
MATERIALIZATION_SCHEMA_VERSION = "github_evidence_materialization.v1"
MAX_REPOSITORIES = 100
MAX_SAVED_CONTEXT_RECORDS = 1000
MAX_RAW_SOURCES_TOTAL = 1000
MAX_RAW_CHARS_PER_SOURCE = 200_000
MAX_RAW_CHARS_TOTAL = 2_000_000
MAX_CHUNKS_TOTAL = 5000
MAX_WARNINGS = 32
MAX_ERRORS = 16
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_WRITE_LOCK = threading.RLock()


class GithubEvidenceMaterializationResult(TypedDict):
    status: str
    source_records_seen: int
    source_records_accepted: int
    source_records_skipped: int
    raw_source_count: int
    chunk_count: int
    project_ids: list[str]
    unresolved_identity_count: int
    conflict_count: int
    content_hash: str
    raw_artifact_hash: str
    chunk_artifact_hash: str
    warnings: list[str]
    errors: list[str]


def is_github_evidence_materialization_enabled() -> bool:
    try:
        value = os.getenv(GITHUB_EVIDENCE_MATERIALIZATION_FLAG, "")
    except Exception:
        return False
    return isinstance(value, str) and value.strip().casefold() in _ENABLED_VALUES


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(record)}\n" for record in records).encode("utf-8")


def _context_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload[:MAX_REPOSITORIES] if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    repositories = payload.get("repositories")
    if isinstance(repositories, Mapping):
        entries = []
        for repo_key in sorted(repositories, key=lambda item: str(item).casefold())[:MAX_REPOSITORIES]:
            entry = repositories[repo_key]
            if not isinstance(entry, Mapping):
                continue
            context = entry.get("context") if isinstance(entry.get("context"), Mapping) else entry
            normalized = dict(context)
            normalized.setdefault("repository", entry.get("repository") or str(repo_key))
            normalized.setdefault("latest_commit_sha", entry.get("latest_commit_sha") or "")
            entries.append(normalized)
        return entries
    for key in ("contexts", "repo_contexts", "github_contexts"):
        values = payload.get(key)
        if isinstance(values, (list, tuple)):
            return [dict(item) for item in values[:MAX_REPOSITORIES] if isinstance(item, Mapping)]
    return [dict(payload)] if payload.get("repository") or payload.get("repo") else []


def list_saved_github_context_repositories(saved_context: Any) -> list[str]:
    """Return bounded canonical repository identities without reading content bodies."""

    repositories = {
        repository
        for context in _context_entries(saved_context)
        if (repository := normalize_repository_identity(
            context.get("repository") or context.get("repo") or context.get("url")
        ))
    }
    return sorted(repositories)[:MAX_REPOSITORIES]


def _context_project_id(
    context: Mapping[str, Any], repository: str, mapping: AuthoritativeRepositoryMapping,
) -> tuple[str, str]:
    explicit_values = {
        project_id.casefold(): project_id
        for key in ("project_id", "project_name")
        if (project_id := normalize_project_id(context.get(key)))
    }
    if len(explicit_values) > 1 or repository in mapping["conflicts"]:
        return "", "identity_conflict"
    explicit = next(iter(explicit_values.values()), "")
    mapped = mapping["repository_to_project"].get(repository)
    if explicit and mapped and explicit.casefold() != mapped.casefold():
        return "", "identity_conflict"
    if explicit:
        return explicit, ""
    if mapped:
        return mapped, ""
    return "", "unresolved_identity"


def _source_inputs(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    repository = normalize_repository_identity(
        context.get("repository") or context.get("repo") or context.get("url")
    )
    latest_sha = str(context.get("latest_commit_sha") or "").strip()
    inputs: list[dict[str, Any]] = []
    readme = context.get("readme")
    if isinstance(readme, str) and readme:
        inputs.append({
            "repo": repository, "source_type": "readme", "path": "README.md",
            "commit_sha": latest_sha, "raw_text": readme, "metadata": {},
        })
    evidence_values = context.get("contribution_evidence")
    if isinstance(evidence_values, (list, tuple)):
        for evidence in evidence_values[:MAX_SAVED_CONTEXT_RECORDS]:
            if not isinstance(evidence, Mapping):
                continue
            commits = evidence.get("commits")
            if not isinstance(commits, (list, tuple)):
                commits = [evidence] if evidence.get("file_changes") else []
            for commit in commits[:MAX_SAVED_CONTEXT_RECORDS]:
                if not isinstance(commit, Mapping):
                    continue
                commit_sha = str(commit.get("sha") or latest_sha).strip()
                message = str(commit.get("message") or "")[:300]
                changes = commit.get("file_changes")
                if not isinstance(changes, (list, tuple)):
                    continue
                for change in changes[:MAX_SAVED_CONTEXT_RECORDS]:
                    if not isinstance(change, Mapping):
                        continue
                    patch = change.get("patch")
                    path = change.get("filename") or change.get("path")
                    if isinstance(patch, str) and patch and isinstance(path, str):
                        inputs.append({
                            "repo": repository, "source_type": "commit_patch", "path": path,
                            "commit_sha": commit_sha, "raw_text": patch,
                            "metadata": {"commit_message": message},
                        })
    for collection_key, source_type in (("issues", "issue"), ("pull_requests", "pr"), ("logs", "log")):
        values = context.get(collection_key)
        if not isinstance(values, (list, tuple)):
            continue
        for item in values[:MAX_SAVED_CONTEXT_RECORDS]:
            if not isinstance(item, Mapping):
                continue
            body = item.get("body") or item.get("text") or item.get("content")
            if isinstance(body, str) and body:
                inputs.append({
                    "repo": repository, "source_type": source_type, "path": "",
                    "commit_sha": "", "raw_text": body, "metadata": {},
                })
    return inputs[:MAX_SAVED_CONTEXT_RECORDS]


def adapt_saved_github_contexts_to_raw_sources(
    *,
    saved_context: Any,
    authoritative_mapping: AuthoritativeRepositoryMapping,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository_count = (
        len(saved_context.get("repositories", {}))
        if isinstance(saved_context, Mapping) and isinstance(saved_context.get("repositories"), Mapping)
        else len(saved_context) if isinstance(saved_context, (list, tuple)) else 1
    )
    entries = _context_entries(saved_context)
    raw_by_id: dict[str, dict[str, Any]] = {}
    seen = accepted = skipped = unresolved = conflicts = total_chars = 0
    warnings: list[str] = []
    limit_reached = repository_count > MAX_REPOSITORIES
    if limit_reached:
        warnings.append("repository_count_limit")
    for context in entries:
        repository = normalize_repository_identity(
            context.get("repository") or context.get("repo") or context.get("url")
        )
        if not repository:
            skipped += 1
            unresolved += 1
            continue
        project_id, reason = _context_project_id(context, repository, authoritative_mapping)
        inputs = _source_inputs(context)
        if not project_id:
            skipped += len(inputs) or 1
            unresolved += int(reason == "unresolved_identity")
            conflicts += int(reason == "identity_conflict")
            continue
        for source in inputs:
            seen += 1
            raw_text = source.get("raw_text")
            if not isinstance(raw_text, str) or not raw_text:
                skipped += 1
                continue
            if len(raw_text) > MAX_RAW_CHARS_PER_SOURCE:
                skipped += 1
                warnings.append("raw_source_char_limit")
                limit_reached = True
                continue
            if total_chars + len(raw_text) > MAX_RAW_CHARS_TOTAL or len(raw_by_id) >= MAX_RAW_SOURCES_TOTAL:
                skipped += 1
                warnings.append("materialization_total_limit")
                limit_reached = True
                continue
            record = build_github_raw_source_record(
                project_id=project_id, repo=source["repo"], source_type=source["source_type"],
                path=source["path"], commit_sha=source["commit_sha"], raw_text=raw_text,
                created_at="", metadata=source["metadata"],
            )
            if record["source_id"] not in raw_by_id:
                raw_by_id[record["source_id"]] = record
                total_chars += len(raw_text)
                accepted += 1
    records = [raw_by_id[key] for key in sorted(raw_by_id)]
    return records, {
        "source_records_seen": seen,
        "source_records_accepted": accepted,
        "source_records_skipped": skipped,
        "unresolved_identity_count": unresolved,
        "conflict_count": conflicts,
        "warnings": sorted(set(warnings))[:MAX_WARNINGS],
        "limit_reached": limit_reached,
    }


def _build_chunks(raw_sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    chunks_by_id: dict[str, dict[str, Any]] = {}
    limit_reached = False
    for raw in raw_sources:
        for chunk in build_github_evidence_chunks_from_raw_source(raw):
            if len(chunks_by_id) >= MAX_CHUNKS_TOTAL:
                limit_reached = True
                break
            chunks_by_id.setdefault(chunk["chunk_id"], chunk)
        if limit_reached:
            break
    return [chunks_by_id[key] for key in sorted(chunks_by_id)], limit_reached


def inspect_saved_github_context_content_hash(
    *, saved_context: Any, authoritative_mapping: AuthoritativeRepositoryMapping,
) -> tuple[str, str]:
    """Compute the materializer's deterministic source hash without writing artifacts."""

    try:
        raw_sources, stats = adapt_saved_github_contexts_to_raw_sources(
            saved_context=saved_context, authoritative_mapping=authoritative_mapping,
        )
        if not raw_sources:
            status = "blocked" if stats["unresolved_identity_count"] or stats["conflict_count"] else "empty"
            return "", status
        source_hash = hashlib.sha256(_canonical_json(raw_sources).encode("utf-8")).hexdigest()
        return source_hash, "partial" if stats.get("limit_reached") else "ready"
    except Exception:
        return "", "error"


def _write_temp(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temp = _write_temp(path, previous)
    os.replace(temp, path)


def _persist_pair(
    payloads: list[tuple[Path, bytes]], *, replace: Callable[[Any, Any], Any] = os.replace,
) -> bool:
    previous = {path: path.read_bytes() if path.exists() else None for path, _ in payloads}
    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in payloads:
            staged.append((_write_temp(path, payload), path))
        for temp, path in staged:
            replace(temp, path)
        return True
    except Exception:
        for path, _payload in payloads:
            _restore(path, previous[path])
        return False
    finally:
        for temp, _path in staged:
            temp.unlink(missing_ok=True)


def _result(status: str, *, stats: Mapping[str, Any] | None = None, **values: Any) -> GithubEvidenceMaterializationResult:
    stats = stats or {}
    return {
        "status": status,
        "source_records_seen": int(stats.get("source_records_seen", 0)),
        "source_records_accepted": int(stats.get("source_records_accepted", 0)),
        "source_records_skipped": int(stats.get("source_records_skipped", 0)),
        "raw_source_count": int(values.get("raw_source_count", 0)),
        "chunk_count": int(values.get("chunk_count", 0)),
        "project_ids": list(values.get("project_ids", [])),
        "unresolved_identity_count": int(stats.get("unresolved_identity_count", 0)),
        "conflict_count": int(stats.get("conflict_count", 0)),
        "content_hash": str(values.get("content_hash", "")),
        "raw_artifact_hash": str(values.get("raw_artifact_hash", "")),
        "chunk_artifact_hash": str(values.get("chunk_artifact_hash", "")),
        "warnings": list(stats.get("warnings", []))[:MAX_WARNINGS],
        "errors": list(values.get("errors", []))[:MAX_ERRORS],
    }


def materialize_saved_github_evidence(
    *,
    saved_context: Any = None,
    project_memory: Any = None,
    saved_context_path: str | Path = DEFAULT_SAVED_GITHUB_CONTEXT_PATH,
    project_memory_path: str | Path = DEFAULT_PROJECT_MEMORY_PATH,
    raw_source_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
    chunk_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    manifest_path: str | Path = DEFAULT_MATERIALIZATION_MANIFEST_PATH,
    authoritative_mapping: AuthoritativeRepositoryMapping | None = None,
    feature_enabled: bool | None = None,
    replace: Callable[[Any, Any], Any] = os.replace,
) -> GithubEvidenceMaterializationResult:
    """Explicitly build a stable raw/chunk/manifest trio; disabled execution performs no I/O."""

    if type(feature_enabled) is bool:
        enabled = feature_enabled
    elif feature_enabled is not None:
        enabled = False
    else:
        enabled = is_github_evidence_materialization_enabled()
    if not enabled:
        return _result("disabled")
    limits = (
        MAX_REPOSITORIES, MAX_SAVED_CONTEXT_RECORDS, MAX_RAW_SOURCES_TOTAL,
        MAX_RAW_CHARS_PER_SOURCE, MAX_RAW_CHARS_TOTAL, MAX_CHUNKS_TOTAL,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
        return _result("error", errors=["invalid_materialization_limits"])
    try:
        if saved_context is None:
            saved_context = json.loads(Path(saved_context_path).read_text(encoding="utf-8"))
        if project_memory is None:
            project_memory = json.loads(Path(project_memory_path).read_text(encoding="utf-8"))
        mapping = authoritative_mapping
        if mapping is None:
            mapping = build_authoritative_repository_project_mapping(project_memory)
        if not isinstance(mapping, Mapping):
            return _result("error", errors=["invalid_authoritative_mapping"])
        raw_sources, stats = adapt_saved_github_contexts_to_raw_sources(
            saved_context=saved_context, authoritative_mapping=mapping
        )
        if not raw_sources:
            status = "blocked" if stats["unresolved_identity_count"] or stats["conflict_count"] else "empty"
            return _result(status, stats=stats)
        chunk_values, chunk_limit = _build_chunks(raw_sources)
        if not chunk_values:
            return _result("blocked", stats=stats, raw_source_count=len(raw_sources), errors=["no_chunks_built"])
        if chunk_limit:
            stats["warnings"] = sorted(set([*stats["warnings"], "chunk_count_limit"]))[:MAX_WARNINGS]
            stats["limit_reached"] = True
        raw_bytes = _jsonl_bytes(raw_sources)
        chunk_bytes = _jsonl_bytes(chunk_values)
        source_hash = hashlib.sha256(_canonical_json(raw_sources).encode("utf-8")).hexdigest()
        raw_hash = _hash_bytes(raw_bytes)
        chunk_hash = _hash_bytes(chunk_bytes)
        materialization_status = "partial" if stats.get("limit_reached") else "ready"
        project_ids = sorted({record["project_id"] for record in raw_sources})
        manifest = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "source_content_hash": source_hash,
            "raw_source_count": len(raw_sources),
            "chunk_count": len(chunk_values),
            "project_ids": project_ids,
            "raw_artifact_hash": raw_hash,
            "chunk_artifact_hash": chunk_hash,
            "status": materialization_status,
        }
        manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")
        raw_path = Path(raw_source_path)
        chunks_path = Path(chunk_path)
        manifest_target = Path(manifest_path)
        payloads = [(raw_path, raw_bytes), (chunks_path, chunk_bytes), (manifest_target, manifest_bytes)]
        unchanged = all(path.exists() and path.read_bytes() == payload for path, payload in payloads)
        existed = any(path.exists() for path, _payload in payloads)
        if not unchanged:
            with _WRITE_LOCK:
                if not _persist_pair(payloads, replace=replace):
                    return _result("error", stats=stats, errors=["atomic_persistence_failed"])
        status = "unchanged" if unchanged else ("updated" if existed else "created")
        if materialization_status == "partial":
            status = "partial"
        return _result(
            status, stats=stats, raw_source_count=len(raw_sources), chunk_count=len(chunk_values),
            project_ids=project_ids, content_hash=source_hash,
            raw_artifact_hash=raw_hash, chunk_artifact_hash=chunk_hash,
        )
    except Exception:
        return _result("error", errors=["materialization_failed"])
