"""Deterministic authority for explicit project-to-GitHub repository links."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping, Sequence, TypedDict
from urllib.parse import urlsplit
from contextlib import contextmanager


IDENTITY_SCHEMA_VERSION = "project_repository_identity.v1"
DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "project_repository_identity.json"
)
DEFAULT_PROJECT_REPOSITORY_CONFIRMATIONS_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "project_repository_confirmations.json"
)
MAX_IDENTITY_CANDIDATES = 1000
MAX_CONFIRMED_MAPPINGS = 500
MAX_REPOSITORY_ALIASES = 16
MAX_SOURCE_REFS = 16
MAX_CONFLICTS = 64
MAX_UNRESOLVED_PROJECTS = 128
MAX_UNRESOLVED_REPOSITORIES = 128
MAX_WARNINGS = 32
MAX_ERRORS = 16
MAX_IDENTITY_INPUT_CHARS = 500

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:api[_-]?key|access[_-]?token|password|credential)\s*[:=])"
)
_REPOSITORY_KEYS = (
    "repository", "repo", "repository_name", "repository_url", "github_repository", "github_url",
)
_WRITE_LOCK = threading.RLock()


class RepositoryAuthorityMapping(TypedDict):
    repository_to_project: dict[str, str]
    alias_to_repository: dict[str, str]
    conflicts: list[str]
    unmapped_projects: list[str]
    mapping_count: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_project_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = "".join(char for char in value.strip() if ord(char) >= 32 and ord(char) != 127)
    return normalized if _PROJECT_ID_RE.fullmatch(normalized) else ""


def normalize_repository_identity(value: Any) -> str:
    """Return canonical ``owner/repository`` for explicit GitHub identities only."""

    if not isinstance(value, str) or not value or len(value) > MAX_IDENTITY_INPUT_CHARS:
        return ""
    text = "".join(char for char in value.strip() if ord(char) >= 32 and ord(char) != 127)
    if not text or _SECRET_RE.search(text) or ".." in text.split("?")[0].split("#")[0].split("/"):
        return ""
    if text.casefold().startswith("git@github.com:"):
        text = text[len("git@github.com:"):]
    elif text.casefold().startswith("ssh://"):
        parsed = urlsplit(text)
        if parsed.hostname is None or parsed.hostname.casefold() != "github.com":
            return ""
        if parsed.username not in (None, "git") or parsed.password is not None or parsed.port is not None:
            return ""
        text = parsed.path.lstrip("/")
    elif "://" in text:
        parsed = urlsplit(text)
        if parsed.scheme.casefold() not in {"http", "https"}:
            return ""
        if parsed.hostname is None or parsed.hostname.casefold() not in {"github.com", "www.github.com"}:
            return ""
        if parsed.username is not None or parsed.password is not None or parsed.port is not None:
            return ""
        text = parsed.path.lstrip("/")
    else:
        text = text.split("?", 1)[0].split("#", 1)[0]
        if text.casefold().startswith("github.com/"):
            text = text[len("github.com/"):]
    text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if text.casefold().endswith(".git"):
        text = text[:-4]
    parts = text.split("/")
    if len(parts) != 2 or not all(_COMPONENT_RE.fullmatch(part or "") for part in parts):
        return ""
    if any(part in {".", ".."} for part in parts):
        return ""
    return f"{parts[0].casefold()}/{parts[1].casefold()}"


def _normalize_alias(value: Any, canonical: str) -> str:
    normalized = normalize_repository_identity(value)
    if normalized:
        return normalized
    if not isinstance(value, str) or len(value) > MAX_IDENTITY_INPUT_CHARS:
        return ""
    alias = "".join(char for char in value.strip() if ord(char) >= 32 and ord(char) != 127).casefold()
    if not alias or "/" in alias or not _COMPONENT_RE.fullmatch(alias) or alias in {".", ".."}:
        return ""
    return alias if canonical else ""


def _project_records(project_memory: Any) -> list[Mapping[str, Any]]:
    values = project_memory.get("projects") if isinstance(project_memory, Mapping) else None
    return [item for item in values[:MAX_UNRESOLVED_PROJECTS] if isinstance(item, Mapping)] if isinstance(values, (list, tuple)) else []


def _known_projects(project_memory: Any) -> set[str]:
    return {project_id for item in _project_records(project_memory) if (project_id := normalize_project_id(item.get("project_id")))}


def _explicit_repositories(record: Mapping[str, Any]) -> list[str]:
    repositories: set[str] = set()
    containers = [record]
    if isinstance(record.get("identity"), Mapping):
        containers.append(record["identity"])
    for container in containers:
        for key in _REPOSITORY_KEYS:
            if repository := normalize_repository_identity(container.get(key)):
                repositories.add(repository)
    return sorted(repositories)


def _context_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, (list, tuple)):
        return [item for item in payload[:MAX_IDENTITY_CANDIDATES] if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    repositories = payload.get("repositories")
    if isinstance(repositories, Mapping):
        result = []
        for key in sorted(repositories, key=lambda value: str(value).casefold())[:MAX_IDENTITY_CANDIDATES]:
            item = repositories[key]
            if not isinstance(item, Mapping):
                continue
            context = item.get("context") if isinstance(item.get("context"), Mapping) else item
            merged = dict(context)
            merged.setdefault("repository", item.get("repository") or key)
            merged.setdefault("project_id", item.get("project_id"))
            result.append(merged)
        return result
    for key in ("contexts", "repo_contexts", "github_contexts"):
        if isinstance(payload.get(key), (list, tuple)):
            return [item for item in payload[key][:MAX_IDENTITY_CANDIDATES] if isinstance(item, Mapping)]
    return [payload]


def _candidate(project_id: str, repository: str, aliases: Sequence[str], source_type: str, source_ref: str) -> dict[str, Any]:
    safe_ref = source_ref[:200]
    core = {"project_id": project_id, "repository_identity": repository, "source_type": source_type, "source_ref": safe_ref}
    return {
        "candidate_id": f"prc_{_hash(core)[:24]}", **core,
        "repository_aliases": sorted(set(aliases))[:MAX_REPOSITORY_ALIASES],
        "source_hash": _hash(core), "explicit": True, "status": "candidate",
    }


def audit_explicit_project_repository_links(
    *, project_memory: Any = None, saved_github_context: Any = None,
    existing_identity_authority: Any = None, user_confirmed_links: Any = None,
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    rejected = 0
    for index, project in enumerate(_project_records(project_memory)):
        project_id = normalize_project_id(project.get("project_id"))
        for repository in _explicit_repositories(project) if project_id else []:
            aliases = project.get("repository_aliases")
            aliases = [_normalize_alias(value, repository) for value in aliases[:MAX_REPOSITORY_ALIASES]] if isinstance(aliases, (list, tuple)) else []
            item = _candidate(project_id, repository, [a for a in aliases if a], "project_memory", f"project:{project_id}")
            candidates[item["candidate_id"]] = item
    for index, context in enumerate(_context_records(saved_github_context)):
        project_id = normalize_project_id(context.get("project_id"))
        repositories = _explicit_repositories(context)
        if project_id and repositories:
            for repository in repositories:
                item = _candidate(project_id, repository, [], "saved_context", f"context:{_hash([project_id, repository])[:16]}")
                candidates[item["candidate_id"]] = item
        elif repositories:
            rejected += 1
    if _valid_authority(existing_identity_authority):
        for item in existing_identity_authority.get("mappings", [])[:MAX_CONFIRMED_MAPPINGS] if isinstance(existing_identity_authority.get("mappings"), list) else []:
            if not isinstance(item, Mapping) or item.get("status") != "confirmed":
                continue
            project_id = normalize_project_id(item.get("project_id")); repository = normalize_repository_identity(item.get("repository_identity"))
            if project_id and repository:
                candidate = _candidate(project_id, repository, [], "existing_authority", str(item.get("mapping_id") or "authority"))
                candidates[candidate["candidate_id"]] = candidate
    confirmations = user_confirmed_links if isinstance(user_confirmed_links, (list, tuple)) else []
    known = _known_projects(project_memory)
    for index, link in enumerate(confirmations[:MAX_IDENTITY_CANDIDATES]):
        if not isinstance(link, Mapping) or link.get("confirmed") is not True:
            rejected += 1; continue
        project_id = normalize_project_id(link.get("project_id")); repository = normalize_repository_identity(link.get("repository"))
        if not project_id or project_id not in known or not repository:
            rejected += 1; continue
        aliases_value = link.get("aliases")
        aliases = [_normalize_alias(value, repository) for value in aliases_value[:MAX_REPOSITORY_ALIASES]] if isinstance(aliases_value, (list, tuple)) else []
        item = _candidate(
            project_id, repository, [a for a in aliases if a], "user_confirmation",
            f"confirmation:{_hash([project_id, repository, sorted(a for a in aliases if a)])[:16]}",
        )
        candidates[item["candidate_id"]] = item
    limited = len(candidates) >= MAX_IDENTITY_CANDIDATES
    return {"candidates": sorted(candidates.values(), key=lambda item: item["candidate_id"])[:MAX_IDENTITY_CANDIDATES], "rejected_count": rejected, "limit_reached": limited}


def validate_user_confirmed_repository_links(*, project_memory: Any, links: Any) -> dict[str, Any]:
    audit = audit_explicit_project_repository_links(project_memory=project_memory, user_confirmed_links=links)
    values = [item for item in audit["candidates"] if item["source_type"] == "user_confirmation"]
    return {"accepted": values, "accepted_count": len(values), "rejected_count": audit["rejected_count"]}


def build_project_repository_identity_authority(
    *, project_memory: Any, saved_github_context: Any = None,
    existing_authority: Any = None, user_confirmed_links: Any = None,
) -> dict[str, Any]:
    audit = audit_explicit_project_repository_links(
        project_memory=project_memory, saved_github_context=saved_github_context,
        existing_identity_authority=existing_authority, user_confirmed_links=user_confirmed_links,
    )
    candidates = audit["candidates"]
    repo_projects: dict[str, set[str]] = {}; alias_targets: dict[str, set[tuple[str, str]]] = {}
    for item in candidates:
        repository = item["repository_identity"]; project_id = item["project_id"]
        repo_projects.setdefault(repository, set()).add(project_id)
        for alias in item["repository_aliases"]:
            alias_targets.setdefault(alias, set()).add((repository, project_id))
    conflict_repositories = {repo for repo, projects in repo_projects.items() if len(projects) != 1}
    conflict_aliases = {alias for alias, targets in alias_targets.items() if len(targets) != 1}
    conflicts = ([{"type": "repository_multiple_projects", "repository_identity": repo} for repo in sorted(conflict_repositories)] +
                 [{"type": "alias_multiple_targets", "repository_alias": alias} for alias in sorted(conflict_aliases)])[:MAX_CONFLICTS]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in candidates:
        key = (item["project_id"], item["repository_identity"])
        if key[1] not in conflict_repositories:
            grouped.setdefault(key, []).append(item)
    mappings = []
    for (project_id, repository), items in sorted(grouped.items()):
        aliases = sorted({alias for item in items for alias in item["repository_aliases"] if alias not in conflict_aliases})[:MAX_REPOSITORY_ALIASES]
        core = {"project_id": project_id, "repository_identity": repository, "repository_aliases": aliases,
                "source_types": sorted({item["source_type"] for item in items}),
                "source_refs": sorted({item["source_ref"] for item in items})[:MAX_SOURCE_REFS], "status": "confirmed"}
        mappings.append({"mapping_id": f"prm_{_hash(core)[:24]}", **core, "content_hash": _hash(core)})
    mappings = mappings[:MAX_CONFIRMED_MAPPINGS]
    known = _known_projects(project_memory); mapped_projects = {item["project_id"] for item in mappings}
    unresolved_projects = sorted(known - mapped_projects)[:MAX_UNRESOLVED_PROJECTS]
    unresolved_repositories = sorted(conflict_repositories)[:MAX_UNRESOLVED_REPOSITORIES]
    warnings = []
    if audit["limit_reached"] or len(grouped) > MAX_CONFIRMED_MAPPINGS:
        warnings.append("identity_limit_reached")
    if conflicts:
        status = "blocked"
    elif warnings:
        status = "partial"
    elif mappings and unresolved_projects:
        status = "partial"
    elif mappings:
        status = "ready"
    elif known:
        status = "blocked"
    else:
        status = "empty"
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION, "status": status,
        "mapping_count": len(mappings), "project_count": len(mapped_projects),
        "repository_count": len(mappings), "mappings": mappings, "conflicts": conflicts,
        "unresolved_projects": unresolved_projects, "unresolved_repositories": unresolved_repositories,
        "warnings": warnings[:MAX_WARNINGS], "errors": [],
    }
    payload["content_hash"] = _hash(payload)
    return payload


def authority_to_repository_mapping(authority: Any) -> RepositoryAuthorityMapping:
    if not _valid_authority(authority):
        return {"repository_to_project": {}, "alias_to_repository": {}, "conflicts": [], "unmapped_projects": [], "mapping_count": 0}
    mapping: dict[str, str] = {}; aliases: dict[str, str] = {}
    conflicts = []
    for conflict in authority.get("conflicts", [])[:MAX_CONFLICTS]:
        if isinstance(conflict, Mapping):
            value = conflict.get("repository_identity") or conflict.get("repository_alias")
            if isinstance(value, str): conflicts.append(value)
    for item in authority.get("mappings", [])[:MAX_CONFIRMED_MAPPINGS]:
        repository = normalize_repository_identity(item.get("repository_identity")); project_id = normalize_project_id(item.get("project_id"))
        if repository and project_id and item.get("status") == "confirmed":
            mapping[repository] = project_id
            for alias in item.get("repository_aliases", [])[:MAX_REPOSITORY_ALIASES] if isinstance(item.get("repository_aliases"), list) else []:
                normalized = _normalize_alias(alias, repository)
                if normalized: aliases[normalized] = repository
    return {"repository_to_project": dict(sorted(mapping.items())), "alias_to_repository": dict(sorted(aliases.items())), "conflicts": sorted(set(conflicts)), "unmapped_projects": list(authority.get("unresolved_projects", []))[:MAX_UNRESOLVED_PROJECTS], "mapping_count": len(mapping)}


def build_authoritative_repository_project_mapping(project_memory: Any) -> RepositoryAuthorityMapping:
    result = authority_to_repository_mapping(build_project_repository_identity_authority(project_memory=project_memory))
    result["unmapped_projects"] = sorted(
        project_id for item in _project_records(project_memory)
        if (project_id := normalize_project_id(item.get("project_id"))) and not _explicit_repositories(item)
    )[:MAX_UNRESOLVED_PROJECTS]
    return result


def safe_repository_identity_report(authority: Any) -> dict[str, Any]:
    valid = authority if _valid_authority(authority) else {}
    unresolved_projects = list(valid.get("unresolved_projects", []))[:MAX_UNRESOLVED_PROJECTS]
    unresolved_repositories = list(valid.get("unresolved_repositories", []))[:MAX_UNRESOLVED_REPOSITORIES]
    conflicts = list(valid.get("conflicts", []))[:MAX_CONFLICTS]
    return {"unresolved_project_ids": unresolved_projects, "unresolved_repository_identities": unresolved_repositories,
            "conflict_count": len(conflicts), "unresolved_count": len(unresolved_projects) + len(unresolved_repositories),
            "mapping_count": int(valid.get("mapping_count", 0)),
            "requires_user_confirmation": bool(conflicts or unresolved_projects or unresolved_repositories or not valid.get("mapping_count"))}


def _valid_authority(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        return False
    mappings = value.get("mappings"); conflicts = value.get("conflicts")
    if not isinstance(mappings, list) or len(mappings) > MAX_CONFIRMED_MAPPINGS or not isinstance(conflicts, list) or len(conflicts) > MAX_CONFLICTS:
        return False
    expected = value.get("content_hash")
    unsigned = dict(value); unsigned.pop("content_hash", None)
    return isinstance(expected, str) and expected == _hash(unsigned)


def load_project_repository_identity_authority(path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH) -> dict[str, Any] | None:
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size > 2_000_000:
            return None
        value = json.loads(candidate.read_text(encoding="utf-8"))
        return dict(value) if _valid_authority(value) else None
    except (OSError, ValueError, TypeError):
        return None


@contextmanager
def _exclusive_artifact_lock(target: Path):
    lock_path = target.with_name(f".{target.name}.lock")
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


def write_project_repository_identity_authority(authority: Any, path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH) -> dict[str, Any]:
    if not _valid_authority(authority):
        return {"status": "error", "error": "invalid_identity_authority"}
    target = Path(path)
    if not target.parent.exists():
        return {"status": "error", "error": "parent_directory_missing"}
    encoded = (_canonical_json(authority) + "\n").encode("utf-8")
    with _WRITE_LOCK:
        try:
            with _exclusive_artifact_lock(target):
                existed = target.exists()
                if existed and target.read_bytes() == encoded:
                    return {"status": "unchanged", "content_hash": authority["content_hash"]}
                handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
                return {"status": "updated" if existed else "created", "content_hash": authority["content_hash"]}
        except OSError:
            return {"status": "error", "error": "identity_authority_write_failed"}
