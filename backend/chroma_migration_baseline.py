"""Capture and verify a non-mutating baseline for protected local Chroma storage."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, NamedTuple

from backend.chroma_baseline_models import (
    CHROMA_MIGRATION_BASELINE_SCHEMA,
    MAX_BASELINE_BYTES,
    MAX_BASELINE_CALL_SITES,
    MAX_BASELINE_FILES,
    PROTECTED_CAPTURE_MODE,
    BaselineValidationError,
    add_baseline_content_hash,
    canonical_json,
    sha256_json,
    validate_chroma_migration_baseline,
    validate_relative_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTECTED_CHROMA_ROOT = PROJECT_ROOT / "information" / "chroma"
DEFAULT_BASELINE_OUTPUT = (
    PROJECT_ROOT / "information" / "chroma_migration_baselines" / "baseline.json"
)
DEFAULT_IDENTITY_AUTHORITY_PATH = PROJECT_ROOT / "information" / "project_repository_identity.json"
DEFAULT_ARTIFACT_SPECS = (
    ("github_raw_sources", "information/github_raw_sources.jsonl"),
    ("github_evidence_chunks", "information/github_evidence_chunks.jsonl"),
    ("github_evidence_materialization", "information/github_evidence_materialization.json"),
    ("project_repository_confirmations", "information/project_repository_confirmations.json"),
    ("project_repository_identity", "information/project_repository_identity.json"),
    ("github_repo_scan_state", "information/github_repo_scan_state.json"),
    ("project_memory", "information/project_memory.json"),
    ("project_evidence_memory", "information/project_evidence_memory.json"),
)
LOGICAL_INVENTORY_UNAVAILABLE_REASON = (
    "Logical collection inventory was not queried because protected embedded Chroma access may mutate database internals."
)
MAX_STORAGE_BYTES = 1_000_000_000_000
MAX_SOURCE_FILES = 20_000
MAX_SOURCE_BYTES = 2_000_000
SCHEMA_SCAN_BYTES = 65_536
_HASH_BLOCK_BYTES = 1024 * 1024
_SCHEMA_MARKER_RE = re.compile(
    rb'"(?:schema|schema_version)"\s*:\s*"([A-Za-z0-9_.:-]{1,128})"'
)
_SAFE_SEMANTIC_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_EXCLUDED_SOURCE_DIRECTORIES = frozenset(
    {
        ".git",
        ".agents",
        ".codex",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "information",
        "logs",
        "outputs",
        "dist",
        "build",
        "venv",
        ".venv",
    }
)
_APPROVED_PRODUCTION_CLIENT_CALLS: frozenset[tuple[str, str, str]] = frozenset()
_APPROVED_MAINTENANCE_CLIENT_CALLS: frozenset[tuple[str, str, str]] = frozenset()


class BaselineCaptureError(RuntimeError):
    """A stable capture failure that never includes protected absolute paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ArtifactSpec(NamedTuple):
    semantic_name: str
    relative_path: str


class _ClientCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.calls: list[tuple[str, int, str]] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        client_type = {"PersistentClient": "persistent", "HttpClient": "http"}.get(call_name)
        if client_type:
            self.calls.append((client_type, int(node.lineno), ".".join(self.stack) or "<module>"))
        self.generic_visit(node)


def _capture_error(code: str) -> None:
    raise BaselineCaptureError(code)


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if callable(is_junction) and is_junction(path):
            return True
        file_attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(file_attributes & reparse_attribute)
    except OSError:
        _capture_error("path_inspection_failed")


def _require_regular_root(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.is_dir() or _is_reparse_point(candidate):
        _capture_error("protected_storage_unavailable")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _capture_error("protected_storage_unavailable")
    if not resolved.is_dir():
        _capture_error("protected_storage_unavailable")
    return candidate, resolved


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        _capture_error("path_escape_detected")
    try:
        return validate_relative_path(relative)
    except BaselineValidationError:
        _capture_error("unsafe_relative_path")


def _hash_regular_file(
    path: Path,
    *,
    relative_path: str,
    opener: Callable[..., BinaryIO] = open,
) -> tuple[int, str, bytes]:
    if _is_reparse_point(path):
        _capture_error("reparse_point_rejected")
    try:
        before = path.stat(follow_symlinks=False)
    except OSError:
        _capture_error("file_stat_failed")
    if not stat.S_ISREG(before.st_mode):
        _capture_error("non_regular_file_rejected")
    digest = hashlib.sha256()
    prefix = bytearray()
    bytes_read = 0
    try:
        with opener(path, "rb") as stream:
            while True:
                block = stream.read(_HASH_BLOCK_BYTES)
                if not block:
                    break
                if not isinstance(block, bytes):
                    _capture_error("invalid_file_reader")
                bytes_read += len(block)
                digest.update(block)
                if len(prefix) < SCHEMA_SCAN_BYTES:
                    prefix.extend(block[: SCHEMA_SCAN_BYTES - len(prefix)])
    except BaselineCaptureError:
        raise
    except (OSError, PermissionError):
        _capture_error("unreadable_file")
    try:
        after = path.stat(follow_symlinks=False)
    except OSError:
        _capture_error("file_stat_failed")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or bytes_read != before.st_size
    ):
        _capture_error("file_changed_while_reading")
    if relative_path != validate_relative_path(relative_path):
        _capture_error("unsafe_relative_path")
    return before.st_size, digest.hexdigest(), bytes(prefix)


def capture_protected_file_inventory(
    root: str | Path,
    *,
    opener: Callable[..., BinaryIO] = open,
) -> dict[str, Any]:
    """Hash a directory tree using ordinary reads without importing or initializing Chroma."""

    root_path, root_resolved = _require_regular_root(root)
    pending = [root_path]
    files: list[dict[str, Any]] = []
    total_bytes = 0
    while pending:
        directory = pending.pop()
        if _is_reparse_point(directory):
            _capture_error("reparse_point_rejected")
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            _capture_error("directory_read_failed")
        child_directories: list[Path] = []
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink() or _is_reparse_point(candidate):
                _capture_error("reparse_point_rejected")
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                _capture_error("path_escape_detected")
            relative = _relative_posix(resolved, root_resolved)
            if entry.is_dir(follow_symlinks=False):
                child_directories.append(candidate)
                continue
            if not entry.is_file(follow_symlinks=False):
                _capture_error("non_regular_file_rejected")
            size, digest, _ = _hash_regular_file(
                candidate, relative_path=relative, opener=opener
            )
            files.append({"relative_path": relative, "size_bytes": size, "sha256": digest})
            total_bytes += size
            if len(files) > MAX_BASELINE_FILES or total_bytes > MAX_STORAGE_BYTES:
                _capture_error("protected_storage_limit_exceeded")
        pending.extend(reversed(child_directories))
    files.sort(key=lambda item: item["relative_path"])
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "aggregate_sha256": sha256_json(files),
    }


def _schema_marker(prefix: bytes) -> str | None:
    match = _SCHEMA_MARKER_RE.search(prefix)
    return match.group(1).decode("ascii") if match else None


def _normalize_artifact_specs(
    artifact_specs: Sequence[ArtifactSpec | tuple[str, str]],
) -> list[ArtifactSpec]:
    normalized: list[ArtifactSpec] = []
    for raw in artifact_specs:
        try:
            spec = raw if isinstance(raw, ArtifactSpec) else ArtifactSpec(*raw)
        except (TypeError, ValueError):
            _capture_error("invalid_artifact_spec")
        if not _SAFE_SEMANTIC_NAME_RE.fullmatch(spec.semantic_name):
            _capture_error("invalid_artifact_name")
        try:
            relative = validate_relative_path(spec.relative_path)
        except BaselineValidationError:
            _capture_error("invalid_artifact_path")
        normalized.append(ArtifactSpec(spec.semantic_name, relative))
    normalized.sort(key=lambda item: item.semantic_name)
    if len({item.semantic_name for item in normalized}) != len(normalized):
        _capture_error("duplicate_artifact_name")
    return normalized


def capture_evidence_artifact_hashes(
    repository_root: str | Path,
    *,
    artifact_specs: Sequence[ArtifactSpec | tuple[str, str]] = DEFAULT_ARTIFACT_SPECS,
    opener: Callable[..., BinaryIO] = open,
) -> dict[str, Any]:
    root_path, root_resolved = _require_regular_root(repository_root)
    artifacts = []
    for spec in _normalize_artifact_specs(artifact_specs):
        candidate = root_path.joinpath(*PurePosixPath(spec.relative_path).parts)
        if _is_reparse_point(candidate):
            _capture_error("artifact_reparse_point_rejected")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            _capture_error("artifact_path_escape")
        size, digest, prefix = _hash_regular_file(
            candidate, relative_path=spec.relative_path, opener=opener
        )
        artifacts.append(
            {
                "semantic_name": spec.semantic_name,
                "relative_path": spec.relative_path,
                "size_bytes": size,
                "sha256": digest,
                "schema_marker": _schema_marker(prefix),
            }
        )
    return {"artifacts": artifacts, "aggregate_sha256": sha256_json(artifacts)}


def _iter_source_files(repository_root: Path) -> Iterable[tuple[Path, str]]:
    _, root_resolved = _require_regular_root(repository_root)
    count = 0
    for directory, directory_names, file_names in os.walk(repository_root, followlinks=False):
        safe_directories = []
        for name in sorted(directory_names):
            if name in _EXCLUDED_SOURCE_DIRECTORIES:
                continue
            if _is_reparse_point(Path(directory) / name):
                _capture_error("source_reparse_point_rejected")
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            if not name.endswith(".py"):
                continue
            candidate = Path(directory) / name
            if _is_reparse_point(candidate):
                _capture_error("source_reparse_point_rejected")
            try:
                resolved = candidate.resolve(strict=True)
                relative = _relative_posix(resolved, root_resolved)
            except OSError:
                _capture_error("source_path_unavailable")
            count += 1
            if count > MAX_SOURCE_FILES:
                _capture_error("source_file_limit_exceeded")
            yield candidate, relative


def _classify_client_call(client_type: str, relative_path: str, callable_name: str) -> str:
    key = (client_type, relative_path, callable_name)
    if relative_path.startswith("tests/"):
        return "test_only"
    if key in _APPROVED_PRODUCTION_CLIENT_CALLS:
        return "production"
    if key in _APPROVED_MAINTENANCE_CLIENT_CALLS:
        return "approved_maintenance"
    return "unknown"


def capture_chroma_client_call_inventory(repository_root: str | Path) -> dict[str, Any]:
    """Statically classify Chroma client constructors without importing application modules."""

    root = Path(repository_root)
    reviewed_manifest = root / "backend" / "chroma_access_manifest.py"
    if reviewed_manifest.is_file():
        try:
            from backend.chroma_access_inventory import (
                ChromaAccessInventoryError,
                baseline_constructor_summary,
            )

            return baseline_constructor_summary(root, reviewed_manifest)
        except ChromaAccessInventoryError as error:
            _capture_error(error.code)
    call_sites: list[dict[str, Any]] = []
    for source_path, relative_path in _iter_source_files(root):
        try:
            size = source_path.stat().st_size
            if size > MAX_SOURCE_BYTES:
                _capture_error("source_file_size_limit_exceeded")
            source = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            _capture_error("source_file_unreadable")
        if "PersistentClient" not in source and "HttpClient" not in source:
            continue
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError:
            _capture_error("source_parse_failed")
        visitor = _ClientCallVisitor()
        visitor.visit(tree)
        for client_type, line, callable_name in visitor.calls:
            call_sites.append(
                {
                    "client_type": client_type,
                    "classification": _classify_client_call(
                        client_type, relative_path, callable_name
                    ),
                    "relative_path": relative_path,
                    "line": line,
                    "callable": callable_name,
                }
            )
    call_sites.sort(
        key=lambda item: (item["relative_path"], item["line"], item["client_type"])
    )
    if len(call_sites) > MAX_BASELINE_CALL_SITES:
        _capture_error("client_call_limit_exceeded")
    state = {
        "production_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "production"
            for item in call_sites
        ),
        "approved_maintenance_persistent_client_call_count": sum(
            item["client_type"] == "persistent"
            and item["classification"] == "approved_maintenance"
            for item in call_sites
        ),
        "test_only_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "test_only"
            for item in call_sites
        ),
        "http_client_call_count": sum(item["client_type"] == "http" for item in call_sites),
        "unknown_unclassified_call_count": sum(
            item["classification"] == "unknown" for item in call_sites
        ),
        "call_sites": call_sites,
        "aggregate_sha256": sha256_json(call_sites),
    }
    if state["unknown_unclassified_call_count"]:
        _capture_error("unclassified_chroma_call_site")
    return state


def unavailable_logical_inventory() -> dict[str, Any]:
    return {
        "source": "unavailable",
        "collections": [],
        "limitations": [LOGICAL_INVENTORY_UNAVAILABLE_REASON],
    }


def _safe_logical_inventory(value: Any) -> tuple[dict[str, Any], bool, str | None]:
    if not isinstance(value, Mapping):
        _capture_error("invalid_logical_provider_result")
    source = value.get("source")
    if source not in {"existing_safe_artifact", "approved_http", "unavailable"}:
        _capture_error("invalid_logical_provider_source")
    raw_collections = value.get("collections")
    raw_limitations = value.get("limitations")
    if not isinstance(raw_collections, list) or not isinstance(raw_limitations, list):
        _capture_error("invalid_logical_provider_result")
    collections = []
    for item in raw_collections:
        if not isinstance(item, Mapping):
            _capture_error("invalid_logical_collection")
        collections.append(
            {
                "semantic_name": item.get("semantic_name"),
                "record_count": item.get("record_count"),
                "record_ids_sha256": item.get("record_ids_sha256"),
                "logical_fingerprint": item.get("logical_fingerprint"),
                "repository_count": item.get("repository_count"),
                "schema_marker": item.get("schema_marker"),
            }
        )
    collections.sort(key=lambda item: str(item.get("semantic_name") or ""))
    limitations = [str(item)[:300] for item in raw_limitations]
    server_reachable = value.get("server_reachable") is True and source == "approved_http"
    server_version = value.get("server_version")
    if server_version is not None:
        server_version = str(server_version)[:64]
    return (
        {"source": source, "collections": collections, "limitations": limitations},
        server_reachable,
        server_version,
    )


def capture_approved_http_logical_inventory(
    *,
    identity_authority_path: str | Path = DEFAULT_IDENTITY_AUTHORITY_PATH,
) -> dict[str, Any]:
    """Use the accepted HTTP-only fingerprint boundary without starting a server."""

    try:
        from backend.chroma_http_vector_search import (
            compute_github_evidence_logical_fingerprint_http,
        )
        from backend.project_repository_identity import (
            load_project_repository_identity_authority,
        )

        authority = load_project_repository_identity_authority(identity_authority_path)
        result = compute_github_evidence_logical_fingerprint_http(authority=authority)
    except Exception:
        return unavailable_logical_inventory()
    if not isinstance(result, Mapping) or result.get("status") != "ready":
        return unavailable_logical_inventory()
    record_ids = result.get("record_ids")
    repositories = result.get("repositories")
    if not isinstance(record_ids, list) or not isinstance(repositories, list):
        return unavailable_logical_inventory()
    safe_ids = sorted(
        item for item in record_ids if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", item)
    )
    if len(safe_ids) != len(record_ids):
        return unavailable_logical_inventory()
    return {
        "source": "approved_http",
        "collections": [
            {
                "semantic_name": result.get("collection_name"),
                "record_count": result.get("record_count"),
                "record_ids_sha256": sha256_json(safe_ids),
                "logical_fingerprint": result.get("fingerprint"),
                "repository_count": len(
                    {item for item in repositories if isinstance(item, str) and item}
                ),
                "schema_marker": None,
            }
        ],
        "limitations": [],
        "server_reachable": True,
        "server_version": None,
    }


def _configured_mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get("GITHUB_EVIDENCE_VECTOR_QUERY_BACKEND", "")).strip().casefold()
    if value in {"", "disabled"}:
        return "disabled"
    if value == "chroma_http":
        return "local_http"
    return "unknown"


def _validate_output_path(
    output_path: str | Path,
    *,
    repository_root: Path,
    protected_root: Path,
) -> Path:
    output = Path(output_path)
    if output.suffix.casefold() != ".json":
        _capture_error("unsafe_output_path")
    try:
        repository = repository_root.resolve(strict=True)
        protected = protected_root.resolve(strict=True)
        output_absolute = output.absolute()
        output_absolute.relative_to(repository)
        expected_parent = repository / "information" / "chroma_migration_baselines"
        output_absolute.relative_to(expected_parent)
    except (OSError, ValueError):
        _capture_error("unsafe_output_path")
    try:
        output_absolute.relative_to(protected)
    except ValueError:
        pass
    else:
        _capture_error("output_inside_protected_storage")
    current = repository
    for part in output_absolute.relative_to(repository).parts[:-1]:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            _capture_error("unsafe_output_parent")
    if output_absolute.exists() and _is_reparse_point(output_absolute):
        _capture_error("unsafe_output_path")
    return output_absolute


def write_baseline_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(content) > MAX_BASELINE_BYTES:
        _capture_error("baseline_size_limit_exceeded")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError:
        _capture_error("baseline_write_failed")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime):
        _capture_error("invalid_capture_clock")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def capture_chroma_migration_baseline(
    *,
    repository_root: str | Path = PROJECT_ROOT,
    protected_root: str | Path = DEFAULT_PROTECTED_CHROMA_ROOT,
    output_path: str | Path = DEFAULT_BASELINE_OUTPUT,
    artifact_specs: Sequence[ArtifactSpec | tuple[str, str]] = DEFAULT_ARTIFACT_SPECS,
    logical_inventory_provider: Callable[[], Mapping[str, Any]] | None = None,
    inventory_reader: Callable[[str | Path], dict[str, Any]] = capture_protected_file_inventory,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture an accepted baseline only after a matching protected-directory recheck."""

    repository_path, _ = _require_regular_root(repository_root)
    protected_path, _ = _require_regular_root(protected_root)
    safe_output = _validate_output_path(
        output_path, repository_root=repository_path, protected_root=protected_path
    )
    before = inventory_reader(protected_path)
    evidence_artifacts = capture_evidence_artifact_hashes(
        repository_path, artifact_specs=artifact_specs
    )
    repository_state = capture_chroma_client_call_inventory(repository_path)
    raw_logical = (
        logical_inventory_provider() if logical_inventory_provider is not None else unavailable_logical_inventory()
    )
    logical_inventory, server_reachable, server_version = _safe_logical_inventory(raw_logical)
    after = inventory_reader(protected_path)
    if before != after:
        _capture_error("protected_storage_changed_during_capture")
    payload = add_baseline_content_hash(
        {
            "schema": CHROMA_MIGRATION_BASELINE_SCHEMA,
            "captured_at": _timestamp(clock),
            "capture_mode": PROTECTED_CAPTURE_MODE,
            "deployment_observation": {
                "configured_mode": _configured_mode(environ),
                "server_reachable": server_reachable,
                "server_version": server_version,
            },
            "protected_storage": before,
            "logical_inventory": logical_inventory,
            "evidence_artifacts": evidence_artifacts,
            "repository_state": repository_state,
            "privacy": {
                "contains_documents": False,
                "contains_embeddings": False,
                "contains_raw_metadata": False,
                "contains_absolute_paths": False,
                "contains_secrets": False,
            },
        }
    )
    try:
        validate_chroma_migration_baseline(payload)
    except BaselineValidationError as error:
        _capture_error(error.code)
    write_baseline_atomic(safe_output, payload)
    return payload


def load_chroma_migration_baseline(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > MAX_BASELINE_BYTES:
            _capture_error("baseline_unavailable")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except BaselineCaptureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        _capture_error("baseline_unavailable")
    try:
        validate_chroma_migration_baseline(payload)
    except BaselineValidationError as error:
        _capture_error(error.code)
    return payload


def verify_chroma_migration_baseline(
    baseline_path: str | Path,
    *,
    repository_root: str | Path = PROJECT_ROOT,
    protected_root: str | Path = DEFAULT_PROTECTED_CHROMA_ROOT,
    compare_protected: bool = False,
    compare_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate an accepted baseline without rewriting or accepting changed hashes."""

    payload = load_chroma_migration_baseline(baseline_path)
    protected_match: bool | None = None
    artifacts_match: bool | None = None
    if compare_protected:
        protected_match = capture_protected_file_inventory(protected_root) == payload["protected_storage"]
    if compare_artifacts:
        specs = [
            ArtifactSpec(item["semantic_name"], item["relative_path"])
            for item in payload["evidence_artifacts"]["artifacts"]
        ]
        artifacts_match = capture_evidence_artifact_hashes(
            repository_root, artifact_specs=specs
        ) == payload["evidence_artifacts"]
    matches = all(value is not False for value in (protected_match, artifacts_match))
    return {
        "status": "verified" if matches else "mismatch",
        "schema_valid": True,
        "historical_byte_baseline": True,
        "protected_storage_compared": compare_protected,
        "protected_storage_match": protected_match,
        "evidence_artifacts_compared": compare_artifacts,
        "evidence_artifacts_match": artifacts_match,
        "logical_inventory_source": payload["logical_inventory"]["source"],
        "logical_inventory_available": payload["logical_inventory"]["source"] != "unavailable",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.chroma_migration_baseline",
        description="Capture or verify a privacy-safe protected Chroma byte baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    capture.add_argument("--protected-root", type=Path, default=DEFAULT_PROTECTED_CHROMA_ROOT)
    capture.add_argument("--output", type=Path, default=DEFAULT_BASELINE_OUTPUT)
    capture.add_argument("--approved-http", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    verify.add_argument("--protected-root", type=Path, default=DEFAULT_PROTECTED_CHROMA_ROOT)
    verify.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_OUTPUT)
    verify.add_argument("--compare-protected", action="store_true")
    verify.add_argument("--compare-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            provider = capture_approved_http_logical_inventory if args.approved_http else None
            payload = capture_chroma_migration_baseline(
                repository_root=args.repository_root,
                protected_root=args.protected_root,
                output_path=args.output,
                logical_inventory_provider=provider,
            )
            print(
                "capture succeeded "
                f"files={payload['protected_storage']['file_count']} "
                f"artifacts={len(payload['evidence_artifacts']['artifacts'])} "
                f"logical_inventory={payload['logical_inventory']['source']}"
            )
            return 0
        result = verify_chroma_migration_baseline(
            args.baseline,
            repository_root=args.repository_root,
            protected_root=args.protected_root,
            compare_protected=args.compare_protected,
            compare_artifacts=args.compare_artifacts,
        )
        print(
            f"verification result={result['status']} "
            f"logical_inventory={result['logical_inventory_source']}"
        )
        return 0 if result["status"] == "verified" else 1
    except (BaselineCaptureError, BaselineValidationError):
        action = "capture" if args.command == "capture" else "verification"
        print(f"{action} failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
