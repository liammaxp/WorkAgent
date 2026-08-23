"""Strict, privacy-safe models for Chroma backup and recovery evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CHROMA_BACKUP_MANIFEST_SCHEMA = "chroma_backup_manifest.v1"
CHROMA_BACKUP_STATUS_SCHEMA = "chroma_backup_status.v1"
CHROMA_RESTORE_STATUS_SCHEMA = "chroma_restore_status.v1"
CHROMA_COMPATIBILITY_STATUS_SCHEMA = "chroma_compatibility_status.v1"
CHROMA_RECOVERY_GATE_SCHEMA = "chroma_recovery_gate.v1"
BACKUP_SOURCE_KIND = "production_chroma_persistence"
TEST_BACKUP_SOURCE_KIND = "test_owned_chroma_persistence"
BACKUP_COMPATIBILITY_POLICY = "executable_probe_required"
BACKUP_CAPTURE_STATES = frozenset({"staging", "verified", "invalid"})
COMPATIBILITY_CLASSIFICATIONS = frozenset(
    {"compatible", "migration_required", "incompatible", "unknown"}
)
MAX_BACKUP_FILES = 100_000
MAX_BACKUP_MANIFEST_BYTES = 8_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BACKUP_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")
_SAFE_DETAIL_RE = re.compile(r"^[a-z0-9_:-]{1,96}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)^[a-z]:[\\/]")
_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "absolute_path",
        "document",
        "documents",
        "embedding",
        "embeddings",
        "environment",
        "metadata",
        "password",
        "raw_metadata",
        "secret",
        "token",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "backup_id",
        "source_kind",
        "source_relative_path",
        "file_count",
        "total_bytes",
        "aggregate_sha256",
        "files",
        "chroma_package_version",
        "compatibility_policy",
        "capture_state",
        "created_at",
        "source_server_state",
    }
)
_FILE_FIELDS = frozenset({"relative_path", "size_bytes", "sha256"})


class ChromaBackupError(RuntimeError):
    """Stable backup/recovery error that never includes local paths or data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ChromaBackupConfigurationError(ChromaBackupError):
    pass


class ChromaBackupPreconditionFailed(ChromaBackupError):
    pass


class ChromaBackupCaptureFailed(ChromaBackupError):
    pass


class ChromaBackupManifestInvalid(ChromaBackupError):
    pass


class ChromaBackupVerificationFailed(ChromaBackupError):
    pass


class ChromaRestoreTargetRejected(ChromaBackupError):
    pass


class ChromaRestoreFailed(ChromaBackupError):
    pass


class ChromaCompatibilityFailed(ChromaBackupError):
    pass


def validate_backup_id(value: Any) -> str:
    if not isinstance(value, str) or not _BACKUP_ID_RE.fullmatch(value):
        raise ChromaBackupManifestInvalid("invalid_chroma_backup_id")
    return value


def validate_sha256(value: Any, code: str = "invalid_chroma_backup_hash") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ChromaBackupManifestInvalid(code)
    return value


def validate_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or _WINDOWS_ABSOLUTE_RE.match(value)
    ):
        raise ChromaBackupManifestInvalid("unsafe_chroma_backup_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ChromaBackupManifestInvalid("unsafe_chroma_backup_relative_path")
    canonical = path.as_posix()
    if canonical != value:
        raise ChromaBackupManifestInvalid("unsafe_chroma_backup_relative_path")
    return canonical


def _validate_nonnegative_int(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromaBackupManifestInvalid(code)
    return value


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ChromaBackupManifestInvalid("invalid_chroma_backup_timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ChromaBackupManifestInvalid("invalid_chroma_backup_timestamp") from None
    if parsed.utcoffset() is None or len(value) > 40:
        raise ChromaBackupManifestInvalid("invalid_chroma_backup_timestamp")
    return value


def _validate_version(value: Any) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ChromaBackupManifestInvalid("invalid_chroma_backup_package_version")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class ChromaBackupFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", validate_relative_path(self.relative_path))
        object.__setattr__(
            self,
            "size_bytes",
            _validate_nonnegative_int(self.size_bytes, "invalid_chroma_backup_file_size"),
        )
        object.__setattr__(
            self,
            "sha256",
            validate_sha256(self.sha256, "invalid_chroma_backup_file_hash"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ChromaBackupFile:
        if not isinstance(payload, Mapping) or frozenset(payload) != _FILE_FIELDS:
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_file_shape")
        return cls(
            relative_path=payload["relative_path"],
            size_bytes=payload["size_bytes"],
            sha256=payload["sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaBackupManifest:
    schema: str
    backup_id: str
    source_kind: str
    source_relative_path: str
    file_count: int
    total_bytes: int
    aggregate_sha256: str
    files: tuple[ChromaBackupFile, ...]
    chroma_package_version: str
    compatibility_policy: str
    capture_state: str
    created_at: str
    source_server_state: str

    def __post_init__(self) -> None:
        if self.schema != CHROMA_BACKUP_MANIFEST_SCHEMA:
            raise ChromaBackupManifestInvalid("unsupported_chroma_backup_schema")
        object.__setattr__(self, "backup_id", validate_backup_id(self.backup_id))
        if self.source_kind not in {BACKUP_SOURCE_KIND, TEST_BACKUP_SOURCE_KIND}:
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_source_kind")
        object.__setattr__(
            self, "source_relative_path", validate_relative_path(self.source_relative_path)
        )
        file_count = _validate_nonnegative_int(
            self.file_count, "invalid_chroma_backup_file_count"
        )
        total_bytes = _validate_nonnegative_int(
            self.total_bytes, "invalid_chroma_backup_total_bytes"
        )
        if not isinstance(self.files, tuple) or len(self.files) > MAX_BACKUP_FILES:
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_files")
        if any(not isinstance(item, ChromaBackupFile) for item in self.files):
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_files")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)):
            raise ChromaBackupManifestInvalid("unordered_chroma_backup_files")
        if len(paths) != len(set(paths)):
            raise ChromaBackupManifestInvalid("duplicate_chroma_backup_file")
        if file_count != len(self.files):
            raise ChromaBackupManifestInvalid("chroma_backup_file_count_mismatch")
        if total_bytes != sum(item.size_bytes for item in self.files):
            raise ChromaBackupManifestInvalid("chroma_backup_total_bytes_mismatch")
        object.__setattr__(self, "file_count", file_count)
        object.__setattr__(self, "total_bytes", total_bytes)
        object.__setattr__(self, "aggregate_sha256", validate_sha256(self.aggregate_sha256))
        object.__setattr__(
            self, "chroma_package_version", _validate_version(self.chroma_package_version)
        )
        if self.compatibility_policy != BACKUP_COMPATIBILITY_POLICY:
            raise ChromaBackupManifestInvalid(
                "unsupported_chroma_backup_compatibility_policy"
            )
        if self.capture_state not in BACKUP_CAPTURE_STATES:
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_state")
        object.__setattr__(self, "created_at", _validate_timestamp(self.created_at))
        if self.source_server_state != "not_running":
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_server_state")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ChromaBackupManifest:
        if not isinstance(payload, Mapping) or frozenset(payload) != _MANIFEST_FIELDS:
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_manifest_shape")
        if frozenset(payload) & _FORBIDDEN_MANIFEST_KEYS:
            raise ChromaBackupManifestInvalid("unsafe_chroma_backup_manifest_field")
        raw_files = payload["files"]
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise ChromaBackupManifestInvalid("invalid_chroma_backup_files")
        return cls(
            schema=payload["schema"],
            backup_id=payload["backup_id"],
            source_kind=payload["source_kind"],
            source_relative_path=payload["source_relative_path"],
            file_count=payload["file_count"],
            total_bytes=payload["total_bytes"],
            aggregate_sha256=payload["aggregate_sha256"],
            files=tuple(ChromaBackupFile.from_mapping(item) for item in raw_files),
            chroma_package_version=payload["chroma_package_version"],
            compatibility_policy=payload["compatibility_policy"],
            capture_state=payload["capture_state"],
            created_at=payload["created_at"],
            source_server_state=payload["source_server_state"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backup_id": self.backup_id,
            "source_kind": self.source_kind,
            "source_relative_path": self.source_relative_path,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.to_dict() for item in self.files],
            "chroma_package_version": self.chroma_package_version,
            "compatibility_policy": self.compatibility_policy,
            "capture_state": self.capture_state,
            "created_at": self.created_at,
            "source_server_state": self.source_server_state,
        }

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backup_id": self.backup_id,
            "capture_state": self.capture_state,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "aggregate_sha256": self.aggregate_sha256,
            "chroma_package_version": self.chroma_package_version,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaBackupPreflight:
    source_state: str
    server_state: str
    endpoint_state: str
    runtime_state_residue: bool
    source_fingerprint: str | None
    capture_allowed: bool
    blocker: str | None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": CHROMA_BACKUP_STATUS_SCHEMA,
            "source_state": self.source_state,
            "server_state": self.server_state,
            "endpoint_state": self.endpoint_state,
            "runtime_state_residue": self.runtime_state_residue,
            "source_fingerprint": self.source_fingerprint,
            "capture_allowed": self.capture_allowed,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaBackupVerification:
    backup_id: str
    verified: bool
    file_count: int
    total_bytes: int
    aggregate_sha256: str

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": CHROMA_BACKUP_STATUS_SCHEMA,
            "backup_id": self.backup_id,
            "state": "verified" if self.verified else "invalid",
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "aggregate_sha256": self.aggregate_sha256,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaRestoreResult:
    backup_id: str
    target_type: str
    file_count: int
    total_bytes: int
    aggregate_sha256: str
    manifest_match: bool
    immutable_backup_unchanged: bool

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": CHROMA_RESTORE_STATUS_SCHEMA,
            "backup_id": self.backup_id,
            "target_type": self.target_type,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "aggregate_sha256": self.aggregate_sha256,
            "manifest_match": self.manifest_match,
            "immutable_backup_unchanged": self.immutable_backup_unchanged,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaCompatibilityResult:
    backup_id: str
    capture_chroma_version: str
    current_chroma_version: str | None
    server_started: bool
    heartbeat_succeeded: bool
    collection_lookup: Mapping[str, str]
    safe_counts: Mapping[str, int | None]
    workspace_fingerprint_before: str | None
    workspace_fingerprint_after: str | None
    server_open_mutated_internal_storage: bool
    changed_file_count: int
    immutable_backup_unchanged: bool
    restored_copy_unchanged: bool
    classification: str
    blocker: str | None

    def __post_init__(self) -> None:
        if self.classification not in COMPATIBILITY_CLASSIFICATIONS:
            raise ChromaCompatibilityFailed("invalid_chroma_compatibility_classification")
        object.__setattr__(self, "collection_lookup", MappingProxyType(dict(self.collection_lookup)))
        object.__setattr__(self, "safe_counts", MappingProxyType(dict(self.safe_counts)))

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": CHROMA_COMPATIBILITY_STATUS_SCHEMA,
            "backup_id": self.backup_id,
            "capture_chroma_version": self.capture_chroma_version,
            "current_chroma_version": self.current_chroma_version,
            "server_started": self.server_started,
            "heartbeat_succeeded": self.heartbeat_succeeded,
            "collection_lookup": dict(self.collection_lookup),
            "safe_counts": dict(self.safe_counts),
            "workspace_fingerprint_before": self.workspace_fingerprint_before,
            "workspace_fingerprint_after": self.workspace_fingerprint_after,
            "server_open_mutated_internal_storage": self.server_open_mutated_internal_storage,
            "changed_file_count": self.changed_file_count,
            "immutable_backup_unchanged": self.immutable_backup_unchanged,
            "restored_copy_unchanged": self.restored_copy_unchanged,
            "classification": self.classification,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ChromaRecoveryGateResult:
    backup_id: str
    backup_verified: bool
    restore_drill_passed: bool
    compatibility: str
    immutable_backup_unchanged: bool
    rollback_source_ready: bool
    production_cutover_recovery_gate: str
    blocker: str | None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": CHROMA_RECOVERY_GATE_SCHEMA,
            "backup_id": self.backup_id,
            "backup_verified": self.backup_verified,
            "restore_drill_passed": self.restore_drill_passed,
            "version_compatibility": self.compatibility,
            "immutable_backup_unchanged": self.immutable_backup_unchanged,
            "rollback_source_ready": self.rollback_source_ready,
            "production_cutover_recovery_gate": self.production_cutover_recovery_gate,
            "blocker": self.blocker,
        }


def build_recovery_gate_result(
    *,
    backup_id: str,
    backup_verified: bool,
    restore_drill_passed: bool,
    compatibility: str,
    immutable_backup_unchanged: bool,
) -> ChromaRecoveryGateResult:
    validate_backup_id(backup_id)
    if compatibility not in COMPATIBILITY_CLASSIFICATIONS:
        raise ChromaCompatibilityFailed("invalid_chroma_compatibility_classification")
    ready = bool(
        backup_verified
        and restore_drill_passed
        and compatibility == "compatible"
        and immutable_backup_unchanged
    )
    if ready:
        blocker = None
    elif not backup_verified:
        blocker = "backup_verification_required"
    elif not restore_drill_passed:
        blocker = "restore_drill_required"
    elif not immutable_backup_unchanged:
        blocker = "immutable_backup_changed"
    else:
        blocker = "compatible_version_evidence_required"
    return ChromaRecoveryGateResult(
        backup_id=backup_id,
        backup_verified=backup_verified,
        restore_drill_passed=restore_drill_passed,
        compatibility=compatibility,
        immutable_backup_unchanged=immutable_backup_unchanged,
        rollback_source_ready=ready,
        production_cutover_recovery_gate="recovery_ready" if ready else "blocked",
        blocker=blocker,
    )
