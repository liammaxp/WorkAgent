"""Filesystem-only Chroma backup, isolated restore, and executable recovery gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import socket
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.chroma_backup_models import (
    BACKUP_COMPATIBILITY_POLICY,
    BACKUP_SOURCE_KIND,
    CHROMA_BACKUP_MANIFEST_SCHEMA,
    MAX_BACKUP_MANIFEST_BYTES,
    TEST_BACKUP_SOURCE_KIND,
    ChromaBackupCaptureFailed,
    ChromaBackupConfigurationError,
    ChromaBackupError,
    ChromaBackupFile,
    ChromaBackupManifest,
    ChromaBackupManifestInvalid,
    ChromaBackupPreconditionFailed,
    ChromaBackupPreflight,
    ChromaBackupVerification,
    ChromaBackupVerificationFailed,
    ChromaCompatibilityResult,
    ChromaRecoveryGateResult,
    ChromaRestoreFailed,
    ChromaRestoreResult,
    ChromaRestoreTargetRejected,
    build_recovery_gate_result,
    validate_backup_id,
)
from backend.chroma_config import (
    EXISTING_LOCAL_HTTP_PORT,
    LOOPBACK_HOST,
    ChromaDeploymentConfig,
    ChromaDeploymentMode,
)
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import ChromaCollectionMissing, ChromaTransportError
from backend.chroma_migration_baseline import (
    DEFAULT_BASELINE_OUTPUT,
    DEFAULT_PROTECTED_CHROMA_ROOT,
    capture_protected_file_inventory,
    load_chroma_migration_baseline,
)
from backend.chroma_persistence_guard import ChromaPersistenceGuard
from backend.chroma_server_lifecycle import (
    ChromaServerLifecycleController,
    inspect_chroma_server_ownership,
)
from backend.chroma_server_lifecycle_models import (
    AtomicChromaServerStateStore,
    ChromaServerLifecycleConfig,
    ChromaServerLifecycleResult,
    build_chroma_server_lifecycle_config,
)
from backend.chroma_collection_registry import list_registered_collections
from backend.chroma_baseline_models import canonical_json, sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = PROJECT_ROOT / "information" / "backups" / "chroma"
DEFAULT_SOURCE_RELATIVE_PATH = "information/chroma"
DEFAULT_TEST_SOURCE_RELATIVE_PATH = "test-owned/chroma"
_BACKUP_DIRECTORY_FIELDS = frozenset({"manifest.json", "snapshot"})
_STAGING_PREFIX = ".workagent-chroma-backup-staging-"
_RESTORE_STAGING_PREFIX = ".workagent-chroma-restore-staging-"


LifecycleObserver = Callable[[ChromaServerLifecycleConfig], ChromaServerLifecycleResult]
FileCopier = Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any]
DirectoryFinalizer = Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any]
ManifestWriter = Callable[[Path, ChromaBackupManifest], None]
VersionProvider = Callable[[str], str]


def _resolved(path: str | Path, code: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ChromaBackupConfigurationError(code)
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise ChromaBackupConfigurationError(code) from None


def _is_root(path: Path) -> bool:
    try:
        return path == Path(path.anchor).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return True


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True, repr=False)
class ChromaBackupConfiguration:
    repository_root: Path
    source_root: Path
    backup_root: Path
    lifecycle_config: ChromaServerLifecycleConfig
    source_relative_path: str
    source_kind: str
    test_owned: bool = False
    test_root: Path | None = None

    def __post_init__(self) -> None:
        repository = _resolved(self.repository_root, "invalid_backup_repository_root")
        source = _resolved(self.source_root, "invalid_backup_source_root")
        backup = _resolved(self.backup_root, "invalid_backup_storage_root")
        if not repository.is_dir() or _is_root(repository) or _is_root(source) or _is_root(backup):
            raise ChromaBackupConfigurationError("unsafe_chroma_backup_configuration")
        if not isinstance(self.lifecycle_config, ChromaServerLifecycleConfig):
            raise ChromaBackupConfigurationError("invalid_backup_lifecycle_configuration")
        if not isinstance(self.test_owned, bool):
            raise ChromaBackupConfigurationError("invalid_backup_test_ownership")
        if source == backup or _is_within(backup, source) or _is_within(source, backup):
            raise ChromaBackupConfigurationError("overlapping_chroma_backup_paths")
        production_source = DEFAULT_PROTECTED_CHROMA_ROOT.resolve(strict=False)
        production_backup = DEFAULT_BACKUP_ROOT.resolve(strict=False)
        if self.test_owned:
            if self.test_root is None or not self.lifecycle_config.test_owned:
                raise ChromaBackupConfigurationError("test_backup_context_required")
            test_root = _resolved(self.test_root, "invalid_backup_test_root")
            if not test_root.is_dir() or _is_root(test_root):
                raise ChromaBackupConfigurationError("invalid_backup_test_root")
            if source in {test_root, production_source} or backup in {
                test_root,
                production_backup,
            }:
                raise ChromaBackupConfigurationError("unsafe_test_backup_path")
            if not _is_within(source, test_root) or not _is_within(backup, test_root):
                raise ChromaBackupConfigurationError("test_backup_path_escape")
            if self.source_kind != TEST_BACKUP_SOURCE_KIND:
                raise ChromaBackupConfigurationError("invalid_test_backup_source_kind")
            object.__setattr__(self, "test_root", test_root)
        else:
            if (
                repository != PROJECT_ROOT.resolve(strict=False)
                or source != production_source
                or backup != production_backup
                or self.lifecycle_config.test_owned
                or self.lifecycle_config.persistence_path != production_source
                or self.source_kind != BACKUP_SOURCE_KIND
                or self.source_relative_path != DEFAULT_SOURCE_RELATIVE_PATH
            ):
                raise ChromaBackupConfigurationError("production_backup_paths_are_fixed")
            if self.test_root is not None:
                raise ChromaBackupConfigurationError("unexpected_backup_test_root")
        object.__setattr__(self, "repository_root", repository)
        object.__setattr__(self, "source_root", source)
        object.__setattr__(self, "backup_root", backup)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "source_scope": "test_owned" if self.test_owned else "production_protected",
            "backup_scope": "test_owned" if self.test_owned else "ignored_local_backup",
            "source_kind": self.source_kind,
            "test_owned": self.test_owned,
        }


def build_production_backup_configuration() -> ChromaBackupConfiguration:
    deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        LOOPBACK_HOST,
        EXISTING_LOCAL_HTTP_PORT,
        False,
        1.0,
    )
    return ChromaBackupConfiguration(
        repository_root=PROJECT_ROOT,
        source_root=DEFAULT_PROTECTED_CHROMA_ROOT,
        backup_root=DEFAULT_BACKUP_ROOT,
        lifecycle_config=build_chroma_server_lifecycle_config(deployment),
        source_relative_path=DEFAULT_SOURCE_RELATIVE_PATH,
        source_kind=BACKUP_SOURCE_KIND,
    )


def build_test_backup_configuration(
    *,
    source_root: str | Path,
    backup_root: str | Path,
    test_root: str | Path,
    lifecycle_config: ChromaServerLifecycleConfig,
) -> ChromaBackupConfiguration:
    return ChromaBackupConfiguration(
        repository_root=PROJECT_ROOT,
        source_root=Path(source_root),
        backup_root=Path(backup_root),
        lifecycle_config=lifecycle_config,
        source_relative_path=DEFAULT_TEST_SOURCE_RELATIVE_PATH,
        source_kind=TEST_BACKUP_SOURCE_KIND,
        test_owned=True,
        test_root=Path(test_root),
    )


def _default_lifecycle_observer(
    config: ChromaServerLifecycleConfig,
) -> ChromaServerLifecycleResult:
    return inspect_chroma_server_ownership(config)


def _fingerprint(root: Path) -> dict[str, Any]:
    try:
        return capture_protected_file_inventory(root)
    except Exception as error:
        code = getattr(error, "code", "backup_source_inventory_failed")
        raise ChromaBackupPreconditionFailed(str(code)) from None


def _expected_production_fingerprint() -> dict[str, Any]:
    try:
        baseline = load_chroma_migration_baseline(DEFAULT_BASELINE_OUTPUT)
        expected = baseline["protected_storage"]
    except Exception:
        raise ChromaBackupPreconditionFailed("accepted_backup_baseline_unavailable") from None
    if not isinstance(expected, Mapping):
        raise ChromaBackupPreconditionFailed("accepted_backup_baseline_unavailable")
    return dict(expected)


def preflight_backup_capture(
    config: ChromaBackupConfiguration | None = None,
    *,
    test_lifecycle_observer: LifecycleObserver | None = None,
    verified_post_cutover_inventory: Mapping[str, Any] | None = None,
) -> ChromaBackupPreflight:
    resolved = config or build_production_backup_configuration()
    if not isinstance(resolved, ChromaBackupConfiguration):
        raise ChromaBackupConfigurationError("invalid_backup_configuration")
    if (
        verified_post_cutover_inventory is not None
        and not isinstance(verified_post_cutover_inventory, Mapping)
    ):
        raise ChromaBackupConfigurationError(
            "invalid_verified_post_cutover_inventory"
        )
    if verified_post_cutover_inventory is not None and resolved.test_owned:
        raise ChromaBackupConfigurationError(
            "verified_post_cutover_inventory_requires_production_backup"
        )
    if test_lifecycle_observer is not None and not resolved.test_owned:
        raise ChromaBackupConfigurationError(
            "lifecycle_observer_injection_requires_test_backup"
        )
    observer = test_lifecycle_observer or _default_lifecycle_observer
    try:
        lifecycle = observer(resolved.lifecycle_config)
    except Exception:
        lifecycle = None
    runtime_residue = resolved.lifecycle_config.runtime_state_path.exists()
    if not isinstance(lifecycle, ChromaServerLifecycleResult):
        return ChromaBackupPreflight(
            "unverified", "ambiguous", "ambiguous", runtime_residue, None, False,
            "lifecycle_ownership_ambiguous",
        )
    endpoint_state = "free" if lifecycle.state == "not_running" else "occupied_or_ambiguous"
    if (
        lifecycle.state != "not_running"
        or lifecycle.process_owned
        or lifecycle.server_reachable
    ):
        return ChromaBackupPreflight(
            "unverified",
            lifecycle.state,
            endpoint_state,
            runtime_residue,
            None,
            False,
            "production_server_not_verified_stopped",
        )
    if runtime_residue:
        return ChromaBackupPreflight(
            "unverified",
            lifecycle.state,
            endpoint_state,
            True,
            None,
            False,
            "runtime_state_residue_present",
        )
    try:
        inventory = _fingerprint(resolved.source_root)
    except ChromaBackupPreconditionFailed as error:
        return ChromaBackupPreflight(
            "invalid",
            lifecycle.state,
            endpoint_state,
            False,
            None,
            False,
            error.code,
        )
    if not resolved.test_owned:
        expected_production_inventory = (
            dict(verified_post_cutover_inventory)
            if verified_post_cutover_inventory is not None
            else _expected_production_fingerprint()
        )
        if inventory != expected_production_inventory:
            return ChromaBackupPreflight(
                "drifted",
                lifecycle.state,
                endpoint_state,
                False,
                inventory["aggregate_sha256"],
                False,
                "protected_backup_source_drift",
            )
    return ChromaBackupPreflight(
        "verified",
        lifecycle.state,
        endpoint_state,
        False,
        inventory["aggregate_sha256"],
        True,
        None,
    )


def _require_capture_preflight(preflight: ChromaBackupPreflight) -> None:
    if preflight.capture_allowed is not True:
        raise ChromaBackupPreconditionFailed(
            preflight.blocker or "backup_capture_not_allowed"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_capture_time(now_provider: Callable[[], datetime]) -> tuple[str, str]:
    try:
        now = now_provider()
    except Exception:
        raise ChromaBackupCaptureFailed("backup_capture_clock_failed") from None
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ChromaBackupCaptureFailed("invalid_backup_capture_time")
    utc = now.astimezone(timezone.utc).replace(microsecond=0)
    return utc.strftime("%Y%m%dT%H%M%SZ"), utc.isoformat().replace("+00:00", "Z")


def _package_version(version_provider: VersionProvider) -> str:
    try:
        version = version_provider("chromadb")
    except Exception:
        raise ChromaBackupCaptureFailed("chroma_package_version_unavailable") from None
    if not isinstance(version, str) or not version or len(version) > 64:
        raise ChromaBackupCaptureFailed("chroma_package_version_unavailable")
    return version


def _files_from_inventory(inventory: Mapping[str, Any]) -> tuple[ChromaBackupFile, ...]:
    try:
        return tuple(
            ChromaBackupFile(
                relative_path=item["relative_path"],
                size_bytes=item["size_bytes"],
                sha256=item["sha256"],
            )
            for item in inventory["files"]
        )
    except (KeyError, TypeError, ChromaBackupManifestInvalid):
        raise ChromaBackupCaptureFailed("invalid_backup_source_inventory") from None


def _copy_inventory(
    source: Path,
    destination: Path,
    inventory: Mapping[str, Any],
    *,
    copier: FileCopier,
) -> None:
    for item in inventory["files"]:
        relative = item["relative_path"]
        source_file = source / Path(*relative.split("/"))
        destination_file = destination / Path(*relative.split("/"))
        try:
            resolved_source = source_file.resolve(strict=True)
            resolved_source.relative_to(source.resolve(strict=True))
            if not resolved_source.is_file() or resolved_source.is_symlink():
                raise OSError
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            copier(resolved_source, destination_file)
        except Exception:
            raise ChromaBackupCaptureFailed("backup_file_copy_failed") from None


def _write_manifest_atomic(path: Path, manifest: ChromaBackupManifest) -> None:
    temporary = path.parent / f".manifest-{uuid4().hex}.tmp"
    encoded = (canonical_json(manifest.to_dict()) + "\n").encode("utf-8")
    if len(encoded) > MAX_BACKUP_MANIFEST_BYTES:
        raise ChromaBackupCaptureFailed("chroma_backup_manifest_too_large")
    try:
        with open(temporary, "xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ChromaBackupCaptureFailed("chroma_backup_manifest_write_failed") from None


def _remove_owned_directory(path: Path, *, parent: Path, prefix: str) -> None:
    if path.parent != parent or not path.name.startswith(prefix):
        raise ChromaBackupCaptureFailed("unsafe_backup_cleanup_target")
    if path.exists():
        shutil.rmtree(path)


def capture_chroma_backup(
    config: ChromaBackupConfiguration | None = None,
    *,
    test_lifecycle_observer: LifecycleObserver | None = None,
    now_provider: Callable[[], datetime] = _utc_now,
    version_provider: VersionProvider = importlib.metadata.version,
    copier: FileCopier = shutil.copyfile,
    manifest_writer: ManifestWriter = _write_manifest_atomic,
    finalizer: DirectoryFinalizer = os.replace,
    verified_post_cutover_inventory: Mapping[str, Any] | None = None,
) -> ChromaBackupManifest:
    """Capture stopped storage; a post-cutover inventory must match byte-for-byte.

    The default production path remains pinned to the accepted offline baseline.
    Callers may supply a full inventory only after a separate online logical
    integrity gate; this function never derives or accepts a new baseline.
    """

    resolved = config or build_production_backup_configuration()
    preflight = preflight_backup_capture(
        resolved,
        test_lifecycle_observer=test_lifecycle_observer,
        verified_post_cutover_inventory=verified_post_cutover_inventory,
    )
    _require_capture_preflight(preflight)
    source_before = _fingerprint(resolved.source_root)
    timestamp_id, created_at = _safe_capture_time(now_provider)
    backup_id = f"{timestamp_id}-{source_before['aggregate_sha256'][:12]}"
    validate_backup_id(backup_id)
    version = _package_version(version_provider)
    try:
        resolved.backup_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ChromaBackupCaptureFailed("backup_root_unavailable") from None
    if resolved.backup_root.is_symlink() or not resolved.backup_root.is_dir():
        raise ChromaBackupCaptureFailed("unsafe_backup_root")
    final_directory = resolved.backup_root / backup_id
    if final_directory.exists():
        raise ChromaBackupCaptureFailed("backup_id_already_exists")
    staging = resolved.backup_root / f"{_STAGING_PREFIX}{backup_id}-{uuid4().hex}"
    snapshot = staging / "snapshot"
    finalized = False
    try:
        snapshot.mkdir(parents=True, exist_ok=False)
        _copy_inventory(
            resolved.source_root,
            snapshot,
            source_before,
            copier=copier,
        )
        copied = _fingerprint(snapshot)
        if copied != source_before:
            raise ChromaBackupCaptureFailed("backup_snapshot_verification_failed")
        source_after = _fingerprint(resolved.source_root)
        if source_after != source_before:
            raise ChromaBackupCaptureFailed("backup_source_changed_during_capture")
        closing_preflight = preflight_backup_capture(
            resolved,
            test_lifecycle_observer=test_lifecycle_observer,
            verified_post_cutover_inventory=verified_post_cutover_inventory,
        )
        _require_capture_preflight(closing_preflight)
        if closing_preflight.source_fingerprint != source_before["aggregate_sha256"]:
            raise ChromaBackupCaptureFailed("backup_source_changed_during_capture")
        manifest = ChromaBackupManifest(
            schema=CHROMA_BACKUP_MANIFEST_SCHEMA,
            backup_id=backup_id,
            source_kind=resolved.source_kind,
            source_relative_path=resolved.source_relative_path,
            file_count=source_before["file_count"],
            total_bytes=source_before["total_bytes"],
            aggregate_sha256=source_before["aggregate_sha256"],
            files=_files_from_inventory(source_before),
            chroma_package_version=version,
            compatibility_policy=BACKUP_COMPATIBILITY_POLICY,
            capture_state="verified",
            created_at=created_at,
            source_server_state="not_running",
        )
        manifest_writer(staging / "manifest.json", manifest)
        if final_directory.exists():
            raise ChromaBackupCaptureFailed("backup_id_already_exists")
        try:
            finalizer(staging, final_directory)
        except Exception:
            raise ChromaBackupCaptureFailed("backup_atomic_finalize_failed") from None
        finalized = True
        verification = verify_chroma_backup(
            backup_id, backup_root=resolved.backup_root
        )
        if not verification.verified:
            raise ChromaBackupCaptureFailed("backup_independent_verification_failed")
        return manifest
    except ChromaBackupError:
        if staging.exists():
            _remove_owned_directory(
                staging, parent=resolved.backup_root, prefix=_STAGING_PREFIX
            )
        if finalized and final_directory.exists():
            # The directory was created by this invocation and has not been returned.
            shutil.rmtree(final_directory)
        raise
    except Exception:
        if staging.exists():
            _remove_owned_directory(
                staging, parent=resolved.backup_root, prefix=_STAGING_PREFIX
            )
        if finalized and final_directory.exists():
            shutil.rmtree(final_directory)
        raise ChromaBackupCaptureFailed("backup_capture_failed") from None


def _backup_directory(backup_root: str | Path, backup_id: str) -> tuple[Path, Path]:
    identifier = validate_backup_id(backup_id)
    root = _resolved(backup_root, "invalid_backup_storage_root")
    directory = (root / identifier).resolve(strict=False)
    if directory.parent != root:
        raise ChromaBackupVerificationFailed("unsafe_backup_identifier")
    return root, directory


def _load_manifest(path: Path) -> ChromaBackupManifest:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BACKUP_MANIFEST_BYTES:
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ChromaBackupManifestInvalid("chroma_backup_manifest_unavailable") from None
    return ChromaBackupManifest.from_mapping(payload)


def _manifest_inventory(manifest: ChromaBackupManifest) -> dict[str, Any]:
    files = [item.to_dict() for item in manifest.files]
    return {
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "files": files,
        "aggregate_sha256": sha256_json(files),
    }


def verify_chroma_backup(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> ChromaBackupVerification:
    root, directory = _backup_directory(backup_root, backup_id)
    try:
        if root.is_symlink() or directory.is_symlink() or not directory.is_dir():
            raise ChromaBackupVerificationFailed("chroma_backup_unavailable")
        entries = frozenset(item.name for item in directory.iterdir())
    except ChromaBackupVerificationFailed:
        raise
    except OSError:
        raise ChromaBackupVerificationFailed("chroma_backup_unavailable") from None
    if entries != _BACKUP_DIRECTORY_FIELDS:
        raise ChromaBackupVerificationFailed("invalid_chroma_backup_layout")
    manifest = _load_manifest(directory / "manifest.json")
    if manifest.backup_id != backup_id or manifest.capture_state != "verified":
        raise ChromaBackupVerificationFailed("chroma_backup_not_verified")
    expected = _manifest_inventory(manifest)
    if expected["aggregate_sha256"] != manifest.aggregate_sha256:
        raise ChromaBackupVerificationFailed("chroma_backup_manifest_aggregate_mismatch")
    try:
        observed = _fingerprint(directory / "snapshot")
    except ChromaBackupPreconditionFailed as error:
        raise ChromaBackupVerificationFailed(error.code) from None
    if observed != expected:
        raise ChromaBackupVerificationFailed("chroma_backup_snapshot_mismatch")
    return ChromaBackupVerification(
        backup_id=backup_id,
        verified=True,
        file_count=observed["file_count"],
        total_bytes=observed["total_bytes"],
        aggregate_sha256=observed["aggregate_sha256"],
    )


def load_verified_chroma_backup_manifest(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> ChromaBackupManifest:
    """Return the strict manifest only after independently verifying its snapshot."""

    verify_chroma_backup(backup_id, backup_root=backup_root)
    _, directory = _backup_directory(backup_root, backup_id)
    return _load_manifest(directory / "manifest.json")


def _validate_restore_target(
    target: str | Path,
    *,
    approved_target_root: str | Path,
    backup_root: str | Path,
) -> tuple[Path, Path]:
    approved = _resolved(approved_target_root, "invalid_restore_target_root")
    candidate = _resolved(target, "invalid_restore_target")
    backups = _resolved(backup_root, "invalid_backup_storage_root")
    if not approved.is_dir() or approved.is_symlink() or _is_root(approved):
        raise ChromaRestoreTargetRejected("unsafe_restore_target_root")
    if candidate == approved or not _is_within(candidate, approved):
        raise ChromaRestoreTargetRejected("restore_target_escape")
    guard = ChromaPersistenceGuard(build_production_backup_configuration().lifecycle_config)
    if guard.is_server_owned_path(candidate) or guard.is_server_owned_path(approved):
        raise ChromaRestoreTargetRejected("protected_restore_target_forbidden")
    if candidate.exists():
        raise ChromaRestoreTargetRejected("restore_target_must_be_absent")
    if _is_within(candidate, backups) or _is_within(backups, candidate):
        raise ChromaRestoreTargetRejected("restore_target_overlaps_backup")
    return approved, candidate


def restore_chroma_backup(
    backup_id: str,
    *,
    target: str | Path,
    approved_target_root: str | Path,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
    copier: FileCopier = shutil.copyfile,
    finalizer: DirectoryFinalizer = os.replace,
) -> ChromaRestoreResult:
    verification = verify_chroma_backup(backup_id, backup_root=backup_root)
    root, directory = _backup_directory(backup_root, backup_id)
    approved, candidate = _validate_restore_target(
        target, approved_target_root=approved_target_root, backup_root=root
    )
    snapshot_before = _fingerprint(directory / "snapshot")
    staging = approved / f"{_RESTORE_STAGING_PREFIX}{candidate.name}-{uuid4().hex}"
    finalized = False
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _copy_inventory(
            directory / "snapshot",
            staging,
            snapshot_before,
            copier=copier,
        )
        staged = _fingerprint(staging)
        if staged != snapshot_before:
            raise ChromaRestoreFailed("restored_bytes_mismatch")
        if candidate.exists():
            raise ChromaRestoreTargetRejected("restore_target_must_be_absent")
        try:
            finalizer(staging, candidate)
        except Exception:
            raise ChromaRestoreFailed("restore_atomic_finalize_failed") from None
        finalized = True
        restored = _fingerprint(candidate)
        snapshot_after = _fingerprint(directory / "snapshot")
        immutable = snapshot_after == snapshot_before
        if restored != snapshot_before or not immutable:
            raise ChromaRestoreFailed("restore_verification_failed")
        return ChromaRestoreResult(
            backup_id=backup_id,
            target_type="isolated_non_production",
            file_count=restored["file_count"],
            total_bytes=restored["total_bytes"],
            aggregate_sha256=restored["aggregate_sha256"],
            manifest_match=(
                restored["aggregate_sha256"] == verification.aggregate_sha256
            ),
            immutable_backup_unchanged=immutable,
        )
    except ChromaBackupError:
        if staging.exists():
            _remove_owned_directory(
                staging, parent=approved, prefix=_RESTORE_STAGING_PREFIX
            )
        if finalized and candidate.exists():
            shutil.rmtree(candidate)
        raise
    except Exception:
        if staging.exists():
            _remove_owned_directory(
                staging, parent=approved, prefix=_RESTORE_STAGING_PREFIX
            )
        if finalized and candidate.exists():
            shutil.rmtree(candidate)
        raise ChromaRestoreFailed("restore_failed") from None


def run_restore_drill(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> ChromaRestoreResult:
    with tempfile.TemporaryDirectory(prefix="workagent-chroma-restore-drill-") as temporary:
        root = Path(temporary)
        result = restore_chroma_backup(
            backup_id,
            target=root / "restored",
            approved_target_root=root,
            backup_root=backup_root,
        )
        return ChromaRestoreResult(
            backup_id=result.backup_id,
            target_type="isolated_temporary_drill",
            file_count=result.file_count,
            total_bytes=result.total_bytes,
            aggregate_sha256=result.aggregate_sha256,
            manifest_match=result.manifest_match,
            immutable_backup_unchanged=result.immutable_backup_unchanged,
        )


def _allocate_loopback_port() -> int:
    for _ in range(20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind((LOOPBACK_HOST, 0))
                port = int(listener.getsockname()[1])
        except OSError:
            raise ChromaBackupConfigurationError("compatibility_port_unavailable") from None
        if port != EXISTING_LOCAL_HTTP_PORT:
            return port
    raise ChromaBackupConfigurationError("compatibility_port_unavailable")


def _inventory_change_count(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:
    left = {item["relative_path"]: (item["size_bytes"], item["sha256"]) for item in before["files"]}
    right = {item["relative_path"]: (item["size_bytes"], item["sha256"]) for item in after["files"]}
    return sum(left.get(path) != right.get(path) for path in set(left) | set(right))


def classify_chroma_compatibility_evidence(
    *,
    manifest: ChromaBackupManifest,
    current_version: str | None,
    server_started: bool,
    heartbeat: bool,
    lookups: Mapping[str, str],
    counts: Mapping[str, int | None],
    workspace_before: Mapping[str, Any] | None,
    workspace_after: Mapping[str, Any] | None,
    immutable: bool,
    restored_unchanged: bool,
    migration_required: bool,
    blocker: str | None,
) -> ChromaCompatibilityResult:
    required = tuple(item.semantic_id for item in list_registered_collections())
    accessible = all(lookups.get(item) == "accessible" for item in required)
    count_safe = all(isinstance(counts.get(item), int) for item in required)
    if migration_required:
        classification = "migration_required"
        blocker = blocker or "explicit_chroma_migration_required"
    elif current_version is None:
        classification = "unknown"
        blocker = blocker or "chroma_package_version_unavailable"
    elif server_started and heartbeat and accessible and count_safe:
        classification = "compatible"
        blocker = None
    elif server_started or blocker is not None:
        classification = "incompatible"
        blocker = blocker or "required_collection_compatibility_failed"
    else:
        classification = "unknown"
        blocker = blocker or "insufficient_compatibility_evidence"
    mutated = bool(
        workspace_before is not None
        and workspace_after is not None
        and workspace_before != workspace_after
    )
    return ChromaCompatibilityResult(
        backup_id=manifest.backup_id,
        capture_chroma_version=manifest.chroma_package_version,
        current_chroma_version=current_version,
        server_started=server_started,
        heartbeat_succeeded=heartbeat,
        collection_lookup=lookups,
        safe_counts=counts,
        workspace_fingerprint_before=(
            workspace_before["aggregate_sha256"] if workspace_before else None
        ),
        workspace_fingerprint_after=(
            workspace_after["aggregate_sha256"] if workspace_after else None
        ),
        server_open_mutated_internal_storage=mutated,
        changed_file_count=(
            _inventory_change_count(workspace_before, workspace_after)
            if workspace_before is not None and workspace_after is not None
            else 0
        ),
        immutable_backup_unchanged=immutable,
        restored_copy_unchanged=restored_unchanged,
        classification=classification,
        blocker=blocker,
    )


def run_chroma_backup_compatibility(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
    version_provider: VersionProvider = importlib.metadata.version,
    port_allocator: Callable[[], int] = _allocate_loopback_port,
    lifecycle_controller_factory: Callable[..., ChromaServerLifecycleController] = ChromaServerLifecycleController,
    explicit_migration_required: bool = False,
) -> ChromaCompatibilityResult:
    verify_chroma_backup(backup_id, backup_root=backup_root)
    _, directory = _backup_directory(backup_root, backup_id)
    manifest = _load_manifest(directory / "manifest.json")
    immutable_before = _fingerprint(directory / "snapshot")
    with tempfile.TemporaryDirectory(prefix="workagent-chroma-compatibility-") as temporary:
        root = Path(temporary)
        restored_path = root / "verified-restored"
        restore_chroma_backup(
            backup_id,
            target=restored_path,
            approved_target_root=root,
            backup_root=backup_root,
        )
        restored_before = _fingerprint(restored_path)
        information = root / "server-information"
        information.mkdir()
        workspace = information / "compatibility-data"
        workspace.mkdir()
        _copy_inventory(
            restored_path,
            workspace,
            restored_before,
            copier=shutil.copyfile,
        )
        workspace_before = _fingerprint(workspace)
        try:
            current_version = version_provider("chromadb")
            if not isinstance(current_version, str) or not current_version:
                current_version = None
        except Exception:
            current_version = None
        lookups = {item.semantic_id: "not_checked" for item in list_registered_collections()}
        counts: dict[str, int | None] = {
            item.semantic_id: None for item in list_registered_collections()
        }
        server_started = False
        heartbeat = False
        blocker: str | None = None
        controller: ChromaServerLifecycleController | None = None
        store: AtomicChromaServerStateStore | None = None
        factory: ChromaHttpClientFactory | None = None
        if current_version is not None:
            port = port_allocator()
            lifecycle_deployment = ChromaDeploymentConfig(
                ChromaDeploymentMode.LOCAL_HTTP,
                LOOPBACK_HOST,
                port,
                False,
                1.0,
            )
            lifecycle_config = build_chroma_server_lifecycle_config(
                lifecycle_deployment,
                information_root=information,
                persistence_path=workspace,
                runtime_state_directory=information / "runtime",
                startup_timeout_seconds=20.0,
                shutdown_timeout_seconds=5.0,
                endpoint_release_timeout_seconds=5.0,
                poll_interval_seconds=0.05,
                test_owned=True,
            )
            store = AtomicChromaServerStateStore(lifecycle_config)
            controller = lifecycle_controller_factory(lifecycle_config, state_store=store)
            try:
                started = controller.start()
                server_started = started.state == "ready" and started.process_owned
                heartbeat = server_started and started.server_reachable
                factory_deployment = ChromaDeploymentConfig(
                    ChromaDeploymentMode.EPHEMERAL_TEST,
                    LOOPBACK_HOST,
                    port,
                    False,
                    1.0,
                )
                factory = ChromaHttpClientFactory(
                    factory_deployment,
                    test_context=True,
                )
                for definition in list_registered_collections():
                    try:
                        handle = factory.get_collection_handle(
                            definition.semantic_id,
                            ChromaAccessLifecycle.TEST_ONLY,
                            "ephemeral_test_fixture",
                        )
                        lookups[definition.semantic_id] = "accessible"
                        counts[definition.semantic_id] = handle.safe_count()
                    except ChromaCollectionMissing:
                        lookups[definition.semantic_id] = "missing"
                    except ChromaTransportError:
                        lookups[definition.semantic_id] = "error"
                if any(value != "accessible" for value in lookups.values()):
                    blocker = "required_collection_unavailable"
            except Exception as error:
                blocker = getattr(error, "code", "compatibility_server_failed")
            finally:
                if factory is not None:
                    try:
                        factory.get_transport().close()
                    except Exception:
                        pass
                if controller is not None and store is not None:
                    try:
                        if store.load() is not None:
                            controller.stop()
                    except Exception:
                        blocker = blocker or "compatibility_server_shutdown_failed"
        workspace_after = _fingerprint(workspace)
        restored_after = _fingerprint(restored_path)
        immutable_after = _fingerprint(directory / "snapshot")
        return classify_chroma_compatibility_evidence(
            manifest=manifest,
            current_version=current_version,
            server_started=server_started,
            heartbeat=heartbeat,
            lookups=lookups,
            counts=counts,
            workspace_before=workspace_before,
            workspace_after=workspace_after,
            immutable=immutable_before == immutable_after,
            restored_unchanged=restored_before == restored_after,
            migration_required=explicit_migration_required,
            blocker=blocker,
        )


def run_chroma_recovery_gate(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> ChromaRecoveryGateResult:
    validate_backup_id(backup_id)
    try:
        verification = verify_chroma_backup(backup_id, backup_root=backup_root)
    except ChromaBackupError as error:
        return ChromaRecoveryGateResult(
            backup_id=backup_id,
            backup_verified=False,
            restore_drill_passed=False,
            compatibility="unknown",
            immutable_backup_unchanged=False,
            rollback_source_ready=False,
            production_cutover_recovery_gate="blocked",
            blocker=error.code,
        )
    try:
        restore = run_restore_drill(backup_id, backup_root=backup_root)
    except ChromaBackupError as error:
        return ChromaRecoveryGateResult(
            backup_id=backup_id,
            backup_verified=True,
            restore_drill_passed=False,
            compatibility="unknown",
            immutable_backup_unchanged=False,
            rollback_source_ready=False,
            production_cutover_recovery_gate="blocked",
            blocker=error.code,
        )
    try:
        compatibility = run_chroma_backup_compatibility(
            backup_id, backup_root=backup_root
        )
    except ChromaBackupError as error:
        return ChromaRecoveryGateResult(
            backup_id=backup_id,
            backup_verified=True,
            restore_drill_passed=True,
            compatibility="unknown",
            immutable_backup_unchanged=restore.immutable_backup_unchanged,
            rollback_source_ready=False,
            production_cutover_recovery_gate="blocked",
            blocker=error.code,
        )
    return build_recovery_gate_result(
        backup_id=backup_id,
        backup_verified=verification.verified,
        restore_drill_passed=(
            restore.manifest_match and restore.immutable_backup_unchanged
        ),
        compatibility=compatibility.classification,
        immutable_backup_unchanged=(
            restore.immutable_backup_unchanged
            and compatibility.immutable_backup_unchanged
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.chroma_backup_recovery",
        description="Capture and verify an offline Chroma recovery source.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capture")
    for command in ("verify", "restore-drill", "compatibility", "gate"):
        child = subparsers.add_parser(command)
        child.add_argument("--backup", required=True)
    return parser


def _print_fields(prefix: str, values: Mapping[str, Any]) -> None:
    rendered = " ".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in values.items()
    )
    print(f"{prefix} {rendered}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "capture":
            config = build_production_backup_configuration()
            preflight = preflight_backup_capture(config)
            _print_fields("backup preflight", preflight.safe_summary())
            _require_capture_preflight(preflight)
            manifest = capture_chroma_backup(config)
            _print_fields("backup capture", manifest.safe_summary())
            return 0
        backup_id = validate_backup_id(arguments.backup)
        if arguments.command == "verify":
            result: Any = verify_chroma_backup(backup_id)
        elif arguments.command == "restore-drill":
            result = run_restore_drill(backup_id)
        elif arguments.command == "compatibility":
            result = run_chroma_backup_compatibility(backup_id)
        else:
            result = run_chroma_recovery_gate(backup_id)
        _print_fields(f"backup {arguments.command}", result.safe_summary())
        if arguments.command == "compatibility":
            return 0 if result.classification == "compatible" else 1
        if arguments.command == "gate":
            return (
                0
                if result.production_cutover_recovery_gate == "recovery_ready"
                else 1
            )
        return 0
    except ChromaBackupError as error:
        print(
            f"backup {arguments.command} failed code={error.code}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
