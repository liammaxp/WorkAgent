from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend import chroma_backup_recovery as recovery
from backend.chroma_backup_models import (
    BACKUP_COMPATIBILITY_POLICY,
    CHROMA_BACKUP_MANIFEST_SCHEMA,
    ChromaBackupCaptureFailed,
    ChromaBackupManifest,
    ChromaBackupManifestInvalid,
    ChromaBackupPreconditionFailed,
    ChromaBackupVerificationFailed,
    ChromaRestoreFailed,
    ChromaRestoreTargetRejected,
    build_recovery_gate_result,
)
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_server_lifecycle_models import (
    ChromaServerLifecycleResult,
    build_chroma_server_lifecycle_config,
)


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ROOT / "information" / "chroma"
FIXED_NOW = datetime(2026, 8, 10, 4, 5, 6, tzinfo=timezone.utc)


def stopped_result() -> ChromaServerLifecycleResult:
    return ChromaServerLifecycleResult(
        state="not_running",
        deployment_mode="local_http",
        endpoint_scope="loopback",
        port=19441,
        process_owned=False,
        server_reachable=False,
        detail="runtime_state_absent",
    )


def lifecycle_result(
    state: str,
    *,
    process_owned: bool = False,
    reachable: bool = False,
) -> ChromaServerLifecycleResult:
    return ChromaServerLifecycleResult(
        state=state,
        deployment_mode="local_http",
        endpoint_scope="loopback",
        port=19441,
        process_owned=process_owned,
        server_reachable=reachable,
        detail="controlled_test_state",
    )


def backup_config(tmp_path: Path) -> recovery.ChromaBackupConfiguration:
    information = tmp_path / "information"
    information.mkdir()
    source = information / "chroma-test"
    source.mkdir()
    (source / "alpha.bin").write_bytes(b"alpha-bytes")
    nested = source / "segments"
    nested.mkdir()
    (nested / "beta.bin").write_bytes(b"beta-bytes" * 3)
    deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        "127.0.0.1",
        19441,
        False,
        0.5,
    )
    lifecycle = build_chroma_server_lifecycle_config(
        deployment,
        information_root=information,
        persistence_path=source,
        runtime_state_directory=information / "runtime",
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        endpoint_release_timeout_seconds=1.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )
    return recovery.build_test_backup_configuration(
        source_root=source,
        backup_root=tmp_path / "backups",
        test_root=tmp_path,
        lifecycle_config=lifecycle,
    )


def capture_fixture(
    tmp_path: Path,
) -> tuple[recovery.ChromaBackupConfiguration, ChromaBackupManifest]:
    config = backup_config(tmp_path)
    manifest = recovery.capture_chroma_backup(
        config,
        test_lifecycle_observer=lambda _config: stopped_result(),
        now_provider=lambda: FIXED_NOW,
        version_provider=lambda _name: "1.5.9",
    )
    return config, manifest


def manifest_path(config: recovery.ChromaBackupConfiguration, backup_id: str) -> Path:
    return config.backup_root / backup_id / "manifest.json"


def snapshot_path(config: recovery.ChromaBackupConfiguration, backup_id: str) -> Path:
    return config.backup_root / backup_id / "snapshot"


def rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_production_preflight_is_bounded_and_enforces_accepted_baseline(
    monkeypatch,
):
    accepted = {
        "file_count": 1,
        "total_bytes": 3,
        "aggregate_sha256": "a" * 64,
        "files": [
            {
                "relative_path": "chroma.sqlite3",
                "size_bytes": 3,
                "sha256": "b" * 64,
            }
        ],
    }
    monkeypatch.setattr(recovery, "_fingerprint", lambda _root: dict(accepted))
    monkeypatch.setattr(
        recovery,
        "_expected_production_fingerprint",
        lambda: dict(accepted),
    )
    monkeypatch.setattr(
        recovery,
        "_default_lifecycle_observer",
        lambda _config: stopped_result(),
    )
    config = recovery.build_production_backup_configuration()
    runtime_state_path = config.lifecycle_config.runtime_state_path
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path == runtime_state_path else original_exists(path),
    )

    result = recovery.preflight_backup_capture(config)

    assert result.safe_summary() == {
        "schema": "chroma_backup_status.v1",
        "source_state": "verified",
        "server_state": "not_running",
        "endpoint_state": "free",
        "runtime_state_residue": False,
        "source_fingerprint": "a" * 64,
        "capture_allowed": True,
        "blocker": None,
    }


def test_verified_post_cutover_inventory_is_exact_and_does_not_change_default(
    monkeypatch,
):
    observed = {
        "file_count": 1,
        "total_bytes": 3,
        "aggregate_sha256": "a" * 64,
        "files": [
            {
                "relative_path": "chroma.sqlite3",
                "size_bytes": 3,
                "sha256": "b" * 64,
            }
        ],
    }
    accepted = {**observed, "aggregate_sha256": "c" * 64}
    monkeypatch.setattr(recovery, "_fingerprint", lambda _root: dict(observed))
    monkeypatch.setattr(
        recovery,
        "_expected_production_fingerprint",
        lambda: dict(accepted),
    )
    monkeypatch.setattr(
        recovery,
        "_default_lifecycle_observer",
        lambda _config: stopped_result(),
    )
    config = recovery.build_production_backup_configuration()
    runtime_state_path = config.lifecycle_config.runtime_state_path
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path == runtime_state_path else original_exists(path),
    )

    default = recovery.preflight_backup_capture(config)
    assert default.capture_allowed is False
    assert default.blocker == "protected_backup_source_drift"

    authorized = recovery.preflight_backup_capture(
        config,
        verified_post_cutover_inventory=observed,
    )
    assert authorized.capture_allowed is True
    assert authorized.source_fingerprint == "a" * 64

    mismatched = recovery.preflight_backup_capture(
        config,
        verified_post_cutover_inventory={**observed, "total_bytes": 4},
    )
    assert mismatched.capture_allowed is False
    assert mismatched.blocker == "protected_backup_source_drift"


def test_verified_post_cutover_inventory_rejects_invalid_and_test_owned_use(
    tmp_path,
):
    with pytest.raises(
        recovery.ChromaBackupConfigurationError,
        match="invalid_verified_post_cutover_inventory",
    ):
        recovery.preflight_backup_capture(
            recovery.build_production_backup_configuration(),
            verified_post_cutover_inventory=object(),
        )

    with pytest.raises(
        recovery.ChromaBackupConfigurationError,
        match="verified_post_cutover_inventory_requires_production_backup",
    ):
        recovery.preflight_backup_capture(
            backup_config(tmp_path),
            verified_post_cutover_inventory={},
        )


@pytest.mark.parametrize(
    "observed",
    [
        lifecycle_result("ready", process_owned=True, reachable=True),
        lifecycle_result("starting", process_owned=True, reachable=True),
        lifecycle_result("foreign_port_conflict"),
        lifecycle_result("ownership_mismatch"),
        lifecycle_result("stale_state"),
        lifecycle_result("unhealthy", process_owned=True),
    ],
)
def test_running_foreign_stale_and_ambiguous_states_block_capture(tmp_path, observed):
    config = backup_config(tmp_path)
    result = recovery.preflight_backup_capture(
        config, test_lifecycle_observer=lambda _config: observed
    )
    assert result.capture_allowed is False
    assert result.blocker == "production_server_not_verified_stopped"


def test_invalid_lifecycle_observation_fails_closed(tmp_path):
    config = backup_config(tmp_path)
    result = recovery.preflight_backup_capture(
        config, test_lifecycle_observer=lambda _config: object()
    )
    assert result.server_state == "ambiguous"
    assert result.capture_allowed is False


def test_runtime_state_residue_blocks_capture_even_with_stopped_result(tmp_path):
    config = backup_config(tmp_path)
    config.lifecycle_config.runtime_state_directory.mkdir(parents=True)
    config.lifecycle_config.runtime_state_path.write_text("{}", encoding="utf-8")
    result = recovery.preflight_backup_capture(
        config, test_lifecycle_observer=lambda _config: stopped_result()
    )
    assert result.runtime_state_residue is True
    assert result.blocker == "runtime_state_residue_present"


def test_missing_source_is_rejected(tmp_path):
    config = backup_config(tmp_path)
    shutil.rmtree(config.source_root)
    result = recovery.preflight_backup_capture(
        config, test_lifecycle_observer=lambda _config: stopped_result()
    )
    assert result.source_state == "invalid"
    assert result.capture_allowed is False


def test_production_lifecycle_observer_cannot_be_injected():
    config = recovery.build_production_backup_configuration()
    with pytest.raises(
        recovery.ChromaBackupConfigurationError,
        match="lifecycle_observer_injection_requires_test_backup",
    ):
        recovery.preflight_backup_capture(
            config, test_lifecycle_observer=lambda _config: stopped_result()
        )


def test_test_configuration_rejects_production_or_escaping_paths(tmp_path):
    config = backup_config(tmp_path)
    with pytest.raises(recovery.ChromaBackupConfigurationError):
        recovery.build_test_backup_configuration(
            source_root=PROTECTED,
            backup_root=config.backup_root,
            test_root=tmp_path,
            lifecycle_config=config.lifecycle_config,
        )
    with pytest.raises(recovery.ChromaBackupConfigurationError):
        recovery.build_test_backup_configuration(
            source_root=config.source_root,
            backup_root=tmp_path.parent / "escaped-backup",
            test_root=tmp_path,
            lifecycle_config=config.lifecycle_config,
        )


def test_capture_is_filesystem_only_and_does_not_construct_chroma(monkeypatch, tmp_path):
    config = backup_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("capture must not construct or start Chroma")

    monkeypatch.setattr(recovery, "ChromaHttpClientFactory", forbidden)
    monkeypatch.setattr(recovery, "ChromaServerLifecycleController", forbidden)
    manifest = recovery.capture_chroma_backup(
        config,
        test_lifecycle_observer=lambda _config: stopped_result(),
        now_provider=lambda: FIXED_NOW,
        version_provider=lambda _name: "1.5.9",
    )
    assert manifest.capture_state == "verified"


def test_file_inventory_is_deterministic_relative_and_drive_independent(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "z.bin").write_bytes(b"z")
        (root / "nested" / "a.bin").write_bytes(b"alpha")
    first = capture_protected_file_inventory(left)
    second = capture_protected_file_inventory(right)
    assert first == second
    assert [item["relative_path"] for item in first["files"]] == [
        "nested/a.bin",
        "z.bin",
    ]
    assert first["file_count"] == 2
    assert first["total_bytes"] == 6
    assert all(len(item["sha256"]) == 64 for item in first["files"])


def test_capture_writes_verified_snapshot_and_strict_manifest(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    source = capture_protected_file_inventory(config.source_root)
    snapshot = capture_protected_file_inventory(snapshot_path(config, manifest.backup_id))
    payload = json.loads(manifest_path(config, manifest.backup_id).read_text(encoding="utf-8"))
    assert source == snapshot
    assert manifest.schema == CHROMA_BACKUP_MANIFEST_SCHEMA
    assert manifest.backup_id.startswith("20260810T040506Z-")
    assert manifest.source_relative_path == "test-owned/chroma"
    assert manifest.chroma_package_version == "1.5.9"
    assert manifest.compatibility_policy == BACKUP_COMPATIBILITY_POLICY
    assert manifest.capture_state == "verified"
    assert set(payload) == set(manifest.to_dict())
    assert not any(Path(item["relative_path"]).is_absolute() for item in payload["files"])
    serialized = json.dumps(payload).casefold()
    for forbidden in ("documents", "embeddings", "environment", "password"):
        assert forbidden not in serialized


def test_capture_rejects_existing_backup_id_without_overwrite(tmp_path):
    config, first = capture_fixture(tmp_path)
    before = capture_protected_file_inventory(snapshot_path(config, first.backup_id))
    with pytest.raises(ChromaBackupCaptureFailed, match="backup_id_already_exists"):
        recovery.capture_chroma_backup(
            config,
            test_lifecycle_observer=lambda _config: stopped_result(),
            now_provider=lambda: FIXED_NOW,
            version_provider=lambda _name: "1.5.9",
        )
    assert capture_protected_file_inventory(snapshot_path(config, first.backup_id)) == before


def test_source_change_during_copy_fails_and_exposes_no_backup(tmp_path):
    config = backup_config(tmp_path)
    calls = 0

    def mutating_copy(source, destination):
        nonlocal calls
        shutil.copyfile(source, destination)
        calls += 1
        if calls == 1:
            (config.source_root / "alpha.bin").write_bytes(b"changed-after-copy")

    with pytest.raises(ChromaBackupCaptureFailed):
        recovery.capture_chroma_backup(
            config,
            test_lifecycle_observer=lambda _config: stopped_result(),
            now_provider=lambda: FIXED_NOW,
            version_provider=lambda _name: "1.5.9",
            copier=mutating_copy,
        )
    assert tuple(config.backup_root.iterdir()) == ()


def test_server_start_during_copy_prevents_verified_finalize(tmp_path):
    config = backup_config(tmp_path)
    observations = iter(
        [stopped_result(), lifecycle_result("ready", process_owned=True, reachable=True)]
    )

    with pytest.raises(ChromaBackupPreconditionFailed):
        recovery.capture_chroma_backup(
            config,
            test_lifecycle_observer=lambda _config: next(observations),
            now_provider=lambda: FIXED_NOW,
            version_provider=lambda _name: "1.5.9",
        )
    assert tuple(config.backup_root.iterdir()) == ()


def test_partial_copy_manifest_failure_and_finalize_failure_leave_no_verified_backup(
    tmp_path,
):
    for failure_kind in ("copy", "manifest", "finalize"):
        case = tmp_path / failure_kind
        case.mkdir()
        config = backup_config(case)

        def failing_copy(_source, _destination):
            raise OSError("controlled")

        def failing_manifest(_path, _manifest):
            raise OSError("controlled")

        def failing_finalize(_source, _destination):
            raise OSError("controlled")

        kwargs = {
            "copier": failing_copy if failure_kind == "copy" else shutil.copyfile,
            "manifest_writer": (
                failing_manifest
                if failure_kind == "manifest"
                else recovery._write_manifest_atomic
            ),
            "finalizer": failing_finalize if failure_kind == "finalize" else os.replace,
        }
        with pytest.raises(ChromaBackupCaptureFailed):
            recovery.capture_chroma_backup(
                config,
                test_lifecycle_observer=lambda _config: stopped_result(),
                now_provider=lambda: FIXED_NOW,
                version_provider=lambda _name: "1.5.9",
                **kwargs,
            )
        assert tuple(config.backup_root.iterdir()) == ()


def test_package_version_unavailable_blocks_capture_before_verified_backup(tmp_path):
    config = backup_config(tmp_path)

    def unavailable(_name):
        raise LookupError

    with pytest.raises(ChromaBackupCaptureFailed, match="version_unavailable"):
        recovery.capture_chroma_backup(
            config,
            test_lifecycle_observer=lambda _config: stopped_result(),
            now_provider=lambda: FIXED_NOW,
            version_provider=unavailable,
        )
    assert not config.backup_root.exists()


def test_untouched_backup_verifies_without_modifying_snapshot(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    before = capture_protected_file_inventory(snapshot_path(config, manifest.backup_id))
    result = recovery.verify_chroma_backup(
        manifest.backup_id, backup_root=config.backup_root
    )
    after = capture_protected_file_inventory(snapshot_path(config, manifest.backup_id))
    assert result.verified is True
    assert result.aggregate_sha256 == manifest.aggregate_sha256
    assert before == after


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "truncated"])
def test_snapshot_corruption_is_detected(tmp_path, mutation):
    config, manifest = capture_fixture(tmp_path)
    snapshot = snapshot_path(config, manifest.backup_id)
    first = next(path for path in snapshot.rglob("*") if path.is_file())
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        (snapshot / "extra.bin").write_bytes(b"extra")
    elif mutation == "changed":
        first.write_bytes(b"changed-same-or-different-size")
    else:
        first.write_bytes(first.read_bytes()[:1])
    with pytest.raises(ChromaBackupVerificationFailed):
        recovery.verify_chroma_backup(manifest.backup_id, backup_root=config.backup_root)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "duplicate",
        "dotdot",
        "absolute",
        "hash",
        "size",
        "aggregate",
        "state",
        "unknown",
        "policy",
    ],
)
def test_manifest_corruption_and_unsafe_paths_are_rejected(tmp_path, mutation):
    config, manifest = capture_fixture(tmp_path)
    path = manifest_path(config, manifest.backup_id)

    def mutate(payload):
        if mutation == "schema":
            payload["schema"] = "chroma_backup_manifest.v999"
        elif mutation == "duplicate":
            duplicate = copy.deepcopy(payload["files"][0])
            payload["files"].insert(1, duplicate)
            payload["file_count"] += 1
            payload["total_bytes"] += duplicate["size_bytes"]
        elif mutation == "dotdot":
            payload["files"][0]["relative_path"] = "../escape.bin"
        elif mutation == "absolute":
            payload["files"][0]["relative_path"] = "C:/private/data.bin"
        elif mutation == "hash":
            payload["files"][0]["sha256"] = "g" * 64
        elif mutation == "size":
            payload["files"][0]["size_bytes"] += 1
        elif mutation == "aggregate":
            payload["aggregate_sha256"] = "0" * 64
        elif mutation == "state":
            payload["capture_state"] = "staging"
        elif mutation == "unknown":
            payload["documents"] = ["forbidden"]
        else:
            payload["compatibility_policy"] = "string_equality_only"

    rewrite_manifest(path, mutate)
    with pytest.raises((ChromaBackupManifestInvalid, ChromaBackupVerificationFailed)):
        recovery.verify_chroma_backup(manifest.backup_id, backup_root=config.backup_root)


def test_corrupt_json_and_extra_backup_layout_entry_fail_verification(tmp_path):
    first_case = tmp_path / "json"
    first_case.mkdir()
    config, manifest = capture_fixture(first_case)
    manifest_path(config, manifest.backup_id).write_text("{", encoding="utf-8")
    with pytest.raises(ChromaBackupManifestInvalid):
        recovery.verify_chroma_backup(manifest.backup_id, backup_root=config.backup_root)

    second_case = tmp_path / "layout"
    second_case.mkdir()
    config, manifest = capture_fixture(second_case)
    (config.backup_root / manifest.backup_id / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ChromaBackupVerificationFailed, match="invalid_chroma_backup_layout"):
        recovery.verify_chroma_backup(manifest.backup_id, backup_root=config.backup_root)


def test_verified_backup_restores_exactly_to_absent_isolated_target(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    restore_root = tmp_path / "restore-root"
    restore_root.mkdir()
    target = restore_root / "restored"
    snapshot_before = capture_protected_file_inventory(snapshot_path(config, manifest.backup_id))
    result = recovery.restore_chroma_backup(
        manifest.backup_id,
        target=target,
        approved_target_root=restore_root,
        backup_root=config.backup_root,
    )
    assert result.manifest_match is True
    assert result.immutable_backup_unchanged is True
    assert capture_protected_file_inventory(target) == snapshot_before
    assert capture_protected_file_inventory(snapshot_path(config, manifest.backup_id)) == snapshot_before


def test_restore_drill_is_temporary_exact_and_preserves_backup(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    result = recovery.run_restore_drill(
        manifest.backup_id, backup_root=config.backup_root
    )
    assert result.target_type == "isolated_temporary_drill"
    assert result.file_count == manifest.file_count
    assert result.total_bytes == manifest.total_bytes
    assert result.aggregate_sha256 == manifest.aggregate_sha256
    assert result.manifest_match is result.immutable_backup_unchanged is True


@pytest.mark.parametrize(
    "target_factory",
    [
        lambda root: PROTECTED,
        lambda root: ROOT / "." / "information" / "chroma",
        lambda root: root,
        lambda root: root / ".." / "escaped",
    ],
)
def test_restore_rejects_protected_alias_root_and_traversal_targets(
    tmp_path, target_factory
):
    config, manifest = capture_fixture(tmp_path)
    restore_root = tmp_path / "restore-root"
    restore_root.mkdir()
    target = target_factory(restore_root)
    with pytest.raises(ChromaRestoreTargetRejected):
        recovery.restore_chroma_backup(
            manifest.backup_id,
            target=target,
            approved_target_root=restore_root if target != PROTECTED else ROOT / "information",
            backup_root=config.backup_root,
        )


def test_restore_rejects_existing_nonempty_target(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    restore_root = tmp_path / "restore-root"
    target = restore_root / "existing"
    target.mkdir(parents=True)
    (target / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(ChromaRestoreTargetRejected, match="restore_target_must_be_absent"):
        recovery.restore_chroma_backup(
            manifest.backup_id,
            target=target,
            approved_target_root=restore_root,
            backup_root=config.backup_root,
        )
    assert (target / "sentinel").read_text(encoding="utf-8") == "preserve"


def test_restore_copy_and_finalize_failures_leave_no_target_or_staging(tmp_path):
    for failure in ("copy", "finalize"):
        case = tmp_path / failure
        case.mkdir()
        config, manifest = capture_fixture(case)
        restore_root = case / "restore"
        restore_root.mkdir()
        target = restore_root / "target"

        def failing(*_args):
            raise OSError("controlled")

        with pytest.raises((ChromaBackupCaptureFailed, ChromaRestoreFailed)):
            recovery.restore_chroma_backup(
                manifest.backup_id,
                target=target,
                approved_target_root=restore_root,
                backup_root=config.backup_root,
                copier=failing if failure == "copy" else shutil.copyfile,
                finalizer=failing if failure == "finalize" else os.replace,
            )
        assert not target.exists()
        assert tuple(restore_root.iterdir()) == ()


def test_restore_rejects_symlink_target_without_touching_destination(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    restore_root = tmp_path / "restore-root"
    restore_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = restore_root / "linked-target"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ChromaRestoreTargetRejected):
        recovery.restore_chroma_backup(
            manifest.backup_id,
            target=link,
            approved_target_root=restore_root,
            backup_root=config.backup_root,
        )
    assert tuple(outside.iterdir()) == ()


def test_version_unavailable_yields_unknown_without_starting_server(tmp_path, monkeypatch):
    config, manifest = capture_fixture(tmp_path)

    def unavailable(_name):
        raise LookupError

    def forbidden(*_args, **_kwargs):
        raise AssertionError("server must not start without version evidence")

    monkeypatch.setattr(recovery, "ChromaServerLifecycleController", forbidden)
    result = recovery.run_chroma_backup_compatibility(
        manifest.backup_id,
        backup_root=config.backup_root,
        version_provider=unavailable,
    )
    assert result.classification == "unknown"
    assert result.server_started is result.heartbeat_succeeded is False
    assert result.immutable_backup_unchanged is result.restored_copy_unchanged is True


def test_server_open_failure_is_incompatible_not_version_string_compatible(tmp_path):
    config, manifest = capture_fixture(tmp_path)

    class FailingController:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("controlled")

    result = recovery.run_chroma_backup_compatibility(
        manifest.backup_id,
        backup_root=config.backup_root,
        version_provider=lambda _name: manifest.chroma_package_version,
        lifecycle_controller_factory=FailingController,
    )
    assert result.classification == "incompatible"
    assert result.server_started is False


def test_different_recorded_version_can_be_compatible_only_with_executable_evidence(tmp_path):
    _config, manifest = capture_fixture(tmp_path)
    inventory = {
        "files": [item.to_dict() for item in manifest.files],
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "aggregate_sha256": manifest.aggregate_sha256,
    }
    result = recovery.classify_chroma_compatibility_evidence(
        manifest=manifest,
        current_version="9.9.9",
        server_started=True,
        heartbeat=True,
        lookups={"github_evidence": "accessible", "profile_facts": "accessible"},
        counts={"github_evidence": 0, "profile_facts": 2},
        workspace_before=inventory,
        workspace_after=inventory,
        immutable=True,
        restored_unchanged=True,
        migration_required=False,
        blocker=None,
    )
    assert result.classification == "compatible"
    assert result.capture_chroma_version == "1.5.9"
    assert result.current_chroma_version == "9.9.9"


def test_explicit_supported_migration_evidence_classifies_migration_required(tmp_path):
    _config, manifest = capture_fixture(tmp_path)
    result = recovery.classify_chroma_compatibility_evidence(
        manifest=manifest,
        current_version="1.5.9",
        server_started=False,
        heartbeat=False,
        lookups={},
        counts={},
        workspace_before=None,
        workspace_after=None,
        immutable=True,
        restored_unchanged=True,
        migration_required=True,
        blocker=None,
    )
    assert result.classification == "migration_required"


@pytest.mark.parametrize(
    "verified,restored,compatibility,immutable,ready",
    [
        (True, True, "compatible", True, True),
        (False, True, "compatible", True, False),
        (True, False, "compatible", True, False),
        (True, True, "unknown", True, False),
        (True, True, "migration_required", True, False),
        (True, True, "incompatible", True, False),
        (True, True, "compatible", False, False),
    ],
)
def test_recovery_gate_requires_verified_restore_compatible_and_immutable(
    verified, restored, compatibility, immutable, ready
):
    result = build_recovery_gate_result(
        backup_id="20260810T040506Z-0123456789ab",
        backup_verified=verified,
        restore_drill_passed=restored,
        compatibility=compatibility,
        immutable_backup_unchanged=immutable,
    )
    assert result.rollback_source_ready is ready
    assert result.production_cutover_recovery_gate == (
        "recovery_ready" if ready else "blocked"
    )


def test_combined_gate_reports_exact_verification_blocker(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    first = next(
        path
        for path in snapshot_path(config, manifest.backup_id).rglob("*")
        if path.is_file()
    )
    first.write_bytes(b"corrupt")
    result = recovery.run_chroma_recovery_gate(
        manifest.backup_id, backup_root=config.backup_root
    )
    assert result.production_cutover_recovery_gate == "blocked"
    assert result.rollback_source_ready is False
    assert result.blocker == "chroma_backup_snapshot_mismatch"


def test_backup_root_is_git_ignored_and_no_real_backup_is_tracked():
    result = subprocess.run(
        ["git", "check-ignore", "information/backups/chroma/probe.snapshot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "information/backups/chroma"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""


def test_backup_sources_have_semantic_names_safe_output_and_no_embedded_client():
    backend_paths = [
        ROOT / "backend" / "chroma_backup_models.py",
        ROOT / "backend" / "chroma_backup_recovery.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in backend_paths)
    lowered = combined.casefold()
    assert "persistent" + "client(" not in lowered
    assert "chromadb" + "." not in lowered
    assert "frontend/" not in lowered
    assert "production in-place" + " restore" not in lowered
    assert "phase" + "6" not in lowered


def test_safe_summaries_do_not_expose_absolute_paths_or_application_bodies(tmp_path):
    config, manifest = capture_fixture(tmp_path)
    summaries = [
        config.safe_summary(),
        manifest.safe_summary(),
        recovery.verify_chroma_backup(
            manifest.backup_id, backup_root=config.backup_root
        ).safe_summary(),
    ]
    serialized = json.dumps(summaries)
    assert str(tmp_path) not in serialized
    assert str(PROTECTED) not in serialized
    assert "alpha-bytes" not in serialized
    assert "documents" not in serialized.casefold()
    assert "embeddings" not in serialized.casefold()
