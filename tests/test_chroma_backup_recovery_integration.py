from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pytest

from backend.chroma_backup_models import build_recovery_gate_result
from backend.chroma_backup_recovery import (
    build_test_backup_configuration,
    capture_chroma_backup,
    restore_chroma_backup,
    run_chroma_backup_compatibility,
    verify_chroma_backup,
)
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_server_lifecycle import ChromaServerLifecycleController
from backend.chroma_server_lifecycle_models import (
    AtomicChromaServerStateStore,
    build_chroma_server_lifecycle_config,
)
from tests.chroma_http_test_support import (
    PROTECTED_CHROMA_ROOT,
    EphemeralChromaEndpoint,
    allocate_dynamic_loopback_endpoint,
    is_loopback_port_releasable,
    prepare_registered_collection_for_test,
    wait_for_loopback_port_release,
)


def protected_fingerprint() -> dict[str, int | str]:
    inventory = capture_protected_file_inventory(PROTECTED_CHROMA_ROOT)
    return {
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }


@pytest.mark.chroma_server_integration
def test_real_backup_restore_and_executable_compatibility_use_only_disposable_storage(
    tmp_path: Path,
):
    protected_before = protected_fingerprint()

    endpoint = allocate_dynamic_loopback_endpoint()
    information = tmp_path / "source-information"
    information.mkdir()
    persistence = information / "chroma-test"
    runtime = information / "runtime"
    deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        endpoint.host,
        endpoint.port,
        False,
        1.0,
    )
    lifecycle = build_chroma_server_lifecycle_config(
        deployment,
        information_root=information,
        persistence_path=persistence,
        runtime_state_directory=runtime,
        startup_timeout_seconds=20.0,
        shutdown_timeout_seconds=5.0,
        endpoint_release_timeout_seconds=5.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )
    store = AtomicChromaServerStateStore(lifecycle)
    controller = ChromaServerLifecycleController(lifecycle, state_store=store)
    source_pid: int | None = None
    try:
        started = controller.start()
        state = store.load()
        assert started.state == "ready"
        assert state is not None
        source_pid = state.pid
        fixture_endpoint = EphemeralChromaEndpoint(endpoint.host, endpoint.port)
        prepare_registered_collection_for_test(
            fixture_endpoint,
            "github_evidence",
            ids=["github-test-record"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"project_id": "test/project"}],
        )
        prepare_registered_collection_for_test(
            fixture_endpoint,
            "profile_facts",
            ids=["profile-test-record"],
            embeddings=[[0.3, 0.2, 0.1]],
            metadatas=[{"index": "0"}],
        )
        stopped = controller.stop()
        assert stopped.state == "not_running"
        assert store.load() is None
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)

        source_before = capture_protected_file_inventory(persistence)
        config = build_test_backup_configuration(
            source_root=persistence,
            backup_root=tmp_path / "backups",
            test_root=tmp_path,
            lifecycle_config=lifecycle,
        )
        manifest = capture_chroma_backup(
            config,
            now_provider=lambda: datetime(
                2026, 8, 10, 5, 6, 7, tzinfo=timezone.utc
            ),
        )
        verification = verify_chroma_backup(
            manifest.backup_id, backup_root=config.backup_root
        )
        immutable_before = capture_protected_file_inventory(
            config.backup_root / manifest.backup_id / "snapshot"
        )

        restore_root = tmp_path / "restore-root"
        restore_root.mkdir()
        restored_path = restore_root / "verified-restored"
        restored = restore_chroma_backup(
            manifest.backup_id,
            target=restored_path,
            approved_target_root=restore_root,
            backup_root=config.backup_root,
        )
        assert capture_protected_file_inventory(restored_path) == immutable_before

        compatibility = run_chroma_backup_compatibility(
            manifest.backup_id,
            backup_root=config.backup_root,
        )
        immutable_after = capture_protected_file_inventory(
            config.backup_root / manifest.backup_id / "snapshot"
        )
        source_after = capture_protected_file_inventory(persistence)

        assert verification.verified is True
        assert restored.manifest_match is restored.immutable_backup_unchanged is True
        assert manifest.chroma_package_version == importlib.metadata.version("chromadb")
        assert compatibility.current_chroma_version == importlib.metadata.version("chromadb")
        assert compatibility.server_started is True
        assert compatibility.heartbeat_succeeded is True
        assert dict(compatibility.collection_lookup) == {
            "github_evidence": "accessible",
            "profile_facts": "accessible",
        }
        assert dict(compatibility.safe_counts) == {
            "github_evidence": 1,
            "profile_facts": 1,
        }
        assert compatibility.workspace_fingerprint_before is not None
        assert compatibility.workspace_fingerprint_after is not None
        assert compatibility.server_open_mutated_internal_storage is (
            compatibility.workspace_fingerprint_before
            != compatibility.workspace_fingerprint_after
        )
        assert compatibility.changed_file_count >= 0
        assert compatibility.immutable_backup_unchanged is True
        assert compatibility.restored_copy_unchanged is True
        assert compatibility.classification == "compatible"
        assert immutable_before == immutable_after
        assert source_before == source_after

        gate = build_recovery_gate_result(
            backup_id=manifest.backup_id,
            backup_verified=verification.verified,
            restore_drill_passed=(
                restored.manifest_match and restored.immutable_backup_unchanged
            ),
            compatibility=compatibility.classification,
            immutable_backup_unchanged=compatibility.immutable_backup_unchanged,
        )
        assert gate.rollback_source_ready is True
        assert gate.production_cutover_recovery_gate == "recovery_ready"
    finally:
        state = store.load()
        if state is not None:
            try:
                controller.stop()
            except Exception:
                if psutil.pid_exists(state.pid):
                    process = psutil.Process(state.pid)
                    process.kill()
                    process.wait(timeout=5.0)
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)
        if source_pid is not None:
            assert not psutil.pid_exists(source_pid)
        assert is_loopback_port_releasable(endpoint.port)
        assert protected_fingerprint() == protected_before
