from __future__ import annotations

import shutil
import time
from pathlib import Path

import psutil
import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_server_lifecycle import ChromaServerLifecycleController
from backend.chroma_server_lifecycle_models import (
    AtomicChromaServerStateStore,
    build_chroma_server_lifecycle_config,
)
from tests.chroma_http_test_support import (
    PROTECTED_CHROMA_ROOT,
    allocate_dynamic_loopback_endpoint,
    is_loopback_port_releasable,
    wait_for_loopback_port_release,
)


def _fingerprint() -> dict[str, int | str]:
    inventory = capture_protected_file_inventory(PROTECTED_CHROMA_ROOT)
    return {
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }


def _wait_until_missing(pid: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.025)
    return not psutil.pid_exists(pid)


@pytest.mark.chroma_server_integration
def test_real_temporary_server_start_health_restart_crash_recovery_and_stop(
    tmp_path: Path,
):
    assert shutil.which("chroma")
    protected_before = _fingerprint()

    endpoint = allocate_dynamic_loopback_endpoint()
    information = tmp_path / "information"
    persistence = information / "chroma-test"
    runtime = information / "runtime" / "chroma"
    information.mkdir()
    sentinel = persistence / "operator-sentinel.txt"
    deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        endpoint.host,
        endpoint.port,
        False,
        1.0,
    )
    config = build_chroma_server_lifecycle_config(
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
    store = AtomicChromaServerStateStore(config)
    controller = ChromaServerLifecycleController(config, state_store=store)
    observed_pids: set[int] = set()

    try:
        started = controller.start()
        first_state = store.load()
        assert started.state == "ready"
        assert started.process_owned is True
        assert started.server_reachable is True
        assert first_state is not None and first_state.lifecycle_state == "ready"
        observed_pids.add(first_state.pid)
        sentinel.write_text("preserve-across-lifecycle-events", encoding="utf-8")

        healthy = controller.health()
        assert healthy.state == "ready"
        assert healthy.process_owned is True
        assert healthy.server_reachable is True

        restarted = controller.restart()
        second_state = store.load()
        assert restarted.state == "ready"
        assert restarted.detail == "restarted"
        assert second_state is not None
        observed_pids.add(second_state.pid)
        assert (
            second_state.pid,
            second_state.process_start_token,
        ) != (
            first_state.pid,
            first_state.process_start_token,
        )
        assert _wait_until_missing(first_state.pid)
        assert sentinel.read_text(encoding="utf-8") == "preserve-across-lifecycle-events"

        psutil.Process(second_state.pid).kill()
        assert _wait_until_missing(second_state.pid)
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)
        stale = controller.health()
        assert stale.state == "stale_state"
        assert stale.process_owned is False

        recovered = controller.start()
        recovered_state = store.load()
        assert recovered.state == "ready"
        assert recovered_state is not None
        observed_pids.add(recovered_state.pid)
        assert recovered_state.pid != second_state.pid
        assert sentinel.read_text(encoding="utf-8") == "preserve-across-lifecycle-events"

        stopped = controller.stop()
        assert stopped.state == "not_running"
        assert stopped.process_owned is True
        assert store.load() is None
        assert is_loopback_port_releasable(endpoint.port)
    finally:
        state = store.load()
        if state is not None:
            try:
                controller.stop()
            except Exception:
                process = psutil.Process(state.pid) if psutil.pid_exists(state.pid) else None
                if process is not None:
                    process.kill()
                    process.wait(timeout=5.0)
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)
        assert all(not psutil.pid_exists(pid) for pid in observed_pids)

    renamed = information / "chroma-test-released"
    persistence.rename(renamed)
    renamed.rename(persistence)
    assert _fingerprint() == protected_before
