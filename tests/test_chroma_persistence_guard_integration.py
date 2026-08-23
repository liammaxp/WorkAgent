from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_persistence_guard import ChromaPersistenceGuard
from backend.chroma_persistence_guard_models import (
    ChromaPersistenceContext,
    ChromaPersistenceRole,
    ChromaProtectedPathAccessDenied,
)
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


ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENVIRONMENT_KEYS = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
)


def _subprocess_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _SUBPROCESS_ENVIRONMENT_KEYS and isinstance(value, str)
    }
    environment["ANONYMIZED_TELEMETRY"] = "FALSE"
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    return environment


def _run_embedded_probe(config) -> subprocess.CompletedProcess[str]:
    creation_flags = (
        subprocess.CREATE_NO_WINDOW
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
        else 0
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.chroma_persistence_test_support",
            str(config.information_root),
            str(config.persistence_path),
            str(config.runtime_state_directory),
            str(config.deployment.port),
        ],
        cwd=ROOT,
        env=_subprocess_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        timeout=30.0,
        creationflags=creation_flags,
        check=False,
    )


@pytest.mark.chroma_server_integration
def test_real_server_exclusively_owns_temp_persistence_until_verified_stop(
    tmp_path: Path,
):
    protected_before = capture_protected_file_inventory(PROTECTED_CHROMA_ROOT)
    endpoint = allocate_dynamic_loopback_endpoint()
    information = tmp_path / "information"
    information.mkdir()
    config = build_chroma_server_lifecycle_config(
        ChromaDeploymentConfig(
            ChromaDeploymentMode.LOCAL_HTTP,
            endpoint.host,
            endpoint.port,
            False,
            1.0,
        ),
        information_root=information,
        persistence_path=information / "chroma-test",
        runtime_state_directory=information / "runtime" / "chroma",
        startup_timeout_seconds=20.0,
        shutdown_timeout_seconds=5.0,
        endpoint_release_timeout_seconds=5.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )
    store = AtomicChromaServerStateStore(config)
    controller = ChromaServerLifecycleController(config, state_store=store)
    guard = ChromaPersistenceGuard(config)
    owned_pids: set[int] = set()

    try:
        assert controller.start().state == "ready"
        state = store.load()
        assert state is not None
        owned_pids.add(state.pid)
        assert guard.verify_dedicated_server_owner().allowed is True

        production_context = ChromaPersistenceContext(
            role=ChromaPersistenceRole.PRODUCTION_CLIENT,
            deployment=config.deployment,
        )
        with pytest.raises(ChromaProtectedPathAccessDenied):
            guard.assert_embedded_access_allowed(
                path=config.persistence_path,
                context=production_context,
            )

        running_probe = _run_embedded_probe(config)
        assert running_probe.returncode == 4
        assert json.loads(running_probe.stdout) == {
            "allowed": False,
            "error": "server_owned_persistence_cannot_be_opened_embedded",
        }
        assert controller.health().state == "ready"

        assert controller.stop().state == "not_running"
        assert store.load() is None
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)

        stopped_probe = _run_embedded_probe(config)
        assert stopped_probe.returncode == 0, stopped_probe.stderr
        stopped_summary = json.loads(stopped_probe.stdout)
        assert stopped_summary["allowed"] is True
        assert stopped_summary["disposition"] == "approved_test_only"
        assert stopped_summary["server_ownership_state"] == "not_running"
        assert is_loopback_port_releasable(endpoint.port)
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
        assert all(not psutil.pid_exists(pid) for pid in owned_pids)

    released = information / "chroma-test-released"
    config.persistence_path.rename(released)
    released.rename(config.persistence_path)
    assert capture_protected_file_inventory(PROTECTED_CHROMA_ROOT) == protected_before
