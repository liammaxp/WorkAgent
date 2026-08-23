from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_server_lifecycle import (
    CHROMA_SERVER_OWNERSHIP_ENV,
    EXIT_MODE,
    EXIT_UNHEALTHY,
    ChromaServerAlreadyRunning,
    ChromaServerDisabled,
    ChromaServerLifecycleController,
    ChromaServerOwnershipMismatch,
    ChromaServerPortConflict,
    ChromaServerShutdownTimeout,
    ChromaServerStartupTimeout,
    ChromaServerUnsupportedMode,
    ObservedChromaProcess,
    _command_identity,
    _executable_identity,
    _ownership_token_hash,
    main,
)
from backend.chroma_server_lifecycle_models import (
    CHROMA_SERVER_RUNTIME_STATE_SCHEMA,
    AtomicChromaServerStateStore,
    ChromaServerLifecycleConfig,
    ChromaServerRuntimeState,
    ChromaServerStateCorrupt,
    InvalidChromaServerLifecycleConfiguration,
    build_chroma_server_lifecycle_config,
)


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_CHROMA_ROOT = ROOT / "information" / "chroma"
LIFECYCLE_SOURCE = ROOT / "backend" / "chroma_server_lifecycle.py"
MODELS_SOURCE = ROOT / "backend" / "chroma_server_lifecycle_models.py"
WRAPPER_SOURCE = ROOT / "windows" / "chroma_server.ps1"
FIXED_TOKEN = "a" * 64


def deployment(
    mode: ChromaDeploymentMode = ChromaDeploymentMode.LOCAL_HTTP,
    *,
    port: int = 18125,
) -> ChromaDeploymentConfig:
    if mode is ChromaDeploymentMode.DISABLED:
        return ChromaDeploymentConfig(mode, None, None, False, 0.2)
    host = "remote.example" if mode is ChromaDeploymentMode.REMOTE_HTTP else "127.0.0.1"
    return ChromaDeploymentConfig(mode, host, port, mode is ChromaDeploymentMode.REMOTE_HTTP, 0.2)


def lifecycle_config(
    tmp_path: Path,
    mode: ChromaDeploymentMode = ChromaDeploymentMode.LOCAL_HTTP,
    *,
    port: int = 18125,
) -> ChromaServerLifecycleConfig:
    information = tmp_path / "information"
    information.mkdir(exist_ok=True)
    return build_chroma_server_lifecycle_config(
        deployment(mode, port=port),
        information_root=information,
        persistence_path=information / "chroma-test",
        runtime_state_directory=information / "runtime" / "chroma",
        startup_timeout_seconds=0.3,
        shutdown_timeout_seconds=0.2,
        endpoint_release_timeout_seconds=0.2,
        poll_interval_seconds=0.02,
        test_owned=True,
    )


class ManualClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.001, seconds)


class RecordingHeartbeat:
    def __init__(self, results: list[bool] | None = None, default: bool = True):
        self.results = list(results or [])
        self.default = default
        self.calls: list[float] = []

    def __call__(self, _deployment: ChromaDeploymentConfig, timeout: float) -> bool:
        self.calls.append(timeout)
        return self.results.pop(0) if self.results else self.default


class FakeProcessManager:
    def __init__(self, *, listen_on_spawn: bool = True, terminate_exits: bool = True):
        self.listen_on_spawn = listen_on_spawn
        self.terminate_exits = terminate_exits
        self.foreign_port = False
        self.next_pid = 4100
        self.processes: dict[int, ObservedChromaProcess] = {}
        self.spawn_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[int] = []
        self.kill_calls: list[int] = []

    def spawn(self, command, *, cwd, environment) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.spawn_calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "environment": dict(environment),
                "shell": False,
            }
        )
        token = environment[CHROMA_SERVER_OWNERSHIP_ENV]
        host = command[command.index("--host") + 1]
        port = int(command[command.index("--port") + 1])
        endpoints = ((host, port),) if self.listen_on_spawn else ()
        self.processes[pid] = ObservedChromaProcess(
            pid=pid,
            process_start_token=str(900_000_000 + pid),
            executable_identity=_executable_identity(command[0]),
            server_command_identity=_command_identity(command),
            ownership_token_hash=_ownership_token_hash(token),
            listening_endpoints=endpoints,
            _command=tuple(command),
        )
        return pid

    def inspect(self, pid: int):
        return self.processes.get(pid)

    def terminate(self, pid: int) -> None:
        self.terminate_calls.append(pid)
        if self.terminate_exits:
            self.processes.pop(pid, None)

    def kill(self, pid: int) -> None:
        self.kill_calls.append(pid)
        self.processes.pop(pid, None)

    def wait(self, pid: int, _timeout: float) -> bool:
        return pid not in self.processes

    def is_port_free(self, host: str, port: int) -> bool:
        if self.foreign_port:
            return False
        return not any((host, port) in item.listening_endpoints for item in self.processes.values())


def controller_parts(
    tmp_path: Path,
    *,
    mode: ChromaDeploymentMode = ChromaDeploymentMode.LOCAL_HTTP,
    process_manager: FakeProcessManager | None = None,
    heartbeat: RecordingHeartbeat | None = None,
):
    config = lifecycle_config(tmp_path, mode)
    manager = process_manager or FakeProcessManager()
    probe = heartbeat or RecordingHeartbeat()
    clock = ManualClock()
    store = AtomicChromaServerStateStore(config)
    controller = ChromaServerLifecycleController(
        config,
        state_store=store,
        process_manager=manager,
        heartbeat_probe=probe,
        executable_resolver=lambda _name: sys.executable,
        token_factory=lambda: FIXED_TOKEN,
        clock=clock,
        sleep=clock.sleep,
    )
    return controller, config, store, manager, probe, clock


def test_lifecycle_config_separates_server_paths_from_deployment_config(tmp_path):
    config = lifecycle_config(tmp_path)
    assert not hasattr(config.deployment, "persistence_path")
    assert config.persistence_path != config.runtime_state_directory
    assert config.test_owned is True
    encoded = json.dumps(config.safe_summary(), sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "chroma-test" not in repr(config)


def test_production_paths_are_deterministic_and_arbitrary_override_fails(tmp_path):
    production = build_chroma_server_lifecycle_config(deployment())
    assert production.persistence_path == PROTECTED_CHROMA_ROOT.resolve()
    assert production.runtime_state_directory == (ROOT / "information" / "runtime" / "chroma").resolve()
    information = tmp_path / "information"
    information.mkdir()
    with pytest.raises(
        InvalidChromaServerLifecycleConfiguration,
        match="production_chroma_server_paths_are_fixed",
    ):
        build_chroma_server_lifecycle_config(
            deployment(),
            information_root=information,
            persistence_path=information / "other",
            runtime_state_directory=information / "runtime",
        )


@pytest.mark.parametrize(
    "mode,error_type",
    (
        (ChromaDeploymentMode.DISABLED, ChromaServerDisabled),
        (ChromaDeploymentMode.REMOTE_HTTP, ChromaServerUnsupportedMode),
        (ChromaDeploymentMode.EPHEMERAL_TEST, ChromaServerUnsupportedMode),
    ),
)
def test_non_local_modes_never_reach_process_launch(tmp_path, mode, error_type):
    controller, _config, _store, manager, _probe, _clock = controller_parts(
        tmp_path, mode=mode
    )
    with pytest.raises(error_type):
        controller.start()
    assert manager.spawn_calls == []


def test_unsafe_local_host_is_rejected_before_process_launch(tmp_path):
    config = lifecycle_config(tmp_path)
    unsafe_deployment = dataclasses.replace(config.deployment, host="0.0.0.0")
    unsafe_config = dataclasses.replace(config, deployment=unsafe_deployment)
    manager = FakeProcessManager()
    controller = ChromaServerLifecycleController(
        unsafe_config,
        process_manager=manager,
        executable_resolver=lambda _name: sys.executable,
    )

    with pytest.raises(
        InvalidChromaServerLifecycleConfiguration,
        match="invalid_local_chroma_server_endpoint",
    ):
        controller.start()

    assert manager.spawn_calls == []


def test_state_schema_roundtrip_is_strict_atomic_and_privacy_safe(tmp_path):
    controller, config, store, _manager, _probe, _clock = controller_parts(tmp_path)
    result = controller.start()
    state = store.load()
    assert result.state == "ready"
    assert state is not None and state.schema == CHROMA_SERVER_RUNTIME_STATE_SCHEMA
    assert state.lifecycle_state == "ready"
    assert store.state_path.read_text(encoding="utf-8") == json.dumps(
        state.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    assert list(config.runtime_state_directory.glob("*.tmp")) == []
    safe = json.dumps(state.safe_summary(), sort_keys=True)
    raw = store.state_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in safe and str(tmp_path) not in raw
    assert "document" not in raw and "embedding" not in raw and FIXED_TOKEN not in raw


def test_corrupt_unknown_and_oversized_state_fail_closed(tmp_path):
    _controller, config, store, _manager, _probe, _clock = controller_parts(tmp_path)
    config.runtime_state_directory.mkdir(parents=True)
    store.state_path.write_text("{", encoding="utf-8")
    with pytest.raises(ChromaServerStateCorrupt, match="state_unreadable"):
        store.load()
    store.state_path.write_text(json.dumps({"unknown": True}), encoding="utf-8")
    with pytest.raises(ChromaServerStateCorrupt, match="state_shape"):
        store.load()
    store.state_path.write_bytes(b"x" * 8_193)
    with pytest.raises(ChromaServerStateCorrupt, match="state_too_large"):
        store.load()


def test_start_uses_argv_loopback_configured_port_path_and_sanitized_environment(tmp_path):
    controller, config, store, manager, probe, _clock = controller_parts(tmp_path)
    result = controller.start()
    assert result.state == "ready" and result.process_owned and result.server_reachable
    assert len(manager.spawn_calls) == 1
    call = manager.spawn_calls[0]
    command = call["command"]
    assert call["shell"] is False
    assert command[1] == "run"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == str(config.deployment.port)
    assert Path(command[command.index("--path") + 1]) == config.persistence_path
    assert CHROMA_SERVER_OWNERSHIP_ENV in call["environment"]
    assert not any(key.startswith("CHROMA_") for key in call["environment"])
    assert probe.calls and all(value <= config.deployment.timeout_seconds for value in probe.calls)
    assert store.load().lifecycle_state == "ready"


def test_healthy_owned_start_is_idempotent_and_does_not_spawn_again(tmp_path):
    controller, _config, _store, manager, _probe, _clock = controller_parts(tmp_path)
    first = controller.start()
    second = controller.start()
    assert first.detail == "started"
    assert second.detail == "already_running"
    assert len(manager.spawn_calls) == 1


def test_foreign_port_without_state_is_conflict_and_never_killed(tmp_path):
    manager = FakeProcessManager()
    manager.foreign_port = True
    controller, _config, _store, manager, _probe, _clock = controller_parts(
        tmp_path, process_manager=manager
    )
    health = controller.health()
    assert health.state == "foreign_port_conflict"
    with pytest.raises(ChromaServerPortConflict):
        controller.start()
    with pytest.raises(ChromaServerPortConflict):
        controller.stop()
    assert manager.terminate_calls == manager.kill_calls == []


def test_pid_reuse_creation_time_mismatch_refuses_stop_and_kill(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    state = store.load()
    observed = manager.processes[state.pid]
    manager.processes[state.pid] = dataclasses.replace(
        observed, process_start_token="123456789"
    )
    assert controller.health().state == "ownership_mismatch"
    with pytest.raises(ChromaServerOwnershipMismatch, match="refusing_to_stop"):
        controller.stop()
    assert manager.terminate_calls == manager.kill_calls == []


@pytest.mark.parametrize(
    "field,value",
    (
        ("ownership_token_hash", "b" * 64),
        ("server_command_identity", "b" * 64),
        ("executable_identity", "b" * 64),
    ),
)
def test_ownership_token_command_and_executable_mismatch_are_foreign(
    tmp_path, field, value
):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    state = store.load()
    manager.processes[state.pid] = dataclasses.replace(
        manager.processes[state.pid], **{field: value}
    )
    with pytest.raises(ChromaServerOwnershipMismatch):
        controller.stop()
    assert manager.terminate_calls == manager.kill_calls == []


def test_health_distinguishes_not_running_stale_starting_ready_and_unhealthy(tmp_path):
    controller, _config, store, manager, probe, _clock = controller_parts(tmp_path)
    assert controller.health().state == "not_running"
    controller.start()
    state = store.load()
    assert controller.health().state == "ready"
    store.write(dataclasses.replace(state, lifecycle_state="starting"))
    assert controller.health().state == "starting"
    probe.default = False
    assert controller.health().state == "unhealthy"
    manager.processes.pop(state.pid)
    assert controller.health().state == "stale_state"


def test_dead_process_stale_state_is_cleaned_only_when_endpoint_is_free(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    old_state = store.load()
    manager.processes.pop(old_state.pid)
    restarted = controller.start()
    assert restarted.state == "ready"
    assert store.load().pid != old_state.pid
    manager.processes.pop(store.load().pid)
    manager.foreign_port = True
    with pytest.raises(ChromaServerPortConflict):
        controller.start()
    assert store.state_path.exists()


def test_startup_timeout_is_bounded_terminates_owned_process_and_clears_state(tmp_path):
    manager = FakeProcessManager(listen_on_spawn=False)
    heartbeat = RecordingHeartbeat(default=False)
    controller, config, store, manager, _probe, clock = controller_parts(
        tmp_path, process_manager=manager, heartbeat=heartbeat
    )
    before = clock()
    with pytest.raises(ChromaServerStartupTimeout, match="startup_timeout"):
        controller.start()
    assert clock() - before <= config.startup_timeout_seconds + config.poll_interval_seconds
    assert manager.terminate_calls and manager.kill_calls == []
    assert store.load() is None


def test_stop_graceful_then_verified_kill_fallback_and_endpoint_release(tmp_path):
    manager = FakeProcessManager(terminate_exits=False)
    controller, _config, store, manager, _probe, _clock = controller_parts(
        tmp_path, process_manager=manager
    )
    controller.start()
    state = store.load()
    result = controller.stop()
    assert manager.terminate_calls == [state.pid]
    assert manager.kill_calls == [state.pid]
    assert result.forced_shutdown is True
    assert store.load() is None
    assert manager.is_port_free("127.0.0.1", 18125)


def test_endpoint_release_failure_retains_state_for_diagnosis(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    manager.foreign_port = True
    with pytest.raises(ChromaServerShutdownTimeout, match="endpoint_release_timeout"):
        controller.stop()
    assert store.state_path.exists()


def test_stop_absent_is_idempotent_and_stale_dead_state_is_removed(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    assert controller.stop().detail == "already_stopped"
    controller.start()
    state = store.load()
    manager.processes.pop(state.pid)
    assert controller.stop().detail == "stale_state_cleaned"
    assert store.load() is None


def test_restart_composes_verified_stop_and_new_start_identity(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    before = store.load()
    result = controller.restart()
    after = store.load()
    assert result.detail == "restarted"
    assert after.pid != before.pid
    assert manager.terminate_calls == [before.pid]


def test_restart_stops_on_ownership_mismatch_without_new_spawn(tmp_path):
    controller, _config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    state = store.load()
    manager.processes[state.pid] = dataclasses.replace(
        manager.processes[state.pid], ownership_token_hash="b" * 64
    )
    with pytest.raises(ChromaServerOwnershipMismatch):
        controller.restart()
    assert len(manager.spawn_calls) == 1
    assert manager.terminate_calls == manager.kill_calls == []


def test_crash_recovery_preserves_persistence_contents(tmp_path):
    controller, config, store, manager, _probe, _clock = controller_parts(tmp_path)
    controller.start()
    sentinel = config.persistence_path / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    state = store.load()
    manager.processes.pop(state.pid)
    controller.start()
    assert sentinel.read_bytes() == b"unchanged"


def test_import_constructor_and_health_disabled_do_not_start_server(tmp_path):
    controller, _config, _store, manager, _probe, _clock = controller_parts(
        tmp_path, mode=ChromaDeploymentMode.DISABLED
    )
    result = controller.health()
    assert result.state == "disabled"
    assert manager.spawn_calls == []
    source = LIFECYCLE_SOURCE.read_text(encoding="utf-8")
    assert "if __name__ == \"__main__\"" in source
    assert "shell=False" in source


def test_cli_disabled_health_and_start_have_deterministic_safe_exit_codes(
    monkeypatch, capsys
):
    monkeypatch.delenv("CHROMA_DEPLOYMENT_MODE", raising=False)
    for key in (
        "CHROMA_HTTP_HOST",
        "CHROMA_HTTP_PORT",
        "CHROMA_HTTP_SSL",
        "CHROMA_HTTP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    assert main(["health", "--json"]) == EXIT_UNHEALTHY
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "disabled"
    assert main(["start", "--json"]) == EXIT_MODE
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "command": "start",
        "error": "chroma_server_lifecycle_disabled",
        "state": "failed",
    }


def test_powershell_wrapper_is_thin_and_propagates_exit_code():
    source = WRAPPER_SOURCE.read_text(encoding="utf-8")
    normalized = source.casefold()
    assert "-m backend.chroma_server_lifecycle" in source
    assert "exit $exitCode" in source
    assert "get-process" not in normalized
    assert "stop-process" not in normalized
    assert "start-process" not in normalized
    assert "get-nettcpconnection" not in normalized
    assert "information\\chroma" not in normalized
    assert "c:\\users" not in normalized


def test_lifecycle_files_have_no_application_or_frontend_startup_integration():
    lifecycle = LIFECYCLE_SOURCE.read_text(encoding="utf-8").casefold()
    models = MODELS_SOURCE.read_text(encoding="utf-8").casefold()
    combined = lifecycle + models
    assert "fastapi" not in combined
    assert "api_server" not in combined
    assert "memory_store" not in combined
    assert "project_retrieval" not in combined
    assert "frontend" not in combined
    assert "get_or_create_collection" not in combined
    assert "persistent" + "client" not in combined
