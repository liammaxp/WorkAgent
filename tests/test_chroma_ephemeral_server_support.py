from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from backend.chroma_config import ChromaDeploymentMode, EXISTING_LOCAL_HTTP_PORT, LOOPBACK_HOST
from tests import chroma_http_test_support as support
from tests.chroma_http_test_support import (
    PROTECTED_CHROMA_ROOT,
    EphemeralChromaEndpoint,
    EphemeralChromaProcessExited,
    EphemeralChromaReadinessTimeout,
    EphemeralChromaServer,
    EphemeralChromaShutdownError,
    EphemeralChromaStartupError,
    EphemeralChromaUnsafeEndpoint,
    EphemeralChromaUnsafeStorage,
    allocate_dynamic_loopback_endpoint,
    create_ephemeral_storage_directory,
    is_loopback_port_releasable,
    remove_ephemeral_storage_directory,
    terminate_process_bounded,
    validate_ephemeral_storage_path,
)


class FakeProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float] = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("ephemeral-test", timeout)
        return self.returncode


class ForceKillProcess(FakeProcess):
    def terminate(self):
        self.terminate_calls += 1


class UnstoppableProcess(ForceKillProcess):
    def kill(self):
        self.kill_calls += 1


def _factory_for(process, captured=None):
    def factory(command, **kwargs):
        if captured is not None:
            captured.update({"command": command, **kwargs})
        return process

    return factory


def test_support_module_has_no_module_level_process_launch():
    tree = ast.parse(Path(support.__file__).read_text(encoding="utf-8"))
    top_level_calls = [
        node
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
        and not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
    ]
    assert top_level_calls == []


def test_server_construction_is_lazy_and_creates_no_storage(tmp_path):
    before = tuple(tmp_path.iterdir())
    server = EphemeralChromaServer(tmp_path)
    assert tuple(tmp_path.iterdir()) == before
    assert server.storage_path is None
    assert server.process_running is server.ready is False
    assert server.safe_summary() == {
        "process_state": "stopped",
        "host_scope": "loopback",
        "port": 0,
        "test_owned": True,
        "storage_scope": "temporary_test_owned",
    }


def test_unique_test_owned_storage_is_created_and_removed(tmp_path):
    first = create_ephemeral_storage_directory(tmp_path)
    second = create_ephemeral_storage_directory(tmp_path)
    try:
        assert first != second
        assert first.parent == second.parent == tmp_path.resolve()
        assert first.name.startswith("workagent-chroma-http-")
        assert second.name.startswith("workagent-chroma-http-")
    finally:
        remove_ephemeral_storage_directory(first, storage_parent=tmp_path)
        remove_ephemeral_storage_directory(second, storage_parent=tmp_path)
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    "candidate",
    [PROTECTED_CHROMA_ROOT, PROTECTED_CHROMA_ROOT / "nested-test-storage"],
)
def test_protected_storage_and_descendants_are_rejected(candidate):
    with pytest.raises(EphemeralChromaUnsafeStorage, match="ephemeral_chroma_storage_is_protected"):
        validate_ephemeral_storage_path(candidate)


def test_cleanup_rejects_non_owned_storage_name(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(EphemeralChromaUnsafeStorage, match="ephemeral_chroma_storage_not_test_owned"):
        remove_ephemeral_storage_directory(unrelated, storage_parent=tmp_path)
    assert unrelated.is_dir()


@pytest.mark.parametrize(
    "host,port,test_owned",
    [
        ("0.0.0.0", 9000, True),
        ("localhost", 9000, True),
        (LOOPBACK_HOST, EXISTING_LOCAL_HTTP_PORT, True),
        (LOOPBACK_HOST, 9000, False),
    ],
)
def test_endpoint_rejects_wildcard_alias_production_port_and_unowned_values(
    host, port, test_owned
):
    with pytest.raises(EphemeralChromaUnsafeEndpoint, match="unsafe_ephemeral_chroma_endpoint"):
        EphemeralChromaEndpoint(host=host, port=port, test_owned=test_owned)


def test_dynamic_endpoint_is_loopback_test_owned_and_not_production_port():
    endpoint = allocate_dynamic_loopback_endpoint()
    assert endpoint.host == LOOPBACK_HOST
    assert endpoint.port != EXISTING_LOCAL_HTTP_PORT
    assert endpoint.test_owned is True
    assert endpoint.safe_summary() == {
        "host_scope": "loopback",
        "port": endpoint.port,
        "test_owned": True,
    }
    assert is_loopback_port_releasable(endpoint.port)


def test_explicit_start_uses_sanitized_environment_temp_cwd_and_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_SERVER_HOST", "production.invalid")
    monkeypatch.setenv("PERSIST_DIRECTORY", str(PROTECTED_CHROMA_ROOT))
    captured = {}
    process = FakeProcess()
    endpoint = EphemeralChromaEndpoint(host=LOOPBACK_HOST, port=9311)
    server = EphemeralChromaServer(
        tmp_path,
        executable=sys.executable,
        process_factory=_factory_for(process, captured),
        endpoint_allocator=lambda: endpoint,
        readiness_probe=lambda *_: True,
    )
    started = server.start()
    storage = server.storage_path
    try:
        assert started == endpoint
        assert server.ready is True
        assert captured["shell"] is False
        assert captured["cwd"] == storage
        assert Path(captured["cwd"]).parent == tmp_path.resolve()
        assert captured["command"][0] == sys.executable
        assert captured["command"][1:3] == ["run", "--path"]
        assert captured["command"][-4:] == ["--host", LOOPBACK_HOST, "--port", "9311"]
        assert "CHROMA_SERVER_HOST" not in captured["env"]
        assert "PERSIST_DIRECTORY" not in captured["env"]
        assert captured["env"]["ANONYMIZED_TELEMETRY"] == "FALSE"
        assert not (Path(captured["cwd"]) / ".env").exists()
    finally:
        server.stop()
    assert storage is not None and not storage.exists()
    assert process.terminate_calls == 1


def test_ephemeral_config_is_explicit_test_owned_and_does_not_read_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DEPLOYMENT_MODE", "remote_http")
    process = FakeProcess()
    endpoint = EphemeralChromaEndpoint(host=LOOPBACK_HOST, port=9312)
    server = EphemeralChromaServer(
        tmp_path,
        executable=sys.executable,
        process_factory=_factory_for(process),
        endpoint_allocator=lambda: endpoint,
        readiness_probe=lambda *_: True,
    )
    server.start()
    try:
        config = server.deployment_config(timeout_seconds=0.25)
        assert config.mode is ChromaDeploymentMode.EPHEMERAL_TEST
        assert config.host == LOOPBACK_HOST
        assert config.port == endpoint.port
        assert config.timeout_seconds == 0.25
    finally:
        server.stop()


def test_config_requires_started_test_owned_endpoint(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    with pytest.raises(EphemeralChromaStartupError, match="ephemeral_chroma_server_not_started"):
        server.deployment_config()


def test_process_exit_before_readiness_is_bounded_and_cleans_storage(tmp_path):
    process = FakeProcess(returncode=7)
    server = EphemeralChromaServer(
        tmp_path,
        executable=sys.executable,
        process_factory=_factory_for(process),
        readiness_probe=lambda *_: False,
    )
    with pytest.raises(EphemeralChromaProcessExited, match="ephemeral_chroma_process_exited_before_ready"):
        server.start()
    assert server.storage_path is None
    assert tuple(tmp_path.iterdir()) == ()
    assert process.terminate_calls == 0


def test_missing_server_executable_cleans_created_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(support.shutil, "which", lambda _name: None)
    server = EphemeralChromaServer(tmp_path)
    with pytest.raises(EphemeralChromaStartupError, match="chroma_server_executable_unavailable"):
        server.start()
    assert server.storage_path is None
    assert tuple(tmp_path.iterdir()) == ()


def test_readiness_timeout_terminates_process_and_cleans_storage(tmp_path):
    process = FakeProcess()
    server = EphemeralChromaServer(
        tmp_path,
        executable=sys.executable,
        startup_timeout_seconds=0.02,
        process_factory=_factory_for(process),
        readiness_probe=lambda *_: False,
    )
    with pytest.raises(EphemeralChromaReadinessTimeout, match="ephemeral_chroma_readiness_timeout"):
        server.start()
    assert process.terminate_calls == 1
    assert server.storage_path is None
    assert tuple(tmp_path.iterdir()) == ()


def test_normal_and_assertion_failure_context_teardown_stop_and_clean(tmp_path):
    processes = []

    def process_factory(*_args, **_kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    with pytest.raises(AssertionError, match="controlled assertion"):
        with EphemeralChromaServer(
            tmp_path,
            executable=sys.executable,
            process_factory=process_factory,
            readiness_probe=lambda *_: True,
        ) as server:
            storage = server.storage_path
            raise AssertionError("controlled assertion")
    assert storage is not None and not storage.exists()
    assert processes[0].terminate_calls == 1
    assert tuple(tmp_path.iterdir()) == ()


def test_forced_termination_fallback_is_bounded():
    process = ForceKillProcess()
    result = terminate_process_bounded(process, timeout_seconds=0.01)
    assert result.forced is True
    assert result.already_stopped is False
    assert process.terminate_calls == process.kill_calls == 1
    assert process.wait_calls == [0.01, 0.01]


def test_unstoppable_process_raises_safe_shutdown_error():
    process = UnstoppableProcess()
    with pytest.raises(EphemeralChromaShutdownError, match="ephemeral_chroma_process_shutdown_timeout"):
        terminate_process_bounded(process, timeout_seconds=0.01)
    assert process.terminate_calls == process.kill_calls == 1


def test_safe_errors_and_summaries_do_not_expose_storage_paths(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    with pytest.raises(EphemeralChromaStartupError) as captured:
        _ = server.endpoint
    assert str(tmp_path) not in str(captured.value)
    assert str(PROTECTED_CHROMA_ROOT) not in str(captured.value)
    assert "documents" not in repr(server.safe_summary()).casefold()
    assert "embeddings" not in repr(server.safe_summary()).casefold()
