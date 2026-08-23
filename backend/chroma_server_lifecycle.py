"""Authoritative explicit Windows-local lifecycle for a dedicated Chroma server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from backend.chroma_config import (
    LOOPBACK_HOST,
    ChromaConfigurationError,
    ChromaDeploymentConfig,
    ChromaDeploymentMode,
    load_chroma_deployment_config,
)
from backend.chroma_http_transport import BoundedChromaHttpTransport, ChromaTransportError
from backend.chroma_server_lifecycle_models import (
    CHROMA_SERVER_RUNTIME_STATE_SCHEMA,
    AtomicChromaServerStateStore,
    ChromaServerLifecycleConfig,
    ChromaServerLifecycleError,
    ChromaServerLifecycleResult,
    ChromaServerRuntimeState,
    ChromaServerStateCorrupt,
    ChromaServerStateWriteFailed,
    InvalidChromaServerLifecycleConfiguration,
    _require_safe_path_chain,
    build_chroma_server_lifecycle_config,
)


CHROMA_SERVER_OWNERSHIP_ENV = "WORKAGENT_CHROMA_SERVER_OWNERSHIP_TOKEN"
CHROMA_SERVER_COMMAND = "chroma"
CHROMA_SERVER_SUBCOMMAND = "run"
MIN_HEALTH_TIMEOUT_SECONDS = 0.1
MAX_READINESS_REQUEST_TIMEOUT_SECONDS = 0.5
OWNERSHIP_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_MODE = 3
EXIT_OWNERSHIP_OR_CONFLICT = 4
EXIT_STARTUP = 5
EXIT_SHUTDOWN = 6
EXIT_UNHEALTHY = 7
EXIT_CONFIGURATION = 8

_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)


class ChromaServerDisabled(ChromaServerLifecycleError):
    pass


class ChromaServerUnsupportedMode(ChromaServerLifecycleError):
    pass


class ChromaServerAlreadyRunning(ChromaServerLifecycleError):
    pass


class ChromaServerNotRunning(ChromaServerLifecycleError):
    pass


class ChromaServerPortConflict(ChromaServerLifecycleError):
    pass


class ChromaServerOwnershipMismatch(ChromaServerLifecycleError):
    pass


class ChromaServerStartupTimeout(ChromaServerLifecycleError):
    pass


class ChromaServerStartupFailed(ChromaServerLifecycleError):
    pass


class ChromaServerShutdownTimeout(ChromaServerLifecycleError):
    pass


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _process_start_token(value: float) -> str:
    return str(int(round(float(value) * 1_000_000)))


def _executable_identity(executable: str) -> str:
    normalized = os.path.normcase(os.path.realpath(executable))
    return _hash_text(normalized)


def _command_identity(command: Sequence[str]) -> str:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise ChromaServerOwnershipMismatch("invalid_chroma_server_command_identity")
    values = list(command)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ChromaServerOwnershipMismatch("invalid_chroma_server_command_identity")
    return _hash_text(
        json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    )


def _ownership_token_hash(value: str) -> str:
    if not isinstance(value, str) or not OWNERSHIP_TOKEN_RE.fullmatch(value):
        raise ChromaServerOwnershipMismatch("invalid_chroma_server_ownership_token")
    return _hash_text(value)


@dataclass(frozen=True, slots=True, repr=False)
class ObservedChromaProcess:
    pid: int
    process_start_token: str
    executable_identity: str
    server_command_identity: str
    ownership_token_hash: str | None
    listening_endpoints: tuple[tuple[str, int], ...]
    _command: tuple[str, ...]

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "process_observed": True,
            "endpoint_count": len(self.listening_endpoints),
            "ownership_token_present": self.ownership_token_hash is not None,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ObservedChromaProcess("
            f"process_observed={summary['process_observed']!r}, "
            f"endpoint_count={summary['endpoint_count']!r}, "
            f"ownership_token_present={summary['ownership_token_present']!r})"
        )


class WindowsChromaProcessManager:
    """Public subprocess, socket, and psutil boundary for Windows process ownership."""

    __slots__ = ()

    def spawn(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
    ) -> int:
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=creation_flags,
        )
        return int(process.pid)

    def inspect(self, pid: int) -> ObservedChromaProcess | None:
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            executable = process.exe()
            command = tuple(process.cmdline())
            environment = process.environ()
            token = environment.get(CHROMA_SERVER_OWNERSHIP_ENV)
            token_hash = (
                _ownership_token_hash(token)
                if isinstance(token, str) and OWNERSHIP_TOKEN_RE.fullmatch(token)
                else None
            )
            endpoints: set[tuple[str, int]] = set()
            process_tree = [process, *process.children(recursive=True)]
            for candidate in process_tree:
                try:
                    candidate_token = candidate.environ().get(
                        CHROMA_SERVER_OWNERSHIP_ENV
                    )
                    if candidate_token != token:
                        continue
                    connections = candidate.net_connections(kind="inet")
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                for connection in connections:
                    if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                        continue
                    address = connection.laddr.ip
                    port = connection.laddr.port
                    if isinstance(address, str) and isinstance(port, int):
                        endpoints.add((address, port))
            return ObservedChromaProcess(
                pid=pid,
                process_start_token=_process_start_token(process.create_time()),
                executable_identity=_executable_identity(executable),
                server_command_identity=_command_identity(command),
                ownership_token_hash=token_hash,
                listening_endpoints=tuple(sorted(endpoints)),
                _command=command,
            )
        except psutil.NoSuchProcess:
            return None
        except (psutil.AccessDenied, psutil.ZombieProcess, OSError, ValueError):
            raise ChromaServerOwnershipMismatch(
                "chroma_server_process_inspection_denied"
            ) from None

    def terminate(self, pid: int) -> None:
        try:
            psutil.Process(pid).terminate()
        except psutil.NoSuchProcess:
            return
        except (psutil.AccessDenied, OSError):
            raise ChromaServerOwnershipMismatch(
                "chroma_server_process_termination_denied"
            ) from None

    def kill(self, pid: int) -> None:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            return
        except (psutil.AccessDenied, OSError):
            raise ChromaServerOwnershipMismatch(
                "chroma_server_process_kill_denied"
            ) from None

    def wait(self, pid: int, timeout_seconds: float) -> bool:
        try:
            psutil.Process(pid).wait(timeout=timeout_seconds)
            return True
        except psutil.NoSuchProcess:
            return True
        except psutil.TimeoutExpired:
            return False
        except (psutil.AccessDenied, OSError):
            raise ChromaServerOwnershipMismatch(
                "chroma_server_process_wait_denied"
            ) from None

    def is_port_free(self, host: str, port: int) -> bool:
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            candidate.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            candidate.close()


def _bounded_heartbeat(
    deployment: ChromaDeploymentConfig,
    timeout_seconds: float,
) -> bool:
    timeout = min(
        deployment.timeout_seconds,
        max(MIN_HEALTH_TIMEOUT_SECONDS, float(timeout_seconds)),
    )
    probe_config = ChromaDeploymentConfig(
        mode=deployment.mode,
        host=deployment.host,
        port=deployment.port,
        ssl=deployment.ssl,
        timeout_seconds=timeout,
    )
    adapter: BoundedChromaHttpTransport | None = None
    try:
        adapter = BoundedChromaHttpTransport(probe_config)
        adapter.heartbeat()
        return True
    except ChromaTransportError:
        return False
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except ChromaTransportError:
                pass


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        return None
    return command[indexes[0] + 1]


def _expected_command(
    observed: ObservedChromaProcess,
    config: ChromaServerLifecycleConfig,
) -> bool:
    command = observed._command
    if CHROMA_SERVER_SUBCOMMAND not in command:
        return False
    if _flag_value(command, "--host") != LOOPBACK_HOST:
        return False
    if _flag_value(command, "--port") != str(config.deployment.port):
        return False
    path_value = _flag_value(command, "--path")
    if path_value is None:
        return False
    try:
        actual_path = Path(path_value).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return actual_path == config.persistence_path


def _server_environment(ownership_token: str) -> dict[str, str]:
    if not OWNERSHIP_TOKEN_RE.fullmatch(ownership_token):
        raise ChromaServerStartupFailed("invalid_chroma_server_ownership_token")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _SAFE_ENVIRONMENT_KEYS and isinstance(value, str)
    }
    environment.update(
        {
            "ANONYMIZED_TELEMETRY": "FALSE",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            CHROMA_SERVER_OWNERSHIP_ENV: ownership_token,
        }
    )
    return environment


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ChromaServerLifecycleController:
    """Single explicit authority for start, health, stop, and restart."""

    __slots__ = (
        "_clock",
        "_config",
        "_executable_resolver",
        "_heartbeat_probe",
        "_processes",
        "_sleep",
        "_state_store",
        "_token_factory",
    )

    def __init__(
        self,
        config: ChromaServerLifecycleConfig,
        *,
        state_store: AtomicChromaServerStateStore | None = None,
        process_manager: Any | None = None,
        heartbeat_probe: Callable[[ChromaDeploymentConfig, float], bool] = _bounded_heartbeat,
        executable_resolver: Callable[[str], str | None] = shutil.which,
        token_factory: Callable[[], str] = lambda: secrets.token_hex(32),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, ChromaServerLifecycleConfig):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_lifecycle_controller_config"
            )
        for callback, code in (
            (heartbeat_probe, "invalid_chroma_server_heartbeat_probe"),
            (executable_resolver, "invalid_chroma_server_executable_resolver"),
            (token_factory, "invalid_chroma_server_token_factory"),
            (clock, "invalid_chroma_server_clock"),
            (sleep, "invalid_chroma_server_sleep"),
        ):
            if not callable(callback):
                raise InvalidChromaServerLifecycleConfiguration(code)
        self._config = config
        self._state_store = state_store or AtomicChromaServerStateStore(config)
        self._processes = process_manager or WindowsChromaProcessManager()
        self._heartbeat_probe = heartbeat_probe
        self._executable_resolver = executable_resolver
        self._token_factory = token_factory
        self._clock = clock
        self._sleep = sleep

    def _result(
        self,
        state: str,
        *,
        process_owned: bool,
        reachable: bool,
        detail: str,
        forced: bool = False,
    ) -> ChromaServerLifecycleResult:
        deployment = self._config.deployment
        endpoint_scope = {
            ChromaDeploymentMode.DISABLED: "none",
            ChromaDeploymentMode.LOCAL_HTTP: "loopback",
            ChromaDeploymentMode.REMOTE_HTTP: "remote",
            ChromaDeploymentMode.EPHEMERAL_TEST: "test_owned",
        }[deployment.mode]
        return ChromaServerLifecycleResult(
            state=state,
            deployment_mode=deployment.mode.value,
            endpoint_scope=endpoint_scope,
            port=deployment.port or 0,
            process_owned=process_owned,
            server_reachable=reachable,
            detail=detail,
            forced_shutdown=forced,
        )

    def _require_local_mode(self) -> None:
        deployment = self._config.deployment
        if deployment.mode is ChromaDeploymentMode.DISABLED:
            raise ChromaServerDisabled("chroma_server_lifecycle_disabled")
        if deployment.mode is not ChromaDeploymentMode.LOCAL_HTTP:
            raise ChromaServerUnsupportedMode(
                "chroma_server_lifecycle_requires_local_http"
            )
        if (
            deployment.host != LOOPBACK_HOST
            or deployment.port is None
            or deployment.ssl is not False
        ):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_local_chroma_server_endpoint"
            )

    def _ownership_status(
        self, state: ChromaServerRuntimeState
    ) -> tuple[str, ObservedChromaProcess | None]:
        if state.port != self._config.deployment.port or state.host_scope != "loopback":
            return "mismatch", None
        observed = self._processes.inspect(state.pid)
        if observed is None:
            return "missing", None
        if (
            observed.pid != state.pid
            or observed.process_start_token != state.process_start_token
            or observed.executable_identity != state.executable_identity
            or observed.server_command_identity != state.server_command_identity
            or observed.ownership_token_hash != state.ownership_token_hash
            or not _expected_command(observed, self._config)
        ):
            return "mismatch", observed
        return "owned", observed

    def _endpoint_owned(self, observed: ObservedChromaProcess) -> bool:
        deployment = self._config.deployment
        return (deployment.host, deployment.port) in observed.listening_endpoints

    def _heartbeat(self, timeout_seconds: float) -> bool:
        return bool(self._heartbeat_probe(self._config.deployment, timeout_seconds))

    def _prepare_server_directories(self) -> None:
        _require_safe_path_chain(
            self._config.information_root,
            self._config.persistence_path,
            "unsafe_chroma_server_persistence_path",
        )
        try:
            self._config.persistence_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise ChromaServerStartupFailed(
                "chroma_server_persistence_unavailable"
            ) from None
        _require_safe_path_chain(
            self._config.information_root,
            self._config.persistence_path,
            "unsafe_chroma_server_persistence_path",
        )
        if not self._config.persistence_path.is_dir():
            raise ChromaServerStartupFailed("chroma_server_persistence_unavailable")
        if any(
            (self._config.persistence_path / name).exists()
            for name in (".env", ".env.local", ".env.production")
        ):
            raise ChromaServerStartupFailed(
                "unsafe_chroma_server_persistence_environment"
            )

    def _resolve_executable(self) -> str:
        executable = self._executable_resolver(CHROMA_SERVER_COMMAND)
        if not executable:
            raise ChromaServerStartupFailed("chroma_server_executable_unavailable")
        try:
            candidate = Path(executable).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ChromaServerStartupFailed(
                "chroma_server_executable_unavailable"
            ) from None
        if not candidate.is_file():
            raise ChromaServerStartupFailed("chroma_server_executable_unavailable")
        return str(candidate)

    def _command(self, executable: str) -> list[str]:
        deployment = self._config.deployment
        return [
            executable,
            CHROMA_SERVER_SUBCOMMAND,
            "--path",
            str(self._config.persistence_path),
            "--host",
            str(deployment.host),
            "--port",
            str(deployment.port),
        ]

    def _state_from_observation(
        self,
        observed: ObservedChromaProcess,
        ownership_token: str,
    ) -> ChromaServerRuntimeState:
        expected_hash = _ownership_token_hash(ownership_token)
        if observed.ownership_token_hash != expected_hash:
            raise ChromaServerOwnershipMismatch(
                "chroma_server_ownership_token_mismatch"
            )
        if not _expected_command(observed, self._config):
            raise ChromaServerOwnershipMismatch(
                "chroma_server_command_identity_mismatch"
            )
        return ChromaServerRuntimeState(
            schema=CHROMA_SERVER_RUNTIME_STATE_SCHEMA,
            pid=observed.pid,
            host_scope="loopback",
            port=int(self._config.deployment.port),
            process_start_token=observed.process_start_token,
            executable_identity=observed.executable_identity,
            server_command_identity=observed.server_command_identity,
            ownership_token_hash=expected_hash,
            lifecycle_state="starting",
            created_at=_utc_timestamp(),
        )

    def health(self) -> ChromaServerLifecycleResult:
        deployment = self._config.deployment
        if deployment.mode is ChromaDeploymentMode.DISABLED:
            return self._result(
                "disabled", process_owned=False, reachable=False, detail="disabled"
            )
        if deployment.mode is not ChromaDeploymentMode.LOCAL_HTTP:
            return self._result(
                "unsupported_mode",
                process_owned=False,
                reachable=False,
                detail="local_ownership_not_applicable",
            )
        self._require_local_mode()
        state = self._state_store.load()
        port_free = self._processes.is_port_free(deployment.host, deployment.port)
        if state is None:
            if port_free:
                return self._result(
                    "not_running",
                    process_owned=False,
                    reachable=False,
                    detail="runtime_state_absent",
                )
            return self._result(
                "foreign_port_conflict",
                process_owned=False,
                reachable=False,
                detail="unowned_endpoint_occupied",
            )
        ownership, observed = self._ownership_status(state)
        if ownership == "missing":
            if port_free:
                return self._result(
                    "stale_state",
                    process_owned=False,
                    reachable=False,
                    detail="owned_process_missing",
                )
            return self._result(
                "foreign_port_conflict",
                process_owned=False,
                reachable=False,
                detail="stale_state_and_endpoint_occupied",
            )
        if ownership != "owned" or observed is None:
            return self._result(
                "ownership_mismatch",
                process_owned=False,
                reachable=False,
                detail="process_identity_mismatch",
            )
        endpoint_owned = self._endpoint_owned(observed)
        if not endpoint_owned:
            if not port_free:
                return self._result(
                    "foreign_port_conflict",
                    process_owned=True,
                    reachable=False,
                    detail="owned_process_foreign_endpoint",
                )
            return self._result(
                "unhealthy",
                process_owned=True,
                reachable=False,
                detail="owned_process_not_listening",
            )
        reachable = self._heartbeat(deployment.timeout_seconds)
        if not reachable:
            return self._result(
                "unhealthy",
                process_owned=True,
                reachable=False,
                detail="owned_endpoint_unhealthy",
            )
        if state.lifecycle_state == "starting":
            return self._result(
                "starting",
                process_owned=True,
                reachable=True,
                detail="owned_process_starting",
            )
        return self._result(
            "ready", process_owned=True, reachable=True, detail="owned_process_ready"
        )

    def _cleanup_started_process(self, state: ChromaServerRuntimeState) -> bool:
        ownership, _ = self._ownership_status(state)
        if ownership == "missing":
            return False
        if ownership != "owned":
            raise ChromaServerOwnershipMismatch(
                "chroma_server_startup_cleanup_ownership_mismatch"
            )
        self._processes.terminate(state.pid)
        if self._processes.wait(state.pid, self._config.shutdown_timeout_seconds):
            return False
        ownership, _ = self._ownership_status(state)
        if ownership == "missing":
            return False
        if ownership != "owned":
            raise ChromaServerOwnershipMismatch(
                "chroma_server_startup_cleanup_ownership_mismatch"
            )
        self._processes.kill(state.pid)
        if not self._processes.wait(
            state.pid, self._config.shutdown_timeout_seconds
        ):
            raise ChromaServerShutdownTimeout(
                "chroma_server_startup_cleanup_timeout"
            )
        return True

    def start(self) -> ChromaServerLifecycleResult:
        self._require_local_mode()
        deployment = self._config.deployment
        state = self._state_store.load()
        if state is not None:
            ownership, observed = self._ownership_status(state)
            if ownership == "owned" and observed is not None:
                if self._endpoint_owned(observed) and self._heartbeat(
                    deployment.timeout_seconds
                ):
                    if state.lifecycle_state != "ready":
                        state = state.mark_ready()
                        self._state_store.write(state)
                    return self._result(
                        "ready",
                        process_owned=True,
                        reachable=True,
                        detail="already_running",
                    )
                raise ChromaServerAlreadyRunning(
                    "owned_chroma_server_is_unhealthy"
                )
            if ownership == "mismatch":
                raise ChromaServerOwnershipMismatch(
                    "chroma_server_runtime_state_ownership_mismatch"
                )
            if not self._processes.is_port_free(deployment.host, deployment.port):
                raise ChromaServerPortConflict(
                    "chroma_server_endpoint_owned_by_foreign_process"
                )
            self._state_store.clear()
        elif not self._processes.is_port_free(deployment.host, deployment.port):
            raise ChromaServerPortConflict(
                "chroma_server_endpoint_owned_by_foreign_process"
            )

        self._prepare_server_directories()
        executable = self._resolve_executable()
        command = self._command(executable)
        ownership_token = self._token_factory()
        if not isinstance(ownership_token, str) or not OWNERSHIP_TOKEN_RE.fullmatch(
            ownership_token
        ):
            raise ChromaServerStartupFailed("invalid_chroma_server_ownership_token")
        try:
            pid = self._processes.spawn(
                command,
                cwd=self._config.persistence_path,
                environment=_server_environment(ownership_token),
            )
        except ChromaServerLifecycleError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise ChromaServerStartupFailed("chroma_server_process_start_failed") from None

        provisional: ChromaServerRuntimeState | None = None
        deadline = self._clock() + self._config.startup_timeout_seconds
        try:
            while self._clock() < deadline and provisional is None:
                observed = self._processes.inspect(pid)
                if observed is None:
                    raise ChromaServerStartupFailed(
                        "chroma_server_exited_before_state_capture"
                    )
                try:
                    provisional = self._state_from_observation(
                        observed, ownership_token
                    )
                except ChromaServerOwnershipMismatch:
                    remaining = deadline - self._clock()
                    if remaining <= self._config.poll_interval_seconds:
                        raise
                    self._sleep(self._config.poll_interval_seconds)
            if provisional is None:
                raise ChromaServerStartupTimeout(
                    "chroma_server_state_capture_timeout"
                )
            self._state_store.write(provisional)
            while self._clock() < deadline:
                ownership, observed = self._ownership_status(provisional)
                if ownership == "missing":
                    self._state_store.clear()
                    raise ChromaServerStartupFailed(
                        "chroma_server_exited_before_ready"
                    )
                if ownership != "owned" or observed is None:
                    raise ChromaServerOwnershipMismatch(
                        "chroma_server_startup_ownership_mismatch"
                    )
                remaining = deadline - self._clock()
                request_timeout = min(
                    MAX_READINESS_REQUEST_TIMEOUT_SECONDS,
                    deployment.timeout_seconds,
                    max(MIN_HEALTH_TIMEOUT_SECONDS, remaining),
                )
                if self._endpoint_owned(observed) and self._heartbeat(request_timeout):
                    ready_state = provisional.mark_ready()
                    self._state_store.write(ready_state)
                    return self._result(
                        "ready",
                        process_owned=True,
                        reachable=True,
                        detail="started",
                    )
                self._sleep(
                    min(self._config.poll_interval_seconds, max(0.0, remaining))
                )
            raise ChromaServerStartupTimeout("chroma_server_startup_timeout")
        except (ChromaServerStartupTimeout, ChromaServerStartupFailed):
            if provisional is not None:
                try:
                    self._cleanup_started_process(provisional)
                    self._state_store.clear()
                except ChromaServerLifecycleError:
                    pass
            raise
        except ChromaServerStateWriteFailed:
            if provisional is not None:
                self._cleanup_started_process(provisional)
                self._state_store.clear()
            raise

    def _wait_for_endpoint_release(self) -> bool:
        deployment = self._config.deployment
        deadline = self._clock() + self._config.endpoint_release_timeout_seconds
        while self._clock() < deadline:
            if self._processes.is_port_free(deployment.host, deployment.port):
                return True
            remaining = deadline - self._clock()
            self._sleep(min(self._config.poll_interval_seconds, max(0.0, remaining)))
        return self._processes.is_port_free(deployment.host, deployment.port)

    def stop(self) -> ChromaServerLifecycleResult:
        self._require_local_mode()
        deployment = self._config.deployment
        state = self._state_store.load()
        if state is None:
            if self._processes.is_port_free(deployment.host, deployment.port):
                return self._result(
                    "not_running",
                    process_owned=False,
                    reachable=False,
                    detail="already_stopped",
                )
            raise ChromaServerPortConflict(
                "cannot_stop_unowned_chroma_endpoint"
            )
        ownership, _ = self._ownership_status(state)
        if ownership == "missing":
            if not self._processes.is_port_free(deployment.host, deployment.port):
                raise ChromaServerPortConflict(
                    "stale_state_with_foreign_endpoint"
                )
            self._state_store.clear()
            return self._result(
                "not_running",
                process_owned=False,
                reachable=False,
                detail="stale_state_cleaned",
            )
        if ownership != "owned":
            raise ChromaServerOwnershipMismatch(
                "refusing_to_stop_unowned_process"
            )

        self._processes.terminate(state.pid)
        forced = False
        if not self._processes.wait(state.pid, self._config.shutdown_timeout_seconds):
            ownership, _ = self._ownership_status(state)
            if ownership == "missing":
                pass
            elif ownership != "owned":
                raise ChromaServerOwnershipMismatch(
                    "refusing_to_kill_unowned_process"
                )
            else:
                self._processes.kill(state.pid)
                forced = True
                if not self._processes.wait(
                    state.pid, self._config.shutdown_timeout_seconds
                ):
                    raise ChromaServerShutdownTimeout(
                        "chroma_server_shutdown_timeout"
                    )
        if not self._wait_for_endpoint_release():
            raise ChromaServerShutdownTimeout(
                "chroma_server_endpoint_release_timeout"
            )
        self._state_store.clear()
        return self._result(
            "not_running",
            process_owned=True,
            reachable=False,
            detail="stopped",
            forced=forced,
        )

    def restart(self) -> ChromaServerLifecycleResult:
        self.stop()
        result = self.start()
        return ChromaServerLifecycleResult(
            **{**result.safe_summary(), "detail": "restarted"}
        )


def inspect_chroma_server_ownership(
    config: ChromaServerLifecycleConfig,
    *,
    state_store: AtomicChromaServerStateStore | None = None,
    process_manager: Any | None = None,
    heartbeat_probe: Callable[[ChromaDeploymentConfig, float], bool] = _bounded_heartbeat,
) -> ChromaServerLifecycleResult:
    """Return lifecycle-authoritative ownership state without starting a server."""

    controller = ChromaServerLifecycleController(
        config,
        state_store=state_store,
        process_manager=process_manager,
        heartbeat_probe=heartbeat_probe,
    )
    return controller.health()


class _Parser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, "error:invalid_arguments\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m backend.chroma_server_lifecycle",
        description="Manage the explicit WorkAgent-owned local Chroma server lifecycle.",
    )
    parser.add_argument("command", choices=("start", "health", "stop", "restart"))
    parser.add_argument("--json", action="store_true", help="Print canonical safe JSON.")
    return parser


def _error_exit_code(error: BaseException) -> int:
    if isinstance(error, (ChromaServerDisabled, ChromaServerUnsupportedMode)):
        return EXIT_MODE
    if isinstance(
        error,
        (
            ChromaServerAlreadyRunning,
            ChromaServerPortConflict,
            ChromaServerOwnershipMismatch,
            ChromaServerStateCorrupt,
        ),
    ):
        return EXIT_OWNERSHIP_OR_CONFLICT
    if isinstance(error, (ChromaServerStartupFailed, ChromaServerStartupTimeout)):
        return EXIT_STARTUP
    if isinstance(error, ChromaServerShutdownTimeout):
        return EXIT_SHUTDOWN
    if isinstance(
        error,
        (InvalidChromaServerLifecycleConfiguration, ChromaConfigurationError),
    ):
        return EXIT_CONFIGURATION
    return EXIT_CONFIGURATION


def _print_result(
    command: str, result: ChromaServerLifecycleResult, *, json_output: bool
) -> None:
    summary = result.safe_summary()
    if json_output:
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return
    print(
        "chroma lifecycle "
        f"command={command} state={summary['state']} "
        f"process_owned={str(summary['process_owned']).lower()} "
        f"server_reachable={str(summary['server_reachable']).lower()} "
        f"detail={summary['detail']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        deployment = load_chroma_deployment_config()
        config = build_chroma_server_lifecycle_config(deployment)
        controller = ChromaServerLifecycleController(config)
        result = getattr(controller, arguments.command)()
    except (ChromaServerLifecycleError, ChromaConfigurationError) as error:
        code = getattr(error, "code", "chroma_server_lifecycle_failed")
        if arguments.json:
            print(
                json.dumps(
                    {"command": arguments.command, "error": code, "state": "failed"},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
        else:
            print(
                f"chroma lifecycle command={arguments.command} failed code={code}",
                file=sys.stderr,
            )
        return _error_exit_code(error)
    _print_result(arguments.command, result, json_output=arguments.json)
    if arguments.command == "health" and result.state != "ready":
        return EXIT_UNHEALTHY
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
