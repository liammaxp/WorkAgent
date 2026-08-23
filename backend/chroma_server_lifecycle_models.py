"""Strict server-only lifecycle configuration, state, and result models."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from backend.chroma_config import ChromaDeploymentConfig
except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
    from chroma_config import ChromaDeploymentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFORMATION_ROOT = PROJECT_ROOT / "information"
DEFAULT_CHROMA_SERVER_PERSISTENCE_PATH = DEFAULT_INFORMATION_ROOT / "chroma"
DEFAULT_CHROMA_SERVER_RUNTIME_DIRECTORY = (
    DEFAULT_INFORMATION_ROOT / "runtime" / "chroma"
)
CHROMA_SERVER_RUNTIME_STATE_SCHEMA = "chroma_server_runtime_state.v1"
CHROMA_SERVER_RUNTIME_STATE_FILENAME = "runtime_state.json"
MAX_RUNTIME_STATE_BYTES = 8_192
DEFAULT_STARTUP_TIMEOUT_SECONDS = 20.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0
DEFAULT_ENDPOINT_RELEASE_TIMEOUT_SECONDS = 5.0
DEFAULT_LIFECYCLE_POLL_INTERVAL_SECONDS = 0.05
MIN_LIFECYCLE_DEADLINE_SECONDS = 0.1
MAX_LIFECYCLE_DEADLINE_SECONDS = 120.0
MAX_LIFECYCLE_POLL_INTERVAL_SECONDS = 1.0

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_START_TOKEN_RE = re.compile(r"^[0-9]{1,24}$")
_CREATED_AT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_RUNTIME_STATE_FIELDS = frozenset(
    {
        "schema",
        "pid",
        "host_scope",
        "port",
        "process_start_token",
        "executable_identity",
        "server_command_identity",
        "ownership_token_hash",
        "lifecycle_state",
        "created_at",
    }
)
_LIFECYCLE_STATES = frozenset({"starting", "ready"})
_RESULT_STATES = frozenset(
    {
        "disabled",
        "unsupported_mode",
        "not_running",
        "starting",
        "ready",
        "unhealthy",
        "stale_state",
        "foreign_port_conflict",
        "ownership_mismatch",
    }
)


class ChromaServerLifecycleError(RuntimeError):
    """Stable lifecycle failure that never includes paths, commands, or environment."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class InvalidChromaServerLifecycleConfiguration(ChromaServerLifecycleError):
    pass


class ChromaServerStateCorrupt(ChromaServerLifecycleError):
    pass


class ChromaServerStateWriteFailed(ChromaServerLifecycleError):
    pass


def _resolved(path: str | Path, code: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise InvalidChromaServerLifecycleConfiguration(code)
    try:
        candidate = Path(path)
        if not str(candidate).strip():
            raise ValueError
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise InvalidChromaServerLifecycleConfiguration(code) from None


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _require_safe_path_chain(root: Path, target: Path, code: str) -> None:
    try:
        target.relative_to(root)
    except ValueError:
        raise InvalidChromaServerLifecycleConfiguration(code) from None
    current = root
    if current.exists() and (current.is_symlink() or _is_reparse_point(current)):
        raise InvalidChromaServerLifecycleConfiguration(code)
    for part in target.relative_to(root).parts:
        current = current / part
        if current.exists() and (current.is_symlink() or _is_reparse_point(current)):
            raise InvalidChromaServerLifecycleConfiguration(code)


def _bounded_deadline(value: Any, code: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidChromaServerLifecycleConfiguration(code)
    result = float(value)
    if (
        not math.isfinite(result)
        or result < MIN_LIFECYCLE_DEADLINE_SECONDS
        or result > MAX_LIFECYCLE_DEADLINE_SECONDS
    ):
        raise InvalidChromaServerLifecycleConfiguration(code)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class ChromaServerLifecycleConfig:
    """Server-operations configuration kept separate from application connection config."""

    deployment: ChromaDeploymentConfig
    information_root: Path
    persistence_path: Path
    runtime_state_directory: Path
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    endpoint_release_timeout_seconds: float = DEFAULT_ENDPOINT_RELEASE_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_LIFECYCLE_POLL_INTERVAL_SECONDS
    test_owned: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, ChromaDeploymentConfig):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_deployment_configuration"
            )
        if not isinstance(self.test_owned, bool):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_test_ownership"
            )
        information_root = _resolved(
            self.information_root, "invalid_chroma_server_information_root"
        )
        persistence_path = _resolved(
            self.persistence_path, "invalid_chroma_server_persistence_path"
        )
        runtime_directory = _resolved(
            self.runtime_state_directory, "invalid_chroma_server_runtime_path"
        )
        if not information_root.is_dir():
            raise InvalidChromaServerLifecycleConfiguration(
                "chroma_server_information_root_unavailable"
            )
        _require_safe_path_chain(
            information_root,
            persistence_path,
            "unsafe_chroma_server_persistence_path",
        )
        _require_safe_path_chain(
            information_root,
            runtime_directory,
            "unsafe_chroma_server_runtime_path",
        )
        if persistence_path == information_root or runtime_directory == information_root:
            raise InvalidChromaServerLifecycleConfiguration(
                "unsafe_chroma_server_root_path"
            )
        try:
            persistence_path.relative_to(runtime_directory)
            nested = True
        except ValueError:
            nested = False
        try:
            runtime_directory.relative_to(persistence_path)
            nested = True
        except ValueError:
            pass
        if nested:
            raise InvalidChromaServerLifecycleConfiguration(
                "overlapping_chroma_server_runtime_and_persistence"
            )
        production_information = DEFAULT_INFORMATION_ROOT.resolve(strict=False)
        production_persistence = DEFAULT_CHROMA_SERVER_PERSISTENCE_PATH.resolve(
            strict=False
        )
        production_runtime = DEFAULT_CHROMA_SERVER_RUNTIME_DIRECTORY.resolve(
            strict=False
        )
        if self.test_owned:
            if (
                information_root == production_information
                or persistence_path == production_persistence
                or runtime_directory == production_runtime
            ):
                raise InvalidChromaServerLifecycleConfiguration(
                    "test_lifecycle_cannot_use_production_paths"
                )
        elif (
            information_root != production_information
            or persistence_path != production_persistence
            or runtime_directory != production_runtime
        ):
            raise InvalidChromaServerLifecycleConfiguration(
                "production_chroma_server_paths_are_fixed"
            )
        startup = _bounded_deadline(
            self.startup_timeout_seconds, "invalid_chroma_server_startup_deadline"
        )
        shutdown = _bounded_deadline(
            self.shutdown_timeout_seconds, "invalid_chroma_server_shutdown_deadline"
        )
        release = _bounded_deadline(
            self.endpoint_release_timeout_seconds,
            "invalid_chroma_server_endpoint_release_deadline",
        )
        if (
            not isinstance(self.poll_interval_seconds, (int, float))
            or isinstance(self.poll_interval_seconds, bool)
        ):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_poll_interval"
            )
        poll = float(self.poll_interval_seconds)
        if (
            not math.isfinite(poll)
            or poll <= 0
            or poll > MAX_LIFECYCLE_POLL_INTERVAL_SECONDS
            or poll >= min(startup, shutdown, release)
        ):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_poll_interval"
            )
        object.__setattr__(self, "information_root", information_root)
        object.__setattr__(self, "persistence_path", persistence_path)
        object.__setattr__(self, "runtime_state_directory", runtime_directory)
        object.__setattr__(self, "startup_timeout_seconds", startup)
        object.__setattr__(self, "shutdown_timeout_seconds", shutdown)
        object.__setattr__(self, "endpoint_release_timeout_seconds", release)
        object.__setattr__(self, "poll_interval_seconds", poll)

    @property
    def runtime_state_path(self) -> Path:
        return self.runtime_state_directory / CHROMA_SERVER_RUNTIME_STATE_FILENAME

    def safe_summary(self) -> dict[str, str | float | bool]:
        return {
            "deployment_mode": self.deployment.mode.value,
            "endpoint_scope": "loopback" if self.deployment.is_local else "none",
            "persistence_scope": "test_owned" if self.test_owned else "server_owned",
            "runtime_state_scope": "test_owned" if self.test_owned else "server_owned",
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "endpoint_release_timeout_seconds": self.endpoint_release_timeout_seconds,
            "test_owned": self.test_owned,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaServerLifecycleConfig("
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"endpoint_scope={summary['endpoint_scope']!r}, "
            f"persistence_scope={summary['persistence_scope']!r}, "
            f"test_owned={summary['test_owned']!r})"
        )


def build_chroma_server_lifecycle_config(
    deployment: ChromaDeploymentConfig,
    *,
    information_root: str | Path = DEFAULT_INFORMATION_ROOT,
    persistence_path: str | Path = DEFAULT_CHROMA_SERVER_PERSISTENCE_PATH,
    runtime_state_directory: str | Path = DEFAULT_CHROMA_SERVER_RUNTIME_DIRECTORY,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    endpoint_release_timeout_seconds: float = DEFAULT_ENDPOINT_RELEASE_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_LIFECYCLE_POLL_INTERVAL_SECONDS,
    test_owned: bool = False,
) -> ChromaServerLifecycleConfig:
    return ChromaServerLifecycleConfig(
        deployment=deployment,
        information_root=Path(information_root),
        persistence_path=Path(persistence_path),
        runtime_state_directory=Path(runtime_state_directory),
        startup_timeout_seconds=startup_timeout_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        endpoint_release_timeout_seconds=endpoint_release_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        test_owned=test_owned,
    )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaServerRuntimeState:
    schema: str
    pid: int
    host_scope: str
    port: int
    process_start_token: str
    executable_identity: str
    server_command_identity: str
    ownership_token_hash: str
    lifecycle_state: str
    created_at: str

    def __post_init__(self) -> None:
        if self.schema != CHROMA_SERVER_RUNTIME_STATE_SCHEMA:
            raise ChromaServerStateCorrupt("unsupported_chroma_server_state_schema")
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_pid")
        if self.host_scope != "loopback":
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_host_scope")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65_535
        ):
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_port")
        if not isinstance(self.process_start_token, str) or not _START_TOKEN_RE.fullmatch(
            self.process_start_token
        ):
            raise ChromaServerStateCorrupt(
                "invalid_chroma_server_process_start_token"
            )
        for value, code in (
            (self.executable_identity, "invalid_chroma_server_executable_identity"),
            (self.server_command_identity, "invalid_chroma_server_command_identity"),
            (self.ownership_token_hash, "invalid_chroma_server_ownership_token"),
        ):
            if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
                raise ChromaServerStateCorrupt(code)
        if self.lifecycle_state not in _LIFECYCLE_STATES:
            raise ChromaServerStateCorrupt("invalid_chroma_server_lifecycle_state")
        if not isinstance(self.created_at, str) or not _CREATED_AT_RE.fullmatch(
            self.created_at
        ):
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_timestamp")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ChromaServerRuntimeState:
        if not isinstance(payload, Mapping) or frozenset(payload) != _RUNTIME_STATE_FIELDS:
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_shape")
        try:
            return cls(**dict(payload))
        except ChromaServerStateCorrupt:
            raise
        except (TypeError, ValueError):
            raise ChromaServerStateCorrupt("invalid_chroma_server_state_shape") from None

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema": self.schema,
            "pid": self.pid,
            "host_scope": self.host_scope,
            "port": self.port,
            "process_start_token": self.process_start_token,
            "executable_identity": self.executable_identity,
            "server_command_identity": self.server_command_identity,
            "ownership_token_hash": self.ownership_token_hash,
            "lifecycle_state": self.lifecycle_state,
            "created_at": self.created_at,
        }

    def mark_ready(self) -> ChromaServerRuntimeState:
        return replace(self, lifecycle_state="ready")

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "schema": self.schema,
            "host_scope": self.host_scope,
            "port": self.port,
            "lifecycle_state": self.lifecycle_state,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaServerRuntimeState("
            f"schema={summary['schema']!r}, "
            f"host_scope={summary['host_scope']!r}, "
            f"port={summary['port']!r}, "
            f"lifecycle_state={summary['lifecycle_state']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaServerLifecycleResult:
    state: str
    deployment_mode: str
    endpoint_scope: str
    port: int
    process_owned: bool
    server_reachable: bool
    detail: str
    forced_shutdown: bool = False

    def __post_init__(self) -> None:
        if self.state not in _RESULT_STATES:
            raise ChromaServerLifecycleError("invalid_chroma_server_result_state")
        if self.deployment_mode not in {
            "disabled",
            "local_http",
            "remote_http",
            "ephemeral_test",
        }:
            raise ChromaServerLifecycleError("invalid_chroma_server_result_mode")
        if self.endpoint_scope not in {"none", "loopback", "remote", "test_owned"}:
            raise ChromaServerLifecycleError("invalid_chroma_server_result_scope")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 0 <= self.port <= 65_535:
            raise ChromaServerLifecycleError("invalid_chroma_server_result_port")
        if not isinstance(self.process_owned, bool) or not isinstance(
            self.server_reachable, bool
        ):
            raise ChromaServerLifecycleError("invalid_chroma_server_result_boolean")
        if not isinstance(self.detail, str) or not self.detail or len(self.detail) > 80:
            raise ChromaServerLifecycleError("invalid_chroma_server_result_detail")
        if not isinstance(self.forced_shutdown, bool):
            raise ChromaServerLifecycleError("invalid_chroma_server_result_shutdown")

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "state": self.state,
            "deployment_mode": self.deployment_mode,
            "endpoint_scope": self.endpoint_scope,
            "port": self.port,
            "process_owned": self.process_owned,
            "server_reachable": self.server_reachable,
            "detail": self.detail,
            "forced_shutdown": self.forced_shutdown,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaServerLifecycleResult("
            f"state={summary['state']!r}, "
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"endpoint_scope={summary['endpoint_scope']!r}, "
            f"process_owned={summary['process_owned']!r}, "
            f"server_reachable={summary['server_reachable']!r}, "
            f"detail={summary['detail']!r})"
        )


class AtomicChromaServerStateStore:
    """Strict single-file runtime state store using same-directory atomic replacement."""

    __slots__ = ("_config",)

    def __init__(self, config: ChromaServerLifecycleConfig):
        if not isinstance(config, ChromaServerLifecycleConfig):
            raise InvalidChromaServerLifecycleConfiguration(
                "invalid_chroma_server_state_store_config"
            )
        self._config = config

    @property
    def state_path(self) -> Path:
        return self._config.runtime_state_path

    def _validate_runtime_directory(self) -> None:
        _require_safe_path_chain(
            self._config.information_root,
            self._config.runtime_state_directory,
            "unsafe_chroma_server_runtime_path",
        )
        if self._config.runtime_state_directory.exists() and not self._config.runtime_state_directory.is_dir():
            raise ChromaServerStateCorrupt("invalid_chroma_server_runtime_directory")

    def load(self) -> ChromaServerRuntimeState | None:
        self._validate_runtime_directory()
        path = self.state_path
        if not path.exists():
            return None
        if path.is_symlink() or _is_reparse_point(path) or not path.is_file():
            raise ChromaServerStateCorrupt("unsafe_chroma_server_state_file")
        try:
            if path.stat().st_size > MAX_RUNTIME_STATE_BYTES:
                raise ChromaServerStateCorrupt("chroma_server_state_too_large")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ChromaServerStateCorrupt:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ChromaServerStateCorrupt("chroma_server_state_unreadable") from None
        return ChromaServerRuntimeState.from_mapping(payload)

    def write(self, state: ChromaServerRuntimeState) -> None:
        if not isinstance(state, ChromaServerRuntimeState):
            raise ChromaServerStateWriteFailed("invalid_chroma_server_state_write")
        deployment_port = self._config.deployment.port
        if state.port != deployment_port:
            raise ChromaServerStateWriteFailed("chroma_server_state_port_mismatch")
        self._validate_runtime_directory()
        temporary: Path | None = None
        try:
            self._config.runtime_state_directory.mkdir(parents=True, exist_ok=True)
            self._validate_runtime_directory()
            encoded = json.dumps(
                state.to_dict(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > MAX_RUNTIME_STATE_BYTES:
                raise ChromaServerStateWriteFailed("chroma_server_state_too_large")
            temporary = self._config.runtime_state_directory / (
                f".{CHROMA_SERVER_RUNTIME_STATE_FILENAME}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            temporary = None
        except ChromaServerStateWriteFailed:
            raise
        except OSError:
            raise ChromaServerStateWriteFailed("chroma_server_state_write_failed") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear(self) -> None:
        self._validate_runtime_directory()
        path = self.state_path
        if not path.exists():
            return
        if path.is_symlink() or _is_reparse_point(path) or not path.is_file():
            raise ChromaServerStateCorrupt("unsafe_chroma_server_state_file")
        try:
            path.unlink()
        except OSError:
            raise ChromaServerStateWriteFailed("chroma_server_state_clear_failed") from None
