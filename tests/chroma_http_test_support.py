"""Test-owned process and loopback helpers for disposable Chroma HTTP tests."""

from __future__ import annotations

import json
import importlib.metadata
import inspect
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.chroma_collection_registry import get_collection_definition
from backend.chroma_config import (
    CHROMA_DEPLOYMENT_MODE_ENV,
    CHROMA_HTTP_HOST_ENV,
    CHROMA_HTTP_PORT_ENV,
    CHROMA_HTTP_SSL_ENV,
    CHROMA_HTTP_TIMEOUT_ENV,
    EXISTING_LOCAL_HTTP_PORT,
    LOOPBACK_HOST,
    ChromaDeploymentConfig,
    load_chroma_deployment_config,
)


PROTECTED_CHROMA_ROOT = Path(__file__).resolve().parents[1] / "information" / "chroma"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 20.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_STORAGE_PREFIX = "workagent-chroma-http-"
_SERVER_ENVIRONMENT_KEYS = frozenset(
    {
        "ANONYMIZED_TELEMETRY",
        "IS_PERSISTENT",
        "PERSIST_DIRECTORY",
    }
)


class EphemeralChromaTestInfrastructureError(RuntimeError):
    """Bounded test-infrastructure failure with a stable non-sensitive code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class EphemeralChromaStartupError(EphemeralChromaTestInfrastructureError):
    pass


class EphemeralChromaReadinessTimeout(EphemeralChromaTestInfrastructureError):
    pass


class EphemeralChromaProcessExited(EphemeralChromaTestInfrastructureError):
    pass


class EphemeralChromaShutdownError(EphemeralChromaTestInfrastructureError):
    pass


class EphemeralChromaUnsafeEndpoint(EphemeralChromaTestInfrastructureError):
    pass


class EphemeralChromaUnsafeStorage(EphemeralChromaTestInfrastructureError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class EphemeralChromaEndpoint:
    host: str
    port: int
    test_owned: bool = True

    def __post_init__(self) -> None:
        if (
            self.host != LOOPBACK_HOST
            or not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65_535
            or self.port == EXISTING_LOCAL_HTTP_PORT
            or self.test_owned is not True
        ):
            raise EphemeralChromaUnsafeEndpoint("unsafe_ephemeral_chroma_endpoint")

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "host_scope": "loopback",
            "port": self.port,
            "test_owned": self.test_owned,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "EphemeralChromaEndpoint("
            f"host_scope={summary['host_scope']!r}, "
            f"port={summary['port']!r}, "
            f"test_owned={summary['test_owned']!r})"
        )


@dataclass(frozen=True, slots=True)
class ProcessTerminationResult:
    forced: bool
    already_stopped: bool


@dataclass(frozen=True, slots=True, repr=False)
class ChromaHttpTimeoutCapability:
    installed_version: str
    http_client_parameters: tuple[str, ...]
    settings_http_fields: tuple[str, ...]
    settings_timeout_fields: tuple[str, ...]
    timeout_support: str
    public_mechanism: str
    production_migration_gate: str
    required_work_item: str

    def safe_summary(self) -> dict[str, str | tuple[str, ...]]:
        return {
            "installed_version": self.installed_version,
            "http_client_parameters": self.http_client_parameters,
            "settings_http_fields": self.settings_http_fields,
            "settings_timeout_fields": self.settings_timeout_fields,
            "timeout_support": self.timeout_support,
            "public_mechanism": self.public_mechanism,
            "production_migration_gate": self.production_migration_gate,
            "required_work_item": self.required_work_item,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaHttpTimeoutCapability("
            f"installed_version={summary['installed_version']!r}, "
            f"timeout_support={summary['timeout_support']!r}, "
            f"public_mechanism={summary['public_mechanism']!r}, "
            f"production_migration_gate={summary['production_migration_gate']!r})"
        )


def inspect_public_http_timeout_capability() -> ChromaHttpTimeoutCapability:
    """Classify only installed public signatures and public Settings fields."""

    import chromadb
    from chromadb.config import Settings

    parameters = tuple(inspect.signature(chromadb.HttpClient).parameters)
    fields = tuple(sorted(Settings.model_fields))
    http_fields = tuple(name for name in fields if "http" in name.casefold())
    timeout_fields = tuple(name for name in fields if "timeout" in name.casefold())
    general_http_timeout_fields = tuple(
        name
        for name in fields
        if "http" in name.casefold() and "timeout" in name.casefold()
    )
    supported = "timeout" in parameters or bool(general_http_timeout_fields)
    return ChromaHttpTimeoutCapability(
        installed_version=importlib.metadata.version("chromadb"),
        http_client_parameters=parameters,
        settings_http_fields=http_fields,
        settings_timeout_fields=timeout_fields,
        timeout_support=(
            "supported"
            if supported
            else "unsupported_by_current_public_client_api"
        ),
        public_mechanism="public_http_timeout_parameter" if supported else "none",
        production_migration_gate="open" if supported else "blocked",
        required_work_item="none" if supported else "Bounded Chroma HTTP Transport Adapter",
    )


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def validate_ephemeral_storage_path(
    path: str | Path,
    *,
    protected_root: str | Path = PROTECTED_CHROMA_ROOT,
) -> Path:
    """Reject the protected persistence root and anything nested below it."""

    candidate = _resolved(path)
    protected = _resolved(protected_root)
    if candidate == protected or protected in candidate.parents:
        raise EphemeralChromaUnsafeStorage("ephemeral_chroma_storage_is_protected")
    return candidate


def create_ephemeral_storage_directory(
    storage_parent: str | Path,
    *,
    protected_root: str | Path = PROTECTED_CHROMA_ROOT,
) -> Path:
    parent = validate_ephemeral_storage_path(
        storage_parent,
        protected_root=protected_root,
    )
    if not parent.is_dir():
        raise EphemeralChromaUnsafeStorage("ephemeral_chroma_storage_parent_unavailable")
    try:
        created = Path(tempfile.mkdtemp(prefix=_STORAGE_PREFIX, dir=parent))
        return validate_ephemeral_storage_path(created, protected_root=protected_root)
    except EphemeralChromaTestInfrastructureError:
        raise
    except (OSError, RuntimeError):
        raise EphemeralChromaUnsafeStorage(
            "ephemeral_chroma_storage_creation_failed"
        ) from None


def remove_ephemeral_storage_directory(
    storage_path: str | Path,
    *,
    storage_parent: str | Path,
    protected_root: str | Path = PROTECTED_CHROMA_ROOT,
) -> None:
    candidate = validate_ephemeral_storage_path(
        storage_path,
        protected_root=protected_root,
    )
    parent = validate_ephemeral_storage_path(
        storage_parent,
        protected_root=protected_root,
    )
    if candidate.parent != parent or not candidate.name.startswith(_STORAGE_PREFIX):
        raise EphemeralChromaUnsafeStorage("ephemeral_chroma_storage_not_test_owned")
    if not candidate.exists():
        return
    try:
        shutil.rmtree(candidate)
    except OSError:
        raise EphemeralChromaShutdownError(
            "ephemeral_chroma_storage_cleanup_failed"
        ) from None


def allocate_dynamic_loopback_endpoint() -> EphemeralChromaEndpoint:
    """Ask the OS for a loopback port and explicitly exclude the production port."""

    for _ in range(20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind((LOOPBACK_HOST, 0))
                port = int(listener.getsockname()[1])
        except OSError:
            raise EphemeralChromaUnsafeEndpoint(
                "ephemeral_chroma_port_allocation_failed"
            ) from None
        if port != EXISTING_LOCAL_HTTP_PORT:
            return EphemeralChromaEndpoint(host=LOOPBACK_HOST, port=port)
    raise EphemeralChromaUnsafeEndpoint("ephemeral_chroma_isolated_port_unavailable")


def is_loopback_port_releasable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((LOOPBACK_HOST, port))
        return True
    except OSError:
        return False


def wait_for_loopback_port_release(port: int, *, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_loopback_port_releasable(port):
            return True
        time.sleep(0.025)
    return is_loopback_port_releasable(port)


def ephemeral_deployment_config(
    endpoint: EphemeralChromaEndpoint,
    *,
    timeout_seconds: float = 1.0,
) -> ChromaDeploymentConfig:
    return load_chroma_deployment_config(
        {},
        overrides={
            CHROMA_DEPLOYMENT_MODE_ENV: "ephemeral_test",
            CHROMA_HTTP_HOST_ENV: endpoint.host,
            CHROMA_HTTP_PORT_ENV: str(endpoint.port),
            CHROMA_HTTP_SSL_ENV: "0",
            CHROMA_HTTP_TIMEOUT_ENV: str(timeout_seconds),
        },
        test_context=True,
        test_endpoint_owned=True,
    )


def terminate_process_bounded(
    process: Any,
    *,
    timeout_seconds: float,
) -> ProcessTerminationResult:
    """Terminate, then kill as a bounded Windows-compatible fallback."""

    if process.poll() is not None:
        return ProcessTerminationResult(forced=False, already_stopped=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
        return ProcessTerminationResult(forced=False, already_stopped=False)
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        process.wait(timeout=timeout_seconds)
        return ProcessTerminationResult(forced=True, already_stopped=False)
    except subprocess.TimeoutExpired:
        raise EphemeralChromaShutdownError(
            "ephemeral_chroma_process_shutdown_timeout"
        ) from None


def _server_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("CHROMA_")
        and key.upper() not in _SERVER_ENVIRONMENT_KEYS
    }
    environment.update(
        {
            "ANONYMIZED_TELEMETRY": "FALSE",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return environment


def _probe_chroma_readiness(endpoint: EphemeralChromaEndpoint, timeout: float) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://{endpoint.host}:{endpoint.port}/api/v2/heartbeat",
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


class EphemeralChromaServer:
    """Explicitly managed separate-process Chroma server for integration tests."""

    def __init__(
        self,
        storage_parent: str | Path,
        *,
        protected_root: str | Path = PROTECTED_CHROMA_ROOT,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        executable: str | None = None,
        process_factory: Any = subprocess.Popen,
        endpoint_allocator: Any = allocate_dynamic_loopback_endpoint,
        readiness_probe: Any = _probe_chroma_readiness,
    ) -> None:
        self._storage_parent = validate_ephemeral_storage_path(
            storage_parent,
            protected_root=protected_root,
        )
        self._protected_root = _resolved(protected_root)
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._executable = executable
        self._process_factory = process_factory
        self._endpoint_allocator = endpoint_allocator
        self._readiness_probe = readiness_probe
        self._storage_path: Path | None = None
        self._endpoint: EphemeralChromaEndpoint | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._ready = False
        self._last_termination: ProcessTerminationResult | None = None

    @property
    def endpoint(self) -> EphemeralChromaEndpoint:
        if self._endpoint is None:
            raise EphemeralChromaStartupError("ephemeral_chroma_server_not_started")
        return self._endpoint

    @property
    def storage_path(self) -> Path | None:
        return self._storage_path

    @property
    def process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def process_id(self) -> int | None:
        if not self.process_running:
            return None
        return int(self._process.pid) if self._process is not None else None

    @property
    def ready(self) -> bool:
        return self._ready and self.process_running

    @property
    def last_termination(self) -> ProcessTerminationResult | None:
        return self._last_termination

    def safe_summary(self) -> dict[str, str | int | bool]:
        endpoint = self._endpoint
        return {
            "process_state": "ready" if self.ready else "stopped",
            "host_scope": "loopback",
            "port": endpoint.port if endpoint is not None else 0,
            "test_owned": True,
            "storage_scope": "temporary_test_owned",
        }

    def _resolve_executable(self) -> str:
        executable = self._executable or shutil.which("chroma")
        if not executable:
            raise EphemeralChromaStartupError("chroma_server_executable_unavailable")
        return executable

    def _cleanup_storage(self) -> None:
        if self._storage_path is None:
            return
        path = self._storage_path
        remove_ephemeral_storage_directory(
            path,
            storage_parent=self._storage_parent,
            protected_root=self._protected_root,
        )
        self._storage_path = None

    def _abort_startup(self) -> None:
        if self._process is not None:
            self._last_termination = terminate_process_bounded(
                self._process,
                timeout_seconds=self._shutdown_timeout_seconds,
            )
            self._process = None
        self._ready = False
        self._endpoint = None
        self._cleanup_storage()

    def start(self) -> EphemeralChromaEndpoint:
        if self._process is not None or self._storage_path is not None:
            raise EphemeralChromaStartupError("ephemeral_chroma_server_already_started")
        if self._startup_timeout_seconds <= 0 or self._shutdown_timeout_seconds <= 0:
            raise EphemeralChromaStartupError("invalid_ephemeral_chroma_deadline")
        self._storage_path = create_ephemeral_storage_directory(
            self._storage_parent,
            protected_root=self._protected_root,
        )
        try:
            self._endpoint = self._endpoint_allocator()
            executable = self._resolve_executable()
            command = [
                executable,
                "run",
                "--path",
                str(self._storage_path),
                "--host",
                self._endpoint.host,
                "--port",
                str(self._endpoint.port),
            ]
            creation_flags = (
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            self._process = self._process_factory(
                command,
                cwd=self._storage_path,
                env=_server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=creation_flags,
            )
            deadline = time.monotonic() + self._startup_timeout_seconds
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise EphemeralChromaProcessExited(
                        "ephemeral_chroma_process_exited_before_ready"
                    )
                remaining = max(0.01, deadline - time.monotonic())
                if self._readiness_probe(self._endpoint, min(0.25, remaining)):
                    self._ready = True
                    return self._endpoint
                time.sleep(0.05)
            raise EphemeralChromaReadinessTimeout(
                "ephemeral_chroma_readiness_timeout"
            )
        except EphemeralChromaTestInfrastructureError:
            self._abort_startup()
            raise
        except (OSError, RuntimeError):
            self._abort_startup()
            raise EphemeralChromaStartupError(
                "ephemeral_chroma_process_start_failed"
            ) from None

    def deployment_config(self, *, timeout_seconds: float = 1.0) -> ChromaDeploymentConfig:
        return ephemeral_deployment_config(
            self.endpoint,
            timeout_seconds=timeout_seconds,
        )

    def stop(self) -> None:
        shutdown_error: EphemeralChromaShutdownError | None = None
        if self._process is not None:
            try:
                self._last_termination = terminate_process_bounded(
                    self._process,
                    timeout_seconds=self._shutdown_timeout_seconds,
                )
                self._process = None
            except EphemeralChromaShutdownError as error:
                shutdown_error = error
        self._ready = False
        self._endpoint = None
        if shutdown_error is None:
            self._cleanup_storage()
        if shutdown_error is not None:
            raise shutdown_error

    def __enter__(self) -> EphemeralChromaServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


def prepare_registered_collection_for_test(
    endpoint: EphemeralChromaEndpoint,
    semantic_collection_id: str,
    *,
    ids: list[str] | None = None,
    embeddings: list[list[float]] | None = None,
    metadatas: list[dict[str, str]] | None = None,
    documents: list[str] | None = None,
) -> Any:
    """Explicit test admin boundary; ordinary factory collection access stays non-creating."""

    definition = get_collection_definition(semantic_collection_id)
    client = _construct_public_chroma_test_client(endpoint)
    return _create_registered_collection_for_test(
        client,
        collection_name=definition.collection_name,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def _create_registered_collection_for_test(
    client: Any,
    *,
    collection_name: str,
    ids: list[str] | None,
    embeddings: list[list[float]] | None,
    metadatas: list[dict[str, str]] | None,
    documents: list[str] | None,
) -> Any:
    collection = client.create_collection(name=collection_name)
    if ids is not None:
        values: dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
        if documents is not None:
            values["documents"] = documents
        collection.add(**values)
    return collection


def read_collection_for_test(collection: Any, *, ids: list[str]) -> dict[str, Any]:
    return collection.get(ids=ids, include=["metadatas"])


def query_collection_for_test(
    collection: Any,
    *,
    query_embeddings: list[list[float]],
) -> dict[str, Any]:
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=1,
        include=["distances", "metadatas"],
    )


def construct_public_http_client_for_timeout_probe(
    endpoint: EphemeralChromaEndpoint,
    *,
    query_timeout_seconds: int,
) -> Any:
    return _construct_public_chroma_test_client(
        endpoint,
        query_timeout_seconds=query_timeout_seconds,
    )


def _construct_public_chroma_test_client(
    endpoint: EphemeralChromaEndpoint,
    *,
    query_timeout_seconds: int | None = None,
) -> Any:
    """Sole fixture-only public Chroma client constructor for temporary setup."""

    settings_kwargs: dict[str, Any] = {
        "_env_file": None,
        "anonymized_telemetry": False,
    }
    if query_timeout_seconds is not None:
        settings_kwargs["chroma_query_request_timeout_seconds"] = query_timeout_seconds
    settings = Settings(
        **settings_kwargs,
    )
    return chromadb.HttpClient(
        host=endpoint.host,
        port=endpoint.port,
        ssl=False,
        settings=settings,
    )


class _DelayedResponseHandler(BaseHTTPRequestHandler):
    server: _DelayedResponseServer

    def _respond(self) -> None:
        time.sleep(self.server.response_delay_seconds)
        body = json.dumps(
            {"error": "Unavailable", "message": "controlled delayed response"}
        ).encode("utf-8")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    do_GET = _respond
    do_POST = _respond

    def log_message(self, format: str, *args: Any) -> None:
        return None


class _DelayedResponseServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, endpoint: EphemeralChromaEndpoint, delay_seconds: float):
        self.response_delay_seconds = delay_seconds
        super().__init__((endpoint.host, endpoint.port), _DelayedResponseHandler)


class DelayedLoopbackHttpServer:
    """Controlled delayed endpoint used only to observe the real HTTP client."""

    def __init__(self, *, delay_seconds: float) -> None:
        self._delay_seconds = float(delay_seconds)
        self._endpoint: EphemeralChromaEndpoint | None = None
        self._server: _DelayedResponseServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> EphemeralChromaEndpoint:
        if self._endpoint is None:
            raise EphemeralChromaStartupError("delayed_http_server_not_started")
        return self._endpoint

    def start(self) -> EphemeralChromaEndpoint:
        if self._server is not None:
            raise EphemeralChromaStartupError("delayed_http_server_already_started")
        endpoint = allocate_dynamic_loopback_endpoint()
        self._server = _DelayedResponseServer(endpoint, self._delay_seconds)
        self._endpoint = endpoint
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="chroma-delayed-http-test",
            daemon=True,
        )
        self._thread.start()
        return endpoint

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise EphemeralChromaShutdownError(
                    "delayed_http_server_shutdown_timeout"
                )
        self._server = None
        self._thread = None
        self._endpoint = None

    def __enter__(self) -> DelayedLoopbackHttpServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
