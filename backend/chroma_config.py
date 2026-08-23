"""Pure, fail-closed deployment configuration for Chroma access."""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


CHROMA_DEPLOYMENT_MODE_ENV = "CHROMA_DEPLOYMENT_MODE"
CHROMA_HTTP_HOST_ENV = "CHROMA_HTTP_HOST"
CHROMA_HTTP_PORT_ENV = "CHROMA_HTTP_PORT"
CHROMA_HTTP_SSL_ENV = "CHROMA_HTTP_SSL"
CHROMA_HTTP_TIMEOUT_ENV = "CHROMA_HTTP_TIMEOUT_SECONDS"

SUPPORTED_ENVIRONMENT_KEYS = frozenset(
    {
        CHROMA_DEPLOYMENT_MODE_ENV,
        CHROMA_HTTP_HOST_ENV,
        CHROMA_HTTP_PORT_ENV,
        CHROMA_HTTP_SSL_ENV,
        CHROMA_HTTP_TIMEOUT_ENV,
    }
)
HTTP_SETTING_KEYS = frozenset(
    {CHROMA_HTTP_HOST_ENV, CHROMA_HTTP_PORT_ENV, CHROMA_HTTP_SSL_ENV, CHROMA_HTTP_TIMEOUT_ENV}
)

LOOPBACK_HOST = "127.0.0.1"
MIN_CHROMA_HTTP_PORT = 1
MAX_CHROMA_HTTP_PORT = 65_535
DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS = 5.0
MIN_CHROMA_HTTP_TIMEOUT_SECONDS = 0.1
MAX_CHROMA_HTTP_TIMEOUT_SECONDS = 30.0
EXISTING_LOCAL_HTTP_PORT = 8100

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_PORT_RE = re.compile(r"^[0-9]{1,5}$")
_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_MISSING = object()


class ChromaDeploymentMode(str, Enum):
    DISABLED = "disabled"
    LOCAL_HTTP = "local_http"
    REMOTE_HTTP = "remote_http"
    EPHEMERAL_TEST = "ephemeral_test"


class ChromaConfigurationError(ValueError):
    """Deterministic configuration failure without rejected values or environment data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class UnsupportedChromaDeploymentMode(ChromaConfigurationError):
    pass


class ContradictoryChromaSettings(ChromaConfigurationError):
    pass


class UnsafeLocalChromaHost(ChromaConfigurationError):
    pass


class InvalidChromaHost(ChromaConfigurationError):
    pass


class InvalidChromaPort(ChromaConfigurationError):
    pass


class InvalidChromaTimeout(ChromaConfigurationError):
    pass


class InvalidChromaSslValue(ChromaConfigurationError):
    pass


class EphemeralTestModeNotAllowed(ChromaConfigurationError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ChromaDeploymentConfig:
    mode: ChromaDeploymentMode
    host: str | None
    port: int | None
    ssl: bool
    timeout_seconds: float

    @property
    def is_disabled(self) -> bool:
        return self.mode is ChromaDeploymentMode.DISABLED

    @property
    def uses_http(self) -> bool:
        return self.mode in {
            ChromaDeploymentMode.LOCAL_HTTP,
            ChromaDeploymentMode.REMOTE_HTTP,
            ChromaDeploymentMode.EPHEMERAL_TEST,
        }

    @property
    def is_local(self) -> bool:
        return self.mode is ChromaDeploymentMode.LOCAL_HTTP

    @property
    def is_remote(self) -> bool:
        return self.mode is ChromaDeploymentMode.REMOTE_HTTP

    @property
    def is_test_only(self) -> bool:
        return self.mode is ChromaDeploymentMode.EPHEMERAL_TEST

    def safe_summary(self) -> dict[str, str | bool | float]:
        host_scope = {
            ChromaDeploymentMode.DISABLED: "none",
            ChromaDeploymentMode.LOCAL_HTTP: "loopback",
            ChromaDeploymentMode.REMOTE_HTTP: "remote",
            ChromaDeploymentMode.EPHEMERAL_TEST: "test_owned",
        }[self.mode]
        return {
            "deployment_mode": self.mode.value,
            "transport": "http" if self.uses_http else "none",
            "host_scope": host_scope,
            "ssl_enabled": self.ssl,
            "timeout_seconds": self.timeout_seconds,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaDeploymentConfig("
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"transport={summary['transport']!r}, "
            f"host_scope={summary['host_scope']!r}, "
            f"ssl_enabled={summary['ssl_enabled']!r}, "
            f"timeout_seconds={summary['timeout_seconds']!r})"
        )


def _value_for(
    key: str,
    environment: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Any:
    if key in overrides:
        return overrides[key]
    return environment.get(key, _MISSING)


def _was_injected(
    key: str,
    provided_environment: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> bool:
    return key in overrides or (
        provided_environment is not None and key in provided_environment
    )


def _parse_mode(value: Any) -> ChromaDeploymentMode:
    if value is _MISSING or value is None:
        return ChromaDeploymentMode.DISABLED
    if not isinstance(value, str):
        raise UnsupportedChromaDeploymentMode("unsupported_chroma_deployment_mode")
    normalized = value.strip().casefold()
    if not normalized:
        return ChromaDeploymentMode.DISABLED
    try:
        return ChromaDeploymentMode(normalized)
    except ValueError as error:
        raise UnsupportedChromaDeploymentMode(
            "unsupported_chroma_deployment_mode"
        ) from error


def _parse_port(value: Any) -> int:
    if value is _MISSING or value is None or value == "":
        raise InvalidChromaPort("chroma_http_port_required")
    if isinstance(value, bool):
        raise InvalidChromaPort("invalid_chroma_http_port")
    if isinstance(value, int):
        port = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not _PORT_RE.fullmatch(normalized):
            raise InvalidChromaPort("invalid_chroma_http_port")
        port = int(normalized)
    else:
        raise InvalidChromaPort("invalid_chroma_http_port")
    if not MIN_CHROMA_HTTP_PORT <= port <= MAX_CHROMA_HTTP_PORT:
        raise InvalidChromaPort("invalid_chroma_http_port")
    return port


def _parse_ssl(value: Any, *, required: bool, default: bool) -> bool:
    if value is _MISSING or value is None or value == "":
        if required:
            raise InvalidChromaSslValue("chroma_http_ssl_required")
        return default
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise InvalidChromaSslValue("invalid_chroma_http_ssl")
    normalized = value.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise InvalidChromaSslValue("invalid_chroma_http_ssl")


def _parse_timeout(value: Any) -> float:
    if value is _MISSING or value is None or value == "":
        return DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS
    if isinstance(value, bool):
        raise InvalidChromaTimeout("invalid_chroma_http_timeout")
    if isinstance(value, (int, float)):
        timeout = float(value)
    elif isinstance(value, str):
        try:
            timeout = float(value.strip())
        except ValueError as error:
            raise InvalidChromaTimeout("invalid_chroma_http_timeout") from error
    else:
        raise InvalidChromaTimeout("invalid_chroma_http_timeout")
    if (
        not math.isfinite(timeout)
        or timeout < MIN_CHROMA_HTTP_TIMEOUT_SECONDS
        or timeout > MAX_CHROMA_HTTP_TIMEOUT_SECONDS
    ):
        raise InvalidChromaTimeout("invalid_chroma_http_timeout")
    return timeout


def _parse_local_host(value: Any) -> str:
    if not isinstance(value, str) or value.strip() != LOOPBACK_HOST:
        raise UnsafeLocalChromaHost("unsafe_local_chroma_host")
    return LOOPBACK_HOST


def _parse_remote_host(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidChromaHost("invalid_remote_chroma_host")
    host = value.strip()
    if not host or len(host) > 253:
        raise InvalidChromaHost("invalid_remote_chroma_host")
    if (
        "://" in host
        or "@" in host
        or "/" in host
        or "?" in host
        or "#" in host
        or ":" in host
        or any(character.isspace() for character in host)
    ):
        raise InvalidChromaHost("invalid_remote_chroma_host")
    if all(character.isdigit() or character == "." for character in host):
        try:
            address = ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError as error:
            raise InvalidChromaHost("invalid_remote_chroma_host") from error
        if address.is_unspecified or address.is_multicast:
            raise InvalidChromaHost("invalid_remote_chroma_host")
        return str(address)
    labels = host.split(".")
    if any(not _HOST_LABEL_RE.fullmatch(label) for label in labels):
        raise InvalidChromaHost("invalid_remote_chroma_host")
    return host.casefold()


def load_chroma_deployment_config(
    environ: Mapping[str, Any] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    test_context: bool = False,
    test_endpoint_owned: bool = False,
) -> ChromaDeploymentConfig:
    """Resolve explicit overrides, then environment values, then safe defaults."""

    environment = os.environ if environ is None else environ
    explicit = {} if overrides is None else overrides
    if not isinstance(environment, Mapping) or not isinstance(explicit, Mapping):
        raise ChromaConfigurationError("invalid_chroma_configuration_source")
    if any(key not in SUPPORTED_ENVIRONMENT_KEYS for key in explicit):
        raise ChromaConfigurationError("unknown_chroma_configuration_override")

    mode = _parse_mode(_value_for(CHROMA_DEPLOYMENT_MODE_ENV, environment, explicit))
    if mode is ChromaDeploymentMode.DISABLED:
        if any(key in explicit or key in environment for key in HTTP_SETTING_KEYS):
            raise ContradictoryChromaSettings("disabled_mode_has_http_settings")
        return ChromaDeploymentConfig(
            mode=mode,
            host=None,
            port=None,
            ssl=False,
            timeout_seconds=DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS,
        )

    host_value = _value_for(CHROMA_HTTP_HOST_ENV, environment, explicit)
    port_value = _value_for(CHROMA_HTTP_PORT_ENV, environment, explicit)
    ssl_value = _value_for(CHROMA_HTTP_SSL_ENV, environment, explicit)
    timeout_value = _value_for(CHROMA_HTTP_TIMEOUT_ENV, environment, explicit)
    port = _parse_port(port_value)
    timeout = _parse_timeout(timeout_value)

    if mode is ChromaDeploymentMode.LOCAL_HTTP:
        host = _parse_local_host(host_value)
        ssl = _parse_ssl(ssl_value, required=False, default=False)
        if ssl:
            raise InvalidChromaSslValue("local_http_ssl_must_be_disabled")
    elif mode is ChromaDeploymentMode.REMOTE_HTTP:
        host = _parse_remote_host(host_value)
        ssl = _parse_ssl(ssl_value, required=True, default=False)
    else:
        if test_context is not True or test_endpoint_owned is not True:
            raise EphemeralTestModeNotAllowed("ephemeral_test_mode_not_allowed")
        if not all(
            _was_injected(key, environ, explicit)
            for key in (CHROMA_HTTP_HOST_ENV, CHROMA_HTTP_PORT_ENV)
        ):
            raise EphemeralTestModeNotAllowed("ephemeral_test_endpoint_not_injected")
        host = _parse_local_host(host_value)
        if port == EXISTING_LOCAL_HTTP_PORT:
            raise EphemeralTestModeNotAllowed("ephemeral_test_endpoint_not_isolated")
        ssl = _parse_ssl(ssl_value, required=False, default=False)
        if ssl:
            raise InvalidChromaSslValue("ephemeral_test_ssl_must_be_disabled")

    return ChromaDeploymentConfig(
        mode=mode,
        host=host,
        port=port,
        ssl=ssl,
        timeout_seconds=timeout,
    )
