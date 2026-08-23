from __future__ import annotations

import ast
import dataclasses
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from backend import chroma_config
from backend.chroma_config import (
    CHROMA_DEPLOYMENT_MODE_ENV,
    CHROMA_HTTP_HOST_ENV,
    CHROMA_HTTP_PORT_ENV,
    CHROMA_HTTP_SSL_ENV,
    CHROMA_HTTP_TIMEOUT_ENV,
    ChromaConfigurationError,
    ChromaDeploymentConfig,
    ChromaDeploymentMode,
    ContradictoryChromaSettings,
    EphemeralTestModeNotAllowed,
    InvalidChromaHost,
    InvalidChromaPort,
    InvalidChromaSslValue,
    InvalidChromaTimeout,
    UnsupportedChromaDeploymentMode,
    UnsafeLocalChromaHost,
    load_chroma_deployment_config,
)


ROOT = Path(__file__).resolve().parents[1]


def local_environment(**changes):
    values = {
        CHROMA_DEPLOYMENT_MODE_ENV: "local_http",
        CHROMA_HTTP_HOST_ENV: "127.0.0.1",
        CHROMA_HTTP_PORT_ENV: "8100",
        CHROMA_HTTP_SSL_ENV: "0",
        CHROMA_HTTP_TIMEOUT_ENV: "5",
    }
    values.update(changes)
    return values


def remote_environment(**changes):
    values = {
        CHROMA_DEPLOYMENT_MODE_ENV: "remote_http",
        CHROMA_HTTP_HOST_ENV: "chroma.internal.example",
        CHROMA_HTTP_PORT_ENV: "8200",
        CHROMA_HTTP_SSL_ENV: "1",
        CHROMA_HTTP_TIMEOUT_ENV: "5",
    }
    values.update(changes)
    return values


def ephemeral_environment(**changes):
    values = {
        CHROMA_DEPLOYMENT_MODE_ENV: "ephemeral_test",
        CHROMA_HTTP_HOST_ENV: "127.0.0.1",
        CHROMA_HTTP_PORT_ENV: "18123",
        CHROMA_HTTP_SSL_ENV: "0",
        CHROMA_HTTP_TIMEOUT_ENV: "1",
    }
    values.update(changes)
    return values


def load_ephemeral(values=None, **kwargs):
    return load_chroma_deployment_config(
        ephemeral_environment() if values is None else values,
        test_context=True,
        test_endpoint_owned=True,
        **kwargs,
    )


def test_missing_and_blank_mode_default_to_disabled():
    missing = load_chroma_deployment_config({})
    blank = load_chroma_deployment_config({CHROMA_DEPLOYMENT_MODE_ENV: "  "})
    assert missing == blank == ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.DISABLED,
        host=None,
        port=None,
        ssl=False,
        timeout_seconds=5.0,
    )
    assert missing.is_disabled is True
    assert missing.uses_http is False


def test_disabled_mode_is_explicitly_accepted():
    config = load_chroma_deployment_config({CHROMA_DEPLOYMENT_MODE_ENV: "DISABLED"})
    assert config.mode is ChromaDeploymentMode.DISABLED


@pytest.mark.parametrize(
    "key",
    (CHROMA_HTTP_HOST_ENV, CHROMA_HTTP_PORT_ENV, CHROMA_HTTP_SSL_ENV, CHROMA_HTTP_TIMEOUT_ENV),
)
def test_disabled_mode_rejects_every_contradictory_http_field_even_when_blank(key):
    with pytest.raises(ContradictoryChromaSettings, match="disabled_mode_has_http_settings"):
        load_chroma_deployment_config({CHROMA_DEPLOYMENT_MODE_ENV: "disabled", key: ""})


@pytest.mark.parametrize(
    "value",
    (
        "unknown",
        "local",
        "server",
        "http",
        "embedded",
        "persistent",
        "local_persistent",
        "auto",
        "default",
        "local_htp",
    ),
)
def test_unknown_misspelled_and_implicit_modes_fail_closed(value):
    with pytest.raises(
        UnsupportedChromaDeploymentMode, match="unsupported_chroma_deployment_mode"
    ):
        load_chroma_deployment_config({CHROMA_DEPLOYMENT_MODE_ENV: value})


def test_supported_modes_are_exactly_the_authoritative_four():
    assert {mode.value for mode in ChromaDeploymentMode} == {
        "disabled",
        "local_http",
        "remote_http",
        "ephemeral_test",
    }


def test_valid_local_http_is_loopback_only_and_http_without_ssl():
    config = load_chroma_deployment_config(local_environment())
    assert config.mode is ChromaDeploymentMode.LOCAL_HTTP
    assert config.host == "127.0.0.1"
    assert config.port == 8100
    assert config.ssl is False
    assert config.uses_http and config.is_local
    assert not config.is_remote and not config.is_test_only


@pytest.mark.parametrize(
    "host",
    (
        "0.0.0.0",
        "::",
        "[::]",
        "192.0.2.10",
        "localhost",
        "chroma.internal",
        "http://127.0.0.1",
        "user@127.0.0.1",
        "127.0.0.1/path",
        "127.0.0.1?query=1",
        "127.0.0.1#fragment",
    ),
)
def test_local_http_rejects_non_loopback_or_composite_hosts(host):
    with pytest.raises(UnsafeLocalChromaHost, match="unsafe_local_chroma_host"):
        load_chroma_deployment_config(local_environment(**{CHROMA_HTTP_HOST_ENV: host}))


def test_local_http_requires_explicit_host_and_port():
    missing_host = local_environment()
    missing_host.pop(CHROMA_HTTP_HOST_ENV)
    with pytest.raises(UnsafeLocalChromaHost):
        load_chroma_deployment_config(missing_host)
    missing_port = local_environment()
    missing_port.pop(CHROMA_HTTP_PORT_ENV)
    with pytest.raises(InvalidChromaPort, match="chroma_http_port_required"):
        load_chroma_deployment_config(missing_port)


def test_remote_http_accepts_safe_hostname_and_ipv4_without_lookup(monkeypatch):
    def forbidden_lookup(*_args, **_kwargs):
        raise AssertionError("configuration parsing must not resolve hosts")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_lookup)
    hostname = load_chroma_deployment_config(remote_environment())
    address = load_chroma_deployment_config(
        remote_environment(**{CHROMA_HTTP_HOST_ENV: "192.0.2.10"})
    )
    assert hostname.host == "chroma.internal.example"
    assert address.host == "192.0.2.10"
    assert hostname.is_remote and hostname.uses_http


def test_remote_http_requires_host_port_and_explicit_ssl():
    for key, error in (
        (CHROMA_HTTP_HOST_ENV, InvalidChromaHost),
        (CHROMA_HTTP_PORT_ENV, InvalidChromaPort),
        (CHROMA_HTTP_SSL_ENV, InvalidChromaSslValue),
    ):
        values = remote_environment()
        values.pop(key)
        with pytest.raises(error):
            load_chroma_deployment_config(values)


@pytest.mark.parametrize(
    "host",
    (
        "https://chroma.example",
        "user:password@chroma.example",
        "chroma.example/path",
        "chroma.example?query=1",
        "chroma.example#fragment",
        "bad host.example",
        "-bad.example",
        "bad-.example",
        "bad..example",
        "999.999.999.999",
        "0.0.0.0",
        "224.0.0.1",
        "[2001:db8::1]",
    ),
)
def test_remote_http_rejects_unsafe_host_syntax(host):
    with pytest.raises(InvalidChromaHost, match="invalid_remote_chroma_host"):
        load_chroma_deployment_config(remote_environment(**{CHROMA_HTTP_HOST_ENV: host}))


@pytest.mark.parametrize("port", (1, 65_535, "1", "65535"))
def test_port_boundaries_are_accepted(port):
    config = load_chroma_deployment_config(
        local_environment(**{CHROMA_HTTP_PORT_ENV: port})
    )
    assert config.port == int(port)


@pytest.mark.parametrize(
    "port", (0, -1, 65_536, 1.0, True, False, "1.0", "-1", "+1", "not-a-port", "")
)
def test_invalid_ports_are_rejected_without_default_replacement(port):
    with pytest.raises(InvalidChromaPort):
        load_chroma_deployment_config(local_environment(**{CHROMA_HTTP_PORT_ENV: port}))


@pytest.mark.parametrize("value", (True, "1", "true", "yes", "on", " TRUE "))
def test_supported_ssl_true_values_are_accepted_for_remote(value):
    assert load_chroma_deployment_config(
        remote_environment(**{CHROMA_HTTP_SSL_ENV: value})
    ).ssl is True


@pytest.mark.parametrize("value", (False, "0", "false", "no", "off", " FALSE "))
def test_supported_ssl_false_values_are_accepted(value):
    assert load_chroma_deployment_config(
        local_environment(**{CHROMA_HTTP_SSL_ENV: value})
    ).ssl is False


@pytest.mark.parametrize("value", ("maybe", "enabled", "2", 1, 0, object()))
def test_ambiguous_ssl_values_are_rejected(value):
    with pytest.raises(InvalidChromaSslValue, match="invalid_chroma_http_ssl"):
        load_chroma_deployment_config(remote_environment(**{CHROMA_HTTP_SSL_ENV: value}))


def test_missing_ssl_uses_false_for_local_but_is_required_for_remote():
    local = local_environment()
    local.pop(CHROMA_HTTP_SSL_ENV)
    assert load_chroma_deployment_config(local).ssl is False
    remote = remote_environment()
    remote.pop(CHROMA_HTTP_SSL_ENV)
    with pytest.raises(InvalidChromaSslValue, match="chroma_http_ssl_required"):
        load_chroma_deployment_config(remote)


def test_local_and_ephemeral_modes_reject_ssl_enablement():
    with pytest.raises(InvalidChromaSslValue, match="local_http_ssl_must_be_disabled"):
        load_chroma_deployment_config(local_environment(**{CHROMA_HTTP_SSL_ENV: "1"}))
    with pytest.raises(InvalidChromaSslValue, match="ephemeral_test_ssl_must_be_disabled"):
        load_ephemeral(ephemeral_environment(**{CHROMA_HTTP_SSL_ENV: "1"}))


@pytest.mark.parametrize("timeout", (0.1, 30.0, "0.1", "30"))
def test_timeout_boundaries_are_accepted(timeout):
    config = load_chroma_deployment_config(
        local_environment(**{CHROMA_HTTP_TIMEOUT_ENV: timeout})
    )
    assert config.timeout_seconds == float(timeout)


def test_missing_timeout_uses_the_documented_five_second_default():
    values = local_environment()
    values.pop(CHROMA_HTTP_TIMEOUT_ENV)
    assert load_chroma_deployment_config(values).timeout_seconds == 5.0
    assert load_chroma_deployment_config(
        local_environment(**{CHROMA_HTTP_TIMEOUT_ENV: ""})
    ).timeout_seconds == 5.0


@pytest.mark.parametrize(
    "timeout",
    (0.09, 30.01, 0, -1, float("nan"), float("inf"), float("-inf"), True, False, "bad"),
)
def test_invalid_timeouts_are_rejected_without_clamping_or_default_replacement(timeout):
    with pytest.raises(InvalidChromaTimeout, match="invalid_chroma_http_timeout"):
        load_chroma_deployment_config(
            local_environment(**{CHROMA_HTTP_TIMEOUT_ENV: timeout})
        )


def test_ephemeral_mode_is_rejected_without_both_explicit_authorizations():
    values = ephemeral_environment()
    with pytest.raises(EphemeralTestModeNotAllowed, match="ephemeral_test_mode_not_allowed"):
        load_chroma_deployment_config(values)
    with pytest.raises(EphemeralTestModeNotAllowed, match="ephemeral_test_mode_not_allowed"):
        load_chroma_deployment_config(values, test_context=True)


def test_ephemeral_mode_accepts_only_injected_test_owned_loopback_endpoint():
    config = load_ephemeral()
    assert config.mode is ChromaDeploymentMode.EPHEMERAL_TEST
    assert config.host == "127.0.0.1"
    assert config.port == 18123
    assert config.is_test_only and config.uses_http


def test_ephemeral_mode_rejects_non_injected_process_endpoint(monkeypatch):
    for key, value in ephemeral_environment().items():
        monkeypatch.setenv(key, str(value))
    with pytest.raises(EphemeralTestModeNotAllowed, match="ephemeral_test_endpoint_not_injected"):
        load_chroma_deployment_config(
            None,
            test_context=True,
            test_endpoint_owned=True,
        )


def test_ephemeral_mode_rejects_existing_local_endpoint_and_non_loopback_host():
    with pytest.raises(
        EphemeralTestModeNotAllowed, match="ephemeral_test_endpoint_not_isolated"
    ):
        load_ephemeral(ephemeral_environment(**{CHROMA_HTTP_PORT_ENV: "8100"}))
    with pytest.raises(UnsafeLocalChromaHost):
        load_ephemeral(ephemeral_environment(**{CHROMA_HTTP_HOST_ENV: "localhost"}))


def test_injected_environment_never_reads_process_or_user_dotenv(monkeypatch):
    monkeypatch.setenv(CHROMA_DEPLOYMENT_MODE_ENV, "remote_http")
    monkeypatch.setenv(CHROMA_HTTP_HOST_ENV, "process.example")
    assert load_chroma_deployment_config({}).is_disabled
    source = Path(chroma_config.__file__).read_text(encoding="utf-8")
    assert "load_" + "dotenv" not in source
    assert "python-" + "dotenv" not in source


def test_explicit_overrides_take_precedence_without_mutating_environment():
    environment = local_environment()
    before = dict(environment)
    config = load_chroma_deployment_config(
        environment,
        overrides={
            CHROMA_DEPLOYMENT_MODE_ENV: "remote_http",
            CHROMA_HTTP_HOST_ENV: "override.example",
            CHROMA_HTTP_PORT_ENV: "9443",
            CHROMA_HTTP_SSL_ENV: "true",
            CHROMA_HTTP_TIMEOUT_ENV: "2.5",
        },
    )
    assert config == ChromaDeploymentConfig(
        mode=ChromaDeploymentMode.REMOTE_HTTP,
        host="override.example",
        port=9443,
        ssl=True,
        timeout_seconds=2.5,
    )
    assert environment == before


def test_unknown_override_key_and_persistence_setting_fail_closed():
    with pytest.raises(ChromaConfigurationError, match="unknown_chroma_configuration_override"):
        load_chroma_deployment_config({}, overrides={"UNKNOWN_CHROMA_KEY": "value"})
    persistence_key = "CHROMA_" + "PERSIST_PATH"
    with pytest.raises(ChromaConfigurationError, match="unknown_chroma_configuration_override"):
        load_chroma_deployment_config({}, overrides={persistence_key: "synthetic"})


def test_configuration_is_immutable_and_has_deterministic_equality():
    first = load_chroma_deployment_config(local_environment())
    second = load_chroma_deployment_config(dict(local_environment()))
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.port = 9000


def test_safe_summary_is_bounded_and_allowlisted():
    config = load_chroma_deployment_config(remote_environment())
    assert config.safe_summary() == {
        "deployment_mode": "remote_http",
        "transport": "http",
        "host_scope": "remote",
        "ssl_enabled": True,
        "timeout_seconds": 5.0,
    }
    serialized = json.dumps(config.safe_summary(), sort_keys=True)
    assert config.host not in serialized
    assert str(config.port) not in serialized


def test_repr_is_redacted_and_model_has_no_path_client_or_collection_fields():
    config = load_chroma_deployment_config(
        remote_environment(**{CHROMA_HTTP_HOST_ENV: "private.internal.example"})
    )
    rendered = repr(config)
    assert "private.internal.example" not in rendered
    assert "8200" not in rendered
    field_names = {field.name for field in dataclasses.fields(ChromaDeploymentConfig)}
    assert field_names == {"mode", "host", "port", "ssl", "timeout_seconds"}
    assert not any(token in field_names for token in {"path", "client", "collection"})


def test_errors_never_echo_environment_secrets_or_absolute_paths():
    unsafe = remote_environment(
        **{CHROMA_HTTP_HOST_ENV: "https://user:secret@C:/Users/example/private"}
    )
    with pytest.raises(InvalidChromaHost) as captured:
        load_chroma_deployment_config(unsafe)
    rendered = str(captured.value)
    assert rendered == "invalid_remote_chroma_host"
    assert "secret" not in rendered.casefold()
    assert "C:/" not in rendered
    assert "CHROMA_HTTP" not in rendered


def test_configuration_module_import_is_side_effect_free_in_fresh_interpreter():
    script = (
        "import sys; import backend.chroma_config as c; "
        "assert not any(n.startswith('chromadb') for n in sys.modules); "
        "assert c.load_chroma_deployment_config({}).is_disabled"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_parser_has_no_chroma_network_collection_or_filesystem_operations():
    source = Path(chroma_config.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "chromadb" not in imported_roots
    assert "dotenv" not in imported_roots
    assert "pathlib" not in imported_roots
    assert "socket" not in imported_roots
    assert not called_attributes & {
        "PersistentClient",
        "HttpClient",
        "heartbeat",
        "get_collection",
        "get_or_create_collection",
        "query",
        "open",
        "read_text",
    }


def test_disabled_parsing_performs_no_network_or_filesystem_io(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("I/O is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert load_chroma_deployment_config({}).is_disabled


def test_tracked_configuration_names_are_semantic_and_backend_only():
    sources = [Path(chroma_config.__file__), Path(__file__)]
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in sources).casefold()
    assert "phase" + "6_5" not in serialized
    assert "phase_" + "6_5" not in serialized
    assert "step" + "2" not in serialized
    assert "frontend/" + "src" not in serialized


def test_access_inventory_remains_synchronized_with_ephemeral_http_tests():
    from backend.chroma_access_inventory import inspect_repository

    report = inspect_repository(ROOT)
    assert report["status"] == "verified"
    assert report["discovered_count"] == report["classified_count"]
    assert report["unknown_count"] == 0
    assert report["forbidden_count"] == 0
    assert report["review_candidates"] == []
