from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from backend import chroma_persistence_guard as persistence_guard
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_persistence_guard import (
    ChromaPersistenceGuard,
    inspect_persistent_client_access,
    verify_persistent_client_access,
)
from backend.chroma_persistence_guard_models import (
    ChromaEmbeddedProductionAccessDenied,
    ChromaMaintenanceAccessNotApproved,
    ChromaPersistenceContext,
    ChromaPersistenceOwnershipAmbiguous,
    ChromaPersistenceRole,
    ChromaProtectedPathAccessDenied,
    InvalidChromaPersistenceContext,
)
from backend.chroma_server_lifecycle_models import (
    ChromaServerLifecycleResult,
    build_chroma_server_lifecycle_config,
)


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ROOT / "information" / "chroma"


def deployment(
    mode: ChromaDeploymentMode = ChromaDeploymentMode.LOCAL_HTTP,
    *,
    port: int = 18140,
) -> ChromaDeploymentConfig:
    if mode is ChromaDeploymentMode.DISABLED:
        return ChromaDeploymentConfig(mode, None, None, False, 0.2)
    if mode is ChromaDeploymentMode.REMOTE_HTTP:
        return ChromaDeploymentConfig(mode, "remote.example", port, True, 0.2)
    return ChromaDeploymentConfig(mode, "127.0.0.1", port, False, 0.2)


def production_guard(
    mode: ChromaDeploymentMode = ChromaDeploymentMode.LOCAL_HTTP,
) -> ChromaPersistenceGuard:
    return ChromaPersistenceGuard(
        build_chroma_server_lifecycle_config(deployment(mode))
    )


def lifecycle_config(tmp_path: Path, *, port: int = 18140):
    information = tmp_path / "information"
    information.mkdir()
    return build_chroma_server_lifecycle_config(
        deployment(port=port),
        information_root=information,
        persistence_path=information / "chroma-test",
        runtime_state_directory=information / "runtime" / "chroma",
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        endpoint_release_timeout_seconds=1.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )


def lifecycle_result(state: str, *, owned: bool = False, reachable: bool = False):
    return ChromaServerLifecycleResult(
        state=state,
        deployment_mode="local_http",
        endpoint_scope="loopback",
        port=18140,
        process_owned=owned,
        server_reachable=reachable,
        detail="synthetic_ownership_state",
    )


def production_context(mode=ChromaDeploymentMode.LOCAL_HTTP):
    return ChromaPersistenceContext(
        role=ChromaPersistenceRole.PRODUCTION_CLIENT,
        deployment=deployment(mode),
    )


def explicit_test_context(tmp_path: Path, *, port: int = 18140):
    return ChromaPersistenceContext.test_owned(
        deployment(ChromaDeploymentMode.EPHEMERAL_TEST, port=port),
        storage_root=tmp_path,
    )


@pytest.mark.parametrize(
    "candidate",
    (
        PROTECTED,
        Path("information") / "chroma",
        Path("information") / "." / "chroma" / ".." / "chroma",
        PROTECTED / "index-data",
    ),
)
def test_protected_path_equivalents_and_children_are_recognized(candidate):
    decision = production_guard().evaluate_embedded_access(
        path=candidate,
        context=production_context(),
    )
    assert decision.allowed is False
    assert decision.persistence_scope == "server_owned"
    assert decision.reason == "local_persistence_is_server_owned"


@pytest.mark.skipif(os.name != "nt", reason="Windows case normalization contract")
def test_windows_case_variation_cannot_bypass_protected_path_guard():
    decision = production_guard().evaluate_embedded_access(
        path=str(PROTECTED.resolve()).swapcase(),
        context=production_context(),
    )
    assert decision.persistence_scope == "server_owned"
    assert decision.allowed is False


def test_existing_symlink_alias_to_protected_path_is_rejected(tmp_path):
    alias = tmp_path / "protected-alias"
    try:
        alias.symlink_to(PROTECTED, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    decision = production_guard().evaluate_embedded_access(
        path=alias,
        context=production_context(),
    )
    assert decision.persistence_scope == "server_owned"
    assert decision.allowed is False


def test_explicit_test_context_allows_only_isolated_child_path(tmp_path):
    context = explicit_test_context(tmp_path)
    guard = production_guard(ChromaDeploymentMode.EPHEMERAL_TEST)
    allowed = guard.assert_embedded_access_allowed(
        path=tmp_path / "chroma-test",
        context=context,
    )
    assert allowed.allowed is True
    assert allowed.persistence_scope == "test_owned"
    with pytest.raises(ChromaEmbeddedProductionAccessDenied):
        guard.assert_embedded_access_allowed(path=Path(PROTECTED.anchor), context=context)
    with pytest.raises(InvalidChromaPersistenceContext):
        ChromaPersistenceContext.test_owned(
            deployment(ChromaDeploymentMode.EPHEMERAL_TEST),
            storage_root=Path(PROTECTED.anchor),
        )


@pytest.mark.parametrize(
    "mode,expected_reason",
    (
        (ChromaDeploymentMode.DISABLED, "disabled_mode_has_no_embedded_fallback"),
        (ChromaDeploymentMode.LOCAL_HTTP, "local_persistence_is_server_owned"),
        (ChromaDeploymentMode.REMOTE_HTTP, "remote_mode_forbids_local_embedded_access"),
        (
            ChromaDeploymentMode.EPHEMERAL_TEST,
            "protected_persistence_forbidden_in_test_mode",
        ),
    ),
)
def test_every_deployment_mode_blocks_protected_embedded_access(
    mode, expected_reason
):
    context = ChromaPersistenceContext(
        role=ChromaPersistenceRole.PRODUCTION_CLIENT,
        deployment=deployment(mode),
    )
    guard = production_guard(mode)
    with pytest.raises(ChromaProtectedPathAccessDenied, match=expected_reason):
        guard.assert_embedded_access_allowed(path=PROTECTED, context=context)


def test_ephemeral_test_context_still_rejects_production_persistence(tmp_path):
    with pytest.raises(ChromaProtectedPathAccessDenied):
        production_guard(ChromaDeploymentMode.EPHEMERAL_TEST).assert_embedded_access_allowed(
            path=PROTECTED,
            context=explicit_test_context(tmp_path),
        )


def test_arbitrary_server_owner_role_cannot_authorize_embedded_access():
    context = ChromaPersistenceContext(
        role=ChromaPersistenceRole.SERVER_OWNER,
        deployment=deployment(),
    )
    with pytest.raises(
        ChromaProtectedPathAccessDenied,
        match="server_owner_role_cannot_authorize_embedded_access",
    ):
        production_guard().assert_embedded_access_allowed(
            path=PROTECTED,
            context=context,
        )


def test_production_lifecycle_observer_cannot_be_replaced_by_a_caller():
    with pytest.raises(
        InvalidChromaPersistenceContext,
        match="lifecycle_observer_injection_requires_test_ownership",
    ):
        ChromaPersistenceGuard(
            build_chroma_server_lifecycle_config(deployment()),
            test_lifecycle_observer=lambda _config: lifecycle_result(
                "ready", owned=True, reachable=True
            ),
        )


@pytest.mark.parametrize(
    "state,owned,reachable,allowed,error",
    (
        ("not_running", False, False, True, None),
        ("ready", True, True, False, ChromaProtectedPathAccessDenied),
        ("starting", True, True, False, ChromaProtectedPathAccessDenied),
        ("unhealthy", True, False, False, ChromaProtectedPathAccessDenied),
        ("stale_state", False, False, False, ChromaPersistenceOwnershipAmbiguous),
        (
            "foreign_port_conflict",
            False,
            False,
            False,
            ChromaPersistenceOwnershipAmbiguous,
        ),
        (
            "ownership_mismatch",
            False,
            False,
            False,
            ChromaPersistenceOwnershipAmbiguous,
        ),
    ),
)
def test_test_owned_target_uses_lifecycle_authority_and_fails_closed(
    tmp_path, state, owned, reachable, allowed, error
):
    config = lifecycle_config(tmp_path)
    observer_calls = []

    def observer(received):
        observer_calls.append(received)
        return lifecycle_result(state, owned=owned, reachable=reachable)

    guard = ChromaPersistenceGuard(config, test_lifecycle_observer=observer)
    context = explicit_test_context(
        config.information_root, port=config.deployment.port
    )
    if allowed:
        decision = guard.assert_embedded_access_allowed(
            path=config.persistence_path,
            context=context,
        )
        assert decision.disposition == "approved_test_only"
    else:
        with pytest.raises(error):
            guard.assert_embedded_access_allowed(
                path=config.persistence_path,
                context=context,
            )
    assert observer_calls == [config]


def test_verified_server_owner_requires_ready_lifecycle_proof(tmp_path):
    config = lifecycle_config(tmp_path)
    ready = ChromaPersistenceGuard(
        config,
        test_lifecycle_observer=lambda _config: lifecycle_result(
            "ready", owned=True, reachable=True
        ),
    )
    decision = ready.verify_dedicated_server_owner()
    assert decision.allowed is True
    assert decision.disposition == "verified_server_owner"
    stopped = ChromaPersistenceGuard(
        config,
        test_lifecycle_observer=lambda _config: lifecycle_result("not_running"),
    )
    with pytest.raises(ChromaPersistenceOwnershipAmbiguous):
        stopped.verify_dedicated_server_owner()


def test_safe_status_is_bounded_and_contains_no_path(tmp_path):
    config = lifecycle_config(tmp_path)
    guard = ChromaPersistenceGuard(
        config,
        test_lifecycle_observer=lambda _config: lifecycle_result(
            "ready", owned=True, reachable=True
        ),
    )
    summary = guard.inspect_status().safe_summary()
    encoded = json.dumps(summary, sort_keys=True)
    assert summary == {
        "production_persistence_role": "server_owned",
        "embedded_production_access": "blocked",
        "server_ownership_state": "ready",
        "legacy_embedded_targets": 0,
    }
    assert str(tmp_path) not in encoded
    assert "pid" not in encoded.casefold()


@pytest.mark.parametrize(
    "role",
    (ChromaPersistenceRole.MAINTENANCE, ChromaPersistenceRole.MIGRATION),
)
def test_maintenance_and_migration_labels_never_grant_access(role):
    context = ChromaPersistenceContext(
        role=role,
        deployment=deployment(),
        operator_invoked=True,
    )
    with pytest.raises(ChromaMaintenanceAccessNotApproved):
        production_guard().assert_embedded_access_allowed(
            path=PROTECTED,
            context=context,
        )


def test_memory_store_has_no_embedded_constructor_or_collection_creation():
    source = (ROOT / "backend" / "memory_store.py").read_text(encoding="utf-8")
    assert "import " + "chromadb" not in source
    assert "Persistent" + "Client" not in source
    assert "get_or_create_" + "collection" not in source
    assert "create_" + "collection" not in source
    assert "_ensure_" + "client" not in source


def test_protected_rejection_needs_no_lifecycle_or_chroma_io(monkeypatch):
    calls = []

    def forbidden_observer(_config):
        calls.append("lifecycle")
        raise AssertionError("protected production rejection must not probe Chroma")

    monkeypatch.setattr(
        persistence_guard,
        "_default_lifecycle_observer",
        forbidden_observer,
    )
    guard = production_guard()
    with pytest.raises(ChromaProtectedPathAccessDenied):
        guard.assert_embedded_access_allowed(
            path=PROTECTED,
            context=production_context(),
        )
    assert calls == []


def _write_source(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_static_guard_reports_exact_repository_dispositions():
    report = verify_persistent_client_access(ROOT)
    assert report.safe_summary() == {
        "schema": "chroma_persistent_client_guard.v1",
        "production_legacy_persistent_client_count": 0,
        "test_only_persistent_client_count": 1,
        "approved_maintenance_persistent_client_count": 0,
        "forbidden_persistent_client_count": 0,
        "unknown_persistent_client_count": 0,
        "embedded_fallback_candidate_count": 0,
    }
    assert {item.disposition for item in report.records} == {"approved_test_only"}
    assert report.records[0].module == "tests/chroma_persistence_test_support.py"


@pytest.mark.parametrize(
    "source",
    (
        "import chromadb\nclient = chromadb.PersistentClient(path='x')\n",
        (
            "from chromadb import PersistentClient as LocalClient\n"
            "client = LocalClient(path='x')\n"
        ),
        (
            "import chromadb\nfactory = chromadb.PersistentClient\n"
            "def create():\n    return factory(path='x')\n"
        ),
        (
            "import chromadb\ndef create():\n"
            "    return chromadb.PersistentClient(path='x')\n"
        ),
    ),
)
def test_new_production_direct_alias_and_wrapper_construction_is_forbidden(
    tmp_path, source
):
    _write_source(tmp_path, "backend/new_client.py", source)
    report = inspect_persistent_client_access(tmp_path)
    assert report.safe_summary()["forbidden_persistent_client_count"] == 1


def test_unknown_constructor_name_fails_closed(tmp_path):
    _write_source(
        tmp_path,
        "backend/unknown.py",
        "def connect(PersistentClient):\n    return PersistentClient(path='x')\n",
    )
    report = inspect_persistent_client_access(tmp_path)
    assert report.safe_summary()["unknown_persistent_client_count"] == 1


def test_classified_test_only_constructor_is_distinguished(tmp_path):
    _write_source(
        tmp_path,
        "tests/chroma_persistence_test_support.py",
        (
            "import chromadb\n"
            "def create_test_owned_persistent_client(tmp_path):\n"
            "    return chromadb.PersistentClient(path=str(tmp_path / 'chroma'))\n"
        ),
    )
    report = inspect_persistent_client_access(tmp_path)
    assert report.safe_summary()["test_only_persistent_client_count"] == 1
    assert report.safe_summary()["forbidden_persistent_client_count"] == 0


def test_static_guard_and_fallback_audit_are_deterministic(tmp_path):
    _write_source(
        tmp_path,
        "backend/fallback.py",
        (
            "import chromadb\n"
            "def connect():\n"
            "    try:\n        return use_http()\n"
            "    except Exception:\n"
            "        return chromadb.PersistentClient(path='x')\n"
        ),
    )
    first = inspect_persistent_client_access(tmp_path)
    second = inspect_persistent_client_access(tmp_path)
    assert first == second
    assert first.safe_summary()["forbidden_persistent_client_count"] == 1
    assert first.safe_summary()["embedded_fallback_candidate_count"] == 1


def test_http_boundaries_contain_no_embedded_constructor_or_fallback():
    report = inspect_persistent_client_access(ROOT)
    assert report.fallback_candidates == ()
    for relative in (
        "backend/chroma_http_client_factory.py",
        "backend/chroma_http_transport.py",
        "backend/chroma_http_vector_search.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "Persistent" + "Client" not in source
