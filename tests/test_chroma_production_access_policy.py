from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend import chroma_access_inventory as access_inventory
from backend.chroma_access_manifest import INVENTORY
from backend.chroma_access_models import (
    ChromaAccessValidationError,
    build_enforced_inventory,
    production_access_policy_violations,
    stable_access_id,
    validate_production_access_policy,
)
from backend.chroma_persistence_guard import verify_persistent_client_access


ROOT = Path(__file__).resolve().parents[1]


def _repository(tmp_path: Path, relative: str, source: str) -> Path:
    root = tmp_path / "repository"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return root


def _production_record(item: dict, **changes) -> dict:
    record = {
        **item,
        "runtime": "production",
        "lifecycle": "read",
        "access_mode": "read_only",
        "may_create_collection": item["operation"]
        in {"create collection", "get or create collection"},
        "may_mutate_records": False,
        "storage_internal_mutation_risk": (
            "server_owned" if item["client_type"] == "http" else "local_process"
        ),
        "current_owner": "bounded_chroma_http_transport",
        "migration_target": "central_http_client",
        "later_work_item": "deprecation_guard",
        "current_state": "synthetic_reviewed",
        "migration_action": "reject_forbidden_access",
        "notes": "Synthetic policy fixture.",
        **changes,
    }
    record["access_id"] = stable_access_id(record)
    return record


def _scan_records(tmp_path: Path, relative: str, source: str) -> list[dict]:
    scan = access_inventory.scan_repository(_repository(tmp_path, relative, source))
    assert scan["review_candidates"] == []
    return [_production_record(item) for item in scan["discoveries"]]


@pytest.mark.parametrize(
    "source",
    (
        "import chromadb\nchromadb.PersistentClient(path='temporary')\n",
        "from chromadb import PersistentClient as Client\nClient(path='temporary')\n",
        "import chromadb\nClient = chromadb.PersistentClient\nClient(path='temporary')\n",
    ),
)
def test_production_persistent_client_direct_alias_and_factory_forms_fail(tmp_path, source):
    records = _scan_records(tmp_path, "backend/example.py", source)
    categories = {item["category"] for item in production_access_policy_violations(records)}
    assert "production_embedded_client" in categories
    with pytest.raises(ChromaAccessValidationError, match="forbidden_chroma_production_access"):
        build_enforced_inventory(records)


@pytest.mark.parametrize(
    "source",
    (
        "import chromadb\nchromadb.HttpClient(host='loopback')\n",
        "from chromadb import HttpClient as Client\nClient(host='loopback')\n",
    ),
)
def test_production_direct_chromadb_http_client_and_alias_fail(tmp_path, source):
    records = _scan_records(tmp_path, "backend/example.py", source)
    categories = {item["category"] for item in production_access_policy_violations(records)}
    assert "production_direct_chromadb_http_client" in categories


def test_manifest_owner_label_cannot_disguise_direct_chromadb_http_client(tmp_path):
    records = _scan_records(
        tmp_path,
        "backend/chroma_http_transport.py",
        (
            "import chromadb\n"
            "def _default_httpx_client_builder():\n"
            "    return chromadb.HttpClient(host='loopback')\n"
        ),
    )
    categories = {item["category"] for item in production_access_policy_violations(records)}
    assert "production_direct_chromadb_http_client" in categories
    assert "production_direct_client_construction" in categories


@pytest.mark.parametrize(
    "source",
    (
        "import httpx\nhttpx.AsyncClient()\n",
        "import requests\nrequests.get('http://loopback')\n",
        "from urllib.request import urlopen\nurlopen('http://loopback')\n",
    ),
)
def test_independent_chroma_http_clients_and_requests_fail(tmp_path, source):
    records = _scan_records(tmp_path, "backend/chroma_consumer.py", source)
    categories = {item["category"] for item in production_access_policy_violations(records)}
    assert categories & {
        "production_direct_client_construction",
        "production_independent_http_access",
    }


def test_unrelated_httpx_usage_is_not_misclassified_as_chroma_access(tmp_path):
    root = _repository(
        tmp_path,
        "backend/unrelated_mail_client.py",
        "import httpx\ndef send():\n    return httpx.Client()\n",
    )
    assert access_inventory.scan_repository(root)["discoveries"] == []
    assert access_inventory.scan_production_source_policy(root) == []


@pytest.mark.parametrize("method", ("create_collection", "get_or_create_collection"))
def test_production_collection_creation_direct_and_aliased_calls_fail(tmp_path, method):
    records = _scan_records(
        tmp_path,
        "backend/chroma_consumer.py",
        (
            "import chromadb\n"
            "client = chromadb.HttpClient(host='loopback')\n"
            f"create = client.{method}\n"
            "create('facts')\n"
        ),
    )
    operations = {item["operation"] for item in records}
    expected = method.replace("_", " ")
    assert expected in operations
    categories = {item["category"] for item in production_access_policy_violations(records)}
    assert "production_collection_creation" in categories


def test_test_only_embedded_boundary_remains_explicitly_valid():
    validate_production_access_policy(INVENTORY)
    embedded = [
        item for item in INVENTORY["records"] if item["client_type"] == "persistent_embedded"
    ]
    assert embedded
    assert {item["runtime"] for item in embedded} == {"test_only"}
    assert {item["module"] for item in embedded} == {
        "tests/chroma_persistence_test_support.py"
    }
    assert sum(item["operation"] == "client construction" for item in embedded) == 1


def test_test_only_label_cannot_expand_embedded_helper_authority():
    original = next(
        item
        for item in INVENTORY["records"]
        if item["client_type"] == "persistent_embedded"
        and item["operation"] == "query"
    )
    expanded = {**original, "operation": "delete", "semantic_role": "delete"}
    expanded["access_id"] = stable_access_id(expanded)
    categories = {
        item["category"] for item in production_access_policy_violations([expanded])
    }
    assert categories == {"test_embedded_boundary_violation"}


def test_production_source_cannot_import_test_persistence_helper(tmp_path):
    root = _repository(
        tmp_path,
        "backend/chroma_consumer.py",
        "from tests.chroma_persistence_test_support import create_test_owned_persistent_client\n",
    )
    violations = access_inventory.scan_production_source_policy(root)
    assert {item["category"] for item in violations} == {
        "production_imports_test_persistence_helper"
    }


def test_semantic_client_cannot_construct_transport_or_low_level_session(tmp_path):
    root = _repository(
        tmp_path,
        "backend/chroma_read_client.py",
        (
            "import httpx\n"
            "from backend.chroma_http_client_factory import ChromaHttpClientFactory\n"
            "from backend.chroma_http_transport import BoundedChromaHttpTransport\n"
            "def read():\n"
            "    BoundedChromaHttpTransport(None)\n"
            "    return httpx.Client()\n"
        ),
    )
    categories = {
        item["category"] for item in access_inventory.scan_production_source_policy(root)
    }
    assert "independent_chroma_http_dependency" in categories
    assert "semantic_client_bypasses_factory" in categories


def test_current_repository_enforces_semantic_closure_without_fixed_total():
    report = access_inventory.inspect_repository(ROOT)
    assert report["status"] == "verified"
    assert report["discovered_count"] == report["classified_count"]
    assert report["unknown_count"] == 0
    assert report["forbidden_count"] == 0
    assert report["policy_violations"] == []
    assert not any(
        item["runtime"] != "test_only"
        and item["client_type"] in {"persistent_embedded", "ephemeral_embedded"}
        for item in INVENTORY["records"]
    )
    assert not any(
        item["runtime"] != "test_only"
        and item["operation"] in {"create collection", "get or create collection"}
        for item in INVENTORY["records"]
    )


def test_only_bounded_transport_constructs_a_production_low_level_http_client():
    constructors = [
        item
        for item in INVENTORY["records"]
        if item["runtime"] != "test_only" and item["operation"] == "client construction"
    ]
    assert {
        (item["module"], item["symbol"], item["semantic_role"], item["current_owner"])
        for item in constructors
    } == {
        (
            "backend/chroma_http_transport.py",
            "_default_httpx_client_builder",
            "low_level_http_client",
            "bounded_chroma_http_transport",
        )
    }


def test_semantic_clients_import_factory_and_do_not_own_forbidden_dependencies():
    for relative in (
        "backend/chroma_operational_reader.py",
        "backend/chroma_read_client.py",
        "backend/chroma_write_client.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "backend.chroma_http_client_factory" in imports
        assert not imports & {"chromadb", "httpx", "requests", "urllib.request"}
    assert access_inventory.scan_production_source_policy(ROOT) == []


def test_policy_diagnostics_are_bounded_source_metadata_only(tmp_path):
    records = _scan_records(
        tmp_path,
        "backend/example.py",
        "import chromadb\nchromadb.PersistentClient(path='private-value')\n",
    )
    violations = production_access_policy_violations(records)
    assert violations
    assert set(violations[0]) == {"module", "line", "symbol", "category"}
    encoded = repr(violations)
    assert str(tmp_path) not in encoded
    assert "private-value" not in encoded


def test_persistent_and_fallback_static_guard_remains_closed():
    summary = verify_persistent_client_access(ROOT).safe_summary()
    assert summary["production_legacy_persistent_client_count"] == 0
    assert summary["forbidden_persistent_client_count"] == 0
    assert summary["unknown_persistent_client_count"] == 0
    assert summary["embedded_fallback_candidate_count"] == 0
    assert summary["test_only_persistent_client_count"] == 1
