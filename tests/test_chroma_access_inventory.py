from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from backend import chroma_access_inventory as inventory
from backend import chroma_migration_baseline as baseline
from backend.chroma_access_manifest import INVENTORY
from backend.chroma_access_models import (
    CHROMA_ACCESS_INVENTORY_SCHEMA,
    ChromaAccessValidationError,
    build_enforced_inventory,
    build_inventory,
    production_access_policy_violations,
    stable_access_id,
    validate_chroma_access_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def repository_with_source(tmp_path: Path, source: str, relative: str = "backend/example.py") -> Path:
    root = tmp_path / "repository"
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return root


def discovered(tmp_path: Path, source: str, relative: str = "backend/example.py") -> dict:
    report = inventory.scan_repository(repository_with_source(tmp_path, source, relative))
    assert report["review_candidates"] == []
    return report


def reviewed_record(item: dict, **changes) -> dict:
    record = {
        **item,
        "runtime": "production",
        "lifecycle": "read",
        "access_mode": "read_only",
        "may_create_collection": False,
        "may_mutate_records": False,
        "storage_internal_mutation_risk": (
            "server_owned" if item["client_type"] == "http" else "local_process"
        ),
        "current_owner": "synthetic_owner",
        "migration_target": "central_http_client",
        "later_work_item": "central_http_client",
        "current_state": "synthetic_reviewed",
        "migration_action": "review_later",
        "notes": "Synthetic reviewed classification.",
        **changes,
    }
    record["access_id"] = stable_access_id(record)
    return record


def test_direct_chromadb_persistent_client_detection(tmp_path):
    scan = discovered(tmp_path, "import chromadb\nclient = chromadb.PersistentClient(path='tmp')\n")
    assert [(item["client_type"], item["operation"]) for item in scan["discoveries"]] == [
        ("persistent_embedded", "client construction")
    ]


def test_imported_persistent_client_alias_detection(tmp_path):
    scan = discovered(
        tmp_path,
        "from chromadb import PersistentClient as LocalClient\n"
        "client = LocalClient(path='tmp')\n",
    )
    assert scan["discoveries"][0]["client_type"] == "persistent_embedded"


def test_direct_chromadb_http_client_detection(tmp_path):
    scan = discovered(tmp_path, "import chromadb as cdb\nclient = cdb.HttpClient(host='local')\n")
    assert scan["discoveries"][0]["client_type"] == "http"


def test_imported_http_client_alias_detection(tmp_path):
    scan = discovered(
        tmp_path,
        "from chromadb import HttpClient as RemoteClient\nclient = RemoteClient(host='local')\n",
    )
    assert scan["discoveries"][0]["client_type"] == "http"


def test_constructor_alias_and_wrapper_factory_are_detected(tmp_path):
    scan = discovered(
        tmp_path,
        "import chromadb\n"
        "client_factory = chromadb.PersistentClient\n"
        "def get_vector_store():\n    return client_factory(path='tmp')\n",
    )
    item = scan["discoveries"][0]
    assert item["symbol"] == "get_vector_store"
    assert item["client_type"] == "persistent_embedded"


def test_collection_operations_require_and_preserve_receiver_provenance(tmp_path):
    scan = discovered(
        tmp_path,
        "import chromadb\n"
        "COLLECTION = 'facts'\n"
        "client = chromadb.PersistentClient(path='tmp')\n"
        "collection = client.get_collection(name=COLLECTION)\n"
        "collection.count()\ncollection.query(query_texts=['x'])\ncollection.get()\n",
    )
    operations = {item["operation"] for item in scan["discoveries"]}
    assert operations == {"client construction", "get collection", "count", "query", "get"}
    collection_entries = [item for item in scan["discoveries"] if item["collection"] == "facts"]
    assert collection_entries
    assert {item["collection_resolution"] for item in collection_entries} == {"shared_constant"}


def test_unrelated_mapping_and_service_methods_are_not_false_positives(tmp_path):
    scan = discovered(
        tmp_path,
        "import chromadb\n"
        "client = chromadb.HttpClient(host='local')\n"
        "payload = {}\npayload.get('x')\n"
        "service.query()\nqueue.add('x')\ncache.delete('x')\n",
    )
    assert [item["operation"] for item in scan["discoveries"]] == ["client construction"]


def test_unresolved_chroma_like_receiver_becomes_review_candidate(tmp_path):
    root = repository_with_source(
        tmp_path,
        "import chromadb\ndef run(unknown_collection):\n    unknown_collection.query()\n",
    )
    scan = inventory.scan_repository(root)
    assert scan["discoveries"] == []
    assert scan["review_candidates"][0]["reason"] == "unresolved_receiver_provenance"


def test_indirect_helper_collection_lookup_becomes_review_candidate(tmp_path):
    root = repository_with_source(
        tmp_path,
        "import chromadb\ndef run(helper):\n    return helper.get_collection('facts')\n",
    )
    scan = inventory.scan_repository(root)
    assert scan["discoveries"] == []
    assert scan["review_candidates"][0]["reason"] == "unresolved_receiver_provenance"


def test_access_ids_are_deterministic_and_exclude_line_numbers(tmp_path):
    first = discovered(tmp_path / "one", "import chromadb\nchromadb.HttpClient(host='local')\n")
    second = discovered(
        tmp_path / "two", "# inserted line\n# another line\nimport chromadb\nchromadb.HttpClient(host='local')\n"
    )
    assert first["discoveries"][0]["line"] != second["discoveries"][0]["line"]
    assert first["discoveries"][0]["access_id"] == second["discoveries"][0]["access_id"]


def test_discovery_sorting_is_deterministic(tmp_path):
    root = repository_with_source(
        tmp_path,
        "import chromadb\nb = chromadb.HttpClient(host='b')\na = chromadb.PersistentClient(path='a')\n",
    )
    first = inventory.scan_repository(root)["discoveries"]
    second = inventory.scan_repository(root)["discoveries"]
    assert first == second
    assert [item["access_id"] for item in first] == sorted(item["access_id"] for item in first)


@pytest.mark.parametrize("runtime", ("production", "maintenance_only", "migration_only", "test_only"))
def test_runtime_classifications_validate(tmp_path, runtime):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    validate_chroma_access_inventory(build_inventory([reviewed_record(item, runtime=runtime)]))


@pytest.mark.parametrize(
    "lifecycle", ("read", "vector_query", "write", "index", "migration", "maintenance", "test_only")
)
def test_lifecycle_classifications_validate(tmp_path, lifecycle):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    validate_chroma_access_inventory(build_inventory([reviewed_record(item, lifecycle=lifecycle)]))


def test_operation_and_collection_resolution_validation(tmp_path):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    invalid_operation = reviewed_record(item, operation="unknown_operation")
    invalid_operation["access_id"] = stable_access_id(invalid_operation)
    with pytest.raises(ChromaAccessValidationError, match="unknown_operation"):
        build_inventory([invalid_operation])
    with pytest.raises(ChromaAccessValidationError, match="unknown_collection_resolution"):
        build_inventory([reviewed_record(item, collection_resolution="unknown")])


def test_mutation_risk_keeps_logical_and_storage_mutation_distinct(tmp_path):
    scan = discovered(
        tmp_path,
        "import chromadb\nclient = chromadb.PersistentClient(path='tmp')\n"
        "collection = client.get_collection('facts')\ncollection.query(query_texts=['x'])\n",
    )
    query = next(item for item in scan["discoveries"] if item["operation"] == "query")
    record = reviewed_record(
        query,
        lifecycle="vector_query",
        may_mutate_records=False,
        storage_internal_mutation_risk="local_process",
    )
    validate_chroma_access_inventory(build_inventory([record]))
    assert record["may_mutate_records"] is False
    assert record["storage_internal_mutation_risk"] == "local_process"


def test_new_unclassified_call_site_and_stale_entry_fail_synchronization(tmp_path):
    one = discovered(tmp_path / "one", "import chromadb\nchromadb.HttpClient(host='local')\n")
    reviewed = build_inventory([reviewed_record(one["discoveries"][0])])
    two = discovered(
        tmp_path / "two",
        "import chromadb\nchromadb.HttpClient(host='local')\n"
        "chromadb.PersistentClient(path='tmp')\n",
    )
    new_result = inventory.compare_discovery_to_inventory(two, reviewed)
    assert new_result["status"] == "mismatch"
    assert len(new_result["unresolved_access_ids"]) == 1
    empty = build_inventory([])
    stale_result = inventory.compare_discovery_to_inventory({"discoveries": [], "review_candidates": []}, reviewed)
    assert stale_result["status"] == "mismatch"
    assert stale_result["stale_access_ids"]
    assert inventory.compare_discovery_to_inventory(
        {"discoveries": [], "review_candidates": []}, empty
    )["status"] == "verified"


def test_changed_operation_and_client_type_fail_synchronization(tmp_path):
    scan = discovered(tmp_path, "import chromadb\nchromadb.PersistentClient(path='tmp')\n")
    reviewed = build_inventory([reviewed_record(scan["discoveries"][0])])
    changed_operation = copy.deepcopy(scan)
    changed_operation["discoveries"][0]["operation"] = "heartbeat"
    changed_operation["discoveries"][0]["access_id"] = stable_access_id(
        changed_operation["discoveries"][0]
    )
    result = inventory.compare_discovery_to_inventory(changed_operation, reviewed)
    assert result["unresolved_access_ids"] and result["stale_access_ids"]

    changed_client = copy.deepcopy(scan)
    changed_client["discoveries"][0]["client_type"] = "http"
    result = inventory.compare_discovery_to_inventory(changed_client, reviewed)
    assert result["mismatched_access_ids"]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("runtime", "unknown", "unknown_runtime"),
        ("lifecycle", "unknown", "unknown_lifecycle"),
        ("client_type", "unknown", "unknown_client_type"),
    ),
)
def test_unknown_classifications_are_rejected(tmp_path, field, value, code):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    with pytest.raises(ChromaAccessValidationError, match=code):
        build_inventory([reviewed_record(item, **{field: value})])


def test_privacy_validator_rejects_absolute_windows_paths_and_secrets(tmp_path):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    with pytest.raises(ChromaAccessValidationError, match="absolute_path_exposure"):
        build_inventory([reviewed_record(item, notes="C:/Users/example/private")])
    with pytest.raises(ChromaAccessValidationError, match="secret_value_exposure"):
        build_inventory([reviewed_record(item, notes="api_key=private")])


def test_privacy_validator_rejects_source_and_diff_bodies(tmp_path):
    item = discovered(tmp_path, "import chromadb\nchromadb.HttpClient(host='local')\n")[
        "discoveries"
    ][0]
    with pytest.raises(ChromaAccessValidationError, match="source_body_exposure"):
        build_inventory([reviewed_record(item, notes="diff --git private body")])
    invalid = reviewed_record(item)
    invalid["source_body"] = "private"
    with pytest.raises(ChromaAccessValidationError, match="forbidden_inventory_field"):
        build_inventory([invalid])


def test_scanner_never_imports_or_executes_scanned_modules(tmp_path):
    root = repository_with_source(
        tmp_path,
        "raise RuntimeError('must not execute')\nimport chromadb\nchromadb.HttpClient(host='local')\n",
        "backend/poison_inventory_module.py",
    )
    before = set(sys.modules)
    result = inventory.scan_repository(root)
    assert result["discoveries"]
    assert "backend.poison_inventory_module" not in set(sys.modules) - before


def test_scanner_never_constructs_clients_or_reads_protected_storage(tmp_path):
    root = repository_with_source(
        tmp_path,
        "import chromadb\nchromadb.PersistentClient(path='must-not-open')\n",
    )
    protected = root / "information" / "chroma"
    protected.mkdir(parents=True)
    (protected / "unreadable.py").write_bytes(b"\xff\xfe")
    result = inventory.scan_repository(root)
    assert len(result["discoveries"]) == 1
    assert result["discoveries"][0]["client_type"] == "persistent_embedded"


def test_executable_document_example_is_a_review_candidate(tmp_path):
    root = repository_with_source(
        tmp_path,
        "# Example\n```powershell\nchroma run --path ./data\n```\n",
        "docs/example.md",
    )
    scan = inventory.scan_repository(root)
    assert scan["review_candidates"][0]["reason"] == "executable_chroma_reference_requires_review"


def test_current_repository_discovery_matches_reviewed_inventory():
    report = inventory.inspect_repository(ROOT)
    assert report["schema"] == CHROMA_ACCESS_INVENTORY_SCHEMA
    assert report["status"] == "verified"
    assert report["discovered_count"] == report["classified_count"]
    assert report["discovered_count"] > 0
    assert report["unknown_count"] == 0
    assert report["forbidden_count"] == 0
    assert report["policy_violations"] == []
    assert report["review_candidates"] == []


def test_vector_bridge_has_no_independent_low_level_http_access():
    bridge = [
        item
        for item in INVENTORY["records"]
        if item["module"] == "backend/chroma_http_vector_search.py"
        and item["client_type"] == "http"
    ]
    assert bridge == []
    source = (ROOT / "backend" / "chroma_http_vector_search.py").read_text(encoding="utf-8")
    assert "ChromaReadClient" in source
    assert "chromadb.Http" + "Client" not in source
    assert "socket.create_connection" not in source


def test_persistent_client_access_is_test_only():
    constructors = [
        item
        for item in INVENTORY["records"]
        if item["operation"] == "client construction"
        and item["client_type"] == "persistent_embedded"
    ]
    assert len(constructors) == 1
    assert constructors[0]["runtime"] == "test_only"
    assert constructors[0]["module"] == "tests/chroma_persistence_test_support.py"
    assert constructors[0]["current_state"] == "accepted_temporary_fixture"


def test_test_only_entries_are_isolated_fakes_or_ephemeral_storage():
    records = [item for item in INVENTORY["records"] if item["runtime"] == "test_only"]
    assert records
    assert {item["client_type"] for item in records} == {
        "http",
        "persistent_embedded",
    }
    assert {item["lifecycle"] for item in records} == {"test_only"}
    ephemeral_http = [
        item for item in records if item["module"] == "tests/chroma_http_test_support.py"
    ]
    assert ephemeral_http
    assert {item["current_owner"] for item in ephemeral_http} == {
        "chroma_http_timeout_probe",
        "ephemeral_http_test_fixture",
    }
    assert all(item["current_state"].startswith("accepted_ephemeral_http_") for item in ephemeral_http)
    persistent = [
        item
        for item in records
        if item["module"] == "tests/chroma_persistence_test_support.py"
    ]
    assert persistent
    assert {item["current_state"] for item in persistent} == {"accepted_temporary_fixture"}
    assert sum(item["operation"] == "client construction" for item in persistent) == 1


def test_baseline_constructor_counts_are_projected_from_authoritative_inventory():
    projected = inventory.baseline_constructor_summary(ROOT)
    captured = baseline.capture_chroma_client_call_inventory(ROOT)
    assert captured == projected
    assert captured["production_persistent_client_call_count"] == 0
    assert captured["approved_maintenance_persistent_client_call_count"] == 0
    assert captured["test_only_persistent_client_call_count"] == 1
    assert captured["http_client_call_count"] == 2
    assert captured["unknown_unclassified_call_count"] == 0


def test_cli_reports_only_bounded_counts(tmp_path, capsys):
    scan = discovered(
        tmp_path,
        "import httpx\ndef _default_httpx_client_builder():\n    return httpx.Client()\n",
        "backend/chroma_http_transport.py",
    )
    record = reviewed_record(
        scan["discoveries"][0],
        current_owner="bounded_chroma_http_transport",
        current_state="accepted_bounded_http_transport",
    )
    payload = build_enforced_inventory([record])
    manifest = tmp_path / "reviewed.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    root = tmp_path / "repository"
    assert inventory.main(
        ["verify", "--repository-root", str(root), "--manifest", str(manifest)]
    ) == 0
    output = capsys.readouterr()
    assert "status=verified" in output.out
    assert "unresolved=0" in output.out
    assert "forbidden=0" in output.out
    assert str(root) not in output.out
    assert output.err == ""


def test_manifest_policy_cannot_reauthorize_production_embedded_access(tmp_path):
    scan = discovered(
        tmp_path,
        "from chromadb import PersistentClient as Client\nClient(path='temporary')\n",
    )
    record = reviewed_record(
        scan["discoveries"][0],
        current_owner="bounded_chroma_http_transport",
    )
    categories = {
        item["category"] for item in production_access_policy_violations([record])
    }
    assert "production_embedded_client" in categories
    with pytest.raises(ChromaAccessValidationError, match="forbidden_chroma_production_access"):
        build_enforced_inventory([record])


def test_inventory_digest_and_order_are_validated():
    validate_chroma_access_inventory(INVENTORY)
    tampered = copy.deepcopy(INVENTORY)
    tampered["inventory_digest"] = "0" * 64
    with pytest.raises(ChromaAccessValidationError, match="inventory_digest_mismatch"):
        validate_chroma_access_inventory(tampered)
    reversed_payload = copy.deepcopy(INVENTORY)
    reversed_payload["records"].reverse()
    with pytest.raises(ChromaAccessValidationError, match="non_deterministic_inventory_order"):
        validate_chroma_access_inventory(reversed_payload)


def test_inventory_sources_are_backend_only_and_semantically_named():
    tracked = [
        ROOT / "backend" / "chroma_access_models.py",
        ROOT / "backend" / "chroma_access_inventory.py",
        ROOT / "backend" / "chroma_access_manifest.py",
        Path(__file__),
    ]
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in tracked).casefold()
    assert "frontend/" + "src" not in serialized
    assert "phase" + "6_5" not in serialized
    assert "phase_" + "6_5" not in serialized
    assert "step_" + "1" not in serialized
    assert all(not item["module"].startswith("frontend/") for item in INVENTORY["records"])


def test_inventory_modules_do_not_import_runtime_chroma_boundaries():
    modules = (
        ROOT / "backend" / "chroma_access_models.py",
        ROOT / "backend" / "chroma_access_inventory.py",
        ROOT / "backend" / "chroma_access_manifest.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in modules)
    assert "import " + "chromadb" not in source
    assert "from backend.memory_store import" not in source
    assert "from backend.chroma_http_vector_search import" not in source
