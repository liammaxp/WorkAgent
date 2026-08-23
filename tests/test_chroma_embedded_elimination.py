from __future__ import annotations

import ast
from pathlib import Path

from backend.chroma_access_manifest import INVENTORY
from backend.chroma_collection_registry import list_registered_collections
from backend.chroma_persistence_guard import verify_persistent_client_access
from backend.memory_store import MemoryVectorStore


ROOT = Path(__file__).resolve().parents[1]
MEMORY_STORE_SOURCE = ROOT / "backend" / "memory_store.py"


def test_production_memory_store_has_no_embedded_client_boundary():
    source = MEMORY_STORE_SOURCE.read_text(encoding="utf-8")
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
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "chromadb" not in imported_roots
    assert not called_attributes & {
        "PersistentClient",
        "HttpClient",
        "create_collection",
        "get_or_create_collection",
    }
    assert not methods & {
        "_ensure_client",
        "_migrate_legacy_profile",
        "_migrate_legacy_github",
        "_legacy_upsert_with_similarity",
        "_legacy_replace_profile",
        "_legacy_store_github_contexts",
        "_legacy_cleanup_github_repositories",
    }


def test_memory_store_construction_does_not_touch_legacy_paths(tmp_path):
    persistence = tmp_path / "persistent"
    memory = tmp_path / "memory.json"
    github = tmp_path / "github"

    store = MemoryVectorStore(persistence, memory, github)

    assert store.persist_directory == persistence
    assert store.legacy_memory_path == memory
    assert store.legacy_github_dir == github
    assert not persistence.exists()
    assert not memory.exists()
    assert not github.exists()
    assert not hasattr(store, "_client")
    assert not hasattr(store, "_profile")
    assert not hasattr(store, "_github")


def test_inventory_has_no_production_embedded_access():
    production = [item for item in INVENTORY["records"] if item["runtime"] != "test_only"]
    assert production
    assert all(item["client_type"] != "persistent_embedded" for item in production)
    assert all(item["runtime"] not in {"migration_only"} for item in production)

    embedded = [
        item for item in INVENTORY["records"] if item["client_type"] == "persistent_embedded"
    ]
    assert len(embedded) == 5
    assert {item["runtime"] for item in embedded} == {"test_only"}
    assert {item["module"] for item in embedded} == {
        "tests/chroma_persistence_test_support.py"
    }
    assert sum(item["operation"] == "client construction" for item in embedded) == 1


def test_static_guard_allows_only_isolated_test_constructor():
    summary = verify_persistent_client_access(ROOT).safe_summary()
    assert summary == {
        "schema": "chroma_persistent_client_guard.v1",
        "production_legacy_persistent_client_count": 0,
        "test_only_persistent_client_count": 1,
        "approved_maintenance_persistent_client_count": 0,
        "forbidden_persistent_client_count": 0,
        "unknown_persistent_client_count": 0,
        "embedded_fallback_candidate_count": 0,
    }


def test_registry_has_no_legacy_consumers_or_automatic_creation():
    definitions = list_registered_collections()
    assert {item.semantic_id for item in definitions} == {
        "github_evidence",
        "profile_facts",
    }
    assert all(item.automatic_creation is False for item in definitions)
    assert all(item.legacy_consumers == () for item in definitions)
