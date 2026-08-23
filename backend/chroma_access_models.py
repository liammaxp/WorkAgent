"""Schema and privacy rules for the reviewed Chroma access inventory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


CHROMA_ACCESS_INVENTORY_SCHEMA = "chroma_access_inventory.v1"
MAX_ACCESS_RECORDS = 500
MAX_INVENTORY_BYTES = 1_000_000

RUNTIME_CATEGORIES = frozenset(
    {"production", "maintenance_only", "migration_only", "test_only"}
)
CLIENT_TYPES = frozenset(
    {"persistent_embedded", "http", "fake_http", "ephemeral_embedded", "unknown"}
)
COLLECTION_RESOLUTIONS = frozenset(
    {"literal", "shared_constant", "registry", "dynamic", "unknown"}
)
LIFECYCLE_CATEGORIES = frozenset(
    {"read", "vector_query", "write", "index", "migration", "maintenance", "test_only"}
)
OPERATIONS = frozenset(
    {
        "client construction",
        "http request",
        "heartbeat",
        "list collections",
        "get collection",
        "get or create collection",
        "create collection",
        "delete collection",
        "count",
        "get",
        "peek",
        "query",
        "add",
        "upsert",
        "update",
        "delete",
        "reset",
        "index",
        "rebuild",
        "migration",
        "backup/recovery inspection",
    }
)
ACCESS_MODES = frozenset({"read_only", "write_only", "read_write"})
STORAGE_INTERNAL_MUTATION_RISKS = frozenset({"none", "local_process", "server_owned"})
LATER_WORK_ITEMS = frozenset(
    {
        "configuration",
        "collection_registry",
        "central_http_client",
        "server_lifecycle",
        "single_owner_guard",
        "status_read_migration",
        "read_vector_migration",
        "write_index_migration",
        "maintenance_recovery",
        "test_infrastructure",
        "deprecation_guard",
    }
)

_RECORD_FIELDS = frozenset(
    {
        "access_id",
        "module",
        "symbol",
        "line",
        "semantic_role",
        "runtime",
        "client_type",
        "collection",
        "collection_resolution",
        "operation",
        "lifecycle",
        "access_mode",
        "may_create_collection",
        "may_mutate_records",
        "storage_internal_mutation_risk",
        "current_owner",
        "migration_target",
        "later_work_item",
        "current_state",
        "migration_action",
        "notes",
    }
)
_IDENTITY_FIELDS = (
    "module",
    "symbol",
    "semantic_role",
    "operation",
    "collection",
)
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/<>|-]{1,180}$")
_ABSOLUTE_WINDOWS_RE = re.compile(r"(?i)(?:^|[\s'\"])[a-z]:[\\/]")
_UNC_RE = re.compile(r"(?:^|[\s'\"])(?:\\\\|//)[^/\\\s]+[/\\]")
_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret|credential)\s*[:=]"
)
_SOURCE_BODY_RE = re.compile(r"(?i)(?:diff\s+--git|@@\s+-\d|begin\s+(?:rsa\s+)?private\s+key)")
_FORBIDDEN_KEYS = frozenset(
    {"documents", "embeddings", "raw_metadata", "source_body", "diff_body", "patch"}
)

_APPROVED_PRODUCTION_OWNER_MODULES = {
    "bounded_chroma_http_transport": "backend/chroma_http_transport.py",
    "central_http_client_factory": "backend/chroma_http_client_factory.py",
    "chroma_operational_reader": "backend/chroma_operational_reader.py",
    "chroma_server_lifecycle": "backend/chroma_server_lifecycle.py",
}
_APPROVED_LOW_LEVEL_HTTP_CONSTRUCTOR = (
    "backend/chroma_http_transport.py",
    "_default_httpx_client_builder",
    "bounded_chroma_http_transport",
    "low_level_http_client",
)
_APPROVED_TEST_PERSISTENCE_ACCESS = frozenset(
    {
        (
            "tests/chroma_persistence_test_support.py",
            "create_test_owned_persistent_client",
            "client construction",
            "default_persistent_client",
        ),
        (
            "tests/chroma_persistence_test_support.py",
            "read_test_owned_collection_snapshot",
            "get collection",
            "get_collection",
        ),
        (
            "tests/chroma_persistence_test_support.py",
            "read_test_owned_collection_snapshot",
            "count",
            "count",
        ),
        (
            "tests/chroma_persistence_test_support.py",
            "read_test_owned_collection_snapshot",
            "get",
            "get",
        ),
        (
            "tests/chroma_persistence_test_support.py",
            "read_test_owned_collection_snapshot",
            "query",
            "query",
        ),
    }
)
_APPROVED_TEST_HTTP_BOUNDARY = "tests/chroma_http_test_support.py"
_COLLECTION_CREATION_OPERATIONS = frozenset(
    {"create collection", "get or create collection"}
)
_EMBEDDED_CLIENT_TYPES = frozenset({"persistent_embedded", "ephemeral_embedded"})


class ChromaAccessValidationError(ValueError):
    """Stable validation failure that never includes repository or user data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fail(code: str) -> None:
    raise ChromaAccessValidationError(code)


def stable_access_id(record: Mapping[str, Any]) -> str:
    """Return a semantic identifier that deliberately excludes the source line."""

    identity = {field: record.get(field) for field in _IDENTITY_FIELDS}
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    return f"chroma_access_{digest}"


def inventory_digest(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(record) for record in records), key=lambda item: item["access_id"])
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def _validate_privacy(value: Any, *, key: str = "") -> None:
    if key.casefold() in _FORBIDDEN_KEYS:
        _fail("forbidden_inventory_field")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                _fail("invalid_inventory_key")
            _validate_privacy(child, key=child_key)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate_privacy(child, key=key)
        return
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if (
            _ABSOLUTE_WINDOWS_RE.search(value)
            or _UNC_RE.search(value)
            or normalized.startswith("/")
        ):
            _fail("absolute_path_exposure")
        if _SECRET_RE.search(value):
            _fail("secret_value_exposure")
        if _SOURCE_BODY_RE.search(value):
            _fail("source_body_exposure")


def _require_safe_text(value: Any, code: str, *, bounded: bool = True) -> str:
    if not isinstance(value, str) or not value or (bounded and len(value) > 300):
        _fail(code)
    return value


def validate_access_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or frozenset(record) != _RECORD_FIELDS:
        _fail("invalid_access_record_shape")
    _validate_privacy(record)
    if record["access_id"] != stable_access_id(record):
        _fail("access_id_mismatch")
    module = _require_safe_text(record["module"], "invalid_module")
    if module.startswith(("/", "\\")) or ".." in module.split("/"):
        _fail("invalid_module")
    _require_safe_text(record["symbol"], "invalid_symbol")
    _require_safe_text(record["semantic_role"], "invalid_semantic_role")
    for field in (
        "collection",
        "current_owner",
        "migration_target",
        "current_state",
        "migration_action",
    ):
        value = _require_safe_text(record[field], f"invalid_{field}")
        if not _SAFE_TOKEN_RE.fullmatch(value):
            _fail(f"invalid_{field}")
    _require_safe_text(record["notes"], "invalid_notes")
    if not isinstance(record["line"], int) or isinstance(record["line"], bool) or record["line"] <= 0:
        _fail("invalid_line")
    allowed_fields = {
        "runtime": RUNTIME_CATEGORIES,
        "client_type": CLIENT_TYPES,
        "collection_resolution": COLLECTION_RESOLUTIONS,
        "operation": OPERATIONS,
        "lifecycle": LIFECYCLE_CATEGORIES,
        "access_mode": ACCESS_MODES,
        "storage_internal_mutation_risk": STORAGE_INTERNAL_MUTATION_RISKS,
        "later_work_item": LATER_WORK_ITEMS,
    }
    for field, allowed in allowed_fields.items():
        if record[field] not in allowed:
            _fail(f"unknown_{field}")
    if record["client_type"] == "unknown":
        _fail("unknown_client_type")
    if record["collection_resolution"] == "unknown":
        _fail("unknown_collection_resolution")
    for field in ("may_create_collection", "may_mutate_records"):
        if not isinstance(record[field], bool):
            _fail(f"invalid_{field}")
    if record["access_mode"] == "read_only" and record["may_mutate_records"]:
        _fail("read_only_record_mutation_conflict")
    if record["operation"] == "query" and record["may_mutate_records"]:
        _fail("query_record_mutation_conflict")
    if record["operation"] == "get or create collection" and not record["may_create_collection"]:
        _fail("collection_creation_risk_missing")
    if record["client_type"] in {"http", "fake_http"}:
        if record["storage_internal_mutation_risk"] not in {"server_owned", "none"}:
            _fail("invalid_http_storage_risk")


def validate_chroma_access_inventory(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or frozenset(payload) != {
        "schema",
        "records",
        "inventory_digest",
    }:
        _fail("invalid_inventory_shape")
    _validate_privacy(payload)
    if payload["schema"] != CHROMA_ACCESS_INVENTORY_SCHEMA:
        _fail("unsupported_inventory_schema")
    records = payload["records"]
    if not isinstance(records, list) or len(records) > MAX_ACCESS_RECORDS:
        _fail("invalid_inventory_records")
    if len(canonical_json(payload).encode("utf-8")) > MAX_INVENTORY_BYTES:
        _fail("inventory_size_limit_exceeded")
    access_ids: list[str] = []
    for record in records:
        validate_access_record(record)
        access_ids.append(record["access_id"])
    if access_ids != sorted(access_ids):
        _fail("non_deterministic_inventory_order")
    if len(access_ids) != len(set(access_ids)):
        _fail("duplicate_access_id")
    if payload["inventory_digest"] != inventory_digest(records):
        _fail("inventory_digest_mismatch")


def production_access_policy_violations(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    """Return bounded semantic violations without source, paths, or runtime data."""

    violations: list[dict[str, str | int]] = []

    def add(record: Mapping[str, Any], category: str) -> None:
        item = {
            "module": str(record.get("module", "unknown")),
            "line": int(record.get("line", 0)),
            "symbol": str(record.get("symbol", "unknown")),
            "category": category,
        }
        if item not in violations:
            violations.append(item)

    for record in records:
        runtime = record.get("runtime")
        client_type = record.get("client_type")
        module = record.get("module")
        owner = record.get("current_owner")
        operation = record.get("operation")
        role = record.get("semantic_role")

        if runtime == "test_only":
            if client_type in _EMBEDDED_CLIENT_TYPES and (
                (
                    module,
                    record.get("symbol"),
                    operation,
                    role,
                )
                not in _APPROVED_TEST_PERSISTENCE_ACCESS
                or owner != "chroma_persistence_test_probe"
                or record.get("lifecycle") != "test_only"
                or not str(module).startswith("tests/")
            ):
                add(record, "test_embedded_boundary_violation")
            if operation in _COLLECTION_CREATION_OPERATIONS and (
                module != _APPROVED_TEST_HTTP_BOUNDARY
                or owner != "ephemeral_http_test_fixture"
                or record.get("lifecycle") != "test_only"
            ):
                add(record, "test_collection_creation_boundary_violation")
            continue

        if client_type in _EMBEDDED_CLIENT_TYPES:
            add(record, "production_embedded_client")
        if client_type != "http":
            add(record, "production_non_http_access")
        if operation in _COLLECTION_CREATION_OPERATIONS or record.get(
            "may_create_collection"
        ):
            add(record, "production_collection_creation")
        if role == "direct_chromadb_http_client":
            add(record, "production_direct_chromadb_http_client")
        if role in {"independent_http_session", "independent_http_request"}:
            add(record, "production_independent_http_access")
        if operation == "client construction" and (
            module,
            record.get("symbol"),
            owner,
            role,
        ) != _APPROVED_LOW_LEVEL_HTTP_CONSTRUCTOR:
            add(record, "production_direct_client_construction")
        if operation == "http request" and (
            module != "backend/chroma_http_transport.py"
            or owner != "bounded_chroma_http_transport"
        ):
            add(record, "production_independent_http_access")
        if _APPROVED_PRODUCTION_OWNER_MODULES.get(str(owner)) != module:
            add(record, "unapproved_production_access_boundary")
        state = str(record.get("current_state", "")).casefold()
        if "embedded" in state or "legacy" in state:
            add(record, "deprecated_production_access_state")

    return sorted(
        violations,
        key=lambda item: (
            str(item["module"]),
            int(item["line"]),
            str(item["symbol"]),
            str(item["category"]),
        ),
    )


def validate_production_access_policy(payload: Mapping[str, Any]) -> None:
    """Reject inventory entries that attempt to authorize forbidden architecture."""

    validate_chroma_access_inventory(payload)
    if production_access_policy_violations(payload["records"]):
        _fail("forbidden_chroma_production_access")


def build_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(record) for record in records), key=lambda item: item["access_id"])
    payload = {
        "schema": CHROMA_ACCESS_INVENTORY_SCHEMA,
        "records": ordered,
        "inventory_digest": inventory_digest(ordered),
    }
    validate_chroma_access_inventory(payload)
    return payload


def build_enforced_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the repository manifest and enforce permanent access boundaries."""

    payload = build_inventory(records)
    validate_production_access_policy(payload)
    return payload
