"""Schema and privacy validation for protected Chroma migration baselines."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping


CHROMA_MIGRATION_BASELINE_SCHEMA = "chroma_migration_baseline.v1"
PROTECTED_CAPTURE_MODE = "protected_read_only"
LOGICAL_INVENTORY_SOURCES = frozenset(
    {"existing_safe_artifact", "approved_http", "unavailable"}
)
MAX_BASELINE_FILES = 100_000
MAX_BASELINE_ARTIFACTS = 64
MAX_BASELINE_CALL_SITES = 256
MAX_BASELINE_COLLECTIONS = 64
MAX_BASELINE_LIMITATIONS = 16
MAX_BASELINE_BYTES = 8_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'=])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)")
_POSIX_ABSOLUTE_RE = re.compile(r"(?:^|\s)/(?:Users|home|var|etc|tmp|mnt|opt|srv)/")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*[:=]|"
    r"access[_-]?token\s*[:=]|password\s*[:=]|secret\s*[:=]|credential\s*[:=])"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "document",
        "documents",
        "embedding",
        "embeddings",
        "source_body",
        "raw_source_body",
        "patch",
        "diff",
        "diff_body",
        "raw_metadata",
        "metadata",
        "token",
        "tokens",
        "api_key",
        "access_token",
        "password",
        "private_key",
        "secret",
        "secrets",
        "environment",
        "environment_values",
    }
)
_ALLOWED_PRIVACY_KEYS = frozenset(
    {
        "contains_documents",
        "contains_embeddings",
        "contains_raw_metadata",
        "contains_absolute_paths",
        "contains_secrets",
    }
)


class BaselineValidationError(ValueError):
    """A stable, privacy-safe baseline validation failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fail(code: str) -> None:
    raise BaselineValidationError(code)


def _exact_keys(value: Any, expected: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(code)
    return value


def _non_negative_int(value: Any, code: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(code)
    if maximum is not None and value > maximum:
        _fail(code)
    return value


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(code)
    return value


def validate_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 800
        or "\\" in value
        or _WINDOWS_ABSOLUTE_RE.search(value)
    ):
        _fail("invalid_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("invalid_relative_path")
    if path.as_posix() != value:
        _fail("noncanonical_relative_path")
    return value


def _validate_file_inventory(value: Any) -> None:
    payload = _exact_keys(
        value,
        {"file_count", "total_bytes", "files", "aggregate_sha256"},
        "invalid_protected_storage",
    )
    files = payload["files"]
    if not isinstance(files, list) or len(files) > MAX_BASELINE_FILES:
        _fail("invalid_protected_files")
    normalized: list[dict[str, Any]] = []
    for item in files:
        entry = _exact_keys(
            item, {"relative_path", "size_bytes", "sha256"}, "invalid_protected_file"
        )
        normalized.append(
            {
                "relative_path": validate_relative_path(entry["relative_path"]),
                "size_bytes": _non_negative_int(entry["size_bytes"], "invalid_file_size"),
                "sha256": _sha256(entry["sha256"], "invalid_file_hash"),
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["relative_path"]):
        _fail("nondeterministic_file_order")
    if len({item["relative_path"] for item in normalized}) != len(normalized):
        _fail("duplicate_file_path")
    if payload["file_count"] != len(normalized):
        _fail("file_count_mismatch")
    if payload["total_bytes"] != sum(item["size_bytes"] for item in normalized):
        _fail("file_size_total_mismatch")
    if _sha256(payload["aggregate_sha256"], "invalid_storage_hash") != sha256_json(normalized):
        _fail("storage_hash_mismatch")


def _validate_logical_inventory(value: Any) -> None:
    payload = _exact_keys(
        value, {"source", "collections", "limitations"}, "invalid_logical_inventory"
    )
    source = payload["source"]
    if source not in LOGICAL_INVENTORY_SOURCES:
        _fail("invalid_logical_source")
    limitations = payload["limitations"]
    if (
        not isinstance(limitations, list)
        or len(limitations) > MAX_BASELINE_LIMITATIONS
        or any(not isinstance(item, str) or not item or len(item) > 300 for item in limitations)
    ):
        _fail("invalid_logical_limitations")
    collections = payload["collections"]
    if not isinstance(collections, list) or len(collections) > MAX_BASELINE_COLLECTIONS:
        _fail("invalid_logical_collections")
    normalized = []
    for item in collections:
        collection = _exact_keys(
            item,
            {
                "semantic_name",
                "record_count",
                "record_ids_sha256",
                "logical_fingerprint",
                "repository_count",
                "schema_marker",
            },
            "invalid_logical_collection",
        )
        name = collection["semantic_name"]
        if not isinstance(name, str) or not _SEMANTIC_NAME_RE.fullmatch(name):
            _fail("invalid_collection_name")
        marker = collection["schema_marker"]
        if marker is not None and (
            not isinstance(marker, str) or not _SEMANTIC_NAME_RE.fullmatch(marker)
        ):
            _fail("invalid_collection_schema_marker")
        normalized.append(
            {
                "semantic_name": name,
                "record_count": _non_negative_int(
                    collection["record_count"], "invalid_record_count", 10_000_000
                ),
                "record_ids_sha256": _sha256(
                    collection["record_ids_sha256"], "invalid_record_ids_hash"
                ),
                "logical_fingerprint": _sha256(
                    collection["logical_fingerprint"], "invalid_logical_fingerprint"
                ),
                "repository_count": _non_negative_int(
                    collection["repository_count"], "invalid_repository_count", 100_000
                ),
                "schema_marker": marker,
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["semantic_name"]):
        _fail("nondeterministic_collection_order")
    if source == "unavailable" and (collections or not limitations):
        _fail("invalid_unavailable_logical_inventory")
    if source != "unavailable" and not collections:
        _fail("empty_available_logical_inventory")


def _validate_evidence_artifacts(value: Any) -> None:
    payload = _exact_keys(
        value, {"artifacts", "aggregate_sha256"}, "invalid_evidence_artifacts"
    )
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) > MAX_BASELINE_ARTIFACTS:
        _fail("invalid_artifact_list")
    normalized = []
    for item in artifacts:
        artifact = _exact_keys(
            item,
            {"semantic_name", "relative_path", "size_bytes", "sha256", "schema_marker"},
            "invalid_artifact",
        )
        name = artifact["semantic_name"]
        if not isinstance(name, str) or not _SEMANTIC_NAME_RE.fullmatch(name):
            _fail("invalid_artifact_name")
        marker = artifact["schema_marker"]
        if marker is not None and (
            not isinstance(marker, str) or not _SEMANTIC_NAME_RE.fullmatch(marker)
        ):
            _fail("invalid_artifact_schema_marker")
        normalized.append(
            {
                "semantic_name": name,
                "relative_path": validate_relative_path(artifact["relative_path"]),
                "size_bytes": _non_negative_int(artifact["size_bytes"], "invalid_artifact_size"),
                "sha256": _sha256(artifact["sha256"], "invalid_artifact_hash"),
                "schema_marker": marker,
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["semantic_name"]):
        _fail("nondeterministic_artifact_order")
    if len({item["semantic_name"] for item in normalized}) != len(normalized):
        _fail("duplicate_artifact_name")
    if _sha256(payload["aggregate_sha256"], "invalid_artifact_aggregate") != sha256_json(
        normalized
    ):
        _fail("artifact_aggregate_mismatch")


def _validate_repository_state(value: Any) -> None:
    payload = _exact_keys(
        value,
        {
            "production_persistent_client_call_count",
            "approved_maintenance_persistent_client_call_count",
            "test_only_persistent_client_call_count",
            "http_client_call_count",
            "unknown_unclassified_call_count",
            "call_sites",
            "aggregate_sha256",
        },
        "invalid_repository_state",
    )
    call_sites = payload["call_sites"]
    if not isinstance(call_sites, list) or len(call_sites) > MAX_BASELINE_CALL_SITES:
        _fail("invalid_call_sites")
    normalized = []
    for item in call_sites:
        site = _exact_keys(
            item,
            {"client_type", "classification", "relative_path", "line", "callable"},
            "invalid_call_site",
        )
        if site["client_type"] not in {"persistent", "http"}:
            _fail("invalid_client_type")
        if site["classification"] not in {
            "production",
            "approved_maintenance",
            "test_only",
            "unknown",
        }:
            _fail("invalid_call_classification")
        callable_name = site["callable"]
        if (
            not isinstance(callable_name, str)
            or not callable_name
            or len(callable_name) > 240
            or not re.fullmatch(r"[A-Za-z0-9_.<>:-]+", callable_name)
        ):
            _fail("invalid_call_name")
        normalized.append(
            {
                "client_type": site["client_type"],
                "classification": site["classification"],
                "relative_path": validate_relative_path(site["relative_path"]),
                "line": _non_negative_int(site["line"], "invalid_call_line", 10_000_000),
                "callable": callable_name,
            }
        )
    expected_order = sorted(
        normalized,
        key=lambda item: (item["relative_path"], item["line"], item["client_type"]),
    )
    if normalized != expected_order:
        _fail("nondeterministic_call_site_order")
    counts = {
        "production_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "production"
            for item in normalized
        ),
        "approved_maintenance_persistent_client_call_count": sum(
            item["client_type"] == "persistent"
            and item["classification"] == "approved_maintenance"
            for item in normalized
        ),
        "test_only_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "test_only"
            for item in normalized
        ),
        "http_client_call_count": sum(item["client_type"] == "http" for item in normalized),
        "unknown_unclassified_call_count": sum(
            item["classification"] == "unknown" for item in normalized
        ),
    }
    for key, expected in counts.items():
        if payload[key] != expected:
            _fail("call_site_count_mismatch")
    if counts["unknown_unclassified_call_count"]:
        _fail("unclassified_chroma_call_site")
    if _sha256(payload["aggregate_sha256"], "invalid_call_site_aggregate") != sha256_json(
        normalized
    ):
        _fail("call_site_aggregate_mismatch")


def _defensive_privacy_scan(value: Any, key: str = "") -> None:
    normalized_key = key.casefold()
    if normalized_key in _FORBIDDEN_KEYS and normalized_key not in _ALLOWED_PRIVACY_KEYS:
        _fail("forbidden_baseline_field")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _defensive_privacy_scan(child_value, str(child_key))
    elif isinstance(value, list):
        for item in value:
            _defensive_privacy_scan(item, key)
    elif isinstance(value, str):
        if _WINDOWS_ABSOLUTE_RE.search(value) or _POSIX_ABSOLUTE_RE.search(value):
            _fail("absolute_path_exposure")
        if _SECRET_VALUE_RE.search(value):
            _fail("secret_value_exposure")


def validate_chroma_migration_baseline(value: Any) -> None:
    payload = _exact_keys(
        value,
        {
            "schema",
            "captured_at",
            "capture_mode",
            "deployment_observation",
            "protected_storage",
            "logical_inventory",
            "evidence_artifacts",
            "repository_state",
            "privacy",
            "content_sha256",
        },
        "invalid_baseline_schema",
    )
    if payload["schema"] != CHROMA_MIGRATION_BASELINE_SCHEMA:
        _fail("unsupported_baseline_schema")
    if payload["capture_mode"] != PROTECTED_CAPTURE_MODE:
        _fail("invalid_capture_mode")
    captured_at = payload["captured_at"]
    if not isinstance(captured_at, str) or len(captured_at) > 64:
        _fail("invalid_capture_timestamp")
    try:
        parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError:
        _fail("invalid_capture_timestamp")
    if parsed.tzinfo is None:
        _fail("capture_timestamp_requires_timezone")

    deployment = _exact_keys(
        payload["deployment_observation"],
        {"configured_mode", "server_reachable", "server_version"},
        "invalid_deployment_observation",
    )
    if deployment["configured_mode"] not in {"disabled", "local_http", "unknown"}:
        _fail("invalid_deployment_mode")
    if type(deployment["server_reachable"]) is not bool:
        _fail("invalid_server_reachability")
    version = deployment["server_version"]
    if version is not None and (
        not isinstance(version, str)
        or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", version)
    ):
        _fail("invalid_server_version")

    _validate_file_inventory(payload["protected_storage"])
    _validate_logical_inventory(payload["logical_inventory"])
    _validate_evidence_artifacts(payload["evidence_artifacts"])
    _validate_repository_state(payload["repository_state"])
    privacy = _exact_keys(
        payload["privacy"], set(_ALLOWED_PRIVACY_KEYS), "invalid_privacy_declaration"
    )
    if any(type(privacy[key]) is not bool or privacy[key] for key in _ALLOWED_PRIVACY_KEYS):
        _fail("unsafe_privacy_declaration")
    _defensive_privacy_scan(payload)
    unsigned = copy.deepcopy(dict(payload))
    content_hash = unsigned.pop("content_sha256")
    if _sha256(content_hash, "invalid_baseline_content_hash") != sha256_json(unsigned):
        _fail("baseline_content_hash_mismatch")


def add_baseline_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("content_sha256", None)
    payload["content_sha256"] = sha256_json(payload)
    return payload
