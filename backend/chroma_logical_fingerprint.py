"""Authoritative bounded logical-integrity fingerprints for Chroma collections.

The implementation deliberately operates through the registry-gated central HTTP
factory.  It never reads Chroma persistence files to infer logical collection
state and never requests documents or embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from backend.chroma_backup_recovery import (
    DEFAULT_BACKUP_ROOT,
    load_verified_chroma_backup_manifest,
    restore_chroma_backup,
    verify_chroma_backup,
)
from backend.chroma_collection_registry import (
    CHROMA_COLLECTION_REGISTRY_SCHEMA,
    EXCLUDED_VOLATILE_LOGICAL_INTEGRITY_METADATA_FIELDS,
    REJECTED_LOGICAL_INTEGRITY_METADATA_FIELDS,
    GITHUB_EVIDENCE_SEMANTIC_ID,
    PROFILE_FACTS_SEMANTIC_ID,
    ChromaCollectionDefinition,
    UnknownCollectionSemanticId,
    get_collection_definition,
    list_registered_collections,
)
from backend.chroma_config import (
    LOOPBACK_HOST,
    ChromaDeploymentConfig,
    ChromaDeploymentMode,
)
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import ChromaCollectionMissing, ChromaTransportError
from backend.chroma_logical_fingerprint_models import (
    CHROMA_LOGICAL_BASELINE_SCHEMA,
    CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
    CHROMA_LOGICAL_GATE_SCHEMA,
    ChromaLogicalBaseline,
    ChromaLogicalComparison,
    ChromaLogicalFingerprint,
    ChromaLogicalIntegrityGate,
    ChromaLogicalModelError,
)
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_server_lifecycle import (
    AtomicChromaServerStateStore,
    ChromaServerLifecycleController,
    build_chroma_server_lifecycle_config,
)
from backend.project_repository_identity import (
    IDENTITY_SCHEMA_VERSION,
    authority_to_repository_mapping,
    build_project_repository_identity_authority,
    load_project_repository_identity_authority,
    normalize_project_id,
    normalize_repository_identity,
)


DEFAULT_LOGICAL_FINGERPRINT_ROOT = (
    Path(__file__).resolve().parents[1] / "information" / "chroma_logical_fingerprints"
)
DEFAULT_PROTECTED_CHROMA_ROOT = Path(__file__).resolve().parents[1] / "information" / "chroma"
DEFAULT_PAGE_SIZE = 200
MAX_LOGICAL_RECORDS = 100_000
MAX_RECORD_ID_BYTES = 512
MAX_METADATA_BYTES_PER_RECORD = 16_384
MAX_TOTAL_METADATA_BYTES = 32_000_000
MAX_LOGICAL_BASELINE_BYTES = 2_000_000

_FORBIDDEN_STRING_RE = re.compile(
    r"(?i)(?:begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*[:=]|"
    r"access[_-]?token\s*[:=]|password\s*[:=]|secret\s*[:=]|credential\s*[:=])"
)
_ABSOLUTE_PATH_RE = re.compile(r"(?i)^(?:[a-z]:[\\/]|/|\\\\)")
_URI_RE = re.compile(r"(?i)^[a-z][a-z0-9+.-]{1,31}://")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ChromaLogicalFingerprintError(RuntimeError):
    """A bounded logical-fingerprint failure with no raw record content."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ChromaLogicalCollectionMissing(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalSnapshotUnstable(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalRecordLimitExceeded(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalMetadataUnsafe(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalAuthorityViolation(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalSchemaMismatch(ChromaLogicalFingerprintError):
    pass


class ChromaLogicalBaselineError(ChromaLogicalFingerprintError):
    pass


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise ChromaLogicalMetadataUnsafe("unsafe_logical_canonical_value") from None


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_nonnegative_count(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromaLogicalSnapshotUnstable(code)
    if value > MAX_LOGICAL_RECORDS:
        raise ChromaLogicalRecordLimitExceeded("logical_record_limit_exceeded")
    return value


def _definition(semantic_collection_id: str) -> ChromaCollectionDefinition:
    try:
        definition = get_collection_definition(semantic_collection_id)
    except UnknownCollectionSemanticId:
        raise ChromaLogicalSchemaMismatch("unknown_logical_collection") from None
    return definition


def _validate_definition(definition: ChromaCollectionDefinition) -> None:
    if not isinstance(definition, ChromaCollectionDefinition):
        raise ChromaLogicalSchemaMismatch("invalid_logical_collection_definition")
    if definition.automatic_creation is not False:
        raise ChromaLogicalSchemaMismatch("unsafe_logical_collection_creation_policy")
    if not definition.schema_version.startswith(f"{definition.semantic_id}."):
        raise ChromaLogicalSchemaMismatch("logical_collection_schema_mismatch")
    allowlist = definition.logical_integrity_metadata_allowlist
    if not isinstance(allowlist, tuple) or allowlist != tuple(sorted(set(allowlist))):
        raise ChromaLogicalSchemaMismatch("invalid_logical_metadata_allowlist")


def _typed_scalar(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChromaLogicalMetadataUnsafe("non_finite_logical_metadata")
        return {"type": "float", "value": value}
    if not isinstance(value, str):
        raise ChromaLogicalMetadataUnsafe("unsupported_logical_metadata_value")
    if (
        _CONTROL_RE.search(value)
        or _FORBIDDEN_STRING_RE.search(value)
        or _ABSOLUTE_PATH_RE.search(value)
        or _URI_RE.search(value)
    ):
        raise ChromaLogicalMetadataUnsafe("unsafe_logical_metadata_string")
    if len(value.encode("utf-8")) > MAX_METADATA_BYTES_PER_RECORD:
        raise ChromaLogicalMetadataUnsafe("logical_metadata_record_limit_exceeded")
    return {"type": "string", "value": value}


def _repository_authority_payload(authority: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not isinstance(authority, Mapping) or authority.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ChromaLogicalAuthorityViolation("repository_mapping_authority_unavailable")
    mapping = authority_to_repository_mapping(authority)
    conflicts = mapping.get("conflicts")
    repositories = mapping.get("repository_to_project")
    aliases = mapping.get("alias_to_repository")
    if (
        not isinstance(conflicts, list)
        or conflicts
        or not isinstance(repositories, Mapping)
        or not isinstance(aliases, Mapping)
    ):
        raise ChromaLogicalAuthorityViolation("repository_mapping_authority_invalid")
    authority_payload = {
        "authority_schema": IDENTITY_SCHEMA_VERSION,
        "repository_to_project": sorted(
            (str(repository), str(project)) for repository, project in repositories.items()
        ),
        "alias_to_repository": sorted(
            (str(alias), str(repository)) for alias, repository in aliases.items()
        ),
    }
    return mapping, authority_payload


def _canonical_repository(value: Any, mapping: Mapping[str, Any]) -> str:
    if not isinstance(value, str) or _URI_RE.search(value) or _ABSOLUTE_PATH_RE.search(value):
        return ""
    normalized = normalize_repository_identity(value)
    repositories = mapping.get("repository_to_project", {})
    if normalized in repositories:
        return normalized
    alias = mapping.get("alias_to_repository", {}).get(normalized)
    return alias if isinstance(alias, str) else ""


def _validate_github_authority(metadata: Mapping[str, Any], mapping: Mapping[str, Any]) -> None:
    if "repository" not in metadata:
        raise ChromaLogicalAuthorityViolation("missing_github_project_authority")
    repository = _canonical_repository(metadata.get("repository"), mapping)
    repositories = mapping.get("repository_to_project", {})
    mapped_project = repositories.get(repository)
    if not repository or not isinstance(mapped_project, str):
        raise ChromaLogicalAuthorityViolation("unknown_github_repository")
    if "project_id" in metadata:
        project_id = normalize_project_id(metadata.get("project_id"))
        if not project_id or project_id != mapped_project:
            raise ChromaLogicalAuthorityViolation("github_repository_project_mismatch")
    if "repository_project_id" in metadata:
        repository_project_id = normalize_project_id(metadata.get("repository_project_id"))
        if not repository_project_id or repository_project_id != mapped_project:
            raise ChromaLogicalAuthorityViolation("github_cross_project_record")


def _authority_payload(
    definition: ChromaCollectionDefinition,
    repository_authority: Any,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    payload: dict[str, Any] = {
        "registry_schema": CHROMA_COLLECTION_REGISTRY_SCHEMA,
        "collection_semantic_id": definition.semantic_id,
        "collection_name": definition.collection_name,
        "collection_schema_version": definition.schema_version,
        "metadata_allowlist": list(definition.logical_integrity_metadata_allowlist),
        "rejected_metadata_fields": list(REJECTED_LOGICAL_INTEGRITY_METADATA_FIELDS),
        "excluded_volatile_metadata_fields": list(
            EXCLUDED_VOLATILE_LOGICAL_INTEGRITY_METADATA_FIELDS
        ),
        "authority_requirements": definition.authority_requirements.to_dict(),
        "automatic_creation": definition.automatic_creation,
    }
    if definition.semantic_id == GITHUB_EVIDENCE_SEMANTIC_ID:
        mapping, repository_payload = _repository_authority_payload(repository_authority)
        payload["repository_mapping_authority"] = repository_payload
        return payload, mapping
    if definition.semantic_id != PROFILE_FACTS_SEMANTIC_ID:
        raise ChromaLogicalSchemaMismatch("unknown_logical_collection")
    payload["profile_scope"] = "existing_single_profile"
    return payload, None


def _canonical_metadata(
    definition: ChromaCollectionDefinition,
    metadata: Any,
    repository_mapping: Mapping[str, Any] | None,
) -> tuple[list[list[Any]], int]:
    if metadata is None:
        source: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        source = metadata
    else:
        raise ChromaLogicalMetadataUnsafe("invalid_logical_metadata_shape")
    keys = set(source)
    if any(not isinstance(key, str) for key in keys):
        raise ChromaLogicalMetadataUnsafe("invalid_logical_metadata_key")
    rejected = set(REJECTED_LOGICAL_INTEGRITY_METADATA_FIELDS)
    if keys & rejected:
        raise ChromaLogicalMetadataUnsafe("forbidden_logical_metadata_field")
    if definition.semantic_id == GITHUB_EVIDENCE_SEMANTIC_ID:
        if repository_mapping is None:
            raise ChromaLogicalAuthorityViolation("repository_mapping_authority_unavailable")
        _validate_github_authority(source, repository_mapping)
    fields: list[list[Any]] = []
    for key in definition.logical_integrity_metadata_allowlist:
        if key not in source:
            fields.append([key, {"state": "absent"}])
        else:
            fields.append([key, {"state": "present", "value": _typed_scalar(source[key])}])
    encoded_size = len(_canonical_json(fields))
    if encoded_size > MAX_METADATA_BYTES_PER_RECORD:
        raise ChromaLogicalMetadataUnsafe("logical_metadata_record_limit_exceeded")
    return fields, encoded_size


def build_logical_fingerprint(
    semantic_collection_id: str,
    records: Sequence[tuple[str, Mapping[str, Any] | None]],
    *,
    repository_authority: Any = None,
    count_before: int | None = None,
    count_after: int | None = None,
) -> ChromaLogicalFingerprint:
    """Build a deterministic fingerprint from one already-bounded logical snapshot."""

    selected = _definition(semantic_collection_id)
    _validate_definition(selected)
    if selected.semantic_id != semantic_collection_id:
        raise ChromaLogicalSchemaMismatch("logical_collection_semantic_id_mismatch")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ChromaLogicalSnapshotUnstable("invalid_logical_records")
    if len(records) > MAX_LOGICAL_RECORDS:
        raise ChromaLogicalRecordLimitExceeded("logical_record_limit_exceeded")
    before = _strict_nonnegative_count(
        len(records) if count_before is None else count_before,
        "invalid_logical_count_before",
    )
    after = _strict_nonnegative_count(
        len(records) if count_after is None else count_after,
        "invalid_logical_count_after",
    )
    if before != after:
        raise ChromaLogicalSnapshotUnstable("logical_snapshot_count_changed")
    if before != len(records):
        raise ChromaLogicalSnapshotUnstable("logical_record_count_mismatch")

    authority_payload, repository_mapping = _authority_payload(
        selected, repository_authority
    )
    normalized: list[tuple[str, Mapping[str, Any] | None]] = []
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ChromaLogicalSnapshotUnstable("invalid_logical_record_shape")
        record_id, metadata = item
        if (
            not isinstance(record_id, str)
            or not record_id
            or len(record_id.encode("utf-8")) > MAX_RECORD_ID_BYTES
        ):
            raise ChromaLogicalSnapshotUnstable("invalid_logical_record_id")
        if record_id in seen:
            raise ChromaLogicalSnapshotUnstable("duplicate_logical_record_id")
        seen.add(record_id)
        normalized.append((record_id, metadata))
    normalized.sort(key=lambda item: item[0])

    metadata_rows: list[list[Any]] = []
    total_metadata_bytes = 0
    for record_id, metadata in normalized:
        fields, encoded_size = _canonical_metadata(selected, metadata, repository_mapping)
        total_metadata_bytes += encoded_size
        if total_metadata_bytes > MAX_TOTAL_METADATA_BYTES:
            raise ChromaLogicalMetadataUnsafe("logical_metadata_total_limit_exceeded")
        metadata_rows.append([record_id, fields])

    record_id_digest = _digest([record_id for record_id, _ in normalized])
    metadata_digest = _digest(metadata_rows)
    authority_digest = _digest(authority_payload)
    aggregate_payload = {
        "schema": CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
        "collection_semantic_id": selected.semantic_id,
        "collection_name": selected.collection_name,
        "collection_schema_version": selected.schema_version,
        "record_count": len(normalized),
        "record_id_digest": record_id_digest,
        "metadata_digest": metadata_digest,
        "authority_digest": authority_digest,
    }
    return ChromaLogicalFingerprint(
        **aggregate_payload,
        aggregate_digest=_digest(aggregate_payload),
        integrity_state="valid",
    )


def retrieve_logical_collection_records(
    handle: Any,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    maximum_records: int = MAX_LOGICAL_RECORDS,
) -> tuple[list[tuple[str, Mapping[str, Any] | None]], int, int]:
    """Read a stable bounded metadata-only snapshot through a validated handle."""

    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= DEFAULT_PAGE_SIZE
    ):
        raise ChromaLogicalRecordLimitExceeded("invalid_logical_page_size")
    if (
        not isinstance(maximum_records, int)
        or isinstance(maximum_records, bool)
        or not 1 <= maximum_records <= MAX_LOGICAL_RECORDS
    ):
        raise ChromaLogicalRecordLimitExceeded("invalid_logical_record_limit")
    try:
        before = handle.safe_count()
    except Exception as error:
        if isinstance(error, ChromaTransportError):
            raise
        raise ChromaLogicalSnapshotUnstable("logical_count_unavailable") from None
    before = _strict_nonnegative_count(before, "invalid_logical_count_before")
    if before > maximum_records:
        raise ChromaLogicalRecordLimitExceeded("logical_record_limit_exceeded")
    records: list[tuple[str, Mapping[str, Any] | None]] = []
    while len(records) < before:
        requested = min(page_size, before - len(records))
        try:
            page = handle.safe_get_page(limit=requested, offset=len(records))
        except Exception as error:
            if isinstance(error, ChromaTransportError):
                raise
            raise ChromaLogicalSnapshotUnstable("logical_page_unavailable") from None
        ids = getattr(page, "ids", None)
        metadatas = getattr(page, "metadatas", None)
        included = getattr(page, "included", None)
        if (
            not isinstance(ids, tuple)
            or not isinstance(metadatas, tuple)
            or included != ("metadatas",)
            or len(ids) != len(metadatas)
            or len(ids) != requested
        ):
            raise ChromaLogicalSnapshotUnstable("logical_pagination_incomplete")
        records.extend(zip(ids, metadatas))
        if len(records) > before:
            raise ChromaLogicalSnapshotUnstable("logical_pagination_overrun")
    try:
        after = handle.safe_count()
    except Exception as error:
        if isinstance(error, ChromaTransportError):
            raise
        raise ChromaLogicalSnapshotUnstable("logical_count_unavailable") from None
    after = _strict_nonnegative_count(after, "invalid_logical_count_after")
    if before != after or len(records) != before:
        raise ChromaLogicalSnapshotUnstable("logical_snapshot_count_changed")
    if len({record_id for record_id, _ in records}) != len(records):
        raise ChromaLogicalSnapshotUnstable("duplicate_logical_record_id")
    return records, before, after


def fingerprint_registered_collection(
    factory: ChromaHttpClientFactory,
    semantic_collection_id: str,
    *,
    repository_authority: Any = None,
    requested_lifecycle: ChromaAccessLifecycle | str = ChromaAccessLifecycle.READ,
    consumer_id: str = "central_http_collection_factory",
    page_size: int = DEFAULT_PAGE_SIZE,
) -> ChromaLogicalFingerprint:
    """Fingerprint one registered existing collection through the central factory."""

    _definition(semantic_collection_id)
    try:
        handle = factory.get_collection_handle(
            semantic_collection_id,
            requested_lifecycle,
            consumer_id,
            creation_requested=False,
        )
    except ChromaCollectionMissing:
        raise ChromaLogicalCollectionMissing("logical_collection_missing") from None
    records, before, after = retrieve_logical_collection_records(
        handle,
        page_size=page_size,
    )
    return build_logical_fingerprint(
        semantic_collection_id,
        records,
        repository_authority=repository_authority,
        count_before=before,
        count_after=after,
    )


def compare_logical_fingerprints(
    expected: ChromaLogicalFingerprint,
    actual: ChromaLogicalFingerprint | None,
) -> ChromaLogicalComparison:
    """Classify a comparison without revealing record IDs or metadata."""

    if not isinstance(expected, ChromaLogicalFingerprint):
        raise ChromaLogicalSchemaMismatch("invalid_expected_logical_fingerprint")
    if actual is None:
        return ChromaLogicalComparison("collection_missing", expected.collection_semantic_id, False)
    if not isinstance(actual, ChromaLogicalFingerprint):
        return ChromaLogicalComparison("invalid", expected.collection_semantic_id, False)
    if (
        expected.schema != actual.schema
        or expected.collection_semantic_id != actual.collection_semantic_id
        or expected.collection_name != actual.collection_name
        or expected.collection_schema_version != actual.collection_schema_version
    ):
        return ChromaLogicalComparison("schema_mismatch", expected.collection_semantic_id, False)
    if expected.record_count != actual.record_count:
        state = "record_count_mismatch"
    elif expected.record_id_digest != actual.record_id_digest:
        state = "record_identity_mismatch"
    elif expected.metadata_digest != actual.metadata_digest:
        state = "metadata_mismatch"
    elif expected.authority_digest != actual.authority_digest:
        state = "authority_mismatch"
    elif expected.integrity_state != "valid" or actual.integrity_state != "valid":
        state = "invalid"
    elif expected.aggregate_digest != actual.aggregate_digest:
        state = "invalid"
    else:
        state = "match"
    return ChromaLogicalComparison(state, expected.collection_semantic_id, state == "match")


def _fingerprint_all(
    factory: ChromaHttpClientFactory,
    repository_authority: Any,
) -> tuple[ChromaLogicalFingerprint, ...]:
    return tuple(
        fingerprint_registered_collection(
            factory,
            definition.semantic_id,
            repository_authority=repository_authority,
            requested_lifecycle=ChromaAccessLifecycle.TEST_ONLY,
            consumer_id="ephemeral_test_fixture",
        )
        for definition in list_registered_collections()
    )


def _close_factory(factory: ChromaHttpClientFactory | None) -> None:
    if factory is None:
        return
    try:
        factory.get_transport().close()
    except Exception:
        pass


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return int(probe.getsockname()[1])


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ChromaLogicalBaselineError("logical_baseline_clock_not_utc")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_logical_fingerprint_self_check() -> bool:
    """Exercise bounded synthetic sensitivity without network, storage, or tokens."""

    authority = build_project_repository_identity_authority(
        project_memory={"projects": [{"project_id": "logical-check"}]},
        user_confirmed_links=[
            {
                "project_id": "logical-check",
                "repository": "workagent/logical-check",
                "confirmed": True,
            }
        ],
    )
    metadata = {
        "project_id": "logical-check",
        "repository_project_id": "logical-check",
        "repository": "workagent/logical-check",
        "source_type": "file",
        "ignored_field": "first",
    }
    first = build_logical_fingerprint(
        GITHUB_EVIDENCE_SEMANTIC_ID,
        [("logical-record", metadata)],
        repository_authority=authority,
    )
    ignored = build_logical_fingerprint(
        GITHUB_EVIDENCE_SEMANTIC_ID,
        [("logical-record", {**metadata, "ignored_field": "second"})],
        repository_authority=authority,
    )
    added = build_logical_fingerprint(
        GITHUB_EVIDENCE_SEMANTIC_ID,
        [("logical-record", metadata), ("logical-record-2", metadata)],
        repository_authority=authority,
    )
    allowed_change = build_logical_fingerprint(
        GITHUB_EVIDENCE_SEMANTIC_ID,
        [("logical-record", {**metadata, "source_type": "commit"})],
        repository_authority=authority,
    )
    cross_project_rejected = False
    unsafe_rejected = False
    try:
        build_logical_fingerprint(
            GITHUB_EVIDENCE_SEMANTIC_ID,
            [("logical-record", {**metadata, "repository_project_id": "other"})],
            repository_authority=authority,
        )
    except ChromaLogicalAuthorityViolation:
        cross_project_rejected = True
    try:
        build_logical_fingerprint(
            GITHUB_EVIDENCE_SEMANTIC_ID,
            [("logical-record", {**metadata, "document": "excluded"})],
            repository_authority=authority,
        )
    except ChromaLogicalMetadataUnsafe:
        unsafe_rejected = True
    return bool(
        first.aggregate_digest == ignored.aggregate_digest
        and first.aggregate_digest != added.aggregate_digest
        and first.aggregate_digest != allowed_change.aggregate_digest
        and cross_project_rejected
        and unsafe_rejected
    )


def _write_baseline_atomic(path: Path, baseline: ChromaLogicalBaseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(baseline.to_dict()) + b"\n"
    if len(payload) > MAX_LOGICAL_BASELINE_BYTES:
        raise ChromaLogicalBaselineError("logical_baseline_too_large")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        raise ChromaLogicalBaselineError("logical_baseline_write_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_logical_baseline(path: str | Path) -> ChromaLogicalBaseline:
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > MAX_LOGICAL_BASELINE_BYTES:
            raise OSError
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return ChromaLogicalBaseline.from_mapping(payload)
    except ChromaLogicalModelError as error:
        raise ChromaLogicalBaselineError(error.code) from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ChromaLogicalBaselineError("logical_baseline_unavailable") from None


def capture_logical_baseline_from_backup(
    backup_id: str,
    *,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
    output_root: str | Path = DEFAULT_LOGICAL_FINGERPRINT_ROOT,
    protected_root: str | Path = DEFAULT_PROTECTED_CHROMA_ROOT,
    repository_authority: Any = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    port_allocator: Callable[[], int] = _allocate_loopback_port,
) -> ChromaLogicalBaseline:
    """Capture pre-cutover logical evidence from a disposable verified restore."""

    manifest = load_verified_chroma_backup_manifest(backup_id, backup_root=backup_root)
    verification_before = verify_chroma_backup(backup_id, backup_root=backup_root)
    production_before = capture_protected_file_inventory(protected_root)
    authority = (
        repository_authority
        if repository_authority is not None
        else load_project_repository_identity_authority()
    )
    if authority is None:
        raise ChromaLogicalAuthorityViolation("repository_mapping_authority_unavailable")
    first: tuple[ChromaLogicalFingerprint, ...] = ()
    second: tuple[ChromaLogicalFingerprint, ...] = ()
    restarted: tuple[ChromaLogicalFingerprint, ...] = ()
    workspace_before: dict[str, Any] | None = None
    workspace_after: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="workagent-chroma-logical-") as temporary:
        root = Path(temporary)
        restored = root / "verified-restored"
        restore_chroma_backup(
            backup_id,
            target=restored,
            approved_target_root=root,
            backup_root=backup_root,
        )
        information = root / "server-information"
        information.mkdir()
        workspace = information / "logical-data"
        shutil.copytree(restored, workspace)
        workspace_before = capture_protected_file_inventory(workspace)
        deployment = ChromaDeploymentConfig(
            ChromaDeploymentMode.LOCAL_HTTP,
            LOOPBACK_HOST,
            port_allocator(),
            False,
            2.0,
        )
        lifecycle_config = build_chroma_server_lifecycle_config(
            deployment,
            information_root=information,
            persistence_path=workspace,
            runtime_state_directory=information / "runtime",
            startup_timeout_seconds=20.0,
            shutdown_timeout_seconds=5.0,
            endpoint_release_timeout_seconds=5.0,
            poll_interval_seconds=0.05,
            test_owned=True,
        )
        store = AtomicChromaServerStateStore(lifecycle_config)
        controller = ChromaServerLifecycleController(lifecycle_config, state_store=store)
        factory: ChromaHttpClientFactory | None = None
        try:
            started = controller.start()
            if started.state != "ready" or not started.process_owned:
                raise ChromaLogicalBaselineError("logical_baseline_server_not_ready")
            factory = ChromaHttpClientFactory(
                ChromaDeploymentConfig(
                    ChromaDeploymentMode.EPHEMERAL_TEST,
                    LOOPBACK_HOST,
                    deployment.port,
                    False,
                    2.0,
                ),
                test_context=True,
            )
            first = _fingerprint_all(factory, authority)
            second = _fingerprint_all(factory, authority)
            _close_factory(factory)
            factory = None
            controller.stop()
            restarted_state = controller.start()
            if restarted_state.state != "ready" or not restarted_state.process_owned:
                raise ChromaLogicalBaselineError("logical_baseline_restart_failed")
            factory = ChromaHttpClientFactory(
                ChromaDeploymentConfig(
                    ChromaDeploymentMode.EPHEMERAL_TEST,
                    LOOPBACK_HOST,
                    deployment.port,
                    False,
                    2.0,
                ),
                test_context=True,
            )
            restarted = _fingerprint_all(factory, authority)
        except ChromaLogicalFingerprintError:
            raise
        except Exception as error:
            code = getattr(error, "code", "logical_baseline_capture_failed")
            raise ChromaLogicalBaselineError(str(code)) from None
        finally:
            _close_factory(factory)
            try:
                if store.load() is not None:
                    controller.stop()
            except Exception:
                raise ChromaLogicalBaselineError("logical_baseline_server_shutdown_failed") from None
        workspace_after = capture_protected_file_inventory(workspace)
    verification_after = verify_chroma_backup(backup_id, backup_root=backup_root)
    production_after = capture_protected_file_inventory(protected_root)
    repeated = first == second
    restart_deterministic = first == restarted
    logical_stable = repeated and restart_deterministic
    immutable = verification_before == verification_after
    production_unchanged = production_before == production_after
    if not first or len(first) != len(list_registered_collections()):
        raise ChromaLogicalBaselineError("logical_baseline_collection_set_incomplete")
    if not logical_stable:
        raise ChromaLogicalSnapshotUnstable("logical_baseline_not_deterministic")
    if not immutable:
        raise ChromaLogicalBaselineError("immutable_backup_changed")
    if not production_unchanged:
        raise ChromaLogicalBaselineError("protected_production_storage_changed")
    if workspace_before is None or workspace_after is None:
        raise ChromaLogicalBaselineError("logical_workspace_fingerprint_unavailable")
    baseline = ChromaLogicalBaseline(
        schema=CHROMA_LOGICAL_BASELINE_SCHEMA,
        backup_id=backup_id,
        backup_aggregate_sha256=manifest.aggregate_sha256,
        chroma_version=manifest.chroma_package_version,
        registry_schema=CHROMA_COLLECTION_REGISTRY_SCHEMA,
        fingerprint_schema=CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
        created_at=_timestamp(clock),
        fingerprints=tuple(sorted(first, key=lambda item: item.collection_semantic_id)),
        repeated_run_deterministic=repeated,
        restart_deterministic=restart_deterministic,
        workspace_byte_fingerprint_before=workspace_before["aggregate_sha256"],
        workspace_byte_fingerprint_after=workspace_after["aggregate_sha256"],
        workspace_byte_mutated=(workspace_before != workspace_after),
        logical_fingerprints_stable=logical_stable,
        immutable_backup_unchanged=immutable,
        production_persistence_unchanged=production_unchanged,
        synthetic_validation_passed=run_logical_fingerprint_self_check(),
        privacy_safe=True,
    )
    target = Path(output_root) / f"{backup_id}.json"
    _write_baseline_atomic(target, baseline)
    return baseline


def evaluate_logical_integrity_gate(
    baseline: ChromaLogicalBaseline,
) -> ChromaLogicalIntegrityGate:
    if not isinstance(baseline, ChromaLogicalBaseline):
        return ChromaLogicalIntegrityGate(
            CHROMA_LOGICAL_GATE_SCHEMA, False, "blocked", 0, "logical_baseline_invalid"
        )
    expected = tuple(item.semantic_id for item in list_registered_collections())
    observed = tuple(item.collection_semantic_id for item in baseline.fingerprints)
    checks = (
        observed == expected,
        all(item.integrity_state == "valid" for item in baseline.fingerprints),
        baseline.registry_schema == CHROMA_COLLECTION_REGISTRY_SCHEMA,
        baseline.fingerprint_schema == CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
        baseline.repeated_run_deterministic,
        baseline.restart_deterministic,
        baseline.workspace_byte_mutated,
        baseline.logical_fingerprints_stable,
        baseline.immutable_backup_unchanged,
        baseline.production_persistence_unchanged,
        baseline.synthetic_validation_passed,
        baseline.privacy_safe,
    )
    ready = all(checks)
    return ChromaLogicalIntegrityGate(
        schema=CHROMA_LOGICAL_GATE_SCHEMA,
        logical_collection_fingerprints_ready=ready,
        production_logical_integrity_gate=("logical_integrity_ready" if ready else "blocked"),
        collection_count=len(baseline.fingerprints),
        blocker=None if ready else "logical_integrity_evidence_incomplete",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.chroma_logical_fingerprint",
        description="Capture or validate privacy-safe logical Chroma fingerprints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--backup", required=True)
    capture.add_argument("--output-root", type=Path, default=DEFAULT_LOGICAL_FINGERPRINT_ROOT)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "capture":
            baseline = capture_logical_baseline_from_backup(
                arguments.backup,
                output_root=arguments.output_root,
            )
            gate = evaluate_logical_integrity_gate(baseline)
            print(json.dumps({"baseline": baseline.safe_summary(), "gate": gate.safe_summary()}, sort_keys=True))
            return 0 if gate.logical_collection_fingerprints_ready else 1
        baseline = load_logical_baseline(arguments.baseline)
        gate = evaluate_logical_integrity_gate(baseline)
        print(json.dumps(gate.safe_summary(), sort_keys=True))
        return 0 if gate.logical_collection_fingerprints_ready else 1
    except (ChromaLogicalFingerprintError, ChromaLogicalModelError) as error:
        print(f"logical fingerprint failed code={error.code}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
