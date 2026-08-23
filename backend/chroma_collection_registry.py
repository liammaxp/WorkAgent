"""Deterministic semantic authority for approved Chroma collections.

This module is intentionally data-only.  It does not parse deployment
configuration, construct a client, inspect storage, or open a collection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from backend.chroma_access_models import (
    LIFECYCLE_CATEGORIES,
    ChromaAccessValidationError,
    validate_chroma_access_inventory,
)


CHROMA_COLLECTION_REGISTRY_SCHEMA = "chroma_collection_registry.v1"
CHROMA_COLLECTION_INVENTORY_SYNC_SCHEMA = "chroma_collection_inventory_sync.v1"

PROFILE_FACTS_SEMANTIC_ID = "profile_facts"
GITHUB_EVIDENCE_SEMANTIC_ID = "github_evidence"
PROFILE_FACTS_COLLECTION_NAME = "profile_facts"
GITHUB_EVIDENCE_COLLECTION_NAME = "github_evidence"

LIFECYCLE_ORDER = (
    "read",
    "vector_query",
    "write",
    "index",
    "migration",
    "maintenance",
    "test_only",
)
CONSUMER_CATEGORY_FOR_LIFECYCLE = MappingProxyType(
    {
        "read": "readers",
        "vector_query": "vector_query_consumers",
        "write": "writers",
        "index": "indexers",
        "migration": "migration_tools",
        "maintenance": "maintenance_tools",
        "test_only": "test_consumers",
    }
)

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SAFE_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}\.v[1-9][0-9]{0,3}$")
_SAFE_DESCRIPTION_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,240}$")
_SAFE_METADATA_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:^|[\s'\"])(?:[a-z]:[\\/]|/|\\\\)")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_-]?key|credential|password|secret|token)\s*[:=]"
)
_UNSAFE_METADATA_KEYS = frozenset(
    {
        "absolute_path",
        "api_key",
        "credential",
        "diff_body",
        "document",
        "documents",
        "embedding",
        "embeddings",
        "filesystem_path",
        "password",
        "patch",
        "patch_body",
        "raw_metadata",
        "raw_text",
        "secret",
        "source_body",
        "token",
    }
)
_UNSTABLE_METADATA_KEYS = frozenset(
    {"created_at", "run_id", "timestamp", "updated_at"}
)
REJECTED_LOGICAL_INTEGRITY_METADATA_FIELDS = tuple(sorted(_UNSAFE_METADATA_KEYS))
EXCLUDED_VOLATILE_LOGICAL_INTEGRITY_METADATA_FIELDS = tuple(
    sorted(_UNSTABLE_METADATA_KEYS)
)
_SECRET_METADATA_FRAGMENTS = (
    "api_key",
    "credential",
    "password",
    "secret",
    "token",
)


class ChromaCollectionRegistryError(ValueError):
    """Bounded semantic validation failure with no runtime or storage data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class DuplicateCollectionSemanticId(ChromaCollectionRegistryError):
    pass


class DuplicatePhysicalCollectionName(ChromaCollectionRegistryError):
    pass


class DuplicateCollectionSchemaVersion(ChromaCollectionRegistryError):
    pass


class UnknownCollectionSemanticId(ChromaCollectionRegistryError):
    pass


class UnknownCollectionName(ChromaCollectionRegistryError):
    pass


class InvalidCollectionLifecycle(ChromaCollectionRegistryError):
    pass


class InvalidCollectionConsumer(ChromaCollectionRegistryError):
    pass


class UnsafeAutomaticCollectionCreation(ChromaCollectionRegistryError):
    pass


class InvalidCollectionAuthorityRequirement(ChromaCollectionRegistryError):
    pass


class InvalidCollectionSchemaVersion(ChromaCollectionRegistryError):
    pass


class UnsafeLogicalIntegrityMetadataField(ChromaCollectionRegistryError):
    pass


class InvalidCollectionDefinition(ChromaCollectionRegistryError):
    pass


class UnknownDynamicCollectionResolution(ChromaCollectionRegistryError):
    pass


class InventoryRegistrySynchronizationError(ChromaCollectionRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class CollectionAuthorityRequirements:
    project_id_required: bool
    repository_identity_required: bool
    repository_mapping_authority_required: bool
    project_isolation_required: bool
    profile_identity_required: bool
    profile_scope_required: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "project_id_required": self.project_id_required,
            "repository_identity_required": self.repository_identity_required,
            "repository_mapping_authority_required": self.repository_mapping_authority_required,
            "project_isolation_required": self.project_isolation_required,
            "profile_identity_required": self.profile_identity_required,
            "profile_scope_required": self.profile_scope_required,
        }


@dataclass(frozen=True, slots=True)
class ApprovedCollectionConsumers:
    readers: tuple[str, ...] = ()
    vector_query_consumers: tuple[str, ...] = ()
    writers: tuple[str, ...] = ()
    indexers: tuple[str, ...] = ()
    migration_tools: tuple[str, ...] = ()
    maintenance_tools: tuple[str, ...] = ()
    test_consumers: tuple[str, ...] = ()

    def for_lifecycle(self, lifecycle: str) -> tuple[str, ...]:
        category = CONSUMER_CATEGORY_FOR_LIFECYCLE.get(lifecycle)
        if category is None:
            raise InvalidCollectionLifecycle("unknown_collection_lifecycle")
        return getattr(self, category)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            category: list(getattr(self, category))
            for category in CONSUMER_CATEGORY_FOR_LIFECYCLE.values()
        }


@dataclass(frozen=True, slots=True)
class LegacyCollectionConsumer:
    consumer_id: str
    inventory_owner: str
    current_states: tuple[str, ...]
    allowed_lifecycles: tuple[str, ...]
    migration_target: str
    later_work_item: str
    approved_future_access: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "inventory_owner": self.inventory_owner,
            "current_states": list(self.current_states),
            "allowed_lifecycles": list(self.allowed_lifecycles),
            "migration_target": self.migration_target,
            "later_work_item": self.later_work_item,
            "approved_future_access": self.approved_future_access,
        }


@dataclass(frozen=True, slots=True)
class ChromaCollectionDefinition:
    semantic_id: str
    collection_name: str
    schema_version: str
    owner: str
    description: str
    allowed_lifecycles: tuple[str, ...]
    automatic_creation: bool
    authority_requirements: CollectionAuthorityRequirements
    approved_consumers: ApprovedCollectionConsumers
    legacy_consumers: tuple[LegacyCollectionConsumer, ...]
    logical_integrity_metadata_allowlist: tuple[str, ...]
    forbidden_metadata_fields: tuple[str, ...]
    migration_owner: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "collection_name": self.collection_name,
            "schema_version": self.schema_version,
            "owner": self.owner,
            "description": self.description,
            "allowed_lifecycles": list(self.allowed_lifecycles),
            "automatic_creation": self.automatic_creation,
            "authority_requirements": self.authority_requirements.to_dict(),
            "approved_consumers": self.approved_consumers.to_dict(),
            "legacy_consumers": [consumer.to_dict() for consumer in self.legacy_consumers],
            "logical_integrity_metadata_allowlist": list(
                self.logical_integrity_metadata_allowlist
            ),
            "forbidden_metadata_fields": list(self.forbidden_metadata_fields),
            "migration_owner": self.migration_owner,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityCollectionAlias:
    module: str
    symbol: str
    semantic_id: str
    migration_target: str
    later_work_item: str


@dataclass(frozen=True, slots=True)
class DynamicCollectionBinding:
    module: str
    symbol: str
    inventory_owner: str
    semantic_ids: tuple[str, ...]
    review_note: str
    later_work_item: str


@dataclass(frozen=True, slots=True)
class InventoryConsumerRule:
    inventory_owner: str
    lifecycle: str
    consumer_id: str
    disposition: str
    accepted_current_states: tuple[str, ...] = ("accepted_existing_http_bridge",)


_COMMON_FORBIDDEN_METADATA_FIELDS = tuple(
    sorted(
        set(REJECTED_LOGICAL_INTEGRITY_METADATA_FIELDS)
        | set(EXCLUDED_VOLATILE_LOGICAL_INTEGRITY_METADATA_FIELDS)
    )
)

PROFILE_FACTS_COLLECTION = ChromaCollectionDefinition(
    semantic_id=PROFILE_FACTS_SEMANTIC_ID,
    collection_name=PROFILE_FACTS_COLLECTION_NAME,
    schema_version="profile_facts.v1",
    owner="profile_memory",
    description="Durable profile facts scoped to the application's existing single profile memory.",
    allowed_lifecycles=(
        "read",
        "vector_query",
        "write",
        "index",
        "migration",
        "maintenance",
        "test_only",
    ),
    automatic_creation=False,
    authority_requirements=CollectionAuthorityRequirements(
        project_id_required=False,
        repository_identity_required=False,
        repository_mapping_authority_required=False,
        project_isolation_required=False,
        profile_identity_required=True,
        profile_scope_required=True,
    ),
    approved_consumers=ApprovedCollectionConsumers(
        readers=(
            "central_http_collection_factory",
            "chroma_operational_reader",
            "profile_memory_reader",
        ),
        vector_query_consumers=("profile_memory_vector_reader",),
        writers=("profile_memory_writer",),
        indexers=("profile_memory_indexer",),
        migration_tools=("profile_memory_migration",),
        maintenance_tools=("profile_memory_maintenance",),
        test_consumers=("ephemeral_test_fixture",),
    ),
    legacy_consumers=(),
    logical_integrity_metadata_allowlist=("index", "is_list", "section"),
    forbidden_metadata_fields=_COMMON_FORBIDDEN_METADATA_FIELDS,
    migration_owner="profile_memory_access_migration",
    status="active",
)

GITHUB_EVIDENCE_COLLECTION = ChromaCollectionDefinition(
    semantic_id=GITHUB_EVIDENCE_SEMANTIC_ID,
    collection_name=GITHUB_EVIDENCE_COLLECTION_NAME,
    schema_version="github_evidence.v1",
    owner="github_evidence",
    description="Project-isolated GitHub evidence indexed under authoritative repository identity.",
    allowed_lifecycles=(
        "read",
        "vector_query",
        "index",
        "migration",
        "maintenance",
        "test_only",
    ),
    automatic_creation=False,
    authority_requirements=CollectionAuthorityRequirements(
        project_id_required=True,
        repository_identity_required=True,
        repository_mapping_authority_required=True,
        project_isolation_required=True,
        profile_identity_required=False,
        profile_scope_required=False,
    ),
    approved_consumers=ApprovedCollectionConsumers(
        readers=(
            "central_http_collection_factory",
            "chroma_operational_reader",
            "github_evidence_metadata_reader",
        ),
        vector_query_consumers=("github_evidence_vector_reader",),
        writers=(),
        indexers=("github_evidence_materializer",),
        migration_tools=("github_evidence_migration",),
        maintenance_tools=("github_evidence_maintenance",),
        test_consumers=("ephemeral_test_fixture",),
    ),
    legacy_consumers=(),
    logical_integrity_metadata_allowlist=(
        "chunk_type",
        "commit_sha",
        "project_id",
        "repository",
        "repository_project_id",
        "source_id",
        "source_type",
    ),
    forbidden_metadata_fields=_COMMON_FORBIDDEN_METADATA_FIELDS,
    migration_owner="github_evidence_access_migration",
    status="active",
)

REGISTERED_COLLECTIONS = tuple(
    sorted(
        (PROFILE_FACTS_COLLECTION, GITHUB_EVIDENCE_COLLECTION),
        key=lambda definition: definition.semantic_id,
    )
)

KNOWN_COLLECTION_SCHEMA_VERSIONS = MappingProxyType(
    {
        GITHUB_EVIDENCE_SEMANTIC_ID: "github_evidence.v1",
        PROFILE_FACTS_SEMANTIC_ID: "profile_facts.v1",
    }
)
KNOWN_COLLECTION_NAMES = MappingProxyType(
    {
        GITHUB_EVIDENCE_SEMANTIC_ID: GITHUB_EVIDENCE_COLLECTION_NAME,
        PROFILE_FACTS_SEMANTIC_ID: PROFILE_FACTS_COLLECTION_NAME,
    }
)
EXPECTED_AUTHORITY_REQUIREMENTS = MappingProxyType(
    {
        GITHUB_EVIDENCE_SEMANTIC_ID: GITHUB_EVIDENCE_COLLECTION.authority_requirements,
        PROFILE_FACTS_SEMANTIC_ID: PROFILE_FACTS_COLLECTION.authority_requirements,
    }
)

COMPATIBILITY_COLLECTION_ALIASES = (
    CompatibilityCollectionAlias(
        module="backend/chroma_http_vector_search.py",
        symbol="GITHUB_EVIDENCE_COLLECTION",
        semantic_id=GITHUB_EVIDENCE_SEMANTIC_ID,
        migration_target="registered_collection_name_import",
        later_work_item="read_vector_migration",
    ),
    CompatibilityCollectionAlias(
        module="backend/memory_store.py",
        symbol="GITHUB_COLLECTION",
        semantic_id=GITHUB_EVIDENCE_SEMANTIC_ID,
        migration_target="registered_collection_name_import",
        later_work_item="write_index_migration",
    ),
    CompatibilityCollectionAlias(
        module="backend/memory_store.py",
        symbol="PROFILE_COLLECTION",
        semantic_id=PROFILE_FACTS_SEMANTIC_ID,
        migration_target="registered_collection_name_import",
        later_work_item="write_index_migration",
    ),
)

DYNAMIC_COLLECTION_BINDINGS = (
    DynamicCollectionBinding(
        module="backend/chroma_http_client_factory.py",
        symbol="ChromaHttpClientFactory.get_collection_handle",
        inventory_owner="central_http_client_factory",
        semantic_ids=(GITHUB_EVIDENCE_SEMANTIC_ID, PROFILE_FACTS_SEMANTIC_ID),
        review_note="The factory passes only validated semantic IDs to the bounded existing-only transport lookup.",
        later_work_item="central_http_client",
    ),
    DynamicCollectionBinding(
        module="tests/chroma_persistence_test_support.py",
        symbol="read_test_owned_collection_snapshot",
        inventory_owner="chroma_persistence_test_probe",
        semantic_ids=(GITHUB_EVIDENCE_SEMANTIC_ID, PROFILE_FACTS_SEMANTIC_ID),
        review_note="Guarded parity reads use only a verified-stopped disposable restored copy.",
        later_work_item="test_infrastructure",
    ),
    DynamicCollectionBinding(
        module="tests/chroma_http_test_support.py",
        symbol="_create_registered_collection_for_test",
        inventory_owner="ephemeral_http_test_fixture",
        semantic_ids=(GITHUB_EVIDENCE_SEMANTIC_ID, PROFILE_FACTS_SEMANTIC_ID),
        review_note="Explicit test preparation resolves only definitions from the production registry into disposable storage.",
        later_work_item="test_infrastructure",
    ),
    DynamicCollectionBinding(
        module="tests/chroma_http_test_support.py",
        symbol="query_collection_for_test",
        inventory_owner="ephemeral_http_test_fixture",
        semantic_ids=(GITHUB_EVIDENCE_SEMANTIC_ID, PROFILE_FACTS_SEMANTIC_ID),
        review_note="Synthetic vector queries receive only collection accessors already gated by the central factory.",
        later_work_item="test_infrastructure",
    ),
    DynamicCollectionBinding(
        module="tests/chroma_http_test_support.py",
        symbol="read_collection_for_test",
        inventory_owner="ephemeral_http_test_fixture",
        semantic_ids=(GITHUB_EVIDENCE_SEMANTIC_ID, PROFILE_FACTS_SEMANTIC_ID),
        review_note="Synthetic reads receive only collection accessors already gated by the central factory.",
        later_work_item="test_infrastructure",
    ),
)

INVENTORY_CONSUMER_RULES = (
    InventoryConsumerRule(
        "central_http_client_factory",
        "read",
        "central_http_collection_factory",
        "approved",
        ("accepted_central_http_factory",),
    ),
    InventoryConsumerRule(
        "github_vector_http_bridge", "read", "github_evidence_metadata_reader", "approved"
    ),
    InventoryConsumerRule(
        "github_vector_http_bridge",
        "vector_query",
        "github_evidence_vector_reader",
        "approved",
    ),
    InventoryConsumerRule(
        "synthetic_sqlite_fixture", "test_only", "ephemeral_test_fixture", "test"
    ),
    InventoryConsumerRule(
        "ephemeral_http_test_fixture",
        "test_only",
        "ephemeral_test_fixture",
        "test",
        ("accepted_ephemeral_http_fixture",),
    ),
    InventoryConsumerRule(
        "chroma_persistence_test_probe",
        "test_only",
        "ephemeral_test_fixture",
        "test",
        ("accepted_temporary_fixture",),
    ),
)


def _is_safe_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_SAFE_IDENTIFIER_RE.fullmatch(value))


def _require_canonical_tuple(values: Any, code: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(not isinstance(value, str) for value in values):
        raise InvalidCollectionDefinition(code)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise InvalidCollectionDefinition(code)
    return values


def _validate_metadata_fields(definition: ChromaCollectionDefinition) -> None:
    allowlist = _require_canonical_tuple(
        definition.logical_integrity_metadata_allowlist,
        "non_deterministic_metadata_allowlist",
    )
    forbidden = _require_canonical_tuple(
        definition.forbidden_metadata_fields,
        "non_deterministic_forbidden_metadata_fields",
    )
    if set(allowlist) & set(forbidden):
        raise UnsafeLogicalIntegrityMetadataField("metadata_allowlist_forbidden_overlap")
    for field in (*allowlist, *forbidden):
        if not _SAFE_METADATA_KEY_RE.fullmatch(field):
            raise UnsafeLogicalIntegrityMetadataField("invalid_metadata_field")
    for field in allowlist:
        if (
            field in _UNSAFE_METADATA_KEYS
            or field in _UNSTABLE_METADATA_KEYS
            or any(fragment in field for fragment in _SECRET_METADATA_FRAGMENTS)
        ):
            raise UnsafeLogicalIntegrityMetadataField("unsafe_logical_integrity_metadata_field")
    if not _UNSAFE_METADATA_KEYS.issubset(set(forbidden)):
        raise UnsafeLogicalIntegrityMetadataField("incomplete_forbidden_metadata_fields")


def _validate_approved_consumers(
    definition: ChromaCollectionDefinition,
) -> None:
    seen: set[str] = set()
    for lifecycle, category in CONSUMER_CATEGORY_FOR_LIFECYCLE.items():
        consumers = _require_canonical_tuple(
            getattr(definition.approved_consumers, category),
            "non_deterministic_approved_consumers",
        )
        if consumers and lifecycle not in definition.allowed_lifecycles:
            raise InvalidCollectionConsumer("consumer_category_lifecycle_not_allowed")
        for consumer_id in consumers:
            if not _is_safe_identifier(consumer_id) or consumer_id in seen:
                raise InvalidCollectionConsumer("invalid_or_duplicate_collection_consumer")
            seen.add(consumer_id)

    legacy_ids: set[str] = set()
    legacy_sort_keys = []
    for consumer in definition.legacy_consumers:
        if not isinstance(consumer, LegacyCollectionConsumer):
            raise InvalidCollectionConsumer("invalid_legacy_collection_consumer")
        legacy_sort_keys.append((consumer.consumer_id, consumer.inventory_owner))
        if (
            not _is_safe_identifier(consumer.consumer_id)
            or not _is_safe_identifier(consumer.inventory_owner)
            or consumer.consumer_id in seen
            or consumer.consumer_id in legacy_ids
        ):
            raise InvalidCollectionConsumer("invalid_or_duplicate_legacy_consumer")
        if consumer.approved_future_access:
            raise InvalidCollectionConsumer("legacy_consumer_permanently_approved")
        if (
            not consumer.migration_target
            or not consumer.later_work_item
            or not consumer.current_states
            or not consumer.allowed_lifecycles
        ):
            raise InvalidCollectionConsumer("legacy_consumer_missing_migration_metadata")
        if consumer.current_states != tuple(sorted(consumer.current_states)):
            raise InvalidCollectionConsumer("non_deterministic_legacy_consumer_states")
        if any(lifecycle not in LIFECYCLE_CATEGORIES for lifecycle in consumer.allowed_lifecycles):
            raise InvalidCollectionLifecycle("unknown_legacy_collection_lifecycle")
        legacy_ids.add(consumer.consumer_id)
    if legacy_sort_keys != sorted(legacy_sort_keys):
        raise InvalidCollectionConsumer("non_deterministic_legacy_consumers")


def _validate_collection_definition(definition: ChromaCollectionDefinition) -> None:
    if not isinstance(definition, ChromaCollectionDefinition):
        raise InvalidCollectionDefinition("invalid_collection_definition")
    for value in (definition.semantic_id, definition.collection_name, definition.owner):
        if not _is_safe_identifier(value):
            raise InvalidCollectionDefinition("invalid_collection_identifier")
    if not _SAFE_DESCRIPTION_RE.fullmatch(definition.description):
        raise InvalidCollectionDefinition("invalid_collection_description")
    if _ABSOLUTE_PATH_RE.search(definition.description) or _SECRET_VALUE_RE.search(
        definition.description
    ):
        raise InvalidCollectionDefinition("unsafe_collection_description")
    if definition.status != "active":
        raise InvalidCollectionDefinition("invalid_collection_status")
    if not _is_safe_identifier(definition.migration_owner):
        raise InvalidCollectionDefinition("invalid_collection_migration_owner")
    expected_schema = KNOWN_COLLECTION_SCHEMA_VERSIONS.get(definition.semantic_id)
    if (
        expected_schema is None
        or not _SAFE_SCHEMA_RE.fullmatch(definition.schema_version)
        or definition.schema_version != expected_schema
    ):
        raise InvalidCollectionSchemaVersion("unsupported_collection_schema_version")
    if definition.collection_name != KNOWN_COLLECTION_NAMES.get(definition.semantic_id):
        raise UnknownCollectionName("unknown_collection_name")
    expected_authority = EXPECTED_AUTHORITY_REQUIREMENTS.get(definition.semantic_id)
    if expected_authority is None or definition.authority_requirements != expected_authority:
        raise InvalidCollectionAuthorityRequirement("invalid_collection_authority_requirement")
    if not isinstance(definition.automatic_creation, bool):
        raise UnsafeAutomaticCollectionCreation("invalid_automatic_creation_policy")
    if definition.automatic_creation:
        raise UnsafeAutomaticCollectionCreation("automatic_collection_creation_forbidden")
    lifecycles = definition.allowed_lifecycles
    if (
        not isinstance(lifecycles, tuple)
        or not lifecycles
        or len(lifecycles) != len(set(lifecycles))
        or any(lifecycle not in LIFECYCLE_CATEGORIES for lifecycle in lifecycles)
        or lifecycles != tuple(item for item in LIFECYCLE_ORDER if item in lifecycles)
    ):
        raise InvalidCollectionLifecycle("invalid_collection_lifecycle_order")
    _validate_approved_consumers(definition)
    _validate_metadata_fields(definition)


def validate_collection_registry(
    definitions: Sequence[ChromaCollectionDefinition] = REGISTERED_COLLECTIONS,
) -> None:
    """Validate deterministic collection semantics without runtime access."""

    if not isinstance(definitions, (tuple, list)) or not definitions:
        raise InvalidCollectionDefinition("invalid_collection_registry")
    semantic_ids = [definition.semantic_id for definition in definitions]
    collection_names = [definition.collection_name for definition in definitions]
    schema_versions = [definition.schema_version for definition in definitions]
    if len(semantic_ids) != len(set(semantic_ids)):
        raise DuplicateCollectionSemanticId("duplicate_collection_semantic_id")
    if len(collection_names) != len(set(collection_names)):
        raise DuplicatePhysicalCollectionName("duplicate_physical_collection_name")
    if len(schema_versions) != len(set(schema_versions)):
        raise DuplicateCollectionSchemaVersion("duplicate_collection_schema_version")
    if semantic_ids != sorted(semantic_ids):
        raise InvalidCollectionDefinition("non_deterministic_collection_order")
    if set(semantic_ids) != set(KNOWN_COLLECTION_SCHEMA_VERSIONS):
        raise UnknownCollectionSemanticId("unknown_or_missing_collection_semantic_id")
    for definition in definitions:
        _validate_collection_definition(definition)


validate_collection_registry()

_COLLECTIONS_BY_SEMANTIC_ID = MappingProxyType(
    {definition.semantic_id: definition for definition in REGISTERED_COLLECTIONS}
)
_SEMANTIC_ID_BY_COLLECTION_NAME = MappingProxyType(
    {definition.collection_name: definition.semantic_id for definition in REGISTERED_COLLECTIONS}
)


def list_registered_collections() -> tuple[ChromaCollectionDefinition, ...]:
    return REGISTERED_COLLECTIONS


def get_collection_definition(semantic_id: str) -> ChromaCollectionDefinition:
    if not isinstance(semantic_id, str) or semantic_id not in _COLLECTIONS_BY_SEMANTIC_ID:
        raise UnknownCollectionSemanticId("unknown_collection_semantic_id")
    return _COLLECTIONS_BY_SEMANTIC_ID[semantic_id]


def resolve_collection_name(semantic_id: str) -> str:
    return get_collection_definition(semantic_id).collection_name


def resolve_collection_semantic_id(collection_name: str) -> str:
    if (
        not isinstance(collection_name, str)
        or collection_name not in _SEMANTIC_ID_BY_COLLECTION_NAME
    ):
        raise UnknownCollectionName("unknown_collection_name")
    return _SEMANTIC_ID_BY_COLLECTION_NAME[collection_name]


def serialize_collection_registry() -> dict[str, Any]:
    """Return a detached, deterministic, JSON-compatible registry payload."""

    return {
        "schema": CHROMA_COLLECTION_REGISTRY_SCHEMA,
        "collections": [definition.to_dict() for definition in REGISTERED_COLLECTIONS],
    }


def safe_collection_registry_summary() -> dict[str, Any]:
    validate_collection_registry()
    return {
        "schema": CHROMA_COLLECTION_REGISTRY_SCHEMA,
        "collection_count": len(REGISTERED_COLLECTIONS),
        "semantic_ids": [definition.semantic_id for definition in REGISTERED_COLLECTIONS],
        "validation_state": "valid",
    }


def validate_collection_lifecycle(semantic_id: str, lifecycle: str) -> None:
    definition = get_collection_definition(semantic_id)
    if not isinstance(lifecycle, str) or lifecycle not in definition.allowed_lifecycles:
        raise InvalidCollectionLifecycle("collection_lifecycle_not_allowed")


def validate_collection_consumer(
    semantic_id: str,
    lifecycle: str,
    consumer_id: str,
    *,
    production_access: bool = True,
) -> None:
    definition = get_collection_definition(semantic_id)
    validate_collection_lifecycle(semantic_id, lifecycle)
    if not isinstance(consumer_id, str):
        raise InvalidCollectionConsumer("unknown_collection_consumer")
    approved = definition.approved_consumers.for_lifecycle(lifecycle)
    if consumer_id not in approved:
        raise InvalidCollectionConsumer("unknown_collection_consumer")
    if production_access and lifecycle == "test_only":
        raise InvalidCollectionConsumer("test_consumer_has_production_authority")


def _dynamic_semantic_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    matches = [
        binding
        for binding in DYNAMIC_COLLECTION_BINDINGS
        if binding.module == record.get("module")
        and binding.symbol == record.get("symbol")
        and binding.inventory_owner == record.get("current_owner")
    ]
    if len(matches) != 1:
        raise UnknownDynamicCollectionResolution("dynamic_collection_binding_missing")
    binding = matches[0]
    if (
        not binding.review_note
        or binding.later_work_item != record.get("later_work_item")
        or tuple(sorted(binding.semantic_ids)) != binding.semantic_ids
    ):
        raise UnknownDynamicCollectionResolution("invalid_dynamic_collection_review")
    for semantic_id in binding.semantic_ids:
        get_collection_definition(semantic_id)
    return binding.semantic_ids


def _inventory_semantic_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    collection = record.get("collection")
    if collection == "not_applicable":
        return ()
    if collection == "dynamic_collection":
        if record.get("collection_resolution") != "dynamic":
            raise UnknownDynamicCollectionResolution("invalid_dynamic_collection_resolution")
        return _dynamic_semantic_ids(record)
    return (resolve_collection_semantic_id(collection),)


def _consumer_rule(record: Mapping[str, Any]) -> InventoryConsumerRule:
    matches = [
        rule
        for rule in INVENTORY_CONSUMER_RULES
        if rule.inventory_owner == record.get("current_owner")
        and rule.lifecycle == record.get("lifecycle")
    ]
    if len(matches) != 1:
        raise InvalidCollectionConsumer("inventory_consumer_not_registered")
    return matches[0]


def _legacy_consumer(
    definition: ChromaCollectionDefinition,
    consumer_id: str,
) -> LegacyCollectionConsumer:
    matches = [
        consumer
        for consumer in definition.legacy_consumers
        if consumer.consumer_id == consumer_id
    ]
    if len(matches) != 1:
        raise InvalidCollectionConsumer("legacy_inventory_consumer_not_registered")
    return matches[0]


def _validate_inventory_access(
    record: Mapping[str, Any],
    definition: ChromaCollectionDefinition,
    rule: InventoryConsumerRule,
) -> None:
    lifecycle = record.get("lifecycle")
    if rule.disposition == "approved":
        if rule.consumer_id not in definition.approved_consumers.for_lifecycle(lifecycle):
            raise InvalidCollectionConsumer("approved_inventory_consumer_mismatch")
        if record.get("current_state") not in rule.accepted_current_states:
            raise InvalidCollectionConsumer("approved_inventory_state_mismatch")
    elif rule.disposition == "test":
        if (
            rule.consumer_id not in definition.approved_consumers.test_consumers
            or lifecycle != "test_only"
            or record.get("runtime") != "test_only"
        ):
            raise InvalidCollectionConsumer("test_inventory_consumer_mismatch")
    elif rule.disposition == "legacy":
        legacy = _legacy_consumer(definition, rule.consumer_id)
        if (
            legacy.approved_future_access
            or legacy.inventory_owner != record.get("current_owner")
            or record.get("current_state") not in legacy.current_states
            or lifecycle not in legacy.allowed_lifecycles
            or record.get("migration_target") != legacy.migration_target
            or record.get("later_work_item") != legacy.later_work_item
            or not record.get("migration_action")
        ):
            raise InvalidCollectionConsumer("legacy_inventory_migration_metadata_mismatch")
    else:
        raise InvalidCollectionConsumer("unknown_inventory_consumer_disposition")

    if lifecycle not in definition.allowed_lifecycles and rule.disposition != "legacy":
        raise InvalidCollectionLifecycle("inventory_collection_lifecycle_not_allowed")
    if lifecycle not in definition.allowed_lifecycles and rule.disposition == "legacy":
        legacy = _legacy_consumer(definition, rule.consumer_id)
        if lifecycle not in legacy.allowed_lifecycles:
            raise InvalidCollectionLifecycle("legacy_collection_lifecycle_not_migrating")

    if record.get("may_create_collection"):
        if lifecycle in {"read", "vector_query"}:
            raise UnsafeAutomaticCollectionCreation("read_lifecycle_collection_creation_forbidden")
        if rule.disposition not in {"legacy", "test"}:
            raise UnsafeAutomaticCollectionCreation("automatic_collection_creation_forbidden")
        if rule.disposition == "legacy" and rule.consumer_id != "legacy_embedded_collection_initializer":
            raise UnsafeAutomaticCollectionCreation("unreviewed_legacy_collection_creation")
    if record.get("operation") == "get or create collection" and lifecycle in {
        "read",
        "vector_query",
    }:
        raise UnsafeAutomaticCollectionCreation("read_lifecycle_collection_creation_forbidden")


def validate_inventory_against_collection_registry(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Synchronize reviewed access semantics with the registry, without source or Chroma I/O."""

    try:
        validate_chroma_access_inventory(inventory)
    except ChromaAccessValidationError as error:
        raise InventoryRegistrySynchronizationError(
            "invalid_chroma_access_inventory"
        ) from error
    validate_collection_registry()
    records = inventory["records"]
    non_collection_count = 0
    resolved_entry_count = 0
    resolved_binding_count = 0
    dispositions = {"approved": 0, "legacy": 0, "test": 0}
    for record in records:
        semantic_ids = _inventory_semantic_ids(record)
        if not semantic_ids:
            non_collection_count += 1
            continue
        lifecycle = record.get("lifecycle")
        matching_rules = [
            rule
            for rule in INVENTORY_CONSUMER_RULES
            if rule.inventory_owner == record.get("current_owner")
            and rule.lifecycle == lifecycle
        ]
        if not matching_rules and any(
            lifecycle not in get_collection_definition(semantic_id).allowed_lifecycles
            for semantic_id in semantic_ids
        ):
            raise InvalidCollectionLifecycle("inventory_collection_lifecycle_not_allowed")
        rule = _consumer_rule(record)
        for semantic_id in semantic_ids:
            _validate_inventory_access(record, get_collection_definition(semantic_id), rule)
            resolved_binding_count += 1
        dispositions[rule.disposition] += 1
        resolved_entry_count += 1
    return {
        "schema": CHROMA_COLLECTION_INVENTORY_SYNC_SCHEMA,
        "inventory_record_count": len(records),
        "resolved_collection_entry_count": resolved_entry_count,
        "resolved_collection_binding_count": resolved_binding_count,
        "non_collection_entry_count": non_collection_count,
        "approved_entry_count": dispositions["approved"],
        "legacy_entry_count": dispositions["legacy"],
        "test_entry_count": dispositions["test"],
        "inventory_digest": inventory["inventory_digest"],
        "validation_state": "valid",
    }
