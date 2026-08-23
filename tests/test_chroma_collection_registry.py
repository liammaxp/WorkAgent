from __future__ import annotations

import ast
import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest

from backend import chroma_collection_registry as registry
from backend.chroma_access_manifest import EXPECTED_INVENTORY_DIGEST, INVENTORY
from backend.chroma_access_models import build_inventory, stable_access_id
from backend.chroma_collection_literal_guard import (
    UnregisteredProductionCollectionLiteral,
    audit_collection_name_literals,
    validate_collection_name_literals,
)
from backend.chroma_collection_registry import (
    CHROMA_COLLECTION_REGISTRY_SCHEMA,
    GITHUB_EVIDENCE_COLLECTION,
    GITHUB_EVIDENCE_COLLECTION_NAME,
    PROFILE_FACTS_COLLECTION,
    PROFILE_FACTS_COLLECTION_NAME,
    ApprovedCollectionConsumers,
    CollectionAuthorityRequirements,
    DuplicateCollectionSchemaVersion,
    DuplicateCollectionSemanticId,
    DuplicatePhysicalCollectionName,
    InvalidCollectionAuthorityRequirement,
    InvalidCollectionConsumer,
    InvalidCollectionDefinition,
    InvalidCollectionLifecycle,
    InvalidCollectionSchemaVersion,
    InventoryRegistrySynchronizationError,
    LegacyCollectionConsumer,
    UnknownCollectionName,
    UnknownCollectionSemanticId,
    UnknownDynamicCollectionResolution,
    UnsafeAutomaticCollectionCreation,
    UnsafeLogicalIntegrityMetadataField,
    get_collection_definition,
    list_registered_collections,
    resolve_collection_name,
    resolve_collection_semantic_id,
    safe_collection_registry_summary,
    serialize_collection_registry,
    validate_collection_consumer,
    validate_collection_lifecycle,
    validate_collection_registry,
    validate_inventory_against_collection_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SOURCE = ROOT / "backend" / "chroma_collection_registry.py"
GUARD_SOURCE = ROOT / "backend" / "chroma_collection_literal_guard.py"


def synthetic_repository(tmp_path: Path, source: str, relative: str) -> Path:
    root = tmp_path / "repository"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return root


def inventory_with_change(predicate, **changes):
    records = copy.deepcopy(INVENTORY["records"])
    matches = [record for record in records if predicate(record)]
    assert len(matches) == 1
    matches[0].update(changes)
    matches[0]["access_id"] = stable_access_id(matches[0])
    return build_inventory(records)


def test_registry_schema_summary_and_serialization_are_deterministic():
    first = serialize_collection_registry()
    second = serialize_collection_registry()
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["schema"] == CHROMA_COLLECTION_REGISTRY_SCHEMA
    assert safe_collection_registry_summary() == {
        "schema": "chroma_collection_registry.v1",
        "collection_count": 2,
        "semantic_ids": ["github_evidence", "profile_facts"],
        "validation_state": "valid",
    }


def test_registry_definitions_are_immutable_and_serialization_is_detached():
    definition = get_collection_definition("github_evidence")
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.collection_name = "changed"
    with pytest.raises(TypeError):
        registry.KNOWN_COLLECTION_NAMES["github_evidence"] = "changed"
    payload = serialize_collection_registry()
    payload["collections"][0]["collection_name"] = "changed"
    assert resolve_collection_name("github_evidence") == "github_evidence"


def test_registry_order_ids_names_and_real_collection_set_are_exact():
    definitions = list_registered_collections()
    assert isinstance(definitions, tuple)
    assert [item.semantic_id for item in definitions] == [
        "github_evidence",
        "profile_facts",
    ]
    assert {item.collection_name for item in definitions} == {
        "github_evidence",
        "profile_facts",
    }
    assert len(definitions) == 2
    assert resolve_collection_name("profile_facts") == PROFILE_FACTS_COLLECTION_NAME
    assert resolve_collection_name("github_evidence") == GITHUB_EVIDENCE_COLLECTION_NAME


def test_duplicate_semantic_ids_physical_names_and_schema_authorities_fail():
    with pytest.raises(DuplicateCollectionSemanticId, match="duplicate_collection_semantic_id"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, GITHUB_EVIDENCE_COLLECTION))
    duplicate_name = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        collection_name=GITHUB_EVIDENCE_COLLECTION.collection_name,
    )
    with pytest.raises(DuplicatePhysicalCollectionName, match="duplicate_physical_collection_name"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, duplicate_name))
    duplicate_schema = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        schema_version=GITHUB_EVIDENCE_COLLECTION.schema_version,
    )
    with pytest.raises(DuplicateCollectionSchemaVersion, match="duplicate_collection_schema_version"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, duplicate_schema))


def test_non_deterministic_or_incomplete_registry_fails():
    with pytest.raises(InvalidCollectionDefinition, match="non_deterministic_collection_order"):
        validate_collection_registry(tuple(reversed(list_registered_collections())))
    with pytest.raises(UnknownCollectionSemanticId, match="unknown_or_missing_collection_semantic_id"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION,))


@pytest.mark.parametrize("value", ("", "unknown", " github_evidence", "GITHUB_EVIDENCE", None))
def test_unknown_semantic_ids_fail_closed(value):
    with pytest.raises(UnknownCollectionSemanticId, match="unknown_collection_semantic_id"):
        get_collection_definition(value)


@pytest.mark.parametrize("value", ("", "unknown", "github_evidence_test", "PROFILE_FACTS", None))
def test_unknown_physical_collection_names_fail_closed(value):
    with pytest.raises(UnknownCollectionName, match="unknown_collection_name"):
        resolve_collection_semantic_id(value)


def test_registry_rejects_unknown_names_and_schema_versions_in_synthetic_definitions():
    unknown_name = dataclasses.replace(PROFILE_FACTS_COLLECTION, collection_name="invented")
    with pytest.raises(UnknownCollectionName, match="unknown_collection_name"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, unknown_name))
    unknown_schema = dataclasses.replace(
        PROFILE_FACTS_COLLECTION, schema_version="profile_facts.v2"
    )
    with pytest.raises(InvalidCollectionSchemaVersion, match="unsupported_collection_schema_version"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, unknown_schema))


def test_environment_and_user_values_cannot_override_collection_names(monkeypatch):
    monkeypatch.setenv("PROFILE_COLLECTION", "attacker_profile")
    monkeypatch.setenv("GITHUB_EVIDENCE_COLLECTION", "attacker_evidence")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "attacker_dynamic")
    assert resolve_collection_name("profile_facts") == "profile_facts"
    assert resolve_collection_name("github_evidence") == "github_evidence"
    for user_value in ("attacker_profile", "attacker_evidence", "${COLLECTION}"):
        with pytest.raises(UnknownCollectionSemanticId):
            resolve_collection_name(user_value)


def test_registry_source_has_no_client_filesystem_network_or_environment_boundary():
    source = REGISTRY_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "chromadb" not in imports
    assert not imports & {"pathlib", "socket", "requests", "httpx", "subprocess"}
    assert "Persistent" + "Client(" not in source
    assert "Http" + "Client(" not in source
    assert "os." + "environ" not in source
    assert "dotenv" not in source.casefold()
    assert "information/" + "chroma" not in source
    assert "get_or_create_" + "collection" not in source
    assert "create_" + "collection(" not in source


def test_registry_resolution_performs_no_io(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registry resolution must perform no I/O")

    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "walk", forbidden)
    assert get_collection_definition("profile_facts") is PROFILE_FACTS_COLLECTION
    assert resolve_collection_name("github_evidence") == "github_evidence"
    assert len(list_registered_collections()) == 2
    validate_collection_registry()


def test_schema_owner_description_and_status_are_explicit_and_bounded():
    expected = {
        "github_evidence": ("github_evidence.v1", "github_evidence"),
        "profile_facts": ("profile_facts.v1", "profile_memory"),
    }
    for definition in list_registered_collections():
        assert (definition.schema_version, definition.owner) == expected[definition.semantic_id]
        assert 1 <= len(definition.description) <= 240
        assert definition.status == "active"
        assert not Path(definition.description).is_absolute()


def test_unsafe_bounded_descriptions_fail_validation():
    for description in ("", "x" * 241, "C:/private/storage", "api_key=private"):
        changed = dataclasses.replace(PROFILE_FACTS_COLLECTION, description=description)
        with pytest.raises(InvalidCollectionDefinition):
            validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))


def test_lifecycle_permissions_are_explicit_and_github_write_is_not_permanently_allowed():
    assert PROFILE_FACTS_COLLECTION.allowed_lifecycles == (
        "read",
        "vector_query",
        "write",
        "index",
        "migration",
        "maintenance",
        "test_only",
    )
    assert GITHUB_EVIDENCE_COLLECTION.allowed_lifecycles == (
        "read",
        "vector_query",
        "index",
        "migration",
        "maintenance",
        "test_only",
    )
    assert "write" not in GITHUB_EVIDENCE_COLLECTION.allowed_lifecycles
    assert GITHUB_EVIDENCE_COLLECTION.approved_consumers.writers == ()


@pytest.mark.parametrize("lifecycle", ("unknown", "status", "config", "create", ""))
def test_unknown_or_disallowed_lifecycle_is_rejected(lifecycle):
    with pytest.raises(InvalidCollectionLifecycle, match="collection_lifecycle_not_allowed"):
        validate_collection_lifecycle("github_evidence", lifecycle)


def test_lifecycle_order_duplicates_and_automatic_creation_fail_validation():
    reordered = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        allowed_lifecycles=("vector_query", "read", "write", "index", "migration", "maintenance", "test_only"),
    )
    with pytest.raises(InvalidCollectionLifecycle, match="invalid_collection_lifecycle_order"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, reordered))
    duplicate = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        allowed_lifecycles=("read", "read", "vector_query", "write", "index", "migration", "maintenance", "test_only"),
    )
    with pytest.raises(InvalidCollectionLifecycle):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, duplicate))
    auto = dataclasses.replace(PROFILE_FACTS_COLLECTION, automatic_creation=True)
    with pytest.raises(UnsafeAutomaticCollectionCreation, match="automatic_collection_creation_forbidden"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, auto))


def test_every_production_collection_prohibits_automatic_creation():
    assert all(definition.automatic_creation is False for definition in list_registered_collections())


def test_approved_consumer_categories_have_stable_disjoint_semantic_ids():
    for definition in list_registered_collections():
        categories = definition.approved_consumers.to_dict()
        all_consumers = [consumer for values in categories.values() for consumer in values]
        assert len(all_consumers) == len(set(all_consumers))
        assert all(consumer.replace("_", "a").isalnum() for consumer in all_consumers)
        assert set(categories["writers"]).isdisjoint(categories["indexers"])
        assert set(categories["readers"]).isdisjoint(categories["writers"])
        assert set(categories["migration_tools"]).isdisjoint(
            categories["maintenance_tools"]
        )


def test_known_consumers_validate_and_unknown_or_cross_category_consumers_fail():
    validate_collection_consumer("profile_facts", "read", "profile_memory_reader")
    validate_collection_consumer(
        "github_evidence", "vector_query", "github_evidence_vector_reader"
    )
    with pytest.raises(InvalidCollectionConsumer, match="unknown_collection_consumer"):
        validate_collection_consumer("profile_facts", "read", "profile_memory_writer")
    with pytest.raises(InvalidCollectionConsumer, match="unknown_collection_consumer"):
        validate_collection_consumer("github_evidence", "read", "unknown_consumer")


def test_test_consumers_do_not_gain_production_authority():
    with pytest.raises(InvalidCollectionConsumer, match="test_consumer_has_production_authority"):
        validate_collection_consumer(
            "profile_facts", "test_only", "ephemeral_test_fixture", production_access=True
        )
    validate_collection_consumer(
        "profile_facts", "test_only", "ephemeral_test_fixture", production_access=False
    )


def test_registry_has_no_remaining_legacy_consumers():
    assert all(not definition.legacy_consumers for definition in list_registered_collections())


def test_legacy_approval_or_missing_migration_metadata_fails():
    first = LegacyCollectionConsumer(
        consumer_id="synthetic_legacy_reader",
        inventory_owner="synthetic_legacy_owner",
        current_states=("legacy_embedded_access",),
        allowed_lifecycles=("read",),
        migration_target="central_http_client",
        later_work_item="synthetic_migration",
    )
    permanently_approved = dataclasses.replace(first, approved_future_access=True)
    changed = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        legacy_consumers=(permanently_approved,),
    )
    with pytest.raises(InvalidCollectionConsumer, match="legacy_consumer_permanently_approved"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))
    missing_target = dataclasses.replace(first, migration_target="")
    changed = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        legacy_consumers=(missing_target,),
    )
    with pytest.raises(InvalidCollectionConsumer, match="legacy_consumer_missing_migration_metadata"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))


def test_existing_http_bridge_and_future_central_client_ownership_are_represented():
    rules = {
        (rule.inventory_owner, rule.lifecycle): (rule.consumer_id, rule.disposition)
        for rule in registry.INVENTORY_CONSUMER_RULES
    }
    assert rules[("github_vector_http_bridge", "read")] == (
        "github_evidence_metadata_reader",
        "approved",
    )
    assert rules[("github_vector_http_bridge", "vector_query")] == (
        "github_evidence_vector_reader",
        "approved",
    )
    assert all(not definition.legacy_consumers for definition in list_registered_collections())


def test_consumer_lists_must_be_deterministic_and_semantic():
    consumers = dataclasses.replace(
        PROFILE_FACTS_COLLECTION.approved_consumers,
        readers=("z_reader", "a_reader"),
    )
    changed = dataclasses.replace(PROFILE_FACTS_COLLECTION, approved_consumers=consumers)
    with pytest.raises(InvalidCollectionDefinition, match="non_deterministic_approved_consumers"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))
    invalid = dataclasses.replace(
        PROFILE_FACTS_COLLECTION.approved_consumers,
        readers=("arbitrary.module.Reader",),
    )
    changed = dataclasses.replace(PROFILE_FACTS_COLLECTION, approved_consumers=invalid)
    with pytest.raises(InvalidCollectionConsumer, match="invalid_or_duplicate_collection_consumer"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))


def test_github_authority_requires_project_repository_mapping_and_isolation():
    authority = GITHUB_EVIDENCE_COLLECTION.authority_requirements
    assert authority == CollectionAuthorityRequirements(
        project_id_required=True,
        repository_identity_required=True,
        repository_mapping_authority_required=True,
        project_isolation_required=True,
        profile_identity_required=False,
        profile_scope_required=False,
    )


def test_profile_authority_matches_existing_single_profile_scope_only():
    authority = PROFILE_FACTS_COLLECTION.authority_requirements
    assert authority == CollectionAuthorityRequirements(
        project_id_required=False,
        repository_identity_required=False,
        repository_mapping_authority_required=False,
        project_isolation_required=False,
        profile_identity_required=True,
        profile_scope_required=True,
    )
    source = REGISTRY_SOURCE.read_text(encoding="utf-8").casefold()
    assert "multi" + "-user" not in source
    assert "auth" + "entication" not in source


def test_authority_requirements_are_exact_and_cannot_be_weakened():
    weakened = dataclasses.replace(
        GITHUB_EVIDENCE_COLLECTION.authority_requirements,
        project_isolation_required=False,
    )
    changed = dataclasses.replace(
        GITHUB_EVIDENCE_COLLECTION,
        authority_requirements=weakened,
    )
    with pytest.raises(InvalidCollectionAuthorityRequirement):
        validate_collection_registry((changed, PROFILE_FACTS_COLLECTION))


def test_registry_references_repository_authority_without_copying_mappings_or_raw_repositories():
    payload = json.dumps(serialize_collection_registry(), sort_keys=True)
    assert "repository_mapping_authority_required" in payload
    assert "repository_to_project" not in payload
    assert "alias_to_repository" not in payload
    assert "owner/repo" not in payload
    assert "github.com/" not in payload


def test_logical_integrity_allowlists_are_exact_stable_metadata_contracts():
    assert PROFILE_FACTS_COLLECTION.logical_integrity_metadata_allowlist == (
        "index",
        "is_list",
        "section",
    )
    assert GITHUB_EVIDENCE_COLLECTION.logical_integrity_metadata_allowlist == (
        "chunk_type",
        "commit_sha",
        "project_id",
        "repository",
        "repository_project_id",
        "source_id",
        "source_type",
    )
    for definition in list_registered_collections():
        assert definition.logical_integrity_metadata_allowlist == tuple(
            sorted(definition.logical_integrity_metadata_allowlist)
        )


@pytest.mark.parametrize(
    "field",
    (
        "document",
        "documents",
        "embedding",
        "embeddings",
        "patch",
        "patch_body",
        "source_body",
        "diff_body",
        "raw_metadata",
        "raw_text",
        "absolute_path",
        "filesystem_path",
        "api_key",
        "access_token",
        "password",
        "secret_value",
        "credential_value",
    ),
)
def test_unsafe_body_path_embedding_and_secret_metadata_cannot_be_allowlisted(field):
    allowlist = tuple(sorted((*PROFILE_FACTS_COLLECTION.logical_integrity_metadata_allowlist, field)))
    changed = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        logical_integrity_metadata_allowlist=allowlist,
    )
    with pytest.raises(UnsafeLogicalIntegrityMetadataField):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))


@pytest.mark.parametrize("field", ("created_at", "run_id", "timestamp", "updated_at"))
def test_transient_metadata_is_rejected_from_logical_integrity_allowlist(field):
    allowlist = tuple(sorted((*PROFILE_FACTS_COLLECTION.logical_integrity_metadata_allowlist, field)))
    changed = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        logical_integrity_metadata_allowlist=allowlist,
    )
    with pytest.raises(UnsafeLogicalIntegrityMetadataField):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, changed))


def test_forbidden_metadata_fields_are_deterministic_complete_and_disjoint():
    for definition in list_registered_collections():
        forbidden = definition.forbidden_metadata_fields
        assert forbidden == tuple(sorted(forbidden))
        assert set(forbidden).isdisjoint(definition.logical_integrity_metadata_allowlist)
        assert {"document", "embedding", "patch", "raw_metadata", "absolute_path"} <= set(
            forbidden
        )
    incomplete = dataclasses.replace(
        PROFILE_FACTS_COLLECTION,
        forbidden_metadata_fields=PROFILE_FACTS_COLLECTION.forbidden_metadata_fields[1:],
    )
    with pytest.raises(UnsafeLogicalIntegrityMetadataField, match="incomplete_forbidden_metadata_fields"):
        validate_collection_registry((GITHUB_EVIDENCE_COLLECTION, incomplete))


def test_inventory_and_registry_synchronize_ephemeral_http_test_extension():
    summary = validate_inventory_against_collection_registry(INVENTORY)
    assert summary["schema"] == "chroma_collection_inventory_sync.v1"
    assert summary["inventory_record_count"] == len(INVENTORY["records"])
    assert (
        summary["resolved_collection_entry_count"]
        + summary["non_collection_entry_count"]
        == summary["inventory_record_count"]
    )
    assert summary["legacy_entry_count"] == 0
    assert summary["inventory_digest"] == EXPECTED_INVENTORY_DIGEST
    assert summary["validation_state"] == "valid"
    assert INVENTORY["inventory_digest"] == EXPECTED_INVENTORY_DIGEST


def test_every_dynamic_inventory_access_has_an_explicit_reviewed_binding():
    dynamic = [record for record in INVENTORY["records"] if record["collection"] == "dynamic_collection"]
    assert len(dynamic) == 9
    keys = {
        (binding.module, binding.symbol, binding.inventory_owner)
        for binding in registry.DYNAMIC_COLLECTION_BINDINGS
    }
    assert all((item["module"], item["symbol"], item["current_owner"]) in keys for item in dynamic)
    assert all(binding.review_note and binding.later_work_item for binding in registry.DYNAMIC_COLLECTION_BINDINGS)


def test_unknown_inventory_collection_and_dynamic_resolution_fail_closed():
    unknown = inventory_with_change(
        lambda item: item["current_owner"] == "central_http_client_factory"
        and item["operation"] == "get collection",
        collection="unknown_collection",
    )
    with pytest.raises(UnknownCollectionName, match="unknown_collection_name"):
        validate_inventory_against_collection_registry(unknown)
    dynamic = inventory_with_change(
        lambda item: item["collection"] == "dynamic_collection"
        and item["current_owner"] == "central_http_client_factory",
        symbol="ChromaHttpClientFactory.unknown_dynamic_helper",
    )
    with pytest.raises(UnknownDynamicCollectionResolution, match="dynamic_collection_binding_missing"):
        validate_inventory_against_collection_registry(dynamic)


def test_impossible_collection_lifecycle_and_unapproved_consumer_fail():
    impossible = inventory_with_change(
        lambda item: item["current_owner"] == "central_http_client_factory"
        and item["collection"] == "dynamic_collection"
        and item["operation"] == "get collection",
        lifecycle="write",
    )
    with pytest.raises(InvalidCollectionLifecycle, match="inventory_collection_lifecycle_not_allowed"):
        validate_inventory_against_collection_registry(impossible)
    unapproved = inventory_with_change(
        lambda item: item["current_owner"] == "central_http_client_factory"
        and item["operation"] == "get collection",
        collection="profile_facts",
        collection_resolution="shared_constant",
        current_owner="arbitrary_consumer",
    )
    with pytest.raises(InvalidCollectionConsumer, match="inventory_consumer_not_registered"):
        validate_inventory_against_collection_registry(unapproved)


def test_removed_legacy_inventory_access_cannot_be_reintroduced():
    changed = inventory_with_change(
        lambda item: item["current_owner"] == "central_http_client_factory"
        and item["operation"] == "get collection",
        collection="profile_facts",
        collection_resolution="shared_constant",
        current_owner="memory_indexing",
        lifecycle="index",
        symbol="MemoryVectorStore._legacy_upsert_with_similarity",
        current_state="legacy_embedded_access",
    )
    with pytest.raises(InvalidCollectionConsumer, match="inventory_consumer_not_registered"):
        validate_inventory_against_collection_registry(changed)


def test_central_read_access_cannot_claim_collection_creation():
    changed = inventory_with_change(
        lambda item: item["current_owner"] == "central_http_client_factory"
        and item["lifecycle"] == "read"
        and item["operation"] == "get collection"
        and item["symbol"] == "ChromaHttpClientFactory.get_collection_handle"
        and item["collection"] != "not_applicable",
        may_create_collection=True,
    )
    with pytest.raises(UnsafeAutomaticCollectionCreation):
        validate_inventory_against_collection_registry(changed)


def test_test_only_fixture_cannot_claim_production_authority():
    changed = inventory_with_change(
        lambda item: item["current_owner"] == "ephemeral_http_test_fixture"
        and item["semantic_role"] == "create_collection",
        runtime="production",
    )
    with pytest.raises(InvalidCollectionConsumer, match="test_inventory_consumer_mismatch"):
        validate_inventory_against_collection_registry(changed)


def test_github_has_no_legacy_or_generic_writer_authority():
    summary = validate_inventory_against_collection_registry(INVENTORY)
    assert summary["legacy_entry_count"] == 0
    assert GITHUB_EVIDENCE_COLLECTION.legacy_consumers == ()
    assert GITHUB_EVIDENCE_COLLECTION.approved_consumers.writers == ()


def test_invalid_inventory_schema_is_reported_with_bounded_registry_error():
    invalid = copy.deepcopy(INVENTORY)
    invalid["inventory_digest"] = "0" * 64
    with pytest.raises(InventoryRegistrySynchronizationError, match="invalid_chroma_access_inventory"):
        validate_inventory_against_collection_registry(invalid)


def test_current_collection_literal_audit_has_only_authority_and_compatibility_aliases():
    report = validate_collection_name_literals(ROOT)
    assert report["candidate_count"] == report["allowed_count"] == 4
    assert report["violation_count"] == report["unknown_count"] == 0
    assert report["classification_counts"] == {
        "authoritative_registry": 2,
        "compatibility_alias": 2,
    }


def test_new_unregistered_or_duplicate_production_assignment_literal_fails(tmp_path):
    for literal in ("new_unregistered_collection", "github_evidence"):
        root = synthetic_repository(
            tmp_path / literal,
            f'collection_name = "{literal}"\n',
            "backend/new_consumer.py",
        )
        with pytest.raises(UnregisteredProductionCollectionLiteral):
            validate_collection_name_literals(root)


@pytest.mark.parametrize("call", ("get_collection", "get_or_create_collection"))
def test_direct_unknown_collection_call_fails_literal_guard(tmp_path, call):
    root = synthetic_repository(
        tmp_path,
        f'def run(client):\n    return client.{call}("unknown_collection")\n',
        "backend/new_consumer.py",
    )
    report = audit_collection_name_literals(root)
    assert report["violation_count"] == 1
    assert report["violations"][0]["source_kind"] == "direct_collection_call"
    with pytest.raises(UnregisteredProductionCollectionLiteral):
        validate_collection_name_literals(root)


def test_authoritative_registry_literal_and_compatibility_alias_pass_guard(tmp_path):
    authority = synthetic_repository(
        tmp_path / "authority",
        'GITHUB_EVIDENCE_COLLECTION_NAME = "github_evidence"\n',
        "backend/chroma_collection_registry.py",
    )
    assert validate_collection_name_literals(authority)["classification_counts"] == {
        "authoritative_registry": 1
    }
    alias = synthetic_repository(
        tmp_path / "alias",
        'PROFILE_COLLECTION = "profile_facts"\n',
        "backend/memory_store.py",
    )
    assert validate_collection_name_literals(alias)["classification_counts"] == {
        "compatibility_alias": 1
    }


def test_test_fixture_literal_passes_only_in_test_context(tmp_path):
    source = 'def fixture(client):\n    return client.get_collection("isolated_fixture")\n'
    test_root = synthetic_repository(tmp_path / "test", source, "tests/test_fixture.py")
    report = validate_collection_name_literals(test_root)
    assert report["classification_counts"] == {"sanitized_test_fixture": 1}
    with pytest.raises(UnregisteredProductionCollectionLiteral):
        validate_collection_name_literals(test_root, allow_test_fixtures=False)
    production_root = synthetic_repository(tmp_path / "production", source, "backend/fixture.py")
    with pytest.raises(UnregisteredProductionCollectionLiteral):
        validate_collection_name_literals(production_root)


def test_literal_guard_avoids_unrelated_string_false_positives(tmp_path):
    root = synthetic_repository(
        tmp_path,
        'message = "github_evidence"\ndef run(service):\n    return service.get("unknown")\n',
        "backend/unrelated.py",
    )
    report = validate_collection_name_literals(root)
    assert report["candidate_count"] == 0


def test_literal_guard_never_executes_sources_or_reads_excluded_storage(tmp_path):
    root = synthetic_repository(
        tmp_path,
        'raise RuntimeError("must not execute")\nmessage = "safe"\n',
        "backend/poison.py",
    )
    protected = root / "information" / "chroma"
    protected.mkdir(parents=True)
    (protected / "unreadable.py").write_bytes(b"\xff\xfe")
    assert validate_collection_name_literals(root)["candidate_count"] == 0


def test_registry_and_guard_have_semantic_names_and_no_frontend_or_later_runtime_work():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REGISTRY_SOURCE, GUARD_SOURCE, Path(__file__))
    )
    lowered = sources.casefold()
    assert "frontend/" + "src" not in lowered
    assert "phase" + "6_5" not in lowered
    assert "phase_" + "6_5" not in lowered
    assert "step" + "3" not in lowered
    assert "start_" + "server" not in lowered
    assert "stop_" + "server" not in lowered
    assert "backup_" + "collection" not in lowered


def test_registry_payload_has_no_runtime_objects_or_sensitive_values():
    payload = serialize_collection_registry()
    encoded = json.dumps(payload, sort_keys=True)
    assert "C:/" not in encoded and "F:/" not in encoded
    assert "BEGIN PRIVATE KEY" not in encoded
    assert "owner/repo" not in encoded
    assert all(
        not hasattr(value, "get_collection")
        for definition in list_registered_collections()
        for value in dataclasses.astuple(definition)
    )
