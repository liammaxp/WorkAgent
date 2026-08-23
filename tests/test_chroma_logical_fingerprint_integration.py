from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import chroma_logical_fingerprint as logical
from backend.chroma_backup_recovery import (
    DEFAULT_BACKUP_ROOT,
    verify_chroma_backup,
)
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaHttpClientFactory,
)
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.project_repository_identity import build_project_repository_identity_authority
from tests.chroma_http_test_support import (
    EphemeralChromaServer,
    is_loopback_port_releasable,
    prepare_registered_collection_for_test,
)


REAL_BACKUP_ID = "20260810T053709Z-a47ec430cbf1"


def authority():
    return build_project_repository_identity_authority(
        project_memory={"projects": [{"project_id": "project-a"}]},
        user_confirmed_links=[
            {
                "project_id": "project-a",
                "repository": "owner/repo",
                "confirmed": True,
            }
        ],
    )


def fingerprint_synthetic_payload(
    root: Path,
    *,
    documents: list[str],
    embeddings: list[list[float]],
):
    """Fingerprint one disposable payload without returning its content."""

    root.mkdir()
    server = EphemeralChromaServer(root)
    endpoint = server.start()
    factory = None
    try:
        prepare_registered_collection_for_test(
            endpoint,
            "github_evidence",
            ids=["record-b", "record-a"],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "project_id": "project-a",
                    "repository_project_id": "project-a",
                    "repository": "owner/repo",
                    "source_type": "file",
                },
                {
                    "project_id": "project-a",
                    "repository_project_id": "project-a",
                    "repository": "owner/repo",
                    "source_type": "commit",
                },
            ],
        )
        factory = ChromaHttpClientFactory(server.deployment_config(), test_context=True)
        return logical.fingerprint_registered_collection(
            factory,
            "github_evidence",
            repository_authority=authority(),
            requested_lifecycle=ChromaAccessLifecycle.TEST_ONLY,
            consumer_id="ephemeral_test_fixture",
        )
    finally:
        if factory is not None:
            factory.get_transport().close()
        port = endpoint.port
        server.stop()
        assert is_loopback_port_releasable(port)


def assert_authoritative_digests_equal(first, second):
    assert first.record_id_digest == second.record_id_digest
    assert first.metadata_digest == second.metadata_digest
    assert first.authority_digest == second.authority_digest
    assert first.aggregate_digest == second.aggregate_digest


@pytest.mark.chroma_http_integration
def test_document_mutation_does_not_change_authoritative_logical_fingerprint(tmp_path):
    first = fingerprint_synthetic_payload(
        tmp_path / "documents-first",
        documents=["first payload", "second payload"],
        embeddings=[[0.0, 1.0], [1.0, 0.0]],
    )
    changed = fingerprint_synthetic_payload(
        tmp_path / "documents-changed",
        documents=["changed first payload", "changed second payload"],
        embeddings=[[0.0, 1.0], [1.0, 0.0]],
    )
    assert_authoritative_digests_equal(first, changed)


@pytest.mark.chroma_http_integration
def test_embedding_mutation_does_not_change_authoritative_logical_fingerprint(tmp_path):
    first = fingerprint_synthetic_payload(
        tmp_path / "embeddings-first",
        documents=["first payload", "second payload"],
        embeddings=[[0.0, 1.0], [1.0, 0.0]],
    )
    changed = fingerprint_synthetic_payload(
        tmp_path / "embeddings-changed",
        documents=["first payload", "second payload"],
        embeddings=[[0.25, 0.75], [0.75, 0.25]],
    )
    assert_authoritative_digests_equal(first, changed)


@pytest.mark.chroma_http_integration
def test_real_http_fingerprint_is_metadata_only_deterministic_and_existing_only(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    endpoint = server.start()
    factory = None
    try:
        prepare_registered_collection_for_test(
            endpoint,
            "github_evidence",
            ids=["record-b", "record-a"],
            embeddings=[[0.0, 1.0], [1.0, 0.0]],
            metadatas=[
                {
                    "project_id": "project-a",
                    "repository_project_id": "project-a",
                    "repository": "owner/repo",
                    "source_type": "file",
                    "ignored": "first",
                },
                {
                    "project_id": "project-a",
                    "repository_project_id": "project-a",
                    "repository": "owner/repo",
                    "source_type": "commit",
                    "ignored": "second",
                },
            ],
        )
        factory = ChromaHttpClientFactory(server.deployment_config(), test_context=True)
        first = logical.fingerprint_registered_collection(
            factory,
            "github_evidence",
            repository_authority=authority(),
            requested_lifecycle=ChromaAccessLifecycle.TEST_ONLY,
            consumer_id="ephemeral_test_fixture",
            page_size=1,
        )
        second = logical.fingerprint_registered_collection(
            factory,
            "github_evidence",
            repository_authority=authority(),
            requested_lifecycle=ChromaAccessLifecycle.TEST_ONLY,
            consumer_id="ephemeral_test_fixture",
            page_size=2,
        )
        assert first == second
        assert first.record_count == 2
        safe = json.dumps(first.safe_summary()).casefold()
        assert "record-a" not in safe and "owner/repo" not in safe
        assert "document" not in safe and "embedding" not in safe
    finally:
        if factory is not None:
            factory.get_transport().close()
        port = endpoint.port
        server.stop()
        assert is_loopback_port_releasable(port)


@pytest.mark.chroma_http_integration
def test_missing_registered_collection_fails_without_automatic_creation(tmp_path):
    server = EphemeralChromaServer(tmp_path)
    endpoint = server.start()
    factory = ChromaHttpClientFactory(server.deployment_config(), test_context=True)
    try:
        with pytest.raises(logical.ChromaLogicalCollectionMissing, match="logical_collection_missing"):
            logical.fingerprint_registered_collection(
                factory,
                "profile_facts",
                requested_lifecycle=ChromaAccessLifecycle.TEST_ONLY,
                consumer_id="ephemeral_test_fixture",
            )
    finally:
        factory.get_transport().close()
        port = endpoint.port
        server.stop()
        assert is_loopback_port_releasable(port)


@pytest.mark.chroma_server_integration
def test_verified_real_backup_chain_produces_stable_safe_logical_baseline(tmp_path):
    backup_directory = Path(DEFAULT_BACKUP_ROOT) / REAL_BACKUP_ID
    if not backup_directory.is_dir():
        pytest.skip("verified operational backup unavailable")
    production_before = capture_protected_file_inventory(logical.DEFAULT_PROTECTED_CHROMA_ROOT)
    backup_before = verify_chroma_backup(REAL_BACKUP_ID)
    baseline = logical.capture_logical_baseline_from_backup(
        REAL_BACKUP_ID,
        output_root=tmp_path / "logical-artifacts",
    )
    gate = logical.evaluate_logical_integrity_gate(baseline)
    production_after = capture_protected_file_inventory(logical.DEFAULT_PROTECTED_CHROMA_ROOT)
    backup_after = verify_chroma_backup(REAL_BACKUP_ID)
    assert production_before == production_after
    assert backup_before == backup_after
    assert tuple(item.collection_semantic_id for item in baseline.fingerprints) == (
        "github_evidence",
        "profile_facts",
    )
    assert all(item.integrity_state == "valid" for item in baseline.fingerprints)
    assert baseline.repeated_run_deterministic is True
    assert baseline.restart_deterministic is True
    assert baseline.logical_fingerprints_stable is True
    assert baseline.immutable_backup_unchanged is True
    assert baseline.production_persistence_unchanged is True
    assert gate.production_logical_integrity_gate == "logical_integrity_ready"
    artifact = tmp_path / "logical-artifacts" / f"{REAL_BACKUP_ID}.json"
    loaded = logical.load_logical_baseline(artifact)
    assert loaded == baseline
    serialized = artifact.read_text(encoding="utf-8").casefold()
    assert "information/chroma" not in serialized
    assert "document" not in serialized and "embedding" not in serialized
