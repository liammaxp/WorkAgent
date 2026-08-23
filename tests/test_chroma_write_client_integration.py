from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import psutil
import pytest

from backend.chroma_backup_recovery import (
    DEFAULT_BACKUP_ROOT,
    restore_chroma_backup,
    verify_chroma_backup,
)
from backend.chroma_collection_registry import get_collection_definition
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_client_factory import ChromaHttpClientFactory
from backend.chroma_http_transport import ChromaCollectionMissing
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_read_client import ChromaReadClient
from backend.chroma_server_lifecycle import ChromaServerLifecycleController
from backend.chroma_server_lifecycle_models import (
    AtomicChromaServerStateStore,
    build_chroma_server_lifecycle_config,
)
from backend.chroma_write_client import ChromaWriteClient
from backend.chroma_write_models import ChromaWriteRecord
from backend.memory_store import LocalHashEmbedding
from backend.project_repository_identity import build_project_repository_identity_authority
from tests.chroma_http_test_support import (
    EphemeralChromaServer,
    allocate_dynamic_loopback_endpoint,
    is_loopback_port_releasable,
    prepare_registered_collection_for_test,
    wait_for_loopback_port_release,
)


pytestmark = pytest.mark.chroma_http_integration
REAL_BACKUP_ID = "20260810T053709Z-a47ec430cbf1"
PROTECTED_CHROMA_ROOT = Path(__file__).resolve().parents[1] / "information" / "chroma"


@pytest.fixture(scope="module")
def write_server(tmp_path_factory):
    storage_parent = tmp_path_factory.mktemp("chroma-write-integration")
    with EphemeralChromaServer(storage_parent) as server:
        prepare_registered_collection_for_test(server.endpoint, "profile_facts")
        prepare_registered_collection_for_test(server.endpoint, "github_evidence")
        yield server


def clients(server):
    config_provider = lambda: server.deployment_config(timeout_seconds=1.0)
    factory_builder = lambda config: ChromaHttpClientFactory(config)
    return (
        ChromaWriteClient(
            config_provider=config_provider,
            factory_builder=factory_builder,
        ),
        ChromaReadClient(
            config_provider=config_provider,
            factory_builder=factory_builder,
        ),
    )


def github_authority():
    return build_project_repository_identity_authority(
        project_memory={
            "projects": [
                {"project_id": "project-a", "repository": "owner/repo-a"},
                {"project_id": "project-b", "repository": "owner/repo-b"},
            ]
        }
    )


def test_real_profile_upsert_delete_and_idempotency_are_logically_visible(write_server):
    writer, reader = clients(write_server)
    records = [
        ChromaWriteRecord("profile-a", "Profile A", {"section": "a"}, (1.0, 0.0)),
        ChromaWriteRecord("profile-b", "Profile B", {"section": "b"}, (0.0, 1.0)),
    ]
    first = writer.upsert_records(
        "profile_facts",
        consumer_id="profile_memory_indexer",
        records=records,
    )
    repeated = writer.upsert_records(
        "profile_facts",
        consumer_id="profile_memory_indexer",
        records=records,
    )
    snapshot = reader.read_records(
        "profile_facts",
        consumer_id="profile_memory_reader",
        include_documents=True,
        metadata_fields=("section",),
        max_records=10,
    )
    assert first.accepted_count == repeated.accepted_count == 2
    assert {record.record_id for record in snapshot.records} == {"profile-a", "profile-b"}
    assert {record.document for record in snapshot.records} == {"Profile A", "Profile B"}

    writer.delete_records(
        "profile_facts",
        consumer_id="profile_memory_writer",
        ids=["profile-a"],
        lifecycle="write",
    )
    remaining = reader.read_records(
        "profile_facts",
        consumer_id="profile_memory_reader",
        include_documents=True,
        metadata_fields=("section",),
        max_records=10,
    )
    assert [record.record_id for record in remaining.records] == ["profile-b"]


def test_real_github_write_preserves_metadata_and_project_delete_isolation(write_server):
    writer, reader = clients(write_server)
    verified = github_authority()
    for suffix in ("a", "b"):
        project_id = f"project-{suffix}"
        repository = f"owner/repo-{suffix}"
        writer.upsert_records(
            "github_evidence",
            consumer_id="github_evidence_materializer",
            records=[
                ChromaWriteRecord(
                    f"evidence-{suffix}",
                    f"Evidence {suffix.upper()}",
                    {"repository": repository, "source": "synthetic"},
                    (1.0, 0.0) if suffix == "a" else (0.0, 1.0),
                )
            ],
            authority_metadata=[
                {
                    "project_id": project_id,
                    "repository": repository,
                    "repository_project_id": project_id,
                }
            ],
            repository_authority=verified,
        )
    visible = reader.read_records(
        "github_evidence",
        consumer_id="github_evidence_metadata_reader",
        include_documents=True,
        metadata_fields=("repository", "source"),
        max_records=10,
    )
    assert {record.record_id for record in visible.records} == {"evidence-a", "evidence-b"}
    assert {tuple(sorted(record.metadata)) for record in visible.records} == {
        ("repository", "source")
    }
    vector = reader.vector_query(
        "github_evidence",
        consumer_id="github_evidence_vector_reader",
        query_embedding=(1.0, 0.0),
        n_results=1,
        include_documents=True,
        metadata_fields=("repository",),
        include_distances=True,
    )
    assert vector.hits[0].record_id == "evidence-a"

    writer.delete_records(
        "github_evidence",
        consumer_id="github_evidence_materializer",
        ids=["evidence-a"],
        lifecycle="index",
        authority_metadata=[
            {
                "project_id": "project-a",
                "repository": "owner/repo-a",
                "repository_project_id": "project-a",
            }
        ],
        repository_authority=verified,
    )
    remaining = reader.read_records(
        "github_evidence",
        consumer_id="github_evidence_metadata_reader",
        include_documents=False,
        metadata_fields=("repository",),
        max_records=10,
    )
    assert [record.record_id for record in remaining.records] == ["evidence-b"]


def test_missing_collection_is_not_created_by_real_write_client(tmp_path):
    with EphemeralChromaServer(tmp_path) as server:
        writer, _reader = clients(server)
        with pytest.raises(ChromaCollectionMissing, match="chroma_collection_missing"):
            writer.upsert_records(
                "profile_facts",
                consumer_id="profile_memory_indexer",
                records=[
                    ChromaWriteRecord("one", "content", {"section": "one"}, (1.0, 0.0))
                ],
            )


def logical_snapshot(reader, semantic_id):
    definition = get_collection_definition(semantic_id)
    result = reader.read_records(
        semantic_id,
        consumer_id=(
            "profile_memory_reader"
            if semantic_id == "profile_facts"
            else "github_evidence_metadata_reader"
        ),
        include_documents=True,
        metadata_fields=definition.logical_integrity_metadata_allowlist,
        max_records=10_000,
    )
    return sorted(
        (
            record.record_id,
            hashlib.sha256((record.document or "").encode("utf-8")).hexdigest(),
            tuple(sorted(record.metadata.items())),
        )
        for record in result.records
    )


@pytest.mark.chroma_server_integration
def test_verified_restored_backup_round_trip_mutation_preserves_logical_state(tmp_path):
    backup_directory = Path(DEFAULT_BACKUP_ROOT) / REAL_BACKUP_ID
    if not backup_directory.is_dir():
        pytest.skip("verified operational backup unavailable")
    protected_before = capture_protected_file_inventory(PROTECTED_CHROMA_ROOT)
    backup_before = verify_chroma_backup(REAL_BACKUP_ID)
    restore_root = tmp_path / "restore-root"
    restore_root.mkdir()
    restored = restore_root / "verified-restored"
    restore_chroma_backup(
        REAL_BACKUP_ID,
        target=restored,
        approved_target_root=restore_root,
    )

    server_information = tmp_path / "server-information"
    server_information.mkdir()
    server_copy = server_information / "server-copy"
    shutil.copytree(restored, server_copy)
    endpoint = allocate_dynamic_loopback_endpoint()
    lifecycle = build_chroma_server_lifecycle_config(
        ChromaDeploymentConfig(
            ChromaDeploymentMode.LOCAL_HTTP,
            endpoint.host,
            endpoint.port,
            False,
            1.0,
        ),
        information_root=server_information,
        persistence_path=server_copy,
        runtime_state_directory=server_information / "runtime",
        startup_timeout_seconds=20.0,
        shutdown_timeout_seconds=5.0,
        endpoint_release_timeout_seconds=5.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )
    state_store = AtomicChromaServerStateStore(lifecycle)
    controller = ChromaServerLifecycleController(lifecycle, state_store=state_store)
    server_pid = None
    try:
        assert controller.start().state == "ready"
        state = state_store.load()
        assert state is not None
        server_pid = state.pid
        deployment = ChromaDeploymentConfig(
            ChromaDeploymentMode.EPHEMERAL_TEST,
            endpoint.host,
            endpoint.port,
            False,
            1.0,
        )
        writer = ChromaWriteClient(config_provider=lambda: deployment)
        reader = ChromaReadClient(config_provider=lambda: deployment)
        before = {
            semantic_id: logical_snapshot(reader, semantic_id)
            for semantic_id in ("profile_facts", "github_evidence")
        }
        embedder = LocalHashEmbedding()
        profile_document = "Profile memory section: restored_parity\n{}"
        github_document = "Approved GitHub evidence for synthetic/restored-parity\n{}"
        verified = build_project_repository_identity_authority(
            project_memory={
                "projects": [
                    {
                        "project_id": "restored-parity-project",
                        "repository": "synthetic/restored-parity",
                    }
                ]
            }
        )
        writer.upsert_records(
            "profile_facts",
            consumer_id="profile_memory_indexer",
            records=[
                ChromaWriteRecord(
                    "profile-restored-parity",
                    profile_document,
                    {"section": "restored_parity", "index": 0, "is_list": 0},
                    embedder.embed(profile_document),
                )
            ],
        )
        writer.upsert_records(
            "github_evidence",
            consumer_id="github_evidence_materializer",
            records=[
                ChromaWriteRecord(
                    "github-restored-parity",
                    github_document,
                    {
                        "repository": "synthetic/restored-parity",
                        "source": "restored-parity",
                    },
                    embedder.embed(github_document),
                )
            ],
            authority_metadata=[
                {
                    "project_id": "restored-parity-project",
                    "repository": "synthetic/restored-parity",
                    "repository_project_id": "restored-parity-project",
                }
            ],
            repository_authority=verified,
        )
        assert len(logical_snapshot(reader, "profile_facts")) == len(before["profile_facts"]) + 1
        assert len(logical_snapshot(reader, "github_evidence")) == len(before["github_evidence"]) + 1

        writer.delete_records(
            "profile_facts",
            consumer_id="profile_memory_writer",
            ids=["profile-restored-parity"],
            lifecycle="write",
        )
        writer.delete_records(
            "github_evidence",
            consumer_id="github_evidence_materializer",
            ids=["github-restored-parity"],
            lifecycle="index",
            authority_metadata=[
                {
                    "project_id": "restored-parity-project",
                    "repository": "synthetic/restored-parity",
                    "repository_project_id": "restored-parity-project",
                }
            ],
            repository_authority=verified,
        )
        assert {
            semantic_id: logical_snapshot(reader, semantic_id)
            for semantic_id in ("profile_facts", "github_evidence")
        } == before
    finally:
        state = state_store.load()
        if state is not None:
            try:
                controller.stop()
            except Exception:
                if psutil.pid_exists(state.pid):
                    process = psutil.Process(state.pid)
                    process.kill()
                    process.wait(timeout=5.0)
        assert wait_for_loopback_port_release(endpoint.port, timeout_seconds=5.0)
        if server_pid is not None:
            assert not psutil.pid_exists(server_pid)
        assert is_loopback_port_releasable(endpoint.port)

    assert capture_protected_file_inventory(PROTECTED_CHROMA_ROOT) == protected_before
    assert verify_chroma_backup(REAL_BACKUP_ID) == backup_before
