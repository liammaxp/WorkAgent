from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
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
from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_read_client import ChromaReadClient
from backend.chroma_server_lifecycle import ChromaServerLifecycleController
from backend.chroma_server_lifecycle_models import (
    AtomicChromaServerStateStore,
    build_chroma_server_lifecycle_config,
)
from backend.memory_store import LocalHashEmbedding
from tests.chroma_http_test_support import (
    PROTECTED_CHROMA_ROOT,
    allocate_dynamic_loopback_endpoint,
    is_loopback_port_releasable,
    wait_for_loopback_port_release,
)


REAL_BACKUP_ID = "20260810T053709Z-a47ec430cbf1"
QUERY = "retrieval evidence repository architecture"
QUERY_RESULTS = 5


def _legacy_snapshot(
    *,
    information_root: Path,
    persistence_path: Path,
    port: int,
) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.chroma_persistence_test_support",
            str(information_root),
            str(persistence_path),
            str(information_root / "runtime"),
            str(port),
            "--snapshot",
            "--query",
            QUERY,
            "--n-results",
            str(QUERY_RESULTS),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _content_hash(value: str | None) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else ""


@pytest.mark.chroma_server_integration
def test_verified_restored_backup_business_read_and_vector_parity(tmp_path):
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

    legacy_information = tmp_path / "legacy-information"
    legacy_information.mkdir()
    legacy_copy = legacy_information / "legacy-copy"
    shutil.copytree(restored, legacy_copy)
    legacy_port = allocate_dynamic_loopback_endpoint().port
    legacy = _legacy_snapshot(
        information_root=legacy_information,
        persistence_path=legacy_copy,
        port=legacy_port,
    )
    assert is_loopback_port_releasable(legacy_port)

    server_information = tmp_path / "server-information"
    server_information.mkdir()
    server_copy = server_information / "server-copy"
    shutil.copytree(restored, server_copy)
    endpoint = allocate_dynamic_loopback_endpoint()
    lifecycle_deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        endpoint.host,
        endpoint.port,
        False,
        1.0,
    )
    lifecycle = build_chroma_server_lifecycle_config(
        lifecycle_deployment,
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
        started = controller.start()
        state = state_store.load()
        assert started.state == "ready" and state is not None
        server_pid = state.pid
        read_deployment = ChromaDeploymentConfig(
            ChromaDeploymentMode.EPHEMERAL_TEST,
            endpoint.host,
            endpoint.port,
            False,
            1.0,
        )
        reader = ChromaReadClient(
            config_provider=lambda: read_deployment,
            factory_builder=lambda config: ChromaHttpClientFactory(
                config,
                test_context=True,
            ),
        )
        embedding = LocalHashEmbedding().embed(QUERY)
        for semantic_id, read_consumer, vector_consumer in (
            (
                "github_evidence",
                "github_evidence_metadata_reader",
                "github_evidence_vector_reader",
            ),
            ("profile_facts", "profile_memory_reader", "profile_memory_vector_reader"),
        ):
            definition = get_collection_definition(semantic_id)
            legacy_collection = legacy["collections"][semantic_id]
            records = reader.read_records(
                semantic_id,
                consumer_id=read_consumer,
                include_documents=True,
                metadata_fields=definition.logical_integrity_metadata_allowlist,
                max_records=10_000,
                page_size=1,
            )
            http_records = sorted(
                [
                    {
                        "id": record.record_id,
                        "content_sha256": _content_hash(record.document),
                        "metadata": dict(record.metadata),
                    }
                    for record in records.records
                ],
                key=lambda item: item["id"],
            )
            assert http_records == legacy_collection["records"]
            assert len(http_records) == legacy_collection["count"]

            vector = reader.vector_query(
                semantic_id,
                consumer_id=vector_consumer,
                query_embedding=embedding,
                n_results=QUERY_RESULTS,
                include_documents=True,
                metadata_fields=definition.logical_integrity_metadata_allowlist,
                include_distances=True,
            )
            expected_query = legacy_collection["query"]
            assert [hit.record_id for hit in vector.hits] == [item["id"] for item in expected_query]
            assert [_content_hash(hit.document) for hit in vector.hits] == [
                item["content_sha256"] for item in expected_query
            ]
            assert [dict(hit.metadata) for hit in vector.hits] == [
                item["metadata"] for item in expected_query
            ]
            assert [hit.distance for hit in vector.hits] == pytest.approx(
                [item["distance"] for item in expected_query],
                rel=1e-7,
                abs=1e-9,
            )
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
