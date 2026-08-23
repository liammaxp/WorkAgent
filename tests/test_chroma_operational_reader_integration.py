from __future__ import annotations

import json

import pytest

from backend.chroma_migration_baseline import capture_protected_file_inventory
from backend.chroma_operational_reader import ChromaOperationalReader
from tests.chroma_http_test_support import (
    EphemeralChromaServer,
    is_loopback_port_releasable,
    prepare_registered_collection_for_test,
)


@pytest.mark.chroma_http_integration
def test_operational_reader_real_server_counts_existence_inventory_and_shutdown(tmp_path):
    protected_before = capture_protected_file_inventory("information/chroma")
    server = EphemeralChromaServer(tmp_path)
    endpoint = server.start()
    deployment = server.deployment_config(timeout_seconds=0.5)
    try:
        prepare_registered_collection_for_test(
            endpoint,
            "profile_facts",
            ids=["profile-1", "profile-2"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[
                {"section": "skills", "index": "0", "is_list": "true"},
                {"section": "summary", "index": "0", "is_list": "false"},
            ],
        )
        prepare_registered_collection_for_test(
            endpoint,
            "github_evidence",
            ids=["private-a", "private-b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[
                {
                    "repository": "owner/repo",
                    "project_id": "project-a",
                    "source_type": "file",
                    "updated_at": "2026-08-10T01:00:00Z",
                    "raw_metadata": "not-returned",
                },
                {
                    "repository": "owner/repo",
                    "repository_project_id": "project-a",
                    "source_type": "commit",
                    "updated_at": "2026-08-10T02:00:00Z",
                    "document": "not-returned",
                },
            ],
        )
        reader = ChromaOperationalReader(config_provider=lambda: deployment)
        profile = reader.read_collection_status("profile_facts")
        github = reader.read_collection_status(
            "github_evidence", include_repository_inventory=True
        )
        assert profile.available and profile.safe_record_count == 2
        assert github.available and github.safe_record_count == 2
        assert [item.repository for item in github.repositories] == ["owner/repo"]
        assert github.repositories[0].updated_at == "2026-08-10T02:00:00Z"
        safe = json.dumps(github.safe_summary()).casefold()
        assert "private-a" not in safe
        assert "not-returned" not in safe
        assert "document" not in safe and "embedding" not in safe
    finally:
        port = endpoint.port
        server.stop()
        assert is_loopback_port_releasable(port)
    unavailable = ChromaOperationalReader(
        config_provider=lambda: deployment
    ).read_collection_status("profile_facts")
    assert unavailable.server_state == "unavailable"
    assert unavailable.integrity_state == "unavailable"
    assert unavailable.safe_record_count is None
    assert capture_protected_file_inventory("information/chroma") == protected_before


@pytest.mark.chroma_http_integration
def test_operational_reader_missing_collection_never_creates_it(tmp_path):
    with EphemeralChromaServer(tmp_path) as server:
        reader = ChromaOperationalReader(
            config_provider=lambda: server.deployment_config(timeout_seconds=0.5)
        )
        missing = reader.read_collection_status("github_evidence")
        assert missing.integrity_state == "collection_missing"
        assert missing.collection_available is False
        assert missing.safe_record_count is None
