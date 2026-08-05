from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


def test_unresolved_endpoint_returns_safe_product_schema():
    payload = {
        "status": "blocked", "repositories": [{"repository": "owner/repo", "canonical": True}],
        "unresolved_count": 1, "conflict_count": 0, "confirmation_required": True,
        "warnings": [], "errors": [],
    }
    with (
        patch.object(api_server.project_repository_mapping_service, "load_project_repository_identity_authority", return_value=None),
        patch.object(api_server.project_repository_mapping_service, "list_unresolved_repository_mappings", return_value=payload) as listing,
    ):
        response = TestClient(api_server.app).get("/api/github/repository-mappings/unresolved")
    assert response.status_code == 200 and response.json() == payload
    assert listing.call_args.kwargs["vector_store"] is api_server.agent.MEMORY_STORE
    assert not ({"raw_text", "documents", "embeddings", "metadata"} & set(json.dumps(response.json())))


def test_projects_endpoint_returns_only_safe_selection_fields():
    project_memory = {"projects": [{"project_id": "A", "project_name": "Alpha", "description": "private"}]}
    with patch.object(
        api_server.project_repository_mapping_service,
        "load_project_memory_for_repository_mapping", return_value=project_memory,
    ):
        response = TestClient(api_server.app).get("/api/github/repository-mappings/projects")
    assert response.status_code == 200
    assert response.json()["projects"] == [{
        "project_id": "A", "project_name": "Alpha", "already_linked_repositories": [],
    }]
    assert "private" not in response.text


def test_confirm_endpoint_is_strict_and_routes_to_service(tmp_path):
    project_memory = {"projects": [{"project_id": "A"}]}
    expected = {"status": "created", "mapping": {"project_id": "A", "repository": "owner/repo"}}
    with (
        patch.object(api_server.project_repository_mapping_service, "load_project_memory_for_repository_mapping", return_value=project_memory),
        patch.object(api_server.project_repository_mapping_service, "confirm_repository_mapping", return_value=expected) as confirm,
    ):
        client = TestClient(api_server.app)
        invalid = client.post("/api/github/repository-mappings/confirm", json={
            "project_id": "A", "repository": "owner/repo", "confirmed": True, "metadata": {"bypass": True},
        })
        response = client.post("/api/github/repository-mappings/confirm", json={
            "project_id": "A", "repository": "https://github.com/owner/repo", "confirmed": True,
        })
    assert invalid.status_code == 422
    assert response.status_code == 200 and response.json() == expected
    assert confirm.call_args.kwargs["request"]["confirmed"] is True


def test_confirm_endpoint_uses_injected_paths_and_never_calls_materialization(tmp_path, monkeypatch):
    confirmation_path = tmp_path / "confirmations.json"
    authority_path = tmp_path / "authority.json"
    monkeypatch.setattr(api_server, "PROJECT_REPOSITORY_CONFIRMATIONS_PATH", confirmation_path)
    monkeypatch.setattr(api_server, "PROJECT_REPOSITORY_IDENTITY_PATH", authority_path)
    monkeypatch.setattr(
        api_server.project_repository_mapping_service, "load_project_memory_for_repository_mapping",
        lambda _path: {"projects": [{"project_id": "A"}]},
    )
    with patch(
        "backend.github_evidence_materializer.materialize_saved_github_evidence",
        side_effect=AssertionError("must not materialize"),
    ):
        response = TestClient(api_server.app).post("/api/github/repository-mappings/confirm", json={
            "project_id": "A", "repository": "owner/repo", "confirmed": True,
        })
    assert response.status_code == 200 and response.json()["status"] == "created"
    assert confirmation_path.exists() and authority_path.exists()
    assert response.json()["materialization_required"] is True


def test_api_errors_do_not_leak_paths_or_exceptions():
    with patch.object(
        api_server.project_repository_mapping_service,
        "load_project_memory_for_repository_mapping", return_value=None,
    ):
        response = TestClient(api_server.app).get("/api/github/repository-mappings/projects")
    assert response.status_code == 503
    assert "Agent Develop" not in response.text and "Traceback" not in response.text
