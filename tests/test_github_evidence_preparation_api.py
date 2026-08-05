from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


SAFE_RESULT = {
    "status": "mapping_required", "enabled": True, "can_prepare": False,
    "repository_mapping_complete": False, "requires_repository_mapping": True,
    "saved_github_context_available": True, "evidence_prepared": False,
    "preparation_complete": False, "preparation_incomplete": False,
    "prepared_project_count": 0, "remaining_repository_count": 5,
    "conflict_count": 0, "ready_for_retrieval_setup": False,
    "message": "Connect all detected GitHub repositories to projects before preparing evidence.",
    "warnings": [], "errors": [],
}


def test_get_endpoint_is_safe_read_only_and_uses_fixed_configuration():
    with patch.object(
        api_server.github_evidence_preparation_service,
        "get_github_evidence_preparation_status", return_value=SAFE_RESULT,
    ) as status:
        response = TestClient(api_server.app).get("/api/github/evidence-preparation")
    assert response.status_code == 200 and response.json() == SAFE_RESULT
    assert status.call_args.kwargs["vector_store"] is api_server.agent.MEMORY_STORE
    assert "path" not in json.dumps(response.json()).casefold()


def test_post_requires_strict_explicit_confirmation_and_no_client_configuration():
    with patch.object(
        api_server.github_evidence_preparation_service,
        "run_github_evidence_preparation", return_value={**SAFE_RESULT, "status": "created"},
    ) as run:
        client = TestClient(api_server.app)
        missing = client.post("/api/github/evidence-preparation/run", json={})
        unknown = client.post("/api/github/evidence-preparation/run", json={"confirmed": True, "path": "C:/private"})
        malformed = client.post("/api/github/evidence-preparation/run", content="not-json", headers={"Content-Type": "application/json"})
        response = client.post("/api/github/evidence-preparation/run", json={"confirmed": True})
    assert missing.status_code == 422 and unknown.status_code == 422 and malformed.status_code == 422
    assert response.status_code == 200
    assert run.call_count == 1 and run.call_args.kwargs["confirmed"] is True
    assert not ({"path", "limit", "repository", "project_id"} & set(run.call_args.kwargs))


def test_false_confirmation_reaches_safe_service_boundary_only():
    with patch.object(
        api_server.github_evidence_preparation_service,
        "run_github_evidence_preparation", return_value={**SAFE_RESULT, "status": "blocked"},
    ) as run:
        response = TestClient(api_server.app).post(
            "/api/github/evidence-preparation/run", json={"confirmed": False},
        )
    assert response.status_code == 200 and response.json()["status"] == "blocked"
    run.assert_called_once()


def test_api_response_does_not_expose_internal_materializer_values():
    safe = {**SAFE_RESULT, "status": "error", "errors": ["evidence_preparation_failed"]}
    with patch.object(
        api_server.github_evidence_preparation_service,
        "run_github_evidence_preparation", return_value=safe,
    ):
        response = TestClient(api_server.app).post(
            "/api/github/evidence-preparation/run", json={"confirmed": True},
        )
    serialized = json.dumps(response.json())
    for forbidden in ("raw_text", "chunk_count", "content_hash", "C:\\", "diff --git", "Traceback"):
        assert forbidden not in serialized
