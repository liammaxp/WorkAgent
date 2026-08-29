"""HTTP boundary tests for the semantic tailoring review."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.hiring_context_ranking_review_api import (
    HIRING_CONTEXT_RANKING_REVIEW_FLAG,
    create_hiring_context_ranking_review_router,
    is_hiring_context_ranking_review_enabled,
)


def _payload():
    return {
        "status": "empty",
        "hiring_context": {
            "company": "Example",
            "team": None,
            "role_title": "Backend Engineer",
            "primary_role_family": "Backend engineering",
            "secondary_role_families": [],
            "context_signals": [],
            "confidence": "Moderate confidence",
        },
        "projects": [],
        "corrections_persisted": False,
    }


def _client(*, enabled=True, prepare=None, read_job=None, normalize=None, read_projects=None):
    app = FastAPI()
    app.include_router(create_hiring_context_ranking_review_router(
        read_job_description=read_job or (lambda: "Backend role JD"),
        normalize_job_context=normalize or (
            lambda _text: {"company": "Example", "job_title": "Backend Engineer"}
        ),
        read_project_memory=read_projects or (lambda: '{"projects": []}'),
        prepare_review=prepare or (lambda **_kwargs: SimpleNamespace(to_dict=_payload)),
        feature_enabled=lambda: enabled,
    ))
    return TestClient(app)


def test_feature_flag_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(HIRING_CONTEXT_RANKING_REVIEW_FLAG, raising=False)
    assert is_hiring_context_ranking_review_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_feature_flag_accepts_repository_truthy_values(monkeypatch, value):
    monkeypatch.setenv(HIRING_CONTEXT_RANKING_REVIEW_FLAG, value)
    assert is_hiring_context_ranking_review_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "NO", "off", "invalid"])
def test_feature_flag_disabled_and_malformed_values_fail_closed(monkeypatch, value):
    monkeypatch.setenv(HIRING_CONTEXT_RANKING_REVIEW_FLAG, value)
    assert is_hiring_context_ranking_review_enabled() is False


def test_availability_reports_disabled_without_reading_product_state():
    calls = []
    response = _client(
        enabled=False,
        read_job=lambda: calls.append("job") or "JD",
    ).get("/api/hiring-context/review/availability")
    assert response.status_code == 200
    assert response.json() == {"available": False}
    assert calls == []


def test_availability_reports_enabled():
    response = _client(enabled=True).get("/api/hiring-context/review/availability")
    assert response.json() == {"available": True}


def test_disabled_post_is_hidden_and_does_not_call_readers():
    calls = []
    response = _client(
        enabled=False,
        read_job=lambda: calls.append("job") or "JD",
        read_projects=lambda: calls.append("projects") or {},
    ).post("/api/hiring-context/review", json={})
    assert response.status_code == 404
    assert calls == []


def test_enabled_post_uses_current_normalized_job_context():
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=_payload)

    response = _client(
        prepare=prepare,
        normalize=lambda text: {
            "company": "Normalized Employer",
            "job_title": "Normalized Role",
            "required_qualifications": [text],
        },
    ).post("/api/hiring-context/review", json={"language": "en"})
    assert response.status_code == 200
    assert captured["company"] == "Normalized Employer"
    assert captured["role_title"] == "Normalized Role"
    assert captured["normalized_job_context"]["required_qualifications"] == ["Backend role JD"]


def test_preview_identity_overrides_do_not_write_job_state():
    calls = []
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=_payload)

    response = _client(
        prepare=prepare,
        read_job=lambda: calls.append("read") or "JD",
    ).post("/api/hiring-context/review", json={
        "company": "Preview Employer",
        "team": "Preview Team",
        "role_title": "Preview Role",
    })
    assert response.status_code == 200
    assert calls == ["read"]
    assert captured["company"] == "Preview Employer"
    assert captured["team"] == "Preview Team"
    assert captured["role_title"] == "Preview Role"


def test_explicit_empty_preview_can_clear_inferred_identity():
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=_payload)

    response = _client(prepare=prepare).post("/api/hiring-context/review", json={
        "company": "",
        "role_title": "",
    })
    assert response.status_code == 200
    assert captured["company"] == ""
    assert captured["role_title"] == ""


def test_project_display_metadata_is_parsed_without_fuzzy_transformation():
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=_payload)

    response = _client(
        prepare=prepare,
        read_projects=lambda: '{"projects":[{"project_id":"Exact_ID","project_name":"Exact Name"}]}',
    ).post("/api/hiring-context/review", json={})
    assert response.status_code == 200
    assert captured["project_memory"]["projects"][0]["project_id"] == "Exact_ID"


def test_invalid_project_metadata_only_removes_display_metadata():
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=_payload)

    response = _client(
        prepare=prepare,
        read_projects=lambda: "not json",
    ).post("/api/hiring-context/review", json={})
    assert response.status_code == 200
    assert captured["project_memory"] == {}


def test_unknown_request_fields_are_rejected():
    response = _client().post("/api/hiring-context/review", json={"candidate_fact": "invented"})
    assert response.status_code == 422


def test_request_does_not_accept_ranking_controls():
    response = _client().post("/api/hiring-context/review", json={"exclude_project": "alpha"})
    assert response.status_code == 422


def test_missing_saved_job_returns_bounded_product_error():
    response = _client(read_job=lambda: (_ for _ in ()).throw(FileNotFoundError())).post(
        "/api/hiring-context/review",
        json={},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "A saved job description is required."


def test_empty_saved_job_returns_bounded_product_error():
    response = _client(read_job=lambda: "   ").post("/api/hiring-context/review", json={})
    assert response.status_code == 400


def test_unexpected_service_failure_returns_no_diagnostics():
    response = _client(
        prepare=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("secret trace")),
    ).post("/api/hiring-context/review", json={})
    assert response.status_code == 500
    assert response.json()["detail"] == "Could not prepare the tailoring review."
    assert "secret trace" not in response.text


def test_response_contract_contains_only_product_review_payload():
    response = _client().post("/api/hiring-context/review", json={})
    assert response.status_code == 200
    assert response.json() == _payload()


def test_router_exposes_only_semantic_review_routes():
    client = _client()
    paths = {
        route.path
        for route in client.app.routes
        if "hiring-context" in route.path
    }
    assert paths == {
        "/api/hiring-context/review",
        "/api/hiring-context/review/availability",
    }


def test_production_app_registers_the_isolated_review_router():
    backend_path = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    api_server = importlib.import_module("api_server")
    paths = {route.path for route in api_server.app.routes}
    assert "/api/hiring-context/review" in paths
    assert "/api/hiring-context/review/availability" in paths
